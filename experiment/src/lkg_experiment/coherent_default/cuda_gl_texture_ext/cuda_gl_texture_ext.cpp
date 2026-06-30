#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <GL/gl.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

void launch_chw_or_hwc_float_to_rgba8(
    torch::Tensor image,
    torch::Tensor rgba,
    int64_t width,
    int64_t height,
    int layout,
    cudaStream_t stream);

namespace {

void check_cuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(error);
    throw std::runtime_error(message.str());
}

cudaGraphicsResource_t as_resource(uintptr_t handle) {
    if (handle == 0) {
        throw std::runtime_error("CUDA graphics resource handle is null");
    }
    return reinterpret_cast<cudaGraphicsResource_t>(handle);
}

int image_layout(torch::Tensor image, int64_t width, int64_t height) {
    if (image.dim() != 3) {
        throw std::runtime_error("texture upload tensor must have shape (3,H,W) or (H,W,3)");
    }
    if (image.size(0) == 3 && image.size(1) == height && image.size(2) == width) {
        return 0;
    }
    if (image.size(0) == height && image.size(1) == width && image.size(2) == 3) {
        return 1;
    }
    std::ostringstream message;
    message << "texture upload tensor shape does not match panel texture: got ("
            << image.size(0) << "," << image.size(1) << "," << image.size(2)
            << "), expected (3," << height << "," << width << ") or ("
            << height << "," << width << ",3)";
    throw std::runtime_error(message.str());
}

class MappedGraphicsResource {
public:
    MappedGraphicsResource(cudaGraphicsResource_t resource, cudaStream_t stream)
        : resource_(resource), stream_(stream), mapped_(false) {
        check_cuda(cudaGraphicsMapResources(1, &resource_, stream_), "cudaGraphicsMapResources");
        mapped_ = true;
    }

    ~MappedGraphicsResource() {
        if (!mapped_) {
            return;
        }
        cudaGraphicsUnmapResources(1, &resource_, stream_);
    }

    cudaArray_t array() {
        cudaArray_t mapped_array = nullptr;
        check_cuda(cudaGraphicsSubResourceGetMappedArray(&mapped_array, resource_, 0, 0),
                   "cudaGraphicsSubResourceGetMappedArray");
        return mapped_array;
    }

    void unmap() {
        if (!mapped_) {
            return;
        }
        check_cuda(cudaGraphicsUnmapResources(1, &resource_, stream_), "cudaGraphicsUnmapResources");
        mapped_ = false;
    }

private:
    cudaGraphicsResource_t resource_;
    cudaStream_t stream_;
    bool mapped_;
};

}  // namespace

uintptr_t register_gl_texture(uint32_t texture, int64_t cuda_device) {
    if (cuda_device >= 0) {
        check_cuda(cudaSetDevice(static_cast<int>(cuda_device)), "cudaSetDevice");
    }
    cudaGraphicsResource_t resource = nullptr;
    check_cuda(
        cudaGraphicsGLRegisterImage(
            &resource,
            static_cast<GLuint>(texture),
            GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsWriteDiscard),
        "cudaGraphicsGLRegisterImage");
    return reinterpret_cast<uintptr_t>(resource);
}

void unregister_gl_texture(uintptr_t handle) {
    cudaGraphicsResource_t resource = as_resource(handle);
    check_cuda(cudaGraphicsUnregisterResource(resource), "cudaGraphicsUnregisterResource");
}

std::vector<int64_t> gl_cuda_devices() {
    unsigned int count = 0;
    int devices[32] = {};
    check_cuda(cudaGLGetDevices(&count, devices, 32, cudaGLDeviceListAll), "cudaGLGetDevices");
    std::vector<int64_t> result;
    result.reserve(count);
    for (unsigned int i = 0; i < count; ++i) {
        result.push_back(static_cast<int64_t>(devices[i]));
    }
    return result;
}

void upload_cuda_tensor(uintptr_t handle, torch::Tensor image, int64_t width, int64_t height) {
    TORCH_CHECK(image.is_cuda(), "texture upload tensor must be on CUDA");
    TORCH_CHECK(image.scalar_type() == torch::kFloat32, "texture upload tensor must be float32");
    TORCH_CHECK(width > 0 && height > 0, "texture dimensions must be positive");

    const int layout = image_layout(image, width, height);
    c10::cuda::CUDAGuard guard(image.device());
    const int device_index = static_cast<int>(image.get_device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index);
    auto rgba = torch::empty({height, width, 4}, image.options().dtype(torch::kUInt8));

    launch_chw_or_hwc_float_to_rgba8(image, rgba, width, height, layout, stream);
    check_cuda(cudaGetLastError(), "rgba conversion kernel launch");

    cudaGraphicsResource_t resource = as_resource(handle);
    MappedGraphicsResource mapped(resource, stream);
    cudaArray_t array = mapped.array();
    const size_t row_bytes = static_cast<size_t>(width) * 4;
    check_cuda(
        cudaMemcpy2DToArrayAsync(
            array,
            0,
            0,
            rgba.data_ptr<uint8_t>(),
            row_bytes,
            row_bytes,
            static_cast<size_t>(height),
            cudaMemcpyDeviceToDevice,
            stream),
        "cudaMemcpy2DToArrayAsync");
    mapped.unmap();
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("register_gl_texture", &register_gl_texture, "Register a GL_TEXTURE_2D for CUDA writes");
    m.def("unregister_gl_texture", &unregister_gl_texture, "Unregister a CUDA graphics resource");
    m.def("gl_cuda_devices", &gl_cuda_devices, "CUDA devices associated with the current GL context");
    m.def("upload_cuda_tensor", &upload_cuda_tensor, "Upload a CUDA RGB tensor into a registered GL texture");
}
