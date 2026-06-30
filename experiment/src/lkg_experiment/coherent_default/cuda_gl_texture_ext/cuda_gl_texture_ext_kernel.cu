#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ uint8_t to_rgba8(float value) {
    if (isnan(value)) {
        value = 0.0f;
    }
    value = fminf(fmaxf(value, 0.0f), 1.0f);
    return static_cast<uint8_t>(value * 255.0f);
}

__global__ void chw_or_hwc_float_to_rgba8_kernel(
    const float* __restrict__ image,
    uint8_t* __restrict__ rgba,
    int64_t width,
    int64_t height,
    int64_t stride0,
    int64_t stride1,
    int64_t stride2,
    int layout) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int64_t out = (static_cast<int64_t>(y) * width + x) * 4;
    if (layout == 0) {
        rgba[out + 0] = to_rgba8(image[0 * stride0 + y * stride1 + x * stride2]);
        rgba[out + 1] = to_rgba8(image[1 * stride0 + y * stride1 + x * stride2]);
        rgba[out + 2] = to_rgba8(image[2 * stride0 + y * stride1 + x * stride2]);
    } else {
        rgba[out + 0] = to_rgba8(image[y * stride0 + x * stride1 + 0 * stride2]);
        rgba[out + 1] = to_rgba8(image[y * stride0 + x * stride1 + 1 * stride2]);
        rgba[out + 2] = to_rgba8(image[y * stride0 + x * stride1 + 2 * stride2]);
    }
    rgba[out + 3] = 255;
}

}  // namespace

void launch_chw_or_hwc_float_to_rgba8(
    torch::Tensor image,
    torch::Tensor rgba,
    int64_t width,
    int64_t height,
    int layout,
    cudaStream_t stream) {
    const dim3 block(16, 16);
    const dim3 grid(
        static_cast<unsigned int>((width + block.x - 1) / block.x),
        static_cast<unsigned int>((height + block.y - 1) / block.y));
    chw_or_hwc_float_to_rgba8_kernel<<<grid, block, 0, stream>>>(
        image.data_ptr<float>(),
        rgba.data_ptr<uint8_t>(),
        width,
        height,
        image.stride(0),
        image.stride(1),
        image.stride(2),
        layout);
}
