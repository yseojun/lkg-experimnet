/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * ------------------------------------------------------------------
 * MODIFICATION NOTICE:
 * This file has been modified by POSTECH and ETRI on 2026-04-14.
 * Changes: Added modifications related to CoherentRaster
 * The modifications are also licensed under the Apache License, Version 2.0.
 * ------------------------------------------------------------------
 */

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <cuda_runtime_api.h>
#include <limits>
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"     // where all the macros are defined
#include "Ops.h"        // a collection of all gsplat operators
#include "Ops_CoherentRaster.h"
#include "CoherentRaster.h" // where the launch function is declared
#include "Cameras.h"

#include "Intersect.h"

namespace gsplat {

namespace {

void check_cuda_status(cudaError_t status, const char *message) {
    TORCH_CHECK(status == cudaSuccess, message, ": ", cudaGetErrorString(status));
}

class CudaStageTimer {
  public:
    explicit CudaStageTimer(bool enabled) : enabled_(enabled) {
        if (!enabled_) {
            return;
        }
        stream_ = at::cuda::getCurrentCUDAStream();
        check_cuda_status(cudaEventCreate(&start_), "cudaEventCreate(start) failed");
        check_cuda_status(cudaEventCreate(&stop_), "cudaEventCreate(stop) failed");
        check_cuda_status(cudaEventRecord(start_, stream_), "cudaEventRecord(start) failed");
    }

    ~CudaStageTimer() {
        if (start_ != nullptr) {
            cudaEventDestroy(start_);
        }
        if (stop_ != nullptr) {
            cudaEventDestroy(stop_);
        }
    }

    float elapsed_ms() {
        if (!enabled_) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        check_cuda_status(cudaEventRecord(stop_, stream_), "cudaEventRecord(stop) failed");
        check_cuda_status(cudaEventSynchronize(stop_), "cudaEventSynchronize(stop) failed");
        float elapsed = 0.0f;
        check_cuda_status(cudaEventElapsedTime(&elapsed, start_, stop_), "cudaEventElapsedTime failed");
        return elapsed;
    }

  private:
    bool enabled_ = false;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

at::Tensor make_cr_timing_tensor(const at::Tensor &reference, float isect_ms, float sort_ms) {
    auto options = reference.options().dtype(at::kFloat).device(at::kCPU);
    return at::tensor({isect_ms, sort_ms}, options);
}

at::Tensor empty_cr_timing_tensor(const at::Tensor &reference) {
    auto options = reference.options().dtype(at::kFloat).device(at::kCPU);
    return at::empty({0}, options);
}

} // namespace


std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_ewa_3dgs_fused_fwd_CR(
    const at::optional<at::Tensor> mask,      // [N, 2]
    const at::Tensor means,                // [N, 3]
    const at::optional<at::Tensor> covars, // [N, 6] optional
    const at::optional<at::Tensor> quats,  // [N, 4] optional
    const at::optional<at::Tensor> scales, // [N, 3] optional
    const at::optional<at::Tensor> opacities, // [N] optional
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model,
    const at::optional<at::Tensor> rtgs_projection_adapter
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    if (rtgs_projection_adapter.has_value()) {
        CHECK_INPUT(rtgs_projection_adapter.value());
        TORCH_CHECK(
            rtgs_projection_adapter.value().dim() == 1 && rtgs_projection_adapter.value().size(0) == 6,
            "rtgs_projection_adapter must have shape [6]"
        );
        TORCH_CHECK(
            rtgs_projection_adapter.value().scalar_type() == means.scalar_type(),
            "rtgs_projection_adapter dtype must match means dtype"
        );
    }

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras

    at::Tensor radii = at::empty({C, N, 2}, means.options().dtype(at::kInt));
    at::Tensor means2d = at::empty({C, N, 2}, means.options());
    at::Tensor depths = at::empty({C, N}, means.options());
    at::Tensor conics = at::empty({C, N, 3}, means.options());
    at::Tensor compensations;
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = at::zeros({C, N}, means.options());
    }

    launch_projection_ewa_3dgs_fused_fwd_kernel_CR(
        // inputs
        mask,
        means,
        covars,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        rtgs_projection_adapter,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        camera_model,
        // outputs
        radii,
        means2d,
        depths,
        conics,
        calc_compensations ? at::optional<at::Tensor>(compensations)
                           : c10::nullopt
    );
    return std::make_tuple(radii, means2d, depths, conics, compensations);
}




std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> intersect_tile_CR(
    const at::Tensor means3d, // [N, 3]
    const at::Tensor means2d,                    // [1, N, 2]
    const at::Tensor radii,                      // [1, N, 2]
    const at::Tensor depths,                     // [1, N]
    const at::Tensor reference_viewmat, // [1, 4, 4]
    const at::Tensor adjacent_viewmats, // [num_of_view_group, view_group_size, 4, 4]
    const at::Tensor K,                         // [3, 3]
    const at::optional<at::Tensor> camera_ids,   // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort,
    const at::optional<at::Tensor> rtgs_projection_adapter,
    const bool return_timing
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);
    if (rtgs_projection_adapter.has_value()) {
        CHECK_INPUT(rtgs_projection_adapter.value());
        TORCH_CHECK(
            rtgs_projection_adapter.value().dim() == 1 && rtgs_projection_adapter.value().size(0) == 6,
            "rtgs_projection_adapter must have shape [6]"
        );
        TORCH_CHECK(
            rtgs_projection_adapter.value().scalar_type() == K.scalar_type(),
            "rtgs_projection_adapter dtype must match K dtype"
        );
    }

    uint32_t n_elements = means2d.numel() / 2;
    bool packed = means2d.dim() == 2;
    if (packed) {
        TORCH_CHECK(
            camera_ids.has_value() && gaussian_ids.has_value(),
            "When packed is set, camera_ids and gaussian_ids must be provided."
        );
        CHECK_INPUT(camera_ids.value());
        CHECK_INPUT(gaussian_ids.value());
    }

    uint32_t n_tiles = tile_width * tile_height;
    // the number of bits needed to encode the camera id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t cam_n_bits = std::bit_width(C);
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    uint32_t N = means2d.size(1);
    uint32_t num_of_view_group = adjacent_viewmats.size(0);
    uint32_t num_of_adjacent = adjacent_viewmats.size(1);
    uint32_t cam_n_bits = (uint32_t)floor(log2(num_of_view_group)) + 1;
    // the first 32 bits are used for the camera id and tile id altogether, so
    // check if we have enough bits for them.
    assert(tile_n_bits + cam_n_bits <= 32);

    // calculate translation values of each gaussian
    

    CudaStageTimer isect_timer(return_timing);

    // first pass: compute number of tiles per gaussian
    at::Tensor tiles_per_gauss =
        at::empty({num_of_view_group, N}, depths.options().dtype(at::kInt));
    int64_t n_isects;
    at::Tensor cum_tiles_per_gauss;
    if (n_elements) {
        launch_intersect_tile_kernel_CR(
            // inputs
            means3d,
            reference_viewmat,
            adjacent_viewmats,
            K,
            rtgs_projection_adapter,
            num_of_adjacent,
            means2d,
            radii,
            depths,
            packed ? camera_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            num_of_view_group,
            tile_size,
            tile_width,
            tile_height,
            c10::nullopt, // cum_tiles_per_gauss
            // outputs
            at::optional<at::Tensor>(tiles_per_gauss),
            c10::nullopt, // isect_ids
            c10::nullopt,  // flatten_ids
            c10::nullopt  // translation_values
        );
        cum_tiles_per_gauss = at::cumsum(tiles_per_gauss.view({-1}), 0);
        n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
    } else {
        n_isects = 0;
    }
    
    // second pass: compute isect_ids and flatten_ids as a packed tensor
    at::Tensor isect_ids =
    at::empty({n_isects}, depths.options().dtype(at::kLong));
    at::Tensor flatten_ids =
    at::empty({n_isects}, depths.options().dtype(at::kInt));

    at::Tensor translation_values = at::empty({N, num_of_view_group, num_of_adjacent, 2}, depths.options());
    if (n_isects) {
        launch_intersect_tile_kernel_CR(
            // inputs
            means3d,
            reference_viewmat,
            adjacent_viewmats,
            K,
            rtgs_projection_adapter,
            num_of_adjacent,
            means2d,
            radii,
            depths,
            packed ? camera_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            num_of_view_group,
            tile_size,
            tile_width,
            tile_height,
            cum_tiles_per_gauss,
            // outputs
            c10::nullopt, // tiles_per_gauss
            at::optional<at::Tensor>(isect_ids),
            at::optional<at::Tensor>(flatten_ids),
            at::optional<at::Tensor>(translation_values)
        );
    }

    float isect_ms = isect_timer.elapsed_ms();
    float sort_ms = sort && n_isects ? 0.0f : std::numeric_limits<float>::quiet_NaN();

    // optionally sort the Gaussians by isect_ids
    if (n_isects && sort) {
        at::Tensor isect_ids_sorted = at::empty_like(isect_ids);
        at::Tensor flatten_ids_sorted = at::empty_like(flatten_ids);
        CudaStageTimer sort_timer(return_timing);
        radix_sort_double_buffer(
            n_isects,
            tile_n_bits,
            cam_n_bits,
            isect_ids,
            flatten_ids,
            isect_ids_sorted,
            flatten_ids_sorted
        );
        sort_ms = sort_timer.elapsed_ms();
        at::Tensor timing_ms = return_timing ? make_cr_timing_tensor(means2d, isect_ms, sort_ms) : empty_cr_timing_tensor(means2d);
        return std::make_tuple(
            tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted, translation_values, timing_ms
        );
    } else {
        at::Tensor timing_ms = return_timing ? make_cr_timing_tensor(means2d, isect_ms, sort_ms) : empty_cr_timing_tensor(means2d);
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids, translation_values, timing_ms);
    }
}

at::Tensor intersect_offset_CR(
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t C,
    const uint32_t tile_width,
    const uint32_t tile_height
) {
    DEVICE_GUARD(isect_ids);
    CHECK_INPUT(isect_ids);

    at::Tensor offsets = at::empty(
        {tile_height, tile_width, C}, isect_ids.options().dtype(at::kInt)
    );
    launch_intersect_offset_kernel_CR(
        isect_ids, C, tile_width, tile_height, offsets
    );
    return offsets;
}



std::tuple<at::Tensor> rasterize_to_pixels_3dgs_fwd_CR(
    const at::Tensor means2d,  // [n_view_group, n_gauss, 2]
    const at::Tensor conics,  // [n_view_group, n_gauss, 3]
    const at::Tensor colors,  // [n_view_group, n_gauss, 3]
    const at::Tensor opacities,  // [n_view_group, n_gauss]
    const at::Tensor translation_values,  // [n_gauss, n_view_group, view_group_size, 2]
    const at::Tensor view_idx_matrix,  // [n_tile_height, n_tile_width, 3, tile_size, tile_size]
    const at::Tensor subpixel_coord_matrix,  // [n_tile_height, n_tile_width, 3, tile_size, tile_size, 3]
    const at::optional<at::Tensor> backgrounds, // [num_of_view_group, channels]
    const at::Tensor tile_offsets,  // [n_view_group, n_tile_height, n_tile_width]
    const at::Tensor flatten_ids  // [n_intersection]
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }

    uint32_t n_tile_height = view_idx_matrix.size(0);
    uint32_t n_tile_width = view_idx_matrix.size(1);
    uint32_t tile_size = view_idx_matrix.size(3);

    at::Tensor renders =
        at::empty({n_tile_height, n_tile_width, 3, tile_size, tile_size}, means2d.options());

    launch_rasterize_to_pixels_3dgs_fwd_kernel_CR(
            means2d,
            conics,
            colors,
            opacities,
            translation_values,
            view_idx_matrix,
            subpixel_coord_matrix,
            backgrounds,
            tile_offsets,
            flatten_ids,
            renders
        );

    return std::make_tuple(renders);
}




} // namespace gsplat
