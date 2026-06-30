/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "CoherentRaster.h"
#include "Utils.cuh"
#include "CoherentRaster_Modified.cuh"
#include "CoherentRaster_CR.cuh"

#include <torch/extension.h>

namespace gsplat {

namespace cg = cooperative_groups;


void launch_projection_ewa_3dgs_fused_fwd_kernel_CR(
    // inputs
    const at::optional<at::Tensor> mask,      // [N, 2]
    const at::Tensor means,                // [N, 3]
    const at::optional<at::Tensor> covars, // [N, 6] optional
    const at::optional<at::Tensor> quats,  // [N, 4] optional
    const at::optional<at::Tensor> scales, // [N, 3] optional
    const at::optional<at::Tensor> opacities, // [N] optional
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const at::optional<at::Tensor> rtgs_projection_adapter, // [6] optional
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    at::Tensor radii,                      // [C, N, 2]
    at::Tensor means2d,                    // [C, N, 2]
    at::Tensor depths,                     // [C, N]
    at::Tensor conics,                     // [C, N, 3]
    at::optional<at::Tensor> compensations // [C, N] optional
) {
    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras

    int64_t n_elements = C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    
    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "projection_ewa_3dgs_fused_fwd_kernel_CR",
        [&]() {
            projection_ewa_3dgs_fused_fwd_kernel_CR<scalar_t>
                <<<grid,
                threads,
                shmem_size,
                at::cuda::getCurrentCUDAStream()>>>(
                    C,
                    N,
                    mask.has_value() ? mask.value().data_ptr<bool>()
                                    : nullptr,
                    means.data_ptr<scalar_t>(),
                    covars.has_value() ? covars.value().data_ptr<scalar_t>()
                                    : nullptr,
                    quats.has_value() ? quats.value().data_ptr<scalar_t>()
                                    : nullptr,
                    scales.has_value() ? scales.value().data_ptr<scalar_t>()
                                    : nullptr,
                    opacities.has_value() ? opacities.value().data_ptr<scalar_t>()
                                        : nullptr,
                    viewmats.data_ptr<scalar_t>(),
                    Ks.data_ptr<scalar_t>(),
                    rtgs_projection_adapter.has_value()
                        ? rtgs_projection_adapter.value().data_ptr<scalar_t>()
                        : nullptr,
                    image_width,
                    image_height,
                    eps2d,
                    near_plane,
                    far_plane,
                    radius_clip,
                    camera_model,
                    radii.data_ptr<int32_t>(),
                    means2d.data_ptr<scalar_t>(),
                    depths.data_ptr<scalar_t>(),
                    conics.data_ptr<scalar_t>(),
                    compensations.has_value()
                        ? compensations.value().data_ptr<scalar_t>()
                        : nullptr
                );
        }
    );
        
}


void launch_intersect_tile_kernel_CR(
    // inputs
    const at::Tensor means3d,                   // [N, 3]
    const at::Tensor reference_viewmat,         // [1, 4, 4]
    const at::Tensor adjacent_viewmats,         // [n_view_group, view_group_size, 4, 4]
    const at::Tensor K,                         // [3, 3]
    const at::optional<at::Tensor> rtgs_projection_adapter, // [6] optional
    const uint32_t view_group_size,             // number of adjacent cameras
    const at::Tensor means2d,                    // [1, N, 2]
    const at::Tensor radii,                      // [1, N, 2]
    const at::Tensor depths,                     // [1, N]
    const at::optional<at::Tensor> camera_ids,   // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t n_view_group,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const at::optional<at::Tensor> cum_tiles_per_gauss, // [n_view_group, N]
    // outputs
    at::optional<at::Tensor> tiles_per_gauss, // [n_view_group, N]
    at::optional<at::Tensor> isect_ids,       // [n_isects]
    at::optional<at::Tensor> flatten_ids,      // [n_isects]
    at::optional<at::Tensor> translation_values  // [N, n_view_group, view_group_size, 2]
) {
    bool packed = means2d.dim() == 2;

    uint32_t N, nnz;
    int64_t n_elements;
    if (packed) {
        nnz = means2d.size(0); // total number of gaussians
        n_elements = nnz;
    } else {
        N = means2d.size(1); // number of gaussians per camera
        n_elements = n_view_group * N;
    }

    uint32_t n_tiles = tile_width * tile_height;
    // the number of bits needed to encode the camera id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t cam_n_bits = std::bit_width(C);
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    uint32_t view_group_n_bits = (uint32_t)floor(log2(n_view_group)) + 1;
    // the first 32 bits are used for the camera id and tile id altogether, so
    // check if we have enough bits for them.
    assert(tile_n_bits + view_group_n_bits <= 32);

    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }


    AT_DISPATCH_FLOATING_TYPES(
        means2d.scalar_type(),
        "intersect_tile_kernel_CR_AccuTile",
        [&]() {
            intersect_tile_kernel_CR_AccuTile
                <<<grid,
                    threads,
                    shmem_size,
                    at::cuda::getCurrentCUDAStream()>>>(
                    n_view_group,
                    N,
                    means3d.data_ptr<float>(),
                    reference_viewmat.data_ptr<float>(),
                    adjacent_viewmats.data_ptr<float>(),
                    K.data_ptr<float>(),
                    rtgs_projection_adapter.has_value()
                        ? rtgs_projection_adapter.value().data_ptr<float>()
                        : nullptr,
                    view_group_size,
                    means2d.data_ptr<float>(),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<float>(),
                    cum_tiles_per_gauss.has_value()
                        ? cum_tiles_per_gauss.value().data_ptr<int64_t>()
                        : nullptr,
                    tile_size,
                    tile_width,
                    tile_height,
                    view_group_n_bits,
                    tiles_per_gauss.has_value()
                        ? tiles_per_gauss.value().data_ptr<int32_t>()
                        : nullptr,
                    isect_ids.has_value()
                        ? isect_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    flatten_ids.has_value()
                        ? flatten_ids.value().data_ptr<int32_t>()
                        : nullptr,
                    translation_values.has_value()
                        ? translation_values.value().data_ptr<float>()
                        : nullptr
                );
        }
    );

}

void launch_intersect_offset_kernel_CR(
    // inputs
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t n_view_group,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // outputs
    at::Tensor offsets // [tile_height, tile_width, n_view_group]
) {
    int64_t n_elements = isect_ids.size(0); // total number of intersections
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        offsets.fill_(0);
        return;
    }

    uint32_t n_tiles = tile_width * tile_height;
    uint32_t view_group_n_bits = (uint32_t)floor(log2(n_view_group)) + 1;
    intersect_offset_kernel_CR<<<
        grid,
        threads,
        shmem_size,
        at::cuda::getCurrentCUDAStream()>>>(
        n_elements,
        isect_ids.data_ptr<int64_t>(),
        n_view_group,
        n_tiles,
        view_group_n_bits,
        offsets.data_ptr<int32_t>()
    );
}


void launch_rasterize_to_pixels_3dgs_fwd_kernel_CR(
    const at::Tensor means2d,  // [n_view_group, n_gauss, 2]
    const at::Tensor conics,  // [n_view_group, n_gauss, 3]
    const at::Tensor colors,  // [n_view_group, n_gauss, 3]
    const at::Tensor opacities,  // [n_gauss]
    const at::Tensor translation_values,  // [n_gauss, n_view_group, view_group_size, 2]
    const at::Tensor view_idx_matrix,  // [n_tile_height, n_tile_width, 3, tile_size, tile_size]
    const at::Tensor subpixel_coord_matrix,  // [n_tile_height, n_tile_width, 3, tile_size, tile_size, 3]
    const at::optional<at::Tensor> backgrounds, // [num_of_view_group, channels]
    const at::Tensor tile_offsets,  // [n_tile_height, n_tile_width, n_view_group]
    const at::Tensor flatten_ids,  // [n_intersection]
    at::Tensor renders  // [n_tile_height, n_tile_width, 3, tile_size, tile_size]
) {
    const uint32_t n_tile_height = tile_offsets.size(0);
    const uint32_t n_tile_width = tile_offsets.size(1);

    const uint32_t tile_size = view_idx_matrix.size(3);
    
    dim3 threads = {tile_size, tile_size, 3};
    dim3 grid = {n_tile_width, n_tile_height, 1};

    int64_t shmem_size = 0;

    if (cudaFuncSetAttribute(
            rasterize_to_pixels_3dgs_fwd_kernel_CR,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_3dgs_fwd_kernel_CR
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            means2d.packed_accessor64<float, 3, torch::RestrictPtrTraits>(),
            conics.packed_accessor64<float, 3, torch::RestrictPtrTraits>(),
            colors.packed_accessor64<float, 3, torch::RestrictPtrTraits>(),
            opacities.packed_accessor64<float, 1, torch::RestrictPtrTraits>(),
            translation_values.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
            view_idx_matrix.packed_accessor64<uint32_t, 5, torch::RestrictPtrTraits>(),
            subpixel_coord_matrix.packed_accessor64<uint32_t, 6, torch::RestrictPtrTraits>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            tile_offsets.packed_accessor64<int32_t, 3, torch::RestrictPtrTraits>(),
            flatten_ids.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
            renders.packed_accessor64<float, 5, torch::RestrictPtrTraits>()
        );
}

} // namespace gsplat
