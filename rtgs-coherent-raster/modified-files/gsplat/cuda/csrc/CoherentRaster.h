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

#pragma once

#include <cstdint>
#include "Cameras.h"

namespace at {
class Tensor;
}

namespace gsplat {

void launch_projection_ewa_3dgs_fused_fwd_kernel_CR(
    // inputs
    const at::optional<at::Tensor> mask,      // [N]
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
);





void launch_intersect_tile_kernel_CR(
    // inputs
    const at::Tensor means3d,  // [N, 3]
    const at::Tensor reference_viewmat,   // [1, 4, 4]
    const at::Tensor adjacent_viewmats, // [num_of_adjacent, 4, 4]
    const at::Tensor K,         // [3, 3]
    const at::optional<at::Tensor> rtgs_projection_adapter, // [6] optional
    const uint32_t num_of_adjacent,
    const at::Tensor means2d,                    // [C, N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [C, N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [C, N] or [nnz]
    const at::optional<at::Tensor> camera_ids,   // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const at::optional<at::Tensor> cum_tiles_per_gauss, // [C, N] or [nnz]
    // outputs
    at::optional<at::Tensor> tiles_per_gauss, // [C, N] or [nnz]
    at::optional<at::Tensor> isect_ids,       // [n_isects]
    at::optional<at::Tensor> flatten_ids,      // [n_isects]
    at::optional<at::Tensor> translation_values  // [N, num_of_adjacent, 2]
);

void launch_intersect_offset_kernel_CR(
    // inputs
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t C,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // outputs
    at::Tensor offsets // [C, tile_height, tile_width]
);




void launch_rasterize_to_pixels_3dgs_fwd_kernel_CR(
    const at::Tensor means2d,  // [n_view_group, n_gauss, 2]
    const at::Tensor conics,  // [n_view_group, n_gauss, 3]
    const at::Tensor colors,  // [n_view_group, n_gauss, 3]
    const at::Tensor opacities,  // [n_view_group, n_gauss]
    const at::Tensor translation_values,  // [n_gauss, n_view_group, view_group_size, 2]
    const at::Tensor view_idx_matrix,  // [n_tile_height, n_tile_width, 3, tile_size, tile_size]
    const at::Tensor subpixel_coord_matrix,  // [n_tile_height, n_tile_width, 3, tile_size, tile_size, 3]
    const at::optional<at::Tensor> backgrounds, // [num_of_view_group, channels]
    const at::Tensor tile_offsets,  // [n_view_group, n_tile_height, n_tile_width]
    const at::Tensor flatten_ids,  // [n_intersection]
    at::Tensor renders  // [n_tile_height, n_tile_width, 3, tile_size, tile_size]
);

} // namespace gsplat
