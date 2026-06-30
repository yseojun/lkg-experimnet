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

#pragma once
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Utils.cuh"

#include <torch/extension.h>

namespace gsplat {

namespace cg = cooperative_groups;


template <typename scalar_t>
__global__ void projection_ewa_3dgs_fused_fwd_kernel_CR(
    const uint32_t C,
    const uint32_t N,
    const bool * __restrict__ mask,  // [N]
    const scalar_t *__restrict__ means,    // [N, 3]
    const scalar_t *__restrict__ covars,   // [N, 6] optional
    const scalar_t *__restrict__ quats,    // [N, 4] optional
    const scalar_t *__restrict__ scales,   // [N, 3] optional
    const scalar_t *__restrict__ opacities, // [N] optional
    const scalar_t *__restrict__ viewmats, // [C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [C, 3, 3]
    const scalar_t *__restrict__ rtgs_projection_adapter, // [6] optional
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    int32_t *__restrict__ radii,         // [C, N, 2]
    scalar_t *__restrict__ means2d,      // [C, N, 2]
    scalar_t *__restrict__ depths,       // [C, N]
    scalar_t *__restrict__ conics,       // [C, N, 3]
    scalar_t *__restrict__ compensations // [C, N] optional
);


__global__ void intersect_offset_kernel_CR(
    const uint32_t n_isects,
    const int64_t *__restrict__ isect_ids,
    const uint32_t n_view_group,
    const uint32_t n_tiles,
    const uint32_t view_group_n_bits,
    int32_t *__restrict__ offsets // [tile_height, tile_width, n_view_group]
);

} // namespace gsplat
