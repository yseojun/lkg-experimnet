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
#include "Utils.cuh"
#include "CoherentRaster_Modified.cuh"

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
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    // glm is column-major but input is row-major
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to camera space
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);
    vec3 projection_mean_c = mean_c;
    scalar_t fx = Ks[0];
    scalar_t fy = Ks[4];
    scalar_t cx = Ks[2];
    scalar_t cy = Ks[5];
    if (rtgs_projection_adapter != nullptr) {
        fx = rtgs_projection_adapter[0];
        fy = rtgs_projection_adapter[1];
        projection_mean_c.x *= rtgs_projection_adapter[2];
        projection_mean_c.y *= rtgs_projection_adapter[3];
        cx = rtgs_projection_adapter[4];
        cy = rtgs_projection_adapter[5];
    }
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // transform Gaussian covariance to camera space
    mat3 covar;
    if (covars != nullptr) {
        covars += gid * 6;
        covar = mat3(
            covars[0],
            covars[1],
            covars[2], // 1st column
            covars[1],
            covars[3],
            covars[4], // 2nd column
            covars[2],
            covars[4],
            covars[5] // 3rd column
        );
    } else {
        // compute from quaternions and scales
        quats += gid * 4;
        scales += gid * 3;
        quat_scale_to_covar_preci(
            glm::make_vec4(quats), glm::make_vec3(scales), &covar, nullptr
        );
    }
    mat3 covar_c;
    covarW2C(R, covar, covar_c);

    // perspective projection
    mat2 covar2d;
    vec2 mean2d;

    switch (camera_model) {
    case CameraModelType::PINHOLE: // perspective projection
        persp_proj(
            projection_mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    case CameraModelType::ORTHO: // orthographic projection
        ortho_proj(
            projection_mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    case CameraModelType::FISHEYE: // fisheye projection
        fisheye_proj(
            projection_mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    }

    float compensation;
    float det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // compute the inverse of the 2d covariance
    mat2 covar2d_inv = glm::inverse(covar2d);

    float extend = 3.33f;
    if (opacities != nullptr) {
        float opacity = opacities[gid];
        if (compensations != nullptr) {
            // we assume compensation term will be applied later on.
            opacity *= compensation;
        }
        if (opacity < ALPHA_THRESHOLD) {
            radii[idx * 2] = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }
        // Compute opacity-aware bounding box.
        // https://arxiv.org/pdf/2402.00525 Section B.2
        extend = min(extend, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
    }

    // compute tight rectangular bounding box (non differentiable)
    // https://arxiv.org/pdf/2402.00525
    float radius_x = ceilf(extend * sqrtf(covar2d[0][0]));
    float radius_y = ceilf(extend * sqrtf(covar2d[1][1]));

    if (radius_x <= radius_clip && radius_y <= radius_clip) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // mask out gaussians outside the image region (clip region)
    if (mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= image_width ||
        mean2d.y + radius_y <= 0 || mean2d.y - radius_y >= image_height) {

        if (mask != nullptr) {
            if (mask[gid] == false) {
                radii[idx * 2] = 0;
                radii[idx * 2 + 1] = 0;
                return;
            }
        }
        else {
            radii[idx * 2] = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }
        

    }

    // write to outputs
    radii[idx * 2] = (int32_t)radius_x;
    radii[idx * 2 + 1] = (int32_t)radius_y;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = mean_c.z;
    conics[idx * 3] = covar2d_inv[0][0];
    conics[idx * 3 + 1] = covar2d_inv[0][1];
    conics[idx * 3 + 2] = covar2d_inv[1][1];
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }
}

template __global__ void projection_ewa_3dgs_fused_fwd_kernel_CR<float>(
    const uint32_t C,
    const uint32_t N,
    const bool * __restrict__ mask,
    const float *__restrict__ means,
    const float *__restrict__ covars,
    const float *__restrict__ quats,
    const float *__restrict__ scales,
    const float *__restrict__ opacities,
    const float *__restrict__ viewmats,
    const float *__restrict__ Ks,
    const float *__restrict__ rtgs_projection_adapter,
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    int32_t *__restrict__ radii,
    float *__restrict__ means2d,
    float *__restrict__ depths,
    float *__restrict__ conics,
    float *__restrict__ compensations
);


template __global__ void projection_ewa_3dgs_fused_fwd_kernel_CR<double>(
    const uint32_t C,
    const uint32_t N,
    const bool * __restrict__ mask,
    const double *__restrict__ means,
    const double *__restrict__ covars,
    const double *__restrict__ quats,
    const double *__restrict__ scales,
    const double *__restrict__ opacities,
    const double *__restrict__ viewmats,
    const double *__restrict__ Ks,
    const double *__restrict__ rtgs_projection_adapter,
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    int32_t *__restrict__ radii,
    double *__restrict__ means2d,
    double *__restrict__ depths,
    double *__restrict__ conics,
    double *__restrict__ compensations
);


__global__ void intersect_offset_kernel_CR(
    const uint32_t n_isects,
    const int64_t *__restrict__ isect_ids,
    const uint32_t n_view_group,
    const uint32_t n_tiles,
    const uint32_t view_group_n_bits,
    int32_t *__restrict__ offsets // [tile_height, tile_width, n_view_group]
) {
    // e.g., ids: [1, 1, 1, 3, 3], n_tiles = 6
    // counts: [0, 3, 0, 2, 0, 0]
    // cumsum: [0, 3, 3, 5, 5, 5]
    // offsets: [0, 0, 3, 3, 5, 5]
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects)
        return;

    int64_t isect_id_curr = isect_ids[idx] >> 32;
    int64_t tile_id_curr = isect_id_curr >> view_group_n_bits;
    int64_t view_group_idx_curr = isect_id_curr & ((1ULL << view_group_n_bits) - 1);
    int64_t id_curr = tile_id_curr * n_view_group + view_group_idx_curr;

    if (idx == 0) {
        // write out the offsets until the first valid tile (inclusive)
        for (uint32_t i = 0; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
    if (idx == n_isects - 1) {
        // write out the rest of the offsets
        for (uint32_t i = id_curr + 1; i < n_tiles * n_view_group; ++i)
            offsets[i] = static_cast<int32_t>(n_isects);
    }

    if (idx > 0) {
        // visit the current and previous isect_id and check if the (cid,
        // tile_id) pair changes.
        int64_t isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth
        if (isect_id_prev == isect_id_curr)
            return;

        // write out the offsets between the previous and current tiles
        int64_t tile_id_prev = isect_id_prev >> view_group_n_bits;
        int64_t view_group_idx_prev = isect_id_prev & ((1 << view_group_n_bits) - 1);
        int64_t id_prev = tile_id_prev * n_view_group + view_group_idx_prev;
        for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
}


} // namespace gsplat
