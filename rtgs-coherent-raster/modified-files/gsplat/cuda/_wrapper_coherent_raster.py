# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------
# MODIFICATION NOTICE:
# This file has been modified by POSTECH and ETRI on 2026-04-14.
# Changes: Added modifications related to CoherentRaster
# The modifications are also licensed under the Apache License, Version 2.0.
# ------------------------------------------------------------------

from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from ._wrapper import _make_lazy_cuda_func, _make_lazy_cuda_obj


def fully_fused_projection_CR(
    mask: Tensor | None, # [N, 2]
    means: Tensor,  # [N, 3]
    covars: Tensor | None,  # [N, 6] or None
    quats: Tensor | None,  # [N, 4] or None
    scales: Tensor | None,  # [N, 3] or None
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    opacities: Tensor | None = None,  # [N] or None
    rtgs_projection_adapter: Tensor | None = None,  # [6] or None
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    assert Ks.size() == (C, 3, 3), Ks.size()
    means = means.contiguous()
    if covars is not None:
        assert covars.size() == (N, 6), covars.size()
        covars = covars.contiguous()
    else:
        assert quats is not None, "covars or quats is required"
        assert scales is not None, "covars or scales is required"
        assert quats.size() == (N, 4), quats.size()
        assert scales.size() == (N, 3), scales.size()
        quats = quats.contiguous()
        scales = scales.contiguous()
    if sparse_grad:
        assert packed, "sparse_grad is only supported when packed is True"
    if opacities is not None:
        assert opacities.size() == (N,), opacities.size()
        opacities = opacities.contiguous()
    if rtgs_projection_adapter is not None:
        assert rtgs_projection_adapter.size() == (6,), rtgs_projection_adapter.size()
        rtgs_projection_adapter = rtgs_projection_adapter.to(device=means.device, dtype=means.dtype).contiguous()

    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()


    camera_model_type = _make_lazy_cuda_obj(
            f"CameraModelType.{camera_model.upper()}"
        )


    # "covars" and {"quats", "scales"} are mutually exclusive
    radii, means2d, depths, conics, compensations = _make_lazy_cuda_func(
        "projection_ewa_3dgs_fused_fwd_CR"
    )(
        mask,
        means,
        covars,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        calc_compensations,
        camera_model_type,
        rtgs_projection_adapter,
    )

    return radii, means2d, depths, conics, compensations













@torch.no_grad()
def isect_tiles_CR(
    means3d: Tensor,  # [N, 3]
    means2d: Tensor,  # [C, N, 2] or [nnz, 2]
    radii: Tensor,  # [C, N, 2] or [nnz, 2]
    depths: Tensor,  # [C, N] or [nnz]
    reference_viewmat: Tensor,  # [1, 4, 4]
    adjacent_viewmats: Tensor,  # [num_of_adjacent_view, 4, 4]
    K: Tensor,  # [3, 3]
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    packed: bool = False,
    n_cameras: int | None = None,
    camera_ids: Tensor | None = None,
    gaussian_ids: Tensor | None = None,
    rtgs_projection_adapter: Tensor | None = None,
) -> Tuple[Tensor, Tensor, Tensor]:

    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.size()
        assert radii.shape == (nnz, 2), radii.size()
        assert depths.shape == (nnz,), depths.size()
        assert camera_ids is not None, "camera_ids is required if packed is True"
        assert gaussian_ids is not None, "gaussian_ids is required if packed is True"
        assert n_cameras is not None, "n_cameras is required if packed is True"
        camera_ids = camera_ids.contiguous()
        gaussian_ids = gaussian_ids.contiguous()
        C = n_cameras

    else:
        C, N, _ = means2d.shape
        assert means2d.shape == (C, N, 2), means2d.size()
        assert radii.shape == (C, N, 2), radii.size()
        assert depths.shape == (C, N), depths.size()
    if rtgs_projection_adapter is not None:
        assert rtgs_projection_adapter.size() == (6,), rtgs_projection_adapter.size()
        rtgs_projection_adapter = rtgs_projection_adapter.to(device=means3d.device, dtype=means3d.dtype).contiguous()

    tiles_per_gauss, isect_ids, flatten_ids, translation_values = _make_lazy_cuda_func("intersect_tile_CR")(
        means3d.contiguous(),
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        reference_viewmat,
        adjacent_viewmats,
        K,         # [3, 3]
        camera_ids,
        gaussian_ids,
        C,
        tile_size,
        tile_width,
        tile_height,
        sort,
        rtgs_projection_adapter,
    )

    
    return tiles_per_gauss, isect_ids, flatten_ids, translation_values


@torch.no_grad()
def isect_offset_encode_CR(
    isect_ids: Tensor, n_cameras: int, tile_width: int, tile_height: int
) -> Tensor:
    """Encodes intersection ids to offsets.

    Args:
        isect_ids: Intersection ids. [n_isects]
        n_cameras: Number of cameras.
        tile_width: Tile width.
        tile_height: Tile height.

    Returns:
        Offsets. [C, tile_height, tile_width]
    """
    return _make_lazy_cuda_func("intersect_offset_CR")(
        isect_ids.contiguous(), n_cameras, tile_width, tile_height
    )







def rasterize_to_pixels_CR(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    translation_values: Tensor,
    view_idx_matrix: Tensor,
    subpixel_coord_matrix: Tensor,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    backgrounds: Tensor | None = None
) -> Tuple[Tensor, Tensor]:

    render_colors = _make_lazy_cuda_func(
            "rasterize_to_pixels_3dgs_fwd_CR"
        )(
            means2d.contiguous(),
            conics.contiguous(),
            colors.contiguous(),
            opacities.contiguous(),
            translation_values.contiguous(),
            view_idx_matrix.contiguous(),
            subpixel_coord_matrix.contiguous(),
            backgrounds,
            isect_offsets.contiguous(),
            flatten_ids.contiguous(),
        )[0]

    return render_colors
