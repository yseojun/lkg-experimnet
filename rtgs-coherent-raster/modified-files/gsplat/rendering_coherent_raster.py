# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

import math
from typing import Any, Dict, Optional, Tuple
import time

import torch
import torch.distributed
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Literal

from .cuda._wrapper import spherical_harmonics

from .cuda._wrapper_coherent_raster import (
    fully_fused_projection_CR,
    isect_tiles_CR,
    isect_offset_encode_CR,
    rasterize_to_pixels_CR
)



def rasterization_CR(
    means: Tensor,  # [N, 3]
    quats: Tensor | None,  # [N, 4]
    scales: Tensor | None,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [(C,) N, D] or [(C,) N, K, 3]
    adjacent_viewmats: Tensor,  # [num_of_viewgroup, view_group_size, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    view_idx_matrix: Tensor,  # [height, width, 3]
    subpixel_coord_matrix: Tensor,  # [C, H, W, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    is_debug: bool = False,
    eps2d: float = 0.3,
    sh_degree: int | None = None,
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Tensor | None = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    covars: Tensor | None = None,
    rtgs_projection_adapter: Tensor | None = None,
    return_timing: bool = False,
) -> Tuple[Tensor, Tensor, Dict]:
    
    # adjacent_viewmats: [num_of_view_group, view_group_size, 4, 4]
    # Ks: [1, 3, 3]
    # means: [N, 3]
    # quats: [N, 4]
    # scales: [N, 3]
    # opacities: [N]
    # colors: [N, 16, 3] = [N, K, 3]
    
    meta = {}
    timing_ms: Dict[str, Any] = {}


    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    projection_start = _timing_stage_start(return_timing, means)
    if is_debug:
        torch.cuda.synchronize()
        tic = time.time()
    proj_results_first = fully_fused_projection_CR(
        None,  # mask
        means,
        covars,
        quats,
        scales,
        adjacent_viewmats[0, :1],
        Ks,
        width,
        height,
        packed=False,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=sparse_grad,
        calc_compensations=(rasterize_mode == "antialiased"),
        camera_model=camera_model,
        opacities=opacities,  # use opacities to compute a tigher bound for radii.
        rtgs_projection_adapter=rtgs_projection_adapter,
    )

    proj_results_last = fully_fused_projection_CR(
        None,  # mask
        means,
        covars,
        quats,
        scales,
        adjacent_viewmats[-1, -1:],
        Ks,
        width,
        height,
        packed=False,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=sparse_grad,
        calc_compensations=(rasterize_mode == "antialiased"),
        camera_model=camera_model,
        opacities=opacities,  # use opacities to compute a tigher bound for radii.
        rtgs_projection_adapter=rtgs_projection_adapter,
    )

    # make mask
    radii_first, _, _, _, _ = proj_results_first
    radii_last, _, _, _, _ = proj_results_last  # [1, N, 2]

    mask_first = (radii_first[:, :, 0] > 0) & (radii_first[:, :, 1] > 0)  
    mask_last = (radii_last[:, :, 0] > 0) & (radii_last[:, :, 1] > 0)

    mask = mask_first | mask_last

    # generate each groups reference_view
    # adjacent_viewmats: [num_of_view_group, view_group_size, 4, 4]
    ref_view_idx = adjacent_viewmats.shape[1] // 2
    reference_viewmat = adjacent_viewmats[:, ref_view_idx, :, :]  # [num_of_view_group, 4, 4]
    Ks = torch.concat([Ks for i in range(reference_viewmat.shape[0])], dim=0)


    proj_results = fully_fused_projection_CR(
        mask,
        means,
        covars,
        quats,
        scales,
        reference_viewmat,
        Ks,
        width,
        height,
        packed=False,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=sparse_grad,
        calc_compensations=(rasterize_mode == "antialiased"),
        camera_model=camera_model,
        opacities=opacities,  # use opacities to compute a tigher bound for radii.
        rtgs_projection_adapter=rtgs_projection_adapter,
    )
    if is_debug:
        torch.cuda.synchronize()
        toc = time.time()
        print(f"projection(): {toc-tic:.8f}")
    timing_ms["cr_projection_ms"] = _timing_stage_elapsed_ms(projection_start, return_timing, means)



    # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
    radii, means2d, depths, conics, compensations = proj_results

    if compensations is not None:
        opacities = opacities * compensations

    
    campos = torch.inverse(reference_viewmat)[:, :3, 3]  # [C, 3]
    dirs = means[None, :, :] - campos[:, None, :]  # [C, N, 3]
    masks = (radii > 0).all(dim=-1)  # [C, N]
    colors = _prepare_colors_for_cr(
        colors,
        sh_degree=sh_degree,
        reference_viewmat_count=reference_viewmat.shape[0],
        dirs=dirs,
        masks=masks,
    )



    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    
    keygen_start = _timing_stage_start(return_timing, means)
    if is_debug:
        torch.cuda.synchronize()
        tic = time.time()
    isect_result = isect_tiles_CR(
        means,
        means2d,
        radii,
        depths,
        reference_viewmat, # [1, 4, 4]
        adjacent_viewmats, # [num_of_adjacent, 4, 4]
        Ks[0],        # [3, 3]
        tile_size,
        tile_width,
        tile_height,
        rtgs_projection_adapter=rtgs_projection_adapter,
        return_timing=return_timing,
    )
    tiles_per_gauss, isect_ids, flatten_ids, translation_values, isect_timing_ms = _unpack_isect_tiles_result(isect_result)
    isect_stage_ms = _timing_stage_elapsed_ms(keygen_start, return_timing, means)
    if is_debug:
        torch.cuda.synchronize()
        toc = time.time()
        print(f"isect_tiles(): {toc-tic:.8f}")
        print(f"n_isects: {isect_ids.shape[0]:,}, inter/tile: {isect_ids.shape[0]/(tile_height*tile_width*adjacent_viewmats.shape[0])}")

    

    # print("rank", world_rank, "Before isect_offset_encode")
    if is_debug:
        torch.cuda.synchronize()
        tic = time.time()
    n_view_group = adjacent_viewmats.shape[0]
    offset_start = _timing_stage_start(return_timing, means)
    isect_offsets = isect_offset_encode_CR(isect_ids, n_view_group, tile_width, tile_height)
    offset_ms = _timing_stage_elapsed_ms(offset_start, return_timing, means)
    if is_debug:
        torch.cuda.synchronize()
        toc = time.time()
        print(f"isect_offsets(): {toc-tic:.8f}")
    cr_isect_ms, cr_sort_ms = _isect_timing_values(isect_timing_ms)
    if return_timing and not math.isfinite(cr_isect_ms):
        cr_isect_ms = isect_stage_ms
    timing_ms["cr_isect_ms"] = cr_isect_ms
    timing_ms["cr_sort_ms"] = cr_sort_ms
    timing_ms["cr_offset_ms"] = offset_ms
    timing_ms["cr_keygen_ms"] = _finite_sum(cr_isect_ms, cr_sort_ms, offset_ms)
    timing_ms["cr_timing_source"] = _isect_timing_source(isect_timing_ms)
    
    means2d = means2d.permute(1, 0, 2)
    conics = conics.permute(1, 0, 2)
    colors = colors.permute(1, 0, 2)

    blend_start = _timing_stage_start(return_timing, means)
    if is_debug:
        torch.cuda.synchronize()
        tic = time.time()
    
    render_colors = rasterize_to_pixels_CR(
        means2d,
        conics,
        colors,
        opacities,
        translation_values,
        view_idx_matrix,
        subpixel_coord_matrix,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds
    )
    if is_debug:
        torch.cuda.synchronize()
        toc = time.time()
        print(f"rasterize_to_pixels(): {toc-tic:.8f}")

    timing_ms["cr_blend_ms"] = _timing_stage_elapsed_ms(blend_start, return_timing, means)
    if return_timing:
        meta["timing_ms"] = timing_ms
    return render_colors, None, meta


def _timing_stage_start(enabled: bool, reference: Tensor) -> float:
    if enabled and reference.is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _timing_stage_elapsed_ms(start: float, enabled: bool, reference: Tensor) -> float:
    if enabled and reference.is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    return float((time.perf_counter() - start) * 1000.0) if enabled else 0.0


def _unpack_isect_tiles_result(result):
    if len(result) == 5:
        return result
    tiles_per_gauss, isect_ids, flatten_ids, translation_values = result
    return tiles_per_gauss, isect_ids, flatten_ids, translation_values, None


def _isect_timing_values(timing_ms) -> tuple[float, float]:
    if timing_ms is None:
        return float("nan"), float("nan")
    try:
        if int(timing_ms.numel()) < 2:
            return float("nan"), float("nan")
        values = timing_ms.detach().cpu().tolist()
        return float(values[0]), float(values[1])
    except Exception:
        return float("nan"), float("nan")


def _isect_timing_source(timing_ms) -> str:
    try:
        if timing_ms is not None and int(timing_ms.numel()) >= 2:
            return "extension_events"
    except Exception:
        pass
    return "legacy_stage_fallback"


def _finite_sum(*values: float) -> float:
    return float(sum(float(value) for value in values if math.isfinite(float(value))))


def _prepare_colors_for_cr(
    colors: Tensor,
    *,
    sh_degree: int | None,
    reference_viewmat_count: int,
    dirs: Tensor,
    masks: Tensor,
) -> Tensor:
    if sh_degree is None:
        if colors.dim() == 2:
            if colors.shape[-1] != 3:
                raise ValueError("precomputed RGB colors must have shape [N,3]")
            return colors.expand(int(reference_viewmat_count), -1, -1).contiguous()
        if colors.dim() == 3:
            if colors.shape[-1] != 3:
                raise ValueError("precomputed RGB colors must have shape [C,N,3]")
            if colors.shape[0] == int(reference_viewmat_count):
                return colors.contiguous()
            if colors.shape[0] == 1:
                return colors.expand(int(reference_viewmat_count), -1, -1).contiguous()
            raise ValueError("precomputed RGB colors camera dimension does not match reference views")
        raise ValueError("precomputed RGB colors must have shape [N,3] or [C,N,3]")

    # Colors are SH coefficients, with shape [N, K, 3] or [C, N, K, 3].
    if colors.dim() == 3:
        shs = colors.expand(int(reference_viewmat_count), -1, -1, -1)  # [C, N, K, 3]
    elif colors.dim() == 4:
        shs = colors
    else:
        raise ValueError("SH colors must have shape [N,K,3] or [C,N,K,3]")
    colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [C, N, 3]
    # make it apple-to-apple with Inria's CUDA Backend.
    return torch.clamp_min(colors + 0.5, 0.0)
