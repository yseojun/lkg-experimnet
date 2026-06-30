from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def install_gsplat_root(gsplat_root: Path | str) -> None:
    import sys

    root = Path(gsplat_root).expanduser().resolve()
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


def render_splats_gsplat(
    splats: Any,
    *,
    viewmat: Any,
    K: Any,
    width: int,
    height: int,
    device: str,
):
    import torch
    import gsplat

    viewmat_t = torch.as_tensor(viewmat, dtype=torch.float32, device=device).contiguous()
    K_t = torch.as_tensor(K, dtype=torch.float32, device=device).contiguous()
    with torch.no_grad():
        image, _alpha, _meta = gsplat.rasterization(
            means=splats.means,
            quats=splats.quats,
            scales=splats.scales,
            opacities=splats.opacities,
            colors=splats.colors,
            viewmats=viewmat_t.unsqueeze(0),
            Ks=K_t.unsqueeze(0),
            sh_degree=int(splats.sh_degree),
            width=int(width),
            height=int(height),
        )
    return image[0].permute(2, 0, 1).clamp(0.0, 1.0).contiguous()


def render_splats_coherent(
    splats: Any,
    *,
    viewmat: Any,
    K: Any,
    width: int,
    height: int,
    device: str,
    tile_size: int,
    near_plane: float,
    far_plane: float,
    camera_model: str,
    debug: bool,
):
    import torch
    from coherent_raster.utils.utils_coherent_raster import unpad, unpatchify_image_shape_matrix
    from gsplat.rendering_coherent_raster import rasterization_CR

    from lkg_experiment.coherent_default.coherent_gsplat_bridge import build_cr_lookup_arrays, lookup_arrays_to_torch

    viewmat_t = torch.as_tensor(viewmat, dtype=torch.float32, device=device).contiguous()
    K_t = torch.as_tensor(K, dtype=torch.float32, device=device).contiguous()
    viewpoint_index = np.zeros((int(height), int(width), 3), dtype=np.uint32)
    lookup = build_cr_lookup_arrays(viewpoint_index, tile_size=int(tile_size), use_remapping=True)
    view_idx_matrix, subpixel_coord_matrix = lookup_arrays_to_torch(lookup, device=str(device))
    adjacent_viewmats = viewmat_t.view(1, 1, 4, 4).contiguous()

    with torch.no_grad():
        colors, _alpha, _meta = rasterization_CR(
            means=splats.means,
            quats=splats.quats,
            scales=splats.scales,
            opacities=splats.opacities,
            colors=splats.colors,
            adjacent_viewmats=adjacent_viewmats,
            Ks=K_t.unsqueeze(0),
            view_idx_matrix=view_idx_matrix,
            subpixel_coord_matrix=subpixel_coord_matrix,
            width=int(width),
            height=int(height),
            sh_degree=int(splats.sh_degree),
            near_plane=float(near_plane),
            far_plane=float(far_plane),
            camera_model=str(camera_model),
            tile_size=int(tile_size),
            is_debug=bool(debug),
        )
        image = unpatchify_image_shape_matrix(colors)
        image = unpad(image, int(height), int(width))
    return image.clamp(0.0, 1.0).contiguous()
