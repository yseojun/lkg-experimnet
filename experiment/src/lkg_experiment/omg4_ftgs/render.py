from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def build_interlaced_viewpoint_index(args: Any, *, width: int, height: int) -> tuple[np.ndarray, dict[str, Any]]:
    mode = str(args.map_mode)
    views = int(args.views)
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("width and height must be positive")
    if views <= 0:
        raise ValueError("views must be positive")
    if mode == "file":
        from lkg_experiment.coherent_default.coherent_raster_experiment import load_viewpoint_index_file

        viewpoint_index, file_view_count, metadata = load_viewpoint_index_file(args.viewpoint_index_path, int(width), int(height))
        if file_view_count is not None and int(file_view_count) != views:
            raise ValueError(f"viewpoint index file has {file_view_count} views, expected {views}")
        return viewpoint_index.astype(np.int32, copy=False), {**dict(metadata), "mode": "file", "views": views}
    if mode == "linear":
        from lkg_experiment.coherent_default.coherent_raster import build_linear_viewpoint_index

        viewpoint_index = build_linear_viewpoint_index(
            int(width),
            int(height),
            views,
            quantize=str(args.coherent_quantize),
        )
        return viewpoint_index.astype(np.int32, copy=False), {
            "mode": "linear",
            "views": views,
            "quantize": str(args.coherent_quantize),
        }
    raise ValueError(f"unsupported map mode: {mode!r}")


def estimate_orbit_center(*, c2w: Any, splat_means: Any, orbit_center_distance: float):
    import torch

    c2w_t = torch.as_tensor(c2w, dtype=torch.float32, device=getattr(splat_means, "device", None))
    means_t = torch.as_tensor(splat_means, dtype=torch.float32, device=c2w_t.device)
    camera_position = c2w_t[:3, 3]
    if float(orbit_center_distance) > 0.0:
        distance = float(orbit_center_distance)
    elif int(means_t.shape[0]) > 0:
        scene_center = means_t.mean(dim=0)
        distance = max(float(torch.linalg.norm(scene_center - camera_position).item()), 1e-4)
    else:
        distance = 1.0
    return (camera_position + c2w_t[:3, 2] * distance).contiguous()


def synthesize_interlaced_viewmats(
    *,
    c2w: Any,
    orbit_center: Any,
    views: int,
    cluster_size: int,
    view_degree: float,
    orbit_direction: int,
    device: str,
):
    import torch
    from lkg_experiment.coherent_default.coherent_raster_experiment import OrbitViewSynthesizer

    if int(views) <= 0:
        raise ValueError("views must be positive")
    if int(cluster_size) <= 0:
        raise ValueError("cluster_size must be positive")
    view_labels = np.arange(int(views), dtype=np.int32)
    synthesizer = OrbitViewSynthesizer(
        view_labels=view_labels,
        source_view_count=int(views),
        cluster_size=int(cluster_size),
        view_degree=float(view_degree),
        orbit_direction=int(orbit_direction),
        device=str(device),
    )
    c2w_t = torch.as_tensor(c2w, dtype=torch.float32, device=str(device)).contiguous()
    center_t = torch.as_tensor(orbit_center, dtype=torch.float32, device=str(device)).contiguous()
    return synthesizer(c2w_t, center_t).contiguous()


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


def render_splats_interlaced_coherent(
    splats: Any,
    *,
    adjacent_viewmats: Any,
    K: Any,
    viewpoint_index: np.ndarray,
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

    viewmats_t = torch.as_tensor(adjacent_viewmats, dtype=torch.float32, device=device).contiguous()
    K_t = torch.as_tensor(K, dtype=torch.float32, device=device).contiguous()
    lookup = build_cr_lookup_arrays(np.asarray(viewpoint_index), tile_size=int(tile_size), use_remapping=True)
    view_idx_matrix, subpixel_coord_matrix = lookup_arrays_to_torch(lookup, device=str(device))

    with torch.no_grad():
        colors, _alpha, _meta = rasterization_CR(
            means=splats.means,
            quats=splats.quats,
            scales=splats.scales,
            opacities=splats.opacities,
            colors=splats.colors,
            adjacent_viewmats=viewmats_t,
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
