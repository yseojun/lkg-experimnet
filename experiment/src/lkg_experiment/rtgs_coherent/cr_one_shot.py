from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from lkg_experiment.coherent_default.coherent_gsplat_bridge import (
    CRLookupArrays,
    build_cr_lookup_arrays,
    lookup_arrays_to_torch,
)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_rtgs_projection_adapter(camera: Any, *, device: str, dtype: Any, enabled: bool):
    import torch

    info: dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": None,
        "values": None,
    }
    if not bool(enabled):
        info["reason"] = "disabled"
        return None, info

    fl_x = _optional_float(getattr(camera, "fl_x", None))
    fl_y = _optional_float(getattr(camera, "fl_y", None))
    fovx = _optional_float(getattr(camera, "FoVx", None))
    fovy = _optional_float(getattr(camera, "FoVy", None))
    cx = _optional_float(getattr(camera, "cx", None))
    cy = _optional_float(getattr(camera, "cy", None))
    width = int(getattr(camera, "image_width", 0) or 0)
    height = int(getattr(camera, "image_height", 0) or 0)
    if fl_x is None or fl_y is None or fl_x <= 0.0 or fl_y <= 0.0 or width <= 0 or height <= 0:
        info["reason"] = "missing_positive_explicit_intrinsics"
        return None, info
    if fovx is None or fovy is None or fovx > 0.0 or fovy > 0.0:
        info["reason"] = "fov_not_nonpositive_sentinel"
        return None, info

    tan_x = math.tan(float(fovx) * 0.5)
    tan_y = math.tan(float(fovy) * 0.5)
    if abs(tan_x) < 1e-12 or abs(tan_y) < 1e-12:
        info["reason"] = "invalid_fov_tangent"
        return None, info

    fov_fx = float(width) / (2.0 * tan_x)
    fov_fy = float(height) / (2.0 * tan_y)
    scale_x = float(fl_x) / fov_fx
    scale_y = float(fl_y) / fov_fy
    values = [fov_fx, fov_fy, scale_x, scale_y, float(cx if cx is not None else width / 2.0), float(cy if cy is not None else height / 2.0)]
    adapter = torch.tensor(values, dtype=dtype, device=device).contiguous()
    info["applied"] = True
    info["reason"] = "explicit_intrinsics_with_nonpositive_fov_sentinel"
    info["values"] = {
        "fov_fx": float(fov_fx),
        "fov_fy": float(fov_fy),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "cx": float(values[4]),
        "cy": float(values[5]),
    }
    return adapter, info


def crop_viewpoint_index_to_viewport(viewpoint_index: np.ndarray, viewport: Any) -> np.ndarray:
    view_hwc = np.asarray(viewpoint_index)
    if view_hwc.ndim != 3 or view_hwc.shape[2] != 3:
        raise ValueError(f"viewpoint_index must have shape [H,W,3], got {view_hwc.shape}")

    offset_x = int(getattr(viewport, "offset_x"))
    offset_y = int(getattr(viewport, "offset_y"))
    render_width = int(getattr(viewport, "render_width"))
    render_height = int(getattr(viewport, "render_height"))
    panel_width = int(getattr(viewport, "panel_width"))
    panel_height = int(getattr(viewport, "panel_height"))
    if min(render_width, render_height, panel_width, panel_height) <= 0:
        raise ValueError("viewport dimensions must be positive")
    if offset_x < 0 or offset_y < 0:
        raise ValueError("viewport offsets must be non-negative")

    x1 = offset_x + render_width
    y1 = offset_y + render_height
    if view_hwc.shape[:2] != (panel_height, panel_width):
        raise ValueError(f"viewpoint_index panel shape must be {(panel_height, panel_width)}, got {view_hwc.shape[:2]}")
    if x1 > panel_width or y1 > panel_height:
        raise ValueError("viewport content bounds exceed panel dimensions")

    return np.ascontiguousarray(view_hwc[offset_y:y1, offset_x:x1, :])


def build_viewport_cr_lookup(
    viewpoint_index: np.ndarray,
    viewport: Any,
    *,
    tile_size: int,
    use_remapping: bool,
) -> CRLookupArrays:
    cropped = crop_viewpoint_index_to_viewport(viewpoint_index, viewport)
    return build_cr_lookup_arrays(cropped, tile_size=int(tile_size), use_remapping=bool(use_remapping))


def paste_viewport_image_into_panel(viewport_image: Any, viewport: Any, *, background: Any | None = None):
    import torch

    image = torch.as_tensor(viewport_image)
    render_width = int(getattr(viewport, "render_width"))
    render_height = int(getattr(viewport, "render_height"))
    panel_width = int(getattr(viewport, "panel_width"))
    panel_height = int(getattr(viewport, "panel_height"))
    offset_x = int(getattr(viewport, "offset_x"))
    offset_y = int(getattr(viewport, "offset_y"))
    expected_shape = (3, render_height, render_width)
    if tuple(image.shape) != expected_shape:
        raise ValueError(f"viewport_image shape {tuple(image.shape)} does not match {expected_shape}")

    panel = image.new_zeros((3, panel_height, panel_width))
    if background is not None:
        bg = torch.as_tensor(background, dtype=image.dtype, device=image.device).reshape(3, 1, 1)
        panel[:] = bg
    panel[:, offset_y : offset_y + render_height, offset_x : offset_x + render_width] = image
    return panel.contiguous()


def render_rtgs_cr_one_shot_interlaced_once(context: Any, *, variant: Any):
    import torch
    from coherent_raster.utils.utils_coherent_raster import unpad, unpatchify_image_shape_matrix
    from gsplat.rendering_coherent_raster import rasterization_CR
    from lkg_experiment.rtgs_coherent import cr_66views
    from lkg_experiment.rtgs_coherent.views66 import synthesize_grouped_viewmats

    device = str(context.args.device)
    lookup = build_viewport_cr_lookup(
        context.viewpoint_index,
        context.viewport,
        tile_size=int(context.args.tile_size),
        use_remapping=bool(variant.use_remapping),
    )
    view_idx_matrix, subpixel_coord_matrix = lookup_arrays_to_torch(lookup, device=device)
    view_labels = np.arange(int(context.source_view_count), dtype=np.int32)
    adjacent_viewmats = synthesize_grouped_viewmats(
        c2w=context.anchor_c2w,
        orbit_center=context.orbit_center,
        view_labels=view_labels,
        source_view_count=int(context.source_view_count),
        cluster_size=int(variant.cluster_size) if bool(variant.reuse_enabled) else 1,
        view_degree=float(context.view_degree),
        orbit_direction=int(context.orbit_direction),
        device=device,
    )
    ref_idx = int(adjacent_viewmats.shape[1]) // 2
    reference_centers = torch.linalg.inv(adjacent_viewmats[:, ref_idx])[:, :3, 3]
    colors = [
        cr_66views.evaluate_rtgs_colors(
            context.runtime.official.gaussians,
            timestamp=float(context.timestamp),
            camera_center=center,
            mask=context.geometry.mask,
        )
        for center in reference_centers
    ]
    snapshot = cr_66views.snapshot_from_geometry(context.geometry, colors=torch.stack(colors, dim=0).contiguous())
    anchor_camera = cr_66views.synthetic_camera_from_viewmat_preserving_rtgs_contract(
        context.cr_anchor_camera_cuda,
        torch.linalg.inv(context.anchor_c2w),
        uid=0,
        image_name="one_shot_clustered_anchor",
        timestamp=float(context.timestamp),
        device=device,
        width=int(context.render_width),
        height=int(context.render_height),
        crop_to_fill=bool(context.crop_to_fill),
    ).cuda()
    _, K = cr_66views.rtgs_camera_to_gsplat_inputs(anchor_camera, device=device)
    rtgs_projection_adapter, _adapter_info = build_rtgs_projection_adapter(
        anchor_camera,
        device=device,
        dtype=snapshot.means.dtype,
        enabled=bool(getattr(context.args, "rtgs_compat_projection", True))
        and not bool(getattr(context, "fov_normalization", {}).get("applied", False)),
    )
    backgrounds = None if context.runtime.background is None else context.runtime.background.contiguous()
    should_sync = device.startswith("cuda") and torch.cuda.is_available()
    if should_sync:
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        rendered, _, _ = rasterization_CR(
            means=snapshot.means,
            quats=None,
            scales=None,
            opacities=snapshot.opacities,
            colors=snapshot.colors,
            adjacent_viewmats=adjacent_viewmats,
            Ks=K.unsqueeze(0).contiguous(),
            view_idx_matrix=view_idx_matrix,
            subpixel_coord_matrix=subpixel_coord_matrix,
            width=int(context.render_width),
            height=int(context.render_height),
            sh_degree=None,
            near_plane=float(context.args.near_plane),
            far_plane=float(context.args.far_plane),
            backgrounds=backgrounds,
            camera_model=str(context.args.camera_model),
            tile_size=int(context.args.tile_size),
            is_debug=bool(context.args.debug_cr),
            covars=snapshot.covars,
            rtgs_projection_adapter=rtgs_projection_adapter,
        )
        viewport_image = unpatchify_image_shape_matrix(rendered)
        viewport_image = unpad(viewport_image, int(context.render_height), int(context.render_width)).clamp(0.0, 1.0).contiguous()
        panel_image = paste_viewport_image_into_panel(
            viewport_image,
            context.viewport,
            background=context.runtime.background,
        ).clamp(0.0, 1.0)
    if should_sync:
        torch.cuda.synchronize()
    return panel_image.contiguous(), (time.perf_counter() - start) * 1000.0
