from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from lkg_experiment.coherent_default.coherent_gsplat_bridge import (
    CRLookupArrays,
    build_cr_lookup_arrays,
    lookup_arrays_to_torch,
)


@dataclass(frozen=True)
class RtgsCrFrameTiming:
    dynamic_geometry_ms: float = 0.0
    temporal_opacity_ms: float = 0.0
    snapshot_compaction_ms: float = 0.0
    dynamic_color_ms: float = 0.0
    cr_projection_ms: float = 0.0
    cr_keygen_ms: float = 0.0
    cr_sort_ms: float = 0.0
    cr_blend_ms: float = 0.0
    lkg_unpatchify_ms: float = 0.0
    lkg_unpad_ms: float = 0.0
    lkg_panel_paste_ms: float = 0.0

    @property
    def rtgs_dynamic_total_ms(self) -> float:
        return float(self.dynamic_geometry_ms + self.temporal_opacity_ms + self.snapshot_compaction_ms + self.dynamic_color_ms)

    @property
    def cr_core_total_ms(self) -> float:
        return float(self.cr_projection_ms + self.cr_keygen_ms + self.cr_sort_ms + self.cr_blend_ms)

    @property
    def lkg_interlace_post_ms(self) -> float:
        return float(self.lkg_unpatchify_ms + self.lkg_unpad_ms + self.lkg_panel_paste_ms)

    @property
    def frame_ms_without_lkg(self) -> float:
        return float(self.rtgs_dynamic_total_ms + self.cr_core_total_ms)

    @property
    def fps_without_lkg(self) -> float:
        return 1000.0 / self.frame_ms_without_lkg if self.frame_ms_without_lkg > 0.0 else 0.0

    @property
    def frame_ms_with_lkg_interlace(self) -> float:
        return float(self.frame_ms_without_lkg + self.lkg_interlace_post_ms)

    @property
    def fps_with_lkg_interlace(self) -> float:
        return 1000.0 / self.frame_ms_with_lkg_interlace if self.frame_ms_with_lkg_interlace > 0.0 else 0.0

    def to_metric_dict(self) -> dict[str, float]:
        return {
            "dynamic_geometry_ms": float(self.dynamic_geometry_ms),
            "temporal_opacity_ms": float(self.temporal_opacity_ms),
            "snapshot_compaction_ms": float(self.snapshot_compaction_ms),
            "dynamic_color_ms": float(self.dynamic_color_ms),
            "rtgs_dynamic_total_ms": float(self.rtgs_dynamic_total_ms),
            "cr_projection_ms": float(self.cr_projection_ms),
            "cr_keygen_ms": float(self.cr_keygen_ms),
            "cr_sort_ms": float(self.cr_sort_ms),
            "cr_blend_ms": float(self.cr_blend_ms),
            "cr_core_total_ms": float(self.cr_core_total_ms),
            "lkg_unpatchify_ms": float(self.lkg_unpatchify_ms),
            "lkg_unpad_ms": float(self.lkg_unpad_ms),
            "lkg_panel_paste_ms": float(self.lkg_panel_paste_ms),
            "lkg_interlace_post_ms": float(self.lkg_interlace_post_ms),
            "frame_ms_without_lkg": float(self.frame_ms_without_lkg),
            "fps_without_lkg": float(self.fps_without_lkg),
            "frame_ms_with_lkg_interlace": float(self.frame_ms_with_lkg_interlace),
            "fps_with_lkg_interlace": float(self.fps_with_lkg_interlace),
        }


@dataclass(frozen=True)
class RtgsCrOneShotState:
    device: str
    adjacent_viewmats: Any
    snapshot: Any
    K: Any
    rtgs_projection_adapter: Any
    backgrounds: Any
    dynamic_geometry_timing: Mapping[str, float]
    dynamic_color_ms: float


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


def _should_sync(device: str) -> bool:
    if not str(device).startswith("cuda"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _stage_start(device: str) -> float:
    if _should_sync(device):
        import torch

        torch.cuda.synchronize()
    return time.perf_counter()


def _stage_elapsed_ms(start: float, device: str) -> float:
    if _should_sync(device):
        import torch

        torch.cuda.synchronize()
    return float((time.perf_counter() - start) * 1000.0)


def prepare_rtgs_cr_one_shot_state(context: Any, *, variant: Any) -> RtgsCrOneShotState:
    import torch
    from lkg_experiment.rtgs_coherent import cr_66views
    from lkg_experiment.rtgs_coherent.views66 import synthesize_grouped_viewmats

    device = str(context.args.device)
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

    geometry, dynamic_geometry_timing = cr_66views.materialize_rtgs_geometry(
        context.runtime.official.gaussians,
        timestamp=float(context.timestamp),
        return_timing=True,
    )
    color_start = _stage_start(device)
    colors = [
        cr_66views.evaluate_rtgs_colors(
            context.runtime.official.gaussians,
            timestamp=float(context.timestamp),
            camera_center=center,
            mask=geometry.mask,
            means_for_color=geometry.means,
        )
        for center in reference_centers
    ]
    dynamic_color_ms = _stage_elapsed_ms(color_start, device)
    snapshot = cr_66views.snapshot_from_geometry(geometry, colors=torch.stack(colors, dim=0).contiguous())
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
    return RtgsCrOneShotState(
        device=device,
        adjacent_viewmats=adjacent_viewmats,
        snapshot=snapshot,
        K=K,
        rtgs_projection_adapter=rtgs_projection_adapter,
        backgrounds=backgrounds,
        dynamic_geometry_timing=dynamic_geometry_timing,
        dynamic_color_ms=float(dynamic_color_ms),
    )


def render_rtgs_cr_one_shot_viewport_image(context: Any, state: RtgsCrOneShotState, lookup: CRLookupArrays):
    import torch
    from coherent_raster.utils.utils_coherent_raster import unpad, unpatchify_image_shape_matrix
    from gsplat.rendering_coherent_raster import rasterization_CR

    device = str(state.device)
    view_idx_matrix, subpixel_coord_matrix = lookup_arrays_to_torch(lookup, device=device)
    with torch.no_grad():
        rendered, _, cr_meta = rasterization_CR(
            means=state.snapshot.means,
            quats=None,
            scales=None,
            opacities=state.snapshot.opacities,
            colors=state.snapshot.colors,
            adjacent_viewmats=state.adjacent_viewmats,
            Ks=state.K.unsqueeze(0).contiguous(),
            view_idx_matrix=view_idx_matrix,
            subpixel_coord_matrix=subpixel_coord_matrix,
            width=int(context.render_width),
            height=int(context.render_height),
            sh_degree=None,
            near_plane=float(context.args.near_plane),
            far_plane=float(context.args.far_plane),
            backgrounds=state.backgrounds,
            camera_model=str(context.args.camera_model),
            tile_size=int(context.args.tile_size),
            is_debug=bool(context.args.debug_cr),
            covars=state.snapshot.covars,
            rtgs_projection_adapter=state.rtgs_projection_adapter,
            return_timing=True,
        )
        unpatchify_start = _stage_start(device)
        viewport_image = unpatchify_image_shape_matrix(rendered)
        lkg_unpatchify_ms = _stage_elapsed_ms(unpatchify_start, device)
        unpad_start = _stage_start(device)
        viewport_image = unpad(viewport_image, int(context.render_height), int(context.render_width)).clamp(0.0, 1.0).contiguous()
        lkg_unpad_ms = _stage_elapsed_ms(unpad_start, device)
    cr_timing = dict((cr_meta or {}).get("timing_ms", {}))
    timing = RtgsCrFrameTiming(
        dynamic_geometry_ms=float(state.dynamic_geometry_timing.get("dynamic_geometry_ms", 0.0)),
        temporal_opacity_ms=float(state.dynamic_geometry_timing.get("temporal_opacity_ms", 0.0)),
        snapshot_compaction_ms=float(state.dynamic_geometry_timing.get("snapshot_compaction_ms", 0.0)),
        dynamic_color_ms=float(state.dynamic_color_ms),
        cr_projection_ms=float(cr_timing.get("cr_projection_ms", 0.0)),
        cr_keygen_ms=float(cr_timing.get("cr_keygen_ms", 0.0)),
        cr_sort_ms=float(cr_timing.get("cr_sort_ms", 0.0)),
        cr_blend_ms=float(cr_timing.get("cr_blend_ms", 0.0)),
        lkg_unpatchify_ms=float(lkg_unpatchify_ms),
        lkg_unpad_ms=float(lkg_unpad_ms),
        lkg_panel_paste_ms=0.0,
    )
    return viewport_image.contiguous(), timing


def render_rtgs_cr_one_shot_interlaced_once(context: Any, *, variant: Any):
    state = prepare_rtgs_cr_one_shot_state(context, variant=variant)
    lookup = build_viewport_cr_lookup(
        context.viewpoint_index,
        context.viewport,
        tile_size=int(context.args.tile_size),
        use_remapping=bool(variant.use_remapping),
    )
    viewport_image, timing = render_rtgs_cr_one_shot_viewport_image(context, state, lookup)
    paste_start = _stage_start(state.device)
    panel_image = paste_viewport_image_into_panel(
        viewport_image,
        context.viewport,
        background=context.runtime.background,
    ).clamp(0.0, 1.0)
    lkg_panel_paste_ms = _stage_elapsed_ms(paste_start, state.device)
    timing = RtgsCrFrameTiming(
        dynamic_geometry_ms=timing.dynamic_geometry_ms,
        temporal_opacity_ms=timing.temporal_opacity_ms,
        snapshot_compaction_ms=timing.snapshot_compaction_ms,
        dynamic_color_ms=timing.dynamic_color_ms,
        cr_projection_ms=timing.cr_projection_ms,
        cr_keygen_ms=timing.cr_keygen_ms,
        cr_sort_ms=timing.cr_sort_ms,
        cr_blend_ms=timing.cr_blend_ms,
        lkg_unpatchify_ms=timing.lkg_unpatchify_ms,
        lkg_unpad_ms=timing.lkg_unpad_ms,
        lkg_panel_paste_ms=float(lkg_panel_paste_ms),
    )
    return panel_image.contiguous(), timing
