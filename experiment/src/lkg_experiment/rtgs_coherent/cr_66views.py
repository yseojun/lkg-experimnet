from __future__ import annotations

import argparse
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from lkg_experiment.coherent_default.coherent_raster import build_linear_viewpoint_index
from lkg_experiment.coherent_default.coherent_raster_experiment import (
    OrbitViewSynthesizer,
    load_viewpoint_index_file,
    tensor_to_hwc_uint8,
)
from lkg_experiment.rtgs_coherent.cli import (
    DEFAULT_GENERATED_ROOT,
    RtgsSimpleCamera,
    _install_gsplat_root,
    _json_ready,
    compute_pair_metrics,
    evaluate_rtgs_colors,
    materialize_rtgs_geometry,
    render_rtgs_coherent,
    rtgs_camera_from_gsplat_viewmat,
    rtgs_camera_to_gsplat_inputs,
    snapshot_from_geometry,
)
from lkg_experiment.rtgs_coherent.cr_1view import (
    adapt_snapshot_for_rtgs_compat_projection,
    normalize_explicit_intrinsics_fov,
)
from lkg_experiment.rtgs_coherent.official_1view import (
    _diff_image,
    build_parser as build_official_parser,
    camera_render_contract_values,
    prepare_official_rtgs_1view,
)


DEFAULT_CR66_OUTPUT_ROOT = DEFAULT_GENERATED_ROOT / "rtgs_cr_66views"


def build_parser() -> argparse.ArgumentParser:
    parser = build_official_parser()
    parser.description = "Render RTGS + CoherentRaster sampled views plus a 66-view LKG interlaced image"
    parser.add_argument("--gsplat-root", default=str(Path(__file__).resolve().parents[4] / "gsplat"))
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=100.0)
    parser.add_argument("--camera-model", choices=("pinhole", "ortho", "fisheye"), default="pinhole")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=2560)
    parser.add_argument("--views", type=int, default=66)
    parser.add_argument("--sample-save-views", type=int, default=5)
    parser.add_argument("--sample-view-indices", default=None)
    parser.add_argument("--view-degree", type=float, default=53.0)
    parser.add_argument("--orbit-direction", type=int, default=-1)
    parser.add_argument("--orbit-center-distance", type=float, default=0.0)
    parser.add_argument("--map-mode", choices=("file", "linear"), default="file")
    parser.add_argument(
        "--viewpoint-index-path",
        default=str(DEFAULT_GENERATED_ROOT / "lkg_go_1440x2560_66_views_lkg_calibration.npz"),
    )
    parser.add_argument("--coherent-quantize", choices=("floor", "nearest"), default="floor")
    parser.add_argument("--cluster-size", type=int, default=1)
    parser.add_argument("--cr-remapping", choices=("on", "off"), default="on")
    parser.add_argument("--interlace-mode", choices=("compose",), default="compose")
    parser.add_argument(
        "--compare-official-sampled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render official RTGS synthetic images for sampled views and record metrics",
    )
    parser.add_argument("--write-interlaced", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--normalize-explicit-intrinsics-fov",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Diagnostic only; keep disabled for N3DV acceptance.",
    )
    parser.add_argument(
        "--rtgs-compat-projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the Task 3 RTGS-compatible CR projection adapter for sentinel-FoV cameras.",
    )
    parser.add_argument("--no-crop-to-fill", action="store_true")
    parser.add_argument("--debug-cr", action="store_true")
    return parser


def default_cr66_output_path(
    model_path: Path | str,
    *,
    split: str,
    camera_index: int,
    timestamp: float,
    run_label: str | None,
    unique_label: str | None = None,
) -> Path:
    scene_name = Path(model_path).expanduser().name
    if run_label:
        leaf = str(run_label)
    else:
        suffix = unique_label or datetime.now().strftime("%Y%m%d_%H%M%S")
        leaf = f"{scene_name}_t{float(timestamp):.6f}_{split}_{int(camera_index)}_{suffix}"
    return DEFAULT_CR66_OUTPUT_ROOT / scene_name / leaf


def resolve_sample_view_indices(views: int, sample_save_views: int, explicit: str | None) -> list[int]:
    view_count = int(views)
    if view_count <= 0:
        raise ValueError("views must be positive")
    if explicit:
        parts = [part for part in re.split(r"[\s,]+", str(explicit).strip()) if part]
        indices = [int(part) for part in parts]
    else:
        sample_count = int(sample_save_views)
        if sample_count <= 0:
            return []
        if sample_count >= view_count:
            indices = list(range(view_count))
        elif sample_count == 1:
            indices = [0]
        else:
            indices = np.linspace(0, view_count - 1, sample_count).round().astype(np.int64).tolist()
    seen = set()
    unique = []
    for index in indices:
        index = int(index)
        if index < 0 or index >= view_count:
            raise ValueError(f"sample view index {index} out of range for {view_count} views")
        if index in seen:
            raise ValueError(f"duplicate sample view index: {index}")
        seen.add(index)
        unique.append(index)
    return unique


def validate_viewpoint_index(viewpoint_index: np.ndarray, *, source_view_count: int) -> dict[str, int]:
    view_hwc = np.asarray(viewpoint_index)
    if view_hwc.ndim != 3 or view_hwc.shape[2] != 3:
        raise ValueError(f"viewpoint_index must have shape [H,W,3], got {view_hwc.shape}")
    if int(source_view_count) <= 0:
        raise ValueError("source_view_count must be positive")
    min_view = int(view_hwc.min()) if view_hwc.size else 0
    max_view = int(view_hwc.max()) if view_hwc.size else 0
    if min_view < 0:
        raise ValueError("viewpoint_index must not contain negative view ids")
    if max_view >= int(source_view_count):
        raise ValueError(f"viewpoint_index max {max_view} exceeds source view count {int(source_view_count)}")
    return {
        "height": int(view_hwc.shape[0]),
        "width": int(view_hwc.shape[1]),
        "min_view": min_view,
        "max_view": max_view,
        "source_view_count": int(source_view_count),
    }


def compose_interlaced_from_views(views: Any, viewpoint_index: np.ndarray):
    import torch

    view_tensor = torch.as_tensor(views)
    if view_tensor.ndim != 4 or int(view_tensor.shape[1]) != 3:
        raise ValueError(f"views must have shape [V,3,H,W], got {tuple(view_tensor.shape)}")
    view_hwc = np.asarray(viewpoint_index)
    validate_viewpoint_index(view_hwc, source_view_count=int(view_tensor.shape[0]))
    height = int(view_tensor.shape[2])
    width = int(view_tensor.shape[3])
    if view_hwc.shape != (height, width, 3):
        raise ValueError(f"viewpoint_index shape must be {(height, width, 3)}, got {view_hwc.shape}")

    device = view_tensor.device
    indices = torch.as_tensor(view_hwc, dtype=torch.long, device=device)
    yy = torch.arange(height, dtype=torch.long, device=device).view(height, 1).expand(height, width)
    xx = torch.arange(width, dtype=torch.long, device=device).view(1, width).expand(height, width)
    channels = [view_tensor[indices[:, :, channel], channel, yy, xx] for channel in range(3)]
    return torch.stack(channels, dim=0).contiguous()


def synthetic_camera_from_viewmat_preserving_rtgs_contract(
    anchor_camera: Any,
    viewmat: Any,
    *,
    uid: int,
    image_name: str,
    timestamp: float,
    device: str,
    width: int | None = None,
    height: int | None = None,
    crop_to_fill: bool = True,
) -> RtgsSimpleCamera:
    import torch

    fovx = _optional_float(getattr(anchor_camera, "FoVx", None))
    fovy = _optional_float(getattr(anchor_camera, "FoVy", None))
    fl_x = _optional_float(getattr(anchor_camera, "fl_x", None))
    fl_y = _optional_float(getattr(anchor_camera, "fl_y", None))
    if fl_x is None or fl_y is None or fl_x <= 0.0 or fl_y <= 0.0 or fovx is None or fovy is None:
        return rtgs_camera_from_gsplat_viewmat(
            anchor_camera,
            viewmat,
            uid=uid,
            image_name=image_name,
            timestamp=timestamp,
            device=device,
            width=width,
            height=height,
            crop_to_fill=crop_to_fill,
        )
    if fovx > 0.0 or fovy > 0.0:
        return rtgs_camera_from_gsplat_viewmat(
            anchor_camera,
            viewmat,
            uid=uid,
            image_name=image_name,
            timestamp=timestamp,
            device=device,
            width=width,
            height=height,
            crop_to_fill=crop_to_fill,
        )

    viewmat_t = torch.as_tensor(viewmat, dtype=torch.float32, device="cpu")
    if tuple(viewmat_t.shape) != (4, 4):
        raise ValueError("viewmat must have shape [4,4]")
    source_width = int(anchor_camera.image_width)
    source_height = int(anchor_camera.image_height)
    target_width = source_width if width is None else int(width)
    target_height = source_height if height is None else int(height)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("width and height must be positive")
    if target_width == source_width and target_height == source_height:
        scaled_fl_x = fl_x
        scaled_fl_y = fl_y
        cx = float(anchor_camera.cx)
        cy = float(anchor_camera.cy)
    else:
        scale_w = float(target_width) / float(source_width)
        scale_h = float(target_height) / float(source_height)
        scale = max(scale_w, scale_h) if bool(crop_to_fill) else min(scale_w, scale_h)
        scaled_fl_x = fl_x * scale
        scaled_fl_y = fl_y * scale
        cx = float(target_width) / 2.0
        cy = float(target_height) / 2.0

    image = getattr(anchor_camera, "image", None)
    if image is None or target_width != source_width or target_height != source_height:
        image = torch.zeros((3, target_height, target_width), dtype=torch.float32)
    elif hasattr(image, "detach"):
        image = image.detach().cpu()

    return RtgsSimpleCamera(
        R=viewmat_t[:3, :3].numpy().T,
        T=viewmat_t[:3, 3].numpy(),
        FoVx=fovx,
        FoVy=fovy,
        image=image,
        image_name=str(image_name),
        uid=int(uid),
        timestamp=float(timestamp),
        fl_x=scaled_fl_x,
        fl_y=scaled_fl_y,
        cx=cx,
        cy=cy,
        resolution=(target_width, target_height),
        data_device=device,
    )


def render_rtgs_cr_66views(args: argparse.Namespace) -> int:
    import torch

    if str(args.interlace_mode) != "compose":
        raise ValueError("only --interlace-mode compose is currently supported")
    if int(args.views) <= 0:
        raise ValueError("--views must be positive")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive")
    if not str(args.device).startswith("cuda"):
        raise RuntimeError("RTGS + CR 66-view rendering requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot run RTGS + CR 66-view rendering")

    _install_gsplat_root(args.gsplat_root)
    runtime = prepare_official_rtgs_1view(args)
    from gaussian_renderer import render

    official_camera_cuda = runtime.camera.cuda()
    cr_anchor_camera_cuda, fov_normalization = normalize_explicit_intrinsics_fov(
        official_camera_cuda,
        enabled=bool(args.normalize_explicit_intrinsics_fov),
    )
    timestamp = float(getattr(cr_anchor_camera_cuda, "timestamp", 0.0))
    width = int(args.width)
    height = int(args.height)
    source_view_count = int(args.views)
    sample_indices = resolve_sample_view_indices(source_view_count, int(args.sample_save_views), args.sample_view_indices)
    sample_set = set(sample_indices)

    viewpoint_index, map_metadata = build_66view_viewpoint_index(
        args,
        width=width,
        height=height,
        source_view_count=source_view_count,
    )
    view_index_stats = validate_viewpoint_index(viewpoint_index, source_view_count=source_view_count)

    anchor_viewmat, _ = rtgs_camera_to_gsplat_inputs(cr_anchor_camera_cuda, device=args.device)
    geometry = materialize_rtgs_geometry(runtime.official.gaussians, timestamp=timestamp)
    c2w = torch.linalg.inv(anchor_viewmat)
    orbit_center = estimate_orbit_center(
        c2w=c2w,
        snapshot_means=geometry.means,
        orbit_center_distance=float(args.orbit_center_distance),
    )
    flat_viewmats = synthesize_flat_viewmats(
        c2w=c2w,
        orbit_center=orbit_center,
        source_view_count=source_view_count,
        view_degree=float(args.view_degree),
        orbit_direction=int(args.orbit_direction),
        device=args.device,
    )

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_cr66_output_path(
        args.model_path,
        split=args.split,
        camera_index=int(args.camera_index),
        timestamp=timestamp,
        run_label=args.run_label,
    )
    sampled_dir = output_dir / "sampled_views"
    comparisons_dir = output_dir / "sampled_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.compare_official_sampled):
        comparisons_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Rendering RTGS+CR {source_view_count} views for {runtime.official.scene_name} "
        f"({width}x{height}, sampled={sample_indices}, interlace={bool(args.write_interlaced)})",
        file=sys.stderr,
        flush=True,
    )

    viewpoint_index_t = torch.as_tensor(viewpoint_index, dtype=torch.long, device=args.device)
    interlaced = torch.zeros((3, height, width), dtype=torch.float32, device=args.device) if bool(args.write_interlaced) else None
    sampled_metrics: dict[str, Any] = {}
    sampled_adapter_info: dict[str, Any] = {}
    render_timings: dict[str, Any] = {"views_ms": {}, "official_sampled_ms": {}}
    adapter_applied_count = 0
    adapter_reason_counts: dict[str, int] = {}

    for view_id in range(source_view_count):
        synthetic_camera = synthetic_camera_from_viewmat_preserving_rtgs_contract(
            cr_anchor_camera_cuda,
            flat_viewmats[view_id],
            uid=view_id,
            image_name=f"view_{view_id:03d}",
            timestamp=timestamp,
            device=args.device,
            width=width,
            height=height,
            crop_to_fill=not bool(args.no_crop_to_fill),
        ).cuda()
        viewmat, K = rtgs_camera_to_gsplat_inputs(synthetic_camera, device=args.device)
        colors = evaluate_rtgs_colors(
            runtime.official.gaussians,
            timestamp=timestamp,
            camera_center=synthetic_camera.camera_center,
            mask=geometry.mask,
        )
        snapshot = snapshot_from_geometry(geometry, colors=colors)
        cr_snapshot, cr_K, cr_projection_adapter = adapt_snapshot_for_rtgs_compat_projection(
            snapshot,
            camera=synthetic_camera,
            viewmat=viewmat,
            K=K,
            enabled=bool(args.rtgs_compat_projection) and not bool(fov_normalization["applied"]),
        )
        if bool(cr_projection_adapter.get("applied")):
            adapter_applied_count += 1
        reason = str(cr_projection_adapter.get("reason"))
        adapter_reason_counts[reason] = adapter_reason_counts.get(reason, 0) + 1

        torch.cuda.synchronize()
        start = time.perf_counter()
        cr_image = render_rtgs_coherent(
            snapshot=cr_snapshot,
            viewmat=viewmat,
            K=cr_K,
            width=width,
            height=height,
            tile_size=int(args.tile_size),
            near_plane=float(args.near_plane),
            far_plane=float(args.far_plane),
            camera_model=str(args.camera_model),
            background=runtime.background,
            debug=bool(args.debug_cr),
        )
        torch.cuda.synchronize()
        render_timings["views_ms"][f"view_{view_id:03d}"] = (time.perf_counter() - start) * 1000.0

        if interlaced is not None:
            accumulate_interlaced_view(interlaced, cr_image, viewpoint_index_t, view_id=view_id)

        if view_id in sample_set:
            save_tensor_image(sampled_dir / f"view_{view_id:03d}_cr.png", cr_image)
            sampled_adapter_info[f"view_{view_id:03d}"] = cr_projection_adapter

            if bool(args.compare_official_sampled):
                torch.cuda.synchronize()
                official_start = time.perf_counter()
                with torch.no_grad():
                    official_image = (
                        render(synthetic_camera, runtime.official.gaussians, runtime.pipe, runtime.background)["render"]
                        .detach()
                        .clamp(0.0, 1.0)
                        .contiguous()
                    )
                torch.cuda.synchronize()
                render_timings["official_sampled_ms"][f"view_{view_id:03d}"] = (time.perf_counter() - official_start) * 1000.0
                metrics = compute_pair_metrics(cr_image, official_image, ssim_metric=runtime.ssim_metric)
                sampled_metrics[f"view_{view_id:03d}"] = metrics
                save_tensor_image(comparisons_dir / f"view_{view_id:03d}_official.png", official_image)
                save_comparison_image(
                    comparisons_dir / f"view_{view_id:03d}_comparison.png",
                    official=official_image,
                    coherent=cr_image,
                )
                (comparisons_dir / f"view_{view_id:03d}_metrics.json").write_text(
                    _json_dumps(metrics),
                    encoding="utf-8",
                )

    if interlaced is not None:
        save_tensor_image(output_dir / "rtgs_cr_lkg_interlaced.png", interlaced)

    metrics_summary = summarize_sampled_metrics(sampled_metrics)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "rtgs_cr_66views",
        "interlace_mode": str(args.interlace_mode),
        "scene": runtime.official.scene_name,
        "dataset_kind": runtime.official.scene_paths.dataset_kind,
        "dataset_path": str(runtime.official.scene_paths.dataset_path),
        "source_path": str(runtime.official.source_path),
        "config_path": str(runtime.official.scene_paths.config_path),
        "model_path": str(Path(args.model_path).expanduser()),
        "checkpoint_path": str(runtime.official.checkpoint_path),
        "iteration": int(runtime.official.iteration),
        "split": str(args.split),
        "camera_index": int(args.camera_index),
        "n3dv_frame_index": int(args.n3dv_frame_index),
        "timestamp": timestamp,
        "width": width,
        "height": height,
        "views": source_view_count,
        "sampled_view_indices": sample_indices,
        "cluster_size": int(args.cluster_size),
        "cr_remapping": str(args.cr_remapping),
        "write_interlaced": bool(args.write_interlaced),
        "compare_official_sampled": bool(args.compare_official_sampled),
        "view_synthesis": {
            "view_degree": float(args.view_degree),
            "orbit_direction": int(args.orbit_direction),
            "orbit_center_distance": float(args.orbit_center_distance),
        },
        "map_metadata": map_metadata,
        "viewpoint_index_stats": view_index_stats,
        "camera_source": str(runtime.camera_source),
        "anchor_camera_contract": camera_render_contract_values(cr_anchor_camera_cuda),
        "camera_fov_normalization": fov_normalization,
        "cr_projection_adapter_summary": {
            "applied_count": int(adapter_applied_count),
            "total_views": int(source_view_count),
            "reason_counts": adapter_reason_counts,
            "sampled": sampled_adapter_info,
        },
        "gaussians_total": int(runtime.official.gaussians.get_xyz.shape[0]),
        "gaussians_after_temporal_mask": int(geometry.means.shape[0]),
        "active_sh_degree": int(runtime.official.gaussians.active_sh_degree),
        "active_sh_degree_t": int(getattr(runtime.official.gaussians, "active_sh_degree_t", 0)),
        "rtgs_git_commit": runtime.official.rtgs_git_commit or None,
        "rtgs_code_status": dict(runtime.official.rtgs_code_status),
        "render_ms": render_timings,
        "metrics": {
            "sampled_cr_vs_official": sampled_metrics,
            "sampled_summary": metrics_summary,
        },
        "args": dict(vars(args)),
    }
    (output_dir / "metrics.json").write_text(
        _json_dumps({"sampled_cr_vs_official": sampled_metrics, "sampled_summary": metrics_summary}),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(_json_dumps(manifest), encoding="utf-8")
    print(
        f"Wrote {output_dir} (sampled_mean_psnr={format_optional_metric(metrics_summary.get('psnr_mean'))})",
        file=sys.stderr,
        flush=True,
    )
    return 0


def build_66view_viewpoint_index(
    args: argparse.Namespace,
    *,
    width: int,
    height: int,
    source_view_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if args.map_mode == "file":
        viewpoint_index, file_view_count, metadata = load_viewpoint_index_file(args.viewpoint_index_path, width, height)
        if file_view_count is not None and int(file_view_count) != int(source_view_count):
            raise ValueError(f"viewpoint index file has {file_view_count} views, expected {source_view_count}")
        metadata = dict(metadata)
        metadata["mode"] = "file"
        return viewpoint_index, metadata
    if args.map_mode == "linear":
        viewpoint_index = build_linear_viewpoint_index(
            width,
            height,
            int(source_view_count),
            quantize=str(args.coherent_quantize),
        )
        return viewpoint_index, {"mode": "linear", "quantize": str(args.coherent_quantize)}
    raise ValueError(f"unsupported map mode: {args.map_mode!r}")


def estimate_orbit_center(*, c2w: Any, snapshot_means: Any, orbit_center_distance: float):
    import torch

    camera_position = c2w[:3, 3]
    if float(orbit_center_distance) > 0.0:
        distance = float(orbit_center_distance)
    elif int(snapshot_means.shape[0]) > 0:
        scene_center = snapshot_means.mean(dim=0)
        distance = max(float(torch.linalg.norm(scene_center - camera_position).item()), 1e-4)
    else:
        distance = 1.0
    return camera_position + c2w[:3, 2] * distance


def synthesize_flat_viewmats(
    *,
    c2w: Any,
    orbit_center: Any,
    source_view_count: int,
    view_degree: float,
    orbit_direction: int,
    device: str,
):
    view_labels = np.arange(int(source_view_count), dtype=np.int32)
    synthesizer = OrbitViewSynthesizer(
        view_labels=view_labels,
        source_view_count=int(source_view_count),
        cluster_size=1,
        view_degree=float(view_degree),
        orbit_direction=int(orbit_direction),
        device=device,
    )
    return synthesizer(c2w, orbit_center).reshape(-1, 4, 4)[: int(source_view_count)].contiguous()


def accumulate_interlaced_view(interlaced: Any, image: Any, viewpoint_index_t: Any, *, view_id: int) -> None:
    if tuple(image.shape) != tuple(interlaced.shape):
        raise ValueError(f"image shape {tuple(image.shape)} does not match interlaced shape {tuple(interlaced.shape)}")
    for channel in range(3):
        mask = viewpoint_index_t[:, :, channel] == int(view_id)
        interlaced[channel][mask] = image[channel][mask]


def summarize_sampled_metrics(sampled_metrics: Mapping[str, Mapping[str, float]]) -> dict[str, float | int | None]:
    summary: dict[str, float | int | None] = {"count": int(len(sampled_metrics))}
    for key in ("psnr", "mae", "mse", "ssim"):
        values = [float(metrics[key]) for metrics in sampled_metrics.values() if key in metrics and math.isfinite(float(metrics[key]))]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_min"] = float(np.min(values))
            summary[f"{key}_max"] = float(np.max(values))
        else:
            summary[f"{key}_mean"] = None
            summary[f"{key}_min"] = None
            summary[f"{key}_max"] = None
    return summary


def format_optional_metric(value: Any, *, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{int(precision)}f}"


def save_tensor_image(path: Path, tensor: Any) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_hwc_uint8(tensor), mode="RGB").save(path)


def save_comparison_image(path: Path, *, official: Any, coherent: Any) -> None:
    from PIL import Image

    official_np = tensor_to_hwc_uint8(official)
    coherent_np = tensor_to_hwc_uint8(coherent)
    diff_np = _diff_image(coherent_np, official_np)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([official_np, coherent_np, diff_np], axis=1), mode="RGB").save(path)


def _json_dumps(value: Any) -> str:
    return __import__("json").dumps(_json_ready(value), indent=2, sort_keys=True) + "\n"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if math.isnan(float(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return render_rtgs_cr_66views(args)
