from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lkg_experiment.coherent_default.coherent_gsplat_bridge import build_cr_lookup_arrays, lookup_arrays_to_torch
from lkg_experiment.coherent_default.coherent_raster import build_linear_viewpoint_index
from lkg_experiment.coherent_default.coherent_raster_experiment import (
    OrbitViewSynthesizer,
    load_viewpoint_index_file,
    tensor_to_hwc_uint8,
)
from lkg_experiment.rtgs_coherent.cli import (
    _background_tensor,
    _install_gsplat_root,
    _make_ssim_metric,
    _pipeline_namespace_for_model,
    compute_pair_metrics,
    default_output_path,
    evaluate_rtgs_colors,
    install_rtgs_code_root,
    load_rtgs_camera,
    load_rtgs_checkpoint,
    materialize_rtgs_geometry,
    render_rtgs_coherent,
    render_rtgs_original,
    rtgs_camera_from_gsplat_viewmat,
    rtgs_camera_to_gsplat_inputs,
    snapshot_from_geometry,
)


def default_views66_output_path(
    model_path: Path | str,
    *,
    split: str,
    camera_index: int,
    timestamp: float,
    views: int,
) -> Path:
    base = default_output_path(model_path, split=split, camera_index=camera_index, timestamp=timestamp)
    return base.with_name(f"{base.name}_views{int(views)}")


def sample_evenly_spaced_view_indices(view_count: int, max_views: int) -> list[int]:
    if int(view_count) <= 0:
        raise ValueError("view_count must be positive")
    if int(max_views) <= 0:
        return []
    if int(max_views) >= int(view_count):
        return list(range(int(view_count)))
    return np.linspace(0, int(view_count) - 1, int(max_views)).astype(np.int64).tolist()


def scale_intrinsics_to_resolution(
    K: Any,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    crop_to_fill: bool = True,
):
    scale_w = float(target_width) / float(source_width)
    scale_h = float(target_height) / float(source_height)
    scale = max(scale_w, scale_h) if bool(crop_to_fill) else min(scale_w, scale_h)
    scaled = K.new_zeros((3, 3))
    scaled[0, 0] = K[0, 0] * scale
    scaled[1, 1] = K[1, 1] * scale
    scaled[0, 2] = float(target_width) / 2.0
    scaled[1, 2] = float(target_height) / 2.0
    scaled[2, 2] = 1.0
    return scaled.contiguous()


def render_rtgs_66_views(args: Any) -> int:
    import torch

    if int(args.views) <= 0:
        raise ValueError("--views must be positive")
    if int(args.cluster_size) <= 0:
        raise ValueError("--cluster-size must be positive")

    install_rtgs_code_root(args.rtgs_code_root)
    _install_gsplat_root(args.gsplat_root)
    checkpoint = load_rtgs_checkpoint(
        model_path=args.model_path,
        checkpoint=args.checkpoint,
        rtgs_code_root=args.rtgs_code_root,
        dataset_root=args.dataset_root,
        n3dv_root=args.n3dv_root,
        config_path=args.config,
        device=args.device,
        checkpoint_load_device=args.checkpoint_load_device,
    )
    _, anchor_camera = load_rtgs_camera(
        checkpoint=checkpoint,
        split=args.split,
        camera_index=args.camera_index,
        device=args.device,
        n3dv_frame_index=args.n3dv_frame_index,
    )
    anchor_viewmat, anchor_K = rtgs_camera_to_gsplat_inputs(anchor_camera, device=args.device)
    width = int(args.width)
    height = int(args.height)
    K = scale_intrinsics_to_resolution(
        anchor_K,
        source_width=int(anchor_camera.image_width),
        source_height=int(anchor_camera.image_height),
        target_width=width,
        target_height=height,
        crop_to_fill=not bool(args.no_crop_to_fill),
    )
    timestamp = float(anchor_camera.timestamp)
    background = _background_tensor(args.background, checkpoint.cfg_args, args.device)

    viewpoint_index, source_view_count, map_metadata = build_views66_viewpoint_index(
        args,
        width=width,
        height=height,
    )
    view_labels = np.arange(source_view_count, dtype=np.int32)

    geometry = materialize_rtgs_geometry(checkpoint.model, timestamp=timestamp)
    c2w = torch.linalg.inv(anchor_viewmat)
    orbit_center = estimate_orbit_center(
        c2w=c2w,
        snapshot_means=geometry.means,
        orbit_center_distance=float(args.orbit_center_distance),
    )
    flat_viewmats = synthesize_flat_viewmats(
        c2w=c2w,
        orbit_center=orbit_center,
        view_labels=view_labels,
        source_view_count=source_view_count,
        view_degree=float(args.view_degree),
        orbit_direction=int(args.orbit_direction),
        device=args.device,
    )

    output_root = Path(args.output_dir).expanduser() if args.output_dir else default_views66_output_path(
        args.model_path,
        split=args.split,
        camera_index=args.camera_index,
        timestamp=timestamp,
        views=source_view_count,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Rendering RTGS coherent {source_view_count} views from {checkpoint.scene_paths.scene_name} "
        f"({width}x{height}, cluster={int(args.cluster_size)})",
        file=sys.stderr,
        flush=True,
    )

    interlaced_ms = None
    if bool(args.write_interlaced):
        torch.cuda.synchronize()
        start = time.perf_counter()
        interlaced = render_rtgs_coherent_interlaced(
            pc=checkpoint.model,
            geometry=geometry,
            timestamp=timestamp,
            adjacent_viewmats=synthesize_grouped_viewmats(
                c2w=c2w,
                orbit_center=orbit_center,
                view_labels=view_labels,
                source_view_count=source_view_count,
                cluster_size=int(args.cluster_size),
                view_degree=float(args.view_degree),
                orbit_direction=int(args.orbit_direction),
                device=args.device,
            ),
            K=K,
            viewpoint_index=viewpoint_index,
            width=width,
            height=height,
            tile_size=int(args.tile_size),
            near_plane=float(args.near_plane),
            far_plane=float(args.far_plane),
            camera_model=str(args.camera_model),
            background=background,
            debug=bool(args.debug_cr),
        )
        torch.cuda.synchronize()
        interlaced_ms = (time.perf_counter() - start) * 1000.0
        save_tensor_image(output_root / "rtgs_coherent_lkg.png", interlaced)

    compare_indices = sample_evenly_spaced_view_indices(source_view_count, int(args.compare_original_views))
    coherent_sampled_views = {}
    per_view_ms = None
    if bool(args.write_per_view) or compare_indices:
        torch.cuda.synchronize()
        start = time.perf_counter()
        coherent_sampled_views = render_rtgs_coherent_views(
            pc=checkpoint.model,
            geometry=geometry,
            timestamp=timestamp,
            flat_viewmats=flat_viewmats,
            K=K,
            width=width,
            height=height,
            tile_size=int(args.tile_size),
            near_plane=float(args.near_plane),
            far_plane=float(args.far_plane),
            camera_model=str(args.camera_model),
            background=background,
            debug=bool(args.debug_cr),
            output_root=output_root / "views" if bool(args.write_per_view) else None,
            keep_indices=set(compare_indices),
        )
        torch.cuda.synchronize()
        per_view_ms = (time.perf_counter() - start) * 1000.0

    comparison_metrics = {}
    comparison_ms = None
    if compare_indices:
        if not coherent_sampled_views:
            raise RuntimeError("internal error: sampled comparison requires coherent per-view renders")
        torch.cuda.synchronize()
        start = time.perf_counter()
        comparison_metrics = render_sampled_original_comparisons(
            output_root=output_root / "comparisons",
            indices=compare_indices,
            flat_viewmats=flat_viewmats,
            coherent_views=coherent_sampled_views,
            anchor_camera=anchor_camera,
            pc=checkpoint.model,
            pipe=_pipeline_namespace_for_model(checkpoint.config.get("PipelineParams", {}), checkpoint.model),
            background=background,
            timestamp=timestamp,
            width=width,
            height=height,
            crop_to_fill=not bool(args.no_crop_to_fill),
            device=args.device,
            ssim_metric=None if args.no_ssim else _make_ssim_metric(args.device),
        )
        torch.cuda.synchronize()
        comparison_ms = (time.perf_counter() - start) * 1000.0

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scene": checkpoint.scene_paths.scene_name,
        "dataset_kind": checkpoint.scene_paths.dataset_kind,
        "dataset_path": str(checkpoint.scene_paths.dataset_path),
        "config_path": str(checkpoint.scene_paths.config_path),
        "model_path": str(Path(args.model_path).expanduser()),
        "checkpoint_path": str(checkpoint.checkpoint_path),
        "iteration": checkpoint.iteration,
        "split": args.split,
        "camera_index": args.camera_index,
        "timestamp": timestamp,
        "width": width,
        "height": height,
        "views": source_view_count,
        "cluster_size": int(args.cluster_size),
        "gaussians_total": int(checkpoint.model.get_xyz.shape[0]),
        "gaussians_snapshot": int(geometry.means.shape[0]),
        "active_sh_degree": int(checkpoint.model.active_sh_degree),
        "active_sh_degree_t": int(checkpoint.model.active_sh_degree_t),
        "map_metadata": map_metadata,
        "write_interlaced": bool(args.write_interlaced),
        "write_per_view": bool(args.write_per_view),
        "compare_original_views": compare_indices,
        "render_ms": {
            "coherent_interlaced": interlaced_ms,
            "coherent_per_view": per_view_ms,
            "sampled_original_comparison": comparison_ms,
        },
        "metrics": {"cr_vs_rtgs_sampled": comparison_metrics},
        "args": vars(args),
    }
    (output_root / "manifest.json").write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "metrics.json").write_text(
        json.dumps(_json_ready({"cr_vs_rtgs_sampled": comparison_metrics}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_root}", file=sys.stderr, flush=True)
    return 0


def build_views66_viewpoint_index(args: Any, *, width: int, height: int) -> tuple[np.ndarray, int, dict[str, Any]]:
    if args.map_mode == "file":
        viewpoint_index, file_view_count, metadata = load_viewpoint_index_file(args.viewpoint_index_path, width, height)
        source_view_count = int(file_view_count or args.views)
        metadata = dict(metadata)
        metadata["mode"] = "file"
        return viewpoint_index, source_view_count, metadata
    if args.map_mode == "linear":
        viewpoint_index = build_linear_viewpoint_index(
            width,
            height,
            int(args.views),
            quantize=str(args.coherent_quantize),
        )
        return viewpoint_index, int(args.views), {"mode": "linear", "quantize": str(args.coherent_quantize)}
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
    view_labels: np.ndarray,
    source_view_count: int,
    view_degree: float,
    orbit_direction: int,
    device: str,
):
    return synthesize_grouped_viewmats(
        c2w=c2w,
        orbit_center=orbit_center,
        view_labels=view_labels,
        source_view_count=source_view_count,
        cluster_size=1,
        view_degree=view_degree,
        orbit_direction=orbit_direction,
        device=device,
    ).reshape(-1, 4, 4)[: int(view_labels.size)].contiguous()


def synthesize_grouped_viewmats(
    *,
    c2w: Any,
    orbit_center: Any,
    view_labels: np.ndarray,
    source_view_count: int,
    cluster_size: int,
    view_degree: float,
    orbit_direction: int,
    device: str,
):
    synthesizer = OrbitViewSynthesizer(
        view_labels=view_labels,
        source_view_count=int(source_view_count),
        cluster_size=int(cluster_size),
        view_degree=float(view_degree),
        orbit_direction=int(orbit_direction),
        device=device,
    )
    return synthesizer(c2w, orbit_center).contiguous()


def render_rtgs_coherent_interlaced(
    *,
    pc: Any,
    geometry: Any,
    timestamp: float,
    adjacent_viewmats: Any,
    K: Any,
    viewpoint_index: np.ndarray,
    width: int,
    height: int,
    tile_size: int,
    near_plane: float,
    far_plane: float,
    camera_model: str,
    background: Any,
    debug: bool,
):
    import torch
    from coherent_raster.utils.utils_coherent_raster import unpad, unpatchify_image_shape_matrix
    from gsplat.rendering_coherent_raster import rasterization_CR

    ref_idx = int(adjacent_viewmats.shape[1]) // 2
    reference_viewmats = adjacent_viewmats[:, ref_idx]
    reference_centers = torch.linalg.inv(reference_viewmats)[:, :3, 3]
    colors = [
        evaluate_rtgs_colors(
            pc,
            timestamp=float(timestamp),
            camera_center=center,
            mask=geometry.mask,
        )
        for center in reference_centers
    ]
    snapshot = snapshot_from_geometry(geometry, colors=torch.stack(colors, dim=0).contiguous())
    lookup = build_cr_lookup_arrays(viewpoint_index, tile_size=int(tile_size), use_remapping=True)
    view_idx_matrix, subpixel_coord_matrix = lookup_arrays_to_torch(lookup, device=str(K.device))
    backgrounds = None if background is None else background.contiguous()

    with torch.no_grad():
        colors, _, _ = rasterization_CR(
            means=snapshot.means,
            quats=None,
            scales=None,
            opacities=snapshot.opacities,
            colors=snapshot.colors,
            adjacent_viewmats=adjacent_viewmats,
            Ks=K.unsqueeze(0).contiguous(),
            view_idx_matrix=view_idx_matrix,
            subpixel_coord_matrix=subpixel_coord_matrix,
            width=int(width),
            height=int(height),
            sh_degree=None,
            near_plane=float(near_plane),
            far_plane=float(far_plane),
            backgrounds=backgrounds,
            camera_model=camera_model,
            tile_size=int(tile_size),
            is_debug=debug,
            covars=snapshot.covars,
        )
        image = unpatchify_image_shape_matrix(colors)
        image = unpad(image, int(height), int(width))
        return image.clamp(0.0, 1.0).contiguous()


def render_rtgs_coherent_views(
    *,
    pc: Any,
    geometry: Any,
    timestamp: float,
    flat_viewmats: Any,
    K: Any,
    width: int,
    height: int,
    tile_size: int,
    near_plane: float,
    far_plane: float,
    camera_model: str,
    background: Any,
    debug: bool,
    output_root: Path | None,
    keep_indices: set[int],
):
    import torch

    kept = {}
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        render_indices = range(int(flat_viewmats.shape[0]))
    else:
        render_indices = sorted(keep_indices)
    for index in render_indices:
        viewmat = flat_viewmats[index]
        center = torch.linalg.inv(viewmat)[:3, 3]
        colors = evaluate_rtgs_colors(
            pc,
            timestamp=float(timestamp),
            camera_center=center,
            mask=geometry.mask,
        )
        snapshot = snapshot_from_geometry(geometry, colors=colors)
        image = render_rtgs_coherent(
            snapshot=snapshot,
            viewmat=viewmat,
            K=K,
            width=width,
            height=height,
            tile_size=tile_size,
            near_plane=near_plane,
            far_plane=far_plane,
            camera_model=camera_model,
            background=background,
            debug=debug,
        )
        if output_root is not None:
            save_tensor_image(output_root / f"view_{index:03d}.png", image)
        if index in keep_indices:
            kept[index] = image.detach().clone()
    return kept


def render_sampled_original_comparisons(
    *,
    output_root: Path,
    indices: list[int],
    flat_viewmats: Any,
    coherent_views: Any,
    anchor_camera: Any,
    pc: Any,
    pipe: Any,
    background: Any,
    timestamp: float,
    width: int,
    height: int,
    crop_to_fill: bool,
    device: str,
    ssim_metric: Any,
) -> dict[str, Mapping[str, float]]:
    output_root.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for index in indices:
        synthetic_camera = rtgs_camera_from_gsplat_viewmat(
            anchor_camera,
            flat_viewmats[int(index)],
            uid=int(index),
            image_name=f"view_{int(index):03d}",
            timestamp=float(timestamp),
            device=device,
            width=width,
            height=height,
            crop_to_fill=crop_to_fill,
        )
        original = render_rtgs_original(synthetic_camera, pc, pipe, background)
        coherent = coherent_views[int(index)]
        metrics[f"view_{int(index):03d}"] = compute_pair_metrics(coherent, original, ssim_metric=ssim_metric)
        save_comparison_image(output_root / f"view_{int(index):03d}_comparison.png", original=original, coherent=coherent)
    return metrics


def save_tensor_image(path: Path, tensor: Any) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_hwc_uint8(tensor), mode="RGB").save(path)


def save_comparison_image(path: Path, *, original: Any, coherent: Any) -> None:
    from PIL import Image

    original_np = tensor_to_hwc_uint8(original)
    coherent_np = tensor_to_hwc_uint8(coherent)
    diff_np = np.clip(np.abs(coherent_np.astype(np.int16) - original_np.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([original_np, coherent_np, diff_np], axis=1), mode="RGB").save(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return str(value)
    return value
