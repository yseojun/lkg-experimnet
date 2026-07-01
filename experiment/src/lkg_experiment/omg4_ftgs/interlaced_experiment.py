from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lkg_experiment.omg4_ftgs import cli as single_cli
from lkg_experiment.omg4_ftgs.camera import load_pose_camera, load_test_frames
from lkg_experiment.omg4_ftgs.model import load_dynamic_gaussians
from lkg_experiment.omg4_ftgs.render import (
    build_interlaced_viewpoint_index,
    estimate_orbit_center,
    install_gsplat_root,
    prepare_cr_lookup_tensors,
    render_splats_interlaced_coherent,
    synthesize_interlaced_viewmats,
)
from lkg_experiment.coherent_default.coherent_raster_experiment import (
    COMMON_EXPERIMENT_COLUMNS,
    DEFAULT_CLUSTERS,
    OMG4_SPECIFIC_EXPERIMENT_COLUMNS,
    ExperimentVariant,
    MetricStats,
    build_experiment_variants,
    compute_interlaced_metric_stats,
    parse_cluster_values,
)


DEFAULT_OUTPUT_ROOT = Path("/data/ysj/result/coherent-raster/generated/omg4_ftgs_interlaced_experiments")
DEFAULT_DATASET_ALIASES = {
    "flame_salmon": "flame_salmon_1",
}
ENGINE_NAME = "omg4_ftgs_interlaced"
SUMMARY_COLUMNS = list(dict.fromkeys(COMMON_EXPERIMENT_COLUMNS + OMG4_SPECIFIC_EXPERIMENT_COLUMNS))


@dataclass(frozen=True)
class SceneJob:
    scene: str
    dataset_scene: str
    checkpoint_path: Path
    data_path: Path


@dataclass(frozen=True)
class PreparedInterlacedScene:
    viewpoint_index: np.ndarray
    map_metadata: Mapping[str, Any]
    view_idx_matrix: Any
    subpixel_coord_matrix: Any
    viewpoint_index_ms: float
    lookup_cpu_ms: float
    lookup_h2d_ms: float
    source_c2w: Any
    adjacent_viewmats: Any
    K: Any
    orbit_center: Any
    setup_materialize_ms: float
    setup_view_setup_ms: float
    setup_gaussians: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure OMG4-FTGS 1440x2560 CoherentRaster interlaced panel generation FPS")
    parser.add_argument("--weights-root", default=str(single_cli.DEFAULT_WEIGHTS_ROOT))
    parser.add_argument("--weight-group", default="ours_L_weight")
    parser.add_argument("--data-root", default="/data/ysj/dataset/N3DV")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-group", default=None)
    parser.add_argument("--scenes", default=None, help="Comma or whitespace separated scene list; default discovers all .xz files")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--panel-width", type=int, default=1440)
    parser.add_argument("--panel-height", type=int, default=2560)
    parser.add_argument("--views", type=int, default=66)
    parser.add_argument("--cluster-size", type=int, default=8)
    parser.add_argument("--clusters", default=",".join(str(cluster) for cluster in DEFAULT_CLUSTERS))
    parser.add_argument("--ablation-cluster", default=8, type=int)
    parser.add_argument("--no-without-remap", action="store_true")
    parser.add_argument("--no-without-reuse", action="store_true")
    parser.add_argument("--view-degree", type=float, default=53.0)
    parser.add_argument("--orbit-direction", type=int, default=-1)
    parser.add_argument("--orbit-center-distance", type=float, default=0.0)
    parser.add_argument("--map-mode", choices=("file", "linear"), default="file")
    parser.add_argument("--viewpoint-index-path", default=str(single_cli.DEFAULT_VIEWPOINT_INDEX_PATH))
    parser.add_argument("--coherent-quantize", choices=("floor", "nearest"), default="floor")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=100.0)
    parser.add_argument("--camera-model", choices=("pinhole", "ortho", "fisheye"), default="pinhole")
    parser.add_argument("--debug-cr", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gsplat-root", default=str(single_cli.DEFAULT_GSPLAT_ROOT))
    parser.add_argument("--allow-missing-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--require-lpips", action="store_true")
    parser.add_argument("--no-save-image", action="store_true")
    return parser


def discover_scene_jobs(
    *,
    weights_root: Path | str,
    weight_group: str,
    data_root: Path | str,
    scenes: str | list[str] | None,
    require_data: bool,
) -> list[SceneJob]:
    weights_dir = Path(weights_root).expanduser() / str(weight_group)
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"OMG4-FTGS weight group not found: {weights_dir}")

    scene_names = _parse_scenes(scenes)
    if scene_names is None:
        scene_names = sorted(path.stem for path in weights_dir.glob("*.xz"))
    jobs: list[SceneJob] = []
    for scene in scene_names:
        checkpoint_path = weights_dir / f"{scene}.xz"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"OMG4-FTGS checkpoint not found for scene {scene}: {checkpoint_path}")
        dataset_scene, data_path = resolve_dataset_scene(scene, data_root=Path(data_root).expanduser(), require_data=require_data)
        jobs.append(
            SceneJob(
                scene=scene,
                dataset_scene=dataset_scene,
                checkpoint_path=checkpoint_path,
                data_path=data_path,
            )
        )
    return jobs


def resolve_dataset_scene(scene: str, *, data_root: Path, require_data: bool) -> tuple[str, Path]:
    candidates = [str(scene)]
    alias = DEFAULT_DATASET_ALIASES.get(str(scene))
    if alias and alias not in candidates:
        candidates.append(alias)
    suffixed = f"{scene}_1"
    if suffixed not in candidates:
        candidates.append(suffixed)

    for candidate in candidates:
        path = data_root / candidate
        if path.is_dir():
            return candidate, path
    dataset_scene = candidates[0]
    data_path = data_root / dataset_scene
    if require_data:
        raise FileNotFoundError(f"N3DV dataset directory not found for scene {scene}; tried: {', '.join(candidates)}")
    return dataset_scene, data_path


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    run_group = _resolve_run_group(args)
    args = argparse.Namespace(**vars(args))
    args.run_group = run_group
    experiment_root = Path(args.output_root).expanduser() / run_group
    install_gsplat_root(args.gsplat_root)
    if not bool(args.dry_run):
        preflight_cuda_runtime(args)

    jobs = discover_scene_jobs(
        weights_root=args.weights_root,
        weight_group=str(args.weight_group),
        data_root=args.data_root,
        scenes=args.scenes,
        require_data=not bool(args.allow_missing_data),
    )
    variants = resolve_experiment_variants(args)
    rows: list[dict[str, Any]] = []
    failures = 0

    for job in jobs:
        metric_reference_image = None
        if not bool(args.dry_run) and not bool(args.skip_metrics):
            metric_reference = metric_reference_variant()
            metric_reference_image = render_metric_reference_interlaced(job, args)
            if not bool(args.no_save_image):
                metric_reference_dir = build_scene_output_dir(args, job, variant=metric_reference)
                metric_reference_dir.mkdir(parents=True, exist_ok=True)
                single_cli._save_tensor_image(metric_reference_dir / "metric_reference.png", metric_reference_image)
        for variant in variants:
            output_dir = build_scene_output_dir(args, job, variant=variant)
            output_dir.mkdir(parents=True, exist_ok=True)
            if bool(args.dry_run):
                rows.append(
                    scene_status_row(
                        status="planned",
                        job=job,
                        output_dir=output_dir,
                        args=args,
                        variant=variant,
                        message="dry run",
                    )
                )
                write_summary(rows, experiment_root)
                continue
            try:
                rows.append(run_scene(job, args, output_dir=output_dir, variant=variant, metric_reference_image=metric_reference_image))
            except Exception as exc:
                failures += 1
                (output_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                rows.append(
                    scene_status_row(
                        status="failed",
                        job=job,
                        output_dir=output_dir,
                        args=args,
                        variant=variant,
                        message=str(exc),
                    )
                )
                write_summary(rows, experiment_root)
                if bool(args.stop_on_failure):
                    write_manifest(experiment_root, args=args, run_group=run_group, scene_count=len(jobs), failure_count=failures)
                    return 1
                continue
            write_summary(rows, experiment_root)

    write_summary(rows, experiment_root)
    write_manifest(experiment_root, args=args, run_group=run_group, scene_count=len(jobs), failure_count=failures)
    return 1 if failures else 0


def validate_args(args: argparse.Namespace) -> None:
    if int(args.panel_width) <= 0 or int(args.panel_height) <= 0:
        raise ValueError("--panel-width and --panel-height must be positive")
    if int(args.views) <= 0:
        raise ValueError("--views must be positive")
    if int(args.cluster_size) <= 0:
        raise ValueError("--cluster-size must be positive")
    if int(args.ablation_cluster) <= 0:
        raise ValueError("--ablation-cluster must be positive")
    if int(args.tile_size) <= 0:
        raise ValueError("--tile-size must be positive")
    if int(args.warmup_iters) < 0 or int(args.measure_iters) <= 0:
        raise ValueError("--warmup-iters must be non-negative and --measure-iters must be positive")
    parse_cluster_values(args.clusters)


def resolve_experiment_variants(args: argparse.Namespace) -> list[ExperimentVariant]:
    return build_experiment_variants(
        parse_cluster_values(args.clusters),
        ablation_cluster=int(args.ablation_cluster),
        include_without_remap=not bool(args.no_without_remap),
        include_without_reuse=not bool(args.no_without_reuse),
    )


def legacy_variant_from_args(args: argparse.Namespace) -> ExperimentVariant:
    cluster_size = int(args.cluster_size)
    return ExperimentVariant(
        name=f"cluster_{cluster_size}",
        cluster_size=cluster_size,
        use_remapping=True,
        reuse_enabled=True,
        group="cluster_sweep",
    )


def resolve_runtime_variant(args: argparse.Namespace, variant: ExperimentVariant | None = None) -> ExperimentVariant:
    return variant if variant is not None else legacy_variant_from_args(args)


def metric_reference_variant() -> ExperimentVariant:
    return ExperimentVariant(
        name="cluster_1",
        cluster_size=1,
        use_remapping=True,
        reuse_enabled=True,
        group="metric_reference",
    )


def effective_cluster_size(variant: Any) -> int:
    cluster_size = int(getattr(variant, "cluster_size"))
    if not bool(getattr(variant, "reuse_enabled", True)):
        return 1
    return cluster_size


def color_eval_view_count(source_views: int, variant: Any) -> int:
    source_views = int(source_views)
    if source_views <= 0:
        return 0
    if not bool(getattr(variant, "reuse_enabled", True)):
        return source_views
    return int(math.ceil(source_views / float(effective_cluster_size(variant))))


def preflight_cuda_runtime(args: argparse.Namespace) -> None:
    if not str(args.device).startswith("cuda"):
        return
    try:
        torch = _import_torch()
    except Exception as exc:
        raise RuntimeError("PyTorch with CUDA support is required for OMG4-FTGS interlaced experiments") from exc
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA is not available; run on a GPU node or use an environment with CUDA visible")

    device_count = int(torch.cuda.device_count())
    capabilities = [tuple(int(item) for item in torch.cuda.get_device_capability(index)) for index in range(device_count)]
    try:
        _import_tinycudann()
    except Exception as exc:
        hint = _cuda_visible_devices_hint(capabilities)
        raise RuntimeError(
            "tinycudann import failed before rendering. "
            f"Visible CUDA compute capabilities are {capabilities}. "
            f"{hint} Original error: {exc}"
        ) from exc


def run_scene(
    job: SceneJob,
    args: argparse.Namespace,
    *,
    output_dir: Path,
    variant: ExperimentVariant | None = None,
    metric_reference_image: Any | None = None,
) -> dict[str, Any]:
    variant = resolve_runtime_variant(args, variant)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_test_frames(job.data_path, camera_index=int(args.camera_index))
    camera_set = load_pose_camera(job.data_path, camera_index=int(args.camera_index), resolution=float(args.resolution))
    dynamic_model = load_dynamic_gaussians(job.checkpoint_path, device=str(args.device))
    frame = single_cli._select_frame(frames, int(args.frame_index))
    prepared = prepare_interlaced_scene(args=args, camera_set=camera_set, dynamic_model=dynamic_model, frame=frame, variant=variant)

    _reset_cuda_peak_memory()
    samples: list[dict[str, Any]] = []
    last_image = None
    total_iterations = int(args.warmup_iters) + int(args.measure_iters)
    for iteration in range(total_iterations):
        phase = "warmup" if iteration < int(args.warmup_iters) else "measure"
        print(
            f"OMG4-FTGS interlaced {job.scene}/{variant.name} timing {iteration + 1}/{total_iterations} {phase}",
            file=sys.stderr,
            flush=True,
        )
        last_image, timing = render_interlaced_once(
            args=args,
            dynamic_model=dynamic_model,
            prepared=prepared,
            timestamp=float(frame.timestamp),
            variant=variant,
        )
        if iteration >= int(args.warmup_iters):
            samples.append(timing)

    averaged = average_measurements(samples)
    lookup_ms = float(prepared.lookup_cpu_ms + prepared.lookup_h2d_ms)
    if "frame_ms_excluding_lookup" not in averaged:
        averaged["frame_ms_excluding_lookup"] = float(averaged["frame_ms_with_lkg_interlace"])
    if "frame_ms_including_lookup" not in averaged:
        averaged["frame_ms_including_lookup"] = float(averaged["frame_ms_with_lkg_interlace"] + prepared.viewpoint_index_ms + lookup_ms)
    averaged["fps_including_lookup"] = fps_from_ms(averaged["frame_ms_including_lookup"])
    _ensure_end_to_end_timing(averaged)
    averaged["viewpoint_index_ms"] = float(prepared.viewpoint_index_ms)
    averaged["lookup_cpu_ms"] = float(prepared.lookup_cpu_ms)
    averaged["lookup_h2d_ms"] = float(prepared.lookup_h2d_ms)
    averaged["setup_materialize_ms"] = float(prepared.setup_materialize_ms)
    averaged["setup_view_setup_ms"] = float(prepared.setup_view_setup_ms)
    averaged["peak_vram_gb"] = _peak_vram_gb()

    image_path = output_dir / "omg4_ftgs_lkg_interlaced.png"
    if last_image is not None and not bool(args.no_save_image):
        single_cli._save_tensor_image(image_path, last_image)
    metric_stats = None
    if last_image is not None and metric_reference_image is not None and not bool(args.skip_metrics):
        metric_stats = compute_interlaced_metric_stats(
            last_image,
            metric_reference_image,
            require_lpips=bool(args.require_lpips),
        )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "omg4_ftgs_interlaced_experiment",
        "engine": ENGINE_NAME,
        "scene": job.scene,
        "dataset_kind": "n3dv",
        "dataset_scene": job.dataset_scene,
        "weight_group": str(args.weight_group),
        "variant": variant.name,
        "group": variant.group,
        "use_remapping": bool(variant.use_remapping),
        "reuse_enabled": bool(variant.reuse_enabled),
        "checkpoint_path": str(job.checkpoint_path),
        "checkpoint": str(job.checkpoint_path),
        "data_path": str(job.data_path),
        "camera_index": int(args.camera_index),
        "frame_index": int(frame.index),
        "timestamp": float(frame.timestamp),
        "resolution": float(args.resolution),
        "panel_width": int(args.panel_width),
        "panel_height": int(args.panel_height),
        "render_width": int(args.panel_width),
        "render_height": int(args.panel_height),
        "views": int(args.views),
        "source_views": int(args.views),
        "cluster_size": int(variant.cluster_size),
        "effective_cluster_size": effective_cluster_size(variant),
        "color_eval_views": color_eval_view_count(int(args.views), variant),
        "tile_size": int(args.tile_size),
        "map_mode": str(args.map_mode),
        "map_metadata": dict(prepared.map_metadata),
        "view_degree": float(args.view_degree),
        "orbit_direction": int(args.orbit_direction),
        "orbit_center_distance": float(args.orbit_center_distance),
        "orbit_center": single_cli._tensor_to_float_list(prepared.orbit_center),
        "warmup_iters": int(args.warmup_iters),
        "measure_iters": int(args.measure_iters),
        "image_path": str(image_path) if not bool(args.no_save_image) else None,
        "timing": averaged,
        "metrics": metric_stats.__dict__ if metric_stats is not None else None,
        "metric_reference_variant": "cluster_1" if metric_stats is not None else None,
        "metric_scope": "interlaced_cluster_reference" if metric_stats is not None else None,
        "samples": samples,
        "gaussians": int(_first_numeric(averaged.get("gaussians"), prepared.setup_gaussians)),
        "total_gaussians": int(_first_numeric(averaged.get("gaussians"), prepared.setup_gaussians)),
        "args": dict(vars(args)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(single_cli._json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return scene_summary_from_manifest(job, output_dir, args=args, variant=variant)


def render_metric_reference_interlaced(job: SceneJob, args: argparse.Namespace):
    frames = load_test_frames(job.data_path, camera_index=int(args.camera_index))
    camera_set = load_pose_camera(job.data_path, camera_index=int(args.camera_index), resolution=float(args.resolution))
    dynamic_model = load_dynamic_gaussians(job.checkpoint_path, device=str(args.device))
    frame = single_cli._select_frame(frames, int(args.frame_index))
    variant = metric_reference_variant()
    prepared = prepare_interlaced_scene(args=args, camera_set=camera_set, dynamic_model=dynamic_model, frame=frame, variant=variant)
    image, _timing = render_interlaced_once(
        args=args,
        dynamic_model=dynamic_model,
        prepared=prepared,
        timestamp=float(frame.timestamp),
        variant=variant,
    )
    return image


def prepare_interlaced_scene(
    *,
    args: argparse.Namespace,
    camera_set: Any,
    dynamic_model: Any,
    frame: Any,
    variant: ExperimentVariant | None = None,
) -> PreparedInterlacedScene:
    import torch

    variant = resolve_runtime_variant(args, variant)
    panel_width = int(args.panel_width)
    panel_height = int(args.panel_height)
    _sync_device(str(args.device))
    viewpoint_start = time.perf_counter()
    viewpoint_index, map_metadata = build_interlaced_viewpoint_index(args, width=panel_width, height=panel_height)
    _sync_device(str(args.device))
    viewpoint_index_ms = (time.perf_counter() - viewpoint_start) * 1000.0
    prepared_lookup = prepare_cr_lookup_tensors(
        viewpoint_index,
        device=str(args.device),
        tile_size=int(args.tile_size),
        use_remapping=bool(variant.use_remapping),
    )

    _sync_device(str(args.device))
    start = time.perf_counter()
    setup_splats = dynamic_model.materialize(float(frame.timestamp))
    _sync_device(str(args.device))
    setup_materialize_ms = (time.perf_counter() - start) * 1000.0

    _sync_device(str(args.device))
    view_setup_start = time.perf_counter()
    source_viewmat = torch.as_tensor(camera_set.viewmat_at(int(args.camera_index)), dtype=torch.float32, device=str(args.device)).contiguous()
    c2w = torch.linalg.inv(source_viewmat)
    orbit_center = estimate_orbit_center(
        c2w=c2w,
        splat_means=setup_splats.means,
        orbit_center_distance=float(args.orbit_center_distance),
    )
    adjacent_viewmats = synthesize_interlaced_viewmats(
        c2w=c2w,
        orbit_center=orbit_center,
        views=int(args.views),
        cluster_size=effective_cluster_size(variant),
        view_degree=float(args.view_degree),
        orbit_direction=int(args.orbit_direction),
        device=str(args.device),
    )
    K = single_cli._scaled_panel_K(args, camera_set)
    _sync_device(str(args.device))
    setup_view_setup_ms = (time.perf_counter() - view_setup_start) * 1000.0
    return PreparedInterlacedScene(
        viewpoint_index=viewpoint_index,
        map_metadata=map_metadata,
        view_idx_matrix=prepared_lookup.view_idx_matrix,
        subpixel_coord_matrix=prepared_lookup.subpixel_coord_matrix,
        viewpoint_index_ms=float(viewpoint_index_ms),
        lookup_cpu_ms=float(prepared_lookup.lookup_cpu_ms),
        lookup_h2d_ms=float(prepared_lookup.lookup_h2d_ms),
        source_c2w=c2w,
        adjacent_viewmats=adjacent_viewmats,
        K=K,
        orbit_center=orbit_center,
        setup_materialize_ms=float(setup_materialize_ms),
        setup_view_setup_ms=float(setup_view_setup_ms),
        setup_gaussians=int(setup_splats.means.shape[0]),
    )


def render_interlaced_once(
    *,
    args: argparse.Namespace,
    dynamic_model: Any,
    prepared: PreparedInterlacedScene,
    timestamp: float,
    variant: ExperimentVariant | None = None,
) -> tuple[Any, dict[str, Any]]:
    variant = resolve_runtime_variant(args, variant)
    device = str(args.device)
    _sync_device(device)
    end_to_end_start = time.perf_counter()
    _sync_device(device)
    materialize_start = time.perf_counter()
    splats = dynamic_model.materialize(float(timestamp))
    _sync_device(device)
    materialize_ms = (time.perf_counter() - materialize_start) * 1000.0

    _sync_device(device)
    view_setup_start = time.perf_counter()
    orbit_center = estimate_orbit_center(
        c2w=prepared.source_c2w,
        splat_means=splats.means,
        orbit_center_distance=float(args.orbit_center_distance),
    )
    adjacent_viewmats = synthesize_interlaced_viewmats(
        c2w=prepared.source_c2w,
        orbit_center=orbit_center,
        views=int(args.views),
        cluster_size=effective_cluster_size(variant),
        view_degree=float(args.view_degree),
        orbit_direction=int(args.orbit_direction),
        device=device,
    )
    _sync_device(device)
    view_setup_ms = (time.perf_counter() - view_setup_start) * 1000.0

    _sync_device(device)
    render_start = time.perf_counter()
    render_result = render_splats_interlaced_coherent(
        splats,
        adjacent_viewmats=adjacent_viewmats,
        K=prepared.K,
        viewpoint_index=prepared.viewpoint_index,
        view_idx_matrix=prepared.view_idx_matrix,
        subpixel_coord_matrix=prepared.subpixel_coord_matrix,
        width=int(args.panel_width),
        height=int(args.panel_height),
        device=device,
        tile_size=int(args.tile_size),
        near_plane=float(args.near_plane),
        far_plane=float(args.far_plane),
        camera_model=str(args.camera_model),
        debug=bool(args.debug_cr),
        return_meta=True,
        return_timing=True,
    )
    _sync_device(device)
    cr_render_ms = (time.perf_counter() - render_start) * 1000.0
    render_once_end_to_end_ms = (time.perf_counter() - end_to_end_start) * 1000.0

    if isinstance(render_result, tuple) and len(render_result) == 2:
        image, render_meta = render_result
    else:
        image, render_meta = render_result, {}
    cr_timing_raw = dict((render_meta or {}).get("timing_ms", {}))
    cr_timing_ms = {
        str(key): numeric
        for key, value in cr_timing_raw.items()
        for numeric in [_float_or_none(value)]
        if numeric is not None
    }
    cr_timing_source = str(cr_timing_raw.get("cr_timing_source", ""))
    cr_core_ms = _finite_sum(
        cr_timing_ms.get("cr_projection_ms", 0.0),
        cr_timing_ms.get("cr_keygen_ms", 0.0),
        cr_timing_ms.get("cr_blend_ms", 0.0),
    )
    post_ms = float((render_meta or {}).get("post_ms", 0.0))
    frame_ms = float(materialize_ms + view_setup_ms + cr_render_ms)
    lookup_ms = float(prepared.viewpoint_index_ms + prepared.lookup_cpu_ms + prepared.lookup_h2d_ms)
    frame_ms_including_lookup = float(frame_ms + lookup_ms)
    frame_ms_end_to_end = float(render_once_end_to_end_ms + lookup_ms)
    timing = {
        "materialize_ms": float(materialize_ms),
        "view_setup_ms": float(view_setup_ms),
        "cr_render_ms": float(cr_render_ms),
        "cr_core_ms": float(cr_core_ms),
        "cr_core_total_ms": float(cr_core_ms),
        "post_ms": post_ms,
        "cr_projection_ms": float(cr_timing_ms.get("cr_projection_ms", 0.0)),
        "cr_isect_ms": float(cr_timing_ms.get("cr_isect_ms", 0.0)),
        "cr_keygen_ms": float(cr_timing_ms.get("cr_keygen_ms", 0.0)),
        "cr_sort_ms": float(cr_timing_ms.get("cr_sort_ms", 0.0)),
        "cr_offset_ms": float(cr_timing_ms.get("cr_offset_ms", 0.0)),
        "cr_blend_ms": float(cr_timing_ms.get("cr_blend_ms", 0.0)),
        "cr_timing_source": cr_timing_source,
        "frame_ms_with_lkg_interlace": frame_ms,
        "frame_ms_excluding_lookup": frame_ms,
        "fps_with_lkg_interlace": fps_from_ms(frame_ms),
        "frame_ms_including_lookup": frame_ms_including_lookup,
        "fps_including_lookup": fps_from_ms(frame_ms_including_lookup),
        "frame_ms_end_to_end": frame_ms_end_to_end,
        "fps_end_to_end": fps_from_ms(frame_ms_end_to_end),
        "viewpoint_index_ms": float(prepared.viewpoint_index_ms),
        "gaussians": float(splats.means.shape[0]),
    }
    return image, timing


def average_measurements(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("at least one measurement sample is required")
    keys = sorted({str(key) for sample in samples for key in sample.keys()})
    averaged: dict[str, Any] = {}
    for key in keys:
        values = [float(sample[key]) for sample in samples if key in sample and _is_finite_number(sample[key])]
        if values:
            averaged[key] = float(np.mean(values))
            continue
        text_values = [str(sample[key]) for sample in samples if key in sample and isinstance(sample[key], str) and str(sample[key])]
        if text_values:
            first = text_values[0]
            averaged[key] = first if all(value == first for value in text_values) else "mixed"
    if "frame_ms_with_lkg_interlace" not in averaged:
        averaged["frame_ms_with_lkg_interlace"] = float(
            averaged.get("materialize_ms", 0.0)
            + averaged.get("view_setup_ms", 0.0)
            + averaged.get("cr_render_ms", averaged.get("cr_core_ms", 0.0) + averaged.get("post_ms", 0.0))
        )
    if "frame_ms_excluding_lookup" not in averaged:
        averaged["frame_ms_excluding_lookup"] = float(averaged["frame_ms_with_lkg_interlace"])
    averaged["fps_with_lkg_interlace"] = fps_from_ms(averaged["frame_ms_with_lkg_interlace"])
    averaged["frame_ms"] = float(averaged["frame_ms_with_lkg_interlace"])
    averaged["fps"] = float(averaged["fps_with_lkg_interlace"])
    if "frame_ms_including_lookup" in averaged:
        averaged["fps_including_lookup"] = fps_from_ms(averaged["frame_ms_including_lookup"])
    _ensure_end_to_end_timing(averaged)
    return averaged


def fps_from_ms(frame_ms: float) -> float:
    value = float(frame_ms)
    return 1000.0 / value if value > 0.0 and math.isfinite(value) else 0.0


def _ensure_end_to_end_timing(timing: dict[str, Any]) -> None:
    end_to_end_ms = _finite_float_or_none(timing.get("frame_ms_end_to_end"))
    if end_to_end_ms is None:
        end_to_end_ms = _finite_float_or_none(timing.get("frame_ms_including_lookup"))
    if end_to_end_ms is None:
        return
    timing["frame_ms_end_to_end"] = float(end_to_end_ms)
    timing["fps_end_to_end"] = fps_from_ms(end_to_end_ms)


def _finite_float_or_none(value: Any) -> float | None:
    numeric = _float_or_none(value)
    if numeric is None or not math.isfinite(float(numeric)):
        return None
    return float(numeric)


def scene_summary_from_manifest(
    job: SceneJob,
    output_dir: Path,
    *,
    args: argparse.Namespace | None = None,
    variant: ExperimentVariant | None = None,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return scene_status_row(
            status="missing",
            job=job,
            output_dir=output_dir,
            args=args,
            variant=variant,
            message="manifest.json not found",
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    timing = dict(manifest.get("timing") or {})
    _ensure_end_to_end_timing(timing)
    metrics = dict(manifest.get("metrics") or {})
    row = scene_status_row(
        status="ok",
        job=job,
        output_dir=output_dir,
        args=args,
        variant=variant,
        message="",
    )
    row.update(
        {
            "dataset_kind": manifest.get("dataset_kind", row["dataset_kind"]),
            "dataset_scene": manifest.get("dataset_scene", job.dataset_scene),
            "weight_group": manifest.get("weight_group", "" if args is None else str(args.weight_group)),
            "checkpoint": manifest.get("checkpoint", manifest.get("checkpoint_path", str(job.checkpoint_path))),
            "camera_index": manifest.get("camera_index", row["camera_index"]),
            "frame_index": manifest.get("frame_index", row["frame_index"]),
            "timestamp": manifest.get("timestamp", ""),
            "engine": manifest.get("engine", row["engine"]),
            "variant": manifest.get("variant", row["variant"]),
            "group": manifest.get("group", row["group"]),
            "panel_width": manifest.get("panel_width", row["panel_width"]),
            "panel_height": manifest.get("panel_height", row["panel_height"]),
            "render_width": manifest.get("render_width", manifest.get("panel_width", row["render_width"])),
            "render_height": manifest.get("render_height", manifest.get("panel_height", row["render_height"])),
            "views": manifest.get("views", row["views"]),
            "source_views": manifest.get("source_views", manifest.get("views", row["source_views"])),
            "cluster_size": manifest.get("cluster_size", row["cluster_size"]),
            "color_eval_views": manifest.get("color_eval_views", row["color_eval_views"]),
            "use_remapping": manifest.get("use_remapping", row["use_remapping"]),
            "reuse_enabled": manifest.get("reuse_enabled", row["reuse_enabled"]),
            "tile_size": manifest.get("tile_size", row["tile_size"]),
            "map_mode": manifest.get("map_mode", row["map_mode"]),
            "metric_reference_variant": manifest.get("metric_reference_variant", row["metric_reference_variant"]),
            "metric_scope": manifest.get("metric_scope", row["metric_scope"]),
            "image_path": manifest.get("image_path") or "",
            "manifest_path": str(manifest_path),
            "checkpoint_path": manifest.get("checkpoint_path", str(job.checkpoint_path)),
            "data_path": manifest.get("data_path", str(job.data_path)),
            "gaussians": manifest.get("gaussians", timing.get("gaussians", "")),
            "total_gaussians": manifest.get("total_gaussians", manifest.get("gaussians", timing.get("gaussians", ""))),
        }
    )
    if "cr_core_total_ms" not in timing and "cr_core_ms" in timing:
        row["cr_core_total_ms"] = timing["cr_core_ms"]
    for key in SUMMARY_COLUMNS:
        if key in timing:
            row[key] = timing[key]
        if key in metrics:
            row[key] = metrics[key]
    return row


def scene_status_row(
    *,
    status: str,
    job: SceneJob,
    output_dir: Path,
    args: argparse.Namespace | None,
    variant: ExperimentVariant | None = None,
    message: str,
) -> dict[str, Any]:
    resolved_variant = resolve_runtime_variant(args, variant) if args is not None else variant
    panel_width = "" if args is None else int(args.panel_width)
    panel_height = "" if args is None else int(args.panel_height)
    source_views = "" if args is None else int(args.views)
    return {
        "status": str(status),
        "dataset_kind": "n3dv",
        "scene": job.scene,
        "checkpoint": str(job.checkpoint_path),
        "dataset_scene": job.dataset_scene,
        "weight_group": "" if args is None else str(args.weight_group),
        "camera_split": "",
        "camera_index": "" if args is None else int(args.camera_index),
        "frame_index": "" if args is None else int(args.frame_index),
        "timestamp": "",
        "output_prefix": "",
        "engine": "" if args is None else ENGINE_NAME,
        "variant": "" if resolved_variant is None else str(resolved_variant.name),
        "group": "" if resolved_variant is None else str(resolved_variant.group),
        "panel_width": panel_width,
        "panel_height": panel_height,
        "render_width": panel_width,
        "render_height": panel_height,
        "views": source_views,
        "source_views": source_views,
        "saved_views": "",
        "cluster_size": "" if resolved_variant is None else int(resolved_variant.cluster_size),
        "color_eval_views": "" if args is None or resolved_variant is None else color_eval_view_count(int(args.views), resolved_variant),
        "use_remapping": "" if resolved_variant is None else bool(resolved_variant.use_remapping),
        "reuse_enabled": "" if resolved_variant is None else bool(resolved_variant.reuse_enabled),
        "tile_size": "" if args is None else int(args.tile_size),
        "map_mode": "" if args is None else str(args.map_mode),
        "frame_ms": "",
        "fps": "",
        "frame_ms_end_to_end": "",
        "fps_end_to_end": "",
        "frame_ms_with_lkg_interlace": "",
        "frame_ms_excluding_lookup": "",
        "fps_with_lkg_interlace": "",
        "frame_ms_including_lookup": "",
        "fps_including_lookup": "",
        "materialize_ms": "",
        "view_setup_ms": "",
        "cr_render_ms": "",
        "cr_core_ms": "",
        "viewpoint_index_ms": "",
        "lookup_cpu_ms": "",
        "lookup_h2d_ms": "",
        "post_ms": "",
        "cr_projection_ms": "",
        "cr_isect_ms": "",
        "cr_keygen_ms": "",
        "cr_sort_ms": "",
        "cr_offset_ms": "",
        "cr_blend_ms": "",
        "psnr_mean": "",
        "psnr_std": "",
        "ssim_mean": "",
        "ssim_std": "",
        "lpips_mean": "",
        "lpips_std": "",
        "metric_view_count": "",
        "metric_reference_variant": "",
        "metric_scope": "",
        "cr_core_total_ms": "",
        "total_gaussians": "",
        "active_gaussians": "",
        "active_gaussian_ratio": "",
        "gaussians": "",
        "peak_vram_gb": "",
        "output_dir": str(output_dir),
        "image_path": str(output_dir / "omg4_ftgs_lkg_interlaced.png"),
        "manifest_path": str(output_dir / "manifest.json"),
        "checkpoint_path": str(job.checkpoint_path),
        "data_path": str(job.data_path),
        "message": str(message),
    }


def build_scene_output_dir(args: argparse.Namespace, job: SceneJob, *, variant: ExperimentVariant | None = None) -> Path:
    path = Path(args.output_root).expanduser() / _resolve_run_group(args) / str(args.weight_group) / job.scene
    if variant is not None:
        return path / str(variant.name)
    return path


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(single_cli._json_ready(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _summary_csv_ready(row.get(key, "")) for key in SUMMARY_COLUMNS})


def write_manifest(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    run_group: str,
    scene_count: int,
    failure_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "omg4_ftgs_interlaced_experiment",
        "run_group": run_group,
        "weight_group": str(args.weight_group),
        "panel_width": int(args.panel_width),
        "panel_height": int(args.panel_height),
        "views": int(args.views),
        "cluster_size": int(args.cluster_size),
        "clusters": str(args.clusters),
        "ablation_cluster": int(args.ablation_cluster),
        "variants": [variant.to_json() for variant in resolve_experiment_variants(args)],
        "warmup_iters": int(args.warmup_iters),
        "measure_iters": int(args.measure_iters),
        "scene_count": int(scene_count),
        "failure_count": int(failure_count),
        "output_dir": str(output_dir),
        "summary_csv": str(output_dir / "summary.csv"),
        "summary_json": str(output_dir / "summary.json"),
        "args": dict(vars(args)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(single_cli._json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


def _summary_csv_ready(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_scenes(scenes: str | list[str] | None) -> list[str] | None:
    if scenes is None:
        return None
    if isinstance(scenes, list):
        return [str(scene) for scene in scenes if str(scene)]
    tokens = str(scenes).replace(",", " ").split()
    return tokens or None


def _resolve_run_group(args: argparse.Namespace) -> str:
    if args.run_group:
        return str(args.run_group)
    return "omg4_ftgs_interlaced_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _sync_device(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _import_torch():
    import torch

    return torch


def _import_tinycudann():
    import tinycudann

    return tinycudann


def _cuda_visible_devices_hint(capabilities: Sequence[tuple[int, int]]) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    unique_caps = sorted(set(capabilities))
    if len(unique_caps) > 1 and not visible:
        preferred = _first_device_index_with_max_capability(capabilities)
        return (
            "The visible GPUs have mixed compute capabilities, so tinycudann may pick the lowest one. "
            f"Retry with CUDA_VISIBLE_DEVICES={preferred} before python, for example: "
            f"CUDA_VISIBLE_DEVICES={preferred} python {_current_script_name()} ..."
        )
    if visible:
        return f"CUDA_VISIBLE_DEVICES is currently {visible!r}; choose a GPU with a tinycudann extension built for its compute capability."
    return "Choose a CUDA device with a tinycudann extension built for its compute capability."


def _first_device_index_with_max_capability(capabilities: Sequence[tuple[int, int]]) -> int:
    if not capabilities:
        return 0
    best = max(capabilities)
    return next((index for index, capability in enumerate(capabilities) if capability == best), 0)


def _current_script_name() -> str:
    name = Path(sys.argv[0]).name
    return name or "omg4_ftgs_interlaced_experiment.py"


def _reset_cuda_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _peak_vram_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / float(2**30)
    except Exception:
        return 0.0
    return 0.0


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _first_numeric(*values: Any) -> float:
    for value in values:
        if _is_finite_number(value):
            return float(value)
    return 0.0


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finite_sum(*values: Any) -> float:
    return float(sum(float(value) for value in values if _is_finite_number(value)))
