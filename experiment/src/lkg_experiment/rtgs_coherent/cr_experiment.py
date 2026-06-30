from __future__ import annotations

import argparse
import inspect
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from lkg_experiment.coherent_default.coherent_raster_experiment import (
    ArtifactWriter,
    ExperimentVariant,
    MetricStats,
    REFERENCE_INTERLACED_VARIANT_NAME,
    TimingStats,
    build_experiment_variants,
    build_experiment_web_assets,
    cluster_index_from_view_index,
    image_artifact_path,
    parse_cluster_values,
    read_metrics_csv,
    reference_interlaced_artifact_path,
    remove_matching_metric_rows,
    select_metric_view_indices,
)
from lkg_experiment.rtgs_coherent import cr_66views
from lkg_experiment.rtgs_coherent.cli import DEFAULT_GENERATED_ROOT, _json_ready, compute_pair_metrics
from lkg_experiment.rtgs_coherent.cr_1view import adapt_snapshot_for_rtgs_compat_projection


DEFAULT_RTGS_CR_EXPERIMENT_ROOT = DEFAULT_GENERATED_ROOT / "rtgs_cr_experiments"
ENGINE_CHOICES = ("one_shot",)


@dataclass(frozen=True)
class RtgsCrTimingStats:
    fps: float
    frame_ms: float
    peak_vram_gb: float
    detailed_metrics: Mapping[str, float]


def build_parser() -> argparse.ArgumentParser:
    parser = cr_66views.build_parser()
    parser.description = "Run one-shot RTGS + CoherentRaster experiments with 3DGS experiment-compatible artifacts"
    parser.add_argument("--artifact-dir", default=str(DEFAULT_RTGS_CR_EXPERIMENT_ROOT))
    parser.add_argument("--run-id", help="Output run directory name")
    parser.add_argument("--output-prefix", default="", help="Prefix for per-camera images inside a run")
    parser.add_argument("--append-metrics", action="store_true", help="Append rows in an existing run directory")
    parser.add_argument(
        "--engine",
        choices=ENGINE_CHOICES,
        default="one_shot",
        help="Experiment engine. Only one_shot grouped CoherentRaster rendering is supported.",
    )
    parser.add_argument("--clusters", default="2,4,8,16")
    parser.add_argument("--ablation-cluster", default=8, type=int)
    parser.add_argument("--no-without-remap", action="store_true")
    parser.add_argument("--no-without-reuse", action="store_true")
    parser.add_argument("--warmup-iters", default=3, type=int)
    parser.add_argument("--measure-iters", default=5, type=int)
    parser.add_argument("--metric-view-stride", default=1, type=int)
    parser.add_argument("--max-metric-views", default=5, type=int)
    parser.add_argument(
        "--all-metric-views",
        action="store_true",
        help="Evaluate metrics on all rendered source views; overrides --max-metric-views.",
    )
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--no-reference-interlaced", action="store_true")
    parser.add_argument("--skip-web-assets", action="store_true")
    parser.add_argument("--write-mapping-artifacts", action="store_true")
    parser.add_argument("--write-mapping-previews", action="store_true")
    parser.add_argument("--progress-every-views", default=5, type=int)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run_rtgs_cr_experiment(args)


def run_rtgs_cr_experiment(args: argparse.Namespace) -> int:
    validate_args(args)

    if not _cuda_available():
        raise RuntimeError("CUDA is required for RTGS + CR experiment rendering")

    import torch

    device = str(args.device)
    context = cr_66views.prepare_rtgs_cr_66_context(args)

    run_id = args.run_id or make_run_id(
        model_path=Path(args.model_path),
        engine=str(args.engine),
        split=str(args.split),
        camera_index=int(args.camera_index),
        n3dv_frame_index=int(args.n3dv_frame_index),
    )
    artifact_root = Path(args.artifact_dir).expanduser() / run_id
    writer = ArtifactWriter(artifact_root)

    variants = resolve_experiment_variants(args)
    metric_view_indices = resolve_metric_view_indices(int(context.source_view_count), args)

    reference_interlaced = None
    if not bool(args.no_reference_interlaced):
        print("Rendering official RTGS reference interlaced image...", file=sys.stderr, flush=True)
        reference_interlaced = render_official_reference_interlaced(context)
        writer.save_tensor_image(
            reference_interlaced_artifact_path(output_prefix=str(args.output_prefix)),
            reference_interlaced,
        )

    if args.append_metrics:
        rows: list[dict[str, Any]] = remove_matching_metric_rows(
            read_metrics_csv(artifact_root / "metrics.csv"),
            output_prefix=str(args.output_prefix),
            camera_split=str(args.split),
            camera_index=int(args.camera_index),
        )
    else:
        rows = []

    cluster_indices = {variant.name: cluster_index_from_view_index(context.viewpoint_index, variant.cluster_size) for variant in variants}
    if bool(args.write_mapping_artifacts) or bool(args.write_mapping_previews):
        writer.write_mapping_npz(context.viewpoint_index, cluster_indices)
        writer.save_mapping_values("view_index", context.viewpoint_index)
        if bool(args.write_mapping_previews):
            writer.save_index_previews("view_index", context.viewpoint_index)
            for variant in variants:
                writer.save_index_previews(f"{variant.name}_cluster_index", cluster_indices[variant.name])

    last_interlaced = None
    for variant in variants:
        print(
            f"Running RTGS+CR experiment variant {variant.name}: engine={args.engine}, "
            f"cluster={variant.cluster_size}, remap={int(variant.use_remapping)}, reuse={int(variant.reuse_enabled)}",
            file=sys.stderr,
            flush=True,
        )
        if is_reference_gt_variant(variant):
            if reference_interlaced is None:
                raise ValueError("without_reuse is the GT/reference variant and requires reference_interlaced")
            interlaced = reference_interlaced
            timing = reference_gt_timing_stats()
        else:
            interlaced, timing = time_one_shot_renderer(
                context,
                variant=variant,
                warmup_iters=int(args.warmup_iters),
                measure_iters=int(args.measure_iters),
            )
        last_interlaced = interlaced
        writer.save_tensor_image(
            image_artifact_path(variant.name, "looking_glass_tensor.png", output_prefix=str(args.output_prefix)),
            interlaced,
        )
        if reference_interlaced is not None:
            if is_reference_gt_variant(variant):
                abs_error = interlaced.new_zeros(interlaced.shape)
            else:
                abs_error = (interlaced - reference_interlaced).abs().mul(8.0).clamp(0.0, 1.0)
            writer.save_tensor_image(
                image_artifact_path(variant.name, "abs_error.png", output_prefix=str(args.output_prefix)),
                abs_error,
            )

        metrics = None
        if not bool(args.skip_metrics):
            if is_reference_gt_variant(variant):
                metrics = reference_gt_metric_stats(metric_view_indices)
            else:
                metrics = compute_sampled_view_metric_stats(context, metric_view_indices)
        rows.append(
            build_metric_row(
                variant=variant,
                engine=str(args.engine),
                camera_split=str(args.split),
                camera_index=int(args.camera_index),
                output_prefix=str(args.output_prefix),
                timing=timing,
                metrics=metrics,
                context=context,
            )
        )
        writer.write_metrics_csv(rows)
        writer.write_metrics_json(rows)

    manifest = build_manifest(
        args=args,
        context=context,
        run_id=run_id,
        artifact_root=artifact_root,
        variants=variants,
        metric_view_indices=metric_view_indices,
        reference_interlaced_written=reference_interlaced is not None,
        last_interlaced=last_interlaced,
    )
    writer.write_manifest(manifest)
    writer.write_metrics_csv(rows)
    writer.write_metrics_json(rows)
    if not bool(args.skip_web_assets):
        build_experiment_web_assets(args.artifact_dir, regenerate_previews=False)
    print(f"RTGS + CR experiment artifacts written to {artifact_root}", file=sys.stderr, flush=True)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if int(args.width) <= 0 or int(args.height) <= 0 or int(args.views) <= 0:
        raise ValueError("--width, --height, and --views must be positive")
    if int(args.tile_size) <= 0:
        raise ValueError("--tile-size must be positive")
    if int(args.warmup_iters) < 0 or int(args.measure_iters) <= 0:
        raise ValueError("--warmup-iters must be non-negative and --measure-iters must be positive")
    if int(args.metric_view_stride) <= 0:
        raise ValueError("--metric-view-stride must be positive")
    if int(args.max_metric_views) < 0:
        raise ValueError("--max-metric-views must be non-negative")
    if int(args.progress_every_views) < 0:
        raise ValueError("--progress-every-views must be non-negative")
    if int(args.ablation_cluster) <= 0:
        raise ValueError("--ablation-cluster must be positive")
    parse_cluster_values(args.clusters)


def resolve_experiment_variants(args: argparse.Namespace) -> list[ExperimentVariant]:
    return build_experiment_variants(
        parse_cluster_values(args.clusters),
        ablation_cluster=int(args.ablation_cluster),
        include_without_remap=not bool(args.no_without_remap),
        include_without_reuse=not bool(args.no_without_reuse),
    )


def is_reference_gt_variant(variant: Any) -> bool:
    return str(getattr(variant, "name", "")) == REFERENCE_INTERLACED_VARIANT_NAME and int(getattr(variant, "cluster_size", 0)) == 1


def reference_gt_timing_stats() -> TimingStats:
    return TimingStats(fps=float("nan"), frame_ms=float("nan"), peak_vram_gb=float("nan"))


def reference_gt_metric_stats(metric_view_indices: Sequence[int]) -> MetricStats:
    return MetricStats(
        psnr_mean=float("inf"),
        psnr_std=0.0,
        ssim_mean=1.0,
        ssim_std=0.0,
        lpips_mean=float("nan"),
        lpips_std=float("nan"),
        metric_view_count=int(len(metric_view_indices)),
    )


def resolve_metric_view_indices(source_view_count: int, args: argparse.Namespace) -> list[int]:
    max_metric_views = 0 if bool(args.all_metric_views) else int(args.max_metric_views)
    return select_metric_view_indices(
        int(source_view_count),
        max_metric_views=max_metric_views,
        stride=int(args.metric_view_stride),
    )


def make_run_id(
    *,
    model_path: Path,
    engine: str,
    split: str,
    camera_index: int,
    n3dv_frame_index: int,
    suffix: str | None = None,
) -> str:
    timestamp = suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    scene = _safe_name(Path(model_path).expanduser().name)
    return (
        f"{scene}_{_safe_name(engine)}_{_safe_name(split)}_cam{int(camera_index):03d}_"
        f"frame{int(n3dv_frame_index):04d}_{timestamp}"
    )


def build_metric_row(
    *,
    variant: Any,
    engine: str,
    camera_split: str,
    camera_index: int,
    output_prefix: str,
    timing: TimingStats,
    metrics: MetricStats | None,
    context: cr_66views.RtgsCr66RenderContext | None = None,
) -> dict[str, Any]:
    detailed_metrics = dict(getattr(timing, "detailed_metrics", {}) or {})
    legacy_frame_ms = float(timing.frame_ms)
    legacy_fps = float(timing.fps)
    frame_ms = float(detailed_metrics.get("frame_ms_with_lkg_interlace", legacy_frame_ms))
    fps = float(detailed_metrics.get("fps_with_lkg_interlace", legacy_fps))
    row: dict[str, Any] = {
        "camera_split": str(camera_split),
        "camera_index": int(camera_index),
        "output_prefix": str(output_prefix),
        "variant": str(variant.name),
        "engine": str(engine),
        "group": str(variant.group),
        "cluster_size": int(variant.cluster_size),
        "use_remapping": bool(variant.use_remapping),
        "reuse_enabled": bool(variant.reuse_enabled),
        "fps": fps,
        "frame_ms": frame_ms,
        "peak_vram_gb": float(timing.peak_vram_gb),
    }
    if context is not None:
        row.update(_context_metric_columns(context, variant=variant))
    row.update(detailed_metrics)
    if "frame_ms_with_lkg_interlace" in detailed_metrics:
        row["frame_ms"] = float(detailed_metrics["frame_ms_with_lkg_interlace"])
    if "fps_with_lkg_interlace" in detailed_metrics:
        row["fps"] = float(detailed_metrics["fps_with_lkg_interlace"])
    if metrics is not None:
        row.update(metrics.__dict__)
    return row


def _context_metric_columns(context: cr_66views.RtgsCr66RenderContext, *, variant: Any) -> dict[str, Any]:
    args = context.args
    official = context.runtime.official
    total_gaussians = int(official.gaussians.get_xyz.shape[0])
    active_gaussians = int(context.geometry.means.shape[0])
    source_views = int(context.source_view_count)
    if bool(variant.reuse_enabled):
        color_eval_views = int(math.ceil(source_views / float(int(variant.cluster_size))))
    else:
        color_eval_views = source_views
    return {
        "dataset_kind": str(getattr(official.scene_paths, "dataset_kind", getattr(args, "dataset_kind", ""))),
        "scene": str(official.scene_name),
        "checkpoint": str(official.checkpoint_path),
        "frame_index": int(getattr(args, "n3dv_frame_index", 0)),
        "timestamp": float(context.timestamp),
        "render_width": int(context.render_width),
        "render_height": int(context.render_height),
        "source_views": source_views,
        "saved_views": int(getattr(args, "sample_save_views", 0)),
        "color_eval_views": color_eval_views,
        "tile_size": int(getattr(args, "tile_size", 0)),
        "map_mode": str(getattr(args, "map_mode", "")),
        "camera_aspect_mode": str(getattr(args, "camera_aspect_mode", "")),
        "total_gaussians": total_gaussians,
        "active_gaussians": active_gaussians,
        "active_gaussian_ratio": float(active_gaussians / total_gaussians) if total_gaussians else float("nan"),
    }


def time_one_shot_renderer(
    context: cr_66views.RtgsCr66RenderContext,
    *,
    variant: Any | None = None,
    warmup_iters: int,
    measure_iters: int,
    render_fn: Callable[[cr_66views.RtgsCr66RenderContext], tuple[Any, Any]] | None = None,
    progress_label: str | None = None,
) -> tuple[Any, TimingStats]:
    if int(warmup_iters) < 0 or int(measure_iters) <= 0:
        raise ValueError("warmup_iters must be non-negative and measure_iters must be positive")
    if render_fn is None:
        if variant is None:
            raise ValueError("variant is required when render_fn is not provided")
        render_fn = lambda current_context: render_one_shot_interlaced_once(current_context, variant=variant)
        if progress_label is None:
            progress_label = f"one_shot/{variant.name}"
    _reset_cuda_peak_memory()
    last = None
    total_ms = 0.0
    count = 0
    detailed_totals: dict[str, float] = {}
    total_iterations = int(warmup_iters) + int(measure_iters)
    for iteration in range(total_iterations):
        phase = "warmup" if iteration < int(warmup_iters) else "measure"
        if progress_label:
            print(f"{progress_label} timing {iteration + 1}/{total_iterations} {phase}", file=sys.stderr, flush=True)
        last, render_timing = _call_render_fn(render_fn, context, progress_label=progress_label)
        render_ms, detailed = _render_timing_values(render_timing)
        if iteration >= int(warmup_iters):
            total_ms += float(render_ms)
            for key, value in detailed.items():
                detailed_totals[key] = detailed_totals.get(key, 0.0) + float(value)
            count += 1
    average_ms = total_ms / float(count) if count else math.inf
    detailed_average = {key: value / float(count) for key, value in detailed_totals.items()} if count else {}
    detailed_average = _normalize_averaged_timing_metrics(detailed_average)
    if "frame_ms_with_lkg_interlace" in detailed_average:
        average_ms = float(detailed_average["frame_ms_with_lkg_interlace"])
    fps = 1000.0 / average_ms if average_ms > 0.0 and math.isfinite(average_ms) else 0.0
    if detailed_average:
        detailed_average["frame_ms_with_lkg_interlace"] = float(average_ms)
        detailed_average["fps_with_lkg_interlace"] = float(fps)
        return last, RtgsCrTimingStats(
            fps=fps,
            frame_ms=average_ms,
            peak_vram_gb=_peak_vram_gb(),
            detailed_metrics=detailed_average,
        )
    return last, TimingStats(fps=fps, frame_ms=average_ms, peak_vram_gb=_peak_vram_gb())


def render_one_shot_interlaced_once(context: cr_66views.RtgsCr66RenderContext, *, variant: Any):
    from lkg_experiment.rtgs_coherent import cr_one_shot

    return cr_one_shot.render_rtgs_cr_one_shot_interlaced_once(context, variant=variant)


def compute_sampled_view_metric_stats(
    context: cr_66views.RtgsCr66RenderContext,
    metric_view_indices: Sequence[int],
) -> MetricStats:
    import torch
    from gaussian_renderer import render

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    c2w = context.anchor_c2w
    orbit_center = context.orbit_center
    flat_viewmats = cr_66views.synthesize_flat_viewmats(
        c2w=c2w,
        orbit_center=orbit_center,
        source_view_count=int(context.source_view_count),
        view_degree=float(context.view_degree),
        orbit_direction=int(context.orbit_direction),
        device=str(context.args.device),
    )
    with torch.no_grad():
        for view_id in metric_view_indices:
            synthetic_camera = cr_66views.synthetic_camera_from_viewmat_preserving_rtgs_contract(
                context.cr_anchor_camera_cuda,
                flat_viewmats[int(view_id)],
                uid=int(view_id),
                image_name=f"metric_view_{int(view_id):03d}",
                timestamp=float(context.timestamp),
                device=str(context.args.device),
                width=int(context.render_width),
                height=int(context.render_height),
                crop_to_fill=bool(context.crop_to_fill),
            ).cuda()
            viewmat, K = cr_66views.rtgs_camera_to_gsplat_inputs(synthetic_camera, device=str(context.args.device))
            colors = cr_66views.evaluate_rtgs_colors(
                context.runtime.official.gaussians,
                timestamp=float(context.timestamp),
                camera_center=synthetic_camera.camera_center,
                mask=context.geometry.mask,
            )
            snapshot = cr_66views.snapshot_from_geometry(context.geometry, colors=colors)
            cr_snapshot, cr_K, _ = adapt_snapshot_for_rtgs_compat_projection(
                snapshot,
                camera=synthetic_camera,
                viewmat=viewmat,
                K=K,
                enabled=bool(context.args.rtgs_compat_projection) and not bool(context.fov_normalization["applied"]),
            )
            cr_image = cr_66views.render_rtgs_coherent(
                snapshot=cr_snapshot,
                viewmat=viewmat,
                K=cr_K,
                width=int(context.render_width),
                height=int(context.render_height),
                tile_size=int(context.args.tile_size),
                near_plane=float(context.args.near_plane),
                far_plane=float(context.args.far_plane),
                camera_model=str(context.args.camera_model),
                background=context.runtime.background,
                debug=bool(context.args.debug_cr),
            )
            official_image = (
                render(synthetic_camera, context.runtime.official.gaussians, context.runtime.pipe, context.runtime.background)["render"]
                .detach()
                .clamp(0.0, 1.0)
                .contiguous()
            )
            pair = compute_pair_metrics(cr_image, official_image, ssim_metric=context.runtime.ssim_metric)
            psnr_values.append(float(pair["psnr"]))
            if "ssim" in pair:
                ssim_values.append(float(pair["ssim"]))
    return MetricStats(
        psnr_mean=_mean(psnr_values),
        psnr_std=_std(psnr_values),
        ssim_mean=_mean(ssim_values),
        ssim_std=_std(ssim_values),
        lpips_mean=float("nan"),
        lpips_std=float("nan"),
        metric_view_count=int(len(metric_view_indices)),
    )


def _reference_flat_viewmats(context: cr_66views.RtgsCr66RenderContext):
    return cr_66views.synthesize_flat_viewmats(
        c2w=context.anchor_c2w,
        orbit_center=context.orbit_center,
        source_view_count=int(context.source_view_count),
        view_degree=float(context.view_degree),
        orbit_direction=int(context.orbit_direction),
        device=str(context.args.device),
    )


def render_official_reference_view(
    context: cr_66views.RtgsCr66RenderContext,
    view_id: int,
    *,
    flat_viewmats: Any | None = None,
):
    from gaussian_renderer import render

    if flat_viewmats is None:
        flat_viewmats = _reference_flat_viewmats(context)
    synthetic_camera = cr_66views.synthetic_camera_from_viewmat_preserving_rtgs_contract(
        context.cr_anchor_camera_cuda,
        flat_viewmats[int(view_id)],
        uid=int(view_id),
        image_name=f"reference_view_{int(view_id):03d}",
        timestamp=float(context.timestamp),
        device=str(context.args.device),
        width=int(context.render_width),
        height=int(context.render_height),
        crop_to_fill=bool(context.crop_to_fill),
    ).cuda()
    return (
        render(synthetic_camera, context.runtime.official.gaussians, context.runtime.pipe, context.runtime.background)["render"]
        .detach()
        .clamp(0.0, 1.0)
        .contiguous()
    )


def render_official_reference_interlaced(context: cr_66views.RtgsCr66RenderContext):
    import torch

    flat_viewmats = _reference_flat_viewmats(context)
    interlaced = torch.zeros(
        (3, int(context.panel_height), int(context.panel_width)),
        dtype=torch.float32,
        device=context.viewpoint_index_t.device,
    )
    with torch.no_grad():
        for view_id in range(int(context.source_view_count)):
            image = render_official_reference_view(context, view_id, flat_viewmats=flat_viewmats)
            cr_66views.accumulate_interlaced_view(
                interlaced,
                image,
                context.viewpoint_index_t,
                view_id=view_id,
                viewport=context.viewport,
            )
    return interlaced.clamp(0.0, 1.0).contiguous()


def build_manifest(
    *,
    args: argparse.Namespace,
    context: cr_66views.RtgsCr66RenderContext,
    run_id: str,
    artifact_root: Path,
    variants: Sequence[Any],
    metric_view_indices: Sequence[int],
    reference_interlaced_written: bool,
    last_interlaced: Any,
) -> dict[str, Any]:
    shape = list(last_interlaced.shape) if last_interlaced is not None and hasattr(last_interlaced, "shape") else None
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "rtgs_cr_experiment",
        "engine": str(args.engine),
        "engine_notes": _engine_notes(str(args.engine)),
        "artifact_root": str(artifact_root),
        "scene": context.runtime.official.scene_name,
        "dataset_kind": context.runtime.official.scene_paths.dataset_kind,
        "dataset_path": str(context.runtime.official.scene_paths.dataset_path),
        "source_path": str(context.runtime.official.source_path),
        "config_path": str(context.runtime.official.scene_paths.config_path),
        "model_path": str(Path(args.model_path).expanduser()),
        "checkpoint": str(context.runtime.official.checkpoint_path),
        "checkpoint_step": int(context.runtime.official.iteration),
        "split": str(args.split),
        "camera_index": int(args.camera_index),
        "n3dv_frame_index": int(args.n3dv_frame_index),
        "timestamp": float(context.timestamp),
        "width": int(context.panel_width),
        "height": int(context.panel_height),
        "source_view_count": int(context.source_view_count),
        "render_view_count": int(context.source_view_count),
        "render_image_shape": shape,
        "camera_aspect_mode": str(args.camera_aspect_mode),
        "content_viewport": context.viewport.to_manifest(),
        "view_degree": float(context.view_degree),
        "orbit_direction": int(context.orbit_direction),
        "viewpoint_index": context.view_index_stats,
        "camera_fov_normalization": dict(context.fov_normalization),
        "reference_interlaced_written": bool(reference_interlaced_written),
        "variants": [variant.to_json() for variant in variants],
        "metric_view_indices": [int(index) for index in metric_view_indices],
        "output_prefix": str(args.output_prefix),
        "gaussians_total": int(context.runtime.official.gaussians.get_xyz.shape[0]),
        "gaussians_after_temporal_mask": int(context.geometry.means.shape[0]),
        "active_sh_degree": int(context.runtime.official.gaussians.active_sh_degree),
        "active_sh_degree_t": int(getattr(context.runtime.official.gaussians, "active_sh_degree_t", 0)),
        "rtgs_git_commit": context.runtime.official.rtgs_git_commit or None,
        "args": dict(vars(args)),
    }


def _call_render_fn(
    render_fn: Callable[..., tuple[Any, Any]],
    context: cr_66views.RtgsCr66RenderContext,
    *,
    progress_label: str | None,
) -> tuple[Any, Any]:
    try:
        parameters = inspect.signature(render_fn).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "progress_label" in parameters:
        return render_fn(context, progress_label=progress_label)
    return render_fn(context)


def _render_timing_values(render_timing: Any) -> tuple[float, dict[str, float]]:
    if hasattr(render_timing, "to_metric_dict"):
        detailed = {str(key): float(value) for key, value in render_timing.to_metric_dict().items()}
        return float(detailed.get("frame_ms_with_lkg_interlace", getattr(render_timing, "frame_ms", 0.0))), detailed
    if isinstance(render_timing, Mapping):
        detailed = {str(key): float(value) for key, value in render_timing.items()}
        return float(detailed.get("frame_ms_with_lkg_interlace", detailed.get("frame_ms", 0.0))), detailed
    return float(render_timing), {}


def _normalize_averaged_timing_metrics(values: Mapping[str, float]) -> dict[str, float]:
    normalized = {str(key): float(value) for key, value in values.items()}
    if not normalized:
        return {}
    normalized["rtgs_dynamic_total_ms"] = float(
        normalized.get("dynamic_geometry_ms", 0.0)
        + normalized.get("temporal_opacity_ms", 0.0)
        + normalized.get("snapshot_compaction_ms", 0.0)
        + normalized.get("dynamic_color_ms", 0.0)
    )
    normalized["cr_core_total_ms"] = float(
        normalized.get("cr_projection_ms", 0.0)
        + normalized.get("cr_keygen_ms", 0.0)
        + normalized.get("cr_sort_ms", 0.0)
        + normalized.get("cr_blend_ms", 0.0)
    )
    normalized["lkg_interlace_post_ms"] = float(
        normalized.get("lkg_unpatchify_ms", 0.0)
        + normalized.get("lkg_unpad_ms", 0.0)
        + normalized.get("lkg_panel_paste_ms", 0.0)
    )
    normalized["frame_ms_without_lkg"] = float(normalized["rtgs_dynamic_total_ms"] + normalized["cr_core_total_ms"])
    normalized["fps_without_lkg"] = (
        1000.0 / normalized["frame_ms_without_lkg"]
        if normalized["frame_ms_without_lkg"] > 0.0 and math.isfinite(normalized["frame_ms_without_lkg"])
        else 0.0
    )
    normalized["frame_ms_with_lkg_interlace"] = float(normalized["frame_ms_without_lkg"] + normalized["lkg_interlace_post_ms"])
    normalized["fps_with_lkg_interlace"] = (
        1000.0 / normalized["frame_ms_with_lkg_interlace"]
        if normalized["frame_ms_with_lkg_interlace"] > 0.0 and math.isfinite(normalized["frame_ms_with_lkg_interlace"])
        else 0.0
    )
    return normalized


def _engine_notes(engine: str) -> dict[str, Any]:
    return {
        "cluster_affects_render": True,
        "description": "One-shot grouped CoherentRaster path with RTGS-compatible projection adapter support for sentinel-FoV cameras.",
    }


def _should_report_view_progress(view_id: int, *, source_view_count: int, every: int) -> bool:
    if int(source_view_count) <= 0:
        return False
    if int(view_id) == 0 or int(view_id) == int(source_view_count) - 1:
        return True
    return int(every) > 0 and (int(view_id) + 1) % int(every) == 0


def _reset_cuda_peak_memory() -> None:
    if not _cuda_available():
        return
    import torch

    torch.cuda.reset_peak_memory_stats()


def _peak_vram_gb() -> float:
    if not _cuda_available():
        return 0.0
    import torch

    return float(torch.cuda.max_memory_reserved()) / float(2**30)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan")
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _std(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan")
    return float(np.std(np.asarray(finite, dtype=np.float64)))


def _safe_name(value: Any) -> str:
    import re

    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return sanitized or "value"


def _json_dumps(value: Mapping[str, Any]) -> str:
    import json

    return json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n"
