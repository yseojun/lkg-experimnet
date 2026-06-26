#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[2]
REPO_ROOT = THIS_FILE.parents[3]
CLEAN_ROOT = THIS_FILE.parents[4]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lkg_experiment.coherent_default.coherent_gsplat_bridge import (
    build_cr_lookup_arrays,
    install_gsplat_root,
    load_splats_from_checkpoint,
    lookup_arrays_to_torch,
    resolve_checkpoint_path,
    scale_intrinsics_to_panel,
)
from lkg_experiment.coherent_default.coherent_raster import (
    LKGViewMappingCalibration,
    build_lkg_viewpoint_index,
    build_linear_viewpoint_index,
)
from lkg_experiment.coherent_default.coherent_raster_experiment import (
    ArtifactWriter,
    CoherentRasterRenderer,
    OfficialGsplatRenderer,
    OrbitViewSynthesizer,
    build_experiment_web_assets,
    build_experiment_variants,
    cluster_index_from_view_index,
    compact_viewpoint_index,
    compute_metric_stats,
    image_artifact_path,
    load_viewpoint_index_file,
    parse_cluster_values,
    read_metrics_csv,
    reference_interlace_from_views,
    remove_matching_metric_rows,
    select_metric_view_indices,
    time_interlaced_render,
)
from lkg_experiment.fourdgs.fourdgs_bridge import (
    DEFAULT_4DGS_CODE_ROOT,
    has_4dgs_dynerf_camera,
    load_4dgs_checkpoint,
    load_4dgs_dynerf_camera,
)


def first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.expanduser().exists():
            return candidate
    return candidates[0]


def env_path(*names: str) -> Optional[Path]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return None


def default_data_root() -> Path:
    env = env_path("DATADIR", "DATA_DIR", "CR_DATASETS_ROOT")
    if env is not None:
        return env
    return first_existing_path(
        Path.home() / "Data" / "datasets",
        Path.home() / "data" / "dataset",
        Path.home() / "data" / "datasets",
        Path("/data/dataset"),
        Path("/data/datasets"),
    )


def default_result_root() -> Path:
    env = env_path("RESULTDIR", "RESULT_DIR", "CR_RESULTS_ROOT")
    if env is not None:
        return env
    return first_existing_path(
        Path.home() / "Data" / "results",
        Path.home() / "data" / "result",
        Path.home() / "data" / "results",
        Path("/data/result"),
        Path("/data/results"),
    )


def default_checkpoint_path() -> Path:
    result_root = default_result_root()
    return first_existing_path(
        result_root / "blender_MCMC100000_init50000" / "drums" / "ckpts" / "ckpt_29999_rank0.pt",
        result_root / "blender_MCMC500000" / "drums" / "ckpts" / "ckpt_29999_rank0.pt",
        result_root / "MipNeRF360_MCMC500000" / "garden" / "ckpts" / "ckpt_29999_rank0.pt",
        result_root / "MipNeRF360" / "garden" / "ckpts" / "ckpt_29999_rank0.pt",
    )


def install_bridge_sdk_root(path: Path | str) -> Path:
    root = Path(path).expanduser().resolve()
    src_root = root / "src"
    if not src_root.is_dir():
        raise FileNotFoundError(f"Bridge SDK src directory not found: {src_root}")
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)
    return root


DEFAULT_BRIDGE_SDK_ROOT = CLEAN_ROOT / "Bridge-Python-SDK-Lab"
DEFAULT_GSPLAT_ROOT = CLEAN_ROOT / "gsplat"
DEFAULT_VIEWPOINT_INDEX_PATH = REPO_ROOT / "generated" / "lkg_go_1440x2560_66_views_lkg_calibration.npz"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "generated" / "coherent_raster_experiments"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run no-Bridge CoherentRaster experiments against original gsplat 3DGS renders"
    )
    parser.add_argument("--checkpoint-path", default=str(default_checkpoint_path()))
    parser.add_argument("--iteration", type=int, help="Checkpoint iteration; omitted means latest rank0 checkpoint")
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--gsplat-root", default=str(DEFAULT_GSPLAT_ROOT))
    parser.add_argument("--bridge-sdk-root", default=str(DEFAULT_BRIDGE_SDK_ROOT))
    parser.add_argument("--four-dgs-model-path", help="4DGaussians model directory; overrides --checkpoint-path")
    parser.add_argument("--four-dgs-code-root", default=str(DEFAULT_4DGS_CODE_ROOT))
    parser.add_argument("--four-dgs-iteration", type=int, help="4DGaussians iteration; omitted means latest")
    parser.add_argument("--four-dgs-time", default=0.0, type=float, help="Timestamp used to materialize 4DGS as 3DGS")
    parser.add_argument("--four-dgs-stage", choices=("fine", "coarse"), default="fine")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--run-id", help="Output subdirectory name; default is timestamp plus checkpoint stem")
    parser.add_argument("--output-prefix", default="", help="Prefix for per-camera image filenames inside a run directory")
    parser.add_argument("--append-metrics", action="store_true", help="Append metrics.csv/json in an existing run directory")

    parser.add_argument("--data-dir", default="auto", help="'auto', a dataset path, or empty to force bbox camera")
    parser.add_argument("--camera-source", choices=("auto", "fourdgs", "dataset", "bbox"), default="auto")
    parser.add_argument("--camera-split", choices=("auto", "val", "train", "test"), default="auto")
    parser.add_argument("--camera-index", default=0, type=int)
    parser.add_argument("--width", default=1440, type=int)
    parser.add_argument("--height", default=2560, type=int)
    parser.add_argument("--views", default=66, type=int)
    parser.add_argument("--view-degree", default=53.0, type=float)
    parser.add_argument("--orbit-direction", default=-1, type=int)
    parser.add_argument("--orbit-center-distance", default=0.0, type=float)
    parser.add_argument("--fov-y", default=60.0, type=float, help="Fallback bbox-camera vertical FOV in degrees")
    parser.add_argument("--no-crop-to-fill", action="store_true")

    parser.add_argument("--map-mode", choices=("file", "linear", "lkg"), default="file")
    parser.add_argument("--viewpoint-index-path", default=str(DEFAULT_VIEWPOINT_INDEX_PATH))
    parser.add_argument("--no-compact-view-index", action="store_true")
    parser.add_argument("--coherent-quantize", choices=("floor", "nearest"), default="floor")
    parser.add_argument("--lkg-pitch", type=float)
    parser.add_argument("--lkg-slope", type=float)
    parser.add_argument("--lkg-center", type=float)
    parser.add_argument("--lkg-subp", type=float)
    parser.add_argument("--lkg-inv-view", action="store_true")

    parser.add_argument("--clusters", default="2,4,8,16")
    parser.add_argument("--ablation-cluster", default=8, type=int)
    parser.add_argument("--no-without-remap", action="store_true")
    parser.add_argument("--no-without-reuse", action="store_true")
    parser.add_argument("--tile-size", default=16, type=int)

    parser.add_argument("--near-plane", default=0.01, type=float)
    parser.add_argument("--far-plane", default=1e10, type=float)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--camera-model", choices=("pinhole", "ortho", "fisheye"), default="pinhole")
    parser.add_argument("--coherent-color-mode", choices=("sh", "dc"), default="sh")
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--packed-original", action="store_true")
    parser.add_argument("--rasterize-mode", choices=("classic", "antialiased"), default="classic")
    parser.add_argument("--max-gaussians", default=0, type=int, help="Debug only; 0 keeps the full checkpoint")

    parser.add_argument("--warmup-iters", default=5, type=int)
    parser.add_argument("--measure-iters", default=10, type=int)
    parser.add_argument("--view-chunk-size", default=1, type=int)
    parser.add_argument("--metric-view-stride", default=1, type=int)
    parser.add_argument("--max-metric-views", default=0, type=int, help="0 evaluates every selected view")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--require-lpips", action="store_true")
    parser.add_argument("--no-reference-interlaced", action="store_true")
    parser.add_argument("--skip-web-assets", action="store_true")
    parser.add_argument(
        "--write-mapping-artifacts",
        action="store_true",
        help="Write mappings/raw_mapping.npz and numeric mapping values; disabled by default",
    )
    parser.add_argument(
        "--write-mapping-previews",
        action="store_true",
        help="Write colorized mapping PNG previews; implies --write-mapping-artifacts",
    )
    parser.add_argument("--debug-render", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CoherentRaster experiment rendering")
    device = "cuda"

    bridge_sdk_root = install_bridge_sdk_root(args.bridge_sdk_root)
    gsplat_root = install_gsplat_root(args.gsplat_root)
    if args.four_dgs_model_path:
        checkpoint = Path(args.four_dgs_model_path).expanduser()
        result_root = checkpoint
        cfg: dict[str, Any] = {}
    else:
        checkpoint = resolve_checkpoint_path(args.checkpoint_path, iteration=args.iteration, rank=args.rank)
        result_root = result_root_from_checkpoint(checkpoint)
        cfg = load_cfg(result_root)
    width, height, source_view_count = int(args.width), int(args.height), int(args.views)

    viewpoint_index, file_view_count, viewpoint_metadata = build_viewpoint_index(args, width, height, source_view_count)
    if file_view_count is not None and int(file_view_count) != source_view_count:
        raise ValueError(f"viewpoint index view_count={file_view_count} but --views={source_view_count}")

    source_viewpoint_index = viewpoint_index.copy()
    view_labels = np.arange(source_view_count, dtype=np.int32)
    render_view_count = source_view_count
    if args.map_mode == "file" and not args.no_compact_view_index:
        viewpoint_index, view_labels = compact_viewpoint_index(viewpoint_index, source_view_count)
        render_view_count = int(view_labels.size)

    run_id = args.run_id or make_run_id(checkpoint)
    artifact_root = Path(args.artifact_dir).expanduser() / run_id
    writer = ArtifactWriter(artifact_root)

    print(f"Loading checkpoint: {checkpoint}", file=sys.stderr, flush=True)
    four_dgs = None
    if args.four_dgs_model_path:
        four_dgs = load_4dgs_checkpoint(
            args.four_dgs_model_path,
            code_root=args.four_dgs_code_root,
            iteration=args.four_dgs_iteration,
            device=device,
        )
        splats = four_dgs.splats_at(args.four_dgs_time, stage=args.four_dgs_stage)
        step = int(four_dgs.iteration)
        cfg["sh_degree"] = int(four_dgs.sh_degree)
    else:
        splats, step = load_splats_from_checkpoint(checkpoint, device=device)
    splats = subset_splats_for_debug(splats, args.max_gaussians)

    data_dir: Optional[Path] = None
    if four_dgs is not None and (
        args.camera_source == "fourdgs"
        or (args.camera_source == "auto" and has_4dgs_dynerf_camera(four_dgs.cfg_args))
    ):
        c2w_np, K_np, scene_center_np, camera_label = load_4dgs_dynerf_camera(
            four_dgs.cfg_args,
            args,
            width,
            height,
        )
        data_dir = Path(str(four_dgs.cfg_args.source_path)).expanduser()
        dataset_category = "fourdgs-dynerf"
    elif args.camera_source == "fourdgs":
        raise RuntimeError("--camera-source fourdgs requires a 4DGS model with dynerf poses_bounds.npy")
    else:
        data_dir = maybe_resolve_data_dir(args, cfg)
        dataset_category = dataset_category_from_cfg(cfg, data_dir)
        if args.camera_source == "dataset" and data_dir is None:
            raise RuntimeError("--camera-source dataset requested but no valid data_dir is available")
    if data_dir is not None and dataset_category != "fourdgs-dynerf" and args.camera_source in {"auto", "dataset"}:
        if dataset_category == "blender":
            c2w_np, K_np, scene_center_np, camera_label = load_blender_camera(args, cfg, data_dir, width, height)
        elif dataset_category == "colmap":
            c2w_np, K_np, scene_center_np, camera_label = load_colmap_camera(args, cfg, data_dir, width, height)
        else:
            raise ValueError(f"Unsupported dataset_category: {dataset_category!r}")
    elif args.camera_source == "bbox" or data_dir is None:
        c2w_np, K_np, scene_center_np, camera_label = estimate_bbox_camera(args, splats["means"], width, height)
        dataset_category = "bbox"

    K = torch.from_numpy(K_np.astype(np.float32)).to(device=device, non_blocking=True).contiguous()
    c2w = torch.from_numpy(c2w_np.astype(np.float32)).to(device=device, non_blocking=True).contiguous()
    initial_distance = float(np.linalg.norm(scene_center_np.astype(np.float64) - c2w_np[:3, 3].astype(np.float64)))
    orbit_center_distance = float(args.orbit_center_distance or max(initial_distance, 1e-4))
    orbit_center = c2w[:3, 3] + c2w[:3, 2] * orbit_center_distance

    use_white_background = bool(
        args.white_background
        or dataset_category == "blender"
        or (four_dgs is not None and getattr(four_dgs.cfg_args, "white_background", False))
    )
    background = torch.ones(3, device=device) if use_white_background else None
    original_renderer = OfficialGsplatRenderer(
        splats,
        sh_degree=int(cfg.get("sh_degree", args.sh_degree)),
        near_plane=float(cfg.get("near_plane", args.near_plane)),
        far_plane=float(cfg.get("far_plane", args.far_plane)),
        camera_model=str(cfg.get("camera_model", args.camera_model)),
        background=background,
        color_mode=args.coherent_color_mode,
        packed=args.packed_original,
        rasterize_mode=args.rasterize_mode,
    )
    coherent_renderer = CoherentRasterRenderer(
        splats,
        sh_degree=int(cfg.get("sh_degree", args.sh_degree)),
        near_plane=float(cfg.get("near_plane", args.near_plane)),
        far_plane=float(cfg.get("far_plane", args.far_plane)),
        camera_model=str(cfg.get("camera_model", args.camera_model)),
        background=background,
        color_mode=args.coherent_color_mode,
    )

    variants = build_experiment_variants(
        parse_cluster_values(args.clusters),
        ablation_cluster=args.ablation_cluster,
        include_without_remap=not args.no_without_remap,
        include_without_reuse=not args.no_without_reuse,
    )
    metric_view_indices = select_metric_view_indices(
        render_view_count,
        max_metric_views=args.max_metric_views,
        stride=args.metric_view_stride,
    )

    base_flat_viewmats = synthesize_flat_viewmats(
        c2w=c2w,
        orbit_center=orbit_center,
        view_labels=view_labels,
        source_view_count=source_view_count,
        view_degree=args.view_degree,
        orbit_direction=args.orbit_direction,
        device=device,
    )
    reference_interlaced = None
    if not args.no_reference_interlaced:
        print("Rendering original gsplat reference interlaced image...", file=sys.stderr, flush=True)
        reference_interlaced = render_reference_interlaced_chunked(
            original_renderer,
            flat_viewmats=base_flat_viewmats,
            K=K,
            viewpoint_index=viewpoint_index,
            width=width,
            height=height,
            tile_size=args.tile_size,
            chunk_size=args.view_chunk_size,
        )

    cluster_indices = {
        variant.name: cluster_index_from_view_index(viewpoint_index, variant.cluster_size)
        for variant in variants
    }
    if args.write_mapping_artifacts or args.write_mapping_previews:
        writer.write_mapping_npz(
            viewpoint_index,
            cluster_indices,
            view_labels=view_labels,
            source_view_index_hwc=source_viewpoint_index,
        )
        writer.save_mapping_values("view_index", viewpoint_index)
        if not np.array_equal(source_viewpoint_index, viewpoint_index):
            writer.save_mapping_values("source_view_index", source_viewpoint_index)
        if args.write_mapping_previews:
            writer.save_index_previews("view_index", viewpoint_index)
            if not np.array_equal(source_viewpoint_index, viewpoint_index):
                writer.save_index_previews("source_view_index", source_viewpoint_index)
            for variant in variants:
                writer.save_index_previews(f"{variant.name}_cluster_index", cluster_indices[variant.name])

    if args.append_metrics:
        rows: list[dict[str, Any]] = remove_matching_metric_rows(
            read_metrics_csv(artifact_root / "metrics.csv"),
            output_prefix=args.output_prefix,
            camera_split=args.camera_split,
            camera_index=args.camera_index,
        )
    else:
        rows = []
    for variant in variants:
        print(
            f"Running variant {variant.name}: cluster={variant.cluster_size}, "
            f"remap={int(variant.use_remapping)}, reuse={int(variant.reuse_enabled)}",
            file=sys.stderr,
            flush=True,
        )
        lookup = build_cr_lookup_arrays(
            viewpoint_index,
            tile_size=args.tile_size,
            use_remapping=variant.use_remapping,
        )
        view_idx_matrix, subpixel_coord_matrix = lookup_arrays_to_torch(lookup, device=device)
        synthesizer = OrbitViewSynthesizer(
            view_labels=view_labels,
            source_view_count=source_view_count,
            cluster_size=variant.cluster_size,
            view_degree=args.view_degree,
            orbit_direction=args.orbit_direction,
            device=device,
        )
        adjacent_viewmats = synthesizer(c2w, orbit_center)

        interlaced, timing = time_interlaced_render(
            coherent_renderer,
            adjacent_viewmats=adjacent_viewmats,
            K=K,
            view_idx_matrix=view_idx_matrix,
            subpixel_coord_matrix=subpixel_coord_matrix,
            width=width,
            height=height,
            tile_size=args.tile_size,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            debug=args.debug_render,
        )
        writer.save_tensor_image(
            image_artifact_path(variant.name, "looking_glass_tensor.png", output_prefix=args.output_prefix),
            interlaced,
        )
        if reference_interlaced is not None:
            writer.save_tensor_image(
                image_artifact_path(variant.name, "reference_interlaced.png", output_prefix=args.output_prefix),
                reference_interlaced,
            )
            writer.save_tensor_image(
                image_artifact_path(variant.name, "abs_error.png", output_prefix=args.output_prefix),
                (interlaced - reference_interlaced).abs().mul(8.0).clamp(0.0, 1.0),
            )

        metrics = None
        if not args.skip_metrics:
            metrics = compute_metric_stats(
                original_renderer,
                coherent_renderer,
                flat_viewmats=base_flat_viewmats,
                adjacent_viewmats=adjacent_viewmats,
                K=K,
                subpixel_coord_matrix=subpixel_coord_matrix,
                view_idx_template=view_idx_matrix,
                metric_view_indices=metric_view_indices,
                width=width,
                height=height,
                tile_size=args.tile_size,
                require_lpips=args.require_lpips,
            )

        row = {
            "camera_split": args.camera_split,
            "camera_index": args.camera_index,
            "output_prefix": args.output_prefix,
            "variant": variant.name,
            "group": variant.group,
            "cluster_size": variant.cluster_size,
            "use_remapping": variant.use_remapping,
            "reuse_enabled": variant.reuse_enabled,
            "fps": timing.fps,
            "frame_ms": timing.frame_ms,
            "peak_vram_gb": timing.peak_vram_gb,
        }
        if metrics is not None:
            row.update(metrics.__dict__)
        rows.append(row)
        writer.write_metrics_csv(rows)
        writer.write_metrics_json(rows)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "result_root": str(result_root),
        "four_dgs": four_dgs_manifest(args) if args.four_dgs_model_path else None,
        "gsplat_root": str(gsplat_root),
        "bridge_sdk_root": str(bridge_sdk_root),
        "width": width,
        "height": height,
        "source_view_count": source_view_count,
        "render_view_count": render_view_count,
        "view_labels": view_labels,
        "view_degree": args.view_degree,
        "orbit_direction": args.orbit_direction,
        "orbit_center_distance": orbit_center_distance,
        "viewpoint_index": viewpoint_metadata,
        "dataset_category": dataset_category,
        "camera": camera_label,
        "background": "white" if use_white_background else "none",
        "variants": [variant.to_json() for variant in variants],
        "metric_view_indices": metric_view_indices,
        "output_prefix": args.output_prefix,
        "artifact_root": str(artifact_root),
        "git": git_summary(REPO_ROOT),
        "args": vars(args),
    }
    writer.write_manifest(manifest)
    writer.write_metrics_csv(rows)
    writer.write_metrics_json(rows)
    if not args.skip_web_assets:
        build_experiment_web_assets(args.artifact_dir, regenerate_previews=False)
    print(f"Experiment artifacts written to {artifact_root}", file=sys.stderr, flush=True)


def validate_args(args) -> None:
    if args.width <= 0 or args.height <= 0 or args.views <= 0:
        raise ValueError("--width, --height, and --views must be positive")
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive")
    if args.ablation_cluster <= 0:
        raise ValueError("--ablation-cluster must be positive")
    if args.warmup_iters < 0 or args.measure_iters <= 0:
        raise ValueError("--warmup-iters must be non-negative and --measure-iters must be positive")
    if args.view_chunk_size <= 0:
        raise ValueError("--view-chunk-size must be positive")
    if args.metric_view_stride <= 0:
        raise ValueError("--metric-view-stride must be positive")
    if args.max_metric_views < 0:
        raise ValueError("--max-metric-views must be non-negative")
    if args.max_gaussians < 0:
        raise ValueError("--max-gaussians must be non-negative")
    if args.four_dgs_time < 0.0:
        raise ValueError("--four-dgs-time must be non-negative")
    parse_cluster_values(args.clusters)


def four_dgs_manifest(args) -> dict[str, Any]:
    return {
        "model_path": args.four_dgs_model_path,
        "code_root": args.four_dgs_code_root,
        "iteration": args.four_dgs_iteration,
        "time": args.four_dgs_time,
        "stage": args.four_dgs_stage,
    }


def build_viewpoint_index(args, width: int, height: int, view_count: int) -> tuple[np.ndarray, Optional[int], dict[str, Any]]:
    if args.map_mode == "file":
        if not args.viewpoint_index_path:
            raise ValueError("--map-mode file requires --viewpoint-index-path")
        viewpoint_index, file_view_count, metadata = load_viewpoint_index_file(args.viewpoint_index_path, width, height)
        metadata["mode"] = "file"
        return viewpoint_index, file_view_count, metadata
    if args.map_mode == "linear":
        viewpoint_index = build_linear_viewpoint_index(width, height, view_count, quantize=args.coherent_quantize)
        return viewpoint_index, view_count, {"mode": "linear", "quantize": args.coherent_quantize}

    required = [args.lkg_pitch, args.lkg_slope, args.lkg_center, args.lkg_subp]
    if any(value is None for value in required):
        raise ValueError("--map-mode lkg requires --lkg-pitch --lkg-slope --lkg-center --lkg-subp")
    calibration = LKGViewMappingCalibration(
        pitch=float(args.lkg_pitch),
        slope=float(args.lkg_slope),
        center=float(args.lkg_center),
        subp=float(args.lkg_subp),
        inv_view=bool(args.lkg_inv_view),
    )
    viewpoint_index = build_lkg_viewpoint_index(width, height, view_count, calibration, quantize=args.coherent_quantize)
    return viewpoint_index, view_count, {
        "mode": "lkg",
        "quantize": args.coherent_quantize,
        "calibration": calibration.__dict__,
    }


def load_cfg(result_root: Path) -> dict[str, Any]:
    cfg_path = result_root / "cfg.yml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml
    except Exception:
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.unsafe_load(f) or {}


def result_root_from_checkpoint(ckpt: Path) -> Path:
    return ckpt.parent.parent if ckpt.parent.name == "ckpts" else ckpt.parent


def maybe_resolve_data_dir(args, cfg: dict[str, Any]) -> Optional[Path]:
    if args.data_dir == "":
        return None
    if args.data_dir != "auto":
        path = Path(args.data_dir).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"--data-dir does not exist: {path}")
        return path
    cfg_data = cfg.get("data_dir")
    if cfg_data:
        path = Path(str(cfg_data)).expanduser()
        if path.exists():
            return path
        rebased = rebase_cfg_data_dir(path, cfg)
        if rebased is not None and rebased.exists():
            print(f"Using DATADIR-relative data_dir instead of missing cfg path: {rebased}", file=sys.stderr, flush=True)
            return rebased
        print(f"WARNING: cfg data_dir does not exist, using bbox camera fallback: {path}", file=sys.stderr, flush=True)
    return None


def rebase_cfg_data_dir(path: Path, cfg: dict[str, Any]) -> Optional[Path]:
    data_root = default_data_root()
    scene = path.name
    parts = set(path.parts)
    category = str(cfg.get("dataset_category", "") or "").strip().lower()
    if "nerf_synthetic" in parts or category == "blender":
        return data_root / "nerf_synthetic" / scene
    if "MipNeRF_360" in parts or "360_v2" in parts or category == "colmap":
        return data_root / "MipNeRF_360" / scene
    return None


def dataset_category_from_cfg(cfg: dict[str, Any], data_dir: Optional[Path]) -> str:
    category = str(cfg.get("dataset_category", "") or "").strip().lower()
    if category:
        return category
    if data_dir is not None and (data_dir / "transforms_train.json").exists():
        return "blender"
    return "colmap"


def load_colmap_metadata(data_dir: Path, factor: int, normalize: bool, test_every: int):
    from examples.datasets.normalize import (
        align_principal_axes,
        similarity_from_cameras,
        transform_cameras,
        transform_points,
    )
    from pycolmap import SceneManager

    colmap_dir = data_dir / "sparse" / "0"
    if not colmap_dir.exists():
        colmap_dir = data_dir / "sparse"
    if not colmap_dir.exists():
        raise FileNotFoundError(f"COLMAP directory does not exist: {colmap_dir}")

    manager = SceneManager(str(colmap_dir))
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()

    w2c_mats = []
    camera_ids = []
    Ks_dict = {}
    imsize_dict = {}
    bottom = np.array([0, 0, 0, 1], dtype=np.float64).reshape(1, 4)
    for image_id in manager.images:
        image = manager.images[image_id]
        rot = image.R()
        trans = image.tvec.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([rot, trans], axis=1), bottom], axis=0)
        w2c_mats.append(w2c)

        camera_id = image.camera_id
        camera_ids.append(camera_id)
        cam = manager.cameras[camera_id]
        K = np.array([[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        K[:2, :] /= float(factor)
        Ks_dict[camera_id] = K
        imsize_dict[camera_id] = (int(cam.width) // int(factor), int(cam.height) // int(factor))

    if len(w2c_mats) == 0:
        raise ValueError("No images found in COLMAP metadata.")

    camtoworlds = np.linalg.inv(np.stack(w2c_mats, axis=0))
    image_names = [manager.images[image_id].name for image_id in manager.images]
    order = np.argsort(image_names)
    image_names = [image_names[i] for i in order]
    camtoworlds = camtoworlds[order]
    camera_ids = [camera_ids[i] for i in order]

    points = manager.points3D.astype(np.float32)
    if normalize:
        T1 = similarity_from_cameras(camtoworlds)
        camtoworlds = transform_cameras(T1, camtoworlds)
        points = transform_points(T1, points)
        T2 = align_principal_axes(points)
        camtoworlds = transform_cameras(T2, camtoworlds)
        points = transform_points(T2, points)
        if np.median(points[:, 2]) > np.mean(points[:, 2]):
            T3 = np.array(
                [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            camtoworlds = transform_cameras(T3, camtoworlds)

    indices = np.arange(len(image_names))
    split_indices = {
        "train": indices[indices % int(test_every) != 0],
        "val": indices[indices % int(test_every) == 0],
    }
    return image_names, camtoworlds.astype(np.float32), camera_ids, Ks_dict, imsize_dict, split_indices


def load_colmap_camera(args, cfg: dict[str, Any], data_dir: Path, width: int, height: int):
    from coherent_raster.utils.utils_coherent_raster import calculate_scene_center
    import torch

    split = "val" if args.camera_split == "auto" else args.camera_split
    if split == "test":
        raise ValueError("COLMAP datasets support --camera-split auto, val, or train; test is Blender-only")
    image_names, camtoworlds, camera_ids, Ks_dict, imsize_dict, split_indices = load_colmap_metadata(
        data_dir=data_dir,
        factor=int(cfg.get("data_factor", 1)),
        normalize=bool(cfg.get("normalize_world_space", True)),
        test_every=int(cfg.get("test_every", 8)),
    )
    selected_indices = split_indices[split]
    if args.camera_index < 0 or args.camera_index >= len(selected_indices):
        raise IndexError(f"--camera-index {args.camera_index} outside {split} split count {len(selected_indices)}")
    index = int(selected_indices[args.camera_index])
    camera_id = camera_ids[index]
    image_width, image_height = imsize_dict[camera_id]
    K_scaled = scale_intrinsics_to_panel(
        Ks_dict[camera_id],
        original_height=image_height,
        original_width=image_width,
        target_height=height,
        target_width=width,
        crop_to_fill=not args.no_crop_to_fill,
    )
    c2w = camtoworlds[index].astype(np.float32)
    w2c_all = torch.linalg.inv(torch.from_numpy(camtoworlds).float())
    scene_center = calculate_scene_center(w2c_all).cpu().numpy().astype(np.float32)
    return c2w, K_scaled, scene_center, f"colmap:{data_dir}:{split}[{args.camera_index}]/{image_names[index]}"


def load_blender_camera(args, cfg: dict[str, Any], data_dir: Path, width: int, height: int):
    from coherent_raster.utils.blender_dataset import BlenderDataset, BlenderParser
    from coherent_raster.utils.utils_coherent_raster import calculate_scene_center
    import torch

    split = "test" if args.camera_split == "auto" else args.camera_split
    parser = BlenderParser(
        data_dir=str(data_dir),
        factor=int(cfg.get("data_factor", 1)),
        normalize=bool(cfg.get("normalize_world_space", False)),
    )
    dataset = BlenderDataset(parser, split=split, white_background=True, load_depths=False)
    if len(dataset) == 0:
        raise RuntimeError(f"Blender split {split!r} is empty for dataset: {data_dir}")
    if args.camera_index < 0 or args.camera_index >= len(dataset):
        raise IndexError(f"--camera-index {args.camera_index} outside {split} split count {len(dataset)}")
    data = dataset[args.camera_index]
    image = data["image"]
    image_height, image_width = int(image.shape[0]), int(image.shape[1])
    K_scaled = scale_intrinsics_to_panel(
        data["K"].cpu().numpy(),
        original_height=image_height,
        original_width=image_width,
        target_height=height,
        target_width=width,
        crop_to_fill=not args.no_crop_to_fill,
    )
    c2w = data["camtoworld"].cpu().numpy().astype(np.float32)
    split_indices = np.asarray(parser.split_indices[split], dtype=np.int64)
    split_c2w = torch.from_numpy(parser.camtoworlds[split_indices]).float()
    scene_center = calculate_scene_center(torch.linalg.inv(split_c2w)).cpu().numpy().astype(np.float32)
    return c2w, K_scaled, scene_center, f"blender:{data_dir}:{split}[{args.camera_index}]"


def estimate_bbox_camera(args, means, width: int, height: int):
    center = means.mean(dim=0)
    std_radius = means.std(dim=0).norm().clamp_min(0.1)
    distance = std_radius * 3.0
    center_np = center.detach().cpu().numpy().astype(np.float32)
    position = center_np + np.array([0.0, 0.0, -float(distance.item())], dtype=np.float32)
    c2w = look_at_c2w(position, center_np)
    fovy = math.radians(float(args.fov_y))
    fy = (float(height) * 0.5) / math.tan(fovy * 0.5)
    K = np.array([[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return c2w, K, center_np, "bbox:auto"


def look_at_c2w(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target.astype(np.float64) - position.astype(np.float64)
    forward /= max(np.linalg.norm(forward), 1e-8)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    right /= max(np.linalg.norm(right), 1e-8)
    up = np.cross(forward, right)
    up /= max(np.linalg.norm(up), 1e-8)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.stack([right, up, forward], axis=1).astype(np.float32)
    c2w[:3, 3] = position.astype(np.float32)
    return c2w


def synthesize_flat_viewmats(
    *,
    c2w,
    orbit_center,
    view_labels: np.ndarray,
    source_view_count: int,
    view_degree: float,
    orbit_direction: int,
    device: str,
):
    synthesizer = OrbitViewSynthesizer(
        view_labels=view_labels,
        source_view_count=source_view_count,
        cluster_size=1,
        view_degree=view_degree,
        orbit_direction=orbit_direction,
        device=device,
    )
    return synthesizer(c2w, orbit_center).reshape(-1, 4, 4)[: int(view_labels.size)].contiguous()


def render_reference_interlaced_chunked(
    original_renderer: OfficialGsplatRenderer,
    *,
    flat_viewmats,
    K,
    viewpoint_index: np.ndarray,
    width: int,
    height: int,
    tile_size: int,
    chunk_size: int,
):
    import torch

    output = torch.empty((3, int(height), int(width)), dtype=torch.float32, device=K.device)
    view_count = int(flat_viewmats.shape[0])
    if chunk_size >= view_count:
        rendered_views = original_renderer.render_views(
            viewmats=flat_viewmats,
            K=K,
            width=width,
            height=height,
            tile_size=tile_size,
        )
        return reference_interlace_from_views(rendered_views, viewpoint_index)

    view_index = torch.from_numpy(viewpoint_index.astype(np.int64, copy=False)).to(device=K.device, non_blocking=True)
    for start in range(0, view_count, int(chunk_size)):
        end = min(start + int(chunk_size), view_count)
        rendered = original_renderer.render_views(
            viewmats=flat_viewmats[start:end],
            K=K,
            width=width,
            height=height,
            tile_size=tile_size,
        )
        for channel in range(3):
            ids = view_index[:, :, channel]
            mask = (ids >= start) & (ids < end)
            if not bool(mask.any()):
                continue
            y, x = torch.nonzero(mask, as_tuple=True)
            local = ids[y, x] - start
            output[channel, y, x] = rendered[local, channel, y, x]
    return output


def subset_splats_for_debug(splats: dict[str, Any], max_gaussians: int) -> dict[str, Any]:
    if max_gaussians <= 0:
        return splats
    current = int(splats["means"].shape[0])
    keep = min(int(max_gaussians), current)
    print(f"WARNING: using only first {keep}/{current} Gaussians for debug", file=sys.stderr, flush=True)
    return {key: value[:keep].contiguous() for key, value in splats.items()}


def make_run_id(checkpoint: Path) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = result_root_from_checkpoint(checkpoint)
    return f"{stamp}_{root.name}"


def git_summary(root: Path) -> dict[str, Any]:
    def run_git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=str(root),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return ""

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(run_git("status", "--short")),
    }


if __name__ == "__main__":
    main()
