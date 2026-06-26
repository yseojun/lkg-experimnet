#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[2]
REPO_ROOT = THIS_FILE.parents[3]
CLEAN_ROOT = THIS_FILE.parents[4]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lkg_experiment.coherent_default.coherent_gsplat_bridge import install_gsplat_root
from lkg_experiment.coherent_default.coherent_raster_experiment import OfficialGsplatRenderer, tensor_to_hwc_uint8
from lkg_experiment.fourdgs.fourdgs_bridge import (
    DEFAULT_4DGS_CODE_ROOT,
    has_4dgs_dynerf_camera,
    load_4dgs_checkpoint,
    load_4dgs_dynerf_camera,
)


DEFAULT_GSPLAT_ROOT = CLEAN_ROOT / "gsplat"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "4dgs-single-view"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render one 4DGS timestamp as a single gsplat view PNG")
    parser.add_argument("--four-dgs-model-path", required=True, help="4DGaussians model directory")
    parser.add_argument("--four-dgs-code-root", default=str(DEFAULT_4DGS_CODE_ROOT))
    parser.add_argument("--four-dgs-iteration", type=int, help="4DGaussians iteration; omitted means latest")
    parser.add_argument("--four-dgs-time", default=0.0, type=float, help="Timestamp used to materialize 4DGS as 3DGS")
    parser.add_argument("--four-dgs-stage", choices=("fine", "coarse"), default="fine")
    parser.add_argument("--gsplat-root", default=str(DEFAULT_GSPLAT_ROOT))
    parser.add_argument("--data-dir", default="auto", help="'auto', a dataset path, or empty to force bbox camera")
    parser.add_argument("--camera-source", choices=("auto", "fourdgs", "dataset", "bbox"), default="fourdgs")
    parser.add_argument("--camera-split", choices=("auto", "val", "train", "test"), default="auto")
    parser.add_argument("--camera-index", default=0, type=int)
    parser.add_argument("--width", default=1352, type=int)
    parser.add_argument("--height", default=1014, type=int)
    parser.add_argument("--output", help="Output PNG path; default writes under experiment/artifacts/4dgs-single-view")
    parser.add_argument("--tile-size", default=16, type=int)
    parser.add_argument("--near-plane", default=0.01, type=float)
    parser.add_argument("--far-plane", default=1e10, type=float)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--camera-model", choices=("pinhole", "ortho", "fisheye"), default="pinhole")
    parser.add_argument("--color-mode", choices=("sh", "dc"), default="sh")
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--rasterize-mode", choices=("classic", "antialiased"), default="classic")
    parser.add_argument("--max-gaussians", default=0, type=int, help="Debug only; 0 keeps the full checkpoint")
    parser.add_argument("--fov-y", default=60.0, type=float, help="Fallback bbox-camera vertical FOV in degrees")
    parser.add_argument("--no-crop-to-fill", action="store_true")
    return parser


def default_output_path(
    model_path: Path | str,
    *,
    iteration: Optional[int],
    time_value: float,
    camera_split: str,
    camera_index: int,
) -> Path:
    model = Path(model_path).expanduser()
    iteration_label = "latest" if iteration is None else str(int(iteration))
    split = camera_split if camera_split != "auto" else "val"
    filename = f"{model.name}_iter{iteration_label}_t{float(time_value):.6f}_{split}_{int(camera_index)}.png"
    return DEFAULT_OUTPUT_DIR / filename


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive")
    if args.max_gaussians < 0:
        raise ValueError("--max-gaussians must be non-negative")
    if args.four_dgs_time < 0.0:
        raise ValueError("--four-dgs-time must be non-negative")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for single-view 4DGS rendering")
    device = "cuda"

    install_gsplat_root(args.gsplat_root)
    cfg: dict[str, Any] = {}

    print(f"Loading 4DGS checkpoint: {args.four_dgs_model_path}", file=sys.stderr, flush=True)
    four_dgs = load_4dgs_checkpoint(
        args.four_dgs_model_path,
        code_root=args.four_dgs_code_root,
        iteration=args.four_dgs_iteration,
        device=device,
    )
    splats = four_dgs.splats_at(args.four_dgs_time, stage=args.four_dgs_stage)
    splats = subset_splats_for_debug(splats, args.max_gaussians)
    cfg["sh_degree"] = int(four_dgs.sh_degree)

    width, height = int(args.width), int(args.height)
    data_dir: Optional[Path] = None
    if args.camera_source == "fourdgs" or (
        args.camera_source == "auto" and has_4dgs_dynerf_camera(four_dgs.cfg_args)
    ):
        c2w_np, K_np, scene_center_np, camera_label = load_4dgs_dynerf_camera(
            four_dgs.cfg_args,
            args,
            width,
            height,
        )
        data_dir = Path(str(four_dgs.cfg_args.source_path)).expanduser()
        dataset_category = "fourdgs-dynerf"
    else:
        data_dir = maybe_resolve_data_dir(args, cfg)
        dataset_category = dataset_category_from_cfg(cfg, data_dir)
        if args.camera_source == "dataset" and data_dir is None:
            raise RuntimeError("--camera-source dataset requested but no valid data_dir is available")
    if args.camera_source not in {"fourdgs"} and data_dir is not None and dataset_category != "fourdgs-dynerf" and args.camera_source in {"auto", "dataset"}:
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
    viewmats = torch.linalg.inv(c2w).unsqueeze(0).contiguous()
    background = (
        torch.ones(3, device=device)
        if bool(args.white_background or dataset_category == "blender" or getattr(four_dgs.cfg_args, "white_background", False))
        else None
    )

    renderer = OfficialGsplatRenderer(
        splats,
        sh_degree=int(cfg.get("sh_degree", args.sh_degree)),
        near_plane=float(args.near_plane),
        far_plane=float(args.far_plane),
        camera_model=str(args.camera_model),
        background=background,
        color_mode=args.color_mode,
        packed=args.packed,
        rasterize_mode=args.rasterize_mode,
    )

    output = Path(args.output).expanduser() if args.output else default_output_path(
        args.four_dgs_model_path,
        iteration=four_dgs.iteration,
        time_value=args.four_dgs_time,
        camera_split=args.camera_split,
        camera_index=args.camera_index,
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        rendered = renderer.render_views(viewmats=viewmats, K=K, width=width, height=height, tile_size=args.tile_size)[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    image = tensor_to_hwc_uint8(rendered)
    Image.fromarray(image, mode="RGB").save(output)

    manifest = {
        "output": str(output),
        "model_path": str(Path(args.four_dgs_model_path).expanduser()),
        "code_root": str(Path(args.four_dgs_code_root).expanduser()),
        "iteration": int(four_dgs.iteration),
        "time": float(args.four_dgs_time),
        "stage": args.four_dgs_stage,
        "gaussians": int(splats["means"].shape[0]),
        "sh_degree": int(cfg.get("sh_degree", args.sh_degree)),
        "width": width,
        "height": height,
        "data_dir": None if data_dir is None else str(data_dir),
        "dataset_category": dataset_category,
        "camera": camera_label,
        "scene_center": np.asarray(scene_center_np).astype(float).tolist(),
        "render_ms": elapsed * 1000.0,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"Wrote {output} ({width}x{height}, gs={manifest['gaussians']}, camera={camera_label}, "
        f"render_ms={manifest['render_ms']:.2f})",
        file=sys.stderr,
        flush=True,
    )


def subset_splats_for_debug(splats: dict[str, Any], max_gaussians: int) -> dict[str, Any]:
    if int(max_gaussians) <= 0:
        return splats
    count = min(int(max_gaussians), int(splats["means"].shape[0]))
    return {key: value[:count].contiguous() if hasattr(value, "__getitem__") else value for key, value in splats.items()}


def maybe_resolve_data_dir(args: argparse.Namespace, cfg: dict[str, Any]) -> Optional[Path]:
    from lkg_experiment.coherent_default.run_coherent_raster_experiment import maybe_resolve_data_dir as _maybe_resolve_data_dir

    return _maybe_resolve_data_dir(args, cfg)


def dataset_category_from_cfg(cfg: dict[str, Any], data_dir: Optional[Path]) -> str:
    from lkg_experiment.coherent_default.run_coherent_raster_experiment import dataset_category_from_cfg as _dataset_category_from_cfg

    return _dataset_category_from_cfg(cfg, data_dir)


def load_colmap_camera(args: argparse.Namespace, cfg: dict[str, Any], data_dir: Path, width: int, height: int):
    from lkg_experiment.coherent_default.run_coherent_raster_experiment import load_colmap_camera as _load_colmap_camera

    return _load_colmap_camera(args, cfg, data_dir, width, height)


def load_blender_camera(args: argparse.Namespace, cfg: dict[str, Any], data_dir: Path, width: int, height: int):
    from lkg_experiment.coherent_default.run_coherent_raster_experiment import load_blender_camera as _load_blender_camera

    return _load_blender_camera(args, cfg, data_dir, width, height)


def estimate_bbox_camera(args: argparse.Namespace, means: Any, width: int, height: int):
    from lkg_experiment.coherent_default.run_coherent_raster_experiment import estimate_bbox_camera as _estimate_bbox_camera

    return _estimate_bbox_camera(args, means, width, height)


if __name__ == "__main__":
    main()
