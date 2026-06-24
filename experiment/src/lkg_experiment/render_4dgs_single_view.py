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
SRC_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]
CLEAN_ROOT = REPO_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lkg_experiment.coherent_gsplat_bridge import install_gsplat_root, scale_intrinsics_to_panel
from lkg_experiment.coherent_raster_experiment import OfficialGsplatRenderer, tensor_to_hwc_uint8
from lkg_experiment.fourdgs_bridge import DEFAULT_4DGS_CODE_ROOT, load_4dgs_checkpoint


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


def has_4dgs_dynerf_camera(cfg_args: Any) -> bool:
    source_path = getattr(cfg_args, "source_path", None)
    if not source_path:
        return False
    return (Path(str(source_path)).expanduser() / "poses_bounds.npy").is_file()


def load_4dgs_dynerf_camera(
    cfg_args: Any,
    args: argparse.Namespace,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    source_path_value = getattr(cfg_args, "source_path", None)
    if not source_path_value:
        raise ValueError("4DGS cfg_args has no source_path for dynerf camera loading")
    source_path = Path(str(source_path_value)).expanduser()
    poses_path = source_path / "poses_bounds.npy"
    if not poses_path.is_file():
        raise FileNotFoundError(f"4DGS dynerf poses_bounds.npy not found: {poses_path}")

    poses_arr = np.load(poses_path)
    if poses_arr.ndim != 2 or poses_arr.shape[1] < 17:
        raise ValueError(f"poses_bounds.npy must have shape (N,17+): {poses_path}")
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])
    if poses.shape[0] == 0:
        raise ValueError(f"poses_bounds.npy contains no cameras: {poses_path}")

    pose_height, pose_width, pose_focal = [float(value) for value in poses[0, :, -1]]
    downsample = max(pose_width / 1352.0, 1e-8)
    native_width = int(round(pose_width / downsample))
    native_height = int(round(pose_height / downsample))
    focal = pose_focal / downsample

    converted = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], axis=-1)
    requested_split = "val" if args.camera_split == "auto" else str(args.camera_split)
    eval_index = 0
    if requested_split in {"val", "test"}:
        camera_indices = [eval_index]
    elif requested_split == "train":
        camera_indices = [index for index in range(converted.shape[0]) if index != eval_index]
    else:
        raise ValueError("4DGS dynerf camera supports --camera-split auto, val, test, or train")
    if not camera_indices:
        raise ValueError(f"4DGS dynerf split {requested_split!r} has no cameras")
    if args.camera_index < 0 or args.camera_index >= len(camera_indices):
        raise IndexError(
            f"--camera-index {args.camera_index} outside {requested_split} split count {len(camera_indices)}"
        )

    all_c2ws = np.stack([_dynerf_pose_to_c2w(converted[index]) for index in range(converted.shape[0])], axis=0)
    selected_camera_index = int(camera_indices[int(args.camera_index)])
    c2w = all_c2ws[selected_camera_index].astype(np.float32)
    K_original = np.array(
        [[focal, 0.0, native_width / 2.0], [0.0, focal, native_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    K_scaled = scale_intrinsics_to_panel(
        K_original,
        original_height=native_height,
        original_width=native_width,
        target_height=int(height),
        target_width=int(width),
        crop_to_fill=not args.no_crop_to_fill,
    )
    scene_center = all_c2ws[:, :3, 3].mean(axis=0).astype(np.float32)
    label = f"fourdgs-dynerf:{source_path}:{requested_split}[{args.camera_index}]/cam{selected_camera_index:02d}"
    return c2w, K_scaled.astype(np.float32), scene_center, label


def _dynerf_pose_to_c2w(pose: np.ndarray) -> np.ndarray:
    R = np.array(pose[:3, :3], dtype=np.float64, copy=True)
    R = -R
    R[:, 0] = -R[:, 0]
    T = -np.asarray(pose[:3, 3], dtype=np.float64).dot(R)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = T
    return np.linalg.inv(w2c).astype(np.float32)


def maybe_resolve_data_dir(args: argparse.Namespace, cfg: dict[str, Any]) -> Optional[Path]:
    from lkg_experiment.run_coherent_raster_experiment import maybe_resolve_data_dir as _maybe_resolve_data_dir

    return _maybe_resolve_data_dir(args, cfg)


def dataset_category_from_cfg(cfg: dict[str, Any], data_dir: Optional[Path]) -> str:
    from lkg_experiment.run_coherent_raster_experiment import dataset_category_from_cfg as _dataset_category_from_cfg

    return _dataset_category_from_cfg(cfg, data_dir)


def load_colmap_camera(args: argparse.Namespace, cfg: dict[str, Any], data_dir: Path, width: int, height: int):
    from lkg_experiment.run_coherent_raster_experiment import load_colmap_camera as _load_colmap_camera

    return _load_colmap_camera(args, cfg, data_dir, width, height)


def load_blender_camera(args: argparse.Namespace, cfg: dict[str, Any], data_dir: Path, width: int, height: int):
    from lkg_experiment.run_coherent_raster_experiment import load_blender_camera as _load_blender_camera

    return _load_blender_camera(args, cfg, data_dir, width, height)


def estimate_bbox_camera(args: argparse.Namespace, means: Any, width: int, height: int):
    from lkg_experiment.run_coherent_raster_experiment import estimate_bbox_camera as _estimate_bbox_camera

    return _estimate_bbox_camera(args, means, width, height)


if __name__ == "__main__":
    main()
