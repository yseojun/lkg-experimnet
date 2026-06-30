from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lkg_experiment.omg4_ftgs.camera import load_pose_camera, load_test_frames
from lkg_experiment.omg4_ftgs.model import load_dynamic_gaussians
from lkg_experiment.omg4_ftgs.render import install_gsplat_root, render_splats_coherent, render_splats_gsplat


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WEIGHTS_ROOT = Path("/data/ysj/result/4dgs/OMG4-FTGS_weights")
DEFAULT_CHECKPOINT_PATH = DEFAULT_WEIGHTS_ROOT / "ours_L_weight" / "cook_spinach.xz"
DEFAULT_DATA_PATH = Path("/data/ysj/dataset/N3DV/cook_spinach")
DEFAULT_OUTPUT_ROOT = Path("/data/ysj/result/coherent-raster/generated/omg4_ftgs")
DEFAULT_GSPLAT_ROOT = REPO_ROOT / "gsplat"
DEFAULT_OMG4_ROOT = REPO_ROOT / "OMG4"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render OMG4-FTGS 1-view images and same-camera videos")
    parser.add_argument("--mode", choices=("single", "video", "both"), default="both")
    parser.add_argument("--renderer", choices=("gsplat", "coherent", "both"), default="gsplat")
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--save-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=100.0)
    parser.add_argument("--camera-model", choices=("pinhole", "ortho", "fisheye"), default="pinhole")
    parser.add_argument("--debug-cr", action="store_true")
    parser.add_argument("--gsplat-root", default=str(DEFAULT_GSPLAT_ROOT))
    parser.add_argument("--omg4-root", default=str(DEFAULT_OMG4_ROOT))
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


def run(args: argparse.Namespace) -> int:
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    install_gsplat_root(args.gsplat_root)
    if str(args.renderer) == "both" and args.mode in {"video", "both"}:
        raise ValueError("--renderer both is currently supported for --mode single only")

    frames = load_test_frames(args.data_path, camera_index=int(args.camera_index))
    camera_set = load_pose_camera(args.data_path, camera_index=int(args.camera_index), resolution=float(args.resolution))
    dynamic_model = load_dynamic_gaussians(args.checkpoint_path, device=str(args.device))

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": str(args.mode),
        "renderer": str(args.renderer),
        "checkpoint_path": str(Path(args.checkpoint_path).expanduser()),
        "data_path": str(Path(args.data_path).expanduser()),
        "camera_index": int(args.camera_index),
        "frame_index": int(args.frame_index),
        "resolution": float(args.resolution),
        "width": int(_render_width(args, camera_set)),
        "height": int(_render_height(args, camera_set)),
        "output_dir": str(output_dir),
        "single": None,
        "video": None,
    }

    if args.mode in {"single", "both"}:
        manifest["single"] = _render_single(args, dynamic_model, camera_set, frames, output_dir)
    if args.mode in {"video", "both"}:
        manifest["video"] = _render_video(args, dynamic_model, camera_set, frames, output_dir)

    _write_manifest(output_dir / "manifest.json", manifest)
    return 0


def _render_single(args: argparse.Namespace, dynamic_model: Any, camera_set: Any, frames: list[Any], output_dir: Path) -> dict[str, Any]:
    frame = _select_frame(frames, int(args.frame_index))
    renderer = str(args.renderer)
    splats = dynamic_model.materialize(float(frame.timestamp))
    base = {
        "renderer": renderer,
        "frame_index": int(frame.index),
        "timestamp": float(frame.timestamp),
    }
    if renderer == "gsplat":
        _image, result = _render_materialized_single(
            args,
            splats,
            camera_set,
            renderer="gsplat",
            output_path=output_dir / "single.png",
            timestamp=float(frame.timestamp),
            frame_index=int(frame.index),
        )
        result["renderer"] = "gsplat"
        return result
    if renderer == "coherent":
        _image, result = _render_materialized_single(
            args,
            splats,
            camera_set,
            renderer="coherent",
            output_path=output_dir / "single_coherent.png",
            timestamp=float(frame.timestamp),
            frame_index=int(frame.index),
        )
        return {**base, "coherent": result}
    if renderer == "both":
        gsplat_image, gsplat_result = _render_materialized_single(
            args,
            splats,
            camera_set,
            renderer="gsplat",
            output_path=output_dir / "single_gsplat.png",
            timestamp=float(frame.timestamp),
            frame_index=int(frame.index),
        )
        coherent_image, coherent_result = _render_materialized_single(
            args,
            splats,
            camera_set,
            renderer="coherent",
            output_path=output_dir / "single_coherent.png",
            timestamp=float(frame.timestamp),
            frame_index=int(frame.index),
        )
        diff_path = output_dir / "single_diff.png"
        _save_tensor_image(diff_path, _diff_tensor(gsplat_image, coherent_image))
        return {
            **base,
            "gsplat": gsplat_result,
            "coherent": coherent_result,
            "diff_path": str(diff_path),
            "metrics": {"gsplat_vs_coherent": _compute_pair_metrics(gsplat_image, coherent_image)},
        }
    raise ValueError(f"unsupported renderer: {renderer}")


def _render_video(args: argparse.Namespace, dynamic_model: Any, camera_set: Any, frames: list[Any], output_dir: Path) -> dict[str, Any]:
    frames_dir = output_dir / "frames"
    if bool(args.save_frames):
        frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    render_ms: list[float] = []
    for frame in frames:
        start = time.perf_counter()
        image = _render_timestamp(args, dynamic_model, camera_set, timestamp=float(frame.timestamp))
        render_ms.append((time.perf_counter() - start) * 1000.0)
        if bool(args.save_frames):
            frame_path = frames_dir / f"frame_{int(frame.index):04d}.png"
            _save_tensor_image(frame_path, image)
            frame_paths.append(frame_path)

    mp4_status = _encode_video_from_frames(frames_dir, output_dir / "video.mp4", fps=float(args.fps)) if frame_paths else {
        "written": False,
        "path": None,
        "error": "frame saving disabled",
    }
    return {
        "frames_dir": str(frames_dir) if frame_paths else None,
        "frame_count": len(frames),
        "timestamps": [float(frame.timestamp) for frame in frames],
        "render_ms_mean": float(np.mean(render_ms)) if render_ms else 0.0,
        "mp4": mp4_status,
    }


def _render_timestamp(args: argparse.Namespace, dynamic_model: Any, camera_set: Any, *, timestamp: float):
    splats = dynamic_model.materialize(float(timestamp))
    return _render_materialized(args, splats, camera_set, renderer=str(args.renderer))


def _render_materialized_single(
    args: argparse.Namespace,
    splats: Any,
    camera_set: Any,
    *,
    renderer: str,
    output_path: Path,
    timestamp: float,
    frame_index: int,
):
    start = time.perf_counter()
    image = _render_materialized(args, splats, camera_set, renderer=renderer)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _save_tensor_image(output_path, image)
    return image, {
        "path": str(output_path),
        "frame_index": int(frame_index),
        "timestamp": float(timestamp),
        "render_ms": float(elapsed_ms),
    }


def _render_materialized(args: argparse.Namespace, splats: Any, camera_set: Any, *, renderer: str):
    if renderer == "both":
        raise ValueError("--renderer both is only supported by single-view comparison")
    viewmat = camera_set.viewmat_at(int(args.camera_index))
    K = _scaled_K(args, camera_set)
    width = _render_width(args, camera_set)
    height = _render_height(args, camera_set)
    if renderer == "gsplat":
        return render_splats_gsplat(
            splats,
            viewmat=viewmat,
            K=K,
            width=width,
            height=height,
            device=str(args.device),
        )
    if renderer == "coherent":
        return render_splats_coherent(
            splats,
            viewmat=viewmat,
            K=K,
            width=width,
            height=height,
            device=str(args.device),
            tile_size=int(args.tile_size),
            near_plane=float(args.near_plane),
            far_plane=float(args.far_plane),
            camera_model=str(args.camera_model),
            debug=bool(args.debug_cr),
        )
    raise ValueError(f"unsupported renderer: {renderer}")


def _select_frame(frames: list[Any], index: int):
    if index < 0 or index >= len(frames):
        raise IndexError(f"frame_index {index} out of range for {len(frames)} frames")
    return frames[index]


def _render_width(args: argparse.Namespace, camera_set: Any) -> int:
    return int(args.width) if int(args.width) > 0 else int(camera_set.width)


def _render_height(args: argparse.Namespace, camera_set: Any) -> int:
    return int(args.height) if int(args.height) > 0 else int(camera_set.height)


def _scaled_K(args: argparse.Namespace, camera_set: Any):
    width = _render_width(args, camera_set)
    height = _render_height(args, camera_set)
    K = np.asarray(camera_set.K, dtype=np.float32).copy()
    if width == int(camera_set.width) and height == int(camera_set.height):
        return K
    K[0, :] *= float(width) / float(camera_set.width)
    K[1, :] *= float(height) / float(camera_set.height)
    return K


def _save_tensor_image(path: Path, image: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image.detach().clamp(0.0, 1.0).cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"image must have shape [3,H,W] or [H,W,3], got {array.shape}")
    if array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"image must have three channels, got {array.shape}")
    Image.fromarray(np.asarray(np.clip(array, 0.0, 1.0) * 255.0, dtype=np.uint8)).save(path)


def _diff_tensor(a: Any, b: Any):
    if hasattr(a, "detach"):
        return (a.detach() - b.detach()).abs().mul(4.0).clamp(0.0, 1.0)
    return np.clip(np.abs(np.asarray(a) - np.asarray(b)) * 4.0, 0.0, 1.0)


def _compute_pair_metrics(a: Any, b: Any) -> dict[str, float]:
    a_np = a.detach().float().cpu().numpy() if hasattr(a, "detach") else np.asarray(a, dtype=np.float32)
    b_np = b.detach().float().cpu().numpy() if hasattr(b, "detach") else np.asarray(b, dtype=np.float32)
    diff = a_np.astype(np.float64) - b_np.astype(np.float64)
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    return {
        "mse": mse,
        "mae": mae,
        "psnr": float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse)),
    }


def _encode_video_from_frames(frames_dir: Path, output_path: Path, *, fps: float) -> dict[str, Any]:
    pattern = frames_dir / "frame_%04d.png"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(float(fps)),
        "-i",
        str(pattern),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        return {"written": False, "path": None, "error": str(exc)}
    except subprocess.CalledProcessError as exc:
        return {"written": False, "path": None, "error": exc.stderr.strip() or str(exc)}
    return {"written": True, "path": str(output_path), "error": None}


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser()
    label = args.run_label or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / str(label)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value
