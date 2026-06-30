from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from lkg_experiment.coherent_default.cuda_gl_texture import (
    TEXTURE_UPLOAD_AUTO,
    TEXTURE_UPLOAD_CHOICES,
)


DEFAULT_CHECKPOINT_ROOT = Path("/data/ysj/result/4dgs/RTGS")
DEFAULT_RTGS_CODE_ROOT = Path(__file__).resolve().parents[4] / "4d-gaussian-splatting"
DEFAULT_GSPLAT_ROOT = Path(__file__).resolve().parents[4] / "gsplat"
DEFAULT_BRIDGE_SDK_ROOT = Path(__file__).resolve().parents[4] / "Bridge-Python-SDK-Lab"
DEFAULT_DATASET_ROOT = Path("/data/ysj/dataset/dnerf")
DEFAULT_N3DV_ROOT = Path("/data/ysj/dataset/N3DV")
DEFAULT_VIEWPOINT_INDEX_PATH = Path("/data/ysj/result/coherent-raster/generated/lkg_go_1440x2560_66_views_lkg_calibration.npz")
LOADING_TEXT = "\uccb4\ud06c\ud3ec\uc778\ud2b8 \ubcc0\uacbd\uc911"
PANEL_WAKE_INTERVAL_S = 30.0


@dataclass(frozen=True)
class CheckpointOption:
    id: str
    scene: str
    dataset_kind: str
    model_path: Path
    checkpoint_path: Path
    checkpoint: str

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scene": self.scene,
            "dataset_kind": self.dataset_kind,
            "model_path": str(self.model_path),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint": self.checkpoint,
        }


class CheckpointCatalog:
    def __init__(self, options: list[CheckpointOption]):
        if not options:
            raise ValueError("checkpoint catalog must contain at least one checkpoint")
        self.options = list(options)
        self._by_id = {option.id: option for option in self.options}

    def get(self, option_id: str) -> CheckpointOption:
        try:
            return self._by_id[str(option_id)]
        except KeyError as exc:
            raise KeyError(f"unknown checkpoint id: {option_id}") from exc

    def default(self, *, scene: str | None = None, checkpoint: str | None = None) -> CheckpointOption:
        for option in self.options:
            if scene is not None and option.scene != scene:
                continue
            if checkpoint is not None and option.checkpoint != checkpoint:
                continue
            return option
        if scene is not None:
            for option in self.options:
                if option.scene == scene:
                    return option
        return self.options[0]

    def to_json(self) -> dict[str, Any]:
        return {"checkpoints": [option.to_json() for option in self.options]}


@dataclass(frozen=True)
class OrbitState:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    distance_scale: float = 1.0

    def to_json(self) -> dict[str, float]:
        return {
            "yaw_deg": float(self.yaw_deg),
            "pitch_deg": float(self.pitch_deg),
            "pan_x": float(self.pan_x),
            "pan_y": float(self.pan_y),
            "distance_scale": float(self.distance_scale),
        }


@dataclass(frozen=True)
class PlaybackFrame:
    frame_index: int
    camera_index: int
    n3dv_frame_index: int | None
    timestamp: float
    camera: Any


@dataclass(frozen=True)
class PlaybackStatus:
    frame_index: int
    frame_count: int
    timestamp: float
    camera_index: int
    n3dv_frame_index: int | None

    def to_json(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "frame_count": int(self.frame_count),
            "timestamp": float(self.timestamp),
            "camera_index": int(self.camera_index),
            "n3dv_frame_index": None if self.n3dv_frame_index is None else int(self.n3dv_frame_index),
        }


class PlaybackCursor:
    def __init__(self, timeline: list[PlaybackFrame]) -> None:
        if not timeline:
            raise ValueError("playback timeline must contain at least one frame")
        self.timeline = list(timeline)
        self.index = 0
        self.current = self.timeline[0]

    def next_frame(self) -> PlaybackFrame:
        frame = self.timeline[self.index]
        self.current = frame
        self.index = (self.index + 1) % len(self.timeline)
        return frame

    def current_status(self) -> PlaybackStatus:
        return PlaybackStatus(
            frame_index=int(self.current.frame_index),
            frame_count=len(self.timeline),
            timestamp=float(self.current.timestamp),
            camera_index=int(self.current.camera_index),
            n3dv_frame_index=self.current.n3dv_frame_index,
        )


class FpsAccumulator:
    def __init__(self) -> None:
        self.frames = 0
        self.current_ms: float | None = None
        self.total_ms = 0.0

    def add_frame(self, elapsed_ms: float) -> None:
        elapsed = max(float(elapsed_ms), 1e-6)
        self.frames += 1
        self.current_ms = elapsed
        self.total_ms += elapsed

    def to_json(self) -> dict[str, Any]:
        average_ms = self.total_ms / self.frames if self.frames else None
        current_fps = 1000.0 / self.current_ms if self.current_ms else None
        average_fps = 1000.0 / average_ms if average_ms else None
        return {
            "frames": int(self.frames),
            "current_ms": self.current_ms,
            "average_ms": average_ms,
            "current_fps": current_fps,
            "average_fps": average_fps,
        }


@dataclass(frozen=True)
class RuntimeBundle:
    checkpoint: CheckpointOption
    args: argparse.Namespace
    context: Any
    cursor: PlaybackCursor


class InteractiveState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.active_checkpoint: CheckpointOption | None = None
        self.loading_checkpoint = False
        self.last_error: str | None = None
        self.orbit = OrbitState()
        self.render_fps = FpsAccumulator()
        self.display_fps = FpsAccumulator()
        self.frame_seq = 0
        self.latest_jpeg: bytes | None = None
        self.latest_jpeg_seq = 0
        self.latest_tensor: Any = None
        self.latest_upload_event: Any = None
        self.playback_status: PlaybackStatus | None = None
        self.runtime_bundle: RuntimeBundle | None = None

    def set_active_checkpoint(self, option: CheckpointOption) -> None:
        with self._lock:
            self.active_checkpoint = option

    def set_runtime_bundle(self, bundle: RuntimeBundle) -> None:
        with self._lock:
            self.runtime_bundle = bundle
            self.active_checkpoint = bundle.checkpoint

    def runtime_snapshot(self) -> RuntimeBundle | None:
        with self._lock:
            return self.runtime_bundle

    def orbit_snapshot(self) -> OrbitState:
        with self._lock:
            return self.orbit

    def apply_control(self, payload: dict[str, Any]) -> OrbitState:
        with self._lock:
            self.orbit = apply_orbit_control(self.orbit, payload)
            return self.orbit

    def set_loading(self, loading: bool) -> None:
        with self._lock:
            self.loading_checkpoint = bool(loading)

    def set_error(self, message: str | None) -> None:
        with self._lock:
            self.last_error = message

    def publish_render_frame(
        self,
        *,
        tensor: Any,
        render_ms: float,
        playback: PlaybackStatus | None,
        upload_event: Any = None,
    ) -> None:
        with self._lock:
            self.latest_tensor = tensor
            self.latest_upload_event = upload_event
            self.frame_seq += 1
            self.latest_jpeg_seq = self.frame_seq
            if playback is not None:
                self.playback_status = playback
            self.render_fps.add_frame(render_ms)

    def publish_preview_jpeg(self, *, seq: int, jpeg: bytes) -> None:
        with self._lock:
            self.latest_jpeg_seq = int(seq)
            self.latest_jpeg = bytes(jpeg)

    def latest_tensor_snapshot(self) -> tuple[int, Any]:
        with self._lock:
            return self.frame_seq, self.latest_tensor

    def latest_display_tensor_snapshot(self) -> tuple[int, Any, Any]:
        with self._lock:
            return self.frame_seq, self.latest_tensor, self.latest_upload_event

    def latest_jpeg_snapshot(self) -> tuple[int, bytes | None]:
        with self._lock:
            return self.latest_jpeg_seq, self.latest_jpeg

    def record_display_frame(self, display_ms: float) -> None:
        with self._lock:
            self.display_fps.add_frame(display_ms)

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "loading_checkpoint": bool(self.loading_checkpoint),
                "last_error": self.last_error,
                "frame_seq": int(self.frame_seq),
                "active_checkpoint": None if self.active_checkpoint is None else self.active_checkpoint.to_json(),
                "orbit": self.orbit.to_json(),
                "playback": None if self.playback_status is None else self.playback_status.to_json(),
                "render_fps": self.render_fps.to_json(),
                "display_fps": self.display_fps.to_json(),
                "preview_available": self.latest_jpeg is not None,
                "preview_seq": int(self.latest_jpeg_seq),
            }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive RTGS+CR LKG web controller")
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--initial-scene", default=None)
    parser.add_argument("--initial-checkpoint", default="checkpoints/chkpnt_best.pth")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--display-mode", choices=("glfw", "none"), default="glfw")
    parser.add_argument("--display-index", default=0, type=int)
    parser.add_argument("--bridge-sdk-root", default=str(DEFAULT_BRIDGE_SDK_ROOT))
    parser.add_argument("--allow-bridge-display-fallback", action="store_true")
    parser.add_argument("--allow-non-native-panel-size", action="store_true")
    parser.add_argument("--window-x", type=int)
    parser.add_argument("--window-y", type=int)
    parser.add_argument("--decorated", action="store_true")
    parser.add_argument("--not-floating", action="store_true")
    parser.add_argument("--swap-interval", default=0, type=int)
    parser.add_argument(
        "--panel-renderer",
        choices=("fixed", "shader"),
        default="fixed",
        help="OpenGL panel blit path; fixed avoids shader compilation for fragile Bridge/OpenGL stacks.",
    )
    parser.add_argument("--rtgs-code-root", default=str(DEFAULT_RTGS_CODE_ROOT))
    parser.add_argument("--rtgs-code-policy", choices=("clean", "as-is"), default="clean")
    parser.add_argument("--rtgs-clean-cache-root", default=None)
    parser.add_argument("--torch-extensions-dir", default=None)
    parser.add_argument("--gsplat-root", default=str(DEFAULT_GSPLAT_ROOT))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--n3dv-root", default=str(DEFAULT_N3DV_ROOT))
    parser.add_argument("--config", default=None)
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--camera-index", default=0, type=int)
    parser.add_argument("--n3dv-frame-index", default=0, type=int)
    parser.add_argument("--background", choices=("auto", "white", "black"), default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-load-device", default="cpu")
    parser.add_argument("--width", default=1440, type=int)
    parser.add_argument("--height", default=2560, type=int)
    parser.add_argument("--views", default=66, type=int)
    parser.add_argument("--aspect-fit", choices=("contain", "fill", "fit"), default="contain")
    parser.add_argument("--camera-aspect-mode", choices=("preserve", "expand"), default="expand")
    parser.add_argument("--view-degree", default=53.0, type=float)
    parser.add_argument("--orbit-direction", default=-1, type=int)
    parser.add_argument("--orbit-center-distance", default=0.0, type=float)
    parser.add_argument("--map-mode", choices=("file", "linear"), default="file")
    parser.add_argument("--viewpoint-index-path", default=str(DEFAULT_VIEWPOINT_INDEX_PATH))
    parser.add_argument("--coherent-quantize", choices=("floor", "nearest"), default="floor")
    parser.add_argument("--cluster-size", default=1, type=int)
    parser.add_argument("--tile-size", default=16, type=int)
    parser.add_argument("--near-plane", default=0.01, type=float)
    parser.add_argument("--far-plane", default=100.0, type=float)
    parser.add_argument("--camera-model", choices=("pinhole", "ortho", "fisheye"), default="pinhole")
    parser.add_argument("--no-remapping", action="store_true")
    parser.add_argument("--no-crop-to-fill", action="store_true")
    parser.add_argument("--debug-cr", action="store_true")
    parser.add_argument("--max-frames", default=0, type=int)
    parser.add_argument("--frame-sleep", default=0.0, type=float)
    parser.add_argument("--playback-mode", choices=("auto", "static"), default="auto")
    parser.add_argument("--playback-fps", default=0.0, type=float)
    parser.add_argument("--web-preview-fps", default=5.0, type=float)
    parser.add_argument(
        "--texture-upload-mode",
        choices=TEXTURE_UPLOAD_CHOICES,
        default=TEXTURE_UPLOAD_AUTO,
        help="LKG panel texture upload path: auto tries CUDA-GL interop, cpu keeps the legacy path, cuda-gl fails if unavailable.",
    )
    parser.add_argument(
        "--rtgs-context-loader",
        choices=("lite", "official"),
        default="lite",
        help="Use lightweight checkpoint+single-camera loading for interactive playback, or official Scene loading for diagnostics.",
    )
    return parser


def infer_dataset_kind(scene: str, *, rtgs_code_root: Path | str) -> str:
    root = Path(rtgs_code_root).expanduser()
    if (root / "configs" / "dnerf" / f"{scene}.yaml").exists():
        return "dnerf"
    if (root / "configs" / "dynerf" / f"{scene}.yaml").exists():
        return "n3dv"
    return "dnerf"


def scan_rtgs_checkpoints(root: Path | str, *, rtgs_code_root: Path | str) -> list[CheckpointOption]:
    checkpoint_root = Path(root).expanduser()
    options: list[CheckpointOption] = []
    for path in sorted(checkpoint_root.glob("*/checkpoints/chkpnt*.pth")):
        model_path = path.parents[1]
        scene = model_path.name
        checkpoint = path.relative_to(model_path).as_posix()
        dataset_kind = infer_dataset_kind(scene, rtgs_code_root=rtgs_code_root)
        option_id = f"{scene}:{checkpoint}"
        options.append(
            CheckpointOption(
                id=option_id,
                scene=scene,
                dataset_kind=dataset_kind,
                model_path=model_path,
                checkpoint_path=path,
                checkpoint=checkpoint,
            )
        )
    return sorted(options, key=lambda item: (item.scene, _checkpoint_sort_key(item.checkpoint)))


def _checkpoint_sort_key(checkpoint: str) -> tuple[int, str]:
    name = Path(checkpoint).name
    return (0 if name == "chkpnt_best.pth" else 1, name)


def apply_orbit_control(state: OrbitState, payload: dict[str, Any]) -> OrbitState:
    shift = bool(payload.get("shift", False))
    drag_dx = _float_payload(payload, "drag_dx", 0.0)
    drag_dy = _float_payload(payload, "drag_dy", 0.0)
    wheel_delta = _float_payload(payload, "wheel_delta", 0.0)
    yaw = float(state.yaw_deg)
    pitch = float(state.pitch_deg)
    pan_x = float(state.pan_x)
    pan_y = float(state.pan_y)
    distance_scale = float(state.distance_scale)
    if shift:
        pan_x += drag_dx * 0.002
        pan_y -= drag_dy * 0.002
    else:
        yaw += drag_dx * 0.1
        pitch += drag_dy * 0.1
    if wheel_delta:
        distance_scale *= math.exp(-wheel_delta * 0.08)
    key = str(payload.get("key") or "")
    if key in {"KeyW", "ArrowUp"}:
        pan_y += 0.05
    elif key in {"KeyS", "ArrowDown"}:
        pan_y -= 0.05
    elif key in {"KeyA", "ArrowLeft"}:
        pan_x -= 0.05
    elif key in {"KeyD", "ArrowRight"}:
        pan_x += 0.05
    elif key == "KeyQ":
        distance_scale *= 1.05
    elif key == "KeyE":
        distance_scale /= 1.05
    pitch = max(-85.0, min(85.0, pitch))
    distance_scale = max(0.1, min(10.0, distance_scale))
    return OrbitState(yaw_deg=yaw, pitch_deg=pitch, pan_x=pan_x, pan_y=pan_y, distance_scale=distance_scale)


def build_rtgs_cr_args_for_checkpoint(base_args: argparse.Namespace, option: CheckpointOption) -> argparse.Namespace:
    from lkg_experiment.rtgs_coherent import cr_66views

    argv = [
        "--dataset-kind",
        option.dataset_kind,
        "--model-path",
        str(option.model_path),
        "--checkpoint",
        option.checkpoint,
        "--rtgs-code-root",
        str(base_args.rtgs_code_root),
        "--gsplat-root",
        str(base_args.gsplat_root),
        "--dataset-root",
        str(base_args.dataset_root),
        "--n3dv-root",
        str(base_args.n3dv_root),
        "--split",
        str(base_args.split),
        "--camera-index",
        str(base_args.camera_index),
        "--n3dv-frame-index",
        str(base_args.n3dv_frame_index),
        "--background",
        str(base_args.background),
        "--device",
        str(base_args.device),
        "--checkpoint-load-device",
        str(base_args.checkpoint_load_device),
        "--width",
        str(base_args.width),
        "--height",
        str(base_args.height),
        "--views",
        str(base_args.views),
        "--aspect-fit",
        str(base_args.aspect_fit),
        "--camera-aspect-mode",
        str(base_args.camera_aspect_mode),
        "--view-degree",
        str(base_args.view_degree),
        "--orbit-direction",
        str(base_args.orbit_direction),
        "--orbit-center-distance",
        str(base_args.orbit_center_distance),
        "--map-mode",
        str(base_args.map_mode),
        "--coherent-quantize",
        str(base_args.coherent_quantize),
        "--cluster-size",
        str(base_args.cluster_size),
        "--tile-size",
        str(base_args.tile_size),
        "--near-plane",
        str(base_args.near_plane),
        "--far-plane",
        str(base_args.far_plane),
        "--camera-model",
        str(base_args.camera_model),
        "--rtgs-code-policy",
        str(base_args.rtgs_code_policy),
        "--no-compare-official-sampled",
        "--no-write-interlaced",
        "--no-ssim",
    ]
    if base_args.config:
        argv.extend(["--config", str(base_args.config)])
    if base_args.source_path:
        argv.extend(["--source-path", str(base_args.source_path)])
    if base_args.rtgs_clean_cache_root:
        argv.extend(["--rtgs-clean-cache-root", str(base_args.rtgs_clean_cache_root)])
    if base_args.torch_extensions_dir:
        argv.extend(["--torch-extensions-dir", str(base_args.torch_extensions_dir)])
    if base_args.map_mode == "file":
        argv.extend(["--viewpoint-index-path", str(base_args.viewpoint_index_path)])
    if bool(base_args.no_remapping):
        argv.extend(["--cr-remapping", "off"])
    if bool(base_args.no_crop_to_fill):
        argv.append("--no-crop-to-fill")
    if bool(base_args.debug_cr):
        argv.append("--debug-cr")
    return cr_66views.build_parser().parse_args(argv)


def build_playback_timeline(args: argparse.Namespace, context: Any) -> list[PlaybackFrame]:
    if str(getattr(args, "playback_mode", "auto")) == "static":
        camera = context.runtime.camera
        return [
            PlaybackFrame(
                frame_index=0,
                camera_index=int(args.camera_index),
                n3dv_frame_index=int(args.n3dv_frame_index) if context.runtime.official.scene_paths.dataset_kind == "n3dv" else None,
                timestamp=float(getattr(camera, "timestamp", context.timestamp)),
                camera=camera,
            )
        ]

    official = context.runtime.official
    checkpoint_like = SimpleNamespace(
        model=official.gaussians,
        iteration=official.iteration,
        config=official.config,
        cfg_args=official.cfg_args,
        scene_paths=official.scene_paths,
        checkpoint_path=official.checkpoint_path,
    )

    if official.scene_paths.dataset_kind == "n3dv":
        frame_indices = _n3dv_playback_frame_indices(args, official)
        if frame_indices:
            frames = []
            for timeline_index, frame_index in enumerate(frame_indices):
                camera = _load_playback_camera(
                    checkpoint_like=checkpoint_like,
                    args=args,
                    camera_index=int(args.camera_index),
                    n3dv_frame_index=int(frame_index),
                )
                frames.append(
                    PlaybackFrame(
                        frame_index=timeline_index,
                        camera_index=int(args.camera_index),
                        n3dv_frame_index=int(frame_index),
                        timestamp=float(getattr(camera, "timestamp", 0.0)),
                        camera=camera,
                    )
                )
            return frames

    if official.scene_paths.dataset_kind == "dnerf":
        return _build_fixed_pose_dnerf_timeline(args, official, checkpoint_like)

    frame_count = _dnerf_playback_camera_count(args, official)
    frames = []
    for timeline_index in range(frame_count):
        camera = _load_playback_camera(
            checkpoint_like=checkpoint_like,
            args=args,
            camera_index=timeline_index,
            n3dv_frame_index=int(args.n3dv_frame_index),
        )
        frames.append(
            PlaybackFrame(
                frame_index=timeline_index,
                camera_index=timeline_index,
                n3dv_frame_index=None,
                timestamp=float(getattr(camera, "timestamp", 0.0)),
                camera=camera,
            )
        )
    return frames


def _build_fixed_pose_dnerf_timeline(args: argparse.Namespace, official: Any, checkpoint_like: Any) -> list[PlaybackFrame]:
    frame_count = _dnerf_playback_camera_count(args, official)
    anchor_camera = _load_playback_camera(
        checkpoint_like=checkpoint_like,
        args=args,
        camera_index=int(args.camera_index),
        n3dv_frame_index=int(args.n3dv_frame_index),
    )
    frames = []
    for timeline_index in range(frame_count):
        time_camera = _load_playback_camera(
            checkpoint_like=checkpoint_like,
            args=args,
            camera_index=timeline_index,
            n3dv_frame_index=int(args.n3dv_frame_index),
        )
        timestamp = float(getattr(time_camera, "timestamp", 0.0))
        playback_camera = clone_camera_with_timestamp(
            anchor_camera,
            timestamp=timestamp,
            timeline_index=timeline_index,
        )
        frames.append(
            PlaybackFrame(
                frame_index=timeline_index,
                camera_index=int(args.camera_index),
                n3dv_frame_index=None,
                timestamp=timestamp,
                camera=playback_camera,
            )
        )
    return frames


def clone_camera_with_timestamp(anchor_camera: Any, *, timestamp: float, timeline_index: int) -> Any:
    camera = deepcopy(anchor_camera)
    if hasattr(camera, "timestamp"):
        camera.timestamp = float(timestamp)
    if hasattr(camera, "uid"):
        camera.uid = int(timeline_index)
    if hasattr(camera, "colmap_id"):
        camera.colmap_id = int(timeline_index)
    if hasattr(camera, "image_name"):
        camera.image_name = f"time_{int(timeline_index):03d}"
    if hasattr(camera, "image"):
        camera.image = None
    return camera


def _load_playback_camera(*, checkpoint_like: Any, args: argparse.Namespace, camera_index: int, n3dv_frame_index: int) -> Any:
    from lkg_experiment.rtgs_coherent.cli import load_rtgs_camera

    _gt, camera = load_rtgs_camera(
        checkpoint=checkpoint_like,
        split=str(args.split),
        camera_index=int(camera_index),
        device=str(args.device),
        n3dv_frame_index=int(n3dv_frame_index),
    )
    if hasattr(camera, "image"):
        camera.image = None
    return camera


def _dnerf_playback_camera_count(args: argparse.Namespace, official: Any) -> int:
    from lkg_experiment.rtgs_coherent.cli import _count_blender_cameras

    return _count_blender_cameras(
        source_path=Path(official.source_path),
        split=str(args.split),
        frame_ratio=int(getattr(official.model_args, "frame_ratio", 1)),
        time_duration=[float(value) for value in official.time_duration],
    )


def _n3dv_playback_frame_indices(args: argparse.Namespace, official: Any) -> list[int]:
    from lkg_experiment.rtgs_coherent.cli import (
        _has_rtgs_n3dv_dynamic_cameras,
        _read_rtgs_n3dv_dynamic_camera_index,
        _rtgs_camera_loader_args,
        _select_rtgs_n3dv_dynamic_cameras,
    )

    loader_args = _rtgs_camera_loader_args(
        config=official.config,
        cfg_args=official.cfg_args,
        scene_paths=official.scene_paths,
        device=str(args.device),
    )
    loader_args.n3dv_frame_index = -1
    if not _has_rtgs_n3dv_dynamic_cameras(loader_args):
        return []
    selected = _select_rtgs_n3dv_dynamic_cameras(
        _read_rtgs_n3dv_dynamic_camera_index(loader_args),
        args=loader_args,
        split=str(args.split),
    )
    return sorted({int(camera["frame_index"]) for camera in selected})


def create_http_server(
    host: str,
    port: int,
    *,
    state: InteractiveState,
    catalog: CheckpointCatalog,
    checkpoint_queue: queue.Queue[CheckpointOption] | None,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            status, content_type, body = handle_http_get(self.path, state=state, catalog=catalog)
            self._send_bytes(body, content_type, status=status, cache=not self.path.startswith("/frame.jpg"))

        def do_POST(self) -> None:
            try:
                payload = self._read_json()
                if self.path == "/api/control":
                    orbit = state.apply_control(payload)
                    self._send_json({"ok": True, "orbit": orbit.to_json()})
                    return
                if self.path == "/api/checkpoint":
                    option = catalog.get(str(payload.get("id") or ""))
                    state.set_loading(True)
                    state.set_error(None)
                    if checkpoint_queue is None:
                        state.set_active_checkpoint(option)
                        state.set_loading(False)
                    else:
                        checkpoint_queue.put(option)
                    self._send_json({"ok": True, "checkpoint": option.to_json()})
                    return
                self.send_response(404)
                self.end_headers()
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            self._send_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"), "application/json", status=status)

        def _send_bytes(self, body: bytes | str, content_type: str, *, status: int = 200, cache: bool = True) -> None:
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            if not cache:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer((host, int(port)), Handler)


def handle_http_get(path: str, *, state: InteractiveState, catalog: CheckpointCatalog) -> tuple[int, str, bytes]:
    if path in {"/", "/index.html"}:
        return 200, "text/html; charset=utf-8", _index_html().encode("utf-8")
    if path.startswith("/api/status"):
        return _json_response(state.status_payload())
    if path.startswith("/api/checkpoints"):
        return _json_response(catalog.to_json())
    if path.startswith("/frame.jpg"):
        _seq, frame = state.latest_jpeg_snapshot()
        if frame is None:
            return 404, "text/plain", b""
        return 200, "image/jpeg", frame
    return 404, "text/plain", b""


def should_wake_panel_display(*, now: float, last_wake: float | None, interval_s: float = PANEL_WAKE_INTERVAL_S) -> bool:
    return last_wake is None or (float(now) - float(last_wake)) >= float(interval_s)


def should_encode_web_preview(*, web_preview_fps: float, now: float, last_preview_at: float | None) -> bool:
    fps = float(web_preview_fps)
    if fps <= 0.0:
        return False
    return last_preview_at is None or (float(now) - float(last_preview_at)) + 1e-9 >= (1.0 / fps)


def _playback_sleep_seconds(playback_fps: float, render_start: float) -> float:
    fps = float(playback_fps)
    if fps <= 0.0:
        return 0.0
    return max(0.0, (1.0 / fps) - (time.perf_counter() - float(render_start)))


def record_cuda_ready_event(tensor: Any) -> Any:
    if not bool(getattr(tensor, "is_cuda", False)):
        return None
    import torch

    device = getattr(tensor, "device", None)
    event = torch.cuda.Event(blocking=False)
    event.record(torch.cuda.current_stream(device))
    return event


def wait_cuda_ready_event(event: Any, tensor: Any) -> None:
    if event is None:
        return
    if bool(getattr(tensor, "is_cuda", False)):
        import torch

        device = getattr(tensor, "device", None)
        torch.cuda.current_stream(device).wait_event(event)
        return
    if hasattr(event, "synchronize"):
        event.synchronize()


def poll_panel_window_events(glfw: Any, window: Any, stop_event: threading.Event) -> bool:
    glfw.poll_events()
    if glfw.window_should_close(window):
        stop_event.set()
        return False
    return True


def _json_response(payload: dict[str, Any], *, status: int = 200) -> tuple[int, str, bytes]:
    return status, "application/json", json.dumps(payload, sort_keys=True).encode("utf-8")


class CheckpointWorker(threading.Thread):
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        state: InteractiveState,
        requests: queue.Queue[CheckpointOption],
        gpu_lock: threading.Lock,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="CheckpointWorker", daemon=True)
        self.args = args
        self.state = state
        self.requests = requests
        self.gpu_lock = gpu_lock
        self.stop_event = stop_event

    def run(self) -> None:
        from lkg_experiment.rtgs_coherent import cr_66views

        while not self.stop_event.is_set():
            try:
                option = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            self.state.set_loading(True)
            self.state.set_error(None)
            try:
                rtgs_args = build_rtgs_cr_args_for_checkpoint(self.args, option)
                with self.gpu_lock:
                    if str(self.args.rtgs_context_loader) == "official":
                        context = cr_66views.prepare_rtgs_cr_66_context(rtgs_args)
                    else:
                        context = cr_66views.prepare_rtgs_cr_66_context_lite(rtgs_args)
                timeline = build_playback_timeline(self.args, context)
                self.state.set_runtime_bundle(
                    RuntimeBundle(
                        checkpoint=option,
                        args=rtgs_args,
                        context=context,
                        cursor=PlaybackCursor(timeline),
                    )
                )
            except Exception as exc:
                self.state.set_error(str(exc))
            finally:
                self.state.set_loading(False)


class RenderingWorker(threading.Thread):
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        state: InteractiveState,
        gpu_lock: threading.Lock,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="RenderingWorker", daemon=True)
        self.args = args
        self.state = state
        self.gpu_lock = gpu_lock
        self.stop_event = stop_event

    def run(self) -> None:
        from lkg_experiment.rtgs_coherent import cr_66views

        while not self.stop_event.is_set():
            bundle = self.state.runtime_snapshot()
            if bundle is None:
                time.sleep(0.05)
                continue
            try:
                frame_info = bundle.cursor.next_frame()
                render_context = cr_66views.context_for_playback_camera(bundle.context, frame_info.camera)
                render_start = time.perf_counter()
                with self.gpu_lock:
                    frame, render_ms = cr_66views.render_rtgs_cr_66_interlaced_frame(
                        render_context,
                        orbit_state=self.state.orbit_snapshot(),
                        debug=bool(self.args.debug_cr),
                    )
                    frame = frame.detach()
                    upload_event = record_cuda_ready_event(frame)
                self.state.publish_render_frame(
                    tensor=frame,
                    render_ms=render_ms,
                    playback=bundle.cursor.current_status(),
                    upload_event=upload_event,
                )
            except Exception as exc:
                self.state.set_error(str(exc))
                time.sleep(0.2)
                continue
            if self.args.max_frames and self.state.status_payload()["frame_seq"] >= int(self.args.max_frames):
                self.stop_event.set()
            sleep_s = max(float(self.args.frame_sleep), _playback_sleep_seconds(float(self.args.playback_fps), render_start))
            if sleep_s > 0.0:
                time.sleep(sleep_s)


class WebPreviewWorker(threading.Thread):
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        state: InteractiveState,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="WebPreviewWorker", daemon=True)
        self.args = args
        self.state = state
        self.stop_event = stop_event

    def run(self) -> None:
        last_seq = -1
        last_preview_at: float | None = None
        while not self.stop_event.is_set():
            now = time.perf_counter()
            if not should_encode_web_preview(
                web_preview_fps=float(self.args.web_preview_fps),
                now=now,
                last_preview_at=last_preview_at,
            ):
                time.sleep(0.01)
                continue
            seq, tensor = self.state.latest_tensor_snapshot()
            if tensor is None or seq == last_seq:
                time.sleep(0.01)
                continue
            try:
                jpeg = tensor_to_jpeg_bytes(tensor)
                self.state.publish_preview_jpeg(seq=seq, jpeg=jpeg)
                last_seq = seq
                last_preview_at = now
            except Exception as exc:
                self.state.set_error(str(exc))
                time.sleep(0.2)


class DisplayWorker(threading.Thread):
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        state: InteractiveState,
        gpu_lock: threading.Lock,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="DisplayWorker", daemon=True)
        self.args = args
        self.state = state
        self.gpu_lock = gpu_lock
        self.stop_event = stop_event

    def run(self) -> None:
        if self.args.display_mode == "none":
            return
        try:
            self._run_glfw()
        except Exception as exc:
            self.state.set_error(str(exc))
            self.stop_event.set()

    def _run_glfw(self) -> None:
        from lkg_experiment.coherent_default.build_lut_npz import _load_bridge_api, install_bridge_sdk_root
        from lkg_experiment.coherent_default.render_looking_glass import (
            configure_x11_environment,
            create_panel_renderer_resources,
            create_panel_texture_uploader,
            delete_panel_renderer_resources,
            destroy_glfw_bootstrap_context,
            draw_panel_texture,
            init_bridge_bootstrap_context,
            init_panel_texture,
            init_panel_window,
            prepare_pyopengl_before_bridge,
            resolve_bridge_or_fallback_display,
            resolve_panel_render_size,
            wake_x11_display_for_panel,
        )

        configure_x11_environment()
        wake_x11_display_for_panel()
        prepare_pyopengl_before_bridge()
        install_bridge_sdk_root(self.args.bridge_sdk_root)
        BridgeAPI, _ = _load_bridge_api()
        bootstrap_window = None
        bridge = None
        window = None
        texture = None
        texture_uploader = None
        program = None
        vao = None
        vbo = None
        try:
            bootstrap_window = init_bridge_bootstrap_context(self.args)
            bridge = BridgeAPI()
            if not bridge.initialize("LkgRtgsWebServer"):
                raise RuntimeError("Bridge initialize failed")
            display_handle, display_info = resolve_bridge_or_fallback_display(bridge, self.args)
            del display_handle
            native_width, native_height = display_info["dimensions"]
            panel_x, panel_y = display_info["position"]
            if self.args.window_x is not None:
                panel_x = int(self.args.window_x)
            if self.args.window_y is not None:
                panel_y = int(self.args.window_y)
            width, height, _ = resolve_panel_render_size(
                requested_width=int(self.args.width),
                requested_height=int(self.args.height),
                native_width=int(native_width),
                native_height=int(native_height),
                allow_non_native=bool(self.args.allow_non_native_panel_size),
            )
            window = init_panel_window(self.args, width, height, int(panel_x), int(panel_y))
            destroy_glfw_bootstrap_context(bootstrap_window)
            bootstrap_window = None
            texture = init_panel_texture(width, height)
            program, vao, vbo = create_panel_renderer_resources(self.args, texture)
            texture_uploader = create_panel_texture_uploader(
                mode=self.args.texture_upload_mode,
                texture=texture,
                width=width,
                height=height,
                torch_extensions_dir=self.args.torch_extensions_dir,
            )
            last_seq = -1
            last_wake = time.perf_counter()
            import glfw

            while not self.stop_event.is_set():
                now = time.perf_counter()
                if should_wake_panel_display(now=now, last_wake=last_wake):
                    wake_x11_display_for_panel()
                    last_wake = now
                if not poll_panel_window_events(glfw, window, self.stop_event):
                    break
                seq, tensor, upload_event = self.state.latest_display_tensor_snapshot()
                if tensor is not None and seq != last_seq:
                    start = time.perf_counter()
                    with self.gpu_lock:
                        wait_cuda_ready_event(upload_event, tensor)
                        texture_uploader.upload(tensor)
                    draw_panel_texture(window, program, vao, texture)
                    glfw.swap_buffers(window)
                    self.state.record_display_frame((time.perf_counter() - start) * 1000.0)
                    last_seq = seq
                else:
                    draw_panel_texture(window, program, vao, texture)
                    glfw.swap_buffers(window)
                    time.sleep(0.005)
        finally:
            if texture_uploader is not None:
                try:
                    texture_uploader.close()
                except Exception:
                    pass
            _cleanup_glfw(
                window=window,
                bootstrap_window=bootstrap_window,
                texture=texture,
                program=program,
                vao=vao,
                vbo=vbo,
                bridge=bridge,
            )


def run_interactive_server(args: argparse.Namespace) -> int:
    options = scan_rtgs_checkpoints(args.checkpoint_root, rtgs_code_root=args.rtgs_code_root)
    catalog = CheckpointCatalog(options)
    initial = catalog.default(scene=args.initial_scene, checkpoint=args.initial_checkpoint)
    state = InteractiveState()
    state.set_active_checkpoint(initial)
    requests: queue.Queue[CheckpointOption] = queue.Queue()
    requests.put(initial)
    gpu_lock = threading.Lock()
    stop_event = threading.Event()
    checkpoint_worker = CheckpointWorker(args=args, state=state, requests=requests, gpu_lock=gpu_lock, stop_event=stop_event)
    rendering_worker = RenderingWorker(args=args, state=state, gpu_lock=gpu_lock, stop_event=stop_event)
    preview_worker = WebPreviewWorker(args=args, state=state, stop_event=stop_event)
    display_worker = DisplayWorker(args=args, state=state, gpu_lock=gpu_lock, stop_event=stop_event)
    server = create_http_server(args.host, int(args.port), state=state, catalog=catalog, checkpoint_queue=requests)
    web_thread = threading.Thread(target=server.serve_forever, name="WebServerWorker", daemon=True)
    print(f"RTGS LKG web server listening on http://{args.host}:{int(args.port)}", file=sys.stderr, flush=True)
    print(f"Initial checkpoint: {initial.checkpoint_path}", file=sys.stderr, flush=True)
    checkpoint_worker.start()
    rendering_worker.start()
    preview_worker.start()
    display_worker.start()
    web_thread.start()
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_http_server(server, web_thread)
        checkpoint_worker.join(timeout=5)
        rendering_worker.join(timeout=5)
        preview_worker.join(timeout=5)
        display_worker.join(timeout=5)
    return 0


def stop_http_server(server: Any, web_thread: threading.Thread, *, timeout: float = 5.0) -> None:
    web_thread_running = bool(web_thread.is_alive())
    try:
        if web_thread_running:
            try:
                server.shutdown()
            except KeyboardInterrupt:
                pass
    finally:
        server.server_close()
        if web_thread_running:
            web_thread.join(timeout=timeout)


def tensor_to_jpeg_bytes(tensor: Any, *, quality: int = 90) -> bytes:
    from io import BytesIO
    from PIL import Image
    from lkg_experiment.coherent_default.coherent_raster_experiment import tensor_to_hwc_uint8

    image = Image.fromarray(tensor_to_hwc_uint8(tensor), mode="RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def _cleanup_glfw(
    *,
    window: Any,
    texture: Any,
    program: Any,
    vao: Any,
    vbo: Any,
    bridge: Any,
    bootstrap_window: Any = None,
) -> None:
    try:
        from lkg_experiment.coherent_default.render_looking_glass import delete_panel_renderer_resources
        from OpenGL import GL

        delete_panel_renderer_resources(program, vao, vbo)
        if texture is not None:
            GL.glDeleteTextures(1, [texture])
    except Exception:
        pass
    if bootstrap_window is not None:
        try:
            from lkg_experiment.coherent_default.render_looking_glass import destroy_glfw_bootstrap_context

            destroy_glfw_bootstrap_context(bootstrap_window)
        except Exception:
            pass
    if window is not None:
        try:
            import glfw

            glfw.destroy_window(window)
            glfw.terminate()
        except Exception:
            pass
    elif bootstrap_window is not None:
        try:
            import glfw

            glfw.terminate()
        except Exception:
            pass
    try:
        if bridge is not None:
            bridge.uninitialize()
    except Exception:
        pass


def _float_payload(payload: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _index_html() -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>RTGS LKG Web Server</title>
  <style>
    html, body {{ margin: 0; height: 100%; background: #111; color: #eee; font-family: sans-serif; }}
    #app {{ display: grid; grid-template-columns: 1fr 320px; height: 100%; }}
    #stage {{ position: relative; display: flex; align-items: center; justify-content: center; overflow: hidden; }}
    #frame {{ max-width: 100%; max-height: 100%; image-rendering: auto; outline: none; }}
    #overlay {{ display: none; position: absolute; inset: 0; align-items: center; justify-content: center; background: rgba(0,0,0,.45); font-size: 24px; }}
    #overlay.visible {{ display: flex; }}
    .spinner {{ width: 24px; height: 24px; margin-right: 12px; border: 3px solid #777; border-top-color: #fff; border-radius: 50%; animation: spin 1s linear infinite; }}
    aside {{ padding: 16px; background: #1c1c1c; overflow: auto; }}
    select, button {{ width: 100%; margin: 6px 0 12px; padding: 8px; background: #2d2d2d; color: #eee; border: 1px solid #555; }}
    pre {{ white-space: pre-wrap; font-size: 12px; }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  </style>
</head>
<body>
<div id="app">
  <main id="stage">
    <img id="frame" tabindex="0">
    <div id="overlay"><div class="spinner"></div><span>{LOADING_TEXT}</span></div>
  </main>
  <aside>
    <label>Checkpoint</label>
    <select id="checkpoint"></select>
    <button id="reload">Load checkpoint</button>
    <pre id="stats"></pre>
  </aside>
</div>
<script>
const frame = document.getElementById('frame');
const overlay = document.getElementById('overlay');
const stats = document.getElementById('stats');
const select = document.getElementById('checkpoint');
let seq = -1, dragging = false, lastX = 0, lastY = 0;
async function post(url, payload) {{
  await fetch(url, {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(payload)}});
}}
async function refreshCheckpoints() {{
  const data = await (await fetch('/api/checkpoints')).json();
  select.innerHTML = '';
  for (const item of data.checkpoints) {{
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = `${{item.scene}} / ${{item.checkpoint}}`;
    select.appendChild(option);
  }}
}}
async function refreshStatus() {{
  const data = await (await fetch('/api/status')).json();
  overlay.classList.toggle('visible', !!data.loading_checkpoint);
  if (data.active_checkpoint) select.value = data.active_checkpoint.id;
  if (data.preview_available && data.preview_seq !== seq) {{
    seq = data.preview_seq;
    frame.src = `/frame.jpg?seq=${{data.preview_seq}}`;
  }} else if (!data.preview_available && frame.hasAttribute('src')) {{
    frame.removeAttribute('src');
  }}
  stats.textContent = JSON.stringify(data, null, 2);
}}
frame.addEventListener('pointerdown', event => {{ dragging = true; lastX = event.clientX; lastY = event.clientY; frame.setPointerCapture(event.pointerId); frame.focus(); }});
frame.addEventListener('pointermove', event => {{
  if (!dragging) return;
  const dx = event.clientX - lastX, dy = event.clientY - lastY;
  lastX = event.clientX; lastY = event.clientY;
  post('/api/control', {{drag_dx: dx, drag_dy: dy, shift: event.shiftKey}});
}});
frame.addEventListener('pointerup', () => {{ dragging = false; }});
frame.addEventListener('wheel', event => {{ event.preventDefault(); post('/api/control', {{wheel_delta: Math.sign(event.deltaY)}}); }}, {{passive: false}});
window.addEventListener('keydown', event => {{ post('/api/control', {{key: event.code}}); }});
document.getElementById('reload').addEventListener('click', () => post('/api/checkpoint', {{id: select.value}}));
refreshCheckpoints().then(refreshStatus);
setInterval(refreshStatus, 250);
</script>
</body>
</html>"""


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run_interactive_server(args)


if __name__ == "__main__":
    raise SystemExit(main())
