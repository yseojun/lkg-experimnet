from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lkg_experiment.omg4_ftgs import cli as single_cli
from lkg_experiment.omg4_ftgs.camera import load_pose_camera, load_test_frames
from lkg_experiment.omg4_ftgs.interlaced_experiment import (
    DEFAULT_DATASET_ALIASES,
    preflight_cuda_runtime,
    resolve_dataset_scene,
)
from lkg_experiment.omg4_ftgs.model import load_dynamic_gaussians
from lkg_experiment.omg4_ftgs.render import (
    build_interlaced_viewpoint_index,
    estimate_orbit_center,
    install_gsplat_root,
    prepare_cr_lookup_tensors,
    render_splats_interlaced_coherent,
    synthesize_interlaced_viewmats,
)
from lkg_experiment.rtgs_coherent import lkg_web_server as rtgs_web
from lkg_experiment.rtgs_coherent.cr_66views import apply_orbit_state_to_c2w


DEFAULT_PORT = 8090


@dataclass(frozen=True)
class Omg4CheckpointOption:
    id: str
    scene: str
    dataset_scene: str
    weight_group: str
    checkpoint_path: Path
    data_path: Path
    checkpoint: str

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scene": self.scene,
            "dataset_scene": self.dataset_scene,
            "weight_group": self.weight_group,
            "checkpoint_path": str(self.checkpoint_path),
            "data_path": str(self.data_path),
            "checkpoint": self.checkpoint,
        }


@dataclass(frozen=True)
class Omg4RenderContext:
    args: argparse.Namespace
    checkpoint: Omg4CheckpointOption
    dynamic_model: Any
    camera_set: Any
    viewpoint_index: Any
    map_metadata: dict[str, Any]
    view_idx_matrix: Any
    subpixel_coord_matrix: Any
    lookup_cpu_ms: float
    lookup_h2d_ms: float
    K: Any
    base_c2w: Any
    orbit_center: Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive OMG4-FTGS + CoherentRaster LKG web controller")
    parser.add_argument("--weights-root", default=str(single_cli.DEFAULT_WEIGHTS_ROOT))
    parser.add_argument("--weight-group", default="ours_L_weight")
    parser.add_argument("--data-root", default="/data/ysj/dataset/N3DV")
    parser.add_argument("--initial-scene", default=None)
    parser.add_argument("--initial-checkpoint", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--display-mode", choices=("glfw", "none"), default="glfw")
    parser.add_argument("--display-index", default=0, type=int)
    parser.add_argument("--bridge-sdk-root", default=str(rtgs_web.DEFAULT_BRIDGE_SDK_ROOT))
    parser.add_argument("--allow-bridge-display-fallback", action="store_true")
    parser.add_argument("--allow-non-native-panel-size", action="store_true")
    parser.add_argument("--window-x", type=int)
    parser.add_argument("--window-y", type=int)
    parser.add_argument("--decorated", action="store_true")
    parser.add_argument("--not-floating", action="store_true")
    parser.add_argument("--swap-interval", default=0, type=int)
    parser.add_argument("--panel-renderer", choices=("fixed", "shader"), default="fixed")
    parser.add_argument("--texture-upload-mode", choices=rtgs_web.TEXTURE_UPLOAD_CHOICES, default=rtgs_web.TEXTURE_UPLOAD_AUTO)
    parser.add_argument("--gsplat-root", default=str(single_cli.DEFAULT_GSPLAT_ROOT))
    parser.add_argument("--torch-extensions-dir", default=None)
    parser.add_argument("--texture-upload-cuda-device", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=2560)
    parser.add_argument("--views", type=int, default=66)
    parser.add_argument("--cluster-size", type=int, default=8)
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
    parser.add_argument("--max-frames", default=0, type=int)
    parser.add_argument("--frame-sleep", default=0.0, type=float)
    parser.add_argument("--playback-mode", choices=("auto", "static"), default="auto")
    parser.add_argument("--playback-fps", default=0.0, type=float)
    parser.add_argument("--web-preview-fps", default=5.0, type=float)
    return parser


def scan_omg4_checkpoints(
    weights_root: Path | str,
    *,
    weight_group: str,
    data_root: Path | str,
) -> list[Omg4CheckpointOption]:
    weights_dir = Path(weights_root).expanduser() / str(weight_group)
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"OMG4-FTGS weight group not found: {weights_dir}")
    root = Path(data_root).expanduser()
    options: list[Omg4CheckpointOption] = []
    for checkpoint_path in sorted(weights_dir.glob("*.xz")):
        scene = checkpoint_path.stem
        dataset_scene, data_path = resolve_dataset_scene(scene, data_root=root, require_data=True)
        checkpoint = checkpoint_path.name
        options.append(
            Omg4CheckpointOption(
                id=f"{weight_group}:{scene}",
                scene=scene,
                dataset_scene=dataset_scene,
                weight_group=str(weight_group),
                checkpoint_path=checkpoint_path,
                data_path=data_path,
                checkpoint=checkpoint,
            )
        )
    if not options:
        raise ValueError(f"no OMG4-FTGS checkpoints found in {weights_dir}")
    return options


def build_playback_timeline(*, camera_index: int, frames: list[Any], playback_mode: str = "auto", frame_index: int = 0) -> list[rtgs_web.PlaybackFrame]:
    if str(playback_mode) == "static":
        frame = single_cli._select_frame(frames, int(frame_index))
        return [
            rtgs_web.PlaybackFrame(
                frame_index=0,
                camera_index=int(camera_index),
                n3dv_frame_index=int(frame.index),
                timestamp=float(frame.timestamp),
                camera=None,
            )
        ]
    return [
        rtgs_web.PlaybackFrame(
            frame_index=timeline_index,
            camera_index=int(camera_index),
            n3dv_frame_index=int(frame.index),
            timestamp=float(frame.timestamp),
            camera=None,
        )
        for timeline_index, frame in enumerate(frames)
    ]


def prepare_runtime_bundle(args: argparse.Namespace, option: Omg4CheckpointOption) -> rtgs_web.RuntimeBundle:
    import torch

    render_args = _panel_args(args)
    install_gsplat_root(args.gsplat_root)
    frames = load_test_frames(option.data_path, camera_index=int(args.camera_index))
    camera_set = load_pose_camera(option.data_path, camera_index=int(args.camera_index), resolution=float(args.resolution))
    dynamic_model = load_dynamic_gaussians(option.checkpoint_path, device=str(args.device))

    viewpoint_index, map_metadata = build_interlaced_viewpoint_index(render_args, width=int(args.width), height=int(args.height))
    lookup = prepare_cr_lookup_tensors(
        viewpoint_index,
        device=str(args.device),
        tile_size=int(args.tile_size),
        use_remapping=True,
    )

    setup_frame = single_cli._select_frame(frames, int(args.frame_index))
    setup_splats = dynamic_model.materialize(float(setup_frame.timestamp))
    source_viewmat = torch.as_tensor(camera_set.viewmat_at(int(args.camera_index)), dtype=torch.float32, device=str(args.device)).contiguous()
    base_c2w = torch.linalg.inv(source_viewmat)
    orbit_center = estimate_orbit_center(
        c2w=base_c2w,
        splat_means=setup_splats.means,
        orbit_center_distance=float(args.orbit_center_distance),
    )
    context = Omg4RenderContext(
        args=render_args,
        checkpoint=option,
        dynamic_model=dynamic_model,
        camera_set=camera_set,
        viewpoint_index=viewpoint_index,
        map_metadata=dict(map_metadata),
        view_idx_matrix=lookup.view_idx_matrix,
        subpixel_coord_matrix=lookup.subpixel_coord_matrix,
        lookup_cpu_ms=float(lookup.lookup_cpu_ms),
        lookup_h2d_ms=float(lookup.lookup_h2d_ms),
        K=single_cli._scaled_panel_K(render_args, camera_set),
        base_c2w=base_c2w,
        orbit_center=orbit_center,
    )
    return rtgs_web.RuntimeBundle(
        checkpoint=option,
        args=render_args,
        context=context,
        cursor=rtgs_web.PlaybackCursor(
            build_playback_timeline(
                camera_index=int(args.camera_index),
                frames=frames,
                playback_mode=str(args.playback_mode),
                frame_index=int(args.frame_index),
            )
        ),
    )


def render_omg4_lkg_frame(context: Omg4RenderContext, frame_info: rtgs_web.PlaybackFrame, *, orbit_state: Any | None = None) -> tuple[Any, float]:
    args = context.args
    _sync_device(str(args.device))
    start = time.perf_counter()
    splats = context.dynamic_model.materialize(float(frame_info.timestamp))
    c2w, orbit_center = apply_orbit_state_to_c2w(context.base_c2w, context.orbit_center, orbit_state or rtgs_web.OrbitState())
    adjacent_viewmats = synthesize_interlaced_viewmats(
        c2w=c2w,
        orbit_center=orbit_center,
        views=int(args.views),
        cluster_size=int(args.cluster_size),
        view_degree=float(args.view_degree),
        orbit_direction=int(args.orbit_direction),
        device=str(args.device),
    )
    render_result = render_splats_interlaced_coherent(
        splats,
        adjacent_viewmats=adjacent_viewmats,
        K=context.K,
        viewpoint_index=context.viewpoint_index,
        view_idx_matrix=context.view_idx_matrix,
        subpixel_coord_matrix=context.subpixel_coord_matrix,
        width=_panel_width(args),
        height=_panel_height(args),
        device=str(args.device),
        tile_size=int(args.tile_size),
        near_plane=float(args.near_plane),
        far_plane=float(args.far_plane),
        camera_model=str(args.camera_model),
        debug=bool(args.debug_cr),
        return_meta=True,
        return_timing=True,
    )
    _sync_device(str(args.device))
    render_ms = (time.perf_counter() - start) * 1000.0
    if isinstance(render_result, tuple):
        image = render_result[0]
    else:
        image = render_result
    return image.clamp(0.0, 1.0).contiguous(), float(render_ms)


class CheckpointWorker(threading.Thread):
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        state: rtgs_web.InteractiveState,
        requests: queue.Queue[Omg4CheckpointOption],
        gpu_lock: threading.Lock,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="Omg4CheckpointWorker", daemon=True)
        self.args = args
        self.state = state
        self.requests = requests
        self.gpu_lock = gpu_lock
        self.stop_event = stop_event

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                option = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            self.state.set_loading(True)
            self.state.set_error(None)
            try:
                with self.gpu_lock:
                    bundle = prepare_runtime_bundle(self.args, option)
                self.state.set_runtime_bundle(bundle)
            except Exception as exc:
                self.state.set_error(str(exc))
            finally:
                self.state.set_loading(False)


class RenderingWorker(threading.Thread):
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        state: rtgs_web.InteractiveState,
        gpu_lock: threading.Lock,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="Omg4RenderingWorker", daemon=True)
        self.args = args
        self.state = state
        self.gpu_lock = gpu_lock
        self.stop_event = stop_event

    def run(self) -> None:
        while not self.stop_event.is_set():
            bundle = self.state.runtime_snapshot()
            if bundle is None:
                time.sleep(0.05)
                continue
            try:
                frame_info = bundle.cursor.next_frame()
                render_start = time.perf_counter()
                with self.gpu_lock:
                    frame, render_ms = render_omg4_lkg_frame(
                        bundle.context,
                        frame_info,
                        orbit_state=self.state.orbit_snapshot(),
                    )
                    frame = frame.detach()
                    upload_event = rtgs_web.record_cuda_ready_event(frame)
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
            sleep_s = max(float(self.args.frame_sleep), rtgs_web._playback_sleep_seconds(float(self.args.playback_fps), render_start))
            if sleep_s > 0.0:
                time.sleep(sleep_s)


def run_interactive_server(args: argparse.Namespace) -> int:
    preflight_cuda_runtime(_panel_args(args))
    options = scan_omg4_checkpoints(args.weights_root, weight_group=str(args.weight_group), data_root=args.data_root)
    catalog = rtgs_web.CheckpointCatalog(options)
    initial = catalog.default(scene=args.initial_scene, checkpoint=args.initial_checkpoint)
    state = rtgs_web.InteractiveState()
    state.set_active_checkpoint(initial)
    requests: queue.Queue[Omg4CheckpointOption] = queue.Queue()
    requests.put(initial)
    gpu_lock = threading.Lock()
    stop_event = threading.Event()
    checkpoint_worker = CheckpointWorker(args=args, state=state, requests=requests, gpu_lock=gpu_lock, stop_event=stop_event)
    rendering_worker = RenderingWorker(args=args, state=state, gpu_lock=gpu_lock, stop_event=stop_event)
    preview_worker = rtgs_web.WebPreviewWorker(args=args, state=state, stop_event=stop_event)
    display_worker = rtgs_web.DisplayWorker(args=args, state=state, gpu_lock=gpu_lock, stop_event=stop_event)
    server = rtgs_web.create_http_server(args.host, int(args.port), state=state, catalog=catalog, checkpoint_queue=requests)
    web_thread = threading.Thread(target=server.serve_forever, name="Omg4WebServerWorker", daemon=True)
    print(f"OMG4-FTGS LKG web server listening on http://{args.host}:{int(args.port)}", file=sys.stderr, flush=True)
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
        rtgs_web.stop_http_server(server, web_thread)
        checkpoint_worker.join(timeout=5)
        rendering_worker.join(timeout=5)
        preview_worker.join(timeout=5)
        display_worker.join(timeout=5)
    return 0


def _panel_args(args: argparse.Namespace) -> argparse.Namespace:
    panel_args = argparse.Namespace(**vars(args))
    panel_args.panel_width = int(args.width)
    panel_args.panel_height = int(args.height)
    return panel_args


def _panel_width(args: argparse.Namespace) -> int:
    return int(getattr(args, "panel_width", getattr(args, "width")))


def _panel_height(args: argparse.Namespace) -> int:
    return int(getattr(args, "panel_height", getattr(args, "height")))


def _sync_device(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def main(argv: list[str] | None = None) -> int:
    return run_interactive_server(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
