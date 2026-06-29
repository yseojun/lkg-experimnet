from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import numpy as np

from lkg_experiment.coherent_default.coherent_raster_experiment import tensor_to_hwc_uint8
from lkg_experiment.rtgs_coherent.cli import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DNERF_ROOT,
    DEFAULT_MODEL_PATH,
    DEFAULT_N3DV_ROOT,
    DEFAULT_RTGS_CODE_ROOT,
    EXPERIMENT_ROOT,
    RtgsScenePaths,
    _background_tensor,
    _camera_manifest_fields,
    _camera_source_path,
    _clear_conflicting_modules,
    _install_pointops_import_stub_if_needed,
    _json_ready,
    _make_ssim_metric,
    _N3DV_SCENE_ALIASES,
    _pipeline_namespace_for_model,
    compute_pair_metrics,
    load_rtgs_yaml_config,
    resolve_checkpoint_path,
)


DEFAULT_OFFICIAL_OUTPUT_ROOT = EXPERIMENT_ROOT / "generated" / "rtgs_official_1view"


@dataclass(frozen=True)
class OfficialRtgsScene:
    scene_name: str
    scene_paths: RtgsScenePaths
    source_path: Path
    config: Mapping[str, Any]
    cfg_args: argparse.Namespace
    model_args: SimpleNamespace
    pipeline_args: SimpleNamespace
    optimization_args: SimpleNamespace
    gaussians: Any
    scene: Any
    checkpoint_path: Path
    iteration: int
    rtgs_code_root: Path
    rtgs_git_commit: str | None
    time_duration: list[float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render one view through the official RTGS Scene/GaussianModel/render flow")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="RTGS scene output directory")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Checkpoint file or path relative to --model-path")
    parser.add_argument("--rtgs-code-root", default=str(DEFAULT_RTGS_CODE_ROOT), help="Cloned 4d-gaussian-splatting code root")
    parser.add_argument("--dataset-kind", choices=("dnerf", "n3dv"), required=True, help="Dataset family; no implicit fallback")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DNERF_ROOT), help="dnerf dataset root")
    parser.add_argument("--n3dv-root", default=str(DEFAULT_N3DV_ROOT), help="N3DV dataset root")
    parser.add_argument("--source-path", default=None, help="Explicit RTGS source_path override for Scene construction")
    parser.add_argument("--config", default=None, help="RTGS YAML config override")
    parser.add_argument("--output-dir", default=None, help="Output directory; defaults under experiment/generated/rtgs_official_1view")
    parser.add_argument("--run-label", default=None, help="Stable output leaf name; omitted labels get a timestamp suffix")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--n3dv-frame-index", type=int, default=0, help="Recorded for N3DV manifests; Scene camera order remains official")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint-load-device",
        default="cpu",
        help="Device used for torch.load map_location before moving the GaussianModel tensors",
    )
    parser.add_argument("--background", choices=("auto", "black", "white"), default="auto")
    parser.add_argument("--no-ssim", action="store_true", help="Skip optional torchmetrics SSIM computation")
    return parser


def default_official_output_path(
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
    return DEFAULT_OFFICIAL_OUTPUT_ROOT / scene_name / leaf


def lkg_display_status(*, connected: bool, image_path: Path | str) -> dict[str, Any]:
    return {
        "connected": bool(connected),
        "validation_mode": "direct_display" if connected else "image_artifact_only",
        "single_view_image_path": str(Path(image_path)),
    }


def install_official_rtgs_code_root(code_root: Path | str) -> Path:
    root = Path(code_root).expanduser().resolve()
    required = [
        root / "scene" / "gaussian_model.py",
        root / "scene" / "__init__.py",
        root / "gaussian_renderer" / "__init__.py",
        root / "arguments" / "__init__.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"RTGS code root is incomplete: {root}; missing {missing}")

    _clear_conflicting_modules(root, ("scene", "arguments", "gaussian_renderer", "utils"))
    root_s = str(root)
    if root_s in sys.path:
        sys.path.remove(root_s)
    sys.path.insert(0, root_s)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(Path("/tmp") / "torch_extensions"))
    _install_pointops_import_stub_if_needed()
    return root


def resolve_official_scene_paths(
    model_path: Path | str,
    *,
    rtgs_code_root: Path | str,
    dataset_kind: str,
    dataset_root: Path | str,
    n3dv_root: Path | str,
    source_path: Path | str | None,
    config_path: Path | str | None,
) -> tuple[RtgsScenePaths, Path]:
    scene_name = Path(model_path).expanduser().name
    rtgs_root = Path(rtgs_code_root).expanduser()
    kind = str(dataset_kind)
    if kind == "dnerf":
        dataset_path = Path(dataset_root).expanduser() / scene_name
        default_config = rtgs_root / "configs" / "dnerf" / f"{scene_name}.yaml"
    elif kind == "n3dv":
        dataset_name = _N3DV_SCENE_ALIASES.get(scene_name, scene_name)
        dataset_path = Path(n3dv_root).expanduser() / dataset_name
        default_config = rtgs_root / "configs" / "dynerf" / f"{scene_name}.yaml"
    else:
        raise ValueError(f"unsupported dataset kind: {dataset_kind!r}")

    selected_config = Path(config_path).expanduser() if config_path else default_config
    if not selected_config.is_file():
        raise FileNotFoundError(f"RTGS config not found: {selected_config}")

    scene_paths = RtgsScenePaths(
        scene_name=scene_name,
        dataset_path=dataset_path,
        config_path=selected_config,
        dataset_kind=kind,
    )
    selected_source = Path(source_path).expanduser() if source_path else _official_scene_source_path(scene_paths)
    if not selected_source.exists():
        raise FileNotFoundError(f"RTGS source_path not found: {selected_source}")
    validate_official_scene_source_path(selected_source)
    return scene_paths, selected_source


def validate_official_scene_source_path(source_path: Path | str) -> None:
    path = Path(source_path).expanduser()
    if (path / "transforms_train.json").is_file():
        return
    if (path / "sparse").exists():
        return
    raise ValueError(
        "RTGS official Scene source_path must contain transforms_train.json for Blender/dnerf "
        f"or sparse for Colmap/N3DV: {path}"
    )


def add_rtgs_train_top_level_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")


def merge_rtgs_config_into_args(args: Any, config: Mapping[str, Any]) -> None:
    def recursive_merge(key: str, host: Mapping[str, Any]) -> None:
        value = host[key]
        if isinstance(value, Mapping):
            for nested_key in value:
                recursive_merge(str(nested_key), value)
            return
        if not hasattr(args, key):
            raise AttributeError(f"unknown RTGS config key: {key}")
        setattr(args, key, value)

    for key in config:
        recursive_merge(str(key), config)


def patch_colmap_camera_info_depth_default(readers_module: Any | None = None) -> None:
    if readers_module is None:
        import scene.dataset_readers as readers_module

    original_camera_info = readers_module.CameraInfo
    if getattr(original_camera_info, "_lkg_depth_default_patch", False):
        return

    def camera_info_with_depth_default(*args: Any, **kwargs: Any):
        kwargs.setdefault("depth", None)
        return original_camera_info(*args, **kwargs)

    camera_info_with_depth_default._lkg_depth_default_patch = True  # type: ignore[attr-defined]
    camera_info_with_depth_default._lkg_original_camera_info = original_camera_info  # type: ignore[attr-defined]
    camera_info_with_depth_default._fields = getattr(original_camera_info, "_fields", ())  # type: ignore[attr-defined]
    readers_module.CameraInfo = camera_info_with_depth_default


def _official_scene_source_path(scene_paths: RtgsScenePaths) -> Path:
    if scene_paths.dataset_kind == "n3dv":
        colmap_path = scene_paths.dataset_path / "colmap"
        if (colmap_path / "sparse").exists():
            return colmap_path
        if (scene_paths.dataset_path / "sparse").exists():
            return scene_paths.dataset_path
    return _camera_source_path(scene_paths)


def seed_official_scene_shuffle() -> None:
    random.seed(0)


def load_official_rtgs_scene(args: argparse.Namespace) -> OfficialRtgsScene:
    import torch

    rtgs_code_root = install_official_rtgs_code_root(args.rtgs_code_root)
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from scene import Scene
    from scene.gaussian_model import GaussianModel

    scene_paths, source_path = resolve_official_scene_paths(
        args.model_path,
        rtgs_code_root=rtgs_code_root,
        dataset_kind=args.dataset_kind,
        dataset_root=args.dataset_root,
        n3dv_root=args.n3dv_root,
        source_path=args.source_path,
        config_path=args.config,
    )
    config = load_rtgs_yaml_config(scene_paths.config_path)
    checkpoint_path = resolve_checkpoint_path(args.model_path, args.checkpoint)
    checkpoint_tuple = torch.load(checkpoint_path, map_location=str(args.checkpoint_load_device), weights_only=False)
    model_params, iteration = checkpoint_tuple

    parser = argparse.ArgumentParser(add_help=False)
    model_group = ModelParams(parser)
    optimization_group = OptimizationParams(parser)
    pipeline_group = PipelineParams(parser)
    add_rtgs_train_top_level_args(parser)
    defaults = parser.parse_args([])
    merge_rtgs_config_into_args(defaults, config)
    defaults.config = str(scene_paths.config_path)
    defaults.model_path = str(Path(args.model_path).expanduser())
    defaults.source_path = str(source_path)
    defaults.data_device = str(args.device)

    model_args = model_group.extract(defaults)
    pipeline_args = pipeline_group.extract(defaults)
    optimization_args = optimization_group.extract(defaults)

    time_duration = [float(value) for value in defaults.time_duration]
    if int(getattr(model_args, "frame_ratio", 1)) > 1:
        frame_ratio = float(model_args.frame_ratio)
        time_duration = [time_duration[0] / frame_ratio, time_duration[1] / frame_ratio]

    gaussians = GaussianModel(
        int(getattr(model_args, "sh_degree", 3)),
        gaussian_dim=int(defaults.gaussian_dim),
        time_duration=time_duration,
        rot_4d=bool(defaults.rot_4d),
        force_sh_3d=bool(defaults.force_sh_3d),
        sh_degree_t=2 if bool(getattr(pipeline_args, "eval_shfs_4d", True)) else 0,
        prefilter_var=float(getattr(model_args, "prefilter_var", -1.0)),
    )

    scene_load_iteration = int(iteration) if int(iteration) > 0 else 1
    seed_official_scene_shuffle()
    patch_colmap_camera_info_depth_default()
    with _skip_scene_model_population(type(gaussians)):
        scene = Scene(
            model_args,
            gaussians,
            load_iteration=scene_load_iteration,
            shuffle=True,
            num_pts=int(defaults.num_pts),
            num_pts_ratio=float(defaults.num_pts_ratio),
            time_duration=time_duration,
        )

    gaussians.restore(model_params, None)
    _move_gaussian_model_to_device(gaussians, str(args.device))

    return OfficialRtgsScene(
        scene_name=scene_paths.scene_name,
        scene_paths=scene_paths,
        source_path=source_path,
        config=config,
        cfg_args=defaults,
        model_args=model_args,
        pipeline_args=pipeline_args,
        optimization_args=optimization_args,
        gaussians=gaussians,
        scene=scene,
        checkpoint_path=checkpoint_path,
        iteration=int(iteration),
        rtgs_code_root=rtgs_code_root,
        rtgs_git_commit=_git_commit(rtgs_code_root),
        time_duration=time_duration,
    )


def render_official_rtgs_1view(args: argparse.Namespace) -> int:
    import torch

    if not str(args.device).startswith("cuda"):
        raise RuntimeError("RTGS official renderer requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot run RTGS official renderer")

    official = load_official_rtgs_scene(args)
    from gaussian_renderer import render

    camera_dataset = official.scene.getTestCameras() if args.split == "test" else official.scene.getTrainCameras()
    if int(args.camera_index) < 0 or int(args.camera_index) >= len(camera_dataset):
        raise IndexError(f"camera index {args.camera_index} out of range for {args.split} split with {len(camera_dataset)} cameras")
    gt, camera = camera_dataset[int(args.camera_index)]
    gt = gt.to(device=args.device, non_blocking=True).contiguous()
    camera_cuda = camera.cuda()
    pipe = _pipeline_namespace_for_model(vars(official.pipeline_args), official.gaussians)
    background = _background_tensor(args.background, official.model_args, args.device)
    ssim_metric = None if args.no_ssim else _make_ssim_metric(args.device)

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        rtgs_render = render(camera_cuda, official.gaussians, pipe, background)["render"].detach().clamp(0.0, 1.0).contiguous()
    torch.cuda.synchronize()
    render_ms = (time.perf_counter() - start) * 1000.0

    timestamp = float(getattr(camera, "timestamp", 0.0))
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_official_output_path(
        args.model_path,
        split=args.split,
        camera_index=args.camera_index,
        timestamp=timestamp,
        run_label=args.run_label,
    )
    render_path = output_dir / "rtgs_official_render.png"
    manifest = _build_manifest(
        args=args,
        official=official,
        camera=camera,
        pipe=pipe,
        gt=gt,
        rtgs_render=rtgs_render,
        render_ms=render_ms,
        ssim_metric=ssim_metric,
        render_path=render_path,
    )
    save_official_outputs(output_dir=output_dir, gt=gt, rtgs_render=rtgs_render, manifest=manifest)
    print(
        f"Wrote {output_dir} ({int(camera.image_width)}x{int(camera.image_height)}, "
        f"rtgs_vs_gt_psnr={manifest['metrics']['rtgs_vs_gt']['psnr']:.3f})",
        file=sys.stderr,
    )
    return 0


def save_official_outputs(*, output_dir: Path, gt: Any, rtgs_render: Any, manifest: Mapping[str, Any]) -> None:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    gt_np = tensor_to_hwc_uint8(gt)
    rtgs_np = tensor_to_hwc_uint8(rtgs_render)
    diff_np = _diff_image(rtgs_np, gt_np)

    Image.fromarray(gt_np, mode="RGB").save(output_dir / "gt.png")
    Image.fromarray(rtgs_np, mode="RGB").save(output_dir / "rtgs_official_render.png")
    Image.fromarray(np.concatenate([gt_np, rtgs_np, diff_np], axis=1), mode="RGB").save(output_dir / "comparison.png")
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_manifest(
    *,
    args: argparse.Namespace,
    official: OfficialRtgsScene,
    camera: Any,
    pipe: Any,
    gt: Any,
    rtgs_render: Any,
    render_ms: float,
    ssim_metric: Any,
    render_path: Path,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "rtgs_official_1view",
        "scene": official.scene_name,
        "dataset_kind": official.scene_paths.dataset_kind,
        "dataset_path": str(official.scene_paths.dataset_path),
        "source_path": str(official.source_path),
        "config_path": str(official.scene_paths.config_path),
        "model_path": str(Path(args.model_path).expanduser()),
        "checkpoint_path": str(official.checkpoint_path),
        "iteration": int(official.iteration),
        "split": args.split,
        "camera_index": int(args.camera_index),
        "n3dv_frame_index": int(args.n3dv_frame_index),
        "timestamp": float(getattr(camera, "timestamp", 0.0)),
        "width": int(camera.image_width),
        "height": int(camera.image_height),
        "gaussians_total": int(official.gaussians.get_xyz.shape[0]),
        "active_sh_degree": int(official.gaussians.active_sh_degree),
        "active_sh_degree_t": int(getattr(official.gaussians, "active_sh_degree_t", 0)),
        "gaussian_dim": int(getattr(official.gaussians, "gaussian_dim", 3)),
        "time_duration": [float(value) for value in official.time_duration],
        "rtgs_code_root": str(official.rtgs_code_root),
        "rtgs_git_commit": official.rtgs_git_commit,
        "rtgs_render_ms": float(render_ms),
        "scene_uses_original_scene": True,
        "scene_model_population": "skipped_then_checkpoint_restored",
        "pipeline": vars(pipe).copy(),
        "model_params": vars(official.model_args).copy(),
        "optimization_params": vars(official.optimization_args).copy(),
        "output_files": {
            "gt": "gt.png",
            "render": "rtgs_official_render.png",
            "comparison": "comparison.png",
            "metrics": "metrics.json",
        },
        "lkg_display": lkg_display_status(connected=False, image_path=render_path),
        "metrics": {
            "rtgs_vs_gt": compute_pair_metrics(rtgs_render, gt, ssim_metric=ssim_metric),
        },
        "args": dict(vars(args)),
    }
    manifest.update(_camera_manifest_fields(camera))
    return manifest


@contextmanager
def _skip_scene_model_population(model_cls: type[Any]):
    missing = object()
    originals = {
        "create_from_pcd": getattr(model_cls, "create_from_pcd", missing),
        "create_from_pth": getattr(model_cls, "create_from_pth", missing),
        "load_ply": getattr(model_cls, "load_ply", missing),
    }

    def _skip(self: Any, *_args: Any, **_kwargs: Any) -> None:
        if len(_args) >= 2:
            self.spatial_lr_scale = _args[1]

    model_cls.create_from_pcd = _skip
    model_cls.create_from_pth = _skip
    model_cls.load_ply = _skip
    try:
        yield
    finally:
        for name, original in originals.items():
            if original is missing:
                delattr(model_cls, name)
            else:
                setattr(model_cls, name, original)


def _move_gaussian_model_to_device(model: Any, device: str) -> None:
    import torch

    for key, value in list(model.__dict__.items()):
        if isinstance(value, torch.nn.Parameter):
            model.__dict__[key] = torch.nn.Parameter(value.detach().to(device), requires_grad=value.requires_grad)
        elif torch.is_tensor(value):
            model.__dict__[key] = value.to(device)


def _git_commit(root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def _diff_image(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"cannot build diff for mismatched images: {a.shape} vs {b.shape}")
    return np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * 4, 0, 255).astype(np.uint8)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return render_official_rtgs_1view(args)
