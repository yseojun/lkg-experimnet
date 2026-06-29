from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from lkg_experiment.coherent_default.coherent_raster_experiment import tensor_to_hwc_uint8
from lkg_experiment.rtgs_coherent.cli import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DNERF_ROOT,
    DEFAULT_GENERATED_ROOT,
    DEFAULT_MODEL_PATH,
    DEFAULT_N3DV_ROOT,
    DEFAULT_RTGS_CODE_ROOT,
    _background_tensor,
    _camera_manifest_fields,
    _json_ready,
    _load_blender_camera,
    _make_ssim_metric,
    _pipeline_namespace_for_model,
    compute_pair_metrics,
    load_rtgs_camera,
    load_rtgs_checkpoint,
    render_rtgs_original,
)


DEFAULT_BASIC_OUTPUT_ROOT = DEFAULT_GENERATED_ROOT / "rtgs_basic_1view"


@dataclass(frozen=True)
class OfficialReferencePaths:
    render: Path
    gt: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render one view through the original RTGS renderer only")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="RTGS scene output directory")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Checkpoint file or path relative to --model-path")
    parser.add_argument("--rtgs-code-root", default=str(DEFAULT_RTGS_CODE_ROOT), help="Cloned 4d-gaussian-splatting code root")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DNERF_ROOT), help="dnerf dataset root")
    parser.add_argument("--n3dv-root", default=str(DEFAULT_N3DV_ROOT), help="N3DV dataset root")
    parser.add_argument(
        "--source-path-override",
        default=None,
        help="Optional RTGS-converted dataset path with transforms_train/test.json; useful for checking official N3DV camera loading",
    )
    parser.add_argument("--config", default=None, help="RTGS YAML config override")
    parser.add_argument("--output-dir", default=None, help=f"Output directory; defaults under {DEFAULT_BASIC_OUTPUT_ROOT}")
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--n3dv-frame-index", type=int, default=0, help="N3DV dynamic frame index used with cameras.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint-load-device",
        default="cpu",
        help="Device used for torch.load map_location; default keeps checkpoint deserialization off CUDA before moving tensors",
    )
    parser.add_argument(
        "--rtgs-rotation-convention",
        choices=("legacy", "current"),
        default="current",
        help="Diagnostic switch for 4D rotation convention; current matches the checked-out RTGS renderer",
    )
    parser.add_argument("--background", choices=("auto", "black", "white"), default="auto")
    parser.add_argument("--compute-cov3d-python", choices=("config", "on", "off"), default="config")
    parser.add_argument("--convert-shs-python", choices=("config", "on", "off"), default="config")
    parser.add_argument("--no-ssim", action="store_true", help="Skip optional torchmetrics SSIM computation")
    parser.add_argument("--compare-official", dest="compare_official", action="store_true", default=True)
    parser.add_argument("--no-compare-official", dest="compare_official", action="store_false")
    parser.add_argument(
        "--official-output-index",
        type=int,
        default=None,
        help="Index under test/RESULT/{renders,gt}; inferred for N3DV test cam0 when omitted",
    )
    parser.add_argument("--official-result-name", default="ours_best")
    parser.add_argument("--diagnostic-sample-count", type=int, default=4096)
    parser.add_argument("--write-diagnostics", dest="write_diagnostics", action="store_true", default=True)
    parser.add_argument("--no-write-diagnostics", dest="write_diagnostics", action="store_false")
    return parser


def default_basic_output_path(model_path: Path | str, *, split: str, camera_index: int, timestamp: float) -> Path:
    scene_name = Path(model_path).expanduser().name
    return DEFAULT_BASIC_OUTPUT_ROOT / scene_name / f"{scene_name}_t{float(timestamp):.6f}_{split}_{int(camera_index)}"


def n3dv_official_test_output_index(
    frame_index: int,
    *,
    frame_count: int = 300,
    train_camera_count: int = 5100,
    seed: int = 0,
) -> int:
    if int(frame_index) < 0 or int(frame_index) >= int(frame_count):
        raise IndexError(f"frame_index {frame_index} out of range for {frame_count} N3DV test frames")
    rng = random.Random(int(seed))
    train_indices = list(range(int(train_camera_count)))
    rng.shuffle(train_indices)
    test_indices = list(range(int(frame_count)))
    rng.shuffle(test_indices)
    return test_indices.index(int(frame_index))


def official_reference_paths(model_path: Path | str, *, output_index: int, result_name: str = "ours_best") -> OfficialReferencePaths:
    root = Path(model_path).expanduser() / "test" / str(result_name)
    filename = f"{int(output_index):05d}.png"
    return OfficialReferencePaths(render=root / "renders" / filename, gt=root / "gt" / filename)


def render_basic_rtgs_1view(args: argparse.Namespace) -> int:
    import torch

    if not str(args.device).startswith("cuda"):
        raise RuntimeError("RTGS renderer requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot run RTGS renderer")

    checkpoint = load_rtgs_checkpoint(
        model_path=args.model_path,
        checkpoint=args.checkpoint,
        rtgs_code_root=args.rtgs_code_root,
        dataset_root=args.dataset_root,
        n3dv_root=args.n3dv_root,
        config_path=args.config,
        device=args.device,
        checkpoint_load_device=args.checkpoint_load_device,
        rotation_convention=args.rtgs_rotation_convention,
    )
    pipe = _pipeline_namespace_for_model(checkpoint.config.get("PipelineParams", {}), checkpoint.model)
    _apply_pipeline_overrides(pipe, compute_cov3d=args.compute_cov3d_python, convert_shs=args.convert_shs_python)
    background = _background_tensor(args.background, checkpoint.cfg_args, args.device)
    ssim_metric = None if args.no_ssim else _make_ssim_metric(args.device)

    gt, camera = load_basic_rtgs_camera(args=args, checkpoint=checkpoint)
    gt = gt.to(device=args.device, non_blocking=True).contiguous()

    torch.cuda.synchronize()
    start = time.perf_counter()
    rtgs_render = render_rtgs_original(camera, checkpoint.model, pipe, background)
    torch.cuda.synchronize()
    rtgs_ms = (time.perf_counter() - start) * 1000.0

    timestamp = float(camera.timestamp)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_basic_output_path(
        args.model_path,
        split=args.split,
        camera_index=args.camera_index,
        timestamp=timestamp,
    )

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "rtgs_basic_1view",
        "scene": checkpoint.scene_paths.scene_name,
        "dataset_kind": checkpoint.scene_paths.dataset_kind,
        "dataset_path": str(checkpoint.scene_paths.dataset_path),
        "config_path": str(checkpoint.scene_paths.config_path),
        "model_path": str(Path(args.model_path).expanduser()),
        "checkpoint_path": str(checkpoint.checkpoint_path),
        "iteration": checkpoint.iteration,
        "split": args.split,
        "camera_index": int(args.camera_index),
        "timestamp": timestamp,
        "width": int(camera.image_width),
        "height": int(camera.image_height),
        "gaussians_total": int(checkpoint.model.get_xyz.shape[0]),
        "active_sh_degree": int(checkpoint.model.active_sh_degree),
        "active_sh_degree_t": int(checkpoint.model.active_sh_degree_t),
        "rtgs_rotation_convention": str(checkpoint.model.rotation_convention),
        "rtgs_render_ms": float(rtgs_ms),
        "pipeline": vars(pipe).copy(),
        "metrics": {
            "rtgs_vs_gt": compute_pair_metrics(rtgs_render, gt, ssim_metric=ssim_metric),
        },
        "args": dict(vars(args)),
    }
    manifest.update(_camera_manifest_fields(camera))
    if bool(args.write_diagnostics):
        manifest["camera_diagnostics"] = camera_diagnostics(camera)
        manifest["model_diagnostics"] = model_diagnostics(checkpoint.model, sample_count=int(args.diagnostic_sample_count))

    official_render = None
    official_gt = None
    if bool(args.compare_official):
        official_index, official_reason = _resolve_official_output_index(args, checkpoint, camera)
        manifest["official_output_index"] = official_index
        manifest["official_output_index_reason"] = official_reason
        if official_index is not None:
            official_paths = official_reference_paths(
                args.model_path,
                output_index=int(official_index),
                result_name=args.official_result_name,
            )
            manifest["official_render_path"] = str(official_paths.render)
            manifest["official_gt_path"] = str(official_paths.gt)
            if official_paths.render.is_file() and official_paths.gt.is_file():
                official_render = _load_image_tensor(official_paths.render, device=args.device, size=(int(camera.image_width), int(camera.image_height)))
                official_gt = _load_image_tensor(official_paths.gt, device=args.device, size=(int(camera.image_width), int(camera.image_height)))
                manifest["metrics"]["rtgs_vs_official_render"] = compute_pair_metrics(rtgs_render, official_render, ssim_metric=ssim_metric)
                manifest["metrics"]["official_render_vs_gt"] = compute_pair_metrics(official_render, gt, ssim_metric=ssim_metric)
                manifest["metrics"]["gt_vs_official_gt"] = compute_pair_metrics(gt, official_gt, ssim_metric=ssim_metric)
            else:
                manifest["official_missing"] = [
                    str(path)
                    for path in (official_paths.render, official_paths.gt)
                    if not path.is_file()
                ]

    save_basic_outputs(
        output_dir=output_dir,
        gt=gt,
        rtgs_render=rtgs_render,
        manifest=manifest,
        official_render=official_render,
        official_gt=official_gt,
        write_diagnostics=bool(args.write_diagnostics),
    )
    print(
        f"Wrote {output_dir} ({int(camera.image_width)}x{int(camera.image_height)}, "
        f"rtgs_vs_gt_psnr={manifest['metrics']['rtgs_vs_gt']['psnr']:.3f})",
        file=sys.stderr,
    )
    return 0


def load_basic_rtgs_camera(*, args: argparse.Namespace, checkpoint: Any):
    if args.source_path_override:
        model_cfg = checkpoint.config.get("ModelParams", {})
        cfg_args = checkpoint.cfg_args
        if args.split == "all":
            raise ValueError("--source-path-override supports train/test splits, not all")
        return _load_blender_camera(
            source_path=Path(args.source_path_override).expanduser(),
            split=args.split,
            camera_index=args.camera_index,
            white_background=bool(model_cfg.get("white_background", getattr(cfg_args, "white_background", False))),
            resolution=int(model_cfg.get("resolution", getattr(cfg_args, "resolution", 2))),
            extension=str(model_cfg.get("extension", getattr(cfg_args, "extension", ".png"))),
            frame_ratio=int(model_cfg.get("frame_ratio", getattr(cfg_args, "frame_ratio", 1))),
            time_duration=checkpoint.model.time_duration,
            device=args.device,
        )
    return load_rtgs_camera(
        checkpoint=checkpoint,
        split=args.split,
        camera_index=args.camera_index,
        device=args.device,
        n3dv_frame_index=args.n3dv_frame_index,
    )


def camera_diagnostics(camera: Any) -> dict[str, Any]:
    return {
        "image_width": int(camera.image_width),
        "image_height": int(camera.image_height),
        "FoVx": float(getattr(camera, "FoVx", 0.0)),
        "FoVy": float(getattr(camera, "FoVy", 0.0)),
        "fl_x": float(getattr(camera, "fl_x", -1.0)),
        "fl_y": float(getattr(camera, "fl_y", -1.0)),
        "cx": float(getattr(camera, "cx", -1.0)),
        "cy": float(getattr(camera, "cy", -1.0)),
        "uses_center_shift_projection": bool(
            float(getattr(camera, "cx", -1.0)) > 0.0
            and float(getattr(camera, "cy", -1.0)) > 0.0
            and float(getattr(camera, "fl_x", -1.0)) > 0.0
            and float(getattr(camera, "fl_y", -1.0)) > 0.0
        ),
        "camera_center": _tensor_to_list(getattr(camera, "camera_center", None)),
        "world_view_transform": _tensor_to_list(getattr(camera, "world_view_transform", None)),
        "projection_matrix": _tensor_to_list(getattr(camera, "projection_matrix", None)),
        "full_proj_transform": _tensor_to_list(getattr(camera, "full_proj_transform", None)),
    }


def model_diagnostics(model: Any, *, sample_count: int = 4096) -> dict[str, Any]:
    features = getattr(model, "get_features", None)
    max_channels = getattr(model, "get_max_sh_channels", None)
    feature_shape = list(features.shape) if hasattr(features, "shape") else None
    max_channels_value = int(max_channels) if max_channels is not None else None
    diagnostics = {
        "gaussian_dim": int(getattr(model, "gaussian_dim", 3)),
        "rot_4d": bool(getattr(model, "rot_4d", False)),
        "force_sh_3d": bool(getattr(model, "force_sh_3d", False)),
        "prefilter_var": float(getattr(model, "prefilter_var", -1.0)),
        "time_duration": [float(value) for value in getattr(model, "time_duration", [])],
        "active_sh_degree": int(getattr(model, "active_sh_degree", 0)),
        "active_sh_degree_t": int(getattr(model, "active_sh_degree_t", 0)),
        "max_sh_degree": int(getattr(model, "max_sh_degree", 0)),
        "max_sh_degree_t": int(getattr(model, "max_sh_degree_t", 0)),
        "get_max_sh_channels": max_channels_value,
        "features_shape": feature_shape,
        "feature_channel_match": bool(feature_shape and max_channels_value is not None and feature_shape[1] == max_channels_value),
        "xyz": tensor_summary(getattr(model, "_xyz", None), sample_count=sample_count),
        "features_dc": tensor_summary(getattr(model, "_features_dc", None), sample_count=sample_count),
        "features_rest": tensor_summary(getattr(model, "_features_rest", None), sample_count=sample_count),
        "scaling": tensor_summary(getattr(model, "_scaling", None), sample_count=sample_count),
        "rotation": tensor_summary(getattr(model, "_rotation", None), sample_count=sample_count),
        "opacity": tensor_summary(getattr(model, "_opacity", None), sample_count=sample_count),
        "t": tensor_summary(getattr(model, "_t", None), sample_count=sample_count),
        "scaling_t": tensor_summary(getattr(model, "_scaling_t", None), sample_count=sample_count),
        "rotation_r": tensor_summary(getattr(model, "_rotation_r", None), sample_count=sample_count),
        "env_map": tensor_summary(getattr(model, "env_map", None), sample_count=sample_count),
    }
    return diagnostics


def tensor_summary(value: Any, *, sample_count: int = 4096) -> dict[str, Any] | None:
    import torch

    if value is None or not torch.is_tensor(value):
        return None
    detached = value.detach()
    summary: dict[str, Any] = {
        "shape": [int(dim) for dim in detached.shape],
        "dtype": str(detached.dtype).replace("torch.", ""),
        "device": str(detached.device),
        "numel": int(detached.numel()),
    }
    if detached.numel() == 0:
        return summary
    flat = detached.reshape(-1)
    max_samples = max(1, int(sample_count))
    if flat.numel() > max_samples:
        sample_step = max(1, int(flat.numel()) // max_samples)
        flat = flat[::sample_step][:max_samples]
        summary["sampled_numel"] = int(flat.numel())
        summary["sample_strategy"] = "stride_slice"
        summary["sample_step"] = int(sample_step)
    values = flat.float()
    summary.update(
        {
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "mean": float(values.mean().item()),
        }
    )
    return summary


def _tensor_to_list(value: Any) -> Any:
    import torch

    if value is None or not torch.is_tensor(value):
        return None
    return value.detach().cpu().tolist()


def _apply_pipeline_overrides(pipe: Any, *, compute_cov3d: str, convert_shs: str) -> None:
    if compute_cov3d != "config":
        pipe.compute_cov3D_python = compute_cov3d == "on"
    if convert_shs != "config":
        pipe.convert_SHs_python = convert_shs == "on"


def save_basic_outputs(
    *,
    output_dir: Path,
    gt: Any,
    rtgs_render: Any,
    manifest: Mapping[str, Any],
    official_render: Any | None = None,
    official_gt: Any | None = None,
    write_diagnostics: bool = True,
) -> None:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    gt_np = tensor_to_hwc_uint8(gt)
    rtgs_np = tensor_to_hwc_uint8(rtgs_render)
    diff_np = _diff_image(rtgs_np, gt_np)

    Image.fromarray(gt_np, mode="RGB").save(output_dir / "gt.png")
    Image.fromarray(rtgs_np, mode="RGB").save(output_dir / "rtgs_render.png")
    Image.fromarray(np.concatenate([gt_np, rtgs_np, diff_np], axis=1), mode="RGB").save(output_dir / "comparison.png")

    if official_render is not None:
        official_np = tensor_to_hwc_uint8(official_render)
        Image.fromarray(official_np, mode="RGB").save(output_dir / "official_render.png")
        columns = [gt_np, rtgs_np, official_np, _diff_image(rtgs_np, official_np)]
        Image.fromarray(np.concatenate(columns, axis=1), mode="RGB").save(output_dir / "comparison_with_official.png")

    if official_gt is not None:
        Image.fromarray(tensor_to_hwc_uint8(official_gt), mode="RGB").save(output_dir / "official_gt.png")

    if write_diagnostics:
        (output_dir / "metrics.json").write_text(
            json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _resolve_official_output_index(args: argparse.Namespace, checkpoint: Any, camera: Any) -> tuple[int | None, str]:
    if args.official_output_index is not None:
        return int(args.official_output_index), "cli"
    if checkpoint.scene_paths.dataset_kind != "n3dv" or args.split != "test":
        return None, "not inferred outside N3DV test split"
    camera_fields = _camera_manifest_fields(camera)
    camera_label = camera_fields.get("n3dv_camera_label")
    frame_index = camera_fields.get("n3dv_frame_index")
    if camera_label not in {"cam00", "cam0"}:
        return None, f"not inferred for N3DV non-test camera label {camera_label!r}"
    if frame_index is None:
        return None, "missing N3DV frame index"
    return n3dv_official_test_output_index(int(frame_index)), "n3dv safe_state(seed=0) train-then-test shuffle"


def _load_image_tensor(path: Path, *, device: str, size: tuple[int, int]):
    import torch
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != size:
            rgb = rgb.resize(size, Image.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(device=device).contiguous()


def _diff_image(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"cannot build diff for mismatched images: {a.shape} vs {b.shape}")
    return np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * 4, 0, 255).astype(np.uint8)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return render_basic_rtgs_1view(args)
