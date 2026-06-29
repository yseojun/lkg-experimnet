from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from lkg_experiment.coherent_default.coherent_raster_experiment import tensor_to_hwc_uint8
from lkg_experiment.rtgs_coherent.cli import (
    DEFAULT_GENERATED_ROOT,
    RtgsSnapshot,
    _install_gsplat_root,
    _json_ready,
    compute_pair_metrics,
    materialize_rtgs_snapshot,
    render_rtgs_coherent,
    rtgs_camera_to_gsplat_inputs,
)
from lkg_experiment.rtgs_coherent.official_1view import (
    _diff_image,
    camera_render_contract_values,
    render_contract_tensor_shapes,
    build_parser as build_official_parser,
    prepare_official_rtgs_1view,
)


DEFAULT_CR_OUTPUT_ROOT = DEFAULT_GENERATED_ROOT / "rtgs_cr_1view"


@dataclass(frozen=True)
class CrSnapshotForDiagnostics:
    means: Any
    covars: Any
    opacities: Any
    colors: Any


class SnapshotGaussianProxy:
    def __init__(self, snapshot: CrSnapshotForDiagnostics, *, active_sh_degree: int) -> None:
        self.snapshot = snapshot
        self.active_sh_degree = int(active_sh_degree)
        self.active_sh_degree_t = 0
        self.gaussian_dim = 3
        self.rot_4d = False
        self.force_sh_3d = True
        self.time_duration = [0.0, 1.0]
        self.env_map = None

    @property
    def get_xyz(self):
        return self.snapshot.means

    @property
    def get_opacity(self):
        opacities = self.snapshot.opacities
        if getattr(opacities, "ndim", 1) == 1:
            return opacities.reshape(-1, 1)
        return opacities

    def get_covariance(self, scaling_modifier: float = 1.0):
        if float(scaling_modifier) == 1.0:
            return self.snapshot.covars
        return self.snapshot.covars * float(scaling_modifier)


def build_parser() -> argparse.ArgumentParser:
    parser = build_official_parser()
    parser.description = "Render one RTGS view through official RTGS and CoherentRaster for comparison"
    parser.add_argument(
        "--compare-official",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render and save the official RTGS baseline next to the CR output",
    )
    parser.add_argument(
        "--normalize-explicit-intrinsics-fov",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic mode: recompute non-positive FoVx/FoVy from positive fl_x/fl_y for CR/snapshot renders. "
            "This is not the default because it changes the RTGS official covariance behavior."
        ),
    )
    parser.add_argument(
        "--rtgs-compat-projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When explicit intrinsics coexist with non-positive FoV sentinels, adapt CR means/K so gsplat projection "
            "matches the RTGS official mean/covariance split used by N3DV checkpoints."
        ),
    )
    parser.add_argument("--gsplat-root", default=str(Path(__file__).resolve().parents[4] / "gsplat"))
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=100.0)
    parser.add_argument("--camera-model", choices=("pinhole", "ortho", "fisheye"), default="pinhole")
    return parser


def default_cr_output_path(
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
    return DEFAULT_CR_OUTPUT_ROOT / scene_name / leaf


def build_one_view_viewpoint_index(*, width: int, height: int) -> np.ndarray:
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("width and height must be positive")
    return np.zeros((int(height), int(width), 3), dtype=np.uint32)


def normalize_explicit_intrinsics_fov(camera: Any, *, enabled: bool) -> tuple[Any, dict[str, Any]]:
    info = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": None,
        "before": {
            "FoVx": _optional_float(getattr(camera, "FoVx", None)),
            "FoVy": _optional_float(getattr(camera, "FoVy", None)),
            "fl_x": _optional_float(getattr(camera, "fl_x", None)),
            "fl_y": _optional_float(getattr(camera, "fl_y", None)),
            "cx": _optional_float(getattr(camera, "cx", None)),
            "cy": _optional_float(getattr(camera, "cy", None)),
        },
        "after": None,
    }
    if not enabled:
        info["reason"] = "disabled"
        info["after"] = dict(info["before"])
        return camera, info

    fl_x = _optional_float(getattr(camera, "fl_x", None))
    fl_y = _optional_float(getattr(camera, "fl_y", None))
    width = int(getattr(camera, "image_width", 0) or 0)
    height = int(getattr(camera, "image_height", 0) or 0)
    if fl_x is None or fl_y is None or fl_x <= 0.0 or fl_y <= 0.0 or width <= 0 or height <= 0:
        info["reason"] = "missing_positive_explicit_intrinsics"
        info["after"] = dict(info["before"])
        return camera, info

    fovx = _optional_float(getattr(camera, "FoVx", None))
    fovy = _optional_float(getattr(camera, "FoVy", None))
    if fovx is not None and fovy is not None and fovx > 0.0 and fovy > 0.0:
        info["reason"] = "fov_already_positive"
        info["after"] = dict(info["before"])
        return camera, info

    normalized = copy.deepcopy(camera)
    normalized.FoVx = 2.0 * math.atan(float(width) / (2.0 * float(fl_x)))
    normalized.FoVy = 2.0 * math.atan(float(height) / (2.0 * float(fl_y)))
    info["applied"] = True
    info["reason"] = "nonpositive_fov_recomputed_from_explicit_intrinsics"
    info["after"] = {
        "FoVx": float(normalized.FoVx),
        "FoVy": float(normalized.FoVy),
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": _optional_float(getattr(normalized, "cx", None)),
        "cy": _optional_float(getattr(normalized, "cy", None)),
    }
    return normalized, info


def adapt_snapshot_for_rtgs_compat_projection(
    snapshot: RtgsSnapshot,
    *,
    camera: Any,
    viewmat: Any,
    K: Any,
    enabled: bool,
) -> tuple[RtgsSnapshot, Any, dict[str, Any]]:
    import torch

    info = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": None,
        "before_K": _matrix_to_float_list(K),
        "after_K": None,
        "mean_camera_scale": None,
    }
    if not enabled:
        info["reason"] = "disabled"
        info["after_K"] = info["before_K"]
        return snapshot, K, info

    fl_x = _optional_float(getattr(camera, "fl_x", None))
    fl_y = _optional_float(getattr(camera, "fl_y", None))
    fovx = _optional_float(getattr(camera, "FoVx", None))
    fovy = _optional_float(getattr(camera, "FoVy", None))
    cx = _optional_float(getattr(camera, "cx", None))
    cy = _optional_float(getattr(camera, "cy", None))
    width = int(getattr(camera, "image_width", 0) or 0)
    height = int(getattr(camera, "image_height", 0) or 0)
    if fl_x is None or fl_y is None or fl_x <= 0.0 or fl_y <= 0.0 or width <= 0 or height <= 0:
        info["reason"] = "missing_positive_explicit_intrinsics"
        info["after_K"] = info["before_K"]
        return snapshot, K, info
    if fovx is None or fovy is None or fovx > 0.0 or fovy > 0.0:
        info["reason"] = "fov_not_nonpositive_sentinel"
        info["after_K"] = info["before_K"]
        return snapshot, K, info

    tan_x = math.tan(float(fovx) * 0.5)
    tan_y = math.tan(float(fovy) * 0.5)
    if abs(tan_x) < 1e-12 or abs(tan_y) < 1e-12:
        info["reason"] = "invalid_fov_tangent"
        info["after_K"] = info["before_K"]
        return snapshot, K, info

    fov_fx = float(width) / (2.0 * tan_x)
    fov_fy = float(height) / (2.0 * tan_y)
    scale_x = float(fl_x) / fov_fx
    scale_y = float(fl_y) / fov_fy

    viewmat_t = torch.as_tensor(viewmat, dtype=snapshot.means.dtype, device=snapshot.means.device)
    R = viewmat_t[:3, :3]
    t = viewmat_t[:3, 3]
    means_cam = (R @ snapshot.means.T).T + t
    means_cam = means_cam.clone()
    means_cam[:, 0] *= scale_x
    means_cam[:, 1] *= scale_y
    adapted_means = (torch.linalg.inv(R) @ (means_cam - t).T).T.contiguous()

    adapted_K = torch.as_tensor(K, dtype=snapshot.means.dtype, device=snapshot.means.device).clone()
    adapted_K[0, 0] = fov_fx
    adapted_K[1, 1] = fov_fy
    if cx is not None:
        adapted_K[0, 2] = cx
    if cy is not None:
        adapted_K[1, 2] = cy

    info["applied"] = True
    info["reason"] = "explicit_intrinsics_with_nonpositive_fov_sentinel"
    info["after_K"] = _matrix_to_float_list(adapted_K)
    info["mean_camera_scale"] = {"x": float(scale_x), "y": float(scale_y)}
    return (
        RtgsSnapshot(
            means=adapted_means,
            covars=snapshot.covars,
            opacities=snapshot.opacities,
            colors=snapshot.colors,
            mask=snapshot.mask,
        ),
        adapted_K.contiguous(),
        info,
    )


def snapshot_tensor_diagnostics(snapshot: CrSnapshotForDiagnostics) -> dict[str, Any]:
    import torch

    means = torch.as_tensor(snapshot.means)
    covars = torch.as_tensor(snapshot.covars)
    opacities = torch.as_tensor(snapshot.opacities)
    colors = torch.as_tensor(snapshot.colors)
    return {
        "means_shape": list(means.shape),
        "covars_shape": list(covars.shape),
        "opacities_shape": list(opacities.shape),
        "colors_shape": list(colors.shape),
        "tensor_nan_inf_counts": {
            "means_nan": int(torch.isnan(means).sum().item()),
            "means_inf": int(torch.isinf(means).sum().item()),
            "covars_nan": int(torch.isnan(covars).sum().item()),
            "covars_inf": int(torch.isinf(covars).sum().item()),
            "opacities_nan": int(torch.isnan(opacities).sum().item()),
            "opacities_inf": int(torch.isinf(opacities).sum().item()),
            "colors_nan": int(torch.isnan(colors).sum().item()),
            "colors_inf": int(torch.isinf(colors).sum().item()),
        },
        "opacity_min_max_mean": _finite_min_max_mean(opacities),
        "color_min_max_mean": _finite_min_max_mean(colors),
        "covariance_min_max_mean": _finite_min_max_mean(covars),
    }


def _finite_min_max_mean(tensor: Any) -> dict[str, float | None]:
    import torch

    values = torch.as_tensor(tensor).detach().float()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "mean": float(finite.mean().item()),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _matrix_to_float_list(value: Any) -> list[list[float]] | None:
    try:
        array = np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, dtype=np.float64)
    except Exception:
        return None
    if array.ndim != 2:
        return None
    return [[float(item) for item in row] for row in array.tolist()]


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return render_rtgs_cr_1view(args)


def render_rtgs_cr_1view(args: argparse.Namespace) -> int:
    import torch

    if not str(args.device).startswith("cuda"):
        raise RuntimeError("RTGS + CoherentRaster renderer requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot run RTGS + CoherentRaster renderer")

    runtime = prepare_official_rtgs_1view(args)
    from gaussian_renderer import render

    gt = runtime.gt.to(device=args.device, non_blocking=True).contiguous()
    official_camera_cuda = runtime.camera.cuda()
    cr_camera_cuda, fov_normalization = normalize_explicit_intrinsics_fov(
        official_camera_cuda,
        enabled=bool(args.normalize_explicit_intrinsics_fov),
    )
    timestamp = float(getattr(cr_camera_cuda, "timestamp", 0.0))
    width = int(cr_camera_cuda.image_width)
    height = int(cr_camera_cuda.image_height)

    official_render = None
    official_cr_camera_render = None
    official_ms = None
    official_cr_camera_ms = None
    if bool(args.compare_official):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            official_render = (
                render(official_camera_cuda, runtime.official.gaussians, runtime.pipe, runtime.background)["render"]
                .detach()
                .clamp(0.0, 1.0)
                .contiguous()
            )
        torch.cuda.synchronize()
        official_ms = (time.perf_counter() - start) * 1000.0
        if bool(fov_normalization["applied"]):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                official_cr_camera_render = (
                    render(cr_camera_cuda, runtime.official.gaussians, runtime.pipe, runtime.background)["render"]
                    .detach()
                    .clamp(0.0, 1.0)
                    .contiguous()
                )
            torch.cuda.synchronize()
            official_cr_camera_ms = (time.perf_counter() - start) * 1000.0

    snapshot = materialize_rtgs_snapshot(
        runtime.official.gaussians,
        timestamp=timestamp,
        camera_center=cr_camera_cuda.camera_center,
    )
    snapshot_proxy = SnapshotGaussianProxy(snapshot, active_sh_degree=int(runtime.official.gaussians.active_sh_degree))
    snapshot_pipe = SimpleNamespace(**vars(runtime.pipe))
    snapshot_pipe.compute_cov3D_python = True
    snapshot_pipe.convert_SHs_python = False
    snapshot_pipe.eval_shfs_4d = False
    snapshot_pipe.env_map_res = 0

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        snapshot_reference = (
            render(cr_camera_cuda, snapshot_proxy, snapshot_pipe, runtime.background, override_color=snapshot.colors)["render"]
            .detach()
            .clamp(0.0, 1.0)
            .contiguous()
        )
    torch.cuda.synchronize()
    snapshot_reference_ms = (time.perf_counter() - start) * 1000.0

    _install_gsplat_root(args.gsplat_root)
    viewmat, K = rtgs_camera_to_gsplat_inputs(cr_camera_cuda, device=args.device)
    cr_snapshot, cr_K, cr_projection_adapter = adapt_snapshot_for_rtgs_compat_projection(
        snapshot,
        camera=cr_camera_cuda,
        viewmat=viewmat,
        K=K,
        enabled=bool(args.rtgs_compat_projection) and not bool(fov_normalization["applied"]),
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    cr_render = render_rtgs_coherent(
        snapshot=cr_snapshot,
        viewmat=viewmat,
        K=cr_K,
        width=width,
        height=height,
        tile_size=int(args.tile_size),
        near_plane=float(args.near_plane),
        far_plane=float(args.far_plane),
        camera_model=str(args.camera_model),
        background=runtime.background,
        debug=False,
    )
    torch.cuda.synchronize()
    cr_ms = (time.perf_counter() - start) * 1000.0

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_cr_output_path(
        args.model_path,
        split=args.split,
        camera_index=args.camera_index,
        timestamp=timestamp,
        run_label=args.run_label,
    )
    manifest = _build_cr_manifest(
        args=args,
        runtime=runtime,
        official_camera=official_camera_cuda,
        cr_camera=cr_camera_cuda,
        gt=gt,
        official_render=official_render,
        official_cr_camera_render=official_cr_camera_render,
        snapshot_reference=snapshot_reference,
        cr_render=cr_render,
        snapshot=snapshot,
        cr_snapshot=cr_snapshot,
        official_ms=official_ms,
        official_cr_camera_ms=official_cr_camera_ms,
        snapshot_reference_ms=snapshot_reference_ms,
        cr_ms=cr_ms,
        fov_normalization=fov_normalization,
        cr_projection_adapter=cr_projection_adapter,
    )
    save_cr_outputs(
        output_dir=output_dir,
        gt=gt,
        official_render=official_render,
        official_cr_camera_render=official_cr_camera_render,
        snapshot_reference=snapshot_reference,
        cr_render=cr_render,
        manifest=manifest,
    )

    metrics = manifest["metrics"]
    official_vs_cr = metrics.get("official_cr_camera_vs_cr") or metrics.get("official_vs_cr", {})
    print(
        f"Wrote {output_dir} ({width}x{height}, "
        f"snapshot_vs_cr_psnr={metrics['snapshot_vs_cr']['psnr']:.3f}, "
        f"official_vs_cr_psnr={official_vs_cr.get('psnr', float('nan')):.3f})",
        file=sys.stderr,
    )
    return 0


def save_cr_outputs(
    *,
    output_dir: Path,
    gt: Any,
    official_render: Any | None,
    official_cr_camera_render: Any | None,
    snapshot_reference: Any,
    cr_render: Any,
    manifest: dict[str, Any],
) -> None:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    gt_np = tensor_to_hwc_uint8(gt)
    snapshot_np = tensor_to_hwc_uint8(snapshot_reference)
    cr_np = tensor_to_hwc_uint8(cr_render)

    Image.fromarray(gt_np, mode="RGB").save(output_dir / "gt.png")
    Image.fromarray(snapshot_np, mode="RGB").save(output_dir / "rtgs_snapshot_reference_render.png")
    Image.fromarray(cr_np, mode="RGB").save(output_dir / "rtgs_cr_render.png")
    Image.fromarray(_diff_image(cr_np, snapshot_np), mode="RGB").save(output_dir / "diff_snapshot_vs_cr.png")
    Image.fromarray(np.concatenate([gt_np, snapshot_np, cr_np, _diff_image(cr_np, snapshot_np)], axis=1), mode="RGB").save(
        output_dir / "comparison_snapshot_vs_cr.png"
    )

    if official_render is not None:
        official_np = tensor_to_hwc_uint8(official_render)
        Image.fromarray(official_np, mode="RGB").save(output_dir / "rtgs_official_render.png")
        Image.fromarray(_diff_image(snapshot_np, official_np), mode="RGB").save(output_dir / "diff_official_vs_snapshot.png")
        Image.fromarray(_diff_image(cr_np, official_np), mode="RGB").save(output_dir / "diff_official_vs_cr.png")
        Image.fromarray(
            np.concatenate([gt_np, official_np, snapshot_np, cr_np, _diff_image(cr_np, official_np)], axis=1),
            mode="RGB",
        ).save(output_dir / "comparison_official_snapshot_cr.png")

    if official_cr_camera_render is not None:
        official_cr_np = tensor_to_hwc_uint8(official_cr_camera_render)
        Image.fromarray(official_cr_np, mode="RGB").save(output_dir / "rtgs_official_cr_camera_render.png")
        Image.fromarray(_diff_image(snapshot_np, official_cr_np), mode="RGB").save(output_dir / "diff_official_cr_camera_vs_snapshot.png")
        Image.fromarray(_diff_image(cr_np, official_cr_np), mode="RGB").save(output_dir / "diff_official_cr_camera_vs_cr.png")
        Image.fromarray(
            np.concatenate([gt_np, official_cr_np, snapshot_np, cr_np, _diff_image(cr_np, official_cr_np)], axis=1),
            mode="RGB",
        ).save(output_dir / "comparison_official_cr_camera_snapshot_cr.png")

    metrics = manifest.get("metrics", {})
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_cr_manifest(
    *,
    args: argparse.Namespace,
    runtime: Any,
    official_camera: Any,
    cr_camera: Any,
    gt: Any,
    official_render: Any | None,
    official_cr_camera_render: Any | None,
    snapshot_reference: Any,
    cr_render: Any,
    snapshot: Any,
    cr_snapshot: Any,
    official_ms: float | None,
    official_cr_camera_ms: float | None,
    snapshot_reference_ms: float,
    cr_ms: float,
    fov_normalization: dict[str, Any],
    cr_projection_adapter: dict[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "snapshot_vs_cr": compute_pair_metrics(snapshot_reference, cr_render, ssim_metric=runtime.ssim_metric),
        "cr_vs_gt": compute_pair_metrics(cr_render, gt, ssim_metric=runtime.ssim_metric),
        "snapshot_reference_vs_gt": compute_pair_metrics(snapshot_reference, gt, ssim_metric=runtime.ssim_metric),
    }
    if official_render is not None:
        metrics.update(
            {
                "rtgs_vs_gt": compute_pair_metrics(official_render, gt, ssim_metric=runtime.ssim_metric),
                "official_vs_snapshot": compute_pair_metrics(official_render, snapshot_reference, ssim_metric=runtime.ssim_metric),
                "official_vs_cr": compute_pair_metrics(official_render, cr_render, ssim_metric=runtime.ssim_metric),
            }
        )
    if official_cr_camera_render is not None:
        metrics.update(
            {
                "official_cr_camera_vs_gt": compute_pair_metrics(official_cr_camera_render, gt, ssim_metric=runtime.ssim_metric),
                "official_cr_camera_vs_snapshot": compute_pair_metrics(
                    official_cr_camera_render,
                    snapshot_reference,
                    ssim_metric=runtime.ssim_metric,
                ),
                "official_cr_camera_vs_cr": compute_pair_metrics(official_cr_camera_render, cr_render, ssim_metric=runtime.ssim_metric),
            }
        )

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "rtgs_cr_1view",
        "scene": runtime.official.scene_name,
        "dataset_kind": runtime.official.scene_paths.dataset_kind,
        "dataset_path": str(runtime.official.scene_paths.dataset_path),
        "source_path": str(runtime.official.source_path),
        "config_path": str(runtime.official.scene_paths.config_path),
        "model_path": str(Path(args.model_path).expanduser()),
        "checkpoint_path": str(runtime.official.checkpoint_path),
        "iteration": int(runtime.official.iteration),
        "split": args.split,
        "camera_index": int(args.camera_index),
        "n3dv_frame_index": int(args.n3dv_frame_index),
        "timestamp": float(getattr(cr_camera, "timestamp", 0.0)),
        "width": int(cr_camera.image_width),
        "height": int(cr_camera.image_height),
        "camera_source": str(runtime.camera_source),
        "gaussians_total": int(runtime.official.gaussians.get_xyz.shape[0]),
        "gaussians_after_temporal_mask": int(snapshot.means.shape[0]),
        "cr_gaussians_after_temporal_mask": int(cr_snapshot.means.shape[0]),
        "active_sh_degree": int(runtime.official.gaussians.active_sh_degree),
        "active_sh_degree_t": int(getattr(runtime.official.gaussians, "active_sh_degree_t", 0)),
        "gaussian_dim": int(getattr(runtime.official.gaussians, "gaussian_dim", 3)),
        "time_duration": [float(value) for value in runtime.official.time_duration],
        "requested_rtgs_code_root": str(runtime.official.requested_rtgs_code_root),
        "rtgs_code_root": str(runtime.official.rtgs_code_root),
        "rtgs_git_commit": runtime.official.rtgs_git_commit or None,
        "rtgs_code_status": dict(runtime.official.rtgs_code_status),
        "torch_extensions_dir": str(runtime.official.torch_extensions_dir),
        "gsplat_root": str(Path(args.gsplat_root).expanduser()),
        "render_ms": {
            "official": None if official_ms is None else float(official_ms),
            "official_cr_camera": None if official_cr_camera_ms is None else float(official_cr_camera_ms),
            "snapshot_reference": float(snapshot_reference_ms),
            "coherent_raster": float(cr_ms),
        },
        "camera_fov_normalization": dict(fov_normalization),
        "cr_projection_adapter": dict(cr_projection_adapter),
        "pipeline": vars(runtime.pipe).copy(),
        "snapshot_reference_pipeline": {
            "compute_cov3D_python": True,
            "convert_SHs_python": False,
            "eval_shfs_4d": False,
            "env_map_res": 0,
        },
        "cr_params": {
            "tile_size": int(args.tile_size),
            "near_plane": float(args.near_plane),
            "far_plane": float(args.far_plane),
            "camera_model": str(args.camera_model),
            "viewpoint_index": "single_view_all_zero_hwc_subpixel_map",
            "cluster_size": 1,
            "use_remapping": True,
        },
        "model_params": vars(runtime.official.model_args).copy(),
        "optimization_params": vars(runtime.official.optimization_args).copy(),
        "gaussian_model_constructor_kwargs": dict(runtime.official.gaussian_constructor_kwargs),
        "official_camera_render_contract": camera_render_contract_values(official_camera),
        "cr_camera_render_contract": camera_render_contract_values(cr_camera),
        "render_contract_tensor_shapes": render_contract_tensor_shapes(runtime.official.gaussians, cr_camera, runtime.background),
        "snapshot_tensor_diagnostics": snapshot_tensor_diagnostics(snapshot),
        "cr_snapshot_tensor_diagnostics": snapshot_tensor_diagnostics(cr_snapshot),
        "output_files": {
            "gt": "gt.png",
            "official_render": "rtgs_official_render.png" if official_render is not None else None,
            "official_cr_camera_render": "rtgs_official_cr_camera_render.png" if official_cr_camera_render is not None else None,
            "snapshot_reference_render": "rtgs_snapshot_reference_render.png",
            "cr_render": "rtgs_cr_render.png",
            "comparison_snapshot_vs_cr": "comparison_snapshot_vs_cr.png",
            "comparison_official_snapshot_cr": "comparison_official_snapshot_cr.png" if official_render is not None else None,
            "comparison_official_cr_camera_snapshot_cr": (
                "comparison_official_cr_camera_snapshot_cr.png" if official_cr_camera_render is not None else None
            ),
            "metrics": "metrics.json",
            "manifest": "manifest.json",
        },
        "metrics": metrics,
        "args": dict(vars(args)),
    }
    return manifest
