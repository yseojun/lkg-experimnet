from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
import time
import types
from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional

import numpy as np

from lkg_experiment.coherent_default.coherent_gsplat_bridge import build_cr_lookup_arrays, lookup_arrays_to_torch
from lkg_experiment.coherent_default.coherent_raster_experiment import tensor_to_hwc_uint8
from lkg_experiment.fourdgs.fourdgs_bridge import parse_4dgs_cfg_args


REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = Path("/data/ysj/result/4dgs/RTGS/jumpingjacks")
DEFAULT_CHECKPOINT = Path("checkpoints/chkpnt_best.pth")
DEFAULT_RTGS_CODE_ROOT = REPO_ROOT / "4d-gaussian-splatting"
DEFAULT_GSPLAT_ROOT = REPO_ROOT / "gsplat"
DEFAULT_DNERF_ROOT = Path("/data/ysj/dataset/dnerf")
DEFAULT_N3DV_ROOT = Path("/data/ysj/dataset/N3DV")
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "generated" / "rtgs_coherent"

_N3DV_SCENE_ALIASES = {
    "flame_salmon": "flame_salmon_1",
}


@dataclass(frozen=True)
class RtgsScenePaths:
    scene_name: str
    dataset_path: Path
    config_path: Path
    dataset_kind: str


@dataclass(frozen=True)
class RtgsSnapshot:
    means: Any
    covars: Any
    opacities: Any
    colors: Any
    mask: Any


@dataclass(frozen=True)
class RtgsCheckpoint:
    model: Any
    iteration: int
    config: Mapping[str, Any]
    cfg_args: Namespace
    scene_paths: RtgsScenePaths
    checkpoint_path: Path


class RtgsLiteGaussianModel:
    def __init__(
        self,
        sh_degree: int,
        *,
        gaussian_dim: int = 4,
        time_duration: list[float] | None = None,
        rot_4d: bool = True,
        force_sh_3d: bool = False,
        sh_degree_t: int = 2,
        prefilter_var: float = -1.0,
    ) -> None:
        self.active_sh_degree = 0
        self.max_sh_degree = int(sh_degree)
        self.gaussian_dim = int(gaussian_dim)
        self.time_duration = list(time_duration or [0.0, 1.0])
        self.rot_4d = bool(rot_4d)
        self.force_sh_3d = bool(force_sh_3d)
        self.active_sh_degree_t = 0
        self.max_sh_degree_t = int(sh_degree_t)
        self.prefilter_var = float(prefilter_var)
        self.env_map = None

    def restore(self, model_args: tuple[Any, ...]) -> None:
        if len(model_args) == 19:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.t_gradient_accum,
                self.denom,
                _opt_dict,
                self.spatial_lr_scale,
                self._t,
                self._scaling_t,
                self._rotation_r,
                self.rot_4d,
                self.env_map,
                self.active_sh_degree_t,
            ) = model_args
            return
        if len(model_args) == 12:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                _opt_dict,
                self.spatial_lr_scale,
            ) = model_args
            return
        raise ValueError(f"unsupported RTGS checkpoint model tuple length: {len(model_args)}")

    @property
    def get_scaling(self):
        import torch

        return torch.exp(self._scaling)

    @property
    def get_scaling_t(self):
        import torch

        return torch.exp(self._scaling_t)

    @property
    def get_scaling_xyzt(self):
        import torch

        return torch.cat([self._scaling.exp(), self._scaling_t.exp()], dim=1)

    @property
    def get_rotation(self):
        import torch

        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_rotation_r(self):
        import torch

        return torch.nn.functional.normalize(self._rotation_r)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_t(self):
        return self._t

    @property
    def get_features(self):
        import torch

        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self):
        import torch

        return torch.sigmoid(self._opacity)

    @property
    def get_max_sh_channels(self):
        if self.gaussian_dim == 3 or self.force_sh_3d:
            return (self.max_sh_degree + 1) ** 2
        if self.gaussian_dim == 4 and self.max_sh_degree_t == 0:
            return (1, 6, 16, 33)[self.max_sh_degree]
        if self.gaussian_dim == 4 and self.max_sh_degree_t > 0:
            return (self.max_sh_degree + 1) ** 2 * (self.max_sh_degree_t + 1)
        raise ValueError(f"unsupported gaussian_dim: {self.gaussian_dim}")

    def get_cov_t(self, scaling_modifier: float = 1.0):
        if self.rot_4d:
            L = _build_scaling_rotation_4d(float(scaling_modifier) * self.get_scaling_xyzt, self._rotation, self._rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance[:, 3, 3].unsqueeze(1)
        return self.get_scaling_t * float(scaling_modifier)

    def get_marginal_t(self, timestamp: float, scaling_modifier: float = 1.0):
        import torch

        sigma = self.get_cov_t(float(scaling_modifier))
        if self.prefilter_var > 0.0:
            sigma = sigma + self.prefilter_var
        return torch.exp(-0.5 * (self.get_t - float(timestamp)) ** 2 / sigma)

    def get_covariance(self, scaling_modifier: float = 1.0):
        L = _build_scaling_rotation(float(scaling_modifier) * self.get_scaling, self._rotation)
        actual_covariance = L @ L.transpose(1, 2)
        return _strip_symmetric(actual_covariance)

    def get_current_covariance_and_mean_offset(self, scaling_modifier: float = 1.0, timestamp: float = 0.0):
        L = _build_scaling_rotation_4d(float(scaling_modifier) * self.get_scaling_xyzt, self._rotation, self._rotation_r)
        actual_covariance = L @ L.transpose(1, 2)
        cov_11 = actual_covariance[:, :3, :3]
        cov_12 = actual_covariance[:, 0:3, 3:4]
        cov_t = actual_covariance[:, 3:4, 3:4]
        current_covariance = cov_11 - cov_12 @ cov_12.transpose(1, 2) / cov_t
        dt = float(timestamp) - self.get_t
        mean_offset = cov_12.squeeze(-1) / cov_t.squeeze(-1) * dt
        return _strip_symmetric(current_covariance), mean_offset


class RtgsSimpleCamera:
    def __init__(
        self,
        *,
        R: np.ndarray,
        T: np.ndarray,
        FoVx: float,
        FoVy: float,
        image: Any,
        image_name: str,
        uid: int,
        timestamp: float,
        fl_x: float,
        fl_y: float,
        cx: float,
        cy: float,
        resolution: tuple[int, int],
        data_device: str,
    ) -> None:
        import torch

        self.uid = uid
        self.colmap_id = uid
        self.R = R
        self.T = T
        self.FoVx = float(FoVx)
        self.FoVy = float(FoVy)
        self.image = image
        self.gt_alpha_mask = None
        self.image_name = image_name
        self.timestamp = float(timestamp)
        self.fl_x = float(fl_x)
        self.fl_y = float(fl_y)
        self.cx = float(cx)
        self.cy = float(cy)
        self.resolution = resolution
        self.image_width = int(resolution[0])
        self.image_height = int(resolution[1])
        self.data_device = torch.device(data_device)
        self.zfar = 100.0
        self.znear = 0.01
        self.world_view_transform = torch.tensor(_get_world_to_view2(R, T), dtype=torch.float32).transpose(0, 1)
        self.projection_matrix = _get_projection_matrix(self.znear, self.zfar, self.FoVx, self.FoVy).transpose(0, 1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def cuda(self):
        copy = deepcopy(self)
        for key, value in copy.__dict__.items():
            if hasattr(value, "to"):
                try:
                    copy.__dict__[key] = value.to(copy.data_device)
                except TypeError:
                    pass
        return copy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a 4d-gaussian-splatting RTGS checkpoint through a 1-view CoherentRaster snapshot path")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="RTGS scene output directory")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Checkpoint file or path relative to --model-path")
    parser.add_argument("--rtgs-code-root", default=str(DEFAULT_RTGS_CODE_ROOT), help="Cloned 4d-gaussian-splatting code root")
    parser.add_argument("--gsplat-root", default=str(DEFAULT_GSPLAT_ROOT), help="Local gsplat code root containing CoherentRaster")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DNERF_ROOT), help="dnerf dataset root")
    parser.add_argument("--n3dv-root", default=str(DEFAULT_N3DV_ROOT), help="N3DV dataset root")
    parser.add_argument("--config", default=None, help="RTGS YAML config override")
    parser.add_argument("--output-dir", default=None, help="Output directory; defaults under experiment/generated/rtgs_coherent")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=100.0)
    parser.add_argument("--camera-model", choices=("pinhole", "ortho", "fisheye"), default="pinhole")
    parser.add_argument("--background", choices=("auto", "black", "white"), default="auto")
    parser.add_argument("--no-ssim", action="store_true", help="Skip optional torchmetrics SSIM computation")
    parser.add_argument("--debug-cr", action="store_true", help="Print CoherentRaster timing/debug output")
    return parser


def default_output_path(model_path: Path | str, *, split: str, camera_index: int, timestamp: float) -> Path:
    scene_name = Path(model_path).expanduser().name
    return DEFAULT_OUTPUT_ROOT / scene_name / f"{scene_name}_t{float(timestamp):.6f}_{split}_{int(camera_index)}"


def resolve_rtgs_scene_paths(
    model_path: Path | str,
    *,
    rtgs_code_root: Path | str,
    dataset_root: Path | str,
    n3dv_root: Path | str,
    config_path: Path | str | None = None,
) -> RtgsScenePaths:
    scene_name = Path(model_path).expanduser().name
    rtgs_root = Path(rtgs_code_root).expanduser()
    dnerf_root = Path(dataset_root).expanduser()
    n3dv_root = Path(n3dv_root).expanduser()

    dnerf_path = dnerf_root / scene_name
    n3dv_name = _N3DV_SCENE_ALIASES.get(scene_name, scene_name)
    n3dv_path = n3dv_root / n3dv_name
    if dnerf_path.exists():
        dataset_path = dnerf_path
        dataset_kind = "dnerf"
        default_config = rtgs_root / "configs" / "dnerf" / f"{scene_name}.yaml"
    elif n3dv_path.exists():
        dataset_path = n3dv_path
        dataset_kind = "n3dv"
        default_config = rtgs_root / "configs" / "dynerf" / f"{scene_name}.yaml"
    else:
        raise FileNotFoundError(f"could not resolve dataset for scene {scene_name!r} under {dnerf_root} or {n3dv_root}")

    selected_config = Path(config_path).expanduser() if config_path else default_config
    if not selected_config.is_file():
        raise FileNotFoundError(f"RTGS config not found: {selected_config}")

    return RtgsScenePaths(
        scene_name=scene_name,
        dataset_path=dataset_path,
        config_path=selected_config,
        dataset_kind=dataset_kind,
    )


def resolve_checkpoint_path(model_path: Path | str, checkpoint: Path | str) -> Path:
    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.is_absolute():
        selected = checkpoint_path
    else:
        selected = Path(model_path).expanduser() / checkpoint_path
    if not selected.is_file():
        raise FileNotFoundError(f"RTGS checkpoint not found: {selected}")
    return selected


def install_rtgs_code_root(code_root: Path | str) -> Path:
    root = Path(code_root).expanduser().resolve()
    required = [
        root / "scene" / "gaussian_model.py",
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
    _install_scene_gaussian_model_stub(root)
    return root


def load_rtgs_yaml_config(path: Path | str) -> dict[str, Any]:
    cfg_path = Path(path).expanduser()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"RTGS config not found: {cfg_path}")

    root: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    for raw_line in cfg_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if indent == 0 and not value:
            current_section = {}
            root[key] = current_section
        elif indent > 0 and current_section is not None:
            current_section[key] = _parse_yaml_scalar(value)
        else:
            current_section = None
            root[key] = _parse_yaml_scalar(value)
    return root


def load_rtgs_checkpoint(
    *,
    model_path: Path | str,
    checkpoint: Path | str,
    rtgs_code_root: Path | str,
    dataset_root: Path | str,
    n3dv_root: Path | str,
    config_path: Path | str | None,
    device: str,
) -> RtgsCheckpoint:
    install_rtgs_code_root(rtgs_code_root)
    import torch

    scene_paths = resolve_rtgs_scene_paths(
        model_path,
        rtgs_code_root=rtgs_code_root,
        dataset_root=dataset_root,
        n3dv_root=n3dv_root,
        config_path=config_path,
    )
    cfg = load_rtgs_yaml_config(scene_paths.config_path)
    cfg_args = _load_cfg_args(Path(model_path).expanduser())
    checkpoint_path = resolve_checkpoint_path(model_path, checkpoint)
    model_args, iteration = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_cfg = cfg.get("ModelParams", {})
    pipe_cfg = cfg.get("PipelineParams", {})
    sh_degree = int(model_cfg.get("sh_degree", getattr(cfg_args, "sh_degree", 3)))
    gaussian_dim = int(cfg.get("gaussian_dim", 4))
    time_duration = [float(value) for value in cfg.get("time_duration", [0.0, 1.0])]
    rot_4d = bool(cfg.get("rot_4d", True))
    force_sh_3d = bool(cfg.get("force_sh_3d", False))
    eval_shfs_4d = bool(pipe_cfg.get("eval_shfs_4d", True))
    prefilter_var = float(model_cfg.get("prefilter_var", getattr(cfg_args, "prefilter_var", -1.0)))

    model = RtgsLiteGaussianModel(
        sh_degree,
        gaussian_dim=gaussian_dim,
        time_duration=time_duration,
        rot_4d=rot_4d,
        force_sh_3d=force_sh_3d,
        sh_degree_t=2 if eval_shfs_4d else 0,
        prefilter_var=prefilter_var,
    )
    model.restore(model_args)

    return RtgsCheckpoint(
        model=model,
        iteration=int(iteration),
        config=cfg,
        cfg_args=cfg_args,
        scene_paths=scene_paths,
        checkpoint_path=checkpoint_path,
    )


def load_rtgs_camera(
    *,
    checkpoint: RtgsCheckpoint,
    split: str,
    camera_index: int,
    device: str,
):
    model_cfg = dict(checkpoint.config.get("ModelParams", {}))
    source_path = _camera_source_path(checkpoint.scene_paths)
    args = SimpleNamespace(
        source_path=str(source_path),
        model_path=str(Path(checkpoint.cfg_args.model_path).expanduser()) if hasattr(checkpoint.cfg_args, "model_path") else "",
        images=model_cfg.get("images", getattr(checkpoint.cfg_args, "images", "images")),
        resolution=int(model_cfg.get("resolution", getattr(checkpoint.cfg_args, "resolution", 2))),
        white_background=bool(model_cfg.get("white_background", getattr(checkpoint.cfg_args, "white_background", False))),
        data_device=device,
        eval=bool(model_cfg.get("eval", getattr(checkpoint.cfg_args, "eval", True))),
        extension=model_cfg.get("extension", getattr(checkpoint.cfg_args, "extension", ".png")),
        num_extra_pts=int(model_cfg.get("num_extra_pts", getattr(checkpoint.cfg_args, "num_extra_pts", 0))),
        frame_ratio=int(model_cfg.get("frame_ratio", getattr(checkpoint.cfg_args, "frame_ratio", 1))),
        dataloader=bool(model_cfg.get("dataloader", getattr(checkpoint.cfg_args, "dataloader", False))),
    )

    if checkpoint.scene_paths.dataset_kind == "dnerf" or (Path(args.source_path) / "transforms_train.json").exists():
        return _load_blender_camera(
            source_path=Path(args.source_path),
            split=split,
            camera_index=camera_index,
            white_background=args.white_background,
            resolution=args.resolution,
            extension=args.extension,
            frame_ratio=args.frame_ratio,
            time_duration=checkpoint.model.time_duration,
            device=device,
        )

    return _load_rtgs_camera_with_original_readers(
        args=args,
        split=split,
        camera_index=camera_index,
        time_duration=checkpoint.model.time_duration,
    )


def rtgs_camera_to_gsplat_inputs(camera: Any, *, device: str):
    import torch

    viewmat = camera.world_view_transform.transpose(0, 1).to(device=device, dtype=torch.float32).contiguous()
    if float(getattr(camera, "fl_x", -1.0)) > 0 and float(getattr(camera, "fl_y", -1.0)) > 0:
        fx = float(camera.fl_x)
        fy = float(camera.fl_y)
        cx = float(camera.cx)
        cy = float(camera.cy)
    else:
        fx = _fov_to_focal(float(camera.FoVx), int(camera.image_width))
        fy = _fov_to_focal(float(camera.FoVy), int(camera.image_height))
        cx = float(camera.image_width) / 2.0
        cy = float(camera.image_height) / 2.0
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    return viewmat, K


def materialize_rtgs_snapshot(
    pc: Any,
    *,
    timestamp: float,
    camera_center: Any,
    scaling_modifier: float = 1.0,
    eval_shfs_4d_fn: Optional[Callable[..., Any]] = None,
    eval_sh_fn: Optional[Callable[..., Any]] = None,
) -> RtgsSnapshot:
    import torch

    means_base = pc.get_xyz
    opacity = pc.get_opacity.reshape(-1)
    if bool(getattr(pc, "rot_4d", False)):
        covars, delta_mean = pc.get_current_covariance_and_mean_offset(float(scaling_modifier), float(timestamp))
        means = means_base + delta_mean
    else:
        covars = pc.get_covariance(float(scaling_modifier))
        means = means_base

    if int(getattr(pc, "gaussian_dim", 3)) == 4:
        marginal_t = pc.get_marginal_t(float(timestamp), float(scaling_modifier)).reshape(-1)
        opacity = opacity * marginal_t
        mask = marginal_t > 0.05
    else:
        mask = torch.ones((means_base.shape[0],), dtype=torch.bool, device=means_base.device)

    colors = evaluate_rtgs_colors(
        pc,
        timestamp=float(timestamp),
        camera_center=camera_center,
        mask=mask,
        eval_shfs_4d_fn=eval_shfs_4d_fn,
        eval_sh_fn=eval_sh_fn,
    )

    return RtgsSnapshot(
        means=means[mask].contiguous(),
        covars=covars[mask].contiguous(),
        opacities=opacity[mask].contiguous(),
        colors=colors.contiguous(),
        mask=mask.contiguous(),
    )


def evaluate_rtgs_colors(
    pc: Any,
    *,
    timestamp: float,
    camera_center: Any,
    mask: Any,
    eval_shfs_4d_fn: Optional[Callable[..., Any]] = None,
    eval_sh_fn: Optional[Callable[..., Any]] = None,
):
    import torch

    means_for_color = pc.get_xyz[mask]
    camera_center = torch.as_tensor(camera_center, dtype=means_for_color.dtype, device=means_for_color.device)
    dir_pp = (means_for_color - camera_center.reshape(1, 3)).detach()
    dir_pp = dir_pp / dir_pp.norm(dim=1, keepdim=True).clamp_min(1e-12)
    shs_view = pc.get_features[mask].transpose(1, 2).reshape(-1, 3, int(pc.get_max_sh_channels))

    if int(getattr(pc, "gaussian_dim", 3)) == 3 or bool(getattr(pc, "force_sh_3d", False)):
        if eval_sh_fn is None:
            from utils.sh_utils import eval_sh

            eval_sh_fn = eval_sh
        rgb = eval_sh_fn(int(pc.active_sh_degree), shs_view, dir_pp)
    else:
        if eval_shfs_4d_fn is None:
            from utils.sh_utils import eval_shfs_4d

            eval_shfs_4d_fn = eval_shfs_4d
        dir_t = (pc.get_t[mask] - float(timestamp)).detach()
        duration = float(pc.time_duration[1] - pc.time_duration[0])
        rgb = eval_shfs_4d_fn(int(pc.active_sh_degree), int(pc.active_sh_degree_t), shs_view, dir_pp, dir_t, duration)
    return torch.clamp_min(rgb + 0.5, 0.0)


def render_rtgs_original(camera: Any, pc: Any, pipe: Any, background: Any):
    from gaussian_renderer import render

    with _torch_no_grad():
        camera_cuda = camera.cuda()
        return render(camera_cuda, pc, pipe, background)["render"].detach().clamp(0.0, 1.0).contiguous()


def render_rtgs_coherent(
    *,
    snapshot: RtgsSnapshot,
    viewmat: Any,
    K: Any,
    width: int,
    height: int,
    tile_size: int,
    near_plane: float,
    far_plane: float,
    camera_model: str,
    background: Any,
    debug: bool,
):
    import numpy as np
    import torch
    from coherent_raster.utils.utils_coherent_raster import unpad, unpatchify_image_shape_matrix
    from gsplat.rendering_coherent_raster import rasterization_CR

    viewpoint_index = np.zeros((int(height), int(width), 3), dtype=np.uint32)
    lookup = build_cr_lookup_arrays(viewpoint_index, tile_size=int(tile_size), use_remapping=True)
    view_idx_matrix, subpixel_coord_matrix = lookup_arrays_to_torch(lookup, device=str(viewmat.device))
    adjacent_viewmats = viewmat.view(1, 1, 4, 4).contiguous()
    backgrounds = None if background is None else background.contiguous()

    with torch.no_grad():
        colors, _, _ = rasterization_CR(
            means=snapshot.means,
            quats=None,
            scales=None,
            opacities=snapshot.opacities,
            colors=snapshot.colors,
            adjacent_viewmats=adjacent_viewmats,
            Ks=K.unsqueeze(0).contiguous(),
            view_idx_matrix=view_idx_matrix,
            subpixel_coord_matrix=subpixel_coord_matrix,
            width=int(width),
            height=int(height),
            sh_degree=None,
            near_plane=float(near_plane),
            far_plane=float(far_plane),
            backgrounds=backgrounds,
            camera_model=camera_model,
            tile_size=int(tile_size),
            is_debug=debug,
            covars=snapshot.covars,
        )
        image = unpatchify_image_shape_matrix(colors)
        image = unpad(image, int(height), int(width))
        return image.clamp(0.0, 1.0).contiguous()


def compute_pair_metrics(a: Any, b: Any, *, ssim_metric: Any = None) -> dict[str, float]:
    import torch

    a = a.float().clamp(0.0, 1.0)
    b = b.float().clamp(0.0, 1.0)
    mse = torch.mean((a - b) ** 2).item()
    mae = torch.mean(torch.abs(a - b)).item()
    metrics = {
        "mse": float(mse),
        "mae": float(mae),
        "psnr": float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse)),
    }
    if ssim_metric is not None:
        metrics["ssim"] = float(ssim_metric(a.unsqueeze(0), b.unsqueeze(0)).item())
    return metrics


def save_outputs(
    *,
    output_stem: Path,
    gt: Any,
    rtgs_original: Any,
    rtgs_coherent: Any,
    manifest: Mapping[str, Any],
) -> None:
    from PIL import Image

    output_stem.mkdir(parents=True, exist_ok=True)
    gt_np = tensor_to_hwc_uint8(gt)
    rtgs_np = tensor_to_hwc_uint8(rtgs_original)
    cr_np = tensor_to_hwc_uint8(rtgs_coherent)
    diff_np = np.clip(np.abs(cr_np.astype(np.int16) - rtgs_np.astype(np.int16)) * 4, 0, 255).astype(np.uint8)

    Image.fromarray(gt_np, mode="RGB").save(output_stem / "gt.png")
    Image.fromarray(rtgs_np, mode="RGB").save(output_stem / "rtgs_original.png")
    Image.fromarray(cr_np, mode="RGB").save(output_stem / "rtgs_coherent.png")
    Image.fromarray(np.concatenate([gt_np, rtgs_np, cr_np, diff_np], axis=1), mode="RGB").save(output_stem / "comparison.png")
    (output_stem / "metrics.json").write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_blender_camera(
    *,
    source_path: Path,
    split: str,
    camera_index: int,
    white_background: bool,
    resolution: int,
    extension: str,
    frame_ratio: int,
    time_duration: list[float],
    device: str,
):
    import torch
    from PIL import Image

    transforms_name = "transforms_test.json" if split == "test" else "transforms_train.json"
    transforms_path = source_path / transforms_name
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Blender transforms file not found: {transforms_path}")
    contents = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = []
    for idx, frame in enumerate(contents["frames"]):
        timestamp = float(frame.get("time", 0.0))
        if int(frame_ratio) > 1:
            timestamp /= int(frame_ratio)
        if time_duration is not None and "time" in frame:
            if timestamp < float(time_duration[0]) or timestamp > float(time_duration[1]):
                continue
        frames.append((idx, frame, timestamp))
    if camera_index < 0 or camera_index >= len(frames):
        raise IndexError(f"camera index {camera_index} out of range for {split} split with {len(frames)} cameras")

    uid, frame, timestamp = frames[int(camera_index)]
    image_path = source_path / f"{frame['file_path']}{extension}"
    with Image.open(image_path) as image_load:
        rgba = image_load.convert("RGBA")
        orig_w, orig_h = rgba.size
        scale = int(resolution) if int(resolution) in {1, 2, 3, 4, 8} else 1
        target_resolution = (round(orig_w / scale), round(orig_h / scale))
        rgb = _compose_rgba_on_background(rgba, white_background).resize(target_resolution, Image.BILINEAR)
        image_np = np.asarray(rgb, dtype=np.float32) / 255.0
    gt = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()

    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3, :3])
    T = w2c[:3, 3]
    if "fl_x" in frame and "fl_y" in frame and "cx" in frame and "cy" in frame:
        fl_x = float(frame["fl_x"]) / scale
        fl_y = float(frame["fl_y"]) / scale
        cx = float(frame["cx"]) / scale
        cy = float(frame["cy"]) / scale
        FoVx = _focal_to_fov(fl_x, target_resolution[0])
        FoVy = _focal_to_fov(fl_y, target_resolution[1])
    elif all(key in contents for key in ("fl_x", "fl_y", "cx", "cy")):
        fl_x = float(contents["fl_x"]) / scale
        fl_y = float(contents["fl_y"]) / scale
        cx = float(contents["cx"]) / scale
        cy = float(contents["cy"]) / scale
        FoVx = _focal_to_fov(fl_x, target_resolution[0])
        FoVy = _focal_to_fov(fl_y, target_resolution[1])
    else:
        FoVx = float(contents["camera_angle_x"])
        focal_x = _fov_to_focal(FoVx, target_resolution[0])
        FoVy = _focal_to_fov(focal_x, target_resolution[1])
        fl_x = focal_x
        fl_y = _fov_to_focal(FoVy, target_resolution[1])
        cx = target_resolution[0] / 2.0
        cy = target_resolution[1] / 2.0

    camera = RtgsSimpleCamera(
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        image=gt,
        image_name=Path(frame["file_path"]).stem,
        uid=int(uid),
        timestamp=timestamp,
        fl_x=fl_x,
        fl_y=fl_y,
        cx=cx,
        cy=cy,
        resolution=target_resolution,
        data_device=device,
    )
    return gt, camera


def _load_rtgs_camera_with_original_readers(
    *,
    args: Any,
    split: str,
    camera_index: int,
    time_duration: list[float],
):
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from utils.camera_utils import loadCam
    from utils.data_utils import CameraDataset

    if (Path(args.source_path) / "sparse").exists():
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
    else:
        scene_info = sceneLoadTypeCallbacks["Blender"](
            args.source_path,
            args.white_background,
            args.eval,
            time_duration=time_duration,
            extension=args.extension,
            num_extra_pts=args.num_extra_pts,
            frame_ratio=args.frame_ratio,
            dataloader=args.dataloader,
        )
    cam_infos = scene_info.test_cameras if split == "test" else scene_info.train_cameras
    if not cam_infos:
        raise ValueError(f"RTGS dataset has no {split} cameras: {args.source_path}")
    if camera_index < 0 or camera_index >= len(cam_infos):
        raise IndexError(f"camera index {camera_index} out of range for {split} split with {len(cam_infos)} cameras")

    camera = loadCam(args, int(camera_index), cam_infos[int(camera_index)], 1.0)
    gt_image, camera = CameraDataset([camera], args.white_background)[0]
    return gt_image.contiguous(), camera


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    import torch

    if not str(args.device).startswith("cuda"):
        raise RuntimeError("RTGS original renderer requires CUDA because 4d-gaussian-splatting hardcodes CUDA tensors")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot run RTGS original renderer")

    install_rtgs_code_root(args.rtgs_code_root)
    _install_gsplat_root(args.gsplat_root)
    checkpoint = load_rtgs_checkpoint(
        model_path=args.model_path,
        checkpoint=args.checkpoint,
        rtgs_code_root=args.rtgs_code_root,
        dataset_root=args.dataset_root,
        n3dv_root=args.n3dv_root,
        config_path=args.config,
        device=args.device,
    )
    gt, camera = load_rtgs_camera(
        checkpoint=checkpoint,
        split=args.split,
        camera_index=args.camera_index,
        device=args.device,
    )
    gt = gt.to(device=args.device, non_blocking=True).contiguous()
    viewmat, K = rtgs_camera_to_gsplat_inputs(camera, device=args.device)
    width = int(camera.image_width)
    height = int(camera.image_height)
    timestamp = float(camera.timestamp)

    pipe = _pipeline_namespace(checkpoint.config.get("PipelineParams", {}))
    background = _background_tensor(args.background, checkpoint.cfg_args, args.device)

    torch.cuda.synchronize()
    start = time.perf_counter()
    rtgs_original = render_rtgs_original(camera, checkpoint.model, pipe, background)
    torch.cuda.synchronize()
    rtgs_ms = (time.perf_counter() - start) * 1000.0

    snapshot = materialize_rtgs_snapshot(
        checkpoint.model,
        timestamp=timestamp,
        camera_center=camera.camera_center.to(device=args.device),
    )

    torch.cuda.synchronize()
    start = time.perf_counter()
    rtgs_coherent = render_rtgs_coherent(
        snapshot=snapshot,
        viewmat=viewmat,
        K=K,
        width=width,
        height=height,
        tile_size=args.tile_size,
        near_plane=args.near_plane,
        far_plane=args.far_plane,
        camera_model=args.camera_model,
        background=background,
        debug=args.debug_cr,
    )
    torch.cuda.synchronize()
    cr_ms = (time.perf_counter() - start) * 1000.0

    ssim_metric = None if args.no_ssim else _make_ssim_metric(args.device)
    metrics = {
        "rtgs_vs_gt": compute_pair_metrics(rtgs_original, gt, ssim_metric=ssim_metric),
        "cr_vs_gt": compute_pair_metrics(rtgs_coherent, gt, ssim_metric=ssim_metric),
        "cr_vs_rtgs": compute_pair_metrics(rtgs_coherent, rtgs_original, ssim_metric=ssim_metric),
    }

    output_stem = Path(args.output_dir).expanduser() if args.output_dir else default_output_path(
        args.model_path,
        split=args.split,
        camera_index=args.camera_index,
        timestamp=timestamp,
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scene": checkpoint.scene_paths.scene_name,
        "dataset_kind": checkpoint.scene_paths.dataset_kind,
        "dataset_path": str(checkpoint.scene_paths.dataset_path),
        "config_path": str(checkpoint.scene_paths.config_path),
        "model_path": str(Path(args.model_path).expanduser()),
        "checkpoint_path": str(checkpoint.checkpoint_path),
        "iteration": checkpoint.iteration,
        "split": args.split,
        "camera_index": args.camera_index,
        "timestamp": timestamp,
        "width": width,
        "height": height,
        "gaussians_total": int(checkpoint.model.get_xyz.shape[0]),
        "gaussians_snapshot": int(snapshot.means.shape[0]),
        "active_sh_degree": int(checkpoint.model.active_sh_degree),
        "active_sh_degree_t": int(checkpoint.model.active_sh_degree_t),
        "rtgs_render_ms": rtgs_ms,
        "coherent_render_ms": cr_ms,
        "metrics": metrics,
        "args": vars(args),
    }
    save_outputs(output_stem=output_stem, gt=gt, rtgs_original=rtgs_original, rtgs_coherent=rtgs_coherent, manifest=manifest)
    print(f"Wrote {output_stem} ({width}x{height}, snapshot_gs={snapshot.means.shape[0]}, cr_vs_rtgs_psnr={metrics['cr_vs_rtgs']['psnr']:.3f})", file=sys.stderr)
    return 0


def _parse_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value in {"True", "true"}:
        return True
    if value in {"False", "false"}:
        return False
    if value in {"None", "null"}:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value.strip("\"'")


def _load_cfg_args(model_path: Path) -> Namespace:
    cfg_path = model_path / "cfg_args"
    if cfg_path.is_file():
        return parse_4dgs_cfg_args(cfg_path)
    return Namespace()


def _install_scene_gaussian_model_stub(root: Path) -> None:
    scene_module = sys.modules.get("scene")
    if scene_module is None:
        scene_module = types.ModuleType("scene")
        scene_module.__package__ = "scene"
        scene_module.__path__ = [str(root / "scene")]
        sys.modules["scene"] = scene_module
    gaussian_model_module = types.ModuleType("scene.gaussian_model")
    gaussian_model_module.GaussianModel = RtgsLiteGaussianModel
    sys.modules["scene.gaussian_model"] = gaussian_model_module


def _camera_source_path(scene_paths: RtgsScenePaths) -> Path:
    if scene_paths.dataset_kind == "n3dv" and (scene_paths.dataset_path / "colmap" / "sparse").exists():
        return scene_paths.dataset_path / "colmap"
    return scene_paths.dataset_path


def _compose_rgba_on_background(image: Any, white_background: bool):
    from PIL import Image

    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    bg = np.array([1.0, 1.0, 1.0], dtype=np.float32) if white_background else np.array([0.0, 0.0, 0.0], dtype=np.float32)
    rgb = rgba[:, :, :3] * rgba[:, :, 3:4] + bg.reshape(1, 1, 3) * (1.0 - rgba[:, :, 3:4])
    return Image.fromarray(np.asarray(np.clip(rgb, 0.0, 1.0) * 255.0, dtype=np.uint8), mode="RGB")


def _strip_symmetric(matrix: Any):
    import torch

    uncertainty = torch.zeros((matrix.shape[0], 6), dtype=matrix.dtype, device=matrix.device)
    uncertainty[:, 0] = matrix[:, 0, 0]
    uncertainty[:, 1] = matrix[:, 0, 1]
    uncertainty[:, 2] = matrix[:, 0, 2]
    uncertainty[:, 3] = matrix[:, 1, 1]
    uncertainty[:, 4] = matrix[:, 1, 2]
    uncertainty[:, 5] = matrix[:, 2, 2]
    return uncertainty


def _build_rotation(r: Any):
    import torch

    q = torch.nn.functional.normalize(r)
    R = torch.zeros((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    qr = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - qr * z)
    R[:, 0, 2] = 2 * (x * z + qr * y)
    R[:, 1, 0] = 2 * (x * y + qr * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - qr * x)
    R[:, 2, 0] = 2 * (x * z - qr * y)
    R[:, 2, 1] = 2 * (y * z + qr * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _build_scaling_rotation(s: Any, r: Any):
    import torch

    L = torch.zeros((s.shape[0], 3, 3), dtype=s.dtype, device=s.device)
    R = _build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    return L @ R


def _build_rotation_4d(l: Any, r: Any):
    import torch

    q_l = torch.nn.functional.normalize(l)
    q_r = torch.nn.functional.normalize(r)
    a, b, c, d = q_l.unbind(-1)
    p, q, rr, s = q_r.unbind(-1)
    M_l = torch.stack(
        [
            a,
            -b,
            -c,
            -d,
            b,
            a,
            -d,
            c,
            c,
            d,
            a,
            -b,
            d,
            -c,
            b,
            a,
        ]
    ).view(4, 4, -1).permute(2, 0, 1)
    M_r = torch.stack(
        [
            p,
            q,
            rr,
            s,
            -q,
            p,
            -s,
            rr,
            -rr,
            s,
            p,
            -q,
            -s,
            -rr,
            q,
            p,
        ]
    ).view(4, 4, -1).permute(2, 0, 1)
    return (M_l @ M_r).flip(1, 2)


def _build_scaling_rotation_4d(s: Any, l: Any, r: Any):
    import torch

    L = torch.zeros((s.shape[0], 4, 4), dtype=s.dtype, device=s.device)
    R = _build_rotation_4d(l, r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    L[:, 3, 3] = s[:, 3]
    return R @ L


def _get_world_to_view2(R: np.ndarray, t: np.ndarray, translate: np.ndarray | None = None, scale: float = 1.0) -> np.ndarray:
    translate = np.array([0.0, 0.0, 0.0]) if translate is None else translate
    Rt = np.zeros((4, 4), dtype=np.float64)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    C2W[:3, 3] = (cam_center + translate) * scale
    return np.linalg.inv(C2W).astype(np.float32)


def _get_projection_matrix(znear: float, zfar: float, fovX: float, fovY: float):
    import torch

    tan_half_fovy = math.tan(float(fovY) / 2.0)
    tan_half_fovx = math.tan(float(fovX) / 2.0)
    top = tan_half_fovy * znear
    bottom = -top
    right = tan_half_fovx * znear
    left = -right
    P = torch.zeros(4, 4, dtype=torch.float32)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def _focal_to_fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def _clear_conflicting_modules(root: Path, names: tuple[str, ...]) -> None:
    for module_name in list(sys.modules):
        if not any(module_name == name or module_name.startswith(f"{name}.") for name in names):
            continue
        module = sys.modules.get(module_name)
        if module is None:
            continue
        origin = _module_path(module)
        if origin is not None and _is_relative_to(origin, root):
            continue
        del sys.modules[module_name]


def _module_path(module: Any) -> Path | None:
    path = getattr(module, "__file__", None)
    if path:
        return Path(path).resolve()
    search_locations = getattr(module, "__path__", None)
    if search_locations:
        try:
            return Path(next(iter(search_locations))).resolve()
        except StopIteration:
            return None
    return None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _fov_to_focal(fov: float, pixels: int) -> float:
    return float(pixels) / (2.0 * math.tan(float(fov) / 2.0))


def _install_gsplat_root(gsplat_root: Path | str) -> None:
    root = Path(gsplat_root).expanduser().resolve()
    if not (root / "gsplat").is_dir():
        raise FileNotFoundError(f"gsplat root not found or incomplete: {root}")
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


def _pipeline_namespace(pipe_cfg: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        convert_SHs_python=bool(pipe_cfg.get("convert_SHs_python", False)),
        compute_cov3D_python=bool(pipe_cfg.get("compute_cov3D_python", False)),
        debug=bool(pipe_cfg.get("debug", False)),
        env_map_res=int(pipe_cfg.get("env_map_res", 0) or 0),
        env_optimize_until=int(pipe_cfg.get("env_optimize_until", 1000000000) or 1000000000),
        env_optimize_from=int(pipe_cfg.get("env_optimize_from", 0) or 0),
        eval_shfs_4d=bool(pipe_cfg.get("eval_shfs_4d", True)),
    )


def _background_tensor(background_mode: str, cfg_args: Namespace, device: str):
    import torch

    if background_mode == "white":
        return torch.ones(3, dtype=torch.float32, device=device)
    if background_mode == "black":
        return torch.zeros(3, dtype=torch.float32, device=device)
    return torch.ones(3, dtype=torch.float32, device=device) if bool(getattr(cfg_args, "white_background", False)) else torch.zeros(3, dtype=torch.float32, device=device)


def _make_ssim_metric(device: str):
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        return StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    except Exception:
        return None


class _torch_no_grad:
    def __enter__(self):
        import torch

        self._ctx = torch.no_grad()
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
