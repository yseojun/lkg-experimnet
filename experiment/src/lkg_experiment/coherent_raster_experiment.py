from __future__ import annotations

import csv
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from PIL import Image


RGB_SUBPIXELS = 3
DEFAULT_CLUSTERS = (2, 4, 8, 16)


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    cluster_size: int
    use_remapping: bool
    reuse_enabled: bool
    group: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimingStats:
    fps: float
    frame_ms: float
    peak_vram_gb: float


@dataclass(frozen=True)
class MetricStats:
    psnr_mean: float
    psnr_std: float
    ssim_mean: float
    ssim_std: float
    lpips_mean: float
    lpips_std: float
    metric_view_count: int


def parse_cluster_values(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
        clusters = tuple(int(part) for part in values)
    else:
        clusters = tuple(int(part) for part in value)
    if not clusters:
        raise ValueError("at least one cluster value is required")
    if any(cluster <= 0 for cluster in clusters):
        raise ValueError("cluster values must be positive")
    return clusters


def build_experiment_variants(
    clusters: Sequence[int] = DEFAULT_CLUSTERS,
    *,
    ablation_cluster: Optional[int] = None,
    include_without_remap: bool = True,
    include_without_reuse: bool = True,
) -> list[ExperimentVariant]:
    cluster_values = parse_cluster_values(clusters)
    if ablation_cluster is None:
        ablation_cluster = 8 if 8 in cluster_values else cluster_values[-1]
    if ablation_cluster <= 0:
        raise ValueError("ablation_cluster must be positive")

    variants = [
        ExperimentVariant(
            name=f"cluster_{cluster}",
            cluster_size=cluster,
            use_remapping=True,
            reuse_enabled=True,
            group="cluster_sweep",
        )
        for cluster in cluster_values
    ]
    if include_without_remap:
        variants.append(
            ExperimentVariant(
                name="without_remap",
                cluster_size=int(ablation_cluster),
                use_remapping=False,
                reuse_enabled=True,
                group="ablation",
            )
        )
    if include_without_reuse:
        variants.extend(
            [
                ExperimentVariant(
                    name="without_reuse",
                    cluster_size=1,
                    use_remapping=True,
                    reuse_enabled=False,
                    group="ablation",
                ),
                ExperimentVariant(
                    name="without_reuse_without_remap",
                    cluster_size=1,
                    use_remapping=False,
                    reuse_enabled=False,
                    group="ablation",
                ),
            ]
        )
    return variants


def cluster_index_from_view_index(view_index_hwc: np.ndarray, cluster_size: int) -> np.ndarray:
    view_index = np.asarray(view_index_hwc)
    if view_index.ndim != 3 or view_index.shape[2] != RGB_SUBPIXELS:
        raise ValueError("view_index_hwc must have shape (height, width, 3)")
    if cluster_size <= 0:
        raise ValueError("cluster_size must be positive")
    if np.any(view_index < 0):
        raise ValueError("view_index_hwc must not contain negative view ids")
    return (view_index.astype(np.int64, copy=False) // int(cluster_size)).astype(np.int32)


def tensor_to_hwc_uint8(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().clamp(0.0, 1.0)
        if getattr(image, "is_cuda", False):
            image = image.cpu()
        image = image.numpy()
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError("image tensor must have shape (3,H,W) or (H,W,3)")
    if array.shape[0] == RGB_SUBPIXELS:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != RGB_SUBPIXELS:
        raise ValueError("image tensor must have three RGB channels")
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    else:
        array = np.clip(array, 0, 255)
    return array.astype(np.uint8, copy=False)


def image_artifact_path(variant_name: str, image_name: str, *, output_prefix: str = "") -> Path:
    safe_variant = _safe_artifact_name(variant_name)
    safe_image_name = Path(image_name).name
    safe_prefix = _safe_artifact_prefix(output_prefix)
    filename = f"{safe_prefix}_{safe_image_name}" if safe_prefix else safe_image_name
    return Path("images") / safe_variant / filename


def read_metrics_csv(path: Path | str) -> list[dict[str, str]]:
    metrics_path = Path(path).expanduser()
    if not metrics_path.is_file():
        return []
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def remove_matching_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_prefix: str,
    camera_split: str,
    camera_index: int,
) -> list[dict[str, Any]]:
    prefix = str(output_prefix)
    split = str(camera_split)
    index = str(camera_index)
    kept: list[dict[str, Any]] = []
    for row in rows:
        if prefix and str(row.get("output_prefix", "")) == prefix:
            continue
        if not prefix and str(row.get("camera_split", "")) == split and str(row.get("camera_index", "")) == index:
            continue
        kept.append(dict(row))
    return kept


def select_metric_view_indices(
    view_count: int,
    *,
    max_metric_views: int = 0,
    stride: int = 1,
) -> list[int]:
    if view_count <= 0:
        raise ValueError("view_count must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    indices = list(range(0, int(view_count), int(stride)))
    if max_metric_views > 0:
        indices = indices[: int(max_metric_views)]
    return indices


def reference_interlace_from_views(rendered_views: Any, viewpoint_index_hwc: np.ndarray):
    import torch

    if rendered_views.ndim != 4:
        raise ValueError("rendered_views must have shape (V,3,H,W) or (V,H,W,3)")
    if rendered_views.shape[1] != RGB_SUBPIXELS and rendered_views.shape[-1] == RGB_SUBPIXELS:
        rendered_views = rendered_views.permute(0, 3, 1, 2).contiguous()
    if rendered_views.shape[1] != RGB_SUBPIXELS:
        raise ValueError("rendered_views must have three RGB channels")

    view_index = torch.as_tensor(
        viewpoint_index_hwc,
        device=rendered_views.device,
        dtype=torch.long,
    )
    if view_index.ndim != 3 or view_index.shape[2] != RGB_SUBPIXELS:
        raise ValueError("viewpoint_index_hwc must have shape (height, width, 3)")
    height, width, _ = viewpoint_index.shape
    if rendered_views.shape[-2:] != (height, width):
        raise ValueError("rendered view size does not match viewpoint index size")
    output = torch.empty((RGB_SUBPIXELS, height, width), dtype=rendered_views.dtype, device=rendered_views.device)
    for channel in range(RGB_SUBPIXELS):
        channel_views = rendered_views[:, channel]
        output[channel] = channel_views.gather(0, view_index[:, :, channel].unsqueeze(0)).squeeze(0)
    return output


class ArtifactWriter:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, manifest: Mapping[str, Any]) -> Path:
        path = self.root / "manifest.json"
        path.write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def write_metrics_json(self, rows: Sequence[Mapping[str, Any]]) -> Path:
        path = self.root / "metrics.json"
        path.write_text(json.dumps(_json_ready(list(rows)), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def write_metrics_csv(self, rows: Sequence[Mapping[str, Any]]) -> Path:
        path = self.root / "metrics.csv"
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        preferred = [
            "camera_split",
            "camera_index",
            "output_prefix",
            "variant",
            "group",
            "cluster_size",
            "use_remapping",
            "reuse_enabled",
            "fps",
            "frame_ms",
            "peak_vram_gb",
            "psnr_mean",
            "psnr_std",
            "ssim_mean",
            "ssim_std",
            "lpips_mean",
            "lpips_std",
            "metric_view_count",
        ]
        fieldnames = [field for field in preferred if any(field in row for row in rows)]
        for row in rows:
            for field in row:
                if field not in fieldnames:
                    fieldnames.append(field)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _csv_ready(row.get(field, "")) for field in fieldnames})
        return path

    def write_mapping_npz(
        self,
        view_index_hwc: np.ndarray,
        cluster_indices: Mapping[str, np.ndarray],
        *,
        view_labels: Optional[np.ndarray] = None,
        source_view_index_hwc: Optional[np.ndarray] = None,
    ) -> Path:
        mapping_dir = self.root / "mappings"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "view_index_hwc": np.asarray(view_index_hwc, dtype=np.int32),
        }
        if view_labels is not None:
            payload["view_labels"] = np.asarray(view_labels, dtype=np.int32)
        if source_view_index_hwc is not None:
            payload["source_view_index_hwc"] = np.asarray(source_view_index_hwc, dtype=np.int32)
        for name, cluster_index in cluster_indices.items():
            payload[f"{name}_cluster_index_hwc"] = np.asarray(cluster_index, dtype=np.int32)
        path = mapping_dir / "raw_mapping.npz"
        np.savez_compressed(path, **payload)
        return path

    def save_index_previews(self, stem: str, index_hwc: np.ndarray) -> list[Path]:
        index = np.asarray(index_hwc)
        if index.ndim != 3 or index.shape[2] != RGB_SUBPIXELS:
            raise ValueError("index_hwc must have shape (height, width, 3)")
        mapping_dir = self.root / "mappings"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        colorized_planes = []
        for channel, suffix in enumerate(("r", "g", "b")):
            path = mapping_dir / f"{stem}_{suffix}.png"
            colorized = _colorize_index_plane(index[:, :, channel])
            Image.fromarray(colorized, mode="RGB").save(path)
            colorized_planes.append(colorized)
            paths.append(path)
        combined_path = mapping_dir / f"{stem}_rgb.png"
        Image.fromarray(np.concatenate(colorized_planes, axis=1), mode="RGB").save(combined_path)
        paths.append(combined_path)
        return paths

    def save_mapping_values(self, stem: str, index_hwc: np.ndarray) -> dict[str, Any]:
        index = np.asarray(index_hwc)
        if index.ndim != 3 or index.shape[2] != RGB_SUBPIXELS:
            raise ValueError("index_hwc must have shape (height, width, 3)")
        mapping_dir = self.root / "mappings"
        mapping_dir.mkdir(parents=True, exist_ok=True)

        min_value = int(index.min()) if index.size else 0
        max_value = int(index.max()) if index.size else 0
        if min_value >= 0 and max_value <= int(np.iinfo(np.uint16).max):
            dtype_name = "uint16"
            encoded = np.ascontiguousarray(index.astype("<u2", copy=False))
        else:
            dtype_name = "int32"
            encoded = np.ascontiguousarray(index.astype("<i4", copy=False))

        path = mapping_dir / f"{stem}_values.{dtype_name}.bin"
        path.write_bytes(encoded.tobytes(order="C"))
        return {
            "path": _relative_posix(self.root, path),
            "dtype": dtype_name,
            "shape": [int(index.shape[0]), int(index.shape[1]), int(index.shape[2])],
            "min": min_value,
            "max": max_value,
        }

    def save_tensor_image(self, relative_path: Path | str, image: Any) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(tensor_to_hwc_uint8(image), mode="RGB").save(path)
        return path


def _safe_artifact_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return sanitized or "artifact"


def _safe_artifact_prefix(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return sanitized


class OfficialGsplatRenderer:
    def __init__(
        self,
        splats: Mapping[str, Any],
        *,
        sh_degree: int,
        near_plane: float,
        far_plane: float,
        camera_model: str,
        background: Any,
        color_mode: str,
        packed: bool = False,
        rasterize_mode: str = "classic",
    ) -> None:
        import torch
        from gsplat.rendering import rasterization

        self.rasterization = rasterization
        self.means = splats["means"].contiguous()
        self.quats = splats["quats"].contiguous()
        self.scales = torch.exp(splats["scales"]).contiguous()
        self.opacities = torch.sigmoid(splats["opacities"]).contiguous()
        if color_mode == "dc":
            self.colors = splats["sh0"].contiguous()
            self.sh_degree = 0
        elif color_mode == "sh":
            self.colors = torch.cat([splats["sh0"], splats["shN"]], dim=1).contiguous()
            self.sh_degree = int(sh_degree)
        else:
            raise ValueError("color_mode must be 'sh' or 'dc'")
        self.near_plane = float(near_plane)
        self.far_plane = float(far_plane)
        self.camera_model = str(camera_model)
        self.background = background
        self.packed = bool(packed)
        self.rasterize_mode = str(rasterize_mode)

    def render_views(self, *, viewmats: Any, K: Any, width: int, height: int, tile_size: int):
        import torch

        if viewmats.ndim != 3:
            raise ValueError("viewmats must have shape (V,4,4)")
        count = int(viewmats.shape[0])
        Ks = K.unsqueeze(0).expand(count, -1, -1).contiguous()
        backgrounds = None
        if self.background is not None:
            backgrounds = self.background.unsqueeze(0).expand(count, -1).contiguous()
        colors, _, _ = self.rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
            viewmats=viewmats.contiguous(),
            Ks=Ks,
            width=int(width),
            height=int(height),
            sh_degree=self.sh_degree,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            packed=self.packed,
            tile_size=int(tile_size),
            backgrounds=backgrounds,
            rasterize_mode=self.rasterize_mode,
            camera_model=self.camera_model,
        )
        return colors[..., :3].permute(0, 3, 1, 2).clamp(0.0, 1.0).contiguous()


class CoherentRasterRenderer:
    def __init__(
        self,
        splats: Mapping[str, Any],
        *,
        sh_degree: int,
        near_plane: float,
        far_plane: float,
        camera_model: str,
        background: Any,
        color_mode: str,
    ) -> None:
        import torch
        from coherent_raster.utils.utils_coherent_raster import unpad, unpatchify_image_shape_matrix
        from gsplat.rendering_coherent_raster import rasterization_CR

        self.rasterization_CR = rasterization_CR
        self.unpatchify_image_shape_matrix = unpatchify_image_shape_matrix
        self.unpad = unpad
        self.means = splats["means"].contiguous()
        self.quats = splats["quats"].contiguous()
        self.scales = torch.exp(splats["scales"]).contiguous()
        self.opacities = torch.sigmoid(splats["opacities"]).contiguous()
        if color_mode == "dc":
            self.colors = splats["sh0"].contiguous()
            self.sh_degree = 0
        elif color_mode == "sh":
            self.colors = torch.cat([splats["sh0"], splats["shN"]], dim=1).contiguous()
            self.sh_degree = int(sh_degree)
        else:
            raise ValueError("color_mode must be 'sh' or 'dc'")
        self.near_plane = float(near_plane)
        self.far_plane = float(far_plane)
        self.camera_model = str(camera_model)
        self.background = background

    def render(
        self,
        *,
        adjacent_viewmats: Any,
        K: Any,
        view_idx_matrix: Any,
        subpixel_coord_matrix: Any,
        width: int,
        height: int,
        tile_size: int,
        debug: bool = False,
    ):
        colors, _, _ = self.rasterization_CR(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
            adjacent_viewmats=adjacent_viewmats,
            Ks=K.unsqueeze(0),
            view_idx_matrix=view_idx_matrix,
            subpixel_coord_matrix=subpixel_coord_matrix,
            width=int(width),
            height=int(height),
            sh_degree=self.sh_degree,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            backgrounds=self.background,
            camera_model=self.camera_model,
            tile_size=int(tile_size),
            is_debug=debug,
        )
        colors = self.unpatchify_image_shape_matrix(colors)
        colors = self.unpad(colors, int(height), int(width))
        return colors.clamp(0.0, 1.0).contiguous()


class OrbitViewSynthesizer:
    def __init__(
        self,
        *,
        view_labels: np.ndarray,
        source_view_count: int,
        cluster_size: int,
        view_degree: float,
        orbit_direction: int,
        device: str,
    ) -> None:
        import torch

        if source_view_count <= 0:
            raise ValueError("source_view_count must be positive")
        if cluster_size <= 0:
            raise ValueError("cluster_size must be positive")
        labels = np.asarray(view_labels, dtype=np.int64)
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError("view_labels must be a non-empty 1D array")
        if int(labels.min()) < 0 or int(labels.max()) >= int(source_view_count):
            raise ValueError("view_labels must be inside source_view_count")

        start = float(view_degree) / 2.0
        end = -float(view_degree) / 2.0
        if int(orbit_direction) > 0:
            start, end = -start, -end
        all_angles = torch.linspace(start, end, int(source_view_count), device=device)
        label_t = torch.from_numpy(labels).to(device=device, dtype=torch.long)
        angles_rad = torch.deg2rad(all_angles.index_select(0, label_t)).view(-1, 1, 1)

        self.device = device
        self.n_valid_views = int(labels.size)
        self.cluster_size = int(cluster_size)
        self.n_pad = (self.cluster_size - (self.n_valid_views % self.cluster_size)) % self.cluster_size
        self.n_clusters = (self.n_valid_views + self.n_pad) // self.cluster_size
        total = self.n_clusters * self.cluster_size
        self.sin_vals = torch.sin(angles_rad)
        self.cos_vals = torch.cos(angles_rad)
        self.one_minus_cos = 1.0 - self.cos_vals
        self.identity_3x3 = torch.eye(3, device=device).unsqueeze(0)
        self.viewmats_buffer = torch.eye(4, device=device).unsqueeze(0).repeat(total, 1, 1).contiguous()

    def __call__(self, c2w: Any, orbit_center: Any):
        import torch
        import torch.nn.functional as F

        R_orig = c2w[:3, :3]
        t_orig = c2w[:3, 3]
        ref_up = R_orig[:, 1]

        R_skew = torch.zeros((3, 3), device=self.device, dtype=c2w.dtype)
        R_skew[0, 1], R_skew[0, 2] = -ref_up[2], ref_up[1]
        R_skew[1, 0], R_skew[1, 2] = ref_up[2], -ref_up[0]
        R_skew[2, 0], R_skew[2, 1] = -ref_up[1], ref_up[0]
        R_skew = R_skew.unsqueeze(0)
        R_rot_batch = self.identity_3x3 + self.sin_vals * R_skew + self.one_minus_cos * torch.matmul(R_skew, R_skew)

        rel_pos = (t_orig - orbit_center).view(1, 3, 1)
        new_t = orbit_center.view(1, 3) + torch.matmul(R_rot_batch, rel_pos).squeeze(-1)

        vec_cam_to_center = F.normalize(orbit_center.unsqueeze(0) - new_t, p=2, dim=1)
        orig_cam_to_center = F.normalize(orbit_center - t_orig, p=2, dim=0)
        orig_z_axis = R_orig[:, 2]
        z_sign = torch.sign(torch.dot(orig_cam_to_center, orig_z_axis))
        new_z_axis = vec_cam_to_center * z_sign

        ref_up_batch = ref_up.unsqueeze(0).expand(self.n_valid_views, -1)
        new_x_axis = F.normalize(torch.linalg.cross(ref_up_batch, new_z_axis), p=2, dim=1)
        orig_cross_up_z = torch.linalg.cross(ref_up, orig_z_axis)
        x_sign = torch.sign(torch.dot(orig_cross_up_z, R_orig[:, 0]))
        new_x_axis = new_x_axis * x_sign

        new_y_axis = torch.linalg.cross(new_z_axis, new_x_axis)
        y_sign = torch.sign(torch.sum(new_y_axis * ref_up_batch, dim=1, keepdim=True))
        new_y_axis = new_y_axis * y_sign
        new_R = torch.stack([new_x_axis, new_y_axis, new_z_axis], dim=2)

        w2c_R = new_R.transpose(1, 2)
        w2c_t = -torch.matmul(w2c_R, new_t.unsqueeze(-1)).squeeze(-1)
        self.viewmats_buffer[: self.n_valid_views, :3, :3] = w2c_R
        self.viewmats_buffer[: self.n_valid_views, :3, 3] = w2c_t
        if self.n_pad > 0:
            self.viewmats_buffer[self.n_valid_views :, :3, :3] = w2c_R[-1]
            self.viewmats_buffer[self.n_valid_views :, :3, 3] = w2c_t[-1]
        return self.viewmats_buffer.view(self.n_clusters, self.cluster_size, 4, 4)


def time_interlaced_render(
    renderer: CoherentRasterRenderer,
    *,
    adjacent_viewmats: Any,
    K: Any,
    view_idx_matrix: Any,
    subpixel_coord_matrix: Any,
    width: int,
    height: int,
    tile_size: int,
    warmup_iters: int,
    measure_iters: int,
    debug: bool = False,
) -> tuple[Any, TimingStats]:
    import torch

    if warmup_iters < 0 or measure_iters <= 0:
        raise ValueError("warmup_iters must be non-negative and measure_iters must be positive")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    last = None
    total = 0.0
    count = 0
    with torch.no_grad():
        for iteration in range(int(warmup_iters) + int(measure_iters)):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            last = renderer.render(
                adjacent_viewmats=adjacent_viewmats,
                K=K,
                view_idx_matrix=view_idx_matrix,
                subpixel_coord_matrix=subpixel_coord_matrix,
                width=width,
                height=height,
                tile_size=tile_size,
                debug=debug,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = max(time.perf_counter() - start, 1e-12)
            if iteration >= warmup_iters:
                total += elapsed
                count += 1
    fps = count / total if total > 0.0 else 0.0
    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_reserved() / float(2**30)
    return last, TimingStats(fps=fps, frame_ms=(1000.0 / fps if fps > 0.0 else math.inf), peak_vram_gb=peak_vram_gb)


def constant_view_index_like(view_idx_matrix: Any, view_id: int):
    import torch

    return torch.full_like(view_idx_matrix, int(view_id), dtype=torch.uint32)


def compute_metric_stats(
    original_renderer: OfficialGsplatRenderer,
    coherent_renderer: CoherentRasterRenderer,
    *,
    flat_viewmats: Any,
    adjacent_viewmats: Any,
    K: Any,
    subpixel_coord_matrix: Any,
    view_idx_template: Any,
    metric_view_indices: Sequence[int],
    width: int,
    height: int,
    tile_size: int,
    require_lpips: bool = False,
) -> MetricStats:
    import torch

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    ssim_metric = _make_ssim_metric(device=K.device)
    lpips_metric = _make_lpips_metric(device=K.device, required=require_lpips)

    with torch.no_grad():
        for view_id in metric_view_indices:
            gt = original_renderer.render_views(
                viewmats=flat_viewmats[int(view_id) : int(view_id) + 1],
                K=K,
                width=width,
                height=height,
                tile_size=tile_size,
            )[0]
            view_idx_matrix = constant_view_index_like(view_idx_template, int(view_id))
            cr = coherent_renderer.render(
                adjacent_viewmats=adjacent_viewmats,
                K=K,
                view_idx_matrix=view_idx_matrix,
                subpixel_coord_matrix=subpixel_coord_matrix,
                width=width,
                height=height,
                tile_size=tile_size,
            )
            psnr_values.append(_psnr_torch(cr, gt))
            if ssim_metric is not None:
                ssim_values.append(float(ssim_metric(cr.unsqueeze(0), gt.unsqueeze(0)).item()))
            if lpips_metric is not None:
                lpips_values.append(float(lpips_metric(cr.unsqueeze(0), gt.unsqueeze(0)).item()))
    return MetricStats(
        psnr_mean=_mean(psnr_values),
        psnr_std=_std(psnr_values),
        ssim_mean=_mean(ssim_values),
        ssim_std=_std(ssim_values),
        lpips_mean=_mean(lpips_values),
        lpips_std=_std(lpips_values),
        metric_view_count=len(metric_view_indices),
    )


def compact_viewpoint_index(viewpoint_index: np.ndarray, source_view_count: int) -> tuple[np.ndarray, np.ndarray]:
    used_views = np.unique(np.asarray(viewpoint_index, dtype=np.int32))
    if used_views.size == 0:
        raise ValueError("viewpoint index does not reference any views")
    if int(used_views.min()) < 0 or int(used_views.max()) >= int(source_view_count):
        raise ValueError("viewpoint index uses values outside source_view_count")
    remap = np.full((int(source_view_count),), -1, dtype=np.int32)
    remap[used_views] = np.arange(used_views.size, dtype=np.int32)
    compacted = remap[viewpoint_index]
    if np.any(compacted < 0):
        raise RuntimeError("internal compact remap failed")
    return compacted.astype(np.int32, copy=False), used_views.astype(np.int32, copy=False)


def load_viewpoint_index_file(path: Path | str, width: int, height: int) -> tuple[np.ndarray, Optional[int], dict[str, Any]]:
    index_path = Path(path).expanduser()
    if not index_path.exists():
        raise FileNotFoundError(f"viewpoint index file not found: {index_path}")
    loaded = np.load(str(index_path), allow_pickle=False)
    metadata: dict[str, Any] = {"path": str(index_path)}
    view_count = None
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            viewpoint_index = loaded["viewpoint_index"]
            if "view_count" in loaded:
                view_count = int(np.asarray(loaded["view_count"]).item())
            for key in ("quilt_cols", "quilt_rows", "layout", "quantize"):
                if key in loaded:
                    metadata[key] = np.asarray(loaded[key]).item()
        finally:
            loaded.close()
    else:
        viewpoint_index = loaded

    if viewpoint_index.ndim == 1:
        expected = int(width) * int(height) * RGB_SUBPIXELS
        if viewpoint_index.size != expected:
            raise ValueError(f"flat viewpoint index has {viewpoint_index.size} entries, expected {expected}")
        viewpoint_index = viewpoint_index.reshape(int(height), int(width), RGB_SUBPIXELS)
    if viewpoint_index.shape != (int(height), int(width), RGB_SUBPIXELS):
        raise ValueError(
            f"viewpoint index shape must be {(int(height), int(width), RGB_SUBPIXELS)}; got {viewpoint_index.shape}"
        )
    return viewpoint_index.astype(np.int32, copy=False), view_count, metadata


def regenerate_mapping_previews(run_root: Path | str) -> list[Path]:
    run_path = Path(run_root).expanduser()
    raw_mapping_path = run_path / "mappings" / "raw_mapping.npz"
    if not raw_mapping_path.exists():
        return []

    written: list[Path] = []
    loaded = np.load(raw_mapping_path, allow_pickle=False)
    try:
        writer = ArtifactWriter(run_path)
        for key in sorted(loaded.files):
            if not key.endswith("_hwc"):
                continue
            value = np.asarray(loaded[key])
            if value.ndim != 3 or value.shape[2] != RGB_SUBPIXELS:
                continue
            stem = _mapping_stem_from_npz_key(key)
            written.extend(writer.save_index_previews(stem, value))
            if stem in {"view_index", "source_view_index"}:
                metadata = writer.save_mapping_values(stem, value)
                written.append(run_path / metadata["path"])
    finally:
        loaded.close()
    return written


def build_experiment_web_assets(
    experiments_root: Path | str,
    *,
    regenerate_previews: bool = False,
) -> Path:
    root = Path(experiments_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    if regenerate_previews:
        for run_root in _iter_experiment_run_dirs(root):
            regenerate_mapping_previews(run_root)

    index = build_experiment_web_index(root)
    index_path = root / "experiments.json"
    index_path.write_text(json.dumps(_json_ready(index), indent=2, sort_keys=True), encoding="utf-8")
    (root / "index.html").write_text(_WEB_INDEX_HTML, encoding="utf-8")
    (root / "viewer.css").write_text(_WEB_VIEWER_CSS, encoding="utf-8")
    (root / "viewer.js").write_text(_WEB_VIEWER_JS, encoding="utf-8")
    return index_path


def build_experiment_web_index(experiments_root: Path | str) -> dict[str, Any]:
    root = Path(experiments_root).expanduser()
    runs = [_build_web_run_entry(root, run_root) for run_root in _iter_experiment_run_dirs(root)]
    runs.sort(key=lambda item: (str(item.get("created_at") or ""), str(item.get("id") or "")), reverse=True)
    return {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": runs,
    }


def _iter_experiment_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "manifest.json").exists())


def _build_web_run_entry(root: Path, run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "manifest.json"
    metrics_path = run_root / "metrics.json"
    manifest = _read_json_file(manifest_path, default={})
    metrics = _read_metrics(run_root)
    variants = _collect_web_variants(root, run_root, manifest, metrics)
    mappings = _collect_mapping_previews(
        root,
        run_root,
        width=_optional_int(manifest.get("width")),
        height=_optional_int(manifest.get("height")),
    )
    _attach_mapping_transforms(mappings, variants)
    return {
        "id": run_root.name,
        "label": str(manifest.get("run_id") or run_root.name),
        "path": _relative_posix(root, run_root),
        "created_at": manifest.get("created_at"),
        "manifest_path": _relative_posix(root, manifest_path),
        "metrics_path": _relative_posix(root, metrics_path) if metrics_path.exists() else None,
        "width": manifest.get("width"),
        "height": manifest.get("height"),
        "source_view_count": manifest.get("source_view_count"),
        "render_view_count": manifest.get("render_view_count"),
        "camera": manifest.get("camera"),
        "dataset_category": manifest.get("dataset_category"),
        "checkpoint": manifest.get("checkpoint"),
        "metrics": metrics,
        "variants": variants,
        "mappings": mappings,
    }


def _collect_web_variants(
    root: Path,
    run_root: Path,
    manifest: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    variants_by_name: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in manifest.get("variants", []) or []:
        if isinstance(item, Mapping):
            name = str(item.get("name") or "")
            variant = dict(item)
        else:
            name = str(item)
            variant = {"name": name}
        if not name:
            continue
        variants_by_name[name] = variant
        order.append(name)

    for row in metrics:
        name = str(row.get("variant") or "")
        if name and name not in variants_by_name:
            variants_by_name[name] = {"name": name}
            order.append(name)

    images_root = run_root / "images"
    if images_root.exists():
        for image_dir in sorted(path for path in images_root.iterdir() if path.is_dir()):
            name = image_dir.name
            if name not in variants_by_name:
                variants_by_name[name] = {"name": name}
                order.append(name)

    variants = []
    for name in order:
        variant = dict(variants_by_name[name])
        variant["images"] = _collect_variant_images(root, run_root, name)
        variants.append(variant)
    return variants


def _collect_variant_images(root: Path, run_root: Path, variant_name: str) -> dict[str, str]:
    image_dir = run_root / "images" / variant_name
    outputs = {}
    for key, filename in (
        ("looking_glass_tensor", "looking_glass_tensor.png"),
        ("reference_interlaced", "reference_interlaced.png"),
        ("abs_error", "abs_error.png"),
    ):
        path = image_dir / filename
        if path.exists():
            outputs[key] = _relative_posix(root, path)
    return outputs


def _collect_mapping_previews(
    root: Path,
    run_root: Path,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> dict[str, dict[str, Any]]:
    mapping_dir = run_root / "mappings"
    previews: dict[str, dict[str, Any]] = {}
    if not mapping_dir.exists():
        return previews
    for path in sorted(mapping_dir.glob("*.png")):
        for suffix in ("rgb", "r", "g", "b"):
            marker = f"_{suffix}.png"
            if path.name.endswith(marker):
                stem = path.name[: -len(marker)]
                previews.setdefault(stem, {})[suffix] = _relative_posix(root, path)
                break
    for path in sorted(mapping_dir.glob("*_values.*.bin")):
        name = path.name
        if "_values." not in name:
            continue
        stem, tail = name.split("_values.", 1)
        dtype_name = tail[: -len(".bin")] if tail.endswith(".bin") else tail
        entry: dict[str, Any] = {
            "path": _relative_posix(root, path),
            "dtype": dtype_name,
            "channels": RGB_SUBPIXELS,
        }
        if width is not None and height is not None:
            entry["shape"] = [int(height), int(width), RGB_SUBPIXELS]
        previews.setdefault(stem, {})["values"] = entry
    return previews


def _attach_mapping_transforms(mappings: dict[str, dict[str, Any]], variants: Sequence[Mapping[str, Any]]) -> None:
    for variant in variants:
        name = str(variant.get("name") or "")
        if not name:
            continue
        stem = f"{name}_cluster_index"
        if stem not in mappings or "values" in mappings[stem]:
            continue
        cluster_size = _optional_int(variant.get("cluster_size"))
        if cluster_size is None:
            continue
        mappings[stem]["transform"] = {
            "type": "cluster_from_view",
            "source": "view_index",
            "cluster_size": int(cluster_size),
        }


def _read_metrics(run_root: Path) -> list[dict[str, Any]]:
    metrics_json = run_root / "metrics.json"
    if metrics_json.exists():
        metrics = _read_json_file(metrics_json, default=[])
        if isinstance(metrics, list):
            return [dict(item) for item in metrics if isinstance(item, Mapping)]
        if isinstance(metrics, Mapping):
            return [dict(metrics)]

    metrics_csv = run_root / "metrics.csv"
    if metrics_csv.exists():
        with metrics_csv.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    return []


def _read_json_file(path: Path, *, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _mapping_stem_from_npz_key(key: str) -> str:
    if key == "view_index_hwc":
        return "view_index"
    if key == "source_view_index_hwc":
        return "source_view_index"
    return key[: -len("_hwc")] if key.endswith("_hwc") else key


def _optional_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _relative_posix(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


_WEB_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Coherent Raster Experiments</title>
  <link rel="stylesheet" href="viewer.css">
</head>
<body>
  <header class="topbar">
    <div>
      <h1>Coherent Raster Experiments</h1>
      <p id="run-summary">Loading experiment index...</p>
    </div>
    <div class="server-note">Serve this folder with a static HTTP server.</div>
  </header>

  <main>
    <section class="controls" aria-label="Experiment controls">
      <label>
        Experiment
        <select id="run-select"></select>
      </label>
      <label>
        Variant
        <select id="variant-select"></select>
      </label>
      <label>
        Image
        <select id="image-select"></select>
      </label>
      <label>
        Overlay
        <select id="mapping-select"></select>
      </label>
      <div class="zoom-controls" aria-label="Zoom controls">
        <button type="button" id="zoom-fit-button">Fit</button>
        <button type="button" id="zoom-actual-button">1:1</button>
        <button type="button" id="zoom-out-button">-</button>
        <button type="button" id="zoom-in-button">+</button>
        <button type="button" id="zoom-numbers-button">Show Numbers</button>
        <span id="zoom-label">Zoom: -</span>
      </div>
    </section>

    <section class="viewer-layout" aria-label="Visual debug view">
      <article class="viewer-card">
        <div class="panel-head">
          <h2>Image View</h2>
          <span id="image-label"></span>
        </div>
        <div class="canvas-frame" id="canvas-frame">
          <canvas id="image-canvas" aria-label="Rendered image with view id overlay"></canvas>
          <div id="canvas-status"></div>
        </div>
      </article>
    </section>

    <section class="metric-panel">
      <div class="panel-head">
        <h2>Metrics</h2>
        <span id="metric-count"></span>
      </div>
      <div class="table-wrap">
        <table id="metrics-table">
          <thead></thead>
          <tbody></tbody>
        </table>
      </div>
    </section>
  </main>

  <script src="viewer.js" defer></script>
</body>
</html>
"""


_WEB_VIEWER_CSS = """* {
  box-sizing: border-box;
}

:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --surface: #ffffff;
  --surface-soft: #eef1f4;
  --text: #18202a;
  --muted: #687384;
  --line: #d9dee6;
  --accent: #0f6b63;
  --accent-soft: #d8ebe8;
  --shadow: 0 12px 28px rgba(21, 32, 43, 0.08);
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.topbar {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 24px;
  padding: 22px 28px 16px;
  border-bottom: 1px solid var(--line);
  background: var(--surface);
}

h1,
h2,
p {
  margin: 0;
}

h1 {
  font-size: 22px;
  font-weight: 700;
}

h2 {
  font-size: 14px;
  font-weight: 700;
}

#run-summary,
.server-note,
.panel-head span {
  color: var(--muted);
  font-size: 12px;
}

main {
  padding: 18px 28px 32px;
}

.controls {
  display: grid;
  grid-template-columns: minmax(220px, 2fr) repeat(3, minmax(140px, 1fr)) minmax(360px, 1.6fr);
  gap: 12px;
  margin-bottom: 16px;
}

label {
  display: grid;
  gap: 6px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 600;
}

select {
  width: 100%;
  min-height: 38px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  padding: 0 10px;
  font: inherit;
}

.zoom-controls {
  display: flex;
  align-items: end;
  gap: 8px;
  min-height: 58px;
}

button {
  min-height: 38px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  padding: 0 11px;
  font: inherit;
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
}

button:hover {
  border-color: var(--accent);
  color: var(--accent);
}

#zoom-label {
  min-width: 82px;
  padding-bottom: 10px;
  color: var(--muted);
  font-size: 12px;
}

.metric-panel,
.viewer-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  box-shadow: var(--shadow);
}

.metric-panel {
  margin-top: 16px;
}

.panel-head {
  min-height: 42px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
}

.table-wrap {
  overflow: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

th,
td {
  padding: 8px 10px;
  border-bottom: 1px solid var(--line);
  text-align: right;
  white-space: nowrap;
}

th:first-child,
td:first-child {
  text-align: left;
}

th {
  color: var(--muted);
  background: var(--surface-soft);
  font-weight: 700;
}

tr.is-selected td {
  background: var(--accent-soft);
}

.viewer-layout {
  display: block;
}

.canvas-frame {
  position: relative;
  min-height: 560px;
  height: calc(100vh - 250px);
  max-height: 980px;
  overflow: auto;
  background: #101418;
  cursor: grab;
  user-select: none;
}

.canvas-frame.is-dragging {
  cursor: grabbing;
}

#image-canvas {
  display: block;
  width: 100%;
  height: 100%;
  image-rendering: pixelated;
}

#canvas-status {
  position: absolute;
  left: 12px;
  bottom: 10px;
  max-width: min(720px, calc(100% - 24px));
  color: #c6ced8;
  background: rgba(16, 20, 24, 0.78);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 6px;
  padding: 6px 8px;
  font-size: 12px;
  pointer-events: none;
}

@media (max-width: 1100px) {
  .controls {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .zoom-controls {
    grid-column: span 2;
  }
}

@media (max-width: 720px) {
  .topbar {
    align-items: flex-start;
    flex-direction: column;
  }

  main,
  .topbar {
    padding-left: 14px;
    padding-right: 14px;
  }

  .controls {
    grid-template-columns: 1fr;
  }

  .zoom-controls {
    grid-column: span 1;
    flex-wrap: wrap;
  }

  .canvas-frame {
    height: 58vh;
    min-height: 420px;
  }
}
"""


_WEB_VIEWER_JS = """const IMAGE_LABELS = {
  looking_glass_tensor: "Rendered",
  reference_interlaced: "Reference",
  abs_error: "Abs error"
};

const METRIC_COLUMNS = [
  "variant",
  "group",
  "cluster_size",
  "use_remapping",
  "reuse_enabled",
  "fps",
  "frame_ms",
  "peak_vram_gb",
  "psnr_mean",
  "ssim_mean",
  "lpips_mean",
  "metric_view_count"
];

const METRIC_FORMATTERS = {
  cluster_size: formatInteger,
  use_remapping: formatBool,
  reuse_enabled: formatBool,
  fps: (value) => formatFixed(value, 2),
  frame_ms: formatInteger,
  peak_vram_gb: (value) => formatFixed(value, 2),
  psnr_mean: (value) => formatFixed(value, 2),
  psnr_std: (value) => formatFixed(value, 2),
  ssim_mean: (value) => formatFixed(value, 3),
  ssim_std: (value) => formatFixed(value, 3),
  lpips_mean: (value) => formatFixed(value, 3),
  lpips_std: (value) => formatFixed(value, 3),
  metric_view_count: formatInteger
};

const NUMBER_SCALE = 22;
const MAX_NUMBERED_PIXELS = 14000;
const MIN_SCALE = 0.02;
const MAX_SCALE = 128;

let indexData = null;
let state = {
  runId: "",
  variant: "",
  imageMode: "looking_glass_tensor",
  mapping: "view_index"
};

const els = {};
const viewer = {
  canvas: null,
  ctx: null,
  frame: null,
  image: null,
  imagePath: "",
  imageLoaded: false,
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  dragging: false,
  dragX: 0,
  dragY: 0,
  mappingName: "",
  mappingData: null,
  mappingError: "",
  mappingCache: new Map()
};

window.addEventListener("DOMContentLoaded", init);

async function init() {
  cacheElements();
  bindEvents();
  resizeCanvas();
  try {
    const response = await fetch("experiments.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    indexData = await response.json();
    readHashState();
    normalizeState();
    render();
  } catch (error) {
    setStatus(`Failed to load experiments.json: ${error.message}`);
  }
}

function cacheElements() {
  for (const id of [
    "run-select",
    "variant-select",
    "image-select",
    "mapping-select",
    "run-summary",
    "metric-count",
    "metrics-table",
    "image-label",
    "canvas-frame",
    "image-canvas",
    "canvas-status",
    "zoom-fit-button",
    "zoom-actual-button",
    "zoom-out-button",
    "zoom-in-button",
    "zoom-numbers-button",
    "zoom-label"
  ]) {
    els[id] = document.getElementById(id);
  }
  viewer.canvas = els["image-canvas"];
  viewer.ctx = viewer.canvas.getContext("2d");
  viewer.frame = els["canvas-frame"];
}

function bindEvents() {
  els["run-select"].addEventListener("change", () => {
    state.runId = els["run-select"].value;
    state.variant = "";
    state.mapping = "view_index";
    normalizeState();
    writeHashState();
    render();
  });
  els["variant-select"].addEventListener("change", () => {
    state.variant = els["variant-select"].value;
    normalizeState();
    writeHashState();
    render();
  });
  els["image-select"].addEventListener("change", () => {
    state.imageMode = els["image-select"].value;
    writeHashState();
    render();
  });
  els["mapping-select"].addEventListener("change", () => {
    state.mapping = els["mapping-select"].value;
    writeHashState();
    render();
  });
  els["zoom-fit-button"].addEventListener("click", fitImage);
  els["zoom-actual-button"].addEventListener("click", () => zoomTo(1));
  els["zoom-out-button"].addEventListener("click", () => zoomBy(1 / 1.5));
  els["zoom-in-button"].addEventListener("click", () => zoomBy(1.5));
  els["zoom-numbers-button"].addEventListener("click", zoomToNumbers);
  viewer.canvas.addEventListener("wheel", onWheel, { passive: false });
  viewer.canvas.addEventListener("mousedown", onMouseDown);
  window.addEventListener("mousemove", onMouseMove);
  window.addEventListener("mouseup", onMouseUp);
  window.addEventListener("resize", () => {
    resizeCanvas();
    drawCanvas();
  });
  window.addEventListener("hashchange", () => {
    readHashState();
    normalizeState();
    render();
  });
}

function readHashState() {
  const params = new URLSearchParams(window.location.hash.slice(1));
  state.runId = params.get("run") || state.runId;
  state.variant = params.get("variant") || state.variant;
  state.imageMode = params.get("image") || state.imageMode;
  state.mapping = params.get("mapping") || state.mapping;
}

function writeHashState() {
  const params = new URLSearchParams();
  if (state.runId) params.set("run", state.runId);
  if (state.variant) params.set("variant", state.variant);
  if (state.imageMode) params.set("image", state.imageMode);
  if (state.mapping) params.set("mapping", state.mapping);
  const next = params.toString();
  if (window.location.hash.slice(1) !== next) {
    window.history.replaceState(null, "", `${window.location.pathname}#${next}`);
  }
}

function normalizeState() {
  const runs = indexData?.runs || [];
  if (!runs.length) return;
  const run = runs.find((item) => item.id === state.runId) || runs[0];
  state.runId = run.id;

  const variants = run.variants || [];
  const variant = variants.find((item) => item.name === state.variant) || variants[0] || null;
  state.variant = variant?.name || "";

  const imageModes = availableImageModes(variant);
  if (!imageModes.includes(state.imageMode)) {
    state.imageMode = imageModes[0] || "looking_glass_tensor";
  }

  const mappings = availableMappings(run);
  if (!mappings.includes(state.mapping)) {
    state.mapping = mappings.includes("view_index") ? "view_index" : mappings[0] || "";
  }
}

function render() {
  const run = currentRun();
  const variant = currentVariant(run);
  renderRunSelect();
  renderVariantSelect(run);
  renderImageSelect(variant);
  renderMappingSelect(run);
  renderSummary(run);
  renderMetrics(run);
  updateImage(variant);
  updateMapping(run);
  els["image-label"].textContent = `${state.variant || "-"} | ${IMAGE_LABELS[state.imageMode] || state.imageMode}`;
  drawCanvas();
}

function currentRun() {
  return (indexData?.runs || []).find((item) => item.id === state.runId) || null;
}

function currentVariant(run) {
  return (run?.variants || []).find((item) => item.name === state.variant) || null;
}

function renderRunSelect() {
  const runs = indexData?.runs || [];
  els["run-select"].innerHTML = runs.map((run) => {
    const label = run.label && run.label !== run.id ? `${run.label} (${run.id})` : run.id;
    return optionHtml(run.id, label, run.id === state.runId);
  }).join("");
}

function renderVariantSelect(run) {
  const variants = run?.variants || [];
  els["variant-select"].innerHTML = variants.map((variant) => (
    optionHtml(variant.name, variant.name, variant.name === state.variant)
  )).join("");
}

function renderImageSelect(variant) {
  const modes = availableImageModes(variant);
  els["image-select"].innerHTML = modes.map((mode) => (
    optionHtml(mode, IMAGE_LABELS[mode] || mode, mode === state.imageMode)
  )).join("");
}

function renderMappingSelect(run) {
  const names = availableMappings(run);
  els["mapping-select"].innerHTML = names.map((name) => optionHtml(name, name, name === state.mapping)).join("");
}

function renderSummary(run) {
  if (!run) {
    els["run-summary"].textContent = "No experiment runs found.";
    return;
  }
  const size = run.width && run.height ? `${run.width}x${run.height}` : "unknown size";
  const views = run.render_view_count || run.source_view_count || "unknown";
  els["run-summary"].textContent = `${run.id} | ${size} | ${views} rendered views | ${run.camera || "camera unknown"}`;
}

function renderMetrics(run) {
  const rows = run?.metrics || [];
  els["metric-count"].textContent = `${rows.length} rows`;
  const columns = METRIC_COLUMNS.filter((column) => rows.some((row) => row[column] !== undefined && row[column] !== ""));
  const thead = els["metrics-table"].querySelector("thead");
  const tbody = els["metrics-table"].querySelector("tbody");
  thead.innerHTML = `<tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>`;
  tbody.innerHTML = rows.map((row) => {
    const selected = row.variant === state.variant ? " class=\\"is-selected\\"" : "";
    return `<tr${selected}>${columns.map((column) => `<td>${formatCell(row[column], column)}</td>`).join("")}</tr>`;
  }).join("");
}

function updateImage(variant) {
  const path = variant?.images?.[state.imageMode] || "";
  if (path === viewer.imagePath) return;
  viewer.imagePath = path;
  viewer.imageLoaded = false;
  viewer.image = null;
  if (!path) {
    drawCanvas();
    return;
  }
  const image = new Image();
  image.onload = () => {
    if (viewer.imagePath !== path) return;
    viewer.image = image;
    viewer.imageLoaded = true;
    fitImage();
  };
  image.onerror = () => {
    if (viewer.imagePath !== path) return;
    viewer.imageLoaded = false;
    setStatus(`Failed to load image: ${path}`);
    drawCanvas();
  };
  image.src = path;
}

function updateMapping(run) {
  const key = run ? `${run.id}::${state.mapping}` : "";
  if (viewer.mappingName === key) return;
  viewer.mappingName = key;
  viewer.mappingData = null;
  viewer.mappingError = "";
  if (!run || !state.mapping) {
    drawCanvas();
    return;
  }
  loadMappingData(run, state.mapping)
    .then((data) => {
      if (viewer.mappingName !== key) return;
      viewer.mappingData = data;
      viewer.mappingError = data ? "" : "No numeric mapping data";
      drawCanvas();
    })
    .catch((error) => {
      if (viewer.mappingName !== key) return;
      viewer.mappingError = error.message;
      viewer.mappingData = null;
      drawCanvas();
    });
}

async function loadMappingData(run, mappingName) {
  const definition = run?.mappings?.[mappingName];
  if (!definition) return null;

  let transform = null;
  let source = definition;
  if (!source.values && definition.transform) {
    transform = definition.transform;
    source = run.mappings?.[transform.source] || {};
  }
  if (!source.values) return null;

  const values = source.values;
  const cacheKey = `${run.id}:${values.path}`;
  let decoded = viewer.mappingCache.get(cacheKey);
  if (!decoded) {
    const response = await fetch(values.path, { cache: "force-cache" });
    if (!response.ok) {
      throw new Error(`Failed to load ${values.path}: HTTP ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    const array = typedArrayFor(values.dtype, buffer);
    const shape = values.shape || [run.height, run.width, values.channels || 3];
    decoded = {
      array,
      height: Number(shape[0]),
      width: Number(shape[1]),
      channels: Number(shape[2] || values.channels || 3)
    };
    viewer.mappingCache.set(cacheKey, decoded);
  }
  return { ...decoded, transform };
}

function typedArrayFor(dtype, buffer) {
  if (dtype === "uint16") return new Uint16Array(buffer);
  if (dtype === "int32") return new Int32Array(buffer);
  throw new Error(`Unsupported mapping dtype: ${dtype}`);
}

function resizeCanvas() {
  if (!viewer.canvas || !viewer.frame) return;
  const rect = viewer.frame.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (viewer.canvas.width !== width || viewer.canvas.height !== height) {
    viewer.canvas.width = width;
    viewer.canvas.height = height;
  }
}

function fitImage() {
  if (!viewer.imageLoaded) {
    drawCanvas();
    return;
  }
  resizeCanvas();
  const frameWidth = canvasCssWidth();
  const frameHeight = canvasCssHeight();
  const fitScale = Math.min(frameWidth / viewer.image.naturalWidth, frameHeight / viewer.image.naturalHeight);
  viewer.scale = clamp(fitScale, MIN_SCALE, MAX_SCALE);
  viewer.offsetX = (frameWidth - viewer.image.naturalWidth * viewer.scale) / 2;
  viewer.offsetY = (frameHeight - viewer.image.naturalHeight * viewer.scale) / 2;
  drawCanvas();
}

function zoomBy(factor) {
  zoomTo(viewer.scale * factor);
}

function zoomTo(scale, centerX = canvasCssWidth() / 2, centerY = canvasCssHeight() / 2) {
  if (!viewer.imageLoaded) return;
  const nextScale = clamp(scale, MIN_SCALE, MAX_SCALE);
  const imageX = (centerX - viewer.offsetX) / viewer.scale;
  const imageY = (centerY - viewer.offsetY) / viewer.scale;
  viewer.scale = nextScale;
  viewer.offsetX = centerX - imageX * nextScale;
  viewer.offsetY = centerY - imageY * nextScale;
  drawCanvas();
}

function zoomToNumbers() {
  if (!viewer.imageLoaded) return;
  zoomTo(Math.max(viewer.scale, NUMBER_SCALE));
}

function onWheel(event) {
  if (!viewer.imageLoaded) return;
  event.preventDefault();
  const rect = viewer.canvas.getBoundingClientRect();
  const factor = event.deltaY < 0 ? 1.18 : 1 / 1.18;
  zoomTo(viewer.scale * factor, event.clientX - rect.left, event.clientY - rect.top);
}

function onMouseDown(event) {
  if (!viewer.imageLoaded) return;
  viewer.dragging = true;
  viewer.dragX = event.clientX;
  viewer.dragY = event.clientY;
  viewer.frame.classList.add("is-dragging");
}

function onMouseMove(event) {
  if (!viewer.dragging) return;
  viewer.offsetX += event.clientX - viewer.dragX;
  viewer.offsetY += event.clientY - viewer.dragY;
  viewer.dragX = event.clientX;
  viewer.dragY = event.clientY;
  drawCanvas();
}

function onMouseUp() {
  viewer.dragging = false;
  viewer.frame.classList.remove("is-dragging");
}

function drawCanvas() {
  resizeCanvas();
  const ctx = viewer.ctx;
  const dpr = window.devicePixelRatio || 1;
  const width = canvasCssWidth();
  const height = canvasCssHeight();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101418";
  ctx.fillRect(0, 0, width, height);

  if (!viewer.imagePath) {
    setStatus("Missing image artifact");
    updateZoomLabel();
    return;
  }
  if (!viewer.imageLoaded) {
    setStatus(`Loading ${viewer.imagePath}`);
    updateZoomLabel();
    return;
  }

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(
    viewer.image,
    viewer.offsetX,
    viewer.offsetY,
    viewer.image.naturalWidth * viewer.scale,
    viewer.image.naturalHeight * viewer.scale
  );
  drawPixelGrid(ctx);
  drawViewNumbers(ctx);
  updateZoomLabel();
}

function drawPixelGrid(ctx) {
  if (viewer.scale < 10) return;
  const bounds = visiblePixelBounds();
  ctx.save();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.16)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = bounds.x0; x <= bounds.x1 + 1; x += 1) {
    const canvasX = viewer.offsetX + x * viewer.scale;
    ctx.moveTo(canvasX, viewer.offsetY + bounds.y0 * viewer.scale);
    ctx.lineTo(canvasX, viewer.offsetY + (bounds.y1 + 1) * viewer.scale);
  }
  for (let y = bounds.y0; y <= bounds.y1 + 1; y += 1) {
    const canvasY = viewer.offsetY + y * viewer.scale;
    ctx.moveTo(viewer.offsetX + bounds.x0 * viewer.scale, canvasY);
    ctx.lineTo(viewer.offsetX + (bounds.x1 + 1) * viewer.scale, canvasY);
  }
  ctx.stroke();
  ctx.restore();
}

function drawViewNumbers(ctx) {
  if (viewer.scale < NUMBER_SCALE) {
    setStatus(`Zoom ${formatZoom()} | numbers render at ${NUMBER_SCALE}x`);
    return;
  }
  if (viewer.mappingError) {
    setStatus(`Zoom ${formatZoom()} | ${viewer.mappingError}`);
    return;
  }
  if (!viewer.mappingData) {
    setStatus(`Zoom ${formatZoom()} | loading numeric overlay`);
    return;
  }
  if (viewer.mappingData.width !== viewer.image.naturalWidth || viewer.mappingData.height !== viewer.image.naturalHeight) {
    setStatus(`Zoom ${formatZoom()} | mapping size ${viewer.mappingData.width}x${viewer.mappingData.height} does not match image`);
    return;
  }

  const bounds = visiblePixelBounds();
  const pixelCount = (bounds.x1 - bounds.x0 + 1) * (bounds.y1 - bounds.y0 + 1);
  if (pixelCount > MAX_NUMBERED_PIXELS) {
    setStatus(`Zoom ${formatZoom()} | zoom further to label ${pixelCount.toLocaleString()} visible pixels`);
    return;
  }

  const fontSize = clamp(viewer.scale * 0.24, 6, 13);
  const lineHeight = fontSize * 1.05;
  const colors = ["#ffb4b4", "#bff5c7", "#b8c8ff"];
  ctx.save();
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = `${fontSize}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
  for (let y = bounds.y0; y <= bounds.y1; y += 1) {
    for (let x = bounds.x0; x <= bounds.x1; x += 1) {
      const centerX = viewer.offsetX + (x + 0.5) * viewer.scale;
      const centerY = viewer.offsetY + (y + 0.5) * viewer.scale;
      ctx.fillStyle = "rgba(0, 0, 0, 0.44)";
      ctx.fillRect(
        viewer.offsetX + x * viewer.scale + 1,
        viewer.offsetY + y * viewer.scale + 1,
        Math.max(0, viewer.scale - 2),
        Math.max(0, viewer.scale - 2)
      );
      for (let channel = 0; channel < 3; channel += 1) {
        ctx.fillStyle = colors[channel];
        ctx.fillText(
          String(mappingValueAt(x, y, channel)),
          centerX,
          centerY + (channel - 1) * lineHeight
        );
      }
    }
  }
  ctx.restore();
  setStatus(`Zoom ${formatZoom()} | ${state.mapping} | ${pixelCount.toLocaleString()} labeled pixels`);
}

function mappingValueAt(x, y, channel) {
  const data = viewer.mappingData;
  const base = (y * data.width + x) * data.channels;
  let value = Number(data.array[base + channel]);
  if (data.transform?.type === "cluster_from_view") {
    value = Math.floor(value / Number(data.transform.cluster_size || 1));
  }
  return value;
}

function visiblePixelBounds() {
  const width = canvasCssWidth();
  const height = canvasCssHeight();
  const imageWidth = viewer.image?.naturalWidth || 1;
  const imageHeight = viewer.image?.naturalHeight || 1;
  const x0 = clamp(Math.floor((-viewer.offsetX) / viewer.scale), 0, imageWidth - 1);
  const y0 = clamp(Math.floor((-viewer.offsetY) / viewer.scale), 0, imageHeight - 1);
  const x1 = clamp(Math.ceil((width - viewer.offsetX) / viewer.scale), 0, imageWidth - 1);
  const y1 = clamp(Math.ceil((height - viewer.offsetY) / viewer.scale), 0, imageHeight - 1);
  return { x0, y0, x1, y1 };
}

function availableImageModes(variant) {
  const images = variant?.images || {};
  return Object.keys(IMAGE_LABELS).filter((key) => images[key]);
}

function availableMappings(run) {
  return Object.keys(run?.mappings || {})
    .filter((name) => mappingHasNumericData(run, name))
    .sort(mappingSort);
}

function mappingHasNumericData(run, name) {
  const mapping = run?.mappings?.[name];
  if (!mapping) return false;
  if (mapping.values) return true;
  if (mapping.transform?.source) {
    return Boolean(run.mappings?.[mapping.transform.source]?.values);
  }
  return false;
}

function mappingSort(a, b) {
  const rank = (name) => {
    if (name === "view_index") return 0;
    if (name === "source_view_index") return 1;
    if (name === `${state.variant}_cluster_index`) return 2;
    if (name.endsWith("_cluster_index")) return 3;
    return 4;
  };
  return rank(a) - rank(b) || a.localeCompare(b);
}

function canvasCssWidth() {
  return viewer.canvas.width / (window.devicePixelRatio || 1);
}

function canvasCssHeight() {
  return viewer.canvas.height / (window.devicePixelRatio || 1);
}

function setStatus(text) {
  els["canvas-status"].textContent = text || "";
}

function updateZoomLabel() {
  els["zoom-label"].textContent = viewer.imageLoaded ? `Zoom: ${formatZoom()}` : "Zoom: -";
}

function formatZoom() {
  return `${viewer.scale.toFixed(viewer.scale >= 10 ? 1 : 2)}x`;
}

function optionHtml(value, label, selected) {
  return `<option value="${escapeHtml(value)}"${selected ? " selected" : ""}>${escapeHtml(label)}</option>`;
}

function formatCell(value, column) {
  if (value === undefined || value === null || value === "") return "-";
  const formatter = METRIC_FORMATTERS[column];
  if (formatter) return formatter(value);
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "-";
    if (Math.abs(value) >= 100) return value.toFixed(2);
    if (Math.abs(value) >= 1) return value.toFixed(4);
    return value.toPrecision(4);
  }
  return escapeHtml(String(value));
}

function formatBool(value) {
  if (typeof value === "boolean") return value ? "True" : "False";
  const normalized = String(value).trim().toLowerCase();
  if (normalized === "true" || normalized === "1") return "True";
  if (normalized === "false" || normalized === "0") return "False";
  return "-";
}

function formatInteger(value) {
  const number = numericValue(value);
  if (number === null) return "-";
  return String(Math.round(number));
}

function formatFixed(value, digits) {
  const number = numericValue(value);
  if (number === null) return "-";
  return number.toFixed(digits);
}

function numericValue(value) {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === "boolean") {
    return value ? 1 : 0;
  }
  const number = Number(String(value).trim());
  return Number.isFinite(number) ? number : null;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
"""


def _make_ssim_metric(device: Any):
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        return StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    except Exception:
        return None


def _make_lpips_metric(device: Any, *, required: bool):
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        return LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    except Exception:
        if required:
            raise
        return None


def _psnr_torch(a: Any, b: Any) -> float:
    import torch

    mse = torch.mean((a.float() - b.float()) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _mean(values: Sequence[float]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _std(values: Sequence[float]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite))


def _colorize_index_plane(index: np.ndarray) -> np.ndarray:
    values = np.asarray(index, dtype=np.int64)
    safe = np.maximum(values, 0)
    rgb = np.empty(values.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = (safe * 37 + 17) % 256
    rgb[..., 1] = (safe * 67 + 29) % 256
    rgb[..., 2] = (safe * 97 + 53) % 256
    rgb[values < 0] = 0
    return rgb


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _csv_ready(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, np.generic):
        return value.item()
    return value
