from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


RGB_SUBPIXELS = 3
DEFAULT_TILE_SIZE = 16


@dataclass(frozen=True)
class CRLookupArrays:
    view_index: np.ndarray
    subpixel_coords: np.ndarray
    original_height: int
    original_width: int
    padded_height: int
    padded_width: int
    tile_size: int


def resolve_checkpoint_path(path: Path | str, iteration: Optional[int] = None, rank: int = 0) -> Path:
    """Resolve a CoherentRaster/gsplat checkpoint file.

    `path` can be a checkpoint file, a result directory containing `ckpts/`, or
    the `ckpts/` directory itself.
    """
    root = Path(path).expanduser()
    if root.is_file():
        return root

    ckpt_dir = root / "ckpts" if (root / "ckpts").is_dir() else root
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoint path is not a file or directory: {root}")

    if iteration is not None:
        ckpt = ckpt_dir / f"ckpt_{int(iteration)}_rank{int(rank)}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint for iteration {iteration} rank {rank} not found: {ckpt}")
        return ckpt

    pattern = re.compile(rf"^ckpt_(\d+)_rank{int(rank)}\.pt$")
    candidates = []
    for candidate in ckpt_dir.iterdir():
        match = pattern.match(candidate.name)
        if match:
            candidates.append((int(match.group(1)), candidate))
    if not candidates:
        raise FileNotFoundError(f"no ckpt_*_rank{int(rank)}.pt files found in {ckpt_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def scale_intrinsics_to_panel(
    K_original: np.ndarray,
    *,
    original_height: int,
    original_width: int,
    target_height: int,
    target_width: int,
    crop_to_fill: bool = True,
) -> np.ndarray:
    """Match CoherentRaster's `scale_K` behavior for a target panel size."""
    K = np.asarray(K_original)
    if K.shape != (3, 3):
        raise ValueError("K_original must have shape (3, 3)")
    if min(original_height, original_width, target_height, target_width) <= 0:
        raise ValueError("image dimensions must be positive")

    scale_w = float(target_width) / float(original_width)
    scale_h = float(target_height) / float(original_height)
    scale_factor = max(scale_w, scale_h) if crop_to_fill else min(scale_w, scale_h)

    scaled = np.zeros((3, 3), dtype=K.dtype)
    scaled[0, 0] = K[0, 0] * scale_factor
    scaled[1, 1] = K[1, 1] * scale_factor
    scaled[0, 2] = float(target_width) / 2.0
    scaled[1, 2] = float(target_height) / 2.0
    scaled[2, 2] = 1.0
    return scaled


def build_cr_lookup_arrays(
    viewpoint_index: np.ndarray,
    *,
    tile_size: int = DEFAULT_TILE_SIZE,
    use_remapping: bool = True,
) -> CRLookupArrays:
    """Convert an HxWx3 LKG view map to CoherentRaster's patchified layout."""
    view_hwc = np.asarray(viewpoint_index)
    if view_hwc.ndim != 3 or view_hwc.shape[2] != RGB_SUBPIXELS:
        raise ValueError("viewpoint_index must have shape (height, width, 3)")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if np.any(view_hwc < 0):
        raise ValueError("viewpoint_index must not contain negative view ids")

    height, width, _ = view_hwc.shape
    padded_height = _round_up(height, tile_size)
    padded_width = _round_up(width, tile_size)

    view_chw = np.zeros((RGB_SUBPIXELS, padded_height, padded_width), dtype=np.uint32)
    view_chw[:, :height, :width] = np.transpose(view_hwc.astype(np.uint32, copy=False), (2, 0, 1))
    subpixel_chw = _subpixel_coord_matrix(padded_height, padded_width)

    view_patch = _patchify_image_shape_matrix(view_chw, tile_size)
    coord_patch = _patchify_coordinate_matrix(subpixel_chw, tile_size)

    if use_remapping:
        view_patch, coord_patch = _remap_subpixel_coord(view_patch, coord_patch)

    return CRLookupArrays(
        view_index=view_patch.astype(np.uint32, copy=False),
        subpixel_coords=coord_patch.astype(np.uint32, copy=False),
        original_height=height,
        original_width=width,
        padded_height=padded_height,
        padded_width=padded_width,
        tile_size=tile_size,
    )


def install_gsplat_root(gsplat_root: Path | str) -> Path:
    root = Path(gsplat_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"gsplat root not found: {root}")
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def lookup_arrays_to_torch(lookup: CRLookupArrays, device: str = "cuda"):
    import torch

    view_index = torch.from_numpy(lookup.view_index).to(device=device, non_blocking=True).contiguous()
    subpixel_coords = torch.from_numpy(lookup.subpixel_coords).to(device=device, non_blocking=True).contiguous()
    return view_index, subpixel_coords


def load_splats_from_checkpoint(checkpoint_path: Path | str, device: str = "cuda") -> tuple[dict, int]:
    import torch

    ckpt = torch.load(str(Path(checkpoint_path).expanduser()), map_location="cpu", weights_only=False)
    if "splats" not in ckpt:
        raise KeyError("checkpoint does not contain a 'splats' dict")
    splats = ckpt["splats"]
    required = ("means", "opacities", "quats", "scales", "sh0", "shN")
    missing = [key for key in required if key not in splats]
    if missing:
        raise KeyError(f"checkpoint splats missing required keys: {missing}")
    splats = {key: value.to(device=device, non_blocking=True).contiguous() for key, value in splats.items()}
    return splats, int(ckpt.get("step", -1))


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _subpixel_coord_matrix(height: int, width: int) -> np.ndarray:
    channel, y, x = np.meshgrid(
        np.arange(RGB_SUBPIXELS, dtype=np.uint32),
        np.arange(height, dtype=np.uint32),
        np.arange(width, dtype=np.uint32),
        indexing="ij",
    )
    return np.stack([channel, y, x], axis=-1)


def _patchify_image_shape_matrix(matrix: np.ndarray, tile_size: int) -> np.ndarray:
    channels, height, width = matrix.shape
    if channels != RGB_SUBPIXELS:
        raise ValueError("matrix must have three subpixel channels")
    n_tile_height = height // tile_size
    n_tile_width = width // tile_size
    patched = matrix.reshape(RGB_SUBPIXELS, n_tile_height, tile_size, n_tile_width, tile_size)
    return patched.transpose(1, 3, 0, 2, 4).copy()


def _patchify_coordinate_matrix(matrix: np.ndarray, tile_size: int) -> np.ndarray:
    channels, height, width, coord_dim = matrix.shape
    if channels != RGB_SUBPIXELS or coord_dim != 3:
        raise ValueError("coordinate matrix must have shape (3, height, width, 3)")
    n_tile_height = height // tile_size
    n_tile_width = width // tile_size
    coord_first = matrix.transpose(0, 3, 1, 2)
    patched = coord_first.reshape(RGB_SUBPIXELS, 3, n_tile_height, tile_size, n_tile_width, tile_size)
    return patched.transpose(2, 4, 0, 3, 5, 1).copy()


def _remap_subpixel_coord(view_patch: np.ndarray, coord_patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_tile_h, n_tile_w, channels, tile_h, tile_w = view_patch.shape
    flat_view = view_patch.reshape(n_tile_h, n_tile_w, channels * tile_h * tile_w)
    flat_coord = coord_patch.reshape(n_tile_h, n_tile_w, channels * tile_h * tile_w, 3)
    sort_indices = np.argsort(flat_view, axis=2, kind="stable")
    remapped_view = np.take_along_axis(flat_view, sort_indices, axis=2)
    remapped_coord = np.take_along_axis(flat_coord, sort_indices[..., None], axis=2)
    return (
        remapped_view.reshape(n_tile_h, n_tile_w, channels, tile_h, tile_w),
        remapped_coord.reshape(n_tile_h, n_tile_w, channels, tile_h, tile_w, 3),
    )
