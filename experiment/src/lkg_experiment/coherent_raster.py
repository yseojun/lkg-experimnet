from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


TILE_WIDTH = 16
TILE_HEIGHT = 16
RGB_SUBPIXELS = 3
SUBPIXELS_PER_TILE = TILE_WIDTH * TILE_HEIGHT * RGB_SUBPIXELS


QuantizationMode = str


@dataclass(frozen=True)
class LKGViewMappingCalibration:
    pitch: float
    slope: float
    center: float
    subp: float
    inv_view: bool = False
    ri: int = 0
    bi: int = 2


def raw_lkg_slope_to_shader_slope(raw_slope: float, display_aspect: float) -> float:
    raw_slope = float(raw_slope)
    display_aspect = float(display_aspect)
    if abs(raw_slope) < 1e-12:
        raise ValueError("raw_slope must be non-zero")
    if display_aspect <= 0.0:
        raise ValueError("display_aspect must be positive")
    return 1.0 / (raw_slope * display_aspect)


def _quantize_view(view_float: np.ndarray, view_count: int, mode: QuantizationMode) -> np.ndarray:
    if mode == "nearest":
        view = np.rint(view_float)
    elif mode == "floor":
        view = np.floor(view_float)
    else:
        raise ValueError("mode must be 'floor' or 'nearest'")
    return np.mod(view.astype(np.int32), int(view_count))


def build_linear_viewpoint_index(
    width: int,
    height: int,
    view_count: int,
    *,
    quantize: QuantizationMode = "floor",
) -> np.ndarray:
    """Build a deterministic fallback subpixel-to-view map.

    This is not a device calibration model. It is useful for smoke tests and
    no-Bridge runs because it exercises the coherent rasterizer's per-subpixel
    view selection without requiring a connected Looking Glass display.
    """
    _validate_dimensions(width, height, view_count)
    x = np.arange(width, dtype=np.float64)[None, :, None]
    channel = np.arange(RGB_SUBPIXELS, dtype=np.float64)[None, None, :]
    phase = (x * RGB_SUBPIXELS + channel) / float(width * RGB_SUBPIXELS)
    row_phase = np.broadcast_to(phase, (height, width, RGB_SUBPIXELS))
    return _quantize_view(row_phase * float(view_count), view_count, quantize).astype(np.int32)


def build_lkg_viewpoint_index(
    width: int,
    height: int,
    view_count: int,
    calibration: LKGViewMappingCalibration,
    *,
    quantize: QuantizationMode = "floor",
) -> np.ndarray:
    """Approximate the Looking Glass quilt shader's subpixel view mapping."""
    _validate_dimensions(width, height, view_count)
    x = (np.arange(width, dtype=np.float64) + 0.5) / float(width)
    y = (np.arange(height, dtype=np.float64) + 0.5) / float(height)
    channel = _physical_channel_indices(calibration.ri, calibration.bi).astype(np.float64)
    channel = channel * float(calibration.subp)

    phase = (
        x[None, :, None]
        + y[:, None, None] * float(calibration.slope)
        + channel[None, None, :]
    )
    phase = np.mod(phase * float(calibration.pitch) - float(calibration.center), 1.0)
    if calibration.inv_view:
        phase = 1.0 - phase
    return _quantize_view(phase * float(view_count), view_count, quantize).astype(np.int32)


def build_view_remap_table(
    viewpoint_index: np.ndarray,
    *,
    tile_width: int = TILE_WIDTH,
    tile_height: int = TILE_HEIGHT,
) -> np.ndarray:
    """Sort each tile's local subpixels by view index for coherent thread order."""
    if viewpoint_index.ndim != 3 or viewpoint_index.shape[2] != RGB_SUBPIXELS:
        raise ValueError("viewpoint_index must have shape (height, width, 3)")
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile_width and tile_height must be positive")

    height, width, _ = viewpoint_index.shape
    tiles_x = (width + tile_width - 1) // tile_width
    tiles_y = (height + tile_height - 1) // tile_height
    subpixels_per_tile = tile_width * tile_height * RGB_SUBPIXELS
    remap = np.empty((tiles_x * tiles_y, subpixels_per_tile), dtype=np.int32)

    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            tile_id = tile_y * tiles_x + tile_x
            entries: list[tuple[int, int]] = []
            for local_y in range(tile_height):
                py = tile_y * tile_height + local_y
                for local_x in range(tile_width):
                    px = tile_x * tile_width + local_x
                    local_pixel = local_y * tile_width + local_x
                    for channel in range(RGB_SUBPIXELS):
                        local_subpixel = local_pixel * RGB_SUBPIXELS + channel
                        if px < width and py < height:
                            view_id = int(viewpoint_index[py, px, channel])
                        else:
                            view_id = np.iinfo(np.int32).max
                        entries.append((view_id, local_subpixel))
            entries.sort(key=lambda item: (item[0], item[1]))
            remap[tile_id, :] = [local_subpixel for _, local_subpixel in entries]

    return remap


def bridge_calibration_to_lkg(
    bridge,
    window_handle,
    *,
    fallback: Optional[LKGViewMappingCalibration] = None,
) -> Optional[LKGViewMappingCalibration]:
    try:
        try:
            raw = bridge.get_calibration(window_handle)
            display_aspect = float(bridge.get_display_aspect(window_handle))
            slope = raw_lkg_slope_to_shader_slope(float(raw.Slope), display_aspect)
        except Exception:
            slope = float(bridge.get_tilt(window_handle))
        return LKGViewMappingCalibration(
            pitch=float(bridge.get_pitch(window_handle)),
            slope=slope,
            center=float(bridge.get_center(window_handle)),
            subp=float(bridge.get_subp(window_handle)),
            inv_view=bool(bridge.get_invview(window_handle)),
            ri=int(bridge.get_ri(window_handle)),
            bi=int(bridge.get_bi(window_handle)),
        )
    except Exception:
        return fallback


def bridge_display_calibration_to_lkg(
    bridge,
    display_handle,
    *,
    fallback: Optional[LKGViewMappingCalibration] = None,
) -> Optional[LKGViewMappingCalibration]:
    try:
        try:
            raw = bridge.get_calibration_for_display(display_handle)
            display_aspect = float(bridge.get_display_aspect_for_display(display_handle))
            slope = raw_lkg_slope_to_shader_slope(float(raw.Slope), display_aspect)
        except Exception:
            slope = float(bridge.get_tilt_for_display(display_handle))
        return LKGViewMappingCalibration(
            pitch=float(bridge.get_pitch_for_display(display_handle)),
            slope=slope,
            center=float(bridge.get_center_for_display(display_handle)),
            subp=float(bridge.get_subp_for_display(display_handle)),
            inv_view=bool(bridge.get_invview_for_display(display_handle)),
            ri=int(bridge.get_ri_for_display(display_handle)),
            bi=int(bridge.get_bi_for_display(display_handle)),
        )
    except Exception:
        return fallback


def _physical_channel_indices(ri: int, bi: int) -> np.ndarray:
    green = 3 - int(ri) - int(bi)
    channels = np.asarray([int(ri), green, int(bi)], dtype=np.int32)
    if np.any(channels < 0) or np.any(channels >= RGB_SUBPIXELS) or len(set(channels.tolist())) != RGB_SUBPIXELS:
        return np.arange(RGB_SUBPIXELS, dtype=np.int32)
    return channels


def _validate_dimensions(width: int, height: int, view_count: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if view_count <= 0:
        raise ValueError("view_count must be positive")
