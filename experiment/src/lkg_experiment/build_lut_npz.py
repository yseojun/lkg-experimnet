#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


RGB_CHANNELS = 3

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
CLEAN_ROOT = REPO_ROOT.parent
DEFAULT_BRIDGE_SDK_ROOT = CLEAN_ROOT / "Bridge-Python-SDK-Lab"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "generated"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an empirical Looking Glass subpixel view-index LUT as an .npz file"
    )
    parser.add_argument("--bridge-sdk-root", default=str(DEFAULT_BRIDGE_SDK_ROOT))
    parser.add_argument("--display-index", default=0, type=int)
    parser.add_argument("--quilt-cols", default=0, type=int, help="0 uses Bridge default quilt columns")
    parser.add_argument("--quilt-rows", default=0, type=int, help="0 uses Bridge default quilt rows")
    parser.add_argument("--layout", choices=("gl-top", "gl-bottom"), default="gl-top")
    parser.add_argument("--index-method", choices=("balanced-ramp",), default="balanced-ramp")
    parser.add_argument(
        "--output",
        help=(
            "Output .npz path. Default is generated/lkg_go_<native_width>x<native_height>_"
            "<view_count>_views_balanced.npz"
        ),
    )
    parser.add_argument(
        "--no-save-responses",
        action="store_true",
        help="Do not store white_response and ramp_response debug arrays in the .npz",
    )
    return parser


def install_bridge_sdk_root(path: Path | str) -> Path:
    root = Path(path).expanduser().resolve()
    src_root = root / "src"
    if not src_root.is_dir():
        raise FileNotFoundError(f"Bridge SDK src directory not found: {src_root}")
    package_root = src_root / "bridge_python_sdk"
    if not package_root.is_dir():
        raise FileNotFoundError(f"Bridge SDK package directory not found: {package_root}")
    for import_root in (src_root, package_root):
        import_root_str = str(import_root)
        if import_root_str not in sys.path:
            sys.path.insert(0, import_root_str)
    return root


def as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if hasattr(value, "value"):
        return int(value.value)
    return int(value)


def ramp_colors(view_count: int) -> np.ndarray:
    if view_count <= 0:
        raise ValueError("view_count must be positive")
    if view_count == 1:
        values = np.zeros((1,), dtype=np.uint8)
    else:
        values = np.rint(np.linspace(0.0, 255.0, int(view_count))).astype(np.uint8)
    return np.repeat(values[:, None], RGB_CHANNELS, axis=1)


def fill_quilt(
    colors: np.ndarray,
    quilt_width: int,
    quilt_height: int,
    cols: int,
    rows: int,
    layout: str,
) -> np.ndarray:
    if quilt_width <= 0 or quilt_height <= 0:
        raise ValueError("quilt_width and quilt_height must be positive")
    if cols <= 0 or rows <= 0:
        raise ValueError("cols and rows must be positive")
    if layout not in {"gl-top", "gl-bottom"}:
        raise ValueError("layout must be gl-top or gl-bottom")

    colors = np.asarray(colors, dtype=np.uint8)
    if colors.ndim != 2 or colors.shape[1] != RGB_CHANNELS:
        raise ValueError("colors must have shape (view_count, 3)")

    quilt = np.zeros((int(quilt_height), int(quilt_width), 4), dtype=np.uint8)
    quilt[:, :, 3] = 255
    view_count = min(colors.shape[0], int(cols) * int(rows))
    for view in range(view_count):
        col = view % int(cols)
        row = view // int(cols)
        x0 = int(round(col * int(quilt_width) / float(cols)))
        x1 = int(round((col + 1) * int(quilt_width) / float(cols)))
        if layout == "gl-bottom":
            y0 = int(round(row * int(quilt_height) / float(rows)))
            y1 = int(round((row + 1) * int(quilt_height) / float(rows)))
        else:
            y0 = int(round((int(rows) - 1 - row) * int(quilt_height) / float(rows)))
            y1 = int(round((int(rows) - row) * int(quilt_height) / float(rows)))
        quilt[y0:y1, x0:x1, :RGB_CHANNELS] = colors[view]
    return quilt


def fill_invalid_view_float(view_float: np.ndarray, valid: np.ndarray) -> np.ndarray:
    filled = np.asarray(view_float, dtype=np.float32).copy()
    valid_mask = np.asarray(valid, dtype=bool) & np.isfinite(filled)
    if filled.ndim != 3:
        raise ValueError("view_float must have shape (height, width, channels)")
    if valid_mask.shape != filled.shape:
        raise ValueError("valid must have the same shape as view_float")

    height, width, channels = filled.shape
    xs_all = np.arange(width, dtype=np.float32)
    ys_all = np.arange(height, dtype=np.float32)

    for channel_index in range(channels):
        channel = filled[:, :, channel_index]
        mask = valid_mask[:, :, channel_index]
        for y in range(height):
            xs = np.flatnonzero(mask[y, :])
            if xs.size:
                channel[y, :] = np.interp(xs_all, xs.astype(np.float32), channel[y, xs]).astype(np.float32)
        finite = np.isfinite(channel)
        for x in range(width):
            ys = np.flatnonzero(finite[:, x])
            if ys.size and ys.size != height:
                channel[:, x] = np.interp(ys_all, ys.astype(np.float32), channel[ys, x]).astype(np.float32)
        finite = np.isfinite(channel)
        if not np.all(finite):
            channel[~finite] = 0.0
        filled[:, :, channel_index] = channel

    return filled


def balanced_view_index_from_float(view_float: np.ndarray, view_count: int) -> np.ndarray:
    if view_count <= 0:
        raise ValueError("view_count must be positive")
    view_float = np.asarray(view_float, dtype=np.float32)
    if view_float.ndim != 3:
        raise ValueError("view_float must have shape (height, width, channels)")

    flat = view_float.reshape(-1)
    order = np.argsort(flat, kind="stable")
    count = flat.size
    bins = (np.arange(count, dtype=np.uint64) * np.uint64(view_count)) // np.uint64(count)
    viewpoint_flat = np.empty(count, dtype=np.int32)
    viewpoint_flat[order] = bins.astype(np.int32, copy=False)
    return viewpoint_flat.reshape(view_float.shape)


def recover_viewpoint_index_from_balanced_ramp(
    bridge: Any,
    bridge_window: Any,
    pixel_format_rgba: Any,
    quilt_width: int,
    quilt_height: int,
    quilt_cols: int,
    quilt_rows: int,
    aspect: float,
    layout: str,
    view_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    white_colors = np.full((int(view_count), RGB_CHANNELS), 255, dtype=np.uint8)
    white_quilt = fill_quilt(white_colors, quilt_width, quilt_height, quilt_cols, quilt_rows, layout)
    white_image = render_bridge_interlaced(
        bridge,
        bridge_window,
        pixel_format_rgba,
        white_quilt,
        quilt_width,
        quilt_height,
        quilt_cols,
        quilt_rows,
        aspect,
    )
    ramp_quilt = fill_quilt(ramp_colors(view_count), quilt_width, quilt_height, quilt_cols, quilt_rows, layout)
    ramp_image = render_bridge_interlaced(
        bridge,
        bridge_window,
        pixel_format_rgba,
        ramp_quilt,
        quilt_width,
        quilt_height,
        quilt_cols,
        quilt_rows,
        aspect,
    )

    white_response = white_image[:, :, :RGB_CHANNELS].copy()
    ramp_response = ramp_image[:, :, :RGB_CHANNELS].copy()
    white_float = white_response.astype(np.float32)
    ramp_float = ramp_response.astype(np.float32)
    valid = white_float > 0.0
    view_float = np.full(white_float.shape, np.nan, dtype=np.float32)
    view_float[valid] = ramp_float[valid] / white_float[valid] * float(view_count - 1)
    np.clip(view_float, 0.0, float(view_count - 1), out=view_float)
    view_float = fill_invalid_view_float(view_float, valid)
    viewpoint_index = balanced_view_index_from_float(view_float, view_count)
    return viewpoint_index, white_response, ramp_response, view_float, valid


def save_viewpoint_index_npz(
    path: Path | str,
    viewpoint_index: np.ndarray,
    *,
    view_count: int,
    quilt_cols: int,
    quilt_rows: int,
    layout: str,
    index_method: str,
    extra_metadata: Optional[dict[str, np.ndarray]] = None,
) -> Path:
    index = np.asarray(viewpoint_index, dtype=np.int32)
    if index.ndim != 3 or index.shape[2] != RGB_CHANNELS:
        raise ValueError("viewpoint_index must have shape (height, width, 3)")
    if view_count <= 0:
        raise ValueError("view_count must be positive")
    if quilt_cols <= 0 or quilt_rows <= 0:
        raise ValueError("quilt_cols and quilt_rows must be positive")

    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "viewpoint_index": index,
        "view_count": np.asarray(int(view_count), dtype=np.int32),
        "quilt_cols": np.asarray(int(quilt_cols), dtype=np.int32),
        "quilt_rows": np.asarray(int(quilt_rows), dtype=np.int32),
        "layout": np.asarray(str(layout)),
        "width": np.asarray(index.shape[1], dtype=np.int32),
        "height": np.asarray(index.shape[0], dtype=np.int32),
        "index_method": np.asarray(str(index_method)),
    }
    if extra_metadata:
        payload.update(extra_metadata)
    np.savez_compressed(str(output), **payload)
    return output


def create_hidden_context():
    import glfw

    if not glfw.init():
        raise RuntimeError("failed to initialize GLFW")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(64, 64, "BuildLutNpz", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("failed to create hidden GLFW context")
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    return window


def init_texture(width: int, height: int, pixels: np.ndarray) -> int:
    from OpenGL import GL

    texture = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGBA8,
        int(width),
        int(height),
        0,
        GL.GL_RGBA,
        GL.GL_UNSIGNED_BYTE,
        pixels,
    )
    return int(texture)


def read_texture(texture: int, width: int, height: int) -> np.ndarray:
    from OpenGL import GL

    pixels = np.empty((int(height), int(width), 4), dtype=np.uint8)
    GL.glBindTexture(GL.GL_TEXTURE_2D, int(texture))
    GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, pixels)
    return pixels


def render_bridge_interlaced(
    bridge: Any,
    bridge_window: Any,
    pixel_format_rgba: Any,
    quilt: np.ndarray,
    quilt_width: int,
    quilt_height: int,
    cols: int,
    rows: int,
    aspect: float,
) -> np.ndarray:
    from OpenGL import GL

    texture = init_texture(quilt_width, quilt_height, quilt)
    try:
        bridge.draw_interop_quilt_texture_gl(
            bridge_window,
            texture,
            pixel_format_rgba,
            int(quilt_width),
            int(quilt_height),
            int(cols),
            int(rows),
            float(aspect),
            1.0,
        )
        GL.glFinish()
        out_texture, _fmt, out_width, out_height = bridge.get_offscreen_window_texture_gl(bridge_window)
        return read_texture(out_texture, out_width, out_height)
    finally:
        GL.glDeleteTextures(1, [texture])


def get_display(bridge: Any, display_index: int) -> Any:
    displays = bridge.get_displays()
    if not displays:
        raise RuntimeError("Bridge did not report any Looking Glass displays")
    if display_index < 0 or display_index >= len(displays):
        raise IndexError(f"--display-index {display_index} outside display count {len(displays)}")
    return displays[display_index]


def quilt_settings_for_display(bridge: Any, display_handle: Any) -> tuple[float, int, int, int, int]:
    quilt = bridge.get_default_quilt_settings_for_display(display_handle)
    aspect = float(getattr(quilt, "Aspect"))
    width = as_int(getattr(quilt, "QuiltWidth"))
    height = as_int(getattr(quilt, "QuiltHeight"))
    cols = as_int(getattr(quilt, "QuiltColumns"))
    rows = as_int(getattr(quilt, "QuiltRows"))
    if width <= 0 or height <= 0 or cols <= 0 or rows <= 0:
        raise RuntimeError(f"invalid Bridge quilt settings: {width}x{height}, {cols}x{rows}")
    return aspect, width, height, cols, rows


def default_output_path(native_width: int, native_height: int, view_count: int) -> Path:
    return DEFAULT_OUTPUT_DIR / f"lkg_go_{int(native_width)}x{int(native_height)}_{int(view_count)}_views_balanced.npz"


def _load_bridge_api():
    from BridgeApi import BridgeAPI, PixelFormats

    return BridgeAPI, PixelFormats


def main() -> None:
    args = build_parser().parse_args()
    if (args.quilt_cols == 0) != (args.quilt_rows == 0):
        raise ValueError("--quilt-cols and --quilt-rows must be provided together")
    if args.quilt_cols < 0 or args.quilt_rows < 0:
        raise ValueError("--quilt-cols and --quilt-rows must be non-negative")

    install_bridge_sdk_root(args.bridge_sdk_root)
    BridgeAPI, PixelFormats = _load_bridge_api()

    context = None
    bridge = BridgeAPI()
    try:
        context = create_hidden_context()
        if not bridge.initialize("BuildLutNpz"):
            raise RuntimeError("Bridge initialize failed")

        display_handle = get_display(bridge, int(args.display_index))
        native_width, native_height = bridge.get_dimensions_for_display(display_handle)
        aspect, quilt_width, quilt_height, default_cols, default_rows = quilt_settings_for_display(bridge, display_handle)
        quilt_cols = int(args.quilt_cols or default_cols)
        quilt_rows = int(args.quilt_rows or default_rows)
        if quilt_cols <= 0 or quilt_rows <= 0:
            raise ValueError("resolved quilt columns and rows must be positive")
        view_count = quilt_cols * quilt_rows

        bridge_window = bridge.instance_offscreen_window_gl(-1)
        print(
            "BuildLutNpz: "
            f"display={args.display_index}, native={native_width}x{native_height}, "
            f"quilt={quilt_width}x{quilt_height} {quilt_cols}x{quilt_rows}, "
            f"views={view_count}, layout={args.layout}, method={args.index_method}",
            file=sys.stderr,
            flush=True,
        )

        viewpoint_index, white_response, ramp_response, normalized_view_float, normalized_valid_mask = (
            recover_viewpoint_index_from_balanced_ramp(
                bridge,
                bridge_window,
                PixelFormats.RGBA,
                quilt_width,
                quilt_height,
                quilt_cols,
                quilt_rows,
                aspect,
                args.layout,
                view_count,
            )
        )
        metadata: dict[str, np.ndarray] = {
            "normalized_view_range": np.asarray(
                [float(np.nanmin(normalized_view_float)), float(np.nanmax(normalized_view_float))],
                dtype=np.float32,
            ),
            "normalized_valid_count": np.asarray(int(np.count_nonzero(normalized_valid_mask)), dtype=np.int64),
            "balanced_rank_remap": np.asarray(True),
        }
        if not args.no_save_responses:
            metadata["white_response"] = white_response
            metadata["ramp_response"] = ramp_response

        output = Path(args.output).expanduser() if args.output else default_output_path(native_width, native_height, view_count)
        path = save_viewpoint_index_npz(
            output,
            viewpoint_index,
            view_count=view_count,
            quilt_cols=quilt_cols,
            quilt_rows=quilt_rows,
            layout=args.layout,
            index_method=args.index_method,
            extra_metadata=metadata,
        )
        unique_views = np.unique(viewpoint_index)
        print(
            f"Saved balanced-ramp viewpoint index to {path} "
            f"shape={viewpoint_index.shape}, view_count={view_count}, "
            f"used_views={int(unique_views.min())}..{int(unique_views.max())} ({len(unique_views)} unique)",
            file=sys.stderr,
            flush=True,
        )
    finally:
        try:
            bridge.uninitialize()
        except Exception:
            pass
        if context is not None:
            import glfw

            glfw.destroy_window(context)
        try:
            import glfw

            glfw.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
