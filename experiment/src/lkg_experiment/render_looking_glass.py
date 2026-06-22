#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from lkg_experiment.build_lut_npz import _load_bridge_api, as_int, install_bridge_sdk_root
from lkg_experiment.coherent_gsplat_bridge import (
    build_cr_lookup_arrays,
    install_gsplat_root,
    load_splats_from_checkpoint,
    lookup_arrays_to_torch,
    resolve_checkpoint_path,
)
from lkg_experiment.coherent_raster import build_linear_viewpoint_index
from lkg_experiment.coherent_raster_experiment import (
    CoherentRasterRenderer,
    OrbitViewSynthesizer,
    compact_viewpoint_index,
    load_viewpoint_index_file,
    tensor_to_hwc_uint8,
)
from lkg_experiment.run_coherent_raster_experiment import (
    DEFAULT_BRIDGE_SDK_ROOT,
    DEFAULT_GSPLAT_ROOT,
    DEFAULT_VIEWPOINT_INDEX_PATH,
    default_checkpoint_path,
    estimate_bbox_camera,
    load_blender_camera,
    load_cfg,
    load_colmap_camera,
    maybe_resolve_data_dir,
    result_root_from_checkpoint,
    subset_splats_for_debug,
)


DEFAULT_PANEL_VIEWS = 45


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a CoherentRaster/gsplat checkpoint to a Looking Glass native interlaced panel window"
    )
    parser.add_argument("--checkpoint-path", default=str(default_checkpoint_path()))
    parser.add_argument("--iteration", type=int, help="Checkpoint iteration; omitted means latest rank0 checkpoint")
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--gsplat-root", default=str(DEFAULT_GSPLAT_ROOT))
    parser.add_argument("--bridge-sdk-root", default=str(DEFAULT_BRIDGE_SDK_ROOT))

    parser.add_argument("--data-dir", default="auto", help="'auto', a dataset path, or empty to force bbox camera")
    parser.add_argument("--camera-source", choices=("auto", "dataset", "bbox"), default="auto")
    parser.add_argument("--camera-split", choices=("auto", "val", "train", "test"), default="auto")
    parser.add_argument("--camera-index", default=0, type=int)
    parser.add_argument("--display-index", default=0, type=int)
    parser.add_argument("--width", default=0, type=int, help="0 uses Bridge native display width")
    parser.add_argument("--height", default=0, type=int, help="0 uses Bridge native display height")
    parser.add_argument("--views", default=0, type=int, help="0 uses LUT metadata, then Bridge quilt count")
    parser.add_argument("--view-degree", default=53.0, type=float)
    parser.add_argument("--orbit-direction", default=-1, type=int)
    parser.add_argument("--orbit-center-distance", default=0.0, type=float, help="0 derives distance from initial camera")
    parser.add_argument("--fov-y", default=60.0, type=float, help="Fallback bbox-camera vertical FOV in degrees")
    parser.add_argument("--no-crop-to-fill", action="store_true")

    parser.add_argument("--map-mode", choices=("file", "linear"), default="file")
    parser.add_argument("--viewpoint-index-path", default=str(DEFAULT_VIEWPOINT_INDEX_PATH))
    parser.add_argument("--no-compact-view-index", action="store_true")
    parser.add_argument("--coherent-quantize", choices=("floor", "nearest"), default="floor")

    parser.add_argument("--coherent-cluster-size", "--cluster", dest="coherent_cluster_size", default=8, type=int)
    parser.add_argument("--tile-size", default=16, type=int)
    parser.add_argument("--no-remapping", action="store_true")
    parser.add_argument("--near-plane", default=0.01, type=float)
    parser.add_argument("--far-plane", default=1e10, type=float)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--camera-model", choices=("pinhole", "ortho", "fisheye"), default="pinhole")
    parser.add_argument("--coherent-color-mode", choices=("sh", "dc"), default="sh")
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--max-gaussians", default=0, type=int, help="Debug only; 0 keeps the full checkpoint")
    parser.add_argument("--debug-render", action="store_true")

    parser.add_argument(
        "--allow-non-native-panel-size",
        action="store_true",
        help="Allow non-native panel size for debug only; exact LKG output requires native size.",
    )
    parser.add_argument("--window-x", type=int)
    parser.add_argument("--window-y", type=int)
    parser.add_argument("--decorated", action="store_true")
    parser.add_argument("--not-floating", action="store_true")
    parser.add_argument("--swap-interval", default=0, type=int)
    parser.add_argument("--max-frames", default=0, type=int, help="0 means hold until the panel window is closed")
    parser.add_argument("--stats", action="store_true")
    return parser


def configure_x11_environment() -> None:
    if not os.environ.get("DISPLAY") and Path("/tmp/.X11-unix/X0").exists():
        os.environ["DISPLAY"] = ":0"
    if not os.environ.get("XAUTHORITY"):
        for candidate in (
            Path(f"/run/user/{os.getuid()}/gdm/Xauthority"),
            Path.home() / ".Xauthority",
        ):
            if candidate.exists():
                os.environ["XAUTHORITY"] = str(candidate)
                break
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")
    os.environ["WAYLAND_DISPLAY"] = ""


def graphics_env_summary() -> str:
    keys = ("DISPLAY", "XAUTHORITY", "XDG_SESSION_TYPE", "WAYLAND_DISPLAY", "PYOPENGL_PLATFORM")
    return ", ".join(f"{key}={os.environ.get(key, '') or '<unset>'}" for key in keys)


def resolve_panel_render_size(
    *,
    requested_width: int,
    requested_height: int,
    native_width: int,
    native_height: int,
    allow_non_native: bool,
) -> tuple[int, int, str]:
    width = int(requested_width or native_width)
    height = int(requested_height or native_height)
    if width <= 0 or height <= 0:
        raise ValueError("resolved render size must be positive")
    if width == int(native_width) and height == int(native_height):
        return width, height, "native"
    if allow_non_native:
        return width, height, "debug-non-native"
    raise ValueError(
        "Native interlaced panel output must use the display native resolution "
        f"{int(native_width)}x{int(native_height)}; got {width}x{height}. "
        "Use --allow-non-native-panel-size only for debug."
    )


def resolve_effective_view_count(
    *,
    requested_views: int,
    file_view_count: Optional[int],
    bridge_view_count: Optional[int],
) -> int:
    if requested_views > 0:
        return int(requested_views)
    if file_view_count is not None and int(file_view_count) > 0:
        return int(file_view_count)
    if bridge_view_count is not None and int(bridge_view_count) > 0:
        return int(bridge_view_count)
    return DEFAULT_PANEL_VIEWS


def bridge_view_count_from_quilt(quilt: Any) -> Optional[int]:
    if quilt is None:
        return None
    cols = as_int(getattr(quilt, "QuiltColumns", None), 0)
    rows = as_int(getattr(quilt, "QuiltRows", None), 0)
    if cols > 0 and rows > 0:
        return cols * rows
    return None


def validate_args(args: argparse.Namespace) -> None:
    if args.width < 0 or args.height < 0:
        raise ValueError("--width and --height must be non-negative; use 0 for Bridge native size")
    if args.views < 0:
        raise ValueError("--views must be non-negative; use 0 for LUT/Bridge default")
    if args.coherent_cluster_size <= 0:
        raise ValueError("--coherent-cluster-size must be positive")
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive")
    if args.max_gaussians < 0:
        raise ValueError("--max-gaussians must be non-negative")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    if args.map_mode == "file" and not args.viewpoint_index_path:
        raise ValueError("--map-mode file requires --viewpoint-index-path")


def select_bridge_display(bridge: Any, display_index: int) -> tuple[int, dict[str, Any]]:
    displays = bridge.get_displays()
    if not displays:
        raise RuntimeError("Bridge did not report any Looking Glass displays")
    if display_index < 0 or display_index >= len(displays):
        raise IndexError(f"--display-index {display_index} outside Bridge display count {len(displays)}")

    handle = displays[display_index]
    info = {
        "handle": int(handle),
        "name": safe_call(lambda: bridge.get_device_name_for_display(handle), ""),
        "serial": safe_call(lambda: bridge.get_device_serial_for_display(handle), ""),
        "dimensions": safe_call(lambda: bridge.get_dimensions_for_display(handle), None),
        "position": safe_call(lambda: bridge.get_window_position_for_display(handle), (0, 0)),
        "quilt": safe_call(lambda: bridge.get_default_quilt_settings_for_display(handle), None),
    }
    if info["dimensions"] is None:
        raise RuntimeError("Bridge display dimensions unavailable")
    return handle, info


def safe_call(fn, fallback: Any = None) -> Any:
    try:
        return fn()
    except Exception:
        return fallback


def build_viewpoint_index(args, width: int, height: int) -> tuple[np.ndarray, int, int, np.ndarray, str]:
    if args.map_mode == "file":
        viewpoint_index, file_view_count, metadata = load_viewpoint_index_file(args.viewpoint_index_path, width, height)
        bridge_count = None
        source_view_count = resolve_effective_view_count(
            requested_views=args.views,
            file_view_count=file_view_count,
            bridge_view_count=bridge_count,
        )
        if file_view_count is not None and int(file_view_count) != int(source_view_count):
            raise ValueError(f"viewpoint index view_count={file_view_count} but renderer views={source_view_count}")
        label_parts = [f"file={args.viewpoint_index_path}"]
        for key in ("layout", "index_method", "quilt_cols", "quilt_rows"):
            if key in metadata:
                label_parts.append(f"{key}={metadata[key]}")
        label = ",".join(label_parts)
    else:
        source_view_count = resolve_effective_view_count(
            requested_views=args.views,
            file_view_count=None,
            bridge_view_count=None,
        )
        viewpoint_index = build_linear_viewpoint_index(
            width,
            height,
            source_view_count,
            quantize=args.coherent_quantize,
        )
        label = f"linear(quantize={args.coherent_quantize})"

    view_labels = np.arange(source_view_count, dtype=np.int32)
    render_view_count = int(source_view_count)
    if args.map_mode == "file" and not args.no_compact_view_index:
        viewpoint_index, view_labels = compact_viewpoint_index(viewpoint_index, source_view_count)
        render_view_count = int(view_labels.size)
    return viewpoint_index, source_view_count, render_view_count, view_labels, label


def init_panel_window(args: argparse.Namespace, width: int, height: int, x: int, y: int):
    import glfw

    if not glfw.init():
        raise RuntimeError("failed to initialize GLFW")
    gl_major, gl_minor = (4, 1) if sys.platform == "darwin" else (4, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, gl_major)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, gl_minor)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    if sys.platform == "darwin":
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.DECORATED, glfw.TRUE if args.decorated else glfw.FALSE)
    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    if not args.not_floating:
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)

    window = glfw.create_window(width, height, "LKG CoherentRaster panel", None, None)
    if not window and sys.platform != "darwin":
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.DECORATED, glfw.TRUE if args.decorated else glfw.FALSE)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        if not args.not_floating:
            glfw.window_hint(glfw.FLOATING, glfw.TRUE)
        window = glfw.create_window(width, height, "LKG CoherentRaster panel", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("failed to create GLFW panel window")

    glfw.set_window_pos(window, int(x), int(y))
    glfw.make_context_current(window)
    glfw.swap_interval(int(args.swap_interval))
    glfw.show_window(window)
    glfw.set_window_pos(window, int(x), int(y))
    glfw.set_window_size(window, int(width), int(height))
    try:
        glfw.focus_window(window)
    except Exception:
        pass
    return window


def init_panel_texture(width: int, height: int) -> int:
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
        None,
    )
    return int(texture)


def compile_shader(shader_type: int, source: str) -> int:
    from OpenGL import GL

    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
        GL.glDeleteShader(shader)
        raise RuntimeError(f"Shader compile failed: {log}")
    return int(shader)


def create_panel_program() -> int:
    from OpenGL import GL

    vertex_source = """
        #version 330 core
        layout (location = 0) in vec2 a_pos;
        layout (location = 1) in vec2 a_uv;
        out vec2 v_uv;
        void main() {
            v_uv = a_uv;
            gl_Position = vec4(a_pos, 0.0, 1.0);
        }
    """
    fragment_source = """
        #version 330 core
        in vec2 v_uv;
        out vec4 frag_color;
        uniform sampler2D u_texture;
        void main() {
            frag_color = texture(u_texture, v_uv);
        }
    """
    vertex_shader = compile_shader(GL.GL_VERTEX_SHADER, vertex_source)
    fragment_shader = compile_shader(GL.GL_FRAGMENT_SHADER, fragment_source)
    program = GL.glCreateProgram()
    GL.glAttachShader(program, vertex_shader)
    GL.glAttachShader(program, fragment_shader)
    GL.glLinkProgram(program)
    GL.glDeleteShader(vertex_shader)
    GL.glDeleteShader(fragment_shader)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        log = GL.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
        GL.glDeleteProgram(program)
        raise RuntimeError(f"Shader link failed: {log}")
    GL.glUseProgram(program)
    GL.glUniform1i(GL.glGetUniformLocation(program, "u_texture"), 0)
    GL.glUseProgram(0)
    return int(program)


def create_panel_quad() -> tuple[int, int]:
    import ctypes
    from OpenGL import GL

    vertices = np.array(
        [
            -1.0, -1.0, 0.0, 1.0,
            1.0, -1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 0.0,
            -1.0, -1.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 0.0,
            -1.0, 1.0, 0.0, 0.0,
        ],
        dtype=np.float32,
    )
    vao = GL.glGenVertexArrays(1)
    vbo = GL.glGenBuffers(1)
    GL.glBindVertexArray(vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
    stride = 4 * vertices.itemsize
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(1)
    GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(2 * vertices.itemsize))
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)
    return int(vao), int(vbo)


def rendered_to_rgba(rendered: Any) -> np.ndarray:
    rgb = tensor_to_hwc_uint8(rendered)
    rgba = np.empty((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    return rgba


def upload_direct(texture: int, width: int, height: int, rgba: np.ndarray) -> None:
    from OpenGL import GL

    GL.glBindTexture(GL.GL_TEXTURE_2D, int(texture))
    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
    GL.glTexSubImage2D(
        GL.GL_TEXTURE_2D,
        0,
        0,
        0,
        int(width),
        int(height),
        GL.GL_RGBA,
        GL.GL_UNSIGNED_BYTE,
        rgba,
    )


def draw_panel_texture(window: Any, program: int, vao: int, texture: int) -> tuple[int, int]:
    import glfw
    from OpenGL import GL

    fb_width, fb_height = glfw.get_framebuffer_size(window)
    if fb_width <= 0 or fb_height <= 0:
        return int(fb_width), int(fb_height)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    GL.glViewport(0, 0, int(fb_width), int(fb_height))
    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glDisable(GL.GL_BLEND)
    GL.glDisable(GL.GL_DITHER)
    try:
        GL.glDisable(GL.GL_FRAMEBUFFER_SRGB)
    except Exception:
        pass
    GL.glClearColor(0.0, 0.0, 0.0, 1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glUseProgram(int(program))
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, int(texture))
    GL.glBindVertexArray(int(vao))
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
    GL.glBindVertexArray(0)
    GL.glUseProgram(0)
    return int(fb_width), int(fb_height)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    configure_x11_environment()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CoherentRaster panel rendering")

    install_bridge_sdk_root(args.bridge_sdk_root)
    BridgeAPI, _PixelFormats = _load_bridge_api()
    gsplat_root = install_gsplat_root(args.gsplat_root)
    checkpoint = resolve_checkpoint_path(args.checkpoint_path, iteration=args.iteration, rank=args.rank)
    cfg = load_cfg(result_root_from_checkpoint(checkpoint))

    bridge = BridgeAPI()
    window = None
    texture = None
    program = None
    vao = None
    vbo = None
    try:
        if not bridge.initialize("LkgExperimentRenderPanel"):
            raise RuntimeError("Bridge initialize failed")
        display_handle, display_info = select_bridge_display(bridge, int(args.display_index))
        native_width, native_height = display_info["dimensions"]
        panel_x, panel_y = display_info["position"]
        if args.window_x is not None:
            panel_x = int(args.window_x)
        if args.window_y is not None:
            panel_y = int(args.window_y)
        width, height, size_label = resolve_panel_render_size(
            requested_width=args.width,
            requested_height=args.height,
            native_width=int(native_width),
            native_height=int(native_height),
            allow_non_native=bool(args.allow_non_native_panel_size),
        )
        try:
            window = init_panel_window(args, width, height, int(panel_x), int(panel_y))
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc}. GLFW needs access to the local X11 display. "
                f"Current graphics environment: {graphics_env_summary()}."
            ) from exc
        texture = init_panel_texture(width, height)
        program = create_panel_program()
        vao, vbo = create_panel_quad()

        viewpoint_index, source_view_count, render_view_count, view_labels, mapping_label = build_viewpoint_index(
            args,
            width,
            height,
        )
        if args.views > 0 and source_view_count != int(args.views):
            raise RuntimeError("internal view count resolution mismatch")
        if args.map_mode == "linear" and args.views == 0:
            bridge_count = bridge_view_count_from_quilt(display_info["quilt"])
            if bridge_count is not None and bridge_count != source_view_count:
                source_view_count = bridge_count
                render_view_count = bridge_count
                view_labels = np.arange(source_view_count, dtype=np.int32)
                viewpoint_index = build_linear_viewpoint_index(
                    width,
                    height,
                    source_view_count,
                    quantize=args.coherent_quantize,
                )
                mapping_label = f"linear(quantize={args.coherent_quantize})"

        lookup = build_cr_lookup_arrays(
            viewpoint_index,
            tile_size=args.tile_size,
            use_remapping=not args.no_remapping,
        )
        view_idx_matrix, subpixel_coord_matrix = lookup_arrays_to_torch(lookup, device="cuda")

        print(f"Loading checkpoint: {checkpoint}", file=sys.stderr, flush=True)
        splats, step = load_splats_from_checkpoint(checkpoint, device="cuda")
        splats = subset_splats_for_debug(splats, args.max_gaussians)

        data_dir = maybe_resolve_data_dir(args, cfg)
        dataset_category = str(cfg.get("dataset_category", "") or "").strip().lower()
        if not dataset_category:
            dataset_category = "blender" if data_dir is not None and (data_dir / "transforms_train.json").exists() else "colmap"
        if args.camera_source == "dataset" and data_dir is None:
            raise RuntimeError("--camera-source dataset requested but no valid data_dir is available")
        if data_dir is not None and args.camera_source in {"auto", "dataset"}:
            if dataset_category == "blender":
                c2w_np, K_np, scene_center_np, camera_label = load_blender_camera(args, cfg, data_dir, width, height)
            elif dataset_category == "colmap":
                c2w_np, K_np, scene_center_np, camera_label = load_colmap_camera(args, cfg, data_dir, width, height)
            else:
                raise ValueError(f"Unsupported dataset_category: {dataset_category!r}")
        else:
            c2w_np, K_np, scene_center_np, camera_label = estimate_bbox_camera(args, splats["means"], width, height)
            dataset_category = "bbox"

        K = torch.from_numpy(K_np.astype(np.float32)).to(device="cuda", non_blocking=True).contiguous()
        c2w = torch.from_numpy(c2w_np.astype(np.float32)).to(device="cuda", non_blocking=True).contiguous()
        initial_distance = float(np.linalg.norm(scene_center_np.astype(np.float64) - c2w_np[:3, 3].astype(np.float64)))
        orbit_center_distance = float(args.orbit_center_distance or max(initial_distance, 1e-4))
        orbit_center = c2w[:3, 3] + c2w[:3, 2] * orbit_center_distance

        background = torch.ones(3, device="cuda") if bool(args.white_background or dataset_category == "blender") else None
        renderer = CoherentRasterRenderer(
            splats,
            sh_degree=int(cfg.get("sh_degree", args.sh_degree)),
            near_plane=float(cfg.get("near_plane", args.near_plane)),
            far_plane=float(cfg.get("far_plane", args.far_plane)),
            camera_model=str(cfg.get("camera_model", args.camera_model)),
            background=background,
            color_mode=args.coherent_color_mode,
        )
        del splats

        synthesizer = OrbitViewSynthesizer(
            view_labels=view_labels,
            source_view_count=source_view_count,
            cluster_size=args.coherent_cluster_size,
            view_degree=args.view_degree,
            orbit_direction=args.orbit_direction,
            device="cuda",
        )
        print(
            "RenderLookingGlass: "
            f"display={display_info['name']} serial={display_info['serial']}, "
            f"native={native_width}x{native_height}, window={width}x{height}@{panel_x},{panel_y}, "
            f"size_mode={size_label}, checkpoint={checkpoint} step={step}, gsplat_root={gsplat_root}, "
            f"camera={camera_label}, views={render_view_count}/{source_view_count}, "
            f"cluster={args.coherent_cluster_size}, remap={not args.no_remapping}, mapping={mapping_label}",
            file=sys.stderr,
            flush=True,
        )

        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            adjacent_viewmats = synthesizer(c2w, orbit_center)
            rendered = renderer.render(
                adjacent_viewmats=adjacent_viewmats,
                K=K,
                view_idx_matrix=view_idx_matrix,
                subpixel_coord_matrix=subpixel_coord_matrix,
                width=width,
                height=height,
                tile_size=args.tile_size,
                debug=args.debug_render,
            )
            torch.cuda.synchronize()
            render_elapsed = time.perf_counter() - t0

        rgba = rendered_to_rgba(rendered)
        upload_direct(texture, width, height, rgba)
        draw_panel_texture(window, program, vao, texture)
        import glfw

        glfw.swap_buffers(window)
        print(
            f"Rendered one panel frame in {render_elapsed * 1000.0:.2f} ms; "
            "holding window open until Esc/window close.",
            file=sys.stderr,
            flush=True,
        )

        total_frames = 0
        stats_frames = 0
        stats_t0 = time.perf_counter()
        while not glfw.window_should_close(window):
            glfw.poll_events()
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                glfw.set_window_should_close(window, True)
                break
            draw_panel_texture(window, program, vao, texture)
            glfw.swap_buffers(window)
            total_frames += 1
            stats_frames += 1
            if args.stats:
                now = time.perf_counter()
                if now - stats_t0 >= 1.0:
                    print(f"RenderLookingGlass display FPS: {stats_frames / (now - stats_t0):.2f}", file=sys.stderr, flush=True)
                    stats_frames = 0
                    stats_t0 = now
            if args.max_frames and total_frames >= args.max_frames:
                break
    finally:
        try:
            from OpenGL import GL

            if vbo is not None:
                GL.glDeleteBuffers(1, [vbo])
            if vao is not None:
                GL.glDeleteVertexArrays(1, [vao])
            if program is not None:
                GL.glDeleteProgram(program)
            if texture is not None:
                GL.glDeleteTextures(1, [texture])
        except Exception:
            pass
        if window is not None:
            try:
                import glfw

                glfw.destroy_window(window)
                glfw.terminate()
            except Exception:
                pass
        try:
            bridge.uninitialize()
        except Exception:
            pass


if __name__ == "__main__":
    main()
