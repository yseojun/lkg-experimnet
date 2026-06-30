import unittest
import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from lkg_experiment.render_looking_glass import (
    build_parser,
    resolve_bridge_or_fallback_display,
    resolve_effective_view_count,
    resolve_panel_render_size,
)


class RenderLookingGlassTest(unittest.TestCase):
    def test_parser_accepts_checkpoint_lut_camera_cluster_and_remap_args(self):
        args = build_parser().parse_args(
            [
                "--checkpoint-path",
                "/tmp/ckpt.pt",
                "--viewpoint-index-path",
                "/tmp/lut.npz",
                "--views",
                "66",
                "--data-dir",
                "/tmp/data",
                "--camera-source",
                "dataset",
                "--camera-split",
                "test",
                "--camera-index",
                "2",
                "--coherent-cluster-size",
                "4",
                "--no-remapping",
                "--max-frames",
                "1",
            ]
        )

        self.assertEqual(args.checkpoint_path, "/tmp/ckpt.pt")
        self.assertEqual(args.viewpoint_index_path, "/tmp/lut.npz")
        self.assertEqual(args.views, 66)
        self.assertEqual(args.data_dir, "/tmp/data")
        self.assertEqual(args.camera_source, "dataset")
        self.assertEqual(args.camera_split, "test")
        self.assertEqual(args.camera_index, 2)
        self.assertEqual(args.coherent_cluster_size, 4)
        self.assertTrue(args.no_remapping)
        self.assertEqual(args.max_frames, 1)

    def test_parser_accepts_4dgs_panel_inputs(self):
        args = build_parser().parse_args(
            [
                "--four-dgs-model-path",
                "/tmp/4dgs-model",
                "--four-dgs-code-root",
                "/tmp/4DGaussians",
                "--four-dgs-iteration",
                "14000",
                "--four-dgs-time",
                "0.75",
            ]
        )

        self.assertEqual(args.four_dgs_model_path, "/tmp/4dgs-model")
        self.assertEqual(args.four_dgs_code_root, "/tmp/4DGaussians")
        self.assertEqual(args.four_dgs_iteration, 14000)
        self.assertEqual(args.four_dgs_time, 0.75)

    def test_resolve_panel_render_size_uses_native_display_by_default(self):
        width, height, label = resolve_panel_render_size(
            requested_width=0,
            requested_height=0,
            native_width=1440,
            native_height=2560,
            allow_non_native=False,
        )

        self.assertEqual((width, height, label), (1440, 2560, "native"))

    def test_resolve_panel_render_size_rejects_partial_non_native_size(self):
        with self.assertRaisesRegex(ValueError, "Native interlaced panel output"):
            resolve_panel_render_size(
                requested_width=720,
                requested_height=1280,
                native_width=1440,
                native_height=2560,
                allow_non_native=False,
            )

    def test_resolve_effective_view_count_prefers_lut_metadata_when_views_omitted(self):
        self.assertEqual(resolve_effective_view_count(requested_views=0, file_view_count=66, bridge_view_count=45), 66)
        self.assertEqual(resolve_effective_view_count(requested_views=64, file_view_count=66, bridge_view_count=45), 64)
        self.assertEqual(resolve_effective_view_count(requested_views=0, file_view_count=None, bridge_view_count=45), 45)

    def test_bridge_display_fallback_uses_explicit_panel_geometry(self):
        class EmptyBridge:
            def get_displays(self):
                return []

        args = Namespace(
            display_index=0,
            allow_bridge_display_fallback=True,
            width=1440,
            height=2560,
            window_x=1920,
            window_y=0,
        )

        handle, info = resolve_bridge_or_fallback_display(EmptyBridge(), args)

        self.assertEqual(handle, -1)
        self.assertEqual(info["dimensions"], (1440, 2560))
        self.assertEqual(info["position"], (1920, 0))
        self.assertEqual(info["name"], "manual-fallback")

    def test_bridge_display_fallback_requires_explicit_panel_size(self):
        class EmptyBridge:
            def get_displays(self):
                return []

        args = Namespace(
            display_index=0,
            allow_bridge_display_fallback=True,
            width=0,
            height=0,
            window_x=None,
            window_y=None,
        )

        with self.assertRaisesRegex(RuntimeError, "--width and --height"):
            resolve_bridge_or_fallback_display(EmptyBridge(), args)

    def test_panel_wrapper_script_exists_and_uses_render_entrypoint(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_drums_panel.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("lkg_experiment.render_looking_glass", text)
        self.assertIn("CHECKPOINT_PATH", text)
        self.assertIn("VIEWPOINT_INDEX_PATH", text)
        self.assertIn("lkg_go_1440x2560_66_views_lkg_calibration.npz", text)

    def test_panel_wrapper_defaults_match_current_cuda_environment_and_server_paths(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_drums_panel.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("rtgs-coherent-cu121", text)
        self.assertIn("/data/ysj/dataset", text)
        self.assertIn("/data/ysj/result/coherent-raster", text)
        self.assertIn("--bridge-sdk-root", text)
        self.assertIn("--gsplat-root", text)

    def test_parser_defaults_to_fixed_panel_renderer(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.panel_renderer, "fixed")
        self.assertEqual(args.texture_upload_mode, "auto")

    def test_parser_accepts_cuda_gl_texture_upload_mode(self):
        args = build_parser().parse_args(["--texture-upload-mode", "cuda-gl"])

        self.assertEqual(args.texture_upload_mode, "cuda-gl")

    def test_prepare_pyopengl_before_bridge_imports_gl_without_shader_entrypoints(self):
        from lkg_experiment.render_looking_glass import prepare_pyopengl_before_bridge

        gl = SimpleNamespace(
            glCreateShader=mock.Mock(),
            glCreateProgram=mock.Mock(),
        )

        with mock.patch.dict("sys.modules", {"OpenGL": SimpleNamespace(GL=gl)}):
            prepare_pyopengl_before_bridge()

        gl.glCreateShader.assert_not_called()
        gl.glCreateProgram.assert_not_called()

    def test_wake_x11_display_for_panel_turns_on_dpms_and_deactivates_screensaver(self):
        from lkg_experiment.render_looking_glass import wake_x11_display_for_panel

        run = mock.Mock()
        with mock.patch.dict(
            os.environ,
            {
                "DISPLAY": ":0",
                "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1003/bus",
            },
            clear=True,
        ), mock.patch("subprocess.run", run):
            wake_x11_display_for_panel()

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0], ["xset", "dpms", "force", "on"])
        self.assertEqual(commands[1], ["xset", "s", "off", "s", "noblank", "-dpms"])
        self.assertEqual(commands[2][:7], ["gdbus", "call", "--session", "--dest", "org.gnome.ScreenSaver", "--object-path", "/org/gnome/ScreenSaver"])
        self.assertEqual(commands[2][-2:], ["org.gnome.ScreenSaver.SetActive", "false"])
        self.assertEqual(run.call_args_list[2].kwargs["env"]["DBUS_SESSION_BUS_ADDRESS"], "unix:path=/run/user/1003/bus")

    def test_wake_x11_display_for_panel_noops_without_display(self):
        from lkg_experiment.render_looking_glass import wake_x11_display_for_panel

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("subprocess.run") as run:
            wake_x11_display_for_panel()

        run.assert_not_called()

    def test_bridge_bootstrap_context_fixed_renderer_avoids_shader_entrypoints(self):
        from lkg_experiment.render_looking_glass import init_bridge_bootstrap_context

        window = object()
        glfw = SimpleNamespace(
            CONTEXT_VERSION_MAJOR=0x00022002,
            CONTEXT_VERSION_MINOR=0x00022003,
            OPENGL_PROFILE=0x00022008,
            OPENGL_CORE_PROFILE=0x00032001,
            VISIBLE=0x00020004,
            FALSE=0,
            TRUE=1,
            init=mock.Mock(return_value=True),
            window_hint=mock.Mock(),
            create_window=mock.Mock(return_value=window),
            make_context_current=mock.Mock(),
            swap_interval=mock.Mock(),
        )
        gl = SimpleNamespace(
            GL_VERTEX_SHADER=0x8B31,
            glCreateShader=mock.Mock(return_value=7),
            glDeleteShader=mock.Mock(),
            glCreateProgram=mock.Mock(return_value=11),
            glDeleteProgram=mock.Mock(),
        )

        with mock.patch.dict("sys.modules", {"glfw": glfw, "OpenGL": SimpleNamespace(GL=gl)}):
            result = init_bridge_bootstrap_context(Namespace(panel_renderer="fixed"))

        self.assertIs(result, window)
        glfw.create_window.assert_called_once_with(1, 1, "LKG Bridge bootstrap", None, None)
        glfw.make_context_current.assert_called_once_with(window)
        glfw.swap_interval.assert_called_once_with(0)
        gl.glCreateShader.assert_not_called()
        gl.glDeleteShader.assert_not_called()
        gl.glCreateProgram.assert_not_called()
        gl.glDeleteProgram.assert_not_called()

    def test_bridge_bootstrap_context_shader_renderer_prewarms_shader_entrypoints(self):
        from lkg_experiment.render_looking_glass import init_bridge_bootstrap_context

        window = object()
        glfw = SimpleNamespace(
            CONTEXT_VERSION_MAJOR=0x00022002,
            CONTEXT_VERSION_MINOR=0x00022003,
            OPENGL_PROFILE=0x00022008,
            OPENGL_CORE_PROFILE=0x00032001,
            VISIBLE=0x00020004,
            FALSE=0,
            TRUE=1,
            init=mock.Mock(return_value=True),
            window_hint=mock.Mock(),
            create_window=mock.Mock(return_value=window),
            make_context_current=mock.Mock(),
            swap_interval=mock.Mock(),
        )
        gl = SimpleNamespace(
            GL_VERTEX_SHADER=0x8B31,
            glCreateShader=mock.Mock(return_value=7),
            glDeleteShader=mock.Mock(),
            glCreateProgram=mock.Mock(return_value=11),
            glDeleteProgram=mock.Mock(),
        )

        with mock.patch.dict("sys.modules", {"glfw": glfw, "OpenGL": SimpleNamespace(GL=gl)}):
            result = init_bridge_bootstrap_context(Namespace(panel_renderer="shader"))

        self.assertIs(result, window)
        gl.glCreateShader.assert_called_once_with(gl.GL_VERTEX_SHADER)
        gl.glDeleteShader.assert_called_once_with(7)
        gl.glCreateProgram.assert_called_once_with()
        gl.glDeleteProgram.assert_called_once_with(11)

    def test_fixed_panel_draw_uses_framebuffer_blit_without_legacy_matrix(self):
        from lkg_experiment.render_looking_glass import draw_panel_texture

        window = object()
        glfw = SimpleNamespace(get_framebuffer_size=mock.Mock(return_value=(1440, 2560)))
        gl = SimpleNamespace(
            GL_FRAMEBUFFER=0x8D40,
            GL_READ_FRAMEBUFFER=0x8CA8,
            GL_DRAW_FRAMEBUFFER=0x8CA9,
            GL_COLOR_ATTACHMENT0=0x8CE0,
            GL_TEXTURE_2D=0x0DE1,
            GL_DEPTH_TEST=0x0B71,
            GL_BLEND=0x0BE2,
            GL_DITHER=0x0BD0,
            GL_FRAMEBUFFER_SRGB=0x8DB9,
            GL_COLOR_BUFFER_BIT=0x4000,
            GL_NEAREST=0x2600,
            GL_BACK=0x0405,
            GL_FRAMEBUFFER_COMPLETE=0x8CD5,
            GL_PROJECTION=0x1701,
            GL_MODELVIEW=0x1700,
            GL_TEXTURE0=0x84C0,
            GL_QUADS=0x0007,
            glBindFramebuffer=mock.Mock(),
            glViewport=mock.Mock(),
            glDisable=mock.Mock(),
            glClearColor=mock.Mock(),
            glClear=mock.Mock(),
            glGenFramebuffers=mock.Mock(return_value=17),
            glFramebufferTexture2D=mock.Mock(),
            glCheckFramebufferStatus=mock.Mock(return_value=0x8CD5),
            glReadBuffer=mock.Mock(),
            glDrawBuffer=mock.Mock(),
            glBlitFramebuffer=mock.Mock(),
            glDeleteFramebuffers=mock.Mock(),
            glMatrixMode=mock.Mock(),
            glLoadIdentity=mock.Mock(),
            glEnable=mock.Mock(),
            glActiveTexture=mock.Mock(),
            glBindTexture=mock.Mock(),
            glBegin=mock.Mock(),
            glTexCoord2f=mock.Mock(),
            glVertex2f=mock.Mock(),
            glEnd=mock.Mock(),
        )

        with mock.patch.dict("sys.modules", {"glfw": glfw, "OpenGL": SimpleNamespace(GL=gl)}):
            draw_panel_texture(window, None, None, 5)

        gl.glMatrixMode.assert_not_called()
        gl.glBegin.assert_not_called()
        gl.glBlitFramebuffer.assert_called_once_with(
            0,
            2560,
            1440,
            0,
            0,
            0,
            1440,
            2560,
            gl.GL_COLOR_BUFFER_BIT,
            gl.GL_NEAREST,
        )


if __name__ == "__main__":
    unittest.main()
