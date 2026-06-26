import unittest
from argparse import Namespace
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
