from __future__ import annotations

import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from lkg_experiment.omg4_ftgs import lkg_panel


class Omg4LkgPanelTest(unittest.TestCase):
    def test_parser_defaults_match_single_panel_runtime(self):
        args = lkg_panel.build_parser().parse_args([])

        self.assertEqual(args.checkpoint_path, "/data/ysj/result/4dgs/OMG4-FTGS_weights/ours_L_weight/cook_spinach.xz")
        self.assertEqual(args.data_path, "/data/ysj/dataset/N3DV/cook_spinach")
        self.assertEqual(args.width, 1440)
        self.assertEqual(args.height, 2560)
        self.assertEqual(args.views, 66)
        self.assertEqual(args.cluster_size, 8)
        self.assertEqual(args.texture_upload_mode, "auto")
        self.assertEqual(args.max_frames, 0)

    def test_render_panel_frame_materializes_and_renders_once(self):
        args = lkg_panel.build_parser().parse_args(
            [
                "--device",
                "cpu",
                "--width",
                "4",
                "--height",
                "3",
                "--views",
                "4",
                "--cluster-size",
                "2",
                "--map-mode",
                "linear",
            ]
        )
        frame = SimpleNamespace(index=7, timestamp=0.25)
        camera_set = SimpleNamespace(
            K=np.eye(3, dtype=np.float32),
            width=4,
            height=3,
            viewmat_at=mock.Mock(return_value=np.eye(4, dtype=np.float32)),
        )
        splats = SimpleNamespace(means=torch.zeros((2, 3), dtype=torch.float32))
        model = SimpleNamespace(materialize=mock.Mock(return_value=splats))
        lookup = SimpleNamespace(
            view_idx_matrix=torch.zeros((1,), dtype=torch.int64),
            subpixel_coord_matrix=torch.zeros((1,), dtype=torch.int64),
            lookup_cpu_ms=1.0,
            lookup_h2d_ms=2.0,
        )
        image = torch.zeros((3, 3, 4), dtype=torch.float32)

        with (
            mock.patch.object(lkg_panel, "install_gsplat_root"),
            mock.patch.object(lkg_panel, "load_test_frames", return_value=[frame]),
            mock.patch.object(lkg_panel, "load_pose_camera", return_value=camera_set),
            mock.patch.object(lkg_panel, "load_dynamic_gaussians", return_value=model),
            mock.patch.object(lkg_panel, "build_interlaced_viewpoint_index", return_value=(np.zeros((3, 4, 3), dtype=np.int32), {"mode": "linear"})),
            mock.patch.object(lkg_panel, "prepare_cr_lookup_tensors", return_value=lookup),
            mock.patch.object(lkg_panel, "estimate_orbit_center", return_value=torch.tensor([0.0, 0.0, 1.0])),
            mock.patch.object(lkg_panel, "synthesize_interlaced_viewmats", return_value=torch.eye(4).view(1, 1, 4, 4)),
            mock.patch.object(lkg_panel, "render_splats_interlaced_coherent", return_value=(image, {"timing_ms": {"cr_projection_ms": 1.0}, "post_ms": 0.5})) as render,
        ):
            result = lkg_panel.render_panel_frame(args)

        model.materialize.assert_called_once_with(0.25)
        render.assert_called_once()
        self.assertIs(result.image, image)
        self.assertEqual(result.frame_index, 7)
        self.assertEqual(result.lookup_cpu_ms, 1.0)
        self.assertEqual(result.lookup_h2d_ms, 2.0)

    def test_wrapper_script_exists_and_uses_module_main(self):
        script = Path(__file__).resolve().parents[1] / "omg4_ftgs_lkg_panel.py"

        text = script.read_text(encoding="utf-8")

        self.assertIn("lkg_experiment.omg4_ftgs.lkg_panel", text)
        self.assertIn("main", text)

    def test_shell_script_uses_single_panel_entrypoint(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_omg4_panel.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("lkg_experiment.omg4_ftgs.lkg_panel", text)
        self.assertIn("cook_spinach.xz", text)
        self.assertIn("lkg_go_1440x2560_66_views_lkg_calibration.npz", text)

    def test_pyproject_registers_console_script(self):
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

        self.assertEqual(
            data["project"]["scripts"]["lkg-omg4-ftgs-lkg-panel"],
            "lkg_experiment.omg4_ftgs.lkg_panel:main",
        )


if __name__ == "__main__":
    unittest.main()
