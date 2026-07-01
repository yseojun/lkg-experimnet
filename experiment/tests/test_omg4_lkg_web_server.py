from __future__ import annotations

import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lkg_experiment.omg4_ftgs import lkg_web_server
from lkg_experiment.rtgs_coherent import lkg_web_server as rtgs_web


class Omg4LkgWebServerTest(unittest.TestCase):
    def test_parser_defaults_match_omg4_lkg_runtime(self):
        args = lkg_web_server.build_parser().parse_args([])

        self.assertEqual(args.weights_root, "/data/ysj/result/4dgs/OMG4-FTGS_weights")
        self.assertEqual(args.weight_group, "ours_L_weight")
        self.assertEqual(args.data_root, "/data/ysj/dataset/N3DV")
        self.assertEqual(args.width, 1440)
        self.assertEqual(args.height, 2560)
        self.assertEqual(args.views, 66)
        self.assertEqual(args.cluster_size, 8)
        self.assertEqual(args.port, 8090)
        self.assertIsNone(args.torch_extensions_dir)
        self.assertIsNone(args.texture_upload_cuda_device)

    def test_parser_accepts_explicit_cuda_gl_upload_device(self):
        args = lkg_web_server.build_parser().parse_args(["--texture-upload-cuda-device", "0"])

        self.assertEqual(args.texture_upload_cuda_device, 0)

    def test_scan_omg4_checkpoints_discovers_dataset_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "weights" / "ours_L_weight"
            data = root / "N3DV"
            weights.mkdir(parents=True)
            data.mkdir()
            for scene in ["cook_spinach", "flame_salmon"]:
                (weights / f"{scene}.xz").write_bytes(b"checkpoint")
            (data / "cook_spinach").mkdir()
            (data / "flame_salmon_1").mkdir()

            options = lkg_web_server.scan_omg4_checkpoints(
                root / "weights",
                weight_group="ours_L_weight",
                data_root=data,
            )

        self.assertEqual([option.scene for option in options], ["cook_spinach", "flame_salmon"])
        self.assertEqual(options[0].id, "ours_L_weight:cook_spinach")
        self.assertEqual(options[0].dataset_scene, "cook_spinach")
        self.assertEqual(options[1].dataset_scene, "flame_salmon_1")
        self.assertEqual(options[1].checkpoint, "flame_salmon.xz")

    def test_build_playback_timeline_uses_same_camera_all_timestamps(self):
        frames = [
            SimpleNamespace(index=0, timestamp=0.0),
            SimpleNamespace(index=7, timestamp=0.25),
        ]

        timeline = lkg_web_server.build_playback_timeline(camera_index=3, frames=frames)

        self.assertEqual([frame.frame_index for frame in timeline], [0, 1])
        self.assertEqual([frame.camera_index for frame in timeline], [3, 3])
        self.assertEqual([frame.n3dv_frame_index for frame in timeline], [0, 7])
        self.assertEqual([frame.timestamp for frame in timeline], [0.0, 0.25])

    def test_prepare_runtime_bundle_loads_model_and_prebuilds_lookup_once(self):
        option = lkg_web_server.Omg4CheckpointOption(
            id="ours_L_weight:cook_spinach",
            scene="cook_spinach",
            dataset_scene="cook_spinach",
            weight_group="ours_L_weight",
            checkpoint_path=Path("/weights/cook_spinach.xz"),
            data_path=Path("/data/cook_spinach"),
            checkpoint="cook_spinach.xz",
        )
        args = lkg_web_server.build_parser().parse_args(["--device", "cpu", "--width", "4", "--height", "3", "--views", "4"])
        frames = [SimpleNamespace(index=0, timestamp=0.0), SimpleNamespace(index=1, timestamp=0.5)]
        camera_set = SimpleNamespace(
            K=np.eye(3, dtype=np.float32),
            width=4,
            height=3,
            viewmat_at=mock.Mock(return_value=np.eye(4, dtype=np.float32)),
        )
        splats = SimpleNamespace(means=np.zeros((2, 3), dtype=np.float32))
        model = SimpleNamespace(materialize=mock.Mock(return_value=splats))
        prepared_lookup = SimpleNamespace(
            view_idx_matrix=object(),
            subpixel_coord_matrix=object(),
            lookup_cpu_ms=1.0,
            lookup_h2d_ms=2.0,
        )

        with (
            mock.patch.object(lkg_web_server, "load_test_frames", return_value=frames),
            mock.patch.object(lkg_web_server, "load_pose_camera", return_value=camera_set),
            mock.patch.object(lkg_web_server, "load_dynamic_gaussians", return_value=model),
            mock.patch.object(lkg_web_server, "build_interlaced_viewpoint_index", return_value=(np.zeros((3, 4, 3), dtype=np.int32), {"mode": "linear"})),
            mock.patch.object(lkg_web_server, "prepare_cr_lookup_tensors", return_value=prepared_lookup),
            mock.patch.object(lkg_web_server, "estimate_orbit_center", return_value=np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ):
            bundle = lkg_web_server.prepare_runtime_bundle(args, option)

        self.assertIs(bundle.checkpoint, option)
        self.assertEqual(bundle.cursor.current_status().frame_count, 2)
        self.assertIs(bundle.context.dynamic_model, model)
        self.assertIs(bundle.context.view_idx_matrix, prepared_lookup.view_idx_matrix)
        self.assertEqual(bundle.context.lookup_cpu_ms, 1.0)
        model.materialize.assert_called_once_with(0.0)

    def test_render_omg4_lkg_frame_applies_orbit_and_returns_timing(self):
        args = lkg_web_server.build_parser().parse_args(["--device", "cpu", "--width", "4", "--height", "3", "--views", "4"])
        context = SimpleNamespace(
            args=args,
            dynamic_model=SimpleNamespace(materialize=mock.Mock(return_value=SimpleNamespace(means=np.zeros((2, 3), dtype=np.float32)))),
            base_c2w=np.eye(4, dtype=np.float32),
            orbit_center=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            K=np.eye(3, dtype=np.float32),
            viewpoint_index=np.zeros((3, 4, 3), dtype=np.int32),
            view_idx_matrix=object(),
            subpixel_coord_matrix=object(),
            lookup_cpu_ms=1.0,
            lookup_h2d_ms=2.0,
        )
        frame = rtgs_web.PlaybackFrame(frame_index=0, camera_index=0, n3dv_frame_index=0, timestamp=0.5, camera=None)
        image = SimpleNamespace(detach=mock.Mock(return_value="detached"), clamp=mock.Mock(return_value=SimpleNamespace(contiguous=mock.Mock(return_value="image"))))

        with (
            mock.patch.object(lkg_web_server, "apply_orbit_state_to_c2w", return_value=("c2w", "center")) as apply_orbit,
            mock.patch.object(lkg_web_server, "synthesize_interlaced_viewmats", return_value="viewmats") as synthesize,
            mock.patch.object(lkg_web_server, "render_splats_interlaced_coherent", return_value=(image, {"timing_ms": {"cr_projection_ms": 4.0}, "post_ms": 1.0})),
        ):
            rendered, render_ms = lkg_web_server.render_omg4_lkg_frame(
                context,
                frame,
                orbit_state=rtgs_web.OrbitState(yaw_deg=1.0),
            )

        self.assertEqual(rendered, "image")
        self.assertGreaterEqual(render_ms, 0.0)
        context.dynamic_model.materialize.assert_called_once_with(0.5)
        apply_orbit.assert_called_once()
        synthesize.assert_called_once_with(
            c2w="c2w",
            orbit_center="center",
            views=4,
            cluster_size=8,
            view_degree=53.0,
            orbit_direction=-1,
            device="cpu",
        )

    def test_wrapper_script_exists_and_uses_module_main(self):
        script = Path(__file__).resolve().parents[1] / "omg4_ftgs_lkg_web_server.py"

        self.assertTrue(script.is_file())
        self.assertIn("lkg_experiment.omg4_ftgs.lkg_web_server", script.read_text(encoding="utf-8"))

    def test_pyproject_registers_console_script(self):
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

        self.assertEqual(
            data["project"]["scripts"]["lkg-omg4-ftgs-lkg-web-server"],
            "lkg_experiment.omg4_ftgs.lkg_web_server:main",
        )


if __name__ == "__main__":
    unittest.main()
