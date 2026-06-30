from __future__ import annotations

import json
import tempfile
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from lkg_experiment.omg4_ftgs import camera, cli, model


class _FakeDynamicModel:
    def materialize(self, timestamp: float):
        return SimpleNamespace(timestamp=float(timestamp))


class _FakeCameraSet:
    K = np.eye(3, dtype=np.float32)
    width = 3
    height = 2

    def viewmat_at(self, camera_index: int):
        return np.eye(4, dtype=np.float32)


class Omg4FtgsTest(unittest.TestCase):
    def test_default_cook_spinach_paths(self):
        self.assertEqual(
            cli.DEFAULT_CHECKPOINT_PATH,
            Path("/data/ysj/result/4dgs/OMG4-FTGS_weights/ours_L_weight/cook_spinach.xz"),
        )
        self.assertEqual(cli.DEFAULT_DATA_PATH, Path("/data/ysj/dataset/N3DV/cook_spinach"))

    def test_parser_defaults_to_both_mode(self):
        args = cli.build_parser().parse_args([])

        self.assertEqual(args.mode, "both")
        self.assertEqual(args.camera_index, 0)
        self.assertEqual(args.frame_index, 0)

    def test_load_test_timestamps_from_real_dataset(self):
        frames = camera.load_test_frames(cli.DEFAULT_DATA_PATH, camera_index=0)

        self.assertGreater(len(frames), 0)
        self.assertIsInstance(frames[0].timestamp, float)
        self.assertGreaterEqual(frames[0].index, 0)

    def test_load_pose_camera_shapes_from_real_dataset(self):
        camera_set = camera.load_pose_camera(cli.DEFAULT_DATA_PATH, camera_index=0, resolution=2.0)

        self.assertEqual(camera_set.viewmats.ndim, 3)
        self.assertEqual(camera_set.viewmats.shape[-2:], (4, 4))
        self.assertEqual(camera_set.K.shape, (3, 3))
        self.assertGreater(camera_set.width, 0)
        self.assertGreater(camera_set.height, 0)

    @unittest.skipUnless(importlib.util.find_spec("dahuffman"), "dahuffman is required to unpickle OMG4-FTGS checkpoints")
    def test_default_checkpoint_has_expected_encoded_keys(self):
        keys = model.peek_checkpoint_keys(cli.DEFAULT_CHECKPOINT_PATH)

        self.assertIn("means", keys)
        self.assertIn("MLP_cont", keys)
        self.assertIn("scale_code", keys)

    def test_single_mode_writes_png_and_manifest_with_mock_renderer(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            args = cli.build_parser().parse_args(["--mode", "single", "--output-dir", str(output_dir)])
            fake_model = _FakeDynamicModel()
            fake_camera = _FakeCameraSet()

            with mock.patch.object(cli, "load_dynamic_gaussians", return_value=fake_model), mock.patch.object(
                cli, "load_pose_camera", return_value=fake_camera
            ), mock.patch.object(cli, "load_test_frames", return_value=[camera.TestFrame(0, None, 0.0)]), mock.patch.object(
                cli, "render_splats_gsplat", return_value=torch.full((3, 2, 3), 0.5)
            ):
                result = cli.run(args)

            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "single.png").is_file())
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "single")
            self.assertEqual(manifest["single"]["timestamp"], 0.0)

    def test_video_mode_writes_frames_and_manifest_with_mock_renderer(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            args = cli.build_parser().parse_args(["--mode", "video", "--output-dir", str(output_dir), "--fps", "12"])
            fake_model = _FakeDynamicModel()
            fake_camera = _FakeCameraSet()
            frames = [
                camera.TestFrame(0, None, 0.0),
                camera.TestFrame(1, None, 0.5),
                camera.TestFrame(2, None, 1.0),
            ]

            with mock.patch.object(cli, "load_dynamic_gaussians", return_value=fake_model), mock.patch.object(
                cli, "load_pose_camera", return_value=fake_camera
            ), mock.patch.object(cli, "load_test_frames", return_value=frames), mock.patch.object(
                cli, "render_splats_gsplat", return_value=torch.full((3, 2, 3), 0.25)
            ), mock.patch.object(
                cli, "_encode_video_from_frames", return_value={"written": False, "path": None, "error": "mock"}
            ):
                result = cli.run(args)

            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "frames" / "frame_0000.png").is_file())
            self.assertTrue((output_dir / "frames" / "frame_0001.png").is_file())
            self.assertTrue((output_dir / "frames" / "frame_0002.png").is_file())
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "video")
            self.assertEqual(manifest["video"]["frame_count"], 3)
            self.assertEqual(manifest["video"]["mp4"]["error"], "mock")


if __name__ == "__main__":
    unittest.main()
