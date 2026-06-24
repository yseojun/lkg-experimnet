import unittest
from pathlib import Path
from argparse import Namespace
import tempfile

import numpy as np

from lkg_experiment.render_4dgs_single_view import (
    build_parser,
    default_output_path,
    load_4dgs_dynerf_camera,
)


class Render4DGSSingleViewTest(unittest.TestCase):
    def test_parser_accepts_4dgs_dataset_camera_and_output_args(self):
        args = build_parser().parse_args(
            [
                "--four-dgs-model-path",
                "/tmp/coffee_martini",
                "--four-dgs-code-root",
                "/tmp/4DGaussians",
                "--four-dgs-iteration",
                "14000",
                "--four-dgs-time",
                "0.5",
                "--data-dir",
                "/tmp/N3DV/coffee_martini/colmap",
                "--camera-source",
                "dataset",
                "--camera-split",
                "val",
                "--camera-index",
                "1",
                "--width",
                "640",
                "--height",
                "480",
                "--output",
                "/tmp/render.png",
            ]
        )

        self.assertEqual(args.four_dgs_model_path, "/tmp/coffee_martini")
        self.assertEqual(args.four_dgs_code_root, "/tmp/4DGaussians")
        self.assertEqual(args.four_dgs_iteration, 14000)
        self.assertEqual(args.four_dgs_time, 0.5)
        self.assertEqual(args.data_dir, "/tmp/N3DV/coffee_martini/colmap")
        self.assertEqual(args.camera_source, "dataset")
        self.assertEqual(args.camera_split, "val")
        self.assertEqual(args.camera_index, 1)
        self.assertEqual((args.width, args.height), (640, 480))
        self.assertEqual(args.output, "/tmp/render.png")

    def test_parser_defaults_to_4dgs_training_camera_source(self):
        args = build_parser().parse_args(["--four-dgs-model-path", "/tmp/coffee_martini"])

        self.assertEqual(args.camera_source, "fourdgs")

    def test_default_output_path_includes_iteration_time_and_camera(self):
        path = default_output_path(
            Path("/data/ysj/result/4dgs/dynerf/coffee_martini"),
            iteration=14000,
            time_value=0.25,
            camera_split="val",
            camera_index=2,
        )

        self.assertEqual(path.name, "coffee_martini_iter14000_t0.250000_val_2.png")

    def test_4dgs_dynerf_camera_uses_held_out_camera_for_val_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp)
            np.save(source_path / "poses_bounds.npy", _poses_bounds_fixture(camera_count=2))
            args = Namespace(camera_split="val", camera_index=0, no_crop_to_fill=False)

            _c2w, K, _center, label = load_4dgs_dynerf_camera(
                Namespace(source_path=str(source_path)),
                args,
                width=1352,
                height=1014,
            )

        self.assertIn(":val[0]/cam00", label)
        np.testing.assert_allclose(
            K,
            np.array([[730.377, 0.0, 676.0], [0.0, 730.377, 507.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_4dgs_dynerf_camera_train_split_skips_eval_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp)
            np.save(source_path / "poses_bounds.npy", _poses_bounds_fixture(camera_count=3))
            args = Namespace(camera_split="train", camera_index=0, no_crop_to_fill=False)

            _c2w, _K, _center, label = load_4dgs_dynerf_camera(
                Namespace(source_path=str(source_path)),
                args,
                width=1352,
                height=1014,
            )

        self.assertIn(":train[0]/cam01", label)


def _poses_bounds_fixture(camera_count: int) -> np.ndarray:
    rows = []
    for index in range(camera_count):
        pose = np.zeros((3, 5), dtype=np.float64)
        pose[:3, :3] = np.eye(3)
        pose[:3, 3] = np.array([float(index), 0.0, 0.0])
        pose[:, 4] = np.array([2028.0, 2704.0, 1460.754])
        rows.append(np.concatenate([pose.reshape(-1), np.array([0.1, 10.0])]))
    return np.stack(rows, axis=0)


if __name__ == "__main__":
    unittest.main()
