import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lkg_experiment.build_lut_npz import (
    DEFAULT_BRIDGE_SDK_ROOT,
    balanced_view_index_from_float,
    build_parser,
    fill_invalid_view_float,
    install_bridge_sdk_root,
    save_viewpoint_index_npz,
)


class BuildLutNpzTest(unittest.TestCase):
    def test_balanced_view_index_rank_maps_all_views(self):
        view_float = np.asarray(
            [
                [[0.0, 0.5, 1.0], [1.5, 2.0, 2.5]],
                [[3.0, 3.5, 4.0], [4.5, 5.0, 5.5]],
            ],
            dtype=np.float32,
        )

        viewpoint_index = balanced_view_index_from_float(view_float, view_count=3)

        self.assertEqual(viewpoint_index.dtype, np.int32)
        self.assertEqual(viewpoint_index.shape, view_float.shape)
        np.testing.assert_array_equal(
            np.bincount(viewpoint_index.reshape(-1), minlength=3),
            np.asarray([4, 4, 4]),
        )
        self.assertEqual(int(viewpoint_index.min()), 0)
        self.assertEqual(int(viewpoint_index.max()), 2)

    def test_fill_invalid_view_float_interpolates_missing_samples(self):
        view_float = np.asarray([[[0.0], [np.nan], [np.nan], [np.nan], [4.0]]], dtype=np.float32)
        valid = np.isfinite(view_float)

        filled = fill_invalid_view_float(view_float, valid)

        np.testing.assert_allclose(filled[:, :, 0], np.asarray([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
        self.assertFalse(np.isnan(filled).any())

    def test_save_viewpoint_index_npz_matches_experiment_loader_metadata(self):
        viewpoint_index = np.arange(2 * 3 * 3, dtype=np.int32).reshape(2, 3, 3) % 6

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lut.npz"
            save_viewpoint_index_npz(
                path,
                viewpoint_index,
                view_count=6,
                quilt_cols=3,
                quilt_rows=2,
                layout="gl-top",
                index_method="balanced-ramp",
                extra_metadata={
                    "normalized_valid_count": np.asarray(18, dtype=np.int64),
                    "balanced_rank_remap": np.asarray(True),
                },
            )

            with np.load(path) as loaded:
                self.assertIn("viewpoint_index", loaded)
                self.assertIn("view_count", loaded)
                self.assertIn("quilt_cols", loaded)
                self.assertIn("quilt_rows", loaded)
                self.assertIn("layout", loaded)
                self.assertIn("width", loaded)
                self.assertIn("height", loaded)
                self.assertIn("index_method", loaded)
                np.testing.assert_array_equal(loaded["viewpoint_index"], viewpoint_index)
                self.assertEqual(int(loaded["view_count"]), 6)
                self.assertEqual(int(loaded["quilt_cols"]), 3)
                self.assertEqual(int(loaded["quilt_rows"]), 2)
                self.assertEqual(str(loaded["layout"]), "gl-top")
                self.assertEqual(int(loaded["width"]), 3)
                self.assertEqual(int(loaded["height"]), 2)
                self.assertEqual(str(loaded["index_method"]), "balanced-ramp")
                self.assertEqual(int(loaded["normalized_valid_count"]), 18)
                self.assertTrue(bool(loaded["balanced_rank_remap"]))

    def test_parser_defaults_to_balanced_ramp(self):
        args = build_parser().parse_args(["--output", "/tmp/lut.npz", "--layout", "gl-top"])

        self.assertEqual(args.output, "/tmp/lut.npz")
        self.assertEqual(args.index_method, "balanced-ramp")
        self.assertEqual(args.layout, "gl-top")

    def test_install_bridge_sdk_root_supports_sdk_absolute_imports(self):
        root = install_bridge_sdk_root(DEFAULT_BRIDGE_SDK_ROOT)

        self.assertIn(str(root / "src"), sys.path)
        self.assertIn(str(root / "src/bridge_python_sdk"), sys.path)


if __name__ == "__main__":
    unittest.main()
