import json
import tempfile
import unittest
from pathlib import Path

from lkg_experiment.dataset_views import (
    count_blender_cameras,
    split_indices_for_colmap_count,
)


class DatasetViewsTest(unittest.TestCase):
    def test_count_blender_cameras_reads_requested_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "drums"
            data_dir.mkdir()
            (data_dir / "transforms_test.json").write_text(
                json.dumps({"frames": [{"file_path": "a"}, {"file_path": "b"}, {"file_path": "c"}]}),
                encoding="utf-8",
            )

            count = count_blender_cameras(data_dir, "test")

        self.assertEqual(count, 3)

    def test_colmap_split_indices_match_experiment_loader_policy(self):
        self.assertEqual(split_indices_for_colmap_count(10, "val", test_every=4), [0, 4, 8])
        self.assertEqual(split_indices_for_colmap_count(10, "train", test_every=4), [1, 2, 3, 5, 6, 7, 9])
        self.assertEqual(split_indices_for_colmap_count(10, "auto", test_every=4), [0, 4, 8])

    def test_colmap_test_split_is_rejected(self):
        with self.assertRaises(ValueError):
            split_indices_for_colmap_count(10, "test", test_every=8)

