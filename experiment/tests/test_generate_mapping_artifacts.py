import tempfile
import unittest
from pathlib import Path

import numpy as np

from lkg_experiment.generate_mapping_artifacts import generate_mapping_artifacts


class GenerateMappingArtifactsTest(unittest.TestCase):
    def test_generate_mapping_artifacts_writes_mappings_from_lut(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lut_path = root / "lut.npz"
            viewpoint_index = np.array(
                [
                    [[0, 1, 2], [3, 4, 5]],
                    [[6, 7, 8], [9, 10, 11]],
                ],
                dtype=np.int32,
            )
            np.savez_compressed(lut_path, viewpoint_index=viewpoint_index, view_count=np.array(12, dtype=np.int32))
            output_dir = root / "mapping_artifacts"

            written = generate_mapping_artifacts(
                viewpoint_index_path=lut_path,
                output_dir=output_dir,
                width=2,
                height=2,
                views=12,
                clusters=(2, 4),
                write_previews=False,
            )

            relative = {path.relative_to(output_dir).as_posix() for path in written}

        self.assertIn("mappings/raw_mapping.npz", relative)
        self.assertIn("mappings/view_index_values.uint16.bin", relative)
        self.assertFalse(any(path.endswith(".png") for path in relative))

