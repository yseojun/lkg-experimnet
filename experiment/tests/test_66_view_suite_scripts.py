import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SixtySixViewSuiteScriptsTest(unittest.TestCase):
    def setUp(self):
        self.experiment_root = Path(__file__).resolve().parents[1]

    def test_blender_script_defaults_to_100000_gaussian_init50000_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_base = root / "dataset"
            result_base = root / "result"
            artifact_dir = root / "artifacts"
            viewpoint_index = root / "lut.npz"
            summary = root / "summary.txt"
            data_dir = dataset_base / "nerf_synthetic" / "drums"
            checkpoint = (
                result_base
                / "blender_MCMC100000_init50000"
                / "drums"
                / "ckpts"
                / "ckpt_29999_rank0.pt"
            )
            data_dir.mkdir(parents=True)
            checkpoint.parent.mkdir(parents=True)
            viewpoint_index.write_bytes(b"viewpoint index")
            checkpoint.write_bytes(b"checkpoint")

            env = self._base_env(root)
            env.update(
                {
                    "ARTIFACT_DIR": str(artifact_dir),
                    "BLENDER_SCENES": "drums",
                    "DATADIR": str(dataset_base),
                    "RESULTDIR": str(result_base),
                    "RUN_GROUP": "split_test",
                    "SUMMARY_TXT": str(summary),
                    "VIEWPOINT_INDEX_PATH": str(viewpoint_index),
                }
            )

            proc = subprocess.run(
                ["bash", str(self.experiment_root / "scripts" / "run_blender_66_views.sh")],
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("blender_MCMC100000_init50000/drums/ckpts/ckpt_29999_rank0.pt", proc.stderr)
            self.assertNotIn("MipNeRF360_MCMC500000", proc.stderr)
            self.assertIn("split_test_blender_MCMC100000_init50000_drums", summary.read_text(encoding="utf-8"))

    def test_mipnerf360_script_runs_only_mipnerf360_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_base = root / "dataset"
            result_base = root / "result"
            artifact_dir = root / "artifacts"
            viewpoint_index = root / "lut.npz"
            summary = root / "summary.txt"
            data_dir = dataset_base / "MipNeRF_360" / "garden"
            checkpoint = (
                result_base
                / "MipNeRF360_MCMC500000"
                / "garden"
                / "ckpts"
                / "ckpt_29999_rank0.pt"
            )
            data_dir.mkdir(parents=True)
            checkpoint.parent.mkdir(parents=True)
            viewpoint_index.write_bytes(b"viewpoint index")
            checkpoint.write_bytes(b"checkpoint")

            env = self._base_env(root)
            env.update(
                {
                    "ARTIFACT_DIR": str(artifact_dir),
                    "DATADIR": str(dataset_base),
                    "MIPNERF360_SCENES": "garden",
                    "RESULTDIR": str(result_base),
                    "RUN_GROUP": "split_test",
                    "SUMMARY_TXT": str(summary),
                    "VIEWPOINT_INDEX_PATH": str(viewpoint_index),
                }
            )

            proc = subprocess.run(
                ["bash", str(self.experiment_root / "scripts" / "run_mipnerf360_66_views.sh")],
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("MipNeRF360_MCMC500000/garden/ckpts/ckpt_29999_rank0.pt", proc.stderr)
            self.assertNotIn("blender_MCMC", proc.stderr)
            self.assertIn("split_test_MipNeRF360_MCMC500000_garden", summary.read_text(encoding="utf-8"))

    def _base_env(self, root: Path) -> dict[str, str]:
        env = os.environ.copy()
        for key in (
            "BLENDER_RESULT_GROUPS",
            "CAMERA_INDICES",
            "CAMERA_INDEXES",
            "GENERATED_DIR",
            "LKG_GENERATED_DIR",
            "MIPNERF360_RESULT_GROUPS",
        ):
            env.pop(key, None)
        env.update(
            {
                "CAMERA_INDEX": "0",
                "DRY_RUN": "1",
                "PYTHON_BIN": sys.executable,
            }
        )
        return env


if __name__ == "__main__":
    unittest.main()
