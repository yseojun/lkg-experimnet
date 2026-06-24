import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class RunAll66ViewsScriptTest(unittest.TestCase):
    def setUp(self):
        self.experiment_root = Path(__file__).resolve().parents[1]
        self.script = self.experiment_root / "scripts" / "run_all_66_views.sh"

    def test_defaults_store_generated_outputs_under_result_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_base = root / "dataset"
            result_base = root / "result"
            data_dir = dataset_base / "nerf_synthetic" / "chair"
            checkpoint = (
                result_base
                / "coherent-raster"
                / "blender_MCMC500000"
                / "chair"
                / "ckpts"
                / "ckpt_29999_rank0.pt"
            )
            viewpoint_index = result_base / "generated" / "lkg_go_1440x2560_66_views_lkg_calibration.npz"
            summary = root / "summary.txt"
            expected_artifact_dir = result_base / "generated" / "coherent_raster_experiments"
            expected_run_dir = expected_artifact_dir / "path_test_blender_MCMC500000_chair"

            data_dir.mkdir(parents=True)
            checkpoint.parent.mkdir(parents=True)
            viewpoint_index.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            viewpoint_index.write_bytes(b"viewpoint index")

            env = os.environ.copy()
            for key in (
                "ARTIFACT_DIR",
                "DATADIR",
                "GENERATED_DIR",
                "LKG_GENERATED_DIR",
                "RESULTDIR",
                "RESULT_DIR_ROOT",
                "VIEWPOINT_INDEX_PATH",
            ):
                env.pop(key, None)
            env.update(
                {
                    "CAMERA_INDEX": "0",
                    "DATADIR": str(dataset_base),
                    "DRY_RUN": "1",
                    "LKG_RESULT_BASE": str(result_base),
                    "MIPNERF360_RESULT_GROUPS": "unused",
                    "MIPNERF360_SCENES": " ",
                    "RUN_GROUP": "path_test",
                    "SUMMARY_TXT": str(summary),
                    "BLENDER_RESULT_GROUPS": "blender_MCMC500000",
                    "BLENDER_SCENES": "chair",
                }
            )

            proc = subprocess.run(
                ["bash", str(self.script)],
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn(f"-> {expected_run_dir} (test_000000)", proc.stderr)
            self.assertIn(f"dry run: {expected_run_dir}/test_000000", summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
