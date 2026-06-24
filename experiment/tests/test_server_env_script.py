import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class ServerEnvScriptTest(unittest.TestCase):
    def setUp(self):
        self.experiment_root = Path(__file__).resolve().parents[1]
        self.script = self.experiment_root / "scripts" / "set_server_dirs.sh"

    def test_source_finds_data_and_result_roots_under_server_bases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_base = root / "dataset"
            result_base = root / "result"
            data_dir = dataset_base / "nerf_synthetic" / "drums"
            checkpoint = (
                result_base
                / "coherent-raster"
                / "blender_MCMC100000_init50000"
                / "drums"
                / "ckpts"
                / "ckpt_29999_rank0.pt"
            )
            data_dir.mkdir(parents=True)
            (data_dir / "transforms_train.json").write_text("{}", encoding="utf-8")
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"")

            env = os.environ.copy()
            for key in (
                "DATADIR",
                "RESULTDIR",
                "datadir",
                "resultdir",
                "DATA_DIR_ROOT",
                "RESULT_DIR_ROOT",
                "DATA_DIR",
                "CHECKPOINT_PATH",
            ):
                env.pop(key, None)
            env["LKG_DATASET_BASE"] = str(dataset_base)
            env["LKG_RESULT_BASE"] = str(result_base)

            proc = subprocess.run(
                [
                    "bash",
                    "-c",
                    (
                        "set -euo pipefail; "
                        f"source {self.script}; "
                        'printf "%s\\n%s\\n%s\\n%s\\n%s\\n%s\\n%s\\n%s\\n" '
                        '"$DATADIR" "$RESULTDIR" "$datadir" "$resultdir" '
                        '"$DATA_DIR_ROOT" "$RESULT_DIR_ROOT" '
                        '"$DATA_DIR" "$CHECKPOINT_PATH"; '
                        'test -d "$DATA_DIR"; '
                        'test -f "$CHECKPOINT_PATH"'
                    ),
                ],
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )

        self.assertEqual(
            proc.stdout.splitlines(),
            [
                str(dataset_base),
                str(result_base / "coherent-raster"),
                str(dataset_base),
                str(result_base / "coherent-raster"),
                str(dataset_base),
                str(result_base / "coherent-raster"),
                str(data_dir),
                str(checkpoint),
            ],
        )

    def test_source_resets_stale_home_defaults_to_server_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home" / "ysj"
            dataset_base = root / "dataset"
            result_base = root / "result"
            data_dir = dataset_base / "nerf_synthetic" / "drums"
            checkpoint = (
                result_base
                / "coherent-raster"
                / "blender_MCMC100000_init50000"
                / "drums"
                / "ckpts"
                / "ckpt_29999_rank0.pt"
            )
            data_dir.mkdir(parents=True)
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"")

            env = os.environ.copy()
            env["HOME"] = str(home)
            env["LKG_DATASET_BASE"] = str(dataset_base)
            env["LKG_RESULT_BASE"] = str(result_base)
            env["DATADIR"] = str(home / "Data" / "datasets")
            env["RESULTDIR"] = str(home / "Data" / "results")
            env["datadir"] = env["DATADIR"]
            env["resultdir"] = env["RESULTDIR"]
            env["DATA_DIR_ROOT"] = env["DATADIR"]
            env["RESULT_DIR_ROOT"] = env["RESULTDIR"]
            env["DATA_DIR"] = str(home / "Data" / "datasets" / "nerf_synthetic" / "drums")
            env["CHECKPOINT_PATH"] = str(
                home
                / "Data"
                / "results"
                / "blender_MCMC100000_init50000"
                / "drums"
                / "ckpts"
                / "ckpt_29999_rank0.pt"
            )

            proc = subprocess.run(
                [
                    "bash",
                    "-c",
                    (
                        "set -euo pipefail; "
                        f"source {self.script}; "
                        'printf "%s\\n%s\\n%s\\n%s\\n" '
                        '"$DATADIR" "$RESULTDIR" "$DATA_DIR" "$CHECKPOINT_PATH"'
                    ),
                ],
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )

        self.assertEqual(
            proc.stdout.splitlines(),
            [
                str(dataset_base),
                str(result_base / "coherent-raster"),
                str(data_dir),
                str(checkpoint),
            ],
        )

    def test_executing_script_prints_exports_that_can_be_evaled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_base = root / "dataset"
            result_base = root / "result"
            data_dir = dataset_base / "nerf_synthetic" / "drums"
            checkpoint = (
                result_base
                / "coherent-raster"
                / "blender_MCMC100000_init50000"
                / "drums"
                / "ckpts"
                / "ckpt_29999_rank0.pt"
            )
            data_dir.mkdir(parents=True)
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"")

            env = os.environ.copy()
            for key in ("DATADIR", "RESULTDIR", "DATA_DIR", "CHECKPOINT_PATH"):
                env.pop(key, None)
            env["LKG_DATASET_BASE"] = str(dataset_base)
            env["LKG_RESULT_BASE"] = str(result_base)

            proc = subprocess.run(
                [
                    "bash",
                    "-c",
                    (
                        "set -euo pipefail; "
                        f'eval "$({self.script})"; '
                        'printf "%s\\n%s\\n%s\\n%s\\n" '
                        '"$DATADIR" "$RESULTDIR" "$DATA_DIR" "$CHECKPOINT_PATH"'
                    ),
                ],
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )

        self.assertEqual(
            proc.stdout.splitlines(),
            [
                str(dataset_base),
                str(result_base / "coherent-raster"),
                str(data_dir),
                str(checkpoint),
            ],
        )
