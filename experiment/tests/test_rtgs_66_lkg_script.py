import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RtgsSixtySixLkgScriptTest(unittest.TestCase):
    def test_dry_run_emits_lkg_only_command_for_each_rtgs_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_root = root / "result" / "RTGS"
            output_root = root / "generated" / "rtgs_coherent"
            viewpoint_index = root / "lut.npz"
            for scene in ("jumpingjacks", "flame_salmon"):
                checkpoint = model_root / scene / "checkpoints" / "chkpnt_best.pth"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"checkpoint")
            viewpoint_index.write_bytes(b"lut")

            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "MODEL_ROOT": str(model_root),
                    "OUTPUT_ROOT": str(output_root),
                    "PYTHON_BIN": sys.executable,
                    "RTGS_SCENES": "jumpingjacks flame_salmon",
                    "RUN_GROUP": "rtgs_lkg_test",
                    "VIEWPOINT_INDEX_PATH": str(viewpoint_index),
                }
            )

            proc = subprocess.run(
                ["bash", str(Path(__file__).resolve().parents[1] / "scripts" / "run_rtgs_66_lkg_all.sh")],
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("--render-mode views66", proc.stderr)
            self.assertIn("--no-write-per-view", proc.stderr)
            self.assertIn("--compare-original-views 0", proc.stderr)
            self.assertIn(f"--model-path {model_root / 'jumpingjacks'}", proc.stderr)
            self.assertIn(f"--model-path {model_root / 'flame_salmon'}", proc.stderr)
            self.assertIn(f"--output-dir {output_root / 'rtgs_lkg_test' / 'jumpingjacks'}", proc.stderr)
            self.assertIn(f"--output-dir {output_root / 'rtgs_lkg_test' / 'flame_salmon'}", proc.stderr)

    def test_views_script_emits_per_view_only_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_root = root / "result" / "RTGS"
            output_root = root / "generated" / "rtgs_coherent"
            viewpoint_index = root / "lut.npz"
            checkpoint = model_root / "jumpingjacks" / "checkpoints" / "chkpnt_best.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            viewpoint_index.write_bytes(b"lut")

            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "MODEL_ROOT": str(model_root),
                    "OUTPUT_ROOT": str(output_root),
                    "PYTHON_BIN": sys.executable,
                    "RTGS_SCENES": "jumpingjacks",
                    "RUN_GROUP": "rtgs_views_test",
                    "VIEWPOINT_INDEX_PATH": str(viewpoint_index),
                }
            )

            proc = subprocess.run(
                ["bash", str(Path(__file__).resolve().parents[1] / "scripts" / "run_rtgs_66_views_all.sh")],
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("--write-per-view", proc.stderr)
            self.assertIn("--no-write-interlaced", proc.stderr)
            self.assertNotIn("--no-write-per-view", proc.stderr)
            self.assertIn(f"--output-dir {output_root / 'rtgs_views_test' / 'jumpingjacks'}", proc.stderr)

    def test_single_view_all_cams_script_emits_one_command_per_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_root = root / "result" / "RTGS"
            output_root = root / "generated" / "rtgs_single_views"
            checkpoint = model_root / "coffee_martini" / "checkpoints" / "chkpnt_best.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")

            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "MODEL_ROOT": str(model_root),
                    "OUTPUT_ROOT": str(output_root),
                    "PYTHON_BIN": sys.executable,
                    "RTGS_SCENES": "coffee_martini",
                    "RUN_GROUP": "single_all_cams",
                }
            )

            proc = subprocess.run(
                ["bash", str(Path(__file__).resolve().parents[1] / "scripts" / "run_rtgs_single_views_all_cams.sh")],
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr.count("--render-mode single-all-cams"), 1)
            self.assertIn("--split all", proc.stderr)
            self.assertNotIn("--camera-index", proc.stderr)
            self.assertIn(f"--output-dir {output_root / 'single_all_cams' / 'coffee_martini'}", proc.stderr)

    def test_single_view_all_cams_script_can_emit_selected_n3dv_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_root = root / "result" / "RTGS"
            output_root = root / "generated" / "rtgs_single_views"
            checkpoint = model_root / "coffee_martini" / "checkpoints" / "chkpnt_best.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")

            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "MODEL_ROOT": str(model_root),
                    "OUTPUT_ROOT": str(output_root),
                    "PYTHON_BIN": sys.executable,
                    "RTGS_SCENES": "coffee_martini",
                    "RUN_GROUP": "single_frames",
                    "CAMERA_INDICES": "0",
                    "N3DV_FRAME_INDICES": "0 150",
                }
            )

            proc = subprocess.run(
                ["bash", str(Path(__file__).resolve().parents[1] / "scripts" / "run_rtgs_single_views_all_cams.sh")],
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr.count("--render-mode single-all-cams"), 2)
            self.assertIn("--n3dv-frame-index 0", proc.stderr)
            self.assertIn("--n3dv-frame-index 150", proc.stderr)
            self.assertIn("--camera-indices 0", proc.stderr)
            self.assertIn(f"--output-dir {output_root / 'single_frames' / 'coffee_martini' / 'frame_0000'}", proc.stderr)
            self.assertIn(f"--output-dir {output_root / 'single_frames' / 'coffee_martini' / 'frame_0150'}", proc.stderr)

    def test_script_prefers_configured_coherent_raster_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_root = root / "result" / "RTGS"
            viewpoint_index = root / "lut.npz"
            coherent_python = root / "coherent_python"
            checkpoint = model_root / "jumpingjacks" / "checkpoints" / "chkpnt_best.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            viewpoint_index.write_bytes(b"lut")
            coherent_python.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
            coherent_python.chmod(0o755)

            env = os.environ.copy()
            env.pop("PYTHON_BIN", None)
            env.update(
                {
                    "COHERENT_RASTER_PYTHON": str(coherent_python),
                    "DRY_RUN": "1",
                    "MODEL_ROOT": str(model_root),
                    "RTGS_SCENES": "jumpingjacks",
                    "VIEWPOINT_INDEX_PATH": str(viewpoint_index),
                }
            )

            proc = subprocess.run(
                ["bash", str(Path(__file__).resolve().parents[1] / "scripts" / "run_rtgs_66_lkg_all.sh")],
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn(f"DRY RUN: {coherent_python}", proc.stderr)


if __name__ == "__main__":
    unittest.main()
