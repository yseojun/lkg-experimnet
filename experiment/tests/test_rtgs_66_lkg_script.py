import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RtgsSixtySixLkgScriptTest(unittest.TestCase):
    def test_cr_experiment_script_emits_experiment_command_for_each_rtgs_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_root = root / "result" / "RTGS"
            artifact_root = root / "generated" / "rtgs_cr_experiments"
            viewpoint_index = root / "lut.npz"
            for scene in ("jumpingjacks", "coffee_martini"):
                checkpoint = model_root / scene / "checkpoints" / "chkpnt_best.pth"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"checkpoint")
            viewpoint_index.write_bytes(b"lut")

            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "GENERATED_ROOT": str(root / "generated"),
                    "MODEL_ROOT": str(model_root),
                    "ARTIFACT_ROOT": str(artifact_root),
                    "PYTHON_BIN": sys.executable,
                    "RTGS_SCENES": "jumpingjacks coffee_martini",
                    "RUN_GROUP": "rtgs_cr_exp_test",
                    "VIEWPOINT_INDEX_PATH": str(viewpoint_index),
                    "CLUSTERS": "2,4",
                    "WARMUP_ITERS": "0",
                    "MEASURE_ITERS": "1",
                    "MAX_METRIC_VIEWS": "1",
                }
            )

            proc = subprocess.run(
                ["bash", str(Path(__file__).resolve().parents[1] / "scripts" / "run_rtgs_cr_experiments_all.sh")],
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("rtgs_cr_experiment.py", proc.stderr)
            self.assertIn("--engine compose", proc.stderr)
            self.assertIn(f"--torch-extensions-dir {root / 'generated' / 'torch_extensions_lkg_rtgs' / 'rtgs_official'}", proc.stderr)
            self.assertIn("--clusters 2\\,4", proc.stderr)
            self.assertIn("--warmup-iters 0", proc.stderr)
            self.assertIn("--measure-iters 1", proc.stderr)
            self.assertIn("--max-metric-views 1", proc.stderr)
            self.assertIn(f"--model-path {model_root / 'jumpingjacks'}", proc.stderr)
            self.assertIn(f"--model-path {model_root / 'coffee_martini'}", proc.stderr)
            self.assertIn(f"--artifact-dir {artifact_root / 'rtgs_cr_exp_test'}", proc.stderr)

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

    def test_1view_video_script_emits_render_frames_and_ffmpeg_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_root = root / "result" / "RTGS"
            output_root = root / "generated" / "rtgs_official_1view_videos"
            dnerf_root = root / "dataset" / "dnerf"
            n3dv_root = root / "dataset" / "N3DV"

            for scene in ("jumpingjacks", "coffee_martini"):
                checkpoint = model_root / scene / "checkpoints" / "chkpnt_best.pth"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"checkpoint")

            dnerf_scene = dnerf_root / "jumpingjacks"
            dnerf_scene.mkdir(parents=True)
            (dnerf_scene / "transforms_test.json").write_text(
                '{"frames": [{"file_path": "r_000"}, {"file_path": "r_001"}]}',
                encoding="utf-8",
            )
            n3dv_images = n3dv_root / "coffee_martini" / "cam00" / "images"
            n3dv_images.mkdir(parents=True)
            (n3dv_images / "0000.png").write_bytes(b"png")
            (n3dv_images / "0001.png").write_bytes(b"png")

            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "MODEL_ROOT": str(model_root),
                    "OUTPUT_ROOT": str(output_root),
                    "PYTHON_BIN": sys.executable,
                    "DNERF_ROOT": str(dnerf_root),
                    "N3DV_ROOT": str(n3dv_root),
                    "DNERF_SCENES": "jumpingjacks",
                    "N3DV_SCENES": "coffee_martini",
                    "RUN_GROUP": "video_test",
                    "FPS": "12",
                }
            )

            proc = subprocess.run(
                ["bash", str(Path(__file__).resolve().parents[1] / "scripts" / "run_rtgs_1view_videos_all.sh")],
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("rtgs_official_1view.py", proc.stderr)
            self.assertIn("--dataset-kind dnerf", proc.stderr)
            self.assertIn("--dataset-kind n3dv", proc.stderr)
            self.assertIn("--camera-index 0", proc.stderr)
            self.assertIn("--camera-index 1", proc.stderr)
            self.assertIn("--n3dv-frame-index 0", proc.stderr)
            self.assertIn("--n3dv-frame-index 1", proc.stderr)
            self.assertIn("ffmpeg", proc.stderr)
            self.assertIn("-framerate 12", proc.stderr)
            self.assertIn(str(output_root / "video_test" / "jumpingjacks" / "jumpingjacks_1view.mp4"), proc.stderr)
            self.assertIn(str(output_root / "video_test" / "coffee_martini" / "coffee_martini_1view.mp4"), proc.stderr)


if __name__ == "__main__":
    unittest.main()
