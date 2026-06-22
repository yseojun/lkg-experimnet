import csv
import tempfile
import unittest
from pathlib import Path

from lkg_experiment.batch_experiments import (
    SUMMARY_COLUMNS,
    append_summary_rows,
    metrics_rows_from_run,
    status_row,
    write_summary_tsv,
)


class BatchExperimentsTest(unittest.TestCase):
    def test_metrics_rows_from_run_adds_dataset_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_a"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(
                '{"run_id": "run_a", "artifact_root": "/tmp/artifacts/run_a"}',
                encoding="utf-8",
            )
            with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "variant",
                        "camera_split",
                        "camera_index",
                        "group",
                        "cluster_size",
                        "use_remapping",
                        "reuse_enabled",
                        "fps",
                        "frame_ms",
                        "peak_vram_gb",
                        "psnr_mean",
                        "ssim_mean",
                        "lpips_mean",
                        "metric_view_count",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "variant": "cluster_8",
                        "camera_split": "test",
                        "camera_index": "16",
                        "group": "cluster_sweep",
                        "cluster_size": "8",
                        "use_remapping": "True",
                        "reuse_enabled": "True",
                        "fps": "10.0",
                        "frame_ms": "100.0",
                        "peak_vram_gb": "6.0",
                        "psnr_mean": "30.0",
                        "ssim_mean": "0.94",
                        "lpips_mean": "0.04",
                        "metric_view_count": "66",
                    }
                )
                writer.writerow(
                    {
                        "variant": "cluster_8",
                        "camera_split": "test",
                        "camera_index": "17",
                        "group": "cluster_sweep",
                        "cluster_size": "8",
                        "use_remapping": "True",
                        "reuse_enabled": "True",
                        "fps": "12.5",
                        "frame_ms": "80.0",
                        "peak_vram_gb": "6.25",
                        "psnr_mean": "31.25",
                        "ssim_mean": "0.95",
                        "lpips_mean": "0.03",
                        "metric_view_count": "66",
                    }
                )

            rows = metrics_rows_from_run(
                run_dir,
                suite="blender",
                result_group="blender_MCMC500000",
                scene="drums",
                camera_split="test",
                camera_index=17,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["suite"], "blender")
        self.assertEqual(rows[0]["result_group"], "blender_MCMC500000")
        self.assertEqual(rows[0]["scene"], "drums")
        self.assertEqual(rows[0]["camera_split"], "test")
        self.assertEqual(rows[0]["camera_index"], "17")
        self.assertEqual(rows[0]["run_id"], "run_a")
        self.assertEqual(rows[0]["variant"], "cluster_8")
        self.assertEqual(rows[0]["fps"], "12.5")
        self.assertEqual(rows[0]["artifact_root"], "/tmp/artifacts/run_a")
        self.assertTrue(rows[0]["metrics_csv"].endswith("run_a/metrics.csv"))

    def test_write_summary_tsv_is_excel_pasteable(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "summary.txt"
            row = status_row(
                status="missing",
                suite="mipnerf360",
                result_group="MipNeRF360_MCMC500000",
                scene="flowers",
                camera_split="val",
                camera_index=0,
                message="missing checkpoint",
            )

            write_summary_tsv([row], output)
            text = output.read_text(encoding="utf-8")

        lines = text.splitlines()
        self.assertEqual(lines[0].split("\t"), SUMMARY_COLUMNS)
        self.assertEqual(
            lines[1].split("\t")[0:7],
            ["missing", "mipnerf360", "MipNeRF360_MCMC500000", "flowers", "val", "0", ""],
        )
        self.assertIn("\tmissing checkpoint", lines[1])

    def test_append_summary_rows_writes_header_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "summary.txt"
            first = status_row(
                status="missing",
                suite="blender",
                result_group="group_a",
                scene="ship",
                camera_split="test",
                camera_index=0,
                message="a",
            )
            second = status_row(
                status="failed",
                suite="blender",
                result_group="group_a",
                scene="lego",
                camera_split="test",
                camera_index=1,
                message="b",
            )

            append_summary_rows([first], output)
            append_summary_rows([second], output)
            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(lines[0].split("\t"), SUMMARY_COLUMNS)
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[1].split("\t")[0], "missing")
        self.assertEqual(lines[2].split("\t")[0], "failed")
