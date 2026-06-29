from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from lkg_experiment.rtgs_coherent import cr_experiment


class RtgsCrExperimentTest(unittest.TestCase):
    def test_parser_defaults_match_rtgs_experiment_layout(self):
        args = cr_experiment.build_parser().parse_args(["--dataset-kind", "dnerf"])

        self.assertEqual(args.artifact_dir, "/data/ysj/result/coherent-raster/generated/rtgs_cr_experiments")
        self.assertEqual(args.engine, "compose")
        self.assertEqual(args.clusters, "2,4,8,16")
        self.assertFalse(args.compose_repeat_variants)
        self.assertEqual(args.progress_every_views, 5)
        self.assertEqual(args.warmup_iters, 3)
        self.assertEqual(args.measure_iters, 5)
        self.assertEqual(args.max_metric_views, 5)
        self.assertEqual(args.width, 1440)
        self.assertEqual(args.height, 2560)
        self.assertEqual(args.views, 66)

    def test_run_id_includes_scene_engine_and_camera(self):
        run_id = cr_experiment.make_run_id(
            model_path=Path("/data/ysj/result/4dgs/RTGS/jumpingjacks"),
            engine="compose",
            split="test",
            camera_index=2,
            n3dv_frame_index=17,
            suffix="20260629_120000",
        )

        self.assertEqual(run_id, "jumpingjacks_compose_test_cam002_frame0017_20260629_120000")

    def test_build_compose_metric_row_uses_existing_schema(self):
        variant = SimpleNamespace(
            name="cluster_4",
            group="cluster_sweep",
            cluster_size=4,
            use_remapping=True,
            reuse_enabled=True,
        )
        timing = cr_experiment.TimingStats(fps=12.5, frame_ms=80.0, peak_vram_gb=3.25)
        metrics = cr_experiment.MetricStats(
            psnr_mean=31.0,
            psnr_std=0.5,
            ssim_mean=0.9,
            ssim_std=0.01,
            lpips_mean=float("nan"),
            lpips_std=float("nan"),
            metric_view_count=2,
        )

        row = cr_experiment.build_metric_row(
            variant=variant,
            engine="compose",
            camera_split="test",
            camera_index=0,
            output_prefix="frame_0000",
            timing=timing,
            metrics=metrics,
        )

        self.assertEqual(row["variant"], "cluster_4")
        self.assertEqual(row["engine"], "compose")
        self.assertEqual(row["cluster_size"], 4)
        self.assertAlmostEqual(row["fps"], 12.5)
        self.assertAlmostEqual(row["peak_vram_gb"], 3.25)
        self.assertAlmostEqual(row["psnr_mean"], 31.0)
        self.assertEqual(row["metric_view_count"], 2)

    def test_resolve_experiment_variants_compose_defaults_to_single_non_cluster_row(self):
        args = cr_experiment.build_parser().parse_args(["--dataset-kind", "dnerf"])

        variants = cr_experiment.resolve_experiment_variants(args)

        self.assertEqual(
            [(variant.name, variant.group, variant.cluster_size, variant.use_remapping, variant.reuse_enabled) for variant in variants],
            [("compose", "compose", 1, True, False)],
        )

    def test_resolve_experiment_variants_clustered_keeps_cluster_sweep(self):
        args = cr_experiment.build_parser().parse_args(
            [
                "--dataset-kind",
                "dnerf",
                "--engine",
                "clustered",
                "--clusters",
                "2,4",
                "--no-without-remap",
                "--no-without-reuse",
            ]
        )

        variants = cr_experiment.resolve_experiment_variants(args)

        self.assertEqual([(variant.name, variant.cluster_size) for variant in variants], [("cluster_2", 2), ("cluster_4", 4)])

    def test_time_compose_renderer_measures_average_and_peak_vram(self):
        calls = []

        def fake_render(_context):
            calls.append(len(calls))
            return torch.zeros((3, 2, 2), dtype=torch.float32), 7.0

        with mock.patch.object(cr_experiment, "_cuda_available", return_value=False):
            image, timing = cr_experiment.time_compose_renderer(
                SimpleNamespace(),
                render_fn=fake_render,
                warmup_iters=1,
                measure_iters=2,
            )

        self.assertEqual(len(calls), 3)
        self.assertEqual(tuple(image.shape), (3, 2, 2))
        self.assertAlmostEqual(timing.frame_ms, 7.0)
        self.assertAlmostEqual(timing.fps, 1000.0 / 7.0)
        self.assertEqual(timing.peak_vram_gb, 0.0)

    def test_time_compose_renderer_reports_iteration_progress(self):
        def fake_render(_context, *, progress_label=None):
            self.assertIsNotNone(progress_label)
            return torch.zeros((3, 2, 2), dtype=torch.float32), 10.0

        stream = io.StringIO()
        with mock.patch.object(cr_experiment, "_cuda_available", return_value=False), mock.patch("sys.stderr", stream):
            cr_experiment.time_compose_renderer(
                SimpleNamespace(),
                render_fn=fake_render,
                warmup_iters=1,
                measure_iters=1,
                progress_label="compose/cluster_2",
            )

        output = stream.getvalue()
        self.assertIn("compose/cluster_2 timing 1/2 warmup", output)
        self.assertIn("compose/cluster_2 timing 2/2 measure", output)

    def test_render_clustered_interlaced_once_delegates_to_one_shot_module(self):
        context = SimpleNamespace(name="context")
        variant = SimpleNamespace(name="cluster_2")
        expected = (torch.zeros((3, 2, 2), dtype=torch.float32), 3.5)

        with mock.patch(
            "lkg_experiment.rtgs_coherent.cr_one_shot.render_rtgs_cr_one_shot_interlaced_once",
            return_value=expected,
        ) as render:
            actual = cr_experiment.render_clustered_interlaced_once(context, variant=variant)

        self.assertIs(actual, expected)
        render.assert_called_once_with(context, variant=variant)

    def test_clustered_engine_allows_rtgs_projection_adapter_context(self):
        context = SimpleNamespace(
            fov_normalization={"applied": False},
            cr_anchor_camera_cuda=SimpleNamespace(FoVx=-1.0, FoVy=-1.0, fl_x=730.0, fl_y=730.0),
            args=SimpleNamespace(rtgs_compat_projection=True),
            viewport=SimpleNamespace(offset_x=0, offset_y=0, render_width=80, render_height=60, panel_width=80, panel_height=60),
        )

        cr_experiment.ensure_clustered_engine_supported(context)

    def test_clustered_engine_allows_letterboxed_viewport_for_positive_fov(self):
        context = SimpleNamespace(
            fov_normalization={"applied": False},
            cr_anchor_camera_cuda=SimpleNamespace(FoVx=0.7, FoVy=0.7, fl_x=-1.0, fl_y=-1.0),
            args=SimpleNamespace(rtgs_compat_projection=True),
            viewport=SimpleNamespace(offset_x=0, offset_y=560, render_width=1440, render_height=1440, panel_width=1440, panel_height=2560),
        )

        cr_experiment.ensure_clustered_engine_supported(context)

    def test_main_dispatches_to_runner(self):
        with mock.patch.object(cr_experiment, "run_rtgs_cr_experiment", return_value=0) as run:
            rc = cr_experiment.main(["--dataset-kind", "dnerf"])

        self.assertEqual(rc, 0)
        run.assert_called_once()

    def test_wrapper_script_exists_and_uses_module_main(self):
        wrapper = Path(__file__).resolve().parents[1] / "rtgs_cr_experiment.py"

        self.assertTrue(wrapper.is_file())
        self.assertIn("lkg_experiment.rtgs_coherent.cr_experiment", wrapper.read_text(encoding="utf-8"))

    def test_pyproject_registers_console_script(self):
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"

        self.assertIn(
            'lkg-rtgs-cr-experiment = "lkg_experiment.rtgs_coherent.cr_experiment:main"',
            pyproject.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
