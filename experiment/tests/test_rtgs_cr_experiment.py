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
        self.assertEqual(args.engine, "one_shot")
        self.assertEqual(args.clusters, "2,4,8,16")
        with mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            cr_experiment.build_parser().parse_args(["--dataset-kind", "dnerf", "--engine", "compose"])
        self.assertEqual(args.progress_every_views, 5)
        self.assertEqual(args.warmup_iters, 3)
        self.assertEqual(args.measure_iters, 5)
        self.assertEqual(args.max_metric_views, 5)
        self.assertFalse(args.all_metric_views)
        self.assertEqual(args.width, 1440)
        self.assertEqual(args.height, 2560)
        self.assertEqual(args.views, 66)
        self.assertEqual(args.camera_aspect_mode, "expand")

    def test_run_id_includes_scene_engine_and_camera(self):
        run_id = cr_experiment.make_run_id(
            model_path=Path("/data/ysj/result/4dgs/RTGS/jumpingjacks"),
            engine="one_shot",
            split="test",
            camera_index=2,
            n3dv_frame_index=17,
            suffix="20260629_120000",
        )

        self.assertEqual(run_id, "jumpingjacks_one_shot_test_cam002_frame0017_20260629_120000")

    def test_build_one_shot_metric_row_uses_existing_schema(self):
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
            engine="one_shot",
            camera_split="test",
            camera_index=0,
            output_prefix="frame_0000",
            timing=timing,
            metrics=metrics,
        )

        self.assertEqual(row["variant"], "cluster_4")
        self.assertEqual(row["engine"], "one_shot")
        self.assertEqual(row["cluster_size"], 4)
        self.assertAlmostEqual(row["fps"], 12.5)
        self.assertAlmostEqual(row["peak_vram_gb"], 3.25)
        self.assertAlmostEqual(row["psnr_mean"], 31.0)
        self.assertEqual(row["metric_view_count"], 2)

    def test_build_one_shot_metric_row_includes_detailed_timing_and_context_columns(self):
        variant = SimpleNamespace(
            name="cluster_2",
            group="cluster_sweep",
            cluster_size=2,
            use_remapping=True,
            reuse_enabled=True,
        )
        timing = SimpleNamespace(
            fps=25.0,
            frame_ms=40.0,
            peak_vram_gb=7.5,
            detailed_metrics={
                "dynamic_geometry_ms": 1.0,
                "temporal_opacity_ms": 2.0,
                "snapshot_compaction_ms": 3.0,
                "dynamic_color_ms": 4.0,
                "rtgs_dynamic_total_ms": 10.0,
                "cr_projection_ms": 5.0,
                "cr_keygen_ms": 6.0,
                "cr_sort_ms": float("nan"),
                "cr_blend_ms": 7.0,
                "cr_core_total_ms": 18.0,
                "lkg_unpatchify_ms": 8.0,
                "lkg_unpad_ms": 9.0,
                "lkg_panel_paste_ms": 10.0,
                "lkg_interlace_post_ms": 27.0,
                "frame_ms_without_lkg": 28.0,
                "fps_without_lkg": 1000.0 / 28.0,
                "frame_ms_with_lkg_interlace": 55.0,
                "fps_with_lkg_interlace": 1000.0 / 55.0,
            },
        )
        context = SimpleNamespace(
            args=SimpleNamespace(
                dataset_kind="n3dv",
                n3dv_frame_index=12,
                sample_save_views=5,
                tile_size=16,
                map_mode="linear",
                camera_aspect_mode="expand",
            ),
            runtime=SimpleNamespace(
                official=SimpleNamespace(
                    scene_name="coffee_martini",
                    checkpoint_path=Path("/data/scene/checkpoints/chkpnt_best.pth"),
                    gaussians=SimpleNamespace(get_xyz=torch.zeros((10, 3), dtype=torch.float32)),
                    scene_paths=SimpleNamespace(dataset_kind="n3dv"),
                )
            ),
            timestamp=0.25,
            render_width=320,
            render_height=568,
            source_view_count=66,
            geometry=SimpleNamespace(means=torch.zeros((4, 3), dtype=torch.float32)),
        )

        row = cr_experiment.build_metric_row(
            variant=variant,
            engine="one_shot",
            camera_split="test",
            camera_index=0,
            output_prefix="frame_0012",
            timing=timing,
            metrics=None,
            context=context,
        )

        self.assertEqual(row["dataset_kind"], "n3dv")
        self.assertEqual(row["scene"], "coffee_martini")
        self.assertEqual(row["checkpoint"], "/data/scene/checkpoints/chkpnt_best.pth")
        self.assertEqual(row["frame_index"], 12)
        self.assertAlmostEqual(row["timestamp"], 0.25)
        self.assertEqual(row["render_width"], 320)
        self.assertEqual(row["render_height"], 568)
        self.assertEqual(row["source_views"], 66)
        self.assertEqual(row["saved_views"], 5)
        self.assertEqual(row["color_eval_views"], 33)
        self.assertEqual(row["tile_size"], 16)
        self.assertEqual(row["map_mode"], "linear")
        self.assertEqual(row["camera_aspect_mode"], "expand")
        self.assertEqual(row["total_gaussians"], 10)
        self.assertEqual(row["active_gaussians"], 4)
        self.assertAlmostEqual(row["active_gaussian_ratio"], 0.4)
        self.assertAlmostEqual(row["dynamic_geometry_ms"], 1.0)
        self.assertAlmostEqual(row["cr_projection_ms"], 5.0)
        self.assertAlmostEqual(row["lkg_interlace_post_ms"], 27.0)
        self.assertAlmostEqual(row["frame_ms"], row["frame_ms_with_lkg_interlace"])
        self.assertAlmostEqual(row["fps"], row["fps_with_lkg_interlace"])

    def test_resolve_experiment_variants_one_shot_defaults_include_ablation_variants(self):
        args = cr_experiment.build_parser().parse_args(["--dataset-kind", "dnerf"])

        variants = cr_experiment.resolve_experiment_variants(args)

        self.assertEqual(
            [(variant.name, variant.cluster_size) for variant in variants[:4]],
            [("cluster_2", 2), ("cluster_4", 4), ("cluster_8", 8), ("cluster_16", 16)],
        )
        self.assertNotIn("cluster_1", [variant.name for variant in variants])
        self.assertEqual(
            [(variant.name, variant.cluster_size, variant.use_remapping, variant.reuse_enabled) for variant in variants[4:]],
            [
                ("without_remap", 8, False, True),
                ("without_reuse", 1, True, False),
                ("without_reuse_without_remap", 1, False, False),
            ],
        )

    def test_resolve_experiment_variants_one_shot_keeps_cluster_sweep(self):
        args = cr_experiment.build_parser().parse_args(
            [
                "--dataset-kind",
                "dnerf",
                "--clusters",
                "2,4",
                "--no-without-remap",
                "--no-without-reuse",
            ]
        )

        variants = cr_experiment.resolve_experiment_variants(args)

        self.assertEqual([(variant.name, variant.cluster_size) for variant in variants], [("cluster_2", 2), ("cluster_4", 4)])

    def test_resolve_metric_view_indices_can_select_all_views(self):
        args = cr_experiment.build_parser().parse_args(
            [
                "--dataset-kind",
                "dnerf",
                "--all-metric-views",
                "--max-metric-views",
                "5",
                "--metric-view-stride",
                "1",
            ]
        )

        self.assertEqual(cr_experiment.resolve_metric_view_indices(6, args), [0, 1, 2, 3, 4, 5])

    def test_resolve_metric_view_indices_keeps_sampled_default(self):
        args = cr_experiment.build_parser().parse_args(
            [
                "--dataset-kind",
                "dnerf",
                "--max-metric-views",
                "3",
                "--metric-view-stride",
                "2",
            ]
        )

        self.assertEqual(cr_experiment.resolve_metric_view_indices(8, args), [0, 2, 4])

    def test_run_uses_cluster_one_interlaced_as_metric_reference(self):
        args = cr_experiment.build_parser().parse_args(
            [
                "--dataset-kind",
                "dnerf",
                "--clusters",
                "2",
                "--no-without-remap",
                "--no-without-reuse",
                "--no-reference-interlaced",
                "--skip-web-assets",
                "--artifact-dir",
                str(Path(tempfile.gettempdir()) / "rtgs_cr_metric_reference_test"),
                "--run-id",
                "metric_reference_case",
                "--warmup-iters",
                "0",
                "--measure-iters",
                "1",
            ]
        )
        args.device = "cpu"
        fake_context = SimpleNamespace(
            args=SimpleNamespace(
                dataset_kind="dnerf",
                n3dv_frame_index=0,
                sample_save_views=5,
                tile_size=16,
                map_mode="linear",
                camera_aspect_mode="expand",
            ),
            runtime=SimpleNamespace(
                official=SimpleNamespace(
                    scene_name="jumpingjacks",
                    checkpoint_path=Path("/data/checkpoints/chkpnt_best.pth"),
                    iteration=100,
                    source_path=Path("/data/dnerf/jumpingjacks"),
                    scene_paths=SimpleNamespace(
                        dataset_kind="dnerf",
                        dataset_path=Path("/data/dnerf/jumpingjacks"),
                        config_path=Path("/data/configs/jumpingjacks.py"),
                    ),
                    gaussians=SimpleNamespace(
                        get_xyz=torch.zeros((4, 3), dtype=torch.float32),
                        active_sh_degree=3,
                        active_sh_degree_t=1,
                    ),
                    rtgs_git_commit="abc123",
                )
            ),
            geometry=SimpleNamespace(means=torch.zeros((4, 3), dtype=torch.float32)),
            viewpoint_index=torch.zeros((2, 2, 3), dtype=torch.int64).numpy(),
            source_view_count=2,
            render_width=2,
            render_height=2,
            panel_width=2,
            panel_height=2,
            timestamp=0.0,
            viewport=SimpleNamespace(to_manifest=lambda: {"render_width": 2, "render_height": 2}),
            view_degree=53.0,
            orbit_direction=-1,
            view_index_stats={"min": 0, "max": 1},
            fov_normalization={},
        )
        writer = mock.Mock()
        metric_reference = torch.zeros((3, 2, 2), dtype=torch.float32)
        cr_render = torch.full((3, 2, 2), 0.5, dtype=torch.float32)
        timing = cr_experiment.TimingStats(fps=10.0, frame_ms=100.0, peak_vram_gb=0.0)
        metric_stats = cr_experiment.MetricStats(
            psnr_mean=12.0,
            psnr_std=0.0,
            ssim_mean=0.8,
            ssim_std=0.0,
            lpips_mean=0.2,
            lpips_std=0.0,
            metric_view_count=1,
        )

        with (
            mock.patch.object(cr_experiment, "_cuda_available", return_value=True),
            mock.patch.object(cr_experiment.cr_66views, "prepare_rtgs_cr_66_context", return_value=fake_context),
            mock.patch.object(cr_experiment, "render_metric_reference_interlaced", return_value=metric_reference) as render_reference,
            mock.patch.object(cr_experiment, "time_one_shot_renderer", return_value=(cr_render, timing)),
            mock.patch.object(cr_experiment, "compute_interlaced_metric_stats", return_value=metric_stats) as compute_metrics,
            mock.patch.object(cr_experiment, "ArtifactWriter", return_value=writer),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            rc = cr_experiment.run_rtgs_cr_experiment(args)

        self.assertEqual(rc, 0)
        render_reference.assert_called_once_with(fake_context)
        compute_metrics.assert_called_once_with(cr_render, metric_reference, require_lpips=False)
        final_rows = writer.write_metrics_csv.call_args_list[-1].args[0]
        self.assertEqual(final_rows[0]["metric_reference_variant"], "cluster_1")
        self.assertEqual(final_rows[0]["metric_scope"], "interlaced_cluster_reference")
        self.assertAlmostEqual(final_rows[0]["psnr_mean"], 12.0)
        self.assertAlmostEqual(final_rows[0]["ssim_mean"], 0.8)
        self.assertAlmostEqual(final_rows[0]["lpips_mean"], 0.2)

    def test_without_reuse_is_measured_cr_variant(self):
        args = cr_experiment.build_parser().parse_args(
            [
                "--dataset-kind",
                "dnerf",
                "--clusters",
                "2",
                "--no-without-remap",
                "--no-reference-interlaced",
                "--skip-metrics",
                "--skip-web-assets",
                "--artifact-dir",
                str(Path(tempfile.gettempdir()) / "rtgs_cr_cluster_one_gt_test"),
                "--run-id",
                "cluster_one_gt_case",
                "--warmup-iters",
                "0",
                "--measure-iters",
                "1",
            ]
        )
        args.device = "cpu"
        fake_context = SimpleNamespace(
            args=SimpleNamespace(
                dataset_kind="dnerf",
                n3dv_frame_index=0,
                sample_save_views=5,
                tile_size=16,
                map_mode="linear",
                camera_aspect_mode="expand",
            ),
            runtime=SimpleNamespace(
                official=SimpleNamespace(
                    scene_name="jumpingjacks",
                    checkpoint_path=Path("/data/checkpoints/chkpnt_best.pth"),
                    iteration=100,
                    source_path=Path("/data/dnerf/jumpingjacks"),
                    scene_paths=SimpleNamespace(
                        dataset_kind="dnerf",
                        dataset_path=Path("/data/dnerf/jumpingjacks"),
                        config_path=Path("/data/configs/jumpingjacks.py"),
                    ),
                    gaussians=SimpleNamespace(
                        get_xyz=torch.zeros((4, 3), dtype=torch.float32),
                        active_sh_degree=3,
                        active_sh_degree_t=1,
                    ),
                    rtgs_git_commit="abc123",
                )
            ),
            geometry=SimpleNamespace(means=torch.zeros((4, 3), dtype=torch.float32)),
            viewpoint_index=torch.zeros((2, 2, 3), dtype=torch.int64).numpy(),
            source_view_count=2,
            render_width=2,
            render_height=2,
            panel_width=2,
            panel_height=2,
            timestamp=0.0,
            viewport=SimpleNamespace(to_manifest=lambda: {"render_width": 2, "render_height": 2}),
            view_degree=53.0,
            orbit_direction=-1,
            view_index_stats={"min": 0, "max": 1},
            fov_normalization={},
        )
        writer = mock.Mock()
        reference = torch.zeros((3, 2, 2), dtype=torch.float32)
        cr_render = torch.full((3, 2, 2), 0.5, dtype=torch.float32)
        timing = cr_experiment.TimingStats(fps=10.0, frame_ms=100.0, peak_vram_gb=0.0)

        with (
            mock.patch.object(cr_experiment, "_cuda_available", return_value=True),
            mock.patch.object(cr_experiment.cr_66views, "prepare_rtgs_cr_66_context", return_value=fake_context),
            mock.patch.object(cr_experiment, "render_official_reference_interlaced", return_value=reference) as render_reference,
            mock.patch.object(cr_experiment, "time_one_shot_renderer", return_value=(cr_render, timing)) as render_cr,
            mock.patch.object(cr_experiment, "ArtifactWriter", return_value=writer),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            rc = cr_experiment.run_rtgs_cr_experiment(args)

        self.assertEqual(rc, 0)
        render_reference.assert_not_called()
        self.assertEqual(render_cr.call_count, 3)
        self.assertEqual(
            [call.kwargs["variant"].name for call in render_cr.call_args_list],
            ["cluster_2", "without_reuse", "without_reuse_without_remap"],
        )
        saved = {call.args[0]: call.args[1] for call in writer.save_tensor_image.call_args_list}
        output_path = Path("images") / "without_reuse" / "looking_glass_tensor.png"
        self.assertTrue(torch.equal(saved[output_path], cr_render))
        self.assertNotIn(Path("images") / "without_reuse" / "abs_error.png", saved)

    def test_time_one_shot_renderer_measures_average_and_peak_vram(self):
        calls = []

        def fake_render(_context):
            calls.append(len(calls))
            return torch.zeros((3, 2, 2), dtype=torch.float32), 7.0

        with mock.patch.object(cr_experiment, "_cuda_available", return_value=False):
            image, timing = cr_experiment.time_one_shot_renderer(
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

    def test_time_one_shot_renderer_averages_only_finite_detailed_measurements(self):
        samples = [
            {
                "cr_sort_ms": float("nan"),
                "cr_projection_ms": 1.0,
                "cr_keygen_ms": 3.0,
                "cr_blend_ms": 5.0,
                "cr_timing_source": "legacy_stage_fallback",
            },
            {
                "cr_sort_ms": 2.0,
                "cr_projection_ms": 3.0,
                "cr_keygen_ms": 5.0,
                "cr_blend_ms": 7.0,
                "cr_timing_source": "legacy_stage_fallback",
            },
        ]

        def fake_render(_context):
            sample = samples.pop(0)
            return torch.zeros((3, 2, 2), dtype=torch.float32), sample

        with mock.patch.object(cr_experiment, "_cuda_available", return_value=False):
            _image, timing = cr_experiment.time_one_shot_renderer(
                SimpleNamespace(),
                render_fn=fake_render,
                warmup_iters=0,
                measure_iters=2,
            )

        self.assertAlmostEqual(timing.detailed_metrics["cr_sort_ms"], 2.0)
        self.assertAlmostEqual(timing.detailed_metrics["cr_projection_ms"], 2.0)
        self.assertAlmostEqual(timing.detailed_metrics["cr_keygen_ms"], 4.0)
        self.assertAlmostEqual(timing.detailed_metrics["cr_core_total_ms"], 12.0)
        self.assertEqual(timing.detailed_metrics["cr_timing_source"], "legacy_stage_fallback")

    def test_time_one_shot_renderer_reports_outer_end_to_end_timing(self):
        def fake_render(_context):
            return torch.zeros((3, 2, 2), dtype=torch.float32), {
                "frame_ms_with_lkg_interlace": 10.0,
                "cr_projection_ms": 1.0,
                "cr_keygen_ms": 2.0,
                "cr_blend_ms": 7.0,
            }

        with (
            mock.patch.object(cr_experiment, "_cuda_available", return_value=False),
            mock.patch.object(cr_experiment.time, "perf_counter", side_effect=[1.0, 1.05, 2.0, 2.09]),
        ):
            _image, timing = cr_experiment.time_one_shot_renderer(
                SimpleNamespace(),
                render_fn=fake_render,
                warmup_iters=0,
                measure_iters=2,
            )

        self.assertAlmostEqual(timing.frame_ms, 10.0)
        self.assertAlmostEqual(timing.detailed_metrics["frame_ms_end_to_end"], 70.0)
        self.assertAlmostEqual(timing.detailed_metrics["fps_end_to_end"], 1000.0 / 70.0)

    def test_cuda_peak_vram_uses_allocated_memory_not_reserved_cache(self):
        with (
            mock.patch.object(cr_experiment, "_cuda_available", return_value=True),
            mock.patch.object(torch.cuda, "max_memory_allocated", return_value=2 * 2**30),
            mock.patch.object(torch.cuda, "max_memory_reserved", return_value=9 * 2**30),
        ):
            self.assertAlmostEqual(cr_experiment._peak_vram_gb(), 2.0)

    def test_cuda_peak_reset_clears_allocator_cache(self):
        with (
            mock.patch.object(cr_experiment, "_cuda_available", return_value=True),
            mock.patch.object(torch.cuda, "empty_cache") as empty_cache,
            mock.patch.object(torch.cuda, "reset_peak_memory_stats") as reset_peak,
        ):
            cr_experiment._reset_cuda_peak_memory()

        empty_cache.assert_called_once()
        reset_peak.assert_called_once()

    def test_time_one_shot_renderer_reports_iteration_progress(self):
        def fake_render(_context, *, progress_label=None):
            self.assertIsNotNone(progress_label)
            return torch.zeros((3, 2, 2), dtype=torch.float32), 10.0

        stream = io.StringIO()
        with mock.patch.object(cr_experiment, "_cuda_available", return_value=False), mock.patch("sys.stderr", stream):
            cr_experiment.time_one_shot_renderer(
                SimpleNamespace(),
                render_fn=fake_render,
                warmup_iters=1,
                measure_iters=1,
                progress_label="one_shot/cluster_2",
            )

        output = stream.getvalue()
        self.assertIn("one_shot/cluster_2 timing 1/2 warmup", output)
        self.assertIn("one_shot/cluster_2 timing 2/2 measure", output)

    def test_render_one_shot_interlaced_once_delegates_to_one_shot_module(self):
        context = SimpleNamespace(name="context")
        variant = SimpleNamespace(name="cluster_2")
        expected = (torch.zeros((3, 2, 2), dtype=torch.float32), 3.5)

        with mock.patch(
            "lkg_experiment.rtgs_coherent.cr_one_shot.render_rtgs_cr_one_shot_interlaced_once",
            return_value=expected,
        ) as render:
            actual = cr_experiment.render_one_shot_interlaced_once(context, variant=variant)

        self.assertIs(actual, expected)
        render.assert_called_once_with(context, variant=variant)

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

    def test_all_experiment_script_defaults_to_one_shot_without_engine_flag(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_rtgs_cr_experiments_all.sh"
        text = script.read_text(encoding="utf-8")

        self.assertNotIn('ENGINE="${ENGINE:-compose}"', text)
        self.assertNotIn("--engine \"$ENGINE\"", text)
        self.assertIn('CLUSTERS="${CLUSTERS:-2,4,8,16}"', text)
        self.assertIn('CAMERA_ASPECT_MODE="${CAMERA_ASPECT_MODE:-expand}"', text)
        self.assertIn('--camera-aspect-mode "$CAMERA_ASPECT_MODE"', text)
        self.assertIn('SAMPLE_SAVE_VIEWS="${SAMPLE_SAVE_VIEWS:-5}"', text)
        self.assertIn('--sample-save-views "$SAMPLE_SAVE_VIEWS"', text)
        self.assertIn('ALL_METRIC_VIEWS="${ALL_METRIC_VIEWS:-0}"', text)
        self.assertIn("cmd+=(--all-metric-views)", text)
        self.assertIn('ALL_TEST_VIEWS="${ALL_TEST_VIEWS:-0}"', text)
        self.assertIn('--output-prefix "$output_prefix"', text)
        self.assertIn("cmd+=(--append-metrics)", text)

    def test_full_view_experiment_script_enables_complete_metrics(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_rtgs_cr_experiments_all_full_views.sh"
        text = script.read_text(encoding="utf-8")

        self.assertIn('SPLIT="${SPLIT:-test}"', text)
        self.assertIn('ALL_TEST_VIEWS="${ALL_TEST_VIEWS:-1}"', text)
        self.assertIn('ALL_METRIC_VIEWS="${ALL_METRIC_VIEWS:-1}"', text)
        self.assertIn('SAMPLE_SAVE_VIEWS="${SAMPLE_SAVE_VIEWS:-66}"', text)
        self.assertIn('NO_REFERENCE_INTERLACED="${NO_REFERENCE_INTERLACED:-0}"', text)
        self.assertIn('RUN_GROUP="${RUN_GROUP:-rtgs_cr_full_view_experiments_', text)
        self.assertIn('exec bash "$SCRIPT_DIR/run_rtgs_cr_experiments_all.sh"', text)


if __name__ == "__main__":
    unittest.main()
