from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from lkg_experiment.omg4_ftgs import interlaced_experiment


class Omg4FtgsInterlacedExperimentTest(unittest.TestCase):
    def test_parser_defaults_match_1440x2560_interlaced_experiment(self):
        args = interlaced_experiment.build_parser().parse_args([])

        self.assertEqual(args.output_root, "/data/ysj/result/coherent-raster/generated/omg4_ftgs_interlaced_experiments")
        self.assertEqual(args.panel_width, 1440)
        self.assertEqual(args.panel_height, 2560)
        self.assertEqual(args.views, 66)
        self.assertEqual(args.cluster_size, 8)
        self.assertEqual(args.clusters, "2,4,8,16")
        self.assertEqual(args.ablation_cluster, 8)
        self.assertFalse(args.no_without_remap)
        self.assertFalse(args.no_without_reuse)
        self.assertEqual(args.warmup_iters, 3)
        self.assertEqual(args.measure_iters, 5)

    def test_resolve_experiment_variants_includes_combined_ablation(self):
        args = interlaced_experiment.build_parser().parse_args(["--clusters", "2,4"])

        variants = interlaced_experiment.resolve_experiment_variants(args)

        self.assertEqual(
            [(variant.name, variant.cluster_size, variant.use_remapping, variant.reuse_enabled) for variant in variants],
            [
                ("cluster_2", 2, True, True),
                ("cluster_4", 4, True, True),
                ("without_remap", 8, False, True),
                ("without_reuse", 1, True, False),
                ("without_reuse_without_remap", 1, False, False),
            ],
        )

    def test_discover_scene_jobs_maps_weight_names_to_n3dv_dataset_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "weights" / "ours_L_weight"
            data = root / "N3DV"
            weights.mkdir(parents=True)
            data.mkdir()
            for scene in ["cook_spinach", "flame_salmon"]:
                (weights / f"{scene}.xz").write_bytes(b"placeholder")
            (data / "cook_spinach").mkdir()
            (data / "flame_salmon_1").mkdir()

            jobs = interlaced_experiment.discover_scene_jobs(
                weights_root=root / "weights",
                weight_group="ours_L_weight",
                data_root=data,
                scenes=None,
                require_data=True,
            )

        self.assertEqual([job.scene for job in jobs], ["cook_spinach", "flame_salmon"])
        self.assertEqual(jobs[0].dataset_scene, "cook_spinach")
        self.assertEqual(jobs[1].dataset_scene, "flame_salmon_1")

    def test_average_interlaced_measurements_uses_panel_generation_as_primary_fps(self):
        samples = [
            {
                "materialize_ms": 3.0,
                "view_setup_ms": 7.0,
                "cr_render_ms": 30.0,
                "frame_ms_with_lkg_interlace": 40.0,
                "frame_ms_including_lookup": 140.0,
                "frame_ms_end_to_end": 150.0,
                "cr_core_ms": 28.0,
                "post_ms": 2.0,
                "cr_timing_source": "extension_events",
            },
            {
                "materialize_ms": 5.0,
                "view_setup_ms": 15.0,
                "cr_render_ms": 40.0,
                "frame_ms_with_lkg_interlace": 60.0,
                "frame_ms_including_lookup": 160.0,
                "frame_ms_end_to_end": 170.0,
                "cr_core_ms": 37.0,
                "post_ms": 3.0,
                "cr_timing_source": "extension_events",
            },
        ]

        averaged = interlaced_experiment.average_measurements(samples)

        self.assertAlmostEqual(averaged["frame_ms"], 50.0)
        self.assertAlmostEqual(averaged["fps"], 20.0)
        self.assertAlmostEqual(averaged["frame_ms_with_lkg_interlace"], 50.0)
        self.assertAlmostEqual(averaged["frame_ms_excluding_lookup"], 50.0)
        self.assertAlmostEqual(averaged["fps_with_lkg_interlace"], 20.0)
        self.assertAlmostEqual(averaged["frame_ms_end_to_end"], 160.0)
        self.assertAlmostEqual(averaged["fps_end_to_end"], 1000.0 / 160.0)
        self.assertAlmostEqual(averaged["materialize_ms"], 4.0)
        self.assertAlmostEqual(averaged["view_setup_ms"], 11.0)
        self.assertAlmostEqual(averaged["cr_render_ms"], 35.0)
        self.assertEqual(averaged["cr_timing_source"], "extension_events")

    def test_cuda_peak_vram_uses_allocated_memory_not_reserved_cache(self):
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "max_memory_allocated", return_value=2 * 2**30),
            mock.patch.object(torch.cuda, "max_memory_reserved", return_value=9 * 2**30),
        ):
            self.assertAlmostEqual(interlaced_experiment._peak_vram_gb(), 2.0)

    def test_cuda_peak_reset_clears_allocator_cache(self):
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "empty_cache") as empty_cache,
            mock.patch.object(torch.cuda, "reset_peak_memory_stats") as reset_peak,
        ):
            interlaced_experiment._reset_cuda_peak_memory()

        empty_cache.assert_called_once()
        reset_peak.assert_called_once()

    def test_render_interlaced_once_measures_dynamic_view_setup(self):
        args = interlaced_experiment.build_parser().parse_args(
            [
                "--device",
                "cpu",
                "--panel-width",
                "4",
                "--panel-height",
                "3",
                "--views",
                "4",
                "--cluster-size",
                "2",
                "--map-mode",
                "linear",
            ]
        )
        splats = SimpleNamespace(means=np.zeros((2, 3), dtype=np.float32))
        dynamic_model = SimpleNamespace(materialize=mock.Mock(return_value=splats))
        prepared = SimpleNamespace(
            viewpoint_index=np.zeros((3, 4, 3), dtype=np.int32),
            view_idx_matrix=object(),
            subpixel_coord_matrix=object(),
            lookup_cpu_ms=1.0,
            lookup_h2d_ms=2.0,
            viewpoint_index_ms=5.0,
            source_c2w=np.eye(4, dtype=np.float32),
            K=np.eye(3, dtype=np.float32),
        )
        image = SimpleNamespace()
        render_meta = {
                "timing_ms": {
                    "cr_projection_ms": 1.0,
                    "cr_keygen_ms": 2.0,
                    "cr_sort_ms": float("nan"),
                    "cr_blend_ms": 3.0,
                },
            "post_ms": 0.5,
        }

        with (
            mock.patch.object(interlaced_experiment, "estimate_orbit_center", return_value="center") as estimate_center,
            mock.patch.object(interlaced_experiment, "synthesize_interlaced_viewmats", return_value="viewmats") as synthesize,
            mock.patch.object(interlaced_experiment, "render_splats_interlaced_coherent", return_value=(image, render_meta)),
            mock.patch.object(
                interlaced_experiment.time,
                "perf_counter",
                side_effect=[10.0, 10.0, 10.1, 10.1, 10.4, 10.4, 10.8, 10.9],
            ),
        ):
            rendered, timing = interlaced_experiment.render_interlaced_once(
                args=args,
                dynamic_model=dynamic_model,
                prepared=prepared,
                timestamp=0.5,
            )

        self.assertIs(rendered, image)
        dynamic_model.materialize.assert_called_once_with(0.5)
        estimate_center.assert_called_once()
        synthesize.assert_called_once_with(
            c2w=prepared.source_c2w,
            orbit_center="center",
            views=4,
            cluster_size=2,
            view_degree=53.0,
            orbit_direction=-1,
            device="cpu",
        )
        self.assertAlmostEqual(timing["materialize_ms"], 100.0)
        self.assertAlmostEqual(timing["view_setup_ms"], 300.0)
        self.assertAlmostEqual(timing["cr_render_ms"], 400.0)
        self.assertAlmostEqual(timing["cr_core_ms"], 6.0)
        self.assertAlmostEqual(timing["frame_ms_with_lkg_interlace"], 800.0)
        self.assertAlmostEqual(timing["frame_ms_excluding_lookup"], 800.0)
        self.assertAlmostEqual(timing["frame_ms_including_lookup"], 808.0)
        self.assertAlmostEqual(timing["frame_ms_end_to_end"], 908.0)
        self.assertAlmostEqual(timing["fps_end_to_end"], 1000.0 / 908.0)
        self.assertAlmostEqual(timing["viewpoint_index_ms"], 5.0)

    def test_prepare_interlaced_scene_applies_variant_remap_and_reuse_flags(self):
        args = interlaced_experiment.build_parser().parse_args(
            [
                "--device",
                "cpu",
                "--panel-width",
                "4",
                "--panel-height",
                "3",
                "--views",
                "4",
                "--cluster-size",
                "2",
                "--map-mode",
                "linear",
            ]
        )
        splats = SimpleNamespace(means=np.zeros((2, 3), dtype=np.float32))
        dynamic_model = SimpleNamespace(materialize=mock.Mock(return_value=splats))
        camera_set = SimpleNamespace(
            K=np.eye(3, dtype=np.float32),
            width=4,
            height=3,
            viewmat_at=mock.Mock(return_value=np.eye(4, dtype=np.float32)),
        )
        frame = SimpleNamespace(timestamp=0.25)
        variant = SimpleNamespace(
            name="without_reuse_without_remap",
            cluster_size=1,
            use_remapping=False,
            reuse_enabled=False,
        )
        prepared_lookup = SimpleNamespace(
            view_idx_matrix=object(),
            subpixel_coord_matrix=object(),
            lookup_cpu_ms=1.0,
            lookup_h2d_ms=2.0,
        )

        with (
            mock.patch.object(
                interlaced_experiment,
                "build_interlaced_viewpoint_index",
                return_value=(np.zeros((3, 4, 3), dtype=np.int32), {"mode": "linear", "views": 4}),
            ),
            mock.patch.object(interlaced_experiment, "prepare_cr_lookup_tensors", return_value=prepared_lookup) as prepare_lookup,
            mock.patch.object(interlaced_experiment, "estimate_orbit_center", return_value="center"),
            mock.patch.object(interlaced_experiment, "synthesize_interlaced_viewmats", return_value="viewmats") as synthesize,
        ):
            prepared = interlaced_experiment.prepare_interlaced_scene(
                args=args,
                camera_set=camera_set,
                dynamic_model=dynamic_model,
                frame=frame,
                variant=variant,
            )

        prepare_lookup.assert_called_once_with(
            prepared.viewpoint_index,
            device="cpu",
            tile_size=16,
            use_remapping=False,
        )
        synthesize.assert_called_once()
        self.assertEqual(synthesize.call_args.kwargs["cluster_size"], 1)

    def test_scene_summary_extracts_interlaced_fps_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "scene"
            output_dir.mkdir()
            (output_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "scene": "cook_spinach",
                        "dataset_scene": "cook_spinach",
                        "weight_group": "ours_L_weight",
                        "camera_index": 0,
                        "frame_index": 0,
                        "timestamp": 0.0,
                        "panel_width": 1440,
                        "panel_height": 2560,
                        "views": 66,
                        "engine": "omg4_ftgs_interlaced",
                        "variant": "cluster_8",
                        "group": "cluster_sweep",
                        "cluster_size": 8,
                        "use_remapping": True,
                        "reuse_enabled": True,
                        "render_width": 1440,
                        "render_height": 2560,
                        "source_views": 66,
                        "color_eval_views": 9,
                        "tile_size": 16,
                        "map_mode": "file",
                        "checkpoint_path": "/weights/cook_spinach.xz",
                        "data_path": "/data/cook_spinach",
                        "image_path": str(output_dir / "omg4_ftgs_lkg_interlaced.png"),
                        "timing": {
                            "frame_ms": 50.0,
                            "fps": 20.0,
                            "frame_ms_with_lkg_interlace": 50.0,
                            "frame_ms_excluding_lookup": 50.0,
                            "fps_with_lkg_interlace": 20.0,
                            "frame_ms_including_lookup": 1200.0,
                            "fps_including_lookup": 1000.0 / 1200.0,
                            "viewpoint_index_ms": 8.0,
                            "view_setup_ms": 6.0,
                            "lookup_cpu_ms": 1100.0,
                            "lookup_h2d_ms": 50.0,
                            "cr_core_ms": 44.0,
                            "post_ms": 2.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            job = interlaced_experiment.SceneJob(
                scene="cook_spinach",
                dataset_scene="cook_spinach",
                checkpoint_path=Path("/weights/cook_spinach.xz"),
                data_path=Path("/data/cook_spinach"),
            )

            row = interlaced_experiment.scene_summary_from_manifest(job, output_dir)

        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["panel_width"], 1440)
        self.assertEqual(row["panel_height"], 2560)
        self.assertEqual(row["engine"], "omg4_ftgs_interlaced")
        self.assertEqual(row["variant"], "cluster_8")
        self.assertEqual(row["group"], "cluster_sweep")
        self.assertTrue(row["use_remapping"])
        self.assertTrue(row["reuse_enabled"])
        self.assertEqual(row["render_width"], 1440)
        self.assertEqual(row["render_height"], 2560)
        self.assertEqual(row["source_views"], 66)
        self.assertEqual(row["color_eval_views"], 9)
        self.assertEqual(row["tile_size"], 16)
        self.assertAlmostEqual(row["frame_ms"], 50.0)
        self.assertAlmostEqual(row["fps"], 20.0)
        self.assertAlmostEqual(row["frame_ms_excluding_lookup"], 50.0)
        self.assertAlmostEqual(row["frame_ms_including_lookup"], 1200.0)
        self.assertAlmostEqual(row["frame_ms_end_to_end"], 1200.0)
        self.assertAlmostEqual(row["fps_end_to_end"], 1000.0 / 1200.0)
        self.assertAlmostEqual(row["viewpoint_index_ms"], 8.0)
        self.assertAlmostEqual(row["view_setup_ms"], 6.0)

    def test_write_summary_orders_common_columns_before_omg4_specific_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            interlaced_experiment.write_summary(
                [
                    {
                        "status": "ok",
                        "dataset_kind": "n3dv",
                        "scene": "cook_spinach",
                        "checkpoint": "/weights/cook_spinach.xz",
                        "variant": "cluster_2",
                        "group": "cluster_sweep",
                        "render_width": 1440,
                        "render_height": 2560,
                        "source_views": 66,
                        "cluster_size": 2,
                        "use_remapping": True,
                        "reuse_enabled": True,
                        "cr_core_total_ms": 10.0,
                        "materialize_ms": 3.0,
                        "dataset_scene": "cook_spinach",
                        "output_dir": str(output_dir / "scene" / "cluster_2"),
                    }
                ],
                output_dir,
            )
            header = (output_dir / "summary.csv").read_text(encoding="utf-8").splitlines()[0].split(",")

        self.assertLess(header.index("variant"), header.index("dataset_scene"))
        self.assertLess(header.index("cr_core_total_ms"), header.index("materialize_ms"))
        self.assertLess(header.index("total_gaussians"), header.index("output_dir"))

    def test_run_dry_run_plans_interlaced_outputs_under_one_run_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "weights" / "ours_L_weight"
            data = root / "N3DV"
            output_root = root / "runs"
            weights.mkdir(parents=True)
            data.mkdir()
            for scene in ["coffee_martini", "cook_spinach"]:
                (weights / f"{scene}.xz").write_bytes(b"placeholder")
                (data / scene).mkdir()
            args = interlaced_experiment.build_parser().parse_args(
                [
                    "--weights-root",
                    str(root / "weights"),
                    "--data-root",
                    str(data),
                    "--output-root",
                    str(output_root),
                    "--run-group",
                    "run_a",
                    "--dry-run",
                ]
            )

            exit_code = interlaced_experiment.run(args)
            summary = json.loads((output_root / "run_a" / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(summary), 14)
        self.assertEqual({row["status"] for row in summary}, {"planned"})
        self.assertEqual(
            sorted({row["variant"] for row in summary}),
            [
                "cluster_16",
                "cluster_2",
                "cluster_4",
                "cluster_8",
                "without_remap",
                "without_reuse",
                "without_reuse_without_remap",
            ],
        )
        self.assertEqual({row["panel_width"] for row in summary}, {1440})
        self.assertEqual({row["panel_height"] for row in summary}, {2560})
        self.assertEqual({Path(row["output_dir"]).parents[2].name for row in summary}, {"run_a"})
        self.assertTrue(all(Path(row["output_dir"]).name == row["variant"] for row in summary))

    def test_cuda_preflight_reports_tinycudann_mixed_gpu_import_error(self):
        args = interlaced_experiment.build_parser().parse_args([])

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=mock.Mock(return_value=True),
                device_count=mock.Mock(return_value=3),
                get_device_capability=mock.Mock(side_effect=[(8, 6), (8, 6), (7, 5)]),
            )
        )

        with (
            mock.patch.object(interlaced_experiment, "_import_torch", return_value=fake_torch),
            mock.patch.object(interlaced_experiment, "_import_tinycudann", side_effect=OSError("Could not find compatible tinycudann extension for compute capability 75.")),
            mock.patch.dict("os.environ", {}, clear=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA_VISIBLE_DEVICES=0"):
                interlaced_experiment.preflight_cuda_runtime(args)

    def test_run_fails_fast_before_scene_loop_when_cuda_preflight_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "weights" / "ours_L_weight"
            data = root / "N3DV"
            weights.mkdir(parents=True)
            data.mkdir()
            (weights / "cook_spinach.xz").write_bytes(b"placeholder")
            (data / "cook_spinach").mkdir()
            args = interlaced_experiment.build_parser().parse_args(
                [
                    "--weights-root",
                    str(root / "weights"),
                    "--data-root",
                    str(data),
                    "--output-root",
                    str(root / "runs"),
                ]
            )

            with (
                mock.patch.object(interlaced_experiment, "preflight_cuda_runtime", side_effect=RuntimeError("tinycudann preflight failed")),
                mock.patch.object(interlaced_experiment, "run_scene") as run_scene,
            ):
                with self.assertRaisesRegex(RuntimeError, "tinycudann preflight failed"):
                    interlaced_experiment.run(args)

        run_scene.assert_not_called()

    def test_run_scene_writes_interlaced_manifest_and_summary_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "scene"
            job = interlaced_experiment.SceneJob(
                scene="cook_spinach",
                dataset_scene="cook_spinach",
                checkpoint_path=Path("/weights/cook_spinach.xz"),
                data_path=Path("/data/cook_spinach"),
            )
            args = interlaced_experiment.build_parser().parse_args(
                [
                    "--warmup-iters",
                    "1",
                    "--measure-iters",
                    "2",
                    "--panel-width",
                    "4",
                    "--panel-height",
                    "3",
                    "--views",
                    "4",
                    "--cluster-size",
                    "2",
                    "--map-mode",
                    "linear",
                    "--device",
                    "cpu",
                ]
            )
            fake_image = mock.Mock()
            fake_image.shape = (3, 3, 4)
            fake_image.detach.return_value = fake_image
            fake_image.clamp.return_value = fake_image
            fake_image.cpu.return_value = fake_image
            fake_image.numpy.return_value = [[[0.0, 0.0, 0.0]]]
            prepared = SimpleNamespace(
                lookup_cpu_ms=10.0,
                lookup_h2d_ms=5.0,
                viewpoint_index_ms=4.0,
                setup_materialize_ms=1.0,
                setup_view_setup_ms=2.0,
                setup_gaussians=7,
                map_metadata={"mode": "linear", "views": 4},
                orbit_center=[0.0, 0.0, 1.0],
            )

            with (
                mock.patch.object(interlaced_experiment, "load_test_frames", return_value=[mock.Mock(index=0, timestamp=0.0)]),
                mock.patch.object(
                    interlaced_experiment,
                    "load_pose_camera",
                    return_value=mock.Mock(
                        width=4,
                        height=3,
                        K=[[1.0, 0.0, 2.0], [0.0, 1.0, 1.5], [0.0, 0.0, 1.0]],
                        viewmat_at=mock.Mock(return_value=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
                    ),
                ),
                mock.patch.object(interlaced_experiment, "load_dynamic_gaussians", return_value=mock.Mock()),
                mock.patch.object(interlaced_experiment, "prepare_interlaced_scene", return_value=prepared),
                mock.patch.object(
                    interlaced_experiment,
                    "render_interlaced_once",
                    side_effect=[
                        (fake_image, {"frame_ms_with_lkg_interlace": 100.0, "frame_ms_including_lookup": 115.0}),
                        (fake_image, {"frame_ms_with_lkg_interlace": 40.0, "frame_ms_including_lookup": 55.0}),
                        (fake_image, {"frame_ms_with_lkg_interlace": 60.0, "frame_ms_including_lookup": 75.0}),
                    ],
                ),
                mock.patch.object(interlaced_experiment.single_cli, "_save_tensor_image"),
            ):
                row = interlaced_experiment.run_scene(job, args, output_dir=output_dir)

            self.assertEqual(row["status"], "ok")
            self.assertAlmostEqual(row["frame_ms"], 50.0)
            self.assertAlmostEqual(row["fps"], 20.0)
            self.assertAlmostEqual(row["frame_ms_end_to_end"], 65.0)
            self.assertAlmostEqual(row["fps_end_to_end"], 1000.0 / 65.0)
            self.assertEqual((output_dir / "manifest.json").is_file(), True)

    def test_run_scene_compares_variant_image_against_cluster_one_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "scene"
            job = interlaced_experiment.SceneJob(
                scene="cook_spinach",
                dataset_scene="cook_spinach",
                checkpoint_path=Path("/weights/cook_spinach.xz"),
                data_path=Path("/data/cook_spinach"),
            )
            args = interlaced_experiment.build_parser().parse_args(
                [
                    "--warmup-iters",
                    "0",
                    "--measure-iters",
                    "1",
                    "--panel-width",
                    "4",
                    "--panel-height",
                    "3",
                    "--views",
                    "4",
                    "--cluster-size",
                    "2",
                    "--map-mode",
                    "linear",
                    "--device",
                    "cpu",
                ]
            )
            fake_image = mock.Mock()
            fake_image.shape = (3, 3, 4)
            fake_image.detach.return_value = fake_image
            fake_image.clamp.return_value = fake_image
            fake_image.cpu.return_value = fake_image
            fake_image.numpy.return_value = [[[0.0, 0.0, 0.0]]]
            metric_reference = object()
            metric_stats = interlaced_experiment.MetricStats(
                psnr_mean=22.0,
                psnr_std=0.0,
                ssim_mean=0.75,
                ssim_std=0.0,
                lpips_mean=0.15,
                lpips_std=0.0,
                metric_view_count=1,
            )
            prepared = SimpleNamespace(
                lookup_cpu_ms=10.0,
                lookup_h2d_ms=5.0,
                viewpoint_index_ms=4.0,
                setup_materialize_ms=1.0,
                setup_view_setup_ms=2.0,
                setup_gaussians=7,
                map_metadata={"mode": "linear", "views": 4},
                orbit_center=[0.0, 0.0, 1.0],
            )

            with (
                mock.patch.object(interlaced_experiment, "load_test_frames", return_value=[mock.Mock(index=0, timestamp=0.0)]),
                mock.patch.object(interlaced_experiment, "load_pose_camera", return_value=mock.Mock()),
                mock.patch.object(interlaced_experiment, "load_dynamic_gaussians", return_value=mock.Mock()),
                mock.patch.object(interlaced_experiment, "prepare_interlaced_scene", return_value=prepared),
                mock.patch.object(
                    interlaced_experiment,
                    "render_interlaced_once",
                    return_value=(fake_image, {"frame_ms_with_lkg_interlace": 40.0, "frame_ms_including_lookup": 55.0}),
                ),
                mock.patch.object(interlaced_experiment, "compute_interlaced_metric_stats", return_value=metric_stats) as compute_metrics,
                mock.patch.object(interlaced_experiment.single_cli, "_save_tensor_image"),
            ):
                row = interlaced_experiment.run_scene(
                    job,
                    args,
                    output_dir=output_dir,
                    metric_reference_image=metric_reference,
                )

        compute_metrics.assert_called_once_with(fake_image, metric_reference, require_lpips=False)
        self.assertEqual(row["metric_reference_variant"], "cluster_1")
        self.assertEqual(row["metric_scope"], "interlaced_cluster_reference")
        self.assertAlmostEqual(row["psnr_mean"], 22.0)
        self.assertAlmostEqual(row["ssim_mean"], 0.75)
        self.assertAlmostEqual(row["lpips_mean"], 0.15)

    def test_run_saves_cluster_one_metric_reference_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = interlaced_experiment.build_parser().parse_args(
                [
                    "--output-root",
                    tmp,
                    "--run-group",
                    "metric_reference_case",
                    "--scenes",
                    "cook_spinach",
                    "--warmup-iters",
                    "0",
                    "--measure-iters",
                    "1",
                ]
            )
            job = interlaced_experiment.SceneJob(
                scene="cook_spinach",
                dataset_scene="cook_spinach",
                checkpoint_path=Path("/weights/cook_spinach.xz"),
                data_path=Path("/data/cook_spinach"),
            )
            variant = interlaced_experiment.ExperimentVariant(
                name="cluster_2",
                cluster_size=2,
                use_remapping=True,
                reuse_enabled=True,
                group="cluster_sweep",
            )
            metric_reference = object()

            with (
                mock.patch.object(interlaced_experiment, "install_gsplat_root"),
                mock.patch.object(interlaced_experiment, "preflight_cuda_runtime"),
                mock.patch.object(interlaced_experiment, "discover_scene_jobs", return_value=[job]),
                mock.patch.object(interlaced_experiment, "resolve_experiment_variants", return_value=[variant]),
                mock.patch.object(interlaced_experiment, "render_metric_reference_interlaced", return_value=metric_reference),
                mock.patch.object(interlaced_experiment.single_cli, "_save_tensor_image") as save_image,
                mock.patch.object(
                    interlaced_experiment,
                    "run_scene",
                    return_value={
                        "status": "ok",
                        "dataset_kind": "n3dv",
                        "scene": "cook_spinach",
                        "variant": "cluster_2",
                    },
                ) as run_scene,
            ):
                rc = interlaced_experiment.run(args)

        self.assertEqual(rc, 0)
        expected_reference_path = (
            Path(tmp)
            / "metric_reference_case"
            / "ours_L_weight"
            / "cook_spinach"
            / "cluster_1"
            / "metric_reference.png"
        )
        save_image.assert_any_call(expected_reference_path, metric_reference)
        run_scene.assert_called_once()
        self.assertIs(run_scene.call_args.kwargs["metric_reference_image"], metric_reference)

    def test_wrapper_script_exists_and_uses_module_main(self):
        wrapper = Path(__file__).resolve().parents[1] / "omg4_ftgs_interlaced_experiment.py"

        self.assertTrue(wrapper.is_file())
        self.assertIn("lkg_experiment.omg4_ftgs.interlaced_experiment", wrapper.read_text(encoding="utf-8"))

    def test_pyproject_registers_console_script(self):
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"

        self.assertIn(
            'lkg-omg4-ftgs-interlaced-experiment = "lkg_experiment.omg4_ftgs.interlaced_experiment:main"',
            pyproject.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
