from __future__ import annotations

import json
import tempfile
import unittest
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from lkg_experiment.omg4_ftgs import camera, cli, model, render


class _FakeDynamicModel:
    def materialize(self, timestamp: float):
        return SimpleNamespace(
            timestamp=float(timestamp),
            means=torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]], dtype=torch.float32),
            quats=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            scales=torch.ones((2, 3), dtype=torch.float32),
            opacities=torch.ones(2, dtype=torch.float32),
            colors=torch.zeros((2, 16, 3), dtype=torch.float32),
            sh_degree=3,
        )


class _FakeCameraSet:
    K = np.eye(3, dtype=np.float32)
    width = 3
    height = 2

    def viewmat_at(self, camera_index: int):
        return np.eye(4, dtype=np.float32)


class Omg4FtgsTest(unittest.TestCase):
    def test_default_cook_spinach_paths(self):
        self.assertEqual(
            cli.DEFAULT_CHECKPOINT_PATH,
            Path("/data/ysj/result/4dgs/OMG4-FTGS_weights/ours_L_weight/cook_spinach.xz"),
        )
        self.assertEqual(cli.DEFAULT_DATA_PATH, Path("/data/ysj/dataset/N3DV/cook_spinach"))

    def test_parser_defaults_to_both_mode(self):
        args = cli.build_parser().parse_args([])

        self.assertEqual(args.mode, "both")
        self.assertEqual(args.renderer, "gsplat")
        self.assertEqual(args.camera_index, 0)
        self.assertEqual(args.frame_index, 0)

    def test_parser_accepts_renderer_modes(self):
        coherent_args = cli.build_parser().parse_args(["--renderer", "coherent"])
        both_args = cli.build_parser().parse_args(["--renderer", "both"])

        self.assertEqual(coherent_args.renderer, "coherent")
        self.assertEqual(both_args.renderer, "both")

    def test_parser_accepts_interlaced_defaults(self):
        args = cli.build_parser().parse_args(["--mode", "interlaced"])

        self.assertEqual(args.mode, "interlaced")
        self.assertEqual(args.views, 66)
        self.assertEqual(args.panel_width, 1440)
        self.assertEqual(args.panel_height, 2560)
        self.assertEqual(args.cluster_size, 8)
        self.assertEqual(args.map_mode, "file")

    def test_build_interlaced_viewpoint_index_linear(self):
        args = SimpleNamespace(map_mode="linear", views=4, coherent_quantize="floor", viewpoint_index_path="/unused")

        viewpoint_index, metadata = render.build_interlaced_viewpoint_index(args, width=5, height=3)

        self.assertEqual(viewpoint_index.shape, (3, 5, 3))
        self.assertEqual(viewpoint_index.dtype, np.int32)
        self.assertGreaterEqual(int(viewpoint_index.min()), 0)
        self.assertLess(int(viewpoint_index.max()), 4)
        self.assertEqual(metadata["mode"], "linear")
        self.assertEqual(metadata["views"], 4)

    def test_synthesize_interlaced_viewmats_groups_views(self):
        c2w = torch.eye(4, dtype=torch.float32)
        means = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]], dtype=torch.float32)

        estimated_center = render.estimate_orbit_center(
            c2w=c2w,
            splat_means=means,
            orbit_center_distance=0.0,
        )
        explicit_center = render.estimate_orbit_center(
            c2w=c2w,
            splat_means=means,
            orbit_center_distance=2.0,
        )
        grouped = render.synthesize_interlaced_viewmats(
            c2w=c2w,
            orbit_center=estimated_center,
            views=5,
            cluster_size=2,
            view_degree=20.0,
            orbit_direction=-1,
            device="cpu",
        )

        torch.testing.assert_close(estimated_center, torch.tensor([0.0, 0.0, 3.0]))
        torch.testing.assert_close(explicit_center, torch.tensor([0.0, 0.0, 2.0]))
        self.assertEqual(tuple(grouped.shape), (3, 2, 4, 4))
        self.assertTrue(torch.isfinite(grouped).all())

    def test_load_test_timestamps_from_real_dataset(self):
        frames = camera.load_test_frames(cli.DEFAULT_DATA_PATH, camera_index=0)

        self.assertGreater(len(frames), 0)
        self.assertIsInstance(frames[0].timestamp, float)
        self.assertGreaterEqual(frames[0].index, 0)

    def test_load_pose_camera_shapes_from_real_dataset(self):
        camera_set = camera.load_pose_camera(cli.DEFAULT_DATA_PATH, camera_index=0, resolution=2.0)

        self.assertEqual(camera_set.viewmats.ndim, 3)
        self.assertEqual(camera_set.viewmats.shape[-2:], (4, 4))
        self.assertEqual(camera_set.K.shape, (3, 3))
        self.assertGreater(camera_set.width, 0)
        self.assertGreater(camera_set.height, 0)

    @unittest.skipUnless(importlib.util.find_spec("dahuffman"), "dahuffman is required to unpickle OMG4-FTGS checkpoints")
    def test_default_checkpoint_has_expected_encoded_keys(self):
        keys = model.peek_checkpoint_keys(cli.DEFAULT_CHECKPOINT_PATH)

        self.assertIn("means", keys)
        self.assertIn("MLP_cont", keys)
        self.assertIn("scale_code", keys)

    def test_render_splats_coherent_uses_single_view_cr_lookup(self):
        calls = {}
        fake_output = torch.full((3, 2, 3), 0.75, dtype=torch.float32)
        utils_module = types.ModuleType("coherent_raster.utils.utils_coherent_raster")
        utils_module.unpatchify_image_shape_matrix = lambda image: image
        utils_module.unpad = lambda image, height, width: image[:, :height, :width]
        raster_module = types.ModuleType("gsplat.rendering_coherent_raster")

        def fake_rasterization_cr(**kwargs):
            calls.update(kwargs)
            return fake_output.clone(), None, {}

        raster_module.rasterization_CR = fake_rasterization_cr
        splats = SimpleNamespace(
            means=torch.zeros((2, 3), dtype=torch.float32),
            quats=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            scales=torch.ones((2, 3), dtype=torch.float32),
            opacities=torch.ones(2, dtype=torch.float32),
            colors=torch.zeros((2, 16, 3), dtype=torch.float32),
            sh_degree=3,
        )

        with mock.patch.dict(
            sys.modules,
            {
                "coherent_raster": types.ModuleType("coherent_raster"),
                "coherent_raster.utils": types.ModuleType("coherent_raster.utils"),
                "coherent_raster.utils.utils_coherent_raster": utils_module,
                "gsplat.rendering_coherent_raster": raster_module,
            },
        ):
            image = render.render_splats_coherent(
                splats,
                viewmat=np.eye(4, dtype=np.float32),
                K=np.eye(3, dtype=np.float32),
                width=3,
                height=2,
                device="cpu",
                tile_size=2,
                near_plane=0.01,
                far_plane=100.0,
                camera_model="pinhole",
                debug=False,
            )

        self.assertEqual(tuple(image.shape), (3, 2, 3))
        self.assertEqual(calls["width"], 3)
        self.assertEqual(calls["height"], 2)
        self.assertEqual(calls["tile_size"], 2)
        self.assertEqual(calls["camera_model"], "pinhole")
        self.assertEqual(tuple(calls["adjacent_viewmats"].shape), (1, 1, 4, 4))
        self.assertEqual(tuple(calls["Ks"].shape), (1, 3, 3))
        self.assertTrue(torch.equal(calls["view_idx_matrix"], torch.zeros_like(calls["view_idx_matrix"])))
        self.assertIs(calls["means"], splats.means)
        self.assertIs(calls["quats"], splats.quats)
        self.assertIs(calls["scales"], splats.scales)
        self.assertIs(calls["opacities"], splats.opacities)
        self.assertIs(calls["colors"], splats.colors)
        self.assertEqual(calls["sh_degree"], 3)

    def test_render_splats_interlaced_coherent_uses_grouped_views_and_mapping(self):
        calls = {}
        fake_output = torch.full((3, 3, 4), 0.4, dtype=torch.float32)
        utils_module = types.ModuleType("coherent_raster.utils.utils_coherent_raster")
        utils_module.unpatchify_image_shape_matrix = lambda image: image
        utils_module.unpad = lambda image, height, width: image[:, :height, :width]
        raster_module = types.ModuleType("gsplat.rendering_coherent_raster")

        def fake_rasterization_cr(**kwargs):
            calls.update(kwargs)
            return fake_output.clone(), None, {"timing_ms": {"cr_projection_ms": 1.0}}

        raster_module.rasterization_CR = fake_rasterization_cr
        splats = SimpleNamespace(
            means=torch.zeros((2, 3), dtype=torch.float32),
            quats=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            scales=torch.ones((2, 3), dtype=torch.float32),
            opacities=torch.ones(2, dtype=torch.float32),
            colors=torch.zeros((2, 16, 3), dtype=torch.float32),
            sh_degree=3,
        )
        viewpoint_index = np.zeros((3, 4, 3), dtype=np.int32)
        adjacent_viewmats = torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4).repeat(2, 2, 1, 1)

        with mock.patch.dict(
            sys.modules,
            {
                "coherent_raster": types.ModuleType("coherent_raster"),
                "coherent_raster.utils": types.ModuleType("coherent_raster.utils"),
                "coherent_raster.utils.utils_coherent_raster": utils_module,
                "gsplat.rendering_coherent_raster": raster_module,
            },
        ):
            image = render.render_splats_interlaced_coherent(
                splats,
                adjacent_viewmats=adjacent_viewmats,
                K=np.eye(3, dtype=np.float32),
                viewpoint_index=viewpoint_index,
                width=4,
                height=3,
                device="cpu",
                tile_size=2,
                near_plane=0.01,
                far_plane=100.0,
                camera_model="pinhole",
                debug=False,
            )

        self.assertEqual(tuple(image.shape), (3, 3, 4))
        self.assertEqual(calls["width"], 4)
        self.assertEqual(calls["height"], 3)
        self.assertEqual(tuple(calls["adjacent_viewmats"].shape), (2, 2, 4, 4))
        self.assertEqual(tuple(calls["Ks"].shape), (1, 3, 3))
        self.assertTrue(torch.equal(calls["view_idx_matrix"], torch.zeros_like(calls["view_idx_matrix"])))
        self.assertIs(calls["means"], splats.means)
        self.assertIs(calls["quats"], splats.quats)
        self.assertIs(calls["scales"], splats.scales)
        self.assertIs(calls["opacities"], splats.opacities)
        self.assertIs(calls["colors"], splats.colors)
        self.assertEqual(calls["sh_degree"], 3)

    def test_single_mode_writes_png_and_manifest_with_mock_renderer(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            args = cli.build_parser().parse_args(["--mode", "single", "--output-dir", str(output_dir)])
            fake_model = _FakeDynamicModel()
            fake_camera = _FakeCameraSet()

            with mock.patch.object(cli, "load_dynamic_gaussians", return_value=fake_model), mock.patch.object(
                cli, "load_pose_camera", return_value=fake_camera
            ), mock.patch.object(cli, "load_test_frames", return_value=[camera.TestFrame(0, None, 0.0)]), mock.patch.object(
                cli, "render_splats_gsplat", return_value=torch.full((3, 2, 3), 0.5)
            ):
                result = cli.run(args)

            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "single.png").is_file())
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "single")
            self.assertEqual(manifest["single"]["timestamp"], 0.0)

    def test_single_mode_renderer_both_writes_gsplat_coherent_diff_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            args = cli.build_parser().parse_args(["--mode", "single", "--renderer", "both", "--output-dir", str(output_dir)])
            fake_model = _FakeDynamicModel()
            fake_camera = _FakeCameraSet()

            with mock.patch.object(cli, "load_dynamic_gaussians", return_value=fake_model), mock.patch.object(
                cli, "load_pose_camera", return_value=fake_camera
            ), mock.patch.object(cli, "load_test_frames", return_value=[camera.TestFrame(0, None, 0.0)]), mock.patch.object(
                cli, "render_splats_gsplat", return_value=torch.full((3, 2, 3), 0.2)
            ), mock.patch.object(
                cli, "render_splats_coherent", create=True, return_value=torch.full((3, 2, 3), 0.6)
            ):
                result = cli.run(args)

            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "single_gsplat.png").is_file())
            self.assertTrue((output_dir / "single_coherent.png").is_file())
            self.assertTrue((output_dir / "single_diff.png").is_file())
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["single"]["renderer"], "both")
            self.assertEqual(manifest["single"]["gsplat"]["timestamp"], 0.0)
            self.assertEqual(manifest["single"]["coherent"]["timestamp"], 0.0)
            self.assertIn("gsplat_vs_coherent", manifest["single"]["metrics"])

    def test_interlaced_mode_writes_panel_and_manifest_with_mock_renderer(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            args = cli.build_parser().parse_args(
                [
                    "--mode",
                    "interlaced",
                    "--map-mode",
                    "linear",
                    "--panel-width",
                    "4",
                    "--panel-height",
                    "3",
                    "--views",
                    "4",
                    "--cluster-size",
                    "2",
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            fake_model = _FakeDynamicModel()
            fake_camera = _FakeCameraSet()

            with mock.patch.object(cli, "load_dynamic_gaussians", return_value=fake_model), mock.patch.object(
                cli, "load_pose_camera", return_value=fake_camera
            ), mock.patch.object(cli, "load_test_frames", return_value=[camera.TestFrame(0, None, 0.0)]), mock.patch.object(
                cli, "render_splats_interlaced_coherent", create=True, return_value=torch.full((3, 3, 4), 0.35)
            ) as render_interlaced:
                result = cli.run(args)

            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "omg4_ftgs_lkg_interlaced.png").is_file())
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "interlaced")
            self.assertEqual(manifest["renderer"], "coherent")
            self.assertEqual(manifest["interlaced"]["timestamp"], 0.0)
            self.assertEqual(manifest["interlaced"]["panel_width"], 4)
            self.assertEqual(manifest["interlaced"]["panel_height"], 3)
            self.assertEqual(manifest["interlaced"]["views"], 4)
            self.assertEqual(manifest["interlaced"]["cluster_size"], 2)
            self.assertEqual(manifest["interlaced"]["map_metadata"]["mode"], "linear")
            render_interlaced.assert_called_once()
            self.assertEqual(tuple(render_interlaced.call_args.kwargs["adjacent_viewmats"].shape), (2, 2, 4, 4))
            self.assertEqual(render_interlaced.call_args.kwargs["viewpoint_index"].shape, (3, 4, 3))

    def test_video_mode_writes_frames_and_manifest_with_mock_renderer(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            args = cli.build_parser().parse_args(["--mode", "video", "--output-dir", str(output_dir), "--fps", "12"])
            fake_model = _FakeDynamicModel()
            fake_camera = _FakeCameraSet()
            frames = [
                camera.TestFrame(0, None, 0.0),
                camera.TestFrame(1, None, 0.5),
                camera.TestFrame(2, None, 1.0),
            ]

            with mock.patch.object(cli, "load_dynamic_gaussians", return_value=fake_model), mock.patch.object(
                cli, "load_pose_camera", return_value=fake_camera
            ), mock.patch.object(cli, "load_test_frames", return_value=frames), mock.patch.object(
                cli, "render_splats_gsplat", return_value=torch.full((3, 2, 3), 0.25)
            ), mock.patch.object(
                cli, "_encode_video_from_frames", return_value={"written": False, "path": None, "error": "mock"}
            ):
                result = cli.run(args)

            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "frames" / "frame_0000.png").is_file())
            self.assertTrue((output_dir / "frames" / "frame_0001.png").is_file())
            self.assertTrue((output_dir / "frames" / "frame_0002.png").is_file())
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "video")
            self.assertEqual(manifest["video"]["frame_count"], 3)
            self.assertEqual(manifest["video"]["mp4"]["error"], "mock")


if __name__ == "__main__":
    unittest.main()
