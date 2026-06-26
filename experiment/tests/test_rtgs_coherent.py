import sys
import tempfile
import types
import unittest
import json
from pathlib import Path

import numpy as np
import torch

import lkg_experiment.rtgs_coherent.cli as rtgs_cli
from lkg_experiment.rtgs_coherent.cli import (
    RtgsLiteGaussianModel,
    _install_pointops_import_stub_if_needed,
    _install_scene_gaussian_model_stub,
    _load_rtgs_colmap_camera_lite,
    _load_rtgs_n3dv_dynamic_camera,
    _count_rtgs_n3dv_dynamic_cameras,
    _pipeline_namespace_for_model,
)
from lkg_experiment.rtgs_coherent import (
    build_parser,
    default_output_path,
    materialize_rtgs_snapshot,
    resolve_rtgs_scene_paths,
    rtgs_camera_from_gsplat_viewmat,
    rtgs_camera_to_gsplat_inputs,
)
from lkg_experiment.rtgs_coherent.views66 import (
    default_views66_output_path,
    sample_evenly_spaced_view_indices,
    scale_intrinsics_to_resolution,
)


class RtgsCoherentTest(unittest.TestCase):
    def test_parser_defaults_to_rtgs_paths_and_jumpingjacks(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.model_path, "/data/ysj/result/4dgs/RTGS/jumpingjacks")
        self.assertEqual(args.checkpoint, "checkpoints/chkpnt_best.pth")
        self.assertEqual(args.rtgs_code_root, "/home/ysj/lkg-experiment/4d-gaussian-splatting")
        self.assertEqual(args.gsplat_root, "/home/ysj/lkg-experiment/gsplat")
        self.assertEqual(args.dataset_root, "/data/ysj/dataset/dnerf")
        self.assertEqual(args.split, "test")
        self.assertEqual(args.camera_index, 0)
        self.assertEqual(args.n3dv_frame_index, 0)
        self.assertEqual(args.render_mode, "single")
        self.assertEqual(args.checkpoint_load_device, "cpu")
        self.assertEqual(args.views, 66)
        self.assertEqual(args.cluster_size, 8)
        self.assertEqual(args.compare_original_views, 5)
        self.assertEqual(args.width, 1440)
        self.assertEqual(args.height, 2560)
        self.assertFalse(args.no_crop_to_fill)
        self.assertTrue(args.write_interlaced)

    def test_parser_accepts_views66_mode(self):
        args = build_parser().parse_args(
            [
                "--render-mode",
                "views66",
                "--views",
                "66",
                "--cluster-size",
                "4",
                "--compare-original-views",
                "0",
                "--split",
                "all",
                "--map-mode",
                "linear",
                "--no-write-per-view",
                "--no-write-interlaced",
            ]
        )

        self.assertEqual(args.render_mode, "views66")
        self.assertEqual(args.views, 66)
        self.assertEqual(args.cluster_size, 4)
        self.assertEqual(args.compare_original_views, 0)
        self.assertEqual(args.split, "all")
        self.assertEqual(args.map_mode, "linear")
        self.assertFalse(args.write_per_view)
        self.assertFalse(args.write_interlaced)

    def test_parser_accepts_single_all_cams_mode(self):
        args = build_parser().parse_args(
            [
                "--render-mode",
                "single-all-cams",
                "--split",
                "all",
                "--camera-indices",
                "0,2,4",
                "--checkpoint-load-device",
                "cuda",
            ]
        )

        self.assertEqual(args.render_mode, "single-all-cams")
        self.assertEqual(args.split, "all")
        self.assertEqual(args.camera_indices, "0,2,4")
        self.assertEqual(args.checkpoint_load_device, "cuda")

    def test_default_output_path_includes_scene_split_and_camera(self):
        path = default_output_path(
            Path("/data/ysj/result/4dgs/RTGS/jumpingjacks"),
            split="test",
            camera_index=2,
            timestamp=0.125,
        )

        self.assertEqual(path.name, "jumpingjacks_t0.125000_test_2")

    def test_default_views66_output_path_suffixes_view_count(self):
        path = default_views66_output_path(
            Path("/data/ysj/result/4dgs/RTGS/jumpingjacks"),
            split="test",
            camera_index=2,
            timestamp=0.125,
            views=66,
        )

        self.assertEqual(path.name, "jumpingjacks_t0.125000_test_2_views66")

    def test_sample_evenly_spaced_view_indices_includes_edges(self):
        self.assertEqual(sample_evenly_spaced_view_indices(66, 5), [0, 16, 32, 48, 65])
        self.assertEqual(sample_evenly_spaced_view_indices(4, 10), [0, 1, 2, 3])
        self.assertEqual(sample_evenly_spaced_view_indices(4, 0), [])

    def test_scale_intrinsics_to_resolution_uses_crop_to_fill(self):
        K = torch.tensor([[100.0, 0.0, 32.0], [0.0, 120.0, 24.0], [0.0, 0.0, 1.0]])

        cropped = scale_intrinsics_to_resolution(
            K,
            source_width=64,
            source_height=48,
            target_width=128,
            target_height=192,
            crop_to_fill=True,
        )
        fitted = scale_intrinsics_to_resolution(
            K,
            source_width=64,
            source_height=48,
            target_width=128,
            target_height=192,
            crop_to_fill=False,
        )

        torch.testing.assert_close(cropped, torch.tensor([[400.0, 0.0, 64.0], [0.0, 480.0, 96.0], [0.0, 0.0, 1.0]]))
        torch.testing.assert_close(fitted, torch.tensor([[200.0, 0.0, 64.0], [0.0, 240.0, 96.0], [0.0, 0.0, 1.0]]))

    def test_resolve_scene_paths_maps_dnerf_scene_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "result" / "RTGS" / "jumpingjacks"
            dataset_root = root / "dataset" / "dnerf"
            dataset_path = dataset_root / "jumpingjacks"
            config_path = root / "code" / "configs" / "dnerf" / "jumpingjacks.yaml"
            model_path.mkdir(parents=True)
            dataset_path.mkdir(parents=True)
            config_path.parent.mkdir(parents=True)
            config_path.write_text("gaussian_dim: 4\n", encoding="utf-8")

            paths = resolve_rtgs_scene_paths(
                model_path,
                rtgs_code_root=root / "code",
                dataset_root=dataset_root,
                n3dv_root=root / "dataset" / "N3DV",
            )

        self.assertEqual(paths.scene_name, "jumpingjacks")
        self.assertEqual(paths.dataset_path, dataset_path)
        self.assertEqual(paths.config_path, config_path)
        self.assertEqual(paths.dataset_kind, "dnerf")

    def test_resolve_scene_paths_maps_flame_salmon_to_n3dv_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "result" / "RTGS" / "flame_salmon"
            n3dv_root = root / "dataset" / "N3DV"
            dataset_path = n3dv_root / "flame_salmon_1"
            config_path = root / "code" / "configs" / "dynerf" / "flame_salmon.yaml"
            model_path.mkdir(parents=True)
            dataset_path.mkdir(parents=True)
            config_path.parent.mkdir(parents=True)
            config_path.write_text("gaussian_dim: 4\n", encoding="utf-8")

            paths = resolve_rtgs_scene_paths(
                model_path,
                rtgs_code_root=root / "code",
                dataset_root=root / "dataset" / "dnerf",
                n3dv_root=n3dv_root,
            )

        self.assertEqual(paths.dataset_path, dataset_path)
        self.assertEqual(paths.dataset_kind, "n3dv")

    def test_materialize_snapshot_filters_time_and_uses_original_xyz_for_color(self):
        pc = _FakeRtgsModel()
        captured = {}

        def fake_eval_shfs_4d(deg, deg_t, shs, dirs, dirs_t, duration):
            captured["deg"] = deg
            captured["deg_t"] = deg_t
            captured["dirs"] = dirs.clone()
            captured["dirs_t"] = dirs_t.clone()
            captured["duration"] = duration
            return torch.zeros((shs.shape[0], 3), dtype=shs.dtype)

        snapshot = materialize_rtgs_snapshot(
            pc,
            timestamp=0.25,
            camera_center=torch.zeros(3),
            eval_shfs_4d_fn=fake_eval_shfs_4d,
        )

        self.assertEqual(tuple(snapshot.means.shape), (1, 3))
        self.assertEqual(tuple(snapshot.covars.shape), (1, 6))
        self.assertEqual(tuple(snapshot.colors.shape), (1, 3))
        torch.testing.assert_close(snapshot.means[0], torch.tensor([1.0, 10.0, 0.0]))
        torch.testing.assert_close(snapshot.opacities, torch.tensor([0.8]))
        torch.testing.assert_close(snapshot.mask, torch.tensor([True, False]))
        torch.testing.assert_close(captured["dirs"][0], torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(captured["dirs_t"], torch.tensor([[-0.25]]))
        self.assertEqual(captured["deg"], 0)
        self.assertEqual(captured["deg_t"], 0)
        self.assertEqual(captured["duration"], 1.0)

    def test_materialize_snapshot_accepts_batched_camera_centers(self):
        pc = _FakeRtgsModel()
        seen_dirs = []

        def fake_eval_shfs_4d(deg, deg_t, shs, dirs, dirs_t, duration):
            seen_dirs.append(dirs.clone())
            base = torch.arange(shs.shape[0], dtype=shs.dtype).view(-1, 1)
            return torch.cat([base, base + dirs[:, :1], base + dirs[:, 1:2]], dim=1)

        snapshot = materialize_rtgs_snapshot(
            pc,
            timestamp=0.25,
            camera_center=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            eval_shfs_4d_fn=fake_eval_shfs_4d,
        )

        self.assertEqual(tuple(snapshot.colors.shape), (2, 1, 3))
        self.assertEqual(len(seen_dirs), 2)
        torch.testing.assert_close(snapshot.colors[0, 0], torch.tensor([0.5, 1.5, 0.5]))
        torch.testing.assert_close(snapshot.colors[1, 0], torch.tensor([0.5, 0.0, 0.5]))

    def test_rtgs_camera_to_gsplat_inputs_uses_transposed_world_view_transform(self):
        camera = _FakeCamera()
        viewmat, K = rtgs_camera_to_gsplat_inputs(camera, device="cpu")

        expected_viewmat = camera.world_view_transform.transpose(0, 1)
        torch.testing.assert_close(viewmat, expected_viewmat)
        torch.testing.assert_close(
            K,
            torch.tensor(
                [[100.0, 0.0, 32.0], [0.0, 120.0, 24.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
        )

    def test_rtgs_camera_from_gsplat_viewmat_preserves_intrinsics_and_pose(self):
        anchor = _FakeCamera()
        viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 1.0, 0.0, 3.0],
                [0.0, 0.0, 1.0, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        camera = rtgs_camera_from_gsplat_viewmat(
            anchor,
            viewmat,
            uid=7,
            image_name="view_007",
            timestamp=0.25,
            device="cpu",
        )
        converted_viewmat, K = rtgs_camera_to_gsplat_inputs(camera, device="cpu")

        torch.testing.assert_close(converted_viewmat, viewmat)
        torch.testing.assert_close(K, torch.tensor([[100.0, 0.0, 32.0], [0.0, 120.0, 24.0], [0.0, 0.0, 1.0]]))
        self.assertEqual(camera.uid, 7)
        self.assertEqual(camera.image_name, "view_007")
        self.assertEqual(camera.timestamp, 0.25)

    def test_rtgs_camera_from_gsplat_viewmat_can_scale_intrinsics(self):
        anchor = _FakeCamera()
        viewmat = torch.eye(4)

        camera = rtgs_camera_from_gsplat_viewmat(
            anchor,
            viewmat,
            uid=1,
            image_name="view_001",
            timestamp=0.0,
            device="cpu",
            width=128,
            height=96,
        )
        _, K = rtgs_camera_to_gsplat_inputs(camera, device="cpu")

        self.assertEqual(camera.image_width, 128)
        self.assertEqual(camera.image_height, 96)
        torch.testing.assert_close(K, torch.tensor([[200.0, 0.0, 64.0], [0.0, 240.0, 48.0], [0.0, 0.0, 1.0]]))

    def test_cr_color_helper_accepts_precomputed_rgb(self):
        gsplat_root = Path(__file__).resolve().parents[2] / "gsplat"
        sys.path.insert(0, str(gsplat_root))
        try:
            from gsplat.rendering_coherent_raster import _prepare_colors_for_cr

            colors = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            prepared = _prepare_colors_for_cr(
                colors,
                sh_degree=None,
                reference_viewmat_count=3,
                dirs=torch.zeros((3, 2, 3)),
                masks=torch.ones((3, 2), dtype=torch.bool),
            )
        finally:
            sys.path.pop(0)

        self.assertEqual(tuple(prepared.shape), (3, 2, 3))
        torch.testing.assert_close(prepared[2], colors)

    def test_rtgs_lite_gaussian_model_to_moves_tensor_attributes(self):
        model = RtgsLiteGaussianModel(3)
        model.active_sh_degree = 1
        model._xyz = torch.ones((1, 3))
        model._features_dc = torch.ones((1, 1, 3))
        model._features_rest = torch.ones((1, 3, 3))
        model._scaling = torch.ones((1, 3))
        model._rotation = torch.ones((1, 4))
        model._opacity = torch.ones((1, 1))
        model.max_radii2D = torch.ones((1,))
        model.xyz_gradient_accum = torch.ones((1, 1))
        model.t_gradient_accum = torch.ones((1, 1))
        model.denom = torch.ones((1, 1))
        model._t = torch.ones((1, 1))
        model._scaling_t = torch.ones((1, 1))
        model._rotation_r = torch.ones((1, 4))
        model.env_map = torch.ones((1, 3))

        moved = model.to("cpu")

        self.assertIs(moved, model)
        self.assertEqual(model._xyz.device.type, "cpu")
        self.assertEqual(model.env_map.device.type, "cpu")

    def test_pipeline_namespace_disables_env_map_when_checkpoint_has_none(self):
        model = types.SimpleNamespace(env_map=None)

        pipe = _pipeline_namespace_for_model({"env_map_res": 500}, model)

        self.assertEqual(pipe.env_map_res, 0)

    def test_rtgs_gaussian_model_stub_reexports_basic_point_cloud(self):
        originals = {
            name: sys.modules.get(name)
            for name in ("scene", "scene.gaussian_model", "utils", "utils.graphics_utils")
        }
        sentinel_basic_point_cloud = object()
        try:
            for name in originals:
                sys.modules.pop(name, None)
            utils_module = types.ModuleType("utils")
            utils_module.__path__ = []
            graphics_module = types.ModuleType("utils.graphics_utils")
            graphics_module.BasicPointCloud = sentinel_basic_point_cloud
            sys.modules["utils"] = utils_module
            sys.modules["utils.graphics_utils"] = graphics_module

            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "scene").mkdir()
                _install_scene_gaussian_model_stub(root)
                from scene.gaussian_model import BasicPointCloud, GaussianModel

            self.assertIs(BasicPointCloud, sentinel_basic_point_cloud)
            self.assertTrue(callable(GaussianModel))
        finally:
            for name, module in originals.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_pointops_import_stub_allows_general_utils_import_without_cuda_extension(self):
        originals = {
            name: sys.modules.get(name)
            for name in (
                "pointops2",
                "pointops2.functions",
                "pointops2.functions.pointops",
                "pointops2_cuda",
            )
        }
        try:
            for name in originals:
                sys.modules.pop(name, None)
            _install_pointops_import_stub_if_needed()
            from pointops2.functions.pointops import furthestsampling, knnquery

            with self.assertRaisesRegex(RuntimeError, "pointops2_cuda"):
                furthestsampling(None, None, None)
            with self.assertRaisesRegex(RuntimeError, "pointops2_cuda"):
                knnquery(None, None, None, None, None)
        finally:
            for name, module in originals.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_colmap_lite_loader_does_not_import_original_dataset_reader(self):
        originals = {
            name: sys.modules.get(name)
            for name in ("scene", "scene.colmap_loader", "scene.dataset_readers")
        }
        try:
            for name in originals:
                sys.modules.pop(name, None)
            scene_module = types.ModuleType("scene")
            scene_module.__path__ = []
            colmap_loader = types.ModuleType("scene.colmap_loader")
            colmap_loader.qvec2rotmat = lambda _qvec: np.eye(3, dtype=np.float64)
            colmap_loader.read_extrinsics_binary = lambda _path: {
                1: types.SimpleNamespace(
                    qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                    tvec=np.array([1.0, 2.0, 3.0]),
                    camera_id=7,
                    name="r_000.png",
                ),
                2: types.SimpleNamespace(
                    qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                    tvec=np.array([4.0, 5.0, 6.0]),
                    camera_id=7,
                    name="r_001.png",
                ),
            }
            colmap_loader.read_extrinsics_text = colmap_loader.read_extrinsics_binary
            colmap_loader.read_intrinsics_binary = lambda _path: {
                7: types.SimpleNamespace(
                    id=7,
                    model="PINHOLE",
                    width=64,
                    height=48,
                    params=np.array([100.0, 120.0, 32.0, 24.0]),
                )
            }
            colmap_loader.read_intrinsics_text = colmap_loader.read_intrinsics_binary
            sys.modules["scene"] = scene_module
            sys.modules["scene.colmap_loader"] = colmap_loader

            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source_path = root / "colmap"
                images_path = source_path / "images"
                (source_path / "sparse" / "0").mkdir(parents=True)
                images_path.mkdir(parents=True)
                from PIL import Image

                Image.new("RGB", (64, 48), color=(255, 0, 0)).save(images_path / "r_000.png")
                Image.new("RGB", (64, 48), color=(0, 255, 0)).save(images_path / "r_001.png")
                args = types.SimpleNamespace(
                    source_path=str(source_path),
                    images="images",
                    eval=True,
                    resolution=1,
                    white_background=False,
                    data_device="cpu",
                )

                gt, camera = _load_rtgs_colmap_camera_lite(
                    args=args,
                    split="all",
                    camera_index=1,
                    device="cpu",
                )

            self.assertNotIn("scene.dataset_readers", sys.modules)
            self.assertEqual(tuple(gt.shape), (3, 48, 64))
            self.assertEqual(camera.image_name, "r_001")
            self.assertEqual(camera.image_width, 64)
            self.assertEqual(camera.image_height, 48)
            self.assertEqual(camera.fl_x, 100.0)
            self.assertEqual(camera.fl_y, 120.0)
            self.assertEqual(camera.cx, 32.0)
            self.assertEqual(camera.cy, 24.0)
        finally:
            for name, module in originals.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_n3dv_dynamic_loader_uses_training_cameras_json_and_frame_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model"
            source_path = root / "coffee_martini"
            image_path = source_path / "cam00" / "images"
            model_path.mkdir()
            image_path.mkdir(parents=True)
            from PIL import Image

            Image.new("RGB", (1352, 1014), color=(64, 128, 192)).save(image_path / "0000.png")
            cameras = [
                {
                    "id": 0,
                    "img_name": "cam00_0000",
                    "width": 2704,
                    "height": 2028,
                    "position": [3.0, 4.0, 5.0],
                    "rotation": [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    "fx": -1.0,
                    "fy": -1.0,
                }
            ]
            (model_path / "cameras.json").write_text(json.dumps(cameras), encoding="utf-8")
            poses_bounds = np.zeros((1, 17), dtype=np.float64)
            poses_bounds[0, :15] = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 2028.0],
                    [0.0, 1.0, 0.0, 0.0, 2704.0],
                    [0.0, 0.0, 1.0, 0.0, 1460.0],
                ]
            ).reshape(-1)
            np.save(source_path / "poses_bounds.npy", poses_bounds)
            args = types.SimpleNamespace(
                source_path=str(source_path),
                model_path=str(model_path),
                resolution=2,
                white_background=False,
                data_device="cpu",
            )

            gt, camera = _load_rtgs_n3dv_dynamic_camera(
                args=args,
                split="all",
                camera_index=0,
                time_duration=[0.0, 10.0],
                device="cpu",
            )

        self.assertEqual(camera.image_name, "cam00_0000")
        self.assertEqual(camera.timestamp, 0.0)
        self.assertEqual(camera.image_width, 1352)
        self.assertEqual(camera.image_height, 1014)
        self.assertEqual(tuple(gt.shape), (3, 1014, 1352))
        self.assertEqual(camera.fl_x, 730.0)
        self.assertEqual(camera.fl_y, 730.0)
        self.assertEqual(camera.cx, 676.0)
        self.assertEqual(camera.cy, 507.0)
        torch.testing.assert_close(camera.camera_center, torch.tensor([3.0, 4.0, 5.0]), atol=1e-5, rtol=1e-5)

    def test_camera_manifest_fields_include_n3dv_frame_identity(self):
        camera = types.SimpleNamespace(
            uid=7,
            image_name="cam02_0150",
            image_width=1352,
            image_height=1014,
            timestamp=5.0,
        )

        self.assertTrue(hasattr(rtgs_cli, "_camera_manifest_fields"))
        fields = rtgs_cli._camera_manifest_fields(camera)

        self.assertEqual(
            fields,
            {
                "camera_uid": 7,
                "camera_image_name": "cam02_0150",
                "camera_timestamp": 5.0,
                "n3dv_camera_label": "cam02",
                "n3dv_frame_index": 150,
            },
        )

    def test_n3dv_dynamic_count_uses_cameras_json_frame_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model"
            source_path = root / "coffee_martini"
            model_path.mkdir()
            source_path.mkdir()
            cameras = [
                {"id": 0, "img_name": "cam00_0000", "width": 2704, "height": 2028, "position": [0, 0, 0], "rotation": np.eye(3).tolist()},
                {"id": 1, "img_name": "cam00_0001", "width": 2704, "height": 2028, "position": [0, 0, 0], "rotation": np.eye(3).tolist()},
                {"id": 2, "img_name": "cam01_0000", "width": 2704, "height": 2028, "position": [1, 0, 0], "rotation": np.eye(3).tolist()},
            ]
            (model_path / "cameras.json").write_text(json.dumps(cameras), encoding="utf-8")
            args = types.SimpleNamespace(source_path=str(source_path), model_path=str(model_path))

            count = _count_rtgs_n3dv_dynamic_cameras(args=args, split="all", time_duration=[0.0, 10.0])

        self.assertEqual(count, 2)


class _FakeRtgsModel:
    gaussian_dim = 4
    rot_4d = True
    force_sh_3d = False
    active_sh_degree = 0
    active_sh_degree_t = 0
    time_duration = [0.0, 1.0]

    @property
    def get_xyz(self):
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    @property
    def get_t(self):
        return torch.tensor([[0.0], [0.9]])

    @property
    def get_opacity(self):
        return torch.tensor([[0.8], [0.5]])

    @property
    def get_features(self):
        return torch.zeros((2, 1, 3))

    @property
    def get_max_sh_channels(self):
        return 1

    def get_current_covariance_and_mean_offset(self, scaling_modifier=1.0, timestamp=0.0):
        return (
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                    [2.0, 0.0, 0.0, 2.0, 0.0, 2.0],
                ]
            ),
            torch.tensor([[0.0, 10.0, 0.0], [20.0, 0.0, 0.0]]),
        )

    def get_marginal_t(self, timestamp, scaling_modifier=1.0):
        return torch.tensor([[1.0], [0.01]])


class _FakeCamera:
    image_width = 64
    image_height = 48
    fl_x = 100.0
    fl_y = 120.0
    cx = 32.0
    cy = 24.0

    world_view_transform = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [3.0, 4.0, 5.0, 1.0],
        ],
        dtype=torch.float32,
    )


if __name__ == "__main__":
    unittest.main()
