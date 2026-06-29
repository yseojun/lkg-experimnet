import random
import subprocess
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

from lkg_experiment.rtgs_coherent import official_1view


class RtgsOfficial1ViewTest(unittest.TestCase):
    def test_default_output_path_is_mode_separated(self):
        path = official_1view.default_official_output_path(
            Path("/data/ysj/result/4dgs/RTGS/jumpingjacks"),
            split="test",
            camera_index=0,
            timestamp=0.0,
            run_label="baseline",
        )

        self.assertIn("rtgs_official_1view", path.parts)
        self.assertEqual(path.name, "baseline")

    def test_parser_requires_dataset_kind_to_avoid_implicit_fallback(self):
        parser = official_1view.build_parser()
        args = parser.parse_args(["--dataset-kind", "dnerf"])

        self.assertEqual(args.dataset_kind, "dnerf")
        self.assertEqual(args.split, "test")
        self.assertEqual(args.camera_index, 0)
        self.assertEqual(args.background, "auto")

    def test_recursive_config_merge_matches_rtgs_flat_arg_namespace(self):
        args = types.SimpleNamespace(
            gaussian_dim=3,
            time_duration=[-0.5, 0.5],
            sh_degree=3,
            source_path="",
            eval=False,
            compute_cov3D_python=False,
            eval_shfs_4d=False,
            iterations=30_000,
        )

        official_1view.merge_rtgs_config_into_args(
            args,
            {
                "gaussian_dim": 4,
                "time_duration": [0.0, 1.0],
                "ModelParams": {
                    "source_path": "data/dnerf/jumpingjacks",
                    "eval": True,
                },
                "PipelineParams": {
                    "compute_cov3D_python": True,
                    "eval_shfs_4d": True,
                },
                "OptimizationParams": {
                    "iterations": 7_000,
                },
            },
        )

        self.assertEqual(args.gaussian_dim, 4)
        self.assertEqual(args.time_duration, [0.0, 1.0])
        self.assertEqual(args.source_path, "data/dnerf/jumpingjacks")
        self.assertTrue(args.eval)
        self.assertTrue(args.compute_cov3D_python)
        self.assertTrue(args.eval_shfs_4d)
        self.assertEqual(args.iterations, 7_000)

    def test_recursive_config_merge_rejects_unknown_rtgs_arg(self):
        args = types.SimpleNamespace(known=True)

        with self.assertRaisesRegex(AttributeError, "unknown RTGS config key"):
            official_1view.merge_rtgs_config_into_args(args, {"ModelParams": {"missing": 1}})

    def test_n3dv_default_resolution_keeps_rtgs_config_value(self):
        parser = official_1view.build_parser()
        args = parser.parse_args(["--dataset-kind", "n3dv"])
        defaults = types.SimpleNamespace(model_path="", source_path="", data_device="", resolution=2)

        official_1view.apply_harness_arg_overrides(defaults, args, Path("/tmp/source"))

        self.assertEqual(defaults.resolution, 2)

    def test_dnerf_default_resolution_keeps_rtgs_config_value(self):
        parser = official_1view.build_parser()
        args = parser.parse_args(["--dataset-kind", "dnerf"])
        defaults = types.SimpleNamespace(model_path="", source_path="", data_device="", resolution=2)

        official_1view.apply_harness_arg_overrides(defaults, args, Path("/tmp/source"))

        self.assertEqual(defaults.resolution, 2)

    def test_explicit_resolution_overrides_dataset_default(self):
        parser = official_1view.build_parser()
        args = parser.parse_args(["--dataset-kind", "n3dv", "--resolution", "2"])
        defaults = types.SimpleNamespace(model_path="", source_path="", data_device="", resolution=1)

        official_1view.apply_harness_arg_overrides(defaults, args, Path("/tmp/source"))

        self.assertEqual(defaults.resolution, 2)

    def test_n3dv_dynamic_camera_path_is_preferred_when_available(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model"
            dataset_path = root / "dataset"
            model_path.mkdir()
            dataset_path.mkdir()
            (model_path / "cameras.json").write_text("[]", encoding="utf-8")
            (dataset_path / "poses_bounds.npy").write_bytes(b"npy marker")
            scene_paths = official_1view.RtgsScenePaths(
                scene_name="coffee_martini",
                dataset_path=dataset_path,
                config_path=root / "config.yaml",
                dataset_kind="n3dv",
            )

            self.assertTrue(official_1view.should_use_n3dv_dynamic_camera(scene_paths, model_path))

    def test_clean_code_policy_archives_dirty_git_root(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "rtgs"
            root.mkdir()
            (root / "tracked.txt").write_text("clean\n", encoding="utf-8")
            subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "-m",
                    "init",
                ],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            (root / "tracked.txt").write_text("dirty\n", encoding="utf-8")

            effective_root, status = official_1view.resolve_effective_rtgs_code_root(
                root,
                policy="clean",
                cache_root=Path(tmp) / "cache",
            )

            self.assertNotEqual(effective_root, root.resolve())
            self.assertTrue(status["dirty"])
            self.assertEqual(status["effective_mode"], "clean_snapshot")
            self.assertEqual((effective_root / "tracked.txt").read_text(encoding="utf-8"), "clean\n")

    def test_as_is_code_policy_keeps_dirty_git_root(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "rtgs"
            root.mkdir()
            (root / "tracked.txt").write_text("clean\n", encoding="utf-8")
            subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "-m",
                    "init",
                ],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            (root / "tracked.txt").write_text("dirty\n", encoding="utf-8")

            effective_root, status = official_1view.resolve_effective_rtgs_code_root(
                root,
                policy="as-is",
                cache_root=Path(tmp) / "cache",
            )

            self.assertEqual(effective_root, root.resolve())
            self.assertTrue(status["dirty"])
            self.assertEqual(status["effective_mode"], "as_is")

    def test_render_contract_tensor_shapes_records_model_and_camera_inputs(self):
        import torch

        gaussians = types.SimpleNamespace(
            _xyz=torch.zeros(5, 3),
            _features_dc=torch.zeros(5, 1, 3),
            _features_rest=torch.zeros(5, 47, 3),
            _scaling=torch.zeros(5, 3),
            _rotation=torch.zeros(5, 4),
            _opacity=torch.zeros(5, 1),
            _t=torch.zeros(5, 1),
            _scaling_t=torch.zeros(5, 1),
            _rotation_r=torch.zeros(5, 4),
            env_map=torch.zeros(3, 500, 500),
        )
        camera = types.SimpleNamespace(
            world_view_transform=torch.zeros(4, 4),
            full_proj_transform=torch.zeros(4, 4),
            camera_center=torch.zeros(3),
        )
        background = torch.zeros(3)

        shapes = official_1view.render_contract_tensor_shapes(gaussians, camera, background)

        self.assertEqual(shapes["gaussians"]["_xyz"], [5, 3])
        self.assertEqual(shapes["gaussians"]["_features_rest"], [5, 47, 3])
        self.assertEqual(shapes["gaussians"]["_t"], [5, 1])
        self.assertEqual(shapes["gaussians"]["env_map"], [3, 500, 500])
        self.assertEqual(shapes["camera"]["world_view_transform"], [4, 4])
        self.assertEqual(shapes["background"], [3])

    def test_camera_render_contract_values_records_intrinsics_and_timestamp(self):
        camera = types.SimpleNamespace(
            image_width=1352,
            image_height=1014,
            FoVx=1.492,
            FoVy=1.214,
            fl_x=730.377,
            fl_y=730.377,
            cx=676.0,
            cy=507.0,
            znear=0.01,
            zfar=100.0,
            timestamp=0.0,
        )

        values = official_1view.camera_render_contract_values(camera)

        self.assertEqual(values["image_width"], 1352)
        self.assertEqual(values["image_height"], 1014)
        self.assertEqual(values["fl_x"], 730.377)
        self.assertEqual(values["cx"], 676.0)
        self.assertEqual(values["timestamp"], 0.0)

    def test_gaussian_model_kwargs_omit_prefilter_when_constructor_does_not_support_it(self):
        class GaussianModelWithoutPrefilter:
            def __init__(
                self,
                sh_degree,
                *,
                gaussian_dim,
                time_duration,
                rot_4d,
                force_sh_3d,
                sh_degree_t,
            ):
                pass

        model_args = types.SimpleNamespace(prefilter_var=0.125)
        pipeline_args = types.SimpleNamespace(eval_shfs_4d=True)
        defaults = types.SimpleNamespace(gaussian_dim=4, rot_4d=True, force_sh_3d=False)

        kwargs = official_1view.gaussian_model_constructor_kwargs(
            GaussianModelWithoutPrefilter,
            model_args=model_args,
            pipeline_args=pipeline_args,
            defaults=defaults,
            time_duration=[0.0, 10.0],
        )

        self.assertNotIn("prefilter_var", kwargs)
        self.assertEqual(kwargs["gaussian_dim"], 4)

    def test_gaussian_model_kwargs_include_prefilter_when_constructor_supports_it(self):
        class GaussianModelWithPrefilter:
            def __init__(
                self,
                sh_degree,
                *,
                gaussian_dim,
                time_duration,
                rot_4d,
                force_sh_3d,
                sh_degree_t,
                prefilter_var,
            ):
                pass

        model_args = types.SimpleNamespace(prefilter_var=0.125)
        pipeline_args = types.SimpleNamespace(eval_shfs_4d=True)
        defaults = types.SimpleNamespace(gaussian_dim=4, rot_4d=True, force_sh_3d=False)

        kwargs = official_1view.gaussian_model_constructor_kwargs(
            GaussianModelWithPrefilter,
            model_args=model_args,
            pipeline_args=pipeline_args,
            defaults=defaults,
            time_duration=[0.0, 10.0],
        )

        self.assertEqual(kwargs["prefilter_var"], 0.125)
        self.assertEqual(kwargs["sh_degree_t"], 2)

    def test_output_dir_without_run_label_is_unique(self):
        first = official_1view.default_official_output_path(
            Path("/data/ysj/result/4dgs/RTGS/jumpingjacks"),
            split="test",
            camera_index=0,
            timestamp=0.0,
            run_label=None,
            unique_label="20260629_010203",
        )
        second = official_1view.default_official_output_path(
            Path("/data/ysj/result/4dgs/RTGS/jumpingjacks"),
            split="test",
            camera_index=0,
            timestamp=0.0,
            run_label=None,
            unique_label="20260629_010204",
        )

        self.assertNotEqual(first, second)
        self.assertEqual(first.name, "jumpingjacks_t0.000000_test_0_20260629_010203")

    def test_lkg_display_status_records_disconnected_image_artifact(self):
        status = official_1view.lkg_display_status(
            connected=False,
            image_path=Path("/tmp/render.png"),
        )

        self.assertFalse(status["connected"])
        self.assertEqual(status["validation_mode"], "image_artifact_only")
        self.assertEqual(status["single_view_image_path"], "/tmp/render.png")

    def test_resolve_scene_paths_uses_explicit_dataset_kind_without_fallback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rtgs_root = root / "rtgs"
            dnerf_root = root / "dnerf"
            n3dv_root = root / "n3dv"
            (rtgs_root / "configs" / "dnerf").mkdir(parents=True)
            (rtgs_root / "configs" / "dynerf").mkdir(parents=True)
            (rtgs_root / "configs" / "dnerf" / "jumpingjacks.yaml").write_text("ModelParams:\n", encoding="utf-8")
            (rtgs_root / "configs" / "dynerf" / "jumpingjacks.yaml").write_text("ModelParams:\n", encoding="utf-8")
            (dnerf_root / "jumpingjacks").mkdir(parents=True)
            (dnerf_root / "jumpingjacks" / "transforms_train.json").write_text("{}", encoding="utf-8")
            (n3dv_root / "jumpingjacks").mkdir(parents=True)
            (n3dv_root / "jumpingjacks" / "poses_bounds.npy").write_text("raw marker", encoding="utf-8")
            (n3dv_root / "jumpingjacks" / "colmap" / "sparse").mkdir(parents=True)

            dnerf_paths, dnerf_source = official_1view.resolve_official_scene_paths(
                root / "models" / "jumpingjacks",
                rtgs_code_root=rtgs_root,
                dataset_kind="dnerf",
                dataset_root=dnerf_root,
                n3dv_root=n3dv_root,
                source_path=None,
                config_path=None,
            )
            n3dv_paths, n3dv_source = official_1view.resolve_official_scene_paths(
                root / "models" / "jumpingjacks",
                rtgs_code_root=rtgs_root,
                dataset_kind="n3dv",
                dataset_root=dnerf_root,
                n3dv_root=n3dv_root,
                source_path=None,
                config_path=None,
            )

            self.assertEqual(dnerf_paths.dataset_kind, "dnerf")
            self.assertEqual(dnerf_source, dnerf_root / "jumpingjacks")
            self.assertEqual(n3dv_paths.dataset_kind, "n3dv")
            self.assertEqual(n3dv_source, n3dv_root / "jumpingjacks" / "colmap")

    def test_scene_source_layout_must_be_official_scene_compatible(self):
        with TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "raw_n3dv"
            source_path.mkdir()

            with self.assertRaisesRegex(ValueError, "transforms_train.json.*sparse"):
                official_1view.validate_official_scene_source_path(source_path)

    def test_scene_shuffle_seed_matches_rtgs_safe_state_random_seed(self):
        official_1view.seed_official_scene_shuffle()
        first = list(range(5))
        random.shuffle(first)

        official_1view.seed_official_scene_shuffle()
        second = list(range(5))
        random.shuffle(second)

        self.assertEqual(first, second)

    def test_skip_scene_model_population_handles_missing_load_ply(self):
        class ModelWithoutLoadPly:
            def create_from_pcd(self):
                raise AssertionError("should be patched")

            def create_from_pth(self):
                raise AssertionError("should be patched")

        self.assertFalse(hasattr(ModelWithoutLoadPly, "load_ply"))
        with official_1view._skip_scene_model_population(ModelWithoutLoadPly):
            instance = ModelWithoutLoadPly()
            instance.create_from_pcd("pcd", 3.5)
            instance.create_from_pth("path", 2.0)
            instance.load_ply("path")
            self.assertEqual(instance.spatial_lr_scale, 2.0)

        self.assertFalse(hasattr(ModelWithoutLoadPly, "load_ply"))

    def test_colmap_camera_info_patch_adds_depth_default(self):
        class CameraInfo(NamedTuple):
            uid: int
            R: object
            T: object
            FovY: float
            FovX: float
            image: object
            depth: object
            image_path: str
            image_name: str
            width: int
            height: int

        readers = types.SimpleNamespace(CameraInfo=CameraInfo)
        official_1view.patch_colmap_camera_info_depth_default(readers)

        camera = readers.CameraInfo(
            uid=1,
            R=None,
            T=None,
            FovY=1.0,
            FovX=1.0,
            image=None,
            image_path="/tmp/image.png",
            image_name="image",
            width=640,
            height=480,
        )

        self.assertIsNone(camera.depth)
        self.assertIsInstance(camera, CameraInfo)


if __name__ == "__main__":
    unittest.main()
