import sys
import tempfile
import unittest
from pathlib import Path

import torch

from lkg_experiment.rtgs_coherent import (
    build_parser,
    default_output_path,
    materialize_rtgs_snapshot,
    resolve_rtgs_scene_paths,
    rtgs_camera_to_gsplat_inputs,
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

    def test_default_output_path_includes_scene_split_and_camera(self):
        path = default_output_path(
            Path("/data/ysj/result/4dgs/RTGS/jumpingjacks"),
            split="test",
            camera_index=2,
            timestamp=0.125,
        )

        self.assertEqual(path.name, "jumpingjacks_t0.125000_test_2")

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
        torch.testing.assert_close(snapshot.means[0], torch.tensor([11.0, 0.0, 0.0]))
        torch.testing.assert_close(snapshot.opacities, torch.tensor([0.8]))
        torch.testing.assert_close(snapshot.mask, torch.tensor([True, False]))
        torch.testing.assert_close(captured["dirs"][0], torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(captured["dirs_t"], torch.tensor([[-0.25]]))
        self.assertEqual(captured["deg"], 0)
        self.assertEqual(captured["deg_t"], 0)
        self.assertEqual(captured["duration"], 1.0)

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
            torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
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
