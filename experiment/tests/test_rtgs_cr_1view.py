import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from lkg_experiment.rtgs_coherent import cr_1view
from lkg_experiment.rtgs_coherent.cli import RtgsSnapshot


class RtgsCr1ViewTest(unittest.TestCase):
    def test_default_output_path_is_separate_and_external(self):
        path = cr_1view.default_cr_output_path(
            Path("/data/ysj/result/4dgs/RTGS/jumpingjacks"),
            split="test",
            camera_index=0,
            timestamp=0.0,
            run_label="cr_one_view",
        )

        self.assertIn("rtgs_cr_1view", path.parts)
        self.assertEqual(path.parts[:5], ("/", "data", "ysj", "result", "coherent-raster"))
        self.assertEqual(path.name, "cr_one_view")

    def test_parser_defaults_compare_against_official_render(self):
        args = cr_1view.build_parser().parse_args(["--dataset-kind", "dnerf"])

        self.assertTrue(args.compare_official)
        self.assertTrue(args.rtgs_compat_projection)
        self.assertFalse(args.normalize_explicit_intrinsics_fov)
        self.assertEqual(args.camera_index, 0)
        self.assertEqual(args.background, "auto")

    def test_build_one_view_viewpoint_index_is_all_zero_hwc_subpixel_map(self):
        view_index = cr_1view.build_one_view_viewpoint_index(width=5, height=3)

        self.assertEqual(view_index.shape, (3, 5, 3))
        self.assertEqual(view_index.dtype, np.uint32)
        self.assertFalse(view_index.any())

    def test_snapshot_tensor_diagnostics_records_shapes_and_basic_stats(self):
        snapshot = cr_1view.CrSnapshotForDiagnostics(
            means=torch.tensor([[0.0, 1.0, 2.0], [float("nan"), 0.0, 1.0]]),
            covars=torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0], [float("inf"), 0.0, 0.0, 1.0, 0.0, 1.0]]),
            opacities=torch.tensor([0.25, 0.75]),
            colors=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        )

        diagnostics = cr_1view.snapshot_tensor_diagnostics(snapshot)

        self.assertEqual(diagnostics["means_shape"], [2, 3])
        self.assertEqual(diagnostics["covars_shape"], [2, 6])
        self.assertEqual(diagnostics["opacities_shape"], [2])
        self.assertEqual(diagnostics["colors_shape"], [2, 3])
        self.assertEqual(diagnostics["tensor_nan_inf_counts"]["means_nan"], 1)
        self.assertEqual(diagnostics["tensor_nan_inf_counts"]["covars_inf"], 1)
        self.assertEqual(diagnostics["opacity_min_max_mean"], {"min": 0.25, "max": 0.75, "mean": 0.5})

    def test_main_dispatches_to_cr_renderer(self):
        with mock.patch.object(cr_1view, "render_rtgs_cr_1view", return_value=7) as render:
            result = cr_1view.main(["--dataset-kind", "dnerf"])

        self.assertEqual(result, 7)
        self.assertEqual(render.call_args.args[0].dataset_kind, "dnerf")

    def test_snapshot_proxy_exposes_static_gaussian_contract(self):
        snapshot = cr_1view.CrSnapshotForDiagnostics(
            means=torch.ones((2, 3)),
            covars=torch.ones((2, 6)),
            opacities=torch.tensor([0.1, 0.2]),
            colors=torch.zeros((2, 3)),
        )

        proxy = cr_1view.SnapshotGaussianProxy(snapshot, active_sh_degree=3)

        self.assertEqual(proxy.gaussian_dim, 3)
        self.assertFalse(proxy.rot_4d)
        self.assertEqual(proxy.active_sh_degree, 3)
        torch.testing.assert_close(proxy.get_xyz, torch.ones((2, 3)))
        torch.testing.assert_close(proxy.get_opacity, torch.tensor([[0.1], [0.2]]))
        torch.testing.assert_close(proxy.get_covariance(), torch.ones((2, 6)))

    def test_normalize_explicit_intrinsics_fov_recomputes_nonpositive_fov_without_projection_change(self):
        camera = mock.Mock()
        camera.image_width = 1352
        camera.image_height = 1014
        camera.fl_x = 730.3771702081667
        camera.fl_y = 730.3771702081667
        camera.cx = 676.0
        camera.cy = 507.0
        camera.FoVx = -1.0
        camera.FoVy = -1.0
        camera.projection_matrix = "projection"
        camera.full_proj_transform = "full_projection"

        normalized, info = cr_1view.normalize_explicit_intrinsics_fov(camera, enabled=True)

        self.assertIsNot(normalized, camera)
        self.assertTrue(info["applied"])
        self.assertGreater(normalized.FoVx, 0.0)
        self.assertGreater(normalized.FoVy, 0.0)
        self.assertIs(normalized.projection_matrix, camera.projection_matrix)
        self.assertIs(normalized.full_proj_transform, camera.full_proj_transform)

    def test_normalize_explicit_intrinsics_fov_is_noop_when_disabled(self):
        camera = mock.Mock()

        normalized, info = cr_1view.normalize_explicit_intrinsics_fov(camera, enabled=False)

        self.assertIs(normalized, camera)
        self.assertFalse(info["applied"])

    def test_rtgs_compat_projection_scales_means_and_uses_fov_focal_for_sentinel_camera(self):
        camera = mock.Mock()
        camera.image_width = 4
        camera.image_height = 4
        camera.fl_x = 2.0
        camera.fl_y = 2.0
        camera.cx = 2.0
        camera.cy = 2.0
        camera.FoVx = -1.0
        camera.FoVy = -1.0
        snapshot = RtgsSnapshot(
            means=torch.tensor([[1.0, 2.0, 4.0]]),
            covars=torch.ones((1, 6)),
            opacities=torch.ones(1),
            colors=torch.zeros((1, 3)),
            mask=torch.ones(1, dtype=torch.bool),
        )
        viewmat = torch.eye(4)
        K = torch.tensor([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]])

        adapted_snapshot, adapted_K, info = cr_1view.adapt_snapshot_for_rtgs_compat_projection(
            snapshot,
            camera=camera,
            viewmat=viewmat,
            K=K,
            enabled=True,
        )

        self.assertTrue(info["applied"])
        self.assertLess(adapted_K[0, 0].item(), 0.0)
        self.assertLess(adapted_K[1, 1].item(), 0.0)
        self.assertNotEqual(adapted_snapshot.means[0, 0].item(), snapshot.means[0, 0].item())
        torch.testing.assert_close(adapted_snapshot.covars, snapshot.covars)


if __name__ == "__main__":
    unittest.main()
