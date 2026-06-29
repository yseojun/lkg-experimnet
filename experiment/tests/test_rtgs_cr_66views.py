from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import unittest

import numpy as np
import torch

from lkg_experiment.rtgs_coherent import cr_66views


class RtgsCr66ViewsTest(unittest.TestCase):
    def test_default_output_path_is_external_and_mode_separated(self):
        path = cr_66views.default_cr66_output_path(
            Path("/data/ysj/result/4dgs/RTGS/coffee_martini"),
            split="test",
            camera_index=0,
            timestamp=0.0,
            run_label="cr66",
        )

        self.assertIn("rtgs_cr_66views", path.parts)
        self.assertEqual(path.name, "cr66")

    def test_parser_defaults_save_five_sampled_views_and_all_66_for_interlaced(self):
        args = cr_66views.build_parser().parse_args(["--dataset-kind", "dnerf"])

        self.assertEqual(args.views, 66)
        self.assertEqual(args.sample_save_views, 5)
        self.assertEqual(args.interlace_mode, "compose")
        self.assertTrue(args.write_interlaced)
        self.assertTrue(args.compare_official_sampled)

    def test_evenly_spaced_sample_indices_cover_first_middle_last(self):
        self.assertEqual(cr_66views.resolve_sample_view_indices(66, 5, None), [0, 16, 32, 49, 65])

    def test_explicit_sample_indices_are_validated(self):
        self.assertEqual(cr_66views.resolve_sample_view_indices(66, 5, "0, 7 65"), [0, 7, 65])
        with self.assertRaises(ValueError):
            cr_66views.resolve_sample_view_indices(66, 5, "0, 66")

    def test_validate_viewpoint_index_rejects_out_of_range_view_id(self):
        viewpoint_index = np.zeros((2, 2, 3), dtype=np.int32)
        viewpoint_index[0, 0, 0] = 66

        with self.assertRaises(ValueError):
            cr_66views.validate_viewpoint_index(viewpoint_index, source_view_count=66)

    def test_compose_interlaced_image_uses_all_subpixel_channels(self):
        views = torch.zeros((2, 3, 2, 2), dtype=torch.float32)
        views[0] = 0.25
        views[1] = 0.75
        viewpoint_index = np.zeros((2, 2, 3), dtype=np.int32)
        viewpoint_index[:, :, 1] = 1

        image = cr_66views.compose_interlaced_from_views(views, viewpoint_index)

        self.assertTrue(torch.allclose(image[0], torch.full((2, 2), 0.25)))
        self.assertTrue(torch.allclose(image[1], torch.full((2, 2), 0.75)))
        self.assertTrue(torch.allclose(image[2], torch.full((2, 2), 0.25)))

    def test_synthetic_camera_preserves_n3dv_negative_fov_sentinel(self):
        anchor = SimpleNamespace(
            image_width=1352,
            image_height=1014,
            FoVx=-1.0,
            FoVy=-1.0,
            fl_x=730.0,
            fl_y=730.0,
            cx=676.0,
            cy=507.0,
            image=torch.zeros((3, 1014, 1352), dtype=torch.float32),
        )

        camera = cr_66views.synthetic_camera_from_viewmat_preserving_rtgs_contract(
            anchor,
            torch.eye(4),
            uid=0,
            image_name="view_000",
            timestamp=0.0,
            device="cuda",
            width=1440,
            height=2560,
            crop_to_fill=True,
        )

        self.assertEqual(camera.FoVx, -1.0)
        self.assertEqual(camera.FoVy, -1.0)
        self.assertGreater(camera.fl_x, 0.0)
        self.assertGreater(camera.fl_y, 0.0)

    def test_main_dispatches_to_66view_renderer(self):
        with mock.patch.object(cr_66views, "render_rtgs_cr_66views", return_value=0) as render:
            rc = cr_66views.main(["--dataset-kind", "dnerf"])

        self.assertEqual(rc, 0)
        render.assert_called_once()


if __name__ == "__main__":
    unittest.main()
