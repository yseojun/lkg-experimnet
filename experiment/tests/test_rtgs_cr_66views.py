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
        self.assertEqual(args.aspect_fit, "contain")
        self.assertEqual(args.camera_aspect_mode, "expand")

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

    def test_resolve_aspect_viewport_contains_n3dv_inside_lkg_panel(self):
        viewport = cr_66views.resolve_aspect_viewport(
            source_width=1352,
            source_height=1014,
            target_width=1440,
            target_height=2560,
            aspect_fit="contain",
        )

        self.assertEqual(viewport.panel_width, 1440)
        self.assertEqual(viewport.panel_height, 2560)
        self.assertEqual(viewport.render_width, 1440)
        self.assertEqual(viewport.render_height, 1080)
        self.assertEqual(viewport.offset_x, 0)
        self.assertEqual(viewport.offset_y, 740)
        self.assertAlmostEqual(viewport.scale, 1440.0 / 1352.0)

    def test_resolve_aspect_viewport_contains_square_dnerf_inside_lkg_panel(self):
        viewport = cr_66views.resolve_aspect_viewport(
            source_width=400,
            source_height=400,
            target_width=1440,
            target_height=2560,
            aspect_fit="contain",
        )

        self.assertEqual(viewport.render_width, 1440)
        self.assertEqual(viewport.render_height, 1440)
        self.assertEqual(viewport.offset_x, 0)
        self.assertEqual(viewport.offset_y, 560)
        self.assertAlmostEqual(viewport.scale, 3.6)

    def test_resolve_aspect_viewport_can_keep_legacy_full_frame_fill_mode(self):
        viewport = cr_66views.resolve_aspect_viewport(
            source_width=1352,
            source_height=1014,
            target_width=1440,
            target_height=2560,
            aspect_fit="fill",
        )

        self.assertEqual(viewport.render_width, 1440)
        self.assertEqual(viewport.render_height, 2560)
        self.assertEqual(viewport.offset_x, 0)
        self.assertEqual(viewport.offset_y, 0)
        self.assertAlmostEqual(viewport.scale, 2560.0 / 1014.0)

    def test_resolve_lkg_camera_viewport_expand_uses_full_9_16_panel_camera(self):
        viewport = cr_66views.resolve_lkg_camera_viewport(
            source_width=1352,
            source_height=1014,
            target_width=1440,
            target_height=2560,
            aspect_fit="contain",
            camera_aspect_mode="expand",
        )

        self.assertEqual(viewport.panel_width, 1440)
        self.assertEqual(viewport.panel_height, 2560)
        self.assertEqual(viewport.render_width, 1440)
        self.assertEqual(viewport.render_height, 2560)
        self.assertEqual(viewport.offset_x, 0)
        self.assertEqual(viewport.offset_y, 0)
        self.assertEqual(viewport.aspect_fit, "fit")
        self.assertAlmostEqual(viewport.scale, 1440.0 / 1352.0)
        self.assertFalse(cr_66views.camera_crop_to_fill_for_viewport(viewport, no_crop_to_fill=False))

    def test_resolve_lkg_camera_viewport_preserve_keeps_contain_letterbox(self):
        viewport = cr_66views.resolve_lkg_camera_viewport(
            source_width=1352,
            source_height=1014,
            target_width=1440,
            target_height=2560,
            aspect_fit="contain",
            camera_aspect_mode="preserve",
        )

        self.assertEqual(viewport.render_width, 1440)
        self.assertEqual(viewport.render_height, 1080)
        self.assertEqual(viewport.offset_y, 740)
        self.assertEqual(viewport.aspect_fit, "contain")

    def test_accumulate_interlaced_view_respects_content_viewport(self):
        interlaced = torch.zeros((3, 4, 5), dtype=torch.float32)
        image = torch.ones((3, 2, 3), dtype=torch.float32)
        viewpoint_index = torch.zeros((4, 5, 3), dtype=torch.long)
        viewpoint_index[1:3, 1:4, :] = 7
        viewport = cr_66views.AspectViewport(
            source_width=6,
            source_height=4,
            panel_width=5,
            panel_height=4,
            render_width=3,
            render_height=2,
            offset_x=1,
            offset_y=1,
            scale=0.5,
            aspect_fit="contain",
        )

        cr_66views.accumulate_interlaced_view(
            interlaced,
            image,
            viewpoint_index,
            view_id=7,
            viewport=viewport,
        )

        self.assertTrue(torch.allclose(interlaced[:, 1:3, 1:4], torch.ones((3, 2, 3))))
        self.assertTrue(torch.allclose(interlaced[:, :1, :], torch.zeros((3, 1, 5))))
        self.assertTrue(torch.allclose(interlaced[:, 3:, :], torch.zeros((3, 1, 5))))

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

    def test_format_optional_metric_handles_missing_sampled_comparisons(self):
        summary = cr_66views.summarize_sampled_metrics({})

        self.assertIsNone(summary["psnr_mean"])
        self.assertEqual(cr_66views.format_optional_metric(summary["psnr_mean"]), "n/a")

    def test_apply_orbit_state_to_c2w_changes_pose_without_moving_center_when_default(self):
        c2w = torch.eye(4)
        orbit_center = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32)
        orbit = SimpleNamespace(yaw_deg=0.0, pitch_deg=0.0, pan_x=0.0, pan_y=0.0, distance_scale=1.0)

        updated_c2w, updated_center = cr_66views.apply_orbit_state_to_c2w(c2w, orbit_center, orbit)

        self.assertTrue(torch.allclose(updated_c2w, c2w))
        self.assertTrue(torch.allclose(updated_center, orbit_center))

    def test_render_interlaced_frame_uses_one_shot_renderer_and_orbit_context(self):
        viewpoint_index = torch.zeros((2, 2, 3), dtype=torch.long)
        viewpoint_index[:, :, 1] = 1
        context = cr_66views.RtgsCr66RenderContext(
            args=SimpleNamespace(
                device="cpu",
                tile_size=16,
                near_plane=0.01,
                far_plane=100.0,
                camera_model="pinhole",
                rtgs_compat_projection=True,
                cluster_size=2,
                cr_remapping="on",
            ),
            runtime=SimpleNamespace(official=SimpleNamespace(gaussians=object()), background=None),
            cr_anchor_camera_cuda=SimpleNamespace(camera_center=torch.zeros(3)),
            fov_normalization={"applied": False},
            timestamp=0.0,
            panel_width=2,
            panel_height=2,
            render_width=2,
            render_height=2,
            viewport=cr_66views.AspectViewport(
                source_width=2,
                source_height=2,
                panel_width=2,
                panel_height=2,
                render_width=2,
                render_height=2,
                offset_x=0,
                offset_y=0,
                scale=1.0,
                aspect_fit="contain",
            ),
            crop_to_fill=False,
            source_view_count=2,
            viewpoint_index=np.zeros((2, 2, 3), dtype=np.int32),
            viewpoint_index_t=viewpoint_index,
            view_index_stats={},
            geometry=SimpleNamespace(mask=None),
            anchor_c2w=torch.eye(4),
            orbit_center=torch.tensor([0.0, 0.0, 1.0]),
            view_degree=10.0,
            orbit_direction=-1,
        )
        one_shot_image = torch.full((3, 2, 2), 0.5, dtype=torch.float32)
        one_shot_timing = SimpleNamespace(frame_ms_with_lkg_interlace=12.5)
        orbit = SimpleNamespace(yaw_deg=10.0, pitch_deg=0.0, pan_x=0.0, pan_y=0.0, distance_scale=1.0)

        def fake_one_shot(received_context, *, variant):
            self.assertEqual(variant.name, "interactive_one_shot")
            self.assertEqual(variant.cluster_size, 2)
            self.assertTrue(variant.use_remapping)
            self.assertTrue(variant.reuse_enabled)
            self.assertFalse(torch.allclose(received_context.anchor_c2w, context.anchor_c2w))
            return one_shot_image, one_shot_timing

        with mock.patch(
            "lkg_experiment.rtgs_coherent.cr_one_shot.render_rtgs_cr_one_shot_interlaced_once",
            side_effect=fake_one_shot,
        ) as render_one_shot:
            image, render_ms = cr_66views.render_rtgs_cr_66_interlaced_frame(context, orbit_state=orbit)

        render_one_shot.assert_called_once()
        self.assertGreaterEqual(render_ms, 0.0)
        self.assertAlmostEqual(render_ms, 12.5)
        self.assertTrue(torch.allclose(image, one_shot_image))


if __name__ == "__main__":
    unittest.main()
