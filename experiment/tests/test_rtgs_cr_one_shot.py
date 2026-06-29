from __future__ import annotations

import unittest
import sys
import types
import math
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from lkg_experiment.coherent_default.coherent_gsplat_bridge import build_cr_lookup_arrays
from lkg_experiment.rtgs_coherent import cr_one_shot


class RtgsCrOneShotTest(unittest.TestCase):
    def test_build_rtgs_projection_adapter_for_sentinel_camera(self):
        camera = SimpleNamespace(
            FoVx=-1.0,
            FoVy=-1.0,
            fl_x=777.916513,
            fl_y=777.916513,
            cx=720.0,
            cy=540.0,
            image_width=1440,
            image_height=1080,
        )

        adapter, info = cr_one_shot.build_rtgs_projection_adapter(camera, device="cpu", dtype=torch.float32, enabled=True)

        fov_fx = 1440.0 / (2.0 * math.tan(-0.5))
        fov_fy = 1080.0 / (2.0 * math.tan(-0.5))
        self.assertTrue(info["applied"])
        self.assertEqual(info["reason"], "explicit_intrinsics_with_nonpositive_fov_sentinel")
        self.assertEqual(tuple(adapter.shape), (6,))
        self.assertTrue(
            torch.allclose(
                adapter,
                torch.tensor(
                    [
                        fov_fx,
                        fov_fy,
                        777.916513 / fov_fx,
                        777.916513 / fov_fy,
                        720.0,
                        540.0,
                    ],
                    dtype=torch.float32,
                ),
            )
        )

    def test_build_rtgs_projection_adapter_returns_none_for_positive_fov(self):
        camera = SimpleNamespace(
            FoVx=0.7,
            FoVy=0.7,
            fl_x=-1.0,
            fl_y=-1.0,
            cx=-1.0,
            cy=-1.0,
            image_width=1440,
            image_height=1440,
        )

        adapter, info = cr_one_shot.build_rtgs_projection_adapter(camera, device="cpu", dtype=torch.float32, enabled=True)

        self.assertIsNone(adapter)
        self.assertFalse(info["applied"])
        self.assertEqual(info["reason"], "missing_positive_explicit_intrinsics")

    def test_crop_viewpoint_index_to_viewport_uses_content_bounds(self):
        viewpoint_index = np.arange(5 * 7 * 3, dtype=np.int32).reshape(5, 7, 3)
        viewport = SimpleNamespace(
            offset_x=2,
            offset_y=1,
            render_width=4,
            render_height=3,
            panel_width=7,
            panel_height=5,
            x1=6,
            y1=4,
        )

        cropped = cr_one_shot.crop_viewpoint_index_to_viewport(viewpoint_index, viewport)

        self.assertEqual(cropped.shape, (3, 4, 3))
        np.testing.assert_array_equal(cropped, viewpoint_index[1:4, 2:6, :])

    def test_build_viewport_cr_lookup_uses_render_dimensions_not_panel_dimensions(self):
        viewpoint_index = np.arange(5 * 7 * 3, dtype=np.int32).reshape(5, 7, 3) % 66
        viewport = SimpleNamespace(
            offset_x=2,
            offset_y=1,
            render_width=4,
            render_height=3,
            panel_width=7,
            panel_height=5,
            x1=6,
            y1=4,
        )

        lookup = cr_one_shot.build_viewport_cr_lookup(
            viewpoint_index,
            viewport,
            tile_size=2,
            use_remapping=False,
        )
        expected = build_cr_lookup_arrays(viewpoint_index[1:4, 2:6, :], tile_size=2, use_remapping=False)

        self.assertEqual(lookup.original_height, 3)
        self.assertEqual(lookup.original_width, 4)
        self.assertEqual(lookup.padded_height, 4)
        self.assertEqual(lookup.padded_width, 4)
        np.testing.assert_array_equal(lookup.view_index, expected.view_index)
        np.testing.assert_array_equal(lookup.subpixel_coords, expected.subpixel_coords)

    def test_paste_viewport_image_into_panel_preserves_background_outside_viewport(self):
        viewport = SimpleNamespace(
            offset_x=2,
            offset_y=1,
            render_width=4,
            render_height=3,
            panel_width=7,
            panel_height=5,
            x1=6,
            y1=4,
        )
        viewport_image = torch.zeros((3, 3, 4), dtype=torch.float32)
        viewport_image[0].fill_(0.25)
        viewport_image[1].fill_(0.5)
        viewport_image[2].fill_(0.75)

        panel = cr_one_shot.paste_viewport_image_into_panel(
            viewport_image,
            viewport,
            background=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        )

        self.assertEqual(tuple(panel.shape), (3, 5, 7))
        self.assertTrue(torch.allclose(panel[:, 0, 0], torch.tensor([0.1, 0.2, 0.3])))
        self.assertTrue(torch.allclose(panel[:, 4, 6], torch.tensor([0.1, 0.2, 0.3])))
        self.assertTrue(torch.allclose(panel[:, 1, 2], torch.tensor([0.25, 0.5, 0.75])))
        self.assertTrue(torch.allclose(panel[:, 3, 5], torch.tensor([0.25, 0.5, 0.75])))

    def test_one_shot_renderer_uses_viewport_lookup_and_render_dimensions(self):
        calls = {}
        viewport = SimpleNamespace(
            offset_x=2,
            offset_y=1,
            render_width=4,
            render_height=3,
            panel_width=7,
            panel_height=5,
            x1=6,
            y1=4,
        )
        context = SimpleNamespace(
            args=SimpleNamespace(
                device="cpu",
                tile_size=2,
                near_plane=0.01,
                far_plane=100.0,
                camera_model="pinhole",
                debug_cr=False,
            ),
            viewpoint_index=np.zeros((5, 7, 3), dtype=np.int32),
            viewport=viewport,
            source_view_count=4,
            anchor_c2w=torch.eye(4),
            orbit_center=torch.zeros(3),
            view_degree=53.0,
            orbit_direction=-1,
            runtime=SimpleNamespace(
                official=SimpleNamespace(gaussians=object()),
                background=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            ),
            timestamp=0.0,
            geometry=SimpleNamespace(mask=torch.ones(2, dtype=torch.bool)),
            cr_anchor_camera_cuda=SimpleNamespace(),
            render_width=4,
            render_height=3,
            crop_to_fill=False,
        )
        variant = SimpleNamespace(cluster_size=2, reuse_enabled=True, use_remapping=False)
        fake_render = torch.zeros((3, 3, 4), dtype=torch.float32)
        fake_render[0].fill_(0.25)
        fake_render[1].fill_(0.5)
        fake_render[2].fill_(0.75)

        utils_module = types.ModuleType("coherent_raster.utils.utils_coherent_raster")
        utils_module.unpatchify_image_shape_matrix = lambda image: image
        utils_module.unpad = lambda image, height, width: image[:, :height, :width]
        raster_module = types.ModuleType("gsplat.rendering_coherent_raster")

        def fake_rasterization_cr(**kwargs):
            calls["rasterization"] = kwargs
            return fake_render.clone(), None, None

        raster_module.rasterization_CR = fake_rasterization_cr

        class FakeCamera:
            FoVx = -1.0
            FoVy = -1.0
            fl_x = 2.0
            fl_y = 2.0
            cx = 2.0
            cy = 1.5
            image_width = 4
            image_height = 3
            camera_center = torch.zeros(3)

            def cuda(self):
                return self

        with mock.patch.dict(
            sys.modules,
            {
                "coherent_raster": types.ModuleType("coherent_raster"),
                "coherent_raster.utils": types.ModuleType("coherent_raster.utils"),
                "coherent_raster.utils.utils_coherent_raster": utils_module,
                "gsplat": types.ModuleType("gsplat"),
                "gsplat.rendering_coherent_raster": raster_module,
            },
        ), mock.patch(
            "lkg_experiment.rtgs_coherent.cr_one_shot.build_viewport_cr_lookup",
            wraps=cr_one_shot.build_viewport_cr_lookup,
        ) as build_lookup, mock.patch(
            "lkg_experiment.rtgs_coherent.views66.synthesize_grouped_viewmats",
            return_value=torch.eye(4).reshape(1, 1, 4, 4),
        ) as synthesize, mock.patch(
            "lkg_experiment.rtgs_coherent.cr_66views.evaluate_rtgs_colors",
            return_value=torch.ones((2, 3), dtype=torch.float32),
        ), mock.patch(
            "lkg_experiment.rtgs_coherent.cr_66views.snapshot_from_geometry",
            return_value=SimpleNamespace(
                means=torch.zeros((2, 3), dtype=torch.float32),
                opacities=torch.ones(2, dtype=torch.float32),
                colors=torch.ones((1, 2, 3), dtype=torch.float32),
                covars=torch.zeros((2, 6), dtype=torch.float32),
            ),
        ), mock.patch(
            "lkg_experiment.rtgs_coherent.cr_66views.synthetic_camera_from_viewmat_preserving_rtgs_contract",
            return_value=FakeCamera(),
        ) as camera_factory, mock.patch(
            "lkg_experiment.rtgs_coherent.cr_66views.rtgs_camera_to_gsplat_inputs",
            return_value=(torch.eye(4), torch.eye(3)),
        ):
            panel, render_ms = cr_one_shot.render_rtgs_cr_one_shot_interlaced_once(context, variant=variant)

        self.assertGreaterEqual(render_ms, 0.0)
        build_lookup.assert_called_once_with(context.viewpoint_index, viewport, tile_size=2, use_remapping=False)
        synthesize.assert_called_once()
        self.assertEqual(synthesize.call_args.kwargs["cluster_size"], 2)
        self.assertEqual(camera_factory.call_args.kwargs["width"], 4)
        self.assertEqual(camera_factory.call_args.kwargs["height"], 3)
        self.assertEqual(calls["rasterization"]["width"], 4)
        self.assertEqual(calls["rasterization"]["height"], 3)
        self.assertIn("rtgs_projection_adapter", calls["rasterization"])
        self.assertEqual(tuple(calls["rasterization"]["rtgs_projection_adapter"].shape), (6,))
        self.assertEqual(tuple(panel.shape), (3, 5, 7))
        self.assertTrue(torch.allclose(panel[:, 0, 0], torch.tensor([0.1, 0.2, 0.3])))
        self.assertTrue(torch.allclose(panel[:, 1, 2], torch.tensor([0.25, 0.5, 0.75])))


if __name__ == "__main__":
    unittest.main()
