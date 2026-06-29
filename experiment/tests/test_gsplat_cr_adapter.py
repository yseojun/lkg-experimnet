from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


class GsplatCrAdapterTest(unittest.TestCase):
    def test_rasterization_cr_forwards_rtgs_projection_adapter(self):
        gsplat_root = Path(__file__).resolve().parents[2] / "gsplat"
        sys.path.insert(0, str(gsplat_root))
        from gsplat import rendering_coherent_raster as rendering_cr

        adapter = torch.tensor([-10.0, -8.0, -0.5, -0.75, 2.0, 1.5], dtype=torch.float32)
        projection_adapters = []
        isect_adapters = []

        def fake_projection(_mask, means, _covars, _quats, _scales, viewmats, _Ks, _width, _height, **kwargs):
            projection_adapters.append(kwargs.get("rtgs_projection_adapter"))
            camera_count = int(viewmats.shape[0])
            gaussian_count = int(means.shape[0])
            return (
                torch.ones((camera_count, gaussian_count, 2), dtype=torch.int32),
                torch.zeros((camera_count, gaussian_count, 2), dtype=torch.float32),
                torch.ones((camera_count, gaussian_count), dtype=torch.float32),
                torch.ones((camera_count, gaussian_count, 3), dtype=torch.float32),
                None,
            )

        def fake_isect_tiles(_means, means2d, _radii, _depths, *_args, **kwargs):
            isect_adapters.append(kwargs.get("rtgs_projection_adapter"))
            gaussian_count = int(means2d.shape[1])
            return (
                torch.zeros((1, gaussian_count), dtype=torch.int32),
                torch.zeros((0,), dtype=torch.int64),
                torch.zeros((0,), dtype=torch.int32),
                torch.zeros((gaussian_count, 1, 1, 2), dtype=torch.float32),
            )

        with mock.patch.object(rendering_cr, "fully_fused_projection_CR", side_effect=fake_projection), mock.patch.object(
            rendering_cr, "isect_tiles_CR", side_effect=fake_isect_tiles
        ), mock.patch.object(
            rendering_cr, "isect_offset_encode_CR", return_value=torch.zeros((1, 1, 1), dtype=torch.int32)
        ), mock.patch.object(
            rendering_cr, "rasterize_to_pixels_CR", return_value=torch.zeros((1, 1, 3, 2, 2), dtype=torch.float32)
        ):
            rendering_cr.rasterization_CR(
                means=torch.zeros((2, 3), dtype=torch.float32),
                quats=None,
                scales=None,
                opacities=torch.ones((2,), dtype=torch.float32),
                colors=torch.ones((1, 2, 3), dtype=torch.float32),
                adjacent_viewmats=torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4),
                Ks=torch.eye(3, dtype=torch.float32).reshape(1, 3, 3),
                view_idx_matrix=torch.zeros((1, 1, 3, 2, 2), dtype=torch.uint32),
                subpixel_coord_matrix=torch.zeros((1, 1, 3, 2, 2, 3), dtype=torch.uint32),
                width=2,
                height=2,
                tile_size=2,
                covars=torch.zeros((2, 6), dtype=torch.float32),
                rtgs_projection_adapter=adapter,
            )

        self.assertEqual(len(projection_adapters), 3)
        self.assertTrue(all(torch.equal(current, adapter) for current in projection_adapters))
        self.assertEqual(len(isect_adapters), 1)
        self.assertTrue(torch.equal(isect_adapters[0], adapter))


if __name__ == "__main__":
    unittest.main()
