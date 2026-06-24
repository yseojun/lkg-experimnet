import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from lkg_experiment.fourdgs_bridge import (
    FourDGSCheckpoint,
    load_4dgs_deform_network_class,
    load_4dgs_static_splats,
    parse_4dgs_cfg_args,
    resolve_4dgs_iteration_dir,
)


def write_ascii_ply(path: Path, rows: list[list[float]], sh_degree: int = 1) -> None:
    f_rest_count = 3 * (sh_degree + 1) ** 2 - 3
    properties = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    properties.extend(f"f_rest_{idx}" for idx in range(f_rest_count))
    properties.extend(["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    header = ["ply", "format ascii 1.0", f"element vertex {len(rows)}"]
    header.extend(f"property float {name}" for name in properties)
    header.append("end_header")
    body = [" ".join(str(value) for value in row) for row in rows]
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")


class FourDGSBridgeTest(unittest.TestCase):
    def test_resolve_iteration_dir_picks_latest_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "point_cloud" / "iteration_10").mkdir(parents=True)
            (root / "point_cloud" / "iteration_14000").mkdir(parents=True)

            self.assertEqual(resolve_4dgs_iteration_dir(root), root / "point_cloud" / "iteration_14000")
            self.assertEqual(resolve_4dgs_iteration_dir(root, iteration=10), root / "point_cloud" / "iteration_10")

    def test_parse_cfg_args_reads_namespace_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg_args"
            path.write_text("Namespace(sh_degree=3, iterations=14000, model_path='/tmp/model')", encoding="utf-8")

            args = parse_4dgs_cfg_args(path)

        self.assertEqual(args.sh_degree, 3)
        self.assertEqual(args.iterations, 14000)
        self.assertEqual(args.model_path, "/tmp/model")

    def test_load_static_splats_converts_4dgs_ply_to_gsplat_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            ply_path = Path(tmp) / "point_cloud.ply"
            row0 = [1, 2, 3, 0, 0, 0, 0.1, 0.2, 0.3] + [0.01 * i for i in range(9)] + [0.4, -1, -2, -3, 1, 0, 0, 0]
            row1 = [4, 5, 6, 0, 0, 0, 0.5, 0.6, 0.7] + [0.02 * i for i in range(9)] + [0.8, -4, -5, -6, 0, 1, 0, 0]
            write_ascii_ply(ply_path, [row0, row1], sh_degree=1)

            splats = load_4dgs_static_splats(ply_path, sh_degree=1, device="cpu")

        self.assertEqual(set(splats), {"means", "opacities", "quats", "scales", "sh0", "shN"})
        self.assertEqual(tuple(splats["means"].shape), (2, 3))
        self.assertEqual(tuple(splats["opacities"].shape), (2,))
        self.assertEqual(tuple(splats["quats"].shape), (2, 4))
        self.assertEqual(tuple(splats["scales"].shape), (2, 3))
        self.assertEqual(tuple(splats["sh0"].shape), (2, 1, 3))
        self.assertEqual(tuple(splats["shN"].shape), (2, 3, 3))
        torch.testing.assert_close(splats["means"][0], torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(splats["sh0"][1, 0], torch.tensor([0.5, 0.6, 0.7]))

    def test_checkpoint_splats_at_uses_deformation_to_generate_3dgs_tensors(self):
        class FakeDeformation(torch.nn.Module):
            def forward(self, means, scales, quats, opacities, shs, times):
                return means + times, scales + 1.0, quats + 2.0, opacities + 3.0, shs + 4.0

        static_splats = {
            "means": torch.zeros((2, 3)),
            "scales": torch.ones((2, 3)),
            "quats": torch.ones((2, 4)),
            "opacities": torch.ones((2,)),
            "sh0": torch.zeros((2, 1, 3)),
            "shN": torch.zeros((2, 3, 3)),
        }
        checkpoint = FourDGSCheckpoint(
            static_splats=static_splats,
            sh_degree=1,
            iteration=14000,
            iteration_dir=Path("/tmp/iteration_14000"),
            cfg_args=Namespace(),
            deformation=FakeDeformation(),
            device="cpu",
        )

        splats = checkpoint.splats_at(0.25)

        torch.testing.assert_close(splats["means"], torch.full((2, 3), 0.25))
        torch.testing.assert_close(splats["scales"], torch.full((2, 3), 2.0))
        torch.testing.assert_close(splats["quats"], torch.full((2, 4), 3.0))
        torch.testing.assert_close(splats["opacities"], torch.full((2,), 4.0))
        torch.testing.assert_close(splats["sh0"], torch.full((2, 1, 3), 4.0))
        torch.testing.assert_close(splats["shN"], torch.full((2, 3, 3), 4.0))

    def test_deform_network_loader_does_not_import_scene_package_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_dir = root / "scene"
            scene_dir.mkdir()
            (scene_dir / "__init__.py").write_text("raise ModuleNotFoundError('plyfile')\n", encoding="utf-8")
            (scene_dir / "deformation.py").write_text(
                "class deform_network:\n"
                "    pass\n",
                encoding="utf-8",
            )

            deform_network = load_4dgs_deform_network_class(root)

        self.assertEqual(deform_network.__name__, "deform_network")


if __name__ == "__main__":
    unittest.main()
