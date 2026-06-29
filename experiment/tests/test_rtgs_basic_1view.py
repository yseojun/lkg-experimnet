import unittest
import types
from pathlib import Path

import torch

from lkg_experiment.rtgs_coherent import basic_1view


class RtgsBasic1ViewTest(unittest.TestCase):
    def test_parser_defaults_to_current_upstream_rtgs_path(self):
        args = basic_1view.build_parser().parse_args([])

        self.assertEqual(args.model_path, "/data/ysj/result/4dgs/RTGS/jumpingjacks")
        self.assertEqual(args.checkpoint, "checkpoints/chkpnt_best.pth")
        self.assertEqual(args.rtgs_code_root, "/home/ysj/lkg-experiment/4d-gaussian-splatting")
        self.assertEqual(args.dataset_root, "/data/ysj/dataset/dnerf")
        self.assertEqual(args.n3dv_root, "/data/ysj/dataset/N3DV")
        self.assertEqual(args.split, "test")
        self.assertEqual(args.camera_index, 0)
        self.assertEqual(args.n3dv_frame_index, 0)
        self.assertEqual(args.rtgs_rotation_convention, "current")
        self.assertIsNone(args.source_path_override)
        self.assertEqual(args.compute_cov3d_python, "config")
        self.assertEqual(args.convert_shs_python, "config")
        self.assertEqual(args.diagnostic_sample_count, 4096)
        self.assertTrue(args.compare_official)
        self.assertTrue(args.write_diagnostics)

    def test_default_output_path_is_separate_from_coherent_outputs(self):
        path = basic_1view.default_basic_output_path(
            Path("/data/ysj/result/4dgs/RTGS/coffee_martini"),
            split="test",
            camera_index=0,
            timestamp=0.0,
        )

        self.assertIn("rtgs_basic_1view", path.parts)
        self.assertEqual(path.name, "coffee_martini_t0.000000_test_0")

    def test_n3dv_official_test_output_index_matches_rtgs_training_shuffle(self):
        self.assertEqual(basic_1view.n3dv_official_test_output_index(0), 196)
        self.assertEqual(basic_1view.n3dv_official_test_output_index(128), 0)
        self.assertEqual(basic_1view.n3dv_official_test_output_index(150), 262)
        self.assertEqual(basic_1view.n3dv_official_test_output_index(299), 272)

    def test_official_reference_paths_follow_rtgs_output_layout(self):
        paths = basic_1view.official_reference_paths(
            Path("/data/ysj/result/4dgs/RTGS/coffee_martini"),
            output_index=196,
            result_name="ours_best",
        )

        self.assertEqual(paths.render.name, "00196.png")
        self.assertEqual(paths.gt.name, "00196.png")
        self.assertEqual(paths.render.parent.name, "renders")
        self.assertEqual(paths.gt.parent.name, "gt")

    def test_model_diagnostics_summarizes_shapes_and_feature_channels(self):
        model = types.SimpleNamespace(
            gaussian_dim=4,
            rot_4d=True,
            force_sh_3d=False,
            prefilter_var=-1.0,
            time_duration=[0.0, 10.0],
            active_sh_degree=3,
            active_sh_degree_t=2,
            max_sh_degree=3,
            max_sh_degree_t=2,
            env_map=None,
            get_features=torch.zeros((2, 48, 3)),
            get_max_sh_channels=48,
            _xyz=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            _t=torch.tensor([[0.0], [10.0]]),
        )

        diagnostics = basic_1view.model_diagnostics(model, sample_count=2)

        self.assertEqual(diagnostics["gaussian_dim"], 4)
        self.assertTrue(diagnostics["feature_channel_match"])
        self.assertEqual(diagnostics["t"]["shape"], [2, 1])
        self.assertEqual(diagnostics["t"]["min"], 0.0)
        self.assertEqual(diagnostics["t"]["max"], 10.0)

    def test_tensor_summary_samples_with_stride_slice(self):
        summary = basic_1view.tensor_summary(torch.arange(10), sample_count=3)

        self.assertEqual(summary["sampled_numel"], 3)
        self.assertEqual(summary["sample_strategy"], "stride_slice")
        self.assertEqual(summary["sample_step"], 3)
        self.assertEqual(summary["min"], 0.0)
        self.assertEqual(summary["max"], 6.0)
        self.assertEqual(summary["mean"], 3.0)


if __name__ == "__main__":
    unittest.main()
