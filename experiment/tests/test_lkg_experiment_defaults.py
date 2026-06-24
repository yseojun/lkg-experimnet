import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lkg_experiment.build_web_index import build_parser as build_web_index_parser
from lkg_experiment.open_experiment import build_experiment_url, resolve_experiment_target
from lkg_experiment.run_coherent_raster_experiment import build_parser
from lkg_experiment.coherent_raster_experiment import (
    build_experiment_variants,
    cluster_index_from_view_index,
    image_artifact_path,
    parse_cluster_values,
)


class LkgExperimentDefaultsTest(unittest.TestCase):
    def test_parser_defaults_match_clean_server_layout(self):
        experiment_root = Path(__file__).resolve().parents[1]
        workspace_root = experiment_root.parent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "datasets"
            result_root = root / "results"
            generated_root = root / "result" / "generated"
            checkpoint = result_root / "blender_MCMC100000_init50000/drums/ckpts/ckpt_29999_rank0.pt"
            viewpoint_index_file = generated_root / "lkg_go_1440x2560_66_views_lkg_calibration.npz"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"")
            viewpoint_index_file.parent.mkdir(parents=True)
            viewpoint_index_file.write_bytes(b"")

            with patch.dict(
                "os.environ",
                {
                    "DATADIR": str(data_root),
                    "LKG_RESULT_BASE": str(root / "result"),
                    "RESULTDIR": str(result_root),
                },
            ):
                parser = build_parser()
                args = parser.parse_args([])

            checkpoint_path = Path(args.checkpoint_path)
            gsplat_root = Path(args.gsplat_root)
            bridge_sdk_root = Path(args.bridge_sdk_root)
            viewpoint_index = Path(args.viewpoint_index_path)
            artifact_dir = Path(args.artifact_dir)

            self.assertEqual(checkpoint_path, checkpoint)
            self.assertEqual(gsplat_root, workspace_root / "gsplat")
            self.assertTrue((gsplat_root / "gsplat").is_dir(), gsplat_root)
            self.assertEqual(bridge_sdk_root, workspace_root / "Bridge-Python-SDK-Lab")
            self.assertTrue((bridge_sdk_root / "src/bridge_python_sdk").is_dir(), bridge_sdk_root)
            self.assertEqual(viewpoint_index, viewpoint_index_file)
            self.assertTrue(viewpoint_index.is_file(), viewpoint_index)
            self.assertEqual(artifact_dir, generated_root / "coherent_raster_experiments")
            self.assertEqual(args.width, 1440)
            self.assertEqual(args.height, 2560)
            self.assertEqual(args.views, 66)
            self.assertEqual(args.clusters, "2,4,8,16")
            self.assertFalse(args.append_metrics)
            self.assertEqual(args.output_prefix, "")
            self.assertFalse(args.write_mapping_artifacts)
            self.assertFalse(args.write_mapping_previews)

    def test_parser_accepts_explicit_experiment_inputs(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--checkpoint-path",
                "/tmp/ckpt.pt",
                "--bridge-sdk-root",
                "/tmp/bridge",
                "--artifact-dir",
                "/tmp/artifacts",
                "--viewpoint-index-path",
                "/tmp/map.npz",
                "--clusters",
                "2,4",
                "--no-compact-view-index",
                "--append-metrics",
                "--output-prefix",
                "test/17",
                "--write-mapping-artifacts",
                "--write-mapping-previews",
            ]
        )

        self.assertEqual(args.checkpoint_path, "/tmp/ckpt.pt")
        self.assertEqual(args.bridge_sdk_root, "/tmp/bridge")
        self.assertEqual(args.artifact_dir, "/tmp/artifacts")
        self.assertEqual(args.viewpoint_index_path, "/tmp/map.npz")
        self.assertEqual(args.clusters, "2,4")
        self.assertTrue(args.no_compact_view_index)
        self.assertTrue(args.append_metrics)
        self.assertEqual(args.output_prefix, "test/17")
        self.assertTrue(args.write_mapping_artifacts)
        self.assertTrue(args.write_mapping_previews)

    def test_experiment_variant_helpers_are_available(self):
        self.assertEqual(parse_cluster_values("2,4,8,16"), (2, 4, 8, 16))
        view_index = np.array([[[0, 1, 2], [3, 4, 5]]], dtype=np.int32)
        variants = build_experiment_variants((2, 4), ablation_cluster=4)
        self.assertEqual(
            [(variant.name, variant.cluster_size, variant.use_remapping, variant.reuse_enabled) for variant in variants],
            [
                ("cluster_2", 2, True, True),
                ("cluster_4", 4, True, True),
                ("without_remap", 4, False, True),
                ("without_reuse", 1, True, False),
                ("without_reuse_without_remap", 1, False, False),
            ],
        )
        np.testing.assert_array_equal(cluster_index_from_view_index(view_index, cluster_size=4), view_index // 4)

    def test_image_artifact_path_prefixes_camera_outputs(self):
        self.assertEqual(
            image_artifact_path("cluster_8", "looking_glass_tensor.png", output_prefix="test/17"),
            Path("images/cluster_8/test_17_looking_glass_tensor.png"),
        )
        self.assertEqual(
            image_artifact_path("cluster_8", "looking_glass_tensor.png"),
            Path("images/cluster_8/looking_glass_tensor.png"),
        )

    def test_web_index_parser_defaults_to_result_generated_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_base = Path(tmp) / "result"
            with patch.dict("os.environ", {"LKG_RESULT_BASE": str(result_base)}):
                args = build_web_index_parser().parse_args([])

        self.assertEqual(args.experiments_root, str(result_base / "generated" / "coherent_raster_experiments"))
        self.assertFalse(args.no_regenerate_previews)

    def test_open_experiment_resolves_run_and_builds_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "coherent_raster_experiments"
            run = root / "run_a"
            run.mkdir(parents=True)
            (run / "manifest.json").write_text("{}", encoding="utf-8")

            target = resolve_experiment_target(run)
