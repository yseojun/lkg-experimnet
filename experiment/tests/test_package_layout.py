import unittest


class PackageLayoutTest(unittest.TestCase):
    def test_canonical_packages_are_importable(self):
        from lkg_experiment.coherent_default.coherent_raster_experiment import tensor_to_hwc_uint8
        from lkg_experiment.coherent_default.run_coherent_raster_experiment import build_parser
        from lkg_experiment.experiment.batch_experiments import SUMMARY_COLUMNS
        from lkg_experiment.fourdgs.fourdgs_bridge import parse_4dgs_cfg_args
        from lkg_experiment.rtgs_coherent.cli import default_output_path

        self.assertTrue(callable(tensor_to_hwc_uint8))
        self.assertTrue(callable(build_parser))
        self.assertIn("scene", SUMMARY_COLUMNS)
        self.assertTrue(callable(parse_4dgs_cfg_args))
        self.assertTrue(callable(default_output_path))

    def test_backward_compatible_imports_still_resolve(self):
        from lkg_experiment.coherent_raster_experiment import tensor_to_hwc_uint8
        from lkg_experiment.fourdgs_bridge import parse_4dgs_cfg_args
        from lkg_experiment.rtgs_coherent import build_parser

        self.assertTrue(callable(tensor_to_hwc_uint8))
        self.assertTrue(callable(parse_4dgs_cfg_args))
        self.assertEqual(build_parser().prog, "python -m unittest")


if __name__ == "__main__":
    unittest.main()
