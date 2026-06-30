import ctypes
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from lkg_experiment.build_lut_npz import (
    DEFAULT_BRIDGE_SDK_ROOT,
    balanced_view_index_from_float,
    build_calculated_lkg_viewpoint_index,
    build_parser,
    default_output_path,
    fill_invalid_view_float,
    index_method_requires_opengl,
    install_bridge_sdk_root,
    quilt_settings_for_display,
    read_lkg_display_calibration,
    save_viewpoint_index_npz,
)
from lkg_experiment.coherent_raster import LKGViewMappingCalibration


class BuildLutNpzTest(unittest.TestCase):
    def test_balanced_view_index_rank_maps_all_views(self):
        view_float = np.asarray(
            [
                [[0.0, 0.5, 1.0], [1.5, 2.0, 2.5]],
                [[3.0, 3.5, 4.0], [4.5, 5.0, 5.5]],
            ],
            dtype=np.float32,
        )

        viewpoint_index = balanced_view_index_from_float(view_float, view_count=3)

        self.assertEqual(viewpoint_index.dtype, np.int32)
        self.assertEqual(viewpoint_index.shape, view_float.shape)
        np.testing.assert_array_equal(
            np.bincount(viewpoint_index.reshape(-1), minlength=3),
            np.asarray([4, 4, 4]),
        )
        self.assertEqual(int(viewpoint_index.min()), 0)
        self.assertEqual(int(viewpoint_index.max()), 2)

    def test_fill_invalid_view_float_interpolates_missing_samples(self):
        view_float = np.asarray([[[0.0], [np.nan], [np.nan], [np.nan], [4.0]]], dtype=np.float32)
        valid = np.isfinite(view_float)

        filled = fill_invalid_view_float(view_float, valid)

        np.testing.assert_allclose(filled[:, :, 0], np.asarray([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
        self.assertFalse(np.isnan(filled).any())

    def test_save_viewpoint_index_npz_matches_experiment_loader_metadata(self):
        viewpoint_index = np.arange(2 * 3 * 3, dtype=np.int32).reshape(2, 3, 3) % 6

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lut.npz"
            save_viewpoint_index_npz(
                path,
                viewpoint_index,
                view_count=6,
                quilt_cols=3,
                quilt_rows=2,
                layout="gl-top",
                index_method="balanced-ramp",
                extra_metadata={
                    "normalized_valid_count": np.asarray(18, dtype=np.int64),
                    "balanced_rank_remap": np.asarray(True),
                },
            )

            with np.load(path) as loaded:
                self.assertIn("viewpoint_index", loaded)
                self.assertIn("view_count", loaded)
                self.assertIn("quilt_cols", loaded)
                self.assertIn("quilt_rows", loaded)
                self.assertIn("layout", loaded)
                self.assertIn("width", loaded)
                self.assertIn("height", loaded)
                self.assertIn("index_method", loaded)
                np.testing.assert_array_equal(loaded["viewpoint_index"], viewpoint_index)
                self.assertEqual(int(loaded["view_count"]), 6)
                self.assertEqual(int(loaded["quilt_cols"]), 3)
                self.assertEqual(int(loaded["quilt_rows"]), 2)
                self.assertEqual(str(loaded["layout"]), "gl-top")
                self.assertEqual(int(loaded["width"]), 3)
                self.assertEqual(int(loaded["height"]), 2)
                self.assertEqual(str(loaded["index_method"]), "balanced-ramp")
                self.assertEqual(int(loaded["normalized_valid_count"]), 18)
                self.assertTrue(bool(loaded["balanced_rank_remap"]))

    def test_parser_defaults_to_lkg_calibration_and_still_accepts_balanced_ramp(self):
        args = build_parser().parse_args(["--output", "/tmp/lut.npz", "--layout", "gl-top"])

        self.assertEqual(args.output, "/tmp/lut.npz")
        self.assertEqual(args.index_method, "lkg-calibration")
        self.assertEqual(args.layout, "gl-top")

        balanced_args = build_parser().parse_args(["--output", "/tmp/lut.npz", "--index-method", "balanced-ramp"])
        self.assertEqual(balanced_args.index_method, "balanced-ramp")

    def test_default_output_path_uses_method_suffix(self):
        self.assertEqual(
            default_output_path(1440, 2560, 66, "lkg-calibration").name,
            "lkg_go_1440x2560_66_views_lkg_calibration.npz",
        )
        self.assertEqual(
            default_output_path(1440, 2560, 66, "balanced-ramp").name,
            "lkg_go_1440x2560_66_views_balanced.npz",
        )

    def test_only_balanced_ramp_requires_opengl(self):
        self.assertFalse(index_method_requires_opengl("lkg-calibration"))
        self.assertTrue(index_method_requires_opengl("balanced-ramp"))

    def test_build_calculated_lkg_viewpoint_index_uses_calibration_formula(self):
        calibration = LKGViewMappingCalibration(
            pitch=1.0,
            slope=0.0,
            center=0.0,
            subp=1.0 / 3.0,
            inv_view=False,
            ri=0,
            bi=2,
        )

        viewpoint_index, metadata = build_calculated_lkg_viewpoint_index(
            width=6,
            height=2,
            view_count=6,
            calibration=calibration,
            quantize="floor",
        )

        self.assertEqual(viewpoint_index.dtype, np.int32)
        self.assertEqual(viewpoint_index.shape, (2, 6, 3))
        np.testing.assert_array_equal(viewpoint_index[0, :, 0], np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int32))
        np.testing.assert_array_equal(viewpoint_index[0, :, 1], np.asarray([2, 3, 4, 5, 0, 1], dtype=np.int32))
        np.testing.assert_array_equal(viewpoint_index[0, :, 2], np.asarray([4, 5, 0, 1, 2, 3], dtype=np.int32))
        self.assertEqual(str(metadata["lkg_quantize"]), "floor")
        self.assertEqual(float(metadata["lkg_pitch"]), 1.0)
        self.assertEqual(float(metadata["lkg_slope"]), 0.0)

    def test_build_calculated_lkg_viewpoint_index_uses_bottom_origin_y(self):
        calibration = LKGViewMappingCalibration(
            pitch=1.0,
            slope=0.5,
            center=0.0,
            subp=0.0,
            inv_view=False,
            ri=0,
            bi=2,
            y_origin="bottom",
        )

        viewpoint_index, metadata = build_calculated_lkg_viewpoint_index(
            width=1,
            height=2,
            view_count=8,
            calibration=calibration,
            quantize="floor",
        )

        np.testing.assert_array_equal(viewpoint_index[:, 0, 0], np.asarray([7, 5], dtype=np.int32))
        np.testing.assert_array_equal(viewpoint_index[:, 0, 1], np.asarray([7, 5], dtype=np.int32))
        np.testing.assert_array_equal(viewpoint_index[:, 0, 2], np.asarray([7, 5], dtype=np.int32))
        self.assertEqual(str(metadata["lkg_y_origin"]), "bottom")

    def test_read_lkg_display_calibration_falls_back_to_raw_display_scalars(self):
        class RawOnlyBridge:
            def get_calibration_for_display(self, _display_handle):
                class RawCalibration:
                    Slope = 2.0

                return RawCalibration()

            def get_display_aspect_for_display(self, _display_handle):
                return 1.25

            def get_pitch_for_display(self, _display_handle):
                raise TypeError("simulated BridgeApi scalar wrapper mismatch")

            def get_center_for_display(self, _display_handle):
                raise TypeError("simulated BridgeApi scalar wrapper mismatch")

            def get_subp_for_display(self, _display_handle):
                raise TypeError("simulated BridgeApi scalar wrapper mismatch")

            def get_invview_for_display(self, _display_handle):
                raise TypeError("simulated BridgeApi scalar wrapper mismatch")

            def get_ri_for_display(self, _display_handle):
                raise TypeError("simulated BridgeApi scalar wrapper mismatch")

            def get_bi_for_display(self, _display_handle):
                raise TypeError("simulated BridgeApi scalar wrapper mismatch")

            def _write_scalar(self, value):
                def write(_display_handle, pointer):
                    pointer._obj.value = value
                    return True

                return write

            _get_pitch_for_display = staticmethod(_write_scalar(None, 7.0))
            _get_center_for_display = staticmethod(_write_scalar(None, 0.25))
            _get_subp_for_display = staticmethod(_write_scalar(None, 1.0 / 3.0))
            _get_invview_for_display = staticmethod(_write_scalar(None, 1))
            _get_ri_for_display = staticmethod(_write_scalar(None, 0))
            _get_bi_for_display = staticmethod(_write_scalar(None, 2))

        calibration = read_lkg_display_calibration(RawOnlyBridge(), display_handle=123)

        self.assertAlmostEqual(calibration.pitch, 7.0)
        self.assertAlmostEqual(calibration.slope, 1.0 / (2.0 * 1.25))
        self.assertAlmostEqual(calibration.center, 0.25)
        self.assertAlmostEqual(calibration.subp, 1.0 / 3.0)
        self.assertTrue(calibration.inv_view)
        self.assertEqual(calibration.ri, 0)
        self.assertEqual(calibration.bi, 2)

    def test_read_lkg_display_calibration_falls_back_to_private_raw_calibration(self):
        class PrivateRawCalibrationBridge:
            def get_calibration_for_display(self, _display_handle):
                raise TypeError("simulated BridgeApi calibration wrapper mismatch")

            def get_tilt_for_display(self, _display_handle):
                return -99.0

            def get_display_aspect_for_display(self, _display_handle):
                raise TypeError("simulated BridgeApi scalar wrapper mismatch")

            def _get_calibration_for_display(
                self,
                _display_handle,
                center,
                pitch,
                slope,
                width,
                height,
                dpi,
                flip_x,
                inv_view,
                viewcone,
                fringe,
                cell_pattern_mode,
                cell_count,
                cells,
            ):
                center._obj.value = 0.5
                pitch._obj.value = 80.0
                slope._obj.value = -8.0
                width._obj.value = 1440
                height._obj.value = 2560
                dpi._obj.value = 338.0
                flip_x._obj.value = 0.0
                inv_view._obj.value = 1
                viewcone._obj.value = 53.0
                fringe._obj.value = 0.0
                cell_pattern_mode._obj.value = 0
                cell_count._obj.value = 0
                self.cells_argument = cells
                return True

            def _write_scalar(self, value):
                def write(_display_handle, pointer):
                    pointer._obj.value = value
                    return True

                return write

            _get_pitch_for_display = staticmethod(_write_scalar(None, 234.0))
            _get_displayaspect_for_display = staticmethod(_write_scalar(None, 0.5))
            _get_center_for_display = staticmethod(_write_scalar(None, 0.5))
            _get_subp_for_display = staticmethod(_write_scalar(None, 1.0 / (3.0 * 1440.0)))
            _get_invview_for_display = staticmethod(_write_scalar(None, 1))
            _get_ri_for_display = staticmethod(_write_scalar(None, 0))
            _get_bi_for_display = staticmethod(_write_scalar(None, 2))

        calibration = read_lkg_display_calibration(PrivateRawCalibrationBridge(), display_handle=123)

        self.assertAlmostEqual(calibration.pitch, 234.0)
        self.assertAlmostEqual(calibration.slope, -0.25)
        self.assertAlmostEqual(calibration.center, 0.5)
        self.assertAlmostEqual(calibration.subp, 1.0 / (3.0 * 1440.0))
        self.assertTrue(calibration.inv_view)
        self.assertEqual(calibration.ri, 0)
        self.assertEqual(calibration.bi, 2)
        self.assertEqual(calibration.y_origin, "bottom")

    def test_read_lkg_display_calibration_converts_scalar_tilt_fallback_to_raw_slope(self):
        class TiltOnlyBridge:
            def get_calibration_for_display(self, _display_handle):
                raise RuntimeError("simulated calibration unavailable")

            def get_display_aspect_for_display(self, _display_handle):
                return 0.5

            def get_tilt_for_display(self, _display_handle):
                return -2.0

            def get_pitch_for_display(self, _display_handle):
                return 234.0

            def get_center_for_display(self, _display_handle):
                return 0.5

            def get_subp_for_display(self, _display_handle):
                return 1.0 / (3.0 * 1440.0)

            def get_invview_for_display(self, _display_handle):
                return 1

            def get_ri_for_display(self, _display_handle):
                return 0

            def get_bi_for_display(self, _display_handle):
                return 2

        calibration = read_lkg_display_calibration(TiltOnlyBridge(), display_handle=123)

        self.assertAlmostEqual(calibration.slope, -0.25)
        self.assertEqual(calibration.y_origin, "bottom")

    def test_quilt_settings_for_display_falls_back_to_raw_bridge_function(self):
        class RawQuiltBridge:
            def get_default_quilt_settings_for_display(self, _display_handle):
                raise TypeError("simulated BridgeApi quilt wrapper mismatch")

            def _get_default_quilt_settings_for_display(
                self,
                _display_handle,
                aspect,
                quilt_width,
                quilt_height,
                quilt_cols,
                quilt_rows,
            ):
                aspect._obj.value = 0.5625
                quilt_width._obj.value = 4092
                quilt_height._obj.value = 4092
                quilt_cols._obj.value = 11
                quilt_rows._obj.value = 6
                return True

        self.assertEqual(
            quilt_settings_for_display(RawQuiltBridge(), display_handle=123),
            (0.5625, 4092, 4092, 11, 6),
        )

    def test_install_bridge_sdk_root_supports_sdk_absolute_imports(self):
        root = install_bridge_sdk_root(DEFAULT_BRIDGE_SDK_ROOT)

        self.assertIn(str(root / "src"), sys.path)
        self.assertIn(str(root / "src/bridge_python_sdk"), sys.path)

    def test_install_bridge_sdk_root_preloads_configured_system_libcurl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Bridge-Python-SDK-Lab"
            (root / "src/bridge_python_sdk").mkdir(parents=True)
            libcurl = Path(tmp) / "libcurl.so.4"
            libcurl.write_bytes(b"")

            with mock.patch.dict("os.environ", {"LKG_BRIDGE_SYSTEM_LIBCURL": str(libcurl)}):
                with mock.patch.object(ctypes, "CDLL") as cdll:
                    install_bridge_sdk_root(root)

        cdll.assert_called_once_with(str(libcurl), mode=ctypes.RTLD_GLOBAL)


if __name__ == "__main__":
    unittest.main()
