from __future__ import annotations

import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from lkg_experiment.rtgs_coherent import lkg_web_server


class RtgsLkgWebServerTest(unittest.TestCase):
    def test_scan_rtgs_checkpoints_discovers_dataset_kind_and_relative_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "RTGS"
            rtgs = Path(tmp) / "4d-gaussian-splatting"
            (root / "jumpingjacks" / "checkpoints").mkdir(parents=True)
            (root / "coffee_martini" / "checkpoints").mkdir(parents=True)
            (rtgs / "configs" / "dnerf").mkdir(parents=True)
            (rtgs / "configs" / "dynerf").mkdir(parents=True)
            (root / "jumpingjacks" / "checkpoints" / "chkpnt_best.pth").write_bytes(b"")
            (root / "coffee_martini" / "checkpoints" / "chkpnt_30000.pth").write_bytes(b"")
            (rtgs / "configs" / "dnerf" / "jumpingjacks.yaml").write_text("ModelParams: {}\n", encoding="utf-8")
            (rtgs / "configs" / "dynerf" / "coffee_martini.yaml").write_text("ModelParams: {}\n", encoding="utf-8")

            options = lkg_web_server.scan_rtgs_checkpoints(root, rtgs_code_root=rtgs)

        self.assertEqual([option.scene for option in options], ["coffee_martini", "jumpingjacks"])
        self.assertEqual(options[0].dataset_kind, "n3dv")
        self.assertEqual(options[0].checkpoint, "checkpoints/chkpnt_30000.pth")
        self.assertEqual(options[1].dataset_kind, "dnerf")
        self.assertEqual(options[1].checkpoint, "checkpoints/chkpnt_best.pth")

    def test_apply_orbit_control_updates_and_clamps_state(self):
        state = lkg_web_server.OrbitState()

        state = lkg_web_server.apply_orbit_control(
            state,
            {
                "drag_dx": 20,
                "drag_dy": -2000,
                "wheel_delta": -3,
                "shift": False,
                "key": "KeyW",
            },
        )
        state = lkg_web_server.apply_orbit_control(
            state,
            {
                "drag_dx": 10,
                "drag_dy": 5,
                "shift": True,
                "key": "KeyD",
            },
        )

        self.assertAlmostEqual(state.yaw_deg, 2.0)
        self.assertEqual(state.pitch_deg, -85.0)
        self.assertGreater(state.distance_scale, 1.0)
        self.assertGreater(state.pan_x, 0.0)
        self.assertGreater(state.pan_y, 0.0)

    def test_fps_accumulator_reports_current_and_average(self):
        stats = lkg_web_server.FpsAccumulator()

        stats.add_frame(25.0)
        stats.add_frame(50.0)

        payload = stats.to_json()
        self.assertEqual(payload["frames"], 2)
        self.assertAlmostEqual(payload["current_ms"], 50.0)
        self.assertAlmostEqual(payload["average_ms"], 37.5)
        self.assertAlmostEqual(payload["current_fps"], 20.0)
        self.assertAlmostEqual(payload["average_fps"], 1000.0 / 37.5)

    def test_build_rtgs_cr_args_for_checkpoint_sets_official_runtime_fields(self):
        base = lkg_web_server.build_parser().parse_args(
            [
                "--display-mode",
                "none",
                "--rtgs-code-root",
                "/tmp/rtgs",
                "--gsplat-root",
                "/tmp/gsplat",
                "--dataset-root",
                "/tmp/dnerf",
                "--n3dv-root",
                "/tmp/n3dv",
                "--width",
                "360",
                "--height",
                "640",
                "--views",
                "6",
            ]
        )
        option = lkg_web_server.CheckpointOption(
            id="jumpingjacks:checkpoints/chkpnt_best.pth",
            scene="jumpingjacks",
            dataset_kind="dnerf",
            model_path=Path("/data/RTGS/jumpingjacks"),
            checkpoint_path=Path("/data/RTGS/jumpingjacks/checkpoints/chkpnt_best.pth"),
            checkpoint="checkpoints/chkpnt_best.pth",
        )

        rtgs_args = lkg_web_server.build_rtgs_cr_args_for_checkpoint(base, option)

        self.assertEqual(rtgs_args.dataset_kind, "dnerf")
        self.assertEqual(rtgs_args.model_path, "/data/RTGS/jumpingjacks")
        self.assertEqual(rtgs_args.checkpoint, "checkpoints/chkpnt_best.pth")
        self.assertEqual(rtgs_args.width, 360)
        self.assertEqual(rtgs_args.height, 640)
        self.assertEqual(rtgs_args.views, 6)
        self.assertFalse(rtgs_args.compare_official_sampled)
        self.assertFalse(rtgs_args.write_interlaced)

    def test_shared_state_status_payload_records_loading_error_and_fps(self):
        option = lkg_web_server.CheckpointOption(
            id="coffee_martini:checkpoints/chkpnt_best.pth",
            scene="coffee_martini",
            dataset_kind="n3dv",
            model_path=Path("/data/RTGS/coffee_martini"),
            checkpoint_path=Path("/data/RTGS/coffee_martini/checkpoints/chkpnt_best.pth"),
            checkpoint="checkpoints/chkpnt_best.pth",
        )
        state = lkg_web_server.InteractiveState()
        state.set_active_checkpoint(option)
        state.set_loading(True)
        state.set_error("load failed")
        state.render_fps.add_frame(40.0)
        state.display_fps.add_frame(10.0)

        payload = state.status_payload()

        self.assertTrue(payload["loading_checkpoint"])
        self.assertEqual(payload["last_error"], "load failed")
        self.assertEqual(payload["active_checkpoint"]["scene"], "coffee_martini")
        self.assertAlmostEqual(payload["render_fps"]["current_fps"], 25.0)
        self.assertAlmostEqual(payload["display_fps"]["current_fps"], 100.0)

    def test_web_routes_serve_status_and_checkpoints(self):
        option = lkg_web_server.CheckpointOption(
            id="jumpingjacks:checkpoints/chkpnt_best.pth",
            scene="jumpingjacks",
            dataset_kind="dnerf",
            model_path=Path("/data/RTGS/jumpingjacks"),
            checkpoint_path=Path("/data/RTGS/jumpingjacks/checkpoints/chkpnt_best.pth"),
            checkpoint="checkpoints/chkpnt_best.pth",
        )
        state = lkg_web_server.InteractiveState()
        state.set_active_checkpoint(option)
        catalog = lkg_web_server.CheckpointCatalog([option])
        status_code, status_type, status_body_raw = lkg_web_server.handle_http_get(
            "/api/status",
            state=state,
            catalog=catalog,
        )
        checkpoints_code, checkpoints_type, checkpoints_body_raw = lkg_web_server.handle_http_get(
            "/api/checkpoints",
            state=state,
            catalog=catalog,
        )

        status_body = json.loads(status_body_raw.decode("utf-8"))
        checkpoints_body = json.loads(checkpoints_body_raw.decode("utf-8"))

        self.assertEqual(status_code, 200)
        self.assertEqual(status_type, "application/json")
        self.assertEqual(status_body["active_checkpoint"]["id"], option.id)
        self.assertEqual(checkpoints_code, 200)
        self.assertEqual(checkpoints_type, "application/json")
        self.assertEqual(checkpoints_body["checkpoints"][0]["id"], option.id)

    def test_main_dispatches_to_interactive_server(self):
        with mock.patch.object(lkg_web_server, "run_interactive_server", return_value=0) as run:
            rc = lkg_web_server.main(["--display-mode", "none", "--max-frames", "1"])

        self.assertEqual(rc, 0)
        run.assert_called_once()

    def test_wrapper_script_exists_and_uses_module_main(self):
        script = Path(__file__).resolve().parents[1] / "rtgs_lkg_web_server.py"

        text = script.read_text(encoding="utf-8")

        self.assertIn("lkg_experiment.rtgs_coherent.lkg_web_server", text)
        self.assertIn("main", text)

    def test_pyproject_registers_console_script(self):
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

        self.assertEqual(
            data["project"]["scripts"]["lkg-rtgs-lkg-web-server"],
            "lkg_experiment.rtgs_coherent.lkg_web_server:main",
        )


if __name__ == "__main__":
    unittest.main()
