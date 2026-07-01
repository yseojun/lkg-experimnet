from __future__ import annotations

import json
import tempfile
import threading
import tomllib
import unittest
from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace
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

    def test_publish_render_frame_does_not_require_web_preview_jpeg(self):
        state = lkg_web_server.InteractiveState()
        tensor = object()
        upload_event = object()
        playback = lkg_web_server.PlaybackStatus(
            frame_index=2,
            frame_count=5,
            timestamp=0.25,
            camera_index=7,
            n3dv_frame_index=11,
        )

        state.publish_render_frame(tensor=tensor, render_ms=20.0, playback=playback, upload_event=upload_event)

        seq, latest_tensor = state.latest_tensor_snapshot()
        display_seq, display_tensor, display_event = state.latest_display_tensor_snapshot()
        jpeg_seq, latest_jpeg = state.latest_jpeg_snapshot()
        payload = state.status_payload()
        self.assertEqual(seq, 1)
        self.assertIs(latest_tensor, tensor)
        self.assertEqual(display_seq, 1)
        self.assertIs(display_tensor, tensor)
        self.assertIs(display_event, upload_event)
        self.assertEqual(jpeg_seq, 1)
        self.assertIsNone(latest_jpeg)
        self.assertEqual(payload["playback"]["frame_index"], 2)
        self.assertEqual(payload["playback"]["frame_count"], 5)
        self.assertAlmostEqual(payload["playback"]["timestamp"], 0.25)

    def test_publish_preview_jpeg_keeps_frame_sequence_and_preview_bytes_separate(self):
        state = lkg_web_server.InteractiveState()
        state.publish_render_frame(tensor=object(), render_ms=20.0, playback=None)

        state.publish_preview_jpeg(seq=1, jpeg=b"jpg")

        seq, latest_jpeg = state.latest_jpeg_snapshot()
        payload = state.status_payload()
        self.assertEqual(seq, 1)
        self.assertEqual(latest_jpeg, b"jpg")
        self.assertTrue(payload["preview_available"])

    def test_playback_cursor_loops_over_timeline(self):
        timeline = [
            lkg_web_server.PlaybackFrame(frame_index=0, camera_index=0, n3dv_frame_index=None, timestamp=0.0, camera=object()),
            lkg_web_server.PlaybackFrame(frame_index=1, camera_index=1, n3dv_frame_index=None, timestamp=0.5, camera=object()),
            lkg_web_server.PlaybackFrame(frame_index=2, camera_index=2, n3dv_frame_index=None, timestamp=1.0, camera=object()),
        ]
        cursor = lkg_web_server.PlaybackCursor(timeline)

        frames = [cursor.next_frame() for _ in range(5)]

        self.assertEqual([frame.frame_index for frame in frames], [0, 1, 2, 0, 1])
        self.assertEqual(cursor.current_status().frame_index, 1)
        self.assertEqual(cursor.current_status().frame_count, 3)

    def test_build_dnerf_playback_timeline_keeps_anchor_camera_pose_and_changes_time(self):
        args = SimpleNamespace(
            playback_mode="auto",
            split="test",
            camera_index=2,
            n3dv_frame_index=0,
            device="cpu",
        )
        official = SimpleNamespace(
            scene_paths=SimpleNamespace(dataset_kind="dnerf"),
            gaussians=object(),
            iteration=30000,
            config={},
            cfg_args=SimpleNamespace(),
            checkpoint_path=Path("/tmp/chkpnt.pth"),
            source_path=Path("/tmp/source"),
            model_args=SimpleNamespace(frame_ratio=1),
            time_duration=[0.0, 1.0],
        )
        context = SimpleNamespace(runtime=SimpleNamespace(official=official))
        loaded = []

        def fake_load_playback_camera(*, checkpoint_like, args, camera_index, n3dv_frame_index):
            del checkpoint_like, args, n3dv_frame_index
            loaded.append(int(camera_index))
            return SimpleNamespace(
                pose_id=f"pose_{int(camera_index)}",
                timestamp=float(camera_index) * 0.25,
                uid=int(camera_index),
                image_name=f"cam_{int(camera_index):03d}",
                image=object(),
            )

        with mock.patch.object(lkg_web_server, "_dnerf_playback_camera_count", return_value=4), mock.patch.object(
            lkg_web_server,
            "_load_playback_camera",
            side_effect=fake_load_playback_camera,
        ):
            timeline = lkg_web_server.build_playback_timeline(args, context)

        self.assertEqual(loaded, [2, 0, 1, 2, 3])
        self.assertEqual([frame.frame_index for frame in timeline], [0, 1, 2, 3])
        self.assertEqual([frame.camera_index for frame in timeline], [2, 2, 2, 2])
        self.assertEqual([frame.timestamp for frame in timeline], [0.0, 0.25, 0.5, 0.75])
        self.assertEqual([frame.camera.pose_id for frame in timeline], ["pose_2", "pose_2", "pose_2", "pose_2"])
        self.assertEqual([frame.camera.image_name for frame in timeline], ["time_000", "time_001", "time_002", "time_003"])
        self.assertTrue(all(frame.camera.image is None for frame in timeline))

    def test_should_encode_web_preview_respects_disabled_and_fps_interval(self):
        self.assertFalse(lkg_web_server.should_encode_web_preview(web_preview_fps=0.0, now=10.0, last_preview_at=None))
        self.assertTrue(lkg_web_server.should_encode_web_preview(web_preview_fps=5.0, now=10.0, last_preview_at=None))
        self.assertFalse(lkg_web_server.should_encode_web_preview(web_preview_fps=5.0, now=10.1, last_preview_at=10.0))
        self.assertTrue(lkg_web_server.should_encode_web_preview(web_preview_fps=5.0, now=10.2, last_preview_at=10.0))

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
        self.assertFalse(status_body["preview_available"])
        self.assertEqual(checkpoints_code, 200)
        self.assertEqual(checkpoints_type, "application/json")
        self.assertEqual(checkpoints_body["checkpoints"][0]["id"], option.id)

    def test_index_html_does_not_request_preview_until_status_reports_available(self):
        html = lkg_web_server._index_html()

        self.assertNotIn('src="/frame.jpg"', html)
        self.assertIn("data.preview_available", html)
        self.assertIn("frame.removeAttribute('src')", html)

    def test_main_dispatches_to_interactive_server(self):
        with mock.patch.object(lkg_web_server, "run_interactive_server", return_value=0) as run:
            rc = lkg_web_server.main(["--display-mode", "none", "--max-frames", "1"])

        self.assertEqual(rc, 0)
        run.assert_called_once()

    def test_default_bridge_sdk_root_matches_workspace_checkout(self):
        args = lkg_web_server.build_parser().parse_args([])

        self.assertEqual(Path(args.bridge_sdk_root).name, "Bridge-Python-SDK-Lab")
        self.assertEqual(args.playback_mode, "auto")
        self.assertEqual(args.playback_fps, 0.0)
        self.assertEqual(args.web_preview_fps, 5.0)
        self.assertEqual(args.rtgs_context_loader, "lite")
        self.assertEqual(args.texture_upload_mode, "auto")
        self.assertIsNone(args.texture_upload_cuda_device)

    def test_parser_accepts_cuda_gl_texture_upload_mode(self):
        args = lkg_web_server.build_parser().parse_args(["--texture-upload-mode", "cuda-gl", "--texture-upload-cuda-device", "0"])

        self.assertEqual(args.texture_upload_mode, "cuda-gl")
        self.assertEqual(args.texture_upload_cuda_device, 0)

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

    def test_should_wake_panel_display_uses_periodic_interval(self):
        self.assertTrue(lkg_web_server.should_wake_panel_display(now=10.0, last_wake=None, interval_s=30.0))
        self.assertFalse(lkg_web_server.should_wake_panel_display(now=20.0, last_wake=10.0, interval_s=30.0))
        self.assertTrue(lkg_web_server.should_wake_panel_display(now=40.0, last_wake=10.0, interval_s=30.0))

    def test_poll_panel_window_events_sets_stop_event_when_window_closes(self):
        stop_event = threading.Event()
        window = object()
        glfw = SimpleNamespace(
            poll_events=mock.Mock(),
            window_should_close=mock.Mock(return_value=True),
        )

        keep_running = lkg_web_server.poll_panel_window_events(glfw, window, stop_event)

        self.assertFalse(keep_running)
        self.assertTrue(stop_event.is_set())
        glfw.poll_events.assert_called_once_with()
        glfw.window_should_close.assert_called_once_with(window)

    def test_display_worker_logs_exception_before_stopping_server(self):
        state = lkg_web_server.InteractiveState()
        stop_event = threading.Event()
        worker = lkg_web_server.DisplayWorker(
            args=SimpleNamespace(display_mode="glfw"),
            state=state,
            gpu_lock=threading.Lock(),
            stop_event=stop_event,
        )

        stderr = StringIO()
        with mock.patch.object(worker, "_run_glfw", side_effect=RuntimeError("panel init failed")):
            with redirect_stderr(stderr):
                worker.run()

        self.assertTrue(stop_event.is_set())
        self.assertEqual(state.status_payload()["last_error"], "panel init failed")
        self.assertIn("DisplayWorker failed: panel init failed", stderr.getvalue())

    def test_stop_http_server_skips_shutdown_when_web_thread_is_not_running(self):
        server = SimpleNamespace(shutdown=mock.Mock(), server_close=mock.Mock())
        web_thread = SimpleNamespace(is_alive=mock.Mock(return_value=False), join=mock.Mock())

        lkg_web_server.stop_http_server(server, web_thread)

        server.shutdown.assert_not_called()
        server.server_close.assert_called_once_with()
        web_thread.join.assert_not_called()

    def test_stop_http_server_shutdowns_and_joins_running_web_thread(self):
        server = SimpleNamespace(shutdown=mock.Mock(), server_close=mock.Mock())
        web_thread = SimpleNamespace(is_alive=mock.Mock(return_value=True), join=mock.Mock())

        lkg_web_server.stop_http_server(server, web_thread, timeout=3.0)

        server.shutdown.assert_called_once_with()
        server.server_close.assert_called_once_with()
        web_thread.join.assert_called_once_with(timeout=3.0)

    def test_stop_http_server_still_closes_when_shutdown_is_interrupted(self):
        server = SimpleNamespace(shutdown=mock.Mock(side_effect=KeyboardInterrupt), server_close=mock.Mock())
        web_thread = SimpleNamespace(is_alive=mock.Mock(return_value=True), join=mock.Mock())

        lkg_web_server.stop_http_server(server, web_thread, timeout=3.0)

        server.shutdown.assert_called_once_with()
        server.server_close.assert_called_once_with()
        web_thread.join.assert_called_once_with(timeout=3.0)


if __name__ == "__main__":
    unittest.main()
