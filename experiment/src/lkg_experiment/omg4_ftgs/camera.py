from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class TestFrame:
    index: int
    image_path: Path | None
    timestamp: float


@dataclass(frozen=True)
class FtgsCameraSet:
    viewmats: np.ndarray
    K: np.ndarray
    width: int
    height: int
    source_width: int
    source_height: int
    focal: float

    def viewmat_at(self, camera_index: int) -> np.ndarray:
        if int(camera_index) < 0 or int(camera_index) >= int(self.viewmats.shape[0]):
            raise IndexError(f"camera_index {camera_index} out of range for {self.viewmats.shape[0]} cameras")
        return np.ascontiguousarray(self.viewmats[int(camera_index)])


def load_test_frames(data_path: Path | str, *, camera_index: int = 0) -> list[TestFrame]:
    root = Path(data_path).expanduser()
    transforms_path = root / "transforms_test.json"
    if transforms_path.is_file():
        return _load_transform_frames(root, transforms_path)
    return _load_video_frames(root, camera_index=int(camera_index))


def load_pose_camera(data_path: Path | str, *, camera_index: int = 0, resolution: float = 2.0) -> FtgsCameraSet:
    root = Path(data_path).expanduser()
    poses_path = root / "poses_bounds.npy"
    if not poses_path.is_file():
        raise FileNotFoundError(f"poses_bounds.npy not found: {poses_path}")
    if float(resolution) <= 0.0:
        raise ValueError("resolution must be positive")

    arr = np.load(str(poses_path))
    if arr.ndim != 2 or arr.shape[1] < 17:
        raise ValueError(f"poses_bounds.npy must have shape (N,17+): {poses_path}")
    cam = arr[:, :-2].reshape((-1, 3, 5))
    if int(camera_index) < 0 or int(camera_index) >= int(cam.shape[0]):
        raise IndexError(f"camera_index {camera_index} out of range for {cam.shape[0]} cameras")

    source_height, source_width, focal = [float(value) for value in cam[0, :, -1]]
    width = int(round(source_width / float(resolution)))
    height = int(round(source_height / float(resolution)))
    scaled_focal = float(focal) / float(resolution)
    K = np.array(
        [[scaled_focal, 0.0, width / 2.0], [0.0, scaled_focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    c2ws = cam[..., :4]
    c2ws = np.stack([c2ws[..., 1], c2ws[..., 0], -c2ws[..., 2], c2ws[..., 3]], axis=1)
    bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64).reshape((1, 4, 1)).repeat(len(c2ws), 0)
    c2ws = np.concatenate([c2ws, bottom], axis=2)
    viewmats = np.transpose(np.linalg.inv(c2ws), (0, 2, 1)).astype(np.float32)
    return FtgsCameraSet(
        viewmats=np.ascontiguousarray(viewmats),
        K=K,
        width=width,
        height=height,
        source_width=int(round(source_width)),
        source_height=int(round(source_height)),
        focal=float(focal),
    )


def _load_transform_frames(root: Path, transforms_path: Path) -> list[TestFrame]:
    contents = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = []
    for index, frame in enumerate(contents.get("frames", [])):
        rel_path = str(frame.get("file_path", ""))
        image_path = root / rel_path
        if image_path.suffix == "":
            image_path = image_path.with_suffix(".png")
        frames.append(TestFrame(index=index, image_path=image_path, timestamp=float(frame.get("time", 0.0))))
    if not frames:
        raise ValueError(f"transforms_test.json contains no frames: {transforms_path}")
    return frames


def _load_video_frames(root: Path, camera_index: int) -> list[TestFrame]:
    video_path = root / f"cam{int(camera_index):02d}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"neither transforms_test.json nor camera video found: {video_path}")
    metadata = _probe_video(video_path)
    frame_count = int(metadata["frame_count"])
    duration = float(metadata["duration"])
    if frame_count <= 0:
        raise ValueError(f"could not determine frame count for {video_path}")
    timestamps = np.linspace(0.0, duration, frame_count, endpoint=False, dtype=np.float64)
    return [TestFrame(index=index, image_path=video_path, timestamp=float(timestamp)) for index, timestamp in enumerate(timestamps)]


def _probe_video(path: Path) -> dict[str, float | int]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,r_frame_rate,duration",
            "-of",
            "default=noprint_wrappers=1",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    values: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    frame_count = int(values.get("nb_frames") or 0)
    duration = float(values.get("duration") or 0.0)
    if duration <= 0.0:
        duration = frame_count / max(_parse_rate(values.get("r_frame_rate", "30/1")), 1e-8)
    return {"frame_count": frame_count, "duration": duration}


def _parse_rate(value: str) -> float:
    if "/" not in value:
        return float(value)
    num_s, den_s = value.split("/", 1)
    den = float(den_s)
    return float(num_s) / den if den else 0.0
