from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
import json
from pathlib import Path


def count_blender_cameras(data_dir: Path | str, split: str) -> int:
    split_name = "test" if split == "auto" else split
    path = Path(data_dir).expanduser() / f"transforms_{split_name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Blender split file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"Blender split file has no frames list: {path}")
    return len(frames)


def split_indices_for_colmap_count(image_count: int, split: str, *, test_every: int) -> list[int]:
    if image_count < 0:
        raise ValueError("image_count must be non-negative")
    if test_every <= 0:
        raise ValueError("test_every must be positive")
    split_name = "val" if split == "auto" else split
    indices = list(range(int(image_count)))
    if split_name == "val":
        return [index for index in indices if index % int(test_every) == 0]
    if split_name == "train":
        return [index for index in indices if index % int(test_every) != 0]
    raise ValueError("COLMAP datasets support split auto, val, or train")


def count_colmap_cameras(data_dir: Path | str, split: str, *, test_every: int = 8) -> int:
    from pycolmap import SceneManager

    data_path = Path(data_dir).expanduser()
    colmap_dir = data_path / "sparse" / "0"
    if not colmap_dir.exists():
        colmap_dir = data_path / "sparse"
    if not colmap_dir.exists():
        raise FileNotFoundError(f"COLMAP directory does not exist: {colmap_dir}")

    with _stdout_to_stderr_fd():
        manager = SceneManager(str(colmap_dir))
        manager.load_images()
    return len(split_indices_for_colmap_count(len(manager.images), split, test_every=test_every))


def read_test_every(cfg_path: Path | str | None, default: int = 8) -> int:
    if cfg_path is None:
        return int(default)
    path = Path(cfg_path).expanduser()
    if not path.is_file():
        return int(default)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith("test_every:"):
                continue
            _, raw_value = stripped.split(":", 1)
            value = raw_value.strip()
            if not value:
                return int(default)
            return int(value)
    return int(default)


def count_dataset_cameras(
    *,
    suite: str,
    data_dir: Path | str,
    split: str,
    cfg_path: Path | str | None = None,
) -> int:
    suite_name = suite.lower()
    if suite_name == "blender":
        return count_blender_cameras(data_dir, split)
    if suite_name in {"mipnerf360", "colmap"}:
        return count_colmap_cameras(data_dir, split, test_every=read_test_every(cfg_path))
    raise ValueError(f"Unsupported suite: {suite}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count dataset cameras for a CoherentRaster experiment split.")
    parser.add_argument("--suite", required=True, choices=("blender", "mipnerf360", "colmap"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--cfg-path")
    return parser


@contextmanager
def _stdout_to_stderr_fd():
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    try:
        os.dup2(stderr_fd, stdout_fd)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, stdout_fd)
        os.close(saved_stdout_fd)


def main() -> None:
    args = build_parser().parse_args()
    count = count_dataset_cameras(
        suite=args.suite,
        data_dir=args.data_dir,
        split=args.split,
        cfg_path=args.cfg_path,
    )
    print(count)


if __name__ == "__main__":
    main()
