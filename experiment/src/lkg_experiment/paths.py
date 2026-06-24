from __future__ import annotations

import os
from pathlib import Path


DEFAULT_LKG_DATASET_BASE = Path("/data/ysj/dataset")
DEFAULT_LKG_RESULT_BASE = Path("/data/ysj/result")
DEFAULT_VIEWPOINT_INDEX_NAME = "lkg_go_1440x2560_66_views_lkg_calibration.npz"


def env_path(*names: str) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return None


def lkg_dataset_base() -> Path:
    return env_path("LKG_DATASET_BASE") or DEFAULT_LKG_DATASET_BASE


def lkg_result_base() -> Path:
    return env_path("LKG_RESULT_BASE") or DEFAULT_LKG_RESULT_BASE


def generated_root() -> Path:
    return env_path("LKG_GENERATED_DIR", "GENERATED_DIR") or lkg_result_base() / "generated"


def coherent_raster_experiments_root() -> Path:
    return generated_root() / "coherent_raster_experiments"


def default_viewpoint_index_path() -> Path:
    return generated_root() / DEFAULT_VIEWPOINT_INDEX_NAME
