#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from lkg_experiment.coherent_default.coherent_raster_experiment import build_experiment_web_assets


DEFAULT_EXPERIMENTS_ROOT = Path("/data/ysj/result/coherent-raster/generated/coherent_raster_experiments")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the static web dashboard for CoherentRaster experiment folders"
    )
    parser.add_argument("--experiments-root", default=str(DEFAULT_EXPERIMENTS_ROOT))
    parser.add_argument(
        "--no-regenerate-previews",
        action="store_true",
        help="Only rebuild index.html/viewer assets; do not regenerate mapping PNG previews from raw_mapping.npz",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    index_path = build_experiment_web_assets(
        args.experiments_root,
        regenerate_previews=not args.no_regenerate_previews,
    )
    print(f"Wrote {index_path}")
    print(f"Serve with: python -m http.server 8000 -d {Path(args.experiments_root).expanduser()}")
    print("Open: http://localhost:8000/index.html")


if __name__ == "__main__":
    main()
