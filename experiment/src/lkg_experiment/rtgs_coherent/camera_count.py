from __future__ import annotations

import argparse
from pathlib import Path

from lkg_experiment.rtgs_coherent.cli import (
    DEFAULT_DNERF_ROOT,
    DEFAULT_N3DV_ROOT,
    DEFAULT_RTGS_CODE_ROOT,
    count_rtgs_cameras,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print the number of RTGS cameras for a split")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--rtgs-code-root", default=str(DEFAULT_RTGS_CODE_ROOT))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DNERF_ROOT))
    parser.add_argument("--n3dv-root", default=str(DEFAULT_N3DV_ROOT))
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--n3dv-frame-index", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    count = count_rtgs_cameras(
        model_path=Path(args.model_path).expanduser(),
        rtgs_code_root=Path(args.rtgs_code_root).expanduser(),
        dataset_root=Path(args.dataset_root).expanduser(),
        n3dv_root=Path(args.n3dv_root).expanduser(),
        config_path=Path(args.config).expanduser() if args.config else None,
        split=args.split,
        n3dv_frame_index=args.n3dv_frame_index,
    )
    print(count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
