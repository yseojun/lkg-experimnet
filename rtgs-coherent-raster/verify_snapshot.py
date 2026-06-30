#!/usr/bin/env python3
from __future__ import annotations

import argparse
import filecmp
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that rtgs-coherent-raster/modified-files matches a target gsplat checkout."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="gsplat",
        help="Path to the target gsplat repository root. Default: ./gsplat",
    )
    args = parser.parse_args()

    archive_root = Path(__file__).resolve().parent
    target_root = Path(args.target).expanduser().resolve()
    manifest = json.loads((archive_root / "manifest.json").read_text(encoding="utf-8"))

    failures: list[str] = []
    for rel in manifest["modified_files"]:
        rel_path = Path(rel)
        archived = archive_root / "modified-files" / rel_path
        target = target_root / rel_path
        if not archived.is_file():
            failures.append(f"missing archived file: {archived}")
            continue
        if not target.is_file():
            failures.append(f"missing target file: {target}")
            continue
        if not filecmp.cmp(archived, target, shallow=False):
            failures.append(f"different: {rel}")

    if failures:
        print("Snapshot verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"Snapshot verification passed for {target_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
