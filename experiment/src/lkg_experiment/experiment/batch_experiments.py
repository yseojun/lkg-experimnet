from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_COLUMNS = [
    "status",
    "suite",
    "result_group",
    "scene",
    "camera_split",
    "camera_index",
    "output_prefix",
    "run_id",
    "variant",
    "group",
    "cluster_size",
    "use_remapping",
    "reuse_enabled",
    "fps",
    "frame_ms",
    "peak_vram_gb",
    "psnr_mean",
    "psnr_std",
    "ssim_mean",
    "ssim_std",
    "lpips_mean",
    "lpips_std",
    "metric_view_count",
    "artifact_root",
    "metrics_csv",
    "message",
]


def status_row(
    *,
    status: str,
    suite: str,
    result_group: str,
    scene: str,
    camera_split: str = "",
    camera_index: int | str | None = None,
    output_prefix: str = "",
    message: str = "",
    run_dir: Path | str | None = None,
) -> dict[str, str]:
    row = {column: "" for column in SUMMARY_COLUMNS}
    row.update(
        {
            "status": str(status),
            "suite": str(suite),
            "result_group": str(result_group),
            "scene": str(scene),
            "camera_split": str(camera_split),
            "camera_index": "" if camera_index is None else str(camera_index),
            "output_prefix": str(output_prefix),
            "message": str(message),
        }
    )
    if run_dir is not None:
        run_path = Path(run_dir).expanduser()
        row["run_id"] = run_path.name
        row["artifact_root"] = str(run_path)
    return row


def metrics_rows_from_run(
    run_dir: Path | str,
    *,
    suite: str,
    result_group: str,
    scene: str,
    camera_split: str = "",
    camera_index: int | str | None = None,
    output_prefix: str = "",
    message: str = "",
) -> list[dict[str, str]]:
    run_path = Path(run_dir).expanduser()
    metrics_path = run_path / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_path}")

    manifest = _read_manifest(run_path / "manifest.json")
    run_id = str(manifest.get("run_id") or run_path.name)
    artifact_root = str(manifest.get("artifact_root") or run_path)

    rows: list[dict[str, str]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for metric_row in reader:
            if not _matches_camera(metric_row, camera_split=camera_split, camera_index=camera_index, output_prefix=output_prefix):
                continue
            row = status_row(
                status="ok",
                suite=suite,
                result_group=result_group,
                scene=scene,
                camera_split=camera_split,
                camera_index=camera_index,
                output_prefix=output_prefix,
                message=message,
            )
            for column in SUMMARY_COLUMNS:
                if column in metric_row and metric_row[column] is not None:
                    row[column] = str(metric_row[column])
            row["status"] = "ok"
            row["suite"] = str(suite)
            row["result_group"] = str(result_group)
            row["scene"] = str(scene)
            row["camera_split"] = str(camera_split)
            row["camera_index"] = "" if camera_index is None else str(camera_index)
            row["output_prefix"] = str(metric_row.get("output_prefix", output_prefix) or "")
            row["run_id"] = run_id
            row["artifact_root"] = artifact_root
            row["metrics_csv"] = str(metrics_path)
            rows.append(row)

    if not rows:
        rows.append(
            status_row(
                status="no_metrics",
                suite=suite,
                result_group=result_group,
                scene=scene,
                camera_split=camera_split,
                camera_index=camera_index,
                output_prefix=output_prefix,
                message=f"metrics.csv has no rows: {metrics_path}",
                run_dir=run_path,
            )
        )
    return rows


def write_summary_tsv(rows: Sequence[Mapping[str, Any]], output_path: Path | str) -> Path:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        _write_summary_rows(rows, f, include_header=True)
    return path


def append_summary_rows(rows: Sequence[Mapping[str, Any]], output_path: Path | str) -> Path:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    include_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        _write_summary_rows(rows, f, include_header=include_header)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append one CoherentRaster experiment result to an Excel-paste TSV.")
    parser.add_argument("--summary-txt", required=True, help="Output TSV .txt file")
    parser.add_argument("--suite", required=True, help="Dataset suite label, e.g. blender or mipnerf360")
    parser.add_argument("--result-group", required=True, help="Result root group, e.g. blender_MCMC500000")
    parser.add_argument("--scene", required=True, help="Scene name")
    parser.add_argument("--camera-split", default="", help="Dataset split used for the camera")
    parser.add_argument("--camera-index", default="", help="Camera index within the selected split")
    parser.add_argument("--output-prefix", default="", help="Optional output_prefix row filter")
    parser.add_argument("--run-dir", help="Experiment artifact run directory")
    parser.add_argument("--status", default="ok", help="ok, missing, failed, or another status label")
    parser.add_argument("--message", default="", help="Optional status/failure message")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.status == "ok":
        if not args.run_dir:
            raise SystemExit("--run-dir is required when --status ok")
        rows = metrics_rows_from_run(
            args.run_dir,
            suite=args.suite,
            result_group=args.result_group,
            scene=args.scene,
            camera_split=args.camera_split,
            camera_index=args.camera_index,
            output_prefix=args.output_prefix,
            message=args.message,
        )
    else:
        rows = [
            status_row(
                status=args.status,
                suite=args.suite,
                result_group=args.result_group,
                scene=args.scene,
                camera_split=args.camera_split,
                camera_index=args.camera_index,
                output_prefix=args.output_prefix,
                message=args.message,
                run_dir=args.run_dir,
            )
        ]
    append_summary_rows(rows, args.summary_txt)


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    return value if isinstance(value, dict) else {}


def _write_summary_rows(rows: Sequence[Mapping[str, Any]], file_obj, *, include_header: bool) -> None:
    writer = csv.DictWriter(
        file_obj,
        fieldnames=SUMMARY_COLUMNS,
        delimiter="\t",
        extrasaction="ignore",
        lineterminator="\n",
    )
    if include_header:
        writer.writeheader()
    for row in rows:
        writer.writerow({column: _tsv_ready(row.get(column, "")) for column in SUMMARY_COLUMNS})


def _tsv_ready(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _matches_camera(metric_row: Mapping[str, Any], *, camera_split: str, camera_index: int | str | None, output_prefix: str = "") -> bool:
    wanted_split = str(camera_split)
    wanted_index = "" if camera_index is None else str(camera_index)
    wanted_prefix = str(output_prefix)
    row_split = str(metric_row.get("camera_split", "") or "")
    row_index = str(metric_row.get("camera_index", "") or "")
    row_prefix = str(metric_row.get("output_prefix", "") or "")
    if wanted_split and row_split and row_split != wanted_split:
        return False
    if wanted_index and row_index and row_index != wanted_index:
        return False
    if wanted_prefix and row_prefix and row_prefix != wanted_prefix:
        return False
    return True


if __name__ == "__main__":
    main()
