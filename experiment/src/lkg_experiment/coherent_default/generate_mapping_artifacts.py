from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from lkg_experiment.coherent_default.coherent_raster_experiment import (
    DEFAULT_CLUSTERS,
    ArtifactWriter,
    build_experiment_variants,
    cluster_index_from_view_index,
    compact_viewpoint_index,
    load_viewpoint_index_file,
    parse_cluster_values,
)


def generate_mapping_artifacts(
    *,
    viewpoint_index_path: Path | str,
    output_dir: Path | str,
    width: int,
    height: int,
    views: int,
    clusters: str | Iterable[int] = DEFAULT_CLUSTERS,
    ablation_cluster: int = 8,
    compact: bool = True,
    write_previews: bool = True,
) -> list[Path]:
    viewpoint_index, file_view_count, _metadata = load_viewpoint_index_file(viewpoint_index_path, width, height)
    if file_view_count is not None and int(file_view_count) != int(views):
        raise ValueError(f"viewpoint index view_count={file_view_count} but views={views}")

    source_viewpoint_index = viewpoint_index.copy()
    view_labels = np.arange(int(views), dtype=np.int32)
    if compact:
        viewpoint_index, view_labels = compact_viewpoint_index(viewpoint_index, int(views))

    variants = build_experiment_variants(
        parse_cluster_values(clusters),
        ablation_cluster=int(ablation_cluster),
        include_without_remap=True,
        include_without_reuse=True,
    )
    cluster_indices = {
        variant.name: cluster_index_from_view_index(viewpoint_index, variant.cluster_size)
        for variant in variants
    }

    writer = ArtifactWriter(output_dir)
    written: list[Path] = [
        writer.write_mapping_npz(
            viewpoint_index,
            cluster_indices,
            view_labels=view_labels,
            source_view_index_hwc=source_viewpoint_index,
        )
    ]

    metadata = writer.save_mapping_values("view_index", viewpoint_index)
    written.append(writer.root / metadata["path"])
    if not np.array_equal(source_viewpoint_index, viewpoint_index):
        metadata = writer.save_mapping_values("source_view_index", source_viewpoint_index)
        written.append(writer.root / metadata["path"])

    if write_previews:
        written.extend(writer.save_index_previews("view_index", viewpoint_index))
        if not np.array_equal(source_viewpoint_index, viewpoint_index):
            written.extend(writer.save_index_previews("source_view_index", source_viewpoint_index))
        for variant in variants:
            written.extend(writer.save_index_previews(f"{variant.name}_cluster_index", cluster_indices[variant.name]))

    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate mapping artifacts from a Looking Glass LUT npz.")
    parser.add_argument("--viewpoint-index-path", required=True)
    parser.add_argument("--output-dir", required=True, help="Artifact root where mappings/ will be created")
    parser.add_argument("--width", default=1440, type=int)
    parser.add_argument("--height", default=2560, type=int)
    parser.add_argument("--views", default=66, type=int)
    parser.add_argument("--clusters", default="2,4,8,16")
    parser.add_argument("--ablation-cluster", default=8, type=int)
    parser.add_argument("--no-compact-view-index", action="store_true")
    parser.add_argument("--no-previews", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written = generate_mapping_artifacts(
        viewpoint_index_path=args.viewpoint_index_path,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        views=args.views,
        clusters=args.clusters,
        ablation_cluster=args.ablation_cluster,
        compact=not args.no_compact_view_index,
        write_previews=not args.no_previews,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
