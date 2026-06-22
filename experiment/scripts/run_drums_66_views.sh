#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATADIR="${DATADIR:-${DATA_DIR_ROOT:-$HOME/Data/datasets}}"
RESULTDIR="${RESULTDIR:-${RESULT_DIR_ROOT:-$HOME/Data/results}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$RESULTDIR/blender_MCMC100000_init50000/drums/ckpts/ckpt_29999_rank0.pt}"
DATA_DIR="${DATA_DIR:-$DATADIR/nerf_synthetic/drums}"
VIEWPOINT_INDEX_PATH="${VIEWPOINT_INDEX_PATH:-$ROOT_DIR/generated/lkg_go_1440x2560_66_views_balanced.npz}"

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "Missing file: $path" >&2
        exit 1
    fi
}

require_dir() {
    local path="$1"
    if [[ ! -d "$path" ]]; then
        echo "Missing directory: $path" >&2
        exit 1
    fi
}

require_file "$CHECKPOINT_PATH"
require_dir "$DATA_DIR"
require_file "$VIEWPOINT_INDEX_PATH"

cd "$ROOT_DIR"
PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" exec "$PYTHON_BIN" -m lkg_experiment.run_coherent_raster_experiment \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --data-dir "$DATA_DIR" \
    --camera-source dataset \
    --camera-split test \
    --camera-index 0 \
    --width 1440 \
    --height 2560 \
    --views 66 \
    --map-mode file \
    --viewpoint-index-path "$VIEWPOINT_INDEX_PATH" \
    "$@"
