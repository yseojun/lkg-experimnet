#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$HOME/anaconda3/envs/coherent_raster/bin/python" ]]; then
        PYTHON_BIN="$HOME/anaconda3/envs/coherent_raster/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi
LKG_DATASET_BASE="${LKG_DATASET_BASE:-/data/ysj/dataset}"
LKG_RESULT_BASE="${LKG_RESULT_BASE:-/data/ysj/result}"
GENERATED_DIR="${LKG_GENERATED_DIR:-${GENERATED_DIR:-$LKG_RESULT_BASE/generated}}"

resolve_result_root() {
    local base="$1"
    if [[ -d "$base/blender_MCMC100000_init50000" ]]; then
        printf '%s\n' "$base"
    elif [[ -d "$base/coherent-raster" ]]; then
        printf '%s\n' "$base/coherent-raster"
    else
        printf '%s\n' "$base"
    fi
}

DATADIR="${DATADIR:-${DATA_DIR_ROOT:-$LKG_DATASET_BASE}}"
RESULTDIR="$(resolve_result_root "${RESULTDIR:-${RESULT_DIR_ROOT:-$LKG_RESULT_BASE}}")"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$RESULTDIR/blender_MCMC100000_init50000/drums/ckpts/ckpt_29999_rank0.pt}"
DATA_DIR="${DATA_DIR:-$DATADIR/nerf_synthetic/drums}"
VIEWPOINT_INDEX_PATH="${VIEWPOINT_INDEX_PATH:-$GENERATED_DIR/lkg_go_1440x2560_66_views_lkg_calibration.npz}"

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
PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" exec "$PYTHON_BIN" -m lkg_experiment.render_looking_glass \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --data-dir "$DATA_DIR" \
    --camera-source dataset \
    --camera-split test \
    --camera-index "${CAMERA_INDEX:-0}" \
    --views "${VIEWS:-66}" \
    --map-mode file \
    --viewpoint-index-path "$VIEWPOINT_INDEX_PATH" \
    --coherent-cluster-size "${CLUSTER:-8}" \
    "$@"
