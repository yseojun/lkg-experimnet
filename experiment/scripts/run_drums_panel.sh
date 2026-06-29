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
DATADIR="${DATADIR:-${DATA_DIR_ROOT:-$HOME/Data/datasets}}"
RESULTDIR="${RESULTDIR:-${RESULT_DIR_ROOT:-$HOME/Data/results}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$RESULTDIR/blender_MCMC100000_init50000/drums/ckpts/ckpt_29999_rank0.pt}"
DATA_DIR="${DATA_DIR:-$DATADIR/nerf_synthetic/drums}"
GENERATED_ROOT="${GENERATED_ROOT:-/data/ysj/result/coherent-raster/generated}"
VIEWPOINT_INDEX_PATH="${VIEWPOINT_INDEX_PATH:-$GENERATED_ROOT/lkg_go_1440x2560_66_views_lkg_calibration.npz}"

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
