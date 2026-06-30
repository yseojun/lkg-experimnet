#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"

first_existing_dir() {
    local candidate
    for candidate in "$@"; do
        if [[ -n "$candidate" && -d "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    echo "$1"
}

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ "${CONDA_DEFAULT_ENV:-}" == "rtgs-coherent-cu121" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        PYTHON_BIN="$CONDA_PREFIX/bin/python"
    elif [[ "${CONDA_DEFAULT_ENV:-}" == "coherent_raster" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        PYTHON_BIN="$CONDA_PREFIX/bin/python"
    elif [[ -x "$HOME/.conda/envs/rtgs-coherent-cu121/bin/python" ]]; then
        PYTHON_BIN="$HOME/.conda/envs/rtgs-coherent-cu121/bin/python"
    elif [[ -x "$HOME/anaconda3/envs/rtgs-coherent-cu121/bin/python" ]]; then
        PYTHON_BIN="$HOME/anaconda3/envs/rtgs-coherent-cu121/bin/python"
    elif [[ -x "/etc/anaconda3/envs/rtgs-coherent-cu121/bin/python" ]]; then
        PYTHON_BIN="/etc/anaconda3/envs/rtgs-coherent-cu121/bin/python"
    elif [[ -x "$HOME/.conda/envs/coherent_raster/bin/python" ]]; then
        PYTHON_BIN="$HOME/.conda/envs/coherent_raster/bin/python"
    elif [[ -x "$HOME/anaconda3/envs/coherent_raster/bin/python" ]]; then
        PYTHON_BIN="$HOME/anaconda3/envs/coherent_raster/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi
DATADIR="$(first_existing_dir "${DATADIR:-}" "${DATA_DIR_ROOT:-}" /data/ysj/dataset "$HOME/Data/datasets" "$HOME/data/dataset" /data/dataset)"
RESULTDIR="$(first_existing_dir "${RESULTDIR:-}" "${RESULT_DIR_ROOT:-}" /data/ysj/result/coherent-raster /data/ysj/results/coherent-raster "$HOME/Data/results" "$HOME/data/result" /data/result)"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$RESULTDIR/blender_MCMC100000_init50000/drums/ckpts/ckpt_29999_rank0.pt}"
DATA_DIR="${DATA_DIR:-$DATADIR/nerf_synthetic/drums}"
GENERATED_ROOT="${GENERATED_ROOT:-/data/ysj/result/coherent-raster/generated}"
VIEWPOINT_INDEX_PATH="${VIEWPOINT_INDEX_PATH:-$GENERATED_ROOT/lkg_go_1440x2560_66_views_lkg_calibration.npz}"
GSPLAT_ROOT="${GSPLAT_ROOT:-$PROJECT_ROOT/gsplat}"
BRIDGE_SDK_ROOT="${BRIDGE_SDK_ROOT:-$PROJECT_ROOT/Bridge-Python-SDK-Lab}"

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
    --gsplat-root "$GSPLAT_ROOT" \
    --bridge-sdk-root "$BRIDGE_SDK_ROOT" \
    --data-dir "$DATA_DIR" \
    --camera-source dataset \
    --camera-split test \
    --camera-index "${CAMERA_INDEX:-0}" \
    --views "${VIEWS:-66}" \
    --map-mode file \
    --viewpoint-index-path "$VIEWPOINT_INDEX_PATH" \
    --coherent-cluster-size "${CLUSTER:-8}" \
    "$@"
