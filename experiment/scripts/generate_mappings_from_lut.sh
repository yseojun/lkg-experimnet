#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$HOME/anaconda3/envs/coherent_raster/bin/python" ]]; then
        PYTHON_BIN="$HOME/anaconda3/envs/coherent_raster/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

GENERATED_ROOT="${GENERATED_ROOT:-/data/ysj/result/coherent-raster/generated}"
VIEWPOINT_INDEX_PATH="${VIEWPOINT_INDEX_PATH:-$GENERATED_ROOT/lkg_go_1440x2560_66_views_lkg_calibration.npz}"
WIDTH="${WIDTH:-1440}"
HEIGHT="${HEIGHT:-2560}"
VIEWS="${VIEWS:-66}"
CLUSTERS="${CLUSTERS:-2,4,8,16}"
ABLATION_CLUSTER="${ABLATION_CLUSTER:-8}"

lut_stem="$(basename "$VIEWPOINT_INDEX_PATH")"
lut_stem="${lut_stem%.*}"
OUTPUT_DIR="${OUTPUT_DIR:-$GENERATED_ROOT/mapping_artifacts/$lut_stem}"

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m lkg_experiment.generate_mapping_artifacts \
    --viewpoint-index-path "$VIEWPOINT_INDEX_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --views "$VIEWS" \
    --clusters "$CLUSTERS" \
    --ablation-cluster "$ABLATION_CLUSTER" \
    "$@"
