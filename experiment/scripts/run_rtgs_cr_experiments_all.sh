#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -n "${COHERENT_RASTER_PYTHON:-}" && -x "$COHERENT_RASTER_PYTHON" ]]; then
        PYTHON_BIN="$COHERENT_RASTER_PYTHON"
    elif [[ -x "$HOME/.conda/envs/rtgs-coherent-cu121/bin/python" ]]; then
        PYTHON_BIN="$HOME/.conda/envs/rtgs-coherent-cu121/bin/python"
    elif [[ -x "$HOME/anaconda3/envs/rtgs-coherent-cu121/bin/python" ]]; then
        PYTHON_BIN="$HOME/anaconda3/envs/rtgs-coherent-cu121/bin/python"
    elif [[ -x "/etc/anaconda3/envs/rtgs-coherent-cu121/bin/python" ]]; then
        PYTHON_BIN="/etc/anaconda3/envs/rtgs-coherent-cu121/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

MODEL_ROOT="${MODEL_ROOT:-/data/ysj/result/4dgs/RTGS}"
RTGS_CODE_ROOT="${RTGS_CODE_ROOT:-$PROJECT_ROOT/4d-gaussian-splatting}"
RTGS_CODE_POLICY="${RTGS_CODE_POLICY:-as-is}"
GSPLAT_ROOT="${GSPLAT_ROOT:-$PROJECT_ROOT/gsplat}"
DNERF_ROOT="${DNERF_ROOT:-/data/ysj/dataset/dnerf}"
N3DV_ROOT="${N3DV_ROOT:-/data/ysj/dataset/N3DV}"
GENERATED_ROOT="${GENERATED_ROOT:-/data/ysj/result/coherent-raster/generated}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$GENERATED_ROOT/rtgs_cr_experiments}"
VIEWPOINT_INDEX_PATH="${VIEWPOINT_INDEX_PATH:-$GENERATED_ROOT/lkg_go_1440x2560_66_views_lkg_calibration.npz}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$GENERATED_ROOT/torch_extensions_lkg_rtgs/rtgs_official}"

RUN_GROUP="${RUN_GROUP:-rtgs_cr_experiments_$(date +%Y%m%d_%H%M%S)}"
ARTIFACT_DIR="$ARTIFACT_ROOT/$RUN_GROUP"
SUMMARY_TXT="${SUMMARY_TXT:-$ARTIFACT_DIR/summary.txt}"
CHECKPOINT="${CHECKPOINT:-checkpoints/chkpnt_best.pth}"
SPLIT="${SPLIT:-test}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
N3DV_FRAME_INDEX="${N3DV_FRAME_INDEX:-0}"
WIDTH="${WIDTH:-1440}"
HEIGHT="${HEIGHT:-2560}"
VIEWS="${VIEWS:-66}"
EXPERIMENT_ENGINE="one_shot"
CLUSTERS="${CLUSTERS:-1,2,4,8,16}"
ABATION_CLUSTER_DEFAULT="8"
ABLATION_CLUSTER="${ABLATION_CLUSTER:-$ABATION_CLUSTER_DEFAULT}"
VIEW_DEGREE="${VIEW_DEGREE:-53.0}"
ORBIT_DIRECTION="${ORBIT_DIRECTION:--1}"
ORBIT_CENTER_DISTANCE="${ORBIT_CENTER_DISTANCE:-0.0}"
MAP_MODE="${MAP_MODE:-file}"
ASPECT_FIT="${ASPECT_FIT:-contain}"
BACKGROUND="${BACKGROUND:-auto}"
TILE_SIZE="${TILE_SIZE:-16}"
WARMUP_ITERS="${WARMUP_ITERS:-3}"
MEASURE_ITERS="${MEASURE_ITERS:-5}"
METRIC_VIEW_STRIDE="${METRIC_VIEW_STRIDE:-1}"
MAX_METRIC_VIEWS="${MAX_METRIC_VIEWS:-5}"

DRY_RUN="${DRY_RUN:-0}"
REQUIRE_ALL="${REQUIRE_ALL:-0}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
NO_SSIM="${NO_SSIM:-1}"
NO_CROP_TO_FILL="${NO_CROP_TO_FILL:-0}"
SKIP_METRICS="${SKIP_METRICS:-0}"
NO_REFERENCE_INTERLACED="${NO_REFERENCE_INTERLACED:-1}"
SKIP_WEB_ASSETS="${SKIP_WEB_ASSETS:-0}"
WRITE_MAPPING_ARTIFACTS="${WRITE_MAPPING_ARTIFACTS:-0}"
WRITE_MAPPING_PREVIEWS="${WRITE_MAPPING_PREVIEWS:-0}"
APPEND_SUMMARY="${APPEND_SUMMARY:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "Missing file: $path" >&2
        exit 1
    fi
}

load_scenes() {
    if [[ -n "${RTGS_SCENES:-}" ]]; then
        read -r -a scenes <<< "$RTGS_SCENES"
        return
    fi
    if [[ ! -d "$MODEL_ROOT" ]]; then
        echo "Missing model root: $MODEL_ROOT" >&2
        exit 1
    fi

    scenes=()
    local scene_dir
    while IFS= read -r -d '' scene_dir; do
        scenes+=("$(basename "$scene_dir")")
    done < <(find "$MODEL_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
}

contains_word() {
    local needle="$1"
    local haystack="$2"
    local item
    for item in $haystack; do
        if [[ "$item" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

dataset_kind_for_scene() {
    local scene="$1"
    if [[ -n "${DNERF_SCENES:-}" ]] && contains_word "$scene" "$DNERF_SCENES"; then
        echo "dnerf"
        return
    fi
    if [[ -n "${N3DV_SCENES:-}" ]] && contains_word "$scene" "$N3DV_SCENES"; then
        echo "n3dv"
        return
    fi
    if [[ -f "$RTGS_CODE_ROOT/configs/dnerf/$scene.yaml" ]]; then
        echo "dnerf"
        return
    fi
    if [[ -f "$RTGS_CODE_ROOT/configs/dynerf/$scene.yaml" ]]; then
        echo "n3dv"
        return
    fi
    echo "${DEFAULT_DATASET_KIND:-n3dv}"
}

print_command() {
    local -a cmd=("$@")
    printf 'DRY RUN:' >&2
    printf ' %q' "${cmd[@]}" >&2
    printf '\n' >&2
}

if [[ "$MAP_MODE" == "file" ]]; then
    require_file "$VIEWPOINT_INDEX_PATH"
fi

load_scenes
if [[ "${#scenes[@]}" -eq 0 ]]; then
    echo "No RTGS scenes found under $MODEL_ROOT" >&2
    exit 1
fi

planned=0
completed=0
missing=0
failures=0

for scene in "${scenes[@]}"; do
    model_path="$MODEL_ROOT/$scene"
    checkpoint_path="$model_path/$CHECKPOINT"
    dataset_kind="$(dataset_kind_for_scene "$scene")"
    printf -v camera_label "cam%03d" "$CAMERA_INDEX"
    printf -v frame_label "frame%04d" "$N3DV_FRAME_INDEX"
    run_id="${scene}_${EXPERIMENT_ENGINE}_${SPLIT}_${camera_label}_${frame_label}"
    run_dir="$ARTIFACT_DIR/$run_id"

    if [[ ! -f "$checkpoint_path" ]]; then
        echo "[$scene] missing checkpoint: $checkpoint_path" >&2
        missing=$((missing + 1))
        if [[ "$REQUIRE_ALL" == "1" ]]; then
            failures=$((failures + 1))
        fi
        continue
    fi

    cmd=(
        "$PYTHON_BIN"
        "$ROOT_DIR/rtgs_cr_experiment.py"
        --dataset-kind "$dataset_kind"
        --model-path "$model_path"
        --checkpoint "$CHECKPOINT"
        --rtgs-code-root "$RTGS_CODE_ROOT"
        --rtgs-code-policy "$RTGS_CODE_POLICY"
        --torch-extensions-dir "$TORCH_EXTENSIONS_DIR"
        --gsplat-root "$GSPLAT_ROOT"
        --dataset-root "$DNERF_ROOT"
        --n3dv-root "$N3DV_ROOT"
        --artifact-dir "$ARTIFACT_DIR"
        --run-id "$run_id"
        --split "$SPLIT"
        --camera-index "$CAMERA_INDEX"
        --n3dv-frame-index "$N3DV_FRAME_INDEX"
        --width "$WIDTH"
        --height "$HEIGHT"
        --views "$VIEWS"
        --aspect-fit "$ASPECT_FIT"
        --clusters "$CLUSTERS"
        --ablation-cluster "$ABLATION_CLUSTER"
        --tile-size "$TILE_SIZE"
        --view-degree "$VIEW_DEGREE"
        --orbit-direction "$ORBIT_DIRECTION"
        --orbit-center-distance "$ORBIT_CENTER_DISTANCE"
        --background "$BACKGROUND"
        --map-mode "$MAP_MODE"
        --warmup-iters "$WARMUP_ITERS"
        --measure-iters "$MEASURE_ITERS"
        --metric-view-stride "$METRIC_VIEW_STRIDE"
        --max-metric-views "$MAX_METRIC_VIEWS"
    )
    if [[ "$MAP_MODE" == "file" ]]; then
        cmd+=(--viewpoint-index-path "$VIEWPOINT_INDEX_PATH")
    fi
    if [[ "$NO_SSIM" == "1" ]]; then
        cmd+=(--no-ssim)
    fi
    if [[ "$NO_CROP_TO_FILL" == "1" ]]; then
        cmd+=(--no-crop-to-fill)
    fi
    if [[ "$SKIP_METRICS" == "1" ]]; then
        cmd+=(--skip-metrics)
    fi
    if [[ "$NO_REFERENCE_INTERLACED" == "1" ]]; then
        cmd+=(--no-reference-interlaced)
    fi
    if [[ "$SKIP_WEB_ASSETS" == "1" ]]; then
        cmd+=(--skip-web-assets)
    fi
    if [[ "$WRITE_MAPPING_ARTIFACTS" == "1" ]]; then
        cmd+=(--write-mapping-artifacts)
    fi
    if [[ "$WRITE_MAPPING_PREVIEWS" == "1" ]]; then
        cmd+=(--write-mapping-previews)
    fi
    if [[ -n "$EXTRA_ARGS" ]]; then
        read -r -a extra_args <<< "$EXTRA_ARGS"
        cmd+=("${extra_args[@]}")
    fi

    summary_cmd=(
        "$PYTHON_BIN"
        "$ROOT_DIR/src/lkg_experiment/experiment/batch_experiments.py"
        --summary-txt "$SUMMARY_TXT"
        --suite "rtgs"
        --result-group "$RUN_GROUP"
        --scene "$scene"
        --camera-split "$SPLIT"
        --camera-index "$CAMERA_INDEX"
        --run-dir "$run_dir"
    )

    planned=$((planned + 1))
    echo "[$scene] dataset=$dataset_kind run: $run_dir" >&2
    if [[ "$DRY_RUN" == "1" ]]; then
        print_command "${cmd[@]}"
        if [[ "$APPEND_SUMMARY" == "1" ]]; then
            print_command "${summary_cmd[@]}"
        fi
        continue
    fi

    mkdir -p "$ARTIFACT_DIR"
    if PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "${cmd[@]}"; then
        completed=$((completed + 1))
        if [[ "$APPEND_SUMMARY" == "1" ]]; then
            PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "${summary_cmd[@]}"
        fi
    else
        rc=$?
        echo "[$scene] failed with exit code $rc" >&2
        failures=$((failures + 1))
        if [[ "$APPEND_SUMMARY" == "1" ]]; then
            status_cmd=(
                "$PYTHON_BIN"
                "$ROOT_DIR/src/lkg_experiment/experiment/batch_experiments.py"
                --summary-txt "$SUMMARY_TXT"
                --suite "rtgs"
                --result-group "$RUN_GROUP"
                --scene "$scene"
                --camera-split "$SPLIT"
                --camera-index "$CAMERA_INDEX"
                --status "failed"
                --message "exit code $rc"
                --run-dir "$run_dir"
            )
            PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "${status_cmd[@]}" || true
        fi
        if [[ "$STOP_ON_FAILURE" == "1" ]]; then
            exit "$rc"
        fi
    fi
done

echo "Run group: $RUN_GROUP" >&2
echo "Artifact root: $ARTIFACT_DIR" >&2
echo "Summary: $SUMMARY_TXT" >&2
echo "Completed: $completed, planned: $planned, missing: $missing, failures: $failures" >&2

if [[ "$failures" -ne 0 ]]; then
    exit 1
fi
