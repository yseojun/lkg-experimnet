#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -n "${COHERENT_RASTER_PYTHON:-}" && -x "$COHERENT_RASTER_PYTHON" ]]; then
        PYTHON_BIN="$COHERENT_RASTER_PYTHON"
    elif [[ -x "$HOME/.conda/envs/coherent_raster/bin/python" ]]; then
        PYTHON_BIN="$HOME/.conda/envs/coherent_raster/bin/python"
    elif [[ -x "$HOME/.conda/envs/coherent_raster/bin/python3.11" ]]; then
        PYTHON_BIN="$HOME/.conda/envs/coherent_raster/bin/python3.11"
    elif [[ -x "$HOME/anaconda3/envs/coherent_raster/bin/python" ]]; then
        PYTHON_BIN="$HOME/anaconda3/envs/coherent_raster/bin/python"
    elif [[ -x "/etc/anaconda3/envs/coherent_raster/bin/python" ]]; then
        PYTHON_BIN="/etc/anaconda3/envs/coherent_raster/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

MODEL_ROOT="${MODEL_ROOT:-/data/ysj/result/4dgs/RTGS}"
RTGS_CODE_ROOT="${RTGS_CODE_ROOT:-$PROJECT_ROOT/4d-gaussian-splatting}"
GSPLAT_ROOT="${GSPLAT_ROOT:-$PROJECT_ROOT/gsplat}"
DNERF_ROOT="${DNERF_ROOT:-/data/ysj/dataset/dnerf}"
N3DV_ROOT="${N3DV_ROOT:-/data/ysj/dataset/N3DV}"
GENERATED_ROOT="${GENERATED_ROOT:-/data/ysj/result/coherent-raster/generated}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$GENERATED_ROOT/rtgs_single_views}"

RUN_GROUP="${RUN_GROUP:-rtgs_single_all_cams_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT="${CHECKPOINT:-checkpoints/chkpnt_best.pth}"
SPLIT="${SPLIT:-all}"
N3DV_FRAME_INDEX="${N3DV_FRAME_INDEX:-0}"
N3DV_FRAME_INDICES="${N3DV_FRAME_INDICES:-}"
BACKGROUND="${BACKGROUND:-auto}"
NO_SSIM="${NO_SSIM:-1}"
DRY_RUN="${DRY_RUN:-0}"
REQUIRE_ALL="${REQUIRE_ALL:-0}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -n "$N3DV_FRAME_INDICES" ]]; then
    read -r -a n3dv_frame_indices <<< "${N3DV_FRAME_INDICES//,/ }"
else
    n3dv_frame_indices=("$N3DV_FRAME_INDEX")
fi

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

print_command() {
    local -a cmd=("$@")
    printf 'DRY RUN:' >&2
    printf ' %q' "${cmd[@]}" >&2
    printf '\n' >&2
}

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

    if [[ ! -f "$checkpoint_path" ]]; then
        echo "[$scene] missing checkpoint: $checkpoint_path" >&2
        missing=$((missing + 1))
        if [[ "$REQUIRE_ALL" == "1" ]]; then
            failures=$((failures + 1))
        fi
        continue
    fi

    for n3dv_frame_index in "${n3dv_frame_indices[@]}"; do
        if [[ ! "$n3dv_frame_index" =~ ^-?[0-9]+$ ]]; then
            echo "[$scene] invalid N3DV frame index: $n3dv_frame_index" >&2
            failures=$((failures + 1))
            continue
        fi
        if [[ -n "$N3DV_FRAME_INDICES" ]]; then
            if ((n3dv_frame_index >= 0)); then
                printf -v frame_dir "frame_%04d" "$n3dv_frame_index"
            else
                frame_dir="frame_all"
            fi
            output_dir="$OUTPUT_ROOT/$RUN_GROUP/$scene/$frame_dir"
        else
            output_dir="$OUTPUT_ROOT/$RUN_GROUP/$scene"
        fi
        cmd=(
            "$PYTHON_BIN"
            "$ROOT_DIR/rtgs_coherent.py"
            --render-mode single-all-cams
            --model-path "$model_path"
            --checkpoint "$CHECKPOINT"
            --rtgs-code-root "$RTGS_CODE_ROOT"
            --gsplat-root "$GSPLAT_ROOT"
            --dataset-root "$DNERF_ROOT"
            --n3dv-root "$N3DV_ROOT"
            --output-dir "$output_dir"
            --split "$SPLIT"
            --n3dv-frame-index "$n3dv_frame_index"
            --background "$BACKGROUND"
        )
        if [[ -n "${CAMERA_INDICES:-}" ]]; then
            cmd+=(--camera-indices "$CAMERA_INDICES")
        elif [[ -n "${CAMERA_COUNT:-}" ]]; then
            camera_indices=()
            for ((camera_index = 0; camera_index < CAMERA_COUNT; camera_index++)); do
                camera_indices+=("$camera_index")
            done
            cmd+=(--camera-indices "$(IFS=,; echo "${camera_indices[*]}")")
        fi
        if [[ "$NO_SSIM" == "1" ]]; then
            cmd+=(--no-ssim)
        fi
        if [[ -n "$EXTRA_ARGS" ]]; then
            read -r -a extra_args <<< "$EXTRA_ARGS"
            cmd+=("${extra_args[@]}")
        fi

        planned=$((planned + 1))
        echo "[$scene] output: $output_dir" >&2
        if [[ "$DRY_RUN" == "1" ]]; then
            print_command "${cmd[@]}"
            continue
        fi

        mkdir -p "$output_dir"
        if PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "${cmd[@]}"; then
            completed=$((completed + 1))
        else
            rc=$?
            echo "[$scene] failed with exit code $rc" >&2
            failures=$((failures + 1))
            if [[ "$STOP_ON_FAILURE" == "1" ]]; then
                exit "$rc"
            fi
        fi
    done
done

echo "Run group: $RUN_GROUP" >&2
echo "Output root: $OUTPUT_ROOT/$RUN_GROUP" >&2
echo "Completed: $completed, planned: $planned, missing: $missing, failures: $failures" >&2

if [[ "$failures" -ne 0 ]]; then
    exit 1
fi
