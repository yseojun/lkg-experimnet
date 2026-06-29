#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -n "${RTGS_COHERENT_PYTHON:-}" && -x "$RTGS_COHERENT_PYTHON" ]]; then
        PYTHON_BIN="$RTGS_COHERENT_PYTHON"
    elif [[ -x "$HOME/.conda/envs/rtgs-coherent-cu121/bin/python" ]]; then
        PYTHON_BIN="$HOME/.conda/envs/rtgs-coherent-cu121/bin/python"
    elif [[ -x "$HOME/anaconda3/envs/rtgs-coherent-cu121/bin/python" ]]; then
        PYTHON_BIN="$HOME/anaconda3/envs/rtgs-coherent-cu121/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

MODEL_ROOT="${MODEL_ROOT:-/data/ysj/result/4dgs/RTGS}"
RTGS_CODE_ROOT="${RTGS_CODE_ROOT:-$PROJECT_ROOT/4d-gaussian-splatting}"
DNERF_ROOT="${DNERF_ROOT:-/data/ysj/dataset/dnerf}"
N3DV_ROOT="${N3DV_ROOT:-/data/ysj/dataset/N3DV}"
GENERATED_ROOT="${GENERATED_ROOT:-/data/ysj/result/coherent-raster/generated}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$GENERATED_ROOT/rtgs_official_1view_videos}"

RUN_GROUP="${RUN_GROUP:-rtgs_1view_videos_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT="${CHECKPOINT:-checkpoints/chkpnt_best.pth}"
SPLIT="${SPLIT:-test}"
FPS="${FPS:-30}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_FRAMES="${MAX_FRAMES:-0}"
BACKGROUND="${BACKGROUND:-auto}"
NO_SSIM="${NO_SSIM:-1}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
RTGS_CODE_POLICY="${RTGS_CODE_POLICY:-clean}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
EXTRA_RENDER_ARGS="${EXTRA_RENDER_ARGS:-}"
EXTRA_FFMPEG_ARGS="${EXTRA_FFMPEG_ARGS:-}"

DNERF_SCENES="${DNERF_SCENES:-bouncingballs hellwarrior hook jumpingjacks lego mutant standup trex}"
N3DV_SCENES="${N3DV_SCENES:-coffee_martini cook_spinach cut_roasted_beef flame_salmon flame_steak sear_steak}"

if [[ "$FRAME_STRIDE" -lt 1 ]]; then
    echo "FRAME_STRIDE must be >= 1" >&2
    exit 1
fi

if [[ "$MAX_FRAMES" -lt 0 ]]; then
    echo "MAX_FRAMES must be >= 0" >&2
    exit 1
fi

if [[ "$DRY_RUN" != "1" ]] && ! command -v "$FFMPEG_BIN" >/dev/null 2>&1; then
    echo "Missing ffmpeg binary: $FFMPEG_BIN" >&2
    exit 1
fi

print_command() {
    local -a cmd=("$@")
    printf 'DRY RUN:' >&2
    printf ' %q' "${cmd[@]}" >&2
    printf '\n' >&2
}

n3dv_dataset_scene() {
    case "$1" in
        flame_salmon)
            printf '%s\n' "flame_salmon_1"
            ;;
        *)
            printf '%s\n' "$1"
            ;;
    esac
}

emit_selected_indices() {
    local ordinal=0
    local emitted=0
    local value
    while IFS= read -r value; do
        if ((MAX_FRAMES > 0 && emitted >= MAX_FRAMES)); then
            ordinal=$((ordinal + 1))
            continue
        fi
        if ((ordinal % FRAME_STRIDE == 0)); then
            printf '%s\n' "$value"
            emitted=$((emitted + 1))
        fi
        ordinal=$((ordinal + 1))
    done
}

list_dnerf_camera_indices() {
    local scene="$1"
    local scene_root="$DNERF_ROOT/$scene"
    local transforms_file="$scene_root/transforms_test.json"
    if [[ "$scene" == "lego" && -f "$scene_root/transforms_val.json" ]]; then
        transforms_file="$scene_root/transforms_val.json"
    fi
    if [[ ! -f "$transforms_file" ]]; then
        echo "Missing dnerf transforms file: $transforms_file" >&2
        return 1
    fi
    "$PYTHON_BIN" - "$transforms_file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
frames = json.loads(path.read_text(encoding="utf-8")).get("frames", [])
for index in range(len(frames)):
    print(index)
PY
}

list_n3dv_frame_indices() {
    local scene="$1"
    local dataset_scene
    dataset_scene="$(n3dv_dataset_scene "$scene")"
    local image_dir="$N3DV_ROOT/$dataset_scene/cam00/images"
    if [[ ! -d "$image_dir" ]]; then
        echo "Missing N3DV cam00 image directory: $image_dir" >&2
        return 1
    fi
    local image_path
    while IFS= read -r -d '' image_path; do
        local stem
        stem="$(basename "$image_path" .png)"
        if [[ "$stem" =~ ^[0-9]+$ ]]; then
            printf '%d\n' "$((10#$stem))"
        fi
    done < <(find "$image_dir" -maxdepth 1 -type f -name '*.png' -print0 | sort -z)
}

render_frame() {
    local dataset_kind="$1"
    local scene="$2"
    local frame_value="$3"
    local frame_number="$4"
    local scene_output_root="$5"
    local padded_frame
    printf -v padded_frame "%04d" "$frame_number"
    local render_output_dir="$scene_output_root/renders/frame_$padded_frame"
    local frames_dir="$scene_output_root/frames"
    local frame_png="$frames_dir/frame_$padded_frame.png"
    local model_path="$MODEL_ROOT/$scene"
    local checkpoint_path="$model_path/$CHECKPOINT"

    if [[ ! -f "$checkpoint_path" ]]; then
        echo "[$scene] missing checkpoint: $checkpoint_path" >&2
        return 1
    fi

    local -a cmd=(
        "$PYTHON_BIN"
        "$ROOT_DIR/rtgs_official_1view.py"
        --dataset-kind "$dataset_kind"
        --model-path "$model_path"
        --checkpoint "$CHECKPOINT"
        --rtgs-code-root "$RTGS_CODE_ROOT"
        --rtgs-code-policy "$RTGS_CODE_POLICY"
        --output-dir "$render_output_dir"
        --split "$SPLIT"
        --background "$BACKGROUND"
    )
    if [[ "$dataset_kind" == "dnerf" ]]; then
        cmd+=(--dataset-root "$DNERF_ROOT" --camera-index "$frame_value")
    else
        cmd+=(--n3dv-root "$N3DV_ROOT" --camera-index 0 --n3dv-frame-index "$frame_value")
    fi
    if [[ "$NO_SSIM" == "1" ]]; then
        cmd+=(--no-ssim)
    fi
    if [[ -n "$EXTRA_RENDER_ARGS" ]]; then
        read -r -a extra_render_args <<< "$EXTRA_RENDER_ARGS"
        cmd+=("${extra_render_args[@]}")
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        print_command "${cmd[@]}"
        print_command cp "$render_output_dir/rtgs_official_render.png" "$frame_png"
        return 0
    fi

    mkdir -p "$render_output_dir" "$frames_dir"
    if PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "${cmd[@]}"; then
        cp "$render_output_dir/rtgs_official_render.png" "$frame_png"
    else
        return "$?"
    fi
}

encode_video() {
    local scene="$1"
    local scene_output_root="$2"
    local frames_dir="$scene_output_root/frames"
    local video_path="$scene_output_root/${scene}_1view.mp4"
    local -a cmd=(
        "$FFMPEG_BIN"
        -y
        -framerate "$FPS"
        -i "$frames_dir/frame_%04d.png"
        -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
        -pix_fmt yuv420p
    )
    if [[ -n "$EXTRA_FFMPEG_ARGS" ]]; then
        read -r -a extra_ffmpeg_args <<< "$EXTRA_FFMPEG_ARGS"
        cmd+=("${extra_ffmpeg_args[@]}")
    fi
    cmd+=("$video_path")

    if [[ "$DRY_RUN" == "1" ]]; then
        print_command "${cmd[@]}"
        return 0
    fi

    "${cmd[@]}"
}

run_scene() {
    local dataset_kind="$1"
    local scene="$2"
    local scene_output_root="$OUTPUT_ROOT/$RUN_GROUP/$scene"
    local -a indices=()
    if [[ "$dataset_kind" == "dnerf" ]]; then
        mapfile -t indices < <(list_dnerf_camera_indices "$scene" | emit_selected_indices)
    else
        mapfile -t indices < <(list_n3dv_frame_indices "$scene" | emit_selected_indices)
    fi
    if [[ "${#indices[@]}" -eq 0 ]]; then
        echo "[$scene] no frames selected" >&2
        return 1
    fi

    echo "[$scene] dataset=$dataset_kind frames=${#indices[@]} output=$scene_output_root" >&2
    local frame_number=0
    local frame_value
    for frame_value in "${indices[@]}"; do
        if render_frame "$dataset_kind" "$scene" "$frame_value" "$frame_number" "$scene_output_root"; then
            frame_number=$((frame_number + 1))
        else
            local rc=$?
            echo "[$scene] frame $frame_value failed with exit code $rc" >&2
            return "$rc"
        fi
    done
    encode_video "$scene" "$scene_output_root"
}

planned=0
completed=0
failures=0

read -r -a dnerf_scenes <<< "$DNERF_SCENES"
read -r -a n3dv_scenes <<< "$N3DV_SCENES"

for scene in "${dnerf_scenes[@]}"; do
    planned=$((planned + 1))
    if run_scene "dnerf" "$scene"; then
        completed=$((completed + 1))
    else
        failures=$((failures + 1))
        if [[ "$STOP_ON_FAILURE" == "1" ]]; then
            exit 1
        fi
    fi
done

for scene in "${n3dv_scenes[@]}"; do
    planned=$((planned + 1))
    if run_scene "n3dv" "$scene"; then
        completed=$((completed + 1))
    else
        failures=$((failures + 1))
        if [[ "$STOP_ON_FAILURE" == "1" ]]; then
            exit 1
        fi
    fi
done

echo "Run group: $RUN_GROUP" >&2
echo "Output root: $OUTPUT_ROOT/$RUN_GROUP" >&2
echo "Completed scenes: $completed, planned scenes: $planned, failures: $failures" >&2

if [[ "$failures" -ne 0 ]]; then
    exit 1
fi
