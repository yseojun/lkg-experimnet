#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

lkg_66_views_init() {
    local default_run_group="$1"

    if [[ -z "${PYTHON_BIN:-}" ]]; then
        if [[ -x "$HOME/anaconda3/envs/coherent_raster/bin/python" ]]; then
            PYTHON_BIN="$HOME/anaconda3/envs/coherent_raster/bin/python"
        else
            PYTHON_BIN="python3"
        fi
    fi

    DATADIR="${DATADIR:-${DATA_DIR_ROOT:-$HOME/Data/datasets}}"
    RESULTDIR="${RESULTDIR:-${RESULT_DIR_ROOT:-$HOME/Data/results}}"
    ARTIFACT_DIR="${ARTIFACT_DIR:-$ROOT_DIR/generated/coherent_raster_experiments}"
    VIEWPOINT_INDEX_PATH="${VIEWPOINT_INDEX_PATH:-$ROOT_DIR/generated/lkg_go_1440x2560_66_views_lkg_calibration.npz}"

    RUN_GROUP="${RUN_GROUP:-$default_run_group}"
    SUMMARY_TXT="${SUMMARY_TXT:-$ARTIFACT_DIR/${RUN_GROUP}_metrics.txt}"

    WIDTH="${WIDTH:-1440}"
    HEIGHT="${HEIGHT:-2560}"
    VIEWS="${VIEWS:-66}"
    CAMERA_INDEX="${CAMERA_INDEX:-}"
    CAMERA_INDEXES="${CAMERA_INDEXES:-${CAMERA_INDICES:-}}"
    CHECKPOINT_ITERATION="${CHECKPOINT_ITERATION:-29999}"
    CHECKPOINT_RANK="${CHECKPOINT_RANK:-0}"

    APPEND_SUMMARY="${APPEND_SUMMARY:-0}"
    DRY_RUN="${DRY_RUN:-0}"
    REQUIRE_ALL="${REQUIRE_ALL:-0}"
    STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"

    if [[ "$APPEND_SUMMARY" != "1" ]]; then
        mkdir -p "$(dirname "$SUMMARY_TXT")"
        : > "$SUMMARY_TXT"
    fi

    lkg_66_views_require_file "$VIEWPOINT_INDEX_PATH"

    failures=0
    missing=0
    planned=0
    completed=0
}

lkg_66_views_require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "Missing file: $path" >&2
        exit 1
    fi
}

lkg_66_views_append_status() {
    local status="$1"
    local suite="$2"
    local result_group="$3"
    local scene="$4"
    local camera_split="$5"
    local camera_index="$6"
    local message="$7"
    PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m lkg_experiment.batch_experiments \
        --summary-txt "$SUMMARY_TXT" \
        --suite "$suite" \
        --result-group "$result_group" \
        --scene "$scene" \
        --camera-split "$camera_split" \
        --camera-index "$camera_index" \
        --status "$status" \
        --message "$message"
}

lkg_66_views_append_metrics() {
    local suite="$1"
    local result_group="$2"
    local scene="$3"
    local camera_split="$4"
    local camera_index="$5"
    local run_dir="$6"
    PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m lkg_experiment.batch_experiments \
        --summary-txt "$SUMMARY_TXT" \
        --suite "$suite" \
        --result-group "$result_group" \
        --scene "$scene" \
        --camera-split "$camera_split" \
        --camera-index "$camera_index" \
        --run-dir "$run_dir"
}

lkg_66_views_sanitize_run_id() {
    local value="$1"
    value="${value//[^A-Za-z0-9_.-]/_}"
    echo "$value"
}

lkg_66_views_run_experiment_unit() {
    local suite="$1"
    local result_group="$2"
    local scene="$3"
    local data_family="$4"
    local camera_split="$5"
    shift 5
    local checkpoint="$RESULTDIR/$result_group/$scene/ckpts/ckpt_${CHECKPOINT_ITERATION}_rank${CHECKPOINT_RANK}.pt"
    local data_dir="$DATADIR/$data_family/$scene"

    if [[ ! -f "$checkpoint" ]]; then
        echo "[$suite/$scene] missing checkpoint: $checkpoint" >&2
        lkg_66_views_append_status "missing" "$suite" "$result_group" "$scene" "$camera_split" "" "missing checkpoint: $checkpoint"
        missing=$((missing + 1))
        if [[ "$REQUIRE_ALL" == "1" ]]; then
            failures=$((failures + 1))
        fi
        return 0
    fi
    if [[ ! -d "$data_dir" ]]; then
        echo "[$suite/$scene] missing data dir: $data_dir" >&2
        lkg_66_views_append_status "missing" "$suite" "$result_group" "$scene" "$camera_split" "" "missing data dir: $data_dir"
        missing=$((missing + 1))
        if [[ "$REQUIRE_ALL" == "1" ]]; then
            failures=$((failures + 1))
        fi
        return 0
    fi

    local camera_indices=()
    if [[ -n "$CAMERA_INDEXES" ]]; then
        read -r -a camera_indices <<< "$CAMERA_INDEXES"
    elif [[ -n "$CAMERA_INDEX" ]]; then
        camera_indices=("$CAMERA_INDEX")
    else
        local cfg_path="$RESULTDIR/$result_group/$scene/cfg.yml"
        local camera_count
        if ! camera_count="$(PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m lkg_experiment.dataset_views \
            --suite "$suite" \
            --data-dir "$data_dir" \
            --split "$camera_split" \
            --cfg-path "$cfg_path")"; then
            echo "[$suite/$scene] failed to count cameras" >&2
            lkg_66_views_append_status "failed" "$suite" "$result_group" "$scene" "$camera_split" "" "failed to count cameras"
            failures=$((failures + 1))
            if [[ "$STOP_ON_FAILURE" == "1" ]]; then
                return 1
            fi
            return 0
        fi
        if [[ "$camera_count" -le 0 ]]; then
            echo "[$suite/$scene] no cameras in split $camera_split" >&2
            lkg_66_views_append_status "failed" "$suite" "$result_group" "$scene" "$camera_split" "" "no cameras in split $camera_split"
            failures=$((failures + 1))
            if [[ "$STOP_ON_FAILURE" == "1" ]]; then
                return 1
            fi
            return 0
        fi
        local camera_index
        for ((camera_index = 0; camera_index < camera_count; camera_index++)); do
            camera_indices+=("$camera_index")
        done
    fi

    local camera_index
    for camera_index in "${camera_indices[@]}"; do
        local run_id
        run_id="$(lkg_66_views_sanitize_run_id "${RUN_GROUP}_${result_group}_${scene}")"
        local run_dir="$ARTIFACT_DIR/$run_id"
        local output_prefix
        printf -v output_prefix "%s_%06d" "$camera_split" "$camera_index"
        planned=$((planned + 1))
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[$suite/$scene/$camera_split:$camera_index] planned: $checkpoint -> $run_dir ($output_prefix)" >&2
            lkg_66_views_append_status "planned" "$suite" "$result_group" "$scene" "$camera_split" "$camera_index" "dry run: $run_dir/$output_prefix"
            continue
        fi

        echo "[$suite/$scene/$camera_split:$camera_index] running $result_group" >&2
        if PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m lkg_experiment.run_coherent_raster_experiment \
            --checkpoint-path "$checkpoint" \
            --data-dir "$data_dir" \
            --camera-source dataset \
            --camera-split "$camera_split" \
            --camera-index "$camera_index" \
            --width "$WIDTH" \
            --height "$HEIGHT" \
            --views "$VIEWS" \
            --map-mode file \
            --viewpoint-index-path "$VIEWPOINT_INDEX_PATH" \
            --artifact-dir "$ARTIFACT_DIR" \
            --run-id "$run_id" \
            --output-prefix "$output_prefix" \
            --append-metrics \
            "$@"; then
            if lkg_66_views_append_metrics "$suite" "$result_group" "$scene" "$camera_split" "$camera_index" "$run_dir"; then
                completed=$((completed + 1))
            else
                echo "[$suite/$scene/$camera_split:$camera_index] metrics aggregation failed for $run_dir" >&2
                lkg_66_views_append_status "failed" "$suite" "$result_group" "$scene" "$camera_split" "$camera_index" "metrics aggregation failed: $run_dir"
                failures=$((failures + 1))
            fi
        else
            local rc=$?
            echo "[$suite/$scene/$camera_split:$camera_index] failed with exit code $rc" >&2
            lkg_66_views_append_status "failed" "$suite" "$result_group" "$scene" "$camera_split" "$camera_index" "experiment exit code $rc"
            failures=$((failures + 1))
            if [[ "$STOP_ON_FAILURE" == "1" ]]; then
                return "$rc"
            fi
        fi
    done
}

lkg_66_views_run_suite() {
    local suite="$1"
    local result_groups="$2"
    local scenes="$3"
    local data_family="$4"
    local camera_split="$5"
    shift 5

    local groups_array=()
    local scenes_array=()
    read -r -a groups_array <<< "$result_groups"
    read -r -a scenes_array <<< "$scenes"

    local result_group
    local scene
    for result_group in "${groups_array[@]}"; do
        for scene in "${scenes_array[@]}"; do
            lkg_66_views_run_experiment_unit "$suite" "$result_group" "$scene" "$data_family" "$camera_split" "$@"
        done
    done
}

lkg_66_views_run_blender() {
    local result_groups="${BLENDER_RESULT_GROUPS:-blender_MCMC100000_init50000}"
    local scenes="${BLENDER_SCENES:-chair drums ficus hotdog lego materials mic ship}"
    local camera_split="${BLENDER_CAMERA_SPLIT:-test}"
    lkg_66_views_run_suite "blender" "$result_groups" "$scenes" "nerf_synthetic" "$camera_split" "$@"
}

lkg_66_views_run_mipnerf360() {
    local result_groups="${MIPNERF360_RESULT_GROUPS:-MipNeRF360_MCMC500000}"
    local scenes="${MIPNERF360_SCENES:-bicycle bonsai counter flowers garden kitchen room stump treehill}"
    local camera_split="${MIPNERF360_CAMERA_SPLIT:-val}"
    lkg_66_views_run_suite "mipnerf360" "$result_groups" "$scenes" "MipNeRF_360" "$camera_split" "$@"
}

lkg_66_views_finish() {
    echo "Summary TSV: $SUMMARY_TXT" >&2
    echo "Completed: $completed, planned: $planned, missing: $missing, failures: $failures" >&2

    if [[ "$failures" -ne 0 ]]; then
        exit 1
    fi
}
