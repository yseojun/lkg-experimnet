#!/usr/bin/env bash

_lkg_checkpoint_rel="blender_MCMC100000_init50000/drums/ckpts/ckpt_29999_rank0.pt"

_lkg_find_data_root() {
    local base="$1"
    local path=""

    if [[ -d "$base/nerf_synthetic/drums" ]]; then
        printf '%s\n' "$base"
        return
    fi

    if [[ -d "$base" ]]; then
        path="$(find "$base" -path '*/nerf_synthetic/drums' -type d -print -quit 2>/dev/null || true)"
        if [[ -n "$path" ]]; then
            printf '%s\n' "${path%/nerf_synthetic/drums}"
            return
        fi
    fi

    printf '%s\n' "$base"
}

_lkg_find_result_root() {
    local base="$1"
    local path=""

    if [[ -f "$base/$_lkg_checkpoint_rel" ]]; then
        printf '%s\n' "$base"
        return
    fi

    if [[ -f "$base/coherent-raster/$_lkg_checkpoint_rel" ]]; then
        printf '%s\n' "$base/coherent-raster"
        return
    fi

    if [[ -d "$base" ]]; then
        path="$(find "$base" -path "*/$_lkg_checkpoint_rel" -type f -print -quit 2>/dev/null || true)"
        if [[ -n "$path" ]]; then
            printf '%s\n' "${path%/$_lkg_checkpoint_rel}"
            return
        fi
    fi

    printf '%s\n' "$base"
}

_lkg_is_under() {
    local path="$1"
    local root="$2"

    [[ "$path" == "$root" || "$path" == "$root"/* ]]
}

_lkg_print_exports() {
    printf 'export LKG_DATASET_BASE=%q\n' "$LKG_DATASET_BASE"
    printf 'export LKG_RESULT_BASE=%q\n' "$LKG_RESULT_BASE"
    printf 'export DATADIR=%q\n' "$DATADIR"
    printf 'export RESULTDIR=%q\n' "$RESULTDIR"
    printf 'export datadir=%q\n' "$datadir"
    printf 'export resultdir=%q\n' "$resultdir"
    printf 'export DATA_DIR_ROOT=%q\n' "$DATA_DIR_ROOT"
    printf 'export RESULT_DIR_ROOT=%q\n' "$RESULT_DIR_ROOT"
    printf 'export DATA_DIR=%q\n' "$DATA_DIR"
    printf 'export CHECKPOINT_PATH=%q\n' "$CHECKPOINT_PATH"
}

_lkg_sourced=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    _lkg_sourced=1
fi

export LKG_DATASET_BASE="${LKG_DATASET_BASE:-/data/ysj/dataset}"
export LKG_RESULT_BASE="${LKG_RESULT_BASE:-/data/ysj/result}"

_lkg_home_data_default="$HOME/Data/datasets"
_lkg_home_result_default="$HOME/Data/results"

_lkg_data_candidate="${DATADIR:-${datadir:-${DATA_DIR_ROOT:-$LKG_DATASET_BASE}}}"
_lkg_result_candidate="${RESULTDIR:-${resultdir:-${RESULT_DIR_ROOT:-$LKG_RESULT_BASE}}}"

if _lkg_is_under "$_lkg_data_candidate" "$_lkg_home_data_default"; then
    _lkg_data_candidate="$LKG_DATASET_BASE"
fi

if _lkg_is_under "$_lkg_result_candidate" "$_lkg_home_result_default"; then
    _lkg_result_candidate="$LKG_RESULT_BASE"
fi

export DATADIR="$(_lkg_find_data_root "$_lkg_data_candidate")"
export RESULTDIR="$(_lkg_find_result_root "$_lkg_result_candidate")"

export datadir="$DATADIR"
export resultdir="$RESULTDIR"
export DATA_DIR_ROOT="$DATADIR"
export RESULT_DIR_ROOT="$RESULTDIR"

_lkg_data_dir_candidate="${DATA_DIR:-$DATADIR/nerf_synthetic/drums}"
_lkg_checkpoint_candidate="${CHECKPOINT_PATH:-$RESULTDIR/$_lkg_checkpoint_rel}"

if _lkg_is_under "$_lkg_data_dir_candidate" "$_lkg_home_data_default"; then
    _lkg_data_dir_candidate="$DATADIR/nerf_synthetic/drums"
fi

if _lkg_is_under "$_lkg_checkpoint_candidate" "$_lkg_home_result_default"; then
    _lkg_checkpoint_candidate="$RESULTDIR/$_lkg_checkpoint_rel"
fi

export DATA_DIR="$_lkg_data_dir_candidate"
export CHECKPOINT_PATH="$_lkg_checkpoint_candidate"

if [[ "$_lkg_sourced" == "1" ]]; then
    echo "DATADIR=$DATADIR" >&2
    echo "RESULTDIR=$RESULTDIR" >&2
    echo "DATA_DIR=$DATA_DIR" >&2
    echo "CHECKPOINT_PATH=$CHECKPOINT_PATH" >&2
else
    _lkg_print_exports
fi

unset -f _lkg_find_data_root _lkg_find_result_root _lkg_is_under _lkg_print_exports
unset _lkg_checkpoint_rel _lkg_sourced
unset _lkg_data_candidate _lkg_result_candidate _lkg_data_dir_candidate _lkg_checkpoint_candidate
unset _lkg_home_data_default _lkg_home_result_default
