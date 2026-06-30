#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  apply_patches.sh [--check] /path/to/gsplat

Applies the RTGS CoherentRaster gsplat patches stored next to this script.

Options:
  --check   Check whether patches apply cleanly without modifying the target.
EOF
}

check_only=0
if [[ "${1:-}" == "--check" ]]; then
    check_only=1
    shift
fi

if [[ "$#" -ne 1 ]]; then
    usage >&2
    exit 2
fi

target="$1"
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
patches=("$script_dir"/patches/*.patch)

if [[ ! -d "$target/.git" ]]; then
    echo "Target is not a git checkout: $target" >&2
    exit 2
fi

if ! git -C "$target" diff --quiet || ! git -C "$target" diff --cached --quiet; then
    echo "Target has uncommitted changes. Commit/stash them before applying patches." >&2
    exit 2
fi

if [[ "$check_only" -eq 1 ]]; then
    tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/rtgs-cr-gsplat-check.XXXXXX")"
    trap 'rm -rf "$tmpdir"' EXIT
    git clone --quiet --shared "$target" "$tmpdir/gsplat-check"
    git -C "$tmpdir/gsplat-check" am --3way "${patches[@]}"
    echo "Patch check passed for $target"
else
    git -C "$target" am --3way "${patches[@]}"
    echo "Applied RTGS CoherentRaster patches to $target"
fi
