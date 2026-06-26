#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WRITE_INTERLACED="${WRITE_INTERLACED:-0}"
export WRITE_PER_VIEW="${WRITE_PER_VIEW:-1}"
export RUN_GROUP="${RUN_GROUP:-rtgs_66_views_$(date +%Y%m%d_%H%M%S)}"

exec "$SCRIPT_DIR/run_rtgs_66_lkg_all.sh" "$@"
