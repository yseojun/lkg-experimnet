#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_66_views_common.sh"

lkg_66_views_init "blender_66_views_$(date +%Y%m%d_%H%M%S)"
lkg_66_views_run_blender "$@"
lkg_66_views_finish
