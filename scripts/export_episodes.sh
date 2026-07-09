#!/usr/bin/env bash
# Split episodes by success/fail and pack for GPU host transfer.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

CONFIG="${1:-$RLT_COLLECT_CONFIG}"
SKIP_LONG="${SKIP_LONG_SEC:-0}"

activate_robot_env
cd "$RLT_ROOT"

EXPORT_DIR="${SMQ_DATA_DIR}/export/plug_insertion"
DATE_TAG="$(date +%Y%m%d)"
ARCHIVE="${SMQ_DATA_DIR}/plug_insertion_export_${DATE_TAG}.tar.gz"

echo "==> Export episodes by label"
python -m rlt.scripts.export_episodes_by_label \
  --config "$CONFIG" \
  --export-dir "$EXPORT_DIR" \
  ${SKIP_LONG:+--skip-long-sec "$SKIP_LONG"}

echo ""
echo "==> Creating archive"
tar -czf "$ARCHIVE" -C "$SMQ_DATA_DIR/export" plug_insertion
ls -lh "$ARCHIVE"
echo ""
echo "Archive: $ARCHIVE"
echo "Contents:"
tar -tzf "$ARCHIVE" | head -8
echo "  ..."
tar -tzf "$ARCHIVE" | wc -l | xargs -I{} echo "  {} entries total"
