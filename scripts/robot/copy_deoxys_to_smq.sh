#!/usr/bin/env bash
# One-time: copy deoxys into SMQ&JGY so we never edit wty/deoxys_control.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_env.sh
source "$SCRIPT_DIR/../_env.sh"

SRC="${DEOXYS_COPY_SRC:-/home/host5010/workspaces/wty/deoxys_control/deoxys}"
DEST="${SMQ_ROOT}/third_party/deoxys"

if [[ ! -d "$SRC/config" ]]; then
  echo "ERROR: source deoxys not found: $SRC" >&2
  exit 1
fi

mkdir -p "${SMQ_ROOT}/third_party"
echo "==> rsync deoxys (~9GB, one-time)"
echo "    from: $SRC"
echo "    to:   $DEST"
rsync -a --info=progress2 --exclude='.git' "$SRC/" "$DEST/"

echo "[OK] deoxys vendored at $DEST"
echo "     DEOXYS_ROOT will auto-prefer this path via scripts/_env.sh"
