#!/usr/bin/env bash
# Pack smq&jgy for GPU host (10.176.53.120) — run on robot PC.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "$SCRIPT_DIR/_env.sh"

DEST="${SMQ_ROOT}/data/export_for_gpu_host"
ARCHIVE="${DEST}/smq_jgy_rl_server_$(date +%Y%m%d).tar.gz"

mkdir -p "$DEST"

echo "==> Packing smq&jgy (rl_server + actor stack, no deoxys 9GB)"
tar -czf "$ARCHIVE" \
  --exclude='third_party/deoxys' \
  --exclude='.git' \
  --exclude='logs' \
  --exclude='__pycache__' \
  --exclude='*.npz' \
  --exclude='data/episodes' \
  -C "$(dirname "$SMQ_ROOT")" \
  "$(basename "$SMQ_ROOT")/rlt_project" \
  "$(basename "$SMQ_ROOT")/scripts" \
  "$(basename "$SMQ_ROOT")/ONLINE_RL_TASKS.md" \
  "$(basename "$SMQ_ROOT")/docs/GPU_SERVER_START.md" 2>/dev/null || \
tar -czf "$ARCHIVE" \
  --exclude='third_party/deoxys' \
  --exclude='.git' \
  --exclude='logs' \
  --exclude='__pycache__' \
  -C "$(dirname "$SMQ_ROOT")" \
  "$(basename "$SMQ_ROOT")"

ls -lh "$ARCHIVE"
echo ""
echo "Copy to GPU (10.176.53.120):"
echo "  scp \"$ARCHIVE\" user@10.176.53.120:~/"
echo "  # on GPU: tar -xzf $(basename "$ARCHIVE") && see docs/GPU_SERVER_START.md"
