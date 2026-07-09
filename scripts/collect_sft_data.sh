#!/usr/bin/env bash
# SFT / imitation learning demo collection (JSONL + PNG, 50 Hz, synced cameras)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

if ! arm_process_running; then
  echo "ERROR: 机械臂服务未运行。请先:" >&2
  echo "  bash scripts/start_arm.sh" >&2
  exit 1
fi

CONFIG="${1:-${SFT_COLLECT_CONFIG:-${SMQ_ROOT}/configs/sft_plug_insertion.yaml}}"

bash "$SCRIPT_DIR/free_cameras.sh"
bash "$SCRIPT_DIR/free_deoxys_client.sh"
bash "$SCRIPT_DIR/check_cameras_config.sh" "$CONFIG"
bash "$SCRIPT_DIR/verify_camera_roles.sh" "$CONFIG"

activate_robot_env
cd "$RLT_ROOT"

MOCK="${MOCK:-0}"
ARGS=(--config "$CONFIG")
if [[ "$MOCK" == "1" ]]; then
  ARGS+=(--mock)
fi

echo "==> SFT 数据采集 (JSONL + images)"
echo "    config=${CONFIG}"
echo "    MOCK=${MOCK}  (MOCK=1 只 reset+遥操，不写盘)"
echo "    输出: data/sft/plug_insertion/ep_XXXXX/"
echo ""
if [[ "$MOCK" == "1" ]]; then
  echo "【MOCK】 r=reset+试遥操  q=退出"
else
  echo "【操作】 r=随机 reset + 开始录  s=成功保存  f=失败保存  q=退出"
fi
echo "SpaceMouse: 移动/旋转 | 左键/g 夹住 | o 松开 | 右键 joint home"
echo ""

exec python -m rlt.scripts.collect_sft_plug_insertion "${ARGS[@]}"
