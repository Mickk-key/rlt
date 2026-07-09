#!/usr/bin/env bash
# Robot-side actor loop for split online RL (Franka PC).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"
if [[ -f "$SCRIPT_DIR/robot/deoxys_actor.env" ]]; then
  # shellcheck source=robot/deoxys_actor.env
  source "$SCRIPT_DIR/robot/deoxys_actor.env"
fi

CONFIG="${RLT_COLLECT_CONFIG}"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    *.yaml|*.yml)
      CONFIG="$1"
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

MOCK="${MOCK:-0}"
EPISODES="${EPISODES:-}"
MAX_STEPS="${MAX_STEPS:-}"
RESET_MODE="${RESET_MODE:-}"
GPU_SERVER_HOST="${GPU_SERVER_HOST:-}"
GPU_SERVER_MOCK="${GPU_SERVER_MOCK:-}"
CONFIRM="${CONFIRM:-0}"

# Pure local mock: no GPU unless caller sets GPU_SERVER_HOST (e.g. tunnel pathway test).
if [[ "$MOCK" == "1" && -z "$GPU_SERVER_HOST" && -z "$GPU_SERVER_MOCK" ]]; then
  GPU_SERVER_MOCK=1
fi

if [[ "$MOCK" != "1" && "$CONFIRM" != "1" ]]; then
  echo "ERROR: 实机 actor loop 需要显式确认。" >&2
  echo "  请先确认工作空间无障碍、急停可用，然后:" >&2
  echo "  CONFIRM=1 bash scripts/run_actor_loop.sh" >&2
  echo "" >&2
  echo "  Mock 冒烟（不连机械臂）:" >&2
  echo "  MOCK=1 bash scripts/run_actor_loop.sh" >&2
  exit 1
fi

if [[ "$MOCK" != "1" ]]; then
  if ! arm_process_running; then
    echo "ERROR: 机械臂服务未运行。请先:" >&2
    echo "  bash scripts/start_arm.sh" >&2
    exit 1
  fi
  if ! gripper_process_running; then
    echo "WARN: 夹爪服务未运行。建议: bash scripts/start_gripper.sh" >&2
  fi
  bash "$SCRIPT_DIR/free_cameras.sh"
  bash "$SCRIPT_DIR/free_deoxys_client.sh"
  bash "$SCRIPT_DIR/check_cameras_config.sh" "$CONFIG"
fi

activate_robot_env
cd "$RLT_ROOT"

ARGS=(--config "$CONFIG")
if [[ "$MOCK" == "1" ]]; then
  ARGS+=(--mock)
fi
if [[ -n "$EPISODES" ]]; then
  ARGS+=(--episodes "$EPISODES")
fi
if [[ -n "$MAX_STEPS" ]]; then
  ARGS+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "$GPU_SERVER_HOST" ]]; then
  ARGS+=(--gpu-host "$GPU_SERVER_HOST")
fi
if [[ -n "$RESET_MODE" ]]; then
  ARGS+=(--reset-mode "$RESET_MODE")
fi

echo "==> RLT Actor Loop（本机 rollout）"
echo "    config=${CONFIG}"
echo "    mock=${MOCK}"
echo "    GPU_SERVER_HOST=${GPU_SERVER_HOST:-<yaml>}"
echo "    episodes=${EPISODES:-<yaml max_episodes, default 1>}"
echo "    max_steps=${MAX_STEPS:-<yaml max_steps_per_episode, default 600>}"
echo ""
echo "【episode 进行中】 s=成功(reward=1)  f=失败  q=退出"
echo "    日志: logs/online_rl/"
echo ""

exec python -m rlt.scripts.actor_loop "${ARGS[@]}" "${EXTRA_ARGS[@]}"
