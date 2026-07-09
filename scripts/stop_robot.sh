#!/usr/bin/env bash
# 停止后台机械臂/夹爪服务（若用 start_robot.sh 一键启动）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

stopped=0

for pidfile in "$SMQ_ARM_PID_FILE" "$SMQ_GRIPPER_PID_FILE"; do
  if [[ -f "$pidfile" ]]; then
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "==> Stopping launcher pid=${pid}"
      kill "$pid" 2>/dev/null || true
      stopped=1
    fi
    rm -f "$pidfile"
  fi
done

if pgrep -f "bin/franka-interface" >/dev/null 2>&1; then
  echo "==> Stopping franka-interface ..."
  pkill -f "bin/franka-interface" || true
  stopped=1
fi

if pgrep -f "bin/gripper-interface" >/dev/null 2>&1; then
  echo "==> Stopping gripper-interface ..."
  pkill -f "bin/gripper-interface" || true
  stopped=1
fi

if [[ "$stopped" -eq 1 ]]; then
  echo "机械臂/夹爪服务已停止。"
else
  echo "未发现运行中的 deoxys 服务（若在前台终端运行，请直接在对应终端 Ctrl+C）。"
fi
