#!/usr/bin/env bash
# 一键后台启动：机械臂 + Franka 原生夹爪（deoxys）
# 等价于桌面 deoxys.txt Terminal 1 + Terminal 2，合并到后台
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"

activate_robot_env
ensure_smq_dirs
cd_deoxys_root

INTERFACE_CFG="${1:-$DEOXYS_INTERFACE_CFG}"

if arm_process_running; then
  echo "==> 机械臂服务已在运行"
else
  echo "==> 后台启动机械臂 ..."
  nohup ./auto_scripts/auto_arm.sh "$INTERFACE_CFG" >"$SMQ_ARM_LOG" 2>&1 &
  echo $! >"$SMQ_ARM_PID_FILE"
  echo "    pid=$(cat "$SMQ_ARM_PID_FILE")  log=${SMQ_ARM_LOG}"
fi

if gripper_process_running; then
  echo "==> 夹爪服务已在运行"
else
  echo "==> 后台启动夹爪 ..."
  nohup ./auto_scripts/auto_gripper.sh "$INTERFACE_CFG" >"$SMQ_GRIPPER_LOG" 2>&1 &
  echo $! >"$SMQ_GRIPPER_PID_FILE"
  echo "    pid=$(cat "$SMQ_GRIPPER_PID_FILE")  log=${SMQ_GRIPPER_LOG}"
fi

echo "==> 等待服务就绪 (12s) ..."
sleep 12

cat <<EOF

============================================================
 deoxys 机械臂 + Franka 夹爪 已启动（后台）
============================================================
 机械臂 : $(arm_process_running && echo RUNNING || echo NOT RUNNING)
 夹爪   : $(gripper_process_running && echo RUNNING || echo NOT RUNNING)
 arm log: ${SMQ_ARM_LOG}
 gripper log: ${SMQ_GRIPPER_LOG}

 下一步（新开终端）:
   cd "${SMQ_ROOT}"
   bash scripts/teleop_test.sh              # 遥操测试
   bash scripts/collect_sync_realsense.sh   # 同步采集
   bash scripts/collect_data.sh             # RLT 任务采集

 停止:
   bash scripts/stop_robot.sh
============================================================
EOF
