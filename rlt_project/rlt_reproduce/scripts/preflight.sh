#!/usr/bin/env bash
# Pre-flight checks before starting FR3 + Franka Hand for RLT collection.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_robot_env.sh
source "$SCRIPT_DIR/_robot_env.sh"

fail=0

warn() { echo "[WARN] $*"; }
ok() { echo "[ OK ] $*"; }
err() { echo "[FAIL] $*" >&2; fail=1; }

echo "==> RLT robot preflight (gripper=${GRIPPER_TYPE})"

if [[ -d "$DEOXYS_ROOT" ]]; then
  ok "DEOXYS_ROOT=$DEOXYS_ROOT"
else
  err "DEOXYS_ROOT missing: $DEOXYS_ROOT"
fi

if [[ -x "${DEOXYS_ROOT}/bin/franka-interface" ]]; then
  ok "franka-interface binary present"
else
  err "franka-interface not found under ${DEOXYS_ROOT}/bin/"
fi

if [[ -x "${DEOXYS_ROOT}/bin/gripper-interface" ]]; then
  ok "gripper-interface binary present"
else
  err "gripper-interface not found under ${DEOXYS_ROOT}/bin/"
fi

RTOS_MODE=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | head -1 || echo unknown)
if [[ "$RTOS_MODE" == "powersave" ]]; then
  warn "CPU governor is powersave — switch to performance mode for stable Franka control"
else
  ok "CPU governor: ${RTOS_MODE:-unknown}"
fi

if command -v conda >/dev/null 2>&1; then
  ok "conda available (target env: ${CONDA_ENV})"
else
  err "conda not found"
fi

if check_franka_fci_ports; then
  ok "Franka FCI ports open on ${FRANKA_ROBOT_IP} (1337 arm, 1338 gripper)"
else
  warn "Franka FCI ports not reachable on ${FRANKA_ROBOT_IP} — unlock brakes and Activate FCI in Desk"
fi

if [[ "$GRIPPER_TYPE" == "robotiq" ]]; then
  port="$(resolve_robotiq_port)"
  if [[ -e "$port" ]]; then
    ok "Robotiq serial port: $port"
  else
    warn "Robotiq serial port not found ($port)"
  fi
else
  if gripper_process_running; then
    ok "deoxys gripper-interface already running"
  else
    warn "deoxys gripper-interface not running (start with: bash scripts/franka/start_gripper.sh)"
  fi
fi

if arm_process_running; then
  ok "deoxys arm service already running"
else
  warn "deoxys arm service not running yet (start with: bash scripts/start_arm.sh)"
fi

activate_robot_env
if python -c "import rlt" 2>/dev/null; then
  ok "python can import rlt"
else
  warn "rlt package not installed in ${CONDA_ENV}; run: pip install -e ${RLT_ROOT}"
fi

if python -c "import deoxys" 2>/dev/null; then
  ok "python can import deoxys"
else
  warn "deoxys not importable in ${CONDA_ENV} — collection may fail"
fi

echo ""
if [[ "$fail" -ne 0 ]]; then
  echo "Preflight failed. Fix errors above before starting the robot."
  exit 1
fi
echo "Preflight passed (warnings are OK if hardware is not connected yet)."
