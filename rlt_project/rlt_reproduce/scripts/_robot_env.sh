#!/usr/bin/env bash
# Shared paths for RLT real-robot data collection (FR3 + gripper).
set -euo pipefail

RLT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RLT_ROOT

export DEOXYS_ROOT="${DEOXYS_ROOT:-${SMQ_ROOT}/third_party/deoxys}"
if [[ ! -f "${DEOXYS_ROOT}/config/charmander.yml" ]]; then
  export DEOXYS_ROOT="/home/host5010/workspaces/wty/deoxys_control/deoxys"
fi
export DEOXYS_INTERFACE_CFG="${DEOXYS_INTERFACE_CFG:-config/charmander.yml}"
export CONDA_ENV="${CONDA_ENV:-franka_mani}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

# franka | robotiq
export GRIPPER_TYPE="${GRIPPER_TYPE:-franka}"
export ROBOTIQ_CONFIG="${ROBOTIQ_CONFIG:-${RLT_ROOT}/configs/robotiq/robotiq.yaml}"
export FRANKA_GRIPPER_CONFIG="${FRANKA_GRIPPER_CONFIG:-${RLT_ROOT}/configs/franka/franka_hand.yaml}"
export ROBOTIQ_PORT="${ROBOTIQ_PORT:-/dev/ttyUSB0}"

export RLT_COLLECT_CONFIG="${RLT_COLLECT_CONFIG:-${RLT_ROOT}/configs/plug_insertion.yaml}"
export PYTHONPATH="${RLT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

SMQ_ROOT="${SMQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export SMQ_ROOT
export SMQ_DATA_DIR="${SMQ_DATA_DIR:-${SMQ_ROOT}/data}"
export SMQ_LOG_DIR="${SMQ_LOG_DIR:-${SMQ_ROOT}/logs}"
export SMQ_RUN_DIR="${SMQ_RUN_DIR:-${SMQ_LOG_DIR}/run}"

export RLT_ARM_PID_FILE="${RLT_ARM_PID_FILE:-${SMQ_RUN_DIR}/arm.pid}"
export RLT_GRIPPER_PID_FILE="${RLT_GRIPPER_PID_FILE:-${SMQ_RUN_DIR}/gripper.pid}"
export RLT_ARM_LOG="${RLT_ARM_LOG:-${SMQ_LOG_DIR}/deoxys_arm.log}"
export RLT_GRIPPER_LOG="${RLT_GRIPPER_LOG:-${SMQ_LOG_DIR}/deoxys_gripper.log}"

export FRANKA_ROBOT_IP="${FRANKA_ROBOT_IP:-172.16.0.3}"

activate_robot_env() {
  local conda_sh
  for conda_sh in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$conda_sh" ]]; then
      # shellcheck source=/dev/null
      source "$conda_sh"
      conda activate "$CONDA_ENV"
      return
    fi
  done

  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    return
  fi

  echo "Could not find conda. Set CONDA_ENV=${CONDA_ENV} manually." >&2
  exit 1
}

cd_deoxys_root() {
  if [[ ! -d "$DEOXYS_ROOT" ]]; then
    echo "DEOXYS_ROOT does not exist: $DEOXYS_ROOT" >&2
    exit 1
  fi
  cd "$DEOXYS_ROOT"
}

resolve_robotiq_port() {
  if [[ -e "$ROBOTIQ_PORT" ]]; then
    echo "$ROBOTIQ_PORT"
    return
  fi
  local by_id="/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAANTKCW-if00-port0"
  if [[ -e "$by_id" ]]; then
    echo "$by_id"
    return
  fi
  echo "$ROBOTIQ_PORT"
}

arm_process_running() {
  pgrep -f "bin/franka-interface" >/dev/null 2>&1
}

gripper_process_running() {
  pgrep -f "bin/gripper-interface" >/dev/null 2>&1
}

check_franka_fci_ports() {
  local ip="${1:-$FRANKA_ROBOT_IP}"
  nc -z -w1 "$ip" 1337 >/dev/null 2>&1 && nc -z -w1 "$ip" 1338 >/dev/null 2>&1
}
