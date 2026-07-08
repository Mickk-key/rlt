#!/usr/bin/env bash
# SMQ&JGY 真机采集公共环境（封装桌面 deoxys.txt 流程，不修改 deoxys 源码）
set -euo pipefail

SMQ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SMQ_ROOT
export RLT_ROOT="${RLT_ROOT:-${SMQ_ROOT}/rlt_project/rlt_reproduce}"

# Robot-side synced modules (actor_loop, gpu_client) override rlt_reproduce copies.
export SMQ_RLT_SRC="${SMQ_ROOT}/src"
export RLT_COLLECT_CONFIG="${RLT_COLLECT_CONFIG:-${SMQ_ROOT}/configs/plug_insertion.yaml}"
export PYTHONPATH="${SMQ_RLT_SRC}:${RLT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export DEOXYS_VENDORED="${SMQ_ROOT}/third_party/deoxys"
export DEOXYS_ROOT="${DEOXYS_ROOT:-${DEOXYS_VENDORED}}"
if [[ ! -f "${DEOXYS_ROOT}/config/charmander.yml" ]]; then
  export DEOXYS_ROOT="/home/host5010/workspaces/wty/deoxys_control/deoxys"
  echo "[env] WARN: third_party/deoxys missing — using wty fallback: ${DEOXYS_ROOT}" >&2
  echo "[env]       Run once: bash scripts/robot/copy_deoxys_to_smq.sh" >&2
fi
export DEOXYS_INTERFACE_CFG="${DEOXYS_INTERFACE_CFG:-config/charmander.yml}"
export CONDA_ENV="${CONDA_ENV:-franka_mani}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

export CONTROLLER_TYPE="${CONTROLLER_TYPE:-OSC_POSE}"
export SMQ_OSC_POSE_CFG="${SMQ_OSC_POSE_CFG:-${SMQ_ROOT}/configs/deoxys/osc-pose-controller.yml}"
export SMQ_OSC_POSITION_CFG="${SMQ_OSC_POSITION_CFG:-${SMQ_ROOT}/configs/deoxys/osc-position-controller.yml}"
export SMQ_OSC_CONTROLLER_CFG="${SMQ_OSC_CONTROLLER_CFG:-}"
export SPACEMOUSE_VENDOR_ID="${SPACEMOUSE_VENDOR_ID:-9583}"
export SPACEMOUSE_PRODUCT_ID="${SPACEMOUSE_PRODUCT_ID:-50746}"

# All runtime outputs stay under SMQ&JGY (never /tmp)
export SMQ_DATA_DIR="${SMQ_DATA_DIR:-${SMQ_ROOT}/data}"
export SMQ_LOG_DIR="${SMQ_LOG_DIR:-${SMQ_ROOT}/logs}"
export SMQ_CAMERA_TEST_DIR="${SMQ_CAMERA_TEST_DIR:-${SMQ_DATA_DIR}/camera_test}"
export SMQ_RUN_DIR="${SMQ_RUN_DIR:-${SMQ_LOG_DIR}/run}"

export SMQ_ARM_PID_FILE="${SMQ_ARM_PID_FILE:-${SMQ_RUN_DIR}/arm.pid}"
export SMQ_GRIPPER_PID_FILE="${SMQ_GRIPPER_PID_FILE:-${SMQ_RUN_DIR}/gripper.pid}"
export SMQ_ARM_LOG="${SMQ_ARM_LOG:-${SMQ_LOG_DIR}/deoxys_arm.log}"
export SMQ_GRIPPER_LOG="${SMQ_GRIPPER_LOG:-${SMQ_LOG_DIR}/deoxys_gripper.log}"

ensure_smq_dirs() {
  mkdir -p "$SMQ_DATA_DIR" "$SMQ_CAMERA_TEST_DIR" "$SMQ_LOG_DIR" "$SMQ_RUN_DIR"
}

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

arm_process_running() {
  pgrep -f "bin/franka-interface" >/dev/null 2>&1
}

gripper_process_running() {
  pgrep -f "bin/gripper-interface" >/dev/null 2>&1
}
