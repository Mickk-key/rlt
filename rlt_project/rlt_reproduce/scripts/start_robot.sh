#!/usr/bin/env bash
# 委托到 SMQ&JGY 统一入口（Franka Hand + deoxys）
SMQ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "${SMQ_ROOT}/scripts/start_robot.sh" "$@"
