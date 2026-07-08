#!/usr/bin/env bash
SMQ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "${SMQ_ROOT}/scripts/start_gripper.sh" "$@"
