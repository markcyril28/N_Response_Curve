#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CALLER_CWD="$PWD"
CONFIG_FILE="$PROJECT_ROOT/scriptCONFIG.toml"
MODULE_SCRIPT="$PROJECT_ROOT/modules/n_response_curve_pipeline.py"
LOG_HELPER="$PROJECT_ROOT/modules/n_response_curve/logging/run_logging.sh"
LOG_ROOT="${N_RESPONSE_LOG_ROOT:-$PROJECT_ROOT/logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

[[ -f "$LOG_HELPER" ]] || {
  printf 'script.sh: missing logging helper: %s\n' "$LOG_HELPER" >&2
  exit 2
}
# shellcheck source=modules/n_response_curve/logging/run_logging.sh
. "$LOG_HELPER"
preflight_die() {
  printf 'script.sh: %s\n' "$*" >&2
  exit 2
}

if (($#)); then
  if [[ "$#" -eq 2 && "$1" == "--config" && -n "$2" ]]; then
    CONFIG_FILE="$2"
  elif [[ "$#" -eq 1 && "$1" == --config=* && -n "${1#--config=}" ]]; then
    CONFIG_FILE="${1#--config=}"
  else
    preflight_die "supported arguments are: --config <path>"
  fi
  if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$CALLER_CWD/$CONFIG_FILE"
  fi
fi

[[ -f "$CONFIG_FILE" ]] || preflight_die "missing configuration: $CONFIG_FILE"
[[ -f "$MODULE_SCRIPT" ]] || preflight_die "missing pipeline entry point: $MODULE_SCRIPT"

if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || preflight_die "PYTHON_BIN is not executable: $PYTHON_BIN"
else
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || preflight_die "Python interpreter not found: $PYTHON_BIN"
fi

LAUNCHER_RUN_ID="n_response_launcher_$(date -u +%Y%m%dT%H%M%SZ)_$$"
export N_RESPONSE_LAUNCHER_RUN_ID="$LAUNCHER_RUN_ID"
"$PYTHON_BIN" -c 'import tomllib' >/dev/null 2>&1 || preflight_die "Python must provide tomllib"

export PYTHONDONTWRITEBYTECODE=1
cd -- "$PROJECT_ROOT"
"$PYTHON_BIN" "$MODULE_SCRIPT" --config "$CONFIG_FILE" --governance-preflight

if ! nrc_setup_logging "$PROJECT_ROOT" "pipeline_launcher" "$LAUNCHER_RUN_ID" "INFO" "$LOG_ROOT"; then
  printf 'script.sh: failed to initialize logging under: %s\n' "$LOG_ROOT" >&2
  exit 2
fi
trap 'nrc_teardown_logging' EXIT

nrc_log INFO "launcher_started" \
  "config_path=$CONFIG_FILE" \
  "module_path=$MODULE_SCRIPT" \
  "full_log_path=$NRC_FULL_LOG_FILE" \
  "event_log_path=$NRC_LOG_FILE" \
  "error_log_path=$NRC_ERROR_WARN_FILE"

nrc_log INFO "launcher_handoff" "python_bin=$PYTHON_BIN"
set +e
"$PYTHON_BIN" "$MODULE_SCRIPT" --config "$CONFIG_FILE"
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
  nrc_log INFO "launcher_completed" "status=$status" || true
else
  nrc_log ERROR "launcher_failed" \
    "status=$status" \
    "failed_stage=python_pipeline" \
    "config_path=$CONFIG_FILE" \
    "full_log_path=$NRC_FULL_LOG_FILE" \
    "event_log_path=$NRC_LOG_FILE" || true
fi
nrc_teardown_logging
trap - EXIT
exit "$status"
