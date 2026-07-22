#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
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
LAUNCHER_RUN_ID="n_response_launcher_$(date -u +%Y%m%dT%H%M%SZ)_$$"
if ! nrc_setup_logging "$PROJECT_ROOT" "pipeline_launcher" "$LAUNCHER_RUN_ID" "INFO" "$LOG_ROOT"; then
  printf 'script.sh: failed to initialize logging under: %s\n' "$LOG_ROOT" >&2
  exit 2
fi
trap 'nrc_teardown_logging' EXIT
export N_RESPONSE_LAUNCHER_RUN_ID="$LAUNCHER_RUN_ID"

die() {
  nrc_log ERROR "launcher_failed" "error=$*"
  printf 'script.sh: %s\n' "$*" >&2
  exit 2
}

nrc_log INFO "launcher_started" \
  "config_path=$CONFIG_FILE" \
  "module_path=$MODULE_SCRIPT" \
  "full_log_path=$NRC_FULL_LOG_FILE" \
  "event_log_path=$NRC_LOG_FILE" \
  "error_log_path=$NRC_ERROR_WARN_FILE"
[[ -f "$CONFIG_FILE" ]] || die "missing configuration: $CONFIG_FILE"
[[ -f "$MODULE_SCRIPT" ]] || die "missing pipeline entry point: $MODULE_SCRIPT"

if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
else
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python interpreter not found: $PYTHON_BIN"
fi

"$PYTHON_BIN" -c 'import tomllib' >/dev/null 2>&1 || die "Python must provide tomllib"

export PYTHONDONTWRITEBYTECODE=1

nrc_log INFO "launcher_handoff" "python_bin=$PYTHON_BIN"
cd -- "$PROJECT_ROOT"
set +e
"$PYTHON_BIN" "$MODULE_SCRIPT" --config "$CONFIG_FILE" "$@"
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
  nrc_log INFO "launcher_completed" "status=$status" || true
else
  nrc_log ERROR "launcher_failed" "status=$status" || true
fi
nrc_teardown_logging
trap - EXIT
exit "$status"
