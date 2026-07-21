#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG_FILE="$PROJECT_ROOT/scriptCONFIG.toml"
MODULE_SCRIPT="$PROJECT_ROOT/modules/n_response_curve_pipeline.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

die() {
  printf 'script.sh: %s\n' "$*" >&2
  exit 2
}

[[ -f "$CONFIG_FILE" ]] || die "missing configuration: $CONFIG_FILE"
[[ -f "$MODULE_SCRIPT" ]] || die "missing pipeline entry point: $MODULE_SCRIPT"

if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
else
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python interpreter not found: $PYTHON_BIN"
fi

"$PYTHON_BIN" -c 'import tomllib' >/dev/null 2>&1 || die "Python must provide tomllib"

export PYTHONDONTWRITEBYTECODE=1

cd -- "$PROJECT_ROOT"
exec "$PYTHON_BIN" "$MODULE_SCRIPT" --config "$CONFIG_FILE" "$@"
