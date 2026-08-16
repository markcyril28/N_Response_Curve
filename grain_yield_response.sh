#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$SCRIPT_DIR"
ENTRYPOINT="$PROJECT_ROOT/modules/grain_yield_response_pipeline.py"

usage() {
  cat <<'EOF'
Usage: grain_yield_response.sh [--config PATH]
  --config PATH  Optional path to configuration file (default: grain_yield_responseCONFIG.toml)
EOF
}

if (( $# > 0 )); then
  if [[ $# -ne 2 || "$1" != --config || -z "$2" ]]; then
    usage
    exit 2
  fi
  CONFIG_PATH="$2"
else
  CONFIG_PATH="$PROJECT_ROOT/grain_yield_responseCONFIG.toml"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && ! [[ -x "$PYTHON_BIN" ]]; then
    printf 'grain_yield_response.sh: PYTHON_BIN is set but not executable: %s\n' "$PYTHON_BIN" >&2
    exit 2
  fi
  PYTHON_BIN_RESOLVED="$PYTHON_BIN"
else
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX%/}/bin/python" ]]; then
    PYTHON_BIN_RESOLVED="${CONDA_PREFIX%/}/bin/python"
  else
    PYTHON_BIN_RESOLVED=""
    COMMON_PREFIXES=(
      "$HOME/miniconda3"
      "$HOME/anaconda3"
      "$HOME/miniforge3"
      "$HOME/mambaforge"
      "/opt/conda"
      "/opt/miniconda3"
      "/usr/local/miniconda3"
      "/usr/local/anaconda3"
    )
    for prefix in "${COMMON_PREFIXES[@]}"; do
      candidate="${prefix%/}/envs/n_response/bin/python"
      if [[ -x "$candidate" ]]; then
        PYTHON_BIN_RESOLVED="$candidate"
        break
      fi
    done
    if [[ -z "${PYTHON_BIN_RESOLVED:-}" ]] && command -v python3 >/dev/null 2>&1; then
      PYTHON_BIN_RESOLVED="python3"
    fi
  fi

  if [[ -z "${PYTHON_BIN_RESOLVED:-}" ]]; then
    printf 'grain_yield_response.sh: no executable Python found; set PYTHON_BIN to a valid interpreter\n' >&2
    exit 2
  fi
fi

export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/n_response_curve_mpl}"

exec "$PYTHON_BIN_RESOLVED" "$ENTRYPOINT" --config "$CONFIG_PATH"

