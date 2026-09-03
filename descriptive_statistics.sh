#!/usr/bin/env bash
# Launcher for the descriptive-statistics companion recipe.
#
# The recipe profiles every registered source dataset and writes a self-verifying
# bundle under WF/02_Quality_Control/. It never touches the promoted N-response
# release package. See descriptive_statisticsCONFIG.toml for the goal and settings.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$SCRIPT_DIR"
ENTRYPOINT="$PROJECT_ROOT/modules/descriptive_statistics_pipeline.py"
CONDA_ENV_NAME="${N_RESPONSE_CONDA_ENV:-n_response}"

die() {
  printf 'descriptive_statistics.sh: %s\n' "$1" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: descriptive_statistics.sh [--config PATH]
  --config PATH  Configuration file (default: descriptive_statisticsCONFIG.toml)
EOF
}

if (($# > 0)); then
  if [[ $# -ne 2 || "$1" != --config || -z "$2" ]]; then
    usage
    exit 2
  fi
  CONFIG_PATH="$2"
else
  CONFIG_PATH="$PROJECT_ROOT/descriptive_statisticsCONFIG.toml"
fi

[[ -f "$ENTRYPOINT" ]] || die "entry point is missing: $ENTRYPOINT"
[[ -f "$CONFIG_PATH" ]] || die "configuration file does not exist: $CONFIG_PATH"

# This environment carries `conda env config vars` — LANG/LC_ALL, PYTHONNOUSERSITE,
# and the single-threaded BLAS pins. Those are applied only by a real activation.
# Putting the environment's bin/ on PATH by hand would load the right packages
# under the operator's ambient locale and thread counts, which is a byte-level
# difference in the emitted tables, not a cosmetic one. So: inherit a live
# activation when there is one, otherwise hand off to `conda run`, which does
# apply them. The bare-interpreter path is a last resort and says so.
ACTIVATION="none"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || [[ -x "$PYTHON_BIN" ]] ||
    die "PYTHON_BIN is set but not executable: $PYTHON_BIN"
  ACTIVATION="operator_python_bin"
elif [[ -n "${CONDA_PREFIX:-}" && "${CONDA_PREFIX##*/}" == "$CONDA_ENV_NAME" &&
        -x "${CONDA_PREFIX%/}/bin/python" ]]; then
  # Already inside the target environment; its variables are in this process.
  PYTHON_BIN="${CONDA_PREFIX%/}/bin/python"
  ACTIVATION="inherited"
fi

export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/n_response_curve_mpl}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  CONDA_EXECUTABLE="${CONDA_EXE:-}"
  if [[ "$CONDA_EXECUTABLE" != /* ]]; then
    CONDA_EXECUTABLE="$(command -v conda 2>/dev/null || true)"
  fi
  if [[ "$CONDA_EXECUTABLE" == /* && -x "$CONDA_EXECUTABLE" ]]; then
    printf 'launcher_handoff conda_activation=conda_run env=%s config=%s\n' \
      "$CONDA_ENV_NAME" "$CONFIG_PATH" >&2
    exec "$CONDA_EXECUTABLE" run --no-capture-output -n "$CONDA_ENV_NAME" \
      python "$ENTRYPOINT" --config "$CONFIG_PATH"
  fi
  for prefix in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
                "$HOME/mambaforge" /opt/conda /opt/miniconda3 \
                /usr/local/miniconda3 /usr/local/anaconda3; do
    candidate="${prefix%/}/envs/$CONDA_ENV_NAME/bin/python"
    if [[ -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      ACTIVATION="prefix_fallback_vars_not_applied"
      printf 'descriptive_statistics.sh: no Conda executable found; using %s directly. The environment variable pins (locale, BLAS thread counts) are NOT applied, so emitted tables may differ byte-for-byte from an activated run.\n' \
        "$candidate" >&2
      break
    fi
  done
fi

[[ -n "${PYTHON_BIN:-}" ]] ||
  die "no usable interpreter; activate the $CONDA_ENV_NAME environment or set PYTHON_BIN"

printf 'launcher_handoff conda_activation=%s config=%s\n' "$ACTIVATION" "$CONFIG_PATH" >&2
exec "$PYTHON_BIN" "$ENTRYPOINT" --config "$CONFIG_PATH"
