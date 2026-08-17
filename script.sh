#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CALLER_CWD="$PWD"
CONFIG_FILE="$PROJECT_ROOT/scriptCONFIG.toml"
MODULE_SCRIPT="$PROJECT_ROOT/modules/n_response_curve_pipeline.py"
SOURCE_DATASET_OVERLAY_SCRIPT="$PROJECT_ROOT/modules/n_response_curve/reporting/generate_source_dataset_overlays.py"
LOG_HELPER="$PROJECT_ROOT/modules/n_response_curve/logging/run_logging.sh"
LOG_ROOT="${N_RESPONSE_LOG_ROOT:-$PROJECT_ROOT/logs}"
# Recorded before defaulting: an operator-supplied interpreter is honoured as
# given, while the default is repointed at the activated environment below.
PYTHON_BIN_FROM_OPERATOR="${PYTHON_BIN:+yes}"
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

# The pipeline is only defined inside the `n_response` Conda environment. The
# engine preflight in modules/n_response_curve/data/config.py resolves the R
# interpreter relative to CONDA_PREFIX, and the environment carries stored
# `conda env config vars` — locale and BLAS thread pins, R user-file
# suppression — that a bare PATH lookup never applies. A base-environment
# interpreter is usually new enough to satisfy the tomllib probe below, so a
# launch from the wrong environment would otherwise fail late, mid-run, with
# the wrong package versions instead of failing here. Activate in place when
# the operator forgot; stay a no-op when they did not.
CONDA_ENV_NAME="${N_RESPONSE_CONDA_ENV:-n_response}"
# An explicit prefix wins over the name, so that both the activation call and
# the already-active check below agree on which environment was asked for.
CONDA_ENV_TARGET="${N_RESPONSE_CONDA_PREFIX:-$CONDA_ENV_NAME}"
if [[ -n "${N_RESPONSE_CONDA_PREFIX:-}" && ! -x "${N_RESPONSE_CONDA_PREFIX%/}/bin/python" ]]; then
  preflight_die "N_RESPONSE_CONDA_PREFIX is not a usable environment prefix: $N_RESPONSE_CONDA_PREFIX"
fi
CONDA_ACTIVATION="inherited"
CONDA_ROOTS=()
# Replays the environment's stored `conda env config vars` in the fallback path
# below, where no Conda executable is available to apply them.
CONDA_STATE_READER='
import json, pathlib, shlex, sys
try:
    state = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)
for name, value in (state.get("env_vars") or {}).items():
    if name and not name[0].isdigit() and name.replace("_", "").isalnum():
        print("export " + name + "=" + shlex.quote(str(value)))
'

# Deliberately reads CONDA_PREFIX rather than CONDA_DEFAULT_ENV: a stale name
# left in a half-initialized shell would otherwise pass for an active
# environment, which is the failure this whole section exists to prevent.
conda_env_is_active() {
  [[ -n "${CONDA_PREFIX:-}" ]] || return 1
  if [[ -n "${N_RESPONSE_CONDA_PREFIX:-}" ]]; then
    [[ "${CONDA_PREFIX%/}" == "${N_RESPONSE_CONDA_PREFIX%/}" ]]
    return
  fi
  [[ "${CONDA_PREFIX##*/}" == "$CONDA_ENV_NAME" ]]
}

# Installation roots that may hold an activation hook, most specific first. The
# inherited CONDA_EXE comes first because it is set precisely in the case this
# is meant to fix: Conda initialized in the operator's shell, wrong environment.
collect_conda_roots() {
  local candidate known discovered duplicate
  local -a raw=()
  if [[ "${CONDA_EXE:-}" == /* ]]; then
    raw+=("${CONDA_EXE%/bin/*}")
  fi
  discovered="$(command -v conda 2>/dev/null || true)"
  if [[ "$discovered" == /* ]]; then
    raw+=("${discovered%/bin/*}")
  fi
  if [[ "${CONDA_PREFIX:-}" == /* ]]; then
    raw+=("${CONDA_PREFIX%/envs/*}" "$CONDA_PREFIX")
  fi
  raw+=(
    "${CONDA_ROOT:-}"
    "${MAMBA_ROOT_PREFIX:-}"
    "$HOME/miniconda3"
    "$HOME/anaconda3"
    "$HOME/miniforge3"
    "$HOME/mambaforge"
    "/opt/conda"
    "/opt/miniconda3"
    "/opt/anaconda3"
    "/usr/local/miniconda3"
    "/usr/local/anaconda3"
  )
  CONDA_ROOTS=()
  for candidate in "${raw[@]}"; do
    candidate="${candidate%/}"
    if [[ -z "$candidate" || ! -d "$candidate" ]]; then
      continue
    fi
    duplicate=0
    for known in ${CONDA_ROOTS[@]+"${CONDA_ROOTS[@]}"}; do
      if [[ "$known" == "$candidate" ]]; then
        duplicate=1
        break
      fi
    done
    if ((duplicate)); then
      continue
    fi
    CONDA_ROOTS+=("$candidate")
  done
}

# Prefix of the target environment, for the no-Conda-executable fallback only.
conda_env_prefix() {
  local line root
  if [[ -n "${N_RESPONSE_CONDA_PREFIX:-}" ]]; then
    printf '%s\n' "${N_RESPONSE_CONDA_PREFIX%/}"
    return 0
  fi
  # The registry lists full prefixes, so it also finds environments created
  # outside <root>/envs.
  if [[ -r "$HOME/.conda/environments.txt" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
      line="${line%/}"
      if [[ -n "$line" && "${line##*/}" == "$CONDA_ENV_NAME" && -x "$line/bin/python" ]]; then
        printf '%s\n' "$line"
        return 0
      fi
    done <"$HOME/.conda/environments.txt"
  fi
  for root in ${CONDA_ROOTS[@]+"${CONDA_ROOTS[@]}"}; do
    if [[ -x "$root/envs/$CONDA_ENV_NAME/bin/python" ]]; then
      printf '%s\n' "$root/envs/$CONDA_ENV_NAME"
      return 0
    fi
  done
  return 1
}

# Last resort: reproduce what `conda activate` does without a Conda executable.
# A bare PATH prepend is not equivalent — the stored environment variables and
# the activate.d hooks carry this environment's determinism pins — so replay
# both rather than leaving the two activation paths silently different.
activate_conda_prefix() {
  local prefix="$1" hook exports
  PATH="$prefix/bin:$PATH"
  export PATH
  export CONDA_PREFIX="$prefix"
  export CONDA_DEFAULT_ENV="$CONDA_ENV_NAME"
  if [[ -r "$prefix/conda-meta/state" && -x "$prefix/bin/python" ]]; then
    exports="$("$prefix/bin/python" -c "$CONDA_STATE_READER" "$prefix/conda-meta/state" 2>/dev/null || true)"
    if [[ -n "$exports" ]]; then
      eval "$exports"
    fi
  fi
  for hook in "$prefix"/etc/conda/activate.d/*.sh; do
    if [[ -r "$hook" ]]; then
      # Activation hooks are not written for `set -eu`, and they report on
      # stdout; keep the launcher's own stdout clean for its callers.
      set +eu
      # shellcheck source=/dev/null
      . "$hook" >&2
      set -eu
    fi
  done
}

activate_conda_env() {
  local root hook conda_exe hook_script prefix
  collect_conda_roots
  for root in ${CONDA_ROOTS[@]+"${CONDA_ROOTS[@]}"}; do
    hook="$root/etc/profile.d/conda.sh"
    if [[ ! -r "$hook" ]]; then
      continue
    fi
    set +eu
    # shellcheck source=/dev/null
    . "$hook" >&2
    conda activate "$CONDA_ENV_TARGET" >&2
    set -eu
    if conda_env_is_active; then
      CONDA_ACTIVATION="conda_activate"
      return 0
    fi
  done
  # No usable hook file, but a Conda executable can still emit one.
  conda_exe="${CONDA_EXE:-}"
  if [[ "$conda_exe" != /* ]]; then
    conda_exe="$(command -v conda 2>/dev/null || true)"
  fi
  if [[ "$conda_exe" == /* && -x "$conda_exe" ]]; then
    hook_script="$("$conda_exe" shell.bash hook 2>/dev/null || true)"
    if [[ -n "$hook_script" ]]; then
      set +eu
      eval "$hook_script" >&2
      conda activate "$CONDA_ENV_TARGET" >&2
      set -eu
      if conda_env_is_active; then
        CONDA_ACTIVATION="conda_shell_hook"
        return 0
      fi
    fi
  fi
  if prefix="$(conda_env_prefix)" && [[ -n "$prefix" ]]; then
    activate_conda_prefix "$prefix"
    if conda_env_is_active; then
      CONDA_ACTIVATION="prefix_fallback"
      printf 'script.sh: no Conda executable was usable; activated %s directly\n' "$prefix" >&2
      return 0
    fi
  fi
  return 1
}

if ! conda_env_is_active; then
  if ! activate_conda_env; then
    printf 'script.sh: Conda environment %s is not active and could not be activated automatically.\n' \
      "$CONDA_ENV_TARGET" >&2
    preflight_die "activate it with: conda activate $CONDA_ENV_NAME (or set N_RESPONSE_CONDA_PREFIX to the environment prefix)"
  fi
fi
# Prefer the environment's own interpreter over a PATH lookup, so an unusual
# PATH cannot reintroduce the interpreter this activation exists to avoid.
if [[ -z "$PYTHON_BIN_FROM_OPERATOR" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
fi

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
[[ -f "$SOURCE_DATASET_OVERLAY_SCRIPT" ]] || preflight_die "missing source dataset overlays entry point: $SOURCE_DATASET_OVERLAY_SCRIPT"

if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || preflight_die "PYTHON_BIN is not executable: $PYTHON_BIN"
else
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || preflight_die "Python interpreter not found: $PYTHON_BIN"
fi

LAUNCHER_RUN_ID="n_response_launcher_$(date -u +%Y%m%dT%H%M%SZ)_$$"
export N_RESPONSE_LAUNCHER_RUN_ID="$LAUNCHER_RUN_ID"
# Matplotlib otherwise falls back noisily when the user's Linux config directory
# is unavailable (for example, in a restricted WSL/session environment).
MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/n-response-matplotlib-${UID:-$$}}"
mkdir -p -- "$MPLCONFIGDIR" || preflight_die "cannot create Matplotlib cache: $MPLCONFIGDIR"
export MPLCONFIGDIR
"$PYTHON_BIN" -c 'import tomllib' >/dev/null 2>&1 || preflight_die "Python must provide tomllib"

export PYTHONDONTWRITEBYTECODE=1
cd -- "$PROJECT_ROOT"
"$PYTHON_BIN" "$MODULE_SCRIPT" --config "$CONFIG_FILE" --governance-preflight
# Any configured log clearing happens here, before the log files are opened.
"$PYTHON_BIN" "$MODULE_SCRIPT" --config "$CONFIG_FILE" --prepare-run-workspace --log-root "$LOG_ROOT"

if ! nrc_setup_logging "$PROJECT_ROOT" "pipeline_launcher" "$LAUNCHER_RUN_ID" "INFO" "$LOG_ROOT"; then
  printf 'script.sh: failed to initialize logging under: %s\n' "$LOG_ROOT" >&2
  exit 2
fi
trap 'nrc_teardown_logging || true' EXIT

nrc_log INFO "launcher_started" \
  "config_path=$CONFIG_FILE" \
  "module_path=$MODULE_SCRIPT" \
  "full_log_path=$NRC_FULL_LOG_FILE" \
  "event_log_path=$NRC_LOG_FILE" \
  "error_log_path=$NRC_ERROR_WARN_FILE"

nrc_log INFO "launcher_handoff" \
  "python_bin=$PYTHON_BIN" \
  "conda_env=$CONDA_ENV_NAME" \
  "conda_prefix=${CONDA_PREFIX:-}" \
  "conda_activation=$CONDA_ACTIVATION"
set +e
"$PYTHON_BIN" "$MODULE_SCRIPT" --config "$CONFIG_FILE"
status=$?
failed_stage="python_pipeline"
if [[ "$status" -eq 0 ]]; then
  nrc_log INFO "source_dataset_overlays_handoff" \
    "layout=figures/overlay/<source_name>" \
    "generation_policy=configured_force_replace"
  "$PYTHON_BIN" "$SOURCE_DATASET_OVERLAY_SCRIPT" --config "$CONFIG_FILE"
  status=$?
  failed_stage="source_dataset_overlays"
  if [[ "$status" -eq 0 ]]; then
    nrc_log INFO "source_dataset_overlays_completed" \
      "layout=figures/overlay/<source_name>"
  fi
fi
set -e
if [[ "$status" -eq 0 ]]; then
  nrc_log INFO "launcher_completed" "status=$status" || true
else
  nrc_log ERROR "launcher_failed" \
    "status=$status" \
    "failed_stage=$failed_stage" \
    "config_path=$CONFIG_FILE" \
    "full_log_path=$NRC_FULL_LOG_FILE" \
    "event_log_path=$NRC_LOG_FILE" || true
fi
logging_status=0
if nrc_teardown_logging; then
  :
else
  logging_status=$?
  printf 'script.sh: logging output could not be completed under: %s\n' "$LOG_ROOT" >&2
fi
trap - EXIT
if [[ "$status" -eq 0 && "$logging_status" -ne 0 ]]; then
  status=2
fi
exit "$status"
