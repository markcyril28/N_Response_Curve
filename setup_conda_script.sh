#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Keep operator switches and paths together. The package/environment definition remains in YAML.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MANIFEST_FILE="${MANIFEST_FILE:-$PROJECT_ROOT/setup_conda_script.yml}"
METADATA_ROOT="${METADATA_ROOT:-$PROJECT_ROOT/WF/99_Run_Metadata/environment}"
MAMBA_BIN="${MAMBA_BIN:-}"
CONDA_BIN="${CONDA_BIN:-}"
DRY_RUN_ONLY=0
MANAGER=""
RUNNER=""
ENV_NAME=""
ENV_EXISTS=0
ENV_PREFIX=""
TARGET_MUTATED=0
BACKUP_PREFIX=""
CANDIDATE_PREFIX=""
RUN_STAMP=""
ROLLBACK_FAILED=0

die() {
  printf 'setup_conda_script.sh: %s\n' "$*" >&2
  exit 2
}

usage() {
  printf 'Usage: %s [--dry-run]\n' "${BASH_SOURCE[0]}"
}

for argument in "$@"; do
  case "$argument" in
    --dry-run)
      DRY_RUN_ONLY=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $argument"
      ;;
  esac
done

[[ -f "$MANIFEST_FILE" ]] || die "missing Conda manifest: $MANIFEST_FILE"

validate_linux_executable() {
  local candidate="$1"
  case "$candidate" in
    *.exe|*.EXE)
      die "Windows manager rejected: $candidate"
      ;;
  esac
  [[ -x "$candidate" ]] || die "manager is not executable: $candidate"
  if command -v file >/dev/null 2>&1 && file -b "$candidate" | grep -Eiq 'PE32|MS-DOS'; then
    die "Windows manager rejected: $candidate"
  fi
}

resolve_manager() {
  local candidate
  if [[ -n "$MAMBA_BIN" ]]; then
    candidate="$MAMBA_BIN"
    [[ "$candidate" == */* ]] || candidate="$(command -v "$candidate" || true)"
    [[ -n "$candidate" ]] || die "MAMBA_BIN was not found: $MAMBA_BIN"
    validate_linux_executable "$candidate"
    MANAGER="$candidate"
    return
  fi
  if command -v mamba >/dev/null 2>&1; then
    candidate="$(command -v mamba)"
    validate_linux_executable "$candidate"
    MANAGER="$candidate"
    return
  fi
  if [[ -n "$CONDA_BIN" ]]; then
    candidate="$CONDA_BIN"
    [[ "$candidate" == */* ]] || candidate="$(command -v "$candidate" || true)"
    [[ -n "$candidate" ]] || die "CONDA_BIN was not found: $CONDA_BIN"
    validate_linux_executable "$candidate"
    MANAGER="$candidate"
    return
  fi
  command -v conda >/dev/null 2>&1 || die "No Linux Mamba or Conda executable was found"
  candidate="$(command -v conda)"
  validate_linux_executable "$candidate"
  MANAGER="$candidate"
}

resolve_runner() {
  local candidate
  if [[ -n "$CONDA_BIN" ]]; then
    candidate="$CONDA_BIN"
    [[ "$candidate" == */* ]] || candidate="$(command -v "$candidate" || true)"
    [[ -n "$candidate" ]] || die "CONDA_BIN was not found: $CONDA_BIN"
  elif command -v conda >/dev/null 2>&1; then
    candidate="$(command -v conda)"
  else
    candidate="$MANAGER"
  fi
  validate_linux_executable "$candidate"
  RUNNER="$candidate"
}

read_environment_name() {
  local line name="" count=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^name:[[:space:]]*([A-Za-z0-9_.-]+)[[:space:]]*$ ]]; then
      name="${BASH_REMATCH[1]}"
      count=$((count + 1))
    fi
  done < "$MANIFEST_FILE"
  [[ "$count" -eq 1 ]] || die "manifest must contain exactly one top-level name"
  ENV_NAME="$name"
}

validate_manifest_shape() {
  grep -Eq '^channels:' "$MANIFEST_FILE" || die "manifest is missing channels"
  grep -Eq '^  - conda-forge$' "$MANIFEST_FILE" || die "manifest must use conda-forge"
  grep -Eq '^  - nodefaults$' "$MANIFEST_FILE" || die "manifest must use nodefaults"
  grep -Eq '^variables:' "$MANIFEST_FILE" || die "manifest is missing environment variables"
}

environment_exists() {
  local environments_json
  environments_json="$($MANAGER env list --json)" || die "could not inspect Conda environments"
  python -c 'import json, pathlib, sys; name = sys.argv[1]; data = json.load(sys.stdin); print("1" if any(pathlib.Path(item).name == name for item in data.get("envs", [])) else "0")' "$ENV_NAME" <<<"$environments_json"
}

resolve_target_prefix() {
  local info_json
  info_json="$($RUNNER info --json)" || die "could not inspect Conda environment directories"
  ENV_PREFIX="$(python -c '
import json, pathlib, sys

name = sys.argv[1]
data = json.load(sys.stdin)
directories = data.get("envs_dirs") or data.get("envs directories") or []
if not directories:
    root = data.get("root_prefix") or data.get("base environment")
    if root:
        directories = [str(pathlib.Path(root) / "envs")]
if not directories:
    raise SystemExit("Conda did not report an environment directory")
print(pathlib.Path(directories[0]).expanduser().resolve() / name)
' "$ENV_NAME" <<<"$info_json")" || die "could not resolve the target environment prefix"
  [[ "$ENV_PREFIX" == /* ]] || die "resolved environment prefix is not absolute: $ENV_PREFIX"
}

validate_absent_target_prefix() {
  [[ "$ENV_EXISTS" -eq 0 ]] || return 0
  [[ ! -e "$ENV_PREFIX" ]] || die "incomplete environment prefix exists but is not registered: $ENV_PREFIX"
}

manager_env_create() {
  "$MANAGER" env create --strict-channel-priority --file "$MANIFEST_FILE" "$@"
}

manager_env_remove() {
  "$MANAGER" env remove "$@"
}

# Invoked indirectly through run_stage/capture_evidence.
# shellcheck disable=SC2329
run_in_env() {
  # A partially initialized parent Conda shell can make `mamba run` activate
  # the wrong prefix. Remove inherited activation state before isolated checks.
  /usr/bin/env \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u CONDA_PROMPT_MODIFIER \
    -u CONDA_SHLVL \
    -u CONDA_EXE \
    -u CONDA_PYTHON_EXE \
    -u CONDA_ROOT \
    PYTHONNOUSERSITE=1 \
    R_ENVIRON_USER=/dev/null \
    R_PROFILE_USER=/dev/null \
    R_LIBS_USER=/dev/null \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    BLIS_NUM_THREADS=1 \
    "$RUNNER" run "$@"
}

# All checks use conda run (or the equivalent mamba run fallback); never activate a parent shell.
run_stage() {
  local stage="$1" log_path status
  shift
  mkdir -p "$METADATA_ROOT"
  log_path="$METADATA_ROOT/${stage}_${RUN_STAMP}.log"
  if "$@" >"$log_path" 2>&1; then
    printf 'Verification stage passed: %s (log: %s)\n' "$stage" "$log_path"
    return 0
  else
    status=$?
  fi
  printf 'Verification stage failed: %s (inspect: %s)\n' "$stage" "$log_path" >&2
  sed -n '1,160p' "$log_path" >&2
  return "$status"
}

capture_evidence() {
  local destination="$1" status
  shift
  if "$@" >"$destination" 2>&1; then
    return 0
  else
    status=$?
  fi
  printf 'Evidence capture failed (inspect: %s)\n' "$destination" >&2
  sed -n '1,160p' "$destination" >&2
  return "$status"
}

verify_runtime() {
  local prefix="$1" label="$2" status
  printf 'Verifying %s runtime through %s run...\n' "$label" "$(basename "$RUNNER")"
  if run_stage "${label}_python_imports" run_in_env --prefix "$prefix" python -c 'import numpy, pandas, scipy, statsmodels, sklearn, matplotlib, pyarrow, openpyxl; print("python-imports-ok")'; then
    :
  else
    status=$?
    return "$status"
  fi
  if run_stage "${label}_r_version" run_in_env --prefix "$prefix" Rscript --version; then
    :
  else
    status=$?
    return "$status"
  fi
  if run_stage "${label}_r_smoke" run_in_env --prefix "$prefix" Rscript --vanilla "$PROJECT_ROOT/tests/r_environment_smoke.R"; then
    :
  else
    status=$?
    return "$status"
  fi
  return 0
}

verify_named_runtime() {
  local status
  printf 'Verifying named runtime through %s run...\n' "$(basename "$RUNNER")"
  if run_stage "named_python_imports" run_in_env --name "$ENV_NAME" python -c 'import numpy, pandas, scipy, statsmodels, sklearn, matplotlib, pyarrow, openpyxl; print("python-imports-ok")'; then
    :
  else
    status=$?
    return "$status"
  fi
  if run_stage "named_r_version" run_in_env --name "$ENV_NAME" Rscript --version; then
    :
  else
    status=$?
    return "$status"
  fi
  if run_stage "named_r_smoke" run_in_env --name "$ENV_NAME" Rscript --vanilla "$PROJECT_ROOT/tests/r_environment_smoke.R"; then
    :
  else
    status=$?
    return "$status"
  fi
  return 0
}

capture_environment_evidence() {
  local destination status
  mkdir -p "$METADATA_ROOT"

  destination="$METADATA_ROOT/conda_explicit_spec.txt"
  if capture_evidence "$destination" "$MANAGER" list --name "$ENV_NAME" --explicit; then
    :
  else
    status=$?
    return "$status"
  fi

  destination="$METADATA_ROOT/conda_environment_export.yml"
  if capture_evidence "$destination" "$MANAGER" env export --name "$ENV_NAME" --no-builds; then
    :
  else
    status=$?
    return "$status"
  fi

  destination="$METADATA_ROOT/r_version.txt"
  if capture_evidence "$destination" run_in_env --name "$ENV_NAME" Rscript --version; then
    :
  else
    status=$?
    return "$status"
  fi

  destination="$METADATA_ROOT/r_session_info.txt"
  if capture_evidence "$destination" run_in_env --name "$ENV_NAME" Rscript --vanilla -e 'sessionInfo()'; then
    :
  else
    status=$?
    return "$status"
  fi

  destination="$METADATA_ROOT/r_direct_packages.csv"
  if capture_evidence "$destination" run_in_env --name "$ENV_NAME" Rscript --vanilla -e 'p <- c("data.table", "arrow", "jsonlite", "digest", "nlme", "lme4", "glmmTMB", "mgcv", "emmeans", "broom", "broom.mixed", "performance", "ggplot2", "testthat", "lintr"); d <- data.frame(package=p, version=vapply(p, function(x) as.character(packageVersion(x)), character(1))); write.csv(d, row.names=FALSE)'; then
    return 0
  else
    status=$?
  fi
  return "$status"
}

solver_and_candidate() {
  local stamp solver_log candidate_prefix status
  mkdir -p "$METADATA_ROOT"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  solver_log="$METADATA_ROOT/solver_${stamp}.log"
  candidate_prefix="$(mktemp -d "${TMPDIR:-/tmp}/n_response_candidate.XXXXXX")"
  rmdir "$candidate_prefix"
  printf 'Solving candidate environment from %s...\n' "$MANIFEST_FILE"
  if manager_env_create --dry-run --prefix "$candidate_prefix" --yes >"$solver_log" 2>&1; then
    :
  else
    status=$?
    cat "$solver_log" >&2
    printf 'Conda solver dry-run failed; no named environment was changed\n' >&2
    return "$status"
  fi
  if [[ "$DRY_RUN_ONLY" -eq 1 ]]; then
    CANDIDATE_PREFIX=""
    printf 'solver-dry-run-ok\n'
    return 0
  fi

  CANDIDATE_PREFIX="$candidate_prefix"
  if run_stage "candidate_create" manager_env_create --prefix "$CANDIDATE_PREFIX" --yes; then
    :
  else
    status=$?
    manager_env_remove --prefix "$CANDIDATE_PREFIX" --yes || true
    CANDIDATE_PREFIX=""
    return "$status"
  fi
  if verify_runtime "$CANDIDATE_PREFIX" candidate; then
    :
  else
    status=$?
    manager_env_remove --prefix "$CANDIDATE_PREFIX" --yes || true
    return "$status"
  fi
  if manager_env_remove --prefix "$CANDIDATE_PREFIX" --yes; then
    :
  else
    status=$?
    return "$status"
  fi
  CANDIDATE_PREFIX=""
  return 0
}

make_backup() {
  local status
  BACKUP_PREFIX="$(mktemp -d "${TMPDIR:-/tmp}/n_response_backup.XXXXXX")"
  rmdir "$BACKUP_PREFIX"
  if "$MANAGER" create --clone "$ENV_NAME" --prefix "$BACKUP_PREFIX" --yes; then
    return 0
  else
    status=$?
  fi
  printf 'Could not clone the verified environment before update\n' >&2
  return "$status"
}

restore_backup() {
  [[ -n "$BACKUP_PREFIX" ]] || return 0
  printf 'Restoring the previous verified environment...\n' >&2
  manager_env_remove --name "$ENV_NAME" --yes || true
  "$MANAGER" create --clone "$BACKUP_PREFIX" --name "$ENV_NAME" --yes
}

rollback() {
  local status="$1" rollback_status
  ROLLBACK_FAILED=0
  if [[ "$TARGET_MUTATED" -eq 1 ]]; then
    if [[ "$ENV_EXISTS" -eq 1 && -n "$BACKUP_PREFIX" ]]; then
      if restore_backup; then
        :
      else
        rollback_status=$?
        ROLLBACK_FAILED=1
        printf 'setup_conda_script.sh: rollback failed (status %s); backup preserved at %s\n' "$rollback_status" "$BACKUP_PREFIX" >&2
      fi
    else
      if manager_env_remove --name "$ENV_NAME" --yes; then
        :
      else
        rollback_status=$?
        ROLLBACK_FAILED=1
        printf 'setup_conda_script.sh: could not remove incomplete environment (status %s)\n' "$rollback_status" >&2
      fi
    fi
  fi
  return "$status"
}

cleanup() {
  if [[ -n "$CANDIDATE_PREFIX" ]]; then
    manager_env_remove --prefix "$CANDIDATE_PREFIX" --yes || true
    CANDIDATE_PREFIX=""
  fi
  if [[ -n "$BACKUP_PREFIX" && -d "$BACKUP_PREFIX" ]]; then
    if [[ "$ROLLBACK_FAILED" -eq 0 ]]; then
      rm -rf -- "$BACKUP_PREFIX"
    else
      printf 'Backup preserved at %s\n' "$BACKUP_PREFIX" >&2
    fi
  fi
}

main() {
  local state status
  resolve_manager
  resolve_runner
  read_environment_name
  validate_manifest_shape
  RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  state="$(environment_exists)"
  if [[ "$state" == "1" ]]; then
    ENV_EXISTS=1
  elif [[ "$state" == "0" ]]; then
    ENV_EXISTS=0
  else
    die "unexpected environment-list response: $state"
  fi
  resolve_target_prefix
  validate_absent_target_prefix

  if solver_and_candidate; then
    :
  else
    status=$?
    return "$status"
  fi
  [[ "$DRY_RUN_ONLY" -eq 1 ]] && return 0

  if [[ "$ENV_EXISTS" -eq 1 ]]; then
    if make_backup; then
      :
    else
      status=$?
      return "$status"
    fi
    TARGET_MUTATED=1
    if "$MANAGER" env update --strict-channel-priority --name "$ENV_NAME" --file "$MANIFEST_FILE" --prune --yes; then
      :
    else
      status=$?
      return "$status"
    fi
  else
    TARGET_MUTATED=1
    if manager_env_create --name "$ENV_NAME" --yes; then
      :
    else
      status=$?
      return "$status"
    fi
  fi

  if verify_named_runtime; then
    :
  else
    status=$?
    return "$status"
  fi
  if capture_environment_evidence; then
    :
  else
    status=$?
    return "$status"
  fi
  printf 'Environment %s is ready.\n' "$ENV_NAME"
  printf 'Optional activation: conda activate %s\n' "$ENV_NAME"
}

status=0
set +e
main
status=$?
set -e
if [[ "$status" -ne 0 ]]; then
  rollback "$status" || status=$?
fi
cleanup
exit "$status"
