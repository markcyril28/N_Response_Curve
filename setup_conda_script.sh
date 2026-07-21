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
MAMBA_MANAGER=""
CONDA_MANAGER=""
ENV_NAME=""
ENV_STATE=""
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

resolve_manager_path() {
  local requested="$1" default_name="$2" candidate
  if [[ -n "$requested" ]]; then
    candidate="$requested"
    [[ "$candidate" == */* ]] || candidate="$(command -v "$candidate" || true)"
    [[ -n "$candidate" ]] || die "${default_name^^}_BIN was not found: $requested"
  elif command -v "$default_name" >/dev/null 2>&1; then
    candidate="$(command -v "$default_name")"
  else
    return 1
  fi
  validate_linux_executable "$candidate"
  printf '%s\n' "$candidate"
}

resolve_manager() {
  MAMBA_MANAGER="$(resolve_manager_path "$MAMBA_BIN" mamba || true)"
  CONDA_MANAGER="$(resolve_manager_path "$CONDA_BIN" conda || true)"
  if [[ -n "$MAMBA_MANAGER" ]]; then
    MANAGER="$MAMBA_MANAGER"
  elif [[ -n "$CONDA_MANAGER" ]]; then
    MANAGER="$CONDA_MANAGER"
  else
    die "No Linux Mamba or Conda executable was found"
  fi
}

resolve_runner() {
  if [[ -n "$CONDA_MANAGER" ]]; then
    RUNNER="$CONDA_MANAGER"
  else
    RUNNER="$MANAGER"
  fi
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

environment_state_with() {
  local manager="$1" environments_json
  if environments_json="$($manager env list --json)"; then
    :
  else
    return $?
  fi
  python -c '
import json
import pathlib
import sys

try:
    data = json.load(sys.stdin)
except (json.JSONDecodeError, UnicodeDecodeError):
    raise SystemExit(2)
envs = data.get("envs") if isinstance(data, dict) else None
if not isinstance(envs, list) or any(not isinstance(item, str) for item in envs):
    raise SystemExit(2)
name = sys.argv[1]
print("1" if any(pathlib.Path(item).name == name for item in envs) else "0")
' "$ENV_NAME" <<<"$environments_json"
}

environment_exists() {
  local state
  if state="$(environment_state_with "$MANAGER")"; then
    ENV_STATE="$state"
    return 0
  fi
  if [[ -n "$MAMBA_MANAGER" && -n "$CONDA_MANAGER" && "$MANAGER" == "$MAMBA_MANAGER" && "$CONDA_MANAGER" != "$MAMBA_MANAGER" ]]; then
    printf 'Mamba environment listing failed or returned invalid JSON. Falling back to Conda.\n' >&2
    MANAGER="$CONDA_MANAGER"
    if state="$(environment_state_with "$MANAGER")"; then
      ENV_STATE="$state"
      return 0
    fi
  fi
  return 1
}

switch_to_conda_after_failure() {
  local reason="$1" state
  [[ -n "$MAMBA_MANAGER" && -n "$CONDA_MANAGER" ]] || return 1
  [[ "$MANAGER" == "$MAMBA_MANAGER" && "$CONDA_MANAGER" != "$MAMBA_MANAGER" ]] || return 1
  printf 'Mamba %s failed. Re-querying environment state and Falling back to Conda.\n' "$reason" >&2
  if state="$(environment_state_with "$CONDA_MANAGER")"; then
    MANAGER="$CONDA_MANAGER"
    ENV_STATE="$state"
    return 0
  fi
  printf 'Conda fallback could not obtain a valid environment listing.\n' >&2
  return 1
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
  local argument r_state_root status
  local is_rscript=0
  local -a run_arguments=()
  local -a clean_environment=(
    -u CONDA_PREFIX
    -u CONDA_DEFAULT_ENV
    -u CONDA_PROMPT_MODIFIER
    -u CONDA_SHLVL
    -u CONDA_EXE
    -u CONDA_PYTHON_EXE
    -u CONDA_ROOT
    PYTHONNOUSERSITE=1
    R_ENVIRON_USER=/dev/null
    R_PROFILE_USER=/dev/null
    R_LIBS_USER=/dev/null
    LANG=C.UTF-8
    LC_ALL=C.UTF-8
    OMP_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1
    MKL_NUM_THREADS=1
    BLIS_NUM_THREADS=1
  )

  # Every Rscript process receives the preferred fresh-session flags exactly
  # once and a unique, disposable user/cache root. No history, workspace,
  # profile, or user cache is read from or written to the operator's home.
  for argument in "$@"; do
    if [[ "${argument##*/}" == "Rscript" ]]; then
      is_rscript=1
      run_arguments+=("$argument" --vanilla --no-save --no-restore)
    elif [[ "$is_rscript" -eq 1 ]]; then
      case "$argument" in
        --vanilla|--no-save|--no-restore) continue ;;
      esac
      run_arguments+=("$argument")
    else
      run_arguments+=("$argument")
    fi
  done

  # A partially initialized parent Conda shell can make `mamba run` activate
  # the wrong prefix. Remove inherited activation state before isolated checks.
  if [[ "$is_rscript" -eq 0 ]]; then
    /usr/bin/env "${clean_environment[@]}" "$RUNNER" run "$@"
    return $?
  fi

  if r_state_root="$(mktemp -d "${TMPDIR:-/tmp}/n_response_r_state.XXXXXX")"; then
    :
  else
    return $?
  fi
  if mkdir -p \
    "$r_state_root/cache/R/pkgcache" \
    "$r_state_root/cache/renv" \
    "$r_state_root/config/R" \
    "$r_state_root/data/R" \
    "$r_state_root/state/R" \
    "$r_state_root/tmp"; then
    :
  else
    status=$?
    rm -rf -- "$r_state_root"
    return "$status"
  fi

  if /usr/bin/env \
    "${clean_environment[@]}" \
    HOME="$r_state_root" \
    R_USER="$r_state_root" \
    R_HISTFILE=/dev/null \
    TMPDIR="$r_state_root/tmp" \
    XDG_CACHE_HOME="$r_state_root/cache" \
    XDG_CONFIG_HOME="$r_state_root/config" \
    XDG_DATA_HOME="$r_state_root/data" \
    XDG_STATE_HOME="$r_state_root/state" \
    R_USER_CACHE_DIR="$r_state_root/cache/R" \
    R_USER_CONFIG_DIR="$r_state_root/config/R" \
    R_USER_DATA_DIR="$r_state_root/data/R" \
    R_PKG_CACHE_DIR="$r_state_root/cache/R/pkgcache" \
    RENV_PATHS_CACHE="$r_state_root/cache/renv" \
    "$RUNNER" run "${run_arguments[@]}"; then
    status=0
  else
    status=$?
  fi
  rm -rf -- "$r_state_root"
  return "$status"
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

capture_environment_export() {
  local destination="$1" raw_export filtered_export line status
  if raw_export="$(mktemp "${destination}.raw.XXXXXX")"; then
    :
  else
    status=$?
    return "$status"
  fi
  if filtered_export="$(mktemp "${destination}.filtered.XXXXXX")"; then
    :
  else
    status=$?
    rm -f -- "$raw_export"
    return "$status"
  fi

  if "$MANAGER" env export --name "$ENV_NAME" --no-builds >"$raw_export" 2>&1; then
    :
  else
    status=$?
    mv -f -- "$raw_export" "$destination"
    rm -f -- "$filtered_export"
    printf 'Evidence capture failed (inspect: %s)\n' "$destination" >&2
    sed -n '1,160p' "$destination" >&2
    return "$status"
  fi

  if while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      prefix:*) continue ;;
    esac
    printf '%s\n' "$line"
  done <"$raw_export" >"$filtered_export"; then
    :
  else
    status=$?
    mv -f -- "$raw_export" "$destination"
    rm -f -- "$filtered_export"
    printf 'Could not remove the machine-specific prefix from %s\n' "$destination" >&2
    return "$status"
  fi

  if [[ -s "$filtered_export" ]]; then
    mv -f -- "$filtered_export" "$destination"
    rm -f -- "$raw_export"
    return 0
  fi
  mv -f -- "$raw_export" "$destination"
  rm -f -- "$filtered_export"
  printf 'Environment export was empty after prefix removal (inspect: %s)\n' "$destination" >&2
  return 1
}

verify_arrow_interchange() {
  local selector="$1" target="$2" label="$3" interchange_root status
  if interchange_root="$(mktemp -d "${TMPDIR:-/tmp}/n_response_interchange.XXXXXX")"; then
    :
  else
    status=$?
    return "$status"
  fi

  if run_stage "${label}_r_to_python_arrow" run_in_env "$selector" "$target" \
    env "N_RESPONSE_INTERCHANGE_DIR=$interchange_root" Rscript --vanilla -e '
root <- Sys.getenv("N_RESPONSE_INTERCHANGE_DIR")
if (!dir.exists(root)) stop("interchange directory is unavailable")
payload <- data.frame(
  stable_id = c("a", "b", "c"),
  label = c("alpha", "β", NA_character_),
  value = c(1.25, NA_real_, 3.5),
  stringsAsFactors = FALSE
)
arrow::write_parquet(payload, file.path(root, "r_to_python.parquet"))
cat("r-to-python-arrow-ok\n")
'; then
    :
  else
    status=$?
    rm -rf -- "$interchange_root"
    return "$status"
  fi

  if run_stage "${label}_python_to_r_arrow" run_in_env "$selector" "$target" \
    env "N_RESPONSE_INTERCHANGE_DIR=$interchange_root" python -c '
import os
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq

root = pathlib.Path(os.environ["N_RESPONSE_INTERCHANGE_DIR"])
payload = pq.read_table(root / "r_to_python.parquet").to_pydict()
expected = {
    "stable_id": ["a", "b", "c"],
    "label": ["alpha", "β", None],
    "value": [1.25, None, 3.5],
}
if payload != expected:
    raise SystemExit(f"R-to-Python Arrow payload changed: {payload!r}")
result = pa.table({
    "stable_id": payload["stable_id"],
    "engine": ["python", "python", "python"],
    "score": [2.5, None, 7.0],
})
pq.write_table(result, root / "python_to_r.parquet")
print("python-to-r-arrow-ok")
'; then
    :
  else
    status=$?
    rm -rf -- "$interchange_root"
    return "$status"
  fi

  # The single-quoted R expression intentionally contains R's `$` operator.
  # shellcheck disable=SC2016
  if run_stage "${label}_python_r_arrow_readback" run_in_env "$selector" "$target" \
    env "N_RESPONSE_INTERCHANGE_DIR=$interchange_root" Rscript --vanilla -e '
root <- Sys.getenv("N_RESPONSE_INTERCHANGE_DIR")
payload <- as.data.frame(
  arrow::read_parquet(file.path(root, "python_to_r.parquet")),
  stringsAsFactors = FALSE
)
stopifnot(
  identical(as.character(payload$stable_id), c("a", "b", "c")),
  identical(as.character(payload$engine), rep("python", 3L)),
  isTRUE(all.equal(as.numeric(payload$score), c(2.5, NA_real_, 7.0), check.attributes = FALSE))
)
cat("python-r-arrow-interchange-ok\n")
'; then
    :
  else
    status=$?
    rm -rf -- "$interchange_root"
    return "$status"
  fi

  rm -rf -- "$interchange_root"
  return 0
}

verify_runtime() {
  local selector="$1" target="$2" label="$3" status
  printf 'Verifying %s runtime through %s run...\n' "$label" "$(basename "$RUNNER")"
  if run_stage "${label}_python_imports" run_in_env "$selector" "$target" python -c '
import importlib
import os
import pathlib
import shutil
import sys

prefix = pathlib.Path(os.environ["CONDA_PREFIX"]).resolve()
executable = pathlib.Path(sys.executable).resolve()
if not executable.is_relative_to(prefix):
    raise SystemExit(f"Python executable is outside CONDA_PREFIX: {executable}")
for name in ("numpy", "pandas", "scipy", "statsmodels", "sklearn", "matplotlib", "seaborn", "pyarrow", "openpyxl", "pytest", "pytest_cov", "coverage"):
    module = importlib.import_module(name)
    module_path = getattr(module, "__file__", None)
    if module_path and not pathlib.Path(module_path).resolve().is_relative_to(prefix):
        raise SystemExit(f"{name} imported outside CONDA_PREFIX: {module_path}")
for command in ("ruff", "shellcheck"):
    command_path = shutil.which(command)
    if not command_path or not pathlib.Path(command_path).resolve().is_relative_to(prefix):
        raise SystemExit(f"{command} is unavailable inside CONDA_PREFIX: {command_path}")
print("python-runtime-ok")
'; then
    :
  else
    status=$?
    return "$status"
  fi
  if run_stage "${label}_pytest_version" run_in_env "$selector" "$target" python -m pytest --version; then
    :
  else
    status=$?
    return "$status"
  fi
  if run_stage "${label}_ruff_version" run_in_env "$selector" "$target" ruff --version; then
    :
  else
    status=$?
    return "$status"
  fi
  if run_stage "${label}_shellcheck_version" run_in_env "$selector" "$target" shellcheck --version; then
    :
  else
    status=$?
    return "$status"
  fi
  if run_stage "${label}_r_version" run_in_env "$selector" "$target" Rscript --version; then
    :
  else
    status=$?
    return "$status"
  fi
  if run_stage "${label}_r_smoke" run_in_env "$selector" "$target" Rscript --vanilla "$PROJECT_ROOT/tests/r_environment_smoke.R"; then
    :
  else
    status=$?
    return "$status"
  fi
  if verify_arrow_interchange "$selector" "$target" "$label"; then
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
  if capture_environment_export "$destination"; then
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
    if switch_to_conda_after_failure "solver dry-run"; then
      cat "$solver_log" >&2
      solver_log="$METADATA_ROOT/solver_conda_fallback_${stamp}.log"
      if manager_env_create --dry-run --prefix "$candidate_prefix" --yes >"$solver_log" 2>&1; then
        :
      else
        status=$?
        cat "$solver_log" >&2
        printf 'Conda solver dry-run failed; no named environment was changed\n' >&2
        return "$status"
      fi
    else
      cat "$solver_log" >&2
      printf 'Conda solver dry-run failed; no named environment was changed\n' >&2
      return "$status"
    fi
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
    if switch_to_conda_after_failure "candidate creation"; then
      "$MANAGER" env remove --prefix "$CANDIDATE_PREFIX" --yes >/dev/null 2>&1 || true
      rm -rf -- "$CANDIDATE_PREFIX"
      if candidate_prefix="$(mktemp -d "${TMPDIR:-/tmp}/n_response_candidate.XXXXXX")"; then
        rmdir "$candidate_prefix"
        CANDIDATE_PREFIX="$candidate_prefix"
      else
        status=$?
        CANDIDATE_PREFIX=""
        return "$status"
      fi
      if run_stage "candidate_create_conda_fallback" manager_env_create --prefix "$CANDIDATE_PREFIX" --yes; then
        :
      else
        status=$?
        manager_env_remove --prefix "$CANDIDATE_PREFIX" --yes || true
        CANDIDATE_PREFIX=""
        return "$status"
      fi
    else
      manager_env_remove --prefix "$CANDIDATE_PREFIX" --yes || true
      CANDIDATE_PREFIX=""
      return "$status"
    fi
  fi
  if verify_runtime --prefix "$CANDIDATE_PREFIX" candidate; then
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

apply_named_environment_state() {
  local state="$1"
  if [[ "$state" == "1" ]]; then
    "$MANAGER" env update --strict-channel-priority --name "$ENV_NAME" --file "$MANIFEST_FILE" --prune --yes
    return $?
  fi
  if [[ "$state" == "0" ]]; then
    # A failed create/update can leave an unregistered partial prefix. This
    # exact target was absent initially or is protected by BACKUP_PREFIX.
    if [[ -e "$ENV_PREFIX" ]]; then
      "$MANAGER" env remove --prefix "$ENV_PREFIX" --yes >/dev/null 2>&1 || true
      rm -rf -- "$ENV_PREFIX"
    fi
    manager_env_create --name "$ENV_NAME" --yes
    return $?
  fi
  printf 'Unexpected environment state while rebuilding action: %s\n' "$state" >&2
  return 2
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
  environment_exists || die "could not inspect Conda environments with valid JSON"
  state="$ENV_STATE"
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
  fi
  TARGET_MUTATED=1
  if apply_named_environment_state "$state"; then
    :
  else
    status=$?
    if switch_to_conda_after_failure "named environment mutation"; then
      if apply_named_environment_state "$ENV_STATE"; then
        :
      else
        status=$?
        return "$status"
      fi
    else
      return "$status"
    fi
  fi

  if verify_runtime --name "$ENV_NAME" named; then
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
