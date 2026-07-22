#!/usr/bin/env bash

# Side-effect-free until nrc_log is called with a nonempty output path.
NRC_LOG_INITIALIZED=0
NRC_LOG_COMPONENT=""
NRC_LOG_RUN_ID=""
NRC_LOG_LEVEL_RANK=20
NRC_LOG_FILE=""
NRC_LOG_SEQUENCE=0
NRC_LOG_ROOT=""
NRC_FULL_LOG_FILE=""
NRC_ERROR_WARN_FILE=""
NRC_LOGGING_ACTIVE=0
NRC_LOG_STDOUT_TEE_PID=""
NRC_LOG_STDERR_TEE_PID=""
NRC_LOG_FDS_SAVED=0

nrc_log_level_rank() {
  case "$1" in
    DEBUG) printf '10\n' ;;
    INFO) printf '20\n' ;;
    WARNING) printf '30\n' ;;
    ERROR) printf '40\n' ;;
    *) return 2 ;;
  esac
}

nrc_json_escape() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\b'/\\b}"
  value="${value//$'\f'/\\f}"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  value="${value//$'\t'/\\t}"
  printf '%s' "$value"
}

nrc_human_escape() {
  local value="$1"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  value="${value//$'\t'/\\t}"
  printf '%s' "$value"
}

nrc_log_init() {
  local component="$1" run_id="$2" level="$3" output_path="${4:-}" rank
  [[ -n "$component" ]] || return 2
  [[ -n "$run_id" ]] || return 2
  if rank="$(nrc_log_level_rank "$level")"; then
    :
  else
    return 2
  fi
  NRC_LOG_COMPONENT="$component"
  NRC_LOG_RUN_ID="$run_id"
  NRC_LOG_LEVEL_RANK="$rank"
  NRC_LOG_FILE="$output_path"
  NRC_LOG_SEQUENCE=0
  NRC_LOG_INITIALIZED=1
}

nrc_setup_logging() {
  local project_root="$1" component="$2" run_id="$3" level="$4" log_root="${5:-}"
  [[ "$NRC_LOGGING_ACTIVE" -eq 0 ]] || return 2
  [[ -n "$project_root" ]] || return 2
  [[ "$component" =~ ^[A-Za-z0-9_.-]+$ ]] || return 2
  [[ "$run_id" =~ ^[A-Za-z0-9_.-]+$ ]] || return 2
  [[ -n "$log_root" ]] || log_root="$project_root/logs"

  NRC_LOG_ROOT="$log_root"
  NRC_FULL_LOG_FILE="$NRC_LOG_ROOT/log_files/${run_id}_${component}_full_log.log"
  NRC_LOG_FILE="$NRC_LOG_ROOT/jsonl_logs/${run_id}_${component}_events.jsonl"
  NRC_ERROR_WARN_FILE="$NRC_LOG_ROOT/error_warn_logs/${run_id}_${component}_errors_warnings.log"

  mkdir -p -- \
    "$NRC_LOG_ROOT/log_files" \
    "$NRC_LOG_ROOT/jsonl_logs" \
    "$NRC_LOG_ROOT/error_warn_logs" || return 2
  [[ ! -L "$NRC_FULL_LOG_FILE" ]] || return 2
  [[ ! -L "$NRC_LOG_FILE" ]] || return 2
  [[ ! -L "$NRC_ERROR_WARN_FILE" ]] || return 2
  touch -- "$NRC_FULL_LOG_FILE" "$NRC_LOG_FILE" "$NRC_ERROR_WARN_FILE" || return 2
  nrc_log_init "$component" "$run_id" "$level" "$NRC_LOG_FILE" || return 2
  command -v tee >/dev/null 2>&1 || return 2

  exec 3>&1 4>&2
  NRC_LOG_FDS_SAVED=1
  exec > >(tee -a -- "$NRC_FULL_LOG_FILE")
  NRC_LOG_STDOUT_TEE_PID="$!"
  exec 2> >(tee -a -- "$NRC_FULL_LOG_FILE" >&4)
  NRC_LOG_STDERR_TEE_PID="$!"
  NRC_LOGGING_ACTIVE=1
}

nrc_teardown_logging() {
  [[ "$NRC_LOGGING_ACTIVE" -eq 1 ]] || return 0
  if [[ "$NRC_LOG_FDS_SAVED" -eq 1 ]]; then
    exec 1>&3 2>&4
    exec 3>&- 4>&-
    NRC_LOG_FDS_SAVED=0
  fi
  if [[ -n "$NRC_LOG_STDOUT_TEE_PID" ]]; then
    wait "$NRC_LOG_STDOUT_TEE_PID" 2>/dev/null || true
    NRC_LOG_STDOUT_TEE_PID=""
  fi
  if [[ -n "$NRC_LOG_STDERR_TEE_PID" ]]; then
    wait "$NRC_LOG_STDERR_TEE_PID" 2>/dev/null || true
    NRC_LOG_STDERR_TEE_PID=""
  fi
  NRC_LOGGING_ACTIVE=0
}

nrc_log() {
  local level="$1" event="$2" rank timestamp line field key value lower_key index
  local human_level human_line human_context=""
  local -a field_keys=() field_values=()
  shift 2
  [[ "$NRC_LOG_INITIALIZED" -eq 1 ]] || return 2
  [[ -n "$event" ]] || return 2
  if rank="$(nrc_log_level_rank "$level")"; then
    :
  else
    return 2
  fi
  [[ "$rank" -ge "$NRC_LOG_LEVEL_RANK" ]] || return 0

  for field in "$@"; do
    [[ "$field" == *=* ]] || return 2
    key="${field%%=*}"
    value="${field#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || return 2
    case "$key" in
      timestamp|sequence|level|run_id|event|component) return 2 ;;
    esac
    lower_key="${key,,}"
    case "$lower_key" in
      *password*|*secret*|*token*|*credential*|*api_key*|*apikey*) value="[REDACTED]" ;;
    esac
    field_keys+=("$key")
    field_values+=("$value")
    human_context+="${human_context:+ }$key=$(nrc_human_escape "$value")"
  done

  NRC_LOG_SEQUENCE=$((NRC_LOG_SEQUENCE + 1))
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
  printf -v line \
    '{"timestamp":"%s","sequence":%d,"level":"%s","run_id":"%s","event":"%s","component":"%s"' \
    "$(nrc_json_escape "$timestamp")" \
    "$NRC_LOG_SEQUENCE" \
    "$(nrc_json_escape "$level")" \
    "$(nrc_json_escape "$NRC_LOG_RUN_ID")" \
    "$(nrc_json_escape "$event")" \
    "$(nrc_json_escape "$NRC_LOG_COMPONENT")"

  for index in "${!field_keys[@]}"; do
    line+=",\"$(nrc_json_escape "${field_keys[$index]}")\":\"$(nrc_json_escape "${field_values[$index]}")\""
  done
  line+='}'

  if [[ -n "$NRC_LOG_FILE" ]]; then
    [[ ! -L "$NRC_LOG_FILE" ]] || return 2
    mkdir -p -- "$(dirname -- "$NRC_LOG_FILE")"
    printf '%s\n' "$line" >>"$NRC_LOG_FILE"
  fi
  human_level="$level"
  [[ "$human_level" != "WARNING" ]] || human_level="WARN"
  printf -v human_line '[%s] [%s] %s' "$timestamp" "$human_level" "$event"
  [[ -z "$human_context" ]] || human_line+=" | $human_context"
  if [[ "$level" == "WARNING" || "$level" == "ERROR" ]]; then
    if [[ -n "$NRC_ERROR_WARN_FILE" ]]; then
      [[ ! -L "$NRC_ERROR_WARN_FILE" ]] || return 2
      printf '%s\n' "$human_line" >>"$NRC_ERROR_WARN_FILE"
    fi
    printf '%s\n' "$human_line" >&2
  else
    printf '%s\n' "$human_line"
  fi
}
