#!/usr/bin/env bash

# Sourcing is side-effect-free; nrc_setup_logging creates the requested log tree.
NRC_LOG_INITIALIZED=0
NRC_LOG_COMPONENT=""
NRC_LOG_RUN_ID=""
NRC_LOG_LEVEL_RANK=20
NRC_LOG_FILE=""
NRC_LOG_SEQUENCE=0
NRC_LOG_ROOT=""
NRC_LOG_PROJECT_ROOT=""
NRC_FULL_LOG_FILE=""
NRC_ERROR_WARN_FILE=""
NRC_LOGGING_ACTIVE=0
NRC_LOG_STDOUT_TEE_PID=""
NRC_LOG_STDERR_TEE_PID=""
NRC_LOG_FULL_WRITER_PID=""
NRC_LOG_ERROR_WRITER_PID=""
NRC_LOG_FDS_SAVED=0
NRC_LOG_COLOR_ACTIVE=0
NRC_LOG_COLOR_AUTO=0
NRC_ANSI_RESET=$'\033[0m'
NRC_ANSI_DIM=$'\033[2m'
NRC_ANSI_BOLD=$'\033[1m'
NRC_ANSI_BLUE=$'\033[34m'
NRC_ANSI_CYAN=$'\033[36m'
NRC_ANSI_GREEN=$'\033[32m'
NRC_ANSI_MAGENTA=$'\033[35m'
NRC_ANSI_YELLOW=$'\033[33m'
NRC_ANSI_BRIGHT_BLUE=$'\033[1;34m'
NRC_ANSI_BRIGHT_CYAN=$'\033[1;36m'
NRC_ANSI_BRIGHT_GREEN=$'\033[1;32m'
NRC_ANSI_BRIGHT_RED=$'\033[1;31m'
NRC_ANSI_BRIGHT_YELLOW=$'\033[1;33m'

nrc_log_level_rank() {
  case "$1" in
    DEBUG) printf '10\n' ;;
    INFO) printf '20\n' ;;
    WARNING) printf '30\n' ;;
    ERROR) printf '40\n' ;;
    *) return 2 ;;
  esac
}

nrc_colors_enabled() {
  local requested="${NRC_CONSOLE_COLOR:-}"
  if [[ -n "${NO_COLOR+x}" ]]; then
    return 1
  fi
  case "${requested,,}" in
    0|false|never) return 1 ;;
    1|true|always) return 0 ;;
  esac
  if [[ "${TERM:-}" == "dumb" ]]; then
    return 1
  fi
  [[ -t 1 || -t 2 ]]
}

# Persisted logs are read in editors, not terminals. Color is added at format
# time in Python and Bash, and the tee splits downstream of that, so the only
# place a plain-text file can be produced is at the sink that owns it.
nrc_strip_ansi() {
  sed -E $'s/\033\\[[0-9;]*[A-Za-z]//g'
}

nrc_human_paint() {
  local code="$1" value="$2"
  if [[ "$NRC_LOG_COLOR_ACTIVE" -eq 1 && -n "$code" ]]; then
    printf '%s%s%s' "$code" "$value" "$NRC_ANSI_RESET"
  else
    printf '%s' "$value"
  fi
}

nrc_human_event_color() {
  local level="$1" event="$2"
  if [[ "$level" == "ERROR" || "$event" == *_failed ]]; then
    printf '%s' "$NRC_ANSI_BRIGHT_RED"
  elif [[ "$level" == "WARNING" ]]; then
    printf '%s' "$NRC_ANSI_BRIGHT_YELLOW"
  elif [[ "$event" == *_completed ]]; then
    printf '%s' "$NRC_ANSI_BRIGHT_GREEN"
  elif [[ "$event" == "resource_snapshot" ]]; then
    printf '%s' "$NRC_ANSI_MAGENTA"
  elif [[ "$event" == "stage_started" || "$event" == "launcher_handoff" || "$event" == *_started ]]; then
    printf '%s' "$NRC_ANSI_BRIGHT_BLUE"
  else
    printf '%s' "$NRC_ANSI_BRIGHT_CYAN"
  fi
}

nrc_human_value_color() {
  local key="$1" value="$2" lower_key
  lower_key="${key,,}"
  if [[ "$value" == *'[REDACTED]'* ]]; then
    printf '%s' "$NRC_ANSI_MAGENTA"
  elif [[ "$key" == "error" || "$key" == "root_cause" || "$key" == "failure_location" ]]; then
    printf '%s' "$NRC_ANSI_BRIGHT_RED"
  elif [[ "$key" == "status" ]]; then
    if [[ "$value" == "0" || "$value" == "ok" || "$value" == "success" ]]; then
      printf '%s' "$NRC_ANSI_GREEN"
    elif [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
      printf '%s' "$NRC_ANSI_BRIGHT_RED"
    fi
  elif [[ "$lower_key" == *_path || "$lower_key" == *_package ]]; then
    printf '%s' "$NRC_ANSI_CYAN"
  elif [[ "$key" == "writes_outputs" ]]; then
    if [[ "$value" == "true" ]]; then
      printf '%s' "$NRC_ANSI_GREEN"
    elif [[ "$value" == "false" ]]; then
      printf '%s' "$NRC_ANSI_YELLOW"
    fi
  fi
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

nrc_human_title() {
  local event="$1" stage="${2:-}" phase="${3:-}" title
  case "$event" in
    launcher_started) printf 'Launcher started' ;;
    launcher_handoff) printf 'Starting pipeline' ;;
    launcher_completed) printf 'Launcher completed' ;;
    launcher_failed) printf 'Launcher failed' ;;
    run_started) printf 'Run started' ;;
    run_completed) printf 'Run completed' ;;
    setup_started) printf 'Setup started' ;;
    setup_completed) printf 'Setup completed' ;;
    controlled_release_started) printf 'Preparing controlled release' ;;
    validation_completed) printf 'Validation completed' ;;
    stage_started|stage_completed|stage_failed)
      title="${stage//_/ }"
      printf '%s %s' "${title^}" "${event#stage_}"
      ;;
    resource_snapshot)
      printf 'Resources · %s' "${phase^}"
      ;;
    *)
      title="${event//_/ }"
      printf '%s' "${title^}"
      ;;
  esac
}

nrc_human_icon() {
  local level="$1" event="$2"
  if [[ "$level" == "ERROR" || "$event" == *_failed ]]; then
    printf '✖'
  elif [[ "$level" == "WARNING" ]]; then
    printf '⚠'
  elif [[ "$event" == *_completed ]]; then
    printf '✓'
  elif [[ "$event" == "stage_started" || "$event" == "launcher_handoff" ]]; then
    printf '▶'
  elif [[ "$event" == "resource_snapshot" ]]; then
    printf '◇'
  elif [[ "$event" == *_started ]]; then
    printf '◆'
  else
    printf '•'
  fi
}

nrc_human_label() {
  case "$1" in
    config_path) printf 'config' ;;
    module_path) printf 'module' ;;
    full_log_path) printf 'full log' ;;
    event_log_path) printf 'event log' ;;
    error_log_path) printf 'warnings' ;;
    launcher_run_id) printf 'launcher' ;;
    python_bin) printf 'Python' ;;
    writes_outputs) printf 'outputs' ;;
    release_target|release_package) printf 'release' ;;
    status) printf 'exit code' ;;
    *) printf '%s' "${1//_/ }" ;;
  esac
}

nrc_human_value() {
  local value="$1"
  if [[ -n "$NRC_LOG_PROJECT_ROOT" && "$value" == "$NRC_LOG_PROJECT_ROOT"/* ]]; then
    value="${value#"$NRC_LOG_PROJECT_ROOT"/}"
  fi
  nrc_human_escape "$value"
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
  NRC_LOG_PROJECT_ROOT=""
  NRC_LOG_SEQUENCE=0
  NRC_LOG_COLOR_AUTO=0
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
  NRC_LOG_PROJECT_ROOT="$project_root"
  if nrc_colors_enabled; then
    NRC_LOG_COLOR_AUTO=1
    export NRC_CONSOLE_COLOR=always
  fi
  command -v tee >/dev/null 2>&1 || return 2
  command -v sed >/dev/null 2>&1 || return 2

  exec 3>&1 4>&2
  NRC_LOG_FDS_SAVED=1
  # One process owns each log file. Both channels forward to its pipe, avoiding
  # concurrent file appenders while preserving terminal stdout/stderr. Each
  # owner strips ANSI on the way in, so the terminal keeps its accents and the
  # files stay plain text.
  exec 5> >(nrc_strip_ansi >>"$NRC_FULL_LOG_FILE")
  NRC_LOG_FULL_WRITER_PID="$!"
  exec 6> >(nrc_strip_ansi >>"$NRC_ERROR_WARN_FILE")
  NRC_LOG_ERROR_WRITER_PID="$!"
  exec > >(tee -- /dev/fd/5 >&3)
  NRC_LOG_STDOUT_TEE_PID="$!"
  # Keep stderr on stderr while retaining every Bash/Python diagnostic in both
  # the full transcript and the dedicated error/warning log.
  exec 2> >(tee -- /dev/fd/6 /dev/fd/5 >&4)
  NRC_LOG_STDERR_TEE_PID="$!"
  NRC_LOGGING_ACTIVE=1
}

nrc_teardown_logging() {
  local wait_status=0
  [[ "$NRC_LOGGING_ACTIVE" -eq 1 ]] || return 0
  if [[ "$NRC_LOG_FDS_SAVED" -eq 1 ]]; then
    exec 1>&3 2>&4
    exec 5>&- 6>&-
    exec 3>&- 4>&-
    NRC_LOG_FDS_SAVED=0
  fi
  if [[ -n "$NRC_LOG_STDOUT_TEE_PID" ]]; then
    wait "$NRC_LOG_STDOUT_TEE_PID" 2>/dev/null || wait_status=2
    NRC_LOG_STDOUT_TEE_PID=""
  fi
  if [[ -n "$NRC_LOG_STDERR_TEE_PID" ]]; then
    wait "$NRC_LOG_STDERR_TEE_PID" 2>/dev/null || wait_status=2
    NRC_LOG_STDERR_TEE_PID=""
  fi
  if [[ -n "$NRC_LOG_FULL_WRITER_PID" ]]; then
    wait "$NRC_LOG_FULL_WRITER_PID" 2>/dev/null || wait_status=2
    NRC_LOG_FULL_WRITER_PID=""
  fi
  if [[ -n "$NRC_LOG_ERROR_WRITER_PID" ]]; then
    wait "$NRC_LOG_ERROR_WRITER_PID" 2>/dev/null || wait_status=2
    NRC_LOG_ERROR_WRITER_PID=""
  fi
  NRC_LOGGING_ACTIVE=0
  return "$wait_status"
}

nrc_log() {
  local level="$1" event="$2" rank timestamp line field key value lower_key index
  local human_level human_line title icon stage="" phase="" label branch rendered event_color value_color
  local label_width=0 visible_count=0 visible_index=0
  local -a field_keys=() field_values=()
  local -a visible_keys=() visible_values=() visible_labels=()
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
    [[ "$key" != "stage" ]] || stage="$value"
    [[ "$key" != "phase" ]] || phase="$value"
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
  title="$(nrc_human_title "$event" "$stage" "$phase")"
  icon="$(nrc_human_icon "$level" "$event")"
  if [[ "$NRC_LOG_COLOR_AUTO" -eq 1 ]] || nrc_colors_enabled; then
    NRC_LOG_COLOR_ACTIVE=1
  else
    NRC_LOG_COLOR_ACTIVE=0
  fi
  event_color="$(nrc_human_event_color "$level" "$event")"
  printf -v human_line '%s  %s  %s %s' \
    "$(nrc_human_paint "$NRC_ANSI_DIM" "${timestamp:11:12}")" \
    "$(nrc_human_paint "$event_color" "$(printf '%-5s' "$human_level")")" \
    "$(nrc_human_paint "$event_color" "$icon")" \
    "$(nrc_human_paint "$NRC_ANSI_BOLD" "$title")"
  for index in "${!field_keys[@]}"; do
    key="${field_keys[$index]}"
    if [[ ( "$event" == "stage_started" || "$event" == "stage_completed" || "$event" == "stage_failed" ) && "$key" == "stage" ]]; then
      continue
    fi
    if [[ "$event" == "resource_snapshot" && "$key" == "phase" ]]; then
      continue
    fi
    label="$(nrc_human_label "$key")"
    visible_keys+=("$key")
    visible_values+=("${field_values[$index]}")
    visible_labels+=("$label")
    if (( ${#label} > label_width )); then
      label_width=${#label}
    fi
  done
  if (( label_width > 24 )); then
    label_width=24
  fi
  visible_count=${#visible_values[@]}
  for visible_index in "${!visible_values[@]}"; do
    branch='├─'
    if (( visible_index + 1 == visible_count )); then
      branch='└─'
    fi
    rendered="$(nrc_human_value "${visible_values[$visible_index]}")"
    value_color="$(nrc_human_value_color "${visible_keys[$visible_index]}" "${visible_values[$visible_index]}")"
    printf -v line '                  %s %s  %s' \
      "$(nrc_human_paint "$NRC_ANSI_DIM" "$branch")" \
      "$(nrc_human_paint "$NRC_ANSI_BLUE" "$(printf "%-*s" "$label_width" "${visible_labels[$visible_index]}")")" \
      "$(nrc_human_paint "$value_color" "$rendered")"
    human_line+=$'\n'"$line"
  done

  if [[ "$level" == "WARNING" || "$level" == "ERROR" ]]; then
    printf '%s\n' "$human_line" >&2
  else
    printf '%s\n' "$human_line"
  fi
}
