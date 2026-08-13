from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
from threading import Lock
import traceback as traceback_module
from typing import Any, TextIO


# Single source for the in-package run-log path. The release manifest advertises
# this same string, so the two must never be written as independent literals.
PACKAGE_LOG_RELATIVE_PATH = "logs/pipeline.jsonl"

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
_SENSITIVE_KEY_FRAGMENTS = ("password", "secret", "token", "credential", "api_key", "apikey")
_ANSI_RESET = "\033[0m"
_ANSI_DIM = "\033[2m"
_ANSI_BOLD = "\033[1m"
_ANSI_BLUE = "\033[34m"
_ANSI_CYAN = "\033[36m"
_ANSI_GREEN = "\033[32m"
_ANSI_MAGENTA = "\033[35m"
_ANSI_YELLOW = "\033[33m"
_ANSI_BRIGHT_BLUE = "\033[1;34m"
_ANSI_BRIGHT_CYAN = "\033[1;36m"
_ANSI_BRIGHT_GREEN = "\033[1;32m"
_ANSI_BRIGHT_RED = "\033[1;31m"
_ANSI_BRIGHT_YELLOW = "\033[1;33m"
_INLINE_SENSITIVE_VALUE = re.compile(
    r"""(?ix)
    (?P<label>
        ["']?\b[A-Z0-9_.-]*(?:password|secret|token|credential|api[_-]?key|apikey)[A-Z0-9_.-]*\b["']?
        \s*(?:=|:)\s*
    )
    (?P<value>
        "(?:[^"\\]|\\.)*"
        | '(?:[^'\\]|\\.)*'
        | [^\s,;\)\]\}'"\r\n]+
    )
    """
)
_RESERVED_FIELDS = {"timestamp", "sequence", "level", "run_id", "event"}
_EVENT_TITLES = {
    "launcher_started": "Launcher started",
    "launcher_handoff": "Starting pipeline",
    "launcher_completed": "Launcher completed",
    "launcher_failed": "Launcher failed",
    "run_started": "Run started",
    "run_completed": "Run completed",
    "setup_started": "Setup started",
    "setup_completed": "Setup completed",
    "controlled_release_started": "Preparing controlled release",
    "output_root_cleared": "Output root cleared",
    "validation_completed": "Validation completed",
    "configuration_error": "Configuration error",
    "runtime_error": "Runtime error",
}
_FIELD_LABELS = {
    "config_path": "config",
    "module_path": "module",
    "full_log_path": "full log",
    "event_log_path": "event log",
    "error_log_path": "warnings",
    "launcher_run_id": "launcher",
    "python_bin": "Python",
    "writes_outputs": "outputs",
    "release_target": "release",
    "output_root": "output root",
    "removed_entries": "removed",
    "release_package": "release",
    "status": "exit code",
    "failed_stage": "failed stage",
    "error_type": "error type",
    "error": "message",
    "root_cause_type": "root cause type",
    "root_cause": "root cause",
    "failure_location": "location",
    "exception_chain": "exception chain",
    "traceback": "traceback",
}
_CONSOLE_OMITTED_FIELDS = frozenset({"exception_chain"})
_LOGGED_EXCEPTION_ATTRIBUTE = "_n_response_logged"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _redact_inline_sensitive_values(value: str) -> str:
    """Redact values named in free-form diagnostics, including tracebacks."""

    def replace(match: re.Match[str]) -> str:
        captured = match.group("value")
        if len(captured) >= 2 and captured[0] == captured[-1] and captured[0] in {"'", '"'}:
            return f"{match.group('label')}{captured[0]}[REDACTED]{captured[0]}"
        return f"{match.group('label')}[REDACTED]"

    return _INLINE_SENSITIVE_VALUE.sub(replace, value)


def _safe_value(value: Any, *, key: str = "") -> Any:
    if any(fragment in key.casefold() for fragment in _SENSITIVE_KEY_FRAGMENTS):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(nested_key): _safe_value(nested, key=str(nested_key)) for nested_key, nested in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe_value(item, key=key) for item in value]
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return _redact_inline_sensitive_values(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, datetime):
        return _timestamp(value)
    if isinstance(value, Path):
        return value.as_posix()
    return _redact_inline_sensitive_values(str(value))


def exception_was_logged(exc: BaseException) -> bool:
    """Return whether a RunLogger already emitted this exception's traceback."""

    return bool(getattr(exc, _LOGGED_EXCEPTION_ATTRIBUTE, False))


def _human_value(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _console_value(value: Any, *, project_root: Path | None) -> str:
    text = _human_value(value)
    if project_root is None or not isinstance(value, str):
        return text
    try:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate.relative_to(project_root).as_posix() or "."
    except ValueError:
        pass
    root_text = project_root.as_posix()
    text = text.replace(f"{root_text}/", "")
    text = text.replace(root_text, ".")
    return text


def _exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    """Return the exception chain from the reported error to its root cause."""

    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return tuple(chain)


def _exception_fields(exc: BaseException) -> dict[str, Any]:
    """Build stable, searchable diagnostics without consuming the exception."""

    chain = _exception_chain(exc)
    root_cause = chain[-1]
    frames = traceback_module.extract_tb(root_cause.__traceback__)
    failure_location = None
    if frames:
        frame = frames[-1]
        failure_location = f"{frame.filename}:{frame.lineno} in {frame.name}"
    return {
        "error_type": type(exc).__name__,
        "error": str(exc) or "<no message>",
        "root_cause_type": type(root_cause).__name__,
        "root_cause": str(root_cause) or "<no message>",
        "failure_location": failure_location,
        "exception_chain": [
            {
                "error_type": type(item).__name__,
                "error": str(item) or "<no message>",
            }
            for item in chain
        ],
        "traceback": "".join(
            traceback_module.format_exception(type(exc), exc, exc.__traceback__, chain=True)
        ).rstrip(),
    }


def _event_title(event: str, fields: Mapping[str, Any]) -> tuple[str, frozenset[str]]:
    stage = fields.get("stage")
    if event in {"stage_started", "stage_completed", "stage_failed"} and isinstance(stage, str):
        state = event.removeprefix("stage_")
        return f"{stage.replace('_', ' ').title()} {state}", frozenset({"stage"})
    phase = fields.get("phase")
    if event == "resource_snapshot" and isinstance(phase, str):
        return f"Resources · {phase.title()}", frozenset({"phase"})
    return _EVENT_TITLES.get(event, event.replace("_", " ").capitalize()), frozenset()


def _event_icon(level: str, event: str) -> str:
    if level == "ERROR" or event.endswith("_failed"):
        return "✖"
    if level == "WARNING":
        return "⚠"
    if event.endswith("_completed"):
        return "✓"
    if event in {"stage_started", "launcher_handoff"}:
        return "▶"
    if event == "resource_snapshot":
        return "◇"
    if event.endswith("_started"):
        return "◆"
    return "•"


def _paint(value: str, code: str, *, color: bool) -> str:
    return f"{code}{value}{_ANSI_RESET}" if color else value


def _event_color(level: str, event: str) -> str:
    if level == "ERROR" or event.endswith("_failed"):
        return _ANSI_BRIGHT_RED
    if level == "WARNING":
        return _ANSI_BRIGHT_YELLOW
    if event.endswith("_completed"):
        return _ANSI_BRIGHT_GREEN
    if event == "resource_snapshot":
        return _ANSI_MAGENTA
    if event in {"stage_started", "launcher_handoff"} or event.endswith("_started"):
        return _ANSI_BRIGHT_BLUE
    return _ANSI_BRIGHT_CYAN


def _value_color(key: str, value: Any) -> str | None:
    if "[REDACTED]" in str(value):
        return _ANSI_MAGENTA
    if key in {"error", "root_cause", "failure_location"}:
        return _ANSI_BRIGHT_RED
    if key == "status":
        return _ANSI_GREEN if str(value) in {"0", "ok", "success"} else _ANSI_BRIGHT_RED
    if key.endswith(("_path", "_package")) or key in {
        "config_path",
        "module_path",
        "full_log_path",
        "event_log_path",
        "error_log_path",
    }:
        return _ANSI_CYAN
    if isinstance(value, bool):
        return _ANSI_GREEN if value else _ANSI_YELLOW
    return None


def console_colors_enabled(stream: TextIO) -> bool:
    """Return whether an interactive console should receive ANSI color accents."""

    if "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb":
        return False
    requested = os.environ.get("NRC_CONSOLE_COLOR", "").casefold()
    if requested in {"0", "false", "never"}:
        return False
    if requested in {"1", "true", "always"}:
        return True
    return bool(hasattr(stream, "isatty") and stream.isatty())


def format_console_event(
    *,
    timestamp: str,
    level: str,
    event: str,
    fields: Mapping[str, Any],
    project_root: Path | None = None,
    color: bool = False,
) -> str:
    """Render one structured event as a compact, readable console block."""

    fields = {
        str(key): _safe_value(value, key=str(key))
        for key, value in fields.items()
    }
    human_level = "WARN" if level == "WARNING" else level
    title, consumed = _event_title(event, fields)
    short_timestamp = timestamp[11:23] if len(timestamp) >= 23 else timestamp
    event_color = _event_color(level, event)
    header = "  ".join(
        (
            _paint(short_timestamp, _ANSI_DIM, color=color),
            _paint(f"{human_level:<5}", event_color, color=color),
            f"{_paint(_event_icon(level, event), event_color, color=color)} "
            f"{_paint(title, _ANSI_BOLD, color=color)}",
        )
    )

    traceback_value = fields.get("traceback")
    details = [
        (key, value)
        for key, value in fields.items()
        if key not in consumed
        and key not in _CONSOLE_OMITTED_FIELDS
        and key != "traceback"
    ]
    if (
        fields.get("root_cause_type") == fields.get("error_type")
        and fields.get("root_cause") == fields.get("error")
    ):
        details = [
            (key, value)
            for key, value in details
            if key not in {"root_cause_type", "root_cause"}
        ]
    traceback_lines = (
        traceback_value.splitlines()
        if isinstance(traceback_value, str) and traceback_value
        else []
    )
    if not details and not traceback_lines:
        return header
    detail_labels = [_FIELD_LABELS.get(key, key.replace("_", " ")) for key, _ in details]
    labels = [*detail_labels, *([_FIELD_LABELS["traceback"]] if traceback_lines else [])]
    width = min(max(len(label) for label in labels), 24)
    lines = [header]
    for index, ((key, value), label) in enumerate(zip(details, detail_labels, strict=True)):
        branch = "└─" if index == len(details) - 1 and not traceback_lines else "├─"
        # Split before rendering: _console_value escapes newlines, so a multi-line
        # value has to become one indented block per line rather than one long line.
        raw_lines = value.split("\n") if isinstance(value, str) else [value]
        rendered_lines = [
            _console_value(item, project_root=project_root) for item in raw_lines
        ]
        value_color = _value_color(key, value)
        painted = [
            _paint(rendered, value_color, color=color) if value_color else rendered
            for rendered in rendered_lines
        ]
        lines.append(
            "                  "
            f"{_paint(branch, _ANSI_DIM, color=color)} "
            f"{_paint(f'{label:<{width}}', _ANSI_BLUE, color=color)}  "
            f"{painted[0]}"
        )
        continuation = " " * len(f"                  ├─ {'':<{width}}  ")
        lines.extend(f"{continuation}{rendered}" for rendered in painted[1:])
    if traceback_lines:
        traceback_label = f"{_FIELD_LABELS['traceback']:<{width}}"
        lines.append(
            "                  "
            f"{_paint('└─', _ANSI_DIM, color=color)} "
            f"{_paint(traceback_label, _ANSI_BRIGHT_RED, color=color)}"
        )
        lines.extend(
            f"                     {_paint(_console_value(line, project_root=project_root), _ANSI_DIM, color=color)}"
            for line in traceback_lines
        )
    return "\n".join(lines)


def format_console_exception(
    *,
    timestamp: str,
    event: str,
    exc: BaseException,
    project_root: Path | None = None,
    color: bool = False,
) -> str:
    """Render one unhandled exception with redacted chain and traceback details."""

    return format_console_event(
        timestamp=timestamp,
        level="ERROR",
        event=event,
        fields=_exception_fields(exc),
        project_root=project_root,
        color=color,
    )


class RunLogger:
    """Level-filtered structured run log with optional human-readable console output."""

    def __init__(
        self,
        *,
        level: str,
        run_id: str,
        clock: Callable[[], datetime] = _utc_now,
        stream: TextIO | None = sys.stderr,
        error_stream: TextIO | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        if level not in _LEVELS:
            raise ValueError(f"Unknown logging level: {level}")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a nonempty string")
        self.level = level
        self.run_id = run_id
        self._clock = clock
        self._stream = stream
        self._error_stream = error_stream
        self._project_root = Path(project_root).resolve() if project_root is not None else None
        self._records: list[dict[str, Any]] = []
        self._lock = Lock()

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(deepcopy(self._records))

    def debug(self, event: str, **fields: Any) -> None:
        self.log("DEBUG", event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self.log("INFO", event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self.log("WARNING", event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self.log("ERROR", event, **fields)

    def exception(self, event: str, exc: BaseException, **fields: Any) -> None:
        """Log a complete exception chain and traceback as one ERROR event."""

        diagnostics = _exception_fields(exc)
        collisions = sorted(set(fields).intersection(diagnostics))
        if collisions:
            raise ValueError(
                "Exception context contains protected diagnostic field(s): "
                + ", ".join(collisions)
            )
        self.error(event, **fields, **diagnostics)
        try:
            setattr(exc, _LOGGED_EXCEPTION_ATTRIBUTE, True)
        except (AttributeError, TypeError):
            pass

    def log(self, level: str, event: str, **fields: Any) -> None:
        if level not in _LEVELS:
            raise ValueError(f"Unknown logging level: {level}")
        if not isinstance(event, str) or not event.strip():
            raise ValueError("event must be a nonempty string")
        if _LEVELS[level] < _LEVELS[self.level]:
            return
        reserved = sorted(_RESERVED_FIELDS.intersection(fields))
        if reserved:
            raise ValueError(f"Event context contains reserved audit field(s): {', '.join(reserved)}")
        safe_fields = {str(key): _safe_value(value, key=str(key)) for key, value in fields.items()}
        with self._lock:
            record = {
                "timestamp": _timestamp(self._clock()),
                "sequence": len(self._records) + 1,
                "level": level,
                "run_id": self.run_id,
                "event": event,
                **safe_fields,
            }
            self._records.append(record)
            destination = (
                self._error_stream
                if level in {"WARNING", "ERROR"} and self._error_stream is not None
                else self._stream
            )
            if destination is not None:
                details = {
                    key: value
                    for key, value in record.items()
                    if key not in {"timestamp", "sequence", "level", "run_id", "event"}
                }
                print(
                    format_console_event(
                        timestamp=record["timestamp"],
                        level=level,
                        event=event,
                        fields=details,
                        project_root=self._project_root,
                        color=console_colors_enabled(destination),
                    ),
                    file=destination,
                    flush=True,
                )

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("stage name must be a nonempty string")
        self.info("stage_started", stage=name)
        try:
            yield
        except Exception as exc:
            self.exception(
                "stage_failed",
                exc,
                stage=name,
            )
            raise
        else:
            self.info("stage_completed", stage=name)

    def write_jsonl(self, stage_root: str | Path) -> Path:
        """Write the current immutable snapshot inside an atomic release staging root."""

        destination = Path(stage_root) / PACKAGE_LOG_RELATIVE_PATH
        destination.parent.mkdir(parents=True, exist_ok=True)
        records = self.records
        payload = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for record in records
        )
        destination.write_text(payload, encoding="utf-8", newline="")
        return destination


__all__ = [
    "PACKAGE_LOG_RELATIVE_PATH",
    "RunLogger",
    "console_colors_enabled",
    "exception_was_logged",
    "format_console_event",
    "format_console_exception",
]
