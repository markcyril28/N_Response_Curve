from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from threading import Lock
from typing import Any, TextIO


_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
_SENSITIVE_KEY_FRAGMENTS = ("password", "secret", "token", "credential", "api_key", "apikey")
_RESERVED_FIELDS = {"timestamp", "sequence", "level", "run_id", "event"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_value(value: Any, *, key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {str(nested_key): _safe_value(nested, key=str(nested_key)) for nested_key, nested in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe_value(item, key=key) for item in value]
    if any(fragment in key.casefold() for fragment in _SENSITIVE_KEY_FRAGMENTS):
        return "[REDACTED]"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, datetime):
        return _timestamp(value)
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _human_value(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class RunLogger:
    """Level-filtered structured run log with optional human-readable console output."""

    def __init__(
        self,
        *,
        level: str,
        run_id: str,
        clock: Callable[[], datetime] = _utc_now,
        stream: TextIO | None = sys.stderr,
    ) -> None:
        if level not in _LEVELS:
            raise ValueError(f"Unknown logging level: {level}")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a nonempty string")
        self.level = level
        self.run_id = run_id
        self._clock = clock
        self._stream = stream
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
            if self._stream is not None:
                details = {
                    key: value
                    for key, value in record.items()
                    if key not in {"timestamp", "sequence", "level", "run_id", "event"}
                }
                detail_text = " ".join(f"{key}={_human_value(value)}" for key, value in details.items())
                suffix = f" | {detail_text}" if detail_text else ""
                human_level = "WARN" if level == "WARNING" else level
                print(f"[{record['timestamp']}] [{human_level}] {event}{suffix}", file=self._stream, flush=True)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("stage name must be a nonempty string")
        self.info("stage_started", stage=name)
        try:
            yield
        except Exception as exc:
            self.error(
                "stage_failed",
                stage=name,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        else:
            self.info("stage_completed", stage=name)

    def write_jsonl(self, stage_root: str | Path) -> Path:
        """Write the current immutable snapshot inside an atomic release staging root."""

        destination = Path(stage_root) / "logs" / "pipeline.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        records = self.records
        payload = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for record in records
        )
        destination.write_text(payload, encoding="utf-8", newline="")
        return destination


__all__ = ["RunLogger"]
