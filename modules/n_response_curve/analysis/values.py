from __future__ import annotations

import math
from typing import Any, Mapping


_MISSING_TEXT_VALUES = frozenset({"na", "n/a", "nan", "none", "null", "not stated", "unresolved"})


def finite_number(value: Any) -> float | None:
    """Return a finite float while rejecting booleans and invalid values."""

    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def is_missing_text(value: object) -> bool:
    """Return whether a text value is empty or uses a supported missing marker."""

    if not isinstance(value, str):
        return False
    normalized = value.strip().casefold()
    return not normalized or normalized in _MISSING_TEXT_VALUES


def outcome_is_present(value: Any) -> bool:
    """Return whether an analysis outcome contains a supported value."""

    if finite_number(value) is not None:
        return True
    return isinstance(value, str) and not is_missing_text(value)


def record_uids(row: Mapping[str, Any]) -> set[str]:
    """Return the stable record IDs represented by one curve-level row."""

    values = row.get("record_uids")
    if isinstance(values, (list, tuple, set, frozenset)):
        return {str(value) for value in values if str(value)}
    value = row.get("record_uid")
    return {str(value)} if value else set()


__all__ = ["finite_number", "is_missing_text", "outcome_is_present", "record_uids"]
