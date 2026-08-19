"""Typed TOML readers shared by the standalone recipe configuration loaders.

``descriptive_statistics/config.py`` and ``grain_yield_response/config.py``
validate different configurations but read them the same way: pull a required
table, pull a typed scalar or list out of it with a message naming the exact
``[table].key`` that failed, and resolve a declared path while proving it stays
inside the project root. Each had grown its own copy of that reader set.

Each recipe raises its **own** ``RecipeConfigError``, and both are caught by
name — by their pipelines and by ``tests/test_grain_yield_response.py``. So the
error class is a parameter here rather than something this module owns: callers
pass ``error=RecipeConfigError``, keep their thin local wrapper, and the two
exception types stay distinct. The message text is identical to what each
recipe raised before, because those messages are what an operator reads when a
run refuses to start.

Booleans are rejected wherever a number is expected. In Python ``True`` is an
``int``, so a configuration reading ``figure_dpi = true`` would otherwise load
as 1 and render an unusable figure rather than failing.

There is deliberately no ``sha256_file`` here: :func:`
n_response_curve.data.provenance.sha256_file` is the one implementation, and
both recipes call it directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "boolean",
    "confined_path",
    "number",
    "string",
    "string_list",
    "table",
]


def table(
    data: Mapping[str, Any], name: str, *, error: type[Exception]
) -> Mapping[str, Any]:
    """Return the required ``[name]`` table."""

    value = data.get(name)
    if not isinstance(value, Mapping):
        raise error(f"Missing or invalid [{name}] table")
    return value


def string(
    section: Mapping[str, Any], key: str, where: str, *, error: type[Exception]
) -> str:
    """Return a required nonempty string, stripped."""

    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise error(f"{where}.{key} must be a non-empty string")
    return value.strip()


def boolean(
    section: Mapping[str, Any], key: str, where: str, *, error: type[Exception]
) -> bool:
    """Return a required Boolean, rejecting the integers that would coerce."""

    value = section.get(key)
    if not isinstance(value, bool):
        raise error(f"{where}.{key} must be a Boolean")
    return value


def number(
    section: Mapping[str, Any],
    key: str,
    where: str,
    *,
    error: type[Exception],
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    """Return a required number inside ``[minimum, maximum]``."""

    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error(f"{where}.{key} must be a number")
    result = float(value)
    if result < minimum:
        raise error(f"{where}.{key} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise error(f"{where}.{key} must be at most {maximum}")
    return result


def string_list(
    section: Mapping[str, Any],
    key: str,
    where: str,
    *,
    error: type[Exception],
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Return a required list of unique nonempty strings, each stripped.

    ``allow_empty`` governs the *list*, not its members: an empty list is a
    meaningful configuration for some keys (no factors declared) and a refusal
    for others (no sources profiled). Members are never allowed to be blank.
    """

    value = section.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise error(f"{where}.{key} must be a list of non-empty strings")
    values = tuple(item.strip() for item in value)
    if not values and not allow_empty:
        raise error(f"{where}.{key} must not be empty")
    if len(values) != len(set(values)):
        raise error(f"{where}.{key} must not contain duplicates")
    return values


def confined_path(
    root: Path, raw: str, where: str, *, error: type[Exception]
) -> Path:
    """Resolve a declared path and prove it stays inside *root*.

    A relative path is taken against *root* rather than the working directory,
    so a recipe launched from anywhere resolves the same files.
    """

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise error(f"{where} must stay inside the project root") from exc
    return resolved
