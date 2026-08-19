"""Calendar-axis helpers shared by the by-planting-year reporting recipes.

Every recipe that draws a quantity against planting year answers the same three
questions, and each had answered them for itself: where does a drawn line have
to break, which decade band does a record belong in, and how wide is that band.

The one that carries real meaning is :func:`contiguous_runs`. These sources
observe most calendar years not at all, and a line drawn straight across a gap
reads as an observed change over years that were never observed. Splitting the
points into runs with no gap, and drawing one line per run, is what keeps the
figure honest. It is separate from any era or colour boundary: IR8's 1984 gap
sits *inside* a single unchanged design era, so the era must not break there but
the line must.

Season vocabulary is deliberately absent. ``core_trial_data`` records dry/wet,
LTCCE records DS/EWS/LWS, and ``ph_combined_nopt_rcm`` records dry/wet with an
``unrecorded`` fallback; those are three different recorded vocabularies, not
one vocabulary spelled three ways, and each recipe keeps its own normalizer.
Only :data:`SEASON_LABELS`, the reader-facing spelling of the two-season
vocabulary, is shared.

This module is importable by the standalone reporting recipes and is not on the
release path.
"""

from __future__ import annotations

from typing import Callable, Sequence, TypeVar

__all__ = [
    "SEASON_LABELS",
    "contiguous_runs",
    "decade_label",
    "decade_spans",
    "year_bounds",
]

_Item = TypeVar("_Item")


# Reader-facing spelling of the recorded two-season vocabulary. The recorded
# value stays the key everywhere it does identifier work.
SEASON_LABELS = {"dry": "dry season", "wet": "wet season"}


def contiguous_runs(
    items: Sequence[_Item],
    *,
    year: Callable[[_Item], int],
    presorted: bool = True,
) -> list[list[_Item]]:
    """Split *items* into runs of consecutive calendar years.

    ``year`` reads the calendar year off one item, so the same grouping serves a
    bare list of years, a list of trend points, and a list of annual summaries.

    *items* are taken in the order given, because most callers have already
    ordered them for drawing and re-sorting would silently reorder ties. Pass
    ``presorted=False`` to sort by year first.
    """

    ordered = list(items) if presorted else sorted(items, key=year)
    runs: list[list[_Item]] = []
    for item in ordered:
        if runs and year(item) == year(runs[-1][-1]) + 1:
            runs[-1].append(item)
        else:
            runs.append([item])
    return runs


def decade_label(year: int) -> str:
    """Return the decade band *year* falls in, as ``"1990s"``."""

    return f"{(year // 10) * 10}s"


def decade_spans(decades: Sequence[str]) -> tuple[tuple[str, int, int], ...]:
    """``(label, first year, last year)`` for each decade band, in the order given.

    Callers that hold decade labels in a mapping should pass ``sorted(mapping)``
    so the bands come out oldest first.
    """

    return tuple((decade, int(decade.rstrip("s")), int(decade.rstrip("s")) + 9) for decade in decades)


def year_bounds(label: str) -> tuple[int, int] | None:
    """Parse a recorded planting year, which may be a range, into ``(first, last)``.

    Some sources record a series against a span rather than a year (``1991-1994``,
    ``2003-04``). They are real observations and belong in a decade band; they
    have no place on a year axis. Returning both ends lets a caller do both, and
    lets it fail loudly if a span ever crosses a decade boundary, where the band
    placement would be ambiguous.

    Returns ``None`` for anything that is not a four-digit year optionally
    followed by a two- or four-digit end that is not earlier than the start.
    """

    text = label.strip()
    if not text:
        return None
    head, _, tail = text.partition("-")
    head = head.strip()
    tail = tail.strip()
    if not (len(head) == 4 and head.isdigit()):
        return None
    first = int(head)
    if not tail:
        return first, first
    if len(tail) == 2 and tail.isdigit():
        last = first - first % 100 + int(tail)
    elif len(tail) == 4 and tail.isdigit():
        last = int(tail)
    else:
        return None
    if last < first:
        return None
    return first, last
