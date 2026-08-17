"""Categorical and identifier-like column profiling.

Emits two tables. ``categorical_summary`` carries one row per column classified
``categorical`` or ``identifier`` and reports cardinality, mode, entropy and the
governance disclosure of how many levels were withheld. ``categorical_levels``
carries the level frequency table, and is built for ``categorical`` columns
only: an ``identifier`` column has more distinct values than
``[structure].maximum_categorical_cardinality``, so tabulating its levels would
publish something close to a row-identity index of a restricted dataset.

Both tables are derived from a single per-column tally so they cannot disagree
about a count, and every column is addressed by ``raw_column_id`` / physical
position because the core extract carries seven duplicated and sixteen blank
header names.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Iterator

import numpy as np
import pandas as pd

from . import contracts
from .config import DescriptiveStatisticsConfig
from .sources import ColumnSpec, LoadedSources, ProfiledSource


SUMMARY_TABLE = "categorical_summary"
LEVELS_TABLE = "categorical_levels"

# Columns classified into either of these kinds are summarized; only
# ``categorical`` is level-tabulated. ``empty`` and ``suppressed`` carry no
# nonblank cell and ``numeric`` is profiled by the numeric module.
_SUMMARIZED_KINDS = ("categorical", "identifier")

_SUMMARY_INTEGER_COLUMNS = (
    "position",
    "nonblank_count",
    "blank_count",
    "distinct_count",
    "mode_count",
    "singleton_level_count",
    "levels_reported",
    "levels_withheld_below_threshold",
)
_LEVELS_INTEGER_COLUMNS = ("position", "level_rank", "count")


@dataclass(frozen=True)
class _ColumnLevels:
    """The complete level tally of one column, ordered for reporting.

    ``ordered`` is sorted by descending count then ascending value, which is
    both the reporting order of ``categorical_levels`` and — at element zero —
    the tie-broken mode: among the levels sharing the maximum count, the
    lexicographically smallest sorts first, so the mode is deterministic
    without a second pass.
    """

    spec: ColumnSpec
    ordered: tuple[tuple[str, int], ...]

    @property
    def total(self) -> int:
        return self.spec.nonblank_count

    @property
    def distinct_count(self) -> int:
        return len(self.ordered)

    @property
    def singleton_level_count(self) -> int:
        return sum(1 for _, count in self.ordered if count == 1)

    @property
    def mode(self) -> tuple[str, int]:
        return self.ordered[0]

    @property
    def shannon_entropy_bits(self) -> float:
        """Entropy over *every* level, including those withheld from the level table.

        Withheld rare levels still shape the dispersion of the column, so
        excluding them would understate entropy and misreport how concentrated
        the recorded values are. The aggregate discloses no individual level.
        """

        total = self.total
        entropy = 0.0
        for _, count in self.ordered:
            share = count / total
            entropy -= share * math.log2(share)
        return entropy

    @property
    def normalized_entropy(self) -> float:
        # log2(1) = 0 for a constant column, and a column with no level has no
        # scale at all; both are undefined rather than zero.
        if self.distinct_count < 2:
            return math.nan
        return self.shannon_entropy_bits / math.log2(self.distinct_count)

    def qualifying(self, minimum_level_count: int) -> tuple[tuple[str, int], ...]:
        """Levels the reporting threshold permits, still in reporting order."""

        return tuple(
            (value, count) for value, count in self.ordered if count >= minimum_level_count
        )


def _tally(source: ProfiledSource, spec: ColumnSpec) -> _ColumnLevels:
    """Count the stripped nonblank strings of one physical column.

    Levels are stripped, so "IRRI " and "IRRI" are one level. Blankness is
    tested on the stripped form, which is what ``ColumnSpec.nonblank_count``
    uses, so the level counts sum to exactly that field.

    Note this makes ``distinct_count`` a stripped-basis count, whereas
    ``ColumnSpec.distinct_nonblank_count`` is tallied on the raw stored strings.
    The stripped basis is required here: entropy, ``singleton_level_count`` and
    ``levels_withheld_below_threshold`` are all defined over the level set that
    ``categorical_levels`` reports, and mixing the two universes would divide a
    stripped-basis entropy by a raw-basis log2.
    """

    counts: Counter[str] = Counter()
    for value in source.text[spec.raw_column_id].to_numpy():
        level = str(value).strip()
        if level:
            counts[level] += 1
    ordered = tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    return _ColumnLevels(spec=spec, ordered=ordered)


def _column_keys(spec: ColumnSpec) -> dict[str, Any]:
    return {
        "source_name": spec.source_name,
        "data_classification": spec.data_classification,
        "position": spec.position,
        "raw_column_id": spec.raw_column_id,
        "header_label": spec.header_label,
    }


def _profiled_columns(source: ProfiledSource) -> Iterator[ColumnSpec]:
    """Summarizable columns of one source, in physical-position order."""

    for spec in source.columns_of_kind(*_SUMMARIZED_KINDS):
        # Belt and braces: a suppressed column is already classified
        # ``suppressed`` and blanked at load, so it cannot reach here. The guard
        # stays because the cost of a future classification change leaking a
        # personal or location identifier is not recoverable.
        if spec.suppressed:
            continue
        yield spec


def _summary_row(
    tally: _ColumnLevels,
    *,
    minimum_level_count: int,
    levels_reported: int,
) -> dict[str, Any]:
    spec = tally.spec
    mode_value, mode_count = tally.mode
    return {
        **_column_keys(spec),
        "nonblank_count": spec.nonblank_count,
        "blank_count": spec.blank_count,
        "distinct_count": tally.distinct_count,
        "is_identifier_like": spec.value_kind == "identifier",
        # A modal level below the reporting threshold is a rare level, and the
        # threshold does not stop applying because the level happens to be the
        # most frequent one: ph_combined_nopt_rcm's rcm_reference is 718 levels
        # of count 1, so its mode is a single restricted record's reference.
        # mode_count and mode_share stay — they are aggregates and disclose the
        # withholding rather than hiding it.
        "mode_value": mode_value if mode_count >= minimum_level_count else np.nan,
        "mode_count": mode_count,
        "mode_share": mode_count / tally.total,
        "singleton_level_count": tally.singleton_level_count,
        "shannon_entropy_bits": tally.shannon_entropy_bits,
        "normalized_entropy": tally.normalized_entropy,
        "levels_reported": levels_reported,
        # Counted over every level, including the levels of an identifier-like
        # column that is never tabulated at all: the reader must be able to see
        # that rare levels exist and how many, without seeing them. The residual
        # distinct_count - levels_reported - levels_withheld_below_threshold is
        # what the identifier rule and the maximum_levels_reported cap removed.
        "levels_withheld_below_threshold": tally.distinct_count
        - len(tally.qualifying(minimum_level_count)),
    }


def _level_rows(tally: _ColumnLevels, reported: tuple[tuple[str, int], ...]) -> list[dict[str, Any]]:
    keys = _column_keys(tally.spec)
    total = tally.total
    rows: list[dict[str, Any]] = []
    running = 0
    for rank, (value, count) in enumerate(reported, start=1):
        running += count
        rows.append(
            {
                **keys,
                "level_rank": rank,
                "level_value": value,
                "count": count,
                "share": count / total,
                # Accumulated on the integer counts, not on the float shares, so
                # a column whose levels all clear the threshold closes on
                # exactly 1.0 instead of 0.9999999999999999.
                "cumulative_share": running / total,
            }
        )
    return rows


def _frame(name: str, rows: list[dict[str, Any]], integer_columns: tuple[str, ...]) -> pd.DataFrame:
    if not rows:
        return contracts.empty_table(name)
    frame = pd.DataFrame.from_records(rows)
    # These counts are always computable — _classify_value_kind returns "empty"
    # before a zero-nonblank column can be summarized — so pin them to int64
    # rather than letting a single NA elsewhere float the whole column and
    # render "1490.0" in the CSV.
    for column in integer_columns:
        frame[column] = frame[column].astype("int64")
    return contracts.conform_table(name, frame)


def analyze_categorical(
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> dict[str, pd.DataFrame]:
    """Profile every categorical and identifier-like column of every source."""

    summary_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []

    for source in loaded.sources:
        for spec in _profiled_columns(source):
            if spec.nonblank_count == 0:
                continue  # unreachable; a blank column classifies as "empty"
            tally = _tally(source, spec)
            qualifying = tally.qualifying(config.minimum_level_count)
            reported: tuple[tuple[str, int], ...] = ()
            if spec.value_kind == "categorical":
                reported = qualifying[: config.maximum_levels_reported]
                level_rows.extend(_level_rows(tally, reported))
            summary_rows.append(
                _summary_row(
                    tally,
                    minimum_level_count=config.minimum_level_count,
                    levels_reported=len(reported),
                )
            )

    summary = _frame(SUMMARY_TABLE, summary_rows, _SUMMARY_INTEGER_COLUMNS)
    if not summary.empty:
        summary["is_identifier_like"] = summary["is_identifier_like"].astype(bool)
    return {
        SUMMARY_TABLE: summary,
        LEVELS_TABLE: _frame(LEVELS_TABLE, level_rows, _LEVELS_INTEGER_COLUMNS),
    }
