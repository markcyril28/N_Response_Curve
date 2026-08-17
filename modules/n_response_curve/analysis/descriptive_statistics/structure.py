"""Structural profile of every registered dataset: shape, headers, missingness.

This module answers question 1 of the recipe: how many physical columns and
logical data rows each source carries, which headers are blank or duplicated,
which columns are wholly empty, and how completeness is distributed across rows.

Everything here is read off the ``ColumnSpec`` tuple that ``sources`` already
built by physical position, plus the aligned ``ProfiledSource.text`` frame. No
statistic is keyed on header text: the core literature extract carries seven
duplicated header names (one of which is the empty string, repeated sixteen
times) so header text does not identify a column in that source at all.

Suppressed columns keep their structural row everywhere in this group. That is
deliberate and is what makes the column arithmetic reconcile — their cells were
blanked at load time, so they contribute zero to every populated-cell total
while still being counted in ``physical_column_count``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterator

import numpy as np
import pandas as pd

from . import contracts
from .config import DescriptiveStatisticsConfig
from .sources import ColumnSpec, LoadedSources, ProfiledSource


# ``header_status`` values that make a column an anomaly worth listing. "named"
# is the only status that is not an anomaly, but enumerate the positives rather
# than negating, so a future status is excluded until someone decides otherwise.
_ANOMALOUS_HEADER_STATUS: dict[str, str] = {
    "blank": "blank_header",
    "duplicated": "duplicated_header",
}


@dataclass(frozen=True)
class _ColumnKey:
    """The five identity fields every per-column structure table repeats."""

    source_name: str
    data_classification: str
    position: int
    raw_column_id: str
    header_label: str

    @classmethod
    def of(cls, spec: ColumnSpec) -> _ColumnKey:
        return cls(
            source_name=spec.source_name,
            data_classification=spec.data_classification,
            position=spec.position,
            raw_column_id=spec.raw_column_id,
            header_label=spec.header_label,
        )

    def as_row(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "data_classification": self.data_classification,
            "position": self.position,
            "raw_column_id": self.raw_column_id,
            "header_label": self.header_label,
        }


@dataclass(frozen=True)
class _CompletenessStatistic:
    """One reported point of the per-row completeness distribution.

    ``quantile is None`` marks the mean, which is the only reported statistic
    that is not an order statistic of the same vector.
    """

    name: str
    quantile: float | None

    def evaluate(self, per_row_counts: np.ndarray) -> float:
        if per_row_counts.size == 0:
            return math.nan
        if self.quantile is None:
            return float(np.mean(per_row_counts))
        return float(np.quantile(per_row_counts, self.quantile))


# Ordered as reported. Reporting minimum and maximum as the 0.0 and 1.0
# quantiles keeps the whole distribution on one interpolation convention.
_COMPLETENESS_STATISTICS: tuple[_CompletenessStatistic, ...] = (
    _CompletenessStatistic("minimum", 0.0),
    _CompletenessStatistic("p05", 0.05),
    _CompletenessStatistic("q1", 0.25),
    _CompletenessStatistic("median", 0.50),
    _CompletenessStatistic("mean", None),
    _CompletenessStatistic("q3", 0.75),
    _CompletenessStatistic("p95", 0.95),
    _CompletenessStatistic("maximum", 1.0),
)


def _ratio(numerator: float, denominator: float) -> float:
    """A share, or NaN when the denominator does not exist.

    Zero is a real fill rate and NaN is "there was nothing to divide by"; a
    reader must be able to tell an empty column from an absent dataset.
    """

    return numerator / denominator if denominator else math.nan


def _row_nonblank_counts(source: ProfiledSource) -> np.ndarray:
    """Per input row, how many of its physical columns hold a nonblank value.

    Whitespace-only is blank, matching ``_build_column_specs`` exactly, so the
    row totals here sum to the same ``populated_cells`` the column totals give.
    """

    if source.data_row_count == 0 or source.physical_column_count == 0:
        return np.zeros(source.data_row_count, dtype=np.int64)
    # ``text`` is an object frame of the exact stored strings; compare on the
    # stripped view without mutating it.
    stripped = source.text.apply(lambda column: column.astype(str).str.strip())
    return stripped.ne("").to_numpy().sum(axis=1).astype(np.int64)


def _count_header_status(source: ProfiledSource, status: str) -> int:
    return sum(1 for spec in source.columns if spec.header_status == status)


def _count_value_kind(source: ProfiledSource, kind: str) -> int:
    return len(source.columns_of_kind(kind))


def _source_inventory_row(source: ProfiledSource) -> dict[str, Any]:
    total_cells = source.data_row_count * source.physical_column_count
    populated_cells = sum(spec.nonblank_count for spec in source.columns)
    return {
        "source_name": source.source_name,
        "data_classification": source.data_classification,
        "source_type": source.source_type,
        "source_family": source.source_family,
        "country_code": source.country_code,
        "shape_adapter_version": source.shape_adapter_version,
        "source_encoding": source.source_encoding,
        "representation_basis": source.representation_basis,
        "representation_basis_status": source.representation_basis_status,
        "restricted_access_status": source.restricted_access_status,
        "source_sha256": source.source_sha256,
        "source_relative_path": source.source_relative_path,
        "physical_column_count": source.physical_column_count,
        "data_row_count": source.data_row_count,
        "blank_row_count": source.blank_row_count,
        "named_header_count": _count_header_status(source, "named"),
        "blank_header_count": _count_header_status(source, "blank"),
        # A blank header is classified as blank even when the empty string
        # repeats, so this counts only duplicated *named* header positions.
        "duplicated_header_count": _count_header_status(source, "duplicated"),
        "numeric_column_count": _count_value_kind(source, "numeric"),
        "categorical_column_count": _count_value_kind(source, "categorical"),
        "identifier_column_count": _count_value_kind(source, "identifier"),
        # "empty" and "suppressed" are disjoint value kinds, so a suppressed
        # column is not also counted as empty even though it holds no values.
        # The five kind counts therefore partition physical_column_count.
        "empty_column_count": _count_value_kind(source, "empty"),
        "suppressed_column_count": sum(1 for spec in source.columns if spec.suppressed),
        "total_cells": total_cells,
        "populated_cells": populated_cells,
        "overall_fill_rate": _ratio(populated_cells, total_cells),
    }


def _column_inventory_row(spec: ColumnSpec) -> dict[str, Any]:
    row = _ColumnKey.of(spec).as_row()
    row.update(
        {
            "header_raw": spec.header_raw,
            "header_status": spec.header_status,
            "duplicate_group_size": spec.duplicate_group_size,
            "value_kind": spec.value_kind,
            "suppressed": spec.suppressed,
            "suppression_reason": spec.suppression_reason,
            "nonblank_count": spec.nonblank_count,
            "blank_count": spec.blank_count,
            "fill_rate": spec.fill_rate,
            "distinct_nonblank_count": spec.distinct_nonblank_count,
            "numeric_parse_rate": spec.numeric_parse_rate,
            # A suppressed column arrives with an empty example tuple, so it
            # renders as an empty cell here without a special case.
            "example_values": " | ".join(spec.example_values),
        }
    )
    return row


def _missingness_row(spec: ColumnSpec, *, data_row_count: int) -> dict[str, Any]:
    row = _ColumnKey.of(spec).as_row()
    row.update(
        {
            "value_kind": spec.value_kind,
            "blank_count": spec.blank_count,
            "nonblank_count": spec.nonblank_count,
            "missing_rate": _ratio(spec.blank_count, data_row_count),
            "fill_rate": spec.fill_rate,
            "is_wholly_empty": spec.nonblank_count == 0,
        }
    )
    return row


def _header_anomaly_row(spec: ColumnSpec, anomaly: str) -> dict[str, Any]:
    row = _ColumnKey.of(spec).as_row()
    row.update(
        {
            "header_raw": spec.header_raw,
            "anomaly": anomaly,
            "duplicate_group_size": spec.duplicate_group_size,
            # For a blank header this lists every blank position, because the
            # empty string is one header-name group like any other; that is the
            # honest reading of the shape and is what makes the blank tail of
            # the core extract visible as a contiguous run.
            "duplicate_positions": ";".join(
                str(position) for position in spec.duplicate_positions
            ),
        }
    )
    return row


def _row_completeness_rows(source: ProfiledSource) -> Iterator[dict[str, Any]]:
    per_row = _row_nonblank_counts(source)
    for statistic in _COMPLETENESS_STATISTICS:
        populated_columns = statistic.evaluate(per_row)
        yield {
            "source_name": source.source_name,
            "data_classification": source.data_classification,
            "statistic": statistic.name,
            "populated_columns": populated_columns,
            # Divided by the physical width including suppressed columns, so a
            # source whose identifiers were blanked reads as less complete than
            # its file is. That is intended: the share describes the data this
            # profile was permitted to see.
            "populated_share": _ratio(
                populated_columns, source.physical_column_count
            ),
        }


def _frame(name: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    # Even the empty frame goes through conform_table, so the schema of a table
    # with nothing to report is checked by the same gate as a populated one.
    built = contracts.empty_table(name) if not rows else pd.DataFrame.from_records(rows)
    return contracts.conform_table(name, built)


def analyze_structure(
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> dict[str, pd.DataFrame]:
    """Build the five structure tables for every profiled source.

    ``config`` is accepted for entry-point uniformity across the analysis
    modules; the only recipe settings this group depends on — the numeric parse
    threshold, the categorical cardinality ceiling, and the example-value count
    — were already applied when ``sources`` classified each column, and are not
    re-applied here so the inventory cannot disagree with what was loaded.
    """

    del config

    inventory: list[dict[str, Any]] = []
    columns: list[dict[str, Any]] = []
    missingness: list[tuple[int, ColumnSpec, dict[str, Any]]] = []
    completeness: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []

    for source_order, source in enumerate(loaded.sources):
        inventory.append(_source_inventory_row(source))
        completeness.extend(_row_completeness_rows(source))
        for spec in source.columns:
            columns.append(_column_inventory_row(spec))
            missingness.append(
                (
                    source_order,
                    spec,
                    _missingness_row(spec, data_row_count=source.data_row_count),
                )
            )
            anomaly = _ANOMALOUS_HEADER_STATUS.get(spec.header_status)
            if anomaly is not None:
                anomalies.append(_header_anomaly_row(spec, anomaly))

    # Worst-first, resolving ties by the configured source order and then by
    # physical position. Sorted in Python rather than by pandas so a NaN
    # missing_rate (a zero-row source) sorts last deterministically instead of
    # wherever the frame sort happens to place it.
    missingness.sort(
        key=lambda item: (
            -item[2]["missing_rate"]
            if isinstance(item[2]["missing_rate"], float)
            and not math.isnan(item[2]["missing_rate"])
            else math.inf,
            item[0],
            item[1].position,
        )
    )

    return {
        "source_inventory": _frame("source_inventory", inventory),
        "column_inventory": _frame("column_inventory", columns),
        "missingness_profile": _frame(
            "missingness_profile", [row for _, _, row in missingness]
        ),
        "row_completeness": _frame("row_completeness", completeness),
        "header_anomalies": _frame("header_anomalies", anomalies),
    }
