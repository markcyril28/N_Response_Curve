from __future__ import annotations

from dataclasses import dataclass
import statistics
from typing import Any, Iterable, Mapping, Sequence

from .factor_catalog import FactorCatalogEntry, factor_value
from .values import finite_number


@dataclass(frozen=True)
class DescriptiveComparison:
    """A non-inferential coverage and outcome summary for one explicit grouping."""

    comparison_type: str
    outcome_name: str
    group_label: str
    row_count: int
    outcome_count: int
    outcome_missing_count: int
    outcome_mean: float | None
    outcome_median: float | None
    outcome_min: float | None
    outcome_max: float | None
    reason_codes: tuple[str, ...]


def _summary(
    comparison_type: str,
    outcome_name: str,
    group_label: str,
    rows: Sequence[Mapping[str, Any]],
) -> DescriptiveComparison:
    values = [finite_number(row.get(outcome_name)) for row in rows]
    observed = [value for value in values if value is not None]
    return DescriptiveComparison(
        comparison_type=comparison_type,
        outcome_name=outcome_name,
        group_label=group_label,
        row_count=len(rows),
        outcome_count=len(observed),
        outcome_missing_count=len(rows) - len(observed),
        outcome_mean=float(statistics.fmean(observed)) if observed else None,
        outcome_median=float(statistics.median(observed)) if observed else None,
        outcome_min=float(min(observed)) if observed else None,
        outcome_max=float(max(observed)) if observed else None,
        reason_codes=("CROSS_SOURCE_COMPARABILITY_NOT_ASSUMED", "DESCRIPTIVE_ONLY"),
    )


def _grouped_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    comparison_type: str,
    outcome_name: str,
    labels: Iterable[tuple[str, Mapping[str, Any]]],
) -> list[DescriptiveComparison]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for label, row in labels:
        groups.setdefault(label, []).append(row)
    return [
        _summary(comparison_type, outcome_name, label, group_rows)
        for label, group_rows in sorted(groups.items())
    ]


def build_descriptive_comparisons(
    curve_rows: Iterable[Mapping[str, Any]],
    *,
    outcome_name: str,
    factor_catalog: Sequence[FactorCatalogEntry],
) -> tuple[DescriptiveComparison, ...]:
    """Summarize coverage/missingness by source, recommendation, and explicit factor strata."""

    if not isinstance(outcome_name, str) or not outcome_name:
        raise ValueError("outcome_name must be a nonempty string")
    rows = tuple(dict(row) for row in curve_rows)
    summaries: list[DescriptiveComparison] = []
    summaries.extend(
        _grouped_summaries(
            rows,
            comparison_type="source_family",
            outcome_name=outcome_name,
            labels=((str(row.get("source_name") or "<missing>"), row) for row in rows),
        )
    )
    summaries.extend(
        _grouped_summaries(
            rows,
            comparison_type="recommendation_class",
            outcome_name=outcome_name,
            labels=((str(row.get("treatment_text_class") or "<missing>"), row) for row in rows),
        )
    )
    seen_factors: set[str] = set()
    for entry in factor_catalog:
        if entry.factor_name in seen_factors:
            raise ValueError("Factor catalog entries must have unique factor names")
        seen_factors.add(entry.factor_name)
        summaries.extend(
            _grouped_summaries(
                rows,
                comparison_type=f"factor_stratum:{entry.factor_name}",
                outcome_name=outcome_name,
                labels=(
                    (str(factor_value(row, entry.factor_name)) if factor_value(row, entry.factor_name) is not None else "<missing>", row)
                    for row in rows
                ),
            )
        )
    return tuple(sorted(summaries, key=lambda summary: (summary.comparison_type, summary.group_label)))


__all__ = ["DescriptiveComparison", "build_descriptive_comparisons"]
