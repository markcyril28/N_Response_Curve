"""Descriptive statistics for every numeric-parsable physical column.

Three tables are produced. ``numeric_summary`` carries one row per numeric
column with counts that reconcile exactly against the physical row count;
``numeric_outlier_audit`` reports Tukey fences at each configured IQR
multiplier; ``numeric_distribution_bins`` carries equal-width histograms for the
agronomically bound columns only, which keeps that table bounded instead of
emitting 269 columns of bins.

Every column is addressed by ``ColumnSpec.raw_column_id`` and reported with its
physical position, never by header text: the core extract has seven duplicated
and sixteen blank header names, so header text is not a key. Rows are ordered by
source and then by physical position and are never ordered by a statistic —
this recipe describes what was recorded and must not read as a ranking.

Suppressed columns are classified ``value_kind == "suppressed"`` upstream and so
never enter these tables; the guard in ``_profile_column`` is a second lock, not
a substitute for the first.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd

from .config import DescriptiveStatisticsConfig
from .contracts import conform_table, empty_table
from .sources import (
    ColumnBinding,
    ColumnSpec,
    LoadedSources,
    ProfiledSource,
    SourceProfileError,
)


# The declared quantile columns of ``numeric_summary`` and the probability each
# one reports. ``load_recipe_config`` already requires every one of these
# probabilities to appear in ``[numeric].quantiles``; the lookup below is written
# to degrade to NA rather than raise, because a declared statistic that cannot be
# computed must still occupy its column.
_DECLARED_QUANTILES: tuple[tuple[float, str], ...] = (
    (0.01, "p01"),
    (0.05, "p05"),
    (0.25, "q1"),
    (0.50, "median"),
    (0.75, "q3"),
    (0.95, "p95"),
    (0.99, "p99"),
)

# Mathematical minima, applied on top of ``[numeric].minimum_observations``.
_MINIMUM_STD_DEV_OBSERVATIONS = 2  # sample sd is undefined at n = 1
_MINIMUM_SKEWNESS_OBSERVATIONS = 3
_MINIMUM_KURTOSIS_OBSERVATIONS = 4  # the bias-corrected estimator needs n >= 4
_MINIMUM_FENCE_OBSERVATIONS = 4  # a box built from fewer points is not a box

# Only the agronomically bound quantities are binned. These are the labels of the
# single-column bindings on ``AgronomicBinding``; context, grouping and series
# bindings are categorical and belong to the categorical profile.
_BINNED_BINDING_LABELS: tuple[str, ...] = (
    "nitrogen_rate",
    "yield_t_ha",
    "yield_kg_ha",
    "zero_n_yield_t_ha",
)


@dataclass(frozen=True)
class _ColumnStatistics:
    """Everything both numeric tables need about one column, computed once.

    ``q1``/``q3``/``iqr`` are shared rather than recomputed per table: the outlier
    audit's fences must reconcile against the summary row exactly
    (``upper_fence == q3 + k * iqr``), and two independent ``.quantile()`` calls
    can disagree in the last ulp.
    """

    spec: ColumnSpec
    values: np.ndarray  # the finite parsed values, in stored row order
    count: int
    missing_count: int
    unparsed_count: int
    quantiles: dict[str, float]
    iqr: float

    @property
    def q1(self) -> float:
        return self.quantiles["q1"]

    @property
    def q3(self) -> float:
        return self.quantiles["q3"]


def _column_keys(spec: ColumnSpec) -> dict[str, object]:
    """The five identity columns every per-column table declares."""

    return {
        "source_name": spec.source_name,
        "data_classification": spec.data_classification,
        "position": spec.position,
        "raw_column_id": spec.raw_column_id,
        "header_label": spec.header_label,
    }


def _quantile_plan(
    config: DescriptiveStatisticsConfig,
) -> tuple[tuple[str, float | None], ...]:
    """Bind each declared quantile column to a configured probability."""

    plan: list[tuple[str, float | None]] = []
    for probability, column in _DECLARED_QUANTILES:
        configured = next(
            (
                value
                for value in config.quantiles
                if math.isclose(value, probability, rel_tol=0.0, abs_tol=1e-12)
            ),
            None,
        )
        plan.append((column, configured))
    return tuple(plan)


def _profile_column(
    source: ProfiledSource,
    spec: ColumnSpec,
    config: DescriptiveStatisticsConfig,
    quantile_plan: Sequence[tuple[str, float | None]],
) -> _ColumnStatistics:
    """Reconcile one numeric column and compute its shared quantile geometry."""

    if spec.suppressed:  # unreachable while suppression forces value_kind
        raise SourceProfileError(
            f"{source.source_name}: refusing to profile suppressed column "
            f"{spec.raw_column_id} (position {spec.position})"
        )

    values = source.numeric[spec.raw_column_id].to_numpy(dtype=float, copy=False)
    finite = values[np.isfinite(values)]
    count = int(finite.size)
    # The real invariant: the numeric matrix is written only where a nonblank cell
    # parsed finite, so a parsed count exceeding the nonblank count means the
    # matrix and the column spec have fallen out of alignment.
    if count > spec.nonblank_count:
        raise SourceProfileError(
            f"{source.source_name}: column {spec.raw_column_id} parsed {count} values "
            f"from {spec.nonblank_count} nonblank cells"
        )
    unparsed_count = spec.nonblank_count - count
    missing_count = spec.blank_count
    # Stated in the contract for this table and re-checked here so a table that
    # ships can be reconciled cell-for-cell against the physical extract.
    if count + missing_count + unparsed_count != source.data_row_count:
        raise SourceProfileError(
            f"{source.source_name}: column {spec.raw_column_id} counts "
            f"{count}+{missing_count}+{unparsed_count} do not reconcile against "
            f"{source.data_row_count} data rows"
        )

    series = pd.Series(finite, dtype=float)
    quantiles: dict[str, float] = {}
    for column, probability in quantile_plan:
        if probability is None or count < config.minimum_numeric_observations:
            quantiles[column] = math.nan
            continue
        # Linear interpolation (Hyndman-Fan type 7), the pandas default and the
        # same convention the sibling grain-yield recipe reports through
        # np.percentile, so quantiles are comparable across the two profiles.
        quantiles[column] = float(series.quantile(probability))
    q1 = quantiles["q1"]
    q3 = quantiles["q3"]
    iqr = q3 - q1 if math.isfinite(q1) and math.isfinite(q3) else math.nan
    return _ColumnStatistics(
        spec=spec,
        values=finite,
        count=count,
        missing_count=missing_count,
        unparsed_count=unparsed_count,
        quantiles=quantiles,
        iqr=iqr,
    )


def _summary_row(
    statistics: _ColumnStatistics, config: DescriptiveStatisticsConfig
) -> dict[str, object]:
    """One ``numeric_summary`` row.

    Counts, extremes and the sum are exact reports of recorded cells and are
    always emitted. The dispersion and shape statistics are estimates, and are
    withheld as NA below ``[numeric].minimum_observations`` so that a reader can
    distinguish "too few observations to characterize" from "column absent" —
    the row is emitted either way.
    """

    finite = statistics.values
    count = statistics.count
    series = pd.Series(finite, dtype=float)
    stable = count >= config.minimum_numeric_observations

    if count:
        mean = float(finite.mean())
        minimum = float(finite.min())
        maximum = float(finite.max())
        total = float(finite.sum())
        span = maximum - minimum
    else:  # a numeric column always parses at least one cell; guarded anyway
        mean = minimum = maximum = total = span = math.nan

    std_dev = (
        float(series.std(ddof=1))
        if stable and count >= _MINIMUM_STD_DEV_OBSERVATIONS
        else math.nan
    )
    # CV is scale-free only around a nonzero mean; on a column centered on zero
    # (or containing a zero-N ladder that averages near it) the ratio explodes and
    # says nothing about spread.
    coefficient_of_variation = (
        std_dev / mean if math.isfinite(std_dev) and mean != 0.0 else math.nan
    )
    median = statistics.quantiles["median"]
    median_absolute_deviation = (
        float(np.median(np.abs(finite - median)))
        if stable and math.isfinite(median)
        else math.nan
    )
    skewness = (
        float(series.skew())
        if stable and count >= _MINIMUM_SKEWNESS_OBSERVATIONS
        else math.nan
    )
    excess_kurtosis = (
        float(series.kurt())
        if stable and count >= _MINIMUM_KURTOSIS_OBSERVATIONS
        else math.nan
    )

    row: dict[str, object] = _column_keys(statistics.spec)
    row.update(
        {
            "count": count,
            "missing_count": statistics.missing_count,
            "unparsed_count": statistics.unparsed_count,
            "mean": mean,
            "std_dev": std_dev,
            "coefficient_of_variation": coefficient_of_variation,
            "minimum": minimum,
        }
    )
    row.update({column: statistics.quantiles[column] for _, column in _DECLARED_QUANTILES})
    row.update(
        {
            "maximum": maximum,
            "range": span,
            "iqr": statistics.iqr,
            "median_absolute_deviation": median_absolute_deviation,
            "skewness": skewness,
            "excess_kurtosis": excess_kurtosis,
            "zero_count": int(np.count_nonzero(finite == 0.0)),
            "negative_count": int(np.count_nonzero(finite < 0.0)),
            # Distinct *parsed* values, which is deliberately not the structure
            # profile's distinct_nonblank_count: "0" and "0.0" are two stored
            # strings and one number, so the two tables differ by design.
            "distinct_count": int(np.unique(finite).size) if count else 0,
            "sum": total,
        }
    )
    return row


def _outlier_rows(
    statistics: _ColumnStatistics, config: DescriptiveStatisticsConfig
) -> list[dict[str, object]]:
    """Tukey fence rows, one per configured multiplier."""

    finite = statistics.values
    count = statistics.count
    iqr = statistics.iqr
    # A degenerate box (q1 == q3, or too few points to have quartiles at all)
    # would fence every value outside a single point and report a fabricated
    # outlier rate. Emit the fences as NA instead of a fence that means nothing.
    evaluable = (
        count >= _MINIMUM_FENCE_OBSERVATIONS and math.isfinite(iqr) and iqr > 0.0
    )
    rows: list[dict[str, object]] = []
    for multiplier in config.outlier_iqr_multipliers:
        if evaluable:
            lower_fence = statistics.q1 - multiplier * iqr
            upper_fence = statistics.q3 + multiplier * iqr
            below = int(np.count_nonzero(finite < lower_fence))
            above = int(np.count_nonzero(finite > upper_fence))
            outlier_rate = (below + above) / count
        else:
            lower_fence = upper_fence = math.nan
            below = above = 0
            # Not 0.0: a zero rate reads as "checked, none found", and no fence
            # was evaluated here at all.
            outlier_rate = math.nan
        row: dict[str, object] = _column_keys(statistics.spec)
        row.update(
            {
                "iqr_multiplier": float(multiplier),
                "lower_fence": lower_fence,
                "upper_fence": upper_fence,
                "below_fence_count": below,
                "above_fence_count": above,
                "outlier_count": below + above,
                "outlier_rate": outlier_rate,
            }
        )
        rows.append(row)
    return rows


def _binned_bindings(source: ProfiledSource) -> list[ColumnBinding]:
    """The bound single-column quantities that exist on this source."""

    bindings: list[ColumnBinding] = []
    for label in _BINNED_BINDING_LABELS:
        binding = getattr(source.binding, label, None)
        if binding is not None:
            bindings.append(binding)
    return bindings


def _histogram_rows(
    source: ProfiledSource,
    spec: ColumnSpec,
    values: np.ndarray,
    config: DescriptiveStatisticsConfig,
) -> list[dict[str, object]]:
    """Equal-width bins spanning the observed range, final bin closed on the right."""

    count = int(values.size)
    if count == 0:
        return []
    low = float(values.min())
    high = float(values.max())
    keys = _column_keys(spec)
    if not high > low:
        # A constant column has no width to divide. numpy would silently widen the
        # range to (v - 0.5, v + 0.5), which would contradict the declared
        # [min, max] span, so emit the single occupied bin instead.
        row: dict[str, object] = dict(keys)
        row.update(
            {
                "bin_index": 0,
                "bin_lower": low,
                "bin_upper": high,
                "count": count,
                "share": 1.0,
            }
        )
        return [row]
    # np.histogram closes the final bin on the right, so the maximum is counted
    # rather than falling past the last edge.
    counts, edges = np.histogram(values, bins=config.histogram_bins, range=(low, high))
    rows: list[dict[str, object]] = []
    for index, binned in enumerate(counts.tolist()):
        row = dict(keys)
        row.update(
            {
                "bin_index": index,
                "bin_lower": float(edges[index]),
                "bin_upper": float(edges[index + 1]),
                "count": int(binned),
                "share": int(binned) / count,
            }
        )
        rows.append(row)
    return rows


def _frame(name: str, rows: Sequence[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows)) if rows else empty_table(name)
    return conform_table(name, frame)


def analyze_numeric(
    loaded: LoadedSources, config: DescriptiveStatisticsConfig
) -> dict[str, pd.DataFrame]:
    """Profile every numeric column of every profiled source."""

    quantile_plan = _quantile_plan(config)
    summary_rows: list[dict[str, object]] = []
    outlier_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []

    for source in loaded.sources:
        profiled: dict[str, _ColumnStatistics] = {}
        # Physical position order, which columns_of_kind preserves.
        for spec in source.columns_of_kind("numeric"):
            statistics = _profile_column(source, spec, config, quantile_plan)
            profiled[spec.raw_column_id] = statistics
            summary_rows.append(_summary_row(statistics, config))
            outlier_rows.extend(_outlier_rows(statistics, config))

        for binding in _binned_bindings(source):
            statistics = profiled.get(binding.raw_column_id)
            if statistics is None:
                # The bound column did not classify as numeric. That is a shape
                # finding for the structure profile to report; binning the subset
                # of cells that happened to parse would misrepresent the column.
                continue
            bin_rows.extend(
                _histogram_rows(source, statistics.spec, statistics.values, config)
            )

    return {
        "numeric_summary": _frame("numeric_summary", summary_rows),
        "numeric_outlier_audit": _frame("numeric_outlier_audit", outlier_rows),
        "numeric_distribution_bins": _frame("numeric_distribution_bins", bin_rows),
    }
