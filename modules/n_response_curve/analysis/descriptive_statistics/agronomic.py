"""Agronomic profile tables: what N rate and grain yield were actually recorded.

The domain view of the descriptive profile. Everything here is read off the
harmonized ``(N kg/ha, yield t/ha)`` observation frame that ``sources`` builds,
so unit conversion, zero-N tolerance, and row retention are settled in one place
and never re-derived here.

Descriptive only. These tables summarize recorded values within recorded
groupings — N bands, years, context levels, zero-N checks. Nothing here fits,
interpolates, extrapolates, ranks, or recommends; a binned mean of what was
recorded is not an estimate of what would be obtained.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import contracts
from .config import DescriptiveStatisticsConfig
from .sources import (
    CONVERTED_FROM_KG_HA,
    NATIVE_T_HA,
    ColumnBinding,
    LoadedSources,
    ProfiledSource,
    build_observation_frame,
)


# Screening bounds for unusual lowland-rice grain yields, not exclusions:
# rows stay in every table and count. A flagged value needs source review;
# it may reflect crop failure, unusual conditions, a unit/area error, or a
# recording error. The flag alone cannot establish which explanation applies
# and must not be used as a distributional trim.
IMPLAUSIBLE_LOW_YIELD_T_HA = 0.1
IMPLAUSIBLE_HIGH_YIELD_T_HA = 15.0

# Zero-N evidence has two independent bases and they are never pooled: one is a
# row whose recorded N rate is zero, the other is a paired check column carried
# alongside a fertilized row. Only ph_combined_nopt_rcm has the second.
ZERO_N_BASIS_TOLERANCE = "n_rate_within_zero_tolerance"
ZERO_N_BASIS_DECLARED_COLUMN = "declared_zero_n_yield_column"

YIELD_UNIT_BASIS = "t_ha"

# Lineage tokens are emitted in this fixed order rather than in count order, so
# the summary string is stable across datasets and across reruns.
_LINEAGE_ORDER = (NATIVE_T_HA, CONVERTED_FROM_KG_HA)

_MAXIMUM_LISTED_RATES = 12
_LIST_SEPARATOR = "; "
_TRUNCATION_MARK = "..."

_TABLE_NAMES = (
    "nitrogen_rate_profile",
    "yield_profile",
    "yield_by_nitrogen_bin",
    "temporal_coverage",
    "context_composition",
    "zero_nitrogen_checks",
)


@dataclass(frozen=True)
class _Distribution:
    """Location and spread of one finite numeric sample."""

    count: int
    mean: float
    median: float
    std_dev: float
    minimum: float
    maximum: float
    q1: float
    q3: float
    skewness: float

    @classmethod
    def of(cls, values: pd.Series) -> _Distribution:
        finite = pd.Series(values, dtype=float).dropna()
        count = int(finite.size)
        if count == 0:
            return cls(
                count=0,
                mean=math.nan,
                median=math.nan,
                std_dev=math.nan,
                minimum=math.nan,
                maximum=math.nan,
                q1=math.nan,
                q3=math.nan,
                skewness=math.nan,
            )
        return cls(
            count=count,
            mean=float(finite.mean()),
            median=float(finite.median()),
            # Sample standard deviation. A single observation carries no spread
            # to report, so pandas returns NA rather than a misleading 0.
            std_dev=float(finite.std(ddof=1)),
            minimum=float(finite.min()),
            maximum=float(finite.max()),
            q1=float(finite.quantile(0.25)),
            q3=float(finite.quantile(0.75)),
            skewness=float(finite.skew()),
        )

    @property
    def span(self) -> float:
        return self.maximum - self.minimum

    @property
    def coefficient_of_variation(self) -> float:
        # Undefined at a zero mean; reported as NA rather than as an infinity.
        if not math.isfinite(self.mean) or self.mean == 0.0:
            return math.nan
        return self.std_dev / self.mean


def _snap_to_tolerance(values: pd.Series, tolerance: float) -> pd.Series:
    """Group sorted rates by distance from each retained level's lowest rate.

    This is the same anchored tolerance rule used by dataset membership and
    curve fitting. Rounding to a grid instead would split arbitrarily close
    rates that straddle a grid boundary. Anchoring also avoids chaining a run
    of near-neighbours into a level wider than the declared tolerance.
    """

    numeric = values.to_numpy(dtype=float)
    grouped = numeric.copy()
    anchor: float | None = None
    for position in np.argsort(numeric, kind="stable"):
        value = float(numeric[position])
        if not math.isfinite(value):
            continue
        if anchor is None or value - anchor > tolerance:
            anchor = value
        grouped[position] = anchor
    return pd.Series(grouped, index=values.index)


def _format_rate_list(rates: Sequence[float]) -> str:
    listed = [f"{float(rate):g}" for rate in rates[:_MAXIMUM_LISTED_RATES]]
    if len(rates) > _MAXIMUM_LISTED_RATES:
        listed.append(_TRUNCATION_MARK)
    return _LIST_SEPARATOR.join(listed)


def _unit_lineage_summary(lineage: pd.Series) -> str:
    counts = lineage.value_counts()
    parts = [
        f"{token}: {int(counts[token])}"
        for token in _LINEAGE_ORDER
        if int(counts.get(token, 0)) > 0
    ]
    # A token the harmonizer gains later must still surface here rather than be
    # silently dropped, so anything unrecognized is appended in sorted order.
    for token in sorted(set(map(str, counts.index)) - set(_LINEAGE_ORDER)):
        if token and int(counts[token]) > 0:
            parts.append(f"{token}: {int(counts[token])}")
    return _LIST_SEPARATOR.join(parts)


def _source_keys(source: ProfiledSource) -> dict[str, Any]:
    return {
        "source_name": source.source_name,
        "data_classification": source.data_classification,
    }


def _aligned_row_positions(
    source: ProfiledSource, observations: pd.DataFrame
) -> np.ndarray:
    """Positional index into ``source.text`` for each harmonized observation.

    ``build_observation_frame`` resets its index, so ``source_row_number`` is the
    only link back to the physical row. Row numbers are unique within a source;
    a duplicate would make this join ambiguous, and ``Series.map`` raises on a
    non-unique lookup index rather than silently taking a first match.
    """

    lookup = pd.Series(
        np.arange(source.data_row_count), index=pd.Index(source.source_row_numbers)
    )
    return observations["source_row_number"].map(lookup).to_numpy(dtype=np.int64)


def _context_level_threshold(config: DescriptiveStatisticsConfig) -> int:
    """The count below which a context level is withheld.

    Two independent floors apply and the stricter wins: the agronomic grouping
    floor keeps thin cells out of summary statistics, while the categorical
    reporting floor is a disclosure control that stops a rare level of a
    restricted context field from becoming a de facto row identifier. Taking the
    maximum is a no-op while both are 3, and keeps the disclosure rule intact if
    they ever diverge.
    """

    return max(config.minimum_group_observations, config.minimum_level_count)


# --------------------------------------------------------------------------
# nitrogen_rate_profile
# --------------------------------------------------------------------------


def _nitrogen_rate_row(
    source: ProfiledSource,
    observations: pd.DataFrame,
    config: DescriptiveStatisticsConfig,
) -> dict[str, Any]:
    binding = source.binding.nitrogen_rate
    rates = observations["n_rate_kg_ha"].astype(float)
    snapped = _snap_to_tolerance(rates, config.n_level_tolerance_kg_ha)
    # One snapped array feeds both the count and the printed list, so the two can
    # never disagree about what "a distinct rate" is.
    distinct = np.sort(pd.unique(snapped.to_numpy(dtype=float)))
    distribution = _Distribution.of(rates)
    observation_count = int(observations.shape[0])
    zero_count = int(observations["is_zero_n"].sum())
    return {
        **_source_keys(source),
        "column_header": binding.header,
        "observation_count": observation_count,
        # Counted over every physical row of the source, whereas
        # observation_count is the harmonized count and additionally requires a
        # finite yield. The two are not complements of data_row_count and are
        # not meant to reconcile: they answer "how often was N left unrecorded"
        # and "how many usable N-yield pairs survived" separately.
        "missing_count": int(source.numeric_series(binding).isna().sum()),
        "distinct_rate_count": int(distinct.size),
        "minimum_kg_ha": distribution.minimum,
        "q1_kg_ha": distribution.q1,
        "median_kg_ha": distribution.median,
        "mean_kg_ha": distribution.mean,
        "q3_kg_ha": distribution.q3,
        "maximum_kg_ha": distribution.maximum,
        "std_dev_kg_ha": distribution.std_dev,
        "span_kg_ha": distribution.span,
        "zero_n_observation_count": zero_count,
        "zero_n_share": (
            zero_count / observation_count if observation_count else math.nan
        ),
        "distinct_rates": _format_rate_list(distinct.tolist()),
    }


# --------------------------------------------------------------------------
# yield_profile
# --------------------------------------------------------------------------


def _yield_missing_count(source: ProfiledSource) -> int:
    """Source rows carrying no finite yield on the common t/ha basis.

    A row counts as recorded when either the native t/ha column or the kg/ha
    column that feeds the same harmonized value is finite, matching exactly what
    the harmonizer accepts.
    """

    binding = source.binding
    recorded = source.numeric_series(binding.yield_t_ha).notna()
    if binding.yield_kg_ha is not None:
        recorded = recorded | source.numeric_series(binding.yield_kg_ha).notna()
    return int((~recorded).sum())


def _yield_row(source: ProfiledSource, observations: pd.DataFrame) -> dict[str, Any]:
    yields = observations["yield_t_ha"].astype(float)
    distribution = _Distribution.of(yields)
    return {
        **_source_keys(source),
        "column_header": source.binding.yield_t_ha.header,
        "unit_basis": YIELD_UNIT_BASIS,
        "unit_lineage": _unit_lineage_summary(observations["yield_unit_lineage"]),
        "observation_count": int(observations.shape[0]),
        "missing_count": _yield_missing_count(source),
        "minimum_t_ha": distribution.minimum,
        "q1_t_ha": distribution.q1,
        "median_t_ha": distribution.median,
        "mean_t_ha": distribution.mean,
        "q3_t_ha": distribution.q3,
        "maximum_t_ha": distribution.maximum,
        "std_dev_t_ha": distribution.std_dev,
        "coefficient_of_variation": distribution.coefficient_of_variation,
        "skewness": distribution.skewness,
        "implausible_low_count": int((yields < IMPLAUSIBLE_LOW_YIELD_T_HA).sum()),
        "implausible_high_count": int((yields > IMPLAUSIBLE_HIGH_YIELD_T_HA).sum()),
    }


# --------------------------------------------------------------------------
# yield_by_nitrogen_bin
# --------------------------------------------------------------------------


def _nitrogen_bin_rows(
    source: ProfiledSource,
    observations: pd.DataFrame,
    config: DescriptiveStatisticsConfig,
) -> list[dict[str, Any]]:
    """Recorded yields summarized inside fixed-width N bands.

    Each band is summarized in isolation: nothing is computed across bands and
    no band is compared with another. Bands are emitted in index order because
    that is their natural key, not because the sequence carries meaning.
    """

    width = config.nitrogen_bin_width_kg_ha
    rates = observations["n_rate_kg_ha"].to_numpy(dtype=float)
    # Half-open bands [lower, upper) anchored at 0, so a rate landing exactly on
    # a boundary always falls in the band that starts there.
    indices = np.floor(rates / width).astype(np.int64)
    binned = observations.assign(_bin_index=indices)

    rows: list[dict[str, Any]] = []
    for bin_index, group in binned.groupby("_bin_index", sort=True):
        count = int(group.shape[0])
        if count < config.minimum_group_observations:
            continue
        distribution = _Distribution.of(group["yield_t_ha"])
        lower = float(bin_index) * width
        rows.append(
            {
                **_source_keys(source),
                "bin_index": int(bin_index),
                "n_lower_kg_ha": lower,
                "n_upper_kg_ha": lower + width,
                "n_midpoint_kg_ha": lower + width / 2.0,
                "observation_count": count,
                "mean_n_kg_ha": float(group["n_rate_kg_ha"].mean()),
                "mean_yield_t_ha": distribution.mean,
                "median_yield_t_ha": distribution.median,
                "std_dev_yield_t_ha": distribution.std_dev,
                "minimum_yield_t_ha": distribution.minimum,
                "maximum_yield_t_ha": distribution.maximum,
            }
        )
    return rows


# --------------------------------------------------------------------------
# temporal_coverage
# --------------------------------------------------------------------------


def _temporal_rows(
    source: ProfiledSource, observations: pd.DataFrame
) -> list[dict[str, Any]]:
    total = int(observations.shape[0])
    dated = observations.loc[observations["year"].notna()]
    if dated.empty:
        return []
    # Years are recorded as whole numbers; rounding guards the float round-trip
    # through the numeric frame rather than reinterpreting a fractional year.
    years = dated["year"].astype(float).round().astype(np.int64)

    rows: list[dict[str, Any]] = []
    for year, group in dated.assign(_year=years).groupby("_year", sort=True):
        count = int(group.shape[0])
        rows.append(
            {
                **_source_keys(source),
                "year": int(year),
                "observation_count": count,
                # Denominator is the source's whole harmonized total, not the
                # dated subset, so observations with no recorded year show up as
                # the shortfall of the shares from 1 instead of disappearing.
                "share": count / total if total else math.nan,
                "mean_n_kg_ha": float(group["n_rate_kg_ha"].mean()),
                "mean_yield_t_ha": float(group["yield_t_ha"].mean()),
            }
        )
    return rows


# --------------------------------------------------------------------------
# context_composition
# --------------------------------------------------------------------------


def _context_rows(
    source: ProfiledSource,
    observations: pd.DataFrame,
    config: DescriptiveStatisticsConfig,
) -> list[dict[str, Any]]:
    if observations.empty or not source.binding.context:
        return []

    total = int(observations.shape[0])
    threshold = _context_level_threshold(config)
    positions = _aligned_row_positions(source, observations)
    rows: list[dict[str, Any]] = []

    for binding in source.binding.context:
        spec = source.column(binding.raw_column_id)
        if spec.suppressed:
            # Values are already blanked upstream; skipping outright means no
            # future change to that blanking can turn this into a leak path.
            continue
        values = (
            source.text[binding.raw_column_id]
            .astype(str)
            .str.strip()
            .to_numpy()[positions]
        )
        frame = observations.assign(_level=values)
        # A blank cell is an unrecorded context, not a level of it. Dropping it
        # here while keeping the source total as the share denominator leaves
        # blanks and withheld levels pooled in the residual, so a withheld count
        # cannot be recovered by subtraction.
        frame = frame.loc[frame["_level"] != ""]
        if frame.empty:
            continue

        grouped = [
            (str(level), group)
            for level, group in frame.groupby("_level", sort=False)
            if int(group.shape[0]) >= threshold
        ]
        # Descending count, then level text, so ties have a defined order.
        grouped.sort(key=lambda item: (-item[1].shape[0], item[0]))
        for level, group in grouped:
            count = int(group.shape[0])
            distribution = _Distribution.of(group["yield_t_ha"])
            rows.append(
                {
                    **_source_keys(source),
                    "context_label": binding.label,
                    "column_header": binding.header,
                    "level_value": level,
                    "observation_count": count,
                    "share": count / total if total else math.nan,
                    "mean_n_kg_ha": float(group["n_rate_kg_ha"].mean()),
                    "mean_yield_t_ha": distribution.mean,
                    "median_yield_t_ha": distribution.median,
                }
            )
    return rows


def build_context_composition(
    source: ProfiledSource,
    observations: pd.DataFrame,
    config: DescriptiveStatisticsConfig,
) -> pd.DataFrame:
    """Build the complete context table for one observation population.

    The ordinary profile calls this with the source's governed N/yield binding.
    A reporting variant may call it with a declared sibling treatment arm (for
    example, Farmer's Practice) so the variant is held to exactly the same
    banding, disclosure threshold, row alignment, and context rules as the
    published table rather than reimplementing those rules in plotting code.
    """

    rows = [
        *_applied_n_band_rows(source, observations, config),
        *_year_band_rows(source, observations, config),
        *_context_rows(source, observations, config),
    ]
    return contracts.conform_table(
        "context_composition", _frame("context_composition", rows)
    )


# The context label under which the derived applied-N bands are filed. Derived,
# not bound: no source records a band column, and the recorded rate is a
# quantity rather than a level — the long-running trials carry thirty distinct
# rates, which is a continuum for the purpose of a composition bar.
APPLIED_N_BAND_CONTEXT = "applied_n_band"


def _applied_n_band(rate: float, width: float) -> tuple[int, str]:
    """The band index and label for one recorded rate.

    Zero is its own band rather than the floor of the first one: it is the
    zero-N check arm, a treatment in its own right, and pooling it with the
    lowest fertilized rates would hide the arm the response curve rests on.
    Above zero the upper bound is inclusive, so a 50 kg rate lands in the band
    named for 50 rather than opening the band above it.
    """

    if rate < 0.0:
        return -1, "<0"
    if rate == 0.0:
        return 0, "0"
    index = int(math.ceil(rate / width))
    return index, f">{(index - 1) * width:g}–{index * width:g}"


def _applied_n_band_rows(
    source: ProfiledSource,
    observations: pd.DataFrame,
    config: DescriptiveStatisticsConfig,
) -> list[dict[str, Any]]:
    """Composition of the recorded N rate, banded, as a context field.

    Emitted in ascending band order — unlike the recorded context fields, which
    are emitted count-descending. The band axis is ordinal, and a reader takes
    the row order for the reading order.
    """

    if observations.empty:
        return []
    total = int(observations.shape[0])
    threshold = _context_level_threshold(config)
    width = config.applied_n_band_width_kg_ha
    # The zero-N arm is identified by the frame's own tolerance-based flag, so
    # this table and zero_nitrogen_checks below count the same rows as zero.
    banded = observations.assign(
        _band=[
            _applied_n_band(0.0 if zero else float(rate), width)
            for rate, zero in zip(
                observations["n_rate_kg_ha"], observations["is_zero_n"]
            )
        ]
    )
    rows: list[dict[str, Any]] = []
    for (index, label), group in sorted(
        banded.groupby("_band", sort=False), key=lambda item: item[0][0]
    ):
        count = int(group.shape[0])
        if count < threshold:
            continue
        distribution = _Distribution.of(group["yield_t_ha"])
        rows.append(
            {
                **_source_keys(source),
                "context_label": APPLIED_N_BAND_CONTEXT,
                "column_header": source.binding.nitrogen_rate.header,
                "level_value": label,
                "observation_count": count,
                "share": count / total if total else math.nan,
                "mean_n_kg_ha": float(group["n_rate_kg_ha"].mean()),
                "mean_yield_t_ha": distribution.mean,
                "median_yield_t_ha": distribution.median,
            }
        )
    return rows


# The context label under which the derived year bands are filed. Derived on the
# same terms as the applied-N band above: the recorded year is a quantity, and
# fifty of them is a continuum for the purpose of a composition bar.
YEAR_BAND_CONTEXT = "year_band"


def _year_band(year: int, span: int) -> tuple[int, str]:
    """The band index and label for one recorded year.

    Aligned to multiples of the span rather than to the first year present, so
    the bands of two datasets that overlap in time are the same bands and can be
    read against each other. At the default span of five that makes 1990-1994,
    1995-1999, and so on; at a span of ten it makes calendar decades.
    """

    start = (year // span) * span
    return start, f"{start}–{start + span - 1}"


def _year_band_rows(
    source: ProfiledSource,
    observations: pd.DataFrame,
    config: DescriptiveStatisticsConfig,
) -> list[dict[str, Any]]:
    """Composition of the recorded year, banded, as a context field.

    Emitted in ascending band order, like the applied-N bands and for the same
    reason. Rows with no recorded year are dropped rather than banded, and the
    share denominator stays the source's whole harmonized total, so an undated
    row shows as the shortfall of the shares from 1 exactly as it does in
    ``temporal_coverage``.
    """

    if observations.empty or source.binding.year is None:
        return []
    total = int(observations.shape[0])
    threshold = _context_level_threshold(config)
    span = config.year_band_span_years
    dated = observations.loc[observations["year"].notna()]
    if dated.empty:
        return []
    # Rounded for the same reason as in _temporal_rows: years are recorded whole,
    # and this guards the float round-trip rather than reinterpreting a fraction.
    years = dated["year"].astype(float).round().astype(np.int64)
    banded = dated.assign(
        _band=[_year_band(int(year), span) for year in years]
    )
    rows: list[dict[str, Any]] = []
    for (start, label), group in sorted(
        banded.groupby("_band", sort=False), key=lambda item: item[0][0]
    ):
        count = int(group.shape[0])
        if count < threshold:
            continue
        distribution = _Distribution.of(group["yield_t_ha"])
        rows.append(
            {
                **_source_keys(source),
                "context_label": YEAR_BAND_CONTEXT,
                "column_header": source.binding.year.header,
                "level_value": label,
                "observation_count": count,
                "share": count / total if total else math.nan,
                "mean_n_kg_ha": float(group["n_rate_kg_ha"].mean()),
                "mean_yield_t_ha": distribution.mean,
                "median_yield_t_ha": distribution.median,
            }
        )
    return rows


# --------------------------------------------------------------------------
# zero_nitrogen_checks
# --------------------------------------------------------------------------


def _zero_nitrogen_row(
    source: ProfiledSource, basis: str, yields: pd.Series
) -> dict[str, Any]:
    distribution = _Distribution.of(yields)
    return {
        **_source_keys(source),
        "basis": basis,
        "observation_count": distribution.count,
        "mean_yield_t_ha": distribution.mean,
        "median_yield_t_ha": distribution.median,
        "std_dev_yield_t_ha": distribution.std_dev,
        "minimum_yield_t_ha": distribution.minimum,
        "maximum_yield_t_ha": distribution.maximum,
    }


def _declared_zero_n_series(
    source: ProfiledSource, binding: ColumnBinding
) -> pd.Series:
    """The paired zero-N check column, read straight off the numeric frame.

    Deliberately not restricted to the harmonized rows: the column is its own
    record of a zero-N arm and is summarized as recorded, independent of whether
    the fertilized row beside it survived harmonization.
    """

    return source.numeric[binding.raw_column_id].astype(float).dropna()


def _zero_nitrogen_rows(
    source: ProfiledSource, observations: pd.DataFrame
) -> list[dict[str, Any]]:
    # Emitted even at a count of 0, so a dataset with no zero-N arm is visibly
    # absent rather than silently missing from the table.
    rows = [
        _zero_nitrogen_row(
            source,
            ZERO_N_BASIS_TOLERANCE,
            observations.loc[observations["is_zero_n"], "yield_t_ha"],
        )
    ]
    declared = source.binding.zero_n_yield_t_ha
    if declared is not None:
        rows.append(
            _zero_nitrogen_row(
                source,
                ZERO_N_BASIS_DECLARED_COLUMN,
                _declared_zero_n_series(source, declared),
            )
        )
    return rows


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def _frame(name: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return contracts.empty_table(name)
    return pd.DataFrame.from_records(rows)


def analyze_agronomic(
    loaded: LoadedSources, config: DescriptiveStatisticsConfig
) -> dict[str, pd.DataFrame]:
    """Build every agronomic table for every profiled source."""

    collected: dict[str, list[dict[str, Any]]] = {name: [] for name in _TABLE_NAMES}

    for source in loaded.sources:
        observations = build_observation_frame(
            source, zero_n_tolerance_kg_ha=config.zero_n_tolerance_kg_ha
        )
        collected["nitrogen_rate_profile"].append(
            _nitrogen_rate_row(source, observations, config)
        )
        collected["yield_profile"].append(_yield_row(source, observations))
        collected["yield_by_nitrogen_bin"].extend(
            _nitrogen_bin_rows(source, observations, config)
        )
        collected["temporal_coverage"].extend(_temporal_rows(source, observations))
        collected["context_composition"].extend(
            build_context_composition(source, observations, config).to_dict(
                orient="records"
            )
        )
        collected["zero_nitrogen_checks"].extend(
            _zero_nitrogen_rows(source, observations)
        )

    return {
        name: contracts.conform_table(name, _frame(name, rows))
        for name, rows in collected.items()
    }
