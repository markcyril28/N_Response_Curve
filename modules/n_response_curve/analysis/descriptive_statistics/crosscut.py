"""Cross-dataset profiling on the harmonized N-yield observation basis.

The four tables here are the only place the recipe puts the three datasets on
one page, so they carry the comparability audit with them: every aggregate is
accompanied by the record of what it was computed from, what unit it arrived
in, and what about it is *not* commensurable with the neighbouring row.

Harmonization is not re-derived here. ``sources.build_all_observations`` is the
single definition of what an observation is, and this module only aggregates
its output. Rows lacking a finite N rate or a finite yield never reach it; the
size of that drop is reported rather than hidden.

Descriptive only. Nothing here fits a response, orders the datasets by any
merit, or reads a rate back out as a recommendation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

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
    build_all_observations,
)

# Absorbs binary-representation dust when comparing gaps that were themselves
# reconstructed from integer tolerance buckets. It is not a second tolerance:
# the operator-facing tolerance is [agronomic].n_level_tolerance_kg_ha.
_FLOAT_SLACK = 1e-9

# Written verbatim into unit_lineage_audit.conversion. Two literals only, so a
# reviewer can grep the released table for every conversion the bundle applied.
_NO_CONVERSION = "none"
_KG_HA_TO_T_HA = "kg/ha / 1000"

_CALENDAR_YEAR = "calendar_year"
_KG_N_HA = "kg N ha-1"
_T_HA = "t ha-1"
_KG_HA = "kg ha-1"

# Per-source lineage note for the N rate. No source records N on anything but a
# kg N/ha basis, so the frame carries no N conversion at all; the counter is
# still emitted so the claim is auditable rather than assumed.
_NATIVE_N_LINEAGE = "native_kg_n_ha"

# One factual sentence per dataset naming what that dataset's rows are and what
# they therefore cannot be compared on. These are statements about what was
# recorded, not about measured effects, and every count in them is substituted
# from the frame so the prose cannot drift away from the table beside it. The
# single literal — LTCCE's 785 registered year-season-variety groups — is the
# registered provenance figure for the source extract rather than a frame
# statistic, and [expected_inputs] pins that extract by SHA-256.
_COMPARABILITY_NOTES: dict[str, str] = {
    "core_trial_data": (
        "Treatment-level rows extracted from heterogeneous published trials whose "
        "representation basis is recorded as {representation_basis} and is still "
        "under review, so a row is a reported treatment mean from one of many "
        "different experimental designs rather than a plot observation, and "
        "{dropped} of {source_rows} data rows are absent here because they record "
        "no finite N rate or no grain yield."
    ),
    "ph_combined_nopt_rcm": (
        "Each row is one linked NOPT/RCM record whose zero-N arm is held in the "
        "paired n0_yield column rather than as its own row, so this dataset "
        "contributes no zero-N observation to the harmonized frame and every one "
        "of its {series} series carries exactly one N rate by construction, which "
        "means no within-series N response can be read from this frame."
    ),
    "ltcce": (
        "Replicated plot-level records from a single long-term experiment spanning "
        "{year_min:.0f}-{year_max:.0f}, registered as carrying four N levels in "
        "every one of its 785 year-season-variety groups, of which {series} groups "
        "survive here and {full_ladder} retain all four once the {dropped} rows "
        "without a recorded yield are dropped, so its rows are not exchangeable "
        "with the treatment-mean rows of the literature extract and its N ladder "
        "is confounded with season and era."
    ),
}


@dataclass(frozen=True)
class _LadderGeometry:
    """Ladder shape of one dataset, measured over series-resolved rows only."""

    series_count: int
    median_levels: float
    minimum_levels: float
    maximum_levels: float
    series_with_zero_n: int
    median_span_kg_ha: float
    median_step_kg_ha: float
    balanced_series: int
    single_level_series: int
    # Series carrying the widest ladder observed in this dataset. For LTCCE,
    # whose registered ladder is four levels everywhere, this is exactly the
    # count of groups that kept all four after the yield drop.
    full_ladder_series: int
    resolved_observations: int
    unresolved_observations: int


@dataclass(frozen=True)
class _UnitLineageRow:
    """One audited quantity: the column it came from and how it reached its unit."""

    source_name: str
    data_classification: str
    quantity: str
    source_column_header: str
    source_unit: str
    target_unit: str
    conversion: str
    converted_observation_count: int
    native_observation_count: int
    note: str


def _distinct_levels(values: np.ndarray, tolerance: float) -> np.ndarray:
    """Sorted distinct N rates, snapped onto integer multiples of ``tolerance``.

    Snapping before uniquing is what makes "distinct within the tolerance" mean
    anything: 112.5 and 112.500000001 are one rung of the ladder, and a bare
    ``np.unique`` on the stored floats would split them into two. Integer
    buckets are used rather than ``round(x, n)`` so the tolerance stays an
    operator-set quantity instead of a hard-coded decimal place.
    """

    finite = values[np.isfinite(values)]
    if tolerance > 0.0:
        return np.unique(np.round(finite / tolerance)) * tolerance
    return np.unique(finite)


def _median_or_nan(values: Sequence[float]) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def _source_rows(observations: pd.DataFrame, source_name: str) -> pd.DataFrame:
    return observations.loc[observations["source_name"] == source_name]


def _series_key_basis(source: ProfiledSource, resolved: pd.DataFrame) -> str:
    """Short phrase naming the physical columns that key one series.

    Derived from the position-verified ``series`` binding rather than restated,
    so a binding change cannot leave the two crosscut tables describing a
    grouping the frame no longer uses. The suffix is data-derived: when no
    series holds more than one row the "series" is a single record, and saying
    so is the difference between "one N rate observed" and "one N rate
    possible".
    """

    headers = [binding.header for binding in source.binding.series]
    if not headers:
        return "(no series binding declared)"
    basis = "|".join(headers)
    if not resolved.empty and int(resolved.groupby("series_key").size().max()) == 1:
        return f"{basis} (one record per series)"
    return basis


def _ladder_geometry(rows: pd.DataFrame, *, tolerance: float) -> _LadderGeometry:
    """Ladder geometry over the series-resolved rows of one dataset.

    Rows whose series key has a blank component are excluded rather than
    bucketed together: ``build_observation_frame`` never forward-fills a key, so
    an unresolved row would otherwise merge unrelated trials into one ladder.
    """

    resolved = rows.loc[rows["is_series_resolved"]]
    level_counts: list[int] = []
    spans: list[float] = []
    steps: list[float] = []
    balanced = 0
    zero_n_series = 0
    for _, group in resolved.groupby("series_key", sort=True):
        levels = _distinct_levels(group["n_rate_kg_ha"].to_numpy(dtype=float), tolerance)
        if levels.size == 0:  # pragma: no cover - retained rows always have a rate
            continue
        level_counts.append(int(levels.size))
        spans.append(float(levels[-1] - levels[0]))
        if bool(group["is_zero_n"].any()):
            zero_n_series += 1
        if levels.size >= 2:
            gaps = np.diff(levels)
            steps.append(float(np.median(gaps)))
            # A two-rung ladder has a single gap and would be trivially "equally
            # spaced", which says nothing about design balance; require three.
            if levels.size >= 3 and float(gaps.max() - gaps.min()) <= tolerance + _FLOAT_SLACK:
                balanced += 1

    counts = np.asarray(level_counts, dtype=float)
    return _LadderGeometry(
        series_count=len(level_counts),
        median_levels=_median_or_nan(counts),
        minimum_levels=float(counts.min()) if counts.size else float("nan"),
        maximum_levels=float(counts.max()) if counts.size else float("nan"),
        series_with_zero_n=zero_n_series,
        median_span_kg_ha=_median_or_nan(spans),
        # NA when no series has two rungs. For ph_combined_nopt_rcm that is by
        # construction, not missing data: a record is its own series there.
        median_step_kg_ha=_median_or_nan(steps),
        balanced_series=balanced,
        single_level_series=int((counts == 1).sum()) if counts.size else 0,
        full_ladder_series=int((counts == counts.max()).sum()) if counts.size else 0,
        resolved_observations=int(len(resolved)),
        unresolved_observations=int(len(rows) - len(resolved)),
    )


def _lineage_summary(rows: pd.DataFrame) -> str:
    """``label=count`` roll-up of the yield unit lineage, widest path first."""

    counts = rows["yield_unit_lineage"].value_counts()
    ordered = [label for label in (NATIVE_T_HA, CONVERTED_FROM_KG_HA) if label in counts.index]
    ordered += [label for label in counts.index if label not in ordered]
    return "; ".join(f"{label}={int(counts[label])}" for label in ordered) or "none"


def _retained_positions(source: ProfiledSource, rows: pd.DataFrame) -> np.ndarray:
    """Positional mask of the physical rows of ``source`` that survived harmonization.

    Joined on ``source_row_number``, which ``ProfiledSource`` records per row, so
    a paired column that the observation frame does not carry (nopt's n0_yield)
    can still be counted on exactly the same row basis as everything else in the
    audit rather than on the full physical extract.
    """

    retained = set(rows["source_row_number"].astype(int).tolist())
    return np.fromiter(
        (number in retained for number in source.source_row_numbers),
        dtype=bool,
        count=source.data_row_count,
    )


def _finite_count(source: ProfiledSource, binding: ColumnBinding, mask: np.ndarray) -> int:
    values = source.numeric_series(binding).to_numpy(dtype=float)
    return int(np.isfinite(values[mask]).sum())


def _unit_lineage_rows(
    source: ProfiledSource,
    rows: pd.DataFrame,
) -> list[_UnitLineageRow]:
    """The full unit lineage of one dataset: every quantity the bundle reports.

    ``native`` and ``converted`` are counted on the harmonized observation
    basis, not on the physical extract, because that is what every other number
    in this bundle's crosscut tables is counted on. Each note names the physical
    row count as well, so this table and the structural inventory can be
    reconciled without either looking wrong.
    """

    binding = source.binding
    retained = _retained_positions(source, rows)
    physical_rows = source.data_row_count
    harmonized = int(len(rows))

    lineage_counts = rows["yield_unit_lineage"].value_counts()
    native_yield = int(lineage_counts.get(NATIVE_T_HA, 0))
    converted_yield = int(lineage_counts.get(CONVERTED_FROM_KG_HA, 0))

    def build(
        quantity: str,
        header: str,
        source_unit: str,
        target_unit: str,
        conversion: str,
        converted: int,
        native: int,
        note: str,
    ) -> _UnitLineageRow:
        return _UnitLineageRow(
            source_name=source.source_name,
            data_classification=source.data_classification,
            quantity=quantity,
            source_column_header=header,
            source_unit=source_unit,
            target_unit=target_unit,
            conversion=conversion,
            converted_observation_count=converted,
            native_observation_count=native,
            note=note,
        )

    parseable_n = _finite_count(source, binding.nitrogen_rate, np.ones(physical_rows, dtype=bool))
    audited: list[_UnitLineageRow] = [
        build(
            "nitrogen_rate",
            binding.nitrogen_rate.header,
            _KG_N_HA,
            _KG_N_HA,
            _NO_CONVERSION,
            0,
            harmonized,
            f"Physical position {binding.nitrogen_rate.position}, recorded on the "
            f"target unit already. {parseable_n} of {physical_rows} data rows carry "
            f"a parseable rate and all {harmonized} harmonized observations do, "
            "because a row without one is not an observation.",
        )
    ]

    parseable_yield = _finite_count(
        source, binding.yield_t_ha, np.ones(physical_rows, dtype=bool)
    )
    if binding.yield_kg_ha is None:
        blank_yield = physical_rows - parseable_yield
        yield_header = binding.yield_t_ha.header
        yield_unit = _T_HA
        yield_conversion = _NO_CONVERSION
        yield_note = (
            f"Physical position {binding.yield_t_ha.position}, recorded on the "
            f"target unit already. {parseable_yield} of {physical_rows} data rows "
            + (
                f"record a parseable yield; the {blank_yield} without one are "
                "dropped rather than imputed."
                if blank_yield
                else "record a parseable yield, so none is dropped for a missing yield."
            )
        )
    else:
        yield_header = f"{binding.yield_t_ha.header}; {binding.yield_kg_ha.header}"
        yield_unit = f"{_T_HA}; {_KG_HA}"
        yield_conversion = _KG_HA_TO_T_HA
        yield_note = (
            f"Physical position {binding.yield_t_ha.position} is the primary t/ha "
            f"column and carries {parseable_yield} of {physical_rows} data rows; "
            f"position {binding.yield_kg_ha.position} fills only the rows whose "
            f"t/ha cell is blank, which is {converted_yield} of the {harmonized} "
            "harmonized observations. The two columns are never averaged and the "
            "kg/ha column never overrides a recorded t/ha value."
        )
    audited.append(
        build(
            "grain_yield",
            yield_header,
            yield_unit,
            _T_HA,
            yield_conversion,
            converted_yield,
            native_yield,
            yield_note,
        )
    )

    if binding.zero_n_yield_t_ha is not None:
        paired = _finite_count(source, binding.zero_n_yield_t_ha, retained)
        audited.append(
            build(
                "zero_n_yield",
                binding.zero_n_yield_t_ha.header,
                _T_HA,
                _T_HA,
                _NO_CONVERSION,
                0,
                paired,
                f"Physical position {binding.zero_n_yield_t_ha.position}: the paired "
                "zero-N arm of the same record, held as a column rather than a row, "
                f"so it contributes no harmonized observation. {paired} of the "
                f"{harmonized} retained records carry a finite value; the zero-N "
                "check table profiles them.",
            )
        )

    if binding.year is None:  # pragma: no cover - every profiled source binds a year
        return audited

    parseable_year = _finite_count(source, binding.year, np.ones(physical_rows, dtype=bool))
    harmonized_year = int(rows["year"].notna().sum())
    # Named on both bases deliberately: the structural inventory counts this
    # column over all physical rows, so a reader comparing the two tables sees
    # two different numbers for the same column and needs to know why.
    year_note = (
        f"Physical position {binding.year.position}, a calendar year with no "
        f"conversion. {harmonized_year} of the {harmonized} harmonized "
        f"observations carry a parseable year, against {parseable_year} of the "
        f"{physical_rows} physical data rows"
    )
    if parseable_year - harmonized_year:
        year_note += (
            "; the difference is rows the harmonized frame does not retain, not a "
            "second parsing rule."
        )
    else:
        year_note += ", with no year lost to the harmonization drop."
    audited.append(
        build(
            "year",
            binding.year.header,
            _CALENDAR_YEAR,
            _CALENDAR_YEAR,
            _NO_CONVERSION,
            0,
            harmonized_year,
            year_note,
        )
    )
    return audited


def _comparability_note(
    source: ProfiledSource,
    rows: pd.DataFrame,
    geometry: _LadderGeometry,
) -> str:
    template = _COMPARABILITY_NOTES.get(source.source_name)
    if template is None:
        # An added source must not silently inherit another dataset's caveats.
        return (
            f"{source.source_name} has no reviewed comparability statement; treat "
            "its rows as not established to be commensurable with the others."
        )
    years = rows["year"].dropna()
    return template.format(
        representation_basis=source.representation_basis,
        dropped=source.data_row_count - int(len(rows)),
        source_rows=source.data_row_count,
        series=geometry.series_count,
        full_ladder=geometry.full_ladder_series,
        year_min=float(years.min()) if not years.empty else float("nan"),
        year_max=float(years.max()) if not years.empty else float("nan"),
    )


def _comparability_record(
    source: ProfiledSource,
    rows: pd.DataFrame,
    geometry: _LadderGeometry,
    *,
    tolerance: float,
) -> dict[str, object]:
    """One source_comparability row.

    The N-rate, yield and year aggregates span every harmonized row of the
    dataset, including any whose series key did not resolve — unlike
    nitrogen_ladder_geometry, which is defined on resolved rows only. Both
    readings coincide in the present data, so only this comment distinguishes
    them for a later reader.
    """

    n_rate = rows["n_rate_kg_ha"].to_numpy(dtype=float)
    yields = rows["yield_t_ha"]
    years = rows["year"].dropna()
    levels = _distinct_levels(n_rate, tolerance)
    year_min = float(years.min()) if not years.empty else float("nan")
    year_max = float(years.max()) if not years.empty else float("nan")
    return {
        "source_name": source.source_name,
        "data_classification": source.data_classification,
        "representation_basis": source.representation_basis,
        "harmonized_observation_count": int(len(rows)),
        "grouping_series_count": geometry.series_count,
        "n_rate_min_kg_ha": float(n_rate.min()) if n_rate.size else float("nan"),
        "n_rate_max_kg_ha": float(n_rate.max()) if n_rate.size else float("nan"),
        "distinct_n_rate_count": int(levels.size),
        "yield_min_t_ha": float(yields.min()) if len(yields) else float("nan"),
        "yield_median_t_ha": float(yields.median()) if len(yields) else float("nan"),
        "yield_max_t_ha": float(yields.max()) if len(yields) else float("nan"),
        "year_min": year_min,
        "year_max": year_max,
        # Observed extent, max minus min, matching the max-minus-min convention
        # used for median_span_kg_ha. It is not a count of distinct years: LTCCE
        # spans 49 on 50 recorded years.
        "year_span": year_max - year_min,
        "yield_unit_lineage": _lineage_summary(rows),
        "n_rate_unit_lineage": f"{_NATIVE_N_LINEAGE}={int(len(rows))}",
        "series_key_basis": _series_key_basis(source, rows.loc[rows["is_series_resolved"]]),
        "dropped_row_count": source.data_row_count - int(len(rows)),
        "comparability_note": _comparability_note(source, rows, geometry),
    }


def _geometry_record(
    source: ProfiledSource,
    rows: pd.DataFrame,
    geometry: _LadderGeometry,
) -> dict[str, object]:
    share = (
        geometry.series_with_zero_n / geometry.series_count
        if geometry.series_count
        else float("nan")
    )
    return {
        "source_name": source.source_name,
        "data_classification": source.data_classification,
        "series_count": geometry.series_count,
        "median_levels_per_series": geometry.median_levels,
        "minimum_levels_per_series": geometry.minimum_levels,
        "maximum_levels_per_series": geometry.maximum_levels,
        "series_with_zero_n_count": geometry.series_with_zero_n,
        "series_with_zero_n_share": share,
        "median_span_kg_ha": geometry.median_span_kg_ha,
        "median_step_kg_ha": geometry.median_step_kg_ha,
        "balanced_ladder_series_count": geometry.balanced_series,
        "single_level_series_count": geometry.single_level_series,
        "resolved_observation_count": geometry.resolved_observations,
        "unresolved_series_observation_count": geometry.unresolved_observations,
        "series_key_basis": _series_key_basis(source, rows.loc[rows["is_series_resolved"]]),
    }


def _frame(records: Iterable[dict[str, object]], name: str) -> pd.DataFrame:
    materialized = list(records)
    if not materialized:
        return contracts.empty_table(name)
    return pd.DataFrame.from_records(materialized)


def analyze_crosscut(
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> dict[str, pd.DataFrame]:
    """Build every crosscut table from the harmonized observation frame.

    Sources are iterated in configured order rather than grouped out of the
    observation frame, so a dataset whose rows all dropped still appears as a
    row of zeros in the two per-source tables instead of vanishing from a
    comparison that is supposed to list every profiled dataset.

    No suppression filtering is applied here: privacy suppression happens in
    ``sources``, which blanks restricted identifier cells before any profile
    sees them, and none of the four quantities audited below is bound to a
    suppressed column.
    """

    observations = build_all_observations(
        loaded, zero_n_tolerance_kg_ha=config.zero_n_tolerance_kg_ha
    )

    comparability: list[dict[str, object]] = []
    geometry_records: list[dict[str, object]] = []
    lineage: list[dict[str, object]] = []
    for source in loaded.sources:
        rows = _source_rows(observations, source.source_name)
        geometry = _ladder_geometry(rows, tolerance=config.n_level_tolerance_kg_ha)
        comparability.append(
            _comparability_record(
                source, rows, geometry, tolerance=config.n_level_tolerance_kg_ha
            )
        )
        geometry_records.append(_geometry_record(source, rows, geometry))
        lineage.extend(asdict(row) for row in _unit_lineage_rows(source, rows))

    if config.include_row_level_observations:
        row_level = observations.copy()
    else:
        # The row-level export is the only artifact that carries restricted rows
        # verbatim, so it stays behind an explicit operator switch.
        row_level = contracts.empty_table("harmonized_observations")

    return {
        "source_comparability": contracts.conform_table(
            "source_comparability", _frame(comparability, "source_comparability")
        ),
        "nitrogen_ladder_geometry": contracts.conform_table(
            "nitrogen_ladder_geometry", _frame(geometry_records, "nitrogen_ladder_geometry")
        ),
        "unit_lineage_audit": contracts.conform_table(
            "unit_lineage_audit", _frame(lineage, "unit_lineage_audit")
        ),
        "harmonized_observations": contracts.conform_table(
            "harmonized_observations", row_level
        ),
    }
