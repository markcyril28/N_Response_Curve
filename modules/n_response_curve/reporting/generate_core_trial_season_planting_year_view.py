#!/usr/bin/env python3
"""Split one recorded season of the governed core-trial overlay by planting year.

This is the by-planting-year refinement of ``generate_core_trial_season_view.py``
and the sibling of ``generate_core_trial_season_variety_view.py``: it takes one
season's stratum of that generator's exact population — the governed
21-series/74-observation core-trial set drawn in
``core_trial_data_source_wide.jpeg`` — and cuts it again on the planting year the
eligibility ledger records. Nothing is clustered and no curve is fitted.

Three things decide whether these panels can be read at all:

* **One frame, pinned once.** Every panel on every plate uses one padded frame
  taken from all 74 governed observations, both seasons pooled — the same
  construction ``season_comparison.jpeg`` uses. Autoscaling a cell would let a
  four-point ladder fill the same box as the eleven-series 1993 cell, and the
  years would stop being comparable to each other or to the parent stratum.

* **Most cells are one repeated trial.** Study 68 contributes a single
  four-level IR72 ladder at Laguna in every dry season 1991-1995 and every wet
  season 1992-1995. A step between those panels is a difference between two
  runs of one experiment, not an estimated year effect. The cells that are
  bigger are bigger because another study joins, and the README states which,
  computed from the data rather than asserted.

* **Season context, not a season comparison.** Each season's directory also
  carries the joint season-by-year grid with the *other* season's row muted, in
  the idiom the variety view already uses for its facets. The muted row is
  there so a year column can be located across seasons; it is not an invitation
  to test one row against the other.

Series recorded against a year *span* (``1991-1994``, ``2003-04``) are refused
rather than binned into their first year. The governed population contains none,
and if one ever arrives it is a decision, not a rounding.

Output is nested at ``clusters/by_season/<season>/by_planting_year/``, the same
shape LTCCE uses at ``clusters/by_season/ds/by_planting_year/``. The season view
preserves non-managed entries of ``by_season/`` across its own replacement and
the raw overlay generator preserves ``clusters/`` whole, so this subtree
survives both.

Exploratory diagnostic; not a governed analysis (ANA-11).

Usage:
    conda run -n n_response python \
      modules/n_response_curve/reporting/generate_core_trial_season_planting_year_view.py \
      --config scriptCONFIG.toml \
      --package WF/z_archive/06_Reports/n_response_full
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import sys
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.reporting.directory_publication import (  # noqa: E402
    nested_publication_container as _publication_container,
    plain_absolute_path as _plain_absolute_path,
    publication_lock as _core_overlay_publication_lock,
    recover_interrupted_directory_publication as _recover_interrupted_directory_publication,  # noqa: E501
)
from n_response_curve.reporting.figure_axis_frames import (  # noqa: E402
    SharedAxisLimits,
    padded_limits as _padded_limits,
)
from n_response_curve.reporting.figure_output import (  # noqa: E402
    save_figure_atomically as _save_governed_figure,
)
from n_response_curve.reporting.generate_core_trial_season_view import (  # noqa: E402
    SeasonStratum,
    _SEASON_LABELS,
    _SEASON_ORDER,
    _draw_stratum,
    _finite_observations,
    group_governed_records_by_season,
)
from n_response_curve.reporting.generate_raw_dataset_overlays import (  # noqa: E402
    DEFAULT_YIELD_THRESHOLD_T_HA,
    _load_governed_core_inputs,
)
from n_response_curve.reporting.planting_year_axis import (  # noqa: E402
    year_bounds as _year_bounds,
)
from n_response_curve.reporting.source_display_names import (  # noqa: E402
    display_source_name,
)

SOURCE_NAME = "core_trial_data"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scriptCONFIG.toml"
SEASON_ROOT = (
    PROJECT_ROOT
    / "WF/03_Response_Curves"
    / "literature_extracted_dataset/clusters/by_season"
)
SUBDIRECTORY_NAME = "by_planting_year"
ALL_SEASONS = "all"

_PANELS_FILENAME = "planting_year_panels.jpeg"
_CONTEXT_FILENAME = "planting_year_across_seasons.jpeg"
_SUMMARY_FILENAME = "planting_year_summary.json"
_FIXED_MANAGED_FILENAMES = frozenset(
    {"README.md", _SUMMARY_FILENAME, _PANELS_FILENAME, _CONTEXT_FILENAME}
)
# One plate per planting year, so the managed names are not a fixed list. Any
# four-digit `<year>.jpeg` is this generator's to replace or drop: leaving a
# plate for a year the population no longer contains would be worse than
# deleting an operator file that happens to be named like one.
_YEAR_PLATE_PATTERN = re.compile(r"^\d{4}\.jpeg$")

# Muted styling for the other season's row on the context grid. Grey at this
# weight reads as background at a glance while still resolving individual
# points when the plate is opened full size.
_MUTED_POINT_COLOUR = "#b8b8b8"
_MUTED_LINE_COLOUR = "#d8d8d8"
_MUTED_LABEL = "other recorded season (shown for position only)"

_PANEL_WIDTH_IN = 5.2
_PANEL_HEIGHT_IN = 6.0
_CONTEXT_PANEL_WIDTH_IN = 4.6
_CONTEXT_PANEL_HEIGHT_IN = 5.0
_SINGLE_FIGURE_SIZE_IN = (10.0, 7.0)


# --------------------------------------------------------------------------
# Planting-year cells
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class YearCell:
    """Every governed observation of one season stratum in one planting year."""

    season: str
    year: int
    series_uids: tuple[str, ...]
    records: tuple[Mapping[str, Any], ...]

    def as_stratum(self) -> SeasonStratum:
        """Reuse the parent view's container so its renderer applies unchanged."""

        return SeasonStratum(
            season=self.season,
            series_uids=self.series_uids,
            records=self.records,
        )

    @property
    def observation_count(self) -> int:
        return len(self.records)


def split_stratum_by_planting_year(stratum: SeasonStratum) -> tuple[YearCell, ...]:
    """Cut one season stratum into recorded planting years, in calendar order.

    Fails closed on a missing year, an unparseable one, a series carrying more
    than one recorded year, and on a year recorded as a span. Every failure is
    the same refusal the season splitter makes: a governed response trajectory
    is never divided between two panels, and never placed in a panel by guess.
    """

    year_by_series: dict[str, int] = {}
    records_by_series: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for record in stratum.records:
        series_uid = str(record.get("response_series_uid") or "").strip()
        records_by_series[series_uid].append(record)

    for series_uid in stratum.series_uids:
        members = records_by_series.get(series_uid, [])
        if not members:
            raise ValueError(
                f"Governed response series {series_uid!r} has no retained records"
            )
        labels = {str(record.get("planting_year") or "").strip() for record in members}
        if len(labels) > 1:
            raise ValueError(
                f"Governed response series {series_uid!r} carries more than one "
                f"recorded planting year: {sorted(labels)}"
            )
        label = next(iter(labels))
        if not label:
            raise ValueError(
                f"Governed response series {series_uid!r} has no recorded planting year"
            )
        bounds = _year_bounds(label)
        if bounds is None:
            raise ValueError(
                f"Governed response series {series_uid!r} has an unparseable "
                f"recorded planting year {label!r}"
            )
        first, last = bounds
        if first != last:
            raise ValueError(
                f"Governed response series {series_uid!r} records planting year "
                f"{label!r} as a span; the year panel it belongs in would be a guess"
            )
        year_by_series[series_uid] = first

    cells: list[YearCell] = []
    for year in sorted(set(year_by_series.values())):
        series = tuple(
            series_uid
            for series_uid in stratum.series_uids
            if year_by_series[series_uid] == year
        )
        series_set = set(series)
        cells.append(
            YearCell(
                season=stratum.season,
                year=year,
                series_uids=series,
                records=tuple(
                    record
                    for record in stratum.records
                    if str(record.get("response_series_uid") or "").strip() in series_set
                ),
            )
        )
    return tuple(cells)


def split_strata_by_planting_year(
    strata: Mapping[str, SeasonStratum],
) -> dict[str, tuple[YearCell, ...]]:
    """Split every season stratum, preserving the parent view's season order."""

    return {
        season: split_stratum_by_planting_year(strata[season])
        for season in _SEASON_ORDER
        if season in strata
    }


def shared_frame(strata: Mapping[str, SeasonStratum]) -> SharedAxisLimits:
    """One padded frame over all governed observations, both seasons pooled.

    Built the way `season_comparison.jpeg` builds its shared axes, so a cell on
    these plates can be laid beside the parent comparison directly. Both
    seasons enter the frame even when only one is being written, or the dry and
    wet directories would carry two different frames.
    """

    observations = [
        observation
        for stratum in strata.values()
        for observation in _finite_observations(stratum)
    ]
    if not observations:
        raise ValueError("The governed population has no finite observations")
    n_values = [observation.n_rate_kg_ha for observation in observations]
    yield_values = [observation.yield_t_ha for observation in observations]
    return SharedAxisLimits(
        x=_padded_limits((min(n_values), max(n_values))),
        y=_padded_limits((min(yield_values), max(yield_values))),
    )


# --------------------------------------------------------------------------
# Cell description
# --------------------------------------------------------------------------


def _distinct(records: Sequence[Mapping[str, Any]], column: str) -> tuple[str, ...]:
    values = {str(record.get(column) or "").strip() for record in records}
    return tuple(sorted(value for value in values if value))


def describe_cell(cell: YearCell) -> dict[str, Any]:
    """The machine-readable and README description of one planting-year cell."""

    observations = _finite_observations(cell.as_stratum())
    n_values = sorted({observation.n_rate_kg_ha for observation in observations})
    zero_n_series = {
        observation.response_series_uid
        for observation in observations
        if observation.n_rate_kg_ha == 0.0
    }
    return {
        "planting_year": cell.year,
        "series_count": len(cell.series_uids),
        "observation_count": len(observations),
        "distinct_n_levels": len(n_values),
        "n_rate_range_kg_ha": [n_values[0], n_values[-1]],
        "series_with_zero_n_anchor": len(zero_n_series),
        "study_ids": list(_distinct(cell.records, "study_id")),
        "rice_varieties": list(_distinct(cell.records, "rice_variety")),
        "provinces": list(_distinct(cell.records, "province")),
        "figure": f"{cell.year}.jpeg",
    }


def _repeated_study_note(cells: Sequence[YearCell]) -> str:
    """State, from the data, which study repeats across this stratum's years.

    The interpretation boundary below turns on whether the year panels are
    separate experiments or repeated runs of one. That is a fact about the
    records, so it is computed here rather than written into the prose.
    """

    years_by_study: dict[str, set[int]] = collections.defaultdict(set)
    for cell in cells:
        for study in _distinct(cell.records, "study_id"):
            years_by_study[study].add(cell.year)
    repeated = {
        study: sorted(years)
        for study, years in years_by_study.items()
        if len(years) > 1
    }
    if not repeated:
        return "No study contributes to more than one planting year in this stratum."
    parts = [
        f"study {study} in {min(years)}-{max(years)} ({len(years)} years)"
        for study, years in sorted(repeated.items())
    ]
    return "Contributing to more than one planting year: " + "; ".join(parts) + "."


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def _draw_cell(axes: Any, cell: YearCell, *, include_legend: bool) -> None:
    """Draw one cell with the parent view's renderer, retitled for the year."""

    stratum = cell.as_stratum()
    _draw_stratum(axes, stratum, include_legend=include_legend)
    observations = _finite_observations(stratum)
    n_values = [observation.n_rate_kg_ha for observation in observations]
    axes.set_title(
        f"{_SEASON_LABELS[cell.season]} {cell.year}\n"
        f"series={len(cell.series_uids)}; observations={len(observations)}; "
        f"N range={min(n_values):g}-{max(n_values):g} kg/ha"
    )


def _draw_muted_cell(axes: Any, cell: YearCell, *, include_label: bool) -> None:
    """Draw a cell of the other season as background position, not as data."""

    observations = _finite_observations(cell.as_stratum())
    by_series: dict[str, list[Any]] = collections.defaultdict(list)
    for observation in observations:
        by_series[observation.response_series_uid].append(observation)
    for series_uid in cell.series_uids:
        ordered = sorted(
            by_series[series_uid],
            key=lambda observation: (observation.n_rate_kg_ha, observation.record_uid),
        )
        axes.plot(
            [observation.n_rate_kg_ha for observation in ordered],
            [observation.yield_t_ha for observation in ordered],
            color=_MUTED_LINE_COLOUR,
            linewidth=1,
            zorder=1,
            label="_nolegend_",
        )
    axes.scatter(
        [observation.n_rate_kg_ha for observation in observations],
        [observation.yield_t_ha for observation in observations],
        color=_MUTED_POINT_COLOUR,
        zorder=2,
        label=_MUTED_LABEL if include_label else "_nolegend_",
    )
    axes.set_xlabel("Applied N (kg N/ha)")
    axes.set_ylabel("Grain yield (t/ha)")
    axes.set_title(
        f"{_SEASON_LABELS[cell.season]} {cell.year}\n"
        f"series={len(cell.series_uids)}; observations={len(observations)}",
        color="grey",
    )


def _draw_absent_cell(axes: Any, season: str, year: int, *, muted: bool) -> None:
    """Keep an empty season-year cell on the grid instead of dropping it.

    The grid is a contingency layout. Hiding the empty cells would let the
    columns shift under one another and make a year with no governed series
    look like a year that was not examined.
    """

    axes.text(
        0.5,
        0.5,
        "no governed series",
        transform=axes.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="grey",
    )
    axes.set_xlabel("Applied N (kg N/ha)")
    axes.set_ylabel("Grain yield (t/ha)")
    axes.set_title(
        f"{_SEASON_LABELS[season]} {year}\nseries=0; observations=0",
        color="grey" if muted else "black",
    )


def _tidy_shared_labels(axes_grid: Sequence[Sequence[Any]]) -> None:
    """Drop the axis labels shared axes have already made redundant.

    ``sharex``/``sharey`` leave tick labels only on the outer panels, so every
    inner panel otherwise keeps an axis label with no numbers under it.
    """

    rows = len(axes_grid)
    for row_index, row in enumerate(axes_grid):
        for column_index, axes in enumerate(row):
            if row_index != rows - 1:
                axes.set_xlabel("")
            if column_index != 0:
                axes.set_ylabel("")


def _figure_legend(figure: Any, axes_list: Sequence[Any]) -> None:
    handles: list[Any] = []
    labels: list[str] = []
    seen: set[str] = set()
    for axes in axes_list:
        for handle, label in zip(*axes.get_legend_handles_labels(), strict=True):
            if label in seen or label.startswith("_"):
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    if not handles:
        return
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=max(1, len(labels)),
        fontsize=8,
        frameon=False,
    )
    # A figure-level legend is outside constrained-layout's axes accounting, so
    # it needs its own reserved strip or it lands on the x labels.
    figure.get_layout_engine().set(rect=(0.0, 0.07, 1.0, 0.93))


def _build_panels_figure(
    season: str,
    cells: Sequence[YearCell],
    limits: SharedAxisLimits,
) -> Any:
    """One panel per planting year of this stratum, on the shared frame."""

    from matplotlib import pyplot as plt

    figure, axes_grid = plt.subplots(
        1,
        len(cells),
        figsize=(_PANEL_WIDTH_IN * len(cells), _PANEL_HEIGHT_IN),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes_list = list(axes_grid[0])
    try:
        for axes, cell in zip(axes_list, cells, strict=True):
            _draw_cell(axes, cell, include_legend=False)
            limits.apply(axes)
        _tidy_shared_labels(axes_grid)
        _figure_legend(figure, axes_list)
        figure.suptitle(
            f"source={display_source_name(SOURCE_NAME)} — {_SEASON_LABELS[season]} "
            "stratum split by recorded planting year\n"
            f"planting years={len(cells)}; "
            f"series={sum(len(cell.series_uids) for cell in cells)}; "
            f"observations={sum(cell.observation_count for cell in cells)}; "
            "all panels share one frame pinned from the whole governed population\n"
            "recorded strata only; no statistical clustering, pooled curve, or fit",
            fontsize=13,
        )
        return figure
    except Exception:
        plt.close(figure)
        raise


def _build_context_figure(
    season: str,
    cells_by_season: Mapping[str, Sequence[YearCell]],
    limits: SharedAxisLimits,
) -> Any:
    """Both seasons as rows on one year axis, with the other season muted."""

    from matplotlib import pyplot as plt

    seasons = tuple(cells_by_season)
    years = sorted({cell.year for cells in cells_by_season.values() for cell in cells})
    figure, axes_grid = plt.subplots(
        len(seasons),
        len(years),
        figsize=(
            _CONTEXT_PANEL_WIDTH_IN * len(years),
            _CONTEXT_PANEL_HEIGHT_IN * len(seasons),
        ),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes_by_row: dict[str, list[Any]] = {name: [] for name in seasons}
    try:
        muted_labelled = False
        for row, row_season in enumerate(seasons):
            by_year = {cell.year: cell for cell in cells_by_season[row_season]}
            for column, year in enumerate(years):
                axes = axes_grid[row][column]
                axes_by_row[row_season].append(axes)
                cell = by_year.get(year)
                if cell is None:
                    _draw_absent_cell(axes, row_season, year, muted=row_season != season)
                elif row_season == season:
                    _draw_cell(axes, cell, include_legend=False)
                else:
                    _draw_muted_cell(axes, cell, include_label=not muted_labelled)
                    muted_labelled = True
                limits.apply(axes)
        _tidy_shared_labels(axes_grid)
        # Focus row first, so the legend leads with the season this plate is
        # about rather than with the row that is only there for position.
        _figure_legend(
            figure,
            [
                axes
                for name in (season, *(n for n in seasons if n != season))
                for axes in axes_by_row[name]
            ],
        )
        other = ", ".join(
            _SEASON_LABELS[name] for name in seasons if name != season
        )
        figure.suptitle(
            f"source={display_source_name(SOURCE_NAME)} — {_SEASON_LABELS[season]} "
            "by recorded planting year, in season context\n"
            "rows=recorded season; columns=recorded planting year; "
            f"{_SEASON_LABELS[season]} drawn in full, {other or 'no other season'} "
            "muted for position only\n"
            "recorded strata only; no statistical clustering, pooled curve, fit, "
            "or between-row comparison",
            fontsize=14,
        )
        return figure
    except Exception:
        plt.close(figure)
        raise


def _build_single_cell_figure(cell: YearCell, limits: SharedAxisLimits) -> Any:
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(figsize=_SINGLE_FIGURE_SIZE_IN, constrained_layout=True)
    try:
        _draw_cell(axes, cell, include_legend=True)
        limits.apply(axes)
        axes.set_title(
            "\n".join(
                (
                    f"source={display_source_name(SOURCE_NAME)}",
                    axes.get_title(),
                    "complete governed response series retained within the cell",
                    "frame shared with every other planting-year panel",
                    "descriptive observed-series overlay; no pooled curve or fit",
                )
            )
        )
        return figure
    except Exception:
        plt.close(figure)
        raise


def _write_figure(figure: Any, destination: Path) -> None:
    from matplotlib import pyplot as plt

    try:
        _save_governed_figure(figure, destination)
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------
# Summary and README
# --------------------------------------------------------------------------


def build_summary(
    season: str,
    cells_by_season: Mapping[str, Sequence[YearCell]],
    limits: SharedAxisLimits,
) -> dict[str, Any]:
    cells = cells_by_season[season]
    return {
        "source_name": SOURCE_NAME,
        "season": season,
        "season_label": _SEASON_LABELS[season],
        "method": "recorded_planting_year_stratification_within_recorded_season",
        "interpretation": (
            "Recorded planting-year cells inside one recorded season stratum of "
            "the governed response series; not an unsupervised cluster analysis "
            "and no curve is fitted."
        ),
        "population": {
            "planting_year_count": len(cells),
            "series_count": sum(len(cell.series_uids) for cell in cells),
            "observation_count": sum(cell.observation_count for cell in cells),
        },
        "governed_population": {
            "series_count": sum(
                len(cell.series_uids)
                for season_cells in cells_by_season.values()
                for cell in season_cells
            ),
            "observation_count": sum(
                cell.observation_count
                for season_cells in cells_by_season.values()
                for cell in season_cells
            ),
        },
        "shared_axis_limits": {
            "n_rate_kg_ha": list(limits.x) if limits.x else None,
            "yield_t_ha": list(limits.y) if limits.y else None,
        },
        "figures": {
            "panels": _PANELS_FILENAME,
            "season_context": _CONTEXT_FILENAME,
        },
        "repeated_study_note": _repeated_study_note(cells),
        "planting_years": [describe_cell(cell) for cell in cells],
        "season_context": {
            name: [cell.year for cell in season_cells]
            for name, season_cells in cells_by_season.items()
        },
    }


def _readme(
    season: str,
    summary: Mapping[str, Any],
    *,
    package_argument: str | None,
    sibling_variety_view: bool,
) -> str:
    label = summary["season_label"]
    population = summary["population"]
    governed = summary["governed_population"]
    other_seasons = [name for name in summary["season_context"] if name != season]
    lines = [
        f"# `{SOURCE_NAME}` — {label} stratum grouped by recorded planting year",
        "",
        "exploratory diagnostic; not a governed analysis (ANA-11) — no curve is fitted.",
        "",
        f"This regroups the exact population drawn in `../../{season}.jpeg` on the",
        "planting year the promoted release's eligibility ledger records. It is not",
        "an unsupervised cluster analysis, and no series is split between panels.",
        "",
        "## Population",
        "",
        f"- {population['planting_year_count']} recorded planting years, "
        f"{population['series_count']} governed response series, "
        f"{population['observation_count']} finite observed N-yield pairs.",
        f"- Drawn from the governed {governed['series_count']}-series / "
        f"{governed['observation_count']}-observation core-trial population; the "
        "other season holds the remainder.",
        "- Series recorded against a year *span* are refused rather than binned",
        "  into their first year. This population contains none.",
        "",
        "## Outputs",
        "",
        f"- `{_PANELS_FILENAME}` — one panel per planting year, on the shared frame.",
        f"- `{_CONTEXT_FILENAME}` — both seasons as rows against one column of",
        f"  planting years, with {label} drawn in full and the other season muted",
        "  for position only. Empty cells are drawn and labelled, not dropped.",
        "- `<year>.jpeg` — one full-size plate per planting year.",
        f"- `{_SUMMARY_FILENAME}` — per-year counts, ladders, studies, varieties",
        "  and provinces.",
        "",
        "## Shared frame",
        "",
        "Every panel here is pinned to one padded frame taken from all",
        f"{governed['observation_count']} governed observations, both seasons pooled —",
        "the same construction `../../season_comparison.jpeg` uses. A four-point",
        "ladder therefore occupies the part of the frame it actually spans instead",
        "of being autoscaled up to fill its panel, and the dry and wet directories",
        "carry the same frame rather than two different ones.",
        "",
        "## What each planting year contains",
        "",
        "| Planting year | Series | Observations | Distinct N levels | "
        "N range (kg/ha) | Zero-N anchored | Studies | Provinces | Varieties |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- |",
    ]
    for cell in summary["planting_years"]:
        low, high = cell["n_rate_range_kg_ha"]
        lines.append(
            "| {year} | {series} | {obs} | {levels} | {low:g}-{high:g} | {zero} "
            "| {studies} | {provinces} | {varieties} |".format(
                year=cell["planting_year"],
                series=cell["series_count"],
                obs=cell["observation_count"],
                levels=cell["distinct_n_levels"],
                low=low,
                high=high,
                zero=f"{cell['series_with_zero_n_anchor']}/{cell['series_count']}",
                studies=", ".join(cell["study_ids"]) or "unrecorded",
                provinces=", ".join(cell["provinces"]) or "unrecorded",
                varieties=", ".join(cell["rice_varieties"]) or "unrecorded",
            )
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Read the table above before reading the plates. These year cells are not",
        "independent samples of a year effect.",
        "",
        f"- {summary['repeated_study_note']} Where one study supplies the same",
        "  variety at the same province in consecutive years, a step between those",
        "  panels is a difference between two runs of one experiment, not an",
        "  estimated year effect.",
        "- A year whose cell is larger is larger because another study joins it, so",
        "  its wider yield spread is a change of composition rather than of year.",
        "- A cell whose study, province or variety differs from its neighbours is a",
        "  different experiment reported in a different place, and the planting year",
        "  is the least of what separates it.",
        "- Response above zero N is undefined for any series with no zero-N anchor;",
        "  the table gives the anchored count per year.",
        "- Connecting lines are a visual aid over a series' own observed points.",
        "- Recorded season and recorded planting year are both compilation facts",
        "  taken from the extracted literature. Neither carries any correction for",
        "  site, management, treatment ladder, variety or reporting practice.",
        "",
        f"On `{_CONTEXT_FILENAME}`, the muted row exists so a year column can be",
        "located across seasons. It is not an invitation to compare the rows: the",
        "two seasons hold different studies in several years, and the parent",
        "`../../season_comparison.jpeg` is where the season strata are shown",
        "against each other with that boundary stated.",
        "",
        "## Neighbouring cuts",
        "",
    ]
    for name in other_seasons:
        lines.append(
            f"- `../../{name}/{SUBDIRECTORY_NAME}/` — the same cut of the "
            f"{_SEASON_LABELS[name]} stratum."
        )
    if sibling_variety_view:
        lines.append(
            "- `../by_variety/` — this same stratum cut by recorded rice variety."
        )
    lines += [
        "- `../../../by_planting_year/` — a by-planting-year sheet over a *different*,",
        "  much wider core-trial population: every series the eligibility ledger",
        "  resolves at two or more distinct applied-N levels, 1991-2018, seasons not",
        "  separated the same way. Its counts do not reconcile with these, so say",
        "  which population a figure uses whenever you take one out of its directory.",
        "",
        "## Regenerating",
        "",
        "```",
        "conda run -n n_response python \\",
        "  modules/n_response_curve/reporting/"
        "generate_core_trial_season_planting_year_view.py \\",
        f"  --config scriptCONFIG.toml --season {season}"
        + (" \\" if package_argument else ""),
    ]
    if package_argument:
        lines.append(f"  --package {package_argument}")
    lines += [
        "```",
        "",
    ]
    if package_argument:
        lines += [
            "`--package` is not optional as things stand: "
            "`[custom_overlays].package_path`",
            "names a release that is not on disk, and this directory was built from",
            "the package above. Drop the flag once one is promoted there again — and",
            "re-run, because the counts will be that package's, not these.",
            "",
        ]
    lines += [
        "The season view replaces `../../` wholesale but carries non-managed entries",
        "across, and the raw overlay generator preserves `clusters/` whole, so this",
        "directory survives both. Re-run it after either, because the population it",
        "reads can have changed.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Publication
# --------------------------------------------------------------------------


def _is_managed(name: str) -> bool:
    return name in _FIXED_MANAGED_FILENAMES or bool(_YEAR_PLATE_PATTERN.match(name))


def _replace_output_unlocked(
    output_dir: Path,
    season: str,
    cells_by_season: Mapping[str, Sequence[YearCell]],
    limits: SharedAxisLimits,
    *,
    package_argument: str | None,
) -> Mapping[str, Any]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.staging.{uuid.uuid4().hex}")
    backup = output_dir.with_name(f".{output_dir.name}.backup.{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        cells = cells_by_season[season]
        _write_figure(
            _build_panels_figure(season, cells, limits), staging / _PANELS_FILENAME
        )
        _write_figure(
            _build_context_figure(season, cells_by_season, limits),
            staging / _CONTEXT_FILENAME,
        )
        for cell in cells:
            _write_figure(
                _build_single_cell_figure(cell, limits), staging / f"{cell.year}.jpeg"
            )
        summary = build_summary(season, cells_by_season, limits)
        (staging / _SUMMARY_FILENAME).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            _readme(
                season,
                summary,
                package_argument=package_argument,
                sibling_variety_view=(output_dir.parent / "by_variety").is_dir(),
            ),
            encoding="utf-8",
        )

        if output_dir.exists():
            if not output_dir.is_dir() or output_dir.is_symlink():
                raise RuntimeError("Planting-year view output is not a plain directory")
            for entry in output_dir.iterdir():
                if _is_managed(entry.name):
                    continue
                destination = staging / entry.name
                if entry.is_dir() and not entry.is_symlink():
                    shutil.copytree(entry, destination)
                else:
                    shutil.copy2(entry, destination, follow_symlinks=False)
            os.replace(output_dir, backup)
        try:
            os.replace(staging, output_dir)
        except Exception:
            if backup.exists() and not output_dir.exists():
                os.replace(backup, output_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)


def _replace_output(
    output_dir: Path,
    season: str,
    cells_by_season: Mapping[str, Sequence[YearCell]],
    limits: SharedAxisLimits,
    *,
    package_argument: str | None,
) -> Mapping[str, Any]:
    output_dir = _plain_absolute_path(output_dir, label="Planting-year view output")
    with _core_overlay_publication_lock(_publication_container(output_dir)):
        _recover_interrupted_directory_publication(output_dir)
        return _replace_output_unlocked(
            output_dir,
            season,
            cells_by_season,
            limits,
            package_argument=package_argument,
        )


def _configured_package_path(config_path: Path) -> Path | None:
    """The `[custom_overlays].package_path` binding, for the "is it there?" note."""

    with config_path.open("rb") as handle:
        table = tomllib.load(handle).get("custom_overlays")
    if not isinstance(table, dict):
        return None
    raw = table.get("package_path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    package = Path(raw)
    if not package.is_absolute():
        package = config_path.resolve().parent / package
    return package.resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Group one recorded season of the governed core-trial overlay by "
            "recorded planting year without fitting or statistical clustering."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--package",
        type=Path,
        default=None,
        help=(
            "Promoted release package to read the eligibility ledger from. "
            "Defaults to the [custom_overlays].package_path binding in --config."
        ),
    )
    parser.add_argument(
        "--season",
        choices=(*sorted(_SEASON_LABELS), ALL_SEASONS),
        default=ALL_SEASONS,
        help="Recorded season stratum to split by planting year, or all of them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Defaults to clusters/by_season/<season>/by_planting_year/. Only "
            "meaningful for a single --season."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output_dir is not None and args.season == ALL_SEASONS:
        raise SystemExit("--output-dir names one directory, so it needs one --season")
    config_path = args.config.resolve()
    inputs = _load_governed_core_inputs(
        config_path,
        yield_threshold_t_ha=DEFAULT_YIELD_THRESHOLD_T_HA,
        require_yield_selection=False,
        package_path=args.package,
    )
    season_strata = group_governed_records_by_season(
        inputs.records,
        inputs.response_series_uids,
    )
    cells_by_season = split_strata_by_planting_year(season_strata)
    limits = shared_frame(season_strata)

    requested = (
        tuple(season for season in _SEASON_ORDER if season in cells_by_season)
        if args.season == ALL_SEASONS
        else (args.season,)
    )
    missing = [season for season in requested if season not in cells_by_season]
    if missing:
        raise SystemExit(
            f"The governed {SOURCE_NAME} population records no {missing} season "
            f"series; found {sorted(cells_by_season)}"
        )

    package_argument = str(args.package) if args.package else None
    written: dict[str, Any] = {}
    for season in requested:
        output_dir = args.output_dir or (SEASON_ROOT / season / SUBDIRECTORY_NAME)
        summary = _replace_output(
            output_dir,
            season,
            cells_by_season,
            limits,
            package_argument=package_argument,
        )
        written[season] = {
            "output_dir": str(output_dir.resolve()),
            "planting_years": [
                cell["planting_year"] for cell in summary["planting_years"]
            ],
            "series_count": summary["population"]["series_count"],
            "observation_count": summary["population"]["observation_count"],
        }
    configured = _configured_package_path(config_path)
    print(
        json.dumps(
            {
                "configured_package_present": bool(configured and configured.is_dir()),
                "package_argument": package_argument,
                "seasons": written,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
