#!/usr/bin/env python3
"""Split one recorded season of the governed core-trial overlay by rice variety.

This is the by-variety refinement of ``generate_core_trial_season_view.py``: it
takes one season's stratum of that generator's exact population — the governed
21-series/74-observation core-trial set drawn in
``core_trial_data_source_wide.jpeg`` — and groups it again on the variety the
eligibility ledger records. Nothing is clustered and no curve is fitted.

The grouping is honest only if the figure says what variety is confounded with
here, and in the dry stratum the confounding is total: the varieties observed in
more than one series come from one repeated study, and the varieties observed in
exactly one series come from one single-year screen. A difference between two
variety panels is therefore a difference of study, year, and treatment ladder as
much as of genotype. Both figures carry that on their face, computed from the
data rather than asserted, and colour is spent on the replication structure —
repeated versus single-series varieties — rather than on eleven cycled hues that
no reader could separate.

Output is nested at ``clusters/by_season/<season>/by_variety/``, the same shape
LTCCE uses at ``clusters/by_season/ds/by_variety/``. The season view preserves
non-managed entries of ``by_season/`` across its own replacement and the raw
overlay generator preserves ``clusters/`` whole, so this subtree survives both.

Usage:
    conda run -n n_response python \
      modules/n_response_curve/reporting/generate_core_trial_season_variety_view.py \
      --config scriptCONFIG.toml \
      --package WF/z_archive/06_Reports/n_response_full
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.analysis.values import finite_number  # noqa: E402
from n_response_curve.reporting.directory_publication import (  # noqa: E402
    nested_publication_container as _publication_container,
    plain_absolute_path as _plain_absolute_path,
    publication_lock as _core_overlay_publication_lock,
    recover_interrupted_directory_publication as _recover_interrupted_directory_publication,  # noqa: E501
)
from n_response_curve.reporting.figure_axis_frames import (  # noqa: E402
    padded_limits as _padded_limits,
)
from n_response_curve.reporting.figure_captions import (  # noqa: E402
    EXPLORATORY_DIAGNOSTIC_DISCLAIMER,
    reserve_suptitle as _reserve_suptitle,
    wrap_title_lines as _wrap_title_lines,
)
from n_response_curve.reporting.figure_output import (  # noqa: E402
    save_figure_atomically as _save_governed_figure,
)
from n_response_curve.reporting.generate_core_trial_season_view import (  # noqa: E402
    SeasonStratum,
    group_governed_records_by_season,
)
from n_response_curve.reporting.generate_raw_dataset_overlays import (  # noqa: E402
    DEFAULT_YIELD_THRESHOLD_T_HA,
    _load_governed_core_inputs,
)
from n_response_curve.reporting.source_display_names import (  # noqa: E402
    display_source_name,
)

SOURCE_NAME = "core_trial_data"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scriptCONFIG.toml"
DEFAULT_SEASON = "dry"
SEASON_ROOT = (
    PROJECT_ROOT
    / "WF/03_Response_Curves"
    / "literature_extracted_dataset/clusters/by_season"
)

_SEASON_LABELS = {"dry": "dry season", "wet": "wet season"}
_OVERLAY_FILENAME = "variety_overlay.jpeg"
_FACETS_FILENAME = "variety_facets.jpeg"
_SUMMARY_FILENAME = "variety_summary.json"
_MANAGED_FILENAMES = frozenset(
    {"README.md", _SUMMARY_FILENAME, _OVERLAY_FILENAME, _FACETS_FILENAME}
)

# Categorical slots 1 and 2 of the validated default palette. Two hues, not one
# per variety: eleven cycled hues fail every colour-vision separation floor, and
# the quantity worth colouring is the replication structure the varieties fall
# into. Variety identity is carried by direct labels and by panel titles, so it
# is never colour-alone.
_REPEATED_COLOR = "#2a78d6"
_SINGLE_COLOR = "#eb6834"
_CONTEXT_COLOR = "#c2c1ba"
_ZERO_N_LABEL = "zero-N anchor observation"
_UNRECORDED_VARIETY = "(variety unrecorded)"

_OVERLAY_TITLE_WIDTH = 130
_FACETS_TITLE_WIDTH = 152
_PANEL_TITLE_WIDTH = 48


@dataclass(frozen=True)
class _Observation:
    """One finite governed N-yield pair inside the season stratum."""

    response_series_uid: str
    record_uid: str
    n_rate_kg_ha: float
    yield_t_ha: float

    @property
    def is_zero_n(self) -> bool:
        return self.n_rate_kg_ha == 0.0


@dataclass(frozen=True)
class _VarietySeries:
    """One complete governed response series, with its recorded context."""

    response_series_uid: str
    variety: str
    study_ids: tuple[str, ...]
    planting_years: tuple[str, ...]
    observations: tuple[_Observation, ...]

    @property
    def has_zero_n(self) -> bool:
        return any(observation.is_zero_n for observation in self.observations)


@dataclass(frozen=True)
class VarietyStratum:
    """Every governed series in one season that records one rice variety."""

    variety: str
    series: tuple[_VarietySeries, ...]

    @property
    def series_count(self) -> int:
        return len(self.series)

    @property
    def is_repeated(self) -> bool:
        """More than one governed series records this variety in this season."""

        return self.series_count > 1

    @property
    def observations(self) -> tuple[_Observation, ...]:
        return tuple(
            observation
            for series in self.series
            for observation in series.observations
        )

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def series_with_zero_n(self) -> int:
        return sum(1 for series in self.series if series.has_zero_n)

    @property
    def study_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({study for series in self.series for study in series.study_ids})
        )

    @property
    def planting_years(self) -> tuple[str, ...]:
        return tuple(
            sorted({year for series in self.series for year in series.planting_years})
        )

    @property
    def n_range(self) -> tuple[float, float]:
        values = [observation.n_rate_kg_ha for observation in self.observations]
        return min(values), max(values)

    @property
    def color(self) -> str:
        return _REPEATED_COLOR if self.is_repeated else _SINGLE_COLOR


def _recorded_values(records: Sequence[Mapping[str, Any]], column: str) -> tuple[str, ...]:
    """Distinct nonempty values of one descriptive ledger column, sorted."""

    return tuple(
        sorted({
            value
            for value in (str(record.get(column) or "").strip() for record in records)
            if value
        })
    )


def _series_observations(
    records: Sequence[Mapping[str, Any]],
) -> tuple[_Observation, ...]:
    """The finite N-yield pairs of one series, ordered along the applied-N ladder."""

    observations: list[_Observation] = []
    for record in records:
        n_rate = finite_number(record.get("n_rate_kg_ha"))
        yield_value = finite_number(record.get("yield_t_ha"))
        if n_rate is None or yield_value is None:
            raise ValueError(
                "A governed season-stratum record carries a nonfinite observation"
            )
        observations.append(
            _Observation(
                response_series_uid=str(record.get("response_series_uid") or ""),
                record_uid=str(record.get("record_uid") or ""),
                n_rate_kg_ha=float(n_rate),
                yield_t_ha=float(yield_value),
            )
        )
    return tuple(
        sorted(
            observations,
            key=lambda observation: (observation.n_rate_kg_ha, observation.record_uid),
        )
    )


def group_stratum_by_variety(stratum: SeasonStratum) -> tuple[VarietyStratum, ...]:
    """Place each complete governed series of *stratum* into exactly one variety.

    A series that carries more than one recorded variety, or none, fails closed
    rather than being split between panels or silently pooled: the series is the
    unit the response curve is drawn on, and a mixed-variety series would make
    every per-variety count in the summary wrong.

    Strata are ordered by series count descending, then by variety name, so the
    replicated variety leads the facet grid and the ordering does not depend on
    ledger row order.
    """

    records_by_series: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for record in stratum.records:
        series_uid = str(record.get("response_series_uid") or "").strip()
        records_by_series[series_uid].append(record)

    series_by_variety: dict[str, list[_VarietySeries]] = collections.defaultdict(list)
    for series_uid in stratum.series_uids:
        members = records_by_series.get(series_uid, [])
        if not members:
            raise ValueError(
                f"Governed response series {series_uid!r} has no retained records"
            )
        varieties = _recorded_values(members, "rice_variety")
        if not varieties:
            raise ValueError(
                f"Governed response series {series_uid!r} records no rice variety"
            )
        if len(varieties) > 1:
            raise ValueError(
                f"Governed response series {series_uid!r} carries more than one "
                f"recorded rice variety: {list(varieties)}"
            )
        series_by_variety[varieties[0]].append(
            _VarietySeries(
                response_series_uid=series_uid,
                variety=varieties[0],
                study_ids=_recorded_values(members, "study_id"),
                planting_years=_recorded_values(members, "planting_year"),
                observations=_series_observations(members),
            )
        )

    strata = tuple(
        VarietyStratum(
            variety=variety,
            series=tuple(
                sorted(series, key=lambda entry: entry.response_series_uid)
            ),
        )
        for variety, series in series_by_variety.items()
    )
    placed = sum(entry.series_count for entry in strata)
    if placed != len(stratum.series_uids):
        raise ValueError(
            f"Variety grouping placed {placed} of {len(stratum.series_uids)} "
            "governed series"
        )
    return tuple(
        sorted(strata, key=lambda entry: (-entry.series_count, entry.variety))
    )


def _stratum_frame(
    strata: Sequence[VarietyStratum],
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """One padded frame pinned from the whole season stratum, before subsetting.

    Every panel is drawn inside it, so a single-series variety panel is directly
    comparable with the eleven beside it and with ``../<season>.jpeg``.
    """

    observations = [
        observation for stratum in strata for observation in stratum.observations
    ]
    n_values = [observation.n_rate_kg_ha for observation in observations]
    yield_values = [observation.yield_t_ha for observation in observations]
    return (
        _padded_limits((min(n_values), max(n_values))),
        _padded_limits((min(yield_values), max(yield_values))),
    )


def _draw_series(
    axes: Any,
    series: _VarietySeries,
    *,
    color: str,
    label: str | None,
    zero_n_label: str | None,
) -> None:
    """Draw one response series: a connecting aid, then its observed points."""

    axes.plot(
        [observation.n_rate_kg_ha for observation in series.observations],
        [observation.yield_t_ha for observation in series.observations],
        color=color,
        linewidth=2,
        alpha=0.85,
        label=label or "_nolegend_",
        zorder=2,
    )
    mineral = [
        observation for observation in series.observations if not observation.is_zero_n
    ]
    zero_n = [
        observation for observation in series.observations if observation.is_zero_n
    ]
    if mineral:
        axes.scatter(
            [observation.n_rate_kg_ha for observation in mineral],
            [observation.yield_t_ha for observation in mineral],
            color=color,
            s=34,
            zorder=3,
            label="_nolegend_",
        )
    if zero_n:
        # Colour already carries the replication structure, so the zero-N anchor
        # is separated by shape and fill instead of by a third hue.
        axes.scatter(
            [observation.n_rate_kg_ha for observation in zero_n],
            [observation.yield_t_ha for observation in zero_n],
            marker="s",
            s=58,
            facecolors="white",
            edgecolors=color,
            linewidths=1.6,
            zorder=4,
            label=zero_n_label or "_nolegend_",
        )


def _draw_context(axes: Any, strata: Sequence[VarietyStratum], *, exclude: str) -> None:
    """Draw every other variety's observations behind one facet, unconnected."""

    others = [
        observation
        for stratum in strata
        if stratum.variety != exclude
        for observation in stratum.observations
    ]
    if not others:
        return
    axes.scatter(
        [observation.n_rate_kg_ha for observation in others],
        [observation.yield_t_ha for observation in others],
        color=_CONTEXT_COLOR,
        s=16,
        alpha=0.9,
        zorder=1,
        label="_nolegend_",
    )


def _composition(strata: Sequence[VarietyStratum]) -> dict[str, Any]:
    """The counts every disclosure line is computed from."""

    repeated = [stratum for stratum in strata if stratum.is_repeated]
    single = [stratum for stratum in strata if not stratum.is_repeated]
    return {
        "variety_count": len(strata),
        "series_count": sum(stratum.series_count for stratum in strata),
        "observation_count": sum(stratum.observation_count for stratum in strata),
        "repeated_varieties": tuple(stratum.variety for stratum in repeated),
        "single_series_varieties": tuple(stratum.variety for stratum in single),
        "repeated_studies": tuple(
            sorted({study for stratum in repeated for study in stratum.study_ids})
        ),
        "single_studies": tuple(
            sorted({study for stratum in single for study in stratum.study_ids})
        ),
        "repeated_years": tuple(
            sorted({year for stratum in repeated for year in stratum.planting_years})
        ),
        "single_years": tuple(
            sorted({year for stratum in single for year in stratum.planting_years})
        ),
        "varieties_with_zero_n": tuple(
            stratum.variety for stratum in strata if stratum.series_with_zero_n
        ),
        "series_with_zero_n": sum(stratum.series_with_zero_n for stratum in strata),
    }


def _join(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "none"


def _plural(count: int, singular: str, plural: str) -> str:
    """``"1 variety"`` / ``"10 varieties"`` — the disclosures read as prose."""

    return f"{count} {singular if count == 1 else plural}"


def _span(values: Sequence[str]) -> str:
    if not values:
        return "unrecorded"
    if len(values) == 1:
        return values[0]
    return f"{values[0]}-{values[-1]}"


def _disclosures(season: str, strata: Sequence[VarietyStratum]) -> tuple[str, ...]:
    """The interpretation boundary, stated from the counts rather than asserted.

    Every clause is derived: nothing here names a study, a variety or a year that
    the grouping did not just find, so the lines stay true if the governed
    population changes.
    """

    facts = _composition(strata)
    repeated = facts["repeated_varieties"]
    single = facts["single_series_varieties"]
    lines = [
        f"source={display_source_name(SOURCE_NAME)} — governed observed series of "
        f"the {_SEASON_LABELS[season]} stratum, grouped by recorded rice variety",
        f"varieties={facts['variety_count']}; series={facts['series_count']}; "
        f"observations={facts['observation_count']}; "
        f"panels and overlay share one frame pinned from the whole stratum",
        EXPLORATORY_DIAGNOSTIC_DISCLAIMER,
    ]
    if repeated and single:
        lines.append(
            "variety is confounded with study and planting year: the "
            f"{_plural(len(repeated), 'variety', 'varieties')} observed in more "
            f"than one series {'comes' if len(repeated) == 1 else 'come'} from "
            f"study {_join(facts['repeated_studies'])} "
            f"({_span(facts['repeated_years'])}), and the "
            f"{_plural(len(single), 'variety', 'varieties')} observed in exactly "
            f"one series {'comes' if len(single) == 1 else 'come'} from study "
            f"{_join(facts['single_studies'])} "
            f"({_span(facts['single_years'])}); a step between two varieties here "
            "is a change of study, year and treatment ladder as much as of genotype"
        )
    if single:
        lines.append(
            f"{len(single)} of {facts['variety_count']} varieties are a single "
            "response series with no within-variety replication, so nothing here "
            "supports a between-variety claim"
        )
    if facts["varieties_with_zero_n"]:
        lines.append(
            f"only {_join(facts['varieties_with_zero_n'])} carries a zero-N anchor "
            f"({facts['series_with_zero_n']} of {facts['series_count']} series); "
            "response above zero N is undefined for every other variety, whose "
            "ladders begin above zero"
        )
    else:
        lines.append(
            "no series in this stratum carries a zero-N anchor, so response above "
            "zero N is undefined throughout"
        )
    lines.append(
        "connecting lines join a series' own observed points as a visual aid; "
        "no curve is fitted, interpolated or extrapolated"
    )
    return tuple(lines)


def _legend_handles(strata: Sequence[VarietyStratum]) -> tuple[list[Any], list[str]]:
    from matplotlib.lines import Line2D

    facts = _composition(strata)
    handles: list[Any] = []
    labels: list[str] = []
    if facts["repeated_varieties"]:
        handles.append(
            Line2D([], [], color=_REPEATED_COLOR, marker="o", linewidth=2, markersize=6)
        )
        labels.append("variety observed in more than one series")
    if facts["single_series_varieties"]:
        handles.append(
            Line2D([], [], color=_SINGLE_COLOR, marker="o", linewidth=2, markersize=6)
        )
        labels.append("variety observed in exactly one series")
    handles.append(
        Line2D(
            [],
            [],
            color="#444444",
            marker="s",
            markerfacecolor="white",
            markersize=7,
            linestyle="none",
        )
    )
    labels.append(_ZERO_N_LABEL)
    handles.append(
        Line2D([], [], color=_CONTEXT_COLOR, marker="o", markersize=5, linestyle="none")
    )
    labels.append("other varieties of the same stratum (context)")
    return handles, labels


def _spread(
    values: Sequence[float], *, gap: float, lower: float, upper: float
) -> list[float]:
    """Push overlapping label positions apart without leaving *lower*..*upper*.

    A single forward pass alone piles the whole run against the top of the frame
    once the anchors are dense, so the sweep is forward (separate), backward
    (pull the run back under the ceiling), then forward again (re-seat it above
    the floor). Returned in the caller's order, not sorted order.
    """

    order = sorted(range(len(values)), key=lambda index: values[index])
    placed = [values[index] for index in order]
    for position in range(1, len(placed)):
        placed[position] = max(placed[position], placed[position - 1] + gap)
    if placed:
        placed[-1] = min(placed[-1], upper)
    for position in range(len(placed) - 2, -1, -1):
        placed[position] = min(placed[position], placed[position + 1] - gap)
    if placed:
        placed[0] = max(placed[0], lower)
    for position in range(1, len(placed)):
        placed[position] = max(placed[position], placed[position - 1] + gap)
    result = [0.0] * len(values)
    for position, index in enumerate(order):
        result[index] = placed[position]
    return result


def _place_labels(
    axes: Any,
    anchors: Sequence[tuple[float, float, str, str]],
    *,
    y_limits: tuple[float, float],
) -> None:
    """Name every variety in a gutter to the right of the frame, on a leader line.

    Identity must not be colour-alone, and eleven labels laid on the frame itself
    collide with the data and with each other. The labels therefore sit outside
    the axes — the frame stays exactly the stratum's own, so this figure remains
    comparable with ``../<season>.jpeg`` and with the facet grid — and the figure
    reserves the gutter width instead of letting the text run off the sheet.
    """

    if not anchors:
        return
    lower, upper = y_limits
    span = upper - lower
    gap = min(span * 0.045, span / max(len(anchors), 1) * 0.94)
    label_y = _spread(
        [anchor[1] for anchor in anchors], gap=gap, lower=lower, upper=upper
    )
    x_lower, x_upper = axes.get_xlim()
    gutter_x = x_upper + (x_upper - x_lower) * 0.035
    for (x_value, y_value, text, color), target in zip(anchors, label_y, strict=True):
        axes.annotate(
            text,
            xy=(x_value, y_value),
            xytext=(gutter_x, target),
            color=color,
            fontsize=8.5,
            va="center",
            ha="left",
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.8,
                "alpha": 0.6,
                "shrinkA": 0.0,
                "shrinkB": 3.0,
            },
        )


def _build_overlay_figure(season: str, strata: Sequence[VarietyStratum]) -> Any:
    from matplotlib import pyplot as plt

    x_limits, y_limits = _stratum_frame(strata)
    if x_limits is None or y_limits is None:
        raise ValueError("The season stratum has no finite frame to draw in")
    title = _wrap_title_lines(
        (
            *_disclosures(season, strata),
            "each line is one response series; the gutter names each variety "
            "and how many series it contributes",
        ),
        width=_OVERLAY_TITLE_WIDTH,
    )
    figure_height = 9.5
    figure = plt.figure(figsize=(15.0, figure_height))
    try:
        # constrained_layout sizes the axes to the artists it knows about, and it
        # knows nothing about the label gutter, so the geometry is placed by hand:
        # the suptitle's own line count sets the top, and the axes stops at 0.72
        # of the width to leave the gutter its room.
        figure.suptitle(title, fontsize=9, y=0.995, va="top")
        title_inches = ((title.count("\n") + 1) * 9 * 1.25 + 20.0) / 72.0
        top = 1.0 - title_inches / figure_height
        axes = figure.add_axes((0.052, 0.075, 0.668, top - 0.075))

        anchors: list[tuple[float, float, str, str]] = []
        for stratum in strata:
            for series in stratum.series:
                _draw_series(
                    axes,
                    series,
                    color=stratum.color,
                    label=None,
                    zero_n_label=None,
                )
            # One label per variety, anchored at the variety's own rightmost
            # observation: anchoring on an arbitrary member series instead drags
            # a leader line for a repeated variety back across the whole frame.
            anchor = max(
                stratum.observations,
                key=lambda observation: (
                    observation.n_rate_kg_ha,
                    observation.yield_t_ha,
                ),
            )
            anchors.append(
                (
                    anchor.n_rate_kg_ha,
                    anchor.yield_t_ha,
                    f"{stratum.variety} "
                    f"({_plural(stratum.series_count, 'series', 'series')})",
                    stratum.color,
                )
            )
        axes.set_xlim(*x_limits)
        axes.set_ylim(*y_limits)
        _place_labels(axes, anchors, y_limits=y_limits)
        axes.set_xlabel("Applied N (kg N/ha)")
        axes.set_ylabel("Grain yield (t/ha)")
        axes.grid(True, color="#e6e5e0", linewidth=0.7, zorder=0)
        axes.set_axisbelow(True)
        handles, labels = _legend_handles(strata)
        axes.legend(
            handles[:-1],
            labels[:-1],
            loc="lower right",
            fontsize=8,
            frameon=True,
        )
        return figure
    except Exception:
        plt.close(figure)
        raise


def _panel_title(stratum: VarietyStratum) -> str:
    low, high = stratum.n_range
    anchored = (
        f"{stratum.series_with_zero_n} of {stratum.series_count} series zero-N anchored"
        if stratum.series_with_zero_n
        else "no zero-N anchor"
    )
    return _wrap_title_lines(
        (
            stratum.variety,
            f"series={stratum.series_count}; observations={stratum.observation_count}; "
            f"N={low:g}-{high:g} kg/ha",
            f"study={_join(stratum.study_ids)}; "
            f"planting year={_span(stratum.planting_years)}",
            anchored,
        ),
        width=_PANEL_TITLE_WIDTH,
    )


def _build_facets_figure(season: str, strata: Sequence[VarietyStratum]) -> Any:
    from matplotlib import pyplot as plt

    x_limits, y_limits = _stratum_frame(strata)
    columns = 4
    rows = -(-(len(strata) + 1) // columns)
    figure, axes_grid = plt.subplots(
        rows,
        columns,
        figsize=(4.6 * columns, 4.3 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    try:
        panels = [axes for row in axes_grid for axes in row]
        for axes, stratum in zip(panels, strata, strict=False):
            _draw_context(axes, strata, exclude=stratum.variety)
            for series in stratum.series:
                _draw_series(
                    axes,
                    series,
                    color=stratum.color,
                    label=None,
                    zero_n_label=None,
                )
            axes.set_title(_panel_title(stratum), fontsize=8)
            axes.grid(True, color="#e6e5e0", linewidth=0.7, zorder=0)
            axes.set_axisbelow(True)
            if x_limits is not None:
                axes.set_xlim(*x_limits)
            if y_limits is not None:
                axes.set_ylim(*y_limits)
        for axes in panels[len(strata):]:
            axes.set_axis_off()
        for axes in axes_grid[-1]:
            axes.set_xlabel("Applied N (kg N/ha)")
        for row in axes_grid:
            row[0].set_ylabel("Grain yield (t/ha)")
        # A shared x label only reaches the bottom row; a panel switched off
        # above an empty slot would leave the column above it unlabelled.
        for column_index in range(columns):
            for row_index in range(rows - 1, -1, -1):
                axes = axes_grid[row_index][column_index]
                if axes.axison:
                    axes.set_xlabel("Applied N (kg N/ha)")
                    axes.tick_params(labelbottom=True)
                    break
        handles, labels = _legend_handles(strata)
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.004),
            ncol=len(labels),
            fontsize=9,
            frameon=False,
        )
        _reserve_suptitle(
            figure,
            _wrap_title_lines(
                (
                    *_disclosures(season, strata),
                    "one panel per recorded variety, ordered by series count; grey "
                    "points are the rest of the stratum, drawn unconnected for "
                    "position only",
                ),
                width=_FACETS_TITLE_WIDTH,
            ),
            legend_strip=0.045,
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


def _summary(season: str, strata: Sequence[VarietyStratum]) -> dict[str, Any]:
    facts = _composition(strata)
    return {
        "source_name": SOURCE_NAME,
        "season": season,
        "method": "recorded_variety_stratification_within_recorded_season",
        "interpretation": (
            "Recorded rice-variety groups of the complete governed response "
            "series in one recorded season stratum; not an unsupervised cluster "
            "analysis and no curve is fitted. Variety is confounded with study, "
            "planting year and treatment ladder."
        ),
        "population": {
            "variety_count": facts["variety_count"],
            "series_count": facts["series_count"],
            "observation_count": facts["observation_count"],
            "series_with_zero_n_anchor": facts["series_with_zero_n"],
            "repeated_varieties": list(facts["repeated_varieties"]),
            "single_series_varieties": list(facts["single_series_varieties"]),
        },
        "figures": {
            "overlay": _OVERLAY_FILENAME,
            "facets": _FACETS_FILENAME,
        },
        "varieties": [
            {
                "rice_variety": stratum.variety,
                "series_count": stratum.series_count,
                "observation_count": stratum.observation_count,
                "series_with_zero_n_anchor": stratum.series_with_zero_n,
                "n_rate_range_kg_ha": list(stratum.n_range),
                "study_ids": list(stratum.study_ids),
                "planting_years": list(stratum.planting_years),
                "response_series_uids": [
                    series.response_series_uid for series in stratum.series
                ],
            }
            for stratum in strata
        ],
    }


def _readme(
    season: str,
    strata: Sequence[VarietyStratum],
    summary: Mapping[str, Any],
    *,
    package_argument: str | None,
) -> str:
    facts = _composition(strata)
    population = summary["population"]
    command = [
        "`conda run -n n_response python \\",
        "  modules/n_response_curve/reporting/generate_core_trial_season_variety_view.py \\",
        f"  --config scriptCONFIG.toml --season {season}",
    ]
    if package_argument:
        command[-1] += " \\"
        command.append(f"  --package {package_argument}`")
    else:
        command[-1] += "`"

    lines = [
        f"# `{SOURCE_NAME}` — {_SEASON_LABELS[season]} stratum grouped by rice variety",
        "",
        f"{EXPLORATORY_DIAGNOSTIC_DISCLAIMER}.",
        "",
        f"This regroups the exact population drawn in `../{season}.jpeg` on the",
        "rice variety the promoted release's eligibility ledger records. It is not",
        "an unsupervised cluster analysis, and no series is split between panels.",
        "",
        "## Population",
        "",
        f"- {population['variety_count']} recorded varieties, "
        f"{population['series_count']} governed response series, "
        f"{population['observation_count']} finite observed N-yield pairs.",
        f"- {population['series_with_zero_n_anchor']} of "
        f"{population['series_count']} series observed zero applied N.",
        f"- Observed in more than one series: {_join(facts['repeated_varieties'])}.",
        f"- Observed in exactly one series: "
        f"{len(facts['single_series_varieties'])} varieties.",
        "",
        "## Outputs",
        "",
        f"- `{_OVERLAY_FILENAME}` — every series on one frame, coloured by whether",
        "  its variety repeats, with each variety direct-labelled.",
        f"- `{_FACETS_FILENAME}` — one panel per variety on that same frame, with",
        "  the rest of the stratum behind it in grey for position.",
        f"- `{_SUMMARY_FILENAME}` — per-variety counts, ladders, studies and years.",
        "",
        "## Interpretation boundary",
        "",
    ]
    if facts["repeated_varieties"] and facts["single_series_varieties"]:
        lines += [
            "- Variety is confounded with study and planting year. The "
            f"{_plural(len(facts['repeated_varieties']), 'variety', 'varieties')} "
            "observed in more than one series "
            f"{'comes' if len(facts['repeated_varieties']) == 1 else 'come'} from "
            f"study {_join(facts['repeated_studies'])} "
            f"({_span(facts['repeated_years'])}); the "
            f"{_plural(len(facts['single_series_varieties']), 'variety', 'varieties')}"
            " observed in exactly one series "
            f"{'comes' if len(facts['single_series_varieties']) == 1 else 'come'} "
            f"from study {_join(facts['single_studies'])} "
            f"({_span(facts['single_years'])}). A step between panels is a change "
            "of study, year and treatment ladder as much as of genotype.",
        ]
    lines += [
        f"- {len(facts['single_series_varieties'])} of {facts['variety_count']} "
        "varieties are one series each, so there is no within-variety replication "
        "and no panel supports a between-variety claim.",
        f"- Zero-N anchors: {_join(facts['varieties_with_zero_n'])}. Response above "
        "zero N is undefined for every other variety.",
        "- Connecting lines are a visual aid over a series' own observed points.",
        "",
        "## Regenerating",
        "",
    ]
    lines += command
    lines += [
        "",
        "The season view replaces `../` wholesale but carries non-managed entries",
        "across, and the raw overlay generator preserves `clusters/` whole, so this",
        "directory survives both. Re-run it after either, because the population it",
        "reads can have changed.",
        "",
    ]
    return "\n".join(lines)


def _replace_output_unlocked(
    output_dir: Path,
    season: str,
    strata: Sequence[VarietyStratum],
    *,
    package_argument: str | None,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.staging.{uuid.uuid4().hex}")
    backup = output_dir.with_name(f".{output_dir.name}.backup.{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        _write_figure(_build_overlay_figure(season, strata), staging / _OVERLAY_FILENAME)
        _write_figure(_build_facets_figure(season, strata), staging / _FACETS_FILENAME)
        summary = _summary(season, strata)
        (staging / _SUMMARY_FILENAME).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            _readme(season, strata, summary, package_argument=package_argument),
            encoding="utf-8",
        )

        if output_dir.exists():
            if not output_dir.is_dir() or output_dir.is_symlink():
                raise RuntimeError("Variety-view output is not a plain directory")
            for entry in output_dir.iterdir():
                if entry.name in _MANAGED_FILENAMES:
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
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)


def _replace_output(
    output_dir: Path,
    season: str,
    strata: Sequence[VarietyStratum],
    *,
    package_argument: str | None,
) -> None:
    output_dir = _plain_absolute_path(output_dir, label="Variety-view output")
    with _core_overlay_publication_lock(_publication_container(output_dir)):
        _recover_interrupted_directory_publication(output_dir)
        _replace_output_unlocked(
            output_dir, season, strata, package_argument=package_argument
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Group one recorded season of the governed core-trial overlay by "
            "recorded rice variety without fitting or statistical clustering."
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
        choices=sorted(_SEASON_LABELS),
        default=DEFAULT_SEASON,
        help="Recorded season stratum to split by variety.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to clusters/by_season/<season>/by_variety/.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config.resolve()
    output_dir = args.output_dir or (SEASON_ROOT / args.season / "by_variety")
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
    stratum = season_strata.get(args.season)
    if stratum is None:
        raise SystemExit(
            f"The governed {SOURCE_NAME} population records no {args.season} season "
            f"series; found {sorted(season_strata)}"
        )
    strata = group_stratum_by_variety(stratum)
    _replace_output(
        output_dir,
        args.season,
        strata,
        package_argument=str(args.package) if args.package else None,
    )
    facts = _composition(strata)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "season": args.season,
                "variety_count": facts["variety_count"],
                "series_count": facts["series_count"],
                "observation_count": facts["observation_count"],
                "series_with_zero_n_anchor": facts["series_with_zero_n"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
