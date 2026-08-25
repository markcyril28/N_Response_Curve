#!/usr/bin/env python3
"""Write an IR8-only, DS-season, by-planting-year breakdown of LTCCE N response.

Standalone companion to `generate_response_curve_season_clusters.py`'s own
`by_season/ds/by_variety/1966_ir8.jpeg` -- the DS season's 935 cluster-eligible
trajectories decomposed by the source `Variety` column, of which IR8 is one
level carrying 88 trajectories, 1968-1990. That figure pools every planting
year of IR8 into one panel; this script splits it back out, one panel per
year, on the same shared axis frame the season module uses. It also draws a
companion annual-trend line chart, in the same spirit as
`by_season/ds/by_planting_year/annotated/annual_trend.jpeg`, restricted to IR8.

IR8 turns out to carry exactly four replicate trajectories in every recorded
DS planting year from 1968 to 1990 except 1984 (unrecorded), all from the same
design (Split-split-plot) and site (IRRI-B5-B8) -- so unlike the whole-season
`by_planting_year/` folder, no stratum-size argument is needed to justify
showing individual years here; the panel grid below needs no banding.

Exploratory diagnostic, outside the governed release inventory (ANA-11): no
curve is fitted, and nothing here is a governed analysis family.

Usage:
    conda run -n n_response python \\
      modules/n_response_curve/reporting/generate_ir8_planting_year_view.py \\
      --config scriptCONFIG.toml
"""

from __future__ import annotations

import argparse
import collections
import math
import shutil
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.reporting.directory_publication import (  # noqa: E402
    nested_publication_container as _nested_publication_container,
    plain_absolute_path as _plain_absolute_path,
    promote_directory as _promote_directory,
    publication_lock as _core_overlay_publication_lock,
)
from n_response_curve.reporting.figure_axis_frames import (  # noqa: E402
    shared_axis_limits as _shared_axis_limits,
)
from n_response_curve.reporting.figure_captions import (  # noqa: E402
    EXPLORATORY_DIAGNOSTIC_DISCLAIMER as _DISCLAIMER,
    reserve_suptitle as _reserve_suptitle,
    wrap_title_lines as _wrap_title_lines,
)
from n_response_curve.reporting.figure_output import (  # noqa: E402
    save_figure_atomically as _save_figure,
)
from n_response_curve.reporting.planting_year_axis import (  # noqa: E402
    contiguous_runs as _contiguous_runs,
    decade_spans as _decade_spans,
)
from n_response_curve.reporting.reference_sheet_layout import (  # noqa: E402
    AXIS_LABEL_FONT_SIZE as _AXIS_LABEL_FONT_SIZE,
    CAPTION_COLUMN_CHARS as _CAPTION_COLUMN_CHARS,
    CAPTION_COLUMN_OFFSET as _CAPTION_COLUMN_OFFSET,
    CAPTION_FONT_SIZE as _CAPTION_FONT_SIZE,
    ERA_MARKERS as _ERA_MARKERS,
    HEADLINE_FONT_SIZE as _HEADLINE_FONT_SIZE,
    HOST_ROW_IN as _HOST_ROW_IN,
    HOST_TREND_FRACTION as _HOST_TREND_FRACTION,
    INSET_BOTTOM_FRACTION as _INSET_BOTTOM_FRACTION,
    INSET_LEGEND_IN as _INSET_LEGEND_IN,
    INSET_READOUT_FONT_SIZE as _INSET_READOUT_FONT_SIZE,
    INSET_TICK_FONT_SIZE as _INSET_TICK_FONT_SIZE,
    INSET_TITLE_FONT_SIZE as _INSET_TITLE_FONT_SIZE,
    INSET_TOP_FRACTION as _INSET_TOP_FRACTION,
    LEGEND_FONT_SIZE as _LEGEND_FONT_SIZE,
    NARROW_PANEL_IN as _NARROW_PANEL_IN,
    PANEL_GUTTER_IN as _PANEL_GUTTER_IN,
    SHEET_BOTTOM_IN as _SHEET_BOTTOM_IN,
    SHEET_LEFT_IN as _SHEET_LEFT_IN,
    SHEET_RIGHT_IN as _SHEET_RIGHT_IN,
    SHEET_WIDTH_IN as _SHEET_WIDTH_IN,
    TICK_FONT_SIZE as _TICK_FONT_SIZE,
    TREND_GAP_IN as _TREND_GAP_IN,
    TREND_LEGEND_IN as _TREND_LEGEND_IN,
    TREND_ROW_IN as _TREND_ROW_IN,
    TREND_XAXIS_IN as _TREND_XAXIS_IN,
    ZERO_N_TREATMENT_CLASS as _ZERO_N_TREATMENT_CLASS,
)
from n_response_curve.reporting.generate_response_curve_season_clusters import (  # noqa: E402
    _draw_overlay_on_axes,
    _treatment_class_colours,
)
from n_response_curve.reporting.source_config_spec import (  # noqa: E402
    load_source_spec as _load_source_spec_for,
)
from n_response_curve.reporting.response_curve_clusters import (  # noqa: E402
    RESPONSE_TYPE_FEATURES,
    centroid_summary,
    read_ltcce_contexts,
    subset_overlay,
)
from n_response_curve.reporting.response_curve_season_clusters import (  # noqa: E402
    AnnualRecord,
    annual_ladder_eras,
    build_season_clustering,
    cluster_year_range,
    decade_band,
    format_shares,
    ladder_text,
    season_label,
)
from n_response_curve.reporting.source_dataset_overlays import (  # noqa: E402
    read_source_dataset_overlay,
)

TARGET_VARIETY = "IR8"
TARGET_SEASON = "DS"
GAP_TOKEN = "—"
SOURCE_NAME = "ltcce"

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "WF/04_Response_Curves/z_n_response_full/ltcce"
    / "clusters/by_season/ds/by_variety/ir8_by_planting_year"
)


def _load_source_spec(config_path: Path) -> tuple[Path, str]:
    return _load_source_spec_for(
        config_path,
        SOURCE_NAME,
        relative_root=PROJECT_ROOT,
        resolve_path=False,
        missing_sources_message="The configuration must contain a [sources] table",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "scriptCONFIG.toml",
        help="Path to scriptCONFIG.toml (read only for [sources.ltcce]).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory (replaced atomically).",
    )
    return parser.parse_args()


def _year_groups(result, season: str, variety: str) -> dict[int, tuple[str, ...]]:
    profile = result.profiles.get(season)
    if profile is None:
        raise ValueError(f"No {season!r} season profile in this clustering result")
    grouped: dict[int, list[str]] = collections.defaultdict(list)
    for trajectory_id in profile.trajectory_ids:
        context = result.contexts.get(trajectory_id)
        if context is None or context.variety.strip() != variety or context.year <= 0:
            continue
        grouped[int(context.year)].append(trajectory_id)
    return {year: tuple(sorted(ids)) for year, ids in sorted(grouped.items())}


def _dominant_ladder(ids: Sequence[str], result) -> tuple[str, bool]:
    counts = collections.Counter(
        ladder_text(result.features[tid].ladder) for tid in ids if tid in result.features
    )
    if not counts:
        return "unknown", False
    return counts.most_common(1)[0][0], len(counts) > 1


def _ladder_shares(ids: Sequence[str], result) -> dict[str, float]:
    counts = collections.Counter(
        ladder_text(result.features[tid].ladder) for tid in ids if tid in result.features
    )
    total = sum(counts.values())
    if not total:
        return {}
    return {
        ladder: count / total for ladder, count in counts.most_common()
    }


def _decade_groups(
    year_groups: dict[int, tuple[str, ...]]
) -> dict[str, tuple[str, ...]]:
    """Band IR8's planting years into calendar decades, the whole-season 'why'."""

    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for year, ids in year_groups.items():
        band = decade_band(str(year))
        if band:
            grouped[band].extend(ids)
    return {decade: tuple(sorted(ids)) for decade, ids in sorted(grouped.items())}


def _year_means(ids: Sequence[str], result) -> tuple[dict[str, float], dict[str, float]]:
    """Replicate mean and standard error for one year's IR8 trajectories.

    Every IR8 planting year is a single design/site/ladder context carrying (at
    most) four replicates, so this is a plain replicate mean -- not the
    multi-context weighting `annual_response_records` uses for the whole
    season, which would be indistinguishable from it here anyway.
    """

    matrix = np.asarray(
        [
            [getattr(result.features[tid], name) for name in RESPONSE_TYPE_FEATURES]
            for tid in ids
            if tid in result.features
        ],
        dtype=float,
    )
    means = matrix.mean(axis=0)
    if matrix.shape[0] > 1:
        errors = matrix.std(axis=0, ddof=1) / math.sqrt(matrix.shape[0])
    else:
        errors = np.zeros(matrix.shape[1])
    return (
        dict(zip(RESPONSE_TYPE_FEATURES, means, strict=True)),
        dict(zip(RESPONSE_TYPE_FEATURES, errors, strict=True)),
    )


def _write_year_panel_grid(
    result,
    overlay,
    year_groups: dict[int, tuple[str, ...]],
    limits,
    destination: Path,
) -> None:
    from matplotlib import pyplot as plt

    years = sorted(year_groups)
    total = sum(len(ids) for ids in year_groups.values())
    columns = 5
    rows = math.ceil(len(years) / columns)
    figure, axes_grid = plt.subplots(
        rows,
        columns,
        figsize=(6.0 * columns, 6.4 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_all = list(np.atleast_1d(axes_grid).ravel())
    axes_list = axes_all[: len(years)]
    for spare in axes_all[len(years) :]:
        spare.set_visible(False)
    try:
        for year, axes in zip(years, axes_list, strict=True):
            ids = year_groups[year]
            _draw_overlay_on_axes(subset_overlay(overlay, ids), axes)
            limits.apply(axes)
            ladder, mixed = _dominant_ladder(ids, result)
            means, _ = _year_means(ids, result)
            axes.set_title(
                f"{year}{' (mixed ladder)' if mixed else ''}\n"
                f"n={len(ids)}; {ladder} kg N/ha\n"
                f"zero-N={means['yield_at_zero_n_t_ha']:.2f} t/ha; "
                f"response={means['response_above_zero_n_t_ha']:.2f} t/ha",
                fontsize=9,
            )
        for position, axes in enumerate(axes_list):
            if position // columns == rows - 1 or position + columns >= len(axes_list):
                axes.set_xlabel("Applied N (kg N/ha)")
            if position % columns == 0:
                axes.set_ylabel("Grain yield (t/ha)")

        handles, labels = axes_list[0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False
        )
        _reserve_suptitle(
            figure,
            _wrap_title_lines(
                [
                    f"source={SOURCE_NAME} — {season_label(TARGET_SEASON)}, "
                    f"{TARGET_VARIETY}: every planting year on its own panel",
                    f"{total} cluster-eligible trajectories, {years[0]}-{years[-1]} "
                    f"({len(years)} of {years[-1] - years[0] + 1} calendar years "
                    "recorded); every panel shares this frame with "
                    "`../1966_ir8.jpeg`, which pools them",
                    "each panel is a recorded stratum, not a cluster: nothing was "
                    "clustered to produce it",
                    "variety is confounded with the calendar period it was grown "
                    "in; a difference between years here is not a genetic effect, "
                    "and every year but one is the same design and site",
                    _DISCLAIMER,
                ],
                width=180,
            ),
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


def _write_decade_panel_grid(
    result,
    overlay,
    decade_groups: dict[str, tuple[str, ...]],
    year_groups: dict[int, tuple[str, ...]],
    limits,
    destination: Path,
) -> None:
    from matplotlib import pyplot as plt

    decades = sorted(decade_groups)
    total = sum(len(ids) for ids in decade_groups.values())
    columns = min(2, len(decades))
    rows = math.ceil(len(decades) / columns)
    figure, axes_grid = plt.subplots(
        rows,
        columns,
        figsize=(8.0 * columns, 7.0 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_all = list(np.atleast_1d(axes_grid).ravel())
    axes_list = axes_all[: len(decades)]
    for spare in axes_all[len(decades) :]:
        spare.set_visible(False)
    try:
        for decade, axes in zip(decades, axes_list, strict=True):
            ids = decade_groups[decade]
            _draw_overlay_on_axes(subset_overlay(overlay, ids), axes)
            limits.apply(axes)
            decade_years = sorted(
                year for year, year_ids in year_groups.items() if set(year_ids) & set(ids)
            )
            year_text = (
                f"{decade_years[0]}-{decade_years[-1]}"
                if len(decade_years) > 1
                else str(decade_years[0])
            )
            means, _ = _year_means(ids, result)
            year_word = "year" if len(decade_years) == 1 else "years"
            axes.set_title(
                f"{decade}\n"
                f"n={len(ids)} of {total} ({100 * len(ids) / total:.0f}%); "
                f"{year_text} ({len(decade_years)} {year_word})\n"
                f"applied-N ladders: {format_shares(_ladder_shares(ids, result), limit=3)}\n"
                f"mean zero-N={means['yield_at_zero_n_t_ha']:.2f} t/ha; "
                f"mean response={means['response_above_zero_n_t_ha']:.2f} t/ha",
                fontsize=9,
            )
        for position, axes in enumerate(axes_list):
            if position // columns == rows - 1 or position + columns >= len(axes_list):
                axes.set_xlabel("Applied N (kg N/ha)")
            if position % columns == 0:
                axes.set_ylabel("Grain yield (t/ha)")

        handles, labels = axes_list[0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False
        )
        _reserve_suptitle(
            figure,
            _wrap_title_lines(
                [
                    f"source={SOURCE_NAME} — {season_label(TARGET_SEASON)}, "
                    f"{TARGET_VARIETY}: planting years banded into decades",
                    f"{total} cluster-eligible trajectories over {len(decades)} "
                    "decades; every panel shares this frame with "
                    "`../1966_ir8.jpeg`, which pools them, and with "
                    "`panel_grid.jpeg`, which does not band them",
                    "each panel is a recorded stratum, not a cluster: nothing was "
                    "clustered to produce it",
                    "a decade carries whatever changed with it -- the applied-N "
                    "ladder, plot design and accumulated soil history among them "
                    "-- so a difference between decades is a difference between "
                    "eras, not a time trend by itself; see `annual_trend.jpeg` "
                    "for the year-by-year detail this banding does not need to "
                    "preserve, since IR8 supports it directly",
                    "variety is confounded with the calendar period it was grown "
                    "in; a difference between decades here is not a genetic "
                    "effect",
                    _DISCLAIMER,
                ],
                width=180,
            ),
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


def _annual_records(
    year_groups: dict[int, tuple[str, ...]], result
) -> tuple[AnnualRecord, ...]:
    """Real `AnnualRecord`s for IR8, so `annual_ladder_eras` can be reused as-is."""

    records = []
    for year in sorted(year_groups):
        ids = year_groups[year]
        means, errors = _year_means(ids, result)
        ladder, mixed = _dominant_ladder(ids, result)
        records.append(
            AnnualRecord(
                year=year,
                decade=decade_band(str(year)),
                trajectory_count=len(ids),
                context_count=1,
                ladder=ladder,
                ladder_is_mixed=mixed,
                means=means,
                standard_errors=errors,
            )
        )
    return tuple(records)


def _decade_inset_readout(
    ids: Sequence[str], result, total: int, *, compact: bool
) -> str:
    centroid = centroid_summary(ids, result.features)
    years = cluster_year_range(ids, result.contexts)
    share = 100 * len(ids) / total
    ladder_shares = _ladder_shares(ids, result)
    single_year = bool(years) and years[0] == years[1]
    if compact:
        year_text = (
            str(years[0])
            if single_year
            else f"{years[0]}-{str(years[1])[2:]}" if years else "years unknown"
        )
        return "\n".join(
            [
                f"n={len(ids)} ({share:.0f}%)",
                year_text,
                f"0N {centroid['yield_at_zero_n_t_ha']:.2f}",
                f"+N {centroid['response_above_zero_n_t_ha']:.2f}",
            ]
        )
    year_text = (
        str(years[0]) if single_year else f"{years[0]}-{years[1]}" if years else "years unknown"
    )
    return "\n".join(
        [
            f"{len(ids)} trajectories ({share:.0f}%)",
            year_text,
            f"applied N {format_shares(ladder_shares, limit=1)}",
            f"mean zero-N {centroid['yield_at_zero_n_t_ha']:.2f} t/ha",
            f"mean response {centroid['response_above_zero_n_t_ha']:.2f} t/ha",
        ]
    )


def _split_calendar_runs(records: Sequence[AnnualRecord]) -> list[list[AnnualRecord]]:
    """Split an era's records at any unrecorded year, so the line breaks there.

    `annual_ladder_eras` merges list-adjacent same-ladder years into one era
    regardless of a calendar gap between them -- correct for the era/colour
    boundary, since the design did not change, but wrong for the line itself:
    IR8's 1984 gap sits inside the single 1970-1990 era, and drawing straight
    through it would read as an observed trend across a year with no data.
    """

    return _contiguous_runs(records, year=lambda record: record.year, presorted=True)


# The reference sheet (`by_planting_year/annotated/decades_and_trend.jpeg`) is 34in wide
# for the whole DS season's 6 decades (1960s-2010s, 60 calendar years). Reusing
# that inches-per-year density here, rather than the fixed 34in, keeps IR8's
# 4-decade sheet proportioned the same way instead of stretching thin insets
# to fill a width sized for half again as many decades.
_REFERENCE_DECADE_COUNT = 6
_REFERENCE_YEAR_SPAN = _REFERENCE_DECADE_COUNT * 10
_INCHES_PER_YEAR = (
    _SHEET_WIDTH_IN - _SHEET_LEFT_IN - _SHEET_RIGHT_IN
) / _REFERENCE_YEAR_SPAN


def _write_decades_and_trend(
    result,
    overlay,
    decade_groups: dict[str, tuple[str, ...]],
    year_groups: dict[int, tuple[str, ...]],
    limits,
    destination: Path,
    *,
    matched_colours: bool = False,
) -> None:
    """IR8's decade insets drawn on the stretch of its own yield trend.

    The IR8-scoped analogue of
    `../../by_planting_year/annotated/decades_and_trend.jpeg`:
    same pairing (each decade's response-curve cloud sitting on the exact
    calendar stretch of the per-year trend it came from, one shared
    inches-per-t/ha between the two trend panels), rebuilt from IR8's own
    per-year and per-decade groupings rather than a `FactorSubstructure`, since
    IR8 needs no stratum-size argument for either resolution.

    `matched_colours` reproduces the reference folder's
    `decades_and_trend_matched_colours.jpeg` treatment: colour is spent on the
    quantity (each trend panel takes the colour its own points already carry
    in every inset) instead of on the applied-N era, which moves to marker
    shape. Written as a separate file, not a replacement -- see
    `_write_decades_and_trend`'s caller.
    """

    from matplotlib import pyplot as plt

    decades = sorted(decade_groups)
    total = sum(len(ids) for ids in decade_groups.values())
    records = _annual_records(year_groups, result)
    spans = _decade_spans(sorted(decade_groups))
    if len(decades) < 2 or len(records) < 2 or len(spans) != len(decades):
        return

    x_lo = spans[0][1] - 0.5
    x_hi = spans[-1][2] + 0.5
    visible = [(first - 0.5, last + 0.5) for _, first, last in spans]
    unsampled = [
        f"{low}-{high}" if low < high else f"{low}"
        for low, high in (
            (spans[0][1], records[0].year - 1),
            (records[-1].year + 1, spans[-1][2]),
        )
        if low <= high
    ]
    gap_years = sorted(
        set(range(records[0].year, records[-1].year + 1))
        - {record.year for record in records}
    )

    colours = _treatment_class_colours(overlay)
    zero_n_colour = colours.get(_ZERO_N_TREATMENT_CLASS, "C1")
    mineral_colour = next(
        (colour for label, colour in colours.items() if label != _ZERO_N_TREATMENT_CLASS),
        "C0",
    )

    mixed = [record.year for record in records if record.ladder_is_mixed]
    headline = (
        f"source={SOURCE_NAME} — {season_label(TARGET_SEASON)}, {TARGET_VARIETY}: "
        "each planting decade's response curves drawn on the stretch of the "
        "yield trend they came from"
    )
    disclosures = [
        *(
            [
                "colour carries the quantity here, not the applied-N era: "
                "orange is yield at zero N — the orange points in every "
                "inset, and the panel that tracks their mean by year — and "
                "blue is the response above zero N, which is computed from "
                "the blue mineral-N points of the same inset relative to its "
                "orange ones rather than read off either directly",
                "the era keeps every encoding it had except colour: the line "
                "is still broken at each boundary, the boundary is still "
                "drawn dashed, and the marker shape now names the ladder as "
                "well",
            ]
            if matched_colours
            else []
        ),
        f"{total} cluster-eligible {TARGET_VARIETY} trajectories over "
        f"{len(records)} planting years ({records[0].year}-{records[-1].year}); "
        "unlike the whole season's `by_planting_year/`, no stratum-size "
        "argument is needed here -- IR8 supports a per-year trend directly, "
        "and decades are offered alongside it as the coarser read",
        "the insets — every trajectory of the decade against applied N, each "
        "a recorded stratum that nothing was clustered to produce, and each "
        "on the same frame at the same size, so any two can be compared "
        "directly",
        "each inset sits on its own calendar decade of the axis beneath it, "
        "whole decade to whole decade"
        + (
            f"; the stretch of a band with no trend under it — "
            f"{' and '.join(unsampled)} — is a decade IR8 was not grown in"
            if unsampled
            else ""
        ),
        "the two trend panels — yield at zero N by planting year, which is "
        "the left-hand end of every curve in the inset above it, and the "
        "response above zero N, which is the other end. Both are the mean of "
        "that year's IR8 replicate trajectories, the bar its standard error "
        "across replicates, and both are drawn to one shared scale so a "
        "change of a given size has the same slope in each",
        "each line is broken at every applied-N era boundary (dashed) and at "
        "any unrecorded year: a change across either is a change of "
        "experiment or a missing observation, not a response to fertilizer",
        *(
            [f"unrecorded planting year within the record: "
             f"{', '.join(str(year) for year in gap_years)}"]
            if gap_years
            else []
        ),
        "variety is confounded with the calendar period it was grown in; a "
        "difference between decades here is not a genetic effect, and every "
        "year but one is the same design and site",
        *(
            [
                "years carrying more than one applied-N ladder (banded on "
                f"the dominant one): {', '.join(str(year) for year in mixed)}"
            ]
            if mixed
            else []
        ),
        _DISCLAIMER,
    ]
    bullets = [
        textwrap.fill(
            entry,
            width=_CAPTION_COLUMN_CHARS,
            initial_indent="—  ",
            subsequent_indent="    ",
            break_long_words=False,
        )
        for entry in disclosures
    ]
    counts = [bullet.count("\n") + 1 for bullet in bullets]
    total_lines = sum(counts)
    split = min(
        range(1, max(len(bullets), 2)),
        key=lambda index: abs(2 * sum(counts[:index]) - total_lines),
    )
    caption_columns = ("\n".join(bullets[:split]), "\n".join(bullets[split:]))
    caption_lines = max(sum(counts[:split]), total_lines - sum(counts[:split]))
    headline_strip_in = _HEADLINE_FONT_SIZE * 1.35 / 72.0 + 0.24
    title_strip_in = (
        headline_strip_in + caption_lines * _CAPTION_FONT_SIZE * 1.45 / 72.0 + 0.45
    )

    year_span = x_hi - x_lo
    plot_width = _INCHES_PER_YEAR * year_span
    sheet_width = plot_width + _SHEET_LEFT_IN + _SHEET_RIGHT_IN

    height = (
        _SHEET_BOTTOM_IN
        + _TREND_LEGEND_IN
        + _TREND_XAXIS_IN
        + _TREND_ROW_IN
        + _TREND_GAP_IN
        + _HOST_ROW_IN
        + _INSET_LEGEND_IN
        + title_strip_in
    )
    figure = plt.figure(figsize=(sheet_width, height))

    def _box(
        left_in: float, bottom_in: float, width_in: float, height_in: float
    ) -> tuple[float, float, float, float]:
        return (
            left_in / sheet_width,
            bottom_in / height,
            width_in / sheet_width,
            height_in / height,
        )

    plot_left = _SHEET_LEFT_IN

    def _year_to_inches(year: float) -> float:
        return plot_left + plot_width * (year - x_lo) / year_span

    response_bottom = _SHEET_BOTTOM_IN + _TREND_LEGEND_IN + _TREND_XAXIS_IN
    host_bottom = response_bottom + _TREND_ROW_IN + _TREND_GAP_IN
    inset_legend_bottom = host_bottom + _HOST_ROW_IN

    eras = annual_ladder_eras(records)
    era_colours = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(eras), 1)))

    try:
        trend_panels = (
            (
                "yield_at_zero_n_t_ha",
                "Yield at zero N (t/ha)",
                host_bottom,
                _HOST_ROW_IN,
                zero_n_colour,
            ),
            (
                "response_above_zero_n_t_ha",
                "Response above zero N (t/ha)",
                response_bottom,
                _TREND_ROW_IN,
                mineral_colour,
            ),
        )
        low = high = None
        for record in records:
            for feature, *_unused in trend_panels:
                value = record.means[feature]
                error = record.standard_errors[feature]
                error = error if error == error else 0.0
                low = value - error if low is None else min(low, value - error)
                high = value + error if high is None else max(high, value + error)
        if low is None or high is None or not high > low:
            low, high = 0.0, 1.0
        pad = 0.06 * (high - low)
        floor, ceiling = low - pad, high + pad
        shared_ticks = [
            tick
            for tick in plt.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(
                floor, ceiling
            )
            if floor <= tick <= ceiling
        ]

        trend_axes: list[Any] = []
        for feature, label, bottom_in, row_in, panel_colour in trend_panels:
            is_host = not trend_axes
            axes = figure.add_axes(
                _box(plot_left, bottom_in, plot_width, row_in),
                sharex=trend_axes[0] if trend_axes else None,
            )
            trend_axes.append(axes)

            for index, (band_lo, band_hi) in enumerate(visible):
                if index % 2 == 0:
                    axes.axvspan(band_lo, band_hi, color="0.94", zorder=0, linewidth=0)
                axes.axvline(band_hi, color="0.85", linewidth=0.8, zorder=0)

            for era_index, (ladder, first, last) in enumerate(eras):
                span = [record for record in records if first <= record.year <= last]
                colour = panel_colour if matched_colours else era_colours[era_index]
                marker = (
                    _ERA_MARKERS[era_index % len(_ERA_MARKERS)]
                    if matched_colours
                    else "o"
                )
                for run_index, run in enumerate(_split_calendar_runs(span)):
                    axes.errorbar(
                        [record.year for record in run],
                        [record.means[feature] for record in run],
                        yerr=[record.standard_errors[feature] for record in run],
                        marker=marker,
                        # Larger where the shape has to be read: with colour
                        # spent on the quantity, a square and a diamond at 5
                        # points are the same smudge.
                        markersize=7 if matched_colours else 5,
                        linewidth=1.8,
                        capsize=3.0,
                        color=colour,
                        ecolor=colour,
                        elinewidth=0.9,
                        zorder=3,
                        label=(
                            f"{ladder} kg N/ha ({first}-{last})"
                            if is_host and run_index == 0
                            else None
                        ),
                    )
            for _, first, _ in eras[1:]:
                axes.axvline(first - 0.5, color="0.55", linewidth=0.9, ls="--", zorder=1)
            axes.set_ylabel(
                label,
                y=_HOST_TREND_FRACTION / 2 if is_host else 0.5,
                fontsize=_AXIS_LABEL_FONT_SIZE,
            )
            axes.grid(axis="y", alpha=0.25, linewidth=0.6)
            axes.tick_params(labelsize=_TICK_FONT_SIZE)
            axes.set_yticks(shared_ticks)
            axes.set_ylim(
                floor,
                floor + (ceiling - floor) / _HOST_TREND_FRACTION if is_host else ceiling,
            )

        trend_axes[0].set_xlim(x_lo, x_hi)
        trend_axes[0].tick_params(labelbottom=False)
        trend_axes[-1].set_xticks([start for _, start, _ in spans])
        trend_axes[-1].set_xticks(
            list(range(spans[0][1], spans[-1][2] + 1, 2)), minor=True
        )
        trend_axes[-1].set_xlabel("Planting year", fontsize=_AXIS_LABEL_FONT_SIZE)

        inset_axes: list[Any] = []
        for index, (decade, (band_lo, band_hi)) in enumerate(
            zip(decades, visible, strict=True)
        ):
            ids = decade_groups[decade]
            band_left = _year_to_inches(band_lo)
            band_right = _year_to_inches(band_hi)
            panel_inches = max(band_right - band_left - _PANEL_GUTTER_IN, 0.4)
            axes = figure.add_axes(
                _box(
                    band_left + _PANEL_GUTTER_IN / 2,
                    host_bottom + _INSET_BOTTOM_FRACTION * _HOST_ROW_IN,
                    panel_inches,
                    (_INSET_TOP_FRACTION - _INSET_BOTTOM_FRACTION) * _HOST_ROW_IN,
                ),
                sharex=inset_axes[0] if inset_axes else None,
                sharey=inset_axes[0] if inset_axes else None,
            )
            inset_axes.append(axes)
            axes.set_facecolor("white")
            axes.set_zorder(trend_axes[0].get_zorder() + 1)
            for spine in axes.spines.values():
                spine.set_color("0.55")

            _draw_overlay_on_axes(subset_overlay(overlay, ids), axes)
            limits.apply(axes)
            axes.grid(alpha=0.25, linewidth=0.6)
            axes.set_xticks([0, 50, 100, 150, 200])
            axes.tick_params(labelsize=_INSET_TICK_FONT_SIZE)
            axes.set_title(decade, fontsize=_INSET_TITLE_FONT_SIZE, pad=6)
            axes.text(
                0.97,
                0.03,
                _decade_inset_readout(
                    ids, result, total, compact=panel_inches < _NARROW_PANEL_IN
                ),
                transform=axes.transAxes,
                ha="right",
                va="bottom",
                fontsize=_INSET_READOUT_FONT_SIZE,
                color="0.25",
                linespacing=1.4,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 3.0,
                },
            )
            if index > 0:
                axes.tick_params(labelleft=False)

        figure.text(
            (plot_left + plot_width / 2) / sheet_width,
            (host_bottom + (_INSET_BOTTOM_FRACTION - 0.05) * _HOST_ROW_IN) / height,
            "insets: applied N (kg N/ha) across, grain yield (t/ha) up — one "
            "shared frame, ticks on the left-hand inset",
            ha="center",
            va="top",
            fontsize=_LEGEND_FONT_SIZE,
            color="0.35",
        )

        handles, labels = inset_axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="center",
            bbox_to_anchor=_box(
                plot_left, inset_legend_bottom, plot_width, _INSET_LEGEND_IN
            ),
            bbox_transform=figure.transFigure,
            ncol=len(labels),
            fontsize=_LEGEND_FONT_SIZE,
            frameon=False,
        )

        trend_handles, trend_labels = trend_axes[0].get_legend_handles_labels()
        figure.legend(
            trend_handles,
            trend_labels,
            loc="center",
            bbox_to_anchor=_box(plot_left, _SHEET_BOTTOM_IN, plot_width, _TREND_LEGEND_IN),
            bbox_transform=figure.transFigure,
            ncol=min(4, max(len(trend_labels), 1)),
            fontsize=_LEGEND_FONT_SIZE,
            frameon=False,
            title=(
                "applied-N era — marker shape; colour on this sheet is the "
                "quantity, not the era"
                if matched_colours
                else "applied-N era"
            ),
            title_fontsize=_LEGEND_FONT_SIZE,
        )

        headline_top = 1.0 - 0.20 / height
        figure.text(
            0.5,
            headline_top,
            headline,
            ha="center",
            va="top",
            fontsize=_HEADLINE_FONT_SIZE,
            color="0.10",
        )
        caption_top = headline_top - headline_strip_in / height
        for column_index, column in enumerate(caption_columns):
            figure.text(
                (plot_left + column_index * plot_width * _CAPTION_COLUMN_OFFSET)
                / sheet_width,
                caption_top,
                column,
                ha="left",
                va="top",
                fontsize=_CAPTION_FONT_SIZE,
                color="0.25",
                linespacing=1.45,
            )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


def _write_annual_trend(
    result,
    year_groups: dict[int, tuple[str, ...]],
    destination: Path,
) -> None:
    from matplotlib import pyplot as plt

    years = sorted(year_groups)
    records = []
    for year in years:
        ids = year_groups[year]
        means, errors = _year_means(ids, result)
        ladder, mixed = _dominant_ladder(ids, result)
        records.append((year, ladder, mixed, len(ids), means, errors))

    runs = _contiguous_runs(years, year=int, presorted=True)
    panels = (
        ("yield_at_zero_n_t_ha", "Yield at zero N (t/ha)"),
        ("response_above_zero_n_t_ha", "Response above zero N (t/ha)"),
    )
    by_year = {record[0]: record for record in records}
    figure, axes_grid = plt.subplots(
        len(panels), 1, figsize=(12.0, 7.5), sharex=True, constrained_layout=True
    )
    axes_list = list(np.atleast_1d(axes_grid).ravel())
    try:
        for (feature, label), axes in zip(panels, axes_list, strict=True):
            for run_index, run in enumerate(runs):
                run_years = [year for year in run]
                values = [by_year[year][4][feature] for year in run_years]
                errs = [by_year[year][5][feature] for year in run_years]
                axes.errorbar(
                    run_years,
                    values,
                    yerr=errs,
                    marker="o",
                    markersize=4,
                    linewidth=1.4,
                    capsize=2.5,
                    color="tab:blue",
                    ecolor="tab:blue",
                    elinewidth=0.8,
                    label="IR8, DS" if (axes is axes_list[0] and run_index == 0) else None,
                )
            # Mark the applied-N era boundary within each contiguous run (0/40-
            # 60-120 & 0/60/100/140 in 1968-1969, 0/50/100/150 from 1970); a
            # gap year (no run to compare across) is not itself a boundary.
            era_boundaries = sorted(
                {
                    run[index]
                    for run in runs
                    for index in range(1, len(run))
                    if by_year[run[index]][1] != by_year[run[index - 1]][1]
                }
            )
            for boundary in era_boundaries:
                axes.axvline(boundary - 0.5, color="0.55", linewidth=0.9, ls="--")
            axes.set_ylabel(label)
            axes.grid(alpha=0.25, linewidth=0.6)

        axes_list[-1].set_xlabel("Planting year")
        handles, labels = axes_list[0].get_legend_handles_labels()
        if handles:
            figure.legend(
                handles, labels, loc="lower center", ncol=1, fontsize=8, frameon=False
            )
        gap_years = [
            year
            for year in range(years[0], years[-1] + 1)
            if year not in by_year
        ]
        mixed_years = [year for year in years if by_year[year][2]]
        _reserve_suptitle(
            figure,
            _wrap_title_lines(
                [
                    f"source={SOURCE_NAME} — {season_label(TARGET_SEASON)}, "
                    f"{TARGET_VARIETY}: observed response by planting year",
                    f"{len(years)} years, {years[0]}-{years[-1]}; each point is the "
                    "mean of that year's IR8 replicate trajectories (4, except "
                    "where noted), the bar its standard error across replicates",
                    "the line is broken at every applied-N era boundary (dashed) "
                    "and at any unrecorded year: a change across either is a "
                    "change of experiment or a missing observation, not a "
                    "response to fertilizer",
                    *(
                        [f"unrecorded planting years: {', '.join(str(y) for y in gap_years)}"]
                        if gap_years
                        else []
                    ),
                    *(
                        [
                            "years carrying more than one applied-N ladder "
                            f"(banded on the dominant one): "
                            f"{', '.join(str(y) for y in mixed_years)}"
                        ]
                        if mixed_years
                        else []
                    ),
                    _DISCLAIMER,
                ],
                width=160,
            ),
            legend_strip=0.06,
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


def _write_readme(
    year_groups: dict[int, tuple[str, ...]],
    decade_groups: dict[str, tuple[str, ...]],
    destination: Path,
) -> None:
    years = sorted(year_groups)
    total = sum(len(ids) for ids in year_groups.values())
    gap_years = [
        year for year in range(years[0], years[-1] + 1) if year not in year_groups
    ]
    decades = sorted(decade_groups)
    lines = [
        f"# {season_label(TARGET_SEASON)} — {TARGET_VARIETY}, by planting year",
        "",
        f"{total} cluster-eligible LTCCE {season_label(TARGET_SEASON)} trajectories "
        f"carrying `Variety={TARGET_VARIETY!r}`, {years[0]}-{years[-1]}, one panel per "
        "recorded planting year -- the year-by-year split of "
        "`../1966_ir8.jpeg`, which pools them.",
        "",
        "**Variety is confounded with the calendar period it was grown in; a "
        "difference between years here is not a genetic effect.**",
        "",
        "## Why individual years, unlike `../../by_planting_year/`",
        "",
        "The whole-season `by_planting_year/` folder bands into decades because "
        "no single calendar year clears this product's 30-trajectory stratum "
        "minimum. IR8 alone needs no such banding: it carries exactly four "
        "replicate trajectories in every recorded planting year from "
        f"{years[0]} to {years[-1]}"
        + (f", except {', '.join(str(y) for y in gap_years)} (unrecorded)" if gap_years else "")
        + ", all from the same design (Split-split-plot) and site (IRRI-B5-B8). "
        "That regularity is specific to this one variety level; it is not "
        "reproduced for the other `by_variety/` levels.",
        "",
        "## What is in this folder",
        "",
        "- `panel_grid.jpeg` -- every planting year on its own panel, on the "
        "frame shared by the whole product. A recorded stratum: nothing was "
        "clustered to produce it.",
        f"- `decade_panel_grid.jpeg` -- the same trajectories banded into "
        f"{len(decades)} calendar decades ({', '.join(decades)}), the same "
        "banding `../../by_planting_year/` uses for the whole season -- "
        "IR8 does not need it for stratum size, but it is offered as the "
        "coarser read.",
        "- `annual_trend.jpeg` -- the same years as one trend, zero-N yield and "
        "response above zero N against planting year, line broken at every "
        "applied-N era boundary and at the unrecorded year.",
        "- `decades_and_trend.jpeg` -- both resolutions on one sheet, with each "
        "decade's response-curve panel set over the exact stretch of the year "
        "axis its trajectories came from and drawn as wide as that stretch, so "
        "a panel and its shaded band are the same interval of calendar time "
        "read two ways -- the IR8-only analogue of "
        "`../../by_planting_year/annotated/decades_and_trend.jpeg`.",
        "- `decades_and_trend_matched_colours.jpeg` -- the same sheet with "
        "colour spent on the quantity instead of the applied-N era: orange "
        "for yield at zero N, blue for response above zero N, matching each "
        "trend panel to its own points in every inset above it. The era it "
        "displaces is re-encoded as marker shape, so nothing the plain sheet "
        "says is lost.",
        "",
        "## Interpretation boundary",
        "",
        "Exploratory diagnostic, outside the governed analysis inventory "
        "(ANA-11). No curve is fitted anywhere in this folder; every mean is "
        "computed from observed yields at observed N rates. Standard errors on "
        "`annual_trend.jpeg` are the spread across each year's own replicate "
        "trajectories, not a fitted quantity's sampling error.",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    source_path, encoding = _load_source_spec(args.config)

    overlay = read_source_dataset_overlay(source_path, SOURCE_NAME, encoding=encoding)
    contexts = read_ltcce_contexts(source_path, encoding=encoding)
    result = build_season_clustering(overlay, contexts)
    limits = _shared_axis_limits(overlay)

    year_groups = _year_groups(result, TARGET_SEASON, TARGET_VARIETY)
    if not year_groups:
        raise SystemExit(
            f"No {TARGET_VARIETY!r} trajectories found in the "
            f"{TARGET_SEASON!r} season profile"
        )
    decade_groups = _decade_groups(year_groups)

    destination = _plain_absolute_path(args.output_dir, label="IR8 output destination")
    publication_container = _nested_publication_container(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = publication_container.parent / (
        f".{publication_container.name}.{destination.name}.nested-staging."
        f"{uuid.uuid4().hex}"
    )
    staging.mkdir()
    try:
        _write_year_panel_grid(
            result, overlay, year_groups, limits, staging / "panel_grid.jpeg"
        )
        _write_decade_panel_grid(
            result,
            overlay,
            decade_groups,
            year_groups,
            limits,
            staging / "decade_panel_grid.jpeg",
        )
        _write_annual_trend(result, year_groups, staging / "annual_trend.jpeg")
        _write_decades_and_trend(
            result,
            overlay,
            decade_groups,
            year_groups,
            limits,
            staging / "decades_and_trend.jpeg",
        )
        _write_decades_and_trend(
            result,
            overlay,
            decade_groups,
            year_groups,
            limits,
            staging / "decades_and_trend_matched_colours.jpeg",
            matched_colours=True,
        )
        _write_readme(year_groups, decade_groups, staging / "README.md")

        with _core_overlay_publication_lock(publication_container):
            _promote_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    total = sum(len(ids) for ids in year_groups.values())
    print(
        f"Wrote panel_grid.jpeg, decade_panel_grid.jpeg, annual_trend.jpeg, "
        f"decades_and_trend.jpeg, decades_and_trend_matched_colours.jpeg, "
        f"README.md for {total} {TARGET_VARIETY} trajectories under {destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
