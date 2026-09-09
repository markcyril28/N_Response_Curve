#!/usr/bin/env python3
"""Write a by-planting-year breakdown of LTCCE N response, one folder per variety.

Standalone companion to `generate_response_curve_season_clusters.py`'s own
`by_season/ds/by_variety/` -- the DS season's 935 cluster-eligible trajectories
decomposed by the source `Variety` column into 76 recorded levels, of which the
eight carrying 30 trajectories or more are drawn as their own figure. Each of
those figures pools every planting year of its level into one panel; this
script splits each back out, one panel per year, on the same shared axis frame
the season module uses, into a `<first planting year>_<level>_by_planting_year/`
folder beside it. It also draws a companion annual-trend line chart per level,
in the same spirit as
`by_season/ds/by_planting_year/annotated/annual_trend.jpeg`.

**The folder's year prefix is not the figure's.** `1966_ir8.jpeg` carries the
NSIC release year; `1968_ir8_by_planting_year/` carries the first planting year
the level appears in this season's record, which is what the folder is about --
a listing of the folders is then the order the experiment took the varieties
up. The two coincide for IR20, IR36 and IR58 and differ by anything from one
year to five elsewhere, so each folder note states both. A breeding-line
designation has no release year at all, and its folder is the only place it
carries a year.

Thirty trajectories is not this recipe's threshold; it is the one
`by_variety/README.md` already states for drawing a level at all, reused so the
folders and the figures beside them have the same roster. `--min-trajectories`
lowers it.

Every one of those levels carries three or four replicate trajectories in each
of its recorded planting years -- so unlike the whole-season
`by_planting_year/` folder, no stratum-size argument is needed to justify
showing individual years here; the panel grids below need no banding.

That regularity is also why the combined reference sheet -- response-curve
insets set over the stretch of the yield trend they came from -- is written at
both resolutions from one writer: `decades_and_trend.jpeg` bands the insets
into calendar decades, the way the whole season has to, and
`years_and_trend.jpeg` gives every recorded planting year its own inset over
its own single year of the axis. The two are the same sheet at two banding
resolutions, not two figures; see `_SheetBanding`. A level whose whole record
sits inside one calendar decade gets no decade sheet and no decade grid: at one
band they would restate `../<level>.jpeg`, which already pools it.

**What differs between levels is stated, never assumed.** The sheet text used
to carry IR8's own regularity as a literal -- four replicates a year, one
design, one site, one unrecorded year. None of it generalizes: IR72 changes
plot design partway through, `IRRI 146 (NSIC Rc158)` is recorded under two
spellings of its site, IR58 loses one 1986 replicate to a missing zero-N
anchor, and IR72's 1992-2001 is a stretch the source records in full and the
clustering excludes whole -- which is a different kind of gap from a year that
was not grown, and is labelled as one. See `_VarietyScope`.

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
from typing import Any, Callable, NamedTuple, Sequence

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
    FACTOR_DEFINITIONS,
    FACTOR_VARIETY,
    MIN_FACTOR_STRATUM,
    AnnualRecord,
    annual_ladder_eras,
    build_season_clustering,
    cluster_year_range,
    decade_band,
    factor_level_stem,
    factor_level_token,
    format_shares,
    ladder_text,
    season_label,
)
# The reason a trajectory did not reach a cluster profile is displayed in one
# vocabulary across the LTCCE recipes; re-spelling it here would put two names
# for one exclusion on two figures of the same record.
from n_response_curve.reporting.generate_ltcce_ds_variety_planting_year_views import (  # noqa: E402
    EXCLUSION_SHORT,
)
from n_response_curve.reporting.source_dataset_overlays import (  # noqa: E402
    read_source_dataset_overlay,
)

DEFAULT_SEASON = "DS"
GAP_TOKEN = "—"
SOURCE_NAME = "ltcce"
_VARIETY_FACTOR = FACTOR_DEFINITIONS[FACTOR_VARIETY]
_NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def _default_output_root(season: str) -> Path:
    """The `by_variety/` folder of a season, which is where the levels are drawn.

    A level's folder is a sibling of its own pooled figure, not a child of it:
    `1966_ir8.jpeg` and `ir8_by_planting_year/` are the same stratum at two
    resolutions, and the folder is where the reader is already looking.
    """

    return (
        PROJECT_ROOT
        / "WF/03_Response_Curves/ltcce"
        / "clusters/by_season"
        / season.strip().lower()
        / "by_variety"
    )


class _VarietyScope(NamedTuple):
    """One variety level, and what its own record actually says.

    Everything here was a literal in the IR8-only version of this recipe --
    four replicates a year, one plot design, one site, 1984 the single gap --
    and none of it survives being pointed at another level. It is therefore
    derived from the record and phrased from what was found, so a sheet drawn
    for IR72 describes IR72.

    `stem` is the pooled figure beside the folder (`1966_ir8`), which the
    captions cross-reference, and `slug` is the folder's own token (`ir8`);
    both come from the season module so the names stay joined to the ones it
    writes. `release_year` is the same NSIC year the stem carries, kept
    separately because the folder is prefixed with `first_year` instead and the
    note has to be able to name both. `designs` and `sites` are runs rather than sets, because a level
    that changed design partway through is saying *when*. `gaps` pairs a year
    with the reason it carries no panel: the empty string where nothing was
    observed, and the exclusion that removed it where the source has the year
    in full.
    """

    variety: str
    season: str
    stem: str
    slug: str
    first_year: int
    release_year: int | None
    designs: tuple[tuple[str, int, int], ...]
    sites: tuple[tuple[str, int, int], ...]
    replicate_counts: tuple[tuple[int, tuple[int, ...]], ...]
    gaps: tuple[tuple[int, str], ...]

    @property
    def season_text(self) -> str:
        return season_label(self.season)

    @property
    def unobserved_years(self) -> tuple[int, ...]:
        return tuple(year for year, reason in self.gaps if not reason)

    @property
    def excluded_years(self) -> tuple[int, ...]:
        return tuple(year for year, reason in self.gaps if reason)

    @property
    def exclusion_reasons(self) -> tuple[str, ...]:
        seen = dict.fromkeys(reason for _year, reason in self.gaps if reason)
        return tuple(seen)

    @property
    def folder_name(self) -> str:
        """`<first planting year>_<token>_by_planting_year`.

        Prefixed with the first year the level was *grown here*, not the year
        it was released: the folder is a record of an experiment, and a
        breeding line that was never released still entered the trial in a
        definite year. The pooled figure beside it keeps the release-year
        prefix the season module gives it, so the two prefixes disagree by a
        year or five for most levels -- `_write_readme` says so, in the folder,
        rather than leaving the reader to notice.
        """

        return f"{self.first_year}_{self.slug}_by_planting_year"


def _factor_runs(
    years: Sequence[int], value: Callable[[int], str]
) -> tuple[tuple[str, int, int], ...]:
    """`(value, first year, last year)` for each run of one recorded factor."""

    runs: list[list[Any]] = []
    for year in years:
        current = value(year)
        if runs and runs[-1][0] == current:
            runs[-1][2] = year
        else:
            runs.append([current, year, year])
    return tuple((name, first, last) for name, first, last in runs)


def _variety_scope(
    result,
    season: str,
    variety: str,
    year_groups: dict[int, tuple[str, ...]],
) -> _VarietyScope:
    """Measure one level's record, so the sheets can describe it rather than assert it."""

    years = sorted(year_groups)

    def recorded(year: int, attribute: str) -> str:
        found = sorted(
            {
                getattr(result.contexts[trajectory_id], attribute).strip()
                for trajectory_id in year_groups[year]
                if trajectory_id in result.contexts
            }
        )
        return " / ".join(found) if found else "unrecorded"

    counts: dict[int, list[int]] = collections.defaultdict(list)
    for year in years:
        counts[len(year_groups[year])].append(year)

    gaps: list[tuple[int, str]] = []
    for year in range(years[0], years[-1] + 1):
        if year in year_groups:
            continue
        reasons = collections.Counter(
            reason
            for trajectory_id, reason in result.excluded.items()
            for context in [result.contexts.get(trajectory_id)]
            if context is not None
            and context.season.strip().upper() == season
            and context.variety.strip() == variety
            and int(context.year or 0) == year
        )
        gaps.append(
            (
                year,
                ", ".join(
                    EXCLUSION_SHORT.get(reason, reason.replace("_", " "))
                    for reason, _count in reasons.most_common()
                ),
            )
        )

    return _VarietyScope(
        variety=variety,
        season=season,
        stem=factor_level_stem(_VARIETY_FACTOR, variety),
        slug=factor_level_token(variety),
        first_year=years[0],
        release_year=_VARIETY_FACTOR.release_year(variety),
        designs=_factor_runs(years, lambda year: recorded(year, "design")),
        sites=_factor_runs(years, lambda year: recorded(year, "site")),
        replicate_counts=tuple(
            (count, tuple(counts[count]))
            for count in sorted(counts, key=lambda size: (-len(counts[size]), -size))
        ),
        gaps=tuple(gaps),
    )


def _year_list(years: Sequence[int]) -> str:
    """Years written for a reader: a consecutive run as a range, the rest listed."""

    return ", ".join(
        f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0])
        for run in _contiguous_runs(sorted(years), year=int, presorted=True)
    )


def _replicate_exceptions(scope: _VarietyScope) -> str:
    """The years whose replicate count is not the modal one, or an empty string.

    Shared by the folder note and the trend plate so a figure that says
    "except where noted" is on the same page as whatever noted it.
    """

    _modal, *exceptions = scope.replicate_counts
    return "; ".join(
        f"{_year_list(sorted(exception_years))}, which "
        f"{'carries' if len(exception_years) == 1 else 'carry'} {count}"
        for count, exception_years in exceptions
    )


def _replicate_phrase(scope: _VarietyScope, years: Sequence[int]) -> str:
    """How many replicate trajectories a planting year of this level carries.

    IR8 carries four in every one of its years, which is what this folder's
    note used to assert of every level. IR58 carries three in 1986 -- one
    replicate has no recorded yield at zero N -- so the count is written from
    the counts.

    Counted over the years actually drawn rather than over the calendar span:
    IR72 spans 1989-2012 and is drawn for thirteen of those years, and "every
    recorded planting year" would be a claim about the ten it is not drawn for.
    """

    (modal, _modal_years), *exceptions = scope.replicate_counts
    word = _NUMBER_WORDS.get(modal, str(modal))
    span = (
        f"in each of the {len(years)} planting years drawn here "
        f"({years[0]}-{years[-1]})"
    )
    if not exceptions:
        return f"exactly {word} replicate trajectories {span}"
    return (
        f"{word} replicate trajectories {span}, except "
        f"{_replicate_exceptions(scope)}"
    )


def _gap_inventory(scope: _VarietyScope) -> str:
    """Every stretch of the year axis carrying no panel, each with its own reason.

    Two different things put a hole in these figures and only one of them is a
    hole in the experiment. IR8's 1984 has plot rows in the source and no yield
    in any of them; IR72's 1991 has no rows at all. IR72's 1992-2001 is neither
    -- the source records that decade in full, and the clustering excludes
    every trajectory in it because two yields are recorded at each applied-N
    level with nothing to tell them apart. The parent folder's combined sheet
    labels that same stretch "no DS record", which is not what the source says;
    saying it again here would spread the error rather than correct it.
    """

    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for year, reason in scope.gaps:
        grouped[reason].append(year)
    return "; ".join(
        f"{_year_list(years)}, "
        + (
            f"recorded in full and excluded whole ({reason})"
            if reason
            else "unrecorded"
        )
        for reason, years in sorted(grouped.items(), key=lambda item: item[1][0])
    )


def _design_site_phrase(scope: _VarietyScope) -> str:
    """What stayed constant across this level's record, and what did not.

    Site is reported as a count of *labels* rather than of sites, because a
    level can change spelling without moving: `IRRI 146 (NSIC Rc158)` is
    recorded at `IRRI-B5-B8` through 2015 and at `B5-B8` from 2016, which is
    almost certainly one place written two ways. Naming it two sites would be
    a claim this recipe cannot make; naming it two labels is what the source
    supports.
    """

    def clause(
        runs: Sequence[tuple[str, int, int]], singular: str, plural: str
    ) -> str:
        if len(runs) == 1:
            return f"one {singular} ({runs[0][0]})"
        spans = ", ".join(
            f"{name} {first}-{last}" if first != last else f"{name} {first}"
            for name, first, last in runs
        )
        return f"{len(runs)} {plural} ({spans})"

    return (
        f"{clause(scope.designs, 'plot design', 'plot designs')} at "
        f"{clause(scope.sites, 'recorded site', 'recorded site labels')}"
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
        "--season",
        default=DEFAULT_SEASON,
        help="LTCCE season code to decompose (DS, EWS, LWS). Default DS.",
    )
    parser.add_argument(
        "--variety",
        action="append",
        default=None,
        metavar="LEVEL",
        help=(
            "Recorded `Variety` level to draw, exactly as the source spells it. "
            "Repeatable. Defaults to every level the season\u2019s `by_variety/` "
            "folder draws as its own figure."
        ),
    )
    parser.add_argument(
        "--min-trajectories",
        type=int,
        default=MIN_FACTOR_STRATUM,
        help=(
            "Smallest level to build a folder for. The default is the threshold "
            "`by_variety/README.md` already states for drawing a level at all, "
            "so the folders and the figures beside them share one roster. "
            "Ignored when --variety is given."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Parent of the per-level folders (each replaced atomically). "
            "Defaults to the season\u2019s own `by_variety/` directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination for a single level, replacing the generated folder "
            "name. Accepted only with exactly one --variety."
        ),
    )
    arguments = parser.parse_args()
    arguments.season = arguments.season.strip().upper()
    if arguments.output_dir is not None and len(arguments.variety or []) != 1:
        parser.error("--output-dir names one destination, so pass exactly one --variety")
    return arguments


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
    """Band a level's planting years into calendar decades, the whole-season 'why'."""

    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for year, ids in year_groups.items():
        band = decade_band(str(year))
        if band:
            grouped[band].extend(ids)
    return {decade: tuple(sorted(ids)) for decade, ids in sorted(grouped.items())}


def _year_means(ids: Sequence[str], result) -> tuple[dict[str, float], dict[str, float]]:
    """Replicate mean and standard error for one year of one variety level.

    A planting year of any level in this folder is a single design/site/ladder
    context carrying three or four replicates, so this is a plain replicate
    mean -- not the multi-context weighting `annual_response_records` uses for
    the whole season, which would be indistinguishable from it here anyway.
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
    scope: _VarietyScope,
    result,
    overlay,
    year_groups: dict[int, tuple[str, ...]],
    limits,
    destination: Path,
) -> bool:
    """Every recorded planting year of one level on its own panel."""

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
                    f"source={SOURCE_NAME} — {scope.season_text}, "
                    f"{scope.variety}: every planting year on its own panel",
                    f"{total} cluster-eligible trajectories, {years[0]}-{years[-1]} "
                    f"({len(years)} of {years[-1] - years[0] + 1} calendar years "
                    f"carry a panel); every panel shares this frame with "
                    f"`../{scope.stem}.jpeg`, which pools them",
                    "each panel is a recorded stratum, not a cluster: nothing was "
                    "clustered to produce it",
                    *(
                        [
                            "a year with no panel: "
                            f"{_gap_inventory(scope)}"
                        ]
                        if scope.gaps
                        else []
                    ),
                    "variety is confounded with the calendar period it was grown "
                    "in; a difference between years here is not a genetic effect, "
                    f"and the record behind it is {_design_site_phrase(scope)}",
                    _DISCLAIMER,
                ],
                width=180,
            ),
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)
    return True


def _write_decade_panel_grid(
    scope: _VarietyScope,
    result,
    overlay,
    decade_groups: dict[str, tuple[str, ...]],
    year_groups: dict[int, tuple[str, ...]],
    limits,
    destination: Path,
) -> bool:
    """The same trajectories banded into calendar decades, where there is more than one.

    A level whose whole record sits inside one decade is declined rather than
    drawn: the single panel would be `../<level>.jpeg` again, which the folder
    already sits beside.
    """

    from matplotlib import pyplot as plt

    decades = sorted(decade_groups)
    if len(decades) < 2:
        return False
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
                    f"source={SOURCE_NAME} — {scope.season_text}, "
                    f"{scope.variety}: planting years banded into decades",
                    f"{total} cluster-eligible trajectories over {len(decades)} "
                    "decades; every panel shares this frame with "
                    f"`../{scope.stem}.jpeg`, which pools them, and with "
                    "`panel_grid.jpeg`, which does not band them",
                    "each panel is a recorded stratum, not a cluster: nothing was "
                    "clustered to produce it",
                    "a decade carries whatever changed with it -- the applied-N "
                    "ladder, plot design and accumulated soil history among them "
                    "-- so a difference between decades is a difference between "
                    "eras, not a time trend by itself; see `annual_trend.jpeg` "
                    "for the year-by-year detail this banding does not need to "
                    f"preserve, since {scope.variety} supports it directly",
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
    return True


def _annual_records(
    year_groups: dict[int, tuple[str, ...]], result
) -> tuple[AnnualRecord, ...]:
    """Real `AnnualRecord`s for one level, so `annual_ladder_eras` is reused as-is."""

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
# that inches-per-year density here, rather than the fixed 34in, keeps a level's
# shorter sheet proportioned the same way instead of stretching thin insets to
# fill a width sized for more decades than it has -- and it is what makes two
# levels' sheets comparable, since an inch is the same number of years on both.
_REFERENCE_DECADE_COUNT = 6
_REFERENCE_YEAR_SPAN = _REFERENCE_DECADE_COUNT * 10
_INCHES_PER_YEAR = (
    _SHEET_WIDTH_IN - _SHEET_LEFT_IN - _SHEET_RIGHT_IN
) / _REFERENCE_YEAR_SPAN

# The decade sheet's vertical strips restated in inches, so a sheet banded at a
# finer resolution can keep the ones that are not about the banding. Only the
# inset band's height belongs to the banding: the trend rows, the strip that
# carries the insets' applied-N tick labels, and the head room the inset titles
# are drawn into are the same however the years are grouped.
_TREND_BAND_IN = _HOST_TREND_FRACTION * _HOST_ROW_IN
_INSET_TICK_STRIP_IN = (_INSET_BOTTOM_FRACTION - _HOST_TREND_FRACTION) * _HOST_ROW_IN
_INSET_HEAD_ROOM_IN = (1.0 - _INSET_TOP_FRACTION) * _HOST_ROW_IN
_DECADE_INSET_ASPECT = (10.0 * _INCHES_PER_YEAR - _PANEL_GUTTER_IN) / (
    (_INSET_TOP_FRACTION - _INSET_BOTTOM_FRACTION) * _HOST_ROW_IN
)

# A year band is a tenth of a decade band, so reusing the decade sheet's
# density would give a 0.44in inset -- a smudge, not a response curve. The year
# sheet is drawn at its own density instead: wide enough that an inset clears
# `_NARROW_PANEL_IN` and keeps the full read-out, and only as tall as that
# width needs to hold the decade inset's shape, so an inset on either sheet is
# the same picture at a different size rather than a squashed one. The sheet
# that results is long rather than large -- 23 recorded years at this density
# is about 59 inches -- which is the honest shape for a record read in calendar
# order.
_ANNUAL_INCHES_PER_YEAR = 2.5
_ANNUAL_INSET_WIDTH_IN = _ANNUAL_INCHES_PER_YEAR - _PANEL_GUTTER_IN
_ANNUAL_INSET_HEIGHT_IN = _ANNUAL_INSET_WIDTH_IN / _DECADE_INSET_ASPECT
_ANNUAL_HOST_ROW_IN = (
    _TREND_BAND_IN
    + _INSET_TICK_STRIP_IN
    + _ANNUAL_INSET_HEIGHT_IN
    + _INSET_HEAD_ROOM_IN
)
# Measure of one wrapped caption column plus the gutter to the next, in inches.
# `_CAPTION_COLUMN_OFFSET` is a fraction of the plot width, which is the right
# unit only while the sheet is about as wide as two columns of text; on the
# year sheet it would open a 30-inch gap between the two columns. Taking the
# smaller of the two leaves the decade sheet exactly where it was.
_CAPTION_COLUMN_MEASURE_IN = 11.9
# An inset narrower than this drops the tick on the frame's right-hand limit.
# The limit tick is the one that cannot be spared: it is drawn centred on the
# panel edge, a tenth of an inch from the next panel's `0`, so on a sheet of
# abutting insets `200` and `0` run together and read as `2000`. Dropping it
# costs nothing -- the highest applied-N level anywhere in LTCCE is 195 kg N/ha,
# and the frame's extent is stated in the line beneath the row.
_DENSE_INSET_TICK_IN = 3.2
_INSET_TICKS = (0, 50, 100, 150, 200)
# Past this plot width the trend panels also carry their y tick labels on the
# right-hand edge. One column of labels is enough to read a 22-inch sheet
# against; it is not enough to read a 59-inch one, where the far end of the
# record is nearly five feet from the only numbers on the axis.
_WIDE_SHEET_LABEL_IN = 30.0
# A sheet narrower than this is padded to it and its axes centred, rather than
# drawn small. The density is deliberately fixed (see `_INCHES_PER_YEAR`), so a
# level occupying two decades genuinely is half the width of one occupying
# four -- but the headline and the two wrapped caption columns are set in
# points and do not shrink with it, and below about twenty inches they overrun
# a sheet sized only by its axes. Padding keeps the type legible and leaves the
# axes measuring calendar time; the white space is the level's own record being
# shorter, which is true. The floor is the four-decade span, so the widest
# levels are untouched.
_MIN_SHEET_SPAN_YEARS = 40
_MIN_SHEET_BAND_IN = _MIN_SHEET_SPAN_YEARS * _INCHES_PER_YEAR


class _SheetBanding(NamedTuple):
    """One resolution of the combined sheet: how the years are grouped, and how wide.

    The sheet pairs a row of response-curve insets with the yield trend beneath
    them, each inset sitting on the exact stretch of the year axis its
    trajectories came from. Nothing in that construction cares whether a
    "stretch" is a decade or a single year, so both resolutions are written by
    `_write_bands_and_trend` and differ only in this record.

    `spans` is `(label, first year, last year)` per band, oldest first, and
    `groups` maps the same labels to trajectory ids. `placement` and `readout`
    are callables rather than strings because both variants say something the
    other cannot: a decade band with no trend under it is a decade the level was
    not grown in, while a year band with no inset over it is a year that was
    either not observed or excluded whole -- which is not the same statement.
    """

    spans: tuple[tuple[str, int, int], ...]
    groups: dict[str, tuple[str, ...]]
    headline: str
    band_word: str
    band_word_plural: str
    scope_tail: str
    placement: Callable[[Sequence[str], Sequence[int]], str]
    readout: Callable[[Sequence[str], Any, int, float], str]
    inches_per_year: float
    host_row_in: float
    trend_row_in: float
    host_trend_fraction: float
    inset_bottom_fraction: float
    inset_top_fraction: float
    x_major_ticks: Callable[[tuple[tuple[str, int, int], ...]], list[int]]
    x_minor_ticks: Callable[[tuple[tuple[str, int, int], ...]], list[int]]


def _year_inset_readout(
    ids: Sequence[str], result, total: int, panel_inches: float
) -> str:
    """The year sheet's inset read-out.

    Deliberately not `_decade_inset_readout`: at year resolution the decade
    lines are either constant or noise. Every year carries the same three or
    four trajectories out of the level's own total, so the share is the same
    small percentage on every panel and costs width it cannot earn; the year is
    already the inset's title; and a single-ladder year does not need `100%`
    after the ladder. A year carrying more than one ladder spends the recovered
    lines on its ladder shares instead.
    """

    centroid = centroid_summary(ids, result.features)
    shares = _ladder_shares(ids, result)
    lines = [f"n={len(ids)}"]
    if len(shares) > 1:
        items = list(shares.items())[:2]
        lines.extend(f"{ladder} {100 * share:.0f}%" for ladder, share in items)
        if len(shares) > len(items):
            lines.append(f"+{len(shares) - len(items)} more")
    else:
        lines.append(next(iter(shares), "ladder unknown"))
    lines.extend(
        [
            f"0N {centroid['yield_at_zero_n_t_ha']:.2f} t/ha",
            f"+N {centroid['response_above_zero_n_t_ha']:.2f} t/ha",
        ]
    )
    return "\n".join(lines)


def _decade_banding(
    scope: _VarietyScope, decade_groups: dict[str, tuple[str, ...]]
) -> _SheetBanding:
    """Insets banded into calendar decades -- the banding the whole season uses."""

    def placement(unsampled: Sequence[str], gap_years: Sequence[int]) -> str:
        return (
            "each inset sits on its own calendar decade of the axis beneath it, "
            "whole decade to whole decade"
            + (
                f"; the stretch of a band with no trend under it — "
                f"{' and '.join(unsampled)} — is a decade {scope.variety} was "
                f"not grown in"
                if unsampled
                else ""
            )
        )

    return _SheetBanding(
        spans=_decade_spans(sorted(decade_groups)),
        groups=decade_groups,
        headline=(
            f"source={SOURCE_NAME} — {scope.season_text}, {scope.variety}: "
            "each planting decade's response curves drawn on the stretch of the "
            "yield trend they came from"
        ),
        band_word="decade",
        band_word_plural="decades",
        scope_tail="and decades are offered alongside it as the coarser read",
        placement=placement,
        readout=lambda ids, result, total, panel_inches: _decade_inset_readout(
            ids, result, total, compact=panel_inches < _NARROW_PANEL_IN
        ),
        inches_per_year=_INCHES_PER_YEAR,
        host_row_in=_HOST_ROW_IN,
        trend_row_in=_TREND_ROW_IN,
        host_trend_fraction=_HOST_TREND_FRACTION,
        inset_bottom_fraction=_INSET_BOTTOM_FRACTION,
        inset_top_fraction=_INSET_TOP_FRACTION,
        x_major_ticks=lambda spans: [start for _, start, _ in spans],
        x_minor_ticks=lambda spans: list(range(spans[0][1], spans[-1][2] + 1, 2)),
    )


def _year_banding(
    scope: _VarietyScope,
    year_groups: dict[int, tuple[str, ...]],
    *,
    decade_sheet: bool,
) -> _SheetBanding:
    """One inset per recorded planting year, each over its own year of the axis.

    A year with no panel gets no band and no inset, so the gap is drawn as the
    width of the missing years rather than closed up -- the same reading the
    decade sheet gives an unsampled decade, one order finer.

    The two reasons a year can be missing are not interchangeable and are not
    written as one. A year nothing was observed in is a hole in the experiment.
    A year the source records in full and the clustering excludes -- IR72's
    1992-2001, where two yields are recorded at every applied-N level with
    nothing to tell them apart -- is a hole in *this* figure only, and calling
    it "not grown" would repeat the error the parent sheet already makes about
    that same stretch.
    """

    def placement(unsampled: Sequence[str], gap_years: Sequence[int]) -> str:
        return (
            "each inset sits on its own calendar year of the axis beneath it, "
            "one year to one year, so the insets are the record in calendar "
            "order and the distance between any two of them is elapsed time"
            + (
                f"; the stretch of axis with no inset over it — "
                f"{_year_list(scope.unobserved_years)} — is a planting "
                f"year {scope.variety} was not grown in"
                if scope.unobserved_years
                else ""
            )
            + (
                f"; {_year_list(scope.excluded_years)} carries no inset for a "
                f"different reason — the source records "
                f"{'it' if len(scope.excluded_years) == 1 else 'them'} in full "
                f"and the clustering excludes "
                f"{'it' if len(scope.excluded_years) == 1 else 'them'} whole "
                f"({', '.join(scope.exclusion_reasons)}), so the gap is in this "
                "figure and not in the experiment"
                if scope.excluded_years
                else ""
            )
        )

    years = sorted(year_groups)
    return _SheetBanding(
        spans=tuple((str(year), year, year) for year in years),
        groups={str(year): year_groups[year] for year in years},
        headline=(
            f"source={SOURCE_NAME} — {scope.season_text}, {scope.variety}: "
            "each planting year's response curves drawn on the stretch of the "
            "yield trend they came from"
        ),
        band_word="year",
        band_word_plural="years",
        scope_tail=(
            "and this sheet is that per-year trend; `decades_and_trend.jpeg` "
            "is the same sheet banded into decades, the coarser read"
            if decade_sheet
            else "and this sheet is that per-year trend; the whole record sits "
            "inside one calendar decade, so there is no coarser read to offer"
        ),
        placement=placement,
        readout=lambda ids, result, total, panel_inches: _year_inset_readout(
            ids, result, total, panel_inches
        ),
        inches_per_year=_ANNUAL_INCHES_PER_YEAR,
        host_row_in=_ANNUAL_HOST_ROW_IN,
        trend_row_in=_TREND_BAND_IN,
        host_trend_fraction=_TREND_BAND_IN / _ANNUAL_HOST_ROW_IN,
        inset_bottom_fraction=(
            (_TREND_BAND_IN + _INSET_TICK_STRIP_IN) / _ANNUAL_HOST_ROW_IN
        ),
        inset_top_fraction=(
            (_TREND_BAND_IN + _INSET_TICK_STRIP_IN + _ANNUAL_INSET_HEIGHT_IN)
            / _ANNUAL_HOST_ROW_IN
        ),
        x_major_ticks=lambda spans: list(range(spans[0][1], spans[-1][2] + 1)),
        x_minor_ticks=lambda spans: [],
    )


def _headline_rows(headline: str, measure_in: float) -> list[str]:
    """The sheet headline split into lines that fit `measure_in` inches.

    Measured against a throwaway canvas rather than estimated from a character
    count, because the sheet is only as wide as the record it draws and the
    level name is part of the headline: `IRRI 146 (NSIC Rc158)` and
    `IR65620-192-3-3-3-2` each add about two and a half inches of type to a
    line that already very nearly fills a two-decade sheet. `figure.text` runs
    off the canvas without clipping and without warning, so an unmeasured
    headline fails silently and is only found by looking at the plate.

    A headline that already fits is returned untouched and costs the strip
    exactly one line, which is what it cost before this existed.
    """

    from matplotlib import pyplot as plt

    probe = plt.figure(figsize=(1.0, 1.0), dpi=150)
    try:
        renderer = probe.canvas.get_renderer()
        artist = probe.text(0.0, 0.0, headline, fontsize=_HEADLINE_FONT_SIZE)

        def fits(candidate: str) -> bool:
            artist.set_text(candidate)
            width = artist.get_window_extent(renderer=renderer).width / 150.0
            return width <= measure_in

        if fits(headline):
            return [headline]
        rows: list[str] = []
        current = ""
        for word in headline.split(" "):
            candidate = f"{current} {word}".strip()
            if current and not fits(candidate):
                rows.append(current)
                current = word
            else:
                current = candidate
        if current:
            rows.append(current)
        return rows
    finally:
        plt.close(probe)


def _write_bands_and_trend(
    scope: _VarietyScope,
    result,
    overlay,
    banding: _SheetBanding,
    year_groups: dict[int, tuple[str, ...]],
    limits,
    destination: Path,
    *,
    matched_colours: bool = False,
) -> bool:
    """One level's response-curve insets drawn on the stretch of its own yield trend.

    The level-scoped analogue of
    `../../by_planting_year/annotated/decades_and_trend.jpeg`:
    same pairing (each band's response-curve cloud sitting on the exact
    calendar stretch of the per-year trend it came from, one shared
    inches-per-t/ha between the two trend panels), rebuilt from the level's own
    per-year and per-decade groupings rather than a `FactorSubstructure`, since
    a level needs no stratum-size argument for either resolution.

    `banding` chooses that resolution -- decades, as the whole season is
    obliged to use, or one inset per recorded planting year, which only a
    stratum as regular as these are supports. Everything the two sheets do not
    share is in the record; nothing below branches on which one it was handed.

    `matched_colours` reproduces the reference folder's
    `decades_and_trend_matched_colours.jpeg` treatment: colour is spent on the
    quantity (each trend panel takes the colour its own points already carry
    in every inset) instead of on the applied-N era, which moves to marker
    shape. Written as a separate file, not a replacement -- see this
    function's caller.

    Returns whether a sheet was written: a banding that resolves to a single
    band is declined rather than drawn, since one inset over the whole trend
    restates the pooled figure the folder already sits beside.
    """

    from matplotlib import pyplot as plt

    inches_per_year = banding.inches_per_year
    host_row_in = banding.host_row_in
    trend_row_in = banding.trend_row_in
    host_trend_fraction = banding.host_trend_fraction
    inset_bottom_fraction = banding.inset_bottom_fraction
    inset_top_fraction = banding.inset_top_fraction

    labels = [label for label, _first, _last in banding.spans]
    total = sum(len(ids) for ids in banding.groups.values())
    records = _annual_records(year_groups, result)
    spans = banding.spans
    if len(labels) < 2 or len(records) < 2 or len(spans) != len(labels):
        return False

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

    year_span = x_hi - x_lo
    plot_width = inches_per_year * year_span
    # The axes keep their density; only the sheet is padded out to the floor,
    # and the axes are then centred in it. `max` returns the axes width itself
    # whenever it is the larger, so a sheet that already clears the floor is
    # laid out to the byte as it was before the floor existed.
    band_width = max(plot_width, _MIN_SHEET_BAND_IN)
    sheet_width = band_width + _SHEET_LEFT_IN + _SHEET_RIGHT_IN

    colours = _treatment_class_colours(overlay)
    zero_n_colour = colours.get(_ZERO_N_TREATMENT_CLASS, "C1")
    mineral_colour = next(
        (colour for label, colour in colours.items() if label != _ZERO_N_TREATMENT_CLASS),
        "C0",
    )

    mixed = [record.year for record in records if record.ladder_is_mixed]
    headline = banding.headline
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
        f"{total} cluster-eligible {scope.variety} trajectories over "
        f"{len(records)} planting years ({records[0].year}-{records[-1].year}); "
        "unlike the whole season's `by_planting_year/`, no stratum-size "
        f"argument is needed here -- {scope.variety} supports a per-year trend "
        f"directly, {banding.scope_tail}",
        f"the insets — every trajectory of the {banding.band_word} against "
        "applied N, each a recorded stratum that nothing was clustered to "
        "produce, and each on the same frame at the same size, so any two can "
        "be compared directly",
        banding.placement(unsampled, gap_years),
        "the two trend panels — yield at zero N by planting year, which is "
        "the left-hand end of every curve in the inset above it, and the "
        "response above zero N, which is the other end. Both are the mean of "
        f"that year's {scope.variety} replicate trajectories, the bar its "
        "standard error across replicates, and both are drawn to one shared "
        "scale so a change of a given size has the same slope in each",
        "each line is broken at every applied-N era boundary (dashed) and at "
        "any unrecorded year: a change across either is a change of "
        "experiment or a missing observation, not a response to fertilizer",
        *(
            [
                f"unrecorded planting "
                f"{'year' if len(scope.unobserved_years) == 1 else 'years'} "
                f"within the record: {_year_list(scope.unobserved_years)}"
            ]
            if scope.unobserved_years
            else []
        ),
        *(
            [
                "planting years the source records in full and the clustering "
                f"excludes whole, so the trend does not cross them: "
                f"{_year_list(scope.excluded_years)} "
                f"({', '.join(scope.exclusion_reasons)}). They are a gap in "
                "this figure, not in the experiment"
            ]
            if scope.excluded_years
            else []
        ),
        "variety is confounded with the calendar period it was grown in; a "
        f"difference between {banding.band_word_plural} here is not a genetic "
        f"effect, and the record behind it is {_design_site_phrase(scope)}",
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
    headline_rows = _headline_rows(headline, band_width)
    headline_strip_in = (
        len(headline_rows) * _HEADLINE_FONT_SIZE * 1.35 / 72.0 + 0.24
    )
    title_strip_in = (
        headline_strip_in + caption_lines * _CAPTION_FONT_SIZE * 1.45 / 72.0 + 0.45
    )

    height = (
        _SHEET_BOTTOM_IN
        + _TREND_LEGEND_IN
        + _TREND_XAXIS_IN
        + trend_row_in
        + _TREND_GAP_IN
        + host_row_in
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

    plot_left = _SHEET_LEFT_IN + (band_width - plot_width) / 2.0

    def _year_to_inches(year: float) -> float:
        return plot_left + plot_width * (year - x_lo) / year_span

    response_bottom = _SHEET_BOTTOM_IN + _TREND_LEGEND_IN + _TREND_XAXIS_IN
    host_bottom = response_bottom + trend_row_in + _TREND_GAP_IN
    inset_legend_bottom = host_bottom + host_row_in

    eras = annual_ladder_eras(records)
    era_colours = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(eras), 1)))

    try:
        trend_panels = (
            (
                "yield_at_zero_n_t_ha",
                "Yield at zero N (t/ha)",
                host_bottom,
                host_row_in,
                zero_n_colour,
            ),
            (
                "response_above_zero_n_t_ha",
                "Response above zero N (t/ha)",
                response_bottom,
                trend_row_in,
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
                y=host_trend_fraction / 2 if is_host else 0.5,
                fontsize=_AXIS_LABEL_FONT_SIZE,
            )
            axes.grid(axis="y", alpha=0.25, linewidth=0.6)
            axes.tick_params(
                labelsize=_TICK_FONT_SIZE,
                labelright=plot_width > _WIDE_SHEET_LABEL_IN,
            )
            axes.set_yticks(shared_ticks)
            axes.set_ylim(
                floor,
                floor + (ceiling - floor) / host_trend_fraction if is_host else ceiling,
            )

        trend_axes[0].set_xlim(x_lo, x_hi)
        trend_axes[0].tick_params(labelbottom=False)
        trend_axes[-1].set_xticks(banding.x_major_ticks(spans))
        trend_axes[-1].set_xticks(banding.x_minor_ticks(spans), minor=True)
        trend_axes[-1].set_xlabel("Planting year", fontsize=_AXIS_LABEL_FONT_SIZE)

        inset_axes: list[Any] = []
        for index, (label, (band_lo, band_hi)) in enumerate(
            zip(labels, visible, strict=True)
        ):
            ids = banding.groups[label]
            band_left = _year_to_inches(band_lo)
            band_right = _year_to_inches(band_hi)
            panel_inches = max(band_right - band_left - _PANEL_GUTTER_IN, 0.4)
            axes = figure.add_axes(
                _box(
                    band_left + _PANEL_GUTTER_IN / 2,
                    host_bottom + inset_bottom_fraction * host_row_in,
                    panel_inches,
                    (inset_top_fraction - inset_bottom_fraction) * host_row_in,
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
            axes.set_xticks(
                _INSET_TICKS
                if panel_inches >= _DENSE_INSET_TICK_IN
                else _INSET_TICKS[:-1]
            )
            axes.tick_params(labelsize=_INSET_TICK_FONT_SIZE)
            axes.set_title(label, fontsize=_INSET_TITLE_FONT_SIZE, pad=6)
            axes.text(
                0.97,
                0.03,
                banding.readout(ids, result, total, panel_inches),
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
            (host_bottom + (inset_bottom_fraction - 0.05) * host_row_in) / height,
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
            "\n".join(headline_rows),
            ha="center",
            va="top",
            fontsize=_HEADLINE_FONT_SIZE,
            color="0.10",
            # Passed only when it does something, so a one-line headline is
            # laid out by exactly the call that drew it before wrapping
            # existed.
            **({"linespacing": 1.35} if len(headline_rows) > 1 else {}),
        )
        caption_top = headline_top - headline_strip_in / height
        # A fraction of the plot width is the right unit only while the sheet
        # is about as wide as two columns of text. Past that the columns are
        # measured in inches instead, or the year sheet opens a 30-inch gap
        # between them; the decade sheet is narrow enough that the fraction
        # still wins.
        caption_column_offset = min(
            _CAPTION_COLUMN_OFFSET, _CAPTION_COLUMN_MEASURE_IN / band_width
        )
        for column_index, column in enumerate(caption_columns):
            figure.text(
                (_SHEET_LEFT_IN + column_index * band_width * caption_column_offset)
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
    return True


def _write_annual_trend(
    scope: _VarietyScope,
    result,
    year_groups: dict[int, tuple[str, ...]],
    destination: Path,
) -> bool:
    """One level's zero-N yield and response above zero N against planting year."""

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
                    label=(
                        f"{scope.variety}, {scope.season}"
                        if (axes is axes_list[0] and run_index == 0)
                        else None
                    ),
                )
            # Mark the applied-N era boundary within each contiguous run --
            # IR8 runs 0/40/60/120 and 0/60/100/140 in 1968-1969 and
            # 0/50/100/150 from 1970, IR72 changes ladder with its design. A
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
        mixed_years = [year for year in years if by_year[year][2]]
        _reserve_suptitle(
            figure,
            _wrap_title_lines(
                [
                    f"source={SOURCE_NAME} — {scope.season_text}, "
                    f"{scope.variety}: observed response by planting year",
                    f"{len(years)} years, {years[0]}-{years[-1]}; each point is the "
                    f"mean of that year's {scope.variety} replicate trajectories "
                    # Named, not deferred to "except where noted": nothing else
                    # on this plate notes it, and IR58 really does lose a
                    # replicate in 1986.
                    + (
                        f"({scope.replicate_counts[0][0]} in every year), the "
                        "bar its standard error across replicates"
                        if len(scope.replicate_counts) == 1
                        else f"({scope.replicate_counts[0][0]}, except "
                        f"{_replicate_exceptions(scope)}), the bar its "
                        "standard error across replicates"
                    ),
                    "the line is broken at every applied-N era boundary (dashed) "
                    "and at any year with no panel: a change across either is a "
                    "change of experiment or a missing observation, not a "
                    "response to fertilizer",
                    *(
                        [f"years with no point: {_gap_inventory(scope)}"]
                        if scope.gaps
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
    return True


def _write_readme(
    scope: _VarietyScope,
    year_groups: dict[int, tuple[str, ...]],
    decade_groups: dict[str, tuple[str, ...]],
    written: Sequence[str],
    destination: Path,
) -> None:
    """The folder note, describing only the plates this level actually got.

    A level whose record sits inside one calendar decade has no decade sheet
    and no decade grid, so the note does not offer them; the entries below are
    filtered against what was written rather than listed as a fixed contents
    page that would be wrong for three of the eight levels.
    """

    years = sorted(year_groups)
    total = sum(len(ids) for ids in year_groups.values())
    decades = sorted(decade_groups)
    entries: list[tuple[str, str]] = [
        (
            "panel_grid.jpeg",
            "- `panel_grid.jpeg` -- every planting year on its own panel, on "
            "the frame shared by the whole product. A recorded stratum: "
            "nothing was clustered to produce it.",
        ),
        (
            "decade_panel_grid.jpeg",
            f"- `decade_panel_grid.jpeg` -- the same trajectories banded into "
            f"{len(decades)} calendar decades ({', '.join(decades)}), the same "
            "banding `../../by_planting_year/` uses for the whole season -- "
            f"{scope.variety} does not need it for stratum size, but it is "
            "offered as the coarser read.",
        ),
        (
            "annual_trend.jpeg",
            "- `annual_trend.jpeg` -- the same years as one trend, zero-N "
            "yield and response above zero N against planting year, line "
            "broken at every applied-N era boundary and at every year with no "
            "panel.",
        ),
        (
            "decades_and_trend.jpeg",
            "- `decades_and_trend.jpeg` -- both resolutions on one sheet, with "
            "each decade\'s response-curve panel set over the exact stretch of "
            "the year axis its trajectories came from and drawn as wide as "
            "that stretch, so a panel and its shaded band are the same "
            "interval of calendar time read two ways -- the level-scoped "
            "analogue of "
            "`../../by_planting_year/annotated/decades_and_trend.jpeg`.",
        ),
        (
            "decades_and_trend_matched_colours.jpeg",
            "- `decades_and_trend_matched_colours.jpeg` -- the same sheet with "
            "colour spent on the quantity instead of the applied-N era: orange "
            "for yield at zero N, blue for response above zero N, matching "
            "each trend panel to its own points in every inset above it. The "
            "era it displaces is re-encoded as marker shape, so nothing the "
            "plain sheet says is lost.",
        ),
        (
            "years_and_trend.jpeg",
            f"- `years_and_trend.jpeg` -- the same sheet at the resolution the "
            f"trend beneath it is already drawn at: one inset per recorded "
            f"planting year, each over its own single year of the axis, "
            f"{len(years)} of them"
            + (
                f" instead of {len(decades)}. The decade sheet asks what "
                "changed between eras; this one asks what each season looked "
                "like, and answers it without the reader having to accept "
                "that a decade is one thing"
                if "decades_and_trend.jpeg" in written
                else ". The whole record sits inside one calendar decade, so "
                "this is the only resolution the sheet is drawn at"
            )
            + (
                f". The stretches with no inset over them -- "
                f"{_gap_inventory(scope)} -- are left as blank axis rather "
                "than closed up, so the spacing stays calendar time"
                if scope.gaps
                else ""
            )
            + ".",
        ),
        (
            "years_and_trend_matched_colours.jpeg",
            "- `years_and_trend_matched_colours.jpeg` -- the year sheet under "
            "the same colour treatment as "
            "`decades_and_trend_matched_colours.jpeg`."
            if "decades_and_trend_matched_colours.jpeg" in written
            else "- `years_and_trend_matched_colours.jpeg` -- the year sheet "
            "with colour spent on the quantity instead of the applied-N era: "
            "orange for yield at zero N, blue for response above zero N. The "
            "era it displaces is re-encoded as marker shape, so nothing the "
            "plain sheet says is lost.",
        ),
    ]
    lines = [
        f"# {scope.season_text} — {scope.variety}, by planting year",
        "",
        f"{total} cluster-eligible LTCCE {scope.season_text} trajectories "
        f"carrying `Variety={scope.variety!r}`, {years[0]}-{years[-1]}, one "
        "panel per recorded planting year -- the year-by-year split of "
        f"`../{scope.stem}.jpeg`, which pools them.",
        "",
        # The two prefixes in this directory count from different events, and
        # a reader who assumes otherwise reads a five-year discrepancy as an
        # error in one of them. Stated here rather than only in the parent
        # note, because the folder is what was opened.
        f"This folder is filed under `{scope.first_year}`, the first planting "
        f"year {scope.variety} appears in the {scope.season} record. "
        + (
            f"`../{scope.stem}.jpeg` beside it is filed under "
            f"`{scope.release_year}`, the NSIC release year, so the folders in "
            "this directory sort in the order the experiment took the "
            "varieties up and the figures sort in the order they were "
            "registered."
            if scope.release_year is not None and scope.release_year != scope.first_year
            else f"That is also the NSIC release year `../{scope.stem}.jpeg` "
            "beside it is filed under, so the two agree here; for most levels "
            "in this directory they do not."
            if scope.release_year is not None
            else "It is a breeding-line designation with no release year on "
            f"record, so `../{scope.stem}.jpeg` beside it carries no year at "
            "all and this folder is the only place the level is dated."
        ),
        "",
        "**Variety is confounded with the calendar period it was grown in; a "
        "difference between years here is not a genetic effect.**",
        "",
        "## Why individual years, unlike `../../by_planting_year/`",
        "",
        "The whole-season `by_planting_year/` folder bands into decades "
        "because no single calendar year clears this product\'s "
        f"{MIN_FACTOR_STRATUM}-trajectory stratum minimum. One variety level "
        f"needs no such banding: {scope.variety} carries "
        f"{_replicate_phrase(scope, years)}, under "
        f"{_design_site_phrase(scope)}. Those counts are read off this level\'s "
        "own record, not assumed from a neighbouring one.",
        "",
    ]
    if scope.gaps:
        lines.extend(
            [
                "## Years with no panel",
                "",
                f"{_gap_inventory(scope)}.",
                "",
                "A year nothing was observed in and a year the source records "
                "in full but the clustering excludes whole are both holes in "
                "these figures, and only the first is a hole in the "
                "experiment. They are labelled apart everywhere in this folder "
                "for that reason -- the parent folder\'s combined sheet calls "
                "the second kind \"no DS record\", which is not what the "
                "source says.",
                "",
            ]
        )
    lines.extend(
        [
            "## What is in this folder",
            "",
            *[entry for name, entry in entries if name in written],
            "",
            "## Interpretation boundary",
            "",
            "Exploratory diagnostic, outside the governed analysis inventory "
            "(ANA-11). No curve is fitted anywhere in this folder; every mean "
            "is computed from observed yields at observed N rates. Standard "
            "errors on `annual_trend.jpeg` are the spread across each year\'s "
            "own replicate trajectories, not a fitted quantity\'s sampling "
            "error.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def _roster(
    result,
    season: str,
    requested: Sequence[str] | None,
    minimum: int,
) -> tuple[str, ...]:
    """Which variety levels get a folder, largest first.

    The default roster is the one the season\'s own `by_variety/` folder
    already draws -- every level carrying `MIN_FACTOR_STRATUM` trajectories or
    more -- so a folder exists exactly where a pooled figure does and the two
    can be read against each other. Ties are broken by name, which is the order
    that folder\'s own table uses.
    """

    profile = result.profiles.get(season)
    if profile is None:
        raise SystemExit(f"No {season!r} season profile in this clustering result")
    counts = collections.Counter(
        result.contexts[trajectory_id].variety.strip()
        for trajectory_id in profile.trajectory_ids
        if trajectory_id in result.contexts
    )
    ordered = [
        level for level, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    if requested:
        unknown = [level for level in requested if level not in counts]
        if unknown:
            raise SystemExit(
                f"No {season} trajectories recorded for "
                + ", ".join(repr(level) for level in unknown)
                + f"; the {season} profile records {len(counts)} levels"
            )
        return tuple(level for level in ordered if level in set(requested))
    return tuple(level for level in ordered if counts[level] >= minimum)


def _write_variety_folder(
    scope: _VarietyScope,
    result,
    overlay,
    year_groups: dict[int, tuple[str, ...]],
    decade_groups: dict[str, tuple[str, ...]],
    limits,
    destination: Path,
) -> tuple[str, ...]:
    """Render one level\'s plates into a staging directory and promote it in place."""

    publication_container = _nested_publication_container(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A sibling of the destination, not of the container: `promote_directory`
    # renames the staging directory into place and refuses a staging path that
    # is not `destination.parent`\'s own child, because a rename across two
    # parents is not the atomic swap the whole publication path is built on.
    # The container is what the *lock* is taken on -- it names the staging
    # directory only so an operator who finds one knows which publisher owns
    # it.
    staging = destination.with_name(
        f".{publication_container.name}.{destination.name}.nested-staging."
        f"{uuid.uuid4().hex}"
    )
    staging.mkdir()
    written: list[str] = []
    try:
        if _write_year_panel_grid(
            scope, result, overlay, year_groups, limits, staging / "panel_grid.jpeg"
        ):
            written.append("panel_grid.jpeg")
        if _write_decade_panel_grid(
            scope,
            result,
            overlay,
            decade_groups,
            year_groups,
            limits,
            staging / "decade_panel_grid.jpeg",
        ):
            written.append("decade_panel_grid.jpeg")
        if _write_annual_trend(scope, result, year_groups, staging / "annual_trend.jpeg"):
            written.append("annual_trend.jpeg")
        decade_sheet = len(decade_groups) > 1
        for banding, stem in (
            (_decade_banding(scope, decade_groups), "decades_and_trend"),
            (
                _year_banding(scope, year_groups, decade_sheet=decade_sheet),
                "years_and_trend",
            ),
        ):
            for suffix, matched in (("", False), ("_matched_colours", True)):
                name = f"{stem}{suffix}.jpeg"
                if _write_bands_and_trend(
                    scope,
                    result,
                    overlay,
                    banding,
                    year_groups,
                    limits,
                    staging / name,
                    matched_colours=matched,
                ):
                    written.append(name)
        _write_readme(
            scope, year_groups, decade_groups, tuple(written), staging / "README.md"
        )
        written.append("README.md")

        with _core_overlay_publication_lock(publication_container):
            _promote_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return tuple(written)


def main() -> int:
    args = _parse_args()
    source_path, encoding = _load_source_spec(args.config)

    overlay = read_source_dataset_overlay(source_path, SOURCE_NAME, encoding=encoding)
    contexts = read_ltcce_contexts(source_path, encoding=encoding)
    result = build_season_clustering(overlay, contexts)
    limits = _shared_axis_limits(overlay)

    varieties = _roster(result, args.season, args.variety, args.min_trajectories)
    if not varieties:
        raise SystemExit(
            f"No {args.season} variety level carries {args.min_trajectories} "
            "or more cluster-eligible trajectories"
        )
    output_root = args.output_root or _default_output_root(args.season)

    for variety in varieties:
        year_groups = _year_groups(result, args.season, variety)
        if not year_groups:
            raise SystemExit(
                f"No {variety!r} trajectories found in the "
                f"{args.season!r} season profile"
            )
        scope = _variety_scope(result, args.season, variety, year_groups)
        decade_groups = _decade_groups(year_groups)
        destination = _plain_absolute_path(
            args.output_dir if args.output_dir is not None else output_root / scope.folder_name,
            label=f"{variety} output destination",
        )
        written = _write_variety_folder(
            scope, result, overlay, year_groups, decade_groups, limits, destination
        )
        total = sum(len(ids) for ids in year_groups.values())
        print(
            f"Wrote {', '.join(written)} for {total} {variety} trajectories "
            f"over {len(year_groups)} planting years under {destination}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
