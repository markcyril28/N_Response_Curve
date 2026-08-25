#!/usr/bin/env python3
"""Draw LTCCE dry-season planting-year plates, one folder per variety.

`by_season/ds/by_planting_year/` has no 1992-2001 in it. The data are not
missing -- `LTCCE_GY_processed.{xlsx,csv}` carry that window in full -- but
every dry-season trajectory in it records two yields at each applied-N level
with nothing in the source to tell the two apart, so
`response_curve_clusters.describe_trajectory` fail-closes it as
`duplicated_n_level_mv_028` and the whole stretch never reaches a cluster
profile. The folder's decade table therefore runs `1990s = 1990-1991` and jumps
to `2000s = 2002-2009`, and its combined sheet labels the stretch "no DS
record", which is not what the source says.

This recipe draws that window back in, from every recorded trajectory instead
of the cluster-eligible ones, for the groups that carry it -- and for IR8,
which carries none of it and is the contrast:

    IR8                  DS 1968-1990 -- entirely outside the window; every
                                         planting year reaches the trend
    IR72                 DS 1989-2012 -- spans the window on both sides
    IR59682-132-1-1-2    DS 1993-1998 -- wholly inside it
    IR60819-*            DS 1993-1998 -- wholly inside it (two lines, never
                                         in the same planting year)
    PSBRc52              DS 1999-2009 -- crosses the far edge

Each group gets its own folder holding the two idioms the parent folder uses:
a decades-and-trend reference sheet on `reference_sheet_layout`'s geometry, and
a panel-per-planting-year grid. One cross-variety plate of the 1992-2001 window
sits at the root beside the folder note.

MV-028 is respected, not worked around, in three places. A duplicated level is
drawn as the two observations the source records, never averaged into one point
and never paired off into two pseudo-ladders -- the file puts the two rows of a
cell adjacent to one another, but nothing in it links the first record at zero
N to the first record at 65 kg N/ha. Its connector is therefore a *band*
spanning the range recorded at each level rather than a line through one path;
a clean trajectory's band collapses to the ordinary connecting line, so one
rule serves both. And a duplicated year carries no zero-N/response read-out and
never reaches a trend panel, because neither summary is defined for a
trajectory with two yields at zero N.

Exploratory diagnostic, outside the governed release inventory (ANA-11): no
curve is fitted, and nothing here is a governed analysis family.

Output lands in a *subdirectory* of the season's `figure_only/` folder, beside
`by_replicate/`. That is deliberate:
`generate_response_curve_season_clusters._carry_unmanaged_directories` carries
whole directories it did not build across a `by_season/` rebuild but lets loose
files inside a directory it did build be destroyed, so plates written flat into
`figure_only/` would not survive the next season-cluster regeneration.

Usage:
    conda run -n n_response python \\
      modules/n_response_curve/reporting/generate_ltcce_ds_variety_planting_year_views.py \\
      --config scriptCONFIG.toml
"""

from __future__ import annotations

import argparse
import collections
import math
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from matplotlib import transforms
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.reporting.figure_axis_frames import (  # noqa: E402
    shared_axis_limits as _shared_axis_limits,
)
from n_response_curve.reporting.figure_captions import (  # noqa: E402
    COMPOSITION_TITLE_WIDTH as _COMPOSITION_TITLE_WIDTH,
    EXPLORATORY_DIAGNOSTIC_DISCLAIMER as _DISCLAIMER,
    TITLE_FONT_SIZE as _TITLE_FONT_SIZE,
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
    PLANTING_YEAR_FIGURE_ONLY_DIRNAME,
    _season_token,
    _treatment_class_colours,
)
from n_response_curve.reporting.response_curve_clusters import (  # noqa: E402
    EXCLUSION_DUPLICATED_N_LEVEL,
    EXCLUSION_INCOMPLETE_LADDER,
    EXCLUSION_NO_ZERO_N_ANCHOR,
    RESPONSE_TYPE_FEATURES,
    TrajectoryContext,
    TrajectoryFeatures,
    centroid_summary,
    describe_trajectory,
    read_ltcce_contexts,
    subset_overlay,
)
from n_response_curve.reporting.response_curve_season_clusters import (  # noqa: E402
    AnnualRecord,
    annual_ladder_eras,
    cluster_year_range,
    decade_band,
    format_shares,
    ladder_text,
    season_label,
)
from n_response_curve.reporting.source_config_spec import (  # noqa: E402
    load_source_spec,
)
from n_response_curve.reporting.source_dataset_overlays import (  # noqa: E402
    SourceDatasetOverlay,
    SourceTrajectory,
    _adaptive_style,
    read_source_dataset_overlay,
)

SOURCE_NAME = "ltcce"
DEFAULT_SEASON = "DS"
SUPPORTED_SEASONS = ("DS", "EWS", "LWS")
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scriptCONFIG.toml"
CLUSTERS_ROOT = PROJECT_ROOT / "WF/04_Response_Curves/z_n_response_full/ltcce/clusters"

# Sibling of PLANTING_YEAR_REPLICATE_DIRNAME. Named here rather than in the
# season-cluster generator because that generator does not own this output and
# must keep treating it as an unmanaged directory to carry across a rebuild.
PLANTING_YEAR_VARIETY_DIRNAME = "by_variety"

# Sibling of `by_variety/`: the same two plates with every recorded variety
# pooled into one membership. It is the parent folder's own season sheet redrawn
# from every recorded trajectory instead of the cluster-eligible ones, so the
# window it labels "no DS record" appears as a cloud.
PLANTING_YEAR_COMBINED_DIRNAME = "all_variety_combined"


README_FILENAME = "README.md"
FIGURE_SUFFIX = "_figure.jpeg"

# The stretch the cluster-eligible planting-year view has no panel for.
MV_028_WINDOW = (1992, 2001)
WINDOW_PLATE_STEM = "mv_028_window_1992_2001"

PANEL_COLUMNS = 5
PANEL_WIDTH_IN = 6.0
PANEL_HEIGHT_IN = 6.4
PANEL_SCALE_YEAR_LIMIT = 24
PANEL_SCALE_FACTOR = 0.68

# Warm tint marking a panel whose every trajectory is MV-028 duplicated, so the
# excluded block reads as a block across the grid rather than one title at a
# time.
MV_028_PANEL_FACECOLOUR = "#fdf2e4"
MV_028_TITLE_COLOUR = "#8c2d04"
CLEAN_PANEL_FACECOLOUR = "white"

MV_028_PANEL_NOTE = "duplicated N levels (MV-028) — not summarized"

# The tint carries one meaning everywhere in this product: the records are
# there and drawn, and only the summary is withheld. The cause varies by year
# (MV-028 on most of them, a missing zero-N anchor at 1986), so the key states
# the rule and points at the title or span label that names the cause, rather
# than asserting a cause the plate may not have.
EXCLUDED_TINT_LABEL = (
    "orange tint — recorded and drawn, not summarized: one unusable "
    "replicate withholds the whole year (label names why)"
)


@dataclass(frozen=True)
class VarietyGroup:
    """One plate's membership rule and how it names itself."""

    slug: str
    display: str
    matches: Callable[[str], bool]
    # Drawn on panel titles when one plate carries more than one recorded
    # `Variety` string, so a panel always names the line it actually holds.
    names_line_per_panel: bool = False


# Pooled views the roster would not produce on its own. A curated group sits
# *beside* the per-variety folders rather than replacing them: `ir60819/` pools
# two released lines that never share a dry-season planting year, and each line
# still gets its own folder.
COMBINED_GROUP = VarietyGroup(
    "all_varieties",
    "all varieties",
    lambda variety: bool(variety),
)

CURATED_GROUPS = (
    VarietyGroup(
        "ir60819",
        "IR60819-*",
        lambda variety: variety.startswith("IR60819"),
        names_line_per_panel=True,
    ),
)


def _slugify(name: str) -> str:
    """Folder name for a recorded variety, from the recorded string alone.

    Recorded names carry spaces, brackets and dots -- `IRRI 156 (NSIC Rc238)`,
    `Taichung (N)1` -- so the folder cannot be the name. Runs of anything that
    is not alphanumeric collapse to one underscore, which is lossy; the caller
    resolves collisions rather than assuming the map is injective.
    """

    out: list[str] = []
    for character in name.strip().casefold():
        if character.isalnum():
            out.append(character)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_") or "variety_unrecorded"


def _exact_matcher(name: str) -> Callable[[str], bool]:
    """A matcher bound to one recorded name, not to a loop variable."""

    return lambda variety: variety == name


def discover_variety_groups(
    contexts: Mapping[str, TrajectoryContext],
    overlay: SourceDatasetOverlay,
    season: str,
) -> tuple[VarietyGroup, ...]:
    """Every variety the season records, oldest first, plus the curated pools.

    Ordered by first recorded planting year so the folder listing reads as the
    order the experiment ran them, and so the roster is stable across runs
    rather than following dictionary order.
    """

    drawable = {trajectory.trajectory_id for trajectory in overlay.trajectories}
    first_year: dict[str, int] = {}
    for trajectory_id, context in contexts.items():
        if trajectory_id not in drawable:
            continue
        if context.season.strip().upper() != season:
            continue
        variety = context.variety.strip()
        year = int(context.year)
        if not variety or year <= 0:
            continue
        first_year[variety] = min(first_year.get(variety, year), year)

    groups: list[VarietyGroup] = []
    taken: set[str] = set()
    for name in sorted(first_year, key=lambda variety: (first_year[variety], variety)):
        slug = base = _slugify(name)
        suffix = 2
        while slug in taken:
            slug = f"{base}_{suffix}"
            suffix += 1
        taken.add(slug)
        groups.append(VarietyGroup(slug, name, _exact_matcher(name)))

    for curated in CURATED_GROUPS:
        if curated.slug in taken:
            continue
        if any(curated.matches(name) for name in first_year):
            taken.add(curated.slug)
            groups.append(curated)
    return tuple(groups)


def default_output_dir(season: str) -> Path:
    """The `by_variety/` folder inside that season's planting-year plates.

    Derived from the season the same way
    `generate_ltcce_ds_replicate_reference_sheets.default_output_dir` derives
    `by_replicate/`, so this recipe and the season-cluster generator cannot
    drift apart on where the plates belong.
    """

    return (
        CLUSTERS_ROOT
        / "by_season"
        / _season_token(season)
        / "by_planting_year"
        / PLANTING_YEAR_FIGURE_ONLY_DIRNAME
        / PLANTING_YEAR_VARIETY_DIRNAME
    )


def default_combined_dir(season: str) -> Path:
    """The `all_variety_combined/` folder beside `by_variety/`."""

    return (
        CLUSTERS_ROOT
        / "by_season"
        / _season_token(season)
        / "by_planting_year"
        / PLANTING_YEAR_FIGURE_ONLY_DIRNAME
        / PLANTING_YEAR_COMBINED_DIRNAME
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--season",
        default=DEFAULT_SEASON,
        choices=SUPPORTED_SEASONS,
        help="Recorded LTCCE season the plates are drawn for (default: DS)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination for the plates (default: the season's own "
            f"by_planting_year/{PLANTING_YEAR_FIGURE_ONLY_DIRNAME}/"
            f"{PLANTING_YEAR_VARIETY_DIRNAME}/)"
        ),
    )
    parser.add_argument(
        "--combined-output-dir",
        type=Path,
        default=None,
        help=(
            "Destination for the pooled all-variety plates (default: the "
            f"season's own by_planting_year/{PLANTING_YEAR_FIGURE_ONLY_DIRNAME}/"
            f"{PLANTING_YEAR_COMBINED_DIRNAME}/)"
        ),
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Skip the pooled all-variety plates.",
    )
    parser.add_argument(
        "--no-window-plate",
        action="store_true",
        help=(
            "Skip the combined "
            f"{MV_028_WINDOW[0]}-{MV_028_WINDOW[1]} plate and write only the "
            "per-variety ones."
        ),
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.season)
    if args.combined_output_dir is None:
        args.combined_output_dir = default_combined_dir(args.season)
    return args


# --------------------------------------------------------------------------
# Membership
# --------------------------------------------------------------------------


def _season_year_groups(
    contexts: Mapping[str, TrajectoryContext],
    overlay: SourceDatasetOverlay,
    season: str,
    matches: Callable[[str], bool],
) -> dict[int, tuple[str, ...]]:
    """Recorded planting year -> that year's trajectory ids, eligibility ignored.

    Deliberately not routed through `build_season_clustering`: most of what this
    recipe exists to draw is what clustering excludes, so a cluster-eligible
    membership would return an empty plate for two of the groups.

    *Drawability* is a different question from eligibility and is enforced here.
    `read_ltcce_contexts` reads the factor columns and so keeps a trajectory
    whose every yield is `NA`, while the overlay drops it for having no finite
    observation; IR8's 1984 is one. Such a year has nothing to plot, and asking
    for its overlay raises rather than returning an empty panel.
    """

    drawable = {trajectory.trajectory_id for trajectory in overlay.trajectories}
    grouped: dict[int, list[str]] = collections.defaultdict(list)
    for trajectory_id, context in contexts.items():
        if trajectory_id not in drawable:
            continue
        if context.season.strip().upper() != season:
            continue
        if context.year <= 0 or not matches(context.variety.strip()):
            continue
        grouped[int(context.year)].append(trajectory_id)
    return {
        year: tuple(sorted(ids)) for year, ids in sorted(grouped.items()) if ids
    }


def _duplicated_levels(trajectory: SourceTrajectory) -> bool:
    """True when the source records more than one yield at some applied-N level."""

    counts = collections.Counter(
        observation.n_rate_kg_ha for observation in trajectory.observations
    )
    return any(count > 1 for count in counts.values())


def _distinct_ladder(trajectories: Sequence[SourceTrajectory]) -> tuple[str, bool]:
    """The applied-N levels a panel holds, and whether its members disagree.

    Distinct levels, not the ladder `describe_trajectory` returns: an MV-028
    trajectory has no ladder, and inventing one by de-duplicating would assert
    exactly the structure the exclusion says the source does not carry.
    """

    ladders = {
        tuple(sorted({observation.n_rate_kg_ha for observation in trajectory.observations}))
        for trajectory in trajectories
    }
    if not ladders:
        return "unknown", False
    counts = collections.Counter(ladders)
    dominant, _ = counts.most_common(1)[0]
    return ladder_text(dominant), len(ladders) > 1


def _clean_year_means(
    trajectories: Sequence[SourceTrajectory],
) -> tuple[float, float] | None:
    """Replicate mean zero-N yield and response, or None if any member is MV-028.

    All-or-nothing on purpose. A year whose members are partly duplicated would
    otherwise be summarized from whichever replicates happened to survive, and
    the read-out would silently describe a different denominator than the panel.
    """

    described = [describe_trajectory(trajectory) for trajectory in trajectories]
    if any(isinstance(item, str) for item in described):
        return None
    matrix = np.asarray(
        [
            [item.yield_at_zero_n_t_ha, item.response_above_zero_n_t_ha]
            for item in described
        ],
        dtype=float,
    )
    means = matrix.mean(axis=0)
    return float(means[0]), float(means[1])


def _exclusion_reasons(
    trajectories: Sequence[SourceTrajectory],
) -> collections.Counter[str]:
    reasons: collections.Counter[str] = collections.Counter()
    for trajectory in trajectories:
        described = describe_trajectory(trajectory)
        if isinstance(described, str):
            reasons[described] += 1
    return reasons


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def _draw_year_panel(
    axes,
    subset: SourceDatasetOverlay,
    colours: Mapping[str, object],
    style: tuple[float, float, float],
    *,
    label_series: bool,
) -> None:
    """Raw observed points, and a connecting line only where one is defined.

    A duplicated trajectory gets a band rather than a line: joining its
    observations in recorded order would draw a zig-zag that reads as a
    response shape the source never measured, and pairing them by order of
    appearance would assert a series the source does not record.
    """

    marker_size, marker_alpha, line_alpha = style

    for treatment_class in subset.summary.treatment_classes:
        xs: list[float] = []
        ys: list[float] = []
        for trajectory in subset.trajectories:
            for observation in trajectory.observations:
                if observation.treatment_class == treatment_class:
                    xs.append(observation.n_rate_kg_ha)
                    ys.append(observation.yield_t_ha)
        if not xs:
            continue
        axes.scatter(
            xs,
            ys,
            s=marker_size,
            alpha=marker_alpha,
            color=colours[treatment_class],
            label=treatment_class if label_series else "_nolegend_",
            zorder=3,
        )

    _draw_connecting_paths(
        axes, subset.trajectories, line_alpha, label_series=label_series
    )
    axes.grid(alpha=0.2, linewidth=0.6)


# --------------------------------------------------------------------------
# Connecting paths
# --------------------------------------------------------------------------

# Kept short because these sit in a single legend row beside the tint key; the
# caption block carries the long form.
CONNECTING_LINE_LABEL = "connector — one trajectory (not a fit)"
CONNECTING_BAND_LABEL = "band — both records, no path chosen (MV-028)"


# Width of one legend key box plus its gap, and the mean glyph width, both as
# a fraction of the type size. Only used to choose a wrap, so an estimate is
# enough; the result is measured afterwards either way.
_LEGEND_HANDLE_IN = 0.45
_LEGEND_CHAR_RATIO = 0.52


def _legend_wrap_chars(available_in: float, fontsize: float, columns: int) -> int:
    per_column_in = available_in / max(columns, 1) - _LEGEND_HANDLE_IN
    char_in = fontsize * _LEGEND_CHAR_RATIO / 72.0
    return max(16, int(per_column_in / char_in))


def _fitted_legend(
    figure,
    handles,
    labels,
    available_in: float,
    fontsize: float,
    *,
    wrap: bool = False,
    max_lines: int = 2,
    **kwargs,
):
    """Draw a legend at the largest type that fits the width it was given.

    These plates size themselves from their own content, so the sheet a variety
    earns and the number of keys it needs are independent: a one-year variety
    gets a six-inch canvas and the full key row, and a two-decade one gets both
    connector keys on a narrow sheet.

    `wrap` folds the keys onto fewer columns and more lines, which is right
    where the legend sits under the panels and can grow downward. The reference
    sheet leaves it off: its key row has a reserved band exactly one line tall,
    so there the type shrinks and the row stays put.

    The legend is rebuilt rather than restyled in place because its padding and
    handle lengths are measured in font-size units: setting the label sizes
    alone leaves the spacing at the original scale and the row still overruns.
    """

    columns = max(len(labels), 1)
    texts = list(labels)
    if wrap and available_in > 0:
        while columns > 1:
            folded = [
                textwrap.fill(
                    label, width=_legend_wrap_chars(available_in, fontsize, columns)
                )
                for label in labels
            ]
            if max(text.count("\n") for text in folded) + 1 <= max_lines:
                texts = folded
                break
            columns -= 1
        else:
            texts = [
                textwrap.fill(label, width=_legend_wrap_chars(available_in, fontsize, 1))
                for label in labels
            ]

    legend = figure.legend(handles, texts, ncol=columns, fontsize=fontsize, **kwargs)
    if available_in <= 0:
        return legend
    renderer = figure.canvas.get_renderer()
    size = fontsize
    floor = fontsize * 0.55
    for _ in range(4):
        width_in = legend.get_window_extent(renderer).width / figure.dpi
        if width_in <= available_in or size <= floor:
            break
        size = max(size * available_in / width_in * 0.98, floor)
        legend.remove()
        legend = figure.legend(handles, texts, ncol=columns, fontsize=size, **kwargs)
    return legend


def _plate_title_width(figure) -> int:
    """Fold the caption to the canvas the plate actually got.

    The shared default is sized for the wide plates; a variety recorded in one
    planting year gets a single-panel canvas, and the caption ran off both
    edges of it.
    """

    char_in = _TITLE_FONT_SIZE * _LEGEND_CHAR_RATIO / 72.0
    return max(40, min(_COMPOSITION_TITLE_WIDTH,
                       int((figure.get_figwidth() - 0.5) / char_in)))


def _legend_strip(figure, legend) -> float:
    """The bottom strip this legend needs, as a fraction of figure height.

    `reserve_suptitle` reserves a fixed strip for a single key row, which a
    wrapped legend under a narrow plate outgrows: constrained_layout makes no
    room for figure-level artists, so the panels would sit on top of it.
    """

    renderer = figure.canvas.get_renderer()
    height_in = legend.get_window_extent(renderer).height / figure.dpi
    return max(0.035, height_in / figure.get_figheight() + 0.012)


def _excluded_tint_handle() -> Patch:
    """The key for the tint, drawn as the tint itself rather than described."""

    return Patch(
        facecolor=MV_028_PANEL_FACECOLOUR,
        edgecolor="0.55",
        linewidth=0.6,
        label=EXCLUDED_TINT_LABEL,
    )


def _trajectory_band(
    trajectory: SourceTrajectory,
) -> tuple[list[float], list[float], list[float]]:
    """Applied-N levels with the lowest and highest yield recorded at each.

    For a clean trajectory the two bounds coincide and the band *is* the
    ordinary connecting line every other figure in this product draws. For an
    MV-028 trajectory they do not, and the band is the honest connector: it
    shows the whole interval the source puts at each level without choosing
    which record continues into the next one. Pairing them by order of
    appearance would draw a series the source does not record — the two rows of
    a cell are adjacent in the file, but nothing says the first row at zero N
    is the same plot as the first row at 65 kg N/ha.
    """

    by_level: dict[float, list[float]] = collections.defaultdict(list)
    for observation in trajectory.observations:
        by_level[observation.n_rate_kg_ha].append(observation.yield_t_ha)
    levels = sorted(by_level)
    return (
        levels,
        [min(by_level[level]) for level in levels],
        [max(by_level[level]) for level in levels],
    )


def _draw_connecting_paths(
    axes,
    trajectories: Sequence[SourceTrajectory],
    line_alpha: float,
    *,
    label_series: bool,
    colour: str = "grey",
    linewidth: float = 1.0,
) -> None:
    """One connector per trajectory: a line when clean, a band when duplicated."""

    labelled: set[str] = set()
    for trajectory in trajectories:
        levels, lows, highs = _trajectory_band(trajectory)
        if len(levels) < 2:
            continue
        duplicated = _duplicated_levels(trajectory)
        label = CONNECTING_BAND_LABEL if duplicated else CONNECTING_LINE_LABEL
        show = label_series and label not in labelled
        labelled.add(label)
        if duplicated:
            axes.fill_between(
                levels,
                lows,
                highs,
                color=colour,
                alpha=min(0.22, line_alpha * 0.45),
                linewidth=0.0,
                zorder=1,
                label=label if show else "_nolegend_",
            )
            # The fill alone disappears where the two records nearly agree, so
            # both bounds are stroked as well; together they read as one
            # slightly thick connector rather than as two separate series.
            for bound in (lows, highs):
                axes.plot(
                    levels,
                    bound,
                    color=colour,
                    linewidth=linewidth * 0.7,
                    alpha=line_alpha,
                    zorder=1,
                    label="_nolegend_",
                )
            continue
        axes.plot(
            levels,
            lows,
            color=colour,
            linewidth=linewidth,
            alpha=line_alpha,
            zorder=1,
            label=label if show else "_nolegend_",
        )


def _panel_title(
    year: int,
    trajectories: Sequence[SourceTrajectory],
    contexts: Mapping[str, TrajectoryContext],
    group: VarietyGroup,
) -> tuple[str, bool]:
    ladder, mixed = _distinct_ladder(trajectories)
    means = _clean_year_means(trajectories)
    lines = [str(year)]
    if group.names_line_per_panel:
        recorded = sorted(
            {
                contexts[trajectory.trajectory_id].variety.strip()
                for trajectory in trajectories
                if trajectory.trajectory_id in contexts
            }
        )
        lines[0] = f"{year} — {', '.join(recorded)}"
    lines.append(
        f"n={len(trajectories)}; applied N {ladder} kg N/ha"
        f"{' (mixed)' if mixed else ''}"
    )
    if means is None:
        reasons = _exclusion_reasons(trajectories)
        if set(reasons) == {EXCLUSION_DUPLICATED_N_LEVEL}:
            lines.append(MV_028_PANEL_NOTE)
        else:
            lines.append(
                "not summarized — "
                + "; ".join(
                    f"{EXCLUSION_SHORT.get(reason, reason.replace('_', ' '))}"
                    f"×{count}"
                    for reason, count in sorted(reasons.items())
                )
            )
        return "\n".join(lines), True
    lines.append(f"zero-N={means[0]:.2f} t/ha; response={means[1]:.2f} t/ha")
    return "\n".join(lines), False


def _panel_grid(years: Sequence[int]):
    from matplotlib import pyplot as plt

    columns = min(PANEL_COLUMNS, len(years))
    rows = math.ceil(len(years) / columns)
    # A whole season pooled is fifty planting years; at the per-variety panel
    # size that is a five-by-ten sheet over five feet tall. Shrink the panel
    # rather than the grid, so every panel keeps the shared response-curve
    # frame and only the plate gets smaller.
    scale = 1.0 if len(years) <= PANEL_SCALE_YEAR_LIMIT else PANEL_SCALE_FACTOR
    figure, axes_grid = plt.subplots(
        rows,
        columns,
        figsize=(
            PANEL_WIDTH_IN * scale * columns,
            PANEL_HEIGHT_IN * scale * rows,
        ),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_all = list(np.atleast_1d(axes_grid).ravel())
    for spare in axes_all[len(years) :]:
        spare.set_visible(False)
    return figure, axes_all[: len(years)], rows, columns


def _label_edges(axes_list, rows: int, columns: int) -> None:
    for position, axes in enumerate(axes_list):
        if position // columns == rows - 1 or position + columns >= len(axes_list):
            axes.set_xlabel("Applied N (kg N/ha)")
            # `sharex` hides tick labels on every row but the last, which leaves
            # a bottom-edge panel of a ragged grid carrying an axis title over a
            # bare axis. Put its own ticks back.
            axes.tick_params(labelbottom=True)
        if position % columns == 0:
            axes.set_ylabel("Grain yield (t/ha)")


def _write_variety_plate(
    group: VarietyGroup,
    season: str,
    overlay: SourceDatasetOverlay,
    contexts: Mapping[str, TrajectoryContext],
    year_groups: Mapping[int, tuple[str, ...]],
    limits,
    destination: Path,
) -> tuple[int, int, tuple[int, ...]]:
    from matplotlib import pyplot as plt

    years = sorted(year_groups)
    every_id = tuple(
        trajectory_id for year in years for trajectory_id in year_groups[year]
    )
    plate_overlay = subset_overlay(overlay, every_id)
    colours = _treatment_class_colours(overlay)
    style = _adaptive_style(
        plate_overlay.summary.finite_observation_count,
        plate_overlay.summary.trajectory_count,
    )

    figure, axes_list, rows, columns = _panel_grid(years)
    excluded_years: list[int] = []
    try:
        for index, (year, axes) in enumerate(zip(years, axes_list, strict=True)):
            subset = subset_overlay(overlay, year_groups[year])
            _draw_year_panel(axes, subset, colours, style, label_series=index == 0)
            limits.apply(axes)
            title, is_excluded = _panel_title(
                year, subset.trajectories, contexts, group
            )
            if is_excluded:
                excluded_years.append(year)
                axes.set_facecolor(MV_028_PANEL_FACECOLOUR)
            else:
                axes.set_facecolor(CLEAN_PANEL_FACECOLOUR)
            axes.set_title(
                title,
                fontsize=9,
                color=MV_028_TITLE_COLOUR if is_excluded else "black",
            )
        _label_edges(axes_list, rows, columns)

        handles, labels = axes_list[0].get_legend_handles_labels()
        if excluded_years:
            handles.append(_excluded_tint_handle())
            labels.append(EXCLUDED_TINT_LABEL)
        legend_strip = _legend_strip(
            figure,
            _fitted_legend(
                figure,
                handles,
                labels,
                figure.get_size_inches()[0] - 0.6,
                8,
                wrap=True,
                loc="lower center",
                frameon=False,
            ),
        )
        excluded_text = (
            "no panel here is excluded from the cluster-eligible view"
            if not excluded_years
            else (
                f"tinted panels ({_year_runs_text(excluded_years)}) are the "
                f"{len(excluded_years)} planting years this variety contributes "
                "to no cluster profile: "
                f"{_reason_sentence(_year_exclusion_reasons(overlay, year_groups, excluded_years))}"
                ", so they carry no read-out"
            )
        )
        _reserve_suptitle(
            figure,
            legend_strip=legend_strip,
            suptitle_text=_wrap_title_lines(
                [
                    f"source={SOURCE_NAME} — {season_label(season)}, "
                    f"{group.display}: every planting year on its own panel",
                    f"{len(every_id)} recorded replicate trajectories, "
                    f"{_span_text(years)} ({len(years)} of "
                    f"{years[-1] - years[0] + 1} calendar years recorded); "
                    "every panel shares the frame the rest of this product uses",
                    "each panel is a recorded stratum, not a cluster: nothing "
                    "was clustered to produce it, and membership is every "
                    "recorded trajectory rather than the cluster-eligible subset",
                    excluded_text,
                    "duplicated levels are drawn as the two observations the "
                    "source records — never averaged, and never paired into two "
                    "ladders",
                    "variety is confounded with the calendar period it was grown "
                    "in, with the applied-N ladder of that period and with the "
                    "accumulated soil history; a difference between panels is "
                    "not a genetic effect",
                    _DISCLAIMER,
                ],
                width=_plate_title_width(figure),
            ),
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)
    return len(every_id), len(years), tuple(excluded_years)


def _write_window_plate(
    groups: Sequence[VarietyGroup],
    season: str,
    overlay: SourceDatasetOverlay,
    contexts: Mapping[str, TrajectoryContext],
    limits,
    destination: Path,
) -> tuple[int, tuple[int, ...], tuple[str, ...]]:
    """The 1992-2001 stretch itself, all four groups on one sheet."""

    from matplotlib import pyplot as plt

    start, end = MV_028_WINDOW
    # Curated pools are a second view of trajectories the per-variety folders
    # already carry. On a per-folder plate that is the point; here it would draw
    # and count the same replicate twice, once per colour.
    curated_slugs = {curated.slug for curated in CURATED_GROUPS}
    drawn_groups = [group for group in groups if group.slug not in curated_slugs]
    uncovered = _uncovered_window_varieties(drawn_groups, contexts, season)
    membership: dict[int, dict[str, tuple[str, ...]]] = collections.defaultdict(dict)
    for group in drawn_groups:
        year_groups = _season_year_groups(contexts, overlay, season, group.matches)
        for year, ids in year_groups.items():
            if start <= year <= end:
                membership[year][group.slug] = ids

    years = sorted(membership)
    if not years:
        raise ValueError(
            f"No {season} trajectories from the requested varieties fall in "
            f"{start}-{end}"
        )

    every_id = tuple(
        trajectory_id
        for year in years
        for ids in membership[year].values()
        for trajectory_id in ids
    )
    plate_overlay = subset_overlay(overlay, every_id)
    marker_size, marker_alpha, _ = _adaptive_style(
        plate_overlay.summary.finite_observation_count,
        plate_overlay.summary.trajectory_count,
    )
    # Only the groups that actually carry the window are drawn or legended, so
    # the palette is built over those rather than over the whole roster: with
    # every recorded variety on the roster, `C0..C9` would run out and most
    # entries would name an empty series.
    present_slugs = {
        slug
        for year in years
        for slug in membership[year]
    }
    present_groups = [
        group for group in drawn_groups if group.slug in present_slugs
    ]
    palette = plt.get_cmap("tab10")
    group_colours = {
        group.slug: palette(index % 10) for index, group in enumerate(present_groups)
    }

    figure, axes_list, rows, columns = _panel_grid(years)
    try:
        for year, axes in zip(years, axes_list, strict=True):
            present: list[str] = []
            for group in present_groups:
                ids = membership[year].get(group.slug)
                if not ids:
                    continue
                present.append(group.display)
                subset = subset_overlay(overlay, ids)
                for marker, is_zero in (("s", True), ("o", False)):
                    points = [
                        observation
                        for trajectory in subset.trajectories
                        for observation in trajectory.observations
                        if (observation.n_rate_kg_ha == 0.0) is is_zero
                    ]
                    if not points:
                        continue
                    axes.scatter(
                        [observation.n_rate_kg_ha for observation in points],
                        [observation.yield_t_ha for observation in points],
                        s=marker_size,
                        alpha=marker_alpha,
                        color=group_colours[group.slug],
                        marker=marker,
                        zorder=3,
                    )
                _draw_connecting_paths(
                    axes,
                    subset.trajectories,
                    0.35,
                    label_series=False,
                    colour=group_colours[group.slug],
                    linewidth=0.9,
                )
            limits.apply(axes)
            axes.grid(alpha=0.2, linewidth=0.6)
            trajectories = subset_overlay(
                overlay,
                tuple(
                    trajectory_id
                    for ids in membership[year].values()
                    for trajectory_id in ids
                ),
            ).trajectories
            ladder, mixed = _distinct_ladder(trajectories)
            # Keyed on what cannot be summarized rather than on duplication
            # alone, so the tint here carries the same meaning its legend key
            # states and the meaning it has on every other plate. Inside this
            # window the two tests coincide; elsewhere they do not.
            reasons = _exclusion_reasons_in_order(trajectories)
            excluded = sum(
                1
                for trajectory in trajectories
                if isinstance(describe_trajectory(trajectory), str)
            )
            axes.set_facecolor(
                MV_028_PANEL_FACECOLOUR if excluded else CLEAN_PANEL_FACECOLOUR
            )
            axes.set_title(
                f"{year} — {len(present)} "
                f"{'variety' if len(present) == 1 else 'varieties'} recorded\n"
                f"n={len(trajectories)}; applied N {ladder} kg N/ha"
                f"{' (mixed)' if mixed else ''}\n"
                f"{excluded} of {len(trajectories)} not summarized"
                f"{f' ({_reason_tag(reasons)})' if excluded else ''}",
                fontsize=9,
                color=MV_028_TITLE_COLOUR if excluded else "black",
            )
        _label_edges(axes_list, rows, columns)

        handles = [
            plt.Line2D(
                [],
                [],
                color=group_colours[group.slug],
                marker="o",
                linestyle="-",
                linewidth=0.9,
                label=group.display,
            )
            for group in present_groups
        ] + [
            plt.Line2D(
                [], [], color="0.35", marker="s", linestyle="none", label="zero N"
            ),
            plt.Line2D(
                [],
                [],
                color="0.35",
                marker="o",
                linestyle="none",
                label="mineral N rate",
            ),
            plt.Line2D(
                [],
                [],
                color="0.45",
                linewidth=5,
                alpha=0.5,
                label=CONNECTING_BAND_LABEL,
            ),
            _excluded_tint_handle(),
        ]
        legend_strip = _legend_strip(
            figure,
            _fitted_legend(
                figure,
                handles,
                [handle.get_label() for handle in handles],
                figure.get_size_inches()[0] - 0.6,
                8,
                wrap=True,
                loc="lower center",
                frameon=False,
            ),
        )
        _reserve_suptitle(
            figure,
            legend_strip=legend_strip,
            suptitle_text=_wrap_title_lines(
                [
                    f"source={SOURCE_NAME} — {season_label(season)}, "
                    f"{start}-{end}: the stretch the cluster-eligible "
                    "planting-year view has no panel for",
                    f"{len(every_id)} recorded replicate trajectories across "
                    f"{len(years)} planting years, from "
                    + ", ".join(group.display for group in present_groups),
                    "the data are present in the source workbook and in the "
                    "curated CSV, with no blank yields; every trajectory here is "
                    "excluded from clustering because the source records two "
                    "yields at each applied-N level and nothing that "
                    "distinguishes them (MV-028)",
                    "both records of a duplicated level are drawn — never "
                    "averaged, and never paired into two ladders; a duplicated "
                    "trajectory's connector is a band spanning the range at "
                    "each level rather than a line through one path",
                    *(
                        [
                            "the groups drawn here do not exhaust the window: "
                            + "; ".join(
                                f"{year} also records "
                                + ", ".join(names)
                                for year, names in sorted(uncovered.items())
                            )
                        ]
                        if uncovered
                        else [
                            "every variety the season records inside this "
                            "window is drawn here"
                        ]
                    ),
                    "year, applied-N ladder, plot design and accumulated soil "
                    "history all change together across this stretch; a "
                    "difference between panels is not a variety effect",
                    _DISCLAIMER,
                ],
                width=_plate_title_width(figure),
            ),
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)

    absent = tuple(
        group.display for group in drawn_groups if group.slug not in present_slugs
    )
    return len(every_id), tuple(years), absent


def _span_text(years: Sequence[int]) -> str:
    """`1968`, or `1993-1998`. A one-year variety must not read `1968-1968`."""

    first, last = min(years), max(years)
    return str(first) if first == last else f"{first}-{last}"


def _year_count_text(count: int) -> str:
    return f"{count} planting {'year' if count == 1 else 'years'}"


def _year_runs_text(years: Sequence[int]) -> str:
    """Fold a sorted year list into `1992-1998, 2001` for a caption."""

    ordered = sorted(set(years))
    runs: list[tuple[int, int]] = []
    for year in ordered:
        if runs and year == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], year)
        else:
            runs.append((year, year))
    return ", ".join(
        str(start) if start == end else f"{start}-{end}" for start, end in runs
    )


# --------------------------------------------------------------------------
# Reference sheet — the parent folder's decades-and-trend idiom
# --------------------------------------------------------------------------

# `by_planting_year/figure_only/decades_and_trend_figure.jpeg` is 34in wide for
# the whole season's six decades. Reusing that inches-per-year density, rather
# than the fixed width, keeps a four-decade or one-decade variety sheet
# proportioned like the parent instead of stretching thin insets to fill a
# canvas sized for more calendar time than the variety was grown in.
_REFERENCE_YEAR_SPAN = 60
_INCHES_PER_YEAR = (
    _SHEET_WIDTH_IN - _SHEET_LEFT_IN - _SHEET_RIGHT_IN
) / _REFERENCE_YEAR_SPAN

# Two caption columns of `_CAPTION_COLUMN_CHARS` at `_CAPTION_FONT_SIZE`, set
# `_CAPTION_COLUMN_OFFSET` apart, need about this much width before the right
# column starts on top of the left one.
_MIN_PLOT_WIDTH_IN = 21.0

# Widest an inset may be, as a multiple of its own height. A decade band on a
# floored sheet is far wider than one on the parent's, and an inset stretched to
# fill it would put this product's shared response-curve frame at an aspect
# nothing else in the folder uses.
_MAX_INSET_ASPECT = 1.3

# A variety with nothing summarizable has no trend to draw. The calendar strip
# stays — it is what puts each inset over its own decade, and it is where the
# window is named — and everything the two trend panels would have occupied is
# given back rather than left blank.
_CALENDAR_STRIP_IN = 1.15
# Below the inset: its own tick labels plus the shared frame caption.
_CALENDAR_STRIP_PAD_IN = 1.15
# Above the inset: its decade title, clear of the series legend above that.
_INSET_TITLE_PAD_IN = 0.80


def _group_features(
    overlay: SourceDatasetOverlay,
    trajectory_ids: Sequence[str],
) -> dict[str, TrajectoryFeatures]:
    """Descriptive features for the trajectories that have them, and only those.

    The excluded ones are not given a stand-in. They are drawn in the insets as
    observations and are absent from every summary on the sheet, which is the
    whole distinction the sheet has to carry.
    """

    wanted = set(trajectory_ids)
    features: dict[str, TrajectoryFeatures] = {}
    for trajectory in overlay.trajectories:
        if trajectory.trajectory_id not in wanted:
            continue
        described = describe_trajectory(trajectory)
        if not isinstance(described, str):
            features[trajectory.trajectory_id] = described
    return features


def _annual_records(
    year_groups: Mapping[int, tuple[str, ...]],
    features: Mapping[str, TrajectoryFeatures],
) -> tuple[AnnualRecord, ...]:
    """One record per planting year every trajectory of which can be summarized.

    A year with any MV-028 member yields no record at all, so the trend line
    breaks there rather than being computed from whichever replicates happened
    to survive. `AnnualRecord` is the real dataclass, so `annual_ladder_eras`
    and the era boundaries below are reused unchanged.
    """

    records = []
    for year in sorted(year_groups):
        ids = year_groups[year]
        if not ids or any(trajectory_id not in features for trajectory_id in ids):
            continue
        matrix = np.asarray(
            [
                [getattr(features[tid], name) for name in RESPONSE_TYPE_FEATURES]
                for tid in ids
            ],
            dtype=float,
        )
        means = matrix.mean(axis=0)
        errors = (
            matrix.std(axis=0, ddof=1) / math.sqrt(matrix.shape[0])
            if matrix.shape[0] > 1
            else np.zeros(matrix.shape[1])
        )
        ladders = collections.Counter(ladder_text(features[tid].ladder) for tid in ids)
        records.append(
            AnnualRecord(
                year=year,
                decade=decade_band(str(year)),
                trajectory_count=len(ids),
                context_count=1,
                ladder=ladders.most_common(1)[0][0],
                ladder_is_mixed=len(ladders) > 1,
                means=dict(zip(RESPONSE_TYPE_FEATURES, means, strict=True)),
                standard_errors=dict(
                    zip(RESPONSE_TYPE_FEATURES, errors, strict=True)
                ),
            )
        )
    return tuple(records)


def _decade_groups(
    year_groups: Mapping[int, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for year, ids in year_groups.items():
        band = decade_band(str(year))
        if band:
            grouped[band].extend(ids)
    return {decade: tuple(sorted(ids)) for decade, ids in sorted(grouped.items())}


def _inset_readout(
    ids: Sequence[str],
    trajectories: Sequence[SourceTrajectory],
    features: Mapping[str, TrajectoryFeatures],
    contexts: Mapping[str, TrajectoryContext],
    total: int,
    *,
    compact: bool,
) -> str:
    summarizable = [trajectory_id for trajectory_id in ids if trajectory_id in features]
    years = cluster_year_range(ids, contexts)
    share = 100 * len(ids) / total if total else 0.0
    single_year = bool(years) and years[0] == years[1]
    long_years = (
        str(years[0]) if single_year else f"{years[0]}-{years[1]}" if years else "years unknown"
    )
    short_years = (
        str(years[0])
        if single_year
        else f"{years[0]}-{str(years[1])[2:]}" if years else "years unknown"
    )
    if not summarizable:
        ladder, _ = _distinct_ladder(trajectories)
        reasons = _exclusion_reasons_in_order(trajectories)
        if compact:
            return "\n".join(
                [
                    f"n={len(ids)} ({share:.0f}%)",
                    short_years,
                    _reason_tag(reasons),
                    "not summarized",
                ]
            )
        return "\n".join(
            [
                f"{len(ids)} trajectories ({share:.0f}%)",
                long_years,
                f"applied N {ladder}",
                "none summarized —",
                f"every one: {_reason_tag(reasons)}",
            ]
        )
    centroid = centroid_summary(summarizable, features)
    ladder_shares = collections.Counter(
        ladder_text(features[tid].ladder) for tid in summarizable
    )
    shares = {
        ladder: count / len(summarizable) for ladder, count in ladder_shares.most_common()
    }
    excluded = len(ids) - len(summarizable)
    excluded_label = _reason_tag(
        _exclusion_reasons_in_order(
            trajectory for trajectory in trajectories if trajectory.trajectory_id not in features
        )
    )
    if compact:
        return "\n".join(
            [
                f"n={len(ids)} ({share:.0f}%)",
                *([f"{excluded} not ({excluded_label})"] if excluded else []),
                short_years,
                f"0N {centroid['yield_at_zero_n_t_ha']:.2f}",
                f"+N {centroid['response_above_zero_n_t_ha']:.2f}",
            ]
        )
    return "\n".join(
        [
            f"{len(ids)} trajectories ({share:.0f}%)"
            + (" drawn" if excluded else ""),
            *(
                [f"{len(summarizable)} summarized, {excluded} not ({excluded_label})"]
                if excluded
                else []
            ),
            long_years,
            f"applied N {format_shares(shares, limit=1)}"
            + (" (of the summarized)" if excluded else ""),
            f"mean zero-N {centroid['yield_at_zero_n_t_ha']:.2f} t/ha "
            + ("(of the summarized)" if excluded else ""),
            f"mean response {centroid['response_above_zero_n_t_ha']:.2f} t/ha"
            + (" (of the summarized)" if excluded else ""),
        ]
    )


EXCLUSION_PHRASES = {
    EXCLUSION_DUPLICATED_N_LEVEL: (
        "the source records two yields at each applied-N level with nothing to "
        "tell them apart (MV-028)"
    ),
    EXCLUSION_NO_ZERO_N_ANCHOR: (
        "a replicate has no recorded yield at zero N, which every summary on "
        "this sheet is measured from"
    ),
    EXCLUSION_INCOMPLETE_LADDER: (
        "a replicate records fewer than three distinct applied-N levels"
    ),
}

EXCLUSION_COMPACT = {
    EXCLUSION_DUPLICATED_N_LEVEL: "MV-028",
    EXCLUSION_NO_ZERO_N_ANCHOR: "no zero-N anchor",
    EXCLUSION_INCOMPLETE_LADDER: "short ladder",
}

EXCLUSION_SHORT = {
    EXCLUSION_DUPLICATED_N_LEVEL: "duplicated N levels (MV-028)",
    EXCLUSION_NO_ZERO_N_ANCHOR: "no zero-N anchor",
    EXCLUSION_INCOMPLETE_LADDER: "incomplete ladder",
}


def _year_exclusion_reasons(
    overlay: SourceDatasetOverlay,
    year_groups: Mapping[int, tuple[str, ...]],
    years: Sequence[int],
) -> tuple[str, ...]:
    """Distinct reasons the given planting years cannot be summarized.

    Not assumed to be MV-028. The window is, but a variety's own record can
    also stop at a replicate with no zero-N yield -- IR58's 1986 does -- and a
    figure that called that MV-028 would be naming the wrong defect.
    """

    by_id = {
        trajectory.trajectory_id: trajectory for trajectory in overlay.trajectories
    }
    reasons: list[str] = []
    for year in years:
        for trajectory_id in year_groups.get(year, ()):
            trajectory = by_id.get(trajectory_id)
            if trajectory is None:
                continue
            described = describe_trajectory(trajectory)
            if isinstance(described, str) and described not in reasons:
                reasons.append(described)
    return tuple(reasons)


def _exclusion_reasons_in_order(trajectories) -> tuple[str, ...]:
    ordered: list[str] = []
    for trajectory in trajectories:
        described = describe_trajectory(trajectory)
        if isinstance(described, str) and described not in ordered:
            ordered.append(described)
    return tuple(ordered)


def _reason_sentence(reasons: Sequence[str]) -> str:
    return "; ".join(
        EXCLUSION_PHRASES.get(reason, reason.replace("_", " ")) for reason in reasons
    ) or "they cannot be summarized"


def _reason_label(reasons: Sequence[str]) -> str:
    return ", ".join(
        EXCLUSION_SHORT.get(reason, reason.replace("_", " ")) for reason in reasons
    ) or "not summarized"


def _reason_tag(reasons: Sequence[str]) -> str:
    return ", ".join(
        EXCLUSION_COMPACT.get(reason, reason.replace("_", " ")) for reason in reasons
    ) or "not summarized"


def _write_reference_sheet(
    group: VarietyGroup,
    season: str,
    overlay: SourceDatasetOverlay,
    contexts: Mapping[str, TrajectoryContext],
    year_groups: Mapping[int, tuple[str, ...]],
    limits,
    destination: Path,
) -> tuple[int, tuple[int, ...]]:
    """This variety's decade insets drawn on the stretch of its own yield trend.

    The variety-scoped analogue of
    `../decades_and_trend_matched_colours_figure.jpeg`, on the same geometry
    block (`reference_sheet_layout`) and with the same pairing: each decade's
    response-curve cloud sits on the exact calendar stretch of the per-year
    trend it came from, both trend panels share one inches-per-t/ha, and colour
    carries the quantity while the applied-N era carries marker shape.

    Two things differ, and they are the point of the sheet. The insets are
    drawn from *every* recorded trajectory rather than the cluster-eligible
    ones, so the MV-028 window appears as a cloud instead of as a hole; and the
    trend panels, which cannot be computed for a duplicated year, break across
    that window and name it. The parent sheet labels the same stretch
    `1992-2001 no DS record`, which is not what the source says.
    """

    from matplotlib import pyplot as plt

    all_ids = tuple(
        trajectory_id for year in sorted(year_groups) for trajectory_id in year_groups[year]
    )
    features = _group_features(overlay, all_ids)
    decade_groups = _decade_groups(year_groups)
    decades = sorted(decade_groups)
    spans = _decade_spans(decades)
    if not decades or len(spans) != len(decades):
        return 0, ()

    records = _annual_records(year_groups, features)
    total = len(all_ids)
    recorded_years = sorted(year_groups)
    summarized = {record.year for record in records}
    unsummarized = tuple(year for year in recorded_years if year not in summarized)
    # Each blank stretch is labelled with *its own* reason. A sheet whose 1986
    # is a missing zero-N anchor and whose 1992-2001 is MV-028 must not put both
    # names on both gaps.
    gap_runs = [
        (run, _year_exclusion_reasons(overlay, year_groups, run))
        for run in _contiguous_runs(unsummarized, year=lambda year: year, presorted=True)
    ]

    x_lo = spans[0][1] - 0.5
    x_hi = spans[-1][2] + 0.5
    visible = [(first - 0.5, last + 0.5) for _, first, last in spans]

    colours = _treatment_class_colours(overlay)
    zero_n_colour = colours.get(_ZERO_N_TREATMENT_CLASS, "C1")
    mineral_colour = next(
        (colour for label, colour in colours.items() if label != _ZERO_N_TREATMENT_CLASS),
        "C0",
    )

    headline = (
        f"source={SOURCE_NAME} — {season_label(season)}, {group.display}: each "
        "planting decade's response curves drawn on the stretch of the yield "
        "trend they came from"
    )
    unsummarized_text = (
        f"{_year_runs_text(unsummarized)} — {len(unsummarized)} of "
        f"{len(recorded_years)} recorded planting "
        f"{'year' if len(unsummarized) == 1 else 'years'}"
    )
    disclosures = [
        "colour carries the quantity here, not the applied-N era: orange is "
        "yield at zero N — the orange points in every inset, and the panel "
        "that tracks their mean by year — and blue is the response above zero "
        "N, computed from the blue mineral-N points of the same inset relative "
        "to its orange ones rather than read off either directly",
        f"{total} recorded {group.display} trajectories over "
        f"{_year_count_text(len(recorded_years))} "
        f"({_span_text(recorded_years)}); membership is every "
        "recorded replicate trajectory, not the cluster-eligible subset the "
        "parent folder's sheets are built from",
        "the insets — every trajectory of the decade against applied N, each a "
        "recorded stratum that nothing was clustered to produce, and each on "
        "the same frame at the same size, so any two can be compared directly",
        "each inset sits on its own calendar decade of the axis beneath it, "
        "whole decade to whole decade",
        "the two trend panels — yield at zero N by planting year, which is the "
        "left-hand end of every curve in the inset above it, and the response "
        "above zero N, which is the other end. Both are the mean of that "
        "year's replicate trajectories, the bar its standard error across "
        "replicates, and both are drawn to one shared scale so a change of a "
        "given size has the same slope in each",
        *(
            [
                "the trend panels are blank across "
                f"{unsummarized_text}. "
                + "; ".join(
                    f"{_year_runs_text(run)}: {_reason_sentence(run_reasons)}"
                    for run, run_reasons in gap_runs
                )
                + ". Neither a zero-N mean nor a response is defined for those "
                "years. The observations themselves are present, and are drawn "
                "in the inset above",
            ]
            if unsummarized
            else []
        ),
        "a connector is a line where the trajectory records one yield per "
        "applied-N level and a band where it records two: the band spans the "
        "range at each level and chooses no path through them, because the "
        "source does not distinguish the records",
        "each line is broken at every applied-N era boundary (dashed) and at "
        "any unrecorded year: a change across either is a change of experiment "
        "or a missing observation, not a response to fertilizer",
        "variety is confounded with the calendar period it was grown in, with "
        "the applied-N ladder of that period and with the accumulated soil "
        "history; a difference between decades here is not a genetic effect",
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
    # A one-decade variety at the parent's year density would be drawn on a
    # canvas narrower than its own two-column caption block, which `figure.text`
    # runs off the edge rather than wrapping. Floor the plot width instead; the
    # year axis stretches to fill it and the insets are capped below, so the
    # sheet keeps the parent's proportions rather than the parent's density.
    plot_width = max(_INCHES_PER_YEAR * year_span, _MIN_PLOT_WIDTH_IN)
    sheet_width = plot_width + _SHEET_LEFT_IN + _SHEET_RIGHT_IN
    plot_left = _SHEET_LEFT_IN

    has_trend = bool(records)
    inset_height_in = (_INSET_TOP_FRACTION - _INSET_BOTTOM_FRACTION) * _HOST_ROW_IN
    if has_trend:
        host_row_in = _HOST_ROW_IN
        lower_row_in = _TREND_ROW_IN
        inset_offset_in = _INSET_BOTTOM_FRACTION * _HOST_ROW_IN
    else:
        host_row_in = (
            _CALENDAR_STRIP_PAD_IN + inset_height_in + _INSET_TITLE_PAD_IN
        )
        lower_row_in = _CALENDAR_STRIP_IN
        inset_offset_in = _CALENDAR_STRIP_PAD_IN

    era_legend_in = _TREND_LEGEND_IN if has_trend else 0.0
    trend_gap_in = _TREND_GAP_IN if has_trend else 0.35
    # Clearance between the inset's own tick labels and the shared-frame caption
    # under them. Sized from the tick type rather than chosen, so a narrower
    # sheet cannot push the caption up into the labels.
    caption_offset_in = 0.30 + _INSET_TICK_FONT_SIZE / 72.0
    height = (
        _SHEET_BOTTOM_IN
        + era_legend_in
        + _TREND_XAXIS_IN
        + lower_row_in
        + trend_gap_in
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

    def _year_to_inches(year: float) -> float:
        return plot_left + plot_width * (year - x_lo) / year_span

    lower_bottom = _SHEET_BOTTOM_IN + era_legend_in + _TREND_XAXIS_IN
    host_bottom = lower_bottom + lower_row_in + trend_gap_in
    inset_legend_bottom = host_bottom + host_row_in

    eras = annual_ladder_eras(records)

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
                lower_bottom,
                lower_row_in,
                mineral_colour,
            ),
        ) if has_trend else (
            ("", "", lower_bottom, lower_row_in, "0.4"),
        )
        low = high = None
        for record in records:
            for feature, *_unused in trend_panels:
                value = record.means[feature]
                error = record.standard_errors[feature]
                error = error if error == error else 0.0
                low = value - error if low is None else min(low, value - error)
                high = value + error if high is None else max(high, value + error)
        if low is None or high is None:
            low, high = 0.0, 1.0
        elif not high > low:
            # One summarizable year, whose standard error across replicates can
            # be zero. Pad around the value: resetting to 0-1 would put the only
            # point on the sheet off its own frame.
            span = max(abs(low) * 0.1, 0.5)
            low, high = low - span, high + span
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
            is_host = has_trend and not trend_axes
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
                marker = _ERA_MARKERS[era_index % len(_ERA_MARKERS)]
                for run_index, run in enumerate(
                    _contiguous_runs(span, year=lambda record: record.year)
                ):
                    axes.errorbar(
                        [record.year for record in run],
                        [record.means[feature] for record in run],
                        yerr=[record.standard_errors[feature] for record in run],
                        marker=marker,
                        markersize=7,
                        linewidth=1.8,
                        capsize=3.0,
                        color=panel_colour,
                        ecolor=panel_colour,
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

            # Name the stretch the trend cannot cross, on the axis where the
            # line stops. The parent sheet calls this same stretch "no DS
            # record"; the record exists and is drawn in the inset above.
            blended = transforms.blended_transform_factory(
                axes.transData, axes.transAxes
            )
            for run, run_reasons in gap_runs:
                axes.axvspan(
                    run[0] - 0.5,
                    run[-1] + 0.5,
                    color=MV_028_PANEL_FACECOLOUR,
                    zorder=0,
                    linewidth=0,
                )
                axes.text(
                    (run[0] + run[-1]) / 2.0,
                    # The host panel's y range is stretched so the trend
                    # occupies its lower fraction and the insets float over the
                    # rest; a label at mid-axes would land behind them.
                    _HOST_TREND_FRACTION / 2 if is_host else 0.5,
                    f"{_year_runs_text(run)}\n{_reason_label(run_reasons)}\n"
                    "drawn in the insets, not summarized",
                    transform=blended,
                    ha="center",
                    va="center",
                    fontsize=_INSET_READOUT_FONT_SIZE,
                    color=MV_028_TITLE_COLOUR,
                    linespacing=1.4,
                    zorder=4,
                )

            axes.tick_params(labelsize=_TICK_FONT_SIZE)
            if not has_trend:
                # The calendar strip carries the decade bands and the window
                # label and nothing else: there is no quantity to put on a y
                # axis, and a 0.00-1.00 scale beside a blank panel would read
                # as measured zeroes.
                axes.set_yticks([])
                continue
            axes.set_ylabel(
                label,
                y=_HOST_TREND_FRACTION / 2 if is_host else 0.5,
                fontsize=_AXIS_LABEL_FONT_SIZE,
            )
            axes.grid(axis="y", alpha=0.25, linewidth=0.6)
            axes.set_yticks(shared_ticks)
            axes.set_ylim(
                floor,
                floor + (ceiling - floor) / _HOST_TREND_FRACTION if is_host else ceiling,
            )

        trend_axes[0].set_xlim(x_lo, x_hi)
        if has_trend:
            trend_axes[0].tick_params(labelbottom=False)
        trend_axes[-1].set_xticks([start for _, start, _ in spans])
        trend_axes[-1].set_xticks(
            list(range(spans[0][1], spans[-1][2] + 1, 2)), minor=True
        )
        trend_axes[-1].set_xlabel("Planting year", fontsize=_AXIS_LABEL_FONT_SIZE)

        inset_axes: list[Any] = []
        tinted_inset = False
        for index, (decade, (band_lo, band_hi)) in enumerate(
            zip(decades, visible, strict=True)
        ):
            ids = decade_groups[decade]
            band_left = _year_to_inches(band_lo)
            band_right = _year_to_inches(band_hi)
            panel_inches = max(
                min(
                    band_right - band_left - _PANEL_GUTTER_IN,
                    _MAX_INSET_ASPECT * inset_height_in,
                ),
                0.4,
            )
            axes = figure.add_axes(
                _box(
                    (band_left + band_right - panel_inches) / 2.0,
                    host_bottom + inset_offset_in,
                    panel_inches,
                    inset_height_in,
                ),
                sharex=inset_axes[0] if inset_axes else None,
                sharey=inset_axes[0] if inset_axes else None,
            )
            inset_axes.append(axes)
            subset = subset_overlay(overlay, ids)
            excluded_count = sum(
                1 for trajectory_id in ids if trajectory_id not in features
            )
            excluded = excluded_count > 0
            tinted_inset = tinted_inset or excluded
            wholly_excluded = excluded_count == len(ids)
            axes.set_facecolor(
                MV_028_PANEL_FACECOLOUR if excluded else CLEAN_PANEL_FACECOLOUR
            )
            axes.set_zorder(trend_axes[0].get_zorder() + 1)
            for spine in axes.spines.values():
                spine.set_color("0.55")

            _draw_year_panel(
                axes,
                subset,
                colours,
                _adaptive_style(
                    subset.summary.finite_observation_count,
                    subset.summary.trajectory_count,
                ),
                # Every inset labels its own series; the legend below dedupes
                # them. Labelling only the first would drop the band entry
                # whenever the oldest decade happens to be a clean one.
                label_series=True,
            )
            limits.apply(axes)
            axes.grid(alpha=0.25, linewidth=0.6)
            axes.set_xticks([0, 50, 100, 150, 200])
            axes.tick_params(labelsize=_INSET_TICK_FONT_SIZE)
            axes.set_title(
                decade,
                fontsize=_INSET_TITLE_FONT_SIZE,
                pad=6,
                color=MV_028_TITLE_COLOUR if wholly_excluded else "black",
            )
            axes.text(
                0.97,
                0.03,
                _inset_readout(
                    ids,
                    subset.trajectories,
                    features,
                    contexts,
                    total,
                    compact=panel_inches < _NARROW_PANEL_IN,
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
            (host_bottom + inset_offset_in - caption_offset_in) / height,
            "insets: applied N (kg N/ha) across, grain yield (t/ha) up — one "
            "shared frame, ticks on the left-hand inset",
            ha="center",
            va="top",
            fontsize=_LEGEND_FONT_SIZE,
            color="0.35",
        )

        handles, labels = [], []
        for axes in inset_axes:
            for handle, label in zip(*axes.get_legend_handles_labels(), strict=True):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        # The tint reaches this sheet twice — the span the trend cannot cross,
        # and an inset whose decade carries an excluded trajectory — and reads
        # the same in both, so it takes one key.
        if gap_runs or tinted_inset:
            handles.append(_excluded_tint_handle())
            labels.append(EXCLUDED_TINT_LABEL)
        if handles:
            _fitted_legend(
                figure,
                handles,
                labels,
                plot_width,
                _LEGEND_FONT_SIZE,
                loc="center",
                bbox_to_anchor=_box(
                    plot_left, inset_legend_bottom, plot_width, _INSET_LEGEND_IN
                ),
                bbox_transform=figure.transFigure,
                frameon=False,
            )

        trend_handles, trend_labels = trend_axes[0].get_legend_handles_labels()
        if trend_handles:
            _fitted_legend(
                figure,
                trend_handles,
                trend_labels,
                plot_width,
                _LEGEND_FONT_SIZE,
                loc="center",
                bbox_to_anchor=_box(
                    plot_left, _SHEET_BOTTOM_IN, plot_width, _TREND_LEGEND_IN
                ),
                bbox_transform=figure.transFigure,
                frameon=False,
                title=(
                    "applied-N era — marker shape; colour on this sheet is "
                    "the quantity, not the era"
                ),
                title_fontsize=_LEGEND_FONT_SIZE,
            )

        headline_top = 1.0 - 0.20 / height
        figure.text(
            0.5,
            headline_top,
            # `figure.text` clips rather than wraps, and a shorter record gets a
            # narrower sheet while the headline keeps its length. Fold it to
            # what this sheet can actually hold.
            textwrap.fill(
                headline,
                width=max(int(72.0 * sheet_width / (0.54 * _HEADLINE_FONT_SIZE)), 40),
                break_long_words=False,
            ),
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
    return len(records), unsummarized


# --------------------------------------------------------------------------
# Folder note
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupSummary:
    """What one variety group produced, for the folder note and the console."""

    group: VarietyGroup
    trajectory_count: int
    year_count: int
    excluded_years: tuple[int, ...]
    grid_filename: str
    sheet_filename: str
    summarized_year_count: int
    unsummarized_years: tuple[int, ...]
    first_year: int
    last_year: int
    exclusion_reasons: tuple[str, ...]


@dataclass(frozen=True)
class WindowTally:
    """What the MV-028 window actually holds, recounted every run.

    The folder note asserts both that the data are present and that the
    exclusion is total. Neither claim may be written into the template: the
    curated CSV is the authority, and a stale literal here would be a note that
    contradicts the plates beside it.
    """

    finite_observations: int
    nonfinite_observations: int
    season_trajectories: int
    duplicated_trajectories: int


def _window_exclusion_tally(
    overlay: SourceDatasetOverlay,
    contexts: Mapping[str, TrajectoryContext],
    season: str,
) -> WindowTally:
    start, end = MV_028_WINDOW
    finite = 0
    nonfinite = 0
    total = 0
    duplicated = 0
    by_id = {
        trajectory.trajectory_id: trajectory for trajectory in overlay.trajectories
    }
    for trajectory_id, context in contexts.items():
        if not start <= int(context.year) <= end:
            continue
        trajectory = by_id.get(trajectory_id)
        in_season = context.season.strip().upper() == season
        if in_season:
            total += 1
        if trajectory is None:
            # Every observation the source recorded for this trajectory is
            # non-finite, so the overlay dropped it whole.
            continue
        finite += len(trajectory.observations)
        nonfinite += trajectory.excluded_nonfinite_observation_count
        if in_season and _duplicated_levels(trajectory):
            duplicated += 1
    return WindowTally(finite, nonfinite, total, duplicated)


def _uncovered_window_varieties(
    groups: Sequence[VarietyGroup],
    contexts: Mapping[str, TrajectoryContext],
    season: str,
) -> dict[int, tuple[str, ...]]:
    """Varieties recorded inside the MV-028 window that no plate here draws.

    Computed from the source rather than written into the note, so the note
    cannot claim a coverage the plates do not have if the curated CSV changes.
    """

    start, end = MV_028_WINDOW
    by_year: dict[int, set[str]] = collections.defaultdict(set)
    for context in contexts.values():
        if context.season.strip().upper() != season:
            continue
        year = int(context.year)
        if not start <= year <= end:
            continue
        variety = context.variety.strip()
        if any(group.matches(variety) for group in groups):
            continue
        by_year[year].add(variety)
    return {year: tuple(sorted(names)) for year, names in sorted(by_year.items())}


def _write_readme(
    season: str,
    summaries: Sequence[GroupSummary],
    window: tuple[int, tuple[int, ...], tuple[str, ...], str] | None,
    uncovered: Mapping[int, tuple[str, ...]],
    window_tally: WindowTally,
    destination: Path,
) -> None:
    start, end = MV_028_WINDOW
    duplication_share = (
        f"every one of its {window_tally.season_trajectories} {season} "
        "trajectories records"
        if window_tally.duplicated_trajectories == window_tally.season_trajectories
        else (
            f"{window_tally.duplicated_trajectories} of its "
            f"{window_tally.season_trajectories} {season} trajectories record"
        )
    )
    lines = [
        f"# {season_label(season)} — planting-year plates, one folder per variety",
        "",
        f"`../` has no {start}-{end} in it. These plates do, drawn from every "
        "recorded trajectory rather than the cluster-eligible ones.",
        "",
        "## Why the parent folder stops at 1991 and restarts at 2002",
        "",
        "Not missing data. `LTCCE_GY_processed.xlsx` and the curated "
        f"`LTCCE_GY_processed.csv` both carry {start}-{end} in full: "
        f"{window_tally.finite_observations:,} finite observations across the "
        f"three seasons, against {window_tally.nonfinite_observations} recorded "
        "`NA`. What the window carries instead is a *duplication*: "
        f"{duplication_share} more than one yield at some applied-N level, "
        "with no column in the source distinguishing the records. "
        "`response_curve_clusters.describe_trajectory` fail-closes on that as "
        "`duplicated_n_level_mv_028` — duplicated levels are never averaged — "
        "so the window contributes to no cluster profile, and the parent "
        "folder's decade table runs `1990s = 1990-1991` straight into "
        "`2000s = 2002-2009`.",
        "",
        "## What is in this folder",
        "",
    ]
    lines.append(
        f"One folder per recorded {season} variety, each carrying the same two "
        "plates — the decades-and-trend sheet the parent folder leads with, and "
        "the panel-per-year grid that carries the detail the sheet compresses. "
        "One cross-variety plate of the window sits at the root beside this "
        "note."
    )
    lines.append("")
    lines.append(
        "`Trend` counts the planting years that reach the sheet's trend panels; "
        "the rest are drawn in the insets but cannot be summarized."
    )
    lines.append("")
    lines.append(
        "| Folder | Recorded variety | Planting years | Trajectories | Trend | "
        "Years not summarized | Why |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | --- | --- |")
    for summary in summaries:
        span = _span_text((summary.first_year, summary.last_year))
        lines.append(
            f"| `{summary.group.slug}/` | {summary.group.display} | "
            f"{span} ({summary.year_count}) | {summary.trajectory_count} | "
            f"{summary.summarized_year_count}/{summary.year_count} | "
            f"{_year_runs_text(summary.excluded_years) or '—'} | "
            f"{_reason_label(summary.exclusion_reasons) if summary.exclusion_reasons else '—'} |"
        )
    lines.append("")
    curated_slugs = {curated.slug for curated in CURATED_GROUPS}
    pooled = [summary for summary in summaries if summary.group.slug in curated_slugs]
    if pooled:
        lines.append(
            "The pooled "
            + ", ".join(f"`{summary.group.slug}/`" for summary in pooled)
            + " folder"
            + ("s sit" if len(pooled) > 1 else " sits")
            + " *beside* the per-variety folders rather than replacing them: "
            "the lines it pools never share a planting year, so pooling them "
            "reads as one series, and each line also keeps its own folder."
        )
        lines.append("")
    if window is not None:
        window_count, window_years, absent, filename = window
        lines.append(f"### `{filename}` — the window itself")
        lines.append("")
        lines.append(
            f"Every variety that carries any of {start}-{end} on one sheet: "
            f"{window_count} recorded trajectories over {len(window_years)} "
            f"planting years. The other {len(absent)} folders record none of it."
        )
        lines.append("")
    lines += [
        "",
        "## How a duplicated level is drawn",
        "",
        "As the two observations the source records. Never averaged into one "
        "point, and never paired off into two ladders — the source carries no "
        "column that would license either, and inventing one would assert "
        "exactly the structure MV-028 says is absent. A trajectory with a "
        "duplicated level is therefore connected by a *band* rather than a "
        "line — the band spans the range recorded at each applied-N level and "
        "chooses no path through them, where a clean trajectory's band "
        "collapses to the ordinary line every other figure in this product "
        "draws. Its panel still carries no zero-N/response read-out, because "
        "neither summary is defined when there are two yields at zero N.",
        "",
        "On a panel grid, where a panel is one planting year, tint and a "
        "coloured title both mean the same thing: every trajectory in that year "
        "is excluded. On a reference sheet, where an inset is a whole decade, "
        "tint means the decade contains *some* excluded trajectories and a "
        "coloured title means *all* of them are; the inset's own read-out gives "
        "the split. Read a tinted panel as observations only.",
        "",
        "## Coverage",
        "",
    ]
    if uncovered:
        lines.append(
            f"Inside {start}-{end} these groups cover every recorded "
            f"{season} variety except the following, which are not drawn here:"
        )
        lines.append("")
        for year in sorted(uncovered):
            lines.append(f"- {year}: {', '.join(uncovered[year])}")
        lines.append("")
    else:
        lines.append(
            f"These groups cover every recorded {season} variety in "
            f"{start}-{end}."
        )
        lines.append("")
    lines += [
        "## Why this is a subdirectory",
        "",
        "`generate_response_curve_season_clusters` replaces `by_season/` as one "
        "snapshot. Its `_carry_unmanaged_directories` copies whole directories "
        "it did not build across that swap, but deliberately lets loose *files* "
        "inside a directory it did build be destroyed. Plates written flat into "
        "`../` would not survive the next regeneration; this folder does, the "
        "same way `../by_replicate/` does.",
        "",
        "## Interpretation boundary",
        "",
        f"{_DISCLAIMER}. Every point is an observed yield at an observed "
        "applied-N rate. Nothing here is clustered, and nothing here enters the "
        "governed release inventory.",
        "",
        "## Regenerating",
        "",
        "```bash",
        "conda run -n n_response python \\",
        "  modules/n_response_curve/reporting/"
        "generate_ltcce_ds_variety_planting_year_views.py \\",
        f"  --config scriptCONFIG.toml --season {season}",
        "```",
        "",
        "Additive: the recipe writes its own files and removes nothing.",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def _write_combined_note(
    season: str,
    trajectory_count: int,
    year_count: int,
    summarized_year_count: int,
    unsummarized_years: Sequence[int],
    unsummarized_reasons: Sequence[str],
    excluded_years: Sequence[int],
    variety_count: int,
    grid_filename: str,
    sheet_filename: str,
    destination: Path,
) -> None:
    start, end = MV_028_WINDOW
    lines = [
        f"# {season_label(season)} — every recorded variety pooled",
        "",
        "The same two plates `../by_variety/` gives each variety, with all "
        f"{variety_count} recorded {season} varieties in one membership.",
        "",
        "## What this is, and how it differs from `../`",
        "",
        "`../decades_designs_and_trend_figure.jpeg` is the same season on the "
        "same geometry, built from the **cluster-eligible** trajectories. This "
        "one is built from **every recorded** trajectory. That is the whole "
        f"difference, and it is why {start}-{end} is a cloud in the insets here "
        "and a panel labelled `no DS record` there — the record exists; it is "
        "excluded, which is not the same thing.",
        "",
        "## What is here",
        "",
        f"- `{sheet_filename}` — the decades-and-trend sheet: one inset per "
        "planting decade over the stretch of the per-year trend it came from. "
        f"{trajectory_count} recorded trajectories over "
        f"{_year_count_text(year_count)}; {summarized_year_count} of them reach "
        "the trend panels.",
        f"- `{grid_filename}` — the panel grid: one panel per planting year, "
        "every recorded trajectory of that year on the shared frame.",
        "",
    ]
    if unsummarized_years:
        lines += [
            "## The years the trend panels cannot carry",
            "",
            f"{_year_runs_text(unsummarized_years)} — "
            f"{_reason_sentence(unsummarized_reasons)}. A pooled year is "
            "summarized only when every one of its recorded trajectories can "
            "be, so one unusable replicate withholds the whole planting year "
            "from the trend rather than quietly shrinking its denominator. The "
            "observations are drawn in the inset above either way.",
            "",
        ]
    if excluded_years:
        lines += [
            "Panels tinted on the grid: "
            f"{_year_runs_text(excluded_years)}.",
            "",
        ]
    lines += [
        "## Interpretation boundary",
        "",
        f"{_DISCLAIMER}. Pooling every variety pools every applied-N ladder, "
        "plot design and era with them: a difference between decades here is a "
        "difference between experiments. Every point is an observed yield at "
        "an observed applied-N rate; nothing here is clustered, and nothing "
        "here enters the governed release inventory.",
        "",
        "## Regenerating",
        "",
        "Written by the same recipe as `../by_variety/`:",
        "",
        "```bash",
        "conda run -n n_response python \\",
        "  modules/n_response_curve/reporting/"
        "generate_ltcce_ds_variety_planting_year_views.py \\",
        f"  --config scriptCONFIG.toml --season {season}",
        "```",
        "",
        "`--no-combined` skips these two plates; `--combined-output-dir` moves "
        "them.",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    season = args.season
    source_path, encoding = load_source_spec(args.config, SOURCE_NAME)
    overlay = read_source_dataset_overlay(source_path, SOURCE_NAME, encoding=encoding)
    contexts = read_ltcce_contexts(source_path, encoding=encoding)

    # From the whole source overlay, before any subsetting, so these plates sit
    # on the same frame as every other figure in the product.
    limits = _shared_axis_limits(overlay)

    destination = args.output_dir.resolve()
    if destination.exists() and (not destination.is_dir() or destination.is_symlink()):
        raise RuntimeError("Variety-plate destination is not a plain directory")
    destination.mkdir(parents=True, exist_ok=True)

    groups = discover_variety_groups(contexts, overlay, season)
    print(
        f"{SOURCE_NAME} {season}: {len(groups)} variety groups "
        f"({len(groups) - len(CURATED_GROUPS)} recorded varieties plus "
        f"{len(CURATED_GROUPS)} curated pool)"
    )

    summaries: list[GroupSummary] = []
    for group in groups:
        year_groups = _season_year_groups(contexts, overlay, season, group.matches)
        if not year_groups:
            raise SystemExit(
                f"No {season} trajectories recorded for {group.display}; "
                "nothing to draw"
            )
        # One folder per variety group. The plates keep the slug in their own
        # names as well: a plate pulled into a document travels without its
        # folder, and `by_planting_year_figure.jpeg` would then name nothing.
        group_directory = destination / group.slug
        group_directory.mkdir(exist_ok=True)
        grid_filename = f"{group.slug}_by_planting_year{FIGURE_SUFFIX}"
        trajectory_count, year_count, excluded_years = _write_variety_plate(
            group,
            season,
            overlay,
            contexts,
            year_groups,
            limits,
            group_directory / grid_filename,
        )
        sheet_filename = f"{group.slug}_decades_and_trend{FIGURE_SUFFIX}"
        summarized_years, unsummarized_years = _write_reference_sheet(
            group,
            season,
            overlay,
            contexts,
            year_groups,
            limits,
            group_directory / sheet_filename,
        )
        summaries.append(
            GroupSummary(
                group=group,
                trajectory_count=trajectory_count,
                year_count=year_count,
                excluded_years=excluded_years,
                grid_filename=f"{group.slug}/{grid_filename}",
                sheet_filename=f"{group.slug}/{sheet_filename}",
                summarized_year_count=summarized_years,
                unsummarized_years=unsummarized_years,
                first_year=min(year_groups),
                last_year=max(year_groups),
                exclusion_reasons=_year_exclusion_reasons(
                    overlay, year_groups, excluded_years
                ),
            )
        )

    if not args.no_combined:
        combined_destination = args.combined_output_dir.resolve()
        if combined_destination.exists() and (
            not combined_destination.is_dir() or combined_destination.is_symlink()
        ):
            raise RuntimeError("Combined destination is not a plain directory")
        combined_destination.mkdir(parents=True, exist_ok=True)
        combined_years = _season_year_groups(
            contexts, overlay, season, COMBINED_GROUP.matches
        )
        combined_grid = f"{COMBINED_GROUP.slug}_by_planting_year{FIGURE_SUFFIX}"
        combined_sheet = f"{COMBINED_GROUP.slug}_decades_and_trend{FIGURE_SUFFIX}"
        combined_count, combined_year_count, combined_excluded = _write_variety_plate(
            COMBINED_GROUP,
            season,
            overlay,
            contexts,
            combined_years,
            limits,
            combined_destination / combined_grid,
        )
        combined_summarized, combined_unsummarized = _write_reference_sheet(
            COMBINED_GROUP,
            season,
            overlay,
            contexts,
            combined_years,
            limits,
            combined_destination / combined_sheet,
        )
        _write_combined_note(
            season,
            combined_count,
            combined_year_count,
            combined_summarized,
            combined_unsummarized,
            _year_exclusion_reasons(overlay, combined_years, combined_unsummarized),
            combined_excluded,
            len(groups) - len(CURATED_GROUPS),
            combined_grid,
            combined_sheet,
            combined_destination / README_FILENAME,
        )
        print(
            f"  pooled: trajectories={combined_count}; "
            f"planting years={combined_year_count}; "
            f"trend years={combined_summarized}; "
            f"not summarized={_year_runs_text(combined_unsummarized) or '—'}; "
            f"{combined_destination}"
        )

    uncovered = _uncovered_window_varieties(groups, contexts, season)

    window: tuple[int, tuple[int, ...], tuple[str, ...], str] | None = None
    if not args.no_window_plate:
        filename = f"{WINDOW_PLATE_STEM}{FIGURE_SUFFIX}"
        window_count, window_years, absent = _write_window_plate(
            groups, season, overlay, contexts, limits, destination / filename
        )
        window = (window_count, window_years, absent, filename)

    _write_readme(
        season,
        summaries,
        window,
        uncovered,
        _window_exclusion_tally(overlay, contexts, season),
        destination / README_FILENAME,
    )

    print(
        f"{SOURCE_NAME} {season}: wrote {2 * len(summaries)} variety plates "
        f"under {destination}"
    )
    print(
        f"  {'folder':<28} {'years':<11} {'traj':>5} {'trend':>6}  "
        "years not summarized"
    )
    for summary in summaries:
        span = _span_text((summary.first_year, summary.last_year))
        print(
            f"  {summary.group.slug:<28} {span:<11} "
            f"{summary.trajectory_count:>5} "
            f"{summary.summarized_year_count:>3}/{summary.year_count:<2} "
            f"{_year_runs_text(summary.excluded_years) or '—'}"
            + (
                f"  [{_reason_label(summary.exclusion_reasons)}]"
                if summary.exclusion_reasons
                else ""
            )
        )
    affected = [summary for summary in summaries if summary.excluded_years]
    print(
        f"  {len(affected)} of {len(summaries)} groups carry planting years that "
        "cannot be summarized"
    )
    if window is not None:
        window_count, window_years, absent, filename = window
        print(
            f"  {MV_028_WINDOW[0]}-{MV_028_WINDOW[1]} window: "
            f"trajectories={window_count}; planting years={len(window_years)}; "
            f"{filename}"
        )
    for year, names in sorted(uncovered.items()):
        print(f"  not drawn — {year}: {', '.join(names)}")
    print(f"  {README_FILENAME} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
