#!/usr/bin/env python3
"""Redraw the planting-decade strata with each connecting line coloured by variety.

`generate_response_curve_season_clusters.py` writes
`by_season/<season>/by_planting_year/annotated/<decade>.jpeg` with every
within-trajectory connector in one grey: the connector is there to show which
four points belong to one replicate, and nothing more. This recipe writes the
same strata on the same shared frame with the connector taking the colour of
the variety the source records for that trajectory, and a legend that names
every colour. Nothing else changes — the same trajectories, the same frame, the
same points in the same two treatment-class colours.

The colours are allocated once over the season's whole cluster-eligible variety
roster, in order of first recorded planting year, so a variety keeps its colour
in every figure here and the decade panels can be read side by side.

Output goes to a *subdirectory* of the season generator's `annotated/`, not
alongside its files. That generator replaces `by_season/` as one snapshot and
carries unmanaged whole directories across the swap, so a subdirectory survives
a rebuild and a loose file next to `<decade>.jpeg` would not.

Exploratory diagnostic, deliberately outside the release inventory (ANA-11);
this module is not on the release path. No curve is fitted anywhere here.

Re-run after any run of the season-cluster generator: the strata are recomputed
from the source, so these figures go stale when that partition changes.

Usage:
    conda run -n n_response python \\
      modules/n_response_curve/reporting/generate_ltcce_planting_year_variety_lines.py \\
      --config scriptCONFIG.toml --season DS
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.reporting.figure_axis_frames import (  # noqa: E402
    SharedAxisLimits,
    shared_axis_limits,
)
from n_response_curve.reporting.figure_captions import (  # noqa: E402
    EXPLORATORY_DIAGNOSTIC_DISCLAIMER as _DISCLAIMER,
    TITLE_FONT_SIZE as _TITLE_FONT_SIZE,
    reserve_suptitle as _reserve_suptitle,
    wrap_title_lines as _wrap_title_lines,
)
from n_response_curve.reporting.figure_output import (  # noqa: E402
    save_figure_atomically as _save_figure,
)
from n_response_curve.reporting.generate_response_curve_season_clusters import (  # noqa: E402
    PLANTING_YEAR_ANNOTATED_DIRNAME,
    PLANTING_YEAR_FIGURE_ONLY_DIRNAME,
    SEPARATED_NOTES_FILENAME,
    _SEPARATED_FIGURE_SUFFIX,
    _factor_agreement_lines,
    _promote,
    _season_token,
    _separated_footer_line,
)
from n_response_curve.reporting.response_curve_clusters import (  # noqa: E402
    TrajectoryContext,
    centroid_summary,
    read_ltcce_contexts,
    subset_overlay,
)
from n_response_curve.reporting.response_curve_season_clusters import (  # noqa: E402
    FACTOR_PLANTING_YEAR,
    MIN_FACTOR_STRATUM,
    FactorStratum,
    FactorSubstructure,
    SeasonClusteringResult,
    build_factor_substructure,
    build_season_clustering,
    cluster_ladder_shares,
    cluster_variety_span,
    cluster_year_range,
    factor_level_stem,
    format_shares,
    season_label,
)
from n_response_curve.reporting.source_config_spec import (  # noqa: E402
    load_source_spec,
)
from n_response_curve.reporting.source_dataset_overlays import (  # noqa: E402
    SourceDatasetOverlay,
    _adaptive_style,
    read_source_dataset_overlay,
)

SOURCE_NAME = "ltcce"
DEFAULT_SEASON = "DS"
SUPPORTED_SEASONS = ("DS", "EWS", "LWS")
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scriptCONFIG.toml"
CLUSTERS_ROOT = PROJECT_ROOT / "WF/03_Response_Curves/ltcce/clusters"

# A subdirectory of the season generator's `annotated/`, for the reason in the
# module docstring: `_carry_unmanaged_directories` preserves whole directories
# across the snapshot swap and deliberately does not preserve loose files.
# `figure_only/by_variety/` is a different product (one plate per variety), so
# the name says what is coloured rather than what it is split by.
VARIETY_LINES_DIRNAME = "lines_by_variety"

README_FILENAME = "README.md"
LEVEL_COMPARISON_STEM = "level_comparison"
LEVEL_COMPARISON_FILENAME = f"{LEVEL_COMPARISON_STEM}.jpeg"

# The recorded `Rep` value drawn, unless `--all-replicates` asks for the whole
# stratum back. Every experimental context in this season is replicated four
# times, and drawing all four lays four near-identical connectors of one colour
# on top of one another: a quarter of what reads as varietal spread on a
# 240-trajectory plate is the replication itself. One replicate is one
# trajectory per experimental context, so the spread that survives on the
# canvas is between contexts rather than within them.
DEFAULT_REPLICATE = "1"

# Matched to the season generator's own overlay canvas, so a decade drawn here
# is the same axes size as the same decade drawn there and the two can be laid
# side by side. The legend gets its own column on top of that width rather than
# eating into it.
_AXES_INCHES = (12.0, 7.5)
_LEGEND_COLUMN_INCHES = 2.1
_OVERLAY_TITLE_WIDTH = 116
# Legend rows one column of the side panel holds. Beyond this the legend takes a
# second column and the canvas widens; the *axes* keep `_AXES_INCHES` either
# way, and every decade in one run is given the same legend geometry — sized
# from the largest roster in the season — so the six plates stay the same shape
# and can be laid side by side.
_LEGEND_MAX_ROWS = 24

# The panel grid, on the season generator's per-panel geometry.
_PANEL_INCHES = (7.5, 8.0)
_PANEL_COLUMNS = 3
_COMPARISON_TITLE_WIDTH = 182
_COMPARISON_LEGEND_COLUMNS = 6
# Vertical inches one legend row occupies at `_LEGEND_FONT_SIZE`, plus the
# frame. Used only to size the strip the union legend is given at the foot of
# the comparison sheet; a generous estimate costs white space, a tight one
# clips the last row.
_LEGEND_ROW_INCHES = 0.24
_LEGEND_FONT_SIZE = 8

# Connector opacity. The season generator's `_adaptive_style` fades a connector
# to 0.08 in a 240-trajectory stratum, which is legible for one grey drawn 240
# times and washes 26 hues out to pastel when they have to be told apart. The
# floor is raised and the taper kept, so a sparse decade reads nearly solid and
# a dense one holds its hues while the points still show through on top.
_LINE_ALPHA_RANGE = (0.50, 0.90)
_LINE_ALPHA_NUMERATOR = 60.0
_LINE_WIDTH = 1.2

_LINE_LEGEND_TITLE = (
    "variety — colour of the within-trajectory\nconnecting lines (visual aid; not a fit)"
)
_COLOUR_ENCODING_LINE = (
    "each connecting line takes the colour of the variety the source records "
    "for that replicate; colours are allocated once across the season, so a "
    "variety keeps its colour in every figure in this folder"
)
_UNRECORDED_VARIETY_LABEL = "variety not recorded"
_UNRECORDED_VARIETY_COLOUR = (0.45, 0.45, 0.45)


def _shade_major(
    name: str,
    families: int,
    *,
    shades: int,
    drop_families: frozenset[int] = frozenset(),
) -> tuple[tuple[float, ...], ...]:
    """Reorder a family-major qualitative colormap so neighbours differ in hue.

    ``tab20``/``tab20b``/``tab20c`` are laid out as consecutive shades of one
    hue, so a run of consecutive entries is one colour fading. Taking one
    family at a time within a shade gives as many distinct hues as there are
    families before any hue is reused at a different lightness. Lifted from
    ``descriptive_statistics_figures._shade_major``, generalized over the shade
    count and over grey families, which are dropped rather than drawn: grey is
    what an unrecorded variety is drawn in below.
    """

    colours = plt.get_cmap(name).colors
    return tuple(
        tuple(colours[family * shades + shade])
        for shade in range(shades)
        for family in range(families)
        if family not in drop_families
    )


def _qualitative(
    name: str,
    count: int,
    *,
    drop: frozenset[int] = frozenset(),
) -> tuple[tuple[float, ...], ...]:
    """The first *count* entries of a small qualitative map, less *drop*."""

    colours = plt.get_cmap(name).colors
    return tuple(
        tuple(colour)
        for index, colour in enumerate(colours[:count])
        if index not in drop
    )


def _variety_palette() -> tuple[tuple[float, ...], ...]:
    """The ordered colour vocabulary a variety roster is allocated from.

    Built the way the descriptive-statistics recipe builds its categorical
    vocabulary — the qualitative colormaps, hue-major, with every grey family
    dropped — and then extended past that recipe's 67 entries, because one
    LTCCE season records more varieties than one stacked bar ever carries.
    Allocation is by first planting year, so a decade takes a nearly
    contiguous run and its members are as far apart in hue as the vocabulary
    allows.
    """

    return (
        # 10 hue families of 2 shades; family 7 is the grey pair.
        _shade_major("tab20", 10, shades=2, drop_families=frozenset({7}))
        # 4 hue families of 5 shades apiece.
        + _shade_major("tab20b", 4, shades=5)
        # ...and the same, without tab20c's fourth family, which is grey.
        + _shade_major("tab20c", 4, shades=5, drop_families=frozenset({3}))
        # Dark2, Set1, Set2 and Accent each close on a grey or a black,
        # dropped for the same reason. Set1's yellow (255,255,51) and Accent's
        # pale yellow (255,255,153) go with them: at this line width and these
        # opacities they are close to invisible on white, and a variety the
        # legend names but the canvas cannot show defeats the point of the
        # colouring.
        + _qualitative("Dark2", 7)
        + _qualitative("Set1", 8, drop=frozenset({5}))
        + _qualitative("Set2", 7)
        + _qualitative("Accent", 7, drop=frozenset({3}))
    )


def _variety_of(contexts: Mapping[str, TrajectoryContext], trajectory_id: str) -> str:
    context = contexts.get(trajectory_id)
    variety = context.variety.strip() if context is not None else ""
    return variety or _UNRECORDED_VARIETY_LABEL


def _variety_colours(
    substructure: FactorSubstructure,
    contexts: Mapping[str, TrajectoryContext],
) -> dict[str, tuple[float, ...]]:
    """Allocate one colour per variety in the season, by first planting year.

    Fail-closed on the roster outgrowing the vocabulary. Wrapping would put two
    varieties of one decade on one colour, and the colour is the only thing
    separating them on the canvas — a silently ambiguous figure is worse than
    no figure.
    """

    first_seen: dict[str, tuple[int, str]] = {}
    for stratum in substructure.strata:
        for trajectory_id in stratum.trajectory_ids:
            variety = _variety_of(contexts, trajectory_id)
            context = contexts.get(trajectory_id)
            year = context.year if context is not None and context.year > 0 else 9999
            existing = first_seen.get(variety)
            if existing is None or (year, variety) < existing:
                first_seen[variety] = (year, variety)

    ordered = sorted(first_seen, key=lambda variety: first_seen[variety])
    palette = _variety_palette()
    # The vocabulary is assembled from seven colormaps that were not designed
    # against one another, so its entries are checked rather than assumed
    # distinct: a repeat would put two varieties on one colour with the length
    # check below still passing.
    if len(set(palette)) != len(palette):
        raise RuntimeError(
            f"the {len(palette)}-colour vocabulary repeats "
            f"{len(palette) - len(set(palette))} colour(s)"
        )
    recorded = [variety for variety in ordered if variety != _UNRECORDED_VARIETY_LABEL]
    if len(recorded) > len(palette):
        raise RuntimeError(
            f"{len(recorded)} varieties in {substructure.season} exceed the "
            f"{len(palette)}-colour vocabulary; two would share a colour"
        )
    colours = {
        variety: palette[index] for index, variety in enumerate(recorded)
    }
    if _UNRECORDED_VARIETY_LABEL in first_seen:
        colours[_UNRECORDED_VARIETY_LABEL] = _UNRECORDED_VARIETY_COLOUR
    return colours


def _varieties_present(
    trajectory_ids: Sequence[str],
    contexts: Mapping[str, TrajectoryContext],
    colours: Mapping[str, tuple[float, ...]],
) -> list[str]:
    """The varieties among *trajectory_ids*, in the roster's colour order.

    Takes trajectory ids rather than a whole stratum because what is drawn is
    no longer necessarily the whole stratum: `--replicate` keeps one quarter of
    it, and a legend naming a variety that no line on the canvas carries is a
    legend that misdescribes the figure it belongs to.
    """

    present = {
        _variety_of(contexts, trajectory_id) for trajectory_id in trajectory_ids
    }
    order = list(colours)
    return [variety for variety in order if variety in present]


def _replicate_of(
    contexts: Mapping[str, TrajectoryContext],
    trajectory_id: str,
) -> str:
    context = contexts.get(trajectory_id)
    return context.replicate.strip() if context is not None else ""


def recorded_replicates(
    substructure: FactorSubstructure,
    contexts: Mapping[str, TrajectoryContext],
) -> tuple[str, ...]:
    """Every `Rep` value the season's cluster-eligible trajectories carry."""

    values = {
        _replicate_of(contexts, trajectory_id)
        for stratum in substructure.strata
        for trajectory_id in stratum.trajectory_ids
    }
    return tuple(sorted(value for value in values if value))


def drawn_ids(
    trajectory_ids: Sequence[str],
    contexts: Mapping[str, TrajectoryContext],
    replicate: str | None,
) -> tuple[str, ...]:
    """The trajectories of one stratum that are actually drawn.

    `replicate` of ``None`` draws the stratum whole. Otherwise only the
    trajectories the source records under that `Rep` value survive.

    The filter deliberately runs *after* stratification and after the colour
    allocation, not before. One replicate is a quarter of each stratum, and
    subsetting first would drop the 1960s (44 to 11) and the 1990s (32 to 8)
    below `--min-factor-stratum` — they would stop being decades at all — and
    would re-allocate the colours over whatever roster the surviving strata
    then held. Filtering at draw time changes which lines are drawn and
    nothing else: the decades, the frame and every colour stay the ones the
    neighbouring products use.
    """

    if replicate is None:
        return tuple(trajectory_ids)
    return tuple(
        trajectory_id
        for trajectory_id in trajectory_ids
        if _replicate_of(contexts, trajectory_id) == replicate
    )


def check_replicate(
    substructure: FactorSubstructure,
    contexts: Mapping[str, TrajectoryContext],
    replicate: str | None,
) -> None:
    """Fail closed on a replicate that draws nothing, or empties a stratum.

    An unrecorded `Rep` value would otherwise render six empty frames, and a
    stratum the value happens to miss would render one — an empty axes on a
    shared frame reads as "this decade has no data", which is a claim about
    the experiment rather than about the filter.
    """

    if replicate is None:
        return
    recorded = recorded_replicates(substructure, contexts)
    if replicate not in recorded:
        raise SystemExit(
            f"Rep={replicate!r} is not recorded in {substructure.season}; "
            f"this season records {', '.join(recorded) or 'none'}"
        )
    empty = [
        stratum.level
        for stratum in substructure.strata
        if not drawn_ids(stratum.trajectory_ids, contexts, replicate)
    ]
    if empty:
        raise SystemExit(
            f"Rep={replicate} has no trajectories in {', '.join(empty)}; "
            "those strata would be drawn as empty frames"
        )


def drawn_member_count(
    substructure: FactorSubstructure,
    contexts: Mapping[str, TrajectoryContext],
    replicate: str | None,
) -> int:
    """The season total the per-stratum shares are taken against.

    The drawn total, not `substructure.member_count`: a share of a population
    a quarter of which is on the canvas is not the share the figure shows.
    """

    return sum(
        len(drawn_ids(stratum.trajectory_ids, contexts, replicate))
        for stratum in substructure.strata
    )


def replicate_headline(replicate: str | None) -> str:
    """The scope label appended to every headline when one replicate is drawn.

    The season generator's own replicate-subset recipe carries the same idea in
    its `headline_context`, and for the same reason: a plate that has left this
    folder must not be mistakable for the all-replicate view.
    """

    return "" if replicate is None else f" — recorded Rep={replicate} only"


def replicate_scope_lines(
    replicate: str | None,
    recorded: Sequence[str],
) -> list[str]:
    """The disclosure a one-replicate plate carries in place of nothing.

    One line, deliberately: this caption already runs to nine disclosures on a
    canvas that gives the title as many rows as it asks for and takes them out
    of the axes, and the fuller account is in the folder's README and in
    `separated_figure_notes.md`, both of which a plate points at. What must be
    on the face of the figure is what the reader would otherwise get wrong —
    the scope, the size of the counts, and what the agreement statistics are
    computed over.
    """

    if replicate is None:
        return []
    return [
        f"one recorded replicate only: Rep={replicate} of the "
        f"{len(recorded)} this season records — one connector per experimental "
        f"context rather than {len(recorded)}, every count about a "
        f"{1 / len(recorded):.0%} share of the season, and the agreement "
        "statistics below still computed over the full stratification",
    ]


def _connector_alpha(trajectory_count: int) -> float:
    return float(
        np.clip(
            _LINE_ALPHA_NUMERATOR / max(trajectory_count, 1),
            *_LINE_ALPHA_RANGE,
        )
    )


def _draw_variety_lines(
    overlay: SourceDatasetOverlay,
    contexts: Mapping[str, TrajectoryContext],
    colours: Mapping[str, tuple[float, ...]],
    axes: Any,
) -> list[Any]:
    """Draw one stratum: points by treatment class, connectors by variety.

    Returns the treatment-class scatter handles, so the caller can put them in
    the legend above the variety block. The points keep the two colours the
    season generator gives them — this recipe recolours the connectors and
    nothing else, so a reader who knows the original figure can still find the
    zero-N column.
    """

    marker_size, marker_alpha, _ = _adaptive_style(
        overlay.summary.finite_observation_count,
        overlay.summary.trajectory_count,
    )
    scatter_handles: list[Any] = []
    for treatment_class in overlay.summary.treatment_classes:
        observations = [
            observation
            for trajectory in overlay.trajectories
            for observation in trajectory.observations
            if observation.treatment_class == treatment_class
        ]
        scatter_handles.append(
            axes.scatter(
                [observation.n_rate_kg_ha for observation in observations],
                [observation.yield_t_ha for observation in observations],
                s=marker_size,
                alpha=marker_alpha,
                label=treatment_class,
                zorder=3,
            )
        )

    alpha = _connector_alpha(overlay.summary.trajectory_count)
    for trajectory in overlay.trajectories:
        variety = _variety_of(contexts, trajectory.trajectory_id)
        axes.plot(
            [observation.n_rate_kg_ha for observation in trajectory.observations],
            [observation.yield_t_ha for observation in trajectory.observations],
            color=colours[variety],
            linewidth=_LINE_WIDTH,
            alpha=alpha,
            zorder=2,
        )
    axes.set_xlabel("Applied N (kg N/ha)")
    axes.set_ylabel("Grain yield (t/ha)")
    return scatter_handles


def _variety_handles(
    varieties: Sequence[str],
    colours: Mapping[str, tuple[float, ...]],
) -> list[Any]:
    """Opaque line handles, one per variety, for the legend.

    Drawn at full opacity whatever the canvas alpha is: the legend's job is to
    name the hue, and a key swatch faded to 0.30 cannot do it.
    """

    return [
        Line2D([0], [0], color=colours[variety], linewidth=2.2, label=variety)
        for variety in varieties
    ]


def level_facts(
    result: SeasonClusteringResult,
    trajectory_ids: Sequence[str],
    member_count: int,
) -> dict[str, Any]:
    """Every number one stratum's caption and its notes-table row both use.

    Recomputed from the ids actually drawn rather than read off the stratum:
    `FactorStratum.ladder_shares` and `variety_names` describe the whole
    stratum, and under `--replicate` that is four times what is on the canvas.
    """

    ids = tuple(trajectory_ids)
    years = cluster_year_range(ids, result.contexts)
    centroid = centroid_summary(ids, result.features)
    return {
        "count": len(ids),
        "share": 100 * len(ids) / member_count if member_count else 0.0,
        "years": f"{years[0]}-{years[1]}" if years else "years unknown",
        "varieties": cluster_variety_span(ids, result.contexts),
        "ladders": format_shares(cluster_ladder_shares(ids, result.features), limit=4),
        "zero_n": centroid["yield_at_zero_n_t_ha"],
        "response": centroid["response_above_zero_n_t_ha"],
        "peak": 100 * centroid["relative_n_at_peak"],
    }


def _level_identity_lines(
    substructure: FactorSubstructure,
    stratum: FactorStratum,
    facts: Mapping[str, Any],
    member_count: int,
    replicate: str | None,
) -> list[str]:
    """What identifies this plate and what it measures — and nothing else.

    This is the half a plates-only figure keeps: a stratum plate carries its
    own numbers nowhere but its caption, so stripping them would leave a plate
    that cannot be read at all. The interpretive half is
    :func:`_level_interpretive_lines`, and the two are concatenated for the
    annotated form, so the plate and its notes cannot drift apart.
    """

    definition = substructure.definition
    return [
        f"source={SOURCE_NAME} — {season_label(substructure.season)}, "
        f"{definition.level_title(stratum.level)} — connectors coloured by "
        f"variety{replicate_headline(replicate)}",
        f"{facts['count']} of {member_count} cluster-eligible trajectories "
        f"in this season ({facts['share']:.0f}%); {facts['years']}; "
        f"{facts['varieties']} varieties",
        f"applied-N ladders: {facts['ladders']}",
        f"mean zero-N yield={facts['zero_n']:.2f} t/ha; "
        f"mean response={facts['response']:.2f} t/ha; "
        f"mean peak at {facts['peak']:.0f}% of the top rate",
    ]


def _level_interpretive_lines(
    substructure: FactorSubstructure,
    stratum: FactorStratum,
    agreement_lines: Sequence[str],
    replicate: str | None,
    recorded: Sequence[str],
    *,
    generic_level: bool = False,
) -> list[str]:
    """What the level is, what it is confounded with, and what not to conclude.

    The half that moves off the plate and into `separated_figure_notes.md`.

    `generic_level` writes the first line to stand for every plate in the
    family rather than for this one. The notes file sets these disclosures once
    beneath a table of six plates, and a line naming the 2010s under a table
    whose first row is the 1960s is simply wrong. Asking for the generic form
    here — rather than editing the list afterwards by index — is what keeps the
    two forms one list, so a change to a disclosure reaches the plate and the
    notes together.
    """

    definition = substructure.definition
    level_line = (
        "every plate here is a recorded stratum, not a cluster: its "
        f"trajectories are the ones the source's {definition.source_column} "
        "column assigns to that level, and nothing was clustered to produce it"
        if generic_level
        else "recorded stratum, not a cluster: every trajectory here carries "
        f"{definition.source_column}={stratum.level!r} in the source"
    )
    return [
        level_line,
        *replicate_scope_lines(replicate, recorded),
        _COLOUR_ENCODING_LINE,
        definition.identity_caveat,
        *agreement_lines,
        _DISCLAIMER,
    ]


def _level_title_lines(
    substructure: FactorSubstructure,
    stratum: FactorStratum,
    facts: Mapping[str, Any],
    member_count: int,
    agreement_lines: Sequence[str],
    replicate: str | None,
    recorded: Sequence[str],
) -> list[str]:
    """The annotated stratum caption: both halves, in that order."""

    return [
        *_level_identity_lines(substructure, stratum, facts, member_count, replicate),
        *_level_interpretive_lines(
            substructure, stratum, agreement_lines, replicate, recorded
        ),
    ]


def _legend_columns(
    drawn: Mapping[str, Sequence[str]],
    contexts: Mapping[str, TrajectoryContext],
    colours: Mapping[str, tuple[float, ...]],
    overlay: SourceDatasetOverlay,
) -> int:
    """Columns the side legend needs, sized from the largest drawn roster.

    One number for the whole run, not one per decade: a legend that widened
    with the roster would change the canvas width from plate to plate, and
    these plates exist to be compared. Sized from what is drawn, so the
    annotated and plates-only families of one run share a canvas too.
    """

    entries = max(
        len(_varieties_present(ids, contexts, colours)) for ids in drawn.values()
    ) + len(overlay.summary.treatment_classes)
    return max(1, math.ceil(entries / _LEGEND_MAX_ROWS))


def _write_level_figure(
    overlay: SourceDatasetOverlay,
    trajectory_ids: Sequence[str],
    contexts: Mapping[str, TrajectoryContext],
    colours: Mapping[str, tuple[float, ...]],
    destination: Path,
    limits: SharedAxisLimits,
    title_lines: Sequence[str],
    legend_columns: int,
) -> None:
    """One planting decade, with its own variety legend beside the axes.

    The caption arrives already assembled, so one function draws both the
    annotated plate and its plates-only twin and the two cannot come out on
    different canvases.
    """

    axes_width, height = _AXES_INCHES
    legend_width = _LEGEND_COLUMN_INCHES * legend_columns
    figure = plt.figure(
        figsize=(axes_width + legend_width, height),
        constrained_layout=True,
    )
    try:
        grid = figure.add_gridspec(1, 2, width_ratios=(axes_width, legend_width))
        axes = figure.add_subplot(grid[0, 0])
        legend_axes = figure.add_subplot(grid[0, 1])
        legend_axes.set_axis_off()

        scatter_handles = _draw_variety_lines(
            subset_overlay(overlay, tuple(trajectory_ids)),
            contexts,
            colours,
            axes,
        )
        limits.apply(axes)
        axes.set_title(
            _wrap_title_lines(title_lines, width=_OVERLAY_TITLE_WIDTH),
            fontsize=_TITLE_FONT_SIZE,
        )

        varieties = _varieties_present(trajectory_ids, contexts, colours)
        legend_axes.legend(
            handles=[*scatter_handles, *_variety_handles(varieties, colours)],
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            ncol=legend_columns,
            fontsize=_LEGEND_FONT_SIZE,
            title=_LINE_LEGEND_TITLE,
            title_fontsize=_LEGEND_FONT_SIZE,
            frameon=False,
            handlelength=2.4,
            borderaxespad=0.0,
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


def _comparison_identity_lines(
    substructure: FactorSubstructure,
    replicate: str | None,
) -> list[str]:
    """What the side-by-side sheet is, and what its shared frame means."""

    definition = substructure.definition
    return [
        f"source={SOURCE_NAME} — {season_label(substructure.season)}: "
        f"{definition.title} levels side by side, connectors coloured by "
        f"variety{replicate_headline(replicate)}",
        "the panels share both axes; each is a recorded stratum, so a "
        "difference between them is whatever the factor is confounded with, "
        "not a result",
    ]


def _comparison_interpretive_lines(
    substructure: FactorSubstructure,
    agreement_lines: Sequence[str],
    replicate: str | None,
    recorded: Sequence[str],
) -> list[str]:
    """The side-by-side sheet's half that moves into the notes file."""

    return [
        *replicate_scope_lines(replicate, recorded),
        _COLOUR_ENCODING_LINE,
        substructure.definition.identity_caveat,
        *agreement_lines,
        _DISCLAIMER,
    ]


def _write_level_comparison(
    result: SeasonClusteringResult,
    overlay: SourceDatasetOverlay,
    substructure: FactorSubstructure,
    drawn: Mapping[str, Sequence[str]],
    member_count: int,
    contexts: Mapping[str, TrajectoryContext],
    colours: Mapping[str, tuple[float, ...]],
    destination: Path,
    limits: SharedAxisLimits,
    suptitle_lines: Sequence[str],
) -> None:
    """Every decade side by side, over one legend for the whole season roster.

    The legend is the union rather than each panel's own, because that is what
    makes a colour mean the same thing in two panels. A variety absent from a
    panel is simply not drawn there.
    """

    strata = substructure.strata
    if len(strata) < 2:
        return

    definition = substructure.definition
    columns = 2 if len(strata) == 4 else min(_PANEL_COLUMNS, len(strata))
    rows = math.ceil(len(strata) / columns)
    panel_width, panel_height = _PANEL_INCHES
    varieties = list(colours)
    legend_rows = math.ceil(
        (len(varieties) + len(overlay.summary.treatment_classes))
        / _COMPARISON_LEGEND_COLUMNS
    )
    # One row for the legend title, and one for the frame's breathing room.
    legend_height = (legend_rows + 2) * _LEGEND_ROW_INCHES

    figure = plt.figure(
        figsize=(panel_width * columns, panel_height * rows + legend_height),
        constrained_layout=True,
    )
    try:
        grid = figure.add_gridspec(
            rows + 1,
            columns,
            height_ratios=(*(panel_height,) * rows, legend_height),
        )
        axes_list = [
            figure.add_subplot(grid[index // columns, index % columns])
            for index in range(len(strata))
        ]
        legend_axes = figure.add_subplot(grid[rows, :])
        legend_axes.set_axis_off()

        scatter_handles: list[Any] = []
        for stratum, axes in zip(strata, axes_list, strict=True):
            ids = tuple(drawn[stratum.level])
            handles = _draw_variety_lines(
                subset_overlay(overlay, ids), contexts, colours, axes
            )
            if not scatter_handles:
                scatter_handles = handles
            limits.apply(axes)
            facts = level_facts(result, ids, member_count)
            axes.set_title(
                f"{definition.level_title(stratum.level)}\n"
                f"{facts['count']} of {member_count} trajectories "
                f"({facts['share']:.0f}%); {facts['years']}; "
                f"{facts['varieties']} varieties\n"
                f"applied-N ladders: "
                f"{format_shares(cluster_ladder_shares(ids, result.features), limit=2)}\n"
                f"mean zero-N yield={facts['zero_n']:.2f} t/ha; "
                f"mean response={facts['response']:.2f} t/ha",
                fontsize=_TITLE_FONT_SIZE,
            )

        legend_axes.legend(
            handles=[*scatter_handles, *_variety_handles(varieties, colours)],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=_COMPARISON_LEGEND_COLUMNS,
            fontsize=_LEGEND_FONT_SIZE,
            title=(
                f"{_LINE_LEGEND_TITLE} — all {len(varieties)} varieties in this "
                "season, on the colours every panel above uses"
            ),
            title_fontsize=_LEGEND_FONT_SIZE,
            frameon=False,
            handlelength=2.4,
            borderaxespad=0.0,
        )

        _reserve_suptitle(
            figure,
            _wrap_title_lines(suptitle_lines, width=_COMPARISON_TITLE_WIDTH),
            # The union legend has its own gridspec row, so no extra strip is
            # reserved for a figure-level one.
            legend_strip=0.0,
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


def replicate_readme_section(
    replicate: str | None,
    recorded: Sequence[str],
) -> list[str]:
    """The section every README in this family carries about the `Rep` filter."""

    if replicate is None:
        return [
            "## Every replicate is drawn",
            "",
            f"This run was given `--all-replicates`, so all {len(recorded)} "
            "recorded replicates of every experimental context are on the "
            f"canvas. {len(recorded)} near-identical connectors of one colour "
            "sit on top of one another wherever the replicates agree, so some "
            "of what reads as spread here is the replication rather than "
            f"anything varietal. The default, `--replicate {DEFAULT_REPLICATE}`, "
            "draws one.",
            "",
        ]
    return [
        "## One replicate only",
        "",
        f"Every trajectory drawn here carries `Rep={replicate}`. The season "
        f"records {len(recorded)} replicates ({', '.join(recorded)}) of every "
        f"experimental context, and drawing all of them lays {len(recorded)} "
        "connectors of one colour along nearly the same path: a "
        f"{1 / len(recorded):.0%} share of what reads as varietal spread on a "
        "crowded plate is the replication itself. With one replicate a context "
        "contributes one line, so the spread left on the canvas is spread "
        "between contexts.",
        "",
        "Every count on every figure and in this file is therefore about a "
        f"{1 / len(recorded):.0%} share of the season. The decades, the frame "
        "and the colours are not: the filter runs after the strata are formed "
        "and after the colours are allocated, precisely so that a decade here "
        "is the decade the neighbouring products draw and a colour means the "
        "same variety in all of them. `--all-replicates` draws the stratum "
        "whole, and `--replicate <n>` picks another one.",
        "",
    ]


def _readme_lines(
    substructure: FactorSubstructure,
    contexts: Mapping[str, TrajectoryContext],
    colours: Mapping[str, tuple[float, ...]],
    drawn: Mapping[str, Sequence[str]],
    written: Sequence[str],
    season: str,
    replicate: str | None,
    recorded: Sequence[str],
) -> list[str]:
    adjective = season_label(season)
    lines = [
        f"# {adjective} — the planting-decade strata, connectors coloured by variety",
        "",
        "The figures in `../` drawn again with one change: a within-trajectory "
        "connecting line takes the colour of the variety the source records for "
        "that replicate, instead of the single grey `../` gives every connector. "
        "The shared frame, the two treatment-class point colours and every "
        "disclosure are the same."
        + (
            " The trajectories are too."
            if replicate is None
            else " The trajectories are the ones a single recorded replicate "
            "contributes, which is the one further difference and is the "
            "subject of the section below."
        ),
        "",
        "## What the colour is, and what it is not",
        "",
        "A connector is a visual aid that says which observed points belong to "
        "one replicate. It is not a fit, and colouring it does not make it one. "
        + (
            "What the colour adds is the ability to ask whether replicates of "
            "one variety travel together inside a decade — and to see, when "
            "they do not, that the spread inside a decade is not varietal."
            if replicate is None
            else "What the colour adds is the ability to ask whether the "
            "varieties of one decade separate at all. With one replicate per "
            "experimental context there is no within-variety replication left "
            "on the canvas to mistake for a varietal difference, and equally "
            "none to check a varietal difference against: what the spread "
            "between two colours is, is the spread between the contexts those "
            "varieties were grown in."
        ),
        "",
        f"Colours are allocated once over all {len(colours)} varieties this "
        "season records among its cluster-eligible trajectories, in order of "
        "first recorded planting year. A variety therefore carries the same "
        "colour in every figure in this folder, and a decade's varieties take a "
        "nearly contiguous run of the vocabulary, which is what keeps them apart "
        "on the canvas. The allocation is fail-closed: a roster larger than the "
        "vocabulary raises rather than letting two varieties share a colour.",
        "",
        *replicate_readme_section(replicate, recorded),
        "## What is here",
        "",
    ]
    for stratum in substructure.strata:
        stem = factor_level_stem(substructure.definition, stratum.level)
        ids = drawn[stratum.level]
        varieties = _varieties_present(ids, contexts, colours)
        lines.append(
            f"- `{stem}.jpeg` — the same stratum as `../{stem}.jpeg`; "
            f"{len(ids)} trajectories drawn of the stratum's "
            f"{len(stratum.trajectory_ids)}, "
            f"{len(varieties)} varieties in the legend."
        )
    lines += [
        f"- `{LEVEL_COMPARISON_FILENAME}` — the decades side by side, over one "
        f"legend carrying all {len(colours)} varieties. The union legend is what "
        "makes a colour mean the same thing in two panels.",
        "",
        "## What is deliberately not here",
        "",
        "`../annual_trend.jpeg` draws no trajectory connectors at all, so there "
        "is nothing to recolour. `../decades_and_trend.jpeg`, "
        "`../decades_and_trend_matched_colours.jpeg` and "
        "`../decades_designs_and_trend.jpeg` carry each decade as a small inset "
        "on a 34-inch sheet whose colour already encodes the applied-N era or "
        "the plotted quantity; a variety legend of this size does not fit an "
        "inset, and a second colour vocabulary on that sheet would contradict "
        "the one it already declares.",
        "",
        "## The plates-only twin",
        "",
        f"Every figure here also exists under `../../"
        f"{PLANTING_YEAR_FIGURE_ONLY_DIRNAME}/{VARIETY_LINES_DIRNAME}/` as "
        f"`<stem>{_SEPARATED_FIGURE_SUFFIX}`: the same canvas with the "
        "interpretive half of the caption lifted off and set in "
        f"`{SEPARATED_NOTES_FILENAME}` beside it. Use these annotated plates "
        "when a figure has to travel alone, and that folder when the prose "
        "belongs in the surrounding document. `--no-separate-notes` suppresses "
        "the whole folder.",
        "",
        "## Interpretation boundary",
        "",
        "Exploratory diagnostic, outside the governed analysis inventory "
        "(ANA-11). No curve is fitted. Every point is an observed yield at an "
        "observed applied-N rate. A decade still carries its applied-N ladder, "
        "its plot design and its era with it, so a difference between panels is "
        "a difference between experiments — colouring by variety does not "
        "separate the variety from any of that.",
        "",
        "## Regenerating",
        "",
        "```bash",
        "conda run -n n_response python \\",
        "  modules/n_response_curve/reporting/"
        "generate_ltcce_planting_year_variety_lines.py \\",
        f"  --config scriptCONFIG.toml --season {season}"
        + ("" if replicate is None else f" --replicate {replicate}"),
        "```",
        "",
        "One run writes this folder and its plates-only twin together, so the "
        "two can never come from different replicate or stratum settings.",
        "",
        "The strata are recomputed from the source on every run, so re-run this "
        "after any run of `generate_response_curve_season_clusters.py`, which "
        "owns `../` and replaces `by_season/` as one snapshot. This folder "
        "survives that swap because it is a directory, not a loose file beside "
        f"`../{LEVEL_COMPARISON_FILENAME}`.",
        "",
        f"Figures written: {len(written)}.",
        "",
    ]
    return lines


def _notes_lines(
    substructure: FactorSubstructure,
    stratum_rows: Sequence[Mapping[str, Any]],
    shared_prose: Sequence[str],
    comparison_prose: Sequence[str],
    season: str,
    replicate: str | None,
    recorded: Sequence[str],
) -> list[str]:
    """The prose the `*_figure.jpeg` plates in `figure_only/` no longer carry.

    Built from the same line lists the annotated captions are built from — see
    :func:`_level_interpretive_lines` — so a plate and its notes cannot
    disagree. A notes file that can contradict the figure it belongs to is
    worse than none, because the reader has no way to tell which is current.

    The six decade plates share every disclosure and differ only in their
    numbers, so the numbers become one table and the disclosures are set once
    beneath it.
    """

    annotated_root = (
        f"../../{PLANTING_YEAR_ANNOTATED_DIRNAME}/{VARIETY_LINES_DIRNAME}"
    )
    lines = [
        f"# {season_label(season)} — the planting-decade strata, connectors "
        "coloured by variety: notes for the separated figures",
        "",
        f"{_DISCLAIMER[0].upper()}{_DISCLAIMER[1:]}.",
        "",
        f"These notes belong to the `*{_SEPARATED_FIGURE_SUFFIX}` plates in "
        "this folder. Each of those carries its identity, its own measurements "
        "and its legend and nothing else; everything about how to read them, "
        "and what not to conclude from them, is here. Every figure also exists "
        f"under `{annotated_root}/` in an annotated form that sets the same "
        "text on its own face, for when a plate has to travel alone.",
        "",
        *replicate_readme_section(replicate, recorded),
    ]

    if stratum_rows:
        lines += [
            f"## The {len(stratum_rows)} `<decade>{_SEPARATED_FIGURE_SUFFIX}` plates",
            "",
            "One recorded stratum each, on the shared frame every figure in "
            "this folder uses, with a legend naming every variety colour the "
            f"panel draws. The annotated form of each is `{annotated_root}/"
            "<decade>.jpeg`.",
            "",
            "| Plate | Level | Trajectories drawn | Years | Varieties | "
            "Applied-N ladders | Mean zero-N (t/ha) | Mean response (t/ha) | "
            "Mean peak (% of top rate) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in stratum_rows:
            lines.append(
                f"| `{row['figure']}` | {row['level']} | "
                f"{row['count']} ({row['share']:.0f}%) | {row['years']} | "
                f"{row['varieties']} | {row['ladders']} | {row['zero_n']:.2f} | "
                f"{row['response']:.2f} | {row['peak']:.0f} |"
            )
        lines.append("")
        if shared_prose:
            lines += ["### True of every plate above", ""]
            lines += [f"- {entry[0].upper()}{entry[1:]}" for entry in shared_prose]
            lines.append("")

    if comparison_prose:
        lines += [
            f"## `{LEVEL_COMPARISON_STEM}{_SEPARATED_FIGURE_SUFFIX}`",
            "",
            "The decades side by side over one legend carrying the season's "
            "whole variety roster. The union legend is what makes a colour "
            "mean the same thing in two panels; a variety absent from a panel "
            "is simply not drawn there. Each panel keeps its own measurements "
            "in its own title. The annotated form is "
            f"`{annotated_root}/{LEVEL_COMPARISON_FILENAME}`.",
            "",
        ]
        extra = [entry for entry in comparison_prose if entry not in shared_prose]
        if extra:
            lines += [f"- {entry[0].upper()}{entry[1:]}" for entry in extra]
        else:
            lines.append(
                "Every disclosure under *True of every plate above* holds here "
                "too — this sheet is those panels, on one canvas."
            )
        lines.append("")

    lines += [
        "## Regenerating",
        "",
        "Written by `generate_ltcce_planting_year_variety_lines.py` as the "
        f"`{PLANTING_YEAR_FIGURE_ONLY_DIRNAME}/` companion to "
        f"`{annotated_root}/`, in the same run; `--no-separate-notes` "
        f"suppresses this file and every `*{_SEPARATED_FIGURE_SUFFIX}` beside "
        "it.",
        "",
        "```bash",
        "conda run -n n_response python \\",
        "  modules/n_response_curve/reporting/"
        "generate_ltcce_planting_year_variety_lines.py \\",
        f"  --config scriptCONFIG.toml --season {season}"
        + ("" if replicate is None else f" --replicate {replicate}"),
        "```",
        "",
    ]
    return lines


def default_output_dir(season: str) -> Path:
    """The variety-coloured folder inside that season's annotated plates.

    Derived from the season and from the season generator's own directory
    names, so this recipe and that one cannot drift apart on where the folder
    belongs.
    """

    return (
        CLUSTERS_ROOT
        / "by_season"
        / _season_token(season)
        / "by_planting_year"
        / PLANTING_YEAR_ANNOTATED_DIRNAME
        / VARIETY_LINES_DIRNAME
    )


def figure_only_output_dir(season: str) -> Path:
    """The plates-only twin, under the season's `figure_only/` tier.

    A sibling of the season generator's own separated plates rather than a
    subdirectory of `annotated/`: `figure_only/` is a presentation tier, and a
    reader who wants plates without prose wants all of them in one place. It is
    a directory for the same reason `annotated/`'s is — `_promote` carries a
    whole directory across a snapshot swap and deliberately does not carry a
    loose file.
    """

    return (
        CLUSTERS_ROOT
        / "by_season"
        / _season_token(season)
        / "by_planting_year"
        / PLANTING_YEAR_FIGURE_ONLY_DIRNAME
        / VARIETY_LINES_DIRNAME
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--season",
        default=DEFAULT_SEASON,
        choices=SUPPORTED_SEASONS,
        help=f"Season whose planting-decade strata are redrawn (default: {DEFAULT_SEASON})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination for the annotated plates (default: that season's "
        f"annotated/{VARIETY_LINES_DIRNAME}/)",
    )
    parser.add_argument(
        "--figure-only-dir",
        type=Path,
        default=None,
        help="Destination for the plates-only twin (default: that season's "
        f"{PLANTING_YEAR_FIGURE_ONLY_DIRNAME}/{VARIETY_LINES_DIRNAME}/)",
    )
    parser.add_argument(
        "--min-factor-stratum",
        type=int,
        default=MIN_FACTOR_STRATUM,
        help="Smallest planting decade drawn as its own figure rather than pooled",
    )
    parser.add_argument(
        "--replicate",
        default=DEFAULT_REPLICATE,
        help="Draw only the trajectories the source records under this Rep "
        f"value (default: {DEFAULT_REPLICATE}). Applied after the strata are "
        "formed and the colours allocated, so the decades and the colour "
        "vocabulary do not move with it",
    )
    parser.add_argument(
        "--all-replicates",
        action="store_true",
        help="Draw every recorded replicate, as this recipe originally did",
    )
    parser.add_argument(
        "--no-level-comparison",
        action="store_true",
        help="Skip the side-by-side sheet and write the per-decade figures only",
    )
    parser.add_argument(
        "--no-separate-notes",
        action="store_true",
        help="Write the annotated plates only, without the plates-only twin "
        f"and its {SEPARATED_NOTES_FILENAME}",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    season = args.season.strip().upper()
    destination = (args.output_dir or default_output_dir(season)).resolve()
    figure_only = (args.figure_only_dir or figure_only_output_dir(season)).resolve()
    separate_notes = not args.no_separate_notes

    source_path, encoding = load_source_spec(
        args.config, SOURCE_NAME, relative_root=PROJECT_ROOT
    )
    overlay = read_source_dataset_overlay(source_path, SOURCE_NAME, encoding=encoding)
    contexts = read_ltcce_contexts(source_path, encoding=encoding)
    result = build_season_clustering(overlay, contexts)
    if season not in result.partitions:
        raise SystemExit(
            f"{season} is not decomposed by planting year in this source "
            f"({result.unclustered_seasons.get(season, 'season absent')})"
        )
    substructure = build_factor_substructure(
        result,
        season,
        FACTOR_PLANTING_YEAR,
        minimum_stratum=args.min_factor_stratum,
    )

    replicate = None if args.all_replicates else args.replicate.strip()
    check_replicate(substructure, contexts, replicate)
    recorded = recorded_replicates(substructure, contexts)
    # One filtered id list per decade, computed once: every caption, every
    # legend, every README line and every notes row is taken from it, so no two
    # of them can describe different sets of trajectories.
    drawn = {
        stratum.level: drawn_ids(stratum.trajectory_ids, contexts, replicate)
        for stratum in substructure.strata
    }
    member_count = drawn_member_count(substructure, contexts, replicate)

    # Pinned from the parent overlay before any subsetting, exactly as the
    # season generator does, so these figures share that product's frame. The
    # colours are allocated over the whole season for the same reason: a
    # variety keeps one colour whatever `--replicate` is set to.
    limits = shared_axis_limits(overlay)
    colours = _variety_colours(substructure, contexts)
    agreement_lines = _factor_agreement_lines(substructure)
    legend_columns = _legend_columns(drawn, contexts, colours, overlay)

    facts = {
        stratum.level: level_facts(result, drawn[stratum.level], member_count)
        for stratum in substructure.strata
    }
    stems = {
        stratum.level: factor_level_stem(substructure.definition, stratum.level)
        for stratum in substructure.strata
    }

    stratum_rows: list[dict[str, Any]] = []
    comparison_prose: list[str] = []
    # Set once, from the generic form of the disclosures every plate carries.
    # The trailing entry is the disclaimer, which stays on every plate and
    # opens the notes file, so it is not restated among the bullets.
    shared_prose = _level_interpretive_lines(
        substructure,
        substructure.strata[0],
        agreement_lines,
        replicate,
        recorded,
        generic_level=True,
    )[:-1]

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.staging.{uuid.uuid4().hex}")
    staging.mkdir()
    notes_staging = figure_only.with_name(
        f".{figure_only.name}.staging.{uuid.uuid4().hex}"
    )
    if separate_notes:
        figure_only.parent.mkdir(parents=True, exist_ok=True)
        notes_staging.mkdir()
    written: list[str] = []
    separated: list[str] = []
    try:
        for stratum in substructure.strata:
            stem = stems[stratum.level]
            ids = drawn[stratum.level]
            identity = _level_identity_lines(
                substructure, stratum, facts[stratum.level], member_count, replicate
            )
            interpretive = _level_interpretive_lines(
                substructure, stratum, agreement_lines, replicate, recorded
            )
            _write_level_figure(
                overlay,
                ids,
                contexts,
                colours,
                staging / f"{stem}.jpeg",
                limits,
                [*identity, *interpretive],
                legend_columns,
            )
            written.append(f"{stem}.jpeg")
            if separate_notes:
                # The plate keeps its identity and its own measurements: a
                # stratum figure carries those in its caption and nowhere else,
                # so stripping them would leave a plate that cannot be read.
                # What moves is the interpretive half.
                figure_name = f"{stem}{_SEPARATED_FIGURE_SUFFIX}"
                _write_level_figure(
                    overlay,
                    ids,
                    contexts,
                    colours,
                    notes_staging / figure_name,
                    limits,
                    [*identity, _separated_footer_line()],
                    legend_columns,
                )
                separated.append(figure_name)
                stratum_rows.append(
                    {
                        "level": substructure.definition.level_title(stratum.level),
                        "figure": figure_name,
                        **facts[stratum.level],
                    }
                )

        if not args.no_level_comparison:
            identity = _comparison_identity_lines(substructure, replicate)
            interpretive = _comparison_interpretive_lines(
                substructure, agreement_lines, replicate, recorded
            )
            _write_level_comparison(
                result,
                overlay,
                substructure,
                drawn,
                member_count,
                contexts,
                colours,
                staging / LEVEL_COMPARISON_FILENAME,
                limits,
                [*identity, *interpretive],
            )
            written.append(LEVEL_COMPARISON_FILENAME)
            if separate_notes:
                figure_name = f"{LEVEL_COMPARISON_STEM}{_SEPARATED_FIGURE_SUFFIX}"
                _write_level_comparison(
                    result,
                    overlay,
                    substructure,
                    drawn,
                    member_count,
                    contexts,
                    colours,
                    notes_staging / figure_name,
                    limits,
                    [*identity, _separated_footer_line()],
                )
                separated.append(figure_name)
                comparison_prose = interpretive[:-1]

        (staging / README_FILENAME).write_text(
            "\n".join(
                _readme_lines(
                    substructure,
                    contexts,
                    colours,
                    drawn,
                    written,
                    season,
                    replicate,
                    recorded,
                )
            ),
            encoding="utf-8",
        )
        written.append(README_FILENAME)

        if separate_notes:
            (notes_staging / SEPARATED_NOTES_FILENAME).write_text(
                "\n".join(
                    _notes_lines(
                        substructure,
                        stratum_rows,
                        shared_prose,
                        comparison_prose,
                        season,
                        replicate,
                        recorded,
                    )
                ),
                encoding="utf-8",
            )
            separated.append(SEPARATED_NOTES_FILENAME)

        carried = _promote(staging, destination)
        carried_separated = (
            _promote(notes_staging, figure_only) if separate_notes else []
        )
    finally:
        for temporary in (staging, notes_staging):
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)

    for folder, names in ((destination, carried), (figure_only, carried_separated)):
        if names:
            noun = "directory" if len(names) == 1 else "directories"
            print(
                f"Carried across {len(names)} unmanaged {noun} under {folder}: "
                f"{', '.join(names)}"
            )
    scope = (
        "all recorded replicates"
        if replicate is None
        else f"recorded Rep={replicate} of {len(recorded)}"
    )
    print(
        f"{SOURCE_NAME} {season}: {len(substructure.strata)} planting-decade strata, "
        f"{member_count} of {substructure.member_count} trajectories drawn "
        f"({scope}), {len(colours)} varieties coloured"
    )
    for stratum in substructure.strata:
        ids = drawn[stratum.level]
        varieties = _varieties_present(ids, contexts, colours)
        print(
            f"  {stratum.level:8s} n={len(ids):4d} of "
            f"{len(stratum.trajectory_ids):4d} varieties={len(varieties):3d} "
            f"connector alpha={_connector_alpha(len(ids)):.2f}"
        )
    print(f"Wrote {len(written)} files under {destination}")
    if separate_notes:
        print(f"Wrote {len(separated)} files under {figure_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
