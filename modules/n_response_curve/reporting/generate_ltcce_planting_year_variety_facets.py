#!/usr/bin/env python3
"""Split each planting-decade plate into one figure per variety, plus one sheet.

`generate_ltcce_planting_year_variety_lines.py` writes
`annotated/lines_by_variety/<decade>.jpeg`: every trajectory of one planting
decade on one axes, each connector in its variety's colour. At 26 varieties
over 240 trajectories that plate answers "is there a varietal pattern here?"
with a tangle. This recipe draws the same decades again, split two ways:

* `<decade>/<variety>.jpeg` — one variety alone on the shared frame, with the
  rest of its decade behind it in light grey for scale.
* `<decade>/side_by_side.jpeg` — those same panels in one image, in planting
  order, so the varieties of a decade can be read against one another.
* `<decade>/side_by_side_by_trajectory_count.jpeg` — the same sheet again with
  the panels ordered by trajectory count, largest first, so the varieties the
  decade actually rests on come first and the four-line ones fall to the end.

Every colour is the one `../lines_by_variety/` gives that variety. The
allocation is *imported* from that recipe rather than copied, so a variety
carries one colour across both products and the two cannot drift apart. That
holds only when both are run with the same `--min-factor-stratum`: the roster
the colours are allocated from is the set of strata that floor admits.

Output goes to a sibling directory of `../lines_by_variety/` inside the season
generator's `annotated/`. That generator replaces `by_season/` as one snapshot
and carries unmanaged whole directories across the swap, so a directory
survives a rebuild and a loose file beside `<decade>.jpeg` would not.

Exploratory diagnostic, deliberately outside the release inventory (ANA-11);
this module is not on the release path. No curve is fitted anywhere here.

Re-run after any run of the season-cluster generator: the strata are recomputed
from the source, so these figures go stale when that partition changes.

Usage:
    conda run -n n_response python \\
      modules/n_response_curve/reporting/generate_ltcce_planting_year_variety_facets.py \\
      --config scriptCONFIG.toml --season DS
"""

from __future__ import annotations

import argparse
import collections
import math
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from n_response_curve.reporting.generate_ltcce_planting_year_variety_lines import (  # noqa: E402
    DEFAULT_REPLICATE,
    README_FILENAME,
    VARIETY_LINES_DIRNAME,
    _AXES_INCHES,
    _LEGEND_FONT_SIZE,
    _OVERLAY_TITLE_WIDTH,
    _UNRECORDED_VARIETY_LABEL,
    _draw_variety_lines,
    _varieties_present,
    _variety_colours,
    _variety_handles,
    _variety_of,
    check_replicate,
    drawn_ids,
    drawn_member_count,
    recorded_replicates,
    replicate_headline,
    replicate_readme_section,
    replicate_scope_lines,
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
    FACTOR_DEFINITIONS,
    FACTOR_PLANTING_YEAR,
    FACTOR_VARIETY,
    MIN_FACTOR_STRATUM,
    FactorStratum,
    FactorSubstructure,
    SeasonClusteringResult,
    build_factor_substructure,
    build_season_clustering,
    cluster_ladder_shares,
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
    read_source_dataset_overlay,
)

SOURCE_NAME = "ltcce"
DEFAULT_SEASON = "DS"
SUPPORTED_SEASONS = ("DS", "EWS", "LWS")
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scriptCONFIG.toml"
CLUSTERS_ROOT = PROJECT_ROOT / "WF/03_Response_Curves/ltcce/clusters"

# A sibling of `lines_by_variety/`, named after it: same colours, same strata,
# same frame — split into one figure per variety instead of overlaid. The name
# deliberately does not shorten to `by_variety/`, which one level up already
# names a different product (`by_planting_year/by_variety/`, whole plates per
# variety across the decades rather than within one).
FACET_DIRNAME = "lines_by_variety_faceted"

SIDE_BY_SIDE_FILENAME = "side_by_side.jpeg"
TRAJECTORY_COUNT_FILENAME = "side_by_side_by_trajectory_count.jpeg"

# The two panel orders, and the line each one puts in its own suptitle. A sheet
# that does not say how it is ordered invites the reader to find meaning in the
# sequence, and on one of these two orders that meaning would be real
# (chronological) while on the other it is only "how much data there is".
_ROSTER_ORDER_LINE = (
    "panels are in the season's colour order — first recorded planting year — "
    "so left to right is roughly the order the varieties entered the trial"
)
_TRAJECTORY_COUNT_ORDER_LINE = (
    "panels are ordered by trajectory count — the number of connecting lines "
    "drawn — largest first; ties keep the planting-year order. That order is "
    "how much data a variety has here and nothing else: it is not a ranking of "
    "the varieties, and it is not chronological"
)

# The rest of the decade, drawn behind the variety. A 4-trajectory variety on
# the season's shared 12x7.5 frame is otherwise mostly empty canvas and says
# nothing about where it sits; this is the only way a per-variety figure keeps
# the comparison the overlaid plate had. Visibly lighter than the 0.45 grey
# `lines_by_variety/` gives an unrecorded variety, so the two greys cannot be
# confused: this one is never a variety.
_BACKDROP_COLOUR = (0.72, 0.72, 0.72)
_BACKDROP_ALPHA = 0.35
_BACKDROP_WIDTH = 0.6
_BACKDROP_LABEL = "the rest of this decade (context; not this variety)"

# One panel size for the whole run, so a panel is the same size in a 7-variety
# decade and a 26-variety one and the sheets can be laid side by side. Column
# count varies; panel inches do not.
_PANEL_INCHES = (4.2, 3.8)
_PANEL_MAX_COLUMNS = 6
_PANEL_TITLE_WIDTH = 52
_PANEL_TITLE_FONT_SIZE = 9
_PANEL_TICK_FONT_SIZE = 8
# The sheet suptitle is folded to the canvas it is actually drawn on, which
# is the column count times the panel width — a fixed character width either
# wraps a short line early on a wide sheet or overruns a narrow one.
_SHEET_TITLE_CHARS_PER_COLUMN = 40
# Vertical inches the sheet's one-row legend needs, title included.
_SHEET_LEGEND_INCHES = 0.62

_COLOUR_SOURCE_LINE = (
    "the connector colour is the one this variety carries in "
    f"`../{VARIETY_LINES_DIRNAME}/`, allocated once across the season, so the "
    "same colour means the same variety in both products"
)


def _suptitle_inches(suptitle: str) -> float:
    """Vertical inches :func:`reserve_suptitle` will take for *suptitle*.

    The same arithmetic that helper does, so the canvas can be grown by what
    the title is about to be given back. Kept in step with it by shape, not by
    import: it reserves a fraction of a figure height it is handed, and there
    is no figure yet at the point this is needed.
    """

    lines = suptitle.count("\n") + 1
    return (lines * _TITLE_FONT_SIZE * 1.2 + 14.0) / 72.0


def _variety_definition() -> Any:
    """The `variety` factor definition, for filename stems only.

    Its `release_year` is what puts a released variety's year in front of its
    filename, so a decade folder lists in release order.
    """

    return FACTOR_DEFINITIONS[FACTOR_VARIETY]


def _variety_members(
    trajectory_ids: Sequence[str],
    contexts: Mapping[str, TrajectoryContext],
) -> dict[str, tuple[str, ...]]:
    """The drawn trajectories of one stratum, split by recorded variety.

    Takes ids rather than the stratum because `--replicate` keeps a quarter of
    it, and the panel count, the panel titles and the trajectory-count ordering
    all have to describe what is on the canvas.
    """

    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for trajectory_id in trajectory_ids:
        grouped[_variety_of(contexts, trajectory_id)].append(trajectory_id)
    return {variety: tuple(ids) for variety, ids in grouped.items()}


def _variety_stems(varieties: Sequence[str]) -> dict[str, str]:
    """One filename stem per variety, fail-closed on a collision.

    Two varieties whose names differ only in punctuation fold to one stem, and
    the second figure would silently overwrite the first — a decade quietly
    short one variety, with nothing on the canvas to say so.
    """

    definition = _variety_definition()
    stems = {variety: factor_level_stem(definition, variety) for variety in varieties}
    if len(set(stems.values())) != len(stems):
        counts = collections.Counter(stems.values())
        clashing = sorted(stem for stem, count in counts.items() if count > 1)
        raise RuntimeError(
            f"variety filename stems collide: {', '.join(clashing)} — "
            "two varieties would write to one file"
        )
    return stems


def _check_treatment_classes(
    parent: SourceDatasetOverlay,
    subset: SourceDatasetOverlay,
    variety: str,
) -> None:
    """Refuse a subset whose treatment classes differ from the parent's.

    The point colours come from the axes property cycle in draw order, so a
    subset missing a class — or carrying them in another order — would give
    that class the other class's colour on this figure alone.
    """

    if subset.summary.treatment_classes != parent.summary.treatment_classes:
        raise RuntimeError(
            f"variety {variety!r} carries treatment classes "
            f"{subset.summary.treatment_classes} against the source's "
            f"{parent.summary.treatment_classes}; point colours would not match"
        )


def _draw_backdrop(
    overlay: SourceDatasetOverlay,
    background_ids: Sequence[str],
    axes: Any,
) -> None:
    """Draw the rest of the stratum as light grey connectors, behind."""

    if not background_ids:
        return
    for trajectory in subset_overlay(overlay, tuple(background_ids)).trajectories:
        axes.plot(
            [observation.n_rate_kg_ha for observation in trajectory.observations],
            [observation.yield_t_ha for observation in trajectory.observations],
            color=_BACKDROP_COLOUR,
            linewidth=_BACKDROP_WIDTH,
            alpha=_BACKDROP_ALPHA,
            zorder=1,
        )


def _backdrop_handle() -> Line2D:
    return Line2D(
        [0],
        [0],
        color=_BACKDROP_COLOUR,
        linewidth=2.2,
        label=_BACKDROP_LABEL,
    )


def _variety_facts(
    result: SeasonClusteringResult,
    variety_ids: Sequence[str],
) -> tuple[str, str, str]:
    """Year range, ladder shares and centroid text for one variety subset."""

    years = cluster_year_range(tuple(variety_ids), result.contexts)
    year_text = f"{years[0]}-{years[1]}" if years else "years unknown"
    ladders = format_shares(
        cluster_ladder_shares(tuple(variety_ids), result.features), limit=2
    )
    centroid = centroid_summary(tuple(variety_ids), result.features)
    centroid_text = (
        f"mean zero-N yield={centroid['yield_at_zero_n_t_ha']:.2f} t/ha; "
        f"mean response={centroid['response_above_zero_n_t_ha']:.2f} t/ha"
    )
    return year_text, ladders, centroid_text


def _variety_identity_lines(
    result: SeasonClusteringResult,
    substructure: FactorSubstructure,
    stratum: FactorStratum,
    stratum_ids: Sequence[str],
    variety: str,
    variety_ids: Sequence[str],
    replicate: str | None,
) -> list[str]:
    """What identifies this panel and what it measures — and nothing else.

    The half a plates-only figure keeps: a per-variety plate carries its own
    numbers in its caption and nowhere else, so stripping them would leave a
    figure that cannot be read at all.
    """

    definition = substructure.definition
    year_text, ladders, centroid_text = _variety_facts(result, variety_ids)
    share = len(variety_ids) / len(stratum_ids) if stratum_ids else 0.0
    return [
        f"source={SOURCE_NAME} — {season_label(substructure.season)}, "
        f"{definition.level_title(stratum.level)} — variety "
        f"{variety}{replicate_headline(replicate)}",
        f"{len(variety_ids)} of this decade's {len(stratum_ids)} drawn "
        f"cluster-eligible trajectories ({100 * share:.0f}%); {year_text}",
        f"applied-N ladders: {ladders}",
        centroid_text,
    ]


def _variety_interpretive_lines(
    substructure: FactorSubstructure,
    stratum: FactorStratum,
    agreement_lines: Sequence[str],
    with_backdrop: bool,
    replicate: str | None,
    recorded: Sequence[str],
    *,
    generic_level: bool = False,
) -> list[str]:
    """The half that moves off the plate and into the notes file.

    The grey backdrop keeps its own entry in the *legend* either way, so the
    plates-only figure still says what the grey is; what moves is the sentence
    about what it is not.

    `generic_level` writes the stratum line to stand for every figure in the
    family rather than for one decade. The notes file sets these disclosures
    once beneath a table of six folders, and a line naming the 1960s under a
    table whose last row is the 2010s is simply wrong. Asking for the generic
    form here — rather than editing the list afterwards by index — keeps the
    two forms one list, so a change to a disclosure reaches the figure and the
    notes together.
    """

    definition = substructure.definition
    lines: list[str] = []
    if with_backdrop:
        lines.append(
            "grey: the rest of the decade, drawn for scale only — it is not "
            "this variety, and it is not a comparison group"
        )
    lines += [
        _COLOUR_SOURCE_LINE,
        (
            "every figure here sits inside a recorded stratum, not a cluster: "
            f"its trajectories are the ones the source's "
            f"{definition.source_column} column assigns to that decade, and "
            "nothing was clustered to produce it"
            if generic_level
            else "recorded stratum, not a cluster: every trajectory here "
            f"carries {definition.source_column}={stratum.level!r} in the "
            "source"
        ),
        *replicate_scope_lines(replicate, recorded),
        definition.identity_caveat,
        *agreement_lines,
        _DISCLAIMER,
    ]
    return lines


def _write_variety_figure(
    overlay: SourceDatasetOverlay,
    stratum_ids: Sequence[str],
    contexts: Mapping[str, TrajectoryContext],
    colours: Mapping[str, tuple[float, ...]],
    variety: str,
    variety_ids: Sequence[str],
    destination: Path,
    limits: SharedAxisLimits,
    title_lines: Sequence[str],
    with_backdrop: bool,
) -> None:
    """One variety of one decade, alone on the season's shared frame.

    The caption arrives assembled, so one function draws both the annotated
    plate and its plates-only twin and the two cannot come out on different
    canvases. The backdrop is drawn from the *drawn* stratum, not the whole
    one: a grey four times denser than the variety in front of it would put
    the replication back on a figure that exists to have it taken out.
    """

    figure = plt.figure(figsize=_AXES_INCHES, constrained_layout=True)
    try:
        axes = figure.add_subplot(1, 1, 1)
        background = [
            trajectory_id
            for trajectory_id in stratum_ids
            if trajectory_id not in set(variety_ids)
        ]
        if with_backdrop:
            _draw_backdrop(overlay, background, axes)
        subset = subset_overlay(overlay, tuple(variety_ids))
        _check_treatment_classes(overlay, subset, variety)
        scatter_handles = _draw_variety_lines(subset, contexts, colours, axes)
        limits.apply(axes)
        axes.set_title(
            _wrap_title_lines(title_lines, width=_OVERLAY_TITLE_WIDTH),
            fontsize=_TITLE_FONT_SIZE,
        )
        handles: list[Any] = [*scatter_handles, *_variety_handles([variety], colours)]
        if with_backdrop and background:
            handles.append(_backdrop_handle())
        axes.legend(
            handles=handles,
            loc="upper left",
            fontsize=_LEGEND_FONT_SIZE,
            frameon=True,
            framealpha=0.85,
            handlelength=2.4,
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


def _by_trajectory_count(
    varieties: Sequence[str],
    members: Mapping[str, tuple[str, ...]],
) -> list[str]:
    """The same varieties, most trajectories first.

    One trajectory is one connecting line, so this is also the panel's drawn
    line count. Ties fall back to the order handed in — the season's colour
    order, i.e. first planting year — so the sheet is reproducible rather than
    dependent on dict order.
    """

    rank = {variety: index for index, variety in enumerate(varieties)}
    return sorted(
        varieties,
        key=lambda variety: (-len(members[variety]), rank[variety]),
    )


def _sheet_identity_lines(
    substructure: FactorSubstructure,
    stratum: FactorStratum,
    stratum_ids: Sequence[str],
    varieties: Sequence[str],
    order_line: str,
    replicate: str | None,
) -> list[str]:
    """What the sheet is, what is on it, and how its panels are ordered.

    The order line is identity, not interpretation, and stays on the plate: a
    sheet that does not say how it is ordered invites the reader to find
    meaning in the sequence, and on one of the two orders that meaning is real.
    """

    definition = substructure.definition
    return [
        f"source={SOURCE_NAME} — {season_label(substructure.season)}, "
        f"{definition.level_title(stratum.level)}: "
        f"{len(varieties)} varieties side by "
        f"side{replicate_headline(replicate)}",
        f"the {len(stratum_ids)} drawn cluster-eligible trajectories of this "
        "decade, split by the variety the source records; the panels share "
        "both axes",
        order_line,
    ]


def _sheet_interpretive_lines(
    substructure: FactorSubstructure,
    agreement_lines: Sequence[str],
    replicate: str | None,
    recorded: Sequence[str],
) -> list[str]:
    """The sheet's half that moves into the notes file."""

    return [
        _COLOUR_SOURCE_LINE,
        "a variety here is not a treatment: it arrives with the year it was "
        "grown, the plot design of the day and that year's applied-N ladder, "
        "so a difference between panels is a difference between experiments",
        *replicate_scope_lines(replicate, recorded),
        substructure.definition.identity_caveat,
        *agreement_lines,
        _DISCLAIMER,
    ]


def _write_side_by_side(
    result: SeasonClusteringResult,
    overlay: SourceDatasetOverlay,
    stratum_ids: Sequence[str],
    contexts: Mapping[str, TrajectoryContext],
    colours: Mapping[str, tuple[float, ...]],
    members: Mapping[str, tuple[str, ...]],
    varieties: Sequence[str],
    destination: Path,
    limits: SharedAxisLimits,
    suptitle_lines: Sequence[str],
    with_backdrop: bool,
) -> None:
    """Every variety of one decade as its own panel, in one image.

    The panels share both axes, so a difference in where a variety's points sit
    is a difference in the data and not in the frame. Variety identity is the
    panel title: a legend naming 26 colours under 26 titled panels would be
    noise, so the figure legend carries only what a title cannot say.
    """

    if not varieties:
        return
    columns = min(_PANEL_MAX_COLUMNS, len(varieties))
    rows = math.ceil(len(varieties) / columns)
    panel_width, panel_height = _PANEL_INCHES

    suptitle = _wrap_title_lines(
        suptitle_lines,
        width=_SHEET_TITLE_CHARS_PER_COLUMN * columns,
    )
    # The suptitle and the legend are both reserved out of the canvas, so the
    # canvas is grown by what they take. Otherwise a two-row sheet — where the
    # title is the same height and the panels are half as many — comes out with
    # visibly squatter panels than a five-row one, and these sheets exist to be
    # compared.
    title_inches = _suptitle_inches(suptitle)

    figure = plt.figure(
        figsize=(
            panel_width * columns,
            panel_height * rows + _SHEET_LEGEND_INCHES + title_inches,
        ),
        constrained_layout=True,
    )
    try:
        axes_grid = figure.subplots(
            rows, columns, sharex=True, sharey=True, squeeze=False
        )
        flat = [
            axes_grid[index // columns][index % columns]
            for index in range(rows * columns)
        ]
        for axes in flat[len(varieties):]:
            axes.set_axis_off()

        scatter_handles: list[Any] = []
        for index, variety in enumerate(varieties):
            axes = flat[index]
            variety_ids = members[variety]
            background = [
                trajectory_id
                for trajectory_id in stratum_ids
                if trajectory_id not in set(variety_ids)
            ]
            if with_backdrop:
                _draw_backdrop(overlay, background, axes)
            subset = subset_overlay(overlay, variety_ids)
            _check_treatment_classes(overlay, subset, variety)
            handles = _draw_variety_lines(subset, contexts, colours, axes)
            if not scatter_handles:
                scatter_handles = handles
            limits.apply(axes)
            year_text, ladders, centroid_text = _variety_facts(result, variety_ids)
            axes.set_title(
                _wrap_title_lines(
                    [
                        variety,
                        f"{len(variety_ids)} trajectories; {year_text}",
                        f"ladders: {ladders}",
                        centroid_text,
                    ],
                    width=_PANEL_TITLE_WIDTH,
                ),
                fontsize=_PANEL_TITLE_FONT_SIZE,
            )
            axes.tick_params(labelsize=_PANEL_TICK_FONT_SIZE)
            # Shared axes hide the inner tick labels but not the axis labels,
            # and `_draw_variety_lines` sets both on every panel it draws.
            if index % columns != 0:
                axes.set_ylabel("")
            if index + columns < len(varieties):
                axes.set_xlabel("")
            else:
                # `sharex` hides the tick labels of every row but the grid's
                # last, and the last row is short whenever the variety count
                # is not a multiple of the column count. Without this the
                # bottom panel of a full column carries an axis label over an
                # unnumbered axis.
                axes.tick_params(labelbottom=True)

        legend_handles: list[Any] = list(scatter_handles)
        if with_backdrop:
            legend_handles.append(_backdrop_handle())
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            fontsize=_LEGEND_FONT_SIZE,
            title=(
                "every panel is one variety of this decade, on the colour it "
                f"carries in `../{VARIETY_LINES_DIRNAME}/`, on one shared frame"
            ),
            title_fontsize=_LEGEND_FONT_SIZE,
            frameon=False,
            handlelength=2.4,
        )

        _reserve_suptitle(
            figure,
            suptitle,
            legend_strip=_SHEET_LEGEND_INCHES / figure.get_figheight(),
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


def _readme_lines(
    substructure: FactorSubstructure,
    contexts: Mapping[str, TrajectoryContext],
    colours: Mapping[str, tuple[float, ...]],
    drawn: Mapping[str, Sequence[str]],
    counts: Mapping[str, int],
    season: str,
    figure_count: int,
    minimum_stratum: int,
    with_backdrop: bool,
    replicate: str | None,
    recorded: Sequence[str],
) -> list[str]:
    adjective = season_label(season)
    lines = [
        f"# {adjective} — each planting decade split into one figure per variety",
        "",
        f"`../{VARIETY_LINES_DIRNAME}/<decade>.jpeg` draws a whole decade on one "
        "axes with every connector in its variety's colour. At "
        f"{max(counts.values(), default=0)} varieties over "
        f"{max((len(ids) for ids in drawn.values()), default=0)} trajectories "
        "the busiest of those plates shows the tangle and little else. Here "
        "the same decades are drawn again, split:",
        "",
        "- `<decade>/<variety>.jpeg` — one variety alone on the season's shared "
        "frame.",
        f"- `<decade>/{SIDE_BY_SIDE_FILENAME}` — those same panels in one image, "
        "in planting order, so the varieties of a decade can be read against "
        "one another.",
        f"- `<decade>/{TRAJECTORY_COUNT_FILENAME}` — the same sheet with the "
        "panels ordered by trajectory count, largest first. One trajectory is "
        "one connecting line, so this puts the varieties the decade actually "
        "rests on first and drops the four-line ones to the end. Ties keep the "
        "planting order, so the two sheets differ only where the counts do.",
        "",
        "## The colours are the other product's colours",
        "",
        f"All {len(colours)} varieties this season records among its "
        "cluster-eligible trajectories are allocated a colour once, in order of "
        f"first recorded planting year, by `../{VARIETY_LINES_DIRNAME}/`'s own "
        "allocator — imported here, not copied. A variety therefore carries one "
        "colour across both products, and the same colour in every decade it "
        "appears in.",
        "",
        "That holds only while both recipes are run with the same "
        f"`--min-factor-stratum` (this run: {minimum_stratum}). The floor decides "
        "which strata exist, the strata decide the roster, and the roster is what "
        "the colours are allocated over. `--replicate` does not enter into it — "
        "it filters after the allocation — but the two products only draw the "
        "same lines when they are given the same value, so run them together.",
        "",
        *replicate_readme_section(replicate, recorded),
    ]
    if with_backdrop:
        lines += [
            "## The grey behind each variety",
            "",
            "The rest of the decade is drawn behind the variety in light grey. A "
            "four-trajectory variety on the season's shared frame is otherwise "
            "mostly empty canvas, and the whole point of a shared frame is to "
            "show where something sits. The grey is context, not a comparison "
            "group, and it is a lighter grey than the one "
            f"`../{VARIETY_LINES_DIRNAME}/` gives a variety the source does not "
            "name — that grey is a variety, this one never is. `--no-backdrop` "
            "drops it.",
            "",
        ]
    lines += ["## What is here", ""]
    for stratum in substructure.strata:
        stem = factor_level_stem(substructure.definition, stratum.level)
        ids = drawn[stratum.level]
        varieties = _varieties_present(ids, contexts, colours)
        lines.append(
            f"- `{stem}/` — {counts.get(stem, 0)} varieties over "
            f"{len(ids)} drawn trajectories of the stratum's "
            f"{len(stratum.trajectory_ids)}, one figure each plus "
            f"the two `side_by_side*` sheets."
            + (
                " Includes trajectories whose variety the source does not record."
                if _UNRECORDED_VARIETY_LABEL in varieties
                else ""
            )
        )
    lines += [
        "",
        "A released variety's filename carries its release year in front, so a "
        "decade folder lists in release order — which for this experiment is "
        "close to calendar order, and that is the confounding the caveat on every "
        "figure is about. A filename with no year is a breeding-line designation "
        "with no release year on record.",
        "",
        "## The plates-only twin",
        "",
        "Every figure here also exists under "
        f"`../../{PLANTING_YEAR_FIGURE_ONLY_DIRNAME}/{FACET_DIRNAME}/` as "
        f"`<decade>/<stem>{_SEPARATED_FIGURE_SUFFIX}`: the same canvas with the "
        "interpretive half of the caption lifted off and set in "
        f"`{SEPARATED_NOTES_FILENAME}` at the root of that folder. Use these "
        "annotated figures when one has to travel alone, and that folder when "
        "the prose belongs in the surrounding document. `--no-separate-notes` "
        "suppresses the whole folder.",
        "",
        "## Interpretation boundary",
        "",
        "Exploratory diagnostic, outside the governed analysis inventory "
        "(ANA-11). No curve is fitted. Every point is an observed yield at an "
        "observed applied-N rate, and a connector says only which points belong "
        "to one replicate. Splitting by variety does not isolate the variety: "
        "each one arrives with the years it was grown, the plot design of those "
        "years and their applied-N ladder.",
        "",
        "## Regenerating",
        "",
        "```bash",
        "conda run -n n_response python \\",
        "  modules/n_response_curve/reporting/"
        "generate_ltcce_planting_year_variety_facets.py \\",
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
        "survives that swap because it is a directory.",
        "",
        "The panel order is stated on every sheet. A sheet that does not say how "
        "it is ordered invites a reader to find meaning in the sequence, and on "
        "one of these two orders that meaning is real — planting order is "
        "roughly chronological — while on the other the sequence says only how "
        "much data a variety has.",
        "",
        "Because a decade is itself a directory here, a decade folder from an "
        "earlier run that is no longer a stratum is *carried across* rather than "
        "removed. The run prints every directory it carries; a carried decade is "
        "stale output, and deleting it is the operator's call.",
        "",
        f"Figures written: {figure_count}.",
        "",
    ]
    return lines


def _notes_lines(
    substructure: FactorSubstructure,
    decade_rows: Sequence[Mapping[str, Any]],
    shared_prose: Sequence[str],
    season: str,
    replicate: str | None,
    recorded: Sequence[str],
    with_backdrop: bool,
) -> list[str]:
    """The prose the `*_figure.jpeg` figures in `figure_only/` no longer carry.

    One file at the root rather than one per decade folder: every figure in
    every decade carries the same disclosures and differs only in its numbers,
    and six near-identical files would be six things to keep in step instead of
    one. The numbers stay on the figures, where they belong; what is here is
    the table that lets a reader find a figure, and the reading it needs.

    Built from the same line lists the annotated captions are built from — see
    :func:`_variety_interpretive_lines` and :func:`_sheet_interpretive_lines` —
    so a figure and its notes cannot disagree.
    """

    annotated_root = f"../../{PLANTING_YEAR_ANNOTATED_DIRNAME}/{FACET_DIRNAME}"
    lines = [
        f"# {season_label(season)} — each planting decade split into one figure "
        "per variety: notes for the separated figures",
        "",
        f"{_DISCLAIMER[0].upper()}{_DISCLAIMER[1:]}.",
        "",
        f"These notes belong to every `*{_SEPARATED_FIGURE_SUFFIX}` under this "
        "folder's decade directories. Each of those carries its identity, its "
        "own measurements and its legend and nothing else; everything about "
        "how to read them, and what not to conclude from them, is here. Every "
        f"figure also exists under `{annotated_root}/` in an annotated form "
        "that sets the same text on its own face, for when a figure has to "
        "travel alone.",
        "",
        "## What is in each decade folder",
        "",
        f"- `<variety>{_SEPARATED_FIGURE_SUFFIX}` — one variety alone on the "
        "season's shared frame.",
        f"- `{Path(SIDE_BY_SIDE_FILENAME).stem}{_SEPARATED_FIGURE_SUFFIX}` — "
        "those same panels in one image, in planting order.",
        f"- `{Path(TRAJECTORY_COUNT_FILENAME).stem}"
        f"{_SEPARATED_FIGURE_SUFFIX}` — the same sheet ordered by trajectory "
        "count, largest first; ties keep the planting order. Each sheet says "
        "on its own face which order it is in, because on one of the two that "
        "order carries meaning and on the other it carries none.",
        "",
        *replicate_readme_section(replicate, recorded),
    ]

    if with_backdrop:
        lines += [
            "## The grey behind each variety",
            "",
            "The rest of the decade is drawn behind the variety in light grey, "
            "and the legend on every figure names it. It is context, not a "
            "comparison group, and it is a lighter grey than the one "
            f"`../{VARIETY_LINES_DIRNAME}/` gives a variety the source does "
            "not name — that grey is a variety, this one never is. It is drawn "
            "from the same replicate as the variety in front of it.",
            "",
        ]

    if decade_rows:
        lines += [
            f"## The {len(decade_rows)} decade folders",
            "",
            "| Folder | Level | Varieties | Trajectories drawn | Years | "
            "Figures |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in decade_rows:
            lines.append(
                f"| `{row['folder']}/` | {row['level']} | {row['varieties']} | "
                f"{row['count']} of the stratum's {row['stratum_count']} | "
                f"{row['years']} | {row['figures']} |"
            )
        lines.append("")
        if shared_prose:
            lines += ["### True of every figure in every folder", ""]
            lines += [f"- {entry[0].upper()}{entry[1:]}" for entry in shared_prose]
            lines.append("")

    lines += [
        "A released variety's filename carries its release year in front, so a "
        "decade folder lists in release order — which for this experiment is "
        "close to calendar order, and that is the confounding the caveat on "
        "every figure is about. A filename with no year is a breeding-line "
        "designation with no release year on record.",
        "",
        "## Regenerating",
        "",
        "Written by `generate_ltcce_planting_year_variety_facets.py` as the "
        f"`{PLANTING_YEAR_FIGURE_ONLY_DIRNAME}/` companion to "
        f"`{annotated_root}/`, in the same run; `--no-separate-notes` "
        "suppresses this file and every figure beside it.",
        "",
        "```bash",
        "conda run -n n_response python \\",
        "  modules/n_response_curve/reporting/"
        "generate_ltcce_planting_year_variety_facets.py \\",
        f"  --config scriptCONFIG.toml --season {season}"
        + ("" if replicate is None else f" --replicate {replicate}"),
        "```",
        "",
        "Because a decade is itself a directory here, a decade folder from an "
        "earlier run that is no longer a stratum is *carried across* rather "
        "than removed. The run prints every directory it carries; a carried "
        "decade is stale output, and deleting it is the operator's call.",
        "",
    ]
    return lines


def default_output_dir(season: str) -> Path:
    """The faceted folder beside that season's `lines_by_variety/`."""

    return (
        CLUSTERS_ROOT
        / "by_season"
        / _season_token(season)
        / "by_planting_year"
        / PLANTING_YEAR_ANNOTATED_DIRNAME
        / FACET_DIRNAME
    )


def figure_only_output_dir(season: str) -> Path:
    """The plates-only twin, under the season's `figure_only/` tier.

    A sibling of `figure_only/lines_by_variety/` and of the season generator's
    own separated plates: `figure_only/` is a presentation tier, and a reader
    who wants figures without prose wants all of them in one place.
    """

    return (
        CLUSTERS_ROOT
        / "by_season"
        / _season_token(season)
        / "by_planting_year"
        / PLANTING_YEAR_FIGURE_ONLY_DIRNAME
        / FACET_DIRNAME
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--season",
        default=DEFAULT_SEASON,
        choices=SUPPORTED_SEASONS,
        help=f"Season whose planting decades are split (default: {DEFAULT_SEASON})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination for the annotated figures (default: that season's "
        f"annotated/{FACET_DIRNAME}/)",
    )
    parser.add_argument(
        "--figure-only-dir",
        type=Path,
        default=None,
        help="Destination for the plates-only twin (default: that season's "
        f"{PLANTING_YEAR_FIGURE_ONLY_DIRNAME}/{FACET_DIRNAME}/)",
    )
    parser.add_argument(
        "--min-factor-stratum",
        type=int,
        default=MIN_FACTOR_STRATUM,
        help="Smallest planting decade split into per-variety figures. Must "
        f"match the value `{VARIETY_LINES_DIRNAME}/` was written with, or the "
        "two products allocate colours over different rosters",
    )
    parser.add_argument(
        "--no-backdrop",
        action="store_true",
        help="Draw each variety alone, without the rest of its decade in grey",
    )
    parser.add_argument(
        "--replicate",
        default=DEFAULT_REPLICATE,
        help="Draw only the trajectories the source records under this Rep "
        f"value (default: {DEFAULT_REPLICATE}). Must match the value "
        f"`{VARIETY_LINES_DIRNAME}/` was written with, or the two products "
        "draw different lines for the same variety",
    )
    parser.add_argument(
        "--all-replicates",
        action="store_true",
        help="Draw every recorded replicate, as this recipe originally did",
    )
    parser.add_argument(
        "--no-side-by-side",
        action="store_true",
        help="Write the per-variety figures only, without the per-decade sheet",
    )
    parser.add_argument(
        "--no-separate-notes",
        action="store_true",
        help="Write the annotated figures only, without the plates-only twin "
        f"and its {SEPARATED_NOTES_FILENAME}",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    season = args.season.strip().upper()
    destination = (args.output_dir or default_output_dir(season)).resolve()
    figure_only = (args.figure_only_dir or figure_only_output_dir(season)).resolve()
    with_backdrop = not args.no_backdrop
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
    drawn = {
        stratum.level: drawn_ids(stratum.trajectory_ids, contexts, replicate)
        for stratum in substructure.strata
    }
    member_count = drawn_member_count(substructure, contexts, replicate)

    # Pinned from the parent overlay before any subsetting, exactly as the
    # season generator and `lines_by_variety/` do, so every panel here shares
    # that frame. The colours come from the whole season for the same reason
    # they do there: a variety keeps one colour whatever is filtered out.
    limits = shared_axis_limits(overlay)
    colours = _variety_colours(substructure, contexts)
    agreement_lines = _factor_agreement_lines(substructure)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.staging.{uuid.uuid4().hex}")
    staging.mkdir()
    notes_staging = figure_only.with_name(
        f".{figure_only.name}.staging.{uuid.uuid4().hex}"
    )
    if separate_notes:
        figure_only.parent.mkdir(parents=True, exist_ok=True)
        notes_staging.mkdir()
    figure_count = 0
    separated_count = 0
    counts: dict[str, int] = {}
    reported: list[tuple[str, int, int, int]] = []
    decade_rows: list[dict[str, Any]] = []
    # Set once, from the generic form of the disclosures every figure carries.
    # `with_backdrop=False` because the notes file gives the grey its own
    # section; the trailing entry is the disclaimer, which stays on every
    # figure and opens the notes file.
    shared_prose = _variety_interpretive_lines(
        substructure,
        substructure.strata[0],
        agreement_lines,
        False,
        replicate,
        recorded,
        generic_level=True,
    )[:-1]
    try:
        for stratum in substructure.strata:
            decade_stem = factor_level_stem(substructure.definition, stratum.level)
            decade_dir = staging / decade_stem
            decade_dir.mkdir()
            separated_dir = notes_staging / decade_stem
            if separate_notes:
                separated_dir.mkdir()
            stratum_ids = drawn[stratum.level]
            members = _variety_members(stratum_ids, contexts)
            varieties = _varieties_present(stratum_ids, contexts, colours)
            stems = _variety_stems(varieties)
            decade_figures = 0
            for variety in varieties:
                identity = _variety_identity_lines(
                    result,
                    substructure,
                    stratum,
                    stratum_ids,
                    variety,
                    members[variety],
                    replicate,
                )
                # The backdrop caveat belongs on a figure only when there is a
                # backdrop to caveat, which a variety that *is* its whole
                # decade does not have.
                background = set(stratum_ids) - set(members[variety])
                interpretive = _variety_interpretive_lines(
                    substructure,
                    stratum,
                    agreement_lines,
                    with_backdrop and bool(background),
                    replicate,
                    recorded,
                )
                _write_variety_figure(
                    overlay,
                    stratum_ids,
                    contexts,
                    colours,
                    variety,
                    members[variety],
                    decade_dir / f"{stems[variety]}.jpeg",
                    limits,
                    [*identity, *interpretive],
                    with_backdrop,
                )
                figure_count += 1
                decade_figures += 1
                if separate_notes:
                    _write_variety_figure(
                        overlay,
                        stratum_ids,
                        contexts,
                        colours,
                        variety,
                        members[variety],
                        separated_dir
                        / f"{stems[variety]}{_SEPARATED_FIGURE_SUFFIX}",
                        limits,
                        [*identity, _separated_footer_line()],
                        with_backdrop,
                    )
                    separated_count += 1

            if not args.no_side_by_side:
                sheets = (
                    (SIDE_BY_SIDE_FILENAME, varieties, _ROSTER_ORDER_LINE),
                    (
                        TRAJECTORY_COUNT_FILENAME,
                        _by_trajectory_count(varieties, members),
                        _TRAJECTORY_COUNT_ORDER_LINE,
                    ),
                )
                for filename, ordered, order_line in sheets:
                    identity = _sheet_identity_lines(
                        substructure,
                        stratum,
                        stratum_ids,
                        ordered,
                        order_line,
                        replicate,
                    )
                    interpretive = _sheet_interpretive_lines(
                        substructure, agreement_lines, replicate, recorded
                    )
                    _write_side_by_side(
                        result,
                        overlay,
                        stratum_ids,
                        contexts,
                        colours,
                        members,
                        ordered,
                        decade_dir / filename,
                        limits,
                        [*identity, *interpretive],
                        with_backdrop,
                    )
                    figure_count += 1
                    decade_figures += 1
                    if separate_notes:
                        _write_side_by_side(
                            result,
                            overlay,
                            stratum_ids,
                            contexts,
                            colours,
                            members,
                            ordered,
                            separated_dir
                            / f"{Path(filename).stem}"
                            f"{_SEPARATED_FIGURE_SUFFIX}",
                            limits,
                            [*identity, _separated_footer_line()],
                            with_backdrop,
                        )
                        separated_count += 1

            counts[decade_stem] = len(varieties)
            years = cluster_year_range(stratum_ids, result.contexts)
            decade_rows.append(
                {
                    "folder": decade_stem,
                    "level": substructure.definition.level_title(stratum.level),
                    "varieties": len(varieties),
                    "count": len(stratum_ids),
                    "stratum_count": len(stratum.trajectory_ids),
                    "years": (
                        f"{years[0]}-{years[1]}" if years else "years unknown"
                    ),
                    "figures": decade_figures,
                }
            )
            reported.append(
                (
                    stratum.level,
                    len(stratum_ids),
                    len(stratum.trajectory_ids),
                    len(varieties),
                )
            )

        (staging / README_FILENAME).write_text(
            "\n".join(
                _readme_lines(
                    substructure,
                    contexts,
                    colours,
                    drawn,
                    counts,
                    season,
                    figure_count,
                    args.min_factor_stratum,
                    with_backdrop,
                    replicate,
                    recorded,
                )
            ),
            encoding="utf-8",
        )

        if separate_notes:
            (notes_staging / SEPARATED_NOTES_FILENAME).write_text(
                "\n".join(
                    _notes_lines(
                        substructure,
                        decade_rows,
                        shared_prose,
                        season,
                        replicate,
                        recorded,
                        with_backdrop,
                    )
                ),
                encoding="utf-8",
            )
            separated_count += 1

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
                f"{', '.join(names)} — a decade folder that is no longer a "
                "stratum is stale output"
            )
    scope = (
        "all recorded replicates"
        if replicate is None
        else f"recorded Rep={replicate} of {len(recorded)}"
    )
    print(
        f"{SOURCE_NAME} {season}: {len(substructure.strata)} planting-decade "
        f"strata, {member_count} of {substructure.member_count} trajectories "
        f"drawn ({scope}), {len(colours)} varieties on the season's colours"
    )
    for level, drawn_count, stratum_count, variety_count in reported:
        print(
            f"  {level:8s} n={drawn_count:4d} of {stratum_count:4d} "
            f"varieties={variety_count:3d} "
            f"figures={variety_count + (0 if args.no_side_by_side else 2):3d}"
        )
    print(f"Wrote {figure_count} figures and 1 README under {destination}")
    if separate_notes:
        print(f"Wrote {separated_count} files under {figure_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
