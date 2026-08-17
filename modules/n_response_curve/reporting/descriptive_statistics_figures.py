"""Figure builders for the descriptive-statistics companion recipe.

One builder per name declared in ``contracts.FIGURE_SPECS``; ``build_figure``
dispatches through :data:`FIGURE_BUILDERS`. Builders return a rendered
``plt.Figure`` or ``None`` — they never save, close, or otherwise touch the
filesystem, and they never emit a blank "no data" panel, because the caller
records a skip and a blank panel would land in the bundle as if it carried
evidence.

Where the inputs come from, and why the split is not arbitrary:

* Anything a sibling analysis module already computed is read from ``tables``.
  The bundle's figures render the bundle's tables, so a figure must show a
  sibling's numbers — including a sibling's mistake — rather than quietly
  recomputing its own.
* Only the row-level cloud is rebuilt from ``loaded`` through the frozen
  ``build_all_observations``: a histogram, a scatter, and a per-series level
  count cannot be reconstructed from summary rows at all.

Governance: restricted datasets appear by ``source_name`` only. Suppressed
columns are blanked upstream and are additionally excluded here by cross-check
against ``ColumnSpec.suppressed``, and no categorical level below
``config.minimum_level_count`` is drawn or labeled. Nothing here fits a model,
ranks a dataset, or draws a trend line.
"""

from __future__ import annotations

from dataclasses import dataclass
import textwrap
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..analysis.descriptive_statistics.config import DescriptiveStatisticsConfig
from ..analysis.descriptive_statistics.contracts import (
    FIGURE_SPECS,
    ProfileContractError,
)
from ..analysis.descriptive_statistics.sources import (
    ColumnBinding,
    LoadedSources,
    ProfiledSource,
    build_all_observations,
)


# The sibling grain-yield bundle's vocabulary (Tableau/Vega-10), reused so the
# two bundles read as one project. Sources take colours in profiling order, so
# a dataset keeps one colour across every figure of a run.
_SOURCE_PALETTE: tuple[str, ...] = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
)
_NEUTRAL = "#9D9D9D"
_WITHHELD_GREY = "#BFBFBF"

def _shade_major(name: str, families: int) -> tuple[tuple[float, ...], ...]:
    """A 4x5 qualitative colormap reordered so neighbours differ in hue.

    ``tab20b`` and ``tab20c`` are laid out as four hue families of five shades,
    so consecutive entries are five shades of one colour. A context field takes
    consecutive entries, which would render a five-level field as one hue
    fading. Taking one family at a time within a shade gives the field four
    distinct hues before it reuses one at a different lightness.
    """

    colours = plt.get_cmap(name).colors
    return tuple(
        tuple(colours[family * 5 + shade])
        for shade in range(5)
        for family in range(families)
    )


# Colour vocabulary for the levels of a categorical context field: tab20 without
# its two greys, which are reserved for the withheld and pooled segments (a solid
# grey level segment beside a hatched grey withheld segment invites reading a
# recorded level as a disclosure control), then tab20b and tab20c — the latter
# without its fourth family, which is also grey.
#
# Sized against one bar, not against the run. Colours are allocated once across
# every field and dataset so a level keeps its colour between figures, and the
# recipe's bindings now allocate well past these 67 entries, so the cycle wraps
# and two levels of two different fields can share a colour. That is accepted:
# the level text inside the segment is the identifier, and no figure asks a
# reader to match a colour across fields.
#
# What must hold is that no two levels of the *same* bar share a colour, since
# there the colour is what separates one segment from the next. It holds by
# construction while every drawn field has at most this many levels: allocation
# is consecutive, so a field's levels take distinct entries even across a wrap.
# A field with more levels than this is coloured from the ramp below instead —
# see ``_composition_level_colours``. Keep the vocabulary at least as large as
# the widest field drawn in the categorical form.
_LEVEL_PALETTE: tuple[tuple[float, ...], ...] = (
    tuple(
        tuple(colour)
        for index, colour in enumerate(plt.get_cmap("tab20").colors)
        if index not in {14, 15}
    )
    + _shade_major("tab20b", 4)
    + _shade_major("tab20c", 3)
    # Dark2 and Set2 close on a grey apiece, dropped for the same reason.
    + tuple(tuple(colour) for colour in plt.get_cmap("Dark2").colors[:7])
    + tuple(tuple(colour) for colour in plt.get_cmap("Set2").colors[:7])
)

# Continuous ramp for a bar with more levels than the vocabulary above holds.
# Perceptually uniform and monotone in lightness, so a long run of narrow
# segments reads as a gradient rather than as noise, and adjacent levels are
# never identical even where they are close. The dark-violet end is cropped:
# every sampled colour has enough contrast for the black inline label used by
# the one wide segment in LTCCE's variety row.
_OVERSIZED_FIELD_COLORMAP = "viridis"
_OVERSIZED_FIELD_COLORMAP_RANGE = (0.45, 0.95)


def _oversized_field_colours(level_count: int) -> tuple[tuple[float, ...], ...]:
    """Sample the readable part of the oversized-field colour ramp."""

    if level_count <= 0:
        return ()
    ramp = plt.get_cmap(_OVERSIZED_FIELD_COLORMAP)
    lower, upper = _OVERSIZED_FIELD_COLORMAP_RANGE
    return tuple(
        tuple(ramp(position))
        for position in np.linspace(lower, upper, num=level_count)
    )

_ANNOTATION_BOX = {"boxstyle": "round", "facecolor": "white", "alpha": 0.88}

# Agronomically bound quantities profiled by ``numeric_spread_overview``. The
# ``year`` binding is deliberately absent: it is a temporal coordinate, and its
# standardized "spread" would describe the calendar, not a measurement.
_SPREAD_QUANTITIES: tuple[tuple[str, str], ...] = (
    ("nitrogen_rate", "Inorganic N rate"),
    ("yield_t_ha", "Grain yield (t ha⁻¹)"),
    ("yield_kg_ha", "Grain yield (kg ha⁻¹)"),
    ("zero_n_yield_t_ha", "Zero-N check yield"),
)

# ``context_composition`` carries every declared context field. Three figures
# divide them, because one stacked bar cannot carry all of them legibly — the
# split is by how many levels a field is recorded at, not by what it means.
#
# The first two figures share the stacked-share form and differ only in which
# fields they draw: the growing-condition pair that every dataset records, and
# the location/management fields that only some do. ``season_coded`` is excluded
# from both on purpose — it is ph_combined_nopt_rcm's second, numeric encoding
# of the same season field and would double-count that dataset.
_COMPOSITION_CONTEXTS: tuple[tuple[str, str], ...] = (
    ("season", "Season"),
    ("water_regime", "Water regime"),
)
_SITE_MANAGEMENT_CONTEXTS: tuple[tuple[str, str], ...] = (
    ("region", "Region"),
    ("site", "Site"),
    ("establishment", "Establishment"),
)

# Variety is the one remaining field, and it is not drawable in that form: it is
# recorded at 69 and 90 distinct levels, so no level reaches the width a segment
# label needs and a stacked bar would render as an unreadable band of colour. It
# gets its own figure, in a per-dataset top-N form.
_VARIETY_CONTEXT = "variety"

# Context fields the agronomic profile derives rather than reads from a bound
# column, as (key, title, caption). They are not in the two tuples above because
# no cross-dataset figure draws them, but every dataset that records the
# underlying quantity has them, so the per-dataset figures pick them up on their
# own. The caption travels with the field because the standing one promises the
# levels are reported as recorded, which is true of every other field here and
# false of these.
_DERIVED_CONTEXTS: tuple[tuple[str, str, str], ...] = (
    (
        "applied_n_band",
        "Applied N (kg ha⁻¹)",
        "The applied-N bands are derived, not recorded: the rate on each row is "
        "banded in {width:g} kg N ha⁻¹ steps, with zero held out as its own "
        "band because it is the zero-N check arm and not the bottom of the "
        "first one.",
    ),
    (
        "year_band",
        "Year",
        "The year bands are derived, not recorded: the recorded year is banded "
        "in {span:d}-year steps aligned to multiples of {span:d}, so two "
        "datasets that overlap in time carry the same bands. Rows with no "
        "recorded year are not banded and fall in the residual.",
    ),
)


def _derivation_parameters(config: DescriptiveStatisticsConfig) -> dict[str, Any]:
    """Values the derived-field captions above interpolate.

    One mapping for every caption rather than one argument per field: a caption
    takes what it names and ignores the rest, so adding a derived field does not
    reach back into the call sites that format them.
    """

    return {
        "width": config.applied_n_band_width_kg_ha,
        "span": config.year_band_span_years,
    }

# The per-dataset figures take their fields from each source's own bindings
# rather than from the curated tuples above, so a context column bound in the
# recipe cannot go undrawn just because nobody added it to a list here. Titles
# come from the tuples where they exist; the rest are derived.
_CONTEXT_TITLES: Mapping[str, str] = dict(
    _COMPOSITION_CONTEXTS
    + _SITE_MANAGEMENT_CONTEXTS
    + tuple((key, title) for key, title, _ in _DERIVED_CONTEXTS)
)

# Excluded from every stacked figure by name, not by measurement: this is
# ph_combined_nopt_rcm's second, numeric encoding of a field it already records
# in words, so drawing both would show one dataset's season twice.
_DUPLICATE_ENCODING_CONTEXTS = frozenset({"season_coded"})

# The governed core-trial modifier panel ranks planting year ahead of every
# applied-N ladder summary. Those are the only ranked quantities that map onto
# rows in this composition form: zero-N yield is an outcome, P/K recording has
# no bound context row, and the remaining ranked quantities are all summaries
# of applied N. Keep every unranked context in recipe order after these two.
# This is source-specific because the ranking is for the governed core-trial
# population, not for either restricted source dataset.
_DATASET_CONTEXT_PRIORITY: Mapping[str, tuple[str, ...]] = {
    "core_trial_data": ("year_band", "applied_n_band"),
}

# A field recorded at fewer levels than this is routed out of a single-dataset
# composition figure. One level means one full-width block: on a denominator
# that is the dataset itself, a constant has no composition to show, and the row
# it spends says only that the column is constant. The caption says so instead,
# and names the level — so the scope statement the binding was for survives
# without costing a bar. The cross-dataset figures deliberately do not apply
# this: there each row is a different dataset, so a full-width bar beside a
# split one is the comparison the figure exists to make.
_MINIMUM_STACKED_LEVELS = 2

# A field recorded at more levels than this is routed out of the stacked form.
# Measured rather than named, so a future high-cardinality binding is handled
# without an edit here; the per-dataset caption names whatever it excluded, so
# the omission is visible in the figure rather than silent. Region, at 13
# reportable levels, is the widest field that still reads.
_MAXIMUM_STACKED_LEVELS = 16

# (source_name, context_key) pairs drawn in the stacked form despite exceeding
# ``_MAXIMUM_STACKED_LEVELS``, by explicit request rather than a raised cap.
# Scoped to one dataset's own composition figure: raising the cap itself would
# also pull core_trial_data's and ph_combined_nopt_rcm's variety fields into
# their composition figures and would change what the cross-dataset figures
# draw. ltcce's own panel is the only one asked to carry all of its bound
# columns, so only its entry is here.
_STACKED_LEVEL_CAP_OVERRIDES: frozenset[tuple[str, str]] = frozenset(
    {("ltcce", _VARIETY_CONTEXT)}
)

# Row slots a stacked composition panel always reserves. Below this, a figure
# with few bars draws them at a thickness that reads as emphasis.
_MINIMUM_COMPOSITION_ROW_SLOTS = 3.95

# Canvas the stacked form needs, split into the part that does not depend on the
# bar count (title, axis labels, wrapped footnote) and the part that does. Fitted
# to the configured 6.4-inch canvas at the row-slot floor above, and applied only
# as a floor-raising term in ``_figure``: a panel at or below the bar count the
# recipe was tuned for keeps exactly the configured canvas, and one above it
# grows rather than thinning its bars until the inline labels collide. Bar
# thickness is set in row units, so a proportional canvas holds it constant.
_COMPOSITION_FIXED_INCHES = 2.05
_COMPOSITION_ROW_INCHES = 0.73

# Named varieties drawn per dataset panel. Ten leaves room on a 6.4-inch canvas
# for the two pooled bars that must sit beside them without the bars thinning to
# where their inline annotations collide.
_VARIETY_LEVELS_PER_PANEL = 10

# Segments narrower than this are left unlabeled; the text would overrun its own
# segment and collide with its neighbours.
_MINIMUM_LABELED_SHARE = 0.08

# Characters a labeled segment may always spend, whatever its width. It is the
# budget at the labeling threshold above, so the narrowest labeled segment is
# the one this floor is sized for.
_MINIMUM_LABEL_CHARACTERS = 14

_TRUE_TEXT = frozenset({"true", "t", "yes", "y", "1"})


@dataclass(frozen=True)
class _BoxSummary:
    """One standardized five-number summary, ready for ``Axes.bxp``."""

    source_name: str
    label: str
    observations: int
    q1: float
    median: float
    q3: float
    low_whisker: float
    high_whisker: float
    minimum: float
    maximum: float


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _figure(
    config: DescriptiveStatisticsConfig, *, height_inches: float | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """The configured canvas, or a taller one where the caller needs the room.

    The configured height is a floor, never a ceiling: a figure whose row count
    is set by the data rather than by the recipe would otherwise thin its rows
    until the inline labels collide, which loses information the figure was
    drawn to carry.
    """

    return plt.subplots(
        figsize=(
            config.figure_width_inches,
            max(config.figure_height_inches, height_inches or 0.0),
        )
    )


def _panel_figure(
    config: DescriptiveStatisticsConfig, rows: int, columns: int
) -> tuple[plt.Figure, np.ndarray]:
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(config.figure_width_inches, config.figure_height_inches),
    )
    return figure, np.asarray(axes).reshape(rows, columns)


_FOOTNOTE_FONT_SIZE = 8
# Mean glyph advance of DejaVu Sans as a fraction of the point size. Matplotlib
# does not lay out figure.text across multiple lines for us, so a caption longer
# than the canvas silently runs off both edges instead of wrapping; wrap it here
# against a measured character budget rather than guessing a fixed width.
_FOOTNOTE_GLYPH_WIDTH_RATIO = 0.52
_FOOTNOTE_SIDE_MARGIN = 0.06
# Caption geometry, fitted as fractions of the 6.4-inch canvas the recipe is
# configured for. A line of 8-point text is the same height on every canvas, so
# these fractions over-reserve on a taller figure and leave a band of white
# between the axes and the caption; ``_footnote`` rescales them by
# 6.4/height. Kept as the fitted fractions and scaled at use, rather than
# resolved to inches and divided back, so that the scale factor is exactly 1.0
# on the tuned canvas and every figure drawn at the configured height keeps the
# bytes it had — converting through inches is off by an ULP for some line counts.
_FOOTNOTE_FITTED_HEIGHT_INCHES = 6.4
_FOOTNOTE_LINE_HEIGHT = 0.030
_FOOTNOTE_PAD = 0.022
_FOOTNOTE_BASELINE = 0.012

# ``yield_versus_nitrogen`` and ``yield_versus_nitrogen_trajectories`` draw the
# same point cloud and are meant to be read as a pair, so they reserve the same
# caption band no matter how many lines their own captions wrap to. Without this
# the trajectory panel's longer caption shrinks its axes, and the identical
# scatter renders at two different scales — the eye reads that as a difference
# in the data. The value is the largest line count either caption reaches; a
# caption that outgrows it still gets the room it needs, it just breaks the
# match, so raise this rather than let a caption exceed it.
_PAIRED_SCATTER_FOOTNOTE_LINES = 3

# The same rule for the yield-distribution set: the combined panel and the three
# single-dataset panels must share an axes rectangle, and their captions differ
# in length.
_YIELD_DISTRIBUTION_FOOTNOTE_LINES = 4

# Axes-fraction heights of the stacked callout boxes in the yield-distribution
# set. The combined panel spends one slot per dataset on its median; each
# single-dataset panel spends all three on that dataset's median, mean, and
# mode. One tuple for both, so the two figures cannot drift apart.
_YIELD_CALLOUT_SLOTS: tuple[float, ...] = (0.96, 0.885, 0.81)
# Half-height of a callout box as an axes fraction, at 8-point text on the
# configured canvas. The lowest slot must clear the tallest bar by this much,
# and the mode line sits at the tallest bar by definition, so without the
# clearance its callout would be drawn inside the bar it labels.
_YIELD_CALLOUT_CLEARANCE = 0.03


def _footnote(
    figure: plt.Figure,
    text: str,
    *,
    top: float = 1.0,
    minimum_lines: int = 1,
) -> None:
    """Caption band under the axes, wrapped to fit the canvas width.

    ``minimum_lines`` reserves the band of a taller caption than this one, so
    figures that must share an axes rectangle keep it when their caption text
    differs in length.
    """

    usable_points = figure.get_figwidth() * 72.0 * (1.0 - _FOOTNOTE_SIDE_MARGIN)
    characters = max(
        40,
        int(usable_points / (_FOOTNOTE_FONT_SIZE * _FOOTNOTE_GLYPH_WIDTH_RATIO)),
    )
    lines = textwrap.wrap(" ".join(text.split()), width=characters) or [""]
    # Exactly 1.0 on the configured canvas, so the fitted fractions apply
    # unchanged there and only a taller figure rescales.
    scale = _FOOTNOTE_FITTED_HEIGHT_INCHES / figure.get_figheight()
    figure.text(
        0.5,
        _FOOTNOTE_BASELINE * scale,
        "\n".join(lines),
        ha="center",
        va="bottom",
        fontsize=_FOOTNOTE_FONT_SIZE,
    )
    # Reserve as much of the canvas as the wrapped block occupies, so a two-line
    # caption does not overlap the x-axis label of the axes above it.
    reserved = (
        _FOOTNOTE_PAD + _FOOTNOTE_LINE_HEIGHT * max(len(lines), minimum_lines)
    ) * scale
    figure.tight_layout(rect=(0, reserved, 1, top))


def _table(
    tables: Mapping[str, pd.DataFrame], name: str
) -> pd.DataFrame | None:
    frame = tables.get(name)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    return frame


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Coerce a declared column to float, tolerating a CSV round trip."""

    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values
    return values.map(
        lambda value: bool(value)
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() in _TRUE_TEXT
    ).astype(bool)


def _text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype=object)
    return frame[column].astype(str).str.strip()


def _source_colours(loaded: LoadedSources) -> dict[str, str]:
    return {
        source.source_name: _SOURCE_PALETTE[index % len(_SOURCE_PALETTE)]
        for index, source in enumerate(loaded.sources)
    }


def _legend_label(source_name: str) -> str:
    """The dataset name alone.

    Figures deliberately do not print the data classification beside the name.
    Classification is recorded per artifact in ``run_manifest.json``, which is
    the binding record; repeating it on every legend entry only crowded the
    panels, and its absence here changes no governance behaviour — restricted
    sources are still labeled, counted, and suppressed exactly as before.
    """

    return source_name


def _tick_label(source_name: str) -> str:
    return source_name


def _ordered_sources(
    loaded: LoadedSources, present: Sequence[str]
) -> tuple[str, ...]:
    """Profiling order, restricted to the sources a frame actually carries."""

    available = set(present)
    return tuple(
        source.source_name
        for source in loaded.sources
        if source.source_name in available
    )


def _suppressed_column_ids(loaded: LoadedSources) -> set[tuple[str, str]]:
    """(source, raw_column_id) pairs that may never appear in a figure."""

    return {
        (source.source_name, spec.raw_column_id)
        for source in loaded.sources
        for spec in source.columns
        if spec.suppressed
    }


def _observations(
    loaded: LoadedSources, config: DescriptiveStatisticsConfig
) -> pd.DataFrame:
    """The harmonized (N kg/ha, yield t/ha) rows, rebuilt from the frozen loader.

    Deterministic and cheap, so each builder that needs the cloud rebuilds it
    rather than the module holding a cache keyed on a mutable caller object.
    """

    return build_all_observations(
        loaded, zero_n_tolerance_kg_ha=config.zero_n_tolerance_kg_ha
    )


def _binding_for(source: ProfiledSource, attribute: str) -> ColumnBinding | None:
    return getattr(source.binding, attribute, None)


def _distinct_level_counts(
    frame: pd.DataFrame, *, tolerance_kg_ha: float
) -> pd.Series:
    """Distinct N levels per series, with rates equal within the tolerance fused.

    Recorded ladders repeat a rate as ``100`` and ``100.0``; snapping to the
    configured tolerance grid before counting keeps those one level instead of
    two.
    """

    working = frame.loc[:, ["series_key", "n_rate_kg_ha"]].copy()
    if tolerance_kg_ha > 0:
        snapped = np.round(
            working["n_rate_kg_ha"].to_numpy(dtype=float) / tolerance_kg_ha
        )
    else:
        snapped = working["n_rate_kg_ha"].to_numpy(dtype=float)
    working["level"] = snapped
    return working.groupby("series_key")["level"].nunique()


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------


def _plot_dataset_scale(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    inventory = _table(tables, "source_inventory")
    if inventory is None:
        return None
    names = _ordered_sources(loaded, _text_column(inventory, "source_name"))
    if not names:
        return None
    indexed = inventory.set_index(_text_column(inventory, "source_name"))
    rows = _numeric_column(indexed, "data_row_count").reindex(names)
    columns = _numeric_column(indexed, "physical_column_count").reindex(names)
    if not (rows.notna().any() and columns.notna().any()):
        return None

    colours = _source_colours(loaded)
    figure, axis = _figure(config)
    positions = np.arange(len(names), dtype=float)
    width = 0.36
    for offset, values, hatch, alpha in (
        (-width / 2, rows, None, 0.95),
        (width / 2, columns, "//", 0.45),
    ):
        for index, name in enumerate(names):
            value = float(values.iloc[index])
            if not np.isfinite(value) or value <= 0:
                continue
            axis.bar(
                positions[index] + offset,
                value,
                width=width,
                color=colours[name],
                alpha=alpha,
                hatch=hatch,
                edgecolor=colours[name],
                linewidth=0.8,
            )
            axis.annotate(
                f"{int(round(value)):,}",
                xy=(positions[index] + offset, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # The counts span three orders of magnitude (13,952 rows against 12
    # columns); on a linear axis every bar but the tallest collapses onto the
    # baseline, so the axis is logarithmic and each bar carries its literal
    # value.
    axis.set_yscale("log")
    finite = np.concatenate(
        [
            rows.to_numpy(dtype=float)[np.isfinite(rows.to_numpy(dtype=float))],
            columns.to_numpy(dtype=float)[np.isfinite(columns.to_numpy(dtype=float))],
        ]
    )
    axis.set_ylim(1.0, float(np.max(finite)) * 6.0)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [_tick_label(name) for name in names],
        fontsize=9,
    )
    axis.set(
        title="Dataset scale: recorded data rows and physical columns",
        xlabel="Registered source dataset",
        ylabel="Count (logarithmic scale)",
    )
    axis.legend(
        handles=[
            Patch(facecolor=_NEUTRAL, edgecolor=_NEUTRAL, label="Data rows"),
            Patch(
                facecolor=_NEUTRAL,
                edgecolor=_NEUTRAL,
                alpha=0.45,
                hatch="//",
                label="Physical columns",
            ),
        ],
        fontsize=8,
    )
    axis.grid(axis="y", alpha=0.2, which="both")
    _footnote(
        figure,
        "Logarithmic axis: row and column counts differ by about three orders "
        "of magnitude. Physical columns include wholly empty and policy-"
        "suppressed columns, so the totals reconcile with the source files.",
    )
    return figure


def _plot_column_fill_profile(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    inventory = _table(tables, "column_inventory")
    if inventory is None:
        return None
    frame = inventory.copy()
    frame["_source"] = _text_column(frame, "source_name")
    frame["_fill"] = _numeric_column(frame, "fill_rate")
    frame["_column_id"] = _text_column(frame, "raw_column_id")
    # A suppressed column reads as 0% filled because policy blanked its cells,
    # not because the source left them empty. Plotting it would overstate
    # missingness in exactly the restricted dataset the suppression protects.
    suppressed = _suppressed_column_ids(loaded)
    declared_suppressed = _boolean_column(frame, "suppressed")
    is_suppressed = declared_suppressed | pd.Series(
        [
            (source, column_id) in suppressed
            for source, column_id in zip(frame["_source"], frame["_column_id"])
        ],
        index=frame.index,
    )
    excluded = (
        is_suppressed.groupby(frame["_source"]).sum().astype(int).to_dict()
    )
    frame = frame.loc[~is_suppressed & frame["_fill"].notna()]
    names = _ordered_sources(loaded, frame["_source"])
    if not names:
        return None

    colours = _source_colours(loaded)
    figure, axis = _figure(config)
    empty_notes: list[str] = []
    for name in names:
        values = (
            frame.loc[frame["_source"] == name, "_fill"]
            .to_numpy(dtype=float)
            .copy()
        )
        if values.size == 0:
            continue
        values.sort()
        values = values[::-1]
        # Rank is expressed as a percentage of each dataset's own columns so a
        # 302-column extract and a 12-column table share one x axis.
        rank = (np.arange(values.size, dtype=float) + 1.0) / values.size * 100.0
        axis.plot(
            rank,
            values * 100.0,
            color=colours[name],
            linewidth=1.9,
            marker="o" if values.size <= 20 else None,
            markersize=3.5,
            # Short label deliberately: a legend wide enough to carry the column
            # counts would cover the steep part of the middle curve. The counts
            # are in the caption instead.
            label=_legend_label(name),
        )
        empty_notes.append(f"{name}: {int(np.sum(values <= 0.0))} of {values.size}")

    axis.set_ylim(-2.0, 104.0)
    axis.set_xlim(0.0, 100.0)
    axis.set(
        title="Column fill-rate profile: physical columns sorted from most to least populated",
        xlabel="Physical columns, ranked by fill rate (percent of the dataset's profiled columns)",
        ylabel="Fill rate (percent of data rows carrying a value)",
    )
    # Centre-right is the only region all three curves leave empty: two of them
    # sit on the 100% ceiling and the third on the 0% floor by mid-rank.
    axis.legend(fontsize=8, loc="center right")
    axis.grid(alpha=0.2)
    suppression_note = ", ".join(
        f"{name}: {excluded.get(name, 0)}" for name in names
    )
    _footnote(
        figure,
        "Policy-suppressed identifier columns are excluded because their cells "
        "are blanked by governance, not missing in the source "
        f"(excluded — {suppression_note}). Wholly empty columns, of those "
        f"profiled — {', '.join(empty_notes)}.",
    )
    return figure


# --------------------------------------------------------------------------
# numeric
# --------------------------------------------------------------------------


def _plot_numeric_spread_overview(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    summary = _table(tables, "numeric_summary")
    if summary is None:
        return None
    frame = summary.copy()
    frame["_source"] = _text_column(frame, "source_name")
    frame["_column_id"] = _text_column(frame, "raw_column_id")
    # Keyed on (source, raw_column_id) and never on header text: the core
    # extract carries seven duplicated and sixteen blank header names.
    indexed = frame.set_index(["_source", "_column_id"])

    boxes: list[_BoxSummary] = []
    for source in loaded.sources:
        for attribute, label in _SPREAD_QUANTITIES:
            binding = _binding_for(source, attribute)
            if binding is None:
                continue
            key = (source.source_name, binding.raw_column_id)
            if key not in indexed.index:
                continue
            row = indexed.loc[key]
            if isinstance(row, pd.DataFrame):  # duplicated summary row
                row = row.iloc[0]
            mean = pd.to_numeric(pd.Series([row.get("mean")]), errors="coerce").iloc[0]
            deviation = pd.to_numeric(
                pd.Series([row.get("std_dev")]), errors="coerce"
            ).iloc[0]
            # Standardizing by the median and IQR would force every box to unit
            # width and erase the comparison; mean/SD keeps the box width, the
            # whisker asymmetry, and the tail reach informative.
            if not np.isfinite(mean) or not np.isfinite(deviation) or deviation <= 0:
                continue
            quantiles = {
                name: pd.to_numeric(
                    pd.Series([row.get(name)]), errors="coerce"
                ).iloc[0]
                for name in ("p05", "q1", "median", "q3", "p95", "minimum", "maximum")
            }
            if any(not np.isfinite(value) for value in quantiles.values()):
                continue
            standardized = {
                name: (value - mean) / deviation for name, value in quantiles.items()
            }
            count = pd.to_numeric(pd.Series([row.get("count")]), errors="coerce").iloc[0]
            boxes.append(
                _BoxSummary(
                    source_name=source.source_name,
                    label=label,
                    observations=int(count) if np.isfinite(count) else 0,
                    q1=float(standardized["q1"]),
                    median=float(standardized["median"]),
                    q3=float(standardized["q3"]),
                    low_whisker=float(standardized["p05"]),
                    high_whisker=float(standardized["p95"]),
                    minimum=float(standardized["minimum"]),
                    maximum=float(standardized["maximum"]),
                )
            )
    if len(boxes) < 2:
        return None

    colours = _source_colours(loaded)
    figure, axis = _figure(config)
    # The tick label names its own dataset, so no colour legend is needed — and
    # none is drawn, because any in-axes legend would cover the extreme ticks it
    # would have to sit beside.
    stats = [
        {
            "label": (
                f"{box.label}\n{box.source_name} (n={box.observations:,})"
            ),
            "med": box.median,
            "q1": box.q1,
            "q3": box.q3,
            "whislo": box.low_whisker,
            "whishi": box.high_whisker,
            "fliers": [],
        }
        for box in boxes
    ]
    positions = np.arange(len(boxes), dtype=float) + 1.0
    artists = axis.bxp(
        stats,
        positions=positions,
        widths=0.6,
        orientation="horizontal",
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.6},
        whiskerprops={"color": _NEUTRAL, "linewidth": 1.2},
        capprops={"color": _NEUTRAL, "linewidth": 1.2},
    )
    for patch, box in zip(artists["boxes"], boxes):
        patch.set_facecolor(colours[box.source_name])
        patch.set_alpha(0.65)
        patch.set_edgecolor(colours[box.source_name])
    axis.scatter(
        [box.minimum for box in boxes] + [box.maximum for box in boxes],
        np.concatenate([positions, positions]),
        marker="|",
        s=90,
        color=_NEUTRAL,
        linewidths=1.1,
        zorder=4,
    )
    axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_yticklabels(
        [stat["label"] for stat in stats],
        fontsize=8,
    )
    axis.set(
        title="Standardized spread of the agronomically bound numeric columns",
        xlabel="Standardized value ((x − mean) ÷ standard deviation)",
        ylabel="Bound quantity and dataset",
    )
    axis.grid(axis="x", alpha=0.2)
    _footnote(
        figure,
        "Boxes span Q1–Q3 around the median, whiskers reach the 5th and 95th "
        "percentiles, and grey ticks mark the recorded minimum and maximum. "
        "Each column is standardized on its own mean and SD, so shapes are "
        "comparable and absolute magnitudes are not.",
    )
    return figure


# --------------------------------------------------------------------------
# categorical
# --------------------------------------------------------------------------


def _plot_categorical_cardinality(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    summary = _table(tables, "categorical_summary")
    if summary is None:
        return None
    frame = summary.copy()
    frame["_source"] = _text_column(frame, "source_name")
    frame["_column_id"] = _text_column(frame, "raw_column_id")
    frame["_distinct"] = _numeric_column(frame, "distinct_count")
    frame["_entropy"] = _numeric_column(frame, "normalized_entropy")
    frame["_identifier"] = _boolean_column(frame, "is_identifier_like")
    # Defensive: a suppressed column classifies as value_kind="suppressed" and
    # should never reach this table, but this is the one figure that labels
    # individual columns by name, so the cross-check is worth its cost.
    suppressed = _suppressed_column_ids(loaded)
    keep = ~pd.Series(
        [
            (source, column_id) in suppressed
            for source, column_id in zip(frame["_source"], frame["_column_id"])
        ],
        index=frame.index,
    )
    frame = frame.loc[keep & frame["_distinct"].notna() & frame["_entropy"].notna()]
    frame = frame.loc[frame["_distinct"] > 0]
    names = _ordered_sources(loaded, frame["_source"])
    if frame.empty or not names:
        return None

    colours = _source_colours(loaded)
    figure, axis = _figure(config)
    for name in names:
        subset = frame.loc[frame["_source"] == name]
        for identifier_like, marker, face in (
            (False, "o", colours[name]),
            (True, "^", "none"),
        ):
            points = subset.loc[subset["_identifier"] == identifier_like]
            if points.empty:
                continue
            axis.scatter(
                points["_distinct"],
                points["_entropy"],
                s=44,
                marker=marker,
                facecolor=face,
                edgecolor=colours[name],
                linewidth=1.0,
                alpha=0.85,
                zorder=3,
            )

    # Only the extremes are labeled: the highest-cardinality and the
    # lowest-entropy column per dataset. Labeling every point would render the
    # panel unreadable at 302 columns.
    labeled: set[tuple[str, str]] = set()
    anchors: list[tuple[float, float]] = []
    log_span = float(
        np.log10(frame["_distinct"].max()) - np.log10(frame["_distinct"].min())
    )
    for name in names:
        subset = frame.loc[frame["_source"] == name]
        for index in (subset["_distinct"].idxmax(), subset["_entropy"].idxmin()):
            row = frame.loc[index]
            key = (str(row["_source"]), str(row["_column_id"]))
            if key in labeled:
                continue
            labeled.add(key)
            header = str(row.get("header_label", "")).strip() or "(blank header)"
            if len(header) > 26:
                header = header[:25] + "…"
            cardinality = float(row["_distinct"])
            entropy = float(row["_entropy"])
            # Every dataset's lowest-entropy column sits at (small k, 0.0), so
            # the extreme labels pile onto the same point. Stack each new label
            # a row higher than the ones already anchored near it.
            collisions = sum(
                1
                for x, y in anchors
                if abs(np.log10(cardinality) - np.log10(x)) < 0.12
                and abs(entropy - y) < 0.08
            )
            anchors.append((cardinality, entropy))
            # A label on a right-edge point would run off the axes, so those
            # are written back towards the interior instead.
            on_right_edge = log_span > 0 and (
                np.log10(cardinality) - np.log10(frame["_distinct"].min())
            ) / log_span > 0.72
            axis.annotate(
                f"pos {int(_numeric_column(frame, 'position').loc[index])} · {header}",
                xy=(cardinality, entropy),
                xytext=(-8 if on_right_edge else 8, 6 + 13 * collisions),
                textcoords="offset points",
                ha="right" if on_right_edge else "left",
                fontsize=7,
                color=colours[name],
            )

    axis.set_xscale("log")
    axis.set_ylim(-0.04, 1.12)
    axis.set(
        title="Categorical columns: cardinality against normalized entropy",
        xlabel="Distinct nonblank levels in the column (count, logarithmic scale)",
        ylabel="Normalized entropy (0 = one level dominates, 1 = levels are uniform)",
    )
    handles: list[Line2D] = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor=colours[name],
            markeredgecolor=colours[name],
            markersize=7,
            label=_legend_label(name),
        )
        for name in names
    ]
    if bool(frame["_identifier"].any()):
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="^",
                markerfacecolor="none",
                markeredgecolor=_NEUTRAL,
                markersize=7,
                label="Identifier-like column",
            )
        )
    # Lower right: the low-entropy columns cluster at low cardinality, so the
    # bottom-left corner is exactly where the labeled extremes live.
    axis.legend(handles=handles, fontsize=8, loc="lower right")
    axis.grid(alpha=0.2, which="both")
    _footnote(
        figure,
        "One point per physical column. Only the highest-cardinality and the "
        "lowest-entropy column of each dataset is labeled, by physical position "
        "and header text. Policy-suppressed columns are excluded entirely.",
    )
    return figure


# --------------------------------------------------------------------------
# agronomic
# --------------------------------------------------------------------------


def _share_histogram(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    edges: np.ndarray,
    colour: str,
    label: str,
) -> None:
    """Within-dataset share histogram; counts differ ~19-fold between sources."""

    weights = np.full(values.size, 100.0 / values.size)
    axis.hist(
        values,
        bins=edges,
        weights=weights,
        histtype="stepfilled",
        alpha=0.35,
        color=colour,
        edgecolor="none",
        label=label,
    )
    # Three translucent fills stacked over each other hide whichever dataset is
    # drawn first; an opaque outline keeps every shape readable. Unlabeled, so
    # it adds no second legend entry.
    axis.hist(
        values,
        bins=edges,
        weights=weights,
        histtype="step",
        color=colour,
        linewidth=1.7,
    )


def _plot_nitrogen_rate_distribution(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    observations = _observations(loaded, config)
    if observations.empty:
        return None
    names = _ordered_sources(loaded, observations["source_name"])
    if not names:
        return None

    rates = observations["n_rate_kg_ha"].to_numpy(dtype=float)
    width = float(config.nitrogen_bin_width_kg_ha)
    lower = float(np.floor(np.nanmin(rates) / width) * width)
    upper = float(np.ceil(np.nanmax(rates) / width) * width) + width
    edges = np.arange(lower, upper + width / 2.0, width)
    if edges.size < 2:
        return None

    colours = _source_colours(loaded)
    figure, axis = _figure(config)
    for name in names:
        values = observations.loc[
            observations["source_name"] == name, "n_rate_kg_ha"
        ].to_numpy(dtype=float)
        if values.size == 0:
            continue
        distinct = int(np.unique(np.round(values, 6)).size)
        _share_histogram(
            axis,
            values,
            edges=edges,
            colour=colours[name],
            label=(
                f"{_legend_label(name)} — "
                f"{values.size:,} observations, {distinct} distinct rates"
            ),
        )
    axis.set(
        title=(
            "Recorded inorganic N-rate distribution per dataset "
            f"({width:g} kg N ha⁻¹ bins)"
        ),
        xlabel="Inorganic N rate (kg N ha⁻¹)",
        ylabel="Share of the dataset's harmonized observations (%)",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    _footnote(
        figure,
        "Bars are within-dataset shares because the observation counts differ "
        "by more than an order of magnitude. The rates are discrete "
        "experimental ladders, not a sample from a continuous distribution.",
    )
    return figure


@dataclass(frozen=True)
class _YieldDistributionAxes:
    """Bin edges and limits shared by the combined panel and the per-source ones.

    Every quantity is computed from the whole observation frame, never from the
    subset a panel happens to draw. Three single-dataset panels are only
    readable as a set if their axes are identical, and the y-axis is the one
    that would silently diverge: shares are within-dataset, so 718 nopt rows
    concentrate into taller bars than 13,653 ltcce rows spread over the same
    bins, and per-panel autoscale would render three different scales as if
    they were three different distributions.
    """

    edges: np.ndarray
    xlim: tuple[float, float]
    ylim: tuple[float, float]


def _yield_distribution_axes(
    observations: pd.DataFrame, *, names: Sequence[str], config: DescriptiveStatisticsConfig
) -> _YieldDistributionAxes | None:
    yields = observations["yield_t_ha"].to_numpy(dtype=float)
    edges = np.linspace(
        float(np.nanmin(yields)),
        float(np.nanmax(yields)),
        int(config.histogram_bins) + 1,
    )
    if edges.size < 2 or not np.isfinite(edges).all():
        return None

    tallest = 0.0
    for name in names:
        values = observations.loc[
            observations["source_name"] == name, "yield_t_ha"
        ].to_numpy(dtype=float)
        if values.size == 0:
            continue
        counts, _ = np.histogram(values, bins=edges)
        tallest = max(tallest, float(counts.max()) * 100.0 / values.size)
    if tallest <= 0.0:
        return None

    span = float(edges[-1] - edges[0])
    return _YieldDistributionAxes(
        edges=edges,
        # The combined panel's autoscale padding, pinned so the per-source
        # panels cannot drift from it.
        xlim=(float(edges[0]) - 0.05 * span, float(edges[-1]) + 0.05 * span),
        # Headroom for the lowest callout slot, derived from the slot heights
        # rather than fixed, so moving a slot cannot silently push its box into
        # the bars.
        ylim=(0.0, tallest / (min(_YIELD_CALLOUT_SLOTS) - _YIELD_CALLOUT_CLEARANCE)),
    )


@dataclass(frozen=True)
class _ModalClass:
    """The tallest histogram bin of one dataset, and its nearest rival.

    The mode of a continuous measurement is only defined against a binning, so
    it is reported as the midpoint of a stated interval rather than as a point.
    The runner-up travels with it because a flat-topped distribution has a
    tallest bin that no reader should treat as a located peak: where several
    bins are within a percentage point of each other, the winner is decided by
    a handful of observations and would move under a different grid. Carrying
    the rival lets the caption show that margin instead of asserting a mode.
    The figures state the values themselves, computed from the data at draw
    time; none are repeated here, because the two restricted sources' derived
    numbers do not belong in tracked source.
    """

    low: float
    high: float
    share: float
    runner_low: float
    runner_high: float
    runner_share: float

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.low + self.high)

    @property
    def width(self) -> float:
        return self.high - self.low


def _modal_class(values: np.ndarray, *, edges: np.ndarray) -> _ModalClass | None:
    """Tallest and second-tallest bin of ``values`` on the shared grid.

    ``np.argmax`` settles a tie on the lowest bin, which is a stable rule rather
    than a meaningful one; the reported runner-up share makes such a tie visible
    to the reader instead of hiding it behind a single number.
    """

    finite = values[np.isfinite(values)]
    if finite.size == 0 or edges.size < 3:
        return None
    counts, _ = np.histogram(finite, bins=edges)
    shares = counts * 100.0 / finite.size
    order = np.argsort(counts, kind="stable")[::-1]
    top, runner = int(order[0]), int(order[1])
    if counts[top] <= 0:
        return None
    return _ModalClass(
        low=float(edges[top]),
        high=float(edges[top + 1]),
        share=float(shares[top]),
        runner_low=float(edges[runner]),
        runner_high=float(edges[runner + 1]),
        runner_share=float(shares[runner]),
    )


def _plot_yield_distribution(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    observations = _observations(loaded, config)
    if observations.empty:
        return None
    names = _ordered_sources(loaded, observations["source_name"])
    if not names:
        return None

    shared = _yield_distribution_axes(observations, names=names, config=config)
    if shared is None:
        return None
    edges = shared.edges

    colours = _source_colours(loaded)
    figure, axis = _figure(config)
    converted = 0
    for index, name in enumerate(names):
        subset = observations.loc[observations["source_name"] == name]
        values = subset["yield_t_ha"].to_numpy(dtype=float)
        if values.size == 0:
            continue
        converted += int(
            (subset["yield_unit_lineage"].astype(str) == "converted_from_kg_ha").sum()
        )
        median = float(np.median(values))
        _share_histogram(
            axis,
            values,
            edges=edges,
            colour=colours[name],
            label=(
                f"{_legend_label(name)} — "
                f"{values.size:,} observations"
            ),
        )
        axis.axvline(median, color=colours[name], linestyle="--", linewidth=1.5)
        axis.annotate(
            f"median {median:.2f}",
            # Deliberately unguarded: a fourth configured source must raise
            # here rather than wrap onto an occupied slot and hide one
            # dataset's median under another's.
            xy=(median, _YIELD_CALLOUT_SLOTS[index]),
            xycoords=("data", "axes fraction"),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color=colours[name],
            bbox=_ANNOTATION_BOX,
        )
    axis.set_xlim(*shared.xlim)
    axis.set_ylim(*shared.ylim)
    axis.set(
        title="Recorded grain-yield distribution per dataset on the common t ha⁻¹ basis",
        xlabel="Grain yield (t ha⁻¹)",
        ylabel="Share of the dataset's harmonized observations (%)",
    )
    axis.legend(fontsize=8, loc="upper right")
    axis.grid(alpha=0.2)
    lineage = (
        f"{converted:,} observations reached t ha⁻¹ by conversion from kg ha⁻¹"
        if converted
        else "every observation is recorded natively in t ha⁻¹"
    )
    _footnote(
        figure,
        "Bars are within-dataset shares; dashed lines mark each dataset's "
        f"median. Unit lineage: {lineage}.",
        minimum_lines=_YIELD_DISTRIBUTION_FOOTNOTE_LINES,
    )
    return figure


def _plot_one_yield_distribution(
    source_name: str,
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    """One dataset's yield distribution, on the axes the whole set shares.

    Same bins, same limits, and same axes rectangle as its two siblings and as
    the combined panel, so the four can be read against one another. Only the
    drawn subset changes.
    """

    observations = _observations(loaded, config)
    if observations.empty:
        return None
    names = _ordered_sources(loaded, observations["source_name"])
    if source_name not in names:
        return None

    shared = _yield_distribution_axes(observations, names=names, config=config)
    if shared is None:
        return None

    subset = observations.loc[observations["source_name"] == source_name]
    values = subset["yield_t_ha"].to_numpy(dtype=float)
    if values.size == 0:
        return None

    colour = _source_colours(loaded)[source_name]
    figure, axis = _figure(config)
    _share_histogram(
        axis,
        values,
        edges=shared.edges,
        colour=colour,
        label=f"{_legend_label(source_name)} — {values.size:,} observations",
    )
    # Median, mean, and mode, each on its own slot. The three coincide closely
    # in some datasets — core_trial_data puts them inside 0.15 t ha⁻¹ — so the
    # boxes are separated vertically rather than horizontally, which lets them
    # overlap in x without colliding.
    median = float(np.median(values))
    axis.axvline(median, color=colour, linestyle="--", linewidth=1.5)
    axis.annotate(
        f"median {median:.2f}",
        xy=(median, _YIELD_CALLOUT_SLOTS[0]),
        xycoords=("data", "axes fraction"),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8,
        color=colour,
        bbox=_ANNOTATION_BOX,
    )
    mean = float(np.mean(values))
    axis.axvline(mean, color=colour, linestyle=":", linewidth=1.5)
    axis.annotate(
        f"mean {mean:.2f}",
        xy=(mean, _YIELD_CALLOUT_SLOTS[1]),
        xycoords=("data", "axes fraction"),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8,
        color=colour,
        bbox=_ANNOTATION_BOX,
    )
    modal = _modal_class(values, edges=shared.edges)
    if modal is not None:
        axis.axvline(modal.midpoint, color=colour, linestyle="-.", linewidth=1.5)
        axis.annotate(
            f"mode {modal.midpoint:.2f}",
            xy=(modal.midpoint, _YIELD_CALLOUT_SLOTS[2]),
            xycoords=("data", "axes fraction"),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color=colour,
            bbox=_ANNOTATION_BOX,
        )
    axis.set_xlim(*shared.xlim)
    axis.set_ylim(*shared.ylim)
    axis.set(
        title=(
            f"Recorded grain-yield distribution — {source_name} "
            "(t ha⁻¹ basis)"
        ),
        xlabel="Grain yield (t ha⁻¹)",
        ylabel="Share of the dataset's harmonized observations (%)",
    )
    axis.legend(fontsize=8, loc="upper right")
    axis.grid(alpha=0.2)
    converted = int(
        (subset["yield_unit_lineage"].astype(str) == "converted_from_kg_ha").sum()
    )
    lineage = (
        f"{converted:,} of them reached t ha⁻¹ by conversion from kg ha⁻¹"
        if converted
        else "every one is recorded natively in t ha⁻¹"
    )
    # The mode of a continuous measurement is a property of the binning, so the
    # caption states the interval it came from and the margin it won by; a bare
    # "mode 5.20" would read as a located peak even where the top bins are
    # separated by a handful of observations.
    if modal is None:
        modal_clause = ""
    else:
        modal_clause = (
            " The dash-dotted line marks the midpoint of its modal class, the "
            f"tallest {modal.width:.2f} t ha⁻¹ bin "
            f"({modal.low:.2f}-{modal.high:.2f} t ha⁻¹, {modal.share:.1f}% of "
            f"the dataset); the next-tallest bin "
            f"({modal.runner_low:.2f}-{modal.runner_high:.2f}) holds "
            f"{modal.runner_share:.1f}%."
        )
    _footnote(
        figure,
        "Bars are shares of this dataset's own harmonized observations; the "
        "dashed line marks its median and the dotted line marks its mean."
        f"{modal_clause} "
        "Bins, axis limits, and panel size are shared with the other "
        "per-dataset panels and with the combined figure, so the four are "
        f"directly comparable. Unit lineage: {lineage}.",
        minimum_lines=_YIELD_DISTRIBUTION_FOOTNOTE_LINES,
    )
    return figure


def _yield_distribution_builder(
    source_name: str,
) -> Callable[..., "plt.Figure | None"]:
    """Bind one source name into the shared single-dataset builder."""

    def builder(
        *,
        tables: Mapping[str, pd.DataFrame],
        loaded: LoadedSources,
        config: DescriptiveStatisticsConfig,
    ) -> plt.Figure | None:
        return _plot_one_yield_distribution(
            source_name, tables=tables, loaded=loaded, config=config
        )

    builder.__name__ = f"_plot_yield_distribution_{source_name}"
    return builder


def _plot_yield_versus_nitrogen(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    observations = _observations(loaded, config)
    if observations.empty:
        return None
    names = _ordered_sources(loaded, observations["source_name"])
    if not names:
        return None

    colours = _source_colours(loaded)
    figure, axis = _figure(config)
    counts = observations["source_name"].value_counts()
    # Largest cloud first so the smaller datasets are not buried beneath it.
    drawing_order = sorted(names, key=lambda name: int(counts.get(name, 0)), reverse=True)
    handles: list[Line2D] = []
    for name in drawing_order:
        subset = observations.loc[observations["source_name"] == name]
        if subset.empty:
            continue
        # 15,558 points overplot heavily; opacity is scaled to each dataset's
        # own density so a 718-row set stays visible beside a 13,653-row one.
        alpha = float(np.clip(1500.0 / len(subset), 0.10, 0.55))
        axis.scatter(
            subset["n_rate_kg_ha"],
            subset["yield_t_ha"],
            s=11,
            color=colours[name],
            alpha=alpha,
            edgecolor="none",
            zorder=2,
        )
    for name in names:
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markerfacecolor=colours[name],
                markeredgecolor="none",
                markersize=7,
                label=(
                    f"{_legend_label(name)} — "
                    f"{int(counts.get(name, 0)):,} observations"
                ),
            )
        )
    axis.text(
        0.02,
        0.02,
        "Scatter of recorded observations only. No fitted line, trend, "
        "smoother, or optimum is drawn:\nthis profile describes what was "
        "recorded and makes no response or recommendation claim.",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=_ANNOTATION_BOX,
    )
    # Headroom so the upper-right legend cannot sit on top of the highest
    # yields, which occur at the mid-range N rates under it.
    low, high = axis.get_ylim()
    axis.set_ylim(low, high + 0.12 * (high - low))
    axis.set(
        title="Recorded grain yield against recorded inorganic N rate",
        xlabel="Inorganic N rate (kg N ha⁻¹)",
        ylabel="Grain yield (t ha⁻¹)",
    )
    axis.legend(handles=handles, fontsize=8, loc="upper right")
    axis.grid(alpha=0.2)
    _footnote(
        figure,
        "Each point is one harmonized row carrying both a finite N rate and a "
        "finite yield. Datasets differ in design, era, and site; the panel is a "
        "coverage view, not a comparison of effects.",
        minimum_lines=_PAIRED_SCATTER_FOOTNOTE_LINES,
    )
    return figure


@dataclass(frozen=True)
class _Trajectories:
    """Drawable polylines for one figure, with the counts that qualify them."""

    paths: dict[str, list[np.ndarray]]
    joined: dict[str, int]
    total: dict[str, int]
    # Joined series whose rows carry more than one recorded year. A connecting
    # line asserts "one experimental unit at different N rates", which stops
    # being true when the series spans two plantings, so the count is surfaced
    # on the panel rather than left for the reader to discover.
    multi_year: dict[str, int]


def _series_trajectories(
    observations: pd.DataFrame, *, tolerance_kg_ha: float
) -> _Trajectories:
    """One ascending-N polyline per series that records more than one N level.

    Replicate rows sharing a series and an N level are reduced to their median
    yield, so a node is one point rather than a vertical stack: ltcce records a
    median of four replicates at each rung of its four-rung ladder, and joining
    the raw rows in sorted order would draw a zigzag between replicates that no
    trial performed. The median is a derived summary and is declared as such on
    the panel.

    Series carrying a single N level yield no segment and are counted, not
    dropped silently — every ph_combined_nopt_rcm record is one, because that
    dataset holds its zero-N arm as a paired column rather than a second row.
    """

    working = observations.loc[
        :, ["source_name", "series_key", "n_rate_kg_ha", "yield_t_ha"]
    ].copy()
    years = _numeric_column(observations, "year")
    # Fuse rates that differ only in recorded precision (100 vs 100.0) onto the
    # configured grid, exactly as the ladder-geometry count does, so a series is
    # not credited with two levels that the trial ran as one.
    if tolerance_kg_ha > 0:
        working["level"] = (
            np.round(working["n_rate_kg_ha"].to_numpy(dtype=float) / tolerance_kg_ha)
            * tolerance_kg_ha
        )
    else:
        working["level"] = working["n_rate_kg_ha"].astype(float)

    nodes = (
        working.groupby(["source_name", "series_key", "level"], sort=True)["yield_t_ha"]
        .median()
        .reset_index()
    )

    # Distinct recorded years per series, blanks ignored: a series with no year
    # at all is not evidence of spanning two.
    dated = observations.loc[years.notna(), ["series_key"]].assign(
        year=years.loc[years.notna()]
    )
    years_per_series = dated.groupby("series_key")["year"].nunique()

    segments: dict[str, list[np.ndarray]] = {}
    joined: dict[str, int] = {}
    total: dict[str, int] = {}
    multi_year: dict[str, int] = {}
    for source_name, source_nodes in nodes.groupby("source_name", sort=True):
        paths: list[np.ndarray] = []
        series_seen = 0
        spanning = 0
        # Sorted series order, so the draw sequence — and therefore the encoded
        # JPEG — does not depend on row order or grouping internals.
        for series_key, series_nodes in source_nodes.groupby("series_key", sort=True):
            series_seen += 1
            if len(series_nodes) < 2:
                continue
            ordered = series_nodes.sort_values("level")
            paths.append(
                np.column_stack(
                    (
                        ordered["level"].to_numpy(dtype=float),
                        ordered["yield_t_ha"].to_numpy(dtype=float),
                    )
                )
            )
            if int(years_per_series.get(series_key, 0)) > 1:
                spanning += 1
        segments[str(source_name)] = paths
        joined[str(source_name)] = len(paths)
        total[str(source_name)] = series_seen
        multi_year[str(source_name)] = spanning
    return _Trajectories(
        paths=segments, joined=joined, total=total, multi_year=multi_year
    )


def _plot_yield_versus_nitrogen_trajectories(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    """The N-yield cloud with each series' own observations joined in place.

    Same point cloud as ``yield_versus_nitrogen`` — identical rows, colours, and
    opacity rule — with within-series connecting lines added beneath it. The two
    panels are meant to be read together: one shows coverage, this one shows
    that the coverage is made of ladders.
    """

    observations = _observations(loaded, config)
    if observations.empty:
        return None
    names = _ordered_sources(loaded, observations["source_name"])
    if not names:
        return None

    trajectories = _series_trajectories(
        observations, tolerance_kg_ha=config.n_level_tolerance_kg_ha
    )
    segments, joined, total = (
        trajectories.paths,
        trajectories.joined,
        trajectories.total,
    )
    if not any(joined.get(name, 0) for name in names):
        return None

    colours = _source_colours(loaded)
    figure, axis = _figure(config)
    counts = observations["source_name"].value_counts()
    drawing_order = sorted(names, key=lambda name: int(counts.get(name, 0)), reverse=True)

    for name in drawing_order:
        paths = segments.get(name, [])
        if not paths:
            continue
        # Lines sit under the points, at their own much lower opacity: 774 ltcce
        # ladders share one four-rung grid and superimpose almost exactly, so an
        # opacity that suits a point cloud renders them as a solid block.
        line_alpha = float(np.clip(120.0 / len(paths), 0.05, 0.45))
        axis.add_collection(
            LineCollection(
                paths,
                colors=colours[name],
                linewidths=0.5,
                alpha=line_alpha,
                zorder=1,
            )
        )
    for name in drawing_order:
        subset = observations.loc[observations["source_name"] == name]
        if subset.empty:
            continue
        alpha = float(np.clip(1500.0 / len(subset), 0.10, 0.55))
        axis.scatter(
            subset["n_rate_kg_ha"],
            subset["yield_t_ha"],
            s=11,
            color=colours[name],
            alpha=alpha,
            edgecolor="none",
            zorder=2,
        )

    handles: list[Line2D] = []
    for name in names:
        drawn = joined.get(name, 0)
        seen = total.get(name, 0)
        detail = (
            f"{drawn:,} of {seen:,} series joined"
            if drawn
            else f"no series joined ({seen:,} single-rate)"
        )
        handles.append(
            Line2D(
                [],
                [],
                linestyle="-" if drawn else "none",
                linewidth=1.2,
                color=colours[name],
                marker="o",
                markerfacecolor=colours[name],
                markeredgecolor="none",
                markersize=7,
                label=f"{_legend_label(name)} — {detail}",
            )
        )

    axis.text(
        0.02,
        0.02,
        "Lines join observations recorded within one N-rate series, in "
        "ascending N order; where a\nseries replicates an N rate, the node is "
        "the median of those replicate rows. Nothing is fitted,\nsmoothed, or "
        "extrapolated, and no optimum is marked.",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=_ANNOTATION_BOX,
    )
    axis.autoscale_view()
    low, high = axis.get_ylim()
    axis.set_ylim(low, high + 0.12 * (high - low))
    axis.set(
        title=(
            "Recorded grain yield against recorded inorganic N rate, "
            "joined within each series"
        ),
        xlabel="Inorganic N rate (kg N ha⁻¹)",
        ylabel="Grain yield (t ha⁻¹)",
    )
    axis.legend(handles=handles, fontsize=8, loc="upper right")
    axis.grid(alpha=0.2)

    lineless = [name for name in names if not joined.get(name, 0)]
    note = (
        "A series is one recorded N ladder, as bound in [agronomic.<source>]. "
        "Series that record a single N rate contribute a point but no line."
    )
    if lineless:
        note += (
            " That is every series of "
            + ", ".join(lineless)
            + ": each record holds one N rate, and its zero-N arm is a paired "
            "column profiled in zero_nitrogen_checks, not a second row."
        )
    spanning = [
        f"{trajectories.multi_year[name]:,} in {name}"
        for name in names
        if trajectories.multi_year.get(name, 0)
    ]
    if spanning:
        note += (
            " Some joined series record more than one year of planting ("
            + ", ".join(spanning)
            + "), so those lines join observations made in different seasons."
        )
    _footnote(figure, note, minimum_lines=_PAIRED_SCATTER_FOOTNOTE_LINES)
    return figure


def _plot_temporal_coverage(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    coverage = _table(tables, "temporal_coverage")
    if coverage is None:
        return None
    frame = coverage.copy()
    frame["_source"] = _text_column(frame, "source_name")
    frame["_year"] = _numeric_column(frame, "year")
    frame["_count"] = _numeric_column(frame, "observation_count")
    frame = frame.loc[frame["_year"].notna() & frame["_count"].notna()]
    names = _ordered_sources(loaded, frame["_source"])
    if frame.empty or not names:
        return None

    pivot = (
        frame.pivot_table(
            index="_year", columns="_source", values="_count", aggfunc="sum"
        )
        .reindex(columns=list(names))
        .fillna(0.0)
        .sort_index()
    )
    years = pivot.index.to_numpy(dtype=float)
    colours = _source_colours(loaded)
    figure, axis = _figure(config)
    bottom = np.zeros(years.size, dtype=float)
    for name in names:
        values = pivot[name].to_numpy(dtype=float)
        axis.bar(
            years,
            values,
            bottom=bottom,
            width=0.86,
            color=colours[name],
            edgecolor="none",
            label=(
                f"{_legend_label(name)} — "
                f"{int(values.sum()):,} observations"
            ),
        )
        bottom += values

    # Make the era gap explicit rather than leaving the reader to notice an
    # empty stack: shade every year before the second dataset's first record.
    first_year = {
        name: float(years[pivot[name].to_numpy(dtype=float) > 0][0])
        for name in names
        if bool((pivot[name].to_numpy(dtype=float) > 0).any())
    }
    if len(first_year) >= 2:
        ordered = sorted(first_year.items(), key=lambda item: item[1])
        earliest_name, earliest_year = ordered[0]
        second_year = ordered[1][1]
        if second_year > earliest_year:
            axis.axvspan(
                earliest_year - 0.6,
                second_year - 0.5,
                color=_NEUTRAL,
                alpha=0.14,
                zorder=0,
            )
            # Sits below the upper-left legend, not behind it: both want the
            # clear space over the early years, and the legend is drawn last.
            axis.annotate(
                f"Only {earliest_name} records\nobservations before {second_year:.0f}",
                xy=((earliest_year + second_year) / 2.0, 0.70),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=8,
                bbox=_ANNOTATION_BOX,
            )

    # Headroom for the legend and that annotation to share the top of the panel
    # without either overlapping the tallest stack.
    tallest = float(np.max(bottom)) if bottom.size else 0.0
    if tallest > 0:
        axis.set_ylim(0.0, tallest * 1.45)

    axis.set_xlim(float(years.min()) - 1.0, float(years.max()) + 1.0)
    tick_start = int(np.floor(years.min() / 5.0) * 5)
    axis.set_xticks(np.arange(tick_start, int(years.max()) + 5, 5))
    axis.set(
        title="Observation coverage by recorded year of planting",
        xlabel="Recorded year of planting",
        ylabel="Harmonized observations recorded (count)",
    )
    axis.legend(fontsize=8, loc="upper left")
    axis.grid(axis="y", alpha=0.2)
    _footnote(
        figure,
        "Bars stack the datasets within each recorded year. Rows without a "
        "recorded year contribute no bar, so the totals are of dated "
        "observations only.",
    )
    return figure


@dataclass(frozen=True)
class _CompositionBar:
    """One dataset × context field, resolved to drawable shares."""

    source_name: str
    classification: str
    context_key: str
    context_title: str
    # The physical column the field was read from, as its header records it.
    column_header: str
    # ``_level`` / ``_count`` / ``_mean_yield``, count-descending.
    levels: pd.DataFrame
    denominator: float

    @property
    def reported_total(self) -> float:
        return float(self.levels["_count"].sum())

    @property
    def withheld_share(self) -> float:
        """Share in no drawn level: below the threshold, or no value recorded.

        The two are pooled deliberately. Reporting them apart would let a reader
        recover a withheld level's count by subtraction, which is the disclosure
        the reporting threshold exists to prevent.
        """

        return max(0.0, 1.0 - self.reported_total / self.denominator)


def _composition_levels(
    tables: Mapping[str, pd.DataFrame], contexts: Sequence[str]
) -> pd.DataFrame | None:
    """``context_composition`` narrowed to the named fields, typed for drawing."""

    composition = _table(tables, "context_composition")
    if composition is None:
        return None
    frame = composition.copy()
    frame["_source"] = _text_column(frame, "source_name")
    frame["_context"] = _text_column(frame, "context_label").str.lower()
    frame["_level"] = _text_column(frame, "level_value")
    frame["_header"] = _text_column(frame, "column_header")
    frame["_count"] = _numeric_column(frame, "observation_count")
    frame["_mean_yield"] = _numeric_column(frame, "mean_yield_t_ha")
    frame = frame.loc[frame["_context"].isin(set(contexts)) & frame["_count"].notna()]
    return None if frame.empty else frame


def _composition_bars(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
    contexts: Sequence[tuple[str, str]],
    source_names: Sequence[str] | None = None,
) -> list[_CompositionBar]:
    """Every (dataset, field) pair with at least one reportable level.

    Shared by every composition figure so that they cannot drift apart on the
    denominator, the threshold, or the level ordering — only on which datasets
    and fields they select, and on how the resulting shares are drawn.

    ``source_names`` restricts the datasets; the per-dataset figures pass one
    name, and the cross-dataset figures pass none. The denominator is a single
    dataset's own observation total either way, so a bar carries the same share
    in both.
    """

    frame = _composition_levels(tables, [key for key, _ in contexts])
    if frame is None:
        return []
    # This module re-applies the reporting threshold rather than trusting that
    # the producing table already did: a level below it may never be drawn or
    # labeled, and its observations belong in the withheld segment.
    reportable = frame.loc[frame["_count"] >= float(config.minimum_level_count)]
    if reportable.empty:
        return []

    # Denominator from the harmonized observation frame, which is the basis the
    # composition describes. Taking the maximum against the group's own total
    # keeps the bar honest if the producing table counted a wider row set: the
    # withheld segment then shrinks to zero instead of going negative.
    observations = _observations(loaded, config)
    source_totals = observations["source_name"].value_counts().to_dict()

    wanted_sources = None if source_names is None else set(source_names)
    bars: list[_CompositionBar] = []
    for source in loaded.sources:
        if wanted_sources is not None and source.source_name not in wanted_sources:
            continue
        for context_key, context_title in contexts:
            group = reportable.loc[
                (reportable["_source"] == source.source_name)
                & (reportable["_context"] == context_key)
            ]
            if group.empty:
                continue
            declared_total = float(
                frame.loc[
                    (frame["_source"] == source.source_name)
                    & (frame["_context"] == context_key),
                    "_count",
                ].sum()
            )
            denominator = max(
                declared_total, float(source_totals.get(source.source_name, 0.0))
            )
            if denominator <= 0:
                continue
            bars.append(
                _CompositionBar(
                    source_name=source.source_name,
                    classification=source.data_classification,
                    context_key=context_key,
                    context_title=context_title,
                    column_header=str(group["_header"].iloc[0]),
                    # The producing table's row order, not a re-sort here. It
                    # emits the recorded fields count-descending and the derived
                    # ordinal fields in their own order, so an applied-N ladder
                    # draws as a ladder while a season draws commonest-first.
                    # Re-sorting on count would scramble the ordinal case.
                    levels=group,
                    denominator=denominator,
                )
            )
    return bars


def _composition_level_colours(
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> dict[tuple[str, str], tuple[float, ...]]:
    """One colour per (context field, level), stable across every figure.

    Assigned from the whole composition table rather than from the subset a
    figure happens to draw. Six figures now draw overlapping slices of the same
    levels, and a level that changes colour between two of them reads as a
    different level. Level text is still the identifier — the label inside the
    segment — so a wrap of the 20-colour cycle costs recognition, not meaning.
    """

    composition = _table(tables, "context_composition")
    if composition is None:
        return {}
    present = set(_text_column(composition, "context_label").str.lower())
    frame = _composition_levels(tables, sorted(present))
    if frame is None:
        return {}
    frame = frame.loc[frame["_count"] >= float(config.minimum_level_count)]
    # Curated fields first, in the order the cross-dataset figures draw them,
    # then anything else the recipe binds, alphabetically. Datasets in profiling
    # order within a field, levels count-descending within a dataset: one bar's
    # levels therefore take consecutive colours, as they did when each figure
    # assigned its own.
    ordered_keys = [key for key in _CONTEXT_TITLES if key in present]
    ordered_keys += sorted(present - set(_CONTEXT_TITLES))
    colours: dict[tuple[str, str], tuple[float, ...]] = {}
    # Counted separately from ``colours`` so that a bar coloured from the ramp
    # below does not advance the categorical cycle: which of the two forms a bar
    # takes is a property of that bar and should not move every colour after it.
    allocated = 0
    for context_key in ordered_keys:
        if context_key in _DUPLICATE_ENCODING_CONTEXTS:
            # Never drawn, so never coloured — the same exclusion the per-dataset
            # plan applies, kept here so it does not consume the vocabulary.
            continue
        for source in loaded.sources:
            # Producer row order here too, so a field's levels take consecutive
            # entries of the vocabulary in the order they are drawn and the
            # neighbouring segments of one bar differ in hue.
            group = frame.loc[
                (frame["_context"] == context_key)
                & (frame["_source"] == source.source_name)
            ]
            # A field routed out of the stacked form draws no coloured segment
            # anywhere, so it takes no colours. Allocating for it would spend
            # the cycle on the 159 variety levels and push the fields that are
            # drawn into a wrap. The one overridden (source, field) pair is
            # drawn, so it is coloured here too, rather than falling back to
            # the per-figure ad hoc assignment in ``_draw_stacked_composition``.
            if (
                len(group) > _MAXIMUM_STACKED_LEVELS
                and (source.source_name, context_key)
                not in _STACKED_LEVEL_CAP_OVERRIDES
            ):
                continue
            if len(group) > len(_LEVEL_PALETTE):
                # One bar with more levels than the categorical vocabulary
                # holds: cycling would repeat a colour inside that single bar,
                # which is the one place reuse actually costs meaning. Sampled
                # across a continuous ramp instead, so neighbouring levels come
                # out close rather than identical — the most a bar of this many
                # levels can carry, and it leaves the cycle for the bars that
                # can use it.
                for level, colour in zip(
                    group["_level"], _oversized_field_colours(len(group))
                ):
                    colours.setdefault(
                        (context_key, str(level)), colour
                    )
                continue
            for level in group["_level"]:
                entry = (context_key, str(level))
                if entry not in colours:
                    colours[entry] = _LEVEL_PALETTE[
                        allocated % len(_LEVEL_PALETTE)
                    ]
                    allocated += 1
    return colours


def _undrawn_source_note(
    loaded: LoadedSources,
    *,
    drawn: Sequence[str],
    contexts: Sequence[str],
    config: DescriptiveStatisticsConfig,
) -> str:
    """Name every profiled dataset the figure could not draw, and say why.

    A composition figure that silently shows two of three datasets reads as a
    dropped dataset. Each of these figures covers fields that only some sources
    declare, so the absence is stated as the finding it is.
    """

    wanted = set(contexts)
    present = set(drawn)
    fragments: list[str] = []
    for source in loaded.sources:
        if source.source_name in present:
            continue
        specs = [
            source.column(binding.raw_column_id)
            for binding in source.binding.context
            if binding.label.strip().lower() in wanted
        ]
        if not specs:
            reason = "declares no such column"
        elif all(spec.suppressed for spec in specs):
            reason = "has that column withheld"
        elif all(spec.nonblank_count == 0 for spec in specs):
            reason = "declares the column but records no value in it"
        else:
            reason = (
                "records no level at least "
                f"{config.minimum_level_count} times"
            )
        fragments.append(f"{source.source_name} ({reason})")
    if not fragments:
        return ""
    return "No bar is drawn for " + "; ".join(fragments) + "."


def _draw_stacked_composition(
    *,
    bars: Sequence[_CompositionBar],
    config: DescriptiveStatisticsConfig,
    title: str,
    footnote: str,
    tick_labels: Sequence[str] | None = None,
    ylabel: str = "Context field and dataset",
    level_colours: Mapping[tuple[str, str], tuple[float, ...]] | None = None,
) -> plt.Figure:
    """The stacked share-of-dataset form shared by every low-cardinality figure.

    ``tick_labels`` defaults to naming the dataset on every row, which is what a
    cross-dataset figure needs and what a single-dataset figure would repeat on
    every row for no gain.
    """

    # Level vocabularies differ between datasets ("Dry"/"DS"), so colours carry
    # no shared meaning and the segments are labeled inline. Where the caller
    # supplies the run-wide map, a level keeps its colour across figures;
    # anything the map does not cover falls back to first appearance here.
    assigned: dict[tuple[str, str], tuple[float, ...]] = dict(level_colours or {})
    slots = max(len(bars) - 1 + 0.95, _MINIMUM_COMPOSITION_ROW_SLOTS)
    figure, axis = _figure(
        config,
        height_inches=_COMPOSITION_FIXED_INCHES + slots * _COMPOSITION_ROW_INCHES,
    )
    positions = np.arange(len(bars), dtype=float)
    any_withheld = False
    for position, bar in zip(positions, bars):
        left = 0.0
        for level, count in zip(bar.levels["_level"], bar.levels["_count"]):
            share = float(count) / bar.denominator
            key = (bar.context_key, str(level))
            if key not in assigned:
                assigned[key] = _LEVEL_PALETTE[len(assigned) % len(_LEVEL_PALETTE)]
            axis.barh(
                position,
                share * 100.0,
                left=left * 100.0,
                height=0.62,
                color=assigned[key],
                edgecolor="white",
                linewidth=0.6,
            )
            if share >= _MINIMUM_LABELED_SHARE:
                # Budget the label against the width of its own segment rather
                # than one fixed cut: a 7-point glyph is about 0.7% of the axis
                # wide, so a segment can hold well over one character per point
                # of share. Held to roughly one per point, which keeps a
                # comfortable margin and still spells out names that a flat
                # 14-character cut used to truncate mid-word.
                label = str(level)
                budget = max(_MINIMUM_LABEL_CHARACTERS, int(share * 100.0))
                text = label if len(label) <= budget else label[: budget - 1] + "…"
                axis.text(
                    (left + share / 2.0) * 100.0,
                    position,
                    f"{text}\n{share * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
            left += share
        remainder = max(0.0, 1.0 - left)
        if remainder > 0.0005:
            any_withheld = True
            axis.barh(
                position,
                remainder * 100.0,
                left=left * 100.0,
                height=0.62,
                color=_WITHHELD_GREY,
                edgecolor="white",
                linewidth=0.6,
                hatch="//",
            )

    axis.set_yticks(positions)
    axis.set_yticklabels(
        list(tick_labels)
        if tick_labels is not None
        else [
            f"{bar.context_title}\n{bar.source_name} ({bar.classification})"
            for bar in bars
        ],
        fontsize=8,
    )
    axis.invert_yaxis()
    # Clear band under the last bar: the withheld-segment legend has nowhere
    # else to go, since the bars themselves span the full width of the axes.
    #
    # The floor keeps a two-bar figure from ballooning its bars to a third of
    # the canvas each. A dataset that declares two context fields and one that
    # declares five then draw bars of comparable thickness, so thickness never
    # reads as a difference in the data.
    axis.set_ylim(slots, -0.55)
    axis.set_xlim(0.0, 100.0)
    axis.set(
        title=title,
        xlabel="Share of the dataset's harmonized observations (%)",
        ylabel=ylabel,
    )
    if any_withheld:
        axis.legend(
            handles=[
                Patch(
                    facecolor=_WITHHELD_GREY,
                    edgecolor="white",
                    hatch="//",
                    label=(
                        "Not shown: level recorded fewer than "
                        f"{config.minimum_level_count} times, or no value recorded"
                    ),
                )
            ],
            fontsize=8,
            loc="lower right",
        )
    axis.grid(axis="x", alpha=0.2)
    _footnote(figure, footnote)
    return figure


_COMPOSITION_FOOTNOTE = (
    "Segments narrower than "
    f"{_MINIMUM_LABELED_SHARE * 100:.0f}% are drawn but not labeled. Level "
    "vocabularies are dataset-specific and are reported as recorded, not "
    "mapped onto a common scheme."
)


def _plot_context_composition(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    bars = _composition_bars(
        tables=tables, loaded=loaded, config=config, contexts=_COMPOSITION_CONTEXTS
    )
    if not bars:
        return None
    return _draw_stacked_composition(
        bars=bars,
        config=config,
        title="Season and water-regime composition of the recorded observations",
        footnote=_COMPOSITION_FOOTNOTE,
        level_colours=_composition_level_colours(tables, loaded, config),
    )


def _plot_context_composition_site_and_management(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    bars = _composition_bars(
        tables=tables,
        loaded=loaded,
        config=config,
        contexts=_SITE_MANAGEMENT_CONTEXTS,
    )
    if not bars:
        return None
    absent = _undrawn_source_note(
        loaded,
        drawn=[bar.source_name for bar in bars],
        contexts=[key for key, _ in _SITE_MANAGEMENT_CONTEXTS],
        config=config,
    )
    return _draw_stacked_composition(
        bars=bars,
        config=config,
        title=(
            "Region, site, and crop-establishment composition of the "
            "recorded observations"
        ),
        footnote=(
            f"{_COMPOSITION_FOOTNOTE} These fields are declared per dataset and "
            f"are not comparable between them: a region is an administrative "
            f"unit of the source country, a site is one experiment station. "
            f"{absent}"
        ),
        level_colours=_composition_level_colours(tables, loaded, config),
    )


def _context_title(key: str) -> str:
    return _CONTEXT_TITLES.get(key, key.replace("_", " ").capitalize())


def _prioritize_dataset_contexts(
    source_name: str, contexts: Sequence[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Apply a source's evidence-led priorities without moving unranked rows."""

    ordered = list(contexts)
    priority = {
        key: position
        for position, key in enumerate(_DATASET_CONTEXT_PRIORITY.get(source_name, ()))
    }
    if not priority:
        return ordered
    return sorted(ordered, key=lambda context: priority.get(context[0], len(priority)))


def _dataset_context_plan(
    source: ProfiledSource,
    *,
    tables: Mapping[str, pd.DataFrame],
    config: DescriptiveStatisticsConfig,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Which of one dataset's context fields the stacked form can carry.

    Returns the drawable ``(key, title)`` pairs in the order the recipe binds
    them, and a plain-language note for every field left out. The notes are not
    diagnostics: a dataset's own composition figure that quietly omits a bound
    field would misreport that dataset's coverage, so the caption says what is
    missing and why.
    """

    keys = [binding.label.strip().lower() for binding in source.binding.context]
    frame = _composition_levels(
        tables, keys + [key for key, _, _ in _DERIVED_CONTEXTS]
    )
    drawable: list[tuple[str, str]] = []
    omitted: list[str] = []

    def drawn_levels(key: str) -> list[str]:
        if frame is None:
            return []
        selected = frame.loc[
            (frame["_source"] == source.source_name)
            & (frame["_context"] == key)
            & (frame["_count"] >= float(config.minimum_level_count))
        ]
        return [str(level) for level in selected["_level"]]

    def constant_note(title: str, levels: Sequence[str]) -> str:
        return (
            f"{title} (one recorded level, {levels[0]!r} on every observation)"
        )

    # Derived fields lead: the applied N rate is the treatment the rest of the
    # profile is about, and a reader scanning down should meet it before the
    # conditions it was applied under.
    for key, title, _ in _DERIVED_CONTEXTS:
        if frame is None:
            break
        levels = drawn_levels(key)
        if len(levels) >= _MINIMUM_STACKED_LEVELS:
            drawable.append((key, title))
        elif levels:
            # A dataset whose whole record falls in one band. Named for the same
            # reason as a constant bound column below, rather than dropped in
            # silence, because a reader cannot tell a band that did not apply
            # from one that applied to everything.
            omitted.append(constant_note(title, levels))
    for binding in source.binding.context:
        key = binding.label.strip().lower()
        title = _context_title(key)
        if key in _DUPLICATE_ENCODING_CONTEXTS:
            omitted.append(f"{title} (a second encoding of a field already shown)")
            continue
        spec = source.column(binding.raw_column_id)
        if spec.suppressed:
            omitted.append(f"{title} (withheld)")
            continue
        if spec.nonblank_count == 0:
            # Distinguished from the threshold case below: a column that is
            # blank on every row is a gap in the dataset, not a disclosure
            # control, and the two would otherwise read the same.
            omitted.append(f"{title} (the column records no value)")
            continue
        recorded = drawn_levels(key)
        levels = len(recorded)
        if levels == 0:
            omitted.append(
                f"{title} (no level recorded at least "
                f"{config.minimum_level_count} times)"
            )
        elif levels < _MINIMUM_STACKED_LEVELS:
            omitted.append(constant_note(title, recorded))
        elif (
            levels > _MAXIMUM_STACKED_LEVELS
            and (source.source_name, key) not in _STACKED_LEVEL_CAP_OVERRIDES
        ):
            omitted.append(
                f"{title} ({levels} recorded levels, too many for this form"
                + (
                    "; drawn in context_composition_variety"
                    if key == _VARIETY_CONTEXT
                    else ""
                )
                + ")"
            )
        else:
            drawable.append((key, title))
    return _prioritize_dataset_contexts(source.source_name, drawable), omitted


def _plot_one_context_composition(
    source_name: str,
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    """Every context field of one dataset, on that dataset's own denominator.

    The cross-dataset figures put one field side by side across datasets; this
    puts one dataset's fields side by side with each other. Same bars, same
    shares — a reader comparing seasons between datasets uses the former, and a
    reader asking what one dataset covers uses this.
    """

    matches = [
        source for source in loaded.sources if source.source_name == source_name
    ]
    if not matches:
        return None
    source = matches[0]
    contexts, omitted = _dataset_context_plan(source, tables=tables, config=config)
    if not contexts:
        return None
    bars = _composition_bars(
        tables=tables,
        loaded=loaded,
        config=config,
        contexts=contexts,
        source_names=(source_name,),
    )
    if not bars:
        return None
    observations = bars[0].denominator
    note = (
        " Context fields not drawn here: " + "; ".join(omitted) + "."
        if omitted
        else ""
    )
    drawn = {key for key, _ in contexts}
    parameters = _derivation_parameters(config)
    derivation = "".join(
        " " + caption.format(**parameters)
        for key, _, caption in _DERIVED_CONTEXTS
        if key in drawn
    )
    return _draw_stacked_composition(
        bars=bars,
        config=config,
        title=(
            f"Context composition of {source.source_name} "
            f"({source.data_classification}) — "
            f"{observations:,.0f} harmonized observations"
        ),
        footnote=(
            f"{_COMPOSITION_FOOTNOTE} Every bar is a share of this dataset's "
            "own harmonized observations, so the bars are comparable with each "
            "other and with this dataset's bars in the cross-dataset "
            f"composition figures.{derivation}{note}"
        ),
        # The source column under each field name, except where the recipe
        # bound a field to a column of the same name and the second line would
        # only repeat the first.
        tick_labels=[
            bar.context_title
            if bar.column_header.strip().lower() == bar.context_title.strip().lower()
            else f"{bar.context_title}\n{bar.column_header}"
            for bar in bars
        ],
        ylabel="Context field, and the column it was read from",
        level_colours=_composition_level_colours(tables, loaded, config),
    )


def _context_composition_builder(
    source_name: str,
) -> Callable[..., "plt.Figure | None"]:
    """Bind one source name into the shared single-dataset builder."""

    def builder(
        *,
        tables: Mapping[str, pd.DataFrame],
        loaded: LoadedSources,
        config: DescriptiveStatisticsConfig,
    ) -> plt.Figure | None:
        return _plot_one_context_composition(
            source_name, tables=tables, loaded=loaded, config=config
        )

    builder.__name__ = f"_plot_context_composition_{source_name}"
    return builder


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Pooled mean of per-level means, weighted by the level counts.

    Exact rather than approximate: every level mean in ``context_composition``
    is an unweighted mean over its own ``observation_count`` rows, so weighting
    by that count reproduces the mean of the pooled rows.
    """

    numeric = pd.to_numeric(values, errors="coerce")
    mass = pd.to_numeric(weights, errors="coerce")
    usable = numeric.notna() & mass.notna() & (mass > 0)
    if not bool(usable.any()):
        return float("nan")
    return float((numeric[usable] * mass[usable]).sum() / mass[usable].sum())


def _variety_entries(
    bar: _CompositionBar, colour: str
) -> list[tuple[str, float, str, str | None, str]]:
    """One panel's bars: the named varieties, then the two pooled remainders.

    The remainders are kept apart because they are not the same statement. The
    pooled-variety bar is this figure's own display choice and could be undone
    by drawing more bars; the withheld bar is the reporting threshold, and no
    figure may undo it. Merging them would present a disclosure control as a
    layout decision.
    """

    named = bar.levels.head(_VARIETY_LEVELS_PER_PANEL)
    further = bar.levels.iloc[_VARIETY_LEVELS_PER_PANEL:]
    entries: list[tuple[str, float, str, str | None, str]] = []
    for level, count, mean_yield in zip(
        named["_level"], named["_count"], named["_mean_yield"]
    ):
        label = str(level)
        entries.append(
            (
                label if len(label) <= 21 else label[:20] + "…",
                float(count) / bar.denominator,
                colour,
                None,
                _variety_annotation(float(count), float(mean_yield)),
            )
        )
    if not further.empty:
        pooled = float(further["_count"].sum())
        entries.append(
            (
                f"{len(further)} further varieties",
                pooled / bar.denominator,
                _NEUTRAL,
                None,
                _variety_annotation(
                    pooled, _weighted_mean(further["_mean_yield"], further["_count"])
                ),
            )
        )
    withheld = bar.withheld_share
    if withheld > 0.0005:
        entries.append(
            (
                "Rare or unrecorded",
                withheld,
                _WITHHELD_GREY,
                "//",
                f"{withheld * bar.denominator:,.0f} obs",
            )
        )
    return entries


def _variety_annotation(count: float, mean_yield: float) -> str:
    if not np.isfinite(mean_yield):
        return f"{count:,.0f} obs"
    return f"{count:,.0f} obs · {mean_yield:.2f} t ha⁻¹"


def _draw_variety_panel(
    axis: plt.Axes,
    *,
    bar: _CompositionBar,
    entries: Sequence[tuple[str, float, str, str | None, str]],
    limit: float,
) -> None:
    positions = np.arange(len(entries), dtype=float)
    for position, (label, share, colour, hatch, annotation) in zip(positions, entries):
        width = share * 100.0
        axis.barh(
            position,
            width,
            height=0.68,
            color=colour,
            edgecolor="white",
            linewidth=0.6,
            hatch=hatch,
        )
        # A bar wide enough to hold its own annotation carries it inside; the
        # alternative is an axis padded out to the widest label, which would
        # squeeze every short bar into the left margin.
        inside = width >= limit * 0.55
        axis.annotate(
            annotation,
            xy=(width, position),
            xytext=(-4 if inside else 4, 0),
            textcoords="offset points",
            ha="right" if inside else "left",
            va="center",
            fontsize=6.5,
            color="white" if inside else "black",
        )
    axis.set_yticks(positions)
    axis.set_yticklabels([label for label, *_ in entries], fontsize=7)
    axis.invert_yaxis()
    axis.set_xlim(0.0, limit)
    axis.set_ylim(len(entries) - 0.4, -0.6)
    axis.set_title(
        f"{bar.source_name} ({bar.classification})\n"
        f"{bar.denominator:,.0f} harmonized observations",
        fontsize=9,
    )
    axis.set_xlabel("Share of the dataset's observations (%)", fontsize=8)
    axis.tick_params(axis="x", labelsize=8)
    axis.grid(axis="x", alpha=0.2)


def _plot_context_composition_variety(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    bars = _composition_bars(
        tables=tables,
        loaded=loaded,
        config=config,
        contexts=((_VARIETY_CONTEXT, "Variety"),),
    )
    if not bars:
        return None
    colours = _source_colours(loaded)
    panels = [(bar, _variety_entries(bar, colours[bar.source_name])) for bar in bars]
    # One x scale across the panels, so a bar of the same length means the same
    # share in both. Padded past the widest bar to leave room for the outside
    # annotations of the short ones.
    widest = max(
        (share for _, entries in panels for _, share, *_ in entries), default=0.0
    )
    limit = max(widest * 100.0 * 1.32, 10.0)

    figure, axes = _panel_figure(config, 1, len(panels))
    for axis, (bar, entries) in zip(axes[0], panels):
        _draw_variety_panel(axis, bar=bar, entries=entries, limit=limit)
    figure.suptitle(
        "Varieties most often recorded in each dataset", fontsize=12
    )
    absent = _undrawn_source_note(
        loaded,
        drawn=[bar.source_name for bar in bars],
        contexts=[_VARIETY_CONTEXT],
        config=config,
    )
    _footnote(
        figure,
        f"Each panel holds the {_VARIETY_LEVELS_PER_PANEL} varieties with the "
        "most recorded observations in that dataset, ordered by observation "
        "count and never by yield: a record of what was grown, not a variety "
        "comparison. Names are counted exactly as recorded, so spelling "
        "variants of one variety appear as separate bars. Grey pools every "
        f"further variety recorded at least {config.minimum_level_count} "
        "times; the hatched bar holds observations whose variety was recorded "
        f"fewer than {config.minimum_level_count} times or not recorded at "
        f"all. Means are recorded means over unequal N rates, years, and "
        f"designs. {absent}",
        top=0.95,
    )
    return figure


# --------------------------------------------------------------------------
# crosscut
# --------------------------------------------------------------------------


def _range_panel(
    axis: plt.Axes,
    *,
    names: Sequence[str],
    low: pd.Series,
    high: pd.Series,
    middle: pd.Series | None,
    colours: Mapping[str, str],
    title: str,
    xlabel: str,
    value_format: str,
) -> None:
    positions = np.arange(len(names), dtype=float)
    for position, name in zip(positions, names):
        start = float(low.get(name, np.nan))
        stop = float(high.get(name, np.nan))
        if not (np.isfinite(start) and np.isfinite(stop)):
            continue
        axis.barh(
            position,
            max(stop - start, 0.0),
            left=start,
            height=0.5,
            color=colours[name],
            alpha=0.55,
            edgecolor=colours[name],
        )
        axis.annotate(
            format(start, value_format),
            xy=(start, position),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=7,
        )
        axis.annotate(
            format(stop, value_format),
            xy=(stop, position),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7,
        )
        if middle is not None:
            centre = float(middle.get(name, np.nan))
            if np.isfinite(centre):
                axis.plot(
                    [centre],
                    [position],
                    marker="D",
                    markersize=5,
                    color="black",
                    zorder=4,
                )
    # The low endpoint is annotated to the LEFT of the bar, so a bar that starts
    # at the data minimum would push its label outside the axes and on top of the
    # y tick label. Reserve margin for both endpoint labels rather than moving
    # them inside the bar, where a short bar would collide with its own high label.
    finite = np.array(
        [
            value
            for name in names
            for value in (float(low.get(name, np.nan)), float(high.get(name, np.nan)))
            if np.isfinite(value)
        ],
        dtype=float,
    )
    if finite.size:
        lowest, highest = float(finite.min()), float(finite.max())
        span = highest - lowest
        pad = span * 0.14 if span > 0 else max(abs(highest) * 0.14, 1.0)
        axis.set_xlim(lowest - pad, highest + pad)

    axis.set_yticks(positions)
    axis.set_yticklabels(list(names), fontsize=8)
    axis.invert_yaxis()
    axis.set_title(title, fontsize=10)
    axis.set_xlabel(xlabel, fontsize=8)
    axis.tick_params(axis="x", labelsize=8)
    axis.grid(axis="x", alpha=0.2)


def _plot_source_comparability(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    comparability = _table(tables, "source_comparability")
    if comparability is None:
        return None
    frame = comparability.copy()
    frame["_source"] = _text_column(frame, "source_name")
    names = _ordered_sources(loaded, frame["_source"])
    if not names:
        return None
    indexed = frame.set_index("_source")

    def column(name: str) -> pd.Series:
        return _numeric_column(indexed, name).reindex(names)

    colours = _source_colours(loaded)
    figure, axes = _panel_figure(config, 2, 2)
    _range_panel(
        axes[0, 0],
        names=names,
        low=column("n_rate_min_kg_ha"),
        high=column("n_rate_max_kg_ha"),
        middle=None,
        colours=colours,
        title="Recorded inorganic N range",
        xlabel="Inorganic N rate (kg N ha⁻¹)",
        value_format=".0f",
    )
    _range_panel(
        axes[0, 1],
        names=names,
        low=column("yield_min_t_ha"),
        high=column("yield_max_t_ha"),
        middle=column("yield_median_t_ha"),
        colours=colours,
        title="Recorded grain-yield range (♦ = median)",
        xlabel="Grain yield (t ha⁻¹)",
        value_format=".2f",
    )
    _range_panel(
        axes[1, 0],
        names=names,
        low=column("year_min"),
        high=column("year_max"),
        middle=None,
        colours=colours,
        title="Recorded year span",
        xlabel="Recorded year of planting",
        value_format=".0f",
    )

    counts = column("harmonized_observation_count")
    series_counts = column("grouping_series_count")
    axis = axes[1, 1]
    positions = np.arange(len(names), dtype=float)
    for position, name in zip(positions, names):
        value = float(counts.get(name, np.nan))
        if not np.isfinite(value) or value <= 0:
            continue
        axis.barh(position, value, height=0.5, color=colours[name], alpha=0.75)
        series_value = float(series_counts.get(name, np.nan))
        suffix = (
            f" · {int(series_value):,} series" if np.isfinite(series_value) else ""
        )
        axis.annotate(
            f"{int(value):,}{suffix}",
            xy=(value, position),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7,
        )
    axis.set_xscale("log")
    finite_counts = counts.to_numpy(dtype=float)
    finite_counts = finite_counts[np.isfinite(finite_counts) & (finite_counts > 0)]
    if finite_counts.size:
        # Wide right margin: each bar carries its count and series total as
        # text, which needs room beyond the longest bar.
        axis.set_xlim(1.0, float(finite_counts.max()) * 60.0)
    axis.set_yticks(positions)
    axis.set_yticklabels(list(names), fontsize=8)
    axis.invert_yaxis()
    axis.set_title("Harmonized observations (logarithmic)", fontsize=10)
    axis.set_xlabel("Observations carrying both N rate and yield (count)", fontsize=8)
    axis.tick_params(axis="x", labelsize=8)
    axis.grid(axis="x", alpha=0.2, which="both")

    figure.suptitle(
        "The registered datasets side by side on common axes", fontsize=12
    )
    classification_note = "; ".join(
        _legend_label(name) for name in names
    )
    _footnote(
        figure,
        f"{classification_note}. Ranges are the extremes recorded, not "
        "tolerance intervals. The panels describe what each dataset covers; "
        "they do not rank the datasets or assert commensurability.",
        top=0.95,
    )
    return figure


def _plot_nitrogen_ladder_geometry(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    observations = _observations(loaded, config)
    if observations.empty:
        return None
    resolved = observations.loc[observations["is_series_resolved"].astype(bool)]
    if resolved.empty:
        return None
    names = _ordered_sources(loaded, resolved["source_name"])
    if not names:
        return None

    # Bars and their annotations share one provenance: both the distribution and
    # the median quoted in the legend come from this recomputation, so the panel
    # cannot contradict itself.
    distributions: dict[str, pd.Series] = {}
    for name in names:
        levels = _distinct_level_counts(
            resolved.loc[resolved["source_name"] == name],
            tolerance_kg_ha=config.n_level_tolerance_kg_ha,
        )
        if not levels.empty:
            distributions[name] = levels
    if not distributions:
        return None

    maximum_levels = int(
        max(int(levels.max()) for levels in distributions.values())
    )
    cap = min(maximum_levels, 12)
    buckets = np.arange(1, cap + 1)
    colours = _source_colours(loaded)
    figure, axis = _figure(config)
    width = 0.8 / max(len(distributions), 1)
    for index, (name, levels) in enumerate(distributions.items()):
        capped = levels.clip(upper=cap)
        shares = np.array(
            [float((capped == bucket).sum()) / len(capped) * 100.0 for bucket in buckets]
        )
        offset = (index - (len(distributions) - 1) / 2.0) * width
        axis.bar(
            buckets + offset,
            shares,
            width=width,
            color=colours[name],
            edgecolor=colours[name],
            alpha=0.85,
            label=(
                f"{_legend_label(name)} — "
                f"{len(levels):,} series, median {float(levels.median()):g} "
                + ("level" if float(levels.median()) == 1.0 else "levels")
            ),
        )
        for bucket, share in zip(buckets, shares):
            # A sliver rounds to "0%", which reads as an absent bar rather than
            # a rare one; leave it to the bar itself.
            if share < 0.5:
                continue
            axis.annotate(
                f"{share:.0f}%",
                xy=(bucket + offset, share),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )

    axis.set_xticks(buckets)
    labels = [str(bucket) for bucket in buckets]
    if maximum_levels > cap:
        labels[-1] = f"{cap}+"
    axis.set_xticklabels(labels)
    axis.set_ylim(0.0, 108.0)
    axis.set(
        title="N-ladder geometry: distinct inorganic N levels recorded per grouping series",
        xlabel="Distinct N levels in the series (count)",
        ylabel="Share of the dataset's series (%)",
    )
    axis.legend(fontsize=8, loc="upper right")
    axis.grid(axis="y", alpha=0.2)
    key_note = "; ".join(
        f"{source.source_name} = "
        + (
            " | ".join(binding.header for binding in source.binding.series)
            or "no series key declared"
        )
        for source in loaded.sources
        if source.source_name in distributions
    )
    _footnote(
        figure,
        f"Series keys — {key_note}. Rates within "
        f"{config.n_level_tolerance_kg_ha:g} kg N ha⁻¹ count as one level, and "
        "only series whose key components are all recorded are counted.",
    )
    return figure


# --------------------------------------------------------------------------
# Registry and dispatch
# --------------------------------------------------------------------------


FIGURE_BUILDERS: Mapping[str, Callable[..., "plt.Figure | None"]] = {
    "dataset_scale": _plot_dataset_scale,
    "column_fill_profile": _plot_column_fill_profile,
    "numeric_spread_overview": _plot_numeric_spread_overview,
    "categorical_cardinality": _plot_categorical_cardinality,
    "nitrogen_rate_distribution": _plot_nitrogen_rate_distribution,
    "yield_distribution": _plot_yield_distribution,
    **{
        spec.name: _yield_distribution_builder(spec.source_name)
        for spec in FIGURE_SPECS
        if spec.name.startswith("yield_distribution_") and spec.source_name
    },
    "yield_versus_nitrogen": _plot_yield_versus_nitrogen,
    "yield_versus_nitrogen_trajectories": _plot_yield_versus_nitrogen_trajectories,
    "temporal_coverage": _plot_temporal_coverage,
    "context_composition": _plot_context_composition,
    "context_composition_site_and_management": (
        _plot_context_composition_site_and_management
    ),
    "context_composition_variety": _plot_context_composition_variety,
    **{
        spec.name: _context_composition_builder(spec.source_name)
        for spec in FIGURE_SPECS
        if spec.name.startswith("context_composition_") and spec.source_name
    },
    "source_comparability": _plot_source_comparability,
    "nitrogen_ladder_geometry": _plot_nitrogen_ladder_geometry,
}


# A missing or misspelled builder must fail at import, not when the bundle is
# already half written — the same rule the contract module applies to its specs.
_DECLARED_FIGURES = {spec.name for spec in FIGURE_SPECS}
if set(FIGURE_BUILDERS) != _DECLARED_FIGURES:
    raise ProfileContractError(
        "Figure builders do not cover the declared figures: "
        f"missing {sorted(_DECLARED_FIGURES - set(FIGURE_BUILDERS))}, "
        f"undeclared {sorted(set(FIGURE_BUILDERS) - _DECLARED_FIGURES)}"
    )


def build_figure(
    name: str,
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> "plt.Figure | None":
    """Render one declared figure, or ``None`` when its inputs cannot support it.

    The figure is returned open: the caller owns saving it once per configured
    format at the configured DPI, and closing it.
    """

    builder = FIGURE_BUILDERS.get(name)
    if builder is None:
        raise ProfileContractError(f"Figure is not declared in the contract: {name!r}")
    return builder(tables=tables, loaded=loaded, config=config)
