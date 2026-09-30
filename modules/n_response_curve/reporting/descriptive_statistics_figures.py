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
from ..analysis.descriptive_statistics.agronomic import (
    APPLIED_N_BAND_CONTEXT,
    CROP_ESTABLISHMENT_CONTEXT,
    CROP_ESTABLISHMENT_LEVELS,
    STRAW_MANAGEMENT_CONTEXT,
    STRAW_MANAGEMENT_LEVELS,
    YEAR_BAND_CONTEXT,
    _aligned_row_positions,
    _temporal_rows,
    build_context_composition,
)
from ..analysis.descriptive_statistics.contracts import (
    FIGURE_SPECS,
    WITH_FARMERS_PRACTICE_SUFFIX,
    ProfileContractError,
    conform_table,
)
from ..analysis.descriptive_statistics.sources import (
    NATIVE_T_HA,
    OBSERVATION_COLUMNS,
    ColumnBinding,
    LoadedSources,
    ProfiledSource,
    build_all_observations,
)
from .source_display_names import display_source_name


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
_MISSING_GREY = "#C7C7C7"
_RARE_GREY = "#858585"
_MINIMUM_MISSING_LABELED_SHARE = 0.05

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
# its two greys, which are reserved for missing, rare, and pooled residual
# segments. A grey recorded-level segment beside those residuals would invite
# reading an ordinary category as missing data. The vocabulary continues with
# tab20b and tab20c — the latter without its fourth family, which is also grey.
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
        "datasets that overlap in time carry the same bands. Nonblank values "
        "that do not encode one finite numeric year are shown as recorded; only "
        "blank source cells are shown as Missing.",
    ),
)

# Unlike the numeric bands above, this field is a conservative harmonization of
# several configured evidence columns. It remains separate so the caption never
# calls a standardized category an as-recorded source level.
_STANDARDIZED_CONTEXTS: tuple[tuple[str, str, str], ...] = (
    (
        CROP_ESTABLISHMENT_CONTEXT,
        "Crop establishment",
        "Crop establishment is standardized from Transplanting Date: explicit "
        "direct-seeded entries are Direct seeded, a recorded transplanting "
        "date is Transplanted, 'not stated' is retained, and blank cells are "
        "Missing.",
    ),
    (
        STRAW_MANAGEMENT_CONTEXT,
        "Straw management",
        "Straw management is conservatively standardized from the configured "
        "source evidence. Retained/returned is not reclassified as "
        "Incorporated; Burned, Removed, and Incorporated require explicit "
        "straw-specific evidence. Fate unspecified means straw was recorded "
        "as an amendment but its disposition was not.",
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
    + tuple((key, title) for key, title, _ in _STANDARDIZED_CONTEXTS)
)

# Excluded from every stacked figure by name, not by measurement: this is
# ph_combined_nopt_rcm's second, numeric encoding of a field it already records
# in words, so drawing both would show one dataset's season twice.
_DUPLICATE_ENCODING_CONTEXTS = frozenset({"season_coded"})

# Each dataset's modifier panel can rank only a subset of the rows in this
# composition form. The governed core-trial panel ranks planting year ahead of
# its applied-N ladder summaries; the ungoverned LTCCE panel ranks mean tested N
# and N-ladder span ahead of planting year. Zero-N yield is an outcome, and the
# other screened quantities have no composition row. Keep every unranked context
# in recipe order after the mapped rows. These priorities are source-specific
# because each modifier ranking describes a different response-series population.
_DATASET_CONTEXT_PRIORITY: Mapping[str, tuple[str, ...]] = {
    "core_trial_data": (
        "year_band",
        "applied_n_band",
        "water_regime",
        "season",
        CROP_ESTABLISHMENT_CONTEXT,
        "soil_type",
    ),
    "ltcce": ("applied_n_band", "year_band"),
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
# also pull unrelated high-cardinality fields into other panels. These entries
# are explicit requests to show the named source field despite its long recorded
# vocabulary; the continuous colour ramp keeps adjacent levels distinct.
_STACKED_LEVEL_CAP_OVERRIDES: frozenset[tuple[str, str]] = frozenset(
    {
        ("core_trial_data", "soil_type"),
        ("ltcce", _VARIETY_CONTEXT),
    }
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


def _extended_callout_slots(count: int) -> tuple[float, ...]:
    """The callout ladder continued at its own step, for a derived series.

    Only a panel that draws a derived arm beside the recorded datasets calls
    this. The governed panels keep ``_YIELD_CALLOUT_SLOTS`` unextended, so a
    fourth *registered* source still raises there rather than acquiring a slot
    nothing reviewed.
    """

    slots = list(_YIELD_CALLOUT_SLOTS)
    if len(slots) < 2:
        return tuple(slots)
    step = slots[0] - slots[1]
    while len(slots) < count:
        slots.append(slots[-1] - step)
    return tuple(slots)


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
    """The dataset's reader-facing name alone.

    Figures deliberately do not print the data classification beside the name.
    Classification is recorded per artifact in ``run_manifest.json``, which is
    the binding record; repeating it on every legend entry only crowded the
    panels, and its absence here changes no governance behaviour — restricted
    sources are still labeled, counted, and suppressed exactly as before.

    The registered ``source_name`` remains the identifier in every table,
    filename, and manifest entry; only rendered text is renamed.
    """

    return display_source_name(source_name)


def _tick_label(source_name: str) -> str:
    return display_source_name(source_name)


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


# --------------------------------------------------------------------------
# agronomic
# --------------------------------------------------------------------------


# ph_combined_nopt_rcm records four N treatments per row, each in its own column
# pair (see [agronomic.ph_combined_nopt_rcm] for the list its registered schema
# evidence declares). ``nitrogen_rate`` binds NOPT-N, the recommended rate; this
# figure draws the farmer's-practice arm instead, because a recorded-rate
# distribution is a statement about what was applied in the field, and FP-N is
# the arm the farmer chose. core_trial_data's own "Type of Experiment/
# Study/Trial" column names 12 of its rows Farmer's Practice comparisons rather
# than a designed dose ladder, so they join that series rather than reading as
# rungs of a literature-extracted ladder.
#
# The governed NOPT series and the farmer-applied arm are intentionally both
# present: they are sibling treatments recorded on the same physical rows, not
# duplicate observations within one treatment. Side-by-side bars keep that
# treatment distinction explicit.
#
# The appended arm and the row reassignment are local to this figure.
# ``build_all_observations`` still returns exactly one row per recorded record
# through the governed binding, so nitrogen_rate_profile.csv and
# context_composition.csv are unchanged and still report core_trial_data's
# recorded Type of Experiment mix in full.
_CORE_TRIAL_SOURCE_NAME = "core_trial_data"
_CORE_TRIAL_EXPERIMENT_TYPE_LABEL = "experiment_type"
_CORE_TRIAL_FARMERS_PRACTICE_LEVEL = "Farmer's Practice"
_FARMERS_PRACTICE_SUFFIX = "_fp"
# Owned by the contract, which uses the same token in the published filenames
# of the panels that draw the arm. One definition, so a figure name and the
# filename beside it cannot disagree about what "with FP" is called.
_WITH_FARMERS_PRACTICE_FIGURE_SUFFIX = WITH_FARMERS_PRACTICE_SUFFIX
# The source whose farmer's-practice series adopts core_trial_data's recorded
# Farmer's Practice rows. Named rather than inferred: a second source binding an
# arm would get its own series, and the 12 rows would still belong to this one.
_FARMERS_PRACTICE_PARENT_SOURCE = "ph_combined_nopt_rcm"
_FARMERS_PRACTICE_SERIES_NAME = (
    _FARMERS_PRACTICE_PARENT_SOURCE + _FARMERS_PRACTICE_SUFFIX
)
_FARMERS_PRACTICE_DISPLAY_LABEL = "Farmer's Practice"
# The parent source records four treatments. The histogram draws NOPT-N and
# Farmer's Practice separately; this note names the remaining treatment whose
# N rate is not represented so neither series reads as the source's whole
# record.
def _ph_combined_treatment_note(mark: str = "bars") -> str:
    """Name the parent source's undrawn treatment, in the panel's own noun.

    One sentence, one set of numbers, whatever the mark: a scatter says
    "points" where a histogram says "bars", and two hand-written copies of a
    governance disclosure would be free to drift apart on everything else.
    """

    return (
        f"The separate ph_combined_nopt_rcm {mark} retain the dataset's NOPT-N "
        "binding (70-150 kg N ha⁻¹, the recommended rate the rest of the "
        "bundle profiles); its RCM-N treatment (37-189 kg N ha⁻¹) is not "
        "drawn."
    )


_PH_COMBINED_TREATMENT_NOTE = _ph_combined_treatment_note()
_ORIGINAL_UNDRAWN_ARM_NOTE = (
    "Its NOPT-N binding (70-150 kg N ha⁻¹, the recommended rate the rest of "
    "the bundle profiles) and its RCM-N treatment (37-189 kg N ha⁻¹) are not "
    "drawn here."
)


@dataclass(frozen=True)
class _FarmersPracticeSeries:
    """The farmer's-practice series and an account of what composes it.

    Every count travels with the frame so the caption can state the composition
    rather than let an appended arm and a reassignment pass as if the series
    were the source's own governed rows.
    """

    observations: pd.DataFrame
    relabeled_rows: int
    arm_rows: int
    arm_source_names: tuple[str, ...]

    @property
    def exists(self) -> bool:
        return self.relabeled_rows > 0 or self.arm_rows > 0


def _core_trial_farmers_practice_index(
    loaded: LoadedSources, observations: pd.DataFrame
) -> pd.Index:
    """Rows of ``observations`` that core_trial_data records as Farmer's Practice."""

    empty = observations.index[:0]
    core = next(
        (
            source
            for source in loaded.sources
            if source.source_name == _CORE_TRIAL_SOURCE_NAME
        ),
        None,
    )
    is_core = observations["source_name"] == _CORE_TRIAL_SOURCE_NAME
    if core is None or not is_core.any():
        return empty
    binding = next(
        (
            candidate
            for candidate in core.binding.context
            if candidate.label == _CORE_TRIAL_EXPERIMENT_TYPE_LABEL
        ),
        None,
    )
    if binding is None:
        return empty

    core_index = observations.index[is_core]
    lookup = pd.Series(
        np.arange(core.data_row_count), index=pd.Index(core.source_row_numbers)
    )
    positions = (
        observations.loc[core_index, "source_row_number"]
        .map(lookup)
        .to_numpy(dtype=np.int64)
    )
    levels = (
        core.text[binding.raw_column_id]
        .astype(str)
        .str.strip()
        .to_numpy()[positions]
    )
    return core_index[levels == _CORE_TRIAL_FARMERS_PRACTICE_LEVEL]


def _farmers_practice_arm(
    source: ProfiledSource, *, zero_n_tolerance_kg_ha: float
) -> pd.DataFrame | None:
    """One source's farmer's-practice arm, read from its sibling column pair.

    Held to the same admission rule as every other observation — a finite rate
    and a finite yield — so the series counts rows the way the recorded series
    do. Returns ``None`` when the source binds no such arm, which is the normal
    case: only a dataset that crosswalks a farmer's treatment has one.
    """

    binding = source.binding
    rate_binding = binding.farmers_practice_n_rate
    yield_binding = binding.farmers_practice_yield_t_ha
    if rate_binding is None or yield_binding is None:
        return None

    n_rate = source.numeric_series(rate_binding).astype(float)
    yield_t_ha = source.numeric_series(yield_binding).astype(float)
    if binding.year is not None:
        year = source.numeric_series(binding.year).astype(float)
    else:
        year = pd.Series(np.nan, index=n_rate.index, dtype=float)
    series_name = source.source_name + _FARMERS_PRACTICE_SUFFIX
    frame = pd.DataFrame(
        {
            "source_name": series_name,
            "data_classification": source.data_classification,
            "source_row_number": list(source.source_row_numbers),
            "study_key": "",
            "trial_key": "",
            "series_key": f"{series_name}::{source.source_name}",
            "year": year,
            "n_rate_kg_ha": n_rate,
            "yield_t_ha": yield_t_ha,
            "yield_unit_lineage": NATIVE_T_HA,
            "is_zero_n": n_rate.abs() <= zero_n_tolerance_kg_ha,
            # A farmer-chosen rate is not a rung of a resolved ladder.
            "is_series_resolved": False,
        },
        index=n_rate.index,
    )
    retained = frame.loc[frame["n_rate_kg_ha"].notna() & frame["yield_t_ha"].notna()]
    return retained.loc[:, list(OBSERVATION_COLUMNS)].reset_index(drop=True)


def _farmers_practice_arm_for_source(
    loaded: LoadedSources,
    source_name: str,
    *,
    zero_n_tolerance_kg_ha: float,
) -> tuple[ProfiledSource, pd.DataFrame] | None:
    """Resolve one declared Farmer's Practice arm without inventing a source."""

    source = next(
        (candidate for candidate in loaded.sources if candidate.source_name == source_name),
        None,
    )
    if source is None:
        return None
    arm = _farmers_practice_arm(
        source, zero_n_tolerance_kg_ha=zero_n_tolerance_kg_ha
    )
    if arm is None or arm.empty:
        return None
    return source, arm


def _figure_uses_farmers_practice(name: str) -> bool:
    """Whether a declared figure name selects the Farmer's Practice arm."""

    return name.endswith(_WITH_FARMERS_PRACTICE_FIGURE_SUFFIX)


def _farmers_practice_series(
    loaded: LoadedSources,
    observations: pd.DataFrame,
    config: DescriptiveStatisticsConfig,
) -> _FarmersPracticeSeries:
    """Append every recorded farmer's-practice rate as a separate series."""

    arms: list[pd.DataFrame] = []
    arm_source_names: list[str] = []
    for source in loaded.sources:
        arm = _farmers_practice_arm(
            source, zero_n_tolerance_kg_ha=config.zero_n_tolerance_kg_ha
        )
        if arm is None or arm.empty:
            continue
        arms.append(arm)
        arm_source_names.append(source.source_name)

    relabeled = _core_trial_farmers_practice_index(loaded, observations)
    if _FARMERS_PRACTICE_PARENT_SOURCE not in arm_source_names:
        # Nothing to adopt the rows; leave them where they were recorded rather
        # than open a series named for an arm this run did not read.
        relabeled = observations.index[:0]

    regrouped = observations
    if len(relabeled) or arms:
        regrouped = observations.copy()
    if len(relabeled):
        regrouped.loc[relabeled, "source_name"] = _FARMERS_PRACTICE_SERIES_NAME
    if arms:
        # The governed source rows retain their recommended-rate binding. The
        # farmer-applied rate is a sibling treatment and is therefore appended
        # as its own bar series rather than substituted for that source.
        regrouped = pd.concat([regrouped, *arms], ignore_index=True)

    return _FarmersPracticeSeries(
        observations=regrouped,
        relabeled_rows=int(len(relabeled)),
        arm_rows=int(sum(len(arm) for arm in arms)),
        arm_source_names=tuple(arm_source_names),
    )


def _farmers_practice_composition(derived: _FarmersPracticeSeries) -> str:
    """How the derived series was assembled, for the captions that draw it.

    A series read from an arm is only readable if the reader is told which arm
    it is and which rows were moved into it, so every panel that draws one
    states the same composition in the same words.
    """

    return (
        "Farmer's Practice (FP) is drawn as its own series: "
        f"{derived.arm_rows:,} FP observations were read from the "
        f"{_FARMERS_PRACTICE_PARENT_SOURCE} sibling column pair, and "
        f"{derived.relabeled_rows:,} literature-extracted rows recorded as "
        "Farmer's Practice were moved into the same series."
    )


def _single_rate_arm_series_keys(
    observations: pd.DataFrame, derived: _FarmersPracticeSeries
) -> pd.DataFrame:
    """Key every appended farmer's-practice row as its own single-rate series.

    ``_farmers_practice_arm`` gives the whole arm one key, which is right for a
    panel that only counts or bins its rows. A trajectory join groups by that
    key, and one key across hundreds of farmer-chosen rates would draw a
    polyline through unrelated trials, so each appended row is keyed by the
    record it came from: one trial, one farmer treatment, one point.

    Rows *relabeled* into the series keep the key they were recorded under.
    They are literature rows that carry a real series identity, and rewriting
    it would discard evidence rather than correct a placeholder.
    """

    keyed = observations.copy()
    row_numbers = keyed["source_row_number"].astype(str)
    for parent in derived.arm_source_names:
        series_name = parent + _FARMERS_PRACTICE_SUFFIX
        placeholder = f"{series_name}::{parent}"
        is_arm = keyed["series_key"] == placeholder
        if not is_arm.any():
            continue
        keyed.loc[is_arm, "series_key"] = (
            placeholder + "::" + row_numbers.loc[is_arm]
        )
    return keyed


def _share_histogram(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    edges: np.ndarray,
    colour: str,
    label: str,
    linestyle: str = "-",
) -> None:
    """Within-series share histogram; counts differ ~20-fold between sources."""

    weights = np.full(values.size, 100.0 / values.size)
    axis.hist(
        values,
        bins=edges,
        weights=weights,
        histtype="stepfilled",
        alpha=0.28,
        color=colour,
        edgecolor="none",
        label=label,
    )
    # Four translucent fills stacked over each other hide whichever series is
    # drawn first; an opaque outline keeps every shape readable. Unlabeled, so
    # it adds no second legend entry. ``linestyle`` marks the derived series,
    # which is the one distinction the fills alone cannot carry.
    axis.hist(
        values,
        bins=edges,
        weights=weights,
        histtype="step",
        color=colour,
        linewidth=1.7,
        linestyle=linestyle,
    )


def _nitrogen_series_label(source_name: str) -> str:
    """Reader-facing legend label for a dataset or its derived FP arm."""

    if source_name == _FARMERS_PRACTICE_SERIES_NAME:
        return _FARMERS_PRACTICE_DISPLAY_LABEL
    if source_name.endswith(_FARMERS_PRACTICE_SUFFIX):
        parent = source_name[: -len(_FARMERS_PRACTICE_SUFFIX)]
        return f"{_legend_label(parent)} — {_FARMERS_PRACTICE_DISPLAY_LABEL}"
    return _legend_label(source_name)


def _plot_nitrogen_rate_distribution(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    """Preserve the original three-series, overlaid N-rate figure."""

    observations = _observations(loaded, config)
    if observations.empty:
        return None
    farmers_practice = _farmers_practice_series(loaded, observations, config)
    observations = farmers_practice.observations.loc[
        ~farmers_practice.observations["source_name"].isin(
            farmers_practice.arm_source_names
        )
    ]
    names = _nitrogen_series_order(loaded, observations, farmers_practice)
    if not names:
        return None

    rates = observations["n_rate_kg_ha"].to_numpy(dtype=float)
    width = float(config.nitrogen_bin_width_kg_ha)
    lower = float(np.floor(np.nanmin(rates) / width) * width)
    upper = float(np.ceil(np.nanmax(rates) / width) * width)
    edges = np.arange(lower, upper + width / 2.0, width)
    if edges.size < 2:
        return None

    colours = _source_colours(loaded)
    for name in farmers_practice.arm_source_names:
        if name in colours:
            colours[name + _FARMERS_PRACTICE_SUFFIX] = colours[name]

    figure, axis = _figure(config, height_inches=7.0)
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
            linestyle="--" if name.endswith(_FARMERS_PRACTICE_SUFFIX) else "-",
        )
    axis.set(
        title=(
            "Recorded inorganic N-rate distribution per dataset "
            f"({width:g} kg N ha⁻¹ bins)"
        ),
        xlabel="Inorganic N rate (kg N ha⁻¹)",
        ylabel="Share of the dataset's recorded observations (%)",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    _footnote(
        figure,
        _original_nitrogen_rate_distribution_caption(farmers_practice),
    )
    return figure


def _plot_nitrogen_rate_distribution_with_separate_farmers_practice(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    observations = _observations(loaded, config)
    if observations.empty:
        return None
    farmers_practice = _farmers_practice_series(loaded, observations, config)
    observations = farmers_practice.observations
    names = _nitrogen_series_order(loaded, observations, farmers_practice)
    if not names:
        return None

    rates = observations["n_rate_kg_ha"].to_numpy(dtype=float)
    width = float(config.nitrogen_bin_width_kg_ha)
    lower = float(np.floor(np.nanmin(rates) / width) * width)
    # No trailing empty bin: numpy closes the last interval on the right, so a
    # maximum sitting exactly on the top edge is already counted.
    upper = float(np.ceil(np.nanmax(rates) / width) * width)
    edges = np.arange(lower, upper + width / 2.0, width)
    if edges.size < 2:
        return None

    colours = _nitrogen_series_colours(loaded, farmers_practice)
    figure, axis = _figure(config, height_inches=7.0)
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
                f"{_nitrogen_series_label(name)} — "
                f"{values.size:,} observations, {distinct} distinct rates"
            ),
            linestyle="--" if name.endswith(_FARMERS_PRACTICE_SUFFIX) else "-",
        )
    axis.set(
        title=(
            "Recorded inorganic N-rate distribution by dataset "
            "and Farmer's Practice "
            f"({width:g} kg N ha⁻¹ bins)"
        ),
        xlabel="Inorganic N rate (kg N ha⁻¹)",
        ylabel="Share within each recorded series (%)",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    _footnote(figure, _nitrogen_rate_distribution_caption(farmers_practice))
    return figure


def _nitrogen_series_order(
    loaded: LoadedSources,
    observations: pd.DataFrame,
    farmers_practice: _FarmersPracticeSeries,
) -> tuple[str, ...]:
    """Profiling order, with a farmer's-practice series after its source.

    The derived names are placed by hand rather than through
    ``_ordered_sources``, which walks ``loaded.sources``: a synthetic entry
    there would shift every other figure's palette index and offer the other
    builders a source with no rows behind it.
    """

    present = set(observations["source_name"])
    ordered = []
    for source in loaded.sources:
        name = source.source_name
        if name in present:
            ordered.append(name)
        arm_name = name + _FARMERS_PRACTICE_SUFFIX
        if arm_name in present:
            ordered.append(arm_name)
    return tuple(ordered)


def _nitrogen_series_colours(
    loaded: LoadedSources, farmers_practice: _FarmersPracticeSeries
) -> dict[str, str]:
    """Source colours plus a distinct colour for each farmer-practice arm.

    A dataset holds one colour across every figure of a run. Farmer's Practice
    is an additional treatment here, so it takes the next unused source-palette
    colour instead of borrowing its parent's colour and disappearing into it.
    """

    colours = _source_colours(loaded)
    for offset, name in enumerate(farmers_practice.arm_source_names):
        colours[name + _FARMERS_PRACTICE_SUFFIX] = _SOURCE_PALETTE[
            (len(loaded.sources) + offset) % len(_SOURCE_PALETTE)
        ]
    return colours


def _nitrogen_rate_distribution_caption(
    farmers_practice: _FarmersPracticeSeries,
) -> str:
    """The caption, carrying the composition of the derived series in full.

    A series read from an arm is only readable if the reader is told which arm
    it is, which rows were moved into it, and which recorded treatments of the
    same source the panel therefore does not show.
    """

    caption = (
        "Bars are within-series shares because the observation counts differ "
        "by more than an order of magnitude. The rates are discrete "
        "experimental ladders, not a sample from a continuous distribution."
    )
    if not farmers_practice.exists:
        return caption

    if farmers_practice.arm_rows:
        caption += (
            f" The dashed {_FARMERS_PRACTICE_DISPLAY_LABEL} series separately "
            f"shows {_FARMERS_PRACTICE_PARENT_SOURCE}'s recorded "
            "farmer-applied N rate "
            f"({farmers_practice.arm_rows:,} records carrying both that rate "
            "and its measured yield), which is farmer-chosen and therefore not "
            f"an experimental ladder. {_PH_COMBINED_TREATMENT_NOTE}"
        )
    if farmers_practice.relabeled_rows:
        caption += (
            f" {farmers_practice.relabeled_rows} core_trial_data row"
            f"{'s' if farmers_practice.relabeled_rows != 1 else ''} recorded as "
            "Farmer's Practice (Type of Experiment/Study/Trial) are included "
            "in those bars rather than in the literature-extracted bars."
        )
    caption += (
        " nitrogen_rate_profile.csv and context_composition.csv are unaffected: "
        "they report every source's recorded rows through its governed rate "
        "binding, so their per-source counts are the recorded ones."
    )
    return caption


def _original_nitrogen_rate_distribution_caption(
    farmers_practice: _FarmersPracticeSeries,
) -> str:
    """Caption retained byte-for-byte in wording for the original figure."""

    caption = (
        "Bars are within-dataset shares because the observation counts differ "
        "by more than an order of magnitude. The rates are discrete "
        "experimental ladders, not a sample from a continuous distribution."
    )
    if not farmers_practice.exists:
        return caption

    if farmers_practice.arm_rows:
        caption += (
            f" The dashed {_FARMERS_PRACTICE_SERIES_NAME} series is "
            f"{_FARMERS_PRACTICE_PARENT_SOURCE} read through its recorded "
            "farmer's-practice arm — the rate the farmer applied "
            f"({farmers_practice.arm_rows:,} records carrying both that rate "
            "and its measured yield), which is farmer-chosen and so is not a "
            f"ladder at all. {_ORIGINAL_UNDRAWN_ARM_NOTE}"
        )
    if farmers_practice.relabeled_rows:
        caption += (
            f" {farmers_practice.relabeled_rows} core_trial_data row"
            f"{'s' if farmers_practice.relabeled_rows != 1 else ''} recorded as "
            "Farmer's Practice (Type of Experiment/Study/Trial) are drawn in "
            "that series rather than as rungs of a literature-extracted ladder."
        )
    caption += (
        " nitrogen_rate_profile.csv and context_composition.csv are unaffected: "
        "they report every source's recorded rows through its governed rate "
        "binding, so their per-source counts are the recorded ones."
    )
    return caption


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
    observations: pd.DataFrame,
    *,
    names: Sequence[str],
    config: DescriptiveStatisticsConfig,
    callout_slots: Sequence[float] = _YIELD_CALLOUT_SLOTS,
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
        ylim=(0.0, tallest / (min(callout_slots) - _YIELD_CALLOUT_CLEARANCE)),
    )


def _yield_axis_population(
    observations: pd.DataFrame,
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """All populations that must share bins and limits across yield panels.

    Farmer's Practice is an alternate declared arm rather than a registered
    source, so it is not drawn in the combined three-dataset panel. It still
    participates in the shared-axis calculation: the explicit with/without
    panels must not acquire different scales merely because one reads sibling
    columns from the same source rows.
    """

    frames = [observations]
    names = list(_ordered_sources(loaded, observations["source_name"]))
    for source in loaded.sources:
        arm = _farmers_practice_arm(
            source, zero_n_tolerance_kg_ha=config.zero_n_tolerance_kg_ha
        )
        if arm is None or arm.empty:
            continue
        frames.append(arm)
        arm_name = str(arm["source_name"].iloc[0])
        if arm_name not in names:
            names.append(arm_name)
    if len(frames) == 1:
        return observations, tuple(names)
    return pd.concat(frames, ignore_index=True), tuple(names)


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
    farmers_practice: bool = False,
) -> plt.Figure | None:
    """Every dataset's yield distribution on one set of shared axes.

    ``farmers_practice`` draws the declared farmer's-practice arm as its own
    series, exactly as ``_plot_one_yield_distribution`` does for a single
    dataset. The axis population is taken from the recorded frame either way,
    so the two panels keep the bins and limits they share with their siblings.
    """

    recorded = _observations(loaded, config)
    if recorded.empty:
        return None
    if farmers_practice:
        derived = _farmers_practice_series(loaded, recorded, config)
        observations = derived.observations
        names = _nitrogen_series_order(loaded, observations, derived)
        colours = _nitrogen_series_colours(loaded, derived)
    else:
        derived = None
        observations = recorded
        names = _ordered_sources(loaded, observations["source_name"])
        colours = _source_colours(loaded)
    if not names:
        return None
    slots = (
        _extended_callout_slots(len(names))
        if farmers_practice
        else _YIELD_CALLOUT_SLOTS
    )

    axis_observations, axis_names = _yield_axis_population(
        recorded, loaded, config
    )
    shared = _yield_distribution_axes(
        axis_observations, names=axis_names, config=config, callout_slots=slots
    )
    if shared is None:
        return None
    edges = shared.edges

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
                f"{_nitrogen_series_label(name)} — "
                f"{values.size:,} observations"
            ),
            linestyle="--" if name.endswith(_FARMERS_PRACTICE_SUFFIX) else "-",
        )
        axis.axvline(median, color=colours[name], linestyle="--", linewidth=1.5)
        axis.annotate(
            f"median {median:.2f}",
            # Deliberately unguarded: a fourth configured source must raise
            # here rather than wrap onto an occupied slot and hide one
            # dataset's median under another's. Only the farmer's-practice
            # path extends the ladder, and it does so by the ladder's own step.
            xy=(median, slots[index]),
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
        title=(
            "Recorded grain-yield distribution per dataset and Farmer's "
            "Practice on the common t ha⁻¹ basis"
            if derived is not None
            else "Recorded grain-yield distribution per dataset on the common "
            "t ha⁻¹ basis"
        ),
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
    caption = (
        "Bars are within-dataset shares; dashed lines mark each dataset's "
        f"median. Unit lineage: {lineage}."
    )
    if derived is not None:
        caption = (
            f"{caption} {_farmers_practice_composition(derived)} Each FP "
            "observation is a second treatment of a trial whose NOPT-N "
            "observation is also drawn, so the two series describe the same "
            f"trials rather than independent ones. {_PH_COMBINED_TREATMENT_NOTE}"
        )
    _footnote(
        figure,
        caption,
        minimum_lines=_YIELD_DISTRIBUTION_FOOTNOTE_LINES,
    )
    return figure


def _plot_published_yield_distribution(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    """The declared ``yield_distribution`` panel, which draws the FP arm.

    Split from the renderer so the population the published figure reports is
    a property of the contract entry rather than of a default argument, and so
    the renderer stays callable without the arm by the standalone variants.
    """

    return _plot_yield_distribution(
        tables=tables, loaded=loaded, config=config, farmers_practice=True
    )


def _plot_one_yield_distribution(
    source_name: str,
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
    farmers_practice: bool = False,
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

    axis_observations, axis_names = _yield_axis_population(
        observations, loaded, config
    )
    shared = _yield_distribution_axes(
        axis_observations, names=axis_names, config=config
    )
    if shared is None:
        return None

    if farmers_practice:
        resolved = _farmers_practice_arm_for_source(
            loaded,
            source_name,
            zero_n_tolerance_kg_ha=config.zero_n_tolerance_kg_ha,
        )
        if resolved is None:
            return None
        _, subset = resolved
    else:
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
        label=(
            f"{_legend_label(source_name)}"
            + (" — Farmer's Practice arm" if farmers_practice else "")
            + f" — {values.size:,} observations"
        ),
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
    if source_name == _FARMERS_PRACTICE_PARENT_SOURCE:
        title = (
            "PH combined yield — Farmer's Practice arm (t ha⁻¹ basis)"
            if farmers_practice
            else "PH combined yield — NOPT arm without Farmer's Practice "
            "(t ha⁻¹ basis)"
        )
    else:
        title = (
            f"Recorded grain-yield distribution — {_legend_label(source_name)} "
            "(t ha⁻¹ basis)"
        )
    axis.set(
        title=title,
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
    if source_name != _FARMERS_PRACTICE_PARENT_SOURCE:
        treatment_clause = ""
    elif farmers_practice:
        treatment_clause = (
            " This version reads the paired Farmer's Practice rate and yield "
            "columns; rows without both finite values are excluded. The NOPT "
            "full-fertilizer arm is not pooled into this distribution."
        )
    else:
        treatment_clause = (
            " This version reads the governed NOPT full-fertilizer rate and "
            "yield columns and excludes the paired Farmer's Practice arm."
        )
    _footnote(
        figure,
        "Bars are shares of this dataset's own harmonized observations; the "
        "dashed line marks its median and the dotted line marks its mean."
        f"{modal_clause} "
        "Bins, axis limits, and panel size are shared with the other "
        "per-dataset panels, the Farmer's Practice companion, and the combined "
        "figure, so they are "
        f"directly comparable. Unit lineage: {lineage}."
        + treatment_clause,
        minimum_lines=_YIELD_DISTRIBUTION_FOOTNOTE_LINES,
    )
    return figure


def _yield_distribution_builder(
    source_name: str,
    *,
    farmers_practice: bool = False,
) -> Callable[..., "plt.Figure | None"]:
    """Bind one source name into the shared single-dataset builder."""

    def builder(
        *,
        tables: Mapping[str, pd.DataFrame],
        loaded: LoadedSources,
        config: DescriptiveStatisticsConfig,
    ) -> plt.Figure | None:
        return _plot_one_yield_distribution(
            source_name,
            tables=tables,
            loaded=loaded,
            config=config,
            farmers_practice=farmers_practice,
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
    farmers_practice: bool = False,
) -> plt.Figure | None:
    """The N-yield cloud with each series' own observations joined in place.

    Same point cloud as ``yield_versus_nitrogen`` — identical rows, colours, and
    opacity rule — with within-series connecting lines added beneath it. The two
    panels are meant to be read together: one shows coverage, this one shows
    that the coverage is made of ladders.

    ``farmers_practice`` adds the declared farmer's-practice arm as its own
    series. Its rates span far wider than any designed ladder, so the panel it
    produces has a different x-range from the recorded-source panel and the two
    are not read against one another.
    """

    recorded = _observations(loaded, config)
    if recorded.empty:
        return None
    if farmers_practice:
        derived = _farmers_practice_series(loaded, recorded, config)
        observations = _single_rate_arm_series_keys(derived.observations, derived)
        names = _nitrogen_series_order(loaded, observations, derived)
        colours = _nitrogen_series_colours(loaded, derived)
    else:
        derived = None
        observations = recorded
        names = _ordered_sources(loaded, observations["source_name"])
        colours = _source_colours(loaded)
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
                label=f"{_nitrogen_series_label(name)} — {detail}",
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
            + (", Farmer's Practice included" if derived is not None else "")
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
            + ", ".join(_nitrogen_series_label(name) for name in lineless)
            + ": each record holds one N rate, and its zero-N arm is a paired "
            "column profiled in zero_nitrogen_checks, not a second row."
        )
    spanning = [
        f"{trajectories.multi_year[name]:,} in {_nitrogen_series_label(name)}"
        for name in names
        if trajectories.multi_year.get(name, 0)
    ]
    if spanning:
        note += (
            " Some joined series record more than one year of planting ("
            + ", ".join(spanning)
            + "), so those lines join observations made in different seasons."
        )
    if derived is not None:
        note += (
            f" {_farmers_practice_composition(derived)} A farmer-chosen rate "
            "is not a rung of a designed ladder, so each appended row is its "
            "own single-rate series and the arm contributes points rather "
            f"than lines. {_ph_combined_treatment_note('points')}"
        )
    _footnote(figure, note, minimum_lines=_PAIRED_SCATTER_FOOTNOTE_LINES)
    return figure


def _plot_temporal_coverage(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
    series_order: Sequence[str] | None = None,
    series_colours: Mapping[str, str] | None = None,
) -> plt.Figure | None:
    """Stacked observation counts per recorded year.

    ``series_order`` and ``series_colours`` are for a caller that stacks a
    derived series beside the recorded ones, such as a farmer's-practice arm.
    ``_ordered_sources`` and ``_source_colours`` both walk ``loaded.sources``,
    which deliberately holds no entry for a derived series, so such a caller
    supplies both itself. Omitting them is the governed path and is unchanged.
    """

    coverage = _table(tables, "temporal_coverage")
    if coverage is None:
        return None
    frame = coverage.copy()
    frame["_source"] = _text_column(frame, "source_name")
    frame["_year"] = _numeric_column(frame, "year")
    frame["_count"] = _numeric_column(frame, "observation_count")
    frame = frame.loc[frame["_year"].notna() & frame["_count"].notna()]
    if series_order is None:
        names = _ordered_sources(loaded, frame["_source"])
    else:
        # Held to the same rule as _ordered_sources: a name with no rows behind
        # it would otherwise reach the legend as a zero-height entry.
        recorded = set(frame["_source"])
        names = tuple(name for name in series_order if name in recorded)
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
    colours = (
        dict(series_colours)
        if series_colours is not None
        else _source_colours(loaded)
    )
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
            # Falls through to _legend_label for every recorded source, so the
            # governed panel is unchanged; only a derived arm renders as one.
            label=(
                f"{_nitrogen_series_label(name)} — "
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
    # Separated only when disclosure permits. Restricted sources leave both as
    # ``None`` so their rare-level count cannot be recovered from the missing
    # count and the visible total.
    missing_count: float | None = None
    rare_count: float | None = None

    @property
    def reported_total(self) -> float:
        return float(self.levels["_count"].sum())

    @property
    def residual_share(self) -> float:
        """Share outside the individually reported recorded levels."""

        return max(0.0, 1.0 - self.reported_total / self.denominator)

    @property
    def separates_residual(self) -> bool:
        return self.missing_count is not None and self.rare_count is not None

    @property
    def missing_share(self) -> float:
        if self.missing_count is None:
            return 0.0
        return max(0.0, float(self.missing_count) / self.denominator)

    @property
    def rare_share(self) -> float:
        if self.rare_count is None:
            return 0.0
        return max(0.0, float(self.rare_count) / self.denominator)

    @property
    def pooled_residual_share(self) -> float:
        """Unseparated residual retained for disclosure-controlled sources."""

        return 0.0 if self.separates_residual else self.residual_share


def _context_residual_counts(
    source: ProfiledSource,
    observations: pd.DataFrame,
    *,
    context_key: str,
    denominator: float,
    reported_total: float,
) -> tuple[float | None, float | None]:
    """Return missing and rare counts when they may be disclosed separately.

    A blank source cell and a nonblank level below the reporting threshold are
    different findings. They remain pooled for restricted sources because
    publishing the blank count would reveal the withheld rare-level total by
    subtraction. Literature-derived internal data carry no such constraint.
    """

    if source.is_restricted or observations.empty:
        return None, None
    if not np.isclose(float(len(observations)), denominator):
        # A treatment-arm override can use a different population from the
        # ordinary source observations. Without its exact row set, retain the
        # disclosure-safe pooled remainder rather than misstate missingness.
        return None, None

    residual = max(0.0, denominator - reported_total)
    if context_key in {APPLIED_N_BAND_CONTEXT, STRAW_MANAGEMENT_CONTEXT}:
        missing = 0.0
    elif context_key == CROP_ESTABLISHMENT_CONTEXT:
        binding = source.binding.crop_establishment
        if binding is None:
            return None, None
        positions = _aligned_row_positions(source, observations)
        values = (
            source.text_series(binding)
            .astype(str)
            .str.strip()
            .to_numpy()[positions]
        )
        missing = float(np.count_nonzero(values == ""))
    elif context_key == YEAR_BAND_CONTEXT:
        if source.binding.year is None:
            return None, None
        positions = _aligned_row_positions(source, observations)
        values = (
            source.text_series(source.binding.year)
            .astype(str)
            .str.strip()
            .to_numpy()[positions]
        )
        missing = float(np.count_nonzero(values == ""))
    else:
        binding = next(
            (
                candidate
                for candidate in source.binding.context
                if candidate.label.strip().lower() == context_key
            ),
            None,
        )
        if binding is None:
            return None, None
        lookup = pd.Series(
            np.arange(source.data_row_count),
            index=pd.Index(source.source_row_numbers),
        )
        positions = (
            observations["source_row_number"].map(lookup).to_numpy(dtype=np.int64)
        )
        values = (
            source.text[binding.raw_column_id]
            .astype(str)
            .str.strip()
            .to_numpy()[positions]
        )
        missing = float(np.count_nonzero(values == ""))

    # Every residual observation is either blank or in a recorded level that
    # fell below the reporting threshold. Clamp against the visible residual to
    # absorb harmless floating arithmetic without allowing a negative rare mass.
    missing = min(max(0.0, missing), residual)
    rare = max(0.0, residual - missing)
    return missing, rare


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
    source_totals: Mapping[str, float] | None = None,
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
    if source_totals is None:
        resolved_source_totals: Mapping[str, float] = (
            observations["source_name"].value_counts().to_dict()
        )
    else:
        resolved_source_totals = source_totals

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
                declared_total,
                float(resolved_source_totals.get(source.source_name, 0.0)),
            )
            if denominator <= 0:
                continue
            source_observations = observations.loc[
                observations["source_name"] == source.source_name
            ]
            reported_total = float(group["_count"].sum())
            missing_count, rare_count = _context_residual_counts(
                source,
                source_observations,
                context_key=context_key,
                denominator=denominator,
                reported_total=reported_total,
            )
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
                    missing_count=missing_count,
                    rare_count=rare_count,
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
        fragments.append(f"{_legend_label(source.source_name)} ({reason})")
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
    any_missing = False
    any_rare = False
    any_pooled_residual = False
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
            if bar.separates_residual:
                missing = min(remainder, bar.missing_share)
                if missing > 0.0005:
                    any_missing = True
                    axis.barh(
                        position,
                        missing * 100.0,
                        left=left * 100.0,
                        height=0.62,
                        color=_MISSING_GREY,
                        edgecolor="white",
                        linewidth=0.6,
                        hatch="//",
                    )
                    if missing >= _MINIMUM_MISSING_LABELED_SHARE:
                        axis.text(
                            (left + missing / 2.0) * 100.0,
                            position,
                            f"Missing\n{missing * 100:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=7,
                        )
                    left += missing

                rare = min(max(0.0, 1.0 - left), bar.rare_share)
                if rare > 0.0005:
                    any_rare = True
                    axis.barh(
                        position,
                        rare * 100.0,
                        left=left * 100.0,
                        height=0.62,
                        color=_RARE_GREY,
                        edgecolor="white",
                        linewidth=0.6,
                        hatch="..",
                    )
                    left += rare

            pooled = max(0.0, 1.0 - left)
            if pooled > 0.0005:
                any_pooled_residual = True
                axis.barh(
                    position,
                    pooled * 100.0,
                    left=left * 100.0,
                    height=0.62,
                    color=_WITHHELD_GREY,
                    edgecolor="white",
                    linewidth=0.6,
                    hatch="xx",
                )

    axis.set_yticks(positions)
    axis.set_yticklabels(
        list(tick_labels)
        if tick_labels is not None
        else [
            f"{bar.context_title}\n{_tick_label(bar.source_name)} "
            f"({bar.classification})"
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
    residual_handles: list[Patch] = []
    if any_missing:
        residual_handles.append(
            Patch(
                facecolor=_MISSING_GREY,
                edgecolor="white",
                hatch="//",
                label="Missing: blank or unrecorded source value",
            )
        )
    if any_rare:
        residual_handles.append(
            Patch(
                facecolor=_RARE_GREY,
                edgecolor="white",
                hatch="..",
                label=(
                    "Rare: recorded levels with fewer than "
                    f"{config.minimum_level_count} observations each"
                ),
            )
        )
    if any_pooled_residual:
        residual_handles.append(
            Patch(
                facecolor=_WITHHELD_GREY,
                edgecolor="white",
                hatch="xx",
                label=(
                    "Unreported remainder: missing or rare (fewer than "
                    f"{config.minimum_level_count} observations)"
                ),
            )
        )
    if residual_handles:
        axis.legend(
            handles=residual_handles,
            fontsize=8,
            loc="lower right",
        )
    axis.grid(axis="x", alpha=0.2)
    _footnote(figure, footnote)
    return figure


_COMPOSITION_FOOTNOTE = (
    "Individually reported level segments narrower than "
    f"{_MINIMUM_LABELED_SHARE * 100:.0f}% are drawn but not labeled. Except "
    "for fields explicitly identified as derived or standardized, level "
    "vocabularies are dataset-specific and are reported as recorded. Missing "
    "segments of at least "
    f"{_MINIMUM_MISSING_LABELED_SHARE * 100:.0f}% are labeled directly."
)


def _series_parent_source(
    loaded: LoadedSources, series_name: str
) -> ProfiledSource:
    """The registered source a drawn series is read from, derived or not."""

    name = series_name
    if name.endswith(_FARMERS_PRACTICE_SUFFIX):
        name = name[: -len(_FARMERS_PRACTICE_SUFFIX)]
    source = next(
        (candidate for candidate in loaded.sources if candidate.source_name == name),
        None,
    )
    if source is None:
        raise ProfileContractError(
            f"No registered source behind series {series_name!r}"
        )
    return source


def _temporal_coverage_with_farmers_practice(
    loaded: LoadedSources,
    derived: _FarmersPracticeSeries,
    names: Sequence[str],
) -> pd.DataFrame:
    """Coverage counts for every drawn series, the derived one included.

    ``_temporal_rows`` is reused rather than reimplemented, so the year
    rounding, the undated-row handling, and the share denominator are the
    published table's own. Only the two identity columns are rewritten,
    because a derived series has no ``ProfiledSource`` to key them from.
    """

    observations = derived.observations
    rows: list[dict[str, Any]] = []
    for series_name in names:
        source = _series_parent_source(loaded, series_name)
        subset = observations.loc[
            observations["source_name"] == series_name
        ].reset_index(drop=True)
        for row in _temporal_rows(source, subset):
            # A derived arm is read from its parent's columns and inherits the
            # parent's classification. The rows adopted from core_trial_data
            # are classified less restrictively than that, so the inherited
            # label never under-declares the series.
            row["source_name"] = series_name
            row["data_classification"] = source.data_classification
            rows.append(row)
    return conform_table("temporal_coverage", pd.DataFrame.from_records(rows))


def _plot_published_temporal_coverage(
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
) -> plt.Figure | None:
    """The declared ``temporal_coverage`` panel, which draws the FP arm.

    Counts are recomputed from the observation cloud rather than read from the
    ``temporal_coverage`` table: the drawn population carries the declared
    farmer's-practice arm, and the published table records the three
    registered sources. The table is unchanged and the caption states the
    difference, so the two cannot be read as disagreeing about one population.
    """

    del tables  # the drawn population is wider than the published table
    recorded = _observations(loaded, config)
    if recorded.empty:
        return None
    derived = _farmers_practice_series(loaded, recorded, config)
    names = _nitrogen_series_order(loaded, derived.observations, derived)
    if not names:
        return None
    coverage = _temporal_coverage_with_farmers_practice(loaded, derived, names)
    figure = _plot_temporal_coverage(
        tables={"temporal_coverage": coverage},
        loaded=loaded,
        config=config,
        series_order=names,
        series_colours=_nitrogen_series_colours(loaded, derived),
    )
    if figure is None or not derived.exists:
        # With no arm to draw this is the recorded-source panel already, and
        # its own title and caption describe it correctly.
        return figure

    dated_arm_rows = int(
        coverage.loc[
            coverage["source_name"] == _FARMERS_PRACTICE_SERIES_NAME,
            "observation_count",
        ].sum()
        - derived.relabeled_rows
    )
    figure.axes[0].set_title(
        "Observation coverage by recorded year of planting, "
        "Farmer's Practice included"
    )
    # Replace the renderer's generic caption with one that states the derived
    # series' composition; re-running the helper reserves the taller band.
    figure.texts.clear()
    _footnote(
        figure,
        f"{_farmers_practice_composition(derived)} {dated_arm_rows:,} of the "
        "appended rows carry a recorded year. Each FP observation is a second "
        "treatment of a trial whose NOPT-N observation is also stacked here, "
        "so a bar counts those trials once per arm rather than once per "
        f"trial. {_PH_COMBINED_TREATMENT_NOTE} Bars stack the series within "
        "each recorded year; rows without a recorded year contribute no bar.",
    )
    return figure


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
        tables,
        keys
        + [key for key, _, _ in _DERIVED_CONTEXTS]
        + [key for key, _, _ in _STANDARDIZED_CONTEXTS],
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

    # Standardized contexts follow the recorded fields, which keeps the new
    # straw row beside the other management fields at the bottom of the panel.
    for key, title, _ in _STANDARDIZED_CONTEXTS:
        if key == STRAW_MANAGEMENT_CONTEXT and not source.binding.straw_management:
            continue
        if (
            key == CROP_ESTABLISHMENT_CONTEXT
            and source.binding.crop_establishment is None
        ):
            continue
        recorded = drawn_levels(key)
        levels = len(recorded)
        if levels == 0:
            omitted.append(
                f"{title} (no standardized level recorded at least "
                f"{config.minimum_level_count} times)"
            )
        elif levels < _MINIMUM_STACKED_LEVELS:
            omitted.append(constant_note(title, recorded))
        elif levels > _MAXIMUM_STACKED_LEVELS:
            omitted.append(f"{title} ({levels} levels, too many for this form)")
        else:
            drawable.append((key, title))
    return _prioritize_dataset_contexts(source.source_name, drawable), omitted


def _straw_management_breakdown(bars: Sequence[_CompositionBar]) -> str:
    """Reader-visible counts for narrow straw segments that cannot hold labels."""

    bar = next(
        (
            candidate
            for candidate in bars
            if candidate.context_key == STRAW_MANAGEMENT_CONTEXT
        ),
        None,
    )
    if bar is None or bar.denominator <= 0:
        return ""
    counts = {
        str(row["_level"]): int(round(float(row["_count"])))
        for _, row in bar.levels.iterrows()
    }
    present = [
        f"{level} {counts[level]:,} ({100.0 * counts[level] / bar.denominator:.1f}%)"
        for level in STRAW_MANAGEMENT_LEVELS
        if counts.get(level, 0) > 0
    ]
    zero_actions = [
        level
        for level in ("Burned", "Incorporated", "Removed", "Retained/returned")
        if counts.get(level, 0) == 0
    ]
    summary = " Standardized straw-management counts: " + ", ".join(present) + "."
    if zero_actions:
        summary += (
            " No observations were identified as "
            + " or ".join(zero_actions)
            + "."
        )
    return summary


def _crop_establishment_breakdown(bars: Sequence[_CompositionBar]) -> str:
    """Reader-visible counts for establishment segments too narrow to label."""

    bar = next(
        (
            candidate
            for candidate in bars
            if candidate.context_key == CROP_ESTABLISHMENT_CONTEXT
        ),
        None,
    )
    if bar is None or bar.denominator <= 0:
        return ""
    counts = {
        str(row["_level"]): int(round(float(row["_count"])))
        for _, row in bar.levels.iterrows()
    }
    parts = [
        f"{level} {counts[level]:,} "
        f"({100.0 * counts[level] / bar.denominator:.1f}%)"
        for level in CROP_ESTABLISHMENT_LEVELS
        if counts.get(level, 0) > 0
    ]
    if bar.missing_count is not None and bar.missing_count > 0:
        missing = int(round(bar.missing_count))
        parts.append(
            f"Missing {missing:,} ({100.0 * missing / bar.denominator:.1f}%)"
        )
    return " Standardized crop-establishment counts: " + ", ".join(parts) + "."


def _farmers_practice_context_tables(
    source: ProfiledSource,
    arm: pd.DataFrame,
    *,
    tables: Mapping[str, pd.DataFrame],
    config: DescriptiveStatisticsConfig,
) -> dict[str, pd.DataFrame]:
    """Replace the context table with one computed on the declared FP arm.

    The physical context columns are shared by the paired NOPT and Farmer's
    Practice measurements. Rebuilding their summaries on the arm's admitted
    row numbers preserves missing-context differences between the 718 governed
    NOPT rows and the 647 rows carrying both a finite FP rate and FP yield.
    """

    composition = build_context_composition(source, arm, config)
    rate_binding = source.binding.farmers_practice_n_rate
    if rate_binding is not None and not composition.empty:
        applied = composition["context_label"] == "applied_n_band"
        composition.loc[applied, "column_header"] = rate_binding.header
    return {**tables, "context_composition": composition}


def _plot_one_context_composition(
    source_name: str,
    *,
    tables: Mapping[str, pd.DataFrame],
    loaded: LoadedSources,
    config: DescriptiveStatisticsConfig,
    farmers_practice: bool = False,
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
    source_totals = None
    selected_tables = tables
    if farmers_practice:
        resolved = _farmers_practice_arm_for_source(
            loaded,
            source_name,
            zero_n_tolerance_kg_ha=config.zero_n_tolerance_kg_ha,
        )
        if resolved is None:
            return None
        _, arm = resolved
        selected_tables = _farmers_practice_context_tables(
            source, arm, tables=tables, config=config
        )
        source_totals = {source_name: float(len(arm))}
    contexts, omitted = _dataset_context_plan(
        source, tables=selected_tables, config=config
    )
    if not contexts:
        return None
    bars = _composition_bars(
        tables=selected_tables,
        loaded=loaded,
        config=config,
        contexts=contexts,
        source_names=(source_name,),
        source_totals=source_totals,
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
    standardization = "".join(
        " " + caption
        for key, _, caption in _STANDARDIZED_CONTEXTS
        if key in drawn
    )
    establishment_breakdown = _crop_establishment_breakdown(bars)
    straw_breakdown = _straw_management_breakdown(bars)
    if source_name != _FARMERS_PRACTICE_PARENT_SOURCE:
        arm_title = ""
        treatment_note = ""
    elif farmers_practice:
        arm_title = " — Farmer's Practice arm"
        treatment_note = (
            " This version includes only rows carrying both a finite Farmer's "
            "Practice N rate and its paired measured yield; the applied-N bar "
            "therefore reads fp_actual_n_kg_per_ha. The NOPT full-fertilizer "
            "arm is not pooled into these shares."
        )
    else:
        arm_title = " — NOPT arm (Farmer's Practice excluded)"
        treatment_note = (
            " This version reads the governed NOPT full-fertilizer population; "
            "the paired Farmer's Practice arm is excluded."
        )
    if source_name == _FARMERS_PRACTICE_PARENT_SOURCE:
        title = (
            "PH combined context (restricted) — "
            + (
                f"Farmer's Practice arm, n={observations:,.0f}"
                if farmers_practice
                else "NOPT arm without Farmer's Practice, "
                f"n={observations:,.0f}"
            )
        )
    else:
        title = (
            f"{_legend_label(source.source_name)} — context composition "
            f"({source.data_classification}; n={observations:,.0f}){arm_title}"
        )
    return _draw_stacked_composition(
        bars=bars,
        config=config,
        title=title,
        footnote=(
            f"{_COMPOSITION_FOOTNOTE} Every bar is a share of this dataset's "
            "own harmonized observations, so the bars are comparable with each "
            "other and with this dataset's bars in the cross-dataset "
            f"composition figures.{derivation}{standardization}"
            f"{establishment_breakdown}{straw_breakdown}{treatment_note}{note}"
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
        level_colours=_composition_level_colours(
            selected_tables, loaded, config
        ),
    )


def _context_composition_builder(
    source_name: str,
    *,
    farmers_practice: bool = False,
) -> Callable[..., "plt.Figure | None"]:
    """Bind one source name into the shared single-dataset builder."""

    def builder(
        *,
        tables: Mapping[str, pd.DataFrame],
        loaded: LoadedSources,
        config: DescriptiveStatisticsConfig,
    ) -> plt.Figure | None:
        return _plot_one_context_composition(
            source_name,
            tables=tables,
            loaded=loaded,
            config=config,
            farmers_practice=farmers_practice,
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
    if bar.separates_residual and bar.missing_share > 0.0005:
        entries.append(
            (
                "Missing",
                bar.missing_share,
                _MISSING_GREY,
                "//",
                f"{bar.missing_count:,.0f} obs",
            )
        )
    if bar.separates_residual and bar.rare_share > 0.0005:
        entries.append(
            (
                "Rare recorded levels",
                bar.rare_share,
                _RARE_GREY,
                "..",
                f"{bar.rare_count:,.0f} obs",
            )
        )
    pooled = bar.pooled_residual_share
    if pooled > 0.0005:
        entries.append(
            (
                "Missing or rare (unreported)",
                pooled,
                _WITHHELD_GREY,
                "xx",
                f"{pooled * bar.denominator:,.0f} obs",
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
        f"{_legend_label(bar.source_name)} ({bar.classification})\n"
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
# Registry and dispatch
# --------------------------------------------------------------------------


FIGURE_BUILDERS: Mapping[str, Callable[..., "plt.Figure | None"]] = {
    "nitrogen_rate_distribution": _plot_nitrogen_rate_distribution,
    "nitrogen_rate_distribution_with_separate_farmers_practice": (
        _plot_nitrogen_rate_distribution_with_separate_farmers_practice
    ),
    "yield_distribution": _plot_published_yield_distribution,
    **{
        spec.name: _yield_distribution_builder(
            spec.source_name,
            farmers_practice=_figure_uses_farmers_practice(spec.name),
        )
        for spec in FIGURE_SPECS
        if spec.name.startswith("yield_distribution_") and spec.source_name
    },
    "yield_versus_nitrogen": _plot_yield_versus_nitrogen,
    "yield_versus_nitrogen_trajectories": _plot_yield_versus_nitrogen_trajectories,
    "temporal_coverage": _plot_published_temporal_coverage,
    "context_composition": _plot_context_composition,
    "context_composition_site_and_management": (
        _plot_context_composition_site_and_management
    ),
    "context_composition_variety": _plot_context_composition_variety,
    **{
        spec.name: _context_composition_builder(
            spec.source_name,
            farmers_practice=_figure_uses_farmers_practice(spec.name),
        )
        for spec in FIGURE_SPECS
        if spec.name.startswith("context_composition_") and spec.source_name
    },
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
