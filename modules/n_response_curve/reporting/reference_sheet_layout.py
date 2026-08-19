"""Geometry and typography of the 34-inch combined reference sheet.

Three recipes draw the same kind of sheet — a row of per-decade response insets
sitting directly above the trend those decades came from, on one shared
year-to-inches mapping. The numbers below are what make that correspondence
true, so they belong to the sheet rather than to whichever recipe was written
first: LTCCE's season sheet, the core-trial planting-year sheet, and IR8's
variety sheet all read the same because they measure from the same block.

They were reached by looking at rendered output at full size, and several are
derived from one another rather than chosen (see the comments). Changing one in
isolation does not scale the sheet — it moves one strip and leaves the rest
where it was.

Everything here is in inches from the bottom-left, or in points for type,
because `constrained_layout` cannot be used on this figure: a layout engine
sizes each axes from its own labels, which would let the insets drift off the
stretch of trend they describe.

A recipe whose sheet is a different width scales from these rather than
restating them — see ``_INCHES_PER_YEAR`` in the planting-year recipes, which
divides this sheet's usable width by its six-decade reference span so a
shorter record keeps the same density instead of being stretched to fill.

This module is importable by the standalone reporting recipes and is not on the
release path.
"""

from __future__ import annotations

__all__ = [
    "AXIS_LABEL_FONT_SIZE",
    "CAPTION_COLUMN_CHARS",
    "CAPTION_COLUMN_OFFSET",
    "CAPTION_FONT_SIZE",
    "CAPTION_LINE_SPACING",
    "CAPTION_PARAGRAPH_GAP_LINES",
    "DESIGN_GAP_IN",
    "DESIGN_ROW_IN",
    "DESIGN_TITLE_FONT_SIZE",
    "DESIGN_TITLE_IN",
    "ERA_MARKERS",
    "HEADLINE_FONT_SIZE",
    "HOST_ROW_IN",
    "HOST_TREND_FRACTION",
    "INSET_BOTTOM_FRACTION",
    "INSET_LEGEND_IN",
    "INSET_READOUT_FONT_SIZE",
    "INSET_TICK_FONT_SIZE",
    "INSET_TITLE_FONT_SIZE",
    "INSET_TOP_FRACTION",
    "LEGEND_FONT_SIZE",
    "NARROW_PANEL_IN",
    "PANEL_GUTTER_IN",
    "SHEET_BOTTOM_IN",
    "SHEET_LEFT_IN",
    "SHEET_RIGHT_IN",
    "SHEET_WIDTH_IN",
    "TICK_FONT_SIZE",
    "TREND_GAP_IN",
    "TREND_LEGEND_IN",
    "TREND_ROW_IN",
    "TREND_XAXIS_IN",
    "ZERO_N_TREATMENT_CLASS",
    "text_inches",
]




# Hand-placed layout for the combined sheet, in inches, measured from the
# bottom edge. `constrained_layout` cannot be used here: the whole point of the
# figure is that each decade's response curve sits on the stretch of the trend
# its trajectories came from, and that requires the insets and the trend axes
# to be positioned from one shared year-to-inches mapping. A layout engine
# sizes every axes from its own labels, so the correspondence would drift.
SHEET_WIDTH_IN = 34.0
# Wide enough for the trend tick labels plus the rotated y label at the type
# sizes set below. Every strip on this sheet that carries text is derived from
# a font size for the same reason: `add_axes` runs text off the canvas without
# raising, so a hard-coded margin fails silently the moment the type grows.
SHEET_LEFT_IN = 1.30
SHEET_RIGHT_IN = 0.35
SHEET_BOTTOM_IN = 0.25
HOST_ROW_IN = 11.6
# How much of the host axes the trend itself is allowed to occupy, measured
# from the bottom. The rest is the band the insets sit in, plus the strip
# between them that carries each inset's applied-N tick labels.
HOST_TREND_FRACTION = 0.42
# Derived, never chosen. The response panel is given exactly the height of the
# host's trend band, and both panels are drawn to one shared y range, so the
# two share an inches-per-t/ha as well as an x axis: a change of a given size
# has the same slope in both. That is the only way the two ends of the same
# curve can be read against each other.
TREND_ROW_IN = HOST_TREND_FRACTION * HOST_ROW_IN
TREND_GAP_IN = 0.12
# Kept out of the shared rectangle so two neighbouring insets never touch;
# taken off the inset, never off the band, so the alignment stays exact.
PANEL_GUTTER_IN = 0.10
# An inset narrower than this gets the abbreviated read-out. Every inset on the
# decade axis clears it comfortably; the branch is kept for a season whose
# record spans enough decades to divide the sheet into thin bands.
NARROW_PANEL_IN = 2.2
# The caption carries the governance disclosures and runs to eight or twelve
# entries. Centred across a 34-inch sheet it reads as a wall of text, and at a
# readable measure a single column leaves half the sheet empty, so it is set as
# two left-aligned columns balanced by wrapped line count.
HEADLINE_FONT_SIZE = 23
CAPTION_FONT_SIZE = 13
CAPTION_COLUMN_CHARS = 108
CAPTION_COLUMN_OFFSET = 0.52
CAPTION_LINE_SPACING = 1.45
# Each bullet is drawn as its own text object separated by this fraction of a
# line, rather than the whole column joined with newlines into one. Joined, the
# gap between two disclosures is exactly the gap between two lines of the same
# disclosure, and the column has no paragraph structure at all — which is what
# made it read as a wall regardless of the type size.
CAPTION_PARAGRAPH_GAP_LINES = 0.55
# Type on this sheet is set much larger than on the 12-inch composition
# figures. Every panel here is several inches across; sizes that read on the
# small canvas render as a grey blur once the sheet is 34 inches wide and the
# reader is looking at it scaled to fit a screen.
AXIS_LABEL_FONT_SIZE = 17
TICK_FONT_SIZE = 14
INSET_TITLE_FONT_SIZE = 20
INSET_TICK_FONT_SIZE = 12
INSET_READOUT_FONT_SIZE = 12
LEGEND_FONT_SIZE = 14
# The optional design row: one panel per recorded plot design, positioned on the
# same year-to-inches mapping as the decade insets, so a design's panel is as
# wide as the stretch of the experiment that ran under it.
DESIGN_ROW_IN = 4.6
DESIGN_GAP_IN = 0.62
DESIGN_TITLE_FONT_SIZE = 21


def text_inches(
    font_size: float, lines: float = 1.0, *, spacing: float = 1.35
) -> float:
    """Height in inches of `lines` lines set at `font_size` points.

    Every strip on the sheet that exists only to carry text is sized through
    this rather than by a chosen number, so raising a font size cannot push its
    own labels off the canvas. `savefig` does not complain when it happens, and
    neither does the linter.
    """

    return font_size * spacing * lines / 72.0


# Derived, never chosen — see `text_inches`. The trend x-axis strip carries a
# row of tick labels *and* the axis label; the trend legend carries its title
# row and its entry row; the design row's `set_title` draws above the axes and
# so needs its own allowance, or it runs into the caption above it.
INSET_LEGEND_IN = text_inches(LEGEND_FONT_SIZE) + 0.22
TREND_XAXIS_IN = (
    text_inches(TICK_FONT_SIZE) + text_inches(AXIS_LABEL_FONT_SIZE) + 0.26
)
TREND_LEGEND_IN = text_inches(LEGEND_FONT_SIZE, 2.0) + 0.24
DESIGN_TITLE_IN = text_inches(DESIGN_TITLE_FONT_SIZE) + 0.16
# Era is carried by marker shape when colour has been spent on the quantity.
# Cycled in this order, which is the order the eras are first observed.
ERA_MARKERS = ("o", "s", "^", "D", "v", "P", "X")

# Every inset draws its treatment classes through `draw_overlay_on_axes`, which
# calls `scatter` once per class in `treatment_classes` order with no explicit
# colour, so each class takes the matching entry of the axes property cycle.
# `treatment_classes` is built with `sorted()`, so the order is stable across
# runs and seasons.
ZERO_N_TREATMENT_CLASS = "zero N"

# The strip between the trend's ceiling and the insets' lower edge has to hold
# the insets' applied-N tick labels *and* the line naming the shared inset
# frame, both of which grew with the type. Raised from 0.50, with `HOST_ROW_IN`
# grown to match so the insets keep their height.
INSET_BOTTOM_FRACTION = 0.53
INSET_TOP_FRACTION = 0.94
