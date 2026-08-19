"""Title, caption, and disclosure text mechanics for the diagnostic figures.

Distinct from :mod:`n_response_curve.reporting.reference_sheet_layout`, which
sizes the 34-inch sheet. Everything here is for the ~12-inch composition and
overlay canvases, where type is set several points smaller and the binding
constraint is different: matplotlib **clips** a title at the axes edge instead
of shrinking it, so a disclosure line that runs long is silently truncated. A
truncated caveat is worse than no caveat, which is why the lines are folded to a
measured width before they are handed over rather than after.

:data:`EXPLORATORY_DIAGNOSTIC_DISCLAIMER` is the ANA-11 line every cluster
figure carries. It is shared rather than restated because it is a governance
disclosure: two copies that drift apart would put two different claims about the
same bundle on two figures a reader sees side by side. A recipe whose disclosure
is genuinely different — the variety recipe names its own confounding — states
its own and does not import this one.

This module is importable by the standalone reporting recipes and is not on the
release path.
"""

from __future__ import annotations

import textwrap
from typing import Any, Sequence

__all__ = [
    "COMPOSITION_TITLE_WIDTH",
    "EXPLORATORY_DIAGNOSTIC_DISCLAIMER",
    "TITLE_FONT_SIZE",
    "reserve_suptitle",
    "wrap_title_lines",
]


EXPLORATORY_DIAGNOSTIC_DISCLAIMER = (
    "exploratory diagnostic; not a governed analysis (ANA-11) — no curve is fitted"
)

# The disclosure discipline costs title lines: a cluster figure carries up to
# nine, including two long ladder/season caveats. On the 10x7 inch canvas
# `create_source_dataset_overlay_figure` builds, that block runs edge to edge and
# squeezes the axes, so these figures are enlarged and the title font raised to
# match the bigger canvas.
TITLE_FONT_SIZE = 10

# The disclosure lines run long, and matplotlib clips a title at the axes edge
# instead of shrinking it — a truncated caveat is worse than no caveat, so the
# lines are folded before they are handed over. Sized for the 12-inch canvas the
# composition figures use at font size 9.
COMPOSITION_TITLE_WIDTH = 130


def wrap_title_lines(
    lines: Sequence[str], width: int = COMPOSITION_TITLE_WIDTH
) -> str:
    """Fold each line to *width* and join them, one disclosure per line.

    Long words are never broken: an applied-N range or a series identifier split
    across two lines reads as two values.
    """

    return "\n".join(
        textwrap.fill(line, width=width, break_long_words=False) for line in lines
    )


def reserve_suptitle(
    figure: Any,
    suptitle_text: str,
    *,
    legend_strip: float = 0.035,
) -> None:
    """Place a multi-line suptitle and reserve exactly the strip it occupies.

    constrained_layout makes no room for figure-level artists, so the top strip
    is reserved by hand, sized from the suptitle's own line count. Note `rect` is
    (left, bottom, width, height) — passing a top edge as the fourth element
    silently over-reserves and lets the panel titles collide with the suptitle.
    """

    figure.suptitle(suptitle_text, fontsize=TITLE_FONT_SIZE, y=0.995, va="top")
    title_points = (suptitle_text.count("\n") + 1) * TITLE_FONT_SIZE * 1.2 + 14.0
    reserved = title_points / (figure.get_figheight() * 72.0)
    figure.get_layout_engine().set(
        rect=(0.0, legend_strip, 1.0, 1.0 - reserved - legend_strip)
    )
