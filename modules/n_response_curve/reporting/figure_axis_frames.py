"""One padded x/y frame shared by every figure a recipe writes from one overlay.

The cluster recipes each render several figures from subsets of one parent
overlay. Letting each figure autoscale would make a four-level stratum reading
0-135 kg N occupy the same width as the full ladder, and would flatten the dry
season's larger response to look the same height as the wet season's. Pinning
the frame once from the *parent* overlay, before any subsetting, is what makes
the panels comparable to one another and to the stratum they came from.

Both floors are the padded data minimum rather than zero. These are descriptive
overlays, not a magnitude comparison against an absolute origin.

``AXIS_PAD_FRACTION`` is hard-coded rather than configured. It is a drawing
choice with no scientific content: it does not select rows, enter any hash, or
change a reported number, and the three recipes that used to carry their own
copy all carried the same 0.04.

This module is importable by the standalone reporting recipes and is not on the
release path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "AXIS_PAD_FRACTION",
    "SharedAxisLimits",
    "padded_limits",
    "shared_axis_limits",
]


# Fraction of the observed span added to each end of an axis.
AXIS_PAD_FRACTION = 0.04


def padded_limits(
    value_range: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Pad one observed ``(low, high)`` range, or return ``None`` if unusable.

    A degenerate range — a single repeated value — has no span to take a
    fraction of, so it is padded by 5% of its own magnitude instead, with a
    floor of 1.0 so a range at zero still opens to a visible frame.
    """

    if value_range is None:
        return None
    low, high = value_range
    if not (math.isfinite(low) and math.isfinite(high)):
        return None
    span = high - low
    pad = span * AXIS_PAD_FRACTION if span > 0 else max(abs(high), 1.0) * 0.05
    return low - pad, high + pad


@dataclass(frozen=True)
class SharedAxisLimits:
    """One padded x/y frame, applied to every axes a recipe draws."""

    x: tuple[float, float] | None
    y: tuple[float, float] | None

    def apply(self, axes: Any) -> None:
        if self.x is not None:
            axes.set_xlim(*self.x)
        if self.y is not None:
            axes.set_ylim(*self.y)


def shared_axis_limits(overlay: Any) -> SharedAxisLimits:
    """Pin one frame from *overlay*'s summary ranges, before any subsetting.

    *overlay* is any object exposing ``summary.n_rate_range_kg_ha`` and
    ``summary.yield_range_t_ha`` — in practice a
    :class:`~n_response_curve.reporting.source_dataset_overlays.SourceDatasetOverlay`.
    It is typed loosely here so this module stays free of that import.
    """

    return SharedAxisLimits(
        x=padded_limits(overlay.summary.n_rate_range_kg_ha),
        y=padded_limits(overlay.summary.yield_range_t_ha),
    )
