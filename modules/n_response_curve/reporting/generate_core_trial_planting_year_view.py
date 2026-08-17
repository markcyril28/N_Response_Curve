#!/usr/bin/env python3
"""Write a by-planting-year `decades_and_trend` sheet for `core_trial_data`.

The core-trial analogue of LTCCE's
`clusters/by_season/ds/by_planting_year/decades_and_trend_matched_colours.jpeg`:
each planting decade's response-curve cloud drawn as an inset sitting on the
exact calendar stretch of the per-year yield trend it came from, with colour
spent on the quantity (yield at zero N versus response above zero N) rather
than on a design era.

Three things had to change, and none of them is cosmetic:

* **Population.** The released `core_trial_data_source_wide.jpeg` draws the 21
  governed v3 response series (74 observations). Those 21 span only 1991-1995
  plus a single 2018 series, so a decade decomposition of them is degenerate.
  This sheet is drawn from every core-trial response series the promoted
  release's eligibility ledger resolves with at least two distinct finite
  applied-N levels -- 63 series, 158 levels, 1991-2018 -- and states the
  relationship to the governed 21 on its own face.

* **Eligibility.** LTCCE's sheet uses `describe_trajectory`, which requires at
  least three distinct applied-N levels *anchored at zero N*. Applied here that
  admits 10 series, all 1991-1995, because 42 of the 63 core trajectories are
  two-point ladders. The trend panels therefore relax the level requirement to
  two while keeping the zero-N anchor, which is exactly what the two plotted
  quantities need and nothing more: no shape feature is computed, and nothing is
  fitted.

* **Season.** LTCCE's sheet lives under `by_season/ds/`, so it never pools
  seasons. Core-trial dry and wet seasons are both present 1991-1995 but the
  record is wet-only from 2003 on, so a pooled yearly mean would turn a change
  of season composition into an apparent trend. Dry and wet are drawn as two
  separate broken series inside each panel instead.

Exploratory diagnostic, outside the governed release inventory (ANA-11): no
curve is fitted, and nothing here is a governed analysis family.

Usage:
    conda run -n n_response python \\
      modules/n_response_curve/reporting/generate_core_trial_planting_year_view.py \\
      --config scriptCONFIG.toml
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import os
import shutil
import statistics
import sys
import textwrap
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.analysis.values import finite_number  # noqa: E402
from n_response_curve.reporting.generate_raw_dataset_overlays import (  # noqa: E402
    _load_governed_core_inputs,
)
from n_response_curve.reporting.generate_response_curve_season_clusters import (  # noqa: E402
    _AXIS_LABEL_FONT_SIZE,
    _CAPTION_COLUMN_CHARS,
    _CAPTION_COLUMN_OFFSET,
    _CAPTION_FONT_SIZE,
    _CAPTION_LINE_SPACING,
    _CAPTION_PARAGRAPH_GAP_LINES,
    _DISCLAIMER,
    _HEADLINE_FONT_SIZE,
    _HOST_ROW_IN,
    _HOST_TREND_FRACTION,
    _INSET_BOTTOM_FRACTION,
    _INSET_LEGEND_IN,
    _INSET_READOUT_FONT_SIZE,
    _INSET_TICK_FONT_SIZE,
    _INSET_TITLE_FONT_SIZE,
    _INSET_TOP_FRACTION,
    _LEGEND_FONT_SIZE,
    _NARROW_PANEL_IN,
    _PANEL_GUTTER_IN,
    _SHEET_BOTTOM_IN,
    _SHEET_LEFT_IN,
    _SHEET_RIGHT_IN,
    _SHEET_WIDTH_IN,
    _TICK_FONT_SIZE,
    _TREND_GAP_IN,
    _TREND_LEGEND_IN,
    _TREND_ROW_IN,
    _TREND_XAXIS_IN,
    _ZERO_N_TREATMENT_CLASS,
    _draw_overlay_on_axes,
    _save_figure,
    _shared_axis_limits,
    _text_inches,
    _treatment_class_colours,
)
from n_response_curve.reporting.response_curve_clusters import (  # noqa: E402
    subset_overlay,
)
from n_response_curve.reporting.source_dataset_overlays import (  # noqa: E402
    SourceObservation,
    _finalize_overlay,
)

SOURCE_NAME = "core_trial_data"
LEDGER_RELATIVE_PATH = "tables/quality/analysis_eligibility_ledger.csv"

# The annotated sheet carries its disclosures on its own face; the separated
# pair moves them to `NOTES_FILENAME` and leaves the plates alone. The figure
# names the notes file, so these two constants are read by both writers rather
# than spelled out at either site.
ANNOTATED_FIGURE_FILENAME = "decades_and_trend.jpeg"
FIGURE_ONLY_FILENAME = "decades_and_trend_figure.jpeg"
NOTES_FILENAME = "decades_and_trend_notes.md"

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "WF/04_Response_Curves/z_n_response_full/overlay/source_dataset"
    / SOURCE_NAME
    / "clusters/by_planting_year"
)

# The sheet reproduces LTCCE's `decades_and_trend_matched_colours.jpeg`, whose
# 34 inches carry six decades. Reusing that inches-per-year density rather than
# the fixed width keeps three core-trial decades proportioned the same way
# instead of stretching three insets across a sheet sized for twice as many.
_REFERENCE_DECADE_COUNT = 6
_INCHES_PER_YEAR = (
    _SHEET_WIDTH_IN - _SHEET_LEFT_IN - _SHEET_RIGHT_IN
) / (_REFERENCE_DECADE_COUNT * 10)

# Colour is spent on the quantity, exactly as on the reference sheet, so season
# takes marker shape and line style. Two levels only, so no ladder of shapes is
# needed -- but the two must differ in *both* channels: at this point size a
# circle and a square alone are one smudge in a JPEG.
_SEASON_ORDER = ("dry", "wet")
_SEASON_STYLES = {
    "dry": ("o", "-"),
    "wet": ("s", (0, (5, 2))),
}
_SEASON_LABELS = {"dry": "dry season", "wet": "wet season"}

_TREND_FEATURES = (
    ("yield_at_zero_n_t_ha", "Yield at zero N (t/ha)"),
    ("response_above_zero_n_t_ha", "Response above zero N (t/ha)"),
)

# A gap shorter than this is disclosed in the caption but not lettered inside
# the panel: three years is about where a label fits between the flanking
# points without touching either.
_MIN_LABELLED_GAP_YEARS = 3


# --------------------------------------------------------------------------
# Ledger population
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Series:
    """One resolved core-trial response series, described from observed points."""

    series_uid: str
    year_label: str
    year_bounds: tuple[int, int]
    decade: str
    season: str
    province: str
    study_id: str
    variety: str
    levels: tuple[float, ...]
    yields: tuple[float, ...]

    @property
    def is_anchored(self) -> bool:
        """Whether the series observed zero applied N."""

        return bool(self.levels) and self.levels[0] == 0.0

    @property
    def placeable_year(self) -> int | None:
        """The single calendar year the series can be drawn at, if there is one."""

        low, high = self.year_bounds
        return low if low == high else None

    @property
    def yield_at_zero_n_t_ha(self) -> float | None:
        return self.yields[0] if self.is_anchored else None

    @property
    def response_above_zero_n_t_ha(self) -> float | None:
        if not self.is_anchored:
            return None
        return max(self.yields) - self.yields[0]

    def feature(self, name: str) -> float | None:
        return getattr(self, name)


def _package_path(config_path: Path) -> Path:
    """The promoted release the ledger is read out of, per `[custom_overlays]`."""

    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    table = config.get("custom_overlays")
    if not isinstance(table, dict):
        raise ValueError("The configuration must contain a [custom_overlays] table")
    raw = table.get("package_path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("[custom_overlays].package_path must be a nonempty string")
    package = Path(raw)
    if not package.is_absolute():
        package = config_path.resolve().parent / package
    return package.resolve()


def _governed_series_uids(config_path: Path) -> tuple[str, ...]:
    """The governed v3 core membership, loaded through its own verifier.

    Called for two reasons and both matter: it is the only honest source of the
    21/74 counts this sheet reconciles itself against, and it verifies the
    eligibility ledger against the package's `CHECKSUMS.sha256` before anything
    reads it. The wider population below is taken from the same file only after
    that check has passed.
    """

    with config_path.open("rb") as handle:
        table = tomllib.load(handle).get("custom_overlays") or {}
    threshold = table.get("yield_threshold_t_ha")
    if not isinstance(threshold, (int, float)):
        raise ValueError("[custom_overlays].yield_threshold_t_ha must be a number")
    inputs = _load_governed_core_inputs(
        config_path,
        yield_threshold_t_ha=float(threshold),
    )
    return inputs.response_series_uids


def _year_bounds(label: str) -> tuple[int, int] | None:
    """Parse a recorded planting year, which may be a range, into (first, last).

    The source records three series against a span rather than a year
    (`1991-1994`, `2003-04`). They are real observations and belong in the
    decade inset; they have no place on a year axis. Returning both ends lets
    the caller do both, and lets it fail loudly if a span ever crosses a decade
    boundary, where the inset placement would be ambiguous.
    """

    text = label.strip()
    if not text:
        return None
    head, _, tail = text.partition("-")
    head = head.strip()
    tail = tail.strip()
    if not (len(head) == 4 and head.isdigit()):
        return None
    first = int(head)
    if not tail:
        return first, first
    if len(tail) == 2 and tail.isdigit():
        last = first - first % 100 + int(tail)
    elif len(tail) == 4 and tail.isdigit():
        last = int(tail)
    else:
        return None
    if last < first:
        return None
    return first, last


def _decade_of(bounds: tuple[int, int], label: str) -> str:
    first, last = bounds
    if first // 10 != last // 10:
        raise ValueError(
            f"Planting year {label!r} spans two decades, so the inset it belongs "
            "in is ambiguous; the sheet's decade placement has to be revisited"
        )
    return f"{first // 10 * 10}s"


def _read_series(ledger_path: Path) -> tuple[_Series, ...]:
    """Every core-trial series the ledger resolves at two or more applied-N levels.

    Duplicated levels are averaged within the series rather than dropped: the
    two plotted quantities are the series' zero-N yield and its observed
    maximum, and both are defined on the level means. LTCCE's clustering path
    refuses duplicated levels instead, because the shape features it computes
    are not.
    """

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    rows_by_series: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    with ledger_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("source_name") != SOURCE_NAME:
                continue
            series_uid = str(row.get("response_series_uid") or "").strip()
            if not series_uid:
                continue
            rows_by_series[series_uid].append(row)

    series: list[_Series] = []
    for series_uid, rows in sorted(rows_by_series.items()):
        by_level: dict[float, list[float]] = collections.defaultdict(list)
        for row in rows:
            n_rate = finite_number(row.get("n_rate_kg_ha"))
            yield_value = finite_number(row.get("yield_t_ha"))
            if n_rate is None or yield_value is None:
                continue
            by_level[float(n_rate)].append(float(yield_value))
        if len(by_level) < 2:
            continue
        bounds = _year_bounds(str(rows[0].get("planting_year") or ""))
        if bounds is None:
            continue
        label = str(rows[0].get("planting_year") or "").strip()
        ordered = sorted(by_level.items())
        series.append(
            _Series(
                series_uid=series_uid,
                year_label=label,
                year_bounds=bounds,
                decade=_decade_of(bounds, label),
                season=str(rows[0].get("season_normalized") or "").strip(),
                province=str(rows[0].get("province") or "").strip() or "site unrecorded",
                study_id=str(rows[0].get("study_id") or "").strip(),
                variety=str(rows[0].get("rice_variety") or "").strip(),
                levels=tuple(level for level, _ in ordered),
                yields=tuple(statistics.fmean(values) for _, values in ordered),
            )
        )
    return tuple(series)


def _build_overlay(series: Sequence[_Series]):
    """An overlay of the same shape the inset renderer already knows how to draw.

    Treatment classes are the two strings `_read_core_trial_overlay` uses, so
    the property-cycle colours the insets take are the ones the trend panels
    below claim to be carrying.
    """

    observations: dict[str, list[SourceObservation]] = {}
    for entry in series:
        observations[entry.series_uid] = [
            SourceObservation(
                trajectory_id=entry.series_uid,
                n_rate_kg_ha=level,
                yield_t_ha=value,
                treatment_class=(
                    _ZERO_N_TREATMENT_CLASS if level == 0.0 else "mineral N rate"
                ),
            )
            for level, value in zip(entry.levels, entry.yields, strict=True)
        ]
    return _finalize_overlay(
        source_name=SOURCE_NAME,
        source_rows=sum(len(values) for values in observations.values()),
        excluded_observation_count=0,
        observations_by_trajectory=observations,
    )


# --------------------------------------------------------------------------
# Trend records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _TrendPoint:
    """One planting year of one season, over its own qualifying series."""

    year: int
    season: str
    series_count: int
    means: dict[str, float]
    standard_errors: dict[str, float]
    province: str
    province_is_mixed: bool


def _trend_points(series: Sequence[_Series]) -> tuple[_TrendPoint, ...]:
    """Per-year, per-season means of the two zero-N-anchored quantities.

    A plain mean over the year's qualifying series, and the standard error
    across them where more than one qualifies. Deliberately *not* LTCCE's
    context-weighted average: that averages replicates inside a
    design-site-year-season-variety-ladder key, and this source records no
    replicate structure at all -- a series' rows are its applied-N levels. The
    weighting would be a no-op and the claim would be false.
    """

    grouped: dict[tuple[int, str], list[_Series]] = collections.defaultdict(list)
    for entry in series:
        year = entry.placeable_year
        if year is None or not entry.is_anchored or not entry.season:
            continue
        grouped[(year, entry.season)].append(entry)

    points: list[_TrendPoint] = []
    for (year, season), members in sorted(grouped.items()):
        means: dict[str, float] = {}
        errors: dict[str, float] = {}
        for feature, _label in _TREND_FEATURES:
            values = [
                value
                for value in (member.feature(feature) for member in members)
                if value is not None
            ]
            means[feature] = statistics.fmean(values) if values else math.nan
            errors[feature] = (
                statistics.stdev(values) / math.sqrt(len(values))
                if len(values) > 1
                else math.nan
            )
        provinces = collections.Counter(member.province for member in members)
        points.append(
            _TrendPoint(
                year=year,
                season=season,
                series_count=len(members),
                means=means,
                standard_errors=errors,
                province=provinces.most_common(1)[0][0],
                province_is_mixed=len(provinces) > 1,
            )
        )
    return tuple(points)


def _decade_spans(decades: Sequence[str]) -> tuple[tuple[str, int, int], ...]:
    """(label, first year, last year) for each decade band, oldest first."""

    spans = []
    for decade in decades:
        start = int(decade.rstrip("s"))
        spans.append((decade, start, start + 9))
    return tuple(spans)


def _calendar_runs(points: Sequence[_TrendPoint]) -> list[list[_TrendPoint]]:
    """Split one season's points at any year it has nothing at, so the line breaks.

    Drawing straight across a gap would read as an observed change over years
    that were never observed, which on this source is most of them.
    """

    runs: list[list[_TrendPoint]] = []
    for point in points:
        if runs and point.year == runs[-1][-1].year + 1:
            runs[-1].append(point)
        else:
            runs.append([point])
    return runs


def _site_eras(series: Sequence[_Series]) -> tuple[tuple[str, int, int], ...]:
    """Contiguous runs of one dominant province, as (province, first, last).

    The structural stand-in for LTCCE's applied-N ladder eras, and for the same
    reason: the sites do not overlap in time, so a step across a boundary is a
    change of experiment and site rather than anything the fertilizer did.

    Counted over the individual series of each year, not over the year-season
    points: a point that is itself an even split between two sites has no
    dominant one to contribute, and letting it carry its whole series count to
    whichever site happened to be read first invented a one-year era.
    """

    by_year: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for entry in series:
        year = entry.placeable_year
        if year is None or not entry.is_anchored or not entry.season:
            continue
        by_year[year][entry.province] += 1

    eras: list[list[Any]] = []
    for year in sorted(by_year):
        province = by_year[year].most_common(1)[0][0]
        if eras and eras[-1][0] == province:
            eras[-1][2] = year
            continue
        eras.append([province, year, year])
    return tuple((province, first, last) for province, first, last in eras)


@dataclass(frozen=True)
class _Gap:
    """A stretch of the calendar axis with no trend on it, and why."""

    first: int
    last: int
    has_series: bool

    def label(self) -> str:
        span = f"{self.first}-{self.last}" if self.first != self.last else f"{self.first}"
        if self.has_series:
            return f"{span}\nseries above,\nnone anchored\nat zero N"
        return f"{span}\nno series"

    def prose(self) -> str:
        span = f"{self.first}-{self.last}" if self.first != self.last else f"{self.first}"
        if self.has_series:
            return (
                f"{span} (the inset above is populated, but no series of those "
                "years observed zero N)"
            )
        return f"{span} (no series of two or more applied-N levels at all)"


def _trend_gaps(
    series: Sequence[_Series],
    points: Sequence[_TrendPoint],
    spans: Sequence[tuple[str, int, int]],
) -> tuple[_Gap, ...]:
    """Every stretch of the drawn decades that carries no trend point.

    Two kinds, and the difference is the whole point: a year with no qualifying
    series at all, and a year whose series are drawn in the inset above but
    observed no zero N, so they can enter neither panel. Without the second
    label the sheet shows a populated 2010s inset over an empty trend and looks
    broken.
    """

    plotted = {point.year for point in points}
    covered: set[int] = set()
    for entry in series:
        first, last = entry.year_bounds
        covered.update(range(first, last + 1))

    gaps: list[_Gap] = []
    run: list[int] = []
    for year in range(spans[0][1], spans[-1][2] + 1):
        if year in plotted:
            if run:
                gaps.append(
                    _Gap(run[0], run[-1], any(y in covered for y in run))
                )
                run = []
            continue
        run.append(year)
    if run:
        gaps.append(_Gap(run[0], run[-1], any(y in covered for y in run)))
    return tuple(gaps)


# --------------------------------------------------------------------------
# Read-outs and caption
# --------------------------------------------------------------------------


def _era_text(eras: Sequence[tuple[str, int, int]]) -> str:
    """`Laguna 1991-2005, Pangasinan 2009-2010` — the site spans, in order."""

    return ", ".join(
        f"{name} {first}-{last}" if first != last else f"{name} {first}"
        for name, first, last in eras
    )


def _gap_text(gaps: Sequence[_Gap]) -> str:
    """Prose for *every* stretch with no trend on it, in axis order.

    All of them, not only the ones the panels letter: a one- or two-year stretch
    is too narrow to write inside but is still a stretch of the axis the reader
    can see nothing on, and leaving it out of the caption is the difference
    between a disclosed gap and an unexplained one.
    """

    return "; ".join(gap.prose() for gap in gaps)


def _inset_readout(
    members: Sequence[_Series], total: int, *, compact: bool
) -> str:
    anchored = [entry for entry in members if entry.is_anchored]
    share = 100 * len(members) / total
    years = sorted(bound for entry in members for bound in entry.year_bounds)
    year_text = f"{years[0]}-{years[-1]}" if years[0] != years[-1] else f"{years[0]}"
    seasons = collections.Counter(entry.season or "unrecorded" for entry in members)
    season_text = ", ".join(
        f"{count} {name}" for name, count in sorted(seasons.items())
    )
    mean_zero_n = (
        statistics.fmean([entry.yield_at_zero_n_t_ha for entry in anchored])
        if anchored
        else None
    )
    mean_response = (
        statistics.fmean([entry.response_above_zero_n_t_ha for entry in anchored])
        if anchored
        else None
    )
    level_count = sum(len(entry.levels) for entry in members)

    if compact:
        lines = [f"n={len(members)} ({share:.0f}%)", year_text, season_text]
        if mean_zero_n is None or mean_response is None:
            lines.append("no zero-N anchor")
        else:
            lines.append(f"0N {mean_zero_n:.2f}")
            lines.append(f"+N {mean_response:.2f}")
        return "\n".join(lines)

    lines = [
        f"{len(members)} series ({share:.0f}%), {level_count} levels",
        year_text,
        f"{season_text} season",
        f"{len({entry.study_id for entry in members})} studies, "
        f"{len({entry.province for entry in members})} sites, "
        f"{len({entry.variety for entry in members})} varieties",
    ]
    if mean_zero_n is None or mean_response is None:
        lines.append("none anchored at zero N — no trend below")
    else:
        lines.append(
            f"{len(anchored)} anchored at zero N: mean zero-N {mean_zero_n:.2f} t/ha"
        )
        lines.append(f"mean response {mean_response:.2f} t/ha")
    return "\n".join(lines)


def _season_response_note(points: Sequence[_TrendPoint]) -> str | None:
    """The dry/wet contrast the season split makes visible, or nothing.

    Stated only where both seasons are actually present in the same years, and
    computed rather than asserted, because it is the one sentence on the sheet
    that makes a claim about the data instead of about the drawing.
    """

    shared = {point.year for point in points if point.season == "dry"} & {
        point.year for point in points if point.season == "wet"
    }
    if len(shared) < 2:
        return None
    by_season = {
        season: [
            point.means["response_above_zero_n_t_ha"]
            for point in points
            if point.season == season and point.year in shared
        ]
        for season in _SEASON_ORDER
    }
    dry = statistics.fmean(by_season["dry"])
    wet = statistics.fmean(by_season["wet"])
    if not (wet > 0 and dry > wet):
        return None
    first, last = min(shared), max(shared)
    return (
        f"across the {len(shared)} years both seasons are recorded in "
        f"({first}-{last}) the dry-season response above zero N averages "
        f"{dry:.2f} t/ha against the wet season's {wet:.2f} — a contrast that "
        "only a pooled yearly mean could hide, and the reason the two are drawn "
        "apart"
    )


def _disclosures(
    series: Sequence[_Series],
    points: Sequence[_TrendPoint],
    gaps: Sequence[_Gap],
    eras: Sequence[tuple[str, int, int]],
    governed_count: int,
    governed_observations: int,
) -> list[str]:
    anchored = [entry for entry in series if entry.is_anchored]
    ranged = [entry for entry in series if entry.placeable_year is None]
    strict = [
        entry
        for entry in anchored
        if len(entry.levels) >= 3 and entry.placeable_year is not None
    ]
    replicated = [point for point in points if point.series_count > 1]
    mixed = [point for point in points if point.province_is_mixed]
    run_count = sum(
        len(_calendar_runs([point for point in points if point.season == season]))
        for season in _SEASON_ORDER
    )
    season_note = _season_response_note(points)
    era_text = _era_text(eras)
    replicated_text = ", ".join(
        f"{point.year} {point.season} (n={point.series_count})"
        for point in replicated
    )
    gap_text = _gap_text(gaps)
    level_count = sum(len(entry.levels) for entry in series)
    two_point_count = sum(1 for entry in series if len(entry.levels) == 2)
    study_count = len({entry.study_id for entry in series})
    site_count = len({entry.province for entry in series})

    return [
        "colour carries the quantity, not the design: orange is yield at zero N "
        "— the orange points in every inset, and the panel that tracks their "
        "mean by year — and blue is the response above zero N, computed from "
        "the same series' own observed maximum relative to its orange point "
        "rather than read off either panel directly",
        "the planting season is the marker shape and the line style, and dry "
        "and wet are drawn as two separate broken series rather than pooled "
        "into one yearly mean: both seasons are recorded 1991-1995 but the "
        "record is wet-only from 2003 on, so a pooled mean would turn a change "
        "of season composition into an apparent trend",
        f"{len(series)} response series ({level_count} applied-N levels) that the "
        "promoted release's eligibility ledger resolves with two or more "
        "distinct levels. This is wider than the "
        f"{governed_count} governed series ({governed_observations} observations) "
        "drawn in `core_trial_data_source_wide.jpeg`, and deliberately so: "
        f"{two_point_count} of the {len(series)} are two-point ladders, and the "
        "reference sheet's eligibility — three or more distinct levels anchored "
        f"at zero N — admits only {len(strict)} core series, all of them "
        "1991-1995",
        "the insets — every one of those series against applied N, each a "
        "recorded stratum that nothing was clustered to produce, and each on "
        "the same frame at the same size, so any two can be compared directly. "
        "The means on each card are over every anchored series of that whole "
        "decade, pooled across its years and its seasons, so a card is not "
        "expected to equal any single marker in the panels below — those are "
        "per year and per season",
        "each inset sits on its own calendar decade of the axis beneath it, "
        "whole decade to whole decade",
        "the two trend panels — yield at zero N by planting year, which is the "
        "left-hand end of every anchored curve in the inset above it, and the "
        "response above zero N, which is that series' observed maximum less "
        "the same point. Each marker is a plain mean over that year and "
        "season's anchored series and the bar its standard error across them, "
        "drawn only where more than one series qualifies; both panels are on "
        "one shared scale, so a change of a given size has the same slope in "
        "each",
        "no replicate weighting is applied, because this source records none: a "
        "series' ledger rows are its applied-N levels, not repeated plots of "
        "one treatment",
        f"this is a compilation of {study_count} independent studies at "
        f"{site_count} sites, not one experiment: the line breaks at every site "
        f"change (dashed, named along the top panel — {era_text}) and at every "
        f"year with nothing to plot, leaving {run_count} disconnected runs. A "
        "step across any of those breaks is a change of study, site, variety "
        "and design, not a response to fertilizer",
        f"{len(series) - len(anchored)} of the {len(series)} series observed no "
        "zero-N treatment, so they are drawn in the insets but enter neither "
        "trend panel"
        + (
            ". Every stretch of the axis with no trend on it, lettered inside "
            f"the panels where it is at least {_MIN_LABELLED_GAP_YEARS} years "
            f"wide: {gap_text}"
            if gap_text
            else ""
        ),
        *(
            [
                f"{len(ranged)} series are recorded against a span of years "
                f"({', '.join(sorted({entry.year_label for entry in ranged}))}) "
                "rather than one year; each is drawn in the decade inset its "
                "span falls inside and left off the year axis, which has no "
                "position for it"
            ]
            if ranged
            else []
        ),
        "most years contribute a single series, so most markers carry no error "
        f"bar; the ones that do are {replicated_text}",
        *(
            [
                "years drawing on more than one site (banded on the dominant "
                "one): "
                + ", ".join(f"{point.year} {point.season}" for point in mixed)
            ]
            if mixed
            else []
        ),
        *([season_note] if season_note else []),
        _DISCLAIMER,
    ]


# --------------------------------------------------------------------------
# The sheet
# --------------------------------------------------------------------------


def _write_decades_and_trend(
    series: Sequence[_Series],
    overlay,
    limits,
    governed_count: int,
    governed_observations: int,
    destination: Path,
    *,
    separate_notes: bool = False,
) -> None:
    """Each planting decade's curves drawn on the stretch of trend they came from.

    `separate_notes` writes the plates-only variant: the disclosure bullets come
    off the sheet and go to `decades_and_trend_notes.md`, which
    `_write_notes` renders from the same `_disclosures` call, so the two cannot
    drift apart. What stays on the figure is what a figure must not travel
    without — the headline that identifies it, the ANA-11 disclaimer, and a
    pointer to the notes file. Stripping those too would produce an
    unattributed, undisclosed plot.
    """

    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    decades = sorted({entry.decade for entry in series})
    spans = _decade_spans(decades)
    points = _trend_points(series)
    if len(decades) < 2 or len(points) < 2:
        raise SystemExit(
            "The sheet needs at least two decades and two trend points; the "
            f"ledger yielded {len(decades)} and {len(points)}"
        )
    gaps = _trend_gaps(series, points, spans)
    eras = _site_eras(series)

    by_decade = {
        decade: [entry for entry in series if entry.decade == decade]
        for decade in decades
    }
    total = len(series)

    x_lo = spans[0][1] - 0.5
    x_hi = spans[-1][2] + 0.5
    visible = [(first - 0.5, last + 0.5) for _, first, last in spans]

    colours = _treatment_class_colours(overlay)
    zero_n_colour = colours.get(_ZERO_N_TREATMENT_CLASS, "C1")
    mineral_colour = next(
        (
            colour
            for label, colour in colours.items()
            if label != _ZERO_N_TREATMENT_CLASS
        ),
        "C0",
    )
    panel_colours = {
        "yield_at_zero_n_t_ha": zero_n_colour,
        "response_above_zero_n_t_ha": mineral_colour,
    }

    year_span = x_hi - x_lo
    plot_width = _INCHES_PER_YEAR * year_span
    sheet_width = plot_width + _SHEET_LEFT_IN + _SHEET_RIGHT_IN

    # The reference measure is calibrated to a 34-inch sheet. Three decades make
    # a narrower one, and a column set to the wide measure would run off it, so
    # the wrap is taken from the column this sheet actually has.
    caption_chars = max(
        54,
        min(
            _CAPTION_COLUMN_CHARS,
            int(plot_width * _CAPTION_COLUMN_OFFSET * 72.0 / (_CAPTION_FONT_SIZE * 0.55)),
        ),
    )
    # Wrapped rather than set on one line: the reference headline fits because
    # its sheet is 34 inches wide. On three decades' worth of width the same
    # sentence at the same size runs off both edges, and `figure.text` clips it
    # without saying so.
    headline = textwrap.fill(
        f"source={SOURCE_NAME} — each planting decade's response curves drawn on "
        "the stretch of the yield trend they came from, by season",
        width=max(
            40,
            int(plot_width * 72.0 / (_HEADLINE_FONT_SIZE * 0.55)),
        ),
        break_long_words=False,
    )
    headline_lines = headline.count("\n") + 1
    caption_line_in = _text_inches(_CAPTION_FONT_SIZE, spacing=_CAPTION_LINE_SPACING)
    paragraph_gap_in = caption_line_in * _CAPTION_PARAGRAPH_GAP_LINES

    footer = textwrap.fill(
        f"{_DISCLAIMER}. Every disclosure this sheet depends on — population, "
        "eligibility, the season split, the site breaks and the stretches with "
        f"no trend on them — is in `{NOTES_FILENAME}`, beside this figure; read "
        "it with the sheet, not after it",
        width=max(
            40, int(plot_width * 72.0 / (_CAPTION_FONT_SIZE * 0.55))
        ),
        break_long_words=False,
    )
    footer_lines = footer.count("\n") + 1

    caption_columns: tuple[list[str], ...] = ((), ())
    caption_counts: tuple[list[int], ...] = ((), ())
    if separate_notes:
        caption_block_in = footer_lines * caption_line_in + 0.10
    else:
        bullets = [
            textwrap.fill(
                entry,
                width=caption_chars,
                initial_indent="—  ",
                subsequent_indent="    ",
                break_long_words=False,
            )
            for entry in _disclosures(
                series, points, gaps, eras, governed_count, governed_observations
            )
        ]
        counts = [bullet.count("\n") + 1 for bullet in bullets]
        weights = [count + _CAPTION_PARAGRAPH_GAP_LINES for count in counts]
        total_weight = sum(weights)
        split = min(
            range(1, max(len(bullets), 2)),
            key=lambda index: abs(2 * sum(weights[:index]) - total_weight),
        )
        caption_columns = (bullets[:split], bullets[split:])
        caption_counts = (counts[:split], counts[split:])
        caption_block_in = max(
            sum(column) * caption_line_in + max(len(column) - 1, 0) * paragraph_gap_in
            for column in caption_counts
        )
    headline_strip_in = _text_inches(_HEADLINE_FONT_SIZE, headline_lines) + 0.24
    title_strip_in = headline_strip_in + caption_block_in + 0.45

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

    try:
        low = high = None
        for point in points:
            for feature, _label in _TREND_FEATURES:
                value = point.means[feature]
                error = point.standard_errors[feature]
                error = error if error == error else 0.0
                if value != value:
                    continue
                low = value - error if low is None else min(low, value - error)
                high = value + error if high is None else max(high, value + error)
        if low is None or high is None or not high > low:
            low, high = 0.0, 1.0
        pad = 0.06 * (high - low)
        floor, ceiling = low - pad, high + pad
        shared_ticks = [
            tick
            for tick in plt.MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]).tick_values(
                floor, ceiling
            )
            if floor <= tick <= ceiling
        ]

        trend_axes: list[Any] = []
        for feature, label in _TREND_FEATURES:
            is_host = not trend_axes
            bottom_in = host_bottom if is_host else response_bottom
            row_in = _HOST_ROW_IN if is_host else _TREND_ROW_IN
            axes = figure.add_axes(
                _box(plot_left, bottom_in, plot_width, row_in),
                sharex=trend_axes[0] if trend_axes else None,
            )
            trend_axes.append(axes)

            for index, (band_lo, band_hi) in enumerate(visible):
                if index % 2 == 0:
                    axes.axvspan(band_lo, band_hi, color="0.94", zorder=0, linewidth=0)
                axes.axvline(band_hi, color="0.85", linewidth=0.8, zorder=0)

            colour = panel_colours[feature]
            for season in _SEASON_ORDER:
                marker, linestyle = _SEASON_STYLES[season]
                season_points = [
                    point for point in points if point.season == season
                ]
                for run_index, run in enumerate(_calendar_runs(season_points)):
                    axes.errorbar(
                        [point.year for point in run],
                        [point.means[feature] for point in run],
                        yerr=[point.standard_errors[feature] for point in run],
                        marker=marker,
                        linestyle=linestyle,
                        markersize=9,
                        linewidth=2.2,
                        capsize=4.0,
                        color=colour,
                        ecolor=colour,
                        elinewidth=1.1,
                        zorder=3,
                        label=(
                            _SEASON_LABELS[season]
                            if is_host and run_index == 0
                            else None
                        ),
                    )

            for _province, first, _last in eras[1:]:
                axes.axvline(
                    first - 0.5, color="0.55", linewidth=1.1, ls="--", zorder=1
                )

            # Each panel's own data band, not the axes: the host is stretched so
            # its points occupy only the bottom `_HOST_TREND_FRACTION`.
            band_middle = _HOST_TREND_FRACTION / 2 if is_host else 0.5
            for gap in gaps:
                if gap.last - gap.first + 1 < _MIN_LABELLED_GAP_YEARS:
                    continue
                axes.text(
                    (gap.first + gap.last) / 2,
                    band_middle,
                    gap.label(),
                    transform=axes.get_xaxis_transform(),
                    ha="center",
                    va="center",
                    fontsize=_LEGEND_FONT_SIZE,
                    color="0.50",
                    linespacing=1.5,
                    zorder=2,
                )
            if is_host:
                # Just above the host's own data band, not at the top of its
                # axes: the axes runs the whole 11.6-inch row, so `0.96` put
                # these among the inset titles and read as labelling the insets.
                for province, first, last in eras:
                    axes.text(
                        (first + last) / 2,
                        _HOST_TREND_FRACTION + 0.012,
                        f"site: {province}",
                        transform=axes.get_xaxis_transform(),
                        ha="center",
                        va="bottom",
                        fontsize=_LEGEND_FONT_SIZE,
                        color="0.45",
                        zorder=2,
                    )

            axes.set_ylabel(
                label,
                y=band_middle,
                fontsize=_AXIS_LABEL_FONT_SIZE,
            )
            axes.grid(axis="y", alpha=0.25, linewidth=0.6)
            axes.tick_params(labelsize=_TICK_FONT_SIZE)
            axes.set_yticks(shared_ticks)
            axes.set_ylim(
                floor,
                floor + (ceiling - floor) / _HOST_TREND_FRACTION
                if is_host
                else ceiling,
            )

        trend_axes[0].set_xlim(x_lo, x_hi)
        trend_axes[0].tick_params(labelbottom=False)
        trend_axes[-1].set_xticks(list(range(spans[0][1], spans[-1][2] + 1, 5)))
        trend_axes[-1].set_xticks(
            list(range(spans[0][1], spans[-1][2] + 1)), minor=True
        )
        trend_axes[-1].set_xlabel("Planting year", fontsize=_AXIS_LABEL_FONT_SIZE)

        inset_axes: list[Any] = []
        for index, (decade, (band_lo, band_hi)) in enumerate(
            zip(decades, visible, strict=True)
        ):
            members = by_decade[decade]
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

            _draw_overlay_on_axes(
                subset_overlay(
                    overlay, [entry.series_uid for entry in members]
                ),
                axes,
            )
            limits.apply(axes)
            axes.grid(alpha=0.25, linewidth=0.6)
            axes.set_xticks([0, 50, 100, 150, 200])
            axes.tick_params(labelsize=_INSET_TICK_FONT_SIZE)
            axes.set_title(
                decade,
                fontsize=_INSET_TITLE_FONT_SIZE,
                fontweight="bold",
                pad=6,
            )
            axes.text(
                0.97,
                0.03,
                _inset_readout(
                    members, total, compact=panel_inches < _NARROW_PANEL_IN
                ),
                transform=axes.transAxes,
                ha="right",
                va="bottom",
                fontsize=_INSET_READOUT_FONT_SIZE,
                color="0.25",
                linespacing=1.4,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.80",
                    "linewidth": 0.6,
                    "alpha": 0.92,
                    "pad": 4.0,
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
        trend_handles = list(trend_handles) + [
            Line2D([], [], color="0.55", linewidth=1.1, ls="--")
        ]
        trend_labels = list(trend_labels) + ["site change"]
        figure.legend(
            trend_handles,
            trend_labels,
            loc="center",
            bbox_to_anchor=_box(
                plot_left, _SHEET_BOTTOM_IN, plot_width, _TREND_LEGEND_IN
            ),
            bbox_transform=figure.transFigure,
            ncol=len(trend_labels),
            fontsize=_LEGEND_FONT_SIZE,
            frameon=False,
            title=(
                "planting season — marker shape and line style; colour on this "
                "sheet is the quantity, not the season"
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
            linespacing=1.35,
        )
        caption_top = headline_top - headline_strip_in / height
        if separate_notes:
            figure.text(
                0.5,
                caption_top,
                footer,
                ha="center",
                va="top",
                fontsize=_CAPTION_FONT_SIZE,
                color="0.35",
                linespacing=_CAPTION_LINE_SPACING,
            )
        for column_index, (column, column_counts) in enumerate(
            zip(caption_columns, caption_counts, strict=True)
        ):
            cursor = caption_top
            for bullet, bullet_lines in zip(column, column_counts, strict=True):
                figure.text(
                    (plot_left + column_index * plot_width * _CAPTION_COLUMN_OFFSET)
                    / sheet_width,
                    cursor,
                    bullet,
                    ha="left",
                    va="top",
                    fontsize=_CAPTION_FONT_SIZE,
                    color="0.25",
                    linespacing=_CAPTION_LINE_SPACING,
                )
                cursor -= (bullet_lines * caption_line_in + paragraph_gap_in) / height
        rule_y = (height - title_strip_in + 0.18) / height
        figure.add_artist(
            Line2D(
                [plot_left / sheet_width, (plot_left + plot_width) / sheet_width],
                [rule_y, rule_y],
                transform=figure.transFigure,
                color="0.82",
                linewidth=1.0,
            )
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------
# README
# --------------------------------------------------------------------------


def _write_notes(
    series: Sequence[_Series],
    points: Sequence[_TrendPoint],
    gaps: Sequence[_Gap],
    eras: Sequence[tuple[str, int, int]],
    governed_count: int,
    governed_observations: int,
    destination: Path,
) -> None:
    """The separated sheet's disclosures, from the same source the sheet uses.

    Rendered from `_disclosures` rather than restated, because a notes file that
    can disagree with the figure it belongs to is worse than no notes file: the
    reader has no way to tell which of the two is current.
    """

    disclosures = _disclosures(
        series, points, gaps, eras, governed_count, governed_observations
    )
    lines = [
        f"# `{FIGURE_ONLY_FILENAME}` — how to read it",
        "",
        f"Notes for **`{FIGURE_ONLY_FILENAME}`**: each planting decade's response",
        "curves drawn on the stretch of the yield trend they came from, by season.",
        "The plates and these notes are one document — the figure is not",
        "interpretable without them, which is why the annotated single sheet",
        f"`{ANNOTATED_FIGURE_FILENAME}` also exists and carries the same text on",
        "its own face.",
        "",
        "## Disclosures",
        "",
    ]
    # Capitalized only. The wording is the sheet's own, character for character,
    # so the two can be diffed; on the sheet these are caption fragments under a
    # dash, and in a markdown document they are sentences.
    lines += [f"- {entry[0].upper()}{entry[1:]}" for entry in disclosures]
    lines += [
        "",
        "## Provenance",
        "",
        f"- Source: `{SOURCE_NAME}`, read from the promoted release's",
        f"  `{LEDGER_RELATIVE_PATH}`, verified against the package's",
        "  `CHECKSUMS.sha256` before use.",
        f"- {len(series)} response series, "
        f"{sum(len(entry.levels) for entry in series)} applied-N levels, "
        f"{min(entry.year_bounds[0] for entry in series)}-"
        f"{max(entry.year_bounds[1] for entry in series)}.",
        "- Regenerate both figures and this file together; see `README.md`.",
        "",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def _write_readme(
    series: Sequence[_Series],
    points: Sequence[_TrendPoint],
    gaps: Sequence[_Gap],
    eras: Sequence[tuple[str, int, int]],
    governed_count: int,
    governed_observations: int,
    destination: Path,
) -> None:
    anchored = [entry for entry in series if entry.is_anchored]
    decades = sorted({entry.decade for entry in series})
    replicated_text = ", ".join(
        f"{point.year} {point.season} (n={point.series_count})"
        for point in points
        if point.series_count > 1
    )
    lines = [
        f"# `{SOURCE_NAME}` — planting-decade response curves on the yield trend",
        "",
        f"{_DISCLAIMER[0].upper()}{_DISCLAIMER[1:]}.",
        "",
        "## The same figure in two forms",
        "",
        "Both are the core-trial analogue of LTCCE's",
        "`ltcce/clusters/by_season/ds/by_planting_year/"
        "decades_and_trend_matched_colours.jpeg`: each planting decade's",
        "response-curve cloud drawn as an inset sitting on the calendar stretch",
        "of the per-year trend it came from, with colour spent on the quantity",
        "(yield at zero N against response above zero N) and the planting season",
        "carried by marker shape and line style. The plates are identical; only",
        "where the disclosures live differs.",
        "",
        f"- **`{ANNOTATED_FIGURE_FILENAME}`** — self-contained. Every disclosure",
        "  is set as a two-column caption on the sheet itself, so the figure can",
        "  travel alone.",
        f"- **`{FIGURE_ONLY_FILENAME}`** + **`{NOTES_FILENAME}`** — separated.",
        "  The caption block comes off the plates and becomes a markdown list.",
        "  The figure keeps only what it must not travel without: the headline,",
        "  the ANA-11 disclaimer, and a pointer to the notes file. Use this pair",
        "  when the plates go into a document that carries its own prose.",
        "",
        "Both figures and the notes are written from one `_disclosures` call, so",
        "the caption on the annotated sheet and the list in the notes file cannot",
        "disagree.",
        "",
        "## Population",
        "",
        f"- {len(series)} response series, "
        f"{sum(len(entry.levels) for entry in series)} applied-N levels, "
        f"{min(entry.year_bounds[0] for entry in series)}-"
        f"{max(entry.year_bounds[1] for entry in series)}, "
        f"decades {', '.join(decades)}.",
        "- Read from the promoted release's",
        f"  `{LEDGER_RELATIVE_PATH}`, verified against the package's",
        "  `CHECKSUMS.sha256` before use, and restricted to series the ledger",
        "  resolves with two or more distinct finite applied-N levels.",
        f"- Wider than the {governed_count} governed v3 series "
        f"({governed_observations} observations) drawn in",
        "  `../../core_trial_data_source_wide.jpeg`. Those 21 span 1991-1995 plus",
        "  one 2018 series, which is not enough calendar for a decade",
        "  decomposition. The sheet states the relationship on its own face.",
        f"- {len(anchored)} of the {len(series)} series observed zero applied N;",
        "  only those can enter the trend panels. The rest are drawn in the",
        "  insets alone, which is why two of the lettered stretches below sit",
        "  under a populated inset.",
        "",
        "## Method",
        "",
        "- Each trend marker is a plain mean over that planting year and",
        "  season's anchored series; the bar is the standard error across those",
        "  series and is drawn only where more than one qualifies.",
        "- The two means on each inset card are over every anchored series of",
        "  that decade, pooled across years and seasons. They are a different",
        "  grouping from the panels below and are not expected to match a marker",
        "  there. Where one happens to, it is because that decade's anchored",
        "  series all fall in a single year and season.",
        "- No replicate weighting: this source records no replicate structure, so",
        "  LTCCE's context-weighted average would be a no-op and its disclosure",
        "  would be false here.",
        "- `response above zero N` is a series' observed maximum yield less its",
        "  own zero-N yield. Nothing is fitted, interpolated or extrapolated.",
        "- Dry and wet are two separate broken series, never pooled: both are",
        "  recorded 1991-1995 but the record is wet-only from 2003, so a pooled",
        "  yearly mean would read a change of season composition as a trend.",
        "",
        "## Reading it honestly",
        "",
        f"- The year axis is a compilation order across "
        f"{len({entry.study_id for entry in series})} independent studies at "
        f"{len({entry.province for entry in series})} sites",
        f"  ({_era_text(eras)}), not one experiment's time trend. The line breaks",
        "  at every site change and at every year with nothing to plot; a step",
        "  across a break is a change of study, site, variety and design.",
        "- Most years contribute a single series, so most markers carry no error",
        f"  bar. The exceptions are {replicated_text}.",
        "- Stretches of the axis with no trend on them, lettered inside the",
        f"  panels where at least {_MIN_LABELLED_GAP_YEARS} years wide:",
        f"  {_gap_text(gaps)}.",
        "",
        "## Regenerating",
        "",
        "```bash",
        "conda run -n n_response python \\",
        "  modules/n_response_curve/reporting/"
        "generate_core_trial_planting_year_view.py \\",
        "  --config scriptCONFIG.toml",
        "```",
        "",
        "The destination directory is replaced atomically, but only the four",
        "files listed above are managed: anything else found here — an archive",
        "folder, a hand-kept copy, a note — is copied into the new snapshot and",
        "reported on stdout, never discarded.",
        "",
        "This directory in turn survives `generate_raw_dataset_overlays.py`'s own",
        f"snapshot replacement of `{SOURCE_NAME}/` because that generator carries",
        "the whole `clusters/` subtree across.",
        "",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "scriptCONFIG.toml",
        help="Path to scriptCONFIG.toml (read for [custom_overlays] and [sources]).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory (replaced atomically).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()

    governed = _governed_series_uids(config_path)
    ledger_path = _package_path(config_path) / LEDGER_RELATIVE_PATH
    series = _read_series(ledger_path)
    if not series:
        raise SystemExit(
            f"No {SOURCE_NAME} series in {ledger_path} carries two distinct "
            "applied-N levels"
        )
    governed_observations = sum(
        len(entry.levels) for entry in series if entry.series_uid in set(governed)
    )

    overlay = _build_overlay(series)
    limits = _shared_axis_limits(overlay)
    points = _trend_points(series)
    spans = _decade_spans(sorted({entry.decade for entry in series}))
    gaps = _trend_gaps(series, points, spans)
    eras = _site_eras(series)

    destination = args.output_dir.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.staging.{uuid.uuid4().hex}")
    staging.mkdir()
    carried: list[str] = []
    try:
        _write_decades_and_trend(
            series,
            overlay,
            limits,
            len(governed),
            governed_observations,
            staging / ANNOTATED_FIGURE_FILENAME,
        )
        _write_decades_and_trend(
            series,
            overlay,
            limits,
            len(governed),
            governed_observations,
            staging / FIGURE_ONLY_FILENAME,
            separate_notes=True,
        )
        _write_notes(
            series,
            points,
            gaps,
            eras,
            len(governed),
            governed_observations,
            staging / NOTES_FILENAME,
        )
        _write_readme(
            series,
            points,
            gaps,
            eras,
            len(governed),
            governed_observations,
            staging / "README.md",
        )
        # This directory is replaced as a whole snapshot, so anything in it that
        # this generator did not write has to be copied into staging or the
        # replacement destroys it. Operators archive and curate under these
        # trees; a rebuild of two figures is not a licence to delete their work.
        owned = {
            ANNOTATED_FIGURE_FILENAME,
            FIGURE_ONLY_FILENAME,
            NOTES_FILENAME,
            "README.md",
        }
        if destination.is_dir() and not destination.is_symlink():
            for entry in sorted(destination.iterdir()):
                if entry.name in owned or entry.is_symlink():
                    continue
                if entry.is_dir():
                    shutil.copytree(entry, staging / entry.name)
                else:
                    shutil.copy2(entry, staging / entry.name)
                carried.append(entry.name)

        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    if carried:
        noun = "entry" if len(carried) == 1 else "entries"
        print(
            f"Carried across {len(carried)} unmanaged {noun}: {', '.join(carried)}"
        )

    print(
        f"Wrote {ANNOTATED_FIGURE_FILENAME}, {FIGURE_ONLY_FILENAME}, "
        f"{NOTES_FILENAME}, README.md for {len(series)} {SOURCE_NAME} series "
        f"({len(points)} year-season trend points) under {destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
