#!/usr/bin/env python3
"""The descriptive-statistics N-yield scatter, redrawn with fitted curves.

The governed ``figures/agronomic/yield_versus_nitrogen.jpeg`` point cloud is
augmented with every declared Farmer's Practice (FP) observation. The PH NOPT
rows, paired PH FP arm, and literature rows explicitly classified as FP are
drawn as one group named ``ph_combined_nopt_rcm_fp``. The same loader, colours,
and opacity rule are retained, with fitted curves drawn over the augmented
cloud. ``--exclude-farmers-practice`` produces the corresponding no-FP panel:
the paired PH FP arm is not appended and the explicitly classified literature
rows are removed, while the PH NOPT arm is retained.

Why this is a separate script rather than a new builder in
``descriptive_statistics_figures.py``: that module's own contract is "Nothing
here fits a model, ranks a dataset, or draws a trend line", and the panel
hard-codes an annotation saying no fitted line is drawn. A fit belongs outside
that bundle, in its own output root, so the governed bundle's figures, tables,
manifest, and checksums are untouched by anything here.

Three project conventions are honoured deliberately:

* **No extrapolation.** Each curve is drawn only across its own dataset's
  observed N range, and a vertex is reported only when it falls inside that
  range.
* **No selection.** The fit form is an operator argument. Nothing here ranks
  forms by R^2 or an information criterion and picks a winner; the reported R^2
  is descriptive of the drawn curve, not a selection statistic.
* **Restricted datasets appear by ``source_name`` only**, exactly as in the
  bundle, and nothing row-level is written out.

``--scope`` chooses whether the fit is per dataset (default), one line across
all datasets stacked, or both. A pooled line is not the default because 13,653
of the 16,205 rows are ltcce, whose N ladder is confounded with season and era:
one line over the three plotted groups summarizes coverage, not a response.

``--weight`` chooses what the least squares is over. The recorded cloud is
severely unbalanced across N — 3,697 rows at zero, one at the top rate — so
``row`` (the default) is decided by where rows pile up, while ``level`` weights
every recorded rate equally and lets the sparse high-N end steer the curve. The
two answer different questions and neither is a correction of the other.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.analysis.descriptive_statistics.config import (  # noqa: E402
    load_recipe_config,
)
from n_response_curve.analysis.descriptive_statistics.sources import (  # noqa: E402
    build_all_observations,
    load_profiled_sources,
)
from n_response_curve.reporting import descriptive_statistics_figures as dsf  # noqa: E402

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "descriptive_statisticsCONFIG.toml"
# A sibling of the governed bundle, never inside it: the bundle verifies its own
# contents against CHECKSUMS, and a stray figure under its tree fails that check.
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "WF/03_Quality_Control/descriptive_statistics_fits"

# The governed PH NOPT rows carry no zero-N arm in this observation basis:
# ``n0_yield`` is a sibling column that ``build_observation_frame`` deliberately
# does not expand. The FP-inclusive plotted group can nevertheless reach zero
# through a literature row explicitly classified as Farmer's Practice. That
# mixed group is a coverage summary, not evidence of a PH response from zero.
_POOLED_LABEL = "all datasets pooled"
_PH_COMBINED_SOURCE_NAME = "ph_combined_nopt_rcm"
_PH_COMBINED_WITH_FP_NAME = "ph_combined_nopt_rcm_fp"

# Curve resolution. 200 points is smooth at this canvas width and keeps the
# encoded JPEG small.
_CURVE_POINTS = 200

# Caption geometry. Everything that qualifies the fit is written below the axes
# rather than over the cloud: the text grew past the two lines the governed panel
# carries, and an annotation box that size either hides the data it sits on or
# gets painted through by the curve it describes.
#
# Wrap budget in characters of 7.5-point DejaVu Sans across the recipe's 10-inch
# canvas, short of the full width so the block keeps a margin.
_CAPTION_CHARACTERS = 150
_CAPTION_FONT_SIZE = 7.5
# One line of that text with normal leading, in inches.
_CAPTION_LINE_INCHES = 0.145
_CAPTION_TOP_PAD_INCHES = 0.22
_CAPTION_BOTTOM_PAD_INCHES = 0.14
# Caption band the configured canvas already budgets for. Only the excess grows
# the figure, so a short caption renders at exactly the recipe's own height and a
# long one buys its own room instead of squeezing the axes.
_CAPTION_BUDGETED_INCHES = 0.72
_CAPTION_LEFT = 0.055


def _caption(figure: plt.Figure, parts: list[str]) -> None:
    """Write the qualifying text under the axes, growing the canvas to fit it."""

    lines = [
        line
        for part in parts
        for line in textwrap.wrap(" ".join(part.split()), width=_CAPTION_CHARACTERS)
    ]
    needed = (
        _CAPTION_TOP_PAD_INCHES
        + _CAPTION_BOTTOM_PAD_INCHES
        + _CAPTION_LINE_INCHES * len(lines)
    )
    if needed > _CAPTION_BUDGETED_INCHES:
        figure.set_figheight(
            figure.get_figheight() + needed - _CAPTION_BUDGETED_INCHES
        )
    height = figure.get_figheight()
    figure.tight_layout(rect=(0, needed / height, 1, 1))
    # Hang the caption off the axes' own tight bounding box — which includes the
    # x-axis label — rather than off a fraction of the canvas. tight_layout does
    # not place the axes flush with the rect it is given, so a fraction computed
    # up front leaves a band of white between the label and the text.
    axes = figure.axes[0]
    box = axes.get_tightbbox(figure.canvas.get_renderer()).transformed(
        figure.transFigure.inverted()
    )
    figure.text(
        _CAPTION_LEFT,
        box.y0 - _CAPTION_TOP_PAD_INCHES / height,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=_CAPTION_FONT_SIZE,
    )


@dataclass(frozen=True)
class _Fit:
    """One drawable curve plus the numbers that qualify it."""

    source_name: str
    form: str
    n_observations: int
    n_levels: int
    n_min: float
    n_max: float
    x: np.ndarray
    y: np.ndarray
    r_squared: float
    # Parameter summary as printed and written; free-form per form.
    parameters: str
    # Turning point of the drawn curve, and whether it sits inside the observed
    # N range. Outside it, no optimum is reported at all.
    vertex_n: float | None
    vertex_yield: float | None
    vertex_inside_range: bool
    # Shape facts the panel must state rather than let a reader infer. Each is
    # tested separately, because the caveat box names the datasets that carry
    # one and naming the wrong dataset is worse than saying nothing.
    plateau_unbounded: bool
    interior_minimum_n: float | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class _FarmersPracticeMerge:
    """The augmented observation cloud and an explicit account of its FP rows."""

    observations: pd.DataFrame
    ph_nopt_rows: int
    paired_ph_fp_rows: int
    relabeled_core_fp_rows: int

    @property
    def merged_rows(self) -> int:
        return (
            self.ph_nopt_rows
            + self.paired_ph_fp_rows
            + self.relabeled_core_fp_rows
        )


@dataclass(frozen=True)
class _FarmersPracticeExclusion:
    """The no-FP cloud and an explicit account of the omitted FP rows."""

    observations: pd.DataFrame
    ph_nopt_rows: int
    omitted_paired_ph_fp_rows: int
    removed_core_fp_rows: int


def _merge_farmers_practice(
    observations: pd.DataFrame,
    loaded,
    config,
) -> _FarmersPracticeMerge:
    """Merge every declared FP observation into the plotted PH dataset group.

    ``_farmers_practice_series`` is the descriptive-statistics recipe's single
    composition rule: it appends the paired PH FP rate/yield columns and moves
    core rows explicitly classified as Farmer's Practice into that derived
    series. This plot then combines the derived series with the PH NOPT rows,
    because its legend is by dataset group rather than by treatment arm.

    Fail loudly if the promised arm disappears from the configuration. The two
    FP bindings are optional at schema level, so silently retaining the old
    three-series cloud would otherwise recreate the omission this plot is meant
    to fix.
    """

    composition = dsf._farmers_practice_series(loaded, observations, config)
    if (
        _PH_COMBINED_SOURCE_NAME not in composition.arm_source_names
        or composition.arm_rows <= 0
    ):
        raise ValueError(
            "No finite Farmer's Practice N-rate/yield pairs were loaded for "
            f"{_PH_COMBINED_SOURCE_NAME}"
        )

    merged = composition.observations.copy()
    ph_nopt_rows = int(
        (merged["source_name"] == _PH_COMBINED_SOURCE_NAME).sum()
    )
    fp_series_name = dsf._FARMERS_PRACTICE_SERIES_NAME
    merge_mask = merged["source_name"].isin(
        (_PH_COMBINED_SOURCE_NAME, fp_series_name)
    )
    merged.loc[merge_mask, "source_name"] = _PH_COMBINED_WITH_FP_NAME
    return _FarmersPracticeMerge(
        observations=merged,
        ph_nopt_rows=ph_nopt_rows,
        paired_ph_fp_rows=composition.arm_rows,
        relabeled_core_fp_rows=composition.relabeled_rows,
    )


def _exclude_farmers_practice(
    observations: pd.DataFrame,
    loaded,
    config,
) -> _FarmersPracticeExclusion:
    """Remove explicit literature FP rows and omit every paired PH FP row."""

    composition = dsf._farmers_practice_series(loaded, observations, config)
    fp_index = dsf._core_trial_farmers_practice_index(loaded, observations)
    if fp_index.empty and composition.arm_rows <= 0:
        raise ValueError("No Farmer's Practice observations were found")

    retained = observations.drop(index=fp_index).reset_index(drop=True)
    ph_nopt_rows = int(
        (retained["source_name"] == _PH_COMBINED_SOURCE_NAME).sum()
    )
    return _FarmersPracticeExclusion(
        observations=retained,
        ph_nopt_rows=ph_nopt_rows,
        omitted_paired_ph_fp_rows=composition.arm_rows,
        removed_core_fp_rows=int(len(fp_index)),
    )


def _fit_source_name(
    source_name: str, *, include_farmers_practice: bool
) -> str:
    """Map the PH source to the FP-inclusive name only for that variant."""

    if include_farmers_practice and source_name == _PH_COMBINED_SOURCE_NAME:
        return _PH_COMBINED_WITH_FP_NAME
    return source_name


def _ordered_sources(
    loaded, present, *, include_farmers_practice: bool
) -> tuple[str, ...]:
    """Registered source order, translated when the plotted group includes FP."""

    available = set(present)
    return tuple(
        plotted_name
        for source in loaded.sources
        if (
            plotted_name := _fit_source_name(
                source.source_name,
                include_farmers_practice=include_farmers_practice,
            )
        )
        in available
    )


def _source_colours(loaded, *, include_farmers_practice: bool) -> dict[str, str]:
    """Keep the registered PH colour when its plotted name gains the FP suffix."""

    registered = dsf._source_colours(loaded)
    return {
        _fit_source_name(
            source_name,
            include_farmers_practice=include_farmers_practice,
        ): colour
        for source_name, colour in registered.items()
    }


def _legend_label(source_name: str, *, include_farmers_practice: bool) -> str:
    """Reader-facing label, including the explicit suffix on the merged group."""

    if include_farmers_practice and source_name == _PH_COMBINED_WITH_FP_NAME:
        return source_name
    return dsf._legend_label(source_name)


def _r_squared(
    observed: np.ndarray, predicted: np.ndarray, weights: np.ndarray
) -> float:
    mean = float(np.sum(weights * observed) / np.sum(weights))
    residual = float(np.sum(weights * (observed - predicted) ** 2))
    total = float(np.sum(weights * (observed - mean) ** 2))
    if total <= 0.0:
        return float("nan")
    return 1.0 - residual / total


def _level_weights(x: np.ndarray, *, tolerance_kg_ha: float) -> np.ndarray:
    """One unit of weight per recorded N level, split among its rows.

    The recorded cloud is not balanced across N: 3,697 rows sit at zero and one
    sit at the top rate, so an unweighted fit is decided almost entirely by where
    the rows happen to pile up. Weighting each row by the reciprocal of its
    level's row count makes every recorded rate count once — the fit then follows
    the level means, and a sparse high-N level is no longer outvoted by a dense
    low-N one. Levels are snapped on the recipe's own tolerance grid, so ``100``
    and ``100.0`` are one level here exactly as they are everywhere else.
    """

    if tolerance_kg_ha > 0:
        levels = np.round(x / tolerance_kg_ha) * tolerance_kg_ha
    else:
        levels = x.astype(float)
    unique, inverse, counts = np.unique(levels, return_inverse=True, return_counts=True)
    del unique
    return 1.0 / counts[inverse].astype(float)


def _polynomial_fit(
    x: np.ndarray, y: np.ndarray, degree: int, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Least-squares polynomial, evaluated across the observed range only."""

    # numpy's ``w`` multiplies the residual, so it takes the square root of a
    # weight expressed in the usual sum(w * residual^2) sense.
    coefficients = np.polyfit(x, y, degree, w=np.sqrt(weights))
    grid = np.linspace(float(x.min()), float(x.max()), _CURVE_POINTS)
    return coefficients, grid, np.polyval(coefficients, grid), _r_squared(
        y, np.polyval(coefficients, x), weights
    )


def _quadratic_plateau(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float, np.ndarray, np.ndarray, float, bool]:
    """Quadratic rising to a smooth join, flat thereafter.

    Parameterized so the join is smooth by construction: with
    ``y = a + b*u + c*u^2`` on ``u = min(N, join)`` and the derivative vanishing
    at the join, ``b = -2*c*join``, which leaves ``a`` and ``c`` linear. The join
    is profiled over a grid of candidate positions inside the observed range and
    the sum of squares picks it. That is estimation of one parameter within a
    single declared form, not selection between forms.

    The last returned value flags a join that ran to the top of the candidate
    grid. That is the signature of data with no plateau in them at all: the
    residual sum of squares keeps falling as the join is pushed right, so the
    fit is a quadratic arm across the whole range and the reported join is the
    grid boundary, not a located plateau. The caller must not report it as one.
    """

    low, high = float(x.min()), float(x.max())
    # Candidate joins are kept away from the extreme ends, where the plateau or
    # the quadratic arm would be supported by a handful of points.
    span = high - low
    candidates = np.linspace(low + 0.15 * span, high - 0.02 * span, 120)
    root = np.sqrt(weights)
    best: tuple[float, float, float, float] | None = None
    for join in candidates:
        u = np.minimum(x, join)
        # Columns of [1, u^2 - 2*join*u] against y, both scaled by the square
        # root of the weight so the profiled join minimizes the weighted sum of
        # squares rather than the raw one.
        design = np.column_stack((np.ones_like(u), u * u - 2.0 * join * u))
        solution, *_ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
        residual = float(np.sum(weights * (y - design @ solution) ** 2))
        if best is None or residual < best[0]:
            best = (residual, float(join), float(solution[0]), float(solution[1]))
    assert best is not None
    _, join, intercept, curvature = best
    linear = -2.0 * curvature * join

    def evaluate(values: np.ndarray) -> np.ndarray:
        u = np.minimum(values, join)
        return intercept + linear * u + curvature * u * u

    grid = np.linspace(low, high, _CURVE_POINTS)
    step = float(candidates[1] - candidates[0])
    at_boundary = join >= float(candidates[-1]) - 0.5 * step
    return (
        join,
        intercept,
        curvature,
        grid,
        evaluate(grid),
        _r_squared(y, evaluate(x), weights),
        at_boundary,
    )


def _loess(
    x: np.ndarray,
    y: np.ndarray,
    fraction: float,
    weights: np.ndarray,
    *,
    level_weighted: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    if level_weighted:
        # ``lowess`` takes no weights, so the level-weighted smoother is run on
        # one point per recorded level — the mean yield there — which is what
        # weighting every level equally amounts to. The neighbourhood fraction
        # then spans levels rather than rows, which is also what makes the top
        # of the range visible to it at all.
        levels, inverse = np.unique(x, return_inverse=True)
        means = np.bincount(inverse, weights=y) / np.bincount(inverse)
        fit_x, fit_y = levels, means
    else:
        fit_x, fit_y = x, y
    span = float(fit_x.max() - fit_x.min())
    smoothed = lowess(
        fit_y,
        fit_x,
        frac=fraction,
        it=1,
        # Points closer than delta share a fitted value; at 13,653 rows on a
        # 30-level ladder this is the difference between seconds and minutes,
        # and 0.3% of the span is far finer than the recorded N grid.
        delta=0.003 * span,
        return_sorted=True,
    )
    grid, fitted = smoothed[:, 0], smoothed[:, 1]
    predicted = np.interp(x, grid, fitted)
    return grid, fitted, _r_squared(y, predicted, weights)


def _fit_one(
    source_name: str,
    frame: pd.DataFrame,
    form: str,
    loess_fraction: float,
    *,
    weighting: str,
    level_tolerance_kg_ha: float,
) -> _Fit | None:
    x = frame["n_rate_kg_ha"].to_numpy(dtype=float)
    y = frame["yield_t_ha"].to_numpy(dtype=float)
    levels = int(np.unique(np.round(x, 6)).size)
    low, high = float(x.min()), float(x.max())
    level_weighted = weighting == "level"
    if level_weighted:
        weights = _level_weights(x, tolerance_kg_ha=level_tolerance_kg_ha)
    else:
        weights = np.ones_like(x)
    notes: list[str] = []
    if low > 0.0:
        notes.append(
            f"no recorded zero-N row; fitted window starts at {low:g} kg N/ha"
        )

    required = {"quadratic": 3, "quadratic-plateau": 4, "loess": 4}[form]
    if levels < required:
        print(
            f"{source_name}: skipped — {levels} distinct N level(s), "
            f"{form} needs at least {required}",
            file=sys.stderr,
        )
        return None

    vertex_n: float | None = None
    vertex_yield: float | None = None
    plateau_unbounded = False
    interior_minimum_n: float | None = None
    if form == "quadratic":
        coefficients, grid, curve, r2 = _polynomial_fit(x, y, 2, weights)
        quadratic, linear, intercept = (float(value) for value in coefficients)
        parameters = (
            f"y = {intercept:.4g} + {linear:.4g}·N + {quadratic:.4g}·N²"
        )
        turning = -linear / (2.0 * quadratic) if quadratic != 0.0 else None
        if quadratic < 0.0 and turning is not None:
            vertex_n = turning
            vertex_yield = float(np.polyval(coefficients, vertex_n))
        elif quadratic > 0.0 and turning is not None and low <= turning <= high:
            # An upward-opening parabola whose vertex is inside the window dips
            # before it rises. That is a minimum, never an optimum, and the
            # panel says so rather than leaving the dip to be read as one.
            interior_minimum_n = float(turning)
    elif form == "quadratic-plateau":
        join, intercept, curvature, grid, curve, r2, at_boundary = _quadratic_plateau(
            x, y, weights
        )
        linear = -2.0 * curvature * join
        parameters = (
            f"y = {intercept:.4g} + {linear:.4g}·N + {curvature:.4g}·N² "
            f"up to N = {join:.4g}, flat above"
        )
        if at_boundary:
            parameters += " [join at the search boundary — no plateau in range]"
            plateau_unbounded = True
            notes.append(
                "the plateau join ran to the top of the observed range, so "
                "these data locate no plateau"
            )
        else:
            vertex_n = join
            vertex_yield = float(np.interp(join, grid, curve))
    else:
        grid, curve, r2 = _loess(
            x, y, loess_fraction, weights, level_weighted=level_weighted
        )
        parameters = f"local linear regression, frac = {loess_fraction:g}"
        peak = int(np.argmax(curve))
        # Only an interior maximum is a turning point; a curve still climbing at
        # the last observed rate has none. The test is on N, not on the index:
        # ``lowess`` returns one row per observation, so the last several rows
        # share the top recorded rate and an index-based interiority test would
        # call the endpoint an optimum.
        margin = 0.02 * (high - low)
        if low + margin < float(grid[peak]) < high - margin:
            vertex_n = float(grid[peak])
            vertex_yield = float(curve[peak])

    inside = bool(vertex_n is not None and low <= vertex_n <= high)
    if vertex_n is not None and not inside:
        vertex_n = vertex_yield = None
    if interior_minimum_n is not None:
        notes.append(
            f"the fitted curve has an interior minimum at {interior_minimum_n:.0f} "
            "kg N/ha, not an optimum"
        )

    return _Fit(
        source_name=source_name,
        form=form,
        n_observations=int(len(frame)),
        n_levels=levels,
        n_min=low,
        n_max=high,
        x=np.asarray(grid, dtype=float),
        y=np.asarray(curve, dtype=float),
        r_squared=float(r2),
        parameters=parameters,
        vertex_n=vertex_n,
        vertex_yield=vertex_yield,
        vertex_inside_range=inside,
        plateau_unbounded=plateau_unbounded,
        interior_minimum_n=interior_minimum_n,
        notes=tuple(notes),
    )


_FORM_LABEL = {
    "quadratic": "quadratic fit",
    "quadratic-plateau": "quadratic-plateau fit",
    "loess": "LOESS smoother",
}


def _build_figure(
    observations: pd.DataFrame,
    loaded,
    config,
    *,
    form: str,
    loess_fraction: float,
    scope: str,
    weighting: str,
    loess_overlay: bool,
    mark_vertex: bool,
    farmers_practice_merge: _FarmersPracticeMerge | None,
    farmers_practice_exclusion: _FarmersPracticeExclusion | None,
    show_description: bool,
) -> tuple[plt.Figure, list[_Fit]]:
    include_farmers_practice = farmers_practice_merge is not None
    if include_farmers_practice == (farmers_practice_exclusion is not None):
        raise ValueError(
            "Exactly one Farmer's Practice inclusion/exclusion summary is required"
        )
    names = _ordered_sources(
        loaded,
        observations["source_name"],
        include_farmers_practice=include_farmers_practice,
    )
    colours = _source_colours(
        loaded, include_farmers_practice=include_farmers_practice
    )
    legend_label = lambda name: _legend_label(  # noqa: E731
        name, include_farmers_practice=include_farmers_practice
    )
    counts = observations["source_name"].value_counts()
    figure, axis = dsf._figure(config)

    # Largest cloud first so the smaller datasets are not buried beneath it —
    # the governed panel's rule, kept so the two read as the same scatter.
    drawing_order = sorted(names, key=lambda name: int(counts.get(name, 0)), reverse=True)
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

    # The level tolerance comes from the recipe, so a level means the same thing
    # here as it does in the bundle's ladder-geometry counts.
    fit_options = {
        "weighting": weighting,
        "level_tolerance_kg_ha": config.n_level_tolerance_kg_ha,
    }
    fits: list[_Fit] = []
    if scope in {"per-dataset", "both"}:
        for name in names:
            subset = observations.loc[observations["source_name"] == name]
            if subset.empty:
                continue
            fit = _fit_one(name, subset, form, loess_fraction, **fit_options)
            if fit is not None:
                fits.append(fit)
    if scope in {"pooled", "both"}:
        pooled_fit = _fit_one(
            _POOLED_LABEL, observations, form, loess_fraction, **fit_options
        )
        if pooled_fit is not None:
            fits.append(pooled_fit)
    # The pooled curve is dashed only where per-dataset curves share the panel
    # and it has to be told apart from them. Drawn alone it is the subject of the
    # figure, so it is solid and darker than the neutral used beside colours.
    pooled_alone = scope == "pooled"
    pooled_colour = "#333333" if pooled_alone else dsf._NEUTRAL
    pooled_style = "-" if pooled_alone else (0, (6, 3))

    # A white halo under each curve: at this opacity the cloud is dense enough
    # that a bare coloured line disappears into its own points.
    halo = [
        path_effects.Stroke(linewidth=4.4, foreground="white", alpha=0.85),
        path_effects.Normal(),
    ]
    handles: list[Line2D] = []
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
                label=f"{legend_label(name)} — {int(counts.get(name, 0)):,} observations",
            )
        )
    for fit in fits:
        is_pooled = fit.source_name == _POOLED_LABEL
        colour = pooled_colour if is_pooled else colours[fit.source_name]
        style = pooled_style if is_pooled else "-"
        axis.plot(
            fit.x,
            fit.y,
            color=colour,
            linewidth=2.6 if pooled_alone else 2.4,
            linestyle=style,
            path_effects=halo,
            zorder=4,
        )
        # No R² in the legend. Three R² values side by side read as a ranking of
        # the datasets, and comparing a smoother's R² against a quadratic's is
        # selection by fit statistic, which this project does not do. The value
        # is descriptive of one drawn curve and lives in the companion CSV.
        label = f"{fit.source_name} — {_FORM_LABEL[fit.form]}"
        if is_pooled:
            label = (
                f"one {_FORM_LABEL[fit.form]} across all "
                f"{fit.n_observations:,} observations (coverage summary)"
            )
        handles.append(
            Line2D(
                [],
                [],
                color=colour,
                linewidth=2.4,
                linestyle=style,
                label=label,
            )
        )
        if mark_vertex and fit.vertex_n is not None:
            axis.plot(
                [fit.vertex_n],
                [fit.vertex_yield],
                marker="v",
                markersize=9,
                color=colour,
                markeredgecolor="white",
                markeredgewidth=1.0,
                zorder=5,
            )
            # Put the numerical result beside the marker so the reader does not
            # have to estimate it from the axes or open the companion CSV. Keep
            # the box on the inward side when a turning point sits in the right
            # half of its fitted range, which prevents an edge vertex from
            # pushing the callout beyond the axes.
            label_to_left = fit.vertex_n > (fit.n_min + fit.n_max) / 2.0
            axis.annotate(
                "Turning point\n"
                f"N = {fit.vertex_n:.1f} kg N ha⁻¹\n"
                f"Yield = {fit.vertex_yield:.2f} t ha⁻¹",
                xy=(fit.vertex_n, fit.vertex_yield),
                xytext=(-12 if label_to_left else 12, 16),
                textcoords="offset points",
                ha="right" if label_to_left else "left",
                va="bottom",
                fontsize=8,
                color=colour,
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "facecolor": "white",
                    "edgecolor": colour,
                    "alpha": 0.92,
                    "linewidth": 0.9,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": colour,
                    "linewidth": 0.9,
                },
                zorder=6,
            )

    if loess_overlay and form != "loess":
        # The shape check follows the scope of the fit it is checking: one pooled
        # smoother beside a pooled curve, per-dataset smoothers beside
        # per-dataset curves.
        if pooled_alone:
            smoothed = [(_POOLED_LABEL, observations, pooled_colour)]
        else:
            smoothed = [
                (name, observations.loc[observations["source_name"] == name], colours[name])
                for name in names
            ]
        for label, subset, colour in smoothed:
            smoother = _fit_one(label, subset, "loess", loess_fraction, **fit_options)
            if smoother is None:
                continue
            axis.plot(
                smoother.x,
                smoother.y,
                color=colour,
                linewidth=1.3,
                linestyle=(0, (2, 2)),
                path_effects=[
                    path_effects.Stroke(linewidth=3.0, foreground="white", alpha=0.7),
                    path_effects.Normal(),
                ],
                zorder=3,
            )
        handles.append(
            Line2D(
                [],
                [],
                color=pooled_colour if pooled_alone else dsf._NEUTRAL,
                linewidth=1.3,
                linestyle=(0, (2, 2)),
                label=(
                    f"LOESS smoother{'' if pooled_alone else ' per dataset'} "
                    f"(frac = {loess_fraction:g})"
                ),
            )
        )

    level_weighted = weighting == "level"
    squares = (
        "weighted least squares, every recorded N level counting once "
        "(each row weighted by the reciprocal of its level's row count)"
        if level_weighted
        else "ordinary least squares on the recorded rows"
    )
    if pooled_alone:
        # A single line over three stacked groups is dominated by whichever one
        # is largest, so the panel prints the composition rather than leaving
        # the reader to infer it from the legend counts.
        composition = ", ".join(
            f"{legend_label(name)} "
            f"{100.0 * int(counts.get(name, 0)) / len(observations):.0f}%"
            for name in sorted(
                names, key=lambda name: int(counts.get(name, 0)), reverse=True
            )
        )
        caveat_parts = [
            f"One {_FORM_LABEL[form]} across all three plotted groups stacked "
            f"({len(observations):,} rows: {composition}), {squares}, drawn only "
            "across the observed N range."
        ]
    else:
        caveat_parts = [
            f"{_FORM_LABEL[form].capitalize()} per dataset, {squares}, drawn only "
            "across each dataset's own observed N range."
        ]
    if level_weighted:
        # Level weighting is what lets the thin top of the range steer the curve,
        # so the panel has to say how thin it is: a rate recorded four times now
        # counts as much as one recorded 3,697 times.
        top_quarter = float(observations["n_rate_kg_ha"].max()) - 0.25 * (
            float(observations["n_rate_kg_ha"].max())
            - float(observations["n_rate_kg_ha"].min())
        )
        tail = observations.loc[observations["n_rate_kg_ha"] >= top_quarter]
        # Name the datasets that actually reach the tail. Under per-dataset scope
        # only one of the three usually gets there, and an unattributed count
        # reads as if the warning applied to every curve on the panel.
        reaching = ", ".join(
            legend_label(name)
            for name in _ordered_sources(
                loaded,
                tail["source_name"].unique(),
                include_farmers_practice=include_farmers_practice,
            )
        )
        caveat_parts.append(
            f"Above {top_quarter:.0f} kg N/ha the cloud holds {len(tail):,} "
            f"observations from {tail['series_key'].nunique()} series across "
            f"{tail['n_rate_kg_ha'].nunique()} levels, all {reaching} — under "
            "level weighting those few levels set the right-hand tail."
        )
    # Each caveat names only the datasets it is true of. Reusing one sentence for
    # every dataset carrying any note would tell the reader, for instance, that
    # ltcce records no zero-N row when it records 3,421 of them.
    unanchored = [legend_label(fit.source_name) for fit in fits if fit.n_min > 0.0]
    if unanchored:
        caveat_parts.append(
            f"{', '.join(unanchored)}: no recorded zero-N row in this basis, so "
            "the curve covers the fertilized window only."
        )
    unbounded = [fit.source_name for fit in fits if fit.plateau_unbounded]
    if unbounded:
        caveat_parts.append(
            f"{', '.join(unbounded)}: the plateau join ran to the top of the "
            "observed range — these data locate no plateau."
        )
    dipping = [
        f"{fit.source_name} ({fit.interior_minimum_n:.0f} kg N/ha)"
        for fit in fits
        if fit.interior_minimum_n is not None
    ]
    if dipping:
        caveat_parts.append(
            f"{', '.join(dipping)}: the curve turns at an interior minimum, not "
            "an optimum."
        )
    if mark_vertex:
        caveat_parts.append(
            "Markers and adjacent labels show a curve's turning point; one "
            "outside the observed range is not reported at all."
        )
    # The standing disclaimer closes the caption: it is identical on every
    # variant, so it reads last, after the facts specific to this panel.
    if farmers_practice_exclusion is not None:
        caveat_parts.append(
            "Farmer's Practice (FP) is excluded: "
            f"{farmers_practice_exclusion.removed_core_fp_rows:,} "
            "literature-extracted FP observations were removed, and "
            f"{farmers_practice_exclusion.omitted_paired_ph_fp_rows:,} paired "
            "PH FP observations were not appended. The ph_combined_nopt_rcm "
            f"group contains only its {farmers_practice_exclusion.ph_nopt_rows:,} "
            "PH NOPT rows. Each retained point carries both a finite N rate and "
            "a finite yield. The fitted curve is added here and is not part of "
            "the governed descriptive-statistics bundle. It fits recorded "
            "yield on recorded N alone, with no term for dataset, series, site, "
            "season, year, or variety, and the datasets differ in design, era, "
            "and site — so the curve summarizes coverage of the retained "
            "recorded cloud. It is not a causal N response and carries no "
            "fertilizer recommendation."
        )
    else:
        assert farmers_practice_merge is not None
        caveat_parts.append(
            f"The {_PH_COMBINED_WITH_FP_NAME} group contains "
            f"{farmers_practice_merge.ph_nopt_rows:,} PH NOPT rows, "
            f"{farmers_practice_merge.paired_ph_fp_rows:,} paired PH FP rows, and "
            f"{farmers_practice_merge.relabeled_core_fp_rows:,} literature rows "
            "explicitly classified as Farmer's Practice. Each point carries both a "
            "finite N rate and a finite yield. The fitted curves are added here and "
            "are not part of the governed descriptive-statistics bundle. They fit "
            "recorded yield on recorded N alone, with no term for dataset, series, "
            "site, season, year, or variety, and the datasets differ in design, era, "
            "and site — so a curve summarizes coverage of the recorded cloud. It is "
            "not a causal N response and carries no fertilizer recommendation."
        )

    # Headroom for the legend, which now carries a fit entry per dataset on top
    # of the scatter entries and would otherwise sit on the highest yields.
    low, high = axis.get_ylim()
    axis.set_ylim(low, high + (0.04 + 0.034 * len(handles)) * (high - low))
    axis.set(
        # The fit clause goes on its own line. Spelled out — form, scope, and
        # weighting — it outgrows a 10-inch canvas on one line and matplotlib
        # centres the overflow off both edges rather than shrinking it.
        title=(
            "Recorded grain yield against recorded inorganic N rate\n"
            + (
                f"with one {_FORM_LABEL[form]} across all datasets"
                if pooled_alone
                else f"with a {_FORM_LABEL[form]} per dataset"
            )
            + (", every N level weighted equally" if weighting == "level" else "")
            + ("\nFarmer's Practice excluded" if not include_farmers_practice else "")
        ),
        xlabel="Inorganic N rate (kg N ha⁻¹)",
        ylabel="Grain yield (t ha⁻¹)",
    )
    axis.legend(handles=handles, fontsize=8, loc="upper right")
    axis.grid(alpha=0.2)
    if show_description:
        _caption(figure, caveat_parts)
    else:
        figure.tight_layout()
    return figure, fits


def _fit_table(fits: list[_Fit], weighting: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_name": fit.source_name,
                "fit_form": fit.form,
                "weighting": weighting,
                "observations": fit.n_observations,
                "distinct_n_levels": fit.n_levels,
                "n_min_kg_ha": fit.n_min,
                "n_max_kg_ha": fit.n_max,
                "r_squared": fit.r_squared,
                "parameters": fit.parameters,
                "turning_point_n_kg_ha": (
                    fit.vertex_n if fit.vertex_n is not None else ""
                ),
                "turning_point_yield_t_ha": (
                    fit.vertex_yield if fit.vertex_yield is not None else ""
                ),
                "turning_point_inside_observed_range": fit.vertex_inside_range,
                "note": "; ".join(fit.notes),
            }
            for fit in fits
        ]
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--fit",
        choices=("quadratic", "quadratic-plateau", "loess"),
        default="quadratic",
        help="Fitted form, per dataset (default: quadratic). Never auto-selected.",
    )
    parser.add_argument(
        "--loess-fraction",
        type=float,
        default=0.6,
        help="LOESS neighbourhood fraction (default: 0.6)",
    )
    parser.add_argument(
        "--loess-overlay",
        action="store_true",
        help="Add a dashed per-dataset LOESS beside the parametric fit as a "
        "shape check",
    )
    parser.add_argument(
        "--scope",
        choices=("per-dataset", "pooled", "both"),
        default="per-dataset",
        help="Whether to fit each dataset separately (default), fit one line "
        "across all datasets stacked, or draw both. Most rows are ltcce, so any "
        "pooled line is labeled a coverage summary.",
    )
    parser.add_argument(
        "--weight",
        choices=("row", "level"),
        default="row",
        help="'row' (default) weights every recorded row equally, so the fit "
        "follows wherever the rows pile up. 'level' weights every recorded N "
        "level equally, so a rate carrying four observations counts as much as "
        "one carrying 3,697 and the sparse high-N end steers the curve.",
    )
    parser.add_argument(
        "--mark-vertex",
        action="store_true",
        help="Mark and label each curve's turning point when it falls inside "
        "the observed N range",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Skip the companion fit-parameter CSV",
    )
    parser.add_argument(
        "--no-description",
        action="store_true",
        help="Render the figure without the qualifying caption below the axes",
    )
    parser.add_argument(
        "--exclude-farmers-practice",
        action="store_true",
        help="Omit the paired PH Farmer's Practice arm and remove literature "
        "rows explicitly classified as Farmer's Practice",
    )
    return parser.parse_args(argv)


def _output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve the figure and table paths for one declared fit variant."""

    # Every switch that changes what is drawn also changes the default filename,
    # so running two variants in a row leaves two artifacts rather than silently
    # overwriting the first.
    fit_family = args.fit.replace("-", "_")
    slug = fit_family
    if args.loess_overlay and args.fit != "loess":
        slug += "_loess"
    if args.scope == "pooled":
        slug += "_single_line"
    elif args.scope == "both":
        slug += "_pooled"
    if args.weight == "level":
        slug += "_level_weighted"
    if args.mark_vertex:
        slug += "_vertex"
    if args.exclude_farmers_practice:
        slug += "_excluding_farmers_practice"
    if args.no_description:
        slug += "_no_description"

    if args.output is not None:
        figure_path = args.output
        return figure_path, figure_path.with_name(figure_path.stem + "_fits.csv")

    filename = f"yield_versus_nitrogen_{slug}"
    family_root = DEFAULT_OUTPUT_ROOT / fit_family
    return (
        family_root / f"{filename}.jpeg",
        family_root / f"{filename}_fits.csv",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not 0.0 < args.loess_fraction <= 1.0:
        raise SystemExit("--loess-fraction must be in (0, 1]")

    config = load_recipe_config(args.config, project_root=PROJECT_ROOT)
    loaded = load_profiled_sources(config)
    governed_observations = build_all_observations(
        loaded, zero_n_tolerance_kg_ha=config.zero_n_tolerance_kg_ha
    )
    if governed_observations.empty:
        raise SystemExit("No harmonized observations were produced")
    farmers_practice_merge: _FarmersPracticeMerge | None = None
    farmers_practice_exclusion: _FarmersPracticeExclusion | None = None
    if args.exclude_farmers_practice:
        farmers_practice_exclusion = _exclude_farmers_practice(
            governed_observations, loaded, config
        )
        observations = farmers_practice_exclusion.observations
    else:
        farmers_practice_merge = _merge_farmers_practice(
            governed_observations, loaded, config
        )
        observations = farmers_practice_merge.observations

    figure, fits = _build_figure(
        observations,
        loaded,
        config,
        form=args.fit,
        loess_fraction=args.loess_fraction,
        scope=args.scope,
        weighting=args.weight,
        loess_overlay=args.loess_overlay,
        mark_vertex=args.mark_vertex,
        farmers_practice_merge=farmers_practice_merge,
        farmers_practice_exclusion=farmers_practice_exclusion,
        show_description=not args.no_description,
    )
    if not fits:
        plt.close(figure)
        raise SystemExit("No dataset carried enough N levels for the requested fit")

    destination, table_path = _output_paths(args)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(
            destination,
            format=destination.suffix.lstrip(".").lower() or "jpeg",
            dpi=config.figure_dpi,
        )
    finally:
        plt.close(figure)

    table = _fit_table(fits, args.weight)
    for row in table.itertuples(index=False):
        turning = (
            f"turning point N = {row.turning_point_n_kg_ha:.1f} kg/ha "
            f"({row.turning_point_yield_t_ha:.2f} t/ha)"
            if row.turning_point_n_kg_ha != ""
            else "no turning point inside the observed range"
        )
        print(
            f"{row.source_name}: n={row.observations:,} over "
            f"{row.n_min_kg_ha:g}-{row.n_max_kg_ha:g} kg N/ha "
            f"({row.distinct_n_levels} levels); {row.parameters}; "
            f"R²={row.r_squared:.3f}; {turning}"
            + (f"; {row.note}" if row.note else "")
        )
    if not args.no_table:
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(table_path, index=False)
        print(f"Wrote {table_path}")
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
