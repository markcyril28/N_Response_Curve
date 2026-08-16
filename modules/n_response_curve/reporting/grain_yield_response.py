from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from ..analysis.grain_yield_response.config import GrainYieldResponseConfig
from ..analysis.grain_yield_response.descriptive import DescriptiveResults
from ..analysis.grain_yield_response.factor_support import FactorSupportResults
from ..analysis.grain_yield_response.heterogeneity import HeterogeneityResults
from ..analysis.grain_yield_response.population import GovernedPopulation, sha256_file
from ..analysis.grain_yield_response.series_covariates import SeriesCovariateResults


MANIFEST_NAME = "run_manifest.json"
CHECKSUMS_NAME = "CHECKSUMS.sha256"
SCHEMA_VERSION = "grain-yield-response-diagnostics-v1"
BUNDLE_STATUS = "diagnostic_internal_not_release"


class DiagnosticBundleError(ValueError):
    """Raised when a diagnostic bundle is incomplete or unsafe."""


# Every bundle artifact lives in a themed subdirectory, and registration here
# is mandatory: writing a table or figure whose name has no group raises, so
# the bundle layout cannot drift silently (the same rule the release package
# enforces through its own table-group registry).
_TABLE_GROUPS: dict[str, str] = {
    "analysis_population": "population",
    "pooled_models": "pooled",
    "model_grid": "pooled",
    "bootstrap_uncertainty": "pooled",
    "leave_one_series_out": "pooled",
    "zero_n_deltas": "pooled",
    "zero_n_delta_summary": "pooled",
    "raw_finite_sensitivity": "pooled",
    "series_slopes": "heterogeneity",
    "series_fixed_intercepts": "heterogeneity",
    "series_covariates": "heterogeneity",
    "heterogeneity_summary": "heterogeneity",
    "heterogeneity_decomposition": "heterogeneity",
    "mixed_model_summary": "heterogeneity",
    "factor_support": "factors",
    "factor_level_summary": "factors",
    "factor_additive_screen": "factors",
    "factor_series_adjusted_screen": "factors",
    "factor_redundancy_audit": "factors",
    "factor_evidence_audit": "factors",
    "series_slope_modifier_screen": "factors",
}

_FIGURE_GROUPS: dict[str, str] = {
    "pooled_linear_response": "pooled",
    "pooled_quadratic_sensitivity": "pooled",
    "raw_finite_population_sensitivity": "pooled",
    "series_response_overlay": "heterogeneity",
    "within_series_response": "heterogeneity",
    "series_slope_distribution": "heterogeneity",
    "factor_baseline_screen": "factors",
    "series_slope_vs_check_yield": "factors",
}

_TABLE_DESCRIPTIONS: dict[str, str] = {
    "analysis_population": (
        "manifest-declared observed-series rows every other artifact uses"
    ),
    "pooled_models": "pooled linear/quadratic/Theil–Sen fits with equations",
    "model_grid": "predicted-yield grid tracing the pooled lines",
    "bootstrap_uncertainty": (
        "seeded cluster-bootstrap intervals for the pooled slope, intercept, "
        "curvature, and turning point"
    ),
    "leave_one_series_out": "pooled linear refits omitting one series at a time",
    "zero_n_deltas": "per-observation yield gain over each series' zero-N check",
    "zero_n_delta_summary": "through-origin slope of the zero-N yield gains",
    "raw_finite_sensitivity": "all-finite-pair source-wide sensitivity fits",
    "series_slopes": "per-series linear fits with slope standard errors",
    "series_fixed_intercepts": "per-series intercepts from the common-slope model",
    "series_covariates": (
        "one row per series: check yield, N-ladder geometry, era, fitted slope"
    ),
    "heterogeneity_summary": "pooled versus within-series slopes and SSE reduction",
    "heterogeneity_decomposition": (
        "study/trial/series fixed-intercept comparison"
    ),
    "mixed_model_summary": "R/lme4 random-intercept random-slope diagnostic",
    "factor_support": "per-factor availability, levels, aliasing, and eligibility",
    "factor_level_summary": "per-level counts and yield summaries",
    "factor_additive_screen": (
        "pooled + factor equations and R² gains (complete-case)"
    ),
    "factor_series_adjusted_screen": (
        "factor gains over a baseline already holding series identity"
    ),
    "factor_redundancy_audit": "aliased and nested factor pairs",
    "factor_evidence_audit": (
        "derived indicators compared with their raw evidence columns"
    ),
    "series_slope_modifier_screen": (
        "series-level covariates screened against fitted series slopes"
    ),
}

_FIGURE_DESCRIPTIONS: dict[str, str] = {
    "pooled_linear_response": "observations by series with the pooled line",
    "pooled_quadratic_sensitivity": "linear versus quadratic pooled shape",
    "raw_finite_population_sensitivity": "all-finite-pair source-wide view",
    "series_response_overlay": "connected series overlaid on the pooled line",
    "within_series_response": "series-centered response association",
    "series_slope_distribution": "per-series slopes with ±1 SE bars",
    "factor_baseline_screen": "complete-case factor screen gains",
    "series_slope_vs_check_yield": "series slope versus zero-N check yield",
}


def table_relative_path(name: str) -> str:
    group = _TABLE_GROUPS.get(name)
    if group is None:
        raise DiagnosticBundleError(
            f"Table {name!r} has no registered bundle group"
        )
    return f"tables/{group}/{name}.csv"


def figure_relative_path(name: str, extension: str) -> str:
    group = _FIGURE_GROUPS.get(name)
    if group is None:
        raise DiagnosticBundleError(
            f"Figure {name!r} has no registered bundle group"
        )
    return f"figures/{group}/{name}.{extension}"


@dataclass(frozen=True)
class WrittenDiagnosticBundle:
    root: Path
    manifest_path: Path
    checksums_path: Path
    artifact_count: int


@dataclass(frozen=True)
class VerifiedDiagnosticBundle:
    root: Path
    status: str
    table_count: int
    figure_count: int
    artifact_count: int


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def _write_csv(root: Path, relative: str, frame: pd.DataFrame) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return {
        "relative_path": relative,
        "artifact_kind": "table",
        "media_type": "text/csv",
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
    }


def _save_figure(
    root: Path,
    relative: str,
    figure: plt.Figure,
    *,
    dpi: int,
) -> dict[str, Any]:
    # The caller owns the figure lifetime: one rendered figure is saved once
    # per configured format, then closed by the caller.
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    format_name = path.suffix.lstrip(".").lower()
    save_format = "jpeg" if format_name in {"jpeg", "jpg"} else format_name
    figure.savefig(path, dpi=dpi, format=save_format, facecolor="white")
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        detected_format = str(image.format)
        mode = image.mode
    return {
        "relative_path": relative,
        "artifact_kind": "figure",
        "media_type": "image/jpeg" if save_format == "jpeg" else f"image/{save_format}",
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "width_pixels": width,
        "height_pixels": height,
        "color_mode": mode,
        "detected_format": detected_format,
    }


def _figure(config: GrainYieldResponseConfig) -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(
        figsize=(config.figure_width_inches, config.figure_height_inches)
    )


def _display_equation(row: pd.Series, *, quadratic: bool = False) -> str:
    intercept = float(row["intercept_t_ha"])
    slope = float(row["slope_t_ha_per_kg_n_ha"])
    equation = f"Ŷ = {intercept:.4f} {'+' if slope >= 0 else '-'} {abs(slope):.7f} N"
    if quadratic:
        curvature = float(row["quadratic_t_ha_per_kg_n_ha_squared"])
        equation += f" {'+' if curvature >= 0 else '-'} {abs(curvature):.7g} N²"
    return equation


_FACTOR_LABELS = {
    "water_regime_normalized": "Water regime",
    "season_normalized": "Season",
    "region_normalized": "Region",
    "province_normalized": "Province",
    "rice_variety_normalized": "Rice variety",
    "planting_year": "Planting year",
    "organic_fertilizer_present": "Organic fertilizer present",
    "biofertilizer_present": "Biofertilizer present",
    "p_rate_kg_p2o5_ha": "P rate (kg P₂O₅ ha⁻¹)",
    "k_rate_kg_k2o_ha": "K rate (kg K₂O ha⁻¹)",
    "series_check_yield_t_ha": "Zero-N check yield (t ha⁻¹)",
    "series_n_span_kg_ha": "N ladder span (kg N ha⁻¹)",
    "series_n_min_kg_ha": "Lowest tested N (kg N ha⁻¹)",
    "series_n_max_kg_ha": "Highest tested N (kg N ha⁻¹)",
    "series_mean_n_rate_kg_ha": "Mean tested N (kg N ha⁻¹)",
    "series_mean_yield_t_ha": "Series mean yield (t ha⁻¹)",
    "series_planting_year_numeric": "Planting year (numeric)",
    "series_observation_count": "Series observation count",
    "series_distinct_n_level_count": "Distinct N levels",
    "series_has_zero_n_control": "Zero-N control present",
    "series_pk_recorded": "P and K rates recorded",
}


def _factor_label(value: object) -> str:
    key = str(value)
    return _FACTOR_LABELS.get(key, key.replace("_", " ").title())


def _scatter_by_series(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    series_key: str,
    connect: bool,
) -> None:
    groups = tuple(frame.groupby(series_key, sort=True))
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(len(groups), 2)))
    for color, (uid, group) in zip(colors, groups):
        ordered = group.sort_values("n_rate_kg_ha")
        label = str(uid) if len(groups) <= 10 else None
        if connect:
            axis.plot(
                ordered["n_rate_kg_ha"],
                ordered["yield_t_ha"],
                marker="o",
                markersize=4,
                linewidth=1.0,
                alpha=0.7,
                color=color,
                label=label,
            )
        else:
            axis.scatter(
                ordered["n_rate_kg_ha"],
                ordered["yield_t_ha"],
                s=28,
                alpha=0.75,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                label=label,
            )
    if len(groups) <= 10:
        axis.legend(title="Response series", fontsize=7, title_fontsize=8)


def _plot_pooled_linear(
    config: GrainYieldResponseConfig,
    population: GovernedPopulation,
    descriptive: DescriptiveResults,
) -> plt.Figure:
    figure, axis = _figure(config)
    _scatter_by_series(
        axis,
        population.frame,
        series_key=config.series_key,
        connect=False,
    )
    grid = descriptive.model_grid[descriptive.model_grid["model"] == "linear"]
    axis.plot(
        grid["n_rate_kg_ha"],
        grid["predicted_yield_t_ha"],
        color="black",
        linewidth=2.2,
        label="Pooled descriptive line",
    )
    linear = descriptive.pooled_models.set_index("model").loc["linear"]
    axis.text(
        0.02,
        0.98,
        f"{_display_equation(linear)}\n"
        f"n={int(linear['observations'])}; series={int(linear['series_count'])}; "
        f"R²={float(linear['r_squared']):.3f}\n"
        "Pooled across series; descriptive only—not causal or a recommendation.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    axis.set(
        title="Grain yield versus inorganic N: pooled descriptive line",
        xlabel="Inorganic N rate (kg N ha⁻¹)",
        ylabel="Grain yield (t ha⁻¹)",
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def _plot_quadratic(
    config: GrainYieldResponseConfig,
    population: GovernedPopulation,
    descriptive: DescriptiveResults,
) -> plt.Figure:
    figure, axis = _figure(config)
    axis.scatter(
        population.frame["n_rate_kg_ha"],
        population.frame["yield_t_ha"],
        s=24,
        color="#4C78A8",
        alpha=0.65,
    )
    for model, color, style, label in (
        ("linear", "black", "--", "Linear diagnostic"),
        ("quadratic", "#E45756", "-", "Quadratic sensitivity"),
    ):
        grid = descriptive.model_grid[descriptive.model_grid["model"] == model]
        axis.plot(
            grid["n_rate_kg_ha"],
            grid["predicted_yield_t_ha"],
            color=color,
            linestyle=style,
            linewidth=2.0,
            label=label,
        )
    quadratic = descriptive.pooled_models.set_index("model").loc["quadratic"]
    turning = quadratic["turning_point_n_kg_ha"]
    turning_text = (
        f"turning point={float(turning):.1f} kg N ha⁻¹; "
        f"inside observed domain={bool(quadratic['turning_point_in_observed_domain'])}"
        if pd.notna(turning)
        else "turning point unavailable"
    )
    axis.text(
        0.02,
        0.98,
        f"Quadratic R²={float(quadratic['r_squared']):.3f}; {turning_text}\n"
        "Visual sensitivity only; no optimum claim.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    axis.set(
        title="Pooled shape sensitivity within the observed N domain",
        xlabel="Inorganic N rate (kg N ha⁻¹)",
        ylabel="Grain yield (t ha⁻¹)",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def _plot_series_overlay(
    config: GrainYieldResponseConfig,
    population: GovernedPopulation,
    descriptive: DescriptiveResults,
) -> plt.Figure:
    figure, axis = _figure(config)
    _scatter_by_series(
        axis,
        population.frame,
        series_key=config.series_key,
        connect=True,
    )
    grid = descriptive.model_grid[descriptive.model_grid["model"] == "linear"]
    axis.plot(
        grid["n_rate_kg_ha"],
        grid["predicted_yield_t_ha"],
        color="black",
        linewidth=2.5,
        label="Pooled descriptive line",
    )
    axis.set(
        title="Manifest-declared observed series and pooled descriptive line",
        xlabel="Inorganic N rate (kg N ha⁻¹)",
        ylabel="Grain yield (t ha⁻¹)",
    )
    axis.text(
        0.02,
        0.02,
        "Colored segments connect observations within series; the black line is "
        "pooled and noncausal.",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def _plot_within_series(
    config: GrainYieldResponseConfig,
    heterogeneity: HeterogeneityResults,
) -> plt.Figure:
    figure, axis = _figure(config)
    frame = heterogeneity.centered_frame
    axis.scatter(
        frame["centered_n_rate_kg_ha"],
        frame["centered_yield_t_ha"],
        s=28,
        alpha=0.7,
        color="#72B7B2",
        edgecolor="white",
        linewidth=0.35,
    )
    x_min = float(frame["centered_n_rate_kg_ha"].min())
    x_max = float(frame["centered_n_rate_kg_ha"].max())
    line_x = np.linspace(x_min, x_max, 201)
    slope = float(
        heterogeneity.summary["within_series_slope_t_ha_per_kg_n_ha"]
    )
    axis.plot(line_x, slope * line_x, color="black", linewidth=2.2)
    axis.text(
        0.02,
        0.98,
        f"Within-series slope={slope:.6f} t ha⁻¹ per kg N ha⁻¹\n"
        f"R²={float(heterogeneity.summary['within_series_r_squared']):.3f}\n"
        "Centered association only; not a causal effect.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    axis.set(
        title="Within-series centered grain-yield response",
        xlabel="N rate minus series mean (kg N ha⁻¹)",
        ylabel="Yield minus series mean (t ha⁻¹)",
    )
    axis.axhline(0, color="grey", linewidth=0.7)
    axis.axvline(0, color="grey", linewidth=0.7)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def _plot_series_slopes(
    config: GrainYieldResponseConfig,
    descriptive: DescriptiveResults,
) -> plt.Figure:
    figure, axis = _figure(config)
    slopes = descriptive.series_slopes[
        descriptive.series_slopes["status"] == "fitted"
    ].sort_values("slope_t_ha_per_kg_n_ha")
    positions = np.arange(len(slopes))
    slope_values = slopes["slope_t_ha_per_kg_n_ha"].to_numpy(dtype=float) * 100
    if "slope_se_t_ha_per_kg_n_ha" in slopes.columns:
        standard_errors = (
            pd.to_numeric(
                slopes["slope_se_t_ha_per_kg_n_ha"], errors="coerce"
            ).to_numpy(dtype=float)
            * 100
        )
        axis.errorbar(
            slope_values,
            positions,
            xerr=np.where(np.isfinite(standard_errors), standard_errors, 0.0),
            fmt="none",
            ecolor="#9D9D9D",
            elinewidth=1.0,
            capsize=2,
            zorder=2,
        )
    axis.scatter(
        slope_values,
        positions,
        color="#F58518",
        s=38,
        zorder=3,
    )
    axis.axvline(0, color="black", linestyle="--", linewidth=1.0)
    if len(slopes) <= 25:
        axis.set_yticks(positions)
        series_column = (
            config.series_key
            if config.series_key in slopes.columns
            else slopes.columns[0]
        )
        axis.set_yticklabels(slopes[series_column].astype(str), fontsize=6)
    else:
        axis.set_yticks([])
    axis.set(
        title="Response-series slope heterogeneity",
        xlabel="Series slope (t ha⁻¹ per 100 kg N ha⁻¹)",
        ylabel="Response series",
    )
    axis.grid(axis="x", alpha=0.2)
    figure.text(
        0.5,
        0.015,
        "Descriptive slopes use only each series' observed points. Bars show "
        "±1 SE where estimable (two-point ladders have none); rows are not "
        "effect rankings.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    return figure


def _plot_factor_screen(
    config: GrainYieldResponseConfig,
    factors: FactorSupportResults,
) -> plt.Figure:
    figure, axis = _figure(config)
    screen = factors.additive_screen.copy()
    adjusted_gain = pd.Series(
        pd.to_numeric(
            screen.loc[:, "adjusted_r_squared_improvement"], errors="coerce"
        ),
        index=screen.index,
        dtype=float,
    )
    fitted_mask = pd.Series(
        screen.loc[:, "screen_status"], index=screen.index
    ).eq("fitted")
    screen = screen.loc[fitted_mask & adjusted_gain.notna(), :].copy()
    screen.loc[:, "adjusted_r_squared_improvement"] = adjusted_gain.loc[
        screen.index
    ]
    screen = screen.sort_values(by=["adjusted_r_squared_improvement"])
    if screen.empty:
        axis.text(
            0.5,
            0.5,
            "No factor passed the descriptive additive-screen gate.",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
        axis.set_axis_off()
    else:
        positions = np.arange(len(screen))
        values = screen["adjusted_r_squared_improvement"].to_numpy(dtype=float)
        colors = [
            "#ECA82C" if str(status).startswith("held_") else "#54A24B"
            for status in screen["modifier_status"]
        ]
        bars = axis.barh(
            positions,
            values,
            color=colors,
        )
        minimum = min(0.0, float(np.min(values)))
        maximum = max(0.0, float(np.max(values)))
        span = max(maximum - minimum, 0.01)
        axis.set_xlim(minimum - 0.04 * span, maximum + 0.20 * span)
        axis.set_yticks(positions)
        axis.set_yticklabels([_factor_label(value) for value in screen["factor"]])
        for bar, (_, row) in zip(bars, screen.iterrows()):
            value = float(row["adjusted_r_squared_improvement"])
            horizontal_alignment = "left" if value >= 0 else "right"
            offset = 3 if value >= 0 else -3
            axis.annotate(
                f"n={int(row['observations'])}; +df={int(row['factor_parameters_added'])}",
                xy=(value, bar.get_y() + bar.get_height() / 2),
                xytext=(offset, 0),
                textcoords="offset points",
                ha=horizontal_alignment,
                va="center",
                fontsize=7,
            )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set(
            title="Complete-case factor baseline associations (noncausal)",
            xlabel="Adjusted R² change versus the same-row N-only model",
            ylabel="Factor",
        )
        axis.grid(axis="x", alpha=0.2)
        figure.text(
            0.5,
            0.015,
            "Rows differ in sample size and degrees of freedom; they are not "
            "effect-size rankings. Orange means the N×factor modifier was held.",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    return figure


def _plot_series_modifiers(
    config: GrainYieldResponseConfig,
    series_covariates: SeriesCovariateResults,
) -> plt.Figure:
    """Series slope against zero-N check yield, the leading modifier candidate."""

    figure, axis = _figure(config)
    frame = series_covariates.series_frame
    slopes = pd.to_numeric(
        frame["series_slope_t_ha_per_kg_n_ha"], errors="coerce"
    )
    checks = pd.to_numeric(frame["series_check_yield_t_ha"], errors="coerce")
    usable = slopes.notna() & checks.notna()
    without_check = int(slopes.notna().sum() - usable.sum())
    if int(usable.sum()) < 3:
        axis.text(
            0.5,
            0.5,
            "Fewer than three series carry both a fitted slope and a zero-N "
            "check yield; the modifier view is withheld.",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
        axis.set_axis_off()
        figure.tight_layout()
        return figure
    x = checks.loc[usable].to_numpy(dtype=float)
    y = slopes.loc[usable].to_numpy(dtype=float) * 100.0
    axis.scatter(
        x,
        y,
        s=52,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    # Reserve headroom so the annotation box cannot cover the highest point.
    y_low, y_high = axis.get_ylim()
    axis.set_ylim(y_low, y_high + 0.22 * (y_high - y_low))
    screen = series_covariates.modifier_screen
    check_row = screen[
        (screen["covariate"] == "series_check_yield_t_ha")
        & (screen["screen_status"] == "fitted")
    ]
    if not check_row.empty:
        row = check_row.iloc[0]
        line_x = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        line_y = (
            float(row["slope_intercept_t_ha_per_kg_n_ha"])
            + float(row["slope_change_per_covariate_unit"]) * line_x
        ) * 100.0
        axis.plot(
            line_x,
            line_y,
            color="black",
            linewidth=1.6,
            linestyle="--",
            label="Series-level association",
        )
        axis.text(
            0.02,
            0.98,
            f"Series used: {int(row['series_used'])} of {len(frame)} "
            f"(the other {without_check} have no zero-N member)\n"
            f"Pearson r = {float(row['pearson_r']):.2f}; "
            f"R² = {float(row['r_squared']):.2f}\n"
            "Series-level association only; confounded with study, site, era, "
            "and design. Not causal.\n"
            "Check yield and slope share observations; mechanical correlation "
            "is possible.",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
        )
        axis.legend(fontsize=8, loc="lower left")
    axis.set(
        title="Series N-response slope versus zero-N check yield",
        xlabel="Zero-N check yield (t ha⁻¹)",
        ylabel="Series slope (t ha⁻¹ per 100 kg N ha⁻¹)",
    )
    axis.axhline(0, color="grey", linewidth=0.7)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def _plot_raw_sensitivity(
    config: GrainYieldResponseConfig,
    raw_sensitivity: Mapping[str, Any],
) -> plt.Figure:
    frame = raw_sensitivity.get("frame")
    summary = raw_sensitivity.get("summary")
    if not isinstance(frame, pd.DataFrame) or not isinstance(summary, pd.DataFrame):
        raise DiagnosticBundleError(
            "Raw sensitivity must provide frame and summary DataFrames"
        )
    figure, axis = _figure(config)
    axis.scatter(
        frame["n_rate_kg_ha"],
        frame["yield_t_ha"],
        s=13,
        alpha=0.35,
        color="#4C78A8",
        edgecolor="none",
    )
    x_min = float(frame["n_rate_kg_ha"].min())
    x_max = float(frame["n_rate_kg_ha"].max())
    grid = np.linspace(x_min, x_max, 301)
    model_rows = summary.set_index("model")
    linear = model_rows.loc["raw_finite_linear"]
    axis.plot(
        grid,
        float(linear["intercept_t_ha"])
        + float(linear["slope_t_ha_per_kg_n_ha"]) * grid,
        color="black",
        linewidth=2.1,
        label="Raw finite-pair linear sensitivity",
    )
    quadratic = model_rows.loc["raw_finite_quadratic_sensitivity"]
    axis.plot(
        grid,
        float(quadratic["intercept_t_ha"])
        + float(quadratic["slope_t_ha_per_kg_n_ha"]) * grid
        + float(quadratic["quadratic_t_ha_per_kg_n_ha_squared"]) * grid**2,
        color="#E45756",
        linestyle="--",
        linewidth=1.8,
        label="Raw finite-pair quadratic sensitivity",
    )
    axis.text(
        0.02,
        0.98,
        f"{_display_equation(linear)}\n"
        f"n={int(linear['observations'])}; R²={float(linear['r_squared']):.3f}\n"
        "Direct finite-pair inventory fit; pooled, noncausal, and not a recommendation.\n"
        "Population differs from the manifest-declared overlay snapshot.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    axis.set(
        title="All-finite-row grain-yield sensitivity",
        xlabel="Inorganic N rate (kg N ha⁻¹)",
        ylabel="Grain yield (t ha⁻¹)",
    )
    axis.legend(fontsize=8, loc="lower right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def _slope_confidence_sentence(row: pd.Series) -> str:
    if str(row.get("cluster_robust_status", "")) != "completed":
        return (
            "A cluster-robust slope interval was withheld "
            f"(`{row.get('cluster_robust_reason_code', 'not_available')}`)."
        )
    low = float(row["cluster_robust_slope_ci95_low"])
    high = float(row["cluster_robust_slope_ci95_high"])
    return (
        "Cluster-robust (by response series) 95% CI for the slope: "
        f"{low:.4f} to {high:.4f} t ha⁻¹ per kg N ha⁻¹ "
        f"({low * 100:.2f} to {high * 100:.2f} t ha⁻¹ per 100 kg N ha⁻¹)."
    )


def _alias_group_lookup(redundancy_audit: pd.DataFrame | None) -> dict[str, str]:
    """Map each factor to a canonical representative of its perfect-alias group."""

    parent: dict[str, str] = {}

    def find(name: str) -> str:
        parent.setdefault(name, name)
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    if redundancy_audit is not None and not redundancy_audit.empty:
        aliases = redundancy_audit[
            redundancy_audit["redundancy_status"] == "perfect_alias"
        ]
        for _, row in aliases.iterrows():
            root_a = find(str(row["factor_a"]))
            root_b = find(str(row["factor_b"]))
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)
    return {name: find(name) for name in list(parent)}


def _augmented_equation_lines(
    factors: FactorSupportResults,
    series_covariates: SeriesCovariateResults | None,
    redundancy_audit: pd.DataFrame | None,
    evidence_audit: pd.DataFrame | None,
    mixed_model: Mapping[str, Any],
) -> list[str]:
    """Render the fitted equations that carry variables beyond nitrogen rate."""

    lines: list[str] = [
        "### Augmented descriptive equations (still not causal)\n",
        "These are the fitted equations that include variables beyond "
        "`n_rate_kg_ha`. Each is descriptive: the added coefficients absorb "
        "whatever the aliased study/site/era structure contributes, so none is "
        "a causal effect of the named variable.\n",
    ]

    evidence_flagged: set[str] = set()
    if evidence_audit is not None and not evidence_audit.empty:
        evidence_flagged = set(
            evidence_audit.loc[
                evidence_audit["evidence_status"].isin(
                    {
                        "no_raw_evidence",
                        "positive_without_raw_evidence",
                        "missing_coerced_to_negative",
                    }
                ),
                "derived_factor",
            ].astype(str)
        )

    screen = factors.additive_screen
    printed_any = False
    if "equation" in screen.columns:
        fitted = screen[
            screen["screen_status"].eq("fitted")
            & screen["equation"].astype(str).str.len().gt(0)
        ].copy()
        fitted["_gain"] = pd.to_numeric(
            fitted.get("adjusted_r_squared_improvement"), errors="coerce"
        )
        fitted = fitted.sort_values("_gain", ascending=False)
        # Headline only full-population fits: complete-case subsets answer a
        # different question and belong in the CSV next to their warnings.
        full_observations = (
            int(pd.to_numeric(fitted["observations"], errors="coerce").max())
            if not fitted.empty
            else 0
        )
        alias_of = _alias_group_lookup(redundancy_audit)
        emitted_groups: set[str] = set()
        shown = 0
        for _, row in fitted.iterrows():
            parameters = pd.to_numeric(
                pd.Series([row.get("factor_parameters_added")]), errors="coerce"
            ).iloc[0]
            if not pd.notna(parameters) or int(parameters) > 3:
                continue
            if int(row["observations"]) < full_observations:
                continue
            if str(row["factor"]) in evidence_flagged:
                continue
            group = alias_of.get(str(row["factor"]), str(row["factor"]))
            if group in emitted_groups:
                continue
            emitted_groups.add(group)
            reference = str(row.get("reference_level", "") or "")
            reference_text = (
                f"; reference level `{reference}`" if reference else ""
            )
            alias_members = [
                name for name, root in alias_of.items()
                if root == group and name != str(row["factor"])
            ]
            alias_text = (
                " (equivalently " + ", ".join(f"`{m}`" for m in sorted(alias_members)) + ")"
                if alias_members
                else ""
            )
            lines.append(
                f"- Pooled + `{row['factor']}`{alias_text} "
                f"(n={int(row['observations'])}, adjusted R² "
                f"{float(row['augmented_adjusted_r_squared']):.3f}"
                f"{reference_text}):\n\n"
                f"  `{row['equation']}`"
            )
            printed_any = True
            shown += 1
            if shown >= 3:
                break
        remaining = int(len(fitted)) - shown
        if remaining > 0:
            lines.append(
                f"- The other {remaining} fitted augmented equations — the "
                "multi-level models, complete-case subsets (P/K rows), and "
                "evidence-flagged factors — are in the `equation` column of "
                "`tables/factors/factor_additive_screen.csv`, next to their "
                "warnings."
            )

    intercept = mixed_model.get("fixed_intercept_t_ha")
    slope = mixed_model.get("fixed_slope_t_ha_per_100_kg_n_ha")
    if isinstance(intercept, (int, float)) and isinstance(slope, (int, float)) and (
        math.isfinite(float(intercept)) and math.isfinite(float(slope))
    ):
        sd_parts: list[str] = []
        for key, label in (
            ("random_intercept_sd_t_ha", "SD(u_series)"),
            ("random_slope_sd_t_ha_per_100_kg_n_ha", "SD(v_series)"),
            ("residual_sd_t_ha", "SD(ε)"),
        ):
            value = mixed_model.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                sd_parts.append(f"{label} = {float(value):.2f}")
        lines.append(
            "- Hierarchical (R/lme4; the variables beyond N are the "
            "per-series effects `u_series` and `v_series`):\n\n"
            f"  `yield_t_ha = {float(intercept):.4g} "
            f"{'+' if float(slope) >= 0 else '-'} {abs(float(slope)):.4g} * "
            "(n_rate_kg_ha/100) + u_series + v_series * (n_rate_kg_ha/100) + ε`"
            + (f" with {', '.join(sd_parts)} (t ha⁻¹)." if sd_parts else "")
        )
        printed_any = True

    if series_covariates is not None:
        modifier = series_covariates.modifier_screen
        if "slope_equation" in modifier.columns:
            fitted_modifiers = modifier[
                modifier["screen_status"].eq("fitted")
                & modifier["slope_equation"].astype(str).str.len().gt(0)
            ].copy()
            fitted_modifiers["_abs_r"] = pd.to_numeric(
                fitted_modifiers["pearson_r"], errors="coerce"
            ).abs()
            fitted_modifiers = fitted_modifiers.sort_values(
                "_abs_r", ascending=False
            )
            for _, row in fitted_modifiers.head(2).iterrows():
                caveat = str(row.get("estimation_artefact_warning", "") or "")
                lines.append(
                    f"- Response-series slope versus `{row['covariate']}` "
                    f"({int(row['series_used'])} series, "
                    f"r = {float(row['pearson_r']):.2f}):\n\n"
                    f"  `{row['slope_equation']}`"
                    + (f"\n\n  {caveat}" if caveat else "")
                )
                printed_any = True

    if not printed_any:
        lines.append(
            "- No augmented equation was estimable in this population; see the "
            "screen tables for the gate that held each factor."
        )
    return lines


def _factors_beyond_nitrogen_markdown(
    factors: FactorSupportResults,
    series_covariates: SeriesCovariateResults | None,
    series_adjusted_screen: pd.DataFrame | None,
    redundancy_audit: pd.DataFrame | None,
    evidence_audit: pd.DataFrame | None,
    mixed_model: Mapping[str, Any],
) -> str:
    lines: list[str] = ["## Factors beyond nitrogen rate\n"]

    zero_variation = int(
        factors.support["within_series_variation_count"].eq(0).sum()
    )
    factor_count = int(len(factors.support))
    if zero_variation == factor_count:
        lines.append(
            "Nitrogen rate is the only quantity that varies within a response "
            "series in this population: every configured factor is constant "
            "inside each series (`within_series_variation_count` in "
            "`tables/factors/factor_support.csv`). Such factors can describe "
            "which series differ, but no regression on these data can separate "
            "them from series identity, and none of them can bend an observed "
            "within-series response curve.\n"
        )
    else:
        lines.append(
            f"{factor_count - zero_variation} of {factor_count} configured "
            "factors vary within at least one response series; the rest are "
            "series-level constants that cannot be separated from series "
            "identity (`tables/factors/factor_support.csv`).\n"
        )

    if series_adjusted_screen is not None and not series_adjusted_screen.empty:
        absorbed = series_adjusted_screen[
            series_adjusted_screen["series_adjusted_status"]
            == "absorbed_by_series_intercepts"
        ]["factor"].astype(str).tolist()
        if absorbed:
            lines.append(
                "- **Series-adjusted screen** "
                "(`tables/factors/factor_series_adjusted_screen.csv`): with one "
                "intercept per series already in the baseline, "
                f"{len(absorbed)} of {len(series_adjusted_screen)} factors "
                "are wholly absorbed by series identity "
                f"({', '.join('`' + name + '`' for name in absorbed)}); their "
                "naive screen gains restate baseline differences between "
                "series, not factor information."
            )

    if redundancy_audit is not None and not redundancy_audit.empty:
        aliases = redundancy_audit[
            redundancy_audit["redundancy_status"] == "perfect_alias"
        ]
        nested_count = int(
            redundancy_audit["redundancy_status"]
            .eq("fully_nested_or_determined")
            .sum()
        )
        if not aliases.empty or nested_count:
            fragments: list[str] = []
            if not aliases.empty:
                shown = [
                    f"`{row['factor_a']}` = `{row['factor_b']}`"
                    for _, row in aliases.head(8).iterrows()
                ]
                overflow = len(aliases) - len(shown)
                pair_text = ", ".join(shown) + (
                    f" (and {overflow} more)" if overflow > 0 else ""
                )
                fragments.append(
                    f"{pair_text} partition these observations identically, so "
                    "identical screen rows for them are one contrast reported "
                    "under several names, not independent corroboration"
                )
            if nested_count:
                fragments.append(
                    f"{nested_count} further pair"
                    f"{'s are' if nested_count != 1 else ' is'} fully nested or "
                    "one-way determined, so their screen rows overlap without "
                    "being separable"
                )
            lines.append(
                "- **Redundant encodings** "
                "(`tables/factors/factor_redundancy_audit.csv`): "
                + "; ".join(fragments)
                + "."
            )

    if evidence_audit is not None and not evidence_audit.empty:
        flagged = evidence_audit[
            evidence_audit["evidence_status"].isin(
                {
                    "no_raw_evidence",
                    "positive_without_raw_evidence",
                    "missing_coerced_to_negative",
                }
            )
        ]
        if not flagged.empty:
            findings = "; ".join(
                f"`{row['derived_factor']}` is `{row['evidence_status']}` "
                f"against `{row['raw_evidence_column']}`"
                for _, row in flagged.iterrows()
            )
            lines.append(
                "- **Evidence audit** "
                "(`tables/factors/factor_evidence_audit.csv`): "
                f"{findings}. These flags track recording practice, not a "
                "verified field condition, and their screen rows must not be "
                "interpreted agronomically."
            )

    if series_covariates is not None:
        screen = series_covariates.modifier_screen
        fitted = screen[screen["screen_status"] == "fitted"].copy()
        if not fitted.empty:
            fitted["abs_r"] = pd.to_numeric(
                fitted["pearson_r"], errors="coerce"
            ).abs()
            fitted = fitted.sort_values("abs_r", ascending=False)
            top = fitted.iloc[0]
            spearman = pd.to_numeric(
                pd.Series([top.get("spearman_rho")]), errors="coerce"
            ).iloc[0]
            spearman_text = (
                f", Spearman ρ = {float(spearman):.2f}"
                if pd.notna(spearman)
                else ""
            )
            shared_outcome_note = (
                " The check yield and fitted slope share observations, so this "
                "correlation may partly be a mechanical estimation artefact."
                if str(top["covariate"]) == "series_check_yield_t_ha"
                else ""
            )
            lines.append(
                "- **Series-level covariates** "
                "(`tables/heterogeneity/series_covariates.csv`, "
                "`tables/factors/series_slope_modifier_screen.csv`): derived "
                "zero-N check yield, N-ladder geometry, era, and P/K recording "
                "state are screened against the fitted series slopes. The "
                "largest absolute correlation among screens with differing "
                f"series availability is `{top['covariate']}` "
                f"(Pearson r = {float(top['pearson_r']):.2f}{spearman_text} "
                f"across {int(top['series_used'])} series). It is the "
                f"strongest of {len(fitted)} fitted screens, and selecting the "
                "maximum inflates apparent strength; the screen is also "
                "unweighted although the series slopes are estimated with "
                f"unequal precision.{shared_outcome_note} "
                "Series-level associations "
                "are confounded with study, site, era, and design; none is a "
                "causal moderator claim."
            )

    pk_sentence = _phosphorus_potassium_sentence(
        factors, series_adjusted_screen, redundancy_audit
    )
    if pk_sentence:
        lines.append(pk_sentence)

    lines.append("")
    lines.extend(
        _augmented_equation_lines(
            factors,
            series_covariates,
            redundancy_audit,
            evidence_audit,
            mixed_model,
        )
    )

    lines.append(
        "\nNo N-by-factor interaction was estimated. Identifying genuine "
        "response-curve modifiers needs series that vary the candidate factor "
        "at matched N rates, which this population does not contain.\n"
    )
    return "\n".join(lines)


def _phosphorus_potassium_sentence(
    factors: FactorSupportResults,
    series_adjusted_screen: pd.DataFrame | None,
    redundancy_audit: pd.DataFrame | None,
) -> str:
    support = factors.support.set_index("factor")
    if (
        "p_rate_kg_p2o5_ha" not in support.index
        or "k_rate_kg_k2o_ha" not in support.index
    ):
        return ""
    p_row = support.loc["p_rate_kg_p2o5_ha"]
    total = int(p_row["observations"]) + int(p_row["missing_observations"])
    fragments = [
        "- **Phosphorus and potassium**: recorded on "
        f"{int(p_row['observations'])} of {total} observations"
    ]
    if redundancy_audit is not None and not redundancy_audit.empty:
        pk_pair = redundancy_audit[
            (
                redundancy_audit["factor_a"].isin(
                    ["p_rate_kg_p2o5_ha", "k_rate_kg_k2o_ha"]
                )
            )
            & (
                redundancy_audit["factor_b"].isin(
                    ["p_rate_kg_p2o5_ha", "k_rate_kg_k2o_ha"]
                )
            )
        ]
        if not pk_pair.empty:
            value = pk_pair.iloc[0]["association_value"]
            if pd.notna(value):
                fragments.append(
                    f", and their recorded rates move in lockstep "
                    f"(r = {float(value):.2f}), so P and K form a single "
                    "contrast rather than two separable factors"
                )
    if series_adjusted_screen is not None and not series_adjusted_screen.empty:
        indexed = series_adjusted_screen.set_index("factor")
        if {"p_rate_kg_p2o5_ha", "k_rate_kg_k2o_ha"} <= set(indexed.index):
            variation = pd.to_numeric(
                indexed.loc[
                    ["p_rate_kg_p2o5_ha", "k_rate_kg_k2o_ha"],
                    "within_series_variation_count",
                ],
                errors="coerce",
            )
            if variation.notna().all() and (variation == 0).all():
                fragments.append(
                    ". Both are constant within every series, so they can "
                    "shift series baselines but cannot modify the shape of "
                    "any observed N-response curve in these data"
                )
    fragments.append(
        "; where they are missing, the missingness itself follows study "
        "membership, so \"P/K recorded\" is also a study label."
    )
    return "".join(fragments)


def _figures_note(figure_paths: tuple[str, ...], *names: str) -> str:
    references = []
    for name in names:
        for relative in figure_paths:
            if PurePosixPath(relative).name.startswith(f"{name}."):
                references.append(f"`{relative}`")
                break
    if not references:
        return ""
    label = "Figure" if len(references) == 1 else "Figures"
    return f"{label}: {', '.join(references)}.\n\n"


def _theil_sentence(descriptive: DescriptiveResults) -> str:
    models = descriptive.pooled_models.set_index("model")
    if "theil_sen_linear_sensitivity" not in models.index:
        return ""
    row = models.loc["theil_sen_linear_sensitivity"]
    slope = pd.to_numeric(
        pd.Series([row.get("slope_t_ha_per_kg_n_ha")]), errors="coerce"
    ).iloc[0]
    if pd.isna(slope):
        return ""
    low = row.get("cluster_robust_slope_ci95_low")
    high = row.get("cluster_robust_slope_ci95_high")
    interval = (
        f" (95% CI {float(low) * 100:.2f} to {float(high) * 100:.2f})"
        if pd.notna(low) and pd.notna(high)
        else ""
    )
    return (
        "A Theil–Sen median-slope fit, robust to individual observations, "
        f"gives {float(slope) * 100:.2f} t ha⁻¹ per 100 kg N ha⁻¹"
        f"{interval}. "
    )


def _loo_sentence(descriptive: DescriptiveResults) -> str:
    loo = descriptive.leave_one_series_out
    if loo.empty:
        return ""
    slopes = pd.to_numeric(
        loo["slope_t_ha_per_kg_n_ha"], errors="coerce"
    ).dropna()
    if slopes.empty:
        return ""
    return (
        "Leave-one-series-out refits keep the pooled slope between "
        f"{float(slopes.min()) * 100:.2f} and {float(slopes.max()) * 100:.2f} "
        "t ha⁻¹ per 100 kg N ha⁻¹ "
        "(`tables/pooled/leave_one_series_out.csv`). "
    )


def _bootstrap_sentences(
    bootstrap_uncertainty: pd.DataFrame | None,
) -> tuple[str, str]:
    """Return (pooled-slope sentence, turning-point sentence), either may be empty."""

    if (
        bootstrap_uncertainty is None
        or bootstrap_uncertainty.empty
        or "quantity" not in bootstrap_uncertainty.columns
    ):
        return "", ""
    indexed = bootstrap_uncertainty.set_index("quantity")
    slope_sentence = ""
    slope_key = "pooled_linear_slope_t_ha_per_kg_n_ha"
    if slope_key in indexed.index:
        row = indexed.loc[slope_key]
        if str(row.get("status", "")) == "fitted":
            slope_sentence = (
                "A cluster bootstrap that resamples whole response series "
                f"({int(row['resamples_used'])} usable resamples, seed "
                f"{int(row['random_seed'])}) puts the slope's 95% interval at "
                f"{float(row['ci95_low']) * 100:.2f} to "
                f"{float(row['ci95_high']) * 100:.2f} t ha⁻¹ per 100 kg N ha⁻¹ "
                "(`tables/pooled/bootstrap_uncertainty.csv`). "
            )
    turning_sentence = ""
    turning_key = "pooled_quadratic_turning_point_n_kg_ha"
    if turning_key in indexed.index:
        row = indexed.loc[turning_key]
        concave = pd.to_numeric(
            pd.Series([row.get("share_concave_resamples")]), errors="coerce"
        ).iloc[0]
        inside = pd.to_numeric(
            pd.Series([row.get("share_turning_point_inside_observed_domain")]),
            errors="coerce",
        ).iloc[0]
        low = row.get("ci95_low")
        high = row.get("ci95_high")
        if pd.notna(concave) and pd.notna(low) and pd.notna(high):
            inside_text = (
                f" and {100 * float(inside):.0f}% of those maxima fall inside "
                "the observed N domain"
                if pd.notna(inside)
                else ""
            )
            turning_sentence = (
                "Cluster-bootstrap check: "
                f"{100 * float(concave):.0f}% of resamples fit a concave "
                "(downward-bending) parabola; among those, the turning point's "
                f"95% interval spans {float(low):.0f} to {float(high):.0f} "
                f"kg N ha⁻¹{inside_text} — the pooled data do not locate a "
                "maximum (`tables/pooled/bootstrap_uncertainty.csv`).\n\n"
            )
    return slope_sentence, turning_sentence


def _zero_n_delta_sentence(descriptive: DescriptiveResults) -> str:
    summary = descriptive.zero_n_delta_summary
    if summary.get("status") != "fitted":
        return ""
    slope = summary.get("slope_through_origin_t_ha_per_kg_n_ha")
    if not isinstance(slope, (int, float)) or not math.isfinite(float(slope)):
        return ""
    return (
        "Measured against each series' own zero-N check, the through-origin "
        f"gain slope is {float(slope) * 100:.2f} t ha⁻¹ per 100 kg N ha⁻¹ "
        f"across {int(summary.get('nonzero_observations', 0))} fertilized "
        f"observations in {int(summary.get('series_with_zero_n', 0))} series "
        "(`tables/pooled/zero_n_delta_summary.csv`).\n\n"
    )


def _key_findings_lines(
    linear: pd.Series,
    quadratic: pd.Series,
    heterogeneity: HeterogeneityResults,
    factors: FactorSupportResults,
    mixed_model: Mapping[str, Any],
    bootstrap_uncertainty: pd.DataFrame | None,
    redundancy_audit: pd.DataFrame | None,
) -> list[str]:
    lines: list[str] = ["## Key findings at a glance\n"]

    slope_text = (
        "- **Pooled descriptive slope**: "
        f"{float(linear['slope_t_ha_per_kg_n_ha']) * 100:.2f} t ha⁻¹ per "
        f"100 kg N ha⁻¹ (R² = {float(linear['r_squared']):.2f})"
    )
    if str(linear.get("cluster_robust_status", "")) == "completed":
        slope_text += (
            "; cluster-robust 95% CI "
            f"{float(linear['cluster_robust_slope_ci95_low']) * 100:.2f} to "
            f"{float(linear['cluster_robust_slope_ci95_high']) * 100:.2f}"
        )
    within = float(
        heterogeneity.summary["within_series_slope_t_ha_per_kg_n_ha"]
    )
    slope_text += (
        f". The within-series slope is smaller ({within * 100:.2f}), so part "
        "of the pooled association is between-series baseline structure."
    )
    lines.append(slope_text)

    sse = heterogeneity.summary.get(
        "series_fixed_intercept_sse_reduction_fraction"
    )
    structure_text = (
        "- **Series identity dominates**: one intercept per series removes "
        f"{100 * float(sse):.0f}% of the pooled residual SSE"
        if isinstance(sse, (int, float)) and math.isfinite(float(sse))
        else "- **Series identity dominates the residual structure**"
    )
    intercept_sd = mixed_model.get("random_intercept_sd_t_ha")
    slope_sd = mixed_model.get("random_slope_sd_t_ha_per_100_kg_n_ha")
    if isinstance(intercept_sd, (int, float)) and math.isfinite(
        float(intercept_sd)
    ):
        structure_text += (
            f"; the hierarchical fit puts the between-series baseline SD at "
            f"{float(intercept_sd):.2f} t ha⁻¹"
        )
        if isinstance(slope_sd, (int, float)) and math.isfinite(float(slope_sd)):
            structure_text += (
                f" and the between-series slope SD at {float(slope_sd):.2f} "
                "t ha⁻¹ per 100 kg N ha⁻¹ — series differ in response, not "
                "just baseline"
            )
    lines.append(structure_text + ".")

    screen = factors.additive_screen
    if "adjusted_r_squared_improvement" in screen.columns:
        fitted_rows = screen[screen["screen_status"].eq("fitted")].copy()
        fitted_rows["_gain"] = pd.to_numeric(
            fitted_rows["adjusted_r_squared_improvement"], errors="coerce"
        )
        fitted_rows = fitted_rows.dropna(subset=["_gain"]).sort_values(
            "_gain", ascending=False
        )
        if not fitted_rows.empty:
            # One entry per alias group, each with its own n: a perfectly
            # aliased pair is one contrast, and complete-case subsets must not
            # read as full-population gains.
            alias_of = _alias_group_lookup(redundancy_audit)
            emitted_groups: set[str] = set()
            entries: list[str] = []
            for _, row in fitted_rows.iterrows():
                group = alias_of.get(str(row["factor"]), str(row["factor"]))
                if group in emitted_groups:
                    continue
                emitted_groups.add(group)
                alias_members = sorted(
                    name
                    for name, root in alias_of.items()
                    if root == group and name != str(row["factor"])
                )
                alias_note = (
                    "; aliased with "
                    + ", ".join(f"`{name}`" for name in alias_members)
                    if alias_members
                    else ""
                )
                entries.append(
                    f"`{row['factor']}` ({float(row['_gain']):+.3f}, "
                    f"n={int(row['observations'])}{alias_note})"
                )
                if len(entries) >= 3:
                    break
            lines.append(
                "- **Largest complete-case screen gains** (adjusted R², versus "
                f"the same-row N-only line): {', '.join(entries)}. All "
                "configured factors are series-constant here, so these gains "
                "restate between-series structure, not separable factor effects."
            )

    turning = quadratic.get("turning_point_n_kg_ha")
    if pd.notna(turning):
        optimum_text = (
            "- **No agronomic optimum**: the quadratic turning point "
            f"({float(turning):.0f} kg N ha⁻¹) sits "
            + (
                "inside the observed domain but is a pooled artifact"
                if bool(quadratic.get("turning_point_in_observed_domain"))
                else "beyond the observed N domain"
            )
        )
        if (
            bootstrap_uncertainty is not None
            and not bootstrap_uncertainty.empty
            and "quantity" in bootstrap_uncertainty.columns
        ):
            indexed = bootstrap_uncertainty.set_index("quantity")
            key = "pooled_quadratic_turning_point_n_kg_ha"
            if key in indexed.index:
                inside_share = pd.to_numeric(
                    pd.Series(
                        [
                            indexed.loc[key].get(
                                "share_turning_point_inside_observed_domain"
                            )
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(inside_share):
                    optimum_text += (
                        f", and only {100 * float(inside_share):.0f}% of the "
                        "bootstrap resamples that fit a concave shape place it "
                        "inside that domain"
                    )
        lines.append(optimum_text + ". No optimum is inferred.")

    lines.append(
        "- **Governance**: everything here is descriptive. No causal claim, "
        "fertilizer recommendation, agronomic optimum, or N-by-factor "
        "interaction is produced."
    )
    lines.append("")
    return lines


def _bundle_contents_lines(
    table_paths: tuple[str, ...],
    figure_paths: tuple[str, ...],
) -> list[str]:
    if not table_paths and not figure_paths:
        return []
    lines: list[str] = [
        "## Bundle contents\n",
        "Artifacts are grouped by theme. `run_manifest.json` inventories every "
        "artifact with hashes and semantic metadata; `CHECKSUMS.sha256` binds "
        "the whole bundle.\n",
    ]

    group_order = {"population": 0, "pooled": 1, "heterogeneity": 2, "factors": 3}

    def emit(paths: tuple[str, ...], descriptions: Mapping[str, str]) -> None:
        by_directory: dict[str, list[str]] = {}
        for relative in paths:
            pure = PurePosixPath(relative)
            by_directory.setdefault(str(pure.parent), []).append(pure.name)
        for directory in sorted(
            by_directory,
            key=lambda name: (
                group_order.get(PurePosixPath(name).name, 99),
                name,
            ),
        ):
            lines.append(f"### `{directory}/`\n")
            for filename in sorted(set(by_directory[directory])):
                stem = filename.split(".", 1)[0]
                description = descriptions.get(stem, "")
                lines.append(
                    f"- `{filename}`"
                    + (f" — {description}" if description else "")
                )
            lines.append("")

    emit(table_paths, _TABLE_DESCRIPTIONS)
    emit(figure_paths, _FIGURE_DESCRIPTIONS)
    return lines


def _summary_markdown(
    population: GovernedPopulation,
    descriptive: DescriptiveResults,
    heterogeneity: HeterogeneityResults,
    factors: FactorSupportResults,
    series_covariates: SeriesCovariateResults | None,
    series_adjusted_screen: pd.DataFrame | None,
    redundancy_audit: pd.DataFrame | None,
    evidence_audit: pd.DataFrame | None,
    mixed_model: Mapping[str, Any],
    raw_sensitivity: Mapping[str, Any] | None,
    bootstrap_uncertainty: pd.DataFrame | None = None,
    table_paths: tuple[str, ...] = (),
    figure_paths: tuple[str, ...] = (),
) -> str:
    linear = descriptive.pooled_models.set_index("model").loc["linear"]
    quadratic = descriptive.pooled_models.set_index("model").loc["quadratic"]
    raw_block = ""
    if raw_sensitivity is not None:
        raw_summary = raw_sensitivity.get("summary")
        if isinstance(raw_summary, pd.DataFrame):
            raw_linear = raw_summary.set_index("model").loc["raw_finite_linear"]
            raw_block = (
                "## Direct source-wide finite-pair fit (sensitivity)\n\n"
                f"`{raw_linear['equation']}`\n\n"
                f"Finite N–yield pairs: {int(raw_linear['observations'])}; "
                f"R² = {float(raw_linear['r_squared']):.6f}; "
                f"RMSE = {float(raw_linear['rmse_t_ha']):.6f} t ha⁻¹.\n\n"
                "This is the direct descriptive fit to all finite pairs in the "
                "authoritative CSV — a population deliberately wider than the "
                "governed overlay snapshot above. It pools heterogeneous studies "
                "and treatments, so it is not a causal N effect, recommendation, "
                "or universal curve.\n\n"
                + _figures_note(figure_paths, "raw_finite_population_sensitivity")
            )
    modifier_eligible = int(
        factors.support["modifier_status"]
        .eq("eligible_for_prespecified_review_not_run")
        .sum()
    )
    fitted_screens = int(factors.additive_screen["screen_status"].eq("fitted").sum())
    source_verification = population.source_release_verification_status
    source_verification_note = (
        "The repository's current strict source-release verifier passed."
        if source_verification == "verified"
        else "The strict source-release verifier failed and the failure was recorded; "
        "current package members still passed the recipe's pinned-hash and checksum "
        "checks. Treat the overlay analysis as an internal snapshot diagnostic."
    )

    zero_series = 0
    total_series = int(len(descriptive.series_slopes))
    domain_line = (
        f"- Observed N domain: {float(linear['n_min_kg_ha']):.0f} to "
        f"{float(linear['n_max_kg_ha']):.0f} kg N ha⁻¹ across "
        f"{int(linear['distinct_n_levels'])} distinct levels"
    )
    if "has_zero_n" in descriptive.series_slopes.columns:
        zero_series = int(descriptive.series_slopes["has_zero_n"].sum())
        domain_line += (
            f"; {zero_series} of {total_series} series include a zero-N check"
        )
    domain_line += "\n"
    yield_line = (
        f"- Observed yield range: {float(linear['yield_min_t_ha']):.2f} to "
        f"{float(linear['yield_max_t_ha']):.2f} t ha⁻¹\n"
    )
    intercept_caveat = (
        f"Only {zero_series} of {total_series} series include a zero-N "
        "member, so the intercept extrapolates below the tested range for "
        "the remaining series and is not an observed unfertilized yield.\n\n"
        if "has_zero_n" in descriptive.series_slopes.columns
        and zero_series < total_series
        else ""
    )

    turning = quadratic["turning_point_n_kg_ha"]
    if pd.notna(turning):
        n_max = float(quadratic["n_max_kg_ha"])
        inside = bool(quadratic["turning_point_in_observed_domain"])
        turning_value = float(turning)
        if inside:
            location_note = "inside the observed N domain"
        else:
            beyond = (
                f", {100 * (turning_value - n_max) / n_max:.0f}% beyond the "
                f"highest observed rate ({n_max:.0f} kg N ha⁻¹)"
                if n_max > 0 and turning_value > n_max
                else " and outside the observed N domain"
            )
            location_note = (
                f"an extrapolation of the fitted parabola{beyond}, "
                "not an observed feature of the data"
            )
        turning_point_text = (
            f"Turning point = {turning_value:.1f} kg N ha⁻¹ — {location_note}; "
            f"inside observed domain = {inside}. No optimum is inferred.\n\n"
        )
    else:
        turning_point_text = (
            "The fitted curvature is degenerate, so no turning point exists. "
            "No optimum is inferred.\n\n"
        )

    mixed_model_sentence = ""
    fixed_slope = mixed_model.get("fixed_slope_t_ha_per_100_kg_n_ha")
    random_slope_sd = mixed_model.get("random_slope_sd_t_ha_per_100_kg_n_ha")
    if isinstance(fixed_slope, (int, float)) and math.isfinite(float(fixed_slope)):
        mixed_model_sentence = (
            f"The random-intercept, random-slope fit estimates a mean slope of "
            f"{float(fixed_slope):.2f} t ha⁻¹ per 100 kg N ha⁻¹"
        )
        if isinstance(random_slope_sd, (int, float)) and math.isfinite(
            float(random_slope_sd)
        ):
            mixed_model_sentence += (
                f" with a between-series slope SD of {float(random_slope_sd):.2f} "
                "t ha⁻¹ per 100 kg N ha⁻¹ — series genuinely differ in response, "
                "not just in baseline. "
            )
        else:
            mixed_model_sentence += ". "

    bootstrap_slope_sentence, bootstrap_turning_sentence = _bootstrap_sentences(
        bootstrap_uncertainty
    )
    uncertainty_parts = "".join(
        part
        for part in (
            bootstrap_slope_sentence,
            _theil_sentence(descriptive),
            _loo_sentence(descriptive),
        )
        if part
    ).rstrip()
    uncertainty_block = f"{uncertainty_parts}\n\n" if uncertainty_parts else ""

    return (
        "# Grain-yield response diagnostics\n\n"
        f"Status: `{BUNDLE_STATUS}`. This is an internal descriptive diagnostic, "
        "not a release certification or fertilizer recommendation.\n\n"
        + "\n".join(
            _key_findings_lines(
                linear,
                quadratic,
                heterogeneity,
                factors,
                mixed_model,
                bootstrap_uncertainty,
                redundancy_audit,
            )
        )
        + "\n"
        + "## Manifest-declared observed-series snapshot\n\n"
        f"- Observations: {len(population.frame)}\n"
        f"- Response series: {len(population.series_uids)}\n"
        f"- Studies: {population.study_count}\n"
        f"- Trials: {population.trial_count}\n"
        + domain_line
        + yield_line
        + f"- Strict source-release verification: `{source_verification}`\n\n"
        f"{source_verification_note}\n\n"
        "## Overlay-snapshot pooled equation\n\n"
        f"`{linear['equation']}`\n\n"
        f"R² = {float(linear['r_squared']):.4f}; "
        f"RMSE = {float(linear['rmse_t_ha']):.4f} t ha⁻¹. The slope is "
        f"{float(linear['slope_t_ha_per_kg_n_ha']) * 100:.2f} t ha⁻¹ per "
        f"100 kg N ha⁻¹. {_slope_confidence_sentence(linear)}\n\n"
        + uncertainty_block
        + intercept_caveat
        + "The pooled intercept and slope combine distinct studies and response "
        "series. They are not causal effects or universal agronomic parameters.\n\n"
        + _figures_note(
            figure_paths, "pooled_linear_response", "series_response_overlay"
        )
        + "## Heterogeneity\n\n"
        f"Within-series slope = "
        f"{float(heterogeneity.summary['within_series_slope_t_ha_per_kg_n_ha']):.5f} "
        "t ha⁻¹ per kg N ha⁻¹ "
        f"({float(heterogeneity.summary['within_series_slope_t_ha_per_kg_n_ha']) * 100:.2f} "
        "t ha⁻¹ per 100 kg N ha⁻¹), smaller than the pooled slope: part of the "
        "pooled association is between-series baseline structure, not "
        "within-series response. Series fixed intercepts reduce pooled residual "
        f"SSE by {100 * float(heterogeneity.summary['series_fixed_intercept_sse_reduction_fraction']):.2f}%.\n\n"
        + _zero_n_delta_sentence(descriptive)
        + "The large baseline-structure reduction is evidence that a single pooled "
        "line suppresses important between-series differences.\n\n"
        + _figures_note(
            figure_paths,
            "within_series_response",
            "series_slope_distribution",
        )
        + "## Pooled quadratic sensitivity\n\n"
        f"`{quadratic['equation']}`\n\n"
        + turning_point_text
        + bootstrap_turning_sentence
        + _figures_note(figure_paths, "pooled_quadratic_sensitivity")
        + "## Hierarchical diagnostic\n\n"
        f"R mixed-model status: `{mixed_model.get('status', 'not_run')}`. "
        f"Singular fit: `{mixed_model.get('singular', 'not_available')}`. "
        + mixed_model_sentence
        + "See `tables/heterogeneity/mixed_model_summary.csv` for the complete "
        "diagnostic record.\n\n"
        "## Candidate-factor support\n\n"
        f"Complete-case additive baseline screens fitted: {fitted_screens}; factors "
        f"with enough within-series support for a future prespecified modifier review: "
        f"{modifier_eligible}. No N×factor modifier effect was estimated here. "
        "Factor rows differ in sample size and degrees of freedom and must not be "
        "read as causal effect rankings. See `tables/factors/factor_support.csv` "
        "and `tables/factors/factor_additive_screen.csv`.\n\n"
        + _factors_beyond_nitrogen_markdown(
            factors,
            series_covariates,
            series_adjusted_screen,
            redundancy_audit,
            evidence_audit,
            mixed_model,
        )
        + "\n"
        + _figures_note(
            figure_paths, "factor_baseline_screen", "series_slope_vs_check_yield"
        )
        + raw_block
        + "\n".join(_bundle_contents_lines(table_paths, figure_paths))
    )


def write_diagnostic_bundle(
    root: str | Path,
    *,
    config: GrainYieldResponseConfig,
    population: GovernedPopulation,
    descriptive: DescriptiveResults,
    heterogeneity: HeterogeneityResults,
    factors: FactorSupportResults,
    mixed_model: Mapping[str, Any],
    raw_sensitivity: Mapping[str, Any] | None,
    implementation_sha256: Mapping[str, str],
    series_covariates: SeriesCovariateResults | None = None,
    series_adjusted_screen: pd.DataFrame | None = None,
    redundancy_audit: pd.DataFrame | None = None,
    evidence_audit: pd.DataFrame | None = None,
    bootstrap_uncertainty: pd.DataFrame | None = None,
) -> WrittenDiagnosticBundle:
    target = Path(root).resolve()
    if target.exists() and any(target.iterdir()):
        raise DiagnosticBundleError(f"Diagnostic staging root is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []

    tables: dict[str, pd.DataFrame] = {
        "analysis_population": population.frame,
        "pooled_models": descriptive.pooled_models,
        "model_grid": descriptive.model_grid,
        "series_slopes": descriptive.series_slopes,
        "leave_one_series_out": descriptive.leave_one_series_out,
        "zero_n_deltas": descriptive.zero_n_deltas,
        "zero_n_delta_summary": pd.DataFrame(
            [descriptive.zero_n_delta_summary]
        ),
        "heterogeneity_summary": pd.DataFrame([heterogeneity.summary]),
        "heterogeneity_decomposition": heterogeneity.decomposition,
        "series_fixed_intercepts": heterogeneity.series_intercepts,
        "factor_support": factors.support,
        "factor_level_summary": factors.level_summary,
        "factor_additive_screen": factors.additive_screen,
        "mixed_model_summary": pd.DataFrame(
            [
                {
                    key: (
                        json.dumps(_jsonable(value), sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in mixed_model.items()
                }
            ]
        ),
    }
    if series_covariates is not None:
        tables["series_covariates"] = series_covariates.series_frame
        tables["series_slope_modifier_screen"] = series_covariates.modifier_screen
    if series_adjusted_screen is not None:
        tables["factor_series_adjusted_screen"] = series_adjusted_screen
    if redundancy_audit is not None:
        tables["factor_redundancy_audit"] = redundancy_audit
    if evidence_audit is not None:
        tables["factor_evidence_audit"] = evidence_audit
    if bootstrap_uncertainty is not None:
        tables["bootstrap_uncertainty"] = bootstrap_uncertainty
    if raw_sensitivity is not None:
        summary = raw_sensitivity.get("summary")
        if isinstance(summary, pd.DataFrame):
            tables["raw_finite_sensitivity"] = summary
    for name, frame in tables.items():
        artifacts.append(_write_csv(target, table_relative_path(name), frame))

    figure_specs = [
        (
            "pooled_linear_response",
            _plot_pooled_linear(config, population, descriptive),
        ),
        (
            "pooled_quadratic_sensitivity",
            _plot_quadratic(config, population, descriptive),
        ),
        (
            "series_response_overlay",
            _plot_series_overlay(config, population, descriptive),
        ),
        ("within_series_response", _plot_within_series(config, heterogeneity)),
        ("series_slope_distribution", _plot_series_slopes(config, descriptive)),
        ("factor_baseline_screen", _plot_factor_screen(config, factors)),
    ]
    if series_covariates is not None:
        figure_specs.append(
            (
                "series_slope_vs_check_yield",
                _plot_series_modifiers(config, series_covariates),
            )
        )
    if raw_sensitivity is not None:
        figure_specs.append(
            (
                "raw_finite_population_sensitivity",
                _plot_raw_sensitivity(config, raw_sensitivity),
            )
        )
    # Every configured figure format is emitted; one rendered figure is saved
    # once per format before being closed.
    for name, figure in figure_specs:
        for extension in config.figure_formats:
            artifacts.append(
                _save_figure(
                    target,
                    figure_relative_path(name, extension),
                    figure,
                    dpi=config.figure_dpi,
                )
            )
        plt.close(figure)

    table_paths = tuple(sorted(table_relative_path(name) for name in tables))
    figure_paths = tuple(
        sorted(
            figure_relative_path(name, extension)
            for name, _ in figure_specs
            for extension in config.figure_formats
        )
    )
    summary_path = target / "summary.md"
    summary_path.write_text(
        _summary_markdown(
            population,
            descriptive,
            heterogeneity,
            factors,
            series_covariates,
            series_adjusted_screen,
            redundancy_audit,
            evidence_audit,
            mixed_model,
            raw_sensitivity,
            bootstrap_uncertainty=bootstrap_uncertainty,
            table_paths=table_paths,
            figure_paths=figure_paths,
        ),
        encoding="utf-8",
        newline="\n",
    )
    artifacts.append(
        {
            "relative_path": "summary.md",
            "artifact_kind": "report",
            "media_type": "text/markdown",
            "sha256": sha256_file(summary_path),
            "bytes": summary_path.stat().st_size,
        }
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": BUNDLE_STATUS,
        "run_id": f"grain_yield_response_{config.mode}",
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "analysis_role": "descriptive_diagnostic_not_causal_not_for_release",
        "source_release": {
            "run_id": population.source_run_id,
            "status": population.source_package_status,
            "strict_verification_status": (
                population.source_release_verification_status
            ),
            "strict_verification_detail": (
                population.source_release_verification_detail
            ),
            "package_path": str(
                config.release_package.relative_to(config.project_root)
            ),
        },
        "analysis_population": {
            "source_name": config.source_name,
            "population_id": config.analysis_population,
            "observations": len(population.frame),
            "series_count": len(population.series_uids),
            "study_count": population.study_count,
            "trial_count": population.trial_count,
            "series_uids": list(population.series_uids),
        },
        "raw_finite_population": (
            {
                "status": "completed",
                "population_id": "raw_finite_inventory_sensitivity",
                "source_nonblank_rows": raw_sensitivity.get(
                    "source_nonblank_rows"
                ),
                "finite_pair_rows": len(raw_sensitivity["frame"]),
                "excluded_nonfinite_pairs": raw_sensitivity.get(
                    "excluded_nonfinite_pairs"
                ),
                "yield_t_source_count": raw_sensitivity.get(
                    "yield_t_source_count"
                ),
                "yield_kg_fallback_count": raw_sensitivity.get(
                    "yield_kg_fallback_count"
                ),
            }
            if raw_sensitivity is not None
            and isinstance(raw_sensitivity.get("frame"), pd.DataFrame)
            else {"status": "disabled"}
        ),
        "input_sha256": {
            "recipe_config": sha256_file(config.config_path),
            **population.input_sha256,
            **dict(config.expected_inputs),
        },
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "mixed_model_status": mixed_model.get("status", "not_run"),
        "mixed_model_singular": mixed_model.get("singular"),
        "factor_modifier_policy": "support_audit_only_no_modifier_estimation",
        "artifacts": sorted(artifacts, key=lambda item: item["relative_path"]),
    }
    manifest_path = target / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    checksum_members = sorted(
        path
        for path in target.rglob("*")
        if path.is_file() and path.name != CHECKSUMS_NAME
    )
    checksums_path = target / CHECKSUMS_NAME
    checksums_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(target).as_posix()}\n"
            for path in checksum_members
        ),
        encoding="utf-8",
        newline="\n",
    )
    verified = verify_diagnostic_bundle(target)
    return WrittenDiagnosticBundle(
        root=target,
        manifest_path=manifest_path,
        checksums_path=checksums_path,
        artifact_count=verified.artifact_count,
    )


def _strict_manifest(path: Path) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DiagnosticBundleError(f"Duplicate manifest key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    if not isinstance(value, dict):
        raise DiagnosticBundleError("Diagnostic manifest must be an object")
    return value


def _parse_checksums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise DiagnosticBundleError(
                f"Malformed checksum line {line_number}"
            ) from exc
        pure = PurePosixPath(relative)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or pure.is_absolute()
            or ".." in pure.parts
            or str(pure) != relative
            or relative in entries
        ):
            raise DiagnosticBundleError(
                f"Unsafe checksum entry on line {line_number}"
            )
        entries[relative] = digest
    return entries


def verify_diagnostic_bundle(root: str | Path) -> VerifiedDiagnosticBundle:
    target = Path(root).resolve()
    if target.is_symlink() or not target.is_dir():
        raise DiagnosticBundleError(f"Diagnostic bundle is missing: {target}")
    manifest_path = target / MANIFEST_NAME
    checksums_path = target / CHECKSUMS_NAME
    for path in (manifest_path, checksums_path):
        if path.is_symlink() or not path.is_file():
            raise DiagnosticBundleError(f"Required bundle artifact is missing: {path}")
    manifest = _strict_manifest(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DiagnosticBundleError("Unsupported diagnostic manifest schema")
    if manifest.get("status") != BUNDLE_STATUS:
        raise DiagnosticBundleError("Unsupported diagnostic bundle status")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise DiagnosticBundleError("Diagnostic manifest artifact inventory is empty")
    inventory: dict[str, dict[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("relative_path"), str):
            raise DiagnosticBundleError("Diagnostic artifact declaration is invalid")
        relative = item["relative_path"]
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
            raise DiagnosticBundleError(f"Unsafe diagnostic artifact path: {relative}")
        if relative in inventory:
            raise DiagnosticBundleError(f"Duplicate diagnostic artifact: {relative}")
        inventory[relative] = item

    actual_files: set[str] = set()
    for path in target.rglob("*"):
        if path.is_symlink():
            raise DiagnosticBundleError(f"Diagnostic bundle contains a symlink: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(target).as_posix())
    expected_files = set(inventory) | {MANIFEST_NAME, CHECKSUMS_NAME}
    if actual_files != expected_files:
        raise DiagnosticBundleError(
            f"Diagnostic bundle inventory mismatch: {sorted(actual_files ^ expected_files)}"
        )

    checksums = _parse_checksums(checksums_path)
    if set(checksums) != expected_files - {CHECKSUMS_NAME}:
        raise DiagnosticBundleError("Diagnostic checksum inventory is incomplete")
    for relative, digest in checksums.items():
        if sha256_file(target / relative) != digest:
            raise DiagnosticBundleError(f"Diagnostic checksum mismatch: {relative}")
    for relative, item in inventory.items():
        path = target / relative
        if sha256_file(path) != item.get("sha256"):
            raise DiagnosticBundleError(f"Manifest hash mismatch: {relative}")
        if path.stat().st_size != item.get("bytes"):
            raise DiagnosticBundleError(f"Manifest byte count mismatch: {relative}")
        kind = item.get("artifact_kind")
        if kind == "table":
            try:
                frame = pd.read_csv(path)
            except Exception as exc:
                raise DiagnosticBundleError(
                    f"Table is unreadable: {relative}"
                ) from exc
            if len(frame) != item.get("rows") or list(frame.columns) != item.get(
                "columns"
            ):
                raise DiagnosticBundleError(f"Table semantic mismatch: {relative}")
        elif kind == "figure":
            with Image.open(path) as image:
                image.load()
                if (
                    image.size
                    != (item.get("width_pixels"), item.get("height_pixels"))
                    or image.mode != item.get("color_mode")
                    or image.format != item.get("detected_format")
                ):
                    raise DiagnosticBundleError(f"Figure semantic mismatch: {relative}")
        elif kind != "report":
            raise DiagnosticBundleError(f"Unknown diagnostic artifact kind: {kind}")

    table_count = sum(
        item.get("artifact_kind") == "table" for item in inventory.values()
    )
    figure_count = sum(
        item.get("artifact_kind") == "figure" for item in inventory.values()
    )
    return VerifiedDiagnosticBundle(
        root=target,
        status=str(manifest["status"]),
        table_count=table_count,
        figure_count=figure_count,
        artifact_count=len(inventory),
    )
