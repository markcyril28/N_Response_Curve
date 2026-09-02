from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

from ..analysis.grain_yield_response.config import GrainYieldResponseConfig
from ..analysis.grain_yield_response.descriptive import DescriptiveResults
from ..analysis.grain_yield_response.factor_support import FactorSupportResults
from ..analysis.grain_yield_response.heterogeneity import HeterogeneityResults
from ..analysis.grain_yield_response.population import GovernedPopulation
from ..analysis.grain_yield_response.series_covariates import SeriesCovariateResults
from ..data.provenance import sha256_file


MANIFEST_NAME = "run_manifest.json"
CHECKSUMS_NAME = "CHECKSUMS.sha256"
SCHEMA_VERSION = "grain-yield-response-diagnostics-v3"
INTERMEDIATE_SCHEMA_VERSION = "grain-yield-response-diagnostics-v2"
LEGACY_SCHEMA_VERSION = "grain-yield-response-diagnostics-v1"
BUNDLE_STATUS = "diagnostic_internal_not_release"
GRAIN_YIELD_FACTOR_CONTRIBUTOR_ROOT = PurePosixPath(
    "factors/grain_yield_factor_contributors"
)
RESPONSE_CURVE_FACTOR_CONTRIBUTOR_ROOT = PurePosixPath(
    "factors/response_curve_factor_contributors"
)
PRIORITIZED_RESPONSE_MODIFIER_ROOT = PurePosixPath(
    RESPONSE_CURVE_FACTOR_CONTRIBUTOR_ROOT / "response_curve_modifier_by_dataset"
)
INTERMEDIATE_PRIORITIZED_RESPONSE_MODIFIER_ROOT = PurePosixPath(
    "factors/response_curve_modifier_by_dataset"
)
LEGACY_PRIORITIZED_RESPONSE_MODIFIER_ROOT = PurePosixPath(
    "figures/factors/response_curve_modifier_by_dataset"
)
_RESPONSE_MODIFIER_SCHEMA_VERSION = "response-curve-modifier-by-dataset-v3"
_INTERMEDIATE_RESPONSE_MODIFIER_SCHEMA_VERSION = (
    "response-curve-modifier-by-dataset-v2"
)
_LEGACY_RESPONSE_MODIFIER_SCHEMA_VERSION = "response-curve-modifier-by-dataset-v1"
_RESPONSE_MODIFIER_PLACEMENT_MODE = "canonical_full_bundle_in_host_subtree"
_RESPONSE_MODIFIER_DATASET_GROUPS = frozenset(
    {"core_trial_data", "ltcce", "ph_combined_nopt_rcm", "all_datasets"}
)


class DiagnosticBundleError(ValueError):
    """Raised when a diagnostic bundle is incomplete or unsafe."""


@contextmanager
def diagnostic_container_publication_lock(destination: Path):
    """Serialize every producer that can replace a diagnostic container."""
    lock_root = Path(tempfile.gettempdir()) / "n_response_curve_publication_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256(
        str(destination.resolve()).encode("utf-8")
    ).hexdigest()[:24]
    lock_path = lock_root / f"diagnostic-container-{token}.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DiagnosticBundleError(
                "Another diagnostic-container publication is already running for "
                f"{destination}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def exchange_directories(left: Path, right: Path) -> bool:
    """Atomically exchange same-filesystem directories when Linux supports it."""
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        return False
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(left),
        -100,
        os.fsencode(right),
        2,
    )
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number in {
        errno.ENOSYS,
        errno.EINVAL,
        errno.EOPNOTSUPP,
        errno.EXDEV,
    }:
        return False
    raise OSError(error_number, os.strerror(error_number), str(left), str(right))


# Every bundle artifact lives in one shared themed subdirectory. Figures and
# tables are intentionally mixed by subject rather than split into top-level
# type buckets. Registration here is mandatory so the layout cannot drift.
_TABLE_GROUPS: dict[str, str] = {
    "factor_support": "factors",
    "factor_level_summary": "factors",
    "factor_additive_screen": "factors",
    "factor_series_adjusted_screen": "factors",
    "factor_redundancy_audit": "factors",
    "factor_evidence_audit": "factors",
    "series_slope_modifier_screen": "factors",
}

_FIGURE_GROUPS: dict[str, str] = {
    "factor_baseline_screen": GRAIN_YIELD_FACTOR_CONTRIBUTOR_ROOT.as_posix(),
    "factor_explanatory_ranking": GRAIN_YIELD_FACTOR_CONTRIBUTOR_ROOT.as_posix(),
    "series_slope_vs_check_yield": (
        RESPONSE_CURVE_FACTOR_CONTRIBUTOR_ROOT.as_posix()
    ),
    "response_curve_modifier_ranking": (
        RESPONSE_CURVE_FACTOR_CONTRIBUTOR_ROOT.as_posix()
    ),
}

# Groups this recipe published before the population/pooled/heterogeneity
# subdirectories were retired. The writer never emits these names again; the
# maps exist so a bundle written under an earlier revision still resolves its
# own manifest paths and stays verifiable — including the one on disk that the
# replaceable-container check reads just before it is overwritten.
_RETIRED_TABLE_GROUPS: dict[str, str] = {
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
}

_RETIRED_FIGURE_GROUPS: dict[str, str] = {
    "pooled_linear_response": "pooled",
    "pooled_quadratic_sensitivity": "pooled",
    "raw_finite_population_sensitivity": "pooled",
    "series_response_overlay": "heterogeneity",
    "within_series_response": "heterogeneity",
    "series_slope_distribution": "heterogeneity",
}

_TABLE_DESCRIPTIONS: dict[str, str] = {
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
    "factor_baseline_screen": "complete-case factor screen gains",
    "series_slope_vs_check_yield": "series slope versus zero-N check yield",
    "factor_explanatory_ranking": (
        "grouping levels and named factors ranked on one common metric, beside "
        "the same factors held against series identity"
    ),
    "response_curve_modifier_ranking": (
        "series-level covariates ranked against the fitted series slopes, "
        "Pearson beside Spearman"
    ),
}

# A prose explainer that sits beside the figure it explains. It is a `report`
# artifact, not a figure: the bundle verifier's exact-inventory check means an
# unregistered .md dropped next to a .jpeg both fails verification and blocks
# the next overwrite run, so the only way to keep a caption alongside its
# figure is to generate and hash it here.
_EXPLAINER_PATHS: dict[str, str] = {
    "response_curve_modifier_ranking": (
        RESPONSE_CURVE_FACTOR_CONTRIBUTOR_ROOT
        / "response_curve_modifier_ranking.md"
    ).as_posix(),
}
# Keyed by filename, not stem, because the explainer shares its stem with the
# figure it documents.
_EXPLAINER_DESCRIPTIONS: dict[str, str] = {
    "response_curve_modifier_ranking.md": (
        "how to read the figure beside it, and what each screened covariate "
        "turns out to be"
    ),
}


def table_relative_path(name: str) -> str:
    group = _TABLE_GROUPS.get(name)
    if group is None:
        raise DiagnosticBundleError(
            f"Table {name!r} has no registered bundle group"
        )
    return f"{group}/{name}.csv"


def figure_relative_path(name: str, extension: str) -> str:
    group = _FIGURE_GROUPS.get(name)
    if group is None:
        raise DiagnosticBundleError(
            f"Figure {name!r} has no registered bundle group"
        )
    return f"{group}/{name}.{extension}"


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


_RANKING_LEVEL_COLOR = "#2a78d6"
_RANKING_FACTOR_COLOR = "#eb6834"
_RANKING_INK_SOFT = "#52514e"

# Aliased factors collapse to one row: the redundancy audit reports Cramér's V =
# 1.0 for region/province/organic and r = -1.0 for the P and K rates, so drawing
# them separately would show one contrast several times over. The survivor is the
# name the audit reports first; the label records the whole alias group.
_RANKING_ALIAS_LABEL: dict[str, str] = {
    "region_normalized": "Region = Province\n= Organic fert.",
    "rice_variety_normalized": "Rice variety",
    "planting_year": "Planting year",
    "biofertilizer_present": "Biofertilizer present",
    "season_normalized": "Season (dry/wet)",
}
_RANKING_ALIAS_DROP = frozenset(
    {"province_normalized", "organic_fertilizer_present", "k_rate_kg_k2o_ha"}
)
# Held below the rule rather than ranked. P/K is fitted on a 39-row complete-case
# subset, so its denominator differs from every other bar and an ordinal position
# would assert a comparison the data does not support; water regime is a single
# level, and a zero-length bar at the foot of a ranking reads as "explains
# nothing" when the truth is that it is not estimable at all.
_RANKING_DETACHED: dict[str, str] = {
    "p_rate_kg_p2o5_ha": "P rate = K rate",
    "water_regime_normalized": "Water regime",
}
_RANKING_STRUCTURE_LABEL: dict[str, str] = {
    "series_fixed_intercepts": "Response series identity",
    "trial_fixed_intercepts": "Trial identity",
    "study_fixed_intercepts": "Study identity",
}


def _plot_factor_ranking(
    config: GrainYieldResponseConfig,
    heterogeneity: HeterogeneityResults,
    factors: FactorSupportResults,
    series_adjusted_screen: pd.DataFrame | None,
) -> plt.Figure:
    """Rank grouping levels and named factors on one common metric.

    The metric is the share of the pooled N-only model's residual sum of squares
    that the structure removes. For the complete-case n=74 factor rows this is
    ``partial_r_squared``, whose base model is exactly the pooled fit anchoring
    ``heterogeneity_decomposition``, so grouping levels and factors are directly
    comparable. The right panel repeats each factor against a baseline that
    already holds series identity, where all of them collapse to zero.
    """
    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=(config.figure_width_inches, config.figure_height_inches),
        gridspec_kw={"width_ratios": [2.4, 1.0], "wspace": 0.10},
    )
    figure.subplots_adjust(left=0.175, right=0.980, top=0.860, bottom=0.300)

    decomposition = heterogeneity.decomposition.set_index("structure")
    screen = factors.additive_screen.set_index("factor")
    adjusted = (
        series_adjusted_screen.set_index("factor")
        if series_adjusted_screen is not None
        else None
    )

    ranked: list[tuple[str, float, int, str, bool, str]] = []
    for key, label in _RANKING_STRUCTURE_LABEL.items():
        if key not in decomposition.index:
            continue
        row = decomposition.loc[key]
        ranked.append(
            (
                label,
                float(row["sse_reduction_fraction_vs_pooled"]),
                int(row["group_count"]) - 1,
                "level",
                False,
                "",
            )
        )
    for key, label in _RANKING_ALIAS_LABEL.items():
        if key not in screen.index:
            continue
        row = screen.loc[key]
        if str(row.get("screen_status")) != "fitted":
            continue
        flagged = key == "biofertilizer_present"
        ranked.append(
            (
                label,
                float(row["partial_r_squared"]),
                int(float(row["factor_parameters_added"])),
                "factor",
                flagged,
                "no raw evidence" if flagged else "",
            )
        )
    ranked.sort(key=lambda item: item[1], reverse=True)

    detached: list[tuple[str, float | None, int, str, bool, str]] = []
    for key, label in _RANKING_DETACHED.items():
        if key not in screen.index:
            continue
        row = screen.loc[key]
        if str(row.get("screen_status")) != "fitted":
            detached.append((label, None, 0, "none", False, ""))
            continue
        detached.append(
            (
                label,
                float(row["partial_r_squared"]),
                int(float(row["factor_parameters_added"])),
                "factor",
                True,
                f"fitted on {int(row['observations'])} rows",
            )
        )

    rows = ranked + detached
    positions = np.arange(len(rows))[::-1]
    split_y = len(detached) - 0.5
    face = {"level": _RANKING_LEVEL_COLOR, "factor": _RANKING_FACTOR_COLOR}

    for position, (label, value, degrees, kind, texture, note) in zip(positions, rows):
        if value is None:
            left.text(
                0.006,
                position,
                "not estimable — a scope limit, not a null result",
                va="center",
                ha="left",
                fontsize=7.5,
                color=_RANKING_INK_SOFT,
                style="italic",
            )
            continue
        left.barh(
            position,
            value,
            height=0.66,
            color=face[kind],
            hatch="///" if texture else None,
            edgecolor="white" if texture else "none",
            linewidth=0.0,
        )
        annotation = f"+df={degrees}" if not note else f"+df={degrees}  ·  {note}"
        left.annotate(
            annotation,
            xy=(value, position),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.5,
            color=_RANKING_INK_SOFT,
        )

    for axis in (left, right):
        axis.axhline(split_y, color="#b8b7b2", linewidth=1.0, linestyle=(0, (5, 3)))
    left.text(
        0.79,
        split_y + 0.16,
        "not ranked below this line — not comparable on this metric",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=_RANKING_INK_SOFT,
        style="italic",
    )
    left.set_yticks(positions)
    left.set_yticklabels([row[0] for row in rows], fontsize=8.5)
    left.set_xlim(0.0, 0.80)
    left.set_ylim(-0.75, len(rows) - 0.3)
    left.set(
        title="On its own, alongside N rate",
        xlabel="Share of the N-only model's residual variation removed",
    )
    left.grid(axis="x", alpha=0.2)
    left.set_axisbelow(True)
    for spine in ("top", "right"):
        left.spines[spine].set_visible(False)

    right.axvspan(0.0, 0.80, color="#f0efec", alpha=0.55)
    for position, (label, value, degrees, kind, texture, note) in zip(positions, rows):
        if value is None:
            continue
        if kind == "level":
            text = (
                "is the baseline"
                if label.startswith("Response series")
                else "nested in series"
            )
            right.text(
                0.035,
                position,
                text,
                va="center",
                ha="left",
                fontsize=7.5,
                color=_RANKING_INK_SOFT,
                style="italic",
            )
            continue
        # A round dot, never a bar-like tick: this is a point estimate at exactly
        # zero and must not read as a very short bar at smaller display sizes.
        right.plot(
            [0.0], [position], marker="o", markersize=5.5, color=_RANKING_FACTOR_COLOR
        )
        adjusted_text = "0.000"
        if adjusted is not None:
            match = [
                key
                for key, alias in {**_RANKING_ALIAS_LABEL, **_RANKING_DETACHED}.items()
                if alias == label
            ]
            if match and match[0] in adjusted.index:
                adjusted_text = (
                    f"{float(adjusted.loc[match[0], 'series_adjusted_partial_r_squared']):.3f}"
                )
        right.annotate(
            adjusted_text,
            xy=(0.0, position),
            xytext=(9, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
        )

    right.set_yticks(positions)
    right.set_yticklabels([])
    right.set_xlim(0.0, 0.80)
    right.set_ylim(-0.75, len(rows) - 0.3)
    right.set_xticks([0.4, 0.8])
    right.set(title="After series identity is held", xlabel="Same scale")
    right.grid(axis="x", alpha=0.2)
    right.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        right.spines[spine].set_visible(False)

    figure.legend(
        handles=[
            Patch(facecolor=_RANKING_LEVEL_COLOR, label="Grouping level"),
            Patch(facecolor=_RANKING_FACTOR_COLOR, label="Named candidate factor"),
            Patch(
                facecolor=_RANKING_FACTOR_COLOR,
                edgecolor="white",
                hatch="///",
                label="Flagged — see caption",
            ),
        ],
        loc="center",
        bbox_to_anchor=(0.5, 0.205),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    figure.suptitle(
        "What explains grain-yield differences besides nitrogen rate", fontsize=12.5
    )
    figure.text(
        0.5,
        0.070,
        "Descriptive associations only; order is confounded with degrees of freedom, and rows are not an "
        "effect-size ranking. Every named factor is\nconstant within a response series, so all of them are "
        "absorbed once series identity is in the baseline. Aliased factors are collapsed to one row.",
        ha="center",
        va="top",
        fontsize=7.2,
        color=_RANKING_INK_SOFT,
        linespacing=1.5,
    )
    return figure


def _plot_slope_modifier_ranking(
    config: GrainYieldResponseConfig,
    series_covariates: SeriesCovariateResults,
    mixed_model: Mapping[str, Any],
) -> plt.Figure:
    """Rank series-level covariates against the fitted per-series slopes.

    Pearson and Spearman are drawn as a pair because their disagreement is the
    diagnostic: a large linear correlation with a near-zero rank correlation
    means one or two outlying series carry it rather than a monotone trend.
    """
    figure, axis = plt.subplots(
        figsize=(config.figure_width_inches, config.figure_height_inches)
    )
    figure.subplots_adjust(left=0.200, right=0.760, top=0.855, bottom=0.245)

    screen = series_covariates.modifier_screen
    screen = screen.loc[screen["screen_status"].astype(str).eq("fitted"), :].copy()
    for column in ("pearson_r", "spearman_rho"):
        screen[column] = pd.to_numeric(screen[column], errors="coerce")
    screen = screen.loc[screen["pearson_r"].notna(), :]
    # Observation count and distinct N-level count are identical columns in this
    # population, so they are one screen reported twice.
    screen = screen.drop_duplicates(subset=["pearson_r", "spearman_rho", "series_used"])
    screen = screen.reindex(
        screen["pearson_r"].abs().sort_values(ascending=True).index
    )

    positions = np.arange(len(screen))
    correlation = mixed_model.get("random_intercept_slope_correlation")
    axis.axvline(0.0, color=_RANKING_INK_SOFT, linewidth=0.9)
    if isinstance(correlation, (int, float)):
        axis.axvline(
            float(correlation), color="#b8b7b2", linewidth=1.2, linestyle=(0, (5, 3))
        )
        axis.annotate(
            f"lme4 baseline↔slope\ncorrelation, {float(correlation):.2f}",
            xy=(float(correlation), len(screen) - 0.52),
            ha="center",
            va="center",
            fontsize=7.2,
            color=_RANKING_INK_SOFT,
            style="italic",
            linespacing=1.4,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "none"},
        )

    for position, (_, row) in zip(positions, screen.iterrows()):
        pearson = float(row["pearson_r"])
        spearman = float(row["spearman_rho"])
        axis.plot(
            [pearson, spearman], [position, position], color="#c9c8c3", linewidth=1.6
        )
        axis.plot(
            [pearson],
            [position],
            marker="o",
            markersize=8,
            color=_RANKING_LEVEL_COLOR,
            markeredgecolor="white",
            markeredgewidth=1.3,
        )
        axis.plot(
            [spearman],
            [position],
            marker="o",
            markersize=8,
            color=_RANKING_FACTOR_COLOR,
            markeredgecolor="white",
            markeredgewidth=1.3,
        )
        note = f"n={int(row['series_used'])} series"
        if str(row.get("estimation_artefact_warning")):
            note += "  ·  shares points with the slope"
        axis.annotate(
            note,
            xy=(1.0, position),
            xycoords=("axes fraction", "data"),
            xytext=(9, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.2,
            color=_RANKING_INK_SOFT,
            annotation_clip=False,
        )

    axis.set_yticks(positions)
    axis.set_yticklabels(
        [_factor_label(str(value)) for value in screen["covariate"]], fontsize=8.2
    )
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-0.75, len(screen) - 0.25)
    axis.set(
        title="What might explain the differences between the response curves",
        xlabel="Correlation with the fitted per-series N-response slope",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)

    figure.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="none", markersize=8,
                color=_RANKING_LEVEL_COLOR, markeredgecolor="white",
                markeredgewidth=1.3, label="Pearson r (linear)",
            ),
            Line2D(
                [], [], marker="o", linestyle="none", markersize=8,
                color=_RANKING_FACTOR_COLOR, markeredgecolor="white",
                markeredgewidth=1.3, label="Spearman ρ (rank / monotone)",
            ),
        ],
        loc="center",
        bbox_to_anchor=(0.5, 0.150),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    figure.text(
        0.5,
        0.105,
        "Where the two dots disagree, the linear association is carried by a few outlying series rather than a "
        "monotone trend. Series-level associations only:\nnone of the configured agronomic factors can appear "
        "here, because all of them are constant within every response series and cannot bend an observed curve.",
        ha="center",
        va="top",
        fontsize=7.2,
        color=_RANKING_INK_SOFT,
        linespacing=1.5,
    )
    return figure


_SLOPE_COLUMN = "series_slope_t_ha_per_kg_n_ha"


def _critical_absolute_r(sample_size: int, alpha: float = 0.05) -> float | None:
    """Two-sided critical |r| for rho = 0 at `sample_size` pairs.

    An orientation aid for the explainer prose only. The bundle's own tables
    report r, rho, R² and residual degrees of freedom and deliberately no
    p-values, because `factor_modifier_policy` is support-audit-only.
    """
    degrees = int(sample_size) - 2
    if degrees < 1:
        return None
    quantile = float(stats.t.ppf(1.0 - alpha / 2.0, degrees))
    return quantile / math.sqrt(quantile * quantile + degrees)


def _pearson(frame: pd.DataFrame, covariate: str) -> float | None:
    pair = frame[[covariate, _SLOPE_COLUMN]].dropna()
    if len(pair) < 3 or pair[covariate].nunique() < 2:
        return None
    value = float(pair[covariate].corr(pair[_SLOPE_COLUMN]))
    return None if math.isnan(value) else value


def _drop_one_extreme(frame: pd.DataFrame, covariate: str) -> dict[str, Any] | None:
    """The single series whose removal moves Pearson r the furthest.

    This is the quantity the Pearson/Spearman pair in the figure only hints at:
    a rank-linear disagreement says leverage is present, and this says how much
    of the correlation one series is carrying.
    """
    pair = frame[["response_series_uid", covariate, _SLOPE_COLUMN]].dropna()
    baseline = _pearson(pair, covariate)
    if baseline is None or len(pair) < 5:
        return None
    worst_uid: str | None = None
    worst_r = baseline
    for uid in pair["response_series_uid"].astype(str):
        reduced = pair.loc[pair["response_series_uid"].astype(str) != uid]
        value = _pearson(reduced, covariate)
        if value is None:
            continue
        if abs(value - baseline) > abs(worst_r - baseline):
            worst_uid, worst_r = uid, value
    if worst_uid is None:
        return None
    return {
        "baseline": baseline,
        "series_uid": worst_uid,
        "reduced": worst_r,
        "sample_size": len(pair) - 1,
    }


def _largest_within_study_pearson(
    frame: pd.DataFrame, covariate: str, minimum_series: int = 4
) -> dict[str, Any] | None:
    """Pearson r inside the single study that contributes the most series.

    A between-study correlation on series-level covariates cannot be told apart
    from study identity; the same correlation recovered inside one study can.
    """
    pair = frame[["study_uid", covariate, _SLOPE_COLUMN]].dropna()
    best: dict[str, Any] | None = None
    for study_uid, block in pair.groupby("study_uid"):
        if len(block) < minimum_series:
            continue
        value = _pearson(block, covariate)
        if value is None:
            continue
        if best is None or len(block) > best["sample_size"]:
            best = {
                "study_uid": str(study_uid),
                "sample_size": len(block),
                "pearson_r": value,
            }
    return best


def _study_series_count(frame: pd.DataFrame, series_uid: str) -> int:
    """How many series the study owning `series_uid` contributes."""
    match = frame.loc[frame["response_series_uid"].astype(str) == str(series_uid)]
    if match.empty:
        return 0
    return int((frame["study_uid"] == match["study_uid"].iloc[0]).sum())


def _study_determined(frame: pd.DataFrame, covariates: tuple[str, ...]) -> tuple[str, ...]:
    """Covariates that never vary inside a study, i.e. study labels in disguise."""
    determined: list[str] = []
    for name in covariates:
        if name not in frame.columns:
            continue
        counts = frame.groupby("study_uid")[name].nunique(dropna=True)
        if not counts.empty and int(counts.max()) <= 1:
            determined.append(name)
    return tuple(determined)


def _modifier_ranking_explainer_markdown(
    population: GovernedPopulation,
    factors: FactorSupportResults,
    series_covariates: SeriesCovariateResults,
    heterogeneity: HeterogeneityResults,
    mixed_model: Mapping[str, Any],
    figure_paths: tuple[str, ...] = (),
) -> str:
    """Explain `response_curve_modifier_ranking` beside the figure itself.

    Every number below is recomputed from the same objects that draw the
    figure, so the prose cannot drift away from the panel it describes when the
    population changes.
    """
    screen = series_covariates.modifier_screen
    screen = screen.loc[screen["screen_status"].astype(str).eq("fitted"), :].copy()
    for column in ("pearson_r", "spearman_rho"):
        screen[column] = pd.to_numeric(screen[column], errors="coerce")
    screen = screen.loc[screen["pearson_r"].notna(), :]
    duplicated = screen.duplicated(subset=["pearson_r", "spearman_rho", "series_used"])
    collapsed = (
        screen.loc[duplicated, "covariate"].astype(str).tolist() if len(screen) else []
    )
    screen = screen.loc[~duplicated, :]
    screen = screen.reindex(
        screen["pearson_r"].abs().sort_values(ascending=False).index
    )

    frame = series_covariates.series_frame
    correlation = mixed_model.get("random_intercept_slope_correlation")
    slopes = pd.to_numeric(frame.get(_SLOPE_COLUMN), errors="coerce").dropna()

    rows: list[dict[str, Any]] = []
    for _, record in screen.iterrows():
        covariate = str(record["covariate"])
        sample_size = int(record["series_used"])
        threshold = _critical_absolute_r(sample_size)
        rows.append(
            {
                "covariate": covariate,
                "label": _factor_label(covariate),
                "pearson_r": float(record["pearson_r"]),
                "spearman_rho": float(record["spearman_rho"]),
                "sample_size": sample_size,
                "threshold": threshold,
                "clears": bool(
                    threshold is not None
                    and abs(float(record["pearson_r"])) >= threshold
                ),
                "shares_points": bool(
                    str(record.get("estimation_artefact_warning") or "").strip()
                ),
                "leverage": _drop_one_extreme(frame, covariate),
                "within_study": _largest_within_study_pearson(frame, covariate),
            }
        )

    study_labels = _study_determined(
        frame, tuple(row["covariate"] for row in rows)
    )
    clearing = [row for row in rows if row["clears"]]
    # Fragile means one series *is* the correlation: dropping it flips the sign
    # or halves the magnitude. A reduced r that merely slips under a critical
    # value computed at the smaller sample size does not qualify -- that is the
    # threshold moving, not the association dissolving.
    fragile = [
        row
        for row in clearing
        if row["leverage"] is not None
        and (
            row["leverage"]["reduced"] * row["leverage"]["baseline"] <= 0.0
            or abs(row["leverage"]["reduced"]) < 0.5 * abs(row["leverage"]["baseline"])
        )
    ]

    lines: list[str] = [
        "# What might explain the differences between the response curves",
        "",
        "Reading notes for `response_curve_modifier_ranking.jpeg`, in this same "
        f"directory. Both describe {len(population.frame)} observations from "
        f"{len(population.series_uids)} response series in "
        f"{population.study_count} studies; bundle status "
        f"`{BUNDLE_STATUS}`.",
        "",
        "The companion figure `factor_explanatory_ranking.jpeg` asks what "
        "explains how *high* a curve sits. This one asks what explains how "
        "*steeply* it rises — the two questions have different answers, and "
        "only the second one is on this page.",
        "",
        "## The short answer",
        "",
    ]

    slope_sentence = ""
    if not slopes.empty:
        negative = int((slopes < 0.0).sum())
        slope_sentence = (
            f"The {len(slopes)} fitted series slopes run from "
            f"{slopes.min() * 100.0:+.2f} to {slopes.max() * 100.0:+.2f} "
            "t ha⁻¹ per 100 kg N"
            + (
                f" ({negative} of them negative)"
                if negative
                else ""
            )
            + ", so there is real variation to explain. "
        )
    collapsed_note = (
        " (one further screen duplicated a column already shown and is folded in)"
        if len(collapsed) == 1
        else f" ({len(collapsed)} further screens duplicated columns already "
        "shown and are folded in)"
        if collapsed
        else ""
    )
    survivors = [row for row in clearing if row not in fragile]
    if fragile:
        fragile_sentence = (
            f"{len(fragile)} of those "
            + ("dissolves" if len(fragile) == 1 else "dissolve")
            + " when a single series is removed. "
        )
    elif clearing:
        fragile_sentence = "None of those collapse when a single series is removed. "
    else:
        fragile_sentence = ""
    if survivors:
        survivor_sentence = (
            "What survives both checks: "
            + ", ".join(
                f"**{row['label'].replace(chr(10), ' ')}**" for row in survivors
            )
            + ". Neither check tests circularity, though, and "
            + (
                "that one is measured from the same points as the slope it "
                "predicts"
                if all(row["shares_points"] for row in survivors)
                else "some of these are measured from the same points as the "
                "slopes they predict"
            )
            + " — read the qualifications below before using any of this."
        )
    else:
        survivor_sentence = (
            "**Nothing on this panel survives both tests**, so the figure's "
            "honest reading is that none of the recorded series attributes "
            "accounts for why the curves differ in steepness."
        )
    lines.append(
        slope_sentence
        + f"{len(rows)} series-level covariates are screened against those "
        f"slopes{collapsed_note}, and {len(clearing)} reach a correlation "
        "larger than sampling noise at its own sample size. "
        + fragile_sentence
        + survivor_sentence
    )
    lines.append("")
    lines.append(
        "**The figure does not rank causes of curve shape. It ranks how "
        "strongly bookkeeping attributes of a series happen to track that "
        "series' fitted slope.** No N × factor interaction was estimated "
        "anywhere in this bundle "
        "(`factor_modifier_policy = support_audit_only_no_modifier_estimation` "
        "in `run_manifest.json`)."
    )
    lines.append("")

    lines += [
        "## What is plotted",
        "",
        "One row per screened covariate, sorted by |Pearson r| — strongest at "
        "the top of the printed figure. Two dots per row:",
        "",
        "- **Blue, Pearson r** — the straight-line association between the "
        "covariate and the fitted slope.",
        "- **Orange, Spearman ρ** — the same association on ranks, which is "
        "blind to how extreme any single value is.",
        "",
        "The grey connector is the gap between them, and that gap is the point "
        "of drawing both. **A long connector means the linear correlation is "
        "carried by a few unusual series rather than by a trend that holds "
        "across the whole set.** Agreement means the association is monotone.",
        "",
        "Each series contributes exactly one point per row, unweighted "
        "(`weighting = unweighted_each_series_counts_once`). The right margin "
        "gives the number of series that had the covariate recorded, which is "
        "not the same for every row.",
        "",
    ]

    if isinstance(correlation, (int, float)):
        check_row = next(
            (row for row in rows if row["shares_points"]), None
        )
        agreement = ""
        if check_row is not None:
            agreement = (
                " It lands close to the "
                f"`{check_row['covariate']}` screen "
                f"({check_row['pearson_r']:+.2f}), which is the crude "
                "two-column version of the same trade-off — mutual support, "
                "not independent confirmation, because both describe the same "
                "series."
            )
        lines += [
            "The dashed vertical rule is not one of the screens. It is the "
            "random-effect intercept↔slope correlation "
            f"({float(correlation):+.2f}) from the lme4 model, "
            f"estimated from all {int(mixed_model.get('observations') or 0)} "
            "observations at "
            "once: series that start high gain less per kilogram of N."
            + agreement,
            "",
        ]

    lines += [
        "## Every row, and what it turns out to be",
        "",
        "Ordered as the figure orders them, strongest linear correlation first.",
        "",
        "| # | Covariate | Pearson r | Spearman ρ | Series | Noise floor | Clears it |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, row in enumerate(rows, start=1):
        floor = "—" if row["threshold"] is None else f"{row['threshold']:.2f}"
        label = row["label"].replace("\n", " ")
        lines.append(
            f"| {index} | {label} — `{row['covariate']}` "
            f"| {row['pearson_r']:+.3f} | {row['spearman_rho']:+.3f} "
            f"| {row['sample_size']} | {floor} "
            f"| {'yes' if row['clears'] else 'no'} |"
        )
    lines.append("")
    lines.append(
        "The noise floor is the two-sided 5% critical correlation magnitude, "
        "computed in this document as a reading aid at each row's own sample "
        "size — which is why row 1 faces a much higher bar than the rest. It "
        "is not a bundle-sanctioned test: the screen table reports r, ρ, R² and "
        "residual degrees of freedom and no p-values by policy, and "
        f"{len(rows)} covariates screened together would need a multiplicity "
        "correction before any of them could be called significant."
    )
    lines.append("")

    for row in rows:
        if not (row["clears"] or row["shares_points"]):
            continue
        label = row["label"].replace("\n", " ")
        lines.append(f"### {label} · `{row['covariate']}`")
        lines.append("")
        detail: list[str] = [
            f"r = {row['pearson_r']:+.3f}, ρ = {row['spearman_rho']:+.3f} on "
            f"{row['sample_size']} series."
        ]
        leverage = row["leverage"]
        if leverage is not None:
            detail.append(
                "Removing the single most influential series "
                f"(`{leverage['series_uid']}`) moves r from "
                f"{leverage['baseline']:+.3f} to {leverage['reduced']:+.3f} on "
                f"the remaining {leverage['sample_size']}."
            )
            if row in fragile:
                detail.append(
                    "**That one series is the correlation.** The wide "
                    "Pearson–Spearman gap on this row in the figure is exactly "
                    "this leverage showing up as a rank/linear disagreement — "
                    "the ranks never supported the association in the first "
                    "place."
                )
                if _study_series_count(frame, leverage["series_uid"]) == 1:
                    detail.append(
                        "That series is also the only series its study "
                        "contributes, so this covariate cannot be separated "
                        "from *that one study* even in principle here."
                    )
            else:
                detail.append(
                    "No single series carries it: the association keeps its "
                    "sign and most of its magnitude under any single removal."
                )
        elif row["clears"]:
            # Never let a row clear the noise floor with no leverage line at
            # all: too few series to run the check reads as unqualified support.
            detail.append(
                "Too few series recorded this covariate to run the "
                "leave-one-series-out check, so nothing here rules out a "
                "single point carrying the whole correlation."
            )
        within = row["within_study"]
        if within is not None:
            detail.append(
                "Inside the single study contributing the most series "
                f"({within['sample_size']} of the {row['sample_size']} that "
                "recorded this covariate), r = "
                f"{within['pearson_r']:+.3f}"
                + (
                    " — the pattern reproduces without leaning on differences "
                    "between studies."
                    if abs(within["pearson_r"]) >= abs(row["pearson_r"]) * 0.75
                    else " — weaker than the pooled value, so part of the "
                    "pooled correlation is a contrast between studies rather "
                    "than a relationship among series."
                )
            )
            if within["sample_size"] >= 0.8 * row["sample_size"]:
                detail.append(
                    "The flip side is scope: nearly every series that recorded "
                    "this covariate belongs to that one study, so the row is "
                    "effectively a single-study result and says nothing about "
                    "the others."
                )
        if row["covariate"] in study_labels:
            detail.append(
                "This covariate never varies inside a study, so it is a study "
                "label in disguise: whatever it appears to track is a "
                "difference between studies, not an agronomic quantity."
            )
        if row["shares_points"]:
            detail.append(
                "**Circularity caveat.** This covariate is measured from the "
                "same observations used to fit the slope, so the two carry "
                "correlated errors by construction and some of the "
                "association is guaranteed by arithmetic rather than agronomy "
                "(`estimation_artefact_warning` in the screen table)."
            )
        lines.append(" ".join(detail))
        lines.append("")

    if study_labels:
        lines += [
            "## Metadata completeness masquerading as agronomy",
            "",
            "These covariates are constant within every study in this "
            "population: "
            + ", ".join(f"`{name}`" for name in study_labels)
            + ". A correlation between one of them and the fitted slopes "
            "cannot be separated from study identity. Read them as *which "
            "studies recorded this* rather than *what this does to a curve* — "
            "for a P/K indicator in particular, the honest reading is that the "
            "studies with fuller fertiliser metadata happen to respond "
            "differently, not that P and K change the response.",
            "",
        ]

    support = factors.support
    zero_variation = int(support["within_series_variation_count"].eq(0).sum())
    modifier_supported = int(
        (~support["modifier_status"].astype(str).str.startswith("held_")).sum()
    )
    factor_names = ", ".join(
        _factor_label(name).replace("\n", " ")
        for name in support["factor"].astype(str)
    )
    held_phrase = (
        "All of them have"
        if zero_variation == len(support)
        else f"{zero_variation} of them have"
    )
    lines += [
        "## Why no agronomic factor appears on this figure at all",
        "",
        f"The {len(support)} configured factors — {factor_names} — are absent "
        "from the panel by construction, not by omission. "
        f"{held_phrase} `within_series_variation_count = 0` in "
        "`factors/factor_support.csv`: they hold one value for the "
        "whole of a response series. A quantity that does not change along a "
        "curve cannot bend that curve, so there is no slope modifier to "
        f"estimate. `factor_support.csv` reports {modifier_supported} factors "
        "with modifier support.",
        "",
        "Some rows on the figure look like exceptions and are not. A P/K "
        "indicator and a zero-N-control indicator are screened here because "
        "*whether a series recorded them* is a property of the series, while "
        "the *rates themselves* are series-level constants. Recording is "
        "bookkeeping; the agronomic rate is still held.",
        "",
    ]

    decomposition = heterogeneity.decomposition
    if isinstance(decomposition, pd.DataFrame) and not decomposition.empty:
        indexed = decomposition.set_index("structure")
        if "series_fixed_intercepts" in indexed.index:
            series_share = float(
                indexed.loc[
                    "series_fixed_intercepts", "sse_reduction_fraction_vs_pooled"
                ]
            )
            lines += [
                "This is the same wall the level-side figure runs into. Series "
                f"identity alone removes {series_share:.1%} of the residual "
                "spread around a pooled N line, and "
                "every named factor is nested inside it. Series identity is "
                "not an explanation — it is a label for whatever differs "
                "between trials that this dataset did not record.",
                "",
            ]

    lines += [
        "## What this figure cannot support",
        "",
        "- **No causal or predictive claim.** Every value carries "
        "`analysis_role = series_level_association_not_causal`. Series-level "
        "covariates are confounded with study, site, calendar era and design.",
        "- **No fertiliser recommendation, agronomic optimum, or transferable "
        "coefficient.**",
        "- **No modifier effect size.** A correlation with a fitted slope is "
        "not an interaction estimate; none was fitted.",
        "- **Unequal precision is ignored.** Each series contributes one slope "
        "unweighted, although the slopes are estimated with very different "
        "standard errors.",
        "- **Slopes are straight-line summaries.** A series fitted with three "
        "or four N levels has a slope, not a curve shape; nothing here speaks "
        "to plateaus or optima.",
        "",
    ]

    lines += [
        "## Provenance",
        "",
        "- Screen values: `factors/series_slope_modifier_screen.csv`",
        "- Series slopes, covariates, and the dashed reference line: derived "
        "in-run and not published as bundle tables",
        "- Factor holds: `factors/factor_support.csv`",
        "- Both the figure and this file are written by "
        "`modules/n_response_curve/reporting/grain_yield_response.py` "
        "(`_plot_slope_modifier_ranking`, "
        "`_modifier_ranking_explainer_markdown`) and are hashed into "
        "`run_manifest.json` and `CHECKSUMS.sha256`. Editing either by hand "
        "breaks bundle verification; rerun the recipe instead.",
    ]
    if collapsed:
        lines.append(
            "- Collapsed as duplicate screens of an identical column in this "
            "population: " + ", ".join(f"`{name}`" for name in collapsed) + "."
        )
    note = _figures_note(figure_paths, "response_curve_modifier_ranking")
    if note:
        lines += ["", note.strip()]
    lines.append("")
    return "\n".join(lines)


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
                "`factors/factor_additive_screen.csv`, next to their "
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
            "`factors/factor_support.csv`). Such factors can describe "
            "which series differ, but no regression on these data can separate "
            "them from series identity, and none of them can bend an observed "
            "within-series response curve.\n"
        )
    else:
        lines.append(
            f"{factor_count - zero_variation} of {factor_count} configured "
            "factors vary within at least one response series; the rest are "
            "series-level constants that cannot be separated from series "
            "identity (`factors/factor_support.csv`).\n"
        )

    if series_adjusted_screen is not None and not series_adjusted_screen.empty:
        absorbed = series_adjusted_screen[
            series_adjusted_screen["series_adjusted_status"]
            == "absorbed_by_series_intercepts"
        ]["factor"].astype(str).tolist()
        if absorbed:
            lines.append(
                "- **Series-adjusted screen** "
                "(`factors/factor_series_adjusted_screen.csv`): with one "
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
                "(`factors/factor_redundancy_audit.csv`): "
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
                "(`factors/factor_evidence_audit.csv`): "
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
                "(`factors/series_slope_modifier_screen.csv`): derived "
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
        "t ha⁻¹ per 100 kg N ha⁻¹. "
    )


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
        f"observations in {int(summary.get('series_with_zero_n', 0))} "
        "series.\n\n"
    )


def _key_findings_lines(
    linear: pd.Series,
    quadratic: pd.Series,
    heterogeneity: HeterogeneityResults,
    factors: FactorSupportResults,
    mixed_model: Mapping[str, Any],
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
    explainer_paths: tuple[str, ...] = (),
) -> list[str]:
    if not table_paths and not figure_paths:
        return []
    lines: list[str] = [
        "## Bundle contents\n",
        "Artifacts are grouped by theme. `run_manifest.json` inventories every "
        "artifact with hashes and semantic metadata; `CHECKSUMS.sha256` binds "
        "the whole bundle.\n",
    ]

    group_order = {"factors": 0}

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
                # Filename first: an explainer shares its stem with the figure
                # it documents, so a stem-only lookup would label it twice.
                description = descriptions.get(filename) or descriptions.get(stem, "")
                lines.append(
                    f"- `{filename}`"
                    + (f" — {description}" if description else "")
                )
            lines.append("")

    emit(table_paths, _TABLE_DESCRIPTIONS)
    emit(
        figure_paths + explainer_paths,
        {**_FIGURE_DESCRIPTIONS, **_EXPLAINER_DESCRIPTIONS},
    )
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
    table_paths: tuple[str, ...] = (),
    figure_paths: tuple[str, ...] = (),
    explainer_paths: tuple[str, ...] = (),
) -> str:
    linear = descriptive.pooled_models.set_index("model").loc["linear"]
    quadratic = descriptive.pooled_models.set_index("model").loc["quadratic"]
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

    uncertainty_parts = "".join(
        part
        for part in (
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
        + "## Pooled quadratic sensitivity\n\n"
        f"`{quadratic['equation']}`\n\n"
        + turning_point_text
        + "## Hierarchical diagnostic\n\n"
        f"R mixed-model status: `{mixed_model.get('status', 'not_run')}`. "
        f"Singular fit: `{mixed_model.get('singular', 'not_available')}`. "
        + mixed_model_sentence.rstrip()
        + "\n\n"
        "## Candidate-factor support\n\n"
        f"Complete-case additive baseline screens fitted: {fitted_screens}; factors "
        f"with enough within-series support for a future prespecified modifier review: "
        f"{modifier_eligible}. No N×factor modifier effect was estimated here. "
        "Factor rows differ in sample size and degrees of freedom and must not be "
        "read as causal effect rankings. See `factors/factor_support.csv` "
        "and `factors/factor_additive_screen.csv`.\n\n"
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
        + "\n".join(
            _bundle_contents_lines(table_paths, figure_paths, explainer_paths)
        )
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
    implementation_sha256: Mapping[str, str],
    series_covariates: SeriesCovariateResults | None = None,
    series_adjusted_screen: pd.DataFrame | None = None,
    redundancy_audit: pd.DataFrame | None = None,
    evidence_audit: pd.DataFrame | None = None,
) -> WrittenDiagnosticBundle:
    target = Path(root).resolve()
    if target.exists() and any(target.iterdir()):
        raise DiagnosticBundleError(f"Diagnostic staging root is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []

    tables: dict[str, pd.DataFrame] = {
        "factor_support": factors.support,
        "factor_level_summary": factors.level_summary,
        "factor_additive_screen": factors.additive_screen,
    }
    if series_covariates is not None:
        tables["series_slope_modifier_screen"] = series_covariates.modifier_screen
    if series_adjusted_screen is not None:
        tables["factor_series_adjusted_screen"] = series_adjusted_screen
    if redundancy_audit is not None:
        tables["factor_redundancy_audit"] = redundancy_audit
    if evidence_audit is not None:
        tables["factor_evidence_audit"] = evidence_audit
    for name, frame in tables.items():
        artifacts.append(_write_csv(target, table_relative_path(name), frame))

    figure_specs = [
        ("factor_baseline_screen", _plot_factor_screen(config, factors)),
        (
            "factor_explanatory_ranking",
            _plot_factor_ranking(
                config, heterogeneity, factors, series_adjusted_screen
            ),
        ),
    ]
    if series_covariates is not None:
        figure_specs.append(
            (
                "series_slope_vs_check_yield",
                _plot_series_modifiers(config, series_covariates),
            )
        )
        figure_specs.append(
            (
                "response_curve_modifier_ranking",
                _plot_slope_modifier_ranking(config, series_covariates, mixed_model),
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
    explainer_paths: list[str] = []
    if series_covariates is not None:
        relative = _EXPLAINER_PATHS["response_curve_modifier_ranking"]
        explainer_path = target / relative
        explainer_path.parent.mkdir(parents=True, exist_ok=True)
        explainer_path.write_text(
            _modifier_ranking_explainer_markdown(
                population,
                factors,
                series_covariates,
                heterogeneity,
                mixed_model,
                figure_paths=figure_paths,
            ),
            encoding="utf-8",
            newline="\n",
        )
        artifacts.append(
            {
                "relative_path": relative,
                "artifact_kind": "report",
                "media_type": "text/markdown",
                "sha256": sha256_file(explainer_path),
                "bytes": explainer_path.stat().st_size,
            }
        )
        explainer_paths.append(relative)

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
            table_paths=table_paths,
            figure_paths=figure_paths,
            explainer_paths=tuple(explainer_paths),
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


def _verify_prioritized_response_modifier_bundle(
    root: Path,
    *,
    placement: PurePosixPath = PRIORITIZED_RESPONSE_MODIFIER_ROOT,
) -> frozenset[str]:
    extension = root / placement
    if not extension.exists():
        return frozenset()
    if extension.is_symlink() or not extension.is_dir():
        raise DiagnosticBundleError(
            f"Prioritized response-modifier bundle is unsafe: {extension}"
        )
    manifest_path = extension / MANIFEST_NAME
    checksums_path = extension / CHECKSUMS_NAME
    for path in (manifest_path, checksums_path):
        if path.is_symlink() or not path.is_file():
            raise DiagnosticBundleError(
                f"Required response-modifier artifact is missing: {path}"
            )

    manifest = _strict_manifest(manifest_path)
    response_modifier_schema = manifest.get("schema_version")
    if placement == LEGACY_PRIORITIZED_RESPONSE_MODIFIER_ROOT:
        expected_response_modifier_schema = _LEGACY_RESPONSE_MODIFIER_SCHEMA_VERSION
    elif placement == INTERMEDIATE_PRIORITIZED_RESPONSE_MODIFIER_ROOT:
        expected_response_modifier_schema = _INTERMEDIATE_RESPONSE_MODIFIER_SCHEMA_VERSION
    elif placement == PRIORITIZED_RESPONSE_MODIFIER_ROOT:
        expected_response_modifier_schema = _RESPONSE_MODIFIER_SCHEMA_VERSION
    else:
        raise DiagnosticBundleError(
            f"Unsupported response-modifier placement: {placement}"
        )
    if response_modifier_schema != expected_response_modifier_schema:
        raise DiagnosticBundleError("Unsupported response-modifier manifest schema")
    if manifest.get("status") != BUNDLE_STATUS:
        raise DiagnosticBundleError("Unsupported response-modifier bundle status")
    if manifest.get("data_classification") != "restricted":
        raise DiagnosticBundleError(
            "Response-modifier bundle must remain classified as restricted"
        )
    package_extension = manifest.get("package_extension")
    if not isinstance(package_extension, dict):
        raise DiagnosticBundleError("Response-modifier package metadata is missing")
    if (
        package_extension.get("placement")
        != placement.as_posix()
        or package_extension.get("placement_mode")
        != _RESPONSE_MODIFIER_PLACEMENT_MODE
        or package_extension.get("in_host_checksum_ledger") is not False
    ):
        raise DiagnosticBundleError(
            "Response-modifier package placement contract is invalid"
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise DiagnosticBundleError(
            "Response-modifier manifest artifact inventory is empty"
        )
    inventory: dict[str, dict[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("relative_path"), str):
            raise DiagnosticBundleError(
                "Response-modifier artifact declaration is invalid"
            )
        relative = item["relative_path"]
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
            raise DiagnosticBundleError(
                f"Unsafe response-modifier artifact path: {relative}"
            )
        if relative in inventory:
            raise DiagnosticBundleError(
                f"Duplicate response-modifier artifact: {relative}"
            )
        if response_modifier_schema in {
            _RESPONSE_MODIFIER_SCHEMA_VERSION,
            _INTERMEDIATE_RESPONSE_MODIFIER_SCHEMA_VERSION,
        }:
            parent = pure.parent.as_posix()
            valid_path = False
            if item.get("artifact_kind") == "document":
                valid_path = relative == "summary.md"
            elif item.get("artifact_kind") == "table" and pure.suffix == ".csv":
                if pure.name in {
                    "series_covariates.csv",
                    "series_slope_modifier_screen.csv",
                }:
                    valid_path = parent in _RESPONSE_MODIFIER_DATASET_GROUPS
                elif pure.name in {
                    "cross_dataset_screen_comparison.csv",
                    "population_summary.csv",
                    "mixed_model_summary.csv",
                }:
                    valid_path = parent == "all_datasets"
            elif item.get("artifact_kind") == "figure" and pure.suffix == ".jpeg":
                if pure.name.startswith("response_curve_modifier_ranking__"):
                    valid_path = parent in _RESPONSE_MODIFIER_DATASET_GROUPS
                elif pure.name == "cross_dataset_comparison.jpeg":
                    valid_path = parent == "all_datasets"
                elif pure.name.startswith(
                    ("core_trial_data__", "ltcce__")
                ):
                    valid_path = parent == "paired_core_vs_ltcce"
            if not valid_path:
                raise DiagnosticBundleError(
                    "Response-modifier artifact path does not match its schema: "
                    f"{relative}"
                )
        inventory[relative] = item

    actual_files: set[str] = set()
    for path in extension.rglob("*"):
        if path.is_symlink():
            raise DiagnosticBundleError(
                f"Response-modifier bundle contains a symlink: {path}"
            )
        if path.is_file():
            actual_files.add(path.relative_to(extension).as_posix())
    expected_files = set(inventory) | {MANIFEST_NAME, CHECKSUMS_NAME}
    if actual_files != expected_files:
        raise DiagnosticBundleError(
            "Response-modifier bundle inventory mismatch: "
            f"{sorted(actual_files ^ expected_files)}"
        )
    if package_extension.get("file_count") != len(expected_files):
        raise DiagnosticBundleError(
            "Response-modifier package file count does not match its inventory"
        )

    checksums = _parse_checksums(checksums_path)
    if set(checksums) != expected_files - {CHECKSUMS_NAME}:
        raise DiagnosticBundleError(
            "Response-modifier checksum inventory is incomplete"
        )
    for relative, digest in checksums.items():
        if sha256_file(extension / relative) != digest:
            raise DiagnosticBundleError(
                f"Response-modifier checksum mismatch: {relative}"
            )
    for relative, item in inventory.items():
        path = extension / relative
        if sha256_file(path) != item.get("sha256"):
            raise DiagnosticBundleError(
                f"Response-modifier manifest hash mismatch: {relative}"
            )
        if path.stat().st_size != item.get("bytes"):
            raise DiagnosticBundleError(
                f"Response-modifier byte count mismatch: {relative}"
            )
        kind = item.get("artifact_kind")
        if kind == "table":
            try:
                frame = pd.read_csv(path)
            except Exception as exc:
                raise DiagnosticBundleError(
                    f"Response-modifier table is unreadable: {relative}"
                ) from exc
            if len(frame) != item.get("rows") or len(frame.columns) != item.get(
                "columns"
            ):
                raise DiagnosticBundleError(
                    f"Response-modifier table semantic mismatch: {relative}"
                )
        elif kind == "figure":
            with Image.open(path) as image:
                image.load()
                if image.size != (
                    item.get("width_pixels"),
                    item.get("height_pixels"),
                ) or image.mode != item.get("color_mode"):
                    raise DiagnosticBundleError(
                        f"Response-modifier figure semantic mismatch: {relative}"
                    )
        elif kind != "document":
            raise DiagnosticBundleError(
                f"Unknown response-modifier artifact kind: {kind}"
            )

    prefix = placement.as_posix()
    return frozenset(f"{prefix}/{relative}" for relative in expected_files)


def _expected_diagnostic_artifact_path(
    item: Mapping[str, Any],
    *,
    schema_version: str,
) -> str:
    relative = str(item["relative_path"])
    pure = PurePosixPath(relative)
    kind = item.get("artifact_kind")
    legacy = schema_version == LEGACY_SCHEMA_VERSION
    intermediate = schema_version == INTERMEDIATE_SCHEMA_VERSION
    if kind == "table":
        group = _TABLE_GROUPS.get(pure.stem) or _RETIRED_TABLE_GROUPS.get(pure.stem)
        if group is not None and pure.suffix == ".csv":
            prefix = f"tables/{group}" if legacy else group
            return f"{prefix}/{pure.name}"
    elif kind == "figure":
        group = _FIGURE_GROUPS.get(pure.stem) or _RETIRED_FIGURE_GROUPS.get(pure.stem)
        if group is not None and pure.suffix:
            if (intermediate or legacy) and pure.stem in {
                "factor_baseline_screen",
                "factor_explanatory_ranking",
                "series_slope_vs_check_yield",
                "response_curve_modifier_ranking",
            }:
                group = "factors"
            prefix = f"figures/{group}" if legacy else group
            return f"{prefix}/{pure.name}"
    elif kind == "report":
        if relative == "summary.md":
            return relative
        if pure.name == "response_curve_modifier_ranking.md":
            if legacy:
                return "figures/factors/response_curve_modifier_ranking.md"
            if intermediate:
                return "factors/response_curve_modifier_ranking.md"
            return _EXPLAINER_PATHS["response_curve_modifier_ranking"]
    raise DiagnosticBundleError(
        f"Diagnostic artifact is not registered for {schema_version}: {relative}"
    )


def _verify_diagnostic_bundle(
    root: str | Path,
    *,
    recognized_extension_files: frozenset[str] = frozenset(),
    schema_version: str = SCHEMA_VERSION,
) -> VerifiedDiagnosticBundle:
    target = Path(root).resolve()
    if target.is_symlink() or not target.is_dir():
        raise DiagnosticBundleError(f"Diagnostic bundle is missing: {target}")
    manifest_path = target / MANIFEST_NAME
    checksums_path = target / CHECKSUMS_NAME
    for path in (manifest_path, checksums_path):
        if path.is_symlink() or not path.is_file():
            raise DiagnosticBundleError(f"Required bundle artifact is missing: {path}")
    manifest = _strict_manifest(manifest_path)
    if manifest.get("schema_version") != schema_version:
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
        expected_relative = _expected_diagnostic_artifact_path(
            item,
            schema_version=schema_version,
        )
        if relative != expected_relative:
            raise DiagnosticBundleError(
                "Diagnostic artifact path does not match its schema: "
                f"{relative} != {expected_relative}"
            )
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
    if recognized_extension_files & expected_files:
        raise DiagnosticBundleError(
            "Recognized extension files overlap the core diagnostic inventory"
        )
    completed_files = expected_files | set(recognized_extension_files)
    if actual_files != completed_files:
        raise DiagnosticBundleError(
            f"Diagnostic bundle inventory mismatch: {sorted(actual_files ^ completed_files)}"
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


def verify_diagnostic_bundle(root: str | Path) -> VerifiedDiagnosticBundle:
    """Verify the strict core bundle; any nested extension remains an error."""

    return _verify_diagnostic_bundle(root)


def verify_completed_diagnostic_container(
    root: str | Path,
) -> VerifiedDiagnosticBundle:
    """Verify the v3 core plus its canonical contributor modifier subtree."""

    target = Path(root).resolve()
    extension_files = _verify_prioritized_response_modifier_bundle(target)
    return _verify_diagnostic_bundle(
        target,
        recognized_extension_files=extension_files,
    )


def verify_replaceable_diagnostic_container(
    root: str | Path,
) -> VerifiedDiagnosticBundle:
    """Verify the current container or an exact replaceable v1/v2 layout."""

    target = Path(root).resolve()
    manifest_path = target / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DiagnosticBundleError(
            f"Required bundle artifact is missing: {manifest_path}"
        )
    schema_version = _strict_manifest(manifest_path).get("schema_version")
    if schema_version == SCHEMA_VERSION:
        return verify_completed_diagnostic_container(root)
    if schema_version == INTERMEDIATE_SCHEMA_VERSION:
        recognized_extension_files = _verify_prioritized_response_modifier_bundle(
            target,
            placement=INTERMEDIATE_PRIORITIZED_RESPONSE_MODIFIER_ROOT,
        )
        return _verify_diagnostic_bundle(
            target,
            recognized_extension_files=recognized_extension_files,
            schema_version=INTERMEDIATE_SCHEMA_VERSION,
        )
    if schema_version != LEGACY_SCHEMA_VERSION:
        raise DiagnosticBundleError("Unsupported diagnostic manifest schema")
    extension_files = _verify_prioritized_response_modifier_bundle(
        target,
        placement=LEGACY_PRIORITIZED_RESPONSE_MODIFIER_ROOT,
    )
    return _verify_diagnostic_bundle(
        target,
        recognized_extension_files=extension_files,
        schema_version=LEGACY_SCHEMA_VERSION,
    )
