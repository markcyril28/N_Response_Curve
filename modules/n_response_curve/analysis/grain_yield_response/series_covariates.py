"""Series-level covariates that may predict grain yield and response shape.

Nitrogen rate is the only treatment factor that varies inside a response
series in this population. Every configured operator factor is a series-level
constant, so it cannot be separated from series identity by any regression on
these data. This module therefore derives the covariates that describe the
*series itself* -- its zero-N check yield, its N-ladder geometry, its design
completeness, and its era -- and screens them against the fitted series slopes.

Nothing here is causal. A series-level association is a description of which
series differ, not evidence that the covariate produced the difference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from .population import distinct_n_level_count


DERIVED_CATEGORICAL_FACTORS: tuple[str, ...] = (
    "series_has_zero_n_control",
    "series_pk_recorded",
)

DERIVED_NUMERIC_FACTORS: tuple[str, ...] = (
    "series_check_yield_t_ha",
    "series_n_span_kg_ha",
    "series_mean_n_rate_kg_ha",
    "series_planting_year_numeric",
)

# Covariates offered to the series-slope modifier screen. Both
# `series_intercept_t_ha` and `series_mean_yield_t_ha` are deliberately excluded:
# they are outcome summaries estimated from the same points as the slope, so their
# errors and algebraic relationship can manufacture an association that is not an
# agronomic modifier signal.
_MODIFIER_COVARIATES: tuple[tuple[str, str], ...] = (
    ("series_check_yield_t_ha", "numeric"),
    ("series_n_span_kg_ha", "numeric"),
    ("series_n_min_kg_ha", "numeric"),
    ("series_n_max_kg_ha", "numeric"),
    ("series_mean_n_rate_kg_ha", "numeric"),
    ("series_planting_year_numeric", "numeric"),
    ("series_observation_count", "numeric"),
    ("series_distinct_n_level_count", "numeric"),
    ("series_has_zero_n_control", "binary"),
    ("series_pk_recorded", "binary"),
)

_ESTIMATION_ARTEFACT_COVARIATES = frozenset({"series_check_yield_t_ha"})


@dataclass(frozen=True)
class SeriesCovariateResults:
    series_frame: pd.DataFrame
    augmented_frame: pd.DataFrame
    modifier_screen: pd.DataFrame
    derived_categorical: tuple[str, ...]
    derived_numeric: tuple[str, ...]


def _numeric_year(values: pd.Series) -> float:
    parsed = pd.to_numeric(values, errors="coerce").dropna()
    if parsed.empty:
        return math.nan
    unique = parsed.unique()
    return float(unique[0]) if len(unique) == 1 else math.nan


def _series_linear_fit(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float]:
    """Return (slope, intercept, r_squared, slope_se); NaNs when degenerate.

    The slope SE is undefined for two-point ladders (no residual degrees of
    freedom); consumers must treat those slopes as unquantified, not precise.
    """

    if len(x) < 2 or len(np.unique(x)) < 2:
        return math.nan, math.nan, math.nan, math.nan
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != 2:
        return math.nan, math.nan, math.nan, math.nan
    residual = y - design @ coefficients
    rss = float(residual @ residual)
    centered_y = y - float(np.mean(y))
    tss = float(centered_y @ centered_y)
    centered_x = x - float(np.mean(x))
    sxx = float(centered_x @ centered_x)
    residual_df = len(x) - 2
    slope_se = (
        float(np.sqrt((rss / residual_df) / sxx))
        if residual_df > 0 and sxx > 0
        else math.nan
    )
    return (
        float(coefficients[1]),
        float(coefficients[0]),
        float(1.0 - rss / tss) if tss > 0 else math.nan,
        slope_se,
    )


def derive_series_covariates(
    frame: pd.DataFrame,
    *,
    series_key: str,
    study_key: str,
    trial_key: str,
    zero_n_tolerance_kg_ha: float,
    n_level_tolerance_kg_ha: float = 0.0,
    pk_columns: tuple[str, ...] = ("p_rate_kg_p2o5_ha", "k_rate_kg_k2o_ha"),
    planting_year_column: str = "planting_year",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive one covariate row per response series and merge them onto the rows."""

    required = {series_key, study_key, trial_key, "n_rate_kg_ha", "yield_t_ha"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Series covariates require columns: {missing}")

    rows: list[dict[str, Any]] = []
    for uid, group in frame.groupby(series_key, sort=True):
        if group[study_key].nunique(dropna=False) != 1 or group[trial_key].nunique(
            dropna=False
        ) != 1:
            raise ValueError(
                f"Response series {uid!s} maps to multiple study or trial identifiers"
            )
        x = group["n_rate_kg_ha"].to_numpy(dtype=float)
        y = group["yield_t_ha"].to_numpy(dtype=float)
        zero_mask = np.abs(x) <= zero_n_tolerance_kg_ha
        has_zero = bool(zero_mask.any())
        slope, intercept, r_squared, slope_se = _series_linear_fit(x, y)
        pk_recorded = set(pk_columns).issubset(group.columns) and all(
            bool(pd.to_numeric(group[column], errors="coerce").notna().all())
            for column in pk_columns
        )
        rows.append(
            {
                series_key: str(uid),
                study_key: str(group[study_key].iloc[0]),
                trial_key: str(group[trial_key].iloc[0]),
                "series_observation_count": int(len(group)),
                "series_distinct_n_level_count": distinct_n_level_count(
                    x, tolerance=n_level_tolerance_kg_ha
                ),
                "series_n_min_kg_ha": float(np.min(x)),
                "series_n_max_kg_ha": float(np.max(x)),
                "series_n_span_kg_ha": float(np.max(x) - np.min(x)),
                "series_mean_n_rate_kg_ha": float(np.mean(x)),
                "series_mean_yield_t_ha": float(np.mean(y)),
                "series_has_zero_n_control": has_zero,
                "series_check_yield_t_ha": (
                    float(np.mean(y[zero_mask])) if has_zero else math.nan
                ),
                "series_pk_recorded": pk_recorded,
                "series_planting_year_numeric": (
                    _numeric_year(group[planting_year_column])
                    if planting_year_column in group.columns
                    else math.nan
                ),
                "series_slope_t_ha_per_kg_n_ha": slope,
                "series_slope_se_t_ha_per_kg_n_ha": slope_se,
                "series_intercept_t_ha": intercept,
                "series_linear_r_squared": r_squared,
                "series_intercept_is_extrapolated": bool(
                    not has_zero and float(np.min(x)) > 0.0
                ),
                "analysis_role": "series_level_descriptive_covariate_not_causal",
            }
        )

    series_frame = pd.DataFrame(rows)
    merge_columns = [
        series_key,
        *DERIVED_CATEGORICAL_FACTORS,
        *DERIVED_NUMERIC_FACTORS,
    ]
    augmented = frame.copy()
    augmented[series_key] = augmented[series_key].astype(str)
    augmented = augmented.merge(
        series_frame.loc[:, merge_columns],
        on=series_key,
        how="left",
        validate="many_to_one",
    )
    if len(augmented) != len(frame):
        raise ValueError("Series covariate merge changed the observation count")
    return series_frame, augmented


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation with average ranks for ties; a robustness cross-check."""

    return _pearson(
        pd.Series(a).rank().to_numpy(dtype=float),
        pd.Series(b).rank().to_numpy(dtype=float),
    )


def screen_series_slope_modifiers(
    series_frame: pd.DataFrame,
    *,
    minimum_series: int,
) -> pd.DataFrame:
    """Associate each series-level covariate with the fitted series slopes."""

    slopes = pd.to_numeric(
        series_frame["series_slope_t_ha_per_kg_n_ha"], errors="coerce"
    )
    rows: list[dict[str, Any]] = []
    for covariate, kind in _MODIFIER_COVARIATES:
        if covariate not in series_frame.columns:
            rows.append(
                {
                    "covariate": covariate,
                    "covariate_kind": kind,
                    "screen_status": "absent",
                    "series_used": 0,
                }
            )
            continue
        raw = series_frame[covariate]
        values = (
            raw.astype(float)
            if kind == "binary" and raw.dtype == bool
            else pd.to_numeric(raw, errors="coerce")
        )
        if kind == "binary" and raw.dtype != bool:
            values = pd.to_numeric(raw.map({True: 1.0, False: 0.0}), errors="coerce")
        usable = values.notna() & slopes.notna()
        used = int(usable.sum())
        covariate_values = values.loc[usable].to_numpy(dtype=float)
        slope_values = slopes.loc[usable].to_numpy(dtype=float)
        if used < minimum_series or len(np.unique(covariate_values)) < 2:
            rows.append(
                {
                    "covariate": covariate,
                    "covariate_kind": kind,
                    "screen_status": (
                        "insufficient_series"
                        if used < minimum_series
                        else "no_covariate_variation"
                    ),
                    "series_used": used,
                    "series_missing": int(len(series_frame) - used),
                    "covariates_screened_in_family": len(_MODIFIER_COVARIATES),
                    "estimation_artefact_warning": (
                        "Covariate is an observed outcome from the same series "
                        "points used to estimate the slope; their errors can be "
                        "correlated by construction."
                        if covariate in _ESTIMATION_ARTEFACT_COVARIATES
                        else ""
                    ),
                    "analysis_role": "series_level_association_not_causal",
                }
            )
            continue
        design = np.column_stack([np.ones(used), covariate_values])
        coefficients, _, rank, _ = np.linalg.lstsq(design, slope_values, rcond=None)
        residual = slope_values - design @ coefficients
        rss = float(residual @ residual)
        centered = slope_values - float(np.mean(slope_values))
        tss = float(centered @ centered)
        correlation = _pearson(covariate_values, slope_values)
        row: dict[str, Any] = {
            "covariate": covariate,
            "covariate_kind": kind,
            "screen_status": "fitted" if rank == 2 else "rank_deficient",
            "series_used": used,
            "series_missing": int(len(series_frame) - used),
            "covariate_min": float(np.min(covariate_values)),
            "covariate_max": float(np.max(covariate_values)),
            "slope_intercept_t_ha_per_kg_n_ha": float(coefficients[0]),
            "slope_change_per_covariate_unit": float(coefficients[1]),
            "slope_equation": (
                "series_slope_t_ha_per_kg_n_ha = "
                f"{float(coefficients[0]):.10g}"
                f" {'+' if float(coefficients[1]) >= 0 else '-'} "
                f"{abs(float(coefficients[1])):.10g} * {covariate}"
            ),
            "pearson_r": correlation,
            "spearman_rho": _spearman(covariate_values, slope_values),
            "r_squared": float(1.0 - rss / tss) if tss > 0 else math.nan,
            "residual_df": int(used - rank),
            "weighting": "unweighted_each_series_counts_once",
            "covariates_screened_in_family": len(_MODIFIER_COVARIATES),
            "estimation_artefact_warning": (
                "Covariate is an observed outcome from the same series points used "
                "to estimate the slope; their errors can be correlated by "
                "construction."
                if covariate in _ESTIMATION_ARTEFACT_COVARIATES
                else ""
            ),
            "interpretation": (
                "Series-level association only. Slopes enter unweighted although "
                "they are estimated with unequal precision (see "
                "series_slope_se_t_ha_per_kg_n_ha in series_covariates); "
                "series-level covariates are confounded with study, site, era, "
                "and design; no causal or predictive claim is licensed."
            ),
            "analysis_role": "series_level_association_not_causal",
        }
        if kind == "binary":
            group_true = slope_values[covariate_values > 0.5]
            group_false = slope_values[covariate_values <= 0.5]
            row["group_true_series"] = int(len(group_true))
            row["group_false_series"] = int(len(group_false))
            row["group_true_mean_slope"] = (
                float(np.mean(group_true)) if len(group_true) else math.nan
            )
            row["group_false_mean_slope"] = (
                float(np.mean(group_false)) if len(group_false) else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)
