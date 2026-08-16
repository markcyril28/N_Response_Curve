from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from .population import distinct_n_level_count


@dataclass(frozen=True)
class DescriptiveResults:
    pooled_models: pd.DataFrame
    model_grid: pd.DataFrame
    series_slopes: pd.DataFrame
    leave_one_series_out: pd.DataFrame
    zero_n_deltas: pd.DataFrame
    zero_n_delta_summary: dict[str, Any]


def _fit_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    *,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    columns = [np.ones(len(x)), x]
    if degree == 2:
        columns.append(x**2)
    design = np.column_stack(columns)
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("Polynomial design matrix is rank deficient")
    fitted = design @ coefficients
    residual = y - fitted
    rss = float(residual @ residual)
    centered = y - float(np.mean(y))
    tss = float(centered @ centered)
    r_squared = float(1.0 - rss / tss) if tss > 0 else math.nan
    residual_df = len(x) - design.shape[1]
    adjusted = (
        float(1.0 - (1.0 - r_squared) * (len(x) - 1) / residual_df)
        if residual_df > 0 and math.isfinite(r_squared)
        else math.nan
    )
    slope_se = math.nan
    slope_ci_low = math.nan
    slope_ci_high = math.nan
    cluster_count = len(set(groups.tolist())) if groups is not None else 0
    cluster_robust_status = "not_requested"
    cluster_robust_reason_code = "NO_CLUSTER_GROUPS"
    # Cluster-robust covariance is deliberately withheld for tiny cluster counts;
    # finite-sample corrections are unstable and can emit invalid standard errors.
    if groups is not None and cluster_count >= 4:
        robust = sm.OLS(y, design).fit(
            cov_type="cluster",
            cov_kwds={"groups": groups, "use_correction": True},
        )
        slope_se = float(robust.bse[1])
        interval = robust.conf_int(alpha=0.05)
        slope_ci_low = float(interval[1, 0])
        slope_ci_high = float(interval[1, 1])
        cluster_robust_status = "completed"
        cluster_robust_reason_code = ""
    elif groups is not None:
        cluster_robust_status = "withheld"
        cluster_robust_reason_code = "FEWER_THAN_FOUR_CLUSTERS"
    return {
        "intercept_t_ha": float(coefficients[0]),
        "slope_t_ha_per_kg_n_ha": float(coefficients[1]),
        "quadratic_t_ha_per_kg_n_ha_squared": (
            float(coefficients[2]) if degree == 2 else math.nan
        ),
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted,
        "rmse_t_ha": float(np.sqrt(np.mean(residual**2))),
        "rss": rss,
        "residual_df": float(residual_df),
        "cluster_robust_slope_se": slope_se,
        "cluster_robust_slope_ci95_low": slope_ci_low,
        "cluster_robust_slope_ci95_high": slope_ci_high,
        "cluster_count": cluster_count,
        "cluster_robust_status": cluster_robust_status,
        "cluster_robust_reason_code": cluster_robust_reason_code,
    }


def _signed_term(value: float, expression: str) -> str:
    sign = "+" if value >= 0 else "-"
    return f" {sign} {abs(value):.10g} * {expression}"


def format_equation(row: dict[str, Any]) -> str:
    equation = (
        "yield_t_ha = "
        f"{float(row['intercept_t_ha']):.10g}"
        + _signed_term(
            float(row["slope_t_ha_per_kg_n_ha"]),
            "n_rate_kg_ha",
        )
    )
    if row["model"] == "quadratic":
        equation += _signed_term(
            float(row["quadratic_t_ha_per_kg_n_ha_squared"]),
            "n_rate_kg_ha^2",
        )
    return equation


def analyze_descriptive(
    frame: pd.DataFrame,
    *,
    series_key: str,
    zero_n_tolerance_kg_ha: float,
    n_level_tolerance_kg_ha: float = 0.0,
) -> DescriptiveResults:
    """Compute bounded descriptive fits without making causal or optimum claims."""

    if frame.empty:
        raise ValueError("Descriptive analysis requires at least one observation")
    x = frame["n_rate_kg_ha"].to_numpy(dtype=float)
    y = frame["yield_t_ha"].to_numpy(dtype=float)
    groups = frame[series_key].astype(str).to_numpy()
    n_min = float(np.min(x))
    n_max = float(np.max(x))
    common = {
        "analysis_role": "descriptive_diagnostic_not_causal",
        "observations": int(len(frame)),
        "series_count": int(frame[series_key].nunique()),
        "distinct_n_levels": distinct_n_level_count(
            frame["n_rate_kg_ha"], tolerance=n_level_tolerance_kg_ha
        ),
        "n_min_kg_ha": n_min,
        "n_max_kg_ha": n_max,
        "yield_min_t_ha": float(np.min(y)),
        "yield_max_t_ha": float(np.max(y)),
    }
    model_rows: list[dict[str, Any]] = []
    for model, degree in (("linear", 1), ("quadratic", 2)):
        row = {
            "model": model,
            **common,
            **_fit_polynomial(x, y, degree, groups=groups),
        }
        curvature = row["quadratic_t_ha_per_kg_n_ha_squared"]
        if model == "quadratic" and curvature != 0:
            turning_point = -row["slope_t_ha_per_kg_n_ha"] / (2 * curvature)
            row["turning_point_n_kg_ha"] = float(turning_point)
            row["turning_point_in_observed_domain"] = bool(
                n_min <= turning_point <= n_max
            )
        else:
            row["turning_point_n_kg_ha"] = math.nan
            row["turning_point_in_observed_domain"] = False
        row["equation"] = format_equation(row)
        model_rows.append(row)

    theil = stats.theilslopes(y, x, alpha=0.95)
    theil_slope, theil_intercept, theil_low, theil_high = np.asarray(
        theil, dtype=float
    ).tolist()
    model_rows.append(
        {
            "model": "theil_sen_linear_sensitivity",
            **common,
            "intercept_t_ha": theil_intercept,
            "slope_t_ha_per_kg_n_ha": theil_slope,
            "quadratic_t_ha_per_kg_n_ha_squared": math.nan,
            "r_squared": math.nan,
            "adjusted_r_squared": math.nan,
            "rmse_t_ha": math.nan,
            "rss": math.nan,
            "residual_df": math.nan,
            "cluster_robust_slope_se": math.nan,
            "cluster_robust_slope_ci95_low": theil_low,
            "cluster_robust_slope_ci95_high": theil_high,
            "cluster_count": int(frame[series_key].nunique()),
            "cluster_robust_status": "not_applicable",
            "cluster_robust_reason_code": "THEIL_SEN_SENSITIVITY",
            "turning_point_n_kg_ha": math.nan,
            "turning_point_in_observed_domain": False,
            "equation": format_equation(
                {
                    "model": "linear",
                    "intercept_t_ha": theil_intercept,
                    "slope_t_ha_per_kg_n_ha": theil_slope,
                }
            ),
            "analysis_role": "robust_descriptive_sensitivity_not_causal",
        }
    )
    pooled_models = pd.DataFrame(model_rows)

    grid_x = np.linspace(n_min, n_max, 201)
    grid_rows: list[dict[str, Any]] = []
    for row in model_rows[:2]:
        predicted = (
            row["intercept_t_ha"]
            + row["slope_t_ha_per_kg_n_ha"] * grid_x
            + np.nan_to_num(row["quadratic_t_ha_per_kg_n_ha_squared"]) * grid_x**2
        )
        grid_rows.extend(
            {
                "model": row["model"],
                "n_rate_kg_ha": float(rate),
                "predicted_yield_t_ha": float(value),
            }
            for rate, value in zip(grid_x, predicted)
        )
    model_grid = pd.DataFrame(grid_rows)

    series_rows: list[dict[str, Any]] = []
    for uid, group in frame.groupby(series_key, sort=True):
        gx = group["n_rate_kg_ha"].to_numpy(dtype=float)
        gy = group["yield_t_ha"].to_numpy(dtype=float)
        base = {
            series_key: str(uid),
            "observations": int(len(group)),
            "distinct_n_levels": distinct_n_level_count(
                group["n_rate_kg_ha"], tolerance=n_level_tolerance_kg_ha
            ),
            "n_min_kg_ha": float(np.min(gx)),
            "n_max_kg_ha": float(np.max(gx)),
            "has_zero_n": bool(np.any(np.abs(gx) <= zero_n_tolerance_kg_ha)),
        }
        if len(group) < 2 or len(np.unique(gx)) < 2:
            series_rows.append(
                {
                    **base,
                    "status": "skipped",
                    "reason_code": "INSUFFICIENT_DISTINCT_N_LEVELS",
                    "intercept_t_ha": math.nan,
                    "slope_t_ha_per_kg_n_ha": math.nan,
                    "slope_se_t_ha_per_kg_n_ha": math.nan,
                    "r_squared": math.nan,
                    "rmse_t_ha": math.nan,
                }
            )
            continue
        fit = _fit_polynomial(gx, gy, 1)
        # Classical OLS slope SE; undefined for two-point ladders (no residual
        # degrees of freedom), which downstream views must not treat as precise.
        centered_gx = gx - float(np.mean(gx))
        sxx = float(centered_gx @ centered_gx)
        series_residual_df = len(gx) - 2
        slope_se = (
            float(np.sqrt((fit["rss"] / series_residual_df) / sxx))
            if series_residual_df > 0 and sxx > 0
            else math.nan
        )
        series_rows.append(
            {
                **base,
                "status": "fitted",
                "reason_code": "",
                "intercept_t_ha": fit["intercept_t_ha"],
                "slope_t_ha_per_kg_n_ha": fit[
                    "slope_t_ha_per_kg_n_ha"
                ],
                "slope_se_t_ha_per_kg_n_ha": slope_se,
                "r_squared": fit["r_squared"],
                "rmse_t_ha": fit["rmse_t_ha"],
            }
        )
    series_slopes = pd.DataFrame(series_rows)

    loo_rows: list[dict[str, Any]] = []
    if len(series_rows) >= 3:
        for uid in sorted(frame[series_key].astype(str).unique()):
            retained = frame[frame[series_key].astype(str) != uid]
            fit = _fit_polynomial(
                retained["n_rate_kg_ha"].to_numpy(dtype=float),
                retained["yield_t_ha"].to_numpy(dtype=float),
                1,
            )
            loo_rows.append(
                {
                    "omitted_series_uid": uid,
                    "retained_observations": int(len(retained)),
                    "slope_t_ha_per_kg_n_ha": fit[
                        "slope_t_ha_per_kg_n_ha"
                    ],
                    "intercept_t_ha": fit["intercept_t_ha"],
                    "r_squared": fit["r_squared"],
                }
            )
    leave_one_series_out = pd.DataFrame(
        loo_rows,
        columns=(
            "omitted_series_uid",
            "retained_observations",
            "slope_t_ha_per_kg_n_ha",
            "intercept_t_ha",
            "r_squared",
        ),
    )

    delta_rows: list[dict[str, Any]] = []
    represented_zero_series = 0
    for uid, group in frame.groupby(series_key, sort=True):
        zero_mask = group["n_rate_kg_ha"].abs() <= zero_n_tolerance_kg_ha
        if not bool(zero_mask.any()):
            continue
        represented_zero_series += 1
        baseline = float(group.loc[zero_mask, "yield_t_ha"].mean())
        for _, row in group.loc[~zero_mask].iterrows():
            delta_rows.append(
                {
                    series_key: str(uid),
                    "n_rate_kg_ha": float(row["n_rate_kg_ha"]),
                    "baseline_zero_n_yield_t_ha": baseline,
                    "yield_delta_from_zero_n_t_ha": float(row["yield_t_ha"])
                    - baseline,
                }
            )
    zero_n_deltas = pd.DataFrame(delta_rows)
    if not zero_n_deltas.empty:
        delta_x = zero_n_deltas["n_rate_kg_ha"].to_numpy(dtype=float)
        delta_y = zero_n_deltas["yield_delta_from_zero_n_t_ha"].to_numpy(
            dtype=float
        )
        denominator = float(delta_x @ delta_x)
        delta_slope = float(delta_x @ delta_y / denominator) if denominator else math.nan
        delta_residual = delta_y - delta_slope * delta_x
        delta_rmse = float(np.sqrt(np.mean(delta_residual**2)))
    else:
        delta_slope = math.nan
        delta_rmse = math.nan
    zero_n_delta_summary = {
        "status": "fitted" if not zero_n_deltas.empty else "skipped",
        "reason_code": "" if not zero_n_deltas.empty else "NO_ZERO_N_SERIES",
        "series_with_zero_n": represented_zero_series,
        "nonzero_observations": int(len(zero_n_deltas)),
        "slope_through_origin_t_ha_per_kg_n_ha": delta_slope,
        "rmse_t_ha": delta_rmse,
        "analysis_role": "zero_n_baseline_sensitivity_not_causal",
    }

    return DescriptiveResults(
        pooled_models=pooled_models,
        model_grid=model_grid,
        series_slopes=series_slopes,
        leave_one_series_out=leave_one_series_out,
        zero_n_deltas=zero_n_deltas,
        zero_n_delta_summary=zero_n_delta_summary,
    )


_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_MINIMUM_SERIES = 4


def bootstrap_pooled_uncertainty(
    frame: pd.DataFrame,
    *,
    series_key: str,
    random_seed: int,
    resamples: int = _BOOTSTRAP_RESAMPLES,
) -> pd.DataFrame:
    """Cluster-bootstrap percentile intervals for the pooled descriptive fits.

    Whole response series are resampled with replacement, preserving each
    series' internal N ladder, so the intervals reflect between-series
    heterogeneity — the dominant source of uncertainty in this population —
    rather than row-level noise alone. The resampling distribution quantifies
    instability of the pooled descriptive summaries; it licenses no causal,
    recommendation, or optimum claim. Deterministic given ``random_seed``.
    """

    groups = [
        (
            group["n_rate_kg_ha"].to_numpy(dtype=float),
            group["yield_t_ha"].to_numpy(dtype=float),
        )
        for _, group in frame.groupby(series_key, sort=True)
    ]
    series_count = len(groups)
    x_full = frame["n_rate_kg_ha"].to_numpy(dtype=float)
    y_full = frame["yield_t_ha"].to_numpy(dtype=float)
    n_min = float(np.min(x_full))
    n_max = float(np.max(x_full))
    common: dict[str, Any] = {
        "method": "cluster_bootstrap_percentile_resampling_response_series",
        "resamples_requested": int(resamples),
        "random_seed": int(random_seed),
        "series_count": int(series_count),
        "analysis_role": "descriptive_uncertainty_not_causal",
    }
    quantities = (
        "pooled_linear_slope_t_ha_per_kg_n_ha",
        "pooled_linear_intercept_t_ha",
        "pooled_quadratic_curvature_t_ha_per_kg_n_ha_squared",
        "pooled_quadratic_turning_point_n_kg_ha",
    )
    # Below four series the resampling distribution collapses onto a handful of
    # cluster combinations and the percentile interval is not meaningful.
    if series_count < _BOOTSTRAP_MINIMUM_SERIES:
        return pd.DataFrame(
            [
                {
                    "quantity": quantity,
                    "status": "withheld",
                    "reason_code": "FEWER_THAN_FOUR_SERIES",
                    **common,
                }
                for quantity in quantities
            ]
        )

    linear_point = _fit_polynomial(x_full, y_full, 1)
    quadratic_point = _fit_polynomial(x_full, y_full, 2)
    point_curvature = float(quadratic_point["quadratic_t_ha_per_kg_n_ha_squared"])
    point_turning = (
        -float(quadratic_point["slope_t_ha_per_kg_n_ha"]) / (2.0 * point_curvature)
        if point_curvature != 0
        else math.nan
    )

    rng = np.random.default_rng(random_seed)
    slopes: list[float] = []
    intercepts: list[float] = []
    curvatures: list[float] = []
    concave_turnings: list[float] = []
    linear_failures = 0
    quadratic_failures = 0
    for _ in range(resamples):
        chosen = rng.integers(0, series_count, size=series_count)
        x = np.concatenate([groups[index][0] for index in chosen])
        y = np.concatenate([groups[index][1] for index in chosen])
        try:
            fit = _fit_polynomial(x, y, 1)
        except ValueError:
            linear_failures += 1
        else:
            slopes.append(float(fit["slope_t_ha_per_kg_n_ha"]))
            intercepts.append(float(fit["intercept_t_ha"]))
        try:
            fit = _fit_polynomial(x, y, 2)
        except ValueError:
            quadratic_failures += 1
            continue
        curvature = float(fit["quadratic_t_ha_per_kg_n_ha_squared"])
        curvatures.append(curvature)
        # The turning point is a fitted maximum only when the parabola bends
        # downward; convex resamples answer a different question and are
        # summarised through the concavity share instead.
        if curvature < 0:
            concave_turnings.append(
                -float(fit["slope_t_ha_per_kg_n_ha"]) / (2.0 * curvature)
            )

    def percentile_row(
        quantity: str,
        estimate: float,
        values: list[float],
        failures: int,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = np.asarray(values, dtype=float)
        usable = int(len(data))
        row: dict[str, Any] = {
            "quantity": quantity,
            "status": "fitted" if usable >= resamples // 2 else "unstable",
            "reason_code": (
                "" if usable >= resamples // 2 else "TOO_FEW_USABLE_RESAMPLES"
            ),
            "estimate": estimate,
            "ci95_low": float(np.percentile(data, 2.5)) if usable else math.nan,
            "ci95_high": float(np.percentile(data, 97.5)) if usable else math.nan,
            "resamples_used": usable,
            "resamples_failed": int(failures),
            **common,
        }
        if extra:
            row.update(extra)
        return row

    quadratic_successes = resamples - quadratic_failures
    turning_extra: dict[str, Any] = {
        "definition": (
            "argmax of the fitted parabola, summarised over concave resamples"
        ),
        "observed_n_min_kg_ha": n_min,
        "observed_n_max_kg_ha": n_max,
        "share_concave_resamples": (
            len(concave_turnings) / quadratic_successes
            if quadratic_successes > 0
            else math.nan
        ),
        "share_turning_point_inside_observed_domain": (
            float(
                np.mean(
                    [n_min <= value <= n_max for value in concave_turnings]
                )
            )
            if concave_turnings
            else math.nan
        ),
        "share_turning_point_beyond_observed_maximum": (
            float(np.mean([value > n_max for value in concave_turnings]))
            if concave_turnings
            else math.nan
        ),
    }
    return pd.DataFrame(
        [
            percentile_row(
                "pooled_linear_slope_t_ha_per_kg_n_ha",
                float(linear_point["slope_t_ha_per_kg_n_ha"]),
                slopes,
                linear_failures,
            ),
            percentile_row(
                "pooled_linear_intercept_t_ha",
                float(linear_point["intercept_t_ha"]),
                intercepts,
                linear_failures,
            ),
            percentile_row(
                "pooled_quadratic_curvature_t_ha_per_kg_n_ha_squared",
                point_curvature,
                curvatures,
                quadratic_failures,
            ),
            percentile_row(
                "pooled_quadratic_turning_point_n_kg_ha",
                point_turning,
                concave_turnings,
                quadratic_failures,
                turning_extra,
            ),
        ]
    )


def analyze_raw_sensitivity(
    frame: pd.DataFrame,
    *,
    n_level_tolerance_kg_ha: float = 0.0,
) -> pd.DataFrame:
    """Fit separately labelled inventory-wide linear and quadratic sensitivities."""

    required = {"n_rate_kg_ha", "yield_t_ha"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Raw sensitivity frame is missing columns: {missing}")
    x = frame["n_rate_kg_ha"].to_numpy(dtype=float)
    y = frame["yield_t_ha"].to_numpy(dtype=float)
    if len(frame) < 3 or not np.isfinite(np.column_stack([x, y])).all():
        raise ValueError("Raw sensitivity requires at least three finite pairs")
    common = {
        "analysis_population": "raw_finite_inventory_sensitivity",
        "analysis_role": "descriptive_sensitivity_not_causal",
        "observations": int(len(frame)),
        "distinct_n_levels": distinct_n_level_count(
            frame["n_rate_kg_ha"], tolerance=n_level_tolerance_kg_ha
        ),
        "n_min_kg_ha": float(np.min(x)),
        "n_max_kg_ha": float(np.max(x)),
        "yield_min_t_ha": float(np.min(y)),
        "yield_max_t_ha": float(np.max(y)),
    }
    rows: list[dict[str, Any]] = []
    for model, degree in (
        ("raw_finite_linear", 1),
        ("raw_finite_quadratic_sensitivity", 2),
    ):
        fit = _fit_polynomial(x, y, degree)
        row: dict[str, Any] = {"model": model, **common, **fit}
        curvature = row["quadratic_t_ha_per_kg_n_ha_squared"]
        if degree == 2 and curvature != 0:
            turning_point = -row["slope_t_ha_per_kg_n_ha"] / (2 * curvature)
            row["turning_point_n_kg_ha"] = float(turning_point)
            row["turning_point_in_observed_domain"] = bool(
                common["n_min_kg_ha"] <= turning_point <= common["n_max_kg_ha"]
            )
            row["equation"] = format_equation(
                {**row, "model": "quadratic"}
            )
        else:
            row["turning_point_n_kg_ha"] = math.nan
            row["turning_point_in_observed_domain"] = False
            row["equation"] = format_equation({**row, "model": "linear"})
        rows.append(row)
    return pd.DataFrame(rows)
