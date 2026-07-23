from __future__ import annotations

from dataclasses import dataclass
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from ..data.provenance import stable_identifier, stable_json_sha256


MODEL_ORDER = (
    "linear",
    "quadratic",
    "linear_plateau",
    "quadratic_plateau",
    "mitscherlich",
)
_MODEL_PARAMETER_COUNTS = {
    "linear": 2,
    "quadratic": 3,
    "linear_plateau": 3,
    "quadratic_plateau": 3,
    "mitscherlich": 3,
}
_MODEL_COMPLEXITY = {
    "linear": 1,
    "quadratic": 2,
    "linear_plateau": 3,
    "quadratic_plateau": 3,
    "mitscherlich": 3,
}


@dataclass(frozen=True)
class ModelAttempt:
    """One fully traceable, observed-domain response-model attempt."""

    model_attempt_uid: str
    response_series_uid: str
    model_name: str
    input_snapshot_sha256: str
    model_policy_sha256: str
    status: str
    reason_codes: tuple[str, ...]
    n_observations: int
    distinct_n_level_count: int
    observed_n_min_kg_ha: float | None
    observed_n_max_kg_ha: float | None
    residual_df: int | None
    rss: float | None
    aicc: float | None
    grouped_prediction_rmse: float | None
    grouped_prediction_fold_count: int
    parameters: Mapping[str, float]
    curve_shape_class: str | None
    optimum_status: str
    agronomic_optimum_n_kg_ha: float | None
    plateau_onset_n_kg_ha: float | None
    predicted_max_yield_t_ha: float | None
    predicted_observed_domain_peak_yield_t_ha: float | None
    finite_maximum_yield_t_ha: float | None
    fitted_asymptote_yield_t_ha: float | None
    supported_max_yield_t_ha: float | None
    maximum_reference_basis: str
    maximum_proximity_status: str
    predictions: tuple[Mapping[str, float], ...]


@dataclass(frozen=True)
class _OptimumSummary:
    optimum_status: str
    agronomic_optimum_n_kg_ha: float | None
    plateau_onset_n_kg_ha: float | None
    observed_domain_peak_yield_t_ha: float
    finite_maximum_yield_t_ha: float | None
    fitted_asymptote_yield_t_ha: float | None
    supported_max_yield_t_ha: float | None
    maximum_reference_basis: str
    maximum_proximity_status: str
    curve_shape_class: str
    reason_codes: tuple[str, ...]


def _frozen_mapping(values: Mapping[str, float] | None = None) -> Mapping[str, float]:
    return MappingProxyType(dict(values or {}))


def _attempt(
    *,
    response_series_uid: str,
    model_name: str,
    identity_payload: Mapping[str, Any],
    status: str,
    reason_codes: Iterable[str] = (),
    n_observations: int = 0,
    distinct_n_level_count: int = 0,
    observed_n_min_kg_ha: float | None = None,
    observed_n_max_kg_ha: float | None = None,
    residual_df: int | None = None,
    rss: float | None = None,
    aicc: float | None = None,
    grouped_prediction_rmse: float | None = None,
    grouped_prediction_fold_count: int = 0,
    parameters: Mapping[str, float] | None = None,
    curve_shape_class: str | None = None,
    optimum_status: str = "NOT_FITTED",
    agronomic_optimum_n_kg_ha: float | None = None,
    plateau_onset_n_kg_ha: float | None = None,
    predicted_max_yield_t_ha: float | None = None,
    predicted_observed_domain_peak_yield_t_ha: float | None = None,
    finite_maximum_yield_t_ha: float | None = None,
    fitted_asymptote_yield_t_ha: float | None = None,
    supported_max_yield_t_ha: float | None = None,
    maximum_reference_basis: str = "none",
    maximum_proximity_status: str = "NO_SUPPORTED_MAXIMUM_REFERENCE",
    predictions: Iterable[Mapping[str, float]] = (),
) -> ModelAttempt:
    input_snapshot_sha256 = stable_json_sha256(identity_payload["observations"])
    model_policy_sha256 = stable_json_sha256(identity_payload["policy"])
    return ModelAttempt(
        model_attempt_uid=stable_identifier(
            "model",
            (
                response_series_uid,
                model_name,
                input_snapshot_sha256,
                model_policy_sha256,
            ),
        ),
        response_series_uid=response_series_uid,
        model_name=model_name,
        input_snapshot_sha256=input_snapshot_sha256,
        model_policy_sha256=model_policy_sha256,
        status=status,
        reason_codes=tuple(sorted(set(reason_codes))),
        n_observations=n_observations,
        distinct_n_level_count=distinct_n_level_count,
        observed_n_min_kg_ha=observed_n_min_kg_ha,
        observed_n_max_kg_ha=observed_n_max_kg_ha,
        residual_df=residual_df,
        rss=rss,
        aicc=aicc,
        grouped_prediction_rmse=grouped_prediction_rmse,
        grouped_prediction_fold_count=grouped_prediction_fold_count,
        parameters=_frozen_mapping(parameters),
        curve_shape_class=curve_shape_class,
        optimum_status=optimum_status,
        agronomic_optimum_n_kg_ha=agronomic_optimum_n_kg_ha,
        plateau_onset_n_kg_ha=plateau_onset_n_kg_ha,
        predicted_max_yield_t_ha=predicted_max_yield_t_ha,
        predicted_observed_domain_peak_yield_t_ha=predicted_observed_domain_peak_yield_t_ha,
        finite_maximum_yield_t_ha=finite_maximum_yield_t_ha,
        fitted_asymptote_yield_t_ha=fitted_asymptote_yield_t_ha,
        supported_max_yield_t_ha=supported_max_yield_t_ha,
        maximum_reference_basis=maximum_reference_basis,
        maximum_proximity_status=maximum_proximity_status,
        predictions=tuple(MappingProxyType(dict(row)) for row in predictions),
    )


def _as_finite_array(values: Sequence[float | int]) -> np.ndarray | None:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or not np.isfinite(array).all():
        return None
    return array


def _observed_bounds(n_rates: np.ndarray) -> tuple[float, float]:
    return float(np.min(n_rates)), float(np.max(n_rates))


def _minimum_distinct_levels(model_name: str) -> int:
    return 3 if model_name == "linear" else 4


def _require_policy_number(policy: Mapping[str, Any], key: str) -> float:
    value = policy.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        raise ValueError(f"Modeling policy must define finite numeric {key!r}")
    return float(value)


def _policy_bounds(policy: Mapping[str, Any]) -> tuple[float, float, float, float, int, float]:
    minimum_yield = _require_policy_number(policy, "plausible_yield_min_t_ha")
    maximum_yield = _require_policy_number(policy, "plausible_yield_max_t_ha")
    if maximum_yield <= minimum_yield:
        raise ValueError("Modeling policy has an invalid plausible-yield range")
    parameter_bounds = policy.get("parameter_bounds")
    if not isinstance(parameter_bounds, Mapping):
        raise ValueError("Modeling policy must define parameter_bounds")
    minimum_n = _require_policy_number(parameter_bounds, "n_rate_min_kg_ha")
    maximum_n = _require_policy_number(parameter_bounds, "n_rate_max_kg_ha")
    if maximum_n <= minimum_n:
        raise ValueError("Modeling policy has an invalid N-rate range")
    residual_df = int(_require_policy_number(policy, "minimum_residual_df"))
    if residual_df < 1:
        raise ValueError("Modeling policy minimum_residual_df must be at least one")
    grid_points = int(_require_policy_number(policy, "plot_grid_points"))
    if grid_points < 2:
        raise ValueError("Modeling policy plot_grid_points must be at least two")
    tolerance = _require_policy_number(policy, "convergence_tolerance")
    if tolerance <= 0:
        raise ValueError("Modeling policy convergence_tolerance must be positive")
    return minimum_yield, maximum_yield, minimum_n, maximum_n, residual_df, tolerance


def _linear(x: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    intercept, slope = parameters
    return intercept + slope * x


def _quadratic(x: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    intercept, slope, curvature = parameters
    return intercept + slope * x + curvature * x**2


def _linear_plateau(x: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    intercept, slope, onset = parameters
    return intercept + slope * np.minimum(x, onset)


def _quadratic_plateau(x: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    baseline, gain, onset = parameters
    scaled = np.clip(x / onset, 0.0, 1.0)
    return baseline + gain * (2.0 * scaled - scaled**2)


def _mitscherlich(x: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    asymptote, amplitude, rate = parameters
    return asymptote - amplitude * np.exp(-rate * x)


_EVALUATORS: Mapping[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "linear": _linear,
    "quadratic": _quadratic,
    "linear_plateau": _linear_plateau,
    "quadratic_plateau": _quadratic_plateau,
    "mitscherlich": _mitscherlich,
}


def evaluate_model(model_name: str, n_rates: Sequence[float | int], parameters: Mapping[str, float]) -> np.ndarray:
    """Evaluate a fitted candidate only at caller-supplied N rates."""

    if model_name not in _EVALUATORS:
        raise ValueError(f"Unknown response model: {model_name}")
    x = _as_finite_array(n_rates)
    if x is None:
        raise ValueError("N rates must be a one-dimensional finite numeric sequence")
    names = {
        "linear": ("intercept", "slope"),
        "quadratic": ("intercept", "slope", "curvature"),
        "linear_plateau": ("intercept", "slope", "plateau_onset"),
        "quadratic_plateau": ("baseline", "gain", "plateau_onset"),
        "mitscherlich": ("asymptote", "amplitude", "rate"),
    }[model_name]
    try:
        values = np.asarray([parameters[name] for name in names], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid parameters for {model_name}") from exc
    return _EVALUATORS[model_name](x, values)


def _fit_parameters(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    minimum_yield: float,
    maximum_yield: float,
    tolerance: float,
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    if model_name == "linear":
        matrix = np.column_stack((np.ones_like(x), x))
        parameters, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
        if rank < 2:
            return None, ("RANK_DEFICIENT_DESIGN",)
        return parameters, ()
    if model_name == "quadratic":
        matrix = np.column_stack((np.ones_like(x), x, x**2))
        parameters, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
        if rank < 3:
            return None, ("RANK_DEFICIENT_DESIGN",)
        return parameters, ()

    x_min, x_max = _observed_bounds(x)
    x_span = x_max - x_min
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    safe_y_span = max(maximum_yield - minimum_yield, y_max - y_min, 1e-6)
    interior_lower = x_min + x_span * 1e-6
    interior_upper = x_max - x_span * 1e-6
    if interior_lower >= interior_upper:
        return None, ("INSUFFICIENT_N_RATE_RANGE",)

    if model_name == "linear_plateau":
        initial_slope = max((y[-1] - y[0]) / x_span, safe_y_span / (1000.0 * x_span))
        maximum_slope = max(safe_y_span * 10.0 / x_span, initial_slope * 10.0)
        initial = np.asarray([y_min, initial_slope, float(np.median(x))], dtype=float)
        lower = np.asarray([minimum_yield, 0.0, interior_lower], dtype=float)
        upper = np.asarray([maximum_yield, maximum_slope, interior_upper], dtype=float)
    elif model_name == "quadratic_plateau":
        initial_gain = max(y_max - y_min, safe_y_span / 1000.0)
        initial = np.asarray([y_min, initial_gain, float(np.median(x))], dtype=float)
        lower = np.asarray([minimum_yield, 0.0, interior_lower], dtype=float)
        upper = np.asarray([maximum_yield, safe_y_span, interior_upper], dtype=float)
    elif model_name == "mitscherlich":
        asymptote = min(maximum_yield, y_max + max(y_max - y_min, safe_y_span / 100.0))
        amplitude = max(asymptote - y_min, safe_y_span / 1000.0)
        initial = np.asarray([asymptote, amplitude, 1.0 / x_span], dtype=float)
        lower = np.asarray([minimum_yield, 0.0, 1e-12], dtype=float)
        upper = np.asarray([maximum_yield, safe_y_span, 10.0 / x_span], dtype=float)
    else:
        return None, ("UNKNOWN_MODEL",)

    initial = np.clip(initial, lower + np.finfo(float).eps, upper - np.finfo(float).eps)
    evaluator = _EVALUATORS[model_name]
    try:
        result = least_squares(
            lambda parameters: evaluator(x, parameters) - y,
            x0=initial,
            bounds=(lower, upper),
            xtol=tolerance,
            ftol=tolerance,
            gtol=tolerance,
            max_nfev=10000,
        )
    except (FloatingPointError, ValueError, RuntimeError) as exc:
        return None, (f"OPTIMIZER_ERROR:{type(exc).__name__}",)
    if not result.success or not np.isfinite(result.x).all():
        return None, ("NONCONVERGENT_FIT",)
    return result.x, ()


def _parameter_mapping(model_name: str, values: np.ndarray) -> dict[str, float]:
    names = {
        "linear": ("intercept", "slope"),
        "quadratic": ("intercept", "slope", "curvature"),
        "linear_plateau": ("intercept", "slope", "plateau_onset"),
        "quadratic_plateau": ("baseline", "gain", "plateau_onset"),
        "mitscherlich": ("asymptote", "amplitude", "rate"),
    }[model_name]
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def _parameter_rank_is_full(
    model_name: str,
    x: np.ndarray,
    parameters: np.ndarray,
) -> bool:
    evaluator = _EVALUATORS[model_name]
    steps = np.maximum(np.abs(parameters), 1.0) * math.sqrt(np.finfo(float).eps)
    columns: list[np.ndarray] = []
    for index, step in enumerate(steps):
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += step
        lower[index] -= step
        columns.append((evaluator(x, upper) - evaluator(x, lower)) / (2.0 * step))
    jacobian = np.column_stack(columns)
    return bool(
        np.isfinite(jacobian).all()
        and np.linalg.matrix_rank(jacobian) == len(parameters)
    )


def _grouped_prediction_summary(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    minimum_yield: float,
    maximum_yield: float,
    minimum_residual_df: int,
    tolerance: float,
) -> tuple[float | None, int]:
    """Evaluate a fitted model by leaving out every distinct N-rate level once."""

    squared_errors: list[float] = []
    held_out_levels = sorted(set(float(value) for value in x))
    parameter_count = _MODEL_PARAMETER_COUNTS[model_name]
    for held_out_level in held_out_levels:
        test_mask = np.isclose(x, held_out_level, rtol=0.0, atol=tolerance)
        train_mask = ~test_mask
        training_x = x[train_mask]
        training_y = y[train_mask]
        if (
            len(set(float(value) for value in training_x)) < _minimum_distinct_levels(model_name)
            or len(training_x) - parameter_count < minimum_residual_df
        ):
            return None, 0
        parameters, _ = _fit_parameters(
            model_name,
            training_x,
            training_y,
            minimum_yield=minimum_yield,
            maximum_yield=maximum_yield,
            tolerance=tolerance,
        )
        if parameters is None:
            return None, 0
        parameter_map = _parameter_mapping(model_name, parameters)
        predicted = evaluate_model(model_name, x[test_mask].tolist(), parameter_map)
        if (
            not np.isfinite(predicted).all()
            or np.min(predicted) < minimum_yield - tolerance
            or np.max(predicted) > maximum_yield + tolerance
        ):
            return None, 0
        squared_errors.extend(float(value) for value in (predicted - y[test_mask]) ** 2)
    if not squared_errors:
        return None, 0
    return float(math.sqrt(float(np.mean(squared_errors)))), len(held_out_levels)


def _aicc(rss: float, n_observations: int, parameter_count: int) -> float | None:
    if n_observations <= parameter_count + 1:
        return None
    mean_square = max(rss / n_observations, np.finfo(float).tiny)
    aic = n_observations * math.log(mean_square) + 2.0 * parameter_count
    return float(aic + (2.0 * parameter_count * (parameter_count + 1)) / (n_observations - parameter_count - 1))


def _predictions(
    model_name: str,
    parameters: Mapping[str, float],
    observed_min: float,
    observed_max: float,
    grid_points: int,
) -> tuple[dict[str, float], ...]:
    n_grid = np.linspace(observed_min, observed_max, grid_points)
    y_grid = evaluate_model(model_name, n_grid, parameters)
    return tuple(
        {
            "n_rate_kg_ha": float(n_rate),
            "predicted_yield_t_ha": float(yield_value),
        }
        for n_rate, yield_value in zip(n_grid, y_grid, strict=True)
    )


def _optimum_summary(
    model_name: str,
    parameters: Mapping[str, float],
    predictions: Sequence[Mapping[str, float]],
    *,
    observed_n_rates: np.ndarray,
    observed_min: float,
    observed_max: float,
    tolerance: float,
    parameter_rank_full: bool,
) -> _OptimumSummary:
    predicted_values = [float(row["predicted_yield_t_ha"]) for row in predictions]
    observed_domain_peak = max(predicted_values)
    boundary_tolerance = max((observed_max - observed_min) * 1e-6, tolerance)
    response_scale = max(max(abs(value) for value in predicted_values), 1.0)
    response_tolerance = max(math.sqrt(tolerance) * response_scale, 1.0e-8)
    span = observed_max - observed_min

    def summary(
        optimum_status: str,
        optimum: float | None,
        plateau_onset: float | None,
        finite_maximum: float | None,
        asymptote: float | None,
        basis: str,
        proximity_status: str,
        shape: str,
        reasons: tuple[str, ...] = (),
    ) -> _OptimumSummary:
        supported_maximum = finite_maximum if finite_maximum is not None else asymptote
        return _OptimumSummary(
            optimum_status,
            optimum,
            plateau_onset,
            observed_domain_peak,
            finite_maximum,
            asymptote,
            supported_maximum,
            basis,
            proximity_status,
            shape,
            reasons,
        )

    if model_name == "linear":
        slope = parameters["slope"]
        shape = "increasing_linear" if slope > tolerance else "decreasing_linear" if slope < -tolerance else "flat_linear"
        return summary(
            "NO_FINITE_OPTIMUM_LINEAR",
            None,
            None,
            None,
            None,
            "none",
            "NO_SUPPORTED_MAXIMUM_REFERENCE",
            shape,
        )
    if model_name == "quadratic":
        curvature = parameters["curvature"]
        curvature_effect = abs(curvature) * span**2
        slope_low = parameters["slope"] + 2.0 * curvature * observed_min
        slope_high = parameters["slope"] + 2.0 * curvature * observed_max
        if curvature_effect <= response_tolerance:
            return summary(
                "NO_IDENTIFIABLE_INTERIOR_MAXIMUM",
                None,
                None,
                None,
                None,
                "none",
                "NO_SUPPORTED_MAXIMUM_REFERENCE",
                "weak_quadratic_curvature",
                ("WEAK_QUADRATIC_CURVATURE",),
            )
        if curvature < 0.0:
            vertex = -parameters["slope"] / (2.0 * curvature)
            if observed_min + boundary_tolerance < vertex < observed_max - boundary_tolerance:
                maximum = float(evaluate_model(model_name, [vertex], parameters)[0])
                return summary(
                    "IDENTIFIABLE_INTERIOR_MAXIMUM",
                    float(vertex),
                    None,
                    maximum,
                    None,
                    "finite_interior_maximum",
                    "SUPPORTED_FINITE_MAXIMUM",
                    "concave_quadratic",
                )
            shape = (
                "diminishing_returns"
                if slope_low > 0.0 and slope_high > 0.0
                else "declining_concave"
                if slope_low < 0.0 and slope_high < 0.0
                else "concave_boundary_peak"
            )
        else:
            shape = (
                "accelerating_returns"
                if slope_low >= 0.0 and slope_high > 0.0
                else "convex_decline"
                if slope_low < 0.0 and slope_high <= 0.0
                else "convex_boundary_minimum"
            )
        return summary(
            "NO_IDENTIFIABLE_INTERIOR_MAXIMUM",
            None,
            None,
            None,
            None,
            "none",
            "NO_SUPPORTED_MAXIMUM_REFERENCE",
            shape,
            ("OPTIMUM_AT_OR_OUTSIDE_OBSERVED_DOMAIN",),
        )
    if model_name in {"linear_plateau", "quadratic_plateau"}:
        onset = parameters["plateau_onset"]
        distinct_rates = sorted(set(float(value) for value in observed_n_rates))
        left_support = sum(value < onset for value in distinct_rates)
        right_support = sum(value >= onset for value in distinct_rates)
        response_gain = (
            parameters["slope"] * max(onset - observed_min, 0.0)
            if model_name == "linear_plateau"
            else parameters["gain"]
        )
        identifiable = (
            parameter_rank_full
            and response_gain > response_tolerance
            and left_support >= 2
            and right_support >= 1
            and observed_min + boundary_tolerance < onset < observed_max - boundary_tolerance
        )
        if identifiable:
            maximum = float(evaluate_model(model_name, [onset], parameters)[0])
            return summary(
                "IDENTIFIABLE_PLATEAU_ONSET",
                float(onset),
                float(onset),
                maximum,
                None,
                "plateau_maximum",
                "SUPPORTED_FINITE_MAXIMUM",
                "plateau",
            )
        return summary(
            "UNSUPPORTED_OR_BOUNDARY_PLATEAU",
            None,
            None,
            None,
            None,
            "none",
            "NO_SUPPORTED_MAXIMUM_REFERENCE",
            "plateau_without_supported_onset",
            ("UNIDENTIFIABLE_SHAPE_PARAMETERS",),
        )
    if model_name == "mitscherlich":
        response_gain = max(predicted_values) - min(predicted_values)
        identifiable = (
            parameter_rank_full
            and parameters["amplitude"] > response_tolerance
            and response_gain > response_tolerance
            and parameters["rate"] * span > math.sqrt(np.finfo(float).eps)
        )
        if identifiable:
            return summary(
                "NO_FINITE_OPTIMUM_ASYMPTOTIC",
                None,
                None,
                None,
                float(parameters["asymptote"]),
                "identifiable_asymptote",
                "SUPPORTED_ASYMPTOTIC_MAXIMUM_REFERENCE",
                "asymptotic_diminishing_returns",
            )
        return summary(
            "UNIDENTIFIABLE_ASYMPTOTE",
            None,
            None,
            None,
            None,
            "none",
            "NO_SUPPORTED_MAXIMUM_REFERENCE",
            "asymptotic_without_supported_asymptote",
            ("UNIDENTIFIABLE_SHAPE_PARAMETERS",),
        )
    raise ValueError(f"Unknown response model: {model_name}")


def fit_candidate_model(
    response_series_uid: str,
    n_rates: Sequence[float | int],
    yields: Sequence[float | int],
    *,
    record_uids: Sequence[str] | None = None,
    model_name: str,
    policy: Mapping[str, Any],
) -> ModelAttempt:
    """Fit one predeclared candidate, retaining unsupported/failure evidence explicitly."""

    if model_name not in MODEL_ORDER:
        raise ValueError(f"Unknown response model: {model_name}")
    if not isinstance(response_series_uid, str) or not response_series_uid:
        raise ValueError("response_series_uid must be a nonempty string")
    if record_uids is not None and len(record_uids) != len(n_rates):
        raise ValueError("record_uids must align one-to-one with N-rate observations")
    if len(n_rates) == len(yields):
        identity_uids: Sequence[str | None] = record_uids if record_uids is not None else (None,) * len(n_rates)
        observations: object = tuple(
            sorted(
                zip(identity_uids, n_rates, yields),
                key=lambda row: json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str),
            )
        )
    else:
        observations = {"n_rates": list(n_rates), "yields": list(yields)}
    identity_payload = {"observations": observations, "policy": dict(policy)}
    if len(n_rates) != len(yields):
        return _attempt(
            response_series_uid=response_series_uid,
            model_name=model_name,
            identity_payload=identity_payload,
            status="unsupported",
            reason_codes=("N_RATE_YIELD_LENGTH_MISMATCH",),
        )
    n_observations = len(n_rates)
    x = _as_finite_array(n_rates)
    y = _as_finite_array(yields)
    if x is None or y is None:
        return _attempt(
            response_series_uid=response_series_uid,
            model_name=model_name,
            identity_payload=identity_payload,
            status="unsupported",
            reason_codes=("NONFINITE_OR_MISSING_OBSERVATION",),
            n_observations=n_observations,
        )
    observed_min, observed_max = _observed_bounds(x)
    distinct_levels = len(set(float(value) for value in x))
    try:
        minimum_yield, maximum_yield, minimum_n, maximum_n, minimum_residual_df, tolerance = _policy_bounds(policy)
    except ValueError:
        raise
    if observed_min < minimum_n - tolerance or observed_max > maximum_n + tolerance:
        return _attempt(
            response_series_uid=response_series_uid,
            model_name=model_name,
            identity_payload=identity_payload,
            status="unsupported",
            reason_codes=("N_RATE_OUTSIDE_CONFIGURED_BOUNDS",),
            n_observations=n_observations,
            distinct_n_level_count=distinct_levels,
            observed_n_min_kg_ha=observed_min,
            observed_n_max_kg_ha=observed_max,
        )
    if distinct_levels < _minimum_distinct_levels(model_name):
        return _attempt(
            response_series_uid=response_series_uid,
            model_name=model_name,
            identity_payload=identity_payload,
            status="unsupported",
            reason_codes=("INSUFFICIENT_DISTINCT_N_LEVELS",),
            n_observations=n_observations,
            distinct_n_level_count=distinct_levels,
            observed_n_min_kg_ha=observed_min,
            observed_n_max_kg_ha=observed_max,
        )
    parameter_count = _MODEL_PARAMETER_COUNTS[model_name]
    residual_df = n_observations - parameter_count
    if residual_df < minimum_residual_df:
        return _attempt(
            response_series_uid=response_series_uid,
            model_name=model_name,
            identity_payload=identity_payload,
            status="unsupported",
            reason_codes=("INSUFFICIENT_RESIDUAL_INFORMATION",),
            n_observations=n_observations,
            distinct_n_level_count=distinct_levels,
            observed_n_min_kg_ha=observed_min,
            observed_n_max_kg_ha=observed_max,
            residual_df=residual_df,
        )
    if np.min(y) < minimum_yield - tolerance or np.max(y) > maximum_yield + tolerance:
        return _attempt(
            response_series_uid=response_series_uid,
            model_name=model_name,
            identity_payload=identity_payload,
            status="unsupported",
            reason_codes=("OBSERVED_YIELD_OUTSIDE_PLAUSIBLE_RANGE",),
            n_observations=n_observations,
            distinct_n_level_count=distinct_levels,
            observed_n_min_kg_ha=observed_min,
            observed_n_max_kg_ha=observed_max,
            residual_df=residual_df,
        )

    parameters, fit_reasons = _fit_parameters(
        model_name,
        x,
        y,
        minimum_yield=minimum_yield,
        maximum_yield=maximum_yield,
        tolerance=tolerance,
    )
    if parameters is None:
        return _attempt(
            response_series_uid=response_series_uid,
            model_name=model_name,
            identity_payload=identity_payload,
            status="failed",
            reason_codes=fit_reasons,
            n_observations=n_observations,
            distinct_n_level_count=distinct_levels,
            observed_n_min_kg_ha=observed_min,
            observed_n_max_kg_ha=observed_max,
            residual_df=residual_df,
        )
    parameter_map = _parameter_mapping(model_name, parameters)
    predicted_observed = evaluate_model(model_name, x, parameter_map)
    if not np.isfinite(predicted_observed).all():
        return _attempt(
            response_series_uid=response_series_uid,
            model_name=model_name,
            identity_payload=identity_payload,
            status="failed",
            reason_codes=("NONFINITE_PREDICTION",),
            n_observations=n_observations,
            distinct_n_level_count=distinct_levels,
            observed_n_min_kg_ha=observed_min,
            observed_n_max_kg_ha=observed_max,
            residual_df=residual_df,
            parameters=parameter_map,
        )
    predictions = _predictions(
        model_name,
        parameter_map,
        observed_min,
        observed_max,
        int(policy["plot_grid_points"]),
    )
    predicted_grid = np.asarray([row["predicted_yield_t_ha"] for row in predictions])
    if (
        not np.isfinite(predicted_grid).all()
        or np.min(predicted_grid) < minimum_yield - tolerance
        or np.max(predicted_grid) > maximum_yield + tolerance
    ):
        return _attempt(
            response_series_uid=response_series_uid,
            model_name=model_name,
            identity_payload=identity_payload,
            status="failed",
            reason_codes=("IMPLAUSIBLE_OR_NONFINITE_DOMAIN_PREDICTION",),
            n_observations=n_observations,
            distinct_n_level_count=distinct_levels,
            observed_n_min_kg_ha=observed_min,
            observed_n_max_kg_ha=observed_max,
            residual_df=residual_df,
            parameters=parameter_map,
        )
    rss = float(np.sum((predicted_observed - y) ** 2))
    optimum_summary = _optimum_summary(
        model_name,
        parameter_map,
        predictions,
        observed_n_rates=x,
        observed_min=observed_min,
        observed_max=observed_max,
        tolerance=tolerance,
        parameter_rank_full=_parameter_rank_is_full(model_name, x, parameters),
    )
    reasons = list(optimum_summary.reason_codes)
    aicc = _aicc(rss, n_observations, parameter_count)
    if aicc is None:
        reasons.append("AICC_UNAVAILABLE")
    grouped_prediction_rmse, grouped_prediction_fold_count = _grouped_prediction_summary(
        model_name,
        x,
        y,
        minimum_yield=minimum_yield,
        maximum_yield=maximum_yield,
        minimum_residual_df=minimum_residual_df,
        tolerance=tolerance,
    )
    if grouped_prediction_rmse is None:
        reasons.append("GROUPED_PREDICTION_UNAVAILABLE")
    if policy.get("allow_uncertainty") is not True:
        reasons.append("UNCERTAINTY_DISABLED_BY_CONFIG")
    return _attempt(
        response_series_uid=response_series_uid,
        model_name=model_name,
        identity_payload=identity_payload,
        status="fitted",
        reason_codes=reasons,
        n_observations=n_observations,
        distinct_n_level_count=distinct_levels,
        observed_n_min_kg_ha=observed_min,
        observed_n_max_kg_ha=observed_max,
        residual_df=residual_df,
        rss=rss,
        aicc=aicc,
        grouped_prediction_rmse=grouped_prediction_rmse,
        grouped_prediction_fold_count=grouped_prediction_fold_count,
        parameters=parameter_map,
        curve_shape_class=optimum_summary.curve_shape_class,
        optimum_status=optimum_summary.optimum_status,
        agronomic_optimum_n_kg_ha=optimum_summary.agronomic_optimum_n_kg_ha,
        plateau_onset_n_kg_ha=optimum_summary.plateau_onset_n_kg_ha,
        predicted_max_yield_t_ha=optimum_summary.finite_maximum_yield_t_ha,
        predicted_observed_domain_peak_yield_t_ha=optimum_summary.observed_domain_peak_yield_t_ha,
        finite_maximum_yield_t_ha=optimum_summary.finite_maximum_yield_t_ha,
        fitted_asymptote_yield_t_ha=optimum_summary.fitted_asymptote_yield_t_ha,
        supported_max_yield_t_ha=optimum_summary.supported_max_yield_t_ha,
        maximum_reference_basis=optimum_summary.maximum_reference_basis,
        maximum_proximity_status=optimum_summary.maximum_proximity_status,
        predictions=predictions,
    )


def fit_response_models(
    records: Iterable[Mapping[str, Any]],
    *,
    model_names: Sequence[str],
    policy: Mapping[str, Any],
) -> tuple[ModelAttempt, ...]:
    """Fit each enabled candidate once per resolved series in deterministic order."""

    unknown_models = set(model_names) - set(MODEL_ORDER)
    if unknown_models:
        raise ValueError(f"Unknown response model(s): {', '.join(sorted(unknown_models))}")
    selected_models = tuple(model for model in MODEL_ORDER if model in set(model_names))
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        series_uid = record.get("response_series_uid")
        if record.get("series_status") != "resolved" or not isinstance(series_uid, str) or not series_uid:
            continue
        grouped.setdefault(series_uid, []).append(record)
    attempts: list[ModelAttempt] = []
    for series_uid in sorted(grouped):
        rows = sorted(grouped[series_uid], key=lambda row: str(row.get("record_uid", "")))
        record_uids = [str(row.get("record_uid", "")) for row in rows]
        n_rates = [row.get("n_rate_kg_ha") for row in rows]
        yields = [row.get("yield_t_ha") for row in rows]
        for model_name in selected_models:
            attempts.append(
                fit_candidate_model(
                    series_uid,
                    n_rates,
                    yields,
                    record_uids=record_uids,
                    model_name=model_name,
                    policy=policy,
                )
            )
    return tuple(attempts)


def select_reportable_model(attempts: Iterable[ModelAttempt]) -> ModelAttempt | None:
    """Select the best fitted candidate without hiding unsupported or failed attempts."""

    fitted = [
        attempt
        for attempt in attempts
        if attempt.status == "fitted"
        and "UNIDENTIFIABLE_SHAPE_PARAMETERS" not in attempt.reason_codes
    ]
    if not fitted:
        return None
    return min(
        fitted,
        key=lambda attempt: (
            math.inf if attempt.aicc is None else attempt.aicc,
            math.inf if attempt.grouped_prediction_rmse is None else attempt.grouped_prediction_rmse,
            _MODEL_COMPLEXITY[attempt.model_name],
            MODEL_ORDER.index(attempt.model_name),
            attempt.model_attempt_uid,
        ),
    )


def model_attempt_record(attempt: ModelAttempt) -> dict[str, Any]:
    """Return a JSON/columnar-friendly copy of a model-attempt ledger entry."""

    return {
        "model_attempt_uid": attempt.model_attempt_uid,
        "response_series_uid": attempt.response_series_uid,
        "model_name": attempt.model_name,
        "input_snapshot_sha256": attempt.input_snapshot_sha256,
        "model_policy_sha256": attempt.model_policy_sha256,
        "status": attempt.status,
        "reason_codes": list(attempt.reason_codes),
        "n_observations": attempt.n_observations,
        "distinct_n_level_count": attempt.distinct_n_level_count,
        "observed_n_min_kg_ha": attempt.observed_n_min_kg_ha,
        "observed_n_max_kg_ha": attempt.observed_n_max_kg_ha,
        "residual_df": attempt.residual_df,
        "rss": attempt.rss,
        "aicc": attempt.aicc,
        "grouped_prediction_rmse": attempt.grouped_prediction_rmse,
        "grouped_prediction_fold_count": attempt.grouped_prediction_fold_count,
        "parameters": dict(attempt.parameters),
        "curve_shape_class": attempt.curve_shape_class,
        "optimum_status": attempt.optimum_status,
        "agronomic_optimum_n_kg_ha": attempt.agronomic_optimum_n_kg_ha,
        "plateau_onset_n_kg_ha": attempt.plateau_onset_n_kg_ha,
        "predicted_max_yield_t_ha": attempt.predicted_max_yield_t_ha,
        "predicted_observed_domain_peak_yield_t_ha": attempt.predicted_observed_domain_peak_yield_t_ha,
        "finite_maximum_yield_t_ha": attempt.finite_maximum_yield_t_ha,
        "fitted_asymptote_yield_t_ha": attempt.fitted_asymptote_yield_t_ha,
        "supported_max_yield_t_ha": attempt.supported_max_yield_t_ha,
        "maximum_reference_basis": attempt.maximum_reference_basis,
        "maximum_proximity_status": attempt.maximum_proximity_status,
        "predictions": [dict(row) for row in attempt.predictions],
    }


__all__ = [
    "MODEL_ORDER",
    "ModelAttempt",
    "evaluate_model",
    "fit_candidate_model",
    "fit_response_models",
    "model_attempt_record",
    "select_reportable_model",
]
