from __future__ import annotations

from dataclasses import dataclass
import json
import math
from statistics import NormalDist
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from ..data.provenance import stable_identifier, stable_json_sha256
from .reviewed_methods import (
    UNCERTAINTY_METHOD_CONFIDENCE_LEVELS,
    UNCERTAINTY_METHOD_REQUIRED_EVIDENCE,
    UNCERTAINTY_METHOD_SPECS,
)


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
_MODEL_PARAMETER_NAMES = {
    "linear": ("intercept", "slope"),
    "quadratic": ("intercept", "slope", "curvature"),
    "linear_plateau": ("intercept", "slope", "plateau_onset"),
    "quadratic_plateau": ("baseline", "gain", "plateau_onset"),
    "mitscherlich": ("asymptote", "amplitude", "rate"),
}
_MODEL_INITIALIZATION_STRATEGIES = {
    "linear": "ordinary_least_squares",
    "quadratic": "ordinary_least_squares",
    "linear_plateau": "deterministic_data_anchored",
    "quadratic_plateau": "deterministic_data_anchored",
    "mitscherlich": "deterministic_data_anchored",
}
_MODEL_COMPLEXITY = {
    "linear": 1,
    "quadratic": 2,
    "linear_plateau": 3,
    "quadratic_plateau": 3,
    "mitscherlich": 3,
}
_MINIMUM_FITTED_LEVEL_COUNT = 4
_BROAD_ROSTER_LEVEL_COUNT = 5


@dataclass(frozen=True)
class _ReviewedModelGate:
    policy_id: str
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    allow_boundary_parameters: bool
    reportable_shape_classes: frozenset[str]
    optimizer_tolerance: float
    optimizer_max_iterations: int
    parameter_boundary_relative_tolerance: float
    optimum_boundary_tolerance_n_kg_ha: float
    flat_response_tolerance_t_ha: float


@dataclass(frozen=True)
class ModelAttempt:
    """One fully traceable, observed-domain response-model attempt."""

    model_attempt_uid: str
    response_series_uid: str
    model_name: str
    input_snapshot_sha256: str
    model_policy_sha256: str
    model_gate_policy_id: str | None
    estimator_policy_id: str
    estimator_name: str
    estimator_status: str
    analysis_grain: str
    weighted_sensitivity_status: str
    weighted_sensitivity_parameters: Mapping[str, float]
    weighted_sensitivity_objective: float | None
    weighted_sensitivity_reason_codes: tuple[str, ...]
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
    uncertainty_status: str
    uncertainty_method: str | None
    uncertainty_evidence_basis: tuple[str, ...]
    feature_variances: Mapping[str, float]
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
    model_gate_policy_id: str | None = None,
    estimator_status: str = "exploratory_unreviewed_grain",
    weighted_sensitivity_status: str = "not_run_incomplete_comparable_se_or_independence_evidence",
    weighted_sensitivity_parameters: Mapping[str, float] | None = None,
    weighted_sensitivity_objective: float | None = None,
    weighted_sensitivity_reason_codes: Iterable[str] = (),
    uncertainty_status: str = "suppressed_no_supported_evidence_basis",
    uncertainty_method: str | None = None,
    uncertainty_evidence_basis: Iterable[str] = (),
    feature_variances: Mapping[str, float] | None = None,
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
        model_gate_policy_id=model_gate_policy_id,
        estimator_policy_id="MOD-09-option-a",
        estimator_name="unweighted_least_squares",
        estimator_status=estimator_status,
        analysis_grain="one_reviewed_treatment_mean_per_distinct_n_level",
        weighted_sensitivity_status=weighted_sensitivity_status,
        weighted_sensitivity_parameters=_frozen_mapping(weighted_sensitivity_parameters),
        weighted_sensitivity_objective=weighted_sensitivity_objective,
        weighted_sensitivity_reason_codes=tuple(
            sorted(set(weighted_sensitivity_reason_codes))
        ),
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
        uncertainty_status=uncertainty_status,
        uncertainty_method=uncertainty_method,
        uncertainty_evidence_basis=tuple(sorted(set(uncertainty_evidence_basis))),
        feature_variances=_frozen_mapping(feature_variances),
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


def _estimator_grain_status(
    evidence: Sequence[Mapping[str, Any]],
    observation_count: int,
) -> str:
    if len(evidence) != observation_count or not evidence:
        return "exploratory_unreviewed_grain"
    if all(
        row.get("analysis_grain_status") == "reviewed_treatment_mean"
        for row in evidence
    ):
        return "primary_reviewed"
    return "exploratory_unreviewed_grain"


def _inverse_variance_sensitivity_weights(
    evidence: Sequence[Mapping[str, Any]],
    observation_count: int,
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    if len(evidence) != observation_count or not evidence:
        return None, ("INCOMPLETE_REPORTED_STANDARD_ERROR_EVIDENCE",)
    standard_errors: list[float] = []
    for row in evidence:
        if row.get("analysis_grain_status") != "reviewed_treatment_mean":
            return None, ("TREATMENT_MEAN_GRAIN_NOT_REVIEWED",)
        if row.get("yield_se_status") != "verified":
            return None, ("REPORTED_STANDARD_ERROR_NOT_VERIFIED",)
        if row.get("experimental_unit_status") != "verified":
            return None, ("EXPERIMENTAL_UNIT_NOT_VERIFIED",)
        if row.get("mean_independence_status") != "verified":
            return None, ("MEAN_INDEPENDENCE_NOT_VERIFIED",)
        raw_standard_error = row.get("yield_se_t_ha")
        if not isinstance(raw_standard_error, (int, float, str)):
            return None, ("INCOMPLETE_REPORTED_STANDARD_ERROR_EVIDENCE",)
        try:
            standard_error = float(raw_standard_error)
        except (TypeError, ValueError):
            return None, ("INCOMPLETE_REPORTED_STANDARD_ERROR_EVIDENCE",)
        if not math.isfinite(standard_error) or standard_error <= 0.0:
            return None, ("INVALID_REPORTED_STANDARD_ERROR",)
        standard_errors.append(standard_error)
    values = np.asarray(standard_errors, dtype=float)
    return 1.0 / np.square(values), ()


def _observed_bounds(n_rates: np.ndarray) -> tuple[float, float]:
    return float(np.min(n_rates)), float(np.max(n_rates))


def _minimum_distinct_levels(model_name: str) -> int:
    if model_name not in MODEL_ORDER:
        raise ValueError(f"Unknown response model: {model_name}")
    return _MINIMUM_FITTED_LEVEL_COUNT


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


def _restricted_fit_models(policy: Mapping[str, Any]) -> frozenset[str] | None:
    raw = policy.get("restricted_fit_models")
    if not isinstance(raw, (list, tuple, set, frozenset)) or isinstance(raw, (str, bytes)):
        return None
    names = tuple(str(value) for value in raw)
    if not names or len(names) != len(set(names)) or set(names) - set(MODEL_ORDER):
        return None
    if set(names) == set(MODEL_ORDER):
        return None
    return frozenset(names)


def _model_level_gate_reason(
    model_name: str,
    distinct_level_count: int,
    policy: Mapping[str, Any],
) -> str | None:
    if distinct_level_count < _MINIMUM_FITTED_LEVEL_COUNT:
        return "DESCRIPTIVE_LEVEL_SUPPORT_ONLY"
    if distinct_level_count >= _BROAD_ROSTER_LEVEL_COUNT:
        return None
    restricted_models = _restricted_fit_models(policy)
    if restricted_models is None:
        return "REVIEWED_RESTRICTED_FIT_ROSTER_UNAVAILABLE"
    if model_name not in restricted_models:
        return "MODEL_OUTSIDE_RESTRICTED_FIT_ROSTER"
    return None


def _reviewed_model_gate(
    model_name: str,
    policy: Mapping[str, Any],
) -> tuple[_ReviewedModelGate | None, str | None]:
    raw_policy = policy.get("model_gate_policy")
    if not isinstance(raw_policy, Mapping):
        return None, "REVIEWED_MODEL_GATE_POLICY_UNAVAILABLE"
    policy_id = raw_policy.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id.strip():
        return None, "REVIEWED_MODEL_GATE_POLICY_UNAVAILABLE"
    if raw_policy.get("review_status") != "approved":
        return None, "MODEL_GATE_POLICY_NOT_APPROVED"
    raw_models = raw_policy.get("models")
    if not isinstance(raw_models, Mapping):
        return None, "REVIEWED_MODEL_GATE_POLICY_UNAVAILABLE"
    raw_model = raw_models.get(model_name)
    if not isinstance(raw_model, Mapping):
        return None, "MODEL_SPECIFIC_GATE_UNAVAILABLE"
    if raw_model.get("initialization_strategy") != _MODEL_INITIALIZATION_STRATEGIES[model_name]:
        return None, "REVIEWED_INITIALIZATION_STRATEGY_UNAVAILABLE"
    allow_boundary = raw_model.get("allow_boundary_parameters")
    if not isinstance(allow_boundary, bool):
        return None, "REVIEWED_BOUNDARY_PARAMETER_RULE_UNAVAILABLE"
    raw_shapes = raw_model.get("reportable_shape_classes")
    if (
        not isinstance(raw_shapes, (list, tuple, set, frozenset))
        or isinstance(raw_shapes, (str, bytes))
        or not raw_shapes
    ):
        return None, "REVIEWED_SHAPE_PLAUSIBILITY_RULE_UNAVAILABLE"
    shape_classes = frozenset(str(value) for value in raw_shapes if str(value))
    if len(shape_classes) != len(raw_shapes):
        return None, "REVIEWED_SHAPE_PLAUSIBILITY_RULE_UNAVAILABLE"
    numeric_controls: dict[str, float] = {}
    for field in (
        "optimizer_tolerance",
        "parameter_boundary_relative_tolerance",
        "optimum_boundary_tolerance_n_kg_ha",
        "flat_response_tolerance_t_ha",
    ):
        value = raw_model.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            return None, "REVIEWED_MODEL_TOLERANCE_UNAVAILABLE"
        numeric_controls[field] = float(value)
    optimizer_max_iterations = raw_model.get("optimizer_max_iterations")
    if (
        isinstance(optimizer_max_iterations, bool)
        or not isinstance(optimizer_max_iterations, int)
        or optimizer_max_iterations <= 0
    ):
        return None, "REVIEWED_MODEL_ITERATION_LIMIT_UNAVAILABLE"
    raw_bounds = raw_model.get("parameter_bounds")
    if not isinstance(raw_bounds, Mapping):
        return None, "MODEL_SPECIFIC_PARAMETER_BOUNDS_UNAVAILABLE"
    names = _MODEL_PARAMETER_NAMES[model_name]
    if set(raw_bounds) != set(names):
        return None, "MODEL_SPECIFIC_PARAMETER_BOUNDS_UNAVAILABLE"
    lower: list[float] = []
    upper: list[float] = []
    for name in names:
        interval = raw_bounds[name]
        if (
            not isinstance(interval, (list, tuple))
            or len(interval) != 2
            or isinstance(interval[0], bool)
            or isinstance(interval[1], bool)
            or not isinstance(interval[0], (float, int))
            or not isinstance(interval[1], (float, int))
        ):
            return None, "MODEL_SPECIFIC_PARAMETER_BOUNDS_UNAVAILABLE"
        low = float(interval[0])
        high = float(interval[1])
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            return None, "MODEL_SPECIFIC_PARAMETER_BOUNDS_INVALID"
        lower.append(low)
        upper.append(high)
    return (
        _ReviewedModelGate(
            policy_id=policy_id.strip(),
            lower_bounds=np.asarray(lower, dtype=float),
            upper_bounds=np.asarray(upper, dtype=float),
            allow_boundary_parameters=allow_boundary,
            reportable_shape_classes=shape_classes,
            optimizer_tolerance=numeric_controls["optimizer_tolerance"],
            optimizer_max_iterations=optimizer_max_iterations,
            parameter_boundary_relative_tolerance=numeric_controls[
                "parameter_boundary_relative_tolerance"
            ],
            optimum_boundary_tolerance_n_kg_ha=numeric_controls[
                "optimum_boundary_tolerance_n_kg_ha"
            ],
            flat_response_tolerance_t_ha=numeric_controls[
                "flat_response_tolerance_t_ha"
            ],
        ),
        None,
    )


def _at_reviewed_parameter_boundary(
    parameters: np.ndarray,
    gate: _ReviewedModelGate,
) -> bool:
    scale = np.maximum(
        np.maximum(np.abs(gate.lower_bounds), np.abs(gate.upper_bounds)),
        1.0,
    )
    tolerance = gate.parameter_boundary_relative_tolerance
    boundary_tolerance = np.maximum(scale * tolerance, tolerance)
    return bool(
        np.any(np.abs(parameters - gate.lower_bounds) <= boundary_tolerance)
        or np.any(np.abs(parameters - gate.upper_bounds) <= boundary_tolerance)
    )


def _uncertainty_gate(
    policy: Mapping[str, Any],
    observation_evidence: Sequence[Mapping[str, Any]],
) -> tuple[str, str | None, tuple[str, ...], tuple[str, ...]]:
    if policy.get("allow_uncertainty") is not True:
        return "suppressed_by_policy", None, (), ("UNCERTAINTY_DISABLED",)
    raw_policy = policy.get("uncertainty_policy")
    if not isinstance(raw_policy, Mapping) or raw_policy.get("review_status") != "approved":
        return (
            "suppressed_unreviewed_method",
            None,
            (),
            ("REVIEWED_UNCERTAINTY_METHOD_UNAVAILABLE",),
        )
    policy_id = raw_policy.get("policy_id")
    method = raw_policy.get("method")
    method_contract = raw_policy.get("method_contract")
    required_basis = raw_policy.get("evidence_basis")
    if (
        not isinstance(policy_id, str)
        or not policy_id.strip()
        or not isinstance(method, str)
        or not method.strip()
        or not isinstance(method_contract, Mapping)
        or not isinstance(required_basis, (list, tuple, set, frozenset))
        or isinstance(required_basis, (str, bytes))
        or not required_basis
    ):
        return (
            "suppressed_unreviewed_method",
            None,
            (),
            ("REVIEWED_UNCERTAINTY_METHOD_UNAVAILABLE",),
        )
    expected_contract = UNCERTAINTY_METHOD_SPECS.get(method)
    method_required_basis = UNCERTAINTY_METHOD_REQUIRED_EVIDENCE.get(method)
    if (
        expected_contract is None
        or method_required_basis is None
        or dict(method_contract) != dict(expected_contract)
    ):
        return (
            "suppressed_unreviewed_method",
            method,
            (),
            ("REVIEWED_UNCERTAINTY_METHOD_UNAVAILABLE",),
        )
    supported_basis = {
        "reported_standard_error",
        "verified_mean_independence",
        "verified_true_replication",
    }
    requested_basis = tuple(sorted({str(value) for value in required_basis}))
    if not set(requested_basis).issubset(supported_basis):
        return (
            "suppressed_unreviewed_method",
            method,
            (),
            ("REVIEWED_UNCERTAINTY_METHOD_UNAVAILABLE",),
        )
    if (
        method not in UNCERTAINTY_METHOD_CONFIDENCE_LEVELS
        or not set(method_required_basis).issubset(requested_basis)
    ):
        return (
            "suppressed_unreviewed_method",
            method,
            (),
            ("REVIEWED_UNCERTAINTY_METHOD_UNAVAILABLE",),
        )
    available: set[str] = set()
    if observation_evidence and all(
        isinstance(row.get("yield_se_t_ha"), (float, int))
        and not isinstance(row.get("yield_se_t_ha"), bool)
        and math.isfinite(float(row["yield_se_t_ha"]))
        and float(row["yield_se_t_ha"]) >= 0.0
        and row.get("yield_se_status") == "verified"
        and row.get("experimental_unit_status") == "verified"
        for row in observation_evidence
    ):
        available.add("reported_standard_error")
    if observation_evidence and all(
        row.get("mean_independence_status") == "verified"
        for row in observation_evidence
    ):
        available.add("verified_mean_independence")
    if observation_evidence and all(
        isinstance(row.get("replicate_count"), int)
        and not isinstance(row.get("replicate_count"), bool)
        and int(row["replicate_count"]) >= 2
        and row.get("replication_status") == "verified"
        and row.get("experimental_unit_status") == "verified"
        for row in observation_evidence
    ):
        available.add("verified_true_replication")
    if not set(requested_basis).issubset(available):
        return (
            "suppressed_unsupported_evidence",
            method,
            tuple(sorted(available)),
            ("UNCERTAINTY_EVIDENCE_BASIS_UNSUPPORTED",),
        )
    return (
        "eligible_for_reviewed_method",
        method,
        requested_basis,
        (),
    )


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
    gate: _ReviewedModelGate,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    if weights is None:
        square_root_weights = np.ones_like(y)
    else:
        if (
            weights.shape != y.shape
            or not np.isfinite(weights).all()
            or np.any(weights <= 0.0)
        ):
            return None, ("INVALID_ESTIMATOR_WEIGHTS",)
        square_root_weights = np.sqrt(weights)
    if model_name == "linear":
        matrix = np.column_stack((np.ones_like(x), x))
        parameters, _, rank, _ = np.linalg.lstsq(
            matrix * square_root_weights[:, None],
            y * square_root_weights,
            rcond=None,
        )
        if rank < 2:
            return None, ("RANK_DEFICIENT_DESIGN",)
        if np.any(parameters < gate.lower_bounds) or np.any(parameters > gate.upper_bounds):
            return None, ("PARAMETERS_OUTSIDE_REVIEWED_BOUNDS",)
        return parameters, ()
    if model_name == "quadratic":
        matrix = np.column_stack((np.ones_like(x), x, x**2))
        parameters, _, rank, _ = np.linalg.lstsq(
            matrix * square_root_weights[:, None],
            y * square_root_weights,
            rcond=None,
        )
        if rank < 3:
            return None, ("RANK_DEFICIENT_DESIGN",)
        if np.any(parameters < gate.lower_bounds) or np.any(parameters > gate.upper_bounds):
            return None, ("PARAMETERS_OUTSIDE_REVIEWED_BOUNDS",)
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

    lower = gate.lower_bounds.copy()
    upper = gate.upper_bounds.copy()
    if model_name in {"linear_plateau", "quadratic_plateau"}:
        lower[2] = max(lower[2], interior_lower)
        upper[2] = min(upper[2], interior_upper)
        if lower[2] >= upper[2]:
            return None, ("NO_REVIEWED_INTERIOR_PLATEAU_DOMAIN",)

    if model_name == "linear_plateau":
        initial_slope = max((y[-1] - y[0]) / x_span, safe_y_span / (1000.0 * x_span))
        initial = np.asarray([y_min, initial_slope, float(np.median(x))], dtype=float)
    elif model_name == "quadratic_plateau":
        initial_gain = max(y_max - y_min, safe_y_span / 1000.0)
        initial = np.asarray([y_min, initial_gain, float(np.median(x))], dtype=float)
    elif model_name == "mitscherlich":
        asymptote = min(maximum_yield, y_max + max(y_max - y_min, safe_y_span / 100.0))
        amplitude = max(asymptote - y_min, safe_y_span / 1000.0)
        initial = np.asarray([asymptote, amplitude, 1.0 / x_span], dtype=float)
    else:
        return None, ("UNKNOWN_MODEL",)

    initial = np.minimum(
        np.maximum(initial, np.nextafter(lower, upper)),
        np.nextafter(upper, lower),
    )
    evaluator = _EVALUATORS[model_name]
    try:
        result = least_squares(
            lambda parameters: (evaluator(x, parameters) - y)
            * square_root_weights,
            x0=initial,
            bounds=(lower, upper),
            xtol=tolerance,
            ftol=tolerance,
            gtol=tolerance,
            max_nfev=gate.optimizer_max_iterations,
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
    gate: _ReviewedModelGate,
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
            gate=gate,
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


def _parameter_jacobian(
    model_name: str,
    n_values: np.ndarray,
    parameters: Mapping[str, float],
) -> np.ndarray:
    parameter_names = _MODEL_PARAMETER_NAMES[model_name]
    parameter_vector = np.asarray(
        [parameters[name] for name in parameter_names],
        dtype=float,
    )
    jacobian = np.empty((len(n_values), len(parameter_names)), dtype=float)
    step_scale = math.sqrt(np.finfo(float).eps)
    for index in range(len(parameter_names)):
        step = step_scale * max(abs(float(parameter_vector[index])), 1.0)
        upper = parameter_vector.copy()
        lower = parameter_vector.copy()
        upper[index] += step
        lower[index] -= step
        upper_values = evaluate_model(
            model_name,
            n_values.tolist(),
            _parameter_mapping(model_name, upper),
        )
        lower_values = evaluate_model(
            model_name,
            n_values.tolist(),
            _parameter_mapping(model_name, lower),
        )
        jacobian[:, index] = (upper_values - lower_values) / (2.0 * step)
    return jacobian


def _reported_se_delta_intervals(
    model_name: str,
    parameters: Mapping[str, float],
    observed_n_rates: np.ndarray,
    observation_evidence: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, float]],
    *,
    method: str,
) -> tuple[
    tuple[dict[str, float], ...],
    str,
    tuple[str, ...],
    np.ndarray | None,
]:
    confidence_level = UNCERTAINTY_METHOD_CONFIDENCE_LEVELS.get(method)
    if confidence_level is None:
        return (
            tuple(dict(row) for row in predictions),
            "suppressed_unreviewed_method",
            ("REVIEWED_UNCERTAINTY_METHOD_UNAVAILABLE",),
            None,
        )
    try:
        standard_errors = np.asarray(
            [float(row["yield_se_t_ha"]) for row in observation_evidence],
            dtype=float,
        )
        observed_jacobian = _parameter_jacobian(
            model_name,
            observed_n_rates,
            parameters,
        )
        if np.linalg.matrix_rank(observed_jacobian) < observed_jacobian.shape[1]:
            raise np.linalg.LinAlgError("rank-deficient parameter Jacobian")
        bread = np.linalg.inv(observed_jacobian.T @ observed_jacobian)
        observation_covariance = np.diag(standard_errors**2)
        parameter_covariance = (
            bread
            @ observed_jacobian.T
            @ observation_covariance
            @ observed_jacobian
            @ bread
        )
        prediction_n_rates = np.asarray(
            [float(row["n_rate_kg_ha"]) for row in predictions],
            dtype=float,
        )
        prediction_jacobian = _parameter_jacobian(
            model_name,
            prediction_n_rates,
            parameters,
        )
        prediction_variances = np.einsum(
            "ij,jk,ik->i",
            prediction_jacobian,
            parameter_covariance,
            prediction_jacobian,
        )
        if not np.isfinite(prediction_variances).all() or np.min(prediction_variances) < -1.0e-10:
            raise np.linalg.LinAlgError("invalid propagated variance")
        prediction_standard_errors = np.sqrt(np.maximum(prediction_variances, 0.0))
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return (
            tuple(dict(row) for row in predictions),
            "suppressed_numerically_unavailable",
            ("UNCERTAINTY_INTERVAL_NUMERICALLY_UNAVAILABLE",),
            None,
        )

    critical_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    bounded_rows: list[dict[str, float]] = []
    for row, standard_error in zip(
        predictions,
        prediction_standard_errors,
        strict=True,
    ):
        fitted_yield = float(row["predicted_yield_t_ha"])
        margin = critical_value * float(standard_error)
        bounded_rows.append(
            {
                **row,
                "confidence_lower_95pct_t_ha": fitted_yield - margin,
                "confidence_upper_95pct_t_ha": fitted_yield + margin,
                "fitted_mean_se_t_ha": float(standard_error),
                "confidence_level": confidence_level,
            }
        )
    return (
        tuple(bounded_rows),
        f"available_{method}",
        (),
        parameter_covariance,
    )


def _optimum_summary(
    model_name: str,
    parameters: Mapping[str, float],
    predictions: Sequence[Mapping[str, float]],
    *,
    observed_n_rates: np.ndarray,
    observed_min: float,
    observed_max: float,
    optimum_boundary_tolerance_n_kg_ha: float,
    flat_response_tolerance_t_ha: float,
    parameter_rank_full: bool,
) -> _OptimumSummary:
    predicted_values = [float(row["predicted_yield_t_ha"]) for row in predictions]
    observed_domain_peak = max(predicted_values)
    boundary_tolerance = optimum_boundary_tolerance_n_kg_ha
    response_tolerance = flat_response_tolerance_t_ha
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
        slope_tolerance = response_tolerance / max(span, 1.0)
        shape = (
            "increasing_linear"
            if slope > slope_tolerance
            else "decreasing_linear"
            if slope < -slope_tolerance
            else "flat_linear"
        )
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
    observation_evidence: Sequence[Mapping[str, Any]] | None = None,
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
    if observation_evidence is not None and len(observation_evidence) != len(n_rates):
        raise ValueError("observation_evidence must align one-to-one with N-rate observations")
    if len(n_rates) == len(yields):
        identity_uids: Sequence[str | None] = record_uids if record_uids is not None else (None,) * len(n_rates)
        identity_evidence: Sequence[Mapping[str, Any]] = (
            observation_evidence
            if observation_evidence is not None
            else ({},) * len(n_rates)
        )
        observations: object = tuple(
            sorted(
                (
                    (
                        record_uid,
                        n_rate,
                        yield_value,
                        dict(evidence),
                    )
                    for record_uid, n_rate, yield_value, evidence in zip(
                        identity_uids,
                        n_rates,
                        yields,
                        identity_evidence,
                        strict=True,
                    )
                ),
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
    uncertainty_status, uncertainty_method, uncertainty_basis, uncertainty_reasons = _uncertainty_gate(
        policy,
        tuple(observation_evidence or ()),
    )
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
    if policy.get("no_extrapolation") is not True:
        return _attempt(
            response_series_uid=response_series_uid,
            model_name=model_name,
            identity_payload=identity_payload,
            status="unsupported",
            reason_codes=("OBSERVED_DOMAIN_RESTRICTION_REQUIRED",),
            n_observations=n_observations,
            distinct_n_level_count=distinct_levels,
            observed_n_min_kg_ha=observed_min,
            observed_n_max_kg_ha=observed_max,
        )
    level_gate_reason = _model_level_gate_reason(model_name, distinct_levels, policy)
    if level_gate_reason is not None:
        return _attempt(
            response_series_uid=response_series_uid,
            model_name=model_name,
            identity_payload=identity_payload,
            status="unsupported",
            reason_codes=(level_gate_reason,),
            n_observations=n_observations,
            distinct_n_level_count=distinct_levels,
            observed_n_min_kg_ha=observed_min,
            observed_n_max_kg_ha=observed_max,
            uncertainty_status=uncertainty_status,
            uncertainty_method=uncertainty_method,
            uncertainty_evidence_basis=uncertainty_basis,
        )
    model_gate, model_gate_reason = _reviewed_model_gate(model_name, policy)
    if model_gate is None:
        return _attempt(
            response_series_uid=response_series_uid,
            model_name=model_name,
            identity_payload=identity_payload,
            status="unsupported",
            reason_codes=(model_gate_reason or "REVIEWED_MODEL_GATE_POLICY_UNAVAILABLE",),
            n_observations=n_observations,
            distinct_n_level_count=distinct_levels,
            observed_n_min_kg_ha=observed_min,
            observed_n_max_kg_ha=observed_max,
            uncertainty_status=uncertainty_status,
            uncertainty_method=uncertainty_method,
            uncertainty_evidence_basis=uncertainty_basis,
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
        tolerance=model_gate.optimizer_tolerance,
        gate=model_gate,
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
            model_gate_policy_id=model_gate.policy_id,
            uncertainty_status=uncertainty_status,
            uncertainty_method=uncertainty_method,
            uncertainty_evidence_basis=uncertainty_basis,
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
            model_gate_policy_id=model_gate.policy_id,
            uncertainty_status=uncertainty_status,
            uncertainty_method=uncertainty_method,
            uncertainty_evidence_basis=uncertainty_basis,
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
            model_gate_policy_id=model_gate.policy_id,
            uncertainty_status=uncertainty_status,
            uncertainty_method=uncertainty_method,
            uncertainty_evidence_basis=uncertainty_basis,
        )
    if (
        not model_gate.allow_boundary_parameters
        and _at_reviewed_parameter_boundary(parameters, model_gate)
    ):
        return _attempt(
            response_series_uid=response_series_uid,
            model_name=model_name,
            identity_payload=identity_payload,
            status="failed",
            reason_codes=("PARAMETER_AT_DISALLOWED_REVIEWED_BOUNDARY",),
            n_observations=n_observations,
            distinct_n_level_count=distinct_levels,
            observed_n_min_kg_ha=observed_min,
            observed_n_max_kg_ha=observed_max,
            residual_df=residual_df,
            parameters=parameter_map,
            model_gate_policy_id=model_gate.policy_id,
            uncertainty_status=uncertainty_status,
            uncertainty_method=uncertainty_method,
            uncertainty_evidence_basis=uncertainty_basis,
        )
    rss = float(np.sum((predicted_observed - y) ** 2))
    optimum_summary = _optimum_summary(
        model_name,
        parameter_map,
        predictions,
        observed_n_rates=x,
        observed_min=observed_min,
        observed_max=observed_max,
        optimum_boundary_tolerance_n_kg_ha=(
            model_gate.optimum_boundary_tolerance_n_kg_ha
        ),
        flat_response_tolerance_t_ha=model_gate.flat_response_tolerance_t_ha,
        parameter_rank_full=_parameter_rank_is_full(model_name, x, parameters),
    )
    if optimum_summary.curve_shape_class not in model_gate.reportable_shape_classes:
        return _attempt(
            response_series_uid=response_series_uid,
            model_name=model_name,
            identity_payload=identity_payload,
            status="failed",
            reason_codes=(
                *optimum_summary.reason_codes,
                "CURVE_SHAPE_OUTSIDE_REVIEWED_PLAUSIBILITY_POLICY",
            ),
            n_observations=n_observations,
            distinct_n_level_count=distinct_levels,
            observed_n_min_kg_ha=observed_min,
            observed_n_max_kg_ha=observed_max,
            residual_df=residual_df,
            parameters=parameter_map,
            curve_shape_class=optimum_summary.curve_shape_class,
            optimum_status=optimum_summary.optimum_status,
            model_gate_policy_id=model_gate.policy_id,
            uncertainty_status=uncertainty_status,
            uncertainty_method=uncertainty_method,
            uncertainty_evidence_basis=uncertainty_basis,
            predictions=predictions,
        )
    if uncertainty_status == "eligible_for_reviewed_method" and uncertainty_method:
        (
            predictions,
            uncertainty_status,
            executed_uncertainty_reasons,
        ) = _reported_se_delta_intervals(
            model_name,
            parameter_map,
            x,
            tuple(observation_evidence or ()),
            predictions,
            method=uncertainty_method,
        )
        uncertainty_reasons = (
            *uncertainty_reasons,
            *executed_uncertainty_reasons,
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
        tolerance=model_gate.optimizer_tolerance,
        gate=model_gate,
    )
    if grouped_prediction_rmse is None:
        reasons.append("GROUPED_PREDICTION_UNAVAILABLE")
    reasons.extend(uncertainty_reasons)
    return _attempt(
        response_series_uid=response_series_uid,
        model_name=model_name,
        identity_payload=identity_payload,
        model_gate_policy_id=model_gate.policy_id,
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
        uncertainty_status=uncertainty_status,
        uncertainty_method=uncertainty_method,
        uncertainty_evidence_basis=uncertainty_basis,
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
        observation_evidence = [
            {
                key: row.get(key)
                for key in (
                    "experimental_unit_status",
                    "mean_independence_review_id",
                    "mean_independence_status",
                    "replicate_count",
                    "replication_status",
                    "yield_se_status",
                    "yield_se_t_ha",
                )
            }
            for row in rows
        ]
        for model_name in selected_models:
            attempts.append(
                fit_candidate_model(
                    series_uid,
                    n_rates,
                    yields,
                    record_uids=record_uids,
                    observation_evidence=observation_evidence,
                    model_name=model_name,
                    policy=policy,
                )
            )
    return tuple(attempts)


def credible_model_attempts(attempts: Iterable[ModelAttempt]) -> tuple[ModelAttempt, ...]:
    """Return every fitted, identifiable candidate in deterministic roster order."""

    fitted = [
        attempt
        for attempt in attempts
        if attempt.status == "fitted"
        and "UNIDENTIFIABLE_SHAPE_PARAMETERS" not in attempt.reason_codes
    ]
    return tuple(
        sorted(
            fitted,
            key=lambda attempt: (
                MODEL_ORDER.index(attempt.model_name),
                attempt.model_attempt_uid,
            ),
        )
    )


def select_reportable_model(attempts: Iterable[ModelAttempt]) -> ModelAttempt | None:
    """Select one fitted candidate for the legacy ranked-selection policy."""

    fitted = credible_model_attempts(attempts)
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
        "model_gate_policy_id": attempt.model_gate_policy_id,
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
        "uncertainty_status": attempt.uncertainty_status,
        "uncertainty_method": attempt.uncertainty_method,
        "uncertainty_evidence_basis": list(attempt.uncertainty_evidence_basis),
        "predictions": [dict(row) for row in attempt.predictions],
    }


__all__ = [
    "MODEL_ORDER",
    "ModelAttempt",
    "credible_model_attempts",
    "evaluate_model",
    "fit_candidate_model",
    "fit_response_models",
    "model_attempt_record",
    "select_reportable_model",
]
