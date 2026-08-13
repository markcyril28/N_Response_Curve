from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
import hashlib
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

from ..data.provenance import stable_identifier, stable_json_sha256
from ..data.schema import CANONICAL_N_RATE_UNIT, CANONICAL_YIELD_UNIT
from ..reporting.plots import prediction_rows
from .models import (
    MODEL_ORDER,
    ModelAttempt,
    credible_model_attempts,
    evaluate_model,
    fit_response_models,
    model_attempt_record,
)
from .reviewed_methods import (
    ECONOMIC_DECISION_RULE,
    ECONOMIC_GRAIN_PRICE_TO_PER_TONNE,
    ECONOMIC_N_COST_UNIT,
    UNCERTAINTY_METHOD_CONFIDENCE_LEVELS,
    UNCERTAINTY_METHOD_REQUIRED_EVIDENCE,
    UNCERTAINTY_METHOD_SPECS,
)
from .values import finite_number


_ALL_CREDIBLE_POLICY = "all_credible_no_selection"
_ASYMPTOTE_REFERENCE_QUANTITIES = frozenset({"ceiling_level", "response_range"})
# The MOD-08 ceiling gate may only rest on an interval method that stays valid
# where the likelihood is a ridge in `(C, A)`. Neither is implemented yet, so a
# policy naming one is still refused at execution time rather than silently
# falling back to the delta-method variance the fitter already computes.
_ASYMPTOTE_INTERVAL_METHODS = frozenset(
    {"profile_likelihood", "design_respecting_bootstrap"}
)
_EXECUTABLE_ASYMPTOTE_INTERVAL_METHODS: frozenset[str] = frozenset()
_SEPARATE_BASELINE_POLICY = "separate_verified_classes"
_P_K_FIT_BLOCK_REASONS = frozenset(
    {
        "P_K_UNRESOLVED",
        "P_K_UNRESOLVED_EXCLUDED",
        "P_K_VARY_WITH_N",
        "P_K_VARY_WITH_N_EXCLUDED",
    }
)
_ORGANIC_FIT_BLOCK_REASONS = frozenset(
    {
        "ORGANIC_OR_BIOFERTILIZER",
        "ORGANIC_OR_BIOFERTILIZER_EXCLUDED",
    }
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _approved_scientific_policy_authority(
    policy: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    authority = policy.get("scientific_policy_authority")
    if not isinstance(authority, Mapping) or set(authority) != {
        "approved_by",
        "approved_on",
        "artifact_sha256",
    }:
        return None
    approved_by = authority.get("approved_by")
    approved_on = authority.get("approved_on")
    if (
        not isinstance(approved_by, str)
        or not approved_by.strip()
        or not isinstance(approved_on, str)
        or not approved_on.strip()
        or not _is_sha256(authority.get("artifact_sha256"))
    ):
        return None
    try:
        date.fromisoformat(approved_on)
    except ValueError:
        return None
    return authority


@dataclass(frozen=True)
class _DisagreementSummary:
    status: str
    curve_shape_class: str
    optimum_status: str
    maximum_reference_basis: str
    maximum_proximity_status: str
    materially_different: bool | None
    values: Mapping[str, float | None]
    ranges: Mapping[str, tuple[float, float] | None]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class CurveEvidenceResult:
    """Per-series model ledger, reporting-policy evidence, and observed-domain predictions."""

    reporting_policy: str
    model_attempts: tuple[ModelAttempt, ...]
    credible_attempts: tuple[ModelAttempt, ...]
    selected_attempts: tuple[ModelAttempt, ...]
    model_attempt_records: tuple[dict[str, Any], ...]
    series_evidence_rows: tuple[dict[str, Any], ...]
    curve_rows: tuple[dict[str, Any], ...]
    prediction_rows: tuple[dict[str, Any], ...]
    economic_optimum_rows: tuple[dict[str, Any], ...]
    efficiency_rows: tuple[dict[str, Any], ...]
    efficiency_operating_point_rows: tuple[dict[str, Any], ...]
    asymptote_support_rows: tuple[dict[str, Any], ...]
    asymptote_reporting_rows: tuple[dict[str, Any], ...]
    environmental_risk_rows: tuple[dict[str, Any], ...]


def _series_rows(records: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw_record in records:
        record = dict(raw_record)
        series_uid = record.get("response_series_uid")
        if record.get("series_status") != "resolved" or not isinstance(series_uid, str) or not series_uid:
            continue
        grouped.setdefault(series_uid, []).append(record)
    for rows in grouped.values():
        rows.sort(key=lambda row: str(row.get("record_uid", "")))
    return grouped


def _approved_efficiency_policy(
    policy: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    raw = policy.get("efficiency_policy")
    if not isinstance(raw, Mapping):
        return None
    if (
        raw.get("review_status") != "approved"
        or raw.get("metric_id")
        != "EFF-01-option-a-partial-factor-productivity"
        or raw.get("basis") != "observed_reviewed_treatment_mean"
        or raw.get("aggregation_level") != "record_by_n_level"
        or not isinstance(raw.get("policy_id"), str)
        or not str(raw["policy_id"]).strip()
    ):
        return None
    if "authority" in raw:
        return None
    authority = _approved_scientific_policy_authority(policy)
    if authority is None:
        return None
    return {**dict(raw), "authority": authority}


def _partial_factor_productivity_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Materialize EFF-01 Option A at the observed record-by-N-level grain."""

    efficiency_policy = _approved_efficiency_policy(policy)
    if efficiency_policy is None:
        return ()
    rows: list[dict[str, Any]] = []
    for record in records:
        record_uid_value = record.get("record_uid")
        series_uid_value = record.get("response_series_uid")
        record_uid = (
            record_uid_value.strip()
            if isinstance(record_uid_value, str)
            else ""
        )
        series_uid = (
            series_uid_value.strip()
            if isinstance(series_uid_value, str)
            else ""
        )
        n_rate = finite_number(record.get("n_rate_kg_ha"))
        yield_t_ha = finite_number(record.get("yield_t_ha"))
        eligibility_tier = record.get("series_eligibility_tier") or record.get(
            "eligibility_tier"
        )
        if (
            not record_uid
            or not series_uid
            or record.get("series_status") != "resolved"
            or eligibility_tier not in {"A", "B"}
            or record.get("analytical_record_status") != "included"
            or record.get("final_analytical_membership_status")
            not in {"included", "included_flagged"}
            or record.get("n_rate_parse_status") != "parsed"
            or record.get("yield_parse_status") != "parsed"
            or record.get("n_rate_unit_status")
            not in {"canonical_reviewed", "converted_reviewed"}
            or record.get("n_rate_canonical_unit") != CANONICAL_N_RATE_UNIT
            or not isinstance(record.get("n_rate_unit_review_id"), str)
            or not str(record["n_rate_unit_review_id"]).strip()
            or record.get("yield_unit_status")
            not in {"consistent", "kg_converted", "t_provided"}
            or record.get("yield_canonical_unit") != CANONICAL_YIELD_UNIT
            or n_rate is None
            or n_rate < 0.0
            or yield_t_ha is None
        ):
            continue
        grain_status = str(
            record.get("analysis_grain_status") or "unreviewed_observed_record"
        )
        if grain_status != "reviewed_treatment_mean":
            continue
        identity = {
            "metric_id": efficiency_policy["metric_id"],
            "efficiency_policy_id": efficiency_policy["policy_id"],
            "efficiency_policy_sha256": efficiency_policy["authority"][
                "artifact_sha256"
            ],
            "record_uid": record_uid,
            "response_series_uid": series_uid,
            "n_rate_kg_ha": n_rate,
            "yield_t_ha": yield_t_ha,
            "analysis_grain_status": grain_status,
        }
        common = {
            "efficiency_metric_uid": stable_identifier(
                "efficiency",
                tuple(identity.values()),
            ),
            **identity,
            "source_name": record.get("source_name"),
            "study_uid": record.get("study_uid"),
            "treatment_class": record.get("treatment_text_class"),
            "metric_name": "partial_factor_productivity",
            "formula": "yield_kg_ha / applied_n_kg_ha",
            "basis": efficiency_policy["basis"],
            "aggregation_level": efficiency_policy["aggregation_level"],
            "unit": "kg_grain_per_kg_n",
        }
        if n_rate == 0.0:
            rows.append(
                {
                    **common,
                    "partial_factor_productivity_kg_grain_per_kg_n": None,
                    "status": "withheld_zero_n_representation_unapproved",
                    "reason_codes": (
                        "EFF02_ZERO_N_REPRESENTATION_UNAPPROVED",
                    ),
                }
            )
            continue
        value = yield_t_ha * 1000.0 / n_rate
        if not math.isfinite(value):
            rows.append(
                {
                    **common,
                    "partial_factor_productivity_kg_grain_per_kg_n": None,
                    "status": "withheld_nonfinite_result",
                    "reason_codes": ("EFF01_NONFINITE_RESULT",),
                }
            )
            continue
        rows.append(
            {
                **common,
                "partial_factor_productivity_kg_grain_per_kg_n": value,
                "status": "computed",
                "reason_codes": (),
            }
        )
    return tuple(sorted(rows, key=lambda row: str(row["record_uid"])))


def _approved_efficiency_operating_point_policy(
    policy: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    raw = policy.get("efficiency_operating_point_policy")
    if not isinstance(raw, Mapping):
        return None
    retention = finite_number(raw.get("yield_retention_fraction"))
    marginal_gain = finite_number(
        raw.get("maximum_marginal_gain_t_ha_per_kg_n")
    )
    concordance = finite_number(
        raw.get("model_concordance_tolerance_n_kg_ha")
    )
    grid_points = raw.get("prediction_grid_points")
    configured_grid_points = policy.get("plot_grid_points")
    uncertainty_policy = policy.get("uncertainty_policy")
    uncertainty_method = (
        uncertainty_policy.get("method")
        if isinstance(uncertainty_policy, Mapping)
        else None
    )
    uncertainty_contract = (
        uncertainty_policy.get("method_contract")
        if isinstance(uncertainty_policy, Mapping)
        else None
    )
    uncertainty_evidence_basis = (
        uncertainty_policy.get("evidence_basis")
        if isinstance(uncertainty_policy, Mapping)
        else None
    )
    uncertainty_method_key = (
        uncertainty_method if isinstance(uncertainty_method, str) else ""
    )
    expected_uncertainty_contract = UNCERTAINTY_METHOD_SPECS.get(
        uncertainty_method_key,
    )
    required_uncertainty_evidence = UNCERTAINTY_METHOD_REQUIRED_EVIDENCE.get(
        uncertainty_method_key,
        (),
    )
    if "authority" in raw or (
        isinstance(uncertainty_policy, Mapping)
        and "authority" in uncertainty_policy
    ):
        return None
    authority = _approved_scientific_policy_authority(policy)
    if authority is None:
        return None
    if (
        raw.get("review_status") != "approved"
        or not isinstance(raw.get("policy_id"), str)
        or not str(raw["policy_id"]).strip()
        or retention is None
        or not 0.0 < retention <= 1.0
        or marginal_gain is None
        or marginal_gain < 0.0
        or concordance is None
        or concordance < 0.0
        or not isinstance(grid_points, int)
        or isinstance(grid_points, bool)
        or grid_points < 3
        or configured_grid_points != grid_points
        or raw.get("prediction_grid_domain") != "observed_n_domain"
        or raw.get("prediction_grid_spacing")
        != "linear_inclusive_endpoints"
        or raw.get("uncertainty_decision_rule")
        != "point_estimate_thresholds_with_validated_fitted_mean_interval_reporting"
        or not isinstance(uncertainty_policy, Mapping)
        or uncertainty_policy.get("review_status") != "approved"
        or not isinstance(uncertainty_policy.get("policy_id"), str)
        or not str(uncertainty_policy["policy_id"]).strip()
        or not uncertainty_method_key
        or expected_uncertainty_contract is None
        or not isinstance(uncertainty_contract, Mapping)
        or dict(uncertainty_contract) != dict(expected_uncertainty_contract)
        or not isinstance(uncertainty_evidence_basis, (tuple, list))
        or not all(
            isinstance(item, str) and bool(item.strip())
            for item in uncertainty_evidence_basis
        )
        or not set(required_uncertainty_evidence).issubset(
            set(uncertainty_evidence_basis)
        )
        or raw.get("marginal_gain_method")
        != "adjacent_prediction_grid_difference"
        or raw.get("zero_n_disposition")
        != "exclude_from_operating_point_search"
        or raw.get("uncertainty_disposition")
        != "require_available_for_all_credible_models"
    ):
        return None
    return {
        **dict(raw),
        "authority": authority,
        "uncertainty_policy_id": uncertainty_policy["policy_id"],
        "uncertainty_method": uncertainty_method_key,
        "uncertainty_confidence_level": (
            UNCERTAINTY_METHOD_CONFIDENCE_LEVELS[uncertainty_method_key]
        ),
        "uncertainty_required_evidence": required_uncertainty_evidence,
    }


def _efficiency_operating_point_rows(
    credible_attempts: Iterable[ModelAttempt],
    *,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Apply reviewed EFF-03 thresholds without supplying scientific defaults."""

    operating_policy = _approved_efficiency_operating_point_policy(policy)
    if operating_policy is None:
        return ()
    grouped: dict[str, list[ModelAttempt]] = {}
    for attempt in credible_attempts:
        grouped.setdefault(attempt.response_series_uid, []).append(attempt)
    rows: list[dict[str, Any]] = []
    retention = float(operating_policy["yield_retention_fraction"])
    maximum_gain = float(
        operating_policy["maximum_marginal_gain_t_ha_per_kg_n"]
    )
    concordance = float(
        operating_policy["model_concordance_tolerance_n_kg_ha"]
    )
    expected_grid_points = int(operating_policy["prediction_grid_points"])
    required_uncertainty_method = str(operating_policy["uncertainty_method"])
    required_confidence_level = float(
        operating_policy["uncertainty_confidence_level"]
    )
    required_interval_critical_value = statistics.NormalDist().inv_cdf(
        0.5 + required_confidence_level / 2.0
    )
    required_uncertainty_evidence = set(
        operating_policy["uncertainty_required_evidence"]
    )
    for series_uid, attempts in sorted(grouped.items()):
        model_rates: dict[str, float] = {}
        qualifying_rates_by_model: dict[str, set[float]] = {}
        model_uncertainty_statuses: dict[str, str] = {}
        model_uncertainty_methods: dict[str, str | None] = {}
        model_attempt_uids: dict[str, str | None] = {}
        model_input_snapshot_sha256s: dict[str, str | None] = {}
        model_policy_sha256s: dict[str, str | None] = {}
        model_prediction_grids: dict[str, tuple[float, ...]] = {}
        model_prediction_evidence_sha256s: dict[str, str] = {}
        model_prediction_intervals: dict[
            str, dict[float, dict[str, float]]
        ] = {}
        reason_codes: set[str] = set()
        for attempt in sorted(attempts, key=lambda item: item.model_name):
            model_name = attempt.model_name
            uncertainty_status = getattr(attempt, "uncertainty_status", "")
            uncertainty_method = getattr(attempt, "uncertainty_method", None)
            model_uncertainty_statuses[model_name] = uncertainty_status
            model_uncertainty_methods[model_name] = uncertainty_method
            model_attempt_uid = getattr(attempt, "model_attempt_uid", None)
            input_snapshot_sha256 = getattr(attempt, "input_snapshot_sha256", None)
            model_policy_sha256 = getattr(attempt, "model_policy_sha256", None)
            model_attempt_uids[model_name] = (
                model_attempt_uid if isinstance(model_attempt_uid, str) else None
            )
            model_input_snapshot_sha256s[model_name] = (
                input_snapshot_sha256
                if isinstance(input_snapshot_sha256, str)
                else None
            )
            model_policy_sha256s[model_name] = (
                model_policy_sha256
                if isinstance(model_policy_sha256, str)
                else None
            )
            expected_model_attempt_uid = (
                stable_identifier(
                    "model",
                    (
                        series_uid,
                        model_name,
                        input_snapshot_sha256,
                        model_policy_sha256,
                    ),
                )
                if _is_sha256(input_snapshot_sha256)
                and _is_sha256(model_policy_sha256)
                else None
            )
            if (
                not isinstance(model_attempt_uid, str)
                or not model_attempt_uid.strip()
                or not _is_sha256(input_snapshot_sha256)
                or not _is_sha256(model_policy_sha256)
                or model_attempt_uid != expected_model_attempt_uid
            ):
                reason_codes.add("MODEL_ATTEMPT_PROVENANCE_UNAVAILABLE")
                continue
            if (
                not isinstance(uncertainty_method, str)
                or not uncertainty_method.strip()
                or uncertainty_method != required_uncertainty_method
                or uncertainty_status != f"available_{uncertainty_method}"
            ):
                reason_codes.add("CREDIBLE_MODEL_UNCERTAINTY_UNAVAILABLE")
                continue
            evidence_basis = getattr(attempt, "uncertainty_evidence_basis", ())
            if (
                not isinstance(evidence_basis, (tuple, list))
                or not evidence_basis
                or not all(
                    isinstance(item, str) and bool(item.strip())
                    for item in evidence_basis
                )
                or not required_uncertainty_evidence.issubset(set(evidence_basis))
            ):
                reason_codes.add("CREDIBLE_MODEL_UNCERTAINTY_EVIDENCE_INVALID")
                continue
            supported_maximum = finite_number(attempt.supported_max_yield_t_ha)
            predictions: list[
                tuple[float, float, float, float, float, float]
            ] = []
            invalid_prediction_evidence = False
            for prediction in attempt.predictions:
                rate = finite_number(prediction.get("n_rate_kg_ha"))
                predicted_yield = finite_number(
                    prediction.get("predicted_yield_t_ha")
                )
                standard_error = finite_number(
                    prediction.get("fitted_mean_se_t_ha")
                )
                lower = finite_number(
                    prediction.get("confidence_lower_95pct_t_ha")
                )
                upper = finite_number(
                    prediction.get("confidence_upper_95pct_t_ha")
                )
                confidence_level = finite_number(
                    prediction.get("confidence_level")
                )
                if (
                    rate is None
                    or predicted_yield is None
                    or standard_error is None
                    or standard_error < 0.0
                    or lower is None
                    or upper is None
                    or lower > predicted_yield
                    or upper < predicted_yield
                    or not math.isclose(
                        predicted_yield - lower,
                        required_interval_critical_value * standard_error,
                        rel_tol=1.0e-9,
                        abs_tol=1.0e-12,
                    )
                    or not math.isclose(
                        upper - predicted_yield,
                        required_interval_critical_value * standard_error,
                        rel_tol=1.0e-9,
                        abs_tol=1.0e-12,
                    )
                    or confidence_level is None
                    or not math.isclose(
                        confidence_level,
                        required_confidence_level,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                ):
                    invalid_prediction_evidence = True
                    break
                predictions.append(
                    (
                        rate,
                        predicted_yield,
                        standard_error,
                        lower,
                        upper,
                        confidence_level,
                    )
                )
            predictions.sort(key=lambda item: item[0])
            prediction_grid = tuple(item[0] for item in predictions)
            if invalid_prediction_evidence:
                reason_codes.add("CREDIBLE_MODEL_UNCERTAINTY_EVIDENCE_INVALID")
                continue
            observed_n_min = finite_number(
                getattr(attempt, "observed_n_min_kg_ha", None)
            )
            observed_n_max = finite_number(
                getattr(attempt, "observed_n_max_kg_ha", None)
            )
            if (
                len(predictions) != expected_grid_points
                or len(set(prediction_grid)) != expected_grid_points
                or observed_n_min is None
                or observed_n_max is None
                or observed_n_max <= observed_n_min
            ):
                reason_codes.add("CREDIBLE_MODEL_PREDICTION_GRID_DEFINITION_INVALID")
                continue
            expected_prediction_grid = tuple(
                observed_n_min
                + index
                * (observed_n_max - observed_n_min)
                / (expected_grid_points - 1)
                for index in range(expected_grid_points)
            )
            if not all(
                math.isclose(
                    actual,
                    expected,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-9,
                )
                for actual, expected in zip(
                    prediction_grid,
                    expected_prediction_grid,
                )
            ):
                reason_codes.add("CREDIBLE_MODEL_PREDICTION_GRID_DEFINITION_INVALID")
                continue
            model_prediction_grids[model_name] = prediction_grid
            model_prediction_intervals[model_name] = {
                rate: {
                    "predicted_yield_t_ha": predicted_yield,
                    "fitted_mean_se_t_ha": standard_error,
                    "confidence_lower_95pct_t_ha": lower,
                    "confidence_upper_95pct_t_ha": upper,
                    "confidence_level": confidence_level,
                }
                for (
                    rate,
                    predicted_yield,
                    standard_error,
                    lower,
                    upper,
                    confidence_level,
                ) in predictions
            }
            model_prediction_evidence_sha256s[model_name] = stable_json_sha256(
                {
                    "model_attempt_uid": model_attempt_uid,
                    "supported_max_yield_t_ha": supported_maximum,
                    "observed_n_min_kg_ha": observed_n_min,
                    "observed_n_max_kg_ha": observed_n_max,
                    "uncertainty_status": uncertainty_status,
                    "uncertainty_method": uncertainty_method,
                    "uncertainty_evidence_basis": tuple(evidence_basis),
                    "predictions": tuple(
                        {
                            "n_rate_kg_ha": rate,
                            "predicted_yield_t_ha": predicted_yield,
                            "fitted_mean_se_t_ha": standard_error,
                            "confidence_lower_95pct_t_ha": lower,
                            "confidence_upper_95pct_t_ha": upper,
                            "confidence_level": confidence_level,
                        }
                        for (
                            rate,
                            predicted_yield,
                            standard_error,
                            lower,
                            upper,
                            confidence_level,
                        ) in predictions
                    ),
                }
            )
            candidate_rate: float | None = None
            qualifying_rates: set[float] = set()
            if supported_maximum is None or supported_maximum <= 0.0:
                reason_codes.add("SUPPORTED_MAXIMUM_UNAVAILABLE")
                continue
            for previous, current in zip(predictions, predictions[1:]):
                previous_rate, previous_yield = previous[:2]
                current_rate, current_yield = current[:2]
                if (
                    current_rate <= previous_rate
                    or current_rate <= 0.0
                ):
                    continue
                gain = (current_yield - previous_yield) / (
                    current_rate - previous_rate
                )
                if (
                    current_yield >= retention * supported_maximum
                    and gain <= maximum_gain
                ):
                    qualifying_rates.add(current_rate)
                    if candidate_rate is None:
                        candidate_rate = current_rate
            qualifying_rates_by_model[model_name] = qualifying_rates
            if candidate_rate is None:
                reason_codes.add("NO_IN_DOMAIN_RATE_SATISFIES_REVIEWED_THRESHOLDS")
            else:
                model_rates[model_name] = candidate_rate
        status = "unavailable"
        operating_rate: float | None = None
        prediction_grid: tuple[float, ...] | None = None
        common_input_snapshot = (
            len(model_input_snapshot_sha256s) == len(attempts)
            and len(set(model_input_snapshot_sha256s.values())) == 1
        )
        if not common_input_snapshot:
            reason_codes.add("CREDIBLE_MODEL_INPUT_SNAPSHOT_MISMATCH")
        common_model_policy = (
            len(model_policy_sha256s) == len(attempts)
            and len(set(model_policy_sha256s.values())) == 1
        )
        if not common_model_policy:
            reason_codes.add("CREDIBLE_MODEL_POLICY_SNAPSHOT_MISMATCH")
        if len(model_prediction_grids) == len(attempts) and model_prediction_grids:
            distinct_grids = set(model_prediction_grids.values())
            if len(distinct_grids) == 1:
                prediction_grid = next(iter(distinct_grids))
            else:
                reason_codes.add("CREDIBLE_MODEL_PREDICTION_GRID_MISMATCH")
        prediction_grid_sha256 = (
            stable_json_sha256(
                {
                    "response_series_uid": series_uid,
                    "n_rates_kg_ha": prediction_grid,
                }
            )
            if prediction_grid is not None
            else None
        )
        if (
            prediction_grid is not None
            and len(model_rates) == len(attempts)
            and model_rates
            and common_input_snapshot
            and common_model_policy
        ):
            rate_values = tuple(model_rates.values())
            if max(rate_values) - min(rate_values) <= concordance:
                common_rates = set.intersection(
                    *(qualifying_rates_by_model[model_name] for model_name in model_rates)
                )
                if common_rates:
                    status = "available"
                    operating_rate = min(common_rates)
                else:
                    reason_codes.add(
                        "NO_COMMON_RATE_SATISFIES_ALL_CREDIBLE_MODELS"
                    )
            else:
                reason_codes.add("CREDIBLE_MODEL_OPERATING_POINT_DISAGREEMENT")
        selected_prediction_intervals = (
            {
                model_name: intervals[operating_rate]
                for model_name, intervals in sorted(
                    model_prediction_intervals.items()
                )
            }
            if operating_rate is not None
            else {}
        )
        identity = {
            "response_series_uid": series_uid,
            "efficiency_operating_point_policy_id": operating_policy["policy_id"],
            "efficiency_operating_point_policy_sha256": operating_policy["authority"][
                "artifact_sha256"
            ],
            "uncertainty_policy_id": operating_policy["uncertainty_policy_id"],
            "uncertainty_method": required_uncertainty_method,
            "prediction_grid_sha256": prediction_grid_sha256,
            "model_attempt_uids": dict(sorted(model_attempt_uids.items())),
            "model_input_snapshot_sha256s": dict(
                sorted(model_input_snapshot_sha256s.items())
            ),
            "model_policy_sha256s": dict(sorted(model_policy_sha256s.items())),
            "model_prediction_evidence_sha256s": dict(
                sorted(model_prediction_evidence_sha256s.items())
            ),
            "status": status,
            "operating_point_n_kg_ha": operating_rate,
            "model_prediction_intervals_at_operating_point": (
                selected_prediction_intervals
            ),
            "reason_codes": tuple(sorted(reason_codes)),
        }
        rows.append(
            {
                "efficiency_operating_point_uid": stable_identifier(
                    "efficiency-operating-point",
                    identity,
                ),
                **identity,
                "basis": (
                    "smallest_prediction_grid_rate_satisfying_all_credible_models"
                ),
                "yield_retention_fraction": retention,
                "maximum_marginal_gain_t_ha_per_kg_n": maximum_gain,
                "marginal_gain_method": operating_policy["marginal_gain_method"],
                "model_concordance_tolerance_n_kg_ha": concordance,
                "prediction_grid_points": expected_grid_points,
                "prediction_grid_domain": operating_policy[
                    "prediction_grid_domain"
                ],
                "prediction_grid_spacing": operating_policy[
                    "prediction_grid_spacing"
                ],
                "uncertainty_disposition": operating_policy[
                    "uncertainty_disposition"
                ],
                "uncertainty_decision_rule": operating_policy[
                    "uncertainty_decision_rule"
                ],
                "model_specific_operating_points_n_kg_ha": dict(
                    sorted(model_rates.items())
                ),
                "model_uncertainty_statuses": dict(
                    sorted(model_uncertainty_statuses.items())
                ),
                "model_uncertainty_methods": dict(
                    sorted(model_uncertainty_methods.items())
                ),
            }
        )
    return tuple(rows)


def _approved_asymptote_support_policy(
    policy: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    raw = policy.get("asymptote_support_policy")
    if not isinstance(raw, Mapping):
        return None
    numeric_fields = (
        "minimum_in_domain_attainment_fraction",
        "maximum_asymptote_relative_se",
        "maximum_influence_relative_shift",
        "maximum_credible_model_relative_difference",
    )
    values = {field: finite_number(raw.get(field)) for field in numeric_fields}
    fold_count = raw.get("minimum_influence_fold_count")
    authority = policy.get("scientific_policy_authority")
    artifact_sha256 = (
        authority.get("artifact_sha256")
        if isinstance(authority, Mapping)
        else None
    )
    if (
        raw.get("review_status") != "approved"
        or not isinstance(raw.get("policy_id"), str)
        or not str(raw["policy_id"]).strip()
        or values["minimum_in_domain_attainment_fraction"] is None
        or not 0.0 < float(values["minimum_in_domain_attainment_fraction"]) <= 1.0
        or any(
            values[field] is None or float(values[field]) < 0.0
            for field in numeric_fields[1:]
        )
        or not isinstance(fold_count, int)
        or isinstance(fold_count, bool)
        or fold_count < 1
        or raw.get("maximum_associated_n_basis")
        != "smallest_prediction_grid_rate_meeting_attainment_threshold"
        or not isinstance(authority, Mapping)
        or not isinstance(authority.get("approved_by"), str)
        or not str(authority["approved_by"]).strip()
        or not isinstance(authority.get("approved_on"), str)
        or not str(authority["approved_on"]).strip()
        or not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in artifact_sha256
        )
    ):
        return None
    return {**dict(raw), **values, "authority": authority}


def _asymptote_support_rows(
    credible_attempts: Iterable[ModelAttempt],
    *,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Evaluate every reviewed MOD-08 support criterion for asymptotic fits."""

    support_policy = _approved_asymptote_support_policy(policy)
    if support_policy is None:
        return ()
    grouped: dict[str, list[ModelAttempt]] = {}
    for attempt in credible_attempts:
        grouped.setdefault(attempt.response_series_uid, []).append(attempt)
    rows: list[dict[str, Any]] = []
    for series_uid, attempts in sorted(grouped.items()):
        asymptotic = next(
            (attempt for attempt in attempts if attempt.model_name == "mitscherlich"),
            None,
        )
        if asymptotic is None:
            continue
        reasons: set[str] = set()
        asymptote = finite_number(asymptotic.fitted_asymptote_yield_t_ha)
        observed_peak = finite_number(
            asymptotic.predicted_observed_domain_peak_yield_t_ha
        )
        attainment = (
            observed_peak / asymptote
            if observed_peak is not None and asymptote not in {None, 0.0}
            else None
        )
        if (
            attainment is None
            or attainment
            < float(support_policy["minimum_in_domain_attainment_fraction"])
        ):
            reasons.add("ASYMPTOTE_IN_DOMAIN_ATTAINMENT_INSUFFICIENT")
        variance = finite_number(
            asymptotic.feature_variances.get("fitted_asymptote_yield_t_ha")
        )
        relative_se = (
            math.sqrt(variance) / abs(asymptote)
            if variance is not None
            and variance > 0.0
            and asymptote not in {None, 0.0}
            else None
        )
        if (
            relative_se is None
            or relative_se > float(support_policy["maximum_asymptote_relative_se"])
        ):
            reasons.add("ASYMPTOTE_PARAMETER_UNCERTAINTY_UNSUPPORTED")
        influence_shift = finite_number(
            asymptotic.asymptote_influence_max_relative_shift
        )
        if (
            influence_shift is None
            or asymptotic.asymptote_influence_fold_count
            < int(support_policy["minimum_influence_fold_count"])
            or influence_shift
            > float(support_policy["maximum_influence_relative_shift"])
        ):
            reasons.add("ASYMPTOTE_INFLUENCE_UNSTABLE")
        comparator_maxima = tuple(
            value
            for attempt in attempts
            if attempt.model_name != "mitscherlich"
            if (value := finite_number(attempt.supported_max_yield_t_ha)) is not None
        )
        concordance = (
            max(abs(value - asymptote) / abs(asymptote) for value in comparator_maxima)
            if comparator_maxima and asymptote not in {None, 0.0}
            else None
        )
        if (
            concordance is None
            or concordance
            > float(
                support_policy["maximum_credible_model_relative_difference"]
            )
        ):
            reasons.add("CREDIBLE_MODEL_ASYMPTOTE_CONCORDANCE_UNSUPPORTED")
        associated_prediction: Mapping[str, Any] | None = None
        if asymptote is not None:
            threshold = (
                float(support_policy["minimum_in_domain_attainment_fraction"])
                * asymptote
            )
            associated_prediction = next(
                (
                    row
                    for row in sorted(
                        asymptotic.predictions,
                        key=lambda row: float(row["n_rate_kg_ha"]),
                    )
                    if float(row["predicted_yield_t_ha"]) >= threshold
                ),
                None,
            )
        if associated_prediction is None:
            reasons.add("MAXIMUM_ASSOCIATED_N_UNAVAILABLE")
        status = "supported" if not reasons else "unsupported"
        identity = {
            "response_series_uid": series_uid,
            "model_attempt_uid": asymptotic.model_attempt_uid,
            "asymptote_support_policy_id": support_policy["policy_id"],
            "asymptote_support_policy_sha256": support_policy["authority"][
                "artifact_sha256"
            ],
        }
        rows.append(
            {
                "asymptote_support_uid": stable_identifier(
                    "asymptote-support",
                    tuple(identity.values()),
                ),
                **identity,
                "status": status,
                "attainable_yield_t_ha": (
                    float(associated_prediction["predicted_yield_t_ha"])
                    if status == "supported" and associated_prediction is not None
                    else None
                ),
                "attainable_yield_basis": "in_domain_fitted_attainment_threshold",
                "maximum_associated_n_kg_ha": (
                    float(associated_prediction["n_rate_kg_ha"])
                    if status == "supported" and associated_prediction is not None
                    else None
                ),
                "maximum_associated_n_basis": support_policy[
                    "maximum_associated_n_basis"
                ],
                "in_domain_attainment_fraction": attainment,
                "asymptote_relative_se": relative_se,
                "asymptote_influence_max_relative_shift": influence_shift,
                "asymptote_influence_fold_count": (
                    asymptotic.asymptote_influence_fold_count
                ),
                "credible_model_maximum_relative_difference": concordance,
                "reason_codes": tuple(sorted(reasons)),
            }
        )
    return tuple(rows)


def _promote_supported_asymptotes(
    credible_attempts: Iterable[ModelAttempt],
    support_rows: Iterable[Mapping[str, Any]],
) -> tuple[ModelAttempt, ...]:
    """Apply positive MOD-08 results without weakening the conservative hold."""

    supported_ids = {
        str(row.get("model_attempt_uid") or "")
        for row in support_rows
        if row.get("status") == "supported"
    }
    promoted: list[ModelAttempt] = []
    for attempt in credible_attempts:
        if attempt.model_attempt_uid not in supported_ids:
            promoted.append(attempt)
            continue
        promoted.append(
            replace(
                attempt,
                supported_max_yield_t_ha=attempt.fitted_asymptote_yield_t_ha,
                maximum_reference_basis="supported_attainable_asymptote",
                maximum_proximity_status="SUPPORTED_ASYMPTOTE",
                reason_codes=tuple(
                    sorted(
                        (
                            set(attempt.reason_codes)
                            - {"ASYMPTOTE_SUPPORT_GATE_NOT_APPROVED"}
                        )
                        | {"ASYMPTOTE_SUPPORT_APPROVED"}
                    )
                ),
            )
        )
    return tuple(promoted)


def _approved_asymptote_reporting_policy(
    policy: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    raw = policy.get("asymptote_reporting_policy")
    if not isinstance(raw, Mapping):
        return None
    fraction = finite_number(raw.get("asymptote_fraction"))
    authority = policy.get("scientific_policy_authority")
    artifact_sha256 = (
        authority.get("artifact_sha256")
        if isinstance(authority, Mapping)
        else None
    )
    if (
        raw.get("review_status") != "approved"
        or not isinstance(raw.get("policy_id"), str)
        or not str(raw["policy_id"]).strip()
        or fraction is None
        or not 0.0 < fraction < 1.0
        or raw.get("reference_quantity") not in {"ceiling_level", "response_range"}
        or raw.get("rate_label") != "N at q% of asymptote"
        or not isinstance(raw.get("uncertainty_method"), str)
        or not str(raw["uncertainty_method"]).strip()
        or raw.get("scope") != "supported_mitscherlich_asymptote"
        or not isinstance(authority, Mapping)
        or not isinstance(authority.get("approved_by"), str)
        or not str(authority["approved_by"]).strip()
        or not isinstance(authority.get("approved_on"), str)
        or not str(authority["approved_on"]).strip()
        or not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in artifact_sha256
        )
    ):
        return None
    return {**dict(raw), "authority": authority}


def _asymptote_reporting_rows(
    credible_attempts: Iterable[ModelAttempt],
    *,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Materialize MOD-07 only after MOD-08 support and uncertainty are present."""

    reporting_policy = _approved_asymptote_reporting_policy(policy)
    if reporting_policy is None:
        return ()
    fraction = float(reporting_policy["asymptote_fraction"])
    reference_quantity = str(reporting_policy["reference_quantity"])
    rows: list[dict[str, Any]] = []
    for attempt in sorted(
        (item for item in credible_attempts if item.model_name == "mitscherlich"),
        key=lambda item: item.response_series_uid,
    ):
        reasons: set[str] = set()
        candidate_rate: float | None = None
        asymptote = finite_number(attempt.fitted_asymptote_yield_t_ha)
        supported = finite_number(attempt.supported_max_yield_t_ha)
        amplitude = finite_number(attempt.parameters.get("amplitude"))
        rate = finite_number(attempt.parameters.get("rate"))
        observed_min = finite_number(attempt.observed_n_min_kg_ha)
        observed_max = finite_number(attempt.observed_n_max_kg_ha)
        if supported is None or asymptote is None:
            reasons.add("MOD08_SUPPORTED_ASYMPTOTE_REQUIRED")
        elif (
            amplitude is None
            or amplitude <= 0.0
            or rate is None
            or rate <= 0.0
            or observed_min is None
            or observed_max is None
        ):
            reasons.add("ASYMPTOTE_FRACTION_RATE_UNIDENTIFIABLE")
        else:
            # `ceiling_level` solves mu(N) = q * (l + A); `response_range` solves
            # A * (1 - exp(-rN)) = q * A, which reduces to the scale-free
            # -ln(1 - q) / r and cannot degenerate.
            ratio = (
                asymptote * (1.0 - fraction) / amplitude
                if reference_quantity == "ceiling_level"
                else 1.0 - fraction
            )
            if ratio <= 0.0:
                reasons.add("ASYMPTOTE_FRACTION_RATE_UNIDENTIFIABLE")
            else:
                solved_rate = -math.log(ratio) / rate
                # Both domain edges refuse. Clamping the lower edge up to
                # observed_min would publish the domain floor as though the
                # criterion had been solved there, paired with the standard
                # error of the unclamped solution -- an interval that does not
                # cover the quantity actually reported.
                if solved_rate < observed_min:
                    reasons.add("ASYMPTOTE_FRACTION_ALREADY_ATTAINED_AT_DOMAIN_FLOOR")
                elif solved_rate > observed_max:
                    reasons.add("ASYMPTOTE_FRACTION_NOT_REACHED_IN_DOMAIN")
                else:
                    candidate_rate = solved_rate
        variance = finite_number(
            attempt.feature_variances.get(
                "asymptote_fraction_reporting_n_kg_ha"
            )
        )
        if variance is None or variance <= 0.0:
            reasons.add("ASYMPTOTE_FRACTION_RATE_UNCERTAINTY_UNAVAILABLE")
        status = "available" if candidate_rate is not None and not reasons else "unavailable"
        identity = {
            "response_series_uid": attempt.response_series_uid,
            "model_attempt_uid": attempt.model_attempt_uid,
            "asymptote_reporting_policy_id": reporting_policy["policy_id"],
            "asymptote_reporting_policy_sha256": reporting_policy["authority"][
                "artifact_sha256"
            ],
        }
        rows.append(
            {
                "asymptote_reporting_uid": stable_identifier(
                    "asymptote-reporting",
                    tuple(identity.values()),
                ),
                **identity,
                "status": status,
                "rate_label": reporting_policy["rate_label"],
                "asymptote_fraction": fraction,
                "asymptote_reference_quantity": reference_quantity,
                "asymptote_fraction_n_kg_ha": (
                    candidate_rate if status == "available" else None
                ),
                "asymptote_fraction_n_se_kg_ha": (
                    math.sqrt(variance)
                    if status == "available" and variance is not None
                    else None
                ),
                "uncertainty_method": reporting_policy["uncertainty_method"],
                "fitted_asymptote_yield_t_ha": asymptote,
                "supported_max_yield_t_ha": supported,
                "basis": "analytic_mitscherlich_fraction_of_asymptote",
                "reason_codes": tuple(sorted(reasons)),
            }
        )
    return tuple(rows)


def _environmental_risk_rows(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Materialize EFF-04 Option B as explicitly noninferential series flags."""

    output: list[dict[str, Any]] = []
    for series_uid, rows in _series_rows(records).items():
        reviewed = [
            row
            for row in rows
            if row.get("analysis_grain_status") == "reviewed_treatment_mean"
            and finite_number(row.get("n_rate_kg_ha")) is not None
            and finite_number(row.get("yield_t_ha")) is not None
        ]
        levels = sorted(
            {
                float(row["n_rate_kg_ha"]): float(row["yield_t_ha"])
                for row in reviewed
            }.items()
        )
        if len(levels) < 2:
            continue
        highest_n, highest_yield = levels[-1]
        lower_max_yield = max(yield_value for _, yield_value in levels[:-1])
        high_n_exposure = any(bool(row.get("is_high_n")) for row in reviewed)
        high_n_decline = highest_yield < lower_max_yield
        severe_stress = any(
            row.get("severe_stress_status") == "verified_present"
            for row in reviewed
        )
        reasons = ["NO_CAUSAL_ENVIRONMENTAL_CLAIM"]
        if high_n_exposure:
            reasons.append("OBSERVED_HIGH_N_EXPOSURE")
        if high_n_decline:
            reasons.append("OBSERVED_YIELD_DECLINE_AT_HIGHEST_N")
        if severe_stress:
            reasons.append("SOURCE_VERIFIED_SEVERE_STRESS")
        identity = {
            "response_series_uid": series_uid,
            "highest_observed_n_kg_ha": highest_n,
            "highest_n_yield_t_ha": highest_yield,
            "maximum_lower_n_yield_t_ha": lower_max_yield,
            "high_n_exposure_flag": high_n_exposure,
            "yield_decline_at_highest_n_flag": high_n_decline,
            "source_verified_severe_stress_flag": severe_stress,
        }
        output.append(
            {
                "environmental_risk_uid": stable_identifier(
                    "environmental_risk",
                    identity,
                ),
                **identity,
                "metric_id": "EFF-04-option-b-descriptive-risk-flags",
                "claim_class": "descriptive_noninferential",
                "status": (
                    "flagged"
                    if high_n_exposure or high_n_decline or severe_stress
                    else "no_flag_observed"
                ),
                "reason_codes": tuple(sorted(reasons)),
            }
        )
    output.sort(key=lambda row: str(row["environmental_risk_uid"]))
    return tuple(output)


def _one_value(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    values = {str(row[key]) for row in rows if row.get(key) not in {None, ""}}
    if len(values) == 1:
        return next(iter(values))
    return None


def _study_uid(rows: Sequence[Mapping[str, Any]], source_name: str) -> str | None:
    canonical = _one_value(rows, "study_uid")
    if canonical is not None:
        return str(canonical)
    source_study_id = _one_value(rows, "study_id")
    if source_study_id is None:
        return None
    payload = f"{source_name}\0{source_study_id}".encode("utf-8")
    return f"study_{hashlib.sha256(payload).hexdigest()[:24]}"


def _record_reason_codes(record: Mapping[str, Any]) -> set[str]:
    reasons: set[str] = set()
    for key in (
        "eligibility_reason_codes",
        "series_eligibility_reason_codes",
        "series_reason_codes",
    ):
        raw = record.get(key, ())
        if isinstance(raw, (list, tuple, set, frozenset)):
            reasons.update(str(value) for value in raw)
    return reasons


def _series_fit_exclusion_reasons(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    reasons = {reason for row in rows for reason in _record_reason_codes(row)}
    exclusions: set[str] = set()
    if reasons.intersection(_P_K_FIT_BLOCK_REASONS):
        exclusions.add("NUTRIENT_CONSTANCY_NOT_VERIFIED_FOR_FITTING")
    if reasons.intersection(_ORGANIC_FIT_BLOCK_REASONS):
        exclusions.add("ORGANIC_OR_BIOFERTILIZER_FITTING_NOT_APPROVED")
    p_statuses = {
        row.get("series_p_constant")
        for row in rows
        if "series_p_constant" in row
    }
    k_statuses = {
        row.get("series_k_constant")
        for row in rows
        if "series_k_constant" in row
    }
    if p_statuses and p_statuses != {True}:
        exclusions.add("NUTRIENT_CONSTANCY_NOT_VERIFIED_FOR_FITTING")
    if k_statuses and k_statuses != {True}:
        exclusions.add("NUTRIENT_CONSTANCY_NOT_VERIFIED_FOR_FITTING")
    if any(
        bool(row.get("organic_fertilizer_present"))
        or bool(row.get("biofertilizer_present"))
        or bool(row.get("series_organic_or_biofertilizer_present"))
        for row in rows
    ):
        exclusions.add("ORGANIC_OR_BIOFERTILIZER_FITTING_NOT_APPROVED")
    return tuple(sorted(exclusions))


def _row_fit_exclusion_reasons(
    record: Mapping[str, Any],
    *,
    primary_only: bool,
) -> tuple[str, ...]:
    reasons: set[str] = set()
    series_uid = record.get("response_series_uid")
    if (
        record.get("series_status") != "resolved"
        or not isinstance(series_uid, str)
        or not series_uid
    ):
        reasons.add("RESPONSE_SERIES_NOT_RESOLVED")
    tier = record.get("series_eligibility_tier") or record.get("eligibility_tier")
    if tier not in {"A", "B"}:
        reasons.add("RESPONSE_SERIES_NOT_FIT_ELIGIBLE")
    if primary_only and tier != "A":
        reasons.add("RESPONSE_SERIES_OUTSIDE_PRIMARY_FIT_TIER")
    role = record.get("treatment_fit_role")
    if role != "curve_candidate":
        reasons.add(
            "COMPARISON_TREATMENT_EXCLUDED_FROM_FIT"
            if role == "comparison_only"
            else "TREATMENT_FIT_ROLE_NOT_VERIFIED"
        )
    if role == "curve_candidate" and not (
        record.get("treatment_class_normalization_status") == "mapped_reviewed"
        and isinstance(record.get("treatment_class_review_id"), str)
        and bool(str(record.get("treatment_class_review_id")).strip())
        and record.get(
            "treatment_fit_role_provenance_status",
            "reviewed_lookup_bound",
        )
        == "reviewed_lookup_bound"
    ):
        reasons.add("TREATMENT_FIT_ROLE_NOT_REVIEW_BOUND")
    treatment_class = str(record.get("treatment_text_class") or "")
    if treatment_class == "FP":
        reasons.add("FARMER_PRACTICE_COMPARISON_EXCLUDED_FROM_FIT")
    if treatment_class == "unresolved":
        reasons.add("TREATMENT_CLASS_UNRESOLVED")
    if record.get("nutrient_control_class") == "absolute_control":
        reasons.add("ABSOLUTE_CONTROL_RESERVED_FOR_SEPARATE_BASELINE")
    treatment_review_status = record.get(
        "treatment_classification_status",
        record.get("treatment_class_review_status"),
    )
    if treatment_review_status in {"pending", "review_required", "unresolved"}:
        reasons.add("TREATMENT_CLASS_REVIEW_REQUIRED")
    return tuple(sorted(reasons))


def _fit_exclusions(
    records: Sequence[Mapping[str, Any]],
    *,
    primary_only: bool,
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        series_uid = record.get("response_series_uid")
        if isinstance(series_uid, str) and series_uid:
            grouped.setdefault(series_uid, []).append(record)
    exclusions: dict[str, tuple[str, ...]] = {}
    series_reasons_by_uid = {
        series_uid: set(_series_fit_exclusion_reasons(rows))
        for series_uid, rows in grouped.items()
    }
    for record in records:
        record_uid = str(record.get("record_uid") or "")
        if not record_uid:
            continue
        series_uid = record.get("response_series_uid")
        series_reasons = (
            series_reasons_by_uid.get(series_uid, set())
            if isinstance(series_uid, str)
            else set()
        )
        reasons = series_reasons | set(
            _row_fit_exclusion_reasons(record, primary_only=primary_only)
        )
        if reasons:
            exclusions[record_uid] = tuple(sorted(reasons))
    return exclusions


def _descriptive_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        row
        for row in rows
        if row.get("treatment_fit_role") == "curve_candidate"
        and str(row.get("treatment_text_class") or "") not in {"FP", "unresolved"}
        and row.get("nutrient_control_class") != "absolute_control"
    )


def _verified_baseline_yields(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_class: str,
    zero_tolerance: float,
) -> list[float]:
    flag_name = {
        "zero_n_with_pk": "is_zero_n_with_pk",
        "absolute_control": "is_absolute_control",
    }[baseline_class]
    return [
        yield_value
        for row in rows
        if row.get("nutrient_control_class") == baseline_class
        and row.get(flag_name) is True
        and row.get("treatment_classification_status") == "resolved"
        and row.get("treatment_class_normalization_status") == "mapped_reviewed"
        and isinstance(row.get("treatment_class_review_id"), str)
        and bool(str(row.get("treatment_class_review_id")).strip())
        and (n_rate := finite_number(row.get("n_rate_kg_ha"))) is not None
        and abs(n_rate) <= zero_tolerance
        and (yield_value := finite_number(row.get("yield_t_ha"))) is not None
    ]


def _baseline_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    response_maximum: float,
    zero_tolerance: float,
    enabled: bool,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    result: dict[str, Any] = {
        "baseline_response_status": (
            "separate_verified_classes"
            if enabled
            else "disabled_without_separate_verified_class_policy"
        ),
        "yield_at_zero_n_t_ha": None,
        "yield_response_above_zero_n_t_ha": None,
    }
    reasons: set[str] = set()
    for baseline_class, label in (
        ("zero_n_with_pk", "zero_n_with_pk"),
        ("absolute_control", "absolute_control"),
    ):
        yields = (
            _verified_baseline_yields(
                rows,
                baseline_class=baseline_class,
                zero_tolerance=zero_tolerance,
            )
            if enabled
            else []
        )
        available = bool(yields)
        result[f"{label}_baseline_status"] = (
            "available"
            if available
            else "unavailable_no_verified_member"
            if enabled
            else "disabled"
        )
        result[f"yield_at_{label}_t_ha"] = (
            float(statistics.fmean(yields)) if available else None
        )
        result[f"yield_response_above_{label}_t_ha"] = (
            float(response_maximum - statistics.fmean(yields))
            if available
            else None
        )
        if enabled and not available:
            reasons.add(f"{label.upper()}_BASELINE_UNAVAILABLE")
    if not enabled:
        reasons.add("SEPARATE_VERIFIED_BASELINE_POLICY_NOT_ENABLED")
    return result, tuple(sorted(reasons))


def _numeric_range(
    attempts: Sequence[ModelAttempt],
    field_name: str,
) -> tuple[float, float] | None:
    values = [
        float(value)
        for attempt in attempts
        if (value := getattr(attempt, field_name)) is not None
    ]
    if len(values) != len(attempts) or not values:
        return None
    return min(values), max(values)


def _common_numeric_value(
    attempts: Sequence[ModelAttempt],
    field_name: str,
) -> float | None:
    values = [getattr(attempt, field_name) for attempt in attempts]
    if values and all(value == values[0] for value in values):
        return float(values[0]) if values[0] is not None else None
    return None


def _reviewed_disagreement_tolerances(
    policy: Mapping[str, Any],
) -> Mapping[str, float] | None:
    raw_policy = policy.get("material_disagreement_policy")
    if not isinstance(raw_policy, Mapping) or raw_policy.get("review_status") != "approved":
        return None
    if not isinstance(raw_policy.get("policy_id"), str) or not str(raw_policy["policy_id"]).strip():
        return None
    raw_tolerances = raw_policy.get("tolerances")
    required = {
        "agronomic_optimum_n_kg_ha",
        "plateau_onset_n_kg_ha",
        "supported_max_yield_t_ha",
    }
    if not isinstance(raw_tolerances, Mapping) or set(raw_tolerances) != required:
        return None
    tolerances: dict[str, float] = {}
    for name in sorted(required):
        value = raw_tolerances[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            return None
        tolerances[name] = float(value)
    return tolerances


def _disagreement_summary(
    credible: Sequence[ModelAttempt],
    *,
    policy: Mapping[str, Any],
) -> _DisagreementSummary:
    numeric_fields = (
        "agronomic_optimum_n_kg_ha",
        "plateau_onset_n_kg_ha",
        "predicted_max_yield_t_ha",
        "predicted_observed_domain_peak_yield_t_ha",
        "finite_maximum_yield_t_ha",
        "fitted_asymptote_yield_t_ha",
        "supported_max_yield_t_ha",
    )
    if not credible:
        return _DisagreementSummary(
            status="no_credible_candidate",
            curve_shape_class="unavailable",
            optimum_status="unavailable",
            maximum_reference_basis="none",
            maximum_proximity_status="unavailable",
            materially_different=None,
            values={name: None for name in numeric_fields},
            ranges={name: None for name in numeric_fields},
            reason_codes=("NO_CREDIBLE_MODEL_CANDIDATE",),
        )
    ranges = {name: _numeric_range(credible, name) for name in numeric_fields}
    values = {name: _common_numeric_value(credible, name) for name in numeric_fields}
    if len(credible) == 1:
        attempt = credible[0]
        return _DisagreementSummary(
            status="single_credible_candidate",
            curve_shape_class=str(attempt.curve_shape_class or "unavailable"),
            optimum_status=attempt.optimum_status,
            maximum_reference_basis=attempt.maximum_reference_basis,
            maximum_proximity_status=attempt.maximum_proximity_status,
            materially_different=False,
            values=values,
            ranges=ranges,
            reason_codes=("ALL_CREDIBLE_MODELS_REPORTED_WITHOUT_SELECTION",),
        )

    shapes = {str(attempt.curve_shape_class or "unavailable") for attempt in credible}
    stable_shape = len(shapes) == 1
    conclusion_signatures = {
        (
            attempt.optimum_status,
            attempt.maximum_reference_basis,
            attempt.agronomic_optimum_n_kg_ha is not None,
            attempt.plateau_onset_n_kg_ha is not None,
            attempt.supported_max_yield_t_ha is not None,
        )
        for attempt in credible
    }
    reasons = {"ALL_CREDIBLE_MODELS_REPORTED_WITHOUT_SELECTION"}
    tolerances = _reviewed_disagreement_tolerances(policy)
    if len(conclusion_signatures) > 1 or not stable_shape:
        materially_different: bool | None = True
    else:
        varying_fields = [
            name
            for name in (
                "agronomic_optimum_n_kg_ha",
                "plateau_onset_n_kg_ha",
                "supported_max_yield_t_ha",
            )
            if ranges[name] is not None and ranges[name][0] != ranges[name][1]
        ]
        if not varying_fields:
            materially_different = False
        elif tolerances is None:
            materially_different = None
            reasons.add("REVIEWED_NUMERIC_DISAGREEMENT_RULE_UNAVAILABLE")
        else:
            materially_different = any(
                ranges[name] is not None
                and ranges[name][1] - ranges[name][0] > tolerances[name]
                for name in varying_fields
            )
    if not stable_shape:
        reasons.add("CREDIBLE_MODEL_SHAPES_DIFFER")
    if materially_different is True:
        reasons.add("SINGLE_CONCLUSION_SUPPRESSED_FOR_MATERIAL_DISAGREEMENT")
    elif materially_different is None:
        reasons.add("SINGLE_CONCLUSION_SUPPRESSED_PENDING_REVIEWED_MATERIALITY_RULE")
    exact_numeric_consensus = all(
        ranges[name] is None or ranges[name][0] == ranges[name][1]
        for name in (
            "agronomic_optimum_n_kg_ha",
            "plateau_onset_n_kg_ha",
            "supported_max_yield_t_ha",
        )
    )
    if materially_different is False:
        status = "stable_credible_conclusions"
        optimum_status = (
            credible[0].optimum_status
            if exact_numeric_consensus
            else "CREDIBLE_MODEL_RANGE_REPORTED"
        )
        maximum_reference_basis = (
            credible[0].maximum_reference_basis
            if len({attempt.maximum_reference_basis for attempt in credible}) == 1
            else "multiple_credible_bases"
        )
        maximum_proximity_status = (
            credible[0].maximum_proximity_status
            if exact_numeric_consensus
            and len({attempt.maximum_proximity_status for attempt in credible}) == 1
            else "CREDIBLE_MODEL_RANGE_REPORTED"
        )
    elif materially_different is True:
        status = (
            "mixed_shape_conclusions"
            if not stable_shape
            else "materially_different_credible_conclusions"
        )
        optimum_status = "SUPPRESSED_MODEL_DISAGREEMENT"
        maximum_reference_basis = "none"
        maximum_proximity_status = "MODEL_DISAGREEMENT"
        values = {name: None for name in numeric_fields}
    else:
        status = "numeric_materiality_rule_unavailable"
        optimum_status = "SUPPRESSED_MODEL_DISAGREEMENT"
        maximum_reference_basis = "none"
        maximum_proximity_status = "MODEL_DISAGREEMENT"
        values = {name: None for name in numeric_fields}
    return _DisagreementSummary(
        status=status,
        curve_shape_class=next(iter(shapes)) if stable_shape else "uncertain_or_mixed",
        optimum_status=optimum_status,
        maximum_reference_basis=maximum_reference_basis,
        maximum_proximity_status=maximum_proximity_status,
        materially_different=materially_different,
        values=values,
        ranges=ranges,
        reason_codes=tuple(sorted(reasons)),
    )


def _economic_scenario_status(
    policy: Mapping[str, Any],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    raw_table = policy.get("economic_scenario_table")
    if not isinstance(raw_table, Mapping) or raw_table.get("review_status") != "approved":
        return (
            "disabled_no_approved_versioned_scenario_table",
            (),
            ("APPROVED_ECONOMIC_SCENARIO_TABLE_UNAVAILABLE",),
        )
    table_id = raw_table.get("table_id")
    version = raw_table.get("version")
    rows = raw_table.get("scenarios")
    required = {
        "scenario_id",
        "grain_price",
        "grain_price_unit",
        "n_cost",
        "n_cost_unit",
        "currency",
        "reference_period",
        "price_basis",
        "tax_subsidy_application_cost_basis",
        "decision_rule",
    }
    if (
        not isinstance(table_id, str)
        or not table_id.strip()
        or not isinstance(version, str)
        or not version.strip()
        or not isinstance(rows, (list, tuple))
        or not rows
        or any(not isinstance(row, Mapping) or set(row) != required for row in rows)
    ):
        return (
            "disabled_invalid_scenario_table",
            (),
            ("APPROVED_ECONOMIC_SCENARIO_TABLE_INVALID",),
        )
    scenario_ids = tuple(str(row["scenario_id"]) for row in rows)
    if (
        any(not scenario_id for scenario_id in scenario_ids)
        or len(scenario_ids) != len(set(scenario_ids))
        or any(
            isinstance(row["grain_price"], bool)
            or not isinstance(row["grain_price"], (float, int))
            or not math.isfinite(float(row["grain_price"]))
            or float(row["grain_price"]) <= 0.0
            or isinstance(row["n_cost"], bool)
            or not isinstance(row["n_cost"], (float, int))
            or not math.isfinite(float(row["n_cost"]))
            or float(row["n_cost"]) < 0.0
            or any(
                not isinstance(row[key], str) or not str(row[key]).strip()
                for key in required - {"scenario_id", "grain_price", "n_cost"}
            )
            for row in rows
        )
        or any(
            row["decision_rule"] != ECONOMIC_DECISION_RULE
            or row["grain_price_unit"] not in ECONOMIC_GRAIN_PRICE_TO_PER_TONNE
            or row["n_cost_unit"] != ECONOMIC_N_COST_UNIT
            for row in rows
        )
    ):
        return (
            "disabled_invalid_scenario_table",
            (),
            ("APPROVED_ECONOMIC_SCENARIO_TABLE_INVALID",),
        )
    return (
        "available_by_scenario_and_credible_model",
        scenario_ids,
        (),
    )


def _model_function_economic_optimum(
    attempt: ModelAttempt,
    *,
    grain_value_per_tonne: float,
    n_cost_per_kg: float,
) -> dict[str, float] | None:
    """Maximize reviewed net return over the continuous observed N domain."""

    lower = finite_number(attempt.observed_n_min_kg_ha)
    upper = finite_number(attempt.observed_n_max_kg_ha)
    if (
        lower is None
        or upper is None
        or lower > upper
        or not math.isfinite(grain_value_per_tonne)
        or grain_value_per_tonne <= 0.0
        or not math.isfinite(n_cost_per_kg)
        or n_cost_per_kg < 0.0
    ):
        return None
    parameters = {
        name: float(value)
        for name, value in attempt.parameters.items()
        if finite_number(value) is not None
    }
    if len(parameters) != len(attempt.parameters):
        return None
    candidates = {lower, upper}

    def add_candidate(value: float | None) -> None:
        if value is not None and math.isfinite(value):
            candidates.add(min(max(float(value), lower), upper))

    if attempt.model_name == "linear":
        pass
    elif attempt.model_name == "quadratic":
        curvature = parameters["curvature"]
        denominator = 2.0 * grain_value_per_tonne * curvature
        if denominator != 0.0:
            add_candidate(
                (n_cost_per_kg - grain_value_per_tonne * parameters["slope"])
                / denominator
            )
    elif attempt.model_name == "linear_plateau":
        add_candidate(parameters["plateau_onset"])
    elif attempt.model_name == "quadratic_plateau":
        onset = parameters["plateau_onset"]
        gain = parameters["gain"]
        if onset <= 0.0:
            return None
        add_candidate(onset)
        denominator = 2.0 * grain_value_per_tonne * gain
        if denominator != 0.0:
            add_candidate(onset - n_cost_per_kg * onset**2 / denominator)
    elif attempt.model_name == "mitscherlich":
        amplitude = parameters["amplitude"]
        rate = parameters["rate"]
        marginal_value_at_zero = grain_value_per_tonne * amplitude * rate
        if n_cost_per_kg > 0.0 and marginal_value_at_zero > 0.0 and rate != 0.0:
            ratio = n_cost_per_kg / marginal_value_at_zero
            if ratio > 0.0:
                add_candidate(-math.log(ratio) / rate)
    else:
        raise ValueError(
            f"Economic optimization is not implemented for model {attempt.model_name!r}"
        )

    rates = sorted(candidates)
    predicted = evaluate_model(attempt.model_name, rates, parameters)
    evaluated = tuple(
        {
            "n_rate_kg_ha": float(n_rate),
            "predicted_yield_t_ha": float(yield_value),
            "net_return_per_ha": (
                float(yield_value) * grain_value_per_tonne
                - float(n_rate) * n_cost_per_kg
            ),
        }
        for n_rate, yield_value in zip(rates, predicted, strict=True)
        if math.isfinite(float(yield_value))
    )
    if not evaluated:
        return None
    maximum_net_return = max(row["net_return_per_ha"] for row in evaluated)
    tolerance = max(abs(maximum_net_return), 1.0) * 1e-12
    return min(
        (
            row
            for row in evaluated
            if abs(row["net_return_per_ha"] - maximum_net_return) <= tolerance
        ),
        key=lambda row: row["n_rate_kg_ha"],
    )


def _economic_optimum_rows(
    credible_attempts: Sequence[ModelAttempt],
    *,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    status, _, _ = _economic_scenario_status(policy)
    if status != "available_by_scenario_and_credible_model":
        return ()
    raw_table = policy["economic_scenario_table"]
    scenarios = raw_table["scenarios"]
    rows: list[dict[str, Any]] = []
    for attempt in sorted(
        credible_attempts,
        key=lambda item: (item.response_series_uid, item.model_name, item.model_attempt_uid),
    ):
        for scenario in sorted(scenarios, key=lambda item: str(item["scenario_id"])):
            grain_price_factor = ECONOMIC_GRAIN_PRICE_TO_PER_TONNE[
                str(scenario["grain_price_unit"])
            ]
            grain_price = float(scenario["grain_price"])
            n_cost = float(scenario["n_cost"])
            optimum = _model_function_economic_optimum(
                attempt,
                grain_value_per_tonne=grain_price_factor * grain_price,
                n_cost_per_kg=n_cost,
            )
            if optimum is None:
                continue
            scenario_payload = dict(scenario)
            scenario_sha256 = stable_json_sha256(scenario_payload)
            row_identity = {
                "decision_rule": ECONOMIC_DECISION_RULE,
                "model_attempt_uid": attempt.model_attempt_uid,
                "scenario_sha256": scenario_sha256,
                "table_id": raw_table["table_id"],
                "table_version": raw_table["version"],
            }
            rows.append(
                {
                    "economic_optimum_uid": stable_identifier(
                        "economic_optimum",
                        row_identity,
                    ),
                    "response_series_uid": attempt.response_series_uid,
                    "model_attempt_uid": attempt.model_attempt_uid,
                    "model_name": attempt.model_name,
                    "model_input_snapshot_sha256": attempt.input_snapshot_sha256,
                    "model_policy_sha256": attempt.model_policy_sha256,
                    "economic_policy_id": raw_table["policy_id"],
                    "economic_table_id": raw_table["table_id"],
                    "economic_table_version": raw_table["version"],
                    "scenario_id": scenario["scenario_id"],
                    "scenario_sha256": scenario_sha256,
                    "decision_rule": ECONOMIC_DECISION_RULE,
                    "currency": scenario["currency"],
                    "reference_period": scenario["reference_period"],
                    "price_basis": scenario["price_basis"],
                    "tax_subsidy_application_cost_basis": scenario[
                        "tax_subsidy_application_cost_basis"
                    ],
                    "grain_price": grain_price,
                    "grain_price_unit": scenario["grain_price_unit"],
                    "n_cost": n_cost,
                    "n_cost_unit": scenario["n_cost_unit"],
                    "economic_optimum_n_kg_ha": optimum["n_rate_kg_ha"],
                    "predicted_yield_t_ha": optimum["predicted_yield_t_ha"],
                    "net_return_per_ha": optimum["net_return_per_ha"],
                    "observed_n_min_kg_ha": attempt.observed_n_min_kg_ha,
                    "observed_n_max_kg_ha": attempt.observed_n_max_kg_ha,
                    "tie_rule": "lowest_n_rate",
                    "optimization_method": "model_function_candidate_points_v1",
                    "status": "computed_observed_domain",
                    "reason_codes": (),
                }
            )
    return tuple(rows)


def _approved_observed_maximum_tie_policy(
    policy: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return the approved observed-maximum tie policy, or None."""

    raw = policy.get("observed_maximum_tie_policy")
    if not isinstance(raw, Mapping):
        return None
    tolerance = finite_number(raw.get("yield_equivalence_tolerance_t_ha"))
    authority = policy.get("scientific_policy_authority")
    artifact_sha256 = (
        authority.get("artifact_sha256") if isinstance(authority, Mapping) else None
    )
    if (
        raw.get("review_status") != "approved"
        or not isinstance(raw.get("policy_id"), str)
        or not str(raw["policy_id"]).strip()
        or tolerance is None
        or tolerance < 0.0
        or raw.get("tie_rule") not in {"retain_all_tied_rates", "lowest_tied_rate"}
        or not isinstance(authority, Mapping)
        or not isinstance(authority.get("approved_by"), str)
        or not str(authority["approved_by"]).strip()
        or not isinstance(authority.get("approved_on"), str)
        or not str(authority["approved_on"]).strip()
        or not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
        or any(character not in "0123456789abcdef" for character in artifact_sha256)
    ):
        return None
    return {**dict(raw), "yield_equivalence_tolerance_t_ha": tolerance}


def _observed_maximum_rate(
    complete: Sequence[tuple[float, float]],
    observed_max: float,
    policy: Mapping[str, Any],
) -> tuple[float | None, str, str, tuple[float, ...]]:
    """Report the N rate of the observed maximum without inventing a tie rule.

    Deciding whether two yields tie needs an approved equivalence tolerance:
    raw float equality would call agronomically indistinguishable values
    distinct, and would fail the unsafe way by presenting a real tie as a unique
    observed maximum. Collapsing a tie to one rate is likewise a policy choice
    that no approved record currently makes, so absent that policy this returns
    no single rate and instead carries every exactly-tied rate as evidence.
    """

    exact_rates = tuple(
        sorted(
            {
                float(n_rate)
                for n_rate, yield_value in complete
                if yield_value == observed_max
            }
        )
    )
    tie_policy = _approved_observed_maximum_tie_policy(policy)
    if tie_policy is None:
        return (
            None,
            "none",
            "unavailable_no_approved_yield_equivalence_tolerance",
            exact_rates,
        )
    tolerance = float(tie_policy["yield_equivalence_tolerance_t_ha"])
    equivalent_rates = tuple(
        sorted(
            {
                float(n_rate)
                for n_rate, yield_value in complete
                if abs(yield_value - observed_max) <= tolerance
            }
        )
    )
    if len(equivalent_rates) == 1:
        return (
            equivalent_rates[0],
            "single_observed_maximum",
            "available_single_observed_maximum",
            equivalent_rates,
        )
    if tie_policy["tie_rule"] == "lowest_tied_rate":
        return (
            equivalent_rates[0],
            "tied_observed_maximum_lowest_rate",
            "available_approved_tie_rule",
            equivalent_rates,
        )
    return (
        None,
        "tied_observed_maximum_retained",
        "unavailable_tied_rates_retained",
        equivalent_rates,
    )


def _maximum_associated_rate(
    attempt: ModelAttempt,
) -> tuple[float | None, str, str]:
    """Return the N rate a supported maximum sits at, with its labelled basis.

    Only a supported finite maximum earns a rate here. An asymptotic curve has
    no finite maximum, so under decided MOD-07 Option A its rate comes from the
    separate `q`-of-asymptote reporting path, and only after MOD-08 support
    passes. Until that policy is bound the rate stays unavailable rather than
    borrowing the observed maximum or the domain edge, either of which would
    silently answer a question the approved policy has not yet answered.
    """

    if attempt.optimum_status == "IDENTIFIABLE_INTERIOR_MAXIMUM":
        return (
            attempt.agronomic_optimum_n_kg_ha,
            "finite_interior_maximum",
            "available_supported_finite_maximum",
        )
    if attempt.optimum_status == "IDENTIFIABLE_PLATEAU_ONSET":
        return (
            attempt.plateau_onset_n_kg_ha,
            "plateau_onset",
            "available_supported_plateau_onset",
        )
    if attempt.maximum_reference_basis == "supported_attainable_asymptote":
        # MOD-08 support passed; the finite rate itself is MOD-07's to supply.
        return (
            None,
            "none",
            "pending_mod07_asymptote_fraction_policy",
        )
    return (None, "none", "unavailable_no_supported_finite_maximum")


def _attainable_yield(attempt: ModelAttempt) -> tuple[float | None, str, str]:
    """Return attainable yield only from an already-supported maximum.

    A fitted asymptote reaches this field solely through the MOD-08 promotion,
    which requires the approved in-domain attainment, uncertainty, influence,
    and credible-model concordance gates to have passed. Absent that promotion
    the ceiling stays in `fitted_asymptote_yield_t_ha` as model-implied only.
    """

    if attempt.maximum_reference_basis == "supported_attainable_asymptote":
        return (
            attempt.supported_max_yield_t_ha,
            "in_domain_fitted_attainment_threshold",
            "available_supported_asymptote",
        )
    if attempt.finite_maximum_yield_t_ha is not None:
        return (
            attempt.finite_maximum_yield_t_ha,
            "supported_finite_maximum",
            "available_supported_finite_maximum",
        )
    return (None, "none", "unavailable_no_supported_maximum")


def _curve_row(
    fit_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    representative: ModelAttempt,
    *,
    zero_tolerance: float,
    baseline_metrics_enabled: bool,
    policy: Mapping[str, Any],
) -> dict[str, Any] | None:
    supported_tiers = {
        str(row.get("series_eligibility_tier") or row.get("eligibility_tier") or "")
        for row in evidence_rows
    }
    if not supported_tiers.intersection({"A", "B"}):
        return None
    complete = [
        (n_rate, yield_value)
        for row in fit_rows
        if (n_rate := finite_number(row.get("n_rate_kg_ha"))) is not None
        and (yield_value := finite_number(row.get("yield_t_ha"))) is not None
    ]
    if not complete:
        return None
    n_rates = [pair[0] for pair in complete]
    yields = [pair[1] for pair in complete]
    series_uid = representative.response_series_uid
    source_name = _one_value(evidence_rows, "source_name")
    if source_name is None:
        return None
    observed_max = float(max(yields))
    (
        observed_max_n,
        observed_max_n_basis,
        observed_max_n_status,
        observed_max_rates,
    ) = _observed_maximum_rate(complete, observed_max, policy)
    (
        maximum_associated_n,
        maximum_associated_n_basis,
        maximum_associated_n_status,
    ) = _maximum_associated_rate(representative)
    (
        attainable_yield,
        attainable_yield_basis,
        attainable_yield_status,
    ) = _attainable_yield(representative)
    supported_max = representative.supported_max_yield_t_ha
    attainment = (
        observed_max / supported_max
        if supported_max is not None and supported_max > 0.0
        else None
    )
    finite_gap = (
        representative.finite_maximum_yield_t_ha - observed_max
        if representative.finite_maximum_yield_t_ha is not None
        else None
    )
    supported_gap = (
        supported_max - observed_max
        if supported_max is not None
        else None
    )
    distinct_level_count = len(set(n_rates))
    evidence_strength = (
        "four_level_restricted_fit"
        if distinct_level_count == 4
        else "five_plus_level_broad_roster"
    )
    baseline_fields, baseline_reasons = _baseline_metrics(
        evidence_rows,
        response_maximum=observed_max,
        zero_tolerance=zero_tolerance,
        enabled=baseline_metrics_enabled,
    )
    economic_status, economic_scenario_ids, economic_reasons = _economic_scenario_status(policy)
    return {
        "response_series_uid": series_uid,
        "selected_model_attempt_uid": None,
        "selected_model_name": None,
        "selected_model_status": "not_selected",
        "record_uids": tuple(str(row["record_uid"]) for row in fit_rows),
        "all_evidence_record_uids": tuple(str(row["record_uid"]) for row in evidence_rows),
        "source_name": source_name,
        "study_uid": _study_uid(evidence_rows, source_name),
        "trial_id": _one_value(evidence_rows, "trial_id"),
        "experiment_type": _one_value(evidence_rows, "experiment_type"),
        "experimental_design": _one_value(evidence_rows, "experimental_design"),
        "water_regime_normalized": _one_value(evidence_rows, "water_regime_normalized"),
        "season_normalized": _one_value(evidence_rows, "season_normalized"),
        "region": _one_value(evidence_rows, "region"),
        "province": _one_value(evidence_rows, "province"),
        "rice_variety": _one_value(evidence_rows, "rice_variety"),
        "planting_year": _one_value(evidence_rows, "planting_year"),
        "comparison_set_uid": _one_value(evidence_rows, "comparison_set_uid"),
        "treatment_uid": _one_value(evidence_rows, "treatment_uid"),
        "treatment_text_class": _one_value(evidence_rows, "treatment_text_class"),
        "treatment_class": _one_value(evidence_rows, "treatment_class"),
        "n_split": _one_value(evidence_rows, "n_split"),
        "organic_fertilizer_present": any(
            bool(row.get("organic_fertilizer_present")) for row in evidence_rows
        ),
        "biofertilizer_present": any(
            bool(row.get("biofertilizer_present")) for row in evidence_rows
        ),
        "series_distinct_n_level_count": distinct_level_count,
        "series_has_zero_n": any(abs(n_rate) <= zero_tolerance for n_rate in n_rates),
        "series_has_high_n": any(bool(row.get("is_high_n")) for row in evidence_rows),
        "series_p_constant": evidence_rows[0].get("series_p_constant"),
        "series_k_constant": evidence_rows[0].get("series_k_constant"),
        "p_rate_kg_p2o5_ha": finite_number(evidence_rows[0].get("p_rate_kg_p2o5_ha")),
        "k_rate_kg_k2o_ha": finite_number(evidence_rows[0].get("k_rate_kg_k2o_ha")),
        "series_observed_n_min_kg_ha": float(min(n_rates)),
        "series_observed_n_max_kg_ha": float(max(n_rates)),
        "curve_shape_class": representative.curve_shape_class,
        "optimum_status": representative.optimum_status,
        "agronomic_optimum_n_kg_ha": representative.agronomic_optimum_n_kg_ha,
        "plateau_onset_n_kg_ha": representative.plateau_onset_n_kg_ha,
        "predicted_max_yield_t_ha": representative.predicted_max_yield_t_ha,
        "predicted_observed_domain_peak_yield_t_ha": representative.predicted_observed_domain_peak_yield_t_ha,
        "finite_maximum_yield_t_ha": representative.finite_maximum_yield_t_ha,
        "fitted_asymptote_yield_t_ha": representative.fitted_asymptote_yield_t_ha,
        "supported_max_yield_t_ha": representative.supported_max_yield_t_ha,
        "maximum_reference_basis": representative.maximum_reference_basis,
        "maximum_proximity_status": representative.maximum_proximity_status,
        "observed_max_yield_t_ha": observed_max,
        "observed_max_n_kg_ha": observed_max_n,
        "observed_max_n_basis": observed_max_n_basis,
        "observed_max_n_status": observed_max_n_status,
        "observed_max_n_rates_kg_ha": observed_max_rates,
        "maximum_associated_n_kg_ha": maximum_associated_n,
        "maximum_associated_n_basis": maximum_associated_n_basis,
        "maximum_associated_n_status": maximum_associated_n_status,
        "attainable_yield_t_ha": attainable_yield,
        "attainable_yield_basis": attainable_yield_basis,
        "attainable_yield_status": attainable_yield_status,
        "observed_domain_boundary_status": (
            representative.observed_domain_boundary_status
        ),
        "observed_domain_boundary_n_kg_ha": (
            representative.observed_domain_boundary_n_kg_ha
        ),
        "observed_domain_boundary_yield_t_ha": (
            representative.observed_domain_boundary_yield_t_ha
        ),
        "observed_max_gap_to_finite_maximum_t_ha": finite_gap,
        "observed_max_gap_to_supported_maximum_t_ha": supported_gap,
        "observed_max_attainment_fraction": attainment,
        "evidence_strength": evidence_strength,
        **baseline_fields,
        "economic_optimum_status": economic_status,
        "economic_optimum_n_kg_ha": None,
        "economic_scenario_ids": economic_scenario_ids,
        "uncertainty_status": representative.uncertainty_status,
        "uncertainty_method": representative.uncertainty_method,
        "recommendation_yield_gap_t_ha": None,
        "target_yield_gap_t_ha": None,
        "target_yield_status": "not_configured",
        "model_attempt_record": None,
        "reason_codes": tuple(sorted(set(baseline_reasons) | set(economic_reasons))),
    }


def _first_stage_uncertainty_fields(
    credible: Sequence[ModelAttempt],
    row: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine within-model and all-credible-model uncertainty for ANA-16."""

    raw_policy = policy.get("first_stage_contextual_uncertainty_policy")
    authority = policy.get("scientific_policy_authority")
    if (
        not isinstance(raw_policy, Mapping)
        or raw_policy.get("review_status") != "approved"
        or raw_policy.get("method_id")
        != "all_credible_equal_weight_total_variance"
        or raw_policy.get("model_selection_uncertainty_method")
        != "all_credible_equal_weight_total_variance"
        or not isinstance(authority, Mapping)
    ):
        return {}
    policy_id = raw_policy.get("policy_id")
    within_method = raw_policy.get("within_model_variance_method")
    reviewer = authority.get("approved_by")
    reviewed_on = authority.get("approved_on")
    eligible_outcomes = raw_policy.get("eligible_outcomes")
    if (
        not isinstance(policy_id, str)
        or not policy_id.strip()
        or not isinstance(within_method, str)
        or not within_method.strip()
        or not isinstance(reviewer, str)
        or not reviewer.strip()
        or not isinstance(reviewed_on, str)
        or not reviewed_on.strip()
        or not isinstance(eligible_outcomes, (list, tuple))
    ):
        return {}
    output: dict[str, Any] = {}
    method_id = (
        f"{policy_id}:{within_method}:"
        "all_credible_equal_weight_total_variance"
    )
    for raw_outcome in eligible_outcomes:
        if not isinstance(raw_outcome, str) or not raw_outcome:
            continue
        outcome = raw_outcome
        prefix = f"{outcome}_"
        output.update(
            {
                f"{prefix}first_stage_variance": None,
                f"{prefix}first_stage_variance_status": "unavailable",
                f"{prefix}model_selection_uncertainty_status": "not_incorporated",
                f"{prefix}first_stage_uncertainty_method_id": None,
                f"{prefix}first_stage_uncertainty_reviewer": None,
                f"{prefix}first_stage_uncertainty_reviewed_on": None,
            }
        )
        reported_value = finite_number(row.get(outcome))
        estimates: list[float] = []
        within_variances: list[float] = []
        valid = reported_value is not None and bool(credible)
        for attempt in credible:
            estimate = finite_number(getattr(attempt, outcome, None))
            variance = finite_number(attempt.feature_variances.get(outcome))
            if (
                estimate is None
                or variance is None
                or variance <= 0.0
                or attempt.uncertainty_method != within_method
                or not attempt.uncertainty_status.startswith("available_")
            ):
                valid = False
                break
            estimates.append(estimate)
            within_variances.append(variance)
        if not valid:
            continue
        mixture_mean = statistics.fmean(estimates)
        total_variance = statistics.fmean(
            within + (estimate - mixture_mean) ** 2
            for estimate, within in zip(
                estimates,
                within_variances,
                strict=True,
            )
        )
        if not math.isfinite(total_variance) or total_variance <= 0.0:
            continue
        output.update(
            {
                f"{prefix}first_stage_variance": total_variance,
                f"{prefix}first_stage_variance_status": "verified_comparable",
                f"{prefix}model_selection_uncertainty_status": "incorporated",
                f"{prefix}first_stage_uncertainty_method_id": method_id,
                f"{prefix}first_stage_uncertainty_reviewer": reviewer,
                f"{prefix}first_stage_uncertainty_reviewed_on": reviewed_on,
            }
        )
    return output


def _all_credible_curve_row(
    fit_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    credible: Sequence[ModelAttempt],
    *,
    zero_tolerance: float,
    baseline_metrics_enabled: bool,
    policy: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build one series row without ranking credible candidates against each other."""

    if not credible:
        return None
    row = _curve_row(
        fit_rows,
        evidence_rows,
        credible[0],
        zero_tolerance=zero_tolerance,
        baseline_metrics_enabled=baseline_metrics_enabled,
        policy=policy,
    )
    if row is None:
        return None
    attempt_uids = tuple(attempt.model_attempt_uid for attempt in credible)
    model_names = tuple(attempt.model_name for attempt in credible)
    row.update(
        {
            "model_reporting_policy": _ALL_CREDIBLE_POLICY,
            "credible_model_count": len(credible),
            "credible_model_attempt_uids": attempt_uids,
            "credible_model_names": model_names,
            "selected_model_attempt_uid": None,
            "selected_model_name": None,
            "selected_model_status": "not_selected",
            "model_attempt_record": None,
        }
    )
    summary = _disagreement_summary(credible, policy=policy)
    row.update(
        {
            "model_disagreement_status": summary.status,
            "materially_different_credible_conclusions": summary.materially_different,
            "curve_shape_class": summary.curve_shape_class,
            "optimum_status": summary.optimum_status,
            "maximum_reference_basis": summary.maximum_reference_basis,
            "maximum_proximity_status": summary.maximum_proximity_status,
            **summary.values,
            "agronomic_optimum_n_range_kg_ha": summary.ranges[
                "agronomic_optimum_n_kg_ha"
            ],
            "plateau_onset_n_range_kg_ha": summary.ranges["plateau_onset_n_kg_ha"],
            "supported_max_yield_range_t_ha": summary.ranges[
                "supported_max_yield_t_ha"
            ],
            "sole_credible_model_attempt_uid": (
                credible[0].model_attempt_uid if len(credible) == 1 else None
            ),
            "sole_credible_model_name": credible[0].model_name if len(credible) == 1 else None,
        }
    )
    # MOD-02 Option D suppression is quantity-specific: a credible set that
    # disagrees materially must not publish one associated rate, one attainable
    # yield, or one boundary reference, even though the observed maximum and its
    # rate remain facts about the data and stay reportable.
    if summary.materially_different is not False:
        suppression = (
            "suppressed_material_credible_model_disagreement"
            if summary.materially_different
            else "suppressed_pending_reviewed_materiality_rule"
        )
        row.update(
            {
                "maximum_associated_n_kg_ha": None,
                "maximum_associated_n_basis": "none",
                "maximum_associated_n_status": suppression,
                "attainable_yield_t_ha": None,
                "attainable_yield_basis": "none",
                "attainable_yield_status": suppression,
                "observed_domain_boundary_status": suppression,
                "observed_domain_boundary_n_kg_ha": None,
                "observed_domain_boundary_yield_t_ha": None,
            }
        )
    else:
        boundary_statuses = {
            attempt.observed_domain_boundary_status for attempt in credible
        }
        boundary_rates = _numeric_range(credible, "observed_domain_boundary_n_kg_ha")
        if len(boundary_statuses) != 1 or (
            boundary_rates is not None and boundary_rates[0] != boundary_rates[1]
        ):
            row.update(
                {
                    "observed_domain_boundary_status": (
                        "CREDIBLE_MODEL_RANGE_REPORTED"
                    ),
                    "observed_domain_boundary_n_kg_ha": None,
                    "observed_domain_boundary_yield_t_ha": None,
                }
            )
    observed_max = float(row["observed_max_yield_t_ha"])
    supported_max = row["supported_max_yield_t_ha"]
    finite_maximum = row["finite_maximum_yield_t_ha"]
    row["observed_max_gap_to_finite_maximum_t_ha"] = (
        finite_maximum - observed_max if finite_maximum is not None else None
    )
    row["observed_max_gap_to_supported_maximum_t_ha"] = (
        supported_max - observed_max if supported_max is not None else None
    )
    row["observed_max_attainment_fraction"] = (
        observed_max / supported_max
        if supported_max is not None and supported_max > 0.0
        else None
    )
    row.update(_first_stage_uncertainty_fields(credible, row, policy=policy))
    row["reason_codes"] = tuple(
        sorted(set(row["reason_codes"]) | set(summary.reason_codes))
    )
    if len(credible) == 1:
        return row
    row["evidence_strength"] = (
        "four_level_restricted_credible_set"
        if row["series_distinct_n_level_count"] == 4
        else "five_plus_level_broad_credible_set"
    )
    return row


def _series_evidence_row(
    rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[ModelAttempt],
    *,
    credible: Sequence[ModelAttempt],
    policy: Mapping[str, Any],
    fit_exclusion_reasons: Sequence[str],
    zero_tolerance: float,
) -> dict[str, Any] | None:
    supported_tiers = {
        str(row.get("series_eligibility_tier") or row.get("eligibility_tier") or "")
        for row in rows
    }
    if not supported_tiers.intersection({"A", "B", "C"}):
        return None
    descriptive_rows = _descriptive_candidate_rows(rows)
    complete = [
        (n_rate, yield_value)
        for row in descriptive_rows
        if (n_rate := finite_number(row.get("n_rate_kg_ha"))) is not None
        and (yield_value := finite_number(row.get("yield_t_ha"))) is not None
    ]
    levels = sorted({n_rate for n_rate, _ in complete})
    low_yields = [value for n_rate, value in complete if levels and abs(n_rate - levels[0]) <= zero_tolerance]
    high_yields = [value for n_rate, value in complete if levels and abs(n_rate - levels[-1]) <= zero_tolerance]
    n_change = levels[-1] - levels[0] if len(levels) >= 2 else None
    yield_change = (
        statistics.fmean(high_yields) - statistics.fmean(low_yields)
        if low_yields and high_yields and len(levels) >= 2
        else None
    )
    summary = _disagreement_summary(credible, policy=policy)
    evidence_reasons: list[str] = list(fit_exclusion_reasons)
    if len(levels) == 2:
        evidence_strength = "two_level_contrast_only"
        evidence_status = "contrast_only"
        evidence_reasons.append("TWO_LEVEL_CONTRAST_ONLY")
    elif len(levels) == 3:
        evidence_strength = "three_level_descriptive_only"
        evidence_status = "descriptive_only"
        evidence_reasons.append("DESCRIPTIVE_LEVEL_SUPPORT_ONLY")
    elif len(credible) > 1:
        evidence_strength = (
            "four_level_restricted_credible_set"
            if len(levels) == 4
            else "five_plus_level_broad_credible_set"
        )
        evidence_status = "credible_model_set_reported"
        evidence_reasons.extend(summary.reason_codes)
    elif len(credible) == 1:
        evidence_strength = (
            "four_level_restricted_fit"
            if len(levels) == 4
            else "five_plus_level_broad_roster"
        )
        evidence_status = "credible_model_reported"
        evidence_reasons.extend(summary.reason_codes)
    elif len(levels) < 2:
        evidence_strength = "insufficient_n_level_support"
        evidence_status = "unsupported"
        evidence_reasons.append("INSUFFICIENT_N_LEVEL_SUPPORT")
    elif len(levels) == 4:
        evidence_strength = "four_level_restricted_fit_unavailable"
        evidence_status = "unsupported"
        evidence_reasons.append("NO_CREDIBLE_RESTRICTED_MODEL")
    else:
        evidence_strength = "five_plus_level_broad_roster_unavailable"
        evidence_status = "unsupported"
        evidence_reasons.append("NO_CREDIBLE_BROAD_ROSTER_MODEL")
    source_name = _one_value(rows, "source_name")
    sole_credible = credible[0] if len(credible) == 1 else None
    return {
        "response_series_uid": str(rows[0].get("response_series_uid") or ""),
        "source_name": source_name,
        "study_uid": _study_uid(rows, source_name) if source_name is not None else None,
        "record_uids": tuple(str(row["record_uid"]) for row in rows),
        "distinct_n_level_count": len(levels),
        "observed_n_min_kg_ha": levels[0] if levels else None,
        "observed_n_max_kg_ha": levels[-1] if levels else None,
        "observed_max_yield_t_ha": max((value for _, value in complete), default=None),
        "observed_low_to_high_yield_change_t_ha": yield_change,
        "observed_low_to_high_n_change_kg_ha": n_change,
        "observed_two_level_slope_t_ha_per_kg_n_ha": (
            yield_change / n_change
            if yield_change is not None and n_change not in {None, 0.0}
            else None
        ),
        "evidence_status": evidence_status,
        "evidence_strength": evidence_strength,
        "curve_shape_class": summary.curve_shape_class,
        "optimum_status": summary.optimum_status,
        "maximum_reference_basis": summary.maximum_reference_basis,
        "maximum_proximity_status": summary.maximum_proximity_status,
        "target_yield_status": "not_configured",
        "model_reporting_policy": _ALL_CREDIBLE_POLICY,
        "model_disagreement_status": summary.status,
        "materially_different_credible_conclusions": summary.materially_different,
        "selected_model_attempt_uid": None,
        "selected_model_name": None,
        "sole_credible_model_attempt_uid": (
            sole_credible.model_attempt_uid if sole_credible is not None else None
        ),
        "sole_credible_model_name": sole_credible.model_name if sole_credible is not None else None,
        "credible_model_attempt_uids": tuple(attempt.model_attempt_uid for attempt in credible),
        "credible_model_names": tuple(attempt.model_name for attempt in credible),
        "credible_model_count": len(credible),
        "model_attempt_uids": tuple(attempt.model_attempt_uid for attempt in attempts),
        "model_reason_codes": tuple(sorted({reason for attempt in attempts for reason in attempt.reason_codes})),
        "reason_codes": tuple(sorted(set(evidence_reasons) | set(summary.reason_codes))),
    }


def curve_fit_record_uids(
    records: Iterable[Mapping[str, Any]],
    *,
    primary_only: bool,
) -> tuple[str, ...]:
    """Return rows permitted in a curve fit without discarding evidence rows."""

    copied_records = tuple(dict(record) for record in records)
    exclusions = _fit_exclusions(copied_records, primary_only=primary_only)
    return tuple(
        str(record["record_uid"])
        for record in copied_records
        if str(record.get("record_uid") or "") not in exclusions
    )


def build_curve_evidence(
    records: Iterable[Mapping[str, Any]],
    *,
    model_names: Sequence[str],
    policy: Mapping[str, Any],
    fit_record_uids: Iterable[str] | None = None,
) -> CurveEvidenceResult:
    """Fit an explicit membership while retaining observational evidence for all rows."""

    copied_records = tuple(dict(record) for record in records)
    if (
        tuple(model_names) != tuple(dict.fromkeys(model_names))
        or set(model_names) != set(MODEL_ORDER)
    ):
        raise ValueError(
            "Curve evidence requires the complete canonical model roster"
        )
    reporting_policy = policy.get("model_selection_metric")
    if reporting_policy != _ALL_CREDIBLE_POLICY or policy.get("tie_breaking") != "not_applicable":
        raise ValueError(
            "Curve evidence requires all credible candidates without model selection"
        )
    permitted_uids = set(curve_fit_record_uids(copied_records, primary_only=False))
    if fit_record_uids is None:
        requested = permitted_uids
    else:
        requested_uids = tuple(str(record_uid) for record_uid in fit_record_uids)
        if len(requested_uids) != len(set(requested_uids)):
            raise ValueError("fit_record_uids must be unique")
        available_uids = {str(record.get("record_uid", "")) for record in copied_records}
        unknown_uids = sorted(set(requested_uids) - available_uids)
        if unknown_uids:
            raise ValueError("fit_record_uids contains unknown records: " + ", ".join(unknown_uids))
        requested = set(requested_uids).intersection(permitted_uids)
    fit_records = tuple(
        record
        for record in copied_records
        if str(record.get("record_uid", "")) in requested
    )
    attempts = fit_response_models(fit_records, model_names=model_names, policy=policy)
    by_series: dict[str, list[ModelAttempt]] = {}
    for attempt in attempts:
        by_series.setdefault(attempt.response_series_uid, []).append(attempt)
    credible_by_series = {
        series_uid: credible_model_attempts(series_attempts)
        for series_uid, series_attempts in sorted(by_series.items())
    }
    credible_attempts_flat = tuple(
        attempt
        for series_uid in sorted(credible_by_series)
        for attempt in credible_by_series[series_uid]
    )
    asymptote_support_rows = _asymptote_support_rows(
        credible_attempts_flat,
        policy=policy,
    )
    credible_attempts_flat = _promote_supported_asymptotes(
        credible_attempts_flat,
        asymptote_support_rows,
    )
    promoted_by_uid = {
        attempt.model_attempt_uid: attempt for attempt in credible_attempts_flat
    }
    attempts = tuple(
        promoted_by_uid.get(attempt.model_attempt_uid, attempt)
        for attempt in attempts
    )
    by_series = {}
    for attempt in attempts:
        by_series.setdefault(attempt.response_series_uid, []).append(attempt)
    credible_by_series = {
        series_uid: tuple(
            promoted_by_uid.get(attempt.model_attempt_uid, attempt)
            for attempt in series_attempts
            if attempt.model_attempt_uid in promoted_by_uid
        )
        for series_uid, series_attempts in sorted(by_series.items())
    }
    selected_attempts: tuple[ModelAttempt, ...] = ()
    grouped_rows = _series_rows(copied_records)
    grouped_fit_rows = _series_rows(fit_records)
    all_exclusions = _fit_exclusions(copied_records, primary_only=False)
    zero_tolerance = float(policy.get("convergence_tolerance", 1e-8))
    baseline_metrics_enabled = (
        policy.get("baseline_response_policy") == _SEPARATE_BASELINE_POLICY
    )
    series_evidence_rows = tuple(
        evidence_row
        for series_uid in sorted(grouped_rows)
        if (
            evidence_row := _series_evidence_row(
                grouped_rows[series_uid],
                by_series.get(series_uid, ()),
                credible=credible_by_series.get(series_uid, ()),
                policy=policy,
                fit_exclusion_reasons=tuple(
                    sorted(
                        {
                            reason
                            for row in grouped_rows[series_uid]
                            for reason in all_exclusions.get(
                                str(row.get("record_uid") or ""),
                                (),
                            )
                        }
                    )
                ),
                zero_tolerance=zero_tolerance,
            )
        ) is not None
    )
    curve_rows = tuple(
        curve
        for series_uid in sorted(credible_by_series)
        if (
            curve := _all_credible_curve_row(
                grouped_fit_rows.get(series_uid, ()),
                grouped_rows.get(series_uid, ()),
                credible_by_series[series_uid],
                zero_tolerance=zero_tolerance,
                baseline_metrics_enabled=baseline_metrics_enabled,
                policy=policy,
            )
        )
        is not None
    )
    prediction_attempts = credible_attempts_flat
    curve_series = {str(row["response_series_uid"]) for row in curve_rows}
    predictions = tuple(
        prediction
        for attempt in prediction_attempts
        if attempt.response_series_uid in curve_series
        for prediction in prediction_rows(attempt)
    )
    credible_ids = {
        attempt.model_attempt_uid for attempt in credible_attempts_flat
    }
    selected_ids = {
        attempt.model_attempt_uid for attempt in selected_attempts
    }
    model_attempt_records = tuple(
        {
            **model_attempt_record(attempt),
            "credible_for_reporting": attempt.model_attempt_uid in credible_ids,
            "selected_for_reporting": attempt.model_attempt_uid in selected_ids,
            "model_reporting_policy": reporting_policy,
        }
        for attempt in attempts
    )
    economic_optimum_rows = _economic_optimum_rows(
        credible_attempts_flat,
        policy=policy,
    )
    efficiency_rows = _partial_factor_productivity_rows(
        copied_records,
        policy=policy,
    )
    efficiency_operating_point_rows = _efficiency_operating_point_rows(
        credible_attempts_flat,
        policy=policy,
    )
    asymptote_reporting_rows = _asymptote_reporting_rows(
        credible_attempts_flat,
        policy=policy,
    )
    asymptote_by_series = {
        str(row["response_series_uid"]): row
        for row in asymptote_reporting_rows
    }
    # An asymptotic curve has no finite maximum, so its associated rate can only
    # come from the approved MOD-07 fraction, and only once MOD-08 support has
    # already promoted the ceiling. Rows left pending by `_maximum_associated_rate`
    # are the only ones eligible; nothing else is overwritten.
    curve_rows = tuple(
        {
            **row,
            **(
                {
                    "maximum_associated_n_kg_ha": asymptote_row[
                        "asymptote_fraction_n_kg_ha"
                    ],
                    "maximum_associated_n_basis": asymptote_row["basis"],
                    "maximum_associated_n_status": (
                        "available_mod07_asymptote_fraction"
                    ),
                }
                if (
                    row.get("maximum_associated_n_status")
                    == "pending_mod07_asymptote_fraction_policy"
                    and (
                        asymptote_row := asymptote_by_series.get(
                            str(row["response_series_uid"])
                        )
                    )
                    is not None
                    and asymptote_row.get("status") == "available"
                )
                else {}
            ),
        }
        for row in curve_rows
    )
    series_evidence_rows = tuple(
        {
            **row,
            **(
                {
                    "asymptote_reporting_status": asymptote_row["status"],
                    "asymptote_rate_label": asymptote_row["rate_label"],
                    "asymptote_fraction": asymptote_row["asymptote_fraction"],
                    "asymptote_fraction_n_kg_ha": asymptote_row[
                        "asymptote_fraction_n_kg_ha"
                    ],
                    "asymptote_fraction_n_se_kg_ha": asymptote_row[
                        "asymptote_fraction_n_se_kg_ha"
                    ],
                    "asymptote_reporting_basis": asymptote_row["basis"],
                    "asymptote_reporting_policy_id": asymptote_row[
                        "asymptote_reporting_policy_id"
                    ],
                }
                if (
                    asymptote_row := asymptote_by_series.get(
                        str(row["response_series_uid"])
                    )
                )
                is not None
                else {}
            ),
        }
        for row in series_evidence_rows
    )
    environmental_risk_rows = _environmental_risk_rows(copied_records)
    return CurveEvidenceResult(
        reporting_policy=reporting_policy,
        model_attempts=tuple(attempts),
        credible_attempts=credible_attempts_flat,
        selected_attempts=selected_attempts,
        model_attempt_records=model_attempt_records,
        series_evidence_rows=series_evidence_rows,
        curve_rows=curve_rows,
        prediction_rows=predictions,
        economic_optimum_rows=economic_optimum_rows,
        efficiency_rows=efficiency_rows,
        efficiency_operating_point_rows=efficiency_operating_point_rows,
        asymptote_support_rows=asymptote_support_rows,
        asymptote_reporting_rows=asymptote_reporting_rows,
        environmental_risk_rows=environmental_risk_rows,
    )


__all__ = ["CurveEvidenceResult", "build_curve_evidence", "curve_fit_record_uids"]
