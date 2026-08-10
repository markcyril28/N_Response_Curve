from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

from ..data.provenance import stable_identifier, stable_json_sha256
from ..reporting.plots import prediction_rows
from .models import (
    MODEL_ORDER,
    ModelAttempt,
    credible_model_attempts,
    fit_response_models,
    model_attempt_record,
)
from .reviewed_methods import (
    ECONOMIC_DECISION_RULE,
    ECONOMIC_GRAIN_PRICE_TO_PER_TONNE,
    ECONOMIC_N_COST_UNIT,
)
from .values import finite_number


_ALL_CREDIBLE_POLICY = "all_credible_no_selection"
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


def _partial_factor_productivity_rows(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Materialize EFF-01 Option A at the observed record-by-N-level grain."""

    rows: list[dict[str, Any]] = []
    for record in records:
        record_uid = str(record.get("record_uid") or "")
        series_uid = str(record.get("response_series_uid") or "")
        n_rate = finite_number(record.get("n_rate_kg_ha"))
        yield_t_ha = finite_number(record.get("yield_t_ha"))
        if (
            not record_uid
            or not series_uid
            or record.get("series_status") != "resolved"
            or n_rate is None
            or n_rate <= 0.0
            or yield_t_ha is None
            or record.get("yield_unit_status") == "conflict"
            or record.get("final_analytical_membership_status")
            == "excluded_by_reviewed_cleaning_rule"
        ):
            continue
        grain_status = str(
            record.get("analysis_grain_status") or "unreviewed_observed_record"
        )
        value = yield_t_ha * 1000.0 / n_rate
        identity = {
            "metric_id": "EFF-01-option-a-partial-factor-productivity",
            "record_uid": record_uid,
            "response_series_uid": series_uid,
            "n_rate_kg_ha": n_rate,
            "yield_t_ha": yield_t_ha,
            "analysis_grain_status": grain_status,
        }
        rows.append(
            {
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
                "basis": (
                    "observed_reviewed_treatment_mean"
                    if grain_status == "reviewed_treatment_mean"
                    else "observed_record"
                ),
                "aggregation_level": "record_by_n_level",
                "unit": "kg_grain_per_kg_n",
                "partial_factor_productivity_kg_grain_per_kg_n": value,
                "status": (
                    "computed"
                    if grain_status == "reviewed_treatment_mean"
                    else "computed_exploratory_unreviewed_grain"
                ),
            }
        )
    return tuple(sorted(rows, key=lambda row: str(row["record_uid"])))


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
            evaluated = tuple(
                {
                    "n_rate_kg_ha": float(prediction["n_rate_kg_ha"]),
                    "predicted_yield_t_ha": float(
                        prediction["predicted_yield_t_ha"]
                    ),
                    "net_return_per_ha": (
                        float(prediction["predicted_yield_t_ha"])
                        * grain_price_factor
                        * grain_price
                        - float(prediction["n_rate_kg_ha"]) * n_cost
                    ),
                }
                for prediction in attempt.predictions
            )
            if not evaluated:
                continue
            maximum_net_return = max(row["net_return_per_ha"] for row in evaluated)
            tolerance = max(abs(maximum_net_return), 1.0) * 1e-12
            optimum = min(
                (
                    row
                    for row in evaluated
                    if abs(row["net_return_per_ha"] - maximum_net_return)
                    <= tolerance
                ),
                key=lambda row: row["n_rate_kg_ha"],
            )
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
                    "status": "computed_observed_domain",
                    "reason_codes": (),
                }
            )
    return tuple(rows)


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
    efficiency_rows = _partial_factor_productivity_rows(copied_records)
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
        environmental_risk_rows=environmental_risk_rows,
    )


__all__ = ["CurveEvidenceResult", "build_curve_evidence", "curve_fit_record_uids"]
