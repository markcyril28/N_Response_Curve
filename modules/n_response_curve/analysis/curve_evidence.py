from __future__ import annotations

from dataclasses import dataclass
import hashlib
import statistics
from typing import Any, Iterable, Mapping, Sequence

from ..reporting.plots import prediction_rows
from .models import (
    ModelAttempt,
    credible_model_attempts,
    fit_response_models,
    model_attempt_record,
    select_reportable_model,
)
from .values import finite_number


_ALL_CREDIBLE_POLICY = "all_credible_no_selection"


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


def _curve_row(
    rows: Sequence[Mapping[str, Any]],
    selected: ModelAttempt,
    *,
    zero_tolerance: float,
    baseline_metrics_enabled: bool,
) -> dict[str, Any] | None:
    supported_tiers = {str(row.get("series_eligibility_tier") or row.get("eligibility_tier") or "") for row in rows}
    if not supported_tiers.intersection({"A", "B"}):
        return None
    complete = [
        (n_rate, yield_value)
        for row in rows
        if (n_rate := finite_number(row.get("n_rate_kg_ha"))) is not None
        and (yield_value := finite_number(row.get("yield_t_ha"))) is not None
    ]
    if not complete:
        return None
    n_rates = [pair[0] for pair in complete]
    yields = [pair[1] for pair in complete]
    zero_yields = [yield_value for n_rate, yield_value in complete if abs(n_rate) <= zero_tolerance]
    series_uid = selected.response_series_uid
    source_name = _one_value(rows, "source_name")
    if source_name is None:
        return None
    selected_record = model_attempt_record(selected)
    observed_max = float(max(yields))
    supported_max = selected.supported_max_yield_t_ha
    attainment = (
        observed_max / supported_max
        if supported_max is not None and supported_max > 0.0
        else None
    )
    finite_gap = (
        selected.finite_maximum_yield_t_ha - observed_max
        if selected.finite_maximum_yield_t_ha is not None
        else None
    )
    supported_gap = (
        supported_max - observed_max
        if supported_max is not None
        else None
    )
    evidence_strength = (
        "three_level_linear_only"
        if len(set(n_rates)) == 3 and selected.model_name == "linear"
        else "four_plus_level_curve"
    )
    return {
        "response_series_uid": series_uid,
        "selected_model_attempt_uid": selected.model_attempt_uid,
        "selected_model_name": selected.model_name,
        "selected_model_status": selected.status,
        "record_uids": tuple(str(row["record_uid"]) for row in rows),
        "source_name": source_name,
        "study_uid": _study_uid(rows, source_name),
        "trial_id": _one_value(rows, "trial_id"),
        "experiment_type": _one_value(rows, "experiment_type"),
        "experimental_design": _one_value(rows, "experimental_design"),
        "water_regime_normalized": _one_value(rows, "water_regime_normalized"),
        "season_normalized": _one_value(rows, "season_normalized"),
        "region": _one_value(rows, "region"),
        "province": _one_value(rows, "province"),
        "rice_variety": _one_value(rows, "rice_variety"),
        "planting_year": _one_value(rows, "planting_year"),
        "treatment_text_class": _one_value(rows, "treatment_text_class"),
        "n_split": _one_value(rows, "n_split"),
        "organic_fertilizer_present": any(bool(row.get("organic_fertilizer_present")) for row in rows),
        "biofertilizer_present": any(bool(row.get("biofertilizer_present")) for row in rows),
        "series_distinct_n_level_count": len(set(n_rates)),
        "series_has_zero_n": bool(zero_yields),
        "series_has_high_n": any(bool(row.get("is_high_n")) for row in rows),
        "series_p_constant": rows[0].get("series_p_constant"),
        "series_k_constant": rows[0].get("series_k_constant"),
        "p_rate_kg_p2o5_ha": finite_number(rows[0].get("p_rate_kg_p2o5_ha")),
        "k_rate_kg_k2o_ha": finite_number(rows[0].get("k_rate_kg_k2o_ha")),
        "series_observed_n_min_kg_ha": float(min(n_rates)),
        "series_observed_n_max_kg_ha": float(max(n_rates)),
        "curve_shape_class": selected.curve_shape_class,
        "optimum_status": selected.optimum_status,
        "agronomic_optimum_n_kg_ha": selected.agronomic_optimum_n_kg_ha,
        "plateau_onset_n_kg_ha": selected.plateau_onset_n_kg_ha,
        "predicted_max_yield_t_ha": selected.predicted_max_yield_t_ha,
        "predicted_observed_domain_peak_yield_t_ha": selected.predicted_observed_domain_peak_yield_t_ha,
        "finite_maximum_yield_t_ha": selected.finite_maximum_yield_t_ha,
        "fitted_asymptote_yield_t_ha": selected.fitted_asymptote_yield_t_ha,
        "supported_max_yield_t_ha": selected.supported_max_yield_t_ha,
        "maximum_reference_basis": selected.maximum_reference_basis,
        "maximum_proximity_status": selected.maximum_proximity_status,
        "observed_max_yield_t_ha": observed_max,
        "observed_max_gap_to_finite_maximum_t_ha": finite_gap,
        "observed_max_gap_to_supported_maximum_t_ha": supported_gap,
        "observed_max_attainment_fraction": attainment,
        "evidence_strength": evidence_strength,
        "baseline_response_status": (
            "available"
            if baseline_metrics_enabled and zero_yields
            else "unavailable_no_numeric_zero_n"
            if baseline_metrics_enabled
            else "disabled_pending_ELG_10"
        ),
        "yield_at_zero_n_t_ha": (
            float(statistics.fmean(zero_yields))
            if baseline_metrics_enabled and zero_yields
            else None
        ),
        "yield_response_above_zero_n_t_ha": (
            float(max(yields) - statistics.fmean(zero_yields))
            if baseline_metrics_enabled and zero_yields
            else None
        ),
        "recommendation_yield_gap_t_ha": None,
        "target_yield_gap_t_ha": None,
        "target_yield_status": "not_configured",
        "model_attempt_record": selected_record,
        "reason_codes": (
            ()
            if baseline_metrics_enabled
            else ("BASELINE_RESPONSE_METRICS_DISABLED_PENDING_ELG_10",)
        ),
    }


def _all_credible_curve_row(
    rows: Sequence[Mapping[str, Any]],
    credible: Sequence[ModelAttempt],
    *,
    zero_tolerance: float,
    baseline_metrics_enabled: bool,
) -> dict[str, Any] | None:
    """Build one series row without ranking credible candidates against each other."""

    if not credible:
        return None
    row = _curve_row(
        rows,
        credible[0],
        zero_tolerance=zero_tolerance,
        baseline_metrics_enabled=baseline_metrics_enabled,
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
    if len(credible) == 1:
        sole = credible[0]
        row.update(
            {
                "model_disagreement_status": "single_credible_candidate",
                "sole_credible_model_attempt_uid": sole.model_attempt_uid,
                "sole_credible_model_name": sole.model_name,
                "reason_codes": tuple(
                    sorted(
                        set(row["reason_codes"])
                        | {"MODEL_SELECTION_DISABLED_MOD_02_OPTION_D"}
                    )
                ),
            }
        )
        return row

    row.update(
        {
            "model_disagreement_status": "multiple_credible_candidates_rule_unapproved",
            "sole_credible_model_attempt_uid": None,
            "sole_credible_model_name": None,
            "curve_shape_class": "uncertain_or_mixed",
            "optimum_status": "SUPPRESSED_MODEL_DISAGREEMENT",
            "agronomic_optimum_n_kg_ha": None,
            "plateau_onset_n_kg_ha": None,
            "predicted_max_yield_t_ha": None,
            "predicted_observed_domain_peak_yield_t_ha": None,
            "finite_maximum_yield_t_ha": None,
            "fitted_asymptote_yield_t_ha": None,
            "supported_max_yield_t_ha": None,
            "maximum_reference_basis": "none",
            "maximum_proximity_status": "MODEL_DISAGREEMENT_UNRESOLVED",
            "observed_max_gap_to_finite_maximum_t_ha": None,
            "observed_max_gap_to_supported_maximum_t_ha": None,
            "observed_max_attainment_fraction": None,
            "evidence_strength": "multiple_credible_models_no_single_summary",
            "reason_codes": tuple(
                sorted(
                    set(row["reason_codes"])
                    | {
                        "MODEL_SELECTION_DISABLED_MOD_02_OPTION_D",
                        "MOD_05_DISAGREEMENT_RULE_UNAPPROVED",
                        "SINGLE_MODEL_CONCLUSION_SUPPRESSED",
                    }
                )
            ),
        }
    )
    return row


def _ranked_curve_row(
    rows: Sequence[Mapping[str, Any]],
    selected: ModelAttempt,
    credible: Sequence[ModelAttempt],
    *,
    zero_tolerance: float,
    baseline_metrics_enabled: bool,
) -> dict[str, Any] | None:
    row = _curve_row(
        rows,
        selected,
        zero_tolerance=zero_tolerance,
        baseline_metrics_enabled=baseline_metrics_enabled,
    )
    if row is None:
        return None
    row.update(
        {
            "model_reporting_policy": "aicc_then_grouped_prediction",
            "credible_model_count": len(credible),
            "credible_model_attempt_uids": tuple(
                attempt.model_attempt_uid for attempt in credible
            ),
            "credible_model_names": tuple(attempt.model_name for attempt in credible),
            "model_disagreement_status": "ranked_selection",
            "sole_credible_model_attempt_uid": (
                credible[0].model_attempt_uid if len(credible) == 1 else None
            ),
            "sole_credible_model_name": credible[0].model_name if len(credible) == 1 else None,
        }
    )
    return row


def _series_evidence_row(
    rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[ModelAttempt],
    selected: ModelAttempt | None,
    *,
    credible: Sequence[ModelAttempt] = (),
    reporting_policy: str = "aicc_then_grouped_prediction",
    zero_tolerance: float,
) -> dict[str, Any] | None:
    supported_tiers = {
        str(row.get("series_eligibility_tier") or row.get("eligibility_tier") or "")
        for row in rows
    }
    if not supported_tiers.intersection({"A", "B", "C"}):
        return None
    complete = [
        (n_rate, yield_value)
        for row in rows
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
    report_all = reporting_policy == _ALL_CREDIBLE_POLICY
    sole_credible = credible[0] if report_all and len(credible) == 1 else None
    reported_attempt = sole_credible if report_all else selected
    evidence_reasons: list[str] = []
    if len(levels) == 2:
        evidence_strength = "two_level_contrast_only"
        evidence_status = "contrast_only"
        evidence_reasons.append("TWO_LEVEL_CONTRAST_ONLY")
    elif report_all and len(credible) > 1:
        evidence_strength = "multiple_credible_models_no_single_summary"
        evidence_status = "credible_model_set_reported"
        evidence_reasons.extend(
            (
                "MODEL_SELECTION_DISABLED_MOD_02_OPTION_D",
                "MOD_05_DISAGREEMENT_RULE_UNAPPROVED",
                "SINGLE_MODEL_CONCLUSION_SUPPRESSED",
            )
        )
    elif len(levels) == 3 and reported_attempt is not None and reported_attempt.model_name == "linear":
        evidence_strength = "three_level_linear_only"
        evidence_status = "credible_model_reported" if report_all else "curve_model_selected"
    elif reported_attempt is not None:
        evidence_strength = "four_plus_level_curve"
        evidence_status = "credible_model_reported" if report_all else "curve_model_selected"
    elif len(levels) < 2:
        evidence_strength = "insufficient_n_level_support"
        evidence_status = "unsupported"
        evidence_reasons.append("INSUFFICIENT_N_LEVEL_SUPPORT")
    else:
        evidence_strength = "no_reportable_curve_model"
        evidence_status = "unsupported"
        evidence_reasons.append("NO_REPORTABLE_CURVE_MODEL")
    if report_all and credible:
        evidence_reasons.append("MODEL_SELECTION_DISABLED_MOD_02_OPTION_D")
    source_name = _one_value(rows, "source_name")
    multiple_credible = report_all and len(credible) > 1
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
        "curve_shape_class": (
            "uncertain_or_mixed"
            if multiple_credible
            else reported_attempt.curve_shape_class
            if reported_attempt is not None
            else "unavailable"
        ),
        "optimum_status": (
            "SUPPRESSED_MODEL_DISAGREEMENT"
            if multiple_credible
            else reported_attempt.optimum_status
            if reported_attempt is not None
            else "unavailable"
        ),
        "maximum_reference_basis": (
            "none"
            if multiple_credible
            else reported_attempt.maximum_reference_basis
            if reported_attempt is not None
            else "none"
        ),
        "maximum_proximity_status": (
            "MODEL_DISAGREEMENT_UNRESOLVED"
            if multiple_credible
            else reported_attempt.maximum_proximity_status
            if reported_attempt is not None
            else "unavailable"
        ),
        "target_yield_status": "not_configured",
        "model_reporting_policy": reporting_policy,
        "model_disagreement_status": (
            "multiple_credible_candidates_rule_unapproved"
            if multiple_credible
            else "single_credible_candidate"
            if sole_credible is not None
            else "ranked_selection"
            if selected is not None
            else "no_credible_candidate"
        ),
        "selected_model_attempt_uid": (
            selected.model_attempt_uid if not report_all and selected is not None else None
        ),
        "selected_model_name": selected.model_name if not report_all and selected is not None else None,
        "sole_credible_model_attempt_uid": (
            sole_credible.model_attempt_uid if sole_credible is not None else None
        ),
        "sole_credible_model_name": sole_credible.model_name if sole_credible is not None else None,
        "credible_model_attempt_uids": tuple(attempt.model_attempt_uid for attempt in credible),
        "credible_model_names": tuple(attempt.model_name for attempt in credible),
        "credible_model_count": len(credible),
        "model_attempt_uids": tuple(attempt.model_attempt_uid for attempt in attempts),
        "model_reason_codes": tuple(sorted({reason for attempt in attempts for reason in attempt.reason_codes})),
        "reason_codes": tuple(sorted(set(evidence_reasons))),
    }


def curve_fit_record_uids(
    records: Iterable[Mapping[str, Any]],
    *,
    primary_only: bool,
) -> tuple[str, ...]:
    """Return rows permitted in a curve fit without discarding evidence rows."""

    return tuple(
        str(record["record_uid"])
        for record in records
        if record.get("treatment_fit_role", "curve_candidate") != "comparison_only"
        and (
            not primary_only
            or (
                record.get("series_status") == "resolved"
                and (record.get("series_eligibility_tier") or record.get("eligibility_tier")) == "A"
            )
        )
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
    if fit_record_uids is None:
        fit_records = copied_records
    else:
        requested_uids = tuple(str(record_uid) for record_uid in fit_record_uids)
        if len(requested_uids) != len(set(requested_uids)):
            raise ValueError("fit_record_uids must be unique")
        available_uids = {str(record.get("record_uid", "")) for record in copied_records}
        unknown_uids = sorted(set(requested_uids) - available_uids)
        if unknown_uids:
            raise ValueError("fit_record_uids contains unknown records: " + ", ".join(unknown_uids))
        requested = set(requested_uids)
        fit_records = tuple(
            record for record in copied_records if str(record.get("record_uid", "")) in requested
        )
    attempts = fit_response_models(fit_records, model_names=model_names, policy=policy)
    by_series: dict[str, list[ModelAttempt]] = {}
    for attempt in attempts:
        by_series.setdefault(attempt.response_series_uid, []).append(attempt)
    reporting_policy = str(
        policy.get("model_selection_metric", "aicc_then_grouped_prediction")
    )
    if reporting_policy not in {"aicc_then_grouped_prediction", _ALL_CREDIBLE_POLICY}:
        raise ValueError(f"Unknown model reporting policy: {reporting_policy}")
    credible_by_series = {
        series_uid: credible_model_attempts(series_attempts)
        for series_uid, series_attempts in sorted(by_series.items())
    }
    credible_attempts_flat = tuple(
        attempt
        for series_uid in sorted(credible_by_series)
        for attempt in credible_by_series[series_uid]
    )
    selected_attempts = (
        ()
        if reporting_policy == _ALL_CREDIBLE_POLICY
        else tuple(
            selected
            for series_uid in sorted(by_series)
            if (selected := select_reportable_model(by_series[series_uid])) is not None
        )
    )
    selected_by_series = {
        attempt.response_series_uid: attempt for attempt in selected_attempts
    }
    grouped_rows = _series_rows(copied_records)
    grouped_fit_rows = _series_rows(fit_records)
    zero_tolerance = float(policy.get("convergence_tolerance", 1e-8))
    baseline_metrics_enabled = bool(
        policy.get("allow_baseline_response_metrics", False)
    )
    series_evidence_rows = tuple(
        evidence_row
        for series_uid in sorted(grouped_rows)
        if (
            evidence_row := _series_evidence_row(
                grouped_rows[series_uid],
                by_series.get(series_uid, ()),
                selected_by_series.get(series_uid),
                credible=credible_by_series.get(series_uid, ()),
                reporting_policy=reporting_policy,
                zero_tolerance=zero_tolerance,
            )
        ) is not None
    )
    if reporting_policy == _ALL_CREDIBLE_POLICY:
        curve_rows = tuple(
            curve
            for series_uid in sorted(credible_by_series)
            if (
                curve := _all_credible_curve_row(
                    grouped_fit_rows.get(series_uid, ()),
                    credible_by_series[series_uid],
                    zero_tolerance=zero_tolerance,
                    baseline_metrics_enabled=baseline_metrics_enabled,
                )
            )
            is not None
        )
        prediction_attempts = credible_attempts_flat
    else:
        curve_rows = tuple(
            curve
            for selected in selected_attempts
            if (
                curve := _ranked_curve_row(
                    grouped_fit_rows.get(selected.response_series_uid, ()),
                    selected,
                    credible_by_series.get(selected.response_series_uid, ()),
                    zero_tolerance=zero_tolerance,
                    baseline_metrics_enabled=baseline_metrics_enabled,
                )
            )
            is not None
        )
        prediction_attempts = selected_attempts
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
    return CurveEvidenceResult(
        reporting_policy=reporting_policy,
        model_attempts=tuple(attempts),
        credible_attempts=credible_attempts_flat,
        selected_attempts=selected_attempts,
        model_attempt_records=model_attempt_records,
        series_evidence_rows=series_evidence_rows,
        curve_rows=curve_rows,
        prediction_rows=predictions,
    )


__all__ = ["CurveEvidenceResult", "build_curve_evidence", "curve_fit_record_uids"]
