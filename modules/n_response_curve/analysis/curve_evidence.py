from __future__ import annotations

from dataclasses import dataclass
import hashlib
import statistics
from typing import Any, Iterable, Mapping, Sequence

from ..reporting.plots import prediction_rows
from .models import ModelAttempt, fit_response_models, model_attempt_record, select_reportable_model
from .values import finite_number


@dataclass(frozen=True)
class CurveEvidenceResult:
    """Per-series model ledger, selected curve features, and observed-domain predictions."""

    model_attempts: tuple[ModelAttempt, ...]
    selected_attempts: tuple[ModelAttempt, ...]
    model_attempt_records: tuple[dict[str, Any], ...]
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


def _curve_row(rows: Sequence[Mapping[str, Any]], selected: ModelAttempt, *, zero_tolerance: float) -> dict[str, Any] | None:
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
        "observed_max_yield_t_ha": float(max(yields)),
        "yield_at_zero_n_t_ha": float(statistics.fmean(zero_yields)) if zero_yields else None,
        "yield_response_above_zero_n_t_ha": float(max(yields) - statistics.fmean(zero_yields)) if zero_yields else None,
        "recommendation_yield_gap_t_ha": None,
        "target_yield_gap_t_ha": None,
        "model_attempt_record": selected_record,
    }


def build_curve_evidence(
    records: Iterable[Mapping[str, Any]],
    *,
    model_names: Sequence[str],
    policy: Mapping[str, Any],
) -> CurveEvidenceResult:
    """Fit configured candidates, retain all attempts, and expose only supported curve-level outcomes."""

    copied_records = tuple(dict(record) for record in records)
    attempts = fit_response_models(copied_records, model_names=model_names, policy=policy)
    by_series: dict[str, list[ModelAttempt]] = {}
    for attempt in attempts:
        by_series.setdefault(attempt.response_series_uid, []).append(attempt)
    selected_attempts = tuple(
        selected
        for series_uid in sorted(by_series)
        if (selected := select_reportable_model(by_series[series_uid])) is not None
    )
    grouped_rows = _series_rows(copied_records)
    zero_tolerance = float(policy.get("convergence_tolerance", 1e-8))
    curve_rows = tuple(
        curve
        for selected in selected_attempts
        if (curve := _curve_row(grouped_rows.get(selected.response_series_uid, ()), selected, zero_tolerance=zero_tolerance)) is not None
    )
    selected_by_series = {attempt.response_series_uid: attempt for attempt in selected_attempts}
    rows_with_predictions = [
        row
        for row in curve_rows
        if row["response_series_uid"] in selected_by_series
    ]
    predictions = tuple(
        prediction
        for row in rows_with_predictions
        for prediction in prediction_rows(selected_by_series[row["response_series_uid"]])
    )
    return CurveEvidenceResult(
        model_attempts=tuple(attempts),
        selected_attempts=selected_attempts,
        model_attempt_records=tuple(model_attempt_record(attempt) for attempt in attempts),
        curve_rows=curve_rows,
        prediction_rows=predictions,
    )


__all__ = ["CurveEvidenceResult", "build_curve_evidence"]
