from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


_TIER_RANK = {"A": 0, "B": 1, "C": 2, "D": 3}
# The simplest supported response curve estimates an intercept and a slope.
_MINIMUM_CURVE_PARAMETER_COUNT = 2


@dataclass(frozen=True)
class EligibilityResult:
    """Row-level A–D eligibility ledger and its resolved-series evidence metrics."""

    ledger: tuple[dict[str, Any], ...]
    series_metrics: Mapping[str, Mapping[str, Any]]
    critical_record_uids: tuple[str, ...]


def _parsed_number(record: Mapping[str, Any], value_key: str, status_key: str) -> float | None:
    value = record.get(value_key)
    if record.get(status_key) != "parsed" or not isinstance(value, (int, float)):
        return None
    return float(value)


def _constant(values: list[float], tolerance: float) -> bool | None:
    if not values:
        return None
    return max(values) - min(values) <= tolerance


def _distinct_n_levels(values: list[float], tolerance: float) -> tuple[float, ...]:
    levels: list[float] = []
    for value in sorted(values):
        if not levels or abs(value - levels[-1]) > tolerance:
            levels.append(value)
    return tuple(levels)


def _series_metrics(records: Iterable[Mapping[str, Any]], policy: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    nutrient_tolerance = float(policy["constant_nutrient_tolerance"])
    n_level_tolerance = float(policy["n_level_tolerance_kg_ha"])
    high_n_threshold = float(policy.get("high_n_review_threshold_kg_ha", float("inf")))
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        series_uid = record.get("response_series_uid")
        if record.get("series_status") == "resolved" and isinstance(series_uid, str) and series_uid:
            grouped.setdefault(series_uid, []).append(record)

    metrics: dict[str, dict[str, Any]] = {}
    for series_uid, rows in grouped.items():
        complete = [
            row
            for row in rows
            if _parsed_number(row, "n_rate_kg_ha", "n_rate_parse_status") is not None
            and _parsed_number(row, "yield_t_ha", "yield_parse_status") is not None
            and row.get("yield_unit_status") != "conflict"
        ]
        n_rates = [
            value
            for row in complete
            if (value := _parsed_number(row, "n_rate_kg_ha", "n_rate_parse_status")) is not None
        ]
        n_levels = _distinct_n_levels(n_rates, n_level_tolerance)
        p_complete = [
            value
            for row in complete
            if (value := _parsed_number(row, "p_rate_kg_p2o5_ha", "p_rate_parse_status")) is not None
        ]
        k_complete = [
            value
            for row in complete
            if (value := _parsed_number(row, "k_rate_kg_k2o_ha", "k_rate_parse_status")) is not None
        ]
        metrics[series_uid] = {
            "row_count": len(rows),
            "complete_observation_count": len(complete),
            "distinct_n_level_count": len(n_levels),
            "observed_n_min_kg_ha": min(n_rates) if n_rates else None,
            "observed_n_max_kg_ha": max(n_rates) if n_rates else None,
            "has_zero_n": any(abs(value) <= n_level_tolerance for value in n_rates),
            "has_high_n": any(value > high_n_threshold for value in n_rates),
            "p_constant": _constant(p_complete, nutrient_tolerance) if len(p_complete) == len(complete) else None,
            "k_constant": _constant(k_complete, nutrient_tolerance) if len(k_complete) == len(complete) else None,
            "organic_or_biofertilizer_present": any(
                bool(row.get("organic_fertilizer_present")) or bool(row.get("biofertilizer_present"))
                for row in rows
            ),
            "experiment_priority_status": (
                "mixed"
                if len({str(row.get("experiment_priority_status", "unresolved")) for row in rows}) > 1
                else str(rows[0].get("experiment_priority_status", "unresolved"))
            ),
        }
    return metrics


def _reason_for_parse_status(prefix: str, status: object) -> str:
    normalized = str(status or "unresolved").upper()
    if normalized == "PARSED":
        return f"{prefix}_PARSED_VALUE_INVALID"
    return f"{prefix}_{normalized}"


def _unresolved_review_controls(record: Mapping[str, Any]) -> tuple[str, ...]:
    unresolved: list[str] = []
    exact_controls = {
        "representation_basis_status": {"reviewed"},
        "representation_review_status": {"resolved"},
        "schema_mapping_status": {"reviewed"},
        "cleaning_review_status": {
            "resolved_excluded",
            "resolved_flagged",
            "resolved_retained",
        },
        "restricted_release_status": {
            "not_restricted",
            "eligible_for_reviewed_public_projection",
            "internal_only_no_public_row_release",
        },
    }
    for field, allowed in exact_controls.items():
        if field in record and str(record.get(field) or "") not in allowed:
            unresolved.append(f"UNRESOLVED_REVIEW_CONTROL:{field}")
    # Category uncertainty is local to analyses that use that category.  It is
    # preserved in QC and the affected factor/fit gate, but it must not turn an
    # otherwise complete N/yield record into Tier D for every analysis.  This
    # implements the selected policy: retain unresolved categories and exclude
    # them only from the analyses that depend on those categories.
    local_category_status_fields = {
        "water_regime_normalization_status",
        "season_normalization_status",
        "region_normalization_status",
        "province_normalization_status",
        "rice_variety_normalization_status",
        "treatment_class_normalization_status",
    }
    for field, value in record.items():
        if not str(field).endswith("_normalization_status"):
            continue
        if str(field) in local_category_status_fields:
            continue
        status = str(value or "").casefold()
        if status.startswith(("unresolved", "review_required")) or status == "invalid_numeric":
            unresolved.append(f"UNRESOLVED_REVIEW_CONTROL:{field}")
    return tuple(sorted(set(unresolved)))


def _tier_and_reasons(
    record: Mapping[str, Any],
    metrics: Mapping[str, Any] | None,
    policy: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]]:
    reasons: set[str] = set(record.get("series_reason_codes", ()))
    unresolved_review_controls = _unresolved_review_controls(record)
    reasons.update(unresolved_review_controls)
    series_status = record.get("series_status")
    duplicate_status = record.get("duplicate_status", "unique")
    raw_duplicate_relationships = record.get("duplicate_relationships", ())
    duplicate_relationships = (
        {str(relationship) for relationship in raw_duplicate_relationships}
        if isinstance(raw_duplicate_relationships, (list, tuple, set, frozenset))
        else set()
    )
    if series_status != "resolved" or not record.get("response_series_uid"):
        reasons.add("UNRESOLVED_RESPONSE_SERIES")
    if (
        duplicate_status == "exact_duplicate_noncanonical"
        or "exact_duplicate_noncanonical" in duplicate_relationships
    ):
        reasons.add("EXACT_DUPLICATE_NONCANONICAL")
    if (
        duplicate_status == "probable_cross_source_duplicate"
        or "probable_cross_source_duplicate" in duplicate_relationships
    ):
        reasons.add("PROBABLE_CROSS_SOURCE_DUPLICATE")

    n_rate = _parsed_number(record, "n_rate_kg_ha", "n_rate_parse_status")
    if n_rate is None:
        reasons.add(_reason_for_parse_status("N_RATE", record.get("n_rate_parse_status")))
    yield_value = _parsed_number(record, "yield_t_ha", "yield_parse_status")
    if yield_value is None:
        reasons.add(_reason_for_parse_status("YIELD", record.get("yield_parse_status")))
    if record.get("yield_unit_status") == "conflict":
        reasons.add("YIELD_UNIT_CONFLICT")
    if record.get("n_rate_unit_status", "canonical") not in {
        "canonical",
        "canonical_reviewed",
        "converted_reviewed",
    }:
        reasons.add("N_RATE_UNIT_CONFLICT")
    if n_rate is not None and not policy["n_rate_min_kg_ha"] <= n_rate <= policy["n_rate_max_kg_ha"]:
        reasons.add("N_RATE_OUT_OF_RANGE")
    if yield_value is not None and not policy["yield_min_t_ha"] <= yield_value <= policy["yield_max_t_ha"]:
        reasons.add("YIELD_OUT_OF_RANGE")
    cleaning_membership = record.get("final_analytical_membership_status")
    if cleaning_membership == "excluded":
        reasons.add("FINAL_CLEANING_EXCLUSION")

    hard_blockers = {
        "UNRESOLVED_RESPONSE_SERIES",
        "EXACT_DUPLICATE_NONCANONICAL",
        "PROBABLE_CROSS_SOURCE_DUPLICATE",
        "YIELD_UNIT_CONFLICT",
        "N_RATE_UNIT_CONFLICT",
        "N_RATE_OUT_OF_RANGE",
        "YIELD_OUT_OF_RANGE",
        "FINAL_CLEANING_EXCLUSION",
    }
    if (
        hard_blockers.intersection(reasons)
        or unresolved_review_controls
        or n_rate is None
        or yield_value is None
    ):
        return "D", tuple(sorted(reasons))

    assert metrics is not None
    insufficient = False
    scope_status = str(record.get("scope_status", "in_scope"))
    if scope_status == "out_of_scope":
        reasons.add("OUT_OF_SCOPE_COUNTRY")
        insufficient = True
    elif scope_status != "in_scope":
        reasons.add("COUNTRY_SCOPE_UNRESOLVED")
        insufficient = True
    if metrics["complete_observation_count"] < policy["minimum_complete_n_yield"]:
        reasons.add("INSUFFICIENT_COMPLETE_N_YIELD_OBSERVATIONS")
        insufficient = True
    minimum_model_residual_df = int(policy["minimum_model_residual_df"])
    if metrics["complete_observation_count"] < _MINIMUM_CURVE_PARAMETER_COUNT + minimum_model_residual_df:
        reasons.add("INSUFFICIENT_MODEL_RESIDUAL_DF")
        insufficient = True
    if metrics["distinct_n_level_count"] < policy["minimum_distinct_n_levels"]:
        reasons.add("INSUFFICIENT_DISTINCT_N_LEVELS")
        insufficient = True
    flagged = False
    excluded = False
    if cleaning_membership == "included_flagged":
        reasons.add("FINAL_CLEANING_FLAG_ONLY")
        flagged = True

    organic_present = metrics["organic_or_biofertilizer_present"]
    organic_policy = policy.get("organic_policy", "retain_flagged")
    if organic_present:
        if organic_policy == "exclude_all":
            reasons.add("ORGANIC_OR_BIOFERTILIZER_EXCLUDED")
            excluded = True
        elif policy.get("primary_inorganic_only", False) or organic_policy in {"retain_flagged", "exclude_primary"}:
            reasons.add("ORGANIC_OR_BIOFERTILIZER")
            flagged = True

    p_k_policy = policy.get("p_k_policy", "retain_flagged")
    if metrics["p_constant"] is False or metrics["k_constant"] is False:
        if p_k_policy == "exclude_all":
            reasons.add("P_K_VARY_WITH_N_EXCLUDED")
            excluded = True
        elif p_k_policy in {"retain_flagged", "require_constant"}:
            reasons.add("P_K_VARY_WITH_N")
            flagged = True
    elif metrics["p_constant"] is None or metrics["k_constant"] is None:
        if p_k_policy == "exclude_all":
            reasons.add("P_K_UNRESOLVED_EXCLUDED")
            excluded = True
        elif p_k_policy in {"retain_flagged", "require_constant"}:
            reasons.add("P_K_UNRESOLVED")
            flagged = True

    if not metrics["has_zero_n"]:
        if policy.get("require_zero_n_for_primary", False) or policy.get("zero_n_policy") == "require_primary":
            reasons.add("ZERO_N_REQUIRED_FOR_PRIMARY")
            flagged = True
        elif policy.get("zero_n_policy") == "allow_flagged":
            reasons.add("ZERO_N_ABSENT_FLAGGED")
            # Interpretation flag only: the approved primary fit does not
            # require a zero-N observation.
    if bool(record.get("is_high_n")):
        reasons.add("HIGH_N_REVIEW")
        # Keep full-range high-N evidence until a trim rule is approved.
    if record.get("same_n_status") == "repeated_measurement":
        reasons.add("REPEATED_MEASUREMENT_AT_N_LEVEL")
        flagged = True

    if excluded:
        return "D", tuple(sorted(reasons))
    if insufficient:
        return "C", tuple(sorted(reasons))
    if flagged:
        return "B", tuple(sorted(reasons))
    reasons.add("PRIMARY_ELIGIBLE")
    return "A", tuple(sorted(reasons))


def _assign_series_tiers(ledger: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in ledger:
        row["series_eligibility_tier"] = None
        row["series_eligibility_reason_codes"] = ()
        series_uid = row.get("response_series_uid")
        if row.get("series_status") == "resolved" and isinstance(series_uid, str) and series_uid:
            grouped.setdefault(series_uid, []).append(row)

    for rows in grouped.values():
        series_tier = max((str(row["eligibility_tier"]) for row in rows), key=_TIER_RANK.__getitem__)
        series_reason_set = {
            reason
            for row in rows
            for reason in row["eligibility_reason_codes"]
        }
        if series_tier != "A":
            series_reason_set.discard("PRIMARY_ELIGIBLE")
        series_reasons = tuple(sorted(series_reason_set))
        for row in rows:
            row["series_eligibility_tier"] = series_tier
            row["series_eligibility_reason_codes"] = series_reasons


def assign_eligibility(
    records: Iterable[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
) -> EligibilityResult:
    """Assign reproducible A–D tiers without deleting observations or reason codes."""

    rows = [dict(record) for record in records]
    record_uids = [str(row.get("record_uid", "")) for row in rows]
    if not all(record_uids):
        raise ValueError("Every eligibility record needs a nonempty record_uid")
    if len(record_uids) != len(set(record_uids)):
        raise ValueError("Eligibility records must have unique record_uid values")
    for key in (
        "minimum_distinct_n_levels",
        "minimum_complete_n_yield",
        "minimum_model_residual_df",
        "constant_nutrient_tolerance",
        "n_level_tolerance_kg_ha",
        "n_rate_min_kg_ha",
        "n_rate_max_kg_ha",
        "yield_min_t_ha",
        "yield_max_t_ha",
    ):
        if key not in policy:
            raise ValueError(f"Eligibility policy is missing {key}")

    if int(policy["minimum_distinct_n_levels"]) < 3:
        raise ValueError("Eligibility policy minimum_distinct_n_levels must be >= 3")
    if float(policy["n_level_tolerance_kg_ha"]) <= 0:
        raise ValueError("Eligibility policy n_level_tolerance_kg_ha must be positive")
    if policy["n_rate_max_kg_ha"] <= policy["n_rate_min_kg_ha"]:
        raise ValueError("Eligibility policy N-rate range is invalid")
    if policy["yield_max_t_ha"] <= policy["yield_min_t_ha"]:
        raise ValueError("Eligibility policy yield range is invalid")

    metrics = _series_metrics(rows, policy)
    critical_codes = {str(code) for code in policy.get("critical_error_codes", ())}
    critical_record_uids: list[str] = []
    ledger: list[dict[str, Any]] = []
    for row in rows:
        series_uid = row.get("response_series_uid")
        tier, reasons = _tier_and_reasons(
            row,
            metrics.get(series_uid) if isinstance(series_uid, str) else None,
            policy,
        )
        row["eligibility_tier"] = tier
        row["eligibility_reason_codes"] = reasons
        row["has_critical_error"] = bool(set(reasons).intersection(critical_codes))
        if row["has_critical_error"]:
            critical_record_uids.append(str(row["record_uid"]))
        if isinstance(series_uid, str) and series_uid in metrics:
            row.update({f"series_{key}": value for key, value in metrics[series_uid].items()})
        ledger.append(row)

    _assign_series_tiers(ledger)

    return EligibilityResult(
        ledger=tuple(ledger),
        series_metrics=metrics,
        critical_record_uids=tuple(critical_record_uids),
    )


__all__ = ["EligibilityResult", "assign_eligibility"]
