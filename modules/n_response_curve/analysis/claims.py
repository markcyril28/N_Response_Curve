from __future__ import annotations

from collections import Counter
import math
from types import MappingProxyType
from typing import Any, Mapping

from n_response_curve.analysis.analysis_matrix import AnalysisCandidate, AnalysisRegistry
from n_response_curve.data.provenance import stable_identifier, stable_json_sha256


_CLAIM_POLICY_FIELDS = frozenset(
    {
        "rule_id",
        "review_status",
        "result_selector",
        "effect_field",
        "meaningful_direction",
        "minimum_meaningful_magnitude",
        "required_sensitivities",
        "sensitivity_require_adjusted_support",
        "predictive_validation_required",
    }
)
_WITHHELD_CLAIM_POLICY_FIELDS = frozenset(
    {"rule_id", "review_status", "reason"}
)
_MEANINGFUL_DIRECTIONS = frozenset({"increase", "decrease", "absolute"})


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _string_list(value: object, *, where: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{where} must be an array")
    normalized = tuple(
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    )
    if len(normalized) != len(value) or len(normalized) != len(set(normalized)):
        raise ValueError(f"{where} must contain unique nonempty strings")
    return normalized


def validate_claim_policy(
    policy: object,
    *,
    declared_sensitivities: object,
    alpha: object,
) -> dict[str, Any]:
    """Validate one reviewed claim policy without supplying scientific defaults."""

    if not isinstance(policy, Mapping):
        raise ValueError("claim policy must be an object")
    review_status = policy.get("review_status")
    required_fields = (
        _WITHHELD_CLAIM_POLICY_FIELDS
        if review_status == "withheld"
        else _CLAIM_POLICY_FIELDS
    )
    observed = set(policy)
    if observed != required_fields:
        missing = ", ".join(sorted(required_fields - observed)) or "none"
        extra = ", ".join(sorted(observed - required_fields)) or "none"
        raise ValueError(
            "claim policy fields do not match the required schema; "
            f"missing={missing}; extra={extra}"
        )
    rule_id = policy["rule_id"]
    if not isinstance(rule_id, str) or not rule_id.strip():
        raise ValueError("claim policy rule_id must be a nonempty string")
    if review_status not in {"approved", "withheld"}:
        raise ValueError("claim policy review_status must be approved or withheld")
    if review_status != "approved":
        reason = policy["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("withheld claim policy reason must be a nonempty string")
        return dict(policy)
    selector = policy["result_selector"]
    if not isinstance(selector, Mapping) or not selector:
        raise ValueError("approved claim policy result_selector must be a nonempty object")
    if any(not isinstance(key, str) or not key.strip() for key in selector):
        raise ValueError("approved claim policy result_selector keys must be nonempty strings")
    effect_field = policy["effect_field"]
    if not isinstance(effect_field, str) or not effect_field.strip():
        raise ValueError("approved claim policy effect_field must be a nonempty string")
    direction = policy["meaningful_direction"]
    if direction not in _MEANINGFUL_DIRECTIONS:
        raise ValueError(
            "approved claim policy meaningful_direction must be increase, decrease, or absolute"
        )
    threshold = _finite_number(policy["minimum_meaningful_magnitude"])
    if threshold is None or threshold <= 0.0:
        raise ValueError(
            "approved claim policy minimum_meaningful_magnitude must be finite and positive"
        )
    required = _string_list(
        policy["required_sensitivities"],
        where="approved claim policy required_sensitivities",
    )
    declared = _string_list(
        declared_sensitivities,
        where="prespecified sensitivities",
    )
    if set(required) != set(declared) or len(required) != len(declared):
        raise ValueError(
            "approved claim policy required_sensitivities must exactly match the "
            "prespecified sensitivity set"
        )
    for field in (
        "sensitivity_require_adjusted_support",
        "predictive_validation_required",
    ):
        if not isinstance(policy[field], bool):
            raise ValueError(f"approved claim policy {field} must be boolean")
    normalized_alpha = _finite_number(alpha)
    if normalized_alpha is None or not 0.0 < normalized_alpha <= 1.0:
        raise ValueError("approved claim policy requires a prespecified alpha in (0, 1]")
    normalized = dict(policy)
    normalized["result_selector"] = MappingProxyType(dict(selector))
    normalized["required_sensitivities"] = required
    normalized["minimum_meaningful_magnitude"] = threshold
    return normalized


def _validated_policy(
    candidate: AnalysisCandidate,
) -> tuple[Mapping[str, Any] | None, tuple[str, ...]]:
    contrast = candidate.prespecified_contrast
    policy = contrast.get("claim_policy")
    if policy is None:
        return None, ("APPROVED_CLAIM_POLICY_MISSING",)
    normalized = validate_claim_policy(
        policy,
        declared_sensitivities=contrast.get("sensitivities", ()),
        alpha=contrast.get("alpha"),
    )
    if normalized["review_status"] != "approved":
        return normalized, ("APPROVED_CLAIM_POLICY_MISSING",)
    return normalized, ()


def _result_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    engine_result = row.get("engine_result")
    payload = dict(engine_result) if isinstance(engine_result, Mapping) else {}
    payload.update(row)
    return payload


def _selector_matches(row: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    return all(row.get(key) == value for key, value in selector.items())


def _meaningful(effect: float, *, direction: str, threshold: float) -> bool:
    if direction == "increase":
        return effect >= threshold
    if direction == "decrease":
        return effect <= -threshold
    return abs(effect) >= threshold


def _evidence_binding_matches(
    evidence: Mapping[str, Any],
    *,
    candidate: AnalysisCandidate,
    result_id: object,
    policy: Mapping[str, Any],
    sensitivity_id: str | None = None,
) -> bool:
    expected = {
        "candidate_id": candidate.candidate_id,
        "result_id": result_id,
        "hypothesis_id": candidate.hypothesis_id,
        "specification_hash": candidate.specification_hash,
        "dataset_version_id": candidate.dataset_version_id,
        "claim_policy_sha256": stable_json_sha256(policy),
    }
    if any(evidence.get(key) != value for key, value in expected.items()):
        return False
    selector = evidence.get("result_selector")
    if not isinstance(selector, Mapping) or dict(selector) != dict(policy["result_selector"]):
        return False
    return sensitivity_id is None or evidence.get("sensitivity_id") == sensitivity_id


def _base_row(
    candidate: AnalysisCandidate,
    *,
    rule_id: str | None,
    result_id: str | None,
) -> dict[str, Any]:
    return {
        "claim_evaluation_id": stable_identifier(
            "claim",
            {
                "candidate_id": candidate.candidate_id,
                "rule_id": rule_id,
                "result_id": result_id,
            },
        ),
        "candidate_id": candidate.candidate_id,
        "hypothesis_id": candidate.hypothesis_id,
        "rule_id": rule_id,
        "result_id": result_id,
        "claim_status": "not_callable",
        "adjusted_support_status": "not_evaluated",
        "meaningful_magnitude_status": "not_evaluated",
        "sensitivity_stability_status": "not_evaluated",
        "predictive_validation_status": "not_required",
        "terminal_interpretability_status": "not_evaluated",
        "reason_codes": [],
    }


def _classify_candidate(
    candidate: AnalysisCandidate,
    result_rows: tuple[Mapping[str, Any], ...],
    *,
    sensitivity_evidence: Mapping[str, Mapping[str, Any]],
    predictive_evidence: Mapping[str, Any] | None,
    terminal_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    policy, unavailable_reasons = _validated_policy(candidate)
    rule_id = (
        str(policy["rule_id"])
        if policy is not None and isinstance(policy.get("rule_id"), str)
        else None
    )
    if unavailable_reasons:
        row = _base_row(candidate, rule_id=rule_id, result_id=None)
        row["reason_codes"] = list(unavailable_reasons)
        return row
    assert policy is not None

    evidence_role = candidate.prespecified_contrast.get("evidence_role")
    if evidence_role not in {"primary", "secondary"}:
        row = _base_row(candidate, rule_id=rule_id, result_id=None)
        row["reason_codes"] = [
            "EXPLORATORY_RESULT_NOT_CLAIM_ELIGIBLE"
            if evidence_role == "exploratory"
            else "CLAIM_EVIDENCE_ROLE_NOT_ELIGIBLE"
        ]
        return row

    terminal_status = (
        str(terminal_evidence.get("terminal_status") or "")
        if isinstance(terminal_evidence, Mapping)
        else ""
    )
    inferential_interpretability = (
        str(terminal_evidence.get("inferential_interpretability") or "")
        if isinstance(terminal_evidence, Mapping)
        else ""
    )
    if terminal_status != "run" or inferential_interpretability != "interpretable":
        row = _base_row(candidate, rule_id=rule_id, result_id=None)
        row["terminal_interpretability_status"] = "not_interpretable"
        row["reason_codes"] = ["TERMINAL_RESULT_NOT_INTERPRETABLE"]
        return row

    selector = policy["result_selector"]
    assert isinstance(selector, Mapping)
    matching = tuple(row for row in result_rows if _selector_matches(row, selector))
    if len(matching) != 1:
        row = _base_row(candidate, rule_id=rule_id, result_id=None)
        row["reason_codes"] = [
            "CLAIM_RESULT_SELECTOR_UNMATCHED"
            if not matching
            else "CLAIM_RESULT_SELECTOR_AMBIGUOUS"
        ]
        return row

    result = matching[0]
    result_id = result.get("result_id")
    row = _base_row(
        candidate,
        rule_id=rule_id,
        result_id=str(result_id) if result_id is not None else None,
    )
    row["claim_policy_sha256"] = stable_json_sha256(policy)
    row["terminal_interpretability_status"] = "interpretable"
    reasons: list[str] = []
    if result.get("status") != "reconciled":
        reasons.append("MULTIPLICITY_RECONCILIATION_INCOMPLETE")
    adjusted_p = _finite_number(result.get("adjusted_p_value"))
    alpha = float(candidate.prespecified_contrast["alpha"])
    if adjusted_p is None:
        reasons.append("ADJUSTED_P_VALUE_UNAVAILABLE")
    elif adjusted_p <= alpha:
        row["adjusted_support_status"] = "supported"
    else:
        row["adjusted_support_status"] = "not_supported"
        reasons.append("ADJUSTED_STATISTICAL_SUPPORT_NOT_MET")

    effect_field = str(policy["effect_field"])
    effect = _finite_number(result.get(effect_field))
    if effect is None:
        reasons.append("CLAIM_EFFECT_ESTIMATE_UNAVAILABLE")
    elif _meaningful(
        effect,
        direction=str(policy["meaningful_direction"]),
        threshold=float(policy["minimum_meaningful_magnitude"]),
    ):
        row["meaningful_magnitude_status"] = "supported"
    else:
        row["meaningful_magnitude_status"] = "not_supported"
        reasons.append("MEANINGFUL_MAGNITUDE_NOT_MET")

    required_sensitivities = tuple(policy["required_sensitivities"])
    missing_sensitivities = tuple(
        sensitivity_id
        for sensitivity_id in required_sensitivities
        if sensitivity_id not in sensitivity_evidence
        or sensitivity_evidence[sensitivity_id].get("status") != "completed"
    )
    if missing_sensitivities:
        reasons.append("SENSITIVITY_EVIDENCE_INCOMPLETE")
    else:
        sensitivity_failures = False
        sensitivity_binding_mismatch = False
        for sensitivity_id in required_sensitivities:
            evidence = sensitivity_evidence[sensitivity_id]
            if not _evidence_binding_matches(
                evidence,
                candidate=candidate,
                result_id=result_id,
                policy=policy,
                sensitivity_id=sensitivity_id,
            ):
                sensitivity_binding_mismatch = True
                continue
            sensitivity_effect = _finite_number(evidence.get(effect_field))
            if sensitivity_effect is None or not _meaningful(
                sensitivity_effect,
                direction=str(policy["meaningful_direction"]),
                threshold=float(policy["minimum_meaningful_magnitude"]),
            ):
                sensitivity_failures = True
            if policy["sensitivity_require_adjusted_support"]:
                sensitivity_p = _finite_number(evidence.get("adjusted_p_value"))
                if sensitivity_p is None or sensitivity_p > alpha:
                    sensitivity_failures = True
        if sensitivity_binding_mismatch:
            reasons.append("SENSITIVITY_EVIDENCE_BINDING_MISMATCH")
        elif sensitivity_failures:
            row["sensitivity_stability_status"] = "not_supported"
            reasons.append("SENSITIVITY_STABILITY_NOT_MET")
        else:
            row["sensitivity_stability_status"] = "supported"

    if policy["predictive_validation_required"]:
        if predictive_evidence is None:
            row["predictive_validation_status"] = "not_evaluated"
            reasons.append("PREDICTIVE_VALIDATION_EVIDENCE_MISSING")
        elif not _evidence_binding_matches(
            predictive_evidence,
            candidate=candidate,
            result_id=result_id,
            policy=policy,
        ):
            row["predictive_validation_status"] = "not_evaluated"
            reasons.append("PREDICTIVE_EVIDENCE_BINDING_MISMATCH")
        elif predictive_evidence.get("status") == "validated":
            row["predictive_validation_status"] = "supported"
        else:
            row["predictive_validation_status"] = "not_supported"
            reasons.append("PREDICTIVE_VALIDATION_NOT_MET")

    incomplete_reasons = {
        "MULTIPLICITY_RECONCILIATION_INCOMPLETE",
        "ADJUSTED_P_VALUE_UNAVAILABLE",
        "CLAIM_EFFECT_ESTIMATE_UNAVAILABLE",
        "SENSITIVITY_EVIDENCE_INCOMPLETE",
        "SENSITIVITY_EVIDENCE_BINDING_MISMATCH",
        "PREDICTIVE_VALIDATION_EVIDENCE_MISSING",
        "PREDICTIVE_EVIDENCE_BINDING_MISMATCH",
    }
    if incomplete_reasons & set(reasons):
        row["claim_status"] = "not_callable"
    elif reasons:
        row["claim_status"] = "not_supported"
    else:
        row["claim_status"] = "supported_association"
    row["reason_codes"] = sorted(set(reasons))
    return row


def build_runtime_claim_evidence(
    registry: AnalysisRegistry,
    multiplicity_reconciliation: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Bind completed prespecified sensitivity candidates to their primary claim.

    Sensitivity candidates are generated from the reviewed effective hypothesis
    snapshot. This producer never infers a missing sensitivity, selector, or
    threshold; absent, ambiguous, or unreconciled outputs remain absent so the
    classifier fails closed.
    """

    raw_results = multiplicity_reconciliation.get("results_by_reconciliation_id", {})
    if not isinstance(raw_results, Mapping):
        raise ValueError("multiplicity reconciliation results must be an object")
    results_by_candidate: dict[str, list[Mapping[str, Any]]] = {}
    for raw_row in raw_results.values():
        if not isinstance(raw_row, Mapping):
            raise ValueError("multiplicity reconciliation result rows must be objects")
        row = _result_payload(raw_row)
        candidate_id = row.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id:
            results_by_candidate.setdefault(candidate_id, []).append(row)

    sensitivity_candidates: dict[tuple[str, str], AnalysisCandidate] = {}
    for candidate in registry.candidates:
        contrast = candidate.prespecified_contrast
        if contrast.get("evidence_role") != "sensitivity":
            continue
        parent_hypothesis_id = contrast.get("parent_hypothesis_id")
        sensitivity_id = contrast.get("sensitivity_id")
        if (
            isinstance(parent_hypothesis_id, str)
            and parent_hypothesis_id
            and isinstance(sensitivity_id, str)
            and sensitivity_id
        ):
            key = (parent_hypothesis_id, sensitivity_id)
            if key in sensitivity_candidates:
                raise ValueError(
                    "multiple runtime candidates satisfy one prespecified sensitivity"
                )
            sensitivity_candidates[key] = candidate

    evidence_by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
    for candidate in registry.candidates:
        if candidate.prespecified_contrast.get("evidence_role") not in {
            "primary",
            "secondary",
        }:
            continue
        policy, reasons = _validated_policy(candidate)
        if reasons or policy is None:
            continue
        selector = policy["result_selector"]
        assert isinstance(selector, Mapping)
        primary_matches = tuple(
            row
            for row in results_by_candidate.get(candidate.candidate_id, ())
            if _selector_matches(row, selector)
        )
        if len(primary_matches) != 1:
            continue
        result_id = primary_matches[0].get("result_id")
        for sensitivity_id in policy["required_sensitivities"]:
            sensitivity_candidate = sensitivity_candidates.get(
                (str(candidate.hypothesis_id), str(sensitivity_id))
            )
            if sensitivity_candidate is None:
                continue
            sensitivity_matches = tuple(
                row
                for row in results_by_candidate.get(
                    sensitivity_candidate.candidate_id,
                    (),
                )
                if _selector_matches(row, selector)
            )
            if len(sensitivity_matches) != 1:
                continue
            sensitivity_result = dict(sensitivity_matches[0])
            if sensitivity_result.get("status") != "reconciled":
                continue
            sensitivity_result.update(
                {
                    "status": "completed",
                    "candidate_id": candidate.candidate_id,
                    "result_id": result_id,
                    "hypothesis_id": candidate.hypothesis_id,
                    "specification_hash": candidate.specification_hash,
                    "dataset_version_id": candidate.dataset_version_id,
                    "claim_policy_sha256": stable_json_sha256(policy),
                    "result_selector": dict(selector),
                    "sensitivity_id": sensitivity_id,
                    "sensitivity_candidate_id": sensitivity_candidate.candidate_id,
                    "sensitivity_specification_hash": (
                        sensitivity_candidate.specification_hash
                    ),
                }
            )
            evidence_by_candidate.setdefault(candidate.candidate_id, {})[
                str(sensitivity_id)
            ] = sensitivity_result
    return evidence_by_candidate


def classify_claim_evidence(
    registry: AnalysisRegistry,
    multiplicity_reconciliation: Mapping[str, Any],
    *,
    sensitivity_evidence_by_candidate: Mapping[
        str, Mapping[str, Mapping[str, Any]]
    ]
    | None = None,
    predictive_evidence_by_candidate: Mapping[str, Mapping[str, Any]] | None = None,
    terminal_statuses_by_candidate: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate noncausal positive-claim prerequisites after multiplicity reconciliation.

    Missing policy or evidence remains explicit and non-callable. Only one reviewed,
    uniquely selected result can become a supported association, and only after its
    direction, meaningful magnitude, adjusted support, required sensitivities, and
    any declared predictive-validation prerequisite all pass.
    """

    raw_results = multiplicity_reconciliation.get("results_by_reconciliation_id", {})
    if not isinstance(raw_results, Mapping):
        raise ValueError("multiplicity reconciliation results must be an object")
    results_by_candidate: dict[str, list[Mapping[str, Any]]] = {}
    for raw_row in raw_results.values():
        if not isinstance(raw_row, Mapping):
            raise ValueError("multiplicity reconciliation result rows must be objects")
        row = _result_payload(raw_row)
        candidate_id = row.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id:
            results_by_candidate.setdefault(candidate_id, []).append(row)

    sensitivities = sensitivity_evidence_by_candidate or {}
    predictions = predictive_evidence_by_candidate or {}
    terminal_statuses = terminal_statuses_by_candidate or {}
    rows = tuple(
        _classify_candidate(
            candidate,
            tuple(results_by_candidate.get(candidate.candidate_id, ())),
            sensitivity_evidence=sensitivities.get(candidate.candidate_id, {}),
            predictive_evidence=predictions.get(candidate.candidate_id),
            terminal_evidence=terminal_statuses.get(candidate.candidate_id),
        )
        for candidate in registry.candidates
        if candidate.hypothesis_id is not None
    )
    counts = Counter(str(row["claim_status"]) for row in rows)
    return {
        "schema_version": 1,
        "claim_vocabulary": [
            "supported_association",
            "not_supported",
            "not_callable",
        ],
        "status": "not_applicable" if not rows else "classified",
        "candidate_count": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "rows": list(rows),
    }


__all__ = [
    "build_runtime_claim_evidence",
    "classify_claim_evidence",
    "validate_claim_policy",
]
