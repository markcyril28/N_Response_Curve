from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
import re
import statistics
from typing import Any, Iterable, Mapping


_CONTEXT_FIELD_ALIASES = {
    "water_regime": "water_regime_normalized",
    "season": "season_normalized",
    "variety": "rice_variety",
    "recommendation_class": "treatment_text_class",
}
_MISSING_CONTEXT_VALUES = {"", "na", "n/a", "not stated", "none", "unresolved"}
_MIXED_CONTEXT_SEPARATOR = re.compile(r"[,;/]|\b(?:and|or)\b", re.IGNORECASE)
_DUPLICATE_STATUS_PRIORITY = {
    "exact_duplicate_noncanonical": 0,
    "probable_duplicate_noncanonical": 1,
    "probable_duplicate_review": 2,
    "exact_duplicate_canonical": 3,
    "probable_duplicate_canonical": 4,
    "unique": 5,
    "not_assessed": 6,
}
_REPEAT_CLASSIFICATIONS = {
    "exchangeable_replicates",
    "management_variant",
    "duplicate",
    "unequal_experimental_units",
}


@dataclass(frozen=True)
class DuplicateRuleSet:
    """Versioned deterministic exact keys and bounded probable-match rules."""

    version: str
    review_id: str
    exact_key_fields: tuple[str, ...]
    probable_key_fields: tuple[str, ...]
    probable_numeric_tolerances: Mapping[str, float]
    casefold_fields: tuple[str, ...] = ()
    probable_cross_source_only: bool = True


@dataclass(frozen=True)
class DuplicateAdjudication:
    """Human disposition for one probable duplicate group."""

    duplicate_group_uid: str
    rules_version: str
    disposition: str
    canonical_record_uid: str | None
    reviewer: str
    reviewed_on: str
    rationale: str


@dataclass(frozen=True)
class RepeatAdjudication:
    """Human classification of all records repeated at one N level."""

    record_uids: tuple[str, ...]
    classification: str
    reviewer: str
    reviewed_on: str
    rationale: str
    review_id: str


@dataclass(frozen=True)
class SeriesResolution:
    """Reviewable source-row ledger plus any approved repeat aggregates."""

    records: tuple[dict[str, Any], ...]
    aggregate_records: tuple[dict[str, Any], ...] = ()

    @property
    def analysis_records(self) -> tuple[dict[str, Any], ...]:
        source_rows = tuple(
            record
            for record in self.records
            if record.get("analytical_record_status") == "included"
        )
        aggregate_rows = tuple(
            record
            for record in self.aggregate_records
            if record.get("analytical_record_status") == "included"
        )
        return source_rows + aggregate_rows


def _stable_identifier(prefix: str, parts: Iterable[object]) -> str:
    encoded = json.dumps(
        list(parts),
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:24]}"


def _record_sort_key(record: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(record.get("source_uid", "")),
        int(record.get("source_row_number", 0) or 0),
        str(record["record_uid"]),
    )


def _nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty")
    return value.strip()


def _review_date(value: object, *, label: str) -> str:
    text = _nonempty(value, label=label)
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc
    return text


def _validate_duplicate_rules(rules: DuplicateRuleSet) -> None:
    _nonempty(rules.version, label="Duplicate rules version")
    _nonempty(rules.review_id, label="Duplicate rules review evidence")
    if not rules.exact_key_fields:
        raise ValueError("Duplicate rules require at least one deterministic exact key")
    if len(rules.exact_key_fields) != len(set(rules.exact_key_fields)):
        raise ValueError("Duplicate exact-key fields must be unique")
    if not rules.probable_key_fields:
        raise ValueError("Duplicate rules require at least one probable-match field")
    if len(rules.probable_key_fields) != len(set(rules.probable_key_fields)):
        raise ValueError("Duplicate probable-key fields must be unique")
    if set(rules.probable_numeric_tolerances) - set(rules.probable_key_fields):
        raise ValueError("Probable numeric tolerances reference fields outside the probable key")
    for field, tolerance in rules.probable_numeric_tolerances.items():
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise ValueError(f"Probable tolerance for {field!r} must be numeric")
        if not math.isfinite(float(tolerance)) or float(tolerance) < 0:
            raise ValueError(f"Probable tolerance for {field!r} must be finite and nonnegative")
    if set(rules.casefold_fields) - (
        set(rules.exact_key_fields) | set(rules.probable_key_fields)
    ):
        raise ValueError("Case-fold fields must belong to an exact or probable key")


def _raw_context_value(record: Mapping[str, Any], dimension: str) -> str | None:
    key = _CONTEXT_FIELD_ALIASES.get(dimension, dimension)
    value = record.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _context_value(record: Mapping[str, Any], dimension: str) -> str | None:
    normalized = _raw_context_value(record, dimension)
    if normalized is None or normalized.casefold() in _MISSING_CONTEXT_VALUES:
        return None
    return normalized


def _has_mixed_context(record: Mapping[str, Any], dimension: str) -> bool:
    raw_value = _raw_context_value(record, dimension)
    return (
        raw_value is not None
        and raw_value.casefold() not in _MISSING_CONTEXT_VALUES
        and _MIXED_CONTEXT_SEPARATOR.search(raw_value) is not None
    )


def _management_signature(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(record.get(key, ""))
        for key in (
            "treatment_id",
            "treatment_text_class",
            "canonical_treatment_class",
            "treatment",
            "p_rate_kg_p2o5_ha",
            "k_rate_kg_k2o_ha",
            "organic_fertilizer_present",
            "biofertilizer_present",
            "source_arm_role",
        )
    )


def _reviewed_series_arm_discriminator(
    record: Mapping[str, Any],
) -> tuple[str, ...]:
    source_arm_id = str(record.get("source_arm_id") or "").strip()
    source_arm_role = str(record.get("source_arm_role") or "").strip()
    if source_arm_id and source_arm_role != "canonical_source_row":
        return ("source_arm", source_arm_id)
    if (
        record.get("recommendation_set_membership_status")
        == "verified_context_comparable"
        and record.get("treatment_classification_status") == "resolved"
    ):
        treatment_class = str(record.get("treatment_text_class") or "").strip()
        if treatment_class in {"RCM", "FP", "NOPT_NPK"}:
            return ("reviewed_recommendation_class", treatment_class)
        if (
            record.get("nutrient_control_class") == "zero_n_with_pk"
            and record.get("is_zero_n_with_pk") is True
        ):
            return ("reviewed_recommendation_class", "zero_n_with_pk")
    return ()


def _reviewed_comparison_set_uid(record: Mapping[str, Any]) -> str:
    comparison_set_uid = str(record.get("comparison_set_uid") or "").strip()
    if not comparison_set_uid:
        return ""
    source_arm_role = str(record.get("source_arm_role") or "").strip()
    if source_arm_role and source_arm_role != "canonical_source_row":
        return comparison_set_uid
    if (
        record.get("recommendation_set_membership_status")
        == "verified_context_comparable"
        and record.get("treatment_classification_status") == "resolved"
    ):
        return comparison_set_uid
    return ""


def _add_sorted_unique(record: dict[str, Any], field: str, value: str) -> tuple[str, ...]:
    existing = record.get(field, ())
    values = (
        {str(item) for item in existing}
        if isinstance(existing, (list, tuple, set, frozenset))
        else set()
    )
    values.add(value)
    normalized = tuple(sorted(values))
    record[field] = normalized
    return normalized


def _append_reason(record: dict[str, Any], reason: str) -> None:
    _add_sorted_unique(record, "series_reason_codes", reason)


def _mark_unresolved(record: dict[str, Any], reason: str) -> None:
    record["response_series_uid"] = None
    record["series_status"] = "review"
    record["analytical_record_status"] = "review"
    _append_reason(record, reason)


def _add_duplicate_relationship(record: dict[str, Any], relationship: str) -> None:
    relationships = _add_sorted_unique(record, "duplicate_relationships", relationship)
    record["duplicate_status"] = min(
        relationships,
        key=lambda item: _DUPLICATE_STATUS_PRIORITY[item],
    )


def _add_duplicate_group(
    record: dict[str, Any],
    *,
    duplicate_group_uid: str,
    relationship: str,
    confidence: str,
    evidence_codes: tuple[str, ...],
    review_status: str,
    canonical_record_uid: str | None,
    rules_version: str,
    adjudication: DuplicateAdjudication | None = None,
) -> None:
    groups = [dict(group) for group in record.get("duplicate_groups", ())]
    group: dict[str, Any] = {
        "duplicate_group_uid": duplicate_group_uid,
        "relationship": relationship,
        "confidence": confidence,
        "evidence_codes": tuple(evidence_codes),
        "review_status": review_status,
        "canonical_record_uid": canonical_record_uid,
        "rules_version": rules_version,
    }
    if adjudication is not None:
        group.update(
            {
                "reviewer": adjudication.reviewer,
                "reviewed_on": adjudication.reviewed_on,
                "rationale": adjudication.rationale,
                "disposition": adjudication.disposition,
            }
        )
    groups.append(group)
    record["duplicate_groups"] = tuple(
        sorted(
            groups,
            key=lambda item: (
                str(item["duplicate_group_uid"]),
                str(item["relationship"]),
            ),
        )
    )


def _normalized_key_value(
    record: Mapping[str, Any],
    field: str,
    *,
    casefold_fields: set[str],
) -> object | None:
    value = record.get(field)
    if value is None:
        return None
    if isinstance(value, str):
        value = " ".join(value.strip().split())
        if not value or value.casefold() in _MISSING_CONTEXT_VALUES:
            return None
        return value.casefold() if field in casefold_fields else value
    if isinstance(value, (tuple, list)):
        return tuple(str(item) for item in value)
    return value


def _exact_signature(
    record: Mapping[str, Any],
    rules: DuplicateRuleSet,
) -> tuple[object, ...] | None:
    casefold_fields = set(rules.casefold_fields)
    values = tuple(
        _normalized_key_value(record, field, casefold_fields=casefold_fields)
        for field in rules.exact_key_fields
    )
    return None if any(value is None for value in values) else values


def _probable_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    rules: DuplicateRuleSet,
) -> bool:
    if rules.probable_cross_source_only and str(left.get("source_uid")) == str(
        right.get("source_uid")
    ):
        return False
    casefold_fields = set(rules.casefold_fields)
    for field in rules.probable_key_fields:
        left_value = _normalized_key_value(left, field, casefold_fields=casefold_fields)
        right_value = _normalized_key_value(right, field, casefold_fields=casefold_fields)
        if left_value is None or right_value is None:
            return False
        if field in rules.probable_numeric_tolerances:
            if isinstance(left_value, bool) or isinstance(right_value, bool):
                return False
            if not isinstance(left_value, (int, float)) or not isinstance(
                right_value,
                (int, float),
            ):
                return False
            if abs(float(left_value) - float(right_value)) > float(
                rules.probable_numeric_tolerances[field]
            ):
                return False
        elif left_value != right_value:
            return False
    return True


def _connected_components(
    records: list[dict[str, Any]],
    rules: DuplicateRuleSet,
) -> list[list[dict[str, Any]]]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(records):
        for right_index in range(left_index + 1, len(records)):
            if _probable_match(left, records[right_index], rules):
                union(left_index, right_index)
    components: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        components.setdefault(find(index), []).append(record)
    return [component for component in components.values() if len(component) > 1]


def _validated_adjudications(
    adjudications: Iterable[DuplicateAdjudication],
    *,
    rules_version: str,
    designated_reviewers: Iterable[str],
) -> dict[str, DuplicateAdjudication]:
    reviewer_set = {
        str(reviewer).strip()
        for reviewer in designated_reviewers
        if str(reviewer).strip()
    }
    indexed: dict[str, DuplicateAdjudication] = {}
    for adjudication in adjudications:
        if adjudication.duplicate_group_uid in indexed:
            raise ValueError("Duplicate group has more than one adjudication")
        if adjudication.rules_version != rules_version:
            raise ValueError("Duplicate adjudication references a different rules version")
        if adjudication.disposition not in {"same_trial", "distinct_trials"}:
            raise ValueError("Duplicate adjudication disposition is invalid")
        if adjudication.reviewer not in reviewer_set:
            raise ValueError("Duplicate adjudication reviewer is not designated")
        _review_date(adjudication.reviewed_on, label="Duplicate adjudication date")
        _nonempty(adjudication.rationale, label="Duplicate adjudication rationale")
        if (
            adjudication.disposition == "same_trial"
            and not adjudication.canonical_record_uid
        ):
            raise ValueError("Same-trial duplicate adjudication requires a canonical record")
        if (
            adjudication.disposition == "distinct_trials"
            and adjudication.canonical_record_uid is not None
        ):
            raise ValueError("Distinct-trial adjudication cannot select a canonical record")
        indexed[adjudication.duplicate_group_uid] = adjudication
    return indexed


def _initialize_duplicate_statuses(
    records: list[dict[str, Any]],
    *,
    rules: DuplicateRuleSet | None,
    adjudications: Iterable[DuplicateAdjudication],
    designated_reviewers: Iterable[str],
) -> None:
    for record in records:
        record["duplicate_relationships"] = ()
        record["duplicate_groups"] = ()
        record["duplicate_status"] = "not_assessed" if rules is None else "unique"
        record["duplicate_of_record_uid"] = None
        record["duplicate_rules_version"] = rules.version if rules else None

    if rules is None:
        return
    _validate_duplicate_rules(rules)
    reviewed = _validated_adjudications(
        adjudications,
        rules_version=rules.version,
        designated_reviewers=designated_reviewers,
    )

    exact_groups: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for record in records:
        signature = _exact_signature(record, rules)
        if signature is not None:
            exact_groups.setdefault(signature, []).append(record)
    for signature, duplicates in exact_groups.items():
        ordered = sorted(duplicates, key=_record_sort_key)
        if len(ordered) == 1:
            continue
        canonical = ordered[0]
        group_uid = _stable_identifier(
            "duplicate",
            ("exact", rules.version, *signature),
        )
        _add_duplicate_relationship(canonical, "exact_duplicate_canonical")
        _add_duplicate_group(
            canonical,
            duplicate_group_uid=group_uid,
            relationship="exact_duplicate_canonical",
            confidence="exact",
            evidence_codes=tuple(f"EXACT_KEY:{field}" for field in rules.exact_key_fields),
            review_status="auto_classified",
            canonical_record_uid=str(canonical["record_uid"]),
            rules_version=rules.version,
        )
        for duplicate in ordered[1:]:
            _add_duplicate_relationship(duplicate, "exact_duplicate_noncanonical")
            duplicate["duplicate_of_record_uid"] = canonical["record_uid"]
            duplicate["analytical_record_status"] = "duplicate_noncanonical"
            _add_duplicate_group(
                duplicate,
                duplicate_group_uid=group_uid,
                relationship="exact_duplicate_noncanonical",
                confidence="exact",
                evidence_codes=tuple(
                    f"EXACT_KEY:{field}" for field in rules.exact_key_fields
                ),
                review_status="auto_classified",
                canonical_record_uid=str(canonical["record_uid"]),
                rules_version=rules.version,
            )

    for candidates in _connected_components(records, rules):
        ordered_uids = tuple(
            sorted(str(record["record_uid"]) for record in candidates)
        )
        group_uid = _stable_identifier(
            "duplicate",
            ("probable", rules.version, *ordered_uids),
        )
        adjudication = reviewed.get(group_uid)
        canonical_uid: str | None = None
        review_status = "review_required"
        if adjudication is not None:
            if adjudication.canonical_record_uid not in set(ordered_uids) and (
                adjudication.canonical_record_uid is not None
            ):
                raise ValueError(
                    "Duplicate adjudication canonical record is outside its candidate group"
                )
            canonical_uid = adjudication.canonical_record_uid
            review_status = "adjudicated"
        for record in candidates:
            record_uid = str(record["record_uid"])
            if adjudication is None or adjudication.disposition == "distinct_trials":
                relationship = "probable_duplicate_review"
            elif record_uid == canonical_uid:
                relationship = "probable_duplicate_canonical"
            else:
                relationship = "probable_duplicate_noncanonical"
                record["duplicate_of_record_uid"] = canonical_uid
                record["analytical_record_status"] = "duplicate_noncanonical"
            _add_duplicate_relationship(record, relationship)
            _add_duplicate_group(
                record,
                duplicate_group_uid=group_uid,
                relationship=relationship,
                confidence="probable",
                evidence_codes=tuple(
                    (
                        f"PROBABLE_KEY:{field}:TOLERANCE="
                        f"{rules.probable_numeric_tolerances[field]}"
                    )
                    if field in rules.probable_numeric_tolerances
                    else f"PROBABLE_KEY:{field}"
                    for field in rules.probable_key_fields
                ),
                review_status=review_status,
                canonical_record_uid=canonical_uid,
                rules_version=rules.version,
                adjudication=adjudication,
            )

    unknown_adjudications = set(reviewed) - {
        str(group["duplicate_group_uid"])
        for record in records
        for group in record["duplicate_groups"]
    }
    if unknown_adjudications:
        raise ValueError(
            "Duplicate adjudication references unknown candidate group(s): "
            + ", ".join(sorted(unknown_adjudications))
        )


def _validated_repeat_adjudications(
    adjudications: Iterable[RepeatAdjudication],
    *,
    designated_reviewers: Iterable[str],
) -> dict[tuple[str, ...], RepeatAdjudication]:
    reviewer_set = {
        str(reviewer).strip()
        for reviewer in designated_reviewers
        if str(reviewer).strip()
    }
    indexed: dict[tuple[str, ...], RepeatAdjudication] = {}
    for adjudication in adjudications:
        record_uids = tuple(sorted(adjudication.record_uids))
        if len(record_uids) < 2 or len(record_uids) != len(set(record_uids)):
            raise ValueError("Repeat adjudication must identify at least two unique records")
        if record_uids in indexed:
            raise ValueError("Repeated-observation group has more than one adjudication")
        if adjudication.classification not in _REPEAT_CLASSIFICATIONS:
            raise ValueError("Repeat adjudication classification is invalid")
        if adjudication.reviewer not in reviewer_set:
            raise ValueError("Repeat adjudication reviewer is not designated")
        _review_date(adjudication.reviewed_on, label="Repeat adjudication date")
        _nonempty(adjudication.rationale, label="Repeat adjudication rationale")
        _nonempty(adjudication.review_id, label="Repeat adjudication review evidence")
        indexed[record_uids] = adjudication
    return indexed


def _repeat_aggregate(
    records: list[dict[str, Any]],
    *,
    series_uid: str,
    comparison_set_uid: str,
    adjudication: RepeatAdjudication,
) -> dict[str, Any]:
    yields = [record.get("yield_t_ha") for record in records]
    if any(
        record.get("yield_parse_status") != "parsed"
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        for record, value in zip(records, yields, strict=True)
    ):
        raise ValueError("Exchangeable replicate aggregation requires complete parsed yields")
    numeric_yields = [float(value) for value in yields]
    ordered_uids = tuple(sorted(str(record["record_uid"]) for record in records))
    aggregate = {
        key: value
        for key, value in records[0].items()
        if key
        not in {
            "raw_cells",
            "raw_headers",
            "raw_column_ids",
            "source_row_number",
            "source_physical_line_start",
            "source_physical_line_end",
        }
    }
    aggregate_uid = _stable_identifier(
        "aggregate",
        ("exchangeable-replicates", series_uid, *ordered_uids),
    )
    aggregate.update(
        {
            "record_uid": aggregate_uid,
            "record_kind": "exchangeable_replicate_aggregate",
            "source_record_uids": ordered_uids,
            "replicate_count": len(records),
            "yield_t_ha": statistics.fmean(numeric_yields),
            "yield_sd_t_ha": (
                statistics.stdev(numeric_yields) if len(numeric_yields) > 1 else None
            ),
            "yield_se_t_ha": (
                statistics.stdev(numeric_yields) / math.sqrt(len(numeric_yields))
                if len(numeric_yields) > 1
                else None
            ),
            "yield_parse_status": "parsed",
            "response_series_uid": series_uid,
            "comparison_set_uid": comparison_set_uid,
            "series_status": "resolved",
            "same_n_status": "exchangeable_replicate_aggregate",
            "analytical_record_status": "included",
            "analysis_grain_status": "reviewed_treatment_mean",
            "repeat_review_id": adjudication.review_id,
            "repeat_reviewer": adjudication.reviewer,
            "repeat_reviewed_on": adjudication.reviewed_on,
            "yield_se_status": "verified",
            "replication_status": "verified",
            "experimental_unit_status": "verified",
            "uncertainty_evidence_review_id": adjudication.review_id,
        }
    )
    return aggregate


def resolve_response_series(
    records: Iterable[Mapping[str, Any]],
    *,
    context_dimensions: tuple[str, ...] | list[str] | None = None,
    series_identity_dimensions: tuple[str, ...] | list[str] | None = None,
    n_level_tolerance_kg_ha: float = 1e-8,
    duplicate_rules: DuplicateRuleSet | None = None,
    duplicate_adjudications: Iterable[DuplicateAdjudication] = (),
    repeat_adjudications: Iterable[RepeatAdjudication] = (),
    designated_reviewers: Iterable[str] = (),
) -> SeriesResolution:
    """Resolve only reviewed contexts, duplicates, and repeated observations."""

    ledger = [dict(record) for record in records]
    record_uids = [str(record.get("record_uid", "")) for record in ledger]
    if not all(record_uids):
        raise ValueError("Every canonical record needs a nonempty record_uid")
    if len(record_uids) != len(set(record_uids)):
        raise ValueError("Canonical record identifiers must be unique")
    dimensions = tuple(
        series_identity_dimensions
        if series_identity_dimensions is not None
        else (context_dimensions or ())
    )
    if not dimensions:
        raise ValueError("Series identity dimensions must be explicitly nonempty")
    if len(dimensions) != len(set(dimensions)):
        raise ValueError("Series identity dimensions must be unique")
    if n_level_tolerance_kg_ha <= 0:
        raise ValueError("N-level tolerance must be positive")

    for record in ledger:
        record["response_series_uid"] = None
        record["comparison_set_uid"] = record.get("comparison_set_uid")
        record["series_status"] = "review"
        record["series_reason_codes"] = ()
        record["same_n_status"] = "not_assessed"
        record["repeat_group_uid"] = None
        record["analytical_record_status"] = "included"
    _initialize_duplicate_statuses(
        ledger,
        rules=duplicate_rules,
        adjudications=duplicate_adjudications,
        designated_reviewers=designated_reviewers,
    )
    reviewed_repeats = _validated_repeat_adjudications(
        repeat_adjudications,
        designated_reviewers=designated_reviewers,
    )

    candidate_groups: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    comparison_keys: dict[tuple[object, ...], tuple[object, ...]] = {}
    for record in ledger:
        if duplicate_rules is None:
            _mark_unresolved(record, "DUPLICATE_RULES_NOT_SUPPLIED")
            continue
        duplicate_relationships = set(record["duplicate_relationships"])
        if {
            "exact_duplicate_noncanonical",
            "probable_duplicate_noncanonical",
        }.intersection(duplicate_relationships):
            _mark_unresolved(record, "DUPLICATE_NONCANONICAL")
            record["analytical_record_status"] = "duplicate_noncanonical"
            continue
        if "probable_duplicate_review" in duplicate_relationships and any(
            group["review_status"] == "review_required"
            for group in record["duplicate_groups"]
            if group["relationship"] == "probable_duplicate_review"
        ):
            _mark_unresolved(record, "PROBABLE_DUPLICATE_REVIEW_REQUIRED")
            continue
        study_id = str(record.get("study_id", "")).strip()
        trial_id = str(record.get("trial_id", "")).strip()
        if not study_id or not trial_id:
            _mark_unresolved(record, "MISSING_STUDY_OR_TRIAL_IDENTIFIER")
            continue
        context_values: list[str] = []
        missing_context = False
        for dimension in dimensions:
            if _has_mixed_context(record, dimension):
                _mark_unresolved(record, f"MIXED_CONTEXT:{dimension}")
                missing_context = True
                continue
            value = _context_value(record, dimension)
            if value is None:
                _mark_unresolved(record, f"MISSING_CONTEXT:{dimension}")
                missing_context = True
            else:
                context_values.append(value)
        if missing_context:
            continue
        comparison_key = (
            str(record.get("source_uid", "")),
            study_id,
            trial_id,
            str(record.get("scope_country_code") or "unresolved"),
            *context_values,
        )
        key = (
            *comparison_key,
            *_reviewed_series_arm_discriminator(record),
        )
        comparison_keys[key] = comparison_key
        candidate_groups.setdefault(key, []).append(record)

    aggregates: list[dict[str, Any]] = []
    used_repeat_adjudications: set[tuple[str, ...]] = set()
    for key, group in sorted(
        candidate_groups.items(),
        key=lambda item: tuple(map(str, item[0])),
    ):
        n_groups: list[tuple[float, list[dict[str, Any]]]] = []
        parsed_n_rows: list[tuple[float, dict[str, Any]]] = []
        for record in group:
            n_rate = record.get("n_rate_kg_ha")
            if (
                record.get("n_rate_parse_status") == "parsed"
                and isinstance(n_rate, (int, float))
                and not isinstance(n_rate, bool)
            ):
                parsed_n_rows.append((float(n_rate), record))
        for n_rate, record in sorted(
            parsed_n_rows,
            key=lambda item: (item[0], _record_sort_key(item[1])),
        ):
            if not n_groups or abs(n_rate - n_groups[-1][0]) > n_level_tolerance_kg_ha:
                n_groups.append((n_rate, [record]))
            else:
                n_groups[-1][1].append(record)

        series_uid = _stable_identifier("series", key)
        reviewed_comparison_uids = {
            _reviewed_comparison_set_uid(record)
            for record in group
            if _reviewed_comparison_set_uid(record)
        }
        if len(reviewed_comparison_uids) > 1:
            for record in group:
                _mark_unresolved(record, "MIXED_COMPARISON_SET_UID")
            continue
        comparison_set_uid = (
            next(iter(reviewed_comparison_uids))
            if reviewed_comparison_uids
            else _stable_identifier("comparison-set", comparison_keys[key])
        )
        group_review_reasons: set[str] = set()
        approved_repeat_groups: list[
            tuple[list[dict[str, Any]], RepeatAdjudication]
        ] = []
        same_n_members = {
            str(record["record_uid"]): rows_at_level
            for _, rows_at_level in n_groups
            if len(rows_at_level) > 1
            for record in rows_at_level
        }
        for _, same_n_records in n_groups:
            if len(same_n_records) == 1:
                same_n_records[0]["same_n_status"] = "unique_n_level"
                continue
            ordered_uids = tuple(
                sorted(str(record["record_uid"]) for record in same_n_records)
            )
            repeat_group_uid = _stable_identifier(
                "repeat",
                (series_uid, *ordered_uids),
            )
            adjudication = reviewed_repeats.get(ordered_uids)
            for record in same_n_records:
                record["repeat_group_uid"] = repeat_group_uid
            same_parent = len(
                {
                    str(record.get("parent_row_uid") or record["record_uid"])
                    for record in same_n_records
                }
            ) < len(same_n_records)
            different_management = len(
                {_management_signature(record) for record in same_n_records}
            ) > 1
            if same_parent:
                status = "linked_arms_same_n"
                reason = "LINKED_ARMS_ARE_NOT_REPLICATES"
            elif adjudication is None:
                status = "repeat_review_required"
                reason = "REPEAT_CLASSIFICATION_REQUIRED"
            elif adjudication.classification == "exchangeable_replicates":
                if different_management:
                    raise ValueError(
                        "Exchangeable replicate adjudication spans different management"
                    )
                status = "confirmed_exchangeable_replicate"
                reason = ""
                approved_repeat_groups.append((same_n_records, adjudication))
                used_repeat_adjudications.add(ordered_uids)
            elif adjudication.classification == "management_variant":
                status = "different_management_same_n"
                reason = "MANAGEMENT_VARIANT_REQUIRES_SERIES_SPLIT"
                used_repeat_adjudications.add(ordered_uids)
            elif adjudication.classification == "duplicate":
                status = "repeat_classified_duplicate"
                reason = "REPEAT_DUPLICATE_REQUIRES_CANONICAL_LINK"
                used_repeat_adjudications.add(ordered_uids)
            else:
                status = "unequal_experimental_units"
                reason = "UNEQUAL_EXPERIMENTAL_UNITS_NOT_AGGREGATED"
                used_repeat_adjudications.add(ordered_uids)
            for record in same_n_records:
                record["same_n_status"] = status
                if adjudication is not None:
                    record["repeat_review_id"] = adjudication.review_id
                    record["repeat_reviewer"] = adjudication.reviewer
                    record["repeat_reviewed_on"] = adjudication.reviewed_on
            if reason:
                group_review_reasons.add(reason)

        if group_review_reasons:
            for record in group:
                for reason in sorted(group_review_reasons):
                    _mark_unresolved(record, reason)
            continue

        for record in group:
            record["response_series_uid"] = series_uid
            record["comparison_set_uid"] = comparison_set_uid
            record["series_status"] = "resolved"
            record["series_reason_codes"] = ()
            if str(record["record_uid"]) in same_n_members:
                record["analytical_record_status"] = "replaced_by_repeat_aggregate"
            elif record.get("analytical_record_status") == "included":
                record["analytical_record_status"] = "included"
        for repeat_records, adjudication in approved_repeat_groups:
            aggregates.append(
                _repeat_aggregate(
                    repeat_records,
                    series_uid=series_uid,
                    comparison_set_uid=comparison_set_uid,
                    adjudication=adjudication,
                )
            )

    unknown_repeat_adjudications = set(reviewed_repeats) - used_repeat_adjudications
    if unknown_repeat_adjudications:
        raise ValueError(
            "Repeat adjudication references records that do not form one same-N group"
        )
    aggregate_uids = [str(record["record_uid"]) for record in aggregates]
    if len(aggregate_uids) != len(set(aggregate_uids)):
        raise ValueError("Repeat aggregate identifiers must be unique")
    return SeriesResolution(
        records=tuple(ledger),
        aggregate_records=tuple(aggregates),
    )


__all__ = [
    "DuplicateAdjudication",
    "DuplicateRuleSet",
    "RepeatAdjudication",
    "SeriesResolution",
    "resolve_response_series",
]
