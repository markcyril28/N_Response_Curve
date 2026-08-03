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
    "probable_cross_source_duplicate": 1,
    "exact_duplicate_canonical": 2,
    "unique": 3,
}


@dataclass(frozen=True)
class SeriesResolution:
    """One reviewable series/duplicate ledger row for every canonical source record."""

    records: tuple[dict[str, Any], ...]


def _stable_identifier(prefix: str, parts: Iterable[object]) -> str:
    encoded = json.dumps(list(parts), ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:24]}"


def _record_sort_key(record: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(record.get("source_uid", "")),
        int(record.get("source_row_number", 0) or 0),
        str(record["record_uid"]),
    )


def _raw_context_value(record: Mapping[str, Any], dimension: str) -> str | None:
    key = _CONTEXT_FIELD_ALIASES.get(dimension, dimension)
    value = record.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _context_value(record: Mapping[str, Any], dimension: str) -> str | None:
    normalized = _raw_context_value(record, dimension)
    if normalized is None:
        return None
    if normalized.casefold() in _MISSING_CONTEXT_VALUES:
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
            "treatment",
            "p_rate_kg_p2o5_ha",
            "k_rate_kg_k2o_ha",
            "organic_fertilizer_present",
            "biofertilizer_present",
        )
    )


def _probable_cross_source_signature(record: Mapping[str, Any]) -> tuple[str, ...] | None:
    study_id = str(record.get("study_id", "")).strip()
    trial_id = str(record.get("trial_id", "")).strip()
    n_rate = record.get("n_rate_kg_ha")
    if not study_id or not trial_id or n_rate is None:
        return None
    return (
        study_id,
        trial_id,
        str(n_rate),
        str(record.get("treatment_id", "")).strip(),
        str(record.get("treatment", "")).strip(),
    )


def _add_sorted_unique(record: dict[str, Any], field: str, value: str) -> tuple[str, ...]:
    existing = record.get(field, ())
    values = {
        str(item)
        for item in existing
    } if isinstance(existing, (list, tuple, set, frozenset)) else set()
    values.add(value)
    normalized = tuple(sorted(values))
    record[field] = normalized
    return normalized


def _append_reason(record: dict[str, Any], reason: str) -> None:
    _add_sorted_unique(record, "series_reason_codes", reason)


def _mark_unresolved(record: dict[str, Any], reason: str) -> None:
    record["response_series_uid"] = None
    record["series_status"] = "review"
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
) -> None:
    groups = [dict(group) for group in record.get("duplicate_groups", ())]
    groups.append(
        {
            "duplicate_group_uid": duplicate_group_uid,
            "relationship": relationship,
            "confidence": confidence,
            "evidence_codes": tuple(evidence_codes),
            "review_status": review_status,
            "canonical_record_uid": canonical_record_uid,
        }
    )
    record["duplicate_groups"] = tuple(
        sorted(
            groups,
            key=lambda group: (
                str(group["duplicate_group_uid"]),
                str(group["relationship"]),
            ),
        )
    )


def _initialize_duplicate_statuses(records: list[dict[str, Any]]) -> None:
    for record in records:
        record["duplicate_relationships"] = ()
        record["duplicate_groups"] = ()
        record["duplicate_status"] = "unique"
        record["duplicate_of_record_uid"] = None

    by_exact_signature: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = {}
    for record in records:
        raw_cells = tuple(str(value) for value in record.get("raw_cells", ()))
        signature = (str(record.get("source_uid", "")), raw_cells)
        by_exact_signature.setdefault(signature, []).append(record)
    for duplicates in by_exact_signature.values():
        ordered = sorted(duplicates, key=_record_sort_key)
        if len(ordered) == 1:
            continue
        canonical = ordered[0]
        signature = (
            str(canonical.get("source_uid", "")),
            tuple(str(value) for value in canonical.get("raw_cells", ())),
        )
        group_uid = _stable_identifier("duplicate", ("exact_source_raw_cells", *signature))
        _add_duplicate_relationship(canonical, "exact_duplicate_canonical")
        _add_duplicate_group(
            canonical,
            duplicate_group_uid=group_uid,
            relationship="exact_duplicate_canonical",
            confidence="exact",
            evidence_codes=("SAME_SOURCE_UID", "IDENTICAL_RAW_CELLS"),
            review_status="auto_classified",
            canonical_record_uid=str(canonical["record_uid"]),
        )
        for duplicate in ordered[1:]:
            _add_duplicate_relationship(duplicate, "exact_duplicate_noncanonical")
            duplicate["duplicate_of_record_uid"] = canonical["record_uid"]
            _add_duplicate_group(
                duplicate,
                duplicate_group_uid=group_uid,
                relationship="exact_duplicate_noncanonical",
                confidence="exact",
                evidence_codes=("SAME_SOURCE_UID", "IDENTICAL_RAW_CELLS"),
                review_status="auto_classified",
                canonical_record_uid=str(canonical["record_uid"]),
            )

    by_probable_signature: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for record in records:
        signature = _probable_cross_source_signature(record)
        if signature is not None:
            by_probable_signature.setdefault(signature, []).append(record)
    for candidates in by_probable_signature.values():
        source_uids = {str(record.get("source_uid", "")) for record in candidates}
        if len(source_uids) < 2:
            continue
        signature = _probable_cross_source_signature(candidates[0])
        if signature is None:
            continue
        group_uid = _stable_identifier("duplicate", ("probable_cross_source", *signature))
        for record in candidates:
            _add_duplicate_relationship(record, "probable_cross_source_duplicate")
            _add_duplicate_group(
                record,
                duplicate_group_uid=group_uid,
                relationship="probable_cross_source_duplicate",
                confidence="probable",
                evidence_codes=(
                    "CROSS_SOURCE_MATCH",
                    "MATCHING_STUDY_TRIAL_N_TREATMENT_SIGNATURE",
                ),
                review_status="review_required",
                canonical_record_uid=None,
            )


def resolve_response_series(
    records: Iterable[Mapping[str, Any]],
    *,
    context_dimensions: tuple[str, ...] | list[str] | None = None,
    series_identity_dimensions: tuple[str, ...] | list[str] | None = None,
    n_level_tolerance_kg_ha: float = 1e-8,
) -> SeriesResolution:
    """Split only resolved contexts and route ambiguous/duplicate candidates to review."""

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
        record["series_status"] = "review"
        record["series_reason_codes"] = ()
        record["same_n_status"] = "not_assessed"
    _initialize_duplicate_statuses(ledger)

    candidate_groups: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for record in ledger:
        duplicate_relationships = set(record["duplicate_relationships"])
        if "exact_duplicate_noncanonical" in duplicate_relationships:
            _mark_unresolved(record, "EXACT_DUPLICATE_NONCANONICAL")
        if "probable_cross_source_duplicate" in duplicate_relationships:
            _mark_unresolved(record, "PROBABLE_CROSS_SOURCE_DUPLICATE")
        if {
            "exact_duplicate_noncanonical",
            "probable_cross_source_duplicate",
        }.intersection(duplicate_relationships):
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
        key = (
            str(record.get("source_uid", "")),
            study_id,
            trial_id,
            str(record.get("scope_country_code") or "unresolved"),
            *context_values,
        )
        candidate_groups.setdefault(key, []).append(record)

    for key, group in sorted(candidate_groups.items(), key=lambda item: tuple(map(str, item[0]))):
        n_groups: list[tuple[float, list[dict[str, Any]]]] = []
        parsed_n_rows: list[tuple[float, dict[str, Any]]] = []
        for record in group:
            n_rate = record.get("n_rate_kg_ha")
            if record.get("n_rate_parse_status") == "parsed" and isinstance(n_rate, (int, float)):
                parsed_n_rows.append((float(n_rate), record))
        for n_rate, record in sorted(parsed_n_rows, key=lambda item: (item[0], _record_sort_key(item[1]))):
            if not n_groups or abs(n_rate - n_groups[-1][0]) > n_level_tolerance_kg_ha:
                n_groups.append((n_rate, [record]))
            else:
                n_groups[-1][1].append(record)
        same_n_count = {
            str(record["record_uid"]): len(rows_at_level)
            for _, rows_at_level in n_groups
            for record in rows_at_level
        }
        has_management_conflict = any(
            len({ _management_signature(record) for record in same_n_records }) > 1
            for _, same_n_records in n_groups
            if len(same_n_records) > 1
        )
        if has_management_conflict:
            for record in group:
                record["same_n_status"] = "different_management_same_n"
                _mark_unresolved(record, "MANAGEMENT_CONFLICT_AT_N_LEVEL")
            continue

        series_uid = _stable_identifier("series", key)
        for record in group:
            count_at_level = same_n_count.get(str(record["record_uid"]), 0)
            record["same_n_status"] = "repeated_measurement" if count_at_level > 1 else "unique_n_level"
            record["response_series_uid"] = series_uid
            record["series_status"] = "resolved"
            record["series_reason_codes"] = ()

    return SeriesResolution(records=tuple(ledger))


__all__ = ["SeriesResolution", "resolve_response_series"]
