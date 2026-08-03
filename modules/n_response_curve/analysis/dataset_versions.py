from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from .values import finite_number


KNOWN_DATASET_VERSIONS = (
    "D00_inventory_all",
    "D01_strict_primary_zero_n",
    "D02_strict_primary_zero_optional",
    "D03_primary_4plus_n_levels",
    "D04_primary_5plus_n_levels",
    "D05_pk_varying_sensitivity",
    "D06_organic_bio_sensitivity",
    "D07_high_n_full_range",
    "D08_high_n_trimmed_sensitivity",
    "D09_complete_recommendation_set",
    "D10_factor_specific_complete_case",
    "D11_balanced_interaction_cells",
    "D12_climate_enriched_future",
)


@dataclass(frozen=True)
class DatasetMembershipDiagnostic:
    """One series-level membership decision for a filtered dataset view."""

    response_series_uid: str
    status: str
    required_classes: tuple[str, ...]
    verified_classes: tuple[str, ...]
    missing_classes: tuple[str, ...]
    ineligible_required_classes: tuple[str, ...]
    ineligible_optional_classes: tuple[str, ...]
    optional_classes_present: tuple[str, ...]
    reason_codes: tuple[str, ...]
    comparison_set_uid: str | None = None
    response_series_uids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetVersion:
    """A deterministic, non-mutating membership view over canonical record IDs."""

    version_id: str
    status: str
    record_uids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    membership_sha256: str
    authority_status: str = "authoritative"
    authority_reason_codes: tuple[str, ...] = ()
    membership_diagnostics: tuple[DatasetMembershipDiagnostic, ...] = ()


def _membership_hash(
    version_id: str,
    status: str,
    record_uids: Sequence[str],
    reason_codes: Sequence[str],
    *,
    authority_status: str,
    authority_reason_codes: Sequence[str],
    membership_diagnostics: Sequence[DatasetMembershipDiagnostic],
) -> str:
    payload = {
        "authority_reason_codes": list(authority_reason_codes),
        "authority_status": authority_status,
        "membership_diagnostics": [
            asdict(diagnostic) for diagnostic in membership_diagnostics
        ],
        "reason_codes": list(reason_codes),
        "record_uids": list(record_uids),
        "status": status,
        "version_id": version_id,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _version(version_id: str, record_uids: Iterable[str], *, status: str = "available", reason_codes: Iterable[str] = ()) -> DatasetVersion:
    sorted_uids = tuple(sorted(set(record_uids)))
    sorted_reasons = tuple(sorted(set(reason_codes)))
    if status == "available" and not sorted_uids:
        status = "unavailable"
        sorted_reasons = ("NO_RECORDS_MATCH_DATASET_VERSION",)
    return DatasetVersion(
        version_id=version_id,
        status=status,
        record_uids=sorted_uids,
        reason_codes=sorted_reasons,
        membership_sha256=_membership_hash(version_id, status, sorted_uids, sorted_reasons),
    )


def _record_uid(record: Mapping[str, Any]) -> str:
    value = record.get("record_uid")
    if not isinstance(value, str) or not value:
        raise ValueError("Each canonical record must have a nonempty record_uid")
    return value


def _eligible(record: Mapping[str, Any]) -> bool:
    tier = record.get("series_eligibility_tier") or record.get("eligibility_tier")
    return record.get("series_status") == "resolved" and tier in {"A", "B"}


def _strict_primary(record: Mapping[str, Any]) -> bool:
    tier = record.get("series_eligibility_tier") or record.get("eligibility_tier")
    return record.get("series_status") == "resolved" and tier == "A"


def _series_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    allowed_tiers: frozenset[str],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        series_uid = record.get("response_series_uid")
        if isinstance(series_uid, str) and series_uid:
            grouped.setdefault(series_uid, []).append(record)
    eligible: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for series_uid, rows in grouped.items():
        statuses = {str(row.get("series_status") or "") for row in rows}
        tiers = {
            str(row.get("series_eligibility_tier") or row.get("eligibility_tier") or "")
            for row in rows
        }
        if statuses == {"resolved"} and len(tiers) == 1 and next(iter(tiers)) in allowed_tiers:
            eligible[series_uid] = tuple(rows)
    return eligible


def _complete_n_levels(rows: Sequence[Mapping[str, Any]]) -> set[float]:
    return {
        n_rate
        for row in rows
        if (n_rate := finite_number(row.get("n_rate_kg_ha"))) is not None
        and finite_number(row.get("yield_t_ha")) is not None
    }


def _uids(groups: Iterable[Sequence[Mapping[str, Any]]]) -> tuple[str, ...]:
    return tuple(
        _record_uid(row)
        for rows in groups
        for row in rows
    )


def _reason_codes(record: Mapping[str, Any]) -> set[str]:
    raw = record.get("eligibility_reason_codes", ())
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return set()
    return {str(reason) for reason in raw}


def _series_complete_recommendation_set(records: Sequence[Mapping[str, Any]]) -> set[str]:
    required = {"zero_n_with_pk", "RCM", "FP", "NOPT_NPK"}
    observed_by_series: dict[str, set[str]] = {}
    for record in records:
        if not _eligible(record):
            continue
        series_uid = record.get("response_series_uid")
        if not isinstance(series_uid, str) or not series_uid:
            continue
        observed = observed_by_series.setdefault(series_uid, set())
        treatment_class = str(record.get("treatment_text_class", ""))
        if treatment_class in {"RCM", "FP", "NOPT_NPK"}:
            observed.add(treatment_class)
        if (
            record.get("nutrient_control_class") == "zero_n_with_pk"
            or record.get("is_zero_n_with_pk") is True
        ):
            observed.add("zero_n_with_pk")
    return {series_uid for series_uid, observed in observed_by_series.items() if required.issubset(observed)}


def _available_membership(version_id: str, records: Sequence[Mapping[str, Any]]) -> DatasetVersion:
    all_uids = [_record_uid(record) for record in records]
    primary = _series_groups(records, allowed_tiers=frozenset({"A"}))
    eligible = _series_groups(records, allowed_tiers=frozenset({"A", "B"}))
    if version_id == "D00_inventory_all":
        return _version(version_id, all_uids)
    if version_id == "D01_strict_primary_zero_n":
        return _version(
            version_id,
            _uids(
                rows
                for rows in primary.values()
                if any(abs(level) <= 1.0e-8 for level in _complete_n_levels(rows))
            ),
        )
    if version_id == "D02_strict_primary_zero_optional":
        return _version(version_id, _uids(primary.values()))
    if version_id == "D03_primary_4plus_n_levels":
        return _version(
            version_id,
            _uids(rows for rows in primary.values() if len(_complete_n_levels(rows)) >= 4),
        )
    if version_id == "D04_primary_5plus_n_levels":
        return _version(
            version_id,
            _uids(rows for rows in primary.values() if len(_complete_n_levels(rows)) >= 5),
        )
    if version_id == "D05_pk_varying_sensitivity":
        return _version(
            version_id,
            _uids(
                rows
                for rows in eligible.values()
                if any(_strict_primary(row) for row in rows)
                or any("P_K_VARY_WITH_N" in _reason_codes(row) for row in rows)
            ),
        )
    if version_id == "D06_organic_bio_sensitivity":
        return _version(
            version_id,
            _uids(
                rows
                for rows in eligible.values()
                if any(_strict_primary(row) for row in rows)
                or any(
                    bool(row.get("organic_fertilizer_present"))
                    or bool(row.get("biofertilizer_present"))
                    or "ORGANIC_OR_BIOFERTILIZER" in _reason_codes(row)
                    for row in rows
                )
            ),
        )
    if version_id == "D07_high_n_full_range":
        return _version(version_id, _uids(eligible.values()))
    if version_id == "D08_high_n_trimmed_sensitivity":
        trimmed_groups = []
        for rows in eligible.values():
            trimmed = tuple(row for row in rows if not bool(row.get("is_high_n")))
            if len(_complete_n_levels(trimmed)) >= 2:
                trimmed_groups.append(trimmed)
        return _version(
            version_id,
            _uids(trimmed_groups),
        )
    if version_id == "D09_complete_recommendation_set":
        complete_series = _series_complete_recommendation_set(records)
        return _version(
            version_id,
            (
                _record_uid(record)
                for rows in eligible.values()
                if str(rows[0].get("response_series_uid") or "") in complete_series
                for record in rows
            ),
        )
    if version_id == "D10_factor_specific_complete_case":
        return _version(version_id, (), status="unavailable", reason_codes=("FACTOR_CONTEXT_REQUIRED",))
    if version_id == "D11_balanced_interaction_cells":
        return _version(version_id, (), status="unavailable", reason_codes=("INTERACTION_CONTEXT_REQUIRED",))
    if version_id == "D12_climate_enriched_future":
        return _version(version_id, (), status="unavailable", reason_codes=("FUTURE_SOURCE_OR_COVARIATE_REQUIRED",))
    raise ValueError(f"Unknown dataset version: {version_id}")


def build_dataset_versions(
    records: Iterable[Mapping[str, Any]],
    *,
    version_names: Sequence[str],
) -> tuple[DatasetVersion, ...]:
    """Build enabled deterministic dataset views without copying or editing raw artifacts."""

    requested = tuple(version_names)
    unknown = set(requested) - set(KNOWN_DATASET_VERSIONS)
    if unknown:
        raise ValueError(f"Unknown dataset version(s): {', '.join(sorted(unknown))}")
    if len(requested) != len(set(requested)):
        raise ValueError("Dataset version names must be unique")
    rows = tuple(dict(record) for record in records)
    record_uids = [_record_uid(record) for record in rows]
    if len(record_uids) != len(set(record_uids)):
        raise ValueError("Canonical records may not have duplicate record_uids")
    return tuple(_available_membership(version_id, rows) for version_id in requested)


def select_dataset_version_records(
    records: Iterable[Mapping[str, Any]],
    dataset_version: DatasetVersion,
) -> tuple[dict[str, Any], ...]:
    """Materialize a membership view only in memory, ordered by stable record ID."""

    if dataset_version.status != "available":
        return ()
    by_uid = {_record_uid(record): dict(record) for record in records}
    if not set(dataset_version.record_uids).issubset(by_uid):
        raise ValueError("Dataset-version membership refers to unavailable canonical record IDs")
    return tuple(by_uid[record_uid] for record_uid in dataset_version.record_uids)


__all__ = [
    "KNOWN_DATASET_VERSIONS",
    "DatasetVersion",
    "build_dataset_versions",
    "select_dataset_version_records",
]
