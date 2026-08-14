from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
    "D13_untrimmed_final_cleaning_sensitivity",
)

DATASET_MEMBERSHIP_RULE_IDS = {
    version_id: f"dataset-membership:{version_id}:v1"
    for version_id in KNOWN_DATASET_VERSIONS
}
DATASET_MEMBERSHIP_RULE_IDS["D07_high_n_full_range"] = (
    "dataset-membership:D07_high_n_full_range:v2"
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
    membership_rule_id: str | None = None
    configuration_sha256: str | None = None
    input_dataset_sha256: str | None = None
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
    membership_rule_id: str | None = None,
    configuration_sha256: str | None = None,
    input_dataset_sha256: str | None = None,
) -> str:
    payload = {
        "authority_reason_codes": list(authority_reason_codes),
        "authority_status": authority_status,
        "configuration_sha256": configuration_sha256,
        "input_dataset_sha256": input_dataset_sha256,
        "membership_diagnostics": [
            asdict(diagnostic) for diagnostic in membership_diagnostics
        ],
        "membership_rule_id": membership_rule_id,
        "reason_codes": list(reason_codes),
        "record_uids": list(record_uids),
        "status": status,
        "version_id": version_id,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _version(
    version_id: str,
    record_uids: Iterable[str],
    *,
    status: str = "available",
    reason_codes: Iterable[str] = (),
    authority_status: str = "authoritative",
    authority_reason_codes: Iterable[str] = (),
    membership_diagnostics: Iterable[DatasetMembershipDiagnostic] = (),
) -> DatasetVersion:
    sorted_uids = tuple(sorted(set(record_uids)))
    sorted_reasons = tuple(sorted(set(reason_codes)))
    sorted_authority_reasons = tuple(sorted(set(authority_reason_codes)))
    sorted_diagnostics = tuple(
        sorted(membership_diagnostics, key=lambda item: item.response_series_uid)
    )
    if status == "available" and not sorted_uids:
        status = "unavailable"
        sorted_reasons = tuple(
            sorted(set(sorted_reasons) | {"NO_RECORDS_MATCH_DATASET_VERSION"})
        )
    return DatasetVersion(
        version_id=version_id,
        status=status,
        record_uids=sorted_uids,
        reason_codes=sorted_reasons,
        membership_sha256=_membership_hash(
            version_id,
            status,
            sorted_uids,
            sorted_reasons,
            authority_status=authority_status,
            authority_reason_codes=sorted_authority_reasons,
            membership_diagnostics=sorted_diagnostics,
        ),
        authority_status=authority_status,
        authority_reason_codes=sorted_authority_reasons,
        membership_diagnostics=sorted_diagnostics,
    )


def _record_uid(record: Mapping[str, Any]) -> str:
    value = record.get("record_uid")
    if not isinstance(value, str) or not value:
        raise ValueError("Each canonical record must have a nonempty record_uid")
    return value


def _eligible(record: Mapping[str, Any]) -> bool:
    tier = record.get("series_eligibility_tier") or record.get("eligibility_tier")
    return (
        record.get("series_status") == "resolved"
        and tier in {"A", "B"}
        and record.get("final_analytical_membership_status")
        not in {"excluded", "review_required"}
    )


def _strict_primary(record: Mapping[str, Any]) -> bool:
    tier = record.get("series_eligibility_tier") or record.get("eligibility_tier")
    return (
        record.get("series_status") == "resolved"
        and tier == "A"
        and record.get("final_analytical_membership_status")
        not in {"excluded", "review_required"}
    )


def _series_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    allowed_tiers: frozenset[str],
    membership_basis: str = "primary_cleaned",
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        if membership_basis == "primary_cleaned" and record.get(
            "final_analytical_membership_status"
        ) in {"excluded", "review_required"}:
            continue
        if membership_basis == "untrimmed" and record.get(
            "untrimmed_sensitivity_membership_status"
        ) != "included":
            continue
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


def _complete_n_levels(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = 0.0,
) -> tuple[float, ...]:
    values = sorted(
        n_rate
        for row in rows
        if (n_rate := finite_number(row.get("n_rate_kg_ha"))) is not None
        and finite_number(row.get("yield_t_ha")) is not None
    )
    levels: list[float] = []
    for value in values:
        if not levels or abs(value - levels[-1]) > tolerance:
            levels.append(value)
    return tuple(levels)


def _supports_response_curve(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_complete_n_yield: int,
    minimum_distinct_n_levels: int,
    n_level_tolerance_kg_ha: float,
) -> bool:
    complete_count = sum(
        1
        for row in rows
        if finite_number(row.get("n_rate_kg_ha")) is not None
        and finite_number(row.get("yield_t_ha")) is not None
    )
    return (
        complete_count >= minimum_complete_n_yield
        and len(
            _complete_n_levels(
                rows,
                tolerance=n_level_tolerance_kg_ha,
            )
        )
        >= minimum_distinct_n_levels
    )


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


def _recommendation_member_class(record: Mapping[str, Any]) -> str | None:
    treatment_class = str(record.get("treatment_text_class", ""))
    if treatment_class in {"RCM", "FP", "NOPT_NPK"}:
        return treatment_class
    n_rate = finite_number(record.get("n_rate_kg_ha"))
    if (
        record.get("nutrient_control_class") == "zero_n_with_pk"
        and record.get("is_zero_n_with_pk") is True
        and n_rate is not None
        and abs(n_rate) <= 1.0e-8
    ):
        return "zero_n_with_pk"
    return None


def _verified_recommendation_member(
    record: Mapping[str, Any],
    *,
    membership_status_field: str,
    verified_status: str,
) -> bool:
    review_id = record.get("recommendation_set_review_id")
    return (
        _eligible(record)
        and record.get("treatment_classification_status") == "resolved"
        and record.get(membership_status_field) == verified_status
        and isinstance(review_id, str)
        and bool(review_id.strip())
        and finite_number(record.get("n_rate_kg_ha")) is not None
        and finite_number(record.get("yield_t_ha")) is not None
    )


def _recommendation_set_membership(
    records: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[DatasetMembershipDiagnostic, ...]]:
    required = frozenset(str(value) for value in policy["required_classes"])
    optional = frozenset(str(value) for value in policy["optional_classes"])
    membership_status_field = str(policy["membership_status_field"])
    verified_status = str(policy["verified_status"])
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        series_uid = record.get("response_series_uid")
        if not isinstance(series_uid, str) or not series_uid:
            continue
        comparison_set_uid = record.get("comparison_set_uid")
        group_uid = (
            str(comparison_set_uid)
            if isinstance(comparison_set_uid, str) and comparison_set_uid
            else series_uid
        )
        grouped.setdefault(group_uid, []).append(record)
    included_uids: list[str] = []
    diagnostics: list[DatasetMembershipDiagnostic] = []
    for group_uid in sorted(grouped):
        rows = grouped[group_uid]
        response_series_uids = tuple(
            sorted(
                {
                    str(row.get("response_series_uid") or "")
                    for row in rows
                    if str(row.get("response_series_uid") or "")
                }
            )
        )
        comparison_set_uids = {
            str(row.get("comparison_set_uid") or "")
            for row in rows
            if str(row.get("comparison_set_uid") or "")
        }
        comparison_set_uid = (
            next(iter(comparison_set_uids))
            if len(comparison_set_uids) == 1
            else None
        )
        observed = {
            member_class
            for row in rows
            if (member_class := _recommendation_member_class(row)) is not None
        }
        verified = {
            member_class
            for row in rows
            if (member_class := _recommendation_member_class(row)) is not None
            and _verified_recommendation_member(
                row,
                membership_status_field=membership_status_field,
                verified_status=verified_status,
            )
        }
        missing = required - observed
        ineligible_required = (required & observed) - verified
        ineligible_optional = (optional & observed) - verified
        complete = not missing and not ineligible_required
        reasons: set[str] = set()
        reasons.update(f"MISSING_REQUIRED_CLASS:{value}" for value in missing)
        reasons.update(
            f"INELIGIBLE_REQUIRED_CLASS:{value}" for value in ineligible_required
        )
        reasons.update(
            f"INELIGIBLE_OPTIONAL_CLASS:{value}" for value in ineligible_optional
        )
        if complete:
            reasons.add("COMPLETE_VERIFIED_RECOMMENDATION_SET")
            included_uids.extend(
                _record_uid(row)
                for row in rows
                if (member_class := _recommendation_member_class(row)) is not None
                and member_class in required | optional
                and _verified_recommendation_member(
                    row,
                    membership_status_field=membership_status_field,
                    verified_status=verified_status,
                )
            )
        else:
            reasons.add("INCOMPLETE_RECOMMENDATION_SET_WITHHELD")
        diagnostics.append(
            DatasetMembershipDiagnostic(
                response_series_uid=(
                    response_series_uids[0]
                    if len(response_series_uids) == 1
                    else f"comparison:{group_uid}"
                ),
                status="included" if complete else "withheld",
                required_classes=tuple(sorted(required)),
                verified_classes=tuple(sorted(verified)),
                missing_classes=tuple(sorted(missing)),
                ineligible_required_classes=tuple(sorted(ineligible_required)),
                ineligible_optional_classes=tuple(sorted(ineligible_optional)),
                optional_classes_present=tuple(sorted(optional & observed)),
                reason_codes=tuple(sorted(reasons)),
                comparison_set_uid=comparison_set_uid,
                response_series_uids=response_series_uids,
            )
        )
    return tuple(included_uids), tuple(diagnostics)


def _available_membership(
    version_id: str,
    records: Sequence[Mapping[str, Any]],
    *,
    recommendation_set_policy: Mapping[str, Any] | None,
    minimum_complete_n_yield: int,
    minimum_distinct_n_levels: int,
    n_level_tolerance_kg_ha: float,
) -> DatasetVersion:
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
        return _version(
            version_id,
            _uids(
                rows
                for rows in eligible.values()
                if _supports_response_curve(
                    rows,
                    minimum_complete_n_yield=minimum_complete_n_yield,
                    minimum_distinct_n_levels=minimum_distinct_n_levels,
                    n_level_tolerance_kg_ha=n_level_tolerance_kg_ha,
                )
            ),
        )
    if version_id == "D08_high_n_trimmed_sensitivity":
        trimmed_groups = []
        for rows in eligible.values():
            trimmed = tuple(row for row in rows if not bool(row.get("is_high_n")))
            if len(_complete_n_levels(trimmed)) >= 2:
                trimmed_groups.append(trimmed)
        return _version(
            version_id,
            _uids(trimmed_groups),
            authority_status="non_authoritative",
            authority_reason_codes=(
                "REVIEWED_HIGH_N_TRIMMING_DIAGNOSTIC_UNAVAILABLE",
            ),
        )
    if version_id == "D09_complete_recommendation_set":
        if (
            not isinstance(recommendation_set_policy, Mapping)
            or recommendation_set_policy.get("review_status") != "approved"
            or not recommendation_set_policy.get("required_classes")
            or not isinstance(
                recommendation_set_policy.get("membership_status_field"),
                str,
            )
            or not isinstance(recommendation_set_policy.get("verified_status"), str)
        ):
            return _version(
                version_id,
                (),
                status="unavailable",
                reason_codes=("REVIEWED_RECOMMENDATION_SET_POLICY_REQUIRED",),
            )
        member_uids, diagnostics = _recommendation_set_membership(
            records,
            policy=recommendation_set_policy,
        )
        withheld = any(diagnostic.status == "withheld" for diagnostic in diagnostics)
        return _version(
            version_id,
            member_uids,
            reason_codes=(
                ("INCOMPLETE_RECOMMENDATION_SETS_WITHHELD",)
                if withheld
                else ()
            ),
            membership_diagnostics=diagnostics,
        )
    if version_id == "D10_factor_specific_complete_case":
        return _version(version_id, (), status="unavailable", reason_codes=("FACTOR_CONTEXT_REQUIRED",))
    if version_id == "D11_balanced_interaction_cells":
        return _version(version_id, (), status="unavailable", reason_codes=("INTERACTION_CONTEXT_REQUIRED",))
    if version_id == "D12_climate_enriched_future":
        return _version(version_id, (), status="unavailable", reason_codes=("FUTURE_SOURCE_OR_COVARIATE_REQUIRED",))
    if version_id == "D13_untrimmed_final_cleaning_sensitivity":
        untrimmed = _series_groups(
            records,
            allowed_tiers=frozenset({"A", "B"}),
            membership_basis="untrimmed",
        )
        return _version(
            version_id,
            _uids(untrimmed.values()),
            reason_codes=("FINAL_CLEANING_UNTRIMMED_SENSITIVITY",),
        )
    raise ValueError(f"Unknown dataset version: {version_id}")


def build_dataset_versions(
    records: Iterable[Mapping[str, Any]],
    *,
    version_names: Sequence[str],
    recommendation_set_policy: Mapping[str, Any] | None = None,
    minimum_complete_n_yield: int = 3,
    minimum_distinct_n_levels: int = 3,
    n_level_tolerance_kg_ha: float = 1.0e-6,
    configuration_sha256: str | None = None,
    input_dataset_sha256: str | None = None,
) -> tuple[DatasetVersion, ...]:
    """Build enabled deterministic dataset views without copying or editing raw artifacts."""

    requested = tuple(version_names)
    unknown = set(requested) - set(KNOWN_DATASET_VERSIONS)
    if unknown:
        raise ValueError(f"Unknown dataset version(s): {', '.join(sorted(unknown))}")
    if len(requested) != len(set(requested)):
        raise ValueError("Dataset version names must be unique")
    for label, value in (
        ("minimum_complete_n_yield", minimum_complete_n_yield),
        ("minimum_distinct_n_levels", minimum_distinct_n_levels),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Dataset-version {label} must be a positive integer")
    if (
        isinstance(n_level_tolerance_kg_ha, bool)
        or not isinstance(n_level_tolerance_kg_ha, (int, float))
        or float(n_level_tolerance_kg_ha) < 0.0
    ):
        raise ValueError(
            "Dataset-version n_level_tolerance_kg_ha must be a nonnegative number"
        )
    if (configuration_sha256 is None) != (input_dataset_sha256 is None):
        raise ValueError(
            "Dataset-version identity requires both configuration and input-dataset SHA-256 values"
        )
    for label, value in (
        ("configuration", configuration_sha256),
        ("input dataset", input_dataset_sha256),
    ):
        if value is not None and (
            len(value) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value.lower()
            )
        ):
            raise ValueError(f"Dataset-version {label} SHA-256 is malformed")
    rows = tuple(dict(record) for record in records)
    record_uids = [_record_uid(record) for record in rows]
    if len(record_uids) != len(set(record_uids)):
        raise ValueError("Canonical records may not have duplicate record_uids")
    versions = tuple(
        _available_membership(
            version_id,
            rows,
            recommendation_set_policy=recommendation_set_policy,
            minimum_complete_n_yield=minimum_complete_n_yield,
            minimum_distinct_n_levels=minimum_distinct_n_levels,
            n_level_tolerance_kg_ha=float(n_level_tolerance_kg_ha),
        )
        for version_id in requested
    )
    return tuple(
        replace(
            version,
            membership_sha256=_membership_hash(
                version.version_id,
                version.status,
                version.record_uids,
                version.reason_codes,
                authority_status=version.authority_status,
                authority_reason_codes=version.authority_reason_codes,
                membership_diagnostics=version.membership_diagnostics,
                membership_rule_id=DATASET_MEMBERSHIP_RULE_IDS[version.version_id],
                configuration_sha256=configuration_sha256,
                input_dataset_sha256=input_dataset_sha256,
            ),
            membership_rule_id=DATASET_MEMBERSHIP_RULE_IDS[version.version_id],
            configuration_sha256=configuration_sha256,
            input_dataset_sha256=input_dataset_sha256,
        )
        for version in versions
    )


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
    "DATASET_MEMBERSHIP_RULE_IDS",
    "DatasetMembershipDiagnostic",
    "DatasetVersion",
    "build_dataset_versions",
    "select_dataset_version_records",
]
