from __future__ import annotations

from dataclasses import dataclass
import hashlib
import statistics
from typing import Any, Iterable, Mapping, Sequence

from .factor_catalog import FactorCatalogEntry, factor_value
from .values import finite_number


@dataclass(frozen=True)
class DescriptiveComparison:
    """A non-inferential coverage and outcome summary for one explicit grouping."""

    comparison_type: str
    outcome_name: str
    group_label: str
    row_count: int
    outcome_count: int
    outcome_missing_count: int
    outcome_mean: float | None
    outcome_median: float | None
    outcome_min: float | None
    outcome_max: float | None
    reason_codes: tuple[str, ...]


MANAGEMENT_SYSTEM_CLASSES = ("RCM", "FP", "NOPT_NPK")


@dataclass(frozen=True)
class ManagementSystemProximity:
    management_proximity_uid: str
    estimand_version: str
    response_series_uid: str
    dataset_version_id: str | None
    dataset_version_status: str
    dataset_membership_sha256: str | None
    system_class: str
    status: str
    source_record_uid: str | None
    system_n_rate_kg_ha: float | None
    n_rate_unit: str
    yield_at_system_rate_t_ha: float | None
    yield_unit: str
    yield_basis: str
    model_reporting_policy: str
    credible_model_attempt_uids: tuple[str, ...]
    selected_model_attempt_uid: str | None
    supported_max_yield_t_ha: float | None
    maximum_reference_basis: str
    observed_bound_status: str
    maximum_gap_t_ha: float | None
    maximum_gap_status: str
    maximum_gap_direction: str
    target_yield_t_ha: float | None
    target_gap_t_ha: float | None
    target_gap_status: str
    target_gap_direction: str
    target_population: str
    same_context_status: str
    reason_codes: tuple[str, ...]


def _stable_uid(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _summary(
    comparison_type: str,
    outcome_name: str,
    group_label: str,
    rows: Sequence[Mapping[str, Any]],
) -> DescriptiveComparison:
    values = [finite_number(row.get(outcome_name)) for row in rows]
    observed = [value for value in values if value is not None]
    return DescriptiveComparison(
        comparison_type=comparison_type,
        outcome_name=outcome_name,
        group_label=group_label,
        row_count=len(rows),
        outcome_count=len(observed),
        outcome_missing_count=len(rows) - len(observed),
        outcome_mean=float(statistics.fmean(observed)) if observed else None,
        outcome_median=float(statistics.median(observed)) if observed else None,
        outcome_min=float(min(observed)) if observed else None,
        outcome_max=float(max(observed)) if observed else None,
        reason_codes=("CROSS_SOURCE_COMPARABILITY_NOT_ASSUMED", "DESCRIPTIVE_ONLY"),
    )


def _grouped_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    comparison_type: str,
    outcome_name: str,
    labels: Iterable[tuple[str, Mapping[str, Any]]],
) -> list[DescriptiveComparison]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for label, row in labels:
        groups.setdefault(label, []).append(row)
    return [
        _summary(comparison_type, outcome_name, label, group_rows)
        for label, group_rows in sorted(groups.items())
    ]


def build_descriptive_comparisons(
    curve_rows: Iterable[Mapping[str, Any]],
    *,
    outcome_name: str,
    factor_catalog: Sequence[FactorCatalogEntry],
) -> tuple[DescriptiveComparison, ...]:
    """Summarize coverage/missingness by source, recommendation, and explicit factor strata."""

    if not isinstance(outcome_name, str) or not outcome_name:
        raise ValueError("outcome_name must be a nonempty string")
    rows = tuple(dict(row) for row in curve_rows)
    summaries: list[DescriptiveComparison] = []
    summaries.extend(
        _grouped_summaries(
            rows,
            comparison_type="source_family",
            outcome_name=outcome_name,
            labels=((str(row.get("source_name") or "<missing>"), row) for row in rows),
        )
    )
    summaries.extend(
        _grouped_summaries(
            rows,
            comparison_type="recommendation_class",
            outcome_name=outcome_name,
            labels=((str(row.get("treatment_text_class") or "<missing>"), row) for row in rows),
        )
    )
    seen_factors: set[str] = set()
    for entry in factor_catalog:
        if entry.factor_name in seen_factors:
            raise ValueError("Factor catalog entries must have unique factor names")
        seen_factors.add(entry.factor_name)
        summaries.extend(
            _grouped_summaries(
                rows,
                comparison_type=f"factor_stratum:{entry.factor_name}",
                outcome_name=outcome_name,
                labels=(
                    (str(factor_value(row, entry.factor_name)) if factor_value(row, entry.factor_name) is not None else "<missing>", row)
                    for row in rows
                ),
            )
        )
    return tuple(sorted(summaries, key=lambda summary: (summary.comparison_type, summary.group_label)))


def build_management_system_proximity(
    records: Iterable[Mapping[str, Any]],
    *,
    curve_rows: Iterable[Mapping[str, Any]],
    dataset_version_id: str | None = None,
    dataset_version_status: str = "not_bound",
    dataset_membership_sha256: str | None = None,
    dataset_record_uids: Iterable[str] | None = None,
) -> tuple[ManagementSystemProximity, ...]:
    """Build ANA-15 Option A system-specific maximum and RCM target gaps.

    The response-series identifier is the same-context boundary. A system yield is
    accepted only when exactly one auditable observed system row is present, so the
    implementation never invents an averaging or pairing rule.
    """

    dataset_members = (
        None
        if dataset_record_uids is None
        else {str(record_uid) for record_uid in dataset_record_uids}
    )
    rows_by_series: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        record_uid = str(record.get("record_uid") or "")
        if dataset_members is not None and record_uid not in dataset_members:
            continue
        series_uid = record.get("response_series_uid")
        if record.get("series_status") != "resolved" or not isinstance(series_uid, str) or not series_uid:
            continue
        if record.get("scope_status") not in {None, "in_scope"}:
            continue
        if (record.get("series_eligibility_tier") or record.get("eligibility_tier")) == "D":
            continue
        rows_by_series.setdefault(series_uid, []).append(record)

    curve_by_series: dict[str, Mapping[str, Any]] = {}
    for row in curve_rows:
        series_uid = row.get("response_series_uid")
        if not isinstance(series_uid, str) or not series_uid:
            continue
        if series_uid in curve_by_series:
            raise ValueError(f"Management proximity requires one curve row per response series: {series_uid}")
        curve_by_series[series_uid] = row

    output: list[ManagementSystemProximity] = []
    for series_uid in sorted(rows_by_series):
        series_records = rows_by_series[series_uid]
        rcm_targets = sorted(
            {
                value
                for record in series_records
                if record.get("treatment_text_class") == "RCM"
                and (value := finite_number(record.get("target_yield_t_ha"))) is not None
            }
        )
        curve_row = curve_by_series.get(series_uid, {})
        supported_max = finite_number(curve_row.get("supported_max_yield_t_ha"))
        model_reporting_policy = str(
            curve_row.get("model_reporting_policy") or "unavailable"
        )
        raw_credible_uids = curve_row.get("credible_model_attempt_uids", ())
        if isinstance(raw_credible_uids, str):
            credible_model_uids = (raw_credible_uids,) if raw_credible_uids else ()
        elif isinstance(raw_credible_uids, (list, tuple, set, frozenset)):
            credible_model_uids = tuple(
                sorted(str(value) for value in raw_credible_uids if str(value))
            )
        else:
            credible_model_uids = ()
        selected_model_uid = curve_row.get("selected_model_attempt_uid")
        selected_model_uid = str(selected_model_uid) if selected_model_uid else None
        maximum_basis = str(curve_row.get("maximum_reference_basis") or "unavailable")
        observed_bound_status = (
            "inside_observed_n_domain"
            if supported_max is not None and maximum_basis not in {"", "none", "unavailable"}
            else "unavailable"
        )

        for system_class in MANAGEMENT_SYSTEM_CLASSES:
            reasons: set[str] = set()
            if dataset_version_id is None or dataset_version_status != "available":
                reasons.add("PRIMARY_DATASET_VERSION_NOT_BOUND")
            if (
                dataset_membership_sha256 is None
                or len(dataset_membership_sha256) != 64
                or any(character not in "0123456789abcdef" for character in dataset_membership_sha256)
            ):
                reasons.add("PRIMARY_DATASET_MEMBERSHIP_HASH_UNAVAILABLE")
            system_records = [
                record
                for record in series_records
                if record.get("treatment_text_class") == system_class
                and finite_number(record.get("n_rate_kg_ha")) is not None
                and finite_number(record.get("yield_t_ha")) is not None
            ]
            if not system_records:
                status = "unavailable"
                source_record_uid = None
                system_n_rate = None
                system_yield = None
                yield_basis = "unavailable"
                reasons.add("MANAGEMENT_SYSTEM_NOT_OBSERVED")
            elif len(system_records) > 1:
                status = "unavailable"
                source_record_uid = None
                system_n_rate = None
                system_yield = None
                yield_basis = "unavailable"
                reasons.add("MANAGEMENT_SYSTEM_OBSERVATION_AMBIGUOUS")
            else:
                status = "available"
                source_record_uid = str(system_records[0].get("record_uid") or "") or None
                system_n_rate = finite_number(system_records[0].get("n_rate_kg_ha"))
                system_yield = finite_number(system_records[0].get("yield_t_ha"))
                yield_basis = "observed_within_response_series"

            if system_yield is not None and supported_max is not None:
                maximum_gap = supported_max - system_yield
                maximum_gap_status = "available"
            else:
                maximum_gap = None
                maximum_gap_status = "unavailable"
                if supported_max is None:
                    reasons.add("SUPPORTED_MAXIMUM_UNAVAILABLE")

            if system_class != "RCM":
                target_yield = None
                target_gap = None
                target_gap_status = "not_applicable"
                target_gap_direction = "not_applicable_non_rcm_system"
            elif not rcm_targets:
                target_yield = None
                target_gap = None
                target_gap_status = "unavailable"
                target_gap_direction = "target_yield_minus_actual_rcm_yield"
                reasons.add("RCM_TARGET_YIELD_UNAVAILABLE")
            elif len(rcm_targets) > 1:
                target_yield = None
                target_gap = None
                target_gap_status = "unavailable"
                target_gap_direction = "target_yield_minus_actual_rcm_yield"
                reasons.add("RCM_TARGET_YIELD_AMBIGUOUS")
            else:
                target_yield = rcm_targets[0]
                target_gap = target_yield - system_yield if system_yield is not None else None
                target_gap_status = "available" if target_gap is not None else "unavailable"
                target_gap_direction = "target_yield_minus_actual_rcm_yield"

            output.append(
                ManagementSystemProximity(
                    management_proximity_uid=_stable_uid("management-proximity-v1", series_uid, system_class),
                    estimand_version="management-system-proximity-v1",
                    response_series_uid=series_uid,
                    dataset_version_id=dataset_version_id,
                    dataset_version_status=dataset_version_status,
                    dataset_membership_sha256=dataset_membership_sha256,
                    system_class=system_class,
                    status=status,
                    source_record_uid=source_record_uid,
                    system_n_rate_kg_ha=system_n_rate,
                    n_rate_unit="kg N/ha",
                    yield_at_system_rate_t_ha=system_yield,
                    yield_unit="t/ha",
                    yield_basis=yield_basis,
                    model_reporting_policy=model_reporting_policy,
                    credible_model_attempt_uids=credible_model_uids,
                    selected_model_attempt_uid=selected_model_uid,
                    supported_max_yield_t_ha=supported_max,
                    maximum_reference_basis=maximum_basis,
                    observed_bound_status=observed_bound_status,
                    maximum_gap_t_ha=maximum_gap,
                    maximum_gap_status=maximum_gap_status,
                    maximum_gap_direction="supported_maximum_minus_system_yield",
                    target_yield_t_ha=target_yield,
                    target_gap_t_ha=target_gap,
                    target_gap_status=target_gap_status,
                    target_gap_direction=target_gap_direction,
                    target_population="resolved_same_response_series",
                    same_context_status="verified_response_series",
                    reason_codes=tuple(sorted(reasons)),
                )
            )
    return tuple(output)


__all__ = [
    "DescriptiveComparison",
    "MANAGEMENT_SYSTEM_CLASSES",
    "ManagementSystemProximity",
    "build_descriptive_comparisons",
    "build_management_system_proximity",
]
