from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from .advanced_analysis import execute_advanced_candidate
from .analysis_matrix import AnalysisCandidate
from .comparisons import build_descriptive_comparisons
from .dataset_versions import DatasetVersion
from .factor_catalog import FactorCatalogEntry
from .values import outcome_is_present, record_uids


@dataclass(frozen=True)
class PythonAnalysisResult:
    """Terminal execution record for a Python-owned analysis candidate."""

    candidate_id: str
    engine: str
    status: str
    result_type: str | None
    reason_codes: tuple[str, ...]
    records: tuple[Mapping[str, Any], ...]


def select_candidate_curve_rows(
    candidate: AnalysisCandidate,
    curve_rows: Iterable[Mapping[str, Any]],
    versions: Mapping[str, DatasetVersion],
) -> tuple[dict[str, Any], ...]:
    version = versions.get(candidate.dataset_version_id)
    if version is None:
        raise ValueError(f"Candidate refers to an unknown dataset version: {candidate.dataset_version_id}")
    if version.status != "available":
        return ()
    selected: list[dict[str, Any]] = []
    for raw_row in curve_rows:
        row = dict(raw_row)
        if str(row.get("source_name", "")) not in candidate.source_families:
            continue
        tagged_version = row.get("dataset_version_id")
        if tagged_version is not None and (
            tagged_version != version.version_id
            or row.get("dataset_version_membership_sha256") != version.membership_sha256
        ):
            continue
        if tagged_version is None and not (
            (row_record_uids := record_uids(row)) and row_record_uids.issubset(version.record_uids)
        ):
            continue
        tagged_combination = row.get("source_combination_id")
        if tagged_combination is not None and tagged_combination != candidate.source_combination_id:
            continue
        selected.append(row)
    return tuple(sorted(selected, key=lambda row: str(row.get("response_series_uid", ""))))


def _coverage_record(
    rows: Sequence[Mapping[str, Any]],
    outcome_name: str,
    eligibility_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observed = [row for row in rows if outcome_is_present(row.get(outcome_name))]
    source_families = sorted({str(row.get("source_name", "")) for row in rows})
    selected_record_uids = set().union(*(record_uids(row) for row in rows)) if rows else set()
    eligibility_uids = {
        str(row.get("record_uid"))
        for row in eligibility_rows
        if row.get("record_uid")
    }
    tier_counts = Counter(str(row.get("eligibility_tier") or "unclassified") for row in eligibility_rows)
    return {
        "curve_row_count": len(rows),
        "outcome_name": outcome_name,
        "outcome_observed_count": len(observed),
        "outcome_missing_count": len(rows) - len(observed),
        "source_families": source_families,
        "eligible_record_count": len(eligibility_uids),
        "curve_selected_record_count": len(eligibility_uids.intersection(selected_record_uids)),
        "not_curve_selected_record_count": len(eligibility_uids - selected_record_uids),
        "eligibility_tier_counts": dict(sorted(tier_counts.items())),
        "reason_codes": ["DESCRIPTIVE_ONLY", "ELIGIBILITY_SELECTION_AUDIT"],
    }


def execute_python_candidates(
    candidates: Iterable[AnalysisCandidate],
    *,
    curve_rows: Iterable[Mapping[str, Any]],
    dataset_versions: Sequence[DatasetVersion],
    factor_catalog: Sequence[FactorCatalogEntry],
    eligibility_records: Iterable[Mapping[str, Any]] = (),
) -> tuple[PythonAnalysisResult, ...]:
    """Run Python-owned candidates; R-owned candidates remain undispatched."""

    versions = {version.version_id: version for version in dataset_versions}
    if len(versions) != len(dataset_versions):
        raise ValueError("Dataset versions must have unique IDs")
    factor_entries = {entry.factor_name: entry for entry in factor_catalog}
    if len(factor_entries) != len(factor_catalog):
        raise ValueError("Factor catalog entries must have unique factor names")
    rows = tuple(dict(row) for row in curve_rows)
    eligibility = tuple(dict(row) for row in eligibility_records)
    results: list[PythonAnalysisResult] = []
    for candidate in candidates:
        unknown_factors = set(candidate.factor_names) - set(factor_entries)
        if unknown_factors:
            raise ValueError(f"Candidate refers to unknown factor(s): {', '.join(sorted(unknown_factors))}")
        selected_rows = select_candidate_curve_rows(candidate, rows, versions)
        version_membership = set(versions[candidate.dataset_version_id].record_uids)
        selected_eligibility = tuple(
            row
            for row in eligibility
            if str(row.get("record_uid") or "") in version_membership
            and str(row.get("source_name") or "") in candidate.source_families
        )
        if candidate.status != "run":
            results.append(
                PythonAnalysisResult(
                    candidate_id=candidate.candidate_id,
                    engine=candidate.engine,
                    status="skipped",
                    result_type=None,
                    reason_codes=candidate.reason_codes,
                    records=(),
                )
            )
            continue
        if candidate.engine == "r":
            results.append(
                PythonAnalysisResult(
                    candidate_id=candidate.candidate_id,
                    engine="r",
                    status="not_dispatched",
                    result_type=None,
                    reason_codes=("OWNED_BY_R_ENGINE",),
                    records=(),
                )
            )
            continue
        if candidate.engine != "python":
            raise ValueError(f"Candidate has unsupported engine: {candidate.engine}")
        if candidate.analysis_family == "coverage_and_missingness":
            results.append(
                PythonAnalysisResult(
                    candidate_id=candidate.candidate_id,
                    engine="python",
                    status="completed",
                    result_type="coverage_and_missingness",
                    reason_codes=(),
                    records=(_coverage_record(selected_rows, candidate.curve_outcome, selected_eligibility),),
                )
            )
            continue
        if candidate.analysis_family == "one_factor_descriptive":
            selected_factors = tuple(factor_entries[name] for name in candidate.factor_names)
            comparisons = build_descriptive_comparisons(
                selected_rows,
                outcome_name=candidate.curve_outcome,
                factor_catalog=selected_factors,
                factor_representations=candidate.factor_representations,
            )
            results.append(
                PythonAnalysisResult(
                    candidate_id=candidate.candidate_id,
                    engine="python",
                    status="completed",
                    result_type="descriptive_comparisons",
                    reason_codes=(),
                    records=tuple(asdict(comparison) for comparison in comparisons),
                )
            )
            continue
        advanced = execute_advanced_candidate(candidate, selected_rows, factor_entries)
        if advanced is not None:
            results.append(
                PythonAnalysisResult(
                    candidate_id=candidate.candidate_id,
                    engine="python",
                    status=advanced.status,
                    result_type=advanced.result_type,
                    reason_codes=advanced.reason_codes,
                    records=advanced.records,
                )
            )
            continue
        results.append(
            PythonAnalysisResult(
                candidate_id=candidate.candidate_id,
                engine="python",
                status="skipped",
                result_type=None,
                reason_codes=("PYTHON_ANALYSIS_FAMILY_NOT_IMPLEMENTED",),
                records=(),
            )
        )
    return tuple(results)


__all__ = ["PythonAnalysisResult", "execute_python_candidates", "select_candidate_curve_rows"]
