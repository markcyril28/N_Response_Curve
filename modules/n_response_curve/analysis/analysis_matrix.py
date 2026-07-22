from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from typing import Any, Iterable, Mapping, Sequence

from ..data.provenance import stable_identifier
from .dataset_versions import DatasetVersion
from .factor_catalog import FactorCatalogEntry, factor_value
from .values import outcome_is_present, record_uids


KNOWN_SOURCE_COMBINATION_MODES = frozenset(
    {
        "each_family_alone",
        "all_nonempty_family_subsets",
        "all_families_deduplicated",
        "leave_one_family_out",
    }
)
_INTERACTION_ORDER_FAMILIES = frozenset(
    {
        "all_supported_interactions",
        "multivariable_mixed_effects",
        "observation_level_curve_modification",
        "penalized_predictive_models",
    }
)


@dataclass(frozen=True)
class SourceCombination:
    """One deterministic source-family membership set."""

    combination_id: str
    source_families: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisCandidate:
    """An engine-neutral analysis specification with a terminal pre-execution status."""

    candidate_id: str
    specification_hash: str
    dataset_version_id: str
    source_combination_id: str
    source_families: tuple[str, ...]
    curve_outcome: str
    analysis_family: str
    factor_names: tuple[str, ...]
    engine: str
    status: str
    reason_codes: tuple[str, ...]
    eligible_curve_rows: int
    independent_study_count: int
    factor_cell_counts: Mapping[str, int]


@dataclass(frozen=True)
class PrunedFamily:
    """Compressed ledger entry retaining a reason and exact number of pruned candidates."""

    reason_code: str
    candidate_count: int


@dataclass(frozen=True)
class AnalysisRegistry:
    """Complete concrete analysis space plus a compressed pruning ledger."""

    candidates: tuple[AnalysisCandidate, ...]
    pruned_families: tuple[PrunedFamily, ...]
    theoretical_candidate_count: int
    accounted_candidate_count: int
    reconciles: bool


def build_source_combinations(
    source_families: Sequence[str],
    *,
    modes: Sequence[str],
) -> tuple[SourceCombination, ...]:
    """Enumerate required source-family combinations once, without duplicate sets."""

    normalized_sources = tuple(sorted({str(source) for source in source_families if str(source)}))
    requested_modes = tuple(modes)
    unknown = set(requested_modes) - KNOWN_SOURCE_COMBINATION_MODES
    if unknown:
        raise ValueError(f"Unknown source-combination mode(s): {', '.join(sorted(unknown))}")
    if len(requested_modes) != len(set(requested_modes)):
        raise ValueError("Source-combination modes must be unique")
    combinations: set[tuple[str, ...]] = set()
    source_count = len(normalized_sources)
    for mode in requested_modes:
        if mode == "each_family_alone":
            combinations.update((source,) for source in normalized_sources)
        elif mode == "all_nonempty_family_subsets":
            for size in range(1, source_count + 1):
                combinations.update(itertools.combinations(normalized_sources, size))
        elif mode == "all_families_deduplicated" and normalized_sources:
            combinations.add(normalized_sources)
        elif mode == "leave_one_family_out" and source_count > 1:
            combinations.update(tuple(source for source in normalized_sources if source != omitted) for omitted in normalized_sources)
    return tuple(
        SourceCombination(
            combination_id=stable_identifier("sources", combination),
            source_families=combination,
        )
        for combination in sorted(combinations, key=lambda item: (len(item), item))
    )


def _factor_combinations(analysis_family: str, factors: Sequence[FactorCatalogEntry], interaction_orders: Sequence[int]) -> tuple[tuple[str, ...], ...]:
    names = tuple(entry.factor_name for entry in factors)
    if analysis_family in {
        "coverage_and_missingness",
        "curve_feature_clustering",
        "dataset_and_source_robustness",
    }:
        return ((),)
    if not names:
        return ((),)
    if analysis_family in _INTERACTION_ORDER_FAMILIES:
        orders = tuple(order for order in interaction_orders if 1 <= order <= len(names))
    else:
        orders = (1,)
    return tuple(
        factor_names
        for order in orders
        for factor_names in itertools.combinations(names, order)
    ) or ((),)


def _row_is_in_dataset_version(row: Mapping[str, Any], version: DatasetVersion) -> bool:
    tagged_version = row.get("dataset_version_id")
    if tagged_version is not None:
        return (
            tagged_version == version.version_id
            and row.get("dataset_version_membership_sha256") == version.membership_sha256
        )
    row_record_uids = record_uids(row)
    return bool(row_record_uids) and row_record_uids.issubset(version.record_uids)


def _selected_rows(
    curve_rows: Sequence[Mapping[str, Any]],
    version: DatasetVersion,
    source_combination: SourceCombination,
    outcome: str,
) -> list[Mapping[str, Any]]:
    if version.status != "available":
        return []
    rows: list[Mapping[str, Any]] = []
    for row in curve_rows:
        if str(row.get("source_name", "")) not in source_combination.source_families:
            continue
        tagged_combination = row.get("source_combination_id")
        if tagged_combination is not None and tagged_combination != source_combination.combination_id:
            continue
        if not _row_is_in_dataset_version(row, version):
            continue
        if not outcome_is_present(row.get(outcome)):
            continue
        rows.append(row)
    return rows


def _support_policy_values(policy: Mapping[str, Any] | None) -> dict[str, int] | None:
    if policy is None:
        return None
    required = (
        "minimum_independent_studies",
        "minimum_factor_cell_count",
        "maximum_factor_cardinality",
        "minimum_residual_information",
    )
    values: dict[str, int] = {}
    for key in required:
        value = policy.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Support policy {key!r} must be an integer >= 1")
        values[key] = value
    return values


def _support_summary(
    rows: Sequence[Mapping[str, Any]],
    factor_names: Sequence[str],
) -> tuple[int, dict[str, int], int, int]:
    complete_rows: list[Mapping[str, Any]] = []
    for row in rows:
        if all(factor_value(row, name) is not None for name in factor_names):
            complete_rows.append(row)
    studies = {str(row.get("study_uid") or "") for row in complete_rows}
    studies.discard("")
    missing_group_count = sum(not str(row.get("study_uid") or "") for row in complete_rows)
    cell_counts: dict[str, int] = {}
    if factor_names:
        for row in complete_rows:
            cell = "|".join(str(factor_value(row, name)) for name in factor_names)
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
    return (
        len(studies),
        {key: cell_counts[key] for key in sorted(cell_counts)},
        len(complete_rows),
        missing_group_count,
    )


def _reasons_for_candidate(
    *,
    version: DatasetVersion,
    rows: Sequence[Mapping[str, Any]],
    factor_entries: Sequence[FactorCatalogEntry],
    analysis_family: str,
    support_policy: Mapping[str, int] | None,
) -> tuple[str, tuple[str, ...], int, Mapping[str, int]]:
    if version.status != "available":
        return "skipped", tuple(version.reason_codes or ("DATASET_VERSION_UNAVAILABLE",)), 0, {}
    if version.version_id == "D00_inventory_all" and analysis_family != "coverage_and_missingness":
        return "skipped", ("DATASET_VERSION_NOT_PERMITTED_FOR_ANALYSIS_FAMILY",), 0, {}
    if not rows:
        return "pruned", ("NO_OUTCOME_SUPPORT",), 0, {}
    if analysis_family in {
        "coverage_and_missingness",
        "curve_feature_clustering",
        "dataset_and_source_robustness",
    }:
        return "run", (), len(rows), {}
    if not factor_entries:
        return "pruned", ("NO_CONFIGURED_FACTORS",), 0, {}
    if any(entry.leakage_restricted for entry in factor_entries):
        return "pruned", ("LEAKAGE_RESTRICTED_FACTOR",), 0, {}
    independent_studies, cell_counts, complete_rows, missing_group_count = _support_summary(
        rows,
        [entry.factor_name for entry in factor_entries],
    )
    if analysis_family == "one_factor_descriptive":
        return ("run", (), independent_studies, cell_counts) if complete_rows else ("pruned", ("NO_FACTOR_COMPLETE_CASES",), independent_studies, cell_counts)
    if support_policy is None:
        return "pruned", ("SUPPORT_POLICY_REQUIRED",), independent_studies, cell_counts
    reasons: list[str] = []
    if missing_group_count:
        reasons.append("MISSING_GROUP_IDENTITY")
    if independent_studies < support_policy["minimum_independent_studies"]:
        reasons.append("INSUFFICIENT_INDEPENDENT_STUDY_SUPPORT")
    if len(cell_counts) > support_policy["maximum_factor_cardinality"]:
        reasons.append("FACTOR_CARDINALITY_EXCEEDS_SUPPORT_POLICY")
    if cell_counts and min(cell_counts.values()) < support_policy["minimum_factor_cell_count"]:
        reasons.append("INSUFFICIENT_FACTOR_CELL_SUPPORT")
    residual_information = complete_rows - len(factor_entries) - 1
    if residual_information < support_policy["minimum_residual_information"]:
        reasons.append("INSUFFICIENT_RESIDUAL_INFORMATION")
    if reasons:
        return "pruned", tuple(sorted(reasons)), independent_studies, cell_counts
    return "run", (), independent_studies, cell_counts


def _candidate(
    *,
    version: DatasetVersion,
    combination: SourceCombination,
    outcome: str,
    analysis_family: str,
    factor_entries: Sequence[FactorCatalogEntry],
    engine: str,
    rows: Sequence[Mapping[str, Any]],
    support_policy: Mapping[str, int] | None,
) -> AnalysisCandidate:
    factor_names = tuple(entry.factor_name for entry in factor_entries)
    status, reasons, independent_studies, cell_counts = _reasons_for_candidate(
        version=version,
        rows=rows,
        factor_entries=factor_entries,
        analysis_family=analysis_family,
        support_policy=support_policy,
    )
    specification = {
        "analysis_family": analysis_family,
        "curve_outcome": outcome,
        "dataset_version_id": version.version_id,
        "engine": engine,
        "factor_names": factor_names,
        "source_families": combination.source_families,
        "version_membership_sha256": version.membership_sha256,
    }
    raw_hash = hashlib.sha256(
        json.dumps(specification, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    specification_hash = f"{engine}_{raw_hash}"
    candidate_id = stable_identifier("analysis", specification)
    return AnalysisCandidate(
        candidate_id=candidate_id,
        specification_hash=specification_hash,
        dataset_version_id=version.version_id,
        source_combination_id=combination.combination_id,
        source_families=combination.source_families,
        curve_outcome=outcome,
        analysis_family=analysis_family,
        factor_names=factor_names,
        engine=engine,
        status=status,
        reason_codes=reasons,
        eligible_curve_rows=len(rows),
        independent_study_count=independent_studies,
        factor_cell_counts=cell_counts,
    )


def build_analysis_registry(
    curve_rows: Iterable[Mapping[str, Any]],
    *,
    dataset_versions: Sequence[DatasetVersion],
    source_combination_modes: Sequence[str],
    curve_outcomes: Sequence[str],
    factor_catalog: Sequence[FactorCatalogEntry],
    analysis_families: Sequence[str],
    engine_assignments: Mapping[str, str],
    interaction_orders: Sequence[int],
    support_policy: Mapping[str, Any] | None,
    source_families: Sequence[str] | None = None,
) -> AnalysisRegistry:
    """Enumerate a complete, engine-owned analysis space before dispatching any model."""

    rows = tuple(dict(row) for row in curve_rows)
    versions = tuple(dataset_versions)
    version_ids = [version.version_id for version in versions]
    if len(version_ids) != len(set(version_ids)):
        raise ValueError("Dataset versions must have unique IDs")
    family_names = tuple(analysis_families)
    if len(family_names) != len(set(family_names)):
        raise ValueError("Analysis families must be unique")
    for family in family_names:
        engine = engine_assignments.get(family)
        if engine is None:
            raise ValueError(f"Analysis family has no primary engine assignment: {family}")
        if engine not in {"python", "r"}:
            raise ValueError(f"Analysis family has unknown engine assignment: {family}={engine}")
    outcomes = tuple(curve_outcomes)
    if not outcomes or any(not isinstance(outcome, str) or not outcome for outcome in outcomes):
        raise ValueError("Curve outcomes must be a nonempty sequence of names")
    if len(outcomes) != len(set(outcomes)):
        raise ValueError("Curve outcomes must be unique")
    known_factors = {entry.factor_name: entry for entry in factor_catalog}
    if len(known_factors) != len(factor_catalog):
        raise ValueError("Factor catalog entries must have unique factor names")
    orders = tuple(interaction_orders)
    if len(orders) != len(set(orders)) or any(not isinstance(order, int) or order < 1 or order > 3 for order in orders):
        raise ValueError("Interaction orders must be unique integers from one through three")
    policy = _support_policy_values(support_policy)
    discovered_source_families = tuple(
        sorted({str(row.get("source_name", "")) for row in rows if str(row.get("source_name", ""))})
    )
    configured_source_families = tuple(source_families) if source_families is not None else discovered_source_families
    if len(configured_source_families) != len(set(configured_source_families)) or any(
        not isinstance(family, str) or not family for family in configured_source_families
    ):
        raise ValueError("Configured source families must be unique nonempty names")
    combinations = build_source_combinations(configured_source_families, modes=source_combination_modes)

    candidates: list[AnalysisCandidate] = []
    theoretical_count = 0
    compressed_prune_count = 0
    expansion_families = {
        "all_supported_interactions",
        "multivariable_mixed_effects",
        "observation_level_curve_modification",
        "penalized_predictive_models",
    }
    for version in versions:
        for combination in combinations:
            for outcome in outcomes:
                applicable_rows = _selected_rows(rows, version, combination, outcome)
                for family in family_names:
                    factor_sets = _factor_combinations(family, tuple(known_factors.values()), orders)
                    theoretical_count += len(factor_sets)
                    if family not in expansion_families or not any(len(names) > 1 for names in factor_sets):
                        materialized_factor_sets = factor_sets
                    else:
                        single_factor_sets = tuple(names for names in factor_sets if len(names) == 1)
                        single_candidates: list[AnalysisCandidate] = []
                        supported_factors: set[str] = set()
                        for factor_names in single_factor_sets:
                            entries = tuple(known_factors[name] for name in factor_names)
                            concrete = _candidate(
                                version=version,
                                combination=combination,
                                outcome=outcome,
                                analysis_family=family,
                                factor_entries=entries,
                                engine=engine_assignments[family],
                                rows=applicable_rows,
                                support_policy=policy,
                            )
                            single_candidates.append(concrete)
                            if concrete.status == "run":
                                supported_factors.update(factor_names)
                        candidates.extend(single_candidates)
                        materialized_factor_sets = tuple(
                            names
                            for names in factor_sets
                            if len(names) > 1 and set(names).issubset(supported_factors)
                        )
                        compressed_prune_count += len(factor_sets) - len(single_factor_sets) - len(materialized_factor_sets)
                    for factor_names in materialized_factor_sets:
                        entries = tuple(known_factors[name] for name in factor_names)
                        candidates.append(
                            _candidate(
                                version=version,
                                combination=combination,
                                outcome=outcome,
                                analysis_family=family,
                                factor_entries=entries,
                                engine=engine_assignments[family],
                                rows=applicable_rows,
                                support_policy=policy,
                            )
                        )
    candidates.sort(
        key=lambda candidate: (
            candidate.dataset_version_id,
            candidate.source_families,
            candidate.curve_outcome,
            candidate.analysis_family,
            candidate.factor_names,
        )
    )
    prune_counts: dict[str, int] = {}
    for candidate in candidates:
        if candidate.status != "pruned":
            continue
        for reason in candidate.reason_codes:
            prune_counts[reason] = prune_counts.get(reason, 0) + 1
    if compressed_prune_count:
        prune_counts["FACTOR_PROFILE_PRUNED_BEFORE_EXPANSION"] = compressed_prune_count
    pruned_families = tuple(
        PrunedFamily(reason_code=reason, candidate_count=count)
        for reason, count in sorted(prune_counts.items())
    )
    accounted_count = (
        sum(1 for candidate in candidates if candidate.status in {"run", "skipped", "pruned", "failed"})
        + compressed_prune_count
    )
    return AnalysisRegistry(
        candidates=tuple(candidates),
        pruned_families=pruned_families,
        theoretical_candidate_count=theoretical_count,
        accounted_candidate_count=accounted_count,
        reconciles=theoretical_count == accounted_count,
    )


__all__ = [
    "AnalysisCandidate",
    "AnalysisRegistry",
    "PrunedFamily",
    "SourceCombination",
    "build_analysis_registry",
    "build_source_combinations",
]
