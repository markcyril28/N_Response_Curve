from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import itertools
import json
import math
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


def _json_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_data(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_data(item) for item in value]
    return value


@dataclass(frozen=True)
class SourceCombination:
    """One deterministic source-family membership set."""

    combination_id: str
    source_families: tuple[str, ...]


@dataclass(frozen=True)
class PrespecifiedHypothesis:
    """One bounded, reviewable hypothesis allowed to become executable."""

    hypothesis_id: str
    dataset_version_id: str
    source_combination_id: str
    curve_outcome: str
    factor_names: tuple[str, ...]
    contrast_specification: Mapping[str, Any]
    analysis_family: str
    engine: str
    multiplicity_family_id: str
    support_rule_id: str | None = None
    support_policy: Mapping[str, Any] = field(default_factory=dict)
    factor_representations: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )


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
    factor_cell_study_counts: Mapping[str, int] = field(default_factory=dict)
    hypothesis_id: str | None = None
    prespecified_contrast: Mapping[str, Any] = field(default_factory=dict)
    multiplicity_family_id: str | None = None
    support_rule_id: str | None = None
    support_policy: Mapping[str, Any] = field(default_factory=dict)
    factor_representations: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class PrunedFamily:
    """Compressed ledger entry retaining a reason and exact number of pruned candidates."""

    reason_code: str
    candidate_count: int


@dataclass(frozen=True)
class MultiplicityFamily:
    """Immutable prespecified membership for one cross-candidate adjustment family."""

    family_id: str
    method: str
    candidate_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisRegistry:
    """Complete concrete analysis space plus a compressed pruning ledger."""

    candidates: tuple[AnalysisCandidate, ...]
    pruned_families: tuple[PrunedFamily, ...]
    theoretical_candidate_count: int
    accounted_candidate_count: int
    reconciles: bool
    multiplicity_families: tuple[MultiplicityFamily, ...] = ()


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
    outcome: str | None,
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
        if outcome is not None and not outcome_is_present(row.get(outcome)):
            continue
        rows.append(row)
    return rows


def _support_policy_values(
    policy: Mapping[str, Any] | None,
) -> dict[str, int | float] | None:
    if policy is None:
        return None
    legacy = {
        "minimum_independent_studies",
        "minimum_factor_cell_count",
        "maximum_factor_cardinality",
        "minimum_residual_information",
    }
    reviewed = {
        "minimum_independent_series",
        "minimum_observations_per_cell",
        "minimum_class_events_per_parameter",
        "minimum_residual_df",
        "maximum_missing_fraction",
        "maximum_factor_cardinality",
        "minimum_independent_studies",
    }
    keys = set(policy)
    if legacy.issubset(keys):
        normalized: dict[str, int | float] = {
            "minimum_independent_series": 1,
            "minimum_observations_per_cell": policy["minimum_factor_cell_count"],
            "minimum_class_events_per_parameter": 1,
            "minimum_residual_df": policy["minimum_residual_information"],
            "maximum_missing_fraction": 1.0,
            "maximum_factor_cardinality": policy["maximum_factor_cardinality"],
            "minimum_independent_studies": policy["minimum_independent_studies"],
            "cell_support_uses_independent_studies": 1,
        }
    elif reviewed.issubset(keys):
        normalized = {key: policy[key] for key in reviewed}
        normalized["cell_support_uses_independent_studies"] = 0
    else:
        required = reviewed if keys & (reviewed - legacy) else legacy
        missing = sorted(required - keys)
        raise ValueError(
            "Support policy is missing required field(s): " + ", ".join(missing)
        )
    for key in reviewed - {"maximum_missing_fraction"}:
        value = normalized[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Support policy {key!r} must be an integer >= 1")
    maximum_missing_fraction = normalized["maximum_missing_fraction"]
    if (
        isinstance(maximum_missing_fraction, bool)
        or not isinstance(maximum_missing_fraction, (int, float))
        or not 0.0 <= float(maximum_missing_fraction) <= 1.0
    ):
        raise ValueError(
            "Support policy 'maximum_missing_fraction' must be numeric in [0, 1]"
        )
    normalized["maximum_missing_fraction"] = float(maximum_missing_fraction)
    return normalized


def _support_summary(
    rows: Sequence[Mapping[str, Any]],
    factor_names: Sequence[str],
    factor_representations: Mapping[str, Mapping[str, Any]],
) -> tuple[int, int, dict[str, int], dict[str, int], int, int, float]:
    complete_rows: list[Mapping[str, Any]] = []
    for row in rows:
        if all(
            factor_value(
                row,
                name,
                representation=factor_representations.get(name),
            )
            is not None
            for name in factor_names
        ):
            complete_rows.append(row)
    studies = {str(row.get("study_uid") or "") for row in complete_rows}
    studies.discard("")
    independent_series = {
        str(row.get("response_series_uid") or "") for row in complete_rows
    }
    independent_series.discard("")
    missing_group_count = sum(not str(row.get("study_uid") or "") for row in complete_rows)
    cell_counts: dict[str, int] = {}
    cell_studies: dict[str, set[str]] = {}
    if factor_names:
        for row in complete_rows:
            cell = "|".join(
                str(
                    factor_value(
                        row,
                        name,
                        representation=factor_representations.get(name),
                    )
                )
                for name in factor_names
            )
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
            cell_studies.setdefault(cell, set())
            study_uid = str(row.get("study_uid") or "")
            if study_uid:
                cell_studies[cell].add(study_uid)
    return (
        len(studies),
        len(independent_series),
        {key: cell_counts[key] for key in sorted(cell_counts)},
        {key: len(cell_studies.get(key, set())) for key in sorted(cell_counts)},
        len(complete_rows),
        missing_group_count,
        (len(rows) - len(complete_rows)) / len(rows) if rows else 1.0,
    )


def _reasons_for_candidate(
    *,
    version: DatasetVersion,
    rows: Sequence[Mapping[str, Any]],
    factor_entries: Sequence[FactorCatalogEntry],
    factor_representations: Mapping[str, Mapping[str, Any]],
    curve_outcome: str,
    analysis_family: str,
    support_policy: Mapping[str, int | float] | None,
) -> tuple[str, tuple[str, ...], int, Mapping[str, int], Mapping[str, int]]:
    if version.status != "available":
        return "skipped", tuple(version.reason_codes or ("DATASET_VERSION_UNAVAILABLE",)), 0, {}, {}
    if version.version_id == "D00_inventory_all" and analysis_family != "coverage_and_missingness":
        return "skipped", ("DATASET_VERSION_NOT_PERMITTED_FOR_ANALYSIS_FAMILY",), 0, {}, {}
    if not rows:
        return "pruned", ("NO_OUTCOME_SUPPORT",), 0, {}, {}
    if analysis_family in {
        "coverage_and_missingness",
        "curve_feature_clustering",
        "dataset_and_source_robustness",
    }:
        studies = {str(row.get("study_uid") or "") for row in rows}
        studies.discard("")
        return "run", (), len(studies), {}, {}
    if not factor_entries:
        return "pruned", ("NO_CONFIGURED_FACTORS",), 0, {}, {}
    if any(entry.leakage_restricted for entry in factor_entries):
        return "pruned", ("LEAKAGE_RESTRICTED_FACTOR",), 0, {}, {}
    (
        independent_studies,
        independent_series,
        cell_counts,
        cell_study_counts,
        complete_rows,
        missing_group_count,
        missing_fraction,
    ) = _support_summary(
        rows,
        [entry.factor_name for entry in factor_entries],
        factor_representations,
    )
    if analysis_family == "one_factor_descriptive":
        return (
            ("run", (), independent_studies, cell_counts, cell_study_counts)
            if complete_rows
            else ("pruned", ("NO_FACTOR_COMPLETE_CASES",), independent_studies, cell_counts, cell_study_counts)
        )
    if support_policy is None:
        return "pruned", ("SUPPORT_POLICY_REQUIRED",), independent_studies, cell_counts, cell_study_counts
    reasons: list[str] = []
    if missing_group_count:
        reasons.append("MISSING_GROUP_IDENTITY")
    if independent_studies < support_policy["minimum_independent_studies"]:
        reasons.append("INSUFFICIENT_INDEPENDENT_STUDY_SUPPORT")
    if independent_series < support_policy["minimum_independent_series"]:
        reasons.append("INSUFFICIENT_INDEPENDENT_SERIES_SUPPORT")
    if missing_fraction > support_policy["maximum_missing_fraction"]:
        reasons.append("MISSING_FRACTION_EXCEEDS_SUPPORT_POLICY")
    if len(cell_counts) > support_policy["maximum_factor_cardinality"]:
        reasons.append("FACTOR_CARDINALITY_EXCEEDS_SUPPORT_POLICY")
    support_counts = (
        cell_study_counts
        if support_policy["cell_support_uses_independent_studies"]
        else cell_counts
    )
    if (
        support_counts
        and min(support_counts.values())
        < support_policy["minimum_observations_per_cell"]
    ):
        reasons.append("INSUFFICIENT_FACTOR_CELL_SUPPORT")
    residual_information = complete_rows - len(factor_entries) - 1
    if residual_information < support_policy["minimum_residual_df"]:
        reasons.append("INSUFFICIENT_RESIDUAL_INFORMATION")
    outcome_counts: dict[str, int] = {}
    for row in rows:
        value = row.get(curve_outcome)
        if isinstance(value, str) and value:
            outcome_counts[value] = outcome_counts.get(value, 0) + 1
    if outcome_counts:
        required_events = (
            support_policy["minimum_class_events_per_parameter"]
            * max(1, len(factor_entries))
        )
        if min(outcome_counts.values()) < required_events:
            reasons.append("INSUFFICIENT_CLASS_EVENTS_PER_PARAMETER")
    if reasons:
        return "pruned", tuple(sorted(reasons)), independent_studies, cell_counts, cell_study_counts
    return "run", (), independent_studies, cell_counts, cell_study_counts


def _candidate(
    *,
    version: DatasetVersion,
    combination: SourceCombination,
    outcome: str,
    analysis_family: str,
    factor_entries: Sequence[FactorCatalogEntry],
    engine: str,
    rows: Sequence[Mapping[str, Any]],
    support_policy: Mapping[str, int | float] | None,
    hypothesis: PrespecifiedHypothesis | None = None,
) -> AnalysisCandidate:
    factor_names = tuple(entry.factor_name for entry in factor_entries)
    effective_support_policy = (
        _support_policy_values(hypothesis.support_policy)
        if hypothesis is not None and hypothesis.support_policy
        else support_policy
    )
    factor_representations = (
        hypothesis.factor_representations if hypothesis is not None else {}
    )
    status, reasons, independent_studies, cell_counts, cell_study_counts = _reasons_for_candidate(
        version=version,
        rows=rows,
        factor_entries=factor_entries,
        factor_representations=factor_representations,
        curve_outcome=outcome,
        analysis_family=analysis_family,
        support_policy=effective_support_policy,
    )
    specification = {
        "analysis_family": analysis_family,
        "curve_outcome": outcome,
        "dataset_version_id": version.version_id,
        "engine": engine,
        "factor_names": factor_names,
        "hypothesis_id": hypothesis.hypothesis_id if hypothesis is not None else None,
        "contrast_specification": (
            _json_data(hypothesis.contrast_specification)
            if hypothesis is not None
            else {}
        ),
        "multiplicity_family_id": hypothesis.multiplicity_family_id if hypothesis is not None else None,
        "support_rule_id": hypothesis.support_rule_id if hypothesis is not None else None,
        "factor_representations": (
            {
                name: _json_data(representation)
                for name, representation in hypothesis.factor_representations.items()
            }
            if hypothesis is not None
            else {}
        ),
        "source_families": combination.source_families,
        "support_policy": _json_data(effective_support_policy or {}),
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
        factor_cell_study_counts=cell_study_counts,
        hypothesis_id=hypothesis.hypothesis_id if hypothesis is not None else None,
        prespecified_contrast=(
            _json_data(hypothesis.contrast_specification)
            if hypothesis is not None
            else {}
        ),
        multiplicity_family_id=hypothesis.multiplicity_family_id if hypothesis is not None else None,
        support_rule_id=hypothesis.support_rule_id if hypothesis is not None else None,
        support_policy=_json_data(effective_support_policy or {}),
        factor_representations=(
            {
                name: _json_data(representation)
                for name, representation in hypothesis.factor_representations.items()
            }
            if hypothesis is not None
            else {}
        ),
    )


def _normalize_hypotheses(
    specifications: Sequence[PrespecifiedHypothesis | Mapping[str, Any]],
) -> tuple[PrespecifiedHypothesis, ...]:
    normalized: list[PrespecifiedHypothesis] = []
    for raw in specifications:
        if isinstance(raw, PrespecifiedHypothesis):
            hypothesis = raw
        elif isinstance(raw, Mapping):
            required = (
                "hypothesis_id",
                "dataset_version_id",
                "source_combination_id",
                "curve_outcome",
                "factor_names",
                "analysis_family",
                "engine",
                "multiplicity_family_id",
            )
            missing = [
                name
                for name in required
                if name not in raw or (name != "factor_names" and not raw.get(name))
            ]
            if missing:
                raise ValueError(
                    "Prespecified hypothesis is missing required field(s): "
                    + ", ".join(missing)
                )
            factor_names = tuple(str(name) for name in raw["factor_names"])
            contrast = raw.get("contrast_specification", {})
            if not isinstance(contrast, Mapping):
                raise ValueError("Prespecified hypothesis contrast_specification must be a mapping")
            hypothesis_support = raw.get("support_policy", {})
            factor_representations = raw.get("factor_representations", {})
            if not isinstance(hypothesis_support, Mapping):
                raise ValueError("Prespecified hypothesis support_policy must be a mapping")
            if not isinstance(factor_representations, Mapping) or any(
                not isinstance(value, Mapping)
                for value in factor_representations.values()
            ):
                raise ValueError(
                    "Prespecified hypothesis factor_representations must map factors to mappings"
                )
            hypothesis = PrespecifiedHypothesis(
                hypothesis_id=str(raw["hypothesis_id"]),
                dataset_version_id=str(raw["dataset_version_id"]),
                source_combination_id=str(raw["source_combination_id"]),
                curve_outcome=str(raw["curve_outcome"]),
                factor_names=factor_names,
                contrast_specification=_json_data(contrast),
                analysis_family=str(raw["analysis_family"]),
                engine=str(raw["engine"]),
                multiplicity_family_id=str(raw["multiplicity_family_id"]),
                support_rule_id=(
                    str(raw["support_rule_id"])
                    if raw.get("support_rule_id") is not None
                    else None
                ),
                support_policy=_json_data(hypothesis_support),
                factor_representations={
                    str(name): _json_data(value)
                    for name, value in factor_representations.items()
                },
            )
        else:
            raise ValueError("Prespecified hypotheses must be mappings or PrespecifiedHypothesis values")
        text_fields = (
            hypothesis.hypothesis_id,
            hypothesis.dataset_version_id,
            hypothesis.source_combination_id,
            hypothesis.curve_outcome,
            hypothesis.analysis_family,
            hypothesis.engine,
            hypothesis.multiplicity_family_id,
        )
        if any(not value for value in text_fields):
            raise ValueError("Prespecified hypothesis identifiers and ownership fields must be nonempty")
        if len(hypothesis.factor_names) != len(set(hypothesis.factor_names)):
            raise ValueError("Prespecified hypothesis factor_names must be unique")
        if set(hypothesis.factor_representations) - set(hypothesis.factor_names):
            raise ValueError(
                "Prespecified hypothesis factor representations must belong to declared factors"
            )
        if hypothesis.support_policy and not hypothesis.support_rule_id:
            raise ValueError(
                "Prespecified hypothesis support_policy requires support_rule_id"
            )
        try:
            json.dumps(_json_data(hypothesis.contrast_specification), allow_nan=False, sort_keys=True)
            json.dumps(_json_data(hypothesis.support_policy), allow_nan=False, sort_keys=True)
            json.dumps(
                _json_data(hypothesis.factor_representations),
                allow_nan=False,
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Prespecified hypothesis semantic controls must be finite JSON data"
            ) from exc
        normalized.append(hypothesis)
    identifiers = [item.hypothesis_id for item in normalized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Prespecified hypothesis IDs must be unique")
    return tuple(normalized)


def _multiplicity_family_registry(
    candidates: Sequence[AnalysisCandidate],
) -> tuple[MultiplicityFamily, ...]:
    grouped: dict[str, list[AnalysisCandidate]] = {}
    for candidate in candidates:
        family_id = candidate.multiplicity_family_id
        if family_id is None:
            continue
        if candidate.hypothesis_id is None:
            raise ValueError(
                "Multiplicity-family membership requires a prespecified hypothesis identifier"
            )
        grouped.setdefault(family_id, []).append(candidate)
    families: list[MultiplicityFamily] = []
    for family_id, members in sorted(grouped.items()):
        candidate_ids = tuple(sorted(member.candidate_id for member in members))
        hypothesis_ids = tuple(sorted(str(member.hypothesis_id) for member in members))
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"Multiplicity family {family_id!r} contains duplicate candidates")
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError(f"Multiplicity family {family_id!r} contains duplicate hypotheses")
        families.append(
            MultiplicityFamily(
                family_id=family_id,
                method="BH",
                candidate_ids=candidate_ids,
                hypothesis_ids=hypothesis_ids,
            )
        )
    return tuple(families)


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
    hypothesis_specifications: Sequence[PrespecifiedHypothesis | Mapping[str, Any]] | None = None,
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
    bounded_hypotheses = (
        None
        if hypothesis_specifications is None
        else _normalize_hypotheses(hypothesis_specifications)
    )
    if bounded_hypotheses is not None:
        known_version_ids = {version.version_id for version in versions}
        known_combination_ids = {combination.combination_id for combination in combinations}
        for hypothesis in bounded_hypotheses:
            if hypothesis.dataset_version_id not in known_version_ids:
                raise ValueError(f"Prespecified hypothesis refers to unknown dataset version: {hypothesis.hypothesis_id}")
            if hypothesis.source_combination_id not in known_combination_ids:
                raise ValueError(f"Prespecified hypothesis refers to unknown source view: {hypothesis.hypothesis_id}")
            if hypothesis.analysis_family not in family_names:
                raise ValueError(f"Prespecified hypothesis refers to unconfigured analysis family: {hypothesis.hypothesis_id}")
            if hypothesis.engine != engine_assignments[hypothesis.analysis_family]:
                raise ValueError(f"Prespecified hypothesis engine conflicts with primary ownership: {hypothesis.hypothesis_id}")
            unknown_hypothesis_factors = set(hypothesis.factor_names) - set(known_factors)
            if unknown_hypothesis_factors:
                raise ValueError(f"Prespecified hypothesis refers to unknown factor(s): {hypothesis.hypothesis_id}")

    candidates: list[AnalysisCandidate] = []
    theoretical_count = 0
    compressed_prune_count = 0
    expansion_families = {
        "all_supported_interactions",
        "multivariable_mixed_effects",
        "observation_level_curve_modification",
        "penalized_predictive_models",
    }
    materialized_hypothesis_ids: set[str] = set()
    for version in versions:
        for combination in combinations:
            for family in family_names:
                if family == "observation_level_curve_modification":
                    family_outcomes = (("yield_t_ha", None),)
                elif family == "curve_feature_clustering":
                    family_outcomes = (("curve_feature_profile", None),)
                else:
                    family_outcomes = tuple((outcome, outcome) for outcome in outcomes)
                for outcome, support_outcome in family_outcomes:
                    applicable_rows = _selected_rows(rows, version, combination, support_outcome)
                    if bounded_hypotheses is not None:
                        scoped_hypotheses = tuple(
                            hypothesis
                            for hypothesis in bounded_hypotheses
                            if hypothesis.dataset_version_id == version.version_id
                            and hypothesis.source_combination_id == combination.combination_id
                            and hypothesis.curve_outcome == outcome
                            and hypothesis.analysis_family == family
                            and hypothesis.engine == engine_assignments[family]
                        )
                        theoretical_count += len(scoped_hypotheses)
                        for hypothesis in scoped_hypotheses:
                            entries = tuple(known_factors[name] for name in hypothesis.factor_names)
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
                                    hypothesis=hypothesis,
                                )
                            )
                            materialized_hypothesis_ids.add(hypothesis.hypothesis_id)
                        continue
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
    if bounded_hypotheses is not None:
        missing_hypotheses = {
            hypothesis.hypothesis_id for hypothesis in bounded_hypotheses
        } - materialized_hypothesis_ids
        if missing_hypotheses:
            raise ValueError(
                "Prespecified hypotheses could not be represented by the configured registry: "
                + ", ".join(sorted(missing_hypotheses))
            )
        if any(candidate.status == "run" and candidate.hypothesis_id is None for candidate in candidates):
            raise ValueError("Bounded hypothesis registry produced an undeclared executable candidate")
    candidates.sort(
        key=lambda candidate: (
            candidate.dataset_version_id,
            candidate.source_families,
            candidate.curve_outcome,
            candidate.analysis_family,
            candidate.factor_names,
            candidate.hypothesis_id or "",
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
        multiplicity_families=_multiplicity_family_registry(candidates),
    )


__all__ = [
    "AnalysisCandidate",
    "AnalysisRegistry",
    "MultiplicityFamily",
    "PrespecifiedHypothesis",
    "PrunedFamily",
    "SourceCombination",
    "build_analysis_registry",
    "build_source_combinations",
]
