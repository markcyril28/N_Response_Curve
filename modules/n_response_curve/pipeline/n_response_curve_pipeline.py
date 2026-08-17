from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import resource
import sys
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from dataclasses import dataclass

from n_response_curve.analysis.policy_artifacts import (
    AnalysisPolicyBundle,
    PolicyArtifactError,
    load_analysis_policy_manifest,
)
from n_response_curve.data.config import ConfigError, ValidatedConfig, load_config
from n_response_curve.data.cleaning import (
    FINAL_CLEANING_METADATA_FIELDS,
    FinalCleaningResult,
    apply_final_cleaning,
)
from n_response_curve.data.curate import CurationResult, curate_ingestion
from n_response_curve.data.duplicates import SeriesResolution, resolve_response_series
from n_response_curve.data.eligibility import EligibilityResult, assign_eligibility
from n_response_curve.data.ingest import (
    BUILTIN_ADAPTER_SPECS,
    IngestionResult,
    SourceAdapterSpec,
    WorkbookCsvReconciliation,
    ingest_configured_sources,
    verify_configured_source_integrity,
)
from n_response_curve.data.policy_artifacts import (
    SourceDataPolicyBundle,
    SourceDataPolicyError,
    load_source_data_policy_manifest,
    validate_source_data_policy_coverage,
    validate_source_scope_activation,
)
from n_response_curve.data.provenance import (
    SourceIntegrityReport,
    VerificationRound,
    plan_literature_verification_round,
    stable_json_sha256,
    verify_source_integrity,  # noqa: F401  (Phase 1 compatibility re-export)
)
from n_response_curve.data.qc import QcReport, build_qc_report
from n_response_curve.data.runtime_source_contracts import (
    build_builtin_source_category_lookups,
    build_builtin_source_maps,
    build_builtin_workbook_reconciliations,
)
from n_response_curve.data.duplicates import DuplicateRuleSet
from n_response_curve.logging.run_logging import (
    RunLogger,
    console_colors_enabled,
    exception_was_logged,
    format_console_exception,
)
from n_response_curve.pipeline.policy_governance import (
    ReviewGatePolicy,
    phase_two_review_disposition,
    release_config_sha256,
    validate_runtime_policy,
)
from n_response_curve.pipeline.workflow import (
    _code_fingerprint,
    _find_reusable_completed_release,
    _release_run_id,
    build_effective_model_policy,
    release_phases_three_to_five,
    run_phase_four,
    run_phase_three,
)
from n_response_curve.pipeline.workspace_reset import (
    clear_log_workspace,
    clear_output_workspace,
)
from n_response_curve.reporting.release import verify_completed_release_package
from n_response_curve.reporting.workspace_outputs import (
    WorkspaceOutputs,
    materialize_workspace_views,
    workspace_view_display_paths,
)


_FACTOR_ANALYSIS_FAMILIES = frozenset(
    {
        "one_factor_descriptive",
        "one_factor_inferential",
        "all_supported_interactions",
        "multivariable_mixed_effects",
        "observation_level_curve_modification",
        "penalized_predictive_models",
        "marginal_contrasts",
    }
)


@dataclass(frozen=True)
class PhaseTwoResult:
    ingestion: IngestionResult
    curation: CurationResult
    resolution: SeriesResolution
    eligibility: EligibilityResult
    analysis_eligibility: EligibilityResult
    qc: QcReport
    source_data_policy: SourceDataPolicyBundle | None = None
    final_cleaning: FinalCleaningResult | None = None
    untrimmed_analysis_eligibility: EligibilityResult | None = None
    literature_verification: VerificationRound | None = None
    source_scope_snapshot: Mapping[str, object] | None = None


def _build_runtime_source_scope_snapshot(
    config: ValidatedConfig,
    source_data_policy: SourceDataPolicyBundle,
    integrity_report: SourceIntegrityReport,
) -> Mapping[str, object]:
    """Materialize the exact approved source registry used by this Phase 2 run."""

    authority = source_data_policy.artifact_authorities["source_scope"]
    manifest_root = config.paths["source_manifest"].resolve().parent
    restricted_artifact_aliases: dict[str, str] = {}
    for source_name, raw in sorted(config.sources.items()):
        if raw.get("data_classification") != "restricted":
            continue
        for field_name in ("data_path", "workbook", "schema_map"):
            value = raw.get(field_name)
            if not isinstance(value, str) or not value.strip():
                continue
            candidate_path = (config.project_root / value).resolve()
            try:
                relative = candidate_path.relative_to(manifest_root).as_posix()
            except ValueError:
                continue
            if relative in integrity_report.artifact_sha256:
                restricted_artifact_aliases[relative] = (
                    "restricted_artifact_"
                    + stable_json_sha256(
                        {"source_name": source_name, "artifact_role": field_name}
                    )[:24]
                )
    sources: list[dict[str, object]] = []
    for source_name in sorted(config.sources):
        raw = config.sources[source_name]
        scope = source_data_policy.source_scope[source_name]
        artifact_path: str | None = None
        artifact_sha256: str | None = None
        nonblank_data_row_count: int | None = None
        availability_observation = "expected_unavailable"
        if raw.get("availability") == "available":
            data_path = (config.project_root / str(raw["data_path"])).resolve()
            try:
                candidate = data_path.relative_to(manifest_root).as_posix()
            except ValueError as exc:
                raise ConfigError(
                    f"Configured source {source_name!r} is outside the manifest root"
                ) from exc
            if candidate not in integrity_report.artifact_sha256:
                raise ConfigError(
                    f"Configured available source {source_name!r} is absent from the verified manifest"
                )
            artifact_sha256 = integrity_report.artifact_sha256[candidate]
            if raw.get("data_classification") != "restricted":
                artifact_path = candidate
            metadata = integrity_report.artifact_metadata.get(candidate, {})
            rows_declaration = metadata.get("rows")
            if isinstance(rows_declaration, str):
                match = re.fullmatch(r"(\d+) data rows", rows_declaration.strip())
                if match is not None:
                    nonblank_data_row_count = int(match.group(1))
            availability_observation = (
                "observed_empty" if nonblank_data_row_count == 0 else "present_artifact"
            )
        sources.append(
            {
                "source_name": source_name,
                "source_type": raw.get("source_type"),
                "source_family": raw.get("source_family"),
                "country_code": raw.get("country_code"),
                "availability": raw.get("availability"),
                "availability_observation": availability_observation,
                "nonblank_data_row_count": nonblank_data_row_count,
                "confirmation_status": raw.get("confirmation_status"),
                "activation_status": scope.activation_status,
                "schema_harmonization_status": scope.schema_harmonization_status,
                "unit_comparability_status": scope.unit_comparability_status,
                "review_id": scope.review_id,
                "shape_adapter_version": raw.get("shape_adapter_version"),
                "encoding": raw.get("encoding"),
                "data_classification": raw.get("data_classification"),
                "data_path": (
                    None
                    if raw.get("data_classification") == "restricted"
                    else raw.get("data_path")
                ),
                "schema_map": (
                    None
                    if raw.get("data_classification") == "restricted"
                    else raw.get("schema_map")
                ),
                "source_locator_disclosure_status": (
                    "withheld_restricted_source_locator"
                    if raw.get("data_classification") == "restricted"
                    else "registered_internal_source_locator"
                ),
                "manifest_artifact_path": artifact_path,
                "manifest_artifact_sha256": artifact_sha256,
            }
        )
    inventory_counts = {
        "expected_source_count": len(sources),
        "source_family_count": len(
            {str(source["source_family"]) for source in sources}
        ),
        "expected_unavailable_source_count": sum(
            source["availability_observation"] == "expected_unavailable"
            for source in sources
        ),
        "observed_empty_source_count": sum(
            source["availability_observation"] == "observed_empty" for source in sources
        ),
        "present_source_count": sum(
            source["availability_observation"] == "present_artifact"
            for source in sources
        ),
        "present_artifact_count": integrity_report.checked_files,
        "present_nonblank_data_row_count": sum(
            source["nonblank_data_row_count"]
            if isinstance(source["nonblank_data_row_count"], int)
            else 0
            for source in sources
        ),
    }
    payload: dict[str, object] = {
        "snapshot_format": "source-scope-v1",
        "scope_revision": authority.artifact_version,
        "scope_artifact_sha256": authority.sha256,
        "approved_by": authority.approved_by,
        "approval_date": authority.approval_date,
        "current_source_cutoff": authority.approval_date,
        "cutoff_semantics": (
            "Sources represented by the approved scope artifact through its approval date; "
            "later receipts require a new approved source-scope artifact."
        ),
        "enabled_sources": tuple(sorted(config.enabled_sources)),
        "scanned_roots": (manifest_root.relative_to(config.project_root).as_posix(),),
        "manifest_artifact_count": integrity_report.checked_files,
        "manifest_artifact_sha256": {
            restricted_artifact_aliases.get(path, path): digest
            for path, digest in integrity_report.artifact_sha256.items()
        },
        "duplicate_byte_groups": tuple(
            tuple(restricted_artifact_aliases.get(path, path) for path in group)
            for group in integrity_report.duplicate_byte_groups
        ),
        "inventory_counts": inventory_counts,
        "sources": tuple(sources),
    }
    payload["snapshot_sha256"] = stable_json_sha256(payload)
    return MappingProxyType(payload)


def _resource_snapshot(
    config: ValidatedConfig,
    *,
    phase: str,
) -> dict[str, Any]:
    """Capture a lightweight planned-versus-observed process resource snapshot."""

    run = config.raw["run"]
    threads_per_job = int(run["r_threads_per_job"])
    parallel_jobs = int(run["max_parallel_r_jobs"])
    visible_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "phase": phase,
        "planned_r_threads_per_job": threads_per_job,
        "planned_max_parallel_r_jobs": parallel_jobs,
        "planned_r_cpu_budget": threads_per_job * parallel_jobs,
        "affinity_visible_cpu_count": visible_cpus,
        "process_cpu_seconds": round(float(usage.ru_utime + usage.ru_stime), 6),
        "process_max_rss_bytes": max(0, int(usage.ru_maxrss) * 1024),
    }


def _load_analysis_policy(
    config: ValidatedConfig,
) -> AnalysisPolicyBundle | None:
    if config.analysis_policy_manifest is None:
        return None
    if config.analysis_policy_manifest_sha256 is None:
        raise ConfigError("Analysis policy manifest hash is unavailable")
    active_factor_engines = {
        config.engine_assignments[family]
        for family in config.analysis_families
        if family in _FACTOR_ANALYSIS_FAMILIES
    }
    required_factor_engine_pairs = tuple(
        sorted(
            (factor_name, engine)
            for factor_name in config.explanatory_factors
            for engine in active_factor_engines
        )
    )
    try:
        return load_analysis_policy_manifest(
            config.analysis_policy_manifest,
            expected_sha256=config.analysis_policy_manifest_sha256,
            project_root=config.project_root,
            required_factor_engine_pairs=required_factor_engine_pairs,
        )
    except PolicyArtifactError as exc:
        raise ConfigError(f"Analysis policy validation failed: {exc}") from exc


def _load_source_data_policy(
    config: ValidatedConfig,
) -> SourceDataPolicyBundle | None:
    if config.source_data_policy_manifest is None:
        return None
    if config.source_data_policy_manifest_sha256 is None:
        raise ConfigError("Source-data policy manifest hash is unavailable")
    secret_env = config.source_data_policy_secret_env
    if secret_env is None:
        raise ConfigError(
            "Source-data policy pseudonym-secret reference is unavailable"
        )
    secret = os.environ.get(secret_env)
    if secret is None:
        raise ConfigError(
            "Source-data policy pseudonym secret is unavailable in the configured "
            "environment variable"
        )
    try:
        return load_source_data_policy_manifest(
            config.source_data_policy_manifest,
            expected_sha256=config.source_data_policy_manifest_sha256,
            project_root=config.project_root,
            secrets={secret_env: secret.encode("utf-8")},
        )
    except SourceDataPolicyError as exc:
        raise ConfigError(f"Source-data policy validation failed: {exc}") from exc


def _plan_literature_verification(
    source_data_policy: SourceDataPolicyBundle,
    records: tuple[dict[str, Any], ...] | list[dict[str, Any]] | Any,
) -> VerificationRound | None:
    """Execute the authenticated SRC-07 sampling policy for its exact source scope."""

    policy = getattr(source_data_policy, "literature_verification_policy", None)
    if policy is None:
        return None
    source_names = set(
        getattr(source_data_policy, "literature_verification_source_names", ())
    )
    scoped_records = tuple(
        record
        for record in records
        if str(record.get("source_name") or "") in source_names
    )
    if not scoped_records:
        raise ConfigError(
            "Literature-verification policy matched no records in the curated inventory"
        )
    try:
        return plan_literature_verification_round(
            scoped_records,
            policy=policy,
            completed_results=(
                getattr(source_data_policy, "literature_verification_results", ())
            ),
            designated_reviewers=source_data_policy.designated_reviewers,
        )
    except ConfigError as exc:
        raise ConfigError(f"Literature-verification planning failed: {exc}") from exc


def _reviewed_source_adapter_specs(
    config: ValidatedConfig,
    source_data_policy: SourceDataPolicyBundle,
) -> dict[str, SourceAdapterSpec]:
    """Materialize versioned ingest contracts from approved source maps."""

    specs: dict[str, SourceAdapterSpec] = {}
    for source_name in config.enabled_sources:
        source_map = source_data_policy.source_maps[source_name]
        positions = [disposition.position for disposition in source_map.dispositions]
        maximum_position = max(positions, default=0)
        if (
            maximum_position < 1
            or len(positions) != maximum_position
            or set(positions) != set(range(1, maximum_position + 1))
        ):
            raise ConfigError(
                f"Reviewed source map for {source_name!r} must cover every physical column exactly once"
            )
        adapter_version = config.sources[source_name].get("shape_adapter_version")
        if (
            not isinstance(adapter_version, str)
            or not adapter_version.strip()
            or adapter_version == "unassigned"
        ):
            raise ConfigError(
                f"Enabled source {source_name!r} lacks an assigned shape adapter version"
            )
        builtin_spec = BUILTIN_ADAPTER_SPECS.get(adapter_version)
        if builtin_spec is not None and (
            maximum_position != builtin_spec.expected_physical_columns
            or any(
                source_map.expected_headers.get(position) != expected_header
                for position, expected_header in builtin_spec.expected_headers.items()
            )
        ):
            raise ConfigError(
                f"Reviewed source map for {source_name!r} conflicts with built-in adapter {adapter_version!r}"
            )
        spec = SourceAdapterSpec(
            version=adapter_version,
            expected_physical_columns=maximum_position,
            expected_headers=source_map.expected_headers,
            map_version=source_map.map_version,
            expected_header_sha256=(
                builtin_spec.expected_header_sha256
                if builtin_spec is not None
                else None
            ),
        )
        existing = specs.get(adapter_version)
        if existing is not None and existing != spec:
            raise ConfigError(
                f"Shape adapter version {adapter_version!r} resolves to conflicting reviewed source maps"
            )
        specs[adapter_version] = spec
    return specs


_GENERIC_SOURCE_LOCAL_KEY_FIELDS = (
    "source_name",
    "study_id",
    "trial_id",
    "treatment",
    "planting_year",
    "season",
    "location",
    "n_rate_kg_ha",
    "yield_t_ha",
)
_GENERIC_SOURCE_LOCAL_CASEFOLD_FIELDS = (
    "source_name",
    "study_id",
    "trial_id",
    "treatment",
    "season",
    "location",
)


def _runtime_source_local_duplicate_rule(source_name: str) -> DuplicateRuleSet:
    """No-source-policy fallback exact/probable duplicate rule for one source.

    LTCCE's generic contextual key omits ``experimental_design``,
    ``rice_variety``, ``source_variety_code``, and ``replicate``, so distinct
    LTCCE trials that share treatment/season/location/N-rate/yield collide as
    exact duplicates. LTCCE therefore binds its exact key to the complete
    preserved physical row (``source_name`` + ``raw_cells``) and widens its
    probable key with the omitted contextual fields; every other source keeps
    the unchanged generic rule.
    """
    if source_name != "ltcce":
        return DuplicateRuleSet(
            version=f"runtime-source-policy-2026-08-13-{source_name}",
            review_id="user-source-policy-decisions-2026-08-13",
            exact_key_fields=_GENERIC_SOURCE_LOCAL_KEY_FIELDS,
            probable_key_fields=_GENERIC_SOURCE_LOCAL_KEY_FIELDS,
            probable_numeric_tolerances=MappingProxyType(
                {"n_rate_kg_ha": 0.01, "yield_t_ha": 0.001}
            ),
            casefold_fields=_GENERIC_SOURCE_LOCAL_CASEFOLD_FIELDS,
            source_names=(source_name,),
            probable_cross_source_only=False,
        )
    return DuplicateRuleSet(
        version="runtime-source-policy-2026-08-14-ltcce-v2",
        review_id="user-source-policy-decisions-2026-08-13",
        exact_key_fields=("source_name", "raw_cells"),
        probable_key_fields=_GENERIC_SOURCE_LOCAL_KEY_FIELDS
        + ("experimental_design", "rice_variety", "source_variety_code", "replicate"),
        probable_numeric_tolerances=MappingProxyType(
            {"n_rate_kg_ha": 0.01, "yield_t_ha": 0.001}
        ),
        casefold_fields=_GENERIC_SOURCE_LOCAL_CASEFOLD_FIELDS
        + ("experimental_design", "rice_variety", "source_variety_code"),
        source_names=("ltcce",),
        probable_cross_source_only=False,
    )


def run_phase_two(
    config: ValidatedConfig,
    *,
    source_data_policy: SourceDataPolicyBundle | None = None,
) -> PhaseTwoResult:
    """Execute the non-writing Phase 2 source-to-analysis-ready data gate."""

    if source_data_policy is not None:
        try:
            validate_source_scope_activation(source_data_policy, config)
        except SourceDataPolicyError as exc:
            raise ConfigError(f"Source-scope activation failed: {exc}") from exc
    if source_data_policy is not None:
        ingestion = ingest_configured_sources(
            config,
            adapter_specs=_reviewed_source_adapter_specs(
                config,
                source_data_policy,
            ),
            checksum_revision_approvals=(
                source_data_policy.checksum_revision_approvals
            ),
            source_representation_bases={
                source_name: source_map.representation_basis
                for source_name, source_map in source_data_policy.source_maps.items()
                if source_map.representation_basis_status == "reviewed"
            },
            source_workbook_reconciliations={
                source_name: WorkbookCsvReconciliation(
                    source_name=source_name,
                    workbook_sha256=source_map.workbook_sha256,
                    csv_sha256=source_map.csv_sha256,
                    review_id=source_map.workbook_csv_reconciliation_review_id,
                    basis=source_map.workbook_csv_basis,
                )
                for source_name, source_map in source_data_policy.source_maps.items()
                if (
                    source_name in config.enabled_sources
                    and source_map.workbook_csv_basis
                    in {
                        "parallel_workbook_csv_verified_equivalent",
                        "parallel_workbook_csv_reviewed_csv_authoritative",
                    }
                    and source_map.workbook_sha256 is not None
                    and source_map.csv_sha256 is not None
                    and source_map.workbook_csv_reconciliation_review_id is not None
                )
            },
            designated_reviewers=source_data_policy.designated_reviewers,
        )
    else:
        ingestion = ingest_configured_sources(
            config,
            source_workbook_reconciliations=(
                build_builtin_workbook_reconciliations(config)
            ),
            source_representation_bases={
                source_name: (
                    "observation_level"
                    if source_name == "ltcce"
                    else "unclear_mixed_scope"
                )
                for source_name in getattr(config, "enabled_sources", ())
            },
        )
    if source_data_policy is not None:
        try:
            validate_source_data_policy_coverage(source_data_policy, ingestion)
        except (SourceDataPolicyError, ValueError) as exc:
            raise ConfigError(f"Source-data policy coverage failed: {exc}") from exc
    source_scope_snapshot = None
    if isinstance(source_data_policy, SourceDataPolicyBundle):
        if ingestion.integrity_report is None:
            raise ConfigError(
                "Source-scope snapshot requires source-integrity evidence"
            )
        source_scope_snapshot = _build_runtime_source_scope_snapshot(
            config,
            source_data_policy,
            ingestion.integrity_report,
        )
    if source_data_policy is not None:
        curation_kwargs = dict(source_data_policy.curation_kwargs)
    else:
        # Full mode is self-contained: byte-bound built-in source contracts
        # implement the user's selected source, unit, yield, and category
        # policies without importing organizational approval artifacts.
        curation_kwargs = {
            "source_maps": build_builtin_source_maps(ingestion, config),
            "source_category_lookups": build_builtin_source_category_lookups(
                ingestion,
                config,
            ),
            "require_reviewed_controls": False,
        }
    curation = curate_ingestion(ingestion, config, **curation_kwargs)
    literature_verification = (
        _plan_literature_verification(source_data_policy, curation.records)
        if source_data_policy is not None
        else None
    )
    resolution_kwargs: dict[str, Any] = (
        dict(source_data_policy.resolution_kwargs)
        if source_data_policy is not None
        else {
            "duplicate_rules": (
                *tuple(
                    _runtime_source_local_duplicate_rule(source_name)
                    for source_name in getattr(config, "enabled_sources", ())
                ),
                *(
                    (
                        DuplicateRuleSet(
                            version="runtime-cross-source-policy-2026-08-13-v1",
                            review_id="user-source-policy-decisions-2026-08-13",
                            exact_key_fields=(
                                "study_id",
                                "trial_id",
                                "treatment",
                                "planting_year",
                                "season",
                                "location",
                                "n_rate_kg_ha",
                                "yield_t_ha",
                            ),
                            probable_key_fields=(
                                "study_id",
                                "trial_id",
                                "treatment",
                                "planting_year",
                                "season",
                                "location",
                                "n_rate_kg_ha",
                                "yield_t_ha",
                            ),
                            probable_numeric_tolerances=MappingProxyType(
                                {
                                    "n_rate_kg_ha": 0.01,
                                    "yield_t_ha": 0.001,
                                }
                            ),
                            casefold_fields=(
                                "study_id",
                                "trial_id",
                                "treatment",
                                "season",
                                "location",
                            ),
                            source_names=tuple(getattr(config, "enabled_sources", ())),
                            probable_cross_source_only=True,
                            scope_kind="cross_source",
                        ),
                    )
                    if len(tuple(getattr(config, "enabled_sources", ()))) >= 2
                    else ()
                ),
            ),
        }
    )
    untrimmed_resolution = resolve_response_series(
        curation.records,
        series_identity_dimensions=config.series_identity_dimensions,
        n_level_tolerance_kg_ha=float(
            config.raw["eligibility"]["n_level_tolerance_kg_ha"]
        ),
        **resolution_kwargs,
    )
    untrimmed_analysis_eligibility: EligibilityResult | None = None
    if source_data_policy is not None:
        untrimmed_analysis_eligibility = assign_eligibility(
            untrimmed_resolution.analysis_records,
            policy=config.raw["eligibility"],
        )
    resolution = untrimmed_resolution
    final_cleaning: FinalCleaningResult | None = None
    if source_data_policy is not None:
        final_cleaning = apply_final_cleaning(
            (
                *untrimmed_resolution.records,
                *untrimmed_resolution.aggregate_records,
            ),
            source_data_policy.final_cleaning_policies,
        )
        cleaned_by_uid = {
            str(record["record_uid"]): record for record in final_cleaning.records
        }
        primary_record_uids = set(final_cleaning.primary_record_uids)
        primary_resolution = resolve_response_series(
            tuple(
                cleaned_by_uid[str(record["record_uid"])]
                for record in untrimmed_resolution.records
                if str(record["record_uid"]) in primary_record_uids
            ),
            series_identity_dimensions=config.series_identity_dimensions,
            n_level_tolerance_kg_ha=float(
                config.raw["eligibility"]["n_level_tolerance_kg_ha"]
            ),
            **resolution_kwargs,
        )
        primary_by_uid = {
            str(record["record_uid"]): record for record in primary_resolution.records
        }
        resolution = SeriesResolution(
            records=tuple(
                primary_by_uid.get(
                    str(record["record_uid"]),
                    cleaned_by_uid[str(record["record_uid"])],
                )
                for record in untrimmed_resolution.records
            ),
            aggregate_records=primary_resolution.aggregate_records,
        )
        cleaning_fields = FINAL_CLEANING_METADATA_FIELDS
        assert untrimmed_analysis_eligibility is not None
        untrimmed_analysis_eligibility = EligibilityResult(
            ledger=tuple(
                {
                    **record,
                    **{
                        field: cleaned_by_uid[str(record["record_uid"])].get(field)
                        for field in cleaning_fields
                    },
                }
                for record in untrimmed_analysis_eligibility.ledger
            ),
            series_metrics=untrimmed_analysis_eligibility.series_metrics,
            critical_record_uids=untrimmed_analysis_eligibility.critical_record_uids,
        )
    eligibility = assign_eligibility(
        resolution.records,
        policy=config.raw["eligibility"],
    )
    analysis_eligibility = assign_eligibility(
        resolution.analysis_records,
        policy=config.raw["eligibility"],
    )
    if untrimmed_analysis_eligibility is None:
        untrimmed_analysis_eligibility = analysis_eligibility
    qc = build_qc_report(
        eligibility.ledger,
        inventory_record_uids=(record["record_uid"] for record in curation.records),
    )
    if not qc.reconciles:
        raise ConfigError(
            "Phase 2 source-to-tier QC flow does not reconcile to the master inventory"
        )
    return PhaseTwoResult(
        ingestion=ingestion,
        curation=curation,
        resolution=resolution,
        eligibility=eligibility,
        analysis_eligibility=analysis_eligibility,
        qc=qc,
        source_data_policy=source_data_policy,
        final_cleaning=final_cleaning,
        untrimmed_analysis_eligibility=untrimmed_analysis_eligibility,
        literature_verification=literature_verification,
        source_scope_snapshot=source_scope_snapshot,
    )


_REVIEW_CONTROL_PREFIX = "UNRESOLVED_REVIEW_CONTROL:"
_QC_GATE_REASON_LIMIT = 8
_QC_GATE_UID_LIMIT = 8
_QC_GATE_UID_PREFIX_CHARS = 12


def _qc_gate_blockers(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[tuple[str, int, int], ...], int]:
    """Count blocking rows per reason code, collapsing the per-field review controls."""

    row_counts: dict[str, int] = {}
    control_fields: set[str] = set()
    for row in rows:
        labels: set[str] = set()
        for reason in row.get("eligibility_reason_codes", ()):
            reason = str(reason)
            if reason.startswith(_REVIEW_CONTROL_PREFIX):
                control_fields.add(reason[len(_REVIEW_CONTROL_PREFIX) :])
                labels.add(f"{_REVIEW_CONTROL_PREFIX}*")
            else:
                labels.add(reason)
        for label in labels:
            row_counts[label] = row_counts.get(label, 0) + 1
    ranked = tuple(
        (
            label,
            count,
            len(control_fields) if label == f"{_REVIEW_CONTROL_PREFIX}*" else 0,
        )
        for label, count in sorted(
            row_counts.items(), key=lambda item: (-item[1], item[0])
        )
    )
    return ranked[:_QC_GATE_REASON_LIMIT], max(len(ranked) - _QC_GATE_REASON_LIMIT, 0)


def _qc_gate_failure_message(
    phase_two: PhaseTwoResult,
    *,
    review_rows: Sequence[Mapping[str, Any]],
    blocking_rows: Sequence[Mapping[str, Any]],
    review_gate_policy: ReviewGatePolicy | None,
) -> str:
    """Explain why the gate blocked in reason-code terms, not as a bare identifier dump."""

    tier_counts: dict[str, int] = {}
    for row in blocking_rows:
        tier = str(row.get("eligibility_tier") or "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    tiers = " ".join(f"{tier}={tier_counts[tier]}" for tier in sorted(tier_counts))
    lines = [
        f"Phase 2 strict QC gate failed: {len(blocking_rows)} of "
        f"{phase_two.qc.inventory_rows} warning, unresolved, or excluded records "
        f"require review (blocking tiers: {tiers})."
    ]

    ranked, remaining = _qc_gate_blockers(blocking_rows)
    if ranked:
        label_width = max(len(label) for label, _, _ in ranked)
        lines.append("Top blockers:")
        lines.extend(
            f"  {label:<{label_width}}  {count:>6}"
            + (f" ({fields} controls)" if fields else "")
            for label, count, fields in ranked
        )
        if remaining:
            lines.append(f"  ... ({remaining} more reason codes)")

    if review_gate_policy is None:
        lines.append(
            "No review-gate policy is loaded; validate mode blocks all "
            f"{len(review_rows)} review rows."
        )
    else:
        permitted = sum(
            1
            for row in review_rows
            if review_gate_policy.permits(phase_two_review_disposition(row))
        )
        lines.append(
            f"Review-gate policy {review_gate_policy.policy_id} classifies {permitted} of "
            f"{len(review_rows)} review rows as permitted for writing-mode ledgering; "
            "validate mode blocks all review rows."
        )

    uids = sorted(str(row.get("record_uid", "unresolved")) for row in blocking_rows)
    preview = ", ".join(
        uid[:_QC_GATE_UID_PREFIX_CHARS]
        + ("..." if len(uid) > _QC_GATE_UID_PREFIX_CHARS else "")
        for uid in uids[:_QC_GATE_UID_LIMIT]
    )
    if len(uids) > _QC_GATE_UID_LIMIT:
        preview += f", ... ({len(uids) - _QC_GATE_UID_LIMIT} more)"
    lines.append(f"First blocking record UIDs: {preview}")
    return "\n".join(lines)


def _enforce_phase_two_qc_gate(
    config: ValidatedConfig,
    phase_two: PhaseTwoResult,
    *,
    review_gate_policy: ReviewGatePolicy | None = None,
) -> None:
    """Fail validation on review states; writing modes preserve them in QC ledgers."""

    if config.run_mode != "validate":
        return
    review_rows = tuple(phase_two.qc.review_rows)
    if not review_rows:
        return
    # Validation is the strict diagnostic mode: a disposition policy may classify
    # findings for writing-mode ledgers, but it never exempts a review-bearing row
    # from this gate.
    blocking_rows = review_rows
    raise ConfigError(
        _qc_gate_failure_message(
            phase_two,
            review_rows=review_rows,
            blocking_rows=blocking_rows,
            review_gate_policy=review_gate_policy,
        )
    )


def _enforce_literature_verification_gate(
    config: ValidatedConfig,
    phase_two: PhaseTwoResult,
) -> None:
    """Preserve the legacy validation hook without blocking complete processing."""

    return


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _display_path(path: Path, root: Path) -> str:
    """Project-relative when possible, absolute otherwise.

    N_RESPONSE_LOG_ROOT may legitimately point outside the project, which
    ``_relative`` rejects.
    """

    try:
        return _relative(path, root)
    except ValueError:
        return path.resolve().as_posix()


def _print_validation_plan(config: ValidatedConfig, phase_two: PhaseTwoResult) -> None:
    report = phase_two.ingestion.integrity_report
    if report is None:
        raise ConfigError(
            "Phase 2 ingestion completed without a source-integrity report"
        )
    tier_counts = phase_two.qc.tier_counts
    resolved_series_count = len(
        {
            record["response_series_uid"]
            for record in phase_two.resolution.records
            if record.get("series_status") == "resolved"
            and record.get("response_series_uid")
        }
    )
    print(f"mode={config.run_mode}")
    print(f"writes_outputs={'true' if config.writes_outputs else 'false'}")
    print(f"source_count={len(config.enabled_sources)}")
    for source_name in config.enabled_sources:
        source_path = (
            config.paths["core_source_csv"]
            if source_name == "core_trial_data"
            else Path(config.sources[source_name]["data_path"])
        )
        if source_name != "core_trial_data":
            source_path = config.project_root / source_path
        print(
            f"source={source_name} path={_relative(source_path, config.project_root)}"
        )
    print(f"enabled_models={','.join(config.enabled_models) or 'observed_only'}")
    print(f"scope_countries={','.join(config.scope_countries)}")
    print(f"series_identity_dimensions={','.join(config.series_identity_dimensions)}")
    print(
        f"n_level_tolerance_kg_ha={config.raw['eligibility']['n_level_tolerance_kg_ha']}"
    )
    print(f"comparison_dimensions={','.join(config.comparison_dimensions) or 'none'}")
    print(f"analysis_families={','.join(config.analysis_families)}")
    print(f"source_integrity=pass checked_files={report.checked_files}")
    print(f"canonical_rows={len(phase_two.curation.records)}")
    print(f"blank_source_rows={len(phase_two.ingestion.blank_rows)}")
    print(f"resolved_response_series={resolved_series_count}")
    print(f"eligibility_rows={len(phase_two.eligibility.ledger)}")
    print(f"analysis_eligibility_rows={len(phase_two.analysis_eligibility.ledger)}")
    print(f"tier_A={tier_counts['A']}")
    print(f"tier_B={tier_counts['B']}")
    print(f"tier_C={tier_counts['C']}")
    print(f"tier_D={tier_counts['D']}")
    print(f"critical_records={len(phase_two.qc.critical_record_uids)}")
    print(f"qc_reconciles={'true' if phase_two.qc.reconciles else 'false'}")
    verification = phase_two.literature_verification
    print(
        "literature_verification_status="
        + (verification.status if verification is not None else "not_configured")
    )
    print(
        "stages=config_validation,source_integrity,position_safe_ingestion,canonical_curation,response_series_resolution,eligibility_qc"
    )


def _preflight_runtime(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> tuple[ValidatedConfig, Any, Any, Any, Any]:
    config = load_config(
        config_path, project_root=project_root, check_files=True, preflight_engines=True
    )
    analysis_policy = _load_analysis_policy(config)
    source_data_policy = _load_source_data_policy(config)
    model_policy = build_effective_model_policy(config, analysis_policy)
    policy_snapshot = validate_runtime_policy(config)
    return (
        config,
        analysis_policy,
        source_data_policy,
        model_policy,
        policy_snapshot,
    )


def _reuse_completed_release(
    config: ValidatedConfig,
    *,
    analysis_policy: AnalysisPolicyBundle | None,
    source_data_policy: SourceDataPolicyBundle | None,
    model_policy: Mapping[str, Any],
    policy_snapshot: Any,
) -> Any | None:
    """Reuse only a complete package bound to the current code and inputs."""

    if not config.reuse_completed_release or not config.writes_outputs:
        return None
    output_root = (
        config.paths["test_output_root"]
        if config.run_mode == "test"
        else config.paths["reports_root"]
    )
    if not output_root.is_dir():
        return None

    integrity = verify_configured_source_integrity(
        config,
        checksum_revision_approvals=(
            source_data_policy.checksum_revision_approvals
            if source_data_policy is not None
            else None
        ),
        designated_reviewers=(
            source_data_policy.designated_reviewers
            if source_data_policy is not None
            else ()
        ),
    )
    source_scope_snapshot = (
        _build_runtime_source_scope_snapshot(
            config,
            source_data_policy,
            integrity,
        )
        if source_data_policy is not None
        else None
    )
    if source_scope_snapshot is not None:
        snapshot_artifacts = source_scope_snapshot.get("manifest_artifact_sha256")
        if not isinstance(snapshot_artifacts, Mapping):
            raise ConfigError(
                "Source-scope snapshot is missing its artifact checksum mapping"
            )
        source_artifact_sha256 = {
            str(path): str(digest)
            for path, digest in sorted(snapshot_artifacts.items())
        }
    else:
        source_artifact_sha256 = dict(sorted(integrity.artifact_sha256.items()))
    analysis_policy_evidence = (
        {
            "status": "validated",
            "manifest_sha256": config.analysis_policy_manifest_sha256,
            "artifact_sha256": dict(analysis_policy.artifact_sha256),
        }
        if analysis_policy is not None
        else {"status": "not_configured"}
    )
    source_policy_evidence = (
        {
            "status": "validated",
            "manifest_sha256": source_data_policy.manifest_authority.sha256,
            "artifact_sha256": dict(source_data_policy.artifact_sha256),
            "source_scope_snapshot_sha256": source_scope_snapshot["snapshot_sha256"],
        }
        if source_data_policy is not None and source_scope_snapshot is not None
        else {"status": "not_configured"}
    )
    expected_identity = {
        "config_sha256": release_config_sha256(config),
        "code_sha256": _code_fingerprint(),
        "policy_content_sha256": policy_snapshot.policy_content_sha256,
        "effective_enablement_sha256": (policy_snapshot.effective_enablement_sha256),
        "mode": config.run_mode,
        "random_seed": config.raw["run"]["random_seed"],
        "scope_countries": list(config.scope_countries),
        "series_identity_dimensions": list(config.series_identity_dimensions),
        "source_artifact_sha256": source_artifact_sha256,
        "source_data_policy": source_policy_evidence,
        "analysis_policy": analysis_policy_evidence,
        "effective_curve_model_policy_sha256": stable_json_sha256(model_policy),
    }
    return _find_reusable_completed_release(
        config,
        expected_identity=expected_identity,
        policy_content_sha256=policy_snapshot.policy_content_sha256,
    )


def _materialize_workspace_views(
    config: ValidatedConfig,
    package: Any,
    *,
    run_log: Any | None = None,
) -> WorkspaceOutputs | None:
    """Project the verified release package into the WF/02-WF/05 roots.

    Full mode only: ``test`` writes its package under the test output root and
    ``validate`` writes nothing at all, so neither one may touch these roots.
    The package itself is read-only here and stays the authoritative deliverable.
    """

    if config.run_mode != "full":
        return None
    return materialize_workspace_views(
        config,
        package,
        view_name=_release_run_id(config),
        run_log=run_log,
    )


def _print_workspace_views(
    config: ValidatedConfig, outputs: WorkspaceOutputs | None
) -> None:
    """Append the view paths to the existing key=value console contract."""

    if outputs is None:
        return
    print(f"workspace_views_reused={'true' if outputs.reused else 'false'}")
    for category, display in workspace_view_display_paths(outputs, config.project_root):
        print(f"workspace_view_{category}={display}")


def _refresh_workspace_outputs(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> int:
    """Re-project the completed full-mode package without rerunning Phases 2-5.

    Loads and validates the configuration exactly as a run does, verifies the
    already-promoted package, and calls the same production function the run
    path uses. It never writes to the release package.
    """

    config, _, _, _, _ = _preflight_runtime(config_path, project_root=project_root)
    if config.run_mode != "full":
        raise ConfigError(
            "--refresh-workspace-outputs requires full mode; "
            f"the configuration selects {config.run_mode!r}"
        )
    target = config.paths["reports_root"] / _release_run_id(config)
    package = verify_completed_release_package(target)
    outputs = _materialize_workspace_views(config, package)
    print(f"mode={config.run_mode}")
    print("workspace_refresh=true")
    print(f"release_package={_relative(package.target_path, config.project_root)}")
    print(f"release_artifacts={len(package.artifact_sha256)}")
    _print_workspace_views(config, outputs)
    if outputs is not None:
        print(f"workspace_view_artifacts={outputs.artifact_count}")
        print(f"workspace_view_bytes={outputs.total_bytes}")
    return 0


def _prepare_run_workspace(
    config_path: str | Path,
    *,
    project_root: str | Path,
    log_root: str | Path,
) -> int:
    """Clear the log root before the launcher opens its log files.

    The launcher runs this between the governance preflight and
    ``nrc_setup_logging``: clearing the log subdirectories from inside the run
    would delete the transcripts the shell helper is writing to. Like the
    preflight, a failure here has no log trail.
    """

    config = load_config(config_path, project_root=project_root, check_files=True)
    cleared = clear_log_workspace(config, log_root=log_root)
    if cleared is not None:
        print(f"log_root_cleared={_display_path(Path(log_root), config.project_root)}")
        print(f"log_entries_removed={len(cleared.removed)}")
    return 0


def run(config_path: str | Path, *, project_root: str | Path) -> int:
    (
        config,
        analysis_policy,
        source_data_policy,
        model_policy,
        policy_snapshot,
    ) = _preflight_runtime(config_path, project_root=project_root)
    run_id = _release_run_id(config)
    run_log = RunLogger(
        level=str(config.raw["logging"]["level"]),
        run_id=run_id,
        stream=sys.stdout,
        error_stream=sys.stderr,
        project_root=config.project_root,
    )
    run_context = {
        "mode": config.run_mode,
        "writes_outputs": config.writes_outputs,
        "config_path": config.config_path,
    }
    launcher_run_id = os.environ.get("N_RESPONSE_LAUNCHER_RUN_ID")
    if launcher_run_id:
        run_context["launcher_run_id"] = launcher_run_id
    run_log.info("run_started", **run_context)
    run_log.info("resource_snapshot", **_resource_snapshot(config, phase="start"))
    if analysis_policy is not None:
        run_log.info(
            "analysis_policy_validated",
            manifest_sha256=config.analysis_policy_manifest_sha256,
            component_sha256=dict(analysis_policy.artifact_sha256),
        )
    if source_data_policy is not None:
        run_log.info(
            "source_data_policy_validated",
            manifest_sha256=source_data_policy.manifest_authority.sha256,
            component_sha256=dict(source_data_policy.artifact_sha256),
        )
    reusable_package = None
    if not config.phases:
        reusable_package = _reuse_completed_release(
            config,
            analysis_policy=analysis_policy,
            source_data_policy=source_data_policy,
            model_policy=model_policy,
            policy_snapshot=policy_snapshot,
        )
        if reusable_package is None:
            raise ConfigError(
                "No reusable completed release matches the current inputs, "
                "configuration, code, and policies; restore the complete "
                "[run].phases list to run Phases 2-5"
            )
    if reusable_package is not None:
        run_log.info(
            "completed_release_reused",
            release_package=reusable_package.target_path,
            skipped_phases="phase_2,phase_3,phase_4,phase_5",
        )
        workspace_outputs = _materialize_workspace_views(
            config, reusable_package, run_log=run_log
        )
        run_log.info("resource_snapshot", **_resource_snapshot(config, phase="end"))
        print(f"mode={config.run_mode}")
        print("writes_outputs=true")
        print(f"status=phase_5_{config.run_mode}_release_complete")
        print("release_reused=true")
        print("phases_executed=none")
        print("phases_skipped=phase_2,phase_3,phase_4,phase_5")
        print(
            "release_package="
            + _relative(reusable_package.target_path, config.project_root)
        )
        _print_workspace_views(config, workspace_outputs)
        return 0
    with run_log.stage("phase_2"):
        phase_two = run_phase_two(
            config,
            source_data_policy=source_data_policy,
        )
        _enforce_phase_two_qc_gate(
            config,
            phase_two,
            review_gate_policy=policy_snapshot.review_gate_policy,
        )
        _enforce_literature_verification_gate(config, phase_two)
    run_log.debug(
        "phase_2_summary",
        canonical_rows=len(phase_two.curation.records),
        eligibility_rows=len(phase_two.eligibility.ledger),
        analysis_eligibility_rows=len(phase_two.analysis_eligibility.ledger),
        critical_records=len(phase_two.qc.critical_record_uids),
        review_records=len(phase_two.qc.review_rows),
    )
    if config.run_mode == "validate":
        _print_validation_plan(config, phase_two)
        run_log.info("resource_snapshot", **_resource_snapshot(config, phase="end"))
        run_log.info("validation_completed", writes_outputs=False)
        return 0
    # Phase 2 writes nothing, so this is the last point before the run's first
    # write: a configuration or QC failure leaves prior packages intact.
    output_clear = clear_output_workspace(config)
    if output_clear is not None:
        run_log.info(
            "output_root_cleared",
            output_root=output_clear.roots[0],
            removed_entries=len(output_clear.removed),
        )
    with run_log.stage("phase_3"):
        phase_three = run_phase_three(
            config,
            phase_two,
            model_policy=model_policy,
        )
    run_log.debug(
        "phase_3_summary",
        model_attempts=len(phase_three.evidence.model_attempts),
        model_reporting_policy=phase_three.evidence.reporting_policy,
        credible_models=len(phase_three.evidence.credible_attempts),
        selected_models=len(phase_three.evidence.selected_attempts),
        curve_rows=len(phase_three.evidence.curve_rows),
        series_evidence_rows=len(phase_three.evidence.series_evidence_rows),
    )
    with run_log.stage("phase_4"):
        phase_four = run_phase_four(
            config,
            phase_two,
            phase_three,
            model_policy=model_policy,
            analysis_policy=analysis_policy,
        )
    run_log.debug(
        "phase_4_summary",
        concrete_candidates=len(phase_four.registry.candidates),
        theoretical_candidates=phase_four.registry.theoretical_candidate_count,
    )
    run_log.info("resource_snapshot", **_resource_snapshot(config, phase="end"))
    with run_log.stage("phase_5"):
        phase_five = release_phases_three_to_five(
            config,
            phase_two,
            phase_three,
            phase_four,
            run_log=run_log,
            policy_snapshot=policy_snapshot,
            analysis_policy=analysis_policy,
        )
    workspace_outputs = _materialize_workspace_views(
        config, phase_five.package, run_log=run_log
    )
    run_log.info(
        "run_completed",
        release_package=phase_five.package.target_path,
        release_reused=phase_five.reused_existing_package,
    )
    print(f"mode={config.run_mode}")
    print("writes_outputs=true")
    print(f"status=phase_5_{config.run_mode}_release_complete")
    print(f"release_reused={'true' if phase_five.reused_existing_package else 'false'}")
    print(
        f"release_package={_relative(phase_five.package.target_path, config.project_root)}"
    )
    print(f"canonical_rows={len(phase_two.curation.records)}")
    print(f"eligibility_rows={len(phase_two.eligibility.ledger)}")
    print(f"analysis_eligibility_rows={len(phase_two.analysis_eligibility.ledger)}")
    print(f"curve_model_attempts={len(phase_three.evidence.model_attempts)}")
    print(f"curve_feature_rows={len(phase_three.evidence.curve_rows)}")
    print(f"series_evidence_rows={len(phase_three.evidence.series_evidence_rows)}")
    print(f"analysis_candidates={phase_four.registry.theoretical_candidate_count}")
    _print_workspace_views(config, workspace_outputs)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and orchestrate the N-response workflow."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--governance-preflight",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--prepare-run-workspace",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--refresh-workspace-outputs",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    try:
        if args.governance_preflight:
            _preflight_runtime(args.config, project_root=project_root)
            return 0
        if args.prepare_run_workspace:
            if args.log_root is None:
                parser.error("--prepare-run-workspace requires --log-root")
            return _prepare_run_workspace(
                args.config,
                project_root=project_root,
                log_root=args.log_root,
            )
        if args.refresh_workspace_outputs:
            return _refresh_workspace_outputs(args.config, project_root=project_root)
        return run(args.config, project_root=project_root)
    except ConfigError as exc:
        if not exception_was_logged(exc):
            print(
                format_console_exception(
                    timestamp=datetime.now(timezone.utc)
                    .isoformat(timespec="milliseconds")
                    .replace("+00:00", "Z"),
                    event="configuration_error",
                    exc=exc,
                    project_root=project_root,
                    color=console_colors_enabled(sys.stderr),
                ),
                file=sys.stderr,
            )
        return 2
    except OSError as exc:
        if not exception_was_logged(exc):
            print(
                format_console_exception(
                    timestamp=datetime.now(timezone.utc)
                    .isoformat(timespec="milliseconds")
                    .replace("+00:00", "Z"),
                    event="runtime_error",
                    exc=exc,
                    project_root=project_root,
                    color=console_colors_enabled(sys.stderr),
                ),
                file=sys.stderr,
            )
        return 2
    except Exception as exc:
        if not exception_was_logged(exc):
            print(
                format_console_exception(
                    timestamp=datetime.now(timezone.utc)
                    .isoformat(timespec="milliseconds")
                    .replace("+00:00", "Z"),
                    event="runtime_error",
                    exc=exc,
                    project_root=project_root,
                    color=console_colors_enabled(sys.stderr),
                ),
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
