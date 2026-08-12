from __future__ import annotations

import argparse
import os
from pathlib import Path
import resource
import sys
from typing import Any

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
    verify_source_integrity,  # noqa: F401  (Phase 1 compatibility re-export)
)
from n_response_curve.data.qc import QcReport, build_qc_report
from n_response_curve.logging.run_logging import RunLogger
from n_response_curve.pipeline.policy_governance import (
    ReviewGatePolicy,
    phase_two_review_disposition,
    validate_policy_authority_bindings,
    validate_runtime_policy,
)
from n_response_curve.pipeline.workflow import (
    build_effective_model_policy,
    release_phases_three_to_five,
    run_phase_four,
    run_phase_three,
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
        raise ConfigError("Source-data policy pseudonym-secret reference is unavailable")
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
                )
                for source_name, source_map in source_data_policy.source_maps.items()
                if (
                    source_name in config.enabled_sources
                    and source_map.workbook_csv_basis
                    == "parallel_workbook_csv_verified_equivalent"
                    and source_map.workbook_sha256 is not None
                    and source_map.csv_sha256 is not None
                    and source_map.workbook_csv_reconciliation_review_id is not None
                )
            },
            designated_reviewers=source_data_policy.designated_reviewers,
        )
    else:
        ingestion = ingest_configured_sources(config)
    if source_data_policy is not None:
        try:
            validate_source_data_policy_coverage(source_data_policy, ingestion)
        except (SourceDataPolicyError, ValueError) as exc:
            raise ConfigError(f"Source-data policy coverage failed: {exc}") from exc
    curation = curate_ingestion(
        ingestion,
        config,
        **(
            dict(source_data_policy.curation_kwargs)
            if source_data_policy is not None
            else {}
        ),
    )
    literature_verification = (
        _plan_literature_verification(source_data_policy, curation.records)
        if source_data_policy is not None
        else None
    )
    resolution_kwargs: dict[str, Any] = (
        dict(source_data_policy.resolution_kwargs)
        if source_data_policy is not None
        else {}
    )
    untrimmed_resolution = resolve_response_series(
        curation.records,
        series_identity_dimensions=config.series_identity_dimensions,
        n_level_tolerance_kg_ha=float(config.raw["eligibility"]["n_level_tolerance_kg_ha"]),
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
            str(record["record_uid"]): record
            for record in final_cleaning.records
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
            str(record["record_uid"]): record
            for record in primary_resolution.records
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
        raise ConfigError("Phase 2 source-to-tier QC flow does not reconcile to the master inventory")
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
    )


def _enforce_phase_two_qc_gate(config: ValidatedConfig, phase_two: PhaseTwoResult) -> None:
    """Fail validate/full after the complete Phase 2 review finds any review state."""

    if config.run_mode not in {"validate", "full"}:
        return
    review_rows = phase_two.qc.review_rows
    if not review_rows:
        return
    affected_uids = sorted(str(row.get("record_uid", "unresolved")) for row in review_rows)
    preview = ", ".join(affected_uids[:20])
    if len(affected_uids) > 20:
        preview += f", ... ({len(affected_uids) - 20} more)"
    raise ConfigError(
        "Phase 2 strict QC gate failed because warning, unresolved, or excluded records require review: "
        + preview
    )


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _print_validation_plan(config: ValidatedConfig, phase_two: PhaseTwoResult) -> None:
    report = phase_two.ingestion.integrity_report
    if report is None:
        raise ConfigError("Phase 2 ingestion completed without a source-integrity report")
    tier_counts = phase_two.qc.tier_counts
    resolved_series_count = len(
        {
            record["response_series_uid"]
            for record in phase_two.resolution.records
            if record.get("series_status") == "resolved" and record.get("response_series_uid")
        }
    )
    print(f"mode={config.run_mode}")
    print(f"writes_outputs={'true' if config.writes_outputs else 'false'}")
    print(f"source_count={len(config.enabled_sources)}")
    for source_name in config.enabled_sources:
        source_path = config.paths["core_source_csv"] if source_name == "core_trial_data" else Path(config.sources[source_name]["data_path"])
        if source_name != "core_trial_data":
            source_path = config.project_root / source_path
        print(f"source={source_name} path={_relative(source_path, config.project_root)}")
    print(f"enabled_models={','.join(config.enabled_models) or 'observed_only'}")
    print(f"scope_countries={','.join(config.scope_countries)}")
    print(f"series_identity_dimensions={','.join(config.series_identity_dimensions)}")
    print(f"n_level_tolerance_kg_ha={config.raw['eligibility']['n_level_tolerance_kg_ha']}")
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
    print("stages=config_validation,source_integrity,position_safe_ingestion,canonical_curation,response_series_resolution,eligibility_qc")


def _preflight_runtime(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> tuple[ValidatedConfig, Any, Any, Any, Any]:
    config = load_config(config_path, project_root=project_root, check_files=True, preflight_engines=True)
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


def run(config_path: str | Path, *, project_root: str | Path) -> int:
    (
        config,
        analysis_policy,
        source_data_policy,
        model_policy,
        policy_snapshot,
    ) = _preflight_runtime(config_path, project_root=project_root)
    run_id = f"n_response_{config.run_mode}_{config.raw['run']['random_seed']}"
    run_log = RunLogger(level=str(config.raw["logging"]["level"]), run_id=run_id)
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
    with run_log.stage("phase_2"):
        phase_two = run_phase_two(
            config,
            source_data_policy=source_data_policy,
        )
        _enforce_phase_two_qc_gate(config, phase_two)
    run_log.debug(
        "phase_2_summary",
        canonical_rows=len(phase_two.curation.records),
        eligibility_rows=len(phase_two.eligibility.ledger),
        analysis_eligibility_rows=len(phase_two.analysis_eligibility.ledger),
        critical_records=len(phase_two.qc.critical_record_uids),
        review_records=len(phase_two.qc.review_rows),
    )
    if config.run_mode == "full" and phase_two.qc.critical_record_uids:
        raise ConfigError(
            "Phase 2 full-mode source gate failed for configured critical records: "
            + ", ".join(phase_two.qc.critical_record_uids)
        )
    if config.run_mode == "validate":
        _print_validation_plan(config, phase_two)
        run_log.info("resource_snapshot", **_resource_snapshot(config, phase="end"))
        run_log.info("validation_completed", writes_outputs=False)
        return 0
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
    phase_five = release_phases_three_to_five(
        config,
        phase_two,
        phase_three,
        phase_four,
        run_log=run_log,
        policy_snapshot=policy_snapshot,
        analysis_policy=analysis_policy,
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
    print(f"release_package={_relative(phase_five.package.target_path, config.project_root)}")
    print(f"canonical_rows={len(phase_two.curation.records)}")
    print(f"eligibility_rows={len(phase_two.eligibility.ledger)}")
    print(f"analysis_eligibility_rows={len(phase_two.analysis_eligibility.ledger)}")
    print(f"curve_model_attempts={len(phase_three.evidence.model_attempts)}")
    print(f"curve_feature_rows={len(phase_three.evidence.curve_rows)}")
    print(f"series_evidence_rows={len(phase_three.evidence.series_evidence_rows)}")
    print(f"analysis_candidates={phase_four.registry.theoretical_candidate_count}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and orchestrate the N-response workflow.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--governance-preflight",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    try:
        if args.governance_preflight:
            _preflight_runtime(args.config, project_root=project_root)
            return 0
        return run(args.config, project_root=project_root)
    except ConfigError as exc:
        print(f"configuration-error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"runtime-error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
