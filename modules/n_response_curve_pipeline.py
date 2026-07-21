from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dataclasses import dataclass

from n_response_curve.config import ConfigError, ValidatedConfig, load_config
from n_response_curve.curate import CurationResult, curate_ingestion
from n_response_curve.duplicates import SeriesResolution, resolve_response_series
from n_response_curve.eligibility import EligibilityResult, assign_eligibility
from n_response_curve.ingest import IngestionResult, ingest_configured_sources
from n_response_curve.provenance import SourceIntegrityReport, verify_source_integrity  # noqa: F401  (Phase 1 compatibility re-export)
from n_response_curve.qc import QcReport, build_qc_report
from n_response_curve.workflow import release_phases_three_to_five, run_phase_four, run_phase_three


@dataclass(frozen=True)
class PhaseTwoResult:
    ingestion: IngestionResult
    curation: CurationResult
    resolution: SeriesResolution
    eligibility: EligibilityResult
    qc: QcReport


def run_phase_two(config: ValidatedConfig) -> PhaseTwoResult:
    """Execute the non-writing Phase 2 source-to-analysis-ready data gate."""

    ingestion = ingest_configured_sources(config)
    curation = curate_ingestion(ingestion, config)
    resolution = resolve_response_series(
        curation.records,
        context_dimensions=config.comparison_dimensions,
    )
    eligibility = assign_eligibility(
        resolution.records,
        policy=config.raw["eligibility"],
    )
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
        qc=qc,
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
    print(f"comparison_dimensions={','.join(config.comparison_dimensions) or 'none'}")
    print(f"analysis_families={','.join(config.analysis_families)}")
    print(f"source_integrity=pass checked_files={report.checked_files}")
    print(f"canonical_rows={len(phase_two.curation.records)}")
    print(f"blank_source_rows={len(phase_two.ingestion.blank_rows)}")
    print(f"resolved_response_series={resolved_series_count}")
    print(f"eligibility_rows={len(phase_two.eligibility.ledger)}")
    print(f"tier_A={tier_counts['A']}")
    print(f"tier_B={tier_counts['B']}")
    print(f"tier_C={tier_counts['C']}")
    print(f"tier_D={tier_counts['D']}")
    print(f"critical_records={len(phase_two.qc.critical_record_uids)}")
    print(f"qc_reconciles={'true' if phase_two.qc.reconciles else 'false'}")
    print("stages=config_validation,source_integrity,position_safe_ingestion,canonical_curation,response_series_resolution,eligibility_qc")


def run(config_path: str | Path, *, project_root: str | Path) -> int:
    config = load_config(config_path, project_root=project_root, check_files=True, preflight_engines=True)
    phase_two = run_phase_two(config)
    if config.run_mode == "full" and phase_two.qc.critical_record_uids:
        raise ConfigError(
            "Phase 2 full-mode source gate failed for configured critical records: "
            + ", ".join(phase_two.qc.critical_record_uids)
        )
    if config.run_mode == "validate":
        _print_validation_plan(config, phase_two)
        return 0
    phase_three = run_phase_three(config, phase_two)
    phase_four = run_phase_four(config, phase_two, phase_three)
    phase_five = release_phases_three_to_five(config, phase_two, phase_three, phase_four)
    print(f"mode={config.run_mode}")
    print("writes_outputs=true")
    print(f"status=phase_5_{config.run_mode}_release_complete")
    print(f"release_reused={'true' if phase_five.reused_existing_package else 'false'}")
    print(f"release_package={_relative(phase_five.package.target_path, config.project_root)}")
    print(f"canonical_rows={len(phase_two.curation.records)}")
    print(f"eligibility_rows={len(phase_two.eligibility.ledger)}")
    print(f"curve_model_attempts={len(phase_three.evidence.model_attempts)}")
    print(f"curve_feature_rows={len(phase_three.evidence.curve_rows)}")
    print(f"analysis_candidates={phase_four.registry.theoretical_candidate_count}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and orchestrate the N-response workflow.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    try:
        return run(args.config, project_root=project_root)
    except ConfigError as exc:
        print(f"configuration-error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"runtime-error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
