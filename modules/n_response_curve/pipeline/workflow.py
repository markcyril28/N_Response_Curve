from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
from statistics import median
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

from n_response_curve.analysis.analysis_matrix import AnalysisRegistry, build_analysis_registry, build_source_combinations
from n_response_curve.analysis.comparisons import ManagementSystemProximity, build_management_system_proximity
from n_response_curve.data.config import ConfigError, ValidatedConfig
from n_response_curve.data.provenance import sha256_file, stable_json_sha256
from n_response_curve.analysis.curve_evidence import CurveEvidenceResult, build_curve_evidence, curve_fit_record_uids
from n_response_curve.analysis.curve_views import DerivedCurveView, build_derived_curve_views
from n_response_curve.analysis.dataset_versions import DatasetVersion, build_dataset_versions
from n_response_curve.analysis.explanatory import PythonAnalysisResult, execute_python_candidates, select_candidate_curve_rows
from n_response_curve.analysis.factor_catalog import FactorCatalogEntry, build_factor_catalog
from n_response_curve.reporting.plots import write_observed_series_figures, write_response_curve_figures
from n_response_curve.analysis.r_bridge import RBridgeError, invoke_r_stage, write_r_stage_contract
from n_response_curve.analysis.r_specs import RAnalysisPreparation, prepare_r_analysis
from n_response_curve.reporting.release import (
    ReleasePackage,
    ReportingError,
    TableArtifact,
    verify_release_package,
    write_release_package,
)
from n_response_curve.logging.run_logging import RunLogger


@dataclass(frozen=True)
class PhaseThreeResult:
    evidence: CurveEvidenceResult
    input_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PhaseFourResult:
    dataset_versions: tuple[DatasetVersion, ...]
    derived_curve_views: tuple[DerivedCurveView, ...]
    curve_rows: tuple[dict[str, Any], ...]
    factor_catalog: tuple[FactorCatalogEntry, ...]
    registry: AnalysisRegistry
    python_results: tuple[PythonAnalysisResult, ...]
    r_preparations: tuple[tuple[str, RAnalysisPreparation], ...]
    management_system_proximity: tuple[ManagementSystemProximity, ...]


@dataclass(frozen=True)
class PhaseFiveResult:
    package: ReleasePackage
    reused_existing_package: bool


_INFERENTIAL_ANALYSIS_FAMILIES = frozenset(
    {
        "all_supported_interactions",
        "marginal_contrasts",
        "multivariable_mixed_effects",
        "observation_level_curve_modification",
        "one_factor_inferential",
    }
)


def _code_fingerprint() -> str:
    module_root = Path(__file__).resolve().parents[1]
    project_root = module_root.parent
    paths = [
        path
        for path in sorted(module_root.rglob("*"))
        if path.is_file()
        and path.suffix in {".py", ".R"}
        and "__pycache__" not in path.parts
    ]
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(project_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _redact(value: Any, *, key: str = "") -> Any:
    lowered = key.casefold()
    if any(fragment in lowered for fragment in ("password", "secret", "token", "credential", "api_key", "apikey")):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(nested_key): _redact(nested, key=str(nested_key)) for nested_key, nested in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_redact(item, key=key) for item in value]
    return value


def _runtime_inventory(rscript_command: str) -> dict[str, Any]:
    packages = ("matplotlib", "numpy", "pandas", "pyarrow", "scikit-learn", "scipy")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    r_executable = shutil.which(str(rscript_command))
    if r_executable is None:
        raise ConfigError(f"Unable to inventory unavailable Rscript command: {rscript_command}")
    r_packages = (
        "arrow",
        "broom",
        "emmeans",
        "glmmTMB",
        "jsonlite",
        "lme4",
        "lmerTest",
        "nnet",
        "performance",
        "reformulas",
        "TMB",
        "testthat",
    )
    package_vector = ",".join(json.dumps(package) for package in r_packages)
    r_expression = (
        f"packages <- c({package_vector}); "
        "installed <- installed.packages()[, 'Version']; "
        "versions <- lapply(packages, function(package) "
        "if (package %in% names(installed)) unname(installed[[package]]) else NA_character_); "
        "names(versions) <- packages; "
        "cat(jsonlite::toJSON(list(version = R.version.string, "
        "executable = normalizePath(Sys.which('Rscript'), winslash = '/', mustWork = TRUE), "
        "packages = versions), auto_unbox = TRUE, null = 'null'))"
    )
    try:
        completed = subprocess.run(
            [r_executable, "--vanilla", "-e", r_expression],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        r_inventory = json.loads(completed.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise ConfigError(f"Unable to capture the R runtime inventory: {exc}") from exc
    if completed.returncode != 0 or not isinstance(r_inventory, Mapping):
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise ConfigError(f"Unable to capture the R runtime inventory: {detail}")
    return {
        "platform": platform.platform(),
        "python": {
            "version": sys.version.split()[0],
            "executable": str(Path(sys.executable).resolve()),
            "packages": versions,
        },
        "r": dict(r_inventory),
    }


def _git_inventory(project_root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    commit = git("rev-parse", "HEAD")
    if commit.returncode != 0:
        return {
            "status": "unavailable",
            "reason": (commit.stderr or commit.stdout).strip() or "not a Git work tree",
        }
    branch = git("branch", "--show-current")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if branch.returncode != 0 or status.returncode != 0:
        raise ConfigError("Git provenance inventory failed for the controlled release")
    changed_entries = tuple(line.rstrip() for line in status.stdout.splitlines() if line.strip())
    return {
        "status": "available",
        "commit": commit.stdout.strip(),
        "branch": branch.stdout.strip() or "DETACHED",
        "dirty": bool(changed_entries),
        "changed_entry_count": len(changed_entries),
        "changed_entries": list(changed_entries),
    }


def _model_input_records(config: ValidatedConfig, ledger: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    records = tuple(dict(record) for record in ledger)
    if config.run_mode != "test":
        return records
    resolved_series = sorted(
        {
            str(record["response_series_uid"])
            for record in records
            if record.get("series_status") == "resolved" and record.get("response_series_uid")
        }
    )
    permitted = set(resolved_series[: int(config.raw["run"]["test_group_limit"])])
    return tuple(record for record in records if str(record.get("response_series_uid", "")) in permitted)


def run_phase_three(config: ValidatedConfig, phase_two: Any) -> PhaseThreeResult:
    """Fit only configured curve candidates and preserve every attempt as evidence."""

    input_records = _model_input_records(config, phase_two.eligibility.ledger)
    evidence = build_curve_evidence(
        input_records,
        model_names=config.enabled_models,
        policy=config.raw["modeling"],
        fit_record_uids=curve_fit_record_uids(input_records, primary_only=True),
    )
    return PhaseThreeResult(evidence=evidence, input_records=input_records)


def run_phase_four(config: ValidatedConfig, phase_two: Any, phase_three: PhaseThreeResult) -> PhaseFourResult:
    """Build immutable analysis views, factor coverage, and an exhaustive dispatch ledger."""

    versions = build_dataset_versions(
        phase_three.input_records,
        version_names=config.dataset_versions,
    )
    source_combinations = build_source_combinations(
        config.enabled_sources,
        modes=config.source_combination_modes,
    )
    derived_curve_views = build_derived_curve_views(
        phase_three.input_records,
        dataset_versions=versions,
        source_combinations=source_combinations,
        model_names=config.enabled_models,
        policy=config.raw["modeling"],
    )
    curve_rows = tuple(row for view in derived_curve_views for row in view.curve_rows)
    factor_catalog = build_factor_catalog(
        phase_three.evidence.curve_rows,
        factor_names=config.explanatory_factors,
    )
    support_policy = config.raw["analysis_matrix"].get("support_policy")
    registry = build_analysis_registry(
        curve_rows,
        dataset_versions=versions,
        source_combination_modes=config.source_combination_modes,
        curve_outcomes=config.curve_outcomes,
        factor_catalog=factor_catalog,
        analysis_families=config.analysis_families,
        engine_assignments=config.engine_assignments,
        interaction_orders=config.interaction_orders,
        support_policy=support_policy,
        source_families=config.enabled_sources,
    )
    python_results = execute_python_candidates(
        registry.candidates,
        curve_rows=curve_rows,
        dataset_versions=versions,
        factor_catalog=factor_catalog,
        eligibility_records=phase_three.input_records,
    )
    r_preparations = _prepare_r_candidates(
        registry,
        versions,
        input_records=phase_three.input_records,
        curve_rows=curve_rows,
    )
    if not registry.reconciles:
        raise ConfigError("Phase 4 analysis registry does not reconcile its configured candidate space")
    primary_dataset_version = next(
        (
            version
            for version in versions
            if version.version_id == "D02_strict_primary_zero_optional"
        ),
        None,
    )
    management_system_proximity = build_management_system_proximity(
        phase_three.input_records,
        curve_rows=phase_three.evidence.curve_rows,
        dataset_version_id=(
            primary_dataset_version.version_id
            if primary_dataset_version is not None
            else None
        ),
        dataset_version_status=(
            primary_dataset_version.status
            if primary_dataset_version is not None
            else "not_configured"
        ),
        dataset_membership_sha256=(
            primary_dataset_version.membership_sha256
            if primary_dataset_version is not None
            else None
        ),
        dataset_record_uids=(
            primary_dataset_version.record_uids
            if primary_dataset_version is not None
            else ()
        ),
    )
    return PhaseFourResult(
        dataset_versions=versions,
        derived_curve_views=derived_curve_views,
        curve_rows=curve_rows,
        factor_catalog=factor_catalog,
        registry=registry,
        python_results=python_results,
        r_preparations=r_preparations,
        management_system_proximity=management_system_proximity,
    )


def _r_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(value, (tuple, list, set, frozenset)):
        return json.dumps(list(value), ensure_ascii=False, default=str)
    return value


def _r_input_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            str(key): _r_value(value)
            for key, value in row.items()
            if key != "model_attempt_record"
        }
        for row in rows
    )


def _select_candidate_observation_rows(
    candidate: Any,
    records: Iterable[Mapping[str, Any]],
    versions: Mapping[str, DatasetVersion],
) -> tuple[dict[str, Any], ...]:
    version = versions.get(candidate.dataset_version_id)
    if version is None or version.status != "available":
        return ()
    membership = set(version.record_uids)
    selected = [
        dict(record)
        for record in records
        if str(record.get("record_uid") or "") in membership
        and str(record.get("source_name") or "") in candidate.source_families
    ]
    return tuple(sorted(selected, key=lambda row: str(row.get("record_uid") or "")))


def _prepare_r_candidates(
    registry: AnalysisRegistry,
    versions: Sequence[DatasetVersion],
    *,
    input_records: Sequence[Mapping[str, Any]],
    curve_rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, RAnalysisPreparation], ...]:
    versions_by_id = {version.version_id: version for version in versions}
    preparations: list[tuple[str, RAnalysisPreparation]] = []
    for candidate in registry.candidates:
        if candidate.engine != "r" or candidate.status != "run":
            continue
        observation_level = candidate.analysis_family == "observation_level_curve_modification"
        selected_rows = (
            _select_candidate_observation_rows(candidate, input_records, versions_by_id)
            if observation_level
            else select_candidate_curve_rows(candidate, curve_rows, versions_by_id)
        )
        preparations.append(
            (
                candidate.candidate_id,
                prepare_r_analysis(
                    candidate=candidate,
                    rows=selected_rows,
                    observation_level=observation_level,
                ),
            )
        )
    return tuple(preparations)


def _r_stage_writer(
    config: ValidatedConfig,
    phase_three: PhaseThreeResult,
    phase_four: PhaseFourResult,
    statuses: list[dict[str, Any]],
    result_rows: dict[str, tuple[dict[str, Any], ...]],
):
    preparations = dict(phase_four.r_preparations)
    if len(preparations) != len(phase_four.r_preparations):
        raise ConfigError("Phase 4 R preparations contain duplicate candidate identifiers")
    entrypoint_value = config.raw["engines"]["r_entrypoint"]
    entrypoint = (config.project_root / str(entrypoint_value)).resolve()
    rscript_command = config.raw["engines"]["rscript_command"]
    timeout_seconds = int(config.raw["run"]["r_stage_timeout_seconds"])
    fail_fast = bool(config.raw["run"]["fail_fast"])
    active_candidates = tuple(
        candidate
        for candidate in phase_four.registry.candidates
        if candidate.engine == "r" and candidate.status == "run"
    )

    def write_r_stages(stage_root: Path) -> tuple[Path, ...]:
        produced: list[Path] = []
        r_root = stage_root / "r_stages"
        for candidate in active_candidates:
            prepared = preparations.get(candidate.candidate_id)
            if prepared is None:
                raise ConfigError(f"R candidate lacks its registry-only preparation: {candidate.candidate_id}")
            if prepared.status != "run":
                statuses.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "status": "skipped",
                        "reason_codes": list(prepared.reason_codes),
                    }
                )
                continue
            contract_root = r_root / candidate.candidate_id
            try:
                contract = write_r_stage_contract(
                    contract_root,
                    specification=prepared.specification,
                    rows=_r_input_rows(prepared.rows),
                    stable_key=prepared.stable_key,
                )
                result = invoke_r_stage(
                    contract,
                    rscript_command=rscript_command,
                    r_entrypoint=entrypoint,
                    timeout_seconds=timeout_seconds,
                    extra_environment={"N_RESPONSE_R_STAGE": candidate.candidate_id},
                )
            except RBridgeError as exc:
                result = None
                statuses.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "status": "failed",
                        "reason_codes": ["R_BRIDGE_ERROR"],
                        "error": str(exc),
                    }
                )
                if fail_fast:
                    raise ConfigError(f"R stage contract failed for {candidate.candidate_id}: {exc}") from exc
            else:
                result_rows[candidate.candidate_id] = tuple(dict(row) for row in result.results)
                structured_reasons = sorted(
                    {
                        str(reason)
                        for row in result.results
                        for reason in (
                            (row.get("reason_codes"),)
                            if isinstance(row.get("reason_codes"), str)
                            else row.get("reason_codes", ())
                        )
                        if str(reason)
                    }
                )
                statuses.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "status": result.status,
                        "return_code": result.return_code,
                        "reason_codes": structured_reasons,
                        "result_count": len(result.results),
                        "metadata": dict(result.metadata),
                        "contract_version": 1,
                        "contract_sha256": contract.contract_sha256,
                        "input_sha256": contract.input_sha256,
                    }
                )
                if result.status == "failed" and fail_fast:
                    raise ConfigError(f"R stage failed for {candidate.candidate_id}: return code {result.return_code}")
                produced.extend(path for path in (contract.contract_path, contract.input_path, contract.output_path) if path.is_file())
        if active_candidates:
            status_path = r_root / "r_stage_statuses.json"
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(json.dumps(statuses, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            produced.append(status_path)
        return tuple(produced)

    return write_r_stages


def _nested_warning_messages(value: Any) -> tuple[str, ...]:
    messages: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() == "warnings" and isinstance(nested, (list, tuple)):
                messages.extend(str(message).strip() for message in nested if str(message).strip())
            else:
                messages.extend(_nested_warning_messages(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            messages.extend(_nested_warning_messages(nested))
    return tuple(messages)


def _raw_probability(row: Mapping[str, Any]) -> tuple[float | None, str | None]:
    values: list[float] = []
    for key in ("p.value_raw", "raw_p_value", "p_value_raw", "p.value", "p_value"):
        if key not in row or row[key] is None or isinstance(row[key], bool):
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            return None, "RAW_P_VALUE_NONFINITE"
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            return None, "RAW_P_VALUE_NONFINITE"
        values.append(value)
    if not values:
        return None, "RAW_P_VALUE_MISSING"
    if any(not math.isclose(value, values[0], rel_tol=1.0e-12, abs_tol=1.0e-15) for value in values[1:]):
        return None, "RAW_P_VALUE_CONFLICT"
    return values[0], None


def _multiplicity_result_id(candidate_id: str, row: Mapping[str, Any]) -> str:
    declared = row.get("result_id")
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    identity = {
        str(key): value
        for key, value in row.items()
        if str(key) not in {"p.value_adjusted", "adjusted_p_value", "p_adjusted", "result_id"}
    }
    encoded = json.dumps(
        {"candidate_id": candidate_id, "result": identity},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "result_" + hashlib.sha256(encoded).hexdigest()


def _multiplicity_reconciliation_id(
    family_id: str,
    candidate_id: str,
    result_id: str,
    occurrence: int,
) -> str:
    encoded = json.dumps(
        [family_id, candidate_id, result_id, occurrence],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "multiplicity_" + hashlib.sha256(encoded).hexdigest()


def _benjamini_hochberg(
    rows: Sequence[tuple[str, str, str, float]],
) -> dict[str, float]:
    ordered = sorted(rows, key=lambda item: (item[3], item[2], item[1], item[0]))
    family_size = len(ordered)
    adjusted: dict[str, float] = {}
    running_minimum = 1.0
    for reverse_index in range(family_size - 1, -1, -1):
        reconciliation_id, _, _, raw_p_value = ordered[reverse_index]
        rank = reverse_index + 1
        running_minimum = min(running_minimum, raw_p_value * family_size / rank)
        adjusted[reconciliation_id] = min(1.0, max(0.0, running_minimum))
    return adjusted


def _reconcile_multiplicity_families(
    registry: AnalysisRegistry,
    r_statuses: Sequence[Mapping[str, Any]],
    r_result_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Reconcile and adjust complete prespecified families without mutating engine outputs."""

    candidates_by_id = {candidate.candidate_id: candidate for candidate in registry.candidates}
    statuses_by_candidate: dict[str, list[Mapping[str, Any]]] = {}
    for status in r_statuses:
        statuses_by_candidate.setdefault(str(status.get("candidate_id") or ""), []).append(status)
    result_rows_by_key: dict[str, dict[str, Any]] = {}
    candidate_statuses: dict[str, dict[str, Any]] = {}
    families: list[dict[str, Any]] = []
    for family in registry.multiplicity_families:
        family_reasons: set[str] = set()
        collected: list[dict[str, Any]] = []
        seen_result_ids: set[str] = set()
        expected_candidate_ids = tuple(family.candidate_ids)
        for candidate_id in expected_candidate_ids:
            candidate = candidates_by_id.get(candidate_id)
            if candidate is None or candidate.multiplicity_family_id != family.family_id:
                family_reasons.add("MULTIPLICITY_REGISTRY_MEMBERSHIP_MISMATCH")
                continue
            if candidate.engine != "r":
                family_reasons.add("MULTIPLICITY_ENGINE_NOT_RECONCILABLE")
                continue
            if candidate.status != "run":
                family_reasons.add("MULTIPLICITY_REGISTRY_MEMBER_NOT_EXECUTABLE")
                continue
            candidate_status_rows = statuses_by_candidate.get(candidate_id, [])
            if not candidate_status_rows:
                family_reasons.add("MULTIPLICITY_CANDIDATE_RESULT_MISSING")
                continue
            if len(candidate_status_rows) != 1:
                family_reasons.add("MULTIPLICITY_CANDIDATE_RESULT_DUPLICATE")
                continue
            if candidate_status_rows[0].get("status") != "completed":
                family_reasons.add("MULTIPLICITY_CANDIDATE_NOT_COMPLETED")
                continue
            candidate_results = r_result_rows.get(candidate_id)
            if not candidate_results:
                family_reasons.add("MULTIPLICITY_RESULT_ROWS_MISSING")
                continue
            for occurrence, raw_row in enumerate(candidate_results):
                row_reasons: set[str] = set()
                if not isinstance(raw_row, Mapping):
                    result_id = f"invalid_result_{occurrence:06d}"
                    raw_p_value = None
                    row_reasons.add("MULTIPLICITY_RESULT_ROW_INVALID")
                else:
                    try:
                        result_id = _multiplicity_result_id(candidate_id, raw_row)
                    except (TypeError, ValueError):
                        result_id = f"invalid_result_{occurrence:06d}"
                        row_reasons.add("MULTIPLICITY_RESULT_ID_INVALID")
                    raw_p_value, raw_reason = _raw_probability(raw_row)
                    if raw_reason is not None:
                        row_reasons.add(raw_reason)
                if result_id in seen_result_ids:
                    row_reasons.add("MULTIPLICITY_RESULT_ID_DUPLICATE")
                seen_result_ids.add(result_id)
                reconciliation_id = _multiplicity_reconciliation_id(
                    family.family_id,
                    candidate_id,
                    result_id,
                    occurrence,
                )
                collected.append(
                    {
                        "reconciliation_id": reconciliation_id,
                        "family_id": family.family_id,
                        "candidate_id": candidate_id,
                        "hypothesis_id": candidate.hypothesis_id,
                        "result_id": result_id,
                        "raw_p_value": raw_p_value,
                        "adjusted_p_value": None,
                        "method": family.method,
                        "family_size": None,
                        "family_complete": False,
                        "status": "not_interpretable",
                        "reason_codes": sorted(row_reasons),
                    }
                )
                family_reasons.update(row_reasons)
        if not collected:
            family_reasons.add("MULTIPLICITY_FAMILY_HAS_NO_RAW_RESULTS")
        family_complete = not family_reasons
        family_size = len(collected) if family_complete else None
        if family_complete:
            adjustments = _benjamini_hochberg(
                tuple(
                    (
                        str(row["reconciliation_id"]),
                        str(row["candidate_id"]),
                        str(row["result_id"]),
                        float(row["raw_p_value"]),
                    )
                    for row in collected
                )
            )
            for row in collected:
                row["adjusted_p_value"] = adjustments[str(row["reconciliation_id"])]
                row["family_size"] = family_size
                row["family_complete"] = True
                row["status"] = "reconciled"
        else:
            for row in collected:
                row["reason_codes"] = sorted(set(row["reason_codes"]) | family_reasons)
        for row in collected:
            result_rows_by_key[str(row["reconciliation_id"])] = row
        family_status = "reconciled" if family_complete else "incomplete"
        for candidate_id in expected_candidate_ids:
            candidate_statuses[candidate_id] = {
                "family_id": family.family_id,
                "status": family_status,
                "reason_codes": [] if family_complete else sorted(family_reasons),
            }
        family_key = hashlib.sha256(
            json.dumps(
                [family.family_id, family.method, list(expected_candidate_ids)],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        families.append(
            {
                "family_key": "family_" + family_key,
                "family_id": family.family_id,
                "method": family.method,
                "status": family_status,
                "complete": family_complete,
                "reason_codes": [] if family_complete else sorted(family_reasons),
                "expected_candidate_ids": list(expected_candidate_ids),
                "expected_hypothesis_ids": list(family.hypothesis_ids),
                "expected_candidate_count": len(expected_candidate_ids),
                "observed_result_count": len(collected),
                "family_size": family_size,
            }
        )
    status_counts = Counter(str(family["status"]) for family in families)
    overall_status = (
        "not_applicable"
        if not families
        else "reconciled"
        if status_counts.get("incomplete", 0) == 0
        else "incomplete"
    )
    return {
        "schema_version": 1,
        "method": "BH",
        "status": overall_status,
        "family_count": len(families),
        "family_status_counts": dict(sorted(status_counts.items())),
        "families": families,
        "candidate_statuses": dict(sorted(candidate_statuses.items())),
        "results_by_reconciliation_id": dict(sorted(result_rows_by_key.items())),
    }


def _multiplicity_reconciliation_stage_writer(
    phase_four: PhaseFourResult,
    r_statuses: list[dict[str, Any]],
    r_result_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    reconciliation_state: dict[str, Any],
    manifest: dict[str, Any],
    report_sections: dict[str, list[str]],
):
    def write_multiplicity_reconciliation(stage_root: Path) -> tuple[Path, ...]:
        payload = _reconcile_multiplicity_families(
            phase_four.registry,
            r_statuses,
            r_result_rows,
        )
        reconciliation_state.clear()
        reconciliation_state.update(payload)
        manifest["multiplicity_reconciliation"] = {
            "artifact_path": "multiplicity_reconciliation.json",
            "method": payload["method"],
            "status": payload["status"],
            "family_count": payload["family_count"],
            "family_status_counts": payload["family_status_counts"],
            "result_count": len(payload["results_by_reconciliation_id"]),
        }
        report_sections["unsupported"].append(
            "Prespecified multiplicity reconciliation status="
            f"{payload['status']} across {payload['family_count']} BH families "
            f"({_format_counts(payload['family_status_counts']) or 'no declared families'}). "
            "Incomplete families remain explicit non-findings; these statuses do not establish causal effects."
        )
        path = stage_root / "multiplicity_reconciliation.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return (path,)

    return write_multiplicity_reconciliation


def _terminal_status_stage_writer(
    phase_three: PhaseThreeResult,
    phase_four: PhaseFourResult,
    r_statuses: list[dict[str, Any]],
    manifest: dict[str, Any],
    report_sections: dict[str, list[str]],
):
    python_by_candidate = {result.candidate_id: result for result in phase_four.python_results}

    def write_terminal_statuses(stage_root: Path) -> tuple[Path, ...]:
        r_by_candidate = {str(status["candidate_id"]): status for status in r_statuses}
        if len(r_by_candidate) != len(r_statuses):
            raise ReportingError("R stage status ledger contains duplicate candidate identifiers")
        status_rows: list[dict[str, Any]] = []
        status_counts: dict[str, int] = {}
        for candidate in phase_four.registry.candidates:
            reason_codes: list[str]
            if candidate.status != "run":
                terminal_status = candidate.status
                reason_codes = list(candidate.reason_codes)
            elif candidate.engine == "python":
                result = python_by_candidate.get(candidate.candidate_id)
                if result is None:
                    raise ReportingError(f"Python candidate lacks a terminal result: {candidate.candidate_id}")
                terminal_status = "run" if result.status == "completed" else result.status
                reason_codes = list(result.reason_codes)
            elif candidate.engine == "r":
                result = r_by_candidate.get(candidate.candidate_id)
                if result is None:
                    raise ReportingError(f"R candidate lacks a terminal result: {candidate.candidate_id}")
                diagnostics = result.get("metadata", {}).get("diagnostics", {})
                if result.get("status") == "completed" and diagnostics.get("converged") is False:
                    terminal_status = "nonconverged"
                elif result.get("status") == "completed" and (
                    diagnostics.get("singular") is True or diagnostics.get("boundary_fit") is True
                ):
                    terminal_status = "not_interpretable"
                else:
                    terminal_status = "run" if result.get("status") == "completed" else str(result.get("status"))
                reason_codes = [str(reason) for reason in result.get("reason_codes", ())]
            else:
                raise ReportingError(f"Candidate has an unknown terminal engine: {candidate.engine}")
            if terminal_status not in {"run", "skipped", "failed", "nonconverged", "not_interpretable", "pruned"}:
                raise ReportingError(
                    f"Candidate {candidate.candidate_id} has an unsupported terminal status: {terminal_status}"
                )
            status_counts[terminal_status] = status_counts.get(terminal_status, 0) + 1
            status_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "analysis_family": candidate.analysis_family,
                    "engine": candidate.engine,
                    "terminal_status": terminal_status,
                    "reason_codes": reason_codes,
                }
            )
        compressed_pruned_count = next(
            (
                item.candidate_count
                for item in phase_four.registry.pruned_families
                if item.reason_code == "FACTOR_PROFILE_PRUNED_BEFORE_EXPANSION"
            ),
            0,
        )
        status_counts["pruned"] = status_counts.get("pruned", 0) + compressed_pruned_count
        terminal_accounted = sum(status_counts.values())
        terminal_reconciles = (
            phase_four.registry.reconciles
            and terminal_accounted == phase_four.registry.accounted_candidate_count
            and terminal_accounted == phase_four.registry.theoretical_candidate_count
        )
        if not terminal_reconciles:
            raise ReportingError("Analysis terminal statuses do not reconcile to the theoretical candidate space")

        contract_inventory = [
            {
                "candidate_id": status["candidate_id"],
                "contract_version": status["contract_version"],
                "contract_sha256": status["contract_sha256"],
                "input_sha256": status["input_sha256"],
            }
            for status in r_statuses
            if "contract_sha256" in status
        ]
        warnings = sorted({message for status in r_statuses for message in _nested_warning_messages(status)})
        model_failures = [
            {
                "model_attempt_uid": attempt.model_attempt_uid,
                "response_series_uid": attempt.response_series_uid,
                "model_name": attempt.model_name,
                "status": attempt.status,
                "reason_codes": list(attempt.reason_codes),
            }
            for attempt in phase_three.evidence.model_attempts
            if attempt.status != "fitted"
        ]
        model_failures.extend(
            {
                "candidate_id": status["candidate_id"],
                "status": "failed",
                "reason_codes": list(status.get("reason_codes", ())),
                "error": status.get("error"),
            }
            for status in r_statuses
            if status.get("status") == "failed"
        )
        manifest["analysis_registry"].update(
            {
                "concrete_candidate_count": len(status_rows),
                "compressed_pruned_candidate_count": compressed_pruned_count,
                "terminal_accounted_candidate_count": terminal_accounted,
                "terminal_status_counts": dict(sorted(status_counts.items())),
                "terminal_reconciles": terminal_reconciles,
            }
        )
        manifest["warnings"] = warnings
        manifest["model_failures"] = model_failures
        manifest["contract_inventory"] = contract_inventory

        supported_factors = sum(
            entry.coverage_count > 0 and entry.cardinality > 1 and not entry.leakage_restricted
            for entry in phase_four.factor_catalog
        )
        available_versions = sum(version.status == "available" for version in phase_four.dataset_versions)
        source_robustness_runs = sum(
            row["analysis_family"] == "dataset_and_source_robustness" and row["terminal_status"] == "run"
            for row in status_rows
        )
        predictive_runs = sum(
            row["analysis_family"] == "penalized_predictive_models" and row["terminal_status"] == "run"
            for row in status_rows
        )
        report_sections["sensitivity"].append(
            f"Factor evidence covered {supported_factors} supported non-leakage factors across "
            f"{available_versions} available dataset versions; {source_robustness_runs} source/dataset robustness analyses completed."
        )
        report_sections["sensitivity"].append(
            f"Model robustness retained {len(phase_three.evidence.model_attempts)} curve-model attempts and "
            f"{len(phase_three.evidence.selected_attempts)} selected models; nonselected or failed attempts remain explicit."
        )
        report_sections["predictive"].append(
            f"{predictive_runs} study-grouped predictive candidates completed; predictive results are not causal estimates."
        )
        report_sections["unsupported"].append(
            f"Terminal analysis reconciliation accounted for {terminal_accounted} of "
            f"{phase_four.registry.theoretical_candidate_count} theoretical candidates."
        )

        payload = {
            "concrete_candidates": status_rows,
            "compressed_pruned_families": [asdict(item) for item in phase_four.registry.pruned_families],
            "terminal_status_counts": dict(sorted(status_counts.items())),
            "terminal_accounted_candidate_count": terminal_accounted,
            "theoretical_candidate_count": phase_four.registry.theoretical_candidate_count,
            "reconciles": terminal_reconciles,
        }
        status_path = stage_root / "analysis_terminal_statuses.json"
        status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return (status_path,)

    return write_terminal_statuses


def _figure_stage_writer(config: ValidatedConfig, phase_three: PhaseThreeResult):
    selected_curve_series = {row["response_series_uid"] for row in phase_three.evidence.curve_rows}

    def write_figures(stage_root: Path) -> tuple[Path, ...]:
        figures: list[Path] = []
        observed_series = sorted(
            {
                str(record["response_series_uid"])
                for record in phase_three.input_records
                if record.get("response_series_uid")
                and record.get("n_rate_kg_ha") is not None
                and record.get("yield_t_ha") is not None
            }
        )
        for response_series_uid in observed_series:
            figures.extend(
                write_observed_series_figures(
                    phase_three.input_records,
                    response_series_uid,
                    output_root=stage_root / "figures" / "observed",
                    formats=config.figure_formats,
                )
            )
        for attempt in phase_three.evidence.selected_attempts:
            if attempt.response_series_uid not in selected_curve_series:
                continue
            figures.extend(
                write_response_curve_figures(
                    phase_three.input_records,
                    attempt,
                    output_root=stage_root / "figures" / "fitted",
                    formats=config.figure_formats,
                )
            )
        return tuple(figures)

    return write_figures


def _table_artifacts(phase_two: Any, phase_three: PhaseThreeResult, phase_four: PhaseFourResult) -> dict[str, TableArtifact]:
    integrity = phase_two.ingestion.integrity_report
    if integrity is None:
        raise ConfigError("Source-integrity report is required before a Phase 5 release")
    return {
        "analysis_candidates": TableArtifact(
            rows=tuple(asdict(candidate) for candidate in phase_four.registry.candidates),
            stable_key="candidate_id",
        ),
        "analysis_pruned_families": TableArtifact(rows=tuple(asdict(item) for item in phase_four.registry.pruned_families)),
        "curve_features": TableArtifact(rows=phase_three.evidence.curve_rows, stable_key="response_series_uid"),
        "derived_curve_features": TableArtifact(
            rows=phase_four.curve_rows,
            stable_key="derived_curve_row_uid",
        ),
        "derived_curve_views": TableArtifact(
            rows=tuple(
                {
                    "view_id": view.view_id,
                    "status": view.status,
                    "reason_codes": view.reason_codes,
                    "dataset_version_id": view.dataset_version_id,
                    "dataset_version_membership_sha256": view.dataset_version_membership_sha256,
                    "source_combination_id": view.source_combination_id,
                    "source_families": view.source_families,
                    "record_uids": view.record_uids,
                    "canonical_input_sha256": view.canonical_input_sha256,
                    "model_policy_sha256": view.model_policy_sha256,
                    "model_attempt_count": len(view.model_attempt_records),
                    "selected_curve_count": len(view.curve_rows),
                }
                for view in phase_four.derived_curve_views
            ),
            stable_key="view_id",
        ),
        "derived_model_attempts": TableArtifact(
            rows=tuple(
                row
                for view in phase_four.derived_curve_views
                for row in view.model_attempt_records
            ),
            stable_key="derived_model_attempt_uid",
        ),
        "derived_model_predictions": TableArtifact(
            rows=tuple(
                row
                for view in phase_four.derived_curve_views
                for row in view.prediction_rows
            ),
            stable_key="derived_prediction_uid",
        ),
        "dataset_versions": TableArtifact(
            rows=tuple(asdict(version) for version in phase_four.dataset_versions),
            stable_key="version_id",
        ),
        "eligibility_ledger": TableArtifact(rows=phase_two.eligibility.ledger, stable_key="record_uid"),
        "factor_catalog": TableArtifact(
            rows=tuple(asdict(entry) for entry in phase_four.factor_catalog),
            stable_key="factor_name",
        ),
        "model_attempts": TableArtifact(rows=phase_three.evidence.model_attempt_records, stable_key="model_attempt_uid"),
        "model_predictions": TableArtifact(rows=phase_three.evidence.prediction_rows),
        "python_analysis_execution": TableArtifact(
            rows=tuple(asdict(result) for result in phase_four.python_results),
            stable_key="candidate_id",
        ),
        "selected_models": TableArtifact(
            rows=tuple(
                row
                for row in phase_three.evidence.model_attempt_records
                if row["model_attempt_uid"]
                in {attempt.model_attempt_uid for attempt in phase_three.evidence.selected_attempts}
            ),
            stable_key="model_attempt_uid",
        ),
        "source_integrity": TableArtifact(
            rows=tuple(
                {"artifact_path": artifact_path, "sha256": sha256}
                for artifact_path, sha256 in sorted(integrity.artifact_sha256.items())
            ),
            stable_key="artifact_path",
        ),
    }


def _source_registry(
    config: ValidatedConfig,
    source_artifact_sha256: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    source_root = config.paths["source_manifest"].parent.resolve()
    registry: dict[str, dict[str, Any]] = {}
    for source_name, source in sorted(config.sources.items()):
        source_path = (config.project_root / str(source["data_path"])).resolve()
        try:
            manifest_artifact_path = source_path.relative_to(source_root).as_posix()
        except ValueError:
            manifest_artifact_path = None
        digest = (
            source_artifact_sha256.get(manifest_artifact_path)
            if manifest_artifact_path is not None
            else None
        )
        registry[source_name] = {
            "source_type": source["source_type"],
            "data_path": source["data_path"],
            "schema_map": source["schema_map"],
            "provider": source["provider"],
            "availability": source["availability"],
            "confirmation_status": source["confirmation_status"],
            "shape_adapter_version": source["shape_adapter_version"],
            "enabled": source_name in config.enabled_sources,
            "manifest_artifact_path": manifest_artifact_path,
            "sha256": digest,
            "checksum_status": (
                "verified"
                if source_name in config.enabled_sources and digest is not None
                else "registered_not_enabled"
                if digest is not None
                else "not_registered"
            ),
        }
    return registry


def _contextual_coverage_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(records)

    def finite_present(value: object) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and value == value
        )

    replication_fields = tuple(
        sorted(
            {
                key
                for row in rows
                for key in ("replication_count", "replicate_count", "replications")
                if key in row
            }
        )
    )
    standard_error_fields = tuple(
        sorted(
            {
                key
                for row in rows
                for key in ("standard_error", "yield_standard_error", "yield_se")
                if key in row
            }
        )
    )
    return {
        "inventory_rows": len(rows),
        "treatment_classified_rows": sum(
            str(row.get("treatment_text_class") or "").strip().casefold()
            not in {"", "unresolved"}
            for row in rows
        ),
        "zero_n_rows": sum(bool(row.get("is_zero_n")) for row in rows),
        "absolute_control_rows": sum(bool(row.get("is_absolute_control")) for row in rows),
        "high_n_rows": sum(bool(row.get("is_high_n")) for row in rows),
        "organic_fertilizer_rows": sum(bool(row.get("organic_fertilizer_present")) for row in rows),
        "biofertilizer_rows": sum(bool(row.get("biofertilizer_present")) for row in rows),
        "p_rate_observed_rows": sum(finite_present(row.get("p_rate_kg_p2o5_ha")) for row in rows),
        "k_rate_observed_rows": sum(finite_present(row.get("k_rate_kg_k2o_ha")) for row in rows),
        "replication_evidence": {
            "registered_fields": list(replication_fields),
            "rows": sum(any(row.get(field) not in {None, ""} for field in replication_fields) for row in rows),
        },
        "standard_error_evidence": {
            "registered_fields": list(standard_error_fields),
            "rows": sum(any(row.get(field) not in {None, ""} for field in standard_error_fields) for row in rows),
        },
    }


def _report_sections(
    config: ValidatedConfig,
    phase_two: Any,
    phase_three: PhaseThreeResult,
    phase_four: PhaseFourResult,
) -> dict[str, list[str]]:
    completed_python = sum(result.status == "completed" for result in phase_four.python_results)
    r_owned = sum(result.engine == "r" for result in phase_four.python_results)
    pruned = sum(candidate.status == "pruned" for candidate in phase_four.registry.candidates)
    unsupported_models = sum(attempt.status != "fitted" for attempt in phase_three.evidence.model_attempts)
    contextual_coverage = _contextual_coverage_summary(phase_two.curation.records)
    unavailable_sources = sum(
        source["availability"] == "expected_unavailable"
        for source in config.sources.values()
    )
    return {
        "primary": [
            f"{contextual_coverage['inventory_rows']} canonical source rows were retained; treatment/control, high-N, P/K, organic/biofertilizer, replication, and standard-error coverage is recorded in the run manifest.",
            f"{len(phase_three.evidence.curve_rows)} supported curve-level outcome rows were generated from selected reportable curve models.",
            f"{len(phase_three.evidence.selected_attempts)} response-series model selections were retained with all candidate attempts.",
        ],
        "sensitivity": [
            f"{len(phase_four.dataset_versions)} immutable dataset versions were registered; unavailable contextual versions remain explicit in the ledger.",
            f"{unavailable_sources} registered source families are explicitly marked expected-unavailable and were not silently omitted.",
        ],
        "exploratory": [f"{completed_python} Python-owned descriptive/coverage candidates completed without inferential claims."],
        "predictive": ["Predictive modeling remains non-reportable unless an approved predictive candidate passes its support and validation gates."],
        "unsupported": [
            f"{pruned} configured candidates were pruned before fitting and remain accounted for in the analysis registry.",
            f"{r_owned} R-owned candidates were retained for R-only dispatch rather than executed by Python.",
            f"{unsupported_models} curve-model attempts were non-reportable and are preserved as explicit failed or unsupported attempts.",
            "Economic optimum and recommendation/target yield gaps remain unavailable until their decision-dependent definitions are approved.",
        ],
    }


def _release_target(config: ValidatedConfig) -> Path:
    seed = int(config.raw["run"]["random_seed"])
    if config.run_mode == "test":
        return config.paths["test_output_root"] / f"n_response_test_{seed}"
    if config.run_mode == "full":
        return config.paths["reports_root"] / f"n_response_full_{seed}"
    raise ConfigError("Validate mode does not have a release target")


def _source_target_paths(config: ValidatedConfig) -> tuple[Path, ...]:
    source_paths = [config.config_path]
    source_paths.extend(config.paths[key] for key in ("core_source_csv", "source_workbook", "source_manifest", "source_checksums", "schema_evidence", "variety_lookup"))
    for source in config.sources.values():
        source_paths.append((config.project_root / str(source["data_path"])).resolve())
        source_paths.append((config.project_root / str(source["schema_map"])).resolve())
    return tuple(source_paths)


def _logged_stage_writer(
    run_log: RunLogger,
    stage_name: str,
    writer: Callable[[Path], Iterable[str | Path]],
) -> Callable[[Path], tuple[str | Path, ...]]:
    def write_with_logging(stage_root: Path) -> tuple[str | Path, ...]:
        with run_log.stage(stage_name):
            written_paths = tuple(writer(stage_root))
        run_log.debug("stage_artifacts_written", stage=stage_name, artifact_count=len(written_paths))
        return written_paths

    return write_with_logging


def release_phases_three_to_five(
    config: ValidatedConfig,
    phase_two: Any,
    phase_three: PhaseThreeResult,
    phase_four: PhaseFourResult,
    *,
    run_log: RunLogger,
) -> PhaseFiveResult:
    """Write the complete Phase 3–5 evidence package only after in-memory gates reconcile."""

    if not config.writes_outputs:
        raise ConfigError("Release packaging is unavailable in validate mode")
    integrity = phase_two.ingestion.integrity_report
    if integrity is None:
        raise ConfigError("Source-integrity report is required before a release")
    target = _release_target(config)
    run_id = target.name
    if run_log.run_id != run_id:
        raise ConfigError("Run logger identity does not match the controlled release target")
    run_log.info("controlled_release_started", release_target=target)
    identity_payload = {
        "config_sha256": sha256_file(config.config_path),
        "code_sha256": _code_fingerprint(),
        "mode": config.run_mode,
        "random_seed": config.raw["run"]["random_seed"],
        "source_artifact_sha256": dict(integrity.artifact_sha256),
    }
    run_identity_sha256 = stable_json_sha256(identity_payload)
    source_registry = _source_registry(config, integrity.artifact_sha256)
    contextual_coverage = _contextual_coverage_summary(phase_two.curation.records)
    r_stage_statuses: list[dict[str, Any]] = []
    r_preparation_status_counts: dict[str, int] = {}
    for _, preparation in phase_four.r_preparations:
        r_preparation_status_counts[preparation.status] = r_preparation_status_counts.get(preparation.status, 0) + 1
    manifest: dict[str, Any] = {
        "status": f"phase_5_{config.run_mode}_release_complete",
        "run_id": run_id,
        "run_identity_sha256": run_identity_sha256,
        "run_identity": identity_payload,
        "code_provenance": {
            "code_sha256": identity_payload["code_sha256"],
            "git": _git_inventory(config.project_root),
        },
        "effective_config": _redact(config.raw),
        "engine_assignments": dict(sorted(config.engine_assignments.items())),
        "runtime_inventory": _runtime_inventory(str(config.raw["engines"]["rscript_command"])),
        "source_integrity": {
            "checked_files": integrity.checked_files,
            "artifact_sha256": dict(integrity.artifact_sha256),
        },
        "source_registry": source_registry,
        "contextual_coverage": contextual_coverage,
        "canonical_rows": len(phase_two.curation.records),
        "eligibility_rows": len(phase_two.eligibility.ledger),
        "tier_counts": dict(phase_two.qc.tier_counts),
        "critical_record_uids": list(phase_two.qc.critical_record_uids),
        "curve_model_attempt_count": len(phase_three.evidence.model_attempts),
        "selected_curve_model_count": len(phase_three.evidence.selected_attempts),
        "curve_feature_row_count": len(phase_three.evidence.curve_rows),
        "analysis_registry": {
            "theoretical_candidate_count": phase_four.registry.theoretical_candidate_count,
            "accounted_candidate_count": phase_four.registry.accounted_candidate_count,
            "reconciles": phase_four.registry.reconciles,
            "r_preparation_status_counts": dict(sorted(r_preparation_status_counts.items())),
        },
        "r_stage_statuses": r_stage_statuses,
        "logging": {
            "artifact_path": "logs/pipeline.jsonl",
            "format": "jsonl",
            "level": run_log.level,
        },
    }
    if target.exists() and not bool(config.raw["run"]["overwrite"]):
        try:
            existing = verify_release_package(target)
            existing_manifest = json.loads(existing.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ReportingError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Existing release package cannot be safely reused: {target}") from exc
        if existing_manifest.get("run_identity_sha256") == run_identity_sha256:
            run_log.info("controlled_release_reused", release_target=target)
            return PhaseFiveResult(package=existing, reused_existing_package=True)
    report_sections = _report_sections(config, phase_two, phase_three, phase_four)

    def write_run_log(stage_root: Path) -> tuple[Path, ...]:
        run_log.info("release_package_ready", release_target=target)
        log_path = run_log.write_jsonl(stage_root)
        manifest["logging"]["record_count"] = len(run_log.records)
        return (log_path,)

    stage_writers = (
        _logged_stage_writer(run_log, "figures", _figure_stage_writer(config, phase_three)),
        _logged_stage_writer(
            run_log,
            "r_stages",
            _r_stage_writer(config, phase_three, phase_four, r_stage_statuses),
        ),
        _logged_stage_writer(
            run_log,
            "terminal_statuses",
            _terminal_status_stage_writer(phase_three, phase_four, r_stage_statuses, manifest, report_sections),
        ),
        write_run_log,
    )
    try:
        package = write_release_package(
            target,
            tables=_table_artifacts(phase_two, phase_three, phase_four),
            manifest=manifest,
            report_sections=report_sections,
            output_formats=config.output_formats,
            overwrite=bool(config.raw["run"]["overwrite"]),
            source_roots=_source_target_paths(config),
            stage_writers=stage_writers,
        )
    except ReportingError as exc:
        raise ConfigError(f"Phase 5 controlled release failed: {exc}") from exc
    return PhaseFiveResult(package=package, reused_existing_package=False)


__all__ = [
    "PhaseFiveResult",
    "PhaseFourResult",
    "PhaseThreeResult",
    "release_phases_three_to_five",
    "run_phase_four",
    "run_phase_three",
]
