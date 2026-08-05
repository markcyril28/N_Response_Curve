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
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from n_response_curve.analysis.analysis_matrix import AnalysisRegistry, build_analysis_registry, build_source_combinations
from n_response_curve.analysis.claims import (
    build_runtime_claim_evidence,
    classify_claim_evidence,
)
from n_response_curve.analysis.comparisons import ManagementSystemProximity, build_management_system_proximity
from n_response_curve.analysis.policy_artifacts import AnalysisPolicyBundle
from n_response_curve.data.config import ConfigError, ValidatedConfig
from n_response_curve.data.curate import project_public_records
from n_response_curve.data.provenance import sha256_file, stable_json_sha256
from n_response_curve.analysis.curve_evidence import CurveEvidenceResult, build_curve_evidence, curve_fit_record_uids
from n_response_curve.analysis.curve_views import DerivedCurveView, build_derived_curve_views
from n_response_curve.analysis.dataset_versions import DatasetVersion, build_dataset_versions
from n_response_curve.analysis.explanatory import PythonAnalysisResult, execute_python_candidates, select_candidate_curve_rows
from n_response_curve.analysis.factor_catalog import FactorCatalogEntry, build_factor_catalog
from n_response_curve.reporting.plots import write_observed_series_figures, write_response_curve_figures
from n_response_curve.analysis.r_bridge import RBridgeError, invoke_r_stage, write_r_stage_contract
from n_response_curve.analysis.r_specs import RAnalysisPreparation, prepare_r_analysis
from n_response_curve.pipeline.policy_governance import (
    RuntimePolicySnapshot,
    effective_analysis_hypotheses,
)
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
    test_subset: Mapping[str, Any] | None
    model_policy: Mapping[str, Any]
    model_policy_sha256: str


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


def _series_coverage_tokens(rows: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    tokens: set[str] = set()
    for row in rows:
        tier = str(
            row.get("series_eligibility_tier")
            or row.get("eligibility_tier")
            or "unresolved"
        ).strip()
        tokens.add(f"eligibility_tier:{tier}")
        treatment = str(
            row.get("treatment_text_class")
            or row.get("treatment_class")
            or "unresolved"
        ).strip()
        tokens.add(f"treatment_class:{treatment}")
        source_type = str(
            row.get("source_type")
            or row.get("source_family")
            or row.get("source_name")
            or "unresolved"
        ).strip()
        tokens.add(f"source_type:{source_type}")
        reasons = tuple(row.get("series_eligibility_reason_codes") or ())
        for reason in reasons:
            if (
                isinstance(reason, str)
                and reason
                and reason != "PRIMARY_ELIGIBLE"
            ):
                tokens.add(f"edge_case:{reason}")
        if bool(row.get("is_high_n")):
            tokens.add("edge_case:high_n")
        if bool(row.get("organic_fertilizer_present")) or bool(row.get("biofertilizer_present")):
            tokens.add("edge_case:organic_or_biofertilizer")
        if row.get("same_n_status") == "repeated_measurement":
            tokens.add("edge_case:repeated_n_level")
    return frozenset(tokens)


def _test_subset_selection(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    seed: int,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        series_uid = record.get("response_series_uid")
        if (
            record.get("series_status") == "resolved"
            and isinstance(series_uid, str)
            and series_uid
        ):
            grouped.setdefault(series_uid, []).append(record)
    tokens_by_series = {
        series_uid: _series_coverage_tokens(rows)
        for series_uid, rows in grouped.items()
    }
    universe = frozenset(
        token for tokens in tokens_by_series.values() for token in tokens
    )

    def seeded_rank(series_uid: str) -> str:
        return hashlib.sha256(f"{seed}\0{series_uid}".encode("utf-8")).hexdigest()

    remaining = set(tokens_by_series)
    selected: list[str] = []
    covered: set[str] = set()
    while remaining and len(selected) < limit:
        chosen = min(
            remaining,
            key=lambda series_uid: (
                -len(tokens_by_series[series_uid] - covered),
                seeded_rank(series_uid),
                series_uid,
            ),
        )
        selected.append(chosen)
        covered.update(tokens_by_series[chosen])
        remaining.remove(chosen)
    permitted = set(selected)
    selected_records = tuple(
        dict(record)
        for record in records
        if str(record.get("response_series_uid") or "") in permitted
    )
    coverage_counts = Counter(token.split(":", 1)[0] for token in covered)
    snapshot = {
        "algorithm": "deterministic_representative_greedy_coverage_v1",
        "random_seed": seed,
        "configured_series_limit": limit,
        "available_resolved_series_count": len(grouped),
        "selected_series_count": len(selected),
        "selected_series_uids": selected,
        "selected_series_uid_sha256": stable_json_sha256(selected),
        "coverage_dimensions": dict(sorted(coverage_counts.items())),
        "covered_tokens": sorted(covered),
        "uncovered_tokens": sorted(universe - covered),
    }
    return selected_records, snapshot


def _model_input_records(
    config: ValidatedConfig,
    ledger: Iterable[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], Mapping[str, Any] | None]:
    records = tuple(dict(record) for record in ledger)
    if config.run_mode != "test":
        return records, None
    return _test_subset_selection(
        records,
        limit=int(config.raw["run"]["test_group_limit"]),
        seed=int(config.raw["run"]["random_seed"]),
    )


def build_effective_model_policy(
    config: ValidatedConfig,
    analysis_policy: AnalysisPolicyBundle | None,
) -> Mapping[str, Any]:
    """Compose one immutable curve policy from runtime mechanics and reviewed controls."""

    mechanics = config.raw["modeling"]
    if not isinstance(mechanics, Mapping):
        raise ConfigError("Validated model mechanics are unavailable")
    if analysis_policy is None:
        return mechanics
    reviewed_controls = analysis_policy.curve_model_policy.effective_controls
    overlap = set(mechanics).intersection(reviewed_controls)
    if overlap:
        raise ConfigError(
            "Reviewed curve controls overlap runtime model mechanics: "
            + ", ".join(sorted(overlap))
        )
    return MappingProxyType({**mechanics, **reviewed_controls})


def run_phase_three(
    config: ValidatedConfig,
    phase_two: Any,
    *,
    model_policy: Mapping[str, Any] | None = None,
) -> PhaseThreeResult:
    """Fit only configured curve candidates and preserve every attempt as evidence."""

    effective_model_policy = (
        model_policy
        if model_policy is not None
        else build_effective_model_policy(config, None)
    )
    model_policy_sha256 = stable_json_sha256(effective_model_policy)
    input_records, test_subset = _model_input_records(
        config,
        phase_two.analysis_eligibility.ledger,
    )
    evidence = build_curve_evidence(
        input_records,
        model_names=config.enabled_models,
        policy=effective_model_policy,
        fit_record_uids=curve_fit_record_uids(input_records, primary_only=True),
    )
    attempt_policy_hashes = {
        attempt.model_policy_sha256
        for attempt in getattr(evidence, "model_attempts", ())
    }
    if attempt_policy_hashes - {model_policy_sha256}:
        raise ConfigError("Curve-model attempts do not share the effective policy hash")
    return PhaseThreeResult(
        evidence=evidence,
        input_records=input_records,
        test_subset=test_subset,
        model_policy=effective_model_policy,
        model_policy_sha256=model_policy_sha256,
    )


def run_phase_four(
    config: ValidatedConfig,
    phase_two: Any,
    phase_three: PhaseThreeResult,
    *,
    model_policy: Mapping[str, Any] | None = None,
    analysis_policy: AnalysisPolicyBundle | None = None,
) -> PhaseFourResult:
    """Build immutable analysis views, factor coverage, and an exhaustive dispatch ledger."""

    effective_model_policy = (
        model_policy if model_policy is not None else phase_three.model_policy
    )
    if stable_json_sha256(effective_model_policy) != phase_three.model_policy_sha256:
        raise ConfigError(
            "Primary and derived curve fits must share one effective model policy"
        )
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
        policy=effective_model_policy,
    )
    derived_policy_hashes = {
        view.model_policy_sha256 for view in derived_curve_views
    }
    if derived_policy_hashes - {phase_three.model_policy_sha256}:
        raise ConfigError("Derived curve views do not share the effective policy hash")
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
        hypothesis_specifications=effective_analysis_hypotheses(
            config,
            analysis_policy,
        ),
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
            contract = None
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
            finally:
                if contract is not None:
                    # Analysis inputs and their local contract may contain row-level
                    # identifiers. Preserve their hashes in the status ledger but
                    # never carry these execution-only files into a release package.
                    contract.input_path.unlink(missing_ok=True)
                    contract.contract_path.unlink(missing_ok=True)
            if contract is not None and contract.output_path.is_file():
                produced.append(contract.output_path)
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
                        "engine_result": (
                            dict(raw_row)
                            if isinstance(raw_row, Mapping)
                            else None
                        ),
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


def _claim_classification_stage_writer(
    phase_four: PhaseFourResult,
    multiplicity_reconciliation: Mapping[str, Any],
    terminal_state: Mapping[str, Any],
    claim_state: dict[str, Any],
    manifest: dict[str, Any],
    report_sections: dict[str, list[str]],
):
    def write_claim_classification(stage_root: Path) -> tuple[Path, ...]:
        terminal_rows = terminal_state.get("rows", ())
        terminal_by_candidate = {
            str(row["candidate_id"]): row
            for row in terminal_rows
            if isinstance(row, Mapping)
            and isinstance(row.get("candidate_id"), str)
            and row.get("candidate_id")
        }
        try:
            sensitivity_evidence = build_runtime_claim_evidence(
                phase_four.registry,
                multiplicity_reconciliation,
            )
            payload = classify_claim_evidence(
                phase_four.registry,
                multiplicity_reconciliation,
                sensitivity_evidence_by_candidate=sensitivity_evidence,
                terminal_statuses_by_candidate=terminal_by_candidate,
            )
        except ValueError as exc:
            raise ReportingError(f"Claim-classification policy failed: {exc}") from exc
        claim_state.clear()
        claim_state.update(payload)
        status_counts = payload["status_counts"]
        manifest["claim_classification"] = {
            "artifact_path": "claim_classification.json",
            "status": payload["status"],
            "candidate_count": payload["candidate_count"],
            "status_counts": status_counts,
        }
        formatted_counts = ", ".join(
            f"{key}={value}" for key, value in sorted(status_counts.items())
        ) or "none"
        report_sections["unsupported"].append(
            "Positive-claim classification is restricted to noncausal supported "
            f"associations after all gates pass ({formatted_counts})."
        )
        path = stage_root / "claim_classification.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return (path,)

    return write_claim_classification


def _terminal_status_stage_writer(
    phase_three: PhaseThreeResult,
    phase_four: PhaseFourResult,
    r_statuses: list[dict[str, Any]],
    multiplicity_reconciliation: Mapping[str, Any],
    terminal_state: dict[str, Any],
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
        multiplicity_by_candidate = multiplicity_reconciliation.get("candidate_statuses", {})
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
            inferential_interpretability = "not_applicable"
            multiplicity_status = None
            if candidate.analysis_family in _INFERENTIAL_ANALYSIS_FAMILIES and terminal_status == "run":
                reconciliation = multiplicity_by_candidate.get(candidate.candidate_id)
                if candidate.multiplicity_family_id is None:
                    terminal_status = "not_interpretable"
                    reason_codes.append("MULTIPLICITY_FAMILY_NOT_DECLARED")
                    multiplicity_status = "missing"
                    inferential_interpretability = "not_interpretable"
                elif not isinstance(reconciliation, Mapping) or reconciliation.get("status") != "reconciled":
                    terminal_status = "not_interpretable"
                    reconciliation_reasons = (
                        reconciliation.get("reason_codes", ())
                        if isinstance(reconciliation, Mapping)
                        else ("MULTIPLICITY_FAMILY_RECONCILIATION_MISSING",)
                    )
                    reason_codes.extend(str(reason) for reason in reconciliation_reasons)
                    multiplicity_status = (
                        str(reconciliation.get("status"))
                        if isinstance(reconciliation, Mapping)
                        else "missing"
                    )
                    inferential_interpretability = "not_interpretable"
                else:
                    multiplicity_status = "reconciled"
                    inferential_interpretability = "interpretable"
            reason_codes = sorted(set(reason_codes))
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
                    "multiplicity_family_id": candidate.multiplicity_family_id,
                    "multiplicity_status": multiplicity_status,
                    "inferential_interpretability": inferential_interpretability,
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
        model_reason_counts = Counter(
            str(reason)
            for failure in model_failures
            for reason in failure.get("reason_codes", ())
        )
        analysis_reason_counts = Counter(
            str(reason)
            for row in status_rows
            for reason in row.get("reason_codes", ())
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
        status_counts_by_engine: dict[str, dict[str, int]] = {}
        for row in status_rows:
            engine_counts = status_counts_by_engine.setdefault(str(row["engine"]), {})
            terminal_status = str(row["terminal_status"])
            engine_counts[terminal_status] = engine_counts.get(terminal_status, 0) + 1
        manifest["analysis_registry"]["terminal_status_counts_by_engine"] = {
            engine: dict(sorted(counts.items()))
            for engine, counts in sorted(status_counts_by_engine.items())
        }
        manifest["warnings"] = warnings
        manifest["model_failures"] = model_failures
        manifest["contract_inventory"] = contract_inventory
        manifest["qc_summary"].update(
            {
                "model_failure_count": len(model_failures),
                "model_reason_counts": dict(sorted(model_reason_counts.items())),
                "analysis_terminal_row_count": len(status_rows),
                "analysis_reason_counts": dict(sorted(analysis_reason_counts.items())),
                "runtime_warning_count": len(warnings),
            }
        )

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
            f"{len(phase_three.evidence.credible_attempts)} credible models under "
            f"{phase_three.evidence.reporting_policy}; ranked selections="
            f"{len(phase_three.evidence.selected_attempts)}. Every noncredible or failed attempt remains explicit."
        )
        report_sections["predictive"].append(
            f"{predictive_runs} study-grouped predictive candidates completed; predictive results are not causal estimates."
        )
        r_terminal_counts = status_counts_by_engine.get("r", {})
        report_sections["exploratory"].append(
            "R-owned terminal outcomes: "
            + (_format_counts(r_terminal_counts) if r_terminal_counts else "no R candidates were dispatched")
            + "."
        )
        report_sections["unsupported"].append(
            f"Terminal analysis reconciliation accounted for {terminal_accounted} of "
            f"{phase_four.registry.theoretical_candidate_count} theoretical candidates."
        )

        payload = {
            "concrete_candidates": status_rows,
            "compressed_pruned_families": [asdict(item) for item in phase_four.registry.pruned_families],
            "terminal_status_counts": dict(sorted(status_counts.items())),
            "reason_counts": dict(sorted(analysis_reason_counts.items())),
            "terminal_accounted_candidate_count": terminal_accounted,
            "theoretical_candidate_count": phase_four.registry.theoretical_candidate_count,
            "reconciles": terminal_reconciles,
        }
        terminal_state.clear()
        terminal_state.update(
            {
                "rows": tuple(status_rows),
                "status_counts": dict(sorted(status_counts.items())),
                "reconciles": terminal_reconciles,
            }
        )
        status_path = stage_root / "analysis_terminal_statuses.json"
        status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return (status_path,)

    return write_terminal_statuses


def _figure_stage_writer(
    config: ValidatedConfig,
    phase_three: PhaseThreeResult,
    series_evidence_rows: Sequence[Mapping[str, Any]],
):
    reportable_curve_series = {
        row["response_series_uid"] for row in phase_three.evidence.curve_rows
    }
    evidence_by_series = {str(row["response_series_uid"]): row for row in series_evidence_rows}
    fitted_attempts = (
        phase_three.evidence.credible_attempts
        if phase_three.evidence.reporting_policy == "all_credible_no_selection"
        else phase_three.evidence.selected_attempts
    )

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
                    evidence_row=evidence_by_series.get(response_series_uid),
                )
            )
        for attempt in fitted_attempts:
            if attempt.response_series_uid not in reportable_curve_series:
                continue
            figures.extend(
                write_response_curve_figures(
                    phase_three.input_records,
                    attempt,
                    output_root=stage_root / "figures" / "fitted",
                    formats=config.figure_formats,
                    evidence_row=evidence_by_series.get(attempt.response_series_uid),
                )
            )
        return tuple(figures)

    return write_figures


def _qc_summary_tables(phase_two: Any) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Build complete series- and source-level QC summaries from the row ledger."""

    ledger = tuple(phase_two.eligibility.ledger)
    review_uids = {str(row["record_uid"]) for row in phase_two.qc.review_rows}
    series_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    source_groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in ledger:
        source_groups.setdefault(str(row["source_name"]), []).append(row)
        series_uid = row.get("response_series_uid")
        if isinstance(series_uid, str) and series_uid:
            series_key = ("resolved", series_uid)
        else:
            series_key = ("unresolved_record", str(row["record_uid"]))
        series_groups.setdefault(series_key, []).append(row)

    def summarize(
        scope: str,
        identifier: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        group_status: str,
    ) -> dict[str, Any]:
        tier_counts = Counter(str(row.get("eligibility_tier") or "unresolved") for row in rows)
        reason_counts = Counter(
            str(reason)
            for row in rows
            for reason in row.get("eligibility_reason_codes", ())
        )
        reasons = sorted(reason_counts)
        row_identifiers = tuple(sorted(str(row["record_uid"]) for row in rows))
        key_name = "response_series_uid" if scope == "series" else "source_name"
        return {
            f"{scope}_qc_uid": hashlib.sha256(
                f"{scope}-qc-v2\0{group_status}\0{identifier}".encode("utf-8")
            ).hexdigest(),
            key_name: identifier if group_status == "resolved" or scope == "source" else None,
            "series_review_record_uid": (
                identifier
                if scope == "series" and group_status == "unresolved_record"
                else None
            ),
            "group_status": group_status,
            "record_uids": row_identifiers,
            "record_count": len(rows),
            "review_record_count": sum(str(row["record_uid"]) in review_uids for row in rows),
            "critical_record_count": sum(bool(row.get("has_critical_error")) for row in rows),
            "tier_A_count": tier_counts.get("A", 0),
            "tier_B_count": tier_counts.get("B", 0),
            "tier_C_count": tier_counts.get("C", 0),
            "tier_D_count": tier_counts.get("D", 0),
            "reason_codes": tuple(reasons),
            "reason_counts": dict(sorted(reason_counts.items())),
            "reconciles": sum(tier_counts.values()) == len(rows),
        }

    series_rows = tuple(
        summarize(
            "series",
            identifier,
            rows,
            group_status=group_status,
        )
        for (group_status, identifier), rows in sorted(series_groups.items())
    )
    source_rows = tuple(
        summarize(
            "source",
            identifier,
            rows,
            group_status="source",
        )
        for identifier, rows in sorted(source_groups.items())
    )
    if sum(row["record_count"] for row in series_rows) != len(ledger):
        raise ReportingError("Series-level QC summaries do not reconcile to the row ledger")
    if sum(row["record_count"] for row in source_rows) != len(ledger):
        raise ReportingError("Source-level QC summaries do not reconcile to the row ledger")
    return series_rows, source_rows


def _table_artifacts(
    phase_two: Any,
    phase_three: PhaseThreeResult,
    phase_four: PhaseFourResult,
    *,
    series_evidence_rows: Sequence[Mapping[str, Any]],
    descriptive_summary_rows: Sequence[Mapping[str, Any]],
) -> dict[str, TableArtifact]:
    integrity = phase_two.ingestion.integrity_report
    if integrity is None:
        raise ConfigError("Source-integrity report is required before a Phase 5 release")
    series_qc_rows, source_qc_rows = _qc_summary_tables(phase_two)
    return {
        "analysis_candidates": TableArtifact(
            rows=tuple(asdict(candidate) for candidate in phase_four.registry.candidates),
            stable_key="candidate_id",
        ),
        "analysis_pruned_families": TableArtifact(rows=tuple(asdict(item) for item in phase_four.registry.pruned_families)),
        "curve_features": TableArtifact(rows=phase_three.evidence.curve_rows, stable_key="response_series_uid"),
        "series_evidence": TableArtifact(rows=series_evidence_rows, stable_key="response_series_uid"),
        "descriptive_summaries": TableArtifact(
            rows=descriptive_summary_rows,
            stable_key="summary_uid",
        ),
        "management_system_proximity": TableArtifact(
            rows=tuple(asdict(row) for row in phase_four.management_system_proximity),
            stable_key="management_proximity_uid",
        ),
        "series_qc": TableArtifact(rows=series_qc_rows, stable_key="series_qc_uid"),
        "source_qc": TableArtifact(rows=source_qc_rows, stable_key="source_qc_uid"),
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
                    "curve_feature_count": len(view.curve_rows),
                    "credible_model_count": len(view.evidence.credible_attempts),
                    "selected_model_count": len(view.evidence.selected_attempts),
                    "model_reporting_policy": view.evidence.reporting_policy,
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
        "credible_models": TableArtifact(
            rows=tuple(
                row
                for row in phase_three.evidence.model_attempt_records
                if row["model_attempt_uid"]
                in {
                    attempt.model_attempt_uid
                    for attempt in phase_three.evidence.credible_attempts
                }
            ),
            stable_key="model_attempt_uid",
        ),
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
            "country_code": source.get("country_code"),
            "source_family": source.get("source_family", source_name),
            "data_path": source["data_path"],
            "schema_map": source["schema_map"],
            "workbook": source.get("workbook"),
            "sheet": source.get("sheet"),
            "provider": source["provider"],
            "provenance_notes": source["provenance_notes"],
            "encoding": source["encoding"],
            "data_classification": source["data_classification"],
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


_CATEGORICAL_SUMMARY_FIELDS = (
    "evidence_status",
    "evidence_strength",
    "curve_shape_class",
    "optimum_status",
    "maximum_reference_basis",
    "maximum_proximity_status",
    "target_yield_status",
)
_NUMERIC_SUMMARY_FIELDS = (
    "agronomic_optimum_n_kg_ha",
    "predicted_observed_domain_peak_yield_t_ha",
    "finite_maximum_yield_t_ha",
    "fitted_asymptote_yield_t_ha",
    "supported_max_yield_t_ha",
    "observed_max_gap_to_finite_maximum_t_ha",
    "observed_max_gap_to_supported_maximum_t_ha",
    "observed_max_attainment_fraction",
)


def _reason_codes(row: Mapping[str, Any], fallback: str) -> tuple[str, ...]:
    raw = row.get("reason_codes", ())
    if isinstance(raw, str):
        values = (raw,) if raw.strip() else ()
    elif isinstance(raw, (list, tuple, set, frozenset)):
        values = tuple(str(value).strip() for value in raw if str(value).strip())
    else:
        values = ()
    return tuple(sorted(set(values))) or (fallback,)


def _validated_series_evidence_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_row in rows:
        row = dict(raw_row)
        series_uid = str(row.get("response_series_uid") or "").strip()
        if not series_uid:
            raise ReportingError("Series evidence contains a missing response_series_uid")
        if series_uid in seen:
            raise ReportingError(f"Series evidence contains a duplicate response_series_uid: {series_uid}")
        seen.add(series_uid)
        row["response_series_uid"] = series_uid
        row["evidence_status"] = str(row.get("evidence_status") or "unreported")
        row["evidence_strength"] = str(row.get("evidence_strength") or "unreported")
        if row["evidence_status"] == "unsupported" and not row.get("reason_codes"):
            row["reason_codes"] = ("UNSUPPORTED_SERIES_WITHOUT_RECORDED_REASON",)
        normalized.append(row)
    return tuple(sorted(normalized, key=lambda row: row["response_series_uid"]))


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _summary_row(**values: Any) -> dict[str, Any]:
    identity = {
        key: values.get(key)
        for key in (
            "summary_scope",
            "context_dimension",
            "context_value",
            "measure_name",
            "measure_type",
            "category_value",
            "availability_status",
            "reason_codes",
        )
    }
    return {"summary_uid": stable_json_sha256(identity), **values}


def _summaries_for_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    summary_scope: str,
    context_dimension: str | None,
    context_value: str | None,
) -> tuple[dict[str, Any], ...]:
    summaries: list[dict[str, Any]] = []
    for field in _CATEGORICAL_SUMMARY_FIELDS:
        counts: Counter[str] = Counter()
        missing_rows: list[Mapping[str, Any]] = []
        for row in rows:
            value = str(row.get(field) or "").strip()
            if value:
                counts[value] += 1
            else:
                missing_rows.append(row)
        for category, count in sorted(counts.items()):
            summaries.append(
                _summary_row(
                    summary_scope=summary_scope,
                    context_dimension=context_dimension,
                    context_value=context_value,
                    measure_name=field,
                    measure_type="categorical",
                    category_value=category,
                    availability_status="available",
                    series_count=count,
                    numeric_count=None,
                    unavailable_series_count=0,
                    minimum=None,
                    median=None,
                    maximum=None,
                    reason_codes=(),
                )
            )
        if missing_rows:
            reasons = tuple(
                sorted(
                    {
                        reason
                        for row in missing_rows
                        for reason in _reason_codes(row, f"{field.upper()}_UNAVAILABLE")
                    }
                )
            )
            summaries.append(
                _summary_row(
                    summary_scope=summary_scope,
                    context_dimension=context_dimension,
                    context_value=context_value,
                    measure_name=field,
                    measure_type="categorical",
                    category_value=None,
                    availability_status="unavailable",
                    series_count=len(missing_rows),
                    numeric_count=None,
                    unavailable_series_count=len(missing_rows),
                    minimum=None,
                    median=None,
                    maximum=None,
                    reason_codes=reasons,
                )
            )
    for field in _NUMERIC_SUMMARY_FIELDS:
        numeric = [value for row in rows if (value := _finite_float(row.get(field))) is not None]
        missing_rows = [row for row in rows if _finite_float(row.get(field)) is None]
        reasons = tuple(
            sorted(
                {
                    reason
                    for row in missing_rows
                    for reason in _reason_codes(row, f"NO_FINITE_{field.upper()}")
                }
            )
        )
        summaries.append(
            _summary_row(
                summary_scope=summary_scope,
                context_dimension=context_dimension,
                context_value=context_value,
                measure_name=field,
                measure_type="numeric",
                category_value=None,
                availability_status="available" if numeric else "unavailable",
                series_count=len(rows),
                numeric_count=len(numeric),
                unavailable_series_count=len(missing_rows),
                minimum=min(numeric) if numeric else None,
                median=median(numeric) if numeric else None,
                maximum=max(numeric) if numeric else None,
                reason_codes=reasons,
            )
        )
    return tuple(summaries)


def _context_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return json.dumps(_redact(value), ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _build_descriptive_summary_rows(
    series_evidence_rows: Sequence[Mapping[str, Any]],
    input_records: Sequence[Mapping[str, Any]],
    *,
    context_dimensions: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    if not series_evidence_rows:
        return (
            _summary_row(
                summary_scope="marginal",
                context_dimension=None,
                context_value=None,
                measure_name="series_evidence",
                measure_type="inventory",
                category_value=None,
                availability_status="unavailable",
                series_count=0,
                numeric_count=None,
                unavailable_series_count=0,
                minimum=None,
                median=None,
                maximum=None,
                reason_codes=("NO_SERIES_EVIDENCE_ROWS",),
            ),
        )
    summaries = list(
        _summaries_for_group(
            series_evidence_rows,
            summary_scope="marginal",
            context_dimension=None,
            context_value=None,
        )
    )
    evidence_by_uid = {str(row["response_series_uid"]): row for row in series_evidence_rows}
    records_by_uid: dict[str, list[Mapping[str, Any]]] = {}
    for record in input_records:
        series_uid = str(record.get("response_series_uid") or "")
        if series_uid in evidence_by_uid:
            records_by_uid.setdefault(series_uid, []).append(record)
    for dimension in context_dimensions:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        missing_rows: list[Mapping[str, Any]] = []
        ambiguous_rows: list[Mapping[str, Any]] = []
        for series_uid, evidence_row in evidence_by_uid.items():
            values = {
                value
                for record in records_by_uid.get(series_uid, ())
                if (value := _context_value(record.get(dimension))) is not None
            }
            if len(values) == 1:
                grouped.setdefault(next(iter(values)), []).append(evidence_row)
            elif values:
                ambiguous_rows.append(evidence_row)
            else:
                missing_rows.append(evidence_row)
        for value, group_rows in sorted(grouped.items()):
            summaries.extend(
                _summaries_for_group(
                    group_rows,
                    summary_scope="context",
                    context_dimension=str(dimension),
                    context_value=value,
                )
            )
        for unavailable_rows, reason in (
            (missing_rows, "CONTEXT_DIMENSION_UNAVAILABLE"),
            (ambiguous_rows, "MULTIPLE_CONTEXT_VALUES_WITHIN_SERIES"),
        ):
            if unavailable_rows:
                summaries.append(
                    _summary_row(
                        summary_scope="context",
                        context_dimension=str(dimension),
                        context_value=None,
                        measure_name=str(dimension),
                        measure_type="context_membership",
                        category_value=None,
                        availability_status="unavailable",
                        series_count=len(unavailable_rows),
                        numeric_count=None,
                        unavailable_series_count=len(unavailable_rows),
                        minimum=None,
                        median=None,
                        maximum=None,
                        reason_codes=(reason,),
                    )
                )
    return tuple(summaries)


def _field_counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or "unavailable") for row in rows).items()))


def _format_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "none"


def _unsupported_reason_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if str(row.get("evidence_status") or "") == "unsupported":
            counts.update(_reason_codes(row, "UNSUPPORTED_SERIES_WITHOUT_RECORDED_REASON"))
    return dict(sorted(counts.items()))


def _descriptive_summary_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reasons: Counter[str] = Counter()
    for row in rows:
        if row.get("availability_status") != "unavailable":
            continue
        raw_reasons = row.get("reason_codes", ())
        reasons.update((raw_reasons,) if isinstance(raw_reasons, str) else map(str, raw_reasons))
    return {
        "row_count": len(rows),
        "available_row_count": sum(row.get("availability_status") == "available" for row in rows),
        "unavailable_row_count": sum(row.get("availability_status") == "unavailable" for row in rows),
        "unavailable_reason_counts": dict(sorted(reasons.items())),
    }


def _hypothesis_snapshot(config: ValidatedConfig) -> dict[str, Any]:
    specifications = _redact(
        tuple(config.raw.get("analysis_hypotheses", {}).get("specifications", ()))
    )
    return {
        "status": "bounded" if specifications else "not_authoritatively_specified",
        "specification_count": len(specifications),
        "specifications": specifications,
        "sha256": stable_json_sha256(specifications),
    }


def _report_sections(
    config: ValidatedConfig,
    phase_two: Any,
    phase_three: PhaseThreeResult,
    phase_four: PhaseFourResult,
    *,
    series_evidence_rows: Sequence[Mapping[str, Any]],
    descriptive_summary_rows: Sequence[Mapping[str, Any]],
    hypothesis_snapshot: Mapping[str, Any],
) -> dict[str, list[str]]:
    completed_python = sum(result.status == "completed" for result in phase_four.python_results)
    pruned = sum(candidate.status == "pruned" for candidate in phase_four.registry.candidates)
    unsupported_models = sum(attempt.status != "fitted" for attempt in phase_three.evidence.model_attempts)
    contextual_coverage = _contextual_coverage_summary(phase_two.curation.records)
    unavailable_sources = sum(
        source["availability"] == "expected_unavailable"
        for source in config.sources.values()
    )
    evidence_status_counts = _field_counts(series_evidence_rows, "evidence_status")
    evidence_strength_counts = _field_counts(series_evidence_rows, "evidence_strength")
    shape_counts = _field_counts(series_evidence_rows, "curve_shape_class")
    optimum_counts = _field_counts(series_evidence_rows, "optimum_status")
    maximum_counts = _field_counts(series_evidence_rows, "maximum_proximity_status")
    target_counts = _field_counts(series_evidence_rows, "target_yield_status")
    management_status_counts = Counter(row.status for row in phase_four.management_system_proximity)
    management_target_counts = Counter(row.target_gap_status for row in phase_four.management_system_proximity)
    unsupported_reasons = _unsupported_reason_counts(series_evidence_rows)
    context_parts: list[str] = []
    for dimension in config.comparison_dimensions:
        counts: Counter[str] = Counter()
        for row in descriptive_summary_rows:
            if (
                row.get("summary_scope") == "context"
                and row.get("context_dimension") == dimension
                and row.get("measure_name") == "evidence_status"
                and row.get("availability_status") == "available"
            ):
                counts[str(row.get("context_value"))] += int(row.get("series_count") or 0)
        context_parts.append(f"{dimension}[{_format_counts(counts)}]")
    return {
        "primary": [
            f"{contextual_coverage['inventory_rows']} canonical source rows were retained; treatment/control, high-N, P/K, organic/biofertilizer, replication, and standard-error coverage is recorded in the run manifest.",
            f"{len(series_evidence_rows)} response series were retained in the all-series evidence ledger ({_format_counts(evidence_status_counts)}); {len(phase_three.evidence.curve_rows)} fitted-evidence rows form the narrower curve_features subset.",
            f"Model reporting policy={phase_three.evidence.reporting_policy}; "
            f"{len(phase_three.evidence.credible_attempts)} credible candidates and "
            f"{len(phase_three.evidence.selected_attempts)} ranked selections were retained with all attempts.",
        ],
        "descriptive": [
            f"Curve-shape distribution: {_format_counts(shape_counts)}.",
            f"Agronomic-optimum status distribution: {_format_counts(optimum_counts)}.",
            f"Observed-to-maximum proximity distribution: {_format_counts(maximum_counts)}.",
            f"Evidence-strength distribution: {_format_counts(evidence_strength_counts)}; two-level contrasts remain weaker evidence rather than fitted curves.",
            f"ANA-15 management-system proximity rows: {_format_counts(management_status_counts)}; target-gap statuses: {_format_counts(management_target_counts)}.",
            "Context-stratified series coverage: "
            + ("; ".join(context_parts) if context_parts else "no comparison dimensions configured")
            + ".",
        ],
        "sensitivity": [
            f"Country scope is {','.join(config.scope_countries)}; response-series boundaries use {','.join(config.series_identity_dimensions)} with N-level tolerance {config.raw['eligibility']['n_level_tolerance_kg_ha']} kg N/ha.",
            f"Bounded-hypothesis snapshot status={hypothesis_snapshot.get('status')}, specifications={hypothesis_snapshot.get('specification_count')}, sha256={hypothesis_snapshot.get('sha256')}.",
            f"{len(phase_four.dataset_versions)} immutable dataset versions were registered; unavailable contextual versions remain explicit in the ledger.",
            f"{unavailable_sources} registered source families are explicitly marked expected-unavailable and were not silently omitted.",
        ],
        "exploratory": [f"{completed_python} Python-owned descriptive/coverage candidates completed without inferential claims."],
        "predictive": ["Predictive modeling remains non-reportable unless an approved predictive candidate passes its support and validation gates."],
        "unsupported": [
            f"{pruned} configured candidates were pruned before fitting and remain accounted for in the analysis registry.",
            f"{unsupported_models} curve-model attempts were non-reportable and are preserved as explicit failed or unsupported attempts.",
            f"Unsupported series reasons: {_format_counts(unsupported_reasons)}.",
            f"Target-yield comparison status: {_format_counts(target_counts)}; not_configured means no target value or gap was inferred.",
            "Baseline-response metrics are disabled until ELG-10 defines the eligible zero-N/control class."
            if not config.raw["modeling"]["allow_baseline_response_metrics"]
            else "Baseline-response metrics use the explicitly enabled ELG-10 policy.",
            "Economic optimum remains unavailable until decision-dependent price scenarios are approved.",
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
        if source.get("workbook"):
            source_paths.append((config.project_root / str(source["workbook"])).resolve())
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
    series_evidence_rows = _validated_series_evidence_rows(phase_three.evidence.series_evidence_rows)
    descriptive_summary_rows = _build_descriptive_summary_rows(
        series_evidence_rows,
        phase_three.input_records,
        context_dimensions=config.comparison_dimensions,
    )
    hypothesis_snapshot = _hypothesis_snapshot(config)
    identity_payload = {
        "config_sha256": sha256_file(config.config_path),
        "code_sha256": _code_fingerprint(),
        "hypothesis_specifications_sha256": hypothesis_snapshot["sha256"],
        "mode": config.run_mode,
        "random_seed": config.raw["run"]["random_seed"],
        "scope_countries": list(config.scope_countries),
        "series_identity_dimensions": list(config.series_identity_dimensions),
        "source_artifact_sha256": dict(integrity.artifact_sha256),
    }
    run_identity_sha256 = stable_json_sha256(identity_payload)
    source_registry = _source_registry(config, integrity.artifact_sha256)
    contextual_coverage = _contextual_coverage_summary(phase_two.curation.records)
    series_qc_rows, source_qc_rows = _qc_summary_tables(phase_two)
    r_stage_statuses: list[dict[str, Any]] = []
    r_stage_result_rows: dict[str, tuple[dict[str, Any], ...]] = {}
    multiplicity_reconciliation: dict[str, Any] = {}
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
        "output_profile": {
            "table_formats": list(config.output_formats),
            "figure_formats": list(config.figure_formats),
            "document_formats": ["html", "md", "pdf"],
            "collision_policy": config.raw["outputs"]["collision_policy"],
            "overwrite": config.raw["run"]["overwrite"],
        },
        "analysis_scope": {
            "country_codes": list(config.scope_countries),
            "series_identity_dimensions": list(config.series_identity_dimensions),
            "n_level_tolerance_kg_ha": config.raw["eligibility"]["n_level_tolerance_kg_ha"],
            "enabled_analysis_families": list(config.analysis_families),
            "deferred_analysis_families": list(
                config.raw["analysis_matrix"]["deferred_analysis_families"]
            ),
            "enabled_interaction_orders": list(config.interaction_orders),
            "deferred_interaction_orders": list(
                config.raw["analysis_matrix"]["deferred_interaction_orders"]
            ),
        },
        "analysis_hypotheses": hypothesis_snapshot,
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
        "qc_summary": {
            "gate_policy": config.raw["run"]["qc_gate"],
            "reconciles": phase_two.qc.reconciles,
            "inventory_row_count": phase_two.qc.inventory_rows,
            "row_ledger_count": len(phase_two.eligibility.ledger),
            "series_ledger_count": len(series_qc_rows),
            "source_ledger_count": len(source_qc_rows),
            "series_ledger_record_count": sum(
                int(row["record_count"]) for row in series_qc_rows
            ),
            "source_ledger_record_count": sum(
                int(row["record_count"]) for row in source_qc_rows
            ),
            "series_ledger_reconciles": (
                sum(int(row["record_count"]) for row in series_qc_rows)
                == phase_two.qc.inventory_rows
            ),
            "source_ledger_reconciles": (
                sum(int(row["record_count"]) for row in source_qc_rows)
                == phase_two.qc.inventory_rows
            ),
            "review_record_count": len(phase_two.qc.review_rows),
            "review_record_uids": sorted(str(row["record_uid"]) for row in phase_two.qc.review_rows),
            "reason_counts": dict(phase_two.qc.reason_counts),
            "row_level_qc_enabled": config.raw["outputs"]["row_level_qc"],
        },
        "curve_model_attempt_count": len(phase_three.evidence.model_attempts),
        "curve_model_reporting_policy": phase_three.evidence.reporting_policy,
        "credible_curve_model_count": len(phase_three.evidence.credible_attempts),
        "selected_curve_model_count": len(phase_three.evidence.selected_attempts),
        "baseline_response_metrics_enabled": config.raw["modeling"][
            "allow_baseline_response_metrics"
        ],
        "curve_feature_row_count": len(phase_three.evidence.curve_rows),
        "series_evidence_summary": {
            "row_count": len(series_evidence_rows),
            "evidence_status_counts": _field_counts(series_evidence_rows, "evidence_status"),
            "evidence_strength_counts": _field_counts(series_evidence_rows, "evidence_strength"),
            "maximum_proximity_status_counts": _field_counts(series_evidence_rows, "maximum_proximity_status"),
            "target_yield_status_counts": _field_counts(series_evidence_rows, "target_yield_status"),
            "unsupported_reason_counts": _unsupported_reason_counts(series_evidence_rows),
        },
        "descriptive_summary": _descriptive_summary_manifest(descriptive_summary_rows),
        "management_system_proximity_summary": {
            "estimand_version": "ANA-15-option-a-v1",
            "row_count": len(phase_four.management_system_proximity),
            "status_counts": dict(sorted(Counter(row.status for row in phase_four.management_system_proximity).items())),
            "target_gap_status_counts": dict(
                sorted(Counter(row.target_gap_status for row in phase_four.management_system_proximity).items())
            ),
            "dataset_version_ids": sorted(
                {
                    row.dataset_version_id
                    for row in phase_four.management_system_proximity
                    if row.dataset_version_id is not None
                }
            ),
            "dataset_membership_sha256": sorted(
                {
                    row.dataset_membership_sha256
                    for row in phase_four.management_system_proximity
                    if row.dataset_membership_sha256 is not None
                }
            ),
            "n_rate_unit": "kg N/ha",
            "yield_unit": "t/ha",
            "target_population": "resolved_same_response_series",
            "maximum_gap_direction": "supported_maximum_minus_system_yield",
            "target_gap_direction": "target_yield_minus_actual_rcm_yield",
        },
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
    report_sections = _report_sections(
        config,
        phase_two,
        phase_three,
        phase_four,
        series_evidence_rows=series_evidence_rows,
        descriptive_summary_rows=descriptive_summary_rows,
        hypothesis_snapshot=hypothesis_snapshot,
    )
    def write_run_log(stage_root: Path) -> tuple[Path, ...]:
        run_log.info("release_package_ready", release_target=target)
        log_path = run_log.write_jsonl(stage_root)
        manifest["logging"]["record_count"] = len(run_log.records)
        return (log_path,)

    stage_writers = (
        _logged_stage_writer(
            run_log,
            "figures",
            _figure_stage_writer(config, phase_three, series_evidence_rows),
        ),
        _logged_stage_writer(
            run_log,
            "r_stages",
            _r_stage_writer(
                config,
                phase_three,
                phase_four,
                r_stage_statuses,
                r_stage_result_rows,
            ),
        ),
        _logged_stage_writer(
            run_log,
            "multiplicity_reconciliation",
            _multiplicity_reconciliation_stage_writer(
                phase_four,
                r_stage_statuses,
                r_stage_result_rows,
                multiplicity_reconciliation,
                manifest,
                report_sections,
            ),
        ),
        _logged_stage_writer(
            run_log,
            "terminal_statuses",
            _terminal_status_stage_writer(
                phase_three,
                phase_four,
                r_stage_statuses,
                multiplicity_reconciliation,
                manifest,
                report_sections,
            ),
        ),
        write_run_log,
    )
    try:
        package = write_release_package(
            target,
            tables=_table_artifacts(
                phase_two,
                phase_three,
                phase_four,
                series_evidence_rows=series_evidence_rows,
                descriptive_summary_rows=descriptive_summary_rows,
            ),
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
