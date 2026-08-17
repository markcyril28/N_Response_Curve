from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import shutil
import sys
from typing import Any, Callable
import uuid

from ..analysis.grain_yield_response.descriptive import (
    analyze_descriptive,
    analyze_raw_sensitivity,
    bootstrap_pooled_uncertainty,
)
from ..analysis.grain_yield_response.factor_support import (
    analyze_factor_support,
    audit_factor_evidence,
    audit_factor_redundancy,
    screen_factors_beyond_series,
)
from ..analysis.grain_yield_response.series_covariates import (
    DERIVED_CATEGORICAL_FACTORS,
    DERIVED_NUMERIC_FACTORS,
    SeriesCovariateResults,
    derive_series_covariates,
    screen_series_slope_modifiers,
)
from ..analysis.grain_yield_response.heterogeneity import (
    analyze_heterogeneity,
    preflight_mixed_model_engine,
    run_mixed_model,
)
from ..analysis.grain_yield_response.config import (
    GrainYieldResponseConfig,
    RecipeConfigError,
    load_recipe_config,
)
from ..analysis.grain_yield_response.population import (
    GovernedPopulation,
    RawFinitePopulation,
    load_governed_population,
    load_raw_finite_population,
    sha256_file,
)
from ..data.config import load_config


@dataclass(frozen=True)
class GrainYieldResponsePipelineResult:
    status: str
    mode: str
    output_root: Path | None
    observations: int
    series_count: int
    artifact_count: int
    source_release_verification_status: str


def _verify_source_release(
    config: GrainYieldResponseConfig,
) -> tuple[str, str]:
    """Run the repository's current strict verifier and preserve any failure."""

    from ..reporting.release import ReportingError, verify_release_package

    try:
        verify_release_package(config.release_package)
    except (ReportingError, OSError, RuntimeError, ValueError) as exc:
        detail = f"{type(exc).__name__}: {exc}"[:1000]
        if config.release_verification_policy == "required":
            raise RecipeConfigError(
                f"Strict source-release verification failed: {detail}"
            ) from exc
        return "failed_recorded", detail
    return "verified", ""


def _load_inputs(
    config: GrainYieldResponseConfig,
    *,
    base_config_loader: Callable[..., Any],
) -> tuple[Any, GovernedPopulation, RawFinitePopulation]:
    base_config = base_config_loader(
        config.base_config_path,
        project_root=config.project_root,
        check_files=True,
        preflight_engines=False,
    )
    population = load_governed_population(config)
    release_status, release_detail = _verify_source_release(config)
    population = replace(
        population,
        source_release_verification_status=release_status,
        source_release_verification_detail=release_detail,
    )
    # The recipe is source-hash-bound even when the raw sensitivity is disabled.
    # Reading this frame during validation proves the registered source identity,
    # positional schema, yield-unit consistency, and before/after stability.
    raw_population = load_raw_finite_population(config, base_config=base_config)
    preflight_mixed_model_engine(config)
    return base_config, population, raw_population


def _implementation_paths(config: GrainYieldResponseConfig) -> tuple[Path, ...]:
    relative_paths = (
        "grain_yield_response.sh",
        "modules/grain_yield_response_pipeline.py",
        "modules/n_response_curve/analysis/grain_yield_response/config.py",
        "modules/n_response_curve/analysis/grain_yield_response/population.py",
        "modules/n_response_curve/analysis/grain_yield_response/descriptive.py",
        "modules/n_response_curve/analysis/grain_yield_response/heterogeneity.py",
        "modules/n_response_curve/analysis/grain_yield_response/factor_support.py",
        "modules/n_response_curve/analysis/grain_yield_response/series_covariates.py",
        "modules/n_response_curve/reporting/grain_yield_response.py",
        "modules/n_response_curve/pipeline/grain_yield_response_pipeline.py",
        "modules/n_response_curve/analysis/stages/grain_yield_response_mixed_models.R",
    )
    paths = tuple((config.project_root / relative).resolve() for relative in relative_paths)
    missing = [path for path in paths if path.is_symlink() or not path.is_file()]
    if missing:
        raise RecipeConfigError(f"Implementation file is missing: {missing[0]}")
    return paths


def _snapshot_paths(paths: tuple[Path, ...], root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in paths
    }


def _snapshot_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _input_snapshot(config: GrainYieldResponseConfig, base_config: Any) -> dict[str, str]:
    source_path = Path(base_config.paths["core_source_csv"]).resolve()
    paths = (
        config.config_path,
        config.base_config_path,
        source_path,
        config.release_package / "run_manifest.json",
        config.release_package / "tables/quality/analysis_eligibility_ledger.csv",
        config.release_package / "CHECKSUMS.sha256",
        *_implementation_paths(config),
    )
    return _snapshot_paths(tuple(paths), config.project_root)


def _recover_interrupted_diagnostic_publication(
    target: Path,
    *,
    verifier: Callable[[Path], Any] | None = None,
) -> None:
    """Recover a verified host stage/backup left by an interrupted promotion."""
    if verifier is None:
        from ..reporting.grain_yield_response import (
            verify_replaceable_diagnostic_container,
        )

        verifier = verify_replaceable_diagnostic_container

    prefixes = (f".{target.name}.staging.", f".{target.name}.backup.")
    residues: list[Path] = []
    if target.parent.is_dir():
        for child in target.parent.iterdir():
            if not child.name.startswith(prefixes):
                continue
            if child.is_symlink() or not child.is_dir():
                raise RecipeConfigError(f"Unsafe diagnostic publication residue: {child}")
            residues.append(child)
    residues.sort(key=lambda path: path.name)

    if target.exists():
        verifier(target)
        for residue in residues:
            shutil.rmtree(residue)
        return
    if not residues:
        return

    valid_stages: list[Path] = []
    valid_backups: list[Path] = []
    for candidate in residues:
        try:
            verifier(candidate)
        except Exception:
            continue
        if candidate.name.startswith(f".{target.name}.staging."):
            valid_stages.append(candidate)
        else:
            valid_backups.append(candidate)

    if len(valid_stages) > 1 or (not valid_stages and len(valid_backups) > 1):
        raise RecipeConfigError(
            f"Ambiguous interrupted diagnostic publication residues for {target}"
        )
    selected = valid_stages[0] if valid_stages else (
        valid_backups[0] if valid_backups else None
    )
    if selected is None:
        raise RecipeConfigError(
            f"No valid diagnostic container can be recovered for {target}"
        )
    selected.replace(target)
    verifier(target)
    for residue in residues:
        if residue.exists():
            shutil.rmtree(residue)


def _promote_stage(
    stage: Path,
    target: Path,
    *,
    overwrite: bool,
) -> None:
    if target.is_symlink():
        raise RecipeConfigError(f"Diagnostic output root is a symlink: {target}")
    backup: Path | None = None
    if target.exists():
        if not overwrite:
            raise RecipeConfigError(
                f"Diagnostic output already exists and overwrite=false: {target}"
            )
        from ..reporting.grain_yield_response import exchange_directories

        try:
            if exchange_directories(stage, target):
                shutil.rmtree(stage)
                return
        except OSError as exc:
            raise RecipeConfigError(
                f"Failed to atomically exchange diagnostic output: {target}"
            ) from exc
        backup = target.with_name(f".{target.name}.backup.{uuid.uuid4().hex}")
        target.replace(backup)
    try:
        stage.replace(target)
    except BaseException:
        try:
            if backup is not None and backup.exists() and not target.exists():
                backup.replace(target)
        except OSError as restore_exc:
            raise RecipeConfigError(
                f"Failed to restore prior diagnostic output: {target}"
            ) from restore_exc
        raise
    if backup is not None:
        shutil.rmtree(backup)


def _run_grain_yield_response_unlocked(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    base_config_loader: Callable[..., Any] = load_config,
    expected_output_root: Path | None = None,
) -> GrainYieldResponsePipelineResult:
    """Validate or execute the governed grain-yield response diagnostics."""

    path = Path(config_path).expanduser().resolve()
    root = Path(project_root or path.parent).expanduser().resolve()
    config = load_recipe_config(path, project_root=root, check_files=True)
    if expected_output_root is not None and config.output_root != expected_output_root:
        raise RecipeConfigError(
            "Diagnostic output root changed while acquiring its publication lock"
        )
    base_config, population, raw_population = _load_inputs(
        config,
        base_config_loader=base_config_loader,
    )
    if config.mode == "validate":
        return GrainYieldResponsePipelineResult(
            status="validated",
            mode=config.mode,
            output_root=None,
            observations=len(population.frame),
            series_count=len(population.series_uids),
            artifact_count=0,
            source_release_verification_status=(
                population.source_release_verification_status
            ),
        )

    target = config.output_root
    target.parent.mkdir(parents=True, exist_ok=True)
    _recover_interrupted_diagnostic_publication(target)
    if target.is_symlink():
        raise RecipeConfigError(f"Diagnostic output root is a symlink: {target}")
    if target.exists() and not config.overwrite:
        raise RecipeConfigError(
            f"Diagnostic output already exists and overwrite=false: {target}"
        )
    input_snapshot_before = _input_snapshot(config, base_config)
    implementation_paths = _implementation_paths(config)
    implementation_sha256 = _snapshot_paths(
        implementation_paths,
        config.project_root,
    )

    descriptive = analyze_descriptive(
        population.frame,
        series_key=config.series_key,
        zero_n_tolerance_kg_ha=config.zero_n_tolerance_kg_ha,
        n_level_tolerance_kg_ha=config.n_level_tolerance_kg_ha,
    )
    bootstrap_uncertainty = bootstrap_pooled_uncertainty(
        population.frame,
        series_key=config.series_key,
        random_seed=config.random_seed,
    )
    heterogeneity = analyze_heterogeneity(
        population.frame,
        series_key=config.series_key,
        study_key=config.study_key,
        trial_key=config.trial_key,
    )
    factors = analyze_factor_support(
        population.frame,
        categorical_factors=config.categorical_factors,
        numeric_factors=config.numeric_factors,
        series_key=config.series_key,
        study_key=config.study_key,
        minimum_level_observations=config.minimum_level_observations,
        minimum_modifier_series=config.minimum_modifier_series,
    )
    series_frame, augmented_frame = derive_series_covariates(
        population.frame,
        series_key=config.series_key,
        study_key=config.study_key,
        trial_key=config.trial_key,
        zero_n_tolerance_kg_ha=config.zero_n_tolerance_kg_ha,
        n_level_tolerance_kg_ha=config.n_level_tolerance_kg_ha,
    )
    series_covariates = SeriesCovariateResults(
        series_frame=series_frame,
        augmented_frame=augmented_frame,
        modifier_screen=screen_series_slope_modifiers(
            series_frame,
            minimum_series=config.minimum_modifier_series,
        ),
        derived_categorical=DERIVED_CATEGORICAL_FACTORS,
        derived_numeric=DERIVED_NUMERIC_FACTORS,
    )
    series_adjusted_screen = screen_factors_beyond_series(
        population.frame,
        categorical_factors=config.categorical_factors,
        numeric_factors=config.numeric_factors,
        series_key=config.series_key,
    )
    redundancy_audit = audit_factor_redundancy(
        population.frame,
        categorical_factors=config.categorical_factors,
        numeric_factors=config.numeric_factors,
    )
    evidence_audit = audit_factor_evidence(
        population.frame,
        evidence_pairs=(
            ("organic_fertilizer_present", "organic_fertilizer"),
            ("biofertilizer_present", "biofertilizer"),
        ),
    )
    mixed_model = run_mixed_model(population.frame, config)
    raw_sensitivity = None
    if config.include_raw_sensitivity:
        raw_sensitivity = {
            "frame": raw_population.frame,
            "summary": analyze_raw_sensitivity(
                raw_population.frame,
                n_level_tolerance_kg_ha=config.n_level_tolerance_kg_ha,
            ),
            "source_nonblank_rows": raw_population.source_nonblank_rows,
            "excluded_nonfinite_pairs": raw_population.excluded_nonfinite_pairs,
            "yield_t_source_count": raw_population.yield_t_source_count,
            "yield_kg_fallback_count": raw_population.yield_kg_fallback_count,
        }

    # Importing pyplot is intentionally deferred until writing mode so validate
    # remains free of Matplotlib cache or font-manager side effects.
    from ..reporting.grain_yield_response import (
        PRIORITIZED_RESPONSE_MODIFIER_ROOT,
        verify_completed_diagnostic_container,
        verify_diagnostic_bundle,
        verify_replaceable_diagnostic_container,
        write_diagnostic_bundle,
    )

    prioritized_extension: Path | None = None
    prioritized_extension_snapshot: dict[str, str] = {}
    if target.exists():
        verify_replaceable_diagnostic_container(target)
        candidate = target / PRIORITIZED_RESPONSE_MODIFIER_ROOT
        if candidate.is_dir():
            prioritized_extension = candidate
            prioritized_extension_snapshot = _snapshot_tree(candidate)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(f".{target.name}.staging.{uuid.uuid4().hex}")
    try:
        write_diagnostic_bundle(
            stage,
            config=config,
            population=population,
            descriptive=descriptive,
            heterogeneity=heterogeneity,
            factors=factors,
            series_covariates=series_covariates,
            series_adjusted_screen=series_adjusted_screen,
            redundancy_audit=redundancy_audit,
            evidence_audit=evidence_audit,
            bootstrap_uncertainty=bootstrap_uncertainty,
            mixed_model=mixed_model,
            raw_sensitivity=raw_sensitivity,
            implementation_sha256=implementation_sha256,
        )
        verify_diagnostic_bundle(stage)
        if prioritized_extension is not None:
            staged_extension = stage / PRIORITIZED_RESPONSE_MODIFIER_ROOT
            staged_extension.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(prioritized_extension, staged_extension)
            if _snapshot_tree(prioritized_extension) != prioritized_extension_snapshot:
                raise RecipeConfigError(
                    "Prioritized response-modifier bundle changed during the diagnostic run"
                )
            if _snapshot_tree(staged_extension) != prioritized_extension_snapshot:
                raise RecipeConfigError(
                    "Prioritized response-modifier bundle was not copied exactly"
                )
            verify_completed_diagnostic_container(stage)
        input_snapshot_after = _input_snapshot(config, base_config)
        if input_snapshot_after != input_snapshot_before:
            raise RecipeConfigError(
                "An input or implementation file changed during the diagnostic run"
            )
        _promote_stage(stage, target, overwrite=config.overwrite)
        verified = verify_completed_diagnostic_container(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    return GrainYieldResponsePipelineResult(
        status="completed",
        mode=config.mode,
        output_root=target,
        observations=len(population.frame),
        series_count=len(population.series_uids),
        artifact_count=verified.artifact_count,
        source_release_verification_status=(
            population.source_release_verification_status
        ),
    )


def run_grain_yield_response(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    base_config_loader: Callable[..., Any] = load_config,
) -> GrainYieldResponsePipelineResult:
    """Validate directly or execute under the shared container writer lock."""
    path = Path(config_path).expanduser().resolve()
    root = Path(project_root or path.parent).expanduser().resolve()
    config = load_recipe_config(path, project_root=root, check_files=True)
    if config.mode == "validate":
        return _run_grain_yield_response_unlocked(
            path,
            project_root=root,
            base_config_loader=base_config_loader,
        )

    from ..reporting.grain_yield_response import (
        diagnostic_container_publication_lock,
    )

    with diagnostic_container_publication_lock(config.output_root):
        return _run_grain_yield_response_unlocked(
            path,
            project_root=root,
            base_config_loader=base_config_loader,
            expected_output_root=config.output_root,
        )


def main(
    argv: list[str] | None = None,
    *,
    project_root: str | Path | None = None,
) -> int:
    root = Path(
        project_root or Path(__file__).resolve().parents[3]
    ).expanduser().resolve()
    parser = argparse.ArgumentParser(
        prog="grain_yield_response",
        description="Run governed grain-yield response diagnostics.",
    )
    parser.add_argument(
        "--config",
        default=str(root / "grain_yield_responseCONFIG.toml"),
    )
    args = parser.parse_args(argv)
    try:
        result = run_grain_yield_response(
            args.config,
            project_root=root,
        )
    except Exception as exc:
        print(f"grain_yield_response_error={exc}", file=sys.stderr)
        return 2
    print(f"status={result.status}")
    print(f"mode={result.mode}")
    print(f"observations={result.observations}")
    print(f"series_count={result.series_count}")
    print(f"artifact_count={result.artifact_count}")
    print(
        "source_release_verification_status="
        f"{result.source_release_verification_status}"
    )
    print(
        "output_root="
        + (str(result.output_root) if result.output_root is not None else "none")
    )
    return 0
