"""Orchestrator for the descriptive-statistics companion recipe.

Phase order is deliberate and mirrors the sibling recipe: read and hash-bind the
inputs, compute every profile in memory, then stage → verify → promote so the
published bundle is never observed half-written. The input and implementation
snapshot is taken before analysis and re-taken after writing; a mismatch aborts
the promotion, which is what catches another session editing a module or a
source underneath a run in progress.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from typing import Any, Callable
import uuid

import pandas as pd

from ..analysis.descriptive_statistics.agronomic import analyze_agronomic
from ..analysis.descriptive_statistics.config import (
    DescriptiveStatisticsConfig,
    RecipeConfigError,
    load_recipe_config,
)
from ..analysis.descriptive_statistics.sources import (
    LoadedSources,
    load_profiled_sources,
)
from ..analysis.descriptive_statistics import contracts
from ..data.config import load_config
from ..data.provenance import sha256_file


@dataclass(frozen=True)
class DescriptiveStatisticsResult:
    status: str
    mode: str
    output_root: Path | None
    profiled_sources: int
    observations: int
    table_count: int
    figure_count: int
    artifact_count: int
    restricted_artifact_count: int
    skipped_figures: tuple[str, ...]


# Every file whose content can change what this recipe emits. Hashing them into
# the manifest makes the bundle reproducible from a known implementation, and
# comparing them before and after the run detects a concurrent edit.
_IMPLEMENTATION_RELATIVE_PATHS = (
    "descriptive_statistics.sh",
    "descriptive_statisticsCONFIG.toml",
    "modules/descriptive_statistics_pipeline.py",
    "modules/n_response_curve/pipeline/descriptive_statistics_pipeline.py",
    "modules/n_response_curve/analysis/descriptive_statistics/contracts.py",
    "modules/n_response_curve/analysis/descriptive_statistics/config.py",
    "modules/n_response_curve/analysis/descriptive_statistics/sources.py",
    "modules/n_response_curve/analysis/descriptive_statistics/agronomic.py",
    "modules/n_response_curve/reporting/descriptive_statistics.py",
    "modules/n_response_curve/reporting/descriptive_statistics_figures.py",
)


def _implementation_paths(config: DescriptiveStatisticsConfig) -> tuple[Path, ...]:
    paths = tuple(
        (config.project_root / relative).resolve()
        for relative in _IMPLEMENTATION_RELATIVE_PATHS
    )
    missing = [path for path in paths if path.is_symlink() or not path.is_file()]
    if missing:
        raise RecipeConfigError(f"Implementation file is missing: {missing[0]}")
    return paths


def _snapshot(paths: tuple[Path, ...], root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in paths:
        try:
            key = path.relative_to(root).as_posix()
        except ValueError:
            key = path.as_posix()
        snapshot[key] = sha256_file(path)
    return snapshot


def _input_snapshot(
    config: DescriptiveStatisticsConfig, loaded: LoadedSources
) -> dict[str, str]:
    paths = (
        config.config_path,
        config.base_config_path,
        *(source.source_path for source in loaded.sources),
        *_implementation_paths(config),
    )
    return _snapshot(paths, config.project_root)


def _compute_tables(
    loaded: LoadedSources, config: DescriptiveStatisticsConfig
) -> dict[str, pd.DataFrame]:
    """Run every profile and reject a name that is not in the frozen contract."""

    tables: dict[str, pd.DataFrame] = {}
    for analyze in (analyze_agronomic,):
        produced = analyze(loaded, config)
        for name, frame in produced.items():
            if name in tables:
                raise RecipeConfigError(
                    f"Table {name!r} was produced by more than one analysis module"
                )
            tables[name] = contracts.conform_table(name, frame)
    missing = [
        spec.name
        for spec in contracts.TABLE_SPECS
        if spec.name not in tables and not spec.optional
    ]
    if missing:
        raise RecipeConfigError(f"Declared tables were not produced: {missing}")
    return tables


def _promote_stage(stage: Path, target: Path, *, overwrite: bool) -> None:
    if target.is_symlink():
        raise RecipeConfigError(f"Output root is a symlink: {target}")
    backup: Path | None = None
    if target.exists():
        if not overwrite:
            raise RecipeConfigError(
                f"Output already exists and overwrite=false: {target}"
            )
        backup = target.with_name(f".{target.name}.backup.{uuid.uuid4().hex}")
        target.replace(backup)
    try:
        stage.replace(target)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            backup.replace(target)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def run_descriptive_statistics(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    base_config_loader: Callable[..., Any] = load_config,
) -> DescriptiveStatisticsResult:
    """Validate or execute the descriptive statistical profile."""

    path = Path(config_path).expanduser().resolve()
    root = Path(project_root or path.parent).expanduser().resolve()
    config = load_recipe_config(path, project_root=root, check_files=True)
    loaded = load_profiled_sources(config, base_config_loader=base_config_loader)
    # Proves the implementation set is complete before any work is done, so a
    # missing module fails in validate rather than halfway through a write.
    implementation_paths = _implementation_paths(config)

    total_rows = sum(source.data_row_count for source in loaded.sources)
    if not config.writes_outputs:
        return DescriptiveStatisticsResult(
            status="validated",
            mode=config.mode,
            output_root=None,
            profiled_sources=len(loaded.sources),
            observations=total_rows,
            table_count=0,
            figure_count=0,
            artifact_count=0,
            restricted_artifact_count=0,
            skipped_figures=(),
        )

    target = config.output_root
    if target.is_symlink():
        raise RecipeConfigError(f"Output root is a symlink: {target}")
    if target.exists() and not config.overwrite:
        raise RecipeConfigError(
            f"Output already exists and overwrite=false: {target}"
        )

    snapshot_before = _input_snapshot(config, loaded)
    implementation_sha256 = _snapshot(implementation_paths, config.project_root)
    tables = _compute_tables(loaded, config)

    # Deferred until writing mode so validate stays free of Matplotlib
    # font-cache and backend side effects.
    from ..reporting.descriptive_statistics import (
        verify_profile_bundle,
        verify_replacement_profile_bundle,
        write_profile_bundle,
    )

    # Verify what is about to be replaced. A published bundle that no longer
    # matches its own checksums has been altered since it was written, and
    # silently overwriting it would destroy the only evidence of that. The
    # operator resolves it by moving the altered bundle aside deliberately.
    if target.exists():
        try:
            verify_replacement_profile_bundle(target)
        except Exception as exc:
            raise RecipeConfigError(
                f"The existing bundle at {target} does not verify ({exc}); it was "
                "altered after it was written. Move it aside deliberately before "
                "regenerating, so the alteration is not silently overwritten."
            ) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(f".{target.name}.staging.{uuid.uuid4().hex}")
    try:
        written = write_profile_bundle(
            stage,
            config=config,
            loaded=loaded,
            tables=tables,
            implementation_sha256=implementation_sha256,
        )
        verify_profile_bundle(stage)
        if _input_snapshot(config, loaded) != snapshot_before:
            raise RecipeConfigError(
                "An input or implementation file changed during the run; "
                "the bundle was discarded rather than published"
            )
        _promote_stage(stage, target, overwrite=config.overwrite)
        verified = verify_profile_bundle(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)

    # One count per source, on the harmonized (N rate, yield) basis every
    # agronomic table is built from — the same number the retired cross-cut
    # comparability table used to report.
    profiled = tables["nitrogen_rate_profile"]["observation_count"]
    observations = int(pd.to_numeric(profiled, errors="coerce").fillna(0).sum())
    return DescriptiveStatisticsResult(
        status="completed",
        mode=config.mode,
        output_root=target,
        profiled_sources=len(loaded.sources),
        observations=observations,
        table_count=verified.table_count,
        figure_count=verified.figure_count,
        artifact_count=verified.artifact_count,
        restricted_artifact_count=verified.restricted_artifact_count,
        skipped_figures=written.skipped_figures,
    )


def main(
    argv: list[str] | None = None,
    *,
    project_root: str | Path | None = None,
) -> int:
    root = (
        Path(project_root or Path(__file__).resolve().parents[3])
        .expanduser()
        .resolve()
    )
    parser = argparse.ArgumentParser(
        prog="descriptive_statistics",
        description="Profile every registered source dataset.",
    )
    parser.add_argument(
        "--config",
        default=str(root / "descriptive_statisticsCONFIG.toml"),
    )
    args = parser.parse_args(argv)
    try:
        result = run_descriptive_statistics(args.config, project_root=root)
    except Exception as exc:  # noqa: BLE001 - the launcher reports the reason
        print(f"descriptive_statistics_error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"status={result.status}")
    print(f"mode={result.mode}")
    print(f"profiled_sources={result.profiled_sources}")
    print(f"observations={result.observations}")
    print(f"table_count={result.table_count}")
    print(f"figure_count={result.figure_count}")
    print(f"artifact_count={result.artifact_count}")
    print(f"restricted_artifact_count={result.restricted_artifact_count}")
    print(
        "skipped_figures="
        + (",".join(result.skipped_figures) if result.skipped_figures else "none")
    )
    print(
        "output_root="
        + (str(result.output_root) if result.output_root is not None else "none")
    )
    return 0
