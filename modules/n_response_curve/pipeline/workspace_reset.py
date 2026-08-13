"""Guarded pre-run clearing of the operational log root and the output root.

Both clears are opt-in booleans in ``scriptCONFIG.toml``
(``[logging].clear_log_root_before_run`` and
``[outputs].clear_output_root_before_run``) and both are rejected outright in
``validate`` mode, which must leave the project tree byte-identical.

Clearing removes directory *contents*, never the directory itself, and refuses
any target that is or contains a protected path: the configuration file, every
declared source artifact, the hash-bound policy manifests, and the run-metadata
root that holds the approval artifacts.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from n_response_curve.data.config import ConfigError, ValidatedConfig

# The launcher's logging helper owns these names; see
# modules/n_response_curve/logging/run_logging.sh:nrc_setup_logging.
LOG_SUBDIRECTORIES = ("log_files", "jsonl_logs", "error_warn_logs")


class WorkspaceClearResult(NamedTuple):
    """Directories that were cleared and the entries removed from them."""

    roots: tuple[Path, ...]
    removed: tuple[Path, ...]


def protected_paths(config: ValidatedConfig) -> tuple[Path, ...]:
    """Paths a clear must never remove, contain, or be contained by."""

    guarded = [config.config_path, config.paths["run_metadata_root"]]
    guarded.extend(
        config.paths[key]
        for key in (
            "core_source_csv",
            "source_workbook",
            "source_manifest",
            "source_checksums",
            "schema_evidence",
            "variety_lookup",
        )
    )
    for source in config.sources.values():
        for key in ("data_path", "schema_map", "workbook"):
            value = source.get(key)
            if value:
                guarded.append(config.project_root / str(value))
    for manifest in (
        config.analysis_policy_manifest,
        config.source_data_policy_manifest,
    ):
        if manifest is not None:
            guarded.append(manifest)
    return tuple(dict.fromkeys(path.resolve() for path in guarded))


def clear_directory_contents(
    directory: str | Path,
    *,
    label: str,
    protected: Iterable[Path] = (),
) -> tuple[Path, ...]:
    """Remove everything inside ``directory``, keeping the directory itself.

    A missing directory is not an error and is not created: nothing to clear.
    """

    raw = Path(directory).expanduser()
    if raw.is_symlink():
        raise ConfigError(f"{label} must not be a symlink: {raw}")
    resolved = raw.resolve()
    if resolved.parent == resolved:
        raise ConfigError(f"{label} must not be a filesystem root: {resolved}")
    if not resolved.exists():
        return ()
    if not resolved.is_dir():
        raise ConfigError(f"{label} must be a directory: {resolved}")
    for guarded in protected:
        if guarded == resolved or guarded.is_relative_to(resolved):
            raise ConfigError(
                f"{label} was not cleared because it holds protected content: {guarded}"
            )
    removed: list[Path] = []
    for entry in sorted(resolved.iterdir()):
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        removed.append(entry)
    return tuple(removed)


def clear_log_workspace(
    config: ValidatedConfig,
    *,
    log_root: str | Path,
) -> WorkspaceClearResult | None:
    """Clear the launcher's log subdirectories when the operator enabled it.

    Returns ``None`` when the control is off. Scoped to the three
    subdirectories the logging helper creates rather than the log root itself,
    because ``N_RESPONSE_LOG_ROOT`` may legitimately point outside the project.
    """

    if not bool(config.raw["logging"]["clear_log_root_before_run"]):
        return None
    guarded = protected_paths(config)
    roots: list[Path] = []
    removed: list[Path] = []
    for name in LOG_SUBDIRECTORIES:
        target = Path(log_root).expanduser() / name
        cleared = clear_directory_contents(
            target,
            label=f"[logging].clear_log_root_before_run target {name}",
            protected=guarded,
        )
        if target.exists():
            roots.append(target.resolve())
        removed.extend(cleared)
    return WorkspaceClearResult(tuple(roots), tuple(removed))


def clear_output_workspace(config: ValidatedConfig) -> WorkspaceClearResult | None:
    """Clear the mode-specific release root when the operator enabled it.

    Returns ``None`` when the control is off or the mode writes no outputs.
    Every run writes a new package directory under this root, so the contents
    accumulate across runs; clearing removes all of them, not just the target
    of the current run.
    """

    if not bool(config.raw["outputs"]["clear_output_root_before_run"]):
        return None
    if not config.writes_outputs:
        return None
    root = output_root(config)
    removed = clear_directory_contents(
        root,
        label="[outputs].clear_output_root_before_run target",
        protected=protected_paths(config),
    )
    return WorkspaceClearResult((root.resolve(),), removed)


def output_root(config: ValidatedConfig) -> Path:
    """The release root the configured mode writes its package into."""

    if config.run_mode == "test":
        return config.paths["test_output_root"]
    if config.run_mode == "full":
        return config.paths["reports_root"]
    raise ConfigError("Validate mode does not have an output root")


__all__ = [
    "LOG_SUBDIRECTORIES",
    "WorkspaceClearResult",
    "clear_directory_contents",
    "clear_log_workspace",
    "clear_output_workspace",
    "output_root",
    "protected_paths",
]
