"""Organized read-only workspace views of one verified release package.

The promoted package under ``[paths].reports_root`` stays the single
authoritative, checksummed deliverable. Full runs additionally project a
deterministic subset of its *already verified* artifacts into the documented
``WF/03``-``WF/04`` roots so downstream readers find QC and curve
outputs where the workflow documents them.

Every projected file is an exact byte copy hashed against the source package's
checksum ledger -- never a symlink, a hardlink, or a recomputed table. Nothing
here writes to the release package; it is opened read-only and re-verified with
the authoritative completed-release verifier before staging and again before the
transaction commits.

Each view target carries its own ``WORKSPACE_VIEW_MANIFEST.json`` and
``CHECKSUMS.sha256`` binding the managed projection to the source release path,
run identity, and checksum ledger. The curves view may additionally carry the
three separately produced dataset directories at its root; their exact boundary
and filesystem safety are verified, while their bytes remain owned by those
producers.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Iterable, Mapping, Sequence

from n_response_curve.data.provenance import sha256_file
from n_response_curve.reporting.release import (
    ReleasePackage,
    ReportingError,
    verify_completed_release_package,
)


class WorkspaceOutputError(ReportingError):
    """Unsafe, incomplete, or unbound workspace-view request."""


WORKSPACE_VIEW_MANIFEST_NAME = "WORKSPACE_VIEW_MANIFEST.json"
WORKSPACE_CHECKSUMS_NAME = "CHECKSUMS.sha256"
WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION = "workspace-view-manifest-v4"
_LEGACY_WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION_V3 = "workspace-view-manifest-v3"
_LEGACY_WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION_V2 = "workspace-view-manifest-v2"
_LEGACY_WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION_V1 = "workspace-view-manifest-v1"
_STANDALONE_EXTENSION_ROOTS_V4 = (
    "literature_extracted_dataset",
    "ltcce",
    "ph_combined_nopt_rcm",
)
_SOURCE_DATASET_EXTENSION_V3 = "source_dataset"
_SOURCE_DATASET_EXTENSION_V2 = "overlay/source_dataset"
_SOURCE_DATASET_EXTENSION_V1 = "figures/overlay/source_dataset"

#: Ordered so plans, promotion, logging, and reporting all traverse identically.
WORKSPACE_VIEW_CATEGORIES = (
    "qc",
    "curves",
)

_CATEGORY_ROOT_KEYS = {
    "qc": "qc_root",
    "curves": "curves_root",
}

_STAGE_SUFFIX = ".workspace-stage"
_BACKUP_SUFFIX = ".workspace-prior"
_TRANSACTION_NAME = ".workspace-output-transaction.json"
_TRANSACTION_SCHEMA_VERSION = "workspace-output-transaction-v4"
_OWNERSHIP_REGISTRY_PREFIX = ".workspace-output-ownership"
_OWNERSHIP_REGISTRY_SCHEMA_VERSION = "workspace-output-ownership-v1"
_PROJECT_LOCK_NAME = ".workspace-outputs.lock"

#: The manifest fields that bind a view to exactly one source release package.
_SOURCE_BINDING_FIELDS = (
    "source_release_path",
    "source_run_identity_sha256",
    "source_checksum_ledger_sha256",
    "source_run_manifest_sha256",
)

# Governance, transcripts, reports, and root-level package files stay only in the
# release root. They are listed rather than inferred so a new root-level package
# artifact fails closed instead of silently vanishing from every view.
_RETAINED_ONLY_PATHS = frozenset(
    {
        "CHECKSUMS.sha256",
        "run_manifest.json",
        "report.md",
        "report.pdf",
        "replacement_record.json",
    }
)
# ``figures/overlay/`` was retired from projection on 2026-09-01: the configured
# descriptive subsets are no longer generated ([custom_overlays].enabled = false)
# and the per-source overlay itself is superseded by the standalone generators.
# It is matched here, *before* _PREFIX_RULES, so the surviving ``figures/`` rule
# cannot silently relocate it into the curves view root instead.
# ``tables/dataset/`` and ``tables/derived/`` were retired from projection on
# 2026-09-03 together with the ``analysis_ready`` view: the controlled package
# under ``[paths].reports_root`` stays their single governed home. They are
# matched here so both trees remain *recognized* -- a new dataset or derived
# path still fails closed rather than vanishing -- while no view projects them.
_RETAINED_ONLY_PREFIXES = (
    "logs/",
    "figures/overlay/",
    "tables/dataset/",
    "tables/derived/",
)

# Explanatory ledgers are recognized so a *new* ledger path still fails closed,
# but they are retained only in the release root: no view projects them.
_RETAINED_ONLY_LEDGERS = frozenset(
    {
        "ledgers/multiplicity_reconciliation.json",
        "ledgers/claim_classification.json",
        "ledgers/analysis_terminal_statuses.json",
    }
)
_QC_LEDGERS = frozenset({"ledgers/review_issue_ledger.json"})

# (source prefix, category, target prefix) applied in order; first match wins.
_PREFIX_RULES = (
    ("tables/quality/", "qc", "quality/"),
    ("tables/curves/", "curves", ""),
    ("figures/", "curves", ""),
)


@dataclass(frozen=True)
class WorkspaceArtifact:
    """One package artifact projected into exactly one view."""

    source_relative_path: str
    target_relative_path: str
    sha256: str


@dataclass(frozen=True)
class PlannedWorkspaceView:
    """The deterministic artifact set one view will contain."""

    category: str
    root_key: str
    artifacts: tuple[WorkspaceArtifact, ...]


@dataclass(frozen=True)
class WorkspaceView:
    """A materialized or verified view target and its measured contents."""

    category: str
    root_key: str
    target_path: Path
    artifacts: tuple[WorkspaceArtifact, ...]
    artifact_count: int
    total_bytes: int

    @property
    def manifest_path(self) -> Path:
        return self.target_path / WORKSPACE_VIEW_MANIFEST_NAME

    @property
    def checksums_path(self) -> Path:
        return self.target_path / WORKSPACE_CHECKSUMS_NAME


@dataclass(frozen=True)
class WorkspaceOutputs:
    """The full set of views bound to one source release package."""

    views: tuple[WorkspaceView, ...]
    reused: bool
    source_release_path: Path
    source_run_identity_sha256: str
    source_checksum_ledger_sha256: str

    @property
    def artifact_count(self) -> int:
        return sum(view.artifact_count for view in self.views)

    @property
    def total_bytes(self) -> int:
        return sum(view.total_bytes for view in self.views)


# --------------------------------------------------------------------------
# Injectable filesystem seams. Kept module-level so transactional failure
# handling can be exercised without monkeypatching os/shutil globally.
# --------------------------------------------------------------------------


def _replace_directory(source: Path, destination: Path) -> None:
    """Atomically move ``source`` onto ``destination`` within one filesystem."""

    os.replace(source, destination)


def _copy_artifact(source: Path, destination: Path) -> int:
    """Copy file bytes only -- no metadata, no links -- and report the size."""

    shutil.copyfile(source, destination)
    return destination.stat().st_size


def _fsync_file(path: Path) -> None:
    """Force one regular file's bytes to stable storage."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_directory(path: Path) -> None:
    """Force one directory's entry list to stable storage."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_mkdir(path: Path, *, exist_ok: bool = False) -> None:
    """Create ``path`` and durably link every directory this call brings into being.

    ``mkdir`` only dirties the parent's entry list. Syncing the new directory
    itself is not enough: if the *parent* entry never reaches the device, a crash
    leaves the directory unreachable, and a transaction journal naming it
    authorizes recovery of paths that do not exist. Each newly created level is
    therefore made durable from the outside in, shallowest first, so the tree is
    always reachable as far as it has been committed.
    """

    missing: list[Path] = []
    probe = path
    while not probe.exists():
        missing.append(probe)
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    path.mkdir(parents=True, exist_ok=exist_ok)
    for created in reversed(missing):
        _sync_directory(created.parent)


def _fsync_tree(root: Path) -> None:
    """Durably sync every regular file and directory under ``root``.

    Files first, then directories deepest-first, so a crash can never leave a
    directory entry pointing at bytes that never reached the device.
    """

    directories: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        directories.append(directory)
        with os.scandir(directory) as entries:
            for entry in entries:
                child = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(child)
                elif entry.is_file(follow_symlinks=False):
                    _fsync_file(child)
    for directory in sorted(
        directories, key=lambda item: len(item.parts), reverse=True
    ):
        _sync_directory(directory)


def _promote_directory(source: Path, destination: Path) -> None:
    """Rename one directory into place and durably record both parent entries."""

    _replace_directory(source, destination)
    _sync_directory(destination.parent)
    if source.parent != destination.parent:
        _sync_directory(source.parent)


def _remove_tree(path: Path) -> None:
    """Remove a path if it exists and durably record the parent entry change."""

    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        return
    _sync_directory(path.parent)


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def _check_relative_path(relative: str, *, where: str) -> None:
    if not relative or relative != relative.strip():
        raise WorkspaceOutputError(f"{where} has an empty or padded path: {relative!r}")
    if "\\" in relative or "\x00" in relative:
        raise WorkspaceOutputError(
            f"{where} has an unsafe path separator: {relative!r}"
        )
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.as_posix() != relative:
        raise WorkspaceOutputError(
            f"{where} is not a normalized relative path: {relative!r}"
        )
    if any(part in ("", ".", "..") for part in candidate.parts):
        raise WorkspaceOutputError(f"{where} escapes its root: {relative!r}")


def _classify(relative: str) -> tuple[str, str] | None:
    """Map one package-relative path to ``(category, view-relative path)``.

    ``None`` means the artifact is deliberately retained only in the release
    root. An unrecognized path is an error: a package layout this module has
    not been taught about must never be projected by guesswork.
    """

    if relative in _RETAINED_ONLY_PATHS or relative.startswith(_RETAINED_ONLY_PREFIXES):
        return None
    for prefix, category, target_prefix in _PREFIX_RULES:
        if relative.startswith(prefix):
            return category, target_prefix + relative[len(prefix) :]
    if relative in _QC_LEDGERS:
        return "qc", relative
    if relative in _RETAINED_ONLY_LEDGERS:
        return None
    # Analysis tables -- comparison tables included -- are recognized so a *new*
    # analysis path still fails closed, but no view projects any of them.
    if relative.startswith("tables/analysis/"):
        return None
    raise WorkspaceOutputError(
        "Release package contains an unsupported path for workspace projection: "
        f"{relative}"
    )


def plan_workspace_views(
    package: ReleasePackage,
) -> dict[str, PlannedWorkspaceView]:
    """Resolve the deterministic, collision-free mapping for every view.

    Refuses traversal, duplicated sources, case-insensitive target aliases,
    file/directory-prefix collisions, and any package path the mapping does not
    cover.
    """

    grouped: dict[str, list[WorkspaceArtifact]] = {
        category: [] for category in WORKSPACE_VIEW_CATEGORIES
    }
    claimed_sources: set[str] = set()
    claimed_targets: dict[str, dict[tuple[str, ...], str]] = {
        category: {} for category in WORKSPACE_VIEW_CATEGORIES
    }
    for relative in sorted(package.artifact_sha256):
        _check_relative_path(relative, where="Release checksum ledger entry")
        classified = _classify(relative)
        if classified is None:
            continue
        category, target_relative = classified
        _check_relative_path(target_relative, where="Workspace view target")
        if relative in claimed_sources:
            raise WorkspaceOutputError(f"Package artifact mapped twice: {relative}")
        target_key = tuple(part.casefold() for part in Path(target_relative).parts)
        for claimed_key, claimed_spelling in claimed_targets[category].items():
            shared = min(len(target_key), len(claimed_key))
            if target_key[:shared] == claimed_key[:shared]:
                raise WorkspaceOutputError(
                    "Workspace view collision in "
                    f"{category}: {claimed_spelling} / {target_relative}"
                )
        claimed_sources.add(relative)
        claimed_targets[category][target_key] = target_relative
        grouped[category].append(
            WorkspaceArtifact(
                source_relative_path=relative,
                target_relative_path=target_relative,
                sha256=str(package.artifact_sha256[relative]),
            )
        )
    return {
        category: PlannedWorkspaceView(
            category=category,
            root_key=_CATEGORY_ROOT_KEYS[category],
            artifacts=tuple(
                sorted(grouped[category], key=lambda item: item.target_relative_path)
            ),
        )
        for category in WORKSPACE_VIEW_CATEGORIES
    }


# --------------------------------------------------------------------------
# Manifest and ledger
# --------------------------------------------------------------------------


def _check_view_name(view_name: str) -> str:
    """The view directory is one safe component, shared by every root."""

    if (
        not view_name
        or view_name in (".", "..")
        or "/" in view_name
        or "\\" in view_name
        or view_name != view_name.strip()
        or view_name.startswith(".")
    ):
        raise WorkspaceOutputError(f"Unsafe workspace view name: {view_name!r}")
    return view_name


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _logical_path(path: str | Path) -> Path:
    """Resolve every component *except* the final name.

    A symlinked leaf is rejected explicitly elsewhere; resolving through it here
    would let a planted link relocate a target and defeat the disjointness
    checks. Resolving the parent still catches a symlinked ancestor aliasing two
    configured roots onto one real directory.
    """

    candidate = Path(path)
    parent = candidate.parent
    if parent == candidate:
        return candidate.resolve()
    return parent.resolve() / candidate.name


def _release_run_identity(package: ReleasePackage) -> tuple[str, str]:
    """Read the source package's run identity and run id from its manifest."""

    try:
        payload = json.loads(package.manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkspaceOutputError("Release run manifest could not be read") from exc
    if not isinstance(payload, Mapping):
        raise WorkspaceOutputError("Release run manifest must be a JSON object")
    identity = payload.get("run_identity_sha256")
    if not isinstance(identity, str) or len(identity) != 64:
        raise WorkspaceOutputError("Release run manifest has no usable run identity")
    run_id = payload.get("run_id")
    return identity, run_id if isinstance(run_id, str) else ""


def _render_manifest(
    *,
    category: str,
    root_key: str,
    view_name: str,
    source_release_path: str,
    source_run_id: str,
    source_run_identity_sha256: str,
    source_checksum_ledger_sha256: str,
    source_run_manifest_sha256: str,
    artifacts: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "schema_version": WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION,
        "category": category,
        "root_key": root_key,
        "view_name": view_name,
        "source_release_path": source_release_path,
        "source_run_id": source_run_id,
        "source_run_identity_sha256": source_run_identity_sha256,
        "source_checksum_ledger_sha256": source_checksum_ledger_sha256,
        "source_run_manifest_sha256": source_run_manifest_sha256,
        "artifact_count": len(artifacts),
        "total_bytes": sum(int(item["bytes"]) for item in artifacts),
        "artifacts": list(artifacts),
    }
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _render_checksums(entries: Mapping[str, str]) -> str:
    return "".join(f"{entries[relative]}  {relative}\n" for relative in sorted(entries))


def _parse_checksums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise WorkspaceOutputError(
            f"Workspace view checksum ledger could not be read: {path}"
        ) from exc
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise WorkspaceOutputError(
                f"Workspace view checksum ledger has an invalid line: {path}"
            ) from exc
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative in entries
        ):
            raise WorkspaceOutputError(
                f"Workspace view checksum ledger is unsafe or malformed: {path}"
            )
        _check_relative_path(relative, where="Workspace view checksum ledger entry")
        entries[relative] = digest
    return entries


def _read_view_manifest(
    target: Path,
    *,
    accepted_schema_versions: frozenset[str] = frozenset(
        {WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION}
    ),
) -> Mapping[str, Any]:
    manifest_path = target / WORKSPACE_VIEW_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise WorkspaceOutputError(
            f"Workspace view manifest is missing: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkspaceOutputError(
            f"Workspace view manifest is not valid JSON: {manifest_path}"
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") not in accepted_schema_versions
    ):
        raise WorkspaceOutputError(
            f"Workspace view manifest has an unrecognized schema: {manifest_path}"
        )
    return payload


def _declared_directory_closure(relatives: Iterable[str]) -> set[str]:
    """Every directory a declared file set requires, and nothing else."""

    closure: set[str] = set()
    for relative in relatives:
        parts = relative.split("/")[:-1]
        for depth in range(1, len(parts) + 1):
            closure.add("/".join(parts[:depth]))
    return closure


def _standalone_extension_prefixes(
    category: str, schema_version: Any
) -> tuple[str, ...]:
    """Return the separately owned dataset roots for one curves schema."""

    if category != "curves":
        return ()
    if schema_version == WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION:
        return _STANDALONE_EXTENSION_ROOTS_V4
    if schema_version == _LEGACY_WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION_V3:
        return (_SOURCE_DATASET_EXTENSION_V3,)
    if schema_version == _LEGACY_WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION_V2:
        return (_SOURCE_DATASET_EXTENSION_V2,)
    if schema_version == _LEGACY_WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION_V1:
        return (_SOURCE_DATASET_EXTENSION_V1,)
    return ()


def _extension_inventory(
    files: Iterable[str],
    directories: Iterable[str],
    *,
    category: str,
    schema_version: Any,
) -> tuple[set[str], set[str]]:
    """Select only entries below the exact separately owned extension roots."""

    prefixes = _standalone_extension_prefixes(category, schema_version)
    if not prefixes:
        return set(), set()
    child_prefixes = tuple(prefix + "/" for prefix in prefixes)
    extension_files = {
        relative for relative in files if relative.startswith(child_prefixes)
    }
    extension_directories = {
        relative
        for relative in directories
        if relative in prefixes or relative.startswith(child_prefixes)
    }
    return extension_files, extension_directories


def _inventory_directory_closure(
    files: Iterable[str], directories: Iterable[str]
) -> set[str]:
    """Directory closure for a validated tree, including intentional empty dirs."""

    listed_directories = set(directories)
    sentinels = {f"{relative}/.workspace-owned" for relative in listed_directories}
    return listed_directories | _declared_directory_closure(set(files) | sentinels)


def _contains_type_bucket(relatives: Iterable[str]) -> bool:
    return any(
        part.casefold() in {"figures", "tables"}
        for relative in relatives
        for part in Path(relative).parts
    )


def _walk_view_entries(target: Path) -> tuple[set[str], set[str]]:
    """Inventory a view with ``lstat`` semantics, refusing every unsafe entry.

    Symlinks, FIFOs, sockets, and devices are rejected outright, and any regular
    file carrying a second link is rejected whatever it is linked to -- a
    paired-source, sibling-source, internal, or external hardlink all present as
    ``st_nlink > 1``. Recursion never follows a link, so a planted directory
    symlink cannot smuggle content in from outside the view.
    """

    files: set[str] = set()
    directories: set[str] = set()
    pending: list[tuple[Path, str]] = [(target, "")]
    while pending:
        directory, prefix = pending.pop()
        try:
            with os.scandir(directory) as entries:
                listing = sorted(entries, key=lambda item: item.name)
        except OSError as exc:
            raise WorkspaceOutputError(
                f"Workspace view directory could not be read: {directory}"
            ) from exc
        for entry in listing:
            relative = f"{prefix}{entry.name}"
            try:
                info = os.lstat(entry.path)
            except OSError as exc:
                raise WorkspaceOutputError(
                    f"Workspace view entry could not be inspected: {target / relative}"
                ) from exc
            mode = info.st_mode
            if stat.S_ISLNK(mode):
                raise WorkspaceOutputError(
                    f"Workspace view contains a symlink: {target / relative}"
                )
            if stat.S_ISDIR(mode):
                directories.add(relative)
                pending.append((Path(entry.path), f"{relative}/"))
                continue
            if not stat.S_ISREG(mode):
                raise WorkspaceOutputError(
                    "Workspace view contains a non-regular filesystem entry: "
                    f"{target / relative}"
                )
            if info.st_nlink > 1:
                raise WorkspaceOutputError(
                    "Workspace view file is a hardlink to another path and is not an "
                    f"independent copy: {target / relative}"
                )
            files.add(relative)
    return files, directories


def verify_workspace_view(target_path: str | Path) -> WorkspaceView:
    """Verify the managed ledger plus any validated standalone curves extension."""

    target = Path(target_path)
    if target.is_symlink() or not target.is_dir():
        raise WorkspaceOutputError(f"Workspace view is missing: {target}")
    manifest = _read_view_manifest(target)
    expected = _parse_checksums(target / WORKSPACE_CHECKSUMS_NAME)
    files, directories = _walk_view_entries(target)
    actual = files - {WORKSPACE_CHECKSUMS_NAME}
    category = manifest.get("category")
    extension_files, extension_directories = _extension_inventory(
        actual,
        directories,
        category=str(category),
        schema_version=manifest.get("schema_version"),
    )
    if actual - extension_files != set(expected):
        raise WorkspaceOutputError(
            f"Workspace view checksum ledger does not cover its contents: {target}"
        )
    allowed_directories = _declared_directory_closure(expected) | _inventory_directory_closure(
        extension_files, extension_directories
    )
    if directories != allowed_directories:
        raise WorkspaceOutputError(
            "Workspace view contains a directory its checksum ledger does not "
            f"account for: {target}"
        )
    for relative, digest in expected.items():
        path = target / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise WorkspaceOutputError(
                f"Workspace view checksum mismatch: {target / relative}"
            )

    declared = manifest.get("artifacts")
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
        raise WorkspaceOutputError(
            f"Workspace view manifest has no artifact list: {target}"
        )
    artifacts: list[WorkspaceArtifact] = []
    total_bytes = 0
    for item in declared:
        if not isinstance(item, Mapping):
            raise WorkspaceOutputError(
                f"Workspace view manifest artifact is invalid: {target}"
            )
        relative = item.get("path")
        source_relative = item.get("source_path")
        digest = item.get("sha256")
        size = item.get("bytes")
        if (
            not isinstance(relative, str)
            or not isinstance(source_relative, str)
            or not isinstance(digest, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise WorkspaceOutputError(
                f"Workspace view manifest artifact is invalid: {target}"
            )
        _check_relative_path(relative, where="Workspace view manifest artifact")
        _check_relative_path(source_relative, where="Workspace view manifest source")
        if expected.get(relative) != digest:
            raise WorkspaceOutputError(
                f"Workspace view manifest disagrees with its checksum ledger: {target / relative}"
            )
        if (target / relative).stat().st_size != size:
            raise WorkspaceOutputError(
                f"Workspace view manifest records the wrong size: {target / relative}"
            )
        total_bytes += size
        artifacts.append(
            WorkspaceArtifact(
                source_relative_path=source_relative,
                target_relative_path=relative,
                sha256=digest,
            )
        )
    if {artifact.target_relative_path for artifact in artifacts} | {
        WORKSPACE_VIEW_MANIFEST_NAME
    } != set(expected):
        raise WorkspaceOutputError(
            f"Workspace view manifest inventory is incomplete: {target}"
        )
    if (
        manifest.get("artifact_count") != len(artifacts)
        or manifest.get("total_bytes") != total_bytes
    ):
        raise WorkspaceOutputError(
            f"Workspace view manifest accounting is wrong: {target}"
        )
    root_key = manifest.get("root_key")
    if category not in WORKSPACE_VIEW_CATEGORIES or root_key != _CATEGORY_ROOT_KEYS.get(
        str(category)
    ):
        raise WorkspaceOutputError(
            f"Workspace view manifest has an unknown category: {target}"
        )
    if category == "curves" and (
        _contains_type_bucket(
            artifact.target_relative_path for artifact in artifacts
        )
        or _contains_type_bucket(directories)
    ):
        raise WorkspaceOutputError(
            "The curves workspace view must mix figures and tables by analytical "
            f"meaning, without figures/ or tables/ directories: {target}"
        )
    return WorkspaceView(
        category=str(category),
        root_key=str(root_key),
        target_path=target,
        artifacts=tuple(artifacts),
        artifact_count=len(artifacts),
        total_bytes=total_bytes,
    )


def _view_targets(config: Any, view_name: str) -> dict[str, Path]:
    """Resolve each category's live target directory.

    ``curves`` is the exception: unlike ``qc``, its root is
    dedicated to this view alone, so the governed ledger lives at
    ``curves_root`` directly rather than under a ``curves_root/view_name``
    subdirectory. The standalone generators' source-named directories
    (``_STANDALONE_EXTENSION_ROOTS_V4``) are unbound siblings there, exactly
    as ``_extension_inventory``/``_copy_standalone_extension`` already allow.
    """

    targets: dict[str, Path] = {}
    for category in WORKSPACE_VIEW_CATEGORIES:
        root_key = _CATEGORY_ROOT_KEYS[category]
        try:
            root = Path(config.paths[root_key])
        except KeyError as exc:
            raise WorkspaceOutputError(f"[paths].{root_key} is not configured") from exc
        targets[category] = root if category == "curves" else root / view_name
    return targets


def verify_workspace_views(
    config: Any,
    *,
    view_name: str,
    expected_package: ReleasePackage | None = None,
) -> tuple[WorkspaceView, ...]:
    """Verify every configured view, optionally requiring one source package."""

    _check_view_name(view_name)
    targets = _view_targets(config, view_name)
    views = tuple(
        verify_workspace_view(targets[category])
        for category in WORKSPACE_VIEW_CATEGORIES
    )
    for category, view in zip(WORKSPACE_VIEW_CATEGORIES, views, strict=True):
        if (
            view.category != category
            or view.root_key != _CATEGORY_ROOT_KEYS[category]
            or view.target_path != targets[category]
        ):
            raise WorkspaceOutputError(
                f"Workspace view at {targets[category]} declares the wrong category"
            )
    set_bindings: set[tuple[str, ...]] = set()
    for view in views:
        manifest = _read_view_manifest(view.target_path)
        if manifest.get("view_name") != view_name:
            raise WorkspaceOutputError(
                f"Workspace view {view.category} declares another view name"
            )
        ownership = _structural_ownership(
            view.target_path,
            category=view.category,
            view_name=view_name,
        )
        # ``run_id`` is optional in a release manifest, so an empty one is
        # legitimate; it still has to be *the same* empty value everywhere.
        run_id = manifest.get("source_run_id")
        if not isinstance(run_id, str):
            raise WorkspaceOutputError(
                f"Workspace view {view.category} has an incomplete source binding"
            )
        set_bindings.add((*ownership.binding, run_id))
    if len(set_bindings) != 1:
        raise WorkspaceOutputError(
            "Configured workspace views are bound to different source releases"
        )
    if expected_package is not None:
        identity, run_id = _release_run_identity(expected_package)
        ledger_digest = sha256_file(expected_package.checksums_path)
        manifest_digest = expected_package.artifact_sha256.get("run_manifest.json")
        if not isinstance(manifest_digest, str) or len(manifest_digest) != 64:
            raise WorkspaceOutputError(
                "Expected release package does not bind its run manifest"
            )
        source_release_path = _project_relative(
            expected_package.target_path,
            getattr(config, "project_root", expected_package.target_path),
        )
        expected_plan = plan_workspace_views(expected_package)
        for category, view in zip(WORKSPACE_VIEW_CATEGORIES, views, strict=True):
            manifest = _read_view_manifest(view.target_path)
            if (
                manifest.get("category") != category
                or manifest.get("root_key") != _CATEGORY_ROOT_KEYS[category]
                or manifest.get("view_name") != view_name
                or manifest.get("source_release_path") != source_release_path
                or manifest.get("source_run_id") != run_id
                or manifest.get("source_run_identity_sha256") != identity
                or manifest.get("source_checksum_ledger_sha256") != ledger_digest
                or manifest.get("source_run_manifest_sha256") != manifest_digest
            ):
                raise WorkspaceOutputError(
                    f"Workspace view {view.category} is bound to a different release package"
                )
            if view.artifacts != expected_plan[category].artifacts:
                raise WorkspaceOutputError(
                    f"Workspace view {category} does not exactly match its release mapping"
                )
            for artifact in expected_plan[category].artifacts:
                source = expected_package.target_path / artifact.source_relative_path
                projected = view.target_path / artifact.target_relative_path
                if source.is_symlink() or not source.is_file():
                    raise WorkspaceOutputError(
                        "Expected release source artifact is missing or unsafe: "
                        f"{artifact.source_relative_path}"
                    )
                if sha256_file(source) != artifact.sha256:
                    raise WorkspaceOutputError(
                        "Expected release source artifact has drifted: "
                        f"{artifact.source_relative_path}"
                    )
                # ``_walk_view_entries`` already refuses every multiply-linked view
                # file, so this only ever fires on a filesystem that misreports
                # ``st_nlink``. Source artifacts are deliberately *not* link-checked:
                # the package may legitimately share inodes internally.
                try:
                    same_file = os.path.samefile(source, projected)
                except OSError as exc:
                    raise WorkspaceOutputError(
                        "Could not prove workspace artifact independence: "
                        f"{artifact.target_relative_path}"
                    ) from exc
                if same_file:
                    raise WorkspaceOutputError(
                        "Workspace artifact is a hardlink to the release package: "
                        f"{artifact.target_relative_path}"
                    )
    return views


# --------------------------------------------------------------------------
# Path-set layout, isolation, and the set-level lock identity
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _WorkspaceLayout:
    """Every path one view-set transaction may touch, validated as disjoint."""

    view_name: str
    release_path: Path
    targets: Mapping[str, Path]
    stages: Mapping[str, Path]
    backups: Mapping[str, Path]
    metadata_root: Path
    journal_path: Path
    journal_temporary_path: Path
    ownership_registry_path: Path
    lock_path: Path
    target_set_fingerprint: str


def _target_set_fingerprint(targets: Mapping[str, Path]) -> str:
    """Canonical identity of the three resolved live targets, order-independent.

    Two configurations naming the same three directories -- however they spell
    them, and whatever ``run_metadata_root`` they declare -- produce the same
    fingerprint and therefore contend on the same lock.
    """

    canonical = "\n".join(
        sorted(
            _logical_path(targets[category]).as_posix()
            for category in WORKSPACE_VIEW_CATEGORIES
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _lock_directory(config: Any, targets: Mapping[str, Path]) -> Path:
    """Return one project-level lock home for every workspace target set.

    Serializing the whole project deliberately trades a little concurrency for
    safety: partially overlapping target sets and a shared journal can never be
    operated on concurrently.
    """

    del targets
    project_root = getattr(config, "project_root", None)
    if project_root is None:
        raise WorkspaceOutputError(
            "Workspace output lock requires a configured project root to bound it"
        )
    root = Path(project_root).resolve()
    return root / "WF"


def _check_workspace_path_isolation(layout: _WorkspaceLayout) -> None:
    """Refuse any overlap between the release, the views, and the bookkeeping.

    Metadata, journal, and lock must live outside the authoritative package and
    outside every live/stage/backup target, and vice versa. This runs before the
    first filesystem write of a run, lock creation included.
    """

    configured_paths = [
        ("release package", layout.release_path),
        ("run metadata root", layout.metadata_root),
        *(
            (f"{category} view target", layout.targets[category])
            for category in WORKSPACE_VIEW_CATEGORIES
        ),
    ]
    for label, path in configured_paths:
        if path.is_symlink():
            raise WorkspaceOutputError(
                f"Workspace {label} is a symlink and cannot be used safely: {path}"
            )

    exclusive: list[tuple[str, Path]] = [
        ("release package", _logical_path(layout.release_path))
    ]
    for category in WORKSPACE_VIEW_CATEGORIES:
        exclusive.append(
            (f"{category} view target", _logical_path(layout.targets[category]))
        )
        exclusive.append(
            (f"{category} stage directory", _logical_path(layout.stages[category]))
        )
        exclusive.append(
            (f"{category} backup directory", _logical_path(layout.backups[category]))
        )
    exclusive.append(("run metadata root", _logical_path(layout.metadata_root)))
    for index, (left_label, left) in enumerate(exclusive):
        for right_label, right in exclusive[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise WorkspaceOutputError(
                    f"Workspace {left_label} overlaps the {right_label}: {left} / {right}"
                )

    metadata_root = _logical_path(layout.metadata_root)
    journal = _logical_path(layout.journal_path)
    temporary = _logical_path(layout.journal_temporary_path)
    registry = _logical_path(layout.ownership_registry_path)
    if (
        journal == temporary
        or registry in (journal, temporary)
        or journal.parent != metadata_root
        or temporary.parent != metadata_root
        or registry.parent != metadata_root
    ):
        raise WorkspaceOutputError(
            "Workspace transaction metadata must live directly under the run metadata "
            f"root: {journal}"
        )

    lock_path = _logical_path(layout.lock_path)
    if lock_path in (journal, temporary, registry):
        raise WorkspaceOutputError(
            f"Workspace output lock collides with the transaction journal: {lock_path}"
        )
    for label, directory in exclusive:
        if lock_path == directory or directory in lock_path.parents:
            raise WorkspaceOutputError(
                f"Workspace output lock path is inside the {label}: {lock_path}"
            )


def _resolve_workspace_layout(
    config: Any, package: ReleasePackage, *, view_name: str
) -> _WorkspaceLayout:
    """Derive and validate every path a run may touch. Writes nothing."""

    _check_view_name(view_name)
    targets = _view_targets(config, view_name)
    try:
        metadata_root = Path(config.paths["run_metadata_root"])
    except KeyError as exc:
        raise WorkspaceOutputError(
            "[paths].run_metadata_root is not configured"
        ) from exc
    journal_path = metadata_root / _TRANSACTION_NAME
    fingerprint = _target_set_fingerprint(targets)
    lock_path = _lock_directory(config, targets) / _PROJECT_LOCK_NAME
    layout = _WorkspaceLayout(
        view_name=view_name,
        release_path=Path(package.target_path),
        targets=dict(targets),
        stages={
            category: targets[category].parent
            / f".{targets[category].name}{_STAGE_SUFFIX}"
            for category in WORKSPACE_VIEW_CATEGORIES
        },
        backups={
            category: targets[category].parent
            / f".{targets[category].name}{_BACKUP_SUFFIX}"
            for category in WORKSPACE_VIEW_CATEGORIES
        },
        metadata_root=metadata_root,
        journal_path=journal_path,
        journal_temporary_path=journal_path.with_name(journal_path.name + ".tmp"),
        ownership_registry_path=(
            metadata_root / f"{_OWNERSHIP_REGISTRY_PREFIX}.{fingerprint}.json"
        ),
        lock_path=lock_path,
        target_set_fingerprint=fingerprint,
    )
    _check_workspace_path_isolation(layout)
    return layout


@contextmanager
def _workspace_output_lock(layout: _WorkspaceLayout):
    """Hold one crash-released inter-process lock for the complete view set."""

    lock_root = layout.lock_path.parent
    if lock_root.is_symlink():
        raise WorkspaceOutputError(
            f"Workspace output lock root is a symlink: {lock_root}"
        )
    _durable_mkdir(lock_root, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(layout.lock_path, flags, 0o600)
    except OSError as exc:
        raise WorkspaceOutputError(
            f"Workspace output lock could not be opened: {layout.lock_path}"
        ) from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkspaceOutputError(
                f"Workspace output lock is held by another process: {layout.lock_path}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


# --------------------------------------------------------------------------
# Source-package snapshots
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _PackageSnapshot:
    """Everything about a verified source package a view set is bound to."""

    target_path: Path
    checksum_ledger_sha256: str
    run_manifest_sha256: str
    run_identity_sha256: str
    run_id: str
    artifacts: tuple[tuple[str, str], ...]


def _verify_source_package(target_path: str | Path) -> ReleasePackage:
    """Re-verify the complete source package with the authoritative verifier."""

    try:
        return verify_completed_release_package(target_path)
    except ReportingError as exc:
        raise WorkspaceOutputError(
            f"Source release package failed complete verification: {target_path}"
        ) from exc


def _snapshot_package(package: ReleasePackage) -> _PackageSnapshot:
    """Capture the complete identity of one freshly verified package."""

    identity, run_id = _release_run_identity(package)
    manifest_digest = package.artifact_sha256.get("run_manifest.json")
    if not isinstance(manifest_digest, str) or len(manifest_digest) != 64:
        raise WorkspaceOutputError(
            "Source release package does not bind its own run manifest"
        )
    return _PackageSnapshot(
        target_path=Path(package.target_path).resolve(),
        checksum_ledger_sha256=sha256_file(package.checksums_path),
        run_manifest_sha256=sha256_file(package.manifest_path),
        run_identity_sha256=identity,
        run_id=run_id,
        artifacts=tuple(
            sorted(
                (str(key), str(value)) for key, value in package.artifact_sha256.items()
            )
        ),
    )


def _check_supplied_package_matches(
    verified: ReleasePackage, supplied: ReleasePackage
) -> None:
    """Refuse a caller-supplied package handle the source no longer matches.

    The supplied handle may have been obtained long before this call. Comparing
    it against a freshly verified snapshot catches a source that has since been
    rewritten *consistently* -- bytes and ledger both -- which no per-artifact
    check inside the projection can see, including drift confined to
    retained-only paths such as ``run_manifest.json``.
    """

    if Path(verified.target_path).resolve() != Path(supplied.target_path).resolve():
        raise WorkspaceOutputError(
            "Verified source release package is not the supplied package: "
            f"{verified.target_path} / {supplied.target_path}"
        )
    for label, left, right in (
        ("checksum ledger", verified.checksums_path, supplied.checksums_path),
        ("run manifest", verified.manifest_path, supplied.manifest_path),
        ("report", verified.report_path, supplied.report_path),
        ("report pdf", verified.report_pdf_path, supplied.report_pdf_path),
    ):
        if Path(left).resolve() != Path(right).resolve():
            raise WorkspaceOutputError(
                f"Supplied release package {label} path does not match the verified "
                f"package: {right}"
            )
    if dict(verified.artifact_sha256) != dict(supplied.artifact_sha256):
        raise WorkspaceOutputError(
            "Source release package has drifted since the supplied package handle was "
            f"obtained: {verified.target_path}"
        )


def _check_package_snapshot_unchanged(
    before: _PackageSnapshot, after: _PackageSnapshot
) -> None:
    """Refuse a source package that changed while the views were being built."""

    if before != after:
        raise WorkspaceOutputError(
            "Source release package changed while its workspace views were being "
            f"materialized: {after.target_path}"
        )


# --------------------------------------------------------------------------
# Structural ownership
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _ViewOwnership:
    """A view target this module structurally owns and may therefore replace."""

    binding: tuple[str, str, str, str]
    artifacts: tuple[WorkspaceArtifact, ...]


def _structural_ownership(
    target: Path,
    *,
    category: str,
    view_name: str,
    expected_plan: PlannedWorkspaceView | None = None,
) -> _ViewOwnership:
    """Prove this module generated the directory now sitting at ``target``.

    Ownership requires the complete generated shape: every source-binding field,
    both accounting fields, a well-formed artifact list on safe unique paths, a
    checksum ledger whose inventory is exactly those artifacts plus the manifest,
    and a ledger entry for the manifest matching the manifest's own bytes. Curves
    views may also carry the exact separately owned dataset roots; every entry
    there is recursively lstat-validated but is not claimed by this ledger.

    With an expected plan, every declared digest and every current artifact byte
    must match that verified package. Drift is accepted only through a separately
    stored generated-metadata attestation; co-located self-signed metadata alone
    never establishes replacement authority.
    """

    manifest = _read_view_manifest(
        target,
        accepted_schema_versions=frozenset(
            {
                WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION,
                _LEGACY_WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION_V3,
                _LEGACY_WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION_V2,
                _LEGACY_WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION_V1,
            }
        ),
    )
    if (
        manifest.get("category") != category
        or manifest.get("root_key") != _CATEGORY_ROOT_KEYS[category]
        or manifest.get("view_name") != view_name
    ):
        raise WorkspaceOutputError("Workspace ownership metadata names another view")

    binding_values: list[str] = []
    for field in _SOURCE_BINDING_FIELDS:
        value = manifest.get(field)
        if not isinstance(value, str) or not value:
            raise WorkspaceOutputError(
                f"Workspace ownership metadata is missing its {field}"
            )
        binding_values.append(value)
    for field in (
        "source_run_identity_sha256",
        "source_checksum_ledger_sha256",
        "source_run_manifest_sha256",
    ):
        digest = str(manifest.get(field))
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise WorkspaceOutputError(
                f"Workspace ownership metadata has an unusable {field}"
            )
    if not isinstance(manifest.get("source_run_id"), str):
        raise WorkspaceOutputError("Workspace ownership metadata is missing its run id")

    declared = manifest.get("artifacts")
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
        raise WorkspaceOutputError("Workspace ownership metadata has no artifact list")
    artifacts: list[WorkspaceArtifact] = []
    declared_paths: set[str] = set()
    total_bytes = 0
    for item in declared:
        if not isinstance(item, Mapping):
            raise WorkspaceOutputError(
                "Workspace ownership metadata has an invalid artifact entry"
            )
        relative = item.get("path")
        source_relative = item.get("source_path")
        digest = item.get("sha256")
        size = item.get("bytes")
        if (
            not isinstance(relative, str)
            or not isinstance(source_relative, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise WorkspaceOutputError(
                "Workspace ownership metadata has an invalid artifact entry"
            )
        _check_relative_path(relative, where="Workspace ownership artifact")
        _check_relative_path(source_relative, where="Workspace ownership source")
        schema_version = manifest.get("schema_version")
        classified = _classify(source_relative)
        if classified is None or classified[0] != category:
            raise WorkspaceOutputError(
                "Workspace ownership metadata contains an artifact from another view"
            )
        if schema_version == _LEGACY_WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION_V1:
            if category == "curves":
                if source_relative.startswith("tables/curves/"):
                    legacy_relative = "tables/" + source_relative[
                        len("tables/curves/") :
                    ]
                elif source_relative.startswith("figures/"):
                    legacy_relative = source_relative
                else:
                    raise WorkspaceOutputError(
                        "Legacy curves ownership metadata contains an unsupported source"
                    )
            else:
                legacy_relative = classified[1]
            if relative != legacy_relative:
                raise WorkspaceOutputError(
                    "Legacy workspace ownership metadata does not match the exact v1 "
                    f"layout: {relative}"
                )
        elif relative != classified[1]:
            raise WorkspaceOutputError(
                "Workspace ownership metadata does not match the exact current "
                f"layout: {relative}"
            )
        if relative in declared_paths or relative == WORKSPACE_VIEW_MANIFEST_NAME:
            raise WorkspaceOutputError(
                f"Workspace ownership metadata declares {relative} more than once"
            )
        declared_paths.add(relative)
        total_bytes += size
        artifacts.append(
            WorkspaceArtifact(
                source_relative_path=source_relative,
                target_relative_path=relative,
                sha256=digest,
            )
        )
    if expected_plan is not None:
        actual_projection = {
            (
                artifact.source_relative_path,
                artifact.sha256,
            )
            for artifact in artifacts
        }
        expected_projection = {
            (
                artifact.source_relative_path,
                artifact.sha256,
            )
            for artifact in expected_plan.artifacts
        }
        if actual_projection != expected_projection:
            raise WorkspaceOutputError(
                "Workspace ownership metadata does not match the verified release mapping"
            )
        for artifact in artifacts:
            artifact_path = target / artifact.target_relative_path
            _assert_plain_regular_file(
                artifact_path, label="Workspace ownership artifact"
            )
            if sha256_file(artifact_path) != artifact.sha256:
                raise WorkspaceOutputError(
                    "Workspace ownership artifact bytes do not match the verified "
                    f"release mapping: {artifact.target_relative_path}"
                )
    if (
        manifest.get("artifact_count") != len(artifacts)
        or manifest.get("total_bytes") != total_bytes
    ):
        raise WorkspaceOutputError("Workspace ownership accounting does not reconcile")

    checksums_path = target / WORKSPACE_CHECKSUMS_NAME
    if checksums_path.is_symlink() or not checksums_path.is_file():
        raise WorkspaceOutputError("Workspace ownership checksum ledger is missing")
    ledger = _parse_checksums(checksums_path)
    if set(ledger) != declared_paths | {WORKSPACE_VIEW_MANIFEST_NAME}:
        raise WorkspaceOutputError(
            "Workspace ownership checksum ledger does not enumerate exactly the "
            "declared artifacts"
        )
    manifest_path = target / WORKSPACE_VIEW_MANIFEST_NAME
    if ledger[WORKSPACE_VIEW_MANIFEST_NAME] != sha256_file(manifest_path):
        raise WorkspaceOutputError(
            "Workspace ownership checksum ledger does not bind its own manifest"
        )
    expected_files = declared_paths | {
        WORKSPACE_VIEW_MANIFEST_NAME,
        WORKSPACE_CHECKSUMS_NAME,
    }
    files, directories = _walk_view_entries(target)
    extension_files, extension_directories = _extension_inventory(
        files,
        directories,
        category=category,
        schema_version=manifest.get("schema_version"),
    )
    if (
        category == "curves"
        and manifest.get("schema_version") == WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION
        and _contains_type_bucket(directories)
    ):
        raise WorkspaceOutputError(
            "The curves workspace view contains a figures/ or tables/ directory"
        )
    actual_managed_files = files - extension_files
    allowed_directories = _declared_directory_closure(
        expected_files
    ) | _inventory_directory_closure(extension_files, extension_directories)
    if actual_managed_files != expected_files or directories != allowed_directories:
        raise WorkspaceOutputError(
            "Workspace ownership inventory contains undeclared or missing filesystem "
            "entries"
        )
    return _ViewOwnership(
        binding=(
            binding_values[0],
            binding_values[1],
            binding_values[2],
            binding_values[3],
        ),
        artifacts=tuple(artifacts),
    )


def _assert_plain_regular_file(path: Path, *, label: str) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise WorkspaceOutputError(f"{label} is unavailable: {path}") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise WorkspaceOutputError(f"{label} is not a plain regular file: {path}")


def _ownership_metadata_fingerprint(target: Path) -> str:
    """Bind ownership to immutable generator metadata kept outside the view."""

    manifest = target / WORKSPACE_VIEW_MANIFEST_NAME
    checksums = target / WORKSPACE_CHECKSUMS_NAME
    _assert_plain_regular_file(manifest, label="Workspace ownership manifest")
    _assert_plain_regular_file(checksums, label="Workspace ownership checksum ledger")
    payload = {
        "manifest_sha256": sha256_file(manifest),
        "checksums_sha256": sha256_file(checksums),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_ownership_registry_file(
    path: Path, layout: _WorkspaceLayout
) -> Mapping[str, str]:
    _assert_plain_regular_file(path, label="Workspace ownership registry")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkspaceOutputError(
            f"Workspace ownership registry is unreadable: {path}"
        ) from exc
    categories = payload.get("categories") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != _OWNERSHIP_REGISTRY_SCHEMA_VERSION
        or payload.get("view_name") != layout.view_name
        or payload.get("target_set_fingerprint") != layout.target_set_fingerprint
        or not isinstance(categories, Mapping)
        or set(categories) != set(WORKSPACE_VIEW_CATEGORIES)
        or any(
            not _is_state_digest(categories.get(category))
            for category in WORKSPACE_VIEW_CATEGORIES
        )
    ):
        raise WorkspaceOutputError(
            f"Workspace ownership registry has an invalid schema: {path}"
        )
    return {category: str(categories[category]) for category in WORKSPACE_VIEW_CATEGORIES}


def _read_ownership_registry(layout: _WorkspaceLayout) -> Mapping[str, str]:
    """Read attestations, completing an interrupted verified registry write."""

    path = layout.ownership_registry_path
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        pending = _parse_ownership_registry_file(temporary, layout)
        try:
            live = {
                category: _ownership_metadata_fingerprint(layout.targets[category])
                for category in WORKSPACE_VIEW_CATEGORIES
            }
        except WorkspaceOutputError as exc:
            raise WorkspaceOutputError(
                "Interrupted workspace ownership registry cannot be reconciled with "
                "the live views; its temporary is preserved"
            ) from exc
        if live != pending:
            raise WorkspaceOutputError(
                "Interrupted workspace ownership registry does not attest the live "
                "views; its temporary is preserved"
            )
        if path.exists() or path.is_symlink():
            _assert_plain_regular_file(path, label="Workspace ownership registry")
        os.replace(temporary, path)
        _sync_directory(path.parent)
        return pending
    if not path.exists() and not path.is_symlink():
        return {}
    return _parse_ownership_registry_file(path, layout)


def _write_ownership_registry(layout: _WorkspaceLayout) -> None:
    """Durably attest the exact generated manifests and ledgers now live."""

    path = layout.ownership_registry_path
    if path.exists() or path.is_symlink():
        _assert_plain_regular_file(path, label="Workspace ownership registry")
    payload = {
        "schema_version": _OWNERSHIP_REGISTRY_SCHEMA_VERSION,
        "view_name": layout.view_name,
        "target_set_fingerprint": layout.target_set_fingerprint,
        "categories": {
            category: _ownership_metadata_fingerprint(layout.targets[category])
            for category in WORKSPACE_VIEW_CATEGORIES
        },
    }
    _write_transaction(path, payload)


def _live_view_state(
    target: Path,
    *,
    category: str,
    view_name: str,
    expected_plan: PlannedWorkspaceView,
    trusted_attestation: str | None = None,
) -> _ViewOwnership | None:
    """Classify a view target: ``None`` when absent, else proven ownership.

    Anything present that this module cannot structurally prove it generated is
    operator content. It is refused, never replaced and never deleted.
    """

    if target.is_symlink():
        raise WorkspaceOutputError(f"Workspace view target is a symlink: {target}")
    if not target.exists():
        return None
    if not target.is_dir():
        raise WorkspaceOutputError(
            f"Workspace view target is not a directory: {target}"
        )
    if not any(target.iterdir()):
        return None
    try:
        if trusted_attestation is not None:
            try:
                ownership = _structural_ownership(
                    target,
                    category=category,
                    view_name=view_name,
                )
                if _ownership_metadata_fingerprint(target) != trusted_attestation:
                    raise WorkspaceOutputError(
                        "Workspace ownership metadata does not match its external attestation"
                    )
                return ownership
            except WorkspaceOutputError:
                # A crash can commit a new exact generation before its registry
                # update. Permit only byte-for-byte bootstrap against this package.
                pass
        return _structural_ownership(
            target,
            category=category,
            view_name=view_name,
            expected_plan=expected_plan,
        )
    except WorkspaceOutputError as exc:
        raise WorkspaceOutputError(
            "Workspace view target holds unmanaged content this run cannot prove it "
            f"generated and will not be replaced; move it aside to regenerate: {target}"
        ) from exc


def _state_fingerprint(path: Path) -> str:
    """Cryptographic identity of one directory tree: inventory, types, and bytes.

    Covers the exact bytes of every regular file *and* the exact directory
    inventory, so no added, removed, renamed, or edited entry -- file or
    directory -- can reproduce a digest. ``_walk_view_entries`` supplies the
    inventory, so a symlink, a non-regular entry, or a multiply-linked file is
    refused here rather than being quietly fingerprinted as ordinary content.
    """

    files, directories = _walk_view_entries(path)
    payload = {
        "directories": sorted(directories),
        "files": {relative: sha256_file(path / relative) for relative in sorted(files)},
    }
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Transaction journal and residue recovery
# --------------------------------------------------------------------------


def _is_state_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_journal_category_record(record: Any) -> bool:
    """One category's journaled state: what existed, and exactly what it was."""

    if not isinstance(record, Mapping) or set(record) != {
        "had_prior",
        "staged_state_sha256",
        "prior_state_sha256",
    }:
        return False
    had_prior = record["had_prior"]
    if not isinstance(had_prior, bool) or not _is_state_digest(
        record["staged_state_sha256"]
    ):
        return False
    prior = record["prior_state_sha256"]
    # ``had_prior`` false is itself a claim -- that the target was absent or
    # empty -- so it must carry no prior fingerprint at all.
    return _is_state_digest(prior) if had_prior else prior is None


def _check_recovery_candidate(
    path: Path,
    *,
    label: str,
    category: str,
    view_name: str,
    expected_plan: PlannedWorkspaceView,
    allowed: Mapping[str, tuple[Any, Any] | None],
    trusted_attestation: str | None,
) -> None:
    """Refuse any candidate that is not exactly one state the journal recorded.

    ``allowed`` maps each acceptable state fingerprint to the source binding that
    state must additionally carry, or ``None`` when any structurally owned
    binding is acceptable -- a prior legitimately predates this transaction's
    source, while a stage must be bound to it.

    Read-only. An absent candidate is valid at several transaction phases. An
    empty candidate is accepted only when its exact empty-tree fingerprint was
    journaled as a prior state; a generated stage must always prove structural
    ownership as well as byte identity.
    """

    if path.is_symlink():
        raise WorkspaceOutputError(
            f"Interrupted workspace transaction {label} is a symlink: {path}"
        )
    if not path.exists():
        return
    if not path.is_dir():
        raise WorkspaceOutputError(
            f"Interrupted workspace transaction {label} is not a directory: {path}"
        )
    is_empty = not any(path.iterdir())
    unproven = (
        f"Interrupted workspace transaction {label} holds content this run cannot "
        "prove it generated; it will not be deleted or replaced. Resolve it "
        f"manually: {path}"
    )
    try:
        digest = _state_fingerprint(path)
        if digest not in allowed:
            raise WorkspaceOutputError(unproven)
        required_binding = allowed[digest]
        if is_empty:
            if required_binding is not None:
                raise WorkspaceOutputError(unproven)
            return
        if required_binding is None and trusted_attestation is not None:
            try:
                ownership = _structural_ownership(
                    path,
                    category=category,
                    view_name=view_name,
                )
                if _ownership_metadata_fingerprint(path) != trusted_attestation:
                    raise WorkspaceOutputError(unproven)
            except WorkspaceOutputError:
                ownership = _structural_ownership(
                    path,
                    category=category,
                    view_name=view_name,
                    expected_plan=expected_plan,
                )
        else:
            ownership = _structural_ownership(
                path,
                category=category,
                view_name=view_name,
                expected_plan=expected_plan,
            )
    except WorkspaceOutputError as exc:
        raise WorkspaceOutputError(unproven) from exc
    if (
        required_binding is not None
        and (
            ownership.binding[1],
            ownership.binding[2],
        )
        != required_binding
    ):
        raise WorkspaceOutputError(unproven)
    if required_binding is not None:
        # Journal and tree fingerprints are attacker-computable. A stage (or a
        # stage already promoted live) therefore also has to pass its complete
        # checksum-ledger verification; structural ownership alone is reserved
        # for rollback priors whose generated bytes may legitimately have drifted.
        try:
            verify_workspace_view(path)
        except WorkspaceOutputError as exc:
            raise WorkspaceOutputError(unproven) from exc


def _authorize_recovery_candidates(
    layout: _WorkspaceLayout,
    payload: Mapping[str, Any],
    expected_plan: Mapping[str, PlannedWorkspaceView],
    trusted_attestations: Mapping[str, str],
) -> None:
    """Prove every path this recovery may delete or rename is journaled state.

    The journal is an ordinary JSON file inside the workspace, so anything able
    to write a view target can write a journal naming it. Its say-so is
    therefore never authority to destroy what is at those paths now. This runs
    before the commit-or-roll-back fork and is all-or-nothing: it either clears
    all nine candidates or raises having touched none of them, leaving the
    journal in place for an operator to adjudicate.
    """

    records = payload["categories"]
    journal_binding = (
        payload.get("source_run_identity_sha256"),
        payload.get("source_checksum_ledger_sha256"),
    )
    for category in WORKSPACE_VIEW_CATEGORIES:
        record = records[category]
        staged = record["staged_state_sha256"]
        prior = record["prior_state_sha256"]
        staged_only: dict[str, tuple[Any, Any] | None] = {staged: journal_binding}
        # Mid-promotion a live target legitimately holds either generation: the
        # prior until its rename lands, the staged state immediately after.
        live_allowed = (
            dict(staged_only) if prior is None else {**staged_only, prior: None}
        )
        for label, path, allowed in (
            (f"{category} stage", layout.stages[category], staged_only),
            (
                f"{category} prior",
                layout.backups[category],
                {} if prior is None else {prior: None},
            ),
            (f"{category} view target", layout.targets[category], live_allowed),
        ):
            _check_recovery_candidate(
                path,
                label=label,
                category=category,
                view_name=layout.view_name,
                expected_plan=expected_plan[category],
                allowed=allowed,
                trusted_attestation=trusted_attestations.get(category),
            )


def _write_transaction(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise WorkspaceOutputError(
            f"Workspace transaction temporary path is already present: {temporary}"
        )
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _remove_transaction(path: Path) -> None:
    if path.is_symlink():
        raise WorkspaceOutputError(f"Workspace transaction path is a symlink: {path}")
    if path.exists():
        path.unlink()
        _sync_directory(path.parent)


def _read_transaction(layout: _WorkspaceLayout) -> Mapping[str, Any]:
    path = layout.journal_path
    if path.is_symlink() or not path.is_file():
        raise WorkspaceOutputError(f"Workspace transaction journal is unsafe: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkspaceOutputError(
            f"Workspace transaction journal is unreadable: {path}"
        ) from exc
    categories = payload.get("categories") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != _TRANSACTION_SCHEMA_VERSION
        or payload.get("view_name") != layout.view_name
        or not isinstance(categories, Mapping)
        or set(categories) != set(WORKSPACE_VIEW_CATEGORIES)
        or any(
            not _is_journal_category_record(categories.get(category))
            for category in WORKSPACE_VIEW_CATEGORIES
        )
    ):
        raise WorkspaceOutputError(
            f"Workspace transaction journal has an invalid schema: {path}"
        )
    if payload.get("target_set_fingerprint") != layout.target_set_fingerprint:
        raise WorkspaceOutputError(
            "Workspace transaction journal was written for a different target set; "
            f"the configured view roots changed: {path}"
        )
    return payload


def _refuse_unauthorized_residue(layout: _WorkspaceLayout) -> None:
    """Refuse stage, prior, or journal-temporary residue no journal authorizes.

    Reaching here means transaction recovery already ran: either it consumed a
    valid journal and cleaned up after itself, or there was no journal at all.
    Residue surviving that is unexplained, so it is left exactly where it is for
    an operator to adjudicate. Deleting it would discard the only copy of a view
    whose provenance this run cannot reconstruct.
    """

    for category in WORKSPACE_VIEW_CATEGORIES:
        for label, path in (
            ("stage", layout.stages[category]),
            ("prior", layout.backups[category]),
        ):
            if path.exists() or path.is_symlink():
                raise WorkspaceOutputError(
                    f"Workspace view {label} residue is present with no transaction "
                    "journal authorizing it; resolve it manually before regenerating: "
                    f"{path}"
                )
    temporary = layout.journal_temporary_path
    if temporary.exists() or temporary.is_symlink():
        raise WorkspaceOutputError(
            "Workspace transaction-temporary residue is present with no journal "
            f"authorizing it; resolve it manually: {temporary}"
        )


def _recovery_source_package(
    layout: _WorkspaceLayout,
    payload: Mapping[str, Any],
    *,
    config: Any,
) -> ReleasePackage:
    """Resolve and verify the package named by an older interrupted generation."""

    project_root = Path(getattr(config, "project_root"))
    expected_binding = (
        payload.get("source_run_identity_sha256"),
        payload.get("source_checksum_ledger_sha256"),
    )
    for category in WORKSPACE_VIEW_CATEGORIES:
        for candidate in (
            layout.stages[category],
            layout.targets[category],
            layout.backups[category],
        ):
            if candidate.is_symlink() or not candidate.is_dir() or not any(candidate.iterdir()):
                continue
            try:
                manifest = _read_view_manifest(
                    candidate,
                    accepted_schema_versions=frozenset(
                        {
                            WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION,
                            _LEGACY_WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION_V3,
                            _LEGACY_WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION_V2,
                            _LEGACY_WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION_V1,
                        }
                    ),
                )
                declared = manifest.get("source_release_path")
                if not isinstance(declared, str) or not declared:
                    continue
                source_path = Path(declared)
                if not source_path.is_absolute():
                    _check_relative_path(
                        source_path.as_posix(), where="Recovery source release path"
                    )
                    source_path = project_root / source_path
                recovered = _verify_source_package(source_path)
                snapshot = _snapshot_package(recovered)
            except (OSError, WorkspaceOutputError):
                continue
            if (
                snapshot.run_identity_sha256,
                snapshot.checksum_ledger_sha256,
            ) == expected_binding:
                return recovered
    raise WorkspaceOutputError(
        "Interrupted workspace transaction refers to a source package that can no "
        "longer be independently verified; recovery will not delete its candidates"
    )


def _recover_workspace_transaction(
    layout: _WorkspaceLayout,
    package: ReleasePackage,
    *,
    config: Any,
    source_run_identity_sha256: str,
    source_checksum_ledger_sha256: str,
    trusted_attestations: Mapping[str, str],
    force_rollback: bool = False,
) -> str | None:
    """Commit or roll back an interrupted three-view transaction as one set.

    ``force_rollback`` skips commit detection. It is used when the caller knows
    the promoted set must be undone even though it verifies -- notably when the
    source package drifted after promotion, which leaves views that are
    internally valid but bound to a package that no longer exists on disk.
    """

    journal = layout.journal_path
    if not journal.exists() and not journal.is_symlink():
        return None
    payload = _read_transaction(layout)
    binding_matches = (
        payload.get("source_run_identity_sha256") == source_run_identity_sha256
        and payload.get("source_checksum_ledger_sha256")
        == source_checksum_ledger_sha256
    )
    recovery_package = package
    if not binding_matches:
        recovery_package = _recovery_source_package(layout, payload, config=config)
    # Before the commit-or-roll-back fork, and before any mutation: a journal is
    # evidence of what this module did, never authority over what is there now.
    _authorize_recovery_candidates(
        layout,
        payload,
        plan_workspace_views(recovery_package),
        trusted_attestations,
    )
    records = payload["categories"]
    had_prior = {
        category: bool(records[category]["had_prior"])
        for category in WORKSPACE_VIEW_CATEGORIES
    }
    stages = layout.stages
    backups = layout.backups
    targets = layout.targets

    committed_live_set = False
    if force_rollback:
        # The caller has already established that this set must be undone, so
        # commit detection is skipped rather than consulted.
        pass
    elif binding_matches:
        try:
            verify_workspace_views(
                config,
                view_name=layout.view_name,
                expected_package=package,
            )
        except WorkspaceOutputError:
            pass
        else:
            committed_live_set = True
    else:
        # The journal belongs to an earlier generation than the package now being
        # projected. The live set counts as committed only if all three views are
        # structurally owned and consistently bound to *that* journal's source.
        live_bindings: set[tuple[str, str, str, str]] = set()
        expected_plan = plan_workspace_views(recovery_package)
        try:
            for category in WORKSPACE_VIEW_CATEGORIES:
                verify_workspace_view(targets[category])
                ownership = _live_view_state(
                    targets[category],
                    category=category,
                    view_name=layout.view_name,
                    expected_plan=expected_plan[category],
                    trusted_attestation=trusted_attestations.get(category),
                )
                if ownership is None:
                    raise WorkspaceOutputError("Live workspace view is absent")
                if ownership.binding[1] != payload.get(
                    "source_run_identity_sha256"
                ) or ownership.binding[2] != payload.get(
                    "source_checksum_ledger_sha256"
                ):
                    raise WorkspaceOutputError(
                        "Live workspace view does not match the interrupted transaction"
                    )
                live_bindings.add(ownership.binding)
        except WorkspaceOutputError:
            pass
        else:
            committed_live_set = len(live_bindings) == 1
    if committed_live_set:
        for category in WORKSPACE_VIEW_CATEGORIES:
            _remove_tree(stages[category])
            _remove_tree(backups[category])
        _remove_transaction(journal)
        _write_ownership_registry(layout)
        return "transaction_commit_completed"

    for category in WORKSPACE_VIEW_CATEGORIES:
        target = targets[category]
        stage = stages[category]
        backup = backups[category]
        if had_prior[category]:
            prior = backup if backup.exists() else target if stage.exists() else None
            if prior is None:
                raise WorkspaceOutputError(
                    "Interrupted workspace transaction cannot prove the prior view: "
                    f"{target}"
                )
            # ``_authorize_recovery_candidates`` already proved this exact prior
            # byte-for-byte and structurally when non-empty. Empty pre-existing
            # directories are legitimate prior state and must be restored too.
        elif backup.exists() or (target.exists() and stage.exists()):
            raise WorkspaceOutputError(
                "Interrupted workspace transaction has ambiguous first-generation residue: "
                f"{target}"
            )

    for category in reversed(WORKSPACE_VIEW_CATEGORIES):
        target = targets[category]
        stage = stages[category]
        backup = backups[category]
        if had_prior[category] and backup.exists():
            _remove_tree(target)
            _promote_directory(backup, target)
        elif not had_prior[category] and target.exists() and not stage.exists():
            _remove_tree(target)
    for category in WORKSPACE_VIEW_CATEGORIES:
        _remove_tree(stages[category])
        _remove_tree(backups[category])
    _remove_transaction(journal)
    return "transaction_rolled_back"


# --------------------------------------------------------------------------
# Production
# --------------------------------------------------------------------------


def _copy_standalone_extension(source_view: Path, stage: Path) -> None:
    """Carry the validated standalone dataset roots into a new curves stage."""

    manifest = _read_view_manifest(
        source_view,
        accepted_schema_versions=frozenset(
            {
                WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION,
                _LEGACY_WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION_V3,
                _LEGACY_WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION_V2,
                _LEGACY_WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION_V1,
            }
        ),
    )
    schema_version = manifest.get("schema_version")
    source_prefixes = _standalone_extension_prefixes("curves", schema_version)
    assert source_prefixes
    files, directories = _walk_view_entries(source_view)
    extension_files, extension_directories = _extension_inventory(
        files,
        directories,
        category="curves",
        schema_version=schema_version,
    )
    if not extension_files and not extension_directories:
        return

    try:
        before = {
            relative: sha256_file(source_view / relative)
            for relative in sorted(extension_files)
        }
    except OSError as exc:
        raise WorkspaceOutputError(
            "Standalone dataset directories changed while they were inventoried"
        ) from exc

    def destination_relative(relative: str) -> Path | None:
        if schema_version == WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION:
            return Path(relative)
        source_prefix = next(
            prefix
            for prefix in source_prefixes
            if relative == prefix or relative.startswith(prefix + "/")
        )
        suffix = Path(relative).relative_to(source_prefix)
        if not suffix.parts:
            return None
        parts = list(suffix.parts)
        if parts[0] == "core_trial_data":
            parts[0] = "literature_extracted_dataset"
        if parts[0] not in _STANDALONE_EXTENSION_ROOTS_V4:
            raise WorkspaceOutputError(
                "Legacy standalone extension contains an unrecognized dataset "
                f"directory: {parts[0]}"
            )
        return Path(*parts)

    for relative in sorted(
        extension_directories, key=lambda item: (len(Path(item).parts), item)
    ):
        destination_relative_path = destination_relative(relative)
        if destination_relative_path is not None:
            (stage / destination_relative_path).mkdir(exist_ok=True)
    for relative in sorted(extension_files):
        source = source_view / relative
        destination_relative_path = destination_relative(relative)
        assert destination_relative_path is not None
        destination = stage / destination_relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise WorkspaceOutputError(
                "Standalone extension migration would overwrite another file: "
                f"{destination}"
            )
        try:
            info = os.lstat(source)
        except OSError as exc:
            raise WorkspaceOutputError(
                f"Standalone extension file disappeared during copy: {source}"
            ) from exc
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink > 1
            or source.is_symlink()
        ):
            raise WorkspaceOutputError(
                f"Standalone extension file became unsafe during copy: {source}"
            )
        _copy_artifact(source, destination)
        if sha256_file(destination) != before[relative]:
            raise WorkspaceOutputError(
                f"Standalone extension file changed during copy: {source}"
            )

    after_files, after_directories = _walk_view_entries(source_view)
    after_extension_files, after_extension_directories = _extension_inventory(
        after_files,
        after_directories,
        category="curves",
        schema_version=schema_version,
    )
    try:
        after = {
            relative: sha256_file(source_view / relative)
            for relative in sorted(after_extension_files)
        }
    except OSError as exc:
        raise WorkspaceOutputError(
            "Standalone dataset directories changed during copy"
        ) from exc
    if (
        after_extension_files != extension_files
        or after_extension_directories != extension_directories
        or after != before
    ):
        raise WorkspaceOutputError(
            "Standalone dataset directories changed during copy"
        )


def _stage_view(
    plan: PlannedWorkspaceView,
    stage: Path,
    *,
    package: ReleasePackage,
    view_name: str,
    source_release_path: str,
    source_run_id: str,
    source_run_identity_sha256: str,
    source_checksum_ledger_sha256: str,
    source_run_manifest_sha256: str,
) -> None:
    _durable_mkdir(stage)
    ledger: dict[str, str] = {}
    manifest_artifacts: list[dict[str, Any]] = []
    for artifact in plan.artifacts:
        source = package.target_path / artifact.source_relative_path
        if source.is_symlink() or not source.is_file():
            raise WorkspaceOutputError(
                f"Release package artifact is missing or unsafe: {artifact.source_relative_path}"
            )
        destination = stage / artifact.target_relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        size = _copy_artifact(source, destination)
        digest = sha256_file(destination)
        if digest != artifact.sha256:
            raise WorkspaceOutputError(
                "Workspace view copy does not match the release checksum ledger: "
                f"{artifact.source_relative_path}"
            )
        ledger[artifact.target_relative_path] = digest
        manifest_artifacts.append(
            {
                "path": artifact.target_relative_path,
                "source_path": artifact.source_relative_path,
                "sha256": digest,
                "bytes": size,
            }
        )
    manifest_text = _render_manifest(
        category=plan.category,
        root_key=plan.root_key,
        view_name=view_name,
        source_release_path=source_release_path,
        source_run_id=source_run_id,
        source_run_identity_sha256=source_run_identity_sha256,
        source_checksum_ledger_sha256=source_checksum_ledger_sha256,
        source_run_manifest_sha256=source_run_manifest_sha256,
        artifacts=manifest_artifacts,
    )
    manifest_path = stage / WORKSPACE_VIEW_MANIFEST_NAME
    manifest_path.write_text(manifest_text, encoding="utf-8")
    ledger[WORKSPACE_VIEW_MANIFEST_NAME] = sha256_file(manifest_path)
    (stage / WORKSPACE_CHECKSUMS_NAME).write_text(
        _render_checksums(ledger), encoding="utf-8"
    )


def _materialize_workspace_views_locked(
    config: Any,
    package: ReleasePackage,
    *,
    layout: _WorkspaceLayout,
    run_log: Any | None = None,
) -> WorkspaceOutputs:
    """Project one verified release package into the configured workspace roots.

    Full mode only. The source package is re-verified in full before anything is
    staged and again immediately before the transaction commits, so a source that
    drifts underneath a long projection can never be silently released. Views are
    staged completely, verified while still staged, and only then promoted; a
    failure at any point leaves every previously complete view intact.
    """

    view_name = layout.view_name
    verified = _verify_source_package(layout.release_path)
    _check_supplied_package_matches(verified, package)
    source_snapshot = _snapshot_package(verified)

    plan = plan_workspace_views(verified)
    targets = layout.targets
    stages = layout.stages

    source_run_identity_sha256 = source_snapshot.run_identity_sha256
    source_run_id = source_snapshot.run_id
    source_checksum_ledger_sha256 = source_snapshot.checksum_ledger_sha256
    source_run_manifest_sha256 = str(verified.artifact_sha256["run_manifest.json"])
    source_release_path = _project_relative(
        verified.target_path, getattr(config, "project_root", verified.target_path)
    )

    trusted_attestations = _read_ownership_registry(layout)
    transaction_recovery = _recover_workspace_transaction(
        layout,
        verified,
        config=config,
        source_run_identity_sha256=source_run_identity_sha256,
        source_checksum_ledger_sha256=source_checksum_ledger_sha256,
        trusted_attestations=trusted_attestations,
    )
    if transaction_recovery and run_log is not None:
        run_log.info(
            "workspace_view_transaction_recovered",
            action=transaction_recovery,
            source_release=verified.target_path,
        )
    if transaction_recovery:
        trusted_attestations = _read_ownership_registry(layout)
    _refuse_unauthorized_residue(layout)

    states = {
        category: _live_view_state(
            targets[category],
            category=category,
            view_name=view_name,
            expected_plan=plan[category],
            trusted_attestation=trusted_attestations.get(category),
        )
        for category in WORKSPACE_VIEW_CATEGORIES
    }
    if all(state is not None for state in states.values()):
        try:
            views = verify_workspace_views(
                config, view_name=view_name, expected_package=verified
            )
        except WorkspaceOutputError:
            views = ()
        if views:
            reuse_source = _verify_source_package(verified.target_path)
            _check_package_snapshot_unchanged(
                source_snapshot, _snapshot_package(reuse_source)
            )
            _write_ownership_registry(layout)
            if run_log is not None:
                run_log.info(
                    "workspace_views_reused",
                    view_count=len(views),
                    source_release=verified.target_path,
                )
            return WorkspaceOutputs(
                views=views,
                reused=True,
                source_release_path=verified.target_path,
                source_run_identity_sha256=source_run_identity_sha256,
                source_checksum_ledger_sha256=source_checksum_ledger_sha256,
            )

    # An empty pre-existing target is real prior state: journal and restore it on
    # rollback rather than collapsing it into the first-generation/absent case.
    had_prior = {
        category: targets[category].exists() for category in WORKSPACE_VIEW_CATEGORIES
    }
    category_records: dict[str, dict[str, Any]] = {}
    try:
        for category in WORKSPACE_VIEW_CATEGORIES:
            _durable_mkdir(targets[category].parent, exist_ok=True)
            _stage_view(
                plan[category],
                stages[category],
                package=verified,
                view_name=view_name,
                source_release_path=source_release_path,
                source_run_id=source_run_id,
                source_run_identity_sha256=source_run_identity_sha256,
                source_checksum_ledger_sha256=source_checksum_ledger_sha256,
                source_run_manifest_sha256=source_run_manifest_sha256,
            )
            if category == "curves" and states[category] is not None:
                _copy_standalone_extension(targets[category], stages[category])
        for category in WORKSPACE_VIEW_CATEGORIES:
            verify_workspace_view(stages[category])
        # Every staged byte and directory entry reaches stable storage before the
        # journal names them, so a journal recovered after a crash can never
        # authorize promoting a stage that was never fully written.
        for category in WORKSPACE_VIEW_CATEGORIES:
            _fsync_tree(stages[category])
        # Fingerprinted from what is actually on disk: the stages after their last
        # write and their fsync, the priors at the last moment before the first
        # rename can move them. Recovery may only delete or restore a candidate
        # whose bytes and inventory still hash to one of these, so the journal
        # authorizes exactly the two states this run observed and nothing an
        # operator put there. Taken inside the staging guard: a fingerprint that
        # cannot be taken unwinds the stages rather than stranding three complete
        # ones that no journal explains and the next run refuses to reap.
        category_records = {
            category: {
                "had_prior": had_prior[category],
                "staged_state_sha256": _state_fingerprint(stages[category]),
                "prior_state_sha256": (
                    _state_fingerprint(targets[category])
                    if had_prior[category]
                    else None
                ),
            }
            for category in WORKSPACE_VIEW_CATEGORIES
        }
    except BaseException as exc:
        for stage in stages.values():
            try:
                _remove_tree(stage)
            except OSError:  # pragma: no cover - best effort during failure handling
                continue
        if not isinstance(exc, Exception):
            raise
        if isinstance(exc, WorkspaceOutputError):
            raise
        raise WorkspaceOutputError(f"Workspace view staging failed: {exc}") from exc

    journal = layout.journal_path
    # The journal's own home is linked from its parent before the journal is
    # written, so ``_write_transaction`` only ever has to sync the metadata root
    # itself -- keeping that sync the event immediately after the journal write.
    try:
        _durable_mkdir(layout.metadata_root, exist_ok=True)
        _write_transaction(
            journal,
            {
                "schema_version": _TRANSACTION_SCHEMA_VERSION,
                "view_name": view_name,
                "target_set_fingerprint": layout.target_set_fingerprint,
                "source_run_identity_sha256": source_run_identity_sha256,
                "source_checksum_ledger_sha256": source_checksum_ledger_sha256,
                "categories": category_records,
            },
        )
    except BaseException as exc:
        try:
            if journal.exists():
                _recover_workspace_transaction(
                    layout,
                    verified,
                    config=config,
                    source_run_identity_sha256=source_run_identity_sha256,
                    source_checksum_ledger_sha256=source_checksum_ledger_sha256,
                    trusted_attestations=trusted_attestations,
                    force_rollback=True,
                )
            else:
                _remove_tree(layout.journal_temporary_path)
                for stage in stages.values():
                    _remove_tree(stage)
        except BaseException as recovery_exc:
            raise WorkspaceOutputError(
                "Workspace transaction journal publication failed and cleanup also "
                f"failed: {recovery_exc}"
            ) from exc
        if not isinstance(exc, Exception):
            raise
        raise WorkspaceOutputError(
            f"Workspace transaction journal publication failed: {exc}"
        ) from exc
    try:
        for category in WORKSPACE_VIEW_CATEGORIES:
            target = targets[category]
            backup = layout.backups[category]
            if had_prior[category]:
                _promote_directory(target, backup)
            elif target.exists():
                if target.is_symlink() or not target.is_dir() or any(target.iterdir()):
                    raise WorkspaceOutputError(
                        f"Workspace view target changed during staging: {target}"
                    )
                target.rmdir()
                _sync_directory(target.parent)
            _promote_directory(stages[category], target)
    except BaseException as exc:
        try:
            _recover_workspace_transaction(
                layout,
                verified,
                config=config,
                source_run_identity_sha256=source_run_identity_sha256,
                source_checksum_ledger_sha256=source_checksum_ledger_sha256,
                trusted_attestations=trusted_attestations,
                force_rollback=True,
            )
        except BaseException as recovery_exc:
            raise WorkspaceOutputError(
                "Workspace view promotion failed and transactional recovery also failed: "
                f"{recovery_exc}"
            ) from exc
        if not isinstance(exc, Exception):
            raise
        raise WorkspaceOutputError(f"Workspace view promotion failed: {exc}") from exc

    try:
        views = verify_workspace_views(
            config,
            view_name=view_name,
            expected_package=verified,
        )
    except BaseException as exc:
        try:
            _recover_workspace_transaction(
                layout,
                verified,
                config=config,
                source_run_identity_sha256=source_run_identity_sha256,
                source_checksum_ledger_sha256=source_checksum_ledger_sha256,
                trusted_attestations=trusted_attestations,
                force_rollback=True,
            )
        except BaseException as recovery_exc:
            raise WorkspaceOutputError(
                "Promoted workspace views failed final verification and transactional "
                f"recovery also failed: {recovery_exc}"
            ) from exc
        if not isinstance(exc, Exception):
            raise
        if isinstance(exc, WorkspaceOutputError):
            raise
        raise WorkspaceOutputError(
            f"Promoted workspace views failed final verification: {exc}"
        ) from exc

    # The last read of the source before the priors become unrecoverable. The
    # promoted views verify against the snapshot they were built from, so only an
    # unconditional rollback can undo them once the source underneath has moved.
    try:
        _check_package_snapshot_unchanged(
            source_snapshot,
            _snapshot_package(_verify_source_package(layout.release_path)),
        )
    except BaseException as exc:
        try:
            _recover_workspace_transaction(
                layout,
                verified,
                config=config,
                source_run_identity_sha256=source_run_identity_sha256,
                source_checksum_ledger_sha256=source_checksum_ledger_sha256,
                trusted_attestations=trusted_attestations,
                force_rollback=True,
            )
        except BaseException as recovery_exc:
            raise WorkspaceOutputError(
                "Source release package changed during promotion and transactional "
                f"recovery also failed: {recovery_exc}"
            ) from exc
        if not isinstance(exc, Exception):
            raise
        if isinstance(exc, WorkspaceOutputError):
            raise
        raise WorkspaceOutputError(
            f"Source release package changed during promotion: {exc}"
        ) from exc

    # Prior complete views remain recoverable until every promoted target has
    # passed its final-path verification and the source has been re-proved
    # unchanged. Deleting them earlier would make a verification failure
    # irreversible even though staging had succeeded. Each removal syncs its
    # parent, so the journal is only dropped once the cleanup is itself durable.
    for category in WORKSPACE_VIEW_CATEGORIES:
        backup = layout.backups[category]
        if had_prior[category] and backup.exists():
            _remove_tree(backup)
    _remove_transaction(journal)
    _write_ownership_registry(layout)
    if run_log is not None:
        run_log.info(
            "workspace_views_materialized",
            view_count=len(views),
            artifact_count=sum(view.artifact_count for view in views),
            total_bytes=sum(view.total_bytes for view in views),
            source_release=verified.target_path,
        )
    return WorkspaceOutputs(
        views=views,
        reused=False,
        source_release_path=verified.target_path,
        source_run_identity_sha256=source_run_identity_sha256,
        source_checksum_ledger_sha256=source_checksum_ledger_sha256,
    )


def materialize_workspace_views(
    config: Any,
    package: ReleasePackage,
    *,
    view_name: str,
    run_log: Any | None = None,
) -> WorkspaceOutputs:
    """Serialize and materialize one complete set of canonical workspace views."""

    # Mode, view name, and the complete path set are settled before the lock,
    # because creating the lock file is itself a filesystem write.
    run_mode = getattr(config, "run_mode", None)
    if run_mode != "full":
        raise WorkspaceOutputError(
            f"Workspace views are produced only by full mode, not {run_mode!r}"
        )
    layout = _resolve_workspace_layout(config, package, view_name=view_name)
    with _workspace_output_lock(layout):
        return _materialize_workspace_views_locked(
            config,
            package,
            layout=layout,
            run_log=run_log,
        )


def workspace_view_display_paths(
    outputs: WorkspaceOutputs, project_root: str | Path
) -> tuple[tuple[str, str], ...]:
    """``(category, project-relative path)`` pairs for logging and the console."""

    root = Path(project_root)
    return tuple(
        (view.category, _project_relative(view.target_path, root))
        for view in outputs.views
    )


__all__ = [
    "PlannedWorkspaceView",
    "WORKSPACE_CHECKSUMS_NAME",
    "WORKSPACE_VIEW_CATEGORIES",
    "WORKSPACE_VIEW_MANIFEST_NAME",
    "WORKSPACE_VIEW_MANIFEST_SCHEMA_VERSION",
    "WorkspaceArtifact",
    "WorkspaceOutputError",
    "WorkspaceOutputs",
    "WorkspaceView",
    "materialize_workspace_views",
    "plan_workspace_views",
    "verify_workspace_view",
    "verify_workspace_views",
    "workspace_view_display_paths",
]
