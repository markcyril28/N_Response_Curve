"""Crash-safe replacement of one whole output directory, shared by the recipes.

Several standalone reporting recipes publish a *directory* rather than a file:
the bundle they write is the unit a reader consumes, and a half-written bundle
is worse than a stale one. They each need the same guarantees, and they need
them against one another, because two of them publish into subdirectories of a
third one's snapshot.

What this module provides, in the order a publisher uses it:

``publication_lock``
    An exclusive advisory lock on one publication container, so a parent writer
    and a nested writer cannot replace overlapping trees at the same time. The
    lock file lives under ``XDG_STATE_HOME``, never inside the operator output
    tree, so it cannot be captured by the snapshot it guards and cannot be
    mistaken for a bundle artifact. ``nested_publication_container`` resolves a
    nested destination up to the snapshot that owns it, so both writers take the
    *same* lock.

``recover_interrupted_directory_publication``
    Reconciles residue left by an interrupted earlier run against the external
    journal that authorized it. Residue with no journal is **preserved** and
    raises, rather than being cleaned up: an unexplained backup beside a live
    destination is operator material, not garbage.

``promote_staged_directory`` / ``promote_directory``
    The replacement itself. The prior generation is renamed aside before the
    staged one is renamed in, never deleted first, so an interruption at any
    point leaves either the prior tree or the new one in place — and the
    unwinding path restores the prior even when the interrupt arrives between
    the two renames. ``promote_directory`` is the unjournaled form for callers
    that hold the lock and do their own recovery; ``promote_staged_directory``
    writes and clears the journal itself.

Everything here refuses symlinks, hardlinks, and non-regular files at every
step, and re-checks them after opening rather than only before, because the
destination is an operator-visible directory that something else may be editing
while the recipe runs.

The on-disk names — the lock and journal filenames, the journal schema string,
and the ``.<name>.backup.``/``.staging.``/``.failed.`` residue prefixes — are
part of the contract between concurrent writers and between a run and its own
interrupted predecessor. They are deliberately literal and must not be changed
to match a renamed function.

This module is importable by the standalone reporting recipes and is not on the
release path.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path

__all__ = [
    "PUBLICATION_JOURNAL_SCHEMA",
    "copy_preserved_plain_tree",
    "directory_state_fingerprint",
    "nested_publication_container",
    "plain_absolute_path",
    "promote_directory",
    "promote_staged_directory",
    "publication_control_paths",
    "publication_journal_path",
    "publication_lock",
    "recover_interrupted_directory_publication",
    "remove_publication_journal",
    "sync_directory",
    "write_publication_journal",
]


PUBLICATION_JOURNAL_SCHEMA = "n-response-directory-publication-v1"


def plain_absolute_path(path: Path, *, label: str) -> Path:
    """Return a lexical absolute path after rejecting existing symlink components."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise RuntimeError(f"{label} contains a symlink component: {current}")
    return absolute

def promote_directory(staging: Path, destination: Path) -> None:
    """Replace one complete directory snapshot without deleting the prior first."""

    staging = plain_absolute_path(staging, label="Snapshot staging path")
    destination = plain_absolute_path(destination, label="Snapshot destination")
    if staging.parent != destination.parent or staging == destination:
        raise RuntimeError("Snapshot staging and destination must be distinct siblings")
    if not staging.is_dir():
        raise RuntimeError(f"Snapshot staging path is not a directory: {staging}")
    if destination.exists() and not destination.is_dir():
        raise RuntimeError(f"Snapshot destination is not a directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(
        f".{destination.name}.backup.{uuid.uuid4().hex}"
    )
    try:
        if destination.exists():
            if not destination.is_dir() or destination.is_symlink():
                raise RuntimeError("Snapshot destination is not a plain directory")
            os.replace(destination, backup)
        os.replace(staging, destination)
    except BaseException:
        if backup.exists():
            displaced = None
            if destination.exists():
                displaced = destination.with_name(
                    f".{destination.name}.failed.{uuid.uuid4().hex}"
                )
                os.replace(destination, displaced)
            try:
                os.replace(backup, destination)
            except BaseException:
                if displaced is not None and displaced.exists() and not destination.exists():
                    os.replace(displaced, destination)
                raise
            if displaced is not None and displaced.exists():
                shutil.rmtree(displaced)
        elif destination.exists() and not staging.exists():
            os.replace(destination, staging)
        raise
    if backup.exists():
        shutil.rmtree(backup)

def publication_control_paths(destination: Path) -> tuple[Path, Path]:
    """Return external lock/journal paths for one publication destination."""

    destination = plain_absolute_path(
        destination,
        label="Core-overlay publication destination",
    )
    state_home = Path(
        os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
    )
    lock_root = state_home / "n_response_curve" / "publication"
    lock_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root_info = os.lstat(lock_root)
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or root_info.st_uid != os.geteuid()
        or stat.S_IMODE(root_info.st_mode) & 0o077
    ):
        raise RuntimeError(
            f"Publication control root must be an owner-only plain directory: {lock_root}"
        )
    token = hashlib.sha256(str(destination).encode("utf-8")).hexdigest()[:24]
    return (
        lock_root / f"core-overlay-{token}.lock",
        lock_root / f"core-overlay-{token}.transaction.json",
    )

def nested_publication_container(destination: Path) -> Path:
    """Resolve the parent snapshot that owns a nested clusters/ publisher."""

    destination = plain_absolute_path(
        destination, label="Nested overlay publication destination"
    )
    for ancestor in destination.parents:
        if ancestor.name == "clusters":
            return ancestor.parent
    return destination

@contextmanager
def publication_lock(destination: Path):
    """Serialize parent and nested writers for one core-overlay container."""

    lock_path, _ = publication_control_paths(destination)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Core-overlay publication is already running for {destination}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)

def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def directory_state_fingerprint(path: Path) -> str:
    """Hash one plain directory tree without following links or hardlinks."""

    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"Publication candidate is not a plain directory: {path}")
    entries: list[tuple[str, str, int, int, str]] = []
    pending = [(path, Path())]
    while pending:
        directory, relative_directory = pending.pop()
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            relative = relative_directory / entry.name
            info = os.lstat(entry.path)
            if stat.S_ISLNK(info.st_mode):
                raise RuntimeError(f"Publication candidate contains a link: {entry.path}")
            if stat.S_ISDIR(info.st_mode):
                entries.append(
                    ("d", relative.as_posix(), stat.S_IMODE(info.st_mode), 0, "")
                )
                pending.append((Path(entry.path), relative))
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise RuntimeError(
                    f"Publication candidate contains an unsafe file: {entry.path}"
                )
            digest = hashlib.sha256()
            with open(entry.path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            entries.append(
                (
                    "f",
                    relative.as_posix(),
                    stat.S_IMODE(info.st_mode),
                    info.st_size,
                    digest.hexdigest(),
                )
            )
    encoded = json.dumps(sorted(entries), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

def publication_journal_path(destination: Path) -> Path:
    return publication_control_paths(destination)[1]

def write_publication_journal(
    destination: Path, staging: Path, backup: Path
) -> Path:
    """Record exact pre/post images outside the operator output tree."""

    journal = publication_journal_path(destination)
    temporary = journal.with_name(journal.name + ".tmp")
    if journal.exists() or journal.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"Directory publication control residue is present: {journal}")
    payload = {
        "schema_version": PUBLICATION_JOURNAL_SCHEMA,
        "destination": str(destination),
        "staging_name": staging.name,
        "backup_name": backup.name,
        "had_prior": destination.exists(),
        "staged_state_sha256": directory_state_fingerprint(staging),
        "prior_state_sha256": (
            directory_state_fingerprint(destination) if destination.exists() else None
        ),
    }
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, journal)
        sync_directory(journal.parent)
    except BaseException:
        for path in (temporary, journal):
            if path.exists() and not path.is_symlink():
                path.unlink()
        sync_directory(journal.parent)
        raise
    return journal

def remove_publication_journal(path: Path) -> None:
    if path.exists():
        path.unlink()
        sync_directory(path.parent)

def recover_interrupted_directory_publication(destination: Path) -> None:
    """Recover only residue authorized by the external publication journal."""

    destination = plain_absolute_path(
        destination,
        label="Directory publication destination",
    )
    parent = destination.parent
    if not parent.exists():
        return
    journal = publication_journal_path(destination)
    journal_temporary = journal.with_name(journal.name + ".tmp")
    backups = sorted(parent.glob(f".{destination.name}.backup.*"))
    staging_directories = sorted(parent.glob(f".{destination.name}.staging.*"))
    failed_directories = sorted(parent.glob(f".{destination.name}.failed.*"))
    residues = (*backups, *staging_directories, *failed_directories)
    for residue in residues:
        if residue.is_symlink() or not residue.is_dir():
            raise RuntimeError(
                f"Directory publication residue is not a plain directory: {residue}"
            )
    if journal_temporary.exists() or journal_temporary.is_symlink():
        if journal.exists() or journal.is_symlink():
            raise RuntimeError(
                "Directory publication has both final and temporary external journals"
            )
        temporary_info = os.lstat(journal_temporary)
        if (
            not stat.S_ISREG(temporary_info.st_mode)
            or temporary_info.st_nlink != 1
        ):
            raise RuntimeError(
                f"Directory publication temporary journal is unsafe: {journal_temporary}"
            )
        try:
            temporary_payload = json.loads(
                journal_temporary.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Directory publication temporary journal is invalid: {journal_temporary}"
            ) from exc
        if (
            not isinstance(temporary_payload, dict)
            or temporary_payload.get("schema_version")
            != PUBLICATION_JOURNAL_SCHEMA
            or temporary_payload.get("destination") != str(destination)
        ):
            raise RuntimeError(
                "Directory publication temporary journal does not bind this destination"
            )
        os.replace(journal_temporary, journal)
        sync_directory(journal.parent)
    if not journal.exists() and not journal.is_symlink():
        if residues:
            raise RuntimeError(
                "Directory publication residue has no external journal authorizing "
                f"recovery; preserve it for operator review: {residues}"
            )
        if destination.exists() and (
            destination.is_symlink() or not destination.is_dir()
        ):
            raise RuntimeError("Directory publication destination is not a plain directory")
        return
    info = os.lstat(journal)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError(f"Directory publication journal is unsafe: {journal}")
    try:
        payload = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Directory publication journal is invalid: {journal}") from exc
    required = {
        "schema_version",
        "destination",
        "staging_name",
        "backup_name",
        "had_prior",
        "staged_state_sha256",
        "prior_state_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload.get("schema_version") != PUBLICATION_JOURNAL_SCHEMA
        or payload.get("destination") != str(destination)
        or not isinstance(payload.get("had_prior"), bool)
        or not isinstance(payload.get("staged_state_sha256"), str)
        or len(payload["staged_state_sha256"]) != 64
    ):
        raise RuntimeError(f"Directory publication journal has an invalid schema: {journal}")
    had_prior = payload["had_prior"]
    prior_fingerprint = payload.get("prior_state_sha256")
    if had_prior != (isinstance(prior_fingerprint, str) and len(prior_fingerprint) == 64):
        raise RuntimeError(f"Directory publication journal has an invalid prior: {journal}")
    staging_name = payload.get("staging_name")
    backup_name = payload.get("backup_name")
    if (
        not isinstance(staging_name, str)
        or Path(staging_name).name != staging_name
        or not staging_name.startswith(f".{destination.name}.staging.")
        or not isinstance(backup_name, str)
        or Path(backup_name).name != backup_name
        or not backup_name.startswith(f".{destination.name}.backup.")
    ):
        raise RuntimeError(f"Directory publication journal names unsafe residue: {journal}")
    staging = parent / staging_name
    backup = parent / backup_name
    if any(path != staging for path in staging_directories) or any(
        path != backup for path in backups
    ):
        raise RuntimeError(
            "Directory publication has residue not authorized by its external journal"
        )
    staged_fingerprint = payload["staged_state_sha256"]

    def require_state(path: Path, expected: str, label: str) -> None:
        if directory_state_fingerprint(path) != expected:
            raise RuntimeError(
                f"Directory publication {label} changed after it was journaled: {path}"
            )

    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise RuntimeError("Directory publication destination is not a plain directory")
        destination_state = directory_state_fingerprint(destination)
    else:
        destination_state = None
    for residue in (*staging_directories, *failed_directories):
        require_state(residue, staged_fingerprint, "generated residue")

    if backup.exists():
        assert prior_fingerprint is not None
        require_state(backup, prior_fingerprint, "prior backup")
        if destination_state is not None:
            if destination_state != staged_fingerprint:
                raise RuntimeError(
                    "Directory publication live destination is neither journaled state"
                )
            shutil.rmtree(destination)
            sync_directory(parent)
        os.replace(backup, destination)
        sync_directory(parent)
    elif had_prior:
        if destination_state not in {prior_fingerprint, staged_fingerprint}:
            raise RuntimeError(
                "Directory publication lost its journaled prior; preserve all residue"
            )
    elif destination_state not in {None, staged_fingerprint}:
        raise RuntimeError(
            "Directory publication first-generation destination is not journaled"
        )
    for residue in (*staging_directories, *failed_directories):
        if residue.exists():
            shutil.rmtree(residue)
    sync_directory(parent)
    remove_publication_journal(journal)

def promote_staged_directory(
    staging: Path, destination: Path, backup: Path
) -> None:
    """Publish one staged snapshot, restoring its prior on any cooperative exit."""

    if staging.is_symlink() or not staging.is_dir():
        raise RuntimeError("Directory publication staging path is not a plain directory")
    journal = write_publication_journal(destination, staging, backup)
    try:
        if destination.exists():
            if not destination.is_dir() or destination.is_symlink():
                raise RuntimeError(
                    "Directory publication destination is not a plain directory"
                )
            os.replace(destination, backup)
            sync_directory(destination.parent)
        os.replace(staging, destination)
        sync_directory(destination.parent)
    except BaseException:
        if backup.exists():
            displaced: Path | None = None
            if destination.exists():
                displaced = destination.with_name(
                    f".{destination.name}.failed.{uuid.uuid4().hex}"
                )
                os.replace(destination, displaced)
                sync_directory(destination.parent)
            try:
                os.replace(backup, destination)
                sync_directory(destination.parent)
            except BaseException:
                if (
                    displaced is not None
                    and displaced.exists()
                    and not destination.exists()
                ):
                    os.replace(displaced, destination)
                    sync_directory(destination.parent)
                raise
            if displaced is not None and displaced.exists():
                shutil.rmtree(displaced)
                sync_directory(destination.parent)
        elif destination.exists() and not staging.exists():
            # First publication has no backup. A rename can still complete
            # before KeyboardInterrupt is delivered, so move that uncommitted
            # generation back to staging and restore the original absence.
            os.replace(destination, staging)
            sync_directory(destination.parent)
        remove_publication_journal(journal)
        raise
    if backup.exists():
        shutil.rmtree(backup)
        sync_directory(destination.parent)
    remove_publication_journal(journal)

def copy_preserved_plain_tree(source: Path, destination: Path) -> None:
    """Copy an owned companion tree without following links of any kind."""

    directories: set[Path] = {Path()}
    files: list[Path] = []
    pending = [(source, Path())]
    while pending:
        directory, relative_directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise RuntimeError(
                f"Preserved cluster tree could not be inspected: {directory}"
            ) from exc
        for entry in entries:
            relative = relative_directory / entry.name
            try:
                info = os.lstat(entry.path)
            except OSError as exc:
                raise RuntimeError(
                    f"Preserved cluster entry could not be inspected: {source / relative}"
                ) from exc
            if stat.S_ISLNK(info.st_mode):
                raise RuntimeError(
                    f"Preserved cluster tree contains a symlink: {source / relative}"
                )
            if stat.S_ISDIR(info.st_mode):
                directories.add(relative)
                pending.append((Path(entry.path), relative))
                continue
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError(
                    "Preserved cluster tree contains a non-regular entry: "
                    f"{source / relative}"
                )
            if info.st_nlink > 1:
                raise RuntimeError(
                    f"Preserved cluster tree contains a hardlink: {source / relative}"
                )
            files.append(relative)

    for relative in sorted(directories, key=lambda path: (len(path.parts), path.parts)):
        (destination / relative).mkdir()
    for relative in sorted(files):
        source_file = source / relative
        destination_file = destination / relative
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(source_file, flags)
        try:
            current = os.fstat(descriptor)
            if not stat.S_ISREG(current.st_mode) or current.st_nlink > 1:
                raise RuntimeError(
                    f"Preserved cluster file changed during copy: {source_file}"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as source_handle:
                with destination_file.open("xb") as destination_handle:
                    shutil.copyfileobj(source_handle, destination_handle)
        finally:
            os.close(descriptor)
