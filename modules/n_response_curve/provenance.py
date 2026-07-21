from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

from .config import ConfigError


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SourceIntegrityReport:
    """Checksum verification outcome for the immutable intake package."""

    checked_files: int
    failed_files: tuple[str, ...]
    artifact_sha256: Mapping[str, str]


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without modifying it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_artifact_path(source_root: Path, relative: str, *, where: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ConfigError(f"{where} artifact path is outside the source root: {relative}")
    resolved = (source_root / candidate).resolve()
    try:
        resolved.relative_to(source_root)
    except ValueError as exc:
        raise ConfigError(f"{where} artifact path is outside the source root: {relative}") from exc
    return resolved


def verify_source_integrity(manifest_path: str | Path, checksums_path: str | Path) -> SourceIntegrityReport:
    """Verify that manifest and checksum ledger entries describe unchanged files."""

    manifest = Path(manifest_path).resolve()
    checksums = Path(checksums_path).resolve()
    source_root = manifest.parent
    expected_from_checksums: dict[str, str] = {}
    checksum_paths: dict[str, Path] = {}
    failed: set[str] = set()

    with checksums.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                expected, relative = stripped.split(maxsplit=1)
            except ValueError as exc:
                raise ConfigError(f"Malformed checksum line {line_number}: {stripped!r}") from exc
            expected = expected.lower()
            if not _SHA256_RE.fullmatch(expected):
                raise ConfigError(f"Malformed SHA-256 value on checksum line {line_number}: {expected!r}")
            if relative in expected_from_checksums:
                raise ConfigError(f"Duplicate checksum artifact path on line {line_number}: {relative}")
            checksum_paths[relative] = _resolve_artifact_path(
                source_root,
                relative,
                where=f"Checksum line {line_number}",
            )
            expected_from_checksums[relative] = expected
    if not expected_from_checksums:
        raise ConfigError(f"Source checksum ledger contains no valid artifact entries: {checksums}")

    checked_paths: set[str] = set()
    manifest_artifacts: dict[str, str] = {}
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        required_columns = {"artifact_path", "sha256"}
        if not required_columns.issubset(rows.fieldnames or set()):
            raise ConfigError("Source manifest must contain required columns: artifact_path, sha256")
        for row_number, row in enumerate(rows, start=2):
            relative = (row.get("artifact_path") or "").strip()
            expected = (row.get("sha256") or "").strip().lower()
            if not relative or not _SHA256_RE.fullmatch(expected):
                failed.add(f"manifest:{row_number}")
                continue
            if relative in checked_paths:
                raise ConfigError(f"Duplicate manifest artifact_path on row {row_number}: {relative}")
            path = _resolve_artifact_path(
                source_root,
                relative,
                where=f"Manifest row {row_number}",
            )
            checked_paths.add(relative)
            manifest_artifacts[relative] = expected
            if not path.is_file() or sha256_file(path) != expected:
                failed.add(relative)
            if expected_from_checksums.get(relative) != expected:
                failed.add(relative)
    if not checked_paths:
        raise ConfigError(f"Source manifest contains no valid artifact entries: {manifest}")

    for relative, expected in expected_from_checksums.items():
        if relative not in checked_paths:
            failed.add(relative)
            continue
        path = checksum_paths[relative]
        if not path.is_file() or sha256_file(path) != expected:
            failed.add(relative)

    return SourceIntegrityReport(
        checked_files=len(checked_paths),
        failed_files=tuple(sorted(failed)),
        artifact_sha256=MappingProxyType(dict(sorted(manifest_artifacts.items()))),
    )


__all__ = ["SourceIntegrityReport", "sha256_file", "verify_source_integrity"]
