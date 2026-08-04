from __future__ import annotations

from dataclasses import dataclass
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping
import zipfile

from .config import ConfigError


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_COLUMNS = (
    "artifact_path",
    "role",
    "priority",
    "source_locator",
    "relationship",
    "rows",
    "columns",
    "bytes",
    "sha256",
    "scope_reason",
)
_MANIFEST_ROLES = frozenset(
    {
        "source_workbook",
        "core_trial_data",
        "paired_management_trial_data",
        "variety_lookup",
        "annual_reports_sheet",
        "annual_report_project_index",
        "field_list",
        "source_notes",
        "duplicate_source_export",
    }
)
_MANIFEST_PRIORITIES = frozenset(
    {"primary", "supporting", "documentation", "provenance", "source_evidence"}
)
_CONTROL_ARTIFACT_NAMES = frozenset({"manifest.csv", "checksums.sha256"})
_ROW_DECLARATION_RE = re.compile(r"^(?P<count>\d+) (?P<kind>data rows|physical rows|sheets)$")


@dataclass(frozen=True)
class SourceIntegrityReport:
    """Checksum verification outcome for the immutable intake package."""

    checked_files: int
    failed_files: tuple[str, ...]
    artifact_sha256: Mapping[str, str]
    artifact_metadata: Mapping[str, Mapping[str, str]]
    duplicate_byte_groups: tuple[tuple[str, ...], ...]
    unverified_relationships: tuple[str, ...]


@dataclass(frozen=True)
class CsvStructure:
    """Read-only structural evidence used when a registered CSV changes."""

    path: Path
    sha256: str
    byte_count: int
    encoding: str
    physical_column_count: int
    logical_row_count: int
    nonblank_data_row_count: int
    header_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "encoding": self.encoding,
            "physical_column_count": self.physical_column_count,
            "logical_row_count": self.logical_row_count,
            "nonblank_data_row_count": self.nonblank_data_row_count,
            "header_sha256": self.header_sha256,
        }


@dataclass(frozen=True)
class ChecksumRevisionComparison:
    """Automated old/new structural comparison; this is not human approval."""

    prior: CsvStructure
    candidate: CsvStructure
    changed_fields: tuple[str, ...]
    comparison_sha256: str


@dataclass(frozen=True)
class ChecksumRevisionApproval:
    """Artifact-specific designated-reviewer evidence for a checksum revision."""

    artifact_path: str
    reviewer: str
    reviewed_on: str
    rationale: str
    old_sha256: str
    new_sha256: str
    manifest_revision: str
    structural_comparison_sha256: str
    prior_registered_path: Path


@dataclass(frozen=True)
class SourceScopeApproval:
    """Human approval bound to one exact authoritative source-scope snapshot."""

    reviewer: str
    reviewed_on: str
    rationale: str
    snapshot_sha256: str
    scope_revision: str


@dataclass(frozen=True)
class SourceActivation:
    """Sources admitted by one exact approved scope and prerequisite set."""

    snapshot_sha256: str
    scope_revision: str
    activated_sources: tuple[str, ...]


@dataclass(frozen=True)
class VerificationSamplingPolicy:
    """Reviewed risk-stratified sequential verification design."""

    version: str
    review_id: str
    stratum_fields: tuple[str, ...]
    risk_field: str
    initial_sample_per_stratum: int
    escalation_sample_per_stratum: int
    discrepancy_thresholds: Mapping[str, int]
    maximum_rounds: int
    seed: str


@dataclass(frozen=True)
class VerificationResult:
    """Completed check for one literature record."""

    record_uid: str
    round_number: int
    severity: str
    reviewer: str
    reviewed_on: str
    evidence: str


@dataclass(frozen=True)
class VerificationRound:
    """Next deterministic sample or a terminal verification state."""

    round_number: int
    status: str
    selected_record_uids: tuple[str, ...]
    escalated_strata: tuple[tuple[str, ...], ...]
    limitation_reasons: tuple[str, ...]


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without modifying it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_value(value: object) -> object:
    """Normalize immutable containers before canonical JSON serialization."""

    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(nested)
            for key, nested in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_json_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
        )
    return value


def stable_json_sha256(payload: object) -> str:
    """Return the SHA-256 digest of the repository's canonical JSON encoding."""

    encoded = json.dumps(
        _canonical_json_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_identifier(prefix: str, payload: object) -> str:
    """Return a prefixed identifier using the canonical 24-character digest."""

    return f"{prefix}_{stable_json_sha256(payload)[:24]}"


def _nonempty_text(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a nonempty string")
    return value.strip()


def _iso_date(value: object, *, where: str) -> str:
    text = _nonempty_text(value, where=where)
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise ConfigError(f"{where} must be an ISO date (YYYY-MM-DD)") from exc
    return text


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


__all__ = [
    "SourceIntegrityReport",
    "sha256_file",
    "stable_identifier",
    "stable_json_sha256",
    "verify_source_integrity",
]
