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
    prior_context: Mapping[str, object]
    candidate_context: Mapping[str, object]
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
    prior_encoding: str
    candidate_encoding: str
    prior_data_classification: str
    candidate_data_classification: str
    prior_workbook_sha256: str | None
    candidate_workbook_sha256: str | None
    prior_workbook_path: Path | None
    prior_sheet: str | None
    candidate_sheet: str | None
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


def _resolve_source_locator(source_root: Path, locator: str, *, where: str) -> Path:
    """Resolve only the file component of a workbook/sheet or cell-range locator."""

    file_component = locator.split("#", maxsplit=1)[0]
    relative = Path(file_component)
    if relative.is_absolute() or ".." in relative.parts:
        raise ConfigError(f"{where} artifact path is outside the intake roots: {locator}")
    within_manifest_root = (source_root / relative).resolve()
    if within_manifest_root.exists():
        return within_manifest_root
    return (source_root.parent / relative).resolve()


def _decode_csv_bytes(path: Path) -> tuple[str, str]:
    payload = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return payload.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ConfigError(f"Manifest CSV artifact cannot be decoded as registered workflow text: {path}")


def _csv_dimensions(path: Path) -> tuple[int, int, int]:
    text, _ = _decode_csv_bytes(path)
    try:
        rows = list(csv.reader(text.splitlines(), strict=True))
    except csv.Error as exc:
        raise ConfigError(f"Malformed registered CSV artifact {path}: {exc}") from exc
    if not rows:
        return 0, 0, 0
    columns = len(rows[0])
    nonblank_data_rows = sum(
        1
        for row in rows[1:]
        if row and any(cell != "" for cell in row)
    )
    return columns, len(text.splitlines()), nonblank_data_rows


def _xlsx_sheet_count(path: Path) -> int:
    try:
        with zipfile.ZipFile(path) as archive:
            workbook_xml = archive.read("xl/workbook.xml")
    except (KeyError, zipfile.BadZipFile, OSError) as exc:
        raise ConfigError(f"Registered workbook is not a readable XLSX package: {path}") from exc
    return len(re.findall(rb"<sheet\b", workbook_xml))


def _validate_manifest_dimensions(path: Path, row: Mapping[str, str], *, row_number: int) -> None:
    try:
        declared_bytes = int((row.get("bytes") or "").strip())
    except ValueError as exc:
        raise ConfigError(f"Manifest row {row_number} has an invalid bytes declaration") from exc
    if declared_bytes < 0:
        raise ConfigError(f"Manifest row {row_number} has a negative bytes declaration")
    if path.is_file() and path.stat().st_size != declared_bytes:
        raise ConfigError(
            f"Manifest row {row_number} byte count differs from the registered artifact: {path}"
        )

    row_declaration = (row.get("rows") or "").strip()
    match = _ROW_DECLARATION_RE.fullmatch(row_declaration)
    if match is None:
        raise ConfigError(
            f"Manifest row {row_number} rows declaration must be '<count> data rows', "
            "'<count> physical rows', or '<count> sheets'"
        )
    declared_count = int(match.group("count"))
    declared_kind = match.group("kind")

    columns_text = (row.get("columns") or "").strip()
    if columns_text:
        try:
            declared_columns = int(columns_text)
        except ValueError as exc:
            raise ConfigError(f"Manifest row {row_number} has an invalid columns declaration") from exc
        if declared_columns < 1:
            raise ConfigError(f"Manifest row {row_number} columns must be positive when present")
    else:
        declared_columns = None

    suffix = path.suffix.casefold()
    if suffix == ".csv":
        observed_columns, physical_rows, nonblank_data_rows = _csv_dimensions(path)
        if declared_columns is not None and observed_columns != declared_columns:
            raise ConfigError(
                f"Manifest row {row_number} physical column count differs from the artifact: {path}"
            )
        observed_rows = (
            nonblank_data_rows if declared_kind == "data rows" else physical_rows
        )
        if declared_kind == "sheets" or observed_rows != declared_count:
            raise ConfigError(
                f"Manifest row {row_number} row declaration differs from the artifact: {path}"
            )
    elif suffix == ".xlsx":
        if declared_columns is not None:
            raise ConfigError(f"Manifest row {row_number} workbook columns must be blank")
        if declared_kind != "sheets" or _xlsx_sheet_count(path) != declared_count:
            raise ConfigError(
                f"Manifest row {row_number} sheet declaration differs from the workbook: {path}"
            )
    else:
        raise ConfigError(f"Manifest row {row_number} uses an unsupported artifact format: {path}")


def _validate_manifest_vocabulary(row: Mapping[str, str], *, row_number: int) -> None:
    role = (row.get("role") or "").strip()
    priority = (row.get("priority") or "").strip()
    relationship = (row.get("relationship") or "").strip()
    if role not in _MANIFEST_ROLES:
        raise ConfigError(f"Manifest row {row_number} has an unregistered role: {role!r}")
    if priority not in _MANIFEST_PRIORITIES:
        raise ConfigError(f"Manifest row {row_number} has an unregistered priority: {priority!r}")
    relationship_head = relationship.partition(":")[0].partition(";")[0].strip()
    if relationship_head not in {"exact_copy", "faithful_cell_value_export", "derived"}:
        raise ConfigError(
            f"Manifest row {row_number} has an unregistered relationship: {relationship!r}"
        )
    _nonempty_text(row.get("source_locator"), where=f"Manifest row {row_number} source_locator")
    _nonempty_text(row.get("scope_reason"), where=f"Manifest row {row_number} scope_reason")


def verify_source_integrity(
    manifest_path: str | Path,
    checksums_path: str | Path,
    *,
    provisional_revision_paths: Iterable[str] = (),
) -> SourceIntegrityReport:
    """Verify intake bytes, deferring only explicitly approved revision candidates.

    A provisional path remains bound to the old manifest digest in the returned
    report.  Its replacement bytes must subsequently pass
    :func:`validate_checksum_revision_approval`; this parameter only prevents
    the old ledger from making that artifact-specific validation unreachable.
    """

    manifest = Path(manifest_path).resolve()
    checksums = Path(checksums_path).resolve()
    source_root = manifest.parent
    provisional_revisions: set[str] = set()
    for raw_path in provisional_revision_paths:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ConfigError("Provisional checksum-revision paths must be nonempty strings")
        relative = raw_path.strip()
        _resolve_artifact_path(source_root, relative, where="Provisional checksum revision")
        provisional_revisions.add(relative)
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
    manifest_metadata: dict[str, Mapping[str, str]] = {}
    unverified_relationships: set[str] = set()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        if tuple(rows.fieldnames or ()) != _MANIFEST_COLUMNS:
            raise ConfigError(
                "Source manifest must contain the exact ten-column contract in order: "
                + ", ".join(_MANIFEST_COLUMNS)
            )
        for row_number, row in enumerate(rows, start=2):
            relative = (row.get("artifact_path") or "").strip()
            expected = (row.get("sha256") or "").strip().lower()
            if not relative or not _SHA256_RE.fullmatch(expected):
                failed.add(f"manifest:{row_number}")
                continue
            if Path(relative).name.casefold() in _CONTROL_ARTIFACT_NAMES:
                raise ConfigError(
                    f"Manifest row {row_number} registers a control file as a source artifact: {relative}"
                )
            if relative in checked_paths:
                raise ConfigError(f"Duplicate manifest artifact_path on row {row_number}: {relative}")
            _validate_manifest_vocabulary(row, row_number=row_number)
            path = _resolve_artifact_path(
                source_root,
                relative,
                where=f"Manifest row {row_number}",
            )
            checked_paths.add(relative)
            manifest_artifacts[relative] = expected
            manifest_metadata[relative] = MappingProxyType(
                {column: (row.get(column) or "").strip() for column in _MANIFEST_COLUMNS}
            )
            if relative not in provisional_revisions and (
                not path.is_file() or sha256_file(path) != expected
            ):
                failed.add(relative)
            if path.is_file() and relative not in provisional_revisions:
                _validate_manifest_dimensions(path, row, row_number=row_number)
            if expected_from_checksums.get(relative) != expected:
                failed.add(relative)

            locator = _resolve_source_locator(
                source_root,
                (row.get("source_locator") or "").strip(),
                where=f"Manifest row {row_number} source_locator",
            )
            relationship = (row.get("relationship") or "").strip()
            if relationship.startswith("exact_copy") and relative not in provisional_revisions:
                if not locator.is_file():
                    unverified_relationships.add(relative)
                elif sha256_file(locator) != expected:
                    failed.add(relative)
    if not checked_paths:
        raise ConfigError(f"Source manifest contains no valid artifact entries: {manifest}")

    for relative, expected in expected_from_checksums.items():
        if relative not in checked_paths:
            failed.add(relative)
            continue
        path = checksum_paths[relative]
        if relative not in provisional_revisions and (
            not path.is_file() or sha256_file(path) != expected
        ):
            failed.add(relative)

    unknown_revisions = provisional_revisions - checked_paths
    if unknown_revisions:
        raise ConfigError(
            "Provisional checksum revision is not registered in the manifest: "
            + ", ".join(sorted(unknown_revisions))
        )

    by_sha256: dict[str, list[str]] = {}
    for relative, expected in manifest_artifacts.items():
        by_sha256.setdefault(expected, []).append(relative)
    duplicate_byte_groups = tuple(
        tuple(sorted(paths))
        for _, paths in sorted(by_sha256.items())
        if len(paths) > 1
    )

    return SourceIntegrityReport(
        checked_files=len(checked_paths),
        failed_files=tuple(sorted(failed)),
        artifact_sha256=MappingProxyType(dict(sorted(manifest_artifacts.items()))),
        artifact_metadata=MappingProxyType(dict(sorted(manifest_metadata.items()))),
        duplicate_byte_groups=duplicate_byte_groups,
        unverified_relationships=tuple(sorted(unverified_relationships)),
    )


def inspect_csv_structure(path: str | Path, *, encoding: str) -> CsvStructure:
    """Collect comparison evidence using the artifact's registered encoding."""

    source = Path(path).resolve()
    if not source.is_file():
        raise ConfigError(f"CSV structure comparison source does not exist: {source}")
    digest_before = sha256_file(source)
    try:
        with source.open("r", encoding=encoding, newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ConfigError(f"CSV structure comparison source is empty: {source}") from exc
            logical_rows = 1
            nonblank_rows = 0
            for row in reader:
                logical_rows += 1
                if row and any(cell != "" for cell in row):
                    nonblank_rows += 1
    except (UnicodeError, LookupError) as exc:
        raise ConfigError(
            f"CSV structure comparison failed under registered encoding {encoding!r}: {source}"
        ) from exc
    except csv.Error as exc:
        raise ConfigError(f"Malformed CSV during structural comparison: {source}: {exc}") from exc
    if sha256_file(source) != digest_before:
        raise ConfigError(f"Source bytes changed during structural comparison: {source}")
    return CsvStructure(
        path=source,
        sha256=digest_before,
        byte_count=source.stat().st_size,
        encoding=encoding,
        physical_column_count=len(header),
        logical_row_count=logical_rows,
        nonblank_data_row_count=nonblank_rows,
        header_sha256=stable_json_sha256(header),
    )


def compare_checksum_revision(
    prior_registered_path: str | Path,
    candidate_path: str | Path,
    *,
    prior_encoding: str,
    candidate_encoding: str,
    prior_data_classification: str = "internal",
    candidate_data_classification: str = "internal",
    prior_workbook_sha256: str | None = None,
    candidate_workbook_sha256: str | None = None,
    prior_sheet: str | None = None,
    candidate_sheet: str | None = None,
) -> ChecksumRevisionComparison:
    """Create byte, structure, encoding, classification, and workbook/sheet evidence."""

    prior = inspect_csv_structure(prior_registered_path, encoding=prior_encoding)
    candidate = inspect_csv_structure(candidate_path, encoding=candidate_encoding)
    comparable_fields = (
        "byte_count",
        "encoding",
        "physical_column_count",
        "logical_row_count",
        "nonblank_data_row_count",
        "header_sha256",
    )
    structural_changed = tuple(
        field
        for field in comparable_fields
        if getattr(prior, field) != getattr(candidate, field)
    )
    prior_context = {
        "data_classification": _source_classification(
            prior_data_classification,
            where="prior source revision classification",
        ),
        "workbook_sha256": _optional_sha256(
            prior_workbook_sha256,
            where="prior source revision workbook_sha256",
        ),
        "workbook_sheet": _optional_text(prior_sheet),
    }
    candidate_context = {
        "data_classification": _source_classification(
            candidate_data_classification,
            where="candidate source revision classification",
        ),
        "workbook_sha256": _optional_sha256(
            candidate_workbook_sha256,
            where="candidate source revision workbook_sha256",
        ),
        "workbook_sheet": _optional_text(candidate_sheet),
    }
    context_changed = tuple(
        field
        for field in ("data_classification", "workbook_sha256", "workbook_sheet")
        if prior_context[field] != candidate_context[field]
    )
    changed = (*structural_changed, *context_changed)
    payload = {
        "prior": prior.as_dict(),
        "candidate": candidate.as_dict(),
        "prior_context": prior_context,
        "candidate_context": candidate_context,
        "changed_fields": changed,
    }
    return ChecksumRevisionComparison(
        prior=prior,
        candidate=candidate,
        prior_context=MappingProxyType(prior_context),
        candidate_context=MappingProxyType(candidate_context),
        changed_fields=changed,
        comparison_sha256=stable_json_sha256(payload),
    )


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return _nonempty_text(value, where="source revision context")


def _optional_sha256(value: object, *, where: str) -> str | None:
    if value is None:
        return None
    digest = _nonempty_text(value, where=where).lower()
    if _SHA256_RE.fullmatch(digest) is None:
        raise ConfigError(f"{where} is not a SHA-256 digest")
    return digest


def _source_classification(value: object, *, where: str) -> str:
    classification = _nonempty_text(value, where=where)
    if classification not in {"internal", "restricted"}:
        raise ConfigError(f"{where} is unsupported")
    return classification


def parse_checksum_revision_approval(payload: Mapping[str, object]) -> ChecksumRevisionApproval:
    """Parse, but do not invent, an artifact-specific checksum acceptance record."""

    required = {
        "artifact_path",
        "reviewer",
        "reviewed_on",
        "rationale",
        "old_sha256",
        "new_sha256",
        "manifest_revision",
        "structural_comparison_sha256",
        "prior_encoding",
        "candidate_encoding",
        "prior_data_classification",
        "candidate_data_classification",
        "prior_workbook_sha256",
        "candidate_workbook_sha256",
        "prior_workbook_path",
        "prior_sheet",
        "candidate_sheet",
        "prior_registered_path",
    }
    missing = required - set(payload)
    if missing:
        raise ConfigError(
            "Checksum revision approval is missing field(s): " + ", ".join(sorted(missing))
        )
    old_sha256 = _nonempty_text(payload["old_sha256"], where="checksum approval old_sha256").lower()
    new_sha256 = _nonempty_text(payload["new_sha256"], where="checksum approval new_sha256").lower()
    comparison_sha256 = _nonempty_text(
        payload["structural_comparison_sha256"],
        where="checksum approval structural_comparison_sha256",
    ).lower()
    for label, digest in (
        ("old_sha256", old_sha256),
        ("new_sha256", new_sha256),
        ("structural_comparison_sha256", comparison_sha256),
    ):
        if _SHA256_RE.fullmatch(digest) is None:
            raise ConfigError(f"checksum approval {label} is not a SHA-256 digest")
    return ChecksumRevisionApproval(
        artifact_path=_nonempty_text(payload["artifact_path"], where="checksum approval artifact_path"),
        reviewer=_nonempty_text(payload["reviewer"], where="checksum approval reviewer"),
        reviewed_on=_iso_date(payload["reviewed_on"], where="checksum approval reviewed_on"),
        rationale=_nonempty_text(payload["rationale"], where="checksum approval rationale"),
        old_sha256=old_sha256,
        new_sha256=new_sha256,
        manifest_revision=_nonempty_text(
            payload["manifest_revision"],
            where="checksum approval manifest_revision",
        ),
        structural_comparison_sha256=comparison_sha256,
        prior_registered_path=Path(
            _nonempty_text(
                payload["prior_registered_path"],
                where="checksum approval prior_registered_path",
            )
        ).resolve(),
    )


def validate_checksum_revision_approval(
    approval: ChecksumRevisionApproval | Mapping[str, object],
    *,
    candidate_path: str | Path,
    expected_old_sha256: str,
    artifact_path: str,
    prior_encoding: str,
    candidate_encoding: str,
    designated_reviewers: Iterable[str],
) -> ChecksumRevisionComparison:
    """Accept only exact, reviewer-bound evidence for one changed artifact."""

    parsed = (
        approval
        if isinstance(approval, ChecksumRevisionApproval)
        else parse_checksum_revision_approval(approval)
    )
    reviewers = {str(reviewer).strip() for reviewer in designated_reviewers if str(reviewer).strip()}
    if not reviewers or parsed.reviewer not in reviewers:
        raise ConfigError("Checksum revision approval reviewer is not a designated reviewer")
    candidate = Path(candidate_path).resolve()
    actual_new_sha256 = sha256_file(candidate)
    expected_old = expected_old_sha256.lower()
    if (
        parsed.artifact_path != artifact_path
        or parsed.old_sha256 != expected_old
        or parsed.new_sha256 != actual_new_sha256
    ):
        raise ConfigError("Checksum revision approval is not bound to the exact old/new artifact")
    if not parsed.prior_registered_path.is_file():
        raise ConfigError("Checksum revision approval does not preserve an accessible prior artifact")
    if sha256_file(parsed.prior_registered_path) != expected_old:
        raise ConfigError("Preserved prior artifact does not match the registered old checksum")
    comparison = compare_checksum_revision(
        parsed.prior_registered_path,
        candidate,
        prior_encoding=prior_encoding,
        candidate_encoding=candidate_encoding,
    )
    if comparison.comparison_sha256 != parsed.structural_comparison_sha256:
        raise ConfigError("Checksum revision approval does not match the automated structural comparison")
    return comparison


def build_source_scope_snapshot(
    *,
    source_entries: Mapping[str, Mapping[str, Any]],
    enabled_sources: Iterable[str],
    integrity_report: SourceIntegrityReport,
    cutoff: str,
    scanned_roots: Iterable[str],
) -> Mapping[str, object]:
    """Build a deterministic, versionable scope candidate without asserting approval."""

    enabled = tuple(sorted(str(name) for name in enabled_sources))
    unknown = set(enabled) - set(source_entries)
    if unknown:
        raise ConfigError("Source scope references unknown source(s): " + ", ".join(sorted(unknown)))
    sources: list[dict[str, object]] = []
    for name in enabled:
        entry = source_entries[name]
        sources.append(
            {
                "source_name": name,
                "source_type": entry.get("source_type"),
                "source_family": entry.get("source_family"),
                "availability": entry.get("availability"),
                "confirmation_status": entry.get("confirmation_status"),
                "shape_adapter_version": entry.get("shape_adapter_version"),
                "encoding": entry.get("encoding"),
                "data_classification": entry.get("data_classification"),
                "data_path": entry.get("data_path"),
                "schema_map": entry.get("schema_map"),
                "manifest_artifact_path": entry.get("manifest_artifact_path"),
                "schema_map_status": entry.get("schema_map_status"),
                "duplicate_review_status": entry.get("duplicate_review_status"),
                "restricted_controls_status": entry.get("restricted_controls_status"),
            }
        )
    payload: dict[str, object] = {
        "snapshot_format": "source-scope-v1",
        "cutoff": _nonempty_text(cutoff, where="source scope cutoff"),
        "scanned_roots": tuple(sorted(_nonempty_text(root, where="scanned root") for root in scanned_roots)),
        "sources": tuple(sources),
        "manifest_artifact_count": integrity_report.checked_files,
        "manifest_artifact_sha256": dict(integrity_report.artifact_sha256),
        "duplicate_byte_groups": integrity_report.duplicate_byte_groups,
    }
    payload["snapshot_sha256"] = stable_json_sha256(payload)
    return MappingProxyType(payload)


def validate_source_scope_approval(
    snapshot: Mapping[str, object],
    approval: SourceScopeApproval | Mapping[str, object],
    *,
    designated_reviewers: Iterable[str],
) -> SourceScopeApproval:
    """Validate explicit approval for the exact source-scope snapshot."""

    if isinstance(approval, SourceScopeApproval):
        parsed = approval
    else:
        parsed = SourceScopeApproval(
            reviewer=_nonempty_text(approval.get("reviewer"), where="source scope reviewer"),
            reviewed_on=_iso_date(approval.get("reviewed_on"), where="source scope reviewed_on"),
            rationale=_nonempty_text(approval.get("rationale"), where="source scope rationale"),
            snapshot_sha256=_nonempty_text(
                approval.get("snapshot_sha256"),
                where="source scope snapshot_sha256",
            ).lower(),
            scope_revision=_nonempty_text(
                approval.get("scope_revision"),
                where="source scope revision",
            ),
        )
    reviewers = {str(reviewer).strip() for reviewer in designated_reviewers if str(reviewer).strip()}
    if parsed.reviewer not in reviewers:
        raise ConfigError("Source-scope approval reviewer is not a designated reviewer")
    expected = snapshot.get("snapshot_sha256")
    if not isinstance(expected, str) or parsed.snapshot_sha256 != expected:
        raise ConfigError("Source-scope approval is not bound to the exact snapshot")
    return parsed


def validate_source_activation(
    snapshot: Mapping[str, object],
    approval: SourceScopeApproval | Mapping[str, object],
    *,
    integrity_report: SourceIntegrityReport,
    designated_reviewers: Iterable[str],
) -> SourceActivation:
    """Admit no source until every source-bound prerequisite is evidenced."""

    parsed = validate_source_scope_approval(
        snapshot,
        approval,
        designated_reviewers=designated_reviewers,
    )
    snapshot_sources = snapshot.get("sources")
    if not isinstance(snapshot_sources, (list, tuple)):
        raise ConfigError("Source-scope snapshot does not contain a source registry")
    activated: list[str] = []
    for item in snapshot_sources:
        if not isinstance(item, Mapping):
            raise ConfigError("Source-scope snapshot source entries must be mappings")
        source_name = _nonempty_text(
            item.get("source_name"),
            where="source-scope source name",
        )
        if item.get("availability") != "available":
            raise ConfigError(f"Source is not available for activation: {source_name}")
        if item.get("confirmation_status") != "verified":
            raise ConfigError(f"Source identity is not verified for activation: {source_name}")
        adapter = item.get("shape_adapter_version")
        if not isinstance(adapter, str) or not adapter.strip() or adapter == "unassigned":
            raise ConfigError(f"Source adapter is not assigned for activation: {source_name}")
        if item.get("schema_map_status") != "reviewed":
            raise ConfigError(f"Source schema map is not reviewed for activation: {source_name}")
        if item.get("duplicate_review_status") != "passed":
            raise ConfigError(f"Source duplicate review has not passed: {source_name}")
        if (
            item.get("data_classification") == "restricted"
            and item.get("restricted_controls_status") != "passed"
        ):
            raise ConfigError(f"Restricted-data controls have not passed: {source_name}")
        artifact_path = _nonempty_text(
            item.get("manifest_artifact_path"),
            where=f"source {source_name} manifest artifact path",
        )
        if (
            artifact_path not in integrity_report.artifact_sha256
            or artifact_path in integrity_report.failed_files
            or artifact_path in integrity_report.unverified_relationships
        ):
            raise ConfigError(
                f"Source artifact relationship and checksum are not fully verified: {source_name}"
            )
        activated.append(source_name)
    if not activated:
        raise ConfigError("Approved source scope activates no sources")
    return SourceActivation(
        snapshot_sha256=str(snapshot["snapshot_sha256"]),
        scope_revision=parsed.scope_revision,
        activated_sources=tuple(sorted(activated)),
    )


def _verification_stratum(
    record: Mapping[str, object],
    policy: VerificationSamplingPolicy,
) -> tuple[str, ...]:
    values: list[str] = []
    for field in (*policy.stratum_fields, policy.risk_field):
        value = record.get(field)
        text = str(value).strip() if value is not None else ""
        values.append(text or "unresolved")
    return tuple(values)


def _validate_verification_policy(policy: VerificationSamplingPolicy) -> None:
    _nonempty_text(policy.version, where="verification policy version")
    _nonempty_text(policy.review_id, where="verification policy review evidence")
    _nonempty_text(policy.risk_field, where="verification risk field")
    _nonempty_text(policy.seed, where="verification sampling seed")
    if not policy.stratum_fields or len(policy.stratum_fields) != len(
        set(policy.stratum_fields)
    ):
        raise ConfigError("Verification strata must be explicitly nonempty and unique")
    if policy.risk_field in policy.stratum_fields:
        raise ConfigError("Verification risk field must be separate from stratum fields")
    if policy.initial_sample_per_stratum < 1 or policy.escalation_sample_per_stratum < 1:
        raise ConfigError("Verification sample sizes must be positive")
    if policy.maximum_rounds < 1:
        raise ConfigError("Verification maximum rounds must be positive")
    if not policy.discrepancy_thresholds:
        raise ConfigError("Verification discrepancy thresholds must be predeclared")
    for severity, threshold in policy.discrepancy_thresholds.items():
        _nonempty_text(severity, where="verification discrepancy severity")
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1:
            raise ConfigError("Verification discrepancy thresholds must be positive integers")


def plan_literature_verification_round(
    records: Iterable[Mapping[str, object]],
    *,
    policy: VerificationSamplingPolicy,
    completed_results: Iterable[VerificationResult] = (),
    designated_reviewers: Iterable[str] = (),
) -> VerificationRound:
    """Plan or stop a deterministic risk-stratified sequential verification."""

    _validate_verification_policy(policy)
    rows = [dict(record) for record in records]
    record_uids = [str(record.get("record_uid", "")).strip() for record in rows]
    if not all(record_uids) or len(record_uids) != len(set(record_uids)):
        raise ConfigError("Literature verification records require unique nonempty identifiers")
    by_uid = dict(zip(record_uids, rows, strict=True))
    by_stratum: dict[tuple[str, ...], list[str]] = {}
    for uid, record in by_uid.items():
        by_stratum.setdefault(_verification_stratum(record, policy), []).append(uid)
    for stratum, members in by_stratum.items():
        members.sort(
            key=lambda uid: stable_json_sha256(
                (policy.seed, policy.version, stratum, uid)
            )
        )

    reviewer_set = {
        str(reviewer).strip()
        for reviewer in designated_reviewers
        if str(reviewer).strip()
    }
    results: dict[str, VerificationResult] = {}
    for result in completed_results:
        if result.record_uid not in by_uid:
            raise ConfigError("Verification result references a record outside the inventory")
        if result.record_uid in results:
            raise ConfigError("Literature record has more than one verification result")
        if result.round_number < 1 or result.round_number > policy.maximum_rounds:
            raise ConfigError("Verification result round number is outside the policy")
        if result.severity != "none" and result.severity not in policy.discrepancy_thresholds:
            raise ConfigError("Verification result uses an undeclared discrepancy severity")
        if result.reviewer not in reviewer_set:
            raise ConfigError("Verification result reviewer is not designated")
        _iso_date(result.reviewed_on, where="verification result reviewed_on")
        _nonempty_text(result.evidence, where="verification result evidence")
        results[result.record_uid] = result

    if not results:
        selected = tuple(
            uid
            for stratum in sorted(by_stratum)
            for uid in by_stratum[stratum][: policy.initial_sample_per_stratum]
        )
        return VerificationRound(
            round_number=1,
            status="sample_required",
            selected_record_uids=selected,
            escalated_strata=(),
            limitation_reasons=(),
        )

    highest_round = max(result.round_number for result in results.values())
    incomplete: list[str] = []
    for stratum, members in sorted(by_stratum.items()):
        expected = min(len(members), policy.initial_sample_per_stratum)
        checked = [uid for uid in members if uid in results]
        if len(checked) < expected:
            needed = expected - len(checked)
            unchecked = [uid for uid in members if uid not in results]
            incomplete.extend(unchecked[:needed])
    if incomplete:
        return VerificationRound(
            round_number=highest_round,
            status="sample_incomplete",
            selected_record_uids=tuple(incomplete),
            escalated_strata=(),
            limitation_reasons=(),
        )

    triggered: list[tuple[str, ...]] = []
    for stratum, members in sorted(by_stratum.items()):
        stratum_results = [results[uid] for uid in members if uid in results]
        if any(
            sum(result.severity == severity for result in stratum_results) >= threshold
            for severity, threshold in policy.discrepancy_thresholds.items()
        ):
            triggered.append(stratum)
    if not triggered:
        return VerificationRound(
            round_number=highest_round,
            status="pass",
            selected_record_uids=(),
            escalated_strata=(),
            limitation_reasons=(),
        )

    if highest_round >= policy.maximum_rounds:
        return VerificationRound(
            round_number=highest_round,
            status="explicit_limitation_required",
            selected_record_uids=(),
            escalated_strata=tuple(triggered),
            limitation_reasons=("MAXIMUM_VERIFICATION_ROUNDS_REACHED",),
        )
    selected_next: list[str] = []
    exhausted: list[tuple[str, ...]] = []
    for stratum in triggered:
        unchecked = [uid for uid in by_stratum[stratum] if uid not in results]
        chosen = unchecked[: policy.escalation_sample_per_stratum]
        selected_next.extend(chosen)
        if not chosen:
            exhausted.append(stratum)
    if exhausted:
        return VerificationRound(
            round_number=highest_round,
            status="explicit_limitation_required",
            selected_record_uids=(),
            escalated_strata=tuple(triggered),
            limitation_reasons=tuple(
                f"STRATUM_EXHAUSTED:{'|'.join(stratum)}" for stratum in exhausted
            ),
        )
    return VerificationRound(
        round_number=highest_round + 1,
        status="escalated_sample_required",
        selected_record_uids=tuple(selected_next),
        escalated_strata=tuple(triggered),
        limitation_reasons=(),
    )


__all__ = [
    "ChecksumRevisionApproval",
    "ChecksumRevisionComparison",
    "CsvStructure",
    "SourceScopeApproval",
    "VerificationResult",
    "VerificationRound",
    "VerificationSamplingPolicy",
    "SourceIntegrityReport",
    "SourceActivation",
    "build_source_scope_snapshot",
    "compare_checksum_revision",
    "inspect_csv_structure",
    "parse_checksum_revision_approval",
    "plan_literature_verification_round",
    "sha256_file",
    "stable_identifier",
    "stable_json_sha256",
    "validate_checksum_revision_approval",
    "validate_source_activation",
    "validate_source_scope_approval",
    "verify_source_integrity",
]
