from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import textwrap
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from n_response_curve.contracts import SUPPORTED_TABLE_FORMATS
from n_response_curve.data.provenance import sha256_file, stable_json_sha256


class ReportingError(RuntimeError):
    """Unsafe, incomplete, or non-reproducible output-package request."""


@dataclass(frozen=True)
class TableArtifact:
    """One normalized table with an optional required unique key."""

    rows: Sequence[Mapping[str, Any]]
    stable_key: str | None = None


@dataclass(frozen=True)
class ReleasePackage:
    """Promoted immutable output package and its audit entry points."""

    target_path: Path
    manifest_path: Path
    report_path: Path
    report_pdf_path: Path
    checksums_path: Path
    artifact_sha256: Mapping[str, str]
    preserved_prior_path: Path | None = None


_SAFE_TABLE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SAFE_RECORD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_ARTIFACT_SUFFIXES = frozenset({".htm", ".html", ".svg"})
_RESERVED_PACKAGE_PATHS = frozenset(
    {"CHECKSUMS.sha256", "report.md", "report.pdf", "run_manifest.json"}
)
_FULL_AUTHORITY_POLICY_VALUES = {
    "role_combination_policy": "accountable_parties_must_be_distinct",
    "substitution_policy": "no_substitution",
    "recusal_policy": "matrix_approval_preclears_assigned_parties",
    "dual_approval_policy": "single_accountable_party_approval",
}


def _governance_calendar_date(value: Any) -> date:
    normalized = str(value)
    if "T" in normalized:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00")).date()
    return date.fromisoformat(normalized)


@dataclass(frozen=True)
class _ValidatedReplacement:
    record: Mapping[str, Any]
    history_entry: Path
    prior_manifest_sha256: str


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _flat_value(value: Any) -> Any:
    if isinstance(value, (Mapping, tuple, list, set, frozenset)):
        return json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def _validate_table(name: str, artifact: TableArtifact) -> tuple[dict[str, Any], ...]:
    if not isinstance(name, str) or not _SAFE_TABLE_NAME.fullmatch(name):
        raise ReportingError(f"Table name is unsafe: {name!r}")
    if not isinstance(artifact, TableArtifact):
        raise ReportingError(f"Table {name!r} must be a TableArtifact")
    rows = tuple(_json_value(dict(row)) for row in artifact.rows)
    if any(not isinstance(row, Mapping) for row in rows):
        raise ReportingError(f"Table {name!r} contains a non-mapping row")
    if artifact.stable_key is not None:
        key = artifact.stable_key
        if not isinstance(key, str) or not key:
            raise ReportingError(f"Table {name!r} stable_key must be a nonempty string")
        values = [row.get(key) for row in rows]
        if any(value in {None, ""} for value in values):
            raise ReportingError(f"Table {name!r} has a missing stable key: {key}")
        normalized = [json.dumps(value, sort_keys=True, default=str) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ReportingError(f"Table {name!r} has a duplicate stable key: {key}")
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_table(stage_root: Path, name: str, artifact: TableArtifact, formats: Sequence[str]) -> tuple[dict[str, Any], dict[str, str]]:
    rows = _validate_table(name, artifact)
    tables_root = stage_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    normalized_rows = [{str(key): _flat_value(value) for key, value in row.items()} for row in rows]
    frame = pd.DataFrame(normalized_rows)
    artifacts: dict[str, str] = {}
    readback_row_counts: dict[str, int] = {}
    for output_format in formats:
        destination = tables_root / f"{name}.{output_format}"
        if output_format == "csv":
            frame.to_csv(destination, index=False)
        elif output_format == "parquet":
            frame.to_parquet(destination, index=False)
        elif output_format == "xlsx":
            frame.to_excel(destination, index=False)
        else:
            raise ReportingError(f"Unsupported output format: {output_format}")
        try:
            if output_format == "csv":
                if frame.empty and not len(frame.columns):
                    readback = pd.DataFrame()
                else:
                    readback = pd.read_csv(destination)
            elif output_format == "parquet":
                readback = pd.read_parquet(destination)
            else:
                readback = pd.read_excel(destination)
        except (ImportError, OSError, TypeError, ValueError, pd.errors.ParserError) as exc:
            raise ReportingError(f"Table {name!r} {output_format} artifact failed read-back validation") from exc
        if len(readback) != len(frame):
            raise ReportingError(
                f"Table {name!r} {output_format} read-back row count {len(readback)} does not match {len(frame)}"
            )
        if set(map(str, readback.columns)) != set(map(str, frame.columns)):
            raise ReportingError(f"Table {name!r} {output_format} read-back columns do not match")
        if artifact.stable_key is not None and artifact.stable_key in readback:
            stable_values = readback[artifact.stable_key]
            if bool(stable_values.isna().any()) or bool(stable_values.astype(str).duplicated().any()):
                raise ReportingError(
                    f"Table {name!r} {output_format} read-back stable key is missing or duplicated"
                )
        readback_row_counts[output_format] = len(readback)
        artifacts[destination.relative_to(stage_root).as_posix()] = sha256_file(destination)
    metadata = {
        "row_count": len(rows),
        "stable_key": artifact.stable_key,
        "column_names": [str(column) for column in frame.columns],
        "artifact_paths": sorted(artifacts),
        "readback_row_counts": dict(sorted(readback_row_counts.items())),
    }
    return metadata, artifacts


def _render_report(report_sections: Mapping[str, Iterable[str]]) -> str:
    headings = (
        ("primary", "Primary findings"),
        ("descriptive", "Descriptive findings"),
        ("sensitivity", "Sensitivity findings"),
        ("exploratory", "Exploratory findings"),
        ("predictive", "Predictive findings"),
        ("unsupported", "Unsupported findings"),
    )
    lines = ["# N-response run report", ""]
    for key, heading in headings:
        lines.extend((f"# {heading}", ""))
        messages = [str(message).strip() for message in report_sections.get(key, ()) if str(message).strip()]
        if messages:
            lines.extend(f"- {message}" for message in messages)
        else:
            lines.append("- No reportable findings in this category.")
        lines.append("")
    lines.extend(
        (
            "# Associational limitations",
            "",
            "- Results are associational and must not be interpreted as causal effects without approved design-specific assumptions.",
            "- Unsupported, sparse, aliased, or implementation-, support-, or artifact-gated analyses remain explicit non-findings rather than negative evidence.",
            "",
        )
    )
    return "\n".join(lines)


def _pdf_text_lines(markdown_text: str) -> tuple[str, ...]:
    lines: list[str] = []
    for raw_line in markdown_text.splitlines():
        normalized = (
            raw_line.replace("\t", "    ")
            .replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u00a0", " ")
        )
        encoded = normalized.encode("cp1252", errors="replace").decode("cp1252")
        wrapped = textwrap.wrap(
            encoded,
            width=92,
            replace_whitespace=False,
            drop_whitespace=True,
            break_long_words=True,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
    return tuple(lines or ("",))


def _pdf_literal(value: str) -> bytes:
    escaped = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return escaped.encode("cp1252", errors="replace")


def _render_deterministic_pdf(markdown_text: str) -> bytes:
    """Render a small deterministic PDF using only the fixed core Helvetica font."""

    text_lines = _pdf_text_lines(markdown_text)
    pages = tuple(
        text_lines[start : start + 52]
        for start in range(0, len(text_lines), 52)
    )
    page_object_numbers = tuple(4 + index * 2 for index in range(len(pages)))
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            b"<< /Type /Pages /Count "
            + str(len(pages)).encode("ascii")
            + b" /Kids ["
            + b" ".join(f"{number} 0 R".encode("ascii") for number in page_object_numbers)
            + b"] >>"
        ),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    }
    for index, lines in enumerate(pages):
        page_number = page_object_numbers[index]
        content_number = page_number + 1
        content_lines = [b"BT", b"/F1 10 Tf", b"50 750 Td", b"13 TL"]
        for line in lines:
            content_lines.append(b"(" + _pdf_literal(line) + b") Tj")
            content_lines.append(b"T*")
        content_lines.append(b"ET")
        content = b"\n".join(content_lines) + b"\n"
        objects[page_number] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents "
            + f"{content_number} 0 R".encode("ascii")
            + b" >>"
        )
        objects[content_number] = (
            b"<< /Length "
            + str(len(content)).encode("ascii")
            + b" >>\nstream\n"
            + content
            + b"endstream"
        )

    result = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {0: 0}
    for object_number in range(1, max(objects) + 1):
        offsets[object_number] = len(result)
        result.extend(f"{object_number} 0 obj\n".encode("ascii"))
        result.extend(objects[object_number])
        result.extend(b"\nendobj\n")
    xref_offset = len(result)
    result.extend(f"xref\n0 {max(objects) + 1}\n".encode("ascii"))
    result.extend(b"0000000000 65535 f \n")
    for object_number in range(1, max(objects) + 1):
        result.extend(f"{offsets[object_number]:010d} 00000 n \n".encode("ascii"))
    result.extend(
        (
            f"trailer\n<< /Size {max(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(result)


def _validate_pdf(path: Path) -> None:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReportingError("Rendered PDF report could not be read back") from exc
    required_tokens = (
        b"%PDF-1.4\n",
        b"/Type /Catalog",
        b"/Type /Pages",
        b"/Type /Page",
        b"/BaseFont /Helvetica",
        b"xref\n",
        b"startxref\n",
        b"%%EOF\n",
    )
    if not payload.startswith(required_tokens[0]) or any(
        token not in payload for token in required_tokens[1:]
    ):
        raise ReportingError("Rendered PDF report failed structural read-back validation")
    if b"/CreationDate" in payload or b"/ModDate" in payload:
        raise ReportingError("Rendered PDF report contains nondeterministic timestamp metadata")
    try:
        startxref = int(payload.rsplit(b"startxref\n", 1)[1].splitlines()[0])
    except (IndexError, ValueError) as exc:
        raise ReportingError("Rendered PDF report has an invalid cross-reference offset") from exc
    if startxref < 1 or payload[startxref : startxref + 5] != b"xref\n":
        raise ReportingError("Rendered PDF report cross-reference offset does not resolve")


def _assert_safe_target(target: Path, source_roots: Iterable[str | Path]) -> None:
    for source_root in source_roots:
        normalized_source = Path(source_root).resolve()
        if target == normalized_source or target.is_relative_to(normalized_source):
            raise ReportingError(f"Refusing source target for release package: {target}")


def _replacement_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReportingError("Approved replacement record has an invalid approval timestamp")
    normalized = value.strip()
    try:
        if "T" in normalized:
            parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("timezone is required")
        else:
            date.fromisoformat(normalized)
    except ValueError as exc:
        raise ReportingError(
            "Approved replacement timestamp must be an ISO date or timezone-qualified datetime"
        ) from exc
    return normalized


def _replacement_text(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ReportingError(f"Approved replacement record has an invalid {field}")
    return value.strip()


def _target_path_sha256(target: Path) -> str:
    return hashlib.sha256(str(target).encode("utf-8")).hexdigest()


def _validate_replacement_record(
    target: Path,
    *,
    manifest: Mapping[str, Any],
    replacement_record: Mapping[str, Any] | None,
) -> _ValidatedReplacement:
    try:
        prior = verify_release_package(target)
    except ReportingError as exc:
        raise ReportingError(
            "Existing release target is not a verified package and cannot be replaced"
        ) from exc
    if not isinstance(replacement_record, Mapping):
        raise ReportingError(
            "Replacing an existing release requires an approved named-target replacement record"
        )
    record = dict(_json_value(replacement_record))
    expected_fields = {
        "record_id",
        "status",
        "target_name",
        "target_path_sha256",
        "replacement_run_id",
        "prior_manifest_sha256",
        "approved_by",
        "approved_at",
        "approval_source",
        "reason",
    }
    if set(record) != expected_fields:
        raise ReportingError("Approved replacement record fields do not match the required schema")
    record_id = _replacement_text(record["record_id"], field="record identifier")
    if _SAFE_RECORD_ID.fullmatch(record_id) is None:
        raise ReportingError("Approved replacement record identifier is unsafe")
    if record["status"] != "APPROVED":
        raise ReportingError("Replacement record is not approved")
    if record["target_name"] != target.name:
        raise ReportingError("Approved replacement record names a different target")
    if record["target_path_sha256"] != _target_path_sha256(target):
        raise ReportingError("Approved replacement record is not bound to this target path")
    replacement_run_id = manifest.get("run_id")
    if not isinstance(replacement_run_id, str) or record["replacement_run_id"] != replacement_run_id:
        raise ReportingError("Approved replacement record names a different replacement run")
    prior_manifest_sha256 = sha256_file(prior.manifest_path)
    if (
        not isinstance(record["prior_manifest_sha256"], str)
        or _SHA256.fullmatch(record["prior_manifest_sha256"]) is None
        or record["prior_manifest_sha256"] != prior_manifest_sha256
    ):
        raise ReportingError("Approved replacement record does not match the prior manifest")
    normalized_record = {
        **record,
        "record_id": record_id,
        "approved_by": _replacement_text(record["approved_by"], field="approver"),
        "approved_at": _replacement_timestamp(record["approved_at"]),
        "approval_source": _replacement_text(
            record["approval_source"],
            field="approval source",
        ),
        "reason": _replacement_text(record["reason"], field="replacement reason"),
    }
    history_entry = (
        target.parent
        / ".release_history"
        / target.name
        / f"{prior_manifest_sha256[:16]}-{record_id}"
    )
    if history_entry.exists():
        raise ReportingError("Approved replacement record has already been used")
    return _ValidatedReplacement(
        record=normalized_record,
        history_entry=history_entry,
        prior_manifest_sha256=prior_manifest_sha256,
    )


def _promote_stage(
    stage: Path,
    target: Path,
    *,
    overwrite: bool,
    replacement: _ValidatedReplacement | None,
) -> Path | None:
    if target.exists() and not overwrite:
        raise ReportingError(f"Release package collision at {target}; set overwrite explicitly to replace it")
    if target.exists() and replacement is None:
        raise ReportingError(
            "Replacing an existing release requires an approved named-target replacement record"
        )
    archived_package: Path | None = None
    history_entry: Path | None = None
    try:
        if target.exists():
            assert replacement is not None
            history_entry = replacement.history_entry
            history_entry.mkdir(parents=True, exist_ok=False)
            replacement_path = history_entry / "replacement_record.json"
            _write_json(replacement_path, replacement.record)
            _write_promotion_state(history_entry, target, "prepared")
            archived_package = history_entry / "package"
            os.replace(target, archived_package)
            _write_promotion_state(history_entry, target, "prior_archived")
        os.replace(stage, target)
        if history_entry is not None:
            _write_promotion_state(history_entry, target, "complete")
    except BaseException as exc:
        try:
            if archived_package is not None and archived_package.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
                os.replace(archived_package, target)
            if history_entry is not None and history_entry.exists():
                shutil.rmtree(history_entry)
        except OSError as restore_error:
            raise ReportingError(
                f"Unable to restore the previous release package after promotion failure: {restore_error}"
            ) from exc
        if isinstance(exc, OSError):
            raise ReportingError(f"Unable to atomically promote release package: {exc}") from exc
        raise
    return archived_package


def _write_promotion_state(history_entry: Path, target: Path, state: str) -> None:
    """Durably journal replacement promotion so a hard-exit can be recovered."""

    payload = {
        "schema_version": "release-promotion-v1",
        "target_name": target.name,
        "target_path_sha256": _target_path_sha256(target),
        "state": state,
    }
    state_path = history_entry / "promotion_state.json"
    temporary = history_entry / ".promotion_state.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, state_path)
    try:
        directory_fd = os.open(history_entry, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # Some mounted filesystems do not support directory fsync; the file itself
        # is still flushed and atomically replaced.
        pass


def _recover_interrupted_promotion(target: Path) -> None:
    """Restore or finish the sole journaled replacement interrupted by hard exit."""

    history_root = target.parent / ".release_history" / target.name
    if not history_root.is_dir():
        return
    pending: list[tuple[Path, str]] = []
    unjournaled: list[Path] = []
    for history_entry in sorted(path for path in history_root.iterdir() if path.is_dir()):
        state_path = history_entry / "promotion_state.json"
        if not state_path.is_file():
            unjournaled.append(history_entry)
            continue
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReportingError("Release-promotion recovery journal is unreadable") from exc
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != "release-promotion-v1"
            or payload.get("target_name") != target.name
            or payload.get("target_path_sha256") != _target_path_sha256(target)
            or payload.get("state") not in {"prepared", "prior_archived", "complete"}
        ):
            raise ReportingError("Release-promotion recovery journal is malformed or misbound")
        state = str(payload["state"])
        if state in {"prepared", "prior_archived"}:
            pending.append((history_entry, state))
    if unjournaled:
        if pending or len(unjournaled) != 1:
            raise ReportingError(
                "Multiple incomplete release-promotion histories require manual review"
            )
        history_entry = unjournaled[0]
        entry_names = {path.name for path in history_entry.iterdir()}
        if (
            not target.exists()
            or (history_entry / "package").exists()
            or entry_names - {"replacement_record.json", ".promotion_state.tmp"}
        ):
            raise ReportingError(
                "Unjournaled release-promotion history cannot be recovered automatically"
            )
        verify_release_package(target)
        shutil.rmtree(history_entry)
        return
    if not pending:
        return
    if len(pending) != 1:
        raise ReportingError("Multiple interrupted release promotions require manual review")
    history_entry, state = pending[0]
    archived_package = history_entry / "package"
    if target.exists() and archived_package.exists():
        try:
            verify_release_package(target)
        except ReportingError:
            failed_target = history_entry / "failed_replacement_package"
            os.replace(target, failed_target)
            verify_release_package(archived_package)
            os.replace(archived_package, target)
            shutil.rmtree(history_entry)
        else:
            _write_promotion_state(history_entry, target, "complete")
        return
    if not target.exists() and archived_package.exists():
        verify_release_package(archived_package)
        os.replace(archived_package, target)
        shutil.rmtree(history_entry)
        return
    if target.exists() and not archived_package.exists() and state == "prepared":
        shutil.rmtree(history_entry)
        return
    raise ReportingError("Interrupted release promotion cannot be recovered automatically")


def _stage_directory_pattern(target: Path) -> re.Pattern[str]:
    """Match only staging directories this module creates for ``target``.

    Anchoring on ``target.name`` is what makes reaping safe: the pattern cannot
    match ``.release_history`` (the promotion journal and preserved priors), the
    promoted package, or another release target's residue in the same root.
    """

    return re.compile(rf"^\.{re.escape(target.name)}\.stage-[A-Za-z0-9_]{{8}}$")


def reap_abandoned_stage_directories(target: Path) -> tuple[str, ...]:
    """Remove staging directories abandoned by a hard exit next to ``target``.

    ``write_release_package`` removes its own staging directory on every Python
    level failure, so a surviving one means the process died without unwinding
    (SIGKILL, OOM, power loss) or that ``shutil.rmtree`` was denied. Nothing
    ever reads these directories again -- ``_recover_interrupted_promotion``
    looks only under ``.release_history`` -- so they accumulate as pure residue
    in the release root.

    A directory is only removed when ``verify_release_package`` rejects it.
    ``CHECKSUMS.sha256`` is written last, immediately before verification and
    promotion, so an in-flight staging directory fails that check for
    essentially its whole life; a complete one is a promotable package and is
    deliberately left alone for manual review rather than deleted.
    """

    release_root = target.parent
    if not release_root.is_dir():
        return ()
    pattern = _stage_directory_pattern(target)
    reaped: list[str] = []
    for candidate in sorted(release_root.iterdir()):
        if candidate.is_symlink() or not candidate.is_dir():
            continue
        if not pattern.fullmatch(candidate.name):
            continue
        try:
            verify_release_package(candidate)
        except ReportingError:
            pass
        except OSError:
            # Unreadable for reasons unrelated to completeness; leave it for a
            # human rather than guess.
            continue
        else:
            continue
        try:
            shutil.rmtree(candidate)
        except OSError:
            continue
        reaped.append(candidate.name)
    return tuple(reaped)


def _prune_empty_stage_directories(stage: Path) -> None:
    """Drop directories left empty inside the staging package.

    The staged inventory counts files only, so an empty directory is invisible
    to it and to ``CHECKSUMS.sha256`` yet is still carried into the promoted
    package by the atomic rename. ``r_stages/<candidate_id>/`` produces these
    whenever a candidate's contract and input are unlinked and no ``result.json``
    is written.
    """

    for directory in sorted(
        (path for path in stage.rglob("*") if path.is_dir() and not path.is_symlink()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            next(directory.iterdir())
        except StopIteration:
            directory.rmdir()
        except OSError:
            continue


def _archived_governance_json(
    target: Path,
    checksums: Mapping[str, str],
    *,
    relative_path: Any,
    expected_sha256: Any,
    label: str,
) -> Mapping[str, Any]:
    """Read one checksummed governance artifact named by an authoritative manifest."""

    if not isinstance(relative_path, str) or not relative_path:
        raise ReportingError(f"Full-release governance is missing archived {label}")
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative_path:
        raise ReportingError(f"Full-release governance has an unsafe archived {label} path")
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise ReportingError(f"Full-release governance has an invalid {label} hash")
    if checksums.get(relative_path) != expected_sha256:
        raise ReportingError(
            f"Full-release governance {label} is not bound to the checksum inventory"
        )
    artifact_path = target / path
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportingError(
            f"Full-release governance archived {label} is unreadable or malformed"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ReportingError(f"Full-release governance archived {label} must be a JSON object")
    return payload


def _verify_archived_policy_bundle(
    target: Path,
    checksums: Mapping[str, str],
    section: Any,
    *,
    label: str,
    default_accountable_party: object,
    accountable_party_by_artifact: Mapping[str, object] | None = None,
) -> None:
    if not isinstance(section, Mapping) or section.get("status") != "validated":
        raise ReportingError(f"Full-release governance requires validated {label}")
    manifest_sha256 = section.get("manifest_sha256")
    component_hashes = section.get("artifact_sha256")
    archived_paths = section.get("archived_artifact_paths")
    if (
        not isinstance(manifest_sha256, str)
        or not _SHA256.fullmatch(manifest_sha256)
        or not isinstance(component_hashes, Mapping)
        or not isinstance(archived_paths, Mapping)
        or not archived_paths
    ):
        raise ReportingError(f"Full-release governance {label} inventory is malformed")
    unmatched_components = set(component_hashes)
    manifest_matches = 0
    for artifact_type, relative_path in archived_paths.items():
        if not isinstance(artifact_type, str):
            raise ReportingError(f"Full-release governance {label} inventory is malformed")
        expected_sha256 = component_hashes.get(artifact_type)
        if (
            expected_sha256 is None
            and manifest_matches == 0
            and isinstance(relative_path, str)
            and checksums.get(relative_path) == manifest_sha256
        ):
            expected_sha256 = manifest_sha256
            manifest_matches += 1
        elif expected_sha256 is not None:
            unmatched_components.discard(artifact_type)
        payload = _archived_governance_json(
            target,
            checksums,
            relative_path=relative_path,
            expected_sha256=expected_sha256,
            label=f"{label} artifact {artifact_type}",
        )
        expected_party = (
            accountable_party_by_artifact or {}
        ).get(artifact_type, default_accountable_party)
        if (
            not isinstance(expected_party, str)
            or not expected_party
            or payload.get("approved_by") != expected_party
        ):
            raise ReportingError(
                f"Full-release governance {label} artifact {artifact_type} "
                "signer does not match its accountable authority"
            )
    if unmatched_components or manifest_matches != 1:
        raise ReportingError(
            f"Full-release governance {label} archive is incomplete or misbound"
        )


def _verify_full_release_governance(
    target: Path,
    checksums: Mapping[str, str],
    manifest: Mapping[str, Any],
) -> None:
    if manifest.get("status") != "phase_5_full_release_complete":
        return
    runtime_policy = manifest.get("runtime_policy")
    run_identity = manifest.get("run_identity")
    run_identity_sha256 = manifest.get("run_identity_sha256")
    if (
        not isinstance(runtime_policy, Mapping)
        or runtime_policy.get("mode") != "full"
        or runtime_policy.get("status") != "approved"
        or not isinstance(run_identity, Mapping)
        or run_identity.get("mode") != "full"
        or not isinstance(run_identity_sha256, str)
        or run_identity_sha256 != stable_json_sha256(run_identity)
    ):
        raise ReportingError(
            "Full-release governance runtime policy or run identity is missing or invalid"
        )

    policy_content_sha256 = runtime_policy.get("policy_content_sha256")
    effective_enablement = runtime_policy.get("effective_enablement")
    effective_enablement_sha256 = runtime_policy.get("effective_enablement_sha256")
    if (
        not isinstance(policy_content_sha256, str)
        or not _SHA256.fullmatch(policy_content_sha256)
        or effective_enablement_sha256 != stable_json_sha256(effective_enablement)
        or run_identity.get("policy_content_sha256") != policy_content_sha256
        or run_identity.get("effective_enablement_sha256")
        != effective_enablement_sha256
    ):
        raise ReportingError("Full-release governance runtime-policy identity is misbound")

    authority_matrix = runtime_policy.get("approval_authority_matrix")
    if not isinstance(authority_matrix, Mapping):
        raise ReportingError("Full-release governance authority matrix is missing")
    matrix_sha256 = authority_matrix.get("artifact_sha256")
    matrix_source = _archived_governance_json(
        target,
        checksums,
        relative_path=authority_matrix.get("archived_artifact_path"),
        expected_sha256=matrix_sha256,
        label="approval authority matrix",
    )
    matrix_approval = authority_matrix.get("approval")
    try:
        matrix_effective_date = _governance_calendar_date(
            matrix_source.get("effective_from")
        )
        matrix_approval_date = _governance_calendar_date(
            matrix_source.get("approved_at")
        )
    except ValueError as exc:
        raise ReportingError(
            "Full-release governance authority matrix dates are invalid"
        ) from exc
    if (
        matrix_source.get("status") != "APPROVED"
        or matrix_source.get("matrix_id") != authority_matrix.get("matrix_id")
        or matrix_source.get("effective_from") != authority_matrix.get("effective_from")
        or matrix_source.get("gate_authorities") != authority_matrix.get("gate_authorities")
        or not isinstance(matrix_approval, Mapping)
        or matrix_source.get("approved_by") != matrix_approval.get("approved_by")
        or matrix_source.get("approved_at") != matrix_approval.get("approved_at")
        or any(
            matrix_source.get(field) != value
            or authority_matrix.get(field) != value
            for field, value in _FULL_AUTHORITY_POLICY_VALUES.items()
        )
        or matrix_approval_date > matrix_effective_date
        or matrix_effective_date > date.today()
        or run_identity.get("approval_authority_matrix_sha256") != matrix_sha256
    ):
        raise ReportingError("Full-release governance authority matrix is misbound")
    gate_authorities = authority_matrix.get("gate_authorities")
    if not isinstance(gate_authorities, Mapping):
        raise ReportingError("Full-release governance authority assignments are malformed")
    accountable_parties = [
        assignment.get("accountable_party")
        for assignment in gate_authorities.values()
        if isinstance(assignment, Mapping)
    ]
    if (
        len(accountable_parties) != 5
        or any(not isinstance(party, str) or not party for party in accountable_parties)
        or len(accountable_parties) != len(set(accountable_parties))
    ):
        raise ReportingError(
            "Full-release governance authority assignments do not enforce role separation"
        )

    snapshot_source = _archived_governance_json(
        target,
        checksums,
        relative_path=runtime_policy.get("archived_artifact_path"),
        expected_sha256=runtime_policy.get("approval_artifact_sha256"),
        label="approved runtime-policy snapshot",
    )
    snapshot_content = snapshot_source.get("policy_content")
    runtime_approval = runtime_policy.get("approval")
    runtime_assignment = gate_authorities.get("runtime_integrity")
    runtime_party = (
        runtime_assignment.get("accountable_party")
        if isinstance(runtime_assignment, Mapping)
        else None
    )
    if (
        snapshot_source.get("status") != "APPROVED"
        or snapshot_source.get("policy_content_sha256") != policy_content_sha256
        or stable_json_sha256(snapshot_content) != policy_content_sha256
        or not isinstance(snapshot_content, Mapping)
        or snapshot_content.get("effective_enablement") != effective_enablement
        or not isinstance(runtime_approval, Mapping)
        or runtime_approval.get("approved_by") != runtime_party
        or snapshot_source.get("approved_by") != runtime_party
        or run_identity.get("approval_artifact_sha256")
        != runtime_policy.get("approval_artifact_sha256")
    ):
        raise ReportingError("Full-release governance runtime-policy snapshot is misbound")

    review_policy = runtime_policy.get("review_gate_policy")
    if review_policy is not None:
        if not isinstance(review_policy, Mapping):
            raise ReportingError("Full-release governance review policy is malformed")
        review_source = _archived_governance_json(
            target,
            checksums,
            relative_path=review_policy.get("archived_artifact_path"),
            expected_sha256=review_policy.get("artifact_sha256"),
            label="review-gate policy",
        )
        scientific_assignment = gate_authorities.get("scientific_methods")
        scientific_party = (
            scientific_assignment.get("accountable_party")
            if isinstance(scientific_assignment, Mapping)
            else None
        )
        if (
            review_source.get("status") != "APPROVED"
            or review_source.get("policy_id") != review_policy.get("policy_id")
            or review_source.get("approved_by") != scientific_party
        ):
            raise ReportingError("Full-release governance review policy is misbound")

    release_approval = manifest.get("release_approval")
    if not isinstance(release_approval, Mapping):
        raise ReportingError("Full-release governance release approval is missing")
    release_source = _archived_governance_json(
        target,
        checksums,
        relative_path=release_approval.get("archived_artifact_path"),
        expected_sha256=release_approval.get("artifact_sha256"),
        label="release approval",
    )
    release_assignment = gate_authorities.get("release_promotion")
    release_party = (
        release_assignment.get("accountable_party")
        if isinstance(release_assignment, Mapping)
        else None
    )
    if (
        release_approval.get("status") != "approved"
        or release_approval.get("run_id") != manifest.get("run_id")
        or release_approval.get("run_identity_sha256") != run_identity_sha256
        or release_approval.get("authority_matrix_sha256") != matrix_sha256
        or not isinstance(release_approval.get("approval"), Mapping)
        or release_approval["approval"].get("approved_by") != release_party
        or release_source.get("status") != "APPROVED"
        or release_source.get("run_id") != manifest.get("run_id")
        or release_source.get("run_identity_sha256") != run_identity_sha256
        or release_source.get("approved_by") != release_party
    ):
        raise ReportingError("Full-release governance release approval is misbound")

    source_assignment = gate_authorities.get("source_integrity")
    restricted_assignment = gate_authorities.get("restricted_data")
    scientific_assignment = gate_authorities.get("scientific_methods")
    source_party = (
        source_assignment.get("accountable_party")
        if isinstance(source_assignment, Mapping)
        else None
    )
    restricted_party = (
        restricted_assignment.get("accountable_party")
        if isinstance(restricted_assignment, Mapping)
        else None
    )
    scientific_party = (
        scientific_assignment.get("accountable_party")
        if isinstance(scientific_assignment, Mapping)
        else None
    )
    _verify_archived_policy_bundle(
        target,
        checksums,
        manifest.get("source_data_policy"),
        label="source-data policy",
        default_accountable_party=source_party,
        accountable_party_by_artifact={"restricted_policy": restricted_party},
    )
    _verify_archived_policy_bundle(
        target,
        checksums,
        manifest.get("analysis_policy"),
        label="analysis policy",
        default_accountable_party=scientific_party,
    )


def verify_release_package(target_path: str | Path) -> ReleasePackage:
    """Verify every promoted artifact against its immutable checksum ledger."""

    target = Path(target_path).resolve()
    manifest_path = target / "run_manifest.json"
    report_path = target / "report.md"
    report_pdf_path = target / "report.pdf"
    checksums_path = target / "CHECKSUMS.sha256"
    if (
        not target.is_dir()
        or not manifest_path.is_file()
        or not report_path.is_file()
        or not report_pdf_path.is_file()
        or not checksums_path.is_file()
    ):
        raise ReportingError(f"Release package is incomplete: {target}")
    expected: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ReportingError("Release checksum ledger has an invalid line") from exc
        relative_path = Path(relative)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in expected
        ):
            raise ReportingError("Release checksum ledger is unsafe or malformed")
        expected[relative] = digest
    actual_paths = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file() and path != checksums_path
    }
    if any(Path(relative).suffix.casefold() in _FORBIDDEN_ARTIFACT_SUFFIXES for relative in actual_paths):
        raise ReportingError("Release package contains a prohibited document or figure artifact")
    if actual_paths != set(expected):
        raise ReportingError("Release checksum ledger does not cover the complete package")
    for relative, digest in expected.items():
        artifact_path = target / relative
        if not artifact_path.is_file() or sha256_file(artifact_path) != digest:
            raise ReportingError(f"Release checksum mismatch: {relative}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReportingError("Release run manifest is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ReportingError("Release run manifest must be a JSON object")
    output_profile = payload.get("output_profile")
    if (
        not isinstance(output_profile, Mapping)
        or tuple(output_profile.get("document_formats", ())) != ("md", "pdf")
    ):
        raise ReportingError("Release run manifest has an invalid report-document profile")
    documents = payload.get("documents")
    if not isinstance(documents, Mapping) or set(documents) != {"markdown", "pdf"}:
        raise ReportingError("Release run manifest has an invalid report-document inventory")
    expected_documents = {
        "markdown": ("report.md", sha256_file(report_path)),
        "pdf": ("report.pdf", sha256_file(report_pdf_path)),
    }
    for name, (relative_path, digest) in expected_documents.items():
        item = documents[name]
        if (
            not isinstance(item, Mapping)
            or item.get("path") != relative_path
            or item.get("sha256") != digest
            or expected.get(relative_path) != digest
        ):
            raise ReportingError("Release run manifest report-document metadata is invalid")
    _verify_full_release_governance(target, expected, payload)
    try:
        report_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReportingError("Markdown report failed UTF-8 read-back validation") from exc
    _validate_pdf(report_pdf_path)
    return ReleasePackage(
        target_path=target,
        manifest_path=manifest_path,
        report_path=report_path,
        report_pdf_path=report_pdf_path,
        checksums_path=checksums_path,
        artifact_sha256=dict(sorted(expected.items())),
    )


_ANALYSIS_POPULATION_DISPOSITIONS = frozenset(
    {
        "retained_descriptive_only",
        "retained_without_credible_model",
        "removed_from_analysis_population",
    }
)


def _validate_analysis_population_selection(manifest: Mapping[str, Any]) -> None:
    """Refuse to promote a successful run that does not record its own selection.

    Under literal `OPS-03` any warning, unresolved field, or excluded series is
    fatal, so a run that reached this point did so with an analysis population
    that is by construction the subset producing none of those states. An
    unrecorded selection of that kind is as release-blocking as an unresolved
    disposition (Plan Section 12, release gate; Phase 5 Task 13 step 7).
    """

    status = manifest.get("status")
    if not isinstance(status, str) or not status.endswith("_release_complete"):
        return
    selection = manifest.get("analysis_population_selection")
    if not isinstance(selection, Mapping):
        raise ReportingError(
            "Successful run manifest is missing its analysis-population selection ledger"
        )
    ledger = selection.get("ledger")
    comparison = selection.get("composition_comparison")
    if not isinstance(ledger, Sequence) or isinstance(ledger, (str, bytes)):
        raise ReportingError(
            "Analysis-population selection ledger must be a list of series records"
        )
    if not isinstance(comparison, Sequence) or isinstance(comparison, (str, bytes)):
        raise ReportingError(
            "Analysis-population selection is missing its included-versus-excluded "
            "composition comparison"
        )
    for entry in ledger:
        if not isinstance(entry, Mapping):
            raise ReportingError(
                "Every analysis-population selection entry must be a record"
            )
        series_uid = entry.get("response_series_uid")
        disposition = entry.get("disposition")
        reason_codes = entry.get("reason_codes")
        if not isinstance(series_uid, str) or not series_uid.strip():
            raise ReportingError(
                "Analysis-population selection entry is missing its response series"
            )
        if disposition not in _ANALYSIS_POPULATION_DISPOSITIONS:
            raise ReportingError(
                "Analysis-population selection entry has no recorded disposition: "
                f"{series_uid}"
            )
        if isinstance(reason_codes, (str, bytes)) or not isinstance(
            reason_codes,
            Sequence,
        ):
            raise ReportingError(
                "Analysis-population selection entry has no recorded reason: "
                f"{series_uid}"
            )
    recorded_count = selection.get("resolved_or_removed_series_count")
    if not isinstance(recorded_count, int) or isinstance(recorded_count, bool):
        raise ReportingError(
            "Analysis-population selection must count the series it resolved or removed"
        )
    if recorded_count != len(ledger):
        raise ReportingError(
            "Analysis-population selection count does not reconcile with its ledger"
        )
    if ledger and not comparison:
        raise ReportingError(
            "Analysis-population selection excluded series without comparing "
            "included and excluded composition"
        )


def write_release_package(
    target_path: str | Path,
    *,
    tables: Mapping[str, TableArtifact],
    manifest: Mapping[str, Any],
    report_sections: Mapping[str, Iterable[str]],
    output_formats: Sequence[str],
    overwrite: bool,
    source_roots: Iterable[str | Path],
    stage_writers: Sequence[Callable[[Path], Iterable[str | Path]]] = (),
    replacement_record: Mapping[str, Any] | None = None,
    release_validator: Callable[[Mapping[str, Any]], None] | None = None,
) -> ReleasePackage:
    """Stage, validate, checksum, and atomically promote one auditable run package."""

    if not isinstance(manifest, Mapping):
        raise ReportingError("Run manifest input must be a mapping")
    target = Path(target_path).resolve()
    _assert_safe_target(target, source_roots)
    target.parent.mkdir(parents=True, exist_ok=True)
    _recover_interrupted_promotion(target)
    initial_manifest = dict(_json_value(manifest))
    initial_profile = initial_manifest.get("output_profile")
    if isinstance(initial_profile, Mapping) and (
        "document_formats" in initial_profile
        and tuple(initial_profile["document_formats"]) != ("md", "pdf")
    ):
        raise ReportingError("Run manifest advertises a prohibited report-document profile")
    _validate_analysis_population_selection(initial_manifest)
    formats = tuple(str(item).lower().lstrip(".") for item in output_formats)
    if not formats or set(formats) - SUPPORTED_TABLE_FORMATS or len(formats) != len(set(formats)):
        raise ReportingError("Output formats must be unique members of csv, parquet, xlsx")
    if target.exists() and not overwrite:
        raise ReportingError(f"Release package collision at {target}; set overwrite explicitly to replace it")
    replacement = (
        _validate_replacement_record(
            target,
            manifest=initial_manifest,
            replacement_record=replacement_record,
        )
        if target.exists()
        else None
    )
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    try:
        table_metadata: dict[str, dict[str, Any]] = {}
        artifact_sha256: dict[str, str] = {}
        for name, artifact in sorted(tables.items()):
            metadata, artifact_hashes = _write_table(stage, name, artifact, formats)
            table_metadata[name] = metadata
            artifact_sha256.update(artifact_hashes)
        for writer in stage_writers:
            if not callable(writer):
                raise ReportingError("Each stage writer must be callable")
            written_paths = writer(stage)
            if written_paths is None:
                raise ReportingError("Stage writer must return its created artifact paths")
            for written_path in written_paths:
                artifact_path = Path(written_path).resolve()
                if not artifact_path.is_relative_to(stage) or not artifact_path.is_file():
                    raise ReportingError("Stage writer returned an artifact outside the staging package or not a file")
                relative_path = artifact_path.relative_to(stage).as_posix()
                if (
                    relative_path in _RESERVED_PACKAGE_PATHS
                    or artifact_path.suffix.casefold() in _FORBIDDEN_ARTIFACT_SUFFIXES
                ):
                    raise ReportingError(
                        "Stage writer returned a reserved or prohibited package artifact"
                    )
                if relative_path in artifact_sha256:
                    raise ReportingError(f"Stage writer artifact collides with an existing package artifact: {relative_path}")
                artifact_sha256[relative_path] = sha256_file(artifact_path)
        if replacement is not None:
            replacement_path = stage / "replacement_record.json"
            _write_json(replacement_path, replacement.record)
            artifact_sha256["replacement_record.json"] = sha256_file(
                replacement_path
            )
        _prune_empty_stage_directories(stage)
        staged_paths = {
            path.relative_to(stage).as_posix()
            for path in stage.rglob("*")
            if path.is_file()
        }
        if any(
            Path(relative).suffix.casefold() in _FORBIDDEN_ARTIFACT_SUFFIXES
            for relative in staged_paths
        ):
            raise ReportingError("Staging package contains a prohibited document or figure artifact")
        if staged_paths != set(artifact_sha256):
            raise ReportingError("Stage artifact inventory does not account for every staged file")
        for relative_path, digest in artifact_sha256.items():
            if sha256_file(stage / relative_path) != digest:
                raise ReportingError(f"Stage artifact changed after registration: {relative_path}")
        if release_validator is not None:
            if not callable(release_validator):
                raise ReportingError("Release validator must be callable")
            release_validator(manifest)
        report_path = stage / "report.md"
        report_text = _render_report(report_sections)
        report_path.write_text(report_text, encoding="utf-8")
        artifact_sha256[report_path.relative_to(stage).as_posix()] = sha256_file(report_path)
        report_pdf_path = stage / "report.pdf"
        report_pdf_path.write_bytes(_render_deterministic_pdf(report_text))
        _validate_pdf(report_pdf_path)
        artifact_sha256[report_pdf_path.relative_to(stage).as_posix()] = sha256_file(
            report_pdf_path
        )
        manifest_path = stage / "run_manifest.json"
        manifest_payload = dict(_json_value(manifest))
        output_profile = manifest_payload.get("output_profile", {})
        if not isinstance(output_profile, Mapping):
            raise ReportingError("Run manifest output profile must be a mapping")
        output_profile = dict(output_profile)
        if (
            "document_formats" in output_profile
            and tuple(output_profile["document_formats"]) != ("md", "pdf")
        ):
            raise ReportingError("Run manifest advertises a prohibited report-document profile")
        output_profile["document_formats"] = ["md", "pdf"]
        manifest_payload["output_profile"] = output_profile
        manifest_payload["documents"] = {
            "markdown": {
                "path": "report.md",
                "sha256": artifact_sha256["report.md"],
            },
            "pdf": {
                "path": "report.pdf",
                "sha256": artifact_sha256["report.pdf"],
            },
        }
        if replacement is not None:
            manifest_payload["replacement"] = {
                "record_path": "replacement_record.json",
                "record_sha256": artifact_sha256["replacement_record.json"],
                "prior_manifest_sha256": replacement.prior_manifest_sha256,
                "preserved_prior_path": (
                    replacement.history_entry.relative_to(target.parent)
                    / "package"
                ).as_posix(),
            }
        manifest_payload["tables"] = table_metadata
        manifest_payload["artifact_sha256_before_manifest"] = dict(sorted(artifact_sha256.items()))
        _write_json(manifest_path, manifest_payload)
        artifact_sha256[manifest_path.relative_to(stage).as_posix()] = sha256_file(manifest_path)
        checksums_path = stage / "CHECKSUMS.sha256"
        checksums_path.write_text(
            "".join(f"{digest}  {relative}\n" for relative, digest in sorted(artifact_sha256.items())),
            encoding="utf-8",
        )
        verify_release_package(stage)
        preserved_prior = _promote_stage(
            stage,
            target,
            overwrite=overwrite,
            replacement=replacement,
        )
        try:
            package = verify_release_package(target)
        except BaseException:
            if preserved_prior is not None and preserved_prior.exists():
                failed_replacement = preserved_prior.parent / "failed_replacement_package"
                if target.exists():
                    os.replace(target, failed_replacement)
                os.replace(preserved_prior, target)
            raise
        return replace(package, preserved_prior_path=preserved_prior)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = [
    "ReleasePackage",
    "ReportingError",
    "TableArtifact",
    "reap_abandoned_stage_directories",
    "verify_release_package",
    "write_release_package",
]
