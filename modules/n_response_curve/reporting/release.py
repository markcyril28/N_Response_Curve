from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
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
from PIL import Image

from n_response_curve.contracts import SUPPORTED_FIGURE_FORMATS, SUPPORTED_TABLE_FORMATS
from n_response_curve.data.provenance import sha256_file, stable_json_sha256


class ReportingError(RuntimeError):
    """Unsafe, incomplete, or non-reproducible output-package request."""


@dataclass(frozen=True)
class TableArtifact:
    """One normalized table with an optional required unique key.

    ``group`` selects the ``tables/<group>/`` subdirectory the artifact is
    written into. It stays optional so ad-hoc callers keep the historical flat
    ``tables/`` layout; the pipeline itself declares a group for every released
    table and refuses to release an ungrouped one (see
    ``pipeline/workflow.py:_RELEASE_TABLE_GROUPS``).
    """

    rows: Sequence[Mapping[str, Any]]
    stable_key: str | None = None
    group: str | None = None


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
_SAFE_TABLE_GROUP = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_ARTIFACT_SUFFIXES = frozenset({".htm", ".html", ".svg"})
_RESERVED_PACKAGE_PATHS = frozenset(
    {"CHECKSUMS.sha256", "report.md", "report.pdf", "run_manifest.json"}
)
_REVIEW_ISSUE_LEDGER_PATH = "ledgers/review_issue_ledger.json"
_REVIEW_GATE_SCHEMA_VERSIONS = frozenset(
    {"ops-03-review-gate-v1", "ops-09-review-gate-v2"}
)
_REVIEW_GATE_EXPECTED_LEDGERS = (
    "phase_2_review",
    "phase_3_series_evidence",
    "model_attempts",
    "runtime_warnings",
    "analysis_terminal_statuses",
    "multiplicity_reconciliation",
    "claim_classification",
)
_REVIEW_ISSUE_UID = re.compile(r"^review_issue_[0-9a-f]{24}$")
_REVIEW_ISSUE_STATES = frozenset(
    {"structural", "unresolved", "warning", "excluded_series"}
)
_REVIEW_ISSUE_STAGES = frozenset({"phase_2", "phase_3", "phase_4", "runtime"})
_FIGURE_LAYOUT_VERSION = "figures-by-source-and-model-v1"
_FIGURE_LAYOUT_VERSION_V2 = "figures-by-source-model-and-overlay-v2"
_KNOWN_FIGURE_LAYOUT_VERSIONS = frozenset(
    {_FIGURE_LAYOUT_VERSION, _FIGURE_LAYOUT_VERSION_V2}
)
_FIGURE_OVERLAY_PATH_TEMPLATE = "figures/overlay/<source>.<format>"
_FIGURE_OVERLAY_SCOPE = "same_governed_series_as_observed_figures"
_SAFE_FIGURE_DIRECTORY_TOKEN = re.compile(r"^[A-Za-z0-9_-]+$")
_WINDOWS_RESERVED_BASENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)
_PIPELINE_RELEASE_STATUSES = frozenset(
    {"phase_5_test_release_complete", "phase_5_full_release_complete"}
)


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
    if artifact.group is not None and (
        not isinstance(artifact.group, str) or not _SAFE_TABLE_GROUP.fullmatch(artifact.group)
    ):
        raise ReportingError(f"Table {name!r} has an unsafe group: {artifact.group!r}")
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
    if artifact.group is not None:
        tables_root = tables_root / artifact.group
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


def _target_path_sha256(target: Path) -> str:
    return hashlib.sha256(str(target).encode("utf-8")).hexdigest()


def _prepare_technical_replacement(
    target: Path,
    *,
    manifest: Mapping[str, Any],
) -> _ValidatedReplacement:
    """Bind an overwrite to one verified prior package without approval semantics."""

    try:
        prior = verify_release_package(target)
    except (OSError, UnicodeError, json.JSONDecodeError, ReportingError) as exc:
        raise ReportingError(
            "Existing release target is not a verified package and cannot be replaced"
        ) from exc
    replacement_run_id = manifest.get("run_id")
    if not isinstance(replacement_run_id, str) or not replacement_run_id.strip():
        raise ReportingError("Technical replacement requires a nonempty replacement run ID")
    prior_manifest_sha256 = sha256_file(prior.manifest_path)
    binding = {
        "schema_version": "technical-release-replacement-v1",
        "target_name": target.name,
        "target_path_sha256": _target_path_sha256(target),
        "replacement_run_id": replacement_run_id.strip(),
        "prior_manifest_sha256": prior_manifest_sha256,
        "preservation_policy": "verify_then_preserve_prior_package",
    }
    record_id = f"technical-{stable_json_sha256(binding)[:24]}"
    normalized_record = {
        **binding,
        "record_id": record_id,
        "status": "verified_prior_bound",
    }
    history_entry = (
        target.parent
        / ".release_history"
        / target.name
        / f"{prior_manifest_sha256[:16]}-{record_id}"
    )
    if history_entry.exists():
        raise ReportingError("Technical replacement binding has already been used")
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
            "Replacing an existing release requires a verified technical preservation binding"
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
        or runtime_policy.get("status") != "runtime_contract"
        or not isinstance(run_identity, Mapping)
        or run_identity.get("mode") != "full"
        or not isinstance(run_identity_sha256, str)
        or run_identity_sha256 != stable_json_sha256(run_identity)
    ):
        raise ReportingError(
            "Full-release runtime contract or run identity is missing or invalid"
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
        raise ReportingError("Full-release runtime-policy identity is misbound")

    # Full mode means complete data processing. Package integrity is enforced
    # through these deterministic content/run bindings and the checksum ledger;
    # it is deliberately independent of organizational approval artifacts.


def _nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _verify_release_status(manifest: Mapping[str, Any]) -> None:
    """Reject a pipeline package whose declared status was downgraded or renamed."""

    status = manifest.get("status")
    runtime_policy = manifest.get("runtime_policy")
    run_identity = manifest.get("run_identity")
    review_gate = manifest.get("review_gate")
    declared_modes = {
        str(candidate.get("mode"))
        for candidate in (runtime_policy, run_identity, review_gate)
        if isinstance(candidate, Mapping) and candidate.get("mode") is not None
    }
    pipeline_modes = declared_modes.intersection({"test", "full"})
    expected_status = (
        "phase_5_full_release_complete"
        if pipeline_modes == {"full"}
        else "phase_5_test_release_complete"
        if pipeline_modes == {"test"}
        else None
    )
    enforce_pipeline_status = status is not None or (
        isinstance(runtime_policy, Mapping)
        and runtime_policy.get("status") == "runtime_contract"
    )
    if enforce_pipeline_status and pipeline_modes and status != expected_status:
        raise ReportingError(
            "Release run manifest has an invalid release status for its pipeline mode"
        )


_QC_SUMMARY_SCOPE_FIELDS = {
    "gate_scope": "validate_only",
    "writing_mode_disposition": "preserve_complete_findings",
    "authority_status": "technical_run_not_scientific_approval",
}


def _verify_qc_summary_scope(manifest: Mapping[str, Any]) -> None:
    """Bind the QC gate's validate-only enforcement scope for pipeline packages.

    ``_enforce_phase_two_qc_gate`` only blocks in validate mode; a test/full
    package preserves review-bearing rows instead. These fields make that
    boundary machine-verifiable so a tampered manifest cannot imply
    scientific approval or organizational sign-off for a technical run.
    """

    if manifest.get("status") not in _PIPELINE_RELEASE_STATUSES:
        return
    qc_summary = manifest.get("qc_summary")
    if not isinstance(qc_summary, Mapping):
        raise ReportingError(
            "Pipeline release run manifest qc_summary is missing or invalid"
        )
    if qc_summary.get("gate_policy") != "fail_on_any_review":
        raise ReportingError(
            "Pipeline release run manifest qc_summary gate_policy is missing or altered"
        )
    for field, expected_value in _QC_SUMMARY_SCOPE_FIELDS.items():
        if qc_summary.get(field) != expected_value:
            raise ReportingError(
                f"Pipeline release run manifest qc_summary {field!r} is missing or altered"
            )


def _verify_manifest_artifact_inventory(
    checksums: Mapping[str, str],
    manifest: Mapping[str, Any],
) -> None:
    """Reconcile the writer's pre-manifest inventory when it is required or advertised."""

    inventory = manifest.get("artifact_sha256_before_manifest")
    if inventory is None:
        if manifest.get("status") in _PIPELINE_RELEASE_STATUSES:
            raise ReportingError(
                "Pipeline release is missing its required manifest artifact inventory"
            )
        return
    expected = {
        relative: digest
        for relative, digest in checksums.items()
        if relative != "run_manifest.json"
    }
    if (
        not isinstance(inventory, Mapping)
        or any(
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            for relative, digest in inventory.items()
        )
        or dict(inventory) != expected
    ):
        raise ReportingError(
            "Release run manifest artifact inventory does not reconcile with the package"
        )


def _verify_replacement_record(
    target: Path,
    checksums: Mapping[str, str],
    manifest: Mapping[str, Any],
) -> None:
    replacement = manifest.get("replacement")
    record_relative = "replacement_record.json"
    if replacement is None:
        if record_relative in checksums:
            raise ReportingError(
                "Release replacement record is not bound by the run manifest"
            )
        return
    if not isinstance(replacement, Mapping) or set(replacement) != {
        "record_path",
        "record_sha256",
        "prior_manifest_sha256",
        "preserved_prior_path",
    }:
        raise ReportingError("Release replacement metadata is invalid")
    if replacement.get("record_path") != record_relative:
        raise ReportingError("Release replacement record path is invalid")
    record_sha256 = replacement.get("record_sha256")
    if not isinstance(record_sha256, str) or checksums.get(record_relative) != record_sha256:
        raise ReportingError("Release replacement record checksum is invalid")
    record_path = target / record_relative
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportingError("Release replacement record is unreadable") from exc
    binding_fields = {
        "schema_version",
        "target_name",
        "target_path_sha256",
        "replacement_run_id",
        "prior_manifest_sha256",
        "preservation_policy",
    }
    if not isinstance(record, Mapping) or set(record) != binding_fields | {
        "record_id",
        "status",
    }:
        raise ReportingError("Release replacement record schema is invalid")
    binding = {field: record.get(field) for field in binding_fields}
    if (
        record.get("schema_version") != "technical-release-replacement-v1"
        or record.get("status") != "verified_prior_bound"
        or record.get("preservation_policy") != "verify_then_preserve_prior_package"
        or record.get("replacement_run_id") != manifest.get("run_id")
        or record.get("prior_manifest_sha256")
        != replacement.get("prior_manifest_sha256")
        or record.get("record_id")
        != f"technical-{stable_json_sha256(binding)[:24]}"
    ):
        raise ReportingError("Release replacement record binding is invalid")
    target_name = record.get("target_name")
    if (
        not isinstance(target_name, str)
        or not target_name
        or Path(target_name).name != target_name
    ):
        raise ReportingError("Release replacement target name is invalid")
    final_target = target.parent / target_name
    is_stage_target = target.name.startswith(f".{target_name}.stage-")
    if target.name != target_name and not is_stage_target:
        raise ReportingError("Release replacement target does not match the package path")
    if record.get("target_path_sha256") != _target_path_sha256(final_target):
        raise ReportingError("Release replacement target identity is invalid")
    record_id = str(record["record_id"])
    prior_manifest_sha256 = str(record["prior_manifest_sha256"])
    expected_preserved_path = (
        Path(".release_history")
        / target_name
        / f"{prior_manifest_sha256[:16]}-{record_id}"
        / "package"
    )
    if replacement.get("preserved_prior_path") != expected_preserved_path.as_posix():
        raise ReportingError("Release replacement preservation path is invalid")
    prior_package = final_target if is_stage_target else target.parent / expected_preserved_path
    if prior_package.resolve() == target.resolve() or not prior_package.is_dir():
        raise ReportingError("Release replacement prior package is missing")
    prior_manifest = prior_package / "run_manifest.json"
    if not prior_manifest.is_file() or sha256_file(prior_manifest) != prior_manifest_sha256:
        raise ReportingError("Release replacement prior manifest identity is invalid")
    try:
        verify_release_package(prior_package)
    except ReportingError as exc:
        raise ReportingError("Release replacement prior package verification failed") from exc


def _verify_table_artifacts(
    target: Path,
    checksums: Mapping[str, str],
    manifest: Mapping[str, Any],
) -> None:
    tables = manifest.get("tables")
    checksummed_table_paths = {
        relative for relative in checksums if relative.startswith("tables/")
    }
    if tables is None and not checksummed_table_paths:
        return
    if not isinstance(tables, Mapping):
        raise ReportingError("Release table inventory must be an object")
    referenced_paths: set[str] = set()
    for table_name, metadata in tables.items():
        if not isinstance(table_name, str) or not _SAFE_TABLE_NAME.fullmatch(table_name):
            raise ReportingError("Release table inventory contains an unsafe table name")
        if not isinstance(metadata, Mapping) or set(metadata) != {
            "row_count",
            "stable_key",
            "column_names",
            "artifact_paths",
            "readback_row_counts",
        }:
            raise ReportingError(f"Release table {table_name!r} metadata is invalid")
        row_count = metadata.get("row_count")
        column_names = metadata.get("column_names")
        stable_key = metadata.get("stable_key")
        artifact_paths = metadata.get("artifact_paths")
        readback_row_counts = metadata.get("readback_row_counts")
        if not _nonnegative_integer(row_count):
            raise ReportingError(f"Release table {table_name!r} row count is invalid")
        if (
            not isinstance(column_names, list)
            or any(not isinstance(column, str) or not column for column in column_names)
            or len(column_names) != len(set(column_names))
        ):
            raise ReportingError(f"Release table {table_name!r} column inventory is invalid")
        if stable_key is not None and (
            not isinstance(stable_key, str)
            or not stable_key
            or bool(column_names)
            and stable_key not in column_names
        ):
            raise ReportingError(f"Release table {table_name!r} stable key is invalid")
        if (
            not isinstance(artifact_paths, list)
            or not artifact_paths
            or any(not isinstance(relative, str) for relative in artifact_paths)
            or len(artifact_paths) != len(set(artifact_paths))
            or not isinstance(readback_row_counts, Mapping)
        ):
            raise ReportingError(f"Release table {table_name!r} artifact inventory is invalid")
        observed_formats: set[str] = set()
        for relative in artifact_paths:
            path = Path(relative)
            output_format = path.suffix.casefold().lstrip(".")
            if (
                len(path.parts) not in {2, 3}
                or path.parts[0] != "tables"
                or path.stem != table_name
                or output_format not in SUPPORTED_TABLE_FORMATS
                or relative not in checksums
                or relative in referenced_paths
            ):
                raise ReportingError(
                    f"Release table {table_name!r} artifact path is invalid"
                )
            referenced_paths.add(relative)
            observed_formats.add(output_format)
            artifact_path = target / relative
            try:
                if output_format == "csv":
                    try:
                        readback = pd.read_csv(artifact_path)
                    except pd.errors.EmptyDataError:
                        if row_count == 0 and column_names == []:
                            readback = pd.DataFrame()
                        else:
                            raise
                elif output_format == "parquet":
                    readback = pd.read_parquet(artifact_path)
                else:
                    readback = pd.read_excel(artifact_path)
            except (
                ImportError,
                OSError,
                TypeError,
                UnicodeError,
                ValueError,
                pd.errors.ParserError,
                pd.errors.EmptyDataError,
            ) as exc:
                raise ReportingError(
                    f"Release table {table_name!r} {output_format} artifact failed read-back"
                ) from exc
            if len(readback) != row_count or list(map(str, readback.columns)) != column_names:
                raise ReportingError(
                    f"Release table {table_name!r} {output_format} read-back does not match its declared schema"
                )
            if stable_key is not None and stable_key in readback:
                stable_values = readback[stable_key]
                if bool(stable_values.isna().any()) or bool(
                    stable_values.astype(str).duplicated().any()
                ):
                    raise ReportingError(
                        f"Release table {table_name!r} {output_format} read-back stable key is invalid"
                    )
        if set(readback_row_counts) != observed_formats or any(
            not _nonnegative_integer(value) or value != row_count
            for value in readback_row_counts.values()
        ):
            raise ReportingError(
                f"Release table {table_name!r} read-back counts do not reconcile"
            )
    if referenced_paths != checksummed_table_paths:
        raise ReportingError(
            "Release table inventory does not account for every checksummed table artifact"
        )


def _verify_review_issue_ledger(
    target: Path,
    checksums: Mapping[str, str],
    manifest: Mapping[str, Any],
) -> None:
    """Validate the complete review ledger semantically, not only by checksum."""

    pipeline_release = manifest.get("status") in _PIPELINE_RELEASE_STATUSES
    manifest_gate = manifest.get("review_gate")
    ledger_advertised = _REVIEW_ISSUE_LEDGER_PATH in checksums
    if not pipeline_release and manifest_gate is None and not ledger_advertised:
        return
    if not isinstance(manifest_gate, Mapping) or not ledger_advertised:
        raise ReportingError(
            "Release review ledger and run-manifest review gate must be present together"
        )
    ledger_path = target / _REVIEW_ISSUE_LEDGER_PATH
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportingError("Release review issue ledger is not valid JSON") from exc
    if not isinstance(ledger, Mapping):
        raise ReportingError("Release review issue ledger must be a JSON object")

    required_fields = {
        "schema_version",
        "policy_id",
        "prospective_effective_version",
        "review_gate_policy_sha256",
        "mode",
        "run_identity_sha256",
        "policy_content_sha256",
        "expected_ledgers",
        "observed_ledgers",
        "evidence_complete",
        "issue_count",
        "permitted_disposition_count",
        "blocking_issue_count",
        "issue_state_counts",
        "issues",
        "decision",
        "complete_processing_allowed",
    }
    if set(ledger) != required_fields:
        raise ReportingError("Release review issue ledger schema is invalid")
    schema_version = ledger.get("schema_version")
    if schema_version not in _REVIEW_GATE_SCHEMA_VERSIONS:
        raise ReportingError("Release review issue ledger schema version is unsupported")

    issues = ledger.get("issues")
    if not isinstance(issues, list):
        raise ReportingError("Release review issue ledger issues must be a list")
    permitted_count = 0
    blocking_count = 0
    issue_uids: set[str] = set()
    issue_state_counts: Counter[str] = Counter()
    required_issue_fields = {
        "review_issue_uid",
        "stage",
        "issue_scope",
        "subject_id",
        "issue_state",
        "status",
        "reason_codes",
        "detail",
        "gate_classification",
        "blocking_reason_codes",
    }
    semantic_issue_fields = (
        "stage",
        "issue_scope",
        "subject_id",
        "issue_state",
        "status",
        "reason_codes",
        "detail",
    )
    for issue in issues:
        if not isinstance(issue, Mapping) or not required_issue_fields <= set(issue):
            raise ReportingError("Release review issue ledger contains an invalid issue record")
        issue_uid = issue.get("review_issue_uid")
        stage = issue.get("stage")
        issue_scope = issue.get("issue_scope")
        subject_id = issue.get("subject_id")
        issue_state = issue.get("issue_state")
        status = issue.get("status")
        reason_codes = issue.get("reason_codes")
        detail = issue.get("detail")
        classification = issue.get("gate_classification")
        blocking_reasons = issue.get("blocking_reason_codes")
        if (
            not isinstance(issue_uid, str)
            or _REVIEW_ISSUE_UID.fullmatch(issue_uid) is None
            or issue_uid in issue_uids
            or stage not in _REVIEW_ISSUE_STAGES
            or not isinstance(issue_scope, str)
            or not issue_scope.strip()
            or not isinstance(subject_id, str)
            or not subject_id.strip()
            or issue_state not in _REVIEW_ISSUE_STATES
            or not isinstance(status, str)
            or not status.strip()
            or not isinstance(reason_codes, list)
            or not reason_codes
            or any(not isinstance(reason, str) or not reason.strip() for reason in reason_codes)
            or (detail is not None and (not isinstance(detail, str) or not detail.strip()))
            or classification not in {"blocking", "permitted_resolved_disposition"}
            or not isinstance(blocking_reasons, list)
            or any(
                not isinstance(reason, str) or not reason.strip()
                for reason in blocking_reasons
            )
            or set(blocking_reasons) - {"ANALYTICAL_LEAKAGE_DETECTED"}
            or (
                classification == "permitted_resolved_disposition"
                and (issue_state != "excluded_series" or blocking_reasons)
            )
        ):
            raise ReportingError("Release review issue ledger contains an invalid issue record")
        expected_issue_uid = "review_issue_" + stable_json_sha256(
            {field: issue.get(field) for field in semantic_issue_fields}
        )[:24]
        if issue_uid != expected_issue_uid:
            raise ReportingError(
                "Release review issue ledger identity is not bound to its semantic payload"
            )
        issue_uids.add(issue_uid)
        issue_state_counts[str(issue_state)] += 1
        if classification == "permitted_resolved_disposition":
            permitted_count += 1
        else:
            blocking_count += 1

    issue_count = ledger.get("issue_count")
    recorded_permitted_count = ledger.get("permitted_disposition_count")
    recorded_blocking_count = ledger.get("blocking_issue_count")
    recorded_state_counts = ledger.get("issue_state_counts")
    if (
        not _nonnegative_integer(issue_count)
        or not _nonnegative_integer(recorded_permitted_count)
        or not _nonnegative_integer(recorded_blocking_count)
        or issue_count != len(issues)
        or recorded_permitted_count != permitted_count
        or recorded_blocking_count != blocking_count
        or issue_count != permitted_count + blocking_count
        or not isinstance(recorded_state_counts, Mapping)
        or any(
            state not in _REVIEW_ISSUE_STATES or not _nonnegative_integer(count)
            for state, count in recorded_state_counts.items()
        )
        or dict(recorded_state_counts) != dict(sorted(issue_state_counts.items()))
    ):
        raise ReportingError("Release review issue ledger counts do not reconcile")

    expected_ledgers = ledger.get("expected_ledgers")
    observed_ledgers = ledger.get("observed_ledgers")
    mode = ledger.get("mode")
    decision = ledger.get("decision")
    expected_decision = "bounded_with_findings" if blocking_count else "pass"
    if (
        not isinstance(expected_ledgers, list)
        or tuple(expected_ledgers) != _REVIEW_GATE_EXPECTED_LEDGERS
        or not isinstance(observed_ledgers, list)
        or tuple(observed_ledgers) != _REVIEW_GATE_EXPECTED_LEDGERS
        or ledger.get("evidence_complete") is not True
        or mode not in {"test", "full"}
        or decision != expected_decision
        or ledger.get("complete_processing_allowed") is not True
    ):
        raise ReportingError(
            "Release review issue ledger completeness or decision does not reconcile"
        )

    run_identity_sha256 = ledger.get("run_identity_sha256")
    policy_content_sha256 = ledger.get("policy_content_sha256")
    runtime_policy = manifest.get("runtime_policy")
    run_identity = manifest.get("run_identity")
    status = manifest.get("status")
    if (
        not isinstance(run_identity_sha256, str)
        or _SHA256.fullmatch(run_identity_sha256) is None
        or manifest.get("run_identity_sha256") != run_identity_sha256
        or not isinstance(policy_content_sha256, str)
        or _SHA256.fullmatch(policy_content_sha256) is None
        or not isinstance(runtime_policy, Mapping)
        or runtime_policy.get("policy_content_sha256") != policy_content_sha256
        or runtime_policy.get("mode") not in {None, mode}
        or (isinstance(run_identity, Mapping) and run_identity.get("mode") != mode)
        or (
            isinstance(run_identity, Mapping)
            and stable_json_sha256(run_identity) != run_identity_sha256
        )
        or (
            isinstance(status, str)
            and status.endswith("_release_complete")
            and status != f"phase_5_{mode}_release_complete"
        )
    ):
        raise ReportingError("Release review issue ledger run or policy identity is misbound")

    policy_id = ledger.get("policy_id")
    prospective_version = ledger.get("prospective_effective_version")
    review_policy_sha256 = ledger.get("review_gate_policy_sha256")
    if schema_version == "ops-03-review-gate-v1":
        if (
            policy_id != "OPS-03-option-b"
            or prospective_version is not None
            or review_policy_sha256 is not None
        ):
            raise ReportingError("Release review issue ledger v1 policy binding is invalid")
    else:
        review_policy = runtime_policy.get("review_gate_policy")
        if (
            not isinstance(policy_id, str)
            or not policy_id.strip()
            or not isinstance(prospective_version, str)
            or not prospective_version.strip()
            or not isinstance(review_policy_sha256, str)
            or _SHA256.fullmatch(review_policy_sha256) is None
            or not isinstance(review_policy, Mapping)
            or review_policy.get("policy_id") != policy_id
            or review_policy.get("prospective_effective_version") != prospective_version
            or review_policy.get("artifact_sha256") != review_policy_sha256
        ):
            raise ReportingError("Release review issue ledger v2 policy binding is invalid")

    manifest_projection = {key: value for key, value in ledger.items() if key != "issues"}
    if dict(manifest_gate) != manifest_projection:
        raise ReportingError(
            "Release review issue ledger does not agree with the run manifest"
        )


def _figure_formats(manifest: Mapping[str, Any], figures: Mapping[str, Any]) -> tuple[str, ...]:
    raw_formats = figures.get("formats")
    output_profile = manifest.get("output_profile")
    if (
        not isinstance(raw_formats, list)
        or not raw_formats
        or any(not isinstance(item, str) for item in raw_formats)
    ):
        raise ReportingError("Release figure format inventory is invalid")
    formats = tuple(item.casefold().lstrip(".") for item in raw_formats)
    if (
        len(formats) != len(set(formats))
        or any(item not in SUPPORTED_FIGURE_FORMATS for item in formats)
        or not isinstance(output_profile, Mapping)
        or tuple(output_profile.get("figure_formats", ())) != formats
    ):
        raise ReportingError("Release figure format inventory is invalid")
    runtime_policy = manifest.get("runtime_policy")
    effective_enablement = (
        runtime_policy.get("effective_enablement")
        if isinstance(runtime_policy, Mapping)
        else None
    )
    if isinstance(effective_enablement, Mapping) and (
        "figure_formats" in effective_enablement
        and tuple(effective_enablement["figure_formats"]) != formats
    ):
        raise ReportingError("Release figure formats disagree with the runtime policy")
    return formats


def _figure_token_map(
    value: Any,
    *,
    name: str,
) -> tuple[dict[str, str], dict[str, str]]:
    if not isinstance(value, Mapping):
        raise ReportingError(f"Release {name} figure-directory token inventory is invalid")
    tokens: dict[str, str] = {}
    owners: dict[str, str] = {}
    for identity, token in value.items():
        if (
            not isinstance(identity, str)
            or not identity.strip()
            or not isinstance(token, str)
            or _SAFE_FIGURE_DIRECTORY_TOKEN.fullmatch(token) is None
            or token.upper() in _WINDOWS_RESERVED_BASENAMES
            or len(token.encode("utf-8")) > 80
            or token.casefold() in owners
        ):
            raise ReportingError(f"Release {name} figure-directory token inventory is invalid")
        tokens[identity] = token
        owners[token.casefold()] = identity
    return tokens, {token: identity for identity, token in tokens.items()}


def _integer_count_map(value: Any, *, where: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not key or not _nonnegative_integer(count)
        for key, count in value.items()
    ):
        raise ReportingError(f"Release figure {where} count inventory is invalid")
    return dict(value)


def _verify_raster_figure(path: Path, *, output_format: str) -> None:
    expected_format = {"jpeg": "JPEG", "png": "PNG"}.get(output_format)
    if expected_format is None:
        raise ReportingError("Release figure artifact format is unsupported")
    try:
        with Image.open(path) as image:
            detected_format = image.format
            image.verify()
        with Image.open(path) as image:
            image.load()
            dimensions = image.size
    except (OSError, SyntaxError, ValueError) as exc:
        raise ReportingError(
            f"Release figure failed image read-back validation: {path.name}"
        ) from exc
    if detected_format != expected_format or any(dimension < 1 for dimension in dimensions):
        raise ReportingError(
            f"Release figure failed image read-back validation: {path.name}"
        )


def _verify_run_figure_inventory(
    target: Path,
    checksums: Mapping[str, str],
    manifest: Mapping[str, Any],
    figures: Mapping[str, Any],
) -> None:
    formats = _figure_formats(manifest, figures)
    source_tokens, source_by_token = _figure_token_map(
        figures.get("source_directory_tokens"),
        name="source",
    )
    model_tokens, model_by_token = _figure_token_map(
        figures.get("model_directory_tokens"),
        name="model",
    )
    layout_version = figures.get("layout_version")
    if layout_version not in _KNOWN_FIGURE_LAYOUT_VERSIONS:
        raise ReportingError("Release figure layout version is unrecognized")
    if (
        figures.get("observed_path_template")
        != "figures/observed/<source>/<series>.<format>"
        or figures.get("fitted_path_template")
        != "figures/fitted/<source>/<model>/<series>__<attempt>.<format>"
    ):
        raise ReportingError("Release figure hierarchy metadata is invalid")
    is_v2_layout = layout_version == _FIGURE_LAYOUT_VERSION_V2
    overlay_declaration = figures.get("overlay")
    if is_v2_layout:
        if (
            not isinstance(overlay_declaration, Mapping)
            or overlay_declaration.get("scope") != _FIGURE_OVERLAY_SCOPE
            or overlay_declaration.get("path_template") != _FIGURE_OVERLAY_PATH_TEMPLATE
        ):
            raise ReportingError("Release figure overlay hierarchy metadata is invalid")
    elif overlay_declaration is not None:
        raise ReportingError(
            "Legacy figure layout must not declare a source overlay"
        )

    runtime_policy = manifest.get("runtime_policy")
    effective_enablement = (
        runtime_policy.get("effective_enablement")
        if isinstance(runtime_policy, Mapping)
        else None
    )
    if isinstance(effective_enablement, Mapping):
        enabled_sources = effective_enablement.get("enabled_sources")
        enabled_models = effective_enablement.get("enabled_models")
        if isinstance(enabled_sources, list) and set(source_tokens) != set(enabled_sources):
            raise ReportingError("Release figure source hierarchy disagrees with runtime policy")
        if isinstance(enabled_models, list) and set(model_tokens) != set(enabled_models):
            raise ReportingError("Release figure model hierarchy disagrees with runtime policy")

    figure_paths = sorted(
        relative for relative in checksums if relative.startswith("figures/")
    )
    logical_formats: dict[str, list[str]] = {}
    logical_metadata: dict[str, tuple[str, str, str | None]] = {}
    for relative in figure_paths:
        relative_path = Path(relative)
        output_format = relative_path.suffix.casefold().lstrip(".")
        if output_format not in formats:
            raise ReportingError("Release figure hierarchy contains an undeclared artifact format")
        _verify_raster_figure(target / relative, output_format=output_format)
        parts = relative_path.parts
        if len(parts) == 4 and parts[:2] == ("figures", "observed"):
            source_name = source_by_token.get(parts[2])
            model_name = None
            kind = "observed"
        elif len(parts) == 5 and parts[:2] == ("figures", "fitted"):
            source_name = source_by_token.get(parts[2])
            model_name = model_by_token.get(parts[3])
            kind = "fitted"
        elif is_v2_layout and len(parts) == 3 and parts[:2] == ("figures", "overlay"):
            source_name = source_by_token.get(relative_path.stem)
            model_name = None
            kind = "overlay"
        else:
            raise ReportingError("Release figure artifact is outside the declared hierarchy")
        if source_name is None or (kind == "fitted" and model_name is None) or not relative_path.stem:
            raise ReportingError("Release figure artifact is outside the declared hierarchy")
        logical_path = relative_path.with_suffix("").as_posix()
        logical_formats.setdefault(logical_path, []).append(output_format)
        logical_metadata[logical_path] = (kind, source_name, model_name)
    if any(
        len(observed) != len(formats) or set(formats) != set(observed)
        for observed in logical_formats.values()
    ):
        raise ReportingError("Release figure artifact formats are incomplete or inconsistent")

    observed_by_source: Counter[str] = Counter()
    fitted_by_source_model: Counter[tuple[str, str]] = Counter()
    overlay_sources: set[str] = set()
    overlay_artifact_count = 0
    for logical_path, (kind, source_name, model_name) in logical_metadata.items():
        if kind == "observed":
            observed_by_source[source_name] += 1
        elif kind == "fitted":
            assert model_name is not None
            fitted_by_source_model[(source_name, model_name)] += 1
        else:
            overlay_sources.add(source_name)
            overlay_artifact_count += len(logical_formats[logical_path])
    observed_count = sum(observed_by_source.values())
    fitted_count = sum(fitted_by_source_model.values())
    expected_observed_by_source = {
        source: observed_by_source[source] for source in sorted(source_tokens)
    }
    expected_fitted_by_source = {
        source: sum(
            fitted_by_source_model[(source, model)] for model in sorted(model_tokens)
        )
        for source in sorted(source_tokens)
    }
    expected_fitted_matrix = {
        source: {
            model: fitted_by_source_model[(source, model)]
            for model in sorted(model_tokens)
        }
        for source in sorted(source_tokens)
    }
    observed_counts = _integer_count_map(
        figures.get("observed_figure_count_by_source"),
        where="observed-by-source",
    )
    fitted_counts = _integer_count_map(
        figures.get("fitted_figure_count_by_source"),
        where="fitted-by-source",
    )
    raw_fitted_matrix = figures.get("fitted_figure_count_by_source_model")
    if not isinstance(raw_fitted_matrix, Mapping):
        raise ReportingError("Release figure fitted-by-source/model count inventory is invalid")
    fitted_matrix = {
        source: _integer_count_map(counts, where="fitted-by-source/model")
        for source, counts in raw_fitted_matrix.items()
        if isinstance(source, str)
    }
    if len(fitted_matrix) != len(raw_fitted_matrix):
        raise ReportingError("Release figure fitted-by-source/model count inventory is invalid")
    if (
        not _nonnegative_integer(figures.get("observed_figure_count"))
        or figures.get("observed_figure_count") != observed_count
        or not _nonnegative_integer(figures.get("fitted_figure_count"))
        or figures.get("fitted_figure_count") != fitted_count
        or not _nonnegative_integer(figures.get("artifact_count"))
        or figures.get("artifact_count") != len(figure_paths)
        or observed_counts != expected_observed_by_source
        or fitted_counts != expected_fitted_by_source
        or fitted_matrix != expected_fitted_matrix
    ):
        raise ReportingError("Release figure counts do not reconcile with the artifact hierarchy")

    if is_v2_layout:
        expected_overlay_figure_count_by_source = {
            source: 1 if expected_observed_by_source[source] > 0 else 0
            for source in sorted(source_tokens)
        }
        expected_overlay_sources = {
            source
            for source, count in expected_overlay_figure_count_by_source.items()
            if count == 1
        }
        if overlay_sources != expected_overlay_sources:
            raise ReportingError(
                "Release figure source overlay is missing or unexpected for an observed source"
            )
        overlay_counts_by_source = _integer_count_map(
            overlay_declaration.get("figure_count_by_source"),
            where="overlay-by-source",
        )
        overlay_series_counts_by_source = _integer_count_map(
            overlay_declaration.get("series_count_by_source"),
            where="overlay-series-by-source",
        )
        if (
            not _nonnegative_integer(overlay_declaration.get("figure_count"))
            or overlay_declaration.get("figure_count") != len(overlay_sources)
            or not _nonnegative_integer(overlay_declaration.get("artifact_count"))
            or overlay_declaration.get("artifact_count") != overlay_artifact_count
            or not _nonnegative_integer(overlay_declaration.get("series_count"))
            or overlay_declaration.get("series_count") != observed_count
            or overlay_counts_by_source != expected_overlay_figure_count_by_source
            or overlay_series_counts_by_source != expected_observed_by_source
            or any(value not in (0, 1) for value in overlay_counts_by_source.values())
        ):
            raise ReportingError(
                "Release figure overlay counts do not reconcile with the artifact hierarchy"
            )
    status = figures.get("status")
    if status == "run_output" and not figure_paths:
        raise ReportingError("Release run-output figure inventory is empty")
    if status == "no_renderable_series" and figure_paths:
        raise ReportingError("Release no-renderable-series figure inventory contains artifacts")


def _verify_sample_figure_inventory(
    target: Path,
    checksums: Mapping[str, str],
    manifest: Mapping[str, Any],
    figures: Mapping[str, Any],
) -> None:
    formats = _figure_formats(manifest, figures)
    if (
        manifest.get("status") != "phase_5_test_release_complete"
        or figures.get("directory") != "figures/sample"
        or figures.get("index_path") != "figures/sample/INDEX.md"
    ):
        raise ReportingError("Synthetic sample figures are only valid in a test package")
    figure_paths = sorted(
        relative for relative in checksums if relative.startswith("figures/")
    )
    if "figures/sample/INDEX.md" not in figure_paths:
        raise ReportingError("Synthetic sample figure index is missing")
    logical_formats: dict[str, list[str]] = {}
    observed_count = 0
    fitted_count = 0
    overlay_count = 0
    for relative in figure_paths:
        if relative == "figures/sample/INDEX.md":
            continue
        path = Path(relative)
        parts = path.parts
        output_format = path.suffix.casefold().lstrip(".")
        if (
            len(parts) != 4
            or parts[:2] != ("figures", "sample")
            or parts[2] not in {"observed", "fitted", "overlay"}
            or output_format not in formats
            or not path.stem
        ):
            raise ReportingError("Synthetic sample figure artifact inventory is invalid")
        _verify_raster_figure(target / relative, output_format=output_format)
        logical_path = path.with_suffix("").as_posix()
        if logical_path not in logical_formats:
            if parts[2] == "observed":
                observed_count += 1
            elif parts[2] == "fitted":
                fitted_count += 1
            else:
                overlay_count += 1
        logical_formats.setdefault(logical_path, []).append(output_format)
    if any(
        len(observed) != len(formats) or set(formats) != set(observed)
        for observed in logical_formats.values()
    ):
        raise ReportingError("Synthetic sample figure formats are incomplete or inconsistent")
    if (
        not _nonnegative_integer(figures.get("observed_figure_count"))
        or figures.get("observed_figure_count") != observed_count
        or not _nonnegative_integer(figures.get("fitted_figure_count"))
        or figures.get("fitted_figure_count") != fitted_count
        or not _nonnegative_integer(figures.get("artifact_count"))
        or figures.get("artifact_count") != len(figure_paths)
    ):
        raise ReportingError("Synthetic sample figure counts do not reconcile")
    declared_overlay_count = figures.get("overlay_figure_count")
    if declared_overlay_count is None:
        if overlay_count != 0:
            raise ReportingError(
                "Synthetic sample figure overlay inventory is undeclared"
            )
    elif (
        not _nonnegative_integer(declared_overlay_count)
        or declared_overlay_count != overlay_count
    ):
        raise ReportingError("Synthetic sample figure overlay counts do not reconcile")


def _verify_figure_inventory(
    target: Path,
    checksums: Mapping[str, str],
    manifest: Mapping[str, Any],
) -> None:
    pipeline_release = manifest.get("status") in _PIPELINE_RELEASE_STATUSES
    figures = manifest.get("figures")
    if figures is None and not pipeline_release:
        return
    if figures is None:
        raise ReportingError(
            "Pipeline release is missing its required figure inventory contract"
        )
    if not isinstance(figures, Mapping):
        raise ReportingError("Release run manifest figure inventory must be an object")
    status = figures.get("status")
    if status in {"run_output", "no_renderable_series"}:
        _verify_run_figure_inventory(target, checksums, manifest, figures)
    elif status == "synthetic_demonstration_substituted":
        _verify_sample_figure_inventory(target, checksums, manifest, figures)
    else:
        raise ReportingError("Release run manifest figure inventory has an invalid status")


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
    _verify_release_status(payload)
    _verify_qc_summary_scope(payload)
    _verify_manifest_artifact_inventory(expected, payload)
    _verify_replacement_record(target, expected, payload)
    _verify_full_release_governance(target, expected, payload)
    _verify_review_issue_ledger(target, expected, payload)
    _verify_figure_inventory(target, expected, payload)
    _verify_table_artifacts(target, expected, payload)
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
    release_validator: Callable[[Mapping[str, Any]], None] | None = None,
    staged_artifact_validator: Callable[[Path], None] | None = None,
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
        _prepare_technical_replacement(
            target,
            manifest=initial_manifest,
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
        if staged_artifact_validator is not None:
            if not callable(staged_artifact_validator):
                raise ReportingError("Staged artifact validator must be callable")
            staged_artifact_validator(stage)
            for relative_path, digest in artifact_sha256.items():
                if sha256_file(stage / relative_path) != digest:
                    raise ReportingError(
                        "Staged artifact validator changed a registered package artifact: "
                        f"{relative_path}"
                    )
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
