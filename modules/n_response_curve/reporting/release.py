from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
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
_FIGURE_LAYOUT_VERSION_V3 = "figures-by-source-model-overlay-only-v3"
_KNOWN_FIGURE_LAYOUT_VERSIONS = frozenset(
    {_FIGURE_LAYOUT_VERSION, _FIGURE_LAYOUT_VERSION_V2, _FIGURE_LAYOUT_VERSION_V3}
)
_FIGURE_OVERLAY_PATH_TEMPLATE = "figures/overlay/<source>.<format>"
_FIGURE_OVERLAY_SCOPE = "same_governed_series_as_observed_figures"
_FIGURE_OVERLAY_SCOPE_V3 = "governed_observed_series_overlay_only"
_SAFE_FIGURE_DIRECTORY_TOKEN = re.compile(r"^[A-Za-z0-9_-]+$")
_WINDOWS_RESERVED_BASENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)
_PIPELINE_RELEASE_STATUSES = frozenset(
    {"phase_5_test_release_complete", "phase_5_full_release_complete"}
)

# Contract for the post-release custom-overlay diagnostic bundle written by
# ``reporting/generate_custom_overlays.py`` under ``figures/overlay/`` atop an
# already-promoted, already-checksummed package. The bundle is intentionally
# unbound (it is not listed in CHECKSUMS.sha256), so it is invisible to that
# generator module and is reproduced here as literal contract constants
# instead of importing it, so a *historical* prior package is verified purely
# from its own bound checksum ledger and this fixed schema -- never from the
# generator's current code or config.
_PRIOR_DIAGNOSTIC_OVERLAY_DIR = "figures/overlay"
_CUSTOM_PLOTS_MANIFEST_NAME = "custom_plots_manifest.json"
_CUSTOM_PLOTS_CHECKSUMS_NAME = "CUSTOM_PLOTS_CHECKSUMS.sha256"
_ZERO_N_DIAGNOSTIC_MANIFEST_NAME = "zero_n_strata_diagnostic_manifest.json"
_ZERO_N_DIAGNOSTIC_CHECKSUMS_NAME = "ZERO_N_STRATA_DIAGNOSTIC_CHECKSUMS.sha256"
_CUSTOM_PLOTS_MANIFEST_SCHEMA_VERSION = "post-release-custom-plots-manifest-v1"
_CUSTOM_PLOTS_MANIFEST_STATUS = "post_release_diagnostics_unbound"
_ZERO_N_DIAGNOSTIC_MANIFEST_SCHEMA_VERSION = "post-release-zero-n-strata-diagnostic-v1"
_ZERO_N_DIAGNOSTIC_MANIFEST_STATUS = "post_release_diagnostic_unbound"
_ORIGINAL_RELEASE_INTEGRITY_BASE_FIELDS = frozenset(
    {
        "bound_artifact_count",
        "bound_artifacts_reverified_unchanged",
        "run_manifest_sha256",
        "release_checksum_ledger_sha256",
    }
)
_ZERO_N_ORIGINAL_RELEASE_INTEGRITY_EXTRA_FIELDS = frozenset(
    {
        "semantic_verifier_before_additions",
        "original_bound_files_unchanged_before_promotion",
    }
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
        return json.dumps(
            _json_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return value


def _validate_table(name: str, artifact: TableArtifact) -> tuple[dict[str, Any], ...]:
    if not isinstance(name, str) or not _SAFE_TABLE_NAME.fullmatch(name):
        raise ReportingError(f"Table name is unsafe: {name!r}")
    if not isinstance(artifact, TableArtifact):
        raise ReportingError(f"Table {name!r} must be a TableArtifact")
    if artifact.group is not None and (
        not isinstance(artifact.group, str)
        or not _SAFE_TABLE_GROUP.fullmatch(artifact.group)
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
        normalized = [
            json.dumps(value, sort_keys=True, default=str) for value in values
        ]
        if len(normalized) != len(set(normalized)):
            raise ReportingError(f"Table {name!r} has a duplicate stable key: {key}")
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            _json_value(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_table(
    stage_root: Path, name: str, artifact: TableArtifact, formats: Sequence[str]
) -> tuple[dict[str, Any], dict[str, str]]:
    rows = _validate_table(name, artifact)
    tables_root = stage_root / "tables"
    if artifact.group is not None:
        tables_root = tables_root / artifact.group
    tables_root.mkdir(parents=True, exist_ok=True)
    normalized_rows = [
        {str(key): _flat_value(value) for key, value in row.items()} for row in rows
    ]
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
        except (
            ImportError,
            OSError,
            TypeError,
            ValueError,
            pd.errors.ParserError,
        ) as exc:
            raise ReportingError(
                f"Table {name!r} {output_format} artifact failed read-back validation"
            ) from exc
        if len(readback) != len(frame):
            raise ReportingError(
                f"Table {name!r} {output_format} read-back row count {len(readback)} does not match {len(frame)}"
            )
        if set(map(str, readback.columns)) != set(map(str, frame.columns)):
            raise ReportingError(
                f"Table {name!r} {output_format} read-back columns do not match"
            )
        if artifact.stable_key is not None and artifact.stable_key in readback:
            stable_values = readback[artifact.stable_key]
            if bool(stable_values.isna().any()) or bool(
                stable_values.astype(str).duplicated().any()
            ):
                raise ReportingError(
                    f"Table {name!r} {output_format} read-back stable key is missing or duplicated"
                )
        readback_row_counts[output_format] = len(readback)
        artifacts[destination.relative_to(stage_root).as_posix()] = sha256_file(
            destination
        )
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
        messages = [
            str(message).strip()
            for message in report_sections.get(key, ())
            if str(message).strip()
        ]
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
        text_lines[start : start + 52] for start in range(0, len(text_lines), 52)
    )
    page_object_numbers = tuple(4 + index * 2 for index in range(len(pages)))
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            b"<< /Type /Pages /Count "
            + str(len(pages)).encode("ascii")
            + b" /Kids ["
            + b" ".join(
                f"{number} 0 R".encode("ascii") for number in page_object_numbers
            )
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
        raise ReportingError(
            "Rendered PDF report failed structural read-back validation"
        )
    if b"/CreationDate" in payload or b"/ModDate" in payload:
        raise ReportingError(
            "Rendered PDF report contains nondeterministic timestamp metadata"
        )
    try:
        startxref = int(payload.rsplit(b"startxref\n", 1)[1].splitlines()[0])
    except (IndexError, ValueError) as exc:
        raise ReportingError(
            "Rendered PDF report has an invalid cross-reference offset"
        ) from exc
    if startxref < 1 or payload[startxref : startxref + 5] != b"xref\n":
        raise ReportingError(
            "Rendered PDF report cross-reference offset does not resolve"
        )


def _assert_safe_target(target: Path, source_roots: Iterable[str | Path]) -> None:
    for source_root in source_roots:
        normalized_source = Path(source_root).resolve()
        if target == normalized_source or target.is_relative_to(normalized_source):
            raise ReportingError(
                f"Refusing source target for release package: {target}"
            )


def _target_path_sha256(target: Path) -> str:
    return hashlib.sha256(str(target).encode("utf-8")).hexdigest()


def _prepare_technical_replacement(
    target: Path,
    *,
    manifest: Mapping[str, Any],
) -> _ValidatedReplacement:
    """Bind an overwrite to one verified prior package without approval semantics."""

    try:
        prior = _verify_prior_release_package(target)
    except (OSError, UnicodeError, json.JSONDecodeError, ReportingError) as exc:
        raise ReportingError(
            "Existing release target is not a verified package and cannot be replaced"
        ) from exc
    replacement_run_id = manifest.get("run_id")
    if not isinstance(replacement_run_id, str) or not replacement_run_id.strip():
        raise ReportingError(
            "Technical replacement requires a nonempty replacement run ID"
        )
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
        raise ReportingError(
            f"Release package collision at {target}; set overwrite explicitly to replace it"
        )
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
            raise ReportingError(
                f"Unable to atomically promote release package: {exc}"
            ) from exc
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

    def verify_prior_for_recovery(package: Path) -> None:
        try:
            _verify_prior_release_package(package)
        except (OSError, UnicodeError, json.JSONDecodeError, ReportingError) as exc:
            raise ReportingError(
                "Release-promotion prior package verification failed during recovery"
            ) from exc

    history_root = target.parent / ".release_history" / target.name
    if not history_root.is_dir():
        return
    pending: list[tuple[Path, str]] = []
    unjournaled: list[Path] = []
    for history_entry in sorted(
        path for path in history_root.iterdir() if path.is_dir()
    ):
        state_path = history_entry / "promotion_state.json"
        if not state_path.is_file():
            unjournaled.append(history_entry)
            continue
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReportingError(
                "Release-promotion recovery journal is unreadable"
            ) from exc
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != "release-promotion-v1"
            or payload.get("target_name") != target.name
            or payload.get("target_path_sha256") != _target_path_sha256(target)
            or payload.get("state") not in {"prepared", "prior_archived", "complete"}
        ):
            raise ReportingError(
                "Release-promotion recovery journal is malformed or misbound"
            )
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
        verify_prior_for_recovery(target)
        shutil.rmtree(history_entry)
        return
    if not pending:
        return
    if len(pending) != 1:
        raise ReportingError(
            "Multiple interrupted release promotions require manual review"
        )
    history_entry, state = pending[0]
    archived_package = history_entry / "package"
    if target.exists() and archived_package.exists():
        try:
            verify_release_package(target)
        except ReportingError:
            failed_target = history_entry / "failed_replacement_package"
            os.replace(target, failed_target)
            verify_prior_for_recovery(archived_package)
            os.replace(archived_package, target)
            shutil.rmtree(history_entry)
        else:
            _write_promotion_state(history_entry, target, "complete")
        return
    if not target.exists() and archived_package.exists():
        verify_prior_for_recovery(archived_package)
        os.replace(archived_package, target)
        shutil.rmtree(history_entry)
        return
    if target.exists() and not archived_package.exists() and state == "prepared":
        shutil.rmtree(history_entry)
        return
    raise ReportingError(
        "Interrupted release promotion cannot be recovered automatically"
    )


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
}


def _verify_qc_summary_scope(manifest: Mapping[str, Any]) -> None:
    """Bind the QC gate's validate-only enforcement scope for pipeline packages.

    ``_enforce_phase_two_qc_gate`` only blocks in validate mode; a test/full
    package preserves review-bearing rows instead. These fields make that
    boundary machine-verifiable so a tampered manifest cannot misrepresent
    which mode's gate ran.
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
    if (
        not isinstance(record_sha256, str)
        or checksums.get(record_relative) != record_sha256
    ):
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
        or record.get("record_id") != f"technical-{stable_json_sha256(binding)[:24]}"
    ):
        raise ReportingError("Release replacement record binding is invalid")
    target_name = record.get("target_name")
    if (
        not isinstance(target_name, str)
        or not target_name
        or Path(target_name).name != target_name
    ):
        raise ReportingError("Release replacement target name is invalid")
    is_stage_target = target.name.startswith(f".{target_name}.stage-")
    is_live_target = target.name == target_name
    is_preserved_prior = (
        target.name == "package"
        and len(target.parents) >= 4
        and target.parent.parent.name == target_name
        and target.parent.parent.parent.name == ".release_history"
        and re.fullmatch(
            rf"{sha256_file(target / 'run_manifest.json')[:16]}-technical-[0-9a-f]{{24}}",
            target.parent.name,
        )
        is not None
    )
    if not is_live_target and not is_stage_target and not is_preserved_prior:
        raise ReportingError(
            "Release replacement target does not match the package path"
        )
    release_root = target.parents[3] if is_preserved_prior else target.parent
    final_target = release_root / target_name
    if record.get("target_path_sha256") != _target_path_sha256(final_target):
        raise ReportingError("Release replacement target identity is invalid")
    record_id = str(record["record_id"])
    prior_manifest_sha256 = record.get("prior_manifest_sha256")
    if (
        not isinstance(prior_manifest_sha256, str)
        or _SHA256.fullmatch(prior_manifest_sha256) is None
    ):
        raise ReportingError("Release replacement prior manifest SHA-256 is invalid")
    expected_preserved_path = (
        Path(".release_history")
        / target_name
        / f"{prior_manifest_sha256[:16]}-{record_id}"
        / "package"
    )
    if replacement.get("preserved_prior_path") != expected_preserved_path.as_posix():
        raise ReportingError("Release replacement preservation path is invalid")
    prior_package = (
        final_target if is_stage_target else release_root / expected_preserved_path
    )
    if prior_package.resolve() == target.resolve() or not prior_package.is_dir():
        raise ReportingError("Release replacement prior package is missing")
    prior_manifest = prior_package / "run_manifest.json"
    if (
        not prior_manifest.is_file()
        or sha256_file(prior_manifest) != prior_manifest_sha256
    ):
        raise ReportingError("Release replacement prior manifest identity is invalid")
    try:
        _verify_prior_release_package(prior_package)
    except ReportingError as exc:
        raise ReportingError(
            "Release replacement prior package verification failed"
        ) from exc


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
        if not isinstance(table_name, str) or not _SAFE_TABLE_NAME.fullmatch(
            table_name
        ):
            raise ReportingError(
                "Release table inventory contains an unsafe table name"
            )
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
            raise ReportingError(
                f"Release table {table_name!r} column inventory is invalid"
            )
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
            raise ReportingError(
                f"Release table {table_name!r} artifact inventory is invalid"
            )
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
            if (
                len(readback) != row_count
                or list(map(str, readback.columns)) != column_names
            ):
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
        raise ReportingError(
            "Release review issue ledger schema version is unsupported"
        )

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
            raise ReportingError(
                "Release review issue ledger contains an invalid issue record"
            )
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
            or any(
                not isinstance(reason, str) or not reason.strip()
                for reason in reason_codes
            )
            or (
                detail is not None
                and (not isinstance(detail, str) or not detail.strip())
            )
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
            raise ReportingError(
                "Release review issue ledger contains an invalid issue record"
            )
        expected_issue_uid = (
            "review_issue_"
            + stable_json_sha256(
                {field: issue.get(field) for field in semantic_issue_fields}
            )[:24]
        )
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
        raise ReportingError(
            "Release review issue ledger run or policy identity is misbound"
        )

    policy_id = ledger.get("policy_id")
    prospective_version = ledger.get("prospective_effective_version")
    review_policy_sha256 = ledger.get("review_gate_policy_sha256")
    if schema_version == "ops-03-review-gate-v1":
        if (
            policy_id != "OPS-03-option-b"
            or prospective_version is not None
            or review_policy_sha256 is not None
        ):
            raise ReportingError(
                "Release review issue ledger v1 policy binding is invalid"
            )
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
            raise ReportingError(
                "Release review issue ledger v2 policy binding is invalid"
            )

    manifest_projection = {
        key: value for key, value in ledger.items() if key != "issues"
    }
    if dict(manifest_gate) != manifest_projection:
        raise ReportingError(
            "Release review issue ledger does not agree with the run manifest"
        )


def _figure_formats(
    manifest: Mapping[str, Any], figures: Mapping[str, Any]
) -> tuple[str, ...]:
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
        raise ReportingError(
            f"Release {name} figure-directory token inventory is invalid"
        )
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
            raise ReportingError(
                f"Release {name} figure-directory token inventory is invalid"
            )
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
    if detected_format != expected_format or any(
        dimension < 1 for dimension in dimensions
    ):
        raise ReportingError(
            f"Release figure failed image read-back validation: {path.name}"
        )


def _declared_overlay_paths(
    value: Any,
    *,
    field: str,
    suffixes: frozenset[str],
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or value != sorted(value)
        or len(value) != len(set(value))
        or any(
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or Path(relative).parts[:2] != ("figures", "overlay")
            or len(Path(relative).parts) != 3
            or Path(relative).suffix.casefold() not in suffixes
            for relative in value
        )
    ):
        raise ReportingError(
            f"Release governed overlay diagnostic {field} inventory is invalid"
        )
    return tuple(value)


def _read_governed_overlay_manifest(path: Path, *, name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportingError(
            f"Release governed overlay {name} manifest is unreadable"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ReportingError(
            f"Release governed overlay {name} manifest must be a JSON object"
        )
    return payload


def _verify_nested_overlay_ledger(
    overlay_root: Path,
    ledger_path: Path,
    *,
    expected_names: set[str],
) -> None:
    entries = _read_release_checksum_ledger(overlay_root, ledger_path)
    if set(entries) != expected_names or any(
        Path(name).name != name for name in entries
    ):
        raise ReportingError(
            "Release governed overlay nested checksum inventory is invalid"
        )
    for name, digest in entries.items():
        path = overlay_root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ReportingError(
                f"Release governed overlay nested checksum mismatch: {name}"
            )


def _verify_governed_diagnostic_raster(
    overlay_root: Path,
    output: Mapping[str, Any],
) -> None:
    expected = {
        "format": "JPEG",
        "width": 1500,
        "height": 1050,
        "mode": "RGB",
    }
    if any(output.get(name) != value for name, value in expected.items()):
        raise ReportingError(
            "Release governed overlay diagnostic raster metadata is invalid"
        )
    path = overlay_root / str(output.get("path"))
    try:
        with Image.open(path) as image:
            image.load()
            actual = {
                "format": image.format,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
            }
    except (OSError, ValueError) as exc:
        raise ReportingError(
            "Release governed overlay diagnostic raster is unreadable"
        ) from exc
    if actual != expected:
        raise ReportingError(
            "Release governed overlay diagnostic raster metadata is invalid"
        )


def _governed_overlay_configuration(
    declaration: Mapping[str, Any],
    *,
    has_zero_n_bundle: bool,
) -> tuple[float, float, float]:
    configuration = declaration.get("configuration")
    expected_keys = {
        "yield_threshold_t_ha",
        "high_n_threshold_kg_ha",
        "generate_zero_n_strata",
        "zero_n_yield_threshold_t_ha",
    }
    if not isinstance(configuration, Mapping) or set(configuration) != expected_keys:
        raise ReportingError(
            "Release governed overlay diagnostic configuration is invalid"
        )

    def finite_positive(name: str) -> float:
        value = configuration.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReportingError(
                "Release governed overlay diagnostic configuration is invalid"
            )
        normalized = float(value)
        if not math.isfinite(normalized) or normalized <= 0:
            raise ReportingError(
                "Release governed overlay diagnostic configuration is invalid"
            )
        return normalized

    generate_zero_n = configuration.get("generate_zero_n_strata")
    if not isinstance(generate_zero_n, bool) or generate_zero_n != has_zero_n_bundle:
        raise ReportingError(
            "Release governed overlay diagnostic configuration is invalid"
        )
    return (
        finite_positive("yield_threshold_t_ha"),
        finite_positive("high_n_threshold_kg_ha"),
        finite_positive("zero_n_yield_threshold_t_ha"),
    )


def _read_governed_overlay_observations(
    target: Path,
    checksums: Mapping[str, str],
    *,
    source_name: str,
    governed_uids: Sequence[str],
) -> dict[str, tuple[tuple[float, float], ...]]:
    ledger_relative = "tables/quality/analysis_eligibility_ledger.csv"
    ledger_path = target / ledger_relative
    if checksums.get(ledger_relative) != sha256_file(ledger_path):
        raise ReportingError(
            "Release governed overlay diagnostic analysis ledger is not checksum-bound"
        )
    governed_uid_set = set(governed_uids)
    observations: dict[str, list[tuple[float, float]]] = {
        uid: [] for uid in governed_uids
    }
    try:
        with ledger_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required_fields = {
                "source_name",
                "response_series_uid",
                "n_rate_kg_ha",
                "yield_t_ha",
            }
            if reader.fieldnames is None or not required_fields.issubset(
                reader.fieldnames
            ):
                raise ReportingError(
                    "Release governed overlay diagnostic analysis ledger schema is invalid"
                )
            for row in reader:
                uid = str(row.get("response_series_uid") or "").strip()
                if row.get("source_name") != source_name or uid not in governed_uid_set:
                    continue
                try:
                    n_rate = float(row["n_rate_kg_ha"])
                    yield_value = float(row["yield_t_ha"])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(n_rate) and math.isfinite(yield_value):
                    observations[uid].append((n_rate, yield_value))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReportingError(
            "Release governed overlay diagnostic analysis ledger is unreadable"
        ) from exc
    if any(not values for values in observations.values()):
        raise ReportingError(
            "Release governed overlay diagnostic analysis ledger lacks governed observations"
        )
    return {uid: tuple(sorted(values)) for uid, values in observations.items()}


def _verify_governed_high_yield_selection(
    plots: Mapping[str, Any],
    observations_by_series: Mapping[str, Sequence[tuple[float, float]]],
    *,
    source_token: str,
    yield_threshold_t_ha: float,
    high_n_threshold_kg_ha: float,
) -> None:
    from n_response_curve.reporting.generate_custom_overlays import (
        select_series_with_any_yield_above,
    )

    threshold_token = (
        format(yield_threshold_t_ha, "g").replace("-", "minus_").replace(".", "_")
    )
    plot = plots.get(f"any_observed_yield_above_{threshold_token}")
    candidates = [
        value
        for value in plots.values()
        if isinstance(value, Mapping) and "yield_threshold_t_ha" in value
    ]
    if not isinstance(plot, Mapping) or candidates != [plot]:
        raise ReportingError("Release governed overlay diagnostic selection is invalid")
    selection = select_series_with_any_yield_above(
        observations_by_series,
        yield_threshold_t_ha=yield_threshold_t_ha,
        high_n_threshold_kg_ha=high_n_threshold_kg_ha,
    )
    expected_above_high_n = [
        {
            "response_series_uid": uid,
            "n_rate_kg_ha": n_rate,
            "yield_t_ha": yield_value,
        }
        for uid, n_rate, yield_value in selection.above_high_n_observations
    ]
    expected_maxima = {
        uid: max(yield_value for _, yield_value in observations_by_series[uid])
        for uid in selection.response_series_uids
    }
    expected_semantics = (
        "Whole series with at least one finite observed grain yield strictly above "
        f"{yield_threshold_t_ha:g} t/ha; all finite observations retained, including "
        f"values at or below the threshold and N rates beyond "
        f"{high_n_threshold_kg_ha:g} kg/ha."
    )
    if (
        plot.get("yield_threshold_t_ha") != yield_threshold_t_ha
        or plot.get("high_n_threshold_kg_ha") != high_n_threshold_kg_ha
        or plot.get("qualifying_response_series_uids")
        != list(selection.response_series_uids)
        or plot.get("series_count") != len(selection.response_series_uids)
        or plot.get("series_max_yield_t_ha") != expected_maxima
        or plot.get("finite_observation_count") != len(selection.selected_observations)
        or plot.get("threshold_exceeding_observation_count")
        != len(selection.threshold_exceeding_observations)
        or plot.get("series_with_max_exactly_threshold_count")
        != len(selection.exact_threshold_response_series_uids)
        or plot.get("series_with_max_exactly_threshold_uids")
        != list(selection.exact_threshold_response_series_uids)
        or plot.get("n_rate_range_kg_ha") != list(selection.n_rate_range_kg_ha or ())
        or plot.get("above_high_n_observation_count")
        != len(selection.above_high_n_observations)
        or plot.get("above_high_n_observations") != expected_above_high_n
        or plot.get("selection_semantics") != expected_semantics
        or not isinstance(plot.get("output"), Mapping)
        or plot["output"].get("path")
        != f"{source_token}_series_with_yield_above_{threshold_token}_t_ha.jpeg"
    ):
        raise ReportingError(
            "Release governed overlay diagnostic selection is inconsistent with the "
            "release-safe analysis ledger"
        )


def _verify_governed_zero_n_classification(
    custom_plots: Mapping[str, Any],
    zero_manifest: Mapping[str, Any],
    observations_by_series: Mapping[str, Sequence[tuple[float, float]]],
    *,
    governed_uids: Sequence[str],
    source_name: str,
    source_token: str,
    zero_n_yield_threshold_t_ha: float,
    high_n_threshold_kg_ha: float,
) -> None:
    from n_response_curve.reporting.generate_custom_overlays import (
        classify_zero_n_strata,
    )

    classification = classify_zero_n_strata(
        observations_by_series,
        threshold_t_ha=zero_n_yield_threshold_t_ha,
        high_n_threshold_kg_ha=high_n_threshold_kg_ha,
    )
    if (
        classification.conflicting_baseline_response_series_uids
        or not classification.below.response_series_uids
        or not classification.above.response_series_uids
    ):
        raise ReportingError(
            "Release governed overlay zero-N classification is not publishable"
        )

    threshold_token = (
        format(zero_n_yield_threshold_t_ha, "g")
        .replace("-", "minus_")
        .replace(".", "_")
    )
    below_key = f"below_{threshold_token}"
    above_key = f"above_{threshold_token}"
    zero_plot_keys = {
        "below": f"zero_n_baseline_{below_key}",
        "above": f"zero_n_baseline_{above_key}",
    }
    summaries = {
        "below": classification.below,
        "above": classification.above,
    }

    def require_fields(
        actual: object,
        expected: Mapping[str, Any],
    ) -> None:
        if not isinstance(actual, Mapping) or any(
            actual.get(name) != value for name, value in expected.items()
        ):
            raise ReportingError(
                "Release governed overlay zero-N classification is inconsistent with "
                "the release-safe analysis ledger"
            )

    zero_groups = zero_manifest.get("groups")
    if not isinstance(zero_groups, Mapping) or set(zero_groups) != {
        below_key,
        above_key,
    }:
        raise ReportingError(
            "Release governed overlay zero-N classification is invalid"
        )
    for direction, key, operator in (
        ("below", below_key, "<"),
        ("above", above_key, ">"),
    ):
        summary = summaries[direction]
        criterion = (
            f"observed yield {operator} {zero_n_yield_threshold_t_ha:g} t/ha "
            "at exactly 0 kg N/ha"
        )
        require_fields(
            zero_groups.get(key),
            {
                "criterion": criterion,
                "response_series_uids": list(summary.response_series_uids),
                "series_count": len(summary.response_series_uids),
                "finite_observation_count": summary.finite_observation_count,
                "n_rate_range_kg_ha": list(summary.n_rate_range_kg_ha or ()),
                "baseline_yield_t_ha_by_series": summary.baseline_yield_t_ha_by_series,
                "baseline_yield_range_t_ha": list(
                    summary.baseline_yield_range_t_ha or ()
                ),
                "above_high_n_observation_count": len(
                    summary.above_high_n_observations
                ),
                "above_high_n_response_series_uids": list(
                    summary.above_high_n_response_series_uids
                ),
            },
        )
        require_fields(
            custom_plots.get(zero_plot_keys[direction]),
            {
                "selection_semantics": (
                    f"Whole series with observed yield {operator} "
                    f"{zero_n_yield_threshold_t_ha:g} t/ha at exactly 0 kg N/ha; "
                    "all finite observations retained."
                ),
                "response_series_uids": list(summary.response_series_uids),
                "series_count": len(summary.response_series_uids),
                "finite_observation_count": summary.finite_observation_count,
            },
        )
        zero_group = zero_groups.get(key)
        custom_plot = custom_plots.get(zero_plot_keys[direction])
        expected_output_name = (
            f"{source_token}_zero_n_{direction}_{threshold_token}_t_ha.jpeg"
        )
        if (
            not isinstance(zero_group, Mapping)
            or not isinstance(custom_plot, Mapping)
            or not isinstance(zero_group.get("output"), Mapping)
            or zero_group.get("output") != custom_plot.get("output")
            or zero_group["output"].get("path") != expected_output_name
        ):
            raise ReportingError(
                "Release governed overlay zero-N output mapping is inconsistent"
            )

    combined_above_high_n = sorted(
        (
            *(
                {
                    "group": below_key,
                    "response_series_uid": uid,
                    "n_rate_kg_ha": n_rate,
                    "yield_t_ha": yield_value,
                }
                for uid, n_rate, yield_value in classification.below.above_high_n_observations
            ),
            *(
                {
                    "group": above_key,
                    "response_series_uid": uid,
                    "n_rate_kg_ha": n_rate,
                    "yield_t_ha": yield_value,
                }
                for uid, n_rate, yield_value in classification.above.above_high_n_observations
            ),
        ),
        key=lambda observation: (
            observation["response_series_uid"],
            observation["n_rate_kg_ha"],
        ),
    )
    require_fields(
        zero_manifest.get("classification"),
        {
            "source_name": source_name,
            "baseline_n_rate_kg_ha": 0.0,
            "yield_threshold_t_ha": zero_n_yield_threshold_t_ha,
            "high_n_threshold_kg_ha": high_n_threshold_kg_ha,
            "comparison_semantics": (
                "Classify a whole governed response series using one distinct finite "
                "yield observed at exactly 0 kg N/ha, then plot all finite observations "
                "from qualifying series."
            ),
            "governed_series_count": classification.governed_series_count,
            "governed_finite_observation_count": (
                classification.governed_finite_observation_count
            ),
            "classified_series_count": (
                len(classification.below.response_series_uids)
                + len(classification.above.response_series_uids)
            ),
            "excluded_missing_zero_n_series_count": len(
                classification.missing_baseline_response_series_uids
            ),
            "excluded_missing_zero_n_response_series_uids": list(
                classification.missing_baseline_response_series_uids
            ),
            "excluded_equal_threshold_series_count": len(
                classification.equal_threshold_response_series_uids
            ),
            "excluded_equal_threshold_response_series_uids": list(
                classification.equal_threshold_response_series_uids
            ),
            "conflicting_zero_n_series_count": len(
                classification.conflicting_baseline_response_series_uids
            ),
            "conflicting_zero_n_response_series_uids": list(
                classification.conflicting_baseline_response_series_uids
            ),
            "series_inventory_sha256": hashlib.sha256(
                "\n".join(governed_uids).encode("utf-8")
            ).hexdigest(),
            "above_high_n_inclusion": {
                "criterion": f"n_rate_kg_ha > {high_n_threshold_kg_ha:g}",
                "disposition": (
                    "included_with_full_trajectory_in_qualifying_baseline_group"
                ),
                "maximum_n_rate_kg_ha": (
                    max(item["n_rate_kg_ha"] for item in combined_above_high_n)
                    if combined_above_high_n
                    else None
                ),
                "observation_count": len(combined_above_high_n),
                "series_count": len(
                    {item["response_series_uid"] for item in combined_above_high_n}
                ),
                "observations": combined_above_high_n,
            },
        },
    )


def _verify_governed_diagnostic_provenance(
    custom_manifest: Mapping[str, Any],
    zero_manifest: Mapping[str, Any] | None,
    checksums: Mapping[str, str],
    *,
    governed_uids: Sequence[str],
    source_token: str,
) -> None:
    inputs = custom_manifest.get("input_sha256")
    provenance = custom_manifest.get("source_provenance")
    if not isinstance(inputs, Mapping) or not isinstance(provenance, Mapping):
        raise ReportingError(
            "Release governed overlay diagnostic provenance is invalid"
        )
    expected_input_keys = {
        "scriptCONFIG.toml",
        "modules/n_response_curve/reporting/generate_custom_overlays.py",
        "modules/n_response_curve/reporting/plots.py",
        "release_package/tables/quality/analysis_eligibility_ledger.csv",
        f"release_package/figures/overlay/{source_token}.jpeg",
    }
    if set(inputs) != expected_input_keys or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in inputs.values()
    ):
        raise ReportingError(
            "Release governed overlay diagnostic provenance is invalid"
        )

    ledger_relative = "tables/quality/analysis_eligibility_ledger.csv"
    canonical_relative = f"figures/overlay/{source_token}.jpeg"
    series_inventory_sha256 = hashlib.sha256(
        "\n".join(governed_uids).encode("utf-8")
    ).hexdigest()
    expected_links = {
        "analysis_eligibility_ledger_sha256": checksums.get(ledger_relative),
        "series_inventory_sha256": series_inventory_sha256,
        "current_custom_plot_config_sha256": inputs["scriptCONFIG.toml"],
        "generator_module_sha256": inputs[
            "modules/n_response_curve/reporting/generate_custom_overlays.py"
        ],
    }
    zero_n_link_names = {
        "zero_n_strata_manifest_sha256",
        "zero_n_strata_checksum_ledger_sha256",
    }
    zero_n_links_valid = (
        all(provenance.get(name) is None for name in zero_n_link_names)
        if zero_manifest is None
        else all(
            isinstance(provenance.get(name), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(provenance.get(name))) is not None
            for name in zero_n_link_names
        )
    )
    if (
        inputs[f"release_package/{ledger_relative}"] != checksums.get(ledger_relative)
        or inputs[f"release_package/{canonical_relative}"]
        != checksums.get(canonical_relative)
        or any(provenance.get(name) != value for name, value in expected_links.items())
        or not zero_n_links_valid
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for name, value in provenance.items()
            if name.endswith("_sha256")
            and not (
                zero_manifest is None and name in zero_n_link_names and value is None
            )
        )
    ):
        raise ReportingError(
            "Release governed overlay diagnostic provenance does not reconcile with "
            "the root checksum-bound inputs"
        )
    if zero_manifest is not None and zero_manifest.get("input_sha256") != inputs:
        raise ReportingError(
            "Release governed overlay diagnostic provenance differs between sidecars"
        )


def _verify_governed_overlay_diagnostics(
    target: Path,
    checksums: Mapping[str, str],
    manifest: Mapping[str, Any],
    figures: Mapping[str, Any],
    source_tokens: Mapping[str, str],
    *,
    is_v3_layout: bool,
) -> tuple[frozenset[str], frozenset[str], str | None, frozenset[str]]:
    declaration = figures.get("diagnostics")
    if declaration is None:
        return frozenset(), frozenset(), None, frozenset()
    if (
        not is_v3_layout
        or not isinstance(declaration, Mapping)
        or declaration.get("schema_version") != "governed-overlay-diagnostics-v1"
        or declaration.get("status") != "governed_release_artifacts"
        or declaration.get("scope")
        != "configured_descriptive_subsets_of_governed_observed_series"
        or declaration.get("directory") != "figures/overlay"
        or declaration.get("generation_status")
        != "generated_governed_release_artifacts"
        or declaration.get("accountable_human_review") != "not_claimed"
    ):
        raise ReportingError(
            "Release governed overlay diagnostic declaration is invalid"
        )
    source_name = declaration.get("source_name")
    if not isinstance(source_name, str) or source_name not in source_tokens:
        raise ReportingError("Release governed overlay diagnostic source is invalid")
    raw_formats = declaration.get("formats")
    if raw_formats != ["jpeg"]:
        raise ReportingError("Release governed overlay diagnostic formats are invalid")
    diagnostic_formats = frozenset({"jpeg"})
    figure_paths = _declared_overlay_paths(
        declaration.get("figure_paths"),
        field="figure",
        suffixes=frozenset({".jpeg"}),
    )
    manifest_paths = _declared_overlay_paths(
        declaration.get("manifest_paths"),
        field="manifest",
        suffixes=frozenset({".json"}),
    )
    ledger_paths = _declared_overlay_paths(
        declaration.get("checksum_ledger_paths"),
        field="checksum-ledger",
        suffixes=frozenset({".sha256"}),
    )
    all_paths = frozenset((*figure_paths, *manifest_paths, *ledger_paths))
    if (
        declaration.get("figure_count") != len(figure_paths)
        or declaration.get("artifact_count") != len(all_paths)
        or len(figure_paths) not in {1, 3}
        or len(manifest_paths) != (2 if len(figure_paths) == 3 else 1)
        or len(ledger_paths) != (2 if len(figure_paths) == 3 else 1)
        or not all_paths.issubset(checksums)
    ):
        raise ReportingError(
            "Release governed overlay diagnostic counts do not reconcile"
        )
    expected_names = {
        "custom_plots_manifest.json",
        "CUSTOM_PLOTS_CHECKSUMS.sha256",
    }
    if len(figure_paths) == 3:
        expected_names.update(
            {
                "zero_n_strata_diagnostic_manifest.json",
                "ZERO_N_STRATA_DIAGNOSTIC_CHECKSUMS.sha256",
            }
        )
    if {
        Path(relative).name for relative in (*manifest_paths, *ledger_paths)
    } != expected_names:
        raise ReportingError(
            "Release governed overlay diagnostic sidecar inventory is invalid"
        )
    (
        yield_threshold_t_ha,
        high_n_threshold_kg_ha,
        zero_n_yield_threshold_t_ha,
    ) = _governed_overlay_configuration(
        declaration,
        has_zero_n_bundle=len(figure_paths) == 3,
    )
    effective_config = manifest.get("effective_config")
    if isinstance(effective_config, Mapping):
        configured = effective_config.get("custom_overlays")
        if not isinstance(configured, Mapping) or any(
            configured.get(name) != expected
            for name, expected in {
                "enabled": True,
                "source_name": source_name,
                "yield_threshold_t_ha": yield_threshold_t_ha,
                "high_n_threshold_kg_ha": high_n_threshold_kg_ha,
                "generate_zero_n_strata": len(figure_paths) == 3,
                "zero_n_yield_threshold_t_ha": zero_n_yield_threshold_t_ha,
            }.items()
        ):
            raise ReportingError(
                "Release governed overlay diagnostic configuration disagrees with "
                "the effective release configuration"
            )

    overlay_declaration = figures.get("overlay")
    governed_uids = (
        overlay_declaration.get("series_uids_by_source", {}).get(source_name)
        if isinstance(overlay_declaration, Mapping)
        else None
    )
    if (
        not isinstance(governed_uids, list)
        or not governed_uids
        or governed_uids != sorted(governed_uids)
        or len(governed_uids) != len(set(governed_uids))
        or any(not isinstance(uid, str) or not uid for uid in governed_uids)
    ):
        raise ReportingError(
            "Release governed overlay diagnostics lack a governed series inventory"
        )
    governed_uid_set = set(governed_uids)
    observations_by_series = _read_governed_overlay_observations(
        target,
        checksums,
        source_name=source_name,
        governed_uids=governed_uids,
    )
    overlay_root = target / "figures" / "overlay"
    custom_manifest_path = overlay_root / "custom_plots_manifest.json"
    custom_manifest = _read_governed_overlay_manifest(
        custom_manifest_path,
        name="custom-plot",
    )
    plots = custom_manifest.get("plots")
    if (
        custom_manifest.get("schema_version") != "governed-custom-plots-manifest-v1"
        or custom_manifest.get("status") != "governed_release_artifacts"
        or custom_manifest.get("source_run_id") != manifest.get("run_id")
        or custom_manifest.get("source_name") != source_name
        or custom_manifest.get("accountable_human_review") != "not_claimed"
        or not isinstance(plots, Mapping)
        or custom_manifest.get("custom_plot_count") != len(plots)
        or len(plots) != len(figure_paths)
    ):
        raise ReportingError("Release governed custom-plot manifest is invalid")
    _verify_governed_high_yield_selection(
        plots,
        observations_by_series,
        source_token=source_tokens[source_name],
        yield_threshold_t_ha=yield_threshold_t_ha,
        high_n_threshold_kg_ha=high_n_threshold_kg_ha,
    )
    figure_names = {Path(relative).name for relative in figure_paths}
    observed_output_names: set[str] = set()
    for plot in plots.values():
        if not isinstance(plot, Mapping):
            raise ReportingError("Release governed custom-plot entry is invalid")
        output = plot.get("output")
        uids = plot.get(
            "qualifying_response_series_uids",
            plot.get("response_series_uids"),
        )
        if (
            not isinstance(output, Mapping)
            or output.get("path") not in figure_names
            or output.get("path") in observed_output_names
            or output.get("sha256")
            != sha256_file(overlay_root / str(output.get("path")))
            or not isinstance(uids, list)
            or not uids
            or len(uids) != len(set(uids))
            or not set(uids).issubset(governed_uid_set)
            or plot.get("series_count") != len(uids)
        ):
            raise ReportingError("Release governed custom-plot entry is invalid")
        _verify_governed_diagnostic_raster(overlay_root, output)
        observed_output_names.add(str(output["path"]))
    if observed_output_names != figure_names:
        raise ReportingError("Release governed custom-plot outputs do not reconcile")
    custom_ledger_path = overlay_root / "CUSTOM_PLOTS_CHECKSUMS.sha256"
    _verify_nested_overlay_ledger(
        overlay_root,
        custom_ledger_path,
        expected_names={*figure_names, custom_manifest_path.name},
    )

    zero_manifest: Mapping[str, Any] | None = None
    if len(figure_paths) == 3:
        zero_manifest_path = overlay_root / "zero_n_strata_diagnostic_manifest.json"
        zero_ledger_path = overlay_root / "ZERO_N_STRATA_DIAGNOSTIC_CHECKSUMS.sha256"
        zero_manifest = _read_governed_overlay_manifest(
            zero_manifest_path,
            name="zero-N",
        )
        groups = zero_manifest.get("groups")
        provenance = custom_manifest.get("source_provenance")
        if (
            zero_manifest.get("schema_version")
            != "governed-zero-n-strata-diagnostic-v1"
            or zero_manifest.get("status") != "governed_release_artifacts"
            or zero_manifest.get("source_run_id") != manifest.get("run_id")
            or zero_manifest.get("source_name") != source_name
            or not isinstance(groups, Mapping)
            or len(groups) != 2
            or not isinstance(provenance, Mapping)
            or provenance.get("zero_n_strata_manifest_sha256")
            != sha256_file(zero_manifest_path)
            or provenance.get("zero_n_strata_checksum_ledger_sha256")
            != sha256_file(zero_ledger_path)
        ):
            raise ReportingError("Release governed zero-N manifest is invalid")
        _verify_governed_zero_n_classification(
            plots,
            zero_manifest,
            observations_by_series,
            governed_uids=governed_uids,
            source_name=source_name,
            source_token=source_tokens[source_name],
            zero_n_yield_threshold_t_ha=zero_n_yield_threshold_t_ha,
            high_n_threshold_kg_ha=high_n_threshold_kg_ha,
        )
        zero_names: set[str] = set()
        for group in groups.values():
            if not isinstance(group, Mapping):
                raise ReportingError("Release governed zero-N group is invalid")
            output = group.get("output")
            uids = group.get("response_series_uids")
            if (
                not isinstance(output, Mapping)
                or output.get("path") not in figure_names
                or output.get("path") in zero_names
                or output.get("sha256")
                != sha256_file(overlay_root / str(output.get("path")))
                or not isinstance(uids, list)
                or not uids
                or len(uids) != len(set(uids))
                or not set(uids).issubset(governed_uid_set)
                or group.get("series_count") != len(uids)
            ):
                raise ReportingError("Release governed zero-N group is invalid")
            zero_names.add(str(output["path"]))
        _verify_nested_overlay_ledger(
            overlay_root,
            zero_ledger_path,
            expected_names={*zero_names, zero_manifest_path.name},
        )

    _verify_governed_diagnostic_provenance(
        custom_manifest,
        zero_manifest,
        checksums,
        governed_uids=governed_uids,
        source_token=source_tokens[source_name],
    )

    return (
        frozenset(figure_paths),
        frozenset((*manifest_paths, *ledger_paths)),
        source_name,
        diagnostic_formats,
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
    is_v2_layout = layout_version == _FIGURE_LAYOUT_VERSION_V2
    is_v3_layout = layout_version == _FIGURE_LAYOUT_VERSION_V3
    expected_observed_path_template = (
        None if is_v3_layout else "figures/observed/<source>/<series>.<format>"
    )
    if (
        (is_v3_layout and "observed_path_template" not in figures)
        or figures.get("observed_path_template") != expected_observed_path_template
        or figures.get("fitted_path_template")
        != "figures/fitted/<source>/<model>/<series>__<attempt>.<format>"
    ):
        raise ReportingError("Release figure hierarchy metadata is invalid")
    overlay_declaration = figures.get("overlay")
    if is_v2_layout:
        if (
            not isinstance(overlay_declaration, Mapping)
            or overlay_declaration.get("scope") != _FIGURE_OVERLAY_SCOPE
            or overlay_declaration.get("path_template") != _FIGURE_OVERLAY_PATH_TEMPLATE
        ):
            raise ReportingError("Release figure overlay hierarchy metadata is invalid")
    elif is_v3_layout:
        if (
            not isinstance(overlay_declaration, Mapping)
            or overlay_declaration.get("scope") != _FIGURE_OVERLAY_SCOPE_V3
            or overlay_declaration.get("path_template") != _FIGURE_OVERLAY_PATH_TEMPLATE
        ):
            raise ReportingError("Release figure overlay hierarchy metadata is invalid")
    elif overlay_declaration is not None:
        raise ReportingError("Legacy figure layout must not declare a source overlay")

    runtime_policy = manifest.get("runtime_policy")
    effective_enablement = (
        runtime_policy.get("effective_enablement")
        if isinstance(runtime_policy, Mapping)
        else None
    )
    if isinstance(effective_enablement, Mapping):
        enabled_sources = effective_enablement.get("enabled_sources")
        enabled_models = effective_enablement.get("enabled_models")
        if isinstance(enabled_sources, list) and set(source_tokens) != set(
            enabled_sources
        ):
            raise ReportingError(
                "Release figure source hierarchy disagrees with runtime policy"
            )
        if isinstance(enabled_models, list) and set(model_tokens) != set(
            enabled_models
        ):
            raise ReportingError(
                "Release figure model hierarchy disagrees with runtime policy"
            )

    (
        diagnostic_figure_paths,
        diagnostic_support_paths,
        diagnostic_source_name,
        diagnostic_formats,
    ) = _verify_governed_overlay_diagnostics(
        target,
        checksums,
        manifest,
        figures,
        source_tokens,
        is_v3_layout=is_v3_layout,
    )
    figure_paths = sorted(
        relative
        for relative in checksums
        if relative.startswith("figures/") and relative not in diagnostic_support_paths
    )
    logical_formats: dict[str, list[str]] = {}
    logical_metadata: dict[str, tuple[str, str, str | None]] = {}
    for relative in figure_paths:
        relative_path = Path(relative)
        output_format = relative_path.suffix.casefold().lstrip(".")
        expected_formats = (
            diagnostic_formats
            if relative in diagnostic_figure_paths
            else frozenset(formats)
        )
        if output_format not in expected_formats:
            raise ReportingError(
                "Release figure hierarchy contains an undeclared artifact format"
            )
        _verify_raster_figure(target / relative, output_format=output_format)
        parts = relative_path.parts
        if relative in diagnostic_figure_paths:
            source_name = diagnostic_source_name
            model_name = None
            kind = "diagnostic"
        elif (
            not is_v3_layout
            and len(parts) == 4
            and parts[:2] == ("figures", "observed")
        ):
            source_name = source_by_token.get(parts[2])
            model_name = None
            kind = "observed"
        elif len(parts) == 5 and parts[:2] == ("figures", "fitted"):
            source_name = source_by_token.get(parts[2])
            model_name = model_by_token.get(parts[3])
            kind = "fitted"
        elif (
            (is_v2_layout or is_v3_layout)
            and len(parts) == 3
            and parts[:2] == ("figures", "overlay")
        ):
            source_name = source_by_token.get(relative_path.stem)
            model_name = None
            kind = "overlay"
        else:
            raise ReportingError(
                "Release figure artifact is outside the declared hierarchy"
            )
        if (
            source_name is None
            or (kind == "fitted" and model_name is None)
            or not relative_path.stem
        ):
            raise ReportingError(
                "Release figure artifact is outside the declared hierarchy"
            )
        logical_path = relative_path.with_suffix("").as_posix()
        logical_formats.setdefault(logical_path, []).append(output_format)
        logical_metadata[logical_path] = (kind, source_name, model_name)
    if any(
        len(observed)
        != len(
            diagnostic_formats
            if logical_metadata[logical_path][0] == "diagnostic"
            else formats
        )
        or set(observed)
        != set(
            diagnostic_formats
            if logical_metadata[logical_path][0] == "diagnostic"
            else formats
        )
        for logical_path, observed in logical_formats.items()
    ):
        raise ReportingError(
            "Release figure artifact formats are incomplete or inconsistent"
        )

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
        elif kind == "overlay":
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
        raise ReportingError(
            "Release figure fitted-by-source/model count inventory is invalid"
        )
    fitted_matrix = {
        source: _integer_count_map(counts, where="fitted-by-source/model")
        for source, counts in raw_fitted_matrix.items()
        if isinstance(source, str)
    }
    if len(fitted_matrix) != len(raw_fitted_matrix):
        raise ReportingError(
            "Release figure fitted-by-source/model count inventory is invalid"
        )
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
        raise ReportingError(
            "Release figure counts do not reconcile with the artifact hierarchy"
        )

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
    elif is_v3_layout:
        raw_series_uids_by_source = overlay_declaration.get("series_uids_by_source")
        if not isinstance(raw_series_uids_by_source, Mapping) or set(
            raw_series_uids_by_source
        ) != set(source_tokens):
            raise ReportingError(
                "Release figure overlay series-uid inventory is invalid"
            )
        series_uids_by_source: dict[str, tuple[str, ...]] = {}
        uid_owners: dict[str, str] = {}
        for source, raw_uids in raw_series_uids_by_source.items():
            if not isinstance(raw_uids, list) or any(
                not isinstance(uid, str) or not uid.strip() or uid != uid.strip()
                for uid in raw_uids
            ):
                raise ReportingError(
                    "Release figure overlay series-uid inventory is invalid"
                )
            if len(raw_uids) != len(set(raw_uids)):
                raise ReportingError(
                    "Release figure overlay series-uid inventory contains a duplicate"
                )
            if raw_uids != sorted(raw_uids):
                raise ReportingError(
                    "Release figure overlay series-uid inventory is not sorted"
                )
            for uid in raw_uids:
                if uid in uid_owners:
                    raise ReportingError(
                        "Release figure overlay series-uid is assigned to multiple sources"
                    )
                uid_owners[uid] = source
            series_uids_by_source[source] = tuple(raw_uids)

        expected_overlay_figure_count_by_source = {
            source: 1 if series_uids_by_source[source] else 0
            for source in sorted(source_tokens)
        }
        expected_overlay_sources = {
            source
            for source, count in expected_overlay_figure_count_by_source.items()
            if count == 1
        }
        if overlay_sources != expected_overlay_sources:
            raise ReportingError(
                "Release figure source overlay is missing or unexpected for a governed source"
            )
        expected_series_counts_by_source = {
            source: len(series_uids_by_source[source])
            for source in sorted(source_tokens)
        }
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
            or overlay_declaration.get("series_count")
            != sum(expected_series_counts_by_source.values())
            or overlay_counts_by_source != expected_overlay_figure_count_by_source
            or overlay_series_counts_by_source != expected_series_counts_by_source
            or any(value not in (0, 1) for value in overlay_counts_by_source.values())
        ):
            raise ReportingError(
                "Release figure overlay counts do not reconcile with the artifact hierarchy"
            )
    status = figures.get("status")
    if status == "run_output" and not figure_paths:
        raise ReportingError("Release run-output figure inventory is empty")
    if status == "no_renderable_series" and figure_paths:
        raise ReportingError(
            "Release no-renderable-series figure inventory contains artifacts"
        )


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
        raise ReportingError(
            "Synthetic sample figures are only valid in a test package"
        )
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
            raise ReportingError(
                "Synthetic sample figure artifact inventory is invalid"
            )
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
        raise ReportingError(
            "Synthetic sample figure formats are incomplete or inconsistent"
        )
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
        raise ReportingError(
            "Release run manifest figure inventory has an invalid status"
        )


def _read_release_checksum_ledger(target: Path, checksums_path: Path) -> dict[str, str]:
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
    return expected


def _reject_symlinked_package_path(target_path: str | Path) -> None:
    """Reject a package root reached through any existing symlink component."""

    absolute = Path(os.path.abspath(os.fspath(target_path)))
    for candidate in (absolute, *absolute.parents):
        try:
            info = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ReportingError(
                f"Could not inspect release package path component: {candidate}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise ReportingError(
                f"Release package path must not traverse a symlink: {candidate}"
            )


def _release_package_inventory(
    target: Path, checksums_path: Path
) -> tuple[set[str], set[str]]:
    """Inventory a package with ``lstat`` semantics, refusing every unsafe entry.

    A checksum ledger can only constrain the bytes of paths it names, so the
    inventory itself has to be closed: recursion never follows a link, symlinks
    and every non-regular/non-directory entry (FIFO, socket, device) are fatal,
    and a regular file carrying a second hard link is refused whatever it points
    at -- a shared inode means the ledgered bytes can be rewritten from outside
    the package. ``CHECKSUMS.sha256`` is stat'd like any other file even though
    it is excluded from the returned file set.

    Returns ``(files, directories)`` as package-relative POSIX paths, with
    ``CHECKSUMS.sha256`` omitted from ``files`` for the callers' ledger
    comparisons.
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
            raise ReportingError(
                f"Release package directory could not be read: {directory}"
            ) from exc
        for entry in listing:
            relative = f"{prefix}{entry.name}"
            try:
                info = os.lstat(entry.path)
            except OSError as exc:
                raise ReportingError(
                    f"Release package entry could not be inspected: {relative}"
                ) from exc
            mode = info.st_mode
            if stat.S_ISLNK(mode):
                raise ReportingError(f"Release package contains a symlink: {relative}")
            if stat.S_ISDIR(mode):
                directories.add(relative)
                pending.append((Path(entry.path), f"{relative}/"))
                continue
            if not stat.S_ISREG(mode):
                raise ReportingError(
                    f"Release package contains a non-regular filesystem entry: {relative}"
                )
            if info.st_nlink != 1:
                raise ReportingError(
                    "Release package file is a hard link and is not an independent "
                    f"copy: {relative}"
                )
            if Path(entry.path) != checksums_path:
                files.add(relative)
    return files, directories


def _release_package_actual_paths(target: Path, checksums_path: Path) -> set[str]:
    return _release_package_inventory(target, checksums_path)[0]


def _declared_directory_closure(relatives: Iterable[str]) -> set[str]:
    """Every directory a declared file set requires, and nothing else."""

    closure: set[str] = set()
    for relative in relatives:
        parts = relative.split("/")[:-1]
        for depth in range(1, len(parts) + 1):
            closure.add("/".join(parts[:depth]))
    return closure


def _verify_release_package_with_allowed_extras(
    target_path: str | Path, *, allowed_extra_paths: frozenset[str]
) -> ReleasePackage:
    """Verify a release with an already-validated set of historical extras."""

    _reject_symlinked_package_path(target_path)
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
    expected = _read_release_checksum_ledger(target, checksums_path)
    actual_paths, actual_directories = _release_package_inventory(
        target, checksums_path
    )
    if any(
        Path(relative).suffix.casefold() in _FORBIDDEN_ARTIFACT_SUFFIXES
        for relative in actual_paths
    ):
        raise ReportingError(
            "Release package contains a prohibited document or figure artifact"
        )
    if (actual_paths - allowed_extra_paths) != set(expected):
        raise ReportingError(
            "Release checksum ledger does not cover the complete package"
        )
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
    if not isinstance(output_profile, Mapping) or tuple(
        output_profile.get("document_formats", ())
    ) != ("md", "pdf"):
        raise ReportingError(
            "Release run manifest has an invalid report-document profile"
        )
    documents = payload.get("documents")
    if not isinstance(documents, Mapping) or set(documents) != {"markdown", "pdf"}:
        raise ReportingError(
            "Release run manifest has an invalid report-document inventory"
        )
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
            raise ReportingError(
                "Release run manifest report-document metadata is invalid"
            )
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
        raise ReportingError(
            "Markdown report failed UTF-8 read-back validation"
        ) from exc
    _validate_pdf(report_pdf_path)
    # Closes the inventory: the governed ledger plus the already-validated
    # allowed extras imply exactly one directory set, so a directory the declared
    # files cannot imply -- empty, or holding only further empty directories --
    # is undeclared state no ledger can describe. Deliberately last: a package
    # that also fails a governance or hierarchy contract must still report *that*
    # diagnosis, since removing a ledgered file routinely empties its directory.
    declared_directories = _declared_directory_closure(
        set(expected) | set(allowed_extra_paths) | {"CHECKSUMS.sha256"}
    )
    if actual_directories != declared_directories:
        raise ReportingError(
            "Release package contains a directory its checksum ledger does not "
            f"account for: {sorted(actual_directories ^ declared_directories)}"
        )
    return ReleasePackage(
        target_path=target,
        manifest_path=manifest_path,
        report_path=report_path,
        report_pdf_path=report_pdf_path,
        checksums_path=checksums_path,
        artifact_sha256=dict(sorted(expected.items())),
    )


def verify_release_package(target_path: str | Path) -> ReleasePackage:
    """Strictly verify every promoted artifact against its immutable ledger."""

    return _verify_release_package_with_allowed_extras(
        target_path,
        allowed_extra_paths=frozenset(),
    )


def _parse_diagnostic_checksum_ledger(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ReportingError(
            f"Prior-package diagnostic checksum ledger is missing: {path.name}"
        )
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, name = line.split("  ", 1)
        except ValueError as exc:
            raise ReportingError(
                f"Prior-package diagnostic checksum ledger {path.name} has an invalid line"
            ) from exc
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or Path(name).name != name
            or name in entries
        ):
            raise ReportingError(
                f"Prior-package diagnostic checksum ledger {path.name} is unsafe or malformed"
            )
        entries[name] = digest
    if not entries:
        raise ReportingError(
            f"Prior-package diagnostic checksum ledger is empty: {path.name}"
        )
    return entries


def _verify_original_release_integrity(
    block: Any,
    *,
    expected_bound_count: int,
    run_manifest_sha256: str,
    checksum_ledger_sha256: str,
    extra_fields: frozenset[str] = frozenset(),
) -> None:
    expected_fields = _ORIGINAL_RELEASE_INTEGRITY_BASE_FIELDS | extra_fields
    if not isinstance(block, Mapping) or set(block) != expected_fields:
        raise ReportingError(
            "Prior-package diagnostic original-release-integrity binding has an invalid schema"
        )
    if (
        block.get("bound_artifact_count") != expected_bound_count
        or block.get("bound_artifacts_reverified_unchanged") is not True
        or block.get("run_manifest_sha256") != run_manifest_sha256
        or block.get("release_checksum_ledger_sha256") != checksum_ledger_sha256
    ):
        raise ReportingError(
            "Prior-package diagnostic original-release-integrity binding does not match the bound package"
        )
    if extra_fields and (
        block.get("semantic_verifier_before_additions") != "not_claimed"
        or block.get("original_bound_files_unchanged_before_promotion") is not True
    ):
        raise ReportingError(
            "Prior-package zero-N diagnostic release-integrity flags are missing or invalid"
        )


def _is_safe_diagnostic_jpeg_name(name: str) -> bool:
    return (
        Path(name).name == name
        and name.casefold().endswith(".jpeg")
        and name
        not in {
            _CUSTOM_PLOTS_MANIFEST_NAME,
            _CUSTOM_PLOTS_CHECKSUMS_NAME,
            _ZERO_N_DIAGNOSTIC_MANIFEST_NAME,
            _ZERO_N_DIAGNOSTIC_CHECKSUMS_NAME,
        }
    )


def _verify_prior_diagnostic_overlay_bundle(
    target: Path,
    *,
    expected: Mapping[str, str],
    actual_paths: set[str],
    manifest: Mapping[str, Any],
) -> frozenset[str]:
    """Accept only an exact three- or seven-file post-release diagnostic bundle.

    Written by ``generate_custom_overlays.py`` atop an already-promoted,
    already-checksummed package: one yield-threshold overlay JPEG and its
    signed manifest/checksum ledger, optionally accompanied by two zero-N
    baseline JPEGs and their legacy signed manifest/checksum ledger. Returns
    the set of paths this bundle legitimately accounts for so the caller can
    re-run strict verification with exactly those paths ignored; raises on
    any unknown, missing, symlinked, unsafe, tampered, or misbound artifact.
    """

    overlay_dir = target / _PRIOR_DIAGNOSTIC_OVERLAY_DIR
    custom_ledger_relative = (
        f"{_PRIOR_DIAGNOSTIC_OVERLAY_DIR}/{_CUSTOM_PLOTS_CHECKSUMS_NAME}"
    )
    zero_ledger_relative = (
        f"{_PRIOR_DIAGNOSTIC_OVERLAY_DIR}/{_ZERO_N_DIAGNOSTIC_CHECKSUMS_NAME}"
    )
    custom_manifest_relative = (
        f"{_PRIOR_DIAGNOSTIC_OVERLAY_DIR}/{_CUSTOM_PLOTS_MANIFEST_NAME}"
    )
    zero_manifest_relative = (
        f"{_PRIOR_DIAGNOSTIC_OVERLAY_DIR}/{_ZERO_N_DIAGNOSTIC_MANIFEST_NAME}"
    )
    custom_bundle_relatives = {
        custom_ledger_relative,
        custom_manifest_relative,
    }
    zero_bundle_relatives = {
        zero_ledger_relative,
        zero_manifest_relative,
    }
    extras_actual = actual_paths - set(expected)
    if not custom_bundle_relatives <= extras_actual:
        raise ReportingError("Prior-package diagnostic overlay bundle is incomplete")
    observed_zero_bundle_relatives = zero_bundle_relatives & extras_actual
    if (
        observed_zero_bundle_relatives
        and observed_zero_bundle_relatives != zero_bundle_relatives
    ):
        raise ReportingError(
            "Prior-package zero-N diagnostic overlay bundle is incomplete"
        )
    has_zero_n_bundle = observed_zero_bundle_relatives == zero_bundle_relatives
    required_bundle_relatives = custom_bundle_relatives | (
        zero_bundle_relatives if has_zero_n_bundle else set()
    )
    for relative in required_bundle_relatives:
        if (target / relative).is_symlink():
            raise ReportingError(
                f"Prior-package diagnostic artifact is a symlink: {relative}"
            )

    custom_entries = _parse_diagnostic_checksum_ledger(
        overlay_dir / _CUSTOM_PLOTS_CHECKSUMS_NAME
    )
    zero_entries = (
        _parse_diagnostic_checksum_ledger(
            overlay_dir / _ZERO_N_DIAGNOSTIC_CHECKSUMS_NAME
        )
        if has_zero_n_bundle
        else {}
    )
    custom_jpeg_names = {
        name for name in custom_entries if name != _CUSTOM_PLOTS_MANIFEST_NAME
    }
    zero_jpeg_names = {
        name for name in zero_entries if name != _ZERO_N_DIAGNOSTIC_MANIFEST_NAME
    }
    expected_custom_entry_count = 4 if has_zero_n_bundle else 2
    expected_custom_jpeg_count = 3 if has_zero_n_bundle else 1
    if not has_zero_n_bundle and (
        len(custom_entries) != expected_custom_entry_count
        or len(custom_jpeg_names) != expected_custom_jpeg_count
    ):
        raise ReportingError("Prior-package diagnostic overlay bundle is incomplete")
    if (
        _CUSTOM_PLOTS_MANIFEST_NAME not in custom_entries
        or len(custom_entries) != expected_custom_entry_count
        or len(custom_jpeg_names) != expected_custom_jpeg_count
        or not all(_is_safe_diagnostic_jpeg_name(name) for name in custom_jpeg_names)
    ):
        raise ReportingError(
            "Prior-package custom-plots checksum ledger does not match the expected diagnostic bundle"
        )
    if has_zero_n_bundle:
        if (
            _ZERO_N_DIAGNOSTIC_MANIFEST_NAME not in zero_entries
            or len(zero_entries) != 3
            or len(zero_jpeg_names) != 2
            or not all(_is_safe_diagnostic_jpeg_name(name) for name in zero_jpeg_names)
        ):
            raise ReportingError(
                "Prior-package zero-N diagnostic checksum ledger does not match the expected diagnostic bundle"
            )
        if not zero_jpeg_names <= custom_jpeg_names:
            raise ReportingError(
                "Prior-package zero-N diagnostic images are not covered by the custom-plots checksum ledger"
            )
        for name in zero_jpeg_names:
            if custom_entries[name] != zero_entries[name]:
                raise ReportingError(
                    f"Prior-package diagnostic checksum ledgers disagree on {name}"
                )

    declared_names = {_CUSTOM_PLOTS_CHECKSUMS_NAME} | set(custom_entries)
    if has_zero_n_bundle:
        declared_names |= {_ZERO_N_DIAGNOSTIC_CHECKSUMS_NAME} | set(zero_entries)
    declared_relatives = frozenset(
        f"{_PRIOR_DIAGNOSTIC_OVERLAY_DIR}/{name}" for name in declared_names
    )
    if declared_relatives != extras_actual:
        raise ReportingError(
            "Prior-package diagnostic overlay bundle does not exactly account for the unbound package extras"
        )

    for name, digest in {**custom_entries, **zero_entries}.items():
        artifact_path = overlay_dir / name
        if artifact_path.is_symlink():
            raise ReportingError(
                f"Prior-package diagnostic artifact is a symlink: {name}"
            )
        if not artifact_path.is_file() or sha256_file(artifact_path) != digest:
            raise ReportingError(f"Prior-package diagnostic checksum mismatch: {name}")
    if (overlay_dir / _CUSTOM_PLOTS_CHECKSUMS_NAME).is_symlink() or (
        has_zero_n_bundle
        and (overlay_dir / _ZERO_N_DIAGNOSTIC_CHECKSUMS_NAME).is_symlink()
    ):
        raise ReportingError("Prior-package diagnostic checksum ledger is a symlink")

    run_manifest_sha256 = sha256_file(target / "run_manifest.json")
    checksum_ledger_sha256 = sha256_file(target / "CHECKSUMS.sha256")
    bound_artifact_count = len(expected)

    try:
        custom_manifest_payload = json.loads(
            (overlay_dir / _CUSTOM_PLOTS_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        zero_manifest_payload = (
            json.loads(
                (overlay_dir / _ZERO_N_DIAGNOSTIC_MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            if has_zero_n_bundle
            else None
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportingError("Prior-package diagnostic manifest is unreadable") from exc
    if not isinstance(custom_manifest_payload, Mapping) or (
        has_zero_n_bundle and not isinstance(zero_manifest_payload, Mapping)
    ):
        raise ReportingError("Prior-package diagnostic manifest must be a JSON object")

    if (
        custom_manifest_payload.get("schema_version")
        != _CUSTOM_PLOTS_MANIFEST_SCHEMA_VERSION
        or custom_manifest_payload.get("status") != _CUSTOM_PLOTS_MANIFEST_STATUS
        or custom_manifest_payload.get("source_run_id") != manifest.get("run_id")
    ):
        raise ReportingError(
            "Prior-package custom-plots manifest schema or binding is invalid"
        )
    if has_zero_n_bundle:
        assert isinstance(zero_manifest_payload, Mapping)
        if (
            zero_manifest_payload.get("schema_version")
            != _ZERO_N_DIAGNOSTIC_MANIFEST_SCHEMA_VERSION
            or zero_manifest_payload.get("status") != _ZERO_N_DIAGNOSTIC_MANIFEST_STATUS
            or zero_manifest_payload.get("source_run_id") != manifest.get("run_id")
            or zero_manifest_payload.get("source_package_status")
            != manifest.get("status")
        ):
            raise ReportingError(
                "Prior-package zero-N diagnostic manifest schema or binding is invalid"
            )

    _verify_original_release_integrity(
        custom_manifest_payload.get("original_release_integrity"),
        expected_bound_count=bound_artifact_count,
        run_manifest_sha256=run_manifest_sha256,
        checksum_ledger_sha256=checksum_ledger_sha256,
    )
    if has_zero_n_bundle:
        assert isinstance(zero_manifest_payload, Mapping)
        _verify_original_release_integrity(
            zero_manifest_payload.get("original_release_integrity"),
            expected_bound_count=bound_artifact_count,
            run_manifest_sha256=run_manifest_sha256,
            checksum_ledger_sha256=checksum_ledger_sha256,
            extra_fields=_ZERO_N_ORIGINAL_RELEASE_INTEGRITY_EXTRA_FIELDS,
        )

    plots = custom_manifest_payload.get("plots")
    if (
        not isinstance(plots, Mapping)
        or not plots
        or custom_manifest_payload.get("custom_plot_count") != len(plots)
    ):
        raise ReportingError(
            "Prior-package custom-plots manifest plot inventory is invalid"
        )
    plot_output_paths: list[str] = []
    for plot_key, plot in plots.items():
        output = plot.get("output") if isinstance(plot, Mapping) else None
        if not isinstance(output, Mapping):
            raise ReportingError(
                f"Prior-package custom-plots manifest plot {plot_key!r} output is invalid"
            )
        output_path = output.get("path")
        output_sha256 = output.get("sha256")
        if (
            not isinstance(output_path, str)
            or output_path not in custom_entries
            or custom_entries[output_path] != output_sha256
        ):
            raise ReportingError(
                f"Prior-package custom-plots manifest plot {plot_key!r} does not "
                "reconcile with its checksum ledger"
            )
        plot_output_paths.append(output_path)
    if (
        len(plot_output_paths) != len(set(plot_output_paths))
        or set(plot_output_paths) != custom_jpeg_names
    ):
        raise ReportingError(
            "Prior-package custom-plots manifest plot inventory does not cover its checksum ledger"
        )

    source_provenance = custom_manifest_payload.get("source_provenance")
    zero_n_provenance_fields = {
        "zero_n_strata_manifest_sha256",
        "zero_n_strata_checksum_ledger_sha256",
    }
    expected_zero_manifest_sha256 = (
        zero_entries[_ZERO_N_DIAGNOSTIC_MANIFEST_NAME] if has_zero_n_bundle else None
    )
    expected_zero_ledger_sha256 = (
        sha256_file(overlay_dir / _ZERO_N_DIAGNOSTIC_CHECKSUMS_NAME)
        if has_zero_n_bundle
        else None
    )
    if (
        not isinstance(source_provenance, Mapping)
        or not zero_n_provenance_fields <= set(source_provenance)
        or source_provenance.get("zero_n_strata_manifest_sha256")
        != expected_zero_manifest_sha256
        or source_provenance.get("zero_n_strata_checksum_ledger_sha256")
        != expected_zero_ledger_sha256
    ):
        raise ReportingError(
            "Prior-package custom-plots manifest does not bind the current zero-N diagnostic artifacts"
        )

    return declared_relatives


def _verify_prior_release_package(target_path: str | Path) -> ReleasePackage:
    """Fail-closed verification for a prior (historical) release package.

    Used only by technical-replacement, preserved-prior, and interrupted-
    promotion-recovery paths that must re-verify a package that was already
    promoted and may since have accumulated the signed post-release
    diagnostic overlay bundle or the exact host-bound restricted extension. A
    newly staged or newly promoted governed core is always verified with strict
    ``verify_release_package`` instead.
    """

    _reject_symlinked_package_path(target_path)
    target = Path(target_path).resolve()
    manifest_path = target / "run_manifest.json"
    checksums_path = target / "CHECKSUMS.sha256"
    if (
        not target.is_dir()
        or not manifest_path.is_file()
        or not (target / "report.md").is_file()
        or not (target / "report.pdf").is_file()
        or not checksums_path.is_file()
    ):
        raise ReportingError(f"Release package is incomplete: {target}")
    expected = _read_release_checksum_ledger(target, checksums_path)
    actual_paths = _release_package_actual_paths(target, checksums_path)
    if actual_paths == set(expected):
        return verify_release_package(target)
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportingError("Release run manifest is not valid JSON") from exc
    if not isinstance(manifest_payload, Mapping):
        raise ReportingError("Release run manifest must be a JSON object")
    restricted_prefix = "restricted_diagnostics/"
    observed_restricted_paths = frozenset(
        relative for relative in actual_paths if relative.startswith(restricted_prefix)
    )
    restricted_extra_paths: frozenset[str] = frozenset()
    if observed_restricted_paths:
        try:
            from n_response_curve.reporting.generate_source_dataset_overlays import (  # noqa: PLC0415
                recognize_restricted_report_extension,
            )

            restricted_extra_paths = recognize_restricted_report_extension(target)
        except RuntimeError as exc:
            raise ReportingError(
                "Completed release restricted extension is not a recognized closed bundle"
            ) from exc
        if restricted_extra_paths != observed_restricted_paths:
            raise ReportingError(
                "Completed release restricted extension inventory does not reconcile"
            )
    remaining_actual_paths = actual_paths - set(restricted_extra_paths)
    if remaining_actual_paths == set(expected):
        diagnostic_extra_paths: frozenset[str] = frozenset()
    else:
        diagnostic_extra_paths = _verify_prior_diagnostic_overlay_bundle(
            target,
            expected=expected,
            actual_paths=remaining_actual_paths,
            manifest=manifest_payload,
        )
    allow_extra_paths = diagnostic_extra_paths | restricted_extra_paths
    return _verify_release_package_with_allowed_extras(
        target,
        allowed_extra_paths=allow_extra_paths,
    )


def verify_completed_release_package(target_path: str | Path) -> ReleasePackage:
    """Verify a reusable completed release and any exact signed diagnostics.

    Unlike :func:`verify_release_package`, this verifier recognizes only the
    post-release custom-overlay bundles or the one host-bound restricted
    extension emitted by their respective generators. It validates complete
    inventories, checksums, manifests, and original-release bindings before
    tolerating them. Arbitrary extras remain fatal.
    """

    return _verify_prior_release_package(target_path)


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


def _register_stage_writer_artifacts(
    stage: Path,
    artifact_sha256: dict[str, str],
    writer: Callable[[Path], Iterable[str | Path]],
    *,
    writer_kind: str,
) -> None:
    """Validate and register every artifact returned by one staging writer."""

    if not callable(writer):
        raise ReportingError(f"Each {writer_kind} must be callable")
    written_paths = writer(stage)
    if written_paths is None:
        raise ReportingError(
            f"{writer_kind.capitalize()} must return its created artifact paths"
        )
    for written_path in written_paths:
        artifact_path = Path(written_path).resolve()
        if not artifact_path.is_relative_to(stage) or not artifact_path.is_file():
            raise ReportingError(
                f"{writer_kind.capitalize()} returned an artifact outside the staging "
                "package or not a file"
            )
        relative_path = artifact_path.relative_to(stage).as_posix()
        if (
            relative_path in _RESERVED_PACKAGE_PATHS
            or artifact_path.suffix.casefold() in _FORBIDDEN_ARTIFACT_SUFFIXES
        ):
            raise ReportingError(
                f"{writer_kind.capitalize()} returned a reserved or prohibited package artifact"
            )
        if relative_path in artifact_sha256:
            raise ReportingError(
                f"{writer_kind.capitalize()} artifact collides with an existing package "
                f"artifact: {relative_path}"
            )
        artifact_sha256[relative_path] = sha256_file(artifact_path)


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
    final_stage_writers: Sequence[Callable[[Path], Iterable[str | Path]]] = (),
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
        raise ReportingError(
            "Run manifest advertises a prohibited report-document profile"
        )
    _validate_analysis_population_selection(initial_manifest)
    formats = tuple(str(item).lower().lstrip(".") for item in output_formats)
    if (
        not formats
        or set(formats) - SUPPORTED_TABLE_FORMATS
        or len(formats) != len(set(formats))
    ):
        raise ReportingError(
            "Output formats must be unique members of csv, parquet, xlsx"
        )
    if target.exists() and not overwrite:
        raise ReportingError(
            f"Release package collision at {target}; set overwrite explicitly to replace it"
        )
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
            _register_stage_writer_artifacts(
                stage,
                artifact_sha256,
                writer,
                writer_kind="stage writer",
            )
        if replacement is not None:
            replacement_path = stage / "replacement_record.json"
            _write_json(replacement_path, replacement.record)
            artifact_sha256["replacement_record.json"] = sha256_file(replacement_path)
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
            raise ReportingError(
                "Staging package contains a prohibited document or figure artifact"
            )
        if staged_paths != set(artifact_sha256):
            raise ReportingError(
                "Stage artifact inventory does not account for every staged file"
            )
        for relative_path, digest in artifact_sha256.items():
            if sha256_file(stage / relative_path) != digest:
                raise ReportingError(
                    f"Stage artifact changed after registration: {relative_path}"
                )
        if release_validator is not None:
            if not callable(release_validator):
                raise ReportingError("Release validator must be callable")
            release_validator(manifest)
        report_path = stage / "report.md"
        report_text = _render_report(report_sections)
        report_path.write_text(report_text, encoding="utf-8")
        artifact_sha256[report_path.relative_to(stage).as_posix()] = sha256_file(
            report_path
        )
        report_pdf_path = stage / "report.pdf"
        report_pdf_path.write_bytes(_render_deterministic_pdf(report_text))
        _validate_pdf(report_pdf_path)
        artifact_sha256[report_pdf_path.relative_to(stage).as_posix()] = sha256_file(
            report_pdf_path
        )
        manifest_path = stage / "run_manifest.json"

        def current_manifest_payload() -> dict[str, Any]:
            payload = dict(_json_value(manifest))
            output_profile = payload.get("output_profile", {})
            if not isinstance(output_profile, Mapping):
                raise ReportingError("Run manifest output profile must be a mapping")
            output_profile = dict(output_profile)
            if "document_formats" in output_profile and tuple(
                output_profile["document_formats"]
            ) != ("md", "pdf"):
                raise ReportingError(
                    "Run manifest advertises a prohibited report-document profile"
                )
            output_profile["document_formats"] = ["md", "pdf"]
            payload["output_profile"] = output_profile
            payload["documents"] = {
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
                payload["replacement"] = {
                    "record_path": "replacement_record.json",
                    "record_sha256": artifact_sha256["replacement_record.json"],
                    "prior_manifest_sha256": replacement.prior_manifest_sha256,
                    "preserved_prior_path": (
                        replacement.history_entry.relative_to(target.parent) / "package"
                    ).as_posix(),
                }
            payload["tables"] = table_metadata
            payload["artifact_sha256_before_manifest"] = dict(
                sorted(
                    (relative, digest)
                    for relative, digest in artifact_sha256.items()
                    if relative != "run_manifest.json"
                )
            )
            return payload

        _write_json(manifest_path, current_manifest_payload())
        artifact_sha256["run_manifest.json"] = sha256_file(manifest_path)
        checksums_path = stage / "CHECKSUMS.sha256"
        checksums_path.write_text(
            "".join(
                f"{digest}  {relative}\n"
                for relative, digest in sorted(artifact_sha256.items())
            ),
            encoding="utf-8",
        )
        for writer in final_stage_writers:
            registered_before = dict(artifact_sha256)
            _register_stage_writer_artifacts(
                stage,
                artifact_sha256,
                writer,
                writer_kind="final stage writer",
            )
            for relative_path, digest in registered_before.items():
                if sha256_file(stage / relative_path) != digest:
                    raise ReportingError(
                        "Final stage writer changed a registered package artifact: "
                        f"{relative_path}"
                    )
        final_paths = {
            path.relative_to(stage).as_posix()
            for path in stage.rglob("*")
            if path.is_file() and path != checksums_path
        }
        if final_paths != set(artifact_sha256):
            raise ReportingError(
                "Final stage artifact inventory does not account for every staged file"
            )
        artifact_sha256.pop("run_manifest.json")
        _write_json(manifest_path, current_manifest_payload())
        artifact_sha256["run_manifest.json"] = sha256_file(manifest_path)
        if final_stage_writers and release_validator is not None:
            release_validator(manifest)
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
        checksums_path.write_text(
            "".join(
                f"{digest}  {relative}\n"
                for relative, digest in sorted(artifact_sha256.items())
            ),
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
                failed_replacement = (
                    preserved_prior.parent / "failed_replacement_package"
                )
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
