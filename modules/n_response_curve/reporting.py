from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4

import pandas as pd


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
    checksums_path: Path
    artifact_sha256: Mapping[str, str]


_SAFE_TABLE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SUPPORTED_FORMATS = frozenset({"csv", "parquet", "xlsx"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        artifacts[destination.relative_to(stage_root).as_posix()] = _sha256(destination)
    metadata = {
        "row_count": len(rows),
        "stable_key": artifact.stable_key,
        "artifact_paths": sorted(artifacts),
    }
    return metadata, artifacts


def _render_report(report_sections: Mapping[str, Iterable[str]]) -> str:
    headings = (
        ("primary", "Primary findings"),
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
            "- Unsupported, sparse, aliased, or decision-gated analyses remain explicit non-findings rather than negative evidence.",
            "",
        )
    )
    return "\n".join(lines)


def _assert_safe_target(target: Path, source_roots: Iterable[str | Path]) -> None:
    for source_root in source_roots:
        normalized_source = Path(source_root).resolve()
        if target == normalized_source or target.is_relative_to(normalized_source):
            raise ReportingError(f"Refusing source target for release package: {target}")


def _promote_stage(stage: Path, target: Path, *, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        raise ReportingError(f"Release package collision at {target}; set overwrite explicitly to replace it")
    backup: Path | None = None
    try:
        if target.exists():
            backup = target.with_name(f".{target.name}.backup-{uuid4().hex}")
            os.replace(target, backup)
        os.replace(stage, target)
    except OSError as exc:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise ReportingError(f"Unable to atomically promote release package: {exc}") from exc
    finally:
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def verify_release_package(target_path: str | Path) -> ReleasePackage:
    """Verify every promoted artifact against its immutable checksum ledger."""

    target = Path(target_path).resolve()
    manifest_path = target / "run_manifest.json"
    report_path = target / "report.md"
    checksums_path = target / "CHECKSUMS.sha256"
    if not target.is_dir() or not manifest_path.is_file() or not report_path.is_file() or not checksums_path.is_file():
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
    if actual_paths != set(expected):
        raise ReportingError("Release checksum ledger does not cover the complete package")
    for relative, digest in expected.items():
        artifact_path = target / relative
        if not artifact_path.is_file() or _sha256(artifact_path) != digest:
            raise ReportingError(f"Release checksum mismatch: {relative}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReportingError("Release run manifest is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ReportingError("Release run manifest must be a JSON object")
    return ReleasePackage(
        target_path=target,
        manifest_path=manifest_path,
        report_path=report_path,
        checksums_path=checksums_path,
        artifact_sha256=dict(sorted(expected.items())),
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
) -> ReleasePackage:
    """Stage, validate, checksum, and atomically promote one auditable run package."""

    if not isinstance(manifest, Mapping):
        raise ReportingError("Run manifest input must be a mapping")
    target = Path(target_path).resolve()
    _assert_safe_target(target, source_roots)
    formats = tuple(str(item).lower().lstrip(".") for item in output_formats)
    if not formats or set(formats) - _SUPPORTED_FORMATS or len(formats) != len(set(formats)):
        raise ReportingError("Output formats must be unique members of csv, parquet, xlsx")
    if target.exists() and not overwrite:
        raise ReportingError(f"Release package collision at {target}; set overwrite explicitly to replace it")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    try:
        table_metadata: dict[str, dict[str, Any]] = {}
        artifact_sha256: dict[str, str] = {}
        for name, artifact in sorted(tables.items()):
            metadata, artifact_hashes = _write_table(stage, name, artifact, formats)
            table_metadata[name] = metadata
            artifact_sha256.update(artifact_hashes)
        report_path = stage / "report.md"
        report_path.write_text(_render_report(report_sections), encoding="utf-8")
        artifact_sha256[report_path.relative_to(stage).as_posix()] = _sha256(report_path)
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
                if relative_path in artifact_sha256:
                    raise ReportingError(f"Stage writer artifact collides with an existing package artifact: {relative_path}")
                artifact_sha256[relative_path] = _sha256(artifact_path)
        manifest_path = stage / "run_manifest.json"
        manifest_payload = dict(_json_value(manifest))
        manifest_payload["tables"] = table_metadata
        manifest_payload["artifact_sha256_before_manifest"] = dict(sorted(artifact_sha256.items()))
        _write_json(manifest_path, manifest_payload)
        artifact_sha256[manifest_path.relative_to(stage).as_posix()] = _sha256(manifest_path)
        checksums_path = stage / "CHECKSUMS.sha256"
        checksums_path.write_text(
            "".join(f"{digest}  {relative}\n" for relative, digest in sorted(artifact_sha256.items())),
            encoding="utf-8",
        )
        _promote_stage(stage, target, overwrite=overwrite)
        return ReleasePackage(
            target_path=target,
            manifest_path=target / "run_manifest.json",
            report_path=target / "report.md",
            checksums_path=target / "CHECKSUMS.sha256",
            artifact_sha256=dict(sorted(artifact_sha256.items())),
        )
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = ["ReleasePackage", "ReportingError", "TableArtifact", "verify_release_package", "write_release_package"]
