from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Iterable, Mapping

from .config import (
    ConfigError,
    KNOWN_DATA_CLASSIFICATIONS,
    KNOWN_SOURCE_ENCODINGS,
    ValidatedConfig,
)
from .provenance import (
    ChecksumRevisionApproval,
    SourceIntegrityReport,
    sha256_file,
    validate_checksum_revision_approval,
    verify_source_integrity,
)


SUPPORTED_SHAPE_ADAPTERS = frozenset({"core-trial-csv-v1", "fixture-csv-v1"})


@dataclass(frozen=True)
class SourceAdapterSpec:
    """Versioned physical-shape contract for exactly one source adapter."""

    version: str
    expected_physical_columns: int
    expected_headers: Mapping[int, str]
    map_version: str


@dataclass(frozen=True)
class RawColumn:
    """One physical input column, identified independently of its header text."""

    source_name: str
    position: int
    raw_column_id: str
    header: str


@dataclass(frozen=True)
class RawRow:
    """A logical CSV record with its logical and physical source location plus cells."""

    source_name: str
    source_path: Path
    source_sha256: str
    source_row_number: int
    source_physical_line_start: int
    source_physical_line_end: int
    raw_cells: tuple[str, ...]


@dataclass(frozen=True)
class IngestedSource:
    """Position-safe, non-mutating representation of one configured source CSV."""

    source_name: str
    source_path: Path
    source_sha256: str
    columns: tuple[RawColumn, ...]
    rows: tuple[RawRow, ...]
    blank_rows: tuple[RawRow, ...]
    source_type: str = "unknown"
    source_family: str = "unknown"
    source_country_code: str = "UNRESOLVED"
    shape_adapter_version: str = "direct-csv-v1"
    schema_map_path: Path | None = None
    schema_map_sha256: str | None = None
    schema_map_version: str | None = None
    source_encoding: str = "utf-8-sig"
    text_decoding_lineage: str = "decoded_from_registered_bytes"
    data_classification: str = "internal"
    workbook_csv_basis: str = "csv_registered_artifact"
    restricted_access_status: str = "not_restricted"
    source_revision_status: str = "registered_checksum"
    source_revision_comparison_sha256: str | None = None


@dataclass(frozen=True)
class IngestionResult:
    """All configured source results plus the input integrity evidence used to admit them."""

    sources: tuple[IngestedSource, ...]
    integrity_report: SourceIntegrityReport | None

    @property
    def rows(self) -> tuple[RawRow, ...]:
        return tuple(row for source in self.sources for row in source.rows)

    @property
    def blank_rows(self) -> tuple[RawRow, ...]:
        return tuple(row for source in self.sources for row in source.blank_rows)


def _is_wholly_blank(row: list[str]) -> bool:
    return not row or all(cell == "" for cell in row)


def _expected_headers_from_config(config: ValidatedConfig) -> dict[int, str]:
    fields = config.raw["schema"]["fields"]
    return {field["position"]: field["header"] for field in fields.values()}


def ingest_csv(
    source_path: str | Path,
    *,
    source_name: str,
    expected_physical_columns: int,
    expected_headers: Mapping[int, str] | None = None,
    expected_sha256: str | None = None,
    source_type: str = "unknown",
    source_family: str | None = None,
    source_country_code: str = "UNRESOLVED",
    shape_adapter_version: str = "direct-csv-v1",
    schema_map_path: str | Path | None = None,
    schema_map_sha256: str | None = None,
    schema_map_version: str | None = None,
    encoding: str = "utf-8-sig",
    data_classification: str = "internal",
    workbook_csv_basis: str = "csv_registered_artifact",
    checksum_revision_approval: ChecksumRevisionApproval | Mapping[str, object] | None = None,
    checksum_revision_artifact_path: str | None = None,
    designated_reviewers: Iterable[str] = (),
) -> IngestedSource:
    """Read one CSV by physical position and fail before accepting shape drift.

    Decoding uses only ``encoding``. A Unicode byte-order mark is removed only
    when the registered codec specifies that behavior. Stored bytes are never
    rewritten.
    """

    if not source_name.strip():
        raise ConfigError("Source name must be nonempty")
    if expected_physical_columns < 1:
        raise ConfigError("Expected physical column count must be at least one")
    if encoding not in KNOWN_SOURCE_ENCODINGS:
        raise ConfigError(f"Source encoding is not supported: {encoding!r}")
    if data_classification not in KNOWN_DATA_CLASSIFICATIONS:
        raise ConfigError(f"Source data classification is not supported: {data_classification!r}")

    path = Path(source_path).resolve()
    if not path.is_file():
        raise ConfigError(f"Source CSV does not exist: {path}")
    digest_before = sha256_file(path)
    revision_comparison_sha256: str | None = None
    source_revision_status = "registered_checksum"
    if expected_sha256 is not None and digest_before != expected_sha256:
        if checksum_revision_approval is None or checksum_revision_artifact_path is None:
            raise ConfigError(f"Source does not match its manifest-bound checksum: {path}")
        comparison = validate_checksum_revision_approval(
            checksum_revision_approval,
            candidate_path=path,
            expected_old_sha256=expected_sha256,
            artifact_path=checksum_revision_artifact_path,
            prior_encoding=encoding,
            candidate_encoding=encoding,
            designated_reviewers=designated_reviewers,
        )
        revision_comparison_sha256 = comparison.comparison_sha256
        source_revision_status = "reviewed_revision_pending_manifest_update"

    try:
        with path.open("r", encoding=encoding, newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ConfigError(f"Source CSV is empty: {path}") from exc
            except csv.Error as exc:
                raise ConfigError(f"Malformed CSV header in {path}: {exc}") from exc
            if len(header) != expected_physical_columns:
                raise ConfigError(
                    f"Source header physical column count is {len(header)}, expected {expected_physical_columns}: {path}"
                )
            for position, expected_header in (expected_headers or {}).items():
                if not isinstance(position, int) or position < 1 or position > len(header):
                    raise ConfigError(f"Configured header position is outside source shape: {position}")
                actual_header = header[position - 1]
                if actual_header != expected_header:
                    raise ConfigError(
                        f"Source header mismatch at physical position {position}: expected {expected_header!r}, got {actual_header!r}"
                    )

            rows: list[RawRow] = []
            blank_rows: list[RawRow] = []
            source_row_number = 2
            while True:
                physical_line_start = reader.line_num + 1
                try:
                    raw_row = next(reader)
                except StopIteration:
                    break
                except csv.Error as exc:
                    raise ConfigError(
                        f"Malformed CSV near physical line {physical_line_start} in {path}: {exc}"
                    ) from exc
                physical_line_end = reader.line_num
                raw = tuple(raw_row)
                row = RawRow(
                    source_name=source_name,
                    source_path=path,
                    source_sha256=digest_before,
                    source_row_number=source_row_number,
                    source_physical_line_start=physical_line_start,
                    source_physical_line_end=physical_line_end,
                    raw_cells=raw,
                )
                if _is_wholly_blank(raw_row):
                    blank_rows.append(row)
                elif len(raw_row) != expected_physical_columns:
                    raise ConfigError(
                        f"Source row {source_row_number} physical column count is {len(raw_row)}, expected {expected_physical_columns}: {path}"
                    )
                else:
                    rows.append(row)
                source_row_number += 1
    except (UnicodeDecodeError, LookupError) as exc:
        raise ConfigError(
            f"Source cannot be decoded using its registered encoding {encoding!r}: {path}"
        ) from exc

    digest_after = sha256_file(path)
    if digest_after != digest_before:
        raise ConfigError(f"Source bytes changed during ingestion: {path}")

    columns = tuple(
        RawColumn(
            source_name=source_name,
            position=position,
            raw_column_id=f"raw_col_{position:03d}",
            header=header_name,
        )
        for position, header_name in enumerate(header, start=1)
    )
    return IngestedSource(
        source_name=source_name,
        source_path=path,
        source_sha256=digest_before,
        columns=columns,
        rows=tuple(rows),
        blank_rows=tuple(blank_rows),
        source_type=source_type,
        source_family=source_family or source_type,
        source_country_code=source_country_code,
        shape_adapter_version=shape_adapter_version,
        schema_map_path=Path(schema_map_path).resolve() if schema_map_path is not None else None,
        schema_map_sha256=schema_map_sha256,
        schema_map_version=schema_map_version,
        source_encoding=encoding,
        data_classification=data_classification,
        workbook_csv_basis=workbook_csv_basis,
        restricted_access_status=(
            "restricted_controls_required"
            if data_classification == "restricted"
            else "not_restricted"
        ),
        source_revision_status=source_revision_status,
        source_revision_comparison_sha256=revision_comparison_sha256,
    )


def _configured_source_path(config: ValidatedConfig, source_name: str) -> Path:
    if source_name == "core_trial_data":
        return config.paths["core_source_csv"]
    raw_path = config.sources[source_name]["data_path"]
    return (config.project_root / raw_path).resolve()


def _manifest_relative_path(config: ValidatedConfig, source_path: Path) -> str:
    source_root = config.paths["source_manifest"].parent.resolve()
    try:
        return source_path.resolve().relative_to(source_root).as_posix()
    except ValueError as exc:
        raise ConfigError(
            f"Enabled source is outside the manifest intake root: {source_path}"
        ) from exc


def ingest_configured_sources(config: ValidatedConfig) -> IngestionResult:
    """Verify the intake package and read every enabled source without writing it."""

    for source_name in config.enabled_sources:
        adapter = str(config.sources[source_name]["shape_adapter_version"])
        if adapter not in SUPPORTED_SHAPE_ADAPTERS:
            raise ConfigError(
                f"Enabled source {source_name!r} declares unsupported shape adapter {adapter!r}"
            )

    integrity_report = verify_source_integrity(
        config.paths["source_manifest"],
        config.paths["source_checksums"],
    )
    if integrity_report.failed_files:
        raise ConfigError(
            "Configured input checksums do not match the intake package: "
            + ", ".join(integrity_report.failed_files)
        )

    schema = config.raw["schema"]
    expected_columns = schema["expected_physical_columns"]
    expected_headers = _expected_headers_from_config(config)
    sources: list[IngestedSource] = []
    for source_name in config.enabled_sources:
        source_path = _configured_source_path(config, source_name)
        source_config = config.sources[source_name]
        schema_map_path = (config.project_root / str(source_config["schema_map"])).resolve()
        if not schema_map_path.is_file():
            raise ConfigError(f"Schema map for enabled source does not exist: {schema_map_path}")
        manifest_relative_path = _manifest_relative_path(config, source_path)
        expected_sha256 = integrity_report.artifact_sha256.get(manifest_relative_path)
        if expected_sha256 is None:
            raise ConfigError(
                f"Enabled source is not registered in the manifest: {source_name} ({manifest_relative_path})"
            )
        sources.append(
            ingest_csv(
                source_path,
                source_name=source_name,
                expected_physical_columns=expected_columns,
                expected_headers=expected_headers,
                expected_sha256=expected_sha256,
                source_type=str(source_config["source_type"]),
                source_family=str(source_config["source_family"]),
                source_country_code=str(source_config["country_code"]),
                shape_adapter_version=str(source_config["shape_adapter_version"]),
                schema_map_path=schema_map_path,
                schema_map_sha256=sha256_file(schema_map_path),
            )
        )
    return IngestionResult(sources=tuple(sources), integrity_report=integrity_report)


__all__ = [
    "IngestedSource",
    "IngestionResult",
    "RawColumn",
    "RawRow",
    "SUPPORTED_SHAPE_ADAPTERS",
    "ingest_configured_sources",
    "ingest_csv",
]
