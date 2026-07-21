from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

from .ingest import IngestedSource, IngestionResult, RawRow
from .schema import classify_missing, classify_treatment, normalize_category, normalize_yield, parse_numeric


@dataclass(frozen=True)
class CurationResult:
    """Traceable canonical records constructed without changing raw source rows."""

    records: tuple[dict[str, Any], ...]


def _stable_uid(*parts: object) -> str:
    encoded = "\x00".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _header_positions(source: IngestedSource, header: str) -> tuple[int, ...]:
    return tuple(column.position for column in source.columns if column.header == header)


def _fill_down_positions(source: IngestedSource, fields: tuple[str, ...]) -> dict[int, str]:
    positions: dict[int, str] = {}
    for header in fields:
        matches = _header_positions(source, header)
        if len(matches) != 1:
            raise ValueError(
                f"Configured fill-down header {header!r} must occur exactly once in source {source.source_name}"
            )
        positions[matches[0]] = header
    return positions


def _field_raw_value(row: RawRow, position: int) -> str:
    return row.raw_cells[position - 1]


def _optional_field(record: Mapping[str, Any], name: str) -> str:
    value = record.get(name)
    return value if isinstance(value, str) else ""


def _curate_source(source: IngestedSource, config: Any) -> list[dict[str, Any]]:
    raw_config = config.raw
    schema = raw_config["schema"]
    schema_fields: Mapping[str, Mapping[str, Any]] = schema["fields"]
    missing_values = raw_config["missing_values"]
    fill_positions = _fill_down_positions(source, tuple(config.fill_down_fields))
    study_id_position = next(
        (position for position, header in fill_positions.items() if header == "Study_ID"),
        None,
    )
    blank_source_row_numbers = {row.source_row_number for row in source.blank_rows}
    fill_state: dict[int, tuple[str, int]] = {}
    source_uid = _stable_uid("source-v1", source.source_name, source.source_sha256)
    records: list[dict[str, Any]] = []
    previous_source_row_number = 1

    for raw_row in source.rows:
        if any(
            previous_source_row_number < blank_row_number < raw_row.source_row_number
            for blank_row_number in blank_source_row_numbers
        ):
            fill_state.clear()
        if (
            study_id_position is not None
            and classify_missing(_field_raw_value(raw_row, study_id_position), missing_values) != "blank"
        ):
            fill_state.clear()
        effective_by_position = {position: _field_raw_value(raw_row, position) for position in range(1, len(source.columns) + 1)}
        filled_by_position = {position: False for position in effective_by_position}
        fill_origin_by_position: dict[int, int | None] = {
            position: None for position in effective_by_position
        }
        for position in fill_positions:
            raw_value = effective_by_position[position]
            if classify_missing(raw_value, missing_values) == "blank" and position in fill_state:
                effective_by_position[position], fill_origin_by_position[position] = fill_state[position]
                filled_by_position[position] = True
            elif classify_missing(raw_value, missing_values) != "blank":
                fill_state[position] = (raw_value, raw_row.source_row_number)

        record: dict[str, Any] = {
            "source_name": source.source_name,
            "source_path": str(source.source_path),
            "source_sha256": source.source_sha256,
            "source_uid": source_uid,
            "record_uid": _stable_uid("record-v1", source_uid, raw_row.source_row_number),
            "source_row_number": raw_row.source_row_number,
            "source_physical_line_start": raw_row.source_physical_line_start,
            "source_physical_line_end": raw_row.source_physical_line_end,
            "raw_column_ids": tuple(column.raw_column_id for column in source.columns),
            "raw_headers": tuple(column.header for column in source.columns),
            "raw_cells": raw_row.raw_cells,
        }
        for canonical_name, field in schema_fields.items():
            position = field["position"]
            raw_value = _field_raw_value(raw_row, position)
            effective_value = effective_by_position[position]
            record[f"{canonical_name}_raw"] = raw_value
            record[canonical_name] = effective_value
            record[f"{canonical_name}_missing_state"] = classify_missing(raw_value, missing_values)
            record[f"{canonical_name}_filled_down"] = filled_by_position[position]
            record[f"{canonical_name}_filled_from_source_row_number"] = fill_origin_by_position[position]

        n_parse = parse_numeric(_optional_field(record, "inorganic_n_rate"), missing_values)
        p_parse = parse_numeric(_optional_field(record, "inorganic_p_rate"), missing_values)
        k_parse = parse_numeric(_optional_field(record, "inorganic_k_rate"), missing_values)
        yield_normalization = normalize_yield(
            _optional_field(record, "yield_kg_ha"),
            _optional_field(record, "yield_t_ha"),
            missing_values,
        )
        record.update(
            {
                "n_rate_kg_ha": n_parse.value,
                "n_rate_parse_status": n_parse.status,
                "p_rate_kg_p2o5_ha": p_parse.value,
                "p_rate_parse_status": p_parse.status,
                "k_rate_kg_k2o_ha": k_parse.value,
                "k_rate_parse_status": k_parse.status,
                "yield_t_ha": yield_normalization.yield_t_ha,
                "yield_parse_status": yield_normalization.parse_status,
                "yield_unit_status": yield_normalization.unit_status,
                "yield_source_unit": yield_normalization.source_unit,
                "water_regime_normalized": normalize_category(
                    _optional_field(record, "water_regime"),
                    schema["normalization"]["water_regime"],
                ),
                "season_normalized": normalize_category(
                    _optional_field(record, "season"),
                    schema["normalization"]["season"],
                ),
            }
        )
        record.update(
            classify_treatment(
                treatment_raw=_optional_field(record, "treatment"),
                n_rate=n_parse.value,
                p_rate=p_parse.value,
                k_rate=k_parse.value,
                organic_raw=_optional_field(record, "organic_fertilizer"),
                bio_raw=_optional_field(record, "biofertilizer"),
                treatment_mapping=schema["normalization"]["treatment_class"],
                missing_values=missing_values,
                high_n_threshold=raw_config["eligibility"]["high_n_review_threshold_kg_ha"],
            )
        )
        records.append(record)
        previous_source_row_number = raw_row.source_row_number
    return records


def curate_ingestion(ingestion: IngestionResult, config: Any) -> CurationResult:
    """Map configured raw positions to canonical values while preserving all raw cells."""

    records = tuple(record for source in ingestion.sources for record in _curate_source(source, config))
    record_uids = [record["record_uid"] for record in records]
    if len(record_uids) != len(set(record_uids)):
        raise ValueError("Canonical record identifiers must be unique")
    return CurationResult(records=records)


__all__ = ["CurationResult", "curate_ingestion"]
