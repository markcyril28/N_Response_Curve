from __future__ import annotations

from dataclasses import dataclass
import hmac
import hashlib
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .ingest import (
    KNOWN_REPRESENTATION_BASES,
    IngestedSource,
    IngestionResult,
    RawRow,
)
from .schema import (
    CANONICAL_N_RATE_UNIT,
    CANONICAL_YIELD_UNIT,
    ReviewedLookupTable,
    canonicalize_irri,
    canonical_unit,
    classify_experiment_priority,
    classify_missing,
    classify_raw_state,
    classify_treatment,
    normalize_category,
    normalize_category_with_evidence,
    normalize_country_code,
    normalize_yield,
    parse_numeric,
    sensitive_path_alias,
    validate_reviewed_lookup_table,
)


_COLUMN_ROLES = frozenset(
    {"canonical", "descriptive", "held", "restricted", "source_metadata", "blank"}
)
REQUIRED_REVIEWED_LOOKUP_FIELDS = (
    "water_regime",
    "season",
    "treatment_class",
)
_KNOWN_TREATMENT_LOOKUP_CLASSES = frozenset(
    {"zero_n", "absolute_control", "RCM", "FP", "NOPT_NPK", "other", "unresolved"}
)
_REQUIRED_TREATMENT_LOOKUP_CLASSES = frozenset(
    {"zero_n", "absolute_control", "RCM", "FP", "NOPT_NPK"}
)


@dataclass(frozen=True)
class PhysicalColumnDisposition:
    """Reviewed analytical role for one physical source column."""

    position: int
    role: str
    canonical_field: str | None = None
    variable_family: str | None = None


@dataclass(frozen=True)
class SourceArmMap:
    """One reviewed long-form arm expanded from a physical parent row."""

    arm_id: str
    role: str
    field_positions: Mapping[str, int]
    constants: Mapping[str, str]


@dataclass(frozen=True)
class ReviewedSourceMap:
    """Versioned, source-bound physical mapping and full column disposition."""

    source_name: str
    map_version: str
    review_id: str
    source_sha256: str
    encoding: str
    workbook_csv_basis: str
    fields: Mapping[str, int]
    expected_headers: Mapping[int, str]
    dispositions: tuple[PhysicalColumnDisposition, ...]
    fill_down_headers: tuple[str, ...] = ()
    arms: tuple[SourceArmMap, ...] = ()
    normalization_map_version: str | None = None
    normalization_review_id: str | None = None


@dataclass(frozen=True)
class RestrictedDataPolicy:
    """Controls required before producing a public projection of restricted rows."""

    pseudonym_salt: bytes
    identifier_fields: tuple[str, ...]
    precise_location_fields: tuple[str, ...]
    detailed_location_fields: tuple[str, ...]
    approved_geography_fields: tuple[str, ...]
    public_release_fields: tuple[str, ...]
    access_review_id: str
    automated_disclosure_review_id: str
    human_disclosure_review_id: str


@dataclass(frozen=True)
class CurationResult:
    """Traceable canonical records constructed without changing raw source rows."""

    records: tuple[dict[str, Any], ...]
    parent_row_uids: tuple[str, ...] = ()


def _stable_uid(*parts: object) -> str:
    encoded = "\x00".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _controlled_pseudonym(value: str, *, salt: bytes) -> str:
    if len(salt) < 16:
        raise ValueError("Restricted-data pseudonym salt must contain at least 16 bytes")
    digest = hmac.new(salt, value.encode("utf-8"), hashlib.sha256).hexdigest()[:24]
    return f"subject_{digest}"


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


def _validate_reviewed_source_map(
    source: IngestedSource,
    source_map: ReviewedSourceMap,
) -> None:
    if not isinstance(source_map, ReviewedSourceMap):
        raise ValueError("Reviewed source map must use the expected source-map type")
    if source_map.source_name != source.source_name:
        raise ValueError("Reviewed source map is bound to a different source")
    for label, value in (
        ("map version", source_map.map_version),
        ("review evidence", source_map.review_id),
        ("workbook/CSV basis", source_map.workbook_csv_basis),
    ):
        if not value.strip():
            raise ValueError(f"Reviewed source map {label} must be nonempty")
    if source_map.source_sha256 != source.source_sha256:
        raise ValueError("Reviewed source map is not bound to the ingested source checksum")
    if source_map.encoding != source.source_encoding:
        raise ValueError("Reviewed source map encoding differs from the ingested source encoding")
    if source_map.workbook_csv_basis != source.workbook_csv_basis:
        raise ValueError(
            "Reviewed source map workbook/CSV basis differs from the ingested source basis"
        )
    positions = tuple(disposition.position for disposition in source_map.dispositions)
    expected_positions = tuple(range(1, len(source.columns) + 1))
    if len(positions) != len(set(positions)) or tuple(sorted(positions)) != expected_positions:
        raise ValueError("Reviewed source map must disposition every physical column exactly once")
    by_position = {item.position: item for item in source_map.dispositions}
    for disposition in source_map.dispositions:
        if disposition.role not in _COLUMN_ROLES:
            raise ValueError(
                f"Physical column {disposition.position} has an unknown disposition role"
            )
        if disposition.role == "canonical" and not disposition.canonical_field:
            raise ValueError(
                f"Canonical physical column {disposition.position} lacks a canonical field"
            )
        if disposition.role in {"descriptive", "held", "restricted"} and not disposition.variable_family:
            raise ValueError(
                f"Physical column {disposition.position} lacks a variable-family disposition"
            )
    mapped_positions: dict[int, str] = {}
    for canonical_name, position in source_map.fields.items():
        if not canonical_name.strip() or position not in by_position:
            raise ValueError("Reviewed source map contains an invalid canonical field mapping")
        prior = mapped_positions.setdefault(position, canonical_name)
        if prior != canonical_name:
            raise ValueError("Reviewed source map maps one position to multiple canonical fields")
        disposition = by_position[position]
        if (
            disposition.role not in {"canonical", "restricted"}
            or disposition.canonical_field != canonical_name
        ):
            raise ValueError(
                f"Canonical mapping {canonical_name!r} disagrees with its column disposition"
            )
    for position, expected_header in source_map.expected_headers.items():
        if position not in by_position:
            raise ValueError("Reviewed source map expected header is outside the physical shape")
        if source.columns[position - 1].header != expected_header:
            raise ValueError(
                f"Reviewed source map header mismatch at physical position {position}"
            )
    arm_ids: set[str] = set()
    for arm in source_map.arms:
        if not arm.arm_id.strip() or arm.arm_id in arm_ids:
            raise ValueError("Reviewed source arm identifiers must be nonempty and unique")
        arm_ids.add(arm.arm_id)
        if not arm.role.strip():
            raise ValueError("Reviewed source arms require an analytical role")
        for canonical_name, position in arm.field_positions.items():
            if not canonical_name.strip() or position not in by_position:
                raise ValueError("Reviewed source arm mapping is outside the physical shape")
        if set(arm.field_positions).intersection(arm.constants):
            raise ValueError("A source-arm field cannot be both positional and constant")


def validate_reviewed_curation_controls(
    ingestion: IngestionResult,
    *,
    source_maps: Mapping[str, ReviewedSourceMap],
    category_lookups: Mapping[str, ReviewedLookupTable],
    required_lookup_fields: Iterable[str] = REQUIRED_REVIEWED_LOOKUP_FIELDS,
) -> None:
    """Require complete reviewed source and normalization controls for an ingestion."""

    source_names = tuple(source.source_name for source in ingestion.sources)
    if len(source_names) != len(set(source_names)):
        raise ValueError("Ingested source names must be unique")
    expected_sources = set(source_names)
    observed_sources = set(source_maps)
    missing_sources = expected_sources - observed_sources
    unexpected_sources = observed_sources - expected_sources
    if missing_sources or unexpected_sources:
        missing = ", ".join(sorted(missing_sources)) or "none"
        unexpected = ", ".join(sorted(unexpected_sources)) or "none"
        raise ValueError(
            "Reviewed source-map coverage must exactly match ingested sources; "
            f"missing={missing}; unexpected={unexpected}"
        )

    normalized_required: list[str] = []
    for field in required_lookup_fields:
        if not isinstance(field, str) or not field.strip():
            raise ValueError("Required reviewed lookup fields must be nonempty strings")
        normalized_required.append(field.strip())
    if len(normalized_required) != len(set(normalized_required)):
        raise ValueError("Required reviewed lookup fields must be unique")

    missing_lookups = set(normalized_required) - set(category_lookups)
    if missing_lookups:
        raise ValueError(
            "Required reviewed category lookup(s) are missing: "
            + ", ".join(sorted(missing_lookups))
        )
    for field, lookup in category_lookups.items():
        if not isinstance(field, str) or not field.strip():
            raise ValueError("Reviewed category lookup field names must be nonempty")
        validate_reviewed_lookup_table(lookup)

    treatment_lookup = category_lookups.get("treatment_class")
    if "treatment_class" in normalized_required:
        if treatment_lookup is None:
            raise ValueError("Required reviewed treatment lookup is missing")
        treatment_classes = set(treatment_lookup.aliases)
        unknown_treatment_classes = (
            treatment_classes - _KNOWN_TREATMENT_LOOKUP_CLASSES
        )
        if unknown_treatment_classes:
            raise ValueError(
                "Reviewed treatment lookup contains unsupported canonical class(es): "
                + ", ".join(sorted(unknown_treatment_classes))
            )
        missing_treatment_classes = (
            _REQUIRED_TREATMENT_LOOKUP_CLASSES - treatment_classes
        )
        if missing_treatment_classes:
            raise ValueError(
                "Reviewed treatment lookup is missing required canonical class(es): "
                + ", ".join(sorted(missing_treatment_classes))
            )

    for source in ingestion.sources:
        _validate_reviewed_source_map(source, source_maps[source.source_name])


def _legacy_source_map(source: IngestedSource, config: Any) -> ReviewedSourceMap:
    schema = config.raw["schema"]
    schema_fields: Mapping[str, Mapping[str, Any]] = schema["fields"]
    mapped = {name: int(field["position"]) for name, field in schema_fields.items()}
    dispositions = tuple(
        PhysicalColumnDisposition(
            position=column.position,
            role="canonical" if column.position in mapped.values() else "held",
            canonical_field=next(
                (name for name, position in mapped.items() if position == column.position),
                None,
            ),
            variable_family=(
                None if column.position in mapped.values() else "unreviewed_unmapped"
            ),
        )
        for column in source.columns
    )
    return ReviewedSourceMap(
        source_name=source.source_name,
        map_version=source.schema_map_version or "legacy-unversioned",
        review_id="review-required",
        source_sha256=source.source_sha256,
        encoding=source.source_encoding,
        workbook_csv_basis=source.workbook_csv_basis,
        fields=MappingProxyType(mapped),
        expected_headers=MappingProxyType(
            {
                int(field["position"]): str(field["header"])
                for field in schema_fields.values()
            }
        ),
        dispositions=dispositions,
        fill_down_headers=tuple(config.fill_down_fields),
    )


def _source_arms(source_map: ReviewedSourceMap) -> tuple[SourceArmMap, ...]:
    if source_map.arms:
        return source_map.arms
    return (
        SourceArmMap(
            arm_id="source_row",
            role="canonical_source_row",
            field_positions=MappingProxyType({}),
            constants=MappingProxyType({}),
        ),
    )


def _reviewed_or_configured_category(
    record: Mapping[str, Any],
    *,
    field: str,
    configured_mapping: Mapping[str, list[str] | tuple[str, ...]],
    lookup: ReviewedLookupTable | None,
) -> tuple[str, str, str | None, str | None]:
    raw_value = _optional_field(record, field)
    if lookup is None:
        return (
            normalize_category(raw_value, configured_mapping),
            "review_required_unversioned",
            None,
            None,
        )
    normalized = normalize_category_with_evidence(raw_value, lookup)
    return (
        normalized.canonical_value or "unresolved",
        normalized.status,
        normalized.map_version,
        normalized.review_id,
    )


def _finalize_canonical_record(
    record: dict[str, Any],
    *,
    source: IngestedSource,
    config: Any,
    schema: Mapping[str, Any],
    missing_values: Mapping[str, Any],
    category_lookups: Mapping[str, ReviewedLookupTable],
    restricted_policy: RestrictedDataPolicy | None,
) -> dict[str, Any]:
    n_parse = parse_numeric(_optional_field(record, "inorganic_n_rate"), missing_values)
    p_parse = parse_numeric(_optional_field(record, "inorganic_p_rate"), missing_values)
    k_parse = parse_numeric(_optional_field(record, "inorganic_k_rate"), missing_values)
    recommended_n_parse = parse_numeric(
        _optional_field(record, "recommended_n_rate"),
        missing_values,
    )
    yield_se_parse = parse_numeric(
        _optional_field(record, "yield_se_t_ha"),
        missing_values,
    )
    yield_normalization = normalize_yield(
        _optional_field(record, "yield_kg_ha"),
        _optional_field(record, "yield_t_ha"),
        missing_values,
    )
    configured_units = schema.get("units", {})
    configured_n_unit = str(configured_units.get("n_rate", CANONICAL_N_RATE_UNIT))
    configured_yield_unit = str(
        configured_units.get("yield_curve", CANONICAL_YIELD_UNIT)
    )
    row_country_raw = _optional_field(record, "country_code") or _optional_field(
        record,
        "country",
    )
    if row_country_raw.strip():
        scope_country_code = normalize_country_code(row_country_raw)
        scope_country_evidence = "row"
    else:
        scope_country_code = normalize_country_code(source.source_country_code)
        scope_country_evidence = "source"
    scope_countries = set(getattr(config, "scope_countries", ("PH",)))
    if scope_country_code is None:
        scope_status = "unresolved"
    elif scope_country_code in scope_countries:
        scope_status = "in_scope"
    else:
        scope_status = "out_of_scope"

    water_value, water_status, water_map_version, water_review_id = (
        _reviewed_or_configured_category(
            record,
            field="water_regime",
            configured_mapping=schema["normalization"]["water_regime"],
            lookup=category_lookups.get("water_regime"),
        )
    )
    season_value, season_status, season_map_version, season_review_id = (
        _reviewed_or_configured_category(
            record,
            field="season",
            configured_mapping=schema["normalization"]["season"],
            lookup=category_lookups.get("season"),
        )
    )
    representation_reasons = list(yield_normalization.review_reasons)
    if canonical_unit(configured_n_unit, "n_rate") is None:
        representation_reasons.append("N_RATE_UNIT_UNSUPPORTED")
    if canonical_unit(configured_yield_unit, "yield") is None:
        representation_reasons.append("YIELD_UNIT_UNSUPPORTED")
    if record.get("workbook_csv_basis") == "parallel_workbook_csv_unresolved":
        representation_reasons.append("WORKBOOK_CSV_BASIS_UNRESOLVED")
    basis_raw = _optional_field(record, "yield_basis")
    moisture_basis_raw = _optional_field(record, "yield_moisture_basis")
    if (basis_raw or moisture_basis_raw) and yield_normalization.yield_t_ha is not None:
        representation_reasons.append("YIELD_BASIS_REVIEW_REQUIRED")

    study_id = _optional_field(record, "study_id")
    trial_id = _optional_field(record, "trial_id")
    site_id = _optional_field(record, "site_id") or _optional_field(record, "location")
    record.update(
        {
            "study_uid": (
                _stable_uid("study-v1", source.source_family, study_id)
                if study_id.strip()
                else None
            ),
            "trial_uid": (
                _stable_uid("trial-v1", source.source_family, study_id, trial_id)
                if study_id.strip() and trial_id.strip()
                else None
            ),
            "site_uid": (
                _stable_uid("site-v1", source.source_family, site_id)
                if site_id.strip()
                else None
            ),
            "n_rate_kg_ha": n_parse.value,
            "n_rate_parse_status": n_parse.status,
            "actual_n_rate_kg_ha": n_parse.value,
            "recommended_n_rate_kg_ha": recommended_n_parse.value,
            "recommended_n_rate_parse_status": recommended_n_parse.status,
            "n_rate_configured_unit": configured_n_unit,
            "n_rate_canonical_unit": CANONICAL_N_RATE_UNIT,
            "n_rate_unit_status": (
                "canonical" if canonical_unit(configured_n_unit, "n_rate") else "unsupported"
            ),
            "p_rate_kg_p2o5_ha": p_parse.value,
            "p_rate_parse_status": p_parse.status,
            "k_rate_kg_k2o_ha": k_parse.value,
            "k_rate_parse_status": k_parse.status,
            "yield_t_ha": yield_normalization.yield_t_ha,
            "yield_se_t_ha": yield_se_parse.value,
            "yield_se_parse_status": yield_se_parse.status,
            "yield_se_status": (
                record.get("yield_se_status")
                if yield_se_parse.value is not None
                else "unavailable"
            ),
            "yield_parse_status": yield_normalization.parse_status,
            "yield_unit_status": yield_normalization.unit_status,
            "yield_source_unit": yield_normalization.source_unit,
            "yield_unit_conversion": yield_normalization.conversion,
            "yield_configured_unit": configured_yield_unit,
            "yield_canonical_unit": CANONICAL_YIELD_UNIT,
            "yield_basis_raw": basis_raw or None,
            "yield_moisture_basis_raw": moisture_basis_raw or None,
            "representation_review_status": (
                "review_required" if representation_reasons else "resolved"
            ),
            "representation_review_reasons": tuple(sorted(set(representation_reasons))),
            "scope_country_code": scope_country_code,
            "scope_country_evidence": scope_country_evidence,
            "scope_status": scope_status,
            "experiment_priority_status": classify_experiment_priority(
                _optional_field(record, "experiment_type"),
                _optional_field(record, "experimental_design"),
            ),
            "water_regime_normalized": water_value,
            "water_regime_normalization_status": water_status,
            "water_regime_map_version": water_map_version,
            "water_regime_review_id": water_review_id,
            "season_normalized": season_value,
            "season_normalization_status": season_status,
            "season_map_version": season_map_version,
            "season_review_id": season_review_id,
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
            treatment_mapping=(
                category_lookups["treatment_class"].aliases
                if "treatment_class" in category_lookups
                else schema["normalization"]["treatment_class"]
            ),
            missing_values=missing_values,
            high_n_threshold=config.raw["eligibility"]["high_n_review_threshold_kg_ha"],
        )
    )
    treatment_lookup = category_lookups.get("treatment_class")
    record["treatment_class_map_version"] = (
        treatment_lookup.map_version if treatment_lookup is not None else None
    )
    record["treatment_class_review_id"] = (
        treatment_lookup.review_id if treatment_lookup is not None else None
    )
    for field, lookup in category_lookups.items():
        if field in {"water_regime", "season", "treatment_class"}:
            continue
        normalized = normalize_category_with_evidence(
            _optional_field(record, field),
            lookup,
        )
        record[f"{field}_normalized"] = normalized.canonical_value or "unresolved"
        record[f"{field}_normalization_status"] = normalized.status
        record[f"{field}_map_version"] = normalized.map_version
        record[f"{field}_review_id"] = normalized.review_id

    if source.data_classification != "restricted":
        record["restricted_release_status"] = "not_restricted"
        record["controlled_subject_uid"] = None
    elif restricted_policy is None:
        record["restricted_release_status"] = "blocked_controls_missing"
        record["controlled_subject_uid"] = None
    else:
        for label, review_id in (
            ("access review", restricted_policy.access_review_id),
            ("automated disclosure review", restricted_policy.automated_disclosure_review_id),
            ("human disclosure review", restricted_policy.human_disclosure_review_id),
        ):
            if not review_id.strip():
                raise ValueError(f"Restricted-data {label} evidence must be nonempty")
        identifier = "\x1f".join(
            _optional_field(record, field)
            for field in restricted_policy.identifier_fields
            if _optional_field(record, field).strip()
        )
        record["controlled_subject_uid"] = (
            _controlled_pseudonym(identifier, salt=restricted_policy.pseudonym_salt)
            if identifier
            else None
        )
        record["restricted_release_status"] = "eligible_for_reviewed_public_projection"
    return record


def _curate_source(
    source: IngestedSource,
    config: Any,
    *,
    source_map: ReviewedSourceMap | None,
    category_lookups: Mapping[str, ReviewedLookupTable],
    restricted_policy: RestrictedDataPolicy | None,
) -> list[dict[str, Any]]:
    raw_config = config.raw
    schema = raw_config["schema"]
    missing_values = raw_config["missing_values"]
    effective_map = source_map or _legacy_source_map(source, config)
    _validate_reviewed_source_map(source, effective_map)
    mapping_review_status = "reviewed" if source_map is not None else "review_required"
    fill_positions = _fill_down_positions(source, effective_map.fill_down_headers)
    schema_fields = effective_map.fields
    study_id_position = next(
        (position for position, header in fill_positions.items() if header == "Study_ID"),
        None,
    )
    blank_source_row_numbers = {row.source_row_number for row in source.blank_rows}
    fill_state: dict[int, tuple[str, int]] = {}
    source_uid = _stable_uid("source-v1", source.source_name, source.source_sha256)
    records: list[dict[str, Any]] = []
    previous_source_row_number = 1
    source_arms = _source_arms(effective_map)
    recommendation_context_verified = {
        arm.role for arm in source_arms
    }.issuperset({"management_comparison", "response_candidate"})

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

        parent_row_uid = _stable_uid(
            "parent-row-v1",
            source_uid,
            raw_row.source_row_number,
        )
        path_aliases = tuple(
            (position, alias)
            for position, value in enumerate(raw_row.raw_cells, start=1)
            if (alias := sensitive_path_alias(value)) is not None
        )
        for arm in source_arms:
            record: dict[str, Any] = {
                "source_name": source.source_name,
                "source_path": str(source.source_path),
                "source_sha256": source.source_sha256,
                "source_type": source.source_type,
                "source_family": source.source_family,
                "source_country_code": source.source_country_code,
                "source_encoding": source.source_encoding,
                "text_decoding_lineage": source.text_decoding_lineage,
                "data_classification": source.data_classification,
                "restricted_access_status": source.restricted_access_status,
                "shape_adapter_version": source.shape_adapter_version,
                "schema_map_path": str(source.schema_map_path) if source.schema_map_path is not None else None,
                "schema_map_sha256": source.schema_map_sha256,
                "schema_map_version": effective_map.map_version,
                "schema_map_review_id": effective_map.review_id,
                "schema_mapping_status": mapping_review_status,
                "normalization_map_version": effective_map.normalization_map_version,
                "normalization_review_id": effective_map.normalization_review_id,
                "workbook_csv_basis": effective_map.workbook_csv_basis,
                "source_uid": source_uid,
                "parent_row_uid": parent_row_uid,
                "source_arm_id": arm.arm_id,
                "source_arm_role": arm.role,
                "source_arm_uid": _stable_uid("source-arm-v1", parent_row_uid, arm.arm_id),
                "comparison_set_uid": _stable_uid("comparison-set-v1", parent_row_uid),
                "recommendation_set_membership_status": (
                    "verified_context_comparable"
                    if recommendation_context_verified
                    and arm.role in {"management_comparison", "response_candidate"}
                    else "not_verified"
                ),
                "recommendation_set_review_id": (
                    effective_map.review_id
                    if recommendation_context_verified
                    and arm.role in {"management_comparison", "response_candidate"}
                    else None
                ),
                "record_uid": _stable_uid(
                    "record-v2",
                    source_uid,
                    raw_row.source_row_number,
                    arm.arm_id,
                ),
                "source_row_number": raw_row.source_row_number,
                "source_physical_line_start": raw_row.source_physical_line_start,
                "source_physical_line_end": raw_row.source_physical_line_end,
                "raw_column_ids": tuple(column.raw_column_id for column in source.columns),
                "raw_headers": tuple(column.header for column in source.columns),
                "raw_cells": raw_row.raw_cells,
                "restricted_path_aliases": path_aliases,
                "public_source_alias": f"source_{source_uid[:16]}",
                "column_dispositions": tuple(
                    {
                        "position": disposition.position,
                        "role": disposition.role,
                        "canonical_field": disposition.canonical_field,
                        "variable_family": disposition.variable_family,
                    }
                    for disposition in effective_map.dispositions
                ),
                "held_variable_families": tuple(
                    sorted(
                        {
                            str(disposition.variable_family)
                            for disposition in effective_map.dispositions
                            if disposition.role in {"held", "descriptive", "restricted"}
                            and disposition.variable_family
                        }
                    )
                ),
            }
            effective_fields = dict(schema_fields)
            effective_fields.update(arm.field_positions)
            for canonical_name, position in effective_fields.items():
                raw_value = _field_raw_value(raw_row, position)
                effective_value = effective_by_position[position]
                record[f"{canonical_name}_raw"] = raw_value
                record[canonical_name] = canonicalize_irri(effective_value)
                record[f"{canonical_name}_missing_state"] = classify_missing(
                    raw_value,
                    missing_values,
                )
                record[f"{canonical_name}_raw_state"] = classify_raw_state(
                    raw_value,
                    missing_values,
                )
                record[f"{canonical_name}_filled_down"] = filled_by_position[position]
                record[f"{canonical_name}_filled_from_source_row_number"] = (
                    fill_origin_by_position[position]
                )
                record[f"{canonical_name}_normalization_status"] = (
                    "filled_down"
                    if filled_by_position[position]
                    else "text_standardized"
                    if record[canonical_name] != effective_value
                    else "unchanged"
                )
            for canonical_name, constant in arm.constants.items():
                record[f"{canonical_name}_raw"] = None
                record[canonical_name] = constant
                record[f"{canonical_name}_missing_state"] = "not_applicable"
                record[f"{canonical_name}_raw_state"] = "not_applicable"
                record[f"{canonical_name}_filled_down"] = False
                record[f"{canonical_name}_filled_from_source_row_number"] = None
                record[f"{canonical_name}_normalization_status"] = "arm_map_constant"
            records.append(
                _finalize_canonical_record(
                    record,
                    source=source,
                    config=config,
                    schema=schema,
                    missing_values=missing_values,
                    category_lookups=category_lookups,
                    restricted_policy=restricted_policy,
                )
            )
        previous_source_row_number = raw_row.source_row_number
    return records


def curate_ingestion(
    ingestion: IngestionResult,
    config: Any,
    *,
    source_maps: Mapping[str, ReviewedSourceMap] | None = None,
    category_lookups: Mapping[str, ReviewedLookupTable] | None = None,
    restricted_policy: RestrictedDataPolicy | None = None,
    require_reviewed_controls: bool = False,
) -> CurationResult:
    """Map configured raw positions to canonical values while preserving all raw cells."""

    maps = source_maps or {}
    lookups = category_lookups or {}
    if require_reviewed_controls:
        validate_reviewed_curation_controls(
            ingestion,
            source_maps=maps,
            category_lookups=lookups,
        )
    else:
        unknown_maps = set(maps) - {
            source.source_name for source in ingestion.sources
        }
        if unknown_maps:
            raise ValueError(
                "Reviewed maps reference source(s) absent from ingestion: "
                + ", ".join(sorted(unknown_maps))
            )
    records = tuple(
        record
        for source in ingestion.sources
        for record in _curate_source(
            source,
            config,
            source_map=maps.get(source.source_name),
            category_lookups=lookups,
            restricted_policy=restricted_policy,
        )
    )
    record_uids = [record["record_uid"] for record in records]
    if len(record_uids) != len(set(record_uids)):
        raise ValueError("Canonical record identifiers must be unique")
    parent_row_uids = tuple(
        sorted({str(record["parent_row_uid"]) for record in records})
    )
    expected_parent_rows = sum(len(source.rows) for source in ingestion.sources)
    if len(parent_row_uids) != expected_parent_rows:
        raise ValueError("Canonical parent-row identities do not reconcile to ingested rows")
    return CurationResult(records=records, parent_row_uids=parent_row_uids)


def project_public_records(
    records: Iterable[Mapping[str, Any]],
    *,
    policy: RestrictedDataPolicy,
) -> tuple[dict[str, Any], ...]:
    """Create a disclosure-reviewed projection without copying restricted raw evidence."""

    for label, value in (
        ("access review", policy.access_review_id),
        ("automated disclosure review", policy.automated_disclosure_review_id),
        ("human disclosure review", policy.human_disclosure_review_id),
    ):
        if not value.strip():
            raise ValueError(f"Restricted-data {label} evidence must be nonempty")
    prohibited_fields = {
        *policy.identifier_fields,
        *policy.precise_location_fields,
        *(set(policy.detailed_location_fields) - set(policy.approved_geography_fields)),
        "source_path",
        "schema_map_path",
        "raw_cells",
        "raw_headers",
        "raw_column_ids",
        "column_dispositions",
        "restricted_path_aliases",
        "record_uid",
        "parent_row_uid",
        "source_uid",
        "source_row_number",
        "source_physical_line_number",
        "controlled_subject_uid",
    }
    prohibited_raw_fields = {
        f"{field}{suffix}"
        for field in (
            *policy.identifier_fields,
            *policy.precise_location_fields,
            *policy.detailed_location_fields,
        )
        for suffix in ("_raw", "_missing_state", "_raw_state")
    }
    allowed_fields = set(policy.public_release_fields)
    forbidden_allowlist_fields = allowed_fields & prohibited_fields
    if forbidden_allowlist_fields:
        raise ValueError(
            "Restricted-data public allowlist contains prohibited field(s): "
            + ", ".join(sorted(forbidden_allowlist_fields))
        )
    public: list[dict[str, Any]] = []
    for source_record in records:
        record = dict(source_record)
        if record.get("data_classification") != "restricted":
            record.pop("source_path", None)
            record.pop("schema_map_path", None)
            record.pop("raw_cells", None)
            record.pop("restricted_path_aliases", None)
            public.append(record)
            continue
        if record.get("restricted_release_status") != "eligible_for_reviewed_public_projection":
            raise ValueError("Restricted record has not completed access and disclosure controls")
        projection = {
            key: value
            for key, value in record.items()
            if key in allowed_fields
            and key not in prohibited_fields
            and key not in prohibited_raw_fields
        }
        record_uid = str(record.get("record_uid") or "").strip()
        if not record_uid:
            raise ValueError("Restricted record lacks an internal record identity")
        projection["public_record_uid"] = _controlled_pseudonym(
            f"public-record:{record_uid}",
            salt=policy.pseudonym_salt,
        )
        projection["data_classification"] = "public_deidentified"
        projection["disclosure_review_status"] = "automated_and_human_review_recorded"
        projection["access_review_id"] = policy.access_review_id
        projection["automated_disclosure_review_id"] = (
            policy.automated_disclosure_review_id
        )
        projection["human_disclosure_review_id"] = policy.human_disclosure_review_id
        public.append(projection)
    return tuple(public)


__all__ = [
    "CurationResult",
    "PhysicalColumnDisposition",
    "REQUIRED_REVIEWED_LOOKUP_FIELDS",
    "RestrictedDataPolicy",
    "ReviewedSourceMap",
    "SourceArmMap",
    "curate_ingestion",
    "project_public_records",
    "validate_reviewed_curation_controls",
]
