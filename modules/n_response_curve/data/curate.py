from __future__ import annotations

from dataclasses import dataclass, field
import hmac
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .config import KNOWN_FILL_DOWN_FIELDS
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
    validate_reviewed_missing_state_table,
)


_COLUMN_ROLES = frozenset(
    {"canonical", "descriptive", "held", "restricted", "source_metadata", "blank"}
)
_SOURCE_VALUE_TYPES = frozenset(
    {"numeric", "text", "date", "mixed", "identifier", "blank"}
)
_PROVIDER_SEMANTICS_STATUSES = frozenset(
    {"verified", "unverified", "not_applicable"}
)
_LEAKAGE_CLASSES = frozenset(
    {
        "approved_predictor",
        "outcome",
        "identifier",
        "post_outcome",
        "economic",
        "descriptive",
        "held",
        "not_applicable",
    }
)
_ADDITIONAL_USE_STATUSES = frozenset(
    {
        "canonical_current_scope",
        "descriptive_only",
        "held_pending_separate_approval",
        "restricted",
        "blank",
    }
)
REQUIRED_REVIEWED_LOOKUP_FIELDS = (
    "water_regime",
    "season",
    "treatment_class",
)
_SERIES_IDENTITY_LOOKUP_FIELDS = MappingProxyType(
    {
        "water_regime": "water_regime",
        "season": "season",
        "region": "region",
        "province": "province",
        "variety": "rice_variety",
        "experimental_design": "experimental_design",
        "soil_texture": "soil_texture",
    }
)
_CURVE_CAPABLE_REPRESENTATION_BASES = frozenset(
    {"observation_level", "treatment_mean"}
)
_REQUIRED_CURVE_FIELD_ROLES = MappingProxyType(
    {
        "study": frozenset({"study_id"}),
        "trial": frozenset({"trial_id"}),
        "treatment": frozenset({"treatment", "treatment_id"}),
        "n_rate": frozenset({"n_rate_kg_ha", "inorganic_n_rate"}),
        "yield": frozenset({"yield_t_ha", "yield_kg_ha"}),
    }
)
_SERIES_IDENTITY_FIELD_ALIASES = MappingProxyType(
    {
        "site": frozenset({"site", "site_id", "location", "location_id"}),
        "experimental_design": frozenset(
            {"experimental_design", "design", "trial_design"}
        ),
        "management_context": frozenset(
            {
                "inorganic_p_rate",
                "inorganic_k_rate",
                "organic_fertilizer",
                "biofertilizer",
                "n_timing_pattern",
                "n_split_pattern",
            }
        ),
        "variety": frozenset({"variety", "rice_variety"}),
        "recommendation_class": frozenset(
            {"recommendation_class", "treatment", "treatment_id"}
        ),
    }
)
_MANDATORY_SERIES_IDENTITY_DIMENSIONS = frozenset(
    {"site", "experimental_design", "management_context"}
)
_KNOWN_TREATMENT_LOOKUP_CLASSES = frozenset(
    {
        "zero_n",
        "absolute_control",
        "mineral_n_rate",
        "RCM",
        "FP",
        "NOPT_NPK",
        "other",
        "unresolved",
    }
)
_REQUIRED_TREATMENT_LOOKUP_CLASSES = frozenset(
    {"zero_n", "absolute_control", "RCM", "FP", "NOPT_NPK"}
)
_FILL_DOWN_CANONICAL_FIELDS = frozenset(
    {"study_id", "trial_id", "source", "authors", "publication_year"}
)
NUTRIENT_CANONICAL_UNITS = MappingProxyType(
    {
        "inorganic_n_rate": CANONICAL_N_RATE_UNIT,
        "recommended_n_rate": CANONICAL_N_RATE_UNIT,
        "inorganic_p_rate": "kg P2O5 ha-1",
        "inorganic_k_rate": "kg K2O ha-1",
    }
)
_OPTIONAL_CANONICAL_SOURCE_FIELDS = frozenset({"recommended_n_rate"})


@dataclass(frozen=True)
class PhysicalColumnDisposition:
    """Reviewed analytical role for one physical source column."""

    position: int
    role: str
    canonical_field: str | None = None
    variable_family: str | None = None
    source_value_type: str | None = None
    provider_semantics_status: str | None = None
    date_conversion_rule: str | None = None
    leakage_class: str | None = None
    additional_use_status: str | None = None


@dataclass(frozen=True)
class SourceArmMap:
    """One reviewed long-form arm expanded from a physical parent row."""

    arm_id: str
    role: str
    field_positions: Mapping[str, int]
    constants: Mapping[str, str]
    comparability_group_id: str | None = None
    comparability_review_id: str | None = None
    recommendation_set_membership_status: str = "not_verified"
    recommendation_set_review_id: str | None = None

    def __post_init__(self) -> None:
        if (self.comparability_group_id is None) != (
            self.comparability_review_id is None
        ):
            raise ValueError(
                "Source-arm comparability group and review ID must be supplied together"
            )
        if self.comparability_group_id is not None and not all(
            value.strip()
            for value in (
                self.comparability_group_id,
                self.comparability_review_id or "",
            )
        ):
            raise ValueError("Source-arm comparability metadata must be nonempty")
        if self.recommendation_set_membership_status not in {
            "not_verified",
            "verified_context_comparable",
        }:
            raise ValueError("Source-arm recommendation-set status is unsupported")
        if self.recommendation_set_membership_status == "verified_context_comparable":
            if not (
                isinstance(self.recommendation_set_review_id, str)
                and self.recommendation_set_review_id.strip()
            ):
                raise ValueError(
                    "Verified source-arm recommendation membership requires a review ID"
                )
            if self.comparability_group_id is None:
                raise ValueError(
                    "Verified source-arm recommendation membership requires reviewed comparability"
                )
        elif self.recommendation_set_review_id is not None:
            raise ValueError(
                "Unverified source-arm recommendation membership cannot declare a review ID"
            )


@dataclass(frozen=True)
class ReviewedNutrientUnitControl:
    """Reviewed raw basis and documented conversion for one nutrient-rate field."""

    canonical_field: str
    source_unit: str
    source_basis: str
    canonical_unit: str
    conversion_factor: float
    conversion_rule: str
    review_id: str


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
    representation_basis: str = "unclear_mixed_scope"
    representation_basis_status: str = "review_required"
    fill_down_headers: tuple[str, ...] = ()
    arms: tuple[SourceArmMap, ...] = ()
    normalization_map_version: str | None = None
    normalization_review_id: str | None = None
    missing_state_maps: Mapping[str, ReviewedLookupTable] = field(
        default_factory=lambda: MappingProxyType({})
    )
    nutrient_unit_controls: Mapping[str, ReviewedNutrientUnitControl] = field(
        default_factory=lambda: MappingProxyType({})
    )
    approved_variable_families: frozenset[str] = frozenset()
    declared_constant_fields: tuple[str, ...] = ()
    workbook_sha256: str | None = None
    csv_sha256: str | None = None
    workbook_csv_reconciliation_review_id: str | None = None


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
    disclosure_review_projection_sha256: str | None = None


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


def validate_reviewed_fill_down_policy(source_map: ReviewedSourceMap) -> None:
    """Limit reviewed fill-down to mapped provenance and hierarchy identifiers."""

    canonical_by_position = {
        position: canonical_name
        for canonical_name, position in source_map.fields.items()
    }
    for header in source_map.fill_down_headers:
        positions = tuple(
            position
            for position, expected_header in source_map.expected_headers.items()
            if expected_header == header
        )
        if len(positions) > 1:
            raise ValueError(
                f"Reviewed fill-down header {header!r} is ambiguous across physical columns"
            )
        canonical_field = (
            canonical_by_position.get(positions[0]) if positions else None
        )
        if canonical_field is not None:
            if canonical_field in _FILL_DOWN_CANONICAL_FIELDS:
                continue
            raise ValueError(
                f"Reviewed fill-down header {header!r} is not a higher-level hierarchy "
                f"or provenance field; it maps to {canonical_field!r}"
            )
        if header not in KNOWN_FILL_DOWN_FIELDS:
            raise ValueError(
                f"Reviewed fill-down header {header!r} is not a higher-level hierarchy "
                "or provenance field and is not an approved physical header"
            )


def validate_reviewed_nutrient_unit_controls(source_map: ReviewedSourceMap) -> None:
    """Require explicit reviewed unit/basis conversion for every mapped nutrient rate."""

    mapped_fields = set(source_map.fields)
    mapped_fields.update(source_map.declared_constant_fields)
    for arm in source_map.arms:
        mapped_fields.update(arm.field_positions)
        mapped_fields.update(arm.constants)
    required_fields = mapped_fields.intersection(NUTRIENT_CANONICAL_UNITS)
    control_fields = set(source_map.nutrient_unit_controls)
    missing_fields = sorted(required_fields - control_fields)
    unexpected_fields = sorted(control_fields - required_fields)
    if missing_fields:
        raise ValueError(
            f"{missing_fields[0]} lacks a reviewed unit/basis control"
        )
    if unexpected_fields:
        raise ValueError(
            "Reviewed nutrient unit/basis controls target unmapped field(s): "
            + ", ".join(unexpected_fields)
        )
    for field_name, control in source_map.nutrient_unit_controls.items():
        expected_unit = NUTRIENT_CANONICAL_UNITS[field_name]
        if control.canonical_field != field_name:
            raise ValueError("Nutrient unit control canonical field does not match its key")
        if control.canonical_unit != expected_unit:
            raise ValueError(
                f"{field_name} nutrient control must canonicalize to {expected_unit}"
            )
        if control.source_basis not in {"elemental", "oxide"}:
            raise ValueError("Nutrient unit control source basis is unsupported")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                control.source_unit,
                control.conversion_rule,
                control.review_id,
            )
        ):
            raise ValueError("Nutrient unit control requires explicit reviewed evidence")
        if not math.isfinite(control.conversion_factor) or control.conversion_factor <= 0:
            raise ValueError("Nutrient unit conversion factor must be finite and positive")
        if (
            control.source_unit == control.canonical_unit
            and control.conversion_factor != 1.0
        ):
            raise ValueError("Identity nutrient-unit conversion must use factor 1")


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
        ("representation basis", source_map.representation_basis),
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
    if source_map.workbook_csv_basis == "parallel_workbook_csv_unresolved":
        raise ValueError(
            "Parallel workbook/CSV source lacks reviewed workbook/CSV reconciliation"
        )
    if source_map.workbook_csv_basis == "parallel_workbook_csv_verified_equivalent":
        reconciliation_values = (
            source_map.workbook_sha256,
            source_map.csv_sha256,
            source_map.workbook_csv_reconciliation_review_id,
        )
        if not all(
            isinstance(value, str) and value.strip()
            for value in reconciliation_values
        ):
            raise ValueError(
                "Verified workbook/CSV equivalence requires both digests and review evidence"
            )
        if any(
            re.fullmatch(r"[0-9a-f]{64}", str(digest).strip().lower()) is None
            for digest in (source_map.workbook_sha256, source_map.csv_sha256)
        ):
            raise ValueError("Workbook/CSV reconciliation digests must be SHA-256 values")
        if source_map.csv_sha256 != source.source_sha256:
            raise ValueError(
                "Workbook/CSV reconciliation is not bound to the ingested CSV bytes"
            )
    if source_map.representation_basis not in KNOWN_REPRESENTATION_BASES:
        raise ValueError("Reviewed source map has an unsupported representation basis")
    if source_map.representation_basis_status not in {"reviewed", "review_required"}:
        raise ValueError("Reviewed source map has an unsupported representation-basis status")
    validate_reviewed_fill_down_policy(source_map)
    if source_map.representation_basis_status == "reviewed":
        validate_reviewed_nutrient_unit_controls(source_map)
    if (
        source.representation_basis_status == "reviewed"
        and source_map.representation_basis != source.representation_basis
    ):
        raise ValueError(
            "Reviewed source map representation basis differs from the ingested source basis"
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
        if (
            source_map.approved_variable_families
            and disposition.variable_family
            and disposition.variable_family not in source_map.approved_variable_families
        ):
            raise ValueError(
                f"Physical column {disposition.position} variable family "
                f"{disposition.variable_family!r} is not approved by the source map"
            )
        for value, allowed, label in (
            (disposition.source_value_type, _SOURCE_VALUE_TYPES, "source value type"),
            (
                disposition.provider_semantics_status,
                _PROVIDER_SEMANTICS_STATUSES,
                "provider semantics status",
            ),
            (disposition.leakage_class, _LEAKAGE_CLASSES, "leakage class"),
            (
                disposition.additional_use_status,
                _ADDITIONAL_USE_STATUSES,
                "additional-use status",
            ),
        ):
            if value is not None and value not in allowed:
                raise ValueError(
                    f"Physical column {disposition.position} has an unknown {label}"
                )
        if disposition.date_conversion_rule is not None and not str(
            disposition.date_conversion_rule
        ).strip():
            raise ValueError(
                f"Physical column {disposition.position} has an empty date conversion rule"
            )
    if source.source_name == "ph_combined_nopt_rcm":
        incomplete_semantic_positions = tuple(
            disposition.position
            for disposition in source_map.dispositions
            if any(
                value is None
                for value in (
                    disposition.source_value_type,
                    disposition.provider_semantics_status,
                    disposition.leakage_class,
                    disposition.additional_use_status,
                )
            )
            or (
                disposition.source_value_type == "date"
                and disposition.date_conversion_rule is None
            )
        )
        if not source_map.approved_variable_families or incomplete_semantic_positions:
            raise ValueError(
                "Combined source requires a complete semantic disposition and approved "
                "variable-family allowlist for every physical column; "
                f"incomplete_positions={incomplete_semantic_positions or 'none'}"
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
            disposition = by_position[position]
            if (
                disposition.role not in {"canonical", "restricted"}
                or disposition.canonical_field != canonical_name
            ):
                raise ValueError(
                    "Reviewed source arm mapping disagrees with its column disposition"
                )
        if set(arm.field_positions).intersection(arm.constants):
            raise ValueError("A source-arm field cannot be both positional and constant")
        undeclared_constants = set(arm.constants) - set(
            source_map.declared_constant_fields
        )
        if undeclared_constants:
            raise ValueError(
                "Reviewed source arm constants require declared canonical fields: "
                + ", ".join(sorted(undeclared_constants))
            )


def validate_reviewed_curation_controls(
    ingestion: IngestionResult,
    *,
    source_maps: Mapping[str, ReviewedSourceMap],
    category_lookups: Mapping[str, ReviewedLookupTable],
    source_category_lookups: Mapping[
        str, Mapping[str, ReviewedLookupTable]
    ] | None = None,
    required_lookup_fields: Iterable[str] = REQUIRED_REVIEWED_LOOKUP_FIELDS,
    required_series_identity_dimensions: Iterable[str] = (),
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
    for required_field in required_lookup_fields:
        if not isinstance(required_field, str) or not required_field.strip():
            raise ValueError("Required reviewed lookup fields must be nonempty strings")
        normalized_required.append(required_field.strip())
    if len(normalized_required) != len(set(normalized_required)):
        raise ValueError("Required reviewed lookup fields must be unique")

    configured_identity_dimensions = {
        str(dimension).strip()
        for dimension in required_series_identity_dimensions
        if str(dimension).strip()
    }
    effective_identity_dimensions = set(configured_identity_dimensions)
    if configured_identity_dimensions:
        effective_identity_dimensions.update(_MANDATORY_SERIES_IDENTITY_DIMENSIONS)

    scoped_lookups = source_category_lookups or {}
    unknown_lookup_sources = set(scoped_lookups) - expected_sources
    if unknown_lookup_sources:
        raise ValueError(
            "Source-specific category lookups reference source(s) absent from ingestion: "
            + ", ".join(sorted(unknown_lookup_sources))
        )
    required_identity_lookup_fields = {
        lookup_field
        for dimension in effective_identity_dimensions
        if (
            lookup_field := _SERIES_IDENTITY_LOOKUP_FIELDS.get(
                str(dimension).strip()
            )
        )
    }
    for source_name in source_names:
        effective_lookups = dict(category_lookups)
        effective_lookups.update(scoped_lookups.get(source_name, {}))
        missing_lookups = (
            set(normalized_required) | required_identity_lookup_fields
        ) - set(effective_lookups)
        if missing_lookups:
            raise ValueError(
                "Required reviewed category lookup(s) are missing for "
                f"source {source_name}: "
                + ", ".join(sorted(missing_lookups))
            )
        for lookup_field, lookup in effective_lookups.items():
            if not isinstance(lookup_field, str) or not lookup_field.strip():
                raise ValueError("Reviewed category lookup field names must be nonempty")
            validate_reviewed_lookup_table(lookup)

        treatment_lookup = effective_lookups.get("treatment_class")
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
        source_map = source_maps[source.source_name]
        _validate_reviewed_source_map(source, source_map)
        physical_canonical_fields = set(source_map.fields)
        available_canonical_fields = set(physical_canonical_fields)
        available_canonical_fields.update(source_map.declared_constant_fields)
        for arm in source_map.arms:
            physical_canonical_fields.update(arm.field_positions)
            available_canonical_fields.update(arm.field_positions)
            available_canonical_fields.update(arm.constants)
        if (
            source_map.representation_basis
            in _CURVE_CAPABLE_REPRESENTATION_BASES
            and source_map.representation_basis_status == "reviewed"
        ):
            missing_roles = sorted(
                role
                for role, alternatives in _REQUIRED_CURVE_FIELD_ROLES.items()
                if not alternatives.intersection(available_canonical_fields)
            )
            required_context = effective_identity_dimensions
            missing_context = sorted(
                field_name
                for field_name in required_context
                if not _SERIES_IDENTITY_FIELD_ALIASES.get(
                    field_name,
                    frozenset({field_name}),
                ).intersection(available_canonical_fields)
            )
            if missing_roles or missing_context:
                raise ValueError(
                    "Reviewed curve-capable source map lacks required curve-role coverage; "
                    f"source={source.source_name}; missing_roles={missing_roles}; "
                    f"missing_context={missing_context}"
                )
        mapped_missing_fields = set(source_map.missing_state_maps)
        missing_state_fields = physical_canonical_fields - mapped_missing_fields
        unexpected_state_fields = mapped_missing_fields - physical_canonical_fields
        if missing_state_fields or unexpected_state_fields:
            missing = ", ".join(sorted(missing_state_fields)) or "none"
            unexpected = ", ".join(sorted(unexpected_state_fields)) or "none"
            raise ValueError(
                "Missing-state maps must cover every physical canonical field; "
                f"missing={missing}; unexpected={unexpected}"
            )
        if not (
            isinstance(source_map.normalization_map_version, str)
            and source_map.normalization_map_version.strip()
            and isinstance(source_map.normalization_review_id, str)
            and source_map.normalization_review_id.strip()
        ):
            raise ValueError(
                "Reviewed missing-state maps require source normalization version and review evidence"
            )
        for field_name, lookup in source_map.missing_state_maps.items():
            validate_reviewed_missing_state_table(lookup)
            if (
                lookup.map_version != source_map.normalization_map_version
                or lookup.review_id != source_map.normalization_review_id
            ):
                raise ValueError(
                    "Missing-state map evidence does not match the source normalization contract: "
                    + field_name
                )


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
    missing_state_maps: Mapping[str, ReviewedLookupTable],
    nutrient_unit_controls: Mapping[str, ReviewedNutrientUnitControl],
    category_lookups: Mapping[str, ReviewedLookupTable],
    restricted_policy: RestrictedDataPolicy | None,
) -> dict[str, Any]:
    n_parse = parse_numeric(
        _optional_field(record, "inorganic_n_rate"),
        missing_values,
        missing_state_lookup=missing_state_maps.get("inorganic_n_rate"),
    )
    p_parse = parse_numeric(
        _optional_field(record, "inorganic_p_rate"),
        missing_values,
        missing_state_lookup=missing_state_maps.get("inorganic_p_rate"),
    )
    k_parse = parse_numeric(
        _optional_field(record, "inorganic_k_rate"),
        missing_values,
        missing_state_lookup=missing_state_maps.get("inorganic_k_rate"),
    )
    recommended_n_parse = parse_numeric(
        _optional_field(record, "recommended_n_rate"),
        missing_values,
        missing_state_lookup=missing_state_maps.get("recommended_n_rate"),
    )
    yield_se_parse = parse_numeric(
        _optional_field(record, "yield_se_t_ha"),
        missing_values,
        missing_state_lookup=missing_state_maps.get("yield_se_t_ha"),
    )
    yield_normalization = normalize_yield(
        _optional_field(record, "yield_kg_ha"),
        _optional_field(record, "yield_t_ha"),
        missing_values,
        kg_missing_state_lookup=missing_state_maps.get("yield_kg_ha"),
        t_missing_state_lookup=missing_state_maps.get("yield_t_ha"),
    )
    configured_units = schema.get("units", {})
    configured_n_unit = str(configured_units.get("n_rate", CANONICAL_N_RATE_UNIT))
    configured_yield_unit = str(
        configured_units.get("yield_curve", CANONICAL_YIELD_UNIT)
    )
    nutrient_values: dict[str, float | None] = {}
    nutrient_metadata: dict[str, dict[str, Any]] = {}
    for field_name, prefix, parsed in (
        ("inorganic_n_rate", "n_rate", n_parse),
        ("recommended_n_rate", "recommended_n_rate", recommended_n_parse),
        ("inorganic_p_rate", "p_rate", p_parse),
        ("inorganic_k_rate", "k_rate", k_parse),
    ):
        control = nutrient_unit_controls.get(field_name)
        if control is None:
            nutrient_values[field_name] = parsed.value
            nutrient_metadata[prefix] = {
                "raw_value": parsed.value,
                "source_unit": (
                    configured_n_unit
                    if field_name in {"inorganic_n_rate", "recommended_n_rate"}
                    else None
                ),
                "source_basis": None,
                "canonical_unit": NUTRIENT_CANONICAL_UNITS[field_name],
                "conversion_rule": None,
                "conversion_factor": None,
                "review_id": None,
                "unit_status": "review_required_unversioned",
            }
            continue
        nutrient_values[field_name] = (
            parsed.value * control.conversion_factor
            if parsed.value is not None
            else None
        )
        nutrient_metadata[prefix] = {
            "raw_value": parsed.value,
            "source_unit": control.source_unit,
            "source_basis": control.source_basis,
            "canonical_unit": control.canonical_unit,
            "conversion_rule": control.conversion_rule,
            "conversion_factor": control.conversion_factor,
            "review_id": control.review_id,
            "unit_status": (
                "canonical_reviewed"
                if control.conversion_factor == 1.0
                and control.source_unit == control.canonical_unit
                else "converted_reviewed"
            ),
        }
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
    for field_name, reason in (
        ("inorganic_n_rate", "N_RATE_UNIT_BASIS_REVIEW_REQUIRED"),
        ("recommended_n_rate", "RECOMMENDED_N_RATE_UNIT_BASIS_REVIEW_REQUIRED"),
        ("inorganic_p_rate", "P_RATE_UNIT_BASIS_REVIEW_REQUIRED"),
        ("inorganic_k_rate", "K_RATE_UNIT_BASIS_REVIEW_REQUIRED"),
    ):
        parsed_value = nutrient_values[field_name]
        if parsed_value is not None and field_name not in nutrient_unit_controls:
            representation_reasons.append(reason)
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
            "n_rate_kg_ha": nutrient_values["inorganic_n_rate"],
            "n_rate_parse_status": n_parse.status,
            "actual_n_rate_kg_ha": nutrient_values["inorganic_n_rate"],
            "recommended_n_rate_kg_ha": nutrient_values["recommended_n_rate"],
            "recommended_n_rate_parse_status": recommended_n_parse.status,
            "recommended_n_rate_raw_value": nutrient_metadata["recommended_n_rate"]["raw_value"],
            "recommended_n_rate_source_unit": nutrient_metadata["recommended_n_rate"]["source_unit"],
            "recommended_n_rate_source_basis": nutrient_metadata["recommended_n_rate"]["source_basis"],
            "recommended_n_rate_canonical_unit": nutrient_metadata["recommended_n_rate"]["canonical_unit"],
            "recommended_n_rate_conversion_rule": nutrient_metadata["recommended_n_rate"]["conversion_rule"],
            "recommended_n_rate_conversion_factor": nutrient_metadata["recommended_n_rate"]["conversion_factor"],
            "recommended_n_rate_unit_review_id": nutrient_metadata["recommended_n_rate"]["review_id"],
            "recommended_n_rate_unit_status": nutrient_metadata["recommended_n_rate"]["unit_status"],
            "n_rate_configured_unit": nutrient_metadata["n_rate"]["source_unit"],
            "n_rate_raw_value": nutrient_metadata["n_rate"]["raw_value"],
            "n_rate_source_unit": nutrient_metadata["n_rate"]["source_unit"],
            "n_rate_source_basis": nutrient_metadata["n_rate"]["source_basis"],
            "n_rate_canonical_unit": nutrient_metadata["n_rate"]["canonical_unit"],
            "n_rate_conversion_rule": nutrient_metadata["n_rate"]["conversion_rule"],
            "n_rate_conversion_factor": nutrient_metadata["n_rate"]["conversion_factor"],
            "n_rate_unit_review_id": nutrient_metadata["n_rate"]["review_id"],
            "n_rate_unit_status": nutrient_metadata["n_rate"]["unit_status"],
            "p_rate_kg_p2o5_ha": nutrient_values["inorganic_p_rate"],
            "p_rate_parse_status": p_parse.status,
            "p_rate_raw_value": nutrient_metadata["p_rate"]["raw_value"],
            "p_rate_source_unit": nutrient_metadata["p_rate"]["source_unit"],
            "p_rate_source_basis": nutrient_metadata["p_rate"]["source_basis"],
            "p_rate_canonical_unit": nutrient_metadata["p_rate"]["canonical_unit"],
            "p_rate_conversion_rule": nutrient_metadata["p_rate"]["conversion_rule"],
            "p_rate_conversion_factor": nutrient_metadata["p_rate"]["conversion_factor"],
            "p_rate_unit_review_id": nutrient_metadata["p_rate"]["review_id"],
            "p_rate_unit_status": nutrient_metadata["p_rate"]["unit_status"],
            "k_rate_kg_k2o_ha": nutrient_values["inorganic_k_rate"],
            "k_rate_parse_status": k_parse.status,
            "k_rate_raw_value": nutrient_metadata["k_rate"]["raw_value"],
            "k_rate_source_unit": nutrient_metadata["k_rate"]["source_unit"],
            "k_rate_source_basis": nutrient_metadata["k_rate"]["source_basis"],
            "k_rate_canonical_unit": nutrient_metadata["k_rate"]["canonical_unit"],
            "k_rate_conversion_rule": nutrient_metadata["k_rate"]["conversion_rule"],
            "k_rate_conversion_factor": nutrient_metadata["k_rate"]["conversion_factor"],
            "k_rate_unit_review_id": nutrient_metadata["k_rate"]["review_id"],
            "k_rate_unit_status": nutrient_metadata["k_rate"]["unit_status"],
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
    if treatment_lookup is None:
        treatment_normalization = None
        record["treatment_class_normalization_status"] = (
            "review_required_unversioned"
        )
    else:
        treatment_normalization = normalize_category_with_evidence(
            _optional_field(record, "treatment"),
            treatment_lookup,
        )
        record["treatment_class_normalization_status"] = (
            treatment_normalization.status
        )
    record["treatment_class_normalized"] = (
        treatment_normalization.canonical_value
        if treatment_normalization is not None
        else None
    )
    record["treatment_class_map_version"] = (
        treatment_lookup.map_version if treatment_lookup is not None else None
    )
    record["treatment_class_review_id"] = (
        treatment_lookup.review_id if treatment_lookup is not None else None
    )
    fit_role_is_review_bound = bool(
        treatment_lookup is not None
        and treatment_normalization is not None
        and treatment_normalization.status == "mapped_reviewed"
        and treatment_normalization.canonical_value
        == record.get("treatment_text_class")
        and record.get("treatment_classification_status") == "resolved"
    )
    record["treatment_fit_role_provenance_status"] = (
        "reviewed_lookup_bound" if fit_role_is_review_bound else "review_required"
    )
    record["treatment_fit_role_map_version"] = (
        treatment_lookup.map_version
        if fit_role_is_review_bound and treatment_lookup is not None
        else None
    )
    record["treatment_fit_role_review_id"] = (
        treatment_lookup.review_id
        if fit_role_is_review_bound and treatment_lookup is not None
        else None
    )
    if record.get("treatment_fit_role") == "curve_candidate" and not fit_role_is_review_bound:
        record["treatment_fit_role"] = "review"
        review_reasons: set[str] = {
            str(reason) for reason in record.get("treatment_review_reasons", ())
        }
        review_reasons.add("TREATMENT_FIT_ROLE_REVIEW_REQUIRED")
        record["treatment_review_reasons"] = tuple(sorted(review_reasons))
        record["treatment_classification_status"] = "review_required"
    for field_name, lookup in category_lookups.items():
        if field_name in {"water_regime", "season", "treatment_class"}:
            continue
        normalized = normalize_category_with_evidence(
            _optional_field(record, field_name),
            lookup,
        )
        record[f"{field_name}_normalized"] = normalized.canonical_value or "unresolved"
        record[f"{field_name}_normalization_status"] = normalized.status
        record[f"{field_name}_map_version"] = normalized.map_version
        record[f"{field_name}_review_id"] = normalized.review_id

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
                "representation_basis": effective_map.representation_basis,
                "representation_basis_status": effective_map.representation_basis_status,
                "source_uid": source_uid,
                "parent_row_uid": parent_row_uid,
                "source_arm_id": arm.arm_id,
                "source_arm_role": arm.role,
                "source_arm_comparability_group_id": arm.comparability_group_id,
                "source_arm_comparability_review_id": arm.comparability_review_id,
                "source_arm_uid": _stable_uid("source-arm-v1", parent_row_uid, arm.arm_id),
                "treatment_uid": _stable_uid(
                    "treatment-v1",
                    parent_row_uid,
                    arm.arm_id,
                ),
                "comparison_set_uid": _stable_uid("comparison-set-v1", parent_row_uid),
                "recommendation_set_membership_status": (
                    arm.recommendation_set_membership_status
                ),
                "recommendation_set_review_id": arm.recommendation_set_review_id,
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
                        "source_value_type": disposition.source_value_type,
                        "provider_semantics_status": disposition.provider_semantics_status,
                        "date_conversion_rule": disposition.date_conversion_rule,
                        "leakage_class": disposition.leakage_class,
                        "additional_use_status": disposition.additional_use_status,
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
                missing_state_lookup = effective_map.missing_state_maps.get(
                    canonical_name
                )
                raw_state = classify_raw_state(raw_value, missing_values)
                canonical_missing_state = classify_missing(
                    raw_value,
                    missing_values,
                    missing_state_lookup=missing_state_lookup,
                )
                record[f"{canonical_name}_raw"] = raw_value
                record[canonical_name] = canonicalize_irri(effective_value)
                record[f"{canonical_name}_missing_state"] = (
                    canonical_missing_state
                )
                record[f"{canonical_name}_raw_state"] = raw_state
                record[f"{canonical_name}_missing_state_normalization_status"] = (
                    "review_required_unversioned"
                    if missing_state_lookup is None
                    else "unresolved_unmapped"
                    if canonical_missing_state == "unresolved_missing"
                    else "preserved_blank"
                    if raw_state == "blank"
                    else "not_missing"
                    if canonical_missing_state == "present"
                    else "mapped_reviewed"
                )
                record[f"{canonical_name}_missing_state_map_version"] = (
                    missing_state_lookup.map_version
                    if missing_state_lookup is not None
                    else None
                )
                record[f"{canonical_name}_missing_state_review_id"] = (
                    missing_state_lookup.review_id
                    if missing_state_lookup is not None
                    else None
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
                record[f"{canonical_name}_missing_state"] = "present"
                record[f"{canonical_name}_raw_state"] = "structural_missing"
                record[f"{canonical_name}_missing_state_normalization_status"] = (
                    "arm_map_constant_reviewed"
                )
                record[f"{canonical_name}_missing_state_map_version"] = (
                    effective_map.map_version
                )
                record[f"{canonical_name}_missing_state_review_id"] = (
                    effective_map.review_id
                )
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
                    missing_state_maps=effective_map.missing_state_maps,
                    nutrient_unit_controls=effective_map.nutrient_unit_controls,
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
    source_category_lookups: Mapping[
        str, Mapping[str, ReviewedLookupTable]
    ] | None = None,
    restricted_policy: RestrictedDataPolicy | None = None,
    require_reviewed_controls: bool = False,
) -> CurationResult:
    """Map configured raw positions to canonical values while preserving all raw cells."""

    maps = source_maps or {}
    lookups = category_lookups or {}
    scoped_lookups = source_category_lookups or {}
    if require_reviewed_controls:
        configured_canonical_fields = (
            set(config.raw["schema"]["fields"])
            | _OPTIONAL_CANONICAL_SOURCE_FIELDS
        )
        for source_name, source_map in maps.items():
            reviewed_canonical_fields = set(source_map.fields)
            reviewed_canonical_fields.update(source_map.declared_constant_fields)
            for arm in source_map.arms:
                reviewed_canonical_fields.update(arm.field_positions)
                reviewed_canonical_fields.update(arm.constants)
            unapproved_canonical_fields = (
                reviewed_canonical_fields - configured_canonical_fields
            )
            if unapproved_canonical_fields:
                raise ValueError(
                    "Reviewed source map contains fields outside the configured canonical "
                    f"data contract for {source_name}: "
                    + ", ".join(sorted(unapproved_canonical_fields))
                )
        validate_reviewed_curation_controls(
            ingestion,
            source_maps=maps,
            category_lookups=lookups,
            source_category_lookups=scoped_lookups,
            required_series_identity_dimensions=getattr(
                config,
                "series_identity_dimensions",
                (),
            ),
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
            category_lookups={
                **lookups,
                **scoped_lookups.get(source.source_name, {}),
            },
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


def _project_public_records(
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
            record_uid = str(record.get("record_uid") or "").strip()
            if record_uid:
                record["release_record_uid"] = record_uid
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
        public_record_uid = _controlled_pseudonym(
            f"public-record:{record_uid}",
            salt=policy.pseudonym_salt,
        )
        projection["public_record_uid"] = public_record_uid
        projection["release_record_uid"] = public_record_uid
        projection["data_classification"] = "public_deidentified"
        projection["public_projection_status"] = (
            "released_after_reviewed_projection"
        )
        projection["disclosure_review_status"] = "automated_and_human_review_recorded"
        projection["access_review_id"] = policy.access_review_id
        projection["automated_disclosure_review_id"] = (
            policy.automated_disclosure_review_id
        )
        projection["human_disclosure_review_id"] = policy.human_disclosure_review_id
        public.append(projection)
    return tuple(public)


def disclosure_review_projection_sha256(
    records: Iterable[Mapping[str, Any]],
    *,
    policy: RestrictedDataPolicy,
) -> str:
    """Hash the exact deidentified restricted projection reviewed for release."""

    projected = _project_public_records(tuple(records), policy=policy)
    restricted_projection = sorted(
        (
            dict(record)
            for record in projected
            if record.get("data_classification") == "public_deidentified"
        ),
        key=lambda record: str(record.get("release_record_uid") or ""),
    )
    payload = json.dumps(
        restricted_projection,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def project_public_records(
    records: Iterable[Mapping[str, Any]],
    *,
    policy: RestrictedDataPolicy,
) -> tuple[dict[str, Any], ...]:
    """Create a projection only when both disclosure reviews bind its exact bytes."""

    source_records = tuple(records)
    projected = _project_public_records(source_records, policy=policy)
    restricted_rows_present = any(
        record.get("data_classification") == "restricted"
        for record in source_records
    )
    if not restricted_rows_present:
        return projected
    reviewed_digest = str(
        policy.disclosure_review_projection_sha256 or ""
    ).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", reviewed_digest) is None:
        raise ValueError(
            "Restricted-data disclosure reviews must bind the exact projected bytes"
        )
    observed_digest = disclosure_review_projection_sha256(
        source_records,
        policy=policy,
    )
    if observed_digest != reviewed_digest:
        raise ValueError(
            "Restricted-data projection does not match the disclosure-reviewed bytes"
        )
    return projected


__all__ = [
    "CurationResult",
    "PhysicalColumnDisposition",
    "REQUIRED_REVIEWED_LOOKUP_FIELDS",
    "RestrictedDataPolicy",
    "ReviewedNutrientUnitControl",
    "ReviewedSourceMap",
    "SourceArmMap",
    "curate_ingestion",
    "disclosure_review_projection_sha256",
    "project_public_records",
    "validate_reviewed_curation_controls",
]
