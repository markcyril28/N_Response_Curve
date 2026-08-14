from __future__ import annotations

from collections import defaultdict
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .curate import (
    NUTRIENT_CANONICAL_UNITS,
    PhysicalColumnDisposition,
    ReviewedNutrientUnitControl,
    ReviewedSourceMap,
    SourceArmMap,
)
from .ingest import IngestedSource, IngestionResult, WorkbookCsvReconciliation
from .schema import ReviewedLookupTable


_CORE_SOURCE_NAME = "core_trial_data"
_COMBINED_SOURCE_NAME = "ph_combined_nopt_rcm"
_LTCCE_SOURCE_NAME = "ltcce"

_CORE_MAP_REVIEW_ID = "source-policy-profile-2026-08-13-core"
_COMBINED_MAP_REVIEW_ID = "source-policy-profile-2026-08-13-combined"
_LTCCE_MAP_REVIEW_ID = "source-policy-profile-2026-08-13-ltcce"
_YIELD_PRECEDENCE_REVIEW_ID = "source-policy-profile-2026-08-13-yield-precedence"

_COMBINED_WORKBOOK_SHA256 = (
    "78e12a020a1a214e95df66b197b6a0793139ae535ed987c8692bf6060046f39f"
)
_COMBINED_CSV_SHA256 = (
    "2f9b05914f50b3935ad52b4d3ee2444a6d9fbd43fcf2f52719ea595e33ac9076"
)
_LTCCE_WORKBOOK_SHA256 = (
    "67339c50bdf23a5b016c12f19a32d63f1251761892e94b47b0cad626df14e3af"
)
_LTCCE_CSV_SHA256 = (
    "f8ab7c50ca51f95c596d15e5d01ba23749dce085da050db56956efb803330ef8"
)

_COMBINED_IDENTIFIER_POSITIONS = frozenset({1, 2, 8, 9, 113, 114, 116})
_COMBINED_PRECISE_LOCATION_POSITIONS = frozenset({117, 118})
_COMBINED_LOCATION_POSITIONS = frozenset({4, 5, 6, 7})
_COMBINED_DATE_POSITIONS = frozenset(
    {25, 49, 50, 51, 52, 76, 77, 78, 79, 96, 127, 128, 129}
)
_COMBINED_ECONOMIC_POSITIONS = frozenset(range(105, 111))
_COMBINED_SOIL_POSITIONS = frozenset((*range(29, 49), *range(161, 229)))
_COMBINED_CLIMATE_POSITIONS = frozenset((*range(21, 24), *range(119, 126)))

_COMBINED_FIELDS = MappingProxyType(
    {
        "country": 3,
        "region": 4,
        "province": 5,
        "planting_year": 11,
        "water_regime": 13,
        "rice_variety": 26,
        "study_id": 113,
        "trial_id": 116,
        "season": 126,
    }
)
_COMBINED_ARMS = (
    SourceArmMap(
        arm_id="fp",
        role="farmer_practice_comparison",
        field_positions=MappingProxyType(
            {
                "inorganic_n_rate": 58,
                "inorganic_p_rate": 63,
                "inorganic_k_rate": 68,
                "yield_t_ha": 103,
            }
        ),
        constants=MappingProxyType({"treatment": "FP", "treatment_id": "FP"}),
    ),
    SourceArmMap(
        arm_id="rcm",
        role="rice_crop_manager_comparison",
        field_positions=MappingProxyType(
            {
                "inorganic_n_rate": 85,
                "inorganic_p_rate": 90,
                "inorganic_k_rate": 95,
                "yield_t_ha": 104,
            }
        ),
        constants=MappingProxyType({"treatment": "RCM", "treatment_id": "RCM"}),
    ),
    SourceArmMap(
        arm_id="nopt_full",
        role="nopt_full_fertilizer_comparison",
        field_positions=MappingProxyType(
            {
                "inorganic_n_rate": 140,
                "inorganic_p_rate": 141,
                "inorganic_k_rate": 143,
                "yield_t_ha": 148,
            }
        ),
        constants=MappingProxyType(
            {"treatment": "NOPT NPK", "treatment_id": "NOPT_NPK"}
        ),
    ),
    SourceArmMap(
        arm_id="nopt_zero_n",
        role="nopt_zero_n_comparison_pk_unresolved",
        field_positions=MappingProxyType({"yield_t_ha": 149}),
        constants=MappingProxyType(
            {
                "treatment": "zero N",
                "treatment_id": "NOPT_ZERO_N",
                "inorganic_n_rate": "0",
            }
        ),
    ),
)

_LTCCE_FIELDS = MappingProxyType(
    {
        "experimental_design": 1,
        "study_id": 2,
        "trial_id": 3,
        "planting_year": 4,
        "season": 5,
        "rice_variety": 8,
        "source_variety_code": 9,
        "inorganic_n_rate": 10,
        "replicate": 11,
        "yield_t_ha": 12,
    }
)


def _expected_headers(source: IngestedSource) -> Mapping[int, str]:
    return MappingProxyType(
        {column.position: column.header for column in source.columns}
    )


def _identity_nutrient_control(
    canonical_field: str,
    *,
    review_id: str,
) -> ReviewedNutrientUnitControl:
    if canonical_field == "inorganic_n_rate":
        source_unit = "kg N ha-1"
        source_basis = "elemental"
    elif canonical_field == "inorganic_p_rate":
        source_unit = "kg P2O5 ha-1"
        source_basis = "oxide"
    elif canonical_field == "inorganic_k_rate":
        source_unit = "kg K2O ha-1"
        source_basis = "oxide"
    else:
        raise ValueError(f"Unsupported built-in nutrient field: {canonical_field}")
    return ReviewedNutrientUnitControl(
        canonical_field=canonical_field,
        source_unit=source_unit,
        source_basis=source_basis,
        canonical_unit=NUTRIENT_CANONICAL_UNITS[canonical_field],
        conversion_factor=1.0,
        conversion_rule="reviewed identity conversion",
        review_id=review_id,
    )


def _nutrient_controls(
    canonical_fields: Iterable[str],
    *,
    review_id: str,
) -> Mapping[str, ReviewedNutrientUnitControl]:
    fields = tuple(
        field_name
        for field_name in (
            "inorganic_n_rate",
            "inorganic_p_rate",
            "inorganic_k_rate",
        )
        if field_name in set(canonical_fields)
    )
    return MappingProxyType(
        {
            field_name: _identity_nutrient_control(
                field_name,
                review_id=review_id,
            )
            for field_name in fields
        }
    )


def _missing_state_lookup(
    *,
    map_version: str,
    review_id: str,
) -> ReviewedLookupTable:
    return ReviewedLookupTable(
        map_version=map_version,
        review_id=review_id,
        aliases=MappingProxyType(
            {
                "not_stated": ("not stated",),
                "not_applicable": ("N/A", "NA", "not applicable"),
                "invalid_numeric": ("invalid_numeric", "not_parseable"),
            }
        ),
    )


def _missing_state_maps(
    canonical_fields: Iterable[str],
    *,
    map_version: str,
    review_id: str,
) -> Mapping[str, ReviewedLookupTable]:
    return MappingProxyType(
        {
            field_name: _missing_state_lookup(
                map_version=map_version,
                review_id=review_id,
            )
            for field_name in dict.fromkeys(canonical_fields)
        }
    )


def _generic_dispositions(
    source: IngestedSource,
    fields: Mapping[str, int],
    *,
    held_family: str,
) -> tuple[PhysicalColumnDisposition, ...]:
    canonical_by_position = {position: name for name, position in fields.items()}
    return tuple(
        PhysicalColumnDisposition(
            position=column.position,
            role=(
                "canonical"
                if column.position in canonical_by_position
                else "held"
            ),
            canonical_field=canonical_by_position.get(column.position),
            variable_family=(
                None if column.position in canonical_by_position else held_family
            ),
        )
        for column in source.columns
    )


def _core_source_map(source: IngestedSource, config: Any) -> ReviewedSourceMap:
    schema_fields = config.raw["schema"]["fields"]
    fields = MappingProxyType(
        {
            name: int(field["position"])
            for name, field in schema_fields.items()
        }
    )
    normalization_version = "core-missing-states-v1"
    canonical_fields = tuple(fields)
    return ReviewedSourceMap(
        source_name=source.source_name,
        map_version="core-trial-runtime-map-v1",
        review_id=_CORE_MAP_REVIEW_ID,
        source_sha256=source.source_sha256,
        encoding=source.source_encoding,
        workbook_csv_basis=source.workbook_csv_basis,
        fields=fields,
        expected_headers=MappingProxyType(
            {
                int(field["position"]): str(field["header"])
                for field in schema_fields.values()
            }
        ),
        dispositions=_generic_dispositions(
            source,
            fields,
            held_family="core_preserved_noncanonical",
        ),
        # The core extraction contains mixed literature reporting grains.  Its
        # mapping and units are reviewed, but its representation grain remains
        # explicit instead of being upgraded to a treatment mean by assumption.
        representation_basis="unclear_mixed_scope",
        representation_basis_status="review_required",
        fill_down_headers=tuple(config.fill_down_fields),
        normalization_map_version=normalization_version,
        normalization_review_id=_CORE_MAP_REVIEW_ID,
        missing_state_maps=_missing_state_maps(
            canonical_fields,
            map_version=normalization_version,
            review_id=_CORE_MAP_REVIEW_ID,
        ),
        nutrient_unit_controls=_nutrient_controls(
            canonical_fields,
            review_id=_CORE_MAP_REVIEW_ID,
        ),
        yield_precedence="prefer_t_ha",
        yield_precedence_review_id=_YIELD_PRECEDENCE_REVIEW_ID,
    )


def _combined_variable_family(position: int) -> str:
    if position in _COMBINED_IDENTIFIER_POSITIONS:
        return "direct_identifier"
    if position in _COMBINED_PRECISE_LOCATION_POSITIONS:
        return "precise_location"
    if position in _COMBINED_LOCATION_POSITIONS:
        return "detailed_location"
    if position in _COMBINED_DATE_POSITIONS:
        return "date_context"
    if position in _COMBINED_ECONOMIC_POSITIONS:
        return "economic_outcome"
    if position in _COMBINED_SOIL_POSITIONS:
        return "soil_context"
    if position in _COMBINED_CLIMATE_POSITIONS:
        return "climate_context"
    return "preserved_unresolved_semantics"


def _combined_source_value_type(position: int) -> str:
    if position in _COMBINED_IDENTIFIER_POSITIONS:
        return "identifier"
    if position in _COMBINED_DATE_POSITIONS:
        return "date"
    return "mixed"


def _combined_leakage_class(position: int, *, canonical: bool) -> str:
    if position in _COMBINED_IDENTIFIER_POSITIONS or position in {
        *_COMBINED_PRECISE_LOCATION_POSITIONS,
        *_COMBINED_LOCATION_POSITIONS,
    }:
        return "identifier"
    if position in _COMBINED_ECONOMIC_POSITIONS:
        return "economic"
    if position in {103, 104, 148, 149}:
        return "outcome"
    return "approved_predictor" if canonical else "held"


def _combined_dispositions(
    source: IngestedSource,
) -> tuple[PhysicalColumnDisposition, ...]:
    canonical_by_position: dict[int, str] = {
        position: name for name, position in _COMBINED_FIELDS.items()
    }
    for arm in _COMBINED_ARMS:
        for canonical_field, position in arm.field_positions.items():
            canonical_by_position[position] = canonical_field

    dispositions: list[PhysicalColumnDisposition] = []
    for column in source.columns:
        position = column.position
        canonical_field = canonical_by_position.get(position)
        restricted = position in {
            *_COMBINED_IDENTIFIER_POSITIONS,
            *_COMBINED_PRECISE_LOCATION_POSITIONS,
            *_COMBINED_LOCATION_POSITIONS,
        }
        canonical = canonical_field is not None
        dispositions.append(
            PhysicalColumnDisposition(
                position=position,
                role=(
                    "restricted"
                    if restricted
                    else "canonical"
                    if canonical
                    else "held"
                ),
                canonical_field=canonical_field,
                variable_family=(
                    _combined_variable_family(position)
                    if restricted or not canonical
                    else None
                ),
                source_value_type=_combined_source_value_type(position),
                provider_semantics_status=(
                    "verified" if canonical else "unverified"
                ),
                date_conversion_rule=(
                    "preserve raw date text; no implicit date conversion"
                    if position in _COMBINED_DATE_POSITIONS
                    else None
                ),
                leakage_class=_combined_leakage_class(
                    position,
                    canonical=canonical,
                ),
                additional_use_status=(
                    "restricted"
                    if restricted
                    else "canonical_current_scope"
                    if canonical
                    else "held_pending_separate_approval"
                ),
            )
        )
    return tuple(dispositions)


def _combined_source_map(source: IngestedSource) -> ReviewedSourceMap:
    canonical_fields = tuple(
        dict.fromkeys(
            (
                *_COMBINED_FIELDS,
                *(
                    field_name
                    for arm in _COMBINED_ARMS
                    for field_name in arm.field_positions
                ),
            )
        )
    )
    normalization_version = "combined-missing-states-v1"
    return ReviewedSourceMap(
        source_name=source.source_name,
        map_version="ph-combined-runtime-map-v1",
        review_id=_COMBINED_MAP_REVIEW_ID,
        source_sha256=source.source_sha256,
        encoding=source.source_encoding,
        workbook_csv_basis=source.workbook_csv_basis,
        fields=_COMBINED_FIELDS,
        expected_headers=_expected_headers(source),
        dispositions=_combined_dispositions(source),
        representation_basis="unclear_mixed_scope",
        representation_basis_status="review_required",
        arms=_COMBINED_ARMS,
        normalization_map_version=normalization_version,
        normalization_review_id=_COMBINED_MAP_REVIEW_ID,
        missing_state_maps=_missing_state_maps(
            canonical_fields,
            map_version=normalization_version,
            review_id=_COMBINED_MAP_REVIEW_ID,
        ),
        nutrient_unit_controls=_nutrient_controls(
            canonical_fields,
            review_id=_COMBINED_MAP_REVIEW_ID,
        ),
        approved_variable_families=frozenset(
            {
                "direct_identifier",
                "precise_location",
                "detailed_location",
                "date_context",
                "economic_outcome",
                "soil_context",
                "climate_context",
                "preserved_unresolved_semantics",
            }
        ),
        declared_constant_fields=(
            "treatment",
            "treatment_id",
            "inorganic_n_rate",
        ),
        workbook_sha256=_COMBINED_WORKBOOK_SHA256,
        csv_sha256=_COMBINED_CSV_SHA256,
        workbook_csv_reconciliation_review_id=(
            "combined-registered-csv-authoritative-2026-08-13"
        ),
        yield_precedence="prefer_t_ha",
        yield_precedence_review_id=_YIELD_PRECEDENCE_REVIEW_ID,
    )


def _ltcce_source_map(source: IngestedSource) -> ReviewedSourceMap:
    normalization_version = "ltcce-missing-states-v1"
    canonical_fields = tuple(_LTCCE_FIELDS)
    dispositions = _generic_dispositions(
        source,
        _LTCCE_FIELDS,
        held_family="ltcce_preserved_context",
    )
    return ReviewedSourceMap(
        source_name=source.source_name,
        map_version="ltcce-runtime-map-v1",
        review_id=_LTCCE_MAP_REVIEW_ID,
        source_sha256=source.source_sha256,
        encoding=source.source_encoding,
        workbook_csv_basis=source.workbook_csv_basis,
        fields=_LTCCE_FIELDS,
        expected_headers=_expected_headers(source),
        dispositions=dispositions,
        representation_basis="observation_level",
        # The runtime fallback can identify the physical observation grain but
        # cannot supply reviewer/date/rationale evidence. Keep it fail-closed.
        representation_basis_status="review_required",
        arms=(
            SourceArmMap(
                arm_id="mineral_n_rate",
                role="ltcce_nominal_plot_observation",
                field_positions=MappingProxyType({}),
                constants=MappingProxyType(
                    {
                        "treatment": "mineral N rate",
                        "treatment_id": "mineral_n_rate",
                    }
                ),
            ),
        ),
        normalization_map_version=normalization_version,
        normalization_review_id=_LTCCE_MAP_REVIEW_ID,
        missing_state_maps=_missing_state_maps(
            canonical_fields,
            map_version=normalization_version,
            review_id=_LTCCE_MAP_REVIEW_ID,
        ),
        nutrient_unit_controls=_nutrient_controls(
            canonical_fields,
            review_id=_LTCCE_MAP_REVIEW_ID,
        ),
        declared_constant_fields=("treatment", "treatment_id"),
        workbook_sha256=_LTCCE_WORKBOOK_SHA256,
        csv_sha256=_LTCCE_CSV_SHA256,
        workbook_csv_reconciliation_review_id=(
            "ltcce-workbook-csv-cell-fidelity-2026-08-13"
        ),
        yield_precedence="prefer_t_ha",
        yield_precedence_review_id=_YIELD_PRECEDENCE_REVIEW_ID,
    )


def build_builtin_source_maps(
    ingestion: IngestionResult,
    config: Any,
) -> Mapping[str, ReviewedSourceMap]:
    """Build byte-bound runtime maps for the three registered source families.

    These maps implement only the selected deterministic source-policy profile.
    They preserve unresolved scientific questions as missing fields, held column
    families, separate linked arms, or downstream review reasons; they do not
    manufacture provider sign-off or silently broaden source semantics.
    """

    builders = {
        _CORE_SOURCE_NAME: lambda source: _core_source_map(source, config),
        _COMBINED_SOURCE_NAME: _combined_source_map,
        _LTCCE_SOURCE_NAME: _ltcce_source_map,
    }
    unknown_sources = {
        source.source_name for source in ingestion.sources
    } - set(builders)
    if unknown_sources:
        raise ValueError(
            "No built-in runtime source map exists for enabled source(s): "
            + ", ".join(sorted(unknown_sources))
        )
    return MappingProxyType(
        {
            source.source_name: builders[source.source_name](source)
            for source in ingestion.sources
        }
    )


def build_builtin_workbook_reconciliations(
    config: Any,
) -> Mapping[str, WorkbookCsvReconciliation]:
    """Return byte-bound workbook/CSV bases before no-policy ingestion."""

    enabled_sources = {
        str(source_name)
        for source_name in getattr(config, "enabled_sources", ())
    }
    reconciliations = {
        _COMBINED_SOURCE_NAME: WorkbookCsvReconciliation(
            source_name=_COMBINED_SOURCE_NAME,
            workbook_sha256=_COMBINED_WORKBOOK_SHA256,
            csv_sha256=_COMBINED_CSV_SHA256,
            review_id="combined-registered-csv-authoritative-2026-08-13",
            basis="parallel_workbook_csv_reviewed_csv_authoritative",
        ),
        _LTCCE_SOURCE_NAME: WorkbookCsvReconciliation(
            source_name=_LTCCE_SOURCE_NAME,
            workbook_sha256=_LTCCE_WORKBOOK_SHA256,
            csv_sha256=_LTCCE_CSV_SHA256,
            review_id="ltcce-workbook-csv-cell-fidelity-2026-08-13",
            basis="parallel_workbook_csv_verified_equivalent",
        ),
    }
    return MappingProxyType(
        {
            source_name: reconciliation
            for source_name, reconciliation in reconciliations.items()
            if source_name in enabled_sources
        }
    )


def _lookup(
    aliases: Mapping[str, Iterable[str]],
    *,
    map_version: str,
    review_id: str,
) -> ReviewedLookupTable:
    return ReviewedLookupTable(
        map_version=map_version,
        review_id=review_id,
        aliases=MappingProxyType(
            {
                canonical: tuple(dict.fromkeys(values))
                for canonical, values in aliases.items()
            }
        ),
    )


def _identity_lookup_from_observed(
    source: IngestedSource,
    *,
    position: int,
    map_version: str,
    review_id: str,
) -> ReviewedLookupTable:
    canonical_by_token: dict[str, str] = {}
    aliases: dict[str, list[str]] = defaultdict(list)
    for row in source.rows:
        raw = row.raw_cells[position - 1]
        normalized = " ".join(raw.strip().split())
        if not normalized or normalized.casefold() in {
            "not stated",
            "n/a",
            "na",
            "not applicable",
        }:
            continue
        canonical = canonical_by_token.setdefault(normalized.casefold(), normalized)
        aliases[canonical].append(raw)
    return _lookup(
        aliases,
        map_version=map_version,
        review_id=review_id,
    )


def _core_treatment_lookup(
    source: IngestedSource,
    config: Any,
    *,
    review_id: str,
) -> ReviewedLookupTable:
    """Bind numeric N-series labels while retaining all non-dose labels unresolved."""

    aliases: dict[str, list[str]] = {
        canonical: list(values)
        for canonical, values in config.raw["schema"]["normalization"][
            "treatment_class"
        ].items()
    }
    aliases.setdefault("mineral_n_rate", []).append("mineral N rate")
    for row in source.rows:
        raw_label = row.raw_cells[7]
        normalized = " ".join(raw_label.strip().split())
        if not normalized:
            continue
        compact = normalized.casefold().replace(" ", "")
        # Examples present in the reviewed core source include N0/N1/N2/N3,
        # 50N, "30 kg N", and "0 N".  These are direct dose-series labels;
        # compound management descriptions remain unresolved and excluded from
        # fitting until their affected semantics are reviewed.
        if (
            compact in {"n0", "n1", "n2", "n3", "n4", "n5", "n6"}
            or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?n", compact)
            or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?kgn", compact)
        ):
            target = "zero_n" if compact in {"n0", "0n", "0kgn"} else "mineral_n_rate"
            aliases[target].append(raw_label)
    return _lookup(
        aliases,
        map_version="core-treatment-class-v2",
        review_id=review_id,
    )


def build_builtin_source_category_lookups(
    ingestion: IngestionResult,
    config: Any,
) -> Mapping[str, Mapping[str, ReviewedLookupTable]]:
    """Return conservative source-scoped category lookups for policy-less runs."""

    sources = {source.source_name: source for source in ingestion.sources}
    output: dict[str, Mapping[str, ReviewedLookupTable]] = {}

    core = sources.get(_CORE_SOURCE_NAME)
    if core is not None:
        core_review = "source-policy-profile-2026-08-13-core-categories"
        output[_CORE_SOURCE_NAME] = MappingProxyType(
            {
                "water_regime": _lookup(
                    config.raw["schema"]["normalization"]["water_regime"],
                    map_version="core-water-regime-v1",
                    review_id=core_review,
                ),
                "season": _lookup(
                    config.raw["schema"]["normalization"]["season"],
                    map_version="core-season-v1",
                    review_id=core_review,
                ),
                "region": _identity_lookup_from_observed(
                    core,
                    position=16,
                    map_version="core-region-preserving-v1",
                    review_id=core_review,
                ),
                "province": _identity_lookup_from_observed(
                    core,
                    position=17,
                    map_version="core-province-preserving-v1",
                    review_id=core_review,
                ),
                "rice_variety": _identity_lookup_from_observed(
                    core,
                    position=27,
                    map_version="core-variety-preserving-v1",
                    review_id=core_review,
                ),
                "treatment_class": _core_treatment_lookup(
                    core,
                    config,
                    review_id=core_review,
                ),
            }
        )

    combined = sources.get(_COMBINED_SOURCE_NAME)
    if combined is not None:
        combined_review = "source-policy-profile-2026-08-13-combined-categories"
        output[_COMBINED_SOURCE_NAME] = MappingProxyType(
            {
                "water_regime": _lookup(
                    {"irrigated": ("Irrigated",), "rainfed": ("Rainfed",)},
                    map_version="combined-water-regime-v1",
                    review_id=combined_review,
                ),
                "season": _lookup(
                    {"dry": ("Dry season",), "wet": ("Wet season",)},
                    map_version="combined-season-v1",
                    review_id=combined_review,
                ),
                "region": _identity_lookup_from_observed(
                    combined,
                    position=4,
                    map_version="combined-region-preserving-v1",
                    review_id=combined_review,
                ),
                "province": _identity_lookup_from_observed(
                    combined,
                    position=5,
                    map_version="combined-province-preserving-v1",
                    review_id=combined_review,
                ),
                "rice_variety": _identity_lookup_from_observed(
                    combined,
                    position=26,
                    map_version="combined-variety-preserving-v1",
                    review_id=combined_review,
                ),
                "treatment_class": _lookup(
                    config.raw["schema"]["normalization"]["treatment_class"],
                    map_version="combined-treatment-class-v1",
                    review_id=combined_review,
                ),
            }
        )

    ltcce = sources.get(_LTCCE_SOURCE_NAME)
    if ltcce is not None:
        ltcce_review = "source-policy-profile-2026-08-13-ltcce-categories"
        output[_LTCCE_SOURCE_NAME] = MappingProxyType(
            {
                # DS, EWS, and LWS intentionally remain distinct.  No mapping
                # silently merges EWS/LWS into a generic wet season.
                "season": _lookup(
                    {"DS": ("DS",), "EWS": ("EWS",), "LWS": ("LWS",)},
                    map_version="ltcce-season-preserving-v1",
                    review_id=ltcce_review,
                ),
                "rice_variety": _identity_lookup_from_observed(
                    ltcce,
                    position=8,
                    map_version="ltcce-variety-preserving-v1",
                    review_id=ltcce_review,
                ),
                "experimental_design": _identity_lookup_from_observed(
                    ltcce,
                    position=1,
                    map_version="ltcce-design-preserving-v1",
                    review_id=ltcce_review,
                ),
                "treatment_class": _lookup(
                    config.raw["schema"]["normalization"]["treatment_class"],
                    map_version="ltcce-treatment-class-v1",
                    review_id=ltcce_review,
                ),
            }
        )

    return MappingProxyType(output)


__all__ = [
    "build_builtin_source_category_lookups",
    "build_builtin_source_maps",
    "build_builtin_workbook_reconciliations",
]
