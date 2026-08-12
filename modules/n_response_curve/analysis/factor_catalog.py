from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .values import finite_number, is_missing_text


@dataclass(frozen=True)
class FactorCatalogEntry:
    """Coverage and role metadata for one configured explanatory factor."""

    factor_name: str
    source_fields: tuple[str, ...]
    role: str
    data_type: str
    unit: str | None
    mapping: str
    coverage_count: int
    missing_count: int
    cardinality: int
    leakage_restricted: bool


_FACTOR_METADATA: Mapping[str, Mapping[str, Any]] = {
    "source_family": {"fields": ("source_name",), "type": "categorical", "role": "candidate_explanatory", "unit": None},
    "experiment_type": {"fields": ("experiment_type",), "type": "categorical", "role": "candidate_explanatory", "unit": None},
    "experimental_design": {"fields": ("experimental_design",), "type": "categorical", "role": "candidate_explanatory", "unit": None},
    "water_regime": {"fields": ("water_regime_normalized",), "type": "categorical", "role": "candidate_explanatory", "unit": None},
    "season": {"fields": ("season_normalized",), "type": "categorical", "role": "candidate_explanatory", "unit": None},
    "region": {"fields": ("region_normalized", "region"), "type": "categorical", "role": "candidate_explanatory", "unit": None},
    "province": {"fields": ("province_normalized", "province"), "type": "categorical", "role": "candidate_explanatory", "unit": None},
    "variety": {"fields": ("rice_variety_normalized", "rice_variety"), "type": "categorical", "role": "candidate_explanatory", "unit": None},
    "planting_year": {"fields": ("planting_year",), "type": "numeric", "role": "candidate_explanatory", "unit": "year"},
    "recommendation_class": {"fields": ("treatment_text_class",), "type": "categorical", "role": "candidate_explanatory", "unit": None},
    "recommendation_scope": {"fields": ("recommendation_scope",), "type": "categorical", "role": "candidate_explanatory", "unit": None},
    "n_level_count": {"fields": ("series_distinct_n_level_count",), "type": "numeric", "role": "design_or_selection", "unit": "count", "leakage": True},
    "observed_n_range": {"fields": ("series_observed_n_min_kg_ha", "series_observed_n_max_kg_ha"), "type": "numeric", "role": "design_or_selection", "unit": "kg N/ha", "leakage": True},
    "has_zero_n": {"fields": ("series_has_zero_n",), "type": "boolean", "role": "design_or_selection", "unit": None, "leakage": True},
    "has_high_n": {"fields": ("series_has_high_n", "is_high_n"), "type": "boolean", "role": "design_or_selection", "unit": None, "leakage": True},
    "p_rate": {"fields": ("p_rate_kg_p2o5_ha",), "type": "numeric", "role": "candidate_explanatory", "unit": "kg P2O5/ha"},
    "p_varies_with_n": {"fields": ("series_p_constant",), "type": "boolean", "role": "candidate_explanatory", "unit": None},
    "k_rate": {"fields": ("k_rate_kg_k2o_ha",), "type": "numeric", "role": "candidate_explanatory", "unit": "kg K2O/ha"},
    "k_varies_with_n": {"fields": ("series_k_constant",), "type": "boolean", "role": "candidate_explanatory", "unit": None},
    "organic_fertilizer_present": {"fields": ("organic_fertilizer_present",), "type": "boolean", "role": "candidate_explanatory", "unit": None},
    "biofertilizer_present": {"fields": ("biofertilizer_present",), "type": "boolean", "role": "candidate_explanatory", "unit": None},
    "n_split_pattern": {"fields": ("n_split",), "type": "categorical", "role": "candidate_explanatory", "unit": None},
    "n_timing_pattern": {"fields": ("n_timing_pattern",), "type": "categorical", "role": "candidate_explanatory", "unit": None},
    "elevation_m": {"fields": ("elevation_m",), "type": "numeric", "role": "candidate_explanatory", "unit": "m"},
    "soil_texture": {"fields": ("soil_texture_normalized", "soil_texture"), "type": "categorical", "role": "candidate_explanatory", "unit": None},
    "soil_ph": {"fields": ("soil_ph",), "type": "numeric", "role": "candidate_explanatory", "unit": "pH"},
    "soil_organic_matter": {"fields": ("soil_organic_matter",), "type": "numeric", "role": "candidate_explanatory", "unit": None},
    "soil_total_n": {"fields": ("soil_total_n",), "type": "numeric", "role": "candidate_explanatory", "unit": None},
    "soil_available_p": {"fields": ("soil_available_p",), "type": "numeric", "role": "candidate_explanatory", "unit": None},
    "soil_exchangeable_k": {"fields": ("soil_exchangeable_k",), "type": "numeric", "role": "candidate_explanatory", "unit": None},
}


KNOWN_FACTORS = frozenset(_FACTOR_METADATA)
_REPRESENTATION_DATA_TYPE_ALIASES = {
    "continuous_numeric": "numeric",
    "categorical_factor": "categorical",
    "ordered_factor": "categorical",
}


def _missing(value: object, data_type: str) -> bool:
    if value is None:
        return True
    if data_type == "numeric":
        return finite_number(value) is None
    if isinstance(value, str):
        return is_missing_text(value)
    return False


def _canonical_value(value: object, data_type: str) -> object:
    if data_type == "numeric":
        parsed = finite_number(value)
        if parsed is None:
            return None
        return parsed
    if data_type == "boolean":
        return bool(value)
    if isinstance(value, str):
        return " ".join(value.split())
    return str(value)


def _reviewed_factor_value(
    record: Mapping[str, Any],
    factor_name: str,
    representation: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> object:
    if representation.get("factor_name") not in {None, factor_name}:
        raise ValueError("Reviewed factor representation is bound to another factor")
    source_fields = representation.get("source_fields")
    if (
        not isinstance(source_fields, (list, tuple))
        or not source_fields
        or any(not isinstance(field, str) or not field for field in source_fields)
    ):
        raise ValueError("Reviewed factor representation requires nonempty source fields")
    if not set(source_fields).issubset(set(metadata["fields"])):
        raise ValueError("Reviewed factor representation uses an unregistered source field")
    leakage_exclusions = representation.get("leakage_exclusions", ())
    if not isinstance(leakage_exclusions, (list, tuple)) or any(
        not isinstance(field, str) or not field for field in leakage_exclusions
    ):
        raise ValueError("Reviewed factor representation leakage exclusions are malformed")
    if set(source_fields) & set(leakage_exclusions):
        raise ValueError("Reviewed factor source field is also a leakage exclusion")
    representation_data_type = str(representation.get("data_type") or "")
    canonical_data_type = _REPRESENTATION_DATA_TYPE_ALIASES.get(
        representation_data_type,
        representation_data_type,
    )
    if canonical_data_type != metadata["type"]:
        raise ValueError("Reviewed factor representation data type conflicts with the catalog")
    if representation.get("unit") != metadata.get("unit"):
        raise ValueError("Reviewed factor representation unit conflicts with the catalog")
    if representation.get("role") != metadata["role"]:
        raise ValueError("Reviewed factor representation role conflicts with the catalog")
    if (
        representation.get("missingness_rule")
        != "factor_outcome_complete_case_no_imputation"
    ):
        raise ValueError(
            "Reviewed factor representation must use "
            "factor_outcome_complete_case_no_imputation"
        )
    if representation.get("learned_within_training_only") is not True:
        raise ValueError(
            "Reviewed factor representation must restrict learned transforms to training"
        )

    transformation = representation.get("transformation")
    raw_values = [record.get(field) for field in source_fields]
    if transformation in {"identity", "direct"}:
        if len(raw_values) != 1:
            raise ValueError("Direct factor representations require exactly one source field")
        return _canonical_value(raw_values[0], str(metadata["type"]))
    if transformation in {"fixed_category_map", "fixed_one_hot"}:
        if len(raw_values) != 1:
            raise ValueError("Fixed categorical representations require one source field")
        category_map = representation.get("category_map")
        if not isinstance(category_map, Mapping) or not category_map:
            raise ValueError("Fixed categorical representation requires category_map")
        raw_value = raw_values[0]
        if raw_value is None:
            return None
        mapped = category_map.get(str(raw_value))
        return None if mapped is None else _canonical_value(mapped, "categorical")
    if transformation == "range_difference":
        if len(raw_values) != 2:
            raise ValueError("Range-difference representation requires two source fields")
        minimum = finite_number(raw_values[0])
        maximum = finite_number(raw_values[1])
        return None if minimum is None or maximum is None else maximum - minimum
    if transformation == "logical_not":
        if len(raw_values) != 1 or raw_values[0] is None:
            return None
        return not bool(raw_values[0])
    if transformation == "normalized_token":
        if len(raw_values) != 1 or raw_values[0] is None:
            return None
        token = re.sub(
            r"[^a-z0-9]+",
            "_",
            str(raw_values[0]).strip().casefold(),
        ).strip("_")
        return token or None
    raise ValueError(
        f"Reviewed factor representation uses unsupported transformation: {transformation!r}"
    )


def factor_value(
    record: Mapping[str, Any],
    factor_name: str,
    *,
    representation: Mapping[str, Any] | None = None,
) -> object:
    """Return a mapped factor value without writing a normalized value into the record."""

    metadata = _FACTOR_METADATA.get(factor_name)
    if metadata is None:
        raise ValueError(f"Unknown explanatory factor: {factor_name}")
    if representation is not None:
        return _reviewed_factor_value(
            record,
            factor_name,
            representation,
            metadata,
        )
    if factor_name == "observed_n_range":
        minimum = finite_number(record.get("series_observed_n_min_kg_ha"))
        maximum = finite_number(record.get("series_observed_n_max_kg_ha"))
        return None if minimum is None or maximum is None else maximum - minimum
    if factor_name == "p_varies_with_n":
        constant = record.get("series_p_constant")
        return None if constant is None else not bool(constant)
    if factor_name == "k_varies_with_n":
        constant = record.get("series_k_constant")
        return None if constant is None else not bool(constant)
    if factor_name == "recommendation_scope":
        raw_scope = record.get("recommendation_scope")
        if raw_scope is None:
            return None
        normalized_scope = re.sub(r"[^a-z0-9]+", "_", str(raw_scope).strip().casefold()).strip("_")
        return normalized_scope or None
    for field in metadata["fields"]:
        if field in record and record[field] is not None:
            return record[field]
    return None


def build_factor_catalog(
    records: Iterable[Mapping[str, Any]],
    *,
    factor_names: Sequence[str],
) -> tuple[FactorCatalogEntry, ...]:
    """Inventory configured explanatory fields and make leakage restrictions explicit."""

    requested = tuple(factor_names)
    unknown = set(requested) - KNOWN_FACTORS
    if unknown:
        raise ValueError(f"Unknown explanatory factor(s): {', '.join(sorted(unknown))}")
    if len(requested) != len(set(requested)):
        raise ValueError("Explanatory factor names must be unique")
    rows = tuple(dict(record) for record in records)
    entries: list[FactorCatalogEntry] = []
    for factor_name in requested:
        metadata = _FACTOR_METADATA[factor_name]
        data_type = str(metadata["type"])
        values = [factor_value(record, factor_name) for record in rows]
        observed = [_canonical_value(value, data_type) for value in values if not _missing(value, data_type)]
        entries.append(
            FactorCatalogEntry(
                factor_name=factor_name,
                source_fields=tuple(metadata["fields"]),
                role=str(metadata["role"]),
                data_type=data_type,
                unit=metadata["unit"],
                mapping="direct" if factor_name not in {"observed_n_range", "p_varies_with_n", "k_varies_with_n"} else "derived_without_mutation",
                coverage_count=len(observed),
                missing_count=len(rows) - len(observed),
                cardinality=len(set(observed)),
                leakage_restricted=bool(metadata.get("leakage", False)),
            )
        )
    return tuple(entries)


__all__ = ["KNOWN_FACTORS", "FactorCatalogEntry", "build_factor_catalog", "factor_value"]
