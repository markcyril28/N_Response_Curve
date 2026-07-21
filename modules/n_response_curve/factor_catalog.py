from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence


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
    "region": {"fields": ("region",), "type": "categorical", "role": "candidate_explanatory", "unit": None},
    "province": {"fields": ("province",), "type": "categorical", "role": "candidate_explanatory", "unit": None},
    "variety": {"fields": ("rice_variety",), "type": "categorical", "role": "candidate_explanatory", "unit": None},
    "planting_year": {"fields": ("planting_year",), "type": "numeric", "role": "candidate_explanatory", "unit": "year"},
    "recommendation_class": {"fields": ("treatment_text_class",), "type": "categorical", "role": "candidate_explanatory", "unit": None},
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
    "soil_texture": {"fields": ("soil_texture",), "type": "categorical", "role": "candidate_explanatory", "unit": None},
    "soil_ph": {"fields": ("soil_ph",), "type": "numeric", "role": "candidate_explanatory", "unit": "pH"},
    "soil_organic_matter": {"fields": ("soil_organic_matter",), "type": "numeric", "role": "candidate_explanatory", "unit": None},
    "soil_total_n": {"fields": ("soil_total_n",), "type": "numeric", "role": "candidate_explanatory", "unit": None},
    "soil_available_p": {"fields": ("soil_available_p",), "type": "numeric", "role": "candidate_explanatory", "unit": None},
    "soil_exchangeable_k": {"fields": ("soil_exchangeable_k",), "type": "numeric", "role": "candidate_explanatory", "unit": None},
}


KNOWN_FACTORS = frozenset(_FACTOR_METADATA)


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _missing(value: object, data_type: str) -> bool:
    if value is None:
        return True
    if data_type == "numeric":
        return _number(value) is None
    if isinstance(value, str):
        return not value.strip() or value.strip().casefold() in {"na", "n/a", "not stated", "unresolved"}
    return False


def _canonical_value(value: object, data_type: str) -> object:
    if data_type == "numeric":
        parsed = _number(value)
        if parsed is None:
            return None
        return parsed
    if data_type == "boolean":
        return bool(value)
    if isinstance(value, str):
        return " ".join(value.split())
    return str(value)


def factor_value(record: Mapping[str, Any], factor_name: str) -> object:
    """Return a mapped factor value without writing a normalized value into the record."""

    metadata = _FACTOR_METADATA.get(factor_name)
    if metadata is None:
        raise ValueError(f"Unknown explanatory factor: {factor_name}")
    if factor_name == "observed_n_range":
        minimum = _number(record.get("series_observed_n_min_kg_ha"))
        maximum = _number(record.get("series_observed_n_max_kg_ha"))
        return None if minimum is None or maximum is None else maximum - minimum
    if factor_name == "p_varies_with_n":
        constant = record.get("series_p_constant")
        return None if constant is None else not bool(constant)
    if factor_name == "k_varies_with_n":
        constant = record.get("series_k_constant")
        return None if constant is None else not bool(constant)
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
