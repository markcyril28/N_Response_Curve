from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Mapping


CANONICAL_N_RATE_UNIT = "kg N ha-1"
CANONICAL_YIELD_UNIT = "t ha-1"
_IRRI_TOKEN = re.compile(r"(?<![\w.])irri(?![\w.])", flags=re.IGNORECASE)
_UNIT_ALIASES = {
    "n_rate": {
        "kg n ha-1": CANONICAL_N_RATE_UNIT,
        "kg n/ha": CANONICAL_N_RATE_UNIT,
        "kg n ha^-1": CANONICAL_N_RATE_UNIT,
    },
    "yield": {
        "t ha-1": CANONICAL_YIELD_UNIT,
        "t/ha": CANONICAL_YIELD_UNIT,
        "tonnes ha-1": CANONICAL_YIELD_UNIT,
    },
}


def canonicalize_irri(value: str | None) -> str | None:
    """Standardize standalone textual references to the IRRI abbreviation.

    URL hostnames are excluded because their conventional representation is
    lowercase, for example ``https://books.irri.org``.
    """

    if value is None:
        return None
    return _IRRI_TOKEN.sub("IRRI", value)


def canonical_unit(value: str | None, quantity: str) -> str | None:
    """Return a canonical analysis unit only for an explicit supported declaration."""

    aliases = _UNIT_ALIASES.get(quantity)
    if aliases is None:
        raise ValueError(f"Unknown unit quantity: {quantity}")
    token = " ".join((value or "").strip().casefold().split())
    return aliases.get(token)


def normalize_country_code(value: str | None) -> str | None:
    """Normalize explicit country evidence without guessing unknown country names."""

    token = " ".join((value or "").strip().casefold().split())
    if not token or token in {"unresolved", "not stated", "n/a", "na"}:
        return None
    if token in {"ph", "phl", "philippines", "republic of the philippines"}:
        return "PH"
    if re.fullmatch(r"[a-z]{2}", token):
        return token.upper()
    return None


def classify_experiment_priority(
    experiment_type_raw: str | None,
    experimental_design_raw: str | None,
) -> str:
    """Expose priority evidence while retaining standard and unresolved trials."""

    evidence = " ".join(
        part.strip().casefold()
        for part in (experiment_type_raw or "", experimental_design_raw or "")
        if part.strip()
    )
    if not evidence:
        return "unresolved"
    priority_tokens = (
        "long term", "long-term", "fertilizer response", "fertiliser response",
        "nitrogen response", "n response",
    )
    return "priority" if any(token in evidence for token in priority_tokens) else "standard"


@dataclass(frozen=True)
class NumericParse:
    value: float | None
    status: str


@dataclass(frozen=True)
class YieldNormalization:
    yield_t_ha: float | None
    parse_status: str
    unit_status: str
    source_unit: str | None


def _normalized_text(value: str | None) -> str:
    return (value or "").strip()


def classify_missing(value: str | None, missing_values: Mapping[str, Any]) -> str:
    """Classify a raw cell without replacing its original text."""

    normalized = _normalized_text(value)
    if not normalized:
        return "blank"
    lowered = normalized.casefold()
    if lowered == str(missing_values["not_stated"]).strip().casefold():
        return "not_stated"
    if lowered in {str(item).strip().casefold() for item in missing_values.get("not_applicable", ())}:
        return "not_applicable"
    if lowered in {str(item).strip().casefold() for item in missing_values.get("invalid_numeric", ())}:
        return "invalid_numeric"
    return "present"


def parse_numeric(value: str | None, missing_values: Mapping[str, Any]) -> NumericParse:
    """Parse locale-light numeric cells while preserving an explicit parse status."""

    missing_state = classify_missing(value, missing_values)
    if missing_state != "present":
        return NumericParse(value=None, status=missing_state)
    normalized = _normalized_text(value).replace("−", "-").replace("–", "-")
    if "," in normalized and re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", normalized) is None:
        return NumericParse(value=None, status="invalid_numeric")
    normalized = normalized.replace(",", "")
    try:
        parsed = float(normalized)
    except ValueError:
        return NumericParse(value=None, status="invalid_numeric")
    if not math.isfinite(parsed):
        return NumericParse(value=None, status="invalid_numeric")
    return NumericParse(value=parsed, status="parsed")


def _combined_missing_status(*statuses: str) -> str:
    for candidate in ("invalid_numeric", "not_stated", "not_applicable", "blank"):
        if candidate in statuses:
            return candidate
    return "blank"


def normalize_yield(
    yield_kg_ha_raw: str | None,
    yield_t_ha_raw: str | None,
    missing_values: Mapping[str, Any],
    *,
    consistency_tolerance_t_ha: float = 0.01,
) -> YieldNormalization:
    """Prefer stated t/ha while retaining kg/ha conversion and conflict evidence."""

    kilogram = parse_numeric(yield_kg_ha_raw, missing_values)
    tonnes = parse_numeric(yield_t_ha_raw, missing_values)
    kilogram_value = kilogram.value
    tonnes_value = tonnes.value
    if kilogram_value is not None and tonnes_value is not None:
        kilogram_as_tonnes = kilogram_value / 1000.0
        if abs(kilogram_as_tonnes - tonnes_value) <= consistency_tolerance_t_ha:
            return YieldNormalization(tonnes_value, "parsed", "consistent", "both")
        return YieldNormalization(None, "parsed", "conflict", "both")
    if kilogram_value is not None and tonnes.status in {"blank", "not_stated", "not_applicable"}:
        return YieldNormalization(kilogram_value / 1000.0, "parsed", "kg_converted", "kg_ha")
    if tonnes_value is not None and kilogram.status in {"blank", "not_stated", "not_applicable"}:
        return YieldNormalization(tonnes_value, "parsed", "t_provided", "t_ha")
    if kilogram_value is not None or tonnes_value is not None:
        return YieldNormalization(None, "invalid_numeric", "conflict", "both")
    return YieldNormalization(
        None,
        _combined_missing_status(kilogram.status, tonnes.status),
        "missing",
        None,
    )


def _category_token(value: str | None) -> str:
    return " ".join(_normalized_text(value).casefold().split())


def normalize_category(value: str | None, mapping: Mapping[str, list[str] | tuple[str, ...]]) -> str:
    """Return a configured category or ``unresolved`` without erasing raw text."""

    normalized = _category_token(value)
    if not normalized:
        return "unresolved"
    for canonical, raw_values in mapping.items():
        if normalized in {_category_token(str(raw)) for raw in raw_values}:
            return canonical
    return "unresolved"


def _text_treatment_class(treatment_raw: str | None, treatment_mapping: Mapping[str, list[str] | tuple[str, ...]]) -> str:
    normalized = _normalized_text(treatment_raw).casefold()
    if not normalized:
        return "unresolved"
    candidates: list[tuple[int, str]] = []
    for canonical, raw_values in treatment_mapping.items():
        for raw in raw_values:
            token = str(raw).strip().casefold()
            if token and re.search(rf"(?<!\w){re.escape(token)}(?!\w)", normalized):
                candidates.append((len(token), canonical))
    if not candidates:
        return "unresolved"
    return max(candidates, key=lambda item: (item[0], item[1]))[1]


def _is_present_indicator(raw: str | None, missing_values: Mapping[str, Any]) -> bool:
    if classify_missing(raw, missing_values) != "present":
        return False
    return _normalized_text(raw).casefold() not in {"n", "no", "false", "0", "none"}


def classify_treatment(
    *,
    treatment_raw: str | None,
    n_rate: float | None,
    p_rate: float | None,
    k_rate: float | None,
    organic_raw: str | None,
    bio_raw: str | None,
    treatment_mapping: Mapping[str, list[str] | tuple[str, ...]],
    missing_values: Mapping[str, Any],
    high_n_threshold: float,
    zero_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Keep textual treatment labels and nutrient-derived treatment facts independent."""

    is_zero_n = n_rate is not None and abs(n_rate) <= zero_tolerance
    has_complete_pk = p_rate is not None and k_rate is not None
    is_absolute_control = is_zero_n and has_complete_pk and abs(p_rate) <= zero_tolerance and abs(k_rate) <= zero_tolerance
    is_zero_n_with_pk = is_zero_n and ((p_rate is not None and p_rate > zero_tolerance) or (k_rate is not None and k_rate > zero_tolerance))
    if is_absolute_control:
        nutrient_control_class = "absolute_control"
    elif is_zero_n_with_pk:
        nutrient_control_class = "zero_n_with_pk"
    elif is_zero_n:
        nutrient_control_class = "zero_n_nutrient_context_unresolved"
    elif n_rate is None:
        nutrient_control_class = "n_rate_unresolved"
    else:
        nutrient_control_class = "nonzero_n"

    treatment_text_class = _text_treatment_class(treatment_raw, treatment_mapping)
    return {
        "treatment_text_class": treatment_text_class,
        "treatment_fit_role": "comparison_only" if treatment_text_class == "FP" else "curve_candidate",
        "nutrient_control_class": nutrient_control_class,
        "is_zero_n": is_zero_n,
        "is_zero_n_with_pk": is_zero_n_with_pk,
        "is_absolute_control": is_absolute_control,
        "is_high_n": n_rate is not None and n_rate > high_n_threshold,
        "organic_fertilizer_present": _is_present_indicator(organic_raw, missing_values),
        "biofertilizer_present": _is_present_indicator(bio_raw, missing_values),
    }


__all__ = [
    "CANONICAL_N_RATE_UNIT",
    "CANONICAL_YIELD_UNIT",
    "NumericParse",
    "YieldNormalization",
    "canonicalize_irri",
    "classify_missing",
    "canonical_unit",
    "classify_experiment_priority",
    "classify_treatment",
    "normalize_category",
    "normalize_country_code",
    "normalize_yield",
    "parse_numeric",
]
