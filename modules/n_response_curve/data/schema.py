from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Mapping


CANONICAL_N_RATE_UNIT = "kg N ha-1"
CANONICAL_YIELD_UNIT = "t ha-1"
KNOWN_MISSING_STATE_CLASSES = frozenset(
    {
        "not_stated",
        "not_applicable",
        "not_collected",
        "structural_missing",
        "below_detection",
        "invalid_numeric",
        "unresolved_missing",
    }
)
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
_SENSITIVE_PATH_PATTERNS = (
    re.compile(r"(?i)\bfile://[^\s\"']+"),
    re.compile(r"(?i)(?<![\w])(?:[a-z]:[\\/])[^\s\"']+"),
    re.compile(r"(?i)(?<![\w])/(?:mnt/[a-z]|home|users)/[^\s\"']+"),
    re.compile(r"(?i)(?<![\w])~[/\\][^\s\"']+"),
)


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
    conversion: str | None = None
    review_required: bool = False
    review_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CategoryNormalization:
    """One raw category mapped only through a reviewed, versioned lookup."""

    raw_value: str | None
    canonical_value: str | None
    status: str
    map_version: str
    review_id: str


@dataclass(frozen=True)
class ReviewedLookupTable:
    """Explicit category aliases and review evidence."""

    map_version: str
    review_id: str
    aliases: Mapping[str, tuple[str, ...] | list[str]]


def _normalized_text(value: str | None) -> str:
    return (value or "").strip()


def classify_missing(
    value: str | None,
    missing_values: Mapping[str, Any],
    *,
    missing_state_lookup: ReviewedLookupTable | None = None,
) -> str:
    """Classify a raw cell without replacing its original text."""

    normalized = _normalized_text(value)
    if not normalized:
        return "blank"
    lowered = normalized.casefold()
    if missing_state_lookup is not None:
        reviewed_owners = _reviewed_lookup_alias_owners(missing_state_lookup)
        reviewed_state = reviewed_owners.get(_category_token(normalized))
        if reviewed_state is not None:
            return reviewed_state
    if lowered == str(missing_values["not_stated"]).strip().casefold():
        return "unresolved_missing" if missing_state_lookup is not None else "not_stated"
    if lowered in {str(item).strip().casefold() for item in missing_values.get("not_applicable", ())}:
        return "unresolved_missing" if missing_state_lookup is not None else "not_applicable"
    if lowered in {str(item).strip().casefold() for item in missing_values.get("invalid_numeric", ())}:
        return "unresolved_missing" if missing_state_lookup is not None else "invalid_numeric"
    return "present"


def classify_raw_state(value: str | None, missing_values: Mapping[str, Any]) -> str:
    """Distinguish numeric zero from missing states without changing parse behavior."""

    missing_state = classify_missing(value, missing_values)
    if missing_state != "present":
        return missing_state
    parsed = parse_numeric(value, missing_values)
    if parsed.status == "parsed" and parsed.value == 0:
        return "reported_zero"
    return "present"


def parse_numeric(
    value: str | None,
    missing_values: Mapping[str, Any],
    *,
    missing_state_lookup: ReviewedLookupTable | None = None,
) -> NumericParse:
    """Parse locale-light numeric cells while preserving an explicit parse status."""

    missing_state = classify_missing(
        value,
        missing_values,
        missing_state_lookup=missing_state_lookup,
    )
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
    for candidate in (
        "invalid_numeric",
        "unresolved_missing",
        "below_detection",
        "not_collected",
        "structural_missing",
        "not_stated",
        "not_applicable",
        "blank",
    ):
        if candidate in statuses:
            return candidate
    return "blank"


def normalize_yield(
    yield_kg_ha_raw: str | None,
    yield_t_ha_raw: str | None,
    missing_values: Mapping[str, Any],
    *,
    kg_missing_state_lookup: ReviewedLookupTable | None = None,
    t_missing_state_lookup: ReviewedLookupTable | None = None,
    consistency_tolerance_t_ha: float | None = None,
    tolerance_review_id: str | None = None,
) -> YieldNormalization:
    """Convert documented mass units and quarantine unresolved disagreements."""

    if consistency_tolerance_t_ha is not None:
        if consistency_tolerance_t_ha < 0:
            raise ValueError("Yield consistency tolerance cannot be negative")
        if not isinstance(tolerance_review_id, str) or not tolerance_review_id.strip():
            raise ValueError("A yield consistency tolerance requires nonempty review evidence")

    kilogram = parse_numeric(
        yield_kg_ha_raw,
        missing_values,
        missing_state_lookup=kg_missing_state_lookup,
    )
    tonnes = parse_numeric(
        yield_t_ha_raw,
        missing_values,
        missing_state_lookup=t_missing_state_lookup,
    )
    kilogram_value = kilogram.value
    tonnes_value = tonnes.value
    if kilogram_value is not None and tonnes_value is not None:
        kilogram_as_tonnes = kilogram_value / 1000.0
        difference = abs(kilogram_as_tonnes - tonnes_value)
        consistent = (
            difference == 0
            if consistency_tolerance_t_ha is None
            else difference <= consistency_tolerance_t_ha
        )
        if consistent:
            return YieldNormalization(
                tonnes_value,
                "parsed",
                "consistent",
                "both",
                conversion="kg_ha / 1000 == t_ha",
            )
        return YieldNormalization(
            None,
            "parsed",
            "conflict",
            "both",
            conversion="kg_ha / 1000 compared with t_ha",
            review_required=True,
            review_reasons=("YIELD_REPRESENTATION_CONFLICT",),
        )
    missing_statuses = {
        "blank",
        "not_stated",
        "not_applicable",
        "not_collected",
        "structural_missing",
    }
    if kilogram_value is not None and tonnes.status in missing_statuses:
        return YieldNormalization(
            kilogram_value / 1000.0,
            "parsed",
            "kg_converted",
            "kg_ha",
            conversion="kg_ha / 1000",
        )
    if tonnes_value is not None and kilogram.status in missing_statuses:
        return YieldNormalization(tonnes_value, "parsed", "t_provided", "t_ha")
    if kilogram_value is not None or tonnes_value is not None:
        return YieldNormalization(
            None,
            "invalid_numeric",
            "conflict",
            "both",
            review_required=True,
            review_reasons=("YIELD_REPRESENTATION_PARSE_CONFLICT",),
        )
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


def _reviewed_lookup_alias_owners(
    lookup: ReviewedLookupTable,
) -> dict[str, str]:
    if not isinstance(lookup, ReviewedLookupTable):
        raise ValueError("Reviewed category lookup must use the expected lookup type")
    if not isinstance(lookup.map_version, str) or not lookup.map_version.strip():
        raise ValueError("A category lookup requires a nonempty map version")
    if not isinstance(lookup.review_id, str) or not lookup.review_id.strip():
        raise ValueError("A category lookup requires nonempty review evidence")
    if not isinstance(lookup.aliases, Mapping) or not lookup.aliases:
        raise ValueError("A category lookup requires at least one canonical category")

    owners: dict[str, str] = {}
    for canonical, raw_values in lookup.aliases.items():
        canonical_name = str(canonical).strip()
        if not canonical_name:
            raise ValueError("Category lookup canonical values must be nonempty")
        if (
            isinstance(raw_values, (str, bytes))
            or not isinstance(raw_values, (list, tuple))
            or not raw_values
        ):
            raise ValueError(
                f"Category lookup aliases for {canonical_name!r} must be a nonempty sequence"
            )
        for raw in raw_values:
            if not isinstance(raw, str):
                raise ValueError("Category lookup aliases must be strings")
            alias = _category_token(raw)
            if not alias:
                raise ValueError("Category lookup aliases must be nonempty")
            owner = owners.setdefault(alias, canonical_name)
            if owner != canonical_name:
                raise ValueError(
                    f"Category alias {raw!r} belongs to multiple canonical values"
                )
    return owners


def validate_reviewed_lookup_table(lookup: ReviewedLookupTable) -> None:
    """Validate all reviewed aliases even when no observed row exercises them."""

    _reviewed_lookup_alias_owners(lookup)


def validate_reviewed_missing_state_table(lookup: ReviewedLookupTable) -> None:
    """Restrict field-specific maps to the approved missing-state vocabulary."""

    validate_reviewed_lookup_table(lookup)
    unknown_states = set(lookup.aliases) - KNOWN_MISSING_STATE_CLASSES
    if unknown_states:
        raise ValueError(
            "Missing-state lookup contains unsupported canonical state(s): "
            + ", ".join(sorted(unknown_states))
        )


def normalize_category_with_evidence(
    value: str | None,
    lookup: ReviewedLookupTable,
) -> CategoryNormalization:
    """Apply a lookup only when its version and review evidence are explicit."""

    owners = _reviewed_lookup_alias_owners(lookup)
    normalized = _category_token(value)
    if not normalized:
        return CategoryNormalization(
            raw_value=value,
            canonical_value=None,
            status="unresolved_missing",
            map_version=lookup.map_version,
            review_id=lookup.review_id,
        )
    canonical = owners.get(normalized)
    return CategoryNormalization(
        raw_value=value,
        canonical_value=canonical,
        status="mapped_reviewed" if canonical is not None else "unresolved_unmapped",
        map_version=lookup.map_version,
        review_id=lookup.review_id,
    )


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
    """Combine aliases and nutrient evidence while retaining every contradiction."""

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
    review_reasons: list[str] = []
    if treatment_text_class == "unresolved":
        review_reasons.append("UNKNOWN_TREATMENT_ALIAS")
    if treatment_text_class == "absolute_control" and not is_absolute_control:
        review_reasons.append("ABSOLUTE_CONTROL_NUMERIC_CONTRADICTION")
    if treatment_text_class == "zero_n" and not is_zero_n:
        review_reasons.append("ZERO_N_ALIAS_NUMERIC_CONTRADICTION")
    if treatment_text_class in {"RCM", "FP", "NOPT_NPK"} and (
        n_rate is None or is_zero_n
    ):
        review_reasons.append("NONZERO_TREATMENT_ALIAS_N_RATE_CONTRADICTION")
    if is_zero_n and not has_complete_pk:
        review_reasons.append("ZERO_N_PK_COMPOSITION_UNRESOLVED")
    if n_rate is None:
        review_reasons.append("N_RATE_UNRESOLVED")
    if is_absolute_control:
        canonical_treatment_class = "absolute_control"
    elif is_zero_n_with_pk:
        canonical_treatment_class = "zero_n"
    else:
        canonical_treatment_class = treatment_text_class
    review_reasons = sorted(set(review_reasons))
    return {
        "treatment_text_class": treatment_text_class,
        "canonical_treatment_class": canonical_treatment_class,
        "treatment_classification_status": (
            "review_required" if review_reasons else "resolved"
        ),
        "treatment_review_reasons": tuple(review_reasons),
        "treatment_fit_role": (
            "review"
            if review_reasons
            else "comparison_only"
            if treatment_text_class == "FP"
            else "baseline_only"
            if treatment_text_class == "absolute_control"
            else "curve_candidate"
            if treatment_text_class in {
                "zero_n",
                "mineral_n_rate",
                "RCM",
                "NOPT_NPK",
            }
            else "held_out"
        ),
        "nutrient_control_class": nutrient_control_class,
        "is_zero_n": is_zero_n,
        "is_zero_n_with_pk": is_zero_n_with_pk,
        "is_absolute_control": is_absolute_control,
        "is_high_n": n_rate is not None and n_rate > high_n_threshold,
        "organic_fertilizer_present": _is_present_indicator(organic_raw, missing_values),
        "biofertilizer_present": _is_present_indicator(bio_raw, missing_values),
    }


def sensitive_path_alias(value: str | None) -> str | None:
    """Return a stable public-safe alias when a cell contains a local path."""

    if value is None or not any(pattern.search(value) for pattern in _SENSITIVE_PATH_PATTERNS):
        return None
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"restricted_path_{digest}"


__all__ = [
    "CANONICAL_N_RATE_UNIT",
    "CANONICAL_YIELD_UNIT",
    "CategoryNormalization",
    "NumericParse",
    "ReviewedLookupTable",
    "YieldNormalization",
    "canonicalize_irri",
    "classify_raw_state",
    "classify_missing",
    "canonical_unit",
    "classify_experiment_priority",
    "classify_treatment",
    "normalize_category",
    "normalize_category_with_evidence",
    "normalize_country_code",
    "normalize_yield",
    "parse_numeric",
    "sensitive_path_alias",
    "validate_reviewed_lookup_table",
]
