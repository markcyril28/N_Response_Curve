from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tomllib
from types import MappingProxyType
from typing import Any, Mapping

from ..contracts import (
    SUPPORTED_FIGURE_FORMATS as _SUPPORTED_FIGURE_FORMATS,
    SUPPORTED_TABLE_FORMATS as _SUPPORTED_TABLE_FORMATS,
)
from .schema import canonical_unit


class ConfigError(ValueError):
    """Raised when the operator configuration is unsafe or incoherent."""


RUN_MODES = {"validate", "test", "full"}
KNOWN_MODELS = {"linear", "quadratic", "linear_plateau", "quadratic_plateau", "mitscherlich"}
KNOWN_SOURCE_TYPES = {"literature", "ltcce", "rcm_validation", "nopt", "future"}
KNOWN_SOURCE_AVAILABILITY = {"available", "expected_unavailable"}
KNOWN_SOURCE_CONFIRMATION_STATUSES = {"verified", "pending"}
KNOWN_COMPARISON_DIMENSIONS = {"water_regime", "season", "region", "province", "variety", "recommendation_class"}
KNOWN_SERIES_IDENTITY_DIMENSIONS = {
    "water_regime", "season", "region", "province", "variety", "planting_year",
    "experiment_type", "experimental_design",
}
KNOWN_OUTPUT_FORMATS = set(_SUPPORTED_TABLE_FORMATS)
KNOWN_FIGURE_FORMATS = set(_SUPPORTED_FIGURE_FORMATS)
KNOWN_TREATMENT_CLASSES = {"zero_n", "absolute_control", "RCM", "FP", "NOPT_NPK", "other", "unresolved"}
KNOWN_FILL_DOWN_FIELDS = {"Study_ID", "Trial ID", "Source", "Author(s)", "Year of Publication"}
KNOWN_CRITICAL_ERROR_CODES = {
    "YIELD_UNIT_CONFLICT", "N_RATE_UNIT_CONFLICT", "N_RATE_OUT_OF_RANGE",
    "YIELD_OUT_OF_RANGE",
}
KNOWN_DATASET_VERSIONS = {f"D{i:02d}_{name}" for i, name in enumerate(
    (
        "inventory_all",
        "strict_primary_zero_n",
        "strict_primary_zero_optional",
        "primary_4plus_n_levels",
        "primary_5plus_n_levels",
        "pk_varying_sensitivity",
        "organic_bio_sensitivity",
        "high_n_full_range",
        "high_n_trimmed_sensitivity",
        "complete_recommendation_set",
        "factor_specific_complete_case",
        "balanced_interaction_cells",
        "climate_enriched_future",
    )
)}
KNOWN_SOURCE_COMBINATION_MODES = {
    "each_family_alone",
    "all_nonempty_family_subsets",
    "all_families_deduplicated",
    "leave_one_family_out",
}
KNOWN_CURVE_OUTCOMES = {
    "curve_shape_class",
    "evidence_status",
    "evidence_strength",
    "optimum_status",
    "agronomic_optimum_n_kg_ha",
    "economic_optimum_n_kg_ha",
    "plateau_onset_n_kg_ha",
    "predicted_max_yield_t_ha",
    "predicted_observed_domain_peak_yield_t_ha",
    "finite_maximum_yield_t_ha",
    "fitted_asymptote_yield_t_ha",
    "supported_max_yield_t_ha",
    "maximum_reference_basis",
    "maximum_proximity_status",
    "observed_max_yield_t_ha",
    "observed_max_gap_to_finite_maximum_t_ha",
    "observed_max_gap_to_supported_maximum_t_ha",
    "observed_max_attainment_fraction",
    "yield_at_zero_n_t_ha",
    "yield_response_above_zero_n_t_ha",
    "recommendation_yield_gap_t_ha",
    "target_yield_gap_t_ha",
    "target_yield_status",
}
KNOWN_FACTORS = {
    "source_family",
    "experiment_type",
    "experimental_design",
    "water_regime",
    "season",
    "region",
    "province",
    "variety",
    "planting_year",
    "recommendation_class",
    "recommendation_scope",
    "n_level_count",
    "observed_n_range",
    "has_zero_n",
    "has_high_n",
    "p_rate",
    "p_varies_with_n",
    "k_rate",
    "k_varies_with_n",
    "organic_fertilizer_present",
    "biofertilizer_present",
    "n_split_pattern",
    "n_timing_pattern",
    "elevation_m",
    "soil_texture",
    "soil_ph",
    "soil_organic_matter",
    "soil_total_n",
    "soil_available_p",
    "soil_exchangeable_k",
}
KNOWN_ANALYSIS_FAMILIES = {
    "coverage_and_missingness",
    "one_factor_descriptive",
    "one_factor_inferential",
    "all_supported_interactions",
    "multivariable_mixed_effects",
    "observation_level_curve_modification",
    "penalized_predictive_models",
    "curve_feature_clustering",
    "dataset_and_source_robustness",
    "marginal_contrasts",
}
KNOWN_ENGINE_RESPONSIBILITIES = KNOWN_ANALYSIS_FAMILIES | {
    "data_pipeline",
    "per_series_curve_fitting",
    "final_reporting",
}
KNOWN_MULTIPLE_TESTING_METHODS = {"benjamini_hochberg", "none"}
KNOWN_MODEL_SELECTION_METRICS = {
    "aicc_then_grouped_prediction",
    "all_credible_no_selection",
}
KNOWN_TIE_BREAKING_RULES = {
    "simpler_model_then_stable_domain",
    "not_applicable",
}
R_OWNED_ANALYSIS_FAMILIES = {
    "one_factor_inferential",
    "all_supported_interactions",
    "multivariable_mixed_effects",
    "observation_level_curve_modification",
    "marginal_contrasts",
}
REQUIRED_PATHS = {
    "core_source_csv",
    "source_workbook",
    "source_manifest",
    "source_checksums",
    "schema_evidence",
    "variety_lookup",
    "analysis_ready_root",
    "qc_root",
    "curves_root",
    "comparisons_root",
    "explanatory_root",
    "reports_root",
    "run_metadata_root",
    "test_output_root",
}
INPUT_PATHS = {
    "core_source_csv",
    "source_workbook",
    "source_manifest",
    "source_checksums",
    "schema_evidence",
    "variety_lookup",
}
OUTPUT_PATHS = {
    "analysis_ready_root",
    "qc_root",
    "curves_root",
    "comparisons_root",
    "explanatory_root",
    "reports_root",
    "run_metadata_root",
    "test_output_root",
}
TOP_LEVEL_SECTIONS = {
    "run",
    "paths",
    "selection",
    "sources",
    "schema",
    "missing_values",
    "eligibility",
    "modeling",
    "outputs",
    "logging",
    "engines",
    "analysis_hypotheses",
    "analysis_matrix",
}


@dataclass(frozen=True)
class ValidatedConfig:
    """Recursively immutable effective configuration used by the runtime."""

    config_path: Path
    project_root: Path
    raw: Mapping[str, Any]
    paths: Mapping[str, Path]
    sources: Mapping[str, Mapping[str, Any]]
    enabled_sources: tuple[str, ...]
    enabled_models: tuple[str, ...]
    comparison_dimensions: tuple[str, ...]
    scope_countries: tuple[str, ...]
    series_identity_dimensions: tuple[str, ...]
    output_formats: tuple[str, ...]
    figure_formats: tuple[str, ...]
    fill_down_fields: tuple[str, ...]
    treatment_classes: tuple[str, ...]
    dataset_versions: tuple[str, ...]
    source_combination_modes: tuple[str, ...]
    curve_outcomes: tuple[str, ...]
    explanatory_factors: tuple[str, ...]
    analysis_families: tuple[str, ...]
    interaction_orders: tuple[int, ...]
    engine_assignments: Mapping[str, str]
    decision_snapshot: Mapping[str, Mapping[str, str | bool]]
    decision_snapshot_hash: str | None
    run_mode: str

    @property
    def writes_outputs(self) -> bool:
        return self.run_mode in {"test", "full"}

    @property
    def enabled_r_families(self) -> tuple[str, ...]:
        return tuple(family for family in self.analysis_families if self.engine_assignments[family] == "r")


def _freeze_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_config_value(nested) for key, nested in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_config_value(item) for item in value)
    return value


def load_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    check_files: bool = True,
    preflight_engines: bool = False,
) -> ValidatedConfig:
    """Read and validate TOML without creating logs, outputs, or directories."""

    path = Path(config_path).expanduser().resolve()
    root = Path(project_root or path.parent).expanduser().resolve()
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file does not exist: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc
    return validate_config(
        data,
        config_path=path,
        project_root=root,
        check_files=check_files,
        preflight_engines=preflight_engines,
    )


def validate_config(
    data: Mapping[str, Any],
    *,
    config_path: str | Path,
    project_root: str | Path,
    check_files: bool = True,
    preflight_engines: bool = False,
) -> ValidatedConfig:
    """Validate a parsed configuration and return only effective operator controls."""

    if not isinstance(data, Mapping):
        raise ConfigError("Top-level TOML value must be a table")
    _check_unknown_keys(data, TOP_LEVEL_SECTIONS, where="top-level")
    root = Path(project_root).expanduser().resolve()
    path = Path(config_path).expanduser().resolve()
    for section in ("run", "paths", "selection", "sources", "schema", "missing_values", "eligibility", "modeling", "outputs", "logging", "engines", "analysis_matrix"):
        if not isinstance(data.get(section), Mapping):
            raise ConfigError(f"Missing or invalid [{section}] table")

    # Freeze effective defaults without mutating a caller-owned mapping.
    data = dict(data)
    data["selection"] = dict(data["selection"])
    data["eligibility"] = dict(data["eligibility"])
    data["sources"] = {
        name: dict(source) if isinstance(source, Mapping) else source
        for name, source in data["sources"].items()
    }
    hypotheses = data.get("analysis_hypotheses", {"specifications": []})
    if not isinstance(hypotheses, Mapping):
        raise ConfigError("[analysis_hypotheses] must be a table")
    hypotheses = dict(hypotheses)
    _check_unknown_keys(hypotheses, {"specifications"}, where="[analysis_hypotheses]")
    specifications = hypotheses.get("specifications", [])
    if not isinstance(specifications, list) or any(not isinstance(item, Mapping) for item in specifications):
        raise ConfigError("[analysis_hypotheses].specifications must be a list of tables")
    hypotheses["specifications"] = [dict(item) for item in specifications]
    data["analysis_hypotheses"] = hypotheses
    selection_defaults = data["selection"]
    selection_defaults.setdefault("scope_countries", ["PH"])
    selection_defaults.setdefault("series_identity_dimensions", ["water_regime", "season"])
    eligibility_defaults = data["eligibility"]
    eligibility_defaults.setdefault(
        "n_level_tolerance_kg_ha",
        eligibility_defaults.get("constant_nutrient_tolerance", 1e-8),
    )
    modeling_defaults = data["modeling"]
    modeling_defaults.setdefault("allow_baseline_response_metrics", False)
    bounds_defaults = modeling_defaults.get("parameter_bounds", {})
    eligibility_defaults.setdefault("n_rate_min_kg_ha", bounds_defaults.get("n_rate_min_kg_ha"))
    eligibility_defaults.setdefault("n_rate_max_kg_ha", bounds_defaults.get("n_rate_max_kg_ha"))
    eligibility_defaults.setdefault("yield_min_t_ha", modeling_defaults.get("plausible_yield_min_t_ha"))
    eligibility_defaults.setdefault("yield_max_t_ha", modeling_defaults.get("plausible_yield_max_t_ha"))

    run = data["run"]
    _check_unknown_keys(
        run,
        {
            "mode", "overwrite", "random_seed", "test_group_limit", "fail_fast", "qc_gate",
            "r_threads_per_job", "max_parallel_r_jobs", "r_stage_timeout_seconds",
            "r_termination_grace_seconds", "cpu_detection", "decisions_approved",
            "required_full_decisions", "required_full_sources",
        },
        where="[run]",
    )
    _require_string(run, "mode", where="[run]")
    mode = run["mode"]
    if mode not in RUN_MODES:
        raise ConfigError(f"[run].mode must be one of {sorted(RUN_MODES)}, got {mode!r}")
    _require_bool(run, "overwrite", where="[run]")
    _require_int(run, "random_seed", where="[run]")
    _require_bool(run, "fail_fast", where="[run]")
    _require_string(run, "qc_gate", where="[run]")
    if run["qc_gate"] not in {"critical_only", "fail_on_any_review"}:
        raise ConfigError("[run].qc_gate is invalid")
    _require_int_at_least(run, "test_group_limit", 1, where="[run]")
    _require_int_at_least(run, "r_threads_per_job", 1, where="[run]")
    _require_int_at_least(run, "max_parallel_r_jobs", 1, where="[run]")
    _require_int_at_least(run, "r_stage_timeout_seconds", 1, where="[run]")
    _require_int_at_least(run, "r_termination_grace_seconds", 0, where="[run]")
    _require_string(run, "cpu_detection", where="[run]")
    if run["cpu_detection"] != "affinity":
        raise ConfigError("[run].cpu_detection must be 'affinity'")
    approved_decisions = _string_list(run.get("decisions_approved", []), where="[run].decisions_approved")
    _check_unique(approved_decisions, where="[run].decisions_approved")
    required_full_decisions = _string_list(
        run.get("required_full_decisions", []),
        where="[run].required_full_decisions",
    )
    _check_unique(required_full_decisions, where="[run].required_full_decisions")
    required_full_sources = _string_list(
        run.get("required_full_sources", []),
        where="[run].required_full_sources",
    )
    _check_unique(required_full_sources, where="[run].required_full_sources")
    decision_snapshot: dict[str, dict[str, str | bool]] = {}
    decision_snapshot_hash: str | None = None
    if mode == "full":
        decision_snapshot, decision_snapshot_hash = _require_full_mode_decision_approvals(
            tuple(approved_decisions),
            tuple(required_full_decisions),
            project_root=root,
        )

    paths = _resolve_paths(data["paths"], root)
    _check_path_overlaps(paths)

    selection = data["selection"]
    _check_unknown_keys(
        selection,
        {
            "enabled_sources", "enabled_models", "comparison_dimensions", "output_formats",
            "figure_formats", "fill_down_fields", "treatment_classes", "scope_countries",
            "series_identity_dimensions",
        },
        where="[selection]",
    )
    sources = data["sources"]
    if not isinstance(sources, Mapping) or not sources:
        raise ConfigError("[sources] must define at least one source")
    enabled_sources = _toggle_list(
        selection, "enabled_sources", set(sources), where="[selection]", allow_empty=False
    )
    enabled_models = _toggle_list(
        selection, "enabled_models", KNOWN_MODELS, where="[selection]", allow_empty=True
    )
    comparison_dimensions = _toggle_list(
        selection,
        "comparison_dimensions",
        KNOWN_COMPARISON_DIMENSIONS,
        where="[selection]",
        allow_empty=True,
    )
    scope_countries = _string_list(selection.get("scope_countries"), where="[selection].scope_countries")
    _check_unique(scope_countries, where="[selection].scope_countries")
    if not scope_countries or any(re.fullmatch(r"[A-Z]{2}", country) is None for country in scope_countries):
        raise ConfigError("[selection].scope_countries must contain unique uppercase ISO alpha-2 codes")
    series_identity_dimensions = _toggle_list(
        selection,
        "series_identity_dimensions",
        KNOWN_SERIES_IDENTITY_DIMENSIONS,
        where="[selection]",
        allow_empty=False,
    )
    output_formats = _toggle_list(
        selection, "output_formats", KNOWN_OUTPUT_FORMATS, where="[selection]", allow_empty=False
    )
    figure_formats = _toggle_list(
        selection, "figure_formats", KNOWN_FIGURE_FORMATS, where="[selection]", allow_empty=False
    )
    fill_down_fields = _toggle_list(
        selection, "fill_down_fields", KNOWN_FILL_DOWN_FIELDS, where="[selection]", allow_empty=True
    )
    treatment_classes = _toggle_list(
        selection,
        "treatment_classes",
        KNOWN_TREATMENT_CLASSES,
        where="[selection]",
        allow_empty=True,
    )

    source_paths: dict[str, Path] = {}
    for source_name, source in sources.items():
        if not isinstance(source, Mapping):
            raise ConfigError(f"[sources.{source_name}] must be a table")
        _check_unknown_keys(
            source,
            {
                "source_type", "data_path", "schema_map", "provider", "provenance_notes",
                "workbook", "sheet", "checksum", "manifest_reference",
                "availability", "confirmation_status", "shape_adapter_version", "country_code",
                "source_family",
            },
            where=f"[sources.{source_name}]",
        )
        for key in (
            "source_type",
            "data_path",
            "schema_map",
            "provider",
            "provenance_notes",
            "availability",
            "confirmation_status",
            "shape_adapter_version",
        ):
            if key not in source:
                raise ConfigError(f"[sources.{source_name}] is missing {key!r}")
        _require_string(source, "source_type", where=f"[sources.{source_name}]")
        if source["source_type"] not in KNOWN_SOURCE_TYPES:
            raise ConfigError(f"[sources.{source_name}].source_type is unknown: {source['source_type']!r}")
        _require_string(source, "availability", where=f"[sources.{source_name}]")
        if source["availability"] not in KNOWN_SOURCE_AVAILABILITY:
            raise ConfigError(
                f"[sources.{source_name}].availability must be one of {sorted(KNOWN_SOURCE_AVAILABILITY)}"
            )
        _require_string(source, "confirmation_status", where=f"[sources.{source_name}]")
        if source["confirmation_status"] not in KNOWN_SOURCE_CONFIRMATION_STATUSES:
            raise ConfigError(
                f"[sources.{source_name}].confirmation_status must be one of "
                f"{sorted(KNOWN_SOURCE_CONFIRMATION_STATUSES)}"
            )
        _require_string(source, "shape_adapter_version", where=f"[sources.{source_name}]")
        source.setdefault("country_code", "PH" if source_name == "core_trial_data" else "UNRESOLVED")
        source.setdefault("source_family", source["source_type"])
        _require_string(source, "country_code", where=f"[sources.{source_name}]")
        if source["country_code"] != "UNRESOLVED" and re.fullmatch(r"[A-Z]{2}", source["country_code"]) is None:
            raise ConfigError(f"[sources.{source_name}].country_code must be an uppercase ISO alpha-2 code or 'UNRESOLVED'")
        _require_string(source, "source_family", where=f"[sources.{source_name}]")
        source_paths[source_name] = _resolve_relative_path(source["data_path"], root, f"[sources.{source_name}].data_path")
        _resolve_relative_path(source["schema_map"], root, f"[sources.{source_name}].schema_map")
        for key in ("provider", "provenance_notes"):
            if not isinstance(source[key], str) or not source[key].strip():
                raise ConfigError(f"[sources.{source_name}].{key} must be a nonempty string")
        for key in ("workbook", "sheet", "checksum", "manifest_reference"):
            if key in source and (not isinstance(source[key], str) or not source[key].strip()):
                raise ConfigError(f"[sources.{source_name}].{key} must be a nonempty string")
    if "core_trial_data" not in sources:
        raise ConfigError("[sources] must define core_trial_data")
    if source_paths["core_trial_data"] != paths["core_source_csv"]:
        raise ConfigError("[sources.core_trial_data].data_path must match [paths].core_source_csv")
    for source_name in enabled_sources:
        if source_name not in sources:
            raise ConfigError(f"[selection].enabled_sources references unknown source: {source_name}")
        source = sources[source_name]
        if source["availability"] != "available":
            raise ConfigError(
                f"[selection].enabled_sources cannot enable {source_name!r} while its availability is "
                f"{source['availability']!r}"
            )
        if check_files:
            _require_file(source_paths[source_name], f"enabled source {source_name}")
            _require_file(_resolve_relative_path(source["schema_map"], root, f"[sources.{source_name}].schema_map"), f"schema map for {source_name}")
    unknown_required_sources = tuple(
        source_name for source_name in required_full_sources if source_name not in sources
    )
    if unknown_required_sources:
        raise ConfigError(
            "[run].required_full_sources references unknown source(s): "
            + ", ".join(unknown_required_sources)
        )
    if mode == "full":
        if not required_full_sources:
            raise ConfigError(
                "full mode requires a nonempty [run].required_full_sources authority gate"
            )
        disabled_required_sources = tuple(
            source_name for source_name in required_full_sources if source_name not in enabled_sources
        )
        if disabled_required_sources:
            raise ConfigError(
                "full mode [run].required_full_sources must all be enabled: "
                + ", ".join(disabled_required_sources)
            )
        unbound_enabled_sources = tuple(
            source_name for source_name in enabled_sources if source_name not in required_full_sources
        )
        if unbound_enabled_sources:
            raise ConfigError(
                "full mode [run].required_full_sources must exactly bind the enabled authoritative "
                "source snapshot; add or disable: "
                + ", ".join(unbound_enabled_sources)
            )
        unverified_required_sources = tuple(
            source_name
            for source_name in required_full_sources
            if sources[source_name]["availability"] != "available"
            or sources[source_name]["confirmation_status"] != "verified"
            or sources[source_name]["shape_adapter_version"] == "unassigned"
        )
        if unverified_required_sources:
            raise ConfigError(
                "full mode [run].required_full_sources must be source-verified with an assigned adapter: "
                + ", ".join(unverified_required_sources)
            )
    if check_files:
        for key in INPUT_PATHS:
            _require_file(paths[key], f"[paths].{key}")

    schema = data["schema"]
    _check_unknown_keys(
        schema,
        {"expected_physical_columns", "fields", "normalization", "units"},
        where="[schema]",
    )
    _require_int_at_least(schema, "expected_physical_columns", 1, where="[schema]")
    fields = schema.get("fields")
    if not isinstance(fields, Mapping) or not fields:
        raise ConfigError("[schema.fields] must define at least one canonical field")
    positions: list[int] = []
    for field_name, field in fields.items():
        if not isinstance(field, Mapping):
            raise ConfigError(f"[schema.fields.{field_name}] must be a table")
        _check_unknown_keys(field, {"header", "position"}, where=f"[schema.fields.{field_name}]")
        if not isinstance(field.get("header"), str) or not field["header"]:
            raise ConfigError(f"[schema.fields.{field_name}].header must be a nonempty string")
        position = field.get("position")
        if isinstance(position, bool) or not isinstance(position, int) or position < 1:
            raise ConfigError(f"[schema.fields.{field_name}].position must be a positive integer")
        positions.append(position)
    _check_unique(positions, where="[schema.fields] positions")
    if max(positions) > schema["expected_physical_columns"]:
        raise ConfigError("[schema.fields] contains a position beyond expected_physical_columns")
    normalization = schema.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ConfigError("[schema.normalization] must be a table")
    _check_unknown_keys(
        normalization,
        {"water_regime", "season", "treatment_class"},
        where="[schema.normalization]",
    )
    for group in ("water_regime", "season", "treatment_class"):
        mapping = normalization.get(group)
        if not isinstance(mapping, Mapping) or not mapping:
            raise ConfigError(f"[schema.normalization.{group}] must define at least one mapping")
        for canonical, raw_values in mapping.items():
            values = _string_list(raw_values, where=f"[schema.normalization.{group}].{canonical}")
            _check_unique(values, where=f"[schema.normalization.{group}].{canonical}")
        if group == "treatment_class":
            unknown_classes = set(mapping) - KNOWN_TREATMENT_CLASSES
            if unknown_classes:
                raise ConfigError(
                    "[schema.normalization.treatment_class] contains unknown class(es): "
                    + ", ".join(sorted(unknown_classes))
                )
            missing_control_classes = {"zero_n", "absolute_control"} - set(mapping)
            if missing_control_classes:
                raise ConfigError(
                    "[schema.normalization.treatment_class] must distinguish zero_n and absolute_control"
                )
            alias_owners: dict[str, str] = {}
            for canonical, raw_values in mapping.items():
                for raw_value in raw_values:
                    alias = " ".join(raw_value.casefold().split())
                    owner = alias_owners.setdefault(alias, canonical)
                    if owner != canonical:
                        raise ConfigError(
                            f"treatment-class alias {raw_value!r} belongs to more than one class: "
                            f"{owner}, {canonical}"
                        )
    units = schema.get("units")
    if not isinstance(units, Mapping) or not units or any(not isinstance(value, str) or not value.strip() for value in units.values()):
        raise ConfigError("[schema.units] must define nonempty unit strings")
    for unit_key, quantity in (("n_rate", "n_rate"), ("yield_curve", "yield")):
        if unit_key not in units or canonical_unit(units[unit_key], quantity) is None:
            raise ConfigError(f"[schema.units].{unit_key} must declare a supported canonical unit")

    missing_values = data["missing_values"]
    _check_unknown_keys(
        missing_values,
        {"blank", "not_stated", "not_applicable", "invalid_numeric"},
        where="[missing_values]",
    )
    _require_string(missing_values, "blank", where="[missing_values]")
    _require_string(missing_values, "not_stated", where="[missing_values]")
    not_applicable = _string_list(missing_values.get("not_applicable"), where="[missing_values].not_applicable")
    _check_unique(not_applicable, where="[missing_values].not_applicable")
    invalid_numeric = _string_list(missing_values.get("invalid_numeric"), where="[missing_values].invalid_numeric")
    _check_unique(invalid_numeric, where="[missing_values].invalid_numeric")

    eligibility = data["eligibility"]
    _check_unknown_keys(
        eligibility,
        {
            "minimum_distinct_n_levels", "minimum_complete_n_yield", "minimum_model_residual_df",
            "require_zero_n_for_primary", "primary_inorganic_only", "organic_policy", "p_k_policy",
            "zero_n_policy", "constant_nutrient_tolerance", "high_n_review_threshold_kg_ha",
            "critical_error_codes", "n_level_tolerance_kg_ha", "n_rate_min_kg_ha",
            "n_rate_max_kg_ha", "yield_min_t_ha", "yield_max_t_ha",
        },
        where="[eligibility]",
    )
    _require_int_at_least(eligibility, "minimum_distinct_n_levels", 3, where="[eligibility]")
    _require_int_at_least(eligibility, "minimum_complete_n_yield", 3, where="[eligibility]")
    _require_int_at_least(eligibility, "minimum_model_residual_df", 1, where="[eligibility]")
    _require_bool(eligibility, "require_zero_n_for_primary", where="[eligibility]")
    _require_bool(eligibility, "primary_inorganic_only", where="[eligibility]")
    for key in ("organic_policy", "p_k_policy", "zero_n_policy"):
        _require_string(eligibility, key, where="[eligibility]")
    if eligibility["organic_policy"] not in {"retain_flagged", "exclude_primary", "exclude_all"}:
        raise ConfigError("[eligibility].organic_policy is invalid")
    if eligibility["p_k_policy"] not in {"retain_flagged", "require_constant", "exclude_all"}:
        raise ConfigError("[eligibility].p_k_policy is invalid")
    if eligibility["zero_n_policy"] not in {"allow_flagged", "require_primary", "ignore"}:
        raise ConfigError("[eligibility].zero_n_policy is invalid")
    _require_number_at_least(
        eligibility,
        "constant_nutrient_tolerance",
        0,
        where="[eligibility]",
    )
    _require_positive_number(eligibility, "n_level_tolerance_kg_ha", where="[eligibility]")
    _require_number_at_least(eligibility, "n_rate_min_kg_ha", 0, where="[eligibility]")
    _require_positive_number(eligibility, "n_rate_max_kg_ha", where="[eligibility]")
    if eligibility["n_rate_max_kg_ha"] <= eligibility["n_rate_min_kg_ha"]:
        raise ConfigError("[eligibility] N-rate maximum must exceed its minimum")
    _require_number_at_least(eligibility, "yield_min_t_ha", 0, where="[eligibility]")
    _require_positive_number(eligibility, "yield_max_t_ha", where="[eligibility]")
    if eligibility["yield_max_t_ha"] <= eligibility["yield_min_t_ha"]:
        raise ConfigError("[eligibility] yield maximum must exceed its minimum")
    _require_number_at_least(
        eligibility,
        "high_n_review_threshold_kg_ha",
        0,
        where="[eligibility]",
    )
    critical_error_codes = _string_list(
        eligibility.get("critical_error_codes", []),
        where="[eligibility].critical_error_codes",
    )
    _check_unique(critical_error_codes, where="[eligibility].critical_error_codes")
    unknown_critical_codes = [code for code in critical_error_codes if code not in KNOWN_CRITICAL_ERROR_CODES]
    if unknown_critical_codes:
        raise ConfigError(
            "[eligibility].critical_error_codes contains unknown member(s): "
            + ", ".join(unknown_critical_codes)
        )

    modeling = data["modeling"]
    _check_unknown_keys(
        modeling,
        {
            "no_extrapolation", "minimum_residual_df", "convergence_tolerance",
            "plausible_yield_min_t_ha", "plausible_yield_max_t_ha", "plot_grid_points",
            "model_selection_metric", "tie_breaking", "candidate_models", "parameter_bounds",
            "allow_uncertainty", "uncertainty_method", "allow_baseline_response_metrics",
        },
        where="[modeling]",
    )
    _require_bool(modeling, "no_extrapolation", where="[modeling]")
    if modeling["no_extrapolation"] is not True:
        raise ConfigError("[modeling].no_extrapolation must remain true")
    _require_int_at_least(modeling, "minimum_residual_df", 1, where="[modeling]")
    _require_positive_number(modeling, "convergence_tolerance", where="[modeling]")
    _require_number_at_least(modeling, "plausible_yield_min_t_ha", 0, where="[modeling]")
    _require_positive_number(modeling, "plausible_yield_max_t_ha", where="[modeling]")
    if modeling["plausible_yield_max_t_ha"] <= modeling["plausible_yield_min_t_ha"]:
        raise ConfigError("[modeling] plausible yield maximum must exceed its minimum")
    _require_int_at_least(modeling, "plot_grid_points", 2, where="[modeling]")
    _require_string(modeling, "model_selection_metric", where="[modeling]")
    if modeling["model_selection_metric"] not in KNOWN_MODEL_SELECTION_METRICS:
        raise ConfigError("[modeling].model_selection_metric is invalid")
    _require_string(modeling, "tie_breaking", where="[modeling]")
    if modeling["tie_breaking"] not in KNOWN_TIE_BREAKING_RULES:
        raise ConfigError("[modeling].tie_breaking is invalid")
    expected_tie_breaking = {
        "aicc_then_grouped_prediction": "simpler_model_then_stable_domain",
        "all_credible_no_selection": "not_applicable",
    }[modeling["model_selection_metric"]]
    if modeling["tie_breaking"] != expected_tie_breaking:
        raise ConfigError(
            "[modeling].tie_breaking is inconsistent with model_selection_metric; "
            f"expected {expected_tie_breaking!r}"
        )
    candidate_models = _toggle_list(
        modeling, "candidate_models", KNOWN_MODELS, where="[modeling]", allow_empty=False
    )
    if any(model not in candidate_models for model in enabled_models):
        raise ConfigError("[selection].enabled_models must be a subset of [modeling].candidate_models")
    parameter_bounds = modeling.get("parameter_bounds")
    if not isinstance(parameter_bounds, Mapping):
        raise ConfigError("[modeling.parameter_bounds] must be a table")
    _check_unknown_keys(
        parameter_bounds,
        {"n_rate_min_kg_ha", "n_rate_max_kg_ha"},
        where="[modeling.parameter_bounds]",
    )
    _require_number_at_least(parameter_bounds, "n_rate_min_kg_ha", 0, where="[modeling.parameter_bounds]")
    _require_positive_number(parameter_bounds, "n_rate_max_kg_ha", where="[modeling.parameter_bounds]")
    if parameter_bounds["n_rate_max_kg_ha"] <= parameter_bounds["n_rate_min_kg_ha"]:
        raise ConfigError("[modeling.parameter_bounds] maximum must exceed minimum")
    _require_bool(modeling, "allow_uncertainty", where="[modeling]")
    _require_bool(
        modeling,
        "allow_baseline_response_metrics",
        where="[modeling]",
    )
    _require_string(modeling, "uncertainty_method", where="[modeling]")
    outputs = data["outputs"]
    _check_unknown_keys(
        outputs,
        {"atomic_writes", "allow_source_targets", "row_level_qc", "collision_policy"},
        where="[outputs]",
    )
    _require_bool(outputs, "atomic_writes", where="[outputs]")
    if outputs["atomic_writes"] is not True:
        raise ConfigError("[outputs].atomic_writes must remain true")
    _require_bool(outputs, "allow_source_targets", where="[outputs]")
    if outputs["allow_source_targets"] is not False:
        raise ConfigError("[outputs].allow_source_targets must remain false")
    _require_bool(outputs, "row_level_qc", where="[outputs]")
    _require_string(outputs, "collision_policy", where="[outputs]")
    if outputs["collision_policy"] not in {"fail", "replace_only_with_overwrite"}:
        raise ConfigError("[outputs].collision_policy is invalid")
    if run["overwrite"] and outputs["collision_policy"] != "replace_only_with_overwrite":
        raise ConfigError(
            "[outputs].collision_policy must be 'replace_only_with_overwrite' when [run].overwrite is true"
        )

    logging = data["logging"]
    _check_unknown_keys(logging, {"level", "write_logs_in_validate"}, where="[logging]")
    _require_string(logging, "level", where="[logging]")
    if logging["level"] not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigError("[logging].level is invalid")
    _require_bool(logging, "write_logs_in_validate", where="[logging]")
    if mode == "validate" and logging["write_logs_in_validate"]:
        raise ConfigError("validate mode cannot write logs")

    engines = data["engines"]
    _check_unknown_keys(
        engines,
        {
            "python_command", "rscript_command", "r_entrypoint", "require_conda_environment",
            "require_linux_rscript", "allow_runtime_r_package_install", "allow_automatic_engine_fallback",
            "r_use_vanilla_session", "assignments",
        },
        where="[engines]",
    )
    for key in ("python_command", "rscript_command", "r_entrypoint"):
        _require_string(engines, key, where="[engines]")
    for key in ("require_conda_environment", "require_linux_rscript", "allow_runtime_r_package_install", "allow_automatic_engine_fallback", "r_use_vanilla_session"):
        _require_bool(engines, key, where="[engines]")
    if engines["allow_runtime_r_package_install"] is not False:
        raise ConfigError("[engines].allow_runtime_r_package_install must remain false")
    if engines["allow_automatic_engine_fallback"] is not False:
        raise ConfigError("[engines].allow_automatic_engine_fallback must remain false")
    if engines["r_use_vanilla_session"] is not True:
        raise ConfigError("[engines].r_use_vanilla_session must remain true")
    r_entrypoint = _resolve_relative_path(engines["r_entrypoint"], root, "[engines].r_entrypoint")

    assignments = data["engines"].get("assignments")
    if not isinstance(assignments, Mapping):
        raise ConfigError("[engines.assignments] must be a table")
    for family, engine in assignments.items():
        if family not in KNOWN_ENGINE_RESPONSIBILITIES:
            raise ConfigError(f"[engines.assignments] contains unknown responsibility: {family}")
        if engine not in {"python", "r"}:
            raise ConfigError(f"[engines.assignments.{family}] must be 'python' or 'r'")
    analysis = data["analysis_matrix"]
    _check_unknown_keys(
        analysis,
        {
            "dataset_versions", "source_combination_modes", "curve_outcomes", "explanatory_factors",
            "analysis_families", "interaction_orders", "combination_mode", "run_supported_only",
            "group_cross_validation_by", "multiple_testing_method", "support_policy",
            "deferred_analysis_families", "deferred_interaction_orders",
        },
        where="[analysis_matrix]",
    )
    dataset_versions = _toggle_list(
        analysis,
        "dataset_versions",
        KNOWN_DATASET_VERSIONS,
        where="[analysis_matrix]",
        allow_empty=False,
    )
    source_combination_modes = _toggle_list(
        analysis,
        "source_combination_modes",
        KNOWN_SOURCE_COMBINATION_MODES,
        where="[analysis_matrix]",
        allow_empty=False,
    )
    curve_outcomes = _toggle_list(
        analysis,
        "curve_outcomes",
        KNOWN_CURVE_OUTCOMES,
        where="[analysis_matrix]",
        allow_empty=False,
    )
    explanatory_factors = _toggle_list(
        analysis,
        "explanatory_factors",
        KNOWN_FACTORS,
        where="[analysis_matrix]",
        allow_empty=True,
    )
    analysis_families = _toggle_list(
        analysis,
        "analysis_families",
        KNOWN_ANALYSIS_FAMILIES,
        where="[analysis_matrix]",
        allow_empty=False,
    )
    deferred_analysis_families = _toggle_list(
        analysis,
        "deferred_analysis_families",
        KNOWN_ANALYSIS_FAMILIES,
        where="[analysis_matrix]",
        allow_empty=True,
    )
    overlapping_families = sorted(set(analysis_families).intersection(deferred_analysis_families))
    if overlapping_families:
        raise ConfigError(
            "[analysis_matrix] analysis_families and deferred_analysis_families overlap: "
            + ", ".join(overlapping_families)
        )
    raw_orders = analysis.get("interaction_orders")
    if not isinstance(raw_orders, list) or any(isinstance(order, bool) or not isinstance(order, int) for order in raw_orders):
        raise ConfigError("[analysis_matrix].interaction_orders must be a list of integers")
    interaction_orders = tuple(raw_orders)
    _check_unique(interaction_orders, where="[analysis_matrix].interaction_orders")
    if any(order < 1 or order > 3 for order in interaction_orders):
        raise ConfigError("[analysis_matrix].interaction_orders must contain only 1, 2, or 3")
    raw_deferred_orders = analysis.get("deferred_interaction_orders")
    if (
        not isinstance(raw_deferred_orders, list)
        or any(isinstance(order, bool) or not isinstance(order, int) for order in raw_deferred_orders)
    ):
        raise ConfigError("[analysis_matrix].deferred_interaction_orders must be a list of integers")
    deferred_interaction_orders = tuple(raw_deferred_orders)
    _check_unique(
        deferred_interaction_orders,
        where="[analysis_matrix].deferred_interaction_orders",
    )
    if any(order < 1 or order > 3 for order in deferred_interaction_orders):
        raise ConfigError(
            "[analysis_matrix].deferred_interaction_orders must contain only 1, 2, or 3"
        )
    overlapping_orders = sorted(set(interaction_orders).intersection(deferred_interaction_orders))
    if overlapping_orders:
        raise ConfigError(
            "[analysis_matrix] interaction_orders and deferred_interaction_orders overlap: "
            + ", ".join(map(str, overlapping_orders))
        )
    _require_string(analysis, "combination_mode", where="[analysis_matrix]")
    if analysis["combination_mode"] != "all_supported":
        raise ConfigError("[analysis_matrix].combination_mode must be 'all_supported'")
    _require_bool(analysis, "run_supported_only", where="[analysis_matrix]")
    _require_string(analysis, "group_cross_validation_by", where="[analysis_matrix]")
    if analysis["group_cross_validation_by"] != "study_uid":
        raise ConfigError("[analysis_matrix].group_cross_validation_by must be 'study_uid'")
    _require_string(analysis, "multiple_testing_method", where="[analysis_matrix]")
    if analysis["multiple_testing_method"] not in KNOWN_MULTIPLE_TESTING_METHODS:
        raise ConfigError("[analysis_matrix].multiple_testing_method is invalid")
    support_policy = analysis.get("support_policy")
    if support_policy is not None:
        if not isinstance(support_policy, Mapping):
            raise ConfigError("[analysis_matrix.support_policy] must be a table")
        support_keys = {
            "minimum_independent_studies",
            "minimum_factor_cell_count",
            "maximum_factor_cardinality",
            "minimum_residual_information",
        }
        _check_unknown_keys(support_policy, support_keys, where="[analysis_matrix.support_policy]")
        missing_support_keys = support_keys - set(support_policy)
        if missing_support_keys:
            raise ConfigError(
                "[analysis_matrix.support_policy] is missing required key(s): "
                + ", ".join(sorted(missing_support_keys))
            )
        for key in sorted(support_keys):
            value = support_policy[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"[analysis_matrix.support_policy].{key} must be an integer >= 1")
    for family in analysis_families:
        if family not in assignments:
            raise ConfigError(f"Enabled analysis family has no primary engine assignment: {family}")
    misassigned_r_families = sorted(
        family
        for family in analysis_families
        if family in R_OWNED_ANALYSIS_FAMILIES and assignments[family] != "r"
    )
    if misassigned_r_families:
        raise ConfigError(
            "R-owned analysis families must be assigned to the R engine: "
            + ", ".join(misassigned_r_families)
        )
    if any(assignments[family] == "r" for family in analysis_families):
        if not engines["require_conda_environment"] or not engines["require_linux_rscript"]:
            raise ConfigError(
                "Enabled R-assigned analysis requires both Conda-environment and Linux-Rscript checks"
            )
        if preflight_engines:
            _preflight_r_contract(engines, r_entrypoint, root)
        if support_policy is None and any(
            assignments[family] == "r" and family in R_OWNED_ANALYSIS_FAMILIES
            for family in analysis_families
        ):
            raise ConfigError(
                "Enabled support-gated R-assigned analysis requires [analysis_matrix.support_policy]"
            )

    if required_full_decisions:
        applicable_decisions: set[str] = set()
        if "ltcce" in enabled_sources:
            applicable_decisions.add("SRC-01")
        if "D12_climate_enriched_future" in dataset_versions:
            applicable_decisions.add("SRC-05")
        if "D09_complete_recommendation_set" in dataset_versions:
            applicable_decisions.add("ELG-11")
        if modeling["allow_baseline_response_metrics"]:
            applicable_decisions.add("ELG-10")
        if modeling["allow_uncertainty"]:
            applicable_decisions.add("MOD-04")
        if "economic_optimum_n_kg_ha" in curve_outcomes:
            applicable_decisions.add("MOD-06")
        if modeling["model_selection_metric"] == "all_credible_no_selection":
            applicable_decisions.update(("MOD-02", "MOD-05"))
        inferential_families = {
            "one_factor_inferential",
            "all_supported_interactions",
            "multivariable_mixed_effects",
            "observation_level_curve_modification",
            "marginal_contrasts",
        }
        if support_policy is not None or inferential_families.intersection(analysis_families):
            applicable_decisions.add("ANA-05")
        if inferential_families.intersection(analysis_families):
            applicable_decisions.update(("ANA-07", "ANA-08", "ANA-10", "ANA-13", "ANA-14"))
        if (
            any(order > 1 for order in interaction_orders)
            or {
                "all_supported_interactions",
                "observation_level_curve_modification",
            }.intersection(analysis_families)
        ):
            applicable_decisions.add("ANA-06")
        if {"penalized_predictive_models", "curve_feature_clustering"}.intersection(
            analysis_families
        ):
            applicable_decisions.update(("ANA-05", "ANA-11", "ANA-13", "ANA-14"))
        if "penalized_predictive_models" in analysis_families:
            applicable_decisions.add("ANA-09")
        if explanatory_factors and {
            "one_factor_descriptive",
            "one_factor_inferential",
            "all_supported_interactions",
            "multivariable_mixed_effects",
            "observation_level_curve_modification",
            "marginal_contrasts",
        }.intersection(analysis_families):
            applicable_decisions.update(("ANA-04", "ANA-08", "ANA-14"))
        if {
            "recommendation_yield_gap_t_ha",
            "target_yield_gap_t_ha",
            "target_yield_status",
        }.intersection(curve_outcomes):
            applicable_decisions.add("ANA-15")
        missing_applicable_decisions = sorted(
            applicable_decisions.difference(required_full_decisions)
        )
        if missing_applicable_decisions:
            raise ConfigError(
                "[run].required_full_decisions omits decision(s) applicable to the "
                "effective operation profile: "
                + ", ".join(missing_applicable_decisions)
            )

    cpu_budget = run["max_parallel_r_jobs"] * run["r_threads_per_job"]
    visible_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if cpu_budget > visible_cpus:
        raise ConfigError(
            f"Configured R CPU budget ({cpu_budget}) exceeds affinity-visible CPUs ({visible_cpus})"
        )

    frozen_raw = _freeze_config_value(data)
    frozen_sources = frozen_raw["sources"]
    return ValidatedConfig(
        config_path=path,
        project_root=root,
        raw=frozen_raw,
        paths=MappingProxyType(dict(paths)),
        sources=frozen_sources,
        enabled_sources=tuple(enabled_sources),
        enabled_models=tuple(enabled_models),
        comparison_dimensions=tuple(comparison_dimensions),
        scope_countries=tuple(scope_countries),
        series_identity_dimensions=tuple(series_identity_dimensions),
        output_formats=tuple(output_formats),
        figure_formats=tuple(figure_formats),
        fill_down_fields=tuple(fill_down_fields),
        treatment_classes=tuple(treatment_classes),
        dataset_versions=tuple(dataset_versions),
        source_combination_modes=tuple(source_combination_modes),
        curve_outcomes=tuple(curve_outcomes),
        explanatory_factors=tuple(explanatory_factors),
        analysis_families=tuple(analysis_families),
        interaction_orders=interaction_orders,
        engine_assignments=MappingProxyType(dict(assignments)),
        decision_snapshot=_freeze_config_value(decision_snapshot),
        decision_snapshot_hash=decision_snapshot_hash,
        run_mode=mode,
    )


def _resolve_paths(values: Mapping[str, Any], root: Path) -> dict[str, Path]:
    missing = REQUIRED_PATHS - set(values)
    if missing:
        raise ConfigError(f"[paths] is missing: {', '.join(sorted(missing))}")
    unknown = set(values) - REQUIRED_PATHS
    if unknown:
        raise ConfigError(f"[paths] contains unknown keys: {', '.join(sorted(unknown))}")
    return {
        key: _resolve_relative_path(values[key], root, f"[paths].{key}")
        for key in REQUIRED_PATHS
    }


def _resolve_relative_path(value: Any, root: Path, where: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a nonempty project-relative path")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ConfigError(f"{where} must be project-relative, not absolute: {value}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConfigError(f"{where} escapes the project root: {value}") from exc
    return resolved


def _check_path_overlaps(paths: Mapping[str, Path]) -> None:
    for input_key in INPUT_PATHS:
        for output_key in OUTPUT_PATHS:
            input_path = paths[input_key]
            output_path = paths[output_key]
            if input_path == output_path or input_path.is_relative_to(output_path) or output_path.is_relative_to(input_path):
                raise ConfigError(f"Input path [paths].{input_key} overlaps output path [paths].{output_key}")


def _check_unknown_keys(table: Mapping[str, Any], allowed: set[str], *, where: str) -> None:
    unknown = set(table) - allowed
    if unknown:
        raise ConfigError(f"{where} contains unknown key(s): {', '.join(sorted(unknown))}")


def _toggle_list(
    table: Mapping[str, Any],
    key: str,
    known: set[str],
    *,
    where: str,
    allow_empty: bool,
) -> list[Any]:
    location = f"{where}.{key}"
    values = _string_list(table.get(key), where=location)
    _check_unique(values, where=location)
    unknown = [value for value in values if value not in known]
    if unknown:
        raise ConfigError(f"{location} contains unknown member(s): {', '.join(unknown)}")
    if not allow_empty and not values:
        raise ConfigError(f"{location} may not be empty")
    return values


def _string_list(value: Any, *, where: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigError(f"{where} must be a list of strings")
    return list(value)


def _check_unique(values: list[Any] | tuple[Any, ...], *, where: str) -> None:
    if len(values) != len(set(values)):
        raise ConfigError(f"{where} contains duplicate members")


def _require_string(table: Mapping[str, Any], key: str, *, where: str) -> None:
    if not isinstance(table.get(key), str) or not table[key].strip():
        raise ConfigError(f"{where}.{key} must be a nonempty string")


def _require_bool(table: Mapping[str, Any], key: str, *, where: str) -> None:
    if type(table.get(key)) is not bool:
        raise ConfigError(f"{where}.{key} must be a Boolean")


def _require_int(table: Mapping[str, Any], key: str, *, where: str) -> None:
    if isinstance(table.get(key), bool) or not isinstance(table.get(key), int):
        raise ConfigError(f"{where}.{key} must be an integer")


def _require_int_at_least(table: Mapping[str, Any], key: str, minimum: int, *, where: str) -> None:
    _require_int(table, key, where=where)
    if table[key] < minimum:
        raise ConfigError(f"{where}.{key} must be >= {minimum}")


def _require_number_at_least(table: Mapping[str, Any], key: str, minimum: float, *, where: str) -> None:
    value = table.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < minimum
    ):
        raise ConfigError(f"{where}.{key} must be a number >= {minimum}")


def _require_positive_number(table: Mapping[str, Any], key: str, *, where: str) -> None:
    value = table.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ConfigError(f"{where}.{key} must be a positive number")


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise ConfigError(f"Required {description} does not exist: {path}")


def _preflight_r_contract(engines: Mapping[str, Any], r_entrypoint: Path, root: Path) -> None:
    if not r_entrypoint.is_file():
        raise ConfigError(f"R contract entry point does not exist: {r_entrypoint}")
    command = engines["rscript_command"]
    command_path = Path(command)
    resolved_command = Path(shutil.which(command) or "")
    if not resolved_command or not resolved_command.is_file() or not os.access(resolved_command, os.X_OK):
        raise ConfigError(f"Rscript command was not found or is not executable: {command}")
    resolved_command = resolved_command.resolve()
    if command_path.name.lower().endswith(".exe") or resolved_command.name.lower().endswith(".exe"):
        raise ConfigError("Rscript command points to a Windows executable")
    if engines["require_conda_environment"]:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix:
            raise ConfigError("R contract preflight requires an active Conda environment")
        try:
            resolved_command.relative_to(Path(conda_prefix).resolve())
        except ValueError as exc:
            raise ConfigError(
                f"Rscript must resolve inside CONDA_PREFIX ({conda_prefix}): {resolved_command}"
            ) from exc
    if engines["r_use_vanilla_session"] is not True:
        raise ConfigError("R contract requires a vanilla R session")


def _approval_fields_from_block(
    block: str,
    *,
    decision_id: str,
) -> dict[str, str]:
    """Read one structured approval record from explicit fields or Approval Evidence."""

    labels = (
        ("Approver", "approver"),
        ("Approval Date", "approval_date"),
        ("Approval Source", "approval_source"),
    )
    combined_matches = re.findall(
        r"^[ \t]*(?:[-*][ \t]+)?\*\*Approval Evidence:\*\*[ \t]*(.*?)[ \t]*$",
        block,
        flags=re.MULTILINE,
    )
    if len(combined_matches) > 1:
        raise ConfigError(
            f"full mode decision {decision_id} must have at most one ledger Approval Evidence field"
        )
    combined_fields: dict[str, str] = {}
    if combined_matches and combined_matches[0].strip():
        combined_value = combined_matches[0].strip()
        combined_pattern = re.compile(
            r"(?:^|[;|])[ \t]*"
            r"(Approver|Approval Date|Approval Source)[ \t]*:[ \t]*"
            r"(.*?)(?=[ \t]*(?:[;|][ \t]*(?:Approver|Approval Date|Approval Source)[ \t]*:|$))",
            flags=re.IGNORECASE,
        )
        parsed = tuple(combined_pattern.finditer(combined_value))
        if parsed:
            consumed = "".join(match.group(0) for match in parsed)
            if re.sub(r"[\s;|]+", "", consumed).casefold() != re.sub(
                r"[\s;|]+", "", combined_value
            ).casefold():
                raise ConfigError(
                    f"full mode decision {decision_id} has malformed structured Approval Evidence"
                )
            canonical_keys = {label.casefold(): key for label, key in labels}
            for match in parsed:
                key = canonical_keys[match.group(1).casefold()]
                if key in combined_fields:
                    raise ConfigError(
                        f"full mode decision {decision_id} repeats {match.group(1)} in Approval Evidence"
                    )
                combined_fields[key] = re.sub(r"[ \t]+", " ", match.group(2).strip())

    approval_fields: dict[str, str] = {}
    for label, key in labels:
        explicit_matches = re.findall(
            rf"^[ \t]*(?:[-*][ \t]+)?\*\*{re.escape(label)}:\*\*[ \t]*(.*?)[ \t]*$",
            block,
            flags=re.MULTILINE,
        )
        values = [match.strip() for match in explicit_matches if match.strip()]
        if key in combined_fields:
            values.append(combined_fields[key])
        if len(values) != 1:
            raise ConfigError(
                f"full mode decision {decision_id} requires exactly one nonblank ledger {label}"
            )
        approval_fields[key] = re.sub(r"[ \t]+", " ", values[0])
    return approval_fields


def _require_full_mode_decision_approvals(
    approved_decisions: tuple[str, ...],
    required_decisions: tuple[str, ...],
    *,
    project_root: Path,
) -> tuple[dict[str, dict[str, str | bool]], str]:
    invalid = tuple(
        decision
        for decision in (*approved_decisions, *required_decisions)
        if re.fullmatch(r"[A-Z][A-Z0-9]*-\d+", decision) is None
    )
    if invalid:
        raise ConfigError(
            "full mode decision entries must use domain-prefixed IDs: "
            + ", ".join(invalid)
        )
    if not required_decisions:
        raise ConfigError(
            "full mode requires a nonempty [run].required_full_decisions authority gate"
        )
    if not approved_decisions:
        raise ConfigError(
            "full mode requires explicit approvals in [run].decisions_approved"
        )
    missing = tuple(
        decision for decision in required_decisions if decision not in approved_decisions
    )
    if missing:
        raise ConfigError(
            "full mode [run].decisions_approved is missing required approvals: "
            + ", ".join(missing)
        )

    ledger_path = project_root / "Docs" / "Plans_Phases" / "03_Decisions_Needed.md"
    try:
        ledger_text = ledger_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(
            f"full mode requires the authoritative decision ledger: {ledger_path}"
        ) from exc
    except OSError as exc:
        raise ConfigError(
            f"full mode could not read the authoritative decision ledger {ledger_path}: {exc}"
        ) from exc

    heading_pattern = re.compile(
        r"^###\s+([A-Z][A-Z0-9]*-\d+)(?=\s|:|$).*$",
        flags=re.MULTILINE,
    )
    heading_matches = tuple(heading_pattern.finditer(ledger_text))
    blocks: dict[str, list[str]] = {}
    for index, heading in enumerate(heading_matches):
        decision_id = heading.group(1)
        end = heading_matches[index + 1].start() if index + 1 < len(heading_matches) else len(ledger_text)
        blocks.setdefault(decision_id, []).append(ledger_text[heading.end():end])

    snapshot: dict[str, dict[str, str | bool]] = {}
    for decision_id in sorted(required_decisions):
        decision_blocks = blocks.get(decision_id, [])
        if len(decision_blocks) != 1:
            raise ConfigError(
                f"full mode requires exactly one authoritative ledger entry for {decision_id}; "
                f"found {len(decision_blocks)}"
            )
        block = decision_blocks[0]
        statuses = re.findall(
            r"^\*\*Status:\*\*[ \t]*(.*?)[ \t]*$",
            block,
            flags=re.MULTILINE,
        )
        if len(statuses) != 1:
            raise ConfigError(
                f"full mode decision {decision_id} must have exactly one ledger Status; "
                f"found {len(statuses)}"
            )
        status = statuses[0].strip()
        if status != "DECIDED":
            raise ConfigError(
                f"full mode decision {decision_id} ledger Status must be exactly DECIDED, "
                f"got {status!r}"
            )

        selected_options = re.findall(
            r"^\*\*(?:Choosed|Chosen) Option:\*\*[ \t]*(.*?)[ \t]*$",
            block,
            flags=re.MULTILINE,
        )
        selected_options.extend(
            re.findall(
                r"^\*\*(?:Choosed|Chosen) Option:[ \t]*(.*?)[ \t]*\*\*[ \t]*$",
                block,
                flags=re.MULTILINE,
            )
        )
        if len(selected_options) != 1:
            raise ConfigError(
                f"full mode decision {decision_id} must have exactly one ledger Choosed Option; "
                f"found {len(selected_options)}"
            )
        selected_option = re.sub(r"[ \t]+", " ", selected_options[0].strip())
        if (
            len(selected_option) >= 2
            and selected_option[0] == selected_option[-1]
            and selected_option[0] in {'"', "'", "`"}
        ):
            selected_option = selected_option[1:-1].strip()
        if not selected_option:
            raise ConfigError(
                f"full mode decision {decision_id} requires a nonblank ledger Choosed Option"
            )
        approval_fields = _approval_fields_from_block(block, decision_id=decision_id)
        invalid_placeholders = {
            "",
            "n/a",
            "na",
            "none",
            "not identified",
            "pending",
            "tbd",
            "unknown",
            "unresolved",
        }
        if approval_fields["approver"].casefold() in invalid_placeholders:
            raise ConfigError(
                f"full mode decision {decision_id} requires a named approving stakeholder"
            )
        if approval_fields["approval_source"].casefold() in invalid_placeholders:
            raise ConfigError(
                f"full mode decision {decision_id} requires a concrete approval source"
            )
        try:
            date.fromisoformat(approval_fields["approval_date"])
        except ValueError as exc:
            raise ConfigError(
                f"full mode decision {decision_id} Approval Date must use a valid YYYY-MM-DD date"
            ) from exc
        snapshot[decision_id] = {
            "status": status,
            "selected_option": selected_option,
            "approved": decision_id in approved_decisions,
            **approval_fields,
        }

    snapshot_payload = [
        {"decision_id": decision_id, **snapshot[decision_id]}
        for decision_id in sorted(snapshot)
    ]
    snapshot_hash = hashlib.sha256(
        json.dumps(
            snapshot_payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return snapshot, snapshot_hash


__all__ = ["ConfigError", "ValidatedConfig", "load_config", "validate_config"]
