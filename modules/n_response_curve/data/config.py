from __future__ import annotations

from dataclasses import dataclass
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


class ConfigError(ValueError):
    """Raised when the operator configuration is unsafe or incoherent."""


RUN_MODES = {"validate", "test", "full"}
KNOWN_MODELS = {"linear", "quadratic", "linear_plateau", "quadratic_plateau", "mitscherlich"}
KNOWN_SOURCE_TYPES = {"literature", "ltcce", "rcm_validation", "nopt", "future"}
KNOWN_SOURCE_AVAILABILITY = {"available", "expected_unavailable"}
KNOWN_SOURCE_CONFIRMATION_STATUSES = {"verified", "pending"}
KNOWN_COMPARISON_DIMENSIONS = {"water_regime", "season", "region", "province", "variety", "recommendation_class"}
KNOWN_OUTPUT_FORMATS = set(_SUPPORTED_TABLE_FORMATS)
KNOWN_FIGURE_FORMATS = set(_SUPPORTED_FIGURE_FORMATS)
KNOWN_TREATMENT_CLASSES = {"zero_n", "absolute_control", "RCM", "FP", "NOPT_NPK", "other", "unresolved"}
KNOWN_FILL_DOWN_FIELDS = {"Study_ID", "Trial ID", "Source", "Author(s)", "Year of Publication"}
KNOWN_CRITICAL_ERROR_CODES = {"YIELD_UNIT_CONFLICT"}
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
    "optimum_status",
    "agronomic_optimum_n_kg_ha",
    "economic_optimum_n_kg_ha",
    "plateau_onset_n_kg_ha",
    "predicted_max_yield_t_ha",
    "observed_max_yield_t_ha",
    "yield_at_zero_n_t_ha",
    "yield_response_above_zero_n_t_ha",
    "recommendation_yield_gap_t_ha",
    "target_yield_gap_t_ha",
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
KNOWN_MODEL_SELECTION_METRICS = {"aicc_then_grouped_prediction"}
KNOWN_TIE_BREAKING_RULES = {"simpler_model_then_stable_domain"}
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

    run = data["run"]
    _check_unknown_keys(
        run,
        {
            "mode", "overwrite", "random_seed", "test_group_limit", "fail_fast",
            "r_threads_per_job", "max_parallel_r_jobs", "r_stage_timeout_seconds",
            "r_termination_grace_seconds", "cpu_detection", "decisions_approved",
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
    if mode == "full":
        _require_approved_decision_ledger(root, tuple(approved_decisions))

    paths = _resolve_paths(data["paths"], root)
    _check_path_overlaps(paths)

    selection = data["selection"]
    _check_unknown_keys(
        selection,
        {
            "enabled_sources", "enabled_models", "comparison_dimensions", "output_formats",
            "figure_formats", "fill_down_fields", "treatment_classes",
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
                "availability", "confirmation_status", "shape_adapter_version",
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
    units = schema.get("units")
    if not isinstance(units, Mapping) or not units or any(not isinstance(value, str) or not value.strip() for value in units.values()):
        raise ConfigError("[schema.units] must define nonempty unit strings")

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
            "critical_error_codes",
        },
        where="[eligibility]",
    )
    _require_int_at_least(eligibility, "minimum_distinct_n_levels", 2, where="[eligibility]")
    _require_int_at_least(eligibility, "minimum_complete_n_yield", 2, where="[eligibility]")
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
            "allow_uncertainty", "uncertainty_method",
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
    raw_orders = analysis.get("interaction_orders")
    if not isinstance(raw_orders, list) or any(isinstance(order, bool) or not isinstance(order, int) for order in raw_orders):
        raise ConfigError("[analysis_matrix].interaction_orders must be a list of integers")
    interaction_orders = tuple(raw_orders)
    _check_unique(interaction_orders, where="[analysis_matrix].interaction_orders")
    if any(order < 1 or order > 3 for order in interaction_orders):
        raise ConfigError("[analysis_matrix].interaction_orders must contain only 1, 2, or 3")
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


def _require_approved_decision_ledger(root: Path, approved_decisions: tuple[str, ...]) -> None:
    ledger = root / "Docs" / "Plans_Phases" / "03_Decisions_Needed.md"
    if not ledger.is_file() or not ledger.read_text(encoding="utf-8").strip():
        raise ConfigError(f"full mode requires a nonempty decision ledger: {ledger}")
    text = ledger.read_text(encoding="utf-8")
    headings = tuple(re.finditer(r"(?m)^###\s+([A-Z][A-Z0-9]*-\d+)\b[^\n]*$", text))
    if not headings:
        raise ConfigError(f"full mode decision ledger contains no domain-prefixed decision IDs: {ledger}")
    decision_ids = tuple(match.group(1) for match in headings)
    if len(decision_ids) != len(set(decision_ids)):
        raise ConfigError(f"full mode decision ledger contains duplicate decision IDs: {ledger}")
    unresolved: list[str] = []
    for index, heading in enumerate(headings):
        decision = heading.group(1)
        section_end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body = text[heading.end():section_end]
        option = re.search(r"(?m)^\s*\*\*Choosed Option:\*\*\s*(.*?)\s*$", body)
        selected = option.group(1).strip() if option is not None else ""
        if not selected or selected.casefold() in {"open", "pending", "tbd", "todo", "unresolved"}:
            unresolved.append(decision)
    if unresolved:
        raise ConfigError(
            "full mode requires a recorded nonempty **Choosed Option:** for "
            f"{', '.join(unresolved)} in {ledger}"
        )
    if approved_decisions != decision_ids:
        raise ConfigError(
            "full mode requires [run].decisions_approved to match every selected domain-prefixed "
            f"decision ID in ledger order: {', '.join(decision_ids)}"
        )


__all__ = ["ConfigError", "ValidatedConfig", "load_config", "validate_config"]
