from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import tomllib
from typing import Any, Mapping


class RecipeConfigError(ValueError):
    """Raised when the grain-yield-response recipe is unsafe or incoherent."""


@dataclass(frozen=True)
class GrainYieldResponseConfig:
    config_path: Path
    project_root: Path
    raw: Mapping[str, Any]
    mode: str
    overwrite: bool
    random_seed: int
    fail_fast: bool
    include_raw_sensitivity: bool
    run_mixed_model: bool
    mixed_model_required: bool
    release_verification_policy: str
    fit_pooled_linear: bool
    fit_pooled_quadratic: bool
    fit_within_series: bool
    fit_zero_n_delta: bool
    fit_series_linear: bool
    leave_one_series_out: bool
    source_name: str
    analysis_population: str
    required_release_status: str
    minimum_observations: int
    minimum_distinct_n_levels: int
    n_level_tolerance_kg_ha: float
    zero_n_tolerance_kg_ha: float
    series_key: str
    trial_key: str
    study_key: str
    categorical_factors: tuple[str, ...]
    numeric_factors: tuple[str, ...]
    minimum_level_observations: int
    minimum_modifier_series: int
    table_formats: tuple[str, ...]
    figure_formats: tuple[str, ...]
    figure_dpi: int
    figure_width_inches: float
    figure_height_inches: float
    expected_inputs: Mapping[str, str]
    base_config_path: Path
    release_package: Path
    output_root: Path
    full_output_root: Path
    test_output_root: Path
    r_stage_path: Path

    @property
    def writes_outputs(self) -> bool:
        return self.mode in {"test", "full"}


def _table(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name)
    if not isinstance(value, Mapping):
        raise RecipeConfigError(f"Missing or invalid [{name}] table")
    return value


_TABLE_KEYS = {
    "run": {
        "mode",
        "overwrite",
        "random_seed",
        "fail_fast",
        "include_raw_sensitivity",
        "run_mixed_model",
        "mixed_model_required",
        "release_verification_policy",
    },
    "population": {
        "source_name",
        "analysis_population",
        "required_release_status",
        "minimum_observations",
        "minimum_distinct_n_levels",
        "n_level_tolerance_kg_ha",
        "zero_n_tolerance_kg_ha",
    },
    "diagnostics": {
        "fit_pooled_linear",
        "fit_pooled_quadratic",
        "fit_within_series",
        "fit_zero_n_delta",
        "fit_series_linear",
        "leave_one_series_out",
    },
    "grouping": {"series_key", "trial_key", "study_key"},
    "factors": {
        "categorical",
        "numeric",
        "minimum_level_observations",
        "minimum_modifier_series",
    },
    "outputs": {
        "table_formats",
        "figure_formats",
        "figure_dpi",
        "figure_width_inches",
        "figure_height_inches",
    },
    "expected_inputs": {
        "base_config_sha256",
        "source_sha256",
        "release_manifest_sha256",
        "release_ledger_sha256",
        "release_checksums_sha256",
    },
    "paths": {
        "base_config",
        "release_package",
        "output_root",
        "test_output_root",
        "r_stage",
    },
}


def _validate_schema(data: Mapping[str, Any]) -> None:
    unknown_tables = sorted(set(data) - set(_TABLE_KEYS))
    if unknown_tables:
        raise RecipeConfigError(f"Unknown TOML tables: {unknown_tables}")
    missing_tables = sorted(set(_TABLE_KEYS) - set(data))
    if missing_tables:
        raise RecipeConfigError(f"Missing TOML tables: {missing_tables}")
    for table_name, allowed in _TABLE_KEYS.items():
        table = _table(data, table_name)
        unknown = sorted(set(table) - allowed)
        if unknown:
            raise RecipeConfigError(f"Unknown keys in [{table_name}]: {unknown}")


def _string(table: Mapping[str, Any], key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RecipeConfigError(f"{where}.{key} must be a non-empty string")
    return value.strip()


def _boolean(table: Mapping[str, Any], key: str, where: str) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        raise RecipeConfigError(f"{where}.{key} must be a Boolean")
    return value


def _positive_int(table: Mapping[str, Any], key: str, where: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RecipeConfigError(f"{where}.{key} must be a positive integer")
    return value


def _number(table: Mapping[str, Any], key: str, where: str, *, minimum: float = 0.0) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeConfigError(f"{where}.{key} must be a number")
    result = float(value)
    if result < minimum:
        raise RecipeConfigError(f"{where}.{key} must be at least {minimum}")
    return result


def _string_list(table: Mapping[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = table.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise RecipeConfigError(f"{where}.{key} must be a list of non-empty strings")
    values = tuple(item.strip() for item in value)
    if len(values) != len(set(values)):
        raise RecipeConfigError(f"{where}.{key} must not contain duplicates")
    return values


def _confined_path(root: Path, raw: str, where: str) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RecipeConfigError(f"{where} must stay inside the project root") from exc
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_recipe_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    check_files: bool = True,
) -> GrainYieldResponseConfig:
    path = Path(config_path).expanduser().resolve()
    root = Path(project_root or path.parent).expanduser().resolve()
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise RecipeConfigError(f"Recipe configuration does not exist: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise RecipeConfigError(f"Invalid TOML in {path}: {exc}") from exc

    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RecipeConfigError(
            "Recipe configuration must stay inside the project root"
        ) from exc
    _validate_schema(data)

    run = _table(data, "run")
    population = _table(data, "population")
    diagnostics = _table(data, "diagnostics")
    grouping = _table(data, "grouping")
    factors = _table(data, "factors")
    outputs = _table(data, "outputs")
    expected_inputs = _table(data, "expected_inputs")
    paths = _table(data, "paths")

    mode = _string(run, "mode", "[run]")
    if mode not in {"validate", "test", "full"}:
        raise RecipeConfigError("[run].mode must be validate, test, or full")

    full_output_root = _confined_path(
        root, _string(paths, "output_root", "[paths]"), "[paths].output_root"
    )
    test_output_root = _confined_path(
        root,
        _string(paths, "test_output_root", "[paths]"),
        "[paths].test_output_root",
    )
    output_root = test_output_root if mode == "test" else full_output_root

    expected = {
        key: _string(expected_inputs, key, "[expected_inputs]")
        for key in (
            "base_config_sha256",
            "source_sha256",
            "release_manifest_sha256",
            "release_ledger_sha256",
            "release_checksums_sha256",
        )
    }
    for key, value in expected.items():
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise RecipeConfigError(
                f"[expected_inputs].{key} must be a lowercase SHA-256 digest"
            )

    release_package = _confined_path(
        root,
        _string(paths, "release_package", "[paths]"),
        "[paths].release_package",
    )
    for candidate in (full_output_root, test_output_root):
        try:
            candidate.relative_to(release_package)
        except ValueError:
            continue
        raise RecipeConfigError(
            "Diagnostic output roots must remain outside the immutable release package"
        )

    config = GrainYieldResponseConfig(
        config_path=path,
        project_root=root,
        raw=data,
        mode=mode,
        overwrite=_boolean(run, "overwrite", "[run]"),
        random_seed=_positive_int(run, "random_seed", "[run]"),
        fail_fast=_boolean(run, "fail_fast", "[run]"),
        include_raw_sensitivity=_boolean(run, "include_raw_sensitivity", "[run]"),
        run_mixed_model=_boolean(run, "run_mixed_model", "[run]"),
        mixed_model_required=_boolean(run, "mixed_model_required", "[run]"),
        release_verification_policy=_string(
            run, "release_verification_policy", "[run]"
        ),
        fit_pooled_linear=_boolean(
            diagnostics, "fit_pooled_linear", "[diagnostics]"
        ),
        fit_pooled_quadratic=_boolean(
            diagnostics, "fit_pooled_quadratic", "[diagnostics]"
        ),
        fit_within_series=_boolean(
            diagnostics, "fit_within_series", "[diagnostics]"
        ),
        fit_zero_n_delta=_boolean(
            diagnostics, "fit_zero_n_delta", "[diagnostics]"
        ),
        fit_series_linear=_boolean(
            diagnostics, "fit_series_linear", "[diagnostics]"
        ),
        leave_one_series_out=_boolean(
            diagnostics, "leave_one_series_out", "[diagnostics]"
        ),
        source_name=_string(population, "source_name", "[population]"),
        analysis_population=_string(
            population, "analysis_population", "[population]"
        ),
        required_release_status=_string(
            population, "required_release_status", "[population]"
        ),
        minimum_observations=_positive_int(
            population, "minimum_observations", "[population]"
        ),
        minimum_distinct_n_levels=_positive_int(
            population, "minimum_distinct_n_levels", "[population]"
        ),
        n_level_tolerance_kg_ha=_number(
            population, "n_level_tolerance_kg_ha", "[population]"
        ),
        zero_n_tolerance_kg_ha=_number(
            population, "zero_n_tolerance_kg_ha", "[population]"
        ),
        series_key=_string(grouping, "series_key", "[grouping]"),
        trial_key=_string(grouping, "trial_key", "[grouping]"),
        study_key=_string(grouping, "study_key", "[grouping]"),
        categorical_factors=_string_list(factors, "categorical", "[factors]"),
        numeric_factors=_string_list(factors, "numeric", "[factors]"),
        minimum_level_observations=_positive_int(
            factors, "minimum_level_observations", "[factors]"
        ),
        minimum_modifier_series=_positive_int(
            factors, "minimum_modifier_series", "[factors]"
        ),
        table_formats=_string_list(outputs, "table_formats", "[outputs]"),
        figure_formats=_string_list(outputs, "figure_formats", "[outputs]"),
        figure_dpi=_positive_int(outputs, "figure_dpi", "[outputs]"),
        figure_width_inches=_number(
            outputs, "figure_width_inches", "[outputs]", minimum=1.0
        ),
        figure_height_inches=_number(
            outputs, "figure_height_inches", "[outputs]", minimum=1.0
        ),
        expected_inputs=expected,
        base_config_path=_confined_path(
            root, _string(paths, "base_config", "[paths]"), "[paths].base_config"
        ),
        release_package=release_package,
        output_root=output_root,
        full_output_root=full_output_root,
        test_output_root=test_output_root,
        r_stage_path=_confined_path(
            root, _string(paths, "r_stage", "[paths]"), "[paths].r_stage"
        ),
    )
    if config.mixed_model_required and not config.run_mixed_model:
        raise RecipeConfigError(
            "[run].mixed_model_required cannot be true when run_mixed_model is false"
        )
    if config.release_verification_policy not in {"required", "record_failure"}:
        raise RecipeConfigError(
            "[run].release_verification_policy must be required or record_failure"
        )
    if config.analysis_population != "governed_overlay":
        raise RecipeConfigError(
            "[population].analysis_population must be governed_overlay"
        )
    if config.table_formats != ("csv",):
        raise RecipeConfigError(
            "[outputs].table_formats currently supports only ['csv']"
        )
    if not config.figure_formats or any(
        value not in {"jpeg", "png"} for value in config.figure_formats
    ):
        raise RecipeConfigError(
            "[outputs].figure_formats must contain only jpeg or png"
        )
    factor_names = config.categorical_factors + config.numeric_factors
    if len(factor_names) != len(set(factor_names)):
        raise RecipeConfigError(
            "[factors] categorical and numeric lists must be disjoint"
        )
    if check_files:
        for required in (config.base_config_path, config.r_stage_path):
            if required.is_symlink() or not required.is_file():
                raise RecipeConfigError(f"Required file is missing: {required}")
        if config.release_package.is_symlink() or not config.release_package.is_dir():
            raise RecipeConfigError(
                f"Release package is missing: {config.release_package}"
            )
        bound_files = {
            "base_config_sha256": config.base_config_path,
            "release_manifest_sha256": config.release_package / "run_manifest.json",
            "release_ledger_sha256": config.release_package
            / "tables/quality/analysis_eligibility_ledger.csv",
            "release_checksums_sha256": config.release_package / "CHECKSUMS.sha256",
        }
        for key, required in bound_files.items():
            if required.is_symlink() or not required.is_file():
                raise RecipeConfigError(f"Required file is missing: {required}")
            actual = _sha256_file(required)
            if actual != config.expected_inputs[key]:
                raise RecipeConfigError(
                    f"Input hash mismatch for {key}: expected "
                    f"{config.expected_inputs[key]}, got {actual}"
                )
    return config
