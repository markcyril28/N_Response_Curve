"""Validation of descriptive_statisticsCONFIG.toml.

Fails closed in the house style: every table is required, every key inside every
table must be declared here, and every path must resolve inside the project
root. The one structural difference from the sibling recipe is ``[agronomic]``,
which mixes scalar tuning keys with one binding sub-table per profiled source.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import tomllib
from typing import Any, Mapping


class RecipeConfigError(ValueError):
    """Raised when the recipe configuration is invalid."""


KNOWN_MODES = ("validate", "test", "full")
SUPPORTED_TABLE_FORMATS = frozenset({"csv"})
SUPPORTED_FIGURE_FORMATS = frozenset({"jpeg", "png"})

# Keys of an [agronomic.<source>] sub-table. ``nitrogen_rate`` and ``yield_t_ha``
# are required; the rest are optional because not every dataset records them.
_SINGLE_BINDING_KEYS = (
    "nitrogen_rate",
    "yield_t_ha",
    "yield_kg_ha",
    "zero_n_yield_t_ha",
    "year",
)
_LIST_BINDING_KEYS = ("context", "grouping", "series")
# Width of the applied-N bands the agronomic profile derives from the recorded
# rate. 50 kg N ha⁻¹ reads as the ladder an agronomist would name (0, then up to
# 50, up to 100, …) rather than as the recorded rates, of which the long-running
# trials carry thirty.
_DEFAULT_APPLIED_N_BAND_WIDTH_KG_HA = 50.0
# Span of the year bands the agronomic profile derives from the recorded year.
# Unrelated to `temporal_coverage`, which reports one row per calendar year and
# is the record of coverage; this is a reading of era as a context field, and the
# two are free to disagree in resolution. Five years rather than a decade because
# the shortest-running dataset spans six years, and a decade would collapse it to
# a single bar stating nothing but its own scope.
_DEFAULT_YEAR_BAND_SPAN_YEARS = 5
_REQUIRED_BINDING_KEYS = ("nitrogen_rate", "yield_t_ha")


@dataclass(frozen=True)
class DescriptiveStatisticsConfig:
    config_path: Path
    project_root: Path
    raw: Mapping[str, Any]
    # [run]
    mode: str
    overwrite: bool
    random_seed: int
    fail_fast: bool
    include_row_level_observations: bool
    # [sources]
    profiled_sources: tuple[str, ...]
    # [privacy]
    suppressed_headers: frozenset[str]
    # [structure]
    numeric_parse_threshold: float
    maximum_categorical_cardinality: int
    example_values_per_column: int
    # [numeric]
    outlier_iqr_multipliers: tuple[float, ...]
    quantiles: tuple[float, ...]
    histogram_bins: int
    minimum_numeric_observations: int
    # [categorical]
    minimum_level_count: int
    maximum_levels_reported: int
    # [agronomic]
    nitrogen_bin_width_kg_ha: float
    applied_n_band_width_kg_ha: float
    year_band_span_years: int
    zero_n_tolerance_kg_ha: float
    n_level_tolerance_kg_ha: float
    minimum_group_observations: int
    agronomic_bindings: Mapping[str, Mapping[str, Any]]
    # [outputs]
    table_formats: tuple[str, ...]
    figure_formats: tuple[str, ...]
    figure_dpi: int
    figure_width_inches: float
    figure_height_inches: float
    # [expected_inputs]
    expected_source_sha256: Mapping[str, str]
    # [paths]
    base_config_path: Path
    output_root: Path
    full_output_root: Path
    test_output_root: Path

    @property
    def writes_outputs(self) -> bool:
        return self.mode in {"test", "full"}

    @property
    def primary_figure_format(self) -> str:
        return self.figure_formats[0]


def _table(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name)
    if not isinstance(value, Mapping):
        raise RecipeConfigError(f"Missing or invalid [{name}] table")
    return value


_TABLE_KEYS: Mapping[str, frozenset[str]] = {
    "run": frozenset(
        {
            "mode",
            "overwrite",
            "random_seed",
            "fail_fast",
            "include_row_level_observations",
        }
    ),
    "sources": frozenset({"profiled"}),
    "privacy": frozenset({"suppressed_headers"}),
    "structure": frozenset(
        {
            "numeric_parse_threshold",
            "maximum_categorical_cardinality",
            "example_values_per_column",
        }
    ),
    "numeric": frozenset(
        {
            "outlier_iqr_multipliers",
            "quantiles",
            "histogram_bins",
            "minimum_observations",
        }
    ),
    "categorical": frozenset({"minimum_level_count", "maximum_levels_reported"}),
    "agronomic": frozenset(
        {
            "nitrogen_bin_width_kg_ha",
            "applied_n_band_width_kg_ha",
            "year_band_span_years",
            "zero_n_tolerance_kg_ha",
            "n_level_tolerance_kg_ha",
            "minimum_group_observations",
        }
    ),
    "outputs": frozenset(
        {
            "table_formats",
            "figure_formats",
            "figure_dpi",
            "figure_width_inches",
            "figure_height_inches",
        }
    ),
    "expected_inputs": frozenset(),  # validated against [sources].profiled
    "paths": frozenset({"base_config", "output_root", "test_output_root"}),
}


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


def _integer(
    table: Mapping[str, Any], key: str, where: str, *, minimum: int = 1
) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RecipeConfigError(f"{where}.{key} must be an integer of at least {minimum}")
    return value


def _number(
    table: Mapping[str, Any],
    key: str,
    where: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeConfigError(f"{where}.{key} must be a number")
    result = float(value)
    if result < minimum:
        raise RecipeConfigError(f"{where}.{key} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise RecipeConfigError(f"{where}.{key} must be at most {maximum}")
    return result


def _string_list(
    table: Mapping[str, Any], key: str, where: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    value = table.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise RecipeConfigError(f"{where}.{key} must be a list of non-empty strings")
    values = tuple(item.strip() for item in value)
    if not values and not allow_empty:
        raise RecipeConfigError(f"{where}.{key} must not be empty")
    if len(values) != len(set(values)):
        raise RecipeConfigError(f"{where}.{key} must not contain duplicates")
    return values


def _number_list(
    table: Mapping[str, Any],
    key: str,
    where: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> tuple[float, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not value:
        raise RecipeConfigError(f"{where}.{key} must be a nonempty list of numbers")
    values: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise RecipeConfigError(f"{where}.{key} must contain only numbers")
        number = float(item)
        if number < minimum or (maximum is not None and number > maximum):
            bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
            raise RecipeConfigError(f"{where}.{key} values must be {bound}")
        values.append(number)
    if len(values) != len(set(values)):
        raise RecipeConfigError(f"{where}.{key} must not contain duplicates")
    return tuple(sorted(values))


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


def _validate_binding_entry(entry: Any, where: str) -> None:
    if not isinstance(entry, Mapping):
        raise RecipeConfigError(f"{where} must be an inline table")
    unknown = sorted(set(entry) - {"position", "header", "label"})
    if unknown:
        raise RecipeConfigError(f"Unknown keys in {where}: {unknown}")
    position = entry.get("position")
    if isinstance(position, bool) or not isinstance(position, int) or position < 1:
        raise RecipeConfigError(f"{where}.position must be a positive integer")
    header = entry.get("header")
    if not isinstance(header, str) or header == "":
        raise RecipeConfigError(f"{where}.header must be a non-empty string")
    label = entry.get("label")
    if label is not None and (not isinstance(label, str) or not label.strip()):
        raise RecipeConfigError(f"{where}.label must be a non-empty string when present")


def _validate_agronomic_bindings(
    agronomic: Mapping[str, Any],
    profiled_sources: tuple[str, ...],
) -> Mapping[str, Mapping[str, Any]]:
    scalar_keys = _TABLE_KEYS["agronomic"]
    sub_tables = {
        key: value for key, value in agronomic.items() if isinstance(value, Mapping)
    }
    unknown_scalars = sorted(set(agronomic) - scalar_keys - set(sub_tables))
    if unknown_scalars:
        raise RecipeConfigError(f"Unknown keys in [agronomic]: {unknown_scalars}")
    unknown_sources = sorted(set(sub_tables) - set(profiled_sources))
    if unknown_sources:
        raise RecipeConfigError(
            f"[agronomic] declares bindings for unprofiled sources: {unknown_sources}"
        )
    missing_sources = sorted(set(profiled_sources) - set(sub_tables))
    if missing_sources:
        raise RecipeConfigError(
            f"[agronomic] is missing a binding table for: {missing_sources}"
        )

    for source_name, table in sub_tables.items():
        where = f"[agronomic.{source_name}]"
        allowed = set(_SINGLE_BINDING_KEYS) | set(_LIST_BINDING_KEYS)
        unknown = sorted(set(table) - allowed)
        if unknown:
            raise RecipeConfigError(f"Unknown keys in {where}: {unknown}")
        for key in _REQUIRED_BINDING_KEYS:
            if key not in table:
                raise RecipeConfigError(f"{where}.{key} is required")
        for key in _SINGLE_BINDING_KEYS:
            if key in table:
                _validate_binding_entry(table[key], f"{where}.{key}")
        for key in _LIST_BINDING_KEYS:
            entries = table.get(key, [])
            if not isinstance(entries, list):
                raise RecipeConfigError(f"{where}.{key} must be an array of inline tables")
            labels: list[str] = []
            for index, entry in enumerate(entries):
                _validate_binding_entry(entry, f"{where}.{key}[{index}]")
                labels.append(str(entry.get("label", f"{key}_{index}")))
            if len(labels) != len(set(labels)):
                raise RecipeConfigError(f"{where}.{key} labels must be unique")
    return sub_tables


def _validate_schema(data: Mapping[str, Any]) -> None:
    unknown_tables = sorted(set(data) - set(_TABLE_KEYS))
    if unknown_tables:
        raise RecipeConfigError(f"Unknown TOML tables: {unknown_tables}")
    missing_tables = sorted(set(_TABLE_KEYS) - set(data))
    if missing_tables:
        raise RecipeConfigError(f"Missing TOML tables: {missing_tables}")
    for table_name, allowed in _TABLE_KEYS.items():
        if table_name in {"agronomic", "expected_inputs"}:
            continue  # validated against [sources].profiled below
        table = _table(data, table_name)
        unknown = sorted(set(table) - allowed)
        if unknown:
            raise RecipeConfigError(f"Unknown keys in [{table_name}]: {unknown}")


def load_recipe_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    check_files: bool = True,
) -> DescriptiveStatisticsConfig:
    """Parse and validate the recipe configuration."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise RecipeConfigError(f"Recipe configuration does not exist: {path}")
    root = Path(project_root or path.parent).expanduser().resolve()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise RecipeConfigError(f"Recipe configuration is not valid TOML: {exc}") from exc
    _validate_schema(data)

    run = _table(data, "run")
    mode = _string(run, "mode", "[run]")
    if mode not in KNOWN_MODES:
        raise RecipeConfigError(f"[run].mode must be one of {KNOWN_MODES}: {mode!r}")

    sources_table = _table(data, "sources")
    profiled_sources = _string_list(sources_table, "profiled", "[sources]")

    privacy = _table(data, "privacy")
    suppressed_headers = frozenset(
        _string_list(privacy, "suppressed_headers", "[privacy]", allow_empty=True)
    )

    structure = _table(data, "structure")
    numeric_parse_threshold = _number(
        structure, "numeric_parse_threshold", "[structure]", minimum=0.0, maximum=1.0
    )
    maximum_categorical_cardinality = _integer(
        structure, "maximum_categorical_cardinality", "[structure]"
    )
    example_values_per_column = _integer(
        structure, "example_values_per_column", "[structure]"
    )

    numeric = _table(data, "numeric")
    outlier_iqr_multipliers = _number_list(
        numeric, "outlier_iqr_multipliers", "[numeric]", minimum=0.0
    )
    quantiles = _number_list(numeric, "quantiles", "[numeric]", minimum=0.0, maximum=1.0)
    required_quantiles = {0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99}
    if not required_quantiles.issubset(set(quantiles)):
        raise RecipeConfigError(
            "[numeric].quantiles must include "
            + ", ".join(f"{value:g}" for value in sorted(required_quantiles))
            + " because numeric_summary declares those columns"
        )
    histogram_bins = _integer(numeric, "histogram_bins", "[numeric]", minimum=2)
    minimum_numeric_observations = _integer(
        numeric, "minimum_observations", "[numeric]", minimum=1
    )

    categorical = _table(data, "categorical")
    minimum_level_count = _integer(categorical, "minimum_level_count", "[categorical]")
    maximum_levels_reported = _integer(
        categorical, "maximum_levels_reported", "[categorical]"
    )

    agronomic = _table(data, "agronomic")
    agronomic_bindings = _validate_agronomic_bindings(agronomic, profiled_sources)
    nitrogen_bin_width_kg_ha = _number(
        agronomic, "nitrogen_bin_width_kg_ha", "[agronomic]", minimum=1e-6
    )
    # Defaulted rather than required, so a configuration written before this key
    # existed still loads. Validated through the same helper once defaulted, so
    # a value that is present is held to the same rules as any other.
    applied_n_band_width_kg_ha = _number(
        {
            "applied_n_band_width_kg_ha": agronomic.get(
                "applied_n_band_width_kg_ha", _DEFAULT_APPLIED_N_BAND_WIDTH_KG_HA
            )
        },
        "applied_n_band_width_kg_ha",
        "[agronomic]",
        minimum=1e-6,
    )
    # Defaulted on the same terms as the band width above.
    year_band_span_years = _integer(
        {
            "year_band_span_years": agronomic.get(
                "year_band_span_years", _DEFAULT_YEAR_BAND_SPAN_YEARS
            )
        },
        "year_band_span_years",
        "[agronomic]",
    )
    zero_n_tolerance_kg_ha = _number(
        agronomic, "zero_n_tolerance_kg_ha", "[agronomic]", minimum=0.0
    )
    n_level_tolerance_kg_ha = _number(
        agronomic, "n_level_tolerance_kg_ha", "[agronomic]", minimum=0.0
    )
    minimum_group_observations = _integer(
        agronomic, "minimum_group_observations", "[agronomic]"
    )

    outputs = _table(data, "outputs")
    table_formats = _string_list(outputs, "table_formats", "[outputs]")
    unsupported_tables = sorted(set(table_formats) - SUPPORTED_TABLE_FORMATS)
    if unsupported_tables:
        raise RecipeConfigError(
            f"[outputs].table_formats supports only {sorted(SUPPORTED_TABLE_FORMATS)}: "
            f"{unsupported_tables}"
        )
    figure_formats = _string_list(outputs, "figure_formats", "[outputs]")
    unsupported_figures = sorted(set(figure_formats) - SUPPORTED_FIGURE_FORMATS)
    if unsupported_figures:
        raise RecipeConfigError(
            f"[outputs].figure_formats supports only {sorted(SUPPORTED_FIGURE_FORMATS)}: "
            f"{unsupported_figures}"
        )
    figure_dpi = _integer(outputs, "figure_dpi", "[outputs]", minimum=72)
    figure_width_inches = _number(
        outputs, "figure_width_inches", "[outputs]", minimum=1.0
    )
    figure_height_inches = _number(
        outputs, "figure_height_inches", "[outputs]", minimum=1.0
    )

    expected_inputs = _table(data, "expected_inputs")
    expected_keys = {f"{name}_sha256" for name in profiled_sources}
    unknown_expected = sorted(set(expected_inputs) - expected_keys)
    if unknown_expected:
        raise RecipeConfigError(
            f"Unknown keys in [expected_inputs]: {unknown_expected}. Declare exactly "
            "one <source>_sha256 per profiled source."
        )
    expected_source_sha256: dict[str, str] = {}
    for source_name in profiled_sources:
        key = f"{source_name}_sha256"
        if key not in expected_inputs:
            raise RecipeConfigError(f"[expected_inputs].{key} is required")
        digest = _string(expected_inputs, key, "[expected_inputs]")
        if len(digest) != 64 or digest != digest.lower() or not all(
            character in "0123456789abcdef" for character in digest
        ):
            raise RecipeConfigError(
                f"[expected_inputs].{key} must be a lowercase 64-character SHA-256"
            )
        expected_source_sha256[source_name] = digest

    paths = _table(data, "paths")
    base_config_path = _confined_path(
        root, _string(paths, "base_config", "[paths]"), "[paths].base_config"
    )
    full_output_root = _confined_path(
        root, _string(paths, "output_root", "[paths]"), "[paths].output_root"
    )
    test_output_root = _confined_path(
        root, _string(paths, "test_output_root", "[paths]"), "[paths].test_output_root"
    )
    if full_output_root == test_output_root:
        raise RecipeConfigError(
            "[paths].output_root and [paths].test_output_root must differ"
        )
    output_root = test_output_root if mode == "test" else full_output_root

    if check_files and not base_config_path.is_file():
        raise RecipeConfigError(
            f"[paths].base_config does not exist: {base_config_path}"
        )

    return DescriptiveStatisticsConfig(
        config_path=path,
        project_root=root,
        raw=data,
        mode=mode,
        overwrite=_boolean(run, "overwrite", "[run]"),
        random_seed=_integer(run, "random_seed", "[run]", minimum=0),
        fail_fast=_boolean(run, "fail_fast", "[run]"),
        include_row_level_observations=_boolean(
            run, "include_row_level_observations", "[run]"
        ),
        profiled_sources=profiled_sources,
        suppressed_headers=suppressed_headers,
        numeric_parse_threshold=numeric_parse_threshold,
        maximum_categorical_cardinality=maximum_categorical_cardinality,
        example_values_per_column=example_values_per_column,
        outlier_iqr_multipliers=outlier_iqr_multipliers,
        quantiles=quantiles,
        histogram_bins=histogram_bins,
        minimum_numeric_observations=minimum_numeric_observations,
        minimum_level_count=minimum_level_count,
        maximum_levels_reported=maximum_levels_reported,
        nitrogen_bin_width_kg_ha=nitrogen_bin_width_kg_ha,
        applied_n_band_width_kg_ha=applied_n_band_width_kg_ha,
        year_band_span_years=year_band_span_years,
        zero_n_tolerance_kg_ha=zero_n_tolerance_kg_ha,
        n_level_tolerance_kg_ha=n_level_tolerance_kg_ha,
        minimum_group_observations=minimum_group_observations,
        agronomic_bindings=agronomic_bindings,
        table_formats=table_formats,
        figure_formats=figure_formats,
        figure_dpi=figure_dpi,
        figure_width_inches=figure_width_inches,
        figure_height_inches=figure_height_inches,
        expected_source_sha256=expected_source_sha256,
        base_config_path=base_config_path,
        output_root=output_root,
        full_output_root=full_output_root,
        test_output_root=test_output_root,
    )


def config_sha256(config: DescriptiveStatisticsConfig) -> str:
    """Digest of the recipe configuration file as it was read."""

    return _sha256_file(config.config_path)
