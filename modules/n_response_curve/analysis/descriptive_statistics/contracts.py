"""Frozen artifact registry and shared row types for the descriptive profile.

Every table and figure the recipe may emit is declared here exactly once, with
its group, its ordered column schema, and the source classification it inherits.
The analysis modules build frames against these schemas, the reporting module
writes them, and ``verify_profile_bundle`` reconciles what landed on disk with
what is declared here. Nothing else may invent an artifact path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


SCHEMA_VERSION = "dataset-descriptive-statistics-v2"
BUNDLE_STATUS = "diagnostic_internal_not_release"
MANIFEST_NAME = "run_manifest.json"
CHECKSUMS_NAME = "CHECKSUMS.sha256"
SUMMARY_NAME = "summary.md"

TABLE_GROUPS = ("structure", "numeric", "categorical", "agronomic", "crosscut")
FIGURE_GROUPS = ("structure", "numeric", "categorical", "agronomic", "crosscut")

RESERVED_PATHS = frozenset({MANIFEST_NAME, CHECKSUMS_NAME, SUMMARY_NAME})


class ProfileContractError(ValueError):
    """Raised when an artifact does not match its declared contract."""


@dataclass(frozen=True)
class TableSpec:
    """One declared output table and the exact columns it must carry."""

    name: str
    group: str
    columns: tuple[str, ...]
    title: str
    # ``per_source`` tables carry one row per source (or per source-column);
    # ``crosscut`` tables mix sources and are labeled restricted when any
    # profiled source is restricted.
    scope: str = "per_source"
    optional: bool = False

    @property
    def relative_path(self) -> str:
        return f"{self.group}/{self.name}.csv"


@dataclass(frozen=True)
class FigureSpec:
    """One declared output figure."""

    name: str
    group: str
    title: str
    optional: bool = False
    # Set when the panel draws exactly one source. It carries only that
    # source's data, so it is classified on that source alone rather than
    # inheriting the run's restricted-if-any fallback — a core_trial_data-only
    # panel is internal, and labeling it restricted would over-classify it.
    #
    # Declared here, one spec per source, rather than expanded from the source
    # list at run time: this module's invariant is that every artifact a run may
    # emit is declared exactly once, and a spec that fans out into N artifacts
    # breaks it. A source that is not profiled simply skips, and the skip is
    # recorded in the manifest.
    #
    # It also files the figure: a single-dataset panel is written to a directory
    # named for its dataset, and the figures that draw every dataset stay at the
    # root of their semantic group beside that group's tables. A distinct output
    # directory name may be declared when the human-facing dataset name differs
    # from its stable internal source key; classification still follows
    # ``source_name``.
    source_name: str | None = None
    output_directory_name: str | None = None
    output_filename_stem: str | None = None

    def relative_path(self, extension: str) -> str:
        directory_name = self.output_directory_name or self.source_name
        directory = f"{directory_name}/" if directory_name else ""
        filename_stem = self.output_filename_stem or self.name
        return f"{self.group}/{directory}{filename_stem}.{extension}"


# --------------------------------------------------------------------------
# Column vocabularies shared by more than one table.
# --------------------------------------------------------------------------

_SOURCE_KEYS = ("source_name", "data_classification")
_COLUMN_KEYS = _SOURCE_KEYS + ("position", "raw_column_id", "header_label")


TABLE_SPECS: tuple[TableSpec, ...] = (
    # ---------------- structure ----------------
    TableSpec(
        name="source_inventory",
        group="structure",
        title="Registered dataset inventory",
        columns=_SOURCE_KEYS
        + (
            "source_type",
            "source_family",
            "country_code",
            "shape_adapter_version",
            "source_encoding",
            "representation_basis",
            "representation_basis_status",
            "restricted_access_status",
            "source_sha256",
            "source_relative_path",
            "physical_column_count",
            "data_row_count",
            "blank_row_count",
            "named_header_count",
            "blank_header_count",
            "duplicated_header_count",
            "numeric_column_count",
            "categorical_column_count",
            "identifier_column_count",
            "empty_column_count",
            "suppressed_column_count",
            "total_cells",
            "populated_cells",
            "overall_fill_rate",
        ),
    ),
    TableSpec(
        name="column_inventory",
        group="structure",
        title="Physical column inventory by position",
        columns=_COLUMN_KEYS
        + (
            "header_raw",
            "header_status",
            "duplicate_group_size",
            "value_kind",
            "suppressed",
            "suppression_reason",
            "nonblank_count",
            "blank_count",
            "fill_rate",
            "distinct_nonblank_count",
            "numeric_parse_rate",
            "example_values",
        ),
    ),
    TableSpec(
        name="missingness_profile",
        group="structure",
        title="Per-column missingness ranked worst first",
        columns=_COLUMN_KEYS
        + (
            "value_kind",
            "blank_count",
            "nonblank_count",
            "missing_rate",
            "fill_rate",
            "is_wholly_empty",
        ),
    ),
    TableSpec(
        name="row_completeness",
        group="structure",
        title="Distribution of per-row completeness",
        columns=_SOURCE_KEYS
        + (
            "statistic",
            "populated_columns",
            "populated_share",
        ),
    ),
    TableSpec(
        name="header_anomalies",
        group="structure",
        title="Blank and duplicated physical headers",
        columns=_COLUMN_KEYS
        + (
            "header_raw",
            "anomaly",
            "duplicate_group_size",
            "duplicate_positions",
        ),
    ),
    # ---------------- numeric ----------------
    TableSpec(
        name="numeric_summary",
        group="numeric",
        title="Descriptive statistics for numeric columns",
        columns=_COLUMN_KEYS
        + (
            "count",
            "missing_count",
            "unparsed_count",
            "mean",
            "std_dev",
            "coefficient_of_variation",
            "minimum",
            "p01",
            "p05",
            "q1",
            "median",
            "q3",
            "p95",
            "p99",
            "maximum",
            "range",
            "iqr",
            "median_absolute_deviation",
            "skewness",
            "excess_kurtosis",
            "zero_count",
            "negative_count",
            "distinct_count",
            "sum",
        ),
    ),
    TableSpec(
        name="numeric_outlier_audit",
        group="numeric",
        title="Tukey fence outlier counts per numeric column",
        columns=_COLUMN_KEYS
        + (
            "iqr_multiplier",
            "lower_fence",
            "upper_fence",
            "below_fence_count",
            "above_fence_count",
            "outlier_count",
            "outlier_rate",
        ),
    ),
    TableSpec(
        name="numeric_distribution_bins",
        group="numeric",
        title="Histogram bins for profiled agronomic numeric columns",
        columns=_COLUMN_KEYS
        + (
            "bin_index",
            "bin_lower",
            "bin_upper",
            "count",
            "share",
        ),
    ),
    # ---------------- categorical ----------------
    TableSpec(
        name="categorical_summary",
        group="categorical",
        title="Descriptive statistics for categorical columns",
        columns=_COLUMN_KEYS
        + (
            "nonblank_count",
            "blank_count",
            "distinct_count",
            "is_identifier_like",
            "mode_value",
            "mode_count",
            "mode_share",
            "singleton_level_count",
            "shannon_entropy_bits",
            "normalized_entropy",
            "levels_reported",
            "levels_withheld_below_threshold",
        ),
    ),
    TableSpec(
        name="categorical_levels",
        group="categorical",
        title="Level frequencies at or above the reporting threshold",
        columns=_COLUMN_KEYS
        + (
            "level_rank",
            "level_value",
            "count",
            "share",
            "cumulative_share",
        ),
    ),
    # ---------------- agronomic ----------------
    TableSpec(
        name="nitrogen_rate_profile",
        group="agronomic",
        title="Inorganic N-rate distribution per dataset",
        columns=_SOURCE_KEYS
        + (
            "column_header",
            "observation_count",
            "missing_count",
            "distinct_rate_count",
            "minimum_kg_ha",
            "q1_kg_ha",
            "median_kg_ha",
            "mean_kg_ha",
            "q3_kg_ha",
            "maximum_kg_ha",
            "std_dev_kg_ha",
            "span_kg_ha",
            "zero_n_observation_count",
            "zero_n_share",
            "distinct_rates",
        ),
    ),
    TableSpec(
        name="yield_profile",
        group="agronomic",
        title="Grain-yield distribution per dataset on a t/ha basis",
        columns=_SOURCE_KEYS
        + (
            "column_header",
            "unit_basis",
            "unit_lineage",
            "observation_count",
            "missing_count",
            "minimum_t_ha",
            "q1_t_ha",
            "median_t_ha",
            "mean_t_ha",
            "q3_t_ha",
            "maximum_t_ha",
            "std_dev_t_ha",
            "coefficient_of_variation",
            "skewness",
            "implausible_low_count",
            "implausible_high_count",
        ),
    ),
    TableSpec(
        name="yield_by_nitrogen_bin",
        group="agronomic",
        title="Grain yield summarized within N-rate bins",
        columns=_SOURCE_KEYS
        + (
            "bin_index",
            "n_lower_kg_ha",
            "n_upper_kg_ha",
            "n_midpoint_kg_ha",
            "observation_count",
            "mean_n_kg_ha",
            "mean_yield_t_ha",
            "median_yield_t_ha",
            "std_dev_yield_t_ha",
            "minimum_yield_t_ha",
            "maximum_yield_t_ha",
        ),
    ),
    TableSpec(
        name="temporal_coverage",
        group="agronomic",
        title="Observation counts by recorded year",
        columns=_SOURCE_KEYS
        + (
            "year",
            "observation_count",
            "share",
            "mean_n_kg_ha",
            "mean_yield_t_ha",
        ),
    ),
    TableSpec(
        name="context_composition",
        group="agronomic",
        title="Composition of agronomic context fields",
        columns=_SOURCE_KEYS
        + (
            "context_label",
            "column_header",
            "level_value",
            "observation_count",
            "share",
            "mean_n_kg_ha",
            "mean_yield_t_ha",
            "median_yield_t_ha",
        ),
    ),
    TableSpec(
        name="zero_nitrogen_checks",
        group="agronomic",
        title="Zero-N check observations per dataset",
        columns=_SOURCE_KEYS
        + (
            "basis",
            "observation_count",
            "mean_yield_t_ha",
            "median_yield_t_ha",
            "std_dev_yield_t_ha",
            "minimum_yield_t_ha",
            "maximum_yield_t_ha",
        ),
    ),
    # ---------------- crosscut ----------------
    TableSpec(
        name="source_comparability",
        group="crosscut",
        title="The three datasets side by side",
        scope="crosscut",
        columns=(
            "source_name",
            "data_classification",
            "representation_basis",
            "harmonized_observation_count",
            "grouping_series_count",
            "n_rate_min_kg_ha",
            "n_rate_max_kg_ha",
            "distinct_n_rate_count",
            "yield_min_t_ha",
            "yield_median_t_ha",
            "yield_max_t_ha",
            "year_min",
            "year_max",
            "year_span",
            "yield_unit_lineage",
            "n_rate_unit_lineage",
            "series_key_basis",
            "dropped_row_count",
            "comparability_note",
        ),
    ),
    TableSpec(
        name="nitrogen_ladder_geometry",
        group="crosscut",
        title="N-ladder geometry of grouping series",
        scope="crosscut",
        columns=(
            "source_name",
            "data_classification",
            "series_count",
            "median_levels_per_series",
            "minimum_levels_per_series",
            "maximum_levels_per_series",
            "series_with_zero_n_count",
            "series_with_zero_n_share",
            "median_span_kg_ha",
            "median_step_kg_ha",
            "balanced_ladder_series_count",
            "single_level_series_count",
            "resolved_observation_count",
            "unresolved_series_observation_count",
            "series_key_basis",
        ),
    ),
    TableSpec(
        name="unit_lineage_audit",
        group="crosscut",
        title="How each reported quantity reached its common unit",
        scope="crosscut",
        columns=(
            "source_name",
            "data_classification",
            "quantity",
            "source_column_header",
            "source_unit",
            "target_unit",
            "conversion",
            "converted_observation_count",
            "native_observation_count",
            "note",
        ),
    ),
    TableSpec(
        name="harmonized_observations",
        group="crosscut",
        title="Row-level harmonized N-yield observations across datasets",
        scope="crosscut",
        optional=True,
        columns=(
            "source_name",
            "data_classification",
            "source_row_number",
            "study_key",
            "trial_key",
            "series_key",
            "year",
            "n_rate_kg_ha",
            "yield_t_ha",
            "yield_unit_lineage",
            "is_zero_n",
            "is_series_resolved",
        ),
    ),
)


FIGURE_SPECS: tuple[FigureSpec, ...] = (
    FigureSpec("dataset_scale", "structure", "Rows and physical columns per dataset"),
    FigureSpec("column_fill_profile", "structure", "Column fill-rate profile per dataset"),
    FigureSpec("numeric_spread_overview", "numeric", "Standardized spread of key numeric columns"),
    FigureSpec("categorical_cardinality", "categorical", "Cardinality and entropy of categorical columns"),
    FigureSpec(
        "nitrogen_rate_distribution",
        "agronomic",
        "Inorganic N-rate distribution per dataset",
    ),
    FigureSpec(
        "nitrogen_rate_distribution_with_separate_farmers_practice",
        "agronomic",
        "Inorganic N-rate distribution with separate Farmer's Practice bars",
    ),
    FigureSpec("yield_distribution", "agronomic", "Grain-yield distribution per dataset"),
    # One single-dataset panel per profiled source, drawn on axes shared with
    # each other and with the combined figure above, so the three read as a set.
    FigureSpec(
        "yield_distribution_core_trial_data",
        "agronomic",
        "Grain-yield distribution — Literature Extracted Datasets",
        source_name="core_trial_data",
        output_directory_name="literature_extracted_datasets",
        output_filename_stem="yield_distribution_literature_extracted_datasets",
    ),
    FigureSpec(
        "yield_distribution_ph_combined_nopt_rcm",
        "agronomic",
        "Grain-yield distribution — ph_combined_nopt_rcm (legacy NOPT-only alias)",
        source_name="ph_combined_nopt_rcm",
    ),
    # The combined PH source records NOPT and Farmer's Practice yields in
    # sibling columns on the same physical rows. Keep both treatment-arm views
    # explicit in the artifact contract. The unqualified panel above remains a
    # path-compatible alias for presentation material that predates these
    # explicit names; it renders the without-Farmer's-Practice population.
    FigureSpec(
        "yield_distribution_ph_combined_nopt_rcm_without_farmers_practice",
        "agronomic",
        "Grain-yield distribution — ph_combined_nopt_rcm, NOPT arm without Farmer's Practice",
        source_name="ph_combined_nopt_rcm",
    ),
    FigureSpec(
        "yield_distribution_ph_combined_nopt_rcm_with_farmers_practice",
        "agronomic",
        "Grain-yield distribution — ph_combined_nopt_rcm, Farmer's Practice arm",
        source_name="ph_combined_nopt_rcm",
    ),
    FigureSpec(
        "yield_distribution_ltcce",
        "agronomic",
        "Grain-yield distribution — ltcce",
        source_name="ltcce",
    ),
    FigureSpec("yield_versus_nitrogen", "agronomic", "Observed grain yield against N rate"),
    FigureSpec(
        "yield_versus_nitrogen_trajectories",
        "agronomic",
        "Observed grain yield against N rate, joined within each N-rate series",
    ),
    FigureSpec("temporal_coverage", "agronomic", "Observations by recorded year"),
    FigureSpec("context_composition", "agronomic", "Season and water-regime composition"),
    # The remaining declared context fields, split off rather than added as more
    # bars to the figure above: a stacked share bar can only label a segment it
    # can fit text into, and variety is recorded at a cardinality (69 and 90
    # levels) where no level reaches that width. Site, region, and establishment
    # keep the stacked form; variety gets a per-dataset top-N form.
    FigureSpec(
        "context_composition_site_and_management",
        "agronomic",
        "Region, site, and crop-establishment composition",
    ),
    FigureSpec(
        "context_composition_variety",
        "agronomic",
        "Most frequently recorded varieties per dataset",
    ),
    # The same composition read the other way: one panel per dataset carrying
    # all of that dataset's context fields, rather than one field across
    # datasets. Declared per source for the reason the yield panels are — a
    # core_trial_data-only figure carries no restricted level and is classified
    # on that source alone.
    FigureSpec(
        "context_composition_core_trial_data",
        "agronomic",
        "Context composition — Literature Extracted Datasets",
        source_name="core_trial_data",
        output_directory_name="literature_extracted_datasets",
        output_filename_stem="context_composition_literature_extracted_datasets",
    ),
    FigureSpec(
        "context_composition_ph_combined_nopt_rcm",
        "agronomic",
        "Context composition — ph_combined_nopt_rcm (legacy NOPT-only alias)",
        source_name="ph_combined_nopt_rcm",
    ),
    FigureSpec(
        "context_composition_ph_combined_nopt_rcm_without_farmers_practice",
        "agronomic",
        "Context composition — ph_combined_nopt_rcm, NOPT arm without Farmer's Practice",
        source_name="ph_combined_nopt_rcm",
    ),
    FigureSpec(
        "context_composition_ph_combined_nopt_rcm_with_farmers_practice",
        "agronomic",
        "Context composition — ph_combined_nopt_rcm, Farmer's Practice arm",
        source_name="ph_combined_nopt_rcm",
    ),
    FigureSpec(
        "context_composition_ltcce",
        "agronomic",
        "Context composition — ltcce",
        source_name="ltcce",
    ),
    FigureSpec("source_comparability", "crosscut", "Datasets compared on common axes"),
    FigureSpec("nitrogen_ladder_geometry", "crosscut", "N-ladder geometry across datasets"),
)


TABLE_SPECS_BY_NAME: Mapping[str, TableSpec] = {spec.name: spec for spec in TABLE_SPECS}
FIGURE_SPECS_BY_NAME: Mapping[str, FigureSpec] = {spec.name: spec for spec in FIGURE_SPECS}


def _require_unique(specs: Sequence[object], attribute: str, label: str) -> None:
    seen: set[str] = set()
    for spec in specs:
        value = getattr(spec, attribute)
        if value in seen:
            raise ProfileContractError(f"Duplicate {label} declared: {value!r}")
        seen.add(value)


_require_unique(TABLE_SPECS, "name", "table name")
_require_unique(FIGURE_SPECS, "name", "figure name")

for _spec in TABLE_SPECS:
    if _spec.group not in TABLE_GROUPS:
        raise ProfileContractError(f"Table {_spec.name!r} declares unknown group {_spec.group!r}")
    if len(set(_spec.columns)) != len(_spec.columns):
        raise ProfileContractError(f"Table {_spec.name!r} declares a duplicate column")
for _spec in FIGURE_SPECS:
    if _spec.group not in FIGURE_GROUPS:
        raise ProfileContractError(f"Figure {_spec.name!r} declares unknown group {_spec.group!r}")
    if _spec.output_directory_name is not None:
        directory = Path(_spec.output_directory_name)
        if _spec.source_name is None:
            raise ProfileContractError(
                f"Figure {_spec.name!r} declares an output directory without a source"
            )
        if directory.is_absolute() or len(directory.parts) != 1 or directory.name in {"", ".", ".."}:
            raise ProfileContractError(
                f"Figure {_spec.name!r} declares an unsafe output directory: "
                f"{_spec.output_directory_name!r}"
            )
    if _spec.output_filename_stem is not None:
        filename = Path(_spec.output_filename_stem)
        if filename.is_absolute() or len(filename.parts) != 1 or filename.name in {"", ".", ".."}:
            raise ProfileContractError(
                f"Figure {_spec.name!r} declares an unsafe output filename stem: "
                f"{_spec.output_filename_stem!r}"
            )


def table_spec(name: str) -> TableSpec:
    try:
        return TABLE_SPECS_BY_NAME[name]
    except KeyError as exc:
        raise ProfileContractError(f"Table is not declared in the contract: {name!r}") from exc


def figure_spec(name: str) -> FigureSpec:
    try:
        return FIGURE_SPECS_BY_NAME[name]
    except KeyError as exc:
        raise ProfileContractError(f"Figure is not declared in the contract: {name!r}") from exc


def table_relative_path(name: str) -> str:
    return table_spec(name).relative_path


def figure_relative_path(name: str, extension: str) -> str:
    return figure_spec(name).relative_path(extension)


def conform_table(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` reindexed to the declared column order, failing on drift.

    Missing declared columns are an error rather than a silent fill: a profile
    module that cannot compute a declared statistic must emit it as NA, so the
    reader can tell "not applicable" from "the writer forgot".
    """

    spec = table_spec(name)
    if not isinstance(frame, pd.DataFrame):
        raise ProfileContractError(f"Table {name!r} is not a DataFrame")
    missing = [column for column in spec.columns if column not in frame.columns]
    if missing:
        raise ProfileContractError(
            f"Table {name!r} is missing declared columns: {missing}"
        )
    extra = [column for column in frame.columns if column not in spec.columns]
    if extra:
        raise ProfileContractError(
            f"Table {name!r} carries undeclared columns: {extra}"
        )
    return frame.loc[:, list(spec.columns)].reset_index(drop=True)


def empty_table(name: str) -> pd.DataFrame:
    """An empty frame carrying exactly the declared schema."""

    return pd.DataFrame(columns=list(table_spec(name).columns))


@dataclass(frozen=True)
class ProfileArtifact:
    """One artifact that landed on disk, with the classification it inherits."""

    relative_path: str
    kind: str
    group: str
    name: str
    sha256: str
    byte_size: int
    row_count: int | None
    data_classification: str


@dataclass(frozen=True)
class VerifiedProfileBundle:
    root: Path
    schema_version: str
    artifact_count: int
    table_count: int
    figure_count: int
    restricted_artifact_count: int
