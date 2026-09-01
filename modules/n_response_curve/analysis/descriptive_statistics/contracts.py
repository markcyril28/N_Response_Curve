"""Frozen artifact registry and shared row types for the descriptive profile.

Every table and figure the recipe may emit is declared here exactly once, with
its group, its ordered column schema, and the source classification it inherits.
The analysis modules build frames against these schemas, the reporting module
writes them, and ``verify_profile_bundle`` reconciles what landed on disk with
what is declared here. Nothing else may invent an artifact path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import pandas as pd


SCHEMA_VERSION = "dataset-descriptive-statistics-v2"
BUNDLE_STATUS = "diagnostic_internal_not_release"
MANIFEST_NAME = "run_manifest.json"
CHECKSUMS_NAME = "CHECKSUMS.sha256"
SUMMARY_NAME = "summary.md"

# The recipe emits the agronomic profile only. The structure, numeric,
# categorical, and cross-cut groups were retired with their analysis modules:
# the recipe no longer profiles physical column shape, generic column
# statistics, or the datasets side by side.
TABLE_GROUPS = ("agronomic",)
FIGURE_GROUPS = ("agronomic",)

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
)


# Filename markers for a panel that varies a declared figure by which
# farmer's-practice evidence it draws. Owned here, beside the declared paths
# they extend, so a published name and the variants rendered beside it cannot
# drift apart: rename a figure through ``output_filename_stem`` and every
# variant filename derived from it follows.
WITH_FARMERS_PRACTICE_SUFFIX = "_with_farmers_practice"
EXCLUDING_FARMERS_PRACTICE_SUFFIX = "_excluding_farmers_practice"


FIGURE_SPECS: tuple[FigureSpec, ...] = (
    # The published filename carries an operator-chosen ``_full`` qualifier that
    # distinguishes this all-evidence panel from the Farmer's-Practice variants
    # rendered beside it; the registry name stays the contract key the builder
    # map, the manifest, and the presentation derivatives are written against.
    FigureSpec(
        "nitrogen_rate_distribution",
        "agronomic",
        "Inorganic N-rate distribution per dataset",
        output_filename_stem="nitrogen_rate_distribution_full",
    ),
    FigureSpec(
        "nitrogen_rate_distribution_with_separate_farmers_practice",
        "agronomic",
        "Inorganic N-rate distribution with separate Farmer's Practice bars",
    ),
    # Draws the declared farmer's-practice arm as a fourth series, so the
    # published filename says so. The registry name stays the contract key the
    # builder map, the manifest, and the presentation derivatives are written
    # against; only the filename carries the qualifier.
    FigureSpec(
        "yield_distribution",
        "agronomic",
        "Grain-yield distribution per dataset and Farmer's Practice",
        output_filename_stem=f"yield_distribution{WITH_FARMERS_PRACTICE_SUFFIX}",
    ),
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
    # Farmer's Practice is stacked as a fourth series here too; same rule as
    # ``yield_distribution`` above — the filename qualifies, the key does not.
    FigureSpec(
        "temporal_coverage",
        "agronomic",
        "Observations by recorded year, Farmer's Practice included",
        output_filename_stem=f"temporal_coverage{WITH_FARMERS_PRACTICE_SUFFIX}",
    ),
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


def variant_figure_relative_path(name: str, suffix: str, extension: str) -> str:
    """The declared path of *name*, carrying *suffix* as its population marker.

    Standalone variants render a declared figure over a different population
    and publish beside it. Deriving their path from the declared one keeps the
    pair named consistently through any rename of the figure they vary, which a
    literal repeated in each script would not.

    The two farmer's-practice markers are mutually exclusive descriptions of
    one population, so an existing marker is replaced rather than appended to:
    a panel published as ``..._with_farmers_practice`` yields
    ``..._excluding_farmers_practice`` here, never a filename claiming both.
    Any other qualifier in the stem — an operator rename, say — is preserved.
    """

    relative = PurePosixPath(figure_relative_path(name, extension))
    stem = relative.stem
    for marker in (
        WITH_FARMERS_PRACTICE_SUFFIX,
        EXCLUDING_FARMERS_PRACTICE_SUFFIX,
    ):
        if stem.endswith(marker):
            stem = stem[: -len(marker)]
            break
    return str(relative.with_name(f"{stem}{suffix}{relative.suffix}"))


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
