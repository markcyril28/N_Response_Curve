"""Position-safe loading of every profiled dataset.

The core literature extract carries 302 physical columns with seven duplicated
and sixteen blank header names, so every column here is keyed by its 1-based
physical position (``raw_col_NNN``) and never by header text. Reading goes
through ``data.ingest.ingest_configured_sources``, which verifies the intake
manifest, enforces the registered per-source encoding and shape adapter, and
rejects a source whose bytes change mid-read.

Privacy suppression happens at this layer, before any profile sees a value:
suppressed columns keep their structural row in the inventory (so column totals
stay reconcilable) but their cells are replaced with empty strings, which makes
it impossible for a downstream module to leak them into a level table.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ...data.config import load_config
from ...data.ingest import IngestedSource, ingest_configured_sources
from ...data.provenance import sha256_file


class SourceProfileError(ValueError):
    """Raised when a profiled source does not match its registered contract."""


# A cell is numeric only if it is a plain decimal or scientific-notation number.
# Deliberately strict: "not stated", "<0.1", "0.05 N HCl extraction" and
# "1,234" are reported as unparsed rather than coerced, because silently
# coercing them would understate how much of each dataset is free text.
_NUMERIC_PATTERN = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")

BLANK_HEADER_LABEL = "(blank header)"


def parse_numeric(text: str) -> float:
    """Return the finite float a cell encodes, or NaN when it encodes none."""

    stripped = text.strip()
    if not stripped or not _NUMERIC_PATTERN.match(stripped):
        return math.nan
    try:
        value = float(stripped)
    except ValueError:  # pragma: no cover - guarded by the pattern
        return math.nan
    return value if math.isfinite(value) else math.nan


@dataclass(frozen=True)
class ColumnSpec:
    """One physical column, classified for profiling."""

    source_name: str
    data_classification: str
    position: int
    raw_column_id: str
    header_raw: str
    header_label: str
    header_status: str  # named | blank | duplicated
    duplicate_group_size: int
    duplicate_positions: tuple[int, ...]
    suppressed: bool
    suppression_reason: str
    value_kind: str  # numeric | categorical | identifier | empty | suppressed
    nonblank_count: int
    blank_count: int
    distinct_nonblank_count: int
    numeric_parse_rate: float
    example_values: tuple[str, ...]

    @property
    def fill_rate(self) -> float:
        total = self.nonblank_count + self.blank_count
        return self.nonblank_count / total if total else 0.0


@dataclass(frozen=True)
class ColumnBinding:
    """A resolved [agronomic.*] binding: a label pointing at a physical column."""

    label: str
    position: int
    header: str
    raw_column_id: str


@dataclass(frozen=True)
class AgronomicBinding:
    """Every declared agronomic column for one source, already position-verified."""

    source_name: str
    nitrogen_rate: ColumnBinding
    yield_t_ha: ColumnBinding
    yield_kg_ha: ColumnBinding | None
    zero_n_yield_t_ha: ColumnBinding | None
    # A second recorded N treatment carried in sibling columns rather than in
    # rows, exactly as ``zero_n_yield_t_ha`` is. Bound so a figure that profiles
    # recorded rates can say what the record actually holds; the harmonized
    # observation frame still expands neither arm into a row.
    farmers_practice_n_rate: ColumnBinding | None
    farmers_practice_yield_t_ha: ColumnBinding | None
    year: ColumnBinding | None
    context: tuple[ColumnBinding, ...]
    grouping: tuple[ColumnBinding, ...]
    # Columns whose combination identifies one N ladder (rows differing only in
    # N rate). Distinct from ``grouping``, which is the study/trial provenance
    # pair: LTCCE's ladder is nested in year-season-variety, not in its design
    # label, so the two lists deliberately differ for that source.
    series: tuple[ColumnBinding, ...]
    # Evidence columns used to derive one conservative, common straw-management
    # vocabulary. Kept separate from ``context`` because no one physical source
    # column carries the standardized value in every dataset.
    straw_management: tuple[ColumnBinding, ...] = ()
    # A source field whose semantics distinguish direct seeding from a recorded
    # transplanting date. Kept separate so raw calendar dates never become
    # categorical context levels.
    crop_establishment: ColumnBinding | None = None


@dataclass(frozen=True)
class ProfiledSource:
    """One dataset, read by physical position and ready to profile.

    ``text`` and ``numeric`` are aligned frames indexed identically: one row per
    logical nonblank data row, one column per physical position keyed by
    ``raw_column_id``. ``text`` holds the exact stored strings (whitespace
    preserved); ``numeric`` holds the parsed float or NaN.
    """

    source_name: str
    source_path: Path
    source_relative_path: str
    source_sha256: str
    data_classification: str
    restricted_access_status: str
    source_encoding: str
    source_type: str
    source_family: str
    country_code: str
    shape_adapter_version: str
    representation_basis: str
    representation_basis_status: str
    physical_column_count: int
    data_row_count: int
    blank_row_count: int
    columns: tuple[ColumnSpec, ...]
    text: pd.DataFrame
    numeric: pd.DataFrame
    source_row_numbers: tuple[int, ...]
    binding: AgronomicBinding

    @property
    def is_restricted(self) -> bool:
        return self.data_classification == "restricted"

    def column(self, raw_column_id: str) -> ColumnSpec:
        for spec in self.columns:
            if spec.raw_column_id == raw_column_id:
                return spec
        raise SourceProfileError(
            f"{self.source_name}: no column {raw_column_id!r}"
        )

    def columns_of_kind(self, *kinds: str) -> tuple[ColumnSpec, ...]:
        wanted = frozenset(kinds)
        return tuple(spec for spec in self.columns if spec.value_kind in wanted)

    def numeric_series(self, binding: ColumnBinding) -> pd.Series:
        return self.numeric[binding.raw_column_id]

    def text_series(self, binding: ColumnBinding) -> pd.Series:
        return self.text[binding.raw_column_id]


@dataclass(frozen=True)
class LoadedSources:
    """Every profiled source plus the base-config evidence used to admit them."""

    sources: tuple[ProfiledSource, ...]
    base_config_path: Path
    base_config_sha256: str
    source_manifest_sha256: str | None

    def by_name(self, source_name: str) -> ProfiledSource:
        for source in self.sources:
            if source.source_name == source_name:
                return source
        raise SourceProfileError(f"Source was not profiled: {source_name!r}")

    @property
    def any_restricted(self) -> bool:
        return any(source.is_restricted for source in self.sources)


def _header_status(
    headers: Sequence[str],
) -> tuple[dict[int, str], dict[int, tuple[int, ...]]]:
    """Classify each 1-based header position as named, blank, or duplicated."""

    groups: dict[str, list[int]] = defaultdict(list)
    for position, header in enumerate(headers, start=1):
        groups[header].append(position)
    status: dict[int, str] = {}
    members: dict[int, tuple[int, ...]] = {}
    for header, positions in groups.items():
        for position in positions:
            if header.strip() == "":
                status[position] = "blank"
            elif len(positions) > 1:
                status[position] = "duplicated"
            else:
                status[position] = "named"
            members[position] = tuple(positions)
    return status, members


def _classify_value_kind(
    *,
    suppressed: bool,
    nonblank_count: int,
    numeric_parse_rate: float,
    distinct_nonblank_count: int,
    numeric_parse_threshold: float,
    maximum_categorical_cardinality: int,
) -> str:
    if suppressed:
        return "suppressed"
    if nonblank_count == 0:
        return "empty"
    if numeric_parse_rate >= numeric_parse_threshold:
        return "numeric"
    if distinct_nonblank_count > maximum_categorical_cardinality:
        return "identifier"
    return "categorical"


def _build_column_specs(
    ingested: IngestedSource,
    cells: list[list[str]],
    *,
    suppressed_headers: frozenset[str],
    numeric_parse_threshold: float,
    maximum_categorical_cardinality: int,
    example_values_per_column: int,
) -> tuple[tuple[ColumnSpec, ...], list[list[str]], np.ndarray]:
    headers = [column.header for column in ingested.columns]
    status, members = _header_status(headers)
    row_count = len(cells)
    specs: list[ColumnSpec] = []
    numeric_matrix = np.full((row_count, len(headers)), math.nan, dtype=float)

    for index, column in enumerate(ingested.columns):
        header = column.header
        suppressed = header in suppressed_headers
        raw_values = [row[index] for row in cells]
        if suppressed:
            # Overwrite in place so no downstream consumer can reach the values.
            for row in cells:
                row[index] = ""
            raw_values = ["" for _ in raw_values]

        nonblank: list[str] = []
        parsed_count = 0
        for row_index, value in enumerate(raw_values):
            if value.strip() == "":
                continue
            nonblank.append(value)
            parsed = parse_numeric(value)
            if not math.isnan(parsed):
                numeric_matrix[row_index, index] = parsed
                parsed_count += 1

        nonblank_count = len(nonblank)
        blank_count = row_count - nonblank_count
        numeric_parse_rate = parsed_count / nonblank_count if nonblank_count else 0.0
        distinct = Counter(nonblank)
        distinct_nonblank_count = len(distinct)
        seen: list[str] = []
        for value in nonblank:
            if value not in seen:
                seen.append(value)
            if len(seen) >= example_values_per_column:
                break

        value_kind = _classify_value_kind(
            suppressed=suppressed,
            nonblank_count=nonblank_count,
            numeric_parse_rate=numeric_parse_rate,
            distinct_nonblank_count=distinct_nonblank_count,
            numeric_parse_threshold=numeric_parse_threshold,
            maximum_categorical_cardinality=maximum_categorical_cardinality,
        )
        specs.append(
            ColumnSpec(
                source_name=ingested.source_name,
                data_classification=ingested.data_classification,
                position=column.position,
                raw_column_id=column.raw_column_id,
                header_raw=header,
                header_label=header if header.strip() else BLANK_HEADER_LABEL,
                header_status=status[column.position],
                duplicate_group_size=len(members[column.position]),
                duplicate_positions=members[column.position],
                suppressed=suppressed,
                suppression_reason=(
                    "restricted_identifier_suppressed_by_policy" if suppressed else ""
                ),
                value_kind=value_kind,
                nonblank_count=nonblank_count,
                blank_count=blank_count,
                distinct_nonblank_count=distinct_nonblank_count,
                numeric_parse_rate=numeric_parse_rate,
                example_values=tuple(seen),
            )
        )
    return tuple(specs), cells, numeric_matrix


def _resolve_binding(
    ingested: IngestedSource,
    raw: Mapping[str, Any],
    key: str,
    *,
    required: bool,
) -> ColumnBinding | None:
    entry = raw.get(key)
    if entry is None:
        if required:
            raise SourceProfileError(
                f"{ingested.source_name}: [agronomic.{ingested.source_name}].{key} is required"
            )
        return None
    return _verify_binding(ingested, entry, label=key)


def _verify_binding(
    ingested: IngestedSource,
    entry: Mapping[str, Any],
    *,
    label: str,
) -> ColumnBinding:
    position = int(entry["position"])
    header = str(entry["header"])
    resolved_label = str(entry.get("label", label))
    if position < 1 or position > len(ingested.columns):
        raise SourceProfileError(
            f"{ingested.source_name}: binding {resolved_label!r} position {position} "
            f"is outside the {len(ingested.columns)}-column shape"
        )
    column = ingested.columns[position - 1]
    if column.header != header:
        raise SourceProfileError(
            f"{ingested.source_name}: binding {resolved_label!r} expects header "
            f"{header!r} at position {position}, found {column.header!r}"
        )
    return ColumnBinding(
        label=resolved_label,
        position=position,
        header=header,
        raw_column_id=column.raw_column_id,
    )


def _resolve_agronomic_binding(
    ingested: IngestedSource,
    raw: Mapping[str, Any],
) -> AgronomicBinding:
    nitrogen_rate = _resolve_binding(ingested, raw, "nitrogen_rate", required=True)
    yield_t_ha = _resolve_binding(ingested, raw, "yield_t_ha", required=True)
    assert nitrogen_rate is not None and yield_t_ha is not None
    return AgronomicBinding(
        source_name=ingested.source_name,
        nitrogen_rate=nitrogen_rate,
        yield_t_ha=yield_t_ha,
        yield_kg_ha=_resolve_binding(ingested, raw, "yield_kg_ha", required=False),
        zero_n_yield_t_ha=_resolve_binding(
            ingested, raw, "zero_n_yield_t_ha", required=False
        ),
        farmers_practice_n_rate=_resolve_binding(
            ingested, raw, "farmers_practice_n_rate", required=False
        ),
        farmers_practice_yield_t_ha=_resolve_binding(
            ingested, raw, "farmers_practice_yield_t_ha", required=False
        ),
        year=_resolve_binding(ingested, raw, "year", required=False),
        context=tuple(
            _verify_binding(ingested, entry, label=f"context_{index}")
            for index, entry in enumerate(raw.get("context", ()))
        ),
        grouping=tuple(
            _verify_binding(ingested, entry, label=f"grouping_{index}")
            for index, entry in enumerate(raw.get("grouping", ()))
        ),
        series=tuple(
            _verify_binding(ingested, entry, label=f"series_{index}")
            for index, entry in enumerate(raw.get("series", ()))
        ),
        straw_management=tuple(
            _verify_binding(ingested, entry, label=f"straw_management_{index}")
            for index, entry in enumerate(raw.get("straw_management", ()))
        ),
        crop_establishment=_resolve_binding(
            ingested, raw, "crop_establishment", required=False
        ),
    )


def _profile_one(
    ingested: IngestedSource,
    *,
    project_root: Path,
    agronomic_raw: Mapping[str, Any],
    suppressed_headers: frozenset[str],
    numeric_parse_threshold: float,
    maximum_categorical_cardinality: int,
    example_values_per_column: int,
) -> ProfiledSource:
    cells = [list(row.raw_cells) for row in ingested.rows]
    specs, cells, numeric_matrix = _build_column_specs(
        ingested,
        cells,
        suppressed_headers=suppressed_headers,
        numeric_parse_threshold=numeric_parse_threshold,
        maximum_categorical_cardinality=maximum_categorical_cardinality,
        example_values_per_column=example_values_per_column,
    )
    column_ids = [column.raw_column_id for column in ingested.columns]
    text = pd.DataFrame(cells, columns=column_ids, dtype=object)
    if text.empty:
        text = pd.DataFrame({column_id: pd.Series(dtype=object) for column_id in column_ids})
    numeric = pd.DataFrame(numeric_matrix, columns=column_ids, dtype=float)
    binding = _resolve_agronomic_binding(ingested, agronomic_raw)
    try:
        relative = ingested.source_path.relative_to(project_root).as_posix()
    except ValueError:
        relative = ingested.source_path.as_posix()
    return ProfiledSource(
        source_name=ingested.source_name,
        source_path=ingested.source_path,
        source_relative_path=relative,
        source_sha256=ingested.source_sha256,
        data_classification=ingested.data_classification,
        restricted_access_status=ingested.restricted_access_status,
        source_encoding=ingested.source_encoding,
        source_type=ingested.source_type,
        source_family=ingested.source_family,
        country_code=ingested.source_country_code,
        shape_adapter_version=ingested.shape_adapter_version,
        representation_basis=ingested.representation_basis,
        representation_basis_status=ingested.representation_basis_status,
        physical_column_count=len(ingested.columns),
        data_row_count=len(ingested.rows),
        blank_row_count=len(ingested.blank_rows),
        columns=specs,
        text=text,
        numeric=numeric,
        source_row_numbers=tuple(row.source_row_number for row in ingested.rows),
        binding=binding,
    )


NATIVE_T_HA = "native_t_ha"
CONVERTED_FROM_KG_HA = "converted_from_kg_ha"

OBSERVATION_COLUMNS = (
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
)


def _labeled(bindings: Sequence[ColumnBinding], label: str) -> ColumnBinding | None:
    for binding in bindings:
        if binding.label == label:
            return binding
    return None


def _key_series(
    source: ProfiledSource, bindings: Sequence[ColumnBinding]
) -> pd.Series:
    """Join the text of several columns into one stable composite key."""

    if not bindings:
        return pd.Series([""] * source.data_row_count, index=source.text.index, dtype=object)
    parts = [
        source.text[binding.raw_column_id].astype(str).str.strip()
        for binding in bindings
    ]
    joined = parts[0]
    for part in parts[1:]:
        joined = joined.str.cat(part, sep="|")
    return joined


def build_observation_frame(
    source: ProfiledSource,
    *,
    zero_n_tolerance_kg_ha: float,
) -> pd.DataFrame:
    """Harmonize one dataset onto the common (N kg/ha, yield t/ha) observation basis.

    Exactly one output row per retained input row: the recorded observation. A
    paired zero-N arm held in its own column — as ph_combined_nopt_rcm holds
    ``n0_yield`` — is deliberately NOT expanded into a second row here, because
    doing so would assert a within-record response structure that the recipe is
    not entitled to assume. Those arms are profiled separately by the zero-N
    check table. Rows lacking a finite N rate or a finite yield are dropped and
    the drop is reported by the unit-lineage audit.
    """

    binding = source.binding
    n_rate = source.numeric_series(binding.nitrogen_rate).astype(float)
    native_yield = source.numeric_series(binding.yield_t_ha).astype(float)

    yield_t_ha = native_yield.copy()
    lineage = pd.Series(
        np.where(native_yield.notna(), NATIVE_T_HA, ""),
        index=native_yield.index,
        dtype=object,
    )
    if binding.yield_kg_ha is not None:
        kg = source.numeric_series(binding.yield_kg_ha).astype(float)
        fillable = native_yield.isna() & kg.notna()
        yield_t_ha = yield_t_ha.where(~fillable, kg / 1000.0)
        lineage = lineage.where(~fillable, CONVERTED_FROM_KG_HA)

    if binding.year is not None:
        year = source.numeric_series(binding.year).astype(float)
    else:
        year = pd.Series(np.nan, index=native_yield.index, dtype=float)

    grouping = binding.grouping
    study = _labeled(grouping, "study")
    trial = _labeled(grouping, "trial")
    # A series key is only usable when every component that composes it carries
    # a value. Blank components are never forward-filled; the row keeps its key
    # but is flagged so ladder geometry can exclude it rather than bucket
    # unrelated rows together under a partially blank key.
    if binding.series:
        resolved = pd.Series(True, index=native_yield.index)
        for component in binding.series:
            values = source.text[component.raw_column_id].astype(str).str.strip()
            resolved &= values != ""
    else:
        resolved = pd.Series(False, index=native_yield.index)
    frame = pd.DataFrame(
        {
            "source_name": source.source_name,
            "data_classification": source.data_classification,
            "source_row_number": list(source.source_row_numbers),
            "study_key": (
                _key_series(source, [study]) if study is not None else ""
            ),
            "trial_key": (
                _key_series(source, [trial]) if trial is not None else ""
            ),
            "series_key": source.source_name
            + "::"
            + _key_series(source, binding.series),
            "year": year,
            "n_rate_kg_ha": n_rate,
            "yield_t_ha": yield_t_ha,
            "yield_unit_lineage": lineage,
            "is_zero_n": n_rate.abs() <= zero_n_tolerance_kg_ha,
            "is_series_resolved": resolved,
        },
        index=native_yield.index,
    )
    retained = frame.loc[frame["n_rate_kg_ha"].notna() & frame["yield_t_ha"].notna()]
    return retained.loc[:, list(OBSERVATION_COLUMNS)].reset_index(drop=True)


def build_all_observations(
    loaded: LoadedSources,
    *,
    zero_n_tolerance_kg_ha: float,
) -> pd.DataFrame:
    """The harmonized observation frame for every profiled dataset, stacked."""

    frames = [
        build_observation_frame(source, zero_n_tolerance_kg_ha=zero_n_tolerance_kg_ha)
        for source in loaded.sources
    ]
    if not frames:
        return pd.DataFrame(columns=list(OBSERVATION_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def load_profiled_sources(
    config: Any,
    *,
    base_config_loader: Callable[..., Any] = load_config,
) -> LoadedSources:
    """Ingest and profile every source named in ``[sources].profiled``."""

    base_config = base_config_loader(
        config.base_config_path,
        project_root=config.project_root,
        check_files=True,
        preflight_engines=False,
    )
    missing = [
        name for name in config.profiled_sources if name not in base_config.enabled_sources
    ]
    if missing:
        raise SourceProfileError(
            "Profiled source is not enabled in the base configuration: "
            + ", ".join(sorted(missing))
        )

    ingestion = ingest_configured_sources(base_config)
    ingested_by_name = {source.source_name: source for source in ingestion.sources}

    profiled: list[ProfiledSource] = []
    for source_name in config.profiled_sources:
        ingested = ingested_by_name.get(source_name)
        if ingested is None:
            raise SourceProfileError(
                f"Profiled source was not returned by ingestion: {source_name!r}"
            )
        expected = config.expected_source_sha256.get(source_name)
        if expected is not None and ingested.source_sha256 != expected:
            raise SourceProfileError(
                f"{source_name}: source SHA-256 {ingested.source_sha256} does not match "
                f"the pinned [expected_inputs] value {expected}"
            )
        agronomic_raw = config.agronomic_bindings.get(source_name)
        if agronomic_raw is None:
            raise SourceProfileError(
                f"No [agronomic.{source_name}] binding table is declared"
            )
        profiled.append(
            _profile_one(
                ingested,
                project_root=config.project_root,
                agronomic_raw=agronomic_raw,
                suppressed_headers=config.suppressed_headers,
                numeric_parse_threshold=config.numeric_parse_threshold,
                maximum_categorical_cardinality=config.maximum_categorical_cardinality,
                example_values_per_column=config.example_values_per_column,
            )
        )

    integrity = ingestion.integrity_report
    manifest_sha256 = getattr(integrity, "manifest_sha256", None) if integrity else None
    return LoadedSources(
        sources=tuple(profiled),
        base_config_path=config.base_config_path,
        base_config_sha256=sha256_file(config.base_config_path),
        source_manifest_sha256=manifest_sha256,
    )
