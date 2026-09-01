"""Bundle writer and verifier for the descriptive-statistics profile.

Everything that lands on disk is declared in
``analysis.descriptive_statistics.contracts`` and bound by ``CHECKSUMS.sha256``,
with ``run_manifest.json`` recording the provenance of every input that produced
it. Unlike the release package, this bundle allows no unchecksummed member at
all: it is a standalone internal diagnostic, so there is no promoted-package
extension case to accommodate.

Every artifact carries the data classification it inherits from the datasets it
was derived from, because two of the three profiled datasets are restricted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import pandas as pd
from PIL import Image

from ..analysis.descriptive_statistics import contracts
from ..analysis.descriptive_statistics.config import DescriptiveStatisticsConfig
from ..analysis.descriptive_statistics.contracts import (
    BUNDLE_STATUS,
    CHECKSUMS_NAME,
    MANIFEST_NAME,
    SCHEMA_VERSION,
    SUMMARY_NAME,
    ProfileArtifact,
    VerifiedProfileBundle,
)
from ..analysis.descriptive_statistics.sources import LoadedSources
from ..data.provenance import sha256_file
from .descriptive_statistics_figures import build_figure


class ProfileBundleError(ValueError):
    """Raised when the bundle does not match its declared contract."""


RESTRICTED = "restricted"
INTERNAL = "internal"
LEGACY_SCHEMA_VERSION = "dataset-descriptive-statistics-v1"


@dataclass(frozen=True)
class WrittenProfileBundle:
    root: Path
    artifacts: tuple[ProfileArtifact, ...]
    skipped_figures: tuple[str, ...]

    @property
    def artifact_count(self) -> int:
        return len(self.artifacts)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def _frame_classification(frame: pd.DataFrame, *, fallback: str) -> str:
    """The strictest classification present in a frame's own rows.

    Reading it off the data rather than the table name means a table that
    happens to contain no restricted row is not labeled restricted, and one that
    does cannot escape the label by being in an innocuous-looking group.
    """

    if "data_classification" in frame.columns and not frame.empty:
        values = {str(value) for value in frame["data_classification"].dropna().unique()}
        return RESTRICTED if RESTRICTED in values else (INTERNAL if values else fallback)
    return fallback


def _write_table(
    root: Path,
    name: str,
    frame: pd.DataFrame,
    *,
    fallback_classification: str,
) -> ProfileArtifact:
    spec = contracts.table_spec(name)
    conformed = contracts.conform_table(name, frame)
    relative = spec.relative_path
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" keeps pandas from emitting CRLF on this Windows-mounted tree,
    # which would make the checksums platform-dependent.
    with path.open("w", encoding="utf-8", newline="") as handle:
        conformed.to_csv(handle, index=False, lineterminator="\n")
    return ProfileArtifact(
        relative_path=relative,
        kind="table",
        group=spec.group,
        name=name,
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
        row_count=int(len(conformed)),
        data_classification=_frame_classification(
            conformed, fallback=fallback_classification
        ),
    )


def _write_figure(
    root: Path,
    name: str,
    figure: Any,
    *,
    config: DescriptiveStatisticsConfig,
    classification: str,
) -> tuple[ProfileArtifact, ...]:
    spec = contracts.figure_spec(name)
    artifacts: list[ProfileArtifact] = []
    for extension in config.figure_formats:
        relative = spec.relative_path(extension)
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        save_format = "jpeg" if extension in {"jpeg", "jpg"} else extension
        figure.savefig(
            path, dpi=config.figure_dpi, format=save_format, facecolor="white"
        )
        with Image.open(path) as image:
            image.load()
            if image.size[0] < 100 or image.size[1] < 100:
                raise ProfileBundleError(
                    f"Figure {name!r} rendered degenerate at {image.size}"
                )
        artifacts.append(
            ProfileArtifact(
                relative_path=relative,
                kind="figure",
                group=spec.group,
                name=name,
                sha256=sha256_file(path),
                byte_size=path.stat().st_size,
                row_count=None,
                data_classification=classification,
            )
        )
    return tuple(artifacts)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (frozenset, set)):
        return sorted(str(item) for item in value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _project_relative(path: Path, root: Path) -> str:
    """Path relative to the project root, or absolute when it lies outside it.

    The recipe configuration is legitimately read from outside the tree — the
    documented way to probe a different mode is to copy the config to a
    scratchpad and point --config at it — so this must not raise the way a bare
    ``relative_to`` does.
    """

    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _source_manifest_block(loaded: LoadedSources) -> list[dict[str, Any]]:
    return [
        {
            "source_name": source.source_name,
            "data_classification": source.data_classification,
            "restricted_access_status": source.restricted_access_status,
            "source_relative_path": source.source_relative_path,
            "source_sha256": source.source_sha256,
            "source_encoding": source.source_encoding,
            "shape_adapter_version": source.shape_adapter_version,
            "representation_basis": source.representation_basis,
            "representation_basis_status": source.representation_basis_status,
            "source_type": source.source_type,
            "source_family": source.source_family,
            "country_code": source.country_code,
            "physical_column_count": source.physical_column_count,
            "data_row_count": source.data_row_count,
            "blank_row_count": source.blank_row_count,
            "suppressed_column_count": sum(
                1 for column in source.columns if column.suppressed
            ),
        }
        for source in loaded.sources
    ]


def _manifest_document(
    *,
    config: DescriptiveStatisticsConfig,
    loaded: LoadedSources,
    artifacts: Sequence[ProfileArtifact],
    skipped_figures: Sequence[str],
    implementation_sha256: Mapping[str, str],
    generated_at: str,
) -> dict[str, Any]:
    restricted = [a for a in artifacts if a.data_classification == RESTRICTED]
    return {
        "schema_version": SCHEMA_VERSION,
        "bundle_status": BUNDLE_STATUS,
        "generated_at_utc": generated_at,
        "recipe": {
            "config_relative_path": _project_relative(
                config.config_path, config.project_root
            ),
            "config_sha256": sha256_file(config.config_path),
            "mode": config.mode,
            "random_seed": config.random_seed,
            "include_row_level_observations": config.include_row_level_observations,
            "profiled_sources": list(config.profiled_sources),
            "suppressed_headers": sorted(config.suppressed_headers),
        },
        # Recorded, deliberately NOT enforced: scriptCONFIG.toml is edited in the
        # ordinary course of operating this repository, so pinning its hash would
        # hard-fail the recipe on an unrelated edit. The audit trail is preserved
        # by recording what was actually read.
        "base_configuration": {
            "relative_path": _project_relative(
                config.base_config_path, config.project_root
            ),
            "observed_sha256": loaded.base_config_sha256,
            "binding": "recorded_not_enforced",
        },
        # Only the settings that still govern an emitted artifact. The recipe
        # profiles the agronomic view alone, so the retired structure, numeric,
        # categorical, and cross-cut tuning is not recorded here as though it
        # shaped a table in this bundle.
        "thresholds": {
            "numeric_parse_threshold": config.numeric_parse_threshold,
            "maximum_categorical_cardinality": config.maximum_categorical_cardinality,
            "example_values_per_column": config.example_values_per_column,
            "histogram_bins": config.histogram_bins,
            "minimum_level_count": config.minimum_level_count,
            "nitrogen_bin_width_kg_ha": config.nitrogen_bin_width_kg_ha,
            "zero_n_tolerance_kg_ha": config.zero_n_tolerance_kg_ha,
            "n_level_tolerance_kg_ha": config.n_level_tolerance_kg_ha,
            "minimum_group_observations": config.minimum_group_observations,
        },
        "outputs": {
            "table_formats": list(config.table_formats),
            "figure_formats": list(config.figure_formats),
            "figure_dpi": config.figure_dpi,
            "figure_width_inches": config.figure_width_inches,
            "figure_height_inches": config.figure_height_inches,
        },
        "sources": _source_manifest_block(loaded),
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "documents": {
            "summary": SUMMARY_NAME,
            "manifest": MANIFEST_NAME,
            "checksums": CHECKSUMS_NAME,
        },
        "artifacts": [
            {
                "relative_path": artifact.relative_path,
                "artifact_kind": artifact.kind,
                "group": artifact.group,
                "name": artifact.name,
                "sha256": artifact.sha256,
                "bytes": artifact.byte_size,
                "row_count": artifact.row_count,
                "data_classification": artifact.data_classification,
            }
            for artifact in artifacts
        ],
        "skipped_figures": list(skipped_figures),
        "counts": {
            "artifacts": len(artifacts),
            "tables": sum(1 for a in artifacts if a.kind == "table"),
            "figures": sum(1 for a in artifacts if a.kind == "figure"),
            "restricted_artifacts": len(restricted),
        },
        "governance": {
            "release_status": "not_a_release_artifact",
            "package_relationship": "standalone_sibling_of_release_package",
            "restricted_sources": sorted(
                source.source_name
                for source in loaded.sources
                if source.data_classification == RESTRICTED
            ),
            "restricted_handling": (
                "Personal and location identifier columns are suppressed before "
                "profiling; categorical levels below the configured minimum count "
                "are withheld and only their withheld count is reported."
            ),
            "analysis_scope": "descriptive_only_no_inference_no_recommendation",
        },
    }


def _write_checksums(root: Path, artifacts: Sequence[ProfileArtifact]) -> None:
    """Bind every file in the bundle except the checksum file itself."""

    entries: list[tuple[str, str]] = [
        (artifact.relative_path, artifact.sha256) for artifact in artifacts
    ]
    for name in (SUMMARY_NAME, MANIFEST_NAME):
        entries.append((name, sha256_file(root / name)))
    lines = [
        f"{digest}  {relative}\n" for relative, digest in sorted(set(entries))
    ]
    with (root / CHECKSUMS_NAME).open("w", encoding="utf-8", newline="") as handle:
        handle.writelines(lines)


def write_profile_bundle(
    root: str | Path,
    *,
    config: DescriptiveStatisticsConfig,
    loaded: LoadedSources,
    tables: Mapping[str, pd.DataFrame],
    implementation_sha256: Mapping[str, str],
) -> WrittenProfileBundle:
    """Write every declared table, figure, the summary, manifest, and checksums."""

    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    fallback = (
        RESTRICTED
        if any(source.data_classification == RESTRICTED for source in loaded.sources)
        else INTERNAL
    )

    artifacts: list[ProfileArtifact] = []
    for spec in contracts.TABLE_SPECS:
        frame = tables.get(spec.name)
        if frame is None:
            if not spec.optional:
                raise ProfileBundleError(
                    f"Declared table was not produced: {spec.name!r}"
                )
            frame = contracts.empty_table(spec.name)
        artifacts.append(
            _write_table(
                target, spec.name, frame, fallback_classification=fallback
            )
        )

    # Deferred so an import of this module does not pull in pyplot.
    import matplotlib.pyplot as plt

    # A panel that draws one source carries only that source's data, so it is
    # classified on that source alone. Inheriting the restricted-if-any fallback
    # would mark a core_trial_data-only figure restricted and overstate what the
    # bundle withholds.
    by_source = {source.source_name: source for source in loaded.sources}

    skipped: list[str] = []
    for spec in contracts.FIGURE_SPECS:
        figure = build_figure(
            spec.name, tables=tables, loaded=loaded, config=config
        )
        if figure is None:
            skipped.append(spec.name)
            continue
        source = by_source.get(spec.source_name) if spec.source_name else None
        try:
            artifacts.extend(
                _write_figure(
                    target,
                    spec.name,
                    figure,
                    config=config,
                    classification=(
                        source.data_classification if source else fallback
                    ),
                )
            )
        finally:
            plt.close(figure)

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    summary = render_summary_markdown(
        config=config,
        loaded=loaded,
        tables=tables,
        artifacts=artifacts,
        skipped_figures=skipped,
        generated_at=generated_at,
    )
    with (target / SUMMARY_NAME).open("w", encoding="utf-8", newline="") as handle:
        handle.write(summary)

    manifest = _manifest_document(
        config=config,
        loaded=loaded,
        artifacts=artifacts,
        skipped_figures=skipped,
        implementation_sha256=implementation_sha256,
        generated_at=generated_at,
    )
    with (target / MANIFEST_NAME).open("w", encoding="utf-8", newline="") as handle:
        json.dump(_jsonable(manifest), handle, indent=2, sort_keys=False)
        handle.write("\n")

    _write_checksums(target, artifacts)
    return WrittenProfileBundle(
        root=target,
        artifacts=tuple(artifacts),
        skipped_figures=tuple(skipped),
    )


# --------------------------------------------------------------------------
# summary.md
# --------------------------------------------------------------------------


def _lookup(tables: Mapping[str, pd.DataFrame], name: str) -> pd.DataFrame:
    frame = tables.get(name)
    return contracts.empty_table(name) if frame is None else frame


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(number):
        return "n/a"
    if number == int(number) and abs(number) < 1e15:
        return f"{int(number):,}"
    return f"{number:,.{digits}f}"


def render_summary_markdown(
    *,
    config: DescriptiveStatisticsConfig,
    loaded: LoadedSources,
    tables: Mapping[str, pd.DataFrame],
    artifacts: Sequence[ProfileArtifact],
    skipped_figures: Sequence[str],
    generated_at: str,
) -> str:
    yields = _lookup(tables, "yield_profile")
    nitrogen = _lookup(tables, "nitrogen_rate_profile")

    lines: list[str] = []
    lines.append("# Descriptive statistical profile of the registered datasets")
    lines.append("")
    lines.append(f"Generated {generated_at} · mode `{config.mode}` · schema `{SCHEMA_VERSION}`")
    lines.append("")
    lines.append(
        "This bundle documents the agronomic coverage of every registered source "
        "dataset: recorded N rates and their ladder, grain yield on a common "
        "t/ha basis, temporal coverage, zero-N checks, and the composition of "
        "the context fields each dataset binds. It is descriptive only: it fits "
        "no response model, selects no optimum, and makes no fertilizer "
        "recommendation. Differences reported between datasets are differences "
        "in what was recorded, not measured effects."
    )
    lines.append("")
    # The roster is read straight off the loader rather than off a profile
    # table: it states what was read, and nothing here is computed from the
    # rows, so it does not reintroduce the retired structural profile.
    lines.append("## Datasets profiled")
    lines.append("")
    lines.append("| Dataset | Class | Rows | Physical columns | Encoding | Adapter |")
    lines.append("| --- | --- | ---: | ---: | --- | --- |")
    for source in loaded.sources:
        lines.append(
            "| `{name}` | {cls} | {rows} | {cols} | {enc} | `{adapter}` |".format(
                name=source.source_name,
                cls=source.data_classification,
                rows=_fmt(source.data_row_count),
                cols=_fmt(source.physical_column_count),
                enc=source.source_encoding,
                adapter=source.shape_adapter_version,
            )
        )
    lines.append("")

    if not nitrogen.empty or not yields.empty:
        lines.append("## Agronomic coverage")
        lines.append("")
        lines.append(
            "| Dataset | Observations | N range (kg/ha) | Distinct N rates | "
            "Zero-N rows | Yield median (t/ha) | Yield range (t/ha) |"
        )
        lines.append("| --- | ---: | --- | ---: | ---: | ---: | --- |")
        yield_by_source = {
            str(row["source_name"]): row for _, row in yields.iterrows()
        }
        for _, row in nitrogen.iterrows():
            name = str(row["source_name"])
            y = yield_by_source.get(name)
            lines.append(
                "| `{n}` | {obs} | {nmin}–{nmax} | {rates} | {zero} | {ymed} | {ymin}–{ymax} |".format(
                    n=name,
                    obs=_fmt(row["observation_count"]),
                    nmin=_fmt(row["minimum_kg_ha"], 1),
                    nmax=_fmt(row["maximum_kg_ha"], 1),
                    rates=_fmt(row["distinct_rate_count"]),
                    zero=_fmt(row["zero_n_observation_count"]),
                    ymed=_fmt(y["median_t_ha"], 2) if y is not None else "n/a",
                    ymin=_fmt(y["minimum_t_ha"], 2) if y is not None else "n/a",
                    ymax=_fmt(y["maximum_t_ha"], 2) if y is not None else "n/a",
                )
            )
        lines.append("")

    restricted_names = sorted(
        source.source_name
        for source in loaded.sources
        if source.data_classification == RESTRICTED
    )
    lines.append("## Data governance")
    lines.append("")
    if restricted_names:
        lines.append(
            "Restricted datasets in this profile: "
            + ", ".join(f"`{name}`" for name in restricted_names)
            + ". Every artifact derived from them is labeled with its data "
            "classification in `run_manifest.json`."
        )
    else:
        lines.append("No restricted dataset contributed to this profile.")
    lines.append("")
    lines.append(
        "Suppressed identifier columns (excluded from every table before profiling): "
        + ", ".join(f"`{header}`" for header in sorted(config.suppressed_headers))
        + "."
    )
    lines.append("")
    lines.append(
        f"Categorical levels occurring fewer than {config.minimum_level_count} times are "
        "withheld; only the count of withheld levels is reported, so rare levels of a "
        "restricted context field cannot act as row identifiers."
    )
    lines.append("")
    lines.append(
        "This bundle is not a release artifact. It is a standalone internal "
        "diagnostic written beside the N-response release package and is never "
        "appended to it."
    )
    lines.append("")

    lines.append("## Bundle contents")
    lines.append("")
    tables_written = [a for a in artifacts if a.kind == "table"]
    figures_written = [a for a in artifacts if a.kind == "figure"]
    lines.append(
        f"{len(artifacts)} checksum-bound artifacts: {len(tables_written)} tables and "
        f"{len(figures_written)} figures, plus `{SUMMARY_NAME}`, `{MANIFEST_NAME}`, "
        f"and `{CHECKSUMS_NAME}`."
    )
    lines.append("")
    for group in contracts.TABLE_GROUPS:
        members = [a for a in artifacts if a.group == group]
        if not members:
            continue
        lines.append(f"**{group}/**")
        lines.append("")
        for artifact in members:
            detail = (
                f"{_fmt(artifact.row_count)} rows"
                if artifact.row_count is not None
                else "figure"
            )
            lines.append(
                f"- `{artifact.relative_path}` — {detail} "
                f"({artifact.data_classification})"
            )
        lines.append("")
    if skipped_figures:
        lines.append(
            "Figures skipped for insufficient data: "
            + ", ".join(f"`{name}`" for name in skipped_figures)
            + "."
        )
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def _read_manifest(
    path: Path, *, expected_schema: str | None = SCHEMA_VERSION
) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileBundleError(f"Manifest is unreadable: {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ProfileBundleError(f"Manifest is not a JSON object: {path}")
    if expected_schema is not None and document.get("schema_version") != expected_schema:
        raise ProfileBundleError(
            f"Manifest schema version is {document.get('schema_version')!r}, "
            f"expected {expected_schema!r}"
        )
    if document.get("bundle_status") != BUNDLE_STATUS:
        raise ProfileBundleError("Manifest bundle status is not the diagnostic status")
    return document


def _parse_checksums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or not relative:
            raise ProfileBundleError(f"Malformed checksum line {number} in {path}")
        if relative in entries:
            raise ProfileBundleError(f"Duplicate checksum entry: {relative}")
        entries[relative] = digest
    return entries


def _require_artifact_path(
    entry: Mapping[str, Any], *, schema_version: str
) -> str:
    relative = str(entry.get("relative_path"))
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or not parsed.parts or ".." in parsed.parts:
        raise ProfileBundleError(f"Artifact path is unsafe: {relative!r}")
    retired_buckets = {"figures", "tables"}.intersection(parsed.parts)
    if schema_version == SCHEMA_VERSION and retired_buckets:
        raise ProfileBundleError(
            f"Current-schema artifact path uses a retired type bucket: {relative}"
        )
    if schema_version not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
        raise ProfileBundleError(
            f"Unsupported descriptive-statistics schema: {schema_version!r}"
        )

    kind = entry.get("artifact_kind")
    name = str(entry.get("name"))
    try:
        if kind == "table":
            spec = contracts.table_spec(name)
            expected = spec.relative_path
        elif kind == "figure":
            extension = parsed.suffix.removeprefix(".")
            if not extension:
                raise ProfileBundleError(
                    f"Figure artifact path has no extension: {relative}"
                )
            spec = contracts.figure_spec(name)
            expected = spec.relative_path(extension)
        else:
            raise ProfileBundleError(
                f"Manifest declares an unsupported artifact kind: {kind!r}"
            )
    except contracts.ProfileContractError as exc:
        raise ProfileBundleError(str(exc)) from exc

    if schema_version == LEGACY_SCHEMA_VERSION:
        prefix = "tables" if kind == "table" else "figures"
        expected = f"{prefix}/{expected}"

    if entry.get("group") != spec.group:
        raise ProfileBundleError(
            f"Manifest group disagrees with the contract for {name!r}: "
            f"{entry.get('group')!r} != {spec.group!r}"
        )
    if relative != expected:
        raise ProfileBundleError(
            f"Artifact path disagrees with the current mixed-layout contract for "
            f"{name!r}: {relative!r} != {expected!r}"
        )
    return relative


def _verify_profile_bundle(
    root: str | Path, *, schema_version: str
) -> VerifiedProfileBundle:
    """Re-derive every checksum and reconcile disk, checksums, and manifest.

    No unchecksummed member is tolerated. This bundle has no promoted-package
    extension case, so any file present but unbound is a defect, not an
    allowance.
    """

    target = Path(root).resolve()
    if not target.is_dir():
        raise ProfileBundleError(f"Bundle root does not exist: {target}")
    for name in (SUMMARY_NAME, MANIFEST_NAME, CHECKSUMS_NAME):
        if not (target / name).is_file():
            raise ProfileBundleError(f"Bundle is missing {name}: {target}")

    manifest = _read_manifest(
        target / MANIFEST_NAME, expected_schema=schema_version
    )
    checksums = _parse_checksums(target / CHECKSUMS_NAME)

    present: set[str] = set()
    for path in sorted(target.rglob("*")):
        if path.is_symlink():
            raise ProfileBundleError(f"Bundle contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(target).as_posix()
        if relative == CHECKSUMS_NAME:
            continue
        present.add(relative)

    unbound = sorted(present - set(checksums))
    if unbound:
        raise ProfileBundleError(f"Files are not bound by checksums: {unbound}")
    absent = sorted(set(checksums) - present)
    if absent:
        raise ProfileBundleError(f"Checksummed files are missing: {absent}")
    for relative, expected in sorted(checksums.items()):
        actual = sha256_file(target / relative)
        if actual != expected:
            raise ProfileBundleError(
                f"Checksum mismatch for {relative}: expected {expected}, got {actual}"
            )

    declared = manifest.get("artifacts")
    if not isinstance(declared, list):
        raise ProfileBundleError("Manifest declares no artifact list")
    declared_paths = {
        _require_artifact_path(entry, schema_version=schema_version)
        for entry in declared
    }
    expected_paths = set(checksums) - {SUMMARY_NAME, MANIFEST_NAME}
    if declared_paths != expected_paths:
        missing = sorted(expected_paths - declared_paths)
        extra = sorted(declared_paths - expected_paths)
        raise ProfileBundleError(
            f"Manifest artifact list disagrees with checksums; missing={missing} extra={extra}"
        )
    for entry in declared:
        relative = str(entry.get("relative_path"))
        if entry.get("sha256") != checksums.get(relative):
            raise ProfileBundleError(
                f"Manifest checksum disagrees with CHECKSUMS.sha256 for {relative}"
            )

    tables_declared = sum(1 for e in declared if e.get("artifact_kind") == "table")
    figures_declared = sum(1 for e in declared if e.get("artifact_kind") == "figure")
    restricted = sum(
        1 for e in declared if e.get("data_classification") == RESTRICTED
    )
    declared_table_names = {
        str(e.get("name")) for e in declared if e.get("artifact_kind") == "table"
    }
    missing_tables = sorted(
        spec.name for spec in contracts.TABLE_SPECS if spec.name not in declared_table_names
    )
    if missing_tables:
        raise ProfileBundleError(f"Contract tables absent from bundle: {missing_tables}")

    return VerifiedProfileBundle(
        root=target,
        schema_version=schema_version,
        artifact_count=len(declared),
        table_count=tables_declared,
        figure_count=figures_declared,
        restricted_artifact_count=restricted,
    )


def verify_profile_bundle(root: str | Path) -> VerifiedProfileBundle:
    """Verify a live bundle against the current mixed-layout schema only."""

    return _verify_profile_bundle(root, schema_version=SCHEMA_VERSION)


def verify_replacement_profile_bundle(root: str | Path) -> VerifiedProfileBundle:
    """Verify a replaceable current bundle or the exact prior type-bucket schema."""

    target = Path(root).resolve()
    manifest = _read_manifest(target / MANIFEST_NAME, expected_schema=None)
    schema_version = manifest.get("schema_version")
    if schema_version not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
        raise ProfileBundleError(
            f"Manifest schema version is {schema_version!r}; replacement accepts "
            f"only {SCHEMA_VERSION!r} or {LEGACY_SCHEMA_VERSION!r}"
        )
    return _verify_profile_bundle(target, schema_version=str(schema_version))
