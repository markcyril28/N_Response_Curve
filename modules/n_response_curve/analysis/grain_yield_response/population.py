from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import pandas as pd

from n_response_curve.data.provenance import sha256_file
from .config import GrainYieldResponseConfig, RecipeConfigError


LEDGER_RELATIVE = "tables/quality/analysis_eligibility_ledger.csv"


@dataclass(frozen=True)
class GovernedPopulation:
    frame: pd.DataFrame
    series_uids: tuple[str, ...]
    study_count: int
    trial_count: int
    source_run_id: str
    source_package_status: str
    input_sha256: dict[str, str]
    source_release_verification_status: str = "not_checked"
    source_release_verification_detail: str = ""


@dataclass(frozen=True)
class RawFinitePopulation:
    frame: pd.DataFrame
    source_nonblank_rows: int
    excluded_nonfinite_pairs: int
    yield_t_source_count: int
    yield_kg_fallback_count: int
    input_sha256: dict[str, str]


def distinct_n_level_count(values: Any, *, tolerance: float) -> int:
    """Count sorted N levels using the canonical adjacent-level tolerance rule."""

    levels: list[float] = []
    numeric_values = np.asarray(values, dtype=float).reshape(-1)
    for value in sorted(float(item) for item in numeric_values):
        if not levels or abs(value - levels[-1]) > tolerance:
            levels.append(value)
    return len(levels)


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RecipeConfigError(
                    f"Duplicate JSON key in {path.name}: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecipeConfigError(f"Cannot parse release manifest: {path}") from exc
    if not isinstance(value, dict):
        raise RecipeConfigError("Release manifest must be a JSON object")
    return value


def _checksum_entries(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RecipeConfigError(f"Release checksum ledger is missing: {path}")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise RecipeConfigError(
                f"Malformed release checksum line {line_number}"
            ) from exc
        pure = PurePosixPath(relative)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or pure.is_absolute()
            or ".." in pure.parts
            or str(pure) != relative
            or relative in entries
        ):
            raise RecipeConfigError(
                f"Unsafe release checksum entry on line {line_number}"
            )
        entries[relative] = digest
    if not entries:
        raise RecipeConfigError("Release checksum ledger is empty")
    return entries


def _verify_release_inputs(config: GrainYieldResponseConfig) -> tuple[Path, Path, Path]:
    package = config.release_package
    manifest_path = package / "run_manifest.json"
    ledger_path = package / LEDGER_RELATIVE
    checksums_path = package / "CHECKSUMS.sha256"
    expected_paths = {
        "release_manifest_sha256": manifest_path,
        "release_ledger_sha256": ledger_path,
        "release_checksums_sha256": checksums_path,
    }
    for expected_key, path in expected_paths.items():
        if path.is_symlink() or not path.is_file():
            raise RecipeConfigError(f"Required release input is missing: {path}")
        actual = sha256_file(path)
        if actual != config.expected_inputs[expected_key]:
            raise RecipeConfigError(
                f"Release input hash mismatch for {path.name}: expected "
                f"{config.expected_inputs[expected_key]}, observed {actual}"
            )

    entries = _checksum_entries(checksums_path)
    for required in ("run_manifest.json", LEDGER_RELATIVE):
        if required not in entries:
            raise RecipeConfigError(
                f"Release checksum ledger does not bind {required}"
            )
    for relative, expected_digest in entries.items():
        member = package / relative
        try:
            member.resolve().relative_to(package.resolve())
        except ValueError as exc:
            raise RecipeConfigError(
                f"Release checksum member escapes package: {relative}"
            ) from exc
        if member.is_symlink() or not member.is_file():
            raise RecipeConfigError(
                f"Release checksum member is missing or unsafe: {relative}"
            )
        actual = sha256_file(member)
        if actual != expected_digest:
            raise RecipeConfigError(
                f"Release checksum mismatch for {relative}"
            )
    return manifest_path, ledger_path, checksums_path


def load_governed_population(
    config: GrainYieldResponseConfig,
) -> GovernedPopulation:
    """Load the exact manifest-declared observed-series population."""

    manifest_path, ledger_path, checksums_path = _verify_release_inputs(config)
    manifest = _strict_json(manifest_path)
    observed_status = manifest.get("status")
    if observed_status != config.required_release_status:
        raise RecipeConfigError(
            "Release manifest status mismatch: expected "
            f"{config.required_release_status!r}, observed {observed_status!r}"
        )
    overlay = manifest.get("figures", {}).get("overlay", {})
    if overlay.get("scope") != "governed_observed_series_overlay_only":
        raise RecipeConfigError(
            "Release overlay does not declare governed observed-series scope"
        )
    by_source = overlay.get("series_uids_by_source")
    if not isinstance(by_source, dict):
        raise RecipeConfigError(
            "Release overlay is missing series_uids_by_source"
        )
    raw_uids = by_source.get(config.source_name)
    if not isinstance(raw_uids, list) or not raw_uids:
        raise RecipeConfigError(
            f"No governed overlay series declared for {config.source_name}"
        )
    if any(not isinstance(uid, str) or not uid for uid in raw_uids):
        raise RecipeConfigError("Governed overlay contains an invalid series UID")
    series_uids = tuple(raw_uids)
    if len(series_uids) != len(set(series_uids)):
        raise RecipeConfigError("Governed overlay contains duplicate series UIDs")
    count_by_source = overlay.get("series_count_by_source")
    if (
        not isinstance(count_by_source, dict)
        or count_by_source.get(config.source_name) != len(series_uids)
    ):
        raise RecipeConfigError(
            "Release overlay series count does not match declared source membership"
        )

    try:
        ledger = pd.read_csv(ledger_path, low_memory=False)
    except Exception as exc:
        raise RecipeConfigError(f"Cannot parse analysis ledger: {ledger_path}") from exc
    required = {
        "release_record_uid",
        "source_name",
        config.series_key,
        config.study_key,
        config.trial_key,
        "n_rate_kg_ha",
        "yield_t_ha",
        *config.categorical_factors,
        *config.numeric_factors,
    }
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise RecipeConfigError(
            f"Analysis ledger is missing required columns: {missing}"
        )
    if ledger["release_record_uid"].isna().any() or ledger[
        "release_record_uid"
    ].duplicated().any():
        raise RecipeConfigError(
            "Analysis ledger release_record_uid values must be unique and nonmissing"
        )

    selected = ledger[
        (ledger["source_name"] == config.source_name)
        & ledger[config.series_key].isin(series_uids)
    ].copy()
    if set(selected[config.series_key].dropna().astype(str)) != set(series_uids):
        raise RecipeConfigError(
            "Release manifest series membership does not resolve exactly in the ledger"
        )
    selected["n_rate_kg_ha"] = pd.to_numeric(
        selected["n_rate_kg_ha"], errors="coerce"
    )
    selected["yield_t_ha"] = pd.to_numeric(
        selected["yield_t_ha"], errors="coerce"
    )
    if not np.isfinite(selected[["n_rate_kg_ha", "yield_t_ha"]].to_numpy()).all():
        raise RecipeConfigError(
            "Governed overlay membership contains nonfinite N-rate or yield values"
        )

    for grouping_key in (config.series_key, config.study_key, config.trial_key):
        cleaned = pd.Series(
            selected.loc[:, grouping_key],
            index=selected.index,
            dtype="string",
        ).str.strip()
        if cleaned.isna().any() or cleaned.eq("").any():
            raise RecipeConfigError(
                f"Governed overlay contains missing {grouping_key} identifiers"
            )
        selected[grouping_key] = cleaned.astype(str)

    for uid, group in selected.groupby(config.series_key, sort=False):
        if len(group) < config.minimum_observations:
            raise RecipeConfigError(
                f"Governed series {uid} has fewer than the required observations"
            )
        if distinct_n_level_count(
            group["n_rate_kg_ha"], tolerance=config.n_level_tolerance_kg_ha
        ) < config.minimum_distinct_n_levels:
            raise RecipeConfigError(
                f"Governed series {uid} has fewer than the required distinct N levels"
            )

    # `organic_fertilizer` / `biofertilizer` are the raw evidence columns behind
    # the derived `*_present` factors; the factor evidence audit compares the two.
    optional = [
        column
        for column in (
            "data_classification",
            "representation_basis",
            "organic_fertilizer",
            "biofertilizer",
        )
        if column in selected.columns
    ]
    projection = [
        "release_record_uid",
        "source_name",
        config.study_key,
        config.trial_key,
        config.series_key,
        "n_rate_kg_ha",
        "yield_t_ha",
        *config.categorical_factors,
        *config.numeric_factors,
        *optional,
    ]
    selected = selected.loc[:, list(dict.fromkeys(projection))].sort_values(
        [config.series_key, "n_rate_kg_ha", "release_record_uid"],
        kind="mergesort",
    ).reset_index(drop=True)

    return GovernedPopulation(
        frame=selected,
        series_uids=series_uids,
        study_count=int(selected[config.study_key].nunique(dropna=True)),
        trial_count=int(selected[config.trial_key].nunique(dropna=True)),
        source_run_id=str(manifest.get("run_id", "")),
        source_package_status=str(manifest.get("status", "")),
        input_sha256={
            "release_manifest": sha256_file(manifest_path),
            "release_ledger": sha256_file(ledger_path),
            "release_checksums": sha256_file(checksums_path),
        },
    )


def _finite_number(value: str) -> float | None:
    text = value.strip().replace(",", "")
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def load_raw_finite_population(
    config: GrainYieldResponseConfig,
    *,
    base_config: Any,
) -> RawFinitePopulation:
    """Load a separately labelled all-finite-pair inventory sensitivity frame."""

    try:
        source_spec: Mapping[str, Any] = base_config.sources[config.source_name]
        schema: Mapping[str, Any] = base_config.raw["schema"]
        fields: Mapping[str, Any] = schema["fields"]
        expected_columns = int(schema["expected_physical_columns"])
        if config.source_name == "core_trial_data":
            source_path = Path(base_config.paths["core_source_csv"]).resolve()
        else:
            source_path = (
                config.project_root / str(source_spec["data_path"])
            ).resolve()
        encoding = str(source_spec["encoding"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RecipeConfigError(
            "Base configuration does not provide the registered raw-source schema"
        ) from exc
    try:
        source_path.relative_to(config.project_root)
    except ValueError as exc:
        raise RecipeConfigError("Registered source path escapes the project root") from exc
    if source_path.is_symlink() or not source_path.is_file():
        raise RecipeConfigError(f"Registered source CSV is missing: {source_path}")
    source_hash_before = sha256_file(source_path)
    if source_hash_before != config.expected_inputs["source_sha256"]:
        raise RecipeConfigError(
            "Registered source SHA-256 does not match [expected_inputs].source_sha256"
        )

    field_specs: dict[str, tuple[int, str]] = {}
    for key in ("inorganic_n_rate", "yield_kg_ha", "yield_t_ha"):
        try:
            spec = fields[key]
            position = int(spec["position"]) - 1
            header = str(spec["header"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RecipeConfigError(f"Invalid registered schema field: {key}") from exc
        if position < 0 or position >= expected_columns:
            raise RecipeConfigError(f"Registered schema position is invalid: {key}")
        field_specs[key] = (position, header)

    output_rows: list[dict[str, float | int | str]] = []
    nonblank_rows = 0
    excluded = 0
    direct_yield_count = 0
    fallback_yield_count = 0
    try:
        with source_path.open("r", encoding=encoding, newline="") as handle:
            reader = csv.reader(handle)
            header_row = next(reader)
            if len(header_row) != expected_columns:
                raise RecipeConfigError(
                    "Registered source header width does not match the base configuration"
                )
            for key, (position, expected_header) in field_specs.items():
                if header_row[position] != expected_header:
                    raise RecipeConfigError(
                        f"Registered source header mismatch for {key}: "
                        f"expected {expected_header!r}, observed {header_row[position]!r}"
                    )
            for source_record_number, row in enumerate(reader, start=1):
                if not row or not any(cell.strip() for cell in row):
                    continue
                nonblank_rows += 1
                if len(row) != expected_columns:
                    raise RecipeConfigError(
                        f"Source record {source_record_number} has {len(row)} columns; "
                        f"expected {expected_columns}"
                    )
                n_rate = _finite_number(row[field_specs["inorganic_n_rate"][0]])
                yield_kg = _finite_number(row[field_specs["yield_kg_ha"][0]])
                yield_t = _finite_number(row[field_specs["yield_t_ha"][0]])
                if yield_t is not None and yield_kg is not None:
                    if abs(yield_t - yield_kg / 1000.0) > 0.01:
                        raise RecipeConfigError(
                            f"Yield-unit conflict in source record {source_record_number}"
                        )
                if n_rate is None or (yield_t is None and yield_kg is None):
                    excluded += 1
                    continue
                if yield_t is not None:
                    normalized_yield = yield_t
                    yield_basis = "source_t_ha"
                    direct_yield_count += 1
                else:
                    normalized_yield = float(yield_kg) / 1000.0
                    yield_basis = "converted_kg_ha_to_t_ha"
                    fallback_yield_count += 1
                output_rows.append(
                    {
                        "source_record_number": source_record_number,
                        "n_rate_kg_ha": n_rate,
                        "yield_t_ha": normalized_yield,
                        "yield_basis": yield_basis,
                    }
                )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise RecipeConfigError(f"Cannot read registered source CSV: {source_path}") from exc

    source_hash_after = sha256_file(source_path)
    if source_hash_after != source_hash_before:
        raise RecipeConfigError("Registered source changed while it was being read")
    frame = pd.DataFrame(output_rows)
    if len(frame) < config.minimum_observations:
        raise RecipeConfigError("Raw finite-pair sensitivity population is too small")
    if distinct_n_level_count(
        frame["n_rate_kg_ha"], tolerance=config.n_level_tolerance_kg_ha
    ) < config.minimum_distinct_n_levels:
        raise RecipeConfigError(
            "Raw finite-pair sensitivity has too few distinct N levels"
        )
    return RawFinitePopulation(
        frame=frame,
        source_nonblank_rows=nonblank_rows,
        excluded_nonfinite_pairs=excluded,
        yield_t_source_count=direct_yield_count,
        yield_kg_fallback_count=fallback_yield_count,
        input_sha256={"source_csv": source_hash_after},
    )
