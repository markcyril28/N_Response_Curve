"""Config-driven governed custom observed-series overlays.

Customized views are generated inside Phase 5 release staging. Selection helpers
classify whole response series and retain complete finite trajectories so that
threshold filtering never severs series context.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image


@dataclass(frozen=True)
class YieldThresholdSelection:
    """Deterministic whole-series selection for an observed-yield threshold."""

    response_series_uids: tuple[str, ...]
    exact_threshold_response_series_uids: tuple[str, ...]
    selected_observations: tuple[tuple[str, float, float], ...]
    threshold_exceeding_observations: tuple[tuple[str, float, float], ...]
    above_high_n_observations: tuple[tuple[str, float, float], ...]
    n_rate_range_kg_ha: tuple[float, float] | None


def select_series_with_any_yield_above(
    observations_by_series: Mapping[str, Sequence[tuple[float, float]]],
    *,
    yield_threshold_t_ha: float,
    high_n_threshold_kg_ha: float = 200.0,
) -> YieldThresholdSelection:
    """Select whole series whose finite observed maximum is strictly above a threshold."""

    if not math.isfinite(yield_threshold_t_ha) or not math.isfinite(
        high_n_threshold_kg_ha
    ):
        raise ValueError("Yield and high-N thresholds must be finite")
    normalized: dict[str, tuple[tuple[float, float], ...]] = {}
    for raw_series_uid, values in observations_by_series.items():
        series_uid = str(raw_series_uid).strip()
        if not series_uid:
            raise ValueError("Response-series identifiers must be nonempty")
        observations = tuple(sorted((float(n_rate), float(yield_value)) for n_rate, yield_value in values))
        if not observations or any(
            not math.isfinite(n_rate) or not math.isfinite(yield_value)
            for n_rate, yield_value in observations
        ):
            raise ValueError("Every response series must contain finite observations")
        normalized[series_uid] = observations

    selected_series = tuple(
        sorted(
            series_uid
            for series_uid, observations in normalized.items()
            if max(yield_value for _, yield_value in observations)
            > yield_threshold_t_ha
        )
    )
    exact_threshold_series = tuple(
        sorted(
            series_uid
            for series_uid, observations in normalized.items()
            if max(yield_value for _, yield_value in observations)
            == yield_threshold_t_ha
        )
    )
    selected_observations = tuple(
        (series_uid, n_rate, yield_value)
        for series_uid in selected_series
        for n_rate, yield_value in normalized[series_uid]
    )
    threshold_exceeding = tuple(
        observation
        for observation in selected_observations
        if observation[2] > yield_threshold_t_ha
    )
    above_high_n = tuple(
        observation
        for observation in selected_observations
        if observation[1] > high_n_threshold_kg_ha
    )
    n_rate_range = (
        (
            min(observation[1] for observation in selected_observations),
            max(observation[1] for observation in selected_observations),
        )
        if selected_observations
        else None
    )
    return YieldThresholdSelection(
        response_series_uids=selected_series,
        exact_threshold_response_series_uids=exact_threshold_series,
        selected_observations=selected_observations,
        threshold_exceeding_observations=threshold_exceeding,
        above_high_n_observations=above_high_n,
        n_rate_range_kg_ha=n_rate_range,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FIGURE_LAYOUT_V3 = "figures-by-source-model-overlay-only-v3"
_LEGACY_FIGURE_LAYOUTS = frozenset(
    {
        "figures-by-source-and-model-v1",
        "figures-by-source-model-and-overlay-v2",
    }
)


def _sha256(path: Path) -> str:
    """Hash a file through the shared reporting helper.

    Keep the import lazy because this file is also a directly executable script;
    :func:`main` adds ``modules/`` to ``sys.path`` before the helper is called.
    """

    from n_response_curve.reporting.source_config_spec import sha256_file

    return sha256_file(path)


def _image_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        return {
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
        }


def _parse_checksum_ledger(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise RuntimeError(f"Malformed checksum entry in {path}: {line!r}") from exc
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or relative in entries:
            raise RuntimeError(f"Invalid or duplicate checksum entry in {path}: {line!r}")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeError(f"Unsafe checksum path in {path}: {relative!r}")
        entries[relative] = digest
    if not entries:
        raise RuntimeError(f"Checksum ledger is empty: {path}")
    return entries


def _verify_bound_files(package: Path, expected: Mapping[str, str]) -> None:
    mismatches = [
        relative
        for relative, digest in expected.items()
        if not (package / relative).is_file()
        or _sha256(package / relative) != digest
    ]
    if mismatches:
        raise RuntimeError(f"Original release artifacts changed: {mismatches}")


def _hash_project_inputs(project_root: Path, paths: Sequence[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted({item.resolve() for item in paths}):
        try:
            relative = path.relative_to(project_root).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"Customized-plot input is outside the project: {path}") from exc
        result[relative] = _sha256(path)
    return result


def _recorded_input_hashes(
    project_root: Path,
    package: Path,
    paths: Sequence[Path],
) -> dict[str, str]:
    """Record stage-local inputs under stable package-relative names."""

    result: dict[str, str] = {}
    for path in sorted({item.resolve() for item in paths}):
        if path.is_relative_to(package):
            relative = f"release_package/{path.relative_to(package).as_posix()}"
        else:
            try:
                relative = path.relative_to(project_root).as_posix()
            except ValueError as exc:
                raise RuntimeError(
                    f"Customized-plot input is outside the project: {path}"
                ) from exc
        result[relative] = _sha256(path)
    return result


def _finite_pair(row: Mapping[str, str]) -> tuple[float, float] | None:
    try:
        n_rate = float(row["n_rate_kg_ha"])
        yield_value = float(row["yield_t_ha"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(n_rate) or not math.isfinite(yield_value):
        return None
    return n_rate, yield_value


def _threshold_token(value: float) -> str:
    token = format(value, "g").replace("-", "minus_").replace(".", "_")
    if not re.fullmatch(r"[A-Za-z0-9_]+", token):
        raise RuntimeError(f"Cannot create a safe threshold token from {value!r}")
    return token


def _manifest_plot_key(yield_threshold_t_ha: float) -> str:
    return f"any_observed_yield_above_{_threshold_token(yield_threshold_t_ha)}"


def _source_token(source_name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", source_name.strip()).strip("._-")
    if not token:
        raise RuntimeError("Cannot create a safe output token from the source name")
    return token


def _is_v3_figure_layout(figure_inventory: Mapping[str, Any]) -> bool:
    """Return whether a package declares the overlay-only v3 figure contract."""

    return figure_inventory.get("layout_version") == _FIGURE_LAYOUT_V3


def resolve_v3_governed_series_uids(
    figure_inventory: Mapping[str, Any], *, source_name: str
) -> tuple[str, ...]:
    """Resolve and validate exact governed series membership from a v3 manifest."""

    if not isinstance(source_name, str) or not source_name.strip():
        raise RuntimeError("The requested custom-overlay source name must be nonempty")
    if not _is_v3_figure_layout(figure_inventory):
        raise RuntimeError("The release package does not declare the v3 figure layout")

    source_tokens = figure_inventory.get("source_directory_tokens")
    if not isinstance(source_tokens, Mapping) or source_name not in source_tokens:
        raise RuntimeError(
            f"Custom-overlay source {source_name!r} is not declared by the release figure inventory"
        )
    declared_source_token = source_tokens.get(source_name)
    if (
        not isinstance(declared_source_token, str)
        or not declared_source_token
        or declared_source_token != _source_token(source_name)
    ):
        raise RuntimeError("Release figure source token mapping is invalid")

    overlay = figure_inventory.get("overlay")
    if (
        not isinstance(overlay, Mapping)
        or overlay.get("scope") != "governed_observed_series_overlay_only"
    ):
        raise RuntimeError("Release figure overlay declaration is invalid")
    raw_uids_by_source = overlay.get("series_uids_by_source")
    raw_counts_by_source = overlay.get("series_count_by_source")
    expected_sources = set(source_tokens)
    if (
        not isinstance(raw_uids_by_source, Mapping)
        or set(raw_uids_by_source) != expected_sources
    ):
        raise RuntimeError("No governed series-uid declaration exists for every source")
    if (
        not isinstance(raw_counts_by_source, Mapping)
        or set(raw_counts_by_source) != expected_sources
    ):
        raise RuntimeError("Release overlay series count-by-source declaration is invalid")

    normalized_by_source: dict[str, tuple[str, ...]] = {}
    uid_owners: dict[str, str] = {}
    for declared_source in sorted(expected_sources):
        raw_uids = raw_uids_by_source.get(declared_source)
        if declared_source == source_name and isinstance(raw_uids, list) and not raw_uids:
            raise RuntimeError(
                f"Governed series-uid inventory is invalid or empty for {source_name!r}"
            )
        if (
            not isinstance(raw_uids, list)
            or any(
                not isinstance(uid, str) or not uid.strip() or uid != uid.strip()
                for uid in raw_uids
            )
        ):
            raise RuntimeError(
                f"Governed series-uid inventory is invalid or empty for {declared_source!r}"
            )
        if len(raw_uids) != len(set(raw_uids)):
            raise RuntimeError(
                f"Governed series-uid inventory contains a duplicate for {declared_source!r}"
            )
        normalized = tuple(sorted(raw_uids))
        declared_count = raw_counts_by_source.get(declared_source)
        if (
            isinstance(declared_count, bool)
            or not isinstance(declared_count, int)
            or declared_count != len(normalized)
        ):
            raise RuntimeError("Release overlay series count-by-source does not reconcile")
        for uid in normalized:
            prior_owner = uid_owners.setdefault(uid, declared_source)
            if prior_owner != declared_source:
                raise RuntimeError(
                    "A governed series uid is assigned to multiple release sources"
                )
        normalized_by_source[declared_source] = normalized

    declared_total = overlay.get("series_count")
    expected_total = sum(len(uids) for uids in normalized_by_source.values())
    if (
        isinstance(declared_total, bool)
        or not isinstance(declared_total, int)
        or declared_total != expected_total
    ):
        raise RuntimeError("Release overlay total series count does not reconcile")
    requested_uids = normalized_by_source[source_name]
    if not requested_uids:
        raise RuntimeError(
            f"Governed series-uid inventory is invalid or empty for {source_name!r}"
        )
    return requested_uids


def _resolve_legacy_governed_series_uids(
    observed_directory: Path,
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """Retain the historical v1/v2 physical observed-image membership rule."""

    observed_paths = tuple(sorted(observed_directory.glob("series_*.jpeg")))
    series_ids = tuple(sorted(path.stem for path in observed_paths))
    if not series_ids or len(series_ids) != len(set(series_ids)):
        raise RuntimeError("Governed observed-series inventory is empty or duplicated")
    return observed_paths, series_ids


_CUSTOM_PLOT_MANIFEST_SCHEMA_VERSION = "post-release-custom-plots-manifest-v1"


def _threshold_image_name_pattern(source_token: str) -> re.Pattern[str]:
    return re.compile(
        rf"^{re.escape(source_token)}_series_with_yield_above_\d+(?:_\d+)?_t_ha\.jpeg$"
    )


def find_superseded_threshold_image(
    *,
    output_root: Path,
    existing_manifest_path: Path,
    existing_checksums_path: Path,
    source_token: str,
    new_image_name: str,
) -> Path | None:
    """Identify a prior generator-managed threshold image safe to retire.

    A candidate is recognized only when it is named by an ``output.path`` of
    a same-family plot entry in the existing custom-plots manifest, covered
    by a matching digest in the existing checksum ledger, resident directly
    under ``output_root``, and matches the tightly scoped
    ``<source_token>_series_with_yield_above_<threshold_token>_t_ha.jpeg``
    filename pattern. Any mismatch returns ``None`` rather than guessing, so
    unrelated or unverifiable files are never retired.
    """

    if not existing_manifest_path.is_file() or not existing_checksums_path.is_file():
        return None
    try:
        manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != _CUSTOM_PLOT_MANIFEST_SCHEMA_VERSION
    ):
        return None
    manifest_source = manifest.get("source_name")
    if not isinstance(manifest_source, str):
        return None
    try:
        if _source_token(manifest_source) != source_token:
            return None
    except RuntimeError:
        return None
    plots = manifest.get("plots")
    if not isinstance(plots, dict):
        return None
    try:
        checksum_entries = _parse_checksum_ledger(existing_checksums_path)
    except (OSError, RuntimeError):
        return None
    manifest_digest = checksum_entries.get(existing_manifest_path.name)
    if manifest_digest is None or _sha256(existing_manifest_path) != manifest_digest:
        return None

    pattern = _threshold_image_name_pattern(source_token)
    candidate_outputs = sorted(
        {
            (plot["output"]["path"], plot["output"].get("sha256"))
            for plot in plots.values()
            if isinstance(plot, dict)
            and isinstance(plot.get("output"), dict)
            and isinstance(plot["output"].get("path"), str)
        }
    )
    for name, declared_digest in candidate_outputs:
        if name == new_image_name or Path(name).name != name or not pattern.fullmatch(name):
            continue
        digest = checksum_entries.get(name)
        if digest is None or declared_digest != digest:
            continue
        candidate_path = output_root / name
        if candidate_path.is_file() and _sha256(candidate_path) == digest:
            return candidate_path
    return None


def _active_pipeline_or_r_processes() -> tuple[tuple[int, str], ...]:
    active: list[tuple[int, str]] = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return ()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) in {os.getpid(), os.getppid()}:
            continue
        try:
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", "replace")
                .strip()
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command and (
            "n_response_curve_pipeline.py" in command or command.startswith("Rscript ")
        ):
            active.append((int(entry.name), command))
    return tuple(active)


@dataclass(frozen=True)
class ZeroNGroupSummary:
    """Whole-series membership and finite-trajectory metadata for one zero-N stratum."""

    response_series_uids: tuple[str, ...]
    baseline_yield_t_ha_by_series: dict[str, float]
    baseline_yield_range_t_ha: tuple[float, float] | None
    finite_observation_count: int
    n_rate_range_kg_ha: tuple[float, float] | None
    above_high_n_response_series_uids: tuple[str, ...]
    above_high_n_observations: tuple[tuple[str, float, float], ...]


@dataclass(frozen=True)
class ZeroNStrataClassification:
    """Deterministic whole-series zero-N baseline classification for a threshold."""

    below: ZeroNGroupSummary
    above: ZeroNGroupSummary
    equal_threshold_response_series_uids: tuple[str, ...]
    missing_baseline_response_series_uids: tuple[str, ...]
    conflicting_baseline_response_series_uids: tuple[str, ...]
    governed_series_count: int
    governed_finite_observation_count: int


def classify_zero_n_strata(
    observations_by_series: Mapping[str, Sequence[tuple[float, float]]],
    *,
    threshold_t_ha: float,
    baseline_n_rate_kg_ha: float = 0.0,
    high_n_threshold_kg_ha: float = 200.0,
) -> ZeroNStrataClassification:
    """Classify whole governed series by one distinct finite yield at the baseline N rate.

    Strict ``<``/``>`` comparisons against ``threshold_t_ha`` decide membership;
    a baseline that equals the threshold, or is missing, is excluded and
    recorded rather than guessed. A baseline N rate with more than one
    distinct finite yield is recorded as conflicting rather than resolved
    here, so callers can fail closed before any output is staged.
    """

    if not math.isfinite(threshold_t_ha) or not math.isfinite(baseline_n_rate_kg_ha) or not math.isfinite(
        high_n_threshold_kg_ha
    ):
        raise ValueError("Zero-N classification thresholds must be finite")
    normalized: dict[str, tuple[tuple[float, float], ...]] = {}
    for raw_series_uid, values in observations_by_series.items():
        series_uid = str(raw_series_uid).strip()
        if not series_uid:
            raise ValueError("Response-series identifiers must be nonempty")
        observations = tuple(sorted((float(n_rate), float(yield_value)) for n_rate, yield_value in values))
        if not observations or any(
            not math.isfinite(n_rate) or not math.isfinite(yield_value)
            for n_rate, yield_value in observations
        ):
            raise ValueError("Every response series must contain finite observations")
        normalized[series_uid] = observations

    below: list[str] = []
    above: list[str] = []
    equal: list[str] = []
    missing: list[str] = []
    conflicts: list[str] = []
    baseline_yield_by_series: dict[str, float] = {}
    for series_uid in sorted(normalized):
        baseline_values = sorted(
            {
                yield_value
                for n_rate, yield_value in normalized[series_uid]
                if n_rate == baseline_n_rate_kg_ha
            }
        )
        if not baseline_values:
            missing.append(series_uid)
        elif len(baseline_values) > 1:
            conflicts.append(series_uid)
        else:
            baseline_yield = baseline_values[0]
            baseline_yield_by_series[series_uid] = baseline_yield
            if baseline_yield < threshold_t_ha:
                below.append(series_uid)
            elif baseline_yield > threshold_t_ha:
                above.append(series_uid)
            else:
                equal.append(series_uid)

    def build_group(series_uids: Sequence[str]) -> ZeroNGroupSummary:
        ordered = tuple(sorted(series_uids))
        observations = tuple(
            (series_uid, n_rate, yield_value)
            for series_uid in ordered
            for n_rate, yield_value in normalized[series_uid]
        )
        baseline_map = {series_uid: baseline_yield_by_series[series_uid] for series_uid in ordered}
        above_high_n = tuple(
            observation for observation in observations if observation[1] > high_n_threshold_kg_ha
        )
        return ZeroNGroupSummary(
            response_series_uids=ordered,
            baseline_yield_t_ha_by_series=baseline_map,
            baseline_yield_range_t_ha=(
                (min(baseline_map.values()), max(baseline_map.values())) if baseline_map else None
            ),
            finite_observation_count=len(observations),
            n_rate_range_kg_ha=(
                (min(o[1] for o in observations), max(o[1] for o in observations))
                if observations
                else None
            ),
            above_high_n_response_series_uids=tuple(sorted({o[0] for o in above_high_n})),
            above_high_n_observations=above_high_n,
        )

    return ZeroNStrataClassification(
        below=build_group(below),
        above=build_group(above),
        equal_threshold_response_series_uids=tuple(sorted(equal)),
        missing_baseline_response_series_uids=tuple(sorted(missing)),
        conflicting_baseline_response_series_uids=tuple(sorted(conflicts)),
        governed_series_count=len(normalized),
        governed_finite_observation_count=sum(len(v) for v in normalized.values()),
    )


def zero_n_strata_artifact_names(
    *, source_token: str, zero_threshold_token: str
) -> tuple[str, str, str, str]:
    """Filenames for the two zero-N images plus their legacy v1 manifest/checksum ledger."""

    return (
        f"{source_token}_zero_n_below_{zero_threshold_token}_t_ha.jpeg",
        f"{source_token}_zero_n_above_{zero_threshold_token}_t_ha.jpeg",
        "zero_n_strata_diagnostic_manifest.json",
        "ZERO_N_STRATA_DIAGNOSTIC_CHECKSUMS.sha256",
    )


def custom_overlay_target_names(
    *,
    source_token: str,
    threshold_token: str,
    zero_threshold_token: str,
    generate_zero_n_strata: bool,
) -> tuple[str, ...]:
    """The full custom-overlay artifact filename set for one run.

    Three names (the yield-threshold image, the unified custom manifest, and
    its checksum ledger) are always present. When ``generate_zero_n_strata``
    is set, the two zero-N images and their legacy v1 manifest/checksum
    ledger are generated too, for seven names total.
    """

    names = [
        f"{source_token}_series_with_yield_above_{threshold_token}_t_ha.jpeg",
        "custom_plots_manifest.json",
        "CUSTOM_PLOTS_CHECKSUMS.sha256",
    ]
    if generate_zero_n_strata:
        names = list(
            zero_n_strata_artifact_names(
                source_token=source_token, zero_threshold_token=zero_threshold_token
            )
        ) + names
    return tuple(names)


def _zero_n_release_metadata(
    *,
    run_manifest: Mapping[str, Any],
    figure_inventory: Mapping[str, Any],
    release_integrity_block: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve the legacy v1 zero-N release context without overstating verification."""

    original_release_integrity = dict(release_integrity_block)
    original_release_integrity.update(
        {
            "semantic_verifier_before_additions": "not_claimed",
            "original_bound_files_unchanged_before_promotion": True,
        }
    )
    return {
        "source_package_status": run_manifest.get("status"),
        "source_package_figure_layout": figure_inventory.get("layout_version"),
        "original_release_integrity": original_release_integrity,
    }


def _cleanup_generation_resources(
    *,
    staging: Path | None,
    lock_fd: int,
    lock_path: Path,
) -> None:
    """Release the lock even when removal of an incomplete staging tree fails."""

    try:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
    finally:
        try:
            os.close(lock_fd)
        finally:
            lock_path.unlink(missing_ok=True)


def _package_files_without_release_ledger(
    package: Path,
    release_checksums_path: Path,
    *,
    ignored: Sequence[Path] = (),
) -> set[str]:
    ignored_resolved = {path.resolve() for path in ignored}
    return {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file()
        and path.resolve() != release_checksums_path.resolve()
        and path.resolve() not in ignored_resolved
    }


def generate_custom_overlays(
    *,
    project_root: Path,
    config_path: Path,
    package: Path,
    source_name: str,
    yield_threshold_t_ha: float,
    high_n_threshold_kg_ha: float,
    replace_existing: bool,
    generate_zero_n_strata: bool,
    zero_n_yield_threshold_t_ha: float,
    configured_figure_formats: Sequence[str],
    governed_release: bool = False,
    allow_active_pipeline: bool = False,
    source_package_path: Path | None = None,
) -> dict[str, Any]:
    """Generate configured custom overlays from a complete staged package.

    Always generates the yield-threshold overlay plus the unified custom-plot
    manifest and checksum ledger. When ``generate_zero_n_strata`` is set, the
    two zero-N baseline strata overlays and their manifest/checksum ledger are
    generated too, from the same governed observed-series inputs -- never read
    back from a pre-existing file. ``governed_release`` is reserved for the
    Phase 5 final-stage writer; the legacy mode remains readable for historical
    package compatibility.
    """

    project_root = project_root.resolve()
    config_path = config_path.resolve()
    package = package.resolve()
    try:
        package.relative_to(project_root)
    except ValueError as exc:
        raise RuntimeError("Configured custom-overlay package is outside the project") from exc
    if not package.is_dir():
        raise RuntimeError(f"Configured custom-overlay package does not exist: {package}")
    active_processes = _active_pipeline_or_r_processes()
    if active_processes and not allow_active_pipeline:
        raise RuntimeError(f"Competing pipeline/R processes detected: {active_processes}")
    logical_package = (source_package_path or package).resolve()
    try:
        source_package_relative = logical_package.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise RuntimeError("Logical custom-overlay package is outside the project") from exc

    source_token = _source_token(source_name)
    threshold_token = _threshold_token(yield_threshold_t_ha)
    zero_threshold_token = _threshold_token(zero_n_yield_threshold_t_ha)

    output_root = package / "figures" / "overlay"
    run_manifest_path = package / "run_manifest.json"
    release_checksums_path = package / "CHECKSUMS.sha256"
    ledger_path = package / "tables" / "quality" / "analysis_eligibility_ledger.csv"
    plots_module_path = project_root / "modules" / "n_response_curve" / "reporting" / "plots.py"
    observed_directory = package / "figures" / "observed" / source_token
    canonical_overlay_path = output_root / f"{source_token}.jpeg"
    new_image_name = f"{source_token}_series_with_yield_above_{threshold_token}_t_ha.jpeg"
    custom_manifest_name = "custom_plots_manifest.json"
    custom_checksums_name = "CUSTOM_PLOTS_CHECKSUMS.sha256"
    zero_below_name, zero_above_name, zero_manifest_name, zero_checksums_name = (
        zero_n_strata_artifact_names(
            source_token=source_token, zero_threshold_token=zero_threshold_token
        )
    )
    target_names = custom_overlay_target_names(
        source_token=source_token,
        threshold_token=threshold_token,
        zero_threshold_token=zero_threshold_token,
        generate_zero_n_strata=generate_zero_n_strata,
    )
    targets = tuple(output_root / name for name in target_names)

    required = [
        config_path,
        Path(__file__).resolve(),
        run_manifest_path,
        release_checksums_path,
        ledger_path,
        plots_module_path,
        canonical_overlay_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing required customized-plot inputs: {missing}")
    existing_targets = [path.name for path in targets if path.exists()]
    if existing_targets and not replace_existing:
        raise RuntimeError(f"Refusing to replace customized outputs: {existing_targets}")

    normalized_formats = tuple(str(value).casefold() for value in configured_figure_formats)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    figure_inventory = run_manifest.get("figures")
    if not isinstance(figure_inventory, dict):
        raise RuntimeError("Release figure inventory is absent")
    package_formats = tuple(
        str(value).casefold() for value in figure_inventory.get("formats", ())
    )
    if "jpeg" not in normalized_formats or "jpeg" not in package_formats:
        raise RuntimeError(
            "Customized JPEG generation requires JPEG in both current and package profiles"
        )

    release_entries = _parse_checksum_ledger(release_checksums_path)
    _verify_bound_files(package, release_entries)
    actual_before = _package_files_without_release_ledger(
        package,
        release_checksums_path,
    )
    target_relative = {
        path.relative_to(package).as_posix() for path in targets if path.is_file()
    }
    superseded_image = find_superseded_threshold_image(
        output_root=output_root,
        existing_manifest_path=output_root / custom_manifest_name,
        existing_checksums_path=output_root / custom_checksums_name,
        source_token=source_token,
        new_image_name=new_image_name,
    )
    superseded_relative = (
        {superseded_image.relative_to(package).as_posix()}
        if superseded_image is not None
        else set()
    )
    unknown_extras = (
        actual_before
        - set(release_entries)
        - target_relative
        - superseded_relative
    )
    if unknown_extras:
        raise RuntimeError(
            f"Unexpected pre-existing unbound package files: {sorted(unknown_extras)}"
        )

    if _is_v3_figure_layout(figure_inventory):
        series_ids = resolve_v3_governed_series_uids(
            figure_inventory,
            source_name=source_name,
        )
        observed_paths: tuple[Path, ...] = ()
    else:
        layout_version = figure_inventory.get("layout_version")
        if layout_version not in _LEGACY_FIGURE_LAYOUTS:
            raise RuntimeError(
                f"Unsupported release figure layout for custom overlays: {layout_version!r}"
            )
        observed_paths, series_ids = _resolve_legacy_governed_series_uids(
            observed_directory
        )

    provenance_inputs = [
        config_path,
        Path(__file__).resolve(),
        ledger_path,
        plots_module_path,
        canonical_overlay_path,
        *observed_paths,
    ]
    input_paths = provenance_inputs if governed_release else [*required, *observed_paths]
    before = _hash_project_inputs(project_root, input_paths)
    recorded_inputs = _recorded_input_hashes(project_root, package, input_paths)
    series_set = set(series_ids)
    records: list[dict[str, str]] = []
    observations_by_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with ledger_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            series_uid = row.get("response_series_uid", "")
            if row.get("source_name") != source_name or series_uid not in series_set:
                continue
            pair = _finite_pair(row)
            if pair is None:
                continue
            records.append(row)
            observations_by_series[series_uid].append(pair)
    if set(observations_by_series) != series_set:
        raise RuntimeError("Finite governed observations do not cover the series inventory")

    selection = select_series_with_any_yield_above(
        observations_by_series,
        yield_threshold_t_ha=yield_threshold_t_ha,
        high_n_threshold_kg_ha=high_n_threshold_kg_ha,
    )
    if not selection.response_series_uids:
        raise RuntimeError(
            f"No governed series has observed grain yield > {yield_threshold_t_ha:g} t/ha"
        )
    zero_classification: ZeroNStrataClassification | None = None
    if generate_zero_n_strata:
        zero_classification = classify_zero_n_strata(
            observations_by_series,
            threshold_t_ha=zero_n_yield_threshold_t_ha,
            high_n_threshold_kg_ha=high_n_threshold_kg_ha,
        )
        if zero_classification.conflicting_baseline_response_series_uids:
            raise RuntimeError(
                "Multiple distinct zero-N yields require an explicit classification rule: "
                f"{zero_classification.conflicting_baseline_response_series_uids}"
            )
        if not zero_classification.below.response_series_uids or not zero_classification.above.response_series_uids:
            raise RuntimeError(
                "Both zero-N baseline strata must be nonempty to generate stratified diagnostics"
            )

    series_inventory_sha256 = hashlib.sha256("\n".join(series_ids).encode("utf-8")).hexdigest()
    release_integrity_block = {
        "bound_artifact_count": len(release_entries),
        "bound_artifacts_reverified_unchanged": True,
        "run_manifest_sha256": _sha256(run_manifest_path),
        "release_checksum_ledger_sha256": _sha256(release_checksums_path),
    }
    release_contract_effect = (
        (
            "These configured descriptive overlays are governed release artifacts under "
            f"the {figure_inventory.get('layout_version')} figure inventory. Their seven-file "
            "inventory is declared in run_manifest.json and covered by CHECKSUMS.sha256."
        )
        if governed_release
        else (
            "These custom plots are post-release diagnostics, not governed "
            f"{figure_inventory.get('layout_version')} release overlays. "
            "The original run_manifest.json and CHECKSUMS.sha256 remain unchanged; the strict "
            "complete-package verifier rejects these extra files as designed."
        )
    )

    module_root = project_root / "modules"
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))
    from matplotlib import pyplot as plt
    from n_response_curve.reporting.plots import create_source_series_overlay_figure

    lock_path = package / "figures" / ".custom-overlay-generation.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Generation lock exists: {lock_path}") from exc
    os.write(lock_fd, f"pid={os.getpid()}\n".encode("utf-8"))
    os.fsync(lock_fd)

    staging: Path | None = None
    try:
        staging = package / "figures" / f".custom-overlay-stage-{uuid.uuid4().hex}"
        staging.mkdir(parents=False, exist_ok=False)

        def render(
            response_series_uids: Sequence[str], destination: Path, title_lines: Sequence[str]
        ) -> dict[str, Any]:
            figure, axes = create_source_series_overlay_figure(
                records,
                source_name,
                response_series_uids=tuple(response_series_uids),
            )
            try:
                axes.set_xlabel("Applied N (kg N/ha)")
                axes.set_ylabel("Grain yield (t/ha)")
                axes.set_title("\n".join(title_lines))
                figure.savefig(destination, format="jpeg", dpi=150)
            finally:
                plt.close(figure)
            metadata = _image_metadata(destination)
            if metadata != {"format": "JPEG", "width": 1500, "height": 1050, "mode": "RGB"}:
                raise RuntimeError(f"Unexpected rendered image metadata: {destination}: {metadata}")
            return metadata

        staged_image = staging / new_image_name
        n_range = selection.n_rate_range_kg_ha
        assert n_range is not None
        new_image_metadata = render(
            selection.response_series_uids,
            staged_image,
            (
                f"source={source_name}",
                f"series with any observed grain yield > {yield_threshold_t_ha:g} t/ha",
                "full series trajectories; descriptive overlay (no pooled curve or fit)",
                f"series={len(selection.response_series_uids)}; "
                f"observations={len(selection.selected_observations)}; "
                f"threshold-exceeding points={len(selection.threshold_exceeding_observations)}; "
                f"N range={n_range[0]:g}-{n_range[1]:g} kg/ha",
            ),
        )

        staged_zero_below: Path | None = None
        staged_zero_above: Path | None = None
        zero_below_metadata: dict[str, Any] | None = None
        zero_above_metadata: dict[str, Any] | None = None
        if generate_zero_n_strata:
            assert zero_classification is not None
            staged_zero_below = staging / zero_below_name
            zero_below_metadata = render(
                zero_classification.below.response_series_uids,
                staged_zero_below,
                (
                    f"source={source_name}",
                    f"observed yield < {zero_n_yield_threshold_t_ha:g} t/ha at exactly 0 kg N/ha",
                    "full series trajectories; descriptive overlay (no pooled curve or fit)",
                    f"series={len(zero_classification.below.response_series_uids)}; "
                    f"observations={zero_classification.below.finite_observation_count}",
                ),
            )
            staged_zero_above = staging / zero_above_name
            zero_above_metadata = render(
                zero_classification.above.response_series_uids,
                staged_zero_above,
                (
                    f"source={source_name}",
                    f"observed yield > {zero_n_yield_threshold_t_ha:g} t/ha at exactly 0 kg N/ha",
                    "full series trajectories; descriptive overlay (no pooled curve or fit)",
                    f"series={len(zero_classification.above.response_series_uids)}; "
                    f"observations={zero_classification.above.finite_observation_count}",
                ),
            )

        if observed_paths:
            current_observed = sorted(observed_directory.glob("series_*.jpeg"))
            if [path.name for path in current_observed] != [
                path.name for path in observed_paths
            ]:
                raise RuntimeError("Governed series inventory changed during rendering")
        after = _hash_project_inputs(project_root, input_paths)
        if after != before:
            changed = sorted(
                relative for relative in set(before) | set(after)
                if before.get(relative) != after.get(relative)
            )
            raise RuntimeError(f"Customized-plot inputs changed during rendering: {changed}")
        _verify_bound_files(package, release_entries)

        staged_zero_manifest: Path | None = None
        staged_zero_checksums: Path | None = None
        if generate_zero_n_strata:
            assert zero_classification is not None
            assert staged_zero_below is not None and staged_zero_above is not None
            assert zero_below_metadata is not None and zero_above_metadata is not None

            def group_block(
                summary: ZeroNGroupSummary,
                criterion: str,
                output_path: Path,
                image_metadata: dict[str, Any],
            ) -> dict[str, Any]:
                return {
                    "criterion": criterion,
                    "response_series_uids": list(summary.response_series_uids),
                    "series_count": len(summary.response_series_uids),
                    "finite_observation_count": summary.finite_observation_count,
                    "n_rate_range_kg_ha": list(summary.n_rate_range_kg_ha or ()),
                    "baseline_yield_t_ha_by_series": summary.baseline_yield_t_ha_by_series,
                    "baseline_yield_range_t_ha": list(summary.baseline_yield_range_t_ha or ()),
                    "above_high_n_observation_count": len(summary.above_high_n_observations),
                    "above_high_n_response_series_uids": list(
                        summary.above_high_n_response_series_uids
                    ),
                    "output": {
                        "path": output_path.name,
                        "sha256": _sha256(output_path),
                        **image_metadata,
                    },
                }

            below_key = f"below_{zero_threshold_token}"
            above_key = f"above_{zero_threshold_token}"
            zero_groups = {
                below_key: group_block(
                    zero_classification.below,
                    f"observed yield < {zero_n_yield_threshold_t_ha:g} t/ha at exactly 0 kg N/ha",
                    staged_zero_below,
                    zero_below_metadata,
                ),
                above_key: group_block(
                    zero_classification.above,
                    f"observed yield > {zero_n_yield_threshold_t_ha:g} t/ha at exactly 0 kg N/ha",
                    staged_zero_above,
                    zero_above_metadata,
                ),
            }
            combined_above_high_n = sorted(
                (
                    *(
                        {
                            "group": below_key,
                            "response_series_uid": series_uid,
                            "n_rate_kg_ha": n_rate,
                            "yield_t_ha": yield_value,
                        }
                        for series_uid, n_rate, yield_value in zero_classification.below.above_high_n_observations
                    ),
                    *(
                        {
                            "group": above_key,
                            "response_series_uid": series_uid,
                            "n_rate_kg_ha": n_rate,
                            "yield_t_ha": yield_value,
                        }
                        for series_uid, n_rate, yield_value in zero_classification.above.above_high_n_observations
                    ),
                ),
                key=lambda observation: (
                    observation["response_series_uid"],
                    observation["n_rate_kg_ha"],
                ),
            )
            zero_classification_block = {
                "source_name": source_name,
                "baseline_n_rate_kg_ha": 0.0,
                "yield_threshold_t_ha": zero_n_yield_threshold_t_ha,
                "high_n_threshold_kg_ha": high_n_threshold_kg_ha,
                "comparison_semantics": (
                    "Classify a whole governed response series using one distinct finite "
                    "yield observed at exactly 0 kg N/ha, then plot all finite observations "
                    "from qualifying series."
                ),
                "governed_series_count": zero_classification.governed_series_count,
                "governed_finite_observation_count": (
                    zero_classification.governed_finite_observation_count
                ),
                "classified_series_count": (
                    len(zero_classification.below.response_series_uids)
                    + len(zero_classification.above.response_series_uids)
                ),
                "excluded_missing_zero_n_series_count": len(
                    zero_classification.missing_baseline_response_series_uids
                ),
                "excluded_missing_zero_n_response_series_uids": list(
                    zero_classification.missing_baseline_response_series_uids
                ),
                "excluded_equal_threshold_series_count": len(
                    zero_classification.equal_threshold_response_series_uids
                ),
                "excluded_equal_threshold_response_series_uids": list(
                    zero_classification.equal_threshold_response_series_uids
                ),
                "conflicting_zero_n_series_count": len(
                    zero_classification.conflicting_baseline_response_series_uids
                ),
                "conflicting_zero_n_response_series_uids": list(
                    zero_classification.conflicting_baseline_response_series_uids
                ),
                "series_inventory_sha256": series_inventory_sha256,
                "above_high_n_inclusion": {
                    "criterion": f"n_rate_kg_ha > {high_n_threshold_kg_ha:g}",
                    "disposition": "included_with_full_trajectory_in_qualifying_baseline_group",
                    "maximum_n_rate_kg_ha": (
                        max(o["n_rate_kg_ha"] for o in combined_above_high_n)
                        if combined_above_high_n
                        else None
                    ),
                    "observation_count": len(combined_above_high_n),
                    "series_count": len(
                        {o["response_series_uid"] for o in combined_above_high_n}
                    ),
                    "observations": combined_above_high_n,
                },
            }
            zero_release_context = (
                {
                    "source_package_status": run_manifest.get("status"),
                    "source_package_figure_layout": figure_inventory.get(
                        "layout_version"
                    ),
                }
                if governed_release
                else _zero_n_release_metadata(
                    run_manifest=run_manifest,
                    figure_inventory=figure_inventory,
                    release_integrity_block=release_integrity_block,
                )
            )
            zero_manifest_content = {
                "schema_version": (
                    "governed-zero-n-strata-diagnostic-v1"
                    if governed_release
                    else "post-release-zero-n-strata-diagnostic-v1"
                ),
                "status": (
                    "governed_release_artifacts"
                    if governed_release
                    else "post_release_diagnostic_unbound"
                ),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_package": source_package_relative,
                "source_run_id": run_manifest.get("run_id"),
                "source_name": source_name,
                **zero_release_context,
                "classification": zero_classification_block,
                "groups": zero_groups,
                "release_contract_effect": release_contract_effect,
                "input_sha256": recorded_inputs,
            }
            staged_zero_manifest = staging / zero_manifest_name
            staged_zero_manifest.write_text(
                json.dumps(zero_manifest_content, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            staged_zero_checksums = staging / zero_checksums_name
            zero_checksum_files = {
                zero_below_name: staged_zero_below,
                zero_above_name: staged_zero_above,
                zero_manifest_name: staged_zero_manifest,
            }
            staged_zero_checksums.write_text(
                "".join(
                    f"{_sha256(path)}  {name}\n"
                    for name, path in sorted(zero_checksum_files.items())
                ),
                encoding="utf-8",
            )

        plots: dict[str, Any] = {}
        if generate_zero_n_strata:
            assert zero_classification is not None
            assert staged_zero_below is not None and staged_zero_above is not None
            assert zero_below_metadata is not None and zero_above_metadata is not None
            plots[f"zero_n_baseline_below_{zero_threshold_token}"] = {
                "selection_semantics": (
                    f"Whole series with observed yield < {zero_n_yield_threshold_t_ha:g} t/ha "
                    "at exactly 0 kg N/ha; all finite observations retained."
                ),
                "response_series_uids": list(zero_classification.below.response_series_uids),
                "series_count": len(zero_classification.below.response_series_uids),
                "finite_observation_count": zero_classification.below.finite_observation_count,
                "output": {
                    "path": zero_below_name,
                    "sha256": _sha256(staged_zero_below),
                    **zero_below_metadata,
                },
            }
            plots[f"zero_n_baseline_above_{zero_threshold_token}"] = {
                "selection_semantics": (
                    f"Whole series with observed yield > {zero_n_yield_threshold_t_ha:g} t/ha "
                    "at exactly 0 kg N/ha; all finite observations retained."
                ),
                "response_series_uids": list(zero_classification.above.response_series_uids),
                "series_count": len(zero_classification.above.response_series_uids),
                "finite_observation_count": zero_classification.above.finite_observation_count,
                "output": {
                    "path": zero_above_name,
                    "sha256": _sha256(staged_zero_above),
                    **zero_above_metadata,
                },
            }
        plots[_manifest_plot_key(yield_threshold_t_ha)] = {
            "selection_semantics": (
                "Whole series with at least one finite observed grain yield strictly above "
                f"{yield_threshold_t_ha:g} t/ha; all finite observations retained, including "
                f"values at or below the threshold and N rates beyond {high_n_threshold_kg_ha:g} kg/ha."
            ),
            "yield_threshold_t_ha": yield_threshold_t_ha,
            "high_n_threshold_kg_ha": high_n_threshold_kg_ha,
            "qualifying_response_series_uids": list(selection.response_series_uids),
            "series_max_yield_t_ha": {
                series_uid: max(yield_value for _, yield_value in observations_by_series[series_uid])
                for series_uid in selection.response_series_uids
            },
            "series_count": len(selection.response_series_uids),
            "finite_observation_count": len(selection.selected_observations),
            "threshold_exceeding_observation_count": len(
                selection.threshold_exceeding_observations
            ),
            "series_with_max_exactly_threshold_count": len(
                selection.exact_threshold_response_series_uids
            ),
            "series_with_max_exactly_threshold_uids": list(
                selection.exact_threshold_response_series_uids
            ),
            "n_rate_range_kg_ha": list(selection.n_rate_range_kg_ha or ()),
            "above_high_n_observation_count": len(selection.above_high_n_observations),
            "above_high_n_observations": [
                {
                    "response_series_uid": series_uid,
                    "n_rate_kg_ha": n_rate,
                    "yield_t_ha": yield_value,
                }
                for series_uid, n_rate, yield_value in selection.above_high_n_observations
            ],
            "output": {
                "path": staged_image.name,
                "sha256": _sha256(staged_image),
                **new_image_metadata,
            },
        }

        staged_manifest = staging / custom_manifest_name
        custom_manifest = {
            "schema_version": (
                "governed-custom-plots-manifest-v1"
                if governed_release
                else "post-release-custom-plots-manifest-v1"
            ),
            "status": (
                "governed_release_artifacts"
                if governed_release
                else "post_release_diagnostics_unbound"
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_package": source_package_relative,
            "source_run_id": run_manifest.get("run_id"),
            "source_name": source_name,
            "custom_plot_count": len(plots),
            "plots": plots,
            "source_provenance": {
                "analysis_eligibility_ledger_sha256": _sha256(ledger_path),
                "series_inventory_sha256": series_inventory_sha256,
                "current_custom_plot_config_sha256": _sha256(config_path),
                "generator_module_sha256": _sha256(Path(__file__).resolve()),
                "zero_n_strata_manifest_sha256": (
                    _sha256(staged_zero_manifest) if generate_zero_n_strata else None
                ),
                "zero_n_strata_checksum_ledger_sha256": (
                    _sha256(staged_zero_checksums) if generate_zero_n_strata else None
                ),
            },
            **(
                {}
                if governed_release
                else {"original_release_integrity": release_integrity_block}
            ),
            "release_contract_effect": release_contract_effect,
            "input_sha256": recorded_inputs,
            "accountable_human_review": "not_claimed",
        }
        staged_manifest.write_text(
            json.dumps(custom_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged_checksums = staging / custom_checksums_name
        checksum_files: dict[str, Path] = {
            new_image_name: staged_image,
            custom_manifest_name: staged_manifest,
        }
        if generate_zero_n_strata:
            assert staged_zero_below is not None and staged_zero_above is not None
            checksum_files[zero_below_name] = staged_zero_below
            checksum_files[zero_above_name] = staged_zero_above
        staged_checksums.write_text(
            "".join(
                f"{_sha256(path)}  {name}\n"
                for name, path in sorted(checksum_files.items())
            ),
            encoding="utf-8",
        )
        if {path.name for path in staging.iterdir() if path.is_file()} != set(target_names):
            raise RuntimeError("Customized output staging inventory is inconsistent")

        backup_root = staging / "backup"
        backup_root.mkdir()
        existed_before: dict[str, bool] = {}
        for destination in targets:
            existed_before[destination.name] = destination.is_file()
            if destination.is_file():
                shutil.copy2(destination, backup_root / destination.name)
        if superseded_image is not None:
            shutil.copy2(superseded_image, backup_root / superseded_image.name)
        promoted: list[Path] = []
        retired_superseded = False
        try:
            for name in target_names:
                destination = output_root / name
                os.replace(staging / name, destination)
                promoted.append(destination)
            if superseded_image is not None:
                superseded_image.unlink()
                retired_superseded = True
            _verify_bound_files(package, release_entries)
            for name, digest in _parse_checksum_ledger(
                output_root / custom_checksums_name
            ).items():
                if Path(name).name != name or not (output_root / name).is_file():
                    raise RuntimeError(f"Unsafe or missing custom-plot artifact: {name}")
                if _sha256(output_root / name) != digest:
                    raise RuntimeError(f"Custom-plot checksum mismatch after promotion: {name}")
            actual_after = _package_files_without_release_ledger(
                package,
                release_checksums_path,
                ignored=(
                    lock_path,
                    *(path for path in staging.rglob("*") if path.is_file()),
                ),
            )
            expected_extras = {path.relative_to(package).as_posix() for path in targets}
            if actual_after - set(release_entries) != expected_extras:
                raise RuntimeError("Post-generation custom-overlay inventory is inconsistent")
        except Exception:
            if retired_superseded and superseded_image is not None:
                os.replace(backup_root / superseded_image.name, superseded_image)
            for destination in reversed(promoted):
                backup = backup_root / destination.name
                if existed_before[destination.name]:
                    os.replace(backup, destination)
                else:
                    destination.unlink(missing_ok=True)
            raise
        shutil.rmtree(staging)
        staging = None
        result = {
            "status": (
                "generated_governed_release_artifacts"
                if governed_release
                else "generated"
            ),
            "output": plots[_manifest_plot_key(yield_threshold_t_ha)]["output"],
            "series_count": len(selection.response_series_uids),
            "observation_count": len(selection.selected_observations),
            "threshold_exceeding_observation_count": len(
                selection.threshold_exceeding_observations
            ),
            "above_high_n_observation_count": len(selection.above_high_n_observations),
            "retired_superseded_output": (
                superseded_image.name if superseded_image is not None else None
            ),
            "custom_manifest_sha256": _sha256(output_root / custom_manifest_name),
            "custom_checksums_sha256": _sha256(output_root / custom_checksums_name),
            "generated_zero_n_strata": generate_zero_n_strata,
            "zero_n_strata_manifest_sha256": (
                _sha256(output_root / zero_manifest_name) if generate_zero_n_strata else None
            ),
            "zero_n_strata_checksums_sha256": (
                _sha256(output_root / zero_checksums_name) if generate_zero_n_strata else None
            ),
            "physical_package_file_count": sum(
                1
                for path in package.rglob("*")
                if path.is_file() and path.resolve() != lock_path.resolve()
            ),
        }
        return result
    finally:
        _cleanup_generation_resources(
            staging=staging,
            lock_fd=lock_fd,
            lock_path=lock_path,
        )


def _validate_mode_response(run_mode: str) -> dict[str, Any] | None:
    """Return a no-op success payload when the effective run mode is 'validate'.

    Validate-mode runs are config/policy preflight only; the standalone
    generator must not resolve a package path or touch the filesystem in
    that mode. Returns ``None`` for every other mode so the caller falls
    through to normal generation.
    """

    if run_mode != "validate":
        return None
    return {
        "status": "validate_mode_no_op",
        "generated": False,
        "reason": (
            "[run].mode is 'validate'; custom-overlay generation writes "
            "post-release diagnostics only and performs no package "
            "resolution or writes in validate mode"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    module_root = PROJECT_ROOT / "modules"
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))
    try:
        from n_response_curve.config import load_config

        config_path = Path(args.config).expanduser().resolve()
        config = load_config(
            config_path,
            project_root=PROJECT_ROOT,
            check_files=False,
        )
        validate_mode_status = _validate_mode_response(config.run_mode)
        if validate_mode_status is not None:
            print(json.dumps(validate_mode_status, sort_keys=True))
            return 0
        settings = config.raw.get("custom_overlays", {})
        if not settings.get("enabled", False):
            print(json.dumps({"status": "disabled", "generated": False}, sort_keys=True))
            return 0
        result = {
            "status": "managed_by_phase_5",
            "generated": False,
            "reason": (
                "Governed custom overlays are generated atomically inside the "
                "Phase 5 release package; rerun the pipeline to regenerate them."
            ),
        }
    except Exception as exc:
        print(f"custom-overlay-error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
