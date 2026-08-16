#!/usr/bin/env python3
"""Generate signed, restricted source-dataset diagnostic overlays.

This standalone post-pipeline stage writes either a disjoint restricted bundle
or an explicitly configured, separately signed restricted extension beneath an
unchanged governed report core. It reads only configured N/yield/context
columns, uses opaque trajectory identifiers, and never records restricted
source locators or raw row identifiers in its manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.config import ConfigError, load_config  # noqa: E402
from n_response_curve.reporting.source_dataset_overlays import (  # noqa: E402
    SUPPORTED_SOURCE_NAMES,
    read_source_dataset_overlay,
    select_source_dataset_overlay_above_yield_threshold,
    write_source_dataset_overlay_figure,
)

MANIFEST_NAME = "source_dataset_overlays_manifest.json"
CHECKSUMS_NAME = "SOURCE_DATASET_OVERLAYS_CHECKSUMS.sha256"
SCHEMA_VERSION = "restricted-source-dataset-overlays-v4"
_FLAT_LAYOUT_SCHEMA_VERSION = "restricted-source-dataset-overlays-v3"
_PREVIOUS_SCHEMA_VERSION = "restricted-source-dataset-overlays-v2"
_LEGACY_SCHEMA_VERSION = "restricted-source-dataset-overlays-v1"
BUNDLE_STATUS = "restricted_internal_diagnostic_not_for_release"
_RESULT_STATUS = "restricted_internal_diagnostics_generated"
RESTRICTED_EXTENSION_DIRECTORY = "restricted_diagnostics"
RESTRICTED_EXTENSION_BUNDLE_DIRECTORY = "source_dataset_overlays"
RESTRICTED_EXTENSION_MANIFEST_NAME = "restricted_extension_manifest.json"
RESTRICTED_EXTENSION_CHECKSUMS_NAME = "RESTRICTED_EXTENSION_CHECKSUMS.sha256"
RESTRICTED_EXTENSION_SCHEMA_VERSION = "mixed-report-restricted-extension-v1"
RESTRICTED_EXTENSION_STATUS = "mixed_restricted_report_container_not_for_release"
_SAFE_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_SOURCE_NAME = re.compile(r"[a-z][a-z0-9_]*")
_SAFE_UTC_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z"
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")

_SOURCE_WIDE_OUTPUT_FILENAMES = {
    "core_trial_data": "core_trial_data_source_wide_internal_diagnostic.jpeg",
    "ltcce": "ltcce_source_wide_internal_diagnostic.jpeg",
    "ph_combined_nopt_rcm": "ph_combined_nopt_rcm_source_wide_internal_diagnostic.jpeg",
}
_PLOT_KINDS = {
    "core_trial_data": "source_observation_inventory_scatter",
    "ltcce": "response_series_overlay",
    "ph_combined_nopt_rcm": "linked_arm_trajectory_overlay",
}
_TREATMENT_CLASSES = {
    "core_trial_data": ["mineral N rate", "zero N"],
    "ltcce": ["mineral N rate", "zero N"],
    "ph_combined_nopt_rcm": ["FP", "NOPT NPK", "RCM", "zero N"],
}
_SOURCE_CAVEATS = {
    "core_trial_data": (
        "Registered raw core-trial extract; pipeline curation and response-series resolution are not applied.",
        "Every finite source row is shown as an independent observation.",
        "No response curve or connected trajectory is reconstructed from source-row order.",
        "Rows with nonfinite N or yield remain counted as explicit exclusions in the signed manifest.",
    ),
    "ltcce": (
        "Registered raw source extract; pipeline curation and response-series resolution are not applied.",
        "Unresolved row multiplicity is retained without deduplication or replicate aggregation.",
        "Rep is retained in the visual trajectory key; absent water, region, and province are not inferred.",
        "Authorized IRRI research use only; confidentiality and prior approval before sharing or publishing remain required.",
    ),
    "ph_combined_nopt_rcm": (
        "Registered raw linked-arm extract; pipeline curation and response-series resolution are not applied.",
        "Same-row storage does not establish experimental-unit or cross-arm comparability.",
        "Within-row connectors are visual aids only; zero-N P/K composition remains unresolved and no response curve or fit is asserted.",
    ),
}

_SELECTION_UNITS = {
    "core_trial_data": "source_row",
    "ltcce": "replicate_specific_trajectory",
    "ph_combined_nopt_rcm": "linked_source_row",
}

_SOURCE_DATA_CLASSIFICATIONS = {
    "core_trial_data": "internal",
    "ltcce": "restricted",
    "ph_combined_nopt_rcm": "restricted",
}

_POTENTIAL_OBSERVATIONS_PER_SOURCE_ROW = {
    "core_trial_data": 1,
    "ltcce": 1,
    "ph_combined_nopt_rcm": 4,
}

_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "generated_at_utc",
        "disclosure_class",
        "governed_release_relationship",
        "governed_figure_inventory_affected",
        "accountable_human_review",
        "yield_threshold_t_ha",
        "source_count",
        "sources",
        "generator",
    }
)
_GENERATOR_KEYS = frozenset(
    {
        "module_sha256",
        "renderer_sha256",
        "config_sha256",
        "atomic_stage_and_replace",
        "source_bytes_reverified_unchanged",
    }
)
_SOURCE_KEYS = frozenset(
    {
        "source_name",
        "source_artifact_token",
        "source_sha256",
        "data_classification",
        "disclosure_class",
        "plot_kind",
        "source_wide_selection_semantics",
        "source_row_count",
        "finite_observation_count",
        "excluded_nonfinite_observation_count",
        "trajectory_count",
        "treatment_classes",
        "n_rate_range_kg_ha",
        "yield_range_t_ha",
        "curation_status",
        "series_resolution_applied",
        "duplicate_adjudication_applied",
        "linked_arm_comparability_assumed",
        "unresolved_provenance_caveats",
        "yield_threshold_selection",
        "outputs",
    }
)
_V2_SELECTION_KEYS = frozenset(
    {
        "operator",
        "threshold_t_ha",
        "selection_unit",
        "retention_rule",
        "selected_unit_count",
        "total_unit_count",
        "selected_observation_count",
        "total_observation_count",
        "threshold_exceeding_observation_count",
        "selected_treatment_classes",
        "selected_n_rate_range_kg_ha",
        "selected_yield_range_t_ha",
    }
)
_SELECTION_KEYS = _V2_SELECTION_KEYS | frozenset(
    {
        "threshold_equal_observation_count",
        "threshold_equal_only_unit_count",
        "excluded_nonfinite_observation_count_within_selected_units",
    }
)
_OUTPUT_KEYS = frozenset({"selection_scope", "path", "sha256", "image"})
_IMAGE_KEYS = frozenset({"format", "width", "height", "mode"})
_SOURCE_VERIFICATION_CONTRACT_KEYS = frozenset(
    {
        "source_sha256",
        "source_row_count",
        "finite_observation_count",
        "excluded_nonfinite_observation_count",
        "trajectory_count",
        "treatment_classes",
        "n_rate_range_kg_ha",
        "yield_range_t_ha",
        "yield_threshold_selection",
        "outputs",
    }
)

_RESTRICTED_EXTENSION_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "generated_at_utc",
        "disclosure_class",
        "accountable_human_review",
        "governed_core",
        "extension",
    }
)
_RESTRICTED_EXTENSION_CORE_KEYS = frozenset(
    {
        "status",
        "run_manifest_sha256",
        "checksum_ledger_sha256",
        "bound_artifact_count",
        "strict_core_verified_before_extension",
    }
)
_RESTRICTED_EXTENSION_BUNDLE_KEYS = frozenset(
    {
        "relative_root",
        "schema_version",
        "manifest_sha256",
        "checksum_ledger_sha256",
        "bundle_file_count",
    }
)


def _threshold_filename_token(threshold_t_ha: float) -> str:
    return format(float(threshold_t_ha), ".15g").replace(".", "_").replace("+", "")


def _output_filenames(source_name: str, threshold_t_ha: float) -> dict[str, str]:
    threshold_token = _threshold_filename_token(threshold_t_ha)
    return {
        "source_wide": _SOURCE_WIDE_OUTPUT_FILENAMES[source_name],
        "yield_threshold_selected": (
            f"{source_name}_series_with_yield_above_{threshold_token}_t_ha_"
            "internal_diagnostic.jpeg"
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_hashes() -> dict[str, str]:
    return {
        "module_sha256": _sha256_file(Path(__file__)),
        "renderer_sha256": _sha256_file(
            Path(__file__).with_name("source_dataset_overlays.py")
        ),
    }


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    return any(candidate.is_symlink() for candidate in (absolute, *absolute.parents))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _safe_relative_path(value: str) -> bool:
    path = Path(value)
    return (
        bool(value)
        and not path.is_absolute()
        and not _WINDOWS_ABSOLUTE_PATH.match(value)
        and "\\" not in value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _bundle_actual_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(
                f"Source-dataset overlay bundle contains a prohibited symlink: "
                f"{path.relative_to(root).as_posix()}"
            )
        if path.is_file():
            paths.add(path.relative_to(root).as_posix())
    return paths


def _parse_checksum_ledger(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Source-dataset overlay checksum ledger is missing or unsafe")
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("Source-dataset overlay checksum ledger is unreadable") from exc
    for line in lines:
        if not line.strip() or "  " not in line:
            raise RuntimeError("Source-dataset overlay checksum ledger contains an invalid line")
        digest, relative = line.split("  ", 1)
        if (
            not _SAFE_SHA256.fullmatch(digest)
            or not _safe_relative_path(relative)
            or relative in entries
            or relative == CHECKSUMS_NAME
        ):
            raise RuntimeError("Source-dataset overlay checksum ledger contains an unsafe entry")
        entries[relative] = digest
    if not entries:
        raise RuntimeError("Source-dataset overlay checksum ledger is empty")
    return entries


def _contains_restricted_locator(payload: Any, *, key: str = "") -> bool:
    normalized_key = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    forbidden_keys = {
        "source_path",
        "data_path",
        "input_path",
        "source_locator",
        "farmer_first_name",
        "farmer_last_name",
        "farmer_id",
        "record_id",
        "row_id",
        "participant_id",
        "latitude",
        "longitude",
        "coordinates",
        "email",
        "phone",
        "address",
    }
    if normalized_key in forbidden_keys or normalized_key.startswith("farmer_"):
        return True
    if isinstance(payload, Mapping):
        return any(
            _contains_restricted_locator(value, key=str(nested_key))
            for nested_key, value in payload.items()
        )
    if isinstance(payload, (tuple, list)):
        return any(_contains_restricted_locator(value, key=key) for value in payload)
    if isinstance(payload, str):
        stripped = payload.strip()
        return (
            "wf/01_curated_dataset" in stripped.casefold().replace("\\", "/")
            or _WINDOWS_ABSOLUTE_PATH.match(stripped) is not None
            or stripped.startswith(("/", "~/", "\\\\", "//"))
            or stripped.casefold().startswith("file:")
        )
    return False


def _read_image_metadata(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.load()
            metadata = {
                "format": str(image.format or "").casefold(),
                "width": int(image.width),
                "height": int(image.height),
                "mode": str(image.mode),
            }
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Source-dataset overlay image failed read-back: {path.name}") from exc
    if metadata != {"format": "jpeg", "width": 1500, "height": 1050, "mode": "RGB"}:
        raise RuntimeError(
            f"Source-dataset overlay image metadata is invalid for {path.name}: {metadata}"
        )
    return metadata


def _is_positive_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _is_nonnegative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _require_exact_keys(
    payload: Any,
    expected: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise RuntimeError(f"Source-dataset overlay {label} schema is not closed")
    return payload


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"Duplicate JSON key is prohibited: {key!r}")
        payload[key] = value
    return payload


def _is_finite_range(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in value
        )
        and float(value[0]) <= float(value[1])
    )


def _is_nonempty_unique_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item) for item in value)
        and len(value) == len(set(value))
    )


def _verify_source_dataset_overlay_bundle(
    output_root: str | Path,
    *,
    allowed_schema_versions: frozenset[str],
    expected_config_sha256: str | None,
    expected_source_sha256: Mapping[str, str] | None,
    expected_source_contracts: Mapping[str, Mapping[str, Any]] | None,
    require_current_bindings: bool,
) -> dict[str, Any]:
    """Verify one exact bundle under an explicitly selected schema policy."""

    lexical_root = Path(output_root).absolute()
    if _has_symlink_component(lexical_root):
        raise RuntimeError("Source-dataset overlay bundle root contains a prohibited symlink")
    root = lexical_root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"Source-dataset overlay bundle does not exist: {root}")
    actual = _bundle_actual_paths(root)
    checksum_path = root / CHECKSUMS_NAME
    manifest_path = root / MANIFEST_NAME
    entries = _parse_checksum_ledger(checksum_path)
    expected_actual = set(entries) | {CHECKSUMS_NAME}
    if actual != expected_actual:
        raise RuntimeError(
            "Source-dataset overlay checksum ledger does not cover the complete bundle"
        )
    if MANIFEST_NAME not in entries or manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("Source-dataset overlay manifest is missing or unsafe")
    for relative, expected_digest in entries.items():
        artifact = root / relative
        if not artifact.is_file() or artifact.is_symlink():
            raise RuntimeError(f"Source-dataset overlay artifact is missing or unsafe: {relative}")
        if _sha256_file(artifact) != expected_digest:
            raise RuntimeError(f"Source-dataset overlay checksum mismatch: {relative}")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError("Source-dataset overlay manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("Source-dataset overlay manifest must be a JSON object")
    _require_exact_keys(manifest, _MANIFEST_KEYS, label="manifest")
    schema_version = manifest.get("schema_version")
    if (
        schema_version not in allowed_schema_versions
        or manifest.get("status") != BUNDLE_STATUS
        or manifest.get("disclosure_class") != "internal_restricted_diagnostic_not_for_release"
        or manifest.get("governed_release_relationship") != "none"
        or manifest.get("governed_figure_inventory_affected") is not False
        or manifest.get("accountable_human_review") != "not_claimed"
    ):
        raise RuntimeError("Source-dataset overlay manifest schema or disclosure status is invalid")
    generated_at_utc = manifest.get("generated_at_utc")
    if (
        not isinstance(generated_at_utc, str)
        or _SAFE_UTC_TIMESTAMP.fullmatch(generated_at_utc) is None
    ):
        raise RuntimeError("Source-dataset overlay generation timestamp is invalid")
    if _contains_restricted_locator(manifest):
        raise RuntimeError("Source-dataset overlay manifest discloses a restricted source locator")
    generator = _require_exact_keys(
        manifest.get("generator"),
        _GENERATOR_KEYS,
        label="generator",
    )
    for key in ("module_sha256", "renderer_sha256", "config_sha256"):
        digest = generator.get(key)
        if not isinstance(digest, str) or not _SAFE_SHA256.fullmatch(digest):
            raise RuntimeError(f"Source-dataset overlay generator {key} is invalid")
    atomic_stage_and_replace = generator.get("atomic_stage_and_replace")
    if (
        not isinstance(atomic_stage_and_replace, bool)
        or generator.get("source_bytes_reverified_unchanged") is not True
    ):
        raise RuntimeError("Source-dataset overlay generator assurances are invalid")
    if require_current_bindings and atomic_stage_and_replace is not False:
        raise RuntimeError(
            "Current source-dataset overlay manifest must not claim interruption-atomic replacement"
        )
    if (
        expected_config_sha256 is not None
        and generator.get("config_sha256") != expected_config_sha256
    ):
        raise RuntimeError("Source-dataset overlay config hash does not match the expected config")
    if require_current_bindings and (
        generator.get("module_sha256") != _sha256_file(Path(__file__))
        or generator.get("renderer_sha256")
        != _sha256_file(Path(__file__).with_name("source_dataset_overlays.py"))
    ):
        raise RuntimeError(
            "Source-dataset overlay generator or renderer hash is not current"
        )
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not sources or manifest.get("source_count") != len(sources):
        raise RuntimeError("Source-dataset overlay manifest source inventory is invalid")
    if expected_source_sha256 is not None and (
        set(expected_source_sha256) != set(sources)
        or any(
            not isinstance(digest, str) or _SAFE_SHA256.fullmatch(digest) is None
            for digest in expected_source_sha256.values()
        )
    ):
        raise RuntimeError("Expected source-dataset overlay source hashes are invalid")
    if expected_source_contracts is not None:
        if set(expected_source_contracts) != set(sources):
            raise RuntimeError("Expected source-dataset overlay contracts are incomplete")
        for source_name, contract in expected_source_contracts.items():
            _require_exact_keys(
                contract,
                _SOURCE_VERIFICATION_CONTRACT_KEYS,
                label=f"expected source contract {source_name!r}",
            )
    threshold_value = manifest.get("yield_threshold_t_ha")
    if schema_version in {
        SCHEMA_VERSION,
        _FLAT_LAYOUT_SCHEMA_VERSION,
        _PREVIOUS_SCHEMA_VERSION,
    }:
        if not _is_positive_finite_number(threshold_value):
            raise RuntimeError("Source-dataset overlay yield threshold is invalid")
        threshold_t_ha = float(cast(float, threshold_value))
    else:
        raise RuntimeError("Source-dataset overlay manifest schema is unsupported")
    output_paths: set[str] = set()
    for source_name, source in sources.items():
        if (
            source_name not in SUPPORTED_SOURCE_NAMES
            or not _SAFE_SOURCE_NAME.fullmatch(source_name)
            or not isinstance(source, dict)
        ):
            raise RuntimeError(
                f"Source-dataset overlay manifest entry is invalid for {source_name!r}"
            )
        source = _require_exact_keys(
            source,
            _SOURCE_KEYS,
            label=f"source {source_name!r}",
        )
        if (
            source.get("source_name") != source_name
            or source.get("data_classification")
            != _SOURCE_DATA_CLASSIFICATIONS[source_name]
            or source.get("disclosure_class") != "internal_diagnostic_not_for_release"
            or source.get("plot_kind") != _PLOT_KINDS[source_name]
            or source.get("source_wide_selection_semantics")
            != "all finite source N-yield observations; no pooled response or fit"
            or source.get("curation_status") != "raw_registered_source_extract_not_analysis_ready"
            or source.get("series_resolution_applied") is not False
            or source.get("duplicate_adjudication_applied") is not False
            or source.get("linked_arm_comparability_assumed") is not False
            or source.get("unresolved_provenance_caveats")
            != list(_SOURCE_CAVEATS[source_name])
        ):
            raise RuntimeError(
                f"Source-dataset overlay manifest entry is invalid for {source_name!r}"
            )
        source_sha256 = source.get("source_sha256")
        if not isinstance(source_sha256, str) or not _SAFE_SHA256.fullmatch(source_sha256):
            raise RuntimeError(
                f"Source-dataset overlay source hash is invalid for {source_name!r}"
            )
        expected_artifact_token = "restricted_artifact_" + hashlib.sha256(
            f"{source_name}\0{source_sha256}".encode("utf-8")
        ).hexdigest()[:24]
        if (
            source.get("source_artifact_token") != expected_artifact_token
            or (
                expected_source_sha256 is not None
                and source_sha256 != expected_source_sha256[source_name]
            )
        ):
            raise RuntimeError(
                f"Source-dataset overlay source hash binding is invalid for {source_name!r}"
            )
        source_row_count = source.get("source_row_count")
        finite_observation_count = source.get("finite_observation_count")
        excluded_observation_count = source.get("excluded_nonfinite_observation_count")
        trajectory_count = source.get("trajectory_count")
        if (
            not isinstance(source.get("source_artifact_token"), str)
            or re.fullmatch(
                r"restricted_artifact_[0-9a-f]{24}",
                source["source_artifact_token"],
            )
            is None
            or not _is_nonnegative_int(source_row_count)
            or source_row_count == 0
            or not _is_nonnegative_int(finite_observation_count)
            or finite_observation_count == 0
            or not _is_nonnegative_int(excluded_observation_count)
            or not _is_nonnegative_int(trajectory_count)
        ):
            raise RuntimeError(
                f"Source-dataset overlay source counts or ranges are invalid for {source_name!r}"
            )
        source_row_count = int(cast(int, source_row_count))
        finite_observation_count = int(cast(int, finite_observation_count))
        excluded_observation_count = int(cast(int, excluded_observation_count))
        trajectory_count = int(cast(int, trajectory_count))
        expected_potential_observations = (
            source_row_count * _POTENTIAL_OBSERVATIONS_PER_SOURCE_ROW[source_name]
        )
        if (
            trajectory_count == 0
            or trajectory_count > finite_observation_count
            or finite_observation_count + excluded_observation_count
            != expected_potential_observations
            or source.get("treatment_classes") != _TREATMENT_CLASSES[source_name]
            or not _is_finite_range(source.get("n_rate_range_kg_ha"))
            or not _is_finite_range(source.get("yield_range_t_ha"))
        ):
            raise RuntimeError(
                f"Source-dataset overlay source counts or ranges are invalid for {source_name!r}"
            )

        expected_selection_keys = (
            _V2_SELECTION_KEYS
            if schema_version == _PREVIOUS_SCHEMA_VERSION
            else _SELECTION_KEYS
        )
        selection = _require_exact_keys(
            source.get("yield_threshold_selection"),
            expected_selection_keys,
            label=f"selection {source_name!r}",
        )
        total_unit_count = trajectory_count
        total_observation_count = finite_observation_count
        if (
            selection.get("operator") != ">"
            or selection.get("threshold_t_ha") != threshold_t_ha
            or selection.get("selection_unit") != _SELECTION_UNITS[source_name]
            or selection.get("retention_rule")
            != "retain every finite observation within each qualifying whole unit"
            or not _is_nonempty_unique_string_list(
                selection.get("selected_treatment_classes")
            )
            or not set(selection["selected_treatment_classes"]).issubset(
                source["treatment_classes"]
            )
            or not _is_finite_range(selection.get("selected_n_rate_range_kg_ha"))
            or not _is_finite_range(selection.get("selected_yield_range_t_ha"))
        ):
            raise RuntimeError(
                f"Source-dataset overlay selection metadata is invalid for {source_name!r}"
            )
        selected_unit_value = selection.get("selected_unit_count")
        selected_observation_value = selection.get("selected_observation_count")
        threshold_exceeding_value = selection.get("threshold_exceeding_observation_count")
        if (
            not _is_nonnegative_int(selected_unit_value)
            or not _is_nonnegative_int(selected_observation_value)
            or not _is_nonnegative_int(threshold_exceeding_value)
            or selection.get("total_unit_count") != total_unit_count
            or selection.get("total_observation_count") != total_observation_count
        ):
            raise RuntimeError(
                f"Source-dataset overlay selection counts are invalid for {source_name!r}"
            )
        selected_unit_count = int(cast(int, selected_unit_value))
        selected_observation_count = int(cast(int, selected_observation_value))
        threshold_exceeding_count = int(cast(int, threshold_exceeding_value))
        if schema_version in {SCHEMA_VERSION, _FLAT_LAYOUT_SCHEMA_VERSION}:
            threshold_equal_value = selection.get("threshold_equal_observation_count")
            threshold_equal_only_unit_value = selection.get("threshold_equal_only_unit_count")
            selected_nonfinite_value = selection.get(
                "excluded_nonfinite_observation_count_within_selected_units"
            )
            if (
                not _is_nonnegative_int(threshold_equal_value)
                or not _is_nonnegative_int(threshold_equal_only_unit_value)
                or not _is_nonnegative_int(selected_nonfinite_value)
            ):
                raise RuntimeError(
                    f"Source-dataset overlay strict-threshold audit counts are invalid for {source_name!r}"
                )
            threshold_equal_count = int(cast(int, threshold_equal_value))
            threshold_equal_only_unit_count = int(
                cast(int, threshold_equal_only_unit_value)
            )
            selected_nonfinite_count = int(cast(int, selected_nonfinite_value))
        else:
            threshold_equal_count = 0
            threshold_equal_only_unit_count = 0
            selected_nonfinite_count = 0
        if (
            selected_unit_count == 0
            or selected_unit_count > total_unit_count
            or selected_unit_count > selected_observation_count
            or threshold_exceeding_count == 0
            or selected_unit_count > threshold_exceeding_count
            or threshold_exceeding_count > selected_observation_count
            or selected_observation_count > total_observation_count
            or threshold_equal_count > total_observation_count
            or threshold_equal_only_unit_count > threshold_equal_count
            or selected_unit_count + threshold_equal_only_unit_count > total_unit_count
            or selected_nonfinite_count > excluded_observation_count
        ):
            raise RuntimeError(
                f"Source-dataset overlay selection counts are invalid for {source_name!r}"
            )
        if expected_source_contracts is not None:
            actual_contract = {
                key: source[key] for key in _SOURCE_VERIFICATION_CONTRACT_KEYS
            }
            if actual_contract != dict(expected_source_contracts[source_name]):
                raise RuntimeError(
                    f"Source-dataset overlay source-derived contract mismatch for {source_name!r}"
                )
        outputs = source.get("outputs")
        expected_filenames = _output_filenames(source_name, threshold_t_ha)
        if not isinstance(outputs, dict) or set(outputs) != set(expected_filenames):
            raise RuntimeError(
                f"Source-dataset overlay output inventory is invalid for {source_name!r}"
            )
        for role, filename in expected_filenames.items():
            output = _require_exact_keys(
                outputs[role],
                _OUTPUT_KEYS,
                label=f"output {source_name!r}/{role}",
            )
            image = _require_exact_keys(
                output.get("image"),
                _IMAGE_KEYS,
                label=f"image {source_name!r}/{role}",
            )
            relative = output.get("path")
            expected_scope = (
                "all_finite_source_observations"
                if role == "source_wide"
                else "whole_units_with_any_observed_yield_strictly_above_threshold"
            )
            expected_relative = (
                f"figures/overlay/{source_name}/{filename}"
                if schema_version == SCHEMA_VERSION
                else f"figures/overlay/{filename}"
            )
            if (
                output.get("selection_scope") != expected_scope
                or not isinstance(relative, str)
                or relative != expected_relative
                or relative not in entries
                or output.get("sha256") != entries[relative]
                or not isinstance(output.get("sha256"), str)
                or _SAFE_SHA256.fullmatch(output["sha256"]) is None
                or image != _read_image_metadata(root / relative)
            ):
                raise RuntimeError(
                    f"Source-dataset overlay output does not reconcile for "
                    f"{source_name!r}/{role}"
                )
            output_paths.add(relative)
    if output_paths != set(entries) - {MANIFEST_NAME}:
        raise RuntimeError(
            "Source-dataset overlay manifest outputs do not cover their checksum ledger"
        )
    return manifest


def verify_source_dataset_overlay_bundle(
    output_root: str | Path,
    *,
    expected_config_sha256: str,
    expected_source_sha256: Mapping[str, str],
    expected_source_contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify one current organized bundle against config and implementation bytes."""

    if _SAFE_SHA256.fullmatch(expected_config_sha256) is None:
        raise RuntimeError("Expected source-dataset overlay config hash is invalid")
    if not isinstance(expected_source_sha256, Mapping) or not expected_source_sha256:
        raise RuntimeError("Expected source-dataset overlay source hashes are required")
    if not isinstance(expected_source_contracts, Mapping) or not expected_source_contracts:
        raise RuntimeError("Expected source-dataset overlay source contracts are required")
    return _verify_source_dataset_overlay_bundle(
        output_root,
        allowed_schema_versions=frozenset({SCHEMA_VERSION}),
        expected_config_sha256=expected_config_sha256,
        expected_source_sha256=expected_source_sha256,
        expected_source_contracts=expected_source_contracts,
        require_current_bindings=True,
    )


def _verify_existing_bundle_for_replacement(output_root: str | Path) -> dict[str, Any]:
    """Recognize a closed v2/v3/v4 bundle for controlled replacement."""

    return _verify_source_dataset_overlay_bundle(
        output_root,
        allowed_schema_versions=frozenset(
            {_PREVIOUS_SCHEMA_VERSION, _FLAT_LAYOUT_SCHEMA_VERSION, SCHEMA_VERSION}
        ),
        expected_config_sha256=None,
        expected_source_sha256=None,
        expected_source_contracts=None,
        require_current_bindings=False,
    )


def _validate_source_specs(
    *,
    project_root: Path,
    source_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not source_specs:
        raise RuntimeError("At least one source dataset overlay must be selected")
    normalized: dict[str, dict[str, Any]] = {}
    for source_name, spec in source_specs.items():
        if source_name not in SUPPORTED_SOURCE_NAMES or not isinstance(spec, Mapping):
            raise RuntimeError(f"Unsupported source dataset overlay: {source_name!r}")
        lexical_source_path = Path(str(spec.get("path", "")))
        if not lexical_source_path.is_absolute():
            lexical_source_path = project_root / lexical_source_path
        lexical_source_path = lexical_source_path.absolute()
        if _has_symlink_component(lexical_source_path):
            raise RuntimeError(
                f"Configured source contains a prohibited symlink: {source_name}"
            )
        source_path = lexical_source_path.resolve()
        if (
            not source_path.is_relative_to(project_root)
            or not source_path.is_file()
        ):
            raise RuntimeError(f"Configured source is missing or unsafe: {source_name}")
        encoding = spec.get("encoding")
        if not isinstance(encoding, str) or not encoding.strip():
            raise RuntimeError(f"Configured source encoding is invalid: {source_name}")
        expected_classification = _SOURCE_DATA_CLASSIFICATIONS[source_name]
        if spec.get("data_classification") != expected_classification:
            raise RuntimeError(
                f"Source-dataset overlay classification is invalid: {source_name}"
            )
        normalized[source_name] = {
            "path": source_path,
            "encoding": encoding,
            "data_classification": expected_classification,
        }
    return normalized


def _source_manifest_entry(
    *,
    source_name: str,
    source_sha256: str,
    overlay: Any,
    selected_overlay: Any,
    outputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    summary = overlay.summary
    selection = selected_overlay.selection
    if selection is None:
        raise RuntimeError("Threshold-selected overlay is missing its selection contract")
    return {
        "source_name": source_name,
        "source_artifact_token": "restricted_artifact_"
        + hashlib.sha256(f"{source_name}\0{source_sha256}".encode("utf-8")).hexdigest()[:24],
        "source_sha256": source_sha256,
        "data_classification": _SOURCE_DATA_CLASSIFICATIONS[source_name],
        "disclosure_class": "internal_diagnostic_not_for_release",
        "plot_kind": _PLOT_KINDS[source_name],
        "source_wide_selection_semantics": (
            "all finite source N-yield observations; no pooled response or fit"
        ),
        "source_row_count": summary.source_rows,
        "finite_observation_count": summary.finite_observation_count,
        "excluded_nonfinite_observation_count": summary.excluded_observation_count,
        "trajectory_count": summary.trajectory_count,
        "treatment_classes": list(summary.treatment_classes),
        "n_rate_range_kg_ha": list(summary.n_rate_range_kg_ha or ()),
        "yield_range_t_ha": list(summary.yield_range_t_ha or ()),
        "curation_status": "raw_registered_source_extract_not_analysis_ready",
        "series_resolution_applied": False,
        "duplicate_adjudication_applied": False,
        "linked_arm_comparability_assumed": False,
        "unresolved_provenance_caveats": list(_SOURCE_CAVEATS[source_name]),
        "yield_threshold_selection": {
            "operator": selection.operator,
            "threshold_t_ha": selection.threshold_t_ha,
            "selection_unit": selection.selection_unit,
            "retention_rule": (
                "retain every finite observation within each qualifying whole unit"
            ),
            "selected_unit_count": selection.selected_unit_count,
            "total_unit_count": selection.total_unit_count,
            "selected_observation_count": selection.selected_observation_count,
            "total_observation_count": selection.total_observation_count,
            "threshold_exceeding_observation_count": selection.threshold_exceeding_observation_count,
            "threshold_equal_observation_count": selection.threshold_equal_observation_count,
            "threshold_equal_only_unit_count": selection.threshold_equal_only_unit_count,
            "excluded_nonfinite_observation_count_within_selected_units": (
                selection.excluded_nonfinite_observation_count_within_selected_units
            ),
            "selected_treatment_classes": list(
                selected_overlay.summary.treatment_classes
            ),
            "selected_n_rate_range_kg_ha": list(
                selected_overlay.summary.n_rate_range_kg_ha or ()
            ),
            "selected_yield_range_t_ha": list(
                selected_overlay.summary.yield_range_t_ha or ()
            ),
        },
        "outputs": {role: dict(metadata) for role, metadata in outputs.items()},
    }


def _source_verification_contract(
    *,
    source_name: str,
    source_sha256: str,
    overlay: Any,
    selected_overlay: Any,
    outputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    entry = _source_manifest_entry(
        source_name=source_name,
        source_sha256=source_sha256,
        overlay=overlay,
        selected_overlay=selected_overlay,
        outputs=outputs,
    )
    return {key: entry[key] for key in _SOURCE_VERIFICATION_CONTRACT_KEYS}


def build_source_dataset_overlay_verification_contracts(
    *,
    project_root: str | Path,
    source_specs: Mapping[str, Mapping[str, Any]],
    yield_threshold_t_ha: float,
) -> dict[str, dict[str, Any]]:
    """Recompute exact source, selection, and rendered-artifact contracts."""

    if not _is_positive_finite_number(yield_threshold_t_ha):
        raise RuntimeError("Source-dataset overlay yield threshold must be positive and finite")
    threshold_t_ha = float(yield_threshold_t_ha)
    specs = _validate_source_specs(
        project_root=Path(project_root).resolve(),
        source_specs=source_specs,
    )
    contracts: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="source-dataset-overlay-verification-") as temp_dir:
        render_root = Path(temp_dir)
        for source_name in sorted(specs):
            spec = specs[source_name]
            source_sha256 = _sha256_file(spec["path"])
            overlay = read_source_dataset_overlay(
                spec["path"],
                source_name,
                encoding=spec["encoding"],
            )
            selected_overlay = select_source_dataset_overlay_above_yield_threshold(
                overlay,
                threshold_t_ha=threshold_t_ha,
            )
            outputs: dict[str, dict[str, Any]] = {}
            overlays_by_role = {
                "source_wide": overlay,
                "yield_threshold_selected": selected_overlay,
            }
            for role, filename in _output_filenames(
                source_name,
                threshold_t_ha,
            ).items():
                output_relative = f"figures/overlay/{source_name}/{filename}"
                rendered_path = render_root / source_name / filename
                write_source_dataset_overlay_figure(
                    overlays_by_role[role],
                    rendered_path,
                )
                outputs[role] = {
                    "selection_scope": (
                        "all_finite_source_observations"
                        if role == "source_wide"
                        else "whole_units_with_any_observed_yield_strictly_above_threshold"
                    ),
                    "path": output_relative,
                    "sha256": _sha256_file(rendered_path),
                    "image": _read_image_metadata(rendered_path),
                }
            if _sha256_file(spec["path"]) != source_sha256:
                raise RuntimeError(
                    f"Restricted source changed while deriving verification contract: {source_name}"
                )
            contracts[source_name] = _source_verification_contract(
                source_name=source_name,
                source_sha256=source_sha256,
                overlay=overlay,
                selected_overlay=selected_overlay,
                outputs=outputs,
            )
    return contracts


def _promote_stage(
    *,
    stage: Path,
    output_root: Path,
    replace_existing: bool,
    config_sha256: str,
    config_path: Path | None,
    source_sha256: Mapping[str, str],
    source_contracts: Mapping[str, Mapping[str, Any]],
) -> None:
    backup = output_root.with_name(
        f".{output_root.name}.source-dataset-overlays-backup-{uuid.uuid4().hex[:8]}"
    )
    prior_moved = False
    promoted = False
    try:
        if output_root.exists():
            if not replace_existing:
                raise RuntimeError(
                    f"Source-dataset overlay output already exists: {output_root}"
                )
            try:
                _verify_existing_bundle_for_replacement(output_root)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Existing source-dataset overlay output is not a verified "
                    "closed-schema v2/v3/v4 bundle"
                ) from exc
            os.replace(output_root, backup)
            prior_moved = True
        os.replace(stage, output_root)
        promoted = True
        if config_path is not None and _sha256_file(config_path) != config_sha256:
            raise RuntimeError("Configuration changed during source-dataset overlay generation")
        verify_source_dataset_overlay_bundle(
            output_root,
            expected_config_sha256=config_sha256,
            expected_source_sha256=source_sha256,
            expected_source_contracts=source_contracts,
        )
    except Exception:
        if promoted and output_root.exists():
            shutil.rmtree(output_root)
        if prior_moved and backup.exists():
            os.replace(backup, output_root)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def generate_source_dataset_overlays(
    *,
    project_root: str | Path,
    output_root: str | Path,
    source_specs: Mapping[str, Mapping[str, Any]],
    replace_existing: bool,
    yield_threshold_t_ha: float,
    config_sha256: str,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate, verify, and promote one exact staged diagnostic bundle."""

    project = Path(project_root).resolve()
    lexical_destination = Path(output_root)
    if not lexical_destination.is_absolute():
        lexical_destination = project / lexical_destination
    lexical_destination = lexical_destination.absolute()
    if _has_symlink_component(lexical_destination):
        raise RuntimeError("Source-dataset overlay output root contains a prohibited symlink")
    destination = lexical_destination.resolve()
    if (
        destination == project
        or not destination.is_relative_to(project)
    ):
        raise RuntimeError("Source-dataset overlay output root is unsafe")
    if _SAFE_SHA256.fullmatch(config_sha256) is None:
        raise RuntimeError("Source-dataset overlay config hash must be a SHA-256 digest")
    verified_config_path = Path(config_path).resolve() if config_path is not None else None
    if (
        verified_config_path is not None
        and _sha256_file(verified_config_path) != config_sha256
    ):
        raise RuntimeError("Configuration changed before source-dataset overlay generation")
    if not _is_positive_finite_number(yield_threshold_t_ha):
        raise RuntimeError("Source-dataset overlay yield threshold must be positive and finite")
    threshold_t_ha = float(yield_threshold_t_ha)
    specs = _validate_source_specs(project_root=project, source_specs=source_specs)
    initial_hashes = {
        source_name: _sha256_file(spec["path"])
        for source_name, spec in specs.items()
    }
    initial_implementation_hashes = _implementation_hashes()

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.source-dataset-overlays.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise RuntimeError("Another source-dataset overlay generation is in progress") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as lock_handle:
            lock_handle.write(f"pid={os.getpid()}\n")
        if destination.exists():
            if not replace_existing:
                raise RuntimeError(
                    f"Source-dataset overlay output already exists: {destination}"
                )
            try:
                _verify_existing_bundle_for_replacement(destination)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Existing source-dataset overlay output is not a verified "
                    "closed-schema v2/v3/v4 bundle"
                ) from exc

        stage = destination.with_name(
            f".{destination.name}.source-dataset-overlays-stage-{uuid.uuid4().hex[:8]}"
        )
        stage.mkdir()
        try:
            source_manifest: dict[str, dict[str, Any]] = {}
            source_contracts: dict[str, dict[str, Any]] = {}
            for source_name in sorted(specs):
                spec = specs[source_name]
                overlay = read_source_dataset_overlay(
                    spec["path"],
                    source_name,
                    encoding=spec["encoding"],
                )
                selected_overlay = select_source_dataset_overlay_above_yield_threshold(
                    overlay,
                    threshold_t_ha=threshold_t_ha,
                )
                if not selected_overlay.trajectories:
                    raise RuntimeError(
                        f"Source {source_name!r} has no whole unit with observed yield "
                        f"> {threshold_t_ha:g} t/ha"
                    )
                outputs: dict[str, dict[str, Any]] = {}
                overlays_by_role = {
                    "source_wide": overlay,
                    "yield_threshold_selected": selected_overlay,
                }
                for role, filename in _output_filenames(
                    source_name,
                    threshold_t_ha,
                ).items():
                    output_relative = f"figures/overlay/{source_name}/{filename}"
                    output_path = stage / output_relative
                    write_source_dataset_overlay_figure(
                        overlays_by_role[role],
                        output_path,
                    )
                    outputs[role] = {
                        "selection_scope": (
                            "all_finite_source_observations"
                            if role == "source_wide"
                            else "whole_units_with_any_observed_yield_strictly_above_threshold"
                        ),
                        "path": output_relative,
                        "sha256": _sha256_file(output_path),
                        "image": _read_image_metadata(output_path),
                    }
                source_manifest[source_name] = _source_manifest_entry(
                    source_name=source_name,
                    source_sha256=initial_hashes[source_name],
                    overlay=overlay,
                    selected_overlay=selected_overlay,
                    outputs=outputs,
                )
                source_contracts[source_name] = _source_verification_contract(
                    source_name=source_name,
                    source_sha256=initial_hashes[source_name],
                    overlay=overlay,
                    selected_overlay=selected_overlay,
                    outputs=outputs,
                )

            final_hashes = {
                source_name: _sha256_file(spec["path"])
                for source_name, spec in specs.items()
            }
            if final_hashes != initial_hashes:
                raise RuntimeError("A restricted source changed during overlay generation")
            if _implementation_hashes() != initial_implementation_hashes:
                raise RuntimeError(
                    "Generator or renderer changed during source-dataset overlay generation"
                )
            if (
                verified_config_path is not None
                and _sha256_file(verified_config_path) != config_sha256
            ):
                raise RuntimeError("Configuration changed during source-dataset overlay generation")

            manifest = {
                "schema_version": SCHEMA_VERSION,
                "status": BUNDLE_STATUS,
                "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "disclosure_class": "internal_restricted_diagnostic_not_for_release",
                "governed_release_relationship": "none",
                "governed_figure_inventory_affected": False,
                "accountable_human_review": "not_claimed",
                "yield_threshold_t_ha": threshold_t_ha,
                "source_count": len(source_manifest),
                "sources": source_manifest,
                "generator": {
                    **initial_implementation_hashes,
                    "config_sha256": config_sha256,
                    "atomic_stage_and_replace": False,
                    "source_bytes_reverified_unchanged": True,
                },
            }
            _write_json(stage / MANIFEST_NAME, manifest)
            entries = {
                path.relative_to(stage).as_posix(): _sha256_file(path)
                for path in stage.rglob("*")
                if path.is_file() and path.name != CHECKSUMS_NAME
            }
            (stage / CHECKSUMS_NAME).write_text(
                "".join(
                    f"{digest}  {relative}\n"
                    for relative, digest in sorted(entries.items())
                ),
                encoding="utf-8",
            )
            verify_source_dataset_overlay_bundle(
                stage,
                expected_config_sha256=config_sha256,
                expected_source_sha256=initial_hashes,
                expected_source_contracts=source_contracts,
            )
            _promote_stage(
                stage=stage,
                output_root=destination,
                replace_existing=replace_existing,
                config_sha256=config_sha256,
                config_path=verified_config_path,
                source_sha256=initial_hashes,
                source_contracts=source_contracts,
            )
        except Exception:
            if stage.exists():
                shutil.rmtree(stage)
            raise
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    verified = verify_source_dataset_overlay_bundle(
        destination,
        expected_config_sha256=config_sha256,
        expected_source_sha256=initial_hashes,
        expected_source_contracts=source_contracts,
    )
    return {
        "status": _RESULT_STATUS,
        "generated": True,
        "source_count": verified["source_count"],
        "output_root": str(destination),
    }


def _parse_restricted_extension_ledger(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Restricted extension checksum ledger is missing or unsafe")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("Restricted extension checksum ledger is unreadable") from exc
    entries: dict[str, str] = {}
    for line in lines:
        if not line.strip() or "  " not in line:
            raise RuntimeError("Restricted extension checksum ledger has an invalid line")
        digest, relative = line.split("  ", 1)
        if (
            _SAFE_SHA256.fullmatch(digest) is None
            or not _safe_relative_path(relative)
            or relative in entries
            or relative == RESTRICTED_EXTENSION_CHECKSUMS_NAME
        ):
            raise RuntimeError("Restricted extension checksum ledger has an unsafe entry")
        entries[relative] = digest
    if not entries:
        raise RuntimeError("Restricted extension checksum ledger is empty")
    return entries


def _governed_core_contract(package_root: Path) -> dict[str, Any]:
    manifest_path = package_root / "run_manifest.json"
    ledger_path = package_root / "CHECKSUMS.sha256"
    if (
        manifest_path.is_symlink()
        or ledger_path.is_symlink()
        or not manifest_path.is_file()
        or not ledger_path.is_file()
    ):
        raise RuntimeError("Restricted extension host package is incomplete or unsafe")
    try:
        ledger_lines = ledger_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("Restricted extension host package ledger is unreadable") from exc
    bound_artifact_count = sum(bool(line.strip()) for line in ledger_lines)
    if bound_artifact_count <= 0:
        raise RuntimeError("Restricted extension host package ledger is empty")
    return {
        "status": "governed_core_unchanged",
        "run_manifest_sha256": _sha256_file(manifest_path),
        "checksum_ledger_sha256": _sha256_file(ledger_path),
        "bound_artifact_count": bound_artifact_count,
        "strict_core_verified_before_extension": True,
    }


def _restricted_extension_actual_paths(extension_root: Path) -> set[str]:
    paths: set[str] = set()
    for path in extension_root.rglob("*"):
        relative = path.relative_to(extension_root).as_posix()
        if path.is_symlink():
            raise RuntimeError(
                f"Restricted extension contains a prohibited symlink: {relative}"
            )
        if path.is_file():
            paths.add(relative)
    return paths


def _verify_restricted_extension_local(
    package_root: Path,
    extension_root: Path,
    *,
    expected_config_sha256: str | None,
    expected_source_sha256: Mapping[str, str] | None,
    expected_source_contracts: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], frozenset[str]]:
    if _has_symlink_component(extension_root.absolute()) or not extension_root.is_dir():
        raise RuntimeError("Restricted extension root is missing or unsafe")
    actual_paths = _restricted_extension_actual_paths(extension_root)
    ledger_path = extension_root / RESTRICTED_EXTENSION_CHECKSUMS_NAME
    manifest_path = extension_root / RESTRICTED_EXTENSION_MANIFEST_NAME
    entries = _parse_restricted_extension_ledger(ledger_path)
    if actual_paths != set(entries) | {RESTRICTED_EXTENSION_CHECKSUMS_NAME}:
        raise RuntimeError(
            "Restricted extension checksum ledger does not cover the complete extension"
        )
    if RESTRICTED_EXTENSION_MANIFEST_NAME not in entries:
        raise RuntimeError("Restricted extension manifest is absent from its checksum ledger")
    for relative, expected_digest in entries.items():
        artifact = extension_root / relative
        if artifact.is_symlink() or not artifact.is_file():
            raise RuntimeError(f"Restricted extension artifact is missing: {relative}")
        if _sha256_file(artifact) != expected_digest:
            raise RuntimeError(f"Restricted extension checksum mismatch: {relative}")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError("Restricted extension manifest is unreadable") from exc
    manifest = _require_exact_keys(
        manifest,
        _RESTRICTED_EXTENSION_MANIFEST_KEYS,
        label="restricted extension manifest",
    )
    timestamp = manifest.get("generated_at_utc")
    if (
        manifest.get("schema_version") != RESTRICTED_EXTENSION_SCHEMA_VERSION
        or manifest.get("status") != RESTRICTED_EXTENSION_STATUS
        or manifest.get("disclosure_class")
        != "mixed_container_with_restricted_internal_extension_not_for_release"
        or manifest.get("accountable_human_review") != "not_claimed"
        or not isinstance(timestamp, str)
        or _SAFE_UTC_TIMESTAMP.fullmatch(timestamp) is None
        or _contains_restricted_locator(manifest)
    ):
        raise RuntimeError("Restricted extension manifest status or privacy contract is invalid")
    governed_core = _require_exact_keys(
        manifest.get("governed_core"),
        _RESTRICTED_EXTENSION_CORE_KEYS,
        label="restricted extension governed core",
    )
    current_core = _governed_core_contract(package_root)
    if governed_core != current_core:
        raise RuntimeError("Restricted extension host package binding does not match")
    extension = _require_exact_keys(
        manifest.get("extension"),
        _RESTRICTED_EXTENSION_BUNDLE_KEYS,
        label="restricted extension bundle",
    )
    if extension.get("relative_root") != RESTRICTED_EXTENSION_BUNDLE_DIRECTORY:
        raise RuntimeError("Restricted extension bundle root is invalid")
    bundle_root = extension_root / RESTRICTED_EXTENSION_BUNDLE_DIRECTORY
    if expected_config_sha256 is None:
        bundle_manifest = _verify_existing_bundle_for_replacement(bundle_root)
    else:
        if expected_source_sha256 is None or expected_source_contracts is None:
            raise RuntimeError("Restricted extension current verification inputs are incomplete")
        bundle_manifest = verify_source_dataset_overlay_bundle(
            bundle_root,
            expected_config_sha256=expected_config_sha256,
            expected_source_sha256=expected_source_sha256,
            expected_source_contracts=expected_source_contracts,
        )
    bundle_paths = _bundle_actual_paths(bundle_root)
    expected_bundle_file_count = 2 * int(bundle_manifest["source_count"]) + 2
    if (
        extension.get("schema_version") != bundle_manifest.get("schema_version")
        or extension.get("manifest_sha256")
        != _sha256_file(bundle_root / MANIFEST_NAME)
        or extension.get("checksum_ledger_sha256")
        != _sha256_file(bundle_root / CHECKSUMS_NAME)
        or extension.get("bundle_file_count") != len(bundle_paths)
        or len(bundle_paths) != expected_bundle_file_count
    ):
        raise RuntimeError("Restricted extension bundle binding does not match")
    expected_bundle_entries = {
        f"{RESTRICTED_EXTENSION_BUNDLE_DIRECTORY}/{relative}"
        for relative in bundle_paths
    }
    if set(entries) != expected_bundle_entries | {RESTRICTED_EXTENSION_MANIFEST_NAME}:
        raise RuntimeError("Restricted extension inventory is not exact")
    allowed_paths = frozenset(
        f"{RESTRICTED_EXTENSION_DIRECTORY}/{relative}"
        for relative in actual_paths
    )
    return manifest, allowed_paths


def recognize_restricted_report_extension(
    package_root: str | Path,
) -> frozenset[str]:
    """Recognize a closed signed extension for prior-package reuse/replacement only."""

    lexical_package = Path(package_root).absolute()
    if _has_symlink_component(lexical_package):
        raise RuntimeError("Restricted report container path contains a prohibited symlink")
    package = lexical_package.resolve()
    extension_root = package / RESTRICTED_EXTENSION_DIRECTORY
    _manifest, allowed_paths = _verify_restricted_extension_local(
        package,
        extension_root,
        expected_config_sha256=None,
        expected_source_sha256=None,
        expected_source_contracts=None,
    )
    return allowed_paths


def verify_restricted_report_container(
    package_root: str | Path,
    *,
    expected_config_sha256: str,
    expected_source_sha256: Mapping[str, str],
    expected_source_contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify the unchanged governed core plus its exact current restricted extension."""

    lexical_package = Path(package_root).absolute()
    if _has_symlink_component(lexical_package):
        raise RuntimeError("Restricted report container path contains a prohibited symlink")
    package = lexical_package.resolve()
    extension_root = package / RESTRICTED_EXTENSION_DIRECTORY
    manifest, allowed_paths = _verify_restricted_extension_local(
        package,
        extension_root,
        expected_config_sha256=expected_config_sha256,
        expected_source_sha256=expected_source_sha256,
        expected_source_contracts=expected_source_contracts,
    )
    from n_response_curve.reporting.release import (  # noqa: PLC0415
        _verify_release_package_with_allowed_extras,
    )

    _verify_release_package_with_allowed_extras(
        package,
        allowed_extra_paths=allowed_paths,
    )
    return {
        "status": "mixed_restricted_report_container_verified",
        "governed_core": dict(manifest["governed_core"]),
        "extension": dict(manifest["extension"]),
    }


def _write_restricted_extension_contract(
    *,
    package_root: Path,
    extension_root: Path,
) -> None:
    bundle_root = extension_root / RESTRICTED_EXTENSION_BUNDLE_DIRECTORY
    bundle_manifest = _verify_existing_bundle_for_replacement(bundle_root)
    bundle_paths = _bundle_actual_paths(bundle_root)
    manifest = {
        "schema_version": RESTRICTED_EXTENSION_SCHEMA_VERSION,
        "status": RESTRICTED_EXTENSION_STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "disclosure_class": (
            "mixed_container_with_restricted_internal_extension_not_for_release"
        ),
        "accountable_human_review": "not_claimed",
        "governed_core": _governed_core_contract(package_root),
        "extension": {
            "relative_root": RESTRICTED_EXTENSION_BUNDLE_DIRECTORY,
            "schema_version": bundle_manifest["schema_version"],
            "manifest_sha256": _sha256_file(bundle_root / MANIFEST_NAME),
            "checksum_ledger_sha256": _sha256_file(bundle_root / CHECKSUMS_NAME),
            "bundle_file_count": len(bundle_paths),
        },
    }
    _write_json(extension_root / RESTRICTED_EXTENSION_MANIFEST_NAME, manifest)
    entries = {
        path.relative_to(extension_root).as_posix(): _sha256_file(path)
        for path in extension_root.rglob("*")
        if path.is_file() and path.name != RESTRICTED_EXTENSION_CHECKSUMS_NAME
    }
    (extension_root / RESTRICTED_EXTENSION_CHECKSUMS_NAME).write_text(
        "".join(
            f"{digest}  {relative}\n"
            for relative, digest in sorted(entries.items())
        ),
        encoding="utf-8",
    )


def generate_restricted_report_extension(
    *,
    project_root: str | Path,
    package_root: str | Path,
    output_root: str | Path,
    source_specs: Mapping[str, Mapping[str, Any]],
    replace_existing: bool,
    yield_threshold_t_ha: float,
    config_sha256: str,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Stage, bind, promote, and verify the restricted report extension."""

    project = Path(project_root).resolve()
    lexical_package = Path(package_root)
    if not lexical_package.is_absolute():
        lexical_package = project / lexical_package
    lexical_package = lexical_package.absolute()
    if _has_symlink_component(lexical_package):
        raise RuntimeError("Restricted report container path contains a prohibited symlink")
    package = lexical_package.resolve()
    if not package.is_dir() or not package.is_relative_to(project):
        raise RuntimeError("Restricted report container host package is unsafe")
    destination = Path(output_root)
    if not destination.is_absolute():
        destination = project / destination
    destination = destination.absolute().resolve()
    extension_root = package / RESTRICTED_EXTENSION_DIRECTORY
    expected_destination = extension_root / RESTRICTED_EXTENSION_BUNDLE_DIRECTORY
    if destination != expected_destination:
        raise RuntimeError(
            "Restricted report extension output must be restricted_diagnostics/"
            "source_dataset_overlays below its host package"
        )
    verified_config_path = Path(config_path).resolve() if config_path is not None else None
    if (
        verified_config_path is not None
        and _sha256_file(verified_config_path) != config_sha256
    ):
        raise RuntimeError("Configuration changed before restricted extension generation")
    source_hashes = {
        source_name: _sha256_file(Path(str(spec["path"])).resolve())
        for source_name, spec in source_specs.items()
    }
    source_contracts = build_source_dataset_overlay_verification_contracts(
        project_root=project,
        source_specs=source_specs,
        yield_threshold_t_ha=yield_threshold_t_ha,
    )
    lock_path = package.parent / f".{package.name}.restricted-extension.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise RuntimeError("Another restricted report extension operation is in progress") from exc
    stage_parent: Path | None = None
    backup: Path | None = None
    promoted = False
    prior_moved = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as lock_handle:
            lock_handle.write(f"pid={os.getpid()}\n")
        from n_response_curve.reporting.release import (  # noqa: PLC0415
            verify_release_package,
        )

        if extension_root.exists():
            if not replace_existing:
                raise RuntimeError(f"Restricted extension already exists: {extension_root}")
            verify_restricted_report_container(
                package,
                expected_config_sha256=config_sha256,
                expected_source_sha256=source_hashes,
                expected_source_contracts=source_contracts,
            )
        else:
            verify_release_package(package)
        core_before = _governed_core_contract(package)
        stage_parent = Path(
            tempfile.mkdtemp(
                prefix=f".{package.name}.restricted-extension-stage-",
                dir=package.parent,
            )
        )
        stage_extension = stage_parent / RESTRICTED_EXTENSION_DIRECTORY
        stage_bundle = stage_extension / RESTRICTED_EXTENSION_BUNDLE_DIRECTORY
        generate_source_dataset_overlays(
            project_root=project,
            output_root=stage_bundle,
            source_specs=source_specs,
            replace_existing=False,
            yield_threshold_t_ha=yield_threshold_t_ha,
            config_sha256=config_sha256,
            config_path=verified_config_path,
        )
        _write_restricted_extension_contract(
            package_root=package,
            extension_root=stage_extension,
        )
        _verify_restricted_extension_local(
            package,
            stage_extension,
            expected_config_sha256=config_sha256,
            expected_source_sha256=source_hashes,
            expected_source_contracts=source_contracts,
        )
        if _governed_core_contract(package) != core_before:
            raise RuntimeError("Restricted extension host package changed during generation")
        if any(
            _sha256_file(Path(str(spec["path"])).resolve()) != source_hashes[source_name]
            for source_name, spec in source_specs.items()
        ):
            raise RuntimeError("A restricted source changed during extension generation")
        if (
            verified_config_path is not None
            and _sha256_file(verified_config_path) != config_sha256
        ):
            raise RuntimeError("Configuration changed during restricted extension generation")
        if extension_root.exists():
            backup = package.parent / (
                f".{package.name}.restricted-extension-backup-{uuid.uuid4().hex[:8]}"
            )
            os.replace(extension_root, backup)
            prior_moved = True
        os.replace(stage_extension, extension_root)
        promoted = True
        verify_restricted_report_container(
            package,
            expected_config_sha256=config_sha256,
            expected_source_sha256=source_hashes,
            expected_source_contracts=source_contracts,
        )
    except Exception:
        if promoted and extension_root.exists():
            shutil.rmtree(extension_root)
        if prior_moved and backup is not None and backup.exists():
            os.replace(backup, extension_root)
        raise
    else:
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage_parent is not None and stage_parent.exists():
            shutil.rmtree(stage_parent)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
    verified = verify_restricted_report_container(
        package,
        expected_config_sha256=config_sha256,
        expected_source_sha256=source_hashes,
        expected_source_contracts=source_contracts,
    )
    return {
        "status": verified["status"],
        "generated": True,
        "source_count": len(source_specs),
        "output_root": str(destination),
    }


def _validate_mode_response(run_mode: str) -> dict[str, Any] | None:
    if run_mode != "validate":
        return None
    return {
        "status": "validate_mode_noop",
        "generated": False,
        "reason": "validate mode never writes source-dataset diagnostic overlays",
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to scriptCONFIG.toml")
    parser.add_argument(
        "--mode",
        choices=("validate", "test", "full"),
        help="Run-mode override supplied by script.sh",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config_path = Path(args.config).resolve()
        config_sha256 = _sha256_file(config_path)
        config = load_config(
            config_path,
            project_root=PROJECT_ROOT,
            check_files=False,
        )
        if _sha256_file(config_path) != config_sha256:
            raise RuntimeError("Configuration changed while it was being parsed")
        run_mode = args.mode or config.run_mode
        validate_response = _validate_mode_response(run_mode)
        if validate_response is not None:
            print(json.dumps(validate_response, sort_keys=True))
            return 0
        settings = config.raw.get("source_dataset_overlays", {})
        if not settings.get("enabled", False):
            print(json.dumps({"status": "disabled", "generated": False}, sort_keys=True))
            return 0
        if run_mode not in {"test", "full"}:
            raise ConfigError(
                "[source_dataset_overlays] can write only after a test/full pipeline run"
            )
        placement = str(settings.get("placement", "separate_bundle"))
        if placement == "restricted_package_extension" and run_mode != "full":
            print(
                json.dumps(
                    {
                        "status": "restricted_package_extension_full_mode_only",
                        "generated": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        source_specs = {
            source_name: {
                "path": (
                    PROJECT_ROOT / str(config.sources[source_name]["data_path"])
                ),
                "encoding": str(config.sources[source_name]["encoding"]),
                "data_classification": config.sources[source_name]["data_classification"],
            }
            for source_name in settings["source_names"]
        }
        if placement == "restricted_package_extension":
            result = generate_restricted_report_extension(
                project_root=PROJECT_ROOT,
                package_root=PROJECT_ROOT / str(settings["package_path"]),
                output_root=PROJECT_ROOT / str(settings["output_root"]),
                source_specs=source_specs,
                replace_existing=bool(settings["replace_existing"]),
                yield_threshold_t_ha=float(settings["yield_threshold_t_ha"]),
                config_sha256=config_sha256,
                config_path=config_path,
            )
        else:
            result = generate_source_dataset_overlays(
                project_root=PROJECT_ROOT,
                output_root=PROJECT_ROOT / str(settings["output_root"]),
                source_specs=source_specs,
                replace_existing=bool(settings["replace_existing"]),
                yield_threshold_t_ha=float(settings["yield_threshold_t_ha"]),
                config_sha256=config_sha256,
                config_path=config_path,
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ConfigError, RuntimeError, OSError, UnicodeError, ValueError) as exc:
        print(f"configuration-error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
