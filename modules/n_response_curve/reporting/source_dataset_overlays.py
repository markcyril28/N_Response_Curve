"""Reusable parsing and rendering for descriptive source-dataset overlays.

These are full or whole-unit-selected overlays over the configured core-trial,
LTCCE, and linked-arm source datasets. No pooled curve or fit is drawn, and no
raw identifiers (references, farmer names, coordinates, source paths) are
carried into plot labels or the returned summary.
"""

from __future__ import annotations

import csv
import codecs
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib.pyplot as plt

from n_response_curve.analysis.values import finite_number
from n_response_curve.data.config import SUPPORTED_SOURCE_DATASET_OVERLAY_NAMES
from .source_display_names import display_source_name

_CORE_TRIAL_SOURCE_NAME = "core_trial_data"
_LTCCE_SOURCE_NAME = "ltcce"
_COMBINED_SOURCE_NAME = "ph_combined_nopt_rcm"
SUPPORTED_SOURCE_NAMES = SUPPORTED_SOURCE_DATASET_OVERLAY_NAMES

# Reviewed per-source encodings from scriptCONFIG.toml's [[sources]] entries
# (both members of data/config.py:KNOWN_SOURCE_ENCODINGS). ph_combined_nopt_rcm
# is registered as cp1252, not UTF-8; reading it as UTF-8 raises
# UnicodeDecodeError on non-ASCII bytes in held free-text columns.
_SOURCE_ENCODINGS = {
    _CORE_TRIAL_SOURCE_NAME: "utf-8-sig",
    _LTCCE_SOURCE_NAME: "utf-8-sig",
    _COMBINED_SOURCE_NAME: "cp1252",
}

_CORE_TRIAL_REQUIRED_HEADERS = frozenset(
    {
        "Study_ID",
        "Trial ID",
        "Treatment",
        "Inorganic N_rate (kg N ha-1)",
        "Grain Yield (kg ha-1)",
        "Grain Yield (t ha-1)",
    }
)

# Grouping/context headers preserved verbatim from the LTCCE curated CSV.
# VarCode is deliberately excluded: it is context only and must not replace
# Variety identity in the trajectory key.
_LTCCE_CONTEXT_HEADERS = (
    "Design",
    "Expt",
    "Site",
    "Year",
    "Season",
    "Crop",
    "Establishment",
    "Variety",
    "Rep",
)
_LTCCE_REQUIRED_HEADERS = frozenset({*_LTCCE_CONTEXT_HEADERS, "VarCode", "Nfert", "GYtha"})

# Canonical PH combined arm columns, matching the reviewed field/arm positions
# in data/runtime_source_contracts.py (fp=58/103, rcm=85/104,
# nopt_full=140/148, nopt_zero_n constant N=0/149). Only the columns needed
# to render finite N/yield observations are read; identifier, location, and
# unrelated columns are never touched by this module.
_COMBINED_ARM_SPECS = (
    ("fp", "FP", "fp_actual_n_kg_per_ha", "fp_measured_grainyield_dry"),
    ("rcm", "RCM", "rcm_actual_n_kg_per_ha", "rcm_measured_grainyield_dry"),
    ("nopt_full", "NOPT NPK", "nrate", "full_fert_yield"),
)
_COMBINED_ZERO_N_ARM = ("nopt_zero_n", "zero N", "n0_yield")
_COMBINED_REQUIRED_HEADERS = frozenset(
    {
        *(n_header for _, _, n_header, _ in _COMBINED_ARM_SPECS),
        *(y_header for _, _, _, y_header in _COMBINED_ARM_SPECS),
        _COMBINED_ZERO_N_ARM[2],
    }
)

# Source-file variants of the PH combined CSV. "with_fp" is the registered file
# named by [sources.ph_combined_nopt_rcm].data_path, which carries the Farmer's
# Practice arm as its fp_* column block. "no_fp" is the sibling file with that
# whole block removed, so the arm cannot reach a figure even by accident.
#
# The variant is always named by the caller and never inferred from which
# headers a file happens to carry. Each variant asserts both what must be
# present and what must be absent, so an FP-free file cannot silently stand in
# for the registered one and the registered file cannot silently satisfy a
# request for file-level FP-free provenance.
COMBINED_VARIANT_WITH_FP = "with_fp"
COMBINED_VARIANT_NO_FP = "no_fp"
COMBINED_SOURCE_VARIANTS = (COMBINED_VARIANT_WITH_FP, COMBINED_VARIANT_NO_FP)
_COMBINED_FP_ARM_ID = "fp"
_COMBINED_FP_HEADER_PREFIX = "fp_"
# Filename stem suffix of each variant, relative to the registered data_path.
# Deriving the sibling from the registered path keeps scriptCONFIG.toml
# untouched and leaves no second path binding that could drift from it.
_COMBINED_VARIANT_STEM_SUFFIXES = {
    COMBINED_VARIANT_WITH_FP: "",
    COMBINED_VARIANT_NO_FP: "_no_FP",
}


def _require_known_variant(variant: str) -> str:
    if variant not in COMBINED_SOURCE_VARIANTS:
        raise ValueError(
            f"Unsupported {_COMBINED_SOURCE_NAME} source variant: {variant!r}; "
            "supported: " + ", ".join(COMBINED_SOURCE_VARIANTS)
        )
    return variant


def combined_variant_source_path(source_path: str | Path, variant: str) -> Path:
    """Return the file holding *variant* of the registered PH combined source."""

    path = Path(source_path)
    suffix = _COMBINED_VARIANT_STEM_SUFFIXES[_require_known_variant(variant)]
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _combined_arm_specs(variant: str) -> tuple[tuple[str, str, str, str], ...]:
    """The arm specs this variant's file is able to supply."""

    if _require_known_variant(variant) == COMBINED_VARIANT_WITH_FP:
        return _COMBINED_ARM_SPECS
    return tuple(
        spec for spec in _COMBINED_ARM_SPECS if spec[0] != _COMBINED_FP_ARM_ID
    )


def _combined_required_headers(variant: str) -> frozenset[str]:
    specs = _combined_arm_specs(variant)
    return frozenset(
        {
            *(n_header for _, _, n_header, _ in specs),
            *(y_header for _, _, _, y_header in specs),
            _COMBINED_ZERO_N_ARM[2],
        }
    )


def _forbid_farmers_practice_headers(
    fieldnames: Sequence[str] | None, variant: str
) -> None:
    """Reject an FP-carrying file when FP-free provenance was requested."""

    if _require_known_variant(variant) == COMBINED_VARIANT_WITH_FP:
        return
    present = sorted(
        header
        for header in (fieldnames or ())
        if header.startswith(_COMBINED_FP_HEADER_PREFIX)
    )
    if present:
        raise ValueError(
            f"Source '{_COMBINED_SOURCE_NAME}' variant "
            f"{COMBINED_VARIANT_NO_FP!r} requires an input with no Farmer's "
            "Practice columns, but the file still carries: "
            + ", ".join(present)
        )

# The whole unit a threshold selection retains, per source. Selecting a unit
# keeps every finite member observation, so threshold filtering never severs a
# trajectory into disconnected points.
_SELECTION_UNIT_BY_SOURCE = {
    _CORE_TRIAL_SOURCE_NAME: "source_row",
    _COMBINED_SOURCE_NAME: "linked_source_row",
    _LTCCE_SOURCE_NAME: "replicate_specific_trajectory",
}
_RETAINED_UNIT_LABELS = {
    "linked_source_row": "linked rows",
    "source_row": "source rows",
    "source_series": "source series",
    "replicate_specific_trajectory": "replicate trajectories",
}


@dataclass(frozen=True)
class SourceObservation:
    trajectory_id: str
    n_rate_kg_ha: float
    yield_t_ha: float
    treatment_class: str


@dataclass(frozen=True)
class SourceTrajectory:
    trajectory_id: str
    observations: tuple[SourceObservation, ...]
    excluded_nonfinite_observation_count: int = 0


@dataclass(frozen=True)
class SourceDatasetOverlaySummary:
    source_name: str
    source_rows: int
    finite_observation_count: int
    trajectory_count: int
    treatment_classes: tuple[str, ...]
    excluded_observation_count: int
    n_rate_range_kg_ha: tuple[float, float] | None
    yield_range_t_ha: tuple[float, float] | None


@dataclass(frozen=True)
class SourceDatasetOverlaySelection:
    operator: str
    threshold_t_ha: float
    selection_unit: str
    selected_unit_count: int
    total_unit_count: int
    selected_observation_count: int
    total_observation_count: int
    threshold_exceeding_observation_count: int
    threshold_equal_observation_count: int
    threshold_equal_only_unit_count: int
    excluded_nonfinite_observation_count_within_selected_units: int


@dataclass(frozen=True)
class SourceDatasetNRateSelection:
    operator: str
    threshold_kg_ha: float
    selection_unit: str
    selected_unit_count: int
    total_unit_count: int
    selected_observation_count: int
    total_observation_count: int
    threshold_exceeding_observation_count: int
    threshold_equal_observation_count: int
    threshold_equal_only_unit_count: int
    excluded_nonfinite_observation_count_within_selected_units: int


@dataclass(frozen=True)
class SourceDatasetZeroNSelection:
    operator: str
    threshold_t_ha: float
    baseline_n_rate_kg_ha: float
    selection_unit: str
    selected_unit_count: int
    total_unit_count: int
    selected_observation_count: int
    total_observation_count: int
    equal_threshold_unit_count: int
    missing_baseline_unit_count: int
    conflicting_baseline_unit_count: int
    duplicate_identical_baseline_unit_count: int
    excluded_nonfinite_observation_count_within_selected_units: int
    unconnected_ambiguous_trajectory_ids: tuple[str, ...]


@dataclass(frozen=True)
class SourceDatasetOverlay:
    source_name: str
    trajectories: tuple[SourceTrajectory, ...]
    summary: SourceDatasetOverlaySummary
    selection: (
        SourceDatasetOverlaySelection
        | SourceDatasetNRateSelection
        | SourceDatasetZeroNSelection
        | None
    ) = None


@dataclass(frozen=True)
class SourceDatasetZeroNStrata:
    below: SourceDatasetOverlay
    above: SourceDatasetOverlay
    equal_threshold_trajectory_ids: tuple[str, ...]
    missing_baseline_trajectory_ids: tuple[str, ...]
    conflicting_baseline_trajectory_ids: tuple[str, ...]


def _trajectory_id(source_name: str, *parts: str) -> str:
    digest_input = "\x1f".join((source_name, *parts))
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16]


def _require_headers(fieldnames: Sequence[str] | None, required: frozenset[str], *, source_name: str) -> None:
    present = set(fieldnames or ())
    missing = sorted(required - present)
    if missing:
        raise ValueError(
            f"Source '{source_name}' overlay input is missing required header(s): "
            + ", ".join(missing)
        )


def _read_rows(
    csv_path: str | Path,
    *,
    source_name: str,
    encoding: str | None = None,
    required_headers: frozenset[str] | None = None,
) -> tuple[Sequence[str], list[dict[str, str]]]:
    reviewed_encoding = _SOURCE_ENCODINGS[source_name]
    if encoding is not None:
        try:
            requested_codec = codecs.lookup(encoding).name
            reviewed_codec = codecs.lookup(reviewed_encoding).name
        except LookupError as exc:
            raise ValueError(
                f"Source '{source_name}' overlay encoding is unknown: {encoding!r}"
            ) from exc
        if requested_codec != reviewed_codec:
            raise ValueError(
                f"Source '{source_name}' overlay encoding does not match the reviewed "
                f"source contract: {encoding!r} != {reviewed_encoding!r}"
            )
    path = Path(csv_path)
    if path.is_symlink():
        raise ValueError(f"Source '{source_name}' overlay input may not be a symlink")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Source '{source_name}' overlay input is unreadable or unsafe") from exc
    if required_headers is None:
        required_headers = {
            _CORE_TRIAL_SOURCE_NAME: _CORE_TRIAL_REQUIRED_HEADERS,
            _LTCCE_SOURCE_NAME: _LTCCE_REQUIRED_HEADERS,
            _COMBINED_SOURCE_NAME: _COMBINED_REQUIRED_HEADERS,
        }[source_name]
    try:
        text_handle = os.fdopen(
            descriptor,
            "r",
            newline="",
            encoding=reviewed_encoding,
        )
        descriptor = -1
        with text_handle as handle:
            reader = csv.reader(handle)
            fieldnames = next(reader, [])
            _require_headers(fieldnames, required_headers, source_name=source_name)
            duplicate_required = sorted(
                header for header in required_headers if fieldnames.count(header) != 1
            )
            if duplicate_required:
                raise ValueError(
                    f"Source '{source_name}' overlay input has ambiguous required header(s): "
                    + ", ".join(duplicate_required)
                )
            projected_headers = tuple(
                header for header in fieldnames if header in required_headers
            )
            indices = {header: fieldnames.index(header) for header in projected_headers}
            rows = [
                {
                    header: row[index] if index < len(row) else ""
                    for header, index in indices.items()
                }
                for row in reader
            ]
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    return tuple(fieldnames), rows


def _finalize_overlay(
    *,
    source_name: str,
    source_rows: int,
    excluded_observation_count: int,
    observations_by_trajectory: dict[str, list[SourceObservation]],
    excluded_observations_by_trajectory: Mapping[str, int] | None = None,
    selection: (
        SourceDatasetOverlaySelection
        | SourceDatasetNRateSelection
        | SourceDatasetZeroNSelection
        | None
    ) = None,
) -> SourceDatasetOverlay:
    excluded_by_trajectory = excluded_observations_by_trajectory or {}
    trajectories = tuple(
        SourceTrajectory(
            trajectory_id=trajectory_id,
            observations=tuple(
                sorted(
                    observations,
                    key=lambda observation: (
                        observation.n_rate_kg_ha,
                        observation.treatment_class,
                    ),
                )
            ),
            excluded_nonfinite_observation_count=excluded_by_trajectory.get(
                trajectory_id,
                0,
            ),
        )
        for trajectory_id, observations in sorted(observations_by_trajectory.items())
        if observations
    )
    all_observations = [
        observation for trajectory in trajectories for observation in trajectory.observations
    ]
    treatment_classes = tuple(sorted({observation.treatment_class for observation in all_observations}))
    n_values = [observation.n_rate_kg_ha for observation in all_observations]
    yield_values = [observation.yield_t_ha for observation in all_observations]
    summary = SourceDatasetOverlaySummary(
        source_name=source_name,
        source_rows=source_rows,
        finite_observation_count=len(all_observations),
        trajectory_count=len(trajectories),
        treatment_classes=treatment_classes,
        excluded_observation_count=excluded_observation_count,
        n_rate_range_kg_ha=(min(n_values), max(n_values)) if n_values else None,
        yield_range_t_ha=(min(yield_values), max(yield_values)) if yield_values else None,
    )
    return SourceDatasetOverlay(
        source_name=source_name,
        trajectories=trajectories,
        summary=summary,
        selection=selection,
    )


def _read_ltcce_overlay(
    csv_path: str | Path,
    *,
    encoding: str | None = None,
) -> SourceDatasetOverlay:
    fieldnames, rows = _read_rows(
        csv_path,
        source_name=_LTCCE_SOURCE_NAME,
        encoding=encoding,
    )
    _require_headers(fieldnames, _LTCCE_REQUIRED_HEADERS, source_name=_LTCCE_SOURCE_NAME)

    observations_by_trajectory: dict[str, list[SourceObservation]] = {}
    excluded_by_trajectory: dict[str, int] = {}
    excluded_observation_count = 0
    for row in rows:
        context = tuple(str(row.get(header, "")).strip() for header in _LTCCE_CONTEXT_HEADERS)
        trajectory_id = _trajectory_id(_LTCCE_SOURCE_NAME, *context)
        n_rate = finite_number(row.get("Nfert"))
        yield_value = finite_number(row.get("GYtha"))
        if n_rate is None or yield_value is None:
            excluded_observation_count += 1
            excluded_by_trajectory[trajectory_id] = excluded_by_trajectory.get(trajectory_id, 0) + 1
            continue
        observations_by_trajectory.setdefault(trajectory_id, []).append(
            SourceObservation(
                trajectory_id=trajectory_id,
                n_rate_kg_ha=n_rate,
                yield_t_ha=yield_value,
                treatment_class="zero N" if n_rate == 0.0 else "mineral N rate",
            )
        )

    return _finalize_overlay(
        source_name=_LTCCE_SOURCE_NAME,
        source_rows=len(rows),
        excluded_observation_count=excluded_observation_count,
        observations_by_trajectory=observations_by_trajectory,
        excluded_observations_by_trajectory=excluded_by_trajectory,
    )


def _read_core_trial_overlay(
    csv_path: str | Path,
    *,
    encoding: str | None = None,
) -> SourceDatasetOverlay:
    """Read finite core-trial rows without inferring response-series identity."""

    fieldnames, raw_rows = _read_rows(
        csv_path,
        source_name=_CORE_TRIAL_SOURCE_NAME,
        encoding=encoding,
    )
    _require_headers(
        fieldnames,
        _CORE_TRIAL_REQUIRED_HEADERS,
        source_name=_CORE_TRIAL_SOURCE_NAME,
    )
    rows = [row for row in raw_rows if any(str(value).strip() for value in row.values())]
    observations_by_trajectory: dict[str, list[SourceObservation]] = {}
    excluded_by_trajectory: dict[str, int] = {}
    excluded_observation_count = 0
    for row_index, row in enumerate(rows, start=1):
        trajectory_id = _trajectory_id(
            _CORE_TRIAL_SOURCE_NAME,
            "source-row",
            str(row_index),
        )
        n_rate = finite_number(row.get("Inorganic N_rate (kg N ha-1)"))
        yield_value = finite_number(row.get("Grain Yield (t ha-1)"))
        if yield_value is None:
            kilograms = finite_number(row.get("Grain Yield (kg ha-1)"))
            if kilograms is not None:
                yield_value = kilograms / 1000.0
        if n_rate is None or yield_value is None:
            excluded_observation_count += 1
            excluded_by_trajectory[trajectory_id] = 1
            continue
        observations_by_trajectory.setdefault(trajectory_id, []).append(
            SourceObservation(
                trajectory_id=trajectory_id,
                n_rate_kg_ha=n_rate,
                yield_t_ha=yield_value,
                treatment_class="zero N" if n_rate == 0.0 else "mineral N rate",
            )
        )

    return _finalize_overlay(
        source_name=_CORE_TRIAL_SOURCE_NAME,
        source_rows=len(rows),
        excluded_observation_count=excluded_observation_count,
        observations_by_trajectory=observations_by_trajectory,
        excluded_observations_by_trajectory=excluded_by_trajectory,
    )


def _read_combined_overlay(
    csv_path: str | Path,
    *,
    encoding: str | None = None,
    variant: str = COMBINED_VARIANT_WITH_FP,
) -> SourceDatasetOverlay:
    required_headers = _combined_required_headers(variant)
    fieldnames, rows = _read_rows(
        csv_path,
        source_name=_COMBINED_SOURCE_NAME,
        encoding=encoding,
        required_headers=required_headers,
    )
    _require_headers(fieldnames, required_headers, source_name=_COMBINED_SOURCE_NAME)
    _forbid_farmers_practice_headers(fieldnames, variant)
    arm_specs = _combined_arm_specs(variant)
    # Arms this variant's file can hold per linked row: the named arms plus the
    # zero-N column. The count follows the variant rather than the registered
    # four, so a dropped arm is never miscounted as a nonfinite exclusion.
    arms_per_row = len(arm_specs) + 1

    observations_by_trajectory: dict[str, list[SourceObservation]] = {}
    excluded_by_trajectory: dict[str, int] = {}
    excluded_observation_count = 0
    for row_index, row in enumerate(rows):
        trajectory_id = _trajectory_id(_COMBINED_SOURCE_NAME, "row", str(row_index))
        row_observations: list[SourceObservation] = []

        for _arm_id, treatment_class, n_header, y_header in arm_specs:
            n_rate = finite_number(row.get(n_header))
            yield_value = finite_number(row.get(y_header))
            if n_rate is None or yield_value is None:
                excluded_observation_count += 1
                continue
            row_observations.append(
                SourceObservation(
                    trajectory_id=trajectory_id,
                    n_rate_kg_ha=n_rate,
                    yield_t_ha=yield_value,
                    treatment_class=treatment_class,
                )
            )

        _arm_id, treatment_class, y_header = _COMBINED_ZERO_N_ARM
        yield_value = finite_number(row.get(y_header))
        if yield_value is None:
            excluded_observation_count += 1
        else:
            row_observations.append(
                SourceObservation(
                    trajectory_id=trajectory_id,
                    n_rate_kg_ha=0.0,
                    yield_t_ha=yield_value,
                    treatment_class=treatment_class,
                )
            )

        if row_observations:
            observations_by_trajectory.setdefault(trajectory_id, []).extend(row_observations)
            excluded_by_trajectory[trajectory_id] = arms_per_row - len(row_observations)

    return _finalize_overlay(
        source_name=_COMBINED_SOURCE_NAME,
        source_rows=len(rows),
        excluded_observation_count=excluded_observation_count,
        observations_by_trajectory=observations_by_trajectory,
        excluded_observations_by_trajectory=excluded_by_trajectory,
    )


def read_source_dataset_overlay(
    csv_path: str | Path,
    source_name: str,
    *,
    encoding: str | None = None,
    variant: str = COMBINED_VARIANT_WITH_FP,
) -> SourceDatasetOverlay:
    """Parse a supported source's curated CSV into an opaque overlay structure.

    Pure/read: no plotting or file writing happens here. Unsupported source
    names and CSVs missing required headers are rejected before any rendering
    is attempted.

    ``variant`` selects which file of the PH combined source is being read; it
    only applies to that source, and only its default is accepted for the
    others, so a variant request can never be silently ignored.
    """

    if source_name == _COMBINED_SOURCE_NAME:
        return _read_combined_overlay(csv_path, encoding=encoding, variant=variant)
    if _require_known_variant(variant) != COMBINED_VARIANT_WITH_FP:
        raise ValueError(
            f"Source '{source_name}' has no source-file variants; "
            f"{variant!r} applies only to '{_COMBINED_SOURCE_NAME}'"
        )
    if source_name == _CORE_TRIAL_SOURCE_NAME:
        return _read_core_trial_overlay(csv_path, encoding=encoding)
    if source_name == _LTCCE_SOURCE_NAME:
        return _read_ltcce_overlay(csv_path, encoding=encoding)
    raise ValueError(
        f"Unsupported source dataset overlay name: {source_name!r}; supported: "
        + ", ".join(sorted(SUPPORTED_SOURCE_NAMES))
    )


@dataclass(frozen=True)
class _WholeUnitThresholdTally:
    """Counts shared by every strictly-above whole-unit selection."""

    selected_trajectories: tuple[SourceTrajectory, ...]
    selection_unit: str
    threshold_exceeding_observation_count: int
    threshold_equal_observation_count: int
    threshold_equal_only_unit_count: int
    selected_observation_count: int
    excluded_nonfinite_observation_count_within_selected_units: int


def _tally_units_above(
    overlay: SourceDatasetOverlay,
    observed_value: Callable[[SourceObservation], float],
    threshold: float,
) -> _WholeUnitThresholdTally:
    """Classify whole units by whether any member observation exceeds a threshold."""

    selected_trajectories = tuple(
        trajectory
        for trajectory in overlay.trajectories
        if any(
            observed_value(observation) > threshold
            for observation in trajectory.observations
        )
    )
    return _WholeUnitThresholdTally(
        selected_trajectories=selected_trajectories,
        selection_unit=_SELECTION_UNIT_BY_SOURCE[overlay.source_name],
        threshold_exceeding_observation_count=sum(
            observed_value(observation) > threshold
            for trajectory in overlay.trajectories
            for observation in trajectory.observations
        ),
        threshold_equal_observation_count=sum(
            observed_value(observation) == threshold
            for trajectory in overlay.trajectories
            for observation in trajectory.observations
        ),
        threshold_equal_only_unit_count=sum(
            any(
                observed_value(observation) == threshold
                for observation in trajectory.observations
            )
            and not any(
                observed_value(observation) > threshold
                for observation in trajectory.observations
            )
            for trajectory in overlay.trajectories
        ),
        selected_observation_count=sum(
            len(trajectory.observations) for trajectory in selected_trajectories
        ),
        excluded_nonfinite_observation_count_within_selected_units=sum(
            trajectory.excluded_nonfinite_observation_count
            for trajectory in selected_trajectories
        ),
    )


def select_source_dataset_overlay_above_yield_threshold(
    overlay: SourceDatasetOverlay,
    *,
    threshold_t_ha: float,
) -> SourceDatasetOverlay:
    """Retain whole source units with any observed yield strictly above a threshold.

    LTCCE's selection unit is one replicate-specific trajectory. The PH combined
    source's selection unit is one linked source row; selecting that row retains
    all of its finite arms. Their connector is a visual aid and does not establish
    scientific comparability.
    """

    if (
        isinstance(threshold_t_ha, bool)
        or not isinstance(threshold_t_ha, (int, float))
        or not math.isfinite(float(threshold_t_ha))
        or float(threshold_t_ha) <= 0
    ):
        raise ValueError("Yield selection threshold must be a finite positive number")
    threshold = float(threshold_t_ha)
    tally = _tally_units_above(
        overlay,
        lambda observation: observation.yield_t_ha,
        threshold,
    )
    selected_trajectories = tally.selected_trajectories
    selection = SourceDatasetOverlaySelection(
        operator=">",
        threshold_t_ha=threshold,
        selection_unit=tally.selection_unit,
        selected_unit_count=len(selected_trajectories),
        total_unit_count=overlay.summary.trajectory_count,
        selected_observation_count=tally.selected_observation_count,
        total_observation_count=overlay.summary.finite_observation_count,
        threshold_exceeding_observation_count=tally.threshold_exceeding_observation_count,
        threshold_equal_observation_count=tally.threshold_equal_observation_count,
        threshold_equal_only_unit_count=tally.threshold_equal_only_unit_count,
        excluded_nonfinite_observation_count_within_selected_units=(
            tally.excluded_nonfinite_observation_count_within_selected_units
        ),
    )
    return _finalize_overlay(
        source_name=overlay.source_name,
        source_rows=overlay.summary.source_rows,
        excluded_observation_count=overlay.summary.excluded_observation_count,
        observations_by_trajectory={
            trajectory.trajectory_id: list(trajectory.observations)
            for trajectory in selected_trajectories
        },
        excluded_observations_by_trajectory={
            trajectory.trajectory_id: trajectory.excluded_nonfinite_observation_count
            for trajectory in selected_trajectories
        },
        selection=selection,
    )


def select_source_dataset_overlay_above_n_rate_threshold(
    overlay: SourceDatasetOverlay,
    *,
    threshold_kg_ha: float,
) -> SourceDatasetOverlay:
    """Retain whole source units with any applied N strictly above a threshold.

    The selection unit matches the yield selector's: one replicate-specific
    trajectory for LTCCE, one linked source row for the PH combined source. A
    unit is kept whole, so the low-N members of a high-N unit stay in the view
    and the trajectory remains readable as a series rather than as an isolated
    high-N point. The empty selection is a legitimate outcome for a source whose
    observed N range does not reach the threshold; callers decide whether that
    warrants a figure.
    """

    if (
        isinstance(threshold_kg_ha, bool)
        or not isinstance(threshold_kg_ha, (int, float))
        or not math.isfinite(float(threshold_kg_ha))
        or float(threshold_kg_ha) < 0
    ):
        raise ValueError(
            "Applied-N selection threshold must be a finite non-negative number"
        )
    threshold = float(threshold_kg_ha)
    tally = _tally_units_above(
        overlay,
        lambda observation: observation.n_rate_kg_ha,
        threshold,
    )
    selected_trajectories = tally.selected_trajectories
    selection = SourceDatasetNRateSelection(
        operator=">",
        threshold_kg_ha=threshold,
        selection_unit=tally.selection_unit,
        selected_unit_count=len(selected_trajectories),
        total_unit_count=overlay.summary.trajectory_count,
        selected_observation_count=tally.selected_observation_count,
        total_observation_count=overlay.summary.finite_observation_count,
        threshold_exceeding_observation_count=tally.threshold_exceeding_observation_count,
        threshold_equal_observation_count=tally.threshold_equal_observation_count,
        threshold_equal_only_unit_count=tally.threshold_equal_only_unit_count,
        excluded_nonfinite_observation_count_within_selected_units=(
            tally.excluded_nonfinite_observation_count_within_selected_units
        ),
    )
    return _finalize_overlay(
        source_name=overlay.source_name,
        source_rows=overlay.summary.source_rows,
        excluded_observation_count=overlay.summary.excluded_observation_count,
        observations_by_trajectory={
            trajectory.trajectory_id: list(trajectory.observations)
            for trajectory in selected_trajectories
        },
        excluded_observations_by_trajectory={
            trajectory.trajectory_id: trajectory.excluded_nonfinite_observation_count
            for trajectory in selected_trajectories
        },
        selection=selection,
    )


def stratify_source_dataset_overlay_by_zero_n_yield(
    overlay: SourceDatasetOverlay,
    *,
    threshold_t_ha: float,
) -> SourceDatasetZeroNStrata:
    """Split LTCCE trajectories by one distinct yield observed at exactly zero N."""

    if overlay.source_name != _LTCCE_SOURCE_NAME:
        raise ValueError("Zero-N yield stratification is defined only for LTCCE trajectories")
    if (
        isinstance(threshold_t_ha, bool)
        or not isinstance(threshold_t_ha, (int, float))
        or not math.isfinite(float(threshold_t_ha))
        or float(threshold_t_ha) <= 0
    ):
        raise ValueError("Zero-N yield threshold must be a finite positive number")
    threshold = float(threshold_t_ha)

    below_ids: list[str] = []
    above_ids: list[str] = []
    equal_ids: list[str] = []
    missing_ids: list[str] = []
    conflicting_ids: list[str] = []
    duplicate_identical_ids: list[str] = []
    trajectories_by_id = {
        trajectory.trajectory_id: trajectory for trajectory in overlay.trajectories
    }
    for trajectory in overlay.trajectories:
        baseline_observations = [
            observation.yield_t_ha
            for observation in trajectory.observations
            if observation.n_rate_kg_ha == 0.0
        ]
        baseline_yields = {
            yield_t_ha for yield_t_ha in baseline_observations
        }
        if not baseline_yields:
            missing_ids.append(trajectory.trajectory_id)
        elif len(baseline_yields) > 1:
            conflicting_ids.append(trajectory.trajectory_id)
        else:
            if len(baseline_observations) > 1:
                duplicate_identical_ids.append(trajectory.trajectory_id)
            baseline_yield = next(iter(baseline_yields))
            if baseline_yield < threshold:
                below_ids.append(trajectory.trajectory_id)
            elif baseline_yield > threshold:
                above_ids.append(trajectory.trajectory_id)
            else:
                equal_ids.append(trajectory.trajectory_id)

    def build_stratum(operator: str, trajectory_ids: Sequence[str]) -> SourceDatasetOverlay:
        selected = tuple(sorted(trajectory_ids))
        unconnected_ambiguous_ids = []
        for trajectory_id in selected:
            yields_by_n_rate: dict[float, set[float]] = {}
            for observation in trajectories_by_id[trajectory_id].observations:
                yields_by_n_rate.setdefault(observation.n_rate_kg_ha, set()).add(
                    observation.yield_t_ha
                )
            if any(len(yields) > 1 for yields in yields_by_n_rate.values()):
                unconnected_ambiguous_ids.append(trajectory_id)
        selected_observation_count = sum(
            len(trajectories_by_id[trajectory_id].observations)
            for trajectory_id in selected
        )
        selected_nonfinite_observation_count = sum(
            trajectories_by_id[trajectory_id].excluded_nonfinite_observation_count
            for trajectory_id in selected
        )
        selection = SourceDatasetZeroNSelection(
            operator=operator,
            threshold_t_ha=threshold,
            baseline_n_rate_kg_ha=0.0,
            selection_unit="replicate_specific_trajectory",
            selected_unit_count=len(selected),
            total_unit_count=overlay.summary.trajectory_count,
            selected_observation_count=selected_observation_count,
            total_observation_count=overlay.summary.finite_observation_count,
            equal_threshold_unit_count=len(equal_ids),
            missing_baseline_unit_count=len(missing_ids),
            conflicting_baseline_unit_count=len(conflicting_ids),
            duplicate_identical_baseline_unit_count=len(duplicate_identical_ids),
            excluded_nonfinite_observation_count_within_selected_units=(
                selected_nonfinite_observation_count
            ),
            unconnected_ambiguous_trajectory_ids=tuple(unconnected_ambiguous_ids),
        )
        return _finalize_overlay(
            source_name=overlay.source_name,
            source_rows=overlay.summary.source_rows,
            excluded_observation_count=overlay.summary.excluded_observation_count,
            observations_by_trajectory={
                trajectory_id: list(trajectories_by_id[trajectory_id].observations)
                for trajectory_id in selected
            },
            excluded_observations_by_trajectory={
                trajectory_id: trajectories_by_id[
                    trajectory_id
                ].excluded_nonfinite_observation_count
                for trajectory_id in selected
            },
            selection=selection,
        )

    return SourceDatasetZeroNStrata(
        below=build_stratum("<", below_ids),
        above=build_stratum(">", above_ids),
        equal_threshold_trajectory_ids=tuple(sorted(equal_ids)),
        missing_baseline_trajectory_ids=tuple(sorted(missing_ids)),
        conflicting_baseline_trajectory_ids=tuple(sorted(conflicting_ids)),
    )


def _adaptive_style(observation_count: int, trajectory_count: int) -> tuple[float, float, float]:
    """Scale marker size / marker alpha / line alpha down as data density grows."""

    marker_size = max(4.0, min(24.0, 6000.0 / max(observation_count, 1)))
    marker_alpha = max(0.25, min(0.85, 300.0 / max(observation_count, 1)))
    line_alpha = max(0.05, min(0.35, 20.0 / max(trajectory_count, 1)))
    return marker_size, marker_alpha, line_alpha


def create_source_dataset_overlay_figure(overlay: SourceDatasetOverlay):
    """Build a descriptive source-dataset overlay figure (no file I/O)."""

    if not overlay.trajectories:
        raise ValueError("A source-dataset overlay requires at least one finite observation")

    marker_size, marker_alpha, line_alpha = _adaptive_style(
        overlay.summary.finite_observation_count,
        overlay.summary.trajectory_count,
    )

    figure, axes = plt.subplots(figsize=(10, 7), constrained_layout=True)

    for treatment_class in overlay.summary.treatment_classes:
        xs = []
        ys = []
        for trajectory in overlay.trajectories:
            for observation in trajectory.observations:
                if observation.treatment_class == treatment_class:
                    xs.append(observation.n_rate_kg_ha)
                    ys.append(observation.yield_t_ha)
        axes.scatter(
            xs,
            ys,
            s=marker_size,
            alpha=marker_alpha,
            label=treatment_class,
            zorder=3,
        )

    if overlay.source_name in {_LTCCE_SOURCE_NAME, _COMBINED_SOURCE_NAME}:
        line_label = "within-trajectory connecting lines (visual aid; not a fit)"
        unconnected_ids = (
            set(overlay.selection.unconnected_ambiguous_trajectory_ids)
            if isinstance(overlay.selection, SourceDatasetZeroNSelection)
            else set()
        )
        line_label_emitted = False
        for trajectory in overlay.trajectories:
            if trajectory.trajectory_id in unconnected_ids:
                continue
            axes.plot(
                [observation.n_rate_kg_ha for observation in trajectory.observations],
                [observation.yield_t_ha for observation in trajectory.observations],
                label=line_label if not line_label_emitted else "_nolegend_",
                color="grey",
                linewidth=1,
                alpha=line_alpha,
                zorder=1,
            )
            line_label_emitted = True

    if overlay.source_name == _COMBINED_SOURCE_NAME:
        plot_description_lines = (
            f"source={display_source_name(overlay.source_name)} — linked-arm trajectories",
            "comparability is not assumed; connecting lines are visual aids",
            "no pooled curve or fit",
        )
    elif overlay.source_name == _CORE_TRIAL_SOURCE_NAME:
        plot_description_lines = (
            f"source={display_source_name(overlay.source_name)} — independent source observations",
            "response-series identity is not inferred from the raw CSV; no connecting lines",
        )
    else:
        plot_description_lines = (
            f"source={display_source_name(overlay.source_name)} — replicate-specific trajectories",
            "connecting lines are visual aids; no pooled curve or fit",
        )

    if overlay.selection is None:
        selection_lines: tuple[str, ...] = ()
        count_line = (
            f"trajectories={overlay.summary.trajectory_count}; "
            f"observations={overlay.summary.finite_observation_count}"
        )
    elif isinstance(overlay.selection, SourceDatasetZeroNSelection):
        selection_lines = (
            f"observed yield {overlay.selection.operator} "
            f"{overlay.selection.threshold_t_ha:g} t/ha at exactly "
            f"{overlay.selection.baseline_n_rate_kg_ha:g} kg N/ha",
            "all finite observations retained; nonfinite excluded="
            f"{overlay.selection.excluded_nonfinite_observation_count_within_selected_units}; "
            "unconnected ambiguous="
            f"{len(overlay.selection.unconnected_ambiguous_trajectory_ids)}",
            f"equal={overlay.selection.equal_threshold_unit_count}; "
            f"missing zero N={overlay.selection.missing_baseline_unit_count}; "
            f"conflicting zero N={overlay.selection.conflicting_baseline_unit_count}; "
            "duplicate-identical zero N="
            f"{overlay.selection.duplicate_identical_baseline_unit_count}",
        )
        count_line = (
            "selected replicate trajectories="
            f"{overlay.selection.selected_unit_count}/{overlay.selection.total_unit_count}; "
            f"observations={overlay.selection.selected_observation_count}/"
            f"{overlay.selection.total_observation_count}"
        )
    elif isinstance(overlay.selection, SourceDatasetNRateSelection):
        retained_unit = _RETAINED_UNIT_LABELS[overlay.selection.selection_unit]
        retained_member = (
            "arms"
            if overlay.selection.selection_unit == "linked_source_row"
            else "points"
        )
        selection_lines = (
            f"selection=whole {retained_unit} with any applied N "
            f"> {overlay.selection.threshold_kg_ha:g} kg N/ha",
            f"all finite {retained_member} retained within selected {retained_unit}, "
            "including their below-threshold N rates",
            "points above threshold="
            f"{overlay.selection.threshold_exceeding_observation_count}; "
            f"points at exactly {overlay.selection.threshold_kg_ha:g} kg N/ha="
            f"{overlay.selection.threshold_equal_observation_count}",
        )
        count_line = (
            f"selected {retained_unit}={overlay.selection.selected_unit_count}/"
            f"{overlay.selection.total_unit_count}; observations="
            f"{overlay.selection.selected_observation_count}/"
            f"{overlay.selection.total_observation_count}"
        )
    else:
        retained_unit = _RETAINED_UNIT_LABELS[overlay.selection.selection_unit]
        retained_member = (
            "arms"
            if overlay.selection.selection_unit == "linked_source_row"
            else "points"
        )
        selection_lines = (
            f"selection=whole {retained_unit} with any observed yield "
            f"> {overlay.selection.threshold_t_ha:g} t/ha",
            f"all finite {retained_member} retained within selected {retained_unit}",
        )
        count_line = (
            f"selected {retained_unit}={overlay.selection.selected_unit_count}/"
            f"{overlay.selection.total_unit_count}; observations="
            f"{overlay.selection.selected_observation_count}/"
            f"{overlay.selection.total_observation_count}"
        )

    n_lower, n_upper = overlay.summary.n_rate_range_kg_ha
    axes.set_xlabel("Applied N (kg N/ha)")
    axes.set_ylabel("Grain yield (t/ha)")
    axes.set_title(
        "\n".join(
            (
                *plot_description_lines,
                *selection_lines,
                count_line,
                f"N range={n_lower:g}-{n_upper:g} kg N/ha",
            )
        )
    )
    axes.legend(loc="best", fontsize=8)
    return figure, axes


def write_source_dataset_overlay_figure(overlay: SourceDatasetOverlay, destination: str | Path) -> Path:
    """Render and save one JPEG to the explicit destination the caller supplies."""

    destination = Path(destination)
    if destination.suffix.lower() != ".jpeg":
        raise ValueError("Source-dataset overlay figures are written as .jpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, _axes = create_source_dataset_overlay_figure(overlay)
    try:
        figure.savefig(destination, format="jpeg", dpi=150)
    finally:
        plt.close(figure)
    return destination


__all__ = [
    "SUPPORTED_SOURCE_NAMES",
    "SourceDatasetNRateSelection",
    "SourceDatasetOverlay",
    "SourceDatasetOverlaySelection",
    "SourceDatasetOverlaySummary",
    "SourceDatasetZeroNSelection",
    "SourceDatasetZeroNStrata",
    "SourceObservation",
    "SourceTrajectory",
    "create_source_dataset_overlay_figure",
    "read_source_dataset_overlay",
    "select_source_dataset_overlay_above_n_rate_threshold",
    "select_source_dataset_overlay_above_yield_threshold",
    "stratify_source_dataset_overlay_by_zero_n_yield",
    "write_source_dataset_overlay_figure",
]
