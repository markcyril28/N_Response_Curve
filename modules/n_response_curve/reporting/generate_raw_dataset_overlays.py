#!/usr/bin/env python3
"""Generate source-dataset overlay views with governed core-series links."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tomllib
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.reporting.source_dataset_overlays import (  # noqa: E402
    COMBINED_SOURCE_VARIANTS,
    COMBINED_VARIANT_NO_FP,
    COMBINED_VARIANT_WITH_FP,
    SourceDatasetOverlay,
    SourceDatasetZeroNStrata,
    combined_variant_source_path,
    read_source_dataset_overlay,
    select_source_dataset_overlay_above_n_rate_threshold,
    select_source_dataset_overlay_above_yield_threshold,
    stratify_source_dataset_overlay_by_zero_n_yield,
    write_source_dataset_overlay_figure,
)
from n_response_curve.analysis.values import finite_number  # noqa: E402
from n_response_curve.reporting.generate_custom_overlays import (  # noqa: E402
    YieldThresholdSelection,
    _parse_checksum_ledger,
    _verify_bound_files,
    resolve_v3_governed_series_uids,
    select_series_with_any_yield_above,
)
from n_response_curve.reporting.plots import (  # noqa: E402
    create_source_series_overlay_figure,
)
from n_response_curve.reporting.source_display_names import (  # noqa: E402
    display_source_name,
)
from n_response_curve.reporting.directory_publication import (  # noqa: E402
    copy_preserved_plain_tree as _copy_preserved_plain_tree,
    plain_absolute_path as _plain_absolute_path,
    promote_staged_directory as _promote_staged_directory,
    publication_lock as _core_overlay_publication_lock,
    recover_interrupted_directory_publication as _recover_interrupted_directory_publication,  # noqa: E501
)
from n_response_curve.reporting.figure_output import (  # noqa: E402
    save_figure_atomically,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scriptCONFIG.toml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "WF/03_Response_Curves"
DEFAULT_YIELD_THRESHOLD_T_HA = 7.8
DEFAULT_ZERO_N_YIELD_THRESHOLD_T_HA = 5.0
DEFAULT_N_RATE_THRESHOLD_KG_HA = 250.0
_COMBINED_SOURCE_NAME = "ph_combined_nopt_rcm"
# A non-default variant writes into its own nested directory rather than
# alongside the registered run's figures: the two runs share figure filenames,
# and _retire_superseded_n_rate_figures only sweeps loose files in one
# directory, so a nested directory is what survives a later default run.
# Keyed by the variant token itself so the directory cannot drift from it.
_VARIANT_SUBDIRECTORY_NAMES = {COMBINED_VARIANT_NO_FP: COMBINED_VARIANT_NO_FP}
_SOURCE_NAMES = (
    "core_trial_data",
    "ltcce",
    "ph_combined_nopt_rcm",
)
_CORE_OUTPUT_DIRECTORY_NAME = "literature_extracted_dataset"
# Owned by the trajectory- and variety-cluster generators. Tolerated in the
# LTCCE destination and carried across this generator's snapshot replacement.
_PRESERVED_SUBDIRECTORY = "clusters"
_REQUIRED_CORE_LEDGER_COLUMNS = frozenset(
    {
        "source_name",
        "source_sha256",
        "data_classification",
        "response_series_uid",
        "record_uid",
        "n_rate_kg_ha",
        "yield_t_ha",
        "treatment_text_class",
    }
)


@dataclass(frozen=True)
class GovernedCoreOverlayInputs:
    records: tuple[dict[str, str], ...]
    response_series_uids: tuple[str, ...]
    selection: YieldThresholdSelection


def _preserve_cluster_tree(destination_dir: Path, staging_dir: Path) -> None:
    preserved = destination_dir / _PRESERVED_SUBDIRECTORY
    if preserved.is_symlink():
        raise RuntimeError(f"Preserved cluster tree is a symlink: {preserved}")
    if not preserved.exists():
        return
    if not preserved.is_dir():
        raise RuntimeError(f"Preserved cluster tree is not a directory: {preserved}")
    _copy_preserved_plain_tree(preserved, staging_dir / _PRESERVED_SUBDIRECTORY)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"Duplicate JSON key in governed run manifest: {key!r}")
        result[key] = value
    return result


def _load_governed_core_inputs(
    config_path: Path,
    *,
    yield_threshold_t_ha: float,
    require_yield_selection: bool = True,
    package_path: Path | None = None,
) -> GovernedCoreOverlayInputs:
    """Load the exact governed core-series membership used by configured overlays.

    *package_path* overrides the ``[custom_overlays].package_path`` binding, for
    operators reading a package that has been moved out of the configured
    release target. It is resolved and containment-checked exactly as the
    configured path is, and the manifest and ledger it names are still verified
    against the package's own ``CHECKSUMS.sha256``: this relocates the input,
    it does not relax the binding.
    """

    project_root = config_path.resolve().parent
    if package_path is None:
        with config_path.open("rb") as handle:
            config = tomllib.load(handle)
        custom_plots = config.get("custom_overlays")
        if not isinstance(custom_plots, dict):
            custom_plots = config.get("custom_plots")
        if not isinstance(custom_plots, dict):
            raise ValueError(
                "The configuration must contain a [custom_overlays] or [custom_plots] table"
            )
        raw_package_path = custom_plots.get("package_path")
        if not isinstance(raw_package_path, str) or not raw_package_path.strip():
            raise ValueError("The custom-overlay package_path must be a nonempty string")
        package = Path(raw_package_path)
    else:
        package = Path(package_path)
    if not package.is_absolute():
        package = project_root / package
    package = package.resolve()
    try:
        package.relative_to(project_root)
    except ValueError as exc:
        raise RuntimeError("The governed custom-overlay package is outside the project") from exc

    manifest_path = package / "run_manifest.json"
    checksums_path = package / "CHECKSUMS.sha256"
    ledger_path = package / "tables/quality/analysis_eligibility_ledger.csv"
    missing = [
        str(path)
        for path in (manifest_path, checksums_path, ledger_path)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(f"Missing governed core-overlay input(s): {missing}")

    release_entries = _parse_checksum_ledger(checksums_path)
    required_relative_paths = (
        manifest_path.relative_to(package).as_posix(),
        ledger_path.relative_to(package).as_posix(),
    )
    missing_bindings = [
        relative for relative in required_relative_paths if relative not in release_entries
    ]
    if missing_bindings:
        raise RuntimeError(
            f"Governed core-overlay inputs are not bound by CHECKSUMS.sha256: {missing_bindings}"
        )
    bound_inputs = {
        relative: release_entries[relative] for relative in required_relative_paths
    }
    _verify_bound_files(package, bound_inputs)

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_json_object,
    )
    figure_inventory = manifest.get("figures")
    if not isinstance(figure_inventory, dict):
        raise RuntimeError("The governed release figure inventory is absent")
    series_uids = resolve_v3_governed_series_uids(
        figure_inventory,
        source_name="core_trial_data",
    )
    series_set = set(series_uids)
    records: list[dict[str, str]] = []
    governed_record_uids: set[str] = set()
    observations_by_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with ledger_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing_columns = sorted(_REQUIRED_CORE_LEDGER_COLUMNS - fieldnames)
        if missing_columns:
            raise RuntimeError(
                f"Governed eligibility-ledger schema is missing columns: {missing_columns}"
            )
        for row in reader:
            series_uid = str(row.get("response_series_uid") or "")
            if row.get("source_name") != "core_trial_data" or series_uid not in series_set:
                continue
            record_uid = str(row.get("record_uid") or "").strip()
            if not record_uid:
                raise RuntimeError("A governed core observation has no record_uid")
            if record_uid in governed_record_uids:
                raise RuntimeError(f"Duplicate governed record_uid: {record_uid}")
            governed_record_uids.add(record_uid)
            n_rate = finite_number(row.get("n_rate_kg_ha"))
            yield_value = finite_number(row.get("yield_t_ha"))
            if n_rate is None or yield_value is None:
                continue
            records.append(dict(row))
            observations_by_series[series_uid].append((n_rate, yield_value))
    if set(observations_by_series) != series_set:
        raise RuntimeError("Finite governed observations do not cover the core series inventory")
    _verify_bound_files(package, bound_inputs)

    selection = select_series_with_any_yield_above(
        observations_by_series,
        yield_threshold_t_ha=yield_threshold_t_ha,
    )
    if require_yield_selection and not selection.response_series_uids:
        raise RuntimeError(
            f"No governed core response series has observed yield > {yield_threshold_t_ha:g} t/ha"
        )
    return GovernedCoreOverlayInputs(
        records=tuple(records),
        response_series_uids=series_uids,
        selection=selection,
    )


def _save_governed_figure(figure: Any, destination: Path) -> None:
    """Write one governed figure through the shared atomic save."""

    save_figure_atomically(figure, destination)


def _write_governed_core_figures_unlocked(
    inputs: GovernedCoreOverlayInputs,
    *,
    source_wide_path: Path,
    selected_path: Path,
    yield_threshold_t_ha: float,
) -> None:
    """Write core overlays using only manifest-governed response-series links."""

    from matplotlib import pyplot as plt

    source_wide_path = source_wide_path.resolve()
    selected_path = selected_path.resolve()
    destination_dir = source_wide_path.parent
    if selected_path.parent != destination_dir:
        raise ValueError("Governed core overlay outputs must share one directory")
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = destination_dir.with_name(
        f".{destination_dir.name}.staging.{uuid.uuid4().hex}"
    )
    backup_dir = destination_dir.with_name(
        f".{destination_dir.name}.backup.{uuid.uuid4().hex}"
    )
    staging_dir.mkdir()

    try:
        source_figure, _ = create_source_series_overlay_figure(
            inputs.records,
            "core_trial_data",
            response_series_uids=inputs.response_series_uids,
            display_name=display_source_name("core_trial_data"),
        )
        try:
            _save_governed_figure(source_figure, staging_dir / source_wide_path.name)
        finally:
            plt.close(source_figure)

        selected_figure, selected_axes = create_source_series_overlay_figure(
            inputs.records,
            "core_trial_data",
            response_series_uids=inputs.selection.response_series_uids,
            display_name=display_source_name("core_trial_data"),
        )
        n_range = inputs.selection.n_rate_range_kg_ha
        assert n_range is not None
        selected_axes.set_title(
            "\n".join(
                (
                    f"source={display_source_name('core_trial_data')}",
                    f"series with any observed grain yield > {yield_threshold_t_ha:g} t/ha",
                    "full series trajectories; descriptive overlay (no pooled curve or fit)",
                    f"series={len(inputs.selection.response_series_uids)}; "
                    f"observations={len(inputs.selection.selected_observations)}; "
                    "threshold-exceeding points="
                    f"{len(inputs.selection.threshold_exceeding_observations)}; "
                    f"N range={n_range[0]:g}-{n_range[1]:g} kg/ha",
                )
            )
        )
        try:
            _save_governed_figure(selected_figure, staging_dir / selected_path.name)
        finally:
            plt.close(selected_figure)

        # Same reason the LTCCE branch below carries it: this destination is
        # replaced as a whole snapshot, so the separately generated cluster
        # views beneath it have to be copied into staging or the replacement
        # silently discards them. The core `clusters/` subtree is owned by the
        # planting-year and recorded-season companion generators, not by this
        # generator.
        _preserve_cluster_tree(destination_dir, staging_dir)

        _promote_staged_directory(staging_dir, destination_dir, backup_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def _write_governed_core_figures(
    inputs: GovernedCoreOverlayInputs,
    *,
    source_wide_path: Path,
    selected_path: Path,
    yield_threshold_t_ha: float,
) -> None:
    """Write the governed pair while excluding nested cluster publishers."""

    source_wide_path = _plain_absolute_path(
        source_wide_path,
        label="Governed core source-wide output",
    )
    selected_path = _plain_absolute_path(
        selected_path,
        label="Governed core selected-series output",
    )
    destination = source_wide_path.parent
    if selected_path.parent != destination:
        raise ValueError("Governed core overlay outputs must share one directory")
    with _core_overlay_publication_lock(destination):
        _recover_interrupted_directory_publication(destination)
        _write_governed_core_figures_unlocked(
            inputs,
            source_wide_path=source_wide_path,
            selected_path=selected_path,
            yield_threshold_t_ha=yield_threshold_t_ha,
        )


def _write_ltcce_figures_unlocked(
    overlay: SourceDatasetOverlay,
    selected: SourceDatasetOverlay,
    zero_n_strata: SourceDatasetZeroNStrata,
    *,
    destination_dir: Path,
    yield_threshold_t_ha: float,
    zero_n_yield_threshold_t_ha: float,
    n_rate_selected: SourceDatasetOverlay | None = None,
    n_rate_threshold_kg_ha: float = DEFAULT_N_RATE_THRESHOLD_KG_HA,
) -> int:
    """Stage and promote the LTCCE overlays as one directory snapshot.

    The applied-N view is written only when the source has a unit above that
    rate. Every applied-N view this generator has ever written for the source
    counts as managed, so re-running at a different rate retires the old one
    instead of leaving two views that disagree about which units qualify.
    """

    destination_dir = destination_dir.resolve()
    yield_token = _threshold_token(yield_threshold_t_ha)
    zero_n_token = _threshold_token(zero_n_yield_threshold_t_ha)
    n_rate_name = _n_rate_figure_name("ltcce", n_rate_threshold_kg_ha)
    outputs = (
        ("ltcce_source_wide.jpeg", overlay),
        (
            f"ltcce_series_with_yield_above_{yield_token}_t_ha.jpeg",
            selected,
        ),
        (f"ltcce_zero_n_below_{zero_n_token}_t_ha.jpeg", zero_n_strata.below),
        (f"ltcce_zero_n_above_{zero_n_token}_t_ha.jpeg", zero_n_strata.above),
        *(
            ((n_rate_name, n_rate_selected),)
            if n_rate_selected is not None and n_rate_selected.trajectories
            else ()
        ),
    )
    expected_names = {name for name, _ in outputs} | {n_rate_name}
    if destination_dir.exists():
        if not destination_dir.is_dir() or destination_dir.is_symlink():
            raise RuntimeError("LTCCE overlay destination is not a plain directory")
        expected_names |= {
            entry.name
            for entry in destination_dir.iterdir()
            if _is_n_rate_figure_name("ltcce", entry.name)
        }
        unexpected_names = sorted(
            entry.name
            for entry in destination_dir.iterdir()
            if entry.name not in expected_names
            and entry.name != _PRESERVED_SUBDIRECTORY
        )
        if unexpected_names:
            raise RuntimeError(
                "LTCCE overlay destination contains unmanaged entries: "
                f"{unexpected_names}"
            )
        if any(
            entry.is_symlink()
            or not (entry.is_file() or entry.name == _PRESERVED_SUBDIRECTORY)
            for entry in destination_dir.iterdir()
        ):
            raise RuntimeError("LTCCE overlay destination contains a non-regular entry")

    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = destination_dir.with_name(
        f".{destination_dir.name}.staging.{uuid.uuid4().hex}"
    )
    backup_dir = destination_dir.with_name(
        f".{destination_dir.name}.backup.{uuid.uuid4().hex}"
    )
    staging_dir.mkdir()
    try:
        for name, selected_overlay in outputs:
            write_source_dataset_overlay_figure(selected_overlay, staging_dir / name)

        # This directory is replaced as a whole snapshot, so the separately
        # generated cluster views beneath it have to be carried across or they
        # would be discarded. The organized `by_trajectory` and `by_variety`
        # products are owned by their dedicated generators, not by this one.
        _preserve_cluster_tree(destination_dir, staging_dir)

        _promote_staged_directory(staging_dir, destination_dir, backup_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
    return len(outputs)


def _write_ltcce_figures(
    overlay: SourceDatasetOverlay,
    selected: SourceDatasetOverlay,
    zero_n_strata: SourceDatasetZeroNStrata,
    *,
    destination_dir: Path,
    yield_threshold_t_ha: float,
    zero_n_yield_threshold_t_ha: float,
    n_rate_selected: SourceDatasetOverlay | None = None,
    n_rate_threshold_kg_ha: float = DEFAULT_N_RATE_THRESHOLD_KG_HA,
) -> int:
    """Serialize recovery and publication of one complete LTCCE snapshot."""

    destination_dir = _plain_absolute_path(
        destination_dir,
        label="LTCCE overlay publication destination",
    )
    with _core_overlay_publication_lock(destination_dir):
        _recover_interrupted_directory_publication(destination_dir)
        return _write_ltcce_figures_unlocked(
            overlay,
            selected,
            zero_n_strata,
            destination_dir=destination_dir,
            yield_threshold_t_ha=yield_threshold_t_ha,
            zero_n_yield_threshold_t_ha=zero_n_yield_threshold_t_ha,
            n_rate_selected=n_rate_selected,
            n_rate_threshold_kg_ha=n_rate_threshold_kg_ha,
        )


def _load_source_specs(config_path: Path) -> dict[str, tuple[Path, str]]:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    sources = config.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("The configuration must contain a [sources] table")

    specs: dict[str, tuple[Path, str]] = {}
    for source_name in _SOURCE_NAMES:
        source = sources.get(source_name)
        if not isinstance(source, dict):
            raise ValueError(f"The configuration is missing [sources.{source_name}]")
        raw_path = source.get("data_path")
        encoding = source.get("encoding")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"[sources.{source_name}].data_path must be a nonempty string")
        if not isinstance(encoding, str) or not encoding.strip():
            raise ValueError(f"[sources.{source_name}].encoding must be a nonempty string")
        source_path = Path(raw_path)
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        specs[source_name] = (source_path, encoding)
    return specs


def _threshold_token(threshold: float) -> str:
    return f"{threshold:g}".replace(".", "_")


def _n_rate_figure_name(source_name: str, threshold_kg_ha: float) -> str:
    return (
        f"{source_name}_series_with_n_rate_above_"
        f"{_threshold_token(threshold_kg_ha)}_kg_n_ha.jpeg"
    )


def _is_n_rate_figure_name(source_name: str, name: str) -> bool:
    """Recognize this generator's applied-N view for a source at any rate."""

    return (
        re.fullmatch(
            rf"{re.escape(source_name)}_series_with_n_rate_above_"
            r"\d+(?:_\d+)?_kg_n_ha\.jpeg",
            name,
        )
        is not None
    )


def _retire_superseded_n_rate_figures(
    directory: Path,
    source_name: str,
    *,
    keep_name: str | None,
) -> None:
    """Drop applied-N views this generator wrote for the source at other rates."""

    if not directory.is_dir():
        return
    for entry in directory.iterdir():
        if entry.name == keep_name or not _is_n_rate_figure_name(source_name, entry.name):
            continue
        if entry.is_file() and not entry.is_symlink():
            entry.unlink()


def _source_n_rate_report(
    overlay: SourceDatasetOverlay,
    n_rate_selected: SourceDatasetOverlay,
    *,
    threshold_kg_ha: float,
) -> str:
    """State the applied-N isolation outcome, including the empty one."""

    observed_range = overlay.summary.n_rate_range_kg_ha
    observed_max = observed_range[1] if observed_range is not None else float("nan")
    selection = n_rate_selected.selection
    assert selection is not None
    if not n_rate_selected.trajectories:
        return (
            f"{overlay.source_name}: no unit exceeds {threshold_kg_ha:g} kg N/ha "
            f"(observed maximum={observed_max:g} kg N/ha); no applied-N view written"
        )
    return (
        f"{overlay.source_name}: units above {threshold_kg_ha:g} kg N/ha="
        f"{selection.selected_unit_count}/{selection.total_unit_count}; "
        f"observations retained={selection.selected_observation_count}/"
        f"{selection.total_observation_count}; points above threshold="
        f"{selection.threshold_exceeding_observation_count}; observed maximum="
        f"{observed_max:g} kg N/ha"
    )


def _core_n_rate_report(
    inputs: GovernedCoreOverlayInputs,
    *,
    threshold_kg_ha: float,
) -> str:
    """State whether any manifest-governed core series reaches the applied-N threshold."""

    rates_by_series: dict[str, list[float]] = defaultdict(list)
    for row in inputs.records:
        n_rate = finite_number(row.get("n_rate_kg_ha"))
        if n_rate is not None:
            rates_by_series[str(row.get("response_series_uid") or "")].append(n_rate)
    selected = sorted(
        series_uid
        for series_uid, rates in rates_by_series.items()
        if any(rate > threshold_kg_ha for rate in rates)
    )
    observed_max = max(
        (rate for rates in rates_by_series.values() for rate in rates),
        default=float("nan"),
    )
    if not selected:
        return (
            f"core_trial_data: no governed response series exceeds "
            f"{threshold_kg_ha:g} kg N/ha (observed maximum={observed_max:g} kg N/ha); "
            "no applied-N view written"
        )
    return (
        f"core_trial_data: governed response series above {threshold_kg_ha:g} kg N/ha="
        f"{len(selected)}/{len(rates_by_series)}; observed maximum={observed_max:g} kg N/ha; "
        "the governed core writer emits only its reviewed figure pair, so no "
        "applied-N view is written here"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--yield-threshold",
        type=float,
        default=DEFAULT_YIELD_THRESHOLD_T_HA,
    )
    parser.add_argument(
        "--zero-n-yield-threshold",
        type=float,
        default=DEFAULT_ZERO_N_YIELD_THRESHOLD_T_HA,
    )
    parser.add_argument(
        "--n-rate-threshold",
        type=float,
        default=DEFAULT_N_RATE_THRESHOLD_KG_HA,
        help=(
            "Isolate whole source units with any applied N strictly above this "
            "rate in kg N/ha; sources that never reach it get no such view"
        ),
    )
    parser.add_argument(
        "--fp-variant",
        choices=COMBINED_SOURCE_VARIANTS,
        default=COMBINED_VARIANT_WITH_FP,
        help=(
            "Which file of the PH combined source to read. The default reads "
            "the registered CSV and writes every source's views as before. "
            f"'{COMBINED_VARIANT_NO_FP}' reads its Farmer's-Practice-free "
            "sibling and writes only that source's views, into a nested "
            f"'{_VARIANT_SUBDIRECTORY_NAMES[COMBINED_VARIANT_NO_FP]}/' "
            "directory so neither run can overwrite the other"
        ),
    )
    return parser.parse_args()


def _write_combined_variant_figures(
    args: argparse.Namespace,
    source_specs: dict[str, tuple[Path, str]],
) -> int:
    """Write the PH combined views from a non-default source-file variant."""

    source_name = _COMBINED_SOURCE_NAME
    registered_path, encoding = source_specs[source_name]
    source_path = combined_variant_source_path(registered_path, args.fp_variant)
    if not source_path.is_file():
        raise ValueError(
            f"Source '{source_name}' variant {args.fp_variant!r} expects "
            f"{source_path}, which does not exist"
        )

    overlay = read_source_dataset_overlay(
        source_path,
        source_name,
        encoding=encoding,
        variant=args.fp_variant,
    )
    selected = select_source_dataset_overlay_above_yield_threshold(
        overlay,
        threshold_t_ha=args.yield_threshold,
    )
    n_rate_selected = select_source_dataset_overlay_above_n_rate_threshold(
        overlay,
        threshold_kg_ha=args.n_rate_threshold,
    )

    token = _threshold_token(args.yield_threshold)
    destination_dir = (
        args.output_dir
        / source_name
        / _VARIANT_SUBDIRECTORY_NAMES[args.fp_variant]
    )
    write_source_dataset_overlay_figure(
        overlay,
        destination_dir / f"{source_name}_source_wide.jpeg",
    )
    write_source_dataset_overlay_figure(
        selected,
        destination_dir / f"{source_name}_series_with_yield_above_{token}_t_ha.jpeg",
    )
    written = 2
    n_rate_name = None
    if n_rate_selected.trajectories:
        n_rate_name = _n_rate_figure_name(source_name, args.n_rate_threshold)
        write_source_dataset_overlay_figure(n_rate_selected, destination_dir / n_rate_name)
        written += 1
    _retire_superseded_n_rate_figures(destination_dir, source_name, keep_name=n_rate_name)

    print(f"{source_name}: source variant={args.fp_variant}; file={source_path.name}")
    print(
        f"{source_name}: treatment classes="
        f"{', '.join(overlay.summary.treatment_classes)}"
    )
    print(
        f"{source_name}: source-wide observations="
        f"{overlay.summary.finite_observation_count}; selected observations="
        f"{selected.summary.finite_observation_count}"
    )
    print(
        _source_n_rate_report(
            overlay,
            n_rate_selected,
            threshold_kg_ha=args.n_rate_threshold,
        )
    )
    print(
        f"Generated {written} source-dataset overlay views under {destination_dir}"
    )
    return 0


def main() -> int:
    args = _parse_args()
    source_specs = _load_source_specs(args.config)
    if args.fp_variant != COMBINED_VARIANT_WITH_FP:
        return _write_combined_variant_figures(args, source_specs)
    token = _threshold_token(args.yield_threshold)
    governed_core = _load_governed_core_inputs(
        args.config,
        yield_threshold_t_ha=args.yield_threshold,
    )
    prepared = []
    ltcce_zero_n_strata = None

    for source_name in ("ltcce", "ph_combined_nopt_rcm"):
        source_path, encoding = source_specs[source_name]
        overlay = read_source_dataset_overlay(
            source_path,
            source_name,
            encoding=encoding,
        )
        selected = select_source_dataset_overlay_above_yield_threshold(
            overlay,
            threshold_t_ha=args.yield_threshold,
        )
        n_rate_selected = select_source_dataset_overlay_above_n_rate_threshold(
            overlay,
            threshold_kg_ha=args.n_rate_threshold,
        )
        if source_name == "ltcce":
            ltcce_zero_n_strata = stratify_source_dataset_overlay_by_zero_n_yield(
                overlay,
                threshold_t_ha=args.zero_n_yield_threshold,
            )
        prepared.append((source_name, overlay, selected, n_rate_selected))

    core_dir = args.output_dir / _CORE_OUTPUT_DIRECTORY_NAME
    _write_governed_core_figures(
        governed_core,
        source_wide_path=core_dir / "core_trial_data_source_wide.jpeg",
        selected_path=(
            core_dir / f"core_trial_data_series_with_yield_above_{token}_t_ha.jpeg"
        ),
        yield_threshold_t_ha=args.yield_threshold,
    )
    written_figure_count = 2
    print(
        "core_trial_data: governed response-series observations="
        f"{len(governed_core.records)}; selected observations="
        f"{len(governed_core.selection.selected_observations)}"
    )
    print(_core_n_rate_report(governed_core, threshold_kg_ha=args.n_rate_threshold))

    for source_name, overlay, selected, n_rate_selected in prepared:
        source_dir = args.output_dir / source_name
        if source_name == "ltcce":
            assert ltcce_zero_n_strata is not None
            written_figure_count += _write_ltcce_figures(
                overlay,
                selected,
                ltcce_zero_n_strata,
                destination_dir=source_dir,
                yield_threshold_t_ha=args.yield_threshold,
                zero_n_yield_threshold_t_ha=args.zero_n_yield_threshold,
                n_rate_selected=n_rate_selected,
                n_rate_threshold_kg_ha=args.n_rate_threshold,
            )
        else:
            write_source_dataset_overlay_figure(
                overlay,
                source_dir / f"{source_name}_source_wide.jpeg",
            )
            write_source_dataset_overlay_figure(
                selected,
                source_dir
                / f"{source_name}_series_with_yield_above_{token}_t_ha.jpeg",
            )
            written_figure_count += 2
            n_rate_name = None
            if n_rate_selected.trajectories:
                n_rate_name = _n_rate_figure_name(source_name, args.n_rate_threshold)
                write_source_dataset_overlay_figure(
                    n_rate_selected,
                    source_dir / n_rate_name,
                )
                written_figure_count += 1
            _retire_superseded_n_rate_figures(
                source_dir,
                source_name,
                keep_name=n_rate_name,
            )
        print(
            f"{source_name}: source-wide observations="
            f"{overlay.summary.finite_observation_count}; selected observations="
            f"{selected.summary.finite_observation_count}"
        )
        print(
            _source_n_rate_report(
                overlay,
                n_rate_selected,
                threshold_kg_ha=args.n_rate_threshold,
            )
        )

    print(
        f"Generated {written_figure_count} source-dataset overlay views "
        f"under {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
