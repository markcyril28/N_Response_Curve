#!/usr/bin/env python3
"""Render a source-dataset overlay with one or more treatment-class arms excluded.

Ad hoc diagnostic variant of generate_raw_dataset_overlays.py's
"*_source_wide.jpeg" figure: same read/render path, but observations for the
given --exclude-treatment-class are dropped before the overlay is finalized,
so the excluded arm never enters the scatter, the legend, or the connecting
lines. A linked row whose only finite arm was the excluded one drops out of
the trajectory count entirely, rather than being drawn as an isolated point.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.reporting import source_dataset_overlays as sdo  # noqa: E402

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scriptCONFIG.toml"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "WF/04_Response_Curves/z_n_response_full/overlay/source_dataset"
)


def _load_source_spec(config_path: Path, source_name: str) -> tuple[Path, str]:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    sources = config.get("sources")
    if not isinstance(sources, dict) or source_name not in sources:
        raise ValueError(f"The configuration is missing [sources.{source_name}]")
    source = sources[source_name]
    raw_path = source.get("data_path")
    encoding = source.get("encoding")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"[sources.{source_name}].data_path must be a nonempty string")
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError(f"[sources.{source_name}].encoding must be a nonempty string")
    source_path = Path(raw_path)
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    return source_path, encoding


def _exclude_treatment_classes(
    overlay: sdo.SourceDatasetOverlay,
    excluded_classes: set[str],
) -> sdo.SourceDatasetOverlay:
    observations_by_trajectory: dict[str, list[sdo.SourceObservation]] = {}
    excluded_count = overlay.summary.excluded_observation_count
    for trajectory in overlay.trajectories:
        kept = [
            observation
            for observation in trajectory.observations
            if observation.treatment_class not in excluded_classes
        ]
        excluded_count += len(trajectory.observations) - len(kept)
        if kept:
            observations_by_trajectory[trajectory.trajectory_id] = kept
    return sdo._finalize_overlay(
        source_name=overlay.source_name,
        source_rows=overlay.summary.source_rows,
        excluded_observation_count=excluded_count,
        observations_by_trajectory=observations_by_trajectory,
    )


def _slug(names: set[str]) -> str:
    return "_".join(
        "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
        for name in sorted(names)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source-name", default="ph_combined_nopt_rcm")
    parser.add_argument(
        "--exclude-treatment-class",
        action="append",
        default=None,
        help="Treatment class to drop (repeatable); defaults to 'NOPT NPK'",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    excluded_classes = set(args.exclude_treatment_class or ["NOPT NPK"])
    source_path, encoding = _load_source_spec(args.config, args.source_name)
    overlay = sdo.read_source_dataset_overlay(source_path, args.source_name, encoding=encoding)

    unknown = excluded_classes - set(overlay.summary.treatment_classes)
    if unknown:
        raise ValueError(
            f"Unknown treatment class(es) for source '{args.source_name}': {sorted(unknown)}; "
            f"available: {sorted(overlay.summary.treatment_classes)}"
        )

    filtered = _exclude_treatment_classes(overlay, excluded_classes)

    destination = args.output
    if destination is None:
        destination = (
            DEFAULT_OUTPUT_DIR
            / args.source_name
            / f"{args.source_name}_source_wide_no_{_slug(excluded_classes)}.jpeg"
        )

    figure, axes = sdo.create_source_dataset_overlay_figure(filtered)
    try:
        axes.set_title(
            axes.get_title()
            + f"\nexcluded treatment class(es): {', '.join(sorted(excluded_classes))}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, format="jpeg", dpi=150)
    finally:
        plt.close(figure)

    print(
        f"{args.source_name}: trajectories={filtered.summary.trajectory_count}/"
        f"{overlay.summary.trajectory_count}; observations="
        f"{filtered.summary.finite_observation_count}/"
        f"{overlay.summary.finite_observation_count}"
    )
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
