#!/usr/bin/env python3
"""Generate LTCCE source-wide overlays stratified by recorded replicate.

The source-wide LTCCE plot contains whole replicate-specific trajectories from
all four values of the source ``Rep`` field.  This recipe writes one comparable
view for each recorded replicate value plus a shared-axis panel comparison.  It
does not pool observations, estimate a response curve, or replace any existing
figure.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.reporting.figure_axis_frames import (  # noqa: E402
    SharedAxisLimits,
    shared_axis_limits,
)
from n_response_curve.reporting.figure_output import (  # noqa: E402
    save_figure_atomically,
)
from n_response_curve.reporting.response_curve_clusters import (  # noqa: E402
    TrajectoryContext,
    read_ltcce_contexts,
    subset_overlay,
)
from n_response_curve.reporting.source_config_spec import (  # noqa: E402
    load_source_spec,
)
from n_response_curve.reporting.source_dataset_overlays import (  # noqa: E402
    SourceDatasetOverlay,
    create_source_dataset_overlay_figure,
    read_source_dataset_overlay,
)

SOURCE_NAME = "ltcce"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scriptCONFIG.toml"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "WF/04_Response_Curves/ltcce/clusters/by_replicate"
)
TREATMENT_COLOURS = {"mineral N rate": "C0", "zero N": "C1"}
CONNECTOR_LABEL = "within-trajectory connecting lines (visual aid; not a fit)"


def _replicate_sort_key(value: str) -> tuple[int, float | str]:
    """Sort numeric replicate labels numerically and all other labels lexically."""

    try:
        number = float(value)
    except ValueError:
        return 1, value.casefold()
    if math.isfinite(number):
        return 0, number
    return 1, value.casefold()


def _partition_trajectory_ids_by_replicate(
    overlay: SourceDatasetOverlay,
    contexts: Mapping[str, TrajectoryContext],
) -> dict[str, tuple[str, ...]]:
    """Partition every finite trajectory by its nonblank source ``Rep`` value."""

    grouped: dict[str, list[str]] = {}
    missing: list[str] = []
    blank: list[str] = []
    for trajectory in overlay.trajectories:
        context = contexts.get(trajectory.trajectory_id)
        if context is None:
            missing.append(trajectory.trajectory_id)
            continue
        replicate = context.replicate.strip()
        if not replicate:
            blank.append(trajectory.trajectory_id)
            continue
        grouped.setdefault(replicate, []).append(trajectory.trajectory_id)

    if missing:
        raise ValueError(
            f"{len(missing)} finite LTCCE trajectories lack source context "
            f"(example: {missing[0]})"
        )
    if blank:
        raise ValueError(
            f"{len(blank)} finite LTCCE trajectories have a blank Rep value "
            f"(example: {blank[0]})"
        )
    assigned = sum(len(trajectory_ids) for trajectory_ids in grouped.values())
    if assigned != overlay.summary.trajectory_count:
        raise ValueError(
            "Replicate partition does not exhaust the finite source-wide overlay"
        )
    return {
        replicate: tuple(sorted(grouped[replicate]))
        for replicate in sorted(grouped, key=_replicate_sort_key)
    }


def _n_range_text(overlay: SourceDatasetOverlay) -> str:
    n_range = overlay.summary.n_rate_range_kg_ha
    if n_range is None:
        return "N range unavailable"
    return f"N range={n_range[0]:g}-{n_range[1]:g} kg N/ha"


def _single_title(replicate: str, overlay: SourceDatasetOverlay) -> str:
    return "\n".join(
        (
            f"source={SOURCE_NAME} — recorded Rep={replicate} stratum",
            "whole replicate-specific trajectories; connecting lines are visual aids; "
            "no pooled curve or fit",
            f"trajectories={overlay.summary.trajectory_count}; "
            f"observations={overlay.summary.finite_observation_count}",
            _n_range_text(overlay),
        )
    )


def _write_single_view(
    replicate: str,
    overlay: SourceDatasetOverlay,
    destination: Path,
    limits: SharedAxisLimits,
) -> None:
    figure, axes = create_source_dataset_overlay_figure(overlay)
    try:
        axes.set_title(_single_title(replicate, overlay))
        limits.apply(axes)
        save_figure_atomically(figure, destination)
    finally:
        plt.close(figure)


def _adaptive_style(
    observation_count: int,
    trajectory_count: int,
) -> tuple[float, float, float]:
    marker_size = max(4.0, min(24.0, 6000.0 / max(observation_count, 1)))
    marker_alpha = max(0.25, min(0.85, 300.0 / max(observation_count, 1)))
    line_alpha = max(0.05, min(0.35, 20.0 / max(trajectory_count, 1)))
    return marker_size, marker_alpha, line_alpha


def _draw_panel(axes, overlay: SourceDatasetOverlay) -> None:
    marker_size, marker_alpha, line_alpha = _adaptive_style(
        overlay.summary.finite_observation_count,
        overlay.summary.trajectory_count,
    )
    for treatment_class in overlay.summary.treatment_classes:
        xs = [
            observation.n_rate_kg_ha
            for trajectory in overlay.trajectories
            for observation in trajectory.observations
            if observation.treatment_class == treatment_class
        ]
        ys = [
            observation.yield_t_ha
            for trajectory in overlay.trajectories
            for observation in trajectory.observations
            if observation.treatment_class == treatment_class
        ]
        axes.scatter(
            xs,
            ys,
            s=marker_size,
            alpha=marker_alpha,
            color=TREATMENT_COLOURS.get(treatment_class),
            label=treatment_class,
            zorder=3,
        )

    line_label_emitted = False
    for trajectory in overlay.trajectories:
        axes.plot(
            [observation.n_rate_kg_ha for observation in trajectory.observations],
            [observation.yield_t_ha for observation in trajectory.observations],
            color="grey",
            linewidth=1,
            alpha=line_alpha,
            label=CONNECTOR_LABEL if not line_label_emitted else "_nolegend_",
            zorder=1,
        )
        line_label_emitted = True


def _write_comparison(
    replicate_overlays: Sequence[tuple[str, SourceDatasetOverlay]],
    destination: Path,
    limits: SharedAxisLimits,
) -> None:
    panel_count = len(replicate_overlays)
    columns = 2 if panel_count > 1 else 1
    rows = math.ceil(panel_count / columns)
    figure, axes_grid = plt.subplots(
        rows,
        columns,
        figsize=(14, 5.2 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = list(axes_grid.flat)
    try:
        for panel_index, (replicate, overlay) in enumerate(replicate_overlays):
            panel = axes[panel_index]
            _draw_panel(panel, overlay)
            limits.apply(panel)
            panel.set_title(
                f"Rep={replicate}: {overlay.summary.trajectory_count} trajectories; "
                f"{overlay.summary.finite_observation_count} observations",
                fontsize=10,
            )
        for unused in axes[panel_count:]:
            unused.set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.91))
        figure.supxlabel("Applied N (kg N/ha)")
        figure.supylabel("Grain yield (t/ha)")
        figure.suptitle(
            "\n".join(
                (
                    f"source={SOURCE_NAME} — source-wide trajectories by recorded replicate",
                    "whole replicate-specific trajectories; connecting lines are visual aids; "
                    "no pooled curve or fit",
                    "shared axes across panels; Rep is the source-recorded layout label",
                )
            ),
            fontsize=14,
        )
        save_figure_atomically(figure, destination)
    finally:
        plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_path, encoding = load_source_spec(args.config, SOURCE_NAME)
    overlay = read_source_dataset_overlay(source_path, SOURCE_NAME, encoding=encoding)
    contexts = read_ltcce_contexts(source_path, encoding=encoding)
    replicate_ids = _partition_trajectory_ids_by_replicate(overlay, contexts)
    replicate_overlays = [
        (replicate, subset_overlay(overlay, trajectory_ids))
        for replicate, trajectory_ids in replicate_ids.items()
    ]
    limits = shared_axis_limits(overlay)

    destination = args.output_dir.resolve()
    if destination.exists() and (not destination.is_dir() or destination.is_symlink()):
        raise RuntimeError("Replicate-view destination is not a plain directory")
    destination.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for replicate, replicate_overlay in replicate_overlays:
        output = destination / f"replicate_{replicate}.jpeg"
        _write_single_view(replicate, replicate_overlay, output, limits)
        written.append(output)

    comparison = destination / "replicate_comparison.jpeg"
    _write_comparison(replicate_overlays, comparison, limits)
    written.append(comparison)

    print(
        f"{SOURCE_NAME}: wrote {len(written)} additive replicate figure(s) under "
        f"{destination}"
    )
    for replicate, replicate_overlay in replicate_overlays:
        print(
            f"  Rep={replicate}: trajectories="
            f"{replicate_overlay.summary.trajectory_count}; observations="
            f"{replicate_overlay.summary.finite_observation_count}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
