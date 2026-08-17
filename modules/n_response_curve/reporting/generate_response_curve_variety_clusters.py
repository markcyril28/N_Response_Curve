#!/usr/bin/env python3
"""Write exploratory LTCCE N-response clusters whose units are rice varieties.

Usage:
    conda run -n n_response python \
      modules/n_response_curve/reporting/generate_response_curve_variety_clusters.py \
      --config scriptCONFIG.toml

Outputs are nested under the trajectory-cluster directory so the raw overlay
generator's existing preservation rule carries both diagnostics across workspace
refreshes.  The trajectory-cluster generator also preserves this subdirectory.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tomllib
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.reporting.response_curve_clusters import (  # noqa: E402
    RESPONSE_TYPE_FEATURES,
    format_season_shares,
    read_ltcce_contexts,
    response_type_label,
    subset_overlay,
)
from n_response_curve.reporting.response_curve_variety_clusters import (  # noqa: E402
    EXCLUSION_INSUFFICIENT_CONTEXTS,
    MIN_VARIETY_CONTEXTS,
    VarietyClusteringResult,
    build_variety_clustering,
    variety_cluster_centroid,
    variety_cluster_members,
)
from n_response_curve.reporting.source_dataset_overlays import (  # noqa: E402
    SourceDatasetOverlay,
    create_source_dataset_overlay_figure,
    read_source_dataset_overlay,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scriptCONFIG.toml"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "WF/04_Response_Curves/z_n_response_full/overlay/source_dataset/ltcce/clusters/by_variety"
)
SOURCE_NAME = "ltcce"

_DISCLAIMER = (
    "exploratory diagnostic; variety is confounded with era, season, and N ladder; "
    "no curve is fitted"
)
_AXIS_PAD_FRACTION = 0.04


def _load_source_spec(config_path: Path) -> tuple[Path, str]:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    sources = config.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("The configuration must contain a [sources] table")
    source = sources.get(SOURCE_NAME)
    if not isinstance(source, dict):
        raise ValueError(f"The configuration is missing [sources.{SOURCE_NAME}]")
    raw_path = source.get("data_path")
    encoding = source.get("encoding")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"[sources.{SOURCE_NAME}].data_path must be nonempty")
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError(f"[sources.{SOURCE_NAME}].encoding must be nonempty")
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path, encoding


def _padded_limits(
    value_range: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if value_range is None:
        return None
    low, high = value_range
    if not (math.isfinite(low) and math.isfinite(high)):
        return None
    span = high - low
    pad = span * _AXIS_PAD_FRACTION if span > 0 else max(abs(high), 1.0) * 0.05
    return low - pad, high + pad


def _save_figure(figure: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        figure.savefig(temporary, format="jpeg", dpi=150)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _plot_overlay(
    overlay: SourceDatasetOverlay,
    destination: Path,
    title_lines: Sequence[str],
    *,
    x_limits: tuple[float, float] | None,
    y_limits: tuple[float, float] | None,
    scatter_only: bool = False,
) -> None:
    from matplotlib import pyplot as plt

    if scatter_only:
        figure, axes = plt.subplots(figsize=(10, 7), constrained_layout=True)
        for treatment_class in overlay.summary.treatment_classes:
            observations = [
                observation
                for trajectory in overlay.trajectories
                for observation in trajectory.observations
                if observation.treatment_class == treatment_class
            ]
            axes.scatter(
                [observation.n_rate_kg_ha for observation in observations],
                [observation.yield_t_ha for observation in observations],
                s=8.0,
                alpha=0.30,
                label=treatment_class,
            )
        axes.set_xlabel("Applied N (kg N/ha)")
        axes.set_ylabel("Grain yield (t/ha)")
        axes.legend(loc="best", fontsize=8)
    else:
        figure, axes = create_source_dataset_overlay_figure(overlay)
    try:
        axes.set_title("\n".join(title_lines), fontsize=9)
        if x_limits is not None:
            axes.set_xlim(*x_limits)
        if y_limits is not None:
            axes.set_ylim(*y_limits)
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


def _variety_token(variety: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", variety.casefold()).strip("_")
    if not token:
        token = "unnamed"
    if len(token) > 72:
        digest = hashlib.sha256(variety.encode("utf-8")).hexdigest()[:8]
        token = f"{token[:63].rstrip('_')}_{digest}"
    return token


def _unique_variety_tokens(varieties: Sequence[str]) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for variety in varieties:
        grouped.setdefault(_variety_token(variety), []).append(variety)
    tokens: dict[str, str] = {}
    for token, names in grouped.items():
        for variety in sorted(names):
            if len(names) == 1:
                tokens[variety] = token
            else:
                digest = hashlib.sha256(variety.encode("utf-8")).hexdigest()[:8]
                tokens[variety] = f"{token}_{digest}"
    return tokens


def _cluster_of(result: VarietyClusteringResult) -> dict[str, int]:
    return {
        variety: int(label)
        for variety, label in zip(
            result.partition.variety_names,
            result.partition.labels,
            strict=True,
        )
    }


def _eligible_trajectory_ids(
    result: VarietyClusteringResult,
    varieties: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            trajectory_id
            for variety in varieties
            for trajectory_id in result.profiles[variety].trajectory_ids
        )
    )


def _write_cluster_views(
    result: VarietyClusteringResult,
    overlay: SourceDatasetOverlay,
    staging: Path,
    x_limits: tuple[float, float] | None,
    y_limits: tuple[float, float] | None,
) -> list[str]:
    written: list[str] = []
    for cluster_id in range(result.partition.cluster_count):
        members = variety_cluster_members(result.partition, cluster_id)
        ids = _eligible_trajectory_ids(result, members)
        profiles = [result.profiles[variety] for variety in members]
        years = [year for profile in profiles if profile.year_range for year in profile.year_range]
        centroid = variety_cluster_centroid(result, cluster_id)
        name = f"variety_cluster_{cluster_id + 1}.jpeg"
        _plot_overlay(
            subset_overlay(overlay, ids),
            staging / name,
            (
                f"source={SOURCE_NAME} — rice-variety cluster {cluster_id + 1} of "
                f"{result.partition.cluster_count}",
                f"{len(members)} varieties; {len(ids)} eligible trajectories; "
                f"{sum(profile.experimental_context_count for profile in profiles)} "
                "year-season experimental contexts; "
                f"{min(years)}-{max(years)}",
                response_type_label(centroid),
                f"k chosen by silhouette={result.partition.silhouette:.3f}; "
                f"seed stability (ARI)={result.partition.assignment_stability:.2f}; "
                "membership is in the CSV ledger and profile heatmap",
                (
                    "weak separation: descriptive banding of a continuum"
                    if result.partition.separation_is_weak
                    else "moderate separation in standardized variety-profile space"
                ),
                _DISCLAIMER,
            ),
            x_limits=x_limits,
            y_limits=y_limits,
        )
        written.append(name)
    return written


def _write_profile_heatmap(
    result: VarietyClusteringResult,
    destination: Path,
) -> None:
    from matplotlib import pyplot as plt

    ordered = sorted(
        result.partition.variety_names,
        key=lambda variety: (
            dict(zip(result.partition.variety_names, result.partition.labels, strict=True))[
                variety
            ],
            variety,
        ),
    )
    matrix = np.asarray([result.profiles[variety].feature_means for variety in ordered])
    center = np.asarray(result.partition.feature_center)
    scale = np.asarray(result.partition.feature_scale)
    standardized = (matrix - center) / scale
    cluster_map = _cluster_of(result)
    height = max(11.0, 0.29 * len(ordered))
    figure, axes = plt.subplots(figsize=(10, height), constrained_layout=True)
    try:
        image = axes.imshow(standardized, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
        axes.set_xticks(range(len(RESPONSE_TYPE_FEATURES)))
        axes.set_xticklabels(
            ("zero-N yield", "response magnitude", "relative N at peak", "saturation index"),
            rotation=25,
            ha="right",
        )
        axes.set_yticks(range(len(ordered)))
        axes.set_yticklabels(
            [f"C{cluster_map[variety] + 1}  {variety}" for variety in ordered],
            fontsize=7,
        )
        previous = cluster_map[ordered[0]]
        for row, variety in enumerate(ordered[1:], start=1):
            current = cluster_map[variety]
            if current != previous:
                axes.axhline(row - 0.5, color="black", linewidth=1.5)
            previous = current
        colorbar = figure.colorbar(image, ax=axes, shrink=0.5)
        colorbar.set_label("standard deviations from the supported-variety mean")
        axes.set_title(
            "LTCCE rice-variety response profiles, ordered by cluster\n"
            "context-weighted means of observed ladder-invariant features; no curve fitted\n"
            "exploratory only: variety is confounded with era, season, and N ladder",
            fontsize=10,
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


def _member_output_path(
    variety: str,
    result: VarietyClusteringResult,
    tokens: Mapping[str, str],
    cluster_map: Mapping[str, int],
) -> Path:
    if variety in cluster_map:
        directory = f"cluster_{cluster_map[variety] + 1}"
    else:
        directory = f"held_out_{result.held_out[variety]}"
    return Path(directory) / f"variety_{tokens[variety]}.jpeg"


def _write_member_views(
    result: VarietyClusteringResult,
    overlay: SourceDatasetOverlay,
    staging: Path,
    x_limits: tuple[float, float] | None,
    y_limits: tuple[float, float] | None,
) -> tuple[list[str], dict[str, str]]:
    written: list[str] = []
    output_by_variety: dict[str, str] = {}
    cluster_map = _cluster_of(result)
    tokens = _unique_variety_tokens(tuple(result.variety_trajectory_ids))

    for variety in sorted(result.variety_trajectory_ids):
        raw_ids = result.variety_trajectory_ids[variety]
        profile = result.profiles.get(variety)
        relative = _member_output_path(variety, result, tokens, cluster_map)
        output_by_variety[variety] = relative.as_posix()
        if variety in cluster_map and profile is not None:
            ids = profile.trajectory_ids
            years = profile.year_range
            year_text = f"{years[0]}-{years[1]}" if years else "years unknown"
            title_lines = (
                f"source={SOURCE_NAME} — rice variety {variety}",
                f"cluster {cluster_map[variety] + 1} of {result.partition.cluster_count}; "
                f"{len(ids)} eligible trajectories across "
                f"{profile.experimental_context_count} year-season contexts; {year_text}; "
                f"{format_season_shares(profile.season_shares)}",
                f"mean zero-N yield={profile.feature_map()['yield_at_zero_n_t_ha']:.2f} t/ha; "
                f"mean response={profile.feature_map()['response_above_zero_n_t_ha']:.2f} t/ha; "
                f"{profile.distinct_ladder_count} observed N ladders",
                "replicates are averaged within context for clustering, but all eligible "
                "replicate trajectories are drawn",
                _DISCLAIMER,
            )
            scatter_only = False
        elif result.held_out[variety] == EXCLUSION_INSUFFICIENT_CONTEXTS and profile:
            ids = profile.trajectory_ids
            years = profile.year_range
            year_text = f"{years[0]}-{years[1]}" if years else "years unknown"
            title_lines = (
                f"source={SOURCE_NAME} — rice variety {variety}",
                f"held out: only {profile.experimental_context_count} eligible year-season "
                f"contexts (minimum={MIN_VARIETY_CONTEXTS}); {year_text}",
                f"{len(ids)} eligible trajectories; {format_season_shares(profile.season_shares)}; "
                f"{profile.distinct_ladder_count} observed N ladders",
                "shown descriptively but not assigned to a variety cluster",
                _DISCLAIMER,
            )
            scatter_only = False
        else:
            ids = raw_ids
            reasons = collections.Counter(
                result.trajectory_exclusions.get(trajectory_id, "unknown")
                for trajectory_id in raw_ids
            )
            reason_text = ", ".join(
                f"{reason}={count}" for reason, count in sorted(reasons.items())
            )
            title_lines = (
                f"source={SOURCE_NAME} — rice variety {variety}",
                "held out: no trajectory satisfies response-shape eligibility",
                f"{len(raw_ids)} raw trajectories; {reason_text}",
                "points only: connecting lines would assert response shapes the source does not support",
                _DISCLAIMER,
            )
            scatter_only = True

        _plot_overlay(
            subset_overlay(overlay, ids),
            staging / relative,
            title_lines,
            x_limits=x_limits,
            y_limits=y_limits,
            scatter_only=scatter_only,
        )
        written.append(relative.as_posix())
    return written, output_by_variety


_LEDGER_COLUMNS = (
    "variety",
    "cluster_eligible",
    "exclusion_reason",
    "cluster_id",
    "cluster_count",
    "silhouette",
    "assignment_stability",
    "raw_trajectory_count",
    "eligible_trajectory_count",
    "excluded_trajectory_count",
    "year_season_context_count",
    "year_min",
    "year_max",
    "season_shares",
    "distinct_applied_n_ladders",
    *RESPONSE_TYPE_FEATURES,
    "figure_path",
)


def _write_ledger(
    result: VarietyClusteringResult,
    destination: Path,
    output_by_variety: Mapping[str, str],
) -> None:
    cluster_map = _cluster_of(result)
    rows: list[dict[str, Any]] = []
    for variety, raw_ids in sorted(result.variety_trajectory_ids.items()):
        profile = result.profiles.get(variety)
        cluster_id = cluster_map.get(variety)
        row: dict[str, Any] = {
            "variety": variety,
            "cluster_eligible": "true" if cluster_id is not None else "false",
            "exclusion_reason": result.held_out.get(variety, ""),
            "cluster_id": cluster_id + 1 if cluster_id is not None else "",
            "cluster_count": result.partition.cluster_count if cluster_id is not None else "",
            "silhouette": f"{result.partition.silhouette:.6f}" if cluster_id is not None else "",
            "assignment_stability": (
                f"{result.partition.assignment_stability:.6f}" if cluster_id is not None else ""
            ),
            "raw_trajectory_count": len(raw_ids),
            "eligible_trajectory_count": len(profile.trajectory_ids) if profile else 0,
            "excluded_trajectory_count": sum(
                trajectory_id in result.trajectory_exclusions for trajectory_id in raw_ids
            ),
            "year_season_context_count": profile.experimental_context_count if profile else 0,
            "year_min": profile.year_range[0] if profile and profile.year_range else "",
            "year_max": profile.year_range[1] if profile and profile.year_range else "",
            "season_shares": (
                json.dumps(profile.season_shares, sort_keys=True) if profile else "{}"
            ),
            "distinct_applied_n_ladders": profile.distinct_ladder_count if profile else 0,
            "figure_path": output_by_variety[variety],
        }
        for index, feature_name in enumerate(RESPONSE_TYPE_FEATURES):
            row[feature_name] = f"{profile.feature_means[index]:.6f}" if profile else ""
        rows.append(row)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    result: VarietyClusteringResult,
    overlay: SourceDatasetOverlay,
    destination: Path,
    *,
    minimum_contexts: int,
    x_limits: tuple[float, float] | None,
    y_limits: tuple[float, float] | None,
) -> dict[str, Any]:
    held_out_counts = collections.Counter(result.held_out.values())
    trajectory_exclusion_counts = collections.Counter(result.trajectory_exclusions.values())
    clusters = []
    for cluster_id in range(result.partition.cluster_count):
        members = variety_cluster_members(result.partition, cluster_id)
        clusters.append(
            {
                "cluster_id": cluster_id + 1,
                "variety_count": len(members),
                "varieties": list(members),
                "centroid": variety_cluster_centroid(result, cluster_id),
            }
        )
    summary = {
        "artifact_class": "exploratory_diagnostic",
        "governed_analysis": False,
        "source_name": SOURCE_NAME,
        "unit_clustered": "rice_variety",
        "method": (
            "Replicate trajectory features are averaged within design-site-year-season-"
            "variety-ladder contexts; context means receive equal weight within each "
            "variety; the four variety means are standardized before k-means; k is "
            "chosen by maximum silhouette over 2..maximum_clusters."
        ),
        "interpretation_caveat": (
            "Variety identity is confounded with calendar period, season, and applied-N "
            "ladder in LTCCE. Clusters describe observed source profiles and are not "
            "genetic, causal, or recommendation classes."
        ),
        "minimum_year_season_contexts": minimum_contexts,
        "observed_variety_count": len(result.variety_trajectory_ids),
        "varieties_with_eligible_profiles": len(result.profiles),
        "clustered_variety_count": len(result.partition.variety_names),
        "held_out_variety_count": len(result.held_out),
        "held_out_by_reason": dict(sorted(held_out_counts.items())),
        "raw_trajectory_count": overlay.summary.trajectory_count,
        "eligible_trajectory_count": len(result.features),
        "excluded_trajectory_count": len(result.trajectory_exclusions),
        "trajectory_exclusions_by_reason": dict(sorted(trajectory_exclusion_counts.items())),
        "observed_yield_range_t_ha": list(overlay.summary.yield_range_t_ha or ()),
        "observed_n_rate_range_kg_ha": list(overlay.summary.n_rate_range_kg_ha or ()),
        "shared_yield_axis_t_ha": list(y_limits or ()),
        "shared_n_rate_axis_kg_ha": list(x_limits or ()),
        "partition": {
            "basis": "context-weighted standardized ladder-invariant variety profiles",
            "feature_names": list(result.partition.feature_names),
            "cluster_count": result.partition.cluster_count,
            "silhouette": result.partition.silhouette,
            "assignment_stability_adjusted_rand": result.partition.assignment_stability,
            "separation_is_weak": result.partition.separation_is_weak,
            "cluster_sizes": [cluster["variety_count"] for cluster in clusters],
        },
        "clusters": clusters,
    }
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_readme(summary: Mapping[str, Any], destination: Path) -> None:
    cluster_count = int(summary["partition"]["cluster_count"])
    cluster_lines = "\n".join(
        f"- `cluster_{cluster_id}/`: one plot for each variety assigned to cluster "
        f"{cluster_id}."
        for cluster_id in range(1, cluster_count + 1)
    )
    destination.write_text(
        "# LTCCE rice-variety response clusters\n\n"
        f"This folder contains {summary['clustered_variety_count']} clustered rice "
        f"varieties and {summary['held_out_variety_count']} held-out varieties from "
        "the LTCCE source. Every observed variety has one individual plot.\n\n"
        "## Top-level files\n\n"
        "- `variety_cluster_*.jpeg`: raw eligible trajectories pooled by assigned "
        "variety cluster.\n"
        "- `variety_profile_heatmap.jpeg`: standardized variety profiles with every "
        "clustered variety labeled.\n"
        "- `variety_cluster_assignments.csv`: membership, support, observed feature "
        "means, exclusions, and the path to each variety plot.\n"
        "- `variety_clustering_summary.json`: method, diagnostics, caveats, counts, "
        "centroids, and complete member lists.\n\n"
        "## Subfolders\n\n"
        f"{cluster_lines}\n"
        "- `held_out_insufficient_year_season_contexts/`: varieties with eligible "
        "curves but too few contexts for clustering.\n"
        "- `held_out_no_eligible_trajectories/`: varieties whose trajectories all "
        "failed response-shape eligibility; these are shown as points only.\n\n"
        "## Interpretation boundary\n\n"
        f"{summary['interpretation_caveat']} No curve is fitted anywhere in this "
        "diagnostic.\n",
        encoding="utf-8",
    )


def _promote(staging: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.backup.{uuid.uuid4().hex}")
    if destination.exists():
        if not destination.is_dir() or destination.is_symlink():
            raise RuntimeError("Variety-cluster destination is not a plain directory")
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--min-contexts",
        type=int,
        default=MIN_VARIETY_CONTEXTS,
        help="Minimum replicate-free year-season contexts required per variety",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=8,
        help="Upper bound of the silhouette search for k",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_path, encoding = _load_source_spec(args.config)
    overlay = read_source_dataset_overlay(source_path, SOURCE_NAME, encoding=encoding)
    contexts = read_ltcce_contexts(source_path, encoding=encoding)
    result = build_variety_clustering(
        overlay,
        contexts,
        minimum_contexts=args.min_contexts,
        maximum_clusters=args.max_clusters,
    )
    x_limits = _padded_limits(overlay.summary.n_rate_range_kg_ha)
    y_limits = _padded_limits(overlay.summary.yield_range_t_ha)

    destination = args.output_dir.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.staging.{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        written = _write_cluster_views(
            result,
            overlay,
            staging,
            x_limits,
            y_limits,
        )
        _write_profile_heatmap(result, staging / "variety_profile_heatmap.jpeg")
        written.append("variety_profile_heatmap.jpeg")
        member_files, output_by_variety = _write_member_views(
            result,
            overlay,
            staging,
            x_limits,
            y_limits,
        )
        written += member_files
        _write_ledger(result, staging / "variety_cluster_assignments.csv", output_by_variety)
        summary = _write_summary(
            result,
            overlay,
            staging / "variety_clustering_summary.json",
            minimum_contexts=args.min_contexts,
            x_limits=x_limits,
            y_limits=y_limits,
        )
        _write_readme(summary, staging / "README.md")
        _promote(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    partition = result.partition
    print(
        f"{SOURCE_NAME}: clustered {summary['clustered_variety_count']} of "
        f"{summary['observed_variety_count']} rice varieties; "
        f"{summary['held_out_variety_count']} held out "
        f"({summary['held_out_by_reason']})"
    )
    print(
        f"  k={partition.cluster_count} silhouette={partition.silhouette:.3f} "
        f"ARI={partition.assignment_stability:.2f}; "
        f"sizes={[len(variety_cluster_members(partition, i)) for i in range(partition.cluster_count)]}"
    )
    print(f"Wrote {len(written)} figures + ledger + summary under {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
