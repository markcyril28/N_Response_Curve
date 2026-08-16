#!/usr/bin/env python3
"""Write the descriptive LTCCE response-curve cluster views.

Exploratory diagnostic, deliberately outside the release inventory. `ANA-11`
disables the governed `curve_feature_clustering` family, so this generator does
not touch `scriptCONFIG.toml`, does not write into a promoted package, and does
not emit records shaped like an authorized analysis family.

The destination lives under `[paths].curves_root`, which
`reporting/workspace_outputs.py` snapshot-replaces on every workspace refresh
(`_promote_directory` renames the whole `<curves_root>/<view>` tree into place).
These figures are therefore **derived artifacts behind a generator**, exactly
like `generate_raw_dataset_overlays.py`: re-run this script after any full run or
`--refresh-workspace-outputs`.

Usage:
    conda run -n n_response python \\
      modules/n_response_curve/reporting/generate_response_curve_clusters.py \\
      --config scriptCONFIG.toml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tomllib
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.reporting.response_curve_clusters import (  # noqa: E402
    EXCLUSION_DUPLICATED_N_LEVEL,
    EXCLUSION_INCOMPLETE_LADDER,
    EXCLUSION_NO_ZERO_N_ANCHOR,
    ClusterPartition,
    ClusteringResult,
    build_clustering,
    centroid_summary,
    cluster_members,
    composition,
    format_season_shares,
    read_ltcce_contexts,
    response_type_label,
    subset_overlay,
)
from n_response_curve.reporting.source_dataset_overlays import (  # noqa: E402
    SourceDatasetOverlay,
    create_source_dataset_overlay_figure,
    read_source_dataset_overlay,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scriptCONFIG.toml"
# Sits beside the four LTCCE source-dataset overlays, in the subdirectory
# generate_raw_dataset_overlays.py:_PRESERVED_SUBDIRECTORY carries across its own
# snapshot replacement. Writing these flat into the parent would trip that
# generator's unmanaged-entry guard.
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "WF/04_Response_Curves/n_response_full/figures/overlay/source_dataset/ltcce/clusters"
)
SOURCE_NAME = "ltcce"

# Carried into every figure so the boundary survives this conversation.
_DISCLAIMER = (
    "exploratory diagnostic; not a governed analysis (ANA-11) — no curve is fitted"
)

# Percentage-point excess of one season in a cluster, over that season's share of
# the parent stratum, at which the split is disclosed as season-confounded.
_SEASON_CONCENTRATION_MARGIN = 0.15


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
        raise ValueError(f"[sources.{SOURCE_NAME}].data_path must be a nonempty string")
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError(f"[sources.{SOURCE_NAME}].encoding must be a nonempty string")
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path, encoding


def _ladder_token(ladder: Sequence[float]) -> str:
    return "n" + "_".join(f"{level:g}" for level in ladder)


def _ladder_text(ladder: Sequence[float]) -> str:
    return "/".join(f"{level:g}" for level in ladder)


def _partition_diagnostics_line(partition: ClusterPartition) -> str:
    coherence = partition.replicate_coherence
    coherence_text = (
        "n/a" if math.isnan(coherence) else f"{100 * coherence:.0f}%"
    )
    return (
        f"k={partition.cluster_count} chosen by silhouette; "
        f"silhouette={partition.silhouette:.3f}; "
        f"seed stability (ARI)={partition.assignment_stability:.2f}; "
        f"replicates co-assigned={coherence_text}"
    )


def _separation_caveat(partition: ClusterPartition) -> str:
    if partition.separation_is_weak:
        return (
            "separation is weak: read these as a descriptive banding of a "
            "continuum, not as discrete response types"
        )
    return (
        f"separation is moderate; the partition mainly tracks {partition.dominant_axis}"
    )


def _season_mixture_caveat(
    member_shares: Mapping[str, float],
    stratum_shares: Mapping[str, float],
) -> str:
    """Warn when a cluster concentrates a season relative to its stratum.

    Applied-N ladders are confounded with season in this source, so a cluster
    that over-represents one season is reporting season as much as N response.
    The test is *concentration*, not mixedness: the 0/50/100/150 stratum is 92%
    DS overall, yet all 44 of its EWS trajectories land in one cluster (37% of
    it). A stratum-level purity check would miss exactly that case.
    """

    if not member_shares or not stratum_shares:
        return ""
    margin, season = max(
        (share - stratum_shares.get(season, 0.0), season)
        for season, share in member_shares.items()
    )
    if margin < _SEASON_CONCENTRATION_MARGIN:
        return ""
    return (
        f"this cluster concentrates {season} "
        f"({100 * member_shares[season]:.0f}% here vs "
        f"{100 * stratum_shares.get(season, 0.0):.0f}% across the stratum): "
        "the split partly tracks season, not N response alone"
    )


def _save_figure(figure: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        figure.savefig(temporary, format="jpeg", dpi=150)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _titled_figure(
    overlay: SourceDatasetOverlay,
    destination: Path,
    title_lines: Sequence[str],
) -> None:
    from matplotlib import pyplot as plt

    figure, axes = create_source_dataset_overlay_figure(overlay)
    try:
        axes.set_title("\n".join(title_lines), fontsize=9)
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


def _scatter_only_figure(
    overlay: SourceDatasetOverlay,
    destination: Path,
    title_lines: Sequence[str],
) -> None:
    """Render points with no connecting lines.

    Used for the trajectories whose replicate identity is ambiguous. Drawing a
    connector there would assert an ordering through two superimposed treatments
    that the source does not support.
    """

    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(figsize=(10, 7), constrained_layout=True)
    try:
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
            axes.scatter(xs, ys, s=8.0, alpha=0.35, label=treatment_class, zorder=3)
        axes.set_xlabel("Applied N (kg N/ha)")
        axes.set_ylabel("Grain yield (t/ha)")
        axes.set_title("\n".join(title_lines), fontsize=9)
        axes.legend(loc="best", fontsize=8)
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


def _write_stratum_views(
    result: ClusteringResult,
    overlay: SourceDatasetOverlay,
    staging: Path,
) -> list[str]:
    written: list[str] = []
    for ladder in result.major_ladders:
        ids = result.strata[ladder]
        partition = result.stratum_partitions[ladder]
        context = composition(ids, result.contexts)
        years = context["year_range"]
        year_text = f"{years[0]}-{years[1]}" if years else "years unknown"
        token = _ladder_token(ladder)

        name = f"stratum_{token}.jpeg"
        _titled_figure(
            subset_overlay(overlay, ids),
            staging / name,
            (
                f"source={SOURCE_NAME} — applied-N design stratum {_ladder_text(ladder)} kg N/ha",
                f"{len(ids)} replicate trajectories; {year_text}; "
                f"{format_season_shares(context['season_shares'])}",
                "exact-ladder stratum: trajectories on other N ladders are held out, not interpolated",
                _DISCLAIMER,
            ),
        )
        written.append(name)

        for cluster_id in range(partition.cluster_count):
            members = cluster_members(partition, cluster_id)
            centroid = centroid_summary(members, result.features)
            member_context = composition(members, result.contexts)
            member_years = member_context["year_range"]
            member_year_text = (
                f"{member_years[0]}-{member_years[1]}" if member_years else "years unknown"
            )
            cluster_name = f"stratum_{token}_cluster_{cluster_id + 1}.jpeg"
            title_lines = [
                f"source={SOURCE_NAME} — stratum {_ladder_text(ladder)} kg N/ha, "
                f"cluster {cluster_id + 1} of {partition.cluster_count}",
                f"{len(members)} trajectories; {member_year_text}; "
                f"{format_season_shares(member_context['season_shares'])}",
                f"mean zero-N yield={centroid['yield_at_zero_n_t_ha']:.2f} t/ha; "
                f"mean response above zero N={centroid['response_above_zero_n_t_ha']:.2f} t/ha; "
                f"mean peak at {100 * centroid['relative_n_at_peak']:.0f}% of the top rate",
                _partition_diagnostics_line(partition),
                _separation_caveat(partition),
            ]
            season_caveat = _season_mixture_caveat(
                member_context["season_shares"], context["season_shares"]
            )
            if season_caveat:
                title_lines.append(season_caveat)
            title_lines.append(_DISCLAIMER)
            _titled_figure(
                subset_overlay(overlay, members), staging / cluster_name, title_lines
            )
            written.append(cluster_name)
    return written


def _write_response_type_views(
    result: ClusteringResult,
    overlay: SourceDatasetOverlay,
    staging: Path,
) -> list[str]:
    partition = result.response_type_partition
    if partition is None:
        return []
    written: list[str] = []
    for cluster_id in range(partition.cluster_count):
        members = cluster_members(partition, cluster_id)
        centroid = centroid_summary(members, result.features)
        context = composition(members, result.contexts)
        years = context["year_range"]
        year_text = f"{years[0]}-{years[1]}" if years else "years unknown"
        ladders = {result.features[member].ladder for member in members}
        name = f"response_type_{cluster_id + 1}.jpeg"
        _titled_figure(
            subset_overlay(overlay, members),
            staging / name,
            (
                f"source={SOURCE_NAME} — response type {cluster_id + 1} of "
                f"{partition.cluster_count} (ladder-invariant clustering)",
                response_type_label(centroid),
                f"{len(members)} trajectories across {len(ladders)} applied-N ladders; "
                f"{year_text}; {format_season_shares(context['season_shares'])}",
                f"mean zero-N yield={centroid['yield_at_zero_n_t_ha']:.2f} t/ha; "
                f"mean response={centroid['response_above_zero_n_t_ha']:.2f} t/ha; "
                f"mean marginal-yield change={100 * centroid['saturation_index']:+.2f} "
                "kg grain per kg N (first step to last)",
                _partition_diagnostics_line(partition),
                _separation_caveat(partition),
                _DISCLAIMER,
            ),
        )
        written.append(name)
    return written


def _write_held_out_views(
    result: ClusteringResult,
    overlay: SourceDatasetOverlay,
    staging: Path,
) -> list[str]:
    written: list[str] = []

    if result.other_ladder_trajectory_ids:
        ids = result.other_ladder_trajectory_ids
        ladders = {result.features[trajectory_id].ladder for trajectory_id in ids}
        context = composition(ids, result.contexts)
        name = "stratum_other_ladders.jpeg"
        _titled_figure(
            subset_overlay(overlay, ids),
            staging / name,
            (
                f"source={SOURCE_NAME} — minor applied-N ladders (not clustered)",
                f"{len(ids)} trajectories spanning {len(ladders)} distinct ladders; "
                f"{format_season_shares(context['season_shares'])}",
                "clustering across incompatible N supports would reintroduce the "
                "incomparability the stratification exists to prevent",
                "partial ladders are held out by exact match rather than imputed",
                _DISCLAIMER,
            ),
        )
        written.append(name)

    grouped: dict[str, list[str]] = {}
    for trajectory_id, reason in result.excluded.items():
        grouped.setdefault(reason, []).append(trajectory_id)

    explanations = {
        EXCLUSION_DUPLICATED_N_LEVEL: (
            "more than one observed yield at the same applied-N level",
            "entirely Split-plot 1991-2001 (MV-028 row multiplicity); consistent with an "
            "uncaptured sub-plot factor, so averaging would fabricate a mean curve",
            "drawn without connecting lines: replicate identity is ambiguous",
        ),
        EXCLUSION_INCOMPLETE_LADDER: (
            "fewer than three distinct applied-N levels observed",
            "a response shape cannot be described from one or two points",
            "",
        ),
        EXCLUSION_NO_ZERO_N_ANCHOR: (
            "no observation at exactly zero applied N",
            "without the unfertilized baseline, response above zero N is undefined",
            "",
        ),
    }

    for reason, ids in sorted(grouped.items()):
        headline, rationale, note = explanations.get(reason, (reason, "", ""))
        context = composition(ids, result.contexts)
        years = context["year_range"]
        year_text = f"{years[0]}-{years[1]}" if years else "years unknown"
        name = f"held_out_{reason}.jpeg"
        title_lines = [
            f"source={SOURCE_NAME} — held out of clustering: {headline}",
            f"{len(ids)} trajectories; {year_text}; "
            f"{format_season_shares(context['season_shares'])}",
        ]
        if rationale:
            title_lines.append(rationale)
        if note:
            title_lines.append(note)
        title_lines.append(_DISCLAIMER)

        subset = subset_overlay(overlay, ids)
        if reason == EXCLUSION_DUPLICATED_N_LEVEL:
            _scatter_only_figure(subset, staging / name, title_lines)
        else:
            _titled_figure(subset, staging / name, title_lines)
        written.append(name)
    return written


_LEDGER_COLUMNS = (
    "trajectory_id",
    "cluster_eligible",
    "exclusion_reason",
    "applied_n_ladder_kg_ha",
    "ladder_stratum",
    "stratum_cluster_id",
    "stratum_cluster_count",
    "stratum_silhouette",
    "stratum_assignment_stability",
    "response_type_id",
    "response_type_count",
    "response_type_silhouette",
    "response_type_assignment_stability",
    "design",
    "site",
    "year",
    "season",
    "variety",
    "replicate",
    "yield_at_zero_n_t_ha",
    "observed_max_yield_t_ha",
    "response_above_zero_n_t_ha",
    "n_at_observed_peak_kg_ha",
    "relative_n_at_peak",
    "saturation_index",
    "decline_from_peak_t_ha",
)


def _write_ledger(result: ClusteringResult, destination: Path) -> None:
    """Emit every trajectory, clustered or held out, with its reason."""

    stratum_of: dict[str, tuple[str, ClusterPartition, int]] = {}
    for ladder, partition in result.stratum_partitions.items():
        token = _ladder_text(ladder)
        for trajectory_id, label in zip(
            partition.trajectory_ids, partition.labels, strict=True
        ):
            stratum_of[trajectory_id] = (token, partition, int(label))

    response_of: dict[str, int] = {}
    response_partition = result.response_type_partition
    if response_partition is not None:
        response_of = {
            trajectory_id: int(label)
            for trajectory_id, label in zip(
                response_partition.trajectory_ids,
                response_partition.labels,
                strict=True,
            )
        }

    rows: list[dict[str, Any]] = []
    for trajectory_id, feature in sorted(result.features.items()):
        context = result.contexts.get(trajectory_id)
        stratum = stratum_of.get(trajectory_id)
        row: dict[str, Any] = {
            "trajectory_id": trajectory_id,
            "cluster_eligible": "true",
            "exclusion_reason": "",
            "applied_n_ladder_kg_ha": _ladder_text(feature.ladder),
            "ladder_stratum": stratum[0] if stratum else "minor_ladder",
            "stratum_cluster_id": stratum[2] + 1 if stratum else "",
            "stratum_cluster_count": stratum[1].cluster_count if stratum else "",
            "stratum_silhouette": f"{stratum[1].silhouette:.6f}" if stratum else "",
            "stratum_assignment_stability": (
                f"{stratum[1].assignment_stability:.6f}" if stratum else ""
            ),
            "response_type_id": (
                response_of[trajectory_id] + 1 if trajectory_id in response_of else ""
            ),
            "response_type_count": (
                response_partition.cluster_count if response_partition else ""
            ),
            "response_type_silhouette": (
                f"{response_partition.silhouette:.6f}" if response_partition else ""
            ),
            "response_type_assignment_stability": (
                f"{response_partition.assignment_stability:.6f}"
                if response_partition
                else ""
            ),
            "yield_at_zero_n_t_ha": f"{feature.yield_at_zero_n_t_ha:.4f}",
            "observed_max_yield_t_ha": f"{feature.observed_max_yield_t_ha:.4f}",
            "response_above_zero_n_t_ha": f"{feature.response_above_zero_n_t_ha:.4f}",
            "n_at_observed_peak_kg_ha": f"{feature.n_at_observed_peak_kg_ha:g}",
            "relative_n_at_peak": f"{feature.relative_n_at_peak:.4f}",
            "saturation_index": f"{feature.saturation_index:.6f}",
            "decline_from_peak_t_ha": f"{feature.decline_from_peak_t_ha:.4f}",
        }
        _attach_context(row, context)
        rows.append(row)

    for trajectory_id, reason in sorted(result.excluded.items()):
        row = {column: "" for column in _LEDGER_COLUMNS}
        row["trajectory_id"] = trajectory_id
        row["cluster_eligible"] = "false"
        row["exclusion_reason"] = reason
        _attach_context(row, result.contexts.get(trajectory_id))
        rows.append(row)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _attach_context(row: dict[str, Any], context: Any) -> None:
    row["design"] = context.design if context else ""
    row["site"] = context.site if context else ""
    row["year"] = context.year if context and context.year > 0 else ""
    row["season"] = context.season if context else ""
    row["variety"] = context.variety if context else ""
    row["replicate"] = context.replicate if context else ""


def _partition_record(partition: ClusterPartition) -> dict[str, Any]:
    return {
        "partition_id": partition.partition_id,
        "basis": partition.basis,
        "member_count": partition.member_count,
        "cluster_count": partition.cluster_count,
        "silhouette": partition.silhouette,
        "assignment_stability_adjusted_rand": partition.assignment_stability,
        "feature_names": list(partition.feature_names),
        "between_cluster_level_variance": _json_number(
            partition.between_cluster_level_variance
        ),
        "between_cluster_shape_variance": _json_number(
            partition.between_cluster_shape_variance
        ),
        "replicate_co_assignment_rate": _json_number(partition.replicate_coherence),
        "separation_is_weak": partition.separation_is_weak,
        "cluster_sizes": [
            sum(1 for label in partition.labels if label == cluster_id)
            for cluster_id in range(partition.cluster_count)
        ],
    }


def _json_number(value: float) -> float | None:
    return None if math.isnan(value) else value


def _write_summary(
    result: ClusteringResult,
    overlay: SourceDatasetOverlay,
    destination: Path,
) -> dict[str, Any]:
    exclusion_counts: dict[str, int] = {}
    for reason in result.excluded.values():
        exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1

    summary = {
        "artifact_class": "exploratory_diagnostic",
        "governed_analysis": False,
        "governance_note": (
            "ANA-11 disables the curve_feature_clustering analysis family for the "
            "current primary release. These views are descriptive only and are not "
            "part of the release inventory or any checksum ledger."
        ),
        "source_name": SOURCE_NAME,
        "total_trajectories": overlay.summary.trajectory_count,
        "total_finite_observations": overlay.summary.finite_observation_count,
        "cluster_eligible_trajectories": len(result.features),
        "held_out_trajectories": len(result.excluded),
        "held_out_by_reason": dict(sorted(exclusion_counts.items())),
        "distinct_applied_n_ladders": len(result.strata),
        "major_ladder_strata": [
            {
                "ladder_kg_ha": list(ladder),
                "trajectories": len(result.strata[ladder]),
                "season_shares": composition(result.strata[ladder], result.contexts)[
                    "season_shares"
                ],
                "year_range": composition(result.strata[ladder], result.contexts)[
                    "year_range"
                ],
            }
            for ladder in result.major_ladders
        ],
        "minor_ladder_trajectories": len(result.other_ladder_trajectory_ids),
        "stratum_partitions": [
            _partition_record(result.stratum_partitions[ladder])
            for ladder in result.major_ladders
        ],
        "response_type_partition": (
            _partition_record(result.response_type_partition)
            if result.response_type_partition
            else None
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return summary


def _promote(staging: Path, destination: Path) -> None:
    """Swap a fully built staging directory into place."""

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.backup.{uuid.uuid4().hex}")
    if destination.exists():
        if not destination.is_dir() or destination.is_symlink():
            raise RuntimeError("Cluster view destination is not a plain directory")
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
        "--min-stratum-size",
        type=int,
        default=100,
        help="Smallest applied-N ladder stratum that is clustered rather than pooled",
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
    result = build_clustering(
        overlay,
        contexts,
        minimum_stratum_size=args.min_stratum_size,
        maximum_clusters=args.max_clusters,
    )

    destination = args.output_dir.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.staging.{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        written = _write_stratum_views(result, overlay, staging)
        written += _write_response_type_views(result, overlay, staging)
        written += _write_held_out_views(result, overlay, staging)
        _write_ledger(result, staging / "cluster_assignments.csv")
        summary = _write_summary(result, overlay, staging / "clustering_summary.json")
        _promote(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    print(
        f"{SOURCE_NAME}: {summary['cluster_eligible_trajectories']} of "
        f"{summary['total_trajectories']} trajectories clustered; "
        f"{summary['held_out_trajectories']} held out "
        f"({summary['held_out_by_reason']})"
    )
    for ladder, partition in result.stratum_partitions.items():
        print(
            f"  stratum {_ladder_text(ladder):22s} n={partition.member_count:4d} "
            f"k={partition.cluster_count} silhouette={partition.silhouette:.3f} "
            f"ARI={partition.assignment_stability:.2f} axis={partition.dominant_axis}"
        )
    response_partition = result.response_type_partition
    if response_partition is not None:
        print(
            f"  response types            n={response_partition.member_count:4d} "
            f"k={response_partition.cluster_count} "
            f"silhouette={response_partition.silhouette:.3f} "
            f"ARI={response_partition.assignment_stability:.2f}"
        )
    print(f"Wrote {len(written)} figures + ledger + summary under {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
