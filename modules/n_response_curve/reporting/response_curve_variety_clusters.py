"""Descriptive clustering of LTCCE rice varieties by observed N response.

This module is the variety-level companion to :mod:`response_curve_clusters`.
It does not fit response curves or interpolate across the source's incompatible
applied-N ladders.  Instead, it:

1. describes each eligible replicate trajectory with the existing four
   ladder-invariant response features;
2. averages replicates within a year-season experimental context;
3. gives those contexts equal weight when forming one profile per variety; and
4. clusters sufficiently supported variety profiles after standardization.

The result is an exploratory diagnostic only.  Variety, calendar period,
season, and applied-N ladder are confounded in LTCCE, so the clusters describe
this source and must not be read as genetic or causal variety effects.
"""

from __future__ import annotations

import collections
import statistics
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from n_response_curve.reporting.response_curve_clusters import (
    RESPONSE_TYPE_FEATURES,
    TrajectoryContext,
    TrajectoryFeatures,
    partition_eligibility,
)
from n_response_curve.reporting.source_dataset_overlays import SourceDatasetOverlay

MIN_VARIETY_CONTEXTS = 3
WEAK_SEPARATION_SILHOUETTE = 0.40

EXCLUSION_INSUFFICIENT_CONTEXTS = "insufficient_year_season_contexts"
EXCLUSION_NO_ELIGIBLE_TRAJECTORIES = "no_eligible_trajectories"

_KMEANS_SEED = 20260720
_KMEANS_N_INIT = 20
_STABILITY_SEEDS = 5


@dataclass(frozen=True)
class VarietyProfile:
    """One equally context-weighted, ladder-invariant variety profile."""

    variety: str
    trajectory_ids: tuple[str, ...]
    experimental_context_count: int
    year_range: tuple[int, int] | None
    season_shares: Mapping[str, float]
    distinct_ladder_count: int
    feature_means: tuple[float, ...]

    def feature_map(self) -> dict[str, float]:
        return dict(zip(RESPONSE_TYPE_FEATURES, self.feature_means, strict=True))


@dataclass(frozen=True)
class VarietyClusterPartition:
    """K-means partition of supported variety profiles."""

    variety_names: tuple[str, ...]
    labels: tuple[int, ...]
    cluster_count: int
    silhouette: float
    assignment_stability: float
    feature_names: tuple[str, ...]
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]

    @property
    def separation_is_weak(self) -> bool:
        return self.silhouette <= WEAK_SEPARATION_SILHOUETTE


@dataclass(frozen=True)
class VarietyClusteringResult:
    """Profiles, partition, exclusions, and source membership for reporting."""

    profiles: Mapping[str, VarietyProfile]
    partition: VarietyClusterPartition
    held_out: Mapping[str, str]
    variety_trajectory_ids: Mapping[str, tuple[str, ...]]
    trajectory_exclusions: Mapping[str, str]
    contexts: Mapping[str, TrajectoryContext]
    features: Mapping[str, TrajectoryFeatures]


def _context_key(
    context: TrajectoryContext,
    feature: TrajectoryFeatures,
) -> tuple[str, str, int, str, str, tuple[float, ...]]:
    """Identify a replicate-free experimental context.

    ``TrajectoryContext`` intentionally omits the LTCCE ``Expt`` field.  The
    exact N ladder is included here so two designs sharing a year/season label
    cannot be accidentally collapsed when their experimental supports differ.
    """

    return (
        context.design,
        context.site,
        context.year,
        context.season,
        context.variety,
        feature.ladder,
    )


def build_variety_profiles(
    features: Mapping[str, TrajectoryFeatures],
    contexts: Mapping[str, TrajectoryContext],
) -> dict[str, VarietyProfile]:
    """Aggregate trajectory features without allowing replicate-rich years to dominate."""

    grouped: dict[
        str,
        dict[
            tuple[str, str, int, str, str, tuple[float, ...]],
            list[tuple[str, TrajectoryFeatures]],
        ],
    ] = collections.defaultdict(lambda: collections.defaultdict(list))
    for trajectory_id, feature in features.items():
        context = contexts.get(trajectory_id)
        if context is None or not context.variety.strip():
            continue
        grouped[context.variety][_context_key(context, feature)].append(
            (trajectory_id, feature)
        )

    profiles: dict[str, VarietyProfile] = {}
    for variety, context_groups in sorted(grouped.items()):
        context_vectors: list[np.ndarray] = []
        trajectory_ids: list[str] = []
        years: list[int] = []
        seasons: collections.Counter[str] = collections.Counter()
        ladders: set[tuple[float, ...]] = set()

        for context_key, members in sorted(context_groups.items()):
            matrix = np.asarray(
                [
                    [getattr(feature, name) for name in RESPONSE_TYPE_FEATURES]
                    for _, feature in members
                ],
                dtype=float,
            )
            context_vectors.append(matrix.mean(axis=0))
            trajectory_ids.extend(trajectory_id for trajectory_id, _ in members)
            year = context_key[2]
            season = context_key[3]
            ladder = context_key[5]
            if year > 0:
                years.append(year)
            seasons[season] += 1
            ladders.add(ladder)

        total_contexts = len(context_vectors)
        season_total = sum(seasons.values())
        profiles[variety] = VarietyProfile(
            variety=variety,
            trajectory_ids=tuple(sorted(trajectory_ids)),
            experimental_context_count=total_contexts,
            year_range=(min(years), max(years)) if years else None,
            season_shares=(
                {
                    season: count / season_total
                    for season, count in seasons.most_common()
                }
                if season_total
                else {}
            ),
            distinct_ladder_count=len(ladders),
            feature_means=tuple(
                float(value)
                for value in np.asarray(context_vectors, dtype=float).mean(axis=0)
            ),
        )
    return profiles


def _select_partition(
    matrix: np.ndarray,
    maximum_clusters: int,
) -> tuple[float, int, np.ndarray]:
    distinct = len(np.unique(matrix, axis=0))
    upper = min(maximum_clusters, len(matrix) - 1, distinct)
    best: tuple[float, int, np.ndarray] | None = None
    for cluster_count in range(2, upper + 1):
        labels = KMeans(
            n_clusters=cluster_count,
            random_state=_KMEANS_SEED,
            n_init=_KMEANS_N_INIT,
        ).fit_predict(matrix)
        if len(set(int(label) for label in labels)) < 2:
            continue
        score = float(silhouette_score(matrix, labels))
        if best is None or score > best[0]:
            best = score, cluster_count, labels
    if best is None:
        raise ValueError("No admissible rice-variety partition was found")
    return best


def _assignment_stability(
    matrix: np.ndarray,
    cluster_count: int,
    labels: np.ndarray,
) -> float:
    alternates = (
        KMeans(
            n_clusters=cluster_count,
            random_state=_KMEANS_SEED + 1 + seed,
            n_init=_KMEANS_N_INIT,
        ).fit_predict(matrix)
        for seed in range(_STABILITY_SEEDS)
    )
    return statistics.fmean(
        adjusted_rand_score(labels, alternate) for alternate in alternates
    )


def _order_cluster_labels(
    matrix: np.ndarray,
    labels: np.ndarray,
    cluster_count: int,
) -> np.ndarray:
    """Give the most N-responsive centroid the first display label."""

    response_index = RESPONSE_TYPE_FEATURES.index("response_above_zero_n_t_ha")
    baseline_index = RESPONSE_TYPE_FEATURES.index("yield_at_zero_n_t_ha")
    centroids = {
        cluster_id: matrix[labels == cluster_id].mean(axis=0)
        for cluster_id in range(cluster_count)
    }
    ordered = sorted(
        centroids,
        key=lambda cluster_id: (
            -centroids[cluster_id][response_index],
            -centroids[cluster_id][baseline_index],
            cluster_id,
        ),
    )
    remap = {old: new for new, old in enumerate(ordered)}
    return np.asarray([remap[int(label)] for label in labels], dtype=int)


def cluster_variety_profiles(
    profiles: Mapping[str, VarietyProfile],
    variety_names: Sequence[str],
    *,
    maximum_clusters: int = 8,
) -> VarietyClusterPartition:
    """Cluster supported variety means on the existing response-type basis."""

    ordered = tuple(sorted(variety_names))
    if len(ordered) < 3:
        raise ValueError("At least three supported rice varieties are required")
    matrix = np.asarray([profiles[name].feature_means for name in ordered], dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    silhouette, cluster_count, labels = _select_partition(scaled, maximum_clusters)
    stability = _assignment_stability(scaled, cluster_count, labels)
    labels = _order_cluster_labels(matrix, labels, cluster_count)
    return VarietyClusterPartition(
        variety_names=ordered,
        labels=tuple(int(label) for label in labels),
        cluster_count=cluster_count,
        silhouette=silhouette,
        assignment_stability=stability,
        feature_names=RESPONSE_TYPE_FEATURES,
        feature_center=tuple(float(value) for value in scaler.mean_),
        feature_scale=tuple(float(value) for value in scaler.scale_),
    )


def build_variety_clustering(
    overlay: SourceDatasetOverlay,
    contexts: Mapping[str, TrajectoryContext],
    *,
    minimum_contexts: int = MIN_VARIETY_CONTEXTS,
    maximum_clusters: int = 8,
) -> VarietyClusteringResult:
    """Build the complete rice-variety diagnostic from one LTCCE overlay."""

    if overlay.source_name != "ltcce":
        raise ValueError("Rice-variety clustering is defined only for LTCCE")
    if minimum_contexts < 1:
        raise ValueError("minimum_contexts must be at least one")
    if maximum_clusters < 2:
        raise ValueError("maximum_clusters must be at least two")

    variety_trajectory_ids: dict[str, list[str]] = collections.defaultdict(list)
    for trajectory in overlay.trajectories:
        context = contexts.get(trajectory.trajectory_id)
        if context is not None and context.variety.strip():
            variety_trajectory_ids[context.variety].append(trajectory.trajectory_id)
    if not variety_trajectory_ids:
        raise ValueError("No named rice varieties were found in the LTCCE overlay")

    features, trajectory_exclusions = partition_eligibility(overlay)
    profiles = build_variety_profiles(features, contexts)
    held_out: dict[str, str] = {}
    supported: list[str] = []
    for variety in sorted(variety_trajectory_ids):
        profile = profiles.get(variety)
        if profile is None:
            held_out[variety] = EXCLUSION_NO_ELIGIBLE_TRAJECTORIES
        elif profile.experimental_context_count < minimum_contexts:
            held_out[variety] = EXCLUSION_INSUFFICIENT_CONTEXTS
        else:
            supported.append(variety)

    partition = cluster_variety_profiles(
        profiles,
        supported,
        maximum_clusters=maximum_clusters,
    )
    return VarietyClusteringResult(
        profiles=profiles,
        partition=partition,
        held_out=held_out,
        variety_trajectory_ids={
            variety: tuple(sorted(ids))
            for variety, ids in sorted(variety_trajectory_ids.items())
        },
        trajectory_exclusions=trajectory_exclusions,
        contexts=dict(contexts),
        features=features,
    )


def variety_cluster_members(
    partition: VarietyClusterPartition,
    cluster_id: int,
) -> tuple[str, ...]:
    return tuple(
        variety
        for variety, label in zip(
            partition.variety_names,
            partition.labels,
            strict=True,
        )
        if label == cluster_id
    )


def variety_cluster_centroid(
    result: VarietyClusteringResult,
    cluster_id: int,
) -> dict[str, float]:
    members = variety_cluster_members(result.partition, cluster_id)
    return {
        feature_name: statistics.fmean(
            result.profiles[variety].feature_means[index] for variety in members
        )
        for index, feature_name in enumerate(RESPONSE_TYPE_FEATURES)
    }


__all__ = [
    "EXCLUSION_INSUFFICIENT_CONTEXTS",
    "EXCLUSION_NO_ELIGIBLE_TRAJECTORIES",
    "MIN_VARIETY_CONTEXTS",
    "VarietyClusterPartition",
    "VarietyClusteringResult",
    "VarietyProfile",
    "build_variety_clustering",
    "build_variety_profiles",
    "cluster_variety_profiles",
    "variety_cluster_centroid",
    "variety_cluster_members",
]
