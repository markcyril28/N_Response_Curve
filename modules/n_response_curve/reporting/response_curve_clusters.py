"""Descriptive clustering of replicate-specific N-response trajectories.

This is an **exploratory diagnostic**, not a governed analysis. `ANA-11` disables
the `curve_feature_clustering` analysis family for the current primary release,
so nothing here may be presented as an authorized curve-feature result, and the
record shapes below deliberately do not imitate
`analysis/advanced_analysis.py:_clustering_result` (`stable_curve_feature_clusters`).

Three findings in the LTCCE source drive the design, and each is enforced rather
than assumed:

1. **Trajectories are not comparable across applied-N ladders.** The source uses
   44 distinct N ladders; the five largest cover ~91% of trajectories. Clustering
   raw yield vectors across different N supports would require interpolating onto
   a shared grid, which `modeling.no_extrapolation` forbids. Tier 2 therefore
   clusters only *within* an exact-match ladder stratum.
2. **The ladder is confounded with season and era.** Dry-season blocks carry the
   high ladders (0/50/100/150 in 1970-1991, 0/65/130/195 in 2002-2017) and wet
   seasons the low ones. So a within-stratum partition is also, unavoidably, a
   season/era partition; Tier 3 exists to cluster on a ladder-invariant basis and
   recover response types that cross those blocks.
3. **316 trajectories carry more than one observation at the same N level.** They
   are entirely `Design == "Split-plot"`, 1991-2001 (the `MV-028` multiplicity),
   with a median within-level yield spread of 0.22 t/ha - consistent with an
   uncaptured sub-plot factor rather than measurement replication. Averaging them
   would fabricate a mean curve across two treatments, so they are excluded from
   clustering and reported as their own stratum. This mirrors how
   `source_dataset_overlays.stratify_source_dataset_overlay_by_zero_n_yield`
   handles ambiguity: refuse to collapse, and refuse to draw the connector.

No curve is fitted anywhere in this module. Cluster centroids are means of
observed yields at observed N levels.
"""

from __future__ import annotations

import collections
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from n_response_curve.analysis.values import finite_number
from n_response_curve.reporting.source_dataset_overlays import (
    SourceDatasetOverlay,
    SourceTrajectory,
    _finalize_overlay,
    _trajectory_id,
)

LTCCE_SOURCE_NAME = "ltcce"

# Verbatim LTCCE grouping headers, identical to
# source_dataset_overlays._LTCCE_CONTEXT_HEADERS so the hashed trajectory_id
# computed here joins exactly onto the overlay this module clusters.
LTCCE_CONTEXT_HEADERS = (
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

# A stratum is only clustered when it can support a stable partition. Below this
# it is reported descriptively with the other irregular ladders.
MIN_STRATUM_SIZE = 100

# Silhouette at or below this is reported as a descriptive banding of a
# continuum rather than as evidence of discrete response types.
WEAK_SEPARATION_SILHOUETTE = 0.40

# Reused from advanced_analysis.py so repeated runs are byte-identical.
_KMEANS_SEED = 20260720
_KMEANS_N_INIT = 20
_STABILITY_SEEDS = 5

EXCLUSION_DUPLICATED_N_LEVEL = "duplicated_n_level_mv_028"
EXCLUSION_INCOMPLETE_LADDER = "incomplete_ladder"
EXCLUSION_NO_ZERO_N_ANCHOR = "no_zero_n_anchor"


@dataclass(frozen=True)
class TrajectoryContext:
    """Experimental factors for one replicate-specific trajectory."""

    trajectory_id: str
    design: str
    site: str
    year: int
    season: str
    variety: str
    replicate: str


@dataclass(frozen=True)
class TrajectoryFeatures:
    """Ladder-invariant descriptive features of one observed trajectory.

    Every field is computed from observed points only. `relative_n_at_peak` and
    `saturation_index` are dimensionless/rate-valued precisely so that they stay
    comparable across the 44 applied-N ladders.
    """

    trajectory_id: str
    ladder: tuple[float, ...]
    yields: tuple[float, ...]
    yield_at_zero_n_t_ha: float
    observed_max_yield_t_ha: float
    response_above_zero_n_t_ha: float
    n_at_observed_peak_kg_ha: float
    relative_n_at_peak: float
    saturation_index: float
    decline_from_peak_t_ha: float


@dataclass(frozen=True)
class ClusterPartition:
    """One k-means partition plus the diagnostics needed to read it honestly."""

    partition_id: str
    basis: str
    member_count: int
    cluster_count: int
    silhouette: float
    assignment_stability: float
    feature_names: tuple[str, ...]
    labels: tuple[int, ...]
    trajectory_ids: tuple[str, ...]
    between_cluster_level_variance: float
    between_cluster_shape_variance: float
    replicate_coherence: float

    @property
    def separation_is_weak(self) -> bool:
        return self.silhouette <= WEAK_SEPARATION_SILHOUETTE

    @property
    def dominant_axis(self) -> str:
        """Whether the partition mainly separates yield level or response shape."""

        if self.between_cluster_shape_variance > self.between_cluster_level_variance:
            return "response shape"
        return "yield level"


@dataclass(frozen=True)
class ClusteringResult:
    """Everything the generator needs to write figures and the ledger."""

    contexts: Mapping[str, TrajectoryContext]
    features: Mapping[str, TrajectoryFeatures]
    excluded: Mapping[str, str]
    strata: Mapping[tuple[float, ...], tuple[str, ...]]
    major_ladders: tuple[tuple[float, ...], ...]
    other_ladder_trajectory_ids: tuple[str, ...]
    stratum_partitions: Mapping[tuple[float, ...], ClusterPartition]
    response_type_partition: ClusterPartition | None


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------


def read_ltcce_contexts(
    csv_path: str | Path,
    *,
    encoding: str = "utf-8-sig",
) -> dict[str, TrajectoryContext]:
    """Recover experimental factors keyed by the overlay's hashed trajectory id.

    `read_source_dataset_overlay` intentionally discards the grouping columns, so
    the context needed to interpret a cluster (season, era) is recomputed here
    with the same hash the overlay uses. Nothing is joined positionally.
    """

    contexts: dict[str, TrajectoryContext] = {}
    with Path(csv_path).open(encoding=encoding, newline="") as handle:
        for row in csv.DictReader(handle):
            parts = tuple(
                str(row.get(header, "")).strip() for header in LTCCE_CONTEXT_HEADERS
            )
            trajectory_id = _trajectory_id(LTCCE_SOURCE_NAME, *parts)
            if trajectory_id in contexts:
                continue
            year = finite_number(parts[3])
            contexts[trajectory_id] = TrajectoryContext(
                trajectory_id=trajectory_id,
                design=parts[0],
                site=parts[2],
                year=int(year) if year is not None else -1,
                season=parts[4],
                variety=parts[7],
                replicate=parts[8],
            )
    return contexts


# --------------------------------------------------------------------------
# Eligibility and features
# --------------------------------------------------------------------------


def describe_trajectory(trajectory: SourceTrajectory) -> TrajectoryFeatures | str:
    """Return descriptive features, or the reason the trajectory cannot cluster.

    Fail-closed: a trajectory qualifies only when it carries exactly one finite
    observation at each of at least three distinct applied-N levels *and* is
    anchored at zero N. Duplicated levels are never averaged.
    """

    by_level: dict[float, list[float]] = collections.defaultdict(list)
    for observation in trajectory.observations:
        by_level[observation.n_rate_kg_ha].append(observation.yield_t_ha)

    if any(len(values) > 1 for values in by_level.values()):
        return EXCLUSION_DUPLICATED_N_LEVEL
    if len(by_level) < 3:
        return EXCLUSION_INCOMPLETE_LADDER
    if 0.0 not in by_level:
        return EXCLUSION_NO_ZERO_N_ANCHOR

    ordered = sorted(by_level.items())
    ladder = tuple(level for level, _ in ordered)
    yields = tuple(values[0] for _, values in ordered)

    array_n = np.asarray(ladder, dtype=float)
    array_y = np.asarray(yields, dtype=float)
    peak_index = int(array_y.argmax())
    increments = np.diff(array_y) / np.diff(array_n)

    return TrajectoryFeatures(
        trajectory_id=trajectory.trajectory_id,
        ladder=ladder,
        yields=yields,
        yield_at_zero_n_t_ha=float(array_y[0]),
        observed_max_yield_t_ha=float(array_y.max()),
        response_above_zero_n_t_ha=float(array_y.max() - array_y[0]),
        n_at_observed_peak_kg_ha=float(array_n[peak_index]),
        # Dimensionless position of the observed peak inside this trajectory's
        # own N range: 1.0 means still climbing at the top rate applied.
        relative_n_at_peak=float(array_n[peak_index] / array_n[-1]),
        # Change in marginal yield per kg N from the first step to the last.
        # Negative means the response is flattening or turning down.
        saturation_index=float(increments[-1] - increments[0]),
        decline_from_peak_t_ha=float(array_y[peak_index] - array_y[-1]),
    )


def partition_eligibility(
    overlay: SourceDatasetOverlay,
) -> tuple[dict[str, TrajectoryFeatures], dict[str, str]]:
    """Split an overlay into cluster-eligible features and labelled exclusions."""

    features: dict[str, TrajectoryFeatures] = {}
    excluded: dict[str, str] = {}
    for trajectory in overlay.trajectories:
        described = describe_trajectory(trajectory)
        if isinstance(described, str):
            excluded[trajectory.trajectory_id] = described
        else:
            features[trajectory.trajectory_id] = described
    return features, excluded


def group_by_ladder(
    features: Mapping[str, TrajectoryFeatures],
) -> dict[tuple[float, ...], tuple[str, ...]]:
    """Stratify by exact applied-N ladder.

    Exact match is deliberate. A (0, 45, 90) trajectory is *not* folded into the
    (0, 45, 90, 135) stratum: its 135 kg N/ha response was never observed, and
    assuming one would be extrapolation.
    """

    grouped: dict[tuple[float, ...], list[str]] = collections.defaultdict(list)
    for trajectory_id, feature in features.items():
        grouped[feature.ladder].append(trajectory_id)
    return {ladder: tuple(sorted(ids)) for ladder, ids in grouped.items()}


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------


def _replicate_coherence(
    trajectory_ids: Sequence[str],
    labels: Sequence[int],
    contexts: Mapping[str, TrajectoryContext],
) -> float:
    """Share of experimental units whose replicates all land in one cluster.

    `Rep` is part of the trajectory key, so one plot contributes up to four
    near-identical trajectories. A high value means clusters partly reflect that
    replicate duplication rather than independent agronomic signal.
    """

    units: dict[tuple[str, str, int, str, str], set[int]] = collections.defaultdict(set)
    for trajectory_id, label in zip(trajectory_ids, labels, strict=True):
        context = contexts.get(trajectory_id)
        if context is None:
            continue
        unit = (
            context.design,
            context.site,
            context.year,
            context.season,
            context.variety,
        )
        units[unit].add(int(label))
    multiples = [members for members in units.values() if members]
    if not multiples:
        return float("nan")
    return statistics.fmean(1.0 if len(members) == 1 else 0.0 for members in multiples)


def _select_partition(
    matrix: np.ndarray,
    *,
    minimum_clusters: int,
    maximum_clusters: int,
) -> tuple[float, int, np.ndarray]:
    """Choose k by silhouette over a range, deterministically."""

    distinct = len(np.unique(matrix, axis=0))
    upper = min(maximum_clusters, len(matrix) - 1, distinct)
    best: tuple[float, int, np.ndarray] | None = None
    for cluster_count in range(minimum_clusters, upper + 1):
        try:
            labels = KMeans(
                n_clusters=cluster_count,
                random_state=_KMEANS_SEED,
                n_init=_KMEANS_N_INIT,
            ).fit_predict(matrix)
            if len(set(int(label) for label in labels)) < 2:
                continue
            score = float(silhouette_score(matrix, labels))
        except (FloatingPointError, ValueError):
            continue
        if best is None or score > best[0]:
            best = (score, cluster_count, labels)
    if best is None:
        raise ValueError("No admissible k-means partition was found")
    return best


def _stability(matrix: np.ndarray, cluster_count: int, labels: np.ndarray) -> float:
    """Mean adjusted Rand index of the chosen labels against alternate seeds."""

    alternates = [
        KMeans(
            n_clusters=cluster_count,
            random_state=_KMEANS_SEED + 1 + seed,
            n_init=_KMEANS_N_INIT,
        ).fit_predict(matrix)
        for seed in range(_STABILITY_SEEDS)
    ]
    return statistics.fmean(
        adjusted_rand_score(labels, alternate) for alternate in alternates
    )


def _level_and_shape_variance(
    matrix: np.ndarray,
    labels: np.ndarray,
    cluster_count: int,
) -> tuple[float, float]:
    """Split between-cluster variance into a level part and a shape part.

    Only meaningful when every column is the same physical quantity, i.e. for the
    Tier 2 raw yield vectors. It answers the question the figure raises: does this
    partition separate curves by height, or by form?
    """

    centroids = np.asarray(
        [matrix[labels == index].mean(axis=0) for index in range(cluster_count)]
    )
    levels = centroids.mean(axis=1)
    shapes = centroids - levels[:, None]
    return float(levels.var()), float(shapes.var())


def cluster_stratum(
    ladder: tuple[float, ...],
    trajectory_ids: Sequence[str],
    features: Mapping[str, TrajectoryFeatures],
    contexts: Mapping[str, TrajectoryContext],
    *,
    maximum_clusters: int = 8,
) -> ClusterPartition:
    """Cluster one ladder stratum on raw observed yield vectors.

    The vector is left **unscaled**: all coordinates are grain yield in t/ha, so
    Euclidean distance is already the agronomically meaningful one. Standardizing
    (as `advanced_analysis.py` correctly does for its mixed kg-N/t-ha feature set)
    would inflate the low-variance zero-N coordinate for no reason.
    """

    ordered = tuple(trajectory_ids)
    widths = {len(features[trajectory_id].yields) for trajectory_id in ordered}
    if widths != {len(ladder)}:
        # Exact-ladder grouping is what makes a fixed-width matrix possible. If the
        # grouping key ever loosens, fail here rather than deep inside numpy.
        raise ValueError(
            f"Stratum {ladder!r} mixes trajectory widths {sorted(widths)}; "
            "clustering requires one shared applied-N support"
        )
    matrix = np.asarray(
        [features[trajectory_id].yields for trajectory_id in ordered], dtype=float
    )
    silhouette, cluster_count, labels = _select_partition(
        matrix, minimum_clusters=2, maximum_clusters=maximum_clusters
    )
    level_variance, shape_variance = _level_and_shape_variance(
        matrix, labels, cluster_count
    )
    return ClusterPartition(
        partition_id="ladder_" + "_".join(f"{level:g}" for level in ladder),
        basis="observed yield vector at the stratum's shared N levels (t/ha, unscaled)",
        member_count=len(ordered),
        cluster_count=cluster_count,
        silhouette=silhouette,
        assignment_stability=_stability(matrix, cluster_count, labels),
        feature_names=tuple(f"yield at {level:g} kg N/ha" for level in ladder),
        labels=tuple(int(label) for label in labels),
        trajectory_ids=ordered,
        between_cluster_level_variance=level_variance,
        between_cluster_shape_variance=shape_variance,
        replicate_coherence=_replicate_coherence(ordered, labels, contexts),
    )


RESPONSE_TYPE_FEATURES = (
    "yield_at_zero_n_t_ha",
    "response_above_zero_n_t_ha",
    "relative_n_at_peak",
    "saturation_index",
)


def cluster_response_types(
    features: Mapping[str, TrajectoryFeatures],
    contexts: Mapping[str, TrajectoryContext],
    *,
    maximum_clusters: int = 8,
) -> ClusterPartition:
    """Cluster every eligible trajectory on a ladder-invariant feature basis.

    Unlike Tier 2 this crosses ladder strata, which is admissible only because
    none of the four features requires a shared N support: two are yields in
    t/ha, one is a dimensionless position inside the trajectory's own observed
    range, and one is a change in marginal yield per kg N. Here the units *are*
    heterogeneous, so the matrix is standardized.
    """

    ordered = tuple(sorted(features))
    matrix = np.asarray(
        [
            [getattr(features[trajectory_id], name) for name in RESPONSE_TYPE_FEATURES]
            for trajectory_id in ordered
        ],
        dtype=float,
    )
    scaled = StandardScaler().fit_transform(matrix)
    silhouette, cluster_count, labels = _select_partition(
        scaled, minimum_clusters=2, maximum_clusters=maximum_clusters
    )
    return ClusterPartition(
        partition_id="response_type",
        basis=(
            "ladder-invariant shape features "
            "(zero-N yield, response magnitude, relative N at peak, saturation index; standardized)"
        ),
        member_count=len(ordered),
        cluster_count=cluster_count,
        silhouette=silhouette,
        assignment_stability=_stability(scaled, cluster_count, labels),
        feature_names=RESPONSE_TYPE_FEATURES,
        labels=tuple(int(label) for label in labels),
        trajectory_ids=ordered,
        # A standardized mixed-unit basis has no meaningful level/shape split.
        between_cluster_level_variance=float("nan"),
        between_cluster_shape_variance=float("nan"),
        replicate_coherence=_replicate_coherence(ordered, labels, contexts),
    )


def build_clustering(
    overlay: SourceDatasetOverlay,
    contexts: Mapping[str, TrajectoryContext],
    *,
    minimum_stratum_size: int = MIN_STRATUM_SIZE,
    maximum_clusters: int = 8,
) -> ClusteringResult:
    """Run the whole two-tier scheme over one LTCCE overlay."""

    if overlay.source_name != LTCCE_SOURCE_NAME:
        raise ValueError(
            "Response-curve clustering is defined for the LTCCE overlay only; "
            f"got {overlay.source_name!r}"
        )
    features, excluded = partition_eligibility(overlay)
    if not features:
        raise ValueError("No LTCCE trajectory is eligible for clustering")

    strata = group_by_ladder(features)
    major = tuple(
        ladder
        for ladder, ids in sorted(strata.items(), key=lambda item: (-len(item[1]), item[0]))
        if len(ids) >= minimum_stratum_size
    )
    other = tuple(
        sorted(
            trajectory_id
            for ladder, ids in strata.items()
            if ladder not in major
            for trajectory_id in ids
        )
    )

    partitions: dict[tuple[float, ...], ClusterPartition] = {}
    for ladder in major:
        partitions[ladder] = cluster_stratum(
            ladder,
            strata[ladder],
            features,
            contexts,
            maximum_clusters=maximum_clusters,
        )

    return ClusteringResult(
        contexts=dict(contexts),
        features=features,
        excluded=excluded,
        strata=strata,
        major_ladders=major,
        other_ladder_trajectory_ids=other,
        stratum_partitions=partitions,
        response_type_partition=cluster_response_types(
            features, contexts, maximum_clusters=maximum_clusters
        ),
    )


# --------------------------------------------------------------------------
# Presentation helpers
# --------------------------------------------------------------------------


def subset_overlay(
    overlay: SourceDatasetOverlay,
    trajectory_ids: Sequence[str],
) -> SourceDatasetOverlay:
    """Rebuild an overlay restricted to the given trajectories.

    `selection` is left `None` on purpose: the existing selection dataclasses
    describe threshold selections, and reusing one would mislabel a cluster
    membership as a yield cut. The caller sets its own title instead.
    """

    wanted = set(trajectory_ids)
    chosen = [
        trajectory for trajectory in overlay.trajectories if trajectory.trajectory_id in wanted
    ]
    if not chosen:
        raise ValueError("A cluster overlay requires at least one trajectory")
    return _finalize_overlay(
        source_name=overlay.source_name,
        source_rows=overlay.summary.source_rows,
        excluded_observation_count=overlay.summary.excluded_observation_count,
        observations_by_trajectory={
            trajectory.trajectory_id: list(trajectory.observations) for trajectory in chosen
        },
        excluded_observations_by_trajectory={
            trajectory.trajectory_id: trajectory.excluded_nonfinite_observation_count
            for trajectory in chosen
        },
    )


def composition(
    trajectory_ids: Sequence[str],
    contexts: Mapping[str, TrajectoryContext],
) -> dict[str, Any]:
    """Aggregate season/era composition for a figure caption.

    Aggregate only. Per-trajectory factors stay in the assignment ledger.
    """

    seasons: collections.Counter[str] = collections.Counter()
    years: list[int] = []
    designs: collections.Counter[str] = collections.Counter()
    for trajectory_id in trajectory_ids:
        context = contexts.get(trajectory_id)
        if context is None:
            continue
        seasons[context.season] += 1
        designs[context.design] += 1
        if context.year > 0:
            years.append(context.year)
    total = sum(seasons.values())
    return {
        "season_shares": {
            season: count / total for season, count in seasons.most_common()
        }
        if total
        else {},
        "year_range": (min(years), max(years)) if years else None,
        "designs": tuple(sorted(designs)),
    }


def format_season_shares(shares: Mapping[str, float]) -> str:
    if not shares:
        return "season unknown"
    return ", ".join(f"{season} {100 * share:.0f}%" for season, share in shares.items())


def cluster_members(partition: ClusterPartition, cluster_id: int) -> tuple[str, ...]:
    return tuple(
        trajectory_id
        for trajectory_id, label in zip(
            partition.trajectory_ids, partition.labels, strict=True
        )
        if label == cluster_id
    )


def centroid_summary(
    members: Sequence[str],
    features: Mapping[str, TrajectoryFeatures],
) -> dict[str, float]:
    """Mean observed features of a cluster. No model is fitted."""

    return {
        "yield_at_zero_n_t_ha": statistics.fmean(
            features[member].yield_at_zero_n_t_ha for member in members
        ),
        "observed_max_yield_t_ha": statistics.fmean(
            features[member].observed_max_yield_t_ha for member in members
        ),
        "response_above_zero_n_t_ha": statistics.fmean(
            features[member].response_above_zero_n_t_ha for member in members
        ),
        "relative_n_at_peak": statistics.fmean(
            features[member].relative_n_at_peak for member in members
        ),
        "saturation_index": statistics.fmean(
            features[member].saturation_index for member in members
        ),
    }


def response_type_label(centroid: Mapping[str, float]) -> str:
    """Name a response type from its observed centroid, not from a fit."""

    response = centroid["response_above_zero_n_t_ha"]
    relative_peak = centroid["relative_n_at_peak"]
    baseline = centroid["yield_at_zero_n_t_ha"]

    if response >= 3.0:
        magnitude = "strongly responsive"
    elif response >= 1.5:
        magnitude = "moderately responsive"
    else:
        magnitude = "weakly responsive"

    if relative_peak >= 0.95:
        shape = "peak at the top rate applied (no observed plateau)"
    elif relative_peak >= 0.75:
        shape = "peak near the top rate applied"
    else:
        shape = "peak well below the top rate applied"

    baseline_text = (
        "high zero-N baseline" if baseline >= 4.0 else "low zero-N baseline"
    )
    return f"{magnitude}; {shape}; {baseline_text}"


__all__ = [
    "EXCLUSION_DUPLICATED_N_LEVEL",
    "EXCLUSION_INCOMPLETE_LADDER",
    "EXCLUSION_NO_ZERO_N_ANCHOR",
    "LTCCE_CONTEXT_HEADERS",
    "LTCCE_SOURCE_NAME",
    "MIN_STRATUM_SIZE",
    "RESPONSE_TYPE_FEATURES",
    "WEAK_SEPARATION_SILHOUETTE",
    "ClusterPartition",
    "ClusteringResult",
    "TrajectoryContext",
    "TrajectoryFeatures",
    "build_clustering",
    "centroid_summary",
    "cluster_members",
    "cluster_response_types",
    "cluster_stratum",
    "composition",
    "describe_trajectory",
    "format_season_shares",
    "group_by_ladder",
    "partition_eligibility",
    "read_ltcce_contexts",
    "response_type_label",
    "subset_overlay",
]
