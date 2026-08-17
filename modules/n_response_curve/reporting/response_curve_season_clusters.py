"""Descriptive clustering of LTCCE N-response trajectories *within* each season.

This is the season-stratified companion to :mod:`response_curve_clusters` and
:mod:`response_curve_variety_clusters`.  It exists because of the confounding the
trajectory module documents: LTCCE assigns the high applied-N ladders to dry
season blocks and the low ones to the wet seasons, so a pooled partition —
whether over raw yield vectors or over ladder-invariant features — is partly a
season contrast wearing a response-type label.

The remedy here is stratification rather than adjustment.  Season is held fixed,
and the existing four ladder-invariant features are clustered *inside* DS, EWS,
and LWS separately.  Whatever structure survives is structure that is not the
season contrast.

Three properties of the design are deliberate:

1. **Cluster labels are local to their season.**  Each season chooses its own
   ``k`` by silhouette and its own centroids, so ``DS`` cluster 1 and ``LWS``
   cluster 1 are unrelated objects.  Nothing in this module cross-matches them,
   and every artifact it writes says so.
2. **Ladder concordance is measured, not assumed away.**  Stratifying by season
   does not remove the ladder: DS is ~95% two ladders, EWS and LWS ~90% three.
   A within-season partition can therefore still re-derive the applied-N design.
   :func:`ladder_concordance` scores that directly with the adjusted Rand index
   of the cluster labels against the ladder labels, and the score is reported
   whichever way it comes out.
3. **No curve is fitted, and no season is compared to another on the strength of
   these partitions.**  The cross-season figure this module supports compares
   observed feature means, which remain confounded with ladder and era; it is
   descriptive context for the within-season partitions, not a season effect.

Like its two companions, this is an exploratory diagnostic outside the governed
analysis inventory (`ANA-11`).
"""

from __future__ import annotations

import collections
import math
import statistics
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score

from n_response_curve.reporting.response_curve_clusters import (
    LTCCE_SOURCE_NAME,
    RESPONSE_TYPE_FEATURES,
    ClusterPartition,
    TrajectoryContext,
    TrajectoryFeatures,
    cluster_response_types,
    cluster_stratum,
    partition_eligibility,
)
from n_response_curve.reporting.response_curve_variety_clusters import _context_key
from n_response_curve.reporting.source_dataset_overlays import SourceDatasetOverlay

# Calendar order of the LTCCE cropping seasons at the IRRI long-term site.
SEASON_DISPLAY_ORDER = ("DS", "EWS", "LWS")

SEASON_LABELS = {
    "DS": "dry season",
    "EWS": "early wet season",
    "LWS": "late wet season",
}

# A season is clustered only when it can support a stable partition; below this
# it is carried descriptively with its exclusions and profile only.
MIN_SEASON_SIZE = 100

# Adjusted Rand index of the within-season cluster labels against the applied-N
# ladder labels, at or above which the partition is disclosed as substantially
# re-deriving the experimental design rather than describing N response. The
# ladder is a nuisance grouping here, so even a modest agreement is worth
# stating plainly.
LADDER_DRIVEN_ADJUSTED_RAND = 0.20

# Share of one ladder inside a single cluster at which that cluster is called
# out individually, regardless of the partition-wide score above.
LADDER_DOMINATED_CLUSTER_SHARE = 0.90

UNCLUSTERED_TOO_FEW_TRAJECTORIES = "too_few_eligible_trajectories"
UNCLUSTERED_NO_ADMISSIBLE_PARTITION = "no_admissible_partition"

# Nested level. Inside one within-season cluster, members still sit on several
# applied-N ladders, and the ladder is the only fertilizer contrast LTCCE varies
# (the source carries no P or K column at all — its own workbook metadata names
# the treatment factors as "N fertilizer rate, Variety"). Splitting a cluster by
# exact ladder therefore separates it by experimental design, and — because a
# ladder sub-stratum shares one applied-N support — it is the first point at
# which raw observed yield vectors may legitimately be clustered.
MIN_LADDER_SUBSTRATUM = 30
MIN_SUBCLUSTERED_SUBSTRATUM = 60

# The same decomposition run over a whole season has roughly four times the
# membership of one cluster, so a lower sub-stratum floor still leaves every
# admitted stratum well supported. At 20 the two one-off DS design years —
# 0/40/60/120 in 1968 (23 trajectories) and 0/60/100/140 in 1969 (20) — get
# their own figures instead of disappearing into the minor-ladder pool. They
# still fall below MIN_SUBCLUSTERED_SUBSTRATUM, so they are shown, not
# partitioned.
MIN_SEASON_LADDER_SUBSTRATUM = 20

SCOPE_SEASON = "season"

# Share of experimental units whose replicates all land in the same sub-cluster,
# at or above which the sub-partition is disclosed as tracking plot identity.
# `Rep` is part of the trajectory key, so one plot contributes up to four
# near-identical trajectories; inside a single ladder they are close enough that
# k-means will co-assign them, and a partition that mostly separates plots is not
# reporting an independent agronomic contrast.
REPLICATE_BOUND_CO_ASSIGNMENT = 0.70


@dataclass(frozen=True)
class SeasonProfile:
    """Observed, replicate-free description of one cropping season."""

    season: str
    trajectory_ids: tuple[str, ...]
    excluded_trajectory_ids: tuple[str, ...]
    exclusions_by_reason: Mapping[str, int]
    experimental_context_count: int
    year_range: tuple[int, int] | None
    ladder_shares: Mapping[str, float]
    distinct_ladder_count: int
    variety_count: int
    feature_means: tuple[float, ...]
    feature_standard_errors: tuple[float, ...]

    def feature_map(self) -> dict[str, float]:
        return dict(zip(RESPONSE_TYPE_FEATURES, self.feature_means, strict=True))


@dataclass(frozen=True)
class LadderConcordance:
    """How much a within-season partition merely recovers the applied-N design."""

    season: str
    adjusted_rand: float
    contingency: Mapping[int, Mapping[str, int]]
    dominant_ladder_share: Mapping[int, float]

    @property
    def is_ladder_driven(self) -> bool:
        return self.adjusted_rand >= LADDER_DRIVEN_ADJUSTED_RAND

    def ladder_dominated_clusters(self) -> tuple[int, ...]:
        return tuple(
            cluster_id
            for cluster_id, share in sorted(self.dominant_ladder_share.items())
            if share >= LADDER_DOMINATED_CLUSTER_SHARE
        )


@dataclass(frozen=True)
class SeasonClusteringResult:
    """Per-season partitions plus everything needed to report them honestly."""

    contexts: Mapping[str, TrajectoryContext]
    features: Mapping[str, TrajectoryFeatures]
    excluded: Mapping[str, str]
    seasons: tuple[str, ...]
    season_trajectory_ids: Mapping[str, tuple[str, ...]]
    profiles: Mapping[str, SeasonProfile]
    partitions: Mapping[str, ClusterPartition]
    ladder_concordance: Mapping[str, LadderConcordance]
    unclustered_seasons: Mapping[str, str]


def season_label(season: str) -> str:
    """Expand an LTCCE season code, leaving unknown codes untouched."""

    described = SEASON_LABELS.get(season)
    return f"{season} ({described})" if described else season or "season unrecorded"


def order_seasons(seasons: Sequence[str]) -> tuple[str, ...]:
    """Calendar order first, then anything unrecognized, alphabetically."""

    known = [season for season in SEASON_DISPLAY_ORDER if season in set(seasons)]
    other = sorted(set(seasons) - set(SEASON_DISPLAY_ORDER))
    return tuple(known + other)


def ladder_text(ladder: Sequence[float]) -> str:
    return "/".join(f"{level:g}" for level in ladder)


def group_by_season(
    trajectory_ids: Sequence[str],
    contexts: Mapping[str, TrajectoryContext],
) -> dict[str, tuple[str, ...]]:
    """Split trajectory ids by recorded season, dropping ids with no context."""

    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for trajectory_id in trajectory_ids:
        context = contexts.get(trajectory_id)
        if context is None or not context.season.strip():
            continue
        grouped[context.season].append(trajectory_id)
    return {season: tuple(sorted(ids)) for season, ids in grouped.items()}


def _context_weighted_moments(
    trajectory_ids: Sequence[str],
    features: Mapping[str, TrajectoryFeatures],
    contexts: Mapping[str, TrajectoryContext],
) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    """Mean and standard error of the feature basis over replicate-free contexts.

    Replicates are averaged inside their own design-site-year-season-variety-
    ladder context first, using the identical key the variety module uses, so a
    year with four replicates does not outweigh a year with one. The standard
    error is taken across those context means and describes the spread of the
    season's experimental contexts, not sampling error of a fitted quantity.
    """

    grouped: dict[tuple, list[np.ndarray]] = collections.defaultdict(list)
    for trajectory_id in trajectory_ids:
        feature = features.get(trajectory_id)
        context = contexts.get(trajectory_id)
        if feature is None or context is None:
            continue
        vector = np.asarray(
            [getattr(feature, name) for name in RESPONSE_TYPE_FEATURES], dtype=float
        )
        grouped[_context_key(context, feature)].append(vector)

    if not grouped:
        nan = tuple(float("nan") for _ in RESPONSE_TYPE_FEATURES)
        return nan, nan, 0

    context_means = np.asarray(
        [np.mean(vectors, axis=0) for vectors in grouped.values()], dtype=float
    )
    means = tuple(float(value) for value in context_means.mean(axis=0))
    if len(context_means) < 2:
        errors = tuple(float("nan") for _ in RESPONSE_TYPE_FEATURES)
    else:
        errors = tuple(
            float(value)
            for value in context_means.std(axis=0, ddof=1) / np.sqrt(len(context_means))
        )
    return means, errors, len(context_means)


def build_season_profile(
    season: str,
    eligible_ids: Sequence[str],
    excluded_ids: Sequence[str],
    features: Mapping[str, TrajectoryFeatures],
    contexts: Mapping[str, TrajectoryContext],
    excluded: Mapping[str, str],
) -> SeasonProfile:
    """Describe one season from observed values only."""

    ladders: collections.Counter[str] = collections.Counter()
    varieties: set[str] = set()
    years: list[int] = []
    for trajectory_id in eligible_ids:
        ladders[ladder_text(features[trajectory_id].ladder)] += 1
    for trajectory_id in (*eligible_ids, *excluded_ids):
        context = contexts.get(trajectory_id)
        if context is None:
            continue
        if context.year > 0:
            years.append(context.year)
        if context.variety.strip():
            varieties.add(context.variety)

    ladder_total = sum(ladders.values())
    means, errors, context_count = _context_weighted_moments(
        eligible_ids, features, contexts
    )
    return SeasonProfile(
        season=season,
        trajectory_ids=tuple(sorted(eligible_ids)),
        excluded_trajectory_ids=tuple(sorted(excluded_ids)),
        exclusions_by_reason=dict(
            sorted(
                collections.Counter(
                    excluded[trajectory_id]
                    for trajectory_id in excluded_ids
                    if trajectory_id in excluded
                ).items()
            )
        ),
        experimental_context_count=context_count,
        year_range=(min(years), max(years)) if years else None,
        ladder_shares=(
            {ladder: count / ladder_total for ladder, count in ladders.most_common()}
            if ladder_total
            else {}
        ),
        distinct_ladder_count=len(ladders),
        variety_count=len(varieties),
        feature_means=means,
        feature_standard_errors=errors,
    )


def cluster_season(
    season: str,
    trajectory_ids: Sequence[str],
    features: Mapping[str, TrajectoryFeatures],
    contexts: Mapping[str, TrajectoryContext],
    *,
    maximum_clusters: int = 8,
) -> ClusterPartition:
    """Cluster one season on the shared ladder-invariant feature basis.

    The basis and the selection rule are exactly those of
    :func:`response_curve_clusters.cluster_response_types`; only the membership
    is restricted. Reusing that function rather than reimplementing it keeps the
    two products' feature definitions and seeds identical, so a season partition
    can be read against the pooled one without wondering whether the difference
    is methodological.
    """

    subset = {
        trajectory_id: features[trajectory_id]
        for trajectory_id in trajectory_ids
        if trajectory_id in features
    }
    if len(subset) < 3:
        raise ValueError(f"Season {season!r} has too few eligible trajectories")
    partition = cluster_response_types(
        subset,
        contexts,
        maximum_clusters=maximum_clusters,
    )
    return replace(
        partition,
        partition_id=f"season_{season.casefold()}",
        basis=(
            f"{partition.basis}; membership restricted to season {season}, so the "
            "partition cannot be reporting the season contrast"
        ),
    )


def ladder_concordance(
    season: str,
    partition: ClusterPartition,
    features: Mapping[str, TrajectoryFeatures],
) -> LadderConcordance:
    """Score how far a within-season partition reproduces the applied-N ladder.

    Season stratification removes the season contrast but not the ladder: the
    ladders are unevenly represented inside every season. If the cluster labels
    agree strongly with the ladder labels, the partition is a relabelling of the
    experimental design and must be read as one.
    """

    ladders = [
        ladder_text(features[trajectory_id].ladder)
        for trajectory_id in partition.trajectory_ids
    ]
    codes = {ladder: index for index, ladder in enumerate(sorted(set(ladders)))}
    score = float(
        adjusted_rand_score(
            list(partition.labels), [codes[ladder] for ladder in ladders]
        )
    )

    contingency: dict[int, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for label, ladder in zip(partition.labels, ladders, strict=True):
        contingency[int(label)][ladder] += 1

    dominant: dict[int, float] = {}
    for cluster_id, counts in contingency.items():
        total = sum(counts.values())
        dominant[cluster_id] = max(counts.values()) / total if total else float("nan")

    return LadderConcordance(
        season=season,
        adjusted_rand=score,
        contingency={
            cluster_id: dict(counts.most_common())
            for cluster_id, counts in sorted(contingency.items())
        },
        dominant_ladder_share=dominant,
    )


def build_season_clustering(
    overlay: SourceDatasetOverlay,
    contexts: Mapping[str, TrajectoryContext],
    *,
    minimum_season_size: int = MIN_SEASON_SIZE,
    maximum_clusters: int = 8,
) -> SeasonClusteringResult:
    """Run the season-stratified scheme over one LTCCE overlay."""

    if overlay.source_name != LTCCE_SOURCE_NAME:
        raise ValueError(
            "Season-stratified clustering is defined for the LTCCE overlay only; "
            f"got {overlay.source_name!r}"
        )
    if minimum_season_size < 3:
        raise ValueError("minimum_season_size must be at least three")
    if maximum_clusters < 2:
        raise ValueError("maximum_clusters must be at least two")

    features, excluded = partition_eligibility(overlay)
    if not features:
        raise ValueError("No LTCCE trajectory is eligible for clustering")

    all_ids = tuple(trajectory.trajectory_id for trajectory in overlay.trajectories)
    season_trajectory_ids = group_by_season(all_ids, contexts)
    if not season_trajectory_ids:
        raise ValueError("No LTCCE trajectory carries a recorded season")

    seasons = order_seasons(tuple(season_trajectory_ids))
    profiles: dict[str, SeasonProfile] = {}
    partitions: dict[str, ClusterPartition] = {}
    concordance: dict[str, LadderConcordance] = {}
    unclustered: dict[str, str] = {}

    for season in seasons:
        ids = season_trajectory_ids[season]
        eligible = tuple(
            trajectory_id for trajectory_id in ids if trajectory_id in features
        )
        held_out = tuple(
            trajectory_id for trajectory_id in ids if trajectory_id in excluded
        )
        profiles[season] = build_season_profile(
            season, eligible, held_out, features, contexts, excluded
        )
        if len(eligible) < minimum_season_size:
            unclustered[season] = UNCLUSTERED_TOO_FEW_TRAJECTORIES
            continue
        try:
            partition = cluster_season(
                season,
                eligible,
                features,
                contexts,
                maximum_clusters=maximum_clusters,
            )
        except ValueError:
            unclustered[season] = UNCLUSTERED_NO_ADMISSIBLE_PARTITION
            continue
        partitions[season] = partition
        concordance[season] = ladder_concordance(season, partition, features)

    if not partitions:
        raise ValueError("No LTCCE season supports a partition at this threshold")

    return SeasonClusteringResult(
        contexts=dict(contexts),
        features=features,
        excluded=excluded,
        seasons=seasons,
        season_trajectory_ids=season_trajectory_ids,
        profiles=profiles,
        partitions=partitions,
        ladder_concordance=concordance,
        unclustered_seasons=unclustered,
    )


# --------------------------------------------------------------------------
# Nested level: applied-N ladder sub-strata inside one within-season cluster
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LadderSubstratum:
    """One exact applied-N ladder inside a cluster, or inside a whole season."""

    season: str
    scope: str
    cluster_id: int | None
    ladder: tuple[float, ...]
    step_kg_ha: float | None
    trajectory_ids: tuple[str, ...]
    partition: ClusterPartition | None
    not_subclustered_reason: str

    @property
    def step_text(self) -> str:
        if self.step_kg_ha is None:
            return "irregular steps"
        return f"uniform {self.step_kg_ha:g} kg N/ha steps"


@dataclass(frozen=True)
class ClusterSubstructure:
    """The applied-N ladder decomposition of one cluster, or of a whole season."""

    season: str
    scope: str
    cluster_id: int | None
    member_count: int
    substrata: tuple[LadderSubstratum, ...]
    minor_ladder_trajectory_ids: tuple[str, ...]
    minor_ladder_count: int

    @property
    def distinct_ladder_count(self) -> int:
        return len(self.substrata) + self.minor_ladder_count

    @property
    def is_season_scope(self) -> bool:
        return self.cluster_id is None

    @property
    def scope_text(self) -> str:
        """How to name this decomposition's parent in prose and figure titles."""

        if self.cluster_id is None:
            return "the whole season"
        return f"cluster {self.cluster_id + 1}"


@dataclass(frozen=True)
class LadderRecord:
    """One applied-N ladder as observed inside one season."""

    ladder: tuple[float, ...]
    step_kg_ha: float | None
    trajectory_count: int
    share: float
    year_range: tuple[int, int] | None
    distinct_year_count: int
    variety_count: int

    @property
    def step_text(self) -> str:
        if self.step_kg_ha is None:
            return "irregular"
        return f"{self.step_kg_ha:g}"

    @property
    def year_text(self) -> str:
        if self.year_range is None:
            return "years unknown"
        low, high = self.year_range
        return f"{low}" if low == high else f"{low}-{high}"


def season_ladder_records(
    result: SeasonClusteringResult,
    season: str,
) -> tuple[LadderRecord, ...]:
    """Describe every applied-N ladder observed in one season, largest first.

    The year range is the interesting column. Inside a season the ladders turn
    out to be disjoint in time — LTCCE replaced one design with the next rather
    than running them side by side — so a ladder label is also an era label, and
    any comparison between ladders is a comparison between calendar periods.
    """

    profile = result.profiles.get(season)
    if profile is None:
        return ()
    grouped: dict[tuple[float, ...], list[str]] = collections.defaultdict(list)
    for trajectory_id in profile.trajectory_ids:
        grouped[result.features[trajectory_id].ladder].append(trajectory_id)

    total = sum(len(ids) for ids in grouped.values())
    records: list[LadderRecord] = []
    for ladder, ids in grouped.items():
        years = [
            result.contexts[trajectory_id].year
            for trajectory_id in ids
            if trajectory_id in result.contexts
            and result.contexts[trajectory_id].year > 0
        ]
        records.append(
            LadderRecord(
                ladder=ladder,
                step_kg_ha=uniform_step(ladder),
                trajectory_count=len(ids),
                share=len(ids) / total if total else 0.0,
                year_range=(min(years), max(years)) if years else None,
                distinct_year_count=len(set(years)),
                variety_count=cluster_variety_span(ids, result.contexts),
            )
        )
    return tuple(
        sorted(records, key=lambda record: (-record.trajectory_count, record.ladder))
    )


def ladders_are_disjoint_in_time(
    records: Sequence[LadderRecord],
    *,
    minimum_trajectories: int = MIN_LADDER_SUBSTRATUM,
) -> bool:
    """True when no two *major* ladders in a season share a calendar year.

    Restricted to ladders above the sub-stratum minimum, because those are the
    ones the nested product actually separates. The long tail of one-off ladders
    is scattered across the whole record and would defeat the test without
    saying anything about the strata a reader will compare.
    """

    spans = sorted(
        record.year_range
        for record in records
        if record.year_range and record.trajectory_count >= minimum_trajectories
    )
    return all(
        earlier[1] < later[0]
        for earlier, later in zip(spans, spans[1:], strict=False)
    )


def uniform_step(ladder: Sequence[float]) -> float | None:
    """Return the ladder's constant N increment, or None when it is irregular.

    The DS ladders that carry this source are arithmetic — 0/50/100/150 steps by
    50, 0/65/130/195 by 65 — so the step is a real property of the design rather
    than a derived summary. Ladders such as 0/40/60/120 have no single step and
    are reported as irregular instead of being given a misleading mean.
    """

    if len(ladder) < 2:
        return None
    steps = [
        round(float(high) - float(low), 6)
        for low, high in zip(ladder, ladder[1:], strict=False)
    ]
    first = steps[0]
    if any(abs(step - first) > 1e-9 for step in steps):
        return None
    return first


def build_ladder_substructure(
    result: SeasonClusteringResult,
    season: str,
    members: Sequence[str],
    *,
    scope: str,
    cluster_id: int | None = None,
    minimum_substratum: int = MIN_LADDER_SUBSTRATUM,
    minimum_subclustered: int = MIN_SUBCLUSTERED_SUBSTRATUM,
    maximum_clusters: int = 6,
) -> ClusterSubstructure:
    """Split a set of trajectories by exact applied-N ladder, then cluster inside.

    Two nested levels with different justifications. The outer split is by
    ladder, which is exact-match and never interpolated, exactly as
    :func:`response_curve_clusters.group_by_ladder` does — a (0, 45, 90)
    trajectory is not folded into (0, 45, 90, 135) because its 135 kg N/ha
    response was never observed. The inner split reuses
    :func:`response_curve_clusters.cluster_stratum`, which clusters the *raw*
    observed yield vector unscaled; that is admissible here and only here,
    because every member of a ladder sub-stratum shares one applied-N support.

    The membership is a parameter so the identical decomposition serves both
    scopes: one within-season cluster, or a whole season. Nothing about the
    method changes between them — only how many trajectories arrive.
    """

    if not members:
        raise ValueError(f"Season {season!r} scope {scope!r} is empty")

    grouped: dict[tuple[float, ...], list[str]] = collections.defaultdict(list)
    for trajectory_id in members:
        grouped[result.features[trajectory_id].ladder].append(trajectory_id)

    major = sorted(
        (
            (ladder, tuple(sorted(ids)))
            for ladder, ids in grouped.items()
            if len(ids) >= minimum_substratum
        ),
        key=lambda item: (-len(item[1]), item[0]),
    )
    minor_ladders = [
        ladder for ladder, ids in grouped.items() if len(ids) < minimum_substratum
    ]

    substrata: list[LadderSubstratum] = []
    for ladder, ids in major:
        sub_partition: ClusterPartition | None = None
        reason = ""
        if len(ids) < minimum_subclustered:
            reason = "too_few_trajectories_for_a_stable_sub_partition"
        else:
            try:
                sub_partition = replace(
                    cluster_stratum(
                        ladder,
                        ids,
                        result.features,
                        result.contexts,
                        maximum_clusters=maximum_clusters,
                    ),
                    partition_id=(
                        f"season_{season.casefold()}_{scope}_"
                        + "_".join(f"{level:g}" for level in ladder)
                    ),
                )
            except ValueError:
                reason = "no_admissible_sub_partition"
        substrata.append(
            LadderSubstratum(
                season=season,
                scope=scope,
                cluster_id=cluster_id,
                ladder=ladder,
                step_kg_ha=uniform_step(ladder),
                trajectory_ids=ids,
                partition=sub_partition,
                not_subclustered_reason=reason,
            )
        )

    minor_ids = tuple(
        sorted(
            trajectory_id
            for ladder in minor_ladders
            for trajectory_id in grouped[ladder]
        )
    )
    return ClusterSubstructure(
        season=season,
        scope=scope,
        cluster_id=cluster_id,
        member_count=len(members),
        substrata=tuple(substrata),
        minor_ladder_trajectory_ids=minor_ids,
        minor_ladder_count=len(minor_ladders),
    )


def build_cluster_substructure(
    result: SeasonClusteringResult,
    season: str,
    cluster_id: int,
    **kwargs: Any,
) -> ClusterSubstructure:
    """Decompose one within-season cluster by applied-N ladder."""

    partition = result.partitions.get(season)
    if partition is None:
        raise ValueError(f"Season {season!r} carries no partition to decompose")
    return build_ladder_substructure(
        result,
        season,
        season_cluster_members(partition, cluster_id),
        scope=f"cluster_{cluster_id + 1}",
        cluster_id=cluster_id,
        **kwargs,
    )


def build_season_ladder_substructure(
    result: SeasonClusteringResult,
    season: str,
    *,
    minimum_substratum: int = MIN_SEASON_LADDER_SUBSTRATUM,
    **kwargs: Any,
) -> ClusterSubstructure:
    """Decompose a whole season by applied-N ladder, ignoring its response clusters.

    This is the season-level counterpart to the per-cluster decomposition: it
    asks what the applied-N design alone separates, before any response-type
    partition is imposed. Because it draws on roughly four times the membership,
    it admits a smaller sub-stratum by default.
    """

    profile = result.profiles.get(season)
    if profile is None:
        raise ValueError(f"Season {season!r} has no profile to decompose")
    return build_ladder_substructure(
        result,
        season,
        profile.trajectory_ids,
        scope=SCOPE_SEASON,
        cluster_id=None,
        minimum_substratum=minimum_substratum,
        **kwargs,
    )


def build_season_substructures(
    result: SeasonClusteringResult,
    season: str,
    **kwargs: Any,
) -> tuple[ClusterSubstructure, ...]:
    """Decompose every cluster of one season, in cluster order."""

    partition = result.partitions.get(season)
    if partition is None:
        return ()
    return tuple(
        build_cluster_substructure(result, season, cluster_id, **kwargs)
        for cluster_id in range(partition.cluster_count)
    )


def season_cluster_members(
    partition: ClusterPartition,
    cluster_id: int,
) -> tuple[str, ...]:
    return tuple(
        trajectory_id
        for trajectory_id, label in zip(
            partition.trajectory_ids, partition.labels, strict=True
        )
        if label == cluster_id
    )


def season_feature_matrix(
    result: SeasonClusteringResult,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Season-by-feature means and standard errors for the comparison figure."""

    seasons = tuple(
        season for season in result.seasons if season in result.profiles
    )
    means = np.asarray(
        [result.profiles[season].feature_means for season in seasons], dtype=float
    )
    errors = np.asarray(
        [result.profiles[season].feature_standard_errors for season in seasons],
        dtype=float,
    )
    return seasons, means, errors


def replicate_coherence_text(partition: ClusterPartition) -> str:
    coherence = partition.replicate_coherence
    if coherence != coherence:  # NaN
        return "n/a"
    return f"{100 * coherence:.0f}%"


def replicate_binding_caveat(partition: ClusterPartition) -> str:
    """State when a partition mostly separates plots rather than N response.

    Returned as two lines. Figure titles here are already long and are joined on
    newlines by the caller, so a single ~180-character sentence would overrun the
    canvas width and be clipped at both ends.
    """

    coherence = partition.replicate_coherence
    if coherence != coherence or coherence < REPLICATE_BOUND_CO_ASSIGNMENT:
        return ""
    return (
        f"{100 * coherence:.0f}% of experimental units have all their replicates "
        "in one sub-cluster: at this level\nthe split largely separates plots, so "
        "the effective sample is the unit count, not the trajectory count"
    )


def cluster_variety_span(
    members: Sequence[str],
    contexts: Mapping[str, TrajectoryContext],
) -> int:
    return len(
        {
            contexts[trajectory_id].variety
            for trajectory_id in members
            if trajectory_id in contexts and contexts[trajectory_id].variety.strip()
        }
    )


def cluster_year_range(
    members: Sequence[str],
    contexts: Mapping[str, TrajectoryContext],
) -> tuple[int, int] | None:
    years = [
        contexts[trajectory_id].year
        for trajectory_id in members
        if trajectory_id in contexts and contexts[trajectory_id].year > 0
    ]
    return (min(years), max(years)) if years else None


def cluster_ladder_shares(
    members: Sequence[str],
    features: Mapping[str, TrajectoryFeatures],
) -> dict[str, float]:
    counts = collections.Counter(
        ladder_text(features[trajectory_id].ladder)
        for trajectory_id in members
        if trajectory_id in features
    )
    total = sum(counts.values())
    if not total:
        return {}
    return {ladder: count / total for ladder, count in counts.most_common()}


def format_shares(shares: Mapping[str, float], *, limit: int = 3) -> str:
    if not shares:
        return "none recorded"
    items = list(shares.items())[:limit]
    text = ", ".join(f"{name} {100 * share:.0f}%" for name, share in items)
    remainder = len(shares) - len(items)
    return f"{text}, +{remainder} more" if remainder > 0 else text


def mean_feature(
    members: Sequence[str],
    features: Mapping[str, TrajectoryFeatures],
    name: str,
) -> float:
    return statistics.fmean(getattr(features[member], name) for member in members)


# --------------------------------------------------------------------------
# Experimental-factor strata
# --------------------------------------------------------------------------
#
# A third nesting axis, alongside the applied-N ladder decomposition above.
# Instead of splitting a season by the fertilizer support, split it by a
# recorded experimental factor — the varietal slot `VarCode`, the plot design,
# or the variety designation — and cluster inside each level.
#
# The methodological difference from `build_ladder_substructure` is the only
# thing that matters here and is not cosmetic. A ladder sub-stratum shares one
# applied-N support, which is what licenses clustering its *raw* observed yield
# vector. A factor level does not: a single `VarCode` spans 1968-2017 and every
# ladder LTCCE ever ran. So factor levels are clustered on the standardized
# ladder-invariant feature basis, exactly as the season partitions are.

FACTOR_VARIETY_CODE = "varcode"
FACTOR_DESIGN = "design"
FACTOR_VARIETY = "variety"
FACTOR_PLANTING_YEAR = "planting_year"

# A factor level is drawn as its own figure at or above this many trajectories;
# below it, levels are pooled into one descriptive "minor levels" figure.
MIN_FACTOR_STRATUM = 30

# ...and sub-clustered only at or above this, the same floor the ladder
# sub-strata use, so the two nested levels are not held to different standards.
MIN_SUBCLUSTERED_FACTOR_STRATUM = MIN_SUBCLUSTERED_SUBSTRATUM

FACTOR_LEVEL_UNRECORDED = "unrecorded"

# Below the LADDER_DRIVEN_ADJUSTED_RAND disclosure threshold there are still two
# very different situations, and calling both "no association" would be false.
# At or under this the association is reported as negligible; between the two,
# as present but under the threshold.
NEGLIGIBLE_ADJUSTED_RAND = 0.05

_NOT_SUBCLUSTERED_TOO_FEW = "too_few_trajectories_for_a_stable_sub_partition"
_NOT_SUBCLUSTERED_NO_PARTITION = "no_admissible_sub_partition"


def decade_band(raw: str) -> str:
    """Bin a recorded planting year into its calendar decade, or "" if unusable.

    Every LTCCE planting year holds about 24 cluster-eligible trajectories, so
    no individual year clears `MIN_FACTOR_STRATUM` and a year-by-year
    decomposition would pool the entire season into `minor_levels`. Binning to
    the decade is what makes the time axis a usable stratification; the
    per-year detail is carried by the annual trend records below instead of by
    forty separate figures.
    """

    try:
        year = int(str(raw).strip())
    except (TypeError, ValueError):
        return ""
    if year <= 0:
        return ""
    return f"{year - year % 10}s"


VARIETY_RELEASE_YEARS: Mapping[str, int] = {
    # Year of release, keyed by the exact `Variety` string LTCCE records.
    #
    # Source: the National Seed Industry Council registry of registered rice
    # varieties (`nsic.buplant.da.gov.ph/registry.php`, as published in the
    # DA-PhilRice RDAC inventory `nsic-registered-rice-varieties`, 1955-2024),
    # matched by IR number, by PSB/NSIC `Rc` number, or by the line designation
    # the registry records.  `IR8` is the single exception: it carries IRRI's
    # own 1966 release rather than the 1968 Philippine Seed Board approval,
    # because that is the year the IRRI literature this experiment belongs to
    # cites.  Every other entry is a Philippine approval year, and every one of
    # them is at or before the level's first planting year in the LTCCE record.
    #
    # A level recorded as a *breeding-line* designation has no entry, even when
    # that line was later released under a variety name (`IR59682-132-1-1-2`
    # became PSB Rc52 in 1997, `IR78581-12-3-2-2` became NSIC Rc222 in 2009).
    # While it was grown under the line designation it was not a released
    # variety, so a release year on it would postdate its own trajectories.
    "IR8": 1966,
    "BPI121-407": 1966,
    "C4-63G": 1968,
    "IR20": 1969,
    "IR22": 1970,
    "IR24": 1971,
    "IR26": 1973,
    "IR28": 1975,
    "IR30": 1975,
    "IR36": 1976,
    "IR40": 1977,
    "IR50": 1980,
    "IR52": 1980,
    "IR54": 1980,
    "IR56": 1982,
    "IR58": 1983,
    "IR64": 1985,
    "IR72": 1988,
    "PSBRc52": 1997,
    "PSBRc54": 1997,
    "PSBRc82": 2000,
    "NSIC Rc110": 2002,
    "NSIC Rc158": 2006,
    "IRRI 146 (NSIC Rc158)": 2006,
    "IR03A433 (NSIC Rc212)": 2009,
    "IRRI 154 (NSIC Rc222)": 2009,
    "IRRI 156 (NSIC Rc238)": 2011,
}

VARIETY_RELEASE_YEAR_SOURCE = (
    "NSIC registry of registered rice varieties (DA-PhilRice RDAC inventory, "
    "1955-2024); IR8 carries IRRI's 1966 release rather than the 1968 "
    "Philippine Seed Board approval"
)


@dataclass(frozen=True)
class FactorDefinition:
    """One recorded experimental factor a season can be decomposed by."""

    key: str
    attribute: str
    source_column: str
    directory: str
    title: str
    level_prefix: str
    # Repeated on every figure and in the README. Each of these factors is
    # misreadable in a specific way, and naming the way is the disclosure.
    identity_caveat: str
    # Optional derivation from the recorded value to the level actually used.
    # Only the calendar factors need one: the source records a year, and the
    # level is its decade. Returning "" routes the trajectory to `unrecorded`.
    level_binner: Callable[[str], str] | None = None
    # Levels of an ordinal factor are ordered by name, not by size — reading a
    # time axis out of calendar order is worse than reading a small stratum
    # late.
    ordinal: bool = False
    # Release year per level, where the factor has one. Only the variety
    # designation does: it names genotypes, which were released in a year the
    # source does not record. The year is not part of the level identity — the
    # stratum ids and the ledger columns stay on the bare level — it only
    # prefixes the figure filenames, so a directory listing reads in release
    # order and the era this factor is confounded with is visible in it.
    release_years: Mapping[str, int] | None = None

    def level_of(self, context: TrajectoryContext) -> str:
        raw = str(getattr(context, self.attribute, "")).strip()
        if self.level_binner is not None:
            return self.level_binner(raw)
        return raw

    def level_title(self, level: str) -> str:
        if level == FACTOR_LEVEL_UNRECORDED:
            return f"{self.title} unrecorded"
        return f"{self.level_prefix}{level}" if self.level_prefix else level

    def release_year(self, level: str) -> int | None:
        """Release year of one level, or None when the level has no released
        identity — an unreleased breeding line, a pooled or unrecorded level,
        or any level of a factor that is not a variety designation."""

        if not self.release_years:
            return None
        return self.release_years.get(level.strip())


FACTOR_DEFINITIONS: Mapping[str, FactorDefinition] = {
    FACTOR_VARIETY_CODE: FactorDefinition(
        key=FACTOR_VARIETY_CODE,
        attribute="variety_code",
        source_column="VarCode",
        directory="by_variety_code",
        title="variety code",
        level_prefix="VarCode ",
        identity_caveat=(
            "VarCode is the varietal slot in the plot layout, not a genotype: one "
            "code carries many varieties across the years and one variety moves "
            "between codes — do not read these as variety effects"
        ),
    ),
    FACTOR_DESIGN: FactorDefinition(
        key=FACTOR_DESIGN,
        attribute="design",
        source_column="Design",
        directory="by_design",
        title="plot design",
        level_prefix="",
        identity_caveat=(
            "design changed once over the experiment's life, so a design level is "
            "also a calendar era and an applied-N ladder — the agreement scores "
            "below say how completely"
        ),
    ),
    FACTOR_VARIETY: FactorDefinition(
        key=FACTOR_VARIETY,
        attribute="variety",
        source_column="Variety",
        directory="by_variety",
        title="variety designation",
        level_prefix="",
        identity_caveat=(
            "variety is confounded with the calendar period it was grown in; a "
            "difference between varieties here is not a genetic effect"
        ),
        release_years=VARIETY_RELEASE_YEARS,
    ),
    FACTOR_PLANTING_YEAR: FactorDefinition(
        key=FACTOR_PLANTING_YEAR,
        attribute="year",
        source_column="Year",
        directory="by_planting_year",
        title="planting decade",
        level_prefix="",
        identity_caveat=(
            "a decade carries whatever changed with it — the applied-N ladder, "
            "the varieties under test, the plot design, the weather and the "
            "accumulated soil history — so a difference between decades is a "
            "difference between experiments, not a time trend of any one of them"
        ),
        level_binner=decade_band,
        ordinal=True,
    ),
}


@dataclass(frozen=True)
class FactorStratum:
    """One level of an experimental factor inside one season."""

    season: str
    factor: str
    level: str
    trajectory_ids: tuple[str, ...]
    partition: ClusterPartition | None
    not_subclustered_reason: str
    variety_names: tuple[str, ...]
    ladder_shares: Mapping[str, float]
    season_cluster_shares: Mapping[str, float]

    @property
    def distinct_ladder_count(self) -> int:
        return len(self.ladder_shares)

    @property
    def spans_one_ladder(self) -> bool:
        return len(self.ladder_shares) == 1


@dataclass(frozen=True)
class FactorSubstructure:
    """A whole season decomposed by one experimental factor."""

    season: str
    factor: str
    member_count: int
    strata: tuple[FactorStratum, ...]
    minor_levels: tuple[str, ...]
    minor_level_trajectory_ids: tuple[str, ...]
    unrecorded_trajectory_ids: tuple[str, ...]
    minimum_stratum: int
    # Adjusted Rand of the *level* labels against the applied-N ladder labels,
    # and against this season's response-cluster labels. The first says whether
    # the factor is a restatement of the fertilizer design and its era; the
    # second says whether the response partition already encodes the factor.
    ladder_agreement: float
    season_cluster_agreement: float

    @property
    def definition(self) -> FactorDefinition:
        return FACTOR_DEFINITIONS[self.factor]

    @property
    def distinct_level_count(self) -> int:
        return len(self.strata) + len(self.minor_levels)

    @property
    def is_ladder_driven(self) -> bool:
        return (
            math.isfinite(self.ladder_agreement)
            and self.ladder_agreement >= LADDER_DRIVEN_ADJUSTED_RAND
        )

    @property
    def is_cluster_driven(self) -> bool:
        return (
            math.isfinite(self.season_cluster_agreement)
            and self.season_cluster_agreement >= LADDER_DRIVEN_ADJUSTED_RAND
        )

    @property
    def cluster_agreement_is_negligible(self) -> bool:
        """True only when the response partition really carries nothing.

        Distinct from ``not is_cluster_driven``: an ARI of 0.17 is below the
        disclosure threshold and still a visible association, which must not be
        written up as an absence of one.
        """

        return (
            math.isfinite(self.season_cluster_agreement)
            and abs(self.season_cluster_agreement) <= NEGLIGIBLE_ADJUSTED_RAND
        )

    @property
    def ladder_agreement_is_negligible(self) -> bool:
        return (
            math.isfinite(self.ladder_agreement)
            and abs(self.ladder_agreement) <= NEGLIGIBLE_ADJUSTED_RAND
        )


def factor_level_token(level: str) -> str:
    """Path-safe token for one factor level, never empty."""

    token = "".join(
        character if character.isalnum() else "_" for character in level.casefold()
    ).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or FACTOR_LEVEL_UNRECORDED


def factor_level_stem(definition: FactorDefinition, level: str) -> str:
    """Figure stem for one factor level: `<release year>_<token>` where the
    level names a variety with a known release year, and the bare token
    otherwise.

    The year leads rather than trails so that a directory listing of
    `by_variety/` sorts into release order — which for this experiment is also
    calendar order, the confounding the folder's own caveat is about. Levels
    without a release year (breeding-line designations, and every level of
    every other factor) keep the unprefixed token, so the presence of a year is
    itself the statement that the level names a released variety.

    :func:`factor_level_token` is deliberately left alone: it also builds the
    stratum ids carried in the summary JSON and the assignment ledger, and
    those identify a level, not a file.
    """

    token = factor_level_token(level)
    year = definition.release_year(level)
    return f"{year}_{token}" if year is not None else token


def group_by_factor(
    trajectory_ids: Sequence[str],
    contexts: Mapping[str, TrajectoryContext],
    definition: FactorDefinition,
) -> dict[str, tuple[str, ...]]:
    """Split trajectory ids by one recorded factor level, largest level first."""

    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for trajectory_id in trajectory_ids:
        context = contexts.get(trajectory_id)
        level = definition.level_of(context) if context is not None else ""
        grouped[level or FACTOR_LEVEL_UNRECORDED].append(trajectory_id)
    return {
        level: tuple(sorted(ids))
        for level, ids in sorted(
            grouped.items(), key=lambda item: (-len(item[1]), item[0])
        )
    }


def _label_agreement(
    left: Sequence[Any],
    right: Sequence[Any],
) -> float:
    """Adjusted Rand between two label sequences; NaN when either is constant."""

    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    if len(set(left)) < 2 or len(set(right)) < 2:
        return float("nan")
    left_codes = {value: index for index, value in enumerate(sorted(set(left)))}
    right_codes = {value: index for index, value in enumerate(sorted(set(right)))}
    return float(
        adjusted_rand_score(
            [left_codes[value] for value in left],
            [right_codes[value] for value in right],
        )
    )


def cluster_factor_stratum(
    season: str,
    definition: FactorDefinition,
    level: str,
    trajectory_ids: Sequence[str],
    features: Mapping[str, TrajectoryFeatures],
    contexts: Mapping[str, TrajectoryContext],
    *,
    maximum_clusters: int = 6,
) -> ClusterPartition:
    """Cluster inside one factor level on the ladder-invariant basis.

    Deliberately *not* :func:`response_curve_clusters.cluster_stratum`. That
    function clusters the raw observed yield vector, which is admissible only
    when every member shares one applied-N support. A factor level generally
    does not, so the standardized ladder-invariant features are used instead —
    the same basis the season partition itself uses.
    """

    subset = {
        trajectory_id: features[trajectory_id]
        for trajectory_id in trajectory_ids
        if trajectory_id in features
    }
    if len(subset) < 3:
        raise ValueError(
            f"Season {season!r} {definition.key} level {level!r} has too few "
            "eligible trajectories"
        )
    partition = cluster_response_types(
        subset,
        contexts,
        maximum_clusters=maximum_clusters,
    )
    return replace(
        partition,
        partition_id=(
            f"season_{season.casefold()}_{definition.key}_{factor_level_token(level)}"
        ),
        basis=(
            f"{partition.basis}; membership restricted to season {season} and "
            f"{definition.title} {level}, which spans more than one applied-N "
            "ladder, so the raw yield vector is not clusterable here"
        ),
    )


def build_factor_substructure(
    result: SeasonClusteringResult,
    season: str,
    factor: str,
    *,
    minimum_stratum: int = MIN_FACTOR_STRATUM,
    minimum_subclustered: int = MIN_SUBCLUSTERED_FACTOR_STRATUM,
    maximum_clusters: int = 6,
) -> FactorSubstructure:
    """Decompose one whole season by a recorded experimental factor."""

    definition = FACTOR_DEFINITIONS.get(factor)
    if definition is None:
        raise ValueError(f"Unknown experimental factor {factor!r}")
    profile = result.profiles.get(season)
    if profile is None:
        raise ValueError(f"Season {season!r} has no profile to decompose")

    members = profile.trajectory_ids
    grouped = group_by_factor(members, result.contexts, definition)
    unrecorded = grouped.pop(FACTOR_LEVEL_UNRECORDED, ())

    partition = result.partitions.get(season)
    cluster_of: dict[str, int] = {}
    if partition is not None:
        cluster_of = {
            trajectory_id: int(label)
            for trajectory_id, label in zip(
                partition.trajectory_ids, partition.labels, strict=True
            )
        }

    strata: list[FactorStratum] = []
    minor_levels: list[str] = []
    minor_ids: list[str] = []
    for level, ids in grouped.items():
        if len(ids) < minimum_stratum:
            minor_levels.append(level)
            minor_ids.extend(ids)
            continue

        sub_partition: ClusterPartition | None = None
        reason = ""
        if len(ids) < minimum_subclustered:
            reason = _NOT_SUBCLUSTERED_TOO_FEW
        else:
            try:
                sub_partition = cluster_factor_stratum(
                    season,
                    definition,
                    level,
                    ids,
                    result.features,
                    result.contexts,
                    maximum_clusters=maximum_clusters,
                )
            except ValueError:
                reason = _NOT_SUBCLUSTERED_NO_PARTITION

        cluster_counts: collections.Counter[str] = collections.Counter(
            f"cluster {cluster_of[trajectory_id] + 1}"
            for trajectory_id in ids
            if trajectory_id in cluster_of
        )
        assigned = sum(cluster_counts.values())
        strata.append(
            FactorStratum(
                season=season,
                factor=factor,
                level=level,
                trajectory_ids=ids,
                partition=sub_partition,
                not_subclustered_reason=reason,
                variety_names=tuple(
                    sorted(
                        {
                            result.contexts[trajectory_id].variety
                            for trajectory_id in ids
                            if trajectory_id in result.contexts
                            and result.contexts[trajectory_id].variety.strip()
                        }
                    )
                ),
                ladder_shares=cluster_ladder_shares(ids, result.features),
                season_cluster_shares=(
                    {
                        name: count / assigned
                        for name, count in sorted(cluster_counts.items())
                    }
                    if assigned
                    else {}
                ),
            )
        )

    # Agreement is scored over every level-bearing trajectory, minor levels
    # included: dropping the small levels would flatter the factor.
    scored = [
        trajectory_id
        for ids in grouped.values()
        for trajectory_id in ids
        if trajectory_id in result.features
    ]
    level_of = {
        trajectory_id: level
        for level, ids in grouped.items()
        for trajectory_id in ids
    }
    ladder_agreement = _label_agreement(
        [level_of[trajectory_id] for trajectory_id in scored],
        [ladder_text(result.features[trajectory_id].ladder) for trajectory_id in scored],
    )
    co_clustered = [
        trajectory_id for trajectory_id in scored if trajectory_id in cluster_of
    ]
    season_cluster_agreement = _label_agreement(
        [level_of[trajectory_id] for trajectory_id in co_clustered],
        [cluster_of[trajectory_id] for trajectory_id in co_clustered],
    )

    if definition.ordinal:
        # Size order is the right default for a nominal factor and the wrong one
        # for a calendar axis: the reader needs the decades left to right in the
        # order the experiment ran them, on the panel and in the table alike.
        strata.sort(key=lambda stratum: stratum.level)
        minor_levels.sort()

    return FactorSubstructure(
        season=season,
        factor=factor,
        member_count=len(members),
        strata=tuple(strata),
        minor_levels=tuple(minor_levels),
        minor_level_trajectory_ids=tuple(sorted(minor_ids)),
        unrecorded_trajectory_ids=tuple(unrecorded),
        minimum_stratum=minimum_stratum,
        ladder_agreement=ladder_agreement,
        season_cluster_agreement=season_cluster_agreement,
    )


@dataclass(frozen=True)
class AnnualRecord:
    """One planting year of one season, summarized over its own contexts."""

    year: int
    decade: str
    trajectory_count: int
    context_count: int
    # The year's dominant applied-N ladder, and whether it was the only one. A
    # handful of LTCCE years carry a single stray trajectory on an off-ladder;
    # banding on the dominant ladder keeps the era boundaries readable, and the
    # flag keeps the stray from being hidden.
    ladder: str
    ladder_is_mixed: bool
    means: Mapping[str, float]
    standard_errors: Mapping[str, float]


def annual_response_records(
    result: SeasonClusteringResult,
    season: str,
) -> tuple[AnnualRecord, ...]:
    """Per-year context-weighted feature means for one season, oldest first.

    The companion to the decade decomposition, and the reason it is acceptable:
    the strata are decades because a single year cannot support one, but the
    year-by-year detail is not thereby lost — it is carried here, and drawn as
    one trend panel rather than forty overlays.

    Weighting is the same replicate-free context averaging the season profile
    uses, so a year with four replicates does not outweigh a year with one. The
    standard error is the spread of that year's experimental contexts, not
    sampling error of a fitted quantity: nothing is fitted.
    """

    profile = result.profiles.get(season)
    if profile is None:
        return ()

    by_year: dict[int, list[str]] = collections.defaultdict(list)
    for trajectory_id in profile.trajectory_ids:
        context = result.contexts.get(trajectory_id)
        if context is None or context.year <= 0:
            continue
        by_year[int(context.year)].append(trajectory_id)

    records: list[AnnualRecord] = []
    for year in sorted(by_year):
        ids = tuple(sorted(by_year[year]))
        means, errors, context_count = _context_weighted_moments(
            ids, result.features, result.contexts
        )
        ladders = collections.Counter(
            ladder_text(result.features[trajectory_id].ladder)
            for trajectory_id in ids
            if trajectory_id in result.features
        )
        records.append(
            AnnualRecord(
                year=year,
                decade=decade_band(str(year)),
                trajectory_count=len(ids),
                context_count=context_count,
                ladder=(
                    ladders.most_common(1)[0][0] if ladders else "unknown"
                ),
                ladder_is_mixed=len(ladders) > 1,
                means=dict(zip(RESPONSE_TYPE_FEATURES, means, strict=True)),
                standard_errors=dict(
                    zip(RESPONSE_TYPE_FEATURES, errors, strict=True)
                ),
            )
        )
    return tuple(records)


def annual_ladder_eras(
    records: Sequence[AnnualRecord],
) -> tuple[tuple[str, int, int], ...]:
    """Contiguous runs of one applied-N ladder, as (ladder, first, last).

    Drawn as boundaries on the trend panel. In this source the runs are also
    calendar eras — LTCCE replaced one N design with the next rather than
    running them side by side — so a step in the trend at a boundary cannot be
    read as a response to the fertilizer change.
    """

    eras: list[tuple[str, int, int]] = []
    for record in records:
        if eras and eras[-1][0] == record.ladder:
            eras[-1] = (record.ladder, eras[-1][1], record.year)
            continue
        eras.append((record.ladder, record.year, record.year))
    return tuple(eras)


def factor_stratum_members(
    stratum: FactorStratum,
    cluster_id: int,
) -> tuple[str, ...]:
    """Trajectories of one sub-cluster inside a factor level."""

    partition = stratum.partition
    if partition is None:
        return ()
    return season_cluster_members(partition, cluster_id)


__all__ = [
    "FACTOR_DEFINITIONS",
    "FACTOR_DESIGN",
    "FACTOR_LEVEL_UNRECORDED",
    "FACTOR_PLANTING_YEAR",
    "FACTOR_VARIETY",
    "FACTOR_VARIETY_CODE",
    "MIN_FACTOR_STRATUM",
    "MIN_SUBCLUSTERED_FACTOR_STRATUM",
    "NEGLIGIBLE_ADJUSTED_RAND",
    "VARIETY_RELEASE_YEARS",
    "VARIETY_RELEASE_YEAR_SOURCE",
    "AnnualRecord",
    "FactorDefinition",
    "FactorStratum",
    "FactorSubstructure",
    "annual_ladder_eras",
    "annual_response_records",
    "build_factor_substructure",
    "cluster_factor_stratum",
    "decade_band",
    "factor_level_stem",
    "factor_level_token",
    "factor_stratum_members",
    "group_by_factor",
    "LADDER_DOMINATED_CLUSTER_SHARE",
    "LADDER_DRIVEN_ADJUSTED_RAND",
    "MIN_LADDER_SUBSTRATUM",
    "MIN_SEASON_LADDER_SUBSTRATUM",
    "MIN_SEASON_SIZE",
    "MIN_SUBCLUSTERED_SUBSTRATUM",
    "REPLICATE_BOUND_CO_ASSIGNMENT",
    "SEASON_DISPLAY_ORDER",
    "SCOPE_SEASON",
    "SEASON_LABELS",
    "UNCLUSTERED_NO_ADMISSIBLE_PARTITION",
    "UNCLUSTERED_TOO_FEW_TRAJECTORIES",
    "ClusterSubstructure",
    "LadderConcordance",
    "LadderRecord",
    "LadderSubstratum",
    "SeasonClusteringResult",
    "SeasonProfile",
    "build_cluster_substructure",
    "build_ladder_substructure",
    "build_season_clustering",
    "build_season_ladder_substructure",
    "build_season_profile",
    "build_season_substructures",
    "uniform_step",
    "cluster_ladder_shares",
    "cluster_season",
    "cluster_variety_span",
    "cluster_year_range",
    "format_shares",
    "group_by_season",
    "ladder_concordance",
    "ladder_text",
    "ladders_are_disjoint_in_time",
    "mean_feature",
    "order_seasons",
    "season_ladder_records",
    "replicate_binding_caveat",
    "replicate_coherence_text",
    "season_cluster_members",
    "season_feature_matrix",
    "season_label",
]
