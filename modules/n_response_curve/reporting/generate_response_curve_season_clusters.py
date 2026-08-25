#!/usr/bin/env python3
"""Write exploratory LTCCE N-response clusters stratified by cropping season.

Exploratory diagnostic, deliberately outside the release inventory. `ANA-11`
disables the governed `curve_feature_clustering` family, so this generator does
not write into a promoted package and emits no record shaped like an authorized
analysis family.

Season is held fixed and the ladder-invariant response features are clustered
inside DS, EWS, and LWS separately, so the resulting partitions cannot be the
season contrast the pooled `by_trajectory/response_types/` product is exposed
to.  Cluster numbering is local to each season.

Outputs are nested under the shared cluster directory, which
`generate_raw_dataset_overlays.py` preserves across its snapshot replacement.
Re-run this script after any full run or `--refresh-workspace-outputs`.

Usage:
    conda run -n n_response python \\
      modules/n_response_curve/reporting/generate_response_curve_season_clusters.py \\
      --config scriptCONFIG.toml
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import shutil
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.reporting.figure_captions import (  # noqa: E402
    COMPOSITION_TITLE_WIDTH as _COMPOSITION_TITLE_WIDTH,
    TITLE_FONT_SIZE as _TITLE_FONT_SIZE,
    reserve_suptitle as _reserve_suptitle,
    wrap_title_lines as _wrap_title_lines,
)
from n_response_curve.reporting.reference_sheet_layout import (  # noqa: E402
    AXIS_LABEL_FONT_SIZE as _AXIS_LABEL_FONT_SIZE,
    CAPTION_COLUMN_CHARS as _CAPTION_COLUMN_CHARS,
    CAPTION_COLUMN_OFFSET as _CAPTION_COLUMN_OFFSET,
    CAPTION_FONT_SIZE as _CAPTION_FONT_SIZE,
    CAPTION_LINE_SPACING as _CAPTION_LINE_SPACING,
    CAPTION_PARAGRAPH_GAP_LINES as _CAPTION_PARAGRAPH_GAP_LINES,
    DESIGN_GAP_IN as _DESIGN_GAP_IN,
    DESIGN_ROW_IN as _DESIGN_ROW_IN,
    DESIGN_TITLE_FONT_SIZE as _DESIGN_TITLE_FONT_SIZE,
    DESIGN_TITLE_IN as _DESIGN_TITLE_IN,
    ERA_MARKERS as _ERA_MARKERS,
    HEADLINE_FONT_SIZE as _HEADLINE_FONT_SIZE,
    HOST_ROW_IN as _HOST_ROW_IN,
    HOST_TREND_FRACTION as _HOST_TREND_FRACTION,
    INSET_BOTTOM_FRACTION as _INSET_BOTTOM_FRACTION,
    INSET_LEGEND_IN as _INSET_LEGEND_IN,
    INSET_READOUT_FONT_SIZE as _INSET_READOUT_FONT_SIZE,
    INSET_TICK_FONT_SIZE as _INSET_TICK_FONT_SIZE,
    INSET_TITLE_FONT_SIZE as _INSET_TITLE_FONT_SIZE,
    INSET_TOP_FRACTION as _INSET_TOP_FRACTION,
    LEGEND_FONT_SIZE as _LEGEND_FONT_SIZE,
    NARROW_PANEL_IN as _NARROW_PANEL_IN,
    PANEL_GUTTER_IN as _PANEL_GUTTER_IN,
    SHEET_BOTTOM_IN as _SHEET_BOTTOM_IN,
    SHEET_LEFT_IN as _SHEET_LEFT_IN,
    SHEET_RIGHT_IN as _SHEET_RIGHT_IN,
    SHEET_WIDTH_IN as _SHEET_WIDTH_IN,
    TICK_FONT_SIZE as _TICK_FONT_SIZE,
    TREND_GAP_IN as _TREND_GAP_IN,
    TREND_LEGEND_IN as _TREND_LEGEND_IN,
    TREND_ROW_IN as _TREND_ROW_IN,
    TREND_XAXIS_IN as _TREND_XAXIS_IN,
    ZERO_N_TREATMENT_CLASS as _ZERO_N_TREATMENT_CLASS,
    text_inches as _text_inches,
)
from n_response_curve.reporting.figure_axis_frames import (  # noqa: E402
    SharedAxisLimits as _SharedAxisLimits,
    shared_axis_limits as _shared_axis_limits,
)
from n_response_curve.reporting.source_config_spec import load_source_spec  # noqa: E402
from n_response_curve.reporting.figure_captions import (  # noqa: E402
    EXPLORATORY_DIAGNOSTIC_DISCLAIMER as _DISCLAIMER,
)
from n_response_curve.reporting.figure_output import (  # noqa: E402
    save_figure_atomically as _save_figure,
)
from n_response_curve.reporting.response_curve_clusters import (  # noqa: E402
    EXCLUSION_DUPLICATED_N_LEVEL,
    EXCLUSION_INCOMPLETE_LADDER,
    EXCLUSION_NO_ZERO_N_ANCHOR,
    RESPONSE_TYPE_FEATURES,
    ClusterPartition,
    centroid_summary,
    read_ltcce_contexts,
    response_type_label,
    subset_overlay,
)
from n_response_curve.reporting.response_curve_season_clusters import (  # noqa: E402
    FACTOR_DEFINITIONS,
    FACTOR_DESIGN,
    FACTOR_PLANTING_YEAR,
    FACTOR_VARIETY,
    FACTOR_VARIETY_CODE,
    LADDER_DOMINATED_CLUSTER_SHARE,
    LADDER_DRIVEN_ADJUSTED_RAND,
    MIN_FACTOR_STRATUM,
    MIN_LADDER_SUBSTRATUM,
    MIN_SEASON_LADDER_SUBSTRATUM,
    MIN_SEASON_SIZE,
    MIN_SUBCLUSTERED_FACTOR_STRATUM,
    MIN_SUBCLUSTERED_SUBSTRATUM,
    SEASON_LABELS,
    VARIETY_RELEASE_YEAR_SOURCE,
    AnnualRecord,
    ClusterSubstructure,
    FactorDefinition,
    FactorStratum,
    FactorSubstructure,
    LadderSubstratum,
    SeasonClusteringResult,
    build_factor_substructure,
    build_season_clustering,
    build_season_ladder_substructure,
    build_season_substructures,
    annual_ladder_eras,
    annual_response_records,
    cluster_ladder_shares,
    cluster_variety_span,
    cluster_year_range,
    factor_level_stem,
    factor_stratum_members,
    format_shares,
    ladder_text,
    ladders_are_disjoint_in_time,
    season_ladder_records,
    replicate_binding_caveat,
    replicate_coherence_text,
    season_cluster_members,
    season_feature_matrix,
    season_label,
)
from n_response_curve.reporting.source_dataset_overlays import (  # noqa: E402
    SourceDatasetOverlay,
    _adaptive_style,
    create_source_dataset_overlay_figure,
    read_source_dataset_overlay,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scriptCONFIG.toml"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "WF/04_Response_Curves/z_n_response_full/ltcce/clusters/by_season"
)
SOURCE_NAME = "ltcce"


# Repeated on every within-season figure. Each season picks its own k and its own
# centroids, so the integers are not a shared vocabulary.
_LOCAL_LABEL_NOTE = "cluster numbering is local to this season and is not comparable across seasons"


# The separated variants: `<name>_figure.jpeg` carries the plates, its identity
# and the disclaimer, and this one markdown file carries the prose for all of
# them. One file rather than one per figure — the six decade plates share every
# disclosure and differ only in their numbers, which read better as a table than
# as six near-identical blocks.
SEPARATED_NOTES_FILENAME = "separated_figure_notes.md"
_SEPARATED_FIGURE_SUFFIX = "_figure.jpeg"
PLANTING_YEAR_ANNOTATED_DIRNAME = "annotated"
PLANTING_YEAR_FIGURE_ONLY_DIRNAME = "figure_only"
PLANTING_YEAR_REPLICATE_DIRNAME = "by_replicate"
COMPACT_DESIGN_FIGURE_FILENAME = (
    "decades_designs_and_trend_figure_no_description.jpeg"
)
COMPACT_DESIGN_DESCRIPTION_FILENAME = (
    "decades_designs_and_trend_figure_description.md"
)

# The disclosure discipline here costs title lines: a cluster figure carries up
# to nine, including two long ladder/season caveats. On the 10x7 inch canvas
# `create_source_dataset_overlay_figure` builds, that block runs edge to edge and
# squeezes the axes. Every figure this generator writes is therefore enlarged,
# and the title font raised to match the bigger canvas.
_OVERLAY_FIGURE_INCHES = (12.0, 7.5)

# Readable names for the four clustering features, in RESPONSE_TYPE_FEATURES order.
_FEATURE_TITLES = (
    "yield at zero N\n(t/ha)",
    "response above zero N\n(t/ha)",
    "relative N at observed peak\n(fraction of top rate)",
    "saturation index\n(change in t/ha per kg N)",
)

_EXCLUSION_EXPLANATIONS = {
    EXCLUSION_DUPLICATED_N_LEVEL: (
        "more than one observed yield at the same applied-N level",
        "entirely Split-plot 1991-2001 (MV-028 row multiplicity); averaging would "
        "fabricate a mean curve across two treatments",
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


def _load_source_spec(config_path: Path) -> tuple[Path, str]:
    return load_source_spec(
        config_path,
        SOURCE_NAME,
        relative_root=PROJECT_ROOT,
        resolve_path=False,
        missing_sources_message="The configuration must contain a [sources] table",
    )


def _season_token(season: str) -> str:
    token = "".join(
        character if character.isalnum() else "_" for character in season.casefold()
    ).strip("_")
    return token or "season_unrecorded"


# --------------------------------------------------------------------------
# Figure primitives, shared frame
# --------------------------------------------------------------------------










def _titled_figure(
    overlay: SourceDatasetOverlay,
    destination: Path,
    title_lines: Sequence[str],
    limits: _SharedAxisLimits,
) -> None:
    from matplotlib import pyplot as plt

    figure, axes = create_source_dataset_overlay_figure(overlay)
    try:
        figure.set_size_inches(*_OVERLAY_FIGURE_INCHES)
        axes.set_title("\n".join(title_lines), fontsize=_TITLE_FONT_SIZE)
        limits.apply(axes)
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


def _scatter_only_figure(
    overlay: SourceDatasetOverlay,
    destination: Path,
    title_lines: Sequence[str],
    limits: _SharedAxisLimits,
) -> None:
    """Points with no connectors, for trajectories whose ordering is ambiguous."""

    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(
        figsize=_OVERLAY_FIGURE_INCHES, constrained_layout=True
    )
    try:
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
                alpha=0.35,
                label=treatment_class,
                zorder=3,
            )
        axes.set_xlabel("Applied N (kg N/ha)")
        axes.set_ylabel("Grain yield (t/ha)")
        axes.set_title("\n".join(title_lines), fontsize=_TITLE_FONT_SIZE)
        axes.legend(loc="best", fontsize=8)
        limits.apply(axes)
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------
# Season views
# --------------------------------------------------------------------------


def _ladder_caveat(result: SeasonClusteringResult, season: str, cluster_id: int) -> str:
    """State the ladder relationship on the figure, whichever way it comes out."""

    concordance = result.ladder_concordance.get(season)
    if concordance is None:
        return ""
    share = concordance.dominant_ladder_share.get(cluster_id, float("nan"))
    if math.isfinite(share) and share >= LADDER_DOMINATED_CLUSTER_SHARE:
        return (
            f"{100 * share:.0f}% of this cluster sits on one applied-N ladder: "
            "read it as a design stratum, not as an independent response type"
        )
    if concordance.is_ladder_driven:
        return (
            f"cluster/ladder agreement across this season is ARI="
            f"{concordance.adjusted_rand:.2f}: the partition substantially "
            "re-derives the applied-N design"
        )
    return (
        f"cluster/ladder agreement across this season is ARI="
        f"{concordance.adjusted_rand:.2f}: the split is not a relabelling of the "
        "applied-N design"
    )


def _separation_caveat(partition: ClusterPartition) -> str:
    if partition.separation_is_weak:
        return (
            "separation is weak: read these as a descriptive banding of a "
            "continuum, not as discrete response types"
        )
    return f"separation is moderate; the partition mainly tracks {partition.dominant_axis}"


def _diagnostics_line(partition: ClusterPartition) -> str:
    return (
        f"k={partition.cluster_count} chosen by silhouette; "
        f"silhouette={partition.silhouette:.3f}; "
        f"seed stability (ARI)={partition.assignment_stability:.2f}; "
        f"replicates co-assigned={replicate_coherence_text(partition)}"
    )


def _write_season_overview(
    result: SeasonClusteringResult,
    overlay: SourceDatasetOverlay,
    season: str,
    staging: Path,
    limits: _SharedAxisLimits,
) -> str:
    profile = result.profiles[season]
    years = profile.year_range
    year_text = f"{years[0]}-{years[1]}" if years else "years unknown"
    relative = Path(_season_token(season)) / "overview.jpeg"
    _titled_figure(
        subset_overlay(overlay, profile.trajectory_ids),
        staging / relative,
        (
            f"source={SOURCE_NAME} — {season_label(season)}: all cluster-eligible trajectories",
            f"{len(profile.trajectory_ids)} replicate trajectories over "
            f"{profile.experimental_context_count} replicate-free experimental contexts; "
            f"{year_text}; {profile.variety_count} varieties",
            f"applied-N ladders: {format_shares(profile.ladder_shares, limit=4)} "
            f"({profile.distinct_ladder_count} distinct)",
            f"mean zero-N yield={profile.feature_map()['yield_at_zero_n_t_ha']:.2f} t/ha; "
            f"mean response above zero N="
            f"{profile.feature_map()['response_above_zero_n_t_ha']:.2f} t/ha "
            "(context-weighted)",
            _DISCLAIMER,
        ),
        limits,
    )
    return relative.as_posix()


def _write_season_cluster_views(
    result: SeasonClusteringResult,
    overlay: SourceDatasetOverlay,
    season: str,
    staging: Path,
    limits: _SharedAxisLimits,
) -> tuple[list[str], dict[str, str]]:
    partition = result.partitions[season]
    written: list[str] = []
    figure_by_trajectory: dict[str, str] = {}
    directory = Path(_season_token(season))

    for cluster_id in range(partition.cluster_count):
        members = season_cluster_members(partition, cluster_id)
        centroid = centroid_summary(members, result.features)
        years = cluster_year_range(members, result.contexts)
        year_text = f"{years[0]}-{years[1]}" if years else "years unknown"
        shares = cluster_ladder_shares(members, result.features)
        relative = directory / f"cluster_{cluster_id + 1}.jpeg"

        title_lines = [
            f"source={SOURCE_NAME} — {season_label(season)}, cluster "
            f"{cluster_id + 1} of {partition.cluster_count}",
            response_type_label(centroid),
            f"{len(members)} trajectories; {year_text}; "
            f"{cluster_variety_span(members, result.contexts)} varieties; "
            f"applied-N ladders: {format_shares(shares)}",
            f"mean zero-N yield={centroid['yield_at_zero_n_t_ha']:.2f} t/ha; "
            f"mean response={centroid['response_above_zero_n_t_ha']:.2f} t/ha; "
            f"mean peak at {100 * centroid['relative_n_at_peak']:.0f}% of the top rate",
            _diagnostics_line(partition),
            _separation_caveat(partition),
        ]
        ladder_caveat = _ladder_caveat(result, season, cluster_id)
        if ladder_caveat:
            title_lines.append(ladder_caveat)
        title_lines.append(_LOCAL_LABEL_NOTE)
        title_lines.append(_DISCLAIMER)

        _titled_figure(
            subset_overlay(overlay, members),
            staging / relative,
            title_lines,
            limits,
        )
        written.append(relative.as_posix())
        for trajectory_id in members:
            figure_by_trajectory[trajectory_id] = relative.as_posix()
    return written, figure_by_trajectory


def _ladder_token(ladder: Sequence[float]) -> str:
    return "n" + "_".join(f"{level:g}" for level in ladder)


def _substructure_directory(substructure: ClusterSubstructure) -> str:
    """Folder name for one ladder decomposition, per scope."""

    if substructure.is_season_scope:
        return "by_applied_n"
    return f"cluster_{substructure.cluster_id + 1}"




def _write_substratum_panel(
    result: SeasonClusteringResult,
    overlay: SourceDatasetOverlay,
    substructure: ClusterSubstructure,
    destination: Path,
    limits: _SharedAxisLimits,
) -> None:
    """Put one cluster's applied-N ladder sub-strata side by side on a shared frame.

    The individual `sub_<j>_<ladder>.jpeg` figures already share this frame, but
    they are read one at a time. Side by side, the thing that matters is
    immediate: the ladders occupy different stretches of the x-axis *and*
    different calendar periods, so what looks like a design contrast is also an
    era contrast.
    """

    from matplotlib import pyplot as plt

    substrata = substructure.substrata
    if len(substrata) < 2:
        return

    season = substructure.season
    # A single row of four panels is unreadably squat. Wrap instead: at most
    # three columns, and a square-ish 2x2 rather than 3+1 when there are four.
    columns = 2 if len(substrata) == 4 else min(3, len(substrata))
    rows = math.ceil(len(substrata) / columns)
    figure, axes_grid = plt.subplots(
        rows,
        columns,
        figsize=(7.5 * columns, 8.0 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_all = list(np.atleast_1d(axes_grid).ravel())
    axes_list = axes_all[: len(substrata)]
    for spare in axes_all[len(substrata) :]:
        spare.set_visible(False)
    try:
        for index, (substratum, axes) in enumerate(
            zip(substrata, axes_list, strict=True), start=1
        ):
            ids = substratum.trajectory_ids
            _draw_overlay_on_axes(subset_overlay(overlay, ids), axes)
            limits.apply(axes)
            centroid = centroid_summary(ids, result.features)
            years = cluster_year_range(ids, result.contexts)
            year_text = f"{years[0]}-{years[1]}" if years else "years unknown"
            partition = substratum.partition
            sub_text = (
                f"{partition.cluster_count} sub-clusters "
                f"(silhouette {partition.silhouette:.3f})"
                if partition is not None
                else f"not sub-clustered ({substratum.not_subclustered_reason})"
            )
            axes.set_title(
                f"sub_{index}  —  {ladder_text(substratum.ladder)} kg N/ha\n"
                f"{substratum.step_text}\n"
                f"{len(ids)} of {substructure.member_count} trajectories "
                f"({100 * len(ids) / substructure.member_count:.0f}%); {year_text}; "
                f"{cluster_variety_span(ids, result.contexts)} varieties\n"
                f"mean zero-N yield={centroid['yield_at_zero_n_t_ha']:.2f} t/ha; "
                f"mean response={centroid['response_above_zero_n_t_ha']:.2f} t/ha\n"
                f"{sub_text}",
                fontsize=_TITLE_FONT_SIZE,
            )
        # Shared axes hide the inner tick labels, so label only the outer edge.
        for position, axes in enumerate(axes_list):
            if position // columns == rows - 1 or position + columns >= len(axes_list):
                axes.set_xlabel("Applied N (kg N/ha)")
            if position % columns == 0:
                axes.set_ylabel("Grain yield (t/ha)")

        handles, labels = axes_list[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            fontsize=8,
            frameon=False,
        )
        _reserve_suptitle(
            figure,
            f"source={SOURCE_NAME} — {season_label(season)}, "
            f"{substructure.scope_text}: applied-N ladder sub-strata side by side\n"
            "the panels share both axes; each is a design stratum, so the "
            "difference between them is the experimental design, not a result\n"
            "these ladders also occupy different calendar periods — see the Years "
            "line on each panel and the season README\n"
            f"{_DISCLAIMER}",
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


def _write_substructure_views(
    result: SeasonClusteringResult,
    overlay: SourceDatasetOverlay,
    substructure: ClusterSubstructure,
    staging: Path,
    limits: _SharedAxisLimits,
    *,
    write_subcluster_figures: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """Write the applied-N ladder decomposition of one within-season cluster.

    Two nested levels, and the figures say which is which. `sub_<j>_<ladder>` is
    a design stratum — every trajectory in it received the same applied-N
    ladder, so nothing was clustered to produce it. `sub_<j>_<ladder>_cluster_<m>`
    is a genuine k-means on the raw observed yield vector, which only becomes
    admissible once the N support is shared.
    """

    season = substructure.season
    directory = Path(_season_token(season)) / _substructure_directory(substructure)
    written: list[str] = []
    figure_by_trajectory: dict[str, str] = {}

    for index, substratum in enumerate(substructure.substrata, start=1):
        stem = f"sub_{index}_{_ladder_token(substratum.ladder)}"
        ids = substratum.trajectory_ids
        share = len(ids) / substructure.member_count
        years = cluster_year_range(ids, result.contexts)
        year_text = f"{years[0]}-{years[1]}" if years else "years unknown"
        centroid = centroid_summary(ids, result.features)

        relative = directory / f"{stem}.jpeg"
        overview_lines = [
            f"source={SOURCE_NAME} — {season_label(season)}, "
            f"{substructure.scope_text}: applied-N ladder "
            f"{ladder_text(substratum.ladder)} kg N/ha",
            f"{len(ids)} of {substructure.member_count} trajectories in "
            f"{substructure.scope_text} "
            f"({100 * share:.0f}%); {substratum.step_text}; {year_text}; "
            f"{cluster_variety_span(ids, result.contexts)} varieties",
            f"mean zero-N yield={centroid['yield_at_zero_n_t_ha']:.2f} t/ha; "
            f"mean response={centroid['response_above_zero_n_t_ha']:.2f} t/ha",
            "design stratum, not a cluster: exact-ladder match, and trajectories on "
            "other ladders are held out rather than interpolated onto this support",
            _LOCAL_LABEL_NOTE,
            _DISCLAIMER,
        ]
        _titled_figure(
            subset_overlay(overlay, ids), staging / relative, overview_lines, limits
        )
        written.append(relative.as_posix())
        for trajectory_id in ids:
            figure_by_trajectory[trajectory_id] = relative.as_posix()

        partition = substratum.partition
        if partition is None or not write_subcluster_figures:
            # The partition is still computed when the figures are suppressed:
            # its k, silhouette and dominant axis are reported on the stratum
            # comparison panel, in this season's README, and in the summary JSON.
            continue
        for inner_id in range(partition.cluster_count):
            members = season_cluster_members(partition, inner_id)
            inner_centroid = centroid_summary(members, result.features)
            inner_years = cluster_year_range(members, result.contexts)
            inner_year_text = (
                f"{inner_years[0]}-{inner_years[1]}" if inner_years else "years unknown"
            )
            inner_relative = directory / f"{stem}_cluster_{inner_id + 1}.jpeg"
            inner_lines = [
                f"source={SOURCE_NAME} — {season_label(season)}, "
                f"{substructure.scope_text}, ladder "
                f"{ladder_text(substratum.ladder)}: sub-cluster "
                f"{inner_id + 1} of {partition.cluster_count}",
                response_type_label(inner_centroid),
                f"{len(members)} of {len(ids)} trajectories on this ladder; "
                f"{inner_year_text}; "
                f"{cluster_variety_span(members, result.contexts)} varieties",
                f"mean zero-N yield={inner_centroid['yield_at_zero_n_t_ha']:.2f} t/ha; "
                f"mean response={inner_centroid['response_above_zero_n_t_ha']:.2f} t/ha; "
                f"mean peak at {100 * inner_centroid['relative_n_at_peak']:.0f}% "
                "of the top rate",
                f"k={partition.cluster_count} chosen by silhouette; "
                f"silhouette={partition.silhouette:.3f}; "
                f"seed stability (ARI)={partition.assignment_stability:.2f}; "
                f"replicates co-assigned={replicate_coherence_text(partition)}",
                "clustered on the raw observed yield vector, unscaled: admissible "
                "here because every member shares this one applied-N support",
                _separation_caveat(partition),
            ]
            binding = replicate_binding_caveat(partition)
            if binding:
                inner_lines.append(binding)
            inner_lines.append(_LOCAL_LABEL_NOTE)
            inner_lines.append(_DISCLAIMER)
            _titled_figure(
                subset_overlay(overlay, members),
                staging / inner_relative,
                inner_lines,
                limits,
            )
            written.append(inner_relative.as_posix())
            for trajectory_id in members:
                figure_by_trajectory[trajectory_id] = inner_relative.as_posix()

    if len(substructure.substrata) > 1:
        panel_relative = directory / "ladder_comparison.jpeg"
        _write_substratum_panel(
            result, overlay, substructure, staging / panel_relative, limits
        )
        written.append(panel_relative.as_posix())

    minor_ids = substructure.minor_ladder_trajectory_ids
    if minor_ids:
        relative = directory / "sub_other_ladders.jpeg"
        years = cluster_year_range(minor_ids, result.contexts)
        year_text = f"{years[0]}-{years[1]}" if years else "years unknown"
        _titled_figure(
            subset_overlay(overlay, minor_ids),
            staging / relative,
            (
                f"source={SOURCE_NAME} — {season_label(season)}, "
                f"{substructure.scope_text}: minor applied-N ladders "
                "(not decomposed further)",
                f"{len(minor_ids)} trajectories spanning "
                f"{substructure.minor_ladder_count} ladders, each below the "
                f"{MIN_LADDER_SUBSTRATUM}-trajectory sub-stratum minimum; {year_text}",
                "pooling them into one figure is descriptive only; they do not share "
                "an applied-N support and are never clustered together",
                _DISCLAIMER,
            ),
            limits,
        )
        written.append(relative.as_posix())
        for trajectory_id in minor_ids:
            figure_by_trajectory[trajectory_id] = relative.as_posix()
    return written, figure_by_trajectory


def _write_ladder_composition(
    result: SeasonClusteringResult,
    season: str,
    destination: Path,
) -> None:
    """Show, per cluster, how the applied-N ladders are distributed.

    This is the figure that decides whether a within-season partition is worth
    anything. If each bar is one colour, the clusters are the experimental
    design; if the ladders are spread across clusters, they are not.
    """

    from matplotlib import pyplot as plt

    partition = result.partitions[season]
    concordance = result.ladder_concordance[season]
    profile = result.profiles[season]

    # The dominant share is spelled out to one decimal because the in-bar labels
    # are whole percentages: an 89.6% cluster prints as "90%" and would otherwise
    # look like it contradicts the call-out threshold stated in this title.
    rows = [
        (
            f"cluster {cluster_id + 1}\n"
            f"n={len(season_cluster_members(partition, cluster_id))}; "
            f"top ladder {100 * concordance.dominant_ladder_share[cluster_id]:.1f}%",
            cluster_ladder_shares(
                season_cluster_members(partition, cluster_id), result.features
            ),
        )
        for cluster_id in range(partition.cluster_count)
    ]
    rows.append(
        (
            f"whole season\nn={len(profile.trajectory_ids)}",
            dict(profile.ladder_shares),
        )
    )

    ladders = list(profile.ladder_shares)
    colours = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(len(ladders), 1)))

    figure, axes = plt.subplots(
        figsize=(12, 1.05 * len(rows) + 3.4), constrained_layout=True
    )
    try:
        positions = np.arange(len(rows))
        left = np.zeros(len(rows))
        for index, ladder in enumerate(ladders):
            widths = np.asarray([shares.get(ladder, 0.0) for _, shares in rows])
            axes.barh(
                positions,
                widths,
                left=left,
                color=colours[index],
                edgecolor="white",
                linewidth=0.6,
                label=f"{ladder} kg N/ha",
            )
            for position, width, start in zip(positions, widths, left, strict=True):
                if width >= 0.06:
                    axes.text(
                        start + width / 2,
                        position,
                        f"{100 * width:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
            left = left + widths

        axes.axhline(len(rows) - 1.5, color="black", linewidth=1.2)
        axes.set_yticks(positions)
        axes.set_yticklabels([label for label, _ in rows], fontsize=9)
        axes.invert_yaxis()
        axes.set_xlim(0.0, 1.0)
        axes.set_xlabel("share of the cluster's trajectories")
        axes.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(4, max(len(ladders), 1)),
            fontsize=8,
            frameon=False,
        )
        verdict = (
            "the partition substantially re-derives the applied-N design"
            if concordance.is_ladder_driven
            else "the partition is not a relabelling of the applied-N design"
        )
        axes.set_title(
            f"source={SOURCE_NAME} — {season_label(season)}: applied-N ladder "
            "composition of each within-season cluster\n"
            f"cluster/ladder agreement ARI={concordance.adjusted_rand:.3f} "
            f"(disclosure threshold {LADDER_DRIVEN_ADJUSTED_RAND:.2f}): {verdict}\n"
            "the bottom row is the season itself, for reference\n"
            f"{_DISCLAIMER}",
            fontsize=9,
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------
# Experimental-factor views
# --------------------------------------------------------------------------


def _factor_agreement_lines(substructure: FactorSubstructure) -> list[str]:
    """The two disclosure lines every figure in a factor folder carries.

    They answer the two ways a factor decomposition can be uninformative: the
    levels merely restate the applied-N design (and therefore the era), or the
    season's own response partition already encodes them.
    """

    definition = substructure.definition
    ladder = substructure.ladder_agreement
    cluster = substructure.season_cluster_agreement
    lines: list[str] = []

    if not math.isfinite(ladder):
        lines.append(
            f"{definition.title} takes one level in this season, so there is "
            "nothing to score against the applied-N ladder"
        )
    elif substructure.is_ladder_driven:
        lines.append(
            f"{definition.title}/ladder agreement ARI={ladder:.3f} (threshold "
            f"{LADDER_DRIVEN_ADJUSTED_RAND:.2f}): the levels substantially "
            "restate the applied-N design, and with it the calendar era"
        )
    elif substructure.ladder_agreement_is_negligible:
        lines.append(
            f"{definition.title}/ladder agreement ARI={ladder:.3f} (threshold "
            f"{LADDER_DRIVEN_ADJUSTED_RAND:.2f}): the levels are spread evenly "
            "across the applied-N ladders rather than restating them"
        )
    else:
        lines.append(
            f"{definition.title}/ladder agreement ARI={ladder:.3f}: below the "
            f"{LADDER_DRIVEN_ADJUSTED_RAND:.2f} disclosure threshold, but the "
            "levels are not evenly spread across the applied-N ladders either"
        )

    if not math.isfinite(cluster):
        lines.append(
            "no within-season response partition to score these levels against"
        )
    elif substructure.is_cluster_driven:
        lines.append(
            f"{definition.title}/response-cluster agreement ARI={cluster:.3f}: "
            "the season's response partition already largely encodes this factor"
        )
    elif substructure.cluster_agreement_is_negligible:
        lines.append(
            f"{definition.title}/response-cluster agreement ARI={cluster:.3f}: "
            "the season's response partition carries essentially no information "
            "about this factor"
        )
    else:
        lines.append(
            f"{definition.title}/response-cluster agreement ARI={cluster:.3f}: "
            f"below the {LADDER_DRIVEN_ADJUSTED_RAND:.2f} disclosure threshold, "
            "but the levels do lean toward particular response clusters"
        )
    return lines


def _factor_directory(season: str, substructure: FactorSubstructure) -> Path:
    return Path(_season_token(season)) / substructure.definition.directory


def _season_adjective(season: str) -> str:
    """`DS` -> `dry-season`: the season name as a prose modifier.

    Hyphenated because it is only ever used attributively, in front of the
    noun it qualifies.
    """

    described = SEASON_LABELS.get(season.strip().upper())
    return described.replace(" ", "-") if described else "per-season"


def _factor_output_directories(
    season: str,
    substructure: FactorSubstructure,
) -> tuple[Path, Path]:
    """Return the annotated and figure-only destinations for one factor.

    Planting-year output is large enough to need two presentation tiers. Every
    other factor retains its established flat directory because it does not
    emit paired annotated/plates-only families.
    """

    directory = _factor_directory(season, substructure)
    if substructure.definition.key != FACTOR_PLANTING_YEAR:
        return directory, directory
    return (
        directory / PLANTING_YEAR_ANNOTATED_DIRNAME,
        directory / PLANTING_YEAR_FIGURE_ONLY_DIRNAME,
    )






# ...and the same for the overlay canvas, which is narrower and set one point
# larger. `_titled_figure` joins its argument with newlines, so a single
# pre-wrapped string is passed through unchanged.
_OVERLAY_TITLE_WIDTH = 116


def _wrapped_title(lines: Sequence[str]) -> tuple[str]:
    return (_wrap_title_lines(lines, width=_OVERLAY_TITLE_WIDTH),)


def _write_separated_notes(
    destination: Path,
    *,
    season: str,
    definition: FactorDefinition,
    sheets: Sequence[Mapping[str, Any]],
    stratum_rows: Sequence[Mapping[str, Any]],
    shared_prose: Sequence[str],
) -> None:
    """The prose the `_figure.jpeg` plates in this folder carry no longer.

    Rendered from the same lists the annotated figures build their captions
    from, so a plate and its notes cannot disagree; a notes file that can
    contradict the figure it belongs to is worse than none, because the reader
    has no way to tell which is current.

    Two foldings, both because repetition here would be noise rather than
    thoroughness. The trend sheets are the same figure with different encodings,
    so the disclosures every one of them carries are set once and each sheet
    keeps only what is true of it alone. The six decade plates share every
    disclosure and differ only in their numbers, so those become one table.
    """

    lines = [
        f"# {season_label(season)} — {definition.title}: notes for the separated figures",
        "",
        f"{_DISCLAIMER[0].upper()}{_DISCLAIMER[1:]}.",
        "",
        "These notes belong to the `*_figure.jpeg` plates in this folder. Each of",
        "those carries its identity and its own measurements and nothing else;",
        "everything about how to read them, and what not to conclude from them,",
        "is here. Every figure also exists under `../annotated/` in an annotated",
        "form that sets the same text on its own face, for when a plate has to",
        "travel alone.",
        "",
    ]

    if sheets:
        common = [
            entry
            for entry in sheets[0]["disclosures"]
            if all(entry in sheet["disclosures"] for sheet in sheets)
        ]
        lines += [
            f"## The {len(sheets)} trend sheets",
            "",
            "One figure in three encodings, on one shared year axis:",
            "",
        ]
        for sheet in sheets:
            lines.append(
                f"- **`{sheet['figure']}`** — {sheet['variant']}. "
                f"Annotated form: `{sheet['annotated']}`."
            )
        lines += ["", "### True of all three", ""]
        lines += [f"- {entry[0].upper()}{entry[1:]}" for entry in common]
        lines.append("")
        for sheet in sheets:
            extra = [
                entry for entry in sheet["disclosures"] if entry not in common
            ]
            if not extra:
                continue
            lines += [f"### Only `{sheet['figure']}`", ""]
            lines += [f"- {entry[0].upper()}{entry[1:]}" for entry in extra]
            lines.append("")

    if stratum_rows:
        lines += [
            f"## The {len(stratum_rows)} `<level>{_SEPARATED_FIGURE_SUFFIX}` plates",
            "",
            "One recorded stratum each, on the shared frame every figure in this",
            "folder uses. The annotated form of each is `../annotated/<level>.jpeg`.",
            "",
            "| Plate | Level | Trajectories | Years | Varieties | Applied-N ladders "
            "| Mean zero-N (t/ha) | Mean response (t/ha) | Mean peak (% of top rate) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in stratum_rows:
            lines.append(
                f"| `{row['figure']}` | {row['level']} | "
                f"{row['count']} ({row['share']:.0f}%) | {row['years']} | "
                f"{row['varieties']} | {row['ladders']} | {row['zero_n']:.2f} | "
                f"{row['response']:.2f} | {row['peak']:.0f} |"
            )
        lines.append("")
        if shared_prose:
            lines += ["### True of every plate above", ""]
            lines += [
                f"- {entry[0].upper()}{entry[1:]}" for entry in shared_prose
            ]
            lines.append("")

    lines += [
        "## Regenerating",
        "",
        "Written by `generate_response_curve_season_clusters.py` as the",
        "`figure_only/` companion to `../annotated/`; `--no-separate-notes`",
        "suppresses this file and every `*_figure.jpeg` beside it.",
        "",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def _write_compact_design_description(destination: Path, *, season: str) -> None:
    """Write the companion note for the description-free design sheet."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(
            (
                f"# LTCCE decades, designs, and trend — {season_label(season)}",
                "",
                f"Figure: `{COMPACT_DESIGN_FIGURE_FILENAME}`",
                "",
                "This compact plate intentionally omits the on-canvas description. "
                f"Read `{SEPARATED_NOTES_FILENAME}` in this folder for every "
                "interpretive disclosure, and use "
                f"`../{PLANTING_YEAR_ANNOTATED_DIRNAME}/"
                "decades_designs_and_trend.jpeg` when the figure must travel "
                "with its explanation attached.",
                "",
                f"{_DISCLAIMER[0].upper()}{_DISCLAIMER[1:]}.",
                "",
            )
        ),
        encoding="utf-8",
    )


def _separated_footer_line() -> str:
    """The one line a plates-only figure keeps in place of its caption."""

    return (
        f"{_DISCLAIMER}; how to read this figure, and what not to conclude from "
        f"it, is in {SEPARATED_NOTES_FILENAME} beside it"
    )


def _write_factor_stratum_views(
    result: SeasonClusteringResult,
    overlay: SourceDatasetOverlay,
    substructure: FactorSubstructure,
    staging: Path,
    limits: _SharedAxisLimits,
    *,
    write_subcluster_figures: bool = False,
    separate_notes: bool = False,
    write_composition: bool = True,
) -> tuple[list[str], dict[str, str]]:
    """Write one season decomposed by a recorded experimental factor.

    Two nested levels again, and again the figures say which is which.
    `<level>.jpeg` is a recorded stratum — nothing was clustered to produce it,
    the source simply says which trajectories carry that level.
    `<level>_cluster_<m>.jpeg` is a k-means *inside* that stratum, and unlike
    the ladder decomposition it runs on the standardized ladder-invariant
    features, because a factor level generally spans several applied-N supports.
    """

    season = substructure.season
    definition = substructure.definition
    directory = _factor_directory(season, substructure)
    annotated_directory, figure_only_directory = _factor_output_directories(
        season,
        substructure,
    )
    agreement_lines = _factor_agreement_lines(substructure)
    written: list[str] = []
    figure_by_trajectory: dict[str, str] = {}
    # Filled only when `separate_notes` is on: the per-level numbers that differ
    # between the plates, and the prose that does not.
    stratum_rows: list[dict[str, Any]] = []
    shared_prose: list[str] = []

    for stratum in substructure.strata:
        ids = stratum.trajectory_ids
        stem = factor_level_stem(definition, stratum.level)
        share = len(ids) / substructure.member_count
        years = cluster_year_range(ids, result.contexts)
        year_text = f"{years[0]}-{years[1]}" if years else "years unknown"
        centroid = centroid_summary(ids, result.features)

        relative = annotated_directory / f"{stem}.jpeg"
        overview_lines = [
            f"source={SOURCE_NAME} — {season_label(season)}, "
            f"{definition.level_title(stratum.level)}",
            f"{len(ids)} of {substructure.member_count} cluster-eligible "
            f"trajectories in this season ({100 * share:.0f}%); {year_text}; "
            f"{len(stratum.variety_names)} varieties",
            f"applied-N ladders: {format_shares(stratum.ladder_shares, limit=4)}",
            f"mean zero-N yield={centroid['yield_at_zero_n_t_ha']:.2f} t/ha; "
            f"mean response={centroid['response_above_zero_n_t_ha']:.2f} t/ha; "
            f"mean peak at {100 * centroid['relative_n_at_peak']:.0f}% of the top rate",
            f"recorded stratum, not a cluster: every trajectory here carries "
            f"{definition.source_column}={stratum.level!r} in the source",
            definition.identity_caveat,
            *agreement_lines,
            _DISCLAIMER,
        ]
        _titled_figure(
            subset_overlay(overlay, ids),
            staging / relative,
            _wrapped_title(overview_lines),
            limits,
        )
        written.append(relative.as_posix())
        for trajectory_id in ids:
            figure_by_trajectory[trajectory_id] = relative.as_posix()

        if separate_notes:
            # The plate keeps its identity and its own measurements — a stratum
            # figure carries those nowhere else, unlike the trend sheets whose
            # numbers sit in in-panel cards. What moves to the notes is the
            # interpretive half: what the level is, what it is confounded with,
            # and what the agreement statistics say about it.
            figure_relative = (
                figure_only_directory / f"{stem}{_SEPARATED_FIGURE_SUFFIX}"
            )
            _titled_figure(
                subset_overlay(overlay, ids),
                staging / figure_relative,
                _wrapped_title([*overview_lines[:4], _separated_footer_line()]),
                limits,
            )
            written.append(figure_relative.as_posix())
            stratum_rows.append(
                {
                    "level": definition.level_title(stratum.level),
                    "figure": f"{stem}{_SEPARATED_FIGURE_SUFFIX}",
                    "count": len(ids),
                    "share": 100 * share,
                    "years": year_text,
                    "varieties": len(stratum.variety_names),
                    "ladders": format_shares(stratum.ladder_shares, limit=4),
                    "zero_n": centroid["yield_at_zero_n_t_ha"],
                    "response": centroid["response_above_zero_n_t_ha"],
                    "peak": 100 * centroid["relative_n_at_peak"],
                }
            )
            # From index 5: index 4 names this level's own value, so it is not
            # shared, and the notes restate it once for the whole family. The
            # final entry is the disclaimer, which stays on every plate.
            shared_prose = [
                "every plate here is a recorded stratum, not a cluster: its "
                "trajectories are the ones the source's "
                f"{definition.source_column} column assigns to that level, and "
                "nothing was clustered to produce it",
                *overview_lines[5:-1],
            ]

        partition = stratum.partition
        if partition is None or not write_subcluster_figures:
            # As in the ladder decomposition: the partition still runs, so its
            # diagnostics reach the level comparison panel and the summary; only
            # the per-sub-cluster figures are suppressed.
            continue
        for inner_id in range(partition.cluster_count):
            members = factor_stratum_members(stratum, inner_id)
            inner_centroid = centroid_summary(members, result.features)
            inner_years = cluster_year_range(members, result.contexts)
            inner_year_text = (
                f"{inner_years[0]}-{inner_years[1]}" if inner_years else "years unknown"
            )
            inner_relative = (
                annotated_directory / f"{stem}_cluster_{inner_id + 1}.jpeg"
            )
            inner_lines = [
                f"source={SOURCE_NAME} — {season_label(season)}, "
                f"{definition.level_title(stratum.level)}: sub-cluster "
                f"{inner_id + 1} of {partition.cluster_count}",
                response_type_label(inner_centroid),
                f"{len(members)} of {len(ids)} trajectories at this level; "
                f"{inner_year_text}; "
                f"{cluster_variety_span(members, result.contexts)} varieties; "
                f"applied-N ladders: "
                f"{format_shares(cluster_ladder_shares(members, result.features))}",
                f"mean zero-N yield={inner_centroid['yield_at_zero_n_t_ha']:.2f} t/ha; "
                f"mean response={inner_centroid['response_above_zero_n_t_ha']:.2f} t/ha; "
                f"mean peak at {100 * inner_centroid['relative_n_at_peak']:.0f}% "
                "of the top rate",
                _diagnostics_line(partition),
                "clustered on the standardized ladder-invariant features, not on "
                "the raw yield vector: this level spans several applied-N supports"
                if not stratum.spans_one_ladder
                else "clustered on the standardized ladder-invariant features, the "
                "same basis the season partition uses",
                _separation_caveat(partition),
            ]
            binding = replicate_binding_caveat(partition)
            if binding:
                inner_lines.append(binding)
            inner_lines.append(definition.identity_caveat)
            inner_lines.append(
                "sub-cluster numbering is local to this level and to this season"
            )
            inner_lines.append(_DISCLAIMER)
            _titled_figure(
                subset_overlay(overlay, members),
                staging / inner_relative,
                _wrapped_title(inner_lines),
                limits,
            )
            written.append(inner_relative.as_posix())
            for trajectory_id in members:
                figure_by_trajectory[trajectory_id] = inner_relative.as_posix()

    minor_ids = substructure.minor_level_trajectory_ids
    if minor_ids:
        relative = annotated_directory / "minor_levels.jpeg"
        years = cluster_year_range(minor_ids, result.contexts)
        year_text = f"{years[0]}-{years[1]}" if years else "years unknown"
        _titled_figure(
            subset_overlay(overlay, minor_ids),
            staging / relative,
            _wrapped_title((
                f"source={SOURCE_NAME} — {season_label(season)}: minor "
                f"{definition.title} levels (not decomposed further)",
                f"{len(minor_ids)} trajectories spanning "
                f"{len(substructure.minor_levels)} levels, each below the "
                f"{substructure.minimum_stratum}-trajectory stratum minimum; "
                f"{year_text}",
                f"levels pooled here: "
                f"{', '.join(substructure.minor_levels[:12])}"
                + (
                    f", +{len(substructure.minor_levels) - 12} more"
                    if len(substructure.minor_levels) > 12
                    else ""
                ),
                "pooling them into one figure is descriptive only; they are never "
                "clustered together, because the pool is not a level of anything",
                definition.identity_caveat,
                _DISCLAIMER,
            )),
            limits,
        )
        written.append(relative.as_posix())
        for trajectory_id in minor_ids:
            figure_by_trajectory[trajectory_id] = relative.as_posix()

    unrecorded_ids = substructure.unrecorded_trajectory_ids
    if unrecorded_ids:
        relative = annotated_directory / "unrecorded_level.jpeg"
        _titled_figure(
            subset_overlay(overlay, unrecorded_ids),
            staging / relative,
            _wrapped_title((
                f"source={SOURCE_NAME} — {season_label(season)}: "
                f"{definition.title} unrecorded",
                f"{len(unrecorded_ids)} trajectories carry no usable "
                f"{definition.source_column} value and are held out of this "
                "decomposition entirely",
                "held out rather than imputed: a trajectory whose source rows "
                "disagree on the value has no single level to report",
                _DISCLAIMER,
            )),
            limits,
        )
        written.append(relative.as_posix())
        for trajectory_id in unrecorded_ids:
            figure_by_trajectory[trajectory_id] = relative.as_posix()

    if len(substructure.strata) > 1:
        panel_relative = annotated_directory / "level_comparison.jpeg"
        _write_factor_panel(
            result, overlay, substructure, staging / panel_relative, limits
        )
        written.append(panel_relative.as_posix())

    if write_composition:
        composition_relative = (
            annotated_directory / f"{definition.key}_composition.jpeg"
        )
        _write_factor_composition(result, substructure, staging / composition_relative)
        written.append(composition_relative.as_posix())

    # The decade strata are a banding of a continuous axis, so this folder owes
    # the reader the axis itself: one panel carrying every planting year.
    annual_records: tuple[AnnualRecord, ...] = ()
    if definition.key == FACTOR_PLANTING_YEAR:
        trend_relative = annotated_directory / "annual_trend.jpeg"
        annual_records = _write_annual_trend(
            result, season, staging / trend_relative
        )
        if annual_records:
            written.append(trend_relative.as_posix())
        # The two resolutions on one sheet. Kept as a third figure rather than
        # replacing either: the panel grid and the trend are each read on their
        # own, and the combined sheet is read when the question is how they
        # line up.
        combined_relative = annotated_directory / "decades_and_trend.jpeg"
        if _write_decades_and_trend(
            result, overlay, substructure, staging / combined_relative, limits
        ):
            written.append(combined_relative.as_posix())
        # Two variations of the same sheet, each its own file. Neither replaces
        # the plain one: spending colour on the quantity costs the era its own
        # hue, and adding the design row costs a third of the sheet's height, so
        # which sheet answers the question depends on the question.
        matched_relative = (
            annotated_directory / "decades_and_trend_matched_colours.jpeg"
        )
        if _write_decades_and_trend(
            result,
            overlay,
            substructure,
            staging / matched_relative,
            limits,
            matched_colours=True,
        ):
            written.append(matched_relative.as_posix())
        design_substructure = build_factor_substructure(result, season, FACTOR_DESIGN)
        designs_relative = (
            annotated_directory / "decades_designs_and_trend.jpeg"
        )
        if _write_decades_and_trend(
            result,
            overlay,
            substructure,
            staging / designs_relative,
            limits,
            matched_colours=True,
            design_substructure=design_substructure,
        ):
            written.append(designs_relative.as_posix())
        if separate_notes:
            # All three trend sheets, each with its plates-only twin. Doing one
            # and not the others would leave the folder half converted, and the
            # three differ only in which disclosures they carry, which the notes
            # file factors out rather than repeating.
            separated_sheets: list[dict[str, Any]] = []
            for annotated_name, variant, kwargs in (
                (
                    combined_relative.name,
                    "colour spent on the applied-N era",
                    {},
                ),
                (
                    matched_relative.name,
                    "colour spent on the quantity, the era moved to marker shape",
                    {"matched_colours": True},
                ),
                (
                    designs_relative.name,
                    "colour on the quantity, plus a row of plot-design panels "
                    "above the decade insets",
                    {
                        "matched_colours": True,
                        "design_substructure": design_substructure,
                    },
                ),
            ):
                sink: dict[str, Any] = {}
                stem = annotated_name.removesuffix(".jpeg")
                separated_relative = (
                    figure_only_directory / f"{stem}{_SEPARATED_FIGURE_SUFFIX}"
                )
                if not _write_decades_and_trend(
                    result,
                    overlay,
                    substructure,
                    staging / separated_relative,
                    limits,
                    separate_notes=True,
                    notes_sink=sink,
                    **kwargs,
                ):
                    continue
                written.append(separated_relative.as_posix())
                separated_sheets.append(
                    {
                        "figure": separated_relative.name,
                        "annotated": (
                            f"../{PLANTING_YEAR_ANNOTATED_DIRNAME}/"
                            f"{annotated_name}"
                        ),
                        "variant": variant,
                        "headline": sink["headline"],
                        "disclosures": sink["disclosures"],
                    }
                )
            if separated_sheets:
                notes_relative = figure_only_directory / SEPARATED_NOTES_FILENAME
                _write_separated_notes(
                    staging / notes_relative,
                    season=season,
                    definition=definition,
                    sheets=separated_sheets,
                    stratum_rows=stratum_rows,
                    shared_prose=shared_prose,
                )
                written.append(notes_relative.as_posix())

                compact_relative = (
                    figure_only_directory / COMPACT_DESIGN_FIGURE_FILENAME
                )
                if _write_decades_and_trend(
                    result,
                    overlay,
                    substructure,
                    staging / compact_relative,
                    limits,
                    matched_colours=True,
                    design_substructure=design_substructure,
                    omit_description=True,
                ):
                    written.append(compact_relative.as_posix())
                    compact_description_relative = (
                        figure_only_directory
                        / COMPACT_DESIGN_DESCRIPTION_FILENAME
                    )
                    _write_compact_design_description(
                        staging / compact_description_relative,
                        season=season,
                    )
                    written.append(compact_description_relative.as_posix())

    readme_relative = directory / "README.md"
    _write_factor_readme(
        result,
        substructure,
        staging / readme_relative,
        write_subcluster_figures=write_subcluster_figures,
        annual_records=annual_records,
        separate_notes=separate_notes,
        write_composition=write_composition,
    )
    written.append(readme_relative.as_posix())
    return written, figure_by_trajectory


def _write_factor_panel(
    result: SeasonClusteringResult,
    overlay: SourceDatasetOverlay,
    substructure: FactorSubstructure,
    destination: Path,
    limits: _SharedAxisLimits,
) -> None:
    """Put one season's factor levels side by side on a shared frame."""

    from matplotlib import pyplot as plt

    strata = substructure.strata
    if len(strata) < 2:
        return

    definition = substructure.definition
    columns = 2 if len(strata) == 4 else min(3, len(strata))
    rows = math.ceil(len(strata) / columns)
    figure, axes_grid = plt.subplots(
        rows,
        columns,
        figsize=(7.5 * columns, 8.0 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_all = list(np.atleast_1d(axes_grid).ravel())
    axes_list = axes_all[: len(strata)]
    for spare in axes_all[len(strata) :]:
        spare.set_visible(False)
    try:
        for stratum, axes in zip(strata, axes_list, strict=True):
            ids = stratum.trajectory_ids
            _draw_overlay_on_axes(subset_overlay(overlay, ids), axes)
            limits.apply(axes)
            centroid = centroid_summary(ids, result.features)
            years = cluster_year_range(ids, result.contexts)
            year_text = f"{years[0]}-{years[1]}" if years else "years unknown"
            partition = stratum.partition
            sub_text = (
                f"{partition.cluster_count} sub-clusters "
                f"(silhouette {partition.silhouette:.3f})"
                if partition is not None
                else f"not sub-clustered ({stratum.not_subclustered_reason})"
            )
            axes.set_title(
                f"{definition.level_title(stratum.level)}\n"
                f"{len(ids)} of {substructure.member_count} trajectories "
                f"({100 * len(ids) / substructure.member_count:.0f}%); {year_text}; "
                f"{len(stratum.variety_names)} varieties\n"
                f"applied-N ladders: "
                f"{format_shares(stratum.ladder_shares, limit=2)}\n"
                f"mean zero-N yield={centroid['yield_at_zero_n_t_ha']:.2f} t/ha; "
                f"mean response={centroid['response_above_zero_n_t_ha']:.2f} t/ha\n"
                f"{sub_text}",
                fontsize=_TITLE_FONT_SIZE,
            )
        for position, axes in enumerate(axes_list):
            if position // columns == rows - 1 or position + columns >= len(axes_list):
                axes.set_xlabel("Applied N (kg N/ha)")
            if position % columns == 0:
                axes.set_ylabel("Grain yield (t/ha)")

        handles, labels = axes_list[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            fontsize=8,
            frameon=False,
        )
        _reserve_suptitle(
            figure,
            _wrap_title_lines(
                [
                    f"source={SOURCE_NAME} — "
                    f"{season_label(substructure.season)}: {definition.title} "
                    "levels side by side",
                    "the panels share both axes; each is a recorded stratum, so a "
                    "difference between them is whatever the factor is confounded "
                    "with, not a result",
                    definition.identity_caveat,
                    *_factor_agreement_lines(substructure),
                    _DISCLAIMER,
                ],
                # The panel canvas is wider than the composition one.
                width=int(_COMPOSITION_TITLE_WIDTH * 1.4),
            ),
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


def _write_factor_composition(
    result: SeasonClusteringResult,
    substructure: FactorSubstructure,
    destination: Path,
) -> None:
    """Show how each factor level distributes across the season's response clusters.

    The companion to `ladder_composition.jpeg`, and it answers the same
    question in the other direction: if every level's bar has the same colour
    profile as the season reference row at the bottom, the factor carries no
    response signal at all, which is a result worth stating plainly.
    """

    from matplotlib import pyplot as plt

    season = substructure.season
    definition = substructure.definition
    partition = result.partitions.get(season)
    if partition is None or not substructure.strata:
        return

    cluster_names = [
        f"cluster {cluster_id + 1}" for cluster_id in range(partition.cluster_count)
    ]
    rows = [
        (
            f"{definition.level_title(stratum.level)}\n"
            f"n={len(stratum.trajectory_ids)}; "
            f"{len(stratum.variety_names)} varieties",
            dict(stratum.season_cluster_shares),
        )
        for stratum in substructure.strata
    ]
    season_counts = collections.Counter(
        f"cluster {int(label) + 1}" for label in partition.labels
    )
    season_total = sum(season_counts.values())
    rows.append(
        (
            f"whole season\nn={season_total}",
            {
                name: count / season_total
                for name, count in sorted(season_counts.items())
            },
        )
    )

    colours = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(cluster_names), 1)))
    figure, axes = plt.subplots(
        figsize=(12, 1.2 * len(rows) + 3.8), constrained_layout=True
    )
    try:
        positions = np.arange(len(rows))
        left = np.zeros(len(rows))
        for index, name in enumerate(cluster_names):
            widths = np.asarray([shares.get(name, 0.0) for _, shares in rows])
            axes.barh(
                positions,
                widths,
                left=left,
                color=colours[index],
                edgecolor="white",
                linewidth=0.6,
                label=name,
            )
            for position, width, start in zip(positions, widths, left, strict=True):
                if width >= 0.06:
                    axes.text(
                        start + width / 2,
                        position,
                        f"{100 * width:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
            left = left + widths

        axes.axhline(len(rows) - 1.5, color="black", linewidth=1.2)
        axes.set_yticks(positions)
        axes.set_yticklabels([label for label, _ in rows], fontsize=9)
        axes.invert_yaxis()
        axes.set_xlim(0.0, 1.0)
        axes.set_xlabel("share of the level's trajectories")
        axes.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=min(4, max(len(cluster_names), 1)),
            fontsize=8,
            frameon=False,
        )
        axes.set_title(
            _wrap_title_lines(
                [
                    f"source={SOURCE_NAME} — {season_label(season)}: within-season "
                    f"response-cluster composition of each {definition.title} level",
                    *_factor_agreement_lines(substructure),
                    "the bottom row is the season itself: a level whose bar "
                    "matches it carries no response signal",
                    definition.identity_caveat,
                    _DISCLAIMER,
                ]
            ),
            fontsize=9,
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


def _write_annual_trend(
    result: SeasonClusteringResult,
    season: str,
    destination: Path,
) -> tuple[AnnualRecord, ...]:
    """Every planting year of one season as a trend, in two stacked panels.

    The counterpart to the decade strata, and the reason banding to decades is
    not a loss: no individual year clears the stratum minimum, but every year
    is still shown here. Points are context-weighted means over replicate-free
    experimental contexts with the standard error across those contexts; the
    line is drawn only within an applied-N era, never across the boundary
    between two, because the ladders do not overlap in time and a segment
    spanning a boundary would draw a trend through a design change.
    """

    from matplotlib import pyplot as plt

    records = annual_response_records(result, season)
    if len(records) < 2:
        return records

    eras = annual_ladder_eras(records)
    panels = (
        ("yield_at_zero_n_t_ha", "Yield at zero N (t/ha)"),
        ("response_above_zero_n_t_ha", "Response above zero N (t/ha)"),
    )
    figure, axes_grid = plt.subplots(
        len(panels),
        1,
        figsize=(14.0, 8.5),
        sharex=True,
        constrained_layout=True,
    )
    axes_list = list(np.atleast_1d(axes_grid).ravel())
    era_colours = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(eras), 1)))
    try:
        for (feature, label), axes in zip(panels, axes_list, strict=True):
            for index, (ladder, first, last) in enumerate(eras):
                span = [
                    record for record in records if first <= record.year <= last
                ]
                years = [record.year for record in span]
                values = [record.means[feature] for record in span]
                errors = [record.standard_errors[feature] for record in span]
                axes.errorbar(
                    years,
                    values,
                    yerr=errors,
                    marker="o",
                    markersize=4,
                    linewidth=1.4,
                    capsize=2.5,
                    color=era_colours[index],
                    ecolor=era_colours[index],
                    elinewidth=0.8,
                    label=(
                        f"{ladder} kg N/ha ({first}-{last})"
                        if axes is axes_list[0]
                        else None
                    ),
                )
            for _, first, _ in eras[1:]:
                axes.axvline(first - 0.5, color="0.55", linewidth=0.9, ls="--")
            axes.set_ylabel(label)
            axes.grid(alpha=0.25, linewidth=0.6)

        axes_list[-1].set_xlabel("Planting year")
        handles, labels = axes_list[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, max(len(labels), 1)),
            fontsize=8,
            frameon=False,
            title="applied-N era",
            title_fontsize=8,
        )
        mixed = [record.year for record in records if record.ladder_is_mixed]
        _reserve_suptitle(
            figure,
            _wrap_title_lines(
                [
                    f"source={SOURCE_NAME} — {season_label(season)}: observed "
                    "response by planting year",
                    f"{len(records)} years, "
                    f"{records[0].year}-{records[-1].year}; each point is a "
                    "context-weighted mean over that year's replicate-free "
                    "experimental contexts, the bar its standard error across "
                    "those contexts",
                    "the line is broken at every applied-N era boundary "
                    "(dashed): the ladders do not overlap in time, so a change "
                    "across a boundary is a change of experiment, not a "
                    "response to fertilizer",
                    "nothing here is a time trend of a single treatment — "
                    "variety, plot design, applied-N ladder and accumulated "
                    "soil history all move with the year",
                    *(
                        [
                            "years carrying more than one applied-N ladder "
                            f"(banded on the dominant one): "
                            f"{', '.join(str(year) for year in mixed)}"
                        ]
                        if mixed
                        else []
                    ),
                    _DISCLAIMER,
                ],
                width=int(_COMPOSITION_TITLE_WIDTH * 1.25),
            ),
            legend_strip=0.06,
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)
    return records


def _decade_band_spans(
    substructure: FactorSubstructure,
) -> tuple[tuple[str, int, int], ...]:
    """(label, first year, last year) for each decade stratum, oldest first."""

    spans: list[tuple[str, int, int]] = []
    for stratum in substructure.strata:
        try:
            start = int(stratum.level.rstrip("s"))
        except ValueError:
            continue
        spans.append((stratum.level, start, start + 9))
    return tuple(spans)









def _treatment_class_colours(overlay: SourceDatasetOverlay) -> dict[str, Any]:
    """The colour every inset on the sheet gives each treatment class.

    Resolved the same way the insets resolve it rather than restated as a
    literal, so the trend panels below can carry the inset colours instead of a
    second palette that happens to look similar. If the two ever drift apart the
    correspondence the figure claims would be false, and nothing else on the
    sheet would show it.
    """

    from matplotlib import pyplot as plt

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {
        treatment_class: cycle[index % len(cycle)]
        for index, treatment_class in enumerate(overlay.summary.treatment_classes)
    }


_LADDER_READOUT_PREFIX = "applied N "


def _draw_readout_card(
    axes: Any,
    lines: Sequence[str],
    *,
    fontsize: float,
    loc: str = "lower right",
) -> None:
    """The per-panel numbers as a bordered card, ladder line in bold.

    One `TextArea` per line inside a `VPacker`, rather than a single multi-line
    `text` with a `bbox`. The applied-N ladder is the line that differs between
    panels — the decades do not share a ladder, and reading a panel without
    noticing which ladder it ran means comparing yields at rates that were never
    the same — and a matplotlib text artist carries one weight for its whole
    string, so per-line emphasis is only possible if each line is its own
    artist. `VPacker` keeps the right edges aligned and sizes the frame to the
    contents, which the manual bbox did for free.
    """

    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

    # Bold alone tops out here: the bundled DejaVu Sans has no heavier face than
    # Bold, so `heavy`/`black` silently render as the same glyphs. The extra
    # weight therefore comes from size and ink as well — a point and a half up
    # and near-black against the 0.25 grey of the rest of the card.
    children = [
        TextArea(
            line,
            textprops=(
                {
                    "fontsize": fontsize + 1.5,
                    "color": "0.05",
                    "fontweight": "bold",
                }
                if line.startswith(_LADDER_READOUT_PREFIX)
                else {"fontsize": fontsize, "color": "0.25"}
            ),
        )
        for line in lines
    ]
    card = AnchoredOffsetbox(
        loc=loc,
        child=VPacker(children=children, align="right", pad=0.0, sep=3.0),
        pad=0.32,
        borderpad=0.3,
        frameon=True,
        bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
        bbox_transform=axes.transAxes,
    )
    card.patch.set(
        facecolor="white", edgecolor="0.80", linewidth=0.6, alpha=0.92
    )
    card.set_zorder(5)
    axes.add_artist(card)


def _span_panel_readout(
    result: SeasonClusteringResult,
    substructure: FactorSubstructure,
    stratum: FactorStratum,
    *,
    compact: bool,
) -> list[str]:
    """The per-panel numbers, short enough to sit inside the panel.

    Serves both rows of the combined sheet — the decade insets and the design
    panels — because both are strata positioned by the stretch of the year axis
    they occupy. Inside the panel rather than in its title: a panel is as wide
    as its stretch is long, so how much room there is for text is decided by
    how many planting years the stratum spans, not by how much there is to say
    about it.
    """

    ids = stratum.trajectory_ids
    centroid = centroid_summary(ids, result.features)
    years = cluster_year_range(ids, result.contexts)
    share = 100 * len(ids) / substructure.member_count
    if compact:
        year_text = f"{years[0]}-{str(years[1])[2:]}" if years else "years unknown"
        return [
            f"n={len(ids)} ({share:.0f}%)",
            year_text,
            f"0N {centroid['yield_at_zero_n_t_ha']:.2f}",
            f"+N {centroid['response_above_zero_n_t_ha']:.2f}",
        ]
    year_text = f"{years[0]}-{years[1]}" if years else "years unknown"
    return [
        f"{len(ids)} trajectories ({share:.0f}%)",
        year_text,
        f"{_LADDER_READOUT_PREFIX}{format_shares(stratum.ladder_shares, limit=1)}",
        f"mean zero-N {centroid['yield_at_zero_n_t_ha']:.2f} t/ha",
        f"mean response {centroid['response_above_zero_n_t_ha']:.2f} t/ha",
    ]


def _write_decades_and_trend(
    result: SeasonClusteringResult,
    overlay: SourceDatasetOverlay,
    substructure: FactorSubstructure,
    destination: Path,
    limits: _SharedAxisLimits,
    *,
    matched_colours: bool = False,
    design_substructure: FactorSubstructure | None = None,
    separate_notes: bool = False,
    notes_sink: dict[str, Any] | None = None,
    headline_context: str = "",
    omit_description: bool = False,
) -> tuple[AnnualRecord, ...]:
    """Each decade's response curve drawn on the trend segment it came from.

    The two views are one object here rather than two stacked figures: every
    decade's cloud of trajectories against applied N is an inset sitting on its
    own decade of the trend and no other. The year axis is pinned to whole
    calendar decades, so every band is ten years wide and every inset is the
    same size; where a band has no trend under it, the experiment did not run
    in those years, and the caption names them.

    The pairing is not arbitrary. The quantity the host trend tracks is yield
    at zero N, which is the left-hand end of every curve in the inset above it,
    so the inset and the line beneath it are the same measurement at two
    resolutions. The response panel underneath carries the other end.

    Nothing is fitted and nothing is aggregated across the join: the insets are
    the same recorded strata drawn everywhere else in this folder, and the
    trend is the same context-weighted per-year means as
    `annotated/annual_trend.jpeg`.

    Two variations, each written as its own file rather than replacing the
    plain sheet:

    `matched_colours` spends colour on the quantity instead of on the applied-N
    era, so the trend panels carry the colours their own points already have in
    every inset — the zero-N panel in the inset's zero-N colour, the response
    panel in its mineral-N colour. The era it displaces is re-encoded as marker
    shape, on top of the broken lines and dashed boundaries that already
    carried it, so nothing is lost.

    `separate_notes` writes the plates-only variant: the disclosure bullets come
    off the sheet and are handed to `notes_sink` for `_write_separated_notes` to
    render as markdown, so the caption and the notes file are the same list and
    cannot drift. What stays on the figure is what a figure must not travel
    without — the headline that identifies it, the ANA-11 disclaimer, and a
    pointer to the notes file. The in-panel read-out cards are untouched, so the
    plates keep every number they carried.

    `headline_context` adds a short scope label after the season name. It is
    used by additive subset recipes (for example, one recorded replicate) so a
    detached sheet cannot be mistaken for the all-replicate source view.

    `omit_description` is the compact plates-only layout: neither disclosure
    bullets nor the notes pointer is drawn, and their vertical strip is removed
    from the canvas. The headline and all in-panel read-outs remain. This is
    intentionally separate from `separate_notes`, whose notes pointer is part
    of its figure contract.

    `design_substructure` adds a row of panels above the insets, one per
    recorded plot design, positioned on the same year-to-inches mapping. Unlike
    the decades these are *not* equalized: a design panel is as wide as the
    stretch of the experiment that ran under it, because that stretch is the
    thing the panel is about. The row makes the sheet a coarse-to-fine cascade
    down one shared year axis — design, then decade, then year — and it puts
    the confound in view, since a design change here is also a ladder change
    and a break in the calendar.
    """

    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    season = substructure.season
    definition = substructure.definition
    strata = substructure.strata
    records = annual_response_records(result, season)
    spans = _decade_band_spans(substructure)
    if len(strata) < 2 or len(records) < 2 or len(spans) != len(strata):
        return records

    # The axis is pinned to whole calendar decades rather than to the observed
    # years, so every band is exactly ten years wide and every inset is exactly
    # the same size. Clipping each band to the years behind it instead made the
    # first and last insets a fifth the width of the others, too small to read
    # and too small to compare against them. The cost is the unsampled stretch
    # each end band now shows, which is a true statement about the record
    # rather than a gap in the drawing, and is named in the caption.
    x_lo = spans[0][1] - 0.5
    x_hi = spans[-1][2] + 0.5
    visible = [(first - 0.5, last + 0.5) for _, first, last in spans]
    unsampled = [
        f"{low}-{high}" if low < high else f"{low}"
        for low, high in (
            (spans[0][1], records[0].year - 1),
            (records[-1].year + 1, spans[-1][2]),
        )
        if low <= high
    ]
    # Runs of years *inside* the observed span that carry no record at all. The
    # line simply stops across them, and on a sixty-year axis a ten-year blank
    # reads as a plotting failure rather than as an interruption of the
    # experiment, so each one is named on the axis where it happens.
    observed_years = {record.year for record in records}
    interior_gaps: list[tuple[int, int]] = []
    gap_start: int | None = None
    for year in range(records[0].year, records[-1].year + 1):
        if year in observed_years:
            if gap_start is not None:
                interior_gaps.append((gap_start, year - 1))
                gap_start = None
        elif gap_start is None:
            gap_start = year

    # The design row is only drawn where the designs actually partition the
    # year axis. A design whose years interleave with another's cannot be given
    # a panel positioned by its span without the two panels overlapping, and an
    # overlap would assert a correspondence the mapping does not support.
    design_panels: list[tuple[FactorStratum, int, int]] = []
    if design_substructure is not None:
        candidates = []
        for stratum in design_substructure.strata:
            years = cluster_year_range(stratum.trajectory_ids, result.contexts)
            if years is not None:
                candidates.append((stratum, years[0], years[1]))
        candidates.sort(key=lambda item: (item[1], item[2]))
        disjoint = all(
            candidates[index][2] < candidates[index + 1][1]
            for index in range(len(candidates) - 1)
        )
        if len(candidates) >= 2 and disjoint:
            design_panels = candidates

    colours = _treatment_class_colours(overlay)
    zero_n_colour = colours.get(_ZERO_N_TREATMENT_CLASS, "C1")
    mineral_colour = next(
        (
            colour
            for label, colour in colours.items()
            if label != _ZERO_N_TREATMENT_CLASS
        ),
        "C0",
    )

    mixed = [record.year for record in records if record.ladder_is_mixed]
    scope_label = season_label(season)
    if headline_context.strip():
        scope_label = f"{scope_label}, {headline_context.strip()}"
    headline = (
        f"source={SOURCE_NAME} — {scope_label}: each planting "
        "decade's response curves drawn on the stretch of the yield trend "
        "they came from"
        + (
            f", above the {len(design_panels)} plot designs the experiment ran "
            "under"
            if design_panels
            else ""
        )
    )
    disclosures = [
            *(
                [
                    "the design row — the same trajectories grouped by the "
                    f"source's {design_substructure.definition.source_column} "
                    "column instead of by decade, each panel as wide as the "
                    "stretch of the experiment that ran under that design. "
                    "These are not equalized the way the decades are: a design "
                    "ran for as long as it ran, and that length is the point",
                    "the design row is not an independent grouping of the "
                    "decades below it — the designs here do not overlap in "
                    "time, so a design is also a calendar era and, at "
                    f"ARI={design_substructure.ladder_agreement:.3f} against "
                    "the applied-N ladder, very nearly the fertilizer design "
                    "as well. A difference between the two panels is a "
                    "difference between experiments",
                ]
                if design_panels and design_substructure is not None
                else []
            ),
            *(
                [
                    "colour carries the quantity here, not the applied-N era: "
                    "orange is yield at zero N — the orange points in every "
                    "inset, and the panel that tracks their mean by year — and "
                    "blue is the response above zero N, which is computed from "
                    "the blue mineral-N points of the same inset relative to "
                    "its orange ones rather than read off either directly",
                    "the era keeps every encoding it had except colour: the "
                    "line is still broken at each boundary, the boundary is "
                    "still drawn dashed, and the marker shape now names the "
                    "ladder as well",
                ]
                if matched_colours
                else []
            ),
            f"{substructure.member_count} cluster-eligible trajectories over "
            f"{len(records)} planting years ({records[0].year}-"
            f"{records[-1].year}); no single year reaches the "
            f"{substructure.minimum_stratum}-trajectory stratum minimum, which "
            "is why the insets are decades and the host axes is a per-year "
            "trend",
            "the insets — every trajectory of the decade against applied N, "
            "each a recorded stratum that nothing was clustered to produce, "
            "and each on the same frame (0-200 kg N/ha across, 0-11 t/ha up) "
            "at the same size, so any two can be compared directly",
            "each inset sits on its own calendar decade of the axis beneath "
            "it, whole decade to whole decade"
            + (
                f"; the stretch of a band with no trend under it — "
                f"{' and '.join(unsampled)} — is a decade the experiment did "
                "not run in"
                if unsampled
                else ""
            ),
            *(
                [
                    "the stretch marked on both trend panels — "
                    + " and ".join(f"{low}-{high}" for low, high in interior_gaps)
                    + " — is inside the record but has no year this season's "
                    "trend could be computed for, so the line is absent across "
                    "it rather than flat"
                ]
                if interior_gaps
                else []
            ),
            "the two trend panels — yield at zero N by planting year, which is "
            "the left-hand end of every curve in the inset above it, and the "
            "response above zero N, which is the other end. Both are "
            "context-weighted means over each year's replicate-free "
            "experimental contexts, with the standard error across those "
            "contexts, and both are drawn to one shared scale so a change of a "
            "given size has the same slope in each",
            "each line is broken at every applied-N era boundary (dashed): the "
            "ladders do not overlap in time, so a change across a boundary is "
            "a change of experiment, not a response to fertilizer",
            *_factor_agreement_lines(substructure),
            "nothing here is a time trend of a single treatment — variety, "
            "plot design, applied-N ladder and accumulated soil history all "
            "move with the year",
            *(
                [
                    "years carrying more than one applied-N ladder (banded on "
                    f"the dominant one): {', '.join(str(y) for y in mixed)}"
                ]
                if mixed
                else []
            ),
            _DISCLAIMER,
    ]
    if notes_sink is not None:
        notes_sink["headline"] = headline
        notes_sink["disclosures"] = list(disclosures)
    # Set as two left-aligned columns, balanced on wrapped line count, rather
    # than one centred block: ragged-centred text at this length is read line by
    # line instead of scanned, and at a measure short enough to scan a single
    # column uses a third of a 34-inch sheet.
    bullets = (
        []
        if separate_notes or omit_description
        else [
            textwrap.fill(
                entry,
                width=_CAPTION_COLUMN_CHARS,
                initial_indent="—  ",
                subsequent_indent="    ",
                break_long_words=False,
            )
            for entry in disclosures
        ]
    )
    # Set at the axis-label size and wrapped to the caption measure rather than
    # run across the whole 34 inches at caption size: this line carries the
    # disclaimer, and a single thin centred rule of text spanning the full sheet
    # reads as a footnote to be skipped.
    footer_font = _AXIS_LABEL_FONT_SIZE
    footer = textwrap.fill(
        f"{_DISCLAIMER}. Every disclosure this sheet depends on is in "
        f"`{SEPARATED_NOTES_FILENAME}`, beside this figure; read it with the "
        "sheet, not after it",
        width=_CAPTION_COLUMN_CHARS,
        break_long_words=False,
    )
    footer_line_in = _text_inches(footer_font, spacing=_CAPTION_LINE_SPACING)
    counts = [bullet.count("\n") + 1 for bullet in bullets]
    # Balanced on the height each bullet actually occupies — its wrapped lines
    # *plus* the paragraph gap under it — not on line count alone. On line count
    # the column with fewer, longer bullets came out visibly shorter, because
    # the gaps it did not have are real inches.
    weights = [count + _CAPTION_PARAGRAPH_GAP_LINES for count in counts]
    total_weight = sum(weights)
    split = min(
        range(1, max(len(bullets), 2)),
        key=lambda index: abs(2 * sum(weights[:index]) - total_weight),
    )
    caption_columns = (bullets[:split], bullets[split:])
    caption_counts = (counts[:split], counts[split:])
    # Sized from the wrapped text rather than fixed: this caption carries the
    # governance disclosures and grows when a season adds an era, a
    # mixed-ladder year or the design row, and a fixed strip lets it run into
    # the figure. The paragraph gaps are part of the height, not slack.
    caption_line_in = _text_inches(_CAPTION_FONT_SIZE, spacing=_CAPTION_LINE_SPACING)
    paragraph_gap_in = caption_line_in * _CAPTION_PARAGRAPH_GAP_LINES
    caption_block_in = (
        0.0
        if omit_description
        else (
            (footer.count("\n") + 1) * footer_line_in + 0.14
            if separate_notes
            else max(
                sum(column) * caption_line_in
                + max(len(column) - 1, 0) * paragraph_gap_in
                for column in caption_counts
            )
        )
    )
    headline_strip_in = _text_inches(_HEADLINE_FONT_SIZE) + 0.24
    title_strip_in = headline_strip_in + caption_block_in + 0.45

    design_strip_in = (
        _DESIGN_ROW_IN + _DESIGN_GAP_IN + _DESIGN_TITLE_IN if design_panels else 0.0
    )
    height = (
        _SHEET_BOTTOM_IN
        + _TREND_LEGEND_IN
        + _TREND_XAXIS_IN
        + _TREND_ROW_IN
        + _TREND_GAP_IN
        + _HOST_ROW_IN
        + _INSET_LEGEND_IN
        + design_strip_in
        + title_strip_in
    )
    figure = plt.figure(figsize=(_SHEET_WIDTH_IN, height))

    def _box(
        left_in: float, bottom_in: float, width_in: float, height_in: float
    ) -> tuple[float, float, float, float]:
        return (
            left_in / _SHEET_WIDTH_IN,
            bottom_in / height,
            width_in / _SHEET_WIDTH_IN,
            height_in / height,
        )

    plot_left = _SHEET_LEFT_IN
    plot_width = _SHEET_WIDTH_IN - _SHEET_LEFT_IN - _SHEET_RIGHT_IN
    year_span = x_hi - x_lo

    def _year_to_inches(year: float) -> float:
        return plot_left + plot_width * (year - x_lo) / year_span

    response_bottom = _SHEET_BOTTOM_IN + _TREND_LEGEND_IN + _TREND_XAXIS_IN
    host_bottom = response_bottom + _TREND_ROW_IN + _TREND_GAP_IN
    inset_legend_bottom = host_bottom + _HOST_ROW_IN
    design_bottom = inset_legend_bottom + _INSET_LEGEND_IN + _DESIGN_GAP_IN

    eras = annual_ladder_eras(records)
    era_colours = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(eras), 1)))

    try:
        # ---- the two trend axes ---------------------------------------------
        # In matched-colour mode each panel takes the colour its own points
        # already carry in every inset above it, and the era moves to marker
        # shape. Otherwise the panels share one neutral styling and colour is
        # the era, as it was before either variation existed.
        trend_panels = (
            (
                "yield_at_zero_n_t_ha",
                "Yield at zero N (t/ha)",
                host_bottom,
                _HOST_ROW_IN,
                zero_n_colour,
            ),
            (
                "response_above_zero_n_t_ha",
                "Response above zero N (t/ha)",
                response_bottom,
                _TREND_ROW_IN,
                mineral_colour,
            ),
        )
        # One y range over both features, computed before either panel is
        # drawn. The two quantities are the two ends of the same curve and
        # their ranges nearly coincide, so a common scale costs no resolution
        # and buys the only thing that makes them comparable: with the response
        # panel already sized to the host's trend band, one shared range means
        # one shared inches-per-t/ha, so the x axis reads identically in both.
        low = high = None
        for record in records:
            for feature, *_unused in trend_panels:
                value = record.means[feature]
                error = record.standard_errors[feature]
                error = error if error == error else 0.0
                low = value - error if low is None else min(low, value - error)
                high = value + error if high is None else max(high, value + error)
        if low is None or high is None or not high > low:
            low, high = 0.0, 1.0
        pad = 0.06 * (high - low)
        floor, ceiling = low - pad, high + pad
        # Ticks come from the data range, never from the host's stretched
        # limit: locating on the full span puts them where there is no data and
        # leaves the host with one label. Shared, so both panels label the same
        # values at the same heights.
        # `nbins=7` rather than 5: on this season's range 5 snapped to a step of
        # 2 and left the panels labelled 2/4/6, three gridlines for a per-year
        # series whose whole point is the size of the year-to-year move.
        shared_ticks = [
            tick
            for tick in plt.MaxNLocator(
                nbins=7, steps=[1, 2, 2.5, 5, 10]
            ).tick_values(floor, ceiling)
            if floor <= tick <= ceiling
        ]

        trend_axes: list[Any] = []
        for feature, label, bottom_in, row_in, panel_colour in trend_panels:
            is_host = not trend_axes
            axes = figure.add_axes(
                _box(plot_left, bottom_in, plot_width, row_in),
                sharex=trend_axes[0] if trend_axes else None,
            )
            trend_axes.append(axes)

            # The bands go down first, so every data mark sits over them. Left
            # neutral rather than coloured: the decade is the interval its
            # inset occupies, not a series of its own, and the four era colours
            # already in this figure are its colour budget.
            for index, (band_lo, band_hi) in enumerate(visible):
                if index % 2 == 0:
                    axes.axvspan(band_lo, band_hi, color="0.94", zorder=0, linewidth=0)
                axes.axvline(band_hi, color="0.85", linewidth=0.8, zorder=0)

            for index, (ladder, first, last) in enumerate(eras):
                span = [record for record in records if first <= record.year <= last]
                colour = panel_colour if matched_colours else era_colours[index]
                axes.errorbar(
                    [record.year for record in span],
                    [record.means[feature] for record in span],
                    yerr=[record.standard_errors[feature] for record in span],
                    marker=(
                        _ERA_MARKERS[index % len(_ERA_MARKERS)]
                        if matched_colours
                        else "o"
                    ),
                    # Larger where the shape has to be read: with colour spent
                    # on the quantity, a square and a diamond at 5 points are
                    # the same smudge.
                    markersize=9 if matched_colours else 6,
                    linewidth=2.2,
                    capsize=4.0,
                    color=colour,
                    ecolor=colour,
                    elinewidth=1.1,
                    zorder=3,
                    label=(
                        f"{ladder} kg N/ha ({first}-{last})" if is_host else None
                    ),
                )
            for _, first, _ in eras[1:]:
                axes.axvline(first - 0.5, color="0.55", linewidth=0.9, ls="--", zorder=1)
            # Placed in the middle of each panel's own trend band — the host's
            # band is the bottom `_HOST_TREND_FRACTION` of a stretched axes, so
            # its midpoint is not the midpoint of the axes.
            for gap_low, gap_high in interior_gaps:
                if gap_high - gap_low < 2:
                    continue
                axes.text(
                    (gap_low + gap_high) / 2,
                    _HOST_TREND_FRACTION / 2 if is_host else 0.5,
                    f"{gap_low}-{gap_high}\nno record",
                    transform=axes.get_xaxis_transform(),
                    ha="center",
                    va="center",
                    fontsize=_LEGEND_FONT_SIZE,
                    color="0.50",
                    linespacing=1.5,
                    zorder=2,
                )
            # The host's y label is pinned to the middle of its trend band, not
            # to the middle of its axes: the axes is stretched to carry the
            # insets, so a centred label would sit beside the empty strip
            # instead of beside the data it names.
            axes.set_ylabel(
                label,
                y=_HOST_TREND_FRACTION / 2 if is_host else 0.5,
                fontsize=_AXIS_LABEL_FONT_SIZE,
            )
            # y only: the decade boundaries are already drawn as band edges, and
            # a vertical grid on top of them runs the full height of the host,
            # through the strip the insets sit in.
            axes.grid(axis="y", alpha=0.25, linewidth=0.6)
            axes.tick_params(labelsize=_TICK_FONT_SIZE)
            axes.set_yticks(shared_ticks)
            # The host keeps its own scale honest: the data occupies the bottom
            # `_HOST_TREND_FRACTION` and the ticks stop where the data stops, so
            # the empty upper strip the insets sit in is never mistaken for
            # headroom the trend could have reached.
            axes.set_ylim(
                floor,
                floor + (ceiling - floor) / _HOST_TREND_FRACTION
                if is_host
                else ceiling,
            )

        trend_axes[0].set_xlim(x_lo, x_hi)
        trend_axes[0].tick_params(labelbottom=False)
        # Labelled every five years, ticked every year. Six decade labels across
        # 34 inches left a per-year trend that could not be read back to a year
        # at all; the band edges are drawn as `axvline`s and as the shading, so
        # they do not need to be the only labels to stay findable.
        trend_axes[-1].set_xticks(
            list(range(spans[0][1], spans[-1][2] + 1, 5))
        )
        trend_axes[-1].set_xticks(
            list(range(spans[0][1], spans[-1][2] + 1)), minor=True
        )
        trend_axes[-1].set_xlabel("Planting year", fontsize=_AXIS_LABEL_FONT_SIZE)

        # ---- the insets: one decade of response curves on its own years -----
        inset_axes: list[Any] = []
        for index, (stratum, (band_lo, band_hi)) in enumerate(
            zip(strata, visible, strict=True)
        ):
            band_left = _year_to_inches(band_lo)
            band_right = _year_to_inches(band_hi)
            panel_inches = max(band_right - band_left - _PANEL_GUTTER_IN, 0.4)
            axes = figure.add_axes(
                _box(
                    band_left + _PANEL_GUTTER_IN / 2,
                    host_bottom + _INSET_BOTTOM_FRACTION * _HOST_ROW_IN,
                    panel_inches,
                    (_INSET_TOP_FRACTION - _INSET_BOTTOM_FRACTION) * _HOST_ROW_IN,
                ),
                sharex=inset_axes[0] if inset_axes else None,
                sharey=inset_axes[0] if inset_axes else None,
            )
            inset_axes.append(axes)
            # Opaque, so the host's bands and gridlines never read as part of
            # the cloud, and above the host's marks in the draw order.
            axes.set_facecolor("white")
            axes.set_zorder(trend_axes[0].get_zorder() + 1)
            for spine in axes.spines.values():
                spine.set_color("0.55")

            _draw_overlay_on_axes(
                subset_overlay(overlay, stratum.trajectory_ids), axes
            )
            limits.apply(axes)
            axes.grid(alpha=0.25, linewidth=0.6)
            axes.set_xticks([0, 50, 100, 150, 200])
            axes.tick_params(labelsize=_INSET_TICK_FONT_SIZE)
            axes.set_title(
                definition.level_title(stratum.level),
                fontsize=_INSET_TITLE_FONT_SIZE,
                fontweight="bold",
                pad=6,
            )
            # The read-out sits bottom-right, the one corner empty in every
            # inset: the highest applied-N rate carries no low yields, and the
            # axis runs to 200 while the ladders stop at 195.
            _draw_readout_card(
                axes,
                _span_panel_readout(
                    result,
                    substructure,
                    stratum,
                    compact=panel_inches < _NARROW_PANEL_IN,
                ),
                fontsize=_INSET_READOUT_FONT_SIZE,
            )
            if index > 0:
                axes.tick_params(labelleft=False)

        # ---- the design row: one panel per plot design, over its own years --
        # Non-empty only when `design_substructure` was supplied and its levels
        # partition the year axis, so the narrowing below always holds.
        design_definition = (
            design_substructure.definition if design_substructure is not None else None
        )
        for index, (stratum, first_year, last_year) in enumerate(design_panels):
            span_left = _year_to_inches(max(first_year - 0.5, x_lo))
            span_right = _year_to_inches(min(last_year + 0.5, x_hi))
            span_inches = max(span_right - span_left - _PANEL_GUTTER_IN, 0.4)
            axes = figure.add_axes(
                _box(
                    span_left + _PANEL_GUTTER_IN / 2,
                    design_bottom,
                    span_inches,
                    _DESIGN_ROW_IN,
                ),
                sharex=inset_axes[0],
                sharey=inset_axes[0],
            )
            for spine in axes.spines.values():
                spine.set_color("0.55")
            _draw_overlay_on_axes(
                subset_overlay(overlay, stratum.trajectory_ids), axes
            )
            limits.apply(axes)
            axes.grid(alpha=0.25, linewidth=0.6)
            axes.set_xticks([0, 50, 100, 150, 200])
            axes.tick_params(labelsize=_INSET_TICK_FONT_SIZE)
            axes.set_title(
                f"{design_definition.level_title(stratum.level)} "
                f"— {first_year}-{last_year}",
                fontsize=_DESIGN_TITLE_FONT_SIZE,
                fontweight="bold",
                pad=6,
            )
            _draw_readout_card(
                axes,
                _span_panel_readout(
                    result,
                    design_substructure,
                    stratum,
                    compact=span_inches < _NARROW_PANEL_IN,
                ),
                fontsize=_INSET_READOUT_FONT_SIZE,
            )
            if index > 0:
                axes.tick_params(labelleft=False)

        # The stretch between two design panels is a stretch with no record at
        # all, which is the one thing about the row that neither a panel title
        # nor the caption can place on the axis. Named only where there is room
        # to write it without touching either panel.
        for index in range(len(design_panels) - 1):
            gap_lo = design_panels[index][2] + 0.5
            gap_hi = design_panels[index + 1][1] - 0.5
            gap_left = _year_to_inches(gap_lo)
            gap_right = _year_to_inches(gap_hi)
            if gap_hi - gap_lo < 1 or gap_right - gap_left < 1.4:
                continue
            # Drawn as a dashed placard rather than left as bare text on white.
            # An empty stretch between two framed panels reads as a drawing
            # error; a bounded region that is deliberately empty does not.
            placard_left = gap_left + 0.30
            placard_right = gap_right - 0.30
            placard_bottom = design_bottom + _DESIGN_ROW_IN * 0.16
            placard_height = _DESIGN_ROW_IN * 0.68
            figure.add_artist(
                Rectangle(
                    (placard_left / _SHEET_WIDTH_IN, placard_bottom / height),
                    (placard_right - placard_left) / _SHEET_WIDTH_IN,
                    placard_height / height,
                    transform=figure.transFigure,
                    facecolor="0.965",
                    edgecolor="0.78",
                    linewidth=1.0,
                    linestyle=(0, (6, 5)),
                    zorder=0,
                )
            )
            figure.text(
                ((placard_left + placard_right) / 2) / _SHEET_WIDTH_IN,
                (placard_bottom + placard_height / 2) / height,
                f"{int(gap_lo + 0.5)}-{int(gap_hi - 0.5)}\nno "
                f"{season_label(season).split(' ')[0]} record",
                ha="center",
                va="center",
                fontsize=_DESIGN_TITLE_FONT_SIZE,
                color="0.45",
                linespacing=1.5,
            )

        figure.text(
            (plot_left + plot_width / 2) / _SHEET_WIDTH_IN,
            (host_bottom + (_INSET_BOTTOM_FRACTION - 0.05) * _HOST_ROW_IN) / height,
            (
                "the design row above and the insets: applied N (kg N/ha) "
                "across, grain yield (t/ha) up — one shared frame throughout, "
                "ticks on the left-hand panel of each row"
                if design_panels
                else "insets: applied N (kg N/ha) across, grain yield (t/ha) "
                "up — one shared frame, ticks on the left-hand inset"
            ),
            ha="center",
            va="top",
            fontsize=_LEGEND_FONT_SIZE,
            color="0.35",
        )

        handles, labels = inset_axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="center",
            bbox_to_anchor=_box(
                plot_left, inset_legend_bottom, plot_width, _INSET_LEGEND_IN
            ),
            bbox_transform=figure.transFigure,
            ncol=len(labels),
            fontsize=_LEGEND_FONT_SIZE,
            frameon=False,
        )

        trend_handles, trend_labels = trend_axes[0].get_legend_handles_labels()
        figure.legend(
            trend_handles,
            trend_labels,
            loc="center",
            bbox_to_anchor=_box(
                plot_left, _SHEET_BOTTOM_IN, plot_width, _TREND_LEGEND_IN
            ),
            bbox_transform=figure.transFigure,
            ncol=min(4, max(len(trend_labels), 1)),
            fontsize=_LEGEND_FONT_SIZE,
            frameon=False,
            title=(
                "applied-N era — marker shape; colour on this sheet is the "
                "quantity, not the era"
                if matched_colours
                else "applied-N era"
            ),
            title_fontsize=_LEGEND_FONT_SIZE,
        )

        headline_top = 1.0 - 0.20 / height
        figure.text(
            0.5,
            headline_top,
            headline,
            ha="center",
            va="top",
            fontsize=_HEADLINE_FONT_SIZE,
            color="0.10",
        )
        caption_top = headline_top - headline_strip_in / height
        if separate_notes and not omit_description:
            figure.text(
                0.5,
                caption_top,
                footer,
                ha="center",
                va="top",
                fontsize=footer_font,
                color="0.35",
                linespacing=_CAPTION_LINE_SPACING,
            )
        for column_index, (column, column_counts) in enumerate(
            zip(caption_columns, caption_counts, strict=True)
        ):
            cursor = caption_top
            for bullet, bullet_lines in zip(column, column_counts, strict=True):
                figure.text(
                    (plot_left + column_index * plot_width * _CAPTION_COLUMN_OFFSET)
                    / _SHEET_WIDTH_IN,
                    cursor,
                    bullet,
                    ha="left",
                    va="top",
                    fontsize=_CAPTION_FONT_SIZE,
                    color="0.25",
                    linespacing=_CAPTION_LINE_SPACING,
                )
                cursor -= (bullet_lines * caption_line_in + paragraph_gap_in) / height
        # One rule under the whole title block. The caption is nine to twelve
        # governance disclosures and the sheet below it is dense; without a
        # boundary the two run together and the reader cannot tell where the
        # prose stops.
        rule_y = (height - title_strip_in + 0.18) / height
        figure.add_artist(
            Line2D(
                [
                    plot_left / _SHEET_WIDTH_IN,
                    (plot_left + plot_width) / _SHEET_WIDTH_IN,
                ],
                [rule_y, rule_y],
                transform=figure.transFigure,
                color="0.82",
                linewidth=1.0,
            )
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)
    return records


def _factor_level_row(
    result: SeasonClusteringResult,
    substructure: FactorSubstructure,
    stratum: FactorStratum,
) -> str:
    definition = substructure.definition
    ids = stratum.trajectory_ids
    years = cluster_year_range(ids, result.contexts)
    year_text = f"{years[0]}-{years[1]}" if years else "unknown"
    if years and years[0] == years[1]:
        year_text = f"{years[0]}"
    centroid = centroid_summary(ids, result.features)
    partition = stratum.partition
    sub_text = (
        f"{partition.cluster_count} (silhouette {partition.silhouette:.3f})"
        if partition is not None
        else f"none — {stratum.not_subclustered_reason}"
    )
    release_year = definition.release_year(stratum.level)
    released_cell = ""
    if definition.release_years:
        # Dashed rather than dropped: a level with no release year is a
        # breeding-line designation, and the blank says so.
        released_cell = f"| {release_year if release_year is not None else '—'} "
    return (
        f"| `{definition.level_title(stratum.level)}` "
        + released_cell
        + f"| {len(ids)} "
        f"| {100 * len(ids) / substructure.member_count:.1f}% "
        f"| {year_text} "
        f"| {len(stratum.variety_names)} "
        f"| {stratum.distinct_ladder_count} "
        f"| {centroid['yield_at_zero_n_t_ha']:.2f} "
        f"| {centroid['response_above_zero_n_t_ha']:.2f} "
        f"| {sub_text} |\n"
    )


def _write_factor_readme(
    result: SeasonClusteringResult,
    substructure: FactorSubstructure,
    destination: Path,
    *,
    write_subcluster_figures: bool = False,
    annual_records: Sequence[AnnualRecord] = (),
    separate_notes: bool = False,
    write_composition: bool = True,
) -> None:
    """Write the explainer inside one factor folder.

    Its first job is to say what the factor actually is in the source, because
    every level name in the folder is short enough to be mistaken for something
    it is not — `VarCode V3` most of all.
    """

    season = substructure.season
    definition = substructure.definition
    partition = result.partitions.get(season)
    is_planting_year = definition.key == FACTOR_PLANTING_YEAR
    annotated_prefix = (
        f"{PLANTING_YEAR_ANNOTATED_DIRNAME}/" if is_planting_year else ""
    )
    figure_only_prefix = (
        f"{PLANTING_YEAR_FIGURE_ONLY_DIRNAME}/" if is_planting_year else ""
    )

    lines: list[str] = [
        f"# {season_label(season)} — decomposed by {definition.title}\n\n",
        f"{substructure.member_count} cluster-eligible replicate trajectories "
        f"from the LTCCE {season_label(season)}, "
        + (
            f"banded from the source's `{definition.source_column}` column into "
            f"{substructure.distinct_level_count} levels"
            if definition.level_binner is not None
            else f"split by the source's `{definition.source_column}` column "
            f"into {substructure.distinct_level_count} recorded levels"
        )
        + ", and clustered again inside each level that is large enough to "
        "support it.\n\n",
        f"**{definition.identity_caveat[0].upper()}{definition.identity_caveat[1:]}.**"
        "\n\n",
    ]

    if definition.key == "varcode":
        lines.append(
            "## `VarCode` is a plot slot, not a variety\n\n"
            "LTCCE records both a `Variety` designation (`IR8`, `PSBRc52`, …) "
            "and a `VarCode` (`V1`…`V6`). Only the first identifies a genotype. "
            "`VarCode` is the varietal position in the experimental layout, and "
            "the experiment re-used those positions for different entries as the "
            "varieties under test changed over fifty years. The count below is "
            "how many distinct varieties each code carried among this season's "
            "cluster-eligible trajectories — the true multiplicity is higher "
            "still, because the held-out trajectories are not counted here:\n\n"
            "| Code | Distinct varieties it carried | Years | Distinct years |\n"
            "| --- | --- | --- | --- |\n"
        )
        # Ordered by code rather than by size: this table is read as a lookup.
        for stratum in sorted(substructure.strata, key=lambda item: item.level):
            years = cluster_year_range(stratum.trajectory_ids, result.contexts)
            year_text = f"{years[0]}-{years[1]}" if years else "unknown"
            distinct_years = len(
                {
                    result.contexts[trajectory_id].year
                    for trajectory_id in stratum.trajectory_ids
                    if trajectory_id in result.contexts
                    and result.contexts[trajectory_id].year > 0
                }
            )
            lines.append(
                f"| `{stratum.level}` | {len(stratum.variety_names)} | {year_text} "
                f"| {distinct_years} |\n"
            )
        lines.append(
            "\nA figure in this folder titled `VarCode V3` is therefore a figure "
            "about a *position in the field layout*, pooling every variety that "
            "ever occupied it. For the genotype reading there are two other "
            "folders, and they answer different questions:\n\n"
            "- `../by_variety/` — this same season split by the `Variety` "
            "designation, one figure per variety large enough to draw. Read it "
            "when you want this season's varieties side by side.\n"
            "- `../../../by_variety/` — a separate product that clusters "
            "*varieties themselves* by their pooled response profile, across "
            "seasons. Read it when you want varieties grouped rather than "
            "listed.\n\n"
        )

    if definition.key == FACTOR_PLANTING_YEAR and annual_records:
        eras = annual_ladder_eras(annual_records)
        lines.append(
            "## Why decades, when the source records a year\n\n"
            f"This season carries {len(annual_records)} planting years "
            f"({annual_records[0].year}-{annual_records[-1].year}) and about "
            f"{round(sum(record.trajectory_count for record in annual_records) / len(annual_records))} "
            "cluster-eligible trajectories in each. No single year reaches the "
            f"{substructure.minimum_stratum}-trajectory stratum minimum used "
            "everywhere in this product, so a year-by-year decomposition would "
            "pool the entire season into `minor_levels.jpeg` and show nothing. "
            "The strata are therefore calendar decades.\n\n"
            "Nothing about the individual years is lost by that: "
            f"`{annotated_prefix}annual_trend.jpeg` carries every one of them, as "
            "context-weighted means with the standard error across each year's "
            "experimental contexts.\n\n"
            "### Applied-N eras behind the decades\n\n"
            "| Applied N (kg N/ha) | Years | Planting years |\n"
            "| --- | --- | --- |\n"
        )
        for ladder, first, last in eras:
            span = f"{first}-{last}" if first != last else f"{first}"
            covered = sum(
                1 for record in annual_records if first <= record.year <= last
            )
            lines.append(f"| `{ladder}` | {span} | {covered} |\n")
        lines.append(
            "\nThe eras do not overlap: LTCCE replaced one N design with the "
            "next rather than running them side by side. A decade is therefore "
            "partly a fertilizer design as well as a period, which is what the "
            "ladder agreement score below measures — read it before comparing "
            "two decades.\n\n"
        )

    lines.append("## Levels\n\n")
    # The variety designation carries a release year the source never records,
    # so its table gains a column and its figures gain a filename prefix. Every
    # other factor keeps the nine-column table.
    released_header = "| Released " if definition.release_years else ""
    released_rule = "| --- " if definition.release_years else ""
    released_blank = "| — " if definition.release_years else ""
    lines.append(
        f"| Level {released_header}| Trajectories | Share | Years | Varieties "
        "| Applied-N ladders "
        "| Mean zero-N yield (t/ha) | Mean response (t/ha) | Sub-clusters |\n"
        f"| --- {released_rule}| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    for stratum in substructure.strata:
        lines.append(_factor_level_row(result, substructure, stratum))
    if substructure.minor_levels:
        lines.append(
            f"| *{len(substructure.minor_levels)} minor levels pooled* "
            f"{released_blank}"
            f"| {len(substructure.minor_level_trajectory_ids)} "
            f"| {100 * len(substructure.minor_level_trajectory_ids) / substructure.member_count:.1f}% "
            "| — | — | — | — | — | none — below the "
            f"{substructure.minimum_stratum}-trajectory minimum |\n"
        )
    if substructure.unrecorded_trajectory_ids:
        lines.append(
            f"| *{definition.source_column} unrecorded* "
            f"{released_blank}"
            f"| {len(substructure.unrecorded_trajectory_ids)} "
            f"| {100 * len(substructure.unrecorded_trajectory_ids) / substructure.member_count:.1f}% "
            "| — | — | — | — | — | held out of this decomposition |\n"
        )
    lines.append(
        f"\nA level is drawn as its own figure at {substructure.minimum_stratum} "
        f"trajectories or more, and sub-clustered at "
        f"{MIN_SUBCLUSTERED_FACTOR_STRATUM} or more. Between the two it is shown "
        "but not partitioned. "
        + (
            "Below the first threshold a level is pooled into "
            f"`{annotated_prefix}minor_levels.jpeg`.\n\n"
            if substructure.minor_levels
            else "No level here falls below the first threshold, so every "
            "trajectory in this season is in one of the figures above.\n\n"
        )
    )

    if definition.release_years:
        released = [
            stratum
            for stratum in substructure.strata
            if definition.release_year(stratum.level) is not None
        ]
        unreleased = [
            stratum
            for stratum in substructure.strata
            if definition.release_year(stratum.level) is None
        ]
        lines.append(
            "### Where the release years come from\n\n"
            "The source records no release year, so the `Released` column and "
            "the `<year>_<level>.jpeg` filenames are joined in from the "
            f"{VARIETY_RELEASE_YEAR_SOURCE}. "
            f"{len(released)} of the {len(substructure.strata)} levels drawn "
            "here resolve to a released variety; "
            + (
                "the rest are breeding-line designations, which were not "
                "released varieties while they were grown under those names, "
                "and carry no year — "
                + ", ".join(f"`{stratum.level}`" for stratum in unreleased)
                + ".\n\n"
                if unreleased
                else "every one of them does.\n\n"
            )
            + "The join is on the recorded designation only — by IR number, by "
            "PSB/NSIC `Rc` number, or by the line designation the registry "
            "records against a released name. Every joined year is at or "
            "before the level's own first planting year above, which is the "
            "only check the trial record can make on it. **A release year is "
            "context, not a covariate**: nothing in this product's clustering, "
            "features or agreement scores uses it, and the levels are ordered "
            "by size here exactly as they are in every other factor folder.\n\n"
        )

    lines.append("## Is this decomposition telling you anything?\n\n")
    lines.append(
        "Two agreement scores, both adjusted Rand indices against the "
        f"{LADDER_DRIVEN_ADJUSTED_RAND:.2f} disclosure threshold this product "
        "uses throughout:\n\n"
    )
    if substructure.is_ladder_driven:
        ladder_verdict = (
            "The levels substantially restate the applied-N design. Since the "
            "ladders in this season do not overlap in time, they restate the "
            "calendar era with it, and no comparison between levels can be "
            "separated from either."
        )
    elif substructure.ladder_agreement_is_negligible:
        ladder_verdict = (
            "The levels are spread evenly across the applied-N ladders rather "
            "than restating them, so a comparison between levels is not "
            "automatically a comparison between fertilizer designs or eras."
        )
    else:
        ladder_verdict = (
            "Below the disclosure threshold, but the levels are not evenly "
            "spread across the ladders either. Check the per-level ladder shares "
            "in the table above before comparing two levels."
        )
    lines.append(
        f"- **{definition.title} vs applied-N ladder: "
        f"ARI={substructure.ladder_agreement:.3f}.** {ladder_verdict}\n"
    )

    if substructure.is_cluster_driven:
        cluster_verdict = (
            "The within-season response partition already largely encodes this "
            "factor, so the two views are not independent readings of the data."
        )
    elif substructure.cluster_agreement_is_negligible:
        cluster_verdict = (
            "The within-season response partition carries essentially no "
            "information about this factor: knowing a trajectory's level tells "
            "you almost nothing about which response cluster it landed in, and "
            "vice versa."
        )
    else:
        cluster_verdict = (
            "Below the disclosure threshold, so this is not reported as a "
            "relabelling — but the levels do lean toward particular response "
            "clusters, and the composition figure shows by how much."
        )
    lines.append(
        f"- **{definition.title} vs the season's response clusters: "
        f"ARI={substructure.season_cluster_agreement:.3f}.** {cluster_verdict}\n\n"
    )

    # Both of these point the reader at the composition figure by name, so they
    # are gated on it being written; a README that names an absent file is worse
    # than one that says less.
    if partition is not None and substructure.cluster_agreement_is_negligible:
        lines.append(
            "That second score is the substantive finding of this folder, and it "
            "is a negative one."
            + (
                " Read `"
                f"{annotated_prefix}{definition.key}_composition.jpeg`: every "
                "level's bar has "
                "close to the same colour profile as the whole-season row "
                "beneath it."
                if write_composition
                else " Every level's response-cluster mix is close to the "
                "whole season's; the per-level shares are in the summary JSON."
            )
            + "\n\n"
        )
    elif partition is not None and write_composition:
        lines.append(
            f"Read `{annotated_prefix}{definition.key}_composition.jpeg` for the "
            "shape of that "
            "second number: it puts each level's response-cluster mix directly "
            "above the whole-season mix, so the size of the lean is visible "
            "rather than only its index.\n\n"
        )

    lines.append("## What is in this folder\n\n")
    if is_planting_year:
        lines.append(
            f"- `{PLANTING_YEAR_ANNOTATED_DIRNAME}/` — self-explaining figures "
            "whose interpretation travels on the canvas.\n"
            f"- `{PLANTING_YEAR_FIGURE_ONLY_DIRNAME}/` — document-ready plates "
            f"and `{PLANTING_YEAR_FIGURE_ONLY_DIRNAME}/"
            f"{SEPARATED_NOTES_FILENAME}`, the shared prose for those plates.\n"
        )
        lines.append(
            f"- `{PLANTING_YEAR_FIGURE_ONLY_DIRNAME}/"
            f"{PLANTING_YEAR_REPLICATE_DIRNAME}/` — optional additive "
            f"{_season_adjective(season)} sheets split by the source's "
            "recorded `Rep` value.\n"
        )
    level_stem = "<year>_<level>" if definition.release_years else "<level>"
    lines.append(
        f"- `{annotated_prefix}{level_stem}.jpeg` — every cluster-eligible "
        "trajectory "
        + (
            f"whose `{definition.source_column}` falls in that band"
            if definition.level_binner is not None
            else f"carrying that `{definition.source_column}` value"
        )
        + ", on the frame shared by the whole product. A recorded stratum: "
        "nothing was clustered to produce it.\n"
        + (
            "  `<year>` is the release year in the table above, so the listing "
            "reads in release order; a level with no release year — a breeding "
            "line — is filed under its bare designation.\n"
            if definition.release_years
            else ""
        )
    )
    if write_subcluster_figures:
        lines.append(
            f"- `{annotated_prefix}{level_stem}_cluster_<m>.jpeg` — the k-means "
            "sub-partition "
            "inside that level, where the level is large enough to support "
            "one.\n"
        )
    if write_composition:
        lines.append(
            f"- `{annotated_prefix}{definition.key}_composition.jpeg` — how each "
            "level distributes "
            "across the season's own response clusters.\n"
        )
    lines.append(
        f"- `{annotated_prefix}level_comparison.jpeg` — the levels side by side "
        "on one shared "
        "frame.\n"
    )
    if definition.key == FACTOR_PLANTING_YEAR and annual_records:
        lines.append(
            f"- `{annotated_prefix}annual_trend.jpeg` — every planting year "
            "individually: zero-N "
            "yield and response above zero N against the year, with the line "
            "broken at each applied-N era boundary.\n"
            f"- `{annotated_prefix}decades_and_trend.jpeg` — both of the above "
            "on one sheet, with "
            "each decade's response-curve panel set over its own decade of the "
            "year axis, so a panel and its shaded band are the same interval "
            "of calendar time read two ways. The axis runs whole decade to "
            "whole decade, so every panel is the same size and any two can be "
            "compared directly; a band with no trend under it is a stretch the "
            "experiment did not run in, and a run of years inside the record "
            "that the trend could not be computed for is named on the axis "
            "where the line stops. Both trend panels share one y range "
            "and one inches-per-t/ha, so a change of a given size has the same "
            "slope in each.\n"
            f"- `{annotated_prefix}decades_and_trend_matched_colours.jpeg` — the "
            "same sheet with "
            "colour spent on the quantity instead of the applied-N era, so "
            "each trend panel carries the colour its own points already have "
            "in every inset: orange for yield at zero N, blue for the response "
            "above it. The era keeps its broken lines and dashed boundaries "
            "and gains a marker shape, so nothing that colour used to say is "
            "lost.\n"
            f"- `{annotated_prefix}decades_designs_and_trend.jpeg` — the "
            "matched-colour sheet with "
            "a row of plot-design panels added above the decade insets, on the "
            "same year axis. A design panel is as wide as the stretch the "
            "design ran for rather than equalized, and the gap between panels "
            "is the gap in the experiment. Read it for the confound: the "
            "designs here do not overlap in time, so a difference between them "
            "is also a difference of ladder and of era.\n"
        )
        if separate_notes:
            lines.append(
                f"- `{figure_only_prefix}<name>{_SEPARATED_FIGURE_SUFFIX}` and "
                f"`{figure_only_prefix}{SEPARATED_NOTES_FILENAME}` — the same "
                "figures with the prose "
                "separated from the plates. A `_figure.jpeg` keeps its identity, "
                "its own read-outs and the ANA-11 disclaimer, and nothing else; "
                "the notes file carries every disclosure for all of them, drawn "
                "from the same text the annotated figures set on their own "
                "faces, so the two cannot disagree. Use the annotated figure "
                "when a plate has to travel alone, and this pair when it goes "
                "into a document that carries its own prose.\n"
                f"- `{figure_only_prefix}{COMPACT_DESIGN_FIGURE_FILENAME}` and "
                f"`{figure_only_prefix}{COMPACT_DESIGN_DESCRIPTION_FILENAME}` — "
                "the compact design/decade/trend plate with no on-canvas "
                "description, plus its companion note.\n"
            )
    if not write_subcluster_figures:
        lines.append(
            "\nThe per-sub-cluster figures are not drawn. The sub-partitions "
            "themselves still run: their k, silhouette and dominant axis are in "
            "the `Sub-clusters` column above, on the "
            f"`{annotated_prefix}level_comparison.jpeg` "
            "panel titles, and in the summary JSON, and every trajectory's "
            "sub-cluster id is in `../../season_cluster_assignments.csv`. Pass "
            "`--write-subcluster-figures` to the generator to draw them.\n"
        )
    if substructure.minor_levels:
        lines.append(
            f"- `{annotated_prefix}minor_levels.jpeg` — the undersized levels, "
            "pooled.\n"
        )
    if substructure.unrecorded_trajectory_ids:
        lines.append(
            f"- `{annotated_prefix}unrecorded_level.jpeg` — trajectories with no "
            "usable "
            f"`{definition.source_column}` value.\n"
        )
    lines.append("\n## How the sub-clusters were built\n\n")
    lines.append(
        "On the standardized ladder-invariant features (zero-N yield, response "
        "above zero N, relative N at the observed peak, saturation index), with "
        "k chosen by maximum silhouette — the same basis the season partition "
        "uses. This differs from the applied-N decomposition in "
        "`../by_applied_n/`, which clusters the *raw* observed yield vector; "
        "that is admissible only inside one exact ladder, and a "
        f"{definition.title} level generally spans several.\n\n"
    )
    lines.append(
        "Sub-cluster numbers are local to their level and to this season. "
        f"`{definition.level_title(substructure.strata[0].level)}` sub-cluster 1 "
        "has no relationship to any other sub-cluster 1 in this product.\n\n"
        if substructure.strata
        else ""
    )
    lines.append(
        "## Interpretation boundary\n\n"
        "Exploratory diagnostic, outside the governed analysis inventory "
        "(ANA-11). No curve is fitted anywhere in this folder; every centroid is "
        "a mean of observed yields at observed N rates. Full method and caveats "
        "are in `../../season_clustering_summary.json`.\n"
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(lines), encoding="utf-8")


def _write_held_out_views(
    result: SeasonClusteringResult,
    overlay: SourceDatasetOverlay,
    season: str,
    staging: Path,
    limits: _SharedAxisLimits,
) -> tuple[list[str], dict[str, str]]:
    profile = result.profiles[season]
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for trajectory_id in profile.excluded_trajectory_ids:
        grouped[result.excluded[trajectory_id]].append(trajectory_id)

    written: list[str] = []
    figure_by_trajectory: dict[str, str] = {}
    directory = Path(_season_token(season))
    for reason, ids in sorted(grouped.items()):
        headline, rationale, note = _EXCLUSION_EXPLANATIONS.get(reason, (reason, "", ""))
        years = cluster_year_range(ids, result.contexts)
        year_text = f"{years[0]}-{years[1]}" if years else "years unknown"
        relative = directory / f"held_out_{reason}.jpeg"
        title_lines = [
            f"source={SOURCE_NAME} — {season_label(season)}, held out of clustering: {headline}",
            f"{len(ids)} trajectories; {year_text}; "
            f"{cluster_variety_span(ids, result.contexts)} varieties",
        ]
        if rationale:
            title_lines.append(rationale)
        if note:
            title_lines.append(note)
        title_lines.append(_DISCLAIMER)

        subset = subset_overlay(overlay, ids)
        if reason == EXCLUSION_DUPLICATED_N_LEVEL:
            _scatter_only_figure(subset, staging / relative, title_lines, limits)
        else:
            _titled_figure(subset, staging / relative, title_lines, limits)
        written.append(relative.as_posix())
        for trajectory_id in ids:
            figure_by_trajectory[trajectory_id] = relative.as_posix()
    return written, figure_by_trajectory


def _draw_overlay_on_axes(overlay: SourceDatasetOverlay, axes: Any) -> None:
    """Render one overlay into an existing axes.

    `create_source_dataset_overlay_figure` owns its own figure, so it cannot be
    used for a shared-axis grid. The marker/line styling is taken from the same
    `_adaptive_style` the single-season figures use, and is computed from each
    season's own density, so a panel here looks like that season's own
    `overview.jpeg` rather than being restyled for the grid.
    """

    marker_size, marker_alpha, line_alpha = _adaptive_style(
        overlay.summary.finite_observation_count,
        overlay.summary.trajectory_count,
    )
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
            s=marker_size,
            alpha=marker_alpha,
            label=treatment_class,
            zorder=3,
        )
    line_label = "within-trajectory connecting lines (visual aid; not a fit)"
    for index, trajectory in enumerate(overlay.trajectories):
        axes.plot(
            [observation.n_rate_kg_ha for observation in trajectory.observations],
            [observation.yield_t_ha for observation in trajectory.observations],
            label=line_label if index == 0 else "_nolegend_",
            color="grey",
            linewidth=1,
            alpha=line_alpha,
            zorder=1,
        )


def _write_season_overview_panel(
    result: SeasonClusteringResult,
    overlay: SourceDatasetOverlay,
    destination: Path,
    limits: _SharedAxisLimits,
) -> None:
    """Put every season's overview side by side on one shared frame.

    The per-season `overview.jpeg` files already share this frame, but they are
    in separate folders and are read one at a time. Side by side on shared axes
    is what makes the size of the dry-season response — roughly three times the
    wet-season response — visible as a difference in the cloud rather than as a
    number in two different title blocks.
    """

    from matplotlib import pyplot as plt

    seasons = [season for season in result.seasons if result.profiles[season].trajectory_ids]
    figure, axes_row = plt.subplots(
        1,
        len(seasons),
        figsize=(7.5 * len(seasons), 8.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_list = list(np.atleast_1d(axes_row))
    try:
        for season, axes in zip(seasons, axes_list, strict=True):
            profile = result.profiles[season]
            partition = result.partitions.get(season)
            _draw_overlay_on_axes(
                subset_overlay(overlay, profile.trajectory_ids), axes
            )
            limits.apply(axes)
            years = profile.year_range
            year_text = f"{years[0]}-{years[1]}" if years else "years unknown"
            features = profile.feature_map()
            cluster_text = (
                f"{partition.cluster_count} within-season clusters "
                f"(silhouette {partition.silhouette:.3f})"
                if partition is not None
                else f"not clustered ({result.unclustered_seasons.get(season, '')})"
            )
            axes.set_title(
                f"{season_label(season)}\n"
                f"{len(profile.trajectory_ids)} eligible trajectories; "
                f"{len(profile.excluded_trajectory_ids)} held out; {year_text}\n"
                f"mean zero-N yield={features['yield_at_zero_n_t_ha']:.2f} t/ha; "
                f"mean response={features['response_above_zero_n_t_ha']:.2f} t/ha\n"
                f"applied-N ladders: {format_shares(profile.ladder_shares, limit=2)}\n"
                f"{cluster_text}",
                fontsize=_TITLE_FONT_SIZE,
            )
            axes.set_xlabel("Applied N (kg N/ha)")
        axes_list[0].set_ylabel("Grain yield (t/ha)")

        handles, labels = axes_list[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            fontsize=8,
            frameon=False,
        )
        _reserve_suptitle(
            figure,
            f"source={SOURCE_NAME} — cluster-eligible trajectories of every LTCCE "
            "cropping season, on one shared frame\n"
            "the panels share both axes, so the seasons are directly comparable in "
            "height and in applied-N reach\n"
            "season is confounded with applied-N ladder and calendar era: a "
            "difference between panels is not a season effect\n"
            f"{_DISCLAIMER}",
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


def _write_season_profile_comparison(
    result: SeasonClusteringResult,
    destination: Path,
) -> None:
    """Compare the three seasons on the clustering basis, without partitioning them.

    Three seasons cannot be clustered; pretending otherwise would be the whole
    error this product exists to avoid. They can be *compared*, which is what
    this figure does: context-weighted observed means with the standard error
    across experimental contexts, and an explicit statement that the difference
    remains confounded with ladder and era.
    """

    from matplotlib import pyplot as plt

    seasons, means, errors = season_feature_matrix(result)
    figure, axes_row = plt.subplots(
        1, len(RESPONSE_TYPE_FEATURES), figsize=(18, 5.0), constrained_layout=True
    )
    try:
        positions = np.arange(len(seasons))
        colours = plt.get_cmap("tab10")(np.linspace(0.0, 0.45, max(len(seasons), 1)))
        for index, axes in enumerate(np.atleast_1d(axes_row)):
            values = means[:, index]
            spread = np.nan_to_num(errors[:, index], nan=0.0)
            axes.bar(
                positions,
                values,
                yerr=spread,
                capsize=4,
                color=colours,
                edgecolor="black",
                linewidth=0.5,
            )
            axes.set_xticks(positions)
            axes.set_xticklabels(seasons)
            axes.set_title(_FEATURE_TITLES[index], fontsize=9)
            axes.axhline(0.0, color="black", linewidth=0.8)
            for position, value, error in zip(positions, values, spread, strict=True):
                # Anchored past the error bar cap, and formatted to significant
                # digits: the saturation index is O(0.02), so a fixed 2-decimal
                # label would render every season as "-0.02".
                axes.annotate(
                    f"{value:.3g}",
                    (position, value + (error if value >= 0 else -error)),
                    textcoords="offset points",
                    xytext=(0, 5 if value >= 0 else -13),
                    ha="center",
                    fontsize=8,
                )
        figure.suptitle(
            f"source={SOURCE_NAME} — observed response profile of each LTCCE cropping season\n"
            "context-weighted means over replicate-free experimental contexts; "
            "bars are the standard error across those contexts\n"
            "three seasons cannot be partitioned, so they are compared, not clustered; "
            "season remains confounded with applied-N ladder and calendar era\n"
            f"{_DISCLAIMER}",
            fontsize=9,
        )
        _save_figure(figure, destination)
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------
# Ledger and summary
# --------------------------------------------------------------------------


_LEDGER_COLUMNS = (
    "trajectory_id",
    "season",
    "season_description",
    "cluster_eligible",
    "exclusion_reason",
    "season_cluster_id",
    "season_cluster_count",
    "season_silhouette",
    "season_assignment_stability",
    "season_replicate_co_assignment",
    "season_cluster_ladder_agreement_ari",
    "applied_n_ladder_kg_ha",
    "applied_n_step_kg_ha",
    "ladder_substratum_id",
    "substratum_cluster_id",
    "substratum_cluster_count",
    "substratum_silhouette",
    "substratum_figure_path",
    "season_ladder_substratum_id",
    "season_ladder_cluster_id",
    "season_ladder_figure_path",
    "design",
    "site",
    "year",
    "variety",
    "replicate",
    "yield_at_zero_n_t_ha",
    "observed_max_yield_t_ha",
    "response_above_zero_n_t_ha",
    "n_at_observed_peak_kg_ha",
    "relative_n_at_peak",
    "saturation_index",
    "decline_from_peak_t_ha",
    "figure_path",
    # Appended, never inserted, so a reader keyed on column position does not
    # break because a new decomposition was added. `variety_code` is the source
    # `VarCode` slot, carried for every trajectory whether or not the varcode
    # decomposition ran; the rest are per-factor and stay blank when it did not.
    "variety_code",
    "variety_code_stratum",
    "variety_code_stratum_cluster_id",
    "variety_code_figure_path",
    "design_stratum",
    "design_stratum_cluster_id",
    "design_figure_path",
    "variety_stratum",
    "variety_stratum_cluster_id",
    "variety_figure_path",
    "planting_decade",
    "planting_decade_stratum_cluster_id",
    "planting_decade_figure_path",
)

# Which ledger columns each factor family fills. Declared once so the writer and
# the column tuple above cannot drift apart.
_FACTOR_LEDGER_COLUMNS = {
    "varcode": (
        "variety_code_stratum",
        "variety_code_stratum_cluster_id",
        "variety_code_figure_path",
    ),
    "design": (
        "design_stratum",
        "design_stratum_cluster_id",
        "design_figure_path",
    ),
    "variety": (
        "variety_stratum",
        "variety_stratum_cluster_id",
        "variety_figure_path",
    ),
    "planting_year": (
        "planting_decade",
        "planting_decade_stratum_cluster_id",
        "planting_decade_figure_path",
    ),
}


def _factor_ledger_rows(
    substructures: Mapping[str, Sequence[FactorSubstructure]],
    figure_by_trajectory: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    """Flatten every factor decomposition into per-trajectory ledger fields."""

    rows: dict[str, dict[str, Any]] = collections.defaultdict(dict)
    for season_substructures in substructures.values():
        for substructure in season_substructures:
            factor = substructure.factor
            level_column, cluster_column, figure_column = _FACTOR_LEDGER_COLUMNS[factor]
            figures = figure_by_trajectory.get(factor, {})
            for stratum in substructure.strata:
                inner_of: dict[str, int] = {}
                partition = stratum.partition
                if partition is not None:
                    inner_of = {
                        trajectory_id: int(label)
                        for trajectory_id, label in zip(
                            partition.trajectory_ids, partition.labels, strict=True
                        )
                    }
                for trajectory_id in stratum.trajectory_ids:
                    inner = inner_of.get(trajectory_id)
                    rows[trajectory_id].update(
                        {
                            level_column: stratum.level,
                            cluster_column: inner + 1 if inner is not None else "",
                            figure_column: figures.get(trajectory_id, ""),
                        }
                    )
            for trajectory_id in substructure.minor_level_trajectory_ids:
                rows[trajectory_id].update(
                    {
                        level_column: "minor_level",
                        cluster_column: "",
                        figure_column: figures.get(trajectory_id, ""),
                    }
                )
            for trajectory_id in substructure.unrecorded_trajectory_ids:
                rows[trajectory_id].update(
                    {
                        level_column: "unrecorded",
                        cluster_column: "",
                        figure_column: figures.get(trajectory_id, ""),
                    }
                )
    return dict(rows)


def _substratum_ledger_rows(
    substructures: Mapping[str, Sequence[ClusterSubstructure]],
    subfigure_by_trajectory: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Flatten the nested ladder decomposition into per-trajectory ledger fields."""

    rows: dict[str, dict[str, Any]] = {}
    for cluster_substructures in substructures.values():
        for substructure in cluster_substructures:
            for index, substratum in enumerate(substructure.substrata, start=1):
                inner_of: dict[str, int] = {}
                partition = substratum.partition
                if partition is not None:
                    inner_of = {
                        trajectory_id: int(label)
                        for trajectory_id, label in zip(
                            partition.trajectory_ids, partition.labels, strict=True
                        )
                    }
                for trajectory_id in substratum.trajectory_ids:
                    inner = inner_of.get(trajectory_id)
                    rows[trajectory_id] = {
                        "applied_n_step_kg_ha": (
                            f"{substratum.step_kg_ha:g}"
                            if substratum.step_kg_ha is not None
                            else "irregular"
                        ),
                        "ladder_substratum_id": f"sub_{index}",
                        "substratum_cluster_id": inner + 1 if inner is not None else "",
                        "substratum_cluster_count": (
                            partition.cluster_count if partition is not None else ""
                        ),
                        "substratum_silhouette": (
                            f"{partition.silhouette:.6f}"
                            if partition is not None
                            else ""
                        ),
                        "substratum_figure_path": subfigure_by_trajectory.get(
                            trajectory_id, ""
                        ),
                    }
            for trajectory_id in substructure.minor_ladder_trajectory_ids:
                rows[trajectory_id] = {
                    "applied_n_step_kg_ha": "",
                    "ladder_substratum_id": "minor_ladder",
                    "substratum_cluster_id": "",
                    "substratum_cluster_count": "",
                    "substratum_silhouette": "",
                    "substratum_figure_path": subfigure_by_trajectory.get(
                        trajectory_id, ""
                    ),
                }
    return rows


def _season_ladder_ledger_rows(
    structures: Mapping[str, ClusterSubstructure],
    figure_by_trajectory: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Per-trajectory fields from the season-level applied-N decomposition."""

    rows: dict[str, dict[str, Any]] = {}
    for structure in structures.values():
        for index, substratum in enumerate(structure.substrata, start=1):
            inner_of: dict[str, int] = {}
            partition = substratum.partition
            if partition is not None:
                inner_of = {
                    trajectory_id: int(label)
                    for trajectory_id, label in zip(
                        partition.trajectory_ids, partition.labels, strict=True
                    )
                }
            for trajectory_id in substratum.trajectory_ids:
                inner = inner_of.get(trajectory_id)
                rows[trajectory_id] = {
                    "season_ladder_substratum_id": f"sub_{index}",
                    "season_ladder_cluster_id": (
                        inner + 1 if inner is not None else ""
                    ),
                    "season_ladder_figure_path": figure_by_trajectory.get(
                        trajectory_id, ""
                    ),
                }
        for trajectory_id in structure.minor_ladder_trajectory_ids:
            rows[trajectory_id] = {
                "season_ladder_substratum_id": "minor_ladder",
                "season_ladder_cluster_id": "",
                "season_ladder_figure_path": figure_by_trajectory.get(
                    trajectory_id, ""
                ),
            }
    return rows


def _write_ledger(
    result: SeasonClusteringResult,
    destination: Path,
    figure_by_trajectory: Mapping[str, str],
    substratum_fields: Mapping[str, Mapping[str, Any]],
    season_ladder_fields: Mapping[str, Mapping[str, Any]],
    factor_fields: Mapping[str, Mapping[str, Any]] = {},
) -> None:
    """Emit every seasoned trajectory, clustered or held out, with its reason."""

    label_of: dict[str, tuple[str, ClusterPartition, int]] = {}
    for season, partition in result.partitions.items():
        for trajectory_id, label in zip(
            partition.trajectory_ids, partition.labels, strict=True
        ):
            label_of[trajectory_id] = (season, partition, int(label))

    rows: list[dict[str, Any]] = []
    for season in result.seasons:
        for trajectory_id in sorted(result.season_trajectory_ids[season]):
            context = result.contexts.get(trajectory_id)
            feature = result.features.get(trajectory_id)
            assigned = label_of.get(trajectory_id)
            concordance = result.ladder_concordance.get(season)
            row: dict[str, Any] = {column: "" for column in _LEDGER_COLUMNS}
            row.update(
                {
                    "trajectory_id": trajectory_id,
                    "season": season,
                    "season_description": season_label(season),
                    "cluster_eligible": "true" if feature is not None else "false",
                    "exclusion_reason": result.excluded.get(trajectory_id, ""),
                    "figure_path": figure_by_trajectory.get(trajectory_id, ""),
                }
            )
            if assigned is not None:
                _, partition, label = assigned
                row.update(
                    {
                        "season_cluster_id": label + 1,
                        "season_cluster_count": partition.cluster_count,
                        "season_silhouette": f"{partition.silhouette:.6f}",
                        "season_assignment_stability": (
                            f"{partition.assignment_stability:.6f}"
                        ),
                        "season_replicate_co_assignment": (
                            ""
                            if math.isnan(partition.replicate_coherence)
                            else f"{partition.replicate_coherence:.6f}"
                        ),
                        "season_cluster_ladder_agreement_ari": (
                            f"{concordance.adjusted_rand:.6f}" if concordance else ""
                        ),
                    }
                )
            if feature is not None:
                row.update(
                    {
                        "applied_n_ladder_kg_ha": ladder_text(feature.ladder),
                        "yield_at_zero_n_t_ha": f"{feature.yield_at_zero_n_t_ha:.4f}",
                        "observed_max_yield_t_ha": f"{feature.observed_max_yield_t_ha:.4f}",
                        "response_above_zero_n_t_ha": (
                            f"{feature.response_above_zero_n_t_ha:.4f}"
                        ),
                        "n_at_observed_peak_kg_ha": f"{feature.n_at_observed_peak_kg_ha:g}",
                        "relative_n_at_peak": f"{feature.relative_n_at_peak:.4f}",
                        "saturation_index": f"{feature.saturation_index:.6f}",
                        "decline_from_peak_t_ha": f"{feature.decline_from_peak_t_ha:.4f}",
                    }
                )
            if context is not None:
                row.update(
                    {
                        "design": context.design,
                        "site": context.site,
                        "year": context.year if context.year > 0 else "",
                        "variety": context.variety,
                        "replicate": context.replicate,
                        "variety_code": context.variety_code,
                    }
                )
            row.update(substratum_fields.get(trajectory_id, {}))
            row.update(season_ladder_fields.get(trajectory_id, {}))
            row.update(factor_fields.get(trajectory_id, {}))
            rows.append(row)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _json_number(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def _partition_record(partition: ClusterPartition) -> dict[str, Any]:
    return {
        "partition_id": partition.partition_id,
        "basis": partition.basis,
        "member_count": partition.member_count,
        "cluster_count": partition.cluster_count,
        "silhouette": partition.silhouette,
        "assignment_stability_adjusted_rand": partition.assignment_stability,
        "feature_names": list(partition.feature_names),
        "replicate_co_assignment_rate": _json_number(partition.replicate_coherence),
        "separation_is_weak": partition.separation_is_weak,
        "cluster_sizes": [
            sum(1 for label in partition.labels if label == cluster_id)
            for cluster_id in range(partition.cluster_count)
        ],
    }


def _substratum_record(substratum: LadderSubstratum) -> dict[str, Any]:
    partition = substratum.partition
    record: dict[str, Any] = {
        "applied_n_ladder_kg_ha": list(substratum.ladder),
        "applied_n_step_kg_ha": substratum.step_kg_ha,
        "step_is_uniform": substratum.step_kg_ha is not None,
        "trajectory_count": len(substratum.trajectory_ids),
        "level": (
            "design stratum: exact applied-N ladder match, nothing was clustered "
            "to produce it"
        ),
    }
    if partition is None:
        record["sub_partition"] = None
        record["not_subclustered_reason"] = substratum.not_subclustered_reason
        return record
    record["sub_partition"] = {
        "basis": partition.basis,
        "cluster_count": partition.cluster_count,
        "silhouette": partition.silhouette,
        "assignment_stability_adjusted_rand": partition.assignment_stability,
        "replicate_co_assignment_rate": _json_number(partition.replicate_coherence),
        "separation_is_weak": partition.separation_is_weak,
        "dominant_axis": partition.dominant_axis,
        "between_cluster_level_variance": _json_number(
            partition.between_cluster_level_variance
        ),
        "between_cluster_shape_variance": _json_number(
            partition.between_cluster_shape_variance
        ),
        "cluster_sizes": [
            sum(1 for label in partition.labels if label == cluster_id)
            for cluster_id in range(partition.cluster_count)
        ],
    }
    return record


def _substructure_record(substructure: ClusterSubstructure) -> dict[str, Any]:
    return {
        "scope": substructure.scope,
        "cluster_id": (
            substructure.cluster_id + 1
            if substructure.cluster_id is not None
            else None
        ),
        "member_count": substructure.member_count,
        "distinct_applied_n_ladders": substructure.distinct_ladder_count,
        "minimum_substratum_size": MIN_LADDER_SUBSTRATUM,
        "minimum_subclustered_size": MIN_SUBCLUSTERED_SUBSTRATUM,
        "minor_ladder_count": substructure.minor_ladder_count,
        "minor_ladder_trajectory_count": len(substructure.minor_ladder_trajectory_ids),
        "ladder_substrata": [
            {"substratum_id": f"sub_{index}", **_substratum_record(substratum)}
            for index, substratum in enumerate(substructure.substrata, start=1)
        ],
    }


def _factor_stratum_record(
    result: SeasonClusteringResult,
    stratum: FactorStratum,
) -> dict[str, Any]:
    partition = stratum.partition
    years = cluster_year_range(stratum.trajectory_ids, result.contexts)
    record: dict[str, Any] = {
        "level": stratum.level,
        "trajectory_count": len(stratum.trajectory_ids),
        "year_range": list(years or ()),
        "distinct_variety_count": len(stratum.variety_names),
        "variety_names": list(stratum.variety_names),
        "applied_n_ladder_shares": dict(stratum.ladder_shares),
        "distinct_applied_n_ladders": stratum.distinct_ladder_count,
        "season_cluster_shares": dict(stratum.season_cluster_shares),
        "observed_centroid": centroid_summary(stratum.trajectory_ids, result.features),
        "stratum_kind": (
            "recorded stratum: every member carries this value in the source, "
            "nothing was clustered to produce it"
        ),
    }
    if partition is None:
        record["sub_partition"] = None
        record["not_subclustered_reason"] = stratum.not_subclustered_reason
        return record
    record["sub_partition"] = {
        "basis": partition.basis,
        "cluster_count": partition.cluster_count,
        "silhouette": partition.silhouette,
        "assignment_stability_adjusted_rand": partition.assignment_stability,
        "replicate_co_assignment_rate": _json_number(partition.replicate_coherence),
        "separation_is_weak": partition.separation_is_weak,
        "cluster_sizes": [
            sum(1 for label in partition.labels if label == cluster_id)
            for cluster_id in range(partition.cluster_count)
        ],
    }
    return record


def _factor_substructure_record(
    result: SeasonClusteringResult,
    substructure: FactorSubstructure,
) -> dict[str, Any]:
    definition = substructure.definition
    return {
        "factor": substructure.factor,
        "source_column": definition.source_column,
        "factor_description": definition.title,
        "directory": definition.directory,
        **(
            {
                "artifact_directories": {
                    "annotated": PLANTING_YEAR_ANNOTATED_DIRNAME,
                    "figure_only": PLANTING_YEAR_FIGURE_ONLY_DIRNAME,
                    **(
                        {
                            "replicate_sheets": (
                                f"{PLANTING_YEAR_FIGURE_ONLY_DIRNAME}/"
                                f"{PLANTING_YEAR_REPLICATE_DIRNAME}"
                            )
                        }
                        if substructure.season.strip().upper() == "DS"
                        else {}
                    ),
                }
            }
            if substructure.factor == FACTOR_PLANTING_YEAR
            else {}
        ),
        "identity_caveat": definition.identity_caveat,
        "member_count": substructure.member_count,
        "distinct_level_count": substructure.distinct_level_count,
        "minimum_stratum_size": substructure.minimum_stratum,
        "minimum_subclustered_size": MIN_SUBCLUSTERED_FACTOR_STRATUM,
        "sub_partition_basis": (
            "standardized ladder-invariant features, not the raw observed yield "
            "vector: a factor level generally spans several applied-N supports, "
            "so the raw vector is not comparable within it"
        ),
        "minor_levels": list(substructure.minor_levels),
        "minor_level_trajectory_count": len(substructure.minor_level_trajectory_ids),
        "unrecorded_trajectory_count": len(substructure.unrecorded_trajectory_ids),
        "agreement": {
            "note": (
                "Adjusted Rand of the factor's level labels against the applied-N "
                "ladder labels, and against this season's response-cluster "
                "labels. High ladder agreement means the factor restates the "
                "experimental design and its calendar era; high cluster "
                "agreement means the response partition already encodes it."
            ),
            "disclosure_threshold": LADDER_DRIVEN_ADJUSTED_RAND,
            "level_vs_applied_n_ladder_adjusted_rand": _json_number(
                substructure.ladder_agreement
            ),
            "level_restates_applied_n_design": substructure.is_ladder_driven,
            "level_vs_season_cluster_adjusted_rand": _json_number(
                substructure.season_cluster_agreement
            ),
            "season_partition_encodes_level": substructure.is_cluster_driven,
        },
        "levels": [
            _factor_stratum_record(result, stratum)
            for stratum in substructure.strata
        ],
        **(
            _annual_trend_record(result, substructure.season)
            if substructure.factor == FACTOR_PLANTING_YEAR
            else {}
        ),
    }


def _annual_trend_record(
    result: SeasonClusteringResult,
    season: str,
) -> dict[str, Any]:
    """The numbers behind `annotated/annual_trend.jpeg`, so it is auditable.

    The decade strata are a banding of a continuous axis; carrying the
    unbanded per-year values here is what keeps that banding a presentational
    choice rather than a loss of resolution.
    """

    records = annual_response_records(result, season)
    if not records:
        return {}
    return {
        "annual_trend": {
            "note": (
                "Per planting year, context-weighted over replicate-free "
                "design-site-year-season-variety-ladder contexts. The standard "
                "error is the spread of that year's experimental contexts, not "
                "sampling error of a fitted quantity: nothing is fitted."
            ),
            "applied_n_eras": [
                {"applied_n_ladder_kg_ha": ladder, "first_year": first,
                 "last_year": last}
                for ladder, first, last in annual_ladder_eras(records)
            ],
            "years": [
                {
                    "year": record.year,
                    "planting_decade": record.decade,
                    "trajectory_count": record.trajectory_count,
                    "experimental_context_count": record.context_count,
                    "dominant_applied_n_ladder_kg_ha": record.ladder,
                    "carries_more_than_one_ladder": record.ladder_is_mixed,
                    "context_weighted_feature_means": {
                        name: _json_number(value)
                        for name, value in record.means.items()
                    },
                    "context_weighted_feature_standard_errors": {
                        name: _json_number(value)
                        for name, value in record.standard_errors.items()
                    },
                }
                for record in records
            ],
        }
    }


def _season_record(
    result: SeasonClusteringResult,
    season: str,
    figure_by_trajectory: Mapping[str, str],
    substructures: Sequence[ClusterSubstructure] = (),
    season_ladder: ClusterSubstructure | None = None,
    factor_substructures: Sequence[FactorSubstructure] = (),
) -> dict[str, Any]:
    profile = result.profiles[season]
    partition = result.partitions.get(season)
    concordance = result.ladder_concordance.get(season)

    record: dict[str, Any] = {
        "season": season,
        "season_description": season_label(season),
        "cluster_eligible_trajectories": len(profile.trajectory_ids),
        "held_out_trajectories": len(profile.excluded_trajectory_ids),
        "held_out_by_reason": dict(profile.exclusions_by_reason),
        "experimental_context_count": profile.experimental_context_count,
        "year_range": list(profile.year_range or ()),
        "variety_count": profile.variety_count,
        "distinct_applied_n_ladders": profile.distinct_ladder_count,
        "applied_n_ladder_shares": dict(profile.ladder_shares),
        "context_weighted_feature_means": dict(
            zip(RESPONSE_TYPE_FEATURES, profile.feature_means, strict=True)
        ),
        "context_weighted_feature_standard_errors": {
            name: _json_number(value)
            for name, value in zip(
                RESPONSE_TYPE_FEATURES, profile.feature_standard_errors, strict=True
            )
        },
    }
    if partition is None:
        record["clustered"] = False
        record["not_clustered_reason"] = result.unclustered_seasons.get(season, "")
        return record

    record["clustered"] = True
    record["partition"] = _partition_record(partition)
    record["ladder_concordance"] = {
        "note": (
            "Adjusted Rand index of the within-season cluster labels against the "
            "applied-N ladder labels. High agreement means the partition recovers "
            "the experimental design rather than an N-response contrast."
        ),
        "adjusted_rand": concordance.adjusted_rand if concordance else None,
        "disclosure_threshold": LADDER_DRIVEN_ADJUSTED_RAND,
        "is_ladder_driven": concordance.is_ladder_driven if concordance else None,
        "ladder_dominated_cluster_ids": (
            [cluster_id + 1 for cluster_id in concordance.ladder_dominated_clusters()]
            if concordance
            else []
        ),
        "ladder_dominated_share_threshold": LADDER_DOMINATED_CLUSTER_SHARE,
        "cluster_by_ladder_counts": (
            {
                str(cluster_id + 1): counts
                for cluster_id, counts in concordance.contingency.items()
            }
            if concordance
            else {}
        ),
    }

    clusters: list[dict[str, Any]] = []
    for cluster_id in range(partition.cluster_count):
        members = season_cluster_members(partition, cluster_id)
        centroid = centroid_summary(members, result.features)
        years = cluster_year_range(members, result.contexts)
        clusters.append(
            {
                "cluster_id": cluster_id + 1,
                "trajectory_count": len(members),
                "observed_centroid": centroid,
                "descriptive_label": response_type_label(centroid),
                "applied_n_ladder_shares": cluster_ladder_shares(
                    members, result.features
                ),
                "dominant_ladder_share": (
                    _json_number(concordance.dominant_ladder_share[cluster_id])
                    if concordance
                    else None
                ),
                "year_range": list(years or ()),
                "variety_count": cluster_variety_span(members, result.contexts),
                "figure": figure_by_trajectory.get(members[0], "") if members else "",
            }
        )
    record["clusters"] = clusters
    if substructures:
        record["nested_ladder_substructure"] = {
            "note": (
                "Each within-season cluster is split by exact applied-N ladder — "
                "the only fertilizer contrast this source varies, since LTCCE "
                "records no P or K column and its workbook names the treatment "
                "factors as N fertilizer rate and Variety. A ladder sub-stratum "
                "shares one applied-N support, which is what makes the raw "
                "observed yield vector clusterable inside it."
            ),
            "clusters": [
                _substructure_record(substructure) for substructure in substructures
            ],
        }
    if season_ladder is not None:
        record["season_level_applied_n_substructure"] = {
            "note": (
                "The same exact-ladder decomposition applied to the whole "
                "season rather than inside one response cluster: what the "
                "applied-N design separates on its own, before any "
                "response-type partition is imposed."
            ),
            "minimum_substratum_size": MIN_SEASON_LADDER_SUBSTRATUM,
            **_substructure_record(season_ladder),
        }
    if factor_substructures:
        record["experimental_factor_substructure"] = {
            "note": (
                "The same season split by a recorded experimental factor rather "
                "than by the applied-N ladder, and clustered again inside each "
                "level. Each family reports how far its levels restate the "
                "applied-N design and how far the season's own response "
                "partition already encodes them."
            ),
            "factors": [
                _factor_substructure_record(result, substructure)
                for substructure in factor_substructures
            ],
        }
    return record


def _write_summary(
    result: SeasonClusteringResult,
    overlay: SourceDatasetOverlay,
    destination: Path,
    limits: _SharedAxisLimits,
    figure_by_trajectory: Mapping[str, str],
    substructures: Mapping[str, Sequence[ClusterSubstructure]],
    season_ladder_structures: Mapping[str, ClusterSubstructure],
    factor_substructures: Mapping[str, Sequence[FactorSubstructure]],
    *,
    minimum_season_size: int,
    maximum_clusters: int,
) -> dict[str, Any]:
    exclusion_counts = collections.Counter(result.excluded.values())
    summary = {
        "artifact_class": "exploratory_diagnostic",
        "governed_analysis": False,
        "governance_note": (
            "ANA-11 disables the curve_feature_clustering analysis family for the "
            "current primary release. These views are descriptive only and are not "
            "part of the release inventory or any checksum ledger."
        ),
        "source_name": SOURCE_NAME,
        "unit_clustered": "replicate trajectory, stratified by cropping season",
        "method": (
            "Trajectories are split by recorded LTCCE season and clustered inside "
            "each season on the same four ladder-invariant features the pooled "
            "response-type product uses (zero-N yield, response above zero N, "
            "relative N at the observed peak, saturation index), standardized, with "
            "k chosen by maximum silhouette over 2..maximum_clusters. Each season "
            "chooses its own k and its own centroids."
        ),
        "cluster_label_scope": (
            "Cluster ids are local to their season. DS cluster 1 and LWS cluster 1 "
            "are unrelated objects and must not be cross-matched or compared by "
            "number."
        ),
        "interpretation_caveat": (
            "Stratifying by season removes the season contrast from each partition "
            "but not the applied-N ladder, which remains unevenly represented "
            "within every season. Each season therefore reports the agreement "
            "between its cluster labels and its ladder labels; read a partition "
            "with high agreement as a restatement of the experimental design."
        ),
        "cross_season_caveat": (
            "The season profile comparison is descriptive. Season is confounded "
            "with applied-N ladder and calendar era in LTCCE, so a difference "
            "between seasons is not a season effect."
        ),
        "minimum_season_size": minimum_season_size,
        "maximum_clusters": maximum_clusters,
        "observed_yield_range_t_ha": list(overlay.summary.yield_range_t_ha or ()),
        "observed_n_rate_range_kg_ha": list(overlay.summary.n_rate_range_kg_ha or ()),
        "shared_yield_axis_t_ha": list(limits.y or ()),
        "shared_n_rate_axis_kg_ha": list(limits.x or ()),
        "total_trajectories": overlay.summary.trajectory_count,
        "total_finite_observations": overlay.summary.finite_observation_count,
        "cluster_eligible_trajectories": len(result.features),
        "held_out_trajectories": len(result.excluded),
        "held_out_by_reason": dict(sorted(exclusion_counts.items())),
        "observed_seasons": list(result.seasons),
        "clustered_season_count": len(result.partitions),
        "unclustered_seasons": dict(sorted(result.unclustered_seasons.items())),
        "seasons_with_nested_ladder_substructure": sorted(substructures),
        "seasons_with_experimental_factor_substructure": {
            season: [substructure.factor for substructure in families]
            for season, families in sorted(factor_substructures.items())
            if families
        },
        "experimental_factor_method": (
            "A season is split by a recorded experimental factor — the LTCCE "
            "VarCode varietal slot, the plot design, or the variety designation "
            "— and each level large enough to support it is clustered on the "
            "same standardized ladder-invariant features the season partition "
            "uses. The raw yield vector is deliberately not used here: unlike an "
            "applied-N sub-stratum, a factor level does not share one N support. "
            "VarCode in particular is a position in the plot layout and not a "
            "genotype; one code carries many varieties across the years."
        ),
        "seasons": [
            _season_record(
                result,
                season,
                figure_by_trajectory,
                substructures.get(season, ()),
                season_ladder_structures.get(season),
                factor_substructures.get(season, ()),
            )
            for season in result.seasons
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return summary


def _write_season_readme(
    result: SeasonClusteringResult,
    season: str,
    substructures: Sequence[ClusterSubstructure],
    destination: Path,
    factor_substructures: Sequence[FactorSubstructure] = (),
    season_ladder: ClusterSubstructure | None = None,
    wrote_cluster_figures: bool = False,
    wrote_ladder_composition: bool = False,
    wrote_held_out_figures: bool = False,
) -> None:
    """Write the explainer that lives inside one season folder.

    Its first job is to say what `0/65/130/195` means, because that notation is
    on nearly every figure in the folder and on none of them is there room to
    define it.
    """

    profile = result.profiles[season]
    partition = result.partitions.get(season)
    concordance = result.ladder_concordance.get(season)
    records = season_ladder_records(result, season)
    disjoint = ladders_are_disjoint_in_time(records)
    features = profile.feature_map()

    lines: list[str] = [
        f"# {season_label(season)} — N-response clusters\n\n",
        f"{len(profile.trajectory_ids)} cluster-eligible replicate trajectories "
        f"from the LTCCE {season_label(season)}, over "
        f"{profile.experimental_context_count} replicate-free experimental "
        f"contexts and {profile.variety_count} varieties. "
        f"{len(profile.excluded_trajectory_ids)} further trajectories are held "
        "out of clustering; see the table at the end.\n\n",
        "## Reading `0/65/130/195` and the other slash-separated numbers\n\n",
        "Those are **applied nitrogen rates in kg N/ha** — the fertilizer ladder "
        "one plot received across the season's treatments, written from lowest "
        "to highest. `0/65/130/195` means four treatments: an unfertilized "
        "control at 0, then 65, then 130, then 195 kg N/ha. They are the "
        "x-axis positions of that trajectory's points in every figure here, and "
        "each trajectory contributes exactly one observed yield at each rate.\n\n",
        "A ladder's **step** is the gap between consecutive rates. "
        "`0/65/130/195` steps uniformly by 65; `0/50/100/150` steps by 50. "
        "`0/40/60/120` has no single step (40, then 20, then 60) and is "
        "reported as *irregular* rather than given a misleading average.\n\n",
        "Ladders are never mixed. A trajectory on `0/45/90` is not folded into "
        "`0/45/90/135`, because its yield at 135 kg N/ha was never observed and "
        "assuming one would be extrapolation.\n\n",
        f"### Applied-N ladders observed in the {season} season\n\n",
        "| Applied N (kg N/ha) | Step | Trajectories | Share | Years | Varieties |\n",
        "| --- | --- | --- | --- | --- | --- |\n",
    ]
    for record in records:
        lines.append(
            f"| `{ladder_text(record.ladder)}` | {record.step_text} | "
            f"{record.trajectory_count} | {100 * record.share:.1f}% | "
            f"{record.year_text} | {record.variety_count} |\n"
        )

    lines.append("\n")
    if disjoint:
        lines.append(
            "**The ladder is also a calendar era.** Read the Years column: the "
            "major ladders above do not overlap in time at all. LTCCE replaced "
            "one N design with the next rather than running them side by side, "
            "so within this season a ladder label is equally an era label, and "
            "any comparison between two ladders is also a comparison between "
            "two periods of the experiment. That confounding cannot be removed "
            "by stratification — only stated.\n\n"
        )
    else:
        lines.append(
            "**The ladder is close to a calendar era.** Read the Years column: "
            "the major ladders sit in largely distinct periods, because LTCCE "
            "replaced one N design with the next. A comparison between two "
            "ladders is therefore substantially a comparison between two "
            "periods of the experiment, and the overlap is too small to "
            "separate the two.\n\n"
        )

    lines.append("## What is in this folder\n\n")
    lines.append(
        "- `overview.jpeg` — every cluster-eligible trajectory in this season, "
        "on the frame shared by the whole product.\n"
    )
    if partition is not None:
        if wrote_cluster_figures:
            lines.append(
                f"- `cluster_1.jpeg` … `cluster_{partition.cluster_count}.jpeg` — "
                f"the {partition.cluster_count} within-season clusters.\n"
            )
        if wrote_ladder_composition:
            lines.append(
                "- `ladder_composition.jpeg` — how the applied-N ladders are "
                f"distributed across this season's {partition.cluster_count} "
                "response clusters.\n"
            )
    if wrote_held_out_figures:
        lines.append(
            "- `held_out_<reason>.jpeg` — one figure per eligibility exclusion.\n"
        )
    if season_ladder is not None:
        lines.append(
            "- `by_applied_n/` — the whole season split by exact applied-N "
            "ladder, with a further clustering inside each ladder large enough "
            "to support one. A ladder sub-stratum shares one applied-N support, "
            "which is the only level at which the raw observed yield vector may "
            "be clustered; it is also a calendar era, so read the Years column "
            "above before comparing two of them.\n"
        )
    if substructures:
        lines.append(
            "- `cluster_<i>/` — the applied-N ladder decomposition of cluster "
            "`i`, and a further clustering inside each ladder. "
            "`ladder_comparison.jpeg` puts that cluster's ladders side by side on "
            "the shared frame. See the top-level `../README.md` for what "
            "separates the two nested levels.\n"
        )
    for substructure in factor_substructures:
        definition = substructure.definition
        lines.append(
            f"- `{definition.directory}/` — the whole season split by "
            f"{definition.title} (source column `{definition.source_column}`), "
            "with a further clustering inside each level large enough to support "
            f"one. {definition.identity_caveat[0].upper()}"
            f"{definition.identity_caveat[1:]}; that folder's own `README.md` "
            "carries the detail.\n"
        )
    lines.append("\n")

    if factor_substructures:
        lines.append("## Does a recorded factor explain the response?\n\n")
        lines.append(
            "Each folder above reports two adjusted Rand indices against the "
            f"{LADDER_DRIVEN_ADJUSTED_RAND:.2f} threshold used throughout this "
            "product — the factor's levels against the applied-N ladder, and "
            "against this season's response clusters.\n\n"
            "| Factor | Source column | Levels | vs applied-N ladder | vs response clusters |\n"
            "| --- | --- | --- | --- | --- |\n"
        )
        for substructure in factor_substructures:
            definition = substructure.definition
            lines.append(
                f"| {definition.title} | `{definition.source_column}` "
                f"| {substructure.distinct_level_count} "
                f"| {substructure.ladder_agreement:.3f}"
                + (
                    " — restates the design"
                    if substructure.is_ladder_driven
                    else " — evenly spread"
                    if substructure.ladder_agreement_is_negligible
                    else " — below threshold, uneven"
                )
                + f" | {substructure.season_cluster_agreement:.3f}"
                + (
                    " — already encoded"
                    if substructure.is_cluster_driven
                    else " — no signal"
                    if substructure.cluster_agreement_is_negligible
                    else " — below threshold, leaning"
                )
                + " |\n"
            )
        lines.append("\n")

    if partition is not None:
        lines.append("## How this season's partition behaves\n\n")
        lines.append(
            f"- **k = {partition.cluster_count}**, chosen by maximum silhouette.\n"
            f"- **Silhouette {partition.silhouette:.3f}** — "
            + (
                "at or below the 0.40 weak-separation threshold, so read the "
                "clusters as a descriptive banding of a continuum, not as "
                "discrete response types.\n"
                if partition.separation_is_weak
                else "above the 0.40 weak-separation threshold.\n"
            )
        )
        lines.append(
            f"- **Seed stability (ARI) {partition.assignment_stability:.2f}** "
            "against alternate k-means seeds.\n"
        )
        lines.append(
            f"- **Replicates co-assigned {replicate_coherence_text(partition)}** — "
            "the share of experimental units whose replicates all land in one "
            "cluster.\n"
        )
        if concordance is not None:
            lines.append(
                f"- **Cluster/ladder agreement ARI {concordance.adjusted_rand:.3f}** — "
                + (
                    "at or above the 0.20 disclosure threshold, so this partition "
                    "substantially re-derives the applied-N design.\n"
                    if concordance.is_ladder_driven
                    else "below the 0.20 disclosure threshold, so this partition is "
                    "not a relabelling of the applied-N design.\n"
                )
            )
        lines.append(
            "\nCluster numbers are local to this season. `cluster_1` here has no "
            "relationship to `cluster_1` in another season folder.\n\n"
        )
    else:
        lines.append(
            "## No partition\n\nThis season was not clustered "
            f"({result.unclustered_seasons.get(season, 'reason unrecorded')}).\n\n"
        )

    lines.append("## Observed season means\n\n")
    lines.append(
        "Context-weighted over replicate-free experimental contexts, so a year "
        "with four replicates does not outweigh a year with one.\n\n"
    )
    lines.append("| Quantity | Value |\n| --- | --- |\n")
    lines.append(
        f"| Yield at zero N | {features['yield_at_zero_n_t_ha']:.2f} t/ha |\n"
        f"| Response above zero N | {features['response_above_zero_n_t_ha']:.2f} t/ha |\n"
        f"| Relative N at observed peak | {features['relative_n_at_peak']:.3f} "
        "of the top rate applied |\n"
        f"| Saturation index | {features['saturation_index']:.4f} t/ha per kg N |\n\n"
    )

    if profile.exclusions_by_reason:
        lines.append("## Held out of clustering\n\n| Reason | Trajectories |\n| --- | --- |\n")
        for reason, count in profile.exclusions_by_reason.items():
            lines.append(f"| `{reason}` | {count} |\n")
        lines.append("\n")
        if not wrote_held_out_figures:
            # The exclusions are still fully disclosed; only the overlays of
            # them are not drawn. Say where they went, or the table above reads
            # as a pointer to figures that are not there.
            lines.append(
                "The held-out trajectories are not drawn. Every one of them is "
                "in `../season_cluster_assignments.csv` with "
                "`cluster_eligible=false` and its `exclusion_reason`, and the "
                "counts are in `../season_clustering_summary.json`. Pass "
                "`--write-held-out-figures` to the generator to draw them.\n\n"
            )

    lines.append(
        "## Interpretation boundary\n\n"
        "Exploratory diagnostic, outside the governed analysis inventory "
        "(ANA-11). No curve is fitted anywhere in this folder; every centroid is "
        "a mean of observed yields at observed N rates. Full method and caveats "
        "are in `../season_clustering_summary.json`.\n"
    )
    destination.write_text("".join(lines), encoding="utf-8")


def _nested_readme_section(
    substructures: Mapping[str, Sequence[ClusterSubstructure]],
    season_ladder_structures: Mapping[str, ClusterSubstructure] = {},
    *,
    write_subcluster_figures: bool = False,
) -> str:
    if not substructures and not season_ladder_structures:
        return ""
    lines: list[str] = [
        "## Nested level: applied-N ladder inside each cluster\n",
        "LTCCE records no P or K column — its workbook names the treatment "
        "factors as N fertilizer rate and Variety — so the applied-N ladder is "
        "the only fertilizer contrast the source varies. Each cluster of the "
        "seasons below is therefore split by exact ladder, and clustered again "
        "inside each ladder.\n",
        "The two nested levels are not the same kind of object:\n",
        f"- `sub_<j>_<ladder>.jpeg` is a **design stratum**. Every trajectory in "
        f"it received the same applied-N ladder; nothing was clustered to "
        f"produce it. Ladders holding fewer than {MIN_LADDER_SUBSTRATUM} of a "
        "cluster's trajectories are pooled into `sub_other_ladders.jpeg` and "
        "never clustered together, because they do not share an N support.\n",
        (
            "- `sub_<j>_<ladder>_cluster_<m>.jpeg` is a **k-means partition** "
            if write_subcluster_figures
            else "- the **k-means sub-partition** inside each sub-stratum, "
            "reported on the panel titles and in the summary JSON rather than "
            "drawn as its own figure, runs "
        )
        + "of the raw observed yield vector, unscaled. That is admissible here "
        "and only here: inside one ladder every trajectory is measured at the "
        "same applied-N levels, so Euclidean distance is already the "
        f"agronomically meaningful one. Sub-strata below "
        f"{MIN_SUBCLUSTERED_SUBSTRATUM} trajectories are not partitioned at "
        "all.\n",
    ]
    for season in sorted(season_ladder_structures):
        structure = season_ladder_structures[season]
        lines.append(
            f"\n`{_season_token(season)}/by_applied_n/` — the same decomposition "
            f"applied to the **whole season** ({structure.member_count} "
            "trajectories) rather than inside one response cluster, so it shows "
            f"what the applied-N design separates on its own. Sub-strata here "
            f"need only {MIN_SEASON_LADDER_SUBSTRATUM} trajectories:\n"
        )
        for index, substratum in enumerate(structure.substrata, start=1):
            partition = substratum.partition
            detail = (
                f"{partition.cluster_count} sub-clusters, silhouette "
                f"{partition.silhouette:.3f}, mainly separating "
                f"{partition.dominant_axis}"
                if partition is not None
                else f"not sub-clustered ({substratum.not_subclustered_reason})"
            )
            lines.append(
                f"- `sub_{index}` {ladder_text(substratum.ladder)} kg N/ha "
                f"({substratum.step_text}), {len(substratum.trajectory_ids)} "
                f"trajectories — {detail}.\n"
            )
        if structure.minor_ladder_trajectory_ids:
            lines.append(
                f"- `sub_other_ladders` — "
                f"{len(structure.minor_ladder_trajectory_ids)} trajectories over "
                f"{structure.minor_ladder_count} minor ladders.\n"
            )

    for season in sorted(substructures):
        for substructure in substructures[season]:
            token = _season_token(season)
            lines.append(
                f"\n`{token}/{_substructure_directory(substructure)}/` — "
                f"{substructure.member_count} trajectories over "
                f"{substructure.distinct_ladder_count} ladders:\n"
            )
            for index, substratum in enumerate(substructure.substrata, start=1):
                partition = substratum.partition
                detail = (
                    f"{partition.cluster_count} sub-clusters, silhouette "
                    f"{partition.silhouette:.3f}, mainly separating "
                    f"{partition.dominant_axis}"
                    if partition is not None
                    else f"not sub-clustered ({substratum.not_subclustered_reason})"
                )
                lines.append(
                    f"- `sub_{index}` {ladder_text(substratum.ladder)} kg N/ha "
                    f"({substratum.step_text}), {len(substratum.trajectory_ids)} "
                    f"trajectories — {detail}.\n"
                )
            if substructure.minor_ladder_trajectory_ids:
                lines.append(
                    f"- `sub_other_ladders` — "
                    f"{len(substructure.minor_ladder_trajectory_ids)} trajectories "
                    f"over {substructure.minor_ladder_count} minor ladders.\n"
                )
    lines.append("\n")
    return "".join(lines)


def _factor_readme_section(
    factor_substructures: Mapping[str, Sequence[FactorSubstructure]],
    *,
    write_subcluster_figures: bool = False,
) -> str:
    """The top-level explainer for the recorded-factor decompositions."""

    families = {
        season: substructures
        for season, substructures in factor_substructures.items()
        if substructures
    }
    if not families:
        return ""

    lines: list[str] = [
        "## Nested level: recorded experimental factors\n",
        "A season can also be split by something the source recorded directly "
        "rather than by the fertilizer support its trajectories were grown on. "
        "Those decompositions sit beside the applied-N ones, one folder per "
        "factor, and each level large enough to support it is clustered again "
        "inside.\n",
        "The nested levels are again two different kinds of object:\n",
        "- `<level>.jpeg` is a **recorded stratum**. Every trajectory in it "
        "carries that value in the source; nothing was clustered to produce it. "
        f"Levels below {MIN_FACTOR_STRATUM} trajectories are pooled into "
        "`minor_levels.jpeg`. In `by_variety/` the filename is "
        "`<release year>_<level>.jpeg`, the year joined in from the "
        f"{VARIETY_RELEASE_YEAR_SOURCE}. The year labels the level and is used "
        "nowhere in the clustering; a level with no released identity keeps its "
        "bare designation.\n",
        (
            "- `<level>_cluster_<m>.jpeg` is a **k-means partition** — but on "
            if write_subcluster_figures
            else "- the **k-means sub-partition** inside each level, reported on "
            "the factor folder's `level_comparison.jpeg` panel titles "
            "(`annotated/level_comparison.jpeg` for planting year) and in the "
            "summary JSON rather than drawn as its own figure, runs on "
        )
        + "the standardized ladder-invariant features, *not* on the raw yield "
        "vector the applied-N sub-strata use. A factor level generally spans "
        "several applied-N supports, so the raw vector is not comparable within "
        f"it. Levels below {MIN_SUBCLUSTERED_FACTOR_STRATUM} trajectories are "
        "not partitioned at all.\n",
        "\nEach folder scores its levels twice, both as adjusted Rand indices "
        f"against the {LADDER_DRIVEN_ADJUSTED_RAND:.2f} threshold used "
        "throughout: against the applied-N ladder labels (does the factor just "
        "restate the design, and with it the era?) and against that season's "
        "response-cluster labels (does the response partition already encode "
        "the factor?).\n",
    ]
    for season in sorted(families):
        token = _season_token(season)
        for substructure in families[season]:
            definition = substructure.definition
            lines.append(
                f"\n`{token}/{definition.directory}/` — {season_label(season)} "
                f"split by {definition.title} (source column "
                f"`{definition.source_column}`), "
                f"{substructure.distinct_level_count} levels over "
                f"{substructure.member_count} trajectories. "
                f"Ladder agreement ARI={substructure.ladder_agreement:.3f}; "
                "response-cluster agreement "
                f"ARI={substructure.season_cluster_agreement:.3f}.\n"
            )
            lines.append(f"  {definition.identity_caveat}.\n")
            for stratum in substructure.strata:
                partition = stratum.partition
                detail = (
                    f"{partition.cluster_count} sub-clusters, silhouette "
                    f"{partition.silhouette:.3f}"
                    if partition is not None
                    else f"not sub-clustered ({stratum.not_subclustered_reason})"
                )
                lines.append(
                    f"- `{factor_level_stem(definition, stratum.level)}` "
                    f"{definition.level_title(stratum.level)}, "
                    f"{len(stratum.trajectory_ids)} trajectories over "
                    f"{len(stratum.variety_names)} varieties and "
                    f"{stratum.distinct_ladder_count} ladders — {detail}.\n"
                )
            if substructure.minor_levels:
                lines.append(
                    f"- `minor_levels` — "
                    f"{len(substructure.minor_level_trajectory_ids)} trajectories "
                    f"over {len(substructure.minor_levels)} undersized levels.\n"
                )
            if substructure.unrecorded_trajectory_ids:
                lines.append(
                    f"- `unrecorded_level` — "
                    f"{len(substructure.unrecorded_trajectory_ids)} trajectories "
                    f"with no usable `{definition.source_column}` value.\n"
                )
    lines.append("\n")
    return "".join(lines)


def _write_readme(
    result: SeasonClusteringResult,
    summary: Mapping[str, Any],
    substructures: Mapping[str, Sequence[ClusterSubstructure]],
    season_ladder_structures: Mapping[str, ClusterSubstructure],
    destination: Path,
    factor_substructures: Mapping[str, Sequence[FactorSubstructure]] = {},
    *,
    write_cluster_figures: bool = False,
    write_subcluster_figures: bool = False,
    write_ladder_composition: bool = False,
    write_held_out_figures: bool = False,
) -> None:
    season_lines = []
    for season in result.seasons:
        token = _season_token(season)
        profile = result.profiles[season]
        partition = result.partitions.get(season)
        if partition is None:
            season_lines.append(
                f"- `{token}/`: {season_label(season)} — "
                f"{len(profile.trajectory_ids)} eligible trajectories, not clustered "
                f"({result.unclustered_seasons.get(season, 'no partition')})."
            )
            continue
        concordance = result.ladder_concordance[season]
        season_lines.append(
            f"- `{token}/`: {season_label(season)} — {len(profile.trajectory_ids)} "
            f"eligible trajectories in {partition.cluster_count} clusters "
            f"(silhouette {partition.silhouette:.3f}); "
            f"{len(profile.excluded_trajectory_ids)} held out; "
            f"cluster/ladder agreement ARI {concordance.adjusted_rand:.2f}."
        )

    destination.write_text(
        "# LTCCE season-stratified response clusters\n\n"
        "Season is held fixed and the ladder-invariant response features are "
        "clustered *within* each cropping season. The pooled products in "
        "`../by_trajectory/` are exposed to the LTCCE season/ladder/era "
        "confounding; these partitions are not, because season does not vary "
        "inside any of them.\n\n"
        f"{summary['cluster_eligible_trajectories']} of "
        f"{summary['total_trajectories']} trajectories are cluster-eligible, "
        f"spread over {len(summary['observed_seasons'])} seasons; "
        f"{summary['clustered_season_count']} seasons carry a partition.\n\n"
        "## Cluster numbering is per season\n\n"
        f"{summary['cluster_label_scope']}\n\n"
        "## Top-level files\n\n"
        "- `season_overview_panel.jpeg`: every season's cluster-eligible "
        "trajectories side by side on one shared frame — the same content as each "
        "season's `overview.jpeg`, arranged so the seasons can be compared "
        "directly.\n"
        "- `season_profile_comparison.jpeg`: the four clustering features compared "
        "across seasons, as context-weighted means with the standard error across "
        "experimental contexts. Three seasons are compared, never partitioned.\n"
        "- `season_cluster_assignments.csv`: every seasoned trajectory with its "
        "season, cluster, diagnostics, observed features, exclusion reason, and "
        "figure path.\n"
        "- `season_clustering_summary.json`: method, per-season diagnostics, "
        "cluster-by-ladder contingency tables, centroids, and caveats.\n\n"
        "## Season folders\n\n"
        + "\n".join(season_lines)
        + "\n\nEach season folder carries its own `README.md`, which explains "
        "the `0/65/130/195` applied-N ladder notation used throughout the "
        "figures and tabulates that season's ladders, steps, and eras. Beyond "
        "that it holds `overview.jpeg` (all its eligible trajectories), "
        + (
            "`cluster_<i>.jpeg` (one per within-season cluster), "
            if write_cluster_figures
            else ""
        )
        + (
            "`ladder_composition.jpeg` (how the applied-N ladders distribute "
            "across that season's clusters), "
            if write_ladder_composition
            else ""
        )
        + (
            "and `held_out_<reason>.jpeg` for each eligibility exclusion.\n\n"
            if write_held_out_figures
            else "and, for the seasons that carry them, the `by_*/` "
            "decomposition folders. Excluded trajectories are not drawn; they "
            "are carried in `season_cluster_assignments.csv` with their "
            "`exclusion_reason`.\n\n"
        )
        + _nested_readme_section(
            substructures,
            season_ladder_structures,
            write_subcluster_figures=write_subcluster_figures,
        )
        + _factor_readme_section(
            factor_substructures,
            write_subcluster_figures=write_subcluster_figures,
        )
        + "## Does the partition just recover the N ladder?\n\n"
        "Stratifying by season removes the season contrast but not the applied-N "
        "ladder, which is still unevenly represented inside each season. Every "
        "season therefore reports the adjusted Rand index between its cluster "
        "labels and its ladder labels, on the figures and in the JSON summary. "
        f"At or above ARI={LADDER_DRIVEN_ADJUSTED_RAND:.2f} the partition is "
        "disclosed as substantially re-deriving the experimental design; a single "
        f"cluster at or above {100 * LADDER_DOMINATED_CLUSTER_SHARE:.0f}% on one "
        "ladder is called out on its own figure.\n\n"
        "## Interpretation boundary\n\n"
        "A within-season cluster is a band of observed trajectories that resemble "
        "one another on four descriptive features. It is not a response class, a "
        "recommendation domain, or a fitted curve, and the separation reported on "
        "each figure is what licenses how hard it can be read.\n\n"
        f"{summary['cross_season_caveat']} These are descriptive, exploratory "
        "views outside the governed analysis inventory (ANA-11), and no curve is "
        "fitted anywhere in this product. The full caveat text is carried in "
        "`season_clustering_summary.json`.\n",
        encoding="utf-8",
    )


def _carry_unmanaged_directories(staging: Path, destination: Path) -> list[str]:
    """Copy directories the run did not build into staging, and name them.

    This generator replaces `by_season/` as one snapshot, so anything under the
    old destination that this run did not write is destroyed by the swap. A
    whole directory absent from staging is by construction not this run's
    output — an operator's `archive/`, or `ds/by_variety/ir8_by_planting_year/`,
    which a different generator owns — and losing it to a figure rebuild has
    already cost real work.

    Directories only, deliberately. A *file* missing from staging inside a
    directory the run did build is either an output a `--write-*` flag has since
    turned off or one the operator deleted, and in both cases it should stay
    gone; carrying files across would resurrect exactly what those flags exist
    to suppress.
    """

    if not destination.is_dir() or destination.is_symlink():
        return []
    carried: list[str] = []
    for source in sorted(path for path in destination.rglob("*") if path.is_dir()):
        if source.is_symlink():
            continue
        relative = source.relative_to(destination)
        if (staging / relative).exists():
            continue
        # A parent already carried takes its whole subtree with it.
        if any(str(relative).startswith(f"{name}{os.sep}") for name in carried):
            continue
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        carried.append(str(relative))
    return carried


def _promote(staging: Path, destination: Path) -> list[str]:
    """Swap a fully built staging directory into place.

    Returns the unmanaged directories carried across, for the caller to report.
    """

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    carried = _carry_unmanaged_directories(staging, destination)
    backup = destination.with_name(f".{destination.name}.backup.{uuid.uuid4().hex}")
    if destination.exists():
        if not destination.is_dir() or destination.is_symlink():
            raise RuntimeError("Season-cluster destination is not a plain directory")
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)
    return carried


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--min-season-size",
        type=int,
        default=MIN_SEASON_SIZE,
        help="Smallest season that is clustered rather than described only",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=8,
        help="Upper bound of the silhouette search for k, within each season",
    )
    parser.add_argument(
        "--subcluster-seasons",
        nargs="*",
        default=["DS"],
        metavar="SEASON",
        help=(
            "Seasons decomposed by applied-N ladder over the whole season "
            "(by_applied_n/) and by recorded factor; 'all' for every season, "
            "'none' to skip (default: DS)"
        ),
    )
    parser.add_argument(
        "--write-ladder-composition",
        action="store_true",
        help=(
            "Also write <season>/ladder_composition.jpeg. Off by default: the "
            "cluster-by-ladder contingency it draws is carried in the season "
            "README and as a table in the summary JSON."
        ),
    )
    parser.add_argument(
        "--write-held-out-figures",
        action="store_true",
        help=(
            "Also write one <season>/held_out_<reason>.jpeg per eligibility "
            "exclusion. Off by default: the excluded trajectories are still "
            "carried in the ledger with their exclusion_reason, and counted in "
            "the season README and the summary JSON."
        ),
    )
    parser.add_argument(
        "--write-subcluster-figures",
        action="store_true",
        help=(
            "Also write one <stratum>_cluster_<m>.jpeg per inner sub-cluster in "
            "by_applied_n/ and every by_<factor>/ folder. Off by default: the "
            "sub-partitions are still computed, and their k, silhouette and "
            "dominant axis are reported on each comparison panel, in the season "
            "README and in the summary JSON."
        ),
    )
    parser.add_argument(
        "--write-cluster-figures",
        action="store_true",
        help=(
            "Also write one cluster_<i>.jpeg per within-season cluster. Off by "
            "default: the by_applied_n/, by_design/ and by_variety_code/ "
            "decompositions carry the season, and the partition's own "
            "diagnostics remain in ladder_composition.jpeg and the summary."
        ),
    )
    parser.add_argument(
        "--cluster-ladder-seasons",
        nargs="*",
        default=["none"],
        metavar="SEASON",
        help=(
            "Seasons whose *individual* response clusters are additionally "
            "decomposed by applied-N ladder into cluster_<i>/ folders; 'all' for "
            "every season, 'none' to skip (default: none)"
        ),
    )
    parser.add_argument(
        "--strata",
        nargs="*",
        default=[
            FACTOR_VARIETY_CODE,
            FACTOR_DESIGN,
            FACTOR_VARIETY,
            FACTOR_PLANTING_YEAR,
        ],
        choices=[*FACTOR_DEFINITIONS, "all", "none"],
        metavar="FACTOR",
        help=(
            "Recorded experimental factors each sub-clustered season is also "
            "decomposed by; 'all' for every factor, 'none' to skip. 'varcode' is "
            "the LTCCE VarCode plot slot, not a genotype; 'variety' is the "
            "genotype designation; 'planting_year' bands the recorded year into "
            "decades and adds the per-year trend panel "
            "(default: varcode design variety planting_year)"
        ),
    )
    parser.add_argument(
        "--min-factor-stratum",
        type=int,
        default=MIN_FACTOR_STRATUM,
        help="Smallest factor level drawn as its own figure rather than pooled",
    )
    parser.add_argument(
        "--separate-notes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "In by_planting_year/, write self-explaining plots under annotated/ "
            "and plates-only `<name>_figure.jpeg` files under figure_only/ with "
            f"one {SEPARATED_NOTES_FILENAME} carrying their prose. "
            "`--no-separate-notes` suppresses the figure_only family."
        ),
    )
    parser.add_argument(
        "--skip-composition-factors",
        nargs="*",
        default=[FACTOR_PLANTING_YEAR, FACTOR_VARIETY],
        choices=[*FACTOR_DEFINITIONS, "none"],
        metavar="FACTOR",
        help=(
            "Factors whose <factor>_composition.jpeg is not written. Defaults to "
            "the two whose composition figure is carried by the level-comparison "
            "panel and the README instead; 'none' writes all of them."
        ),
    )
    args = parser.parse_args()
    if "none" in args.skip_composition_factors:
        args.skip_composition_factors = []
    return args


def _requested_factors(requested: Sequence[str]) -> tuple[str, ...]:
    tokens = [token.strip().casefold() for token in requested if token.strip()]
    if not tokens or "none" in tokens:
        return ()
    if "all" in tokens:
        return tuple(FACTOR_DEFINITIONS)
    return tuple(factor for factor in FACTOR_DEFINITIONS if factor in tokens)


def _requested_substructure_seasons(
    requested: Sequence[str],
    available: Sequence[str],
) -> tuple[str, ...]:
    tokens = [token.strip() for token in requested if token.strip()]
    lowered = {token.casefold() for token in tokens}
    if not tokens or "none" in lowered:
        return ()
    if "all" in lowered:
        return tuple(available)
    known = {season.casefold(): season for season in available}
    unknown = sorted(token for token in tokens if token.casefold() not in known)
    if unknown:
        raise ValueError(
            f"Unknown season(s) {unknown}; this source carries {list(available)}"
        )
    return tuple(
        season for season in available if season.casefold() in lowered
    )


def main() -> int:
    args = _parse_args()
    source_path, encoding = _load_source_spec(args.config)

    overlay = read_source_dataset_overlay(source_path, SOURCE_NAME, encoding=encoding)
    contexts = read_ltcce_contexts(source_path, encoding=encoding)
    result = build_season_clustering(
        overlay,
        contexts,
        minimum_season_size=args.min_season_size,
        maximum_clusters=args.max_clusters,
    )

    # Taken from the parent overlay before any season subsetting, so the DS
    # response magnitude reads against the wet seasons rather than autoscaling.
    limits = _shared_axis_limits(overlay)

    destination = args.output_dir.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.staging.{uuid.uuid4().hex}")
    staging.mkdir()
    substructure_seasons = _requested_substructure_seasons(
        args.subcluster_seasons,
        [season for season in result.seasons if season in result.partitions],
    )
    # Decomposing each response cluster separately is a third nesting level and
    # is off by default: the season-level by_applied_n/ view below answers the
    # same question over the whole season, without multiplying figures.
    cluster_ladder_seasons = _requested_substructure_seasons(
        args.cluster_ladder_seasons,
        [season for season in result.seasons if season in result.partitions],
    )
    substructures = {
        season: build_season_substructures(result, season)
        for season in cluster_ladder_seasons
    }
    # The same decomposition over the whole season, before any response-type
    # partition is imposed: what does the applied-N design separate on its own?
    season_ladder_structures = {
        season: build_season_ladder_substructure(result, season)
        for season in substructure_seasons
    }
    # A third axis: the same season split by a recorded experimental factor
    # rather than by the fertilizer support it was grown on.
    requested_factors = _requested_factors(args.strata)
    factor_substructures = {
        season: tuple(
            build_factor_substructure(
                result,
                season,
                factor,
                minimum_stratum=args.min_factor_stratum,
            )
            for factor in requested_factors
        )
        for season in substructure_seasons
    }

    carried: list[str] = []
    try:
        written: list[str] = []
        figure_by_trajectory: dict[str, str] = {}
        subfigure_by_trajectory: dict[str, str] = {}
        season_ladder_figures: dict[str, str] = {}
        factor_figures: dict[str, dict[str, str]] = collections.defaultdict(dict)
        for season in result.seasons:
            overview_relative = _write_season_overview(
                result, overlay, season, staging, limits
            )
            written.append(overview_relative)
            # Baseline figure for every eligible trajectory, so the ledger still
            # resolves when the per-cluster figures are not written. Anything
            # more specific written below overwrites it.
            for trajectory_id in result.profiles[season].trajectory_ids:
                figure_by_trajectory[trajectory_id] = overview_relative
            if season in result.partitions:
                if args.write_cluster_figures:
                    cluster_files, cluster_figures = _write_season_cluster_views(
                        result, overlay, season, staging, limits
                    )
                    written += cluster_files
                    figure_by_trajectory.update(cluster_figures)
                if args.write_ladder_composition:
                    composition_relative = (
                        Path(_season_token(season)) / "ladder_composition.jpeg"
                    )
                    _write_ladder_composition(
                        result, season, staging / composition_relative
                    )
                    written.append(composition_relative.as_posix())
                for substructure in substructures.get(season, ()):
                    nested_files, nested_figures = _write_substructure_views(
                        result,
                        overlay,
                        substructure,
                        staging,
                        limits,
                        write_subcluster_figures=args.write_subcluster_figures,
                    )
                    written += nested_files
                    subfigure_by_trajectory.update(nested_figures)
                season_ladder = season_ladder_structures.get(season)
                if season_ladder is not None:
                    ladder_files, ladder_figures = _write_substructure_views(
                        result,
                        overlay,
                        season_ladder,
                        staging,
                        limits,
                        write_subcluster_figures=args.write_subcluster_figures,
                    )
                    written += ladder_files
                    season_ladder_figures.update(ladder_figures)
                for factor_substructure in factor_substructures.get(season, ()):
                    factor_files, level_figures = _write_factor_stratum_views(
                        result,
                        overlay,
                        factor_substructure,
                        staging,
                        limits,
                        write_subcluster_figures=args.write_subcluster_figures,
                        separate_notes=(
                            args.separate_notes
                            and factor_substructure.factor == FACTOR_PLANTING_YEAR
                        ),
                        write_composition=(
                            factor_substructure.factor
                            not in args.skip_composition_factors
                        ),
                    )
                    written += factor_files
                    factor_figures[factor_substructure.factor].update(level_figures)
            if args.write_held_out_figures:
                held_out_files, held_out_figures = _write_held_out_views(
                    result, overlay, season, staging, limits
                )
                written += held_out_files
                figure_by_trajectory.update(held_out_figures)
            season_readme = Path(_season_token(season)) / "README.md"
            _write_season_readme(
                result,
                season,
                substructures.get(season, ()),
                staging / season_readme,
                factor_substructures.get(season, ()),
                season_ladder_structures.get(season),
                args.write_cluster_figures,
                args.write_ladder_composition,
                args.write_held_out_figures,
            )
            written.append(season_readme.as_posix())

        _write_season_overview_panel(
            result, overlay, staging / "season_overview_panel.jpeg", limits
        )
        written.append("season_overview_panel.jpeg")
        _write_season_profile_comparison(
            result, staging / "season_profile_comparison.jpeg"
        )
        written.append("season_profile_comparison.jpeg")
        _write_ledger(
            result,
            staging / "season_cluster_assignments.csv",
            figure_by_trajectory,
            _substratum_ledger_rows(substructures, subfigure_by_trajectory),
            _season_ladder_ledger_rows(
                season_ladder_structures, season_ladder_figures
            ),
            _factor_ledger_rows(factor_substructures, factor_figures),
        )
        summary = _write_summary(
            result,
            overlay,
            staging / "season_clustering_summary.json",
            limits,
            figure_by_trajectory,
            substructures,
            season_ladder_structures,
            factor_substructures,
            minimum_season_size=args.min_season_size,
            maximum_clusters=args.max_clusters,
        )
        _write_readme(
            result,
            summary,
            substructures,
            season_ladder_structures,
            staging / "README.md",
            factor_substructures,
            write_cluster_figures=args.write_cluster_figures,
            write_subcluster_figures=args.write_subcluster_figures,
            write_ladder_composition=args.write_ladder_composition,
            write_held_out_figures=args.write_held_out_figures,
        )
        carried = _promote(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    if carried:
        noun = "directory" if len(carried) == 1 else "directories"
        print(
            f"Carried across {len(carried)} unmanaged {noun}: {', '.join(carried)}"
        )
    print(
        f"{SOURCE_NAME}: {summary['cluster_eligible_trajectories']} of "
        f"{summary['total_trajectories']} trajectories clustered within "
        f"{summary['clustered_season_count']} of {len(result.seasons)} seasons; "
        f"{summary['held_out_trajectories']} held out "
        f"({summary['held_out_by_reason']})"
    )
    for season in result.seasons:
        partition = result.partitions.get(season)
        if partition is None:
            print(
                f"  {season:4s} n={len(result.profiles[season].trajectory_ids):4d} "
                f"not clustered ({result.unclustered_seasons.get(season, '')})"
            )
            continue
        concordance = result.ladder_concordance[season]
        print(
            f"  {season:4s} n={partition.member_count:4d} k={partition.cluster_count} "
            f"silhouette={partition.silhouette:.3f} "
            f"seed-ARI={partition.assignment_stability:.2f} "
            f"ladder-ARI={concordance.adjusted_rand:.3f} "
            f"{'(ladder-driven)' if concordance.is_ladder_driven else '(not ladder-driven)'}"
        )
        for substructure in (
            *substructures.get(season, ()),
            *(
                (season_ladder_structures[season],)
                if season in season_ladder_structures
                else ()
            ),
        ):
            for index, substratum in enumerate(substructure.substrata, start=1):
                inner = substratum.partition
                detail = (
                    f"k={inner.cluster_count} silhouette={inner.silhouette:.3f} "
                    f"axis={inner.dominant_axis}"
                    if inner is not None
                    else f"not sub-clustered ({substratum.not_subclustered_reason})"
                )
                print(
                    f"       {substructure.scope_text:16s} sub_{index} "
                    f"{ladder_text(substratum.ladder):16s} "
                    f"n={len(substratum.trajectory_ids):4d} {detail}"
                )
    print(f"Wrote {len(written)} figures + ledger + summary under {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
