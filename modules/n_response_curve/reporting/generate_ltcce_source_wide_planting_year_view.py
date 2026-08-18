#!/usr/bin/env python3
"""Build LTCCE source-wide planting-decade overlays from whole trajectories.

This view keeps every finite observation in whole replicate-specific trajectories and
writes only source-wide recorded decade partitions (1960s–2010s) with no pooled
curve or pooled fit.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import textwrap
import tomllib
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.reporting.response_curve_clusters import (  # noqa: E402
    TrajectoryContext,
    read_ltcce_contexts,
    subset_overlay,
)
from n_response_curve.reporting.source_dataset_overlays import (  # noqa: E402
    _adaptive_style,
    SourceDatasetOverlay,
    read_source_dataset_overlay,
)

SOURCE_NAME = "ltcce"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "WF/04_Response_Curves/z_n_response_full/overlay/source_dataset"
    / "ltcce/clusters/by_planting_year"
)
KNOWN_DECADES = ("1960s", "1970s", "1980s", "1990s", "2000s", "2010s")
SEASONS = ("DS", "EWS", "LWS")
SEASON_MARKERS = {"DS": "o", "EWS": "s", "LWS": "^"}
SEASON_COLOURS = {"DS": "C0", "EWS": "C1", "LWS": "C2"}
TREATMENT_COLOURS = {"mineral N rate": "C0", "zero N": "C1"}
SUMMARY_FILENAME = "planting_decade_summary.json"
README_FILENAME = "README.md"
KNOWN_ARTIFACTS = frozenset(
    {
        "1960s.jpeg",
        "1970s.jpeg",
        "1980s.jpeg",
        "1990s.jpeg",
        "2000s.jpeg",
        "2010s.jpeg",
        "level_comparison.jpeg",
        "decades_and_annual_counts.jpeg",
        SUMMARY_FILENAME,
        README_FILENAME,
    }
)


@dataclass(frozen=True)
class DecadeProfile:
    decade: str
    trajectory_ids: tuple[str, ...]
    trajectory_count: int
    observation_count: int
    season_composition: tuple[tuple[str, int], ...]
    year_range: tuple[int, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_spec(config_path: Path) -> tuple[Path, str]:
    with Path(config_path).open("rb") as handle:
        config = tomllib.load(handle)
    sources = config.get("sources")
    section = sources.get(SOURCE_NAME) if isinstance(sources, dict) else None
    if not isinstance(section, dict):
        raise ValueError(f"The configuration is missing [{SOURCE_NAME}] under [sources]")
    data_path = section.get("data_path")
    encoding = section.get("encoding")
    if not isinstance(data_path, str) or not data_path.strip():
        raise ValueError(f"[sources.{SOURCE_NAME}].data_path must be a non-empty string")
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError(f"[sources.{SOURCE_NAME}].encoding must be a non-empty string")
    source_path = Path(data_path)
    if not source_path.is_absolute():
        source_path = Path(config_path).resolve().parent / source_path
    return source_path.resolve(), encoding


def _normalize_season(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        return ""
    mapping = {
        "DRY": "DS",
        "DRY SEASON": "DS",
        "DS": "DS",
        "EWS": "EWS",
        "WS": "EWS",
        "WET": "EWS",
        "WET SEASON": "EWS",
        "LWS": "LWS",
    }
    return mapping.get(normalized, normalized)


def _decade_label(year: int) -> str:
    return f"{(year // 10) * 10}s"


def _point_encoding(treatment_class: str, season: str) -> dict[str, str]:
    if treatment_class not in TREATMENT_COLOURS:
        raise ValueError(f"Unsupported LTCCE treatment class: {treatment_class}")
    if season not in SEASON_MARKERS:
        raise ValueError(f"Unsupported LTCCE season: {season}")
    return {
        "color": TREATMENT_COLOURS[treatment_class],
        "marker": SEASON_MARKERS[season],
    }


def _encoding_legend_handles() -> tuple[tuple[Line2D, ...], tuple[str, ...]]:
    treatment_labels = tuple(TREATMENT_COLOURS)
    season_labels = tuple(SEASONS)
    connector_label = "within-trajectory connecting lines (visual aid; not a fit)"
    handles = (
        *(
            Line2D(
                [],
                [],
                color=colour,
                marker="o",
                linestyle="none",
                markersize=5,
            )
            for colour in TREATMENT_COLOURS.values()
        ),
        *(
            Line2D(
                [],
                [],
                color="0.35",
                marker=SEASON_MARKERS[season],
                linestyle="none",
                markersize=5,
            )
            for season in SEASONS
        ),
        Line2D([], [], color="0.6", linewidth=0.9),
    )
    return handles, (*treatment_labels, *season_labels, connector_label)


def _require_complete_contexts(
    overlay: SourceDatasetOverlay,
    contexts: Mapping[str, TrajectoryContext],
) -> dict[str, TrajectoryContext]:
    complete: dict[str, TrajectoryContext] = {}
    missing: list[str] = []
    invalid: list[str] = []
    for trajectory in overlay.trajectories:
        context = contexts.get(trajectory.trajectory_id)
        if context is None:
            missing.append(trajectory.trajectory_id)
            continue
        season = _normalize_season(context.season)
        if not season:
            invalid.append(f"{trajectory.trajectory_id}: season blank")
            continue
        if season not in SEASONS:
            invalid.append(f"{trajectory.trajectory_id}: season={context.season!r}")
            continue
        if context.year <= 0:
            invalid.append(f"{trajectory.trajectory_id}: year={context.year}")
            continue
        complete[trajectory.trajectory_id] = replace(context, season=season)

    if missing:
        preview = ", ".join(missing[:6])
        raise ValueError(
            f"{len(missing)} trajectories are missing context rows for "
            f"{SOURCE_NAME} (example: {preview})"
        )
    if invalid:
        preview = ", ".join(invalid[:6])
        raise ValueError(
            "One or more contexts are invalid for complete-context partitioning: "
            f"{preview}"
        )
    return complete


def _build_decade_profile(
    overlay: SourceDatasetOverlay,
    context_by_id: Mapping[str, TrajectoryContext],
) -> tuple[DecadeProfile, ...]:
    if not overlay.trajectories:
        raise ValueError("LTCCE source-wide overlay is empty")

    observation_counts = {
        trajectory.trajectory_id: len(trajectory.observations)
        for trajectory in overlay.trajectories
    }
    trajectory_ids = [trajectory.trajectory_id for trajectory in overlay.trajectories]

    if len(context_by_id) != len(trajectory_ids):
        raise ValueError(
            "Context coverage is incomplete for complete-context partitioning"
        )

    groups: dict[str, list[str]] = {decade: [] for decade in KNOWN_DECADES}
    years: dict[str, list[int]] = {decade: [] for decade in KNOWN_DECADES}
    seasons: dict[str, collections.Counter[str]] = {
        decade: collections.Counter() for decade in KNOWN_DECADES
    }

    for trajectory_id in trajectory_ids:
        context = context_by_id.get(trajectory_id)
        if context is None:
            raise ValueError(f"Missing LTCCE context for trajectory {trajectory_id}")
        decade = _decade_label(context.year)
        if decade not in groups:
            raise ValueError(
                f"Unexpected planting decade {decade} for trajectory {trajectory_id}"
            )
        groups[decade].append(trajectory_id)
        years[decade].append(context.year)
        seasons[decade][context.season] += 1

    profiles: list[DecadeProfile] = []
    for decade in KNOWN_DECADES:
        ids = tuple(sorted(groups[decade]))
        if not ids:
            raise ValueError(f"LTCCE source missing expected decade {decade}")
        decade_years = sorted(years[decade])
        profile_year_range = (decade_years[0], decade_years[-1]) if decade_years else (0, 0)
        profiles.append(
            DecadeProfile(
                decade=decade,
                trajectory_ids=ids,
                trajectory_count=len(ids),
                observation_count=sum(
                    observation_counts[trajectory_id] for trajectory_id in ids
                ),
                season_composition=tuple(
                    (season, int(seasons[decade][season]))
                    for season in SEASONS
                ),
                year_range=profile_year_range,
            )
        )

    covered = set(trajectory_id for decade in groups.values() for trajectory_id in decade)
    if len(covered) != len(trajectory_ids):
        raise ValueError(
            "Decade partition is not exhaustive for source-wide trajectories "
            f"(covered={len(covered)} expected={len(trajectory_ids)})"
        )
    return tuple(profiles)


def _annual_trajectory_counts(
    overlay: SourceDatasetOverlay,
    context_by_id: Mapping[str, TrajectoryContext],
) -> tuple[dict[str, int], ...]:
    if not overlay.trajectories:
        raise ValueError("LTCCE source-wide overlay has no trajectories")

    per_year: dict[int, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    years: list[int] = []
    for trajectory in overlay.trajectories:
        context = context_by_id[trajectory.trajectory_id]
        per_year[context.year][context.season] += 1
        years.append(context.year)

    start = min(years)
    end = max(years)
    output = []
    for year in range(start, end + 1):
        counter = per_year.get(year, collections.Counter())
        ds = int(counter.get("DS", 0))
        ews = int(counter.get("EWS", 0))
        lws = int(counter.get("LWS", 0))
        output.append(
            {
                "year": year,
                "DS": ds,
                "EWS": ews,
                "LWS": lws,
                "total": ds + ews + lws,
            }
        )
    return tuple(output)


def _draw_overlay_panel(
    axes,
    overlay: SourceDatasetOverlay,
    context_by_id: Mapping[str, TrajectoryContext],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    title: str,
    season_composition: Sequence[tuple[str, int]],
    show_season_composition: bool = True,
) -> None:
    marker_size, marker_alpha, line_alpha = _adaptive_style(
        overlay.summary.finite_observation_count,
        overlay.summary.trajectory_count,
    )

    for trajectory in overlay.trajectories:
        xs = [observation.n_rate_kg_ha for observation in trajectory.observations]
        ys = [observation.yield_t_ha for observation in trajectory.observations]
        if len(xs) > 1:
            axes.plot(
                xs,
                ys,
                color="0.6",
                alpha=line_alpha,
                linewidth=0.9,
                zorder=1,
            )

    for season in SEASONS:
        for treatment_class in TREATMENT_COLOURS:
            xs: list[float] = []
            ys: list[float] = []
            for trajectory in overlay.trajectories:
                context = context_by_id[trajectory.trajectory_id]
                if context.season != season:
                    continue
                for observation in trajectory.observations:
                    if observation.treatment_class == treatment_class:
                        xs.append(observation.n_rate_kg_ha)
                        ys.append(observation.yield_t_ha)
            if xs:
                encoding = _point_encoding(treatment_class, season)
                axes.scatter(
                    xs,
                    ys,
                    s=marker_size,
                    alpha=marker_alpha,
                    marker=encoding["marker"],
                    color=encoding["color"],
                    zorder=2,
                )

    axes.set_title(title)
    axes.set_xlim(x_limits)
    axes.set_ylim(y_limits)
    axes.set_xlabel("Applied N (kg N/ha)")
    axes.set_ylabel("Grain yield (t/ha)")
    axes.grid(alpha=0.2, linewidth=0.6)

    if show_season_composition:
        labels = ", ".join(f"{season}={count}" for season, count in season_composition)
        axes.text(
            0.01,
            0.98,
            labels,
            ha="left",
            va="top",
            fontsize=8,
            color="0.3",
            transform=axes.transAxes,
        )


def _build_decade_figure(
    profile: DecadeProfile,
    overlay: SourceDatasetOverlay,
    context_by_id: Mapping[str, TrajectoryContext],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> Figure:
    subset = subset_overlay(overlay, profile.trajectory_ids)
    figure, axes = plt.subplots(figsize=(10.0, 8.0))
    _draw_overlay_panel(
        axes,
        subset,
        context_by_id,
        x_limits=x_limits,
        y_limits=y_limits,
        title=(
            f"{profile.decade}: {profile.trajectory_count} trajectories, "
            f"{profile.observation_count} observations"
        ),
        season_composition=profile.season_composition,
    )
    handles, labels = _encoding_legend_handles()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    footer = figure.text(
        0.5,
        0.012,
        (
            "all-season source-wide recorded decade strata, not statistical clusters; "
            "DS, EWS, and LWS are retained; no pooled response curve or fit."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="0.45",
    )
    footer.set_gid("figure-footer")
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.21, top=0.94)
    return figure


def _write_decade_figure(
    profile: DecadeProfile,
    overlay: SourceDatasetOverlay,
    context_by_id: Mapping[str, TrajectoryContext],
    destination: Path,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    figure = _build_decade_figure(
        profile,
        overlay,
        context_by_id,
        x_limits=x_limits,
        y_limits=y_limits,
    )
    try:
        figure.savefig(destination, format="jpeg", dpi=150)
    finally:
        plt.close(figure)


def _build_level_comparison(
    profiles: Sequence[DecadeProfile],
    overlay: SourceDatasetOverlay,
    context_by_id: Mapping[str, TrajectoryContext],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> Figure:
    figure, axes_grid = plt.subplots(
        2,
        3,
        figsize=(18.0, 10.0),
        sharex=True,
        sharey=True,
    )
    axes_list = list(axes_grid.ravel())
    for index, (axes, profile) in enumerate(
        zip(axes_list, sorted(profiles, key=lambda item: item.decade), strict=True)
    ):
        subset = subset_overlay(overlay, profile.trajectory_ids)
        _draw_overlay_panel(
            axes,
            subset,
            context_by_id,
            x_limits=x_limits,
            y_limits=y_limits,
            title=(
                f"{profile.decade} — {profile.trajectory_count} trajectories, "
                f"{profile.observation_count} observations"
            ),
            season_composition=profile.season_composition,
        )
        if index < 3:
            axes.set_xlabel("")
        if index % 3:
            axes.set_ylabel("")

    figure.suptitle(
        "All-season LTCCE source-wide response trajectories by planting decade",
        fontsize=14,
    )
    handles, labels = _encoding_legend_handles()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.052),
        ncol=6,
        fontsize=8,
        frameon=False,
    )
    footer = figure.text(
        0.5,
        0.012,
        (
            "all-season source-wide recorded decade strata, not statistical clusters; "
            "DS, EWS, and LWS are retained; no pooled response curve or fit."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="0.45",
    )
    footer.set_gid("figure-footer")
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.16,
        top=0.91,
        hspace=0.25,
        wspace=0.12,
    )
    return figure


def _write_level_comparison(
    profiles: Sequence[DecadeProfile],
    overlay: SourceDatasetOverlay,
    context_by_id: Mapping[str, TrajectoryContext],
    destination: Path,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    figure = _build_level_comparison(
        profiles,
        overlay,
        context_by_id,
        x_limits=x_limits,
        y_limits=y_limits,
    )
    try:
        figure.savefig(destination, format="jpeg", dpi=150)
    finally:
        plt.close(figure)


def _build_decades_and_annual_counts(
    profiles: Sequence[DecadeProfile],
    overlay: SourceDatasetOverlay,
    context_by_id: Mapping[str, TrajectoryContext],
    annual_counts: Sequence[Mapping[str, int]],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> Figure:
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    x_pad = max(2.0, 0.05 * (x_max - x_min))
    y_pad = max(0.2, 0.05 * (y_max - y_min))

    # Mirror the reference DS sheet's hand-placed calendar layout: each inset is
    # physically inside the decade band of the host year axis rather than merely
    # arranged in an unrelated row above it.
    sheet_width = 34.0
    sheet_left = 1.30
    sheet_right = 0.35
    sheet_bottom = 0.25
    trend_legend_height = 0.72
    trend_xaxis_height = 0.70
    host_height = 11.60
    host_trend_fraction = 0.42
    inset_bottom_fraction = 0.53
    inset_top_fraction = 0.94
    inset_legend_height = 0.54
    title_height = 1.65
    panel_gutter = 0.10
    sheet_height = (
        sheet_bottom
        + trend_legend_height
        + trend_xaxis_height
        + host_height
        + inset_legend_height
        + title_height
    )
    figure = plt.figure(figsize=(sheet_width, sheet_height))

    def box(
        left_inches: float,
        bottom_inches: float,
        width_inches: float,
        height_inches: float,
    ) -> tuple[float, float, float, float]:
        return (
            left_inches / sheet_width,
            bottom_inches / sheet_height,
            width_inches / sheet_width,
            height_inches / sheet_height,
        )

    plot_width = sheet_width - sheet_left - sheet_right
    host_bottom = sheet_bottom + trend_legend_height + trend_xaxis_height
    inset_bottom = host_bottom + inset_bottom_fraction * host_height
    inset_height = (inset_top_fraction - inset_bottom_fraction) * host_height
    band_width = plot_width / len(KNOWN_DECADES)
    axes_by_decade = []
    for index in range(len(KNOWN_DECADES)):
        axis = figure.add_axes(
            box(
                sheet_left + index * band_width + panel_gutter / 2,
                inset_bottom,
                band_width - panel_gutter,
                inset_height,
            )
        )
        axis.set_gid(f"decade-inset-{KNOWN_DECADES[index]}")
        axis.set_facecolor("white")
        axis.set_zorder(2)
        for spine in axis.spines.values():
            spine.set_color("0.55")
        axes_by_decade.append(axis)

    for index, profile in enumerate(sorted(profiles, key=lambda item: item.decade)):
        subset = subset_overlay(overlay, profile.trajectory_ids)
        composition = dict(profile.season_composition)
        _draw_overlay_panel(
            axes_by_decade[index],
            subset,
            context_by_id,
            x_limits=(x_min - x_pad, x_max + x_pad),
            y_limits=(y_min - y_pad, y_max + y_pad),
            title=profile.decade,
            season_composition=profile.season_composition,
            show_season_composition=False,
        )
        axes = axes_by_decade[index]
        axes.set_title(profile.decade, fontsize=20, fontweight="bold", pad=6)
        axes.set_xlabel("")
        axes.set_ylabel("")
        axes.set_xticks([0, 50, 100, 150, 200])
        axes.tick_params(labelsize=10)
        share = 100.0 * profile.trajectory_count / overlay.summary.trajectory_count
        readout = axes.text(
            0.98,
            0.02,
            (
                f"{profile.trajectory_count} trajectories ({share:.0f}%)\n"
                f"{profile.year_range[0]}-{profile.year_range[1]}\n"
                f"DS {composition['DS']} · EWS {composition['EWS']} · "
                f"LWS {composition['LWS']}\n"
                f"{profile.observation_count} observations"
            ),
            ha="right",
            va="bottom",
            fontsize=11,
            color="0.2",
            linespacing=1.35,
            transform=axes.transAxes,
            bbox={
                "boxstyle": "square,pad=0.22",
                "facecolor": "white",
                "edgecolor": "0.82",
                "alpha": 0.92,
                "linewidth": 0.8,
            },
            zorder=4,
        )
        readout.set_gid("decade-readout")

    for extra in axes_by_decade[len(profiles) :]:
        extra.set_visible(False)

    for index, axis in enumerate(axes_by_decade[: len(profiles)]):
        if index:
            axis.set_ylabel("")
            axis.tick_params(labelleft=False)

    frame_note = figure.text(
        (sheet_left + plot_width / 2) / sheet_width,
        (
            host_bottom
            + (inset_bottom_fraction - 0.05) * host_height
        )
        / sheet_height,
        (
            "insets: applied N (kg N/ha) across, grain yield (t/ha) up — "
            "one shared frame, ticks on the left-hand inset"
        ),
        ha="center",
        va="top",
        fontsize=14,
        color="0.35",
    )
    frame_note.set_gid("inset-frame-note")

    annual_axis = figure.add_axes(
        box(sheet_left, host_bottom, plot_width, host_height)
    )
    annual_axis.set_gid("annual-count-host")
    annual_axis.set_zorder(0)
    years = [int(record["year"]) for record in annual_counts]
    for index, decade in enumerate(KNOWN_DECADES):
        start = int(decade.rstrip("s"))
        if index % 2 == 0:
            annual_axis.axvspan(start - 0.5, start + 9.5, color="0.94", zorder=0)
        if index:
            annual_axis.axvline(
                start - 0.5,
                color="0.55",
                linewidth=0.9,
                linestyle="--",
                zorder=1,
            )
    for season in SEASONS:
        annual_axis.plot(
            years,
            [int(record[season]) for record in annual_counts],
            marker="o",
            linewidth=1.4,
            markersize=3.0,
            alpha=0.95,
            color=SEASON_COLOURS[season],
            label=season,
        )
    annual_axis.set_xlabel("Planting year")
    annual_axis.set_ylabel("Trajectory count", y=host_trend_fraction / 2)
    annual_axis.set_xlim(1959.5, 2019.5)
    count_ceiling = max(
        int(record[season])
        for record in annual_counts
        for season in SEASONS
    )
    annual_axis.set_ylim(-1.0, (count_ceiling + 2.0) / host_trend_fraction)
    annual_axis.set_yticks(range(0, count_ceiling + 1, 5))
    annual_axis.set_xticks(range(1960, 2020, 5))
    annual_axis.set_xticks(range(1960, 2020), minor=True)
    annual_axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    annual_axis.tick_params(labelsize=10)

    handles, labels = _encoding_legend_handles()
    figure.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=box(
            sheet_left,
            host_bottom + host_height,
            plot_width,
            inset_legend_height,
        ),
        bbox_transform=figure.transFigure,
        ncol=6,
        fontsize=10,
        frameon=False,
    )
    trend_handles, trend_labels = annual_axis.get_legend_handles_labels()
    figure.legend(
        trend_handles,
        trend_labels,
        loc="center",
        bbox_to_anchor=box(
            sheet_left,
            sheet_bottom,
            plot_width,
            trend_legend_height,
        ),
        bbox_transform=figure.transFigure,
        ncol=3,
        fontsize=10,
        frameon=False,
        title="annual trajectory count — colour and marker identify season",
        title_fontsize=10,
    )
    footer = figure.text(
        0.5,
        1.0 - 0.72 / sheet_height,
        (
            "exploratory source-wide diagnostic — recorded decade strata, not "
            "statistical clusters; DS, EWS, and LWS are retained; no pooled "
            "response curve or fit is shown"
        ),
        ha="center",
        va="top",
        fontsize=13,
        color="0.35",
    )
    footer.set_gid("figure-footer")
    headline = figure.suptitle(
        "source=ltcce — all seasons: each planting decade's response trajectories "
        "drawn on the stretch of the annual count record they came from",
        fontsize=23,
        y=1.0 - 0.16 / sheet_height,
    )
    headline.set_gid("sheet-headline")
    header_rule = Line2D(
        [sheet_left / sheet_width, (sheet_left + plot_width) / sheet_width],
        [
            (host_bottom + host_height + inset_legend_height + 0.08) / sheet_height,
            (host_bottom + host_height + inset_legend_height + 0.08) / sheet_height,
        ],
        transform=figure.transFigure,
        color="0.82",
        linewidth=1.0,
    )
    header_rule.set_gid("header-rule")
    figure.add_artist(header_rule)
    return figure


def _write_decades_and_annual_counts(
    profiles: Sequence[DecadeProfile],
    overlay: SourceDatasetOverlay,
    context_by_id: Mapping[str, TrajectoryContext],
    annual_counts: Sequence[Mapping[str, int]],
    destination: Path,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    figure = _build_decades_and_annual_counts(
        profiles,
        overlay,
        context_by_id,
        annual_counts,
        x_limits=x_limits,
        y_limits=y_limits,
    )
    try:
        figure.savefig(destination, format="jpeg", dpi=150)
    finally:
        plt.close(figure)


def _write_summary(
    overlay: SourceDatasetOverlay,
    source_path: Path,
    source_sha: str,
    profiles: Sequence[DecadeProfile],
    annual_counts: Sequence[Mapping[str, int]],
    destination: Path,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "artifact": "ltcce_source_wide_planting_year_view",
        "source_name": SOURCE_NAME,
        # LTCCE is restricted; the content hash is sufficient provenance and the
        # basename identifies the configured input without disclosing the host's
        # absolute filesystem/project layout in copied artifacts.
        "source_file": source_path.name,
        "source_sha256": source_sha,
        "artifact_files": sorted(KNOWN_ARTIFACTS),
        "population": {
            "trajectory_count": overlay.summary.trajectory_count,
            "finite_observation_count": overlay.summary.finite_observation_count,
            "year_range": [annual_counts[0]["year"], annual_counts[-1]["year"]],
            "decades": list(KNOWN_DECADES),
            "season_composition": {
                season: sum(
                    int(dict(profile.season_composition)[season]) for profile in profiles
                )
                for season in SEASONS
            },
        },
        "decade_strata": [
            {
                "decade": profile.decade,
                "trajectory_count": profile.trajectory_count,
                "observation_count": profile.observation_count,
                "season_composition": {
                    season: int(dict(profile.season_composition)[season])
                    for season in SEASONS
                },
                "year_range": list(profile.year_range),
            }
            for profile in profiles
        ],
        "annual_trajectory_counts": [
            {
                "year": int(record["year"]),
                "DS": int(record["DS"]),
                "EWS": int(record["EWS"]),
                "LWS": int(record["LWS"]),
                "total": int(record["total"]),
            }
            for record in annual_counts
        ],
        "disclosures": [
            "The six panels are all-season source-wide recorded strata, "
            "not a statistical cluster.",
            "DS, EWS, and LWS are included in every panel.",
            "No pooled response curve or pooled fit is shown.",
            "Decade differences are confounded by season composition, N ladder, "
            "variety, design, weather, and soil history.",
            (
                "Annual trajectory counts use full-population complete-context trajectories "
                "and do not filter by response-curve-eligibility criteria."
            ),
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _write_readme(
    source_path: Path,
    source_sha: str,
    overlay: SourceDatasetOverlay,
    profiles: Sequence[DecadeProfile],
    annual_counts: Sequence[Mapping[str, int]],
    destination: Path,
) -> None:
    per_decade = []
    for profile in profiles:
        comps = ", ".join(
            f"{season} {int(dict(profile.season_composition)[season])}"
            for season in SEASONS
        )
        per_decade.append(
            f"- {profile.decade}: {profile.trajectory_count} trajectories, "
            f"{profile.observation_count} finite observations ({comps})"
        )
    years = [int(record["year"]) for record in annual_counts]
    lines = [
        "# LTCCE source-wide planting-decade view",
        "",
        "This folder contains **all-season source-wide recorded decade strata**, not "
        "statistical clusters.",
        "DS, EWS, and LWS are retained.",
        "No pooled response curve or pooled fit is shown.",
        "",
        "## Files",
        "- `1960s.jpeg` through `2010s.jpeg`: source-wide overlay panels by decade.",
        "- `level_comparison.jpeg`: all six decades on a shared N/yield frame.",
        "- `decades_and_annual_counts.jpeg`: six decade overlays embedded in their "
        "calendar bands above the annual trajectory counts by season, following the "
        "reference DS sheet's shared-year-axis presentation.",
        "- `planting_decade_summary.json`: exact partition identity and yearly counts.",
        "- `README.md`: this note.",
        "",
        "## Population",
        f"- configured source file: `{source_path.name}`",
        f"- source SHA-256: `{source_sha}`",
        f"- trajectories: {overlay.summary.trajectory_count}",
        f"- finite observations: {overlay.summary.finite_observation_count}",
        f"- years: {years[0]}-{years[-1]}",
        "",
        "## Decade-by-decade composition",
        *per_decade,
        "",
        "## Disclosures",
        "- This is a complete source-wide partition by planting decade (1960s–2010s); every finite observation stays with its whole replicate-specific trajectory.",
        "- Decade differences are confounded by season composition, N ladder, variety, "
        "design, weather, and soil history.",
        "- Per-season composition for annual trajectories is DS/EWS/LWS.",
        "- Annual trends show full-population trajectory counts and do not assume pooled trendability.",
        "",
        "## Deterministic build policy",
        (
            textwrap.fill(
                "Fail-closed checks require one complete LTCCE context with valid year and "
                "nonblank season for every trajectory.",
                width=78,
            )
        ),
        (
            textwrap.fill(
                "No seasons are filtered and no trajectory is excluded after context checks.",
                width=78,
            )
        ),
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_owned_snapshot(path: Path) -> bool:
    """Return whether *path* is exactly a bundle this generator may replace."""

    if path.is_symlink() or not path.is_dir():
        return False
    entries = tuple(path.iterdir())
    if {entry.name for entry in entries} != KNOWN_ARTIFACTS:
        return False
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        return False
    try:
        summary = json.loads((path / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(summary, dict)
        and summary.get("artifact") == "ltcce_source_wide_planting_year_view"
        and summary.get("source_name") == SOURCE_NAME
    )


def _replace_with_snapshot(staging: Path, destination: Path) -> None:
    # `resolve()` would follow a destination symlink before we can reject it.
    # `absolute()` makes the sibling check deterministic without erasing that
    # security-relevant path identity.
    staging = staging.absolute()
    destination = destination.absolute()
    if staging == destination or staging.parent != destination.parent:
        raise RuntimeError("Snapshot staging must be a distinct sibling directory")

    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(
        f".{destination.name}.backup.{uuid.uuid4().hex}"
    )
    promotion_succeeded = False
    try:
        if destination.is_symlink():
            raise RuntimeError("Destination snapshot must be a plain directory")
        if destination.exists():
            if not _is_owned_snapshot(destination):
                raise RuntimeError(
                    "Destination is not an owned LTCCE planting-decade bundle"
                )
        if not _is_owned_snapshot(staging):
            raise RuntimeError("Staging snapshot is incomplete or unowned")
        if destination.exists():
            os.replace(destination, backup)
        os.replace(staging, destination)
        promotion_succeeded = True
    except BaseException:
        if backup.exists():
            displaced_promotion: Path | None = None
            if destination.exists() or destination.is_symlink():
                displaced_promotion = destination.with_name(
                    f".{destination.name}.failed-promotion.{uuid.uuid4().hex}"
                )
                os.replace(destination, displaced_promotion)
            try:
                os.replace(backup, destination)
            except BaseException:
                # Keep both the previous snapshot and any promoted candidate as
                # physical recovery material if rollback itself cannot finish.
                raise
            else:
                if displaced_promotion is not None:
                    if displaced_promotion.is_dir():
                        shutil.rmtree(displaced_promotion, ignore_errors=True)
                    elif displaced_promotion.exists() or displaced_promotion.is_symlink():
                        displaced_promotion.unlink(missing_ok=True)
        elif (destination.exists() or destination.is_symlink()) and not staging.exists():
            # There was no previous snapshot. A rename can take effect and still
            # be followed immediately by KeyboardInterrupt; move that candidate
            # back out of the canonical path before propagating the interruption.
            os.replace(destination, staging)
        raise
    finally:
        if promotion_succeeded and backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def _publish(
    overlay: SourceDatasetOverlay,
    source_path: Path,
    source_sha: str,
    profiles: Sequence[DecadeProfile],
    annual_counts: Sequence[Mapping[str, int]],
    context_by_id: Mapping[str, TrajectoryContext],
    destination: Path,
) -> None:
    destination = destination.resolve()
    if not destination.parent.is_dir():
        destination.parent.mkdir(parents=True, exist_ok=True)

    staging = destination.with_name(f".{destination.name}.staging.{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        x_limits = overlay.summary.n_rate_range_kg_ha
        y_limits = overlay.summary.yield_range_t_ha
        if x_limits is None or y_limits is None:
            raise ValueError("Overlay missing numeric range required for plotting")
        for profile in profiles:
            _write_decade_figure(
                profile,
                overlay,
                context_by_id,
                staging / f"{profile.decade}.jpeg",
                x_limits=x_limits,
                y_limits=y_limits,
            )

        _write_level_comparison(
            profiles,
            overlay,
            context_by_id,
            staging / "level_comparison.jpeg",
            x_limits=x_limits,
            y_limits=y_limits,
        )
        _write_decades_and_annual_counts(
            profiles,
            overlay,
            context_by_id,
            annual_counts,
            staging / "decades_and_annual_counts.jpeg",
            x_limits=x_limits,
            y_limits=y_limits,
        )
        _write_summary(
            overlay=overlay,
            source_path=source_path,
            source_sha=source_sha,
            profiles=profiles,
            annual_counts=annual_counts,
            destination=staging / SUMMARY_FILENAME,
        )
        _write_readme(
            source_path=source_path,
            source_sha=source_sha,
            overlay=overlay,
            profiles=profiles,
            annual_counts=annual_counts,
            destination=staging / README_FILENAME,
        )
        if _sha256(source_path) != source_sha:
            raise RuntimeError("LTCCE source changed while rendering outputs")
        _replace_with_snapshot(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "scriptCONFIG.toml",
        help="Path to scriptCONFIG.toml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory replaced atomically.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(list(argv) if argv is not None else None)

    source_path, encoding = _load_source_spec(args.config)
    source_sha = _sha256(source_path)
    overlay = read_source_dataset_overlay(source_path, SOURCE_NAME, encoding=encoding)
    contexts = read_ltcce_contexts(source_path, encoding=encoding)
    context_by_id = _require_complete_contexts(overlay, contexts)

    if len(context_by_id) != overlay.summary.trajectory_count:
        raise RuntimeError("Context coverage changed while partitioning by complete context")

    profiles = _build_decade_profile(overlay, context_by_id)
    annual_counts = _annual_trajectory_counts(overlay, context_by_id)

    if _sha256(source_path) != source_sha:
        raise RuntimeError("LTCCE source changed while preparing outputs")

    _publish(
        overlay=overlay,
        source_path=source_path,
        source_sha=source_sha,
        profiles=profiles,
        annual_counts=annual_counts,
        context_by_id=context_by_id,
        destination=args.output_dir,
    )

    print(
        f"{SOURCE_NAME}: {overlay.summary.trajectory_count} trajectories, "
        f"{overlay.summary.finite_observation_count} finite observations, "
        f"years {annual_counts[0]['year']}-{annual_counts[-1]['year']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
