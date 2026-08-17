#!/usr/bin/env python3
"""Build a planting-year view for the PH combined overlay with NOPT NPK excluded.

The source stores FP, RCM, NOPT full-fertilizer, and zero-N arms on one linked
row. This view keeps the exact linked-row/finite-arm population used by
``ph_combined_nopt_rcm_source_wide_no_nopt_npk.jpeg``. It never interprets the
remaining same-row arms as a response series: annual summaries are computed
separately for FP, RCM, and zero N.
"""

from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import math
import os
import shutil
import statistics
import sys
import textwrap
import tomllib
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.reporting import source_dataset_overlays as sdo  # noqa: E402
from n_response_curve.reporting.generate_combined_overlay_excluding_arm import (  # noqa: E402
    _exclude_treatment_classes,
)

SOURCE_NAME = "ph_combined_nopt_rcm"
EXCLUDED_TREATMENT_CLASSES = frozenset({"NOPT NPK"})
TREND_TREATMENT_CLASSES = ("FP", "RCM", "zero N")
ANNOTATED_FIGURE_FILENAME = "decades_and_trend.jpeg"
FIGURE_ONLY_FILENAME = "decades_and_trend_figure.jpeg"
NOTES_FILENAME = "decades_and_trend_notes.md"
_OWNED_FILENAMES = frozenset(
    {
        ANNOTATED_FIGURE_FILENAME,
        FIGURE_ONLY_FILENAME,
        NOTES_FILENAME,
        "README.md",
    }
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "WF/04_Response_Curves/z_n_response_full/overlay/source_dataset"
    / "ph_combined_nopt_rcm/clusters/by_planting_year/no_nopt_npk"
)
_ARM_COLOURS = {"FP": "C0", "RCM": "C1", "zero N": "C2"}
_SEASON_STYLES = {
    "dry": ("o", "-", "dry season"),
    "wet": ("s", (0, (5, 2)), "wet season"),
}
_CONTEXT_HEADERS = ("year", "season")
_SEASON_ALIASES = {
    "dry": "dry",
    "dry season": "dry",
    "ds": "dry",
    "wet": "wet",
    "wet season": "wet",
    "ws": "wet",
}


@dataclass(frozen=True)
class LinkedRowContext:
    trajectory_id: str
    year: int | None
    season: str


@dataclass(frozen=True)
class AnnualArmSummary:
    year: int
    season: str
    treatment_class: str
    observation_count: int
    mean_yield_t_ha: float
    standard_error_t_ha: float


@dataclass(frozen=True)
class PreparedAnalysis:
    overlay: sdo.SourceDatasetOverlay
    contexts: Mapping[str, LinkedRowContext]
    annual_summaries: tuple[AnnualArmSummary, ...]
    years: tuple[int, ...]
    decades: tuple[str, ...]
    source_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_season(value: str) -> str:
    return _SEASON_ALIASES.get(value.strip().casefold(), "unrecorded")


def _parse_year(value: str) -> int | None:
    text = value.strip()
    if not (len(text) == 4 and text.isdigit()):
        return None
    year = int(text)
    return year if 1900 <= year <= 2100 else None


def _read_contexts(csv_path: Path, *, encoding: str) -> dict[str, LinkedRowContext]:
    if codecs.lookup(encoding).name != codecs.lookup("cp1252").name:
        raise ValueError("PH combined planting-year input must use the reviewed cp1252 encoding")
    if csv_path.is_symlink():
        raise ValueError("PH combined planting-year input may not be a symlink")

    with csv_path.open("r", encoding="cp1252", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        missing = [name for name in _CONTEXT_HEADERS if header.count(name) != 1]
        if missing:
            raise ValueError(
                "PH combined planting-year input requires one unambiguous "
                + ", ".join(missing)
                + " header"
            )
        indices = {name: header.index(name) for name in _CONTEXT_HEADERS}
        contexts = {}
        for row_index, row in enumerate(reader):
            values = {
                name: row[index] if index < len(row) else ""
                for name, index in indices.items()
            }
            trajectory_id = sdo._trajectory_id(SOURCE_NAME, "row", str(row_index))
            contexts[trajectory_id] = LinkedRowContext(
                trajectory_id=trajectory_id,
                year=_parse_year(values["year"]),
                season=_normalize_season(values["season"]),
            )
    return contexts


def _annual_arm_summaries(
    overlay: sdo.SourceDatasetOverlay,
    contexts: Mapping[str, LinkedRowContext],
) -> tuple[AnnualArmSummary, ...]:
    values: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for trajectory in overlay.trajectories:
        context = contexts.get(trajectory.trajectory_id)
        if context is None or context.year is None or context.season == "unrecorded":
            continue
        for observation in trajectory.observations:
            if observation.treatment_class not in TREND_TREATMENT_CLASSES:
                continue
            values[
                (context.year, context.season, observation.treatment_class)
            ].append(observation.yield_t_ha)

    summaries = []
    for (year, season, treatment_class), yields in sorted(values.items()):
        standard_error = (
            statistics.stdev(yields) / math.sqrt(len(yields))
            if len(yields) > 1
            else math.nan
        )
        summaries.append(
            AnnualArmSummary(
                year=year,
                season=season,
                treatment_class=treatment_class,
                observation_count=len(yields),
                mean_yield_t_ha=statistics.fmean(yields),
                standard_error_t_ha=standard_error,
            )
        )
    return tuple(summaries)


def _prepare_analysis(csv_path: Path, *, encoding: str) -> PreparedAnalysis:
    csv_path = Path(csv_path)
    before = _sha256(csv_path)
    source_overlay = sdo.read_source_dataset_overlay(
        csv_path,
        SOURCE_NAME,
        encoding=encoding,
    )
    overlay = _exclude_treatment_classes(
        source_overlay,
        set(EXCLUDED_TREATMENT_CLASSES),
    )
    contexts = _read_contexts(csv_path, encoding=encoding)
    after = _sha256(csv_path)
    if after != before:
        raise RuntimeError("PH combined source bytes changed while preparing the figure")

    annual_summaries = _annual_arm_summaries(overlay, contexts)
    years = tuple(
        sorted(
            {
                context.year
                for trajectory in overlay.trajectories
                if (context := contexts.get(trajectory.trajectory_id)) is not None
                and context.year is not None
            }
        )
    )
    if not years or not annual_summaries:
        raise ValueError("PH combined source has no usable planting-year arm summaries")
    decades = tuple(sorted({f"{year // 10 * 10}s" for year in years}))
    return PreparedAnalysis(
        overlay=overlay,
        contexts=contexts,
        annual_summaries=annual_summaries,
        years=years,
        decades=decades,
        source_sha256=before,
    )


def _contiguous_runs(entries: list[AnnualArmSummary]) -> list[list[AnnualArmSummary]]:
    runs: list[list[AnnualArmSummary]] = []
    for entry in sorted(entries, key=lambda item: item.year):
        if runs and entry.year == runs[-1][-1].year + 1:
            runs[-1].append(entry)
        else:
            runs.append([entry])
    return runs


def _arm_counts(analysis: PreparedAnalysis) -> dict[str, int]:
    return {
        treatment_class: sum(
            observation.treatment_class == treatment_class
            for trajectory in analysis.overlay.trajectories
            for observation in trajectory.observations
        )
        for treatment_class in TREND_TREATMENT_CLASSES
    }


def _season_row_counts(analysis: PreparedAnalysis) -> dict[str, int]:
    counts = {"dry": 0, "wet": 0, "unrecorded": 0}
    for trajectory in analysis.overlay.trajectories:
        context = analysis.contexts.get(trajectory.trajectory_id)
        if context is not None:
            counts[context.season] = counts.get(context.season, 0) + 1
    return counts


def _disclosures(analysis: PreparedAnalysis) -> list[str]:
    arm_counts = _arm_counts(analysis)
    season_counts = _season_row_counts(analysis)
    year_text = f"{analysis.years[0]}-{analysis.years[-1]}"
    return [
        f"Population: {analysis.overlay.summary.trajectory_count} linked source rows and "
        f"{analysis.overlay.summary.finite_observation_count} retained finite arms "
        f"({arm_counts['FP']} FP, {arm_counts['RCM']} RCM, "
        f"{arm_counts['zero N']} zero N), matching the no-NOPT-NPK source-wide view.",
        "NOPT NPK is excluded from the inset, annual summaries, legends, and "
        "connecting lines; FP, RCM, and zero N remain separate treatment classes.",
        "The source stores these arms on one linked row; comparability is not "
        "assumed; gray within-row connectors in the inset are visual aids, not "
        "response trajectories, fits, or causal contrasts.",
        "Each annual point is the plain mean grain yield for one treatment class, "
        "planting year, and season; its error bar is the standard error across "
        "finite linked rows when at least two rows contribute. No arm is subtracted "
        "from another.",
        f"The recorded years are {year_text}, all within the {analysis.decades[0]}. "
        "One real decade inset is shown; empty decades are not invented.",
        f"Season is kept separate: {season_counts['dry']} dry-season rows and "
        f"{season_counts['wet']} wet-season rows. Lines break across unrecorded "
        "calendar years rather than bridging missing evidence.",
        "Rows contributing to different years are not a longitudinal cohort. Lines "
        "between annual means are orientation aids over a compiled record, not an "
        "estimated time effect.",
        "Exploratory observed-data diagnostic only: no pooled curve, response "
        "difference, fitted trend, recommendation, or treatment-comparability claim.",
    ]


def _draw_linked_arm_inset(analysis: PreparedAnalysis, axes) -> None:
    marker_size, marker_alpha, line_alpha = sdo._adaptive_style(
        analysis.overlay.summary.finite_observation_count,
        analysis.overlay.summary.trajectory_count,
    )
    for treatment_class in TREND_TREATMENT_CLASSES:
        observations = [
            observation
            for trajectory in analysis.overlay.trajectories
            for observation in trajectory.observations
            if observation.treatment_class == treatment_class
        ]
        axes.scatter(
            [observation.n_rate_kg_ha for observation in observations],
            [observation.yield_t_ha for observation in observations],
            s=marker_size,
            alpha=marker_alpha,
            color=_ARM_COLOURS[treatment_class],
            label=treatment_class,
            zorder=3,
        )
    line_label_emitted = False
    for trajectory in analysis.overlay.trajectories:
        if len(trajectory.observations) < 2:
            continue
        axes.plot(
            [entry.n_rate_kg_ha for entry in trajectory.observations],
            [entry.yield_t_ha for entry in trajectory.observations],
            color="0.65",
            linewidth=0.8,
            alpha=line_alpha,
            label=(
                "within-row connecting lines (visual aid; not a fit)"
                if not line_label_emitted
                else "_nolegend_"
            ),
            zorder=1,
        )
        line_label_emitted = True

    n_range = analysis.overlay.summary.n_rate_range_kg_ha
    yield_range = analysis.overlay.summary.yield_range_t_ha
    assert n_range is not None and yield_range is not None
    n_pad = max(5.0, 0.03 * (n_range[1] - n_range[0]))
    yield_pad = max(0.25, 0.04 * (yield_range[1] - yield_range[0]))
    axes.set_xlim(n_range[0] - n_pad, n_range[1] + n_pad)
    axes.set_ylim(yield_range[0] - yield_pad, yield_range[1] + yield_pad)
    axes.set_xlabel("Applied N (kg N/ha)")
    axes.set_ylabel("Grain yield (t/ha)")
    axes.grid(alpha=0.25, linewidth=0.6)
    axes.set_title(
        f"{analysis.decades[0]} linked-arm cloud ({analysis.years[0]}-"
        f"{analysis.years[-1]})\n"
        f"{analysis.overlay.summary.trajectory_count} linked rows; "
        f"{analysis.overlay.summary.finite_observation_count} finite arms; "
        "NOPT NPK excluded",
        fontsize=14,
        fontweight="bold",
    )
    axes.legend(loc="upper right", fontsize=9, framealpha=0.92)


def _draw_trend_panels(analysis: PreparedAnalysis, axes_list) -> None:
    finite_bounds = []
    for entry in analysis.annual_summaries:
        error = (
            entry.standard_error_t_ha
            if math.isfinite(entry.standard_error_t_ha)
            else 0.0
        )
        finite_bounds.extend(
            (entry.mean_yield_t_ha - error, entry.mean_yield_t_ha + error)
        )
    low, high = min(finite_bounds), max(finite_bounds)
    padding = max(0.15, 0.08 * (high - low))

    for axes, treatment_class in zip(
        axes_list,
        TREND_TREATMENT_CLASSES,
        strict=True,
    ):
        for season, (marker, linestyle, label) in _SEASON_STYLES.items():
            entries = [
                entry
                for entry in analysis.annual_summaries
                if entry.treatment_class == treatment_class and entry.season == season
            ]
            for run_index, run in enumerate(_contiguous_runs(entries)):
                axes.errorbar(
                    [entry.year for entry in run],
                    [entry.mean_yield_t_ha for entry in run],
                    yerr=[entry.standard_error_t_ha for entry in run],
                    marker=marker,
                    linestyle=linestyle,
                    markersize=7,
                    linewidth=1.8,
                    capsize=3,
                    color=_ARM_COLOURS[treatment_class],
                    ecolor=_ARM_COLOURS[treatment_class],
                    elinewidth=1.0,
                    label=label if run_index == 0 else "_nolegend_",
                    zorder=3,
                )
        axes.set_xlim(analysis.years[0] - 0.5, analysis.years[-1] + 0.5)
        axes.set_ylim(low - padding, high + padding)
        axes.set_ylabel(f"{treatment_class} yield\n(t/ha)")
        axes.grid(axis="y", alpha=0.25, linewidth=0.6)
        axes.set_title(
            f"Observed {treatment_class} grain yield by planting year and season",
            fontsize=11,
            color=_ARM_COLOURS[treatment_class],
        )
        axes.tick_params(labelsize=9)
    axes_list[-1].set_xticks(analysis.years)
    axes_list[-1].set_xlabel("Planting year")
    for axes in axes_list[:-1]:
        axes.tick_params(labelbottom=False)


def _build_figure(
    analysis: PreparedAnalysis,
    *,
    annotated: bool,
):
    from matplotlib import pyplot as plt

    height = 20.0 if annotated else 15.0
    figure = plt.figure(figsize=(12.0, height))
    bottom = 0.30 if annotated else 0.20
    grid = figure.add_gridspec(
        4,
        1,
        left=0.09,
        right=0.97,
        top=0.86,
        bottom=bottom,
        hspace=0.33,
        height_ratios=(1.65, 0.78, 0.78, 0.78),
    )
    inset_axes = figure.add_subplot(grid[0])
    trend_axes = [figure.add_subplot(grid[index]) for index in range(1, 4)]
    _draw_linked_arm_inset(analysis, inset_axes)
    _draw_trend_panels(analysis, trend_axes)
    handles, labels = trend_axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035 if not annotated else 0.17),
        ncol=2,
        fontsize=10,
        frameon=False,
        title=(
            "planting season — marker shape and line style; panel colour is "
            "the treatment class"
        ),
        title_fontsize=10,
    )
    figure.suptitle(
        "source=ph_combined_nopt_rcm\n"
        "2010s linked-arm N-yield cloud over observed annual arm-yield trends, "
        "by season\n"
        "comparability is not assumed; NOPT NPK excluded; no pooled curve or fit",
        y=0.985,
        fontsize=15,
        linespacing=1.25,
    )
    footer = (
        "Exploratory observed-data diagnostic. Same-row connectors and annual "
        "summary lines are visual aids; no response difference or fitted trend. "
        f"Read {NOTES_FILENAME} with this figure."
    )
    footer_text = figure.text(
        0.5,
        0.13 if not annotated else 0.245,
        textwrap.fill(footer, width=135),
        ha="center",
        va="bottom",
        fontsize=9,
        color="0.35",
    )
    footer_text.set_gid("figure-footer")
    if annotated:
        wrapped = [
            textwrap.fill(
                disclosure,
                width=90,
                initial_indent="— ",
                subsequent_indent="   ",
            )
            for disclosure in _disclosures(analysis)
        ]
        left = "\n\n".join(wrapped[:4])
        right = "\n\n".join(wrapped[4:])
        figure.text(0.06, 0.025, left, ha="left", va="bottom", fontsize=8.2)
        figure.text(0.52, 0.025, right, ha="left", va="bottom", fontsize=8.2)
    return figure


def _write_figure(
    analysis: PreparedAnalysis,
    destination: Path,
    *,
    annotated: bool,
) -> None:
    from matplotlib import pyplot as plt

    figure = _build_figure(analysis, annotated=annotated)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, format="jpeg", dpi=150)
    finally:
        plt.close(figure)


def _write_notes(analysis: PreparedAnalysis, destination: Path) -> None:
    lines = [
        f"# `{FIGURE_ONLY_FILENAME}` — how to read it",
        "",
        "These notes and the figure are one document. The plot adapts the core-trial",
        "decades-and-trend layout to the PH combined linked-row source without",
        "turning linked arms into response curves.",
        "",
        "## Disclosures",
        "",
        *[f"- {entry}" for entry in _disclosures(analysis)],
        "",
        "## Provenance",
        "",
        f"- Source token: `{SOURCE_NAME}`.",
        f"- Source SHA-256: `{analysis.source_sha256}`.",
        f"- Recorded planting years: {analysis.years[0]}-{analysis.years[-1]}; "
        f"decade coverage: {', '.join(analysis.decades)}.",
        "- Regenerate the two figures, these notes, and README together.",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def _write_readme(analysis: PreparedAnalysis, destination: Path) -> None:
    lines = [
        "# PH combined planting-year linked-arm view",
        "",
        "This folder applies the core-trial `decades_and_trend` presentation to",
        "`ph_combined_nopt_rcm_source_wide_no_nopt_npk.jpeg` while preserving the",
        "PH source's linked-row semantics.",
        "",
        f"- `{ANNOTATED_FIGURE_FILENAME}` is self-contained with disclosures.",
        f"- `{FIGURE_ONLY_FILENAME}` is the plates-only form used with `{NOTES_FILENAME}`.",
        f"- Population: {analysis.overlay.summary.trajectory_count} linked rows, "
        f"{analysis.overlay.summary.finite_observation_count} retained observations.",
        f"- Coverage: {analysis.years[0]}-{analysis.years[-1]} ({', '.join(analysis.decades)}).",
        "- NOPT NPK is excluded; FP, RCM, and zero N are summarized separately.",
        "- Comparability is not assumed and no pooled curve or fit is drawn.",
        "",
        "## Regenerate",
        "",
        "```bash",
        "conda run -n n_response python --no-capture-output \\",
        "  modules/n_response_curve/reporting/generate_ph_combined_planting_year_view.py \\",
        "  --config scriptCONFIG.toml",
        "```",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def _publish_document_set(
    analysis: PreparedAnalysis,
    destination: Path,
    *,
    source_path: Path | None = None,
) -> None:
    """Stage the managed files and replace the directory with rollback."""

    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(
        f".{destination.name}.staging.{uuid.uuid4().hex}"
    )
    backup = destination.with_name(
        f".{destination.name}.backup.{uuid.uuid4().hex}"
    )
    staging.mkdir()
    promoted = False
    try:
        _write_figure(
            analysis,
            staging / ANNOTATED_FIGURE_FILENAME,
            annotated=True,
        )
        _write_figure(
            analysis,
            staging / FIGURE_ONLY_FILENAME,
            annotated=False,
        )
        _write_notes(analysis, staging / NOTES_FILENAME)
        _write_readme(analysis, staging / "README.md")

        if destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise RuntimeError("Planting-year output destination is not a plain directory")
            for entry in sorted(destination.iterdir()):
                if entry.name in _OWNED_FILENAMES:
                    continue
                if entry.is_symlink():
                    raise RuntimeError(
                        f"Planting-year output contains an unmanaged symlink: {entry.name}"
                    )
                if entry.is_dir():
                    shutil.copytree(entry, staging / entry.name)
                elif entry.is_file():
                    shutil.copy2(entry, staging / entry.name)
                else:
                    raise RuntimeError(
                        f"Planting-year output contains an unsafe entry: {entry.name}"
                    )

        if source_path is not None and _sha256(source_path) != analysis.source_sha256:
            raise RuntimeError("PH combined source bytes changed while rendering the figure")

        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
            promoted = True
        except Exception:
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and not promoted and not destination.exists():
            os.replace(backup, destination)
        elif backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def _load_source_spec(config_path: Path) -> tuple[Path, str]:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    sources = config.get("sources")
    source = sources.get(SOURCE_NAME) if isinstance(sources, dict) else None
    if not isinstance(source, dict):
        raise ValueError(f"The configuration is missing [sources.{SOURCE_NAME}]")
    raw_path = source.get("data_path")
    encoding = source.get("encoding")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"[sources.{SOURCE_NAME}].data_path must be a nonempty string")
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError(f"[sources.{SOURCE_NAME}].encoding must be a nonempty string")
    source_path = Path(raw_path)
    if not source_path.is_absolute():
        source_path = config_path.resolve().parent / source_path
    return source_path.resolve(), encoding


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "scriptCONFIG.toml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_path, encoding = _load_source_spec(args.config.resolve())
    analysis = _prepare_analysis(source_path, encoding=encoding)
    destination = args.output_dir.resolve()
    _publish_document_set(
        analysis,
        destination,
        source_path=source_path,
    )
    print(
        f"{SOURCE_NAME}: linked rows={analysis.overlay.summary.trajectory_count}; "
        f"retained observations={analysis.overlay.summary.finite_observation_count}; "
        f"years={analysis.years[0]}-{analysis.years[-1]}; "
        f"decades={','.join(analysis.decades)}"
    )
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
