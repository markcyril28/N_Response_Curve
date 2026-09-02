#!/usr/bin/env python3
"""Group the governed core-trial source-wide overlay by recorded season.

This is a recorded-factor stratification of the exact 21-series/74-observation
population used by ``core_trial_data_source_wide.jpeg``. It is deliberately not
an unsupervised cluster analysis: the source's normalized season field assigns
each complete response series to dry or wet season, and no curve is fitted.

Usage:
    conda run -n n_response python \
      modules/n_response_curve/reporting/generate_core_trial_season_view.py \
      --config scriptCONFIG.toml
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.analysis.values import finite_number  # noqa: E402
from n_response_curve.reporting.directory_publication import (  # noqa: E402
    plain_absolute_path as _plain_absolute_path,
    publication_lock as _core_overlay_publication_lock,
    recover_interrupted_directory_publication as _recover_interrupted_directory_publication,  # noqa: E501
)
from n_response_curve.reporting.figure_output import (  # noqa: E402
    save_figure_atomically as _save_governed_figure,
)
from n_response_curve.reporting.generate_raw_dataset_overlays import (  # noqa: E402
    DEFAULT_YIELD_THRESHOLD_T_HA,
    _load_governed_core_inputs,
)
from n_response_curve.reporting.plots import _plot_observations  # noqa: E402
from n_response_curve.reporting.source_display_names import (  # noqa: E402
    display_source_name,
)

SOURCE_NAME = "core_trial_data"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scriptCONFIG.toml"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "WF/04_Response_Curves"
    / "literature_extracted_dataset/clusters/by_season"
)

_SEASON_ORDER = ("dry", "wet")
_SEASON_LABELS = {"dry": "dry season", "wet": "wet season"}
_MANAGED_FILENAMES = frozenset(
    {
        "README.md",
        "season_summary.json",
        "season_comparison.jpeg",
        "dry.jpeg",
        "wet.jpeg",
    }
)


@dataclass(frozen=True)
class SeasonStratum:
    """All governed observations for the complete series in one season."""

    season: str
    series_uids: tuple[str, ...]
    records: tuple[Mapping[str, Any], ...]

    @property
    def observation_count(self) -> int:
        return len(self.records)


@dataclass(frozen=True)
class _Observation:
    response_series_uid: str
    record_uid: str
    n_rate_kg_ha: float
    yield_t_ha: float
    treatment_text_class: str

    def as_plot_record(self) -> dict[str, Any]:
        return {
            "response_series_uid": self.response_series_uid,
            "record_uid": self.record_uid,
            "n_rate_kg_ha": self.n_rate_kg_ha,
            "yield_t_ha": self.yield_t_ha,
            "treatment_text_class": self.treatment_text_class,
        }


def group_governed_records_by_season(
    records: Sequence[Mapping[str, Any]],
    response_series_uids: Sequence[str],
) -> dict[str, SeasonStratum]:
    """Place each complete governed series into exactly one recorded season.

    The record order is preserved within each stratum. Missing, unexpected, or
    mixed season assignments fail closed instead of splitting one response
    trajectory between panels.
    """

    ordered_uids = tuple(str(value) for value in response_series_uids)
    if len(ordered_uids) != len(set(ordered_uids)):
        raise ValueError("Governed response-series ids must be unique")
    expected = set(ordered_uids)
    records_by_series: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for record in records:
        series_uid = str(record.get("response_series_uid") or "").strip()
        if series_uid in expected:
            records_by_series[series_uid].append(record)

    season_by_series: dict[str, str] = {}
    for series_uid in ordered_uids:
        members = records_by_series.get(series_uid, [])
        if not members:
            raise ValueError(
                f"Governed response series {series_uid!r} has no retained records"
            )
        season_values = tuple(
            str(record.get("season_normalized") or "").strip().casefold()
            for record in members
        )
        if any(not season for season in season_values):
            raise ValueError(
                f"Governed response series {series_uid!r} has a missing recorded season"
            )
        seasons = set(season_values)
        if len(seasons) > 1:
            raise ValueError(
                f"Governed response series {series_uid!r} carries more than one "
                f"recorded season: {sorted(seasons)}"
            )
        if not seasons:
            raise ValueError(
                f"Governed response series {series_uid!r} has no recorded season"
            )
        season = next(iter(seasons))
        if season not in _SEASON_ORDER:
            raise ValueError(
                f"Governed response series {series_uid!r} has unsupported normalized "
                f"season {season!r}; expected dry or wet"
            )
        season_by_series[series_uid] = season

    grouped: dict[str, SeasonStratum] = {}
    for season in _SEASON_ORDER:
        series = tuple(
            series_uid
            for series_uid in ordered_uids
            if season_by_series[series_uid] == season
        )
        if not series:
            continue
        series_set = set(series)
        season_records = tuple(
            record
            for record in records
            if str(record.get("response_series_uid") or "").strip() in series_set
        )
        grouped[season] = SeasonStratum(
            season=season,
            series_uids=series,
            records=season_records,
        )
    return grouped


def _finite_observations(stratum: SeasonStratum) -> tuple[_Observation, ...]:
    observations: list[_Observation] = []
    for record in stratum.records:
        n_rate = finite_number(record.get("n_rate_kg_ha"))
        yield_value = finite_number(record.get("yield_t_ha"))
        if n_rate is None or yield_value is None:
            continue
        observations.append(
            _Observation(
                response_series_uid=str(record.get("response_series_uid") or ""),
                record_uid=str(record.get("record_uid") or ""),
                n_rate_kg_ha=float(n_rate),
                yield_t_ha=float(yield_value),
                treatment_text_class=str(
                    record.get("treatment_text_class") or "unresolved"
                ),
            )
        )
    if len(observations) != stratum.observation_count:
        raise ValueError(
            f"Season {stratum.season!r} contains a nonfinite governed observation"
        )
    return tuple(observations)


def _draw_stratum(axes: Any, stratum: SeasonStratum, *, include_legend: bool) -> None:
    observations = _finite_observations(stratum)
    plot_records = tuple(observation.as_plot_record() for observation in observations)
    _plot_observations(axes, plot_records)

    by_series: dict[str, list[_Observation]] = collections.defaultdict(list)
    for observation in observations:
        by_series[observation.response_series_uid].append(observation)
    line_label = "within-series connecting lines (visual aid; not a fit)"
    for index, series_uid in enumerate(stratum.series_uids):
        ordered = sorted(
            by_series[series_uid],
            key=lambda observation: (
                observation.n_rate_kg_ha,
                observation.record_uid,
            ),
        )
        axes.plot(
            [observation.n_rate_kg_ha for observation in ordered],
            [observation.yield_t_ha for observation in ordered],
            label=line_label if index == 0 else "_nolegend_",
            color="grey",
            linewidth=1,
            alpha=0.35,
            zorder=1,
        )

    n_values = [observation.n_rate_kg_ha for observation in observations]
    axes.set_xlabel("Applied N (kg N/ha)")
    axes.set_ylabel("Grain yield (t/ha)")
    axes.set_title(
        f"{_SEASON_LABELS[stratum.season]}\n"
        f"series={len(stratum.series_uids)}; observations={len(observations)}; "
        f"N range={min(n_values):g}-{max(n_values):g} kg/ha\n"
        "recorded season stratum; not a statistical cluster or fit"
    )
    if include_legend:
        axes.legend(loc="best", fontsize=8)


def _padded_range(values: Sequence[float]) -> tuple[float, float]:
    lower, upper = min(values), max(values)
    span = upper - lower
    padding = span * 0.04 if span > 0 else max(abs(upper), 1.0) * 0.05
    return lower - padding, upper + padding


def _all_plot_values(
    strata: Mapping[str, SeasonStratum],
) -> tuple[list[float], list[float]]:
    observations = [
        observation
        for stratum in strata.values()
        for observation in _finite_observations(stratum)
    ]
    return (
        [observation.n_rate_kg_ha for observation in observations],
        [observation.yield_t_ha for observation in observations],
    )


def _build_comparison_figure(strata: Mapping[str, SeasonStratum]) -> Any:
    """Build the shared-frame comparison with a reserved legend strip."""

    from matplotlib import pyplot as plt

    seasons = tuple(season for season in _SEASON_ORDER if season in strata)
    figure, axes_grid = plt.subplots(
        1,
        len(seasons),
        figsize=(7.5 * len(seasons), 7.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes_list = list(axes_grid[0])
    try:
        for axes, season in zip(axes_list, seasons, strict=True):
            _draw_stratum(axes, strata[season], include_legend=False)

        n_values, yield_values = _all_plot_values(strata)
        x_limits = _padded_range(n_values)
        y_limits = _padded_range(yield_values)
        for axes in axes_list:
            axes.set_xlim(*x_limits)
            axes.set_ylim(*y_limits)

        handles: list[Any] = []
        labels: list[str] = []
        seen: set[str] = set()
        for axes in axes_list:
            for handle, label in zip(
                *axes.get_legend_handles_labels(), strict=True
            ):
                if label in seen or label.startswith("_"):
                    continue
                seen.add(label)
                handles.append(handle)
                labels.append(label)
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=max(1, len(labels)),
            fontsize=8,
            frameon=False,
        )
        # A figure-level legend is not included in constrained-layout's axes
        # accounting. Reserve its own strip so it cannot sit on the x labels.
        layout_engine: Any = figure.get_layout_engine()
        layout_engine.set(rect=(0.0, 0.07, 1.0, 0.93))
        total_series = sum(len(stratum.series_uids) for stratum in strata.values())
        total_observations = sum(
            stratum.observation_count for stratum in strata.values()
        )
        figure.suptitle(
            f"source={display_source_name(SOURCE_NAME)} — governed observed series "
            "grouped by recorded planting season\n"
            f"series={total_series}; observations={total_observations}; panels share "
            "both axes\n"
            "recorded strata only; no statistical clustering, pooled curve, or fit",
            fontsize=13,
        )
        return figure
    except Exception:
        plt.close(figure)
        raise


def _write_comparison(
    strata: Mapping[str, SeasonStratum], destination: Path
) -> None:
    from matplotlib import pyplot as plt

    figure = _build_comparison_figure(strata)
    try:
        _save_governed_figure(figure, destination)
    finally:
        plt.close(figure)


def _write_individual(stratum: SeasonStratum, destination: Path) -> None:
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(figsize=(10, 7), constrained_layout=True)
    try:
        _draw_stratum(axes, stratum, include_legend=True)
        axes.set_title(
            "\n".join(
                (
                    f"source={display_source_name(SOURCE_NAME)}",
                    axes.get_title(),
                    "complete governed response series retained within the stratum",
                    "descriptive observed-series overlay; no pooled curve or fit",
                )
            )
        )
        _save_governed_figure(figure, destination)
    finally:
        plt.close(figure)


def _summary(
    strata: Mapping[str, SeasonStratum],
) -> dict[str, Any]:
    return {
        "source_name": SOURCE_NAME,
        "method": "recorded_season_stratification",
        "interpretation": (
            "Recorded dry/wet strata of complete governed response series; "
            "not an unsupervised cluster analysis and no curve is fitted."
        ),
        "population": {
            "series_count": sum(
                len(stratum.series_uids) for stratum in strata.values()
            ),
            "observation_count": sum(
                stratum.observation_count for stratum in strata.values()
            ),
        },
        "seasons": {
            season: {
                "label": _SEASON_LABELS[season],
                "series_count": len(stratum.series_uids),
                "observation_count": stratum.observation_count,
                "figure": f"{season}.jpeg",
            }
            for season, stratum in strata.items()
        },
        "comparison_figure": "season_comparison.jpeg",
    }


def _readme(summary: Mapping[str, Any]) -> str:
    population = summary["population"]
    lines = [
        "# `core_trial_data` grouped by recorded season",
        "",
        "This is a descriptive stratification of the exact governed population in",
        "`../../core_trial_data_source_wide.jpeg`. Complete response series are",
        "placed into the normalized season recorded in the eligibility ledger.",
        "It is not an unsupervised cluster analysis, and no curve is fitted.",
        "",
        "## Population",
        "",
        f"- {population['series_count']} governed response series.",
        f"- {population['observation_count']} finite observed N-yield pairs.",
        "- No series is split between season panels.",
        "",
        "## Outputs",
        "",
        "- `season_comparison.jpeg` — dry and wet strata side by side on shared axes.",
    ]
    for season, details in summary["seasons"].items():
        lines.append(
            f"- `{details['figure']}` — {details['label']}: "
            f"{details['series_count']} series, "
            f"{details['observation_count']} observations."
        )
    lines += [
        "- `season_summary.json` — machine-readable counts and method boundary.",
        "",
        "## Interpretation boundary",
        "",
        "The panels show recorded dry/wet strata. Differences between them are",
        "descriptive and may also reflect study, site, planting year, treatment",
        "ladder, variety, and other design differences. They are not season effects.",
        "",
        "## Regenerating",
        "",
        "`conda run -n n_response python modules/n_response_curve/reporting/",
        "generate_core_trial_season_view.py --config scriptCONFIG.toml`",
        "",
    ]
    return "\n".join(lines)


def _replace_output_unlocked(
    output_dir: Path,
    strata: Mapping[str, SeasonStratum],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.staging.{uuid.uuid4().hex}")
    backup = output_dir.with_name(f".{output_dir.name}.backup.{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        _write_comparison(strata, staging / "season_comparison.jpeg")
        for season, stratum in strata.items():
            _write_individual(stratum, staging / f"{season}.jpeg")
        summary = _summary(strata)
        (staging / "season_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "README.md").write_text(_readme(summary), encoding="utf-8")

        if output_dir.exists():
            if not output_dir.is_dir() or output_dir.is_symlink():
                raise RuntimeError("Season-view output is not a plain directory")
            for entry in output_dir.iterdir():
                if entry.name in _MANAGED_FILENAMES:
                    continue
                destination = staging / entry.name
                if entry.is_dir() and not entry.is_symlink():
                    shutil.copytree(entry, destination)
                else:
                    shutil.copy2(entry, destination, follow_symlinks=False)
            os.replace(output_dir, backup)
        try:
            os.replace(staging, output_dir)
        except Exception:
            if backup.exists() and not output_dir.exists():
                os.replace(backup, output_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)


def _season_publication_container(output_dir: Path) -> Path:
    """Return the parent container also replaced by the source-wide writer."""

    if output_dir.name == "by_season" and output_dir.parent.name == "clusters":
        return output_dir.parent.parent
    return output_dir


def _replace_output(
    output_dir: Path,
    strata: Mapping[str, SeasonStratum],
) -> None:
    output_dir = _plain_absolute_path(output_dir, label="Season-view output")
    with _core_overlay_publication_lock(_season_publication_container(output_dir)):
        _recover_interrupted_directory_publication(output_dir)
        _replace_output_unlocked(output_dir, strata)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Group the governed core-trial source-wide overlay by recorded dry/wet "
            "season without fitting or statistical clustering."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config.resolve()
    inputs = _load_governed_core_inputs(
        config_path,
        yield_threshold_t_ha=DEFAULT_YIELD_THRESHOLD_T_HA,
        require_yield_selection=False,
    )
    strata = group_governed_records_by_season(
        inputs.records,
        inputs.response_series_uids,
    )
    _replace_output(args.output_dir, strata)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "series_count": sum(
                    len(stratum.series_uids) for stratum in strata.values()
                ),
                "observation_count": sum(
                    stratum.observation_count for stratum in strata.values()
                ),
                "seasons": {
                    season: len(stratum.series_uids)
                    for season, stratum in strata.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
