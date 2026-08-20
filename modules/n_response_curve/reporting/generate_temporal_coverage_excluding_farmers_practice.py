#!/usr/bin/env python3
"""Render temporal observation coverage with all FP evidence excluded.

The governed temporal-coverage figure uses the registered NOPT binding for
``ph_combined_nopt_rcm`` but still contains the literature-extracted rows whose
experiment type is ``Farmer's Practice``. This variant removes those rows before
counting observations by year. The PH sibling FP arm is not appended, so no FP
evidence contributes to the bars, legend counts, limits, or caption.

The output is a standalone requested variant rather than a member of the
hash-bound descriptive-statistics bundle.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.analysis.descriptive_statistics import contracts  # noqa: E402
from n_response_curve.analysis.descriptive_statistics.agronomic import (  # noqa: E402
    _temporal_rows,
)
from n_response_curve.analysis.descriptive_statistics.config import (  # noqa: E402
    DescriptiveStatisticsConfig,
    load_recipe_config,
)
from n_response_curve.analysis.descriptive_statistics.sources import (  # noqa: E402
    LoadedSources,
    load_profiled_sources,
)
from n_response_curve.reporting import descriptive_statistics_figures as dsf  # noqa: E402


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "descriptive_statisticsCONFIG.toml"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "WF/03_Quality_Control/descriptive_statistics/agronomic"
    / "temporal_coverage_excluding_farmers_practice.jpeg"
)


def _coverage_without_farmers_practice(
    loaded: LoadedSources,
    retained: pd.DataFrame,
) -> pd.DataFrame:
    """Recount the governed observations by year after FP rows are removed."""

    rows: list[dict[str, object]] = []
    for source in loaded.sources:
        observations = retained.loc[
            retained["source_name"] == source.source_name
        ].reset_index(drop=True)
        rows.extend(_temporal_rows(source, observations))
    return contracts.conform_table(
        "temporal_coverage",
        pd.DataFrame.from_records(rows),
    )


def build_figure(
    config: DescriptiveStatisticsConfig,
    loaded: LoadedSources,
) -> tuple[plt.Figure, int, int]:
    """Build the no-FP panel and return total and dated exclusion counts."""

    recorded = dsf._observations(loaded, config)
    fp_index = dsf._core_trial_farmers_practice_index(loaded, recorded)
    if fp_index.empty:
        raise ValueError("No literature-extracted Farmer's Practice rows were found")

    excluded_dated = int(recorded.loc[fp_index, "year"].notna().sum())
    retained = recorded.drop(index=fp_index).reset_index(drop=True)
    coverage = _coverage_without_farmers_practice(loaded, retained)
    figure = dsf._plot_temporal_coverage(
        tables={"temporal_coverage": coverage},
        loaded=loaded,
        config=config,
    )
    if figure is None:
        raise ValueError("The retained observations cannot define temporal coverage")

    figure.axes[0].set_title(
        "Observation coverage by recorded year of planting — "
        "Farmer's Practice excluded"
    )
    # Replace the governed figure's generic caption with one that makes the
    # variant's population explicit. Re-running the shared helper also reserves
    # the correct caption band for the longer text.
    figure.texts.clear()
    dsf._footnote(
        figure,
        f"Farmer's Practice (FP) is excluded: {len(fp_index):,} "
        "literature-extracted FP observations were removed "
        f"({excluded_dated:,} with a recorded year), and "
        "ph_combined_nopt_rcm is represented only by its NOPT "
        "full-fertilizer arm. Bars stack the retained datasets within each "
        "recorded year; rows without a recorded year contribute no bar.",
    )
    return figure, int(len(fp_index)), excluded_dated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render temporal observation coverage without FP.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    output = DEFAULT_OUTPUT_PATH.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")

    config = load_recipe_config(args.config, project_root=PROJECT_ROOT)
    loaded = load_profiled_sources(config)
    figure, excluded, excluded_dated = build_figure(config, loaded)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output,
            dpi=config.figure_dpi,
            format="jpeg",
            facecolor="white",
        )
    finally:
        plt.close(figure)

    print(f"output={output}")
    print(f"excluded_farmers_practice_observations={excluded}")
    print(f"excluded_dated_farmers_practice_observations={excluded_dated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
