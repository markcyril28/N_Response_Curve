#!/usr/bin/env python3
"""Render the N-yield trajectory panel with Farmer's Practice as its own series.

The declared ``yield_versus_nitrogen_trajectories`` panel draws the three
registered datasets and represents ``ph_combined_nopt_rcm`` by its NOPT-N
binding alone, because that source records its farmer's-practice treatment in
sibling columns rather than in rows. This variant adds that arm and moves the
literature-extracted rows recorded as ``Farmer's Practice`` into it.

The arm contributes points and no lines: a farmer-chosen rate is one treatment
on one trial, not a rung of a designed ladder. Its rates reach far beyond any
designed ladder in the profiled sources, so this panel's x-range differs from
the declared one and the two are not read against each other.

The output is a standalone requested variant rather than a member of the
hash-bound descriptive-statistics bundle; the declared panel it varies is
published unchanged beside it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.analysis.descriptive_statistics import contracts  # noqa: E402
from n_response_curve.analysis.descriptive_statistics.config import (  # noqa: E402
    DescriptiveStatisticsConfig,
    load_recipe_config,
)
from n_response_curve.analysis.descriptive_statistics.sources import (  # noqa: E402
    LoadedSources,
    load_profiled_sources,
)
from n_response_curve.reporting import descriptive_statistics_figures as dsf  # noqa: E402
from n_response_curve.reporting.figure_output import (  # noqa: E402
    save_figure_atomically,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "descriptive_statisticsCONFIG.toml"
# The declared figure this one varies. Its declared filename, not a literal
# repeated here, supplies the stem below.
BASE_FIGURE_NAME = "yield_versus_nitrogen_trajectories"


def default_output_path(config: DescriptiveStatisticsConfig) -> Path:
    """Where this variant lands, derived from the figure it varies.

    The stem comes from the declared panel's own path, so a rename of that
    panel — through the contract's ``output_filename_stem`` — carries into this
    variant instead of leaving two filenames that disagree about which figure
    they vary.
    """

    return config.output_root / contracts.variant_figure_relative_path(
        BASE_FIGURE_NAME,
        contracts.WITH_FARMERS_PRACTICE_SUFFIX,
        config.primary_figure_format,
    )


def build_figure(
    config: DescriptiveStatisticsConfig,
    loaded: LoadedSources,
) -> plt.Figure:
    """Build the FP-inclusive trajectory panel."""

    # ``tables`` is unused by this builder, which reads the observation cloud
    # directly; passing an empty mapping keeps the whole profile off the path.
    figure = dsf._plot_yield_versus_nitrogen_trajectories(
        tables={},
        loaded=loaded,
        config=config,
        farmers_practice=True,
    )
    if figure is None:
        raise ValueError("The observations define no joinable series")
    return figure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the N-yield trajectory panel including FP.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination; defaults beside the declared panel it varies.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    config = load_recipe_config(args.config, project_root=PROJECT_ROOT)
    output = (args.output or default_output_path(config)).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")

    loaded = load_profiled_sources(config)
    figure = build_figure(config, loaded)
    try:
        save_figure_atomically(
            figure,
            output,
            image_format=config.primary_figure_format,
            dpi=config.figure_dpi,
        )
    finally:
        plt.close(figure)

    print(f"output={output}")
    for handle in figure.axes[0].get_legend().get_texts():
        print(f"series={handle.get_text()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
