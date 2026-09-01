#!/usr/bin/env python3
"""Render the inorganic N-rate distribution with all FP evidence excluded.

The source figure appends the paired farmer-applied N-rate arm from
``ph_combined_nopt_rcm`` and moves literature rows explicitly classified as
``Farmer's Practice`` into that series. This variant does neither: it removes
the literature FP rows and retains the Philippine source only through its
governed NOPT full-fertilizer binding.

The output is a standalone requested variant rather than a member of the
hash-bound descriptive-statistics bundle.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


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


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "descriptive_statisticsCONFIG.toml"
# The governed figure this one varies. Its declared filename, not a literal
# repeated here, supplies the stem below.
BASE_FIGURE_NAME = "nitrogen_rate_distribution"


def default_output_path(config: DescriptiveStatisticsConfig) -> Path:
    """Where this variant lands, derived from the figure it varies.

    The stem comes from the governed panel's own declared path, so a rename of
    that panel — through the contract's ``output_filename_stem`` — carries into
    this variant instead of leaving two filenames that disagree about which
    figure they vary.
    """

    return config.output_root / contracts.variant_figure_relative_path(
        BASE_FIGURE_NAME,
        contracts.EXCLUDING_FARMERS_PRACTICE_SUFFIX,
        config.primary_figure_format,
    )


def build_figure(
    config: DescriptiveStatisticsConfig,
    loaded: LoadedSources,
) -> tuple[plt.Figure, int, int]:
    """Build the no-FP panel and return explicit and paired FP row counts."""

    recorded = dsf._observations(loaded, config)
    if recorded.empty:
        raise ValueError("No governed N-rate observations were loaded")

    fp_index = dsf._core_trial_farmers_practice_index(loaded, recorded)
    farmers_practice = dsf._farmers_practice_series(loaded, recorded, config)
    if fp_index.empty and farmers_practice.arm_rows == 0:
        raise ValueError("No Farmer's Practice observations were found")

    observations = recorded.drop(index=fp_index).reset_index(drop=True)
    names = dsf._ordered_sources(loaded, observations["source_name"])
    if not names:
        raise ValueError("No retained dataset has N-rate observations")

    rates = observations["n_rate_kg_ha"].to_numpy(dtype=float)
    width = float(config.nitrogen_bin_width_kg_ha)
    lower = float(np.floor(np.nanmin(rates) / width) * width)
    upper = float(np.ceil(np.nanmax(rates) / width) * width)
    edges = np.arange(lower, upper + width / 2.0, width)
    if edges.size < 2:
        raise ValueError("The retained observations cannot define histogram bins")

    colours = dsf._source_colours(loaded)
    figure, axis = dsf._figure(config, height_inches=7.0)
    for name in names:
        values = observations.loc[
            observations["source_name"] == name, "n_rate_kg_ha"
        ].to_numpy(dtype=float)
        if values.size == 0:
            continue
        distinct = int(np.unique(np.round(values, 6)).size)
        dsf._share_histogram(
            axis,
            values,
            edges=edges,
            colour=colours[name],
            label=(
                f"{dsf._legend_label(name)} — "
                f"{values.size:,} observations, {distinct} distinct rates"
            ),
        )

    axis.set(
        title=(
            "Recorded inorganic N-rate distribution by dataset — "
            "Farmer's Practice excluded "
            f"({width:g} kg N ha⁻¹ bins)"
        ),
        xlabel="Inorganic N rate (kg N ha⁻¹)",
        ylabel="Share within each retained dataset (%)",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    dsf._footnote(
        figure,
        f"Farmer's Practice (FP) is excluded: {len(fp_index):,} "
        "literature-extracted FP observations were removed, and the "
        f"{farmers_practice.arm_rows:,} paired farmer-applied rates in "
        "ph_combined_nopt_rcm were not appended. That source is represented "
        "only by its NOPT full-fertilizer binding; its RCM-N treatment is not "
        "drawn. Bars are within-retained-dataset shares because observation "
        "counts differ by more than an order of magnitude. Rates are discrete "
        "experimental ladders, not a sample from a continuous distribution.",
    )
    return figure, int(len(fp_index)), farmers_practice.arm_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the inorganic N-rate distribution without FP.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    config = load_recipe_config(args.config, project_root=PROJECT_ROOT)
    output = default_output_path(config).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")

    loaded = load_profiled_sources(config)
    figure, explicit_fp, paired_fp = build_figure(config, loaded)
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
    print(f"excluded_literature_farmers_practice_observations={explicit_fp}")
    print(f"omitted_paired_ph_farmers_practice_observations={paired_fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
