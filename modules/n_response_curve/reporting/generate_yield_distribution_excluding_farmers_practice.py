#!/usr/bin/env python3
"""Render the combined grain-yield distribution with all FP evidence excluded.

The governed combined figure uses the registered NOPT yield binding for
``ph_combined_nopt_rcm`` but still contains the literature-extracted rows whose
experiment type is ``Farmer's Practice``.  This variant removes those rows as
well, so neither the PH sibling FP arm nor the literature FP records contribute
to the bars, medians, counts, axes, or caption.

The output location follows the requested descriptive-statistics agronomic
directory.  The variant is not part of the bundle's current artifact contract,
so it is not represented in that bundle's manifest or checksum ledger.
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
    / "yield_distribution_excluding_farmers_practice.jpeg"
)


def build_figure(
    config: DescriptiveStatisticsConfig,
    loaded: LoadedSources,
) -> tuple[plt.Figure, int]:
    """Build the no-FP panel and return it with the excluded-row count."""

    recorded = dsf._observations(loaded, config)
    fp_index = dsf._core_trial_farmers_practice_index(loaded, recorded)
    if fp_index.empty:
        raise ValueError("No literature-extracted Farmer's Practice rows were found")

    observations = recorded.drop(index=fp_index).reset_index(drop=True)
    names = dsf._ordered_sources(loaded, observations["source_name"])
    shared = dsf._yield_distribution_axes(
        observations,
        names=names,
        config=config,
    )
    if shared is None:
        raise ValueError("The retained observations cannot define histogram axes")

    colours = dsf._source_colours(loaded)
    figure, axis = dsf._figure(config)
    converted = 0
    for index, name in enumerate(names):
        subset = observations.loc[observations["source_name"] == name]
        values = subset["yield_t_ha"].to_numpy(dtype=float)
        if values.size == 0:
            continue
        converted += int(
            (
                subset["yield_unit_lineage"].astype(str)
                == "converted_from_kg_ha"
            ).sum()
        )
        median = float(np.median(values))
        dsf._share_histogram(
            axis,
            values,
            edges=shared.edges,
            colour=colours[name],
            label=f"{dsf._legend_label(name)} — {values.size:,} observations",
        )
        axis.axvline(
            median,
            color=colours[name],
            linestyle="--",
            linewidth=1.5,
        )
        axis.annotate(
            f"median {median:.2f}",
            xy=(median, dsf._YIELD_CALLOUT_SLOTS[index]),
            xycoords=("data", "axes fraction"),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color=colours[name],
            bbox=dsf._ANNOTATION_BOX,
        )

    axis.set_xlim(*shared.xlim)
    axis.set_ylim(*shared.ylim)
    axis.set(
        title=(
            "Recorded grain-yield distribution per dataset — "
            "Farmer's Practice excluded"
        ),
        xlabel="Grain yield (t ha⁻¹)",
        ylabel="Share of each dataset's retained observations (%)",
    )
    axis.legend(fontsize=8, loc="upper right")
    axis.grid(alpha=0.2)

    lineage = (
        f"{converted:,} retained observations reached t ha⁻¹ by conversion "
        "from kg ha⁻¹"
        if converted
        else "every retained observation is recorded natively in t ha⁻¹"
    )
    dsf._footnote(
        figure,
        f"Farmer's Practice (FP) is excluded: {len(fp_index):,} "
        "literature-extracted FP observations were removed, and "
        "ph_combined_nopt_rcm is represented only by its NOPT "
        "full-fertilizer arm. Bars are within-dataset shares; dashed lines "
        f"mark each retained dataset's median. Unit lineage: {lineage}.",
        minimum_lines=dsf._YIELD_DISTRIBUTION_FOOTNOTE_LINES,
    )
    return figure, int(len(fp_index))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the combined grain-yield distribution without FP.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    # This figure has one fixed project destination.  Do not accept a caller-
    # supplied path: that previously allowed runs to recreate copies outside
    # the required descriptive-statistics agronomic directory.
    output = DEFAULT_OUTPUT_PATH.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")

    config = load_recipe_config(args.config, project_root=PROJECT_ROOT)
    loaded = load_profiled_sources(config)
    figure, excluded = build_figure(config, loaded)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
