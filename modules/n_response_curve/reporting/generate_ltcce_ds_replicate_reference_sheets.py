#!/usr/bin/env python3
"""Write one LTCCE decade/design/trend sheet per replicate, for one season.

Each output is the compact, no-description version of that season's existing
reference sheet.  The full plate structure is retained: design-era overlays,
six planting-decade overlays, annual zero-N yield, and annual response above
zero N.  Membership is restricted to one recorded source ``Rep`` value at a
time; no curve is fitted and no existing figure is removed.

``--season`` selects which season is drawn and where the sheets land; the dry
season is only the default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_ROOT = PROJECT_ROOT / "modules"
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from n_response_curve.reporting.figure_axis_frames import (  # noqa: E402
    shared_axis_limits,
)
from n_response_curve.reporting.generate_ltcce_replicate_views import (  # noqa: E402
    _partition_trajectory_ids_by_replicate,
)
from n_response_curve.reporting.generate_response_curve_season_clusters import (  # noqa: E402
    PLANTING_YEAR_FIGURE_ONLY_DIRNAME,
    PLANTING_YEAR_REPLICATE_DIRNAME,
    _season_token,
    _write_decades_and_trend,
)
from n_response_curve.reporting.response_curve_clusters import (  # noqa: E402
    read_ltcce_contexts,
    subset_overlay,
)
from n_response_curve.reporting.response_curve_season_clusters import (  # noqa: E402
    FACTOR_DESIGN,
    FACTOR_PLANTING_YEAR,
    build_factor_substructure,
    build_season_clustering,
)
from n_response_curve.reporting.source_config_spec import (  # noqa: E402
    load_source_spec,
)
from n_response_curve.reporting.source_dataset_overlays import (  # noqa: E402
    read_source_dataset_overlay,
)

SOURCE_NAME = "ltcce"
DEFAULT_SEASON = "DS"
SUPPORTED_SEASONS = ("DS", "EWS", "LWS")
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scriptCONFIG.toml"
CLUSTERS_ROOT = (
    PROJECT_ROOT / "WF/04_Response_Curves/z_n_response_full/ltcce/clusters"
)
OUTPUT_STEM = "decades_designs_and_trend_figure_no_description_rep"
EXPECTED_DECADES = ("1960s", "1970s", "1980s", "1990s", "2000s", "2010s")


def default_output_dir(season: str) -> Path:
    """The `by_replicate/` folder inside that season's planting-year plates.

    Derived from the season rather than written out, so this recipe and the
    season-cluster generator cannot drift apart on where the sheets belong.
    """

    return (
        CLUSTERS_ROOT
        / "by_season"
        / _season_token(season)
        / "by_planting_year"
        / PLANTING_YEAR_FIGURE_ONLY_DIRNAME
        / PLANTING_YEAR_REPLICATE_DIRNAME
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--season",
        default=DEFAULT_SEASON,
        choices=SUPPORTED_SEASONS,
        help="Recorded LTCCE season the sheets are drawn for (default: DS)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination for the replicate sheets "
            "(default: the season's own by_planting_year/"
            f"{PLANTING_YEAR_FIGURE_ONLY_DIRNAME}/"
            f"{PLANTING_YEAR_REPLICATE_DIRNAME}/)"
        ),
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.season)
    return args


def main() -> int:
    args = _parse_args()
    season = args.season
    source_path, encoding = load_source_spec(args.config, SOURCE_NAME)
    source_overlay = read_source_dataset_overlay(
        source_path,
        SOURCE_NAME,
        encoding=encoding,
    )
    contexts = read_ltcce_contexts(source_path, encoding=encoding)
    replicate_groups = _partition_trajectory_ids_by_replicate(
        source_overlay,
        contexts,
    )
    limits = shared_axis_limits(source_overlay)

    destination = args.output_dir.resolve()
    if destination.exists() and (not destination.is_dir() or destination.is_symlink()):
        raise RuntimeError("Replicate-sheet destination is not a plain directory")
    destination.mkdir(parents=True, exist_ok=True)

    summaries: list[tuple[str, int, int, Path]] = []
    for replicate, trajectory_ids in replicate_groups.items():
        season_ids = tuple(
            trajectory_id
            for trajectory_id in trajectory_ids
            if contexts[trajectory_id].season.strip().upper() == season
        )
        if not season_ids:
            raise ValueError(f"Recorded Rep={replicate} has no {season} trajectories")

        replicate_overlay = subset_overlay(source_overlay, season_ids)
        result = build_season_clustering(
            replicate_overlay,
            contexts,
            minimum_season_size=3,
        )
        profile = result.profiles.get(season)
        if profile is None or not profile.trajectory_ids:
            raise ValueError(
                f"Recorded Rep={replicate} has no cluster-eligible {season} trajectories"
            )

        # Keep every recorded decade and design even when its per-replicate
        # membership is below the clustering threshold. These are descriptive
        # strata; the reference sheet does not need a within-stratum partition.
        decades = build_factor_substructure(
            result,
            season,
            FACTOR_PLANTING_YEAR,
            minimum_stratum=1,
            minimum_subclustered=10**9,
        )
        decade_levels = tuple(stratum.level for stratum in decades.strata)
        if decade_levels != EXPECTED_DECADES:
            raise ValueError(
                f"Recorded Rep={replicate} has unexpected decade coverage: "
                f"{decade_levels}"
            )
        designs = build_factor_substructure(
            result,
            season,
            FACTOR_DESIGN,
            minimum_stratum=1,
            minimum_subclustered=10**9,
        )
        if len(designs.strata) != 2:
            raise ValueError(
                f"Recorded Rep={replicate} has {len(designs.strata)} design strata; "
                "two are required by this reference-sheet layout"
            )

        output = destination / f"{OUTPUT_STEM}_{replicate}.jpeg"
        records = _write_decades_and_trend(
            result,
            replicate_overlay,
            decades,
            output,
            limits,
            matched_colours=True,
            design_substructure=designs,
            headline_context=f"recorded Rep={replicate}",
            omit_description=True,
        )
        if not output.is_file():
            raise RuntimeError(f"Replicate reference sheet was not written: {output}")
        summaries.append(
            (
                replicate,
                len(profile.trajectory_ids),
                len(records),
                output,
            )
        )

    eligible_total = sum(trajectory_count for _, trajectory_count, _, _ in summaries)
    print(
        f"{SOURCE_NAME} {season}: wrote {len(summaries)} additive replicate "
        f"reference sheets under {destination}"
    )
    for replicate, trajectory_count, year_count, output in summaries:
        print(
            f"  Rep={replicate}: cluster-eligible trajectories={trajectory_count}; "
            f"planting years={year_count}; {output.name}"
        )
    print(f"  reconciled cluster-eligible trajectory total={eligible_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
