"""Synthetic demonstration figures for `test`-mode release packages.

`test` runs in this workspace resolve zero response series, so the phase 5 figure
stage has nothing to draw and `figures/observed/` and `figures/fitted/` come out
empty. A reviewer then has no way to see what the figure contract actually renders.
This module supplies a fixed synthetic stand-in set, written to `figures/sample/`,
that exercises the same `reporting.plots` entry points the real stage uses.

Everything produced here is an **illustration**. The observations are invented, and
where the run carries no reviewed curve gates the missing controls are filled from
demonstration stubs that no accountable party has approved. Sample output is not run
output, is not evidence, and must not be cited. `write_sample_figures` refuses to run
outside `test` mode, so a `full` release package can never contain it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from n_response_curve.analysis.models import MODEL_ORDER, fit_candidate_model
from n_response_curve.reporting.plots import (
    write_observed_series_figures,
    write_response_curve_figures,
)


class SampleFigureError(RuntimeError):
    """A sample-figure request that would leave demonstration output unlabeled."""


SAMPLE_SERIES_UID_PREFIX = "SAMPLE_SYNTHETIC_"
SAMPLE_FIGURE_DIRECTORY = "figures/sample"

_DEMONSTRATION_POLICY_ID = "demonstration-sample-figures-unreviewed-v1"
_DEMONSTRATION_CREDIBILITY_POLICY_ID = "demonstration-sample-credibility-unreviewed-v1"

# (n_rate_kg_ha, yield_t_ha, treatment_text_class) per synthetic series. The four
# shapes cover what the figure contract has to render: an asymptote, a plateau, an
# interior optimum, and a series too sparse for any curve.
SAMPLE_SERIES: Mapping[str, tuple[tuple[float, float, str], ...]] = {
    "SAMPLE_SYNTHETIC_01_asymptotic": (
        (0.0, 3.05, "zero_n"),
        (30.0, 4.10, "mineral_n_rate"),
        (60.0, 4.95, "mineral_n_rate"),
        (90.0, 5.50, "mineral_n_rate"),
        (120.0, 5.78, "mineral_n_rate"),
        (150.0, 5.85, "mineral_n_rate"),
    ),
    "SAMPLE_SYNTHETIC_02_plateau": (
        (0.0, 2.80, "zero_n"),
        (25.0, 3.65, "mineral_n_rate"),
        (50.0, 4.48, "mineral_n_rate"),
        (75.0, 5.30, "mineral_n_rate"),
        (100.0, 5.55, "RCM"),
        (125.0, 5.52, "mineral_n_rate"),
    ),
    "SAMPLE_SYNTHETIC_03_interior_optimum": (
        (0.0, 3.20, "zero_n"),
        (40.0, 4.55, "mineral_n_rate"),
        (80.0, 5.60, "mineral_n_rate"),
        (120.0, 6.05, "NOPT_NPK"),
        (160.0, 5.90, "mineral_n_rate"),
        (200.0, 5.35, "mineral_n_rate"),
    ),
    # Two N levels only: a curve is not supportable, so this one stays observed-only.
    "SAMPLE_SYNTHETIC_04_two_level": (
        (0.0, 3.10, "zero_n"),
        (90.0, 5.25, "RCM"),
    ),
}

# Supplied for one series only, so both the annotated and unannotated renderings show.
SAMPLE_EVIDENCE_ROW: Mapping[str, Any] = {
    "response_series_uid": "SAMPLE_SYNTHETIC_01_asymptotic",
    "evidence_status": "curve_model_credible",
    "evidence_strength": "four_plus_level_curve",
    "curve_shape_class": "asymptotic_diminishing_returns",
    "optimum_status": "interior",
    "agronomic_optimum_n_kg_ha": 132.0,
    "predicted_observed_domain_peak_yield_t_ha": 5.85,
    "finite_maximum_yield_t_ha": None,
    "fitted_asymptote_yield_t_ha": 6.02,
    "supported_max_yield_t_ha": 6.02,
    "maximum_reference_basis": "fitted_asymptote",
    "maximum_proximity_status": "SUPPORTED_ASYMPTOTE",
    "observed_max_gap_to_finite_maximum_t_ha": None,
    "observed_max_gap_to_supported_maximum_t_ha": 0.17,
    "observed_max_attainment_fraction": 0.972,
    "target_yield_status": "not_configured",
    "reason_codes": ("SAMPLE_SYNTHETIC_ILLUSTRATION",),
}

# Mirrors analysis.models initialization strategies. Drift only degrades the sample
# set to observed-only with the reason recorded in INDEX.md; it never fails a run.
_DEMONSTRATION_MODEL_GATES: Mapping[str, Mapping[str, Any]] = {
    "linear": {
        "initialization_strategy": "ordinary_least_squares",
        "parameter_bounds": {"intercept": [-30.0, 30.0], "slope": [-1.0, 1.0]},
        "reportable_shape_classes": [
            "increasing_linear",
            "decreasing_linear",
            "flat_linear",
        ],
    },
    "quadratic": {
        "initialization_strategy": "ordinary_least_squares",
        "parameter_bounds": {
            "intercept": [-30.0, 30.0],
            "slope": [-1.0, 1.0],
            "curvature": [-0.1, 0.1],
        },
        "reportable_shape_classes": [
            "weak_quadratic_curvature",
            "concave_quadratic",
            "diminishing_returns",
            "declining_concave",
            "concave_boundary_peak",
            "accelerating_returns",
            "convex_decline",
            "convex_boundary_minimum",
        ],
    },
    "linear_plateau": {
        "initialization_strategy": "deterministic_data_anchored",
        "parameter_bounds": {
            "intercept": [-30.0, 30.0],
            "slope": [0.0, 1.0],
            "plateau_onset": [0.0, 400.0],
        },
        "reportable_shape_classes": ["plateau", "plateau_without_supported_onset"],
    },
    "quadratic_plateau": {
        "initialization_strategy": "deterministic_data_anchored",
        "parameter_bounds": {
            "baseline": [-30.0, 30.0],
            "gain": [0.0, 60.0],
            "plateau_onset": [0.0, 400.0],
        },
        "reportable_shape_classes": ["plateau", "plateau_without_supported_onset"],
    },
    "mitscherlich": {
        "initialization_strategy": "deterministic_data_anchored",
        "parameter_bounds": {
            "asymptote": [0.0, 30.0],
            "amplitude": [0.0, 60.0],
            "rate": [1.0e-12, 1.0],
        },
        "reportable_shape_classes": [
            "asymptotic_diminishing_returns",
            "asymptotic_without_supported_asymptote",
        ],
    },
}

_DEMONSTRATION_GATE_TOLERANCES: Mapping[str, Any] = {
    "allow_boundary_parameters": True,
    "optimizer_tolerance": 1.0e-8,
    "optimizer_max_iterations": 10000,
    "parameter_boundary_relative_tolerance": 1.0e-8,
    "optimum_boundary_tolerance_n_kg_ha": 1.0e-4,
    "flat_response_tolerance_t_ha": 1.0e-6,
}

# Deliberately permissive: a demonstration must render every shape rather than
# adjudicate it. These thresholds decide nothing about real data — the sample path
# is the only caller, and it runs in `test` mode alone.
_DEMONSTRATION_SUBSTITUTES: Mapping[str, Any] = {
    "model_gate_policy": {
        "policy_id": _DEMONSTRATION_POLICY_ID,
        "review_status": "approved",
        "models": {
            name: {**gate, **_DEMONSTRATION_GATE_TOLERANCES}
            for name, gate in _DEMONSTRATION_MODEL_GATES.items()
        },
    },
    "model_credibility_policy": {
        "policy_id": _DEMONSTRATION_CREDIBILITY_POLICY_ID,
        "review_status": "approved",
        "maximum_normalized_rmse": 10.0,
        "maximum_parameter_influence_relative_shift": 1.0e6,
        "minimum_influence_folds": 2,
        "maximum_parameter_relative_standard_error": 1.0e6,
        "parameter_scale_floor": 0.001,
        "maximum_observed_step_decline_t_ha": 1.0e6,
    },
    "restricted_fit_models": ["linear", "quadratic"],
}


@dataclass(frozen=True)
class SampleFigureSet:
    """Written demonstration figures and the audit record that labels them."""

    paths: tuple[Path, ...]
    summary: Mapping[str, Any]


def demonstration_model_policy(
    base_policy: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    """Fill only the reviewed curve controls the run does not already carry.

    A run that has a bound analysis policy keeps its own reviewed gates; the
    substitution list names every control that came from a demonstration stub, so the
    index and manifest can say exactly what was unreviewed.
    """

    policy = dict(base_policy)
    substituted: list[str] = []
    for key, value in _DEMONSTRATION_SUBSTITUTES.items():
        if policy.get(key) is None:
            policy[key] = value
            substituted.append(key)
    return policy, tuple(substituted)


def _records_for(series_uid: str) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "record_uid": f"{series_uid}__obs{index:02d}",
            "response_series_uid": series_uid,
            "n_rate_kg_ha": n_rate,
            "yield_t_ha": yield_value,
            "treatment_text_class": treatment,
        }
        for index, (n_rate, yield_value, treatment) in enumerate(SAMPLE_SERIES[series_uid])
    )


_INDEX_HEADER = (
    "# Sample figures — synthetic illustrations, not run output\n"
    "\n"
    "This `test`-mode run resolved no response series, so the figure stage had no\n"
    "observation to draw. The figures beside this file were rendered from invented\n"
    "observations through the same `reporting.plots` entry points phase 5 uses, so the\n"
    "figure contract can be reviewed while the real series resolution is unavailable.\n"
    "\n"
    "**Nothing here is evidence.** No value below describes any trial, and no sample\n"
    "figure may be cited, published, or copied into a report as a result.\n"
)


def _write_index(
    output_root: Path,
    rows: Sequence[tuple[str, ...]],
    *,
    formats: Sequence[str],
    substituted_controls: Sequence[str],
) -> Path:
    """Map the hash-named figure files back to their series and model."""

    lines = [_INDEX_HEADER]
    if substituted_controls:
        lines.append(
            "Reviewed curve controls this run does not carry were filled from "
            "unreviewed demonstration stubs so the fitted renderings could be produced: "
            + ", ".join(f"`{control}`" for control in substituted_controls)
            + ".\n"
        )
    lines.append(f"Formats written per figure: {', '.join(formats)}.\n")
    lines.append(
        "| series | model | status | shape class | optimum | file stem |\n"
        "| --- | --- | --- | --- | --- | --- |"
    )
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    index_path = output_root / "INDEX.md"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def write_sample_figures(
    output_root: str | Path,
    *,
    run_mode: str,
    model_policy: Mapping[str, Any],
    formats: Sequence[str],
) -> SampleFigureSet:
    """Render the synthetic demonstration set and the index that labels it.

    Refuses any mode but `test`: demonstration content must never reach an
    authoritative package.
    """

    if run_mode != "test":
        raise SampleFigureError(
            f"Sample figures are a test-mode demonstration; refusing run_mode={run_mode!r}"
        )
    root = Path(output_root)
    policy, substituted = demonstration_model_policy(model_policy)

    written: list[Path] = []
    index_rows: list[tuple[str, ...]] = []
    unsupported: list[str] = []
    observed_count = 0
    fitted_count = 0

    for series_uid in SAMPLE_SERIES:
        records = _records_for(series_uid)
        evidence_row = (
            SAMPLE_EVIDENCE_ROW
            if series_uid == SAMPLE_EVIDENCE_ROW["response_series_uid"]
            else None
        )
        observed_paths = write_observed_series_figures(
            records,
            series_uid,
            output_root=root / "observed",
            formats=formats,
            evidence_row=evidence_row,
        )
        written.extend(observed_paths)
        observed_count += 1
        index_rows.append(
            (series_uid, "—", "observed_only", "n/a", "n/a", f"observed/{observed_paths[0].stem}")
        )

        n_rates = [row["n_rate_kg_ha"] for row in records]
        yields = [row["yield_t_ha"] for row in records]
        record_uids = [row["record_uid"] for row in records]
        for model_name in MODEL_ORDER:
            attempt = fit_candidate_model(
                series_uid,
                n_rates,
                yields,
                record_uids=record_uids,
                model_name=model_name,
                policy=policy,
            )
            if attempt.status != "fitted":
                reasons = ",".join(attempt.reason_codes) or "none"
                unsupported.append(f"{series_uid}/{model_name}: {attempt.status} ({reasons})")
                index_rows.append(
                    (
                        series_uid,
                        model_name,
                        attempt.status,
                        "n/a",
                        "n/a",
                        "(no figure — curve not supportable)",
                    )
                )
                continue
            fitted_paths = write_response_curve_figures(
                records,
                attempt,
                output_root=root / "fitted",
                formats=formats,
                evidence_row=evidence_row,
            )
            written.extend(fitted_paths)
            fitted_count += 1
            index_rows.append(
                (
                    series_uid,
                    model_name,
                    attempt.status,
                    str(attempt.curve_shape_class or "—"),
                    str(attempt.optimum_status or "—"),
                    f"fitted/{fitted_paths[0].stem}",
                )
            )

    index_path = _write_index(
        root,
        index_rows,
        formats=tuple(formats),
        substituted_controls=substituted,
    )
    written.append(index_path)

    summary = {
        "status": "synthetic_demonstration_substituted",
        "reason": "no response series resolved, so no observed or fitted figure was renderable",
        "scientific_status": "illustration_only_not_run_output_not_citable",
        "directory": SAMPLE_FIGURE_DIRECTORY,
        "index_path": f"{SAMPLE_FIGURE_DIRECTORY}/INDEX.md",
        "formats": list(formats),
        "series_uids": list(SAMPLE_SERIES),
        "observed_figure_count": observed_count,
        "fitted_figure_count": fitted_count,
        "artifact_count": len(written),
        "unsupported_attempts": unsupported,
        "unreviewed_substituted_controls": list(substituted),
        "demonstration_policy_ids": {
            "model_gate_policy": _DEMONSTRATION_POLICY_ID,
            "model_credibility_policy": _DEMONSTRATION_CREDIBILITY_POLICY_ID,
        },
    }
    return SampleFigureSet(paths=tuple(written), summary=summary)


__all__ = [
    "SAMPLE_EVIDENCE_ROW",
    "SAMPLE_FIGURE_DIRECTORY",
    "SAMPLE_SERIES",
    "SAMPLE_SERIES_UID_PREFIX",
    "SampleFigureError",
    "SampleFigureSet",
    "demonstration_model_policy",
    "write_sample_figures",
]
