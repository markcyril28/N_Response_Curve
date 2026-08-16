from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .config import GrainYieldResponseConfig, RecipeConfigError


@dataclass(frozen=True)
class HeterogeneityResults:
    summary: dict[str, Any]
    centered_frame: pd.DataFrame
    decomposition: pd.DataFrame
    series_intercepts: pd.DataFrame


def _rss(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("Heterogeneity design matrix is rank deficient")
    residuals = y - design @ coefficients
    return coefficients, float(residuals @ residuals), residuals


def _fixed_intercept_fit(
    frame: pd.DataFrame,
    *,
    group_key: str,
) -> dict[str, Any]:
    labels = frame[group_key].astype(str)
    levels = tuple(sorted(labels.unique()))
    indicators = np.column_stack(
        [(labels == level).to_numpy(dtype=float) for level in levels]
    )
    x = frame["n_rate_kg_ha"].to_numpy(dtype=float)
    y = frame["yield_t_ha"].to_numpy(dtype=float)
    design = np.column_stack([indicators, x])
    coefficients, rss, residuals = _rss(design, y)
    return {
        "group_key": group_key,
        "group_count": len(levels),
        "slope_t_ha_per_kg_n_ha": float(coefficients[-1]),
        "rss": rss,
        "rmse_t_ha": float(np.sqrt(np.mean(residuals**2))),
        "intercepts": pd.DataFrame(
            {
                group_key: levels,
                "fixed_intercept_t_ha": coefficients[:-1],
            }
        ),
    }


def analyze_heterogeneity(
    frame: pd.DataFrame,
    *,
    series_key: str,
    study_key: str,
    trial_key: str,
) -> HeterogeneityResults:
    """Separate pooled, study/trial, and response-series structure."""

    if frame.empty:
        raise ValueError("Heterogeneity analysis requires observations")
    required = {
        series_key,
        study_key,
        trial_key,
        "n_rate_kg_ha",
        "yield_t_ha",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Heterogeneity analysis is missing columns: {missing}")

    data = frame.copy()
    x = data["n_rate_kg_ha"].to_numpy(dtype=float)
    y = data["yield_t_ha"].to_numpy(dtype=float)
    pooled_design = np.column_stack([np.ones(len(data)), x])
    pooled_coefficients, pooled_rss, pooled_residuals = _rss(pooled_design, y)

    data["centered_n_rate_kg_ha"] = data["n_rate_kg_ha"] - data.groupby(
        series_key
    )["n_rate_kg_ha"].transform("mean")
    data["centered_yield_t_ha"] = data["yield_t_ha"] - data.groupby(
        series_key
    )["yield_t_ha"].transform("mean")
    centered_x = data["centered_n_rate_kg_ha"].to_numpy(dtype=float)
    centered_y = data["centered_yield_t_ha"].to_numpy(dtype=float)
    denominator = float(centered_x @ centered_x)
    if denominator <= 0:
        raise ValueError("Within-series N-rate variation is required")
    within_slope = float(centered_x @ centered_y / denominator)
    within_residuals = centered_y - within_slope * centered_x
    within_rss = float(within_residuals @ within_residuals)
    within_tss = float(centered_y @ centered_y)
    data["within_series_fitted_centered_yield_t_ha"] = within_slope * centered_x
    data["within_series_residual_t_ha"] = within_residuals

    fits = {
        key: _fixed_intercept_fit(data, group_key=key)
        for key in (study_key, trial_key, series_key)
    }
    decomposition_rows: list[dict[str, Any]] = [
        {
            "structure": "pooled_intercept",
            "group_key": "none",
            "group_count": 1,
            "slope_t_ha_per_kg_n_ha": float(pooled_coefficients[1]),
            "rss": pooled_rss,
            "rmse_t_ha": float(np.sqrt(np.mean(pooled_residuals**2))),
            "sse_reduction_fraction_vs_pooled": 0.0,
        }
    ]
    for label, key in (
        ("study_fixed_intercepts", study_key),
        ("trial_fixed_intercepts", trial_key),
        ("series_fixed_intercepts", series_key),
    ):
        fit = fits[key]
        reduction = (
            float(1.0 - fit["rss"] / pooled_rss) if pooled_rss > 0 else math.nan
        )
        decomposition_rows.append(
            {
                "structure": label,
                "group_key": key,
                "group_count": fit["group_count"],
                "slope_t_ha_per_kg_n_ha": fit[
                    "slope_t_ha_per_kg_n_ha"
                ],
                "rss": fit["rss"],
                "rmse_t_ha": fit["rmse_t_ha"],
                "sse_reduction_fraction_vs_pooled": reduction,
            }
        )
    decomposition = pd.DataFrame(decomposition_rows)

    series_fit = fits[series_key]
    summary = {
        "analysis_role": "descriptive_heterogeneity_not_causal",
        "observations": int(len(data)),
        "study_count": int(data[study_key].nunique(dropna=True)),
        "trial_count": int(data[trial_key].nunique(dropna=True)),
        "series_count": int(data[series_key].nunique(dropna=True)),
        "pooled_slope_t_ha_per_kg_n_ha": float(pooled_coefficients[1]),
        "pooled_rmse_t_ha": float(np.sqrt(np.mean(pooled_residuals**2))),
        "within_series_slope_t_ha_per_kg_n_ha": within_slope,
        "within_series_r_squared": (
            float(1.0 - within_rss / within_tss) if within_tss > 0 else math.nan
        ),
        "within_series_rmse_t_ha": float(
            np.sqrt(np.mean(within_residuals**2))
        ),
        "series_fixed_intercept_sse_reduction_fraction": (
            float(1.0 - series_fit["rss"] / pooled_rss)
            if pooled_rss > 0
            else math.nan
        ),
        "interpretation": (
            "Series fixed intercepts quantify baseline structure; neither the "
            "decomposition nor the within-series slope identifies a causal factor."
        ),
    }

    centered_projection = data[
        [
            series_key,
            study_key,
            trial_key,
            "n_rate_kg_ha",
            "yield_t_ha",
            "centered_n_rate_kg_ha",
            "centered_yield_t_ha",
            "within_series_fitted_centered_yield_t_ha",
            "within_series_residual_t_ha",
        ]
    ].copy()
    return HeterogeneityResults(
        summary=summary,
        centered_frame=centered_projection,
        decomposition=decomposition,
        series_intercepts=series_fit["intercepts"],
    )


def _rscript_path(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    candidates.append(Path(sys.executable).with_name("Rscript"))
    discovered = shutil.which("Rscript")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_symlink() and resolved.is_file():
            return resolved
    raise RecipeConfigError("Rscript executable was not found")


def preflight_mixed_model_engine(
    config: GrainYieldResponseConfig,
    *,
    rscript: str | Path | None = None,
) -> dict[str, Any]:
    if not config.run_mixed_model:
        return {"status": "skipped", "reason_code": "DISABLED_BY_CONFIG"}
    executable = _rscript_path(rscript)
    completed = subprocess.run(
        [
            str(executable),
            "--no-save",
            "--no-restore",
            "-e",
            (
                "stopifnot(requireNamespace('lme4', quietly=TRUE), "
                "requireNamespace('jsonlite', quietly=TRUE)); "
                "cat(as.character(getRversion()))"
            ),
        ],
        cwd=config.project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RecipeConfigError(
            "R mixed-model engine preflight failed: "
            + (completed.stderr.strip() or completed.stdout.strip())[-1000:]
        )
    return {
        "status": "validated",
        "engine": "R/lme4",
        "r_version": completed.stdout.strip(),
        "rscript": str(executable),
    }


def run_mixed_model(
    frame: pd.DataFrame,
    config: GrainYieldResponseConfig,
    *,
    rscript: str | Path | None = None,
) -> dict[str, Any]:
    if not config.run_mixed_model:
        return {
            "status": "skipped",
            "reason_code": "DISABLED_BY_CONFIG",
            "analysis_role": "exploratory_heterogeneity_diagnostic_not_causal",
        }
    executable = _rscript_path(rscript)
    required = [
        "release_record_uid",
        config.series_key,
        config.study_key,
        config.trial_key,
        "n_rate_kg_ha",
        "yield_t_ha",
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RecipeConfigError(f"Mixed-model input is missing columns: {missing}")
    renamed = frame.loc[:, required].rename(
        columns={
            config.series_key: "response_series_uid",
            config.study_key: "study_uid",
            config.trial_key: "trial_uid",
        }
    )
    with tempfile.TemporaryDirectory(prefix="grain_yield_response_mixed_") as temp_dir:
        input_path = Path(temp_dir) / "input.csv"
        output_path = Path(temp_dir) / "output.json"
        renamed.to_csv(input_path, index=False, lineterminator="\n")
        completed = subprocess.run(
            [
                str(executable),
                "--no-save",
                "--no-restore",
                str(config.r_stage_path),
                str(input_path),
                str(output_path),
            ],
            cwd=config.project_root,
            text=True,
            capture_output=True,
            check=False,
        )
        result: dict[str, Any] | None = None
        if output_path.is_file():
            try:
                parsed = json.loads(output_path.read_text(encoding="utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise RecipeConfigError("R mixed-model output is invalid JSON") from exc
            if isinstance(parsed, dict):
                result = parsed
        if completed.returncode != 0 or result is None or result.get("status") != "completed":
            reason = (
                str(result.get("reason_code"))
                if result is not None
                else "R_STAGE_EXECUTION_FAILED"
            )
            detail = (completed.stderr.strip() or completed.stdout.strip())[-1000:]
            if config.mixed_model_required or config.fail_fast:
                raise RecipeConfigError(
                    f"R mixed-model stage failed ({reason}): {detail}"
                )
            return {
                "status": "skipped",
                "reason_code": reason,
                "error_detail": detail,
                "analysis_role": "exploratory_heterogeneity_diagnostic_not_causal",
            }
    if int(result.get("observations", -1)) != len(frame):
        raise RecipeConfigError("R mixed-model output observation count mismatch")
    if int(result.get("series_count", -1)) != frame[config.series_key].nunique():
        raise RecipeConfigError("R mixed-model output series count mismatch")
    result["rscript"] = str(executable)
    return result
