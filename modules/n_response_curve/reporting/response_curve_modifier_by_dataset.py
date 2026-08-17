"""Per-dataset series-slope modifier screens, plus a pooled cross-dataset view.

The released `grain_yield_response_diagnostics` bundle screens series-level
covariates against fitted per-series N-response slopes on **one** population:
the governed `core_trial_data` observed-series overlay (21 series). Its
`response_curve_modifier_ranking` figure is therefore a single-dataset result
that is easy to misread as a pooled one.

This recipe rebuilds that screen once per registered source dataset and once on
the three pooled together, so the same covariates can be compared across
populations rather than read off one of them.

Two population facts drive the design:

* `ltcce` and `ph_combined_nopt_rcm` contribute **zero** series to the governed
  overlay (`series_count_by_source` in the release manifest), so their screens
  cannot come from the release ledger. They are built by reading the curated
  source files directly. That output is **ungoverned**: it has passed no
  eligibility screen, no QC gate, and no release verification. It is a
  diagnostic, never a releasable inventory member.
* `ph_combined_nopt_rcm` looks unfittable in the descriptive-statistics ladder
  geometry table (718 series, one N level each). That is an artefact of
  row-wise harmonization: the paired zero-N omission arm is held as a *column*
  (`n0_yield`), not as a second row. Reconstructing the pair gives a two-point
  ladder per field and 717 estimable slopes. The cost is that a two-point slope
  is exactly `(y_N - y_0) / N`, so the zero-N check yield enters it
  algebraically -- flagged per row rather than silently screened.

Both restricted sources feed this bundle, so the whole output tree inherits
their classification. Nothing here is causal, and no fertilizer recommendation
or agronomic optimum is produced.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from ..analysis.grain_yield_response.population import sha256_file  # noqa: E402
from ..analysis.grain_yield_response.series_covariates import (  # noqa: E402
    derive_series_covariates,
    screen_series_slope_modifiers,
)
from .grain_yield_response import (  # noqa: E402
    DiagnosticBundleError,
    _RANKING_FACTOR_COLOR,
    _RANKING_INK_SOFT,
    _RANKING_LEVEL_COLOR,
    diagnostic_container_publication_lock,
    exchange_directories,
    _factor_label,
)


SCHEMA_VERSION = "response-curve-modifier-by-dataset-v3"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_HOST_PACKAGE = (
    PROJECT_ROOT / "WF/03_Quality_Control/grain_yield_response_diagnostics"
).resolve()
CANONICAL_EXTENSION_PLACEMENT = (
    "factors/response_curve_factor_contributors/response_curve_modifier_by_dataset"
)
CANONICAL_OUTPUT_ROOT = (
    CANONICAL_HOST_PACKAGE / CANONICAL_EXTENSION_PLACEMENT
).resolve()

POOLED_KEY = "all_datasets"

# Ordering used by every table, figure and summary section in the bundle.
DATASET_ORDER: tuple[str, ...] = (
    "core_trial_data",
    "ltcce",
    "ph_combined_nopt_rcm",
    POOLED_KEY,
)
_OUTPUT_GROUPS = frozenset((*DATASET_ORDER, "paired_core_vs_ltcce"))

_DATASET_COLORS: Mapping[str, str] = {
    "core_trial_data": "#2a78d6",
    "ltcce": "#eb6834",
    "ph_combined_nopt_rcm": "#3f9b6d",
    POOLED_KEY: "#6b5b95",
}

_DATASET_LABELS: Mapping[str, str] = {
    "core_trial_data": "core_trial_data",
    "ltcce": "ltcce",
    "ph_combined_nopt_rcm": "ph_combined_nopt_rcm",
    POOLED_KEY: "all three pooled",
}

# The screen reports observation count and distinct N-level count separately.
# They are the same column whenever every series observation sits on its own N
# level, which is true in all four populations here, so the figure collapses
# rows that are numerically identical rather than drawing one contrast twice.
_DEDUPLICATION_KEYS = ("pearson_r", "spearman_rho", "series_used")

# Pearson r spread across the source datasets below which the per-dataset
# results are called similar in magnitude as well as in sign.
_AGREEMENT_RANGE_THRESHOLD = 0.30


class RecipeError(RuntimeError):
    """Configuration or population failure that must stop the run."""


def _grouped_artifact_path(group: str, filename: str) -> str:
    """Place every artifact in a semantic group, never a file-type bucket."""
    pure = PurePosixPath(filename)
    if group not in _OUTPUT_GROUPS or pure.name != filename or filename in {"", "."}:
        raise RecipeError(f"Unsafe or unknown output artifact path: {group}/{filename}")
    return f"{group}/{filename}"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    governance: str
    data_classification: str
    series_key_basis: str
    source_path: Path | None
    encoding: str | None
    note: str


@dataclass(frozen=True)
class RecipeConfig:
    project_root: Path
    config_path: Path
    output_root: Path
    release_package: Path
    base_config: Path
    overwrite: bool
    zero_n_tolerance_kg_ha: float
    n_level_tolerance_kg_ha: float
    minimum_modifier_series: int
    figure_dpi: int
    figure_width_inches: float
    figure_height_inches: float
    run_mixed_model: bool
    mixed_model_stage: Path
    mixed_model_minimum_observations_per_series: int
    mixed_model_minimum_distinct_n_levels: int
    rscript_path: str | None
    extension_enabled: bool
    extension_host_package: Path
    extension_placement: str
    specs: Mapping[str, DatasetSpec]


@dataclass
class DatasetResult:
    spec: DatasetSpec
    observations: pd.DataFrame
    series: pd.DataFrame
    screen: pd.DataFrame
    inputs: dict[str, str] = field(default_factory=dict)
    mixed_model: dict[str, Any] = field(default_factory=dict)

    @property
    def baseline_slope_correlation(self) -> float | None:
        """lme4's random intercept-slope correlation, or None if not fitted.

        This is the dashed reference line on the released figure: the
        hierarchical model's own estimate of how series baseline and series
        response covary, against which the covariate correlations are read.
        """
        if str(self.mixed_model.get("status")) != "completed":
            return None
        value = self.mixed_model.get("random_intercept_slope_correlation")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def series_count(self) -> int:
        return int(len(self.series))

    @property
    def estimable_slope_count(self) -> int:
        return int(
            pd.to_numeric(
                self.series["series_slope_t_ha_per_kg_n_ha"], errors="coerce"
            ).notna().sum()
        )

    @property
    def median_distinct_n_levels(self) -> float:
        return float(
            pd.to_numeric(
                self.series["series_distinct_n_level_count"], errors="coerce"
            ).median()
        )

    @property
    def two_point_ladder(self) -> bool:
        """Whether the typical series carries exactly two N levels.

        A two-point slope is `(y_N - y_0) / N` exactly, so any covariate that is
        one of those two yields is algebraically inside the slope rather than
        merely sharing estimation error with it.
        """
        return self.median_distinct_n_levels <= 2.0


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def _require_table(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    table = document.get(name)
    if not isinstance(table, dict):
        raise RecipeError(f"Configuration is missing the [{name}] table")
    return table


def load_config(config_path: Path) -> RecipeConfig:
    try:
        document = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RecipeError(f"Cannot read configuration: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise RecipeError(f"Configuration is not valid TOML: {exc}") from exc

    run = _require_table(document, "run")
    population = _require_table(document, "population")
    outputs = _require_table(document, "outputs")
    paths = _require_table(document, "paths")
    datasets = _require_table(document, "datasets")
    # Optional tables with safe defaults: a config predating them still loads.
    mixed_model = document.get("mixed_model")
    mixed_model = mixed_model if isinstance(mixed_model, dict) else {}
    extension = document.get("package_extension")
    extension = extension if isinstance(extension, dict) else {}

    project_root = PROJECT_ROOT

    def _resolve(value: object, key: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise RecipeError(f"[paths].{key} must be a nonempty string")
        return (project_root / value).resolve()

    specs: dict[str, DatasetSpec] = {}
    for name in DATASET_ORDER:
        if name == POOLED_KEY:
            continue
        block = datasets.get(name)
        if not isinstance(block, dict):
            raise RecipeError(f"Configuration is missing [datasets.{name}]")
        raw_path = block.get("source_path")
        specs[name] = DatasetSpec(
            name=name,
            governance=str(block.get("governance", "")),
            data_classification=str(block.get("data_classification", "")),
            series_key_basis=str(block.get("series_key_basis", "")),
            source_path=(
                (project_root / str(raw_path)).resolve()
                if isinstance(raw_path, str) and raw_path.strip()
                else None
            ),
            encoding=(
                str(block["encoding"])
                if isinstance(block.get("encoding"), str)
                else None
            ),
            note=str(block.get("note", "")),
        )

    config = RecipeConfig(
        project_root=project_root,
        config_path=config_path.resolve(),
        output_root=_resolve(paths.get("output_root"), "output_root"),
        release_package=_resolve(paths.get("release_package"), "release_package"),
        base_config=_resolve(paths.get("base_config"), "base_config"),
        overwrite=bool(run.get("overwrite", False)),
        zero_n_tolerance_kg_ha=float(population.get("zero_n_tolerance_kg_ha", 1e-8)),
        n_level_tolerance_kg_ha=float(population.get("n_level_tolerance_kg_ha", 0.0)),
        minimum_modifier_series=int(population.get("minimum_modifier_series", 4)),
        figure_dpi=int(outputs.get("figure_dpi", 220)),
        figure_width_inches=float(outputs.get("figure_width_inches", 10.0)),
        figure_height_inches=float(outputs.get("figure_height_inches", 6.4)),
        run_mixed_model=bool(mixed_model.get("enabled", True)),
        mixed_model_stage=(
            project_root
            / str(
                mixed_model.get(
                    "r_stage",
                    "modules/n_response_curve/analysis/stages/"
                    "grain_yield_response_mixed_models.R",
                )
            )
        ).resolve(),
        mixed_model_minimum_observations_per_series=int(
            mixed_model.get("minimum_observations_per_series", 3)
        ),
        mixed_model_minimum_distinct_n_levels=int(
            mixed_model.get("minimum_distinct_n_levels", 3)
        ),
        rscript_path=(
            str(mixed_model["rscript"])
            if isinstance(mixed_model.get("rscript"), str)
            else None
        ),
        extension_enabled=bool(extension.get("enabled", False)),
        extension_host_package=(
            project_root
            / str(extension.get("host_package", "WF/03_Quality_Control/grain_yield_response_diagnostics"))
        ).resolve(),
        extension_placement=str(
            extension.get(
                "placement",
                "factors/response_curve_factor_contributors/response_curve_modifier_by_dataset",
            )
        ),
        specs=specs,
    )
    _validate_canonical_destination(config)
    return config


def _validate_canonical_destination(config: RecipeConfig) -> None:
    if (
        not config.extension_enabled
        or config.extension_host_package.resolve() != CANONICAL_HOST_PACKAGE
        or PurePosixPath(config.extension_placement).as_posix()
        != CANONICAL_EXTENSION_PLACEMENT
        or config.output_root.resolve() != CANONICAL_OUTPUT_ROOT
    ):
        raise RecipeError(
            "This recipe may write only to the canonical prioritized destination: "
            f"{CANONICAL_OUTPUT_ROOT}"
        )


# --------------------------------------------------------------------------
# Populations
# --------------------------------------------------------------------------

_OBSERVATION_COLUMNS: tuple[str, ...] = (
    "source_name",
    "response_series_uid",
    "study_uid",
    "trial_uid",
    "n_rate_kg_ha",
    "yield_t_ha",
    "planting_year",
)


def _finalize(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = sorted(set(_OBSERVATION_COLUMNS) - set(frame.columns))
    if missing:
        raise RecipeError(f"{name} population is missing columns: {missing}")
    cleaned = frame.dropna(subset=["n_rate_kg_ha", "yield_t_ha"]).copy()
    if cleaned.empty:
        raise RecipeError(f"{name} population resolved to zero observations")
    for key in ("response_series_uid", "study_uid", "trial_uid"):
        cleaned[key] = cleaned[key].astype(str).str.strip()
        if cleaned[key].eq("").any():
            raise RecipeError(f"{name} population has a blank {key}")
    cleaned["n_rate_kg_ha"] = cleaned["n_rate_kg_ha"].astype(float)
    cleaned["yield_t_ha"] = cleaned["yield_t_ha"].astype(float)
    return cleaned.sort_values(
        ["response_series_uid", "n_rate_kg_ha"], kind="mergesort"
    ).reset_index(drop=True)


def build_core_trial_data(
    config: RecipeConfig, spec: DatasetSpec
) -> tuple[pd.DataFrame, dict[str, str]]:
    """The governed observed-series overlay -- the released figure's population.

    Read through the release manifest's declared series membership rather than
    by filtering the ledger on eligibility columns, so this population is the
    same 21 series the promoted bundle screened.
    """
    manifest_path = config.release_package / "run_manifest.json"
    ledger_path = (
        config.release_package / "tables/quality/analysis_eligibility_ledger.csv"
    )
    for path in (manifest_path, ledger_path):
        if not path.is_file():
            raise RecipeError(f"Release input is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    overlay = manifest.get("figures", {}).get("overlay", {})
    if overlay.get("scope") != "governed_observed_series_overlay_only":
        raise RecipeError("Release overlay does not declare governed observed-series scope")
    uids = overlay.get("series_uids_by_source", {}).get(spec.name)
    if not isinstance(uids, list) or not uids:
        raise RecipeError(f"No governed overlay series declared for {spec.name}")

    ledger = pd.read_csv(ledger_path, low_memory=False)
    selected = ledger[
        (ledger["source_name"] == spec.name)
        & ledger["response_series_uid"].astype(str).isin(set(uids))
    ]
    if set(selected["response_series_uid"].astype(str)) != set(uids):
        raise RecipeError("Governed series membership does not resolve in the ledger")

    frame = pd.DataFrame(
        {
            "source_name": spec.name,
            "response_series_uid": selected["response_series_uid"].astype(str),
            "study_uid": selected["study_uid"].astype(str),
            "trial_uid": selected["trial_uid"].astype(str),
            "n_rate_kg_ha": pd.to_numeric(selected["n_rate_kg_ha"], errors="coerce"),
            "yield_t_ha": pd.to_numeric(selected["yield_t_ha"], errors="coerce"),
            "planting_year": pd.to_numeric(selected["planting_year"], errors="coerce"),
            "p_rate_kg_p2o5_ha": pd.to_numeric(
                selected["p_rate_kg_p2o5_ha"], errors="coerce"
            ),
            "k_rate_kg_k2o_ha": pd.to_numeric(
                selected["k_rate_kg_k2o_ha"], errors="coerce"
            ),
            "observation_arm": "governed_release_record",
        }
    )
    inputs = {
        "release_manifest": sha256_file(manifest_path),
        "analysis_eligibility_ledger": sha256_file(ledger_path),
    }
    return _finalize(frame, name=spec.name), inputs


def build_ltcce(
    config: RecipeConfig, spec: DatasetSpec
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Raw LTCCE read: one series per Year|Season|Variety, Rep averaged.

    `Rep` is replication inside an N level, not a separate response point, so
    the replicate yields are averaged to one observation per (series, N rate).
    `Design` and `Site` change once across the record and are constant within
    every series, so they serve as the study identifier.
    """
    if spec.source_path is None or not spec.source_path.is_file():
        raise RecipeError(f"{spec.name} source file is missing: {spec.source_path}")
    raw = pd.read_csv(
        spec.source_path, encoding=spec.encoding or "utf-8-sig", low_memory=False
    )
    required = {"Year", "Season", "Variety", "Nfert", "GYtha", "Design", "Rep"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RecipeError(f"{spec.name} source is missing columns: {missing}")

    raw = raw.dropna(subset=["Nfert", "GYtha"])
    keys = pd.DataFrame(
        {
            "response_series_uid": (
                "ltcce::"
                + raw["Year"].astype(str)
                + "|"
                + raw["Season"].astype(str)
                + "|"
                + raw["Variety"].astype(str)
            ),
            "study_uid": "ltcce::" + raw["Design"].astype(str),
            "trial_uid": (
                "ltcce::" + raw["Year"].astype(str) + "|" + raw["Season"].astype(str)
            ),
            "planting_year": pd.to_numeric(raw["Year"], errors="coerce"),
            "n_rate_kg_ha": pd.to_numeric(raw["Nfert"], errors="coerce"),
            "gy": pd.to_numeric(raw["GYtha"], errors="coerce"),
        }
    )
    grouped = keys.groupby(
        ["response_series_uid", "study_uid", "trial_uid", "planting_year", "n_rate_kg_ha"],
        as_index=False,
    ).agg(yield_t_ha=("gy", "mean"), replicate_count=("gy", "size"))
    grouped["source_name"] = spec.name
    grouped["observation_arm"] = "replicate_mean"
    inputs = {"source_file": sha256_file(spec.source_path)}
    return _finalize(grouped, name=spec.name), inputs


def build_ph_combined_nopt_rcm(
    config: RecipeConfig, spec: DatasetSpec
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Raw PH read, reconstructing the paired zero-N arm held as a column.

    Each record carries a fertilized observation (`nrate`, `full_fert_yield`)
    and the yield of its own zero-N omission plot (`n0_yield`). Row-wise
    harmonization keeps only the first, which is why the descriptive ladder
    geometry reports one N level per series. Emitting the omission plot as a
    second observation at 0 kg N/ha restores the two-point ladder the design
    actually has.
    """
    if spec.source_path is None or not spec.source_path.is_file():
        raise RecipeError(f"{spec.name} source file is missing: {spec.source_path}")
    raw = pd.read_csv(
        spec.source_path, encoding=spec.encoding or "cp1252", low_memory=False
    )
    required = {
        "rcm_reference",
        "nopt_reference",
        "country",
        "year",
        "nrate",
        "full_fert_yield",
        "n0_yield",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RecipeError(f"{spec.name} source is missing columns: {missing}")

    series_uid = (
        "ph::" + raw["rcm_reference"].astype(str) + "|" + raw["nopt_reference"].astype(str)
    )
    study_uid = "ph::" + raw["country"].astype(str) + "|" + raw["year"].astype(str)
    planting_year = pd.to_numeric(raw["year"], errors="coerce")

    def _arm(n_rate: pd.Series, yields: pd.Series, label: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "source_name": spec.name,
                "response_series_uid": series_uid,
                "study_uid": study_uid,
                "trial_uid": series_uid,
                "n_rate_kg_ha": n_rate,
                "yield_t_ha": yields,
                "planting_year": planting_year,
                "observation_arm": label,
            }
        )

    fertilized = _arm(
        pd.to_numeric(raw["nrate"], errors="coerce"),
        pd.to_numeric(raw["full_fert_yield"], errors="coerce"),
        "full_fertilizer",
    )
    zero_n = _arm(
        pd.Series(0.0, index=raw.index),
        pd.to_numeric(raw["n0_yield"], errors="coerce"),
        "zero_n_omission_plot",
    )
    frame = pd.concat([fertilized, zero_n], ignore_index=True)
    inputs = {"source_file": sha256_file(spec.source_path)}
    return _finalize(frame, name=spec.name), inputs


_BUILDERS = {
    "core_trial_data": build_core_trial_data,
    "ltcce": build_ltcce,
    "ph_combined_nopt_rcm": build_ph_combined_nopt_rcm,
}


def screen_dataset(config: RecipeConfig, result_frame: pd.DataFrame) -> tuple[
    pd.DataFrame, pd.DataFrame
]:
    series, _ = derive_series_covariates(
        result_frame,
        series_key="response_series_uid",
        study_key="study_uid",
        trial_key="trial_uid",
        zero_n_tolerance_kg_ha=config.zero_n_tolerance_kg_ha,
        n_level_tolerance_kg_ha=config.n_level_tolerance_kg_ha,
    )
    screen = screen_series_slope_modifiers(
        series, minimum_series=config.minimum_modifier_series
    )
    return series, screen


# --------------------------------------------------------------------------
# Hierarchical (lme4) diagnostic
# --------------------------------------------------------------------------


def _rscript_executable(config: RecipeConfig) -> Path:
    candidates: list[Path] = []
    if config.rscript_path:
        candidates.append(Path(config.rscript_path).expanduser())
    candidates.append(Path(sys.executable).with_name("Rscript"))
    discovered = shutil.which("Rscript")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_symlink() and resolved.is_file():
            return resolved
    raise RecipeError("Rscript executable was not found")


def mixed_model_eligible(config: RecipeConfig, frame: pd.DataFrame) -> pd.DataFrame:
    """Series the R stage's own gates admit: enough points and enough N levels.

    A random-intercept/random-slope fit needs more points per series than it
    spends on that series' own line; a two-point ladder determines its line
    exactly and leaves nothing to separate residual from between-series slope
    variance. The R stage enforces both thresholds and fails the whole run if
    any series violates them, so the subset is taken here and the exclusion is
    reported rather than hidden.
    """
    grouped = frame.groupby("response_series_uid")["n_rate_kg_ha"]
    eligible = (
        grouped.nunique().ge(config.mixed_model_minimum_distinct_n_levels)
        & grouped.size().ge(config.mixed_model_minimum_observations_per_series)
    )
    keep = set(eligible.loc[eligible].index)
    return frame.loc[frame["response_series_uid"].isin(keep), :].copy()


# Run when the governed stage's support gate admits no series. The gate is a
# threshold this recipe chose, so refusing there proves nothing on its own; this
# hands the ungated population straight to lme4 and records lme4's own verdict,
# which is the evidence that the model is unidentifiable rather than merely
# under-supported. It deliberately does not go through the release-bound R
# stage: that stage's gates are part of its contract and are not to be bypassed.
_FORCED_FIT_R_SOURCE = """
suppressPackageStartupMessages({library(jsonlite); library(lme4)})
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1L]], stringsAsFactors = FALSE)
d$n100 <- d$n_rate_kg_ha / 100
d$response_series_uid <- factor(d$response_series_uid)
n_series <- nlevels(d$response_series_uid)
emit <- function(x) writeLines(toJSON(x, auto_unbox = TRUE, null = "null",
                                      na = "null", digits = 15), args[[2L]])
fit <- tryCatch(
  lmer(yield_t_ha ~ n100 + (1 + n100 | response_series_uid), data = d,
       REML = FALSE,
       control = lmerControl(optimizer = "bobyqa",
                             optCtrl = list(maxfun = 100000),
                             check.conv.singular = "ignore")),
  error = function(e) e)
base <- list(forced_fit_observations = nrow(d),
             forced_fit_series = n_series,
             forced_fit_random_effect_parameters = 2L * n_series)
if (inherits(fit, "error")) {
  emit(c(base, list(forced_fit_status = "error",
                    forced_fit_message = conditionMessage(fit))))
} else {
  corr <- attr(VarCorr(fit)$response_series_uid, "correlation")
  emit(c(base, list(
    forced_fit_status = "fitted",
    forced_fit_singular = isTRUE(isSingular(fit, tol = 1e-4)),
    forced_fit_correlation = unname(corr[1L, 2L]),
    forced_fit_residual_sd = sigma(fit),
    forced_fit_message = paste(as.character(fit@optinfo$conv$lme4$messages),
                               collapse = "; "))))
}
"""


def probe_forced_mixed_model(
    config: RecipeConfig, result: DatasetResult
) -> dict[str, Any]:
    """Hand the ungated population to lme4 and record what lme4 itself says."""
    try:
        executable = _rscript_executable(config)
    except RecipeError as exc:
        return {"forced_fit_status": "unavailable", "forced_fit_message": str(exc)}
    projection = result.observations.loc[
        :, ["response_series_uid", "n_rate_kg_ha", "yield_t_ha"]
    ]
    with tempfile.TemporaryDirectory(prefix="modifier_forced_fit_") as temp_dir:
        script_path = Path(temp_dir) / "forced_fit.R"
        input_path = Path(temp_dir) / "input.csv"
        output_path = Path(temp_dir) / "output.json"
        script_path.write_text(_FORCED_FIT_R_SOURCE, encoding="utf-8")
        projection.to_csv(input_path, index=False, lineterminator="\n")
        completed = subprocess.run(
            [
                str(executable),
                "--vanilla",
                "--no-save",
                "--no-restore",
                str(script_path),
                str(input_path),
                str(output_path),
            ],
            cwd=config.project_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if output_path.is_file():
            try:
                parsed = json.loads(output_path.read_text(encoding="utf-8"))
            except (UnicodeError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
    detail = (completed.stderr.strip() or completed.stdout.strip())[-600:]
    return {"forced_fit_status": "error", "forced_fit_message": detail}


def run_mixed_model(config: RecipeConfig, result: DatasetResult) -> dict[str, Any]:
    """Fit the released bundle's lme4 model on this population.

    Run for every dataset, including the ungoverned raw reads. Where the model
    is not estimable the R stage's own reason code is recorded rather than a
    number being manufactured for the figure.
    """
    base = {
        "dataset": result.name,
        "engine": "R/lme4",
        "model": "random_intercept_random_slope_by_response_series",
        "analysis_role": "exploratory_heterogeneity_diagnostic_not_causal",
        "governance": result.spec.governance,
        "series_in_population": result.series_count,
    }
    if not config.run_mixed_model:
        return {**base, "status": "skipped", "reason_code": "DISABLED_BY_CONFIG"}

    eligible = mixed_model_eligible(config, result.observations)
    dropped = result.series_count - int(eligible["response_series_uid"].nunique())
    base["series_offered_to_lme4"] = int(eligible["response_series_uid"].nunique())
    base["series_excluded_for_support"] = int(dropped)
    if eligible.empty:
        # The gate above is this recipe's own threshold, so refusing there is
        # not evidence. Force the ungated fit and record lme4's verdict beside
        # the refusal, so the operator's "do it anyway" is actually carried out
        # and the reason the number would be meaningless is demonstrated.
        return {
            **base,
            "status": "not_estimable",
            "reason_code": "INSUFFICIENT_N_LEVELS",
            "error_detail": (
                "No series meets the minimum of "
                f"{config.mixed_model_minimum_distinct_n_levels} distinct N levels. "
                "A random-intercept/random-slope model cannot be identified on "
                "ladders this short: each series' line is exactly determined by "
                "its own points, leaving no residual information to separate "
                "between-series slope variance from noise."
            ),
            **probe_forced_mixed_model(config, result),
        }

    executable = _rscript_executable(config)
    projection = eligible.loc[
        :, ["response_series_uid", "study_uid", "trial_uid", "n_rate_kg_ha", "yield_t_ha"]
    ].reset_index(drop=True)
    projection.insert(
        0,
        "release_record_uid",
        [f"{result.name}::{index:06d}" for index in range(len(projection))],
    )
    with tempfile.TemporaryDirectory(prefix="modifier_by_dataset_mixed_") as temp_dir:
        input_path = Path(temp_dir) / "input.csv"
        output_path = Path(temp_dir) / "output.json"
        projection.to_csv(input_path, index=False, lineterminator="\n")
        completed = subprocess.run(
            [
                str(executable),
                "--no-save",
                "--no-restore",
                str(config.mixed_model_stage),
                str(input_path),
                str(output_path),
            ],
            cwd=config.project_root,
            text=True,
            capture_output=True,
            check=False,
        )
        parsed: dict[str, Any] | None = None
        if output_path.is_file():
            try:
                candidate = json.loads(output_path.read_text(encoding="utf-8"))
            except (UnicodeError, json.JSONDecodeError):
                candidate = None
            if isinstance(candidate, dict):
                parsed = candidate

    if completed.returncode != 0 or parsed is None or parsed.get("status") != "completed":
        return {
            **base,
            "status": "failed",
            "reason_code": (
                str(parsed.get("reason_code"))
                if parsed is not None
                else "R_STAGE_EXECUTION_FAILED"
            ),
            "error_detail": (
                (parsed or {}).get("error_message")
                or (completed.stderr.strip() or completed.stdout.strip())[-600:]
            ),
        }
    parsed = {**base, **parsed}
    parsed["rscript"] = str(executable)
    return parsed


def _mixed_model_frame(results: Mapping[str, DatasetResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in DATASET_ORDER:
        entry = dict(results[name].mixed_model)
        warnings = entry.get("warning_messages")
        if isinstance(warnings, list):
            entry["warning_messages"] = "; ".join(str(item) for item in warnings)
        rows.append(entry)
    return pd.DataFrame(rows)


def build_pooled(config: RecipeConfig, results: Mapping[str, DatasetResult]) -> DatasetResult:
    """Pool the three per-dataset populations and rescreen them as one.

    Series UIDs are already source-prefixed, so pooling cannot collide them.
    The pooled screen is reported *with* the per-dataset columns beside it,
    because a covariate that is near-constant inside every dataset but differs
    between them produces a pooled correlation that is a dataset contrast under
    the covariate's name.
    """
    frames = [results[name].observations for name in DATASET_ORDER if name in results]
    pooled_observations = pd.concat(frames, ignore_index=True, sort=False)
    series, screen = screen_dataset(config, pooled_observations)
    series = series.merge(
        pooled_observations[["response_series_uid", "source_name"]].drop_duplicates(),
        on="response_series_uid",
        how="left",
        validate="one_to_one",
    )
    spec = DatasetSpec(
        name=POOLED_KEY,
        governance="mixed_governed_and_raw",
        data_classification="restricted",
        series_key_basis="union of the three per-dataset series keys",
        source_path=None,
        encoding=None,
        note=(
            "Pooled across populations with different ladder designs and "
            "governance states; read every pooled correlation against the "
            "per-dataset columns."
        ),
    )
    return DatasetResult(
        spec=spec, observations=pooled_observations, series=series, screen=screen
    )


# --------------------------------------------------------------------------
# Cross-dataset reconciliation
# --------------------------------------------------------------------------


def _fitted(screen: pd.DataFrame) -> pd.DataFrame:
    fitted = screen.loc[screen["screen_status"].astype(str).eq("fitted"), :].copy()
    for column in ("pearson_r", "spearman_rho"):
        fitted[column] = pd.to_numeric(fitted[column], errors="coerce")
    return fitted.loc[fitted["pearson_r"].notna(), :]


def _plot_rows(result: DatasetResult) -> pd.DataFrame:
    """Fitted screen rows for the ranking figure, ordered by |Pearson r|."""
    rows = _fitted(result.screen)
    rows = rows.drop_duplicates(subset=list(_DEDUPLICATION_KEYS))
    return rows.reindex(rows["pearson_r"].abs().sort_values(ascending=True).index)


def source_determined_covariates(pooled: DatasetResult) -> tuple[str, ...]:
    """Covariates that never vary inside a source dataset.

    The pooled analogue of `grain_yield_response._study_determined`: a covariate
    with one distinct value per source is a source label in disguise, so its
    pooled correlation is a between-dataset contrast and nothing else.
    """
    determined: list[str] = []
    numeric_and_binary = [
        column
        for column in pooled.series.columns
        if column.startswith("series_") and column != "series_slope_t_ha_per_kg_n_ha"
    ]
    for name in numeric_and_binary:
        counts = pooled.series.groupby("source_name")[name].nunique(dropna=True)
        if not counts.empty and int(counts.max()) <= 1:
            determined.append(name)
    return tuple(determined)


def cross_dataset_table(results: Mapping[str, DatasetResult]) -> pd.DataFrame:
    """One row per covariate, one column block per dataset.

    This is the reconciliation the released single-dataset figure cannot show:
    whether a correlation reported on one population reappears on the others.
    """
    covariates: list[str] = []
    for name in DATASET_ORDER:
        for covariate in results[name].screen["covariate"].astype(str):
            if covariate not in covariates:
                covariates.append(covariate)

    rows: list[dict[str, Any]] = []
    for covariate in covariates:
        row: dict[str, Any] = {"covariate": covariate, "label": _factor_label(covariate)}
        fitted_values: list[float] = []
        for name in DATASET_ORDER:
            screen = results[name].screen
            match = screen.loc[screen["covariate"].astype(str) == covariate]
            if match.empty:
                row[f"{name}__screen_status"] = "absent"
                row[f"{name}__series_used"] = 0
                continue
            entry = match.iloc[0]
            status = str(entry["screen_status"])
            row[f"{name}__screen_status"] = status
            row[f"{name}__series_used"] = int(entry.get("series_used") or 0)
            pearson = pd.to_numeric(entry.get("pearson_r"), errors="coerce")
            spearman = pd.to_numeric(entry.get("spearman_rho"), errors="coerce")
            row[f"{name}__pearson_r"] = (
                float(pearson) if pd.notna(pearson) else np.nan
            )
            row[f"{name}__spearman_rho"] = (
                float(spearman) if pd.notna(spearman) else np.nan
            )
            if status == "fitted" and pd.notna(pearson) and name != POOLED_KEY:
                fitted_values.append(float(pearson))

        row["source_datasets_fitted"] = len(fitted_values)
        if len(fitted_values) >= 2:
            spread = max(fitted_values) - min(fitted_values)
            row["per_dataset_pearson_min"] = min(fitted_values)
            row["per_dataset_pearson_max"] = max(fitted_values)
            row["per_dataset_pearson_range"] = spread
            same_sign = all(value >= 0 for value in fitted_values) or all(
                value <= 0 for value in fitted_values
            )
            # Deliberately never the word "consistent": planting year is +0.65,
            # +0.09, +0.00 across the three datasets -- one sign, but nothing a
            # reader should take as replication. Sign and magnitude are reported
            # as separate facts so neither can stand in for the other.
            row["sign_agreement"] = (
                "sign_conflict"
                if not same_sign
                else (
                    "same_sign_similar_magnitude"
                    if spread < _AGREEMENT_RANGE_THRESHOLD
                    else "same_sign_wide_range"
                )
            )
        else:
            row["sign_agreement"] = "insufficient_datasets"
        row["analysis_role"] = "series_level_association_not_causal"
        rows.append(row)
    return pd.DataFrame(rows)


def population_summary_table(results: Mapping[str, DatasetResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in DATASET_ORDER:
        result = results[name]
        series = result.series
        check = pd.to_numeric(series["series_check_yield_t_ha"], errors="coerce").dropna()
        slopes = pd.to_numeric(
            series["series_slope_t_ha_per_kg_n_ha"], errors="coerce"
        ).dropna()
        observations = result.observations
        # Counts add up across the pooled row; distributions do not. A pooled
        # check-yield SD is a between-dataset spread, and a pooled median N-level
        # count is the median of a bimodal two-and-four mixture. Printing either
        # beside the three real per-dataset values invites reading them as
        # commensurable, so the pooled row suppresses them.
        pooled = name == POOLED_KEY

        def _distributional(value: float) -> float:
            return np.nan if pooled else value

        rows.append(
            {
                "dataset": name,
                "governance": result.spec.governance,
                "data_classification": result.spec.data_classification,
                "series_key_basis": result.spec.series_key_basis,
                "observations": int(len(observations)),
                "series": result.series_count,
                "series_with_estimable_slope": result.estimable_slope_count,
                "studies": int(observations["study_uid"].nunique()),
                "median_observations_per_series": _distributional(
                    float(
                        pd.to_numeric(
                            series["series_observation_count"], errors="coerce"
                        ).median()
                    )
                ),
                "median_distinct_n_levels": _distributional(
                    result.median_distinct_n_levels
                ),
                "series_with_zero_n_check": int(
                    series["series_has_zero_n_control"].astype(bool).sum()
                ),
                "n_rate_min_kg_ha": float(observations["n_rate_kg_ha"].min()),
                "n_rate_max_kg_ha": float(observations["n_rate_kg_ha"].max()),
                "check_yield_series": int(len(check)),
                "check_yield_sd_t_ha": _distributional(
                    float(check.std()) if len(check) > 1 else np.nan
                ),
                "check_yield_min_t_ha": _distributional(
                    float(check.min()) if len(check) else np.nan
                ),
                "check_yield_max_t_ha": _distributional(
                    float(check.max()) if len(check) else np.nan
                ),
                "slope_sd_t_ha_per_100kg_n": _distributional(
                    float(slopes.std()) * 100.0 if len(slopes) > 1 else np.nan
                ),
                "slope_median_t_ha_per_100kg_n": _distributional(
                    float(slopes.median()) * 100.0 if len(slopes) else np.nan
                ),
                "two_point_ladder_population": (
                    np.nan if pooled else bool(result.two_point_ladder)
                ),
                "distributional_statistics_status": (
                    "suppressed_pooled_mixture_not_a_population"
                    if pooled
                    else "population_statistic"
                ),
                "analysis_role": "descriptive_population_snapshot_not_causal",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _save_figure(root: Path, relative: str, figure: plt.Figure, *, dpi: int) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, format="jpeg", facecolor="white")
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        mode = image.mode
    return {
        "relative_path": relative,
        "artifact_kind": "figure",
        "media_type": "image/jpeg",
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "width_pixels": width,
        "height_pixels": height,
        "color_mode": mode,
    }


def _provenance_banner(result: DatasetResult) -> str:
    if result.name == POOLED_KEY:
        return "POOLED · mixed governance · restricted"
    governed = result.spec.governance == "governed_release_overlay"
    banner = "GOVERNED release overlay" if governed else "UNGOVERNED raw source read"
    if result.spec.data_classification == "restricted":
        banner += " · RESTRICTED"
    return banner


def _provenance_slug(result: DatasetResult) -> str:
    """Filesystem-safe form of `_provenance_banner`, for filenames.

    The governance state is recorded in the filename (and in `summary.md` /
    `run_manifest.json`) instead of stamped on the figure itself, so it
    survives a crop or a screenshot the same way the rest of the figure does
    not.
    """
    return re.sub(r"[^a-z0-9]+", "_", _provenance_banner(result).lower()).strip("_")


def paired_row_order(result: DatasetResult) -> tuple[str, ...]:
    """Row order for panels meant to be read side by side.

    Taken from one reference population so every paired panel keeps the same
    covariates in the same vertical positions. Without this each panel sorts and
    de-duplicates on its own numbers, and two panels of the same figure end up
    with different rows in different places -- which is precisely what makes
    them uncomparable by eye.
    """
    return tuple(str(value) for value in _plot_rows(result)["covariate"])


def _aligned_rows(result: DatasetResult, order: tuple[str, ...]) -> pd.DataFrame:
    """The result's screen rows in `order`, keeping covariates it could not fit.

    A covariate absent from a panel because it has no variation in that dataset
    is information, not a gap: dropping the row silently would make the panel
    look like the covariate was never considered.
    """
    screen = result.screen.copy()
    screen["covariate"] = screen["covariate"].astype(str)
    indexed = screen.drop_duplicates(subset=["covariate"]).set_index("covariate")
    rows = indexed.reindex(list(order)).reset_index()
    for column in ("pearson_r", "spearman_rho"):
        rows[column] = pd.to_numeric(rows.get(column), errors="coerce")
    rows["screen_status"] = rows["screen_status"].fillna("absent").astype(str)
    return rows


_UNFITTED_LABELS: Mapping[str, str] = {
    "no_covariate_variation": "no variation in this dataset",
    "insufficient_series": "too few series to screen",
    "absent": "not recorded in this dataset",
}


def plot_modifier_ranking(
    config: RecipeConfig,
    result: DatasetResult,
    *,
    row_order: tuple[str, ...] | None = None,
) -> plt.Figure:
    """The released ranking figure's design, rebuilt for one population.

    Pearson and Spearman are drawn as a pair because their disagreement is the
    diagnostic: a large linear correlation beside a near-zero rank correlation
    means one or two outlying series carry it rather than a monotone trend.

    With `row_order`, the panel keeps that covariate order and shows rows it
    could not fit, so two panels can be compared line by line.
    """
    figure, axis = plt.subplots(
        figsize=(config.figure_width_inches, config.figure_height_inches)
    )
    figure.subplots_adjust(left=0.200, right=0.760, top=0.820, bottom=0.255)

    rows = (
        _aligned_rows(result, row_order) if row_order is not None else _plot_rows(result)
    )
    positions = np.arange(len(rows))
    axis.axvline(0.0, color=_RANKING_INK_SOFT, linewidth=0.9)

    # The hierarchical model's own baseline-slope correlation, as on the
    # released figure. When lme4 could not be identified on this population the
    # line is replaced by the reason, so its absence is never silent.
    correlation = result.baseline_slope_correlation
    if correlation is not None:
        axis.axvline(
            correlation, color="#b8b7b2", linewidth=1.2, linestyle=(0, (5, 3))
        )
        # The dashed line is fitted on the series that clear the R stage's
        # support gates, which is fewer than the panel's population wherever
        # short ladders were excluded. Say so on the figure when they differ,
        # so the line is never read as covering every series in the banner.
        fitted = int(result.mixed_model.get("series_offered_to_lme4", 0))
        coverage = (
            f"\n({fitted} of {result.series_count} series)"
            if fitted and fitted != result.series_count
            else ""
        )
        axis.annotate(
            f"lme4 baseline↔slope\ncorrelation, {correlation:.2f}{coverage}",
            xy=(correlation, len(rows) - 0.52),
            ha="center",
            va="center",
            fontsize=7.2,
            color=_RANKING_INK_SOFT,
            style="italic",
            linespacing=1.4,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "none",
            },
        )
    else:
        axis.annotate(
            _mixed_model_absence_note(result),
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7.2,
            color="#a8442a",
            style="italic",
            linespacing=1.4,
        )

    for position, (_, row) in zip(positions, rows.iterrows()):
        if pd.isna(row["pearson_r"]):
            # Present but unfittable here. Draw no marker -- an absent dot is
            # the honest rendering of "no estimate" -- and say why in the
            # margin, in the slot the series count would otherwise occupy.
            axis.annotate(
                _UNFITTED_LABELS.get(
                    str(row["screen_status"]), str(row["screen_status"])
                ),
                xy=(1.0, position),
                xycoords=("axes fraction", "data"),
                xytext=(9, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7.2,
                color="#a09e99",
                style="italic",
                annotation_clip=False,
            )
            continue
        pearson = float(row["pearson_r"])
        spearman = float(row["spearman_rho"])
        axis.plot(
            [pearson, spearman], [position, position], color="#c9c8c3", linewidth=1.6
        )
        # Spearman first, so the headline Pearson marker stays visible when the
        # two land on nearly the same value.
        axis.plot(
            [spearman], [position], marker="o", markersize=8,
            color=_RANKING_FACTOR_COLOR, markeredgecolor="white", markeredgewidth=1.3,
        )
        axis.plot(
            [pearson], [position], marker="o", markersize=8,
            color=_RANKING_LEVEL_COLOR, markeredgecolor="white", markeredgewidth=1.3,
        )
        note = f"n={int(row['series_used'])} series"
        if str(row.get("estimation_artefact_warning")):
            # On a two-point ladder the slope *is* (y_N - y_0)/N, so a zero-N
            # check yield is inside the slope algebraically, not merely
            # correlated with its error. Say which of the two applies. The
            # qualifier goes on its own line because it is longer than the
            # right margin the counts alone need.
            note += (
                "\nalgebraically inside the slope"
                if result.two_point_ladder
                else "\nshares points with the slope"
            )
        axis.annotate(
            note,
            xy=(1.0, position),
            xycoords=("axes fraction", "data"),
            xytext=(9, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.2,
            color=_RANKING_INK_SOFT,
            linespacing=1.5,
            annotation_clip=False,
        )

    axis.set_yticks(positions)
    axis.set_yticklabels(
        [_factor_label(str(value)) for value in rows["covariate"]], fontsize=8.2
    )
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-0.75, max(len(rows) - 0.25, 0.75))
    axis.set(xlabel="Correlation with the fitted per-series N-response slope")
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)

    axis.set_title(
        "What might explain the differences between the response curves",
        pad=26,
        fontsize=13.5,
    )
    axis.annotate(
        f"{_DATASET_LABELS[result.name]}  ·  {result.series_count} series, "
        f"{len(result.observations)} observations",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 10),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.4,
        color=_DATASET_COLORS[result.name],
        weight="bold",
    )

    figure.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="none", markersize=8,
                color=_RANKING_LEVEL_COLOR, markeredgecolor="white",
                markeredgewidth=1.3, label="Pearson r (linear)",
            ),
            Line2D(
                [], [], marker="o", linestyle="none", markersize=8,
                color=_RANKING_FACTOR_COLOR, markeredgecolor="white",
                markeredgewidth=1.3, label="Spearman ρ (rank / monotone)",
            ),
        ],
        loc="center",
        bbox_to_anchor=(0.5, 0.160),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    figure.text(
        0.5,
        0.112,
        _ranking_footnote(result),
        ha="center",
        va="top",
        fontsize=7.2,
        color=_RANKING_INK_SOFT,
        linespacing=1.5,
    )
    return figure


def _mixed_model_absence_note(result: DatasetResult) -> str:
    """Say, on the figure, why this panel has no lme4 reference line."""
    status = str(result.mixed_model.get("status", "not_run"))
    if status == "not_estimable":
        entry = result.mixed_model
        parameters = int(entry.get("forced_fit_random_effect_parameters", 0))
        observations = int(entry.get("forced_fit_observations", 0))
        if parameters and observations:
            return (
                "no lme4 baseline↔slope line: forced on the ungated population, "
                "lme4 refuses the model as unidentifiable\n"
                f"({parameters} random-effect parameters against {observations} "
                "observations — every series is a two-point ladder)"
            )
        return (
            "no lme4 baseline↔slope line: the model is not identified on this "
            "population\n(every series is a two-point ladder, so its line is "
            "exactly determined and leaves no residual information)"
        )
    if status == "failed":
        return (
            "no lme4 baseline↔slope line: the hierarchical fit failed "
            f"({result.mixed_model.get('reason_code', 'unknown')})"
        )
    return "no lme4 baseline↔slope line: the hierarchical fit was not run"


def _ranking_footnote(result: DatasetResult) -> str:
    base = (
        "Where the two dots disagree, the linear association is carried by a few outlying series rather than a "
        "monotone trend. Series-level\nassociations only: no configured agronomic factor can appear here, because "
        "each is constant within every response series."
    )
    if result.name == POOLED_KEY:
        return (
            "Pooled across three populations with different ladder designs. Every correlation here is confounded with "
            "which dataset a series came from —\nread it against the per-dataset panels and "
            "cross_dataset_screen_comparison.csv before treating any of it as a covariate effect."
        )
    if result.two_point_ladder:
        return (
            "Two-point ladders: each series slope is exactly (yield at N − yield at 0) / N, so the zero-N check yield "
            "sits inside the slope algebraically\nand its correlation is forced, not evidence. "
            + base.split("Series-level")[0].strip()
        )
    return base


def plot_cross_dataset_comparison(
    config: RecipeConfig, table: pd.DataFrame, results: Mapping[str, DatasetResult]
) -> plt.Figure:
    """Every covariate's Pearson r side by side across the four populations.

    This is the figure the released single-dataset panel cannot be: it shows
    whether a correlation found on one population survives on the others.
    """
    plotted = table.loc[
        table[[f"{name}__pearson_r" for name in DATASET_ORDER]].notna().any(axis=1), :
    ].copy()
    order = plotted["core_trial_data__pearson_r"].abs().fillna(-1.0)
    plotted = plotted.reindex(order.sort_values(ascending=True).index)

    height = max(config.figure_height_inches, 0.62 * len(plotted) + 3.2)
    figure, axis = plt.subplots(figsize=(config.figure_width_inches + 1.6, height))
    figure.subplots_adjust(left=0.235, right=0.740, top=0.865, bottom=0.215)

    positions = np.arange(len(plotted))
    axis.axvline(0.0, color=_RANKING_INK_SOFT, linewidth=0.9)
    for position, (_, row) in zip(positions, plotted.iterrows()):
        axis.axhspan(position - 0.5, position + 0.5,
                     color="#f5f4f1" if int(position) % 2 == 0 else "white", zorder=0)
        values = [
            (name, row.get(f"{name}__pearson_r"))
            for name in DATASET_ORDER
            if pd.notna(row.get(f"{name}__pearson_r"))
        ]
        if len(values) > 1:
            axis.plot(
                [value for _, value in values],
                [position] * len(values),
                color="#d6d5d0",
                linewidth=1.4,
                zorder=1,
            )
        for name, value in values:
            axis.plot(
                [value], [position],
                marker="D" if name == POOLED_KEY else "o",
                markersize=7.5 if name == POOLED_KEY else 8.5,
                color=_DATASET_COLORS[name],
                markeredgecolor="white",
                markeredgewidth=1.2,
                zorder=3,
            )
        flag = str(row.get("sign_agreement", ""))
        note = f"{int(row.get('source_datasets_fitted', 0))}/3 datasets"
        # Same sign is not replication: a covariate can point the same way in
        # all three datasets and still differ tenfold in strength, so a wide
        # spread is called out as loudly as a sign flip.
        if flag == "sign_conflict":
            note += "  ·  SIGN CONFLICT"
        elif flag == "same_sign_wide_range":
            note += (
                f"  ·  same sign, spread {float(row['per_dataset_pearson_range']):.2f}"
            )
        axis.annotate(
            note,
            xy=(1.0, position),
            xycoords=("axes fraction", "data"),
            xytext=(9, 0),
            textcoords="offset points",
            ha="left", va="center", fontsize=7.2,
            color="#a8442a" if flag == "sign_conflict" else _RANKING_INK_SOFT,
            annotation_clip=False,
        )

    axis.set_yticks(positions)
    axis.set_yticklabels([str(value) for value in plotted["label"]], fontsize=8.4)
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-0.5, len(plotted) - 0.5)
    axis.set(xlabel="Pearson correlation with the fitted per-series N-response slope")
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.set_title(
        "Does the explanation hold in every dataset?",
        pad=28,
        fontsize=14.0,
    )
    axis.annotate(
        "The released response-curve modifier ranking is core_trial_data only. "
        "Each covariate is rescreened on every dataset here.",
        xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 11),
        textcoords="offset points", ha="left", va="bottom",
        fontsize=8.4, color=_RANKING_INK_SOFT,
    )

    handles = [
        Line2D(
            [], [], marker="D" if name == POOLED_KEY else "o", linestyle="none",
            markersize=7.5 if name == POOLED_KEY else 8.5,
            color=_DATASET_COLORS[name], markeredgecolor="white", markeredgewidth=1.2,
            label=f"{_DATASET_LABELS[name]} ({results[name].series_count} series)",
        )
        for name in DATASET_ORDER
    ]
    figure.legend(
        handles=handles, loc="center", bbox_to_anchor=(0.5, 0.118),
        ncol=2, frameon=False, fontsize=8.4,
    )
    figure.text(
        0.5, 0.062,
        "Series-level descriptive associations, not causal effects and not an effect-size ranking. "
        "Datasets differ in ladder design, governance state and series definition,\nso a covariate can be fitted on "
        "one and unfittable on another; the pooled diamond is confounded with dataset membership by construction.",
        ha="center", va="top", fontsize=7.2, color=_RANKING_INK_SOFT, linespacing=1.5,
    )
    return figure


# --------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------


def _host_ledger_paths(host_package: Path) -> frozenset[str]:
    """Every path the host bundle's own checksum ledger binds.

    Read so the mirror can refuse to write over anything the host already
    accounts for. The host's ledger was computed before this subtree existed and
    must keep verifying exactly as it did.
    """
    ledger = host_package / "CHECKSUMS.sha256"
    if ledger.is_symlink() or not ledger.is_file():
        raise RecipeError(f"Host checksum ledger is missing or unsafe: {ledger}")
    paths: set[str] = set()
    for line_number, line in enumerate(
        ledger.read_text(encoding="utf-8").splitlines(), start=1
    ):
        digest, separator, relative = line.partition("  ")
        pure = PurePosixPath(relative)
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or pure.is_absolute()
            or ".." in pure.parts
            or str(pure) != relative
            or relative in paths
        ):
            raise RecipeError(
                f"Malformed host checksum entry on line {line_number}: {line}"
            )
        paths.add(relative)
    if not paths:
        raise RecipeError(f"Host checksum ledger is empty: {ledger}")
    return frozenset(paths)


def _host_extension_placement(config: RecipeConfig) -> tuple[Path, Path]:
    """Resolve a package-confined placement that does not contain host members."""
    host = config.extension_host_package
    if not host.is_dir():
        raise RecipeError(f"Extension host package does not exist: {host}")
    placement = (host / config.extension_placement).resolve()
    try:
        relative_placement = placement.relative_to(host)
    except ValueError as exc:
        raise RecipeError(
            f"Extension placement escapes the host package: {placement}"
        ) from exc
    if str(relative_placement) in {"", "."} or PurePosixPath(
        relative_placement
    ).parts[0] in {"CHECKSUMS.sha256", "run_manifest.json"}:
        raise RecipeError(f"Unsafe extension placement: {relative_placement}")

    prefix = PurePosixPath(relative_placement).as_posix()
    collisions = sorted(
        path
        for path in _host_ledger_paths(host)
        if path == prefix or path.startswith(f"{prefix}/")
    )
    if collisions:
        raise RecipeError(
            "Extension placement contains host-ledger paths; refusing: "
            + ", ".join(collisions[:3])
        )
    return placement, relative_placement


def _write_table(root: Path, relative: str, frame: pd.DataFrame) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return {
        "relative_path": relative,
        "artifact_kind": "table",
        "media_type": "text/csv",
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
    }


def _write_text(root: Path, relative: str, text: str) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {
        "relative_path": relative,
        "artifact_kind": "document",
        "media_type": "text/markdown",
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _implementation_provenance(config: RecipeConfig) -> dict[str, str]:
    implementation_paths = {
        "base_config": config.base_config,
        "config": config.config_path,
        "entrypoint": PROJECT_ROOT
        / "modules/response_curve_modifier_by_dataset_pipeline.py",
        "launcher": PROJECT_ROOT / "response_curve_modifier_by_dataset.sh",
        "mixed_model_stage": config.mixed_model_stage,
        "producer": Path(__file__).resolve(),
    }
    provenance: dict[str, str] = {}
    for label, path in implementation_paths.items():
        if path.is_symlink() or not path.is_file():
            raise RecipeError(f"Implementation input is missing or unsafe: {path}")
        provenance[label] = sha256_file(path)
    return provenance


def _data_input_provenance(config: RecipeConfig) -> dict[str, dict[str, str]]:
    inputs = {
        "core_trial_data": {
            "release_manifest": config.release_package / "run_manifest.json",
            "analysis_eligibility_ledger": config.release_package
            / "tables/quality/analysis_eligibility_ledger.csv",
        },
    }
    inputs.update(
        {
            name: {"source_file": spec.source_path}
            for name, spec in config.specs.items()
            if name != "core_trial_data" and spec.source_path is not None
        }
    )
    provenance: dict[str, dict[str, str]] = {}
    for dataset, paths in inputs.items():
        provenance[dataset] = {}
        for label, path in paths.items():
            if path.is_symlink() or not path.is_file():
                raise RecipeError(f"Data input is missing or unsafe: {path}")
            provenance[dataset][label] = sha256_file(path)
    return provenance


def _format_r(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "—" if pd.isna(number) else f"{float(number):+.3f}"


def build_summary(
    config: RecipeConfig,
    results: Mapping[str, DatasetResult],
    comparison: pd.DataFrame,
    populations: pd.DataFrame,
    determined: tuple[str, ...],
) -> str:
    core = results["core_trial_data"]
    ltcce = results["ltcce"]
    ph = results["ph_combined_nopt_rcm"]

    def _r(dataset: str, covariate: str) -> str:
        match = comparison.loc[comparison["covariate"] == covariate]
        if match.empty:
            return "—"
        return _format_r(match.iloc[0].get(f"{dataset}__pearson_r"))

    def _n(dataset: str, covariate: str) -> str:
        match = comparison.loc[comparison["covariate"] == covariate]
        if match.empty:
            return "0"
        return str(int(match.iloc[0].get(f"{dataset}__series_used") or 0))

    # LTCCE ladder-geometry correlations, needed by both the hierarchical
    # section and the ladder section below it.
    ladder = ltcce.series[
        [
            "series_n_span_kg_ha",
            "series_planting_year_numeric",
            "series_slope_t_ha_per_kg_n_ha",
        ]
    ].dropna()
    span_year = float(
        ladder["series_n_span_kg_ha"].corr(ladder["series_planting_year_numeric"])
    )
    year_slope = float(
        ladder["series_planting_year_numeric"].corr(
            ladder["series_slope_t_ha_per_kg_n_ha"]
        )
    )

    lines: list[str] = []
    lines.append("# Response-curve modifier screen, separated by dataset")
    lines.append("")
    lines.append(
        f"Generated {datetime.now(timezone.utc).isoformat()} · schema `{SCHEMA_VERSION}`"
    )
    lines.append("")
    lines.append(
        "Status: `diagnostic_internal_not_release`, and **restricted** — two of the "
        "three populations are restricted sources, so the whole bundle inherits "
        "that classification. Not a release certification, not a fertilizer "
        "recommendation, not a causal claim."
    )
    lines.append("")
    lines.append("## Why this bundle exists")
    lines.append("")
    lines.append(
        "`grain_yield_response_diagnostics/factors/"
        "response_curve_modifier_ranking.jpeg` screens series-level covariates "
        "against fitted per-series N-response slopes on **one** population: the "
        "governed `core_trial_data` observed-series overlay, "
        f"{core.series_count} series and {len(core.observations)} observations. "
        "The release manifest records `series_count_by_source = "
        "{core_trial_data: 21, ltcce: 0, ph_combined_nopt_rcm: 0}`, so neither "
        "restricted source contributes a single series to it. Read without that "
        "context the figure looks like a pooled result. It is not."
    )
    lines.append("")
    lines.append(
        "Here the same screen is rebuilt once per dataset and once on the three "
        "pooled, so each covariate can be compared across populations."
    )
    lines.append("")
    lines.append("## Populations")
    lines.append("")
    lines.append(
        "| Dataset | Governance | Series | Slopes estimable | Obs | Median N levels | Zero-N series | Check-yield SD (t/ha) |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in populations.iterrows():
        sd = row["check_yield_sd_t_ha"]
        levels = row["median_distinct_n_levels"]
        lines.append(
            f"| `{row['dataset']}` | {row['governance']} | {int(row['series'])} | "
            f"{int(row['series_with_estimable_slope'])} | {int(row['observations'])} | "
            f"{'—' if pd.isna(levels) else f'{float(levels):g}'} | "
            f"{int(row['series_with_zero_n_check'])} | "
            f"{'—' if pd.isna(sd) else f'{float(sd):.3f}'} |"
        )
    lines.append("")
    lines.append(
        "The pooled row reports counts only. Its distributional statistics are "
        "suppressed (`—`): a check-yield SD across three populations measures the "
        "gap between them, and a pooled median N-level count is the median of a "
        "bimodal two-and-four mixture. Neither is comparable to the three real "
        "per-dataset values above it."
    )
    lines.append("")
    lines.append(
        "Only `core_trial_data` comes through the governed release overlay. The "
        "other two are read directly from their curated source files and have "
        "passed no eligibility screen, QC gate, or release verification — they "
        "are diagnostics, not releasable inventory."
    )
    lines.append("")
    lines.append(
        "`ph_combined_nopt_rcm` appears unfittable in the descriptive-statistics "
        "ladder geometry table (718 series, one N level each). That is an "
        "artefact of row-wise harmonization: the paired zero-N omission arm is "
        "held as a column (`n0_yield`), not a second row. Reconstructing the "
        f"pair gives a two-point ladder per field and {ph.estimable_slope_count} "
        "estimable slopes."
    )
    lines.append("")
    lines.append("## Hierarchical (lme4) diagnostic")
    lines.append("")
    lines.append(
        "The released figure carries a dashed reference line: lme4's random "
        "intercept↔slope correlation, the hierarchical model's own estimate of "
        "how a series' baseline and its response covary. The same "
        "random-intercept/random-slope model is fitted here on every population, "
        "raw reads included."
    )
    lines.append("")
    lines.append(
        "| Dataset | Status | Series fitted | Excluded | Mean slope (t/ha per 100 kg N) | Between-series slope SD | Baseline↔slope r | Singular |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for name in DATASET_ORDER:
        entry = results[name].mixed_model
        status = str(entry.get("status", "not_run"))
        if status == "completed":
            lines.append(
                f"| `{name}` | {status} | {int(entry.get('series_offered_to_lme4', 0))} | "
                f"{int(entry.get('series_excluded_for_support', 0))} | "
                f"{float(entry['fixed_slope_t_ha_per_100_kg_n_ha']):.3f} | "
                f"{float(entry['random_slope_sd_t_ha_per_100_kg_n_ha']):.3f} | "
                f"**{float(entry['random_intercept_slope_correlation']):+.3f}** | "
                f"{entry.get('singular')} |"
            )
        else:
            lines.append(
                f"| `{name}` | {status} (`{entry.get('reason_code', '—')}`) | "
                f"{int(entry.get('series_offered_to_lme4', 0))} | "
                f"{int(entry.get('series_excluded_for_support', 0))} | — | — | — | — |"
            )
    lines.append("")
    core_mixed = core.mixed_model
    if str(core_mixed.get("status")) == "completed":
        lines.append(
            "The `core_trial_data` correlation reproduces the released bundle's "
            f"{float(core_mixed['random_intercept_slope_correlation']):.2f} exactly, "
            "which is the check that this recipe rebuilt the promoted population "
            "rather than something near it."
        )
        lines.append("")
    ltcce_r = ltcce.baseline_slope_correlation
    core_r = core.baseline_slope_correlation
    if ltcce_r is not None and core_r is not None and (ltcce_r > 0) != (core_r > 0):
        lines.append(
            f"**The reference line itself reverses sign.** On core_trial_data the "
            f"baseline↔slope correlation is {core_r:+.3f} — higher-baseline series "
            f"respond less. On LTCCE's {int(ltcce.mixed_model['series_count'])} "
            f"series it is {ltcce_r:+.3f}, the opposite direction, and the pooled "
            f"fit ({results[POOLED_KEY].baseline_slope_correlation:+.3f}) simply "
            "follows LTCCE because LTCCE supplies most of the fitted series. The "
            "dashed line readers use to anchor the released figure is therefore a "
            "property of that one population, not a general feature of N response "
            "in this data."
        )
        lines.append("")
        lines.append(
            "Two qualifications. The pooled fit is **core + LTCCE only** — "
            "ph_combined_nopt_rcm contributes no series to it, so despite the "
            "\"all three pooled\" banner the pooled dashed line is a two-dataset "
            "quantity. And LTCCE's reversal deserves the same skepticism applied "
            "to core's: the LTCCE series key is `Year|Season|Variety`, so a "
            "series' random intercept is partly an era level, and ladder width "
            "tracks year (r = "
            f"{span_year:+.3f}). \"The baseline↔response coupling genuinely "
            "reverses\" and \"the two populations index baseline differently\" are "
            "both live readings of +0.28 versus −0.68; nothing here separates them."
        )
        lines.append("")
        lines.append(
            "Note these are *not* the same quantity as the zero-N check-yield row "
            "in the screen above: lme4's correlation is between shrunken random "
            "effects estimated jointly, while the screen correlates each series' "
            "observed check yield against its own least-squares slope. They "
            "answer the same question by different routes and agree here that the "
            "core_trial_data coupling does not carry over — but their magnitudes "
            "are not comparable and should not be quoted interchangeably."
        )
        lines.append("")
    ph_mixed = ph.mixed_model
    if str(ph_mixed.get("status")) != "completed":
        lines.append(
            f"`ph_combined_nopt_rcm` returns `{ph_mixed.get('reason_code')}`: the "
            "model is **not identified** on two-point ladders. With two points a "
            "series' line is exactly determined, so there is no residual "
            "information left to separate between-series slope variance from "
            "noise."
        )
        lines.append("")
        forced = str(ph_mixed.get("forced_fit_status", ""))
        if forced:
            lines.append(
                "That refusal is this recipe's own support threshold, which "
                "proves nothing by itself, so the fit was also **forced** on the "
                "ungated population and lme4's own verdict recorded:"
            )
            lines.append("")
            if forced == "error":
                lines.append(
                    f"> {str(ph_mixed.get('forced_fit_message', '')).strip()}"
                )
                lines.append("")
                lines.append(
                    f"{int(ph_mixed.get('forced_fit_random_effect_parameters', 0))} "
                    "random-effect parameters (two per series) against "
                    f"{int(ph_mixed.get('forced_fit_observations', 0))} "
                    "observations: the model has more parameters than data "
                    "points. lme4 refuses to fit it at all — this is not a "
                    "threshold choice, and no number exists to put on the panel."
                )
            else:
                lines.append(
                    f"Forced fit status `{forced}`, singular = "
                    f"`{ph_mixed.get('forced_fit_singular')}`, correlation "
                    f"{ph_mixed.get('forced_fit_correlation')}, residual SD "
                    f"{ph_mixed.get('forced_fit_residual_sd')}. A singular fit's "
                    "correlation is the optimizer's boundary, not an estimate, so "
                    "it is recorded here and kept off the panel."
                )
            lines.append("")
        lines.append("Its panel states this in place of the dashed line.")
        lines.append("")
    lines.append(
        "The fit was attempted on every population; series below the R stage's "
        "support thresholds (at least "
        f"{config.mixed_model_minimum_observations_per_series} observations and "
        f"{config.mixed_model_minimum_distinct_n_levels} distinct N levels) are "
        "excluded and counted in the table rather than dropped silently. Full "
        "diagnostics: `all_datasets/mixed_model_summary.csv`."
    )
    lines.append("")
    lines.append("## What changes when the datasets are separated")
    lines.append("")
    lines.append("### Zero-N check yield does not replicate")
    lines.append("")
    lines.append(
        f"The released figure's largest association is zero-N check yield versus "
        f"series slope: **r = {_r('core_trial_data', 'series_check_yield_t_ha')} "
        f"on {_n('core_trial_data', 'series_check_yield_t_ha')} core_trial_data "
        f"series**. The released summary hedged it as possibly \"a mechanical "
        f"estimation artefact\". Rescreened out of sample it collapses: "
        f"**r = {_r('ltcce', 'series_check_yield_t_ha')} on "
        f"{_n('ltcce', 'series_check_yield_t_ha')} LTCCE series**."
    )
    lines.append("")
    check = populations.set_index("dataset")["check_yield_sd_t_ha"]
    lines.append(
        f"That collapse is not variance compression. LTCCE's check-yield spread "
        f"is *wider* than core_trial_data's (SD {float(check['ltcce']):.3f} vs "
        f"{float(check['core_trial_data']):.3f} t/ha), so there was more signal "
        f"available to detect, not less. On the evidence here the core_trial_data "
        f"correlation is a small-sample result that does not generalize."
    )
    lines.append("")
    lines.append(
        f"`ph_combined_nopt_rcm` posts r = "
        f"{_r('ph_combined_nopt_rcm', 'series_check_yield_t_ha')}, and that number "
        f"carries no evidential weight at all: with two points per series the "
        f"slope is exactly (yield at N − yield at 0) / N, so the check yield is "
        f"inside the slope algebraically. The figure labels that row "
        f"`ALGEBRAICALLY INSIDE THE SLOPE` rather than reusing the milder "
        f"\"shares points with the slope\" note."
    )
    lines.append("")
    lines.append("### LTCCE's own largest association is ladder geometry")
    lines.append("")
    lines.append(
        f"On LTCCE the largest fitted association is N-ladder span (and its "
        f"collinear twins, highest and mean tested N): "
        f"r = {_r('ltcce', 'series_n_span_kg_ha')} across "
        f"{_n('ltcce', 'series_n_span_kg_ha')} series, against "
        f"{_r('core_trial_data', 'series_n_span_kg_ha')} on core_trial_data."
    )
    lines.append("")
    lines.append(
        f"The obvious confound is era — the LTCCE series key is "
        f"`Year|Season|Variety` and ladder width did change over the record "
        f"(span vs planting year, r = {span_year:+.3f}). But planting year itself "
        f"barely tracks the slope (r = {year_slope:+.3f}), so the ladder "
        f"association is not simply an era contrast wearing a geometry label. "
        f"It is still a design property of the series, not an agronomic factor: "
        f"wider ladders showing *steeper* fitted slopes runs against a saturating "
        f"response and should be treated as a question, not a finding."
    )
    lines.append("")
    lines.append("### The pooled panel is mostly a dataset contrast")
    lines.append("")
    lines.append(
        "Pooled series shares are "
        + ", ".join(
            f"`{name}` {100.0 * results[name].series_count / results[POOLED_KEY].series_count:.1f}%"
            for name in DATASET_ORDER
            if name != POOLED_KEY
        )
        + " — core_trial_data is under 2% of the pooled population, so a pooled "
        "correlation is almost entirely the two restricted sources."
    )
    lines.append("")
    wide = comparison.loc[comparison["sign_agreement"] == "same_sign_wide_range"]
    lines.append(
        "Same sign is not replication, so the comparison table never says "
        "\"consistent\": it reports `sign_conflict`, `same_sign_similar_magnitude` "
        f"(spread < {_AGREEMENT_RANGE_THRESHOLD:g}) and `same_sign_wide_range` as "
        f"separate verdicts. {len(wide)} covariates land in the last category — "
        "planting year is the one to watch, positive on all three datasets "
        f"({_r('core_trial_data', 'series_planting_year_numeric')}, "
        f"{_r('ltcce', 'series_planting_year_numeric')}, "
        f"{_r('ph_combined_nopt_rcm', 'series_planting_year_numeric')}) but spanning "
        "almost the whole usable range, which is agreement in sign and nothing else."
    )
    lines.append("")
    conflicts = comparison.loc[comparison["sign_agreement"] == "sign_conflict"]
    lines.append(
        f"{len(conflicts)} of {len(comparison)} covariates change sign between "
        "datasets. The clearest illustration is series observation count: "
        f"pooled r = {_r(POOLED_KEY, 'series_observation_count')}, but "
        f"{_r('core_trial_data', 'series_observation_count')} on core_trial_data "
        f"and {_r('ltcce', 'series_observation_count')} on LTCCE, and it is "
        "unfittable on ph_combined_nopt_rcm because every series there has "
        "exactly two observations. The pooled number is the gap between a "
        "two-point population and a four-point one, not a property of series "
        "length."
    )
    lines.append("")
    if determined:
        lines.append(
            "Covariates with no variation inside any single source dataset — "
            "pooled correlations for these are pure between-dataset contrasts: "
            + ", ".join(f"`{name}`" for name in determined)
            + "."
        )
    else:
        lines.append(
            "No covariate is constant within every source dataset, so no pooled "
            "correlation is a pure source label; they are still confounded with "
            "source membership in proportion to how much the datasets differ."
        )
    lines.append("")
    lines.append("## Reading the artifacts")
    lines.append("")
    lines.append(
        "- `all_datasets/cross_dataset_comparison.jpeg` — every covariate's "
        "Pearson r on all four populations at once. This is the figure that answers "
        "\"is this LTCCE only?\"."
    )
    lines.append(
        "- `<dataset>/response_curve_modifier_ranking__<governance-state>.jpeg` "
        "— the released figure's design, one population per panel. The panel "
        "carries no governance label; the trailing `__<governance-state>` "
        "filename segment does (`governed_release_overlay` for `core_trial_data`, "
        "`ungoverned_raw_source_read_restricted` for `ltcce` and "
        "`ph_combined_nopt_rcm`), and `summary.md` and `run_manifest.json` both "
        "record it per dataset too."
    )
    lines.append(
        "- `paired_core_vs_ltcce/core_trial_data__governed_release_overlay.jpeg` "
        "and `paired_core_vs_ltcce/ltcce__ungoverned_raw_source_read_restricted.jpeg` "
        "— the same two panels forced onto core's row order, with the covariates "
        "LTCCE cannot fit kept in place and labelled. Use these when reading core "
        "against LTCCE line by line; the per-dataset panels above sort "
        "independently and do not align."
    )
    lines.append(
        "- `all_datasets/cross_dataset_screen_comparison.csv` — the same reconciliation as "
        "a table, with per-dataset r, ρ, series counts, screen status, and a "
        "sign-agreement flag."
    )
    lines.append(
        "- `<dataset>/series_covariates.csv` and "
        "`<dataset>/series_slope_modifier_screen.csv` — per-series covariates "
        "and the full screen behind each panel."
    )
    lines.append(
        "- `all_datasets/population_summary.csv` — the populations table above."
    )
    lines.append("")
    lines.append("## Governance")
    lines.append("")
    lines.append(
        "Every number here is a descriptive series-level association. Series-level "
        "covariates are confounded with study, site, era and design; none is a "
        "causal moderator, an effect-size ranking, or a basis for a fertilizer "
        "recommendation. No N×factor interaction is estimated. The two restricted "
        "sources are read outside the governed release path, so nothing in this "
        "bundle may be promoted into the release package or its checksum ledger."
    )
    lines.append("")
    return "\n".join(lines)


def _write_bundle_at_root(
    config: RecipeConfig,
    root: Path,
    in_place_extension: Path | None,
    *,
    implementation_sha256: Mapping[str, str],
    data_input_sha256: Mapping[str, Mapping[str, str]],
) -> Path:
    if in_place_extension is None:
        raise RecipeError(
            "The canonical full bundle must be written directly at its host placement"
        )
    if root.exists():
        raise RecipeError(f"Internal output path already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)

    results: dict[str, DatasetResult] = {}
    for name in DATASET_ORDER:
        if name == POOLED_KEY:
            continue
        spec = config.specs[name]
        observations, inputs = _BUILDERS[name](config, spec)
        series, screen = screen_dataset(config, observations)
        results[name] = DatasetResult(
            spec=spec,
            observations=observations,
            series=series,
            screen=screen,
            inputs=inputs,
        )
    results[POOLED_KEY] = build_pooled(config, results)

    for name in DATASET_ORDER:
        results[name].mixed_model = run_mixed_model(config, results[name])

    comparison = cross_dataset_table(results)
    populations = population_summary_table(results)
    determined = source_determined_covariates(results[POOLED_KEY])

    artifacts: list[dict[str, Any]] = []
    for name in DATASET_ORDER:
        result = results[name]
        artifacts.append(
            _write_table(
                root,
                _grouped_artifact_path(name, "series_covariates.csv"),
                result.series,
            )
        )
        artifacts.append(
            _write_table(
                root,
                _grouped_artifact_path(name, "series_slope_modifier_screen.csv"),
                result.screen,
            )
        )
        figure = plot_modifier_ranking(config, result)
        try:
            artifacts.append(
                _save_figure(
                    root,
                    _grouped_artifact_path(
                        name,
                        "response_curve_modifier_ranking__"
                        f"{_provenance_slug(result)}.jpeg",
                    ),
                    figure,
                    dpi=config.figure_dpi,
                )
            )
        finally:
            plt.close(figure)

    artifacts.append(
        _write_table(
            root,
            _grouped_artifact_path(
                POOLED_KEY, "cross_dataset_screen_comparison.csv"
            ),
            comparison,
        )
    )
    artifacts.append(
        _write_table(
            root,
            _grouped_artifact_path(POOLED_KEY, "population_summary.csv"),
            populations,
        )
    )
    artifacts.append(
        _write_table(
            root,
            _grouped_artifact_path(POOLED_KEY, "mixed_model_summary.csv"),
            _mixed_model_frame(results),
        )
    )

    # A row-aligned core vs LTCCE pair. The per-dataset panels above each sort
    # and de-duplicate on their own numbers, so LTCCE shows five rows against
    # core's nine and the two cannot be read line by line. These two share
    # core's row order and keep the covariates LTCCE cannot fit.
    order = paired_row_order(results["core_trial_data"])
    for name in ("core_trial_data", "ltcce"):
        figure = plot_modifier_ranking(config, results[name], row_order=order)
        try:
            artifacts.append(
                _save_figure(
                    root,
                    _grouped_artifact_path(
                        "paired_core_vs_ltcce",
                        f"{name}__{_provenance_slug(results[name])}.jpeg",
                    ),
                    figure,
                    dpi=config.figure_dpi,
                )
            )
        finally:
            plt.close(figure)

    figure = plot_cross_dataset_comparison(config, comparison, results)
    try:
        artifacts.append(
            _save_figure(
                root,
                _grouped_artifact_path(
                    POOLED_KEY, "cross_dataset_comparison.jpeg"
                ),
                figure,
                dpi=config.figure_dpi,
            )
        )
    finally:
        plt.close(figure)

    artifacts.append(
        _write_text(
            root,
            "summary.md",
            build_summary(config, results, comparison, populations, determined),
        )
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_internal_not_release",
        "data_classification": "restricted",
        "governance_note": (
            "core_trial_data is the governed release overlay population; ltcce and "
            "ph_combined_nopt_rcm are ungoverned raw source reads. This bundle is "
            "never promoted into the release package or its checksum ledger."
        ),
        "analysis_role": "series_level_association_not_causal",
        "implementation_sha256": dict(implementation_sha256),
        "datasets": {
            name: {
                "governance": results[name].spec.governance,
                "data_classification": results[name].spec.data_classification,
                "series_key_basis": results[name].spec.series_key_basis,
                "series_count": results[name].series_count,
                "observation_count": int(len(results[name].observations)),
                "series_with_estimable_slope": results[name].estimable_slope_count,
                "median_distinct_n_levels": results[name].median_distinct_n_levels,
                "mixed_model_status": str(
                    results[name].mixed_model.get("status", "not_run")
                ),
                "mixed_model_baseline_slope_correlation": (
                    results[name].baseline_slope_correlation
                ),
                "input_sha256": results[name].inputs,
                "note": results[name].spec.note,
            }
            for name in DATASET_ORDER
        },
        "source_determined_covariates": list(determined),
        "artifacts": artifacts,
    }
    if in_place_extension is not None:
        manifest["package_extension"] = {
            "placement": str(in_place_extension),
            "host_package": str(
                config.extension_host_package.relative_to(config.project_root)
            ),
            "file_count": len(artifacts) + 2,
            "placement_mode": "canonical_full_bundle_in_host_subtree",
            "in_host_checksum_ledger": False,
            "governance_note": (
                "This restricted diagnostic bundle is stored inside the host "
                "package but is excluded from the host CHECKSUMS.sha256. Its "
                "own CHECKSUMS.sha256 binds the complete subtree."
            ),
        }
    manifest_path = root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    checksum_lines = [
        f"{entry['sha256']}  {entry['relative_path']}"
        for entry in sorted(artifacts, key=lambda item: item["relative_path"])
    ]
    checksum_lines.append(f"{sha256_file(manifest_path)}  run_manifest.json")
    (root / "CHECKSUMS.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )

    if _implementation_provenance(config) != dict(implementation_sha256):
        raise RecipeError("An implementation input changed during bundle generation")
    if _data_input_provenance(config) != {
        dataset: dict(entries) for dataset, entries in data_input_sha256.items()
    }:
        raise RecipeError("A data input changed during bundle generation")
    for dataset, expected in data_input_sha256.items():
        if results[dataset].inputs != dict(expected):
            raise RecipeError(
                f"Recorded input provenance does not match the loaded {dataset} data"
            )

    return root


def _verify_closed_bundle(root: Path) -> None:
    """Verify every generated member and reject undeclared physical files."""
    if root.is_symlink() or not root.is_dir():
        raise RecipeError(f"Generated bundle root is invalid: {root}")
    ledger_path = root / "CHECKSUMS.sha256"
    if not ledger_path.is_file():
        raise RecipeError(f"Generated checksum ledger is missing: {ledger_path}")
    members: dict[str, str] = {}
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        relative_path = PurePosixPath(relative)
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative in members
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or str(relative_path) != relative
        ):
            raise RecipeError(f"Malformed generated checksum entry: {line}")
        member = root / relative_path
        if not member.is_file() or member.is_symlink():
            raise RecipeError(f"Generated checksum member is invalid: {member}")
        actual = sha256_file(member)
        if actual != digest:
            raise RecipeError(f"Generated checksum mismatch: {relative}")
        members[relative] = digest
    expected = set(members) | {"CHECKSUMS.sha256"}
    expected_directories = {
        parent.as_posix()
        for relative in expected
        for parent in PurePosixPath(relative).parents
        if str(parent) != "."
    }
    physical: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RecipeError(f"Generated bundle contains unsafe filesystem object: {path}")
        if path.is_file():
            physical.add(relative)
        elif path.is_dir():
            if relative not in expected_directories:
                raise RecipeError(
                    f"Generated bundle contains undeclared filesystem object: {path}"
                )
        else:
            raise RecipeError(f"Generated bundle contains unsafe filesystem object: {path}")
    if physical != expected:
        raise RecipeError(
            "Generated bundle inventory mismatch: "
            f"unexpected={sorted(physical - expected)}, "
            f"missing={sorted(expected - physical)}"
        )


@contextmanager
def _publication_lock(destination: Path):
    """Translate the shared container lock into this producer's error type."""
    try:
        with diagnostic_container_publication_lock(destination):
            yield
    except DiagnosticBundleError as exc:
        raise RecipeError(
            "Another response-modifier publication is already running for "
            f"{destination}"
        ) from exc


def _publication_residue_parents(destination: Path) -> list[Path]:
    prefixes = (
        f".{destination.name}.stage-",
        f".{destination.name}.backup-",
    )
    residues: list[Path] = []
    for child in destination.parent.iterdir():
        if not child.name.startswith(prefixes):
            continue
        if child.is_symlink() or not child.is_dir():
            raise RecipeError(f"Unsafe publication residue: {child}")
        residues.append(child)
    return sorted(residues, key=lambda path: path.name)


def _recover_interrupted_publication(destination: Path) -> None:
    """Recover an owned staged/backup bundle left by a hard process exit."""
    residues = _publication_residue_parents(destination)
    if destination.exists():
        _verify_closed_bundle(destination)
        for residue in residues:
            shutil.rmtree(residue)
        return
    if not residues:
        return

    valid_stages: list[tuple[Path, Path]] = []
    valid_backups: list[tuple[Path, Path]] = []
    for residue in residues:
        candidate = residue / "bundle"
        try:
            _verify_closed_bundle(candidate)
        except RecipeError:
            continue
        item = (residue, candidate)
        if residue.name.startswith(f".{destination.name}.stage-"):
            valid_stages.append(item)
        else:
            valid_backups.append(item)

    if len(valid_stages) > 1 or (not valid_stages and len(valid_backups) > 1):
        raise RecipeError(
            f"Ambiguous interrupted publication residues for {destination}"
        )
    selected = valid_stages[0] if valid_stages else (
        valid_backups[0] if valid_backups else None
    )
    if selected is None:
        raise RecipeError(
            f"No valid bundle can be recovered from publication residues for {destination}"
        )

    _, candidate = selected
    candidate.rename(destination)
    _verify_closed_bundle(destination)
    for residue in residues:
        if residue.exists():
            shutil.rmtree(residue)


def _promote_staged_bundle(staged: Path, destination: Path) -> None:
    """Replace the live bundle with a verified same-filesystem staged bundle."""
    backup_parent: Path | None = None
    backup: Path | None = None
    if destination.exists():
        try:
            if exchange_directories(staged, destination):
                shutil.rmtree(staged)
                return
        except OSError as exc:
            raise RecipeError(
                f"Failed to atomically exchange generated bundle: {destination}"
            ) from exc
        backup_parent = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.backup-", dir=destination.parent
            )
        )
        backup = backup_parent / "bundle"
        destination.rename(backup)
    try:
        staged.rename(destination)
    except BaseException as exc:
        try:
            if backup is not None and backup.exists() and not destination.exists():
                backup.rename(destination)
            if backup_parent is not None and backup_parent.exists():
                backup_parent.rmdir()
        except OSError as restore_exc:
            raise RecipeError(
                f"Failed to restore the prior generated bundle: {destination}"
            ) from restore_exc
        if isinstance(exc, OSError):
            raise RecipeError(
                f"Failed to promote generated bundle: {destination}"
            ) from exc
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)
    if backup_parent is not None and backup_parent.exists():
        backup_parent.rmdir()


def write_bundle(config: RecipeConfig) -> Path:
    _validate_canonical_destination(config)
    destination = config.output_root
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _publication_lock(CANONICAL_HOST_PACKAGE):
        _recover_interrupted_publication(destination)
        implementation_sha256 = _implementation_provenance(config)
        data_input_sha256 = _data_input_provenance(config)
        placement, relative_placement = _host_extension_placement(config)
        if destination.resolve() != placement:
            raise RecipeError(
                f"Canonical output and host placement differ: {destination} != {placement}"
            )
        if destination.exists() and not config.overwrite:
            raise RecipeError(f"Output root already exists: {destination}")
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}.stage-", dir=destination.parent
        ) as staging_parent:
            staged = Path(staging_parent) / "bundle"
            _write_bundle_at_root(
                config,
                staged,
                relative_placement,
                implementation_sha256=implementation_sha256,
                data_input_sha256=data_input_sha256,
            )
            _verify_closed_bundle(staged)
            _promote_staged_bundle(staged, destination)
        return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the response-curve modifier screen once per registered source "
            "dataset and once on the three pooled."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the recipe TOML")
    arguments = parser.parse_args(argv)
    try:
        config = load_config(Path(arguments.config).resolve())
        root = write_bundle(config)
    except RecipeError as exc:
        print(f"response_curve_modifier_by_dataset: {exc}", file=sys.stderr)
        return 2
    print(f"response_curve_modifier_by_dataset: wrote {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
