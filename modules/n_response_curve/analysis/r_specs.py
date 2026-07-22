from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .analysis_matrix import AnalysisCandidate
from .factor_catalog import factor_value
from .values import finite_number


@dataclass(frozen=True)
class RAnalysisPreparation:
    status: str
    reason_codes: tuple[str, ...]
    specification: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    stable_key: str


_CATEGORICAL_OUTCOMES = frozenset({"curve_shape_class", "optimum_status"})


def _present(value: object) -> bool:
    if finite_number(value) is not None:
        return True
    return isinstance(value, (str, bool)) and bool(str(value).strip())


def _has_supported_random_intercept(rows: Sequence[Mapping[str, Any]]) -> bool:
    counts = Counter(str(row.get("study_uid") or "") for row in rows)
    counts.pop("", None)
    return len(counts) >= 2 and any(count >= 2 for count in counts.values())


def _normalized_rows(
    rows: Sequence[Mapping[str, Any]],
    factor_names: Sequence[str],
    *,
    outcome_name: str,
    stable_key: str,
    observation_level: bool,
) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        factors = {
            name: factor_value(row, name)
            for name in factor_names
        }
        if (
            not row.get(stable_key)
            or not row.get("study_uid")
            or not _present(row.get(outcome_name))
            or any(not _present(value) for value in factors.values())
        ):
            continue
        if observation_level and finite_number(row.get("n_rate_kg_ha")) is None:
            continue
        row.update(factors)
        normalized.append(row)
    return tuple(
        sorted(
            normalized,
            key=lambda row: str(row.get(stable_key, "")),
        )
    )


def _curve_formula(
    candidate: AnalysisCandidate,
    *,
    include_random_intercept: bool,
) -> str:
    if candidate.analysis_family == "all_supported_interactions":
        fixed_terms = " * ".join(candidate.factor_names)
    else:
        fixed_terms = " + ".join(candidate.factor_names)
    formula = f"{candidate.curve_outcome} ~ {fixed_terms}"
    if include_random_intercept:
        formula += " + (1 | study_uid)"
    return formula


def _observation_formula(
    factor_names: Sequence[str],
    *,
    include_random_intercept: bool,
) -> str:
    terms = ["n_rate_kg_ha", "I(n_rate_kg_ha^2)"]
    terms.extend(factor_names)
    for factor_name in factor_names:
        terms.extend(
            (
                f"n_rate_kg_ha:{factor_name}",
                f"I(n_rate_kg_ha^2):{factor_name}",
            )
        )
    if include_random_intercept:
        terms.append("(1 | study_uid)")
    return "yield_t_ha ~ " + " + ".join(terms)


def _design_gate_reason(
    rows: Sequence[Mapping[str, Any]],
    factor_names: Sequence[str],
    *,
    outcome_name: str,
    outcome_kind: str,
    observation_level: bool,
) -> str | None:
    frame = pd.DataFrame([{name: row[name] for name in factor_names} for row in rows])
    categorical: list[str] = []
    for factor_name in factor_names:
        values = tuple(frame[factor_name])
        if len({str(value) for value in values}) < 2:
            return "UNIDENTIFIED_FACTOR_VARIATION"
        if all(not isinstance(value, bool) and finite_number(value) is not None for value in values):
            frame[factor_name] = [float(value) for value in values]
        else:
            categorical.append(factor_name)
    design = pd.get_dummies(frame, columns=categorical, drop_first=True, dtype=float)
    if observation_level:
        n_rate = np.asarray([float(row["n_rate_kg_ha"]) for row in rows], dtype=float)
        design["n_rate_kg_ha"] = n_rate
        design["n_rate_squared"] = n_rate**2
    matrix = np.column_stack((np.ones(len(design), dtype=float), design.to_numpy(dtype=float)))
    rank = int(np.linalg.matrix_rank(matrix))
    if rank < matrix.shape[1]:
        return "ALIASED_DESIGN_MATRIX"
    predictors = design.to_numpy(dtype=float)
    if predictors.shape[1] > 1:
        scales = predictors.std(axis=0)
        standardized = (predictors - predictors.mean(axis=0)) / scales
        condition_number = np.linalg.cond(standardized)
        if not np.isfinite(condition_number) or condition_number > 1.0e8:
            return "COLLINEAR_DESIGN_MATRIX"
    if len(rows) - rank < 3:
        return "INSUFFICIENT_RESIDUAL_INFORMATION"
    if outcome_kind == "categorical":
        outcome_by_pattern: dict[tuple[str, ...], set[str]] = {}
        for row in rows:
            pattern = tuple(str(row[name]) for name in factor_names)
            outcome_by_pattern.setdefault(pattern, set()).add(str(row[outcome_name]))
        if len(outcome_by_pattern) >= 2 and all(len(outcomes) == 1 for outcomes in outcome_by_pattern.values()):
            return "COMPLETE_SEPARATION_RISK"
    return None


def prepare_r_analysis(
    *,
    candidate: AnalysisCandidate,
    rows: Sequence[Mapping[str, Any]],
    observation_level: bool = False,
) -> RAnalysisPreparation:
    """Materialize a fully normalized, immutable R-stage specification."""

    if candidate.engine != "r":
        raise ValueError("R analysis preparation requires an R-owned candidate")
    if not candidate.factor_names:
        return RAnalysisPreparation(
            "skipped",
            ("R_FACTOR_SET_REQUIRED",),
            {},
            (),
            "record_uid" if observation_level else "response_series_uid",
        )
    stable_key = "record_uid" if observation_level else "response_series_uid"
    outcome_name = "yield_t_ha" if observation_level else candidate.curve_outcome
    normalized = _normalized_rows(
        rows,
        candidate.factor_names,
        outcome_name=outcome_name,
        stable_key=stable_key,
        observation_level=observation_level,
    )
    minimum_rows = 8 if observation_level else max(4, len(candidate.factor_names) + 3)
    if len(normalized) < minimum_rows:
        return RAnalysisPreparation(
            "skipped",
            ("INSUFFICIENT_NORMALIZED_R_ROWS",),
            {},
            normalized,
            stable_key,
        )
    studies = {str(row["study_uid"]) for row in normalized}
    if len(studies) < 2:
        return RAnalysisPreparation(
            "skipped",
            ("INSUFFICIENT_R_STUDY_GROUPS",),
            {},
            normalized,
            stable_key,
        )

    random_intercept = _has_supported_random_intercept(normalized)
    if observation_level:
        n_levels = {
            finite_number(row.get("n_rate_kg_ha"))
            for row in normalized
        }
        n_levels.discard(None)
        if len(n_levels) < 3:
            return RAnalysisPreparation(
                "skipped",
                ("INSUFFICIENT_OBSERVATION_LEVEL_N_SUPPORT",),
                {},
                normalized,
                stable_key,
            )
        outcome_kind = "continuous"
        gate_reason = _design_gate_reason(
            normalized,
            candidate.factor_names,
            outcome_name=outcome_name,
            outcome_kind=outcome_kind,
            observation_level=True,
        )
        if gate_reason is not None:
            return RAnalysisPreparation("skipped", (gate_reason,), {}, normalized, stable_key)
        model_kind = "lmer" if random_intercept else "lm"
        formula = _observation_formula(
            candidate.factor_names,
            include_random_intercept=random_intercept,
        )
    else:
        outcome_kind = (
            "categorical"
            if candidate.curve_outcome in _CATEGORICAL_OUTCOMES
            else "continuous"
        )
        if outcome_kind == "categorical":
            level_counts = Counter(
                str(row[candidate.curve_outcome])
                for row in normalized
            )
            if len(level_counts) < 2 or min(level_counts.values()) < 2:
                return RAnalysisPreparation(
                    "skipped",
                    ("INSUFFICIENT_CATEGORICAL_OUTCOME_SUPPORT",),
                    {},
                    normalized,
                    stable_key,
                )
            if len(level_counts) > 2:
                model_kind = "multinom"
                random_intercept = False
            else:
                model_kind = "glmmTMB" if random_intercept else "glm"
        else:
            model_kind = "lmer" if random_intercept else "lm"
        gate_reason = _design_gate_reason(
            normalized,
            candidate.factor_names,
            outcome_name=outcome_name,
            outcome_kind=outcome_kind,
            observation_level=False,
        )
        if gate_reason is not None:
            return RAnalysisPreparation("skipped", (gate_reason,), {}, normalized, stable_key)
        formula = _curve_formula(
            candidate,
            include_random_intercept=random_intercept,
        )

    specification: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "specification_hash": candidate.specification_hash,
        "dataset_version_id": candidate.dataset_version_id,
        "source_combination_id": candidate.source_combination_id,
        "curve_outcome": candidate.curve_outcome,
        "analysis_family": candidate.analysis_family,
        "factor_names": list(candidate.factor_names),
        "engine": "r",
        "support_gates_passed": True,
        "model_formula": formula,
        "model_kind": model_kind,
        "outcome_kind": outcome_kind,
        "multiplicity": {
            "method": "BH",
            "family_id": "|".join(
                (
                    candidate.dataset_version_id,
                    candidate.source_combination_id,
                    candidate.curve_outcome,
                    candidate.analysis_family,
                )
            ),
        },
        "grouping_column": "study_uid",
        "missing_data_policy": "factor-specific complete cases; no imputation",
    }
    if observation_level:
        specification.update(
            {
                "n_rate_column": "n_rate_kg_ha",
                "require_quadratic_n": True,
                "observation_outcome": "yield_t_ha",
            }
        )
    elif candidate.analysis_family == "marginal_contrasts":
        specification["contrast_specification"] = {
            "factor_name": candidate.factor_names[0],
            "adjustment": "BH",
        }
    return RAnalysisPreparation(
        "run",
        (),
        specification,
        normalized,
        stable_key,
    )


__all__ = ["RAnalysisPreparation", "prepare_r_analysis"]
