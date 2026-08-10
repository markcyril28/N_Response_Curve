from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import itertools
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .analysis_matrix import AnalysisCandidate, first_stage_uncertainty_reasons
from .factor_catalog import factor_value
from .values import finite_number


@dataclass(frozen=True)
class RAnalysisPreparation:
    status: str
    reason_codes: tuple[str, ...]
    specification: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    stable_key: str
    membership_rows: tuple[Mapping[str, Any], ...] = ()


_CATEGORICAL_OUTCOMES = frozenset({"curve_shape_class", "optimum_status"})
_FITTED_FEATURE_INFERENTIAL_FAMILIES = frozenset(
    {
        "one_factor_inferential",
        "all_supported_interactions",
        "multivariable_mixed_effects",
        "marginal_contrasts",
    }
)


def _present(value: object) -> bool:
    if finite_number(value) is not None:
        return True
    return isinstance(value, (str, bool)) and bool(str(value).strip())


def _has_supported_random_intercept(
    rows: Sequence[Mapping[str, Any]],
    grouping_column: str,
) -> bool:
    counts = Counter(str(row.get(grouping_column) or "") for row in rows)
    counts.pop("", None)
    return len(counts) >= 3 and sum(count >= 2 for count in counts.values()) >= 2


def _normalized_rows(
    rows: Sequence[Mapping[str, Any]],
    factor_names: Sequence[str],
    *,
    candidate_id: str,
    factor_representations: Mapping[str, Mapping[str, Any]],
    outcome_name: str,
    stable_key: str,
    observation_level: bool,
    first_stage_variance_field: str | None = None,
    dependence_unit_field: str | None = None,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    normalized: list[dict[str, Any]] = []
    membership: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        factors = {
            name: factor_value(
                row,
                name,
                representation=factor_representations.get(name),
            )
            for name in factor_names
        }
        exclusion_reasons: list[str] = []
        if not row.get(stable_key):
            exclusion_reasons.append("MISSING_STABLE_RECORD_ID")
        if not row.get("study_uid"):
            exclusion_reasons.append("MISSING_STUDY_ID")
        if dependence_unit_field is not None and not row.get(dependence_unit_field):
            exclusion_reasons.append("MISSING_DEPENDENCE_UNIT")
        if not _present(row.get(outcome_name)):
            exclusion_reasons.append("MISSING_OUTCOME")
        exclusion_reasons.extend(
            f"MISSING_FACTOR:{name}"
            for name, value in factors.items()
            if not _present(value)
        )
        if observation_level and finite_number(row.get("n_rate_kg_ha")) is None:
            exclusion_reasons.append("MISSING_N_RATE")
        variance: float | None = None
        if first_stage_variance_field is not None:
            variance = finite_number(row.get(first_stage_variance_field))
            if variance is None or variance <= 0.0:
                exclusion_reasons.append("INVALID_FIRST_STAGE_VARIANCE")
        membership_row = {
            "candidate_id": candidate_id,
            "membership_record_uid": str(
                row.get(stable_key)
                or row.get("record_uid")
                or row.get("response_series_uid")
                or row.get("source_record_uid")
                or ""
            ),
            "membership_status": "excluded" if exclusion_reasons else "included",
            "exclusion_reasons": tuple(exclusion_reasons),
            "source_name": row.get("source_name"),
            "study_uid": row.get("study_uid"),
            "response_series_uid": row.get("response_series_uid"),
            **factors,
        }
        if dependence_unit_field is not None:
            membership_row[dependence_unit_field] = row.get(dependence_unit_field)
        membership.append(membership_row)
        if exclusion_reasons:
            continue
        analysis_row = {
            stable_key: row[stable_key],
            "study_uid": row["study_uid"],
            outcome_name: row[outcome_name],
            **factors,
        }
        if dependence_unit_field is not None:
            analysis_row[dependence_unit_field] = row[dependence_unit_field]
        if observation_level:
            analysis_row["response_series_uid"] = row["response_series_uid"]
            analysis_row["n_rate_kg_ha"] = row["n_rate_kg_ha"]
        elif first_stage_variance_field is not None:
            assert variance is not None
            analysis_row["first_stage_variance"] = variance
            analysis_row["first_stage_weight"] = 1.0 / variance
        normalized.append(analysis_row)
    return (
        tuple(sorted(normalized, key=lambda row: str(row.get(stable_key, "")))),
        tuple(
            sorted(
                membership,
                key=lambda row: (
                    str(row.get("membership_record_uid") or ""),
                    str(row.get("study_uid") or ""),
                ),
            )
        ),
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
        terms.append("(1 | study_uid/response_series_uid)")
    return "yield_t_ha ~ " + " + ".join(terms)


def _design_gate_reason(
    rows: Sequence[Mapping[str, Any]],
    factor_names: Sequence[str],
    *,
    outcome_name: str,
    outcome_kind: str,
    observation_level: bool,
    all_factor_interactions: bool,
) -> str | None:
    frame = pd.DataFrame([{name: row[name] for name in factor_names} for row in rows])
    factor_blocks: dict[str, np.ndarray] = {}
    for factor_name in factor_names:
        values = tuple(frame[factor_name])
        if len({str(value) for value in values}) < 2:
            return "UNIDENTIFIED_FACTOR_VARIATION"
        if all(not isinstance(value, bool) and finite_number(value) is not None for value in values):
            factor_blocks[factor_name] = np.asarray(
                [float(value) for value in values],
                dtype=float,
            ).reshape(-1, 1)
        else:
            encoded = pd.get_dummies(
                pd.Series([str(value) for value in values], name=factor_name),
                drop_first=True,
                dtype=float,
            )
            factor_blocks[factor_name] = encoded.to_numpy(dtype=float)
    columns: list[np.ndarray] = [factor_blocks[name] for name in factor_names]
    if all_factor_interactions and len(factor_names) > 1:
        for order in range(2, len(factor_names) + 1):
            for names in itertools.combinations(factor_names, order):
                products = factor_blocks[names[0]]
                for name in names[1:]:
                    products = np.column_stack(
                        [
                            products[:, left] * factor_blocks[name][:, right]
                            for left in range(products.shape[1])
                            for right in range(factor_blocks[name].shape[1])
                        ]
                    )
                columns.append(products)
    if observation_level:
        n_rate = np.asarray([float(row["n_rate_kg_ha"]) for row in rows], dtype=float)
        n_columns = np.column_stack((n_rate, n_rate**2))
        columns.append(n_columns)
        for name in factor_names:
            block = factor_blocks[name]
            columns.append(
                np.column_stack(
                    [
                        n_columns[:, n_index] * block[:, factor_index]
                        for n_index in range(n_columns.shape[1])
                        for factor_index in range(block.shape[1])
                    ]
                )
            )
    predictors = np.column_stack(columns)
    matrix = np.column_stack((np.ones(len(rows), dtype=float), predictors))
    rank = int(np.linalg.matrix_rank(matrix))
    if rank < matrix.shape[1]:
        return "ALIASED_DESIGN_MATRIX"
    if predictors.shape[1] > 1:
        scales = predictors.std(axis=0)
        if np.any(scales <= np.finfo(float).eps):
            return "ALIASED_DESIGN_MATRIX"
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
        factor_representations=candidate.factor_representations,
        outcome_name=outcome_name,
        stable_key=stable_key,
        observation_level=observation_level,
    )
    if observation_level:
        levels_by_series: dict[str, set[float]] = {}
        for row in normalized:
            series_uid = str(row.get("response_series_uid") or "")
            n_rate = finite_number(row.get("n_rate_kg_ha"))
            if series_uid and n_rate is not None:
                levels_by_series.setdefault(series_uid, set()).add(n_rate)
        supported_series = {
            series_uid
            for series_uid, levels in levels_by_series.items()
            if len(levels) >= 3
        }
        normalized = tuple(
            row
            for row in normalized
            if str(row.get("response_series_uid") or "") in supported_series
        )
        if len(supported_series) < 3:
            return RAnalysisPreparation(
                "skipped",
                ("INSUFFICIENT_WITHIN_SERIES_N_SUPPORT",),
                {},
                normalized,
                stable_key,
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
    minimum_studies = 3 if observation_level else 2
    if len(studies) < minimum_studies:
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
            all_factor_interactions=False,
        )
        if gate_reason is not None:
            return RAnalysisPreparation("skipped", (gate_reason,), {}, normalized, stable_key)
        if not random_intercept:
            return RAnalysisPreparation(
                "skipped",
                ("INSUFFICIENT_NESTED_RANDOM_EFFECT_SUPPORT",),
                {},
                normalized,
                stable_key,
            )
        model_kind = "lmer"
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
            all_factor_interactions=candidate.analysis_family == "all_supported_interactions",
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
        "hypothesis_id": candidate.hypothesis_id,
        "factor_names": list(candidate.factor_names),
        "factor_representations": {
            name: dict(representation)
            for name, representation in candidate.factor_representations.items()
        },
        "support_rule_id": candidate.support_rule_id,
        "support_policy": dict(candidate.support_policy),
        "estimand": dict(candidate.prespecified_contrast),
        "engine": "r",
        "support_gates_passed": True,
        "model_formula": formula,
        "model_kind": model_kind,
        "outcome_kind": outcome_kind,
        "multiplicity": {
            "method": "BH",
            "family_id": candidate.multiplicity_family_id,
            "hypothesis_id": candidate.hypothesis_id,
            "registry_authority": "prespecified_analysis_registry",
            "adjustment_status": "pending_central_reconciliation",
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
        specification["contrast_specification"] = dict(candidate.prespecified_contrast) or {
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
