from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import itertools
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .analysis_matrix import (
    AnalysisCandidate,
    first_stage_policy_reasons,
    first_stage_uncertainty_reasons,
)
from .factor_catalog import factor_data_type, factor_value
from .values import finite_number, is_missing_text


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
    return isinstance(value, (str, bool)) and not is_missing_text(str(value))


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
    require_verified_comparability: bool = False,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    normalized: list[dict[str, Any]] = []
    membership: list[dict[str, Any]] = []
    review_ids_by_context: dict[str, set[str]] = {}
    if require_verified_comparability:
        for row in rows:
            context_uid = str(row.get("comparison_set_uid") or "").strip()
            review_id = str(row.get("recommendation_set_review_id") or "").strip()
            if (
                context_uid
                and review_id
                and row.get("recommendation_set_membership_status")
                == "verified_context_comparable"
            ):
                review_ids_by_context.setdefault(context_uid, set()).add(review_id)
    conflicting_review_contexts = {
        context_uid
        for context_uid, review_ids in review_ids_by_context.items()
        if len(review_ids) > 1
    }
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
        if require_verified_comparability and (
            not isinstance(row.get("treatment_uid"), str)
            or not row.get("treatment_uid")
            or row.get("treatment_classification_status") != "resolved"
            or row.get("recommendation_set_membership_status")
            != "verified_context_comparable"
            or not isinstance(row.get("recommendation_set_review_id"), str)
            or not str(row.get("recommendation_set_review_id") or "").strip()
        ):
            exclusion_reasons.append("COMPARABILITY_EVIDENCE_REQUIRED")
        if (
            require_verified_comparability
            and str(row.get("comparison_set_uid") or "").strip()
            in conflicting_review_contexts
        ):
            exclusion_reasons.append("COMPARABILITY_REVIEW_AUTHORITY_CONFLICT")
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
            "treatment_uid": row.get("treatment_uid"),
            "recommendation_set_membership_status": row.get(
                "recommendation_set_membership_status"
            ),
            "recommendation_set_review_id": row.get(
                "recommendation_set_review_id"
            ),
            "outcome_name": outcome_name,
            "outcome_value": row.get(outcome_name),
            # Plan Section 10.8: the tested N design travels with every
            # curve-feature row into Phase 4. A domain-dependent feature is
            # withheld exactly when its turning point reaches the series' own
            # highest tested rate, so without the truncation point a contrast
            # cannot tell an agronomic difference from a difference between the
            # two groups' N ladders.
            "tested_n_min_kg_ha": row.get("tested_n_min_kg_ha"),
            "tested_n_max_kg_ha": row.get("tested_n_max_kg_ha"),
            "tested_n_level_count": row.get("tested_n_level_count"),
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
        if require_verified_comparability:
            analysis_row["treatment_uid"] = row["treatment_uid"]
            analysis_row["recommendation_set_membership_status"] = row[
                "recommendation_set_membership_status"
            ]
            analysis_row["recommendation_set_review_id"] = row[
                "recommendation_set_review_id"
            ]
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
    grouping_column: str,
) -> str:
    if candidate.analysis_family == "all_supported_interactions":
        fixed_terms = " * ".join(candidate.factor_names)
    else:
        fixed_terms = " + ".join(candidate.factor_names)
    formula = f"{candidate.curve_outcome} ~ {fixed_terms}"
    formula += f" + (1 | {grouping_column})"
    return formula


def _observation_formula(
    factor_names: Sequence[str],
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
    minimum_residual_df: int = 3,
    minimum_class_events_per_parameter: int | None = None,
) -> str | None:
    frame = pd.DataFrame([{name: row[name] for name in factor_names} for row in rows])
    factor_blocks: dict[str, np.ndarray] = {}
    for factor_name in factor_names:
        values = tuple(frame[factor_name])
        if len({str(value) for value in values}) < 2:
            return "UNIDENTIFIED_FACTOR_VARIATION"
        if factor_data_type(factor_name) == "numeric":
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
    if len(rows) - rank < minimum_residual_df:
        return "INSUFFICIENT_RESIDUAL_INFORMATION"
    if outcome_kind == "categorical":
        outcomes = np.asarray([str(row[outcome_name]) for row in rows])
        levels = np.unique(outcomes)
        if minimum_class_events_per_parameter is not None and len(levels):
            # A k-level factor contributes k-1 coefficients; interactions add
            # further columns. Count events on these normalized complete cases,
            # not on the larger registry population before exclusions.
            required_events = minimum_class_events_per_parameter * max(rank - 1, 1)
            if min(Counter(outcomes).values()) < required_events:
                return "INSUFFICIENT_CLASS_EVENTS_PER_PARAMETER"
        if len(levels) == 2:
            # Unique continuous values make every empirical pattern "pure";
            # that does not imply logistic separation. Test whether the actual
            # fixed-effect design admits a strictly separating hyperplane.
            scales = predictors.std(axis=0)
            design = np.column_stack((np.ones(len(rows)), (predictors - predictors.mean(axis=0)) / scales))
            signs = np.where(outcomes == levels[1], 1.0, -1.0)
            separation = linprog(
                np.zeros(design.shape[1]), A_ub=-signs[:, None] * design,
                b_ub=-np.ones(len(rows)), bounds=(None, None), method="highs",
            )
            if separation.success:
                return "COMPLETE_SEPARATION_RISK"
            if separation.status != 2:  # Only proven infeasibility rules out separation.
                return "SEPARATION_CHECK_UNRESOLVED"
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
    stable_key = "record_uid" if observation_level else "response_series_uid"
    model_specification = dict(candidate.model_specification)
    required_model_fields = {
        "outcome_kind",
        "model_kind",
        "dependence_structure",
        "link_function",
        "focal_factor_names",
        "adjustment_factor_names",
        "multiplicity_test_ids",
        "interval_method",
    }
    if not required_model_fields.issubset(model_specification):
        return RAnalysisPreparation(
            "skipped",
            ("PREDECLARED_MODEL_SPECIFICATION_REQUIRED",),
            {},
            (),
            stable_key,
        )
    expected_structure = (
        "nested_random_intercept" if observation_level else "random_intercept"
    )
    if model_specification["dependence_structure"] != expected_structure:
        return RAnalysisPreparation(
            "skipped",
            ("PREDECLARED_MODEL_STRUCTURE_UNSUPPORTED",),
            {},
            (),
            stable_key,
        )
    focal_factors = tuple(model_specification["focal_factor_names"])
    adjustment_factors = tuple(model_specification["adjustment_factor_names"])
    if (
        set(focal_factors) & set(adjustment_factors)
        or set(focal_factors) | set(adjustment_factors)
        != set(candidate.factor_names)
    ):
        return RAnalysisPreparation(
            "skipped",
            ("PREDECLARED_FACTOR_ROLE_PARTITION_INVALID",),
            {},
            (),
            stable_key,
        )
    expected_test_ids = tuple(model_specification["multiplicity_test_ids"])
    if not expected_test_ids or len(expected_test_ids) != len(set(expected_test_ids)):
        return RAnalysisPreparation(
            "skipped",
            ("PREDECLARED_MULTIPLICITY_TEST_IDS_REQUIRED",),
            {},
            (),
            stable_key,
        )
    if (
        candidate.decision_alpha is None
        or not 0.0 < float(candidate.decision_alpha) <= 1.0
    ):
        return RAnalysisPreparation(
            "skipped",
            ("PREDECLARED_DECISION_ALPHA_REQUIRED",),
            {},
            (),
            stable_key,
        )
    contrast_specification = dict(candidate.prespecified_contrast)
    same_context_management_contrast = (
        candidate.analysis_family == "marginal_contrasts"
        and contrast_specification.get("estimand_type")
        == "same_context_management_contrast"
    )
    dependence_unit_field: str | None = None
    if same_context_management_contrast:
        raw_dependence_unit = contrast_specification.get("dependence_unit")
        if (
            contrast_specification.get("same_context_required") is not True
            or not isinstance(raw_dependence_unit, str)
            or raw_dependence_unit != "comparison_set_uid"
        ):
            return RAnalysisPreparation(
                "skipped",
                ("VERIFIED_SAME_CONTEXT_ESTIMAND_REQUIRED",),
                {},
                (),
                "record_uid" if observation_level else "response_series_uid",
            )
        dependence_unit_field = raw_dependence_unit
    expected_grouping = (
        ("study_uid", "response_series_uid")
        if observation_level
        else (
            (dependence_unit_field,)
            if dependence_unit_field is not None
            else ("study_uid",)
        )
    )
    if not candidate.grouping:
        return RAnalysisPreparation(
            "skipped",
            ("PREDECLARED_GROUPING_REQUIRED",),
            {},
            (),
            "record_uid" if observation_level else "response_series_uid",
        )
    if candidate.grouping != expected_grouping:
        return RAnalysisPreparation(
            "skipped",
            ("PREDECLARED_GROUPING_UNSUPPORTED",),
            {},
            (),
            "record_uid" if observation_level else "response_series_uid",
        )
    if not candidate.factor_names:
        return RAnalysisPreparation(
            "skipped",
            ("R_FACTOR_SET_REQUIRED",),
            {},
            (),
            "record_uid" if observation_level else "response_series_uid",
        )
    if candidate.analysis_family == "marginal_contrasts":
        required_contrast_fields = {
            "estimand_id",
            "treatment",
            "comparator",
            "direction",
        }
        if not required_contrast_fields.issubset(contrast_specification):
            return RAnalysisPreparation(
                "skipped",
                ("PRESPECIFIED_CONTRAST_REQUIRED",),
                {},
                (),
                stable_key,
            )
        contrast_specification.setdefault(
            "factor_name",
            candidate.factor_names[0],
        )
        if contrast_specification["factor_name"] != candidate.factor_names[0]:
            return RAnalysisPreparation(
                "skipped",
                ("PRESPECIFIED_CONTRAST_FACTOR_MISMATCH",),
                {},
                (),
                stable_key,
            )
        estimand_id = str(contrast_specification["estimand_id"])
        if expected_test_ids != (estimand_id,):
            return RAnalysisPreparation(
                "skipped",
                ("PREDECLARED_MULTIPLICITY_TEST_MISMATCH",),
                {},
                (),
                stable_key,
            )
    if same_context_management_contrast:
        treatment = contrast_specification.get("treatment")
        comparator = contrast_specification.get("comparator")
        if (
            len(candidate.factor_names) != 1
            or not isinstance(treatment, str)
            or not treatment
            or not isinstance(comparator, str)
            or not comparator
            or treatment == comparator
        ):
            return RAnalysisPreparation(
                "skipped",
                ("PRESPECIFIED_MANAGEMENT_CONTRAST_REQUIRED",),
                {},
                (),
                "record_uid" if observation_level else "response_series_uid",
            )
        contrast_specification["factor_name"] = candidate.factor_names[0]
        contrast_specification.setdefault("adjustment", "BH")
    outcome_name = "yield_t_ha" if observation_level else candidate.curve_outcome
    first_stage_variance_field: str | None = None
    fitted_feature_inference = (
        not observation_level
        and candidate.analysis_family in _FITTED_FEATURE_INFERENTIAL_FAMILIES
    )
    if fitted_feature_inference:
        first_stage_variance_field = f"{outcome_name}_first_stage_variance"
    normalized, membership_rows = _normalized_rows(
        rows,
        candidate.factor_names,
        candidate_id=candidate.candidate_id,
        factor_representations=candidate.factor_representations,
        outcome_name=outcome_name,
        stable_key=stable_key,
        observation_level=observation_level,
        first_stage_variance_field=(
            None if fitted_feature_inference else first_stage_variance_field
        ),
        dependence_unit_field=dependence_unit_field,
        require_verified_comparability=same_context_management_contrast,
    )
    if fitted_feature_inference and normalized:
        included_keys = {str(row[stable_key]) for row in normalized}
        included_rows = tuple(
            row
            for row in rows
            if str(row.get(stable_key) or "") in included_keys
        )
        policy_reasons = first_stage_policy_reasons(
            included_rows,
            outcome_name,
            candidate.first_stage_uncertainty_policy,
        )
        if policy_reasons:
            return RAnalysisPreparation(
                "skipped",
                policy_reasons,
                {},
                normalized,
                stable_key,
                membership_rows,
            )
        first_stage_reasons = first_stage_uncertainty_reasons(
            included_rows,
            outcome_name,
        )
        if first_stage_reasons:
            return RAnalysisPreparation(
                "skipped",
                first_stage_reasons,
                {},
                normalized,
                stable_key,
                membership_rows,
            )
        normalized, membership_rows = _normalized_rows(
            rows,
            candidate.factor_names,
            candidate_id=candidate.candidate_id,
            factor_representations=candidate.factor_representations,
            outcome_name=outcome_name,
            stable_key=stable_key,
            observation_level=observation_level,
            first_stage_variance_field=first_stage_variance_field,
            dependence_unit_field=dependence_unit_field,
            require_verified_comparability=same_context_management_contrast,
        )
    factor_representation_contracts: dict[str, dict[str, Any]] = {}
    for factor_name, raw_representation in candidate.factor_representations.items():
        representation = dict(raw_representation)
        if representation.get("data_type") in {
            "categorical",
            "ordinal",
            "boolean",
        }:
            reference_level = representation.get("reference_level")
            observed_levels = {
                str(row[factor_name])
                for row in normalized
                if row.get(factor_name) is not None
            }
            if (
                not isinstance(reference_level, str)
                or reference_level not in observed_levels
            ):
                return RAnalysisPreparation(
                    "skipped",
                    (f"REVIEWED_REFERENCE_LEVEL_UNAVAILABLE:{factor_name}",),
                    {},
                    normalized,
                    stable_key,
                    membership_rows,
                )
            representation["approved_levels"] = [
                reference_level,
                *sorted(observed_levels - {reference_level}),
            ]
        factor_representation_contracts[factor_name] = representation
    if same_context_management_contrast:
        assert dependence_unit_field is not None
        factor_name = candidate.factor_names[0]
        treatment = str(contrast_specification["treatment"])
        comparator = str(contrast_specification["comparator"])
        levels_by_context: dict[str, set[str]] = {}
        for row in normalized:
            context_uid = str(row.get(dependence_unit_field) or "")
            if context_uid:
                levels_by_context.setdefault(context_uid, set()).add(
                    str(row.get(factor_name) or "")
                )
        complete_contexts = {
            context_uid
            for context_uid, levels in levels_by_context.items()
            if {treatment, comparator}.issubset(levels)
        }
        normalized = tuple(
            row
            for row in normalized
            if str(row.get(dependence_unit_field) or "") in complete_contexts
            and str(row.get(factor_name) or "") in {treatment, comparator}
        )
        membership_rows = tuple(
            {
                **row,
                "membership_status": "excluded",
                "exclusion_reasons": (
                    *tuple(row.get("exclusion_reasons") or ()),
                    "PRESPECIFIED_CONTRAST_PAIR_INCOMPLETE",
                ),
            }
            if row.get("membership_status") == "included"
            and (
                str(row.get(dependence_unit_field) or "") not in complete_contexts
                or str(row.get(factor_name) or "") not in {treatment, comparator}
            )
            else row
            for row in membership_rows
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
        membership_rows = tuple(
            {
                **row,
                "membership_status": "excluded",
                "exclusion_reasons": (
                    *tuple(row.get("exclusion_reasons") or ()),
                    "INSUFFICIENT_WITHIN_SERIES_N_SUPPORT",
                ),
            }
            if row.get("membership_status") == "included"
            and str(row.get("response_series_uid") or "") not in supported_series
            else row
            for row in membership_rows
        )
        if len(supported_series) < 3:
            return RAnalysisPreparation(
                "skipped",
                ("INSUFFICIENT_WITHIN_SERIES_N_SUPPORT",),
                {},
                normalized,
                stable_key,
                membership_rows,
            )
    minimum_rows = 8 if observation_level else max(4, len(candidate.factor_names) + 3)
    if len(normalized) < minimum_rows:
        return RAnalysisPreparation(
            "skipped",
            ("INSUFFICIENT_NORMALIZED_R_ROWS",),
            {},
            normalized,
            stable_key,
            membership_rows,
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
            membership_rows,
        )

    grouping_column = str(expected_grouping[0])
    random_intercept = _has_supported_random_intercept(
        normalized,
        grouping_column,
    )
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
                membership_rows,
            )
        outcome_kind = "continuous"
        if (
            model_specification["outcome_kind"] != outcome_kind
            or model_specification["model_kind"] != "lmer"
            or model_specification["link_function"] != "identity"
        ):
            return RAnalysisPreparation(
                "skipped",
                ("PREDECLARED_MODEL_KIND_OUTCOME_MISMATCH",),
                {},
                normalized,
                stable_key,
                membership_rows,
            )
        gate_reason = _design_gate_reason(
            normalized,
            candidate.factor_names,
            outcome_name=outcome_name,
            outcome_kind=outcome_kind,
            observation_level=True,
            all_factor_interactions=False,
            minimum_residual_df=int(candidate.support_policy.get("minimum_residual_df", 3)),
            minimum_class_events_per_parameter=candidate.support_policy.get(
                "minimum_class_events_per_parameter"
            ),
        )
        if gate_reason is not None:
            return RAnalysisPreparation(
                "skipped",
                (gate_reason,),
                {},
                normalized,
                stable_key,
                membership_rows,
            )
        if not random_intercept:
            return RAnalysisPreparation(
                "skipped",
                ("PREDECLARED_GROUPING_NOT_IDENTIFIABLE",),
                {},
                normalized,
                stable_key,
                membership_rows,
            )
        model_kind = str(model_specification["model_kind"])
        formula = _observation_formula(candidate.factor_names)
    else:
        outcome_kind = (
            "categorical"
            if candidate.curve_outcome in _CATEGORICAL_OUTCOMES
            else "continuous"
        )
        if model_specification["outcome_kind"] != outcome_kind:
            return RAnalysisPreparation(
                "skipped",
                ("PREDECLARED_MODEL_KIND_OUTCOME_MISMATCH",),
                {},
                normalized,
                stable_key,
                membership_rows,
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
                    membership_rows,
                )
            if len(level_counts) > 2:
                return RAnalysisPreparation(
                    "skipped",
                    ("PREDECLARED_GROUPING_UNSUPPORTED_FOR_MULTINOMIAL_OUTCOME",),
                    {},
                    normalized,
                    stable_key,
                    membership_rows,
                )
            expected_model_kind = "glmmTMB"
            expected_link_function = "logit"
        else:
            expected_model_kind = "lmer"
            expected_link_function = "identity"
        if (
            model_specification["model_kind"] != expected_model_kind
            or model_specification["link_function"] != expected_link_function
        ):
            return RAnalysisPreparation(
                "skipped",
                ("PREDECLARED_MODEL_KIND_OUTCOME_MISMATCH",),
                {},
                normalized,
                stable_key,
                membership_rows,
            )
        model_kind = str(model_specification["model_kind"])
        gate_reason = _design_gate_reason(
            normalized,
            candidate.factor_names,
            outcome_name=outcome_name,
            outcome_kind=outcome_kind,
            observation_level=False,
            all_factor_interactions=candidate.analysis_family == "all_supported_interactions",
            minimum_residual_df=int(candidate.support_policy.get("minimum_residual_df", 3)),
            minimum_class_events_per_parameter=candidate.support_policy.get(
                "minimum_class_events_per_parameter"
            ),
        )
        if gate_reason is not None:
            return RAnalysisPreparation(
                "skipped",
                (gate_reason,),
                {},
                normalized,
                stable_key,
                membership_rows,
            )
        if not random_intercept:
            return RAnalysisPreparation(
                "skipped",
                ("PREDECLARED_GROUPING_NOT_IDENTIFIABLE",),
                {},
                normalized,
                stable_key,
                membership_rows,
            )
        formula = _curve_formula(
            candidate,
            grouping_column=grouping_column,
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
        "factor_representations": factor_representation_contracts,
        "support_rule_id": candidate.support_rule_id,
        "support_policy": dict(candidate.support_policy),
        "estimand": dict(candidate.prespecified_contrast),
        "engine": "r",
        "support_gates_passed": True,
        "model_formula": formula,
        "model_kind": model_kind,
        "model_structure": expected_structure,
        "model_specification": model_specification,
        "outcome_kind": outcome_kind,
        "multiplicity": {
            "method": "BH",
            "family_id": candidate.multiplicity_family_id,
            "hypothesis_id": candidate.hypothesis_id,
            "family_scope_complete": (
                candidate.multiplicity_family_id is None
                or bool(candidate.hypothesis_id)
            ),
            "registry_authority": "prespecified_analysis_registry",
            "adjustment_status": "pending_central_reconciliation",
            "expected_test_ids": list(expected_test_ids),
            "alpha": float(candidate.decision_alpha),
        },
        "grouping_column": grouping_column,
        "predeclared_grouping": list(candidate.grouping),
        "missing_data_policy": "factor-specific complete cases; no imputation",
    }
    if first_stage_variance_field is not None:
        method_field = f"{outcome_name}_first_stage_uncertainty_method_id"
        method_ids = sorted(
            {
                str(row[method_field])
                for row in rows
                if isinstance(row.get(method_field), str) and row.get(method_field)
            }
        )
        specification["first_stage_uncertainty"] = {
            "decision_id": "ANA-16",
            "mode": "variance_aware_two_stage",
            "policy_id": candidate.first_stage_uncertainty_policy["policy_id"],
            "policy_authority_sha256": candidate.first_stage_uncertainty_policy[
                "authority_sha256"
            ],
            "method_id": method_ids[0] if len(method_ids) == 1 else None,
            "source_variance_field": first_stage_variance_field,
            "variance_column": "first_stage_variance",
            "weight_column": "first_stage_weight",
            "weighting": "inverse_first_stage_variance",
            "model_selection_uncertainty": "incorporated",
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
        specification["contrast_specification"] = contrast_specification
    return RAnalysisPreparation(
        "run",
        (),
        specification,
        normalized,
        stable_key,
        membership_rows,
    )


__all__ = ["RAnalysisPreparation", "prepare_r_analysis"]
