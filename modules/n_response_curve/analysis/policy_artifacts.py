from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..data.provenance import sha256_file
from .analysis_matrix import INFERENTIAL_ANALYSIS_FAMILIES
from .claims import validate_claim_policy
from .reviewed_methods import (
    ECONOMIC_DECISION_RULE,
    ECONOMIC_GRAIN_PRICE_TO_PER_TONNE,
    ECONOMIC_N_COST_UNIT,
    UNCERTAINTY_METHOD_CONFIDENCE_LEVELS,
    UNCERTAINTY_METHOD_REQUIRED_EVIDENCE,
    UNCERTAINTY_METHOD_SPECS,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_HYPOTHESIS_IDS = frozenset(
    {
        "curve_form_context",
        "agronomic_optimum_context",
        "maximum_yield_context",
        "management_system_contrasts",
        "target_yield_contrast",
    }
)
_ANALYSIS_ENGINES = frozenset({"python", "r"})
_ANA_APPROVED_FACTOR_ROSTER = frozenset(
    {
        "water_regime",
        "season",
        "planting_year",
        "region",
        "province",
        "variety",
        "recommendation_scope",
        "recommendation_class",
    }
)
_ANA03_SECONDARY_OUTCOMES = frozenset(
    {"yield_at_zero_n_t_ha", "yield_response_above_zero_n_t_ha"}
)
_ANA03_PRIMARY_OUTCOMES = frozenset(
    {
        "curve_shape_class",
        "optimum_status",
        "agronomic_optimum_n_kg_ha",
        "plateau_onset_n_kg_ha",
        "predicted_observed_domain_peak_yield_t_ha",
        "finite_maximum_yield_t_ha",
        "supported_max_yield_t_ha",
    }
)
_MODEL_ROSTER = (
    "linear",
    "quadratic",
    "linear_plateau",
    "quadratic_plateau",
    "mitscherlich",
)
_MODEL_PARAMETER_NAMES = {
    "linear": ("intercept", "slope"),
    "quadratic": ("intercept", "slope", "curvature"),
    "linear_plateau": ("intercept", "slope", "plateau_onset"),
    "quadratic_plateau": ("baseline", "gain", "plateau_onset"),
    "mitscherlich": ("asymptote", "amplitude", "rate"),
}
_MODEL_INITIALIZATION_STRATEGIES = {
    "linear": "ordinary_least_squares",
    "quadratic": "ordinary_least_squares",
    "linear_plateau": "deterministic_data_anchored",
    "quadratic_plateau": "deterministic_data_anchored",
    "mitscherlich": "deterministic_data_anchored",
}
_MODEL_SHAPE_CLASSES = {
    "linear": frozenset(
        {"increasing_linear", "decreasing_linear", "flat_linear"}
    ),
    "quadratic": frozenset(
        {
            "weak_quadratic_curvature",
            "concave_quadratic",
            "diminishing_returns",
            "declining_concave",
            "concave_boundary_peak",
            "accelerating_returns",
            "convex_decline",
            "convex_boundary_minimum",
        }
    ),
    "linear_plateau": frozenset(
        {"plateau", "plateau_without_supported_onset"}
    ),
    "quadratic_plateau": frozenset(
        {"plateau", "plateau_without_supported_onset"}
    ),
    "mitscherlich": frozenset(
        {
            "asymptotic_diminishing_returns",
            "asymptotic_without_supported_asymptote",
        }
    ),
}
_DISAGREEMENT_FIELDS = frozenset(
    {
        "agronomic_optimum_n_kg_ha",
        "plateau_onset_n_kg_ha",
        "supported_max_yield_t_ha",
    }
)
_UNCERTAINTY_EVIDENCE_BASES = frozenset(
    {
        "reported_standard_error",
        "verified_mean_independence",
        "verified_true_replication",
    }
)
_FIRST_STAGE_NUMERIC_OUTCOMES = frozenset(
    {
        "agronomic_optimum_n_kg_ha",
        "plateau_onset_n_kg_ha",
        "predicted_max_yield_t_ha",
        "predicted_observed_domain_peak_yield_t_ha",
        "finite_maximum_yield_t_ha",
        "fitted_asymptote_yield_t_ha",
        "supported_max_yield_t_ha",
    }
)
_BASELINE_CLASSES = ("zero_n_with_pk", "absolute_control")
_RECOMMENDATION_REQUIRED_CLASSES = ("zero_n_with_pk", "RCM", "NOPT_NPK")
_RECOMMENDATION_OPTIONAL_CLASSES = ("FP",)
_ECONOMIC_SCENARIO_FIELDS = frozenset(
    {
        "scenario_id",
        "grain_price",
        "grain_price_unit",
        "n_cost",
        "n_cost_unit",
        "currency",
        "reference_period",
        "price_basis",
        "tax_subsidy_application_cost_basis",
        "decision_rule",
    }
)
class PolicyArtifactError(ValueError):
    """Raised when a scientific policy artifact is incomplete or unauthenticated."""


@dataclass(frozen=True)
class ArtifactAuthority:
    artifact_type: str
    artifact_version: str
    path: Path
    sha256: str
    approved_by: str
    approval_date: str
    scope: str
    archive_location: str


@dataclass(frozen=True)
class SupportRule:
    rule_id: str
    outcome: str
    analysis_family: str
    factor_type: str
    minimum_independent_series: int
    minimum_independent_studies: int
    minimum_observations_per_cell: int
    minimum_class_events_per_parameter: int
    minimum_residual_df: int
    maximum_missing_fraction: float
    maximum_factor_cardinality: int
    maximum_loso_studies: int
    sensitivity_checks: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class FactorRepresentation:
    representation_id: str
    factor_name: str
    engine: str
    role: str
    source_fields: tuple[str, ...]
    data_type: str
    unit: str | None
    transformation: str
    reference_level: str | None
    category_map: Mapping[str, str]
    missingness_rule: str
    leakage_exclusions: tuple[str, ...]
    learned_within_training_only: bool


@dataclass(frozen=True)
class Estimand:
    estimand_id: str
    hypothesis_id: str
    estimand_type: str
    outcome: str
    treatment: str
    comparator: str
    direction: str
    target_population: str
    same_context_required: bool
    dependence_unit: str
    unit: str
    eligibility_rule: str
    unavailable_reason_field: str


@dataclass(frozen=True)
class EffectiveHypothesis:
    hypothesis_id: str
    status: str
    disabled_reason: str | None
    evidence_role: str
    dataset_version_id: str
    source_view: str
    population: str
    outcome: str
    estimand_id: str
    analysis_family: str
    factor_names: tuple[str, ...]
    grouping: tuple[str, ...]
    model_specification: Mapping[str, Any]
    support_rule_id: str
    engine: str
    multiplicity_family_id: str
    alpha: float
    sensitivities: tuple[str, ...]
    sensitivity_specifications: tuple[Mapping[str, Any], ...]
    claim_policy: Mapping[str, Any]


@dataclass(frozen=True)
class CurveModelGate:
    """One reviewed candidate's deterministic fitting and plausibility controls."""

    initialization_strategy: str
    parameter_bounds: Mapping[str, tuple[float, float]]
    allow_boundary_parameters: bool
    reportable_shape_classes: tuple[str, ...]
    optimizer_tolerance: float
    optimizer_max_iterations: int
    parameter_boundary_relative_tolerance: float
    optimum_boundary_tolerance_n_kg_ha: float
    flat_response_tolerance_t_ha: float


@dataclass(frozen=True)
class CurveModelPolicy:
    """Authenticated scientific controls consumed by curve and dataset behavior."""

    policy_id: str
    restricted_fit_models: tuple[str, ...]
    model_gates: Mapping[str, CurveModelGate]
    model_credibility_policy: Mapping[str, Any]
    baseline_policy_id: str
    baseline_response_policy: str
    baseline_classes: tuple[str, ...]
    recommendation_policy_id: str
    recommendation_required_classes: tuple[str, ...]
    recommendation_optional_classes: tuple[str, ...]
    recommendation_membership_status_field: str
    recommendation_verified_status: str
    material_disagreement_policy_id: str
    material_disagreement_review_status: str
    material_disagreement_tolerances: Mapping[str, float]
    uncertainty_policy_id: str
    uncertainty_review_status: str
    uncertainty_method: str | None
    uncertainty_method_contract: Mapping[str, Any]
    uncertainty_evidence_basis: tuple[str, ...]
    first_stage_contextual_uncertainty_policy: Mapping[str, Any]
    economic_policy_id: str
    economic_review_status: str
    economic_table_id: str | None
    economic_table_version: str | None
    economic_scenarios: tuple[Mapping[str, Any], ...]
    efficiency_metric_policy: Mapping[str, Any]
    efficiency_operating_point_policy: Mapping[str, Any]
    asymptote_reporting_policy: Mapping[str, Any]
    asymptote_support_policy: Mapping[str, Any]

    @property
    def effective_controls(self) -> Mapping[str, Any]:
        """Return deeply immutable controls shaped for the curve runtime."""

        models = {
            model_name: MappingProxyType(
                {
                    "initialization_strategy": gate.initialization_strategy,
                    "parameter_bounds": gate.parameter_bounds,
                    "allow_boundary_parameters": gate.allow_boundary_parameters,
                    "reportable_shape_classes": gate.reportable_shape_classes,
                    "optimizer_tolerance": gate.optimizer_tolerance,
                    "optimizer_max_iterations": gate.optimizer_max_iterations,
                    "parameter_boundary_relative_tolerance": gate.parameter_boundary_relative_tolerance,
                    "optimum_boundary_tolerance_n_kg_ha": gate.optimum_boundary_tolerance_n_kg_ha,
                    "flat_response_tolerance_t_ha": gate.flat_response_tolerance_t_ha,
                }
            )
            for model_name, gate in self.model_gates.items()
        }
        return MappingProxyType(
            {
                "restricted_fit_models": self.restricted_fit_models,
                "model_gate_policy": MappingProxyType(
                    {
                        "policy_id": self.policy_id,
                        "review_status": "approved",
                        "models": MappingProxyType(models),
                    }
                ),
                "model_credibility_policy": self.model_credibility_policy,
                "baseline_response_policy": self.baseline_response_policy,
                "baseline_response_classes": self.baseline_classes,
                "baseline_response_policy_id": self.baseline_policy_id,
                "recommendation_set_policy": MappingProxyType(
                    {
                        "policy_id": self.recommendation_policy_id,
                        "review_status": "approved",
                        "required_classes": self.recommendation_required_classes,
                        "optional_classes": self.recommendation_optional_classes,
                        "membership_status_field": (
                            self.recommendation_membership_status_field
                        ),
                        "verified_status": self.recommendation_verified_status,
                    }
                ),
                "material_disagreement_policy": MappingProxyType(
                    {
                        "policy_id": self.material_disagreement_policy_id,
                        "review_status": (
                            self.material_disagreement_review_status
                        ),
                        "tolerances": self.material_disagreement_tolerances,
                    }
                ),
                "uncertainty_policy": MappingProxyType(
                    {
                        "policy_id": self.uncertainty_policy_id,
                        "review_status": self.uncertainty_review_status,
                        "method": self.uncertainty_method,
                        "method_contract": self.uncertainty_method_contract,
                        "evidence_basis": self.uncertainty_evidence_basis,
                    }
                ),
                "first_stage_contextual_uncertainty_policy": (
                    self.first_stage_contextual_uncertainty_policy
                ),
                "economic_scenario_table": MappingProxyType(
                    {
                        "policy_id": self.economic_policy_id,
                        "review_status": self.economic_review_status,
                        "table_id": self.economic_table_id,
                        "version": self.economic_table_version,
                        "scenarios": self.economic_scenarios,
                    }
                ),
                "efficiency_policy": self.efficiency_metric_policy,
                "efficiency_operating_point_policy": (
                    self.efficiency_operating_point_policy
                ),
                "asymptote_reporting_policy": self.asymptote_reporting_policy,
                "asymptote_support_policy": self.asymptote_support_policy,
            }
        )


@dataclass(frozen=True)
class AnalysisPolicyBundle:
    support_authority: ArtifactAuthority
    representation_authority: ArtifactAuthority
    estimand_authority: ArtifactAuthority
    hypothesis_authority: ArtifactAuthority
    model_authority: ArtifactAuthority
    support_rules: tuple[SupportRule, ...]
    factor_representations: tuple[FactorRepresentation, ...]
    estimands: tuple[Estimand, ...]
    hypotheses: tuple[EffectiveHypothesis, ...]
    curve_model_policy: CurveModelPolicy
    manifest_authority: ArtifactAuthority | None = None

    @property
    def artifact_sha256(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "support_table": self.support_authority.sha256,
                "factor_representations": self.representation_authority.sha256,
                "estimands": self.estimand_authority.sha256,
                "hypotheses": self.hypothesis_authority.sha256,
                "curve_model_policy": self.model_authority.sha256,
            }
        )

    def runtime_hypothesis_specifications(
        self,
        *,
        source_view_ids: Mapping[str, str],
    ) -> tuple[Mapping[str, Any], ...]:
        """Materialize enabled reviewed rows for the runtime registry.

        The artifact bundle remains the sole semantic authority.  This adapter
        only binds a reviewed source-view name to the deterministic combination
        identifier created for the current run; it does not infer or broaden a
        hypothesis, support rule, factor representation, or estimand.
        """

        support_by_id = {rule.rule_id: rule for rule in self.support_rules}
        estimand_by_id = {item.estimand_id: item for item in self.estimands}
        representation_by_key = {
            (item.factor_name, item.engine): item
            for item in self.factor_representations
        }
        first_stage_uncertainty_policy = MappingProxyType(
            {
                **dict(
                    self.curve_model_policy.
                    first_stage_contextual_uncertainty_policy
                ),
                "authority_sha256": self.model_authority.sha256,
            }
        )
        specifications: list[Mapping[str, Any]] = []
        for hypothesis in self.hypotheses:
            if hypothesis.status != "enabled":
                continue
            source_combination_id = source_view_ids.get(hypothesis.source_view)
            if not isinstance(source_combination_id, str) or not source_combination_id:
                raise PolicyArtifactError(
                    f"Enabled hypothesis {hypothesis.hypothesis_id} source view "
                    f"{hypothesis.source_view!r} is not bound to one runtime source combination"
                )
            support_rule = support_by_id[hypothesis.support_rule_id]
            estimand = estimand_by_id[hypothesis.estimand_id]
            factor_representations: dict[str, Mapping[str, Any]] = {}
            for factor_name in hypothesis.factor_names:
                representation = representation_by_key[(factor_name, hypothesis.engine)]
                factor_representations[factor_name] = MappingProxyType(
                    {
                        "representation_id": representation.representation_id,
                        "factor_name": representation.factor_name,
                        "engine": representation.engine,
                        "role": representation.role,
                        "data_type": representation.data_type,
                        "unit": representation.unit,
                        "source_fields": representation.source_fields,
                        "transformation": representation.transformation,
                        "reference_level": representation.reference_level,
                        "category_map": representation.category_map,
                        "missingness_rule": representation.missingness_rule,
                        "leakage_exclusions": representation.leakage_exclusions,
                        "learned_within_training_only": (
                            representation.learned_within_training_only
                        ),
                    }
                )
            support_policy = MappingProxyType(
                {
                    "factor_type": support_rule.factor_type,
                    "minimum_independent_series": support_rule.minimum_independent_series,
                    "minimum_observations_per_cell": support_rule.minimum_observations_per_cell,
                    "minimum_class_events_per_parameter": (
                        support_rule.minimum_class_events_per_parameter
                    ),
                    "minimum_residual_df": support_rule.minimum_residual_df,
                    "maximum_missing_fraction": support_rule.maximum_missing_fraction,
                    "maximum_factor_cardinality": support_rule.maximum_factor_cardinality,
                    "minimum_independent_studies": support_rule.minimum_independent_studies,
                    "maximum_loso_studies": support_rule.maximum_loso_studies,
                }
            )
            contrast_specification = MappingProxyType(
                {
                    "estimand_id": estimand.estimand_id,
                    "estimand_type": estimand.estimand_type,
                    "outcome": estimand.outcome,
                    "treatment": estimand.treatment,
                    "comparator": estimand.comparator,
                    "direction": estimand.direction,
                    "target_population": estimand.target_population,
                    "same_context_required": estimand.same_context_required,
                    "dependence_unit": estimand.dependence_unit,
                    "unit": estimand.unit,
                    "eligibility_rule": estimand.eligibility_rule,
                    "unavailable_reason_field": estimand.unavailable_reason_field,
                    "hypothesis_population": hypothesis.population,
                    "grouping": hypothesis.grouping,
                    "evidence_role": hypothesis.evidence_role,
                    "alpha": hypothesis.alpha,
                    "sensitivities": hypothesis.sensitivities,
                    "claim_policy": hypothesis.claim_policy,
                }
            )
            specifications.append(
                MappingProxyType(
                    {
                        "specification_id": hypothesis.hypothesis_id,
                        "hypothesis_id": hypothesis.hypothesis_id,
                        "dataset_version_id": hypothesis.dataset_version_id,
                        "source_combination_id": source_combination_id,
                        "curve_outcome": hypothesis.outcome,
                        "outcome_role": hypothesis.evidence_role,
                        "factor_names": hypothesis.factor_names,
                        "grouping": hypothesis.grouping,
                        "model_specification": hypothesis.model_specification,
                        "alpha": hypothesis.alpha,
                        "contrast_specification": contrast_specification,
                        "analysis_family": hypothesis.analysis_family,
                        "engine": hypothesis.engine,
                        "multiplicity_family_id": hypothesis.multiplicity_family_id,
                        "support_rule_id": hypothesis.support_rule_id,
                        "support_policy": support_policy,
                        "factor_representations": MappingProxyType(
                            factor_representations
                        ),
                        "first_stage_uncertainty_policy": first_stage_uncertainty_policy,
                    }
                )
            )
            for check_index, support_sensitivity in enumerate(
                support_rule.sensitivity_checks,
                start=1,
            ):
                field = str(support_sensitivity["field"])
                for value_index, value in enumerate(
                    support_sensitivity["values"],
                    start=1,
                ):
                    if support_policy[field] == value:
                        continue
                    sensitivity_id = (
                        f"support_threshold_{check_index}_{value_index}"
                    )
                    sensitivity_policy = dict(support_policy)
                    sensitivity_policy[field] = value
                    sensitivity_contrast = dict(contrast_specification)
                    sensitivity_contrast.update(
                        {
                            "evidence_role": "sensitivity",
                            "sensitivity_type": "support_threshold",
                            "sensitivity_id": sensitivity_id,
                            "support_threshold_field": field,
                            "support_threshold_value": value,
                            "parent_hypothesis_id": hypothesis.hypothesis_id,
                            "sensitivities": (),
                            "claim_policy": MappingProxyType(
                                {
                                    "rule_id": (
                                        f"{hypothesis.hypothesis_id}-{sensitivity_id}"
                                    ),
                                    "review_status": "withheld",
                                    "reason": (
                                        "support_threshold_sensitivity_not_primary_claim"
                                    ),
                                }
                            ),
                        }
                    )
                    specifications.append(
                        MappingProxyType(
                            {
                                "specification_id": (
                                    f"{hypothesis.hypothesis_id}:{sensitivity_id}"
                                ),
                                "hypothesis_id": hypothesis.hypothesis_id,
                                "dataset_version_id": hypothesis.dataset_version_id,
                                "source_combination_id": source_combination_id,
                                "curve_outcome": hypothesis.outcome,
                                "outcome_role": hypothesis.evidence_role,
                                "factor_names": hypothesis.factor_names,
                                "grouping": hypothesis.grouping,
                                "model_specification": hypothesis.model_specification,
                                "alpha": hypothesis.alpha,
                                "contrast_specification": MappingProxyType(
                                    sensitivity_contrast
                                ),
                                "analysis_family": hypothesis.analysis_family,
                                "engine": hypothesis.engine,
                                "multiplicity_family_id": (
                                    f"{hypothesis.multiplicity_family_id}:support_sensitivity"
                                ),
                                "support_rule_id": hypothesis.support_rule_id,
                                "support_policy": MappingProxyType(
                                    sensitivity_policy
                                ),
                                "factor_representations": MappingProxyType(
                                    factor_representations
                                ),
                                "first_stage_uncertainty_policy": (
                                    first_stage_uncertainty_policy
                                ),
                            }
                        )
                    )
            for sensitivity in hypothesis.sensitivity_specifications:
                sensitivity_id = str(sensitivity["sensitivity_id"])
                sensitivity_source_view = str(sensitivity["source_view"])
                sensitivity_source_id = source_view_ids.get(sensitivity_source_view)
                if not isinstance(sensitivity_source_id, str) or not sensitivity_source_id:
                    raise PolicyArtifactError(
                        f"Enabled hypothesis {hypothesis.hypothesis_id} sensitivity "
                        f"{sensitivity_id} source view {sensitivity_source_view!r} is not "
                        "bound to one runtime source combination"
                    )
                sensitivity_support = support_by_id[str(sensitivity["support_rule_id"])]
                sensitivity_support_policy = MappingProxyType(
                    {
                        "factor_type": sensitivity_support.factor_type,
                        "minimum_independent_series": (
                            sensitivity_support.minimum_independent_series
                        ),
                        "minimum_observations_per_cell": (
                            sensitivity_support.minimum_observations_per_cell
                        ),
                        "minimum_class_events_per_parameter": (
                            sensitivity_support.minimum_class_events_per_parameter
                        ),
                        "minimum_residual_df": sensitivity_support.minimum_residual_df,
                        "maximum_missing_fraction": (
                            sensitivity_support.maximum_missing_fraction
                        ),
                        "maximum_factor_cardinality": (
                            sensitivity_support.maximum_factor_cardinality
                        ),
                        "minimum_independent_studies": (
                            sensitivity_support.minimum_independent_studies
                        ),
                        "maximum_loso_studies": (
                            sensitivity_support.maximum_loso_studies
                        ),
                    }
                )
                sensitivity_representations: dict[str, Mapping[str, Any]] = {}
                sensitivity_engine = str(sensitivity["engine"])
                for factor_name in sensitivity["factor_names"]:
                    representation = representation_by_key[(factor_name, sensitivity_engine)]
                    sensitivity_representations[factor_name] = MappingProxyType(
                        {
                            "representation_id": representation.representation_id,
                            "factor_name": representation.factor_name,
                            "engine": representation.engine,
                            "role": representation.role,
                            "data_type": representation.data_type,
                            "unit": representation.unit,
                            "source_fields": representation.source_fields,
                            "transformation": representation.transformation,
                            "reference_level": representation.reference_level,
                            "category_map": representation.category_map,
                            "missingness_rule": representation.missingness_rule,
                            "leakage_exclusions": representation.leakage_exclusions,
                            "learned_within_training_only": (
                                representation.learned_within_training_only
                            ),
                        }
                    )
                sensitivity_contrast = dict(contrast_specification)
                sensitivity_contrast.update(
                    {
                        "evidence_role": "sensitivity",
                        "sensitivity_id": sensitivity_id,
                        "parent_hypothesis_id": hypothesis.hypothesis_id,
                        "sensitivities": (),
                        "claim_policy": MappingProxyType(
                            {
                                "rule_id": f"{hypothesis.hypothesis_id}-{sensitivity_id}",
                                "review_status": "withheld",
                                "reason": "sensitivity_result_not_primary_claim",
                            }
                        ),
                    }
                )
                specifications.append(
                    MappingProxyType(
                        {
                            "specification_id": (
                                f"{hypothesis.hypothesis_id}:{sensitivity_id}"
                            ),
                            "hypothesis_id": hypothesis.hypothesis_id,
                            "dataset_version_id": sensitivity["dataset_version_id"],
                            "source_combination_id": sensitivity_source_id,
                            "curve_outcome": hypothesis.outcome,
                            "outcome_role": hypothesis.evidence_role,
                            "factor_names": sensitivity["factor_names"],
                            "grouping": sensitivity["grouping"],
                            "model_specification": sensitivity[
                                "model_specification"
                            ],
                            "alpha": hypothesis.alpha,
                            "contrast_specification": MappingProxyType(
                                sensitivity_contrast
                            ),
                            "analysis_family": sensitivity["analysis_family"],
                            "engine": sensitivity_engine,
                            "multiplicity_family_id": (
                                f"{hypothesis.multiplicity_family_id}:sensitivity"
                            ),
                            "support_rule_id": sensitivity["support_rule_id"],
                            "support_policy": sensitivity_support_policy,
                            "factor_representations": MappingProxyType(
                                sensitivity_representations
                            ),
                            "first_stage_uncertainty_policy": (
                                first_stage_uncertainty_policy
                            ),
                        }
                    )
                )
        return tuple(specifications)


def _nonempty_text(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicyArtifactError(f"{where} must be a nonempty string")
    return value.strip()


def _string_tuple(value: object, *, where: str, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise PolicyArtifactError(f"{where} must be a JSON array")
    normalized = tuple(_nonempty_text(item, where=f"{where} member") for item in value)
    if not allow_empty and not normalized:
        raise PolicyArtifactError(f"{where} must not be empty")
    if len(normalized) != len(set(normalized)):
        raise PolicyArtifactError(f"{where} must contain unique values")
    return normalized


def _positive_integer(value: object, *, where: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PolicyArtifactError(f"{where} must be an integer >= {minimum}")
    return value


def _fraction(value: object, *, where: str, include_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyArtifactError(f"{where} must be numeric")
    normalized = float(value)
    lower_ok = normalized >= 0.0 if include_zero else normalized > 0.0
    if not lower_ok or normalized > 1.0:
        raise PolicyArtifactError(f"{where} must be in {'[0, 1]' if include_zero else '(0, 1]'}")
    return normalized


def _finite_number(
    value: object,
    *,
    where: str,
    minimum: float | None = None,
    strictly_greater: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise PolicyArtifactError(f"{where} must be a finite number")
    normalized = float(value)
    if minimum is not None:
        allowed = normalized > minimum if strictly_greater else normalized >= minimum
        if not allowed:
            comparison = ">" if strictly_greater else ">="
            raise PolicyArtifactError(f"{where} must be {comparison} {minimum}")
    return normalized


def _exact_object(
    value: object,
    *,
    required_keys: set[str] | frozenset[str],
    where: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyArtifactError(f"{where} must be an object")
    observed = set(value)
    if observed != set(required_keys):
        missing = ", ".join(sorted(set(required_keys) - observed)) or "none"
        extra = ", ".join(sorted(observed - set(required_keys))) or "none"
        raise PolicyArtifactError(
            f"{where} fields do not match the required schema; "
            f"missing={missing}; extra={extra}"
        )
    return value


def _reviewed_model_specification(
    value: object,
    *,
    factor_names: tuple[str, ...],
    required: bool,
    where: str,
) -> Mapping[str, Any]:
    """Validate one frozen inferential model and scientific-test contract."""

    if not required and value is None:
        return MappingProxyType({})
    raw = _exact_object(
        value,
        required_keys={
            "outcome_kind",
            "model_kind",
            "dependence_structure",
            "link_function",
            "focal_factor_names",
            "adjustment_factor_names",
            "multiplicity_test_ids",
            "interval_method",
        },
        where=where,
    )
    outcome_kind = _nonempty_text(
        raw["outcome_kind"],
        where=f"{where}.outcome_kind",
    )
    model_kind = _nonempty_text(raw["model_kind"], where=f"{where}.model_kind")
    dependence_structure = _nonempty_text(
        raw["dependence_structure"],
        where=f"{where}.dependence_structure",
    )
    link_function = _nonempty_text(
        raw["link_function"],
        where=f"{where}.link_function",
    )
    supported_models = {
        ("continuous", "lmer", "identity"),
        ("categorical", "glmmTMB", "logit"),
    }
    if (outcome_kind, model_kind, link_function) not in supported_models:
        raise PolicyArtifactError(
            f"{where} declares an unsupported outcome/model/link combination"
        )
    if dependence_structure not in {
        "random_intercept",
        "nested_random_intercept",
    }:
        raise PolicyArtifactError(
            f"{where}.dependence_structure has no executable runtime path"
        )
    focal = _string_tuple(
        raw["focal_factor_names"],
        where=f"{where}.focal_factor_names",
    )
    adjustments = _string_tuple(
        raw["adjustment_factor_names"],
        where=f"{where}.adjustment_factor_names",
        allow_empty=True,
    )
    if set(focal) & set(adjustments) or set(focal) | set(adjustments) != set(
        factor_names
    ):
        raise PolicyArtifactError(
            f"{where} focal and adjustment factors must partition factor_names"
        )
    test_ids = _string_tuple(
        raw["multiplicity_test_ids"],
        where=f"{where}.multiplicity_test_ids",
    )
    if len(test_ids) != len(set(test_ids)):
        raise PolicyArtifactError(f"{where}.multiplicity_test_ids must be unique")
    interval_method = _nonempty_text(
        raw["interval_method"],
        where=f"{where}.interval_method",
    )
    if interval_method != "wald_95":
        raise PolicyArtifactError(
            f"{where}.interval_method must name the implemented wald_95 method"
        )
    return MappingProxyType(
        {
            "outcome_kind": outcome_kind,
            "model_kind": model_kind,
            "dependence_structure": dependence_structure,
            "link_function": link_function,
            "focal_factor_names": focal,
            "adjustment_factor_names": adjustments,
            "multiplicity_test_ids": test_ids,
            "interval_method": interval_method,
        }
    )


def _review_status(value: object, *, where: str) -> str:
    status = _nonempty_text(value, where=where)
    if status not in {"approved", "withheld"}:
        raise PolicyArtifactError(f"{where} must be approved or withheld")
    return status


def _records(payload: Mapping[str, Any], *, where: str) -> tuple[Mapping[str, Any], ...]:
    raw = payload.get("records")
    if not isinstance(raw, list) or not raw:
        raise PolicyArtifactError(f"{where}.records must be a nonempty JSON array")
    records: list[Mapping[str, Any]] = []
    for index, record in enumerate(raw):
        if not isinstance(record, Mapping):
            raise PolicyArtifactError(f"{where}.records[{index}] must be an object")
        records.append(MappingProxyType(dict(record)))
    return tuple(records)


def _load_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_type: str,
) -> tuple[ArtifactAuthority, Mapping[str, Any]]:
    artifact_path = Path(path).resolve()
    if artifact_path.suffix.casefold() != ".json":
        raise PolicyArtifactError(
            f"{expected_type} artifact must use the JSON format"
        )
    if not artifact_path.is_file():
        raise PolicyArtifactError(f"{expected_type} artifact does not exist: {artifact_path}")
    expected = expected_sha256.lower()
    if not _SHA256_RE.fullmatch(expected):
        raise PolicyArtifactError(f"{expected_type} expected SHA-256 is malformed")
    actual = sha256_file(artifact_path)
    if actual != expected:
        raise PolicyArtifactError(
            f"{expected_type} SHA-256 mismatch: expected {expected}, observed {actual}"
        )
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PolicyArtifactError(f"{expected_type} artifact is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise PolicyArtifactError(f"{expected_type} artifact root must be an object")
    artifact_type = _nonempty_text(payload.get("artifact_type"), where="artifact_type")
    if artifact_type != expected_type:
        raise PolicyArtifactError(
            f"Expected artifact_type {expected_type!r}, observed {artifact_type!r}"
        )
    version = _nonempty_text(payload.get("artifact_version"), where="artifact_version")
    if not _VERSION_RE.fullmatch(version):
        raise PolicyArtifactError("artifact_version contains unsupported characters")
    approved_by = _nonempty_text(payload.get("approved_by"), where="approved_by")
    approval_date = _nonempty_text(payload.get("approval_date"), where="approval_date")
    try:
        date.fromisoformat(approval_date)
    except ValueError as exc:
        raise PolicyArtifactError("approval_date must be an ISO-8601 calendar date") from exc
    scope = _nonempty_text(payload.get("scope"), where="scope")
    archive_location = _nonempty_text(
        payload.get("archive_location"),
        where="archive_location",
    )
    authority = ArtifactAuthority(
        artifact_type=artifact_type,
        artifact_version=version,
        path=artifact_path,
        sha256=actual,
        approved_by=approved_by,
        approval_date=approval_date,
        scope=scope,
        archive_location=archive_location,
    )
    return authority, MappingProxyType(dict(payload))


def load_support_table(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[ArtifactAuthority, tuple[SupportRule, ...]]:
    authority, payload = _load_artifact(
        path,
        expected_sha256=expected_sha256,
        expected_type="outcome_support_table",
    )
    rules: list[SupportRule] = []
    for index, record in enumerate(_records(payload, where="support table")):
        where = f"support table record {index}"
        sensitivity_raw = record.get("sensitivity_checks")
        if not isinstance(sensitivity_raw, list) or not sensitivity_raw:
            raise PolicyArtifactError(f"{where}.sensitivity_checks must be a nonempty array")
        sensitivities: list[Mapping[str, Any]] = []
        sensitivity_fields: set[str] = set()
        integer_threshold_fields = {
            "minimum_independent_series",
            "minimum_independent_studies",
            "minimum_observations_per_cell",
            "minimum_class_events_per_parameter",
            "minimum_residual_df",
            "maximum_factor_cardinality",
            "maximum_loso_studies",
        }
        fraction_threshold_fields = {"maximum_missing_fraction"}
        for sensitivity_index, sensitivity in enumerate(sensitivity_raw):
            if not isinstance(sensitivity, Mapping) or not sensitivity:
                raise PolicyArtifactError(
                    f"{where}.sensitivity_checks[{sensitivity_index}] must be a nonempty object"
                )
            sensitivity_where = (
                f"{where}.sensitivity_checks[{sensitivity_index}]"
            )
            sensitivity_contract = _exact_object(
                sensitivity,
                required_keys={"field", "values"},
                where=sensitivity_where,
            )
            field = _nonempty_text(
                sensitivity_contract["field"],
                where=f"{sensitivity_where}.field",
            )
            if field not in integer_threshold_fields | fraction_threshold_fields:
                raise PolicyArtifactError(
                    f"{sensitivity_where}.field names an unsupported threshold field"
                )
            if field in sensitivity_fields:
                raise PolicyArtifactError(
                    f"{where}.sensitivity_checks must use unique threshold fields"
                )
            sensitivity_fields.add(field)
            raw_values = sensitivity_contract["values"]
            if not isinstance(raw_values, list) or not raw_values:
                raise PolicyArtifactError(
                    f"{sensitivity_where}.values must be a nonempty array"
                )
            if len(raw_values) > 8:
                raise PolicyArtifactError(
                    f"{sensitivity_where}.values must contain at most 8 thresholds"
                )
            values: list[int | float] = []
            for value_index, value in enumerate(raw_values):
                value_where = f"{sensitivity_where}.values[{value_index}]"
                if field in integer_threshold_fields:
                    values.append(_positive_integer(value, where=value_where))
                else:
                    values.append(_fraction(value, where=value_where))
            if len(values) != len(set(values)):
                raise PolicyArtifactError(
                    f"{sensitivity_where}.values must contain unique thresholds"
                )
            sensitivities.append(
                MappingProxyType(
                    {
                        "field": field,
                        "values": tuple(values),
                    }
                )
            )
        factor_type = _nonempty_text(
            record.get("factor_type"),
            where=f"{where}.factor_type",
        )
        if factor_type not in {
            "numeric",
            "categorical",
            "ordinal",
            "boolean",
            "mixed",
            "none",
        }:
            raise PolicyArtifactError(f"{where}.factor_type is unsupported")
        rules.append(
            SupportRule(
                rule_id=_nonempty_text(record.get("rule_id"), where=f"{where}.rule_id"),
                outcome=_nonempty_text(record.get("outcome"), where=f"{where}.outcome"),
                analysis_family=_nonempty_text(
                    record.get("analysis_family"),
                    where=f"{where}.analysis_family",
                ),
                factor_type=factor_type,
                minimum_independent_series=_positive_integer(
                    record.get("minimum_independent_series"),
                    where=f"{where}.minimum_independent_series",
                ),
                minimum_independent_studies=_positive_integer(
                    record.get("minimum_independent_studies"),
                    where=f"{where}.minimum_independent_studies",
                ),
                minimum_observations_per_cell=_positive_integer(
                    record.get("minimum_observations_per_cell"),
                    where=f"{where}.minimum_observations_per_cell",
                ),
                minimum_class_events_per_parameter=_positive_integer(
                    record.get("minimum_class_events_per_parameter"),
                    where=f"{where}.minimum_class_events_per_parameter",
                ),
                minimum_residual_df=_positive_integer(
                    record.get("minimum_residual_df"),
                    where=f"{where}.minimum_residual_df",
                ),
                maximum_missing_fraction=_fraction(
                    record.get("maximum_missing_fraction"),
                    where=f"{where}.maximum_missing_fraction",
                ),
                maximum_factor_cardinality=_positive_integer(
                    record.get("maximum_factor_cardinality"),
                    where=f"{where}.maximum_factor_cardinality",
                ),
                maximum_loso_studies=_positive_integer(
                    record.get("maximum_loso_studies"),
                    where=f"{where}.maximum_loso_studies",
                ),
                sensitivity_checks=tuple(sensitivities),
            )
        )
    identifiers = [rule.rule_id for rule in rules]
    keys = [
        (rule.outcome, rule.analysis_family, rule.factor_type)
        for rule in rules
    ]
    if len(identifiers) != len(set(identifiers)):
        raise PolicyArtifactError("Support rule IDs must be unique")
    if len(keys) != len(set(keys)):
        raise PolicyArtifactError(
            "Support table outcome/analysis-family/factor-type keys must be unique"
        )
    return authority, tuple(rules)


def load_factor_representations(
    path: str | Path,
    *,
    expected_sha256: str,
    required_pairs: Sequence[tuple[str, str]] = (),
) -> tuple[ArtifactAuthority, tuple[FactorRepresentation, ...]]:
    authority, payload = _load_artifact(
        path,
        expected_sha256=expected_sha256,
        expected_type="factor_representation_table",
    )
    representations: list[FactorRepresentation] = []
    for index, record in enumerate(_records(payload, where="factor representations")):
        where = f"factor representation record {index}"
        engine = _nonempty_text(record.get("engine"), where=f"{where}.engine")
        if engine not in _ANALYSIS_ENGINES:
            raise PolicyArtifactError(f"{where}.engine must be python or r")
        data_type = _nonempty_text(record.get("data_type"), where=f"{where}.data_type")
        if data_type not in {"numeric", "categorical", "ordinal", "boolean"}:
            raise PolicyArtifactError(f"{where}.data_type is unsupported")
        category_raw = record.get("category_map", {})
        if not isinstance(category_raw, Mapping):
            raise PolicyArtifactError(f"{where}.category_map must be an object")
        category_map = {
            _nonempty_text(key, where=f"{where}.category_map key"): _nonempty_text(
                value,
                where=f"{where}.category_map value",
            )
            for key, value in category_raw.items()
        }
        reference = record.get("reference_level")
        if reference is not None:
            reference = _nonempty_text(reference, where=f"{where}.reference_level")
        if data_type in {"categorical", "ordinal", "boolean"}:
            if not category_map:
                raise PolicyArtifactError(f"{where} requires a nonempty category_map")
            if reference is None or reference not in set(category_map.values()):
                raise PolicyArtifactError(
                    f"{where}.reference_level must name a mapped category"
                )
        unit = record.get("unit")
        if unit is not None:
            unit = _nonempty_text(unit, where=f"{where}.unit")
        missingness_rule = _nonempty_text(
            record.get("missingness_rule"),
            where=f"{where}.missingness_rule",
        )
        if missingness_rule != "factor_outcome_complete_case_no_imputation":
            raise PolicyArtifactError(
                f"{where}.missingness_rule must enforce factor/outcome complete cases"
            )
        learned_only = record.get("learned_within_training_only")
        if not isinstance(learned_only, bool):
            raise PolicyArtifactError(
                f"{where}.learned_within_training_only must be boolean"
            )
        representations.append(
            FactorRepresentation(
                representation_id=_nonempty_text(
                    record.get("representation_id"),
                    where=f"{where}.representation_id",
                ),
                factor_name=_nonempty_text(
                    record.get("factor_name"),
                    where=f"{where}.factor_name",
                ),
                engine=engine,
                role=_nonempty_text(record.get("role"), where=f"{where}.role"),
                source_fields=_string_tuple(
                    record.get("source_fields"),
                    where=f"{where}.source_fields",
                ),
                data_type=data_type,
                unit=unit,
                transformation=_nonempty_text(
                    record.get("transformation"),
                    where=f"{where}.transformation",
                ),
                reference_level=reference,
                category_map=MappingProxyType(dict(sorted(category_map.items()))),
                missingness_rule=missingness_rule,
                leakage_exclusions=_string_tuple(
                    record.get("leakage_exclusions"),
                    where=f"{where}.leakage_exclusions",
                ),
                learned_within_training_only=learned_only,
            )
        )
    identifiers = [item.representation_id for item in representations]
    pairs = [(item.factor_name, item.engine) for item in representations]
    if len(identifiers) != len(set(identifiers)):
        raise PolicyArtifactError("Factor representation IDs must be unique")
    if len(pairs) != len(set(pairs)):
        raise PolicyArtifactError("Each factor/engine pair must have one representation")
    required = set(required_pairs)
    missing = required - set(pairs)
    if missing:
        formatted = ", ".join(f"{factor}/{engine}" for factor, engine in sorted(missing))
        raise PolicyArtifactError(f"Required factor representations are missing: {formatted}")
    return authority, tuple(representations)


def load_estimands(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[ArtifactAuthority, tuple[Estimand, ...]]:
    authority, payload = _load_artifact(
        path,
        expected_sha256=expected_sha256,
        expected_type="estimand_registry",
    )
    estimands: list[Estimand] = []
    for index, record in enumerate(_records(payload, where="estimands")):
        where = f"estimand record {index}"
        hypothesis_id = _nonempty_text(
            record.get("hypothesis_id"),
            where=f"{where}.hypothesis_id",
        )
        if hypothesis_id not in _HYPOTHESIS_IDS:
            raise PolicyArtifactError(
                f"{where}.hypothesis_id is outside the bounded hypothesis registry"
            )
        same_context = record.get("same_context_required")
        if same_context is not True:
            raise PolicyArtifactError(f"{where} must require verified same-context evidence")
        estimands.append(
            Estimand(
                estimand_id=_nonempty_text(
                    record.get("estimand_id"),
                    where=f"{where}.estimand_id",
                ),
                hypothesis_id=hypothesis_id,
                estimand_type=_nonempty_text(
                    record.get("estimand_type"),
                    where=f"{where}.estimand_type",
                ),
                outcome=_nonempty_text(record.get("outcome"), where=f"{where}.outcome"),
                treatment=_nonempty_text(
                    record.get("treatment"),
                    where=f"{where}.treatment",
                ),
                comparator=_nonempty_text(
                    record.get("comparator"),
                    where=f"{where}.comparator",
                ),
                direction=_nonempty_text(
                    record.get("direction"),
                    where=f"{where}.direction",
                ),
                target_population=_nonempty_text(
                    record.get("target_population"),
                    where=f"{where}.target_population",
                ),
                same_context_required=True,
                dependence_unit=_nonempty_text(
                    record.get("dependence_unit"),
                    where=f"{where}.dependence_unit",
                ),
                unit=_nonempty_text(record.get("unit"), where=f"{where}.unit"),
                eligibility_rule=_nonempty_text(
                    record.get("eligibility_rule"),
                    where=f"{where}.eligibility_rule",
                ),
                unavailable_reason_field=_nonempty_text(
                    record.get("unavailable_reason_field"),
                    where=f"{where}.unavailable_reason_field",
                ),
            )
        )
    identifiers = [item.estimand_id for item in estimands]
    if len(identifiers) != len(set(identifiers)):
        raise PolicyArtifactError("Estimand IDs must be unique")
    return authority, tuple(estimands)


def load_hypothesis_snapshot(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[ArtifactAuthority, tuple[EffectiveHypothesis, ...]]:
    authority, payload = _load_artifact(
        path,
        expected_sha256=expected_sha256,
        expected_type="effective_hypothesis_snapshot",
    )
    hypotheses: list[EffectiveHypothesis] = []
    for index, record in enumerate(_records(payload, where="hypotheses")):
        where = f"hypothesis record {index}"
        hypothesis_id = _nonempty_text(
            record.get("hypothesis_id"),
            where=f"{where}.hypothesis_id",
        )
        if hypothesis_id not in _HYPOTHESIS_IDS:
            raise PolicyArtifactError(
                f"{where}.hypothesis_id is outside the bounded hypothesis registry"
            )
        status = _nonempty_text(record.get("status"), where=f"{where}.status")
        if status not in {"enabled", "disabled"}:
            raise PolicyArtifactError(f"{where}.status must be enabled or disabled")
        disabled_reason = record.get("disabled_reason")
        if status == "disabled":
            disabled_reason = _nonempty_text(
                disabled_reason,
                where=f"{where}.disabled_reason",
            )
        elif disabled_reason not in {None, ""}:
            raise PolicyArtifactError(
                f"{where}.disabled_reason must be empty for an enabled hypothesis"
            )
        evidence_role = _nonempty_text(
            record.get("evidence_role"),
            where=f"{where}.evidence_role",
        )
        if evidence_role not in {"primary", "secondary", "sensitivity", "exploratory"}:
            raise PolicyArtifactError(
                f"{where}.evidence_role must be primary, secondary, sensitivity, or exploratory"
            )
        analysis_family = _nonempty_text(
            record.get("analysis_family"),
            where=f"{where}.analysis_family",
        )
        engine = _nonempty_text(record.get("engine"), where=f"{where}.engine")
        if engine not in _ANALYSIS_ENGINES:
            raise PolicyArtifactError(f"{where}.engine must be python or r")
        if (
            status == "enabled"
            and analysis_family in INFERENTIAL_ANALYSIS_FAMILIES
            and engine != "r"
        ):
            raise PolicyArtifactError(
                f"{where} enabled inferential hypotheses require the R engine until "
                "a shared Python raw-test multiplicity contract is implemented"
            )
        alpha = _fraction(
            record.get("alpha"),
            where=f"{where}.alpha",
            include_zero=False,
        )
        factor_names = _string_tuple(
            record.get("factor_names"),
            where=f"{where}.factor_names",
            allow_empty=status == "disabled",
        )
        unapproved_factors = sorted(
            set(factor_names) - _ANA_APPROVED_FACTOR_ROSTER
        )
        if status == "enabled" and unapproved_factors:
            raise PolicyArtifactError(
                f"{where}.factor_names contains factors outside the ANA-04 approved "
                "roster: " + ", ".join(unapproved_factors)
            )
        grouping = _string_tuple(
            record.get("grouping"),
            where=f"{where}.grouping",
        )
        model_specification = _reviewed_model_specification(
            record.get("model_specification"),
            factor_names=factor_names,
            required=status == "enabled",
            where=f"{where}.model_specification",
        )
        sensitivities = _string_tuple(
            record.get("sensitivities"),
            where=f"{where}.sensitivities",
        )
        raw_sensitivity_specifications = record.get("sensitivity_specifications")
        if not isinstance(raw_sensitivity_specifications, list):
            raise PolicyArtifactError(
                f"{where}.sensitivity_specifications must be a JSON array"
            )
        sensitivity_specifications: list[Mapping[str, Any]] = []
        for sensitivity_index, raw_specification in enumerate(
            raw_sensitivity_specifications,
            start=1,
        ):
            sensitivity_where = (
                f"{where}.sensitivity_specifications[{sensitivity_index}]"
            )
            specification = _exact_object(
                raw_specification,
                required_keys={
                    "sensitivity_id",
                    "dataset_version_id",
                    "source_view",
                    "analysis_family",
                    "factor_names",
                    "grouping",
                    "model_specification",
                    "support_rule_id",
                    "engine",
                },
                where=sensitivity_where,
            )
            sensitivity_engine = _nonempty_text(
                specification["engine"],
                where=f"{sensitivity_where}.engine",
            )
            if sensitivity_engine not in _ANALYSIS_ENGINES:
                raise PolicyArtifactError(
                    f"{sensitivity_where}.engine must be python or r"
                )
            sensitivity_analysis_family = _nonempty_text(
                specification["analysis_family"],
                where=f"{sensitivity_where}.analysis_family",
            )
            if (
                status == "enabled"
                and sensitivity_analysis_family in INFERENTIAL_ANALYSIS_FAMILIES
                and sensitivity_engine != "r"
            ):
                raise PolicyArtifactError(
                    f"{sensitivity_where} enabled inferential sensitivities require "
                    "the R engine until a shared Python raw-test multiplicity contract "
                    "is implemented"
                )
            sensitivity_factor_names = _string_tuple(
                specification["factor_names"],
                where=f"{sensitivity_where}.factor_names",
                allow_empty=status == "disabled",
            )
            unapproved_sensitivity_factors = sorted(
                set(sensitivity_factor_names) - _ANA_APPROVED_FACTOR_ROSTER
            )
            if status == "enabled" and unapproved_sensitivity_factors:
                raise PolicyArtifactError(
                    f"{sensitivity_where}.factor_names contains factors outside the "
                    "ANA-04 approved roster: "
                    + ", ".join(unapproved_sensitivity_factors)
                )
            sensitivity_grouping = _string_tuple(
                specification["grouping"],
                where=f"{sensitivity_where}.grouping",
            )
            sensitivity_model_specification = _reviewed_model_specification(
                specification["model_specification"],
                factor_names=sensitivity_factor_names,
                required=status == "enabled",
                where=f"{sensitivity_where}.model_specification",
            )
            sensitivity_source_view = _nonempty_text(
                specification["source_view"],
                where=f"{sensitivity_where}.source_view",
            )
            if (
                status == "enabled"
                and sensitivity_source_view != "all_families_deduplicated"
            ):
                raise PolicyArtifactError(
                    f"{sensitivity_where}.source_view must use the ANA-02 "
                    "all_families_deduplicated source pool"
                )
            sensitivity_specifications.append(
                MappingProxyType(
                    {
                        "sensitivity_id": _nonempty_text(
                            specification["sensitivity_id"],
                            where=f"{sensitivity_where}.sensitivity_id",
                        ),
                        "dataset_version_id": _nonempty_text(
                            specification["dataset_version_id"],
                            where=f"{sensitivity_where}.dataset_version_id",
                        ),
                        "source_view": sensitivity_source_view,
                        "analysis_family": sensitivity_analysis_family,
                        "factor_names": sensitivity_factor_names,
                        "grouping": sensitivity_grouping,
                        "model_specification": sensitivity_model_specification,
                        "support_rule_id": _nonempty_text(
                            specification["support_rule_id"],
                            where=f"{sensitivity_where}.support_rule_id",
                        ),
                        "engine": sensitivity_engine,
                    }
                )
            )
        if {
            str(specification["sensitivity_id"])
            for specification in sensitivity_specifications
        } != set(sensitivities):
            raise PolicyArtifactError(
                f"{where}.sensitivity_specifications must exactly cover sensitivities"
            )
        if status == "enabled" and any(
            str(specification["dataset_version_id"]) == "D12_no_numeric_n_inventory"
            for specification in sensitivity_specifications
        ):
            raise PolicyArtifactError(
                f"{where} ANA-01 D12_no_numeric_n_inventory is inventory-only and "
                "cannot be an enabled analysis sensitivity"
            )
        if status == "enabled" and evidence_role == "primary":
            has_required_d01_sensitivity = any(
                str(specification["sensitivity_id"])
                == "D01_strict_primary_zero_n"
                and str(specification["dataset_version_id"])
                == "D01_strict_primary_zero_n"
                for specification in sensitivity_specifications
            )
            if not has_required_d01_sensitivity:
                raise PolicyArtifactError(
                    f"{where} ANA-01 primary evidence requires the "
                    "D01_strict_primary_zero_n dataset sensitivity"
                )
        try:
            claim_policy = validate_claim_policy(
                record.get("claim_policy"),
                declared_sensitivities=sensitivities,
                alpha=alpha,
            )
        except ValueError as exc:
            raise PolicyArtifactError(f"{where}.claim_policy is invalid: {exc}") from exc
        dataset_version_id = _nonempty_text(
            record.get("dataset_version_id"),
            where=f"{where}.dataset_version_id",
        )
        source_view = _nonempty_text(
            record.get("source_view"),
            where=f"{where}.source_view",
        )
        if (
            status == "enabled"
            and evidence_role == "primary"
            and dataset_version_id != "D02_strict_primary_zero_optional"
        ):
            raise PolicyArtifactError(
                f"{where} primary evidence must use D02_strict_primary_zero_optional"
            )
        if status == "enabled" and source_view != "all_families_deduplicated":
            raise PolicyArtifactError(
                f"{where}.source_view must use the ANA-02 all_families_deduplicated "
                "primary source pool; source-combination inference is deferred"
            )
        outcome = _nonempty_text(record.get("outcome"), where=f"{where}.outcome")
        if status == "enabled" and outcome == "economic_optimum_n_kg_ha":
            raise PolicyArtifactError(f"{where} ANA-03 keeps {outcome} disabled")
        if (
            status == "enabled"
            and outcome in _ANA03_PRIMARY_OUTCOMES
            and evidence_role != "primary"
        ):
            raise PolicyArtifactError(
                f"{where} ANA-03 requires {outcome} to use the primary evidence role"
            )
        if (
            status == "enabled"
            and outcome in _ANA03_SECONDARY_OUTCOMES
            and evidence_role != "secondary"
        ):
            raise PolicyArtifactError(
                f"{where} ANA-03 requires {outcome} to use the secondary evidence role"
            )
        hypotheses.append(
            EffectiveHypothesis(
                hypothesis_id=hypothesis_id,
                status=status,
                disabled_reason=disabled_reason or None,
                evidence_role=evidence_role,
                dataset_version_id=dataset_version_id,
                source_view=source_view,
                population=_nonempty_text(
                    record.get("population"),
                    where=f"{where}.population",
                ),
                outcome=outcome,
                estimand_id=_nonempty_text(
                    record.get("estimand_id"),
                    where=f"{where}.estimand_id",
                ),
                analysis_family=analysis_family,
                factor_names=factor_names,
                grouping=grouping,
                model_specification=model_specification,
                support_rule_id=_nonempty_text(
                    record.get("support_rule_id"),
                    where=f"{where}.support_rule_id",
                ),
                engine=engine,
                multiplicity_family_id=_nonempty_text(
                    record.get("multiplicity_family_id"),
                    where=f"{where}.multiplicity_family_id",
                ),
                alpha=alpha,
                sensitivities=sensitivities,
                sensitivity_specifications=tuple(sensitivity_specifications),
                claim_policy=MappingProxyType(claim_policy),
            )
        )
    identifiers = {item.hypothesis_id for item in hypotheses}
    if len(hypotheses) != len(identifiers):
        raise PolicyArtifactError("Hypothesis IDs must be unique")
    if identifiers != _HYPOTHESIS_IDS:
        missing = ", ".join(sorted(_HYPOTHESIS_IDS - identifiers)) or "none"
        extra = ", ".join(sorted(identifiers - _HYPOTHESIS_IDS)) or "none"
        raise PolicyArtifactError(
            "Effective snapshot must account for the complete bounded hypothesis registry; "
            f"missing={missing}; extra={extra}"
        )
    enabled = tuple(item for item in hypotheses if item.status == "enabled")
    if enabled and not any(item.evidence_role == "primary" for item in enabled):
        raise PolicyArtifactError(
            "Effective snapshot must designate at least one enabled primary hypothesis"
        )
    return authority, tuple(sorted(hypotheses, key=lambda item: item.hypothesis_id))


def _curve_model_gates(value: object) -> Mapping[str, CurveModelGate]:
    raw_models = _exact_object(
        value,
        required_keys=set(_MODEL_ROSTER),
        where="curve model policy model_gates",
    )
    gates: dict[str, CurveModelGate] = {}
    required_gate_fields = {
        "initialization_strategy",
        "parameter_bounds",
        "allow_boundary_parameters",
        "reportable_shape_classes",
        "optimizer_tolerance",
        "optimizer_max_iterations",
        "parameter_boundary_relative_tolerance",
        "optimum_boundary_tolerance_n_kg_ha",
        "flat_response_tolerance_t_ha",
    }
    for model_name in _MODEL_ROSTER:
        where = f"curve model policy model_gates.{model_name}"
        raw_gate = _exact_object(
            raw_models[model_name],
            required_keys=required_gate_fields,
            where=where,
        )
        initialization = _nonempty_text(
            raw_gate["initialization_strategy"],
            where=f"{where}.initialization_strategy",
        )
        if initialization != _MODEL_INITIALIZATION_STRATEGIES[model_name]:
            raise PolicyArtifactError(
                f"{where}.initialization_strategy is incompatible with the "
                "deterministic candidate implementation"
            )
        raw_bounds = _exact_object(
            raw_gate["parameter_bounds"],
            required_keys=set(_MODEL_PARAMETER_NAMES[model_name]),
            where=f"{where}.parameter_bounds",
        )
        bounds: dict[str, tuple[float, float]] = {}
        for parameter_name in _MODEL_PARAMETER_NAMES[model_name]:
            interval = raw_bounds[parameter_name]
            if not isinstance(interval, list) or len(interval) != 2:
                raise PolicyArtifactError(
                    f"{where}.parameter_bounds.{parameter_name} must be a "
                    "two-member JSON array"
                )
            lower = _finite_number(
                interval[0],
                where=f"{where}.parameter_bounds.{parameter_name}[0]",
            )
            upper = _finite_number(
                interval[1],
                where=f"{where}.parameter_bounds.{parameter_name}[1]",
            )
            if upper <= lower:
                raise PolicyArtifactError(
                    f"{where}.parameter_bounds.{parameter_name} upper bound "
                    "must exceed its lower bound"
                )
            bounds[parameter_name] = (lower, upper)
        allow_boundary = raw_gate["allow_boundary_parameters"]
        if not isinstance(allow_boundary, bool):
            raise PolicyArtifactError(
                f"{where}.allow_boundary_parameters must be boolean"
            )
        shape_classes = _string_tuple(
            raw_gate["reportable_shape_classes"],
            where=f"{where}.reportable_shape_classes",
        )
        unsupported_shapes = set(shape_classes) - _MODEL_SHAPE_CLASSES[model_name]
        if unsupported_shapes:
            raise PolicyArtifactError(
                f"{where}.reportable_shape_classes contains unsupported "
                "class(es): "
                + ", ".join(sorted(unsupported_shapes))
            )
        optimizer_tolerance = _finite_number(
            raw_gate["optimizer_tolerance"],
            where=f"{where}.optimizer_tolerance",
            minimum=0.0,
        )
        optimizer_max_iterations = _positive_integer(
            raw_gate["optimizer_max_iterations"],
            where=f"{where}.optimizer_max_iterations",
        )
        parameter_boundary_relative_tolerance = _finite_number(
            raw_gate["parameter_boundary_relative_tolerance"],
            where=f"{where}.parameter_boundary_relative_tolerance",
            minimum=0.0,
        )
        optimum_boundary_tolerance_n_kg_ha = _finite_number(
            raw_gate["optimum_boundary_tolerance_n_kg_ha"],
            where=f"{where}.optimum_boundary_tolerance_n_kg_ha",
            minimum=0.0,
        )
        flat_response_tolerance_t_ha = _finite_number(
            raw_gate["flat_response_tolerance_t_ha"],
            where=f"{where}.flat_response_tolerance_t_ha",
            minimum=0.0,
        )
        if (
            optimizer_tolerance <= 0.0
            or parameter_boundary_relative_tolerance <= 0.0
            or optimum_boundary_tolerance_n_kg_ha <= 0.0
            or flat_response_tolerance_t_ha <= 0.0
        ):
            raise PolicyArtifactError(f"{where} reviewed tolerances must be positive")
        gates[model_name] = CurveModelGate(
            initialization_strategy=initialization,
            parameter_bounds=MappingProxyType(bounds),
            allow_boundary_parameters=allow_boundary,
            reportable_shape_classes=shape_classes,
            optimizer_tolerance=optimizer_tolerance,
            optimizer_max_iterations=optimizer_max_iterations,
            parameter_boundary_relative_tolerance=parameter_boundary_relative_tolerance,
            optimum_boundary_tolerance_n_kg_ha=optimum_boundary_tolerance_n_kg_ha,
            flat_response_tolerance_t_ha=flat_response_tolerance_t_ha,
        )
    return MappingProxyType(gates)


def _material_disagreement_policy(
    value: object,
) -> tuple[str, str, Mapping[str, float]]:
    raw = _exact_object(
        value,
        required_keys={"policy_id", "review_status", "tolerances"},
        where="curve model policy material_disagreement",
    )
    policy_id = _nonempty_text(
        raw["policy_id"],
        where="curve model policy material_disagreement.policy_id",
    )
    status = _review_status(
        raw["review_status"],
        where="curve model policy material_disagreement.review_status",
    )
    raw_tolerances = raw["tolerances"]
    if not isinstance(raw_tolerances, Mapping):
        raise PolicyArtifactError(
            "curve model policy material_disagreement.tolerances must be an object"
        )
    if status == "withheld":
        if raw_tolerances:
            raise PolicyArtifactError(
                "Withheld material-disagreement controls cannot carry active tolerances"
            )
        return policy_id, status, MappingProxyType({})
    if set(raw_tolerances) != _DISAGREEMENT_FIELDS:
        raise PolicyArtifactError(
            "Approved material-disagreement tolerances must cover exactly the "
            "supported conclusion fields"
        )
    tolerances = {
        field: _finite_number(
            raw_tolerances[field],
            where=(
                "curve model policy material_disagreement.tolerances."
                f"{field}"
            ),
            minimum=0.0,
        )
        for field in sorted(_DISAGREEMENT_FIELDS)
    }
    return policy_id, status, MappingProxyType(tolerances)


def _uncertainty_policy(
    value: object,
) -> tuple[str, str, str | None, Mapping[str, Any], tuple[str, ...]]:
    raw = _exact_object(
        value,
        required_keys={
            "policy_id",
            "review_status",
            "method",
            "method_contract",
            "evidence_basis",
        },
        where="curve model policy uncertainty",
    )
    policy_id = _nonempty_text(
        raw["policy_id"],
        where="curve model policy uncertainty.policy_id",
    )
    status = _review_status(
        raw["review_status"],
        where="curve model policy uncertainty.review_status",
    )
    evidence_basis = _string_tuple(
        raw["evidence_basis"],
        where="curve model policy uncertainty.evidence_basis",
    )
    unsupported = set(evidence_basis) - _UNCERTAINTY_EVIDENCE_BASES
    if unsupported:
        raise PolicyArtifactError(
            "curve model policy uncertainty.evidence_basis contains "
            "unsupported member(s): "
            + ", ".join(sorted(unsupported))
        )
    raw_method = raw["method"]
    if status == "approved":
        method = _nonempty_text(
            raw_method,
            where="curve model policy uncertainty.method",
        )
        expected_contract = UNCERTAINTY_METHOD_SPECS.get(method)
        required_evidence = UNCERTAINTY_METHOD_REQUIRED_EVIDENCE.get(method)
        if (
            method not in UNCERTAINTY_METHOD_CONFIDENCE_LEVELS
            or expected_contract is None
            or required_evidence is None
            or not set(required_evidence).issubset(evidence_basis)
        ):
            raise PolicyArtifactError(
                "Approved curve model policy uncertainty must name an executable method "
                "with its required evidence basis"
            )
        raw_contract = _exact_object(
            raw["method_contract"],
            required_keys=set(expected_contract),
            where="curve model policy uncertainty.method_contract",
        )
        normalized_contract: dict[str, Any] = {}
        for field, expected in expected_contract.items():
            where = f"curve model policy uncertainty.method_contract.{field}"
            if isinstance(expected, float):
                observed = _finite_number(raw_contract[field], where=where)
            else:
                observed = _nonempty_text(raw_contract[field], where=where)
            if observed != expected:
                raise PolicyArtifactError(
                    "curve model policy uncertainty.method_contract must exactly match "
                    "the executable method specification"
                )
            normalized_contract[field] = observed
        method_contract: Mapping[str, Any] = MappingProxyType(normalized_contract)
    else:
        if raw_method is not None:
            raise PolicyArtifactError(
                "Withheld uncertainty controls cannot name an active method"
            )
        method = None
        _exact_object(
            raw["method_contract"],
            required_keys=set(),
            where="curve model policy uncertainty.method_contract",
        )
        method_contract = MappingProxyType({})
    return policy_id, status, method, method_contract, evidence_basis


def _economic_policy(
    value: object,
) -> tuple[str, str, str | None, str | None, tuple[Mapping[str, Any], ...]]:
    raw = _exact_object(
        value,
        required_keys={
            "policy_id",
            "review_status",
            "table_id",
            "version",
            "scenarios",
        },
        where="curve model policy economic_scenarios",
    )
    policy_id = _nonempty_text(
        raw["policy_id"],
        where="curve model policy economic_scenarios.policy_id",
    )
    status = _review_status(
        raw["review_status"],
        where="curve model policy economic_scenarios.review_status",
    )
    raw_scenarios = raw["scenarios"]
    if not isinstance(raw_scenarios, list):
        raise PolicyArtifactError(
            "curve model policy economic_scenarios.scenarios must be a JSON array"
        )
    scenarios: list[Mapping[str, Any]] = []
    scenario_ids: set[str] = set()
    for index, raw_scenario in enumerate(raw_scenarios):
        where = f"curve model policy economic_scenarios.scenarios[{index}]"
        scenario = _exact_object(
            raw_scenario,
            required_keys=_ECONOMIC_SCENARIO_FIELDS,
            where=where,
        )
        scenario_id = _nonempty_text(
            scenario["scenario_id"],
            where=f"{where}.scenario_id",
        )
        if scenario_id in scenario_ids:
            raise PolicyArtifactError(
                "Economic scenario identifiers must be unique"
            )
        scenario_ids.add(scenario_id)
        normalized: dict[str, Any] = {
            "scenario_id": scenario_id,
            "grain_price": _finite_number(
                scenario["grain_price"],
                where=f"{where}.grain_price",
                minimum=0.0,
                strictly_greater=True,
            ),
            "n_cost": _finite_number(
                scenario["n_cost"],
                where=f"{where}.n_cost",
                minimum=0.0,
            ),
        }
        for field in sorted(
            _ECONOMIC_SCENARIO_FIELDS - {"scenario_id", "grain_price", "n_cost"}
        ):
            normalized[field] = _nonempty_text(
                scenario[field],
                where=f"{where}.{field}",
            )
        if (
            normalized["decision_rule"] != ECONOMIC_DECISION_RULE
            or normalized["grain_price_unit"] not in ECONOMIC_GRAIN_PRICE_TO_PER_TONNE
            or normalized["n_cost_unit"] != ECONOMIC_N_COST_UNIT
        ):
            raise PolicyArtifactError(
                f"{where} must use the executable decision rule and supported units"
            )
        if normalized["tax_subsidy_application_cost_basis"] not in {
            "excluded",
            "included_in_n_cost",
        }:
            raise PolicyArtifactError(
                f"{where} must use a supported cost-accounting basis"
            )
        scenarios.append(MappingProxyType(normalized))
    raw_table_id = raw["table_id"]
    raw_version = raw["version"]
    if status == "approved":
        table_id = _nonempty_text(
            raw_table_id,
            where="curve model policy economic_scenarios.table_id",
        )
        version = _nonempty_text(
            raw_version,
            where="curve model policy economic_scenarios.version",
        )
        if _VERSION_RE.fullmatch(version) is None:
            raise PolicyArtifactError(
                "curve model policy economic_scenarios.version contains "
                "unsupported characters"
            )
        if not scenarios:
            raise PolicyArtifactError(
                "Approved economic-scenario controls require at least one scenario"
            )
    else:
        if raw_table_id is not None or raw_version is not None or scenarios:
            raise PolicyArtifactError(
                "Withheld economic-scenario controls cannot carry an active table"
            )
        table_id = None
        version = None
    return (
        policy_id,
        status,
        table_id,
        version,
        tuple(sorted(scenarios, key=lambda item: str(item["scenario_id"]))),
    )


def _first_stage_contextual_uncertainty_policy(
    value: object | None,
    *,
    active_within_model_method: str | None,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType(
            {
                "policy_id": "ANA-16-withheld",
                "review_status": "withheld",
                "method_id": None,
                "eligible_outcomes": (),
                "within_model_variance_method": None,
                "model_selection_uncertainty_method": None,
            }
        )
    raw = _exact_object(
        value,
        required_keys={
            "policy_id",
            "review_status",
            "method_id",
            "eligible_outcomes",
            "within_model_variance_method",
            "model_selection_uncertainty_method",
        },
        where="curve model policy first_stage_contextual_uncertainty",
    )
    policy_id = _nonempty_text(
        raw["policy_id"],
        where="curve model policy first_stage_contextual_uncertainty.policy_id",
    )
    status = _review_status(
        raw["review_status"],
        where="curve model policy first_stage_contextual_uncertainty.review_status",
    )
    if status != "approved":
        if any(
            raw[field] not in (None, [])
            for field in (
                "method_id",
                "eligible_outcomes",
                "within_model_variance_method",
                "model_selection_uncertainty_method",
            )
        ):
            raise PolicyArtifactError(
                "Withheld first-stage contextual uncertainty cannot enable a method or outcome"
            )
        return MappingProxyType(
            {
                "policy_id": policy_id,
                "review_status": status,
                "method_id": None,
                "eligible_outcomes": (),
                "within_model_variance_method": None,
                "model_selection_uncertainty_method": None,
            }
        )
    method_id = _nonempty_text(
        raw["method_id"],
        where="curve model policy first_stage_contextual_uncertainty.method_id",
    )
    model_selection_method = _nonempty_text(
        raw["model_selection_uncertainty_method"],
        where=(
            "curve model policy first_stage_contextual_uncertainty."
            "model_selection_uncertainty_method"
        ),
    )
    if (
        method_id != "all_credible_equal_weight_total_variance"
        or model_selection_method != method_id
    ):
        raise PolicyArtifactError(
            "Approved first-stage contextual uncertainty must use the executable "
            "all-credible total-variance method"
        )
    within_method = _nonempty_text(
        raw["within_model_variance_method"],
        where=(
            "curve model policy first_stage_contextual_uncertainty."
            "within_model_variance_method"
        ),
    )
    if within_method != active_within_model_method:
        raise PolicyArtifactError(
            "First-stage contextual uncertainty must reference the active reviewed "
            "within-model uncertainty method"
        )
    outcomes = _string_tuple(
        raw["eligible_outcomes"],
        where="curve model policy first_stage_contextual_uncertainty.eligible_outcomes",
    )
    if set(outcomes) - _FIRST_STAGE_NUMERIC_OUTCOMES:
        raise PolicyArtifactError(
            "First-stage contextual uncertainty contains a nonnumeric or unsupported outcome"
        )
    return MappingProxyType(
        {
            "policy_id": policy_id,
            "review_status": status,
            "method_id": method_id,
            "eligible_outcomes": outcomes,
            "within_model_variance_method": within_method,
            "model_selection_uncertainty_method": model_selection_method,
        }
    )


def _asymptote_support_policy(value: object) -> Mapping[str, Any]:
    fields = (
        "minimum_in_domain_attainment_fraction",
        "maximum_asymptote_relative_se",
        "maximum_influence_relative_shift",
        "minimum_influence_fold_count",
        "maximum_credible_model_relative_difference",
        "maximum_associated_n_basis",
    )
    if value is None:
        return MappingProxyType(
            {
                "policy_id": "MOD-08-withheld",
                "review_status": "withheld",
                **{field: None for field in fields},
            }
        )
    raw = _exact_object(
        value,
        required_keys={"policy_id", "review_status", *fields},
        where="curve model policy asymptote_support",
    )
    policy_id = _nonempty_text(
        raw["policy_id"],
        where="curve model policy asymptote_support.policy_id",
    )
    review_status = _review_status(
        raw["review_status"],
        where="curve model policy asymptote_support.review_status",
    )
    if review_status != "approved":
        if any(raw[field] is not None for field in fields):
            raise PolicyArtifactError(
                "Withheld asymptote-support policy cannot supply scientific thresholds"
            )
        return MappingProxyType({"policy_id": policy_id, **dict(raw)})
    attainment = _fraction(
        raw["minimum_in_domain_attainment_fraction"],
        where=(
            "curve model policy asymptote_support."
            "minimum_in_domain_attainment_fraction"
        ),
        include_zero=False,
    )
    relative_se = _finite_number(
        raw["maximum_asymptote_relative_se"],
        where="curve model policy asymptote_support.maximum_asymptote_relative_se",
        minimum=0.0,
    )
    influence = _finite_number(
        raw["maximum_influence_relative_shift"],
        where=(
            "curve model policy asymptote_support."
            "maximum_influence_relative_shift"
        ),
        minimum=0.0,
    )
    concordance = _finite_number(
        raw["maximum_credible_model_relative_difference"],
        where=(
            "curve model policy asymptote_support."
            "maximum_credible_model_relative_difference"
        ),
        minimum=0.0,
    )
    fold_count = raw["minimum_influence_fold_count"]
    if not isinstance(fold_count, int) or isinstance(fold_count, bool) or fold_count < 1:
        raise PolicyArtifactError(
            "Asymptote-support minimum influence fold count must be a positive integer"
        )
    n_basis = _nonempty_text(
        raw["maximum_associated_n_basis"],
        where="curve model policy asymptote_support.maximum_associated_n_basis",
    )
    if n_basis != "smallest_prediction_grid_rate_meeting_attainment_threshold":
        raise PolicyArtifactError(
            "MOD-08 maximum-associated-N basis is not executable"
        )
    return MappingProxyType(
        {
            "policy_id": policy_id,
            "review_status": review_status,
            "minimum_in_domain_attainment_fraction": attainment,
            "maximum_asymptote_relative_se": relative_se,
            "maximum_influence_relative_shift": influence,
            "minimum_influence_fold_count": fold_count,
            "maximum_credible_model_relative_difference": concordance,
            "maximum_associated_n_basis": n_basis,
        }
    )


_ASYMPTOTE_REFERENCE_QUANTITIES = frozenset({"ceiling_level", "response_range"})


def _asymptote_reporting_policy(value: object) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType(
            {
                "policy_id": "MOD-07-withheld",
                "review_status": "withheld",
                "asymptote_fraction": None,
                "reference_quantity": None,
                "rate_label": None,
                "uncertainty_method": None,
                "scope": None,
            }
        )
    raw = _exact_object(
        value,
        required_keys={
            "policy_id",
            "review_status",
            "asymptote_fraction",
            "reference_quantity",
            "rate_label",
            "uncertainty_method",
            "scope",
        },
        where="curve model policy asymptote_reporting",
    )
    policy_id = _nonempty_text(
        raw["policy_id"],
        where="curve model policy asymptote_reporting.policy_id",
    )
    review_status = _review_status(
        raw["review_status"],
        where="curve model policy asymptote_reporting.review_status",
    )
    if review_status != "approved":
        if any(
            raw[field] is not None
            for field in (
                "asymptote_fraction",
                "reference_quantity",
                "rate_label",
                "uncertainty_method",
                "scope",
            )
        ):
            raise PolicyArtifactError(
                "Withheld asymptote-reporting policy cannot supply scientific controls"
            )
        return MappingProxyType({"policy_id": policy_id, **dict(raw)})
    fraction = _fraction(
        raw["asymptote_fraction"],
        where="curve model policy asymptote_reporting.asymptote_fraction",
        include_zero=False,
    )
    if fraction >= 1.0:
        raise PolicyArtifactError(
            "Asymptote-reporting fraction must be strictly below one"
        )
    # MOD-07 binds a fraction `q`, but a fraction is meaningless until the
    # quantity it is a fraction of is named. The two candidates give different
    # rates and one of them degenerates: a fraction of the ceiling *level* is
    # already satisfied at N = 0 whenever the zero-N yield exceeds q * ceiling,
    # collapsing the reported rate onto the domain floor, while a fraction of
    # the *response range* is scale-free. The plan forbids resolving this in
    # code, so the approver must state it and an unapproved value fails closed.
    reference_quantity = _nonempty_text(
        raw["reference_quantity"],
        where="curve model policy asymptote_reporting.reference_quantity",
    )
    if reference_quantity not in _ASYMPTOTE_REFERENCE_QUANTITIES:
        raise PolicyArtifactError(
            "MOD-07 reporting policy reference_quantity must be one of "
            + ", ".join(sorted(_ASYMPTOTE_REFERENCE_QUANTITIES))
        )
    rate_label = _nonempty_text(
        raw["rate_label"],
        where="curve model policy asymptote_reporting.rate_label",
    )
    uncertainty_method = _nonempty_text(
        raw["uncertainty_method"],
        where="curve model policy asymptote_reporting.uncertainty_method",
    )
    scope = _nonempty_text(
        raw["scope"],
        where="curve model policy asymptote_reporting.scope",
    )
    if rate_label != "N at q% of asymptote":
        raise PolicyArtifactError(
            "MOD-07 reporting policy must retain the approved asymptote-rate label"
        )
    if scope != "supported_mitscherlich_asymptote":
        raise PolicyArtifactError(
            "MOD-07 reporting policy scope must require a supported Mitscherlich asymptote"
        )
    return MappingProxyType(
        {
            "policy_id": policy_id,
            "review_status": review_status,
            "asymptote_fraction": fraction,
            "reference_quantity": reference_quantity,
            "rate_label": rate_label,
            "uncertainty_method": uncertainty_method,
            "scope": scope,
        }
    )


def _efficiency_metric_policy(value: object) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType(
            {
                "policy_id": "EFF-01-withheld",
                "review_status": "withheld",
                "metric_id": None,
                "basis": None,
                "aggregation_level": None,
            }
        )
    raw = _exact_object(
        value,
        required_keys={
            "policy_id",
            "review_status",
            "metric_id",
            "basis",
            "aggregation_level",
        },
        where="curve model policy efficiency_metric",
    )
    policy_id = _nonempty_text(
        raw["policy_id"],
        where="curve model policy efficiency_metric.policy_id",
    )
    review_status = _review_status(
        raw["review_status"],
        where="curve model policy efficiency_metric.review_status",
    )
    if review_status != "approved":
        if any(
            raw[field] is not None
            for field in ("metric_id", "basis", "aggregation_level")
        ):
            raise PolicyArtifactError(
                "Withheld EFF-01 efficiency metric policy cannot supply controls"
            )
        return MappingProxyType(
            {
                "policy_id": policy_id,
                "review_status": review_status,
                "metric_id": None,
                "basis": None,
                "aggregation_level": None,
            }
        )
    metric_id = _nonempty_text(
        raw["metric_id"],
        where="curve model policy efficiency_metric.metric_id",
    )
    basis = _nonempty_text(
        raw["basis"],
        where="curve model policy efficiency_metric.basis",
    )
    aggregation_level = _nonempty_text(
        raw["aggregation_level"],
        where="curve model policy efficiency_metric.aggregation_level",
    )
    if (
        metric_id != "EFF-01-option-a-partial-factor-productivity"
        or basis != "observed_reviewed_treatment_mean"
        or aggregation_level != "record_by_n_level"
    ):
        raise PolicyArtifactError(
            "EFF-01 efficiency metric policy must retain approved Option A semantics"
        )
    return MappingProxyType(
        {
            "policy_id": policy_id,
            "review_status": review_status,
            "metric_id": metric_id,
            "basis": basis,
            "aggregation_level": aggregation_level,
        }
    )


def _efficiency_operating_point_policy(value: object) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType(
            {
                "policy_id": "EFF-03-withheld",
                "review_status": "withheld",
                "yield_retention_fraction": None,
                "maximum_marginal_gain_t_ha_per_kg_n": None,
                "marginal_gain_method": None,
                "model_concordance_tolerance_n_kg_ha": None,
                "zero_n_disposition": None,
                "prediction_grid_points": None,
                "prediction_grid_domain": None,
                "prediction_grid_spacing": None,
                "uncertainty_decision_rule": None,
                "uncertainty_disposition": None,
            }
        )
    raw = _exact_object(
        value,
        required_keys={
            "policy_id",
            "review_status",
            "yield_retention_fraction",
            "maximum_marginal_gain_t_ha_per_kg_n",
            "marginal_gain_method",
            "model_concordance_tolerance_n_kg_ha",
            "zero_n_disposition",
            "prediction_grid_points",
            "prediction_grid_domain",
            "prediction_grid_spacing",
            "uncertainty_decision_rule",
            "uncertainty_disposition",
        },
        where="curve model policy efficiency_operating_point",
    )
    policy_id = _nonempty_text(
        raw["policy_id"],
        where="curve model policy efficiency_operating_point.policy_id",
    )
    review_status = _review_status(
        raw["review_status"],
        where="curve model policy efficiency_operating_point.review_status",
    )
    if review_status != "approved":
        if any(
            raw[field] is not None
            for field in (
                "yield_retention_fraction",
                "maximum_marginal_gain_t_ha_per_kg_n",
                "marginal_gain_method",
                "model_concordance_tolerance_n_kg_ha",
                "zero_n_disposition",
                "prediction_grid_points",
                "prediction_grid_domain",
                "prediction_grid_spacing",
                "uncertainty_decision_rule",
                "uncertainty_disposition",
            )
        ):
            raise PolicyArtifactError(
                "Withheld efficiency operating-point policy cannot supply thresholds"
            )
        return MappingProxyType({"policy_id": policy_id, **dict(raw)})
    retention = _fraction(
        raw["yield_retention_fraction"],
        where=(
            "curve model policy efficiency_operating_point."
            "yield_retention_fraction"
        ),
        include_zero=False,
    )
    maximum_gain = _finite_number(
        raw["maximum_marginal_gain_t_ha_per_kg_n"],
        where=(
            "curve model policy efficiency_operating_point."
            "maximum_marginal_gain_t_ha_per_kg_n"
        ),
        minimum=0.0,
    )
    concordance = _finite_number(
        raw["model_concordance_tolerance_n_kg_ha"],
        where=(
            "curve model policy efficiency_operating_point."
            "model_concordance_tolerance_n_kg_ha"
        ),
        minimum=0.0,
    )
    marginal_gain_method = _nonempty_text(
        raw["marginal_gain_method"],
        where="curve model policy efficiency_operating_point.marginal_gain_method",
    )
    zero_n_disposition = _nonempty_text(
        raw["zero_n_disposition"],
        where="curve model policy efficiency_operating_point.zero_n_disposition",
    )
    prediction_grid_points = _positive_integer(
        raw["prediction_grid_points"],
        where=(
            "curve model policy efficiency_operating_point."
            "prediction_grid_points"
        ),
    )
    if prediction_grid_points < 3:
        raise PolicyArtifactError(
            "EFF-03 operating-point policy requires at least three prediction-grid points"
        )
    if raw["prediction_grid_domain"] != "observed_n_domain":
        raise PolicyArtifactError(
            "curve model policy efficiency_operating_point.prediction_grid_domain "
            "must be 'observed_n_domain'"
        )
    if raw["prediction_grid_spacing"] != "linear_inclusive_endpoints":
        raise PolicyArtifactError(
            "curve model policy efficiency_operating_point.prediction_grid_spacing "
            "must be 'linear_inclusive_endpoints'"
        )
    uncertainty_decision_rule = _nonempty_text(
        raw["uncertainty_decision_rule"],
        where=(
            "curve model policy efficiency_operating_point."
            "uncertainty_decision_rule"
        ),
    )
    if uncertainty_decision_rule != (
        "point_estimate_thresholds_with_validated_fitted_mean_interval_reporting"
    ):
        raise PolicyArtifactError(
            "curve model policy efficiency_operating_point.uncertainty_decision_rule "
            "must use point-estimate thresholds with validated fitted-mean interval "
            "reporting"
        )
    uncertainty_disposition = _nonempty_text(
        raw["uncertainty_disposition"],
        where=(
            "curve model policy efficiency_operating_point."
            "uncertainty_disposition"
        ),
    )
    if marginal_gain_method != "adjacent_prediction_grid_difference":
        raise PolicyArtifactError(
            "EFF-03 operating-point policy must use the executable adjacent-grid marginal-gain method"
        )
    if zero_n_disposition != "exclude_from_operating_point_search":
        raise PolicyArtifactError(
            "EFF-03 operating-point policy must state the executable zero-N disposition"
        )
    if uncertainty_disposition != "require_available_for_all_credible_models":
        raise PolicyArtifactError(
            "EFF-03 operating-point policy must require uncertainty for every credible model"
        )
    return MappingProxyType(
        {
            "policy_id": policy_id,
            "review_status": review_status,
            "yield_retention_fraction": retention,
            "maximum_marginal_gain_t_ha_per_kg_n": maximum_gain,
            "marginal_gain_method": marginal_gain_method,
            "model_concordance_tolerance_n_kg_ha": concordance,
            "zero_n_disposition": zero_n_disposition,
            "prediction_grid_points": prediction_grid_points,
            "prediction_grid_domain": "observed_n_domain",
            "prediction_grid_spacing": "linear_inclusive_endpoints",
            "uncertainty_decision_rule": uncertainty_decision_rule,
            "uncertainty_disposition": uncertainty_disposition,
        }
    )


def _model_credibility_policy(value: object) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({"review_status": "not_approved"})
    raw = _exact_object(
        value,
        required_keys={
            "policy_id",
            "review_status",
            "maximum_normalized_rmse",
            "maximum_parameter_influence_relative_shift",
            "minimum_influence_folds",
            "maximum_parameter_relative_standard_error",
            "parameter_scale_floor",
            "maximum_observed_step_decline_t_ha",
        },
        where="curve model policy model_credibility",
    )
    policy_id = _nonempty_text(
        raw["policy_id"],
        where="curve model policy model_credibility.policy_id",
    )
    review_status = _review_status(
        raw["review_status"],
        where="curve model policy model_credibility.review_status",
    )
    minimum_influence_folds = _positive_integer(
        raw["minimum_influence_folds"],
        where=(
            "curve model policy model_credibility.minimum_influence_folds"
        ),
    )
    if minimum_influence_folds < 2:
        raise PolicyArtifactError(
            "curve model policy model_credibility.minimum_influence_folds "
            "must be >= 2"
        )
    return MappingProxyType(
        {
            "policy_id": policy_id,
            "review_status": review_status,
            "maximum_normalized_rmse": _finite_number(
                raw["maximum_normalized_rmse"],
                where=(
                    "curve model policy model_credibility."
                    "maximum_normalized_rmse"
                ),
                minimum=0.0,
                strictly_greater=True,
            ),
            "maximum_parameter_influence_relative_shift": _finite_number(
                raw["maximum_parameter_influence_relative_shift"],
                where=(
                    "curve model policy model_credibility."
                    "maximum_parameter_influence_relative_shift"
                ),
                minimum=0.0,
                strictly_greater=True,
            ),
            "minimum_influence_folds": minimum_influence_folds,
            "maximum_parameter_relative_standard_error": _finite_number(
                raw["maximum_parameter_relative_standard_error"],
                where=(
                    "curve model policy model_credibility."
                    "maximum_parameter_relative_standard_error"
                ),
                minimum=0.0,
                strictly_greater=True,
            ),
            "parameter_scale_floor": _finite_number(
                raw["parameter_scale_floor"],
                where=(
                    "curve model policy model_credibility.parameter_scale_floor"
                ),
                minimum=0.0,
                strictly_greater=True,
            ),
            "maximum_observed_step_decline_t_ha": _finite_number(
                raw["maximum_observed_step_decline_t_ha"],
                where=(
                    "curve model policy model_credibility."
                    "maximum_observed_step_decline_t_ha"
                ),
                minimum=0.0,
                strictly_greater=True,
            ),
        }
    )


def load_curve_model_policy(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[ArtifactAuthority, CurveModelPolicy]:
    """Load the complete reviewed curve, uncertainty, and dataset-view controls."""

    authority, payload = _load_artifact(
        path,
        expected_sha256=expected_sha256,
        expected_type="curve_model_policy",
    )
    records = _records(payload, where="curve model policy")
    if len(records) != 1:
        raise PolicyArtifactError(
            "Curve model policy must contain exactly one complete record"
        )
    required_record_keys = {
        "policy_id",
        "restricted_fit_models",
        "model_gates",
        "baseline_response",
        "recommendation_set",
        "material_disagreement",
        "uncertainty",
        "economic_scenarios",
    }
    if not isinstance(records[0], Mapping):
        raise PolicyArtifactError("curve model policy record must be an object")
    observed_record_keys = set(records[0])
    optional_record_keys = {
        "first_stage_contextual_uncertainty",
        "efficiency_metric",
        "efficiency_operating_point",
        "asymptote_reporting",
        "asymptote_support",
        "model_credibility",
    }
    if (
        required_record_keys <= observed_record_keys
        and observed_record_keys <= required_record_keys | optional_record_keys
    ):
        required_record_keys = observed_record_keys
    raw = _exact_object(
        records[0],
        required_keys=required_record_keys,
        where="curve model policy record",
    )
    policy_id = _nonempty_text(
        raw["policy_id"],
        where="curve model policy record.policy_id",
    )
    raw_restricted_models = _string_tuple(
        raw["restricted_fit_models"],
        where="curve model policy record.restricted_fit_models",
    )
    restricted_set = set(raw_restricted_models)
    if (
        restricted_set - set(_MODEL_ROSTER)
        or restricted_set == set(_MODEL_ROSTER)
    ):
        raise PolicyArtifactError(
            "Restricted-fit models must be a nonempty proper subset of the "
            "canonical model roster"
        )
    restricted_models = tuple(
        model_name
        for model_name in _MODEL_ROSTER
        if model_name in restricted_set
    )
    model_gates = _curve_model_gates(raw["model_gates"])

    raw_baseline = _exact_object(
        raw["baseline_response"],
        required_keys={"policy_id", "mode", "classes"},
        where="curve model policy baseline_response",
    )
    baseline_policy_id = _nonempty_text(
        raw_baseline["policy_id"],
        where="curve model policy baseline_response.policy_id",
    )
    baseline_mode = _nonempty_text(
        raw_baseline["mode"],
        where="curve model policy baseline_response.mode",
    )
    if baseline_mode != "separate_verified_classes":
        raise PolicyArtifactError(
            "Baseline-response policy must keep verified control classes separate"
        )
    baseline_classes = _string_tuple(
        raw_baseline["classes"],
        where="curve model policy baseline_response.classes",
    )
    if baseline_classes != _BASELINE_CLASSES:
        raise PolicyArtifactError(
            "Baseline-response classes must be the two verified, separately "
            "reported control classes"
        )

    raw_recommendation = _exact_object(
        raw["recommendation_set"],
        required_keys={
            "policy_id",
            "required_classes",
            "optional_classes",
            "membership_status_field",
            "verified_status",
        },
        where="curve model policy recommendation_set",
    )
    recommendation_policy_id = _nonempty_text(
        raw_recommendation["policy_id"],
        where="curve model policy recommendation_set.policy_id",
    )
    required_classes = _string_tuple(
        raw_recommendation["required_classes"],
        where="curve model policy recommendation_set.required_classes",
    )
    optional_classes = _string_tuple(
        raw_recommendation["optional_classes"],
        where="curve model policy recommendation_set.optional_classes",
    )
    if (
        required_classes != _RECOMMENDATION_REQUIRED_CLASSES
        or optional_classes != _RECOMMENDATION_OPTIONAL_CLASSES
    ):
        raise PolicyArtifactError(
            "Recommendation-set controls must declare the complete required "
            "and optional class membership"
        )
    membership_status_field = _nonempty_text(
        raw_recommendation["membership_status_field"],
        where="curve model policy recommendation_set.membership_status_field",
    )
    verified_status = _nonempty_text(
        raw_recommendation["verified_status"],
        where="curve model policy recommendation_set.verified_status",
    )
    if (
        membership_status_field != "recommendation_set_membership_status"
        or verified_status != "verified_context_comparable"
    ):
        raise PolicyArtifactError(
            "Recommendation-set controls must require verified "
            "context-comparable membership evidence"
        )

    (
        disagreement_policy_id,
        disagreement_status,
        disagreement_tolerances,
    ) = _material_disagreement_policy(raw["material_disagreement"])
    (
        uncertainty_policy_id,
        uncertainty_status,
        uncertainty_method,
        uncertainty_method_contract,
        uncertainty_basis,
    ) = _uncertainty_policy(raw["uncertainty"])
    first_stage_contextual_uncertainty = (
        _first_stage_contextual_uncertainty_policy(
            raw.get("first_stage_contextual_uncertainty"),
            active_within_model_method=uncertainty_method,
        )
    )
    (
        economic_policy_id,
        economic_status,
        economic_table_id,
        economic_table_version,
        economic_scenarios,
    ) = _economic_policy(raw["economic_scenarios"])

    return (
        authority,
        CurveModelPolicy(
            policy_id=policy_id,
            restricted_fit_models=restricted_models,
            model_gates=model_gates,
            baseline_policy_id=baseline_policy_id,
            baseline_response_policy=baseline_mode,
            baseline_classes=baseline_classes,
            recommendation_policy_id=recommendation_policy_id,
            recommendation_required_classes=required_classes,
            recommendation_optional_classes=optional_classes,
            recommendation_membership_status_field=membership_status_field,
            recommendation_verified_status=verified_status,
            material_disagreement_policy_id=disagreement_policy_id,
            material_disagreement_review_status=disagreement_status,
            material_disagreement_tolerances=disagreement_tolerances,
            uncertainty_policy_id=uncertainty_policy_id,
            uncertainty_review_status=uncertainty_status,
            uncertainty_method=uncertainty_method,
            uncertainty_method_contract=uncertainty_method_contract,
            uncertainty_evidence_basis=uncertainty_basis,
            first_stage_contextual_uncertainty_policy=(
                first_stage_contextual_uncertainty
            ),
            economic_policy_id=economic_policy_id,
            economic_review_status=economic_status,
            economic_table_id=economic_table_id,
            economic_table_version=economic_table_version,
            economic_scenarios=economic_scenarios,
        ),
    )


def load_analysis_policy_bundle(
    *,
    support_path: str | Path,
    support_sha256: str,
    representation_path: str | Path,
    representation_sha256: str,
    estimand_path: str | Path,
    estimand_sha256: str,
    hypothesis_path: str | Path,
    hypothesis_sha256: str,
    model_policy_path: str | Path,
    model_policy_sha256: str,
    required_factor_engine_pairs: Sequence[tuple[str, str]] = (),
) -> AnalysisPolicyBundle:
    support_authority, support_rules = load_support_table(
        support_path,
        expected_sha256=support_sha256,
    )
    representation_authority, factor_representations = load_factor_representations(
        representation_path,
        expected_sha256=representation_sha256,
        required_pairs=required_factor_engine_pairs,
    )
    estimand_authority, estimands = load_estimands(
        estimand_path,
        expected_sha256=estimand_sha256,
    )
    hypothesis_authority, hypotheses = load_hypothesis_snapshot(
        hypothesis_path,
        expected_sha256=hypothesis_sha256,
    )
    model_authority, curve_model_policy = load_curve_model_policy(
        model_policy_path,
        expected_sha256=model_policy_sha256,
    )
    support_ids = {rule.rule_id for rule in support_rules}
    estimand_by_id = {estimand.estimand_id: estimand for estimand in estimands}
    support_by_id = {rule.rule_id: rule for rule in support_rules}
    representations = {
        (representation.factor_name, representation.engine)
        for representation in factor_representations
    }
    for hypothesis in hypotheses:
        if hypothesis.support_rule_id not in support_ids:
            raise PolicyArtifactError(
                f"{hypothesis.hypothesis_id} refers to unknown support rule "
                f"{hypothesis.support_rule_id!r}"
            )
        if hypothesis.estimand_id not in estimand_by_id:
            raise PolicyArtifactError(
                f"{hypothesis.hypothesis_id} refers to unknown estimand "
                f"{hypothesis.estimand_id!r}"
            )
        estimand = estimand_by_id[hypothesis.estimand_id]
        if estimand.hypothesis_id != hypothesis.hypothesis_id:
            raise PolicyArtifactError(
                f"{hypothesis.hypothesis_id} refers to an estimand owned by another hypothesis"
            )
        if estimand.outcome != hypothesis.outcome:
            raise PolicyArtifactError(
                f"{hypothesis.hypothesis_id} has inconsistent estimand and hypothesis outcomes"
            )
        support_rule = support_by_id[hypothesis.support_rule_id]
        if (
            support_rule.outcome != hypothesis.outcome
            or support_rule.analysis_family != hypothesis.analysis_family
        ):
            raise PolicyArtifactError(
                f"{hypothesis.hypothesis_id} has an inconsistent outcome-support rule"
            )
        if hypothesis.status == "enabled":
            missing = {
                (factor_name, hypothesis.engine)
                for factor_name in hypothesis.factor_names
            } - representations
            if missing:
                formatted = ", ".join(
                    f"{factor}/{engine}"
                    for factor, engine in sorted(missing)
                )
                raise PolicyArtifactError(
                    f"{hypothesis.hypothesis_id} lacks factor representations: {formatted}"
                )
            for sensitivity in hypothesis.sensitivity_specifications:
                sensitivity_id = str(sensitivity["sensitivity_id"])
                sensitivity_support_id = str(sensitivity["support_rule_id"])
                sensitivity_support = support_by_id.get(sensitivity_support_id)
                if sensitivity_support is None:
                    raise PolicyArtifactError(
                        f"{hypothesis.hypothesis_id} sensitivity {sensitivity_id} refers "
                        f"to unknown support rule {sensitivity_support_id!r}"
                    )
                if (
                    sensitivity_support.outcome != hypothesis.outcome
                    or sensitivity_support.analysis_family
                    != sensitivity["analysis_family"]
                ):
                    raise PolicyArtifactError(
                        f"{hypothesis.hypothesis_id} sensitivity {sensitivity_id} has "
                        "an inconsistent outcome-support rule"
                    )
                sensitivity_missing = {
                    (factor_name, str(sensitivity["engine"]))
                    for factor_name in sensitivity["factor_names"]
                } - representations
                if sensitivity_missing:
                    formatted = ", ".join(
                        f"{factor}/{engine}"
                        for factor, engine in sorted(sensitivity_missing)
                    )
                    raise PolicyArtifactError(
                        f"{hypothesis.hypothesis_id} sensitivity {sensitivity_id} "
                        f"lacks factor representations: {formatted}"
                    )
    return AnalysisPolicyBundle(
        support_authority=support_authority,
        representation_authority=representation_authority,
        estimand_authority=estimand_authority,
        hypothesis_authority=hypothesis_authority,
        model_authority=model_authority,
        support_rules=support_rules,
        factor_representations=factor_representations,
        estimands=estimands,
        hypotheses=hypotheses,
        curve_model_policy=curve_model_policy,
    )


def load_analysis_policy_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    project_root: str | Path,
    required_factor_engine_pairs: Sequence[tuple[str, str]] = (),
) -> AnalysisPolicyBundle:
    """Load a hash-bound bundle whose component paths cannot escape the project."""

    manifest_path = Path(path).resolve()
    root = Path(project_root).resolve()
    if not manifest_path.is_relative_to(root):
        raise PolicyArtifactError("Analysis policy manifest must be inside the project root")
    if manifest_path.suffix.casefold() != ".json":
        raise PolicyArtifactError("Analysis policy manifest must use the JSON format")
    _, payload = _load_artifact(
        manifest_path,
        expected_sha256=expected_sha256,
        expected_type="analysis_policy_bundle",
    )
    raw_artifacts = payload.get("artifacts")
    required_artifacts = {
        "support_table",
        "factor_representations",
        "estimands",
        "hypotheses",
        "curve_model_policy",
    }
    if not isinstance(raw_artifacts, Mapping) or set(raw_artifacts) != required_artifacts:
        raise PolicyArtifactError(
            "Analysis policy manifest must declare exactly the required component artifacts"
        )

    resolved: dict[str, tuple[Path, str]] = {}
    for name in sorted(required_artifacts):
        reference = raw_artifacts[name]
        if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
            raise PolicyArtifactError(
                f"Analysis policy manifest reference {name!r} must contain path and sha256"
            )
        raw_path = reference["path"]
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise PolicyArtifactError(
                f"Analysis policy manifest reference {name!r} has an invalid path"
            )
        relative_path = Path(raw_path)
        if relative_path.is_absolute():
            raise PolicyArtifactError(
                f"Analysis policy manifest reference {name!r} must be project-relative"
            )
        component_path = (root / relative_path).resolve()
        if not component_path.is_relative_to(root):
            raise PolicyArtifactError(
                f"Analysis policy manifest reference {name!r} escapes the project root"
            )
        if component_path.suffix.casefold() != ".json":
            raise PolicyArtifactError(
                f"Analysis policy manifest reference {name!r} must use the JSON format"
            )
        component_sha256 = reference["sha256"]
        if (
            not isinstance(component_sha256, str)
            or _SHA256_RE.fullmatch(component_sha256.lower()) is None
        ):
            raise PolicyArtifactError(
                f"Analysis policy manifest reference {name!r} has an invalid SHA-256"
            )
        resolved[name] = (component_path, component_sha256.lower())

    return load_analysis_policy_bundle(
        support_path=resolved["support_table"][0],
        support_sha256=resolved["support_table"][1],
        representation_path=resolved["factor_representations"][0],
        representation_sha256=resolved["factor_representations"][1],
        estimand_path=resolved["estimands"][0],
        estimand_sha256=resolved["estimands"][1],
        hypothesis_path=resolved["hypotheses"][0],
        hypothesis_sha256=resolved["hypotheses"][1],
        model_policy_path=resolved["curve_model_policy"][0],
        model_policy_sha256=resolved["curve_model_policy"][1],
        required_factor_engine_pairs=required_factor_engine_pairs,
    )


__all__ = [
    "AnalysisPolicyBundle",
    "ArtifactAuthority",
    "CurveModelGate",
    "CurveModelPolicy",
    "EffectiveHypothesis",
    "Estimand",
    "FactorRepresentation",
    "PolicyArtifactError",
    "SupportRule",
    "load_analysis_policy_bundle",
    "load_analysis_policy_manifest",
    "load_curve_model_policy",
    "load_estimands",
    "load_factor_representations",
    "load_hypothesis_snapshot",
    "load_support_table",
]
