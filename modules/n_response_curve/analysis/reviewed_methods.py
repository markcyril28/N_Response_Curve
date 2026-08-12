from __future__ import annotations

from types import MappingProxyType


ECONOMIC_DECISION_RULE = "maximize_net_return_within_observed_n_domain"
ECONOMIC_GRAIN_PRICE_TO_PER_TONNE = MappingProxyType(
    {
        "currency_per_kg_grain": 1000.0,
        "currency_per_t_grain": 1.0,
    }
)
ECONOMIC_N_COST_UNIT = "currency_per_kg_n"

# MOD-09 Option A: the one immutable estimator contract consumed by both curve
# fitting and uncertainty validation. Duplicating any of these values in a
# calling module reintroduces the divergence this object exists to prevent.
#
# `primary_*` is the authoritative-target design. `sensitivity_*` is the
# separately labelled inverse-variance arm, which is never selected
# automatically: `MOD09_ESTIMATOR_SWITCHING` records that prohibition so a
# reader cannot mistake the sensitivity arm for a fallback.
#
# `nuisance_parameter_count` is the residual-variance parameter that the
# Gaussian likelihood estimates alongside the mean parameters. Information
# criteria must count it in `k`; omitting it understates every candidate's
# complexity by exactly one and does so unequally across comparisons.
MOD09_ESTIMATOR_SPECIFICATION = MappingProxyType(
    {
        "estimator_policy_id": "MOD-09-option-a",
        "estimator_spec_version": "1.0.0",
        "analysis_grain": "one_reviewed_treatment_mean_per_distinct_n_level",
        "primary_estimator_name": "unweighted_least_squares",
        "primary_objective": "sum_of_squared_residuals",
        "primary_likelihood": "gaussian_equal_variance_treatment_mean",
        "primary_weights": "equal_weight_per_reviewed_treatment_mean",
        "residual_covariance_structure": (
            "independent_equal_variance_treatment_mean_errors"
        ),
        "nuisance_parameter_names": ("residual_variance",),
        "nuisance_parameter_count": 1,
        "sensitivity_estimator_name": "inverse_variance_weighted_least_squares",
        "sensitivity_weights": "inverse_reported_mean_standard_error_squared",
        "sensitivity_label": "separately_labelled_inverse_variance_sensitivity",
        "sensitivity_required_evidence": (
            "reported_standard_error",
            "verified_experimental_unit",
            "verified_mean_independence",
        ),
        "prediction_domain": "observed_n_domain",
        # The equal-variance assumption above is exactly right only when every
        # treatment mean rests on the same number of replicates. With unequal
        # replication Var(mean_i) = sigma^2 / n_i, so residuals are
        # heteroscedastic and nominal standard errors are wrong. Per-level
        # replication counts are therefore carried on every attempt so this
        # check is available to the reviewer who binds the authority artifact.
        "replication_balance_requirement": (
            "equal_replicate_count_per_reviewed_treatment_mean"
        ),
        "authoritative_status": (
            "exploratory_until_scientific_policy_authority_approved"
        ),
    }
)
MOD09_ESTIMATOR_SWITCHING = "no_automatic_estimator_or_weighting_fallback"
UNCERTAINTY_METHOD_SPECS = MappingProxyType(
    {
        "reported_se_delta_interval_95pct": MappingProxyType(
            {
                "method_spec_version": "1.0.0",
                "linkage_rule": "row_aligned_reported_mean_standard_error",
                "point_estimation_weighting": MOD09_ESTIMATOR_SPECIFICATION[
                    "primary_estimator_name"
                ],
                "likelihood": "none_delta_covariance_propagation",
                "observation_error_covariance": (
                    "diagonal_reported_mean_se_squared"
                ),
                "independent_unit_assumption": (
                    "treatment_mean_errors_independent"
                ),
                "parameter_covariance": "linearized_delta_sandwich",
                "interval_target": "fitted_mean",
                "interval_distribution": "normal",
                "confidence_level": 0.95,
                "prediction_domain": "observed_n_domain",
            }
        ),
    }
)
UNCERTAINTY_METHOD_REQUIRED_EVIDENCE = MappingProxyType(
    {
        "reported_se_delta_interval_95pct": (
            "reported_standard_error",
            "verified_mean_independence",
        ),
    }
)
UNCERTAINTY_METHOD_CONFIDENCE_LEVELS = MappingProxyType(
    {
        method: float(spec["confidence_level"])
        for method, spec in UNCERTAINTY_METHOD_SPECS.items()
    }
)


__all__ = [
    "ECONOMIC_DECISION_RULE",
    "ECONOMIC_GRAIN_PRICE_TO_PER_TONNE",
    "ECONOMIC_N_COST_UNIT",
    "MOD09_ESTIMATOR_SPECIFICATION",
    "MOD09_ESTIMATOR_SWITCHING",
    "UNCERTAINTY_METHOD_CONFIDENCE_LEVELS",
    "UNCERTAINTY_METHOD_REQUIRED_EVIDENCE",
    "UNCERTAINTY_METHOD_SPECS",
]
