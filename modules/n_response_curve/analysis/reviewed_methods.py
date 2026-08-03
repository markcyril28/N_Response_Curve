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
UNCERTAINTY_METHOD_SPECS = MappingProxyType(
    {
        "reported_se_delta_interval_95pct": MappingProxyType(
            {
                "method_spec_version": "1.0.0",
                "linkage_rule": "row_aligned_reported_mean_standard_error",
                "point_estimation_weighting": "unweighted_least_squares",
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
    "UNCERTAINTY_METHOD_CONFIDENCE_LEVELS",
    "UNCERTAINTY_METHOD_REQUIRED_EVIDENCE",
    "UNCERTAINTY_METHOD_SPECS",
]
