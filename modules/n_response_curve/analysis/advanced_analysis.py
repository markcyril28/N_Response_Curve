from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .analysis_matrix import AnalysisCandidate
from .factor_catalog import FactorCatalogEntry, factor_value
from .values import finite_number, outcome_is_present


@dataclass(frozen=True)
class AdvancedAnalysisResult:
    status: str
    result_type: str | None
    reason_codes: tuple[str, ...]
    records: tuple[Mapping[str, Any], ...]


def _skipped(reason: str) -> AdvancedAnalysisResult:
    return AdvancedAnalysisResult("skipped", None, (reason,), ())


def _finite_mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else None


def _predictive_result(
    candidate: AnalysisCandidate,
    rows: Sequence[Mapping[str, Any]],
    factor_entries: Mapping[str, FactorCatalogEntry],
) -> AdvancedAnalysisResult:
    if not candidate.factor_names:
        return _skipped("PREDICTIVE_FACTORS_REQUIRED")
    entries = [factor_entries[name] for name in candidate.factor_names]
    normalized: list[dict[str, Any]] = []
    for row in rows:
        outcome = row.get(candidate.curve_outcome)
        values = {
            name: factor_value(
                row,
                name,
                representation=candidate.factor_representations.get(name),
            )
            for name in candidate.factor_names
        }
        group = str(row.get("study_uid") or "")
        if (
            not group
            or not outcome_is_present(outcome)
            or any(value is None for value in values.values())
        ):
            continue
        normalized.append(
            {**values, "__outcome": outcome, "__group": group}
        )
    groups = np.asarray(
        [row["__group"] for row in normalized],
        dtype=object,
    )
    if len(normalized) < 8 or len(set(groups)) < 4:
        return _skipped("INSUFFICIENT_GROUPED_PREDICTIVE_SUPPORT")

    frame = pd.DataFrame(normalized)
    features = frame[list(candidate.factor_names)]
    numeric = [
        entry.factor_name
        for entry in entries
        if entry.data_type == "numeric"
    ]
    categorical = [
        entry.factor_name
        for entry in entries
        if entry.data_type != "numeric"
    ]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical,
            )
        )
    preprocess = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    numeric_outcome = all(
        finite_number(value) is not None
        for value in frame["__outcome"]
    )
    if numeric_outcome:
        response = np.asarray(
            [float(value) for value in frame["__outcome"]],
            dtype=float,
        )
        parameter_grid: tuple[Mapping[str, float], ...] = (
            {"alpha": 0.001, "l1_ratio": 0.2},
            {"alpha": 0.01, "l1_ratio": 0.5},
            {"alpha": 0.1, "l1_ratio": 0.8},
        )
        scoring = "neg_root_mean_squared_error"
    else:
        response = np.asarray(
            [str(value) for value in frame["__outcome"]],
            dtype=object,
        )
        if len(set(response)) < 2:
            return _skipped("PREDICTIVE_OUTCOME_HAS_ONE_CLASS")
        parameter_grid = (
            {"C": 0.1, "l1_ratio": 0.2},
            {"C": 1.0, "l1_ratio": 0.5},
            {"C": 10.0, "l1_ratio": 0.8},
        )
        scoring = "balanced_accuracy"

    group_count = len(set(groups))
    maximum_loso_studies = candidate.support_policy.get(
        "maximum_loso_studies"
    )
    if (
        isinstance(maximum_loso_studies, bool)
        or not isinstance(maximum_loso_studies, int)
        or maximum_loso_studies < 4
    ):
        return _skipped("GROUPED_VALIDATION_POLICY_REQUIRED")
    outcome_classes = set(response) if not numeric_outcome else set()

    def training_splits_are_estimable(
        splits: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> bool:
        return all(
            len(train) > 0
            and len(test) > 0
            and (
                numeric_outcome
                or (
                    set(response[train]) == outcome_classes
                    and set(response[test]) == outcome_classes
                )
            )
            for train, test in splits
        )

    if group_count <= maximum_loso_studies:
        outer_splits = tuple(
            LeaveOneGroupOut().split(features, response, groups)
        )
        resampling_design = "leave_one_study_out"
    else:
        outer_splits = ()
        resampling_design = "grouped_k_fold"
    if not outer_splits or not training_splits_are_estimable(outer_splits):
        n_splits = min(5, group_count)
        class_study_support = (
            numeric_outcome
            or all(
                len(set(groups[response == level])) >= n_splits
                for level in outcome_classes
            )
        )
        if not numeric_outcome and class_study_support:
            outer = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=False,
            )
            resampling_design = "stratified_grouped_k_fold"
        else:
            outer = GroupKFold(n_splits=n_splits)
            resampling_design = "grouped_k_fold"
        outer_splits = tuple(outer.split(features, response, groups))
    if not training_splits_are_estimable(outer_splits):
        return _skipped("GROUPED_FOLD_MISSING_OUTCOME_CLASS")
    performance: list[dict[str, Any]] = []
    importances: dict[str, list[float]] = {
        name: [] for name in candidate.factor_names
    }
    selected_parameters: list[Mapping[str, float]] = []
    balanced_accuracy_unavailable_study_count = 0

    for fold, (train, test) in enumerate(
        outer_splits,
        start=1,
    ):
        train_groups = groups[train]
        if not numeric_outcome and len(set(response[train])) < 2:
            return _skipped("GROUPED_FOLD_HAS_ONE_OUTCOME_CLASS")
        inner_group_count = len(set(train_groups))
        inner_n_splits = min(3, inner_group_count)
        if inner_n_splits < 2:
            return _skipped("INNER_GROUPED_VALIDATION_UNSUPPORTED")
        if (
            not numeric_outcome
            and all(
                len(set(train_groups[response[train] == level]))
                >= inner_n_splits
                for level in outcome_classes
            )
        ):
            inner = StratifiedGroupKFold(
                n_splits=inner_n_splits,
                shuffle=False,
            )
        else:
            inner = GroupKFold(n_splits=inner_n_splits)
        inner_splits = tuple(
            inner.split(
                features.iloc[train],
                response[train],
                train_groups,
            )
        )
        if not numeric_outcome and any(
            set(response[train][inner_train]) != outcome_classes
            for inner_train, _ in inner_splits
        ):
            return _skipped("INNER_GROUPED_FOLD_MISSING_OUTCOME_CLASS")
        best_score = -math.inf
        best_model: Pipeline | None = None
        best_parameters: Mapping[str, float] | None = None
        for parameters in parameter_grid:
            if numeric_outcome:
                estimator = ElasticNet(
                    random_state=20260720,
                    max_iter=20000,
                    **parameters,
                )
            else:
                estimator = LogisticRegression(
                    solver="saga",
                    max_iter=10000,
                    random_state=20260720,
                    C=float(parameters["C"]),
                    l1_ratio=float(parameters["l1_ratio"]),
                )
            model = Pipeline(
                [
                    ("preprocess", clone(preprocess)),
                    ("model", estimator),
                ]
            )
            try:
                scores = cross_val_score(
                    model,
                    features.iloc[train],
                    response[train],
                    groups=train_groups,
                    cv=inner_splits,
                    scoring=scoring,
                    error_score="raise",
                )
                score = float(np.mean(scores))
            except ValueError:
                score = -math.inf
            if score > best_score:
                best_score = score
                best_model = model
                best_parameters = parameters
        if best_model is None or best_parameters is None:
            return _skipped("GROUPED_TUNING_FAILED")

        best_model.fit(features.iloc[train], response[train])
        predicted = best_model.predict(features.iloc[test])
        if numeric_outcome:
            performance.append(
                {
                    "rmse": math.sqrt(
                        mean_squared_error(
                            response[test],
                            predicted,
                        )
                    ),
                    "mae": mean_absolute_error(
                        response[test],
                        predicted,
                    ),
                    "r2": (
                        r2_score(response[test], predicted)
                        if len(test) > 1
                        else math.nan
                    ),
                }
            )
            importance_scoring = "neg_root_mean_squared_error"
        else:
            probabilities = best_model.predict_proba(
                features.iloc[test]
            )
            performance.append(
                {
                    "balanced_accuracy": balanced_accuracy_score(
                        response[test],
                        predicted,
                    ),
                    "log_loss": log_loss(
                        response[test],
                        probabilities,
                        labels=best_model.classes_,
                    ),
                }
            )
            importance_scoring = "balanced_accuracy"
        measured = permutation_importance(
            best_model,
            features.iloc[test],
            response[test],
            scoring=importance_scoring,
            n_repeats=5,
            random_state=20260720 + fold,
        )
        for name, value in zip(
            candidate.factor_names,
            measured.importances_mean,
            strict=True,
        ):
            importances[name].append(float(value))
        selected_parameters.append(dict(best_parameters))

    metric_names = sorted(
        {name for fold_result in performance for name in fold_result}
    )
    performance_record: dict[str, Any] = {
        "record_type": "performance",
        "candidate_id": candidate.candidate_id,
        "outcome": candidate.curve_outcome,
        "outcome_kind": (
            "continuous" if numeric_outcome else "categorical"
        ),
        "group_cross_validation_by": "study_uid",
        "resampling_design": "adaptive_repeated_group_shuffle",
        "configured_outer_repeat_count": outer_repeat_count,
        "outer_fold_count": len(performance),
        "tuning_search_space": list(parameter_grid),
        "selected_parameters_by_fold": selected_parameters,
        "preprocessing_scope": "fit within each training fold",
    }
    for metric in metric_names:
        performance_record[f"mean_{metric}"] = _finite_mean(
            [fold_result.get(metric, math.nan) for fold_result in performance]
        )
    records: list[dict[str, Any]] = [performance_record]
    records.extend(
        {
            "record_type": "permutation_importance",
            "candidate_id": candidate.candidate_id,
            "outcome": candidate.curve_outcome,
            "factor_name": name,
            "mean_held_out_importance": statistics.fmean(values),
            "importance_fold_sd": (
                statistics.stdev(values) if len(values) > 1 else 0.0
            ),
            "group_cross_validation_by": "study_uid",
        }
        for name, values in importances.items()
    )
    return AdvancedAnalysisResult(
        "completed",
        "grouped_penalized_prediction",
        (),
        tuple(records),
    )


_CLUSTER_FEATURES = (
    "agronomic_optimum_n_kg_ha",
    "plateau_onset_n_kg_ha",
    "predicted_max_yield_t_ha",
    "observed_max_yield_t_ha",
    "yield_at_zero_n_t_ha",
    "yield_response_above_zero_n_t_ha",
)


def _clustering_result(
    candidate: AnalysisCandidate,
    rows: Sequence[Mapping[str, Any]],
) -> AdvancedAnalysisResult:
    features = [
        name
        for name in _CLUSTER_FEATURES
        if sum(
            finite_number(row.get(name)) is not None
            for row in rows
        )
        >= 6
    ]
    complete = [
        row
        for row in rows
        if row.get("response_series_uid")
        and all(
            finite_number(row.get(name)) is not None
            for name in features
        )
    ]
    if len(features) < 2 or len(complete) < 6:
        return _skipped("INSUFFICIENT_COMPARABLE_CURVE_FEATURES")
    matrix = np.asarray(
        [
            [float(row[name]) for name in features]
            for row in complete
        ],
        dtype=float,
    )
    if np.any(np.var(matrix, axis=0) <= np.finfo(float).eps):
        return _skipped("ZERO_VARIANCE_CLUSTER_FEATURE")
    scaled = StandardScaler().fit_transform(matrix)
    distinct_point_count = len(np.unique(scaled, axis=0))
    if distinct_point_count < 2:
        return _skipped("INSUFFICIENT_DISTINCT_CLUSTER_POINTS")
    best: tuple[float, int, np.ndarray] | None = None
    for cluster_count in range(
        2,
        min(5, len(complete) - 1, distinct_point_count) + 1,
    ):
        try:
            labels = KMeans(
                n_clusters=cluster_count,
                random_state=20260720,
                n_init=20,
            ).fit_predict(scaled)
            if len(set(int(label) for label in labels)) < 2:
                continue
            score = float(silhouette_score(scaled, labels))
        except (FloatingPointError, ValueError):
            continue
        if best is None or (score, -cluster_count) > (
            best[0],
            -best[1],
        ):
            best = (score, cluster_count, labels)
    if best is None:
        return _skipped("CLUSTER_SELECTION_FAILED")
    silhouette, cluster_count, labels = best
    try:
        alternate = [
            KMeans(
                n_clusters=cluster_count,
                random_state=20260721 + seed,
                n_init=20,
            ).fit_predict(scaled)
            for seed in range(5)
        ]
    except (FloatingPointError, ValueError):
        return _skipped("CLUSTER_STABILITY_ESTIMATION_FAILED")
    stability = statistics.fmean(
        adjusted_rand_score(labels, other)
        for other in alternate
    )
    records: list[dict[str, Any]] = [
        {
            "record_type": "cluster_metrics",
            "candidate_id": candidate.candidate_id,
            "cluster_count": cluster_count,
            "silhouette": silhouette,
            "assignment_stability": stability,
            "feature_names": features,
            "missing_data_policy": (
                "factor-specific complete cases; no imputation"
            ),
        }
    ]
    records.extend(
        {
            "record_type": "cluster_assignment",
            "candidate_id": candidate.candidate_id,
            "response_series_uid": str(row["response_series_uid"]),
            "cluster_id": int(label),
            "cluster_count": cluster_count,
            "assignment_stability": stability,
        }
        for row, label in zip(complete, labels, strict=True)
    )
    return AdvancedAnalysisResult(
        "completed",
        "stable_curve_feature_clusters",
        (),
        tuple(records),
    )


def _robustness_result(
    candidate: AnalysisCandidate,
    rows: Sequence[Mapping[str, Any]],
) -> AdvancedAnalysisResult:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        value = finite_number(row.get(candidate.curve_outcome))
        source = str(row.get("source_name") or "")
        study_uid = str(row.get("study_uid") or "")
        if value is not None and source and study_uid:
            grouped.setdefault(source, {}).setdefault(study_uid, []).append(value)
    if len(grouped) < 2 or any(
        len(studies) < 2 for studies in grouped.values()
    ):
        return _skipped("INSUFFICIENT_SOURCE_ROBUSTNESS_SUPPORT")
    study_means = {
        source: {
            study_uid: statistics.fmean(values)
            for study_uid, values in studies.items()
        }
        for source, studies in grouped.items()
    }
    pooled_study_means = [
        value
        for studies in study_means.values()
        for value in studies.values()
    ]
    pooled_mean = statistics.fmean(pooled_study_means)
    source_means = {
        source: statistics.fmean(studies.values())
        for source, studies in study_means.items()
    }
    records: list[dict[str, Any]] = []
    for source, studies in sorted(study_means.items()):
        mean = source_means[source]
        leave_one_source_out_study_means = [
            value
            for other_source, other_studies in study_means.items()
            if other_source != source
            for value in other_studies.values()
        ]
        leave_one_source_out_mean = statistics.fmean(leave_one_source_out_study_means)
        omission_shift = leave_one_source_out_mean - pooled_mean
        standard_error = (
            statistics.stdev(studies.values()) / math.sqrt(len(studies))
            if len(studies) > 1
            else 0.0
        )
        records.append(
            {
                "record_type": "source_omission_sensitivity",
                "candidate_id": candidate.candidate_id,
                "outcome": candidate.curve_outcome,
                "omitted_source_family": source,
                "omitted_source_study_count": len(studies),
                "total_independent_study_count": len(pooled_study_means),
                "source_mean": mean,
                "pooled_mean": pooled_mean,
                "leave_one_source_out_mean": leave_one_source_out_mean,
                "omission_shift_from_pooled": omission_shift,
                "absolute_omission_shift": abs(omission_shift),
                "relative_omission_shift": (
                    omission_shift / abs(pooled_mean)
                    if pooled_mean != 0.0
                    else None
                ),
                "interval_low": mean - 1.96 * standard_error,
                "interval_high": mean + 1.96 * standard_error,
                "interpretation_status": "DESCRIPTIVE_SOURCE_OMISSION_ONLY",
            }
        )
    return AdvancedAnalysisResult(
        "completed",
        "source_omission_sensitivity",
        (),
        tuple(records),
    )


def execute_advanced_candidate(
    candidate: AnalysisCandidate,
    rows: Sequence[Mapping[str, Any]],
    factor_entries: Mapping[str, FactorCatalogEntry],
) -> AdvancedAnalysisResult | None:
    """Execute supported Python analysis families with explicit gates."""

    if candidate.analysis_family == "penalized_predictive_models":
        return _predictive_result(
            candidate,
            rows,
            factor_entries,
        )
    if candidate.analysis_family == "curve_feature_clustering":
        return _clustering_result(candidate, rows)
    if candidate.analysis_family == "dataset_and_source_robustness":
        return _robustness_result(candidate, rows)
    return None


__all__ = ["AdvancedAnalysisResult", "execute_advanced_candidate"]
