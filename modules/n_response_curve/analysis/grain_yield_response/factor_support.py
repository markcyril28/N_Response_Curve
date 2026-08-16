from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorSupportResults:
    support: pd.DataFrame
    level_summary: pd.DataFrame
    additive_screen: pd.DataFrame


def _fit_rss(
    design: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, int, int, np.ndarray]:
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("rank_deficient")
    residual = y - design @ coefficients
    rss = float(residual @ residual)
    tss = float((y - np.mean(y)) @ (y - np.mean(y)))
    return (
        rss,
        float(1.0 - rss / tss) if tss > 0 else math.nan,
        int(len(y) - rank),
        int(rank),
        coefficients,
    )


def _signed_equation_term(value: float, expression: str) -> str:
    sign = "+" if value >= 0 else "-"
    return f" {sign} {abs(value):.10g} * {expression}"


def _augmented_equation(
    coefficients: np.ndarray,
    *,
    factor_terms: tuple[str, ...],
) -> str:
    """Render `yield = a + b*N + …factor terms…` from a fitted design.

    The design is always [1, n_rate_kg_ha, *factor columns], so the first two
    coefficients are the intercept and the N slope.
    """

    equation = (
        f"yield_t_ha = {float(coefficients[0]):.10g}"
        + _signed_equation_term(float(coefficients[1]), "n_rate_kg_ha")
    )
    for value, term in zip(coefficients[2:], factor_terms):
        equation += _signed_equation_term(float(value), term)
    return equation


def _adjusted_r_squared(
    r_squared: float,
    *,
    observations: int,
    residual_df: int,
) -> float:
    if residual_df <= 0 or not math.isfinite(r_squared):
        return math.nan
    return float(1.0 - (1.0 - r_squared) * (observations - 1) / residual_df)


def _fitted_screen_row(
    *,
    factor: str,
    observations: int,
    base_rss: float,
    base_r_squared: float,
    base_residual_df: int,
    base_rank: int,
    augmented_rss: float,
    augmented_r_squared: float,
    augmented_residual_df: int,
    augmented_rank: int,
    equation: str = "",
    reference_level: str = "",
) -> dict[str, object]:
    base_adjusted = _adjusted_r_squared(
        base_r_squared,
        observations=observations,
        residual_df=base_residual_df,
    )
    augmented_adjusted = _adjusted_r_squared(
        augmented_r_squared,
        observations=observations,
        residual_df=augmented_residual_df,
    )
    partial_r_squared = (
        float(1.0 - augmented_rss / base_rss) if base_rss > 0 else math.nan
    )
    return {
        "factor": factor,
        "screen_status": (
            "fitted" if augmented_residual_df > 0 else "insufficient_residual_df"
        ),
        "observations": observations,
        "factor_parameters_added": augmented_rank - base_rank,
        "base_r_squared": base_r_squared,
        "augmented_r_squared": augmented_r_squared,
        "r_squared_improvement": float(augmented_r_squared - base_r_squared),
        "base_adjusted_r_squared": base_adjusted,
        "augmented_adjusted_r_squared": augmented_adjusted,
        "adjusted_r_squared_improvement": float(augmented_adjusted - base_adjusted),
        "partial_r_squared": partial_r_squared,
        "rss_reduction_fraction": partial_r_squared,
        "residual_df": augmented_residual_df,
        "equation": equation,
        "reference_level": reference_level,
        "comparability_warning": (
            "Complete-case baseline association; rows can differ in sample size and "
            "factor degrees of freedom and are not effect-size rankings."
        ),
        "analysis_role": "descriptive_baseline_screen_not_causal",
    }


# Vocabulary that records "no usable evidence" rather than a measured absence.
# Shared by `_clean_categorical` and `audit_factor_evidence` so the support
# audit and the evidence audit agree on what counts as an informative cell.
_NO_EVIDENCE_TOKENS = frozenset(
    {"", "nan", "none", "not stated", "not applicable", "n/a", "na"}
)


def _clean_categorical(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.mask(
        values.isna() | values.str.casefold().isin(_NO_EVIDENCE_TOKENS)
    )


def _additive_categorical_screen(
    complete: pd.DataFrame,
    *,
    factor: str,
) -> dict[str, object]:
    y = complete["yield_t_ha"].to_numpy(dtype=float)
    n_rate = complete["n_rate_kg_ha"].to_numpy(dtype=float)
    base_design = np.column_stack([np.ones(len(complete)), n_rate])
    base_rss, base_r_squared, base_residual_df, base_rank, _ = _fit_rss(
        base_design, y
    )
    all_levels = tuple(sorted(complete[factor].unique()))
    dummies = pd.get_dummies(complete[factor], drop_first=True, dtype=float)
    dummies = dummies.reindex(sorted(dummies.columns), axis=1)
    reference_level = str(all_levels[0]) if all_levels else ""
    augmented = np.column_stack([base_design, dummies.to_numpy(dtype=float)])
    try:
        (
            augmented_rss,
            augmented_r_squared,
            augmented_residual_df,
            augmented_rank,
            coefficients,
        ) = _fit_rss(augmented, y)
    except ValueError:
        return {
            "factor": factor,
            "screen_status": "rank_deficient",
            "observations": int(len(complete)),
            "factor_parameters_added": math.nan,
            "base_r_squared": base_r_squared,
            "augmented_r_squared": math.nan,
            "r_squared_improvement": math.nan,
            "base_adjusted_r_squared": _adjusted_r_squared(
                base_r_squared,
                observations=len(complete),
                residual_df=base_residual_df,
            ),
            "augmented_adjusted_r_squared": math.nan,
            "adjusted_r_squared_improvement": math.nan,
            "partial_r_squared": math.nan,
            "rss_reduction_fraction": math.nan,
            "residual_df": math.nan,
            "comparability_warning": "Rank-deficient additive screen.",
            "analysis_role": "descriptive_baseline_screen_not_causal",
        }
    return _fitted_screen_row(
        factor=factor,
        observations=len(complete),
        base_rss=base_rss,
        base_r_squared=base_r_squared,
        base_residual_df=base_residual_df,
        base_rank=base_rank,
        augmented_rss=augmented_rss,
        augmented_r_squared=augmented_r_squared,
        augmented_residual_df=augmented_residual_df,
        augmented_rank=augmented_rank,
        equation=_augmented_equation(
            coefficients,
            factor_terms=tuple(
                f"1[{factor}={column}]" for column in dummies.columns
            ),
        ),
        reference_level=reference_level,
    )


def _additive_numeric_screen(
    complete: pd.DataFrame,
    *,
    factor: str,
) -> dict[str, object]:
    y = complete["yield_t_ha"].to_numpy(dtype=float)
    n_rate = complete["n_rate_kg_ha"].to_numpy(dtype=float)
    factor_values = complete[factor].to_numpy(dtype=float)
    base_design = np.column_stack([np.ones(len(complete)), n_rate])
    base_rss, base_r_squared, base_residual_df, base_rank, _ = _fit_rss(
        base_design, y
    )
    augmented = np.column_stack([base_design, factor_values])
    try:
        (
            augmented_rss,
            augmented_r_squared,
            augmented_residual_df,
            augmented_rank,
            coefficients,
        ) = _fit_rss(augmented, y)
    except ValueError:
        return {
            "factor": factor,
            "screen_status": "rank_deficient",
            "observations": int(len(complete)),
            "factor_parameters_added": math.nan,
            "base_r_squared": base_r_squared,
            "augmented_r_squared": math.nan,
            "r_squared_improvement": math.nan,
            "base_adjusted_r_squared": _adjusted_r_squared(
                base_r_squared,
                observations=len(complete),
                residual_df=base_residual_df,
            ),
            "augmented_adjusted_r_squared": math.nan,
            "adjusted_r_squared_improvement": math.nan,
            "partial_r_squared": math.nan,
            "rss_reduction_fraction": math.nan,
            "residual_df": math.nan,
            "comparability_warning": "Rank-deficient additive screen.",
            "analysis_role": "descriptive_baseline_screen_not_causal",
        }
    return _fitted_screen_row(
        factor=factor,
        observations=len(complete),
        base_rss=base_rss,
        base_r_squared=base_r_squared,
        base_residual_df=base_residual_df,
        base_rank=base_rank,
        augmented_rss=augmented_rss,
        augmented_r_squared=augmented_r_squared,
        augmented_residual_df=augmented_residual_df,
        augmented_rank=augmented_rank,
        equation=_augmented_equation(coefficients, factor_terms=(factor,)),
    )


def analyze_factor_support(
    frame: pd.DataFrame,
    *,
    categorical_factors: tuple[str, ...],
    numeric_factors: tuple[str, ...],
    series_key: str,
    study_key: str,
    minimum_level_observations: int,
    minimum_modifier_series: int,
) -> FactorSupportResults:
    """Audit factor support and run only noncausal additive baseline screens."""

    support_rows: list[dict[str, object]] = []
    level_rows: list[dict[str, object]] = []
    screen_rows: list[dict[str, object]] = []
    total = len(frame)

    for factor in categorical_factors:
        values = _clean_categorical(frame[factor])
        complete_mask = values.notna()
        complete = frame.loc[complete_mask].copy()
        complete[factor] = values.loc[complete_mask].astype(str)
        counts = complete[factor].value_counts().sort_index()
        level_count = int(len(counts))
        within_series_variation = int(
            complete.groupby(series_key)[factor].nunique().gt(1).sum()
        ) if not complete.empty else 0
        if complete.empty:
            support_status = "unavailable"
        elif level_count < 2:
            support_status = "single_level"
        elif int(counts.min()) < minimum_level_observations:
            support_status = "sparse_levels"
        else:
            support_status = "descriptive_supported"
        modifier_status = (
            "eligible_for_prespecified_review_not_run"
            if support_status == "descriptive_supported"
            and within_series_variation >= minimum_modifier_series
            else "held_no_within_series_variation"
            if within_series_variation == 0
            else "held_insufficient_modifier_series"
        )
        level_study_counts = (
            complete.groupby(factor)[study_key].nunique(dropna=True)
            if not complete.empty
            else pd.Series(dtype=int)
        )
        if complete[study_key].nunique(dropna=True) < 2:
            study_alias_status = "single_study"
        elif not level_study_counts.empty and bool((level_study_counts == 1).all()):
            study_alias_status = "fully_nested_in_study"
        else:
            study_alias_status = "cross_study_support_present"
        support_rows.append(
            {
                "factor": factor,
                "factor_type": "categorical",
                "observations": int(complete_mask.sum()),
                "missing_observations": int(total - complete_mask.sum()),
                "level_count": level_count,
                "minimum_level_observations": int(counts.min()) if level_count else 0,
                "dominant_level_fraction": (
                    float(counts.max() / counts.sum()) if level_count else math.nan
                ),
                "represented_studies": int(complete[study_key].nunique(dropna=True)),
                "represented_series": int(complete[series_key].nunique(dropna=True)),
                "within_series_variation_count": within_series_variation,
                "study_alias_status": study_alias_status,
                "support_status": support_status,
                "modifier_status": modifier_status,
                "causal_claim_status": "not_supported",
            }
        )
        for level, group in complete.groupby(factor, sort=True):
            level_rows.append(
                {
                    "factor": factor,
                    "level": str(level),
                    "observations": int(len(group)),
                    "series_count": int(group[series_key].nunique(dropna=True)),
                    "study_count": int(group[study_key].nunique(dropna=True)),
                    "n_rate_mean_kg_ha": float(group["n_rate_kg_ha"].mean()),
                    "n_rate_min_kg_ha": float(group["n_rate_kg_ha"].min()),
                    "n_rate_max_kg_ha": float(group["n_rate_kg_ha"].max()),
                    "yield_mean_t_ha": float(group["yield_t_ha"].mean()),
                    "yield_sd_t_ha": (
                        float(group["yield_t_ha"].std(ddof=1))
                        if len(group) > 1
                        else math.nan
                    ),
                }
            )
        if support_status in {"descriptive_supported", "sparse_levels"}:
            screen_rows.append(
                _additive_categorical_screen(complete, factor=factor)
            )
        else:
            screen_rows.append(
                {
                    "factor": factor,
                    "screen_status": f"skipped_{support_status}",
                    "observations": int(len(complete)),
                    "base_r_squared": math.nan,
                    "augmented_r_squared": math.nan,
                    "r_squared_improvement": math.nan,
                    "rss_reduction_fraction": math.nan,
                    "residual_df": math.nan,
                    "analysis_role": "descriptive_baseline_screen_not_causal",
                }
            )

    for factor in numeric_factors:
        numeric = pd.to_numeric(frame[factor], errors="coerce")
        finite_mask = np.isfinite(numeric.to_numpy(dtype=float))
        complete = frame.loc[finite_mask].copy()
        complete[factor] = numeric.loc[finite_mask].astype(float)
        unique_count = int(complete[factor].nunique())
        within_series_variation = int(
            complete.groupby(series_key)[factor].nunique().gt(1).sum()
        ) if not complete.empty else 0
        if complete.empty:
            support_status = "unavailable"
        elif unique_count < 2:
            support_status = "constant_numeric"
        else:
            support_status = "descriptive_supported"
        modifier_status = (
            "eligible_for_prespecified_review_not_run"
            if support_status == "descriptive_supported"
            and within_series_variation >= minimum_modifier_series
            else "held_no_within_series_variation"
            if within_series_variation == 0
            else "held_insufficient_modifier_series"
        )
        support_rows.append(
            {
                "factor": factor,
                "factor_type": "numeric",
                "observations": int(finite_mask.sum()),
                "missing_observations": int(total - finite_mask.sum()),
                "level_count": unique_count,
                "minimum_level_observations": math.nan,
                "dominant_level_fraction": math.nan,
                "represented_studies": int(complete[study_key].nunique(dropna=True)),
                "represented_series": int(complete[series_key].nunique(dropna=True)),
                "within_series_variation_count": within_series_variation,
                "study_alias_status": (
                    "single_study"
                    if complete[study_key].nunique(dropna=True) < 2
                    else "not_assessed_for_numeric"
                ),
                "support_status": support_status,
                "modifier_status": modifier_status,
                "causal_claim_status": "not_supported",
                "numeric_min": float(complete[factor].min()) if not complete.empty else math.nan,
                "numeric_max": float(complete[factor].max()) if not complete.empty else math.nan,
            }
        )
        if support_status == "descriptive_supported":
            screen_rows.append(_additive_numeric_screen(complete, factor=factor))
        else:
            screen_rows.append(
                {
                    "factor": factor,
                    "screen_status": f"skipped_{support_status}",
                    "observations": int(len(complete)),
                    "base_r_squared": math.nan,
                    "augmented_r_squared": math.nan,
                    "r_squared_improvement": math.nan,
                    "rss_reduction_fraction": math.nan,
                    "residual_df": math.nan,
                    "analysis_role": "descriptive_baseline_screen_not_causal",
                }
            )

    support_frame = pd.DataFrame(support_rows)
    screen_frame = pd.DataFrame(screen_rows)
    if not screen_frame.empty:
        screen_frame = screen_frame.merge(
            support_frame[
                [
                    "factor",
                    "factor_type",
                    "support_status",
                    "modifier_status",
                    "within_series_variation_count",
                    "study_alias_status",
                ]
            ],
            on="factor",
            how="left",
            validate="one_to_one",
        )
    return FactorSupportResults(
        support=support_frame,
        level_summary=pd.DataFrame(level_rows),
        additive_screen=screen_frame,
    )


def _series_indicator_matrix(labels: pd.Series) -> np.ndarray:
    """Full-rank series indicator block, first level dropped."""

    dummies = pd.get_dummies(labels.astype(str), drop_first=True, dtype=float)
    return dummies.to_numpy(dtype=float)


def _factor_design_block(values: pd.Series, *, numeric: bool) -> np.ndarray:
    if numeric:
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
    dummies = pd.get_dummies(values.astype(str), drop_first=True, dtype=float)
    return dummies.to_numpy(dtype=float)


def screen_factors_beyond_series(
    frame: pd.DataFrame,
    *,
    categorical_factors: tuple[str, ...],
    numeric_factors: tuple[str, ...],
    series_key: str,
) -> pd.DataFrame:
    """Screen each factor against a baseline that already holds series identity.

    The plain additive screen in `analyze_factor_support` compares a factor with
    a pooled N-only line. Every series-level factor beats that baseline simply by
    standing in for series identity, so the improvement is not attributable to
    the factor. This screen puts series fixed intercepts in the *baseline*, which
    is the only comparison that can distinguish a factor from the series it
    labels. A factor with no within-series variation is a linear combination of
    the series indicators and is reported as `absorbed_by_series_intercepts`.
    """

    rows: list[dict[str, object]] = []
    specifications = [(factor, False) for factor in categorical_factors]
    specifications += [(factor, True) for factor in numeric_factors]

    for factor, numeric in specifications:
        if factor not in frame.columns:
            rows.append(
                {
                    "factor": factor,
                    "factor_type": "numeric" if numeric else "categorical",
                    "series_adjusted_status": "absent",
                    "observations": 0,
                    "analysis_role": "series_adjusted_screen_not_causal",
                }
            )
            continue
        if numeric:
            values = pd.to_numeric(frame[factor], errors="coerce")
            complete_mask = values.notna() & np.isfinite(
                values.to_numpy(dtype=float), where=values.notna().to_numpy(), out=np.zeros(len(values), dtype=bool)
            )
        else:
            values = _clean_categorical(frame[factor])
            complete_mask = values.notna()
        complete = frame.loc[complete_mask]
        observations = int(len(complete))
        within_series_variation = (
            int(
                complete.assign(_factor=values.loc[complete_mask].astype(str))
                .groupby(series_key)["_factor"]
                .nunique()
                .gt(1)
                .sum()
            )
            if observations
            else 0
        )
        base_row: dict[str, object] = {
            "factor": factor,
            "factor_type": "numeric" if numeric else "categorical",
            "observations": observations,
            "series_retained": (
                int(complete[series_key].nunique(dropna=True)) if observations else 0
            ),
            "within_series_variation_count": within_series_variation,
            "analysis_role": "series_adjusted_screen_not_causal",
            "interpretation": (
                "Improvement measured against a baseline that already contains N "
                "rate and one intercept per response series."
            ),
        }
        if observations < 4:
            rows.append(
                {
                    **base_row,
                    "series_adjusted_status": "insufficient_observations",
                }
            )
            continue

        y = complete["yield_t_ha"].to_numpy(dtype=float)
        n_rate = complete["n_rate_kg_ha"].to_numpy(dtype=float)
        series_block = _series_indicator_matrix(complete[series_key])
        baseline = np.column_stack(
            [np.ones(observations), n_rate, series_block]
            if series_block.size
            else [np.ones(observations), n_rate]
        )
        try:
            base_rss, base_r_squared, base_residual_df, base_rank, _ = _fit_rss(
                baseline, y
            )
        except ValueError:
            rows.append(
                {
                    **base_row,
                    "series_adjusted_status": "baseline_rank_deficient",
                }
            )
            continue
        if base_residual_df <= 0:
            rows.append(
                {
                    **base_row,
                    "series_adjusted_status": "baseline_saturated",
                    "series_adjusted_baseline_r_squared": base_r_squared,
                    "residual_df": base_residual_df,
                }
            )
            continue

        factor_block = _factor_design_block(
            values.loc[complete_mask], numeric=numeric
        )
        if factor_block.size == 0:
            rows.append({**base_row, "series_adjusted_status": "no_factor_columns"})
            continue
        augmented = np.column_stack([baseline, factor_block])
        try:
            (
                augmented_rss,
                augmented_r_squared,
                augmented_residual_df,
                augmented_rank,
                _,
            ) = _fit_rss(augmented, y)
        except ValueError:
            rows.append(
                {
                    **base_row,
                    "series_adjusted_status": "absorbed_by_series_intercepts",
                    "series_adjusted_baseline_r_squared": base_r_squared,
                    "series_adjusted_partial_r_squared": 0.0,
                    "residual_df": base_residual_df,
                    "absorption_note": (
                        "The factor is a linear combination of the response-series "
                        "indicators, so it carries no information beyond series "
                        "identity and no separate effect is estimable."
                    ),
                }
            )
            continue
        partial = float(1.0 - augmented_rss / base_rss) if base_rss > 0 else math.nan
        rows.append(
            {
                **base_row,
                "series_adjusted_status": (
                    "fitted" if augmented_residual_df > 0 else "insufficient_residual_df"
                ),
                "factor_parameters_added": augmented_rank - base_rank,
                "series_adjusted_baseline_r_squared": base_r_squared,
                "series_adjusted_augmented_r_squared": augmented_r_squared,
                "series_adjusted_partial_r_squared": partial,
                "series_adjusted_baseline_adjusted_r_squared": _adjusted_r_squared(
                    base_r_squared,
                    observations=observations,
                    residual_df=base_residual_df,
                ),
                "series_adjusted_augmented_adjusted_r_squared": _adjusted_r_squared(
                    augmented_r_squared,
                    observations=observations,
                    residual_df=augmented_residual_df,
                ),
                "residual_df": augmented_residual_df,
            }
        )
    return pd.DataFrame(rows)


def _cramers_v(first: pd.Series, second: pd.Series) -> tuple[float, bool]:
    table = pd.crosstab(first.astype(str), second.astype(str))
    counts = table.to_numpy(dtype=float)
    total = counts.sum()
    if total <= 0 or counts.shape[0] < 2 or counts.shape[1] < 2:
        return math.nan, False
    occupied = counts > 0
    one_to_one = bool(
        (occupied.sum(axis=1) == 1).all() and (occupied.sum(axis=0) == 1).all()
    )
    expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_square = float(
            np.nansum(np.where(expected > 0, (counts - expected) ** 2 / expected, 0.0))
        )
    denominator = total * (min(counts.shape) - 1)
    value = float(np.sqrt(chi_square / denominator)) if denominator > 0 else math.nan
    return (min(value, 1.0) if math.isfinite(value) else value), one_to_one


def _correlation_ratio(categories: pd.Series, values: np.ndarray) -> float:
    labels = categories.astype(str).to_numpy()
    grand_mean = float(np.mean(values))
    total = float(((values - grand_mean) ** 2).sum())
    if total <= 0:
        return math.nan
    between = 0.0
    for level in np.unique(labels):
        subset = values[labels == level]
        between += len(subset) * (float(np.mean(subset)) - grand_mean) ** 2
    return float(np.sqrt(between / total))


def audit_factor_redundancy(
    frame: pd.DataFrame,
    *,
    categorical_factors: tuple[str, ...],
    numeric_factors: tuple[str, ...],
    minimum_overlap: int = 4,
) -> pd.DataFrame:
    """Flag factor pairs that encode the same contrast under different names.

    Two perfectly aliased factors produce byte-identical screen rows, which reads
    as independent corroboration when it is one contrast counted twice. This
    audit makes the duplication explicit before the screen is interpreted.
    """

    specifications = [(factor, False) for factor in categorical_factors]
    specifications += [(factor, True) for factor in numeric_factors]
    rows: list[dict[str, object]] = []

    for index, (first, first_numeric) in enumerate(specifications):
        for second, second_numeric in specifications[index + 1 :]:
            if first not in frame.columns or second not in frame.columns:
                rows.append(
                    {
                        "factor_a": first,
                        "factor_b": second,
                        "redundancy_status": "absent",
                        "overlapping_observations": 0,
                        "analysis_role": "factor_redundancy_audit",
                    }
                )
                continue
            first_values = (
                pd.to_numeric(frame[first], errors="coerce")
                if first_numeric
                else _clean_categorical(frame[first])
            )
            second_values = (
                pd.to_numeric(frame[second], errors="coerce")
                if second_numeric
                else _clean_categorical(frame[second])
            )
            overlap = first_values.notna() & second_values.notna()
            count = int(overlap.sum())
            row: dict[str, object] = {
                "factor_a": first,
                "factor_b": second,
                "pair_kind": (
                    "numeric_numeric"
                    if first_numeric and second_numeric
                    else "categorical_categorical"
                    if not first_numeric and not second_numeric
                    else "mixed"
                ),
                "overlapping_observations": count,
                "association_measure": "",
                "association_value": math.nan,
                "redundancy_status": "insufficient_overlap",
                "analysis_role": "factor_redundancy_audit",
                "interpretation": "",
            }
            if count < minimum_overlap:
                rows.append(row)
                continue
            left = first_values.loc[overlap]
            right = second_values.loc[overlap]
            one_to_one = False
            if first_numeric and second_numeric:
                a = left.to_numpy(dtype=float)
                b = right.to_numpy(dtype=float)
                value = (
                    float(np.corrcoef(a, b)[0, 1])
                    if np.std(a) > 0 and np.std(b) > 0
                    else math.nan
                )
                row["association_measure"] = "pearson_r"
                row["association_value"] = value
                magnitude = abs(value) if math.isfinite(value) else math.nan
                # |r| = 1 is an invertible linear map: a genuine two-way alias.
                one_to_one = magnitude >= 0.9999 if math.isfinite(magnitude) else False
            elif not first_numeric and not second_numeric:
                value, one_to_one = _cramers_v(left, right)
                row["association_measure"] = "cramers_v"
                row["association_value"] = value
                row["levels_are_one_to_one"] = one_to_one
                magnitude = 1.0 if one_to_one else value
            else:
                categories = right if first_numeric else left
                numbers = (left if first_numeric else right).to_numpy(dtype=float)
                value = _correlation_ratio(categories, numbers)
                row["association_measure"] = "correlation_ratio_eta"
                row["association_value"] = value
                magnitude = value
            if not isinstance(magnitude, float) or not math.isfinite(magnitude):
                row["redundancy_status"] = "not_assessable"
            elif magnitude >= 0.9999 and one_to_one:
                row["redundancy_status"] = "perfect_alias"
                row["interpretation"] = (
                    "These factors partition the observations identically. Their "
                    "screen rows are the same contrast reported twice and must not "
                    "be read as separate evidence."
                )
            elif magnitude >= 0.9999:
                # Cramér's V and the correlation ratio reach 1 for nesting or
                # one-way determinism as well as for a two-way alias; keep the
                # two situations apart because a nested factor still carries
                # its own finer contrast.
                row["redundancy_status"] = "fully_nested_or_determined"
                row["interpretation"] = (
                    "One factor determines the other on these rows (nesting or "
                    "one-way determinism), so their screen rows overlap but are "
                    "not necessarily the same contrast."
                )
            elif magnitude >= 0.95:
                row["redundancy_status"] = "near_collinear"
                row["interpretation"] = (
                    "Nearly the same contrast; separate effects are not estimable."
                )
            elif magnitude >= 0.70:
                row["redundancy_status"] = "strongly_associated"
                row["interpretation"] = (
                    "Substantially overlapping information; treat screen rows as "
                    "correlated rather than independent."
                )
            else:
                row["redundancy_status"] = "distinct"
            rows.append(row)
    return pd.DataFrame(rows)


def audit_factor_evidence(
    frame: pd.DataFrame,
    *,
    evidence_pairs: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    """Compare a derived indicator with the raw column it claims to summarise.

    A derived boolean that reports a value on rows where the raw column carries
    no usable evidence is encoding metadata availability, not the agronomic
    condition it is named for. Any screen result for such a factor describes the
    recording process instead of the trial.
    """

    rows: list[dict[str, object]] = []
    for derived, raw in evidence_pairs:
        row: dict[str, object] = {
            "derived_factor": derived,
            "raw_evidence_column": raw,
            "observations": int(len(frame)),
            "analysis_role": "factor_evidence_audit",
            "evidence_status": "unavailable",
            "interpretation": "",
        }
        if derived not in frame.columns or raw not in frame.columns:
            row["evidence_status"] = "column_absent"
            rows.append(row)
            continue
        derived_values = _clean_categorical(frame[derived])
        raw_text = frame[raw].astype("string").str.strip()
        informative_raw = raw_text.notna() & ~raw_text.str.casefold().isin(
            _NO_EVIDENCE_TOKENS
        )
        asserted = derived_values.notna()
        positive = derived_values.astype("string").str.casefold().isin(
            {"true", "yes", "y", "1"}
        ) & asserted
        row.update(
            {
                "derived_nonmissing": int(asserted.sum()),
                "derived_positive": int(positive.sum()),
                "raw_informative": int(informative_raw.sum()),
                "raw_uninformative_or_missing": int((~informative_raw).sum()),
                "derived_asserted_without_raw_evidence": int(
                    (asserted & ~informative_raw).sum()
                ),
                "derived_positive_without_raw_evidence": int(
                    (positive & ~informative_raw).sum()
                ),
                "raw_distinct_informative_values": int(
                    raw_text.loc[informative_raw].nunique()
                ),
            }
        )
        if int(informative_raw.sum()) == 0:
            row["evidence_status"] = "no_raw_evidence"
            row["interpretation"] = (
                "The raw column carries no informative value anywhere in this "
                "population, so the derived indicator encodes recording state "
                "rather than an observed condition."
            )
        elif int((positive & ~informative_raw).sum()) > 0:
            row["evidence_status"] = "positive_without_raw_evidence"
            row["interpretation"] = (
                "The derived indicator asserts the condition on rows whose raw "
                "column is missing or explicitly uninformative."
            )
        elif int((asserted & ~informative_raw).sum()) > 0:
            row["evidence_status"] = "missing_coerced_to_negative"
            row["interpretation"] = (
                "Rows without raw evidence are carried as a definite level, so "
                "absence of information is being read as absence of the condition."
            )
        else:
            row["evidence_status"] = "raw_evidence_complete"
        rows.append(row)
    return pd.DataFrame(rows)
