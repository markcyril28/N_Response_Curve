from __future__ import annotations

import hashlib
from pathlib import Path
import re
import textwrap
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt

from n_response_curve.analysis.models import ModelAttempt
from n_response_curve.analysis.values import finite_number
from n_response_curve.contracts import SUPPORTED_FIGURE_FORMATS
from n_response_curve.data.curate import _KNOWN_TREATMENT_LOOKUP_CLASSES


_SAFE_FILENAME_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_SERIES_FILENAME_TOKEN_BYTES = 160
_SAFE_DIRECTORY_TOKEN = re.compile(r"^[A-Za-z0-9_-]+$")
_UNSAFE_DIRECTORY_TOKEN = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_DIRECTORY_TOKEN_BYTES = 80
_WINDOWS_RESERVED_BASENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)


def sanitize_figure_directory_token(value: str, *, fallback: str) -> str:
    """Return a deterministic portable directory token for figure grouping."""

    normalized = str(value).strip()
    if not normalized:
        raise ValueError("Figure directory identity must be nonempty")
    token = _UNSAFE_DIRECTORY_TOKEN.sub("_", normalized).strip("_") or fallback
    safe_unchanged = (
        token == normalized
        and _SAFE_DIRECTORY_TOKEN.fullmatch(token) is not None
        and token.upper() not in _WINDOWS_RESERVED_BASENAMES
        and len(token.encode("utf-8")) <= _MAX_DIRECTORY_TOKEN_BYTES
    )
    if safe_unchanged:
        return token
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    suffix = f"__{digest}"
    byte_budget = _MAX_DIRECTORY_TOKEN_BYTES - len(suffix.encode("ascii"))
    prefix = token.encode("utf-8")[:byte_budget].decode(
        "utf-8", errors="ignore"
    ).rstrip("_-")
    return f"{prefix or fallback}{suffix}"


def sanitize_series_filename(response_series_uid: str) -> str:
    """Return a deterministic portable filename token without changing the source ID."""

    token = _SAFE_FILENAME_TOKEN.sub("_", response_series_uid).strip("._")
    token = token or "response_series"
    encoded = token.encode("utf-8")
    if len(encoded) <= _MAX_SERIES_FILENAME_TOKEN_BYTES:
        return token
    digest = hashlib.sha256(response_series_uid.encode("utf-8")).hexdigest()[:16]
    suffix = f"__{digest}"
    byte_budget = _MAX_SERIES_FILENAME_TOKEN_BYTES - len(suffix.encode("ascii"))
    prefix = encoded[:byte_budget].decode("utf-8", errors="ignore").rstrip("._-")
    return f"{prefix or 'response_series'}{suffix}"


def _validated_predictions(attempt: ModelAttempt) -> tuple[Mapping[str, float], ...]:
    if attempt.observed_n_min_kg_ha is None or attempt.observed_n_max_kg_ha is None:
        raise ValueError("A curve plot requires observed N-domain bounds")
    lower = attempt.observed_n_min_kg_ha
    upper = attempt.observed_n_max_kg_ha
    tolerance = max(abs(upper - lower) * 1e-9, 1e-12)
    predictions = tuple(attempt.predictions)
    for row in predictions:
        n_rate = finite_number(row.get("n_rate_kg_ha"))
        yield_value = finite_number(row.get("predicted_yield_t_ha"))
        if n_rate is None or yield_value is None:
            raise ValueError("Curve predictions must be finite numeric values")
        if n_rate < lower - tolerance or n_rate > upper + tolerance:
            raise ValueError("Curve predictions must remain within the observed N domain")
    return predictions


def _observations(records: Iterable[Mapping[str, Any]], response_series_uid: str) -> tuple[dict[str, Any], ...]:
    observations: list[dict[str, Any]] = []
    for record in records:
        if record.get("response_series_uid") != response_series_uid:
            continue
        n_rate = finite_number(record.get("n_rate_kg_ha"))
        yield_value = finite_number(record.get("yield_t_ha"))
        if n_rate is None or yield_value is None:
            continue
        observations.append(
            {
                "record_uid": str(record.get("record_uid", "")),
                "n_rate_kg_ha": n_rate,
                "yield_t_ha": yield_value,
                "treatment_text_class": str(record.get("treatment_text_class", "unresolved")),
            }
        )
    return tuple(sorted(observations, key=lambda row: (row["n_rate_kg_ha"], row["record_uid"])))


_TREATMENT_CLASS_COLORS = {
    category: f"C{index}"
    for index, category in enumerate(sorted(_KNOWN_TREATMENT_LOOKUP_CLASSES))
}


def _plot_observations(axes: Any, observations: Sequence[Mapping[str, Any]]) -> None:
    categories = sorted({str(row["treatment_text_class"]) for row in observations})
    for category in categories:
        category_rows = [row for row in observations if row["treatment_text_class"] == category]
        observation_label = (
            "observed (treatment class unresolved)"
            if category.casefold() == "unresolved"
            else f"observed {category}"
        )
        axes.scatter(
            [row["n_rate_kg_ha"] for row in category_rows],
            [row["yield_t_ha"] for row in category_rows],
            label=observation_label,
            color=_TREATMENT_CLASS_COLORS[category],
            zorder=3,
        )


def _mark_observed_domain(axes: Any, lower: float, upper: float, *, mark_equal_upper: bool) -> None:
    axes.axvline(lower, color="grey", linestyle="--", linewidth=1, label="observed N bounds")
    if mark_equal_upper or upper != lower:
        axes.axvline(upper, color="grey", linestyle="--", linewidth=1)


def _wrapped_plot_text(lines: Iterable[str], *, width: int = 84) -> str:
    """Wrap unbounded source identifiers and reason ledgers for stable layouts."""

    wrapped: list[str] = []
    for line in lines:
        parts = textwrap.wrap(
            str(line),
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        )
        wrapped.extend(parts or [""])
    return "\n".join(wrapped)


def _finalize_axes(axes: Any, title_lines: Iterable[str]) -> None:
    axes.set_xlabel("Applied N (kg N/ha)")
    axes.set_ylabel("Grain yield (t/ha)")
    axes.set_title(_wrapped_plot_text(title_lines))
    axes.legend(loc="best", fontsize=8)


def _display_number(value: Any) -> str | None:
    number = finite_number(value)
    return None if number is None else f"{number:.3g}"


def _evidence_annotation(
    evidence_row: Mapping[str, Any] | None,
    *,
    attempt: ModelAttempt | None = None,
) -> tuple[str, ...]:
    if evidence_row is None:
        if attempt is None or attempt.status != "fitted":
            return ()
        optimum = f"optimum={attempt.optimum_status}"
        if attempt.agronomic_optimum_n_kg_ha is not None:
            optimum += f" ({attempt.agronomic_optimum_n_kg_ha:.2f} kg N/ha)"
        return (optimum,)
    status = str(evidence_row.get("evidence_status") or "unreported")
    strength = str(evidence_row.get("evidence_strength") or "unreported")
    lines = [f"evidence={status}; strength={strength}"]
    shape = str(evidence_row.get("curve_shape_class") or "unavailable")
    optimum_status = str(evidence_row.get("optimum_status") or "unavailable")
    optimum_value = _display_number(evidence_row.get("agronomic_optimum_n_kg_ha"))
    optimum = f"shape={shape}; optimum={optimum_status}"
    if optimum_value is not None:
        optimum += f" ({optimum_value} kg N/ha)"
    lines.append(optimum)
    for field, label in (
        ("predicted_observed_domain_peak_yield_t_ha", "observed-domain fitted peak"),
        ("finite_maximum_yield_t_ha", "finite maximum"),
        ("fitted_asymptote_yield_t_ha", "fitted asymptote"),
    ):
        value = _display_number(evidence_row.get(field))
        if value is not None:
            lines.append(f"{label}={value} t/ha")
    supported = _display_number(evidence_row.get("supported_max_yield_t_ha"))
    basis = str(evidence_row.get("maximum_reference_basis") or "none")
    if supported is not None:
        lines.append(f"supported maximum={supported} t/ha; basis={basis}")
    proximity = str(evidence_row.get("maximum_proximity_status") or "unavailable")
    proximity_parts = [f"maximum proximity={proximity}"]
    for field, label in (
        ("observed_max_gap_to_finite_maximum_t_ha", "gap to finite maximum"),
        ("observed_max_gap_to_supported_maximum_t_ha", "gap to supported maximum"),
        ("observed_max_attainment_fraction", "attainment fraction"),
    ):
        value = _display_number(evidence_row.get(field))
        if value is not None:
            proximity_parts.append(f"{label}={value}")
    lines.append("; ".join(proximity_parts))
    asymptote_rate = _display_number(
        evidence_row.get("asymptote_fraction_n_kg_ha")
    )
    if asymptote_rate is not None:
        rate_label = str(
            evidence_row.get("asymptote_rate_label")
            or "N at q% of asymptote"
        )
        fraction = _display_number(evidence_row.get("asymptote_fraction"))
        standard_error = _display_number(
            evidence_row.get("asymptote_fraction_n_se_kg_ha")
        )
        lines.append(f"{rate_label}={asymptote_rate} kg N/ha")
        details = []
        if fraction is not None:
            details.append(f"q={fraction}")
        if standard_error is not None:
            details.append(f"SE={standard_error} kg N/ha")
        if details:
            lines.append("; ".join(details))
    lines.append(f"target yield={evidence_row.get('target_yield_status') or 'not_configured'}")
    raw_reasons = evidence_row.get("reason_codes", ())
    if isinstance(raw_reasons, str):
        reasons = raw_reasons.strip()
    elif isinstance(raw_reasons, (list, tuple, set, frozenset)):
        reasons = ",".join(str(reason) for reason in raw_reasons if str(reason).strip())
    else:
        reasons = ""
    if reasons:
        lines.append(f"reasons={reasons}")
    return tuple(lines)


def create_observed_series_figure(
    records: Iterable[Mapping[str, Any]],
    response_series_uid: str,
    *,
    evidence_row: Mapping[str, Any] | None = None,
):
    """Build an observed-only figure even when no model is supportable."""

    if not isinstance(response_series_uid, str) or not response_series_uid:
        raise ValueError("An observed-series plot requires a nonempty response series ID")
    observations = _observations(records, response_series_uid)
    if not observations:
        raise ValueError("An observed-series plot requires at least one finite N/yield pair")
    figure, axes = plt.subplots(figsize=(10, 7), constrained_layout=True)
    _plot_observations(axes, observations)
    axes.plot(
        [row["n_rate_kg_ha"] for row in observations],
        [row["yield_t_ha"] for row in observations],
        label="observed connecting line (visual aid)",
        color="grey",
        linewidth=1,
        alpha=0.65,
        zorder=1,
    )
    n_values = [row["n_rate_kg_ha"] for row in observations]
    lower, upper = min(n_values), max(n_values)
    _mark_observed_domain(axes, lower, upper, mark_equal_upper=False)
    _finalize_axes(
        axes,
        (
            f"series={response_series_uid}",
            "observed only",
            f"distinct N={len(set(n_values))}",
            f"N range={lower:g}-{upper:g} kg/ha",
        ),
    )
    annotation = _evidence_annotation(evidence_row)
    if annotation:
        axes.text(
            0.01,
            0.01,
            _wrapped_plot_text(annotation),
            transform=axes.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
        )
    return figure, axes


def create_source_series_overlay_figure(
    records: Iterable[Mapping[str, Any]],
    source_name: str,
    *,
    response_series_uids: Sequence[str],
    display_name: str | None = None,
):
    """Overlay governed observed series without pooling or cross-series inference.

    ``display_name`` renames the source only in the drawn title. It defaults to
    ``source_name``, so release-path callers that omit it render exactly as
    before; ``source_name`` remains the value that selects rows.
    """

    if not isinstance(source_name, str) or not source_name.strip():
        raise ValueError("A source-series overlay requires a nonempty source name")
    selected_series = tuple(sorted(set(response_series_uids)))
    if (
        not selected_series
        or any(not isinstance(series_uid, str) or not series_uid for series_uid in selected_series)
        or len(selected_series) != len(tuple(response_series_uids))
    ):
        raise ValueError("A source-series overlay requires unique nonempty response series IDs")

    observations_by_series: dict[str, list[dict[str, Any]]] = {
        series_uid: [] for series_uid in selected_series
    }
    for record in records:
        series_uid = record.get("response_series_uid")
        if series_uid not in observations_by_series:
            continue
        if str(record.get("source_name") or "").strip() != source_name:
            raise ValueError(
                "Every selected response series row must belong to the requested overlay source"
            )
        n_rate = finite_number(record.get("n_rate_kg_ha"))
        yield_value = finite_number(record.get("yield_t_ha"))
        if n_rate is None or yield_value is None:
            continue
        observations_by_series[str(series_uid)].append(
            {
                "record_uid": str(record.get("record_uid", "")),
                "n_rate_kg_ha": n_rate,
                "yield_t_ha": yield_value,
                "treatment_text_class": str(
                    record.get("treatment_text_class", "unresolved")
                ),
            }
        )
    missing_series = [
        series_uid
        for series_uid, observations in observations_by_series.items()
        if not observations
    ]
    if missing_series:
        raise ValueError(
            "A source-series overlay requires at least one finite N/yield pair for every "
            "selected series"
        )
    for observations in observations_by_series.values():
        observations.sort(
            key=lambda row: (row["n_rate_kg_ha"], row["record_uid"])
        )

    figure, axes = plt.subplots(figsize=(10, 7), constrained_layout=True)
    combined_observations = tuple(
        observation
        for series_uid in selected_series
        for observation in observations_by_series[series_uid]
    )
    _plot_observations(axes, combined_observations)
    line_label = "within-series connecting lines (visual aid; not a fit)"
    for index, series_uid in enumerate(selected_series):
        observations = observations_by_series[series_uid]
        axes.plot(
            [row["n_rate_kg_ha"] for row in observations],
            [row["yield_t_ha"] for row in observations],
            label=line_label if index == 0 else "_nolegend_",
            color="grey",
            linewidth=1,
            alpha=0.35,
            zorder=1,
        )
    n_values = [row["n_rate_kg_ha"] for row in combined_observations]
    _finalize_axes(
        axes,
        (
            f"source={display_name or source_name}",
            "observed-series overlay (descriptive; no pooled curve or fit)",
            f"series={len(selected_series)}; observations={len(combined_observations)}",
            f"N range={min(n_values):g}-{max(n_values):g} kg/ha",
        ),
    )
    return figure, axes


def create_response_curve_figure(
    records: Iterable[Mapping[str, Any]],
    attempt: ModelAttempt,
    *,
    evidence_row: Mapping[str, Any] | None = None,
):
    """Build a structural, observed-domain curve figure without writing it to disk."""

    observations = _observations(records, attempt.response_series_uid)
    if not observations:
        raise ValueError("A curve plot requires at least one finite observation for its response series")
    predictions = _validated_predictions(attempt)
    figure, axes = plt.subplots(figsize=(10, 7), constrained_layout=True)
    _plot_observations(axes, observations)
    if attempt.status == "fitted" and predictions:
        axes.plot(
            [row["n_rate_kg_ha"] for row in predictions],
            [row["predicted_yield_t_ha"] for row in predictions],
            label=f"fitted {attempt.model_name}",
            color="black",
            zorder=2,
        )
    assert attempt.observed_n_min_kg_ha is not None
    assert attempt.observed_n_max_kg_ha is not None
    _mark_observed_domain(
        axes,
        attempt.observed_n_min_kg_ha,
        attempt.observed_n_max_kg_ha,
        mark_equal_upper=True,
    )
    _finalize_axes(
        axes,
        (
            f"series={attempt.response_series_uid}",
            f"model={attempt.model_name}",
            f"status={attempt.status}",
            f"distinct N={attempt.distinct_n_level_count}",
        ),
    )
    annotation = _evidence_annotation(evidence_row, attempt=attempt)
    if annotation:
        axes.text(
            0.01,
            0.01,
            _wrapped_plot_text(annotation),
            transform=axes.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
        )
    return figure, axes


def prediction_rows(attempt: ModelAttempt) -> tuple[dict[str, Any], ...]:
    """Expose machine-readable prediction rows linked to both series and model attempt."""

    rows: list[dict[str, Any]] = []
    for prediction in _validated_predictions(attempt):
        row = {
                "response_series_uid": attempt.response_series_uid,
                "model_attempt_uid": attempt.model_attempt_uid,
                "model_name": attempt.model_name,
                "model_status": attempt.status,
                "uncertainty_status": attempt.uncertainty_status,
                "uncertainty_method": attempt.uncertainty_method,
                "uncertainty_evidence_basis": attempt.uncertainty_evidence_basis,
                "n_rate_kg_ha": float(prediction["n_rate_kg_ha"]),
                "predicted_yield_t_ha": float(prediction["predicted_yield_t_ha"]),
            }
        for field in (
            "confidence_lower_95pct_t_ha",
            "confidence_upper_95pct_t_ha",
            "fitted_mean_se_t_ha",
            "confidence_level",
        ):
            if field in prediction:
                row[field] = float(prediction[field])
        rows.append(row)
    return tuple(rows)


def model_attempt_display_rows(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Apply Plan Section 10.2's display rules to emitted model-attempt records.

    Two diagnostics need a display rule that no per-candidate computation can
    supply, because both are properties of the *set* a reader compares.

    **AICc** (STAT-009). A three-mean-parameter candidate reaches `k = 4` once
    residual variance is counted, so AICc needs six distinct N levels, while
    `linear` reaches `k = 3` and qualifies from five. At exactly five levels the
    column is therefore populated for `linear` alone, and a single populated cell
    among unavailable ones reads as the candidate that passed a comparison no
    other candidate was eligible for. The column is suppressed entirely below two
    available candidates, with the reason recorded rather than the field
    silently emptied.

    **Leave-one-N-level-out** (PRF-014). Folds are dropped per candidate, by the
    roster gate or as boundary folds, so two candidates on one series can carry
    values computed from different folds. Those are different quantities — the
    statistic conditions on which predictions were attempted, exactly as AICc is
    comparable only across identical response records — so the fold basis travels
    with every value and unequal fold sets are marked not comparable rather than
    juxtaposed as though they measured the same thing.

    Neither diagnostic selects a model under `MOD-02`, so a suppressed value
    removes a display rather than a decision rule.
    """

    rows = [dict(record) for record in records]
    by_series: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_series.setdefault(str(row.get("response_series_uid") or ""), []).append(row)
    for series_rows in by_series.values():
        with_aicc = [row for row in series_rows if row.get("aicc") is not None]
        if len(with_aicc) >= 2:
            aicc_status = "available_comparable_candidate_set"
        elif len(with_aicc) == 1:
            aicc_status = "suppressed_single_available_candidate"
        else:
            aicc_status = "unavailable_no_candidate"
        fold_signatures = {
            (
                str(row.get("grouped_prediction_basis") or ""),
                row.get("grouped_prediction_fold_count"),
            )
            for row in series_rows
            if row.get("grouped_prediction_rmse") is not None
        }
        if len(fold_signatures) > 1:
            fold_status = "not_comparable_unequal_fold_sets"
        elif len(fold_signatures) == 1:
            fold_status = "comparable_common_fold_set"
        else:
            fold_status = "unavailable_no_candidate"
        for row in series_rows:
            if aicc_status == "suppressed_single_available_candidate":
                row["aicc"] = None
            row["aicc_display_status"] = aicc_status
            row["grouped_prediction_comparability_status"] = fold_status
    return tuple(rows)


def _normalize_figure_formats(formats: Sequence[str], *, figure_type: str) -> tuple[str, ...]:
    normalized = tuple(str(item).lower().lstrip(".") for item in formats)
    if not normalized or any(item not in SUPPORTED_FIGURE_FORMATS for item in normalized):
        supported = ", ".join(sorted(SUPPORTED_FIGURE_FORMATS))
        raise ValueError(f"{figure_type} figures support one or more of: {supported}")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{figure_type} figure formats must be unique")
    return normalized


def _write_figure(figure: Any, *, root: Path, stem: str, formats: Sequence[str]) -> tuple[Path, ...]:
    try:
        paths: list[Path] = []
        for output_format in formats:
            destination = root / f"{stem}.{output_format}"
            figure.savefig(destination, format=output_format, dpi=150)
            paths.append(destination)
        return tuple(paths)
    finally:
        plt.close(figure)


def write_response_curve_figures(
    records: Iterable[Mapping[str, Any]],
    attempt: ModelAttempt,
    *,
    output_root: str | Path,
    formats: Sequence[str],
    evidence_row: Mapping[str, Any] | None = None,
) -> tuple[Path, ...]:
    """Write one deterministic figure per requested format, after domain validation."""

    normalized_formats = _normalize_figure_formats(formats, figure_type="Curve")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    figure, _ = create_response_curve_figure(records, attempt, evidence_row=evidence_row)
    stem = f"{sanitize_series_filename(attempt.response_series_uid)}__{attempt.model_attempt_uid}"
    return _write_figure(figure, root=root, stem=stem, formats=normalized_formats)


def write_observed_series_figures(
    records: Iterable[Mapping[str, Any]],
    response_series_uid: str,
    *,
    output_root: str | Path,
    formats: Sequence[str],
    evidence_row: Mapping[str, Any] | None = None,
) -> tuple[Path, ...]:
    """Write deterministic observed-only figures without requiring a fit."""

    normalized_formats = _normalize_figure_formats(formats, figure_type="Observed-series")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    figure, _ = create_observed_series_figure(records, response_series_uid, evidence_row=evidence_row)
    stem = sanitize_series_filename(response_series_uid)
    return _write_figure(figure, root=root, stem=stem, formats=normalized_formats)


def write_source_series_overlay_figures(
    records: Iterable[Mapping[str, Any]],
    source_name: str,
    *,
    response_series_uids: Sequence[str],
    output_root: str | Path,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    """Write one additive observed-series overlay per requested format."""

    normalized_formats = _normalize_figure_formats(
        formats,
        figure_type="Source-series overlay",
    )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    figure, _ = create_source_series_overlay_figure(
        records,
        source_name,
        response_series_uids=response_series_uids,
    )
    stem = sanitize_figure_directory_token(source_name, fallback="source")
    return _write_figure(figure, root=root, stem=stem, formats=normalized_formats)


__all__ = [
    "create_observed_series_figure",
    "create_response_curve_figure",
    "create_source_series_overlay_figure",
    "model_attempt_display_rows",
    "prediction_rows",
    "sanitize_figure_directory_token",
    "sanitize_series_filename",
    "write_observed_series_figures",
    "write_response_curve_figures",
    "write_source_series_overlay_figures",
]
