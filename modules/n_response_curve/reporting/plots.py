from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt

from n_response_curve.analysis.models import ModelAttempt
from n_response_curve.analysis.values import finite_number
from n_response_curve.contracts import SUPPORTED_FIGURE_FORMATS


_SAFE_FILENAME_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_series_filename(response_series_uid: str) -> str:
    """Return a deterministic portable filename token without changing the source ID."""

    token = _SAFE_FILENAME_TOKEN.sub("_", response_series_uid).strip("._")
    return token or "response_series"


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


def _plot_observations(axes: Any, observations: Sequence[Mapping[str, Any]]) -> None:
    categories = sorted({str(row["treatment_text_class"]) for row in observations})
    for category in categories:
        category_rows = [row for row in observations if row["treatment_text_class"] == category]
        axes.scatter(
            [row["n_rate_kg_ha"] for row in category_rows],
            [row["yield_t_ha"] for row in category_rows],
            label=f"observed {category}",
            zorder=3,
        )


def _mark_observed_domain(axes: Any, lower: float, upper: float, *, mark_equal_upper: bool) -> None:
    axes.axvline(lower, color="grey", linestyle="--", linewidth=1, label="observed N bounds")
    if mark_equal_upper or upper != lower:
        axes.axvline(upper, color="grey", linestyle="--", linewidth=1)


def _finalize_axes(axes: Any, title: str) -> None:
    axes.set_xlabel("kg N/ha")
    axes.set_ylabel("t/ha")
    axes.set_title(title)
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
    figure, axes = plt.subplots(figsize=(8, 5), constrained_layout=True)
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
        " | ".join(
            (
                f"series={response_series_uid}",
                "observed only",
                f"distinct N={len(set(n_values))}",
                f"N range={lower:g}-{upper:g} kg/ha",
            )
        ),
    )
    annotation = _evidence_annotation(evidence_row)
    if annotation:
        axes.text(
            0.01,
            0.01,
            "\n".join(annotation),
            transform=axes.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
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
    figure, axes = plt.subplots(figsize=(8, 5), constrained_layout=True)
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
        " | ".join(
            (
                f"series={attempt.response_series_uid}",
                f"model={attempt.model_name}",
                f"status={attempt.status}",
                f"distinct N={attempt.distinct_n_level_count}",
            )
        ),
    )
    annotation = _evidence_annotation(evidence_row, attempt=attempt)
    if annotation:
        axes.text(
            0.01,
            0.01,
            "\n".join(annotation),
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
        rows.append(
            {
                "response_series_uid": attempt.response_series_uid,
                "model_attempt_uid": attempt.model_attempt_uid,
                "model_name": attempt.model_name,
                "model_status": attempt.status,
                "n_rate_kg_ha": float(prediction["n_rate_kg_ha"]),
                "predicted_yield_t_ha": float(prediction["predicted_yield_t_ha"]),
            }
        )
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


__all__ = [
    "create_observed_series_figure",
    "create_response_curve_figure",
    "prediction_rows",
    "sanitize_series_filename",
    "write_observed_series_figures",
    "write_response_curve_figures",
]
