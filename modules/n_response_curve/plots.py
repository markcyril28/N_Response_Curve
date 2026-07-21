from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt

from .models import ModelAttempt


_SAFE_FILENAME_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_series_filename(response_series_uid: str) -> str:
    """Return a deterministic portable filename token without changing the source ID."""

    token = _SAFE_FILENAME_TOKEN.sub("_", response_series_uid).strip("._")
    return token or "response_series"


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _validated_predictions(attempt: ModelAttempt) -> tuple[Mapping[str, float], ...]:
    if attempt.observed_n_min_kg_ha is None or attempt.observed_n_max_kg_ha is None:
        raise ValueError("A curve plot requires observed N-domain bounds")
    lower = attempt.observed_n_min_kg_ha
    upper = attempt.observed_n_max_kg_ha
    tolerance = max(abs(upper - lower) * 1e-9, 1e-12)
    predictions = tuple(attempt.predictions)
    for row in predictions:
        n_rate = _finite_number(row.get("n_rate_kg_ha"))
        yield_value = _finite_number(row.get("predicted_yield_t_ha"))
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
        n_rate = _finite_number(record.get("n_rate_kg_ha"))
        yield_value = _finite_number(record.get("yield_t_ha"))
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


def create_observed_series_figure(
    records: Iterable[Mapping[str, Any]],
    response_series_uid: str,
):
    """Build an observed-only figure even when no model is supportable."""

    if not isinstance(response_series_uid, str) or not response_series_uid:
        raise ValueError("An observed-series plot requires a nonempty response series ID")
    observations = _observations(records, response_series_uid)
    if not observations:
        raise ValueError("An observed-series plot requires at least one finite N/yield pair")
    figure, axes = plt.subplots(figsize=(8, 5), constrained_layout=True)
    categories = sorted({row["treatment_text_class"] for row in observations})
    for category in categories:
        category_rows = [row for row in observations if row["treatment_text_class"] == category]
        axes.scatter(
            [row["n_rate_kg_ha"] for row in category_rows],
            [row["yield_t_ha"] for row in category_rows],
            label=f"observed {category}",
            zorder=3,
        )
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
    axes.axvline(lower, color="grey", linestyle="--", linewidth=1, label="observed N bounds")
    if upper != lower:
        axes.axvline(upper, color="grey", linestyle="--", linewidth=1)
    axes.set_xlabel("kg N/ha")
    axes.set_ylabel("t/ha")
    axes.set_title(
        " | ".join(
            (
                f"series={response_series_uid}",
                "observed only",
                f"distinct N={len(set(n_values))}",
                f"N range={lower:g}-{upper:g} kg/ha",
            )
        )
    )
    axes.legend(loc="best", fontsize=8)
    return figure, axes


def create_response_curve_figure(
    records: Iterable[Mapping[str, Any]],
    attempt: ModelAttempt,
):
    """Build a structural, observed-domain curve figure without writing it to disk."""

    observations = _observations(records, attempt.response_series_uid)
    if not observations:
        raise ValueError("A curve plot requires at least one finite observation for its response series")
    predictions = _validated_predictions(attempt)
    figure, axes = plt.subplots(figsize=(8, 5), constrained_layout=True)
    categories = sorted({row["treatment_text_class"] for row in observations})
    for category in categories:
        category_rows = [row for row in observations if row["treatment_text_class"] == category]
        axes.scatter(
            [row["n_rate_kg_ha"] for row in category_rows],
            [row["yield_t_ha"] for row in category_rows],
            label=f"observed {category}",
            zorder=3,
        )
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
    axes.axvline(attempt.observed_n_min_kg_ha, color="grey", linestyle="--", linewidth=1, label="observed N bounds")
    axes.axvline(attempt.observed_n_max_kg_ha, color="grey", linestyle="--", linewidth=1)
    axes.set_xlabel("kg N/ha")
    axes.set_ylabel("t/ha")
    axes.set_title(
        " | ".join(
            (
                f"series={attempt.response_series_uid}",
                f"model={attempt.model_name}",
                f"status={attempt.status}",
                f"distinct N={attempt.distinct_n_level_count}",
            )
        )
    )
    if attempt.status == "fitted":
        annotation = f"optimum={attempt.optimum_status}"
        if attempt.agronomic_optimum_n_kg_ha is not None:
            annotation += f" ({attempt.agronomic_optimum_n_kg_ha:.2f} kg N/ha)"
        axes.text(0.01, 0.01, annotation, transform=axes.transAxes, va="bottom", ha="left", fontsize=8)
    axes.legend(loc="best", fontsize=8)
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


def write_response_curve_figures(
    records: Iterable[Mapping[str, Any]],
    attempt: ModelAttempt,
    *,
    output_root: str | Path,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    """Write one deterministic figure per requested format, after domain validation."""

    normalized_formats = tuple(str(item).lower().lstrip(".") for item in formats)
    if not normalized_formats or any(item not in {"png", "svg"} for item in normalized_formats):
        raise ValueError("Curve figures support one or more of: png, svg")
    if len(normalized_formats) != len(set(normalized_formats)):
        raise ValueError("Curve figure formats must be unique")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    figure, _ = create_response_curve_figure(records, attempt)
    try:
        stem = f"{sanitize_series_filename(attempt.response_series_uid)}__{attempt.model_attempt_uid}"
        paths: list[Path] = []
        for output_format in normalized_formats:
            destination = root / f"{stem}.{output_format}"
            figure.savefig(destination, format=output_format, dpi=150)
            paths.append(destination)
        return tuple(paths)
    finally:
        plt.close(figure)


def write_observed_series_figures(
    records: Iterable[Mapping[str, Any]],
    response_series_uid: str,
    *,
    output_root: str | Path,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    """Write deterministic observed-only figures without requiring a fit."""

    normalized_formats = tuple(str(item).lower().lstrip(".") for item in formats)
    if not normalized_formats or any(item not in {"png", "svg"} for item in normalized_formats):
        raise ValueError("Observed-series figures support one or more of: png, svg")
    if len(normalized_formats) != len(set(normalized_formats)):
        raise ValueError("Observed-series figure formats must be unique")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    figure, _ = create_observed_series_figure(records, response_series_uid)
    try:
        stem = sanitize_series_filename(response_series_uid)
        paths: list[Path] = []
        for output_format in normalized_formats:
            destination = root / f"{stem}.{output_format}"
            figure.savefig(destination, format=output_format, dpi=150)
            paths.append(destination)
        return tuple(paths)
    finally:
        plt.close(figure)


__all__ = [
    "create_observed_series_figure",
    "create_response_curve_figure",
    "prediction_rows",
    "sanitize_series_filename",
    "write_observed_series_figures",
    "write_response_curve_figures",
]
