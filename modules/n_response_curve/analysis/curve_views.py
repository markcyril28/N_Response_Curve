from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..data.provenance import stable_identifier, stable_json_sha256
from .analysis_matrix import SourceCombination
from .curve_evidence import CurveEvidenceResult, build_curve_evidence
from .dataset_versions import DatasetVersion, select_dataset_version_records


@dataclass(frozen=True)
class DerivedCurveView:
    """Curve evidence fitted for one exact dataset-version and source view."""

    view_id: str
    status: str
    reason_codes: tuple[str, ...]
    dataset_version_id: str
    dataset_version_membership_sha256: str
    source_combination_id: str
    source_families: tuple[str, ...]
    record_uids: tuple[str, ...]
    canonical_input_sha256: str
    model_policy_sha256: str
    evidence: CurveEvidenceResult
    model_attempt_records: tuple[dict[str, Any], ...]
    curve_rows: tuple[dict[str, Any], ...]
    prediction_rows: tuple[dict[str, Any], ...]


def _view_metadata(
    *,
    version: DatasetVersion,
    source_combination: SourceCombination,
    record_uids: Sequence[str],
    canonical_input_sha256: str,
    model_policy_sha256: str,
) -> dict[str, Any]:
    return {
        "dataset_version_id": version.version_id,
        "dataset_version_membership_sha256": version.membership_sha256,
        "source_combination_id": source_combination.combination_id,
        "source_families": list(source_combination.source_families),
        "record_uids": list(record_uids),
        "canonical_input_sha256": canonical_input_sha256,
        "model_policy_sha256": model_policy_sha256,
    }


def build_derived_curve_views(
    records: Iterable[Mapping[str, Any]],
    *,
    dataset_versions: Sequence[DatasetVersion],
    source_combinations: Sequence[SourceCombination],
    model_names: Sequence[str],
    policy: Mapping[str, Any],
) -> tuple[DerivedCurveView, ...]:
    """Refit response curves for every exact version/source membership.

    A fitted row from one view is never relabelled or reused in another view.  The
    view and model-attempt identities bind the membership, source subset, inputs,
    and model policy used to produce the evidence.
    """

    canonical_records = tuple(dict(record) for record in records)
    model_policy_sha256 = stable_json_sha256(dict(policy))
    views: list[DerivedCurveView] = []
    for version in dataset_versions:
        version_records = select_dataset_version_records(canonical_records, version)
        for source_combination in source_combinations:
            view_records = tuple(
                record
                for record in version_records
                if str(record.get("source_name", "")) in source_combination.source_families
            )
            view_records = tuple(sorted(view_records, key=lambda row: str(row.get("record_uid", ""))))
            record_uids = tuple(str(record["record_uid"]) for record in view_records)
            canonical_input_sha256 = stable_json_sha256(view_records)
            metadata = _view_metadata(
                version=version,
                source_combination=source_combination,
                record_uids=record_uids,
                canonical_input_sha256=canonical_input_sha256,
                model_policy_sha256=model_policy_sha256,
            )
            view_id = stable_identifier("curve_view", metadata)
            evidence = build_curve_evidence(
                view_records,
                model_names=model_names,
                policy=policy,
            )
            common_fields = {
                "accepted_model_view_id": view_id,
                "dataset_version_id": version.version_id,
                "dataset_version_membership_sha256": version.membership_sha256,
                "source_combination_id": source_combination.combination_id,
                "source_families": source_combination.source_families,
                "canonical_input_sha256": canonical_input_sha256,
                "model_policy_sha256": model_policy_sha256,
            }
            selected_ids = {attempt.model_attempt_uid for attempt in evidence.selected_attempts}
            attempt_records = tuple(
                {
                    **record,
                    **common_fields,
                    "derived_model_attempt_uid": stable_identifier(
                        "derived_attempt", (view_id, record["model_attempt_uid"])
                    ),
                    "selected_for_view": record["model_attempt_uid"] in selected_ids,
                }
                for record in evidence.model_attempt_records
            )
            curve_rows = tuple(
                {
                    **row,
                    **common_fields,
                    "derived_curve_row_uid": stable_identifier(
                        "derived_curve", (view_id, row["response_series_uid"])
                    ),
                }
                for row in evidence.curve_rows
            )
            prediction_records = tuple(
                {
                    **row,
                    **common_fields,
                    "derived_prediction_uid": stable_identifier(
                        "derived_prediction", (view_id, row, index)
                    ),
                }
                for index, row in enumerate(evidence.prediction_rows)
            )
            if version.status != "available":
                status = version.status
                reason_codes = version.reason_codes or ("DATASET_VERSION_UNAVAILABLE",)
            elif not view_records:
                status = "unsupported"
                reason_codes = ("NO_SOURCE_COMBINATION_RECORDS",)
            elif not curve_rows:
                status = "unsupported"
                reason_codes = ("NO_REPORTABLE_CURVE_EVIDENCE",)
            else:
                status = "available"
                reason_codes = ()
            views.append(
                DerivedCurveView(
                    view_id=view_id,
                    status=status,
                    reason_codes=tuple(reason_codes),
                    dataset_version_id=version.version_id,
                    dataset_version_membership_sha256=version.membership_sha256,
                    source_combination_id=source_combination.combination_id,
                    source_families=source_combination.source_families,
                    record_uids=record_uids,
                    canonical_input_sha256=canonical_input_sha256,
                    model_policy_sha256=model_policy_sha256,
                    evidence=evidence,
                    model_attempt_records=attempt_records,
                    curve_rows=curve_rows,
                    prediction_rows=prediction_records,
                )
            )
    views.sort(key=lambda view: (view.dataset_version_id, view.source_families))
    return tuple(views)


__all__ = ["DerivedCurveView", "build_derived_curve_views"]
