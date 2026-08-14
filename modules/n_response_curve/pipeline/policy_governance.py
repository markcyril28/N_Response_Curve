from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

from n_response_curve.analysis.analysis_matrix import build_source_combinations
from n_response_curve.data.config import ConfigError, ValidatedConfig
from n_response_curve.data.provenance import sha256_file, stable_json_sha256


_POLICY_SNAPSHOT_SCHEMA_VERSION = 1
_AUTHORITY_MATRIX_SCHEMA_VERSION = 1
_REVIEW_GATE_POLICY_SCHEMA_VERSION = 1
_RELEASE_APPROVAL_SCHEMA_VERSION = 1
_APPROVED_STATUS = "APPROVED"
_SAFE_APPROVAL_TEXT = re.compile(r"^[^\x00-\x1f\x7f]+$")
_FATAL_REVIEW_ISSUE_STATES = ("structural", "unresolved", "warning")
_AUTHORITY_GATES = (
    "source_integrity",
    "restricted_data",
    "scientific_methods",
    "runtime_integrity",
    "release_promotion",
)
_AUTHORITY_POLICY_VALUES = MappingProxyType(
    {
        "role_combination_policy": "accountable_parties_must_be_distinct",
        "substitution_policy": "no_substitution",
        "recusal_policy": "matrix_approval_preclears_assigned_parties",
        "dual_approval_policy": "single_accountable_party_approval",
    }
)


@dataclass(frozen=True)
class ApprovalAuthorityMatrix:
    """Authenticated OPS-08 gate ownership; never inferred from a job title."""

    matrix_id: str
    effective_from: str
    approval: Mapping[str, str]
    gate_authorities: Mapping[str, Mapping[str, str]]
    role_combination_policy: str
    substitution_policy: str
    recusal_policy: str
    dual_approval_policy: str
    artifact_path: Path
    artifact_sha256: str

    def manifest_payload(self, *, project_root: Path) -> dict[str, Any]:
        try:
            relative_path = self.artifact_path.relative_to(project_root).as_posix()
        except ValueError:
            relative_path = None
        return {
            "matrix_id": self.matrix_id,
            "effective_from": self.effective_from,
            "approval": dict(self.approval),
            "gate_authorities": _json_value(self.gate_authorities),
            "role_combination_policy": self.role_combination_policy,
            "substitution_policy": self.substitution_policy,
            "recusal_policy": self.recusal_policy,
            "dual_approval_policy": self.dual_approval_policy,
            "artifact_path": relative_path,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True)
class ReviewGatePolicy:
    """Approved prospective treatment of fully resolved nonanalytical evidence."""

    policy_id: str
    prospective_effective_version: str
    effective_from: str
    approval: Mapping[str, str]
    fatal_issue_states: tuple[str, ...]
    permitted_resolved_dispositions: tuple[Mapping[str, str], ...]
    source_accountability_policy: str
    analytical_leakage_policy: str
    artifact_path: Path
    artifact_sha256: str

    def permits(self, issue: Mapping[str, Any]) -> bool:
        identity = {
            field: str(issue.get(field) or "")
            for field in ("stage", "issue_scope", "issue_state", "status")
        }
        return any(
            identity == dict(disposition)
            for disposition in self.permitted_resolved_dispositions
        )

    def manifest_payload(
        self,
        *,
        project_root: Path | None = None,
    ) -> dict[str, Any]:
        artifact_path: str | None = str(self.artifact_path)
        if project_root is not None:
            try:
                artifact_path = self.artifact_path.relative_to(project_root).as_posix()
            except ValueError:
                artifact_path = None
        return {
            "policy_id": self.policy_id,
            "prospective_effective_version": self.prospective_effective_version,
            "effective_from": self.effective_from,
            "approval": dict(self.approval),
            "fatal_issue_states": list(self.fatal_issue_states),
            "permitted_resolved_dispositions": [
                dict(disposition)
                for disposition in self.permitted_resolved_dispositions
            ],
            "source_accountability_policy": self.source_accountability_policy,
            "analytical_leakage_policy": self.analytical_leakage_policy,
            "artifact_path": artifact_path,
            "artifact_sha256": self.artifact_sha256,
        }

    def semantic_payload(self) -> dict[str, Any]:
        payload = self.manifest_payload()
        payload.pop("approval")
        payload.pop("artifact_path")
        return payload


@dataclass(frozen=True)
class ReleaseApproval:
    """Exact release-owner authorization for one target and run identity."""

    record_id: str
    run_id: str
    release_target: str
    run_identity_sha256: str
    authority_matrix_sha256: str
    approval: Mapping[str, str]
    artifact_path: Path
    artifact_sha256: str

    def manifest_payload(
        self,
        *,
        project_root: Path | None = None,
    ) -> dict[str, Any]:
        artifact_path: str | None = str(self.artifact_path)
        if project_root is not None:
            try:
                artifact_path = self.artifact_path.relative_to(project_root).as_posix()
            except ValueError:
                artifact_path = None
        return {
            "status": "approved",
            "record_id": self.record_id,
            "scope": "release_promotion",
            "run_id": self.run_id,
            "release_target": self.release_target,
            "run_identity_sha256": self.run_identity_sha256,
            "authority_matrix_sha256": self.authority_matrix_sha256,
            "approval": dict(self.approval),
            "artifact_path": artifact_path,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True)
class RuntimePolicySnapshot:
    """Validated semantic policy state for one requested runtime operation."""

    mode: str
    status: str
    policy_content_sha256: str
    effective_enablement_sha256: str
    effective_enablement: Mapping[str, Any]
    approval: Mapping[str, str] | None
    artifact_path: Path | None
    artifact_sha256: str | None
    authority_matrix: ApprovalAuthorityMatrix | None = None
    review_gate_policy: ReviewGatePolicy | None = None

    def manifest_payload(self, *, project_root: Path) -> dict[str, Any]:
        payload = {
            "mode": self.mode,
            "status": self.status,
            "policy_content_sha256": self.policy_content_sha256,
            "effective_enablement_sha256": self.effective_enablement_sha256,
            "effective_enablement": _json_value(self.effective_enablement),
        }
        if self.review_gate_policy is not None:
            payload["review_gate_policy"] = self.review_gate_policy.manifest_payload(
                project_root=project_root
            )
        return payload


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def effective_analysis_hypotheses(
    config: ValidatedConfig,
    analysis_policy: Any | None,
) -> tuple[Any, ...] | None:
    """Return the sole bounded hypothesis authority for this run."""

    configured = config.raw.get("analysis_hypotheses", {}).get(
        "specifications",
        [],
    )
    if analysis_policy is None:
        return tuple(configured)
    if configured:
        raise ConfigError(
            "The hash-bound analysis-policy bundle must be the sole hypothesis "
            "authority; remove analysis_hypotheses.specifications overrides"
        )
    source_view_ids: dict[str, str] = {}
    combinations = build_source_combinations(
        config.enabled_sources,
        modes=config.source_combination_modes,
    )
    for combination in combinations:
        source_view_ids[combination.combination_id] = combination.combination_id
    for mode in config.source_combination_modes:
        mode_combinations = build_source_combinations(
            config.enabled_sources,
            modes=(mode,),
        )
        if len(mode_combinations) == 1:
            source_view_ids[mode] = mode_combinations[0].combination_id
    return tuple(
        analysis_policy.runtime_hypothesis_specifications(
            source_view_ids=source_view_ids,
        )
    )


def _semantic_policy(
    review_gate_policy: ReviewGatePolicy | None = None,
    *,
    table_formats: tuple[str, ...],
) -> dict[str, Any]:
    policy = {
        "test_subset": {
            "selection": "deterministic_representative_greedy_coverage",
            "coverage_dimensions": [
                "eligibility_tier",
                "treatment_class",
                "source_type",
                "edge_case",
            ],
            "seed_source": "run.random_seed",
        },
        "review_gate": {
            "failure_modes": ["validate"],
            "ledger_modes": ["test", "full"],
            "failure_states": ["warning", "unresolved", "excluded_series"],
            "writing_mode_disposition": "complete_processing_with_findings_ledgered",
            "failure_timing": "after_complete_phase_2_review_before_later_phases",
        },
        "tables": {
            "required_formats": list(table_formats),
        },
        "figures": {
            "required_any_of": ["png", "jpeg"],
            "minimum_required_format_count": 1,
            "media_type": "raster",
            "forbidden_formats": ["svg"],
        },
        "reports": {
            "required_formats": ["md", "pdf"],
            "forbidden_formats": ["html"],
            "deterministic_pdf": True,
        },
        "replacement": {
            "control": "run.overwrite",
            "requires_named_target": True,
            "requires_verified_prior_package": True,
            "requires_technical_hash_binding": True,
            "requires_organizational_approval": False,
            "requires_validated_stage": True,
            "preserve_prior_package": True,
        },
        "logging": {
            "operational_level": "INFO",
            "complete_machine_readable_qc": True,
        },
    }
    if review_gate_policy is not None:
        policy["review_gate"] = {
            "failure_modes": ["validate"],
            "ledger_modes": ["test", "full"],
            **review_gate_policy.semantic_payload(),
        }
    return policy


def effective_enablement(config: ValidatedConfig) -> dict[str, Any]:
    """Return the complete operator-effective list state in canonical order."""

    return {
        "enabled_sources": list(config.enabled_sources),
        "enabled_models": list(config.enabled_models),
        "comparison_dimensions": list(config.comparison_dimensions),
        "scope_countries": list(config.scope_countries),
        "series_identity_dimensions": list(config.series_identity_dimensions),
        "table_formats": list(config.output_formats),
        "figure_formats": list(config.figure_formats),
        "document_formats": ["md", "pdf"],
        "dataset_versions": list(config.dataset_versions),
        "source_combination_modes": list(config.source_combination_modes),
        "curve_outcomes": list(config.curve_outcomes),
        "explanatory_factors": list(config.explanatory_factors),
        "analysis_families": list(config.analysis_families),
        "interaction_orders": list(config.interaction_orders),
        "engine_assignments": dict(sorted(config.engine_assignments.items())),
    }


def _policy_content(
    config: ValidatedConfig,
    review_gate_policy: ReviewGatePolicy | None = None,
    authority_matrix: ApprovalAuthorityMatrix | None = None,
) -> dict[str, Any]:
    enablement = effective_enablement(config)
    content = {
        "schema_version": _POLICY_SNAPSHOT_SCHEMA_VERSION,
        "mode": config.run_mode,
        "config_sha256": sha256_file(config.config_path),
        "semantic_policy": _semantic_policy(
            review_gate_policy,
            table_formats=config.output_formats,
        ),
        "effective_enablement": enablement,
        "effective_enablement_sha256": stable_json_sha256(enablement),
    }
    if authority_matrix is not None:
        content["approval_authority_matrix"] = {
            "matrix_id": authority_matrix.matrix_id,
            "effective_from": authority_matrix.effective_from,
            "artifact_sha256": authority_matrix.artifact_sha256,
        }
    return content


def _enabled_restricted_sources(config: ValidatedConfig) -> tuple[str, ...]:
    raw_sources = config.raw.get("sources")
    if not isinstance(raw_sources, Mapping):
        return ()
    return tuple(
        source_name
        for source_name in config.enabled_sources
        if isinstance(raw_sources.get(source_name), Mapping)
        and raw_sources[source_name].get("data_classification") == "restricted"
    )


def _authority_matrix_path(config: ValidatedConfig) -> Path:
    path = (
        config.paths["run_metadata_root"]
        / "approvals"
        / "authority_matrix.json"
    ).resolve()
    if not path.is_relative_to(config.project_root):
        raise ConfigError("Approval authority matrix path escapes the project root")
    return path


def _review_gate_policy_path(config: ValidatedConfig) -> Path:
    path = (
        config.paths["run_metadata_root"]
        / "approvals"
        / "review_gate_policy.json"
    ).resolve()
    if not path.is_relative_to(config.project_root):
        raise ConfigError("Review-gate policy path escapes the project root")
    return path


def _approval_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("Approved policy snapshot has an invalid approval timestamp")
    normalized = value.strip()
    try:
        if "T" in normalized:
            parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("timezone is required")
        else:
            date.fromisoformat(normalized)
    except ValueError as exc:
        raise ConfigError(
            "Approved policy snapshot approval timestamp must be an ISO date or timezone-qualified datetime"
        ) from exc
    return normalized


def _approval_calendar_date(value: str) -> date:
    if "T" in value:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    return date.fromisoformat(value)


def _approval_text(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or _SAFE_APPROVAL_TEXT.fullmatch(value.strip()) is None
    ):
        raise ConfigError(f"Approved policy snapshot has an invalid {field}")
    return value.strip()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(
            "The requested authoritative operation requires a standalone approved policy snapshot"
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError("Approved policy snapshot is unreadable or malformed") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("Approved policy snapshot must be a JSON object")
    return dict(payload)


def load_approval_authority_matrix(path: str | Path) -> ApprovalAuthorityMatrix:
    """Load the independently approved OPS-08 matrix with exact gate ownership."""

    artifact_path = Path(path).resolve()
    try:
        payload = _load_json_object(artifact_path)
    except ConfigError as exc:
        raise ConfigError(
            "Authoritative full mode requires a separately approved OPS-08 authority matrix"
        ) from exc
    required_keys = {
        "schema_version",
        "status",
        "matrix_id",
        "effective_from",
        "approved_by",
        "approved_at",
        "approval_source",
        "gate_authorities",
        "role_combination_policy",
        "substitution_policy",
        "recusal_policy",
        "dual_approval_policy",
    }
    if set(payload) != required_keys:
        raise ConfigError("Approval authority matrix fields do not match the required schema")
    if payload["schema_version"] != _AUTHORITY_MATRIX_SCHEMA_VERSION:
        raise ConfigError("Approval authority matrix schema version is unsupported")
    if payload["status"] != _APPROVED_STATUS:
        raise ConfigError("Approval authority matrix is not approved")
    matrix_id = _approval_text(payload["matrix_id"], field="matrix identifier")
    effective_from = _approval_timestamp(payload["effective_from"])
    approval = {
        "approved_by": _approval_text(payload["approved_by"], field="matrix approver"),
        "approved_at": _approval_timestamp(payload["approved_at"]),
        "approval_source": _approval_text(
            payload["approval_source"],
            field="matrix approval source",
        ),
    }
    effective_date = _approval_calendar_date(effective_from)
    approval_date = _approval_calendar_date(approval["approved_at"])
    if approval_date > effective_date:
        raise ConfigError(
            "Approval authority matrix cannot become effective before its approval date"
        )
    if effective_date > date.today():
        raise ConfigError("Approval authority matrix is not yet effective")
    raw_gates = payload["gate_authorities"]
    if not isinstance(raw_gates, Mapping) or set(raw_gates) != set(_AUTHORITY_GATES):
        raise ConfigError(
            "Approval authority matrix must assign every source/privacy, scientific, "
            "runtime, and release gate"
        )
    gate_authorities: dict[str, Mapping[str, str]] = {}
    required_gate_fields = {
        "accountable_role",
        "accountable_party",
        "authority_scope",
        "approval_source",
    }
    for gate in _AUTHORITY_GATES:
        raw_assignment = raw_gates[gate]
        if (
            not isinstance(raw_assignment, Mapping)
            or set(raw_assignment) != required_gate_fields
        ):
            raise ConfigError(
                f"Approval authority assignment for {gate!r} does not match the required schema"
            )
        assignment = {
            field: _approval_text(
                raw_assignment[field],
                field=f"{gate} {field}",
            )
            for field in sorted(required_gate_fields)
        }
        if assignment["authority_scope"] != gate:
            raise ConfigError(
                f"Approval authority assignment for {gate!r} has a mismatched scope"
            )
        gate_authorities[gate] = assignment
    policies = {
        field: _approval_text(payload[field], field=field.replace("_", " "))
        for field in (
            "role_combination_policy",
            "substitution_policy",
            "recusal_policy",
            "dual_approval_policy",
        )
    }
    if policies != dict(_AUTHORITY_POLICY_VALUES):
        raise ConfigError(
            "Approval authority matrix governance policies do not match the "
            "supported fail-closed policy values"
        )
    accountable_parties = [
        assignment["accountable_party"] for assignment in gate_authorities.values()
    ]
    if len(accountable_parties) != len(set(accountable_parties)):
        raise ConfigError(
            "Approval authority matrix requires distinct accountable parties for every gate"
        )
    return ApprovalAuthorityMatrix(
        matrix_id=matrix_id,
        effective_from=effective_from,
        approval=approval,
        gate_authorities=gate_authorities,
        artifact_path=artifact_path,
        artifact_sha256=sha256_file(artifact_path),
        **policies,
    )


def load_review_gate_policy(
    path: str | Path,
    *,
    authority_matrix: ApprovalAuthorityMatrix,
) -> ReviewGatePolicy:
    """Load the approved disposition-aware review policy without inferring states."""

    artifact_path = Path(path).resolve()
    payload = _load_json_object(artifact_path)
    required_keys = {
        "schema_version",
        "status",
        "policy_id",
        "prospective_effective_version",
        "effective_from",
        "approved_by",
        "approved_at",
        "approval_source",
        "fatal_issue_states",
        "permitted_resolved_dispositions",
        "source_accountability_policy",
        "analytical_leakage_policy",
    }
    if set(payload) != required_keys:
        raise ConfigError("Review-gate policy fields do not match the required schema")
    if payload["schema_version"] != _REVIEW_GATE_POLICY_SCHEMA_VERSION:
        raise ConfigError("Review-gate policy schema version is unsupported")
    if payload["status"] != _APPROVED_STATUS:
        raise ConfigError("Review-gate policy is not approved")
    approval = {
        "approved_by": _approval_text(
            payload["approved_by"],
            field="review-gate policy approver",
        ),
        "approved_at": _approval_timestamp(payload["approved_at"]),
        "approval_source": _approval_text(
            payload["approval_source"],
            field="review-gate policy approval source",
        ),
    }
    scientific_party = authority_matrix.gate_authorities["scientific_methods"][
        "accountable_party"
    ]
    if approval["approved_by"] != scientific_party:
        raise ConfigError(
            "Review-gate policy signer is not the accountable scientific-methods party "
            "in the OPS-08 authority matrix"
        )
    fatal_states = payload["fatal_issue_states"]
    if (
        not isinstance(fatal_states, list)
        or sorted(fatal_states) != list(_FATAL_REVIEW_ISSUE_STATES)
        or len(fatal_states) != len(set(fatal_states))
    ):
        raise ConfigError(
            "Review-gate policy must keep structural, unresolved, and warning states fatal"
        )
    raw_dispositions = payload["permitted_resolved_dispositions"]
    if not isinstance(raw_dispositions, list) or not raw_dispositions:
        raise ConfigError(
            "Review-gate policy must enumerate at least one exact permitted resolved disposition"
        )
    required_disposition_fields = {
        "stage",
        "issue_scope",
        "issue_state",
        "status",
    }
    permitted: list[Mapping[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for index, raw_disposition in enumerate(raw_dispositions):
        if (
            not isinstance(raw_disposition, Mapping)
            or set(raw_disposition) != required_disposition_fields
        ):
            raise ConfigError(
                f"Review-gate permitted disposition {index} does not match the required schema"
            )
        disposition = {
            field: _approval_text(
                raw_disposition[field],
                field=f"review-gate disposition {index} {field}",
            )
            for field in sorted(required_disposition_fields)
        }
        if disposition["stage"] not in {"phase_2", "phase_3", "phase_4"}:
            raise ConfigError("Review-gate permitted disposition has an invalid stage")
        if disposition["issue_state"] != "excluded_series":
            raise ConfigError(
                "Only exact resolved excluded-series dispositions may be permitted"
            )
        identity = (
            disposition["stage"],
            disposition["issue_scope"],
            disposition["issue_state"],
            disposition["status"],
        )
        if identity in seen:
            raise ConfigError("Review-gate permitted dispositions must be unique")
        seen.add(identity)
        permitted.append(disposition)
    effective_from = _approval_timestamp(payload["effective_from"])
    if _approval_calendar_date(approval["approved_at"]) < _approval_calendar_date(
        authority_matrix.effective_from
    ):
        raise ConfigError(
            "Review-gate policy approval predates the effective OPS-08 authority matrix"
        )
    if _approval_calendar_date(effective_from) < _approval_calendar_date(
        approval["approved_at"]
    ):
        raise ConfigError(
            "Review-gate policy cannot become effective before its approval date"
        )
    if _approval_calendar_date(effective_from) < _approval_calendar_date(
        authority_matrix.effective_from
    ):
        raise ConfigError(
            "Review-gate policy predates the effective OPS-08 authority matrix"
        )
    source_accountability_policy = _approval_text(
        payload["source_accountability_policy"],
        field="review-gate source-accountability policy",
    )
    if (
        source_accountability_policy
        != "retain_in_authoritative_source_accountability_ledger"
    ):
        raise ConfigError(
            "Review-gate policy must retain every disposition in source accountability"
        )
    analytical_leakage_policy = _approval_text(
        payload["analytical_leakage_policy"],
        field="review-gate analytical-leakage policy",
    )
    if analytical_leakage_policy != "prohibit_analysis_use":
        raise ConfigError(
            "Review-gate policy must keep analytical leakage prohibited and fatal"
        )
    return ReviewGatePolicy(
        policy_id=_approval_text(payload["policy_id"], field="review-gate policy identifier"),
        prospective_effective_version=_approval_text(
            payload["prospective_effective_version"],
            field="review-gate prospective effective version",
        ),
        effective_from=effective_from,
        approval=approval,
        fatal_issue_states=tuple(sorted(fatal_states)),
        permitted_resolved_dispositions=tuple(permitted),
        source_accountability_policy=source_accountability_policy,
        analytical_leakage_policy=analytical_leakage_policy,
        artifact_path=artifact_path,
        artifact_sha256=sha256_file(artifact_path),
    )


def load_release_approval(
    path: str | Path,
    *,
    authority_matrix: ApprovalAuthorityMatrix,
    run_id: str,
    release_target: str,
    run_identity_sha256: str,
) -> ReleaseApproval:
    """Require release-owner approval bound to the exact promotion request."""

    artifact_path = Path(path).resolve()
    if not artifact_path.is_file():
        raise ConfigError(
            "Authoritative release requires a release-owner approval record at "
            f"{artifact_path}; expected run_identity_sha256={run_identity_sha256}"
        )
    payload = _load_json_object(artifact_path)
    required_keys = {
        "schema_version",
        "status",
        "record_id",
        "scope",
        "run_id",
        "release_target",
        "run_identity_sha256",
        "authority_matrix_sha256",
        "approved_by",
        "approved_at",
        "approval_source",
    }
    if set(payload) != required_keys:
        raise ConfigError("Release approval fields do not match the required schema")
    if payload["schema_version"] != _RELEASE_APPROVAL_SCHEMA_VERSION:
        raise ConfigError("Release approval schema version is unsupported")
    if payload["status"] != _APPROVED_STATUS:
        raise ConfigError("Release approval record is not approved")
    if payload["scope"] != "release_promotion":
        raise ConfigError("Release approval scope must be release_promotion")
    expected_values = {
        "run_id": run_id,
        "release_target": release_target,
        "run_identity_sha256": run_identity_sha256,
        "authority_matrix_sha256": authority_matrix.artifact_sha256,
    }
    for field, expected in expected_values.items():
        if payload[field] != expected:
            raise ConfigError(
                f"Release approval {field} does not match the current promotion request"
            )
    approval = {
        "approved_by": _approval_text(
            payload["approved_by"],
            field="release approval approver",
        ),
        "approved_at": _approval_timestamp(payload["approved_at"]),
        "approval_source": _approval_text(
            payload["approval_source"],
            field="release approval source",
        ),
    }
    release_party = authority_matrix.gate_authorities["release_promotion"][
        "accountable_party"
    ]
    if approval["approved_by"] != release_party:
        raise ConfigError(
            "Release approval signer is not the accountable release-promotion party "
            "in the OPS-08 authority matrix"
        )
    if _approval_calendar_date(approval["approved_at"]) < _approval_calendar_date(
        authority_matrix.effective_from
    ):
        raise ConfigError(
            "Release approval predates the effective OPS-08 authority matrix"
        )
    return ReleaseApproval(
        record_id=_approval_text(
            payload["record_id"],
            field="release approval record identifier",
        ),
        run_id=run_id,
        release_target=release_target,
        run_identity_sha256=run_identity_sha256,
        authority_matrix_sha256=authority_matrix.artifact_sha256,
        approval=approval,
        artifact_path=artifact_path,
        artifact_sha256=sha256_file(artifact_path),
    )


def phase_two_review_disposition(row: Mapping[str, Any]) -> Mapping[str, str]:
    """Return the exact state-map identity used by both early and release gates."""

    raw_reasons = row.get("eligibility_reason_codes", row.get("reason_codes", ()))
    if isinstance(raw_reasons, str):
        reasons = (raw_reasons,)
    elif isinstance(raw_reasons, (list, tuple, set, frozenset)):
        reasons = tuple(str(reason) for reason in raw_reasons)
    else:
        reasons = ()
    normalized_reasons = tuple(reason.strip().upper() for reason in reasons)
    if any(
        token in reason
        for reason in normalized_reasons
        for token in ("DATA_ERROR", "PROHIBITED_USE", "STRUCTURAL")
    ):
        issue_state = "structural"
    elif any(
        "UNRESOLVED" in reason or "REVIEW" in reason
        for reason in normalized_reasons
    ):
        issue_state = "unresolved"
    elif any("WARNING" in reason for reason in normalized_reasons) or row.get(
        "status"
    ) == "warning":
        issue_state = "warning"
    elif row.get("analytical_record_status") not in {None, "included"}:
        issue_state = "excluded_series"
    else:
        issue_state = "structural"
    return MappingProxyType(
        {
            "stage": "phase_2",
            "issue_scope": "record",
            "issue_state": issue_state,
            "status": str(row.get("eligibility_tier") or "review"),
        }
    )


def validate_policy_authority_bindings(
    authority_matrix: ApprovalAuthorityMatrix,
    *,
    source_data_policy: Any | None,
    analysis_policy: Any | None,
) -> None:
    """Bind every supplied policy artifact to its OPS-08 accountable party."""

    def require_authority(authority: Any, *, gate: str, artifact: str) -> None:
        expected_party = authority_matrix.gate_authorities[gate]["accountable_party"]
        if getattr(authority, "approved_by", None) != expected_party:
            raise ConfigError(
                f"{artifact} signer is not the accountable {gate} party in the OPS-08 authority matrix"
            )
        approval_date = getattr(authority, "approval_date", None)
        try:
            parsed_date = date.fromisoformat(str(approval_date))
        except ValueError as exc:
            raise ConfigError(f"{artifact} approval date is invalid") from exc
        if parsed_date < date.fromisoformat(authority_matrix.effective_from):
            raise ConfigError(
                f"{artifact} approval predates the effective OPS-08 authority matrix"
            )

    if source_data_policy is not None:
        require_authority(
            source_data_policy.manifest_authority,
            gate="source_integrity",
            artifact="source-data policy manifest",
        )
        source_gate_by_artifact = {
            "restricted_policy": "restricted_data",
            "final_cleaning_policy": "scientific_methods",
        }
        for name, authority in source_data_policy.artifact_authorities.items():
            require_authority(
                authority,
                gate=source_gate_by_artifact.get(name, "source_integrity"),
                artifact=f"source-data policy component {name!r}",
            )
    if analysis_policy is not None:
        for name, authority in (
            ("manifest", analysis_policy.manifest_authority),
            ("support table", analysis_policy.support_authority),
            ("factor representations", analysis_policy.representation_authority),
            ("estimands", analysis_policy.estimand_authority),
            ("hypotheses", analysis_policy.hypothesis_authority),
            ("curve-model policy", analysis_policy.model_authority),
        ):
            require_authority(
                authority,
                gate="scientific_methods",
                artifact=f"analysis-policy {name}",
            )


def _validate_approved_snapshot(
    config: ValidatedConfig,
    *,
    path: Path,
    expected_content: Mapping[str, Any],
) -> RuntimePolicySnapshot:
    payload = _load_json_object(path)
    expected_keys = {
        "schema_version",
        "status",
        "approved_by",
        "approved_at",
        "approval_source",
        "policy_content",
        "policy_content_sha256",
    }
    if set(payload) != expected_keys:
        raise ConfigError("Approved policy snapshot fields do not match the required schema")
    if payload["schema_version"] != _POLICY_SNAPSHOT_SCHEMA_VERSION:
        raise ConfigError("Approved policy snapshot schema version is unsupported")
    if payload["status"] != _APPROVED_STATUS:
        raise ConfigError("Policy snapshot is not approved")
    approval = {
        "approved_by": _approval_text(payload["approved_by"], field="approver"),
        "approved_at": _approval_timestamp(payload["approved_at"]),
        "approval_source": _approval_text(payload["approval_source"], field="approval source"),
    }
    content = payload["policy_content"]
    if not isinstance(content, Mapping):
        raise ConfigError("Approved policy snapshot content must be a JSON object")
    content = dict(content)
    supplied_hash = payload["policy_content_sha256"]
    if (
        not isinstance(supplied_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", supplied_hash)
        or stable_json_sha256(content) != supplied_hash
    ):
        raise ConfigError("Approved policy snapshot content hash is invalid")
    expected_hash = stable_json_sha256(expected_content)
    if supplied_hash != expected_hash or content != dict(expected_content):
        raise ConfigError("Approved policy snapshot is stale or does not match the effective runtime")
    enablement = dict(expected_content["effective_enablement"])
    return RuntimePolicySnapshot(
        mode=config.run_mode,
        status="approved",
        policy_content_sha256=supplied_hash,
        effective_enablement_sha256=str(expected_content["effective_enablement_sha256"]),
        effective_enablement=enablement,
        approval=approval,
        artifact_path=path,
        artifact_sha256=sha256_file(path),
    )


def validate_runtime_policy(config: ValidatedConfig) -> RuntimePolicySnapshot:
    """Validate the effective runtime contract without organizational approvals."""

    allowed_raster_figures = {"png", "jpeg"}
    configured_figures = set(config.figure_formats)
    forbidden_figures = configured_figures - allowed_raster_figures
    if forbidden_figures:
        raise ConfigError("The effective figure-format profile contains a prohibited format")
    if not configured_figures.intersection(allowed_raster_figures):
        raise ConfigError(
            "The effective figure-format profile requires at least one PNG or JPEG raster format"
        )
    if config.writes_outputs:
        if str(config.raw["logging"]["level"]).upper() != "INFO":
            raise ConfigError("Writing modes require INFO operational logging")
        if not bool(config.raw["outputs"]["row_level_qc"]):
            raise ConfigError("Writing modes require complete machine-readable row-level QC")

    content = _policy_content(config)
    content_hash = stable_json_sha256(content)
    enablement = dict(content["effective_enablement"])
    return RuntimePolicySnapshot(
        mode=config.run_mode,
        status="runtime_contract",
        policy_content_sha256=content_hash,
        effective_enablement_sha256=str(content["effective_enablement_sha256"]),
        effective_enablement=enablement,
        approval=None,
        artifact_path=None,
        artifact_sha256=None,
        authority_matrix=None,
        review_gate_policy=None,
    )


def write_policy_snapshot_template(
    config: ValidatedConfig,
    destination: str | Path,
) -> Path:
    """Write an explicitly unapproved review template; never an approval record."""

    destination_path = Path(destination).resolve()
    review_gate_path = _review_gate_policy_path(config)
    authority_matrix = None
    if config.run_mode == "full" or review_gate_path.is_file():
        authority_matrix = load_approval_authority_matrix(
            _authority_matrix_path(config)
        )
    review_gate_policy = None
    if review_gate_path.is_file():
        if authority_matrix is None:
            raise ConfigError("Review-gate authority matrix was not resolved")
        review_gate_policy = load_review_gate_policy(
            review_gate_path,
            authority_matrix=authority_matrix,
        )
    content = _policy_content(config, review_gate_policy, authority_matrix)
    payload = {
        "schema_version": _POLICY_SNAPSHOT_SCHEMA_VERSION,
        "status": "REVIEW_REQUIRED",
        "approved_by": "",
        "approved_at": "",
        "approval_source": "",
        "policy_content": content,
        "policy_content_sha256": stable_json_sha256(content),
    }
    destination_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination_path


def write_approval_authority_matrix_template(destination: str | Path) -> Path:
    """Write a non-approval OPS-08 matrix template for independent completion."""

    destination_path = Path(destination).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _AUTHORITY_MATRIX_SCHEMA_VERSION,
        "status": "REVIEW_REQUIRED",
        "matrix_id": "",
        "effective_from": "",
        "approved_by": "",
        "approved_at": "",
        "approval_source": "",
        "gate_authorities": {
            gate: {
                "accountable_role": "",
                "accountable_party": "",
                "authority_scope": gate,
                "approval_source": "",
            }
            for gate in _AUTHORITY_GATES
        },
        **dict(_AUTHORITY_POLICY_VALUES),
    }
    destination_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination_path


def write_review_gate_policy_template(destination: str | Path) -> Path:
    """Write a non-approval template for the prospective disposition state map."""

    destination_path = Path(destination).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _REVIEW_GATE_POLICY_SCHEMA_VERSION,
        "status": "REVIEW_REQUIRED",
        "policy_id": "",
        "prospective_effective_version": "",
        "effective_from": "",
        "approved_by": "",
        "approved_at": "",
        "approval_source": "",
        "fatal_issue_states": list(_FATAL_REVIEW_ISSUE_STATES),
        "permitted_resolved_dispositions": [],
        "source_accountability_policy": (
            "retain_in_authoritative_source_accountability_ledger"
        ),
        "analytical_leakage_policy": "prohibit_analysis_use",
    }
    destination_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination_path


def write_release_approval_template(
    destination: str | Path,
    *,
    run_id: str,
    release_target: str,
    run_identity_sha256: str,
    authority_matrix_sha256: str,
) -> Path:
    """Write a non-approval release record bound to a computed promotion request."""

    for field, value in (
        ("run identity SHA-256", run_identity_sha256),
        ("authority matrix SHA-256", authority_matrix_sha256),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ConfigError(f"Release approval template {field} is malformed")
    destination_path = Path(destination).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _RELEASE_APPROVAL_SCHEMA_VERSION,
        "status": "REVIEW_REQUIRED",
        "record_id": "",
        "scope": "release_promotion",
        "run_id": run_id,
        "release_target": release_target,
        "run_identity_sha256": run_identity_sha256,
        "authority_matrix_sha256": authority_matrix_sha256,
        "approved_by": "",
        "approved_at": "",
        "approval_source": "",
    }
    destination_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination_path


__all__ = [
    "ApprovalAuthorityMatrix",
    "ReleaseApproval",
    "ReviewGatePolicy",
    "RuntimePolicySnapshot",
    "effective_analysis_hypotheses",
    "effective_enablement",
    "load_approval_authority_matrix",
    "load_release_approval",
    "load_review_gate_policy",
    "phase_two_review_disposition",
    "validate_policy_authority_bindings",
    "validate_runtime_policy",
    "write_approval_authority_matrix_template",
    "write_policy_snapshot_template",
    "write_release_approval_template",
    "write_review_gate_policy_template",
]
