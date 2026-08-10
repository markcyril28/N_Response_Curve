from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
import json
from pathlib import Path
import re
from typing import Any, Mapping

from n_response_curve.analysis.analysis_matrix import build_source_combinations
from n_response_curve.data.config import ConfigError, ValidatedConfig
from n_response_curve.data.provenance import sha256_file, stable_json_sha256


_POLICY_SNAPSHOT_SCHEMA_VERSION = 1
_AUTHORITY_MATRIX_SCHEMA_VERSION = 1
_APPROVED_STATUS = "APPROVED"
_SAFE_APPROVAL_TEXT = re.compile(r"^[^\x00-\x1f\x7f]+$")
_AUTHORITY_GATES = (
    "source_integrity",
    "restricted_data",
    "scientific_methods",
    "runtime_integrity",
    "release_promotion",
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

    def manifest_payload(self, *, project_root: Path) -> dict[str, Any]:
        artifact_relative: str | None = None
        if self.artifact_path is not None:
            try:
                artifact_relative = self.artifact_path.relative_to(project_root).as_posix()
            except ValueError:
                artifact_relative = None
        return {
            "mode": self.mode,
            "status": self.status,
            "policy_content_sha256": self.policy_content_sha256,
            "effective_enablement_sha256": self.effective_enablement_sha256,
            "effective_enablement": _json_value(self.effective_enablement),
            "approval": dict(self.approval) if self.approval is not None else None,
            "approval_artifact_path": artifact_relative,
            "approval_artifact_sha256": self.artifact_sha256,
            "approval_authority_matrix": (
                self.authority_matrix.manifest_payload(project_root=project_root)
                if self.authority_matrix is not None
                else None
            ),
        }


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


def _semantic_policy() -> dict[str, Any]:
    return {
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
            "modes": ["validate", "full"],
            "failure_states": ["warning", "unresolved", "excluded_series"],
            "failure_timing": "before_authoritative_promotion",
        },
        "tables": {
            "required_formats": ["csv", "parquet"],
        },
        "figures": {
            "required_formats": ["png"],
            "optional_formats": ["jpeg"],
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
            "requires_approved_record": True,
            "requires_validated_stage": True,
            "preserve_prior_package": True,
        },
        "logging": {
            "operational_level": "INFO",
            "complete_machine_readable_qc": True,
        },
    }


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


def _policy_content(config: ValidatedConfig) -> dict[str, Any]:
    enablement = effective_enablement(config)
    return {
        "schema_version": _POLICY_SNAPSHOT_SCHEMA_VERSION,
        "mode": config.run_mode,
        "config_sha256": sha256_file(config.config_path),
        "semantic_policy": _semantic_policy(),
        "effective_enablement": enablement,
        "effective_enablement_sha256": stable_json_sha256(enablement),
    }


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
    return ApprovalAuthorityMatrix(
        matrix_id=matrix_id,
        effective_from=effective_from,
        approval=approval,
        gate_authorities=gate_authorities,
        artifact_path=artifact_path,
        artifact_sha256=sha256_file(artifact_path),
        **policies,
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
    """Validate runtime policy before restricted access or authoritative use."""

    forbidden_figures = set(config.figure_formats) - {"png", "jpeg"}
    if forbidden_figures:
        raise ConfigError("The effective figure-format profile contains a prohibited format")
    if config.run_mode == "full":
        if set(config.output_formats) != {"csv", "parquet"}:
            raise ConfigError("Authoritative mode requires the approved CSV-and-Parquet table profile")
        if "png" not in config.figure_formats:
            raise ConfigError("Authoritative mode requires PNG figures")
    if config.writes_outputs:
        if str(config.raw["logging"]["level"]).upper() != "INFO":
            raise ConfigError("Writing modes require INFO operational logging")
        if not bool(config.raw["outputs"]["row_level_qc"]):
            raise ConfigError("Writing modes require complete machine-readable row-level QC")

    content = _policy_content(config)
    content_hash = stable_json_sha256(content)
    enablement = dict(content["effective_enablement"])
    authority_matrix = None
    if _enabled_restricted_sources(config):
        authority_matrix = load_approval_authority_matrix(
            _authority_matrix_path(config)
        )
    if config.run_mode != "full":
        return RuntimePolicySnapshot(
            mode=config.run_mode,
            status="non_authoritative_runtime_contract",
            policy_content_sha256=content_hash,
            effective_enablement_sha256=str(content["effective_enablement_sha256"]),
            effective_enablement=enablement,
            approval=None,
            artifact_path=None,
            artifact_sha256=None,
            authority_matrix=authority_matrix,
        )

    snapshot_path = (
        config.paths["run_metadata_root"]
        / "approvals"
        / "policy_snapshots"
        / "full.json"
    ).resolve()
    if not snapshot_path.is_relative_to(config.project_root):
        raise ConfigError("Approved policy snapshot path escapes the project root")
    snapshot = _validate_approved_snapshot(
        config,
        path=snapshot_path,
        expected_content=content,
    )
    if authority_matrix is None:
        authority_matrix = load_approval_authority_matrix(
            _authority_matrix_path(config)
        )
    runtime_party = authority_matrix.gate_authorities["runtime_integrity"][
        "accountable_party"
    ]
    if snapshot.approval is None or snapshot.approval["approved_by"] != runtime_party:
        raise ConfigError(
            "Approved runtime snapshot signer is not the accountable runtime-integrity "
            "party in the OPS-08 authority matrix"
        )
    return replace(snapshot, authority_matrix=authority_matrix)


def write_policy_snapshot_template(
    config: ValidatedConfig,
    destination: str | Path,
) -> Path:
    """Write an explicitly unapproved review template; never an approval record."""

    destination_path = Path(destination).resolve()
    content = _policy_content(config)
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


__all__ = [
    "RuntimePolicySnapshot",
    "effective_analysis_hypotheses",
    "effective_enablement",
    "validate_runtime_policy",
    "write_policy_snapshot_template",
]
