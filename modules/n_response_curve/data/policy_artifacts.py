from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .curate import (
    NUTRIENT_CANONICAL_UNITS,
    PhysicalColumnDisposition,
    REQUIRED_REVIEWED_LOOKUP_FIELDS,
    RestrictedDataPolicy,
    ReviewedNutrientUnitControl,
    ReviewedSourceMap,
    SourceArmMap,
    validate_reviewed_curation_controls,
    validate_reviewed_fill_down_policy,
    validate_reviewed_nutrient_unit_controls,
)
from .config import ConfigError
from .duplicates import (
    DuplicateAdjudication,
    DuplicateRuleSet,
    RepeatAdjudication,
)
from .provenance import (
    ChecksumRevisionApproval,
    VerificationResult,
    VerificationSamplingPolicy,
    parse_checksum_revision_approval,
    sha256_file,
)
from .schema import (
    ReviewedLookupTable,
    validate_reviewed_lookup_table,
    validate_reviewed_missing_state_table,
)
from .ingest import KNOWN_REPRESENTATION_BASES, IngestionResult
from .cleaning import (
    FinalCleaningRule,
    SourceCleaningPolicy,
    validate_numeric_rule_unit,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ARTIFACT_KEYS = (
    "source_scope",
    "source_maps",
    "category_lookups",
    "restricted_policy",
    "duplicate_rules",
    "final_cleaning_policy",
    "checksum_revision_approvals",
    "duplicate_adjudications",
    "repeat_adjudications",
)
_OPTIONAL_ARTIFACT_KEYS = ("literature_verification",)


class SourceDataPolicyError(ValueError):
    """Raised when source-data controls are incomplete or unauthenticated."""


@dataclass(frozen=True)
class PolicyAuthority:
    artifact_type: str
    artifact_version: str
    path: Path
    sha256: str
    approved_by: str
    approval_date: str
    scope: str
    archive_location: str


@dataclass(frozen=True)
class SourceScopeRecord:
    source_name: str
    country_code: str
    activation_status: str
    schema_harmonization_status: str
    unit_comparability_status: str
    review_id: str


@dataclass(frozen=True)
class SourceDataPolicyBundle:
    """Runtime-ready source-data controls from one authenticated bundle."""

    manifest_authority: PolicyAuthority
    artifact_authorities: Mapping[str, PolicyAuthority]
    source_scope: Mapping[str, SourceScopeRecord]
    source_maps: Mapping[str, ReviewedSourceMap]
    category_lookups: Mapping[str, ReviewedLookupTable]
    restricted_policy: RestrictedDataPolicy
    restricted_secret_reference: str
    duplicate_rules: DuplicateRuleSet
    final_cleaning_policies: Mapping[str, SourceCleaningPolicy]
    checksum_revision_approvals: Mapping[str, ChecksumRevisionApproval]
    duplicate_adjudications: tuple[DuplicateAdjudication, ...]
    repeat_adjudications: tuple[RepeatAdjudication, ...]
    designated_reviewers: tuple[str, ...]

    @property
    def artifact_sha256(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                key: authority.sha256
                for key, authority in self.artifact_authorities.items()
            }
        )

    @property
    def active_source_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                source_name
                for source_name, record in self.source_scope.items()
                if record.activation_status == "approved_active"
            )
        )

    @property
    def curation_kwargs(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "source_maps": self.source_maps,
                "category_lookups": self.category_lookups,
                "restricted_policy": self.restricted_policy,
                "require_reviewed_controls": True,
            }
        )

    @property
    def ingestion_kwargs(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "checksum_revision_approvals": self.checksum_revision_approvals,
                "designated_reviewers": self.designated_reviewers,
                "source_representation_bases": MappingProxyType(
                    {
                        source_name: source_map.representation_basis
                        for source_name, source_map in self.source_maps.items()
                        if source_map.representation_basis_status == "reviewed"
                    }
                ),
            }
        )

    @property
    def resolution_kwargs(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "duplicate_rules": self.duplicate_rules,
                "duplicate_adjudications": self.duplicate_adjudications,
                "repeat_adjudications": self.repeat_adjudications,
                "designated_reviewers": self.designated_reviewers,
            }
        )


def _nonempty_text(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceDataPolicyError(f"{where} must be a nonempty string")
    return value.strip()


def _exact_object(
    value: object,
    *,
    required_keys: Iterable[str],
    where: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceDataPolicyError(f"{where} must be an object")
    required = set(required_keys)
    observed = set(value)
    if observed != required:
        missing = ", ".join(sorted(required - observed)) or "none"
        extra = ", ".join(sorted(observed - required)) or "none"
        raise SourceDataPolicyError(
            f"{where} fields do not match the required schema; "
            f"missing={missing}; extra={extra}"
        )
    return value


def _iso_date(value: object, *, where: str) -> str:
    text = _nonempty_text(value, where=where)
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise SourceDataPolicyError(f"{where} must be an ISO date") from exc
    return text


def _version(value: object, *, where: str) -> str:
    text = _nonempty_text(value, where=where)
    if _VERSION_RE.fullmatch(text) is None:
        raise SourceDataPolicyError(f"{where} contains unsupported characters")
    return text


def _string_tuple(
    value: object,
    *,
    where: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SourceDataPolicyError(f"{where} must be a JSON array")
    normalized = tuple(
        _nonempty_text(item, where=f"{where} member") for item in value
    )
    if not allow_empty and not normalized:
        raise SourceDataPolicyError(f"{where} must not be empty")
    if len(normalized) != len(set(normalized)):
        raise SourceDataPolicyError(f"{where} must contain unique values")
    return normalized


def _records(
    payload: Mapping[str, Any],
    *,
    where: str,
    allow_empty: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    raw = payload.get("records")
    if not isinstance(raw, list) or (not raw and not allow_empty):
        qualifier = "a JSON array" if allow_empty else "a nonempty JSON array"
        raise SourceDataPolicyError(f"{where}.records must be {qualifier}")
    records: list[Mapping[str, Any]] = []
    for index, record in enumerate(raw):
        if not isinstance(record, Mapping):
            raise SourceDataPolicyError(f"{where}.records[{index}] must be an object")
        records.append(MappingProxyType(dict(record)))
    return tuple(records)


def _resolve_project_path(
    project_root: Path,
    relative: object,
    *,
    where: str,
) -> Path:
    text = _nonempty_text(relative, where=where)
    candidate = Path(text)
    if candidate.is_absolute():
        raise SourceDataPolicyError(f"{where} must be project-relative")
    resolved = (project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise SourceDataPolicyError(f"{where} escapes the project root") from exc
    return resolved


def _json_payload(path: Path, *, where: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise SourceDataPolicyError(f"{where} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceDataPolicyError(f"{where} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise SourceDataPolicyError(f"{where} root must be an object")
    return MappingProxyType(dict(payload))


def _authority(
    payload: Mapping[str, Any],
    *,
    path: Path,
    actual_sha256: str,
    expected_type: str,
) -> PolicyAuthority:
    artifact_type = _nonempty_text(
        payload.get("artifact_type"),
        where=f"{expected_type} artifact_type",
    )
    if artifact_type != expected_type:
        raise SourceDataPolicyError(
            f"Expected artifact_type {expected_type!r}, observed {artifact_type!r}"
        )
    return PolicyAuthority(
        artifact_type=artifact_type,
        artifact_version=_version(
            payload.get("artifact_version"),
            where=f"{expected_type} artifact_version",
        ),
        path=path,
        sha256=actual_sha256,
        approved_by=_nonempty_text(
            payload.get("approved_by"),
            where=f"{expected_type} approved_by",
        ),
        approval_date=_iso_date(
            payload.get("approval_date"),
            where=f"{expected_type} approval_date",
        ),
        scope=_nonempty_text(
            payload.get("scope"),
            where=f"{expected_type} scope",
        ),
        archive_location=_nonempty_text(
            payload.get("archive_location"),
            where=f"{expected_type} archive_location",
        ),
    )


def _load_artifact(
    descriptor: object,
    *,
    project_root: Path,
    expected_type: str,
) -> tuple[PolicyAuthority, Mapping[str, Any]]:
    component = _exact_object(
        descriptor,
        required_keys=("path", "sha256"),
        where=f"Manifest artifact descriptor {expected_type!r}",
    )
    path = _resolve_project_path(
        project_root,
        component["path"],
        where=f"{expected_type} path",
    )
    if path.suffix.casefold() != ".json":
        raise SourceDataPolicyError(
            f"{expected_type} artifact must use the JSON format"
        )
    expected_sha256 = _nonempty_text(
        component["sha256"],
        where=f"{expected_type} sha256",
    ).lower()
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise SourceDataPolicyError(f"{expected_type} sha256 is malformed")
    if not path.is_file():
        raise SourceDataPolicyError(f"{expected_type} artifact does not exist: {path}")
    try:
        actual_sha256 = sha256_file(path)
    except OSError as exc:
        raise SourceDataPolicyError(
            f"{expected_type} artifact could not be read"
        ) from exc
    if actual_sha256 != expected_sha256:
        raise SourceDataPolicyError(
            f"{expected_type} SHA-256 mismatch: expected {expected_sha256}, "
            f"observed {actual_sha256}"
        )
    payload = _json_payload(path, where=f"{expected_type} artifact")
    return (
        _authority(
            payload,
            path=path,
            actual_sha256=actual_sha256,
            expected_type=expected_type,
        ),
        payload,
    )


def _integer(value: object, *, where: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SourceDataPolicyError(f"{where} must be an integer >= {minimum}")
    return value


def _position_mapping(value: object, *, where: str) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise SourceDataPolicyError(f"{where} must be an object")
    parsed: dict[str, int] = {}
    for field, position in value.items():
        name = _nonempty_text(field, where=f"{where} field")
        parsed[name] = _integer(position, where=f"{where}.{name}")
    if len(parsed.values()) != len(set(parsed.values())):
        raise SourceDataPolicyError(f"{where} positions must be unique")
    return MappingProxyType(parsed)


def _header_mapping(value: object, *, where: str) -> Mapping[int, str]:
    if not isinstance(value, Mapping):
        raise SourceDataPolicyError(f"{where} must be an object")
    parsed: dict[int, str] = {}
    for position, header in value.items():
        try:
            numeric_position = int(position)
        except (TypeError, ValueError) as exc:
            raise SourceDataPolicyError(
                f"{where} keys must be positive physical positions"
            ) from exc
        if numeric_position < 1 or str(numeric_position) != str(position):
            raise SourceDataPolicyError(
                f"{where} keys must be canonical positive integers"
            )
        parsed[numeric_position] = _nonempty_text(
            header,
            where=f"{where}.{position}",
        )
    return MappingProxyType(parsed)


def _source_scope(
    payload: Mapping[str, Any],
) -> Mapping[str, SourceScopeRecord]:
    scope: dict[str, SourceScopeRecord] = {}
    for index, record in enumerate(_records(payload, where="source scope")):
        where = f"source scope record {index}"
        exact = _exact_object(
            record,
            required_keys={
                "source_name",
                "country_code",
                "activation_status",
                "schema_harmonization_status",
                "unit_comparability_status",
                "review_id",
            },
            where=where,
        )
        source_name = _nonempty_text(
            exact["source_name"],
            where=f"{where}.source_name",
        )
        if source_name in scope:
            raise SourceDataPolicyError(
                f"Source scope contains duplicate source_name: {source_name}"
            )
        country_code = _nonempty_text(
            exact["country_code"],
            where=f"{where}.country_code",
        )
        if re.fullmatch(r"[A-Z]{2}", country_code) is None:
            raise SourceDataPolicyError(
                f"{where}.country_code must be an uppercase ISO alpha-2 code"
            )
        activation_status = _nonempty_text(
            exact["activation_status"],
            where=f"{where}.activation_status",
        )
        if activation_status not in {"approved_active", "approved_inactive"}:
            raise SourceDataPolicyError(
                f"{where}.activation_status must be approved_active or approved_inactive"
            )
        schema_status = _nonempty_text(
            exact["schema_harmonization_status"],
            where=f"{where}.schema_harmonization_status",
        )
        unit_status = _nonempty_text(
            exact["unit_comparability_status"],
            where=f"{where}.unit_comparability_status",
        )
        allowed_review_statuses = {"verified", "unresolved", "not_applicable"}
        if schema_status not in allowed_review_statuses:
            raise SourceDataPolicyError(
                f"{where}.schema_harmonization_status is unsupported"
            )
        if unit_status not in allowed_review_statuses:
            raise SourceDataPolicyError(
                f"{where}.unit_comparability_status is unsupported"
            )
        if activation_status == "approved_active" and (
            schema_status != "verified" or unit_status != "verified"
        ):
            raise SourceDataPolicyError(
                f"{where} cannot activate a source until schema harmonization and unit comparability are verified"
            )
        scope[source_name] = SourceScopeRecord(
            source_name=source_name,
            country_code=country_code,
            activation_status=activation_status,
            schema_harmonization_status=schema_status,
            unit_comparability_status=unit_status,
            review_id=_nonempty_text(
                exact["review_id"],
                where=f"{where}.review_id",
            ),
        )
    if not any(
        record.activation_status == "approved_active"
        for record in scope.values()
    ):
        raise SourceDataPolicyError(
            "Source scope must approve at least one active source"
        )
    return MappingProxyType(dict(sorted(scope.items())))


def _source_maps(payload: Mapping[str, Any]) -> Mapping[str, ReviewedSourceMap]:
    maps: dict[str, ReviewedSourceMap] = {}
    for index, record in enumerate(_records(payload, where="source maps")):
        where = f"source maps record {index}"
        source_name = _nonempty_text(
            record.get("source_name"),
            where=f"{where}.source_name",
        )
        if source_name in maps:
            raise SourceDataPolicyError("Source-map source names must be unique")
        raw_dispositions = record.get("dispositions")
        if not isinstance(raw_dispositions, list) or not raw_dispositions:
            raise SourceDataPolicyError(f"{where}.dispositions must be nonempty")
        dispositions: list[PhysicalColumnDisposition] = []
        for disposition_index, raw in enumerate(raw_dispositions):
            if not isinstance(raw, Mapping):
                raise SourceDataPolicyError(
                    f"{where}.dispositions[{disposition_index}] must be an object"
                )
            canonical_field = raw.get("canonical_field")
            variable_family = raw.get("variable_family")
            dispositions.append(
                PhysicalColumnDisposition(
                    position=_integer(
                        raw.get("position"),
                        where=f"{where}.dispositions[{disposition_index}].position",
                    ),
                    role=_nonempty_text(
                        raw.get("role"),
                        where=f"{where}.dispositions[{disposition_index}].role",
                    ),
                    canonical_field=(
                        _nonempty_text(
                            canonical_field,
                            where=(
                                f"{where}.dispositions[{disposition_index}]"
                                ".canonical_field"
                            ),
                        )
                        if canonical_field is not None
                        else None
                    ),
                    variable_family=(
                        _nonempty_text(
                            variable_family,
                            where=(
                                f"{where}.dispositions[{disposition_index}]"
                                ".variable_family"
                            ),
                        )
                        if variable_family is not None
                        else None
                    ),
                )
            )
        raw_arms = record.get("arms", [])
        if not isinstance(raw_arms, list):
            raise SourceDataPolicyError(f"{where}.arms must be a JSON array")
        arms: list[SourceArmMap] = []
        for arm_index, raw_arm in enumerate(raw_arms):
            if not isinstance(raw_arm, Mapping):
                raise SourceDataPolicyError(
                    f"{where}.arms[{arm_index}] must be an object"
                )
            raw_constants = raw_arm.get("constants", {})
            if not isinstance(raw_constants, Mapping):
                raise SourceDataPolicyError(
                    f"{where}.arms[{arm_index}].constants must be an object"
                )
            arms.append(
                SourceArmMap(
                    arm_id=_nonempty_text(
                        raw_arm.get("arm_id"),
                        where=f"{where}.arms[{arm_index}].arm_id",
                    ),
                    role=_nonempty_text(
                        raw_arm.get("role"),
                        where=f"{where}.arms[{arm_index}].role",
                    ),
                    field_positions=_position_mapping(
                        raw_arm.get("field_positions", {}),
                        where=f"{where}.arms[{arm_index}].field_positions",
                    ),
                    constants=MappingProxyType(
                        {
                            _nonempty_text(
                                key,
                                where=f"{where}.arms[{arm_index}].constants key",
                            ): _nonempty_text(
                                value,
                                where=f"{where}.arms[{arm_index}].constants value",
                            )
                            for key, value in raw_constants.items()
                        }
                    ),
                )
            )
        source_sha256 = _nonempty_text(
            record.get("source_sha256"),
            where=f"{where}.source_sha256",
        ).lower()
        if _SHA256_RE.fullmatch(source_sha256) is None:
            raise SourceDataPolicyError(f"{where}.source_sha256 is malformed")
        normalization_map_version = record.get("normalization_map_version")
        normalization_review_id = record.get("normalization_review_id")
        raw_representation_basis = record.get("representation_basis")
        representation_basis = (
            _nonempty_text(
                raw_representation_basis,
                where=f"{where}.representation_basis",
            )
            if raw_representation_basis is not None
            else "unclear_mixed_scope"
        )
        if representation_basis not in KNOWN_REPRESENTATION_BASES:
            raise SourceDataPolicyError(
                f"{where}.representation_basis is not a supported category"
            )
        maps[source_name] = ReviewedSourceMap(
            source_name=source_name,
            map_version=_version(
                record.get("map_version"),
                where=f"{where}.map_version",
            ),
            review_id=_nonempty_text(
                record.get("review_id"),
                where=f"{where}.review_id",
            ),
            source_sha256=source_sha256,
            encoding=_nonempty_text(
                record.get("encoding"),
                where=f"{where}.encoding",
            ),
            workbook_csv_basis=_nonempty_text(
                record.get("workbook_csv_basis"),
                where=f"{where}.workbook_csv_basis",
            ),
            fields=_position_mapping(
                record.get("fields"),
                where=f"{where}.fields",
            ),
            expected_headers=_header_mapping(
                record.get("expected_headers"),
                where=f"{where}.expected_headers",
            ),
            dispositions=tuple(dispositions),
            representation_basis=representation_basis,
            representation_basis_status=(
                "reviewed"
                if raw_representation_basis is not None
                else "review_required"
            ),
            fill_down_headers=_string_tuple(
                record.get("fill_down_headers", []),
                where=f"{where}.fill_down_headers",
                allow_empty=True,
            ),
            arms=tuple(arms),
            normalization_map_version=(
                _version(
                    normalization_map_version,
                    where=f"{where}.normalization_map_version",
                )
                if normalization_map_version is not None
                else None
            ),
            normalization_review_id=(
                _nonempty_text(
                    normalization_review_id,
                    where=f"{where}.normalization_review_id",
                )
                if normalization_review_id is not None
                else None
            ),
            declared_constant_fields=_string_tuple(
                record.get("declared_constant_fields", []),
                where=f"{where}.declared_constant_fields",
                allow_empty=True,
            ),
        )
    return MappingProxyType(maps)


def _category_lookups(
    payload: Mapping[str, Any],
) -> Mapping[str, ReviewedLookupTable]:
    lookups: dict[str, ReviewedLookupTable] = {}
    for index, record in enumerate(_records(payload, where="category lookups")):
        where = f"category lookups record {index}"
        field = _nonempty_text(record.get("field"), where=f"{where}.field")
        if field in lookups:
            raise SourceDataPolicyError("Category lookup fields must be unique")
        raw_aliases = record.get("aliases")
        if not isinstance(raw_aliases, Mapping) or not raw_aliases:
            raise SourceDataPolicyError(f"{where}.aliases must be a nonempty object")
        aliases = {
            _nonempty_text(key, where=f"{where}.aliases key"): _string_tuple(
                value,
                where=f"{where}.aliases.{key}",
            )
            for key, value in raw_aliases.items()
        }
        lookups[field] = ReviewedLookupTable(
            map_version=_version(
                record.get("map_version"),
                where=f"{where}.map_version",
            ),
            review_id=_nonempty_text(
                record.get("review_id"),
                where=f"{where}.review_id",
            ),
            aliases=MappingProxyType(aliases),
        )
        try:
            validate_reviewed_lookup_table(lookups[field])
        except ValueError as exc:
            raise SourceDataPolicyError(
                f"{where} is not a valid reviewed lookup: {exc}"
            ) from exc
    return MappingProxyType(lookups)


def _restricted_policy(
    payload: Mapping[str, Any],
    *,
    secrets: Mapping[str, bytes],
) -> tuple[RestrictedDataPolicy, str]:
    records = _records(payload, where="restricted policy")
    if len(records) != 1:
        raise SourceDataPolicyError("Restricted policy must contain exactly one record")
    record = records[0]
    secret_reference = _nonempty_text(
        record.get("pseudonym_secret_reference"),
        where="restricted policy pseudonym_secret_reference",
    )
    if secret_reference not in secrets:
        raise SourceDataPolicyError(
            "Restricted policy pseudonym-secret reference does not match any "
            "supplied secret reference"
        )
    secret = secrets[secret_reference]
    if not isinstance(secret, bytes):
        raise SourceDataPolicyError(
            "Resolved restricted-policy pseudonym secret must be provided as bytes"
        )
    if len(secret) < 16:
        raise SourceDataPolicyError(
            "Resolved restricted-policy pseudonym secret must contain at least 16 bytes"
        )
    public_release_fields = _string_tuple(
        record.get("public_release_fields"),
        where="restricted policy public_release_fields",
    )
    identifier_fields = _string_tuple(
        record.get("identifier_fields"),
        where="restricted policy identifier_fields",
    )
    precise_location_fields = _string_tuple(
        record.get("precise_location_fields"),
        where="restricted policy precise_location_fields",
    )
    detailed_location_fields = _string_tuple(
        record.get("detailed_location_fields"),
        where="restricted policy detailed_location_fields",
        allow_empty=True,
    )
    approved_geography_fields = _string_tuple(
        record.get("approved_geography_fields"),
        where="restricted policy approved_geography_fields",
        allow_empty=True,
    )
    prohibited_public_fields = {
        *identifier_fields,
        *precise_location_fields,
        *(set(detailed_location_fields) - set(approved_geography_fields)),
        "source_path",
        "schema_map_path",
        "raw_cells",
        "raw_headers",
        "raw_column_ids",
        "column_dispositions",
        "restricted_path_aliases",
        "record_uid",
        "parent_row_uid",
        "source_uid",
        "source_row_number",
        "source_physical_line_number",
        "controlled_subject_uid",
    }
    invalid_public_fields = set(public_release_fields) & prohibited_public_fields
    if invalid_public_fields:
        raise SourceDataPolicyError(
            "Restricted policy public_release_fields contains prohibited field(s): "
            + ", ".join(sorted(invalid_public_fields))
        )
    return (
        RestrictedDataPolicy(
            pseudonym_salt=secret,
            identifier_fields=identifier_fields,
            precise_location_fields=precise_location_fields,
            detailed_location_fields=detailed_location_fields,
            approved_geography_fields=approved_geography_fields,
            public_release_fields=public_release_fields,
            access_review_id=_nonempty_text(
                record.get("access_review_id"),
                where="restricted policy access_review_id",
            ),
            automated_disclosure_review_id=_nonempty_text(
                record.get("automated_disclosure_review_id"),
                where="restricted policy automated_disclosure_review_id",
            ),
            human_disclosure_review_id=_nonempty_text(
                record.get("human_disclosure_review_id"),
                where="restricted policy human_disclosure_review_id",
            ),
        ),
        secret_reference,
    )


def _duplicate_rules(payload: Mapping[str, Any]) -> DuplicateRuleSet:
    records = _records(payload, where="duplicate rules")
    if len(records) != 1:
        raise SourceDataPolicyError("Duplicate rules must contain exactly one record")
    record = records[0]
    raw_tolerances = record.get("probable_numeric_tolerances")
    if not isinstance(raw_tolerances, Mapping):
        raise SourceDataPolicyError(
            "duplicate rules probable_numeric_tolerances must be an object"
        )
    tolerances: dict[str, float] = {}
    for field, value in raw_tolerances.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SourceDataPolicyError(
                "Duplicate-rule numeric tolerances must be numeric"
            )
        tolerance = float(value)
        if tolerance < 0:
            raise SourceDataPolicyError(
                "Duplicate-rule numeric tolerances must be nonnegative"
            )
        tolerances[_nonempty_text(field, where="duplicate tolerance field")] = tolerance
    cross_source_only = record.get("probable_cross_source_only")
    if not isinstance(cross_source_only, bool):
        raise SourceDataPolicyError(
            "duplicate rules probable_cross_source_only must be boolean"
        )
    return DuplicateRuleSet(
        version=_version(record.get("version"), where="duplicate rules version"),
        review_id=_nonempty_text(
            record.get("review_id"),
            where="duplicate rules review_id",
        ),
        exact_key_fields=_string_tuple(
            record.get("exact_key_fields"),
            where="duplicate rules exact_key_fields",
        ),
        probable_key_fields=_string_tuple(
            record.get("probable_key_fields"),
            where="duplicate rules probable_key_fields",
        ),
        probable_numeric_tolerances=MappingProxyType(tolerances),
        casefold_fields=_string_tuple(
            record.get("casefold_fields", []),
            where="duplicate rules casefold_fields",
            allow_empty=True,
        ),
        probable_cross_source_only=cross_source_only,
    )


def _optional_finite_number(value: object, *, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SourceDataPolicyError(f"{where} must be numeric or null")
    number = float(value)
    if not math.isfinite(number):
        raise SourceDataPolicyError(f"{where} must be finite")
    return number


def _final_cleaning_policies(
    payload: Mapping[str, Any],
    *,
    designated_reviewers: tuple[str, ...],
) -> Mapping[str, SourceCleaningPolicy]:
    policies: dict[str, SourceCleaningPolicy] = {}
    policy_keys = {
        "source_name",
        "policy_id",
        "review_id",
        "reviewer",
        "reviewed_on",
        "default_action",
        "untrimmed_sensitivity_required",
        "rules",
    }
    rule_keys = {
        "rule_id",
        "rule_type",
        "field",
        "raw_position",
        "unit",
        "lower_bound",
        "upper_bound",
        "lower_bound_inclusive",
        "upper_bound_inclusive",
        "values",
        "action",
        "reason_code",
    }
    for index, raw_policy in enumerate(
        _records(payload, where="final cleaning policy")
    ):
        where = f"final cleaning policy record {index}"
        exact = _exact_object(
            raw_policy,
            required_keys=policy_keys,
            where=where,
        )
        source_name = _nonempty_text(
            exact["source_name"],
            where=f"{where}.source_name",
        )
        if source_name in policies:
            raise SourceDataPolicyError(
                f"Final cleaning policy contains duplicate source_name: {source_name}"
            )
        reviewer = _nonempty_text(exact["reviewer"], where=f"{where}.reviewer")
        if reviewer not in designated_reviewers:
            raise SourceDataPolicyError(
                f"{where}.reviewer is not in the designated reviewer registry"
            )
        raw_rules = exact["rules"]
        if not isinstance(raw_rules, list):
            raise SourceDataPolicyError(f"{where}.rules must be a JSON array")
        rules: list[FinalCleaningRule] = []
        for rule_index, raw_rule in enumerate(raw_rules):
            rule_where = f"{where}.rules[{rule_index}]"
            rule = _exact_object(
                raw_rule,
                required_keys=rule_keys,
                where=rule_where,
            )
            field = rule["field"]
            raw_position = rule["raw_position"]
            lower_inclusive = rule["lower_bound_inclusive"]
            upper_inclusive = rule["upper_bound_inclusive"]
            if not isinstance(lower_inclusive, bool) or not isinstance(
                upper_inclusive,
                bool,
            ):
                raise SourceDataPolicyError(
                    f"{rule_where} bound-inclusion fields must be boolean"
                )
            try:
                rules.append(
                    FinalCleaningRule(
                        rule_id=_nonempty_text(
                            rule["rule_id"], where=f"{rule_where}.rule_id"
                        ),
                        rule_type=_nonempty_text(
                            rule["rule_type"], where=f"{rule_where}.rule_type"
                        ),
                        field=(
                            _nonempty_text(field, where=f"{rule_where}.field")
                            if field is not None
                            else None
                        ),
                        raw_position=(
                            _integer(
                                raw_position,
                                where=f"{rule_where}.raw_position",
                            )
                            if raw_position is not None
                            else None
                        ),
                        unit=_nonempty_text(
                            rule["unit"], where=f"{rule_where}.unit"
                        ),
                        lower_bound=_optional_finite_number(
                            rule["lower_bound"],
                            where=f"{rule_where}.lower_bound",
                        ),
                        upper_bound=_optional_finite_number(
                            rule["upper_bound"],
                            where=f"{rule_where}.upper_bound",
                        ),
                        lower_bound_inclusive=lower_inclusive,
                        upper_bound_inclusive=upper_inclusive,
                        values=_string_tuple(
                            rule["values"],
                            where=f"{rule_where}.values",
                            allow_empty=True,
                        ),
                        action=_nonempty_text(
                            rule["action"], where=f"{rule_where}.action"
                        ),
                        reason_code=_nonempty_text(
                            rule["reason_code"],
                            where=f"{rule_where}.reason_code",
                        ),
                    )
                )
            except ValueError as exc:
                raise SourceDataPolicyError(f"{rule_where} is invalid: {exc}") from exc
        untrimmed_required = exact["untrimmed_sensitivity_required"]
        if not isinstance(untrimmed_required, bool):
            raise SourceDataPolicyError(
                f"{where}.untrimmed_sensitivity_required must be boolean"
            )
        try:
            policies[source_name] = SourceCleaningPolicy(
                source_name=source_name,
                policy_id=_version(
                    exact["policy_id"],
                    where=f"{where}.policy_id",
                ),
                review_id=_nonempty_text(
                    exact["review_id"],
                    where=f"{where}.review_id",
                ),
                reviewed_by=reviewer,
                reviewed_on=_iso_date(
                    exact["reviewed_on"],
                    where=f"{where}.reviewed_on",
                ),
                default_action=_nonempty_text(
                    exact["default_action"],
                    where=f"{where}.default_action",
                ),
                untrimmed_sensitivity_required=untrimmed_required,
                rules=tuple(rules),
            )
        except ValueError as exc:
            raise SourceDataPolicyError(f"{where} is invalid: {exc}") from exc
    return MappingProxyType(dict(sorted(policies.items())))


def _checksum_revision_approvals(
    payload: Mapping[str, Any],
    *,
    project_root: Path,
    designated_reviewers: tuple[str, ...],
) -> Mapping[str, ChecksumRevisionApproval]:
    approvals: dict[str, ChecksumRevisionApproval] = {}
    approval_fields = {
        "artifact_path",
        "reviewer",
        "reviewed_on",
        "rationale",
        "old_sha256",
        "new_sha256",
        "manifest_revision",
        "structural_comparison_sha256",
        "prior_registered_path",
    }
    for index, record in enumerate(
        _records(
            payload,
            where="checksum revision approvals",
            allow_empty=True,
        )
    ):
        where = f"checksum revision approval record {index}"
        exact = _exact_object(
            record,
            required_keys={"source_name", *approval_fields},
            where=where,
        )
        source_name = _nonempty_text(
            exact["source_name"],
            where=f"{where}.source_name",
        )
        if source_name in approvals:
            raise SourceDataPolicyError(
                f"Checksum revision approvals contain duplicate source_name: {source_name}"
            )
        approval_payload = {
            key: exact[key]
            for key in approval_fields
        }
        approval_payload["prior_registered_path"] = str(
            _resolve_project_path(
                project_root,
                exact["prior_registered_path"],
                where=f"{where}.prior_registered_path",
            )
        )
        try:
            parsed = parse_checksum_revision_approval(approval_payload)
        except ConfigError as exc:
            raise SourceDataPolicyError(
                f"{where} is invalid: {exc}"
            ) from exc
        if parsed.reviewer not in designated_reviewers:
            raise SourceDataPolicyError(
                f"{where}.reviewer is not a designated reviewer"
            )
        approvals[source_name] = parsed
    return MappingProxyType(dict(sorted(approvals.items())))


def _duplicate_adjudications(
    payload: Mapping[str, Any],
    *,
    designated_reviewers: tuple[str, ...],
) -> tuple[DuplicateAdjudication, ...]:
    records = _records(
        payload,
        where="duplicate adjudications",
        allow_empty=True,
    )
    adjudications: list[DuplicateAdjudication] = []
    for index, record in enumerate(records):
        where = f"duplicate adjudications record {index}"
        reviewer = _nonempty_text(record.get("reviewer"), where=f"{where}.reviewer")
        if reviewer not in designated_reviewers:
            raise SourceDataPolicyError(
                f"{where}.reviewer is not in the designated reviewer registry"
            )
        canonical = record.get("canonical_record_uid")
        adjudications.append(
            DuplicateAdjudication(
                duplicate_group_uid=_nonempty_text(
                    record.get("duplicate_group_uid"),
                    where=f"{where}.duplicate_group_uid",
                ),
                rules_version=_version(
                    record.get("rules_version"),
                    where=f"{where}.rules_version",
                ),
                disposition=_nonempty_text(
                    record.get("disposition"),
                    where=f"{where}.disposition",
                ),
                canonical_record_uid=(
                    _nonempty_text(
                        canonical,
                        where=f"{where}.canonical_record_uid",
                    )
                    if canonical is not None
                    else None
                ),
                reviewer=reviewer,
                reviewed_on=_iso_date(
                    record.get("reviewed_on"),
                    where=f"{where}.reviewed_on",
                ),
                rationale=_nonempty_text(
                    record.get("rationale"),
                    where=f"{where}.rationale",
                ),
            )
        )
    return tuple(adjudications)


def _repeat_adjudications(
    payload: Mapping[str, Any],
    *,
    designated_reviewers: tuple[str, ...],
) -> tuple[RepeatAdjudication, ...]:
    records = _records(
        payload,
        where="repeat adjudications",
        allow_empty=True,
    )
    adjudications: list[RepeatAdjudication] = []
    for index, record in enumerate(records):
        where = f"repeat adjudications record {index}"
        reviewer = _nonempty_text(record.get("reviewer"), where=f"{where}.reviewer")
        if reviewer not in designated_reviewers:
            raise SourceDataPolicyError(
                f"{where}.reviewer is not in the designated reviewer registry"
            )
        adjudications.append(
            RepeatAdjudication(
                record_uids=_string_tuple(
                    record.get("record_uids"),
                    where=f"{where}.record_uids",
                ),
                classification=_nonempty_text(
                    record.get("classification"),
                    where=f"{where}.classification",
                ),
                reviewer=reviewer,
                reviewed_on=_iso_date(
                    record.get("reviewed_on"),
                    where=f"{where}.reviewed_on",
                ),
                rationale=_nonempty_text(
                    record.get("rationale"),
                    where=f"{where}.rationale",
                ),
                review_id=_nonempty_text(
                    record.get("review_id"),
                    where=f"{where}.review_id",
                ),
            )
        )
    return tuple(adjudications)


def load_source_data_policy_manifest(
    manifest_path: str | Path,
    *,
    expected_sha256: str,
    project_root: str | Path,
    secrets: Mapping[str, bytes],
) -> SourceDataPolicyBundle:
    """Load one complete, hash-bound source-data policy bundle."""

    root = Path(project_root).resolve()
    manifest = Path(manifest_path).resolve()
    try:
        manifest.relative_to(root)
    except ValueError as exc:
        raise SourceDataPolicyError("Source-data policy manifest is outside the project root") from exc
    if manifest.suffix.casefold() != ".json":
        raise SourceDataPolicyError(
            "Source-data policy manifest must use the JSON format"
        )
    if not manifest.is_file():
        raise SourceDataPolicyError(
            f"Source-data policy manifest does not exist: {manifest}"
        )
    expected = expected_sha256.lower()
    if _SHA256_RE.fullmatch(expected) is None:
        raise SourceDataPolicyError("Source-data policy manifest SHA-256 is malformed")
    try:
        actual = sha256_file(manifest)
    except OSError as exc:
        raise SourceDataPolicyError(
            "Source-data policy manifest could not be read"
        ) from exc
    if actual != expected:
        raise SourceDataPolicyError(
            f"Source-data policy manifest SHA-256 mismatch: expected {expected}, observed {actual}"
        )
    payload = _json_payload(manifest, where="source-data policy manifest")
    manifest_authority = _authority(
        payload,
        path=manifest,
        actual_sha256=actual,
        expected_type="source_data_policy_bundle",
    )
    designated_reviewers = _string_tuple(
        payload.get("designated_reviewers"),
        where="source-data policy designated_reviewers",
    )
    raw_artifacts = payload.get("artifacts")
    if (
        not isinstance(raw_artifacts, Mapping)
        or set(raw_artifacts) != set(_ARTIFACT_KEYS)
        or len(raw_artifacts) != len(_ARTIFACT_KEYS)
    ):
        raise SourceDataPolicyError(
            "Source-data policy artifacts must contain exactly: "
            + ", ".join(_ARTIFACT_KEYS)
        )
    authorities: dict[str, PolicyAuthority] = {}
    artifacts: dict[str, Mapping[str, Any]] = {}
    for key in _ARTIFACT_KEYS:
        authority, artifact_payload = _load_artifact(
            raw_artifacts[key],
            project_root=root,
            expected_type=key,
        )
        authorities[key] = authority
        artifacts[key] = artifact_payload

    restricted_policy, secret_reference = _restricted_policy(
        artifacts["restricted_policy"],
        secrets=secrets,
    )
    return SourceDataPolicyBundle(
        manifest_authority=manifest_authority,
        artifact_authorities=MappingProxyType(authorities),
        source_scope=_source_scope(artifacts["source_scope"]),
        source_maps=_source_maps(artifacts["source_maps"]),
        category_lookups=_category_lookups(artifacts["category_lookups"]),
        restricted_policy=restricted_policy,
        restricted_secret_reference=secret_reference,
        duplicate_rules=_duplicate_rules(artifacts["duplicate_rules"]),
        final_cleaning_policies=_final_cleaning_policies(
            artifacts["final_cleaning_policy"],
            designated_reviewers=designated_reviewers,
        ),
        checksum_revision_approvals=_checksum_revision_approvals(
            artifacts["checksum_revision_approvals"],
            project_root=root,
            designated_reviewers=designated_reviewers,
        ),
        duplicate_adjudications=_duplicate_adjudications(
            artifacts["duplicate_adjudications"],
            designated_reviewers=designated_reviewers,
        ),
        repeat_adjudications=_repeat_adjudications(
            artifacts["repeat_adjudications"],
            designated_reviewers=designated_reviewers,
        ),
        designated_reviewers=designated_reviewers,
    )


def validate_source_data_policy_coverage(
    bundle: SourceDataPolicyBundle,
    ingestion: IngestionResult,
    *,
    required_lookup_fields: Iterable[str] = REQUIRED_REVIEWED_LOOKUP_FIELDS,
) -> None:
    """Validate the bundle's exact controls against the sources actually ingested."""

    validate_reviewed_curation_controls(
        ingestion,
        source_maps=bundle.source_maps,
        category_lookups=bundle.category_lookups,
        required_lookup_fields=required_lookup_fields,
    )
    source_names = {source.source_name for source in ingestion.sources}
    cleaning_sources = set(bundle.final_cleaning_policies)
    if source_names != cleaning_sources:
        raise SourceDataPolicyError(
            "Final cleaning policy coverage does not match the ingested sources: "
            f"ingested={sorted(source_names)}; reviewed={sorted(cleaning_sources)}"
        )


def validate_source_scope_activation(
    bundle: SourceDataPolicyBundle,
    config: object,
) -> None:
    """Require config activation to exactly match one reviewed source scope."""

    enabled_sources = tuple(getattr(config, "enabled_sources", ()))
    approved_active = bundle.active_source_names
    if set(enabled_sources) != set(approved_active) or len(enabled_sources) != len(
        approved_active
    ):
        raise SourceDataPolicyError(
            "Configured enabled sources do not match the approved source scope: "
            f"configured={sorted(enabled_sources)}; approved={list(approved_active)}"
        )
    sources = getattr(config, "sources", None)
    scope_countries = tuple(getattr(config, "scope_countries", ()))
    if not isinstance(sources, Mapping):
        raise SourceDataPolicyError("Configured source registry is unavailable")
    if set(sources) != set(bundle.source_scope):
        missing = sorted(set(sources) - set(bundle.source_scope))
        extra = sorted(set(bundle.source_scope) - set(sources))
        raise SourceDataPolicyError(
            "Approved source scope does not reconcile to the configured source registry: "
            f"missing={missing}; extra={extra}"
        )
    for source_name, scope_record in bundle.source_scope.items():
        raw_source = sources[source_name]
        if not isinstance(raw_source, Mapping):
            raise SourceDataPolicyError(
                f"Configured source {source_name!r} must be an object"
            )
        configured_country = raw_source.get("country_code")
        if configured_country != scope_record.country_code:
            raise SourceDataPolicyError(
                f"Configured country for source {source_name!r} does not match the approved source scope"
            )
        if scope_record.activation_status != "approved_active":
            continue
        if scope_record.country_code not in scope_countries:
            raise SourceDataPolicyError(
                f"Approved active source {source_name!r} is outside configured country scope"
            )
        if raw_source.get("availability") != "available":
            raise SourceDataPolicyError(
                f"Approved active source {source_name!r} is not configured as available"
            )
        if raw_source.get("confirmation_status") != "verified":
            raise SourceDataPolicyError(
                f"Approved active source {source_name!r} is not configured as verified"
            )
        if source_name not in bundle.source_maps:
            raise SourceDataPolicyError(
                f"Approved active source {source_name!r} has no reviewed source map"
            )
        if source_name not in bundle.final_cleaning_policies:
            raise SourceDataPolicyError(
                f"Approved active source {source_name!r} has no reviewed final cleaning policy"
            )


__all__ = [
    "PolicyAuthority",
    "SourceDataPolicyBundle",
    "SourceDataPolicyError",
    "SourceScopeRecord",
    "load_source_data_policy_manifest",
    "validate_source_data_policy_coverage",
    "validate_source_scope_activation",
]
