from __future__ import annotations

from dataclasses import dataclass
import math
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence


# Row-local rule types decide one record from that record alone. Grouped rule
# types need the whole distribution the record sits in, so they are evaluated
# once per source before any record is dispositioned.
_ROW_LOCAL_RULE_TYPES = frozenset({"numeric_outside_range", "remark_match"})
_GROUPED_RULE_TYPES = frozenset({"distributional_outlier", "influence_outlier"})
_RULE_TYPES = _ROW_LOCAL_RULE_TYPES | _GROUPED_RULE_TYPES
_RULE_ACTIONS = frozenset({"exclude_primary", "flag_only", "retain"})
_DEFAULT_ACTIONS = frozenset({"retain"})
_DISTRIBUTIONAL_STATISTICS = frozenset({"modified_z_score", "iqr_fence"})
_INFLUENCE_STATISTICS = frozenset(
    {"cooks_distance", "leverage", "dffits", "studentized_residual"}
)
_GROUPING_SCOPES = frozenset({"source", "response_series"})
_GROUPING_SCOPE_KEYS = {
    "source": "source_name",
    "response_series": "response_series_uid",
}
# Thresholds on a distribution are multiples of a robust spread, never a
# physical quantity, so their declared unit is dimensionless by contract.
_DIMENSIONLESS_UNIT = "dimensionless"
# A robust centre and spread are not defined below four values; a smaller group
# is a review hold rather than a silently retained record.
_MINIMUM_DISTRIBUTIONAL_GROUP_SIZE = 4
_PRESPECIFICATION_STATUSES = frozenset({"prespecified_before_fitted_conclusions"})
_MODIFIED_Z_CONSTANT = 0.6745
_RULE_MATCHED = "match"
_RULE_NOT_MATCHED = "no_match"
_SPACE_RE = re.compile(r"\s+")
FINAL_CLEANING_METADATA_FIELDS = (
    "cleaning_policy_id",
    "cleaning_review_id",
    "cleaning_reviewer",
    "cleaning_reviewed_on",
    "cleaning_prespecification_status",
    "cleaning_rule_ids",
    "cleaning_unevaluable_rule_ids",
    "cleaning_reason_codes",
    "cleaning_review_status",
    "final_analytical_membership_status",
    "untrimmed_sensitivity_membership_status",
)
_CANONICAL_NUMERIC_FIELD_UNITS = {
    "yield_t_ha": frozenset({"t/ha", "t ha-1", "tonnes ha-1"}),
    "yield_kg_ha": frozenset({"kg/ha", "kg ha-1", "kg ha^-1"}),
    "n_rate_kg_ha": frozenset({"kg n/ha", "kg n ha-1", "kg n ha^-1"}),
    "inorganic_n_rate": frozenset({"kg n/ha", "kg n ha-1", "kg n ha^-1"}),
    "p_rate_kg_p2o5_ha": frozenset({"kg p2o5/ha", "kg p2o5 ha-1"}),
    "inorganic_p_rate": frozenset({"kg p2o5/ha", "kg p2o5 ha-1"}),
    "k_rate_kg_k2o_ha": frozenset({"kg k2o/ha", "kg k2o ha-1"}),
    "inorganic_k_rate": frozenset({"kg k2o/ha", "kg k2o ha-1"}),
}


def _normalized_unit(value: str) -> str:
    return _SPACE_RE.sub(" ", value.strip()).casefold()


def validate_numeric_rule_unit(
    field: str,
    unit: str,
    *,
    require_known: bool = False,
) -> None:
    """Fail closed unless a cleaning threshold has a known physical quantity."""

    accepted_units = _CANONICAL_NUMERIC_FIELD_UNITS.get(field)
    if accepted_units is None:
        if require_known:
            raise ValueError(
                f"Final cleaning field {field} has no verified quantity/unit contract"
            )
        return
    if _normalized_unit(unit) not in accepted_units:
        raise ValueError(f"Final cleaning field {field} has an incompatible rule unit")


@dataclass(frozen=True)
class FinalCleaningRule:
    """One pre-reviewed, source-specific final cleaning rule.

    ``statistic``, ``threshold``, ``grouping_scope``, and ``minimum_group_size``
    carry the decided `ELG-12` Option D distributional and influence thresholds.
    They are required together for the two grouped rule types and prohibited
    outright for the two row-local types, so a rule can never half-declare a
    threshold whose evaluation basis is unstated.
    """

    rule_id: str
    rule_type: str
    field: str | None
    raw_position: int | None
    unit: str
    lower_bound: float | None
    upper_bound: float | None
    lower_bound_inclusive: bool
    upper_bound_inclusive: bool
    values: tuple[str, ...]
    action: str
    reason_code: str
    statistic: str | None = None
    threshold: float | None = None
    grouping_scope: str | None = None
    minimum_group_size: int | None = None

    def __post_init__(self) -> None:
        if not self.rule_id.strip() or not self.reason_code.strip() or not self.unit.strip():
            raise ValueError("Final cleaning rules require identifiers, units, and reason codes")
        if self.rule_type not in _RULE_TYPES:
            raise ValueError(f"Unsupported final cleaning rule type: {self.rule_type!r}")
        if self.action not in _RULE_ACTIONS:
            raise ValueError(f"Unsupported final cleaning action: {self.action!r}")
        if (self.field is None) == (self.raw_position is None):
            raise ValueError("A final cleaning rule must identify exactly one canonical field or raw position")
        if self.raw_position is not None and self.raw_position < 1:
            raise ValueError("Final cleaning raw positions are one-based positive integers")
        if self.rule_type in _ROW_LOCAL_RULE_TYPES:
            self._validate_row_local_rule()
        else:
            self._validate_grouped_rule()

    def _validate_row_local_rule(self) -> None:
        if any(
            value is not None
            for value in (
                self.statistic,
                self.threshold,
                self.grouping_scope,
                self.minimum_group_size,
            )
        ):
            raise ValueError(
                f"Final cleaning rule type {self.rule_type!r} cannot declare "
                "distributional or influence threshold controls"
            )
        if self.rule_type == "numeric_outside_range":
            if self.lower_bound is None and self.upper_bound is None:
                raise ValueError("Numeric final cleaning rules require at least one reviewed bound")
            if self.values:
                raise ValueError("Numeric final cleaning rules cannot contain remark values")
            if (
                self.lower_bound is not None
                and self.upper_bound is not None
                and self.lower_bound > self.upper_bound
            ):
                raise ValueError("Final cleaning lower bounds cannot exceed upper bounds")
            if self.field is not None:
                validate_numeric_rule_unit(self.field, self.unit)
        elif not self.values:
            raise ValueError("Remark final cleaning rules require at least one reviewed value")

    def _validate_grouped_rule(self) -> None:
        allowed_statistics = (
            _DISTRIBUTIONAL_STATISTICS
            if self.rule_type == "distributional_outlier"
            else _INFLUENCE_STATISTICS
        )
        if self.statistic not in allowed_statistics:
            raise ValueError(
                f"Final cleaning rule type {self.rule_type!r} requires a reviewed "
                f"statistic from {sorted(allowed_statistics)}"
            )
        if (
            self.threshold is None
            or not math.isfinite(self.threshold)
            or self.threshold <= 0.0
        ):
            raise ValueError(
                f"Final cleaning rule type {self.rule_type!r} requires a finite positive threshold"
            )
        if self.grouping_scope not in _GROUPING_SCOPES:
            raise ValueError(
                f"Final cleaning rule type {self.rule_type!r} requires a reviewed "
                f"grouping scope from {sorted(_GROUPING_SCOPES)}"
            )
        if (
            self.minimum_group_size is None
            or self.minimum_group_size < _MINIMUM_DISTRIBUTIONAL_GROUP_SIZE
        ):
            raise ValueError(
                f"Final cleaning rule type {self.rule_type!r} requires a reviewed "
                f"minimum group size of at least {_MINIMUM_DISTRIBUTIONAL_GROUP_SIZE}"
            )
        if self.values or self.lower_bound is not None or self.upper_bound is not None:
            raise ValueError(
                f"Final cleaning rule type {self.rule_type!r} cannot carry fixed "
                "bounds or remark values"
            )
        if self.field is None or self.field not in _CANONICAL_NUMERIC_FIELD_UNITS:
            raise ValueError(
                f"Final cleaning rule type {self.rule_type!r} requires a canonical "
                "numeric field with a verified quantity contract"
            )
        if _normalized_unit(self.unit) != _DIMENSIONLESS_UNIT:
            raise ValueError(
                f"Final cleaning rule type {self.rule_type!r} thresholds are "
                f"multiples of a robust spread and must declare unit {_DIMENSIONLESS_UNIT!r}"
            )


@dataclass(frozen=True)
class SourceCleaningPolicy:
    """Approved ELG-12 cleaning policy for one registered source."""

    source_name: str
    policy_id: str
    review_id: str
    reviewed_by: str
    reviewed_on: str
    default_action: str
    untrimmed_sensitivity_required: bool
    rules: tuple[FinalCleaningRule, ...]

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.source_name,
                self.policy_id,
                self.review_id,
                self.reviewed_by,
                self.reviewed_on,
            )
        ):
            raise ValueError("Source cleaning policies require complete authority metadata")
        if self.default_action not in _DEFAULT_ACTIONS:
            raise ValueError("Final cleaning policies must retain records that match no reviewed rule")
        if not self.untrimmed_sensitivity_required:
            raise ValueError("ELG-12 policies must preserve an untrimmed sensitivity membership")
        if not self.rules:
            raise ValueError(
                "Final cleaning policies require at least one reviewed rule"
            )
        rule_ids = tuple(rule.rule_id for rule in self.rules)
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("Final cleaning rule identifiers must be unique within a source policy")


@dataclass(frozen=True)
class FinalCleaningResult:
    """Full row ledger plus primary and untrimmed membership snapshots."""

    records: tuple[dict[str, Any], ...]
    decision_rows: tuple[dict[str, Any], ...]
    primary_record_uids: tuple[str, ...]
    untrimmed_record_uids: tuple[str, ...]


def _normalized_text(value: object) -> str:
    if value is None:
        return ""
    return _SPACE_RE.sub(" ", str(value).strip()).casefold()


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rule_value(record: Mapping[str, Any], rule: FinalCleaningRule) -> object:
    if rule.field is not None:
        return record.get(rule.field)
    raw_cells = record.get("raw_cells")
    if not isinstance(raw_cells, (tuple, list)) or rule.raw_position is None:
        return None
    index = rule.raw_position - 1
    return raw_cells[index] if index < len(raw_cells) else None


def _rule_matches(record: Mapping[str, Any], rule: FinalCleaningRule) -> bool:
    value = _rule_value(record, rule)
    if rule.rule_type == "remark_match":
        reviewed_values = {_normalized_text(item) for item in rule.values}
        return _normalized_text(value) in reviewed_values

    number = _finite_number(value)
    if number is None:
        return False
    below = False
    above = False
    if rule.lower_bound is not None:
        below = (
            number < rule.lower_bound
            if rule.lower_bound_inclusive
            else number <= rule.lower_bound
        )
    if rule.upper_bound is not None:
        above = (
            number > rule.upper_bound
            if rule.upper_bound_inclusive
            else number >= rule.upper_bound
        )
    return below or above


def _decision_row(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "record_uid": record.get("record_uid"),
        "source_name": record.get("source_name"),
        "cleaning_policy_id": record.get("cleaning_policy_id"),
        "cleaning_review_id": record.get("cleaning_review_id"),
        "cleaning_reviewer": record.get("cleaning_reviewer"),
        "cleaning_reviewed_on": record.get("cleaning_reviewed_on"),
        "cleaning_rule_ids": record.get("cleaning_rule_ids", ()),
        "cleaning_reason_codes": record.get("cleaning_reason_codes", ()),
        "cleaning_review_status": record.get("cleaning_review_status"),
        "final_analytical_membership_status": record.get(
            "final_analytical_membership_status"
        ),
        "untrimmed_sensitivity_membership_status": record.get(
            "untrimmed_sensitivity_membership_status"
        ),
    }


def apply_final_cleaning(
    records: Iterable[Mapping[str, Any]],
    policies: Mapping[str, SourceCleaningPolicy],
) -> FinalCleaningResult:
    """Apply only approved rules while preserving every row for audit and sensitivity.

    Missing policy coverage is a review hold. It never becomes an implicit deletion.
    Raw values are deliberately excluded from the decision ledger.
    """

    cleaned: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for source_record in records:
        record = dict(source_record)
        record_uid = record.get("record_uid")
        source_name = record.get("source_name")
        if not isinstance(record_uid, str) or not record_uid:
            raise ValueError("Final cleaning requires a stable record_uid on every record")
        policy = policies.get(str(source_name))
        if policy is None:
            record.update(
                {
                    "cleaning_policy_id": None,
                    "cleaning_review_id": None,
                    "cleaning_reviewer": None,
                    "cleaning_reviewed_on": None,
                    "cleaning_rule_ids": (),
                    "cleaning_reason_codes": ("FINAL_CLEANING_POLICY_REQUIRED",),
                    "cleaning_review_status": "review_required",
                    "final_analytical_membership_status": "review_required",
                    "untrimmed_sensitivity_membership_status": "included",
                }
            )
        else:
            matches = tuple(rule for rule in policy.rules if _rule_matches(record, rule))
            actions = {rule.action for rule in matches}
            if "exclude_primary" in actions:
                membership = "excluded"
                review_status = "resolved_excluded"
                record["analytical_record_status"] = "excluded_by_final_cleaning"
            elif "flag_only" in actions:
                membership = "included_flagged"
                review_status = "resolved_flagged"
            else:
                membership = "included"
                review_status = "resolved_retained"
            record.update(
                {
                    "cleaning_policy_id": policy.policy_id,
                    "cleaning_review_id": policy.review_id,
                    "cleaning_reviewer": policy.reviewed_by,
                    "cleaning_reviewed_on": policy.reviewed_on,
                    "cleaning_rule_ids": tuple(sorted(rule.rule_id for rule in matches)),
                    "cleaning_reason_codes": tuple(
                        sorted(rule.reason_code for rule in matches)
                    ),
                    "cleaning_review_status": review_status,
                    "final_analytical_membership_status": membership,
                    "untrimmed_sensitivity_membership_status": "included",
                }
            )
        cleaned.append(record)
        decisions.append(_decision_row(record))

    ordered = tuple(sorted(cleaned, key=lambda row: str(row["record_uid"])))
    decision_rows = tuple(sorted(decisions, key=lambda row: str(row["record_uid"])))
    primary_uids = tuple(
        str(row["record_uid"])
        for row in ordered
        if row.get("final_analytical_membership_status")
        in {"included", "included_flagged"}
        and row.get("analytical_record_status") == "included"
    )
    untrimmed_uids = tuple(str(row["record_uid"]) for row in ordered)
    return FinalCleaningResult(
        records=ordered,
        decision_rows=decision_rows,
        primary_record_uids=primary_uids,
        untrimmed_record_uids=untrimmed_uids,
    )


__all__ = [
    "FINAL_CLEANING_METADATA_FIELDS",
    "FinalCleaningResult",
    "FinalCleaningRule",
    "SourceCleaningPolicy",
    "apply_final_cleaning",
]
