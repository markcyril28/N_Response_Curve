from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


_TIERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class QcReport:
    """Reconciled, row-level quality-control evidence for the Phase 2 ledger."""

    inventory_rows: int
    tier_counts: Mapping[str, int]
    source_to_tier_counts: Mapping[str, Mapping[str, int]]
    reason_counts: Mapping[str, int]
    review_rows: tuple[dict[str, Any], ...]
    critical_record_uids: tuple[str, ...]
    reconciles: bool


def build_qc_report(
    ledger: Iterable[Mapping[str, Any]],
    *,
    inventory_record_uids: Iterable[str] | None = None,
) -> QcReport:
    """Build source-to-tier flows and non-lossy review rows from an eligibility ledger."""

    rows = [dict(row) for row in ledger]
    if inventory_record_uids is None:
        raise ValueError("inventory_record_uids is required for authoritative QC reconciliation")
    record_uids = [str(row.get("record_uid", "")) for row in rows]
    if not all(record_uids):
        raise ValueError("Every QC ledger row needs a nonempty record_uid")
    if len(record_uids) != len(set(record_uids)):
        raise ValueError("QC ledger contains a duplicate record_uid")
    inventory_uids = tuple(str(record_uid) for record_uid in inventory_record_uids)
    if not all(inventory_uids) or len(inventory_uids) != len(set(inventory_uids)):
        raise ValueError("Canonical inventory record identifiers must be nonempty and unique")
    if set(record_uids) != set(inventory_uids):
        raise ValueError("QC ledger does not match the canonical inventory")

    tier_counts = {tier: 0 for tier in _TIERS}
    source_to_tier_counts: dict[str, dict[str, int]] = {}
    reason_counts: dict[str, int] = {}
    review_rows: list[dict[str, Any]] = []
    critical_record_uids: list[str] = []

    for row in rows:
        tier = row.get("eligibility_tier")
        if tier not in _TIERS:
            raise ValueError(f"Unknown eligibility tier: {tier!r}")
        source_name = str(row.get("source_name", "")).strip()
        if not source_name:
            raise ValueError("Every QC ledger row needs a nonempty source_name")
        tier_counts[tier] += 1
        source_counts = source_to_tier_counts.setdefault(source_name, {item: 0 for item in _TIERS})
        source_counts[tier] += 1

        reasons = tuple(row.get("eligibility_reason_codes", ()))
        if any(not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError("Eligibility reason codes must be nonempty strings")
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if tier != "A" or set(reasons) != {"PRIMARY_ELIGIBLE"}:
            review_rows.append(row)
        if bool(row.get("has_critical_error")):
            critical_record_uids.append(str(row["record_uid"]))

    source_total = sum(sum(counts.values()) for counts in source_to_tier_counts.values())
    reconciles = (
        sum(tier_counts.values()) == len(inventory_uids)
        and source_total == len(inventory_uids)
        and set(record_uids) == set(inventory_uids)
    )
    return QcReport(
        inventory_rows=len(inventory_uids),
        tier_counts=tier_counts,
        source_to_tier_counts={source: source_to_tier_counts[source] for source in sorted(source_to_tier_counts)},
        reason_counts={reason: reason_counts[reason] for reason in sorted(reason_counts)},
        review_rows=tuple(review_rows),
        critical_record_uids=tuple(critical_record_uids),
        reconciles=reconciles,
    )


__all__ = ["QcReport", "build_qc_report"]
