"""Uncertainty-aware, failure-explicit analysis of evaluation JSONL rows."""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping


VALID_STATUSES = frozenset({"success", "task_failure", "timeout", "safety_stop"})
ALL_STATUSES = VALID_STATUSES | {"system_error"}


def wilson_interval(successes: int, total: int, *, z: float = 1.96) -> tuple[float | None, float | None]:
    """Return a two-sided Wilson interval for a Bernoulli success rate."""
    if successes < 0 or total < 0 or successes > total:
        raise ValueError("successes and total must satisfy 0 <= successes <= total")
    if total == 0:
        return None, None
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def summarize_results(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot summarize an empty result set")
    statuses = [str(row.get("status")) for row in materialized]
    unknown = sorted(set(statuses) - ALL_STATUSES)
    if unknown:
        raise ValueError(f"Unknown statuses: {unknown}")
    valid = [row for row in materialized if row["status"] in VALID_STATUSES]
    successes = sum(row["status"] == "success" for row in valid)
    low, high = wilson_interval(successes, len(valid))
    return {
        "episodes_total": len(materialized),
        "episodes_valid": len(valid),
        "successes": successes,
        "success_rate": successes / len(valid) if valid else None,
        "success_rate_wilson95": [low, high],
        "system_errors": sum(row["status"] == "system_error" for row in materialized),
        "task_failures": sum(row["status"] == "task_failure" for row in materialized),
        "timeouts": sum(row["status"] == "timeout" for row in materialized),
        "safety_stops": sum(row["status"] == "safety_stop" for row in materialized),
    }


def compare_paired(
    left: Iterable[Mapping[str, Any]],
    right: Iterable[Mapping[str, Any]],
    *,
    identity_key: str = "case_id",
) -> dict[str, Any]:
    """Compare two checkpoints on the same valid case identities.

    The output reports discordant wins/losses and ties. It deliberately avoids
    claiming a significance test for tiny development sets.
    """
    left_by_id = {str(row[identity_key]): row for row in left}
    right_by_id = {str(row[identity_key]): row for row in right}
    overlap = sorted(set(left_by_id) & set(right_by_id))
    if not overlap:
        raise ValueError("Paired comparison requires at least one shared case identity")
    pairs = []
    for case_id in overlap:
        left_row, right_row = left_by_id[case_id], right_by_id[case_id]
        if left_row["status"] not in VALID_STATUSES or right_row["status"] not in VALID_STATUSES:
            continue
        pairs.append(
            {
                "case_id": case_id,
                "left_success": left_row["status"] == "success",
                "right_success": right_row["status"] == "success",
            }
        )
    left_wins = sum(pair["left_success"] and not pair["right_success"] for pair in pairs)
    right_wins = sum(pair["right_success"] and not pair["left_success"] for pair in pairs)
    return {
        "paired_cases": len(pairs),
        "left_wins": left_wins,
        "right_wins": right_wins,
        "ties": len(pairs) - left_wins - right_wins,
        "discordant_pairs": left_wins + right_wins,
        "pairs": pairs,
        "interpretation": "Descriptive paired comparison; no significance claim.",
    }


def build_analysis_report(
    rows: Iterable[Mapping[str, Any]],
    *,
    manifest_identity_hash: str | None = None,
    scope: str = "development_only",
) -> dict[str, Any]:
    """Build condition/checkpoint/task tables without hiding invalid runs."""
    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot analyze an empty result set")

    def group(key: str) -> dict[str, Any]:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in materialized:
            if key not in row:
                raise ValueError(f"Result row lacks grouping key {key!r}")
            groups[str(row[key])].append(row)
        return {name: summarize_results(groups[name]) for name in sorted(groups)}

    return {
        "status": "ANALYSIS_ONLY",
        "scope": scope,
        "manifest_identity_hash": manifest_identity_hash,
        "overall": summarize_results(materialized),
        "by_condition": group("condition"),
        "by_checkpoint": group("checkpoint_id"),
        "by_task": group("task_id"),
    }
