"""Deterministic metrics aggregation for episode JSONL evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .runner import STATUSES


VALID_EPISODE_STATUSES = frozenset({"success", "task_failure", "timeout", "safety_stop"})


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in sorted({str(row[key]) for row in rows}):
        result[value] = _summarize([row for row in rows if str(row[key]) == value])
    return result


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if row.get("status") not in STATUSES:
            raise ValueError(f"Unknown episode status: {row.get('status')}")
    valid = [row for row in rows if row["status"] in VALID_EPISODE_STATUSES]
    successes = sum(row["status"] == "success" for row in valid)
    system_errors = sum(row["status"] == "system_error" for row in rows)
    latencies = [float(value) for row in rows for value in row.get("latency_ms", [])]
    return {
        "episodes_total": len(rows),
        "episodes_valid": len(valid),
        "successes": successes,
        "success_rate": successes / len(valid) if valid else None,
        "system_errors": system_errors,
        "timeouts": sum(row["status"] == "timeout" for row in rows),
        "task_failures": sum(row["status"] == "task_failure" for row in rows),
        "safety_stops": sum(row["status"] == "safety_stop" for row in rows),
        "inference_latency_ms_p50": _percentile(latencies, 0.50),
        "inference_latency_ms_p95": _percentile(latencies, 0.95),
    }


def aggregate_results(rows: Iterable[dict[str, Any]], *, manifest_identity_hash: str | None = None) -> dict[str, Any]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot aggregate an empty result set")
    case_ids = [row.get("case_id") for row in materialized]
    if any(not case_id for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("Results must contain unique nonempty case_id values")
    for row in materialized:
        if row.get("status") not in STATUSES:
            raise ValueError(f"Unknown episode status: {row.get('status')}")
    summary = _summarize(materialized)
    output: dict[str, Any] = {
        "status": "COMPLETED" if summary["system_errors"] == 0 else "PARTIAL_WITH_SYSTEM_ERRORS",
        "manifest_identity_hash": manifest_identity_hash,
        "overall": summary,
        "by_task": _group(materialized, "task_id"),
        "by_condition": _group(materialized, "condition"),
        "by_checkpoint": _group(materialized, "checkpoint_id"),
    }
    return output


def aggregate_jsonl(path: str | Path, *, manifest_identity_hash: str | None = None) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return aggregate_results(rows, manifest_identity_hash=manifest_identity_hash)
