"""Resumable episode result writer for LIBERO evaluation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from typing import Any, Callable, Mapping

from .case_manifest import CaseManifest, EvalCase


STATUSES = frozenset({"success", "task_failure", "timeout", "system_error", "safety_stop"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EpisodeResult:
    run_id: str
    case_id: str
    task_id: int
    condition: str
    checkpoint_id: str
    status: str
    action_steps: int
    inference_count: int
    latency_ms: list[float]
    started_at_utc: str
    ended_at_utc: str
    video_path: str | None = None
    error_type: str | None = None
    error: str | None = None
    traceback: str | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"Unknown episode status: {self.status}")
        if self.action_steps < 0 or self.inference_count < 0:
            raise ValueError("action_steps and inference_count must be nonnegative")
        if any(value < 0 for value in self.latency_ms):
            raise ValueError("latency_ms cannot contain negative values")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvalRunner:
    """Execute cases through a callback while preserving resumable JSONL evidence.

    The callback must return a mapping with ``status`` and optional result fields.
    Environment/model code stays outside this CPU-testable protocol layer.
    """

    def __init__(self, manifest: CaseManifest, output_dir: str | Path, *, resume: bool = True):
        self.manifest = manifest
        self.output_dir = Path(output_dir)
        self.results_path = self.output_dir / "episodes.jsonl"
        self.state_path = self.output_dir / "state.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "case_manifest.json"
        if manifest_path.exists():
            existing = CaseManifest.read(manifest_path)
            if existing.identity_hash != manifest.identity_hash:
                raise ValueError("Existing output directory belongs to a different case identity set")
            if existing.manifest_hash != manifest.manifest_hash:
                raise ValueError("Existing output directory belongs to a different bound manifest")
        else:
            manifest.write(manifest_path)
        self.completed: dict[str, dict[str, Any]] = {}
        if resume and self.results_path.exists():
            for line_number, line in enumerate(self.results_path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                case_id = row.get("case_id")
                if case_id not in {case.case_id for case in manifest.cases}:
                    raise ValueError(f"Result line {line_number} has unknown case_id: {case_id}")
                if case_id in self.completed:
                    raise ValueError(f"Duplicate result for case_id: {case_id}")
                self.completed[case_id] = row

    def _append(self, result: EpisodeResult) -> None:
        with self.results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            stream.flush()
        state = {
            "run_id": self.manifest.run_id,
            "manifest_identity_hash": self.manifest.identity_hash,
            "completed_cases": sorted(set(self.completed) | {result.case_id}),
            "last_case_id": result.case_id,
            "status": "RUNNING",
        }
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temp.replace(self.state_path)

    def run(self, execute: Callable[[EvalCase], Mapping[str, Any]]) -> list[dict[str, Any]]:
        for case in self.manifest.cases:
            if case.case_id in self.completed:
                continue
            started = _now()
            try:
                payload = dict(execute(case))
                status = str(payload.pop("status"))
                result = EpisodeResult(
                    run_id=self.manifest.run_id,
                    case_id=case.case_id,
                    task_id=case.task_id,
                    condition=case.condition,
                    checkpoint_id=case.checkpoint_id,
                    status=status,
                    started_at_utc=started,
                    ended_at_utc=_now(),
                    action_steps=int(payload.pop("action_steps", 0)),
                    inference_count=int(payload.pop("inference_count", 0)),
                    latency_ms=[float(value) for value in payload.pop("latency_ms", [])],
                    video_path=payload.pop("video_path", None),
                    error_type=payload.pop("error_type", None),
                    error=payload.pop("error", None),
                    traceback=payload.pop("traceback", None),
                )
                if payload:
                    raise ValueError(f"Executor returned unsupported fields: {sorted(payload)}")
            except Exception as exc:
                result = EpisodeResult(
                    run_id=self.manifest.run_id,
                    case_id=case.case_id,
                    task_id=case.task_id,
                    condition=case.condition,
                    checkpoint_id=case.checkpoint_id,
                    status="system_error",
                    action_steps=0,
                    inference_count=0,
                    latency_ms=[],
                    started_at_utc=started,
                    ended_at_utc=_now(),
                    error_type=type(exc).__name__,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
            row = result.to_dict()
            self.completed[case.case_id] = row
            self._append(result)
        state = {
            "run_id": self.manifest.run_id,
            "manifest_identity_hash": self.manifest.identity_hash,
            "completed_cases": sorted(self.completed),
            "status": "COMPLETED" if len(self.completed) == len(self.manifest.cases) else "PARTIAL",
        }
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temp.replace(self.state_path)
        return [self.completed[case.case_id] for case in self.manifest.cases if case.case_id in self.completed]
