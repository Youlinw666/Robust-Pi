"""Immutable case identities and model-specific checkpoint bindings."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
MANIFEST_STATUSES = frozenset({"DRAFT", "READY"})
SPLITS = frozenset({"development", "final"})
CONDITIONS = frozenset({"id", "visual_ood", "language_ood", "visual_language_ood"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class EvalCase:
    """One fixed episode specification shared by every evaluated checkpoint."""

    case_id: str
    task_id: int
    init_state_id: int
    prompt_id: str
    perturbation_id: str
    flow_seed: int
    checkpoint_id: str
    max_steps: int
    replan_steps: int
    condition: str = "id"

    def __post_init__(self) -> None:
        if not self.case_id or not self.prompt_id or not self.perturbation_id or not self.checkpoint_id:
            raise ValueError("case_id, prompt_id, perturbation_id, and checkpoint_id are required")
        if self.task_id < 0 or self.init_state_id < 0 or self.flow_seed < 0:
            raise ValueError("task_id, init_state_id, and flow_seed must be nonnegative")
        if self.max_steps < 1 or self.replan_steps < 1 or self.replan_steps > self.max_steps:
            raise ValueError("Require 1 <= replan_steps <= max_steps")
        if self.condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {self.condition}")

    def identity_dict(self) -> dict[str, Any]:
        """Return fields that must match across model/checkpoint runs."""
        return {
            "case_id": self.case_id,
            "task_id": self.task_id,
            "init_state_id": self.init_state_id,
            "prompt_id": self.prompt_id,
            "perturbation_id": self.perturbation_id,
            "flow_seed": self.flow_seed,
            "max_steps": self.max_steps,
            "replan_steps": self.replan_steps,
            "condition": self.condition,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_dict(), "checkpoint_id": self.checkpoint_id}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvalCase":
        fields = {
            "case_id",
            "task_id",
            "init_state_id",
            "prompt_id",
            "perturbation_id",
            "flow_seed",
            "checkpoint_id",
            "max_steps",
            "replan_steps",
            "condition",
        }
        missing = fields - value.keys()
        if missing:
            raise ValueError(f"Case is missing fields: {sorted(missing)}")
        return cls(**{field: value[field] for field in fields})


@dataclass(frozen=True)
class CaseManifest:
    """A persisted development/final case set with stable identity hashes."""

    run_id: str
    suite: str
    split: str
    status: str
    cases: tuple[EvalCase, ...]
    created_at_utc: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id or not self.suite or not self.created_at_utc:
            raise ValueError("run_id, suite, and created_at_utc are required")
        if self.split not in SPLITS:
            raise ValueError(f"Unknown split: {self.split}")
        if self.status not in MANIFEST_STATUSES:
            raise ValueError(f"Unknown manifest status: {self.status}")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported case manifest schema: {self.schema_version}")
        if not self.cases:
            raise ValueError("A case manifest must contain at least one case")
        ids = [case.case_id for case in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate case_id in manifest")
        identity_ids = [case.identity_dict()["case_id"] for case in self.cases]
        if identity_ids != sorted(identity_ids):
            raise ValueError("Cases must be sorted by case_id for deterministic manifests")

    @property
    def identity_hash(self) -> str:
        return _sha256([case.identity_dict() for case in self.cases])

    @property
    def manifest_hash(self) -> str:
        return _sha256(self.to_dict(include_hashes=False))

    def to_dict(self, *, include_hashes: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "suite": self.suite,
            "split": self.split,
            "status": self.status,
            "created_at_utc": self.created_at_utc,
            "cases": [case.to_dict() for case in self.cases],
        }
        if include_hashes:
            value["identity_hash"] = self.identity_hash
            value["manifest_hash"] = self.manifest_hash
        return value

    def bind_checkpoint(self, checkpoint_id: str, *, run_id: str | None = None) -> "CaseManifest":
        """Bind one checkpoint while preserving the exact case identity set."""
        if not checkpoint_id:
            raise ValueError("checkpoint_id is required")
        return replace(
            self,
            run_id=run_id or self.run_id,
            status="READY",
            cases=tuple(replace(case, checkpoint_id=checkpoint_id) for case in self.cases),
        )

    def write(self, path: str | Path, *, overwrite: bool = False) -> None:
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite case manifest: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n")
        temp.replace(path)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CaseManifest":
        required = {"schema_version", "run_id", "suite", "split", "status", "created_at_utc", "cases"}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"Manifest is missing fields: {sorted(missing)}")
        manifest = cls(
            schema_version=int(value["schema_version"]),
            run_id=str(value["run_id"]),
            suite=str(value["suite"]),
            split=str(value["split"]),
            status=str(value["status"]),
            created_at_utc=str(value["created_at_utc"]),
            cases=tuple(EvalCase.from_dict(case) for case in value["cases"]),
        )
        if "identity_hash" in value and value["identity_hash"] != manifest.identity_hash:
            raise ValueError("Case identity hash mismatch")
        if "manifest_hash" in value and value["manifest_hash"] != manifest.manifest_hash:
            raise ValueError("Case manifest hash mismatch")
        return manifest

    @classmethod
    def read(cls, path: str | Path) -> "CaseManifest":
        return cls.from_dict(json.loads(Path(path).read_text()))


def make_level_a_development_manifest(
    *,
    run_id: str = "m4_level_a_dev",
    checkpoint_id: str = "PENDING_M3_CHECKPOINT",
    created_at_utc: str = "2026-10-21T00:00:00Z",
) -> CaseManifest:
    """Create a draft five-task, four-condition development matrix.

    Task and initial-state identifiers are protocol inputs; runtime LIBERO
    validation is intentionally separate and is required before READY status.
    """
    cases: list[EvalCase] = []
    for task_id in range(5):
        for condition in ("id", "visual_ood", "language_ood", "visual_language_ood"):
            cases.append(
                EvalCase(
                    case_id=f"spatial_t{task_id:02d}_init00_{condition}",
                    task_id=task_id,
                    init_state_id=0,
                    prompt_id="canonical",
                    perturbation_id=condition,
                    flow_seed=20261021 + task_id,
                    checkpoint_id=checkpoint_id,
                    max_steps=220,
                    replan_steps=5,
                    condition=condition,
                )
            )
    return CaseManifest(
        run_id=run_id,
        suite="libero_spatial",
        split="development",
        status="DRAFT",
        cases=tuple(sorted(cases, key=lambda case: case.case_id)),
        created_at_utc=created_at_utc,
    )
