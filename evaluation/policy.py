"""Model-agnostic policy adapter contract for LIBERO evaluation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PolicySpec:
    model_id: str
    checkpoint_id: str
    action_horizon: int
    flow_steps: int
    internal_action_dim: int = 32
    libero_action_dim: int = 7

    def __post_init__(self) -> None:
        if not self.model_id or not self.checkpoint_id:
            raise ValueError("model_id and checkpoint_id are required")
        if self.action_horizon < 1 or self.flow_steps < 1:
            raise ValueError("action_horizon and flow_steps must be positive")
        if self.internal_action_dim < self.libero_action_dim:
            raise ValueError("Internal action dimension cannot be smaller than LIBERO dimension")


class PolicyAdapter(ABC):
    """Common H/K, timing, and 32-to-7 action contract.

    Concrete adapters may wrap an openpi policy server or an in-process policy.
    The protocol layer performs the dimensionality check so internal padding can
    never be sent to ``env.step``.
    """

    def __init__(self, spec: PolicySpec):
        self.spec = spec

    @abstractmethod
    def _infer_raw(self, observation: dict[str, Any], prompt: str, *, flow_seed: int) -> Any:
        raise NotImplementedError

    def infer(self, observation: dict[str, Any], prompt: str, *, flow_seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter()
        raw = self._infer_raw(observation, prompt, flow_seed=flow_seed)
        actions = np.asarray(raw)
        if actions.ndim != 2 or actions.shape[0] < self.spec.action_horizon:
            raise ValueError(
                f"Policy returned {actions.shape}; expected at least "
                f"({self.spec.action_horizon}, {self.spec.internal_action_dim})"
            )
        if actions.shape[1] < self.spec.libero_action_dim:
            raise ValueError(f"Policy returned too few action dimensions: {actions.shape}")
        actions = np.asarray(actions[: self.spec.action_horizon, : self.spec.libero_action_dim], dtype=np.float32)
        if not np.isfinite(actions).all():
            raise ValueError("Policy returned NaN/Inf actions")
        return actions, {
            "model_id": self.spec.model_id,
            "checkpoint_id": self.spec.checkpoint_id,
            "action_horizon": self.spec.action_horizon,
            "flow_steps": self.spec.flow_steps,
            "flow_seed": int(flow_seed),
            "libero_action_dim": self.spec.libero_action_dim,
            "inference_ms": (time.perf_counter() - started) * 1000.0,
        }
