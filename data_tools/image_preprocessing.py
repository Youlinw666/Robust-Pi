"""Explicit CPU image contract for newly prepared data and future rollouts.

Raw OpenGL images rotate by 180 degrees. Canonical images are copied unchanged.
Resize, if supplied, runs AFTER orientation. State/actions are never rotated.
Historical conversion and rollout entry points are deliberately left unchanged.
"""
from __future__ import annotations

import numpy as np

CONTRACT = "robust_pi_canonical_rot180_v1"


def canonicalize_rgb(image, *, source_orientation: str) -> np.ndarray:
    if source_orientation not in ("raw_opengl", "canonical"):
        raise ValueError("source_orientation must be raw_opengl or canonical")
    value = np.asarray(image)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3 or min(value.shape[:2]) < 1:
        raise ValueError("Expected nonempty uint8 HWC RGB")
    if source_orientation == "raw_opengl":
        value = value[::-1, ::-1]
    return value.copy(order="C")


def prepare_camera_pair(base, wrist, *, source_orientation: str, resize=None):
    images = tuple(canonicalize_rgb(x, source_orientation=source_orientation) for x in (base, wrist))
    return tuple(resize(x) for x in images) if resize is not None else images


def rollout_policy_inputs(observation, state, prompt, *, source_orientation: str, resize=None):
    """For future rollouts, replace the old inline rotation with this function.

    Pass raw camera arrays once; never rotate before calling with raw_opengl.
    Leaving resize=None delegates 224 padding/resize to official model transforms.
    """
    base, wrist = prepare_camera_pair(observation["agentview_image"], observation["robot0_eye_in_hand_image"],
                                      source_orientation=source_orientation, resize=resize)
    state = np.asarray(state)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Expected finite state[8]")
    return {"observation/image": base, "observation/wrist_image": wrist,
            "observation/state": state.copy(), "prompt": prompt}
