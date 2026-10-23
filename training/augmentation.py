"""Deterministic, label-preserving visual and language augmentation primitives."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Sequence

import numpy as np


LANGUAGE_SPLITS = frozenset({"train", "validation", "test"})


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


@dataclass(frozen=True)
class VisualAugmentationConfig:
    """Only photometric transforms are permitted by this protocol."""

    probability: float = 0.5
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.9, 1.1)
    rgb_noise_std: float = 2.0
    seed: int = 20261021

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        for name, value in (
            ("brightness_range", self.brightness_range),
            ("contrast_range", self.contrast_range),
        ):
            if len(value) != 2 or value[0] <= 0 or value[0] > value[1]:
                raise ValueError(f"{name} must be a positive ordered pair")
        if self.rgb_noise_std < 0:
            raise ValueError("rgb_noise_std must be nonnegative")


class VisualAugmentation:
    """Apply seeded brightness/contrast/noise without changing geometry or labels."""

    def __init__(self, config: VisualAugmentationConfig):
        self.config = config

    def apply(self, image: np.ndarray, *, sample_id: str, view_id: str = "base") -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise ValueError("Visual augmentation expects uint8 HWC RGB input")
        if not sample_id:
            raise ValueError("sample_id is required for deterministic augmentation")
        rng = np.random.default_rng(_stable_seed(self.config.seed, sample_id, view_id))
        if rng.random() >= self.config.probability:
            return array.copy()
        output = array.astype(np.float32)
        brightness = rng.uniform(*self.config.brightness_range)
        contrast = rng.uniform(*self.config.contrast_range)
        output = (output * brightness - 127.5) * contrast + 127.5
        if self.config.rgb_noise_std:
            output += rng.normal(0.0, self.config.rgb_noise_std, size=output.shape)
        return np.clip(np.rint(output), 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class LanguageVariant:
    """An explicitly authored equivalent instruction.

    ``protected_phrases`` is required metadata rather than an inferred
    semantic representation.  The policy refuses a variant that omits one of
    these phrases, so paraphrases cannot silently change object, color, count,
    relation, target, or operation semantics.
    """

    task_id: int
    split: str
    template_id: str
    text: str
    protected_phrases: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.task_id < 0 or self.split not in LANGUAGE_SPLITS:
            raise ValueError("invalid task_id or language split")
        if not self.template_id or not self.text.strip():
            raise ValueError("template_id and text are required")
        lowered = self.text.casefold()
        missing = [phrase for phrase in self.protected_phrases if phrase.casefold() not in lowered]
        if missing:
            raise ValueError(
                f"Language variant {self.template_id} omits protected phrases: {missing}"
            )


def validate_language_bank(variants: Iterable[LanguageVariant]) -> dict[str, int]:
    """Check that authored templates are isolated across train/validation/test."""
    rows = list(variants)
    if not rows:
        raise ValueError("language bank cannot be empty")
    seen_keys: set[tuple[int, str, str]] = set()
    split_templates: dict[int, dict[str, set[str]]] = {}
    for variant in rows:
        key = (variant.task_id, variant.split, variant.template_id)
        if key in seen_keys:
            raise ValueError(f"duplicate language template: {key}")
        seen_keys.add(key)
        per_task = split_templates.setdefault(variant.task_id, {split: set() for split in LANGUAGE_SPLITS})
        per_task[variant.split].add(variant.template_id)
    for task_id, per_task in split_templates.items():
        splits = list(per_task)
        for index, left in enumerate(splits):
            for right in splits[index + 1 :]:
                overlap = per_task[left] & per_task[right]
                if overlap:
                    raise ValueError(
                        f"task {task_id} reuses template IDs across {left}/{right}: {sorted(overlap)}"
                    )
    return {
        split: sum(1 for variant in rows if variant.split == split)
        for split in sorted(LANGUAGE_SPLITS)
    }


def choose_language_variant(
    variants: Sequence[LanguageVariant],
    *,
    task_id: int,
    split: str,
    sample_id: str,
    seed: int,
) -> LanguageVariant:
    """Choose one authored variant deterministically within a split."""
    validate_language_bank(variants)
    candidates = sorted(
        (
            variant
            for variant in variants
            if variant.task_id == task_id and variant.split == split
        ),
        key=lambda variant: variant.template_id,
    )
    if not candidates:
        raise ValueError(f"No language variants for task {task_id} split {split}")
    index = _stable_seed(seed, task_id, split, sample_id) % len(candidates)
    return candidates[index]
