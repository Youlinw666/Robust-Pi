"""Map-style dataset wrapper for the audited M5 augmentation protocol."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .augmentation import (
    LanguageVariant,
    VisualAugmentation,
    choose_language_variant,
    validate_language_bank,
)


class AugmentedActionChunkDataset:
    """Apply optional M5 transforms while preserving action-chunk labels.

    The wrapped dataset must return the project action-chunk keys ``image``,
    ``wrist_image``, ``state``, ``actions``, ``prompt``, ``task_id``,
    ``episode_index`` and ``frame_index``. Only the two images and prompt are
    replaced; state, actions and identity fields are carried through intact.
    """

    def __init__(
        self,
        dataset: Sequence,
        *,
        visual: VisualAugmentation | None = None,
        language_variants: Sequence[LanguageVariant] = (),
        language_split: str = "train",
        language_seed: int = 20261021,
    ) -> None:
        self.dataset = dataset
        self.visual = visual
        self.language_variants = tuple(language_variants)
        self.language_split = language_split
        self.language_seed = int(language_seed)
        if self.language_variants:
            validate_language_bank(self.language_variants)
            if language_split not in {"train", "validation", "test"}:
                raise ValueError(f"Unknown language split: {language_split}")

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _sample_id(sample: dict[str, Any]) -> str:
        try:
            task_id = int(sample["task_id"])
            episode_id = int(sample["episode_index"])
            frame_id = int(sample["frame_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("augmented samples require task_id/episode_index/frame_index") from exc
        return f"task{task_id}:episode{episode_id}:frame{frame_id}"

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        required = {
            "image",
            "wrist_image",
            "state",
            "actions",
            "prompt",
            "task_id",
            "episode_index",
            "frame_index",
        }
        missing = required - sample.keys()
        if missing:
            raise ValueError(f"augmented sample is missing fields: {sorted(missing)}")
        sample_id = self._sample_id(sample)
        if self.visual is not None:
            sample["image"] = self.visual.apply(sample["image"], sample_id=sample_id, view_id="base")
            sample["wrist_image"] = self.visual.apply(
                sample["wrist_image"], sample_id=sample_id, view_id="wrist"
            )
        if self.language_variants:
            variant = choose_language_variant(
                self.language_variants,
                task_id=int(sample["task_id"]),
                split=self.language_split,
                sample_id=sample_id,
                seed=self.language_seed,
            )
            sample["prompt"] = variant.text
        return sample
