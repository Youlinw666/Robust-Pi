#!/usr/bin/env python3
"""Run one official openpi data-loader batch from the formal M2 source."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_tools.dataset_adapter import ActionChunkDataset, DatasetAdapter


class FormalSampleDataset:
    """Adapt the project sample keys to the official LIBERO repack contract."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        value = self.dataset[index]
        return {
            # LeRobotLiberoDataConfig's RepackTransform maps these source
            # names to observation/image, observation/wrist_image, and state.
            "image": value["image"],
            "wrist_image": value["wrist_image"],
            "state": value["state"],
            "actions": value["actions"],
            "prompt": value["prompt"],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("pi0", "pi05"), default="pi05")
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refuse to overwrite output " + str(args.output))

    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation
    from openpi.shared import normalize
    from openpi.training import config as openpi_config
    from openpi.training.data_loader import TorchDataLoader, transform_dataset

    model_name = "pi0_libero_low_mem_finetune" if args.model == "pi0" else "pi05_libero"
    native = openpi_config.get_config(model_name)
    formal_inventory = ROOT / "reports/m2_formal_inventory.json"
    manifest = ROOT / "manifests/m2_dataset_manifest.json"
    data_root = ROOT / "data/libero_offset_diagnostic/f13aa24a3da8c43c7225569f28c562979fa0e35a"
    stats_path = ROOT / "artifacts/m2_formal_norm_stats" / args.model / "{}shot".format(args.shots)
    adapter = DatasetAdapter(data_root, formal_inventory, manifest=manifest, split="train", shots=args.shots)
    dataset = FormalSampleDataset(ActionChunkDataset(adapter, native.model.action_horizon))

    # Reuse official LIBERO repack/data/model transforms, injecting the
    # persisted pool-specific statistics instead of any upstream dataset.
    data_config = native.data.create(ROOT / "artifacts/m3_loader_assets", native.model)
    data_config = dataclasses.replace(
        data_config,
        repo_id="robust-pi-formal",
        norm_stats=normalize.load(stats_path),
    )
    transformed = transform_dataset(dataset, data_config)
    data_loader = TorchDataLoader(
        transformed,
        local_batch_size=args.batch_size,
        num_batches=args.num_batches,
        num_workers=0,
        shuffle=False,
        seed=20261020,
        framework="jax",
    )
    batch = next(iter(data_loader))
    observation = Observation.from_dict(batch)
    if jax.default_backend() != "cpu":
        raise RuntimeError("This loader smoke is explicitly CPU-only")
    if batch["actions"].shape[-2:] != (native.model.action_horizon, native.model.action_dim):
        raise RuntimeError("Unexpected action shape {}".format(batch["actions"].shape))
    if observation.state.shape[-1] != native.model.action_dim:
        raise RuntimeError("Unexpected padded state shape {}".format(observation.state.shape))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "PASSED",
        "model": args.model,
        "config": model_name,
        "shots": args.shots,
        "batch_size": args.batch_size,
        "dataset_samples": len(dataset),
        "dataset_episodes": len(adapter.allowed_ids),
        "jax_backend": jax.default_backend(),
        "model_action_horizon": native.model.action_horizon,
        "model_action_dim": native.model.action_dim,
        "actions": {"shape": list(batch["actions"].shape), "dtype": str(batch["actions"].dtype)},
        "state": {"shape": list(observation.state.shape), "dtype": str(observation.state.dtype)},
        "images": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in observation.images.items()
        },
        "tokens": {
            "shape": list(observation.tokenized_prompt.shape),
            "dtype": str(observation.tokenized_prompt.dtype),
        },
        "formal_inventory": str(formal_inventory.relative_to(ROOT)),
        "manifest": str(manifest.relative_to(ROOT)),
        "stats": str((stats_path / "norm_stats.json").relative_to(ROOT)),
        "weights_loaded": False,
        "training_started": False,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
