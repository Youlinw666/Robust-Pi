#!/usr/bin/env python3
"""Load one real base checkpoint and execute one JAX training step on GPU.

This is an integration smoke only.  It never creates a checkpoint, enables
Weights & Biases, or enters a long training loop.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OPENPI = ROOT / "external/openpi"
sys.path.insert(0, str(OPENPI / "scripts"))

from data_tools.dataset_adapter import ActionChunkDataset, DatasetAdapter
from scripts.m3_formal_loader_smoke import FormalSampleDataset


class DeviceCheckpointWeightLoader:
    """Restore base params directly as device arrays to avoid a host NumPy copy."""

    def __init__(self, params_path: Path, sharding):
        self.params_path = str(params_path)
        self.sharding = sharding

    def load(self, params):
        import flax.traverse_util as traverse_util
        import jax
        import jax.numpy as jnp
        from openpi.training import weight_loaders
        import orbax.checkpoint as ocp

        # The stock CheckpointWeightLoader restores the entire float32 tree as
        # NumPy first (about 13 GB for these bases), which exceeds this WSL
        # host's 15 GiB RAM before JAX can place the arrays on the GPU.  Orbax
        # supports placeholders in the requested tree; restore one leaf at a
        # time directly to bfloat16 device arrays and release each staging
        # buffer before reading the next leaf.
        checkpoint = Path(self.params_path).resolve()
        with ocp.PyTreeCheckpointer() as checkpointer:
            metadata = checkpointer.metadata(checkpoint)["params"]
            flat_metadata = traverse_util.flatten_dict(metadata)
            flat_loaded = {}
            for key, leaf_metadata in flat_metadata.items():
                item_params = traverse_util.unflatten_dict(
                    {
                        candidate: (
                            candidate_metadata
                            if candidate == key
                            else ocp.PLACEHOLDER
                        )
                        for candidate, candidate_metadata in flat_metadata.items()
                    }
                )
                item = {"params": item_params}
                restore_args = jax.tree.map(
                    lambda value: (
                        ocp.ArrayRestoreArgs(
                            restore_type=jax.Array,
                            dtype=jnp.bfloat16,
                            sharding=self.sharding,
                        )
                        if value is not ocp.PLACEHOLDER
                        else ocp.PLACEHOLDER
                    ),
                    item,
                )
                restored = checkpointer.restore(
                    checkpoint,
                    ocp.args.PyTreeRestore(
                        item=item, restore_args=restore_args
                    ),
                )["params"]
                flat_loaded[key] = traverse_util.flatten_dict(restored)[key]
                jax.block_until_ready(flat_loaded[key])
                del restored, item, item_params, restore_args
                gc.collect()
        loaded = traverse_util.unflatten_dict(flat_loaded)
        return weight_loaders._merge_params(
            loaded, params, missing_regex=".*lora.*"
        )


def gpu_memory() -> dict[str, int] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        used, total = result.stdout.strip().splitlines()[0].split(",")
        return {"used_mib": int(used.strip()), "total_mib": int(total.strip())}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _make_batch(model_name: str, shots: int, batch_size: int, mesh):
    import jax
    from openpi.models.model import Observation
    from openpi.shared import normalize
    from openpi.training import config as openpi_config
    from openpi.training.data_loader import TorchDataLoader, transform_dataset
    from openpi.training import sharding

    native = openpi_config.get_config(
        "pi0_libero_low_mem_finetune" if model_name == "pi0" else "pi05_libero"
    )
    formal_inventory = ROOT / "reports/m2_formal_inventory.json"
    manifest = ROOT / "manifests/m2_dataset_manifest.json"
    data_root = (
        ROOT
        / "data/libero_offset_diagnostic/"
        / "f13aa24a3da8c43c7225569f28c562979fa0e35a"
    )
    stats_path = (
        ROOT / "artifacts/m2_formal_norm_stats" / model_name / f"{shots}shot"
    )
    adapter = DatasetAdapter(
        data_root, formal_inventory, manifest=manifest, split="train", shots=shots
    )
    dataset = FormalSampleDataset(
        ActionChunkDataset(adapter, native.model.action_horizon)
    )
    data_config = native.data.create(ROOT / "artifacts/m3_loader_assets", native.model)
    data_config = dataclasses.replace(
        data_config,
        repo_id="robust-pi-formal",
        norm_stats=normalize.load(stats_path),
    )
    transformed = transform_dataset(dataset, data_config)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    loader = TorchDataLoader(
        transformed,
        local_batch_size=batch_size,
        num_batches=1,
        num_workers=0,
        shuffle=False,
        seed=20261021,
        framework="jax",
        sharding=data_sharding,
    )
    raw_batch = next(iter(loader))
    observation = Observation.from_dict(raw_batch)
    actions = raw_batch["actions"]
    return native, observation, actions, {
        "shots": shots,
        "batch_size": batch_size,
        "dataset_samples": len(dataset),
        "dataset_episodes": len(adapter.allowed_ids),
        "stats": str((stats_path / "norm_stats.json").relative_to(ROOT)),
        "actions_shape": list(actions.shape),
        "state_shape": list(observation.state.shape),
        "image_shapes": {
            key: list(value.shape) for key, value in observation.images.items()
        },
    }


def _make_config(model_name: str, native, base_params: Path, mesh):
    import jax
    import flax.nnx as nnx

    if model_name == "pi0":
        model = native.model
        freeze_filter = native.freeze_filter
    else:
        model = dataclasses.replace(
            native.model,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
        freeze_filter = model.get_freeze_filter()
    return dataclasses.replace(
        native,
        name=f"robust_pi_m3_{model_name}_gpu_smoke",
        exp_name=f"m3_gpu_smoke_{model_name}",
        model=model,
        weight_loader=DeviceCheckpointWeightLoader(
            base_params,
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
        ),
        freeze_filter=freeze_filter if freeze_filter is not None else nnx.Nothing(),
        batch_size=1,
        num_train_steps=1,
        num_workers=0,
        log_interval=1,
        save_interval=10_000,
        checkpoint_base_dir=str(ROOT / "artifacts/m3_gpu_smoke_checkpoints"),
        overwrite=True,
        resume=False,
        wandb_enabled=False,
    )


def run(model_name: str, shots: int, output: Path) -> dict:
    started = time.monotonic()
    report: dict = {
        "status": "STARTED",
        "model": model_name,
        "shots": shots,
        "weights_loaded": False,
        "training_started": False,
        "openpi_commit": "215abfb217dbac7d5f1273282331b9b1866c0479",
        "base_checkpoint": str(
            (ROOT / "checkpoints" / f"{model_name}_base" / "params").relative_to(ROOT)
        ),
        "gpu_before": gpu_memory(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import jax
        from openpi.training import sharding
        import train as openpi_train

        report["jax_backend"] = jax.default_backend()
        report["jax_devices"] = [str(device) for device in jax.devices()]
        if jax.default_backend() not in {"gpu", "cuda"}:
            raise RuntimeError(
                f"GPU smoke refuses non-GPU backend: {jax.default_backend()}"
            )
        mesh = sharding.make_mesh(1)
        native, observation, actions, batch_info = _make_batch(
            model_name, shots, 1, mesh
        )
        report["batch"] = batch_info
        config = _make_config(
            model_name,
            native,
            ROOT / "checkpoints" / f"{model_name}_base" / "params",
            mesh,
        )
        rng = jax.random.key(20261021)
        train_rng, init_rng = jax.random.split(rng)
        report["init_started_at"] = time.monotonic()
        train_state, train_state_sharding = openpi_train.init_train_state(
            config, init_rng, mesh, resume=False
        )
        jax.block_until_ready(train_state)
        report["weights_loaded"] = True
        report["init_seconds"] = time.monotonic() - report["init_started_at"]

        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
        )
        replicated_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec()
        )
        batch = (observation, actions)
        ptrain_step = jax.jit(
            lambda step_rng, state, current_batch: openpi_train.train_step(
                config, step_rng, state, current_batch
            ),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
        report["step_started_at"] = time.monotonic()
        train_state, info = ptrain_step(train_rng, train_state, batch)
        jax.block_until_ready((train_state, info))
        report["training_started"] = True
        report["step_seconds"] = time.monotonic() - report["step_started_at"]
        report["info"] = {
            key: float(np.asarray(value)) for key, value in info.items()
        }
        report["status"] = "PASSED"
    except Exception as exc:
        report.update(
            status="FAILED",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
    report["gpu_after"] = gpu_memory()
    report["elapsed_seconds"] = time.monotonic() - started
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("pi0", "pi05"), required=True)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.model, args.shots, args.output)
    if report["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
