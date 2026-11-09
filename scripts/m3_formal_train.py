#!/usr/bin/env python3
"""Run guarded historical M3 pilots or an explicit canonical-data training plan.

This entry point is intentionally guarded.  It does not run unless the caller
provides a human approval reference and ``--allow-compute``.  It uses the
pinned openpi JAX/Flax train step, the project's immutable manifest/statistics,
and a streamed base-weight restore that fits the local WSL host.

--training-plan binds accepted canonical data; --preflight checks that exact
training path on CPU without model creation or parameter restoration.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OPENPI = ROOT / "external/openpi"
sys.path.insert(0, str(OPENPI / "scripts"))

from data_tools.dataset_adapter import ActionChunkDataset, DatasetAdapter
from scripts.m3_formal_loader_smoke import FormalSampleDataset
from scripts.m3_gpu_smoke import DeviceCheckpointWeightLoader
from training import data_binding


FORMAL_PILOT_SCOPE = {
    "shots": 40,
    "steps": 100,
    "batch_size": 1,
    "seed": 20261021,
}


class _LoaderProxy:
    """Minimal official DataLoader contract needed by checkpoint saving."""

    def __init__(self, data_config):
        self._data_config = data_config

    def data_config(self):
        return self._data_config


def _display_path(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def _build_data(model_name: str, shots: int, batch_size: int, mesh, seed: int,
                dataset_config=None, split="train"):
    import jax
    from openpi.models.model import Observation
    from openpi.shared import normalize
    from openpi.training import config as openpi_config
    from openpi.training.data_loader import TorchDataLoader, transform_dataset
    from openpi.training import sharding

    native = openpi_config.get_config(
        "pi0_libero_low_mem_finetune" if model_name == "pi0" else "pi05_libero"
    )
    inventory = ROOT / "reports/m2_formal_inventory.json"
    manifest = ROOT / "manifests/m2_dataset_manifest.json"
    data_root = ROOT / "data/libero_offset_diagnostic/f13aa24a3da8c43c7225569f28c562979fa0e35a"
    stats_path = ROOT / "artifacts/m2_formal_norm_stats" / model_name / f"{shots}shot"
    if dataset_config is not None:
        paths = data_binding.canonical_paths(dataset_config, model=model_name, shots=shots)
        inventory, manifest, data_root = paths["inventory"], paths["manifest"], paths["data_root"]
        stats_path = paths["norm_stats"].parent
    elif split != "train":
        raise ValueError("Validation binding requires an explicit canonical dataset config")
    adapter = DatasetAdapter(
        data_root,
        inventory,
        manifest=manifest,
        split=split,
        shots=shots if split == "train" else None,
    )
    dataset = FormalSampleDataset(
        ActionChunkDataset(adapter, native.model.action_horizon)
    )
    data_config = native.data.create(ROOT / "artifacts/m3_loader_assets", native.model)
    data_config = dataclasses.replace(
        data_config,
        repo_id="robust-pi-canonical-v1" if dataset_config else "robust-pi-formal",
        norm_stats=normalize.load(stats_path),
    )
    transformed = transform_dataset(dataset, data_config)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    loader = TorchDataLoader(
        transformed,
        local_batch_size=batch_size,
        num_batches=None,
        num_workers=0,
        shuffle=(split == "train"),
        seed=seed,
        framework="jax",
        sharding=data_sharding,
    )

    def next_batch():
        raw = next(iterator)
        return Observation.from_dict(raw), raw["actions"]

    iterator = iter(loader)
    return native, data_config, _LoaderProxy(data_config), next_batch, {
        "inventory": _display_path(inventory),
        "manifest": _display_path(manifest),
        "stats": _display_path(stats_path / "norm_stats.json"),
        "data_root": _display_path(data_root),
        "dataset_config": str(dataset_config) if dataset_config else None,
        "split": split,
        "shots": shots if split == "train" else None,
        "normalization_shots": shots,
        "dataset_samples": len(dataset),
        "dataset_episodes": len(adapter.allowed_ids),
        "action_horizon": native.model.action_horizon,
        "action_dim": native.model.action_dim,
    }


def _make_config(
    model_name: str,
    native,
    base_params: Path,
    mesh,
    run_id: str,
    steps: int,
    seed: int,
    checkpoint_base_dir: Path,
    training_plan=None,
):
    import jax
    import flax.nnx as nnx

    if training_plan is not None:
        data_binding.validate_native(native, training_plan)
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
        name=f"robust_pi_m3_{model_name}_lora",
        exp_name=run_id,
        model=model,
        weight_loader=DeviceCheckpointWeightLoader(
            base_params,
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
        ),
        freeze_filter=freeze_filter if freeze_filter is not None else nnx.Nothing(),
        batch_size=1,
        num_train_steps=steps,
        num_workers=0,
        seed=seed,
        log_interval=1,
        save_interval=training_plan["save_interval"] if training_plan else steps,
        checkpoint_base_dir=str(checkpoint_base_dir),
        overwrite=False,
        resume=False,
        wandb_enabled=False,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def _initialize_low_memory_checkpoint_dir(checkpoints, checkpoint_dir: Path):
    """Create an inference checkpoint manager with bounded host-side writes.

    The M3 pilot needs a final inference checkpoint, not optimizer-resume
    state.  Saving only params and assets avoids the extra train-state copy
    that exhausted this 15 GiB host.  The params handler keeps the official
    Orbax format and caps concurrent serialization to 3 GiB, above the largest
    individual parameter shard.
    """
    import orbax.checkpoint as ocp

    checkpoint_dir = checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": checkpoints.CallbackHandler(),
            "params": ocp.PyTreeCheckpointHandler(save_concurrent_gb=3, restore_concurrent_gb=3),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=None,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )


def _save_inference_checkpoint(checkpoints, manager, state, data_loader, step: int) -> None:
    """Save params/assets required by ``create_trained_policy``."""
    from openpi.shared import normalize

    def save_assets(directory: Path) -> None:
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            normalize.save(directory / data_config.asset_id, norm_stats)

    params = state.ema_params if state.ema_params is not None else state.params
    manager.save(step, {"assets": save_assets, "params": {"params": params}})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_provenance(model_name: str, shots: int) -> dict[str, Any]:
    """Hash the small, immutable inputs that define a guarded pilot run."""
    stats_path = ROOT / "artifacts/m2_formal_norm_stats" / model_name / f"{shots}shot" / "norm_stats.json"
    paths = {
        "m2_formal_inventory": ROOT / "reports/m2_formal_inventory.json",
        "m2_dataset_manifest": ROOT / "manifests/m2_dataset_manifest.json",
        "norm_stats": stats_path,
        "m3_training_plan": ROOT / "configs/m3_formal_training_plan.json",
        "base_object_inventory": ROOT / f"environment/{model_name}_base_objects.json",
        "base_download_receipt": ROOT / "environment/m3_base_download_receipt.json",
    }
    missing = [str(path.relative_to(ROOT)) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing guarded pilot provenance files: {missing}")
    return {
        name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
    }


def validate_pilot_scope(args: argparse.Namespace) -> None:
    """Reject compute settings outside the explicitly budgeted E1/E2 pilot."""
    if args.shots != FORMAL_PILOT_SCOPE["shots"]:
        raise ValueError(
            f"Guarded M3 pilot requires --shots {FORMAL_PILOT_SCOPE['shots']}; "
            f"got {args.shots}"
        )
    if args.steps != FORMAL_PILOT_SCOPE["steps"]:
        raise ValueError(
            f"Guarded M3 pilot requires --steps {FORMAL_PILOT_SCOPE['steps']}; "
            f"got {args.steps}"
        )
    if args.batch_size != FORMAL_PILOT_SCOPE["batch_size"]:
        raise ValueError(
            f"Guarded M3 pilot requires --batch-size {FORMAL_PILOT_SCOPE['batch_size']}; "
            f"got {args.batch_size}"
        )
    if args.seed != FORMAL_PILOT_SCOPE["seed"]:
        raise ValueError(
            f"Guarded M3 pilot requires --seed {FORMAL_PILOT_SCOPE['seed']}; "
            f"got {args.seed}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_compute:
        raise RuntimeError("Refusing to train without --allow-compute")
    if not args.approval_ref.strip():
        raise RuntimeError("Refusing to train without --approval-ref")
    plan, paths = (data_binding.bind_args(args) if getattr(args, "training_plan", None) else (None, None))
    if plan is None:
        validate_pilot_scope(args)
        provenance = _source_provenance(args.model, args.shots)
    else:
        data_binding.verify_preflight(args.preflight_receipt, plan, paths)
        provenance = data_binding.provenance(plan, paths)
        provenance["training_entry_preflight"] = dict(
            path=str(Path(args.preflight_receipt).resolve()), sha256=data_binding.sha256(args.preflight_receipt))

    started = time.monotonic()
    run_root = (Path(plan["output_root"]) / args.run_id if plan else ROOT / "artifacts/m3_runs" / args.run_id)
    base_params = paths["base_params"] if paths else ROOT / "checkpoints" / f"{args.model}_base" / "params"
    if run_root.exists():
        raise FileExistsError(f"Refusing to reuse run_id directory: {run_root}")
    run_root.mkdir(parents=True)
    report: dict[str, Any] = {
        "status": "STARTED",
        "run_id": args.run_id,
        "model": args.model,
        "shots": args.shots,
        "steps_requested": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "approval_ref": args.approval_ref,
        "openpi_commit": "215abfb217dbac7d5f1273282331b9b1866c0479",
        "formal_training": plan["phase"] == "train" if plan else True,
        "phase": plan["phase"] if plan else "historical_pilot",
        "provenance": provenance,
        "base_checkpoint": _display_path(base_params),
        "checkpoint_dir": None,
        "loss_history": _display_path(run_root / "loss.jsonl"),
        "gpu_before": None,
        "steps_completed": 0,
        "weights_loaded": False,
        "training_started": False,
    }
    report_path = run_root / "run.json"
    loss_path = run_root / "loss.jsonl"
    checkpoint_manager = None
    try:
        import jax
        from openpi.training import checkpoints as openpi_checkpoints
        from openpi.training import sharding
        import train as openpi_train

        report["jax_backend"] = jax.default_backend()
        report["jax_devices"] = [str(device) for device in jax.devices()]
        if jax.default_backend() not in {"gpu", "cuda"}:
            raise RuntimeError(f"Formal training refuses non-GPU backend: {jax.default_backend()}")
        report["gpu_before"] = _gpu_memory()
        mesh = sharding.make_mesh(1)
        native, data_config, loader_proxy, next_batch, data_info = _build_data(
            args.model, args.shots, args.batch_size, mesh, args.seed,
            dataset_config=args.dataset_config if plan else None
        )
        report["data"] = data_info
        config = _make_config(
            args.model,
            native,
            base_params,
            mesh,
            args.run_id,
            args.steps,
            args.seed,
            run_root,
            training_plan=plan,
        )
        report["checkpoint_dir"] = None
        if not plan or plan["checkpoint_policy"] != "none":
            checkpoint_manager = _initialize_low_memory_checkpoint_dir(
                openpi_checkpoints, config.checkpoint_dir
            )

        rng = jax.random.key(args.seed)
        train_rng, init_rng = jax.random.split(rng)
        report["init_started_at"] = time.time()
        train_state, state_sharding = openpi_train.init_train_state(
            config, init_rng, mesh, resume=False
        )
        jax.block_until_ready(train_state)
        report["weights_loaded"] = True
        report["init_seconds"] = time.monotonic() - started
        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
        )
        replicated_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec()
        )
        ptrain_step = jax.jit(
            lambda step_rng, state, batch: openpi_train.train_step(
                config, step_rng, state, batch
            ),
            in_shardings=(replicated_sharding, state_sharding, data_sharding),
            out_shardings=(state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

        report["training_started"] = True
        with loss_path.open("w", encoding="utf-8") as loss_stream:
            for _ in range(args.steps):
                batch = next_batch()
                train_state, info = ptrain_step(train_rng, train_state, batch)
                jax.block_until_ready((train_state, info))
                values = {key: float(np.asarray(value)) for key, value in info.items()}
                if not all(np.isfinite(value) for value in values.values()):
                    raise FloatingPointError(f"Non-finite training metrics: {values}")
                loss_stream.write(json.dumps(values, ensure_ascii=False) + "\n")
                loss_stream.flush()
                report["steps_completed"] += 1
                if (plan and checkpoint_manager is not None
                        and report["steps_completed"] % plan["save_interval"] == 0
                        and report["steps_completed"] < args.steps):
                    _save_inference_checkpoint(openpi_checkpoints, checkpoint_manager, train_state, loader_proxy, int(train_state.step))
                    checkpoint_manager.wait_until_finished()

        if checkpoint_manager is not None:
            _save_inference_checkpoint(
                openpi_checkpoints, checkpoint_manager, train_state, loader_proxy, int(train_state.step)
            )
            checkpoint_manager.wait_until_finished()
            report["checkpoint_step"] = int(train_state.step)
            report["checkpoint_dir"] = _display_path(config.checkpoint_dir / str(train_state.step))
            report["checkpoint_kind"] = "orbax_params_assets_inference_only"
        else:
            report["checkpoint_kind"] = "none_smoke_only"
        report["status"] = "PASSED"
    except Exception as exc:
        report.update(
            status="FAILED",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
    finally:
        if checkpoint_manager is not None:
            try:
                checkpoint_manager.wait_until_finished()
            except Exception:
                pass
        gc.collect()
        report["gpu_after"] = _gpu_memory()
        report["elapsed_seconds"] = time.monotonic() - started
        _write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _gpu_memory() -> dict[str, int] | None:
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        used, total = result.stdout.strip().splitlines()[0].split(",")
        return {"used_mib": int(used.strip()), "total_mib": int(total.strip())}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-plan", type=Path)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--also-check-plan", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", choices=("pi0", "pi05"))
    parser.add_argument("--shots", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--approval-ref", default="")
    parser.add_argument("--allow-compute", action="store_true")
    args = parser.parse_args()
    if args.training_plan:
        data_binding.bind_args(args)
        if args.preflight:
            if args.allow_compute:
                parser.error("CPU preflight cannot be combined with --allow-compute")
            if args.output is None:
                parser.error("CPU preflight requires a fresh --output receipt")
            from training.cpu_entry_preflight import preflight
            preflight(args)
            return
    else:
        if args.dataset_config or args.preflight or args.also_check_plan or args.preflight_receipt:
            parser.error("New dataset/preflight options require --training-plan")
        if not args.model or not args.run_id or not args.approval_ref:
            parser.error("Legacy mode requires --model, --run-id and --approval-ref")
        for key, value in FORMAL_PILOT_SCOPE.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    result = run(args)
    if result["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
