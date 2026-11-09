"""Exercise the formal training data/config functions on CPU; no model init."""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import resource
import subprocess
from unittest import mock

import numpy as np

from training import data_binding as binding


def preflight(args):
    import jax
    from openpi.training import sharding
    from openpi.models.pi0_config import Pi0Config
    from scripts import m3_formal_train as trainer

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    binding.require(jax.default_backend() == "cpu", "Preflight must run with JAX_PLATFORMS=cpu")
    plan_paths = [Path(args.training_plan).resolve(), *[Path(p).resolve() for p in args.also_check_plan]]
    plans = [binding.load_plan(p) for p in plan_paths]
    plan, paths = plans[0]
    for p, refs in plans:
        binding.require(refs["dataset_config"] == paths["dataset_config"] and refs["base_params"] == paths["base_params"], "Preflight plans use different data/base weights")
        binding.require(p["batch_size"] == plan["batch_size"] and p["seed"] == plan["seed"], "Preflight plans use different batching/seed")
        binding.require(not (Path(p["output_root"]) / p["run_id"]).exists(), "Future GPU run directory already exists")
    root = trainer.ROOT
    checkouts = {"robust_pi": root, "openpi": root / "external/openpi", "libero": root / "external/openpi/third_party/libero"}
    commits = {name: subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() for name, path in checkouts.items()}
    binding.require(commits == plan["source_commits"], "Source commits differ from plan")
    # Current bytes are verified once on CPU. Later GPU commands compare the
    # recorded size/mtime fingerprints, plan hashes, source hashes and receipts.
    assets = []
    inventory = binding.read(paths["inventory"])
    for item in inventory["selected_files"]:
        f = paths["data_root"] / item["path"]
        binding.require(binding.sha256(f) == item["sha256"], "Parquet changed: " + str(f))
        stat = f.stat()
        assets.append(dict(path=str(f), size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=item["sha256"]))
    weights = binding.read(plan["artifacts"]["weight_audit"]["path"])
    for item in weights["files"]:
        f = Path(item["path"])
        binding.require(binding.sha256(f) == item["sha256"], "Weight SHA256 changed: " + str(f))
        stat = f.stat()
        assets.append(dict(path=str(f), size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=item["sha256"]))
    mesh = sharding.make_mesh(1)
    checks = {}
    # These are the SAME functions called by run() on GPU. The guards make
    # accidental parameter restoration or network creation fail during preflight.
    with mock.patch.object(trainer.DeviceCheckpointWeightLoader, "load", side_effect=AssertionError("CPU preflight cannot load weights")) as load_guard, \
         mock.patch.object(Pi0Config, "create", side_effect=AssertionError("CPU preflight cannot create a model")) as create_guard:
        native, config, proxy, next_batch, info = trainer._build_data(
            plan["model"], plan["shots"], plan["batch_size"], mesh, plan["seed"], dataset_config=paths["dataset_config"])
        obs, actions = next_batch()
        binding.require(actions.shape == (1, 50, 32) and obs.state.shape == (1, 32), "Formal train batch has wrong dimensions")
        for value in (actions, obs.state, *obs.images.values()):
            binding.require(np.isfinite(np.asarray(value)).all(), "Nonfinite formal train batch")
        checks["train"] = dict(**info, actions_shape=list(actions.shape), state_shape=list(obs.state.shape), tokens_shape=list(obs.tokenized_prompt.shape))
        _, val_config, _, next_val, val_info = trainer._build_data(
            plan["model"], plan["shots"], plan["batch_size"], mesh, plan["seed"], dataset_config=paths["dataset_config"], split="val")
        val_obs, val_actions = next_val()
        binding.require(val_actions.shape == (1, 50, 32), "Validation batch has wrong dimensions")
        for value in (val_actions, val_obs.state, *val_obs.images.values()):
            binding.require(np.isfinite(np.asarray(value)).all(), "Nonfinite validation batch")
        for key in ("state", "actions"):
            np.testing.assert_array_equal(config.norm_stats[key].mean, val_config.norm_stats[key].mean)
            np.testing.assert_array_equal(config.norm_stats[key].std, val_config.norm_stats[key].std)
        checks["val"] = dict(**val_info, actions_shape=list(val_actions.shape), state_shape=list(val_obs.state.shape), uses_training_stats=True)
        resolved = []
        for p, refs in plans:
            run_root = Path(p["output_root"]) / p["run_id"]
            train_config = trainer._make_config(p["model"], native, refs["base_params"], mesh,
                p["run_id"], p["steps"], p["seed"], run_root, training_plan=p)
            binding.require(train_config.num_train_steps == p["steps"] and train_config.batch_size == p["batch_size"], "Effective budget differs from plan")
            binding.require(train_config.weight_loader.params_path == str(refs["base_params"]), "Effective weight loader uses wrong path")
            resolved.append(dict(phase=p["phase"], plan=str(refs["plan_path"]), run_root=str(run_root),
                steps=train_config.num_train_steps, batch_size=train_config.batch_size, seed=train_config.seed,
                checkpoint_dir=str(train_config.checkpoint_dir), checkpoint_policy=p["checkpoint_policy"],
                save_interval=train_config.save_interval, lr_schedule=dataclasses.asdict(train_config.lr_schedule),
                optimizer=dataclasses.asdict(train_config.optimizer), lora=p["lora"]))
        binding.require(not load_guard.called and not create_guard.called, "A model/weight method was invoked")
    sources = [root / name for name in (
        "scripts/m3_formal_train.py", "training/data_binding.py", "training/cpu_entry_preflight.py",
        "scripts/m3_gpu_smoke.py", "scripts/m3_formal_loader_smoke.py", "data_tools/dataset_adapter.py",
        "data_tools/splits.py", "data_tools/image_preprocessing.py", "external/openpi/scripts/train.py",
        "external/openpi/src/openpi/training/config.py", "external/openpi/src/openpi/training/optimizer.py",
        "external/openpi/src/openpi/transforms.py", "external/openpi/src/openpi/models/gemma.py",
        "external/openpi/src/openpi/models/pi0_config.py")]
    receipt = dict(status="TRAIN_ENTRY_CPU_ACCEPTED", completed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        source_commits=commits, source_hashes={str(p): binding.sha256(p) for p in sources},
        plans={str(refs["plan_path"]): binding.sha256(refs["plan_path"]) for _, refs in plans},
        resolved_plans=resolved, batches=checks, dataset_config=str(paths["dataset_config"]),
        asset_fingerprints=assets, parquet_files_sha256_verified=250, weight_files_sha256_verified=len(weights["files"]),
        backend=jax.default_backend(), model_created=False, model_weights_loaded=False,
        checkpoint_created=False, training_started=False, gpu_started=False,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        adaptation_scope="Pinned native low_mem_finetune: LoRA plus other unfrozen parameters; not adapter-only",
        checkpoint_scope="params/assets inference checkpoints; optimizer resume state is not saved")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        f.write(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    for p, refs in plans:
        binding.verify_preflight(output, p, refs)
    print(json.dumps(dict(status=receipt["status"], output=str(output), batches=checks, plans=resolved,
                         model_created=False, model_weights_loaded=False, training_started=False), indent=2))
