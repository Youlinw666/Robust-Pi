"""Bind a new canonical dataset to the formal trainer without legacy fallbacks.

The metadata checks are CPU-only. No model or accelerator libraries import here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

CONTRACT = "robust_pi_canonical_rot180_v1"
EXPECTED_COMMITS = {
    "robust_pi": "3d635222bca63459cb631c66524371f9e406adf8",
    "openpi": "215abfb217dbac7d5f1273282331b9b1866c0479",
    "libero": "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_paths(dataset_config, *, model="pi0", shots=40):
    path = Path(dataset_config).resolve()
    c = read(path)
    require(model == "pi0" and shots == 40, "This accepted dataset has pi0 40-shot statistics only")
    require(c["dataset_version"] == CONTRACT and c["image_orientation"] == "canonical", "Unknown image contract")
    expected = dict(model_config="pi0_libero_low_mem_finetune", horizon=50, action_dim=7,
                    model_action_dim=32, state_dim=8, shots_per_task=40, task_ids=list(range(5)))
    require(all(c.get(k) == v for k, v in expected.items()), "Dataset protocol differs from accepted pi0 five-task 40-shot")
    result = {k: Path(c[k]).resolve() for k in ("data_root", "inventory", "manifest", "norm_stats")}
    require(result["data_root"].is_dir(), "Missing Parquet root")
    require(result["norm_stats"].name == "norm_stats.json", "Expected an explicit norm_stats.json file")
    for k in ("inventory", "manifest", "norm_stats"):
        require(result[k].is_file(), "Missing " + k)
    inv, manifest = read(result["inventory"]), read(result["manifest"])
    require(inv.get("image_contract") == CONTRACT and inv.get("image_orientation") == "canonical", "Inventory orientation mismatch")
    require(manifest["status"] == "AUDITED_FINAL" and manifest["task_ids"] == list(range(5)), "Manifest is not the audited five-task split")
    require(manifest["source_inventory_sha256"] == sha256(result["inventory"]), "Manifest/inventory hash mismatch")
    from data_tools.splits import validate_manifest
    validate_manifest(manifest)
    expected_train = [eid for ids in manifest["fewshot_episode_ids"]["40"].values() for eid in ids]
    audit_path = path.parent / "normalization_audit.json"
    norm_audit = read(audit_path)
    require(norm_audit["only_training_pool"] and norm_audit["shots_per_task"] == 40, "Stats are not train-only 40-shot")
    require(set(norm_audit["source_episode_ids"]) == set(expected_train) and len(expected_train) == 200, "Stats episode pool mismatch")
    require(norm_audit["manifest_sha256"] == sha256(result["manifest"]), "Stats/manifest mismatch")
    require(norm_audit["norm_stats_sha256"] == sha256(result["norm_stats"]), "Stats hash mismatch")
    require(norm_audit["extra_delta_transform"] is True and norm_audit["m2_sample_action_horizon"] == 50, "Stats numeric transform mismatch")
    acceptance_path = path.parent / "acceptance.json"
    acceptance = read(acceptance_path)
    require(acceptance["status"] == "CPU_PREPARATION_ACCEPTED", "CPU data acceptance is missing")
    for key, target in [("dataset_config.json", path), ("inventory.json", result["inventory"]),
                        ("manifest.json", result["manifest"]), ("pi0/40shot/norm_stats.json", result["norm_stats"]),
                        ("normalization_audit.json", audit_path)]:
        require(acceptance["artifacts"][key]["sha256"] == sha256(target), "Accepted artifact changed: " + key)
    result.update(dataset_config=path, normalization_audit=audit_path, acceptance=acceptance_path)
    return result


def load_plan(path):
    path = Path(path).resolve()
    plan = read(path)
    require(plan["schema_version"] == 1 and plan["phase"] in ("smoke", "pilot", "train"), "Unknown training plan")
    require(plan["native_config"] == "pi0_libero_low_mem_finetune" and plan["task_ids"] == list(range(5)), "Native config/task set mismatch")
    require(plan["action_horizon"] == 50 and plan["model_action_dim"] == 32, "Plan model dimensions mismatch")
    require(plan["model"] == "pi0" and plan["shots"] == 40 and plan["batch_size"] == 1, "Unsupported protocol")
    require(type(plan["seed"]) is int and type(plan["steps"]) is int and plan["steps"] > 0, "Invalid seed or step budget")
    require(type(plan["save_interval"]) is int and plan["save_interval"] > 0, "Invalid checkpoint interval")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", plan["run_id"]) is not None, "Unsafe run_id")
    require(plan["checkpoint_policy"] in ("none", "inference_params_assets"), "Unknown checkpoint policy")
    require(not plan["resume"] and not plan["overwrite"], "Only fresh non-overwriting runs are supported")
    require(plan["phase"] != "smoke" or (plan["steps"] == 1 and plan["checkpoint_policy"] == "none"), "Smoke must be one step without checkpoint")
    require(plan["num_workers"] == 0 and plan["source_commits"] == EXPECTED_COMMITS, "Worker/source version mismatch")
    paths = canonical_paths(plan["dataset_config"], model=plan["model"], shots=plan["shots"])
    for key, ref in plan["artifacts"].items():
        require(sha256(ref["path"]) == ref["sha256"], "Training plan artifact changed: " + key)
    require(plan["artifacts"]["dataset_config"]["sha256"] == sha256(paths["dataset_config"]), "Plan uses different dataset config")
    for key in ("inventory", "manifest", "norm_stats", "acceptance", "normalization_audit"):
        require(Path(plan["artifacts"][key]["path"]).resolve() == paths[key], "Plan reference path mismatch: " + key)
    base = Path(plan["base_params"]).resolve()
    require(base.is_dir(), "Missing accepted base weights")
    weights = read(plan["artifacts"]["weight_audit"]["path"])
    accepted = read(paths["acceptance"])
    for key, artifact in (("weight_audit", "weight_audit.json"), ("official_weight_audit", "weight_official_origin_audit.json")):
        require(plan["artifacts"][key]["sha256"] == accepted["artifacts"][artifact]["sha256"], "Weights no longer match CPU acceptance")
    require(weights["status"] == "PASSED_TRANSPORT_INTEGRITY", "Missing weight integrity acceptance")
    expected = {Path(row["path"]).resolve(): row for row in weights["files"]}
    actual = {x.resolve() for x in base.rglob("*") if x.is_file()}
    require(set(expected) == actual, "Weight directory differs from accepted file set")
    for f, row in expected.items():
        require(f.stat().st_size == row["size"], "Base weight size changed: " + str(f))
    paths.update(base_params=base, plan_path=path)
    return plan, paths


def bind_args(args):
    plan, paths = load_plan(args.training_plan)
    for key in ("model", "shots", "steps", "batch_size", "seed", "run_id"):
        supplied = getattr(args, key)
        require(supplied is None or supplied == plan[key], "CLI conflicts with immutable plan: " + key)
        setattr(args, key, plan[key])
    require(args.dataset_config is None or Path(args.dataset_config).resolve() == paths["dataset_config"], "CLI dataset config conflicts with plan")
    args.dataset_config = paths["dataset_config"]
    return plan, paths


def validate_native(native, plan):
    import dataclasses
    from openpi.models import gemma
    require(native.model.action_horizon == 50 and native.model.action_dim == 32, "Native model dimensions changed")
    require(native.data.extra_delta_transform is True, "Native legacy DeltaActions changed")
    require(native.model.paligemma_variant == plan["lora"]["paligemma_variant"], "PaliGemma variant mismatch")
    require(native.model.action_expert_variant == plan["lora"]["action_expert_variant"], "Action expert variant mismatch")
    for key, variant in [("paligemma_rank", native.model.paligemma_variant), ("action_expert_rank", native.model.action_expert_variant)]:
        require({x.rank for x in gemma.get_config(variant).lora_configs.values()} == {plan["lora"][key]}, "LoRA rank mismatch")
    require(dataclasses.asdict(native.lr_schedule) == plan["lr_schedule"], "Learning rate schedule changed")
    require(dataclasses.asdict(native.optimizer) == plan["optimizer"], "Optimizer changed")
    require(native.ema_decay == plan["ema_decay"], "EMA configuration changed")


def provenance(plan, paths):
    refs = dict(plan["artifacts"])
    refs["training_plan"] = dict(path=str(paths["plan_path"]), sha256=sha256(paths["plan_path"]))
    return refs


def verify_preflight(receipt_path, plan, paths):
    require(receipt_path is not None, "New-plan GPU execution requires its completed CPU preflight receipt")
    receipt = read(receipt_path)
    require(receipt["status"] == "TRAIN_ENTRY_CPU_ACCEPTED", "Training-entry CPU preflight did not pass")
    require(receipt["plans"].get(str(paths["plan_path"])) == sha256(paths["plan_path"]), "Plan differs from CPU preflight")
    for name, digest in receipt["source_hashes"].items():
        require(sha256(name) == digest, "Code changed after CPU preflight: " + name)
    for item in receipt["asset_fingerprints"]:
        stat = Path(item["path"]).stat()
        require(stat.st_size == item["size"] and stat.st_mtime_ns == item["mtime_ns"], "Asset changed after CPU preflight: " + item["path"])
