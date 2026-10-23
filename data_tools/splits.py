"""Strict episode-level 40/5/5 planning. No automatic fallback to fewer shots."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_tools.dataset_adapter import DataError, sha256_file


def make_manifest(inventory_path: str | Path, *, task_ids=(0, 1, 2, 3, 4), seed=20261020,
                  shots=(5, 10, 20, 40), val_count=5, holdout_count=5) -> dict:
    """Pure planning; call only after a separate complete data audit.

    Raises on insufficient episodes. Test fixtures are not a real split.
    Source file/hash/validity audit must be added before a plan becomes final.
    """
    inventory_path = Path(inventory_path)
    source = json.loads(inventory_path.read_text())
    if not shots or sorted(set(shots)) != list(shots) or min(shots) <= 0:
        raise DataError("Few-shot sizes must be positive, unique and increasing")
    if len(set(task_ids)) != len(task_ids) or val_count < 1 or holdout_count < 1:
        raise DataError("Invalid task IDs or validation/holdout sizes")
    result = dict(status="PLAN_NOT_AUDITED", repo_id=source["repo_id"], revision=source["revision"],
        source_inventory_sha256=sha256_file(inventory_path), split_seed=seed, task_ids=list(task_ids),
        fewshot_sizes=list(shots), val_count=val_count, holdout_count=holdout_count,
        train_episode_ids={}, val_episode_ids={}, holdout_episode_ids={},
        fewshot_episode_ids={str(n): {} for n in shots})
    for task in task_ids:
        ids = sorted(f["episode_id"] for f in source["selected_files"] if f["task_id"] == task)
        if len(ids) != len(set(ids)):
            raise DataError("Duplicate source episode IDs")
        if len(ids) < max(shots) + val_count + holdout_count:
            raise DataError("Task {} has {} episodes, requires {} for {}+{}+{}; no fallback permitted".format(
                task, len(ids), max(shots) + val_count + holdout_count, max(shots), val_count, holdout_count))
        shuffled = np.random.default_rng(np.random.SeedSequence([seed, task])).permutation(ids).tolist()
        key = str(task)
        result["val_episode_ids"][key] = shuffled[:val_count]
        result["holdout_episode_ids"][key] = shuffled[val_count:val_count + holdout_count]
        pool = shuffled[val_count + holdout_count:]
        result["train_episode_ids"][key] = pool
        for n in shots:
            result["fewshot_episode_ids"][str(n)][key] = pool[:n]
    validate_manifest(result)
    return result


def validate_manifest(document: dict) -> None:
    tasks = {str(task) for task in document["task_ids"]}
    if len(tasks) != len(document["task_ids"]):
        raise DataError("Duplicate task IDs")
    if document["fewshot_sizes"] != sorted(set(document["fewshot_sizes"])):
        raise DataError("Invalid few-shot sizes")
    all_ids = set()
    for split in ("train", "val", "holdout"):
        mapping = document[split + "_episode_ids"]
        if set(mapping) != tasks:
            raise DataError("Manifest task set differs between splits")
        for task, ids in mapping.items():
            if len(ids) != len(set(ids)) or all_ids.intersection(ids):
                raise DataError("Episode duplicated within/across task/split")
            if split != "train" and len(ids) != document[split + "_count"]:
                raise DataError("Invalid split count")
            all_ids.update(ids)
    for task in tasks:
        train = document["train_episode_ids"][task]
        previous = set()
        for n in document["fewshot_sizes"]:
            mapping = document["fewshot_episode_ids"][str(n)]
            if set(mapping) != tasks:
                raise DataError("Few-shot task set mismatch")
            ids = mapping[task]
            if len(ids) != n or len(set(ids)) != n or ids != train[:n]:
                raise DataError("Few-shot IDs must be ordered prefixes of their training pool")
            if previous and not previous < set(ids):
                raise DataError("Few-shot nesting is not strict")
            previous = set(ids)
