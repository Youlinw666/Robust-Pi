#!/usr/bin/env python3
"""Run one real π0 E1 action-chunk inference from the params-only checkpoint.

This is a read-only diagnostic: it loads the trained inference checkpoint, uses
one audited M2 observation, and writes shapes/finite checks/resource facts. It
does not run LIBERO or claim closed-loop performance.
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _gpu_memory() -> dict[str, int | None]:
    try:
        value = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()[0]
        used, total = (int(part.strip()) for part in value.split(","))
        return {"used_mib": used, "total_mib": total}
    except (FileNotFoundError, IndexError, ValueError, subprocess.CalledProcessError):
        return {"used_mib": None, "total_mib": None}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "artifacts/m3_runs/m3_e1_pi0_20261021_params_only_01/robust_pi_m3_pi0_lora/"
        / "m3_e1_pi0_20261021_params_only_01/100",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=1)
    args = parser.parse_args()

    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    if args.num_steps < 1:
        raise ValueError("--num-steps must be positive")

    started = time.monotonic()
    report: dict = {
        "status": "STARTED",
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "model_config": "pi0_libero_low_mem_finetune",
        "task_id": args.task_id,
        "episode_id": args.episode_id,
        "frame_index": args.frame_index,
        "flow_num_steps": args.num_steps,
        "gpu_before": _gpu_memory(),
    }
    try:
        from data_tools.dataset_adapter import DatasetAdapter
        from openpi.policies import policy_config
        from openpi.training import config as openpi_config

        source = ROOT / "data/libero_offset_diagnostic/f13aa24a3da8c43c7225569f28c562979fa0e35a"
        inventory = ROOT / "reports/m2_formal_inventory.json"
        manifest = ROOT / "manifests/m2_dataset_manifest.json"
        adapter = DatasetAdapter(source, inventory, manifest=manifest, split="train", shots=40)
        episode = adapter.load_episode(args.task_id, args.episode_id)
        sample = episode.sample(args.frame_index)
        observation = {
            "observation/image": sample["image"],
            "observation/wrist_image": sample["wrist_image"],
            "observation/state": sample["state"],
            "prompt": sample["prompt"],
        }
        report["input"] = {
            "image_shape": list(sample["image"].shape),
            "wrist_image_shape": list(sample["wrist_image"].shape),
            "state_shape": list(sample["state"].shape),
            "state_dtype": str(sample["state"].dtype),
            "prompt_chars": len(sample["prompt"]),
        }

        config = openpi_config.get_config("pi0_libero_low_mem_finetune")
        policy = policy_config.create_trained_policy(
            config,
            args.checkpoint,
            sample_kwargs={"num_steps": args.num_steps},
        )
        result = policy.infer(observation)
        actions = np.asarray(result["actions"])
        finite = bool(np.isfinite(actions).all())
        report.update(
            {
                "status": "PASSED_ACTION_CHUNK_INFERENCE" if finite else "FAILED_NONFINITE_ACTIONS",
                "jax_backend": __import__("jax").default_backend(),
                "actions_shape": list(actions.shape),
                "actions_dtype": str(actions.dtype),
                "actions_finite": finite,
                "actions_min": float(actions.min()),
                "actions_max": float(actions.max()),
                "policy_timing": result.get("policy_timing"),
            }
        )
    except Exception as exc:  # preserve a structured traceback for resource diagnosis
        import traceback

        report.update(
            {
                "status": "FAILED_ACTION_CHUNK_INFERENCE",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(args.output, report)
        raise
    finally:
        report["gpu_after"] = _gpu_memory()
        report["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["elapsed_seconds"] = time.monotonic() - started
        _write_json(args.output, report)


if __name__ == "__main__":
    main()
