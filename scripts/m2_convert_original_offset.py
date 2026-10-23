#!/usr/bin/env python3
"""Convert raw post-action observations to an explicit offset Parquet contract.

The raw creator stores obs[t] after action[t]. Therefore this diagnostic
conversion stores obs[t] -> action[t+1], discarding action[0] because its
pre-action observation is not present in the raw file. It never claims that
the resulting source is the formal 40/5/5 training pool.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def png_bytes(array):
    from io import BytesIO
    stream = BytesIO()
    Image.fromarray(array).save(stream, format="PNG", optimize=False)
    return stream.getvalue()


def parse_prompt(attrs):
    try:
        value = json.loads(attrs["problem_info"])
        return value.get("language_instruction", "")
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/libero_original/f13aa24a3da8c43c7225569f28c562979fa0e35a")
    parser.add_argument("--inventory", type=Path, default=ROOT / "reports/m2_original_hdf5_inventory.json")
    parser.add_argument("--replay-audit", type=Path, default=ROOT / "reports/m2_replay_alignment_audit.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/libero_offset_diagnostic/f13aa24a3da8c43c7225569f28c562979fa0e35a")
    parser.add_argument("--output-inventory", type=Path, required=True)
    args = parser.parse_args()
    if args.output_inventory.exists():
        parser.error("Refuse to overwrite output inventory")
    audit = json.loads(args.replay_audit.read_text())
    replay = {(row["task_id"], demo["demo_id"]): demo
              for row in audit["tasks"] for demo in row["demos"]}
    source = json.loads(args.inventory.read_text())
    selected, converted = [], []
    for item in source["selected_files"]:
        raw_path = args.data_root / item["path"]
        with h5py.File(raw_path, "r") as source_file:
            data = source_file["data"]
            prompt = parse_prompt(data.attrs)
            names = sorted(data.keys(), key=lambda key: int(key.split("demo_", 1)[1]))
            for name in names:
                demo = data[name]
                n = int(demo["actions"].shape[0])
                if n < 2:
                    continue
                # Explicitly offset observations and actions by one transition.
                images = demo["obs/agentview_rgb"][:-1]
                wrists = demo["obs/eye_in_hand_rgb"][:-1]
                ee = demo["obs/ee_states"][:-1].astype(np.float32)
                grip = demo["obs/gripper_states"][:-1].astype(np.float32)
                state = np.concatenate((ee, grip), axis=1)
                actions = demo["actions"][1:].astype(np.float32)
                rows = dict(
                    image=[{"bytes": png_bytes(value), "path": None} for value in images],
                    wrist_image=[{"bytes": png_bytes(value), "path": None} for value in wrists],
                    state=pa.array(state.tolist(), pa.list_(pa.float32(), 8)),
                    actions=pa.array(actions.tolist(), pa.list_(pa.float32(), 7)),
                    timestamp=pa.array(np.arange(n - 1, dtype=np.float32) / 20, pa.float32()),
                    frame_index=pa.array(np.arange(n - 1), pa.int64()),
                    episode_index=pa.array([item["task_id"] * 1000 + int(name.split("demo_", 1)[1])] * (n - 1), pa.int64()),
                    # Keep the source index unique across demos while retaining
                    # contiguous increments within each episode.
                    index=pa.array(np.arange(n - 1) + ((item["task_id"] * 1000 + int(name.split("demo_", 1)[1])) * 1000), pa.int64()),
                    task_index=pa.array([item["task_id"]] * (n - 1), pa.int64()),
                )
                out_rel = Path(item["path"]).with_suffix("") / (name + ".parquet")
                out_path = args.output_root / out_rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.table(rows), out_path, compression="zstd")
                replay_row = replay.get((item["task_id"], name), {})
                digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
                episode_id = item["task_id"] * 1000 + int(name.split("demo_", 1)[1])
                selected.append(dict(task_id=item["task_id"], dataset_task_index=item["task_id"],
                                     episode_id=episode_id, path=str(out_rel), length=n - 1,
                                     size=out_path.stat().st_size, sha256=digest,
                                     source_demo_id=name, source_file=item["path"],
                                     replay_success=bool(replay_row.get("done", False)),
                                     replay_done_at=replay_row.get("done_at"),
                                     conversion_status="DIAGNOSTIC_OFFSET_OBS_T_TO_ACTION_T_PLUS_1",
                                     dropped_source_actions=[0], prompt=prompt))
                converted.append(dict(task_id=item["task_id"], demo_id=name, path=str(out_rel),
                                      length=n - 1, replay_success=bool(replay_row.get("done", False))))
    formal_counts = {str(task): sum(row["replay_success"] for row in selected if row["task_id"] == task)
                     for task in range(5)}
    prompts = {}
    for row in selected:
        prompts.setdefault(row["task_id"], row["prompt"])
    spatial_tasks = [
        dict(task_id=task_id, dataset_task_index=task_id,
             prompt=prompts.get(task_id, ""))
        for task_id in range(5)
    ]
    result = dict(status="DIAGNOSTIC_NOT_FORMAL_SPLIT", generated_at=datetime.now(timezone.utc).isoformat(),
                  repo_id=source["repo_id"], revision=source["revision"], source_revision=source["revision"],
                  source_inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
                  replay_audit_sha256=hashlib.sha256(args.replay_audit.read_bytes()).hexdigest(),
                  source_fps=20,
                  spatial_tasks=spatial_tasks,
                  source_role="Original HDF5 offset conversion for audit and interface testing only",
                  temporal_contract="Raw obs[t] was recorded after action[t]; output row t uses raw obs[t] and action[t+1].",
                  dropped_first_action=True, output_root=str(args.output_root), selected_files=selected,
                  replay_success_count_by_task=formal_counts,
                  formal_40_5_5_ready=False,
                  reason="Replay success counts are below 50 for every task; no formal manifest or norm stats is created.")
    args.output_inventory.parent.mkdir(parents=True, exist_ok=True)
    args.output_inventory.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(converted=len(converted), replay_success_count_by_task=formal_counts), indent=2))


if __name__ == "__main__":
    main()
