#!/usr/bin/env python3
"""LIBERO single-task teaching: recorded M1 playback or an instrumented live episode.

Run with .venv-libero/bin/python. Archive mode never creates an environment or
loads a model; live mode calls the same fixed official helpers as M1.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import importlib.util
import hashlib
import json
import os
import random
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from urllib.request import urlopen

import imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external/openpi"
sys.path.insert(0, str(ROOT))
KEYS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
        "robot0_eef_quat", "robot0_gripper_qpos")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "environment/libero"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def emit(message):
    print(message, flush=True)


def describe(name, value):
    array = np.asarray(value)
    emit("{}: shape={} dtype={}".format(name, array.shape, array.dtype))
    if array.size <= 80:
        emit(np.array2string(array, precision=8, suppress_small=False))


def official_client():
    sys.path.insert(0, str(UPSTREAM / "third_party/libero"))
    spec = importlib.util.spec_from_file_location("teaching_official_client", UPSTREAM / "examples/libero/main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def policy_input(official, obs, prompt, *, visual_ood: bool = False, visual_ood_seed: int = 20261021,
                 sample_id: str = "sample", visual_ood_probability: float = 1.0):
    """Build the verified M1 input, optionally applying M5 photometric OOD."""
    def image(key):
        rotated = np.ascontiguousarray(obs[key][::-1, ::-1])
        output = official.image_tools.convert_to_uint8(official.image_tools.resize_with_pad(rotated, 224, 224))
        if visual_ood:
            from training.augmentation import VisualAugmentation, VisualAugmentationConfig

            augmenter = VisualAugmentation(
                VisualAugmentationConfig(
                    probability=visual_ood_probability,
                    brightness_range=(0.8, 1.2),
                    contrast_range=(0.9, 1.1),
                    rgb_noise_std=2.0,
                    seed=visual_ood_seed,
                )
            )
            view_id = "base" if key == "agentview_image" else "wrist"
            output = augmenter.apply(output, sample_id=sample_id, view_id=view_id)
        return output
    return {
        "observation/image": image("agentview_image"),
        "observation/wrist_image": image("robot0_eye_in_hand_image"),
        "observation/state": np.concatenate((obs["robot0_eef_pos"],
            official._quat2axisangle(obs["robot0_eef_quat"].copy()), obs["robot0_gripper_qpos"])),
        "prompt": str(prompt),
    }


def archive(args):
    source = args.source.resolve()
    result = json.loads((source / "result.json").read_text())
    steps = [json.loads(line) for line in (source / "steps.jsonl").read_text().splitlines()]
    with np.load(source / "first_inference.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    np.testing.assert_array_equal(data["actions"][:result["replan_steps"]],
        np.asarray([row["action"] for row in steps[:result["replan_steps"]]], dtype=data["actions"].dtype))
    args.output.mkdir(parents=True, exist_ok=False)
    emit("[归档读取] {}；不是重新推理，也没有调用 env.step".format(result["run_id"]))
    emit("prompt: {!r} (Python str，无图像那样的tensor shape)".format(result["prompt"]))
    for key in ("image", "wrist_image", "state", "actions"):
        describe(key, data[key])
    for key in ("image", "wrist_image"):
        imageio.imwrite(str(args.output / (key + ".png")), data[key])
    for row in steps:
        if row["step"] % result["replan_steps"] == 0:
            emit("[归档调度] step={} 到达K={}重规划边界；仅首次完整chunk有归档，不重建后续预测".format(
                row["step"], result["replan_steps"]))
        emit("[已记录动作] step={step:03d} action={action} success={success}".format(**row))
    summary = dict(mode="archive_read_only", source=str(source), source_run_id=result["run_id"],
        prompt=result["prompt"], state=data["state"].tolist(), actions=data["actions"].tolist(),
        arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in data.items()},
        action_steps=len(steps), success=result["success"],
        first_five_actions_match_execution=True,
        limitation="M1 saved first preprocessed policy input, not the complete raw observation or later chunks.")
    (args.output / "archive_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


def live(args):
    cfg = json.loads(args.config.read_text())
    cfg.update(run_id=args.output.name, scope="LIBERO learning only; not M2 or a benchmark run",
               output_dir=str(args.output.resolve()), video_dir=str(args.output.resolve()),
               replan_steps=args.replan_steps, seed=args.seed,
               max_steps=cfg["max_steps"] if args.max_steps is None else args.max_steps)
    if not 1 <= cfg["replan_steps"] <= cfg["action_horizon"]:
        raise ValueError("Require 1 <= K <= H (replan_steps <= action_horizon)")
    if not 0 <= cfg["seed"] < 2**32:
        raise ValueError("Require 0 <= seed < 2**32")
    if cfg["max_steps"] < 1:
        raise ValueError("Require max_steps >= 1")
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = args.output / "config.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    result = dict(run_id=cfg["run_id"], mode="teaching_live_single_task", status="RUNNING",
        suite=cfg["suite"], task_id=cfg["task_id"], init_state_id=cfg["init_state_id"],
        action_horizon=cfg["action_horizon"], replan_steps=cfg["replan_steps"],
        seed=cfg["seed"], flow_seed=cfg["flow_seed"], action_steps=0, replans=0,
        inference_count=0, checkpoint_dir=cfg["checkpoint_dir"], config_name=cfg["config_name"],
        max_steps=cfg["max_steps"], settle_steps=cfg["settle_steps"],
        latency_definition="Client infer round trip in ms: serialization, transport, server preprocessing, "
                           "sampling and output transfer; first call includes JIT. No warmup call.",
        success=False, timeout=False, system_error=None, server=None,
        started_at_utc=datetime.now(timezone.utc).isoformat())
    env = None
    server = None
    server_log = None
    writer = None
    latencies = []
    run_start = time.perf_counter()
    loop_start = None

    def save_result():
        temp = args.output / "result.tmp"
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        temp.replace(args.output / "result.json")

    save_result()
    try:
        result["checkpoint_files_before"] = checkpoint_files(Path(cfg["checkpoint_dir"]))
        result["norm_stats_sha256"] = hashlib.sha256((Path(cfg["checkpoint_dir"]) /
            "assets/physical-intelligence/libero/norm_stats.json").read_bytes()).hexdigest()
        result["source_commits"] = {name: subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            for name, path in (("openpi", UPSTREAM), ("libero", UPSTREAM / "third_party/libero"))}
        if args.start_server:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", args.port))
            server_script = Path(cfg.get("server_script", ROOT / "scripts/serve_e0.py"))
            if not server_script.is_absolute():
                server_script = ROOT / server_script
            argv = [str(UPSTREAM / ".venv/bin/python"), str(server_script),
                    "--config", str(config_path.resolve()), "--port", str(args.port)]
            server_log = (args.output / "server.log").open("w")
            server = subprocess.Popen(argv, cwd=ROOT, stdout=server_log, stderr=subprocess.STDOUT)
            result["server"] = dict(argv=argv, cwd=str(ROOT), pid=server.pid, exit_code=None)
            save_result()
            deadline = time.monotonic() + 600
            emit("[在线教学] 正在加载官方模型，服务仅绑定127.0.0.1；日志在 " + str(args.output / "server.log"))
            while True:
                if server.poll() is not None:
                    raise RuntimeError("Policy server exited during loading; see server.log")
                try:
                    with urlopen("http://127.0.0.1:{}/healthz".format(args.port), timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server not ready after 600 seconds")
                time.sleep(2)
        else:
            with urlopen("http://127.0.0.1:{}/healthz".format(args.port), timeout=2) as response:
                if response.status != 200:
                    raise RuntimeError("Server is not ready")
        official = official_client()
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        suite = official.benchmark.get_benchmark_dict()[cfg["suite"]]()
        task = suite.get_task(cfg["task_id"])
        env, prompt = official._get_libero_env(task, official.LIBERO_ENV_RESOLUTION, cfg["seed"])
        result["environment_prompt"] = str(prompt)
        policy_prompt = str(cfg.get("prompt_override") or prompt)
        result["prompt"] = policy_prompt
        result["visual_ood"] = bool(cfg.get("visual_ood", False))
        result["visual_ood_seed"] = int(cfg.get("visual_ood_seed", 20261021))
        result["visual_ood_probability"] = float(cfg.get("visual_ood_probability", 1.0))
        result["task_name"] = task.name
        emit("[任务] " + str(prompt))
        obs = env.reset()
        emit("[reset] 返回{}个字段；接下来加载官方init_state，而非使用本次随机reset布局".format(len(obs)))
        init_state = suite.get_task_init_states(cfg["task_id"])[cfg["init_state_id"]]
        state_array = np.asarray(init_state)
        result["init_state"] = dict(shape=list(state_array.shape), dtype=str(state_array.dtype),
            sha256=hashlib.sha256(state_array.tobytes(order="C")).hexdigest())
        obs = env.set_init_state(init_state)
        result["raw_observation_schema"] = {k: dict(shape=list(v.shape), dtype=str(v.dtype))
                                              for k, v in obs.items() if isinstance(v, np.ndarray)}
        for key in KEYS:
            describe("raw." + key, obs[key])
        emit("[稳定] 执行{}次官方dummy action；不触发VLA重规划".format(cfg["settle_steps"]))
        for _ in range(cfg["settle_steps"]):
            obs, _, _, _ = env.step(official.LIBERO_DUMMY_ACTION)
        controller = env.env.robots[0].controller
        result["controller"] = dict(class_name=type(controller).__name__, control_freq=env.env.control_freq,
            model_timestep=env.env.model_timestep, control_timestep=env.env.control_timestep,
            input_min=controller.input_min.tolist(), input_max=controller.input_max.tolist(),
            output_min=controller.output_min.tolist(), output_max=controller.output_max.tolist())
        emit("[控制器] " + json.dumps(result["controller"]))
        client = official._websocket_client_policy.WebsocketClientPolicy("127.0.0.1", args.port)
        if client.get_server_metadata().get("robust_pi_run_id") != cfg["run_id"]:
            raise RuntimeError("Server run_id mismatch; use --start-server for an isolated session")
        writer = imageio.get_writer(str(args.output / "teaching.mp4"), fps=20)
        plan = deque()
        done = False
        loop_start = time.perf_counter()
        with (args.output / "events.jsonl").open("x") as events:
            for step in range(cfg["max_steps"]):
                if not plan:
                    element = policy_input(
                        official,
                        obs,
                        policy_prompt,
                        visual_ood=result["visual_ood"],
                        visual_ood_seed=result["visual_ood_seed"],
                        visual_ood_probability=result["visual_ood_probability"],
                        sample_id=f"{cfg['run_id']}_step{step}",
                    )
                    raw_name = "raw_obs_{:04d}.npz".format(step)
                    input_name = "policy_input_{:04d}.npz".format(step)
                    np.savez_compressed(args.output / raw_name, **{k: v for k, v in obs.items() if isinstance(v, np.ndarray)})
                    np.savez_compressed(args.output / input_name, image=element["observation/image"],
                        wrist_image=element["observation/wrist_image"], state=element["observation/state"], prompt=np.asarray(prompt))
                    emit("[重规划 {}] step={}；队列为空；H={} K={}".format(result["replans"], step, cfg["action_horizon"], cfg["replan_steps"]))
                    describe("policy.state", element["observation/state"])
                    if step == 0:
                        for key in KEYS:
                            describe("after_settle.raw." + key, obs[key])
                        for key, filename in (("observation/image", "image.png"), ("observation/wrist_image", "wrist_image.png")):
                            describe("policy." + key, element[key])
                            imageio.imwrite(str(args.output / filename), element[key])
                    start = time.perf_counter()
                    response = client.infer(element)
                    infer_ms = (time.perf_counter() - start) * 1000
                    latencies.append(infer_ms)
                    chunk = np.asarray(response["actions"])
                    if chunk.shape != (cfg["action_horizon"], 7) or not np.isfinite(chunk).all():
                        raise ValueError("Invalid chunk: " + str(chunk.shape))
                    chunk_name = "chunk_{:04d}.npy".format(step)
                    np.save(args.output / chunk_name, chunk)
                    describe("actions (已由policy反归一化)", chunk)
                    events.write(json.dumps(dict(event="replan", step=step, replan=result["replans"],
                        raw_observation=raw_name, policy_input=input_name, chunk=chunk_name,
                        actions=chunk.tolist(), dtype=str(chunk.dtype), infer_ms=infer_ms)) + "\n")
                    result["replans"] += 1
                    result["inference_count"] += 1
                    emit("[推理延迟] round_trip_ms={:.3f} first_call={}".format(infer_ms, step == 0))
                    plan.extend((index, action) for index, action in enumerate(chunk[:cfg["replan_steps"]]))
                    save_result()
                writer.append_data(
                    policy_input(
                        official,
                        obs,
                        policy_prompt,
                        visual_ood=result["visual_ood"],
                        visual_ood_seed=result["visual_ood_seed"],
                        visual_ood_probability=result["visual_ood_probability"],
                        sample_id=f"{cfg['run_id']}_step{step}",
                    )["observation/image"]
                )
                index, action = plan.popleft()
                if args.pause:
                    input("按Enter执行step={}，chunk内索引={}：".format(step, index))
                obs, reward, done, info = env.step(action.tolist())
                record = dict(event="step", step=step, chunk_action_index=index, remaining_in_plan=len(plan),
                    action=action.tolist(), reward=float(reward), success=bool(done),
                    next_eef_pos=obs["robot0_eef_pos"].tolist(), next_gripper_qpos=obs["robot0_gripper_qpos"].tolist())
                events.write(json.dumps(record) + "\n")
                events.flush()
                emit("[执行] " + json.dumps(record))
                result["action_steps"] += 1
                if done:
                    break
            result["rollout_loop_wall_seconds"] = time.perf_counter() - loop_start
            writer.append_data(
                policy_input(
                    official,
                    obs,
                    policy_prompt,
                    visual_ood=result["visual_ood"],
                    visual_ood_seed=result["visual_ood_seed"],
                    visual_ood_probability=result["visual_ood_probability"],
                    sample_id=f"{cfg['run_id']}_final",
                )["observation/image"]
            )
        np.savez_compressed(args.output / "final_raw_obs.npz", **{k: v for k, v in obs.items() if isinstance(v, np.ndarray)})
        result.update(status="COMPLETED", success=bool(done), timeout=not bool(done),
            checked_success=bool(env.check_success()), video_frames=result["action_steps"] + 1, video_fps=20)
        # Record diagnostics outside the policy request; never feed predicates to VLA.
        result["goal_predicates"] = env.env.parsed_problem["goal_state"]
    except KeyboardInterrupt:
        result.update(status="INTERRUPTED", system_error="Learner interrupted before episode completion")
        emit(result["system_error"])
    except Exception:
        result.update(status="SYSTEM_ERROR", system_error=traceback.format_exc())
        emit(result["system_error"])
    finally:
        try:
            if writer is not None:
                writer.close()
            if env is not None:
                env.close()
        finally:
            if server is not None:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
                result["server"]["exit_code"] = server.returncode
            if server_log is not None:
                server_log.close()
            result["inference_latency_ms"] = latency_summary(latencies)
            result["run_wall_seconds"] = time.perf_counter() - run_start
            if loop_start is not None and "rollout_loop_wall_seconds" not in result:
                result["rollout_loop_wall_seconds"] = time.perf_counter() - loop_start
            result["action_simulation_seconds"] = result["action_steps"] * result.get("controller", {}).get("control_timestep", 0)
            result["checkpoint_files_after"] = checkpoint_files(Path(cfg["checkpoint_dir"]))
            result["checkpoint_metadata_unchanged"] = result.get("checkpoint_files_before") == result["checkpoint_files_after"]
            result["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            save_result()
    emit("[教学结果] " + json.dumps(result, ensure_ascii=False))
    if result["system_error"]:
        raise RuntimeError("Teaching episode failed; see result.json")


def checkpoint_files(directory):
    """Size/mtime identity only; this is not a new cryptographic weight audit."""
    return {str(path.relative_to(directory)): dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def latency_summary(values):
    def stats(items):
        if not items:
            return dict(count=0, total=None, mean=None, p50=None, p95=None, min=None, max=None)
        array = np.asarray(items, dtype=np.float64)
        return dict(count=len(items), total=float(array.sum()), mean=float(array.mean()),
            p50=float(np.percentile(array, 50)), p95=float(np.percentile(array, 95)),
            min=float(array.min()), max=float(array.max()))
    return dict(first_call=values[0] if values else None, all_calls=stats(values),
                subsequent_calls=stats(values[1:]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    recorded = sub.add_parser("archive", help="Print recorded M1 observation/action and executed steps; no GPU")
    recorded.add_argument("--source", type=Path, default=ROOT / "artifacts/m1_e0_20261018_01")
    recorded.add_argument("--output", type=Path, required=True)
    online = sub.add_parser("live", help="Exactly one task/initial state, using official policy and environment")
    online.add_argument("--output", type=Path, required=True)
    online.add_argument("--config", type=Path, default=ROOT / "configs/m1_e0.json")
    online.add_argument("--start-server", action="store_true")
    online.add_argument("--port", type=int, default=8000)
    online.add_argument("--replan-steps", type=int, default=5, help="K: execute this prefix of each configured H chunk")
    online.add_argument("--seed", type=int, default=7, help="Python/NumPy/environment seed; official policy flow seed stays 0")
    online.add_argument("--max-steps", type=int, default=None, help="Optional action-step cap for a controlled timeout probe")
    online.add_argument("--pause", action="store_true", help="Wait for Enter before every policy action")
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "archive":
        archive(args)
    else:
        live(args)


if __name__ == "__main__":
    main()
