#!/usr/bin/env python3
"""One new frame, one inference, bounded time and sampled resource guards.

Uses the existing E1 inference entry point without changing model/data/config.
Run from the project root via scripts/run_logged.py. No automatic retries.
"""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/g1_bounded_resource_probe_20261021.json"
CHILD = ROOT / "reports/g1_frame1_action_chunk_20261021.json"
SAMPLES = ROOT / "logs/sessions/g1_resource_samples_20261021.jsonl"
CHECKPOINT = ROOT / "artifacts/m3_runs/m3_e1_pi0_20261021_params_only_01/robust_pi_m3_pi0_lora/m3_e1_pi0_20261021_params_only_01/100"
SMI = "/usr/lib/wsl/lib/nvidia-smi"


def save(data):
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(OUT)


def snapshot(pid=None):
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        mem[key] = int(value.strip().split()[0])
    gpu = subprocess.check_output([SMI, "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"], text=True, timeout=5)
    used, free, total = map(int, gpu.strip().splitlines()[0].split(","))
    s = dict(elapsed_s=round(time.monotonic()-START, 3), mem_available_kib=mem["MemAvailable"], mem_total_kib=mem["MemTotal"], swap_used_kib=mem["SwapTotal"]-mem["SwapFree"], gpu_used_mib=used, gpu_free_mib=free, gpu_total_mib=total)
    if pid:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, val = line.split(":", 1)
                    s[key + "_kib"] = int(val.strip().split()[0])
        except FileNotFoundError:
            pass
    return s


def metadata():
    return {str(p.relative_to(CHECKPOINT)): [p.stat().st_size, p.stat().st_mtime_ns] for p in CHECKPOINT.rglob("*") if p.is_file()}


START = time.monotonic()
if any(p.exists() for p in (OUT, CHILD, SAMPLES)):
    raise SystemExit("Refusing to overwrite an existing G1 probe; no automatic retry")
before = metadata()
pre = snapshot()
report = dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), status="PREFLIGHT", preflight=pre, limits=dict(timeout_s=180, min_mem_available_kib=2*1024**2, min_gpu_free_mib=1024), frame_index=1, flow_num_steps=10, checkpoint_files=len(before))
save(report)
# Historical E1 process HWM ~12.04 GiB and GPU delta ~6.4 GiB.
if pre["mem_available_kib"] < 15*1024**2 or pre["gpu_free_mib"] < 8*1024:
    report["status"] = "NOT_RUN_PREFLIGHT_RESOURCE_MARGIN"
    save(report)
    raise SystemExit(2)
env = dict(os.environ, JAX_PLATFORMS="cuda", XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_ALLOCATOR="platform", OPENPI_DATA_HOME=str(ROOT/"artifacts/openpi_cache"))
env["PATH"] = "/usr/lib/wsl/lib:" + env.get("PATH", "")
argv = [str(ROOT/"external/openpi/.venv/bin/python"), "scripts/m3_e1_action_inference.py", "--output", str(CHILD), "--frame-index", "1", "--num-steps", "10"]
report.update(argv=argv, environment={k:env[k] for k in ("JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_ALLOCATOR", "OPENPI_DATA_HOME")}, status="RUNNING")
save(report)
samples = [pre]
reason = None
p = subprocess.Popen(argv, cwd=ROOT, env=env, start_new_session=True)
try:
    with SAMPLES.open("x") as log:
        while p.poll() is None:
            s = snapshot(p.pid)
            samples.append(s)
            log.write(json.dumps(s)+"\n")
            log.flush()
            if s["mem_available_kib"] < 2*1024**2:
                reason = "STOPPED_HOST_RAM_MARGIN"
            elif s["gpu_free_mib"] < 1024:
                reason = "STOPPED_GPU_MARGIN"
            elif time.monotonic()-START > 180:
                reason = "STOPPED_TIMEOUT"
            if reason:
                break
            time.sleep(0.5)
except Exception as exc:
    reason = "STOPPED_MONITOR_ERROR: " + repr(exc)
finally:
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
    code = p.wait()
    child = json.loads(CHILD.read_text()) if CHILD.exists() else None
    report.update(status=reason or ("PASSED" if code == 0 and child and child.get("actions_finite") and child.get("jax_backend")=="gpu" else "FAILED_CHILD"), exit_code=code, child_report=child, checkpoint_metadata_unchanged=(before==metadata()), samples=len(samples), elapsed_s=time.monotonic()-START, sampled_gpu_peak_mib=max(s["gpu_used_mib"] for s in samples), sampled_min_gpu_free_mib=min(s["gpu_free_mib"] for s in samples), sampled_min_mem_available_kib=min(s["mem_available_kib"] for s in samples), sampled_child_rss_peak_kib=max(s.get("VmRSS_kib", 0) for s in samples), sampled_swap_peak_kib=max(s["swap_used_kib"] for s in samples), ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
raise SystemExit(0 if report["status"] == "PASSED" else 3)
