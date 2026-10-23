#!/usr/bin/env python3
"""G2: immutable registration, guarded single cases, and saved-evidence analysis.

No training, model imports, automatic retries, or batch expansion in this tool.
Run with the existing openpi Python for NumPy; rollout uses the LIBERO Python.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BATCH = "g2_e1_visual_ood_20261021"
BASE = ROOT / "artifacts" / BATCH
MANIFEST = ROOT / "manifests" / (BATCH + ".json")
LIBERO = ROOT / ".venv-libero/bin/python"
SMI = "/usr/lib/wsl/lib/nvidia-smi"
OLD = [ROOT / f"artifacts/m3_e1_v5_task0_init{i}_k5_20261021_01" for i in (0, 1)]


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value, *, exclusive=False):
    p = Path(p)
    if exclusive and p.exists():
        raise FileExistsError(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def tree(p):
    return {rel(f): {"sha256": sha(f), "bytes": f.stat().st_size} for f in sorted(Path(p).rglob("*")) if f.is_file()}


def verify_hashes(items):
    for name, info in items.items():
        assert sha(ROOT/name) == info["sha256"], "Hash mismatch: " + name


def checkpoint_hashes(cp):
    expected = {}
    for line in (ROOT/"reports/CLOUD_MIGRATION_E1_SHA256.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.lstrip("*")] = digest
    actual = {str(f.relative_to(cp)): sha(f) for f in cp.rglob("*") if f.is_file()}
    assert actual == expected, "Checkpoint differs from G0 SHA256 inventory"
    return {rel(cp/name): {"sha256": digest, "bytes": (cp/name).stat().st_size} for name, digest in actual.items()}


def decode_video(p):
    code = """import imageio_ffmpeg,json,subprocess,sys
p=sys.argv[1]; exe=imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([exe,'-v','error','-i',p,'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
g=imageio_ffmpeg.read_frames(p); meta=next(g); count=sum(1 for _ in g)
print(json.dumps(dict(frames=count,metadata=meta,decoder=exe)))
"""
    return json.loads(subprocess.check_output([str(LIBERO), "-c", code, str(p)], text=True, timeout=60))


def audit_episode(dr):
    """Validate saved chunks against every actual action and terminal classification."""
    r = read(dr/"result.json")
    assert r["status"] == "COMPLETED" and r["system_error"] is None
    assert r["checkpoint_metadata_unchanged"]
    assert r["success"] == r["checked_success"]
    assert bool(r["timeout"]) == (not r["success"] and r["action_steps"] == r["max_steps"])
    events = [json.loads(l) for l in (dr/"events.jsonl").read_text().splitlines()]
    steps = [e for e in events if e["event"] == "step"]
    replans = [e for e in events if e["event"] == "replan"]
    assert len(steps) == r["action_steps"] and len(replans) == r["inference_count"]
    assert len(replans) == math.ceil(len(steps)/r["replan_steps"])
    plans = {}
    for e in replans:
        a = np.load(dr/e["chunk"], allow_pickle=False)
        assert a.shape == (50,7) and np.isfinite(a).all()
        np.testing.assert_array_equal(a, np.asarray(e["actions"]))
        assert e["step"] % 5 == 0
        plans[e["step"]] = a
        with np.load(dr/e["policy_input"], allow_pickle=False) as data:
            assert data["state"].shape == (8,) and np.isfinite(data["state"]).all()
            assert data["image"].shape == data["wrist_image"].shape == (224,224,3)
            assert data["image"].dtype == data["wrist_image"].dtype == np.uint8
            assert str(data["prompt"]) == r["prompt"]
        assert (dr/e["raw_observation"]).exists()
    for i,e in enumerate(steps):
        assert e["step"] == i and e["chunk_action_index"] == i%5
        np.testing.assert_array_equal(np.asarray(e["action"]), plans[i//5*5][i%5])
        assert bool(e["success"]) == (r["success"] and i==len(steps)-1)
        assert np.isfinite(e["next_eef_pos"]).all()
    video = decode_video(dr/"teaching.mp4")
    assert video["frames"] == r["video_frames"] == len(steps)+1
    actions = np.asarray([s["action"] for s in steps])
    positions = np.asarray([s["next_eef_pos"] for s in steps])
    latencies = [e["infer_ms"] for e in replans]
    np.testing.assert_allclose(sum(latencies), r["inference_latency_ms"]["all_calls"]["total"])
    return dict(status="success" if r["success"] else "timeout", success=r["success"], timeout=r["timeout"], system_error=r["system_error"], action_steps=len(steps), inference_count=len(replans), latency=r["inference_latency_ms"], wall_seconds=r["run_wall_seconds"], video=rel(dr/"teaching.mp4"), video_decode=video, action_statistics=dict(mean=actions.mean(0).tolist(), std=actions.std(0).tolist(), min=actions.min(0).tolist(), max=actions.max(0).tolist(), abs_gt_one_fraction=float((np.abs(actions)>1).mean()), translational_abs_gt_one_fraction=float((np.abs(actions[:,:3])>1).mean()), rotational_abs_gt_one_fraction=float((np.abs(actions[:,3:6])>1).mean()), final_eef_pos=positions[-1].tolist(), eef_path_length=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum())), init_state=r["init_state"], controller=r["controller"], prompt=r["prompt"], norm_stats_sha256=r["norm_stats_sha256"], checkpoint_dir=r["checkpoint_dir"], source_commits=r["source_commits"])


def prepare():
    assert not MANIFEST.exists() and not BASE.exists(), "Registration exists; refuse overwrite"
    audits = [audit_episode(d) for d in OLD]
    results = [read(d/"result.json") for d in OLD]
    assert all(a["action_steps"] == 220 and a["inference_count"] == 44 and a["status"] == "timeout" for a in audits)
    assert results[0]["checkpoint_dir"] == results[1]["checkpoint_dir"]
    cp = Path(results[0]["checkpoint_dir"])
    checks = checkpoint_hashes(cp)
    protocol = read(ROOT/"configs/m5_augmentation_protocol.json")["visual"]
    assert protocol == dict(probability=0.5, brightness_range=[0.8,1.2], contrast_range=[0.9,1.1], rgb_noise_std=2.0, geometry_changes=False, label_changes=False, seed=20261021)
    BASE.mkdir()
    sources = ["scripts/g2_e1_visual_ood.py", "logs/sessions/g2_libero_teaching_before_20261021.py", "scripts/libero_teaching.py", "scripts/serve_e0.py", "training/augmentation.py", "configs/m5_augmentation_protocol.json", "external/openpi/examples/libero/main.py", "external/openpi/src/openpi/models/model.py", "external/openpi/src/openpi/models/pi0.py", "external/openpi/src/openpi/policies/policy.py", "external/openpi/src/openpi/policies/policy_config.py", "external/openpi/src/openpi/training/config.py", "reports/CLOUD_MIGRATION_E1_SHA256.txt"]
    source_hashes = {p: {"sha256": sha(ROOT/p)} for p in sources}
    environment = {}
    for name, executable in [("model",ROOT/"external/openpi/.venv/bin/python"),("simulation",LIBERO)]:
        output = subprocess.check_output([str(executable), "-c", "import importlib.metadata,json,sys; print(json.dumps({'python':sys.version,'distributions':sorted([(d.metadata['Name'],d.version) for d in importlib.metadata.distributions()])}))"], text=True)
        environment[name] = json.loads(output)
    cases=[]
    invariants = ["suite","task_id","init_state_id","seed","flow_seed","config_name","checkpoint_dir","checkpoint_role","action_horizon","replan_steps","max_steps","settle_steps","server_script","prompt_id"]
    for i, (dr, old, audit) in enumerate(zip(OLD, results, audits)):
        cfg=read(dr/"config.json")
        assert cfg["task_id"]==0 and cfg["init_state_id"]==i and cfg["seed"]==7 and cfg["flow_seed"]==0
        assert cfg["action_horizon"]==50 and cfg["replan_steps"]==5 and cfg["max_steps"]==220 and cfg["settle_steps"]==10
        assert not old["visual_ood"]
        idcase=f"{BATCH}_t00_i{i:02d}_clean_reused"
        cases.append(dict(case_id=idcase,evidence="reused",condition="clean_id",origin_run_id=old["run_id"],artifact_dir=rel(dr),protocol={k:cfg[k] for k in invariants},audit=audit,files=tree(dr)))
        new_id=f"{BATCH}_t00_i{i:02d}_visual_ood"
        out=BASE/"episodes"/new_id
        newcfg=dict(cfg)
        newcfg.update(run_id=new_id,scope="G2 two-pair descriptive E1 evaluation; not LIBERO benchmark",output_dir=str(out),video_dir=str(out),visual_ood=True,visual_ood_probability=protocol["probability"],visual_ood_seed=protocol["seed"],perturbation_id="m5_photometric_p05_seed20261021")
        config_path=BASE/"configs"/(new_id+".json")
        write(config_path,newcfg,exclusive=True)
        assert all(newcfg[k]==cfg[k] for k in invariants)
        cases.append(dict(case_id=new_id,evidence="new",condition="visual_ood",paired_clean_case_id=idcase,artifact_dir=rel(out),config_path=rel(config_path),config_sha256=sha(config_path),protocol={k:cfg[k] for k in invariants},visual=protocol,sample_id_rule=new_id+"_step{step}",view_ids=["base","wrist"],port=8060+i))
    manifest=dict(schema_version=1,run_id=BATCH,registered_at_utc=stamp(),status="PREREGISTERED",checkpoint_role="self_trained",checkpoint_files=checks,source_hashes=source_hashes,environment=environment,flow_num_steps=10,flow_num_steps_evidence="pinned pi0.sample_actions default=10; serve_e0 does not override",cases=cases,resources=dict(preflight_mem_available_kib=15*1024**2,preflight_gpu_free_mib=8*1024,min_mem_available_kib=2*1024**2,min_gpu_free_mib=1024,timeout_s=300,poll_sleep_s=0.5),protocol_difference_from_legacy_e0=dict(probability={"G2_M5":0.5,"legacy_E0":1.0},seed={"G2":20261021,"legacy_E0":"20261021+task_id*100+init_id"},sample_id="run_id dependent; E0 and G2 have distinct perturbation draws",comparison="Do not merge E0/E1 or label identical OOD protocols"),limitations=["two pairs only", "clean ID 0/2", "retention undefined for zero ID success", "reused and new runs are different historical executions", "sampled resources may miss transients"])
    write(MANIFEST,manifest,exclusive=True)
    write(ROOT/"reports/g2_case_identity_mapping.json",dict(manifest=rel(MANIFEST),manifest_sha256=sha(MANIFEST),cases=[{k:c[k] for k in ("case_id","evidence","condition","artifact_dir")} | ({"origin_run_id":c["origin_run_id"]} if c["evidence"]=="reused" else {"paired_clean_case_id":c["paired_clean_case_id"]}) for c in cases]),exclusive=True)
    print(json.dumps(dict(status="PREREGISTERED",manifest=rel(MANIFEST),manifest_sha256=sha(MANIFEST),clean_audits=audits),ensure_ascii=False,indent=2))


def proc_status(pid):
    try:
        d={}
        for l in Path(f"/proc/{pid}/status").read_text().splitlines():
            if l.startswith(("VmRSS:","VmHWM:")):
                k,v=l.split(":",1);d[k+"_kib"]=int(v.split()[0])
        return d
    except (FileNotFoundError,PermissionError,ProcessLookupError):
        return {}


def snapshot(start, group=None):
    mem={l.split(":")[0]:int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()}
    value=subprocess.check_output([SMI,"--query-gpu=memory.used,memory.free,memory.total","--format=csv,noheader,nounits"],text=True,timeout=5)
    used,free,total=map(int,value.splitlines()[0].split(","))
    vm=dict(l.split() for l in Path("/proc/vmstat").read_text().splitlines())
    s=dict(elapsed_s=time.monotonic()-start,mem_available_kib=mem["MemAvailable"],swap_used_kib=mem["SwapTotal"]-mem["SwapFree"],gpu_used_mib=used,gpu_free_mib=free,gpu_total_mib=total,oom_kill=int(vm.get("oom_kill",0)),processes=[])
    if group:
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():continue
            try:
                parts=(d/"stat").read_text().rsplit(")",1)[1].split()
                if int(parts[2])==group:
                    argv=(d/"cmdline").read_bytes().split(b"\0")
                    s["processes"].append(dict(pid=int(d.name),argv=[a.decode(errors="replace") for a in argv[:3]],**proc_status(d.name)))
            except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    s["tree_rss_kib"]=sum(p.get("VmRSS_kib",0) for p in s["processes"])
    return s


def heavy_processes():
    found=[]
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():continue
        try:
            argv=(d/"cmdline").read_bytes().split(b"\0")
            if not argv or b"python" not in argv[0]:continue
            if any(any(w in a for w in (b"serve_e0.py",b"libero_teaching.py",b"m3_formal_train.py",b"m3_gpu_smoke.py",b"m3_e1_action_inference.py",b"m4_e0_")) for a in argv[1:3]):
                found.append({"pid":int(d.name),"argv":[x.decode(errors="replace") for x in argv[:3]]})
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return found


def outcome_path(case):
    return BASE/"resources"/(case["case_id"]+".json")


def pair_checks(case):
    """Initial state/raw observation/state/prompt equality, and exact M5 application."""
    from training.augmentation import VisualAugmentation,VisualAugmentationConfig
    from openpi_client import image_tools
    dr=ROOT/case["artifact_dir"]
    old=OLD[case["protocol"]["init_state_id"]]
    a,b=read(old/"result.json"),read(dr/"result.json")
    for key in ["task_id","init_state_id","seed","flow_seed","action_horizon","replan_steps","max_steps","settle_steps","init_state","controller","norm_stats_sha256","checkpoint_dir","prompt","environment_prompt","source_commits"]:
        assert a[key]==b[key],key
    with np.load(old/"raw_obs_0000.npz",allow_pickle=False) as x,np.load(dr/"raw_obs_0000.npz",allow_pickle=False) as y:
        assert set(x.files)==set(y.files)
        for k in x.files:np.testing.assert_array_equal(x[k],y[k],err_msg=k)
    with np.load(old/"policy_input_0000.npz",allow_pickle=False) as x,np.load(dr/"policy_input_0000.npz",allow_pickle=False) as y:
        np.testing.assert_array_equal(x["state"],y["state"])
        np.testing.assert_array_equal(x["prompt"],y["prompt"])
    visual=case["visual"]
    aug=VisualAugmentation(VisualAugmentationConfig(probability=visual["probability"],brightness_range=tuple(visual["brightness_range"]),contrast_range=tuple(visual["contrast_range"]),rgb_noise_std=visual["rgb_noise_std"],seed=visual["seed"]))
    changes={"base":0,"wrist":0};mae={"base":[],"wrist":[]}
    for e in [json.loads(l) for l in (dr/"events.jsonl").read_text().splitlines() if json.loads(l)["event"]=="replan"]:
        with np.load(dr/e["raw_observation"],allow_pickle=False) as raw,np.load(dr/e["policy_input"],allow_pickle=False) as actual:
            for view,key,saved in [("base","agentview_image","image"),("wrist","robot0_eye_in_hand_image","wrist_image")]:
                clean=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(raw[key][::-1,::-1]),224,224))
                expected=aug.apply(clean,sample_id=f"{case['case_id']}_step{e['step']}",view_id=view)
                np.testing.assert_array_equal(expected,actual[saved])
                changes[view]+=int(not np.array_equal(clean,expected))
                mae[view].append(float(np.abs(clean.astype(float)-expected).mean()))
    assert sum(changes.values())>0, "No actual photometric intervention"
    return dict(initial_raw_arrays_equal=True,initial_state_and_prompt_equal=True,controller_and_model_identity_equal=True,all_recorded_inputs_match_registered_M5=True,changed_policy_inputs=changes,mean_pixel_mae={k:float(np.mean(v)) for k,v in mae.items()})


def run_case(index):
    m=read(MANIFEST);cases=[c for c in m["cases"] if c["evidence"]=="new"]
    assert index in (0,1)
    c=cases[index];out=ROOT/c["artifact_dir"];rp=outcome_path(c)
    assert not out.exists() and not rp.exists(), "Never overwrite or automatically retry a case"
    if index==1:
        assert read(outcome_path(cases[0]))["status"]=="ACCEPTED", "First case has not passed all gates"
    registration_sha=sha(MANIFEST)
    assert registration_sha==read(ROOT/"reports/g2_case_identity_mapping.json")["manifest_sha256"], "Manifest differs from registration hash"
    verify_hashes(m["source_hashes"])
    verify_hashes(m["checkpoint_files"])
    for old in [c for c in m["cases"] if c["evidence"]=="reused"]:verify_hashes(old["files"])
    assert sha(ROOT/c["config_path"])==c["config_sha256"]
    assert not heavy_processes(),"Another training/inference/simulation process exists"
    start=time.monotonic();pre=snapshot(start)
    r=dict(case_id=c["case_id"],started_at_utc=stamp(),status="PREFLIGHT",manifest_sha256=registration_sha,preflight=pre,config_sha256=c["config_sha256"],checkpoint_hashes_verified=True,source_hashes_verified=True,limits=m["resources"])
    write(rp,r,exclusive=True)
    limits=m["resources"]
    if pre["mem_available_kib"]<limits["preflight_mem_available_kib"] or pre["gpu_free_mib"]<limits["preflight_gpu_free_mib"]:
        r["status"]="BLOCKED_PREFLIGHT_RESOURCE_MARGIN";write(rp,r);print(json.dumps(r,indent=2));return 2
    argv=[str(LIBERO),str(ROOT/"scripts/libero_teaching.py"),"live","--config",str(ROOT/c["config_path"]),"--start-server","--port",str(c["port"]),"--replan-steps","5","--seed","7","--max-steps","220","--output",str(out)]
    env=dict(os.environ,JAX_PLATFORMS="cuda",XLA_PYTHON_CLIENT_PREALLOCATE="false",XLA_PYTHON_CLIENT_ALLOCATOR="platform",OPENPI_DATA_HOME=str(ROOT/"artifacts/openpi_cache"))
    env["PATH"]="/usr/lib/wsl/lib:"+env.get("PATH","")
    logpath=BASE/"logs"/(c["case_id"]+".log");logpath.parent.mkdir(parents=True,exist_ok=True)
    samplepath=BASE/"resources"/(c["case_id"]+".samples.jsonl")
    r.update(status="RUNNING",argv=argv,cwd=str(ROOT),environment={k:env[k] for k in ["JAX_PLATFORMS","XLA_PYTHON_CLIENT_PREALLOCATE","XLA_PYTHON_CLIENT_ALLOCATOR","OPENPI_DATA_HOME"]},output_log=rel(logpath),samples_path=rel(samplepath));write(rp,r)
    stop=None;samples=[pre];process_peaks={}
    out.parent.mkdir(parents=True,exist_ok=True)
    with logpath.open("x") as log,samplepath.open("x") as stream:
        p=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        r["process_group"]=p.pid;write(rp,r)
        try:
            while p.poll() is None:
                s=snapshot(start,p.pid);samples.append(s);stream.write(json.dumps(s)+"\n");stream.flush()
                for process in s["processes"]:
                    key=str(process["pid"]);entry=process_peaks.setdefault(key,dict(argv=process["argv"],rss_peak_kib=0,hwm_peak_kib=0))
                    entry["rss_peak_kib"]=max(entry["rss_peak_kib"],process.get("VmRSS_kib",0));entry["hwm_peak_kib"]=max(entry["hwm_peak_kib"],process.get("VmHWM_kib",0))
                if s["mem_available_kib"]<limits["min_mem_available_kib"]:stop="STOPPED_RAM_MARGIN"
                elif s["gpu_free_mib"]<limits["min_gpu_free_mib"]:stop="STOPPED_GPU_MARGIN"
                elif s["oom_kill"]>pre["oom_kill"]:stop="STOPPED_SYSTEM_OOM_COUNTER"
                elif time.monotonic()-start>limits["timeout_s"]:stop="STOPPED_TIME_LIMIT"
                if stop:break
                time.sleep(limits["poll_sleep_s"])
        except BaseException as exc:
            stop="STOPPED_MONITOR_ERROR: "+repr(exc)
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
            code=p.wait()
            # A crashed client must not leave its model server running.
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    post=snapshot(start)
    if post["oom_kill"]>pre["oom_kill"]:stop=stop or "STOPPED_SYSTEM_OOM_COUNTER"
    r.update(status=stop or ("AWAITING_EVIDENCE_AUDIT" if code==0 else "FAILED_PROCESS"),exit_code=code,ended_at_utc=stamp(),wall_seconds=time.monotonic()-start,postflight=post,sample_count=len(samples),sampled_gpu_peak_mib=max(s["gpu_used_mib"] for s in samples),sampled_gpu_free_min_mib=min(s["gpu_free_mib"] for s in samples),sampled_mem_available_min_kib=min(s["mem_available_kib"] for s in samples),sampled_tree_rss_peak_kib=max(s["tree_rss_kib"] for s in samples),sampled_swap_peak_kib=max(s["swap_used_kib"] for s in samples),per_process_peaks=process_peaks)
    write(rp,r)
    if r["status"]=="AWAITING_EVIDENCE_AUDIT":
        try:
            server_log=(out/"server.log").read_text()
            assert "CudaDevice" in server_log and "Loading self_trained checkpoint" in server_log, "GPU/self-trained identity missing from server log"
            r["server_gpu_identity_verified"]=True
            r["episode_audit"]=audit_episode(out)
            r["pair_checks"]=pair_checks(c)
            verify_hashes(m["checkpoint_files"])
            assert sha(MANIFEST)==registration_sha
            assert sha(ROOT/c["config_path"])==c["config_sha256"]
            assert not heavy_processes(),"Process cleanup incomplete"
            r["files"]=tree(out)
            r["status"]="ACCEPTED"
        except Exception as exc:
            import traceback
            r.update(status="FAILED_EVIDENCE_AUDIT",audit_error=repr(exc),audit_traceback=traceback.format_exc())
        write(rp,r)
    print(json.dumps(r,ensure_ascii=False,indent=2))
    return 0 if r["status"]=="ACCEPTED" else 3


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["prepare","run"]);p.add_argument("--case",type=int,choices=[0,1]);args=p.parse_args()
    if args.mode=="prepare":prepare();return 0
    assert args.case is not None,"Use --case 0 or 1; no automatic batch expansion"
    return run_case(args.case)


if __name__=="__main__":
    raise SystemExit(main())
