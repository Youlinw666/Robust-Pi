#!/usr/bin/env python3
"""CPU-only G3 diagnosis from saved arrays and selected pinned source AST nodes.

Does not import JAX/Torch, load weights, simulate, or modify original evidence.
Use --output-dir to write a fresh audit; existing outputs are refused.
"""
from __future__ import annotations
import argparse
import ast
import dataclasses
from datetime import datetime,timezone
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import sys
from types import SimpleNamespace
import numpy as np
import h5py
import pyarrow.parquet as pq
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data_tools.dataset_adapter import DatasetAdapter,future_actions


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def source_nodes(path,names,ns):
    s=Path(path).read_text();tree=ast.parse(s);found=[]
    for n in tree.body:
        if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names:
            found.append(n)
    assert {n.name for n in found}==set(names)
    code=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),*found],type_ignores=[])
    exec(compile(ast.fix_missing_locations(code),str(path),"exec"),ns)


def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=ROOT/"reports/g3_static_audit_20261022_v2");args=p.parse_args()
    out=args.output_dir
    assert not out.exists(),"Refuse overwrite"
    out.mkdir(parents=True)
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),mode="CPU_STATIC_NO_MODEL_NO_SIMULATION",model_loaded=False,gpu_rollout_started=False)
    statsfile=ROOT/"artifacts/m2_formal_norm_stats/pi0/40shot/norm_stats.json"
    stats=read(statsfile)["norm_stats"];ss=SimpleNamespace(**{k:np.asarray(v) for k,v in stats["state"].items()});aa=SimpleNamespace(**{k:np.asarray(v) for k,v in stats["actions"].items()})
    inv=read(ROOT/"reports/m2_formal_inventory.json");manifest=read(ROOT/"manifests/m2_dataset_manifest.json")
    adapter=DatasetAdapter(ROOT/"data/libero_offset_diagnostic/f13aa24a3da8c43c7225569f28c562979fa0e35a",ROOT/"reports/m2_formal_inventory.json",manifest=ROOT/"manifests/m2_dataset_manifest.json",split="train",shots=40)
    e=adapter.load_episode(0,0);sample=e.sample(0,50)
    rawpath=ROOT/"data/libero_original"/inv["revision"]/adapter.files[0]["source_file"]
    ns=dict(np=np,dataclasses=dataclasses,DataTransformFn=object,transforms=SimpleNamespace(DataTransformFn=object))
    transforms=ROOT/"external/openpi/src/openpi/transforms.py"
    source_nodes(transforms,["Normalize","Unnormalize","DeltaActions","AbsoluteActions","PadStatesAndActions","pad_to_dim"],ns)
    source_nodes(ROOT/"external/openpi/src/openpi/policies/libero_policy.py",["LiberoOutputs"],ns)
    delta=ns["DeltaActions"]([True]*6+[False]);absolute=ns["AbsoluteActions"]([True]*6+[False])
    normal=ns["Normalize"](None);unnormal=ns["Unnormalize"](None)
    numeric=delta(dict(state=sample["state"].copy(),actions=sample["actions"].copy()))
    state_z=normal._normalize(numeric["state"],ss);action_z=normal._normalize(numeric["actions"],aa)
    padded=ns["PadStatesAndActions"](32)(dict(state=state_z,actions=action_z))
    decoded={"state":unnormal._unnormalize(padded["state"].astype(np.float32),ss),"actions":unnormal._unnormalize(padded["actions"].astype(np.float32),aa)}
    recovered=ns["LiberoOutputs"]()(absolute(decoded))["actions"]
    error=float(np.max(np.abs(recovered-sample["actions"])))
    assert error<1e-5
    report["action_roundtrip"]=dict(uses_actual_source_AST=True,raw_shape=list(sample["actions"].shape),padded_shape=list(padded["actions"].shape),output_shape=list(recovered.shape),max_abs_error_with_float32_boundary=error,legacy_extra_delta=True,normalization="mean/std",stats_sha256=sha(statsfile),gripper_excluded_from_extra_delta=True)
    with h5py.File(rawpath,"r") as f:
        d=f["data/demo_0"];attrs=f["data"].attrs;env=json.loads(attrs["env_args"])
        np.testing.assert_array_equal(e.actions,d["actions"][1:].astype(np.float32))
        np.testing.assert_array_equal(e.state,np.concatenate([d["obs/ee_states"][:-1],d["obs/gripper_states"][:-1]],axis=1).astype(np.float32))
        raw=d["obs/agentview_rgb"][0];wrist=d["obs/eye_in_hand_rgb"][0]
        np.testing.assert_array_equal(sample["image"],raw);np.testing.assert_array_equal(sample["wrist_image"],wrist)
        imagecheck=dict(train_parquet_equals_raw_hdf5=True,train_equals_rot180=bool(np.array_equal(sample["image"],raw[::-1,::-1])),train_image_shape=list(raw.shape),raw_macros_image_convention=attrs["macros_image_convention"],train_episode_id=0,parquet_sha256=sha(e.path),conversion_file="scripts/m2_convert_original_offset.py",adapter_has_no_rotation=True,rollout_source_rotation="obs[key][::-1, ::-1]",confirmed_mismatch="TRAIN_RAW_VS_ROLLOUT_ROT180")
        report["image_orientation"]=imagecheck
        report["dataset_action_contract"]=dict(env_controller=env["env_kwargs"]["controller_configs"],control_freq=env["env_kwargs"]["control_freq"],raw_episode_actions_equal_parquet_offset=True,state_mapping_equal=True,temporal_contract=inv["temporal_contract"],reference_demo_id="demo_0",raw_source=str(rawpath.relative_to(ROOT)),raw_source_sha256_from_inventory=read(ROOT/"reports/m2_original_hdf5_inventory.json")["selected_files"][0].get("sha256"))
        demo_init=np.asarray(d.attrs["init_state"],dtype=np.float64)
        init_hash=hashlib.sha256(demo_init.tobytes()).hexdigest()
        all_init_hashes={name:hashlib.sha256(np.asarray(g.attrs["init_state"],dtype=np.float64).tobytes()).hexdigest() for name,g in f["data"].items()}
        demo_trace=dict(episode_id=0,actions=e.actions.tolist(),eef_positions=e.state[:,:3].tolist(),gripper_positions=e.state[:,6:].tolist(),state=e.state.tolist(),fps=inv["source_fps"],init_state_sha256=init_hash,recorded_replay_success=adapter.files[0]["replay_success"],not_new_replay=True)
    # Recompute only numeric mean/std from 200 training episodes, no images/weights.
    sums={"state":np.zeros(8),"actions":np.zeros(7)};squares={k:np.zeros_like(v) for k,v in sums.items()};counts={"state":0,"actions":0};raw_min=np.full(7,np.inf);raw_max=-raw_min.copy()
    for eid in adapter.episode_ids():
        item=adapter.files[eid];table=pq.read_table(adapter.root/item["path"],columns=["state","actions"])
        state=np.asarray(table["state"].to_pylist(),dtype=np.float32);actions=np.asarray(table["actions"].to_pylist(),dtype=np.float32)
        raw_min=np.minimum(raw_min,actions.min(0));raw_max=np.maximum(raw_max,actions.max(0))
        sums["state"]+=state.astype(float).sum(0);squares["state"]+=np.square(state.astype(float)).sum(0);counts["state"]+=len(state)
        for i in range(len(actions)):
            chunk,_=future_actions(actions,i,50);chunk=delta(dict(state=state[i],actions=chunk))["actions"].astype(float)
            sums["actions"]+=chunk.sum(0);squares["actions"]+=np.square(chunk).sum(0);counts["actions"]+=len(chunk)
    check={}
    for k in sums:
        mean=sums[k]/counts[k];std=np.sqrt(np.maximum(0,squares[k]/counts[k]-mean*mean))
        mean_error=float(np.abs(mean-np.asarray(stats[k]["mean"])).max());std_error=float(np.abs(std-np.asarray(stats[k]["std"])).max())
        assert mean_error<1e-6 and std_error<1e-6
        check[k]=dict(vectors=counts[k],max_mean_error=mean_error,max_std_error=std_error)
    report["numeric_stats_recheck"]=dict(episodes=200,checks=check,all_start_frames=counts["state"],tail_repeat_weighting=True,raw_action_min=raw_min.tolist(),raw_action_max=raw_max.tolist(),stats_not_modified=True)
    g2=read(ROOT/"reports/g2_e1_case_results.json");cases=[];traces=[]
    for c in g2["cases"]:
        dr=ROOT/c["artifact_dir"];r=read(dr/"result.json")
        events=[json.loads(l) for l in (dr/"events.jsonl").read_text().splitlines()];steps=[x for x in events if x["event"]=="step"];replans=[x for x in events if x["event"]=="replan"]
        with np.load(dr/"raw_obs_0000.npz",allow_pickle=False) as obs:target=obs["akita_black_bowl_1_pos"].copy();initial_eef=obs["robot0_eef_pos"].copy()
        positions=np.asarray([x["next_eef_pos"] for x in steps]);actions=np.asarray([x["action"] for x in steps]);distance=np.linalg.norm(positions-target,axis=1)
        state_zs=[];deltas=[]
        for ev in replans:
            with np.load(dr/ev["policy_input"],allow_pickle=False) as a:state_zs.append((a["state"]-ss.mean)/(ss.std+1e-6))
            with np.load(dr/ev["raw_observation"],allow_pickle=False) as raw:deltas.append(float(np.linalg.norm(raw["akita_black_bowl_1_pos"]-target)))
        initial_match={name:v for name,v in all_init_hashes.items() if v==r["init_state"]["sha256"]}
        controller=r["controller"];cfg=report["dataset_action_contract"]["env_controller"]
        assert controller["control_freq"]==20 and controller["input_min"]==[-1.]*6 and controller["input_max"]==[1.]*6
        assert controller["output_min"]==cfg["output_min"] and controller["output_max"]==cfg["output_max"]
        row=dict(case_id=c["case_id"],condition=c["condition"],init_state_id=c["init_state_id"],initial_eef_to_initial_bowl_m=float(np.linalg.norm(initial_eef-target)),min_recorded_post_action_eef_to_initial_bowl_m=float(distance.min()),min_at_step=int(distance.argmin()),final_eef_to_initial_bowl_m=float(distance[-1]),fraction_post_action_farther_than_initial=float((distance>np.linalg.norm(initial_eef-target)).mean()),target_max_drift_at_replans_m=max(deltas),controller_contract_fields_match=True,state_abs_z_max_per_dimension=np.abs(state_zs).max(0).tolist(),rotation_cmd_abs_gt1_fraction=float((np.abs(actions[:,3:6])>1).mean()),translation_cmd_abs_gt1_fraction=float((np.abs(actions[:,:3])>1).mean()),matching_hdf5_initial_state_hashes=initial_match,reference_pointwise_comparison_supported=bool(initial_match))
        cases.append(row);traces.append(dict(case_id=c["case_id"],condition=c["condition"],init_state_id=c["init_state_id"],eef_positions=positions.tolist(),distance_to_initial_target=distance.tolist(),initial_target=target.tolist()))
    report["rollout_diagnosis"]=cases
    run=read(ROOT/"artifacts/m3_runs/m3_e1_pi0_20261021_params_only_01/run.json");lossfile=ROOT/run["loss_history"]
    losses=np.asarray([json.loads(l)["loss"] for l in lossfile.read_text().splitlines()]);assert len(losses)==100 and np.isfinite(losses).all()
    init_lr=2.5e-5/1001;last_lr=init_lr+(2.5e-5-init_lr)*99/1000
    report["pilot_scope"]=dict(steps=100,batch_size=1,episodes=200,anchors=counts["state"],sampled_anchor_fraction_upper_bound=100/counts["state"],warmup_steps=1000,estimated_default_lr_at_step0=init_lr,estimated_default_lr_at_step99=last_lr,lr_schedule_from_source=True,loss_first10_mean=float(losses[:10].mean()),loss_last10_mean=float(losses[-10:].mean()),loss_min=float(losses.min()),loss_max=float(losses.max()),loss_sha256=sha(lossfile),warning="No held-out loss or converged-training evidence; different random batches/noise/time make train loss non-comparable as success")
    report["lora_evidence_limit"]=dict(frozen_base_match=17,trainable_non_lora_base_changed=33,nonzero_lora_leaves=20,lora_initialization="both A/B use normal stddev0.01 in this pinned implementation",adapter_nonzero_is_not_proof_of_update=True,initial_adapter_snapshot_unavailable=True)
    sources=["scripts/m3_formal_train.py","scripts/m3_formal_loader_smoke.py","scripts/m2_convert_original_offset.py","data_tools/dataset_adapter.py","data_tools/openpi_bridge.py","scripts/libero_teaching.py","scripts/serve_e0.py","external/openpi/src/openpi/models/pi0.py","external/openpi/src/openpi/models/pi0_config.py","external/openpi/src/openpi/models/gemma.py","external/openpi/src/openpi/models/lora.py","external/openpi/src/openpi/models/model.py","external/openpi/src/openpi/transforms.py","external/openpi/src/openpi/training/config.py","external/openpi/src/openpi/training/optimizer.py","external/openpi/src/openpi/policies/policy.py","external/openpi/src/openpi/policies/policy_config.py","external/openpi/src/openpi/policies/libero_policy.py","external/openpi/third_party/libero/scripts/create_dataset.py",".venv-libero/lib/python3.8/site-packages/robosuite/controllers/osc.py",".venv-libero/lib/python3.8/site-packages/robosuite/controllers/base_controller.py",".venv-libero/lib/python3.8/site-packages/robosuite/utils/control_utils.py",".venv-libero/lib/python3.8/site-packages/robosuite/macros.py"]
    report["source_hashes"]={n:sha(ROOT/n) for n in sources}
    report["source_anchors"]={}
    anchors=["class LeRobotLiberoDataConfig","pi0_libero_low_mem_finetune","def create(","def get_freeze_filter","class Pi0(","def embed_prefix","def embed_suffix","def compute_loss","def sample_actions","class DeltaActions","class AbsoluteActions","class Normalize","class Unnormalize","class LiberoOutputs","class LiberoInputs","def policy_input","def scale_action","def set_goal(","def set_goal_orientation","def transform_dataset","class CosineDecaySchedule","class FormalSampleDataset","def _build_data","def _save_inference_checkpoint","def main(","def compute_pool_stats"]
    for n in sources:
        matches=[dict(line=i,text=l.strip()) for i,l in enumerate((ROOT/n).read_text().splitlines(),1) if any(a in l for a in anchors)]
        if matches:report["source_anchors"][n]=matches
    report["max_rss_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert "jax" not in sys.modules and "torch" not in sys.modules
    (out/"audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    (out/"plot_data.json").write_text(json.dumps(dict(losses=losses.tolist(),reference_demo=demo_trace,rollout_traces=traces),indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ["source_hashes","source_anchors"]},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
