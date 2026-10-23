#!/usr/bin/env python3
"""Synthetic CPU protocol demo. Not model inference or a robot evaluation."""
from pathlib import Path
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_tools.dataset_adapter import future_actions
from evaluation.case_manifest import CaseManifest,EvalCase
from training.augmentation import VisualAugmentation,VisualAugmentationConfig


def main():
    actions=np.arange(21,dtype=np.float32).reshape(3,7)
    chunk,valid=future_actions(actions,1,50)
    rgb=np.arange(12*12*3,dtype=np.uint8).reshape(12,12,3)
    aug=VisualAugmentation(VisualAugmentationConfig(seed=7)).apply(rgb,sample_id="synthetic-cpu",view_id="base")
    case=EvalCase(case_id="synthetic-0",task_id=0,init_state_id=0,prompt_id="synthetic",perturbation_id="synthetic",flow_seed=0,checkpoint_id="NO_MODEL",max_steps=5,replan_steps=5)
    manifest=CaseManifest(run_id="synthetic-cpu",suite="synthetic",split="development",status="READY",cases=(case,),created_at_utc="2026-10-22T00:00:00Z")
    assert chunk.shape==(50,7) and valid.sum()==2 and aug.shape==rgb.shape
    assert not any(n in sys.modules for n in ["jax","torch","mujoco","robosuite"])
    print(json.dumps(dict(mode="SYNTHETIC_CPU_NO_MODEL_NO_SIMULATOR",chunk_shape=list(chunk.shape),real_synthetic_actions=int(valid.sum()),image_shape=list(aug.shape),synthetic_manifest_identity_hash=manifest.identity_hash,task_performance_claim=False),indent=2))


if __name__=="__main__":main()
