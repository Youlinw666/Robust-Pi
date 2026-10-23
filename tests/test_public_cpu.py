"""Asset-free CPU integration checks; no model or simulator imports."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from data_tools.dataset_adapter import future_actions
from evaluation.case_manifest import EvalCase,CaseManifest
from evaluation.runner import EvalRunner
from analysis.metrics import summarize_results
from training.augmentation import VisualAugmentation,VisualAugmentationConfig
from training.augmented_dataset import AugmentedActionChunkDataset


def make_case(case_id="synthetic-0"):
    return EvalCase(case_id=case_id,task_id=0,init_state_id=0,prompt_id="synthetic",perturbation_id="id",flow_seed=0,checkpoint_id="NO_MODEL",max_steps=5,replan_steps=5)


def make_manifest():
    return CaseManifest(run_id="synthetic-cpu",suite="synthetic",split="development",status="READY",cases=(make_case(),),created_at_utc="2026-10-22T00:00:00Z")


def test_future_actions_never_wraps_into_another_episode():
    a=np.arange(21,dtype=np.float32).reshape(3,7)
    chunk,valid=future_actions(a,2,50)
    assert chunk.shape==(50,7) and valid.sum()==1
    np.testing.assert_array_equal(chunk,np.repeat(a[2:3],50,axis=0))
    with pytest.raises(IndexError):future_actions(a,3,50)


def test_augmentation_preserves_non_visual_labels_and_global_rng():
    sample=dict(image=np.full((12,12,3),128,dtype=np.uint8),wrist_image=np.full((12,12,3),64,dtype=np.uint8),state=np.arange(8,dtype=np.float32),actions=np.ones((50,7),dtype=np.float32),prompt="synthetic instruction",task_id=0,episode_index=1,frame_index=0)
    np.random.seed(99);before=np.random.get_state()
    dataset=AugmentedActionChunkDataset([sample],visual=VisualAugmentation(VisualAugmentationConfig(probability=1,seed=7)))
    first=dataset[0];second=dataset[0]
    for k in ["state","actions"]:np.testing.assert_array_equal(first[k],sample[k])
    for k in ["image","wrist_image"]:np.testing.assert_array_equal(first[k],second[k])
    assert first["prompt"]==sample["prompt"]
    after=np.random.get_state();np.testing.assert_array_equal(before[1],after[1]);assert before[2:]==after[2:]


def test_checkpoint_binding_does_not_change_case_identity():
    m=make_manifest();bound=m.bind_checkpoint("ANOTHER_SYNTHETIC_ID",run_id="rebound")
    assert m.identity_hash==bound.identity_hash and m.manifest_hash!=bound.manifest_hash
    with pytest.raises(ValueError):CaseManifest(run_id="x",suite="synthetic",split="development",status="READY",cases=(make_case(),make_case()),created_at_utc="synthetic")


def test_timeout_and_system_error_have_different_denominators(tmp_path):
    timeout=EvalRunner(make_manifest(),tmp_path/"timeout",resume=False).run(lambda c:dict(status="timeout",action_steps=5,inference_count=1,latency_ms=[1.]))
    def fail(c):raise RuntimeError("synthetic system error")
    error=EvalRunner(make_manifest(),tmp_path/"error",resume=False).run(fail)
    result=summarize_results([*timeout,*error])
    assert result["episodes_total"]==2 and result["episodes_valid"]==1
    assert result["timeouts"]==1 and result["system_errors"]==1 and result["success_rate"]==0


def test_public_results_retain_real_negative_outcomes():
    d=json.loads((Path(__file__).parents[1]/"reports/public_results.json").read_text())
    assert len(d["cases"])==4 and d["ood_retention"] is None
    assert all(c["success"] is False and c["timeout"] is True and c["system_error"] is None for c in d["cases"])
    assert {c["evidence"] for c in d["cases"]}=={"reused","new"}
    assert all(c["action_steps"]==220 and c["inference_count"]==44 for c in d["cases"])


def test_no_model_or_simulator_was_imported():
    assert not any(n in sys.modules for n in ["jax","torch","tensorflow","mujoco","robosuite","libero"])
