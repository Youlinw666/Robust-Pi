"""Five-task clean rollout protocol from locked OpenPI; no model imports.

H50/K5, flow10, environment seed7, settle10, max220, official initial states.
Each episode resets policy flow seed to0 as in the original teaching runner.
"""
from pathlib import Path
import argparse,collections,hashlib,json,random,sys,time,traceback
import imageio
import numpy as np
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from scripts.libero_teaching import official_client,policy_input

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--output',required=True); parser.add_argument('--inventory',type=Path,required=True); parser.add_argument('--smoke',action='store_true'); parser.add_argument('--port',type=int,default=8066);args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(exist_ok=False)
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    official=official_client();random.seed(7);np.random.seed(7)
    suite=official.benchmark.get_benchmark_dict()['libero_spatial']()
    client=official._websocket_client_policy.WebsocketClientPolicy('127.0.0.1',args.port)
    assert client.get_server_metadata()['run_id']==cfg['run_id']
    inv=json.loads(args.inventory.read_text())
    all_results=[]
    start=time.monotonic()
    for taskid in ([0] if args.smoke else range(5)):
        task=suite.get_task(taskid)
        assert task.language==next(t['prompt'] for t in inv['spatial_tasks'] if t['task_id']==taskid)
        initstates=suite.get_task_init_states(taskid)
        assert len(initstates)==50
        env,prompt=official._get_libero_env(task,official.LIBERO_ENV_RESOLUTION,7)
        try:
            for initid in range(1 if args.smoke else 50):
                case='spatial_t%02d_i%02d_clean'%(taskid,initid)
                ep=out/case;ep.mkdir()
                row=dict(case_id=case,task_id=taskid,task_name=task.name,init_state_id=initid,
                         prompt=str(prompt),environment_seed=7,flow_seed=0,max_steps=220,settle_steps=10,
                         action_horizon=50,replan_steps=5,flow_steps=10,condition='clean',
                         success=False,status='STARTED',steps=0,inference_count=0,latencies_ms=[],model_role=cfg['model_role'])
                writer=None;episodestart=time.monotonic()
                try:
                    env.reset();state=np.asarray(initstates[initid]); row['init_state_sha256']=hashlib.sha256(state.tobytes()).hexdigest()
                    obs=env.set_init_state(initstates[initid])
                    for _ in range(10): obs,_,_,_=env.step(official.LIBERO_DUMMY_ACTION)
                    np.savez_compressed(ep/'initial_raw_obs.npz',**{k:v for k,v in obs.items() if isinstance(v,np.ndarray)})
                    writer=imageio.get_writer(str(ep/'rollout.mp4'),fps=20)
                    queue=collections.deque();done=False
                    with (ep/'trace.jsonl').open('x') as trace:
                        for step in range(220):
                            element=policy_input(official,obs,prompt)
                            writer.append_data(element['observation/image'])
                            if not queue:
                                if step==0:
                                    np.savez_compressed(ep/'initial_policy_input.npz',image=element['observation/image'],wrist_image=element['observation/wrist_image'],state=element['observation/state'],prompt=np.asarray(prompt))
                                    element['_episode_flow_seed']=0
                                t=time.monotonic();chunk=np.asarray(client.infer(element)['actions']);ms=(time.monotonic()-t)*1000
                                assert chunk.shape==(50,7) and np.isfinite(chunk).all(),chunk.shape
                                row['inference_count']+=1;row['latencies_ms'].append(ms)
                                if step==0:np.save(ep/'first_action_chunk.npy',chunk)
                                queue.extend((i,action) for i,action in enumerate(chunk[:5]))
                                trace.write(json.dumps(dict(event='replan',step=step,actions=chunk.tolist(),latency_ms=ms))+'\n')
                            index,action=queue.popleft();obs,reward,done,info=env.step(action.tolist())
                            trace.write(json.dumps(dict(event='step',step=step,chunk_action_index=index,action=action.tolist(),reward=float(reward),success=bool(done),eef_pos=obs['robot0_eef_pos'].tolist()))+'\n');trace.flush()
                            row['steps']=step+1
                            if done:break
                        writer.append_data(policy_input(official,obs,prompt)['observation/image'])
                    checked=bool(env.check_success());assert bool(done)==checked
                    row.update(status='success' if done else 'timeout',success=bool(done),checked_success=checked,
                               video_frames=row['steps']+1)
                except Exception as exc:
                    row.update(status='system_error',error=repr(exc),traceback=traceback.format_exc())
                finally:
                    if writer is not None:writer.close()
                    row['elapsed_seconds']=time.monotonic()-episodestart
                    (ep/'result.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))
                    with (out/'episodes.jsonl').open('a') as fp:fp.write(json.dumps(row,ensure_ascii=False)+'\n')
                    all_results.append(row)
                    print('EPISODE',case,row['status'],'steps',row['steps'],flush=True)
                if row['status']=='system_error':raise RuntimeError('Stop evaluation on system error: '+case)
        finally:env.close()
    tasks=[]
    for taskid in sorted(set(r['task_id'] for r in all_results)):
        rows=[r for r in all_results if r['task_id']==taskid]
        tasks.append(dict(task_id=taskid,task_name=rows[0]['task_name'],rollouts=len(rows),successes=sum(r['success'] for r in rows),success_rate=sum(r['success'] for r in rows)/len(rows)))
    result=dict(status='PASSED_SYSTEM_SMOKE' if args.smoke else 'COMPLETED',smoke_only=args.smoke,
                model_role=cfg['model_role'],condition='clean',task_results=tasks,rollouts=len(all_results),
                successes=sum(r['success'] for r in all_results),success_rate=sum(r['success'] for r in all_results)/len(all_results),
                elapsed_seconds=time.monotonic()-start,protocol_sources=cfg['protocol_sources'],
                comparison_scope='LIBERO Spatial task IDs 0-4 only; not full ten-task LIBERO Spatial benchmark')
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
