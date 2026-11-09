from pathlib import Path
import argparse, dataclasses, json, logging, os, sys
os.environ.setdefault('JAX_PLATFORMS','cuda')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--port',type=int,default=8066); args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    import jax
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.training import config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    assert jax.default_backend()=='gpu'
    native=config.get_config('pi0_libero_low_mem_finetune')
    if cfg['model_role']=='base':
        native=dataclasses.replace(native,model=dataclasses.replace(native.model,
            paligemma_variant='gemma_2b',action_expert_variant='gemma_300m'))
        policy=policy_config.create_trained_policy(native,Path(cfg['checkpoint']),norm_stats=normalize.load(Path(cfg['norm_stats']).parent))
    else:
        policy=policy_config.create_trained_policy(native,Path(cfg['checkpoint']))
    class EpisodePolicy:
        metadata={**policy.metadata,'run_id':cfg['run_id'],'model_role':cfg['model_role'],
                  'flow_seed_mode':'reset to 0 at each episode, matching original single-episode runner'}
        def infer(self,obs):
            obs=dict(obs)
            seed=obs.pop('_episode_flow_seed',None)
            if seed is not None:
                assert seed==0
                policy._rng=jax.random.key(0)
            return policy.infer(obs)
    logging.info('Checkpoint loaded on %s; role=%s',jax.devices(),cfg['model_role'])
    WebsocketPolicyServer(policy=EpisodePolicy(),host='127.0.0.1',port=args.port,metadata=EpisodePolicy.metadata).serve_forever()

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,force=True);main()
