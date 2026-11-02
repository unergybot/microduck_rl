"""Content verified, private sibling artifacts; submitted host paths never accepted."""
import json
from pathlib import Path
from .configuration import verify_experiment,canonical_json
from .engine import checked


def compatible(source,target):
    for key in ('robotProfileId','backlash'):checked(source[key]==target[key])
    # Teachers have different reward/command recipes, but identical physical,
    # proprioception, action and empirical normalization contracts are required.
    a,b=source['effectiveConfig'],target['effectiveConfig']
    for key in ('servo','gains','calibration','physics','actionScale','actionFilter','controlFrequencyHz','observations'):
        checked(a.get(key)==b.get(key))
    checked(source['terrain']==target['terrain'])


def trusted_checkpoint(directory,job,ref):
    from mjlab_microduck.managed_training import artifact
    import re
    checked(type(ref.get('jobId')) is str and bool(re.fullmatch('[-A-Za-z0-9_]{1,80}',ref['jobId'])))
    for key in ('checkpointSha256','configDigest'):checked(type(ref.get(key)) is str and bool(re.fullmatch('sha256:[a-f0-9]{64}',ref[key])))
    root=Path(directory).parent/ref['jobId']
    checked(not root.is_symlink() and root.is_dir())
    for name in ('checkpoint.json','experiment.json','request.json'):
        checked((root/name).is_file() and not (root/name).is_symlink() and (root/name).stat().st_size<=131072)
    parent=json.loads((root/'request.json').read_text())
    def scope(value):return {k:v for k,v in value.items() if k!='containerId'}
    checked(scope(parent['scope'])==scope(job['scope']))
    checked(all(key in parent['limits'] and parent['limits'][key]==job['limits'].get(key) for key in ('modelDigest','recipeDigest','dependencyDigest')))
    spec=verify_experiment(json.loads((root/'experiment.json').read_text()))
    checked(spec['configDigest']==ref['configDigest'])
    checked(parent.get('experimentJson')==canonical_json(spec))
    manifest=json.loads((root/'checkpoint.json').read_text())
    candidates=[a for a in manifest['artifacts'] if a['kind']=='CHECKPOINT' and a['sha256']==ref['checkpointSha256']]
    # Bounded history generations remain usable after latest moves.
    index=root/'checkpoint-index.json'
    if index.is_file() and not index.is_symlink():
        history=json.loads(index.read_text());checked(history['configDigest']==spec['configDigest'])
        candidates += [e['artifact'] for e in history['checkpoints'] if e['artifact']['sha256']==ref['checkpointSha256']]
    checked(bool(candidates));entry=candidates[0]
    import re
    checked(bool(re.fullmatch('checkpoint-[a-f0-9]{64}\\.pt',entry['artifactId'])))
    path=root/entry['artifactId'];checked(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=256<<20)
    checked(artifact(path,'CHECKPOINT')==entry)
    return path,spec


def load_teachers(spec,job,directory,device):
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from .engine import build_configs
    refs=spec['effectiveConfig'].get('teacherCheckpoints',{})
    if not spec['teacherRequirements']:
        checked(not refs);return {}
    checked(set(refs)=={'walk','stand'})
    result={}
    for requirement in spec['teacherRequirements']:
        role=requirement['actionId'];path,source=trusted_checkpoint(directory,job,refs[role])
        checked(source['recipeId']==requirement['recipeId']);compatible(source,spec)
        _,agent=build_configs(source)
        actor=MLPModel(TensorDict({'actor':torch.zeros(1,61,device=device)},batch_size=[1]),{'actor':['actor']},'actor',14,hidden_dims=agent.actor.hidden_dims,activation=agent.actor.activation,obs_normalization=True,distribution_cfg=dict(agent.actor.distribution_cfg)).to(device)
        saved=torch.load(path,map_location=device,weights_only=True)
        state=saved['actor_state_dict']
        checked(any(k.startswith('obs_normalizer.') for k in state))
        actor.load_state_dict(state,strict=True);actor.eval();actor.requires_grad_(False)
        result[role]=actor
    return result


def attach_teachers(runner,teachers,coefficient=.1):
    """Independent bounded supervised step on the same PPO rollout, no extra steps.

    Teachers receive raw actor observations and apply their own frozen normalizer.
    Projected gravity z in raw proprioception chooses walk (upright) or recovery.
    """
    checked(set(teachers)=={'walk','stand'} and 0<coefficient<=1)
    import torch
    alg=runner.alg;update=alg.update
    def supervised_update():
        # Keep a bounded deterministic sample before PPO clears its rollout cursor.
        obs=alg.storage.observations.flatten(0,1)[:1024].detach().clone()
        result=update()
        with torch.no_grad():
            walk=teachers['walk'](obs);stand=teachers['stand'](obs)
            target=torch.where((obs['actor'][:,5]<-.5).unsqueeze(-1),walk,stand)
        prediction=alg.actor(obs)
        loss=coefficient*(prediction-target).square().mean()
        checked(bool(torch.isfinite(loss)))
        alg.optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(alg.actor.parameters(),alg.max_grad_norm)
        alg.optimizer.step();result['teacher_loss']=float(loss.detach())
        return result
    alg.update=supervised_update
