"""Bounded CPU checkpoint, teacher and actual state observation contracts."""
from types import SimpleNamespace
import json
import threading
import time
import pytest
import torch
from mjlab_microduck.workbench import canonical_json,resolve_experiment
from mjlab_microduck.workbench.observer import LatestObserver
from mjlab_microduck.workbench.checkpoints import attach_teachers,trusted_checkpoint,compatible
from mjlab_microduck.managed_training import experiment,verify_image_experiment,publish_checkpoint,atomic_json


def make_spec(recipe='Mjlab-Velocity-Flat-MicroDuck',**config):
    return resolve_experiment(dict(schemaVersion='ROM_TRAINING_EXPERIMENT_V1',recipeId=recipe,robotProfileId='rom-xl330-v1',config=config))


def test_experiment_is_canonical_and_parameters_bound():
    spec=make_spec(numEnvs=64,iterations=5,seed=7)
    job=dict(experimentJson=canonical_json(spec),numEnvs=64,iterations=5,seed=7)
    assert experiment(job)==spec
    for changes in ({'seed':9},{'experimentJson':json.dumps(spec)},{'experimentJson':'{"recipeId":1,"recipeId":2}'}):
        with pytest.raises(ValueError):experiment({**job,**changes})
    with pytest.raises(ValueError):verify_image_experiment({},spec)


def test_actual_state_latest_frames_are_bounded_nonblocking_and_readonly(tmp_path):
    position=torch.arange(20*2*3).reshape(20,2,3).float();matrix=torch.zeros(20,2,3,3)
    env=SimpleNamespace(num_envs=20,step_dt=.02,common_step_counter=4,sim=SimpleNamespace(data=SimpleNamespace(geom_xpos=position,geom_xmat=matrix)))
    clock=[1.];entered=threading.Event();release=threading.Event();frames=[]
    def sink(frame):frames.append(frame);entered.set();release.wait(2)
    job=dict(jobId='job-a',requestDigest='sha256:'+'a'*64,limits=dict(modelDigest='sha256:'+'b'*64))
    observer=LatestObserver(env,tmp_path,job,make_spec(),clock=lambda:clock[0],sink=sink)
    try:
        original=position.clone();observer.sample();assert entered.wait(1)
        observer.sample();assert observer.sequence==1
        for i in range(10):clock[0]+=.051;observer.sample()
        assert observer.sequence==11 and observer.pending is not None
        assert torch.equal(position,original) and env.common_step_counter==4
        assert frames[0]['environmentIndices']==list(range(9))
        assert frames[0]['positions'][0]==original[0].flatten().tolist()
        assert frames[0]['jobId']=='job-a' and frames[0]['configDigest']==make_spec()['configDigest']
    finally:release.set();observer.close()


def test_observer_sink_failure_is_optional(tmp_path):
    env=SimpleNamespace(num_envs=1,step_dt=.02,common_step_counter=1,sim=SimpleNamespace(data=SimpleNamespace(geom_xpos=torch.zeros(1,2,3),geom_xmat=torch.zeros(1,2,3,3))))
    observer=LatestObserver(env,tmp_path,dict(jobId='a',requestDigest='b',limits=dict(modelDigest='c')),make_spec(),sink=lambda frame:(_ for _ in ()).throw(OSError()))
    observer.sample();observer.close();assert env.common_step_counter==1


def test_teacher_uses_raw_observations_and_frozen_own_normalizers():
    from tensordict import TensorDict
    class Teacher(torch.nn.Module):
        def __init__(self,bias):super().__init__();self.bias=torch.nn.Parameter(torch.tensor(bias),requires_grad=False);self.seen=[]
        def forward(self,obs):self.seen.append(obs['actor'].clone());return (obs['actor'][:,:14]-self.bias)/2
    walk,stand=Teacher(2.),Teacher(8.)
    actor=torch.nn.Linear(61,14,bias=False)
    observations=TensorDict({'actor':torch.ones(1,2,61)*4},batch_size=[1,2]);observations['actor'][0,0,5]=-1
    original=observations.clone();optimizer=torch.optim.SGD(actor.parameters(),lr=.01)
    class Actor(torch.nn.Module):
        def __init__(self):super().__init__();self.model=actor
        def forward(self,obs):return actor(obs['actor'])
    alg=SimpleNamespace(storage=SimpleNamespace(observations=observations),actor=Actor(),optimizer=optimizer,max_grad_norm=1.,update=lambda:{'ppo':1})
    runner=SimpleNamespace(alg=alg);before=actor.weight.detach().clone()
    attach_teachers(runner,{'walk':walk,'stand':stand});result=alg.update()
    assert result['teacher_loss']>0 and result['ppo']==1
    assert not torch.equal(actor.weight,before)
    assert torch.equal(walk.seen[0],original['actor'].flatten(0,1))
    assert walk.bias.grad is None and stand.bias.grad is None
    assert torch.equal(observations['actor'],original['actor'])


def test_sibling_artifact_hash_scope_and_physics_are_required(tmp_path):
    spec=make_spec();source=tmp_path/'parent';source.mkdir();directory=tmp_path/'child';directory.mkdir()
    scope=dict(tenantId=1,operatorId=2,containerId='old')
    atomic_json(source/'request.json',dict(scope=scope,limits=dict(modelDigest='model',recipeDigest='recipe',dependencyDigest='deps'),experimentJson=canonical_json(spec)));atomic_json(source/'experiment.json',spec)
    checkpoint=source/'model.pt';checkpoint.write_bytes(b'checkpoint')
    artifact=publish_checkpoint(checkpoint,source)
    ref=dict(jobId='parent',checkpointSha256=artifact['sha256'],configDigest=spec['configDigest'])
    job=dict(scope={**scope,'containerId':'new'},limits=dict(modelDigest='model',recipeDigest='recipe',dependencyDigest='deps'))
    assert trusted_checkpoint(directory,job,ref)[1]==spec
    for changes in ({'scope':{**scope,'tenantId':9},'limits':dict(modelDigest='model',recipeDigest='recipe',dependencyDigest='deps')},):
        with pytest.raises(ValueError):trusted_checkpoint(directory,changes,ref)
    with pytest.raises(ValueError):compatible(spec,make_spec(gains=dict(kpFirmware=190)))
    (source/artifact['artifactId']).write_bytes(b'tampered')
    with pytest.raises(ValueError):trusted_checkpoint(directory,job,ref)


def test_frozen_normalized_teacher_loads_real_checkpoint_and_rejects_shape(tmp_path):
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    from mjlab_microduck.workbench.checkpoints import load_teachers
    from mjlab_microduck.workbench.engine import build_configs
    refs={};scope=dict(tenantId=1,operatorId=2,environment='SIMULATOR')
    for role,recipe in [('walk','Mjlab-Velocity-Flat-MicroDuck'),('stand','Mjlab-StandUp-Flat-MicroDuck')]:
        source=make_spec(recipe,network=dict(actorHiddenDims=[16]))
        _,agent=build_configs(source)
        actor=MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),{'actor':['actor']},'actor',14,hidden_dims=agent.actor.hidden_dims,activation=agent.actor.activation,obs_normalization=True,distribution_cfg=dict(agent.actor.distribution_cfg))
        directory=tmp_path/role;directory.mkdir();atomic_json(directory/'request.json',dict(scope=scope,limits=dict(modelDigest='model',recipeDigest='recipe',dependencyDigest='deps'),experimentJson=canonical_json(source)));atomic_json(directory/'experiment.json',source)
        torch.save({'actor_state_dict':actor.state_dict()},directory/'model.pt')
        artifact=publish_checkpoint(directory/'model.pt',directory)
        refs[role]=dict(jobId=role,checkpointSha256=artifact['sha256'],configDigest=source['configDigest'])
    joint=make_spec('Mjlab-VelStand-Flat-MicroDuck',teacherCheckpoints=refs)
    current=tmp_path/'joint';current.mkdir()
    teachers=load_teachers(joint,dict(scope=scope,limits=dict(modelDigest='model',recipeDigest='recipe',dependencyDigest='deps')),current,'cpu')
    assert set(teachers)=={'walk','stand'}
    assert all(not p.requires_grad for model in teachers.values() for p in model.parameters())
    assert all(model.obs_normalization and not model.training for model in teachers.values())
    bad=make_spec('Mjlab-VelStand-Flat-MicroDuck',teacherCheckpoints=refs,actionScale=.5)
    with pytest.raises(ValueError):load_teachers(bad,dict(scope=scope,limits=dict(modelDigest='model',recipeDigest='recipe',dependencyDigest='deps')),current,'cpu')


def test_recipe_aware_bundle_is_v2_and_preserves_image_and_request_pins(tmp_path):
    import zipfile
    from tests.test_rom_bundle import write_normalized_onnx
    from mjlab_microduck.workbench.bundle import publish_bundle
    from mjlab_microduck.rom.managed_bundle import verify_bundle
    source=write_normalized_onnx(tmp_path/'policy.onnx');output=tmp_path/'policy.zip'
    pins=dict(modelDigest='sha256:'+'a'*64,recipeDigest='sha256:'+'b'*64,configDigest='sha256:'+'c'*64,sourceCommit='d'*40)
    spec=make_spec('Mjlab-BallKickLeft-Flat-MicroDuck')
    publish_bundle(source,output,pins,spec)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist())=={'manifest.json','policy.onnx'}
        manifest=json.loads(archive.read('manifest.json'))
    assert set(manifest)=={'schema','policy','policySizeBytes','experiment'}
    assert manifest['schema']=='ROM_MANAGED_POLICY_BUNDLE_V2'
    metadata=manifest['policy'];assert metadata['recipeDigest']==pins['recipeDigest']
    assert metadata['configDigest']==pins['configDigest']
    assert metadata['experimentRecipeDigest']==spec['recipeDigest']
    assert metadata['experimentConfigDigest']==spec['configDigest']
    assert metadata['actionId']=='kick_left' and metadata['provenance']=='MANAGED'
    assert metadata['executionEnvironment']=='SIMULATOR' and metadata['normalizerBaked']
    with pytest.raises(ValueError):verify_bundle(output)


def test_actual_env_hook_does_not_advance_again_and_preserves_step_result(tmp_path):
    from mjlab_microduck.workbench.engine import build_configs
    from mjlab_microduck.workbench.observer import attach_observer
    env_cfg,_=build_configs(make_spec());model=env_cfg.scene.entities['robot'].build().compile()
    env=SimpleNamespace(num_envs=2,step_dt=.02,common_step_counter=0,sim=SimpleNamespace(mj_model=model,data=SimpleNamespace(geom_xpos=torch.zeros(2,model.ngeom,3),geom_xmat=torch.zeros(2,model.ngeom,3,3))))
    sentinel=object()
    def step(actions):env.common_step_counter+=1;return sentinel
    env.step=step
    observer=attach_observer(env,tmp_path,dict(jobId='a',requestDigest='b',limits=dict(modelDigest='c')),make_spec())
    try:
        assert env.step(None) is sentinel
        assert env.common_step_counter==1
        assert (tmp_path/'observer/geometry.json').is_file()
    finally:observer.close()


def test_experimental_resume_selects_retained_history_and_refuses_evicted_or_mismatched(tmp_path):
    from mjlab_microduck import managed_training as managed
    spec=make_spec();parent=tmp_path/'parent';parent.mkdir();current=tmp_path/'child';current.mkdir()
    limits=dict(modelDigest='model',recipeDigest='recipe',dependencyDigest='deps')
    scope=dict(tenantId=1,operatorId=2,containerId='old')
    atomic_json(parent/'request.json',dict(scope=scope,limits=limits,experimentJson=canonical_json(spec)))
    atomic_json(parent/'experiment.json',spec)
    checkpoint=parent/'model.pt';checkpoint.write_bytes(b'older complete generation')
    old=publish_checkpoint(checkpoint,parent,retain_history=True)
    checkpoint.write_bytes(b'latest complete generation')
    latest=publish_checkpoint(checkpoint,parent,retain_history=True)
    atomic_json(parent/'checkpoint-index.json',dict(configDigest=spec['configDigest'],checkpoints=[dict(iteration=1,artifact=old),dict(iteration=2,artifact=latest)]))
    job=dict(scope={**scope,'containerId':'new'},limits=limits,parentJobId='parent',checkpointSha256=old['sha256'])
    assert managed.select_resume_checkpoint(job,current,spec)==parent/old['artifactId']
    # A retained file without an available index entry is evicted, not resumable.
    atomic_json(parent/'checkpoint-index.json',dict(configDigest=spec['configDigest'],checkpoints=[dict(iteration=2,artifact=latest)],evictedCheckpoints=[dict(iteration=1,artifact=old)]))
    with pytest.raises(ValueError):managed.select_resume_checkpoint(job,current,spec)
    atomic_json(parent/'checkpoint-index.json',dict(configDigest=spec['configDigest'],checkpoints=[dict(iteration=1,artifact=old),dict(iteration=2,artifact=latest)]))
    with pytest.raises(ValueError):managed.select_resume_checkpoint(job,current,make_spec(actionScale=.5))
    (parent/old['artifactId']).write_bytes(b'tampered')
    with pytest.raises(ValueError):managed.select_resume_checkpoint(job,current,spec)
    # Legacy jobs remain intentionally latest-only.
    assert managed.select_resume_checkpoint({**job,'checkpointSha256':latest['sha256']},current,None)==parent/latest['artifactId']
    with pytest.raises(ValueError):managed.select_resume_checkpoint(job,current,None)
