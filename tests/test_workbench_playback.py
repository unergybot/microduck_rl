"""Matched normalized PT/ONNX playback on exactly the same raw observations."""
import hashlib
import json
import pytest
import torch
from mjlab_microduck.workbench import resolve_experiment
from mjlab_microduck.workbench.playback import PlaybackPolicy,compare_playback


def sha(path):return 'sha256:'+hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def policies(tmp_path):
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    from mjlab_microduck.workbench.engine import build_configs
    from mjlab_microduck.rom.export_metadata import attach_microduck_metadata
    import onnx
    spec=resolve_experiment(dict(schemaVersion='ROM_TRAINING_EXPERIMENT_V1',recipeId='Mjlab-Velocity-Flat-MicroDuck',robotProfileId='rom-xl330-v1',config=dict(numEnvs=1,iterations=1,network=dict(actorHiddenDims=[16]))))
    _,agent=build_configs(spec);torch.manual_seed(17)
    actor=MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),{'actor':['actor']},'actor',14,hidden_dims=agent.actor.hidden_dims,activation=agent.actor.activation,obs_normalization=True,distribution_cfg=dict(agent.actor.distribution_cfg))
    actor.obs_normalizer._mean.fill_(.3);actor.obs_normalizer._std.fill_(2);actor.obs_normalizer._var.fill_(4)
    actor.eval();actor.requires_grad_(False)
    checkpoint=tmp_path/'checkpoint.pt';torch.save({'actor_state_dict':actor.state_dict()},checkpoint)
    exported=tmp_path/'policy.onnx'
    torch.onnx.export(actor.as_onnx(False),torch.zeros(1,61),str(exported),input_names=['obs'],output_names=['actions'],opset_version=17,dynamo=False)
    attach_microduck_metadata(exported,task_id=spec['taskId'],source_commit='a'*40,checkpoint='checkpoint.pt',run_identity='eval-fixture')
    model=onnx.load(str(exported))
    for key,value in {'microduck.experiment_config_digest':spec['configDigest'],'microduck.recipe_digest':spec['recipeDigest'],'microduck.robot_profile_id':spec['robotProfileId'],'microduck.action_id':spec['actionId'],'microduck.action_filter':'NONE','microduck.command_profile':json.dumps(spec['commandProfile'],sort_keys=True)}.items():
        entry=model.metadata_props.add();entry.key=key;entry.value=value
    onnx.save(model,str(exported))
    return spec,checkpoint,exported,actor


def test_matched_pt_onnx_use_frozen_normalizers_and_report_numeric_consistency(policies):
    spec,checkpoint,exported,actor=policies
    pt=PlaybackPolicy.from_checkpoint(checkpoint,spec,sha(checkpoint));onnx=PlaybackPolicy.from_onnx(exported,spec,sha(exported))
    samples=torch.stack((torch.zeros(61),torch.full((61,),.7),torch.linspace(-1,1,61)))
    before={k:v.clone() for k,v in pt.module.state_dict().items()}
    report=compare_playback(pt,onnx,samples)
    assert report['samples']==3 and report['maxAbsoluteError']<1e-5
    assert report['configDigest']==spec['configDigest'] and report['evaluation']=='CONSISTENCY_ONLY'
    assert report['ptSha256']==sha(checkpoint) and report['onnxSha256']==sha(exported)
    assert all(torch.equal(before[k],v) for k,v in pt.module.state_dict().items())
    from tensordict import TensorDict
    expected=actor(TensorDict({'actor':samples[:1]},batch_size=[1]))
    assert torch.allclose(pt.predict(samples[:1]),expected,atol=1e-6)
    assert not pt.module.training and all(not p.requires_grad for p in pt.module.parameters())


def test_playback_refuses_wrong_hash_layout_profile_and_unbaked_graph(policies,tmp_path):
    spec,checkpoint,exported,_=policies
    with pytest.raises(ValueError):PlaybackPolicy.from_checkpoint(checkpoint,spec,'sha256:'+'f'*64)
    alias=tmp_path/'link.pt';alias.symlink_to(checkpoint)
    with pytest.raises(ValueError):PlaybackPolicy.from_checkpoint(alias,spec,sha(checkpoint))
    other=resolve_experiment(dict(schemaVersion='ROM_TRAINING_EXPERIMENT_V1',recipeId=spec['recipeId'],robotProfileId='hd1910-reference-v1',config=spec['requestedConfig']))
    with pytest.raises(ValueError):PlaybackPolicy.from_onnx(exported,other,sha(exported))
    from tests.test_rom_bundle import write_minimal_onnx
    bare=write_minimal_onnx(tmp_path/'bare.onnx')
    with pytest.raises(ValueError):PlaybackPolicy.from_onnx(bare,spec,sha(bare))
    malformed=tmp_path/'missing-normalizer.pt';state=torch.load(checkpoint,weights_only=True)
    state['actor_state_dict']={k:v for k,v in state['actor_state_dict'].items() if not k.startswith('obs_normalizer.')};torch.save(state,malformed)
    with pytest.raises(ValueError):PlaybackPolicy.from_checkpoint(malformed,spec,sha(malformed))


def test_nonfinite_inputs_and_different_configs_cannot_be_compared(policies):
    spec,checkpoint,exported,_=policies
    pt=PlaybackPolicy.from_checkpoint(checkpoint,spec,sha(checkpoint));onnx=PlaybackPolicy.from_onnx(exported,spec,sha(exported))
    for bad in (torch.zeros(1,60),torch.full((1,61),float('nan'))):
        with pytest.raises(ValueError):pt.predict(bad)
        with pytest.raises(ValueError):onnx.predict(bad)
    onnx.config_digest='sha256:'+'f'*64
    with pytest.raises(ValueError):compare_playback(pt,onnx,torch.zeros(1,61))


@pytest.mark.parametrize('kind',['PT','ONNX'])
def test_actual_isolated_session_runs_the_verified_policy_and_binds_its_config(policies,kind):
    from mjlab_microduck.workbench.evaluation import EvaluationSession
    from tests.test_workbench_evaluation import owner
    spec,checkpoint,exported,_=policies
    policy=PlaybackPolicy.from_checkpoint(checkpoint,spec,sha(checkpoint)) if kind=='PT' else PlaybackPolicy.from_onnx(exported,spec,sha(exported))
    current=EvaluationSession(spec,owner(),policy=policy)
    try:
        result=current.tick()
        assert current.status(owner())['mode']==kind
        expected=policy.predict(torch.tensor(result['observation'])[None])
        assert result['action']==pytest.approx(expected[0].tolist())
    finally:current.close()
    request=dict(schemaVersion=spec['schemaVersion'],recipeId=spec['recipeId'],robotProfileId=spec['robotProfileId'],config={**spec['requestedConfig'],'seed':19})
    with pytest.raises(ValueError):EvaluationSession(resolve_experiment(request),owner(),policy=policy)


def test_large_finite_predictions_have_finite_consistency_metrics(policies,monkeypatch):
    import math
    spec,checkpoint,exported,_=policies
    pt=PlaybackPolicy.from_checkpoint(checkpoint,spec,sha(checkpoint));onnx=PlaybackPolicy.from_onnx(exported,spec,sha(exported))
    compare_playback(pt,onnx,torch.zeros(1,61))
    monkeypatch.setattr(pt,'predict',lambda row:torch.full((1,14),3e38))
    monkeypatch.setattr(onnx,'predict',lambda row:torch.full((1,14),-3e38))
    report=compare_playback(pt,onnx,torch.zeros(1,61))
    assert all(math.isfinite(report[key]) for key in ('maxAbsoluteError','meanAbsoluteError','rootMeanSquaredError'))
    assert report['maxAbsoluteError']>5e38
