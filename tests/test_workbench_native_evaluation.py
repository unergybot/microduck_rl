"""Real CPU ONNX proof and native walking semantics, without a training engine."""
import copy
import json
import zipfile
import numpy as np
import pytest
from tests.managed_onnx_fixture import write_normalized_onnx
from mjlab_microduck.workbench import resolve_experiment
from mjlab_microduck.workbench.bundle import publish_bundle
from mjlab_microduck.rom.managed_bundle import publish_bundle as publish_v1
from mjlab_microduck.workbench.native_evaluation import verify_native_bundle, native_evaluation_support

PROFILE='microduck-walk-flat-native-v1'
def spec(recipe='Mjlab-Velocity-Flat-MicroDuck', profile='rom-xl330-v1', **config):
    return resolve_experiment(dict(schemaVersion='ROM_TRAINING_EXPERIMENT_V1',recipeId=recipe,robotProfileId=profile,config=config))
def pins():
    return dict(modelDigest='sha256:'+'a'*64,recipeDigest='sha256:'+'b'*64,configDigest='sha256:'+'c'*64,sourceCommit='d'*40,checkpointSha256='sha256:'+'e'*64)
def bundle(tmp_path,experiment):
    output=tmp_path/'policy.zip'
    publish_bundle(write_normalized_onnx(tmp_path/'policy.onnx'),output,pins(),experiment)
    return output

def test_v2_proves_real_graph_and_keeps_experiment_metadata(tmp_path):
    experiment=spec();path=bundle(tmp_path,experiment);before=path.read_bytes()
    result=verify_native_bundle(path)
    assert result['nativeEvaluation']==dict(supported=True,profileId=PROFILE,reasonCode='')
    assert result['metadata']['experimentConfigDigest']==experiment['configDigest']
    assert result['metadata']['qualified'] is False
    assert result['wireMetadata']['taskId']=='Mjlab-Velocity-Flat-MicroDuck'
    assert 'experimentConfigDigest' not in result['wireMetadata']
    assert path.read_bytes()==before

def test_v1_keeps_existing_metadata_and_profile(tmp_path):
    path=tmp_path/'v1.zip';publish_v1(write_normalized_onnx(tmp_path/'p.onnx'),path,pins())
    result=verify_native_bundle(path)
    assert result['metadata']==result['wireMetadata']
    assert result['nativeEvaluation']==dict(supported=True,profileId=PROFILE,reasonCode='')

@pytest.mark.parametrize('config',[dict(seed=9,iterations=6,learningRate=.002),dict(rewards={'action_rate_l2':{'weight':-.2}}),dict(observations={'critic':{'joint_pos':{'scale':2}}})])
def test_learning_changes_do_not_change_native_support(tmp_path,config):
    assert verify_native_bundle(bundle(tmp_path,spec(**config)))['nativeEvaluation']['supported']

@pytest.mark.parametrize('experiment,reason',[
    (spec(actionScale=.5),'POLICY_NATIVE_ACTION_UNSUPPORTED'),
    (spec(calibration={'imuPitchOffset':.1}),'POLICY_NATIVE_OBSERVATION_UNSUPPORTED'),
    (spec(calibration={'jointZeroOffsets':{'left_knee':.1}}),'POLICY_NATIVE_OBSERVATION_UNSUPPORTED'),
    (spec(observations={'actor':{'joint_pos':{'scale':2}}}),'POLICY_NATIVE_OBSERVATION_UNSUPPORTED'),
    (spec(observations={'actor':{'joint_pos':{'clip':[-1,1]}}}),'POLICY_NATIVE_OBSERVATION_UNSUPPORTED'),
    (spec(recipe='Mjlab-SitStand-Flat-MicroDuck'),'POLICY_NATIVE_PROFILE_UNSUPPORTED'),
    (spec(recipe='Mjlab-Velocity-Rough-MicroDuck'),'POLICY_NATIVE_PROFILE_UNSUPPORTED'),
    (spec(profile='hd1910-reference-v1'),'POLICY_NATIVE_PROFILE_UNSUPPORTED'),
])
def test_valid_other_policies_have_explicit_unsupported_result(tmp_path,experiment,reason):
    value=verify_native_bundle(bundle(tmp_path,experiment))
    assert value['nativeEvaluation']==dict(supported=False,profileId='',reasonCode=reason)
    assert value['metadata']['experimentConfigDigest']==experiment['configDigest']

@pytest.mark.parametrize('change',['experiment','onnx','unsafe_member'])
def test_tampered_bundle_is_invalid_not_unsupported(tmp_path,change):
    path=bundle(tmp_path,spec())
    with zipfile.ZipFile(path) as z:manifest=json.loads(z.read('manifest.json'));data=z.read('policy.onnx')
    if change=='experiment':manifest['experiment']['effectiveConfig']['actionScale']=.5
    if change=='onnx':data=b'not an ONNX graph'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('manifest.json',json.dumps(manifest));z.writestr('../policy.onnx' if change=='unsafe_member' else 'policy.onnx',data)
    with pytest.raises(ValueError):verify_native_bundle(path)

def test_native_raw_vector_and_unfiltered_unit_scale_targets():
    from mjlab_microduck.rom.observation import DeploymentState,DeploymentCommand,DEFAULT_JOINT_POSE,build_actor_observation
    offsets=np.arange(14,dtype=np.float32)/100
    velocity=np.arange(14,dtype=np.float32)/10
    previous=-offsets
    state=DeploymentState([.1,.2,.3],[1,0,0,0],DEFAULT_JOINT_POSE+offsets,velocity,previous)
    command=DeploymentCommand([.05,0,.2],[0,0,0,0],[0,0,.01,.02,.03,0])
    expected=np.concatenate(([.1,.2,.3],[0,0,-1],offsets,velocity,previous,[.05,0,.2],[0,0,0,0],[0,0,.01,.02,.03,0]))
    np.testing.assert_allclose(build_actor_observation(state,command),expected,atol=1e-7)
    # Native qualification must use home + 1*raw action, without alpha lowpass.
    np.testing.assert_allclose(DEFAULT_JOINT_POSE+offsets, state.joint_positions_rad)
