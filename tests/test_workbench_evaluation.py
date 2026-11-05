"""Task-aware isolated evaluation; CPU simulation is not policy qualification."""
import json
import math
from pathlib import Path
import pytest
from mjlab_microduck.workbench import catalog,resolve_experiment
from mjlab_microduck.workbench.evaluation import EvaluationSession,command_schema,validate_commands


def spec(recipe='Mjlab-Velocity-Flat-MicroDuck',profile='rom-xl330-v1'):
    return resolve_experiment(dict(schemaVersion='ROM_TRAINING_EXPERIMENT_V1',recipeId=recipe,robotProfileId=profile,config=dict(numEnvs=1,iterations=1,seed=11)))


def owner():
    return dict(sessionId='session-1',operatorDigest='sha256:'+'a'*64,scopeDigest='sha256:'+'b'*64,modelDigest='sha256:'+'c'*64)


def test_command_surface_covers_all_catalog_semantics_without_velocity_relabeling():
    from mjlab_microduck.workbench.engine import build_configs
    semantics=set()
    recipes=catalog()['recipes'];assert len(recipes)==35
    for recipe in recipes:
        value=spec(recipe['recipeId']);surface=command_schema(value);semantics.add(surface['twistSemantics'])
        cfg,_=build_configs(value)
        terms={name for name,term in cfg.commands.items() if term is not None}
        for name in ('twist','head_pose','body_pose'):
            expected=name in terms and (name!='twist' or surface['twistSemantics'] not in ('phase-clock','autonomous-maneuver','passive-balance'))
            assert surface[name]['editable'] is expected,(recipe['recipeId'],name)
            if name not in terms:
                with pytest.raises(ValueError,match='EVALUATION_COMMAND_UNSUPPORTED'):
                    validate_commands(value,{name:[0]*surface[name]['dimensions']})
        assert surface['head_pose']['dimensions']==4 and surface['body_pose']['dimensions']==6
        if surface['twistSemantics'] in ('phase-clock','autonomous-maneuver','passive-balance'):
            assert surface['twist']['editable'] is False
            with pytest.raises(ValueError):validate_commands(value,{'twist':[0,0,0]})
        else:assert surface['twist']['editable'] is True
    assert len(semantics)>=6
    sit=spec('Mjlab-SitStand-Flat-MicroDuck')
    assert validate_commands(sit,{'twist':[1,0,0]})=={'twist':[1,0,0]}
    for value in ([.5,0,0],[1,.1,0],[2,0,0]):
        with pytest.raises(ValueError):validate_commands(sit,{'twist':value})


def test_actual_config_support_keeps_present_head_but_refuses_none_body_and_twist(monkeypatch):
    from mjlab_microduck.workbench import engine
    value=spec();cfg,agent=engine.build_configs(value)
    cfg.commands['body_pose']=None;cfg.commands['twist']=None
    monkeypatch.setattr(engine,'build_configs',lambda requested:(cfg,agent))
    surface=command_schema(value)
    assert surface['head_pose']['editable'] is True
    assert surface['body_pose']['editable'] is False and surface['twist']['editable'] is False
    assert validate_commands(value,{'head_pose':[0]*4})=={'head_pose':[0]*4}
    for commands in ({'body_pose':[0]*6},{'twist':[0]*3},{'head_pose':[0]*4,'body_pose':[0]*6}):
        with pytest.raises(ValueError,match='EVALUATION_COMMAND_UNSUPPORTED'):validate_commands(value,commands)


@pytest.mark.parametrize('commands',[{'unknown':[0]},{'twist':[float('nan'),0,0]},{'twist':[True,0,0]},{'twist':[100,0,0]},{'head_pose':[0]*3}])
def test_command_validation_refuses_nonfinite_wrong_shapes_and_outside_declared_envelope(commands):
    with pytest.raises(ValueError):validate_commands(spec(),commands)


@pytest.fixture
def session():
    current=EvaluationSession(spec(),owner(),clock=lambda:10.)
    try:yield current
    finally:current.close()


def test_actual_preview_pause_reset_disturbance_and_readonly_frames(session,monkeypatch):
    from mjlab_microduck.workbench import engine,evaluation
    def no_rebuild(*args,**kwargs):raise AssertionError('session operations must use their own command schema')
    monkeypatch.setattr(engine,'build_configs',no_rebuild)
    monkeypatch.setattr(evaluation,'command_schema',no_rebuild)
    identity=owner();assert session.status(identity)['mode']=='INITIAL_MODEL'
    surface=session.status(identity)['commandSchema']
    assert all(surface[name]['editable'] for name in ('twist','head_pose','body_pose'))
    surface['head_pose']['editable']=False;surface['body_pose']['ranges'][0][0]=999
    assert session.status(identity)['commandSchema']['head_pose']['editable'] is True
    assert session.status(identity)['commandSchema']['body_pose']['ranges'][0][0]!=999
    assert session.env.num_envs==1 and session.env.step_dt==.02
    initial=session.env.common_step_counter
    frame=session.frame(identity);geometry=session.geometry(identity)
    assert frame['configDigest']==spec()['configDigest'] and len(frame['positions'])==3*session.env.sim.mj_model.ngeom
    assert geometry['geometry'] and geometry['modelDigest']==identity['modelDigest']
    assert session.env.common_step_counter==initial
    session.control(identity,1,'PAUSE',{})
    assert session.tick() is None and session.env.common_step_counter==initial
    session.control(identity,2,'COMMAND',{'twist':[.1,0,0],'head_pose':[0]*4,'body_pose':[0]*6})
    before=session.env.scene['robot'].data.root_link_vel_w.clone()
    session.control(identity,3,'DISTURB',{'linearVelocityDelta':[.1,0,0],'angularVelocityDelta':[0,0,.2]})
    assert session.env.scene['robot'].data.root_link_vel_w[0].tolist()==pytest.approx((before[0]+before.new_tensor([.1,0,0,0,0,.2])).tolist())
    assert session.status(identity)['lastControlSequence']==3
    session.control(identity,4,'RUN',{});result=session.tick()
    assert result['step']==1 and math.isfinite(result['reward'])
    assert session.env.command_manager.get_command('twist')[0].tolist()==pytest.approx([.1,0,0])
    session.control(identity,5,'RESET',{})
    assert session.status(identity)['episodeStep']==0
    assert session.status(identity)['commands']['twist']==[.1,0,0]


def test_owner_sequence_and_lease_cannot_revive_or_cross_control(session):
    other={**owner(),'operatorDigest':'sha256:'+'d'*64}
    with pytest.raises(ValueError):session.control(other,1,'RUN',{})
    assert session.status(owner())['lastControlSequence']==0
    session.control(owner(),1,'PAUSE',{})
    with pytest.raises(ValueError):session.control(owner(),1,'RUN',{})
    assert session.tick() is None
    session.clock=lambda:101.
    assert session.tick() is None and session.closed
    with pytest.raises(ValueError):session.control(owner(),2,'RENEW',{'seconds':90})


def test_saved_episode_has_exact_inputs_states_and_no_implicit_qualification(session,tmp_path):
    session.control(owner(),1,'COMMAND',{'twist':[.1,0,0]})
    session.tick();session.tick();session.control(owner(),2,'PAUSE',{})
    path=tmp_path/'episode.json';session.save_replay(owner(),path)
    saved=json.loads(path.read_text())
    assert saved['experiment']==spec() and saved['identity']==owner()
    assert saved['evaluation']=='NOT_QUALIFIED' and saved['policy']['kind']=='INITIAL_MODEL'
    assert len(saved['steps'])==2 and len(saved['steps'][0]['observation'])==61
    assert len(saved['steps'][0]['action'])==14 and saved['steps'][0]['qpos']
    assert saved['controls'][0]['parameters']['twist']==[.1,0,0]
    assert saved['steps'][0]['step']==1 and saved['steps'][1]['simulationSeconds']==pytest.approx(.04)


def test_sitstand_uses_actual_posture_command_and_slew_not_walking_velocity():
    current=EvaluationSession(spec('Mjlab-SitStand-Flat-MicroDuck'),owner(),clock=lambda:10.)
    try:
        status=current.status(owner())
        assert status['commandSchema']['head_pose']['editable'] is True
        assert status['commandSchema']['body_pose']['editable'] is False
        with pytest.raises(ValueError,match='EVALUATION_COMMAND_UNSUPPORTED'):
            current.control(owner(),1,'COMMAND',{'head_pose':[0]*4,'body_pose':[0]*6})
        assert current.status(owner())==status
        term=current.env.command_manager.get_term('twist')
        current.control(owner(),1,'COMMAND',{'twist':[1,0,0],'head_pose':[0]*4});current.tick()
        assert type(term).__name__=='SitStandCommand' and term.command[0,0]==1
        assert term.alpha.shape==(1,)
        # Episode-start compute reinitializes from the actual spawn height.
        # Measure the held flag's slew after that initialization window.
        current.tick();before=term.alpha.clone();current.tick()
        assert 0<float(term.alpha[0]-before[0])<1
        assert current.spec['robotProfileId']=='rom-xl330-v1'
    finally:current.close()


def test_picking_uses_actual_continuous_phase_instead_of_held_velocity(monkeypatch):
    from copy import deepcopy
    import torch
    from mjlab_microduck.workbench import engine
    current=EvaluationSession(spec('Mjlab-GroundPick-Flat-MicroDuck'),owner(),clock=lambda:10.)
    try:
        def no_rebuild(*args,**kwargs):raise AssertionError('session must use active command terms')
        monkeypatch.setattr(engine,'build_configs',no_rebuild)
        assert set(current.env.command_manager.active_terms)=={'twist'}
        assert all(current.status(owner())['commandSchema'][name]['editable'] is False for name in ('twist','head_pose','body_pose'))
        for commands in ({'head_pose':[0]*4},{'body_pose':[0]*6},{'head_pose':[0]*4,'body_pose':[0]*6}):
            status=current.status(owner());payload=deepcopy(current._payload());frame=current.frame(owner())
            tensors=[current.env.sim.data.qpos,current.env.sim.data.qvel,current.observation['actor'],current.env.command_manager.get_command('twist')]
            before=[value.clone() for value in tensors];counter=current.env.common_step_counter;size=current.bytes
            with pytest.raises(ValueError,match='EVALUATION_COMMAND_UNSUPPORTED'):
                current.control(owner(),1,'COMMAND',commands)
            assert current.status(owner())==status and current._payload()==payload
            assert current.frame(owner())==frame and current.bytes==size and current.env.common_step_counter==counter
            assert all(torch.equal(old,new) for old,new in zip(before,tensors))
        current.control(owner(),1,'PAUSE',{});assert current.tick() is None
        current.control(owner(),2,'RUN',{})
        term=current.env.command_manager.get_term('twist')
        assert type(term).__name__=='GroundPickPhaseCommand'
        before=term.command.clone();current.tick()
        assert not (before==term.command).all()
        assert float(term.command[0,:2].square().sum())==pytest.approx(1,abs=1e-5)
        with pytest.raises(ValueError,match='EVALUATION_AUTONOMOUS_COMMAND'):current.control(owner(),3,'COMMAND',{'twist':[0,0,0]})
        assert not current.closed and current.failure is None and current.sequence==2
        current.control(owner(),3,'RESET',{});assert current.tick()['step']==2
    finally:current.close()


def test_replay_quota_includes_large_geometry_controls_and_resets(monkeypatch,tmp_path):
    import base64
    import torch
    from mjlab_microduck.workbench import evaluation
    original=EvaluationSession._make_geometry
    def large_geometry(current):
        geometry=original(current)
        mesh=geometry['meshes'][0];vertices=base64.b64decode(mesh['verticesBase64'])
        # Repeated unused vertices keep the real mesh indices valid and stay
        # within the viewer's 8MiB mesh budget, above the old 2MiB reserve.
        repeat=(3*1024*1024*3//4)//len(vertices)+1
        mesh['verticesBase64']=base64.b64encode(vertices*repeat).decode()
        return geometry
    monkeypatch.setattr(EvaluationSession,'_make_geometry',large_geometry)
    current=EvaluationSession(spec(),owner(),clock=lambda:10.)
    try:
        baseline=tmp_path/'baseline.json';current.save_replay(owner(),baseline)
        monkeypatch.setattr(evaluation,'_MAX_REPLAY_BYTES',baseline.stat().st_size+2*1024*1024)
        # Physics is covered by other real-step cases. This test fills history
        # without waiting for hundreds of physical intervals.
        def stationary_step(action):
            current.env.common_step_counter+=1
            return current.observation,torch.tensor([0.]),torch.tensor([False]),torch.tensor([False]),{}
        monkeypatch.setattr(current.env,'step',stationary_step)
        sequence=0
        for index in range(2000):
            if current.tick() is None or current.status(owner())['paused']:break
            if index%20==0:
                sequence+=1;current.control(owner(),sequence,'COMMAND',{'twist':[1e-323,0,0]})
                sequence+=1;current.control(owner(),sequence,'RESET',{})
        assert current.status(owner())['paused']
        output=tmp_path/'bounded.json';current.save_replay(owner(),output)
        assert output.stat().st_size<=evaluation._MAX_REPLAY_BYTES
        saved=json.loads(output.read_text())
        assert len(saved['initialStates'])>1 and len(saved['controls'])>1 and saved['steps']
    finally:current.close()


def test_nonfinite_preview_input_fails_before_physics_and_cannot_continue(session,tmp_path):
    import torch
    before=session.env.common_step_counter
    session.observation['actor']=torch.full((1,61),float('nan'))
    with pytest.raises(ValueError):session.tick()
    assert session.env.common_step_counter==before
    assert session.closed and session.tick() is None
    output=tmp_path/'failed.json';session.save_replay(owner(),output)
    saved=json.loads(output.read_text())
    assert saved['executionState']=='FAILED' and saved['historyComplete'] is False
    assert saved['failure']['physicalStepMayHaveAdvanced'] is False


@pytest.mark.parametrize('fault',['reward','frame'])
def test_failing_interval_latches_terminal_failure_and_marks_incomplete_history(session,monkeypatch,tmp_path,fault):
    import torch
    original=session.env.step;before=session.env.common_step_counter
    if fault=='reward':
        def bad_step(action):
            result=list(original(action));result[1]=torch.tensor([float('nan')]);return tuple(result)
        monkeypatch.setattr(session.env,'step',bad_step)
    else:
        monkeypatch.setattr(session,'_sample_frame',lambda:(_ for _ in ()).throw(ValueError('frame')))
    with pytest.raises(ValueError):session.tick()
    assert session.env.common_step_counter==before+1
    assert session.closed and session.tick() is None
    with pytest.raises(ValueError):session.control(owner(),1,'RUN',{})
    output=tmp_path/'failed.json';session.save_replay(owner(),output)
    saved=json.loads(output.read_text())
    assert saved['executionState']=='FAILED' and saved['historyComplete'] is False
    assert saved['failure']['atStep']==1 and saved['failure']['physicalStepMayHaveAdvanced'] is True
    assert saved['failure']['lastVerifiedStep']==0


def test_native_reset_failure_closes_session_instead_of_allowing_unrecorded_state(session,monkeypatch,tmp_path):
    monkeypatch.setattr(session.env,'reset',lambda **kwargs:(_ for _ in ()).throw(ValueError('native reset')))
    with pytest.raises(ValueError):session.control(owner(),1,'RESET',{})
    assert session.closed and session.tick() is None
    output=tmp_path/'failed-control.json';session.save_replay(owner(),output)
    assert json.loads(output.read_text())['historyComplete'] is False
