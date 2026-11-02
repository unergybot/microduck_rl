"""CPU-only real factory/physical model checks. No CUDA simulation or learning."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from mjlab_microduck.workbench import catalog, resolve_experiment
from mjlab_microduck.workbench.engine import build_configs


def spec(recipe='Mjlab-Velocity-Flat-MicroDuck', profile='rom-xl330-v1', **config):
    return resolve_experiment(dict(schemaVersion='ROM_TRAINING_EXPERIMENT_V1',recipeId=recipe,robotProfileId=profile,config=config))


@pytest.mark.parametrize('recipe', [r['recipeId'] for r in catalog()['recipes']])
@pytest.mark.parametrize('profile', ['rom-xl330-v1','hd1910-reference-v1'])
def test_actual_robot_build_and_actuator_physics_all_70(recipe, profile):
    import mujoco
    from bam.model import load_model
    s=spec(recipe,profile)
    env,agent=build_configs(s)
    robot=env.scene.entities['robot'].build()
    model=robot.compile()
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(model.njnt)]
    servo=[n for n in names if n and not n.startswith('passive_') and model.joint(n).type != mujoco.mjtJoint.mjJNT_FREE]
    from mjlab_microduck.rom.contracts import CONTROLLED_SERVO_JOINTS
    assert tuple(servo)==CONTROLLED_SERVO_JOINTS
    assert model.nu==14
    assert all(not model.joint(int(i)).name.startswith('passive_') for i in model.actuator_trnid[:,0])
    actuator=env.scene.entities['robot'].articulation.actuators[0]
    assert hashlib.sha256(Path(actuator._resolved_json_path).read_bytes()).hexdigest()==s['effectiveConfig']['servo']['parameterDigest'][7:]
    physics=load_model(actuator._resolved_json_path)
    assert physics is not None
    assert actuator.kp_fw == (5 if profile.startswith('hd') else 200)
    assert env.decimation*env.sim.mujoco.timestep==pytest.approx(.02)
    if s['backlash']: assert len([n for n in names if n and n.endswith('_backlash')])==14
    if 'Roller' in recipe or 'Swizzle' in recipe or 'Spin' in recipe: assert any(n and 'wheel' in n for n in names)
    if s['actionId']=='kick_left': assert env.events['reset_ball'].params['offset'][1]>0


def test_typed_edits_apply_and_do_not_mutate_factory():
    env,agent=build_configs(spec(ppo={'gamma':.97,'numLearningEpochs':2},network={'actorHiddenDims':[64,32]},rewards={'action_rate_l2':{'weight':-.7}},physics={'gravity':[0,0,-9.7]},gains={'kpFirmware':180},calibration={'jointZeroOffsets':{'left_knee':.01}}))
    assert agent.algorithm.gamma==.97 and agent.algorithm.num_learning_epochs==2
    assert agent.actor.hidden_dims==(64,32)
    assert env.rewards['action_rate_l2'].weight==-.7
    assert env.sim.mujoco.gravity==(0,0,-9.7)
    assert env.scene.entities['robot'].articulation.actuators[0].kp_fw==180
    assert build_configs(spec())[1].algorithm.gamma==.99


def test_untrusted_terms_and_executable_overrides_rejected():
    for changes in ({'rewards':{'missing':{'weight':1}}},{'observations':{'actor':{'joint_pos':{'params':{'asset_cfg':'os.system'}}}}}):
        with pytest.raises(ValueError):build_configs(spec(**changes))


def test_curriculum_is_scaled_by_sample_count():
    default,_=build_configs(spec())
    changed,_=build_configs(spec(numEnvs=2))
    for name,term in default.curriculum.items():
        for key,stages in term.params.items():
            if key.endswith('_stages'):
                assert changed.curriculum[name].params[key][1]['step']==round(stages[1]['step']/2) if len(stages)>1 else True


def test_named_physics_changes_reach_actual_compiled_model():
    env,_=build_configs(spec(physics={'bodies':{'trunk_base':{'mass':.3,'ipos':[.001,.002,.003]}},'joints':{'left_knee':{'range':[-1,1],'stiffness':.02}},'actuatorParameters':{'friction_base':.03}}))
    robot=env.scene.entities['robot'].build();model=robot.compile()
    assert model.body('trunk_base').mass[0]==pytest.approx(.3)
    assert model.body('trunk_base').ipos.tolist()==pytest.approx([.001,.002,.003])
    assert model.joint('left_knee').range.tolist()==pytest.approx([-1,1])
    assert model.jnt_stiffness[model.joint('left_knee').id]==pytest.approx(.02)
    assert robot.actuators[0]._bam_model.friction_base.value==pytest.approx(.03)


def test_left_factory_profile_cannot_leak_into_other_recipes():
    build_configs(spec('Mjlab-BallKickLeft-Flat-Backlash-MicroDuck','hd1910-reference-v1'))
    env,_=build_configs(spec('Mjlab-StandUp-Flat-MicroDuck'))
    assert env.scene.entities['robot'].articulation.actuators[0].motor_name=='xl330'


def test_commands_and_observation_edits_freeze_effective_contract():
    s=spec(commands={'twist':{'ranges':{'lin_vel_x':[-.1,.2]}},'head_pose':{'ranges':[[0,.1]]*4}},observations={'actor':{'joint_vel':{'scale':.5,'noise':{'n_min':-.1,'n_max':.1}}}})
    env,_=build_configs(s)
    assert env.commands['twist'].ranges.lin_vel_x==(-.1,.2)
    assert s['commandProfile']['twistRanges'][0]==[-.1,.2]
    assert s['commandProfile']['headPoseRanges']==[[0,.1]]*4
    assert env.observations['actor'].terms['joint_vel'].scale==.5
    assert env.commands['head_pose'].ranges==((0,.1),)*4


@pytest.mark.parametrize('field,value',[('damping',.02),('armature',.04),('frictionloss',.03)])
def test_named_bam_owned_joint_parameters_are_explicitly_refused(field,value):
    with pytest.raises(ValueError,match='^MANAGED_BAM_JOINT_PARAMETER_UNSUPPORTED$'):
        env,_=build_configs(spec(physics={'joints':{'left_knee':{field:value}}}))
        env.scene.entities['robot'].build()


def test_bam_model_parameters_survive_actual_actuator_construction():
    env,_=build_configs(spec(physics={'actuatorParameters':{'armature':.04,'friction_base':.03,'friction_viscous':.02}}))
    robot=env.scene.entities['robot'].build();model=robot.compile()
    for name in ('left_knee','right_knee'):
        dof=int(model.jnt_dofadr[model.joint(name).id])
        assert model.dof_armature[dof]==pytest.approx(.04)
        # BAM initializes these to zero, then owns them via its runtime budget.
        assert model.dof_damping[dof]==0 and model.dof_frictionloss[dof]==0
    bam=robot.actuators[0]._bam_model
    assert bam.friction_base.value==pytest.approx(.03)
    assert bam.friction_viscous.value==pytest.approx(.02)
    import torch
    zero=torch.zeros(1,14)
    budget=robot.actuators[0]._compute_friction_budget(zero,zero,zero)
    assert torch.allclose(budget,torch.full_like(zero,.03))
