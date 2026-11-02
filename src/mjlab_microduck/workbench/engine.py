"""Independent ROM adapter over the installed upstream Apache task factories."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib
from .configuration import verify_experiment
from .parameters import PPO, GAINS


def checked(condition):
    if not condition:raise ValueError('MANAGED_EXPERIMENT_CONFIG_INVALID')


def edit_numeric(original, override):
    """Only numeric/bool leaves of existing factory data; never object/callable selectors."""
    if type(override) is dict:
        checked(type(original) is dict and not override.keys()-original.keys())
        result=deepcopy(original)
        for key,value in override.items():result[key]=edit_numeric(original[key],value)
        return result
    if type(override) is list:
        checked(type(original) in (list,tuple) and len(original)==len(override))
        result=[edit_numeric(a,b) for a,b in zip(original,override)]
        return tuple(result) if type(original) is tuple else result
    if type(original) is bool:checked(type(override) is bool)
    elif type(original) is int:checked(type(override) is int)
    else:checked(type(original) in (int,float) and type(override) in (int,float))
    return override


def scale_stages(value,factor):
    if isinstance(value,dict):
        for key,item in value.items():
            if key=='step' and type(item) is int:value[key]=round(item*factor)
            else:scale_stages(item,factor)
    elif isinstance(value,(list,tuple)):
        for item in value:scale_stages(item,factor)


def build_configs(spec):
    """Copy actual configs, apply typed values, leave the task registry immutable.

    Curriculum boundaries count aggregate samples at upstream's 4096 env training
    reference. numStepsPerEnv changes rollout batching, not sample boundaries.
    """
    spec=verify_experiment(spec);values=spec['effectiveConfig']
    import mjlab_microduck.tasks
    from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
    env=load_env_cfg(spec['taskId']);agent=load_rl_cfg(spec['taskId'])
    if spec['factoryOverrides']:
        from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import make_microduck_ball_kick_env_cfg
        from mjlab_microduck.tasks.backlash import make_backlash_variant
        env=make_microduck_ball_kick_env_cfg(kick_foot='left')
        if spec['backlash']:env=make_backlash_variant(env)
    # Factories can retain shared robot constants; detach before adapting profile.
    env=deepcopy(env)
    env.scene.num_envs=values['numEnvs'];env.seed=values['seed'];env.episode_length_s=values['episodeLengthSeconds']
    agent.seed=values['seed'];agent.max_iterations=values['iterations'];agent.num_steps_per_env=values['numStepsPerEnv'];agent.algorithm.learning_rate=values['learningRate']
    checked(len(env.actions)==1)
    next(iter(env.actions.values())).scale=values['actionScale']
    for key,value in values.get('ppo',{}).items():setattr(agent.algorithm,PPO[key][0],value)
    network=values.get('network',{})
    for role in ('actor','critic'):
        model=getattr(agent,role)
        if role+'HiddenDims' in network:model.hidden_dims=tuple(network[role+'HiddenDims'])
        if 'activation' in network:model.activation=network['activation']
    if 'initialStd' in network:agent.actor.distribution_cfg['init_std']=network['initialStd']
    for group in ('rewards','events','curriculum'):
        target=getattr(env,group)
        for name,changes in values.get(group,{}).items():
            checked(name in target and changes.keys() <= ({'weight','params'} if group=='rewards' else {'params'}))
            term=target[name]
            if 'weight' in changes:term.weight=changes['weight']
            if 'params' in changes:term.params=edit_numeric(term.params,changes['params'])
    for group,changes in values.get('observations',{}).items():
        checked(group in env.observations)
        for name,fields in changes.items():
            checked(name in env.observations[group].terms and fields.keys()<={'params','noise','scale','clip','delay_min_lag','delay_max_lag'})
            term=env.observations[group].terms[name]
            for key,value in fields.items():
                if key=='scale':
                    checked(type(value) in (int,float) and abs(value)<=100);term.scale=value
                elif key=='clip':
                    checked(type(value) is list and len(value)==2 and value[0]<value[1]);term.clip=tuple(value)
                elif key=='noise':
                    checked(term.noise is not None and value.keys()<={'n_min','n_max'})
                    for field,n in value.items():setattr(term.noise,field,edit_numeric(getattr(term.noise,field),n))
                    checked(term.noise.n_min<=term.noise.n_max)
                else:setattr(term,key,edit_numeric(getattr(term,key),value))
            checked(0<=term.delay_min_lag<=term.delay_max_lag<=20)
    for name,changes in values.get('commands',{}).items():
        checked(name in env.commands and changes.keys()<={'ranges','period','resampling_time_range'})
        command=env.commands[name]
        for key,value in changes.items():
            old=getattr(command,key,None)
            if key=='ranges':
                if type(old) in (tuple,list):
                    command.ranges=edit_numeric(old,value);continue
                checked(old is not None and type(value) is dict)
                for field,n in value.items():
                    checked(not field.startswith('_') and hasattr(old,field))
                    setattr(old,field,edit_numeric(getattr(old,field),n))
            else:setattr(command,key,edit_numeric(old,value))
    physics=values.get('physics',{});checked(physics.keys()<={'gravity','solverIterations','bodies','joints','contacts','actuatorParameters'})
    if 'gravity' in physics:
        checked(len(physics['gravity'])==3 and all(abs(x)<=20 for x in physics['gravity']))
        env.sim.mujoco.gravity=tuple(physics['gravity'])
    if 'solverIterations' in physics:
        checked(type(physics['solverIterations']) is int and 1<=physics['solverIterations']<=100)
        env.sim.mujoco.iterations=physics['solverIterations']
    robot=env.scene.entities['robot'];servo=values['servo']
    # BAM rewrites these controlled-joint fields at construction/runtime. Model
    # parameters, not raw joint attributes, own motor inertia and friction.
    for fields in physics.get('joints',{}).values():
        if fields.keys() & {'damping','armature','frictionloss'}:
            raise ValueError('MANAGED_BAM_JOINT_PARAMETER_UNSUPPORTED')
    if any(k in physics for k in ('bodies','joints','contacts')):
        original_spec=robot.spec_fn
        def physical_spec():
            result=original_spec()
            for group,finder,allowed in [('bodies',result.body,{'mass','ipos','inertia'}),('joints',result.joint,{'range','stiffness'}),('contacts',result.geom,{'friction','solref','solimp','margin','gap'})]:
                for name,fields in physics.get(group,{}).items():
                    checked(fields.keys()<=allowed and not (group=='joints' and name.startswith('passive_')))
                    target=finder(name);checked(target is not None)
                    for key,value in fields.items():
                        old=getattr(target,key)
                        if group=='joints' and key=='stiffness' and type(value) in (int,float):value=[value,0,0]
                        if hasattr(old,'shape'):
                            checked(type(value) is list and len(value)==len(old));setattr(target,key,value)
                        else:
                            checked(type(value) in (int,float));setattr(target,key,value)
                    if group=='bodies':
                        checked(target.mass>0)
                        if 'inertia' in fields:
                            checked(all(x>0 for x in target.inertia));target.fullinertia=[0,0,0,0,0,0]
                    if group=='joints' and 'range' in fields:checked(target.range[0]<target.range[1])
                    if group=='contacts' and 'friction' in fields:checked(all(x>=0 for x in target.friction))
            return result
        robot.spec_fn=physical_spec
        # Existing contact editors apply after spec construction. Mirror explicit
        # friction settings into a final editor so they cannot silently overwrite it.
        from mjlab.utils.spec_config import CollisionCfg
        import re
        for name,fields in physics.get('contacts',{}).items():
            robot.collisions=(*robot.collisions,CollisionCfg(geom_names_expr=[re.escape(name)],disable_other_geoms=False,**{key:tuple(v) if type(v) is list else v for key,v in fields.items()}))
    actuators=[]
    for actuator in robot.articulation.actuators:
        settings=dict(kp_fw=servo['kpFirmware'],vin_range=tuple(servo['voltageRange']),vin_drop_gain_range=tuple(servo['voltageDropGainRange']),vin_min=servo['voltageMinimum'],delay_min_lag=servo['delayMinLag'],delay_max_lag=servo['delayMaxLag'])
        if spec['robotProfileId']=='hd1910-reference-v1':settings.update(motor_name=None,model=None,json_path=str(Path(__file__).parents[1]/'robot/hd1910/hd1910_m6.json'))
        settings['parameter_overrides']=physics.get('actuatorParameters',{})
        settings.update({GAINS[k][0]:v for k,v in values.get('gains',{}).items()})
        actuator=replace(actuator,**settings)
        checked(hashlib.sha256(Path(actuator._resolved_json_path).read_bytes()).hexdigest()==servo['parameterDigest'][7:])
        checked(actuator.delay_min_lag<=actuator.delay_max_lag)
        actuators.append(actuator)
    robot.articulation=replace(robot.articulation,actuators=tuple(actuators))
    calibration=values.get('calibration',{});checked(calibration.keys()<={'jointZeroOffsets'})
    offsets=calibration.get('jointZeroOffsets',{})
    if offsets:
        import re
        model=robot.spec_fn().compile()
        for name,offset in offsets.items():
            joint=model.joint(name)
            checked(not name.startswith('passive_') and abs(offset)<=.2)
            defaults=robot.init_state.joint_pos
            initial=next((v for pattern,v in defaults.items() if re.fullmatch(pattern,name)),0)
            defaults={f'^{re.escape(name)}$':initial+offset,**defaults}
            robot.init_state=replace(robot.init_state,joint_pos=defaults)
    for name,term in list(env.curriculum.items()):
        if name in values.get('curriculum',{}):continue
        params=term.params
        reward=params.get('reward_name');event=params.get('event_name');command=params.get('command_name')
        if (reward in values.get('rewards',{}) and 'weight' in values['rewards'][reward]) or event in values.get('events',{}) or command in values.get('commands',{}):del env.curriculum[name]
    for term in env.curriculum.values():scale_stages(term.params,4096/values['numEnvs'])
    checked(abs(env.decimation*env.sim.mujoco.timestep-.02)<1e-8)
    checked(agent.actor.obs_normalization and agent.critic.obs_normalization)
    return env,agent
