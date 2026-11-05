"""Actual task-factory CPU evaluation, isolated from the training process.

This kernel has no network, scheduling, artifact lookup or launch authority. Its
managed caller must revalidate the enrolled scope and operator on every command.
"""
from copy import deepcopy
import math
import json
import time
from .configuration import verify_experiment,canonical_json

_AUTONOMOUS={'phase-clock','autonomous-maneuver','passive-balance'}
_MAX_STEPS=6000
_MAX_CONTROLS=1024
_MAX_REPLAY_BYTES=64*1024*1024
_STEP_RESERVE=1024*1024
_FAILURE_RESERVE=1024


def require(condition,code='EVALUATION_REQUEST_INVALID'):
    if not condition:raise ValueError(code)


def vector(value,size,limit):
    require(type(value) is list and len(value)==size)
    require(all(type(v) in (int,float) and math.isfinite(v) and abs(v)<=limit for v in value))
    return list(value)


def command_schema(spec):
    from .engine import build_configs
    spec=verify_experiment(spec)
    cfg,_=build_configs(spec)
    return _command_schema(spec,{name for name,term in cfg.commands.items() if term is not None})


def _command_schema(spec,terms):
    profile=spec['commandProfile']
    return dict(twistSemantics=profile['twistSemantics'],twist=dict(dimensions=3,editable='twist' in terms and profile['twistSemantics'] not in _AUTONOMOUS,ranges=deepcopy(profile['twistRanges'])),head_pose=dict(dimensions=4,editable='head_pose' in terms,ranges=deepcopy(profile['headPoseRanges'])),body_pose=dict(dimensions=6,editable='body_pose' in terms,ranges=deepcopy(profile['bodyPoseRanges'])))


def validate_commands(spec,commands):
    return _validate_commands(commands,command_schema(spec))


def _validate_commands(commands,schema):
    require(type(commands) is dict and not commands.keys()-{'twist','head_pose','body_pose'})
    result={}
    for name,value in commands.items():
        field=schema[name]
        require(field['editable'],'EVALUATION_AUTONOMOUS_COMMAND' if name=='twist' and schema['twistSemantics'] in _AUTONOMOUS else 'EVALUATION_COMMAND_UNSUPPORTED')
        value=vector(value,field['dimensions'],1000)
        require(all(lo<=v<=hi for v,(lo,hi) in zip(value,field['ranges'])),'EVALUATION_COMMAND_OUTSIDE_ENVELOPE')
        if name=='twist' and schema['twistSemantics']=='sit-stand-vx-flag':require(value[0] in (0,1) and value[1:]==[0,0])
        result[name]=value
    return result


class EvaluationSession:
    """One actual environment, explicit owner, bounded lease/history and controls.

    Each tick advances exactly one 50Hz control interval. Reads never step. The
    owning managed process drives ticks; this object starts no execution thread.
    """
    def __init__(self,spec,identity,*,policy=None,clock=time.monotonic):
        import re
        require(type(identity) is dict and set(identity)=={'sessionId','operatorDigest','scopeDigest','modelDigest'})
        require(type(identity['sessionId']) is str and bool(re.fullmatch('[-A-Za-z0-9_]{1,80}',identity['sessionId'])))
        for key in ('operatorDigest','scopeDigest','modelDigest'):require(type(identity[key]) is str and bool(re.fullmatch('sha256:[a-f0-9]{64}',identity[key])))
        self.spec=verify_experiment(spec);self.identity=deepcopy(identity);self.clock=clock
        self.created=clock();self.expires=self.created+90;self.closed=False;self.paused=False;self.ended=False
        self.sequence=0;self.episode=0;self.episode_step=0;self.total_steps=0;self.steps=[];self.controls=[];self.initial_states=[];self.frame_sequence=0;self.bytes=0;self.next_frame=0
        self.commands={};self.policy=policy;self.failure=None;self.limit_reached=False
        if policy is not None:
            from .playback import PlaybackPolicy
            require(isinstance(policy,PlaybackPolicy),'EVALUATION_POLICY_INVALID')
            require(policy.config_digest==self.spec['configDigest'],'EVALUATION_POLICY_CONFIG_MISMATCH')
        import torch,warp as wp
        from mjlab_microduck.managed_training import configure_warp_runtime
        from .engine import build_configs
        from mjlab.envs import ManagerBasedRlEnv
        configure_warp_runtime(wp,'0.0')
        cfg,self.agent=build_configs(self.spec)
        cfg.scene.num_envs=1;cfg.auto_reset=False
        self.env=ManagerBasedRlEnv(cfg=cfg,device='cpu')
        try:
            require(abs(self.env.step_dt-.02)<1e-8,'EVALUATION_CONTROL_RATE_INVALID')
            self._command_schema=_command_schema(self.spec,self.env.command_manager.active_terms)
            self._hold_commands()
            self._geometry=self._make_geometry()
            self.bytes=self._size(self._payload())+_FAILURE_RESERVE
            require(self._room(_STEP_RESERVE),'EVALUATION_REPLAY_TOO_LARGE')
            self._reset()
        except Exception:
            self.env.close();self.closed=True;raise

    def _authorize(self,identity,*,allow_closed=False):
        require(identity==self.identity,'EVALUATION_OWNER_MISMATCH')
        if not self.closed and self.clock()>=self.expires:self.close()
        require(allow_closed or not self.closed,'EVALUATION_SESSION_CLOSED')

    def _hold_commands(self):
        # Keep task compute logic (including sit/stand slew), but prevent random
        # sampling from replacing an explicitly held operator command.
        for name in ('twist','head_pose','body_pose'):
            if name not in self.env.command_manager.active_terms:continue
            term=self.env.command_manager.get_term(name);original=term.compute
            def compute(dt,*,name=name,term=term,original=original):
                if name in self.commands:
                    term.time_left.fill_(1e9)
                    term.command[:]=term.command.new_tensor(self.commands[name])
                original(dt)
                if name in self.commands:term.command[:]=term.command.new_tensor(self.commands[name])
            term.compute=compute

    def _refresh_commands(self):
        for name,values in self.commands.items():
            term=self.env.command_manager.get_term(name)
            term.command[:]=term.command.new_tensor(values)
        self.env.obs_buf=self.env.observation_manager.compute(update_history=False)
        self.observation=self.env.obs_buf

    def _reset(self):
        # Every reset uses a declared deterministic seed; no global state is shared
        # with PPO because the managed caller runs this kernel in a separate process.
        self.observation,_=self.env.reset(seed=self.spec['effectiveConfig']['seed']+self.episode)
        self.episode_step=0;self.ended=False;self._refresh_commands()
        self._append(self.initial_states,dict(episode=self.episode,seed=self.spec['effectiveConfig']['seed']+self.episode,qpos=self.env.sim.data.qpos[0].tolist(),qvel=self.env.sim.data.qvel[0].tolist()))
        self._frame=self._sample_frame()

    def _make_geometry(self):
        from mjlab_microduck.rom.viewer import export_geometry,export_meshes
        model=self.env.sim.mj_model
        require(model.ngeom<=256 and model.nmeshvert<=1000000 and model.nmeshface<=2000000,'EVALUATION_MODEL_TOO_LARGE')
        geoms=export_geometry(model)
        value=dict(schemaVersion='ROM_EVALUATION_GEOMETRY_V1',**self.identity,configDigest=self.spec['configDigest'],geometry=geoms,meshes=export_meshes(model,geoms))
        require(len(canonical_json(value).encode())<=64*1024*1024,'EVALUATION_MODEL_TOO_LARGE')
        return value

    def _sample_frame(self):
        result=dict(schemaVersion='ROM_EVALUATION_FRAME_V1',**self.identity,configDigest=self.spec['configDigest'],sequence=self.frame_sequence,episode=self.episode,episodeStep=self.episode_step,simulationSeconds=self.total_steps*.02,positions=self.env.sim.data.geom_xpos[0].flatten().tolist(),matrices=self.env.sim.data.geom_xmat[0].flatten().tolist())
        require(len(canonical_json(result).encode())<=262144,'EVALUATION_FRAME_TOO_LARGE')
        self.frame_sequence+=1
        return result

    def status(self,identity):
        self._authorize(identity,allow_closed=True)
        return dict(**self.identity,configDigest=self.spec['configDigest'],mode='INITIAL_MODEL' if self.policy is None else self.policy.kind,closed=self.closed,paused=self.paused,episodeEnded=self.ended,lastControlSequence=self.sequence,episode=self.episode,episodeStep=self.episode_step,totalSteps=self.total_steps,commands=deepcopy(self.commands),commandSchema=deepcopy(self._command_schema),remainingLeaseSeconds=max(0,self.expires-self.clock()),evaluation='NOT_QUALIFIED',executionState='FAILED' if self.failure else 'CLOSED' if self.closed else 'REPLAY_LIMIT' if self.limit_reached else 'OPEN',replayLimitReached=self.limit_reached,failure=deepcopy(self.failure))

    def geometry(self,identity):
        self._authorize(identity);return deepcopy(self._geometry)

    def frame(self,identity):
        self._authorize(identity);return deepcopy(self._frame)

    def _control_parameters(self,action,parameters):
        require(type(parameters) is dict and type(action) is str)
        if action=='COMMAND':return _validate_commands(parameters,self._command_schema)
        if action in ('PAUSE','RUN','RESET'):
            require(not parameters);return {}
        if action=='DISTURB':
            require(set(parameters)=={'linearVelocityDelta','angularVelocityDelta'})
            return dict(linearVelocityDelta=vector(parameters['linearVelocityDelta'],3,2),angularVelocityDelta=vector(parameters['angularVelocityDelta'],3,5))
        if action=='RENEW':
            require(set(parameters)=={'seconds'} and type(parameters['seconds']) in (int,float) and math.isfinite(parameters['seconds']) and 1<=parameters['seconds']<=90)
            return deepcopy(parameters)
        raise ValueError('EVALUATION_ACTION_UNSUPPORTED')

    def control(self,identity,sequence,action,parameters):
        self._authorize(identity)
        require(type(sequence) is int and sequence==self.sequence+1,'EVALUATION_SEQUENCE_CONFLICT')
        require(len(self.controls)<_MAX_CONTROLS,'EVALUATION_CONTROL_LIMIT')
        parameters=self._control_parameters(action,parameters)
        record=dict(sequence=sequence,action=action,parameters=deepcopy(parameters),atStep=self.total_steps,episode=self.episode+(action=='RESET'))
        reserve=_STEP_RESERVE if action=='RESET' else 0
        if not self._room(self._size(record)+2+reserve):
            self.paused=True;self.limit_reached=True
            raise ValueError('EVALUATION_REPLAY_LIMIT')
        try:
            if action=='COMMAND':self.commands.update(parameters);self._refresh_commands()
            elif action=='PAUSE':self.paused=True
            elif action=='RUN':self.paused=False
            elif action=='RESET':self.episode+=1;self._reset()
            elif action=='DISTURB':
                robot=self.env.scene['robot'];state=robot.data.root_link_vel_w.clone()
                state[:,:3]+=state.new_tensor(parameters['linearVelocityDelta']);state[:,3:]+=state.new_tensor(parameters['angularVelocityDelta'])
                robot.write_root_link_velocity_to_sim(state);self.env.sim.forward();self.env.scene.update(dt=0);self._refresh_commands()
                self._frame=self._sample_frame()
            else:self.expires=min(self.created+600,self.clock()+parameters['seconds'])
            self._append(self.controls,record);self.sequence=sequence
        except Exception:
            self._fail('CONTROL_APPLY',self.total_steps,True)
            raise ValueError('EVALUATION_CONTROL_FAILED') from None
        return self.status(identity)

    def tick(self):
        if self.closed:return None
        if self.clock()>=self.expires:self.close();return None
        if self.paused or self.ended:return None
        if self.total_steps>=_MAX_STEPS or not self._room(_STEP_RESERVE):
            self.paused=True;self.limit_reached=True;return None
        import torch
        attempted=self.total_steps+1;phase='PRE_STEP';advanced=False
        try:
            raw=self.observation['actor'].detach().clone()
            require(tuple(raw.shape)==(1,61) and bool(torch.isfinite(raw).all()),'EVALUATION_OBSERVATION_INVALID')
            for value in (self.env.sim.data.qpos,self.env.sim.data.qvel):
                require(value.ndim==2 and value.shape[0]==1 and value.numel()<=4096 and bool(torch.isfinite(value).all()),'EVALUATION_STATE_INVALID')
            with torch.inference_mode():
                action=torch.zeros((1,14)) if self.policy is None else self.policy.predict(raw)
            require(tuple(action.shape)==(1,14) and bool(torch.isfinite(action).all()),'EVALUATION_POLICY_OUTPUT_INVALID')
            if self.agent.clip_actions is not None:action=action.clamp(-self.agent.clip_actions,self.agent.clip_actions)
            record=dict(step=attempted,episode=self.episode,observation=raw[0].tolist(),action=action[0].tolist(),qpos=self.env.sim.data.qpos[0].tolist(),qvel=self.env.sim.data.qvel[0].tolist(),commands={name:self.env.command_manager.get_command(name)[0].tolist() for name in self.env.command_manager.active_terms})
            self._size(record)
            phase='PHYSICS_STEP';advanced=True
            self.observation,reward,terminated,truncated,extras=self.env.step(action)
            self.total_steps+=1;self.episode_step+=1;self.ended=bool(terminated[0] or truncated[0])
            phase='EVIDENCE_VALIDATION'
            record.update(reward=float(reward[0]),terminated=bool(terminated[0]),truncated=bool(truncated[0]),simulationSeconds=self.total_steps*.02,rewardTerms={name:values for name,values in self.env.reward_manager.get_active_iterable_terms(0)})
            require(math.isfinite(record['reward']),'EVALUATION_NONFINITE_RESULT')
            if self.total_steps*.02>=self.next_frame:
                self._frame=self._sample_frame();record['frame']=deepcopy(self._frame);self.next_frame=self.total_steps*.02+.05
            self._append(self.steps,record)
            if not self._room(_STEP_RESERVE):self.paused=True;self.limit_reached=True
            return deepcopy(record)
        except Exception:
            self._fail(phase,attempted,advanced)
            raise ValueError('EVALUATION_STEP_FAILED') from None

    @staticmethod
    def _size(value):
        # Match atomic_json's actual encoding, including separators/ASCII escapes.
        return len(json.dumps(value,sort_keys=True,allow_nan=False).encode('utf-8'))

    def _room(self,cost):return self.bytes+cost<=_MAX_REPLAY_BYTES

    def _append(self,records,value):
        cost=self._size(value)+2
        require(self._room(cost),'EVALUATION_REPLAY_LIMIT')
        records.append(value);self.bytes+=cost

    def _fail(self,phase,at_step,advanced):
        self.failure=dict(phase=phase,atStep=at_step,physicalStepMayHaveAdvanced=advanced,lastVerifiedStep=self.steps[-1]['step'] if self.steps else 0)
        self.paused=True
        try:self.close()
        except Exception:pass  # Failure is latched even if native cleanup fails.

    def _payload(self):
        policy=dict(kind='INITIAL_MODEL') if self.policy is None else dict(kind=self.policy.kind,sha256=self.policy.sha256)
        return dict(schemaVersion='ROM_EVALUATION_REPLAY_V1',identity=self.identity,experiment=self.spec,policy=policy,evaluation='NOT_QUALIFIED',executionState='FAILED' if self.failure else 'CLOSED' if self.closed else 'REPLAY_LIMIT' if self.limit_reached else 'OPEN',historyComplete=self.failure is None,failure=self.failure,controlFrequencyHz=50,initialStates=self.initial_states,controls=self.controls,steps=self.steps,geometry=self._geometry)

    def save_replay(self,identity,path):
        self._authorize(identity,allow_closed=True)
        from mjlab_microduck.managed_training import atomic_json
        value=self._payload()
        require(self._size(value)<=_MAX_REPLAY_BYTES,'EVALUATION_REPLAY_TOO_LARGE')
        atomic_json(path,value)

    def close(self):
        if not self.closed:
            self.closed=True
            self.env.close()
