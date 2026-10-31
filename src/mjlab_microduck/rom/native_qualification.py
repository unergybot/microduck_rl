"""Immutable native smoke profile. Transport is an Agent port, never a robot socket.

CLI: ``python -m mjlab_microduck.rom.native_qualification report.json`` evaluates
collected native evidence. Missing evidence is INCONCLUSIVE, never inferred from ACKs.
"""
from __future__ import annotations
import hashlib
import json
import math
import sys
import time
from typing import Protocol
import copy

PROFILE = {'id':'microduck-walk-flat-native-v1','seeds':[7,11,29],
           'scenarios':{'stand':[0.,0.,0.],'forward':[.05,0.,0.],'turn':[0.,0.,.2]},
           'settleSeconds':3.,'motionSeconds':10.,'stopSeconds':2.,
           'startupWallSeconds':60.,'episodeWallSeconds':120.,'fallDegrees':60.,
           'fallSeconds':.2,'fallHeightRatio':.5,'jointLimitTolerance':.001,
           'stopLinear':.02,'stopYaw':.1,'resetGenerator':'PCG64',
           'jointPerturbation':.005,'yawPerturbation':.01,'sampleSeconds':.005,
           'footGeoms':['left_foot_collision','right_foot_collision']}

def digest(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

PROFILE_DIGEST = digest(PROFILE)
IDENTITY_FIELDS = ('releaseId','simulatorImage','adapterImage','modelDigest','defaultPolicyDigest','profileDigest','runtimeRevision','candidateSha256')
FAULTS = ('connection_loss','duplicate','restart','cancel')
# Frozen reference vectors from NumPy Generator(PCG64(seed)); body generates independently.
RESET_VECTORS = {7: {'jointPerturbations': [0.0012509546660466695, 0.0039721380096957555, 0.0027568569024519355, -0.0027479281000940816, -0.0019983371508877456, 0.003735534453962619, -0.004947346954344253, 0.0032122841838276633, 0.0029706942875204626, -0.00032065047156279187, -0.001969675731806865, -0.0022157438789922666, -0.002451304123458754, -0.0005492369411735339], 'yawPerturbation': 9.09651791590662e-05}, 11: {'jointPerturbations': [-0.003714297972308004, -7.221375598850596e-06, 0.0010149835762335744, -0.004713109916280555, -0.0035207391542254407, 0.004282110229603695, -0.0042957942384580314, -0.0037022605060070203, 0.004483284532917751, 0.0012188359279638288, -0.00131006876270209, 0.00011390021803262679, 0.0016284295251679922, -0.002246911842388707], 'yawPerturbation': -0.007240638542660893}, 29: {'jointPerturbations': [-0.004499530296459311, 6.322298571449433e-05, 0.00019234029629711536, -0.00234796786062789, -0.003707783551481659, -0.004792689674231341, -0.0010617203907634177, -0.0011977139812342041, -0.004765480466599985, -0.0026178826135474886, 0.002881063937984786, 0.001175953818296679, 0.004827933264471253, 0.003611118791447844], 'yawPerturbation': 0.0026285839266456112}}


def scenarios():
    return [{'scenario':name,'seed':seed,'twist':twist.copy()} for seed in PROFILE['seeds'] for name,twist in PROFILE['scenarios'].items()]


def perturbations(seed):
    if seed not in PROFILE['seeds']: raise ValueError('PROFILE_SEED_MISMATCH')
    return copy.deepcopy(RESET_VECTORS[seed])


def _vector(value, size):
    if not isinstance(value,list) or len(value)!=size: raise ValueError('PHYSICAL_EVIDENCE_SHAPE')
    if any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) for x in value): raise ValueError('PHYSICAL_EVIDENCE_NONFINITE')
    return value


def _mean(rows, value, start, end):
    area=0.
    for left,right in zip(rows,rows[1:]):
        dt=max(0.,min(end,right['simTime'])-max(start,left['simTime']))
        area+=dt*value(left)
    return area/(end-start)


def evaluate_episode(episode):
    result={'scenario':episode.get('scenario'),'seed':episode.get('seed'),'evaluation':'INCONCLUSIVE','reasonCodes':[], 'metrics':{}}
    reasons=result['reasonCodes']; metrics=result['metrics']
    try:
        if episode.get('completed') is not True: raise ValueError('EPISODE_INCOMPLETE')
        rows=episode['samples']; reset=episode['reset']; commands=episode['commands']
        _vector(reset['jointPerturbations'],14)
        if not reset['qpos'] or not reset['qvel'] or any(not math.isfinite(float(x)) for x in reset['qpos']+reset['qvel']+[reset['yawPerturbation']]): raise ValueError('RESET_EVIDENCE_INVALID')
        coverage=episode['contactCoverage']
        if coverage['configured'] is not True or coverage['footGeoms']!=PROFILE['footGeoms'] or not isinstance(coverage['disabledGeoms'],list): raise ValueError('CONTACT_COVERAGE_MISSING')
        height=episode['standHeight']
        if not math.isfinite(height) or height<=0: raise ValueError('STAND_HEIGHT_INVALID')
        enable,motion,stop=(commands[key] for key in ('enable','motion','stop'))
        if any(not math.isfinite(v) for v in (enable,motion,stop)) or motion-enable<3.-1e-6 or stop-motion<10.-1e-6: raise ValueError('PHASE_TIMING_INVALID')
        end=stop+2.
        if len(rows)<2 or rows[0]['simTime']>enable+.005001 or rows[-1]['simTime']<end-1e-6: raise ValueError('PHYSICAL_EVIDENCE_TRUNCATED')
        fallen_since=None; previous=None
        for row in rows:
            stamp=row['simTime']; wall=row['wallTime']; seq=row['sequence']
            if not math.isfinite(stamp) or not math.isfinite(wall): raise ValueError('TIMESTAMP_INVALID')
            if previous and (seq!=previous['sequence']+1 or not 0.<stamp-previous['simTime']<=.005001 or wall<previous['wallTime']): raise ValueError('PHYSICAL_EVIDENCE_GAP_OR_DUPLICATE')
            previous=row
            for key in ('finiteState','finiteActions'):
                if type(row[key]) is not bool: raise ValueError('FINITE_EVIDENCE_MISSING')
                if row[key] is False: reasons.append('NONFINITE_STATE_OR_ACTION')
            position=_vector(row['trunkPosition'],3); up=_vector(row['trunkUp'],3)
            _vector(row['linearVelocity'],3)
            if not math.isfinite(row['yawRate']): reasons.append('NONFINITE_STATE_OR_ACTION')
            joints=_vector(row['jointPositions'],14)
            ranges=row['jointRanges']
            if len(ranges)!=14: raise ValueError('JOINT_RANGES_MISSING')
            for q,limits in zip(joints,ranges):
                lo,hi=_vector(limits,2)
                if lo>=hi: raise ValueError('JOINT_RANGE_INVALID')
                if q<lo-.001 or q>hi+.001: reasons.append('JOINT_LIMIT')
            if not isinstance(row['forbiddenContacts'],list): raise ValueError('CONTACT_EVIDENCE_MISSING')
            if row['forbiddenContacts']: reasons.append('FORBIDDEN_CONTACT')
            if up[2]<.5 or position[2]<.5*height:
                if fallen_since is None: fallen_since=stamp
                if stamp-fallen_since>=.2-1e-9: reasons.append('FALL')
            else: fallen_since=None
        if rows[-1]['wallTime']-rows[0]['wallTime']>120.: raise ValueError('EPISODE_WALL_TIMEOUT')
        linear=_mean(rows,lambda r:math.hypot(*r['linearVelocity'][:2]),end-1.,end)
        yaw=_mean(rows,lambda r:abs(r['yawRate']),end-1.,end)
        if linear>.02+1e-12 or yaw>.1+1e-12: reasons.append('PHYSICAL_STOP_FAILED')
        twist=PROFILE['scenarios'][episode['scenario']]
        metrics.update(stopMeanLinearVelocity=linear,stopMeanAbsoluteYawRate=yaw,
                       trackingLinearMAE=_mean(rows,lambda r:abs(r['linearVelocity'][0]-twist[0]),motion,stop),
                       trackingYawMAE=_mean(rows,lambda r:abs(r['yawRate']-twist[2]),motion,stop),
                       distance=math.dist(rows[0]['trunkPosition'][:2],rows[-1]['trunkPosition'][:2]),
                       yawDisplacement=_mean(rows,lambda r:r['yawRate'],enable,end)*(end-enable),
                       realtimeFactor=(rows[-1]['simTime']-rows[0]['simTime'])/max(rows[-1]['wallTime']-rows[0]['wallTime'],1e-9),
                       contactCoverage=coverage)
        for name in ('energy','saturation','replay'):
            metrics[name]=episode.get(name,{'available':False,'reason':'NOT_RECORDED'})
        result['evaluation']='FAILED' if reasons else 'PASSED'
    except (KeyError,ValueError,TypeError,IndexError,OverflowError) as error:
        reasons.append(str(error) or 'REQUIRED_EVIDENCE_MISSING')
        if 'NONFINITE_STATE_OR_ACTION' in reasons: result['evaluation']='FAILED'
    result['reasonCodes']=sorted(set(reasons))
    return result


def evaluate_fault(fault):
    kind=fault.get('kind'); reasons=[]
    if fault.get('parentActive') is not True or fault.get('completed') is not True: reasons.append('FAULT_EVIDENCE_INCOMPLETE')
    if fault.get('motionReplayed') is not False: reasons.append('MOTION_REPLAY_NOT_EXCLUDED')
    if kind in ('restart','cancel') and (fault.get('oldAuthorityRejected') is not True or not isinstance(fault.get('newGeneration'),int) or fault['newGeneration']<=fault.get('oldGeneration',0)): reasons.append('GENERATION_NOT_FENCED')
    if kind=='duplicate' and (fault.get('duplicateApplied') is not False or fault.get('watchdogExtended') is not False): reasons.append('DUPLICATE_NOT_FENCED')
    if kind=='connection_loss':
        try:
            delay=fault['zeroRequestedWallTime']-fault['disconnectedWallTime']
            if not math.isfinite(delay) or not 0<=delay<=.5: reasons.append('UPDATE_LOSS_BOUND_FAILED')
            rows=fault['stopSamples']; start=fault['stopSimTime']; end=start+2.
            if len(rows)<2 or rows[0]['simTime']>start+.005001 or rows[-1]['simTime']<end: raise ValueError('STOP_EVIDENCE_TRUNCATED')
            for a,b in zip(rows,rows[1:]):
                if b['sequence']!=a['sequence']+1 or not 0<b['simTime']-a['simTime']<=.005001: raise ValueError('STOP_EVIDENCE_GAP')
                _vector(a['linearVelocity'],3)
                if not math.isfinite(a['yawRate']): raise ValueError('STOP_NONFINITE')
            if _mean(rows,lambda r:math.hypot(*r['linearVelocity'][:2]),end-1,end)>.02 or _mean(rows,lambda r:abs(r['yawRate']),end-1,end)>.1: reasons.append('PHYSICAL_STOP_FAILED')
        except (KeyError,ValueError,TypeError): reasons.append('FAULT_STOP_EVIDENCE_MISSING')
    return {'kind':kind,'evaluation':'PASSED' if not reasons else 'INCONCLUSIVE','reasonCodes':reasons}


def evaluate_report(report):
    result={'evaluation':'INCONCLUSIVE','reasonCodes':[],'episodes':[],'faults':[],'profileId':PROFILE['id'],'profileDigest':PROFILE_DIGEST}
    if report.get('runOutcome')!='COMPLETED': result['reasonCodes'].append('PARENT_NOT_COMPLETED'); return result
    if report.get('runtimeKind')!='NATIVE_ROBOTD': result['reasonCodes'].append('NATIVE_RUNTIME_REQUIRED'); return result
    try:
        actual=report['identity']; expected=report['expectedIdentity']
        if any(not actual.get(k) or actual[k]!=expected[k] for k in IDENTITY_FIELDS) or actual['profileDigest']!=PROFILE_DIGEST: raise ValueError('IDENTITY_MISMATCH')
        episodes=report['episodes']; faults=report['faults']
        if [(x['scenario'],x['seed']) for x in episodes]!=[(x['scenario'],x['seed']) for x in scenarios()]: raise ValueError('NINE_SEEDED_EPISODES_REQUIRED')
        for item in episodes:
            if item.get('loadedPolicyDigest')!=actual['candidateSha256']:raise ValueError('NATIVE_LOAD_READBACK_REQUIRED')
            reset=perturbations(item['seed'])
            if any(item['reset'][key]!=reset[key] for key in reset): raise ValueError('RESET_VECTOR_MISMATCH')
        if [f['kind'] for f in faults]!=list(FAULTS): raise ValueError('FAULT_EPISODES_REQUIRED')
        result['episodes']=[evaluate_episode(x) for x in episodes]
        result['faults']=[evaluate_fault(x) for x in faults]
        verdicts=[x['evaluation'] for x in result['episodes']+result['faults']]
        result['evaluation']='INCONCLUSIVE' if 'INCONCLUSIVE' in verdicts else 'FAILED' if 'FAILED' in verdicts else 'PASSED'
    except (KeyError,ValueError,TypeError) as error: result['reasonCodes'].append(str(error))
    return result


class AgentPort(Protocol):
    """Implemented by governed Agent transport; no robotd/Adapter socket access here.

    Each open creates a fresh Manager child world and authority. Calls enforce supplied
    wall deadline. fault() injects only child faults while parent_state remains RUNNING.
    """
    def open_episode(self, scenario: dict, deadline: float) -> dict: ...
    def command(self, operation: str, parameters: dict, deadline: float) -> dict: ...
    def evidence(self, after_sequence: int, deadline: float) -> dict: ...
    def fault(self, kind: str, deadline: float) -> dict: ...
    def close_episode(self) -> None: ...
    def parent_state(self) -> str: ...


def run_profile(port: AgentPort, identity: dict, *, clock=time.monotonic, sleep=time.sleep):
    report={'schema':'ROM_NATIVE_QUALIFICATION_V1','runtimeKind':'NATIVE_ROBOTD','runOutcome':'RUNNING','expectedIdentity':identity,'identity':{},'episodes':[],'faults':[]}
    try:
        for scenario in scenarios():
            if port.parent_state()!='RUNNING': raise InterruptedError(port.parent_state())
            startup=clock()+60.; info=port.open_episode(scenario,startup)
            report['identity']=info['identity']
            if any(info['identity'].get(k)!=identity.get(k) for k in IDENTITY_FIELDS): raise ValueError('IDENTITY_MISMATCH')
            reset=port.command('RESET',{'seed':scenario['seed']},startup)
            loaded=port.command('LOAD',{},startup)
            enabled=port.command('ENABLE',{},startup)
            begin=enabled['appliedSimTime']; deadline=clock()+120.
            episode={**scenario,**reset,'loadedPolicyDigest':loaded['loadedPolicyDigest'],'completed':False,'commands':{'enable':begin},'samples':[]}
            seq=0; moving=False; stopped=False; last_update=float('-inf')
            while True:
                if port.parent_state()!='RUNNING': raise InterruptedError(port.parent_state())
                if clock()>=deadline: raise TimeoutError('EPISODE_WALL_TIMEOUT')
                evidence=port.evidence(seq,deadline)
                rows=evidence['samples']; episode['samples'].extend(rows)
                if rows: seq=rows[-1]['sequence']
                sim=rows[-1]['simTime'] if rows else begin
                if not moving and sim-begin>=3.:
                    receipt=port.command('MOVE',dict(zip(('vx','vy','wz'),scenario['twist'])),deadline)
                    episode['commands']['motion']=receipt['appliedSimTime']; moving=True; last_update=clock()
                if moving and not stopped and sim-episode['commands']['motion']>=10.:
                    receipt=port.command('STOP',{},deadline); episode['commands']['stop']=receipt['appliedSimTime']; stopped=True
                elif moving and not stopped and clock()-last_update>=.1:
                    port.command('MOVE',dict(zip(('vx','vy','wz'),scenario['twist'])),deadline); last_update=clock()
                if stopped and sim-episode['commands']['stop']>=2.:
                    episode['completed']=True;break
                sleep(.01)
            # Drop samples preceding enable; body records every physics step.
            episode['samples']=[row for row in episode['samples'] if row['simTime']>=begin-1e-8]
            report['episodes'].append(episode); port.close_episode()
        for kind in FAULTS:
            if port.parent_state()!='RUNNING': raise InterruptedError(port.parent_state())
            report['faults'].append(port.fault(kind,clock()+120.))
        if port.parent_state()!='RUNNING': raise InterruptedError(port.parent_state())
        report['runOutcome']='COMPLETED'
    except InterruptedError as error: report.update(runOutcome='TIMED_OUT' if str(error)=='TIMED_OUT' else 'CANCELLED')
    except TimeoutError: report.update(runOutcome='TIMED_OUT')
    except (OSError,RuntimeError,ValueError,KeyError) as error: report.update(runOutcome='FAILED',runtimeFault=str(error))
    finally: port.close_episode()
    report['result']=evaluate_report(report)
    report['evidenceDigest']=digest(report)
    return report


if __name__=='__main__':
    with open(sys.argv[1]) as stream: evidence=json.load(stream)
    print(json.dumps(evaluate_report(evidence),sort_keys=True,allow_nan=False))
