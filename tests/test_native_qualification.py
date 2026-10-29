import copy
import importlib
import math
import pytest


def api():
    spec = importlib.util.find_spec('mjlab_microduck.rom.native_qualification')
    assert spec is not None, 'native qualification evaluator is missing'
    return importlib.import_module(spec.name)


def episode():
    rows=[]
    for i in range(3001):
        rows.append(dict(sequence=i+1,simTime=i*.005,wallTime=i*.005,finiteState=True,finiteActions=True,
                         trunkPosition=[0.,0.,.15],trunkUp=[0.,0.,1.],linearVelocity=[0.,0.,0.],yawRate=0.,
                         jointPositions=[0.]*14,jointRanges=[[-1.,1.]]*14,forbiddenContacts=[]))
    return dict(scenario='stand',seed=7,completed=True,reset={'jointPerturbations':[0.]*14,'yawPerturbation':0.,'qpos':[0.]*21,'qvel':[0.]*20},
                standHeight=.15,contactCoverage={'configured':True,'disabledGeoms':[],'footGeoms':['left_foot_collision','right_foot_collision']},
                commands={'enable':0.,'motion':3.,'stop':13.},samples=rows)


def test_complete_physical_evidence_passes_without_optional_replay():
    result=api().evaluate_episode(episode())
    assert result['evaluation']=='PASSED'
    assert result['metrics']['replay']['available'] is False


@pytest.mark.parametrize('change', ['missing','duplicate','truncated','gap','nonfinite','limit','contact','fall','stop'])
def test_evidence_is_never_promoted_by_ack_or_incomplete_sampling(change):
    e=episode()
    if change=='missing': del e['samples'][10]['finiteActions']
    elif change=='duplicate': e['samples'][10]['sequence']=10
    elif change=='truncated': e['samples']=e['samples'][:-100]
    elif change=='gap': del e['samples'][500:550]
    elif change=='nonfinite': e['samples'][10]['finiteState']=False
    elif change=='limit': e['samples'][10]['jointPositions'][0]=1.002
    elif change=='contact': e['samples'][10]['forbiddenContacts']=['head']
    elif change=='fall':
        for row in e['samples'][100:150]: row['trunkUp']=[1.,0.,0.]
    elif change=='stop':
        for row in e['samples'][2800:]: row['linearVelocity']=[.021,0.,0.]
    assert api().evaluate_episode(e)['evaluation']==('INCONCLUSIVE' if change in ('missing','duplicate','truncated','gap') else 'FAILED')


def test_fall_debounce_uses_simulation_time_and_resets_after_recovery():
    e=episode()
    for row in e['samples'][100:130]+e['samples'][150:180]: row['trunkUp']=[1.,0.,0.]
    assert api().evaluate_episode(e)['evaluation']=='PASSED'


def test_pcg64_reset_and_nine_scenarios():
    scenarios=api().scenarios()
    assert [(x['scenario'],x['seed']) for x in scenarios]==[(name,seed) for seed in (7,11,29) for name in ('stand','forward','turn')]
    reset=api().perturbations(7)
    assert reset['jointPerturbations'][0]==pytest.approx(.0012509546660466698)
    assert len(reset['jointPerturbations'])==14
    assert all(abs(x)<=.005 for x in reset['jointPerturbations'])
    assert abs(reset['yawPerturbation'])<=.01


def test_cancelled_or_legacy_evaluation_can_never_pass():
    mod=api()
    assert mod.evaluate_report({'runOutcome':'CANCELLED'})['evaluation']=='INCONCLUSIVE'
    assert mod.evaluate_report({'runtimeKind':'PYTHON_LEGACY','runOutcome':'COMPLETED'})['evaluation']=='INCONCLUSIVE'


def test_report_requires_actual_native_load_readback_in_each_fresh_world():
    mod=api(); identity={key:'sha256:'+'a'*64 for key in mod.IDENTITY_FIELDS};identity['profileDigest']=mod.PROFILE_DIGEST
    episodes=[]
    for scenario in mod.scenarios():
        e=episode();e.update(scenario);e['reset'].update(mod.perturbations(e['seed']));episodes.append(e)
    report={'runOutcome':'COMPLETED','runtimeKind':'NATIVE_ROBOTD','identity':identity,'expectedIdentity':identity,'episodes':episodes,'faults':[]}
    result=mod.evaluate_report(report)
    assert result['evaluation']=='INCONCLUSIVE'
    assert 'NATIVE_LOAD_READBACK_REQUIRED' in result['reasonCodes']


def test_full_native_profile_needs_nine_results_and_all_four_measured_faults():
    mod=api();identity={key:'sha256:'+'a'*64 for key in mod.IDENTITY_FIELDS};identity['profileDigest']=mod.PROFILE_DIGEST
    episodes=[]
    for scenario in mod.scenarios():
        e=episode();e.update(scenario);e['reset'].update(mod.perturbations(e['seed']));e['loadedPolicyDigest']=identity['candidateSha256'];episodes.append(e)
    faults=[{'kind':kind,'parentActive':True,'completed':True,'motionReplayed':False} for kind in mod.FAULTS]
    faults[0].update(disconnectedWallTime=1.,zeroRequestedWallTime=1.4,stopSimTime=13.,stopSamples=episode()['samples'][2600:])
    faults[1].update(duplicateApplied=False,watchdogExtended=False)
    for fault in faults[2:]:fault.update(oldAuthorityRejected=True,oldGeneration=1,newGeneration=2)
    report={'runOutcome':'COMPLETED','runtimeKind':'NATIVE_ROBOTD','identity':identity,'expectedIdentity':identity,'episodes':episodes,'faults':faults}
    assert mod.evaluate_report(report)['evaluation']=='PASSED'
    faults[0]['zeroRequestedWallTime']=1.501
    assert mod.evaluate_report(report)['evaluation']!='PASSED'
    report['runOutcome']='CANCELLED'
    assert mod.evaluate_report(report)['evaluation']=='INCONCLUSIVE'
