"""V2 envelope proof and the one supported native walking evaluation profile.

Full experiment provenance stays in the original bundle. The transport projection
contains only the existing PolicyMetadata fields and is never a rewritten bundle.
"""
from copy import deepcopy
from pathlib import Path
from .configuration import verify_experiment
from mjlab_microduck.rom.native_qualification import PROFILE_DIGEST
from mjlab_microduck.rom.managed_bundle import read_envelope, metadata_valid, _prove, require, SCHEMA

PROFILE = 'microduck-walk-flat-native-v1'
CORE_FIELDS = frozenset(('policySha256','modelDigest','recipeDigest','configDigest','taskId','sourceCommit',
    'checkpointSha256','observationDimension','actionDimension','controlHz','observationLayout',
    'jointNames','normalizerDigest','normalizerBaked','provenance','subjectDigest'))
EXPERIMENT_FIELDS = frozenset(('actionId','robotProfileId','experimentConfigDigest','experimentRecipeDigest',
    'commandProfile','actionFilter','executionEnvironment','qualified'))


def native_evaluation_support(experiment, metadata):
    spec = verify_experiment(experiment)
    def unsupported(reason):return dict(supported=False,profileId='',profileDigest='',reasonCode=reason)
    if (spec['recipeId'] != 'Mjlab-Velocity-Flat-MicroDuck' or spec['actionId'] != 'walk'
            or spec['robotProfileId'] != 'rom-xl330-v1' or spec['terrain'] != 'flat' or spec['backlash']
            or spec['teacherRequirements'] or spec['commandProfile']['id'] != 'walk-v1'
            or spec['commandProfile']['twistSemantics'] != 'body-velocity'):
        return unsupported('POLICY_NATIVE_PROFILE_UNSUPPORTED')
    config = spec['effectiveConfig']
    if config['actionScale'] != 1 or metadata['actionFilter'] != 'NONE':
        return unsupported('POLICY_NATIVE_ACTION_UNSUPPORTED')
    # Learning/critic/noise/curriculum changes do not redefine raw actor inputs.
    for fields in config.get('observations',{}).get('actor',{}).values():
        if (fields.get('scale',1) != 1 or 'clip' in fields or fields.get('params')
                or fields.get('delay_min_lag',0) != 0 or fields.get('delay_max_lag',0) != 0):
            return unsupported('POLICY_NATIVE_OBSERVATION_UNSUPPORTED')
    def nonzero(value):
        if isinstance(value,dict):return any(nonzero(v) for v in value.values())
        if isinstance(value,list):return any(nonzero(v) for v in value)
        return value != 0
    if nonzero(config.get('calibration',{})):
        return unsupported('POLICY_NATIVE_OBSERVATION_UNSUPPORTED')
    return dict(supported=True,profileId=PROFILE,profileDigest=PROFILE_DIGEST,reasonCode='')


def verify_native_bundle(path):
    manifest, data = read_envelope(Path(path),manifest_limit=131072)
    require(type(manifest) is dict,'BUNDLE_MANIFEST_INVALID')
    metadata=manifest['policy']
    if manifest.get('schema') == SCHEMA:
        require(set(manifest)=={'schema','policy','policySizeBytes'},'BUNDLE_SCHEMA_INVALID')
        metadata_valid(metadata)
        support=dict(supported=True,profileId=PROFILE,profileDigest=PROFILE_DIGEST,reasonCode='')
    else:
        require(manifest.get('schema')=='ROM_MANAGED_POLICY_BUNDLE_V2'
                and set(manifest)=={'schema','policy','policySizeBytes','experiment'},'BUNDLE_SCHEMA_INVALID')
        spec=verify_experiment(manifest['experiment'])
        require(set(metadata)==(CORE_FIELDS-{'subjectDigest'})|EXPERIMENT_FIELDS,'POLICY_METADATA_INVALID')
        for key in ('taskId','actionId','robotProfileId','commandProfile'):
            require(metadata[key]==spec[key],'POLICY_EXPERIMENT_INVALID')
        require(metadata['experimentConfigDigest']==spec['configDigest']
                and metadata['experimentRecipeDigest']==spec['recipeDigest'],'POLICY_EXPERIMENT_INVALID')
        require(metadata['executionEnvironment']=='SIMULATOR' and metadata['provenance']=='MANAGED'
                and metadata['qualified'] is False and metadata['actionFilter']=='NONE','POLICY_CONTRACT_INCOMPATIBLE')
        metadata_valid({key:value for key,value in metadata.items() if key in CORE_FIELDS},task_id=spec['taskId'])
        support=native_evaluation_support(spec,metadata)
    require(_prove(data)==metadata['normalizerDigest'],'POLICY_NORMALIZER_MISMATCH')
    return dict(metadata=deepcopy(metadata),wireMetadata={key:deepcopy(value) for key,value in metadata.items() if key in CORE_FIELDS},nativeEvaluation=support)
