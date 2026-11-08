"""Strict typed resolution against the versioned portable recipe snapshot."""
from copy import deepcopy
from .parameters import GROUPS, validate_groups
from decimal import Decimal
import hashlib
from importlib.resources import files
import json
import math

_SCHEMA = 'ROM_TRAINING_EXPERIMENT_V1'
_FIELDS = {
    'numEnvs': (int, 1, 16384),
    'iterations': (int, 1, 100000),
    'seed': (int, 0, 2147483647),
    'numStepsPerEnv': (int, 8, 128),
    'learningRate': (float, 0.000001, 0.01),
    'episodeLengthSeconds': (float, 1, 120),
    'actionScale': (float, 0.01, 2),
}


def canonical_json(value: object) -> str:
    """Canonical UTF-8 JSON text: sorted keys and plain minimal decimal numbers.

    Integral numeric values share an encoding (1.0 -> 1); -0 encodes as 0.
    Booleans are JSON booleans, not numeric values. No nonfinite numbers,
    executable objects, lone Unicode surrogates or non-string object keys.
    """
    def encode(item, depth):
        if depth > 20:
            raise ValueError('WORKBENCH_DATA_INVALID')
        if item is None:
            return 'null'
        if type(item) is bool:
            return 'true' if item else 'false'
        if type(item) in (int, float):
            if type(item) is float and not math.isfinite(item):
                raise ValueError('WORKBENCH_DATA_INVALID')
            text = format(Decimal(str(item)), 'f')
            if '.' in text:
                text = text.rstrip('0').rstrip('.')
            return '0' if text == '-0' else text
        if type(item) is str:
            try:
                item.encode('utf-8')
            except UnicodeError as exc:
                raise ValueError('WORKBENCH_DATA_INVALID') from exc
            return json.dumps(item, ensure_ascii=False, separators=(',', ':'))
        if type(item) is list:
            return '[' + ','.join(encode(v, depth + 1) for v in item) + ']'
        if type(item) is dict and all(type(k) is str for k in item):
            return '{' + ','.join(encode(k, depth + 1) + ':' + encode(item[k], depth + 1)
                                  for k in sorted(item)) + '}'
        raise ValueError('WORKBENCH_DATA_INVALID')
    return encode(value, 0)


def _digest(value):
    return 'sha256:' + hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def catalog() -> dict:
    """Return a detached portable catalog, with no engine imports or side effects."""
    return json.loads(files(__package__).joinpath('resources/catalog.json').read_text(encoding='utf-8'))


def _config(value):
    if type(value) is not dict or value.keys() - (_FIELDS.keys() | GROUPS):
        raise ValueError('WORKBENCH_CONFIG_INVALID')
    validate_groups(value)
    for key, number in value.items():
        if key in GROUPS: continue
        kind, low, high = _FIELDS[key]
        allowed = (int,) if kind is int else (int, float)
        if type(number) not in allowed or not low <= number <= high:
            raise ValueError('WORKBENCH_CONFIG_INVALID')
    # Normalize representation for cross-language JSON round trips.
    return json.loads(canonical_json(value))


def resolve_experiment(request: dict) -> dict:
    """Resolve a recipe/profile and bounded overrides without modifying inputs."""
    if type(request) is not dict or request.keys() - {'schemaVersion', 'recipeId', 'robotProfileId', 'config'}:
        raise ValueError('WORKBENCH_REQUEST_INVALID')
    if request.get('schemaVersion') != _SCHEMA:
        raise ValueError('WORKBENCH_SCHEMA_INVALID')
    data = catalog()
    recipe = next((r for r in data['recipes'] if r['recipeId'] == request.get('recipeId')), None)
    profile = next((p for p in data['robotProfiles'] if p['robotProfileId'] == request.get('robotProfileId')), None)
    if recipe is None:
        raise ValueError('WORKBENCH_RECIPE_INVALID')
    if profile is None:
        raise ValueError('WORKBENCH_PROFILE_INVALID')
    requested = _config(request.get('config', {}))
    effective = {**deepcopy(recipe['defaults']), **requested,
                 'servo': deepcopy(profile['servo']), 'actionFilter': 'NONE',
                 'controlFrequencyHz': 50, 'normalizeObservations': True}
    teachers = []
    for action in recipe['teacherActionIds']:
        teacher = next(r for r in data['recipes'] if r['actionId'] == action
                       and r['terrain'] == recipe['terrain'] and r['backlash'] == recipe['backlash'])
        teachers.append({'actionId': action, 'recipeId': teacher['recipeId'],
                         'robotProfileId': profile['robotProfileId']})
    result = {'schemaVersion': _SCHEMA, 'catalogRevision': data['catalogRevision'],
              **{key: deepcopy(recipe[key]) for key in
                 ('recipeId', 'taskId', 'actionId', 'terrain', 'backlash', 'factoryOverrides', 'commandProfile')},
              'robotProfileId': profile['robotProfileId'], 'teacherRequirements': teachers,
              'requestedConfig': requested, 'effectiveConfig': effective, 'recipeDigest': _digest(recipe)}
    commands=requested.get('commands',{})
    for name,changes in commands.items():
        if name not in ('twist','head_pose','body_pose') or type(changes) is not dict or changes.keys()-{'ranges','period','resampling_time_range'}:
            raise ValueError('WORKBENCH_CONFIG_INVALID')
        if 'resampling_time_range' in changes:
            duration=changes['resampling_time_range']
            if type(duration) is not list or len(duration)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in duration) or not .1<=duration[0]<=duration[1]<=120:
                raise ValueError('WORKBENCH_CONFIG_INVALID')
        ranges=changes.get('ranges')
        if ranges is not None:
            if name=='twist':
                if result['commandProfile']['twistSemantics'] not in ('body-velocity','bidirectional-propulsion-heading-error','propulsion-brake-heading-error'):raise ValueError('WORKBENCH_CONFIG_INVALID')
                fields={'lin_vel_x':0,'lin_vel_y':1,'ang_vel_z':2}
                if type(ranges) is not dict or ranges.keys()-fields.keys():raise ValueError('WORKBENCH_CONFIG_INVALID')
                pairs=[(fields[k],v) for k,v in ranges.items()]
                target='twistRanges'
            else:
                size=4 if name=='head_pose' else 6
                if type(ranges) is not list or len(ranges)!=size:raise ValueError('WORKBENCH_CONFIG_INVALID')
                pairs=list(enumerate(ranges));target='headPoseRanges' if name=='head_pose' else 'bodyPoseRanges'
            for index,pair in pairs:
                if type(pair) is not list or len(pair)!=2 or any(type(v) not in (int,float) or abs(v)>1000 for v in pair) or pair[0]>pair[1]:raise ValueError('WORKBENCH_CONFIG_INVALID')
                result['commandProfile'][target][index]=deepcopy(pair)
        if 'period' in changes:
            if name!='twist' or result['commandProfile']['twistSemantics']!='phase-clock' or type(changes['period']) not in (int,float) or not .1<=changes['period']<=120:raise ValueError('WORKBENCH_CONFIG_INVALID')
            result['commandProfile']['phasePeriodSeconds']=changes['period']
    result['configDigest'] = _digest(result)
    return result


def verify_experiment(spec: dict) -> dict:
    """Re-resolve intent and reject any altered derived data, schema or digest.

    Verification returns a fresh specification. It does not bless checkpoints,
    provenance, engine availability, numerical qualification or admission.
    """
    if type(spec) is not dict:
        raise ValueError('WORKBENCH_SPEC_INVALID')
    expected = resolve_experiment({'schemaVersion': spec.get('schemaVersion'),
                                   'recipeId': spec.get('recipeId'),
                                   'robotProfileId': spec.get('robotProfileId'),
                                   'config': spec.get('requestedConfig')})
    # The request config check is strict for integer fields. Canonical comparison
    # additionally treats JSON numeric representations consistently on transport.
    if canonical_json(spec) != canonical_json(expected):
        raise ValueError('WORKBENCH_SPEC_INVALID')
    return expected
