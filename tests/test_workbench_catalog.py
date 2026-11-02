"""CPU contract checks for canonical recipes and immutable experiment intent."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from mjlab_microduck.workbench import catalog, resolve_experiment, verify_experiment

SCHEMA = 'ROM_TRAINING_EXPERIMENT_V1'
WALK = 'Mjlab-Velocity-Flat-MicroDuck'


def request(recipe=WALK, profile='rom-xl330-v1', **config):
    return {'schemaVersion': SCHEMA, 'recipeId': recipe, 'robotProfileId': profile, 'config': config}


def test_catalog_has_all_35_recipes_and_14_distinct_action_families():
    value = catalog()
    assert value['catalogRevision'] == 'microduck-workbench-v1'
    assert Counter(r['actionId'] for r in value['recipes']) == {
        'joint': 4, 'sitstand': 4, 'ground_pick': 4, 'kick_left': 2, 'kick_right': 2,
        'roulade': 1, 'walk': 4, 'stand': 4, 'roller': 2, 'crouch': 2,
        'swizzle': 2, 'slope': 2, 'roller_stand': 1, 'spin': 1,
    }
    assert {p['robotProfileId'] for p in value['robotProfiles']} == {'rom-xl330-v1', 'hd1910-reference-v1'}
    for recipe in value['recipes']:
        for profile in value['robotProfiles']:
            spec = resolve_experiment(request(recipe['recipeId'], profile['robotProfileId']))
            assert verify_experiment(json.loads(json.dumps(spec))) == spec
            assert spec['effectiveConfig']['actionFilter'] == 'NONE'
            assert spec['effectiveConfig']['controlFrequencyHz'] == 50


def test_task_identity_preserves_left_kick_and_non_flat_roller_semantics():
    left = resolve_experiment(request('Mjlab-BallKickLeft-Flat-Backlash-MicroDuck'))
    assert left['taskId'] == 'Mjlab-BallKick-Flat-Backlash-MicroDuck'
    assert left['factoryOverrides'] == {'kick_foot': 'left'}
    assert left['actionId'] == 'kick_left'
    right = resolve_experiment(request('Mjlab-BallKick-Flat-Backlash-MicroDuck'))
    assert right['factoryOverrides'] == {}
    assert left['recipeDigest'] != right['recipeDigest']
    roller = resolve_experiment(request('Mjlab-Velocity-Flat-MicroDuck-Rollers'))
    swizzle = resolve_experiment(request('Mjlab-Velocity-Swizzle-MicroDuck'))
    slope = resolve_experiment(request('Mjlab-RollerSlope-Flat-MicroDuck'))
    assert roller['commandProfile']['twistSemantics'] == 'propulsion-brake-heading-error'
    assert swizzle['commandProfile']['twistSemantics'] == 'bidirectional-propulsion-heading-error'
    assert slope['terrain'] == 'slope'
    assert slope['commandProfile']['twistRanges'] == [[0, 0], [0, 0], [0, 0]]
    assert slope['commandProfile']['twistSemantics'] == 'passive-balance'


def test_profile_servo_settings_and_teacher_dependencies_are_explicit():
    xl = resolve_experiment(request())
    hd = resolve_experiment(request(profile='hd1910-reference-v1'))
    assert xl['effectiveConfig']['servo']['motorName'] == 'xl330'
    assert xl['effectiveConfig']['servo']['kpFirmware'] == 200
    assert hd['effectiveConfig']['servo']['motorName'] == 'hd1910'
    assert hd['effectiveConfig']['servo']['kpFirmware'] == 5
    assert xl['configDigest'] != hd['configDigest']
    joint = resolve_experiment(request('Mjlab-VelStand-Rough-Backlash-MicroDuck'))
    assert joint['teacherRequirements'] == [
        {'actionId': 'walk', 'recipeId': 'Mjlab-Velocity-Rough-Backlash-MicroDuck', 'robotProfileId': 'rom-xl330-v1'},
        {'actionId': 'stand', 'recipeId': 'Mjlab-StandUp-Rough-Backlash-MicroDuck', 'robotProfileId': 'rom-xl330-v1'},
    ]


def test_config_resolution_is_detached_and_digest_binds_requested_and_effective_data():
    value = request(numEnvs=64, iterations=5, seed=7)
    before = copy.deepcopy(value)
    spec = resolve_experiment(value)
    assert value == before
    assert spec['requestedConfig'] == before['config']
    assert spec['effectiveConfig']['numEnvs'] == 64
    assert spec['effectiveConfig']['iterations'] == 5
    assert spec['effectiveConfig']['seed'] == 7
    changed = resolve_experiment(request(numEnvs=64, iterations=6, seed=7))
    assert changed['configDigest'] != spec['configDigest']
    assert changed['recipeDigest'] == spec['recipeDigest']
    body = {k: v for k, v in spec.items() if k != 'configDigest'}
    # This example has no exponent-form or integral float edge cases.
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    assert spec['configDigest'] == 'sha256:' + hashlib.sha256(canonical.encode()).hexdigest()
    saved = copy.deepcopy(spec)
    value['config']['seed'] = 99
    next(r for r in catalog()['recipes'] if r['recipeId'] == WALK)['defaults']['seed'] = 99
    assert resolve_experiment(before) == saved
    for field in ('effectiveConfig', 'requestedConfig', 'commandProfile'):
        tampered = copy.deepcopy(saved)
        tampered[field]['unknown'] = True
        with pytest.raises(ValueError):
            verify_experiment(tampered)


@pytest.mark.parametrize('config', [
    {'numEnvs': 1.5}, {'numEnvs': True}, {'numEnvs': 0}, {'numEnvs': 16385},
    {'iterations': 0}, {'iterations': 1.0}, {'seed': -1}, {'seed': 2147483648},
    {'learningRate': float('nan')}, {'learningRate': float('inf')}, {'learningRate': '0.001'},
    {'learningRate': 0}, {'actionScale': 3}, {'episodeLengthSeconds': 121},
    {'numStepsPerEnv': 129}, {'ema': 0.5}, {'command': 'shell'}, {'servo': {'kpFirmware': 5}},
])
def test_resolver_refuses_unknown_nonfinite_fractional_or_out_of_bounds_settings(config):
    with pytest.raises(ValueError):
        resolve_experiment(request(**config))


@pytest.mark.parametrize('patch', [
    {'schemaVersion': 'ROM_TRAINING_EXPERIMENT_V2'}, {'robotProfileId': 'custom'},
    {'recipeId': 'Mjlab-SquatReference-Flat-MicroDuck'}, {'extra': True}, {'config': []},
])
def test_resolver_refuses_unknown_schema_profile_recipe_and_fields(patch):
    value = request(); value.update(patch)
    with pytest.raises(ValueError):
        resolve_experiment(value)


def test_shipped_json_schema_checks_examples_and_rejects_extra_fields():
    from jsonschema import Draft202012Validator
    root = Path(__file__).parents[1] / 'src/mjlab_microduck/workbench/resources'
    schema = json.loads((root / 'experiment.schema.json').read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    spec = resolve_experiment(request())
    validator.validate(spec)
    invalid = copy.deepcopy(spec); invalid['extra'] = True
    assert list(validator.iter_errors(invalid))


def test_canonical_numbers_are_portable_and_reject_executable_values():
    from mjlab_microduck.workbench import canonical_json
    assert canonical_json({'z': -0.0, 'b': 1.0, 'a': [0.000001, True, '中文']}) == '{"a":[0.000001,true,"中文"],"b":1,"z":0}'
    for invalid in ({'a': float('nan')}, {'a': object()}, {1: 'a'}, {'a': '\ud800'}):
        with pytest.raises(ValueError):
            canonical_json(invalid)


def test_catalog_defaults_and_command_ranges_match_real_registered_factories():
    # This optional heavy import builds configs only; it does not create an env,
    # touch a CUDA context, compile a model or step physics.
    import mjlab_microduck.tasks  # noqa: F401 - register the real local tasks
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
    registered = set(list_tasks())
    for recipe in catalog()['recipes']:
        assert recipe['taskId'] in registered
        env, rl = load_env_cfg(recipe['taskId']), load_rl_cfg(recipe['taskId'])
        assert recipe['defaults'] == {
            'numEnvs': env.scene.num_envs, 'iterations': rl.max_iterations, 'seed': rl.seed,
            'numStepsPerEnv': rl.num_steps_per_env, 'learningRate': rl.algorithm.learning_rate,
            'episodeLengthSeconds': env.episode_length_s, 'actionScale': env.actions['joint_pos'].scale,
        }
        command = recipe['commandProfile']
        assert command['twistRanges'] == [list(getattr(env.commands['twist'].ranges, key))
                                           for key in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z')]
        for key, name, size in [('headPoseRanges', 'head_pose', 4), ('bodyPoseRanges', 'body_pose', 6)]:
            if name in env.commands:
                assert command[key] == [list(pair) for pair in env.commands[name].ranges]
            else:
                assert command[key] == [[0, 0]] * size
    roller = load_env_cfg('Mjlab-Velocity-Flat-MicroDuck-Rollers')
    swizzle = load_env_cfg('Mjlab-Velocity-Swizzle-MicroDuck')
    slope = load_env_cfg('Mjlab-RollerSlope-Flat-MicroDuck')
    assert 'braking' in roller.rewards and 'braking' not in swizzle.rewards
    assert swizzle.rewards['wheel_speed'].params['bidirectional'] is True
    assert 'leg_symmetry' in swizzle.rewards and 'single_support' not in swizzle.rewards
    assert 'flat_ramp' in slope.scene.terrain.terrain_generator.sub_terrains


def test_left_factory_override_builds_distinct_real_ball_reset_semantics():
    from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import make_microduck_ball_kick_env_cfg
    left = make_microduck_ball_kick_env_cfg(kick_foot='left')
    right = make_microduck_ball_kick_env_cfg(kick_foot='right')
    # A left recipe must alter the actual ball side and support foot, not just
    # relabel a right-foot checkpoint. The exact event/term are upstream contracts.
    assert left.events['reset_ball'].params['offset'] == (0.09, 0.042)
    assert right.events['reset_ball'].params['offset'] == (0.09, -0.042)
    left_support = next(s for s in left.scene.sensors if s.name == 'support_foot_ground_contact')
    right_support = next(s for s in right.scene.sensors if s.name == 'support_foot_ground_contact')
    assert left_support.primary.pattern == '^right_foot_collision$'
    assert right_support.primary.pattern == '^left_foot_collision$'


def test_portable_examples_match_canonical_bytes_and_reject_modified_effective_config():
    from mjlab_microduck.workbench import canonical_json
    root = Path(__file__).parents[1] / 'src/mjlab_microduck/workbench/resources'
    example_request = json.loads((root / 'example-request.json').read_text())
    example = json.loads((root / 'example-resolved.json').read_text())
    assert resolve_experiment(example_request) == example
    assert canonical_json(example) + '\n' == (root / 'example-resolved.canonical.json').read_text()
    assert example['configDigest'] == 'sha256:8c562ba28830e1c40c08f7f16d2e553113e0780c8b211f3f82f5cec24aa365be'
    with pytest.raises(ValueError):
        verify_experiment(json.loads((root / 'example-invalid.json').read_text()))


def test_effective_command_ranges_and_phase_period_are_frozen_without_reinterpreting_tasks():
    value=resolve_experiment(request(commands={'twist':{'ranges':{'lin_vel_x':[-.1,.2]}}}))
    assert value['commandProfile']['twistRanges'][0]==[-.1,.2]
    phase=resolve_experiment(request('Mjlab-RollerCrouch-Flat-MicroDuck',commands={'twist':{'period':6}}))
    assert phase['commandProfile']['phasePeriodSeconds']==6
    for recipe in ('Mjlab-RollerSlope-Flat-MicroDuck','Mjlab-RollerCrouch-Flat-MicroDuck','Mjlab-SitStand-Flat-MicroDuck'):
        with pytest.raises(ValueError):resolve_experiment(request(recipe,commands={'twist':{'ranges':{'lin_vel_x':[-.1,.1]}}}))


@pytest.mark.parametrize('name',['twist','head_pose','body_pose'])
@pytest.mark.parametrize('duration',[True,[10,-1],[1],[1,2,3],[False,2],[.09,1],[1,121],[2,1],['1',2],float('inf')])
def test_command_resampling_duration_rejects_invalid_shapes_and_bounds(name,duration):
    with pytest.raises(ValueError):resolve_experiment(request(commands={name:{'resampling_time_range':duration}}))


@pytest.mark.parametrize('name',['twist','head_pose','body_pose'])
def test_command_resampling_duration_keeps_valid_bounds(name):
    value=resolve_experiment(request(commands={name:{'resampling_time_range':[.1,120]}}))
    assert value['effectiveConfig']['commands'][name]['resampling_time_range']==[.1,120]
