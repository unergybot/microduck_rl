"""CPU evidence that observation hooks report the pinned logger's real values."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('training_console', Path(__file__).parents[1] / 'src/mjlab_microduck/training_console.py')


def drain(collector):
    collector.close()
    collector._thread.join(timeout=3)
    assert not collector._thread.is_alive()


def console():
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('start_it,total_it', [(0,5), (3,8)])
def test_actual_pinned_logger_scalars_and_iteration_are_preserved(tmp_path, start_it, total_it):
    module = console()
    import torch
    from rsl_rl.utils.logger import Logger
    logger = Logger(str(tmp_path / 'work'), {'num_steps_per_env': 24, 'algorithm': {'rnd_cfg': None}}, {}, 64, False, 1, 0, 'cpu')
    runner = SimpleNamespace(logger=logger)
    collector = module.observe_runner(runner, tmp_path, {'jobId': 'job-a', 'requestDigest': 'sha256:' + 'a'*64})
    logger.init_logging_writer()
    logger.rewbuffer.extend([2., 4.]); logger.lenbuffer.extend([10., 20.])
    logger.log(it=3, start_it=start_it, total_it=total_it, collect_time=1., learn_time=1., loss_dict={'value': .25}, learning_rate=.01, action_std=torch.tensor([.5]), rnd_weight=None)
    drain(collector)
    body = json.loads((tmp_path / 'console/metrics.json').read_text())
    assert body['progress'] == {'currentIteration': 3, 'targetIterations': total_it, 'elapsedSeconds': 2., 'samplesPerSecond': 768.}
    metrics = {m['key']: m['points'] for m in body['metrics']}
    assert metrics['Train/mean_reward'] == [{'step': 3, 'value': 3.}]
    assert metrics['Train/mean_episode_length'] == [{'step': 3, 'value': 15.}]
    assert metrics['Loss/value'] == [{'step': 3, 'value': .25}]
    assert logger.tot_timesteps == 1536 and logger.tot_time == 2.
    logger.writer.close()


def test_collector_bounds_history_series_and_finite_numbers(tmp_path):
    module = console()
    collector = module.Collector(tmp_path, {'jobId': 'job-a', 'requestDigest': 'sha256:' + 'a'*64})
    for value in [float('nan'), float('inf'), True]: collector.scalar('invalid', value, 1)
    collector.scalar('invalid-step', 2, -1)
    for step in range(2002): collector.scalar('reward', step / 10, step)
    for key in range(80): collector.scalar('metric' + str(key), 1, 1)
    collector.flush({'currentIteration': 2001, 'targetIterations': 3000, 'elapsedSeconds': float('inf'), 'samplesPerSecond': 3})
    drain(collector)
    raw = (tmp_path / 'console/metrics.json').read_bytes(); body = json.loads(raw)
    assert len(raw) <= 2 << 20
    assert len([m for m in body['metrics'] if not m['key'].startswith(module.PREFIXES)]) == 28
    assert next(m for m in body['metrics'] if m['key'] == 'Measurement/droppedPoints')['points'][0]['value'] > 0
    reward = next(x for x in body['metrics'] if x['key'] == 'reward')['points']
    assert len(reward) == 2000 and reward[0] == {'step': 2, 'value': .2}
    assert body['progress']['elapsedSeconds'] is None
    assert all(x['key'] not in ('invalid', 'invalid-step') for x in body['metrics'])


def test_observation_failure_does_not_change_logger_return_or_exception(tmp_path):
    module = console()
    class Logger:
        writer = None
        tot_time = 1
        def init_logging_writer(self): return 17
        def log(self, **kwargs):
            if kwargs.get('fail'): raise RuntimeError('original failure')
            return 23
    logger = Logger()
    (tmp_path / 'console').write_text('unavailable sink')
    collector = module.observe_runner(SimpleNamespace(logger=logger), tmp_path, {'jobId': 'job-a', 'requestDigest': 'sha256:'+'a'*64})
    assert logger.init_logging_writer() == 17
    assert logger.log(it=0, total_it=5) == 23
    with pytest.raises(RuntimeError, match='original failure'): logger.log(fail=True)
    drain(collector)
    assert collector.state == 2


def test_serialized_observations_fit_snapshot_budget_at_maximum_series(tmp_path):
    module=console();collector=module.Collector(tmp_path, {'jobId':'job-a','requestDigest':'sha256:'+'a'*64})
    for key in range(64):
        for step in range(2000): collector.scalar('metric'+str(key), 1.23456789123456e100, step)
    collector.flush({})
    drain(collector)
    body=json.loads((tmp_path/'console/metrics.json').read_bytes())
    assert (tmp_path/'console/metrics.json').stat().st_size <= 2 << 20
    assert len([m for m in body['metrics'] if not m['key'].startswith(module.PREFIXES)]) == 28
    assert all(0 < len(series['points']) <= 2000 for series in body['metrics'])


@pytest.mark.parametrize('points', [0, 1, 2])
def test_oversize_metadata_terminates_without_publishing(tmp_path, points):
    import subprocess
    import sys
    script = '''
import importlib.util
from pathlib import Path
import sys
spec = importlib.util.spec_from_file_location('console', sys.argv[1])
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
collector = module.Collector(sys.argv[2], {'jobId': 'x' * (module.MAX_BYTES + 1), 'requestDigest': 'sha256:' + 'a' * 64})
for step in range(int(sys.argv[3])): collector.scalar('reward', 1, step)
try: collector.flush({})
except ValueError: pass
collector.close(); collector._thread.join(1)
assert not (Path(sys.argv[2]) / 'console/metrics.json').exists()
'''
    subprocess.run([sys.executable, '-c', script, str(Path(spec.origin)), str(tmp_path), str(points)],
                   timeout=3, check=True, capture_output=True)


def test_writer_hook_setup_failure_preserves_original_return(tmp_path):
    module = console()
    class Writer:
        @property
        def add_scalar(self): raise OSError('optional writer interception unavailable')
    logger = SimpleNamespace(writer=Writer(), init_logging_writer=lambda: 17, log=lambda: 23)
    collector = module.observe_runner(SimpleNamespace(logger=logger), tmp_path, {'jobId': 'job-a', 'requestDigest': 'sha256:' + 'a'*64})
    assert logger.init_logging_writer() == 17
    drain(collector)


@pytest.mark.parametrize('start', [0, 100])
def test_actual_cpu_logger_counts_invocation_iterations_with_actual_seven_steps(tmp_path, start):
    module = console()
    import torch
    from rsl_rl.utils.logger import Logger
    assert not torch.cuda.is_initialized()
    cfg = {'num_steps_per_env': 7, 'algorithm': {'rnd_cfg': None}}
    logger = Logger(str(tmp_path / 'work'), cfg, {}, 64, False, 1, 0, 'cpu')
    runner = SimpleNamespace(logger=logger, cfg=cfg, env=SimpleNamespace(num_envs=64))
    clock = [1.]
    measurements = module.RunMeasurements({'numEnvs': 64}, clock=lambda: clock[0])
    collector = module.observe_runner(runner, tmp_path, {'jobId': 'job-a', 'requestDigest': 'sha256:' + 'a' * 64}, measurements=measurements)
    try:
        logger.init_logging_writer()
        measurements.begin('training')
        for offset in range(2):
            clock[0] += 1
            logger.log(it=start + offset, start_it=start, total_it=start + 5, collect_time=1.,
                       learn_time=1., loss_dict={'value': .25}, learning_rate=.01,
                       action_std=torch.tensor([.5]), rnd_weight=None)
        measurements.end('training')
        step, points = measurements.snapshot()
        assert step == 2
        assert points['MeasureTraining/completedTransitions'] == 896
        assert points['MeasureTraining/transitionsPerSecond'] == 448
        assert points['MeasurePhase/training/state'] == 2
        assert logger.tot_timesteps == 896 and logger.tot_time == 4
        measurements.begin('cleanup'); measurements.end('cleanup')
        collector.flush({})
        measurements.terminal_written(True)
        collector.flush({})
    finally:
        drain(collector); logger.writer.close()
    body = json.loads((tmp_path / 'console/metrics.json').read_text())
    assert body['progress'] == {'currentIteration': start + 1, 'targetIterations': start + 5,
                                'elapsedSeconds': 4., 'samplesPerSecond': 224.}
    assert next(m for m in body['metrics'] if m['key'] == 'Measurement/version')['points'][0]['step'] == 2
    assert not torch.cuda.is_initialized()


def test_actual_logger_partial_iteration_preserves_original_exception_and_omits_rate(tmp_path):
    module = console()
    import torch
    from rsl_rl.utils.logger import Logger
    cfg = {'num_steps_per_env': 7, 'algorithm': {'rnd_cfg': None}}
    logger = Logger(str(tmp_path / 'work'), cfg, {}, 64, False, 1, 0, 'cpu')
    measurements = module.RunMeasurements({'numEnvs': 64})
    collector = module.observe_runner(SimpleNamespace(logger=logger, cfg=cfg, env=SimpleNamespace(num_envs=64)),
        tmp_path, {'jobId': 'job-a', 'requestDigest': 'sha256:' + 'a' * 64}, measurements=measurements)
    original = RuntimeError('original upstream writer exception')
    try:
        logger.init_logging_writer()
        measurements.begin('training')
        def broken(*args, **kwargs): raise original
        logger.writer.add_scalar = broken
        with pytest.raises(RuntimeError) as caught:
            logger.log(it=100, start_it=100, total_it=105, collect_time=1., learn_time=1.,
                       loss_dict={'value': .25}, learning_rate=.01, action_std=torch.tensor([.5]), rnd_weight=None)
        assert caught.value is original
        measurements.end('training', 3)
        step, points = measurements.snapshot()
        assert step == 0 and points['MeasurePhase/training/state'] == 3
        assert 'MeasureTraining/completedTransitions' not in points
        assert 'MeasureTraining/transitionsPerSecond' not in points
    finally:
        drain(collector); logger.writer.close()
    assert not torch.cuda.is_initialized()



def test_summary_preserves_unsupplied_progress_but_explicit_unknown_stays_unknown(tmp_path):
    module = console()
    collector = module.Collector(tmp_path, {'jobId': 'job-a', 'requestDigest': 'sha256:' + 'a' * 64})
    collector.flush({'currentIteration': 4, 'targetIterations': 5, 'elapsedSeconds': 10., 'samplesPerSecond': 100.})
    collector.flush({'elapsedSeconds': float('inf'), 'samplesPerSecond': None})
    collector.flush({})
    drain(collector)
    body = json.loads((tmp_path / 'console/metrics.json').read_text())
    assert body['progress'] == {'currentIteration': 4., 'targetIterations': 5.,
                                'elapsedSeconds': None, 'samplesPerSecond': None}
