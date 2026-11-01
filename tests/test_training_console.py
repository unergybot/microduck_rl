"""CPU evidence that observation hooks report the pinned logger's real values."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('training_console', Path(__file__).parents[1] / 'src/mjlab_microduck/training_console.py')


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
    module.observe_runner(runner, tmp_path, {'jobId': 'job-a', 'requestDigest': 'sha256:' + 'a'*64})
    logger.init_logging_writer()
    logger.rewbuffer.extend([2., 4.]); logger.lenbuffer.extend([10., 20.])
    logger.log(it=3, start_it=start_it, total_it=total_it, collect_time=1., learn_time=1., loss_dict={'value': .25}, learning_rate=.01, action_std=torch.tensor([.5]), rnd_weight=None)
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
    raw = (tmp_path / 'console/metrics.json').read_bytes(); body = json.loads(raw)
    assert len(raw) <= 2 << 20
    assert len(body['metrics']) == 64
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
    module.observe_runner(SimpleNamespace(logger=logger), tmp_path, {'jobId': 'job-a', 'requestDigest': 'sha256:'+'a'*64})
    assert logger.init_logging_writer() == 17
    assert logger.log(it=0, total_it=5) == 23
    with pytest.raises(RuntimeError, match='original failure'): logger.log(fail=True)


def test_serialized_observations_fit_snapshot_budget_at_maximum_series(tmp_path):
    module=console();collector=module.Collector(tmp_path, {'jobId':'job-a','requestDigest':'sha256:'+'a'*64})
    for key in range(64):
        for step in range(2000): collector.scalar('metric'+str(key), 1.23456789123456e100, step)
    collector.flush({})
    body=json.loads((tmp_path/'console/metrics.json').read_bytes())
    assert (tmp_path/'console/metrics.json').stat().st_size <= 2 << 20
    assert len(body['metrics'])==64
    assert all(0 < len(series['points']) <= 2000 for series in body['metrics'])
