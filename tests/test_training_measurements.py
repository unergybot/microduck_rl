"""CPU contracts for honest host measurements and optional console isolation."""
import importlib.util
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def console():
    spec = importlib.util.spec_from_file_location(
        'measure_console', Path(__file__).parents[1] / 'src/mjlab_microduck/training_console.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


JOB = {'jobId': 'job-a', 'requestDigest': 'sha256:' + 'a' * 64, 'numEnvs': 64}


def test_upstream_cannot_spoof_reserved_measurements(tmp_path):
    """A logger collision must not manufacture an owned measurement."""
    module = console()
    collector = module.Collector(tmp_path, JOB)
    collector.scalar('MeasureMemory/deviceUsedSampledMaxBytes', 123, 9)
    try:
        assert 'MeasureMemory/deviceUsedSampledMaxBytes' not in collector.series
    finally:
        collector.close(); collector._thread.join(timeout=1)


EXPECTED_KEYS = {
    'MeasurePhase/initialization/seconds', 'MeasurePhase/initialization/state',
    'MeasurePhase/training/seconds', 'MeasurePhase/training/state',
    'MeasurePhase/checkpoint/seconds', 'MeasurePhase/checkpoint/state',
    'MeasurePhase/onnx/seconds', 'MeasurePhase/onnx/state',
    'MeasurePhase/cleanup/seconds', 'MeasurePhase/cleanup/state',
    'MeasurePhase/package/seconds', 'MeasurePhase/package/state',
    'MeasureCheckpoint/calls', 'MeasureMemory/deviceUsedSampledMaxBytes',
    'MeasureMemory/deviceSampleCount', 'MeasureMemory/deviceSampleFailures',
    'MeasureMemory/deviceFirstSampleSeconds', 'MeasureMemory/deviceLastSampleSeconds',
    'MeasureMemory/deviceStatus', 'MeasureMemory/torchAllocatedPeakBytes',
    'MeasureMemory/torchReservedPeakBytes', 'MeasureMemory/torchStatus',
    'MeasureTraining/completedTransitions', 'MeasureTraining/transitionsPerSecond',
    'MeasureObserver/hookCalls', 'MeasureObserver/hookSeconds', 'MeasureObserver/state',
    'Measurement/version', 'Measurement/snapshotElapsedSeconds', 'Measurement/terminalState',
    'Measurement/consoleState', 'Measurement/droppedPoints',
}


def drain(collector):
    collector.close()
    collector._thread.join(timeout=3)
    assert not collector._thread.is_alive()


def metrics(path):
    return {m['key']: m['points'] for m in json.loads(path.read_bytes())['metrics']}


def test_initial_summary_omits_unknowns_and_keeps_exact_contract():
    module = console()
    recorder = module.RunMeasurements(JOB, clock=lambda: 10., started=7.)
    step, values = recorder.snapshot()
    assert step == 0 and values['MeasurePhase/initialization/seconds'] == 3
    assert values['MeasurePhase/initialization/state'] == 1
    assert values['MeasureObserver/state'] == 0
    assert values['MeasureMemory/deviceStatus'] == 0
    assert 'MeasureObserver/hookCalls' not in values
    assert 'MeasureMemory/deviceUsedSampledMaxBytes' not in values
    assert 'MeasureTraining/completedTransitions' not in values
    assert 'MeasurePhase/training/seconds' not in values
    assert module.DIAGNOSTIC_KEYS == EXPECTED_KEYS and len(EXPECTED_KEYS) == 32
    assert set(values) <= EXPECTED_KEYS


def test_aggregate_phases_repeated_checkpoints_and_training_overlap():
    module = console()
    clock = [10.]
    recorder = module.RunMeasurements(JOB, clock=lambda: clock[0], started=1.)
    recorder.configure_iterations(7, 64)
    recorder.end('initialization')
    recorder.begin('training')
    clock[0] = 11.
    with recorder.phase('checkpoint'):
        clock[0] = 13.
    with pytest.raises(RuntimeError, match='save original'), recorder.phase('checkpoint'):
        clock[0] = 16.
        raise RuntimeError('save original')
    clock[0] = 18.
    recorder.end('training')
    with recorder.phase('checkpoint'):
        clock[0] = 19.
        values = recorder.snapshot()[1]
        assert values['MeasurePhase/checkpoint/state'] == 3
        assert values['MeasurePhase/checkpoint/seconds'] == 6
    values = recorder.snapshot()[1]
    assert values['MeasureCheckpoint/calls'] == 3
    assert values['MeasurePhase/training/seconds'] == 8
    assert values['MeasurePhase/checkpoint/seconds'] == 6
    assert values['MeasurePhase/initialization/seconds'] == 9
    recorder.end('package', 4)
    assert 'MeasurePhase/package/seconds' not in recorder.snapshot()[1]


@pytest.mark.parametrize('steps,envs', [(None, 64), (7, 63), (0, 64), (True, 64)])
def test_unproven_configuration_omits_throughput_and_marks_partial(steps, envs):
    module = console()
    recorder = module.RunMeasurements(JOB, clock=lambda: 1.)
    recorder.configure_iterations(steps, envs)
    recorder.begin('training'); recorder.end('training')
    values = recorder.snapshot()[1]
    assert values['MeasurePhase/training/state'] == 5
    assert 'MeasureTraining/completedTransitions' not in values
    assert 'MeasureTraining/transitionsPerSecond' not in values


def test_resumed_iterations_zero_duration_and_partial_callback():
    module = console()
    clock = [0.]
    recorder = module.RunMeasurements(JOB, clock=lambda: clock[0])
    recorder.configure_iterations(7, 64); recorder.begin('training')
    recorder.completed_iteration(100, 100, steps=7, envs=64, writer=True)
    step, values = recorder.snapshot()
    assert step == 1 and values['MeasureTraining/completedTransitions'] == 448
    assert values['MeasurePhase/training/seconds'] == 0.
    assert 'MeasureTraining/transitionsPerSecond' not in values
    clock[0] = 2.
    recorder.completed_iteration(101, 100, steps=7, envs=64, writer=True)
    assert recorder.snapshot()[1]['MeasureTraining/transitionsPerSecond'] == 448
    recorder.completed_iteration(103, 100, steps=7, envs=64, writer=True)
    assert recorder.snapshot()[0] == 3
    recorder.completed_iteration(103, 100, steps=7, envs=64, writer=True)
    assert recorder.snapshot()[0] == 3
    recorder.end('training')
    assert recorder.snapshot()[1]['MeasurePhase/training/state'] == 5
    assert 'MeasureTraining/completedTransitions' not in recorder.snapshot()[1]


def test_allocator_values_are_distinct_and_failed_read_retains_partial_values():
    module = console()
    recorder = module.RunMeasurements(JOB)
    cuda = SimpleNamespace(max_memory_allocated=lambda _: 123, max_memory_reserved=lambda _: 456)
    recorder.device = {'MeasureMemory/deviceUsedSampledMaxBytes': 789}
    recorder.read_torch(cuda, final=True)
    values = recorder.snapshot()[1]
    assert values['MeasureMemory/torchAllocatedPeakBytes'] == 123
    assert values['MeasureMemory/torchReservedPeakBytes'] == 456
    assert values['MeasureMemory/deviceUsedSampledMaxBytes'] == 789
    assert values['MeasureMemory/torchStatus'] == 2
    cuda.max_memory_reserved = lambda _: (_ for _ in ()).throw(RuntimeError('query'))
    recorder.read_torch(cuda, final=True)
    assert recorder.snapshot()[1]['MeasureMemory/torchStatus'] == 3
    assert recorder.snapshot()[1]['MeasureMemory/torchReservedPeakBytes'] == 456


def full_recorder(module):
    class ReservedCapacity(module.RunMeasurements):
        """Capacity fixture for future M2's two otherwise absent hook values."""
        def snapshot(self, **kwargs):
            step, values = super().snapshot(**kwargs)
            values.update({'MeasureObserver/hookCalls': 0, 'MeasureObserver/hookSeconds': 0})
            return step, values
    recorder = ReservedCapacity(JOB, clock=lambda: 3., started=0.)
    recorder.configure_iterations(7, 64)
    for phase in module.PHASES:
        recorder.begin(phase); recorder.end(phase)
    recorder.device = {
        'MeasureMemory/deviceUsedSampledMaxBytes': 123,
        'MeasureMemory/deviceSampleCount': 1, 'MeasureMemory/deviceSampleFailures': 0,
        'MeasureMemory/deviceFirstSampleSeconds': 1, 'MeasureMemory/deviceLastSampleSeconds': 1,
        'MeasureMemory/deviceStatus': 2,
    }
    recorder.read_torch(SimpleNamespace(max_memory_allocated=lambda _: 10, max_memory_reserved=lambda _: 20), final=True)
    # Equal-clock training has no valid rate; an observed 2 s span supplies one.
    recorder.phases['training'][1] = 2
    return recorder


def test_32_normal_and_32_reserved_series_survive_byte_pruning(tmp_path):
    module = console()
    collector = module.Collector(tmp_path, JOB, measurements=full_recorder(module))
    for key in [*module.PREFERRED, *('r' * 120 + str(i) for i in range(28))]:
        for step in range(2000):
            collector.scalar(key, 1.23456789123456e100, step)
    collector.scalar('33rd-normal', 1, 0)
    collector.scalar('MeasurePhase/forged/state', 2, 0)
    collector.flush({}); drain(collector)
    raw = (tmp_path / 'console/metrics.json').read_bytes()
    body = metrics(tmp_path / 'console/metrics.json')
    assert len(raw) <= 1 << 20 and len(body) == 64
    assert EXPECTED_KEYS <= body.keys()
    assert all(len(body[key]) == 1 for key in EXPECTED_KEYS)
    assert all(0 < len(points) <= 2000 for key, points in body.items() if key not in EXPECTED_KEYS)
    lost = 64000 - sum(len(v) for k, v in body.items() if k not in EXPECTED_KEYS)
    assert body['Measurement/droppedPoints'][0]['value'] == lost + 2
    assert '33rd-normal' not in body and 'MeasurePhase/forged/state' not in body


def test_history_eviction_invalid_scalar_and_unmaterialized_tensor_loss(tmp_path):
    module = console()
    collector = module.Collector(tmp_path, JOB)
    class Tensor:
        def __float__(self): raise AssertionError('must never copy tensor scalar')
    for value in (float('nan'), float('inf'), True, Tensor()): collector.scalar('bad', value, 0)
    collector.scalar('bad', 1, -1)
    for i in range(2002): collector.scalar('reward', i, i)
    collector.flush({}); drain(collector)
    body = metrics(tmp_path / 'console/metrics.json')
    assert body['Measurement/droppedPoints'][0]['value'] == 7
    assert body['reward'][0] == {'step': 2, 'value': 2.}


def test_blocked_sink_coalesces_bounded_snapshots_and_close_is_partial(tmp_path):
    module = console()
    entered, release = threading.Event(), threading.Event()
    collector = module.Collector(tmp_path, JOB)
    original_write = collector._write
    writes = []
    def blocked(data):
        writes.append(data)
        entered.set()
        assert release.wait(3)
        original_write(data)
    collector._write = blocked
    try:
        collector.scalar('reward', 1, 0); collector.flush({})
        assert entered.wait(1)
        worker = collector._thread
        for i in range(1, 100):
            collector.scalar('reward', i, i)
            collector.flush({})
        assert collector._thread is worker and len(writes) == 1
        assert isinstance(collector._pending, tuple)
        assert len(collector._pending[1]) == 1
        assert len(collector._pending[1][0][1]) == 100
        assert collector.dropped == 0
        before = time.monotonic(); collector.close()
        assert time.monotonic() - before < .5
        assert collector.state == 3 and worker.is_alive()
        assert collector.measurements.snapshot(console_state=collector.state)[1]['Measurement/consoleState'] == 3
    finally:
        release.set(); drain(collector)
    assert len(writes) == 2
    assert metrics(tmp_path / 'console/metrics.json')['reward'][-1]['value'] == 99
    assert metrics(tmp_path / 'console/metrics.json')['Measurement/consoleState'][0]['value'] == 3
    assert collector.dropped == 0


def test_trylock_refusal_preserves_carried_history_and_prior_sink_state(tmp_path):
    module = console()
    collector = module.Collector(tmp_path, JOB)
    with collector._lock:
        collector.scalar('reward', 1, 0)
        collector.flush({})
        assert collector._pending is None and collector.dropped == 0
    collector.flush({}); drain(collector)
    body = metrics(tmp_path / 'console/metrics.json')
    assert body['Measurement/consoleState'] == [{'step': 0, 'value': 0}]
    assert collector.state == 1 and body['reward'][0]['value'] == 1


def test_sink_failure_preserves_last_valid_file_and_exposes_prior_error(tmp_path):
    module = console()
    collector = module.Collector(tmp_path, JOB)
    collector.scalar('reward', 1, 0)
    # Use a completed write event to establish the prior valid file deterministically.
    original = collector._write
    written = threading.Event()
    def write(data):
        original(data); written.set()
    collector._write = write
    collector.flush({}); assert written.wait(1)
    old = (tmp_path / 'console/metrics.json').read_bytes()
    failed = threading.Event()
    def fail(data):
        failed.set(); raise OSError('optional filesystem failure')
    collector._write = fail
    collector.scalar('reward', 2, 1); collector.flush({})
    assert failed.wait(1)
    drain(collector)
    assert collector.state == 2
    assert (tmp_path / 'console/metrics.json').read_bytes() == old
    assert collector.measurements.snapshot(console_state=collector.state)[1]['Measurement/consoleState'] == 2


def test_missing_terminal_logger_iteration_or_changed_config_omits_derived_counts():
    module = console()
    recorder = module.RunMeasurements({**JOB, 'iterations': 2})
    recorder.configure_iterations(7, 64); recorder.begin('training')
    recorder.completed_iteration(100, 100, steps=7, envs=64, writer=True)
    recorder.end('training')
    assert recorder.snapshot()[1]['MeasurePhase/training/state'] == 5
    assert 'MeasureTraining/completedTransitions' not in recorder.snapshot()[1]
    recorder.completed_iteration(101, 100, steps=None, envs=64, writer=True)
    assert recorder.snapshot()[0] == 2


def test_coalesced_byte_pruning_then_history_eviction_counts_each_lost_point_once(tmp_path):
    module = console()
    collector = module.Collector(tmp_path, JOB)
    # One long-key history fits at 2000 points. A tiny encoder cap deliberately
    # forces prefix pruning so its overlap with later ring eviction is testable.
    original = module.MAX_BYTES
    module.MAX_BYTES = 8000
    entered, release = threading.Event(), threading.Event()
    original_write = collector._write
    def blocked(data):
        entered.set(); assert release.wait(3); original_write(data)
    collector._write = blocked
    try:
        for i in range(2000): collector.scalar('reward', 1.23456789123456e100, i)
        collector.flush({}); assert entered.wait(1)
        for i in range(2000, 4100): collector.scalar('reward', 1.23456789123456e100, i)
        collector.flush({}); collector.flush({})
    finally:
        release.set(); drain(collector); module.MAX_BYTES = original
    body = metrics(tmp_path / 'console/metrics.json')
    assert body['Measurement/droppedPoints'][0]['value'] == 4100 - len(body['reward'])
    assert collector.dropped == 4100 - len(body['reward'])



class FaultClock:
    def __init__(self, *readings):
        self.readings = iter(readings)

    def __call__(self):
        reading = next(self.readings)
        if isinstance(reading, Exception): raise reading
        return reading


@pytest.mark.parametrize('operation_fails', [False, True])
def test_failed_begin_counts_entered_checkpoint_without_fabricated_zero(operation_fails):
    module = console()
    recorder = module.RunMeasurements(JOB, started=0., clock=FaultClock(
        OSError('begin unavailable'), 10., 11., 12., 14., 15.))
    original = RuntimeError('original operation failure')
    result = object()
    def operation():
        with recorder.phase('checkpoint'):
            if operation_fails: raise original
            return result
    if operation_fails:
        with pytest.raises(RuntimeError) as caught: operation()
        assert caught.value is original
    else:
        assert operation() is result
    values = recorder.snapshot()[1]
    assert values['MeasureCheckpoint/calls'] == 1
    assert values['MeasurePhase/checkpoint/state'] == 5
    assert 'MeasurePhase/checkpoint/seconds' not in values
    with recorder.phase('checkpoint'):
        pass
    values = recorder.snapshot()[1]
    assert values['MeasureCheckpoint/calls'] == 2
    assert values['MeasurePhase/checkpoint/state'] == 5
    assert values['MeasurePhase/checkpoint/seconds'] == 2.


@pytest.mark.parametrize('operation_fails', [False, True])
@pytest.mark.parametrize('prior', [False, True])
def test_failed_end_closes_span_retains_known_sum_and_original_operation(operation_fails, prior):
    module = console()
    readings = ([1., 3.] if prior else []) + [4., OSError('end unavailable'), 100., 200.]
    recorder = module.RunMeasurements(JOB, started=0., clock=FaultClock(*readings))
    if prior:
        with recorder.phase('checkpoint'):
            pass
    original = ValueError('original checkpoint exception')
    result = object()
    def operation():
        with recorder.phase('checkpoint'):
            if operation_fails: raise original
            return result
    if operation_fails:
        with pytest.raises(ValueError) as caught: operation()
        assert caught.value is original
    else:
        assert operation() is result
    for _ in range(2):
        values = recorder.snapshot()[1]
        assert values['MeasureCheckpoint/calls'] == (2 if prior else 1)
        assert values['MeasurePhase/checkpoint/state'] == 5
        if prior: assert values['MeasurePhase/checkpoint/seconds'] == 2.
        else: assert 'MeasurePhase/checkpoint/seconds' not in values


def test_failed_snapshot_clock_omits_unobserved_numeric_values_and_keeps_partial_sticky():
    module = console()
    recorder = module.RunMeasurements(JOB, started=0., clock=FaultClock(
        1., OSError('snapshot unavailable'), 3., 100.))
    recorder.begin('checkpoint')
    observed = module.diagnose(recorder, 'snapshot')
    assert observed is not None
    values = observed[1]
    assert values['MeasureCheckpoint/calls'] == 1
    assert values['MeasurePhase/checkpoint/state'] == 5
    assert 'MeasurePhase/checkpoint/seconds' not in values
    assert 'Measurement/snapshotElapsedSeconds' not in values
    recorder.end('checkpoint')
    values = recorder.snapshot()[1]
    assert values['MeasurePhase/checkpoint/state'] == 5
    assert values['MeasurePhase/checkpoint/seconds'] == 2.


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -1., True])
def test_invalid_clock_values_are_unknown_instead_of_zero_or_negative_residence(invalid):
    module = console()
    recorder = module.RunMeasurements(JOB, started=0., clock=FaultClock(invalid, 10., 11.))
    with recorder.phase('checkpoint'):
        pass
    values = recorder.snapshot()[1]
    assert values['MeasureCheckpoint/calls'] == 1
    assert values['MeasurePhase/checkpoint/state'] == 5
    assert 'MeasurePhase/checkpoint/seconds' not in values


def test_observer_close_failure_keeps_observed_matching_totals_and_failure_state():
    module = console()
    recorder = module.RunMeasurements(JOB)
    source = SimpleNamespace(_measurement_snapshot=lambda: {
        'MeasureObserver/state': 2, 'MeasureObserver/hookCalls': 2,
        'MeasureObserver/hookSeconds': 4.,
    })
    recorder._observer_datum(1, observer=source)
    recorder._observer_datum(3)
    values = recorder.snapshot()[1]
    assert values['MeasureObserver/state'] == 3
    assert values['MeasureObserver/hookCalls'] == 2
    assert values['MeasureObserver/hookSeconds'] == 4
    assert recorder.device == {'MeasureMemory/deviceStatus': 0}
    assert recorder.torch == {'MeasureMemory/torchStatus': 0}
    assert set(values) <= EXPECTED_KEYS


def test_observer_snapshot_error_is_optional_and_sticky_incomplete():
    module = console()
    recorder = module.RunMeasurements(JOB)
    queries = []
    def fail():
        queries.append(1)
        raise RuntimeError('optional snapshot')
    source = SimpleNamespace(_measurement_snapshot=fail)
    recorder._observer_datum(1, observer=source)
    for _ in range(2):
        values = recorder.snapshot()[1]
        assert values['MeasureObserver/state'] == 5
        assert 'MeasureObserver/hookCalls' not in values
        assert 'MeasureObserver/hookSeconds' not in values
    assert len(queries) == 1  # A broken source is detached rather than retried.


@pytest.mark.parametrize('failure_channel', ['datum', 'source'])
def test_known_observer_failure_survives_snapshot_error_and_source_detachment(failure_channel):
    module = console()
    recorder = module.RunMeasurements(JOB)
    queries = []
    broken = [False]
    def snapshot():
        queries.append(1)
        if broken[0]:
            raise OSError('optional snapshot after observed failure')
        return {'MeasureObserver/state': 3 if failure_channel == 'source' else 2,
                'MeasureObserver/hookCalls': 2, 'MeasureObserver/hookSeconds': 4.}
    source = SimpleNamespace(_measurement_snapshot=snapshot)
    recorder._observer_datum(1, observer=source)
    if failure_channel == 'datum':
        recorder._observer_datum(3)
    observed = recorder.snapshot()[1]
    assert observed['MeasureObserver/state'] == 3
    assert observed['MeasureObserver/hookCalls'] == 2
    assert observed['MeasureObserver/hookSeconds'] == 4
    broken[0] = True
    for _ in range(3):
        values = recorder.snapshot()[1]
        assert values['MeasureObserver/state'] == 3
        assert 'MeasureObserver/hookCalls' not in values
        assert 'MeasureObserver/hookSeconds' not in values
        broken[0] = False  # Detached sources must stay detached after recovery.
    assert len(queries) == 2


@pytest.mark.parametrize('failure_channel', ['datum', 'source'])
def test_known_observer_failure_survives_later_clean_source_snapshot(failure_channel):
    module = console()
    recorder = module.RunMeasurements(JOB)
    state = [3 if failure_channel == 'source' else 1]
    source = SimpleNamespace(_measurement_snapshot=lambda: {
        'MeasureObserver/state': state[0], 'MeasureObserver/hookCalls': 1,
        'MeasureObserver/hookSeconds': 2.,
    })
    recorder._observer_datum(1, observer=source)
    if failure_channel == 'datum':
        recorder._observer_datum(3)
    assert recorder.snapshot()[1]['MeasureObserver/state'] == 3
    state[0] = 2
    for _ in range(2):
        values = recorder.snapshot()[1]
        assert values['MeasureObserver/state'] == 3
        assert values['MeasureObserver/hookCalls'] == 1
        assert values['MeasureObserver/hookSeconds'] == 2
