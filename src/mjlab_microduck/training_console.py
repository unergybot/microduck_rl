"""Bounded, optional host diagnostics; no native/package imports or job authority."""
import json
import math
import os
import re
import secrets
import stat
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

MAX_BYTES = 1 << 20
_UNSET = object()
PHASES = ('initialization', 'training', 'checkpoint', 'onnx', 'cleanup', 'package')
PREFERRED = ('Perf/total_fps', 'Train/mean_reward', 'Train/mean_episode_length', 'Loss/value')
PREFIXES = ('MeasurePhase/', 'MeasureMemory/', 'MeasureCheckpoint/', 'MeasureTraining/',
            'MeasureObserver/', 'Measurement/')
DIAGNOSTIC_KEYS = frozenset(
    [f'MeasurePhase/{name}/{field}' for name in PHASES for field in ('seconds', 'state')]
    + ['MeasureCheckpoint/calls', 'MeasureMemory/deviceUsedSampledMaxBytes',
       'MeasureMemory/deviceSampleCount', 'MeasureMemory/deviceSampleFailures',
       'MeasureMemory/deviceFirstSampleSeconds', 'MeasureMemory/deviceLastSampleSeconds',
       'MeasureMemory/deviceStatus', 'MeasureMemory/torchAllocatedPeakBytes',
       'MeasureMemory/torchReservedPeakBytes', 'MeasureMemory/torchStatus',
       'MeasureTraining/completedTransitions', 'MeasureTraining/transitionsPerSecond',
       'MeasureObserver/hookCalls', 'MeasureObserver/hookSeconds', 'MeasureObserver/state',
       'Measurement/version', 'Measurement/snapshotElapsedSeconds', 'Measurement/terminalState',
       'Measurement/consoleState', 'Measurement/droppedPoints'])


def finite(value):
    # Never materialize a tensor/device scalar for optional diagnostics.
    if type(value) not in (int, float): return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError, OverflowError):
        return None


def diagnose(target, method, *args, **kwargs):
    """Contain our bookkeeping only; authoritative calls remain outside this guard."""
    try:
        if target is not None: return getattr(target, method)(*args, **kwargs)
    except Exception: pass  # noqa: BLE001, S110 -- optional observations


class RunMeasurements:
    """One invocation's host spans. Device snapshots have a single sampler owner."""
    def __init__(self, job, *, clock=time.monotonic, started=_UNSET):
        self.clock = clock
        self._last_clock = None
        self.started = self._read_clock() if started is _UNSET else finite(started)
        self._last_clock = self.started
        self._open = set()
        self._incomplete = set()
        self.num_envs = job.get('numEnvs')
        self.expected_iterations = job.get('iterations')
        # None means no measured contribution; a measured zero remains 0.0.
        self.phases = {name: [0, None, None] for name in PHASES}
        if started is not _UNSET:
            self._open.add('initialization')
            self.phases['initialization'] = [1 if self.started is not None else 5, None, self.started]
            if self.started is None: self._incomplete.add('initialization')
        self.calls = 0
        self.completed = 0
        self.next_iteration = None
        self.iteration_size = None
        self.coverage = False
        self.device = {'MeasureMemory/deviceStatus': 0}
        self.torch = {'MeasureMemory/torchStatus': 0}
        self.terminal = 0
        self._observer = None
        self._observer_state = 0

    def _read_clock(self):
        try: now = finite(self.clock())
        except Exception: return None
        if now is None or self._last_clock is not None and now < self._last_clock: return None
        self._last_clock = now
        return now

    def _partial(self, name):
        self._incomplete.add(name)
        self.phases[name][0] = 5
        if name == 'training': self.coverage = False

    def begin(self, name):
        phase = self.phases[name]
        # The wrapper was entered even when its optional timestamp is unavailable.
        if name == 'checkpoint': self.calls += 1
        self._open.add(name)
        phase[2] = self._read_clock()
        if phase[0] not in (3, 5): phase[0] = 1
        if phase[2] is None: self._partial(name)

    def end(self, name, state=2):
        phase = self.phases[name]
        active = phase[2]
        entered = name in self._open
        # End ownership before reading the clock: a failed read cannot keep the
        # returned operation active or charge unrelated later time to it.
        self._open.discard(name)
        phase[2] = None
        if entered:
            now = self._read_clock()
            if active is None or now is None or now < active:
                self._partial(name)
            else:
                phase[1] = (phase[1] if phase[1] is not None else 0.) + now - active
        elif state in (2, 3):
            self._partial(name)
        if name in self._incomplete: phase[0] = 5
        elif phase[0] != 3 or state == 3: phase[0] = state
        if (name == 'training' and state == 2 and type(self.expected_iterations) is int
                and self.completed != self.expected_iterations): self.partial_training()
        if name == 'training' and not self.coverage and phase[0] != 3: phase[0] = 5

    def fail_active(self):
        for name in ('initialization', 'onnx'):
            if name in self._open: self.end(name, 3)

    def terminal_written(self, success):
        self.terminal = 1 if success else 2

    def partial_training(self):
        self.coverage = False
        if self.phases['training'][0] not in (0, 3): self.phases['training'][0] = 5

    @contextmanager
    def phase(self, name):
        diagnose(self, 'begin', name)
        try:
            yield
        except BaseException:
            diagnose(self, 'end', name, 3)
            raise
        else:
            diagnose(self, 'end', name)

    def configure_iterations(self, steps, envs):
        self.coverage = (type(steps) is int and steps > 0 and type(envs) is int
                         and envs > 0 and envs == self.num_envs)
        self.iteration_size = steps * envs if self.coverage else None

    def completed_iteration(self, iteration, start, *, steps, envs, writer):
        if (not writer or type(iteration) is not int or type(start) is not int
                or iteration < 0 or self.next_iteration is not None and iteration < self.next_iteration):
            self.partial_training(); return
        expected = start if self.next_iteration is None else self.next_iteration
        self.completed += 1
        self.next_iteration = iteration + 1
        if (iteration != expected or type(steps) is not int or type(envs) is not int
                or self.iteration_size != steps * envs or envs != self.num_envs):
            self.partial_training()

    def read_torch(self, cuda, *, final=False):
        values = dict(self.torch)
        try:
            for key, query in (('torchAllocatedPeakBytes', cuda.max_memory_allocated),
                               ('torchReservedPeakBytes', cuda.max_memory_reserved)):
                value = query(0)
                if type(value) is not int or value < 0: raise ValueError('invalid allocator count')
                values['MeasureMemory/' + key] = value
            values['MeasureMemory/torchStatus'] = 2 if final else 1
        except Exception:
            values['MeasureMemory/torchStatus'] = 3
        self.torch = values

    def _observer_datum(self, state, *, observer=None):
        """Private handoff; producer/worker remain the owners of observer facts."""
        self._observer_state = state
        if state != 3 or observer is not None:
            self._observer = observer

    def _observer_snapshot(self):
        if self._observer is None:
            return {'MeasureObserver/state': self._observer_state}
        try:
            values = self._observer._measurement_snapshot()
            if values.get('MeasureObserver/state') == 3: self._observer_state = 3
            if self._observer_state == 3: values['MeasureObserver/state'] = 3
            return values
        except Exception:
            # Incomplete accounting cannot erase an already observed failure.
            if self._observer_state != 3: self._observer_state = 5
            self._observer = None
            return {'MeasureObserver/state': self._observer_state}

    def snapshot(self, *, console_state=0, dropped=0):
        now = self._read_clock()
        if now is None:
            for name in self._open: self._partial(name)
        values = {'Measurement/version': 1,
                  'Measurement/terminalState': self.terminal, 'Measurement/consoleState': console_state,
                  'Measurement/droppedPoints': dropped, **self._observer_snapshot(),
                  'MeasureCheckpoint/calls': self.calls, **self.device, **self.torch}
        if now is not None and self.started is not None:
            values['Measurement/snapshotElapsedSeconds'] = now - self.started
        for name, (state, elapsed, active) in self.phases.items():
            values[f'MeasurePhase/{name}/state'] = state
            if active is not None and now is not None:
                elapsed = (elapsed if elapsed is not None else 0.) + now - active
            if elapsed is not None:
                values[f'MeasurePhase/{name}/seconds'] = elapsed
        if self.coverage and self.phases['training'][0] != 0:
            transitions = self.completed * self.iteration_size
            values['MeasureTraining/completedTransitions'] = transitions
            seconds = values.get('MeasurePhase/training/seconds')
            if seconds is not None and seconds > 0:
                values['MeasureTraining/transitionsPerSecond'] = transitions / seconds
        return self.completed, {key: value for key, value in values.items() if finite(value) is not None}


class Collector:
    def __init__(self, directory, job, *, measurements=None):
        self.directory = Path(directory) / 'console'
        self.valid = all(isinstance(job.get(k), str) and re.fullmatch(pattern, job[k]) for k, pattern in
                         (('jobId', r'[-A-Za-z0-9_]{1,80}'), ('requestDigest', r'sha256:[a-f0-9]{64}')))
        self.job = {key: job[key] for key in ('jobId', 'requestDigest')} if self.valid else {}
        self.measurements = measurements if measurements is not None else RunMeasurements(job)
        self.series = {}
        self._progress = {key: None for key in
                          ('currentIteration', 'targetIterations', 'elapsedSeconds', 'samplesPerSecond')}
        self._sequence = {}; self._evicted = {}; self._pruned = {}
        self._rejected = 0
        self._lock = threading.Lock(); self._wake = threading.Event()
        self._pending = None; self._closing = False; self.state = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def dropped(self):
        return self._rejected + sum(max(self._evicted.get(k, 0), self._pruned.get(k, 0)) for k in tuple(self.series))

    def scalar(self, key, value, step):
        if (not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_./:-]{1,128}', key)
                or key.startswith(PREFIXES)):
            self._rejected += 1; return
        value = finite(value)
        if value is None or type(step) is not int or not 0 <= step <= 2147483647:
            self._rejected += 1; return
        if key not in self.series:
            if key not in PREFERRED and sum(k not in PREFERRED for k in self.series) >= 28:
                self._rejected += 1; return
            self.series[key] = deque(maxlen=2000)
            self._sequence[key] = 0
        if len(self.series[key]) == 2000: self._evicted[key] = self.series[key][0][0]
        self._sequence[key] += 1
        self.series[key].append((self._sequence[key], step, value))

    def flush(self, progress):
        if not self.valid: raise ValueError('console metadata exceeds diagnostic budget')
        summary = diagnose(self.measurements, 'snapshot', console_state=self.state, dropped=self.dropped)
        step, values = summary if summary is not None else (0, {})
        # Immutable bounded scalar copies only: no encoding, probes or filesystem here.
        histories = tuple((key, tuple(point for point in points if point[0] > self._pruned.get(key, 0)))
                          for key, points in self.series.items())
        for key in self._progress:
            if key in progress: self._progress[key] = finite(progress[key])
        snapshot = (tuple(self._progress.items()),
                    histories, step, tuple(values.items()), self._rejected, tuple(self._evicted.items()))
        if not self._lock.acquire(blocking=False): return
        try: self._pending = snapshot
        finally: self._lock.release()
        self._wake.set()

    def _encode(self, snapshot):
        progress, histories, step, diagnostics, rejected, evictions = snapshot
        evictions = dict(evictions)
        normal = [(key, [p for p in points if p[0] > self._pruned.get(key, 0)]) for key, points in histories]
        values = dict(diagnostics)
        # Merge disjoint loss watermarks; carrying the same history never counts twice.
        while True:
            values['Measurement/droppedPoints'] = rejected + sum(max(evictions.get(key, 0), self._pruned.get(key, 0)) for key, _ in histories)
            values['Measurement/consoleState'] = self.state
            body = {**self.job, 'progress': dict(progress), 'metrics':
                    [{'key': key, 'points': [{'step': p[1], 'value': p[2]} for p in points]} for key, points in normal]
                    + [{'key': key, 'points': [{'step': step, 'value': value}]} for key, value in values.items()]}
            data = json.dumps(body, allow_nan=False, separators=(',', ':')).encode()
            if len(data) <= MAX_BYTES: return data
            if not any(points for _, points in normal):
                raise ValueError('console metadata exceeds diagnostic budget')
            for key, points in normal:
                remove = (len(points)+1)//2
                if remove:
                    self._pruned[key] = max(self._pruned.get(key, 0), points[remove-1][0])
                    del points[:remove]

    def _write(self, data):
        self.directory.mkdir(mode=0o700, exist_ok=True)
        info = self.directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise OSError('console directory unavailable')
        name = self.directory / ('.metrics-' + secrets.token_hex(8))
        fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            os.replace(name, self.directory / 'metrics.json')
        finally:
            name.unlink(missing_ok=True)

    def _run(self):
        while True:
            self._wake.wait()
            with self._lock:
                snapshot = self._pending; self._pending = None; self._wake.clear()
                closing = self._closing
            if snapshot is None:
                if closing: return
                continue
            try:
                self._write(self._encode(snapshot))
                if self.state not in (2, 3): self.state = 1
            except Exception:
                if self.state != 3: self.state = 2
            if self._closing: self._wake.set()

    def close(self, timeout=.1):
        self._closing = True; self._wake.set()
        self._thread.join(timeout=min(.1, max(0., timeout)))
        if self._thread.is_alive(): self.state = 3


def observe_runner(runner, directory, job, *, collector=None, measurements=None):
    """Original logger/writer execute first. Only successful log returns count."""
    logger = runner.logger
    initialize, log = logger.init_logging_writer, logger.log
    collector = collector if collector is not None else Collector(directory, job, measurements=measurements)
    measurements = measurements if measurements is not None else collector.measurements
    # Only this pinned logger's full per-iteration callback is proven here.
    pinned = type(logger).__module__ == 'rsl_rl.utils.logger' and type(logger).__name__ == 'Logger'
    try:
        proven = (pinned and runner.cfg['num_steps_per_env'] == logger.cfg['num_steps_per_env']
                  and runner.env.num_envs == logger.num_envs)
        if proven: measurements.configure_iterations(logger.cfg['num_steps_per_env'], logger.num_envs)
        else: measurements.partial_training()
    except Exception:
        proven = False
        diagnose(measurements, 'partial_training')

    def init_writer(*args, **kwargs):
        result = initialize(*args, **kwargs)
        try: attach_writer()
        except Exception: pass  # noqa: BLE001, S110
        return result

    def attach_writer():
        if logger.writer is not None:
            add = logger.writer.add_scalar
            def scalar(tag, value, global_step=None, *args, **kwargs):
                result = add(tag, value, global_step, *args, **kwargs)
                diagnose(collector, 'scalar', tag, value, global_step)
                return result
            logger.writer.add_scalar = scalar

    def observed_log(*args, **kwargs):
        try: result = log(*args, **kwargs)
        except BaseException:
            diagnose(measurements, 'partial_training')
            raise
        try:
            if pinned:
                it = kwargs.get('it', args[0] if args else None)
                start = kwargs.get('start_it', args[1] if len(args) > 1 else None)
                measurements.completed_iteration(it, start, steps=logger.cfg['num_steps_per_env'],
                                                 envs=logger.num_envs, writer=logger.writer is not None)
            fps = collector.series.get('Perf/total_fps')
            collector.flush({'currentIteration': kwargs.get('it', args[0] if args else None),
                             'targetIterations': kwargs.get('total_it', args[2] if len(args) > 2 else None),
                             'elapsedSeconds': logger.tot_time,
                             'samplesPerSecond': fps[-1][2] if fps else None})
        except Exception: pass  # noqa: BLE001, S110
        return result

    logger.init_logging_writer, logger.log = init_writer, observed_log
    return collector
