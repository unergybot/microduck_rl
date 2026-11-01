"""Bounded diagnostics tapped from rsl-rl 5.0.1's actual logger, never PPO state."""
import json
import math
import os
import re
import secrets
import stat
from collections import deque
from pathlib import Path

MAX_BYTES = 2 << 20


def finite(value):
    if isinstance(value, bool): return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError, OverflowError):
        return None


class Collector:
    def __init__(self, directory, job):
        self.directory = Path(directory) / 'console'
        self.job = {'jobId': job['jobId'], 'requestDigest': job['requestDigest']}
        self.series = {}

    def scalar(self, key, value, step):
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_./:-]{1,128}', key): return
        value = finite(value)
        if value is None or type(step) is not int or not 0 <= step <= 2147483647: return
        if key not in self.series:
            if len(self.series) >= 64: return
            self.series[key] = deque(maxlen=2000)
        self.series[key].append({'step': step, 'value': value})

    def flush(self, progress):
        body = {**self.job, 'progress': {key: finite(progress.get(key)) for key in
                ('currentIteration', 'targetIterations', 'elapsedSeconds', 'samplesPerSecond')},
                'metrics': [{'key': key, 'points': list(points)} for key, points in self.series.items()]}
        while True:
            data = json.dumps(body, allow_nan=False, separators=(',', ':')).encode()
            if len(data) <= MAX_BYTES: break
            # Trim all histories proportionally in one pass; repeatedly encoding
            # the longest series would stall a fully populated 64-series logger.
            ratio = MAX_BYTES / len(data) * .95
            if not any(metric['points'] for metric in body['metrics']):
                raise ValueError('console metadata exceeds diagnostic budget')
            for metric in body['metrics']:
                # Halving guarantees progress even for one point. At most 12
                # encodings for the bounded 2000-point histories, then reject
                # metadata that cannot fit without changing job identity.
                keep = min(len(metric['points']) // 2, int(len(metric['points']) * ratio))
                metric['points'] = metric['points'][-keep:] if keep else []
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


def observe_runner(runner, directory, job):
    """Attach after runner construction, before learn initializes its writer.

    Original logger and writer calls execute first and keep their return/exception
    semantics. Observation I/O failure is intentionally unable to fail training.
    """
    collector = Collector(directory, job)
    logger = runner.logger
    initialize, log = logger.init_logging_writer, logger.log

    def init_writer(*args, **kwargs):
        result = initialize(*args, **kwargs)
        try:
            attach_writer()
        except Exception: pass  # noqa: BLE001, S110 - optional writer interception
        return result

    def attach_writer():
        if logger.writer is not None:
            add = logger.writer.add_scalar
            def scalar(tag, value, global_step=None, *args, **kwargs):
                result = add(tag, value, global_step, *args, **kwargs)
                try: collector.scalar(tag, value, global_step)
                except Exception: pass  # noqa: BLE001, S110 - diagnostics cannot change runner semantics
                return result
            logger.writer.add_scalar = scalar

    def observed_log(*args, **kwargs):
        result = log(*args, **kwargs)
        try:
            fps = collector.series.get('Perf/total_fps')
            collector.flush({'currentIteration': kwargs.get('it'), 'targetIterations': kwargs.get('total_it'),
                             'elapsedSeconds': logger.tot_time,
                             'samplesPerSecond': fps[-1]['value'] if fps else None})
        except Exception: pass  # noqa: BLE001, S110 - diagnostics cannot change runner semantics
        return result

    logger.init_logging_writer, logger.log = init_writer, observed_log
    return collector
