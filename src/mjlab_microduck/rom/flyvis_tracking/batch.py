"""Independent local rollout processes with bounded worker count and owned runtimes."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.util import Finalize

_world = _vision = _readout = _readout_path = None


def _initialize(bundle, model, kind):
    global _world, _vision
    from .simulation import TrackingWorld
    from .vision import FlyvisVision, RetinaVision

    _vision = FlyvisVision(model) if kind == "flyvis" else RetinaVision()
    _world = TrackingWorld(bundle)
    Finalize(_world, _world.close, exitpriority=10)


def _execute(job):
    global _readout, _readout_path
    from .experiment import run_episode
    from .learning import Readout

    options = dict(job)
    path = options.pop("readout_path", None)
    if path != _readout_path:
        _readout = Readout.load(path) if path else None
        _readout_path = path
    spec = options.pop("spec")
    return run_episode(_world, _vision, spec, readout=_readout, **options)


class RolloutPool:
    def __init__(self, bundle, model, kind, jobs):
        if not 1 <= jobs <= 8:
            raise ValueError("jobs must be between 1 and 8")
        self.executor = ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
            initargs=(bundle, model, kind),
        )

    def run(self, jobs):
        return self.executor.map(_execute, jobs, chunksize=1)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.executor.shutdown(wait=True, cancel_futures=True)
