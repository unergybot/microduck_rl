"""Optional actual-environment snapshots, one pending frame, no control authority."""
import json
import math
import os
import threading
import time
from pathlib import Path
from .configuration import canonical_json


class LatestObserver:
    def __init__(self,env,directory,job,spec,*,clock=time.monotonic,sink=None):
        self.env=env;self.clock=clock;self.next_at=0.;self.sequence=0;self.pending=None;self.closed=False
        self.condition=threading.Condition();self.directory=Path(directory)/'observer'
        self.identity=dict(jobId=job['jobId'],requestDigest=job['requestDigest'],configDigest=spec['configDigest'],robotProfileId=spec['robotProfileId'],modelDigest=job['limits']['modelDigest'])
        self.count=min(9,env.num_envs);self.sink=sink or self.write
        self.thread=threading.Thread(target=self.consume,daemon=True);self.thread.start()

    def write(self,frame):
        from mjlab_microduck.managed_training import atomic_json
        self.directory.mkdir(exist_ok=True)
        atomic_json(self.directory/'latest.json',frame)

    def sample(self):
        now=self.clock()
        if self.closed or now<self.next_at:return
        self.next_at=now+.05
        # Avoid waiting for observer worker or disk. GPU transfer synchronization is
        # owned by the worker; producer enqueues copies from tiny cloned snapshots.
        if not self.condition.acquire(blocking=False):return
        try:
            import torch
            data=self.env.sim.data
            positions=data.geom_xpos[:self.count];matrices=data.geom_xmat[:self.count]
            payload=[];event=None
            if isinstance(positions,torch.Tensor) and positions.is_cuda:
                stream=torch.cuda.current_stream(positions.device)
                for value in (positions,matrices):
                    target=torch.empty_like(value,device='cpu',pin_memory=True)
                    target.copy_(value.clone(),non_blocking=True);payload.append(target)
                event=torch.cuda.Event();event.record(stream)
            else:
                for value in (positions,matrices):payload.append(value.clone() if hasattr(value,'clone') else value.copy())
            self.pending=(now,time.time(),self.sequence,payload,event,float(self.env.step_dt*self.env.common_step_counter))
            self.sequence+=1;self.condition.notify()
        finally:self.condition.release()

    def consume(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda:self.pending is not None or self.closed)
                if self.closed:return
                now,wall,seq,payload,event,simtime=self.pending;self.pending=None
            try:
                if event:event.synchronize()
                frame=dict(schemaVersion='ROM_TRAINING_OBSERVER_FRAME_V1',**self.identity,sequence=seq,capturedAtUnixSeconds=wall,monotonicSeconds=now,simulationSeconds=simtime,environmentIndices=list(range(self.count)),positions=payload[0].reshape(self.count,-1).tolist(),matrices=payload[1].reshape(self.count,-1).tolist())
                if len(canonical_json(frame).encode())>262144:continue
                self.sink(frame)
            except Exception:pass  # optional diagnostics cannot change PPO

    def close(self):
        with self.condition:self.closed=True;self.pending=None;self.condition.notify()
        self.thread.join(timeout=.1)


def attach_observer(env,directory,job,spec,*,version=1,measurements=None):
    try:
        observer = _attach_observer(env, directory, job, spec, version=version)
    except Exception:
        try:
            if measurements is not None: measurements._observer_datum(3)
        except Exception: pass  # Optional publication cannot mask the attach failure.
        raise
    if version == 2 and measurements is not None:
        observer._hook_instrumented = True
    try:
        if measurements is not None:
            measurements._observer_datum(1 if version == 2 else 0,
                                         observer=observer if version == 2 else None)
    except Exception: pass  # Optional publication cannot change the installed hook.
    return observer


def _attach_observer(env,directory,job,spec,*,version=1):
    if type(version) is not int or version not in (1, 2):
        raise ValueError('OBSERVER_VERSION_UNSUPPORTED')
    if version == 2:
        step = env.step
        observer = ProjectedObserver(env, directory, job, spec, _start=False)
        def projected_step(*args, **kwargs):
            result = step(*args, **kwargs)
            if observer._hook_instrumented:
                observer._measured_sample()
            else:
                observer.sample()  # Optional producer contains its own failures.
            return result
        # Install the hook before starting a worker that could block on disk.
        env.step = projected_step
        try:
            observer.thread.start()
        except Exception:
            env.step = step
            observer.close()
            raise
        return observer
    from mjlab_microduck.rom.viewer import export_geometry,export_meshes
    from mjlab_microduck.managed_training import atomic_json
    model=env.sim.mj_model
    if model.ngeom>256 or model.nmeshvert>1000000 or model.nmeshface>2000000:raise ValueError('observer model exceeds bound')
    geometry=export_geometry(model)
    body=dict(schemaVersion='ROM_TRAINING_OBSERVER_GEOMETRY_V1',geometry=geometry,meshes=export_meshes(model,geometry),modelDigest=job['limits']['modelDigest'],configDigest=spec['configDigest'],jobId=job['jobId'],requestDigest=job['requestDigest'])
    if len(canonical_json(body).encode())>64*1024*1024:raise ValueError('observer geometry exceeds bound')
    observer=LatestObserver(env,directory,job,spec)
    observer.directory.mkdir(exist_ok=True);atomic_json(observer.directory/'geometry.json',body)
    step=env.step
    def observed_step(*args,**kwargs):
        result=step(*args,**kwargs)
        try:observer.sample()
        except Exception:pass
        return result
    env.step=observed_step
    return observer



def _encode_bounded(value, limit, code):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if len(encoded) > limit:
        raise ValueError(code)
    return encoded


class _SelectedTransfer:
    """One reusable host slot; fresh selected CUDA tensors live until completion.

    Allocation and device-index creation occur at attach, never under the worker
    lock on PPO. Both source axes are selected in one gather. In a partial-copy
    failure the worker fences the recorded stream before releasing any storage.
    """

    def __init__(self, positions, matrices, count, sources):
        import numpy as np
        import torch

        self.shapes = (tuple(positions.shape), tuple(matrices.shape))
        self.torch = isinstance(positions, torch.Tensor)
        self.device = positions.device if self.torch else None
        self.selected = []
        self.stream = None
        self.recorded = False
        self.event = None
        self._validate(positions, matrices)
        shapes = ((count, len(sources), 3), (count, len(sources), 3, 3))
        if self.torch:
            self.env_index = torch.arange(count, device=self.device)[:, None]
            self.source_index = torch.tensor(
                sources, dtype=torch.long, device=self.device
            )
            self.buffers = [
                torch.empty(
                    shape,
                    dtype=torch.float32,
                    device="cpu",
                    pin_memory=positions.is_cuda,
                )
                for shape in shapes
            ]
            if positions.is_cuda:
                self.event = torch.cuda.Event()
        else:
            self.env_index = np.arange(count)[:, None]
            self.source_index = np.asarray(sources, dtype=np.intp)
            self.buffers = [np.empty(shape, dtype=np.float32) for shape in shapes]

    def _validate(self, positions, matrices):
        import numpy as np
        import torch

        if (tuple(positions.shape), tuple(matrices.shape)) != self.shapes:
            raise ValueError("POSE_SHAPE")
        if self.torch:
            valid = all(
                isinstance(a, torch.Tensor)
                and a.device == self.device
                and a.is_floating_point()
                for a in (positions, matrices)
            )
        else:
            valid = all(
                isinstance(a, np.ndarray) and a.dtype.kind == "f"
                for a in (positions, matrices)
            )
        if not valid:
            raise ValueError("POSE_TYPE")

    def capture(self, positions, matrices):
        import torch

        self._validate(positions, matrices)
        self.recorded = False
        self.selected = []
        if self.event is not None:
            self.stream = torch.cuda.current_stream(self.device)
        for source, target in zip((positions, matrices), self.buffers):
            value = source[self.env_index, self.source_index]
            if self.torch:
                value = value.detach()
            self.selected.append(value)
            if self.torch:
                target.copy_(value, non_blocking=self.event is not None)
            else:
                target[...] = value
        if self.event is not None:
            self.event.record(self.stream)
            self.recorded = True

    def wait(self):
        # Worker only. Never free/reuse a slot if either fence raises.
        if self.recorded:
            self.event.synchronize()
        elif self.stream is not None:
            self.stream.synchronize()
        self.selected = []


class ProjectedObserver:
    """Opt-in V2 with TWO total slots across copy, pending, packing and IO.

    Slots transition idle -> queued -> working -> idle only after completion.
    Nothing replaces queued/working copies. Stop drains owned transfers on the
    worker; a failed fence quarantines its slot. Close never fences or performs
    IO and joins for at most 0.1 s. A permanently blocked sink/fence may retain
    this daemon and its two bounded slots, but cannot block the training lane.
    """

    def __init__(
        self,
        env,
        directory,
        job,
        spec,
        *,
        clock=time.monotonic,
        wall_clock=time.time,
        sink=None,
        geometry_sink=None,
        _transfer_factory=_SelectedTransfer,
        _start=True,
    ):
        from .geometry import MAX_FRAME_BYTES, MAX_GEOMETRY_BYTES, build_projection

        self.env = env
        self.clock = clock
        self._hook_clock = time.monotonic
        self._last_hook_clock = None
        self._hook_instrumented = False
        self._hooks = (0, 0.)  # One immutable, matching count/time producer snapshot.
        self._producer_failed = False
        self._worker_failed = False
        self._hook_partial = False
        self._close_partial = False
        self._closed = False
        self.wall_clock = wall_clock
        self.next_at = float("-inf")
        self.sequence = 0
        self.stopped = threading.Event()
        self.condition = threading.Condition()
        self.queue = []
        self.directory = Path(directory) / "observer"
        self.sink = sink
        self.geometry_sink = geometry_sink
        self.identity = {
            "jobId": job["jobId"],
            "requestDigest": job["requestDigest"],
            "configDigest": spec["configDigest"],
            "robotProfileId": spec["robotProfileId"],
            "modelDigest": job["limits"]["modelDigest"],
        }
        _encode_bounded(self.identity, 2048, "OBSERVER_IDENTITY_BUDGET")
        if type(env.num_envs) is not int or env.num_envs < 1:
            raise ValueError("ENVIRONMENT_INDICES")
        self.count = min(9, env.num_envs)
        self.environment_count = env.num_envs
        self.environments = tuple(range(self.count))
        self.projection = build_projection(
            env.sim.mj_model,
            robot_root_body_id=env.scene.entities["robot"].indexing.root_body_id,
        )
        geometry = dict(
            schemaVersion="ROM_TRAINING_OBSERVER_GEOMETRY_V2",
            **{k: v for k, v in self.identity.items() if k != "robotProfileId"},
            **self.projection.export(),
        )
        self.geometry_bytes = _encode_bounded(
            geometry, MAX_GEOMETRY_BYTES, "GEOMETRY_BYTES"
        )
        slots = len(self.projection.source_membership)
        if 4 * ((self.count * slots * 12 * 4 + 2) // 3) + 4096 > MAX_FRAME_BYTES:
            raise ValueError("FRAME_BUDGET")
        positions, matrices = env.sim.data.geom_xpos, env.sim.data.geom_xmat
        source_count = self.projection.source_geom_count
        if tuple(positions.shape) != (env.num_envs, source_count, 3) or tuple(
            matrices.shape
        ) != (env.num_envs, source_count, 3, 3):
            raise ValueError("POSE_SHAPE")
        sources = tuple(g for _, g in self.projection.dynamic_mapping)
        # All potentially failing preflight/allocation precedes thread start.
        self.slots = [
            {
                "transfer": _transfer_factory(positions, matrices, self.count, sources),
                "state": "idle",
            }
            for _ in range(2)
        ]
        self.thread = threading.Thread(target=self.consume, daemon=True)
        if _start:
            self.thread.start()

    def _write(self, filename, encoded):
        self.directory.mkdir(exist_ok=True)
        path = self.directory / filename
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _read_hook_clock(self):
        try:
            value = self._hook_clock()
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or self._last_hook_clock is not None and value < self._last_hook_clock):
                return None
            self._last_hook_clock = value
            return value
        except Exception:
            return None

    def _measured_sample(self):
        # Completed-hook host residence only: begins AFTER the original step.
        # Excludes worker GPU waits/packing/IO and final datum publication.
        start = self._read_hook_clock()
        try:
            self.sample()
        except Exception:
            self._producer_failed = True
            self.stopped.set()
            return
        end = self._read_hook_clock()
        if start is None or end is None:
            self._hook_partial = True
            return
        calls, seconds = self._hooks
        total = seconds + (end - start)
        if not math.isfinite(total):
            self._hook_partial = True
            return
        self._hooks = (calls + 1, total)

    def _measurement_snapshot(self):
        calls, seconds = self._hooks
        state = (3 if self._producer_failed or self._worker_failed else
                 5 if self._hook_partial or self._close_partial else
                 2 if self._closed and not self.thread.is_alive() else 1)
        values = {'MeasureObserver/state': state}
        if calls or not self._hook_partial:
            values.update({'MeasureObserver/hookCalls': calls,
                           'MeasureObserver/hookSeconds': seconds})
        return values

    def sample(self):
        # Even allocation/copy/clock errors are optional; the step result survives.
        try:
            now = self.clock()
            if self.stopped.is_set() or now < self.next_at:
                return
            if not self.condition.acquire(blocking=False):
                return
            try:
                if self.stopped.is_set():
                    return
                slot = next((s for s in self.slots if s["state"] == "idle"), None)
                if slot is None:
                    return
                wall = self.wall_clock()
                simtime = float(self.env.step_dt * self.env.common_step_counter)
                slot["metadata"] = {
                    "sequence": self.sequence,
                    "capturedAtUnixSeconds": wall,
                    "monotonicSeconds": now,
                    "simulationSeconds": simtime,
                }
                # Queue ownership BEFORE any copy can be submitted. The worker
                # cannot touch this slot until this short nonblocking lock exits.
                slot["state"] = "queued"
                self.queue.append(slot)
                self.next_at = now + 0.05
                try:
                    data = self.env.sim.data
                    slot["transfer"].capture(data.geom_xpos, data.geom_xmat)
                    self.sequence += 1
                except Exception:  # noqa: BLE001 -- Optional observer must never fail PPO.
                    self._producer_failed = True
                    self.stopped.set()
                self.condition.notify()
            finally:
                self.condition.release()
        except Exception:  # noqa: BLE001 -- Optional observer must never fail PPO.
            self._producer_failed = True
            self.stopped.set()

    def consume(self):
        from .geometry import MAX_FRAME_BYTES

        try:
            if self.geometry_sink is None:
                self._write("geometry.json", self.geometry_bytes)
            else:
                self.geometry_sink(json.loads(self.geometry_bytes))
        except Exception:  # noqa: BLE001 -- Optional observer must never fail PPO.
            self._worker_failed = True
            self.stopped.set()
        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda: self.queue or self.stopped.is_set(), timeout=0.05
                )
                if not self.queue:
                    if self.stopped.is_set():
                        return
                    continue
                slot = self.queue.pop(0)
                slot["state"] = "working"
            completed = False
            try:
                slot["transfer"].wait()
                completed = True
                if not self.stopped.is_set():
                    packed = self.projection.pack_selected(
                        *slot["transfer"].buffers,
                        environment_indices=self.environments,
                        environment_count=self.environment_count,
                    )
                    frame = dict(
                        schemaVersion="ROM_TRAINING_OBSERVER_FRAME_V2",
                        **self.identity,
                        **slot["metadata"],
                        **packed,
                    )
                    encoded = _encode_bounded(frame, MAX_FRAME_BYTES, "FRAME_BUDGET")
                    if self.sink is None:
                        self._write("latest.json", encoded)
                    else:
                        self.sink(frame)
            except Exception:  # noqa: BLE001 -- Optional observer must never fail PPO.
                self._worker_failed = True
                self.stopped.set()
            finally:
                # Release completed encodings before admitting reuse of this slot.
                packed = frame = encoded = None
                with self.condition:
                    slot["state"] = "idle" if completed else "quarantined"

    def close(self):
        self.stopped.set()
        # No condition lock or GPU fence on caller. Worker polls stop if idle.
        if self.thread.ident is not None:
            self.thread.join(timeout=0.1)
        self._closed = True
        if self.thread.is_alive(): self._close_partial = True
