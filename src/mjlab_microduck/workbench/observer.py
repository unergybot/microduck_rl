"""Optional actual-environment snapshots, one pending frame, no control authority."""
import hashlib
import json
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


def attach_observer(env,directory,job,spec):
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
