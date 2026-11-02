"""Projected observer CPU contracts; fake transfers are not native CUDA proof."""

import base64
import copy
import hashlib
import json
import threading
import time
from importlib.resources import files
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from mjlab_microduck import managed_training as managed
from mjlab_microduck.workbench import observer as observers
from mjlab_microduck.workbench.geometry import build_projection

JOB = {
    "jobId": "test-job",
    "requestDigest": "sha256:" + "a" * 64,
    "limits": {"modelDigest": "sha256:" + "b" * 64},
}
SPEC = {"configDigest": "sha256:" + "c" * 64, "robotProfileId": "rom-xl330-v1"}
PINS = (
    "workbench/observer.py",
    "workbench/geometry.py",
    "workbench/checkpoints.py",
    "workbench/engine.py",
    "workbench/resources/request.schema.json",
    "managed_training.py",
    "rom/viewer.py",
)
CHANNELS = [
    "OBSERVER_FRAME",
    "MODEL_GEOMETRY",
    "CHECKPOINT_INDEX",
    "ENGINE_CONFIGURATION",
    "EXPERIMENT_ARCHIVE",
]


def release():
    return {
        "trainingWorkbenchReadV2": {
            "schemaVersion": "ROM_TRAINING_WORKBENCH_READ_CAPABILITY_V2",
            "channels": CHANNELS.copy(),
            "producerSha256": {
                p: "sha256:"
                + hashlib.sha256(
                    files("mjlab_microduck").joinpath(p).read_bytes()
                ).hexdigest()
                for p in PINS
            },
        }
    }


@pytest.fixture
def env():
    # Compiled synthetic many-source scene; stub indexing is trusted by this test,
    # not evidence of a native training Env or nine simulated environments.
    boxes = "".join(
        f'<geom type="box" size=".01 .01 .02" pos="{i % 20 * 0.1 + 0.1} {i // 20 * 0.1 + 0.1} 0" rgba=".2 .4 .6 1"/>'
        for i in range(300)
    )
    model = mujoco.MjModel.from_xml_string(f"""<mujoco><worldbody>{boxes}
      <body name="robot"><freejoint/><geom size=".1"/></body>
      <body name="object"><freejoint/><geom size=".2"/></body>
      </worldbody></mujoco>""")
    positions = np.arange(12 * model.ngeom * 3, dtype=np.float32).reshape(
        12, model.ngeom, 3
    )
    matrices = np.tile(np.eye(3, dtype=np.float32), (12, model.ngeom, 1, 1))
    value = SimpleNamespace(
        num_envs=12,
        step_dt=0.02,
        common_step_counter=0,
        scene=SimpleNamespace(
            entities={
                "robot": SimpleNamespace(
                    indexing=SimpleNamespace(root_body_id=model.body("robot").id)
                )
            }
        ),
        sim=SimpleNamespace(
            mj_model=model,
            data=SimpleNamespace(geom_xpos=positions, geom_xmat=matrices),
        ),
    )
    value.result = object()

    def step(_):
        value.common_step_counter += 1
        return value.result

    value.step = step
    return value


def arrays(frame):
    n, s = len(frame["environmentIndices"]), frame["slotCount"]
    return (
        np.frombuffer(
            base64.b64decode(frame["positionsBase64"], validate=True), "<f4"
        ).reshape(n, s, 3),
        np.frombuffer(
            base64.b64decode(frame["matricesBase64"], validate=True), "<f4"
        ).reshape(n, s, 3, 3),
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_v2_attach_real_projection_and_frame(env, tmp_path, backend):
    old = env.sim.data.geom_xpos.copy()
    if backend == "torch":
        env.sim.data.geom_xpos = torch.from_numpy(env.sim.data.geom_xpos)
        env.sim.data.geom_xmat = torch.from_numpy(env.sim.data.geom_xmat)
    observer = observers.attach_observer(env, tmp_path, JOB, SPEC, version=2)
    try:
        assert env.step(None) is env.result
        assert env.common_step_counter == 1
        deadline = time.monotonic() + 2
        while (
            not (tmp_path / "observer/latest.json").exists()
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        geometry = json.loads((tmp_path / "observer/geometry.json").read_text())
        frame = json.loads((tmp_path / "observer/latest.json").read_text())
        expected = build_projection(
            env.sim.mj_model,
            robot_root_body_id=env.scene.entities["robot"].indexing.root_body_id,
        )
        for key, value in expected.export().items():
            assert geometry[key] == value
        assert geometry["schemaVersion"] == "ROM_TRAINING_OBSERVER_GEOMETRY_V2"
        assert frame["schemaVersion"] == "ROM_TRAINING_OBSERVER_FRAME_V2"
        assert "positions" not in frame and "matrices" not in frame
        assert frame["environmentIndices"] == list(range(9))
        assert frame["mappingDigest"] == expected.mapping_digest
        pos, mat = arrays(frame)
        for slot, source in expected.dynamic_mapping:
            np.testing.assert_array_equal(pos[:, slot], old[:9, source])
            np.testing.assert_array_equal(
                mat[:, slot], env.sim.data.geom_xmat[:9, source]
            )
        np.testing.assert_array_equal(env.sim.data.geom_xpos, old)
        assert len((tmp_path / "observer/latest.json").read_bytes()) <= 262144
    finally:
        observer.close()


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_selected_pack_matches_capture(env, backend):
    p = build_projection(env.sim.mj_model, robot_root_body_id=1)
    ids = [11, 9, 7, 5, 3, 1, 0, 2, 4]
    src = [g for _, g in p.dynamic_mapping]
    selected = [
        a[np.asarray(ids)[:, None], src]
        for a in (env.sim.data.geom_xpos, env.sim.data.geom_xmat)
    ]
    if backend == "torch":
        selected = [torch.from_numpy(a) for a in selected]
    assert p.pack_selected(
        *selected, environment_indices=ids, environment_count=12
    ) == p.capture(
        env.sim.data.geom_xpos, env.sim.data.geom_xmat, environment_indices=ids
    )


def test_image_proof_is_exact_and_legacy_default():
    assert managed.observer_version({}) == 1
    assert managed.observer_version(release()) == 2
    bad = release()
    bad["trainingWorkbenchReadV1"] = {}
    assert managed.observer_version(bad) == 0


@pytest.mark.parametrize("version", [True, False, 0, 3, "2", 2.0, None])
def test_unknown_version_refuses_before_hook(env, tmp_path, version):
    step = env.step
    with pytest.raises(ValueError, match="OBSERVER_VERSION_UNSUPPORTED"):
        observers.attach_observer(env, tmp_path, JOB, SPEC, version=version)
    assert env.step is step


@pytest.mark.parametrize(
    "mutation",
    [
        "null",
        "list",
        "field",
        "extra_field",
        "channels",
        "reorder",
        "missing_pin",
        "extra_path",
        "traversal",
        "hash",
        "uppercase",
        "hash_type",
        "schema",
        "ambiguous",
    ],
)
def test_invalid_image_proof_disables_without_fallback(mutation):
    value = release()
    descriptor = value["trainingWorkbenchReadV2"]
    if mutation == "null":
        value["trainingWorkbenchReadV2"] = None
    elif mutation == "list":
        value["trainingWorkbenchReadV2"] = []
    elif mutation == "field":
        descriptor.pop("producerSha256")
    elif mutation == "extra_field":
        descriptor["optional"] = True
    elif mutation == "channels":
        descriptor["channels"].pop()
    elif mutation == "reorder":
        descriptor["channels"].reverse()
    elif mutation == "missing_pin":
        descriptor["producerSha256"].pop(PINS[0])
    elif mutation == "extra_path":
        descriptor["producerSha256"]["other.py"] = "sha256:" + "a" * 64
    elif mutation == "traversal":
        descriptor["producerSha256"]["../observer.py"] = descriptor[
            "producerSha256"
        ].pop(PINS[0])
    elif mutation == "hash":
        descriptor["producerSha256"][PINS[0]] = "sha256:" + "0" * 64
    elif mutation == "uppercase":
        descriptor["producerSha256"][PINS[0]] = descriptor["producerSha256"][
            PINS[0]
        ].upper()
    elif mutation == "hash_type":
        descriptor["producerSha256"][PINS[0]] = None
    elif mutation == "schema":
        descriptor["schemaVersion"] = "ROM_TRAINING_WORKBENCH_READ_CAPABILITY_V1"
    elif mutation == "ambiguous":
        value["trainingWorkbenchReadV1"] = None
    original = copy.deepcopy(value)
    assert managed.observer_version(value) == 0
    assert value == original


def test_proof_reads_only_known_resources_and_io_failure_is_optional(monkeypatch):
    value = release()
    calls = []

    class Package:
        def joinpath(self, name):
            calls.append(name)
            raise OSError("resource unavailable")

    monkeypatch.setattr("importlib.resources.files", lambda _: Package())
    assert managed.observer_version(value) == 0
    assert calls == [PINS[0]]
    calls.clear()
    value["trainingWorkbenchReadV2"]["producerSha256"]["../../private"] = (
        "sha256:" + "a" * 64
    )
    assert managed.observer_version(value) == 0
    assert not calls


@pytest.mark.parametrize(
    "fault,code",
    [
        ("shape", "POSE_SHAPE"),
        ("matrix", "POSE_SHAPE"),
        ("nan", "NONFINITE_POSE"),
        ("range", "NONFINITE_POSE"),
        ("integer", "POSE_TYPE"),
        ("count", "ENVIRONMENT_INDICES"),
        ("duplicate", "ENVIRONMENT_INDICES"),
        ("bool", "ENVIRONMENT_INDICES"),
        ("negative", "ENVIRONMENT_INDICES"),
        ("past_end", "ENVIRONMENT_INDICES"),
        ("empty", "ENVIRONMENT_INDICES"),
        ("ten", "ENVIRONMENT_INDICES"),
    ],
)
def test_selected_pack_refuses_invalid_inputs(env, fault, code):
    p = build_projection(env.sim.mj_model, robot_root_body_id=1)
    pos = np.zeros((2, len(p.dynamic_mapping), 3), np.float32)
    mat = np.zeros((2, len(p.dynamic_mapping), 3, 3), np.float32)
    ids, count = [0, 1], 12
    if fault == "shape":
        pos = pos[:, :1]
    elif fault == "matrix":
        mat = mat.reshape(2, -1)
    elif fault == "nan":
        pos[0, 0, 0] = np.nan
    elif fault == "range":
        pos[0, 0, 0] = 1_000_001
    elif fault == "integer":
        pos = pos.astype(int)
    elif fault == "count":
        count = True
    elif fault == "duplicate":
        ids = [0, 0]
    elif fault == "bool":
        ids = [0, True]
    elif fault == "negative":
        ids = [-1, 0]
    elif fault == "past_end":
        ids = [0, 12]
    elif fault == "empty":
        ids = []
    elif fault == "ten":
        ids = list(range(10))
    with pytest.raises(ValueError, match=code):
        p.pack_selected(pos, mat, environment_indices=ids, environment_count=count)


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.002)
    assert predicate()


@pytest.mark.parametrize("blocked", ["geometry", "sink", "transfer"])
def test_two_total_snapshots_when_any_stage_blocks(env, tmp_path, blocked):
    entered, unblock = threading.Event(), threading.Event()
    frames, captures, waits, threads = [], [], [], []
    clock = [1.0]
    main = threading.get_ident()

    class Transfer(observers._SelectedTransfer):
        def capture(self, *args):
            super().capture(*args)
            captures.append(self.buffers[0].copy())

        def wait(self):
            waits.append(threading.get_ident())
            if blocked == "transfer":
                entered.set()
                assert unblock.wait(3)
            super().wait()

    def geometry_sink(_):
        threads.append(threading.get_ident())
        if blocked == "geometry":
            entered.set()
            assert unblock.wait(3)

    def sink(frame):
        threads.append(threading.get_ident())
        frames.append(frame)
        if blocked == "sink":
            entered.set()
            assert unblock.wait(3)

    observer = observers.ProjectedObserver(
        env,
        tmp_path,
        JOB,
        SPEC,
        clock=lambda: clock[0],
        sink=sink,
        geometry_sink=geometry_sink,
        _transfer_factory=Transfer,
    )
    try:
        if blocked == "geometry":
            assert entered.wait(1)
        observer.sample()
        if blocked != "geometry":
            assert entered.wait(1)
        env.sim.data.geom_xpos += 1
        clock[0] += 0.051
        observer.sample()
        original = [a.copy() for a in captures]
        start = time.monotonic()
        for _ in range(100):
            clock[0] += 0.051
            env.sim.data.geom_xpos += 1
            observer.sample()
        assert time.monotonic() - start < 0.25
        assert len(captures) == 2  # Behavioral transfer admission, not queue counters.
        assert all(a.shape == (9, 2, 3) for a in captures)
        for a, b in zip(captures, original):
            np.testing.assert_array_equal(a, b)
        unblock.set()
        wait_until(lambda: len(frames) == 2)
        for frame, capture in zip(frames, original):
            p, _ = arrays(frame)
            for i, (slot, _) in enumerate(observer.projection.dynamic_mapping):
                np.testing.assert_array_equal(p[:, slot], capture[:, i])
        assert all(t != main for t in waits + threads)
    finally:
        unblock.set()
        observer.close()


def test_capture_rate_capture_timestamps_and_nonblocking_condition(env, tmp_path):
    entered, unblock = threading.Event(), threading.Event()
    frames = []
    clock, wall = [10.0], [100.0]

    def geometry_sink(_):
        entered.set()
        assert unblock.wait(3)

    observer = observers.ProjectedObserver(
        env,
        tmp_path,
        JOB,
        SPEC,
        clock=lambda: clock[0],
        wall_clock=lambda: wall[0],
        sink=frames.append,
        geometry_sink=geometry_sink,
    )
    try:
        assert entered.wait(1)
        observer.sample()
        clock[0] = 10.049
        wall[0] = 101.0
        observer.sample()
        clock[0] = 10.05
        wall[0] = 102.0
        env.common_step_counter = 3
        observer.sample()
        clock[0] = 200.0
        wall[0] = 500.0
        unblock.set()
        wait_until(lambda: len(frames) == 2)
        assert [
            (
                f["sequence"],
                f["monotonicSeconds"],
                f["capturedAtUnixSeconds"],
                f["simulationSeconds"],
            )
            for f in frames
        ] == [(0, 10.0, 100.0, 0.0), (1, 10.05, 102.0, 0.06)]
        hold, release_lock = threading.Event(), threading.Event()

        def locker():
            with observer.condition:
                hold.set()
                release_lock.wait(2)

        thread = threading.Thread(target=locker)
        thread.start()
        assert hold.wait(1)
        start = time.monotonic()
        observer.sample()
        assert time.monotonic() - start < 0.1
        release_lock.set()
        thread.join()
        observer.sample()
        wait_until(lambda: len(frames) == 3)
        assert (
            frames[2]["sequence"] == 2 and frames[2]["capturedAtUnixSeconds"] == 500.0
        )
    finally:
        unblock.set()
        observer.close()


@pytest.mark.parametrize("blocked", ["geometry", "sink", "transfer"])
def test_close_never_waits_on_blocked_stage(env, tmp_path, blocked):
    entered, unblock = threading.Event(), threading.Event()
    copies = []

    class Transfer(observers._SelectedTransfer):
        def capture(self, *args):
            super().capture(*args)
            copies.append(1)

        def wait(self):
            if blocked == "transfer":
                entered.set()
                assert unblock.wait(3)
            super().wait()

    def geometry_sink(_):
        if blocked == "geometry":
            entered.set()
            assert unblock.wait(3)

    def sink(_):
        if blocked == "sink":
            entered.set()
            assert unblock.wait(3)

    observer = observers.ProjectedObserver(
        env,
        tmp_path,
        JOB,
        SPEC,
        geometry_sink=geometry_sink,
        sink=sink,
        _transfer_factory=Transfer,
    )
    try:
        observer.sample()
        assert entered.wait(1)
        start = time.monotonic()
        observer.close()
        assert time.monotonic() - start < 0.25
        count = len(copies)
        observer.sample()
        assert len(copies) == count
    finally:
        unblock.set()
        observer.thread.join(2)
    assert not observer.thread.is_alive()


@pytest.mark.parametrize(
    "fault", ["capture", "wait", "geometry", "sink", "serialize", "disk"]
)
def test_optional_failures_stop_observer_without_changing_step(
    env, tmp_path, monkeypatch, fault
):
    entered = threading.Event()

    def fail(*_, **__):
        entered.set()
        raise OSError("optional failure")

    if fault == "capture":
        monkeypatch.setattr(observers._SelectedTransfer, "capture", fail)
    elif fault == "wait":
        monkeypatch.setattr(observers._SelectedTransfer, "wait", fail)
    elif fault in ("geometry", "disk"):
        original_write = observers.ProjectedObserver._write

        def write(self, name, data):
            if fault == "geometry" or name == "latest.json":
                fail()
            original_write(self, name, data)

        monkeypatch.setattr(observers.ProjectedObserver, "_write", write)
    elif fault == "sink":
        monkeypatch.setattr(
            observers.ProjectedObserver,
            "_write",
            lambda self, name, data: fail() if name == "latest.json" else None,
        )
    elif fault == "serialize":
        from mjlab_microduck.workbench.geometry import GeometryProjection

        monkeypatch.setattr(GeometryProjection, "pack_selected", fail)
    observer = observers.attach_observer(env, tmp_path, JOB, SPEC, version=2)
    try:
        assert env.step(None) is env.result
        assert entered.wait(1)
        for _ in range(4):
            assert env.step(None) is env.result
        assert env.common_step_counter == 5
        wait_until(lambda: not observer.thread.is_alive())
    finally:
        observer.close()


@pytest.mark.parametrize(
    "fault", ["allocation", "root", "identity", "geometry_bound", "shape"]
)
def test_attach_failure_does_not_start_worker_or_change_hook(
    env, tmp_path, monkeypatch, fault
):
    step, starts = env.step, []
    monkeypatch.setattr(threading.Thread, "start", lambda self: starts.append(self))
    job = copy.deepcopy(JOB)
    if fault == "allocation":
        monkeypatch.setattr(
            observers._SelectedTransfer,
            "__init__",
            lambda *_: (_ for _ in ()).throw(MemoryError()),
        )
    elif fault == "root":
        env.scene.entities["robot"].indexing.root_body_id = True
    elif fault == "identity":
        job["jobId"] = "x" * 4096
    elif fault == "geometry_bound":
        monkeypatch.setattr(
            "mjlab_microduck.workbench.geometry.MAX_GEOMETRY_BYTES", 500
        )
    elif fault == "shape":
        env.sim.data.geom_xpos = env.sim.data.geom_xpos[:1]
    with pytest.raises((ValueError, MemoryError)):
        observers.attach_observer(env, tmp_path, job, SPEC, version=2)
    assert not starts and env.step is step
    assert not (tmp_path / "observer").exists()


@pytest.mark.parametrize("fault", [None, "second_copy", "record"])
def test_simulated_cuda_route_gathers_before_copy_and_worker_owns_fences(
    env, tmp_path, monkeypatch, fault
):
    """CPU tensor backend with simulated events/stream; NOT real CUDA proof."""
    env.sim.data.geom_xpos = torch.from_numpy(env.sim.data.geom_xpos)
    env.sim.data.geom_xmat = torch.from_numpy(env.sim.data.geom_xmat)
    original = env.sim.data.geom_xpos.clone()
    main = threading.get_ident()
    entered, unblock = threading.Event(), threading.Event()
    copies, allocations, fences, records, captures = [], [], [], [], []
    clock = [1.0]

    class Stream:
        def synchronize(self):
            fences.append(("stream", threading.get_ident()))
            entered.set()
            assert unblock.wait(3)

    stream = Stream()

    class Event:
        def record(self, current):
            assert current is stream
            records.append(threading.get_ident())
            if fault == "record":
                raise RuntimeError("record failed after enqueued copies")

        def synchronize(self):
            fences.append(("event", threading.get_ident()))
            entered.set()
            assert unblock.wait(3)

    empty, copy_, to, clone = (
        torch.empty,
        torch.Tensor.copy_,
        torch.Tensor.to,
        torch.Tensor.clone,
    )

    def allocating(shape, **kwargs):
        assert kwargs.pop("pin_memory") is True
        allocations.append(tuple(shape))
        return empty(shape, **kwargs)

    def copying(self, source, *, non_blocking):
        assert non_blocking is True
        assert source.shape[0] == 9 and source.shape[1] == 2
        copies.append((tuple(source.shape), threading.get_ident()))
        if fault == "second_copy" and len(copies) == 2:
            raise RuntimeError("second asynchronous copy failed")
        return copy_(self, source)

    def converting(self, *args, **kwargs):
        assert threading.get_ident() != main, "producer must not call to(cpu)"
        return to(self, *args, **kwargs)

    def cloning(self, *args, **kwargs):
        assert self.shape[1] <= 2, "full source clone"
        return clone(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(torch, "empty", allocating)
    monkeypatch.setattr(torch.Tensor, "copy_", copying)
    monkeypatch.setattr(torch.Tensor, "to", converting)
    monkeypatch.setattr(torch.Tensor, "clone", cloning)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    observer = observers.ProjectedObserver(
        env,
        tmp_path,
        JOB,
        SPEC,
        clock=lambda: clock[0],
        sink=captures.append,
        geometry_sink=lambda _: None,
    )
    try:
        observer.sample()
        assert entered.wait(1)
        for _ in range(100):
            clock[0] += 0.051
            observer.sample()
        assert allocations == [(9, 2, 3), (9, 2, 3, 3)] * 2
        assert len(copies) == (4 if fault is None else 2)
        assert all(t == main for _, t in copies)
        assert all(t == main for t in records)
        assert all(t != main for _, t in fences)
        assert fences[0][0] == ("event" if fault is None else "stream")
        torch.testing.assert_close(env.sim.data.geom_xpos, original)
        start = time.monotonic()
        observer.close()
        assert time.monotonic() - start < 0.25
    finally:
        unblock.set()
        observer.thread.join(2)
    assert not observer.thread.is_alive()
    assert not torch.cuda.is_initialized()


def test_actual_flat_scene_v2_metadata_only(tmp_path):
    """Self-contained real Scene compile, trusted indexing stub, no native Env."""
    import signal

    from mjlab.scene import Scene

    from mjlab_microduck.workbench import resolve_experiment
    from mjlab_microduck.workbench.engine import build_configs

    def timeout_handler(*_):
        raise TimeoutError("90-second flat Scene ceiling")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(90)
    observer = None
    try:
        spec = resolve_experiment(
            {
                "schemaVersion": "ROM_TRAINING_EXPERIMENT_V1",
                "recipeId": "Mjlab-Velocity-Flat-MicroDuck",
                "robotProfileId": "rom-xl330-v1",
                "config": {"numEnvs": 9, "iterations": 1, "seed": 11},
            }
        )
        cfg, _ = build_configs(spec)
        scene = Scene(cfg.scene, device="cpu")
        model = scene.compile()
        root = model.body("robot/trunk_base").id
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        positions = np.tile(data.geom_xpos, (9, 1, 1))
        positions += np.arange(9)[:, None, None]
        matrices = np.tile(data.geom_xmat.reshape(model.ngeom, 3, 3), (9, 1, 1, 1))
        env = SimpleNamespace(
            num_envs=9,
            step_dt=0.02,
            common_step_counter=1,
            scene=SimpleNamespace(
                entities={
                    "robot": SimpleNamespace(
                        indexing=SimpleNamespace(root_body_id=root)
                    )
                }
            ),
            sim=SimpleNamespace(
                mj_model=model,
                data=SimpleNamespace(geom_xpos=positions, geom_xmat=matrices),
            ),
        )
        frames, geometry = [], []
        observer = observers.ProjectedObserver(
            env, tmp_path, JOB, spec, sink=frames.append, geometry_sink=geometry.append
        )
        observer.sample()
        wait_until(lambda: bool(frames))
        projection = observer.projection
        assert {k: geometry[0][k] for k in projection.export()} == projection.export()
        assert len(geometry[0]["focusSlotIds"]) >= 60
        p, _ = arrays(frames[0])
        for slot, source in projection.dynamic_mapping:
            np.testing.assert_allclose(p[:, slot], positions[:, source], atol=1e-6)
        assert not torch.cuda.is_initialized()
    finally:
        if observer:
            observer.close()
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def test_legacy_default_ignores_job_opt_in_and_preserves_payload_bytes(
    tmp_path, monkeypatch
):
    from mjlab_microduck.rom.viewer import export_geometry, export_meshes

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
    )
    env = SimpleNamespace(
        num_envs=1,
        step_dt=0.02,
        common_step_counter=0,
        sim=SimpleNamespace(
            mj_model=model,
            data=SimpleNamespace(
                geom_xpos=np.zeros((1, 1, 3)), geom_xmat=np.zeros((1, 1, 3, 3))
            ),
        ),
    )
    result = object()
    env.step = lambda _: result
    job = dict(
        JOB,
        observerVersion=2,
        trainingWorkbenchReadV2=release()["trainingWorkbenchReadV2"],
    )
    monkeypatch.setattr(observers.time, "time", lambda: 123.0)
    observer = observers.attach_observer(env, tmp_path, job, SPEC)
    observer.clock = lambda: 7.0
    try:
        assert type(observer) is observers.LatestObserver
        assert env.step(None) is result
        wait_until(lambda: (tmp_path / "observer/latest.json").exists())
        expected_geometry = {
            "schemaVersion": "ROM_TRAINING_OBSERVER_GEOMETRY_V1",
            "geometry": export_geometry(model),
            "meshes": export_meshes(model, export_geometry(model)),
            "modelDigest": JOB["limits"]["modelDigest"],
            "configDigest": SPEC["configDigest"],
            "jobId": JOB["jobId"],
            "requestDigest": JOB["requestDigest"],
        }
        expected_frame = {
            "schemaVersion": "ROM_TRAINING_OBSERVER_FRAME_V1",
            "jobId": JOB["jobId"],
            "requestDigest": JOB["requestDigest"],
            "configDigest": SPEC["configDigest"],
            "robotProfileId": SPEC["robotProfileId"],
            "modelDigest": JOB["limits"]["modelDigest"],
            "sequence": 0,
            "capturedAtUnixSeconds": 123.0,
            "monotonicSeconds": 7.0,
            "simulationSeconds": 0.0,
            "environmentIndices": [0],
            "positions": [[0.0, 0.0, 0.0]],
            "matrices": [[0.0] * 9],
        }
        assert (tmp_path / "observer/geometry.json").read_text() == json.dumps(
            expected_geometry, sort_keys=True, allow_nan=False
        )
        assert (tmp_path / "observer/latest.json").read_text() == json.dumps(
            expected_frame, sort_keys=True, allow_nan=False
        )
    finally:
        observer.close()
    from tests.test_managed_training import job as managed_job

    forged = managed_job()
    forged["observerVersion"] = 2
    with pytest.raises(ValueError, match="MANAGED_SIMULATOR_REQUIRED"):
        managed.validate_request(forged)


@pytest.mark.parametrize(
    "proof,expected", [("absent", 1), ("valid", 2), ("invalid", 0), ("ambiguous", 0)]
)
def test_real_preflight_selects_from_single_image_read_and_preserves_evidence(
    monkeypatch, proof, expected
):
    from pathlib import Path

    from tests.test_managed_training import preflight_host

    value, host = preflight_host.__wrapped__(monkeypatch)
    identity = {
        key: value["limits"][key]
        for key in ("recipeDigest", "modelDigest", "dependencyDigest")
    }
    if proof != "absent":
        identity.update(release())
    if proof == "invalid":
        identity["trainingWorkbenchReadV2"]["producerSha256"][PINS[0]] = "bad"
    if proof == "ambiguous":
        identity["trainingWorkbenchReadV1"] = {}
    original_read, reads = Path.read_text, []

    def read(path, *args, **kwargs):
        if str(path) == "/opt/rom-trainer/recipe.json":
            reads.append(str(path))
            return json.dumps(identity)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    before = copy.deepcopy(value)
    runtime = managed.Runtime()
    assert runtime.preflight(value) == {
        "cudaVerified": True,
        "warpInteropVerified": True,
        "gpuUuid": host["gpu"],
        "driverVersion": host["driver"],
        "computeCapability": "8.6",
    }
    assert runtime._observer_version == expected
    assert reads == ["/opt/rom-trainer/recipe.json"]
    assert value == before


@pytest.mark.parametrize(
    "version,experiment_job,expected",
    [(None, True, [1]), (0, True, []), (1, True, [1]), (2, True, [2]), (2, False, [])],
)
def test_runtime_handoff_uses_only_image_selection_and_preserves_training(
    tmp_path, monkeypatch, version, experiment_job, expected
):
    """Real execute flow with fake GPU/Env/runner; never creates native Env."""
    import sys
    from dataclasses import dataclass
    from types import ModuleType

    from mjlab_microduck.workbench import (
        canonical_json,
        checkpoints,
        engine,
        resolve_experiment,
    )
    from tests.test_managed_training import job as managed_job

    value = managed_job()
    value["requestDigest"] = JOB["requestDigest"]
    if experiment_job:
        spec = resolve_experiment(
            {
                "schemaVersion": "ROM_TRAINING_EXPERIMENT_V1",
                "recipeId": "Mjlab-Velocity-Flat-MicroDuck",
                "robotProfileId": "rom-xl330-v1",
                "config": {"numEnvs": 64, "iterations": 5, "seed": 7},
            }
        )
        value["experimentJson"] = canonical_json(spec)
    original = copy.deepcopy(value)
    events, selected = [], []

    @dataclass
    class AgentConfig:
        clip_actions: float = 1.0
        num_steps_per_env: int = 24

    cfg = SimpleNamespace(scene=SimpleNamespace(), curriculum={})
    env = SimpleNamespace(step_dt=0.02, close=lambda: events.append("env-close"))

    class Runner:
        def save(self, *args):
            raise AssertionError("test stops at learn")

        def learn(self, **kwargs):
            events.append("learn")
            raise RuntimeError("authoritative learn stop")

    runner = Runner()
    modules = {
        "mjlab": {},
        "mjlab.tasks": {},
        "mjlab.envs": {"ManagerBasedRlEnv": lambda **kw: env},
        "mjlab.rl": {
            "MjlabOnPolicyRunner": lambda *a: runner,
            "RslRlVecEnvWrapper": lambda *a, **kw: env,
        },
        "mjlab.tasks.registry": {
            "load_env_cfg": lambda task: cfg,
            "load_rl_cfg": lambda task: AgentConfig(),
            "load_runner_cls": lambda task: None,
        },
        "mjlab.utils.torch": {"configure_torch_backends": lambda: None},
        "mjlab.rl.exporter_utils": {
            "attach_metadata_to_onnx": lambda *a: None,
            "get_base_metadata": lambda *a: {},
        },
        "mjlab_microduck.rom.export_metadata": {
            "attach_microduck_metadata": lambda *a: None
        },
        "mjlab_microduck.training_console": {"observe_runner": lambda *a: None},
        "torch": {"cuda": SimpleNamespace(mem_get_info=lambda _: (1, 1))},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(engine, "build_configs", lambda _: (cfg, AgentConfig()))
    monkeypatch.setattr(checkpoints, "load_teachers", lambda *a: {})

    def attach(*args, version=1):
        selected.append(version)
        return SimpleNamespace(close=lambda: events.append("observer-close"))

    monkeypatch.setattr(observers, "attach_observer", attach)

    class Runtime(managed.Runtime):
        def preflight(self, job):
            if version is not None:
                self._observer_version = version
            return {}

    with pytest.raises(RuntimeError, match="authoritative learn stop"):
        Runtime().execute(value, tmp_path)
    assert selected == expected
    assert events == ["learn"] + (["observer-close"] if expected else []) + [
        "env-close"
    ]
    assert value == original


def test_wrapper_keeps_shared_meshes_materials_heightfield_colors_and_focus(tmp_path):
    model = mujoco.MjModel.from_xml_string("""<mujoco><asset>
    <material name="mat" rgba=".2 .3 .4 1" emission=".3"/>
    <hfield name="h" nrow="2" ncol="3" size="1 2 .5 .2"/>
    <mesh name="m" vertex="0 0 0 1 0 0 0 1 0 0 0 1" face="0 1 2 0 1 3 0 2 3 1 2 3"/>
    </asset><worldbody><geom type="hfield" hfield="h" material="mat"/>
    <body name="robot"><freejoint/><geom type="mesh" mesh="m"/></body>
    <body name="other"><freejoint/><geom type="mesh" mesh="m"/></body>
    </worldbody></mujoco>""")
    model.hfield_data[:] = [0, 0.2, 0.3, 0.8, 1, 0.5]
    env = SimpleNamespace(
        num_envs=1,
        step_dt=0.02,
        common_step_counter=0,
        scene=SimpleNamespace(
            entities={
                "robot": SimpleNamespace(
                    indexing=SimpleNamespace(root_body_id=model.body("robot").id)
                )
            }
        ),
        sim=SimpleNamespace(
            mj_model=model,
            data=SimpleNamespace(
                geom_xpos=np.zeros((1, model.ngeom, 3)),
                geom_xmat=np.zeros((1, model.ngeom, 3, 3)),
            ),
        ),
    )
    geometries = []
    observer = observers.ProjectedObserver(
        env, tmp_path, JOB, SPEC, geometry_sink=geometries.append
    )
    try:
        wait_until(lambda: bool(geometries))
        geometry = geometries[0]
        assert {
            k: geometry[k] for k in observer.projection.export()
        } == observer.projection.export()
        assert len(geometry["focusSlotIds"]) == 1
        assert len(geometry["materials"]) == 1
        shared = [
            g["meshId"]
            for g in geometry["geometry"]
            if g["id"] in [s for s, _ in observer.projection.dynamic_mapping]
        ]
        assert len(shared) == 2 and shared[0] == shared[1]
        assert any("colorsBase64" in m for m in geometry["meshes"])
    finally:
        observer.close()


def test_pack_budget_precedes_dense_allocation_and_checks_encoded_result(
    env, monkeypatch
):
    from mjlab_microduck.workbench import geometry

    p = build_projection(env.sim.mj_model, robot_root_body_id=1)
    pos = np.zeros((1, len(p.dynamic_mapping), 3), np.float32)
    mat = np.zeros((1, len(p.dynamic_mapping), 3, 3), np.float32)
    with monkeypatch.context() as patch:
        patch.setattr(geometry, "MAX_FRAME_BYTES", 1)
        patch.setattr(
            np, "broadcast_to", lambda *a: pytest.fail("allocation before preflight")
        )
        with pytest.raises(ValueError, match="FRAME_BUDGET"):
            p.pack_selected(pos, mat, environment_indices=[0], environment_count=12)
    monkeypatch.setattr(geometry, "_json", lambda _: "x" * 262145)
    with pytest.raises(ValueError, match="FRAME_BUDGET"):
        p.pack_selected(pos, mat, environment_indices=[0], environment_count=12)


def test_final_frame_limit_includes_identity_and_times(env, tmp_path, monkeypatch):
    from mjlab_microduck.workbench import geometry

    ready, release_geometry = threading.Event(), threading.Event()
    frames = []

    def sink(_):
        ready.set()
        assert release_geometry.wait(3)

    observer = observers.ProjectedObserver(
        env, tmp_path, JOB, SPEC, geometry_sink=sink, sink=frames.append
    )
    try:
        assert ready.wait(1)
        # Isolate the wrapper check: packing already has its own pre/post checks.
        original_pack = geometry.GeometryProjection.pack_selected

        def pack(self, *args, **kwargs):
            frame = original_pack(self, *args, **kwargs)
            frame["positionsBase64"] = "A" * 262000
            return frame

        monkeypatch.setattr(geometry.GeometryProjection, "pack_selected", pack)
        observer.sample()
        release_geometry.set()
        wait_until(lambda: not observer.thread.is_alive())
        assert not frames and not (tmp_path / "observer/latest.json").exists()
    finally:
        release_geometry.set()
        observer.close()


def test_rejected_step_hook_starts_no_worker(env, tmp_path, monkeypatch):
    starts = []

    class RejectingEnv:
        def __getattr__(self, name):
            return getattr(env, name)

        def __setattr__(self, name, value):
            if name == "step":
                raise RuntimeError("hook unavailable")
            super().__setattr__(name, value)

    # No fake start: if called, this records a real worker that must be drained.
    start = threading.Thread.start

    def starting(thread):
        starts.append(thread)
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", starting)
    with pytest.raises(RuntimeError, match="hook unavailable"):
        observers.attach_observer(RejectingEnv(), tmp_path, JOB, SPEC, version=2)
    for thread in starts:
        thread.join(1)
    assert not starts


def test_thread_start_failure_restores_original_step(env, tmp_path, monkeypatch):
    step = env.step

    def failing_start(_):
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr(threading.Thread, "start", failing_start)
    with pytest.raises(RuntimeError, match="thread unavailable"):
        observers.attach_observer(env, tmp_path, JOB, SPEC, version=2)
    assert env.step is step
    assert not (tmp_path / "observer").exists()
