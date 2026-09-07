"""Offline comparison contract: strict windows and immutable publication."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/evaluate_sitstand_comparison.py"


def module():
    assert SCRIPT.exists(), "comparison producer must exist"
    spec = importlib.util.spec_from_file_location("comparison", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_strict_hold_does_not_accept_intermittent_settlement():
    m = module()
    w = m.StrictWindow()
    assert w.step(0, True) is None
    assert w.step(0.02, False) is None
    for i in range(500):
        assert w.step(i * 0.02, False) is None
    assert w.step(10, False) == "TRANSITION_TIMEOUT"
    w = m.StrictWindow()
    for i in range(10):
        assert w.step(i * 0.02, True) is None
    assert w.step(29.98, False) == "STRICT_HOLD_BROKEN"
    w = m.StrictWindow()
    for i in range(10):
        assert w.step(9.82 + i * 0.02, True) is None
    assert w.step(40, True) == "PASSED"


def test_atomic_manifest_hashes_and_refuses_overwrite(tmp_path):
    m = module()
    staging = tmp_path / ".staging"
    staging.mkdir()
    (staging / "trace.csv").write_text("time,value\n0,1\n")
    report = {"schema": m.SCHEMA, "experimentId": "test-7", "cases": []}
    final = m.publish(staging, tmp_path, report)
    manifest = json.loads((final / "manifest.json").read_text())
    assert {a["id"] for a in manifest["artifacts"]} == {"trace", "report"}
    for item in manifest["artifacts"]:
        assert item["sha256"] == m.digest((final / item["path"]).read_bytes())
    again = tmp_path / ".again"
    again.mkdir()
    with pytest.raises(FileExistsError):
        m.publish(again, tmp_path, report)


def test_closure_checks_nested_assets_and_hashes(tmp_path):
    m = module()
    (tmp_path / "root.xml").write_text('<mujoco><include file="child.xml"/></mujoco>')
    (tmp_path / "child.xml").write_text(
        '<mujoco><asset><mesh file="mesh.stl"/></asset></mujoco>'
    )
    (tmp_path / "mesh.stl").write_bytes(b"mesh")
    closure = m.model_closure(tmp_path / "root.xml")
    assert set(closure) == {"root.xml", "child.xml", "mesh.stl"}
    with pytest.raises(ValueError, match="digest"):
        m.verify_digest(tmp_path / "mesh.stl", "sha256:" + "0" * 64)
    (tmp_path / "child.xml").write_text(
        '<mujoco><include file="../outside.xml"/></mujoco>'
    )
    with pytest.raises(ValueError, match="escape"):
        m.model_closure(tmp_path / "root.xml")


def test_simulator_canonical_mapping_with_extra_free_body(tmp_path):
    m = module()
    m.load_dependencies()
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    joints = m.CONTROLLED_SERVO_JOINTS
    bodies = "".join(
        f'<body pos="0 0 {i * 0.002}"><joint name="{name}" type="hinge" range="-3 3"/><geom type="sphere" size=".001" mass=".001"/></body>'
        for i, name in enumerate(joints)
    )
    actuators = "".join(
        f'<position joint="{name}" kp="1" ctrlrange="-3 3"/>'
        for name in reversed(joints)
    )
    xml = f'<mujoco><option timestep=".002" gravity="0 0 0"/><default><joint armature=".01" damping=".1"/><geom contype="0" conaffinity="0"/></default><worldbody><body name="trunk_base" pos="0 0 .115"><freejoint/><geom type="sphere" size=".01" mass="1"/><site name="imu"/>{bodies}</body><body name="ball" pos="1 0 1"><freejoint/><geom type="sphere" size=".1"/></body></worldbody><sensor><gyro name="imu_ang_vel" site="imu"/></sensor><actuator>{actuators}</actuator></mujoco>'
    modelpath = tmp_path / "model.xml"
    modelpath.write_text(xml)
    output = helper.make_tensor("value", TensorProto.FLOAT, [1, 14], [0] * 14)
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["action"], value=output)],
        "actor",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 14])],
    )
    policy = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    policypath = tmp_path / "policy.onnx"
    onnx.save(policy, policypath)
    sim = m.Simulator(
        {"modelPath": str(modelpath), "policyPath": str(policypath), "id": "candidate"}
    )
    sim.reset(7, False)
    assert sim.aids == list(reversed(range(14)))
    clamps, _violations = sim.step(False)
    assert not clamps
    assert np.allclose(sim.data.ctrl[sim.aids], sim.home)
    assert sim.previous.shape == (14,)
    assert sim.session.get_providers() == ["CPUExecutionProvider"]
    assert m.SEEDS == (7, 11, 29, 43, 71, 101, 137, 173)
    sim.reset(7, False)
    sim.data.qvel[0] = float("nan")
    with pytest.raises(FloatingPointError, match="NUMERICAL"):
        sim.step(False)
    nan_output = helper.make_tensor(
        "value", TensorProto.FLOAT, [1, 14], [float("nan")] * 14
    )
    policy.graph.node[0].attribute[0].t.CopyFrom(nan_output)
    onnx.save(policy, policypath)
    bad = m.Simulator(
        {"modelPath": str(modelpath), "policyPath": str(policypath), "id": "bad"}
    )
    bad.reset(7, False)
    with pytest.raises(FloatingPointError, match="NONFINITE"):
        bad.step(False)


def test_failed_case_keeps_joint_and_action_trace_without_renderer(
    tmp_path, monkeypatch
):
    m = module()
    m.load_dependencies()
    from types import SimpleNamespace

    import numpy as np

    class Sim:
        def __init__(self):
            self.cfg = {"id": "c"}

        data = SimpleNamespace(qpos=np.zeros(14), qvel=np.zeros(14))
        qids = vids = np.arange(14)
        previous = np.ones(14)
        last_obs = np.zeros(61)

        def reset(self, *args):
            pass

        def step(self, *args):
            return False, 0

        def measure(self, *args):
            return {
                "maxPoseErrorRad": 0.0,
                "poseRmseRad": 0.0,
                "heightDeviationM": 0.0,
                "heightM": 0.01,
                "maxTiltRad": 0.0,
                "maxJointSpeedRadps": 0.0,
                "settled": False,
                "fall": True,
            }

    def broken_video(*args):
        raise RuntimeError("EGL unavailable")

    monkeypatch.setattr(m, "Video", broken_video)
    case = m.evaluate(Sim(), "STAND_HOLD", 7, tmp_path, True)
    assert case["videoArtifactId"] is None
    assert "EGL unavailable" in case["videoUnavailableReason"]
    import csv

    rows = list(csv.DictReader((tmp_path / (case["csvArtifactId"] + ".csv")).open()))
    assert case["status"] == "FAILED" and case["reason"] == "FALL"
    assert rows[0]["jointPosition13"] == "0.0"
    assert rows[0]["action13"] == "1.0"
    sequence = m.evaluate(Sim(), "STAND_SIT_STAND", 7, tmp_path, False)
    assert [p["status"] for p in sequence["phases"]] == ["FAILED", "NOT_RUN", "NOT_RUN"]
    assert sequence["metrics"]["controlSteps"] == 1
    assert sequence["metrics"]["completedPhases"] == 0


def test_stand_uses_fork_rmse_but_sit_uses_max_error():
    m = module()
    m.load_dependencies()
    from types import SimpleNamespace

    import numpy as np

    sim = m.Simulator.__new__(m.Simulator)
    sim.cfg = {"sitHeightM": 0.115}
    sim.home = sim.sit = np.zeros(14)
    sim.qids = sim.vids = np.arange(14)
    sim.trunk = 0
    q = np.zeros(14)
    q[0] = 0.1
    sim.data = SimpleNamespace(
        qpos=q,
        qvel=np.zeros(14),
        xpos=np.array([[0, 0, 0.115]]),
        xquat=np.array([[1, 0, 0, 0]]),
    )
    assert sim.measure(False)["settled"]
    assert not sim.measure(True)["settled"]
    assert sim.measure(False)["maxPoseErrorRad"] == 0.1


def test_malformed_identity_fails_before_creating_output(tmp_path):
    import subprocess
    import sys

    config = tmp_path / "bad.json"
    config.write_text(json.dumps({"experimentId": "bad", "candidates": [{"id": "c"}]}))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config),
            "--output-root",
            str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires valid policySha256" in result.stderr
    assert not (tmp_path / "out").exists()


def test_effective_identity_deduplicates_alias_but_keeps_behavior_differences():
    m = module()
    m.load_dependencies()
    original = {
        "policySha256": "sha256:" + "a" * 64,
        "modelSha256": "sha256:" + "b" * 64,
    }
    closure = "sha256:" + "c" * 64
    identity = m.candidate_identity(original, closure)
    alias = {**original, "id": "another", "source": "another url", "actionScale": 0.9}
    assert m.candidate_identity(alias, closure) == identity
    assert m.candidate_identity({**original, "actionScale": 1}, closure) != identity
    assert m.candidate_identity(original, "sha256:" + "d" * 64) != identity
    assert (
        m.candidate_identity(
            {**original, "resetPreviousActionOnCommandChange": True}, closure
        )
        != identity
    )


def test_sequence_reports_phase_progress_and_retains_state(tmp_path):
    m = module()
    m.load_dependencies()
    from types import SimpleNamespace

    import numpy as np

    class Sim:
        def __init__(self):
            self.cfg = {
                "id": "sequence",
                "resetPreviousActionOnCommandChange": True,
                "sitCommandDelaySeconds": 0.8,
            }
            self.data = SimpleNamespace(qpos=np.zeros(14), qvel=np.zeros(14))
            self.qids = self.vids = np.arange(14)
            self.previous = np.zeros(14)
            self.last_obs = np.zeros(61)
            self.resets = 0
            self.edges = []
            self.last_sit = False

        def reset(self, *args):
            self.resets += 1

        def step(self, sitting):
            if sitting != self.last_sit:
                self.edges.append((sitting, float(self.previous[0])))
            self.last_sit = sitting
            self.previous[:] = 1
            return False, 0

        def measure(self, sitting):
            return {
                "maxPoseErrorRad": 0.0,
                "poseRmseRad": 0.0,
                "heightDeviationM": 0.0,
                "heightM": 0.06 if sitting else 0.115,
                "maxTiltRad": 0.0,
                "maxJointSpeedRadps": 0.0,
                "settled": sitting == self.last_sit,
                "fall": False,
            }

    sim = Sim()
    case = m.evaluate(sim, "STAND_SIT_STAND", 7, tmp_path, False)
    assert case["status"] == "PASSED"
    assert sim.resets == 1
    assert sim.edges == [(True, 0.0), (False, 0.0)]
    assert len(case["phases"]) == 3
    assert [p["target"] for p in case["phases"]] == ["STAND", "SIT", "STAND"]
    assert all(
        p["holdSteps"] == 1500 and p["status"] == "PASSED" for p in case["phases"]
    )
    assert case["phases"][1]["transitionSeconds"] == pytest.approx(0.98)
    assert case["metrics"]["completedPhases"] == 3
    assert case["metrics"]["holdSteps"] == 4500
    assert case["metrics"]["durationSeconds"] > 90


def test_main_collapses_verified_aliases_and_records_fixed_revisions(
    tmp_path, monkeypatch
):
    m = module()
    m.load_dependencies()
    import sys
    from types import SimpleNamespace

    model = tmp_path / "scene.xml"
    model.write_text("<mujoco/>")
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"policy-boundary-fixture")
    candidate = {
        "id": "first",
        "label": "First",
        "policyPath": "policy.onnx",
        "policySha256": m.digest(policy.read_bytes()),
        "modelPath": "scene.xml",
        "modelSha256": m.digest(model.read_bytes()),
        "modelClosure": [{"path": "scene.xml", "sha256": m.digest(model.read_bytes())}],
    }
    config = {
        "experimentId": "aliases",
        "codeRevisions": {"rl": "a" * 40, "runtime": "b" * 40},
        "candidates": [
            candidate,
            {**candidate, "id": "alias", "source": "another source"},
            {**candidate, "id": "different-scale", "actionScale": 1.0},
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(m, "Simulator", lambda c: SimpleNamespace(cfg=c, metadata={}))

    def evaluate(sim, scenario, seed, *args):
        return {
            "id": f"{sim.cfg['id']}-{scenario}-{seed}",
            "candidateId": sim.cfg["id"],
            "scenario": scenario,
            "seed": seed,
            "status": "PASSED",
            "reason": "fixture",
            "metrics": {},
            "videoArtifactId": None,
            "csvArtifactId": None,
        }

    monkeypatch.setattr(m, "evaluate", evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate", "--config", str(path), "--output-root", str(tmp_path / "out")],
    )
    m.main()
    report = json.loads((tmp_path / "out/aliases/report.json").read_text())
    assert report["provenance"]["codeRevisions"] == config["codeRevisions"]
    assert len(report["candidates"]) == 2
    assert len(report["cases"]) == 64
    assert report["candidates"][0]["aliases"][0]["id"] == "alias"
    assert report["candidates"][1]["effectiveBehavior"]["actionScale"] == 1.0
    with pytest.raises(ValueError, match="full fixed"):
        m.code_revisions({"codeRevisions": {"rl": "a" * 40, "runtime": "branch-main"}})
