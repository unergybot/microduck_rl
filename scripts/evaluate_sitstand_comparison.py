#!/usr/bin/env python3
"""Immutable, CPU-only diagnostic evaluation; never authorizes runtime SIT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "MICRODUCK_POLICY_COMPARISON_V1"
SEEDS = (7, 11, 29, 43, 71, 101, 137, 173)
SCENARIOS = ("STAND_HOLD", "STAND_TO_SIT", "SIT_TO_STAND", "STAND_SIT_STAND")
DT = 0.02


def digest(content):
    return "sha256:" + hashlib.sha256(content).hexdigest()


def verify_digest(path, expected):
    actual = digest(path.read_bytes())
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected) or actual != expected:
        raise ValueError(f"digest mismatch: {path}")
    return actual


def model_closure(path):
    """Discover all includes and file assets, rejecting traversal and symlinks."""
    root = path.absolute().parent
    found = {}

    def visit(file, meshdir="", texturedir=""):
        absolute = file.absolute()
        resolved = absolute.resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise ValueError("model dependency escapes root")
        if any(p.is_symlink() for p in (absolute, *absolute.parents)):
            raise ValueError("model dependency symlink")
        key = resolved.relative_to(root.resolve()).as_posix()
        if key in found:
            return
        found[key] = digest(resolved.read_bytes())
        if resolved.suffix.lower() != ".xml":
            return
        tree = ET.parse(resolved).getroot()
        compiler = tree.find("compiler")
        if compiler is not None:
            meshdir = compiler.get("meshdir", compiler.get("assetdir", meshdir))
            texturedir = compiler.get(
                "texturedir", compiler.get("assetdir", texturedir)
            )
        for node in tree.iter():
            ref = node.get("file")
            if ref:
                base = resolved.parent if node.tag == "include" else root
                directory = (
                    meshdir
                    if node.tag == "mesh"
                    else texturedir
                    if node.tag == "texture"
                    else ""
                )
                visit(base / directory / ref, meshdir, texturedir)

    visit(path)
    return found


class StrictWindow:
    """First settled sample starts a strict hold; no retries mask instability."""

    def __init__(self):
        self.settled_at = None
        self.consecutive = 0

    def step(self, seconds, settled):
        if self.settled_at is None:
            if seconds > 10 + 1e-9:
                return "TRANSITION_TIMEOUT"
            self.consecutive = self.consecutive + 1 if settled else 0
            if self.consecutive >= 10:
                self.settled_at = seconds
            elif seconds >= 10 - 1e-9:
                return "TRANSITION_TIMEOUT"
        if self.settled_at is not None:
            if not settled:
                return "STRICT_HOLD_BROKEN"
            if seconds - self.settled_at >= 30 - 1e-9:
                return "PASSED"
        return None


def publish(staging, output_root, report):
    final = output_root / report["experimentId"]
    if final.exists():
        raise FileExistsError(final)
    (staging / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    artifacts = []
    for path in sorted(staging.iterdir()):
        if not path.is_file() or path.is_symlink():
            raise ValueError("artifacts must be regular files")
        media = {".json": "application/json", ".csv": "text/csv", ".mp4": "video/mp4"}[
            path.suffix
        ]
        artifacts.append(
            {
                "id": path.stem,
                "path": path.name,
                "sha256": digest(path.read_bytes()),
                "sizeBytes": path.stat().st_size,
                "mediaType": media,
            }
        )
    if len(artifacts) > 256:
        raise ValueError("artifact limit exceeded")
    manifest = {
        "schema": SCHEMA,
        "experimentId": report["experimentId"],
        "completedAt": datetime.now(UTC).isoformat(),
        "artifacts": artifacts,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # mkdir lock prevents simultaneous producers replacing the same experiment.
    lock = output_root / ("." + report["experimentId"] + ".lock")
    lock.mkdir()
    try:
        if final.exists():
            raise FileExistsError(final)
        os.rename(staging, final)
    finally:
        lock.rmdir()
    return final


def load_dependencies():
    global \
        np, \
        mujoco, \
        ort, \
        DEFAULT_JOINT_POSE, \
        DeploymentCommand, \
        DeploymentState, \
        build_actor_observation, \
        project_gravity_wxyz, \
        CONTROLLED_SERVO_JOINTS, \
        STAND_SETTLEMENT_LIMITS
    import mujoco
    import numpy as np
    import onnxruntime as ort

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mjlab_microduck.rom.action_specs import STAND_SETTLEMENT_LIMITS
    from mjlab_microduck.rom.contracts import CONTROLLED_SERVO_JOINTS
    from mjlab_microduck.rom.observation import (
        DEFAULT_JOINT_POSE,
        DeploymentCommand,
        DeploymentState,
        build_actor_observation,
        project_gravity_wxyz,
    )


class Simulator:
    def __init__(self, candidate):
        self.cfg = candidate
        self.model = mujoco.MjModel.from_xml_path(candidate["modelPath"])
        self.data = mujoco.MjData(self.model)
        self.substeps = round(DT / self.model.opt.timestep)
        if self.substeps < 1 or not math.isclose(
            self.substeps * self.model.opt.timestep, DT, abs_tol=1e-10
        ):
            raise ValueError("model timestep must divide 20ms exactly")
        if self.model.nu != 14:
            raise ValueError("model must have 14 actuators")
        self.jids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                for n in CONTROLLED_SERVO_JOINTS
            ]
        )
        if np.any(self.jids < 0):
            raise ValueError("canonical servo joints missing")
        self.qids = self.model.jnt_qposadr[self.jids]
        self.vids = self.model.jnt_dofadr[self.jids]
        self.aids = []
        for jid in self.jids:
            matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == jid)
            if len(matches) != 1:
                raise ValueError("servo actuator mapping must be one-to-one")
            self.aids.append(int(matches[0]))
        self.trunk = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base"
        )
        free = np.flatnonzero(
            (self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
            & (self.model.jnt_bodyid == self.trunk)
        )
        if self.trunk < 0 or len(free) != 1:
            raise ValueError("expected trunk_base with one freejoint")
        self.free = int(self.model.jnt_qposadr[free[0]])
        self.gyro = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel"
        )
        if self.gyro < 0:
            raise ValueError("imu_ang_vel sensor missing")
        self.gyroadr = int(self.model.sensor_adr[self.gyro])
        self.home = np.asarray(
            candidate.get("homePose", DEFAULT_JOINT_POSE), dtype=np.float32
        )
        self.sit = self.home.copy()
        self.sit[[1, 2, 3, 4, 10, 11, 12, 13]] = [
            0,
            -0.4079,
            1.35,
            0,
            0,
            0.4079,
            -1.35,
            0,
        ]
        self.sit = np.asarray(candidate.get("sitPose", self.sit), dtype=np.float32)
        if (
            self.home.shape != (14,)
            or self.sit.shape != (14,)
            or not np.isfinite([self.home, self.sit]).all()
        ):
            raise ValueError("poses must contain 14 finite angles")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            candidate["policyPath"],
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        ins, outs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(ins) != 1
            or len(outs) != 1
            or ins[0].shape[-1] != 61
            or outs[0].shape[-1] != 14
            or ins[0].type != "tensor(float)"
        ):
            raise ValueError("ONNX requires one float32 61D input and 14D output")
        self.input_name = ins[0].name
        self.metadata = self.session.get_modelmeta().custom_metadata_map

    def reset(self, seed, sitting):
        mujoco.mj_resetData(self.model, self.data)
        pose = self.sit if sitting else self.home
        self.data.qpos[self.qids] = pose + np.random.default_rng(seed).uniform(
            -0.005, 0.005, 14
        )
        if sitting:
            self.data.qpos[self.free + 2] = self.cfg.get("sitHeightM", 0.060)
        self.data.ctrl[self.aids] = self.data.qpos[self.qids]
        self.previous = np.zeros(14, dtype=np.float32)
        mujoco.mj_forward(self.model, self.data)

    def step(self, sitting):
        command = DeploymentCommand([float(sitting), 0, 0], np.zeros(4), np.zeros(6))
        state = DeploymentState(
            self.data.sensordata[self.gyroadr : self.gyroadr + 3],
            self.data.xquat[self.trunk],
            self.data.qpos[self.qids],
            self.data.qvel[self.vids],
            self.previous,
        )
        obs = build_actor_observation(state, command)
        # Shared builder is canonically centered; explicit experimental home is auditable.
        obs[6:20] += DEFAULT_JOINT_POSE - self.home
        self.last_obs = obs.copy()
        action = np.asarray(
            self.session.run(None, {self.input_name: obs[None]})[0], dtype=np.float32
        )
        if action.shape != (1, 14) or not np.isfinite(action).all():
            raise FloatingPointError("NONFINITE_OR_INVALID_ACTION")
        self.previous = action[0].copy()
        target = self.home + self.previous * self.cfg.get("actionScale", 0.9)
        if not np.isfinite(target).all():
            raise FloatingPointError("NONFINITE_TARGET")
        limits = self.model.actuator_ctrlrange[self.aids]
        limited = np.where(
            self.model.actuator_ctrllimited[self.aids],
            np.clip(target, limits[:, 0], limits[:, 1]),
            target,
        )
        clamps = bool(np.any(limited != target))
        self.data.ctrl[self.aids] = limited
        violations = 0
        for _ in range(self.substeps):
            before_time = self.data.time
            mujoco.mj_step(self.model, self.data)
            numerical_warnings = (
                mujoco.mjtWarning.mjWARN_BADQPOS,
                mujoco.mjtWarning.mjWARN_BADQVEL,
                mujoco.mjtWarning.mjWARN_BADQACC,
                mujoco.mjtWarning.mjWARN_BADCTRL,
            )
            if any(
                self.data.warning[int(w)].number for w in numerical_warnings
            ) or not math.isclose(
                self.data.time, before_time + self.model.opt.timestep, abs_tol=1e-10
            ):
                raise FloatingPointError("NUMERICAL_STATE_AUTORESET")
            q = self.data.qpos[self.qids]
            bounds = self.model.jnt_range[self.jids]
            violations += int(
                np.count_nonzero(
                    self.model.jnt_limited[self.jids]
                    & ((q < bounds[:, 0]) | (q > bounds[:, 1]))
                )
            )
            if (
                not np.isfinite(self.data.qpos).all()
                or not np.isfinite(self.data.qvel).all()
            ):
                raise FloatingPointError("NONFINITE_STATE")
        return clamps, violations

    def measure(self, sitting):
        q = self.data.qpos[self.qids]
        height = float(self.data.xpos[self.trunk, 2])
        tilt = math.acos(
            float(np.clip(-project_gravity_wxyz(self.data.xquat[self.trunk])[2], -1, 1))
        )
        error = float(np.max(np.abs(q - (self.sit if sitting else self.home))))
        speed = float(np.max(np.abs(self.data.qvel[self.vids])))
        lim = STAND_SETTLEMENT_LIMITS
        height_ok = (
            abs(height - self.cfg.get("sitHeightM", 0.060)) <= 0.015
            if sitting
            else lim.trunk_height_min_m <= height <= lim.trunk_height_max_m
        )
        rmse = float(np.sqrt(np.mean(np.square(q - self.home))))
        settled = (
            (error if sitting else rmse) <= lim.pose_error_max_rad
            and height_ok
            and tilt <= lim.trunk_tilt_max_rad
            and speed <= lim.joint_speed_max_radps
        )
        return {
            "maxPoseErrorRad": error,
            "heightM": height,
            "maxTiltRad": tilt,
            "maxJointSpeedRadps": speed,
            "settled": bool(settled),
            "fall": height < 0.025 or tilt > math.radians(75),
        }


class Video:
    def __init__(self, sim, path):
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg executable unavailable")
        self.renderer = mujoco.Renderer(sim.model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 0.65
        self.camera.azimuth = 135
        self.camera.elevation = -15
        self.sim = sim
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "640x480",
                "-r",
                "25",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def frame(self):
        self.camera.lookat[:] = self.sim.data.xpos[self.sim.trunk]
        self.renderer.update_scene(self.sim.data, camera=self.camera)
        self.process.stdin.write(self.renderer.render().tobytes())

    def close(self):
        self.process.stdin.close()
        code = self.process.wait()
        self.renderer.close()
        if code:
            raise RuntimeError("ffmpeg video encoding failed")


def evaluate(sim, scenario, seed, staging, record):
    cid = f"{sim.cfg['id']}-{scenario.lower()}-{seed}"
    csvpath, videopath = staging / (cid + ".csv"), staging / (cid + ".mp4")
    sim.reset(seed, scenario == "SIT_TO_STAND")
    phases = (
        [False, True, False]
        if scenario == "STAND_SIT_STAND"
        else [scenario == "STAND_TO_SIT"]
    )
    metrics = {
        "falls": 0,
        "strictHoldPassed": False,
        "maxPoseErrorRad": 0.0,
        "maxTiltRad": 0.0,
        "maxJointSpeedRadps": 0.0,
        "actuatorClampSteps": 0,
        "physicalJointLimitViolations": 0,
        "transitionSeconds": 0.0,
    }
    video = None
    video_unavailable = None

    def video_failed(error):
        nonlocal video, video_unavailable
        video_unavailable = str(error)
        if video:
            try:
                video.close()
            except Exception as cleanup_error:  # noqa: BLE001 - optional renderer cleanup
                video_unavailable += "; cleanup: " + str(cleanup_error)
        video = None
        videopath.unlink(missing_ok=True)

    if record:
        try:
            video = Video(sim, videopath)
            video.frame()
        except Exception as error:  # noqa: BLE001 - rendering never masks physics evidence
            video_failed(error)
    reason = "PASSED"
    elapsed = 0
    last_command_sit = False
    try:
        with csvpath.open("w", newline="") as stream:
            fields = [
                "timeSeconds",
                "phase",
                "commandSit",
                "maxPoseErrorRad",
                "heightM",
                "maxTiltRad",
                "maxJointSpeedRadps",
                "settled",
                "fall",
                "actuatorClampSteps",
                "physicalJointLimitViolations",
            ]
            fields += [
                f"{prefix}{i}"
                for prefix, count in [
                    ("jointPosition", 14),
                    ("jointVelocity", 14),
                    ("action", 14),
                    ("observation", 61),
                ]
                for i in range(count)
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for phase, sitting in enumerate(phases):
                window = StrictWindow()
                for step in range(2001):
                    local = (step + 1) * DT
                    command_sit = sitting and local >= sim.cfg.get(
                        "sitCommandDelaySeconds", 0.0
                    )
                    if (
                        sim.cfg.get("resetPreviousActionOnCommandChange", False)
                        and command_sit != last_command_sit
                    ):
                        sim.previous[:] = 0
                    last_command_sit = command_sit
                    try:
                        clamps, violations = sim.step(command_sit)
                        measured = sim.measure(sitting)
                    except (FloatingPointError, ValueError) as error:
                        reason = str(error)
                        break
                    metrics["actuatorClampSteps"] += int(clamps)
                    metrics["physicalJointLimitViolations"] += violations
                    for key in ("maxPoseErrorRad", "maxTiltRad", "maxJointSpeedRadps"):
                        metrics[key] = max(metrics[key], measured[key])
                    trace = {
                        f"{prefix}{i}": float(value)
                        for prefix, values in [
                            ("jointPosition", sim.data.qpos[sim.qids]),
                            ("jointVelocity", sim.data.qvel[sim.vids]),
                            ("action", sim.previous),
                            ("observation", sim.last_obs),
                        ]
                        for i, value in enumerate(values)
                    }
                    writer.writerow(
                        dict(
                            timeSeconds=(elapsed + 1) * DT,
                            phase=phase,
                            commandSit=int(command_sit),
                            **measured,
                            **trace,
                            actuatorClampSteps=int(clamps),
                            physicalJointLimitViolations=violations,
                        )
                    )
                    if video and elapsed % 2 == 0:
                        try:
                            video.frame()
                        except Exception as error:  # noqa: BLE001 - keep CSV if rendering fails
                            video_failed(error)
                    elapsed += 1
                    if measured["fall"]:
                        metrics["falls"] += 1
                        reason = "FALL"
                        break
                    outcome = window.step(local, measured["settled"])
                    if outcome:
                        reason = outcome
                        break
                if window.settled_at is not None:
                    metrics["transitionSeconds"] = max(
                        metrics["transitionSeconds"], window.settled_at
                    )
                if reason != "PASSED":
                    break
            metrics["strictHoldPassed"] = reason == "PASSED"
    finally:
        if video:
            try:
                video.close()
            except Exception as error:  # noqa: BLE001 - preserve evaluation if encoding fails
                video_failed(error)
    return {
        "id": cid,
        "candidateId": sim.cfg["id"],
        "scenario": scenario,
        "seed": seed,
        "status": "PASSED" if reason == "PASSED" else "FAILED",
        "reason": reason,
        "metrics": metrics,
        "videoArtifactId": cid if video else None,
        "csvArtifactId": cid,
        "videoUnavailableReason": video_unavailable,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, required=True, help="policy_comparisons directory"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    experiment = config["experimentId"]
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}", experiment):
        raise ValueError("invalid experimentId")
    if not 1 <= len(config["candidates"]) <= 6:
        raise ValueError("1..6 candidates required (256-case/artifact limit)")
    ids = [c["id"] for c in config["candidates"]]
    if len(set(ids)) != len(ids) or any(
        not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,31}", i) for i in ids
    ):
        raise ValueError("candidate ids must be unique safe identifiers")
    for candidate in config["candidates"]:
        for key in ("policySha256", "modelSha256"):
            if not isinstance(candidate.get(key), str) or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", candidate[key]
            ):
                raise ValueError(f"candidate {candidate['id']} requires valid {key}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.output_root / experiment).exists():
        raise FileExistsError(experiment)
    load_dependencies()
    staging = Path(
        tempfile.mkdtemp(prefix="." + experiment + "-", dir=args.output_root)
    )
    report = {
        "schema": SCHEMA,
        "experimentId": experiment,
        "title": config.get("title", experiment),
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "COMPLETED",
        "criteriaVersion": "SITSTAND_DIAGNOSTIC_V1",
        "provenance": {
            "config": config,
            "configSha256": digest(args.config.read_bytes()),
            "evaluatorSha256": digest(Path(__file__).read_bytes()),
            "cpuOnly": True,
            "controlHz": 50,
            "seeds": list(SEEDS),
            "transitionLimitSeconds": 10,
            "strictHoldSeconds": 30,
            "runtimeQualificationChanged": False,
            "versions": {
                "mujoco": mujoco.__version__,
                "onnxruntime": ort.__version__,
                "numpy": np.__version__,
            },
        },
        "candidates": [],
        "cases": [],
        "checks": [],
    }
    try:
        for original in config["candidates"]:
            c = dict(original)
            for key in ("policyPath", "modelPath"):
                c[key] = str((args.config.resolve().parent / c[key]).resolve())
            entry = {
                key: c.get(key, "UNKNOWN")
                for key in (
                    "id",
                    "label",
                    "policySha256",
                    "modelSha256",
                    "source",
                    "normalizationEvidence",
                    "licenseStatus",
                )
            }
            report["candidates"].append(entry)
            sim = None
            try:
                for prefix in ("policy", "model"):
                    verify_digest(Path(c[prefix + "Path"]), c[prefix + "Sha256"])
                closure = model_closure(Path(c["modelPath"]))
                declared = {i["path"]: i["sha256"] for i in c["modelClosure"]}
                if declared != closure:
                    raise ValueError(
                        "modelClosure must exactly match all XML and asset digests"
                    )
                entry["modelClosureSha256"] = digest(
                    json.dumps(closure, sort_keys=True, separators=(",", ":")).encode()
                )
                for key in ("actionScale", "sitHeightM", "sitCommandDelaySeconds"):
                    value = c.get(
                        key,
                        {
                            "actionScale": 0.9,
                            "sitHeightM": 0.060,
                            "sitCommandDelaySeconds": 0.0,
                        }[key],
                    )
                    if (
                        not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                    ):
                        raise ValueError("invalid " + key)
                sim = Simulator(c)
                entry["onnxMetadata"] = sim.metadata
                entry["experimentalOnly"] = True
            except Exception as error:  # noqa: BLE001 - preserve each invalid candidate as UNAVAILABLE
                unavailable = str(error)
            first_failure_recorded = False
            for seed in SEEDS:
                for scenario in SCENARIOS:
                    if sim is None:
                        case = {
                            "id": f"{c['id']}-{scenario.lower()}-{seed}",
                            "candidateId": c["id"],
                            "scenario": scenario,
                            "seed": seed,
                            "status": "UNAVAILABLE",
                            "reason": unavailable,
                            "metrics": {},
                            "videoArtifactId": None,
                            "csvArtifactId": None,
                        }
                    else:
                        case = evaluate(
                            sim,
                            scenario,
                            seed,
                            staging,
                            seed == 7 or not first_failure_recorded,
                        )
                        if case["status"] == "FAILED":
                            first_failure_recorded = True
                        elif seed != 7 and case["videoArtifactId"]:
                            (staging / (case["videoArtifactId"] + ".mp4")).unlink()
                            case["videoArtifactId"] = None
                    # Media IDs must be distinct even though case filenames share a stem.
                    if case["csvArtifactId"]:
                        old = staging / (case["csvArtifactId"] + ".csv")
                        case["csvArtifactId"] += "-trace"
                        old.rename(staging / (case["csvArtifactId"] + ".csv"))
                    report["cases"].append(case)
                    print(
                        f"{case['id']}: {case['status']} {case['reason']}", flush=True
                    )
        report["checks"] = [
            {
                "name": "RUNTIME_SIT_UNCHANGED",
                "status": "PASSED",
                "reason": "Offline diagnostic only; no qualification or runtime changes",
            },
            {
                "name": "EXPERIMENTAL_PROVENANCE",
                "status": "INFO",
                "reason": "External ONNX metadata is recorded; missing metadata does not confer normalization or deployment approval",
            },
        ]
        final = publish(staging, args.output_root, report)
        print(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
