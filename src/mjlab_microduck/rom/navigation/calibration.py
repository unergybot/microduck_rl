"""Offline command-response measurements using the actual ROM MuJoCo runtime.

No network, provider credentials, policy mutation, or benchmark promotion.
Run with ``python -m mjlab_microduck.rom.navigation.calibration --help``.
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from ..contracts import TaskCreateRequest
from ..main import load_verified_bundle
from ..mujoco_runtime import MicroduckMujocoRuntime
from .installation import source_digest


def summarize(samples, command):
    start, end = samples[0], samples[-1]
    measured = samples[1:]
    dx, dy = end["x"] - start["x"], end["y"] - start["y"]
    yaw = sum(
        math.atan2(math.sin(b["yaw"] - a["yaw"]), math.cos(b["yaw"] - a["yaw"]))
        for a, b in itertools.pairwise(samples)
    )
    return {
        "durationS": end["time"] - start["time"],
        "forwardDisplacementM": math.cos(start["yaw"]) * dx
        + math.sin(start["yaw"]) * dy,
        "lateralDisplacementM": -math.sin(start["yaw"]) * dx
        + math.cos(start["yaw"]) * dy,
        "planarDisplacementM": math.hypot(dx, dy),
        "verticalDisplacementM": end["z"] - start["z"],
        "yawRotationRad": yaw,
        "maxTiltRad": max(s["tilt"] for s in samples),
        "peakPlanarSpeedMps": max(math.hypot(s["vx"], s["vy"]) for s in samples),
        "trackingRmse": {
            key: math.sqrt(
                sum((s[key] - target) ** 2 for s in measured) / len(measured)
            )
            if measured
            else None
            for key, target in zip(("vx", "vy", "yawRate"), command, strict=True)
        },
    }


def _observe(runtime, command, phase):
    position = runtime._base_position()
    yaw = runtime._yaw_rad()
    velocity = runtime._data.qvel[
        runtime._free_qvel_address : runtime._free_qvel_address + 2
    ]
    _w, x, y, _z = runtime._base_quaternion_wxyz()
    return {
        "time": float(runtime._data.time),
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]),
        "yaw": yaw,
        "vx": float(math.cos(yaw) * velocity[0] + math.sin(yaw) * velocity[1]),
        "vy": float(-math.sin(yaw) * velocity[0] + math.cos(yaw) * velocity[1]),
        "yawRate": float(runtime._base_angular_velocity()[2]),
        "tilt": math.acos(float(np.clip(1 - 2 * (x * x + y * y), -1, 1))),
        "command": list(command),
        "phase": phase,
    }


def _settlement_seconds(samples):
    since = None
    for sample in samples:
        if (
            math.hypot(sample["vx"], sample["vy"]) <= 0.02
            and abs(sample["yawRate"]) <= 0.05
        ):
            if since is None:
                since = sample["time"]
            if sample["time"] - since >= 0.5 - 1e-9:
                return sample["time"] - samples[0]["time"]
        else:
            since = None
    return None


def run_case(root, bundle, command, seed, *, move_seconds=10.0, stop_seconds=2.0):
    for duration in (move_seconds, stop_seconds):
        if (
            not math.isfinite(duration)
            or not 0.02 <= duration <= 60
            or not math.isclose(duration * 50, round(duration * 50))
        ):
            raise ValueError(
                "durations must be whole 50 Hz ticks between 0.02 and 60 seconds"
            )
    request = TaskCreateRequest(
        schema="MICRODUCK_SIM_TASK_V1",
        taskId=f"{seed:032x}",
        actionCode="WALK_VELOCITY",
        bundleVersion=bundle.bundleVersion,
        bundleDigest=bundle.bundleDigest,
        parameters=dict(zip(("vxMps", "vyMps", "yawRateRadps"), command, strict=True)),
        scenario={"terrain": "flat", "seed": seed},
        leaseMs=1000,
        requestedBy="offline-calibration",
    )
    runtime = MicroduckMujocoRuntime(Path(root), bundle, realtime=False)
    handle = None
    try:
        action = next(a for a in bundle.actions if a.actionCode == "WALK_VELOCITY")
        handle = runtime.start(action, request)
        move = [_observe(runtime, command, "MOVE")]
        terminal = None
        for _ in range(round(move_seconds * 50)):
            sample = runtime.sample(handle)
            move.append(_observe(runtime, command, "MOVE"))
            if not sample.running:
                terminal = sample.stopReason
                break
        zero = (0.0, 0.0, 0.0)
        stop = []
        if terminal is None:
            runtime.command(handle, {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0})
            stop = [_observe(runtime, zero, "STOP")]
            for _ in range(round(stop_seconds * 50)):
                sample = runtime.sample(handle)
                stop.append(_observe(runtime, zero, "STOP"))
                if not sample.running:
                    terminal = sample.stopReason
                    break
        runtime.safe_stop(handle, terminal or "CALIBRATION_END")
        stopped = runtime.status()
        handle = None
        return {
            "seed": seed,
            "command": list(command),
            "bundleDigest": bundle.bundleDigest,
            "move": summarize(move, command),
            "stop": summarize(stop, zero) if stop else None,
            "settledAfterZeroCommandS": _settlement_seconds(stop) if stop else None,
            "stoppedCommandConfirmed": stopped.activeTaskId is None
            and stopped.appliedMotion["twist"] == [0.0, 0.0, 0.0],
            "terminalReason": terminal,
            "actuatorClampSteps": runtime._actuator_clamp_steps,
            "physicalJointLimitViolations": runtime._physical_joint_limit_violations,
            "trace": move + stop,
        }
    finally:
        try:
            if handle is not None:
                runtime.safe_stop(handle, "CALIBRATION_ABORTED")
        finally:
            runtime._snapshot.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 29])
    args = parser.parse_args()
    root = args.bundle.resolve()
    if args.output.resolve().is_relative_to(root):
        parser.error("calibration output must be outside the installed bundle")
    bundle = load_verified_bundle(root)
    commands = [(speed, 0.0, 0.0) for speed in (0.05, 0.1, 0.2, 0.4)]
    commands += [(0.0, 0.0, turn) for turn in (-1.0, -0.4, -0.2, 0.2, 0.4, 1.0)]
    report = {
        "schema": "MICRODUCK_COMMAND_CALIBRATION_V1",
        "engine": "MUJOCO_ONNX",
        "provenance": "SIM_GROUND_TRUTH",
        "runtimeSourceDigest": source_digest(),
        "bundleDigest": bundle.bundleDigest,
        "modelDigest": bundle.model.digest,
        "policyDigests": [p.digest for p in bundle.policies],
        "runs": [],
    }
    for command in commands:
        for seed in args.seeds:
            result = run_case(root, bundle, command, seed)
            report["runs"].append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(args.output)
            print(
                json.dumps(
                    {
                        "command": command,
                        "seed": seed,
                        "move": result["move"],
                        "stop": result["stop"],
                        "settledS": result["settledAfterZeroCommandS"],
                    }
                ),
                flush=True,
            )
    return (
        0
        if all(
            r["terminalReason"] is None and r["stoppedCommandConfirmed"]
            for r in report["runs"]
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
