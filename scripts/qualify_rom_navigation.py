#!/usr/bin/env python3
"""Offline real MuJoCo/ONNX navigation qualification; never connects to a robot.

Failed reports are retained and cannot enable navigation. Candidate injection is
limited to offline stepping; ordinary runtime construction requires a passed report.
"""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from mjlab_microduck.rom.main import load_verified_bundle
from mjlab_microduck.rom.mujoco_runtime import MicroduckMujocoRuntime
from mjlab_microduck.rom.navigation.installation import Installation, source_digest
from mjlab_microduck.rom.navigation_contracts import (
    NavigationProfile,
    NavigationTaskRequest,
    Scene,
    digest,
)
from mjlab_microduck.rom.navigation_service import NavigationRuntimeRequest


def run(root, bundle, scene, profile, scenario, seed):
    clock = [0.0]
    installed = Installation(
        scene,
        profile,
        "qualification",
        "qualification",
        bundle.bundleDigest,
        "sha256:" + "0" * 64,
    )
    runtime = MicroduckMujocoRuntime(
        root,
        bundle,
        realtime=False,
        monotonic_clock=lambda: clock[0],
        _navigation_candidate=installed,
    )
    with runtime._lock:
        runtime._reset_model_locked(np.random.default_rng(seed))
        x, y, yaw = scenario["start"]
        index = runtime._free_qpos_address
        runtime._data.qpos[index : index + 2] = [x, y]
        runtime._data.qpos[index + 3 : index + 7] = [
            math.cos(yaw / 2),
            0,
            0,
            math.sin(yaw / 2),
        ]
        mujoco.mj_forward(runtime._model, runtime._data)
    proposal = {
        "schema": "ROM_MICRODUCK_NAVIGATION_PROPOSAL_V2",
        "actionCode": "NAVIGATE_TO_LANDMARK",
        "binding": {
            "robotId": "qualification",
            "targetId": "qualification",
            "runtimeSession": "offline",
        },
        "landmarkId": scenario["landmarkId"],
        "destination": scene.landmarks[scenario["landmarkId"]].model_dump(),
        "scene": scene.model_dump(),
        "mapDigest": digest(scene),
        "bundleVersion": bundle.bundleVersion,
        "bundleDigest": bundle.bundleDigest,
        "navigationProfileDigest": digest(profile),
        "profile": profile.model_dump(),
        "safeStopBehavior": "ZERO_THEN_HOLD",
        "planning": {
            "source": "MANUAL",
            "requestId": "qualification",
            "inputSnapshotDigest": "sha256:" + "0" * 64,
            "providerResultRef": None,
        },
    }
    nav = NavigationTaskRequest(
        schema="MICRODUCK_NAVIGATION_TASK_V2",
        taskId=f"{seed:032x}",
        proposalDigest=digest(proposal),
        proposal=proposal,
        requestedBy="qualification",
    )
    request = NavigationRuntimeRequest(
        schema="MICRODUCK_SIM_TASK_V1",
        taskId=nav.taskId,
        actionCode="WALK_VELOCITY",
        bundleVersion=bundle.bundleVersion,
        bundleDigest=bundle.bundleDigest,
        parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
        scenario={"terrain": "flat", "seed": seed},
        leaseMs=profile.leaseMs,
        requestedBy="qualification",
        navigation=nav,
    )
    action = next(a for a in bundle.actions if a.actionCode == "WALK_VELOCITY")
    handle = runtime.start(action, request)
    collision = False
    for _ in range(profile.deadlineMs // 20 + 2):
        clock[0] += 0.02
        sample = runtime.sample(handle)
        for contact in runtime._data.contact:
            names = [
                mujoco.mj_id2name(runtime._model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                or ""
                for g in (contact.geom1, contact.geom2)
            ]
            if any(name.startswith("rom_navigation_obstacle_") for name in names):
                collision = True
        if not sample.running:
            break
    evidence = runtime.safe_stop(handle, sample.stopReason or "QUALIFICATION_END")
    expected = scenario["expectedReason"]
    passed = (
        sample.terminalState == "SUCCEEDED"
        if expected == "ARRIVED"
        else sample.stopReason == expected
    )
    passed = (
        passed
        and not collision
        and evidence.metrics.get("stoppedCommandConfirmed") is True
    )
    return {
        "scenario": scenario["name"],
        "seed": seed,
        "passed": passed,
        "state": sample.terminalState,
        "reason": sample.stopReason,
        "collision": collision,
        "durationS": clock[0],
        "evidence": evidence.metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("tests/fixtures/navigation/scenarios.json"),
    )
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.scenarios.read_text())
    scene = Scene.model_validate(data["scene"])
    profile = NavigationProfile.model_validate(data["profile"])
    bundle = load_verified_bundle(args.bundle)
    report = {
        "schema": "MICRODUCK_NAVIGATION_QUALIFICATION_V1",
        "engine": "MUJOCO_ONNX",
        "provenance": "SIM_GROUND_TRUTH",
        "runtimeSourceDigest": source_digest(),
        "scenarioDigest": digest(data),
        "bundleDigest": bundle.bundleDigest,
        "mapDigest": digest(scene),
        "profileDigest": digest(profile),
        "runs": [],
    }
    for scenario in data["scenarios"]:
        for seed in range(args.seed_count):
            try:
                result = run(args.bundle, bundle, scene, profile, scenario, seed)
            except Exception as error:  # noqa: BLE001 - retain every failed qualification result
                result = {
                    "scenario": scenario["name"],
                    "seed": seed,
                    "passed": False,
                    "error": type(error).__name__ + ": " + str(error),
                }
            report["runs"].append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                scenario["name"],
                seed,
                result.get("passed"),
                result.get("reason", result.get("error")),
                flush=True,
            )
    return (
        0
        if len(report["runs"]) >= 50 and all(r["passed"] for r in report["runs"])
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
