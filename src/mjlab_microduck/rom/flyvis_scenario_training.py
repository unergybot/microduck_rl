"""Collection, readout fitting and held-out motion/obstacle comparisons."""

import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.util import Finalize
from pathlib import Path

import numpy as np

from .flyvis_scenarios import (
    SCENARIO_PROTOCOL,
    SCENARIOS,
    ScenarioWorld,
    scenario_episode,
    validate_scenario,
)
from .flyvis_tracking.core import PHYSICS_PROTOCOL, RANDOMIZATION_PROTOCOL, EpisodeSpec
from .flyvis_tracking.experiment import (
    STARTUP_PROTOCOL,
    load_dataset,
    provenance,
    validate_protocol,
    write_json,
)
from .flyvis_tracking.learning import Readout, fit_readout
from .flyvis_tracking.simulation import file_digest
from .flyvis_tracking.vision import FlyvisVision, RetinaVision

_world = _vision = None


def specs(partition, count, offset=0):
    if count < 1 or offset < 0 or count + offset >= 100_000:
        raise ValueError("invalid count/offset")
    base = {"train": 0, "validation": 100_000, "probe": 200_000, "test": 300_000}[
        partition
    ]
    return [
        EpisodeSpec(base + i, SCENARIOS[i % len(SCENARIOS)], partition)
        for i in range(offset, offset + count)
    ]


def load_scenarios(paths, kind, partition, identity=None):
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
        validate_scenario(meta)
        if identity:
            fields = ("extent", "kernelSize", "grayscaleWeights", "retinaSites")
            actual = (
                meta["vision"]
                if kind == "flyvis"
                else {k: meta["vision"][k] for k in fields}
            )
            expected = (
                identity["vision"]
                if kind == "flyvis"
                else {k: identity["vision"][k] for k in fields}
            )
            if meta["bundleDigest"] != identity["bundleDigest"] or actual != expected:
                raise ValueError("mixed scenario dataset identities")
    return load_dataset(paths, kind, partition)


def scenario_provenance():
    result = provenance()
    root = Path(__file__).resolve().parents[3]
    paths = [
        Path(__file__),
        Path(__file__).with_name("flyvis_scenarios.py"),
        root / "scripts/evaluate_flyvis_microduck_scenarios.py",
    ]
    result["scenarioSourceFiles"] = {
        str(p.relative_to(root)): file_digest(p) for p in paths
    }
    return result


def _initialize(bundle, model, kind):
    global _world, _vision
    _world = ScenarioWorld(bundle)
    _vision = FlyvisVision(model) if kind == "flyvis" else RetinaVision()
    Finalize(_world, _world.close, exitpriority=10)


def _execute(job):
    options = dict(job)
    checkpoint = options.pop("checkpoint", None)
    readout = Readout.load(checkpoint) if checkpoint else None
    return scenario_episode(_world, _vision, readout=readout, **options)


class ScenarioPool:
    def __init__(self, bundle, model, kind, jobs):
        if not 1 <= jobs <= 4:
            raise ValueError("scenario jobs must be 1 through 4")
        self.executor = ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
            initargs=(bundle, model, kind),
        )

    def run(self, work):
        return self.executor.map(_execute, work, chunksize=1)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.executor.shutdown(wait=True, cancel_futures=True)


def collect(bundle, model, output, count=30, validation_count=10, seconds=20.0, jobs=2):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    work = [
        {
            "spec": spec,
            "record": output / f"{partition}-{spec.seed}.npz",
            "seconds": seconds,
            "video": i < 5,
        }
        for partition, n in [("train", count), ("validation", validation_count)]
        for i, spec in enumerate(specs(partition, n))
    ]
    results = []
    with ScenarioPool(bundle, model, "flyvis", jobs) as pool:
        for result in pool.run(work):
            results.append(result)
            print(json.dumps({"phase": "collect", **result}), flush=True)
    with np.load(work[0]["record"], allow_pickle=False) as data:
        identity = json.loads(str(data["metadata"]))
    write_json(
        output / "dataset.json",
        {
            "schema": "MICRODUCK_SCENARIO_DATASET_V1",
            "scenarioProtocol": SCENARIO_PROTOCOL,
            "startupProtocol": STARTUP_PROTOCOL,
            "randomizationProtocol": RANDOMIZATION_PROTOCOL,
            "physicsProtocol": PHYSICS_PROTOCOL,
            "count": count,
            "validationCount": validation_count,
            "seconds": seconds,
            "bundleDigest": identity["bundleDigest"],
            "vision": identity["vision"],
            "provenance": scenario_provenance(),
            "teacherResults": results,
        },
    )


def train(
    bundle,
    model,
    dataset,
    output,
    kind="flyvis",
    seed=1,
    rounds=1,
    round_episodes=20,
    epochs=30,
    jobs=2,
):
    dataset, output = Path(dataset), Path(output)
    info = json.loads((dataset / "dataset.json").read_text())
    validate_scenario(info)
    validate_protocol(info)
    if rounds < 0 or epochs < 1:
        raise ValueError("invalid training budget")
    output.mkdir(parents=True, exist_ok=False)
    train_paths = sorted(dataset.glob("train-*.npz"))
    validation_paths = sorted(dataset.glob("validation-*.npz"))
    vx, vy = load_scenarios(validation_paths, kind, "validation", info)
    visual_identity = info["vision"] if kind == "flyvis" else RetinaVision().manifest
    with ScenarioWorld(bundle) as world:
        if world.bundle.bundleDigest != info["bundleDigest"]:
            raise ValueError("scenario bundle differs from dataset")
    best, selected = -1.0, None
    with ScenarioPool(bundle, model, kind, jobs) as pool:
        for stage in range(rounds + 1):
            x, y = load_scenarios(train_paths, kind, "train", info)
            readout, history = fit_readout(x, y, vx, vy, seed=seed, epochs=epochs)
            checkpoint = output / f"round-{stage}.npz"
            readout.save(
                checkpoint,
                {
                    "scenarioProtocol": SCENARIO_PROTOCOL,
                    "startupProtocol": STARTUP_PROTOCOL,
                    "randomizationProtocol": RANDOMIZATION_PROTOCOL,
                    "physicsProtocol": PHYSICS_PROTOCOL,
                    "kind": kind,
                    "seed": seed,
                    "stage": stage,
                    "vision": visual_identity,
                    "bundleDigest": info["bundleDigest"],
                    "datasetFiles": {
                        str(p.resolve()): file_digest(p) for p in train_paths
                    },
                    "validationFiles": {
                        str(p.resolve()): file_digest(p) for p in validation_paths
                    },
                    "history": history,
                    "provenance": scenario_provenance(),
                },
            )
            stage_dir = output / f"validation-{stage}"
            stage_dir.mkdir()
            results = list(
                pool.run(
                    [
                        {
                            "spec": spec,
                            "record": stage_dir / f"{spec.seed}.npz",
                            "checkpoint": checkpoint,
                            "seconds": info["seconds"],
                        }
                        for spec in specs("validation", info["validationCount"])
                    ]
                )
            )
            score = float(
                np.mean(
                    [
                        float(r["passed"])
                        + 0.1 * max(0.0, min(1.0, r["distanceReductionM"]))
                        if not (r["fallen"] or r["collision"] or r["fault"])
                        else 0.0
                        for r in results
                    ]
                )
            )
            write_json(output / f"round-{stage}-validation.json", results)
            if score > best:
                best, selected = score, checkpoint
            print(
                json.dumps(
                    {
                        "phase": "train",
                        "kind": kind,
                        "stage": stage,
                        "selectionScore": score,
                    }
                ),
                flush=True,
            )
            if stage < rounds:
                dagger = output / f"dagger-{stage}"
                dagger.mkdir()
                work = [
                    {
                        "spec": spec,
                        "record": dagger / f"{spec.seed}.npz",
                        "checkpoint": checkpoint,
                        "seconds": info["seconds"],
                    }
                    for spec in specs(
                        "train", round_episodes, info["count"] + stage * round_episodes
                    )
                ]
                for result in pool.run(work):
                    print(json.dumps({"phase": "dagger", **result}), flush=True)
                train_paths.extend(job["record"] for job in work)
    shutil.copyfile(selected, output / "selected.npz")
    write_json(
        output / "selection.json", {"selected": selected.name, "selectionScore": best}
    )


def evaluate(
    bundle,
    model,
    output,
    checkpoint=None,
    kind="flyvis",
    count=10,
    seconds=20.0,
    jobs=2,
    interventions=("none",),
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if checkpoint:
        metadata = Readout.load(checkpoint).metadata
        validate_scenario(metadata)
        kind = metadata["kind"]
    report = {
        "physicsProtocol": PHYSICS_PROTOCOL,
        "schema": "MICRODUCK_MOTION_OBSTACLES_V1",
        "scenarioProtocol": SCENARIO_PROTOCOL,
        "controllerRole": "learned_sensor_readout"
        if checkpoint
        else "privileged_geometry_teacher",
        "checkpointDigest": file_digest(checkpoint) if checkpoint else None,
        "provenance": scenario_provenance(),
        "runs": [],
        "criteria": {
            "minimumDistanceReductionM": 0.2,
            "minimumDisplacementM": 0.2,
            "finalDistanceBandM": [0.4, 0.8],
            "finalBearingDegrees": 15,
            "noFallsCollisionsOrFaults": True,
        },
        "observerCameraFeedsController": False,
        "scientificAdvantageEstablished": False,
    }
    with ScenarioPool(bundle, model, kind, jobs) as pool:
        for intervention in interventions:
            directory = output / intervention
            directory.mkdir()
            work = [
                {
                    "spec": spec,
                    "record": directory / f"{spec.family}-{spec.seed}.npz",
                    "checkpoint": checkpoint,
                    "seconds": seconds,
                    "video": i < 5,
                    "intervention": intervention,
                }
                for i, spec in enumerate(
                    specs("test" if checkpoint else "probe", count)
                )
            ]
            for job, result in zip(work, pool.run(work), strict=True):
                result.update(
                    artifact=str(job["record"].relative_to(output)),
                    artifactDigest=file_digest(job["record"]),
                )
                report["runs"].append(result)
                write_json(output / "report.json", report)
                print(json.dumps({"phase": "evaluate", **result}), flush=True)
    lines = [
        "# MicroDuck motion and obstacle results",
        "",
        report["controllerRole"],
        "",
        "Pass requires reaching the target with at least 0.2 m approach progress and displacement, without collision, fall or fault.",
        "",
        "| Scenario | Input | Passed | Progress (m) | Collision |",
        "|---|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {r['family']} | {r['intervention']} | {r['passed']} | {r['distanceReductionM']:.3f} | {r['collision']} |"
        for r in report["runs"]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
