"""Collection, DAgger readout training, held-out evaluation and synchronized artifacts."""

import hashlib
import importlib.metadata
import json
import math
import resource
import time
from dataclasses import asdict, replace
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .batch import RolloutPool
from .core import (
    PHYSICS_PROTOCOL,
    RANDOMIZATION_PROTOCOL,
    Action,
    MovementGate,
    episode_specs,
    score_episode,
    teacher_action,
)
from .learning import Readout, fit_readout
from .simulation import TrackingWorld, file_digest
from .vision import FlyvisVision, RetinaVision

STARTUP_PROTOCOL = "immediate_sensor_control"


def validate_protocol(metadata):
    if metadata.get("startupProtocol") != STARTUP_PROTOCOL:
        raise ValueError("startup protocol differs from immediate sensor control")
    if metadata.get("randomizationProtocol") != RANDOMIZATION_PROTOCOL:
        raise ValueError(
            "randomization protocol differs from independent target stream"
        )

    if metadata.get("physicsProtocol") != PHYSICS_PROTOCOL:
        raise ValueError("physics protocol differs from isolated rendering")


def provenance():
    root = Path(__file__).resolve().parents[4]
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += [root / "scripts/evaluate_flyvis_microduck.py", root / "uv.lock"]
    digests = {str(p.relative_to(root)): file_digest(p) for p in paths}
    return {
        "sourceFiles": digests,
        "sourceSha256": hashlib.sha256(
            json.dumps(digests, sort_keys=True).encode()
        ).hexdigest(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("flyvis", "torch", "mujoco", "numpy", "scipy", "onnxruntime")
        },
    }


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def temporal_metrics(rows):
    losses, settlements, reacquisitions = [], [], []
    for i, row in enumerate(rows):
        if i and not row["visible"] and rows[i - 1]["visible"]:
            end = next(
                (j for j in range(i + 1, len(rows)) if rows[j]["visible"]), len(rows)
            )
            window = rows[i:end]
            stopped = next((r for r in window if r["requested"] == 0), None)
            losses.append(stopped["time"] - row["time"] if stopped else None)
            since, settled = None, None
            for r in window:
                if r["speed"] <= 0.02 and abs(r["yaw_rate"]) <= 0.05:
                    since = r["time"] if since is None else since
                    if r["time"] - since >= 0.5 - 1e-9:
                        settled = r["time"] - row["time"]
                        break
                else:
                    since = None
            settlements.append(settled)
        if i and row["visible"] and not rows[i - 1]["visible"]:
            end = next(
                (j for j in range(i + 1, len(rows)) if not rows[j]["visible"]),
                len(rows),
            )
            tracked = next(
                (
                    r
                    for r in rows[i:end]
                    if abs(r["bearing"]) <= math.radians(15)
                    and 0.4 <= r["distance"] <= 0.8
                ),
                None,
            )
            reacquisitions.append(tracked["time"] - row["time"] if tracked else None)
    return {
        "lossStopCommandLatencyS": losses,
        "lossSettlementLatencyS": settlements,
        "reacquisitionLatencyS": reacquisitions,
    }


def compose_frame(rgb, retina, features, row):
    """Pixels remain untouched in the upper-left panel; annotations never feed inference."""
    canvas = Image.new("RGB", (640, 480), "#161a22")
    canvas.paste(Image.fromarray(rgb), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((330, 12), "HEAD CAMERA / BODY FEEDBACK", fill="white")
    text = (
        f"t={row['time']:.2f}s  visible={row['visible']}\n"
        f"requested: {Action(row['requested']).name}\n"
        f"applied: {Action(row['applied']).name}\n"
        f"bearing: {math.degrees(row['bearing']):.1f} deg\n"
        f"distance: {row['distance']:.2f}m\n"
        f"speed: {row['speed']:.3f}m/s\n"
        f"yaw rate: {row['yaw_rate']:.3f}rad/s"
    )
    draw.multiline_text((330, 40), text, fill="white", spacing=8)
    # The lattice is shown in the same receptor ordering used by BoxEye.
    index = 0
    for u in range(-15, 16):
        for v in range(max(-15, -15 - u), min(15, 15 - u) + 1):
            x, y = 150 + v * 4, 365 + (u + v / 2) * 4
            grey = int(np.clip(retina[index], 0, 1) * 255)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(grey,) * 3)
            index += 1
    draw.text((12, 255), "RETINAL INPUT", fill="white")
    draw.text((330, 255), "VISUAL FEATURES (activity, not spikes)", fill="white")
    values = np.asarray(features[:270], float)
    scale = max(float(np.max(np.abs(values))), 1e-6)
    for i, value in enumerate(values):
        height = int(130 * abs(value) / scale)
        draw.line((335 + i, 440, 335 + i, 440 - height), fill="#6bcea7")
    return np.asarray(canvas)


def run_episode(
    world,
    vision,
    spec,
    *,
    readout=None,
    intervention="none",
    seconds=20.0,
    record=None,
    video=False,
):
    if not 0.08 <= seconds <= 60 or not math.isclose(seconds * 25, round(seconds * 25)):
        raise ValueError("seconds must be 25Hz periods within [.08, 60]")
    if intervention not in {
        "none",
        "frozen_camera",
        "zero_vision",
        "no_body",
        "stale_camera",
    }:
        raise ValueError("unknown intervention")
    if readout is not None:
        validate_protocol(readout.metadata)
    world.reset(spec)
    vision.reset()
    gate, rows, neural, retinal, bodies, labels = MovementGate(), [], [], [], [], []
    rgb_frames = []
    first_rgb = None
    writer = (
        imageio.get_writer(str(Path(record).with_suffix(".mp4")), fps=25)
        if video
        else None
    )
    started = time.perf_counter()
    try:
        for _ in range(round(seconds * 25)):
            observation = world.observe()
            if first_rgb is None:
                first_rgb = observation.rgb.copy()
            rgb = first_rgb if intervention == "frozen_camera" else observation.rgb
            features, retina = vision.encode(rgb)
            if intervention == "zero_vision":
                features = np.zeros_like(features)
            body = (
                np.zeros_like(observation.body)
                if intervention == "no_body"
                else observation.body
            )
            # The readout call occurs before accessing privileged target/scoring information.
            requested = (
                int(readout.predict_batch(np.concatenate((features, body))[None])[0])
                if readout
                else 0
            )
            truth = world.truth()
            label = int(
                teacher_action(truth["bearing"], truth["distance"], truth["visible"])
            )
            if readout is None:
                requested = label
            if intervention == "stale_camera":
                observation = replace(observation, captured_tick=0)
            command = gate.apply(requested, observation)
            truth.update(
                requested=requested,
                applied=int(gate.action),
                command=list(command),
                capturedTick=observation.captured_tick,
            )
            rows.append(truth)
            neural.append(features)
            retinal.append(retina)
            bodies.append(body)
            labels.append(label)
            if video:
                rgb_frames.append(rgb.copy())
                writer.append_data(compose_frame(rgb, retina, features, truth))
            if gate.fault:
                world.step((0.0, 0.0, 0.0))
                break
            if not world.step(command) or not world.step(command):
                break
    finally:
        if writer:
            writer.close()
    elapsed = time.perf_counter() - started
    terminal = gate.fault or world.terminal
    result = {
        **asdict(spec),
        **score_episode(rows, terminal),
        **temporal_metrics(rows),
        "intervention": intervention,
        "wallSeconds": elapsed,
        "simulatedSeconds": world.tick * 0.02,
        "simulationToWallRatio": world.tick * 0.02 / max(elapsed, 1e-9),
    }
    # Measure actual settling under zero twist; this is separate from command acceptance.
    settled_since, settled_after = None, None
    for i in range(100):
        if not world.step((0.0, 0.0, 0.0)):
            break
        truth = world.truth()
        if truth["speed"] <= 0.02 and abs(truth["yaw_rate"]) <= 0.05:
            settled_since = i if settled_since is None else settled_since
            if i - settled_since >= 25:
                settled_after = (i + 1) * 0.02
                break
        else:
            settled_since = None
    result["endSettlementS"] = settled_after
    result["meanMeasuredSpeedMps"] = float(np.mean([r["speed"] for r in rows]))
    result["maxCameraAgeMs"] = max(
        (r["time"] - r["capturedTick"] * 0.02) * 1000 for r in rows
    )
    result["processPeakRssMiB"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )
    safety = world.safety_outcome()
    result["fallen"] |= safety["fallen"]
    result["collision"] |= safety["collision"]
    result["fault"] = result["fault"] or safety["terminalReason"]
    result["passed"] &= not (result["fallen"] or result["collision"] or result["fault"])
    result["terminalObservation"] = world.truth()
    if record:
        metadata = {
            **asdict(spec),
            "startupProtocol": STARTUP_PROTOCOL,
            "randomizationProtocol": RANDOMIZATION_PROTOCOL,
            "physicsProtocol": PHYSICS_PROTOCOL,
            "vision": vision.manifest,
            "camera": world.camera_manifest(),
            "bundleDigest": world.bundle.bundleDigest,
            "intervention": intervention,
            "readoutMetadata": readout.metadata if readout else None,
        }
        arrays = {
            "neural": np.asarray(neural, np.float32),
            "retina": np.asarray(retinal, np.float32),
            "body": np.asarray(bodies, np.float32),
            "labels": np.asarray(labels, np.int64),
            "ticks": np.array([round(r["time"] * 50) for r in rows]),
            "metadata": json.dumps(metadata, allow_nan=False),
        }
        if video:
            arrays["rgb"] = np.asarray(rgb_frames)
        np.savez_compressed(record, **arrays)
        write_json(
            Path(record).with_suffix(".json"),
            {"result": result, "metadata": metadata, "trace": rows},
        )
    return result


def validate_dataset_identity(metadata, identity, kind):
    try:
        actual, expected = metadata["vision"], identity["vision"]
        if kind == "retina":
            fields = ("extent", "kernelSize", "grayscaleWeights", "retinaSites")
            actual = {key: actual[key] for key in fields}
            expected = {key: expected[key] for key in fields}
        matches = (
            metadata["bundleDigest"] == identity["bundleDigest"] and actual == expected
        )
    except (KeyError, TypeError) as error:
        raise ValueError("dataset episode identity is missing or invalid") from error
    if not matches:
        raise ValueError("dataset episode identity mismatch")


def load_dataset(paths, kind, partition, *, identity=None):
    xs, ys = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata["partition"] != partition:
                raise ValueError("dataset partition mismatch")
            if metadata.get("intervention", "none") != "none":
                raise ValueError("intervention data cannot train a readout")
            validate_protocol(metadata)
            if identity is not None:
                validate_dataset_identity(metadata, identity, kind)
            features = data["neural"] if kind == "flyvis" else data["retina"]
            xs.append(np.concatenate((features, data["body"]), axis=1))
            ys.append(data["labels"])
    if not xs:
        raise ValueError("no dataset episodes")
    return np.concatenate(xs), np.concatenate(ys)


def collect(
    bundle, model, output, *, count=80, validation_count=20, seconds=20.0, jobs=1
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    vision = FlyvisVision(model)
    with (
        TrackingWorld(bundle) as world,
        RolloutPool(bundle, model, "flyvis", jobs) as pool,
    ):
        work = [
            {
                "spec": spec,
                "seconds": seconds,
                "record": output / f"{partition}-{spec.seed}.npz",
                "video": spec.seed % 100_000 == 0,
            }
            for partition, n in (("train", count), ("validation", validation_count))
            for spec in episode_specs(partition, n)
        ]
        for result in pool.run(work):
            print(json.dumps({"phase": "collect", **result}), flush=True)
    write_json(
        output / "dataset.json",
        {
            "schema": "MICRODUCK_TRACKING_DATASET_V1",
            "startupProtocol": STARTUP_PROTOCOL,
            "randomizationProtocol": RANDOMIZATION_PROTOCOL,
            "physicsProtocol": PHYSICS_PROTOCOL,
            "count": count,
            "validationCount": validation_count,
            "seconds": seconds,
            "vision": vision.manifest,
            "bundleDigest": world.bundle.bundleDigest,
            "provenance": provenance(),
        },
    )


def train(
    bundle,
    model,
    dataset,
    output,
    *,
    kind="flyvis",
    seeds=(1, 2, 3),
    rounds=2,
    round_episodes=40,
    epochs=50,
    jobs=1,
):
    dataset, output = Path(dataset), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    info = json.loads((dataset / "dataset.json").read_text())
    validate_protocol(info)
    validation_x, validation_y = load_dataset(
        sorted(dataset.glob("validation-*.npz")), kind, "validation", identity=info
    )
    vision = FlyvisVision(model) if kind == "flyvis" else RetinaVision()
    if kind == "flyvis" and vision.manifest != info["vision"]:
        raise ValueError("dataset neural identity differs from current model")
    validate_dataset_identity(
        info, {"bundleDigest": info["bundleDigest"], "vision": vision.manifest}, kind
    )
    with TrackingWorld(bundle) as world, RolloutPool(bundle, model, kind, jobs) as pool:
        if world.bundle.bundleDigest != info["bundleDigest"]:
            raise ValueError("dataset bundle identity mismatch")
        for seed in seeds:
            seed_dir = output / f"seed-{seed}"
            seed_dir.mkdir()
            paths = sorted(dataset.glob("train-*.npz"))
            best_score, best_path = -1.0, None
            for stage in range(rounds + 1):
                x, y = load_dataset(paths, kind, "train", identity=info)
                readout, history = fit_readout(
                    x, y, validation_x, validation_y, seed=seed, epochs=epochs
                )
                path = seed_dir / f"round-{stage}.npz"
                readout.save(
                    path,
                    {
                        "kind": kind,
                        "startupProtocol": STARTUP_PROTOCOL,
                        "randomizationProtocol": RANDOMIZATION_PROTOCOL,
                        "physicsProtocol": PHYSICS_PROTOCOL,
                        "seed": seed,
                        "stage": stage,
                        "vision": vision.manifest,
                        "bundleDigest": info["bundleDigest"],
                        "samples": len(y),
                        "datasetFiles": {
                            str(p.resolve()): file_digest(p) for p in paths
                        },
                        "validationFiles": {
                            str(p.resolve()): file_digest(p)
                            for p in sorted(dataset.glob("validation-*.npz"))
                        },
                        "history": history,
                    },
                )
                # Checkpoint selection uses closed-loop validation, never the held-out set.
                results = list(
                    pool.run(
                        [
                            {
                                "spec": spec,
                                "readout_path": path,
                                "seconds": info["seconds"],
                            }
                            for spec in episode_specs(
                                "validation", info["validationCount"]
                            )
                        ]
                    )
                )
                score = float(np.mean([r["trackingFraction"] or 0.0 for r in results]))
                write_json(seed_dir / f"round-{stage}-validation.json", results)
                if score > best_score:
                    best_score, best_path = score, path
                print(
                    json.dumps(
                        {
                            "phase": "train",
                            "kind": kind,
                            "seed": seed,
                            "stage": stage,
                            "validationTracking": score,
                            "samples": len(y),
                        }
                    ),
                    flush=True,
                )
                if stage < rounds:
                    offset = info["count"] + stage * round_episodes
                    work = [
                        {
                            "spec": spec,
                            "readout_path": path,
                            "seconds": info["seconds"],
                            "record": seed_dir / f"dagger-{stage}-{spec.seed}.npz",
                        }
                        for spec in episode_specs("train", round_episodes, offset)
                    ]
                    for result in pool.run(work):
                        print(
                            json.dumps(
                                {
                                    "phase": "dagger",
                                    "kind": kind,
                                    "readoutSeed": seed,
                                    "stage": stage,
                                    "episodeSeed": result["seed"],
                                }
                            ),
                            flush=True,
                        )
                    paths.extend(job["record"] for job in work)
            import shutil

            shutil.copyfile(best_path, seed_dir / "selected.npz")
            write_json(
                seed_dir / "selection.json",
                {"selected": best_path.name, "validationTracking": best_score},
            )


def evaluate(
    bundle,
    model,
    checkpoints,
    output,
    *,
    count=50,
    seconds=20.0,
    interventions=("none", "frozen_camera", "zero_vision", "no_body"),
    jobs=1,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "MICRODUCK_FLYVIS_TRACKING_V1",
        "startupProtocol": STARTUP_PROTOCOL,
        "randomizationProtocol": RANDOMIZATION_PROTOCOL,
        "physicsProtocol": PHYSICS_PROTOCOL,
        "runs": [],
        "criteria": {
            "bearingDegrees": 15,
            "distanceBandM": [0.4, 0.8],
            "trackingFraction": 0.8,
            "acquisitionAllowanceS": 2,
        },
        "timingHz": {"locomotion": 50, "cameraReadout": 25, "neural": 100},
        "execution": "OFFLINE_SIMULATION_TIME",
        "scientificAdvantageEstablished": False,
        "provenance": provenance(),
    }
    with TrackingWorld(bundle) as world:
        report["bundleDigest"] = world.bundle.bundleDigest
        report["camera"] = world.camera_manifest()
        report["bundleSource"] = json.loads(
            (Path(bundle) / "experiment-source.json").read_text()
        )
        for checkpoint in checkpoints:
            readout = Readout.load(checkpoint)
            validate_protocol(readout.metadata)
            kind = readout.metadata["kind"]
            vision = FlyvisVision(model) if kind == "flyvis" else RetinaVision()
            if (
                readout.metadata["bundleDigest"] != world.bundle.bundleDigest
                or readout.metadata["vision"] != vision.manifest
            ):
                raise ValueError("checkpoint executable identity mismatch")
            identity = f"{kind}-seed-{readout.metadata['seed']}"
            with RolloutPool(bundle, model, kind, jobs) as pool:
                work = []
                for intervention in interventions:
                    directory = output / identity / intervention
                    directory.mkdir(parents=True)
                    work.extend(
                        {
                            "spec": spec,
                            "readout_path": checkpoint,
                            "seconds": seconds,
                            "intervention": intervention,
                            "record": directory / f"{spec.family}-{spec.seed}.npz",
                            "video": i < 5,
                        }
                        for i, spec in enumerate(episode_specs("test", count))
                    )
                for job, result in zip(work, pool.run(work), strict=True):
                    artifact = job["record"]
                    result.update(
                        controller=identity,
                        checkpointDigest=file_digest(checkpoint),
                        artifact=str(artifact.relative_to(output)),
                        artifactDigest=file_digest(artifact),
                    )
                    report["runs"].append(result)
                    write_json(output / "report.json", report)
                    print(json.dumps({"phase": "evaluate", **result}), flush=True)
    groups = {}
    for row in report["runs"]:
        groups.setdefault((row["controller"], row["intervention"]), []).append(row)
    lines = [
        "# MicroDuck flyvis tracking results",
        "",
        "Offline simulation; performance findings are advisory.",
        "",
        "| Controller | Intervention | Passed | Mean tracking | Falls |",
        "|---|---|---:|---:|---:|",
    ]
    for (controller, intervention), rows in groups.items():
        mean = np.mean([r["trackingFraction"] or 0.0 for r in rows])
        lines.append(
            f"| {controller} | {intervention} | {sum(r['passed'] for r in rows)}/{len(rows)} | {mean:.3f} | {sum(r['fallen'] for r in rows)} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report


def replay(record, output):
    record = Path(record)
    trace = json.loads(record.with_suffix(".json").read_text())["trace"]
    with np.load(record, allow_pickle=False) as data:
        if "rgb" not in data:
            raise ValueError(
                "this episode has no camera recording; use a representative video episode"
            )
        with imageio.get_writer(str(output), fps=25) as writer:
            for rgb, retina, features, row in zip(
                data["rgb"], data["retina"], data["neural"], trace, strict=True
            ):
                writer.append_data(compose_frame(rgb, retina, features, row))
