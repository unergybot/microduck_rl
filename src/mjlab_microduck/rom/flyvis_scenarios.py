"""Motion-required obstacle experiments, isolated from the running tracking baseline.

World geometry and observer images are evidence/teacher inputs only. The learned
controller still receives the original head-camera features and 34 body channels.
"""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .contracts import PolicyBundle, sha256_prefixed, unsigned_policy_bundle_manifest
from .flyvis_tracking.core import (
    PHYSICS_PROTOCOL,
    RANDOMIZATION_PROTOCOL,
    TARGET_RNG_DOMAIN,
    Action,
    teacher_action,
)
from .flyvis_tracking.experiment import (
    STARTUP_PROTOCOL,
    compose_frame,
    run_episode,
    write_json,
)
from .flyvis_tracking.simulation import TrackingWorld, file_digest, prepare_bundle

SCENARIO_PROTOCOL = "microduck_motion_obstacles_v2"
SCENARIOS = ("approach", "turn_left", "turn_right", "obstacle_left", "obstacle_right")


def validate_scenario(metadata):
    if metadata.get("scenarioProtocol") != SCENARIO_PROTOCOL:
        raise ValueError("incompatible scenario protocol")


def prepare_scenarios(source, destination):
    destination = prepare_bundle(source, destination)
    manifest_path = destination / "microduck-policy-bundle.json"
    manifest = json.loads(manifest_path.read_text())
    scene = destination / manifest["model"]["path"]
    tree = ET.parse(scene)
    obstacle = ET.SubElement(
        tree.getroot().find("worldbody"),
        "body",
        name="tracking_obstacle",
        mocap="true",
        pos="5 5 .14",
    )
    ET.SubElement(
        obstacle,
        "geom",
        name="tracking_obstacle_geom",
        type="box",
        size=".08 .10 .14",
        rgba=".8 .35 .08 1",
        contype="3",
        conaffinity="3",
    )
    tree.write(scene, encoding="unicode")
    artifacts = [
        manifest["model"],
        *manifest["policies"],
        *manifest["qualification"].get("artifacts", []),
        *manifest["qualification"].get("modelClosure", []),
        *manifest["license"]["artifacts"],
    ]
    digests = {}
    for artifact in artifacts:
        artifact["digest"] = file_digest(destination / artifact["path"])
        digests[artifact["path"]] = artifact["digest"]
    manifest["bundleVersion"] += "-obstacles"
    candidate = PolicyBundle.model_validate(manifest)
    manifest["bundleDigest"] = sha256_prefixed(
        {
            "manifest": unsigned_policy_bundle_manifest(candidate).model_dump(
                mode="json", by_alias=True
            ),
            "artifacts": digests,
        }
    )
    write_json(manifest_path, manifest)
    source_info = json.loads((destination / "experiment-source.json").read_text())
    source_info.update(
        scenarioProtocol=SCENARIO_PROTOCOL,
        obstacle="physical box, half extents .08 .10 .14 m",
    )
    write_json(destination / "experiment-source.json", source_info)
    return destination


class ScenarioWorld(TrackingWorld):
    def __init__(self, bundle_root):
        super().__init__(bundle_root)
        self.obstacle_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "tracking_obstacle_geom"
        )
        if self.obstacle_id < 0:
            self.close()
            raise ValueError("scenario bundle requires a physical obstacle")
        body_id = self.model.geom_bodyid[self.obstacle_id]
        self.obstacle_mocap = self.model.body_mocapid[body_id]
        self.robot_geoms = {
            i
            for i, body in enumerate(self.model.geom_bodyid)
            if body != 0 and self.model.body_mocapid[body] < 0
        }
        self.observer = mujoco.Renderer(self.model, height=480, width=640)
        self.record_observer = False
        self.observer_frames = []

    def reset(self, spec):
        if spec.family not in SCENARIOS:
            raise ValueError("unknown motion scenario")
        recording = self.record_observer
        super().reset(spec)
        self.record_observer = recording
        self.observer_frames = []
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([spec.seed, TARGET_RNG_DOMAIN]))
        )
        angle = rng.uniform(-0.08, 0.08)
        if spec.family in ("turn_left", "turn_right"):
            angle = rng.uniform(0.40, 0.55) * (1 if spec.family == "turn_left" else -1)
        distance = rng.uniform(1.1, 1.3)
        self.target_start = np.array(
            [distance * math.cos(angle), distance * math.sin(angle), 0.30]
        )
        self.target_start[:2] += self.origin[:2]
        self.obstacle_active = spec.family.startswith("obstacle_")
        side = 1 if spec.family == "obstacle_left" else -1
        obstacle = np.array([0.48, side * rng.uniform(0.09, 0.13), 0.14])
        obstacle[:2] += self.origin[:2]
        self.obstacle_position = obstacle
        self.detour_side = -side
        clear_y = obstacle[1] + self.detour_side * 0.42
        self.route = (
            [[obstacle[0] - 0.22, clear_y], [obstacle[0] + 0.24, clear_y]]
            if self.obstacle_active
            else []
        )
        self.route_index = 0
        self.data.mocap_pos[self.obstacle_mocap] = (
            obstacle if self.obstacle_active else [5, 5, 0.14]
        )
        self.update_target()

    def external_view(self):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [self.origin[0] + 0.55, self.origin[1], 0.18]
        camera.distance, camera.azimuth, camera.elevation = 2.0, 125, -25
        self.observer.update_scene(self.rendering_state(), camera=camera)
        return self.observer.render().copy()

    def observe(self):
        observation = super().observe()
        if self.record_observer:
            self.observer_frames.append(self.external_view())
        return observation

    def obstacle_contact(self):
        return any(
            (c.geom1 == self.obstacle_id and c.geom2 in self.robot_geoms)
            or (c.geom2 == self.obstacle_id and c.geom1 in self.robot_geoms)
            for c in self.data.contact
        )

    def truth(self):
        row = super().truth()
        row["collision"] |= self.obstacle_contact()
        row["waypoint"] = None
        if getattr(self, "obstacle_active", False):
            ox, oy = self.obstacle_position[:2]
            # Privileged teacher route only. Never appended to learner observations.
            if self.route_index < len(self.route):
                row["waypoint"] = self.route[self.route_index]
            dx = max(abs(row["x"] - ox) - 0.08, 0.0)
            dy = max(abs(row["y"] - oy) - 0.10, 0.0)
            row["obstacleBaseClearanceM"] = math.hypot(dx, dy)
        else:
            row["obstacleBaseClearanceM"] = None
        return row

    def step(self, command):
        self.safety["collision"] |= self.obstacle_contact()
        running = super().step(command)
        if self.route_index < len(self.route):
            position = self.runtime._base_position()[:2]
            if np.linalg.norm(position - self.route[self.route_index]) < 0.10:
                self.route_index += 1
        self.safety["collision"] |= self.obstacle_contact()
        return running

    def close(self):
        if hasattr(self, "observer"):
            self.observer.close()
        super().close()


def scenario_teacher(row):
    waypoint = row.get("waypoint")
    if waypoint is not None:
        delta = np.asarray(waypoint) - [row["x"], row["y"]]
        bearing = math.atan2(delta[1], delta[0]) - row["yaw"]
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        if abs(bearing) > math.radians(6):
            return Action.TURN_LEFT if bearing > 0 else Action.TURN_RIGHT
        return Action.ADVANCE
    # This privileged reference knows the target exists even when temporarily occluded.
    return teacher_action(row["bearing"], row["distance"], True)


def score_motion(rows, fault, *, final=None):
    first, last = rows[0], final if final is not None else rows[-1]
    progress = first["distance"] - last["distance"]
    displacement = math.hypot(last["x"] - first["x"], last["y"] - first["y"])
    collision = any(row["collision"] for row in rows)
    fallen = any(row["fallen"] for row in rows)
    reached = (
        last["visible"]
        and 0.4 <= last["distance"] <= 0.8
        and abs(last["bearing"]) <= math.radians(15)
    )
    return {
        "distanceReductionM": progress,
        "displacementM": displacement,
        "reachedTarget": bool(reached),
        "collision": collision,
        "fallen": fallen,
        "passed": bool(
            reached
            and progress >= 0.2
            and displacement >= 0.2
            and not collision
            and not fallen
            and not fault
        ),
    }


class GeometryTeacher:
    """Explicit privileged training reference, never presented as a learned controller."""

    def __init__(self, world):
        self.world = world
        self.metadata = {
            "startupProtocol": STARTUP_PROTOCOL,
            "randomizationProtocol": RANDOMIZATION_PROTOCOL,
            "physicsProtocol": PHYSICS_PROTOCOL,
            "scenarioProtocol": SCENARIO_PROTOCOL,
            "kind": "geometry_teacher",
        }

    def predict_batch(self, _sensors):
        return np.array([scenario_teacher(self.world.truth())])


def scenario_episode(
    world,
    vision,
    spec,
    record,
    *,
    readout=None,
    seconds=20.0,
    video=False,
    intervention="none",
):
    record = Path(record)
    if readout is not None:
        validate_scenario(readout.metadata)
        if (
            readout.metadata["bundleDigest"] != world.bundle.bundleDigest
            or readout.metadata["vision"] != vision.manifest
        ):
            raise ValueError("scenario checkpoint executable identity mismatch")
    controller = readout if readout is not None else GeometryTeacher(world)
    world.record_observer = video
    result = run_episode(
        world,
        vision,
        spec,
        readout=controller,
        seconds=seconds,
        record=record,
        video=video,
        intervention=intervention,
    )
    episode = json.loads(record.with_suffix(".json").read_text())
    rows = episode["trace"]
    # Refresh visibility from the settled physics state without appending a video frame.
    world.record_observer = False
    world.observe()
    final = world.truth()
    result.update(score_motion(rows, result["fault"], final=final))
    result["terminalObservation"] = final
    result["finalMeasurement"] = "fresh observation after zero-command settling attempt"
    result["finalMeasurementTimeS"] = final["time"]
    safety = world.safety_outcome()
    result["collision"] |= safety["collision"]
    result["fallen"] |= safety["fallen"]
    result["passed"] &= not (result["collision"] or result["fallen"])
    result["controllerRole"] = (
        "learned_sensor_readout" if readout else "privileged_geometry_teacher"
    )
    with np.load(record, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    meta = json.loads(str(arrays["metadata"]))
    meta.update(
        scenarioProtocol=SCENARIO_PROTOCOL, controllerRole=result["controllerRole"]
    )
    arrays["metadata"] = json.dumps(meta)
    arrays["labels"] = np.array([scenario_teacher(row) for row in rows], dtype=np.int64)
    if video:
        arrays["observer_rgb"] = np.asarray(world.observer_frames, dtype=np.uint8)
    np.savez_compressed(record, **arrays)
    episode.update(result=result, metadata=meta)
    write_json(record.with_suffix(".json"), episode)
    if video:
        replay_scenario(record, record.with_suffix(".mp4"))
    return result


def replay_scenario(record, output):
    record = Path(record)
    episode = json.loads(record.with_suffix(".json").read_text())
    validate_scenario(episode["metadata"])
    role = episode["metadata"]["controllerRole"]
    with np.load(record, allow_pickle=False) as data:
        if "observer_rgb" not in data:
            raise ValueError("episode has no synchronized outside camera recording")
        with imageio.get_writer(str(output), fps=25) as writer:
            for outside, rgb, retina, features, row in zip(
                data["observer_rgb"],
                data["rgb"],
                data["retina"],
                data["neural"],
                episode["trace"],
                strict=True,
            ):
                frame = Image.fromarray(
                    np.concatenate(
                        (outside, compose_frame(rgb, retina, features, row)), axis=1
                    )
                )
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 639, 36), fill=(20, 25, 30))
                draw.text((10, 5), "MUJOCO OUTSIDE VIEW - REVIEW ONLY", fill="white")
                draw.text((10, 20), role, fill="white")
                writer.append_data(np.asarray(frame))
