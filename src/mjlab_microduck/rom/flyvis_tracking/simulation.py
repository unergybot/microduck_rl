"""Offline adapter around the verified ROM locomotion implementation."""

import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from ..contracts import (
    PolicyBundle,
    TaskCreateRequest,
    sha256_prefixed,
    unsigned_policy_bundle_manifest,
)
from ..main import _bundle_artifacts, load_verified_bundle
from ..mujoco_runtime import MicroduckMujocoRuntime
from ..observation import project_gravity_wxyz
from .core import TARGET_RNG_DOMAIN, SensorFrame


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def prepare_bundle(source, destination):
    """Copy declared artifacts only, add a visible target, and rebind all hashes."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError("experiment bundle must be outside source bundle")
    bundle = load_verified_bundle(source)
    destination.mkdir(parents=True, exist_ok=False)
    for artifact in _bundle_artifacts(bundle):
        dest = destination / artifact.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / artifact.path, dest)
    camera_correction = None
    for xml in sorted((destination / "models").glob("*.xml")):
        camera_tree = ET.parse(xml)
        camera = camera_tree.find(".//camera[@name='head_camera']")
        if camera is not None:
            # Exported optical frame looks back into the lens and has sideways up.
            # Compose a camera-local rotation: new right=-old up, new up=-old right.
            original = np.fromstring(camera.get("quat", "1 0 0 0"), sep=" ")
            corrected = np.zeros(4)
            mujoco.mju_mulQuat(
                corrected, original, np.array([0.0, 2**-0.5, -(2**-0.5), 0.0])
            )
            camera.set("quat", " ".join(map(str, corrected)))
            camera_tree.write(xml, encoding="unicode")
            camera_correction = {
                "originalQuaternion": original.tolist(),
                "experimentQuaternion": corrected.tolist(),
                "reason": "exported camera faces into head; correct optical axes only",
            }
    scene_path = destination / bundle.model.path
    tree = ET.parse(scene_path)
    root = tree.getroot()
    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")
    target = ET.SubElement(
        world, "body", name="tracking_target", mocap="true", pos=".8 0 .3"
    )
    ET.SubElement(
        target,
        "geom",
        name="tracking_target_geom",
        type="sphere",
        size=".08",
        rgba=".03 .03 .03 1",
        contype="3",
        conaffinity="3",
    )
    tree.write(scene_path, encoding="unicode")
    manifest = bundle.model_dump(mode="json", by_alias=True)
    manifest["bundleVersion"] += "-flyvis"
    artifacts = [
        manifest["model"],
        *manifest["policies"],
        *manifest["qualification"].get("artifacts", []),
        *manifest["qualification"].get("modelClosure", []),
        *manifest["license"]["artifacts"],
    ]
    digests = {}
    for item in artifacts:
        item["digest"] = file_digest(destination / item["path"])
        digests[item["path"]] = item["digest"]
    candidate = PolicyBundle.model_validate(manifest)
    manifest["bundleDigest"] = sha256_prefixed(
        {
            "manifest": unsigned_policy_bundle_manifest(candidate).model_dump(
                mode="json", by_alias=True
            ),
            "artifacts": digests,
        }
    )
    (destination / "microduck-policy-bundle.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (destination / "experiment-source.json").write_text(
        json.dumps(
            {
                "sourceBundleDigest": bundle.bundleDigest,
                "sourceModelDigest": bundle.model.digest,
                "sceneAddition": "tracking_target sphere radius .08m, mocap",
                "cameraCorrection": camera_correction,
            },
            indent=2,
        )
        + "\n"
    )
    load_verified_bundle(destination)
    return destination


class TrackingWorld:
    """Owns one offline runtime. World truth is deliberately a separate method."""

    def __init__(self, bundle_root):
        self.root = Path(bundle_root)
        self.bundle = load_verified_bundle(self.root)
        self.runtime = MicroduckMujocoRuntime(self.root, self.bundle, realtime=False)
        self.model, self.data = self.runtime._model, self.runtime._data
        self.render_data = mujoco.MjData(self.model)
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera"
        )
        self.target_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "tracking_target_geom"
        )
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "tracking_target"
        )
        if min(self.camera_id, self.target_id, body_id) < 0:
            self.close()
            raise ValueError("experiment requires head_camera and tracking target")
        self.mocap_id = self.model.body_mocapid[body_id]
        self.renderer = mujoco.Renderer(self.model, height=240, width=320)
        self.handle = None
        self.tick = 0
        self.visible = False
        self.terminal = None

    def reset(self, spec):
        if self.runtime._fatal_reason is not None:
            # Runtime faults are intentionally latched. A new episode owns a new
            # isolated runtime rather than clearing a live runtime's safety state.
            self.close()
            self.__init__(self.root)
        if self.handle is not None:
            self.runtime.safe_stop(self.handle, "EPISODE_END")
        self.spec, self.tick, self.terminal = spec, 0, None
        self.visible = False
        self.safety = {"fallen": False, "collision": False}
        request = TaskCreateRequest(
            schema="MICRODUCK_SIM_TASK_V1",
            taskId=f"{spec.seed:032x}",
            actionCode="WALK_VELOCITY",
            bundleVersion=self.bundle.bundleVersion,
            bundleDigest=self.bundle.bundleDigest,
            parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
            scenario={"terrain": "flat", "seed": spec.seed},
            leaseMs=1000,
            requestedBy="offline-flyvis-tracking",
        )
        action = next(a for a in self.bundle.actions if a.actionCode == "WALK_VELOCITY")
        self.handle = self.runtime.start(action, request)
        # Domain separation prevents target geometry from encoding reset joint noise.
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([spec.seed, TARGET_RNG_DOMAIN]))
        )
        angle = rng.uniform(-0.3, 0.3)
        distance = rng.uniform(0.65, 0.95)
        self.origin = self.runtime._base_position().copy()
        yaw = self.runtime._yaw_rad()
        self.target_start = np.array(
            [
                self.origin[0] + distance * math.cos(yaw + angle),
                self.origin[1] + distance * math.sin(yaw + angle),
                0.30,
            ]
        )
        self.update_target()

    def update_target(self):
        t = self.tick * 0.02
        position = self.target_start.copy()
        if self.spec.family == "lateral":
            position[1] += 0.06 * math.sin(t / 6)
        if self.spec.family == "receding":
            position[0] += 0.01 * t
        if self.spec.family == "disappearance" and 8 <= t < 10:
            position[2] = -2.0
        self.data.mocap_pos[self.mocap_id] = position

    def rendering_state(self):
        # Refresh camera geometry on a copy. Running forward dynamics on the live
        # data changes the next locomotion observation/solver state.
        mujoco.mj_copyData(self.render_data, self.model, self.data)
        mujoco.mj_forward(self.model, self.render_data)
        return self.render_data

    def observe(self):
        render_data = self.rendering_state()
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(render_data, camera=self.camera_id)
        rgb = self.renderer.render().copy()
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(render_data, camera=self.camera_id)
        segmentation = self.renderer.render()
        self.visible = bool(
            np.count_nonzero(
                (segmentation[..., 0] == self.target_id)
                & (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            )
            >= 5
        )
        self.renderer.disable_segmentation_rendering()
        r = self.runtime
        body = np.concatenate(
            (
                r._base_angular_velocity(),
                project_gravity_wxyz(r._base_quaternion_wxyz()),
                r._encoder_positions(),
                r._encoder_velocities(),
            )
        ).astype(np.float32)
        return SensorFrame(self.tick, self.tick, rgb, body)

    def truth(self):
        r = self.runtime
        position = r._base_position()
        delta = self.data.mocap_pos[self.mocap_id, :2] - position[:2]
        yaw = r._yaw_rad()
        bearing = math.atan2(
            math.sin(math.atan2(delta[1], delta[0]) - yaw),
            math.cos(math.atan2(delta[1], delta[0]) - yaw),
        )
        collisions = any(
            self.target_id in (c.geom1, c.geom2) for c in self.data.contact
        )
        velocity = self.data.qvel[r._free_qvel_address : r._free_qvel_address + 2]
        return {
            "time": self.tick * 0.02,
            "visible": self.visible,
            "bearing": bearing,
            "distance": float(np.linalg.norm(delta)),
            "yaw": yaw,
            "x": float(position[0]),
            "y": float(position[1]),
            "fallen": bool(r._fallen),
            "collision": bool(collisions),
            "speed": float(np.linalg.norm(velocity)),
            "yaw_rate": float(r._base_angular_velocity()[2]),
        }

    def step(self, command):
        self.runtime.command(
            self.handle,
            dict(zip(("vxMps", "vyMps", "yawRateRadps"), command, strict=True)),
        )
        trunk = self.model.body("trunk_base").id
        self.data.xfrc_applied[trunk] = 0
        if self.spec.family == "perturbation" and 400 <= self.tick < 405:
            self.data.xfrc_applied[trunk, 5] = 0.02
        sample = self.runtime.sample(self.handle)
        self.safety["fallen"] |= (
            bool(self.runtime._fallen) or sample.stopReason == "FALLEN"
        )
        self.safety["collision"] |= any(
            self.target_id in (c.geom1, c.geom2) for c in self.data.contact
        )
        self.tick += 1
        self.update_target()
        self.terminal = sample.stopReason if not sample.running else None
        return sample.running

    def safety_outcome(self):
        return {**self.safety, "terminalReason": self.terminal}

    def camera_manifest(self):
        return {
            "name": "head_camera",
            "width": 320,
            "height": 240,
            "fovy": float(self.model.cam_fovy[self.camera_id]),
            "localPosition": self.model.cam_pos[self.camera_id].tolist(),
            "localQuaternion": self.model.cam_quat[self.camera_id].tolist(),
        }

    def close(self):
        try:
            if getattr(self, "handle", None) is not None:
                self.runtime.safe_stop(self.handle, "EXPERIMENT_END")
                self.handle = None
        finally:
            if hasattr(self, "renderer"):
                self.renderer.close()
            if hasattr(self, "runtime"):
                self.runtime._snapshot.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
