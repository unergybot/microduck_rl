"""Bounded read-only display data; never advances or resets the simulator."""

from __future__ import annotations

import base64
import json
import math
from datetime import UTC, datetime
from uuid import uuid4

MODEL_MAX_BYTES = 8 * 1024 * 1024
FRAME_MAX_BYTES = 262144
CHUNK_CHARS = 24000
MAX_GEOMS = 512


def encode_display(value, limit=MODEL_MAX_BYTES):
    text = json.dumps(value, separators=(",", ":"), allow_nan=False)
    if len(text.encode("utf-8")) > limit:
        raise ValueError("viewer data exceeds bound")
    return text


def finite_list(array):
    values = array.reshape(-1).tolist()
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite viewer data")
    return values


def export_geometry(model):
    if not 0 < model.ngeom <= MAX_GEOMS:
        raise ValueError("viewer geometry count exceeds bound")
    geometry = []
    estimated = 0
    for index in range(model.ngeom):
        if int(model.geom_group[index]) == 3:
            continue
        kind = int(model.geom_type[index])
        if kind not in {0, 2, 3, 4, 5, 6, 7}:
            raise ValueError("unsupported viewer geometry")
        color = model.geom_rgba[index]
        material = int(model.geom_matid[index])
        if material >= 0 and list(color) == [0.5, 0.5, 0.5, 1.0]:
            color = model.mat_rgba[material]
        geom = dict(
            id=index,
            type=kind,
            size=finite_list(model.geom_size[index]),
            rgba=finite_list(color),
        )
        if kind == 7:
            mesh = int(model.geom_dataid[index])
            if not 0 <= mesh < min(model.nmesh, MAX_GEOMS):
                raise ValueError("viewer mesh id out of range")
            geom["meshId"] = mesh
        estimated += len(encode_display(geom))
        if estimated > MODEL_MAX_BYTES - 1024:
            raise ValueError("viewer geometry exceeds bound")
        geometry.append(geom)
    return geometry


def export_meshes(model, geometry):
    # Native dependencies stay in the model-owning child, not the supervisor.
    import numpy as np

    meshes = []
    estimated = 0
    for mesh in sorted({g["meshId"] for g in geometry if g["type"] == 7}):
        va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        if not (9 <= vn * 3 <= 1_500_000 and 0 < fn * 3 <= 1_500_000):
            raise ValueError("viewer mesh exceeds component bound")
        if (
            va < 0
            or fa < 0
            or va + vn > len(model.mesh_vert)
            or fa + fn > len(model.mesh_face)
        ):
            raise ValueError("viewer mesh slice out of range")
        vertices = model.mesh_vert[va : va + vn]
        faces = model.mesh_face[fa : fa + fn]
        if not np.isfinite(vertices).all() or (np.abs(vertices) > 1e6).any():
            raise ValueError("invalid viewer mesh coordinate")
        if faces.min() < 0 or faces.max() >= vn:
            raise ValueError("viewer mesh index out of range")
        index_type = "uint16" if vn <= 65536 else "uint32"
        index_bytes = 2 if index_type == "uint16" else 4
        estimated += 4 * ((vn * 12 + 2) // 3) + 4 * ((fn * 3 * index_bytes + 2) // 3)
        if estimated > MODEL_MAX_BYTES:
            raise ValueError("viewer meshes exceed bound")
        meshes.append(
            dict(
                id=mesh,
                verticesBase64=base64.b64encode(
                    vertices.astype("<f4", copy=False).tobytes()
                ).decode("ascii"),
                indicesBase64=base64.b64encode(
                    faces.astype("<u2" if index_bytes == 2 else "<u4").tobytes()
                ).decode("ascii"),
                indexType=index_type,
            )
        )
    return meshes


def robot_geometry_ids(model, geometry, trunk_body_id):
    if not 0 < trunk_body_id < model.nbody:
        raise ValueError("invalid viewer robot root")
    result = []
    for geom in geometry:
        body = int(model.geom_bodyid[geom["id"]])
        for _ in range(model.nbody):
            if body == trunk_body_id:
                result.append(geom["id"])
                break
            if body == 0:
                break
            if not 0 < body < model.nbody:
                raise ValueError("invalid viewer body ancestry")
            body = int(model.body_parentid[body])
        else:
            raise ValueError("cyclic viewer body ancestry")
    return result


class RuntimeViewer:
    def __init__(self, model, bundle, *, trunk_body_id):
        self.identity = dict(
            runtimeSession=uuid4().hex,
            modelDigest=bundle.model.digest,
            bundleDigest=bundle.bundleDigest,
        )
        geometry = export_geometry(model)
        self.model = dict(
            schema="MICRODUCK_VIEWER_MODEL_V2",
            **self.identity,
            geometry=geometry,
            meshes=export_meshes(model, geometry),
            robotGeometryIds=robot_geometry_ids(model, geometry, trunk_body_id),
        )
        self.model_text = encode_display(self.model)
        self.frame = None
        self.sampled_at = float("-inf")
        self.sequence = 0

    def sample(self, data, task_id, now):
        if self.frame is not None and now - self.sampled_at < 0.2:
            return self.frame
        frame = dict(
            schema="MICRODUCK_VIEWER_FRAME_V1",
            **self.identity,
            capturedAt=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            sequence=self.sequence,
            activeTaskId=task_id,
            positions=finite_list(data.geom_xpos),
            matrices=finite_list(data.geom_xmat),
        )
        encode_display(frame, FRAME_MAX_BYTES)
        self.frame, self.sampled_at = frame, now
        self.sequence += 1
        return frame
