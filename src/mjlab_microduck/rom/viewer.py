"""Bounded read-only display data; never advances or resets the simulator."""

from __future__ import annotations

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
            va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
            # Conservative allocation bound before materializing Python lists.
            if (vn + fn) * 3 > MODEL_MAX_BYTES // 2:
                raise ValueError("viewer mesh exceeds bound")
            geom["vertices"] = finite_list(model.mesh_vert[va : va + vn])
            geom["indices"] = model.mesh_face[fa : fa + fn].reshape(-1).tolist()
            if any(index < 0 or index >= vn for index in geom["indices"]):
                raise ValueError("viewer mesh index out of range")
        estimated += len(encode_display(geom))
        if estimated > MODEL_MAX_BYTES - 1024:
            raise ValueError("viewer geometry exceeds bound")
        geometry.append(geom)
    return geometry


class RuntimeViewer:
    def __init__(self, model, bundle):
        self.identity = dict(
            runtimeSession=uuid4().hex,
            modelDigest=bundle.model.digest,
            bundleDigest=bundle.bundleDigest,
        )
        self.model = dict(
            schema="MICRODUCK_VIEWER_MODEL_V1",
            **self.identity,
            geometry=export_geometry(model),
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
