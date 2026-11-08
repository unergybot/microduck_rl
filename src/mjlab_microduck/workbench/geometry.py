"""Private bounded projection of a compiled MuJoCo scene.

Display IDs are dense slots, NOT MuJoCo geom IDs. This payload needs new V2
consumers: existing observer/evaluation/viewer wire formats are unchanged.
See GEOMETRY.md for encoding, fidelity and integration limitations.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

MAX_SOURCE_GEOMS = 32768
MAX_SLOTS = 256
MAX_VERTICES = 1_000_000
MAX_TRIANGLES = 2_000_000
MAX_COMPONENTS = 1_500_000
MAX_GEOMETRY_BYTES = 64 * 1024 * 1024
MAX_FRAME_BYTES = 262144
CHUNK_METERS = 16.0


class ProjectionError(ValueError):
    """Stable private refusal code; never a partial geometry fallback."""


def _require(condition, code):
    if not condition:
        raise ProjectionError(code)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite(value, code="NONFINITE_GEOMETRY"):
    _require(np.isfinite(value).all() and (np.abs(value) <= 1e6).all(), code)
    return value


def _buffer(value, dtype):
    return base64.b64encode(np.asarray(value, dtype=dtype).tobytes()).decode("ascii")


def _mesh(vertices, faces, mesh_id, colors=None):
    _finite(vertices)
    _require(
        len(vertices) * 3 <= MAX_COMPONENTS and len(faces) * 3 <= MAX_COMPONENTS,
        "MESH_COMPONENT_LIMIT",
    )
    _require(
        len(faces) > 0 and faces.min() >= 0 and faces.max() < len(vertices),
        "MESH_INDEX",
    )
    wide = len(vertices) > 65536
    result = {
        "id": mesh_id,
        "verticesBase64": _buffer(vertices, "<f4"),
        "indicesBase64": _buffer(faces, "<u4" if wide else "<u2"),
        "indexType": "uint32" if wide else "uint16",
    }
    if colors is not None:
        result["colorsBase64"] = _buffer(colors, "<f4")
    return result


@dataclass(frozen=True)
class GeometryProjection:
    """Immutable geometry/mapping plus a bounded two-device gather-index cache.

    export() returns a detached JSON copy. capture() returns fresh packed CPU
    buffers. A mapping digest covers geometry, root, all source membership and
    constant poses, so display identities cannot alias another projection.
    """

    geometry_json: str
    source_geom_count: int
    mapping_digest: str
    dynamic_mapping: tuple[tuple[int, int], ...]
    source_membership: tuple[tuple[int, ...], ...]
    _positions: tuple[tuple[float, ...], ...]
    _matrices: tuple[tuple[float, ...], ...]
    _device_cache: OrderedDict = field(
        default_factory=OrderedDict, compare=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, compare=False, repr=False
    )

    def export(self):
        return json.loads(self.geometry_json)

    def capture(self, positions, matrices, *, environment_indices):
        """Gather actual [env, sourceGeom, 3/3x3] poses before any CPU transfer.

        Float32 packed poses retain the source sample to float32 precision;
        static slots use precomputed world transforms. No model/data stepping,
        reset, mutation, full-source clone, or implicit environment selection.
        """
        _require(
            isinstance(environment_indices, (list, tuple))
            and 1 <= len(environment_indices) <= 9,
            "ENVIRONMENT_INDICES",
        )
        _require(
            all(type(i) is int for i in environment_indices)
            and len(set(environment_indices)) == len(environment_indices),
            "ENVIRONMENT_INDICES",
        )
        _require(
            len(positions.shape) == 3
            and tuple(positions.shape[1:]) == (self.source_geom_count, 3),
            "POSE_SHAPE",
        )
        _require(
            tuple(matrices.shape) == (positions.shape[0], self.source_geom_count, 3, 3),
            "POSE_SHAPE",
        )
        _require(
            all(0 <= i < positions.shape[0] for i in environment_indices),
            "ENVIRONMENT_INDICES",
        )
        slots = len(self._positions)
        count = len(environment_indices)
        # Fixed-width frame preflight occurs before gather/output allocation.
        _require(
            4 * ((count * slots * 12 * 4 + 2) // 3) + 4096 <= MAX_FRAME_BYTES,
            "FRAME_BUDGET",
        )
        sources = [g for _, g in self.dynamic_mapping]
        if isinstance(positions, np.ndarray):
            _require(
                isinstance(matrices, np.ndarray)
                and positions.dtype.kind == "f"
                and matrices.dtype.kind == "f",
                "POSE_TYPE",
            )
            selected_p = positions[np.asarray(environment_indices)[:, None], sources]
            selected_m = matrices[np.asarray(environment_indices)[:, None], sources]
        else:
            import torch

            _require(
                isinstance(positions, torch.Tensor)
                and isinstance(matrices, torch.Tensor),
                "POSE_TYPE",
            )
            _require(
                positions.device == matrices.device
                and positions.is_floating_point()
                and matrices.is_floating_point(),
                "POSE_TYPE",
            )
            key = str(positions.device)
            with self._lock:
                if key not in self._device_cache:
                    self._device_cache[key] = torch.tensor(
                        sources, dtype=torch.long, device=positions.device
                    )
                    while len(self._device_cache) > 2:
                        self._device_cache.popitem(last=False)
                source_index = self._device_cache[key]
                self._device_cache.move_to_end(key)
            env_index = torch.tensor(
                environment_indices, dtype=torch.long, device=positions.device
            )
            # Combined advanced indexing selects BOTH axes before detach/CPU.
            selected_p = (
                positions[env_index[:, None], source_index]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .numpy()
            )
            selected_m = (
                matrices[env_index[:, None], source_index]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .numpy()
            )
        return self.pack_selected(
            selected_p, selected_m, environment_indices=environment_indices,
            environment_count=positions.shape[0],
        )

    def pack_selected(self, positions, matrices, *, environment_indices, environment_count):
        """Pack already gathered, completed CPU buffers in dynamic_mapping order.

        Trusted callers must gather actual source/env indices before transfer and
        complete asynchronous copies before calling. This helper never transfers
        from a GPU or reconstructs full-source arrays. Constants remain immutable.
        """
        _require(
            type(environment_count) is int and environment_count > 0
            and isinstance(environment_indices, (list, tuple))
            and 1 <= len(environment_indices) <= 9
            and all(type(i) is int and 0 <= i < environment_count for i in environment_indices)
            and len(set(environment_indices)) == len(environment_indices),
            "ENVIRONMENT_INDICES",
        )
        count = len(environment_indices)
        slots = len(self._positions)
        _require(
            4 * ((count * slots * 12 * 4 + 2) // 3) + 4096 <= MAX_FRAME_BYTES,
            "FRAME_BUDGET",
        )
        dynamic_slots = [s for s, _ in self.dynamic_mapping]
        _require(
            tuple(positions.shape) == (count, len(dynamic_slots), 3)
            and tuple(matrices.shape) == (count, len(dynamic_slots), 3, 3),
            "POSE_SHAPE",
        )
        selected = []
        for value in (positions, matrices):
            if isinstance(value, np.ndarray):
                _require(value.dtype.kind == "f", "POSE_TYPE")
            else:
                import torch

                _require(
                    isinstance(value, torch.Tensor) and value.device.type == "cpu"
                    and value.is_floating_point(), "POSE_TYPE",
                )
                value = value.detach().to(dtype=torch.float32).numpy()
            selected.append(_finite(value, "NONFINITE_POSE"))
        selected_p, selected_m = selected
        out_p = np.broadcast_to(
            np.asarray(self._positions, dtype="<f4"), (count, slots, 3)
        ).copy()
        out_m = np.broadcast_to(
            np.asarray(self._matrices, dtype="<f4").reshape(slots, 3, 3),
            (count, slots, 3, 3),
        ).copy()
        out_p[:, dynamic_slots] = selected_p
        out_m[:, dynamic_slots] = selected_m
        result = {
            "mappingDigest": self.mapping_digest,
            "environmentIndices": list(environment_indices),
            "slotCount": slots,
            "positionsBase64": _buffer(out_p, "<f4"),
            "matricesBase64": _buffer(out_m, "<f4"),
            "poseEncoding": "float32-le",
        }
        _require(len(_json(result).encode()) <= MAX_FRAME_BYTES, "FRAME_BUDGET")
        return result


def _body_transforms(model, root):
    import mujoco

    _require(0 < model.nbody <= 65536, "BODY_LIMIT")
    _require(type(root) is int and 0 < root < model.nbody, "ROBOT_ROOT")
    world_p = [np.zeros(3)]
    world_r = [np.eye(3)]
    dynamic = [False]
    focus = [False]
    for body in range(1, model.nbody):
        parent = int(model.body_parentid[body])
        _require(0 <= parent < body, "BODY_ANCESTRY")
        local_r = np.empty(9)
        _finite(model.body_quat[body])
        _finite(model.body_pos[body])
        mujoco.mju_quat2Mat(local_r, model.body_quat[body])
        local_r = local_r.reshape(3, 3)
        world_p.append(world_p[parent] + world_r[parent] @ model.body_pos[body])
        world_r.append(world_r[parent] @ local_r)
        dynamic.append(
            dynamic[parent]
            or int(model.body_dofnum[body]) > 0
            or int(model.body_mocapid[body]) >= 0
        )
        focus.append(body == root or focus[parent])
    return world_p, world_r, dynamic, focus


_BOX_VERTS = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    float,
)
_BOX_FACES = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ],
    np.int32,
)


def _heightfield(model, index):
    nr = int(model.hfield_nrow[index])
    nc = int(model.hfield_ncol[index])
    start = int(model.hfield_adr[index])
    _require(nr >= 2 and nc >= 2, "HFIELD_SHAPE")
    perimeter = 2 * nr + 2 * nc - 4
    return (
        nr,
        nc,
        start,
        nr * nc + perimeter,
        2 * (nr - 1) * (nc - 1) + 2 * perimeter + 2,
    )


def _hfield_mesh(model, index):
    nr, nc, start, _, _ = _heightfield(model, index)
    size = model.hfield_size[index]
    data = model.hfield_data[start : start + nr * nc]
    _finite(size)
    _finite(data)
    _require(
        len(data) == nr * nc
        and (size > 0).all()
        and (data >= 0).all()
        and (data <= 1).all(),
        "HFIELD_DATA",
    )
    x, y = np.meshgrid(
        np.linspace(-size[0], size[0], nc), np.linspace(-size[1], size[1], nr)
    )
    vertices = np.column_stack((x.ravel(), y.ravel(), data * size[2]))
    a = (np.arange(nr - 1)[:, None] * nc + np.arange(nc - 1)).ravel()
    # MuJoCo's diagonal goes from (row,col) to (row+1,col+1).
    faces = np.concatenate(
        (
            np.column_stack((a, a + 1, a + nc + 1)),
            np.column_stack((a, a + nc + 1, a + nc)),
        )
    )
    border = np.concatenate(
        (
            np.arange(nc),
            np.arange(1, nr) * nc + nc - 1,
            (nr - 1) * nc + np.arange(nc - 2, -1, -1),
            np.arange(nr - 2, 0, -1) * nc,
        )
    )
    bottom = vertices[border].copy()
    bottom[:, 2] = -size[3]
    count = len(border)
    b = np.arange(count) + len(vertices)
    next_b = np.roll(b, -1)
    next_top = np.roll(border, -1)
    faces = np.concatenate(
        (
            faces,
            np.column_stack((border, b, next_b)),
            np.column_stack((border, next_b, next_top)),
        )
    )
    corners = b[[0, nc - 1, nc + nr - 2, 2 * nc + nr - 3]]
    faces = np.concatenate(
        (
            faces,
            np.array(
                [
                    [corners[0], corners[2], corners[1]],
                    [corners[0], corners[3], corners[2]],
                ]
            ),
        )
    )
    return np.concatenate((vertices, bottom)), faces


def _cell_bounds(low, high):
    lower = np.floor(np.asarray(low) / CHUNK_METERS).astype(np.int64)
    upper = np.maximum(
        lower, np.ceil(np.asarray(high) / CHUNK_METERS).astype(np.int64) - 1
    )
    return lower, upper


def _spatial_preflight(model, item):
    if item["kind"] == 6:
        corners = _BOX_VERTS * item["size"]
    else:
        size = _finite(model.hfield_size[item["hfield"]])
        _require((size > 0).all(), "HFIELD_DATA")
        corners = _BOX_VERTS * [size[0], size[1], (size[2] + size[3]) / 2]
        corners[:, 2] += (size[2] - size[3]) / 2
    corners = corners @ item["rotation"].T + item["position"]
    _finite(corners)
    low, high = _cell_bounds(corners[:, :2].min(axis=0), corners[:, :2].max(axis=0))
    spans = high - low + 1
    # Conservative bounding-volume guard, before cell enumeration or hfield arrays.
    _require(int(spans[0]) * int(spans[1]) <= MAX_SLOTS, "SPATIAL_CELL_LIMIT")


def _clip_triangle(triangle, cell):
    polygon = list(triangle)
    for axis in (0, 1):
        for boundary, sign in (
            (cell[axis] * CHUNK_METERS, 1),
            ((cell[axis] + 1) * CHUNK_METERS, -1),
        ):
            clipped = []
            if not polygon:
                return np.empty((0, 3))
            previous = polygon[-1]
            previous_inside = sign * (previous[axis] - boundary) >= 0
            for current in polygon:
                inside = sign * (current[axis] - boundary) >= 0
                if inside != previous_inside:
                    fraction = (boundary - previous[axis]) / (
                        current[axis] - previous[axis]
                    )
                    point = previous + fraction * (current - previous)
                    point[axis] = boundary
                    clipped.append(point)
                if inside:
                    clipped.append(current)
                previous, previous_inside = current, inside
            polygon = clipped
    distinct = [
        point
        for i, point in enumerate(polygon)
        if not np.array_equal(point, polygon[i - 1])
    ]
    _require(len(distinct) <= 7, "CLIP_TOPOLOGY")
    return np.asarray(distinct)


class _SpatialChunks:
    """Guarded exact surface clipping; no internal cut faces are introduced."""

    def __init__(self, vertex_count, face_count, primitive_count, material_count):
        self.vertices = vertex_count
        self.faces = face_count
        self.primitives = primitive_count
        self.materials = material_count
        self.chunks = {}

    def _preflight(self, plans):
        _require(
            self.primitives + len(self.chunks.keys() | plans.keys()) <= MAX_SLOTS,
            "DISPLAY_SLOT_LIMIT",
        )
        vertices = self.vertices + sum(v for v, _ in plans.values())
        faces = self.faces + sum(f for _, f in plans.values())
        _require(vertices <= MAX_VERTICES and faces <= MAX_TRIANGLES, "MESH_BUDGET")
        for key, (vn, fn) in plans.items():
            old = self.chunks.get(key, {"vn": 0, "fn": 0})
            _require(
                (old["vn"] + vn) * 3 <= MAX_COMPONENTS
                and (old["fn"] + fn) * 3 <= MAX_COMPONENTS,
                "MESH_COMPONENT_LIMIT",
            )
        predicted = (
            4 * ((vertices * 28 + faces * 12 + 2) // 3)
            + MAX_SLOTS * 1024
            + self.materials * 512
        )
        _require(predicted <= MAX_GEOMETRY_BYTES, "GEOMETRY_BYTES")

    def _store(self, key, sources, vertices, faces, colors):
        chunk = self.chunks.setdefault(
            key, {"pieces": [], "sources": set(), "vn": 0, "fn": 0}
        )
        chunk["pieces"].append((vertices, faces, colors))
        chunk["sources"].update(sources)
        chunk["vn"] += len(vertices)
        chunk["fn"] += len(faces)
        self.vertices += len(vertices)
        self.faces += len(faces)

    def _append(self, key, item, vertices, faces):
        if not len(faces):
            return
        self._preflight({key: (len(vertices), len(faces))})
        self._store(
            key,
            (item["source"],),
            vertices,
            faces,
            np.tile(item["rgba"], (len(vertices), 1)),
        )

    def add_contained_boxes(self, boxes):
        groups = {}
        for item in boxes:
            low, high = item["cell_bounds"]
            if np.array_equal(low, high):
                groups.setdefault((*map(int, low), item["material"]), []).append(item)
        # All buffers/colors for the contained batch follow one exact aggregate
        # reservation. Original source/coordinate checks have already passed.
        self._preflight(
            {key: (8 * len(items), 12 * len(items)) for key, items in groups.items()}
        )
        for key, items in sorted(groups.items()):
            count = len(items)
            vertices = np.concatenate([item["world_vertices"] for item in items])
            faces = (
                _BOX_FACES[None, :, :] + (8 * np.arange(count))[:, None, None]
            ).reshape(-1, 3)
            colors = np.repeat(np.asarray([item["rgba"] for item in items]), 8, axis=0)
            self._store(
                key, (item["source"] for item in items), vertices, faces, colors
            )

    def add(self, item, vertices, faces):
        _finite(vertices)
        if "cell_bounds" in item:
            low, high = item["cell_bounds"]
        else:
            low, high = _cell_bounds(
                vertices[:, :2].min(axis=0), vertices[:, :2].max(axis=0)
            )
        if np.array_equal(low, high):
            self._append((*map(int, low), item["material"]), item, vertices, faces)
            return
        # Original arrays already passed source/aggregate bounds; this temporary
        # indexes only one source surface. Only crossing triangles are clipped.
        triangles = vertices[faces]
        lower, upper = _cell_bounds(
            triangles[:, :, :2].min(axis=1), triangles[:, :, :2].max(axis=1)
        )
        contained = (lower == upper).all(axis=1)
        plans = {}
        for cell in np.unique(lower[contained], axis=0):
            ids = np.flatnonzero(contained & (lower == cell).all(axis=1))
            used, remap = np.unique(faces[ids], return_inverse=True)
            plans[tuple(map(int, cell))] = {
                "used": used,
                "faces": remap.reshape(-1, 3),
                "cross": [],
            }
        pair_count = 0
        for face_id in np.flatnonzero(~contained):
            spans = upper[face_id] - lower[face_id] + 1
            pair_count += int(spans[0]) * int(spans[1])
            _require(
                pair_count * 7 <= MAX_VERTICES and pair_count * 5 <= MAX_TRIANGLES,
                "MESH_BUDGET",
            )
            for x in range(int(lower[face_id, 0]), int(upper[face_id, 0]) + 1):
                for y in range(int(lower[face_id, 1]), int(upper[face_id, 1]) + 1):
                    plan = plans.setdefault(
                        (x, y),
                        {
                            "used": np.empty(0, np.int64),
                            "faces": np.empty((0, 3), np.int64),
                            "cross": [],
                        },
                    )
                    plan["cross"].append(face_id)
        # Triangle/rectangle intersection has at most seven vertices/five faces.
        # Reserve this conservative output BEFORE polygon/vertex expansion.
        self._preflight(
            {
                (*cell, item["material"]): (
                    len(plan["used"]) + 7 * len(plan["cross"]),
                    len(plan["faces"]) + 5 * len(plan["cross"]),
                )
                for cell, plan in plans.items()
            }
        )
        for cell, plan in sorted(plans.items()):
            vertex_parts = [vertices[plan["used"]]]
            face_parts = [plan["faces"]]
            offset = len(plan["used"])
            for face_id in plan["cross"]:
                polygon = _clip_triangle(triangles[face_id], cell)
                if len(polygon) < 3:
                    continue
                vertex_parts.append(polygon)
                face_parts.append(
                    np.array(
                        [
                            [offset, offset + i, offset + i + 1]
                            for i in range(1, len(polygon) - 1)
                        ]
                    )
                )
                offset += len(polygon)
            # Exact-equality deduplication preserves all geometric detail/colors.
            unique, inverse = np.unique(
                np.concatenate(vertex_parts), axis=0, return_inverse=True
            )
            indices = inverse[np.concatenate(face_parts)]
            tri = unique[indices]
            valid = np.any(
                np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) != 0, axis=1
            )
            indices = indices[valid]
            if not len(indices):
                continue
            used, remap = np.unique(indices, return_inverse=True)
            self._append(
                (*cell, item["material"]), item, unique[used], remap.reshape(-1, 3)
            )


def build_projection(model, *, robot_root_body_id):
    """Bake fixed terrain exactly; keep actual dynamic objects in dense slots.

    Reject before expansion if aggregate/component/source/slot bounds fail.
    Clip crossing source surface triangles to deterministic 16 m XY cells.
    Shared material/color semantics and exact surfaces are preserved.
    """
    import mujoco

    _require(0 < model.ngeom <= MAX_SOURCE_GEOMS, "SOURCE_GEOMETRY_LIMIT")
    wp, wr, dynamic, focus = _body_transforms(model, robot_root_body_id)
    descriptors = []
    terrain_items = []
    mesh_sources = {}
    materials = {}
    vertices_total = 0
    faces_total = 0
    # Gather only visible numeric fields after the source cap; each bounded
    # array is validated once, including source RGBA before material resolution.
    visible = np.flatnonzero(model.geom_group != 3)
    for field_name in ("geom_size", "geom_rgba", "geom_quat", "geom_pos"):
        _finite(getattr(model, field_name)[visible])
    material_ids = model.geom_matid[visible]
    _require(
        ((material_ids >= -1) & (material_ids < model.nmat)).all(), "MATERIAL_REFERENCE"
    )
    colors = model.geom_rgba[visible].copy()
    use_material = (material_ids >= 0) & (colors == [0.5, 0.5, 0.5, 1.0]).all(axis=1)
    colors[use_material] = model.mat_rgba[material_ids[use_material]]
    _finite(colors)
    for visible_index, source_id in enumerate(visible):
        source = int(source_id)
        kind = int(model.geom_type[source])
        body = int(model.geom_bodyid[source])
        _require(0 <= body < model.nbody, "BODY_ANCESTRY")
        _require(kind in {0, 1, 2, 3, 4, 5, 6, 7}, "UNSUPPORTED_GEOMETRY")
        _require(not (kind == 1 and dynamic[body]), "DYNAMIC_HEIGHTFIELD_UNSUPPORTED")
        size = model.geom_size[source]
        color = colors[visible_index]
        material = int(model.geom_matid[source])
        if material >= 0 and material not in materials:
            properties = {
                name: float(getattr(model, "mat_" + name)[material])
                for name in (
                    "emission",
                    "specular",
                    "shininess",
                    "reflectance",
                    "metallic",
                    "roughness",
                )
                if hasattr(model, "mat_" + name)
            }
            _finite(np.array(list(properties.values())))
            properties["textureOmitted"] = bool((model.mat_texid[material] >= 0).any())
            materials[material] = dict(id=material, **properties)
        local_r = np.empty(9)
        mujoco.mju_quat2Mat(local_r, model.geom_quat[source])
        rotation = wr[body] @ local_r.reshape(3, 3)
        position = wp[body] + wr[body] @ model.geom_pos[source]
        item = {
            "source": source,
            "kind": kind,
            "size": size.tolist(),
            "rgba": color.tolist(),
            "material": material,
            "position": position,
            "rotation": rotation,
            "dynamic": dynamic[body],
            "focus": focus[body],
        }
        if kind in (1, 6) and not dynamic[body] and not focus[body]:
            if kind == 1:
                idx = int(model.geom_dataid[source])
                _require(0 <= idx < model.nhfield, "HFIELD_REFERENCE")
                nr, nc, start, vn, fn = _heightfield(model, idx)
                item["hfield"] = idx
                _require(vn <= MAX_VERTICES and fn <= MAX_TRIANGLES, "MESH_BUDGET")
                _require(
                    start >= 0 and start + nr * nc <= len(model.hfield_data),
                    "HFIELD_DATA",
                )
            else:
                vn, fn = 8, 12
            _require(
                vn * 3 <= MAX_COMPONENTS and fn * 3 <= MAX_COMPONENTS,
                "MESH_COMPONENT_LIMIT",
            )
            vertices_total += vn
            faces_total += fn
            if kind == 1:
                _spatial_preflight(model, item)
            terrain_items.append(item)
        else:
            _require(kind != 1, "ROBOT_HEIGHTFIELD_UNSUPPORTED")
            if kind == 7:
                idx = int(model.geom_dataid[source])
                _require(0 <= idx < model.nmesh, "MESH_REFERENCE")
                item["sourceMesh"] = idx
                if idx not in mesh_sources:
                    va, vn, fa, fn = map(
                        int,
                        (
                            model.mesh_vertadr[idx],
                            model.mesh_vertnum[idx],
                            model.mesh_faceadr[idx],
                            model.mesh_facenum[idx],
                        ),
                    )
                    _require(
                        3 <= vn
                        and 0 < fn
                        and vn * 3 <= MAX_COMPONENTS
                        and fn * 3 <= MAX_COMPONENTS,
                        "MESH_COMPONENT_LIMIT",
                    )
                    _require(
                        0 <= va
                        and va + vn <= len(model.mesh_vert)
                        and 0 <= fa
                        and fa + fn <= len(model.mesh_face),
                        "MESH_SLICE",
                    )
                    vertices_total += vn
                    faces_total += fn
                    mesh_sources[idx] = (va, vn, fa, fn)
            descriptors.append(item)
        _require(
            vertices_total <= MAX_VERTICES and faces_total <= MAX_TRIANGLES,
            "MESH_BUDGET",
        )
    _finite(np.asarray([item["position"] for item in descriptors + terrain_items]))
    # At most 32768 boxes/eight corners each. Source geometry budgets already
    # passed; no unbounded cell enumeration or clipped output is allocated here.
    boxes = [item for item in terrain_items if item["kind"] == 6]
    if boxes:
        sizes = np.asarray([item["size"] for item in boxes])
        rotations = np.asarray([item["rotation"] for item in boxes])
        origins = np.asarray([item["position"] for item in boxes])
        corners = (_BOX_VERTS[None, :, :] * sizes[:, None, :]) @ rotations.transpose(
            0, 2, 1
        ) + origins[:, None, :]
        _finite(corners)
        lower, upper = _cell_bounds(
            corners[:, :, :2].min(axis=1), corners[:, :, :2].max(axis=1)
        )
        spans = upper - lower + 1
        _require((spans[:, 0] * spans[:, 1] <= MAX_SLOTS).all(), "SPATIAL_CELL_LIMIT")
        for index, item in enumerate(boxes):
            item["world_vertices"] = corners[index]
            item["cell_bounds"] = (lower[index], upper[index])
    _require(len(descriptors) <= MAX_SLOTS, "DISPLAY_SLOT_LIMIT")
    spatial = _SpatialChunks(
        sum(vn for _, vn, _, _ in mesh_sources.values()),
        sum(fn for _, _, _, fn in mesh_sources.values()),
        len(descriptors),
        len(materials),
    )
    spatial._preflight({})
    spatial.add_contained_boxes(boxes)
    for item in terrain_items:
        if item["kind"] == 6:
            low, high = item["cell_bounds"]
            if np.array_equal(low, high):
                continue
            spatial.add(item, item["world_vertices"], _BOX_FACES)
        else:
            vertices, faces = _hfield_mesh(model, item["hfield"])
            spatial.add(item, vertices @ item["rotation"].T + item["position"], faces)
    chunks = spatial.chunks
    geometry = []
    meshes = []
    positions = []
    matrices = []
    mapping = []
    dynamic_mapping = []
    focus_slots = []
    source_mesh_ids = {}
    for source, (va, vn, fa, fn) in sorted(mesh_sources.items()):
        source_mesh_ids[source] = len(meshes)
        meshes.append(
            _mesh(
                model.mesh_vert[va : va + vn],
                model.mesh_face[fa : fa + fn],
                len(meshes),
            )
        )
    for item in descriptors:
        slot = len(geometry)
        geom = {
            "id": slot,
            "type": item["kind"],
            "size": item["size"],
            "rgba": item["rgba"],
            "materialId": item["material"],
        }
        if "sourceMesh" in item:
            geom["meshId"] = source_mesh_ids[item["sourceMesh"]]
        geometry.append(geom)
        positions.append(
            tuple([0.0, 0.0, 0.0] if item["dynamic"] else item["position"].tolist())
        )
        matrices.append(
            tuple(np.eye(3).ravel() if item["dynamic"] else item["rotation"].ravel())
        )
        mapping.append(
            {
                "slot": slot,
                "sourceGeomIds": [item["source"]],
                "dynamic": item["dynamic"],
            }
        )
        if item["dynamic"]:
            dynamic_mapping.append((slot, item["source"]))
        if item["focus"]:
            focus_slots.append(slot)
    for key, chunk in sorted(chunks.items()):
        verts = []
        faces = []
        colors = []
        offset = 0
        for v, f, color in chunk["pieces"]:
            verts.append(v)
            faces.append(f + offset)
            colors.append(color)
            offset += len(v)
        mesh_id = len(meshes)
        meshes.append(
            _mesh(
                np.concatenate(verts),
                np.concatenate(faces),
                mesh_id,
                np.concatenate(colors),
            )
        )
        slot = len(geometry)
        geometry.append(
            {
                "id": slot,
                "type": 7,
                "size": [1.0, 1.0, 1.0],
                "rgba": [1.0, 1.0, 1.0, 1.0],
                "materialId": key[2],
                "meshId": mesh_id,
            }
        )
        positions.append((0.0, 0.0, 0.0))
        matrices.append(tuple(np.eye(3).ravel()))
        mapping.append(
            {
                "slot": slot,
                "sourceGeomIds": sorted(chunk["sources"]),
                "dynamic": False,
            }
        )
    _require(bool(focus_slots), "ROBOT_GEOMETRY_EMPTY")
    payload = {
        "geometry": geometry,
        "meshes": meshes,
        "materials": list(materials.values()),
        "focusSlotIds": focus_slots,
        "sourceGeomCount": int(model.ngeom),
        "slotIdentity": "dense-display-slot",
        "meshEncoding": "float32-le",
        "colorEncoding": "rgba-float32-le",
    }
    identity = {
        "payload": payload,
        "mapping": mapping,
        "positions": positions,
        "matrices": matrices,
        "robotRootBodyId": robot_root_body_id,
    }
    digest = "sha256:" + hashlib.sha256(_json(identity).encode()).hexdigest()
    payload["mappingDigest"] = digest
    encoded = _json(payload)
    _require(len(encoded.encode()) <= MAX_GEOMETRY_BYTES, "GEOMETRY_BYTES")
    return GeometryProjection(
        encoded,
        int(model.ngeom),
        digest,
        tuple(dynamic_mapping),
        tuple(tuple(x["sourceGeomIds"]) for x in mapping),
        tuple(positions),
        tuple(matrices),
    )
