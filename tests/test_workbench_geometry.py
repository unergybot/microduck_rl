"""Private projection fidelity: compiled source geometry, never a simulator loop."""

import base64
import json
import os
import random
import signal
import time

import mujoco
import numpy as np
import pytest
import torch

torch.set_num_threads(1)
from mjlab_microduck.workbench.geometry import ProjectionError, build_projection


def model(body="", asset=""):
    return mujoco.MjModel.from_xml_string(f"""<mujoco><asset>{asset}</asset><worldbody>
    {body}<body name="robot" pos="0 0 2"><freejoint/><geom size=".1"/>
    <body name="child" pos="0 0 .2"><geom size=".05"/><geom size=".02" group="3"/></body>
    </body></worldbody></mujoco>""")


def build(m):
    return build_projection(m, robot_root_body_id=m.body("robot").id)


def mesh_arrays(entry):
    return (
        np.frombuffer(
            base64.b64decode(entry["verticesBase64"], validate=True), "<f4"
        ).reshape(-1, 3),
        np.frombuffer(
            base64.b64decode(entry["indicesBase64"], validate=True),
            "<u2" if entry["indexType"] == "uint16" else "<u4",
        ).reshape(-1, 3),
    )


def frame_arrays(frame):
    shape = (len(frame["environmentIndices"]), frame["slotCount"])
    return (
        np.frombuffer(
            base64.b64decode(frame["positionsBase64"], validate=True), "<f4"
        ).reshape(*shape, 3),
        np.frombuffer(
            base64.b64decode(frame["matricesBase64"], validate=True), "<f4"
        ).reshape(*shape, 3, 3),
    )


def test_fixed_rotated_box_world_corners_colors_focus_and_immutable_export():
    m = model(
        '<body pos="2 3 4" euler="0 0 90"><body pos="1 0 0"><geom type="box" size="1 2 3" rgba=".2 .4 .6 1"/></body></body>'
    )
    p = build(m)
    g = p.export()
    assert p.source_geom_count == 4
    assert [v["id"] for v in g["geometry"]] == list(range(3))
    assert len(g["focusSlotIds"]) == 2
    terrain = next(v for v in g["geometry"] if v["type"] == 7)
    v, f = mesh_arrays(g["meshes"][terrain["meshId"]])
    assert len(v) == 8 and len(f) == 12
    expected = np.array([[x, y, z] for x in [0, 4] for y in [3, 5] for z in [1, 7]])
    np.testing.assert_allclose(
        sorted(map(tuple, v)), sorted(map(tuple, expected)), atol=1e-6
    )
    rgba = np.frombuffer(
        base64.b64decode(g["meshes"][terrain["meshId"]]["colorsBase64"]), "<f4"
    ).reshape(-1, 4)
    np.testing.assert_allclose(rgba, np.tile([0.2, 0.4, 0.6, 1], (8, 1)))
    original = p.mapping_digest
    g["geometry"].clear()
    assert len(p.export()["geometry"]) == 3
    assert build(m).mapping_digest == original
    m.geom_pos[0, 0] += 0.1
    assert build(m).mapping_digest != original


def ray_triangles(origin, direction, vertices, faces):
    a, b, c = (vertices[faces[:, i]].astype(float) for i in range(3))
    edge1 = b - a
    edge2 = c - a
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    det = np.einsum("ij,ij->i", edge1, h)
    ok = np.abs(det) > 1e-10
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=ok)
    s = origin - a
    u = inv * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inv * (q @ direction)
    t = inv * np.einsum("ij,ij->i", edge2, q)
    valid = ok & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6) & (t >= 0)
    return t[valid].min() if valid.any() else -1


def test_anisotropic_heightfield_rays_rotation_border_sides_and_base():
    m = model(
        '<body pos="3 -2 1" euler="20 15 30"><geom type="hfield" hfield="h"/></body>',
        '<hfield name="h" nrow="3" ncol="4" size="2 1 .7 .3"/>',
    )
    m.hfield_data[:] = [0, 0.2, 0.8, 0.1, 0.6, 1, 0.1, 0.4, 0.1, 0.3, 0.7, 0.2]
    p = build(m)
    g = p.export()
    v, f = mesh_arrays(g["meshes"][0])
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    rotation = d.geom_xmat[0].reshape(3, 3)
    translation = d.geom_xpos[0]
    for x, y in [
        (-1.7, -0.7),
        (-0.2, -0.4),
        (0.7, 0.3),
        (1.9, 0.8),
        (-2.0, 0),
        (2.0, 0.7),
        (0, -1 + 1e-6),
        (0, 1),
    ]:
        origin = translation + rotation @ np.array([x, y, 3.0])
        direction = rotation @ np.array([0.0, 0.0, -1.0])
        expected = mujoco.mj_ray(
            m, d, origin, direction, None, 1, -1, np.zeros(1, np.int32)
        )
        assert expected >= 0
        assert ray_triangles(origin, direction, v, f) == pytest.approx(
            expected, abs=3e-6
        )
    # Side wall and underside are part of MuJoCo's solid hfield, not a top-only patch.
    for origin, direction in [([-3, 0, -0.1], [1, 0, 0]), ([0, 0, -2], [0, 0, 1])]:
        o = translation + rotation @ origin
        r = rotation @ direction
        expected = mujoco.mj_ray(m, d, o, r, None, 1, -1, np.zeros(1, np.int32))
        assert ray_triangles(o, r, v, f) == pytest.approx(expected, abs=3e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_selected_actual_dynamic_pose_capture_nine_envs_is_readonly(backend):
    m = model(
        '<geom type="box" size="1 1 .1"/><body name="object"><freejoint/><geom size=".2"/></body>'
    )
    p = build(m)
    pos = np.arange(12 * m.ngeom * 3, dtype=np.float32).reshape(12, m.ngeom, 3)
    mat = np.tile(np.eye(3, dtype=np.float32), (12, m.ngeom, 1, 1))
    old = pos.copy()
    pos[:, 0] = np.nan  # static source coordinates must never be gathered
    old = pos.copy()
    envs = [11, 9, 7, 5, 3, 1, 0, 2, 4]
    args = (
        (torch.from_numpy(pos), torch.from_numpy(mat))
        if backend == "torch"
        else (pos, mat)
    )
    frame = p.capture(*args, environment_indices=envs)
    out, rot = frame_arrays(frame)
    assert len(json.dumps(frame, separators=(",", ":")).encode()) <= 262144
    assert frame["mappingDigest"] == p.mapping_digest
    assert frame["environmentIndices"] == envs
    for slot, source in p.dynamic_mapping:
        np.testing.assert_array_equal(out[:, slot], old[envs, source])
        np.testing.assert_array_equal(rot[:, slot], mat[envs, source])
    assert len(p.export()["focusSlotIds"]) == 2  # excludes independent dynamic object
    np.testing.assert_array_equal(pos, old)
    assert not torch.cuda.is_initialized()
    with pytest.raises(ProjectionError, match="ENVIRONMENT_INDICES"):
        p.capture(*args, environment_indices=list(range(10)))
    with pytest.raises(ProjectionError, match="ENVIRONMENT_INDICES"):
        p.capture(*args, environment_indices=[12])
    pos[0, 1, 0] = np.nan
    with pytest.raises(ProjectionError, match="NONFINITE_POSE"):
        p.capture(*args, environment_indices=[0])


def test_shared_compiled_mesh_bitexact_dynamic_child_and_focus():
    mesh = '<mesh name="m" vertex="0 0 0 1 0 0 0 1 0 0 0 1" face="0 1 2 0 1 3 0 2 3 1 2 3"/>'
    m = model(
        '<body><freejoint/><geom type="mesh" mesh="m"/><body><geom type="mesh" mesh="m"/></body></body>',
        mesh,
    )
    p = build(m)
    g = p.export()
    assert len(g["meshes"]) == 1
    v, f = mesh_arrays(g["meshes"][0])
    assert v.tobytes() == m.mesh_vert.astype("<f4").tobytes()
    np.testing.assert_array_equal(f, m.mesh_face)
    assert len([a for a in g["geometry"] if a.get("meshId") == 0]) == 2


@pytest.mark.parametrize(
    "failure",
    [
        "root",
        "unsupported",
        "nonfinite",
        "source_cap",
        "hfield_predicted",
        "slots",
        "bad_mesh",
    ],
)
def test_explicit_refusal_before_expansion(failure):
    m = model(
        '<geom type="box" size="1 1 .1"/>',
        '<hfield name="h" nrow="2" ncol="2" size="1 1 1 .1"/>',
    )
    root = m.body("robot").id
    if failure == "root":
        root = 0
        code = "ROBOT_ROOT"
    elif failure == "unsupported":
        m.geom_type[0] = 100
        code = "UNSUPPORTED_GEOMETRY"
    elif failure == "nonfinite":
        m.geom_size[0, 0] = np.nan
        code = "NONFINITE_GEOMETRY"
    elif failure == "source_cap":

        class TooLarge:
            ngeom = 32769

            def __getattr__(self, name):
                raise AssertionError("allocated/read after source cap")

        m = TooLarge()
        code = "SOURCE_GEOMETRY_LIMIT"
    elif failure == "hfield_predicted":
        m.geom_type[0] = 1
        m.geom_dataid[0] = 0
        m.hfield_nrow[0] = 1000000
        m.hfield_ncol[0] = 1000000
        code = "MESH_BUDGET"
    elif failure == "slots":
        m = model("".join('<geom type="sphere" size=".1"/>' for _ in range(255)))
        root = m.body("robot").id
        code = "DISPLAY_SLOT_LIMIT"
    else:
        m.geom_type[0] = 7
        m.geom_dataid[0] = 50
        code = "MESH_REFERENCE"
    with pytest.raises(ProjectionError, match=code):
        build_projection(m, robot_root_body_id=root)


def test_spatial_chunks_retain_distant_boxes_and_material_colors():
    m = model(
        "".join(
            f'<geom type="box" pos="{x} 4 0" size="1 1 .1" rgba="{r} 0 0 1"/>'
            for x, r in [(4, 0.2), (5, 0.5), (44, 0.8)]
        )
    )
    g = build(m).export()
    assert len(g["meshes"]) == 2
    all_vertices = np.concatenate([mesh_arrays(v)[0] for v in g["meshes"]])
    assert all_vertices[:, 0].min() == 3 and all_vertices[:, 0].max() == 45
    assert sum(len(mesh_arrays(v)[1]) for v in g["meshes"]) == 36


@pytest.mark.parametrize(
    "recipe,num_envs",
    [
        ("Mjlab-Velocity-Flat-MicroDuck", 1),
        ("Mjlab-Velocity-Rough-MicroDuck", 4096),
        ("Mjlab-RollerSlope-Flat-MicroDuck", 1),
    ],
)
def test_actual_factory_scene_projection_cpu_metadata_only(recipe, num_envs):
    _check_scene_projection(recipe, num_envs)


def _check_scene_projection(recipe, num_envs, intended_grid=False):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    from mjlab.scene import Scene

    from mjlab_microduck.workbench import resolve_experiment
    from mjlab_microduck.workbench.engine import build_configs

    def timeout_handler(*_):
        raise TimeoutError("90-second Scene diagnostic ceiling")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(90)
    started = time.monotonic()
    try:
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        spec = resolve_experiment(
            {
                "schemaVersion": "ROM_TRAINING_EXPERIMENT_V1",
                "recipeId": recipe,
                "robotProfileId": "rom-xl330-v1",
                "config": {"numEnvs": num_envs, "iterations": 1, "seed": 11},
            }
        )
        cfg, _ = build_configs(spec)
        if intended_grid:
            # Explicit stress input only: not unchanged-factory acceptance.
            from copy import deepcopy

            cfg = deepcopy(cfg)
            cfg.scene.terrain.terrain_generator.num_rows = 10
            cfg.scene.terrain.terrain_generator.num_cols = 20
        scene = Scene(cfg.scene, device="cpu")
        m = scene.compile()
        root = m.body("robot/trunk_base").id
        p = build_projection(m, robot_root_body_id=root)
        g = p.export()
        assert (
            len(g["geometry"]) <= 256
            and len(json.dumps(g, separators=(",", ":")).encode()) <= 64 * 1024 * 1024
        )
        for geom in g["geometry"]:
            if (
                geom.get("meshId") is not None
                and "colorsBase64" in g["meshes"][geom["meshId"]]
            ):
                vertices, _ = mesh_arrays(g["meshes"][geom["meshId"]])
                assert np.ptp(vertices[:, :2], axis=0).max() <= 16.00002
        assert p.source_geom_count == m.ngeom and len(g["focusSlotIds"]) >= 60
        assert (
            sorted({source for members in p.source_membership for source in members})
            == np.flatnonzero(m.geom_group != 3).tolist()
        )
        if num_envs == 4096:
            assert tuple(scene.env_origins.shape) == (4096, 3)
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        # Only nine CPU pose samples, never a 4096-instance Simulation allocation.
        frame = p.capture(
            np.broadcast_to(d.geom_xpos, (9, m.ngeom, 3)),
            np.broadcast_to(d.geom_xmat.reshape(m.ngeom, 3, 3), (9, m.ngeom, 3, 3)),
            environment_indices=list(range(9)),
        )
        assert len(json.dumps(frame, separators=(",", ":")).encode()) < 262144
        print(
            "GEOMETRY_PROJECTION",
            json.dumps(
                {
                    "recipe": recipe,
                    "intendedGridStress": intended_grid,
                    "numEnvs": num_envs,
                    "sourceGeoms": m.ngeom,
                    "displaySlots": len(g["geometry"]),
                    "meshes": len(g["meshes"]),
                    "vertices": sum(len(mesh_arrays(mesh)[0]) for mesh in g["meshes"]),
                    "triangles": sum(len(mesh_arrays(mesh)[1]) for mesh in g["meshes"]),
                    "geometryBytes": len(p.geometry_json),
                    "frameBytes": len(
                        json.dumps(frame, separators=(",", ":")).encode()
                    ),
                    "seconds": time.monotonic() - started,
                    "steps": 0,
                }
            ),
        )
        assert not torch.cuda.is_initialized()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def test_exact_unrotated_heightfield_borders_and_corners():
    m = model(
        '<geom type="hfield" hfield="h"/>',
        '<hfield name="h" nrow="2" ncol="3" size="2 1 1 .2"/>',
    )
    m.hfield_data[:] = [0, 0.3, 1, 0.2, 0.9, 0.5]
    v, f = _all_terrain_meshes(build(m))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    for x, y in [(-2, 0), (2, 0.5), (0.5, -1), (0.5, 1), (-2, -1), (2, 1)]:
        o = np.array([x, y, 3.0])
        r = np.array([0.0, 0.0, -1.0])
        expected = mujoco.mj_ray(m, d, o, r, None, 1, -1, np.zeros(1, np.int32))
        assert ray_triangles(o, r, v, f) == pytest.approx(expected, abs=2e-6)


def test_static_robot_pose_capture_needs_no_dynamic_source():
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="robot" pos="1 2 3"><geom size=".1"/></body></worldbody></mujoco>'
    )
    p = build(m)
    pos, mat = frame_arrays(
        p.capture(np.zeros((1, 1, 3)), np.zeros((1, 1, 3, 3)), environment_indices=[0])
    )
    np.testing.assert_array_equal(pos, [[[1, 2, 3]]])
    np.testing.assert_array_equal(mat, [[np.eye(3)]])
    assert p.source_membership == ((0,),)


def test_torch_transfers_only_selected_dynamic_values(monkeypatch):
    m = model('<geom type="box" size="1 1 .1"/>')
    p = build(m)
    positions = torch.zeros(20, m.ngeom, 3)
    matrices = torch.eye(3).expand(20, m.ngeom, 3, 3)
    original_to = torch.Tensor.to
    transfers = []

    def selected_to(tensor, *args, **kwargs):
        transfers.append(tuple(tensor.shape))
        assert tensor.shape[:2] == (2, len(p.dynamic_mapping))
        return original_to(tensor, *args, **kwargs)

    def no_clone(*_args, **_kwargs):
        raise AssertionError("source clone is forbidden")

    monkeypatch.setattr(torch.Tensor, "to", selected_to)
    monkeypatch.setattr(torch.Tensor, "clone", no_clone)
    p.capture(positions, matrices, environment_indices=[19, 5])
    assert transfers == [(2, 2, 3), (2, 2, 3, 3)]


@pytest.mark.parametrize("failure", ["nan", "negative", "range"])
def test_corrupt_actual_shared_mesh_refuses(failure):
    asset = '<mesh name="m" vertex="0 0 0 1 0 0 0 1 0 0 0 1" face="0 1 2 0 1 3 0 2 3 1 2 3"/>'
    m = model('<geom type="mesh" mesh="m"/>', asset)
    if failure == "nan":
        m.mesh_vert[0, 0] = np.nan
    elif failure == "negative":
        m.mesh_face[0, 0] = -1
    else:
        m.mesh_face[0, 0] = m.mesh_vertnum[0]
    with pytest.raises(ProjectionError, match="NONFINITE_GEOMETRY|MESH_INDEX"):
        build(m)


def test_intended_10x20_terrain_stress_metadata_only():
    _check_scene_projection("Mjlab-Velocity-Rough-MicroDuck", 4096, intended_grid=True)


def test_uint32_mesh_indices_endianness_and_topology():
    m = model(
        '<geom type="hfield" hfield="h" pos="4 4 0"/>',
        '<hfield name="h" nrow="2" ncol="17000" size="2 1 1 .2"/>',
    )
    entry = build(m).export()["meshes"][0]
    v, f = mesh_arrays(entry)
    assert entry["indexType"] == "uint32" and len(v) > 65536
    assert (
        base64.b64decode(entry["indicesBase64"])[:12]
        == np.array([0, 1, 17001], dtype="<u4").tobytes()
    )
    assert (
        base64.b64decode(entry["verticesBase64"])[:12]
        == np.array([2, 3, 0], dtype="<f4").tobytes()
    )
    assert f.max() == len(v) - 1
    colors = base64.b64decode(entry["colorsBase64"])
    assert len(colors) == len(v) * 16


def test_maximum_256_slots_nine_environment_frame_fits():
    m = model("".join('<geom size=".1"/>' for _ in range(254)))
    p = build(m)
    frame = p.capture(
        np.zeros((9, m.ngeom, 3)),
        np.zeros((9, m.ngeom, 3, 3)),
        environment_indices=list(range(9)),
    )
    assert frame["slotCount"] == 256
    assert len(json.dumps(frame, separators=(",", ":")).encode()) < 262144


def test_fixed_children_of_dynamic_and_mocap_ancestors_stay_dynamic():
    m = model(
        '<body><freejoint/><geom type="box" size=".1 .1 .1"/><body><geom type="box" size=".1 .1 .1"/></body></body><body mocap="true"><geom type="box" size=".1 .1 .1"/></body>'
    )
    p = build(m)
    assert len(p.dynamic_mapping) == 5 and not p.export()["meshes"]


def _all_terrain_meshes(projection):
    exported = projection.export()
    terrain = [g for g in exported["geometry"] if g["type"] == 7]
    vertices, faces = [], []
    offset = 0
    for geom in terrain:
        v, f = mesh_arrays(exported["meshes"][geom["meshId"]])
        assert np.ptp(v[:, :2], axis=0).max() <= 16.00002
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces)


def test_long_box_clips_to_actual_16m_extents_preserving_surface_and_color():
    m = model('<geom type="box" pos="3 4 2" size="40 3 1" rgba=".2 .4 .7 1"/>')
    p = build(m)
    v, f = _all_terrain_meshes(p)
    assert sum(0 in members for members in p.source_membership) > 1
    tri = v[f].astype(float)
    area = (
        np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
        ).sum()
        / 2
    )
    assert area == pytest.approx(1304, abs=2e-4)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    for origin, direction in [
        ([-30, 4, 9], [0, 0, -1]),
        ([16, 4, 9], [0, 0, -1]),
        ([40, 4, 9], [0, 0, -1]),
        ([-50, 4, 2], [1, 0, 0]),
        ([3, -2, 2], [0, 1, 0]),
    ]:
        origin = np.array(origin, float)
        direction = np.array(direction, float)
        expected = mujoco.mj_ray(
            m, d, origin, direction, None, 1, -1, np.zeros(1, np.int32)
        )
        assert ray_triangles(origin, direction, v, f) == pytest.approx(
            expected, abs=2e-5
        )
    for mesh in p.export()["meshes"]:
        rgba = np.frombuffer(base64.b64decode(mesh["colorsBase64"]), "<f4").reshape(
            -1, 4
        )
        np.testing.assert_allclose(rgba, np.tile([0.2, 0.4, 0.7, 1], (len(rgba), 1)))


def test_rotated_large_heightfield_clips_without_surface_or_ray_loss():
    m = model(
        '<body pos="5 7 2" euler="9 13 31"><geom type="hfield" hfield="h"/></body>',
        '<hfield name="h" nrow="4" ncol="5" size="35 23 2 .4"/>',
    )
    m.hfield_data[:] = np.array(
        [
            0,
            0.2,
            0.8,
            0.1,
            0.4,
            0.6,
            1,
            0.1,
            0.4,
            0.8,
            0.1,
            0.3,
            0.7,
            0.2,
            0.6,
            0.9,
            0.1,
            0.3,
            0.7,
            0.2,
        ]
    )
    p = build(m)
    v, f = _all_terrain_meshes(p)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    r = d.geom_xmat[0].reshape(3, 3)
    t = d.geom_xpos[0]
    for x in [-34, -24, -8, 4, 18, 34]:
        for y in [-22, -12, 3, 16, 22]:
            origin = t + r @ np.array([x, y, 10.0])
            direction = r @ np.array([0.0, 0.0, -1.0])
            expected = mujoco.mj_ray(
                m, d, origin, direction, None, 1, -1, np.zeros(1, np.int32)
            )
            assert ray_triangles(origin, direction, v, f) == pytest.approx(
                expected, abs=1e-5
            )
    # Independent original cell triangles, edge walls and bottom compute area.
    x, y = np.meshgrid(np.linspace(-35, 35, 5), np.linspace(-23, 23, 4))
    z = m.hfield_data.reshape(4, 5) * 2
    surface = np.stack([x, y, z], axis=-1)
    expected = 0.0
    for row in range(3):
        for col in range(4):
            a, b, c, e = (
                surface[row, col],
                surface[row, col + 1],
                surface[row + 1, col + 1],
                surface[row + 1, col],
            )
            expected += (
                np.linalg.norm(np.cross(b - a, c - a))
                + np.linalg.norm(np.cross(c - a, e - a))
            ) / 2
    for edge in [surface[0], surface[-1], surface[:, 0], surface[:, -1]]:
        expected += np.sum(
            np.linalg.norm(np.diff(edge[:, :2], axis=0), axis=1)
            * ((edge[:-1, 2] + edge[1:, 2]) / 2 + 0.4)
        )
    expected += 70 * 46
    tri = v[f].astype(float)
    actual = (
        np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
        ).sum()
        / 2
    )
    assert actual == pytest.approx(expected, rel=1e-7)
    assert {source for members in p.source_membership for source in members} == set(
        np.flatnonzero(m.geom_group != 3)
    )


def test_huge_spatial_span_refuses_before_surface_expansion(monkeypatch):
    from mjlab_microduck.workbench import geometry

    m = model(
        '<geom type="hfield" hfield="h"/>',
        '<hfield name="h" nrow="2" ncol="2" size="100000 100000 1 .2"/>',
    )

    def no_expansion(*args):
        raise AssertionError("expanded before spatial preflight")

    monkeypatch.setattr(geometry, "_hfield_mesh", no_expansion)
    with pytest.raises(ProjectionError, match="SPATIAL_CELL_LIMIT"):
        build(m)


@pytest.mark.parametrize("budget", ["vertices", "slots"])
def test_split_result_preflight_precedes_clipped_polygon_allocation(
    monkeypatch, budget
):
    from mjlab_microduck.workbench import geometry

    if budget == "vertices":
        m = model('<geom type="box" pos="3 4 2" size="40 3 1"/>')
        monkeypatch.setattr(geometry, "MAX_VERTICES", 32)
        code = "MESH_BUDGET"
    else:
        m = model(
            '<geom type="box" size="1000 1 1"/>'
            + "".join('<geom size=".1"/>' for _ in range(3))
        )
        code = "DISPLAY_SLOT_LIMIT"

    def no_clipping(*args):
        raise AssertionError("clipping allocated before result preflight")

    monkeypatch.setattr(geometry, "_clip_triangle", no_clipping)
    with pytest.raises(ProjectionError, match=code):
        build(m)


def test_grid_coincident_faces_have_no_duplicate_surface():
    m = model('<geom type="box" size="32 16 1"/>')
    v, f = _all_terrain_meshes(build(m))
    tri = v[f].astype(float)
    area = (
        np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
        ).sum()
        / 2
    )
    assert area == pytest.approx(4480, abs=1e-5)


@pytest.mark.parametrize("field", ["geom_size", "geom_rgba", "geom_quat", "geom_pos"])
def test_bulk_validation_retains_visible_refusal_and_hidden_omission(field):
    m = model('<geom type="box" pos="4 4 1" size="1 1 .1"/>')
    getattr(m, field)[-1, 0] = np.nan
    p = build(m)
    assert p.source_geom_count == m.ngeom
    getattr(m, field)[0, 0] = np.nan
    with pytest.raises(ProjectionError, match="NONFINITE_GEOMETRY"):
        build(m)


def test_contained_box_batch_preserves_each_world_corner_color_and_membership():
    m = model(
        '<geom type="box" pos="4 4 1" euler="0 0 30" size="1 2 .1" rgba=".2 .4 .6 1"/><geom type="box" pos="10 10 2" size="1 1 .2" rgba=".9 .1 .3 .5"/>'
    )
    p = build(m)
    g = p.export()
    assert len(g["meshes"]) == 1
    v, f = mesh_arrays(g["meshes"][0])
    assert len(v) == 16 and len(f) == 24
    rgba = np.frombuffer(
        base64.b64decode(g["meshes"][0]["colorsBase64"]), "<f4"
    ).reshape(-1, 4)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    for source in [0, 1]:
        selected = v[:, 0] < 7 if source == 0 else v[:, 0] > 8
        local = (
            np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
            * m.geom_size[source]
        )
        expected = local @ d.geom_xmat[source].reshape(3, 3).T + d.geom_xpos[source]
        np.testing.assert_allclose(
            sorted(map(tuple, v[selected])), sorted(map(tuple, expected)), atol=1e-6
        )
        np.testing.assert_allclose(rgba[selected], np.tile(m.geom_rgba[source], (8, 1)))
    assert any(members == (0, 1) for members in p.source_membership)
    assert build(m).geometry_json == p.geometry_json


def test_contained_batch_output_budget_refuses_before_color_allocation(monkeypatch):
    from mjlab_microduck.workbench import geometry

    m = model(
        "".join(
            f'<geom type="box" pos="{x} 4 1" size=".1 .1 .1"/>' for x in [2, 4, 6, 8]
        )
    )
    monkeypatch.setattr(geometry, "MAX_COMPONENTS", 90)

    def no_repeat(*args, **kwargs):
        raise AssertionError("expanded batch colors before component preflight")

    monkeypatch.setattr(geometry.np, "repeat", no_repeat)
    with pytest.raises(ProjectionError, match="MESH_COMPONENT_LIMIT"):
        build(m)
