# MicroDuck Native Scene Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a versioned MicroDuck MuJoCo navigation scene whose physical desk and door posts match its map, while preserving v1 behavior.

**Architecture:** A canonical v2 scene document supplies obstacle rectangles to both the planner grid and MJCF generator. Version-aware runtime code activates the exact generated collision set for navigation, retains visual scenery for other actions, and leaves v1 untouched. Qualification and immutable deployment use the new scene digest.

**Tech Stack:** Python 3.12, Pydantic, MuJoCo, OpenCV AprilTag, pytest, ROM MicroDuck simulator container.

**Spec:** `docs/superpowers/specs/2026-09-26-microduck-native-scene-layout-design.md`

## Global Constraints

- Keep `microduck-navigation-calibration-v1`, its installed map, and historical qualification valid.
- v2 desk footprint: `[0.4, 0.6] x [-0.1, 0.3]` m; door posts: `[-0.325, -0.275] x [0.975, 1.025]` m and `[0.275, 0.325] x [0.975, 1.025]` m.
- Keep `home`, `desk`, and `door` IDs/poses and the 0.55 m door opening. The approved 0.15 m robot radius, 0.05 m clearance, and 0.05 m grid half-cell diagonal leave the center grid cell free.
- AprilTag IDs/poses and 0.0875 m plane half extents remain unchanged; tags and landmark pads never collide.
- Desk, posts, and overhead header collide for navigation; non-navigation actions retain collision-free scenery. v2 scenery remains visible in both modes.
- Reject unmapped static colliders; do not change robot, policy, observation, actuator, or promote visual odometry automatically.
- Preserve exact simulator source, scene, model, bundle, and qualification identities in release evidence.

## Review Focus

1. A scene that claims v2 but moves one post must fail at load and before MuJoCo compilation; Task 1 tests it.
2. A valid scene with an unknown additional static collider must invalidate navigation; Task 3 tests it.
3. An ordinary action following navigation must have no active environmental collision while retaining v2 visuals; Task 3 tests it.
4. A door post close to a map cell boundary must remain blocked after profile inflation while the door center remains reachable; Task 2 tests it.
5. A tag partly hidden by the physical header or desk at an approach pose must fail the rendered-camera acceptance; Task 4 tests it.

---

### Task 1: Canonical v2 scene contract

**Files:**
- Create: `src/mjlab_microduck/rom/navigation/calibrated_v2.json` (one document with `scene`, `profile`, and the five qualification scenarios).
- Create: `src/mjlab_microduck/rom/navigation/calibrated_scene.py` (load and validate the canonical scene).
- Modify: `src/mjlab_microduck/rom/navigation/installation.py:39-70` (fail closed on noncanonical v2 installations).
- Modify: `docker/rom-simulator/Dockerfile`, `.dockerignore`, `docker/rom-simulator/Dockerfile.dockerignore` (ship the canonical module and JSON in the exact simulator source closure).
- Test: `tests/test_rom_navigation_installation.py` (existing file; add v2 cases).
- Test: `tests/test_rom_navigation_geometry.py` (v1 regression and v2 map contract).
- Test: `tests/test_rom_process_container.py` (container file allowlist).

**Interfaces:**
- Produces `V2_REVISION = "microduck-navigation-calibration-v2"`.
- Produces `canonical_v2_scene() -> Scene`, reading the `scene` field of the package JSON.
- Produces `validate_v2_scene(scene: Scene) -> None`; no-op for other revisions, `ValueError("noncanonical v2 scene")` for any v2 mismatch. Compare normalized `Scene.model_dump()` values, including area, ordered obstacles, and landmarks.
- The CLI can consume `calibrated_v2.json` directly through its existing `--scenarios` option.

- [ ] **Step 1: Write failing tests** for the exact three rectangles and landmarks, rejected moved/missing/reordered post, accepted v1, `load()` rejecting malformed v2, and both new files present in the simulator Docker copy/allowlist.
- [ ] **Step 2: Run** `uv run --with pytest pytest -q tests/test_rom_navigation_installation.py tests/test_rom_navigation_geometry.py tests/test_rom_process_container.py`; expect the new v2 cases to fail.
- [ ] **Step 3: Add the canonical JSON and `canonical_v2_scene()` / `validate_v2_scene()`** with the values above; call validation from installation load immediately after `Scene.model_validate`, and include both files in Docker and ignore rules.
- [ ] **Step 4: Run** the same command; expect all tests to pass. Check `git diff --check`.
- [ ] **Step 5: Commit** only the files in this task with `feat(rom): define canonical MicroDuck navigation scene v2`.

### Task 2: Generate physical native geometry from mapped footprints

**Files:**
- Modify: `src/mjlab_microduck/rom/navigation/environment.py:7-117`.
- Modify: `tests/test_rom_navigation_geometry.py`.

**Interfaces:**
- Consumes `validate_v2_scene(scene: Scene) -> None` from Task 1.
- Produces `v2_collision_geom_names(scene: Scene) -> frozenset[str]`, containing every desk top/leg, both post, and overhead-header geom generated for v2.
- `add_geometry(model_path, scene)` retains its existing signature and v1 branch. For v2, take the three obstacle rectangles from `scene.obstacles`, build the tabletop/legs and posts at their mapped footprints, and place the header above the 0.25 m robot with its lower face at 0.37 m.

- [ ] **Step 1: Write failing MuJoCo tests** for the three obstacle footprints, desk top/legs, two post names, header lower face and span, non-colliding tags/pads, unchanged v1 geometry, and grid blocking around each post while the door center and desk approach remain free.
- [ ] **Step 2: Run** `uv run --with pytest pytest -q tests/test_rom_navigation_geometry.py`; expect v2 cases to fail.
- [ ] **Step 3: Implement v2 geometry and `v2_collision_geom_names()`** in `environment.py`; generate all planar sizes/positions from `scene.obstacles`, not duplicate numeric positions. Keep v1 code path intact.
- [ ] **Step 4: Run** the geometry tests and `git diff --check`; expect pass.
- [ ] **Step 5: Commit** only these files with `feat(rom): build physical desk and doorway from scene map`.

### Task 3: Runtime collision and display lifecycle

**Files:**
- Modify: `src/mjlab_microduck/rom/mujoco_runtime.py:220-260,860-880`.
- Modify: `src/mjlab_microduck/rom/navigation/environment.py` (only if needed to initialize v2 visual alpha).
- Test: `tests/test_rom_navigation_runtime.py`.
- Test: `tests/test_rom_mujoco_runtime.py` (exact action-to-navigation transition).

**Interfaces:**
- Consumes `v2_collision_geom_names(scene: Scene) -> frozenset[str]` from Task 2.
- The runtime records the validated v2 set after model compilation. The static-collider guard allows precisely that set, and only when its corresponding validated installation is active; unknown fixed colliders still disable navigation.
- `start()` activates the exact set for a navigation task and deactivates it for an ordinary action. For v2, alpha remains visible; v1 keeps its existing alpha and collision behavior.

- [ ] **Step 1: Write failing tests** for v2 navigation physical contacts, ordinary-action collision-off/visual-on, transition from navigation to ordinary action, v1 unchanged behavior, and unknown fixed collider rejection.
- [ ] **Step 2: Run** `uv run --with pytest pytest -q tests/test_rom_navigation_runtime.py tests/test_rom_mujoco_runtime.py`; expect new v2 cases to fail.
- [ ] **Step 3: Implement exact-set activation and guard** without broadening name-prefix trust; preserve v1 branch.
- [ ] **Step 4: Run** the same tests plus `git diff --check`; expect pass.
- [ ] **Step 5: Commit** only these files with `fix(rom): activate mapped native collisions for navigation`.

### Task 4: AprilTag v2 compatibility and camera evidence

**Files:**
- Modify: `src/mjlab_microduck/rom/navigation/environment.py:118-180`.
- Modify: `tests/test_rom_apriltag.py`.
- Modify: `tests/test_rom_navigation_vision.py` if its existing fixtures assume v1 only.

**Interfaces:**
- Consumes `validate_v2_scene(scene: Scene) -> None` and the v2 model from Tasks 1-3.
- `add_apriltag_probe(model_path, scene)` continues to use `PROBE_TAGS`; accepts canonical v2 and preserves the existing v1 guard. Tag plane `size` remains `0.0875 0.0875 0.001` and `contype=conaffinity=0`.

- [ ] **Step 1: Write failing tests** for canonical v2 accepted, moved-post v2 rejected, unchanged six tag IDs/poses/plane size, and rendered detection at home, desk approach, and door approach. The rendered test uses `MICRODUCK_TEST_BUNDLE` and records a skip only when no explicit verified bundle is available.
- [ ] **Step 2: Run** `uv run --with pytest pytest -q tests/test_rom_apriltag.py tests/test_rom_navigation_vision.py`; expect new v2 cases to fail or the opt-in render to skip without a bundle.
- [ ] **Step 3: Extend the calibrated-scene guard** in `add_apriltag_probe()`; do not change `PROBE_TAGS` or camera calibration.
- [ ] **Step 4: Run** the same tests with an explicit verified bundle for the render gate, then `git diff --check`; expect pass.
- [ ] **Step 5: Commit** only these files with `test(rom): preserve AprilTag localization in scene v2`.

### Task 5: Full local qualification and operator instructions

**Files:**
- Modify: `docs/rom/sensor-navigation-experiment.md` (v2 identities and qualification procedure, no premature pass claim).
- Modify: `docs/rom-simulator.md` (new scene installation and rollback steps).
- Modify: `scripts/qualify_rom_navigation.py:145-155` only if the header's collision name is not already included by the existing prefix; keep collision reporting exact.
- Test: `tests/test_rom_navigation_live_bundle.py` (opt-in v2 installed bundle path).

**Interfaces:**
- Consumes `src/mjlab_microduck/rom/navigation/calibrated_v2.json` as `--scenarios` and an explicitly selected verified immutable bundle; creates separate ground-truth and visual-odometry JSON reports bound to exact scene/source/bundle digests.

- [ ] **Step 1: Add failing opt-in live-bundle tests** that install the canonical v2 `scene` and `profile`, submit door and desk tasks, and assert terminal stop/zero, no mapped collision, and v2 scene digest in capabilities.
- [ ] **Step 2: Run** `MICRODUCK_TEST_BUNDLE=<verified-absolute-path> uv run --with pytest pytest -q tests/test_rom_navigation_live_bundle.py`; expect v2 cases to fail before implementation is complete. If no verified bundle exists, stop this gate and record the blocker.
- [ ] **Step 3: Complete the minimal qualification collision reporting or fixture wiring** needed for v2; update the two operator documents with immutable installation, digest, rollback, and evidence requirements.
- [ ] **Step 4: Run** the narrow unit suites, opt-in live-bundle tests, and `uv run python scripts/qualify_rom_navigation.py --bundle <verified-absolute-path> --scenarios src/mjlab_microduck/rom/navigation/calibrated_v2.json --seed-count 10 --pose-source SIM_GROUND_TRUTH --output <private-ground-truth-report>`; expect all 50 cases to pass with no collision and confirmed stop.
- [ ] **Step 5: Run** the same qualifier with `--pose-source SIM_VISUAL_ODOMETRY --output <private-visual-report>`; expect all 50 cases to pass with tag fixes and hidden-truth margins inside the approved thresholds. Report this separately from ground truth.
- [ ] **Step 6: Check** `git diff --check`, exact scene/source/bundle digests, and both report outcomes; commit only the task files with `docs(rom): document and qualify physical scene v2`.

### Task 6: Release and duale5 acceptance

**Files:**
- No source changes; save private release and rollback evidence outside Git.

**Interfaces:**
- Consumes the reviewed branch, exact verified bundle, canonical v2 scene, two passing qualification reports, and the documented release procedures. Produces immutable simulator image identity, protected v2 installation, route/collision/visual evidence, and rollback record.

- [ ] **Step 1: Review the completed branch and run** `uv run --with pytest pytest -q tests/test_rom_navigation_geometry.py tests/test_rom_navigation_runtime.py tests/test_rom_apriltag.py tests/test_rom_navigation_live_bundle.py`; expect no failures or unexplained skips. Confirm `git status --short` is clean and record the exact commit.
- [ ] **Step 2: Follow** `docs/rom-simulator.md` and unergy-platform `docs/deploy/rom-microduck-simulator.md` to build the simulator image from that exact commit and verify image ID/digest. Record current duale5 container/image IDs and health before any mutation.
- [ ] **Step 3: Compile** a fresh TAIROS map artifact from the exact v2 navigation installation with unergy-platform `deploy/scripts/microduck-tairos-map-compile.py`; inspect its occupied desk/post cells, free doorway, scene revision, and input digest before upload.
- [ ] **Step 4: Stage** a protected v2 navigation installation and provider map with the exact canonical scene/profile and qualification digests; preserve the v1 installation, immutable image, provider map, and rollback files. Use the existing preflight and Compose release entry points; do not edit a running container or reuse a mutable tag.
- [ ] **Step 5: Verify** live MuJoCo desk/post contact behavior, doorway traversal, desk route, stop/zero, matching provider map and simulator scene, current model/bundle/scene identities, Boot and simulator health, and Web viewer visual appearance. Do not infer physical collision from a screenshot.
- [ ] **Step 6: Record** pass/fail and restore the previous immutable image, v1 installation, and provider map if any gate fails. Keep qualification and rollout evidence private and free of tokens.
