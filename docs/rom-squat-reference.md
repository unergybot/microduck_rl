# MicroDuck ROM squat reference: design, enablement, and deployment record

This note records how `SQUAT_REFERENCE` moved from a catalog-only entry to a
trained, qualified, deployable action, and every operational fact discovered
on the way. It is the handoff for the next person touching this pipeline.

## 1. What the action is

`SQUAT_REFERENCE` is the video-PoC reference-assisted squat:

- task family: `Mjlab-SquatReference-Flat-MicroDuck`
- command profile: `COS_SIN_SQUAT_REFERENCE_PHASE` — the runtime publishes
  `twist = (cos 2πφ, sin 2πφ, 0)` and advances φ over the spec-owned 5.0 s
  period; the policy infers the Blender reference target from the phase
- reset: `DEFAULT_STANDING`, phase starts at 0
- completion: `RETURN_STAND_AFTER_SQUAT` — at phase wrap the runtime settles
  `SUCCEEDED` when the mean-absolute servo error vs the code-owned standing
  pose is ≤ `SQUAT_RETURN_LIMITS.return_pose_error_max_rad` (0.12 rad)
- evidence metrics: `minimumCrouchHeightM` (episode-minimum base height,
  lower = deeper crouch) and `returnPoseError`
- safety: `FAIL_ON_FALL`, `HOLD_CURRENT_POSITION` safe stop, empty parameters

## 2. Merged work (all on `unergybot/microduck_rl` `develop`)

| PR | content |
| --- | --- |
| #6 | SQUAT_REFERENCE catalog entry (UNAVAILABLE / REFERENCE_POLICY_UNQUALIFIED) |
| #7 | missing reference artifacts keep `REFERENCE_POLICY_UNQUALIFIED` instead of the generic `POLICY_ARTIFACT_MISSING` |
| #8 | runtime semantics + qualification branches to make the spec `supported=True` |
| #9 | governed phase-zero expected motion for the squat battery |
| #10 | reward shaping: height std 0.04 → 0.008 (weight 3.0), joint std 0.35 → 0.2 |
| #11 | crouch completion bound 0.115 → 0.117 (deployment-model geometry) |
| #12 | qualification tracking aggregation skips rollouts without a tracking series |

Deployment 1.0.3 (`20260902154000-rom-microduck-sim-222855b` image, bundle
`minimal-1.0.3-rom-reason-222855b-28ed6400`) is the currently serving release:
16 actions, `WALK_VELOCITY` + `STAND` AVAILABLE, `SQUAT_REFERENCE` UNAVAILABLE.

## 3. Training record

- Reference archive: `/home/mcao/Downloads/microduck-alpha.npz`
  (200 frames @ 50 fps = 4.0 s, root height 0.1127–0.1215 m, ~9 mm crouch).
- Run 1 (`logs/rsl_rl/squat_reference/2026-09-02_09-21-18_squat_reference`,
  1500 iters, 1024 envs): learns the cycle and the return (returnPoseError
  ≈ 0.07) but never dips below the reset height — governed rollouts pin
  `minimumCrouchHeightM` at exactly 0.1200. Root cause: with height std
  0.04 the standing pose earns ≥ 96.7 % of the height reward, so depth has
  no gradient (fixed by PR #10).
- Run 2 (`.../2026-09-02_11-17-38_squat_reference`, resumed from run 1
  `model_1250.pt` with the shaped rewards): the squat becomes real.
  `model_1850.pt` is the shipped checkpoint: 3/3 seeds settle
  `RETURN_STAND_AFTER_SQUAT` at 250 steps, `minimumCrouchHeightM` ≈
  0.1156–0.1160 (≈ 4–4.4 mm dip), `returnPoseError` ≈ 0.051.

## 4. The serving model is a welded kinematic fixture (important)

Every release up to 1.0.3 packages `models/robot.xml` = a minimal Apache-2.0
test MJCF (`microduck-runtime-fixture`): trunk `weld`ed to the world,
`gravity="0 0 0"`, 14 yaw-axis hinges. The walk/stand "policies" it serves
are synthetic zero-weight ONNX stubs (walk bias 0.13, stand all-zero); they
cannot fall on a welded robot, which is why their qualification passed.

Consequences:

- on the fixture, base height is constant 0.1200: a real crouch metric can
  never move, and the squat completion bound can only be met by a policy
  that does nothing (meaningless). The fixture cannot serve a real squat.
- on the real model (`scene_walk.xml`), the walk/stand stubs fall within
  15–36 steps, so the stubs are fixture-only artifacts.

## 5. The validated 1.0.4 real-model release (built, qualified, NOT deployed)

- candidate `1.0.4-candidate.20260902134000.5a653c7` (sha256:5f0cfa7c…),
  model = real `scene_walk.xml` closure, license
  `LicenseRef-MicroDuck-CC-BY-SA-NC` / `DEVELOPMENT_ONLY`.
- provenance triple (all three artifacts match):
  `source_commit 5a653c73820c02c0e26880be9686d734efd77c82`,
  `checkpoint model_1850.pt`,
  `run_identity logs/rsl_rl/squat_reference/2026-09-02_11-17-38_squat_reference/model_1850.pt`.
  The walk/stand stub ONNX are re-stamped to this triple (graphs unchanged).
- qualified inside the production image
  `20260902150000-rom-microduck-sim-5f19fd4` (MuJoCo 3.10): SQUAT_REFERENCE
  PASSED (crouch mean 0.1158 ≤ 0.117, success 1.0, 0 falls);
  WALK_VELOCITY / STAND FAILED (stubs fall) → UNAVAILABLE /
  QUALIFICATION_FAILED in the promoted catalog.
- promoted bundle published (not deployed):
  `releases/minimal-1.0.4-rom-real-squat-50538f1a` (bundle
  sha256:50538f1a…).

Why not deployed: switching serving to the real model would drop walk/stand
from the catalog (their stubs fail there) and changes the model license to
DEVELOPMENT_ONLY. Two decisions belong to the maintainer:

1. train real WALK_VELOCITY / STAND policies (the infrastructure in this
   note applies directly: same training command, `--checkpoint-file` export,
   same qualification path), then
2. accept a locally-loaded image (no registry push) or clear the model
   license for registry distribution.

## 6. Pipeline commands (duale5, `/home/mcao/MyCode/microduck_rl`)

Training (HomeLab venv; repo `.venv` torch is cu128 and cannot run Pascal):

```bash
source /home/mcao/MyCode/HomeLab/.venv/bin/activate
export MICRODUCK_REFERENCE_MOTION=/home/mcao/Downloads/microduck-alpha.npz
python -m mjlab.scripts.train Mjlab-SquatReference-Flat-MicroDuck \
  --env.scene.num-envs 1024 --agent.max_iterations 1500 \
  --agent.save_interval 50 --enable-nan-guard True --agent.logger tensorboard \
  --env.sim.mujoco.jacobian sparse --env.sim.mujoco.solver cg \
  --env.sim.mujoco.iterations 10 --env.sim.mujoco.ls_iterations 20
# resume: --agent.resume True --agent.load_run <run-dir> --agent.load_checkpoint model_N.pt
```

Export (needs the same sim overrides; `scripts/export.py` does not accept
`--env.*`, so patch `ManagerBasedRlEnv.__init__` in a wrapper):
`--checkpoint-file <run>/model_N.pt --onnx-file <out> --num-envs 64`.
The ONNX metadata carries the provenance triple: `run_identity` is the
checkpoint path exactly as passed, `source_commit` is git HEAD.

Bundle + qualify: `scripts/build_rom_bundle.py` with `--source-commit`,
`--checkpoint`, `--experiment-ref` EXACTLY equal to the ONNX metadata
(mismatch ⇒ `POLICY_PROVENANCE_MISMATCH`); candidate `--release` must be a
candidate version distinct from the release-config `release`. The qualify
step (`scripts/qualify_rom_bundle.py`) must run inside the production image
(mount repo at `/work:ro`, candidate at `/build`, entrypoint override to
`/app/.venv/bin/python`): host MuJoCo 3.12 evidence fails the serving
preflight of the MuJoCo 3.10 image (see
`build/quarantine-invalid-host-mujoco-3.12-eeb44dc3`).

Container swap (as done for 1.0.3): stop + rename the running
`unergy-rom-microduck-sim` as a timestamped backup, `docker run` the new
image with the release dir at `/bundle:ro`, shared `/state`, bearer token
file, `--network unergy_prod`, `--restart unless-stopped`, the healthcheck
(python urllib GET `/v1/health`, 15 s interval, 30 s start period), then
verify `/v1/ready` + `/v1/catalog` from inside the container (the token file
is 10001-owned; read it there).

## 7. Rollback inventory

- Images: `111.230.197.171:5000/unergy/unergy-rom-microduck-sim:<ts>-rom-microduck-sim-<sha>`
  (`…f5c76e3` = 1.0.2, `…222855b` = 1.0.3 serving, `…5f19fd4` = 1.0.4 tooling).
- Bundles: `releases/minimal-1.0.1-*`, `minimal-1.0.2-rom-squat-f5c76e3-f4c3c1e5`,
  `minimal-1.0.3-rom-reason-222855b-28ed6400` (serving),
  `minimal-1.0.4-rom-real-squat-50538f1a` (validated, not deployed).
- Stopped backups: `unergy-rom-microduck-sim-backup-20260902051000`,
  `unergy-rom-microduck-sim-backup-20260902160000`.
- Rollback = `docker run` the previous image with the previous release dir.
