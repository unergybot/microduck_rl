# Embodied MicroDuck visual tracking

> **2026-09-15 diagnostic correction:** Earlier performance results below used a
> harness that called `mj_forward` on live runtime data after every control tick.
> A controlled comparison reproduced artificial stalls. These results are retained
> as historical observations, not corrected-runtime performance evidence. Rendering
> now refreshes a separate `MjData` copy; new artifacts require
> `physicsProtocol=runtime_owned_physics_render_copy_v1`. Existing training data and
> checkpoints must be regenerated before assessing autonomous navigation.


This local experiment connects **head-camera pixels → a frozen flyvis visual circuit →
a learned movement readout → the existing walking policy → MuJoCo sensors**.
The readout also receives angular velocity, projected gravity, and 14 joint positions
and velocities. It chooses STOP, ADVANCE, TURN_LEFT or TURN_RIGHT. This is a visual
circuit with an engineered body-feedback readout, not an emulated whole fly brain.

## Installation and provenance

Use an isolated Python 3.12 environment. From this worktree:

```bash
uv venv .venv
uv export --frozen --only-group rom --only-group flyvis --only-group dev \
  --no-emit-project --output-file /tmp/microduck-flyvis-requirements.txt
uv pip sync --python .venv/bin/python --require-hashes \
  /tmp/microduck-flyvis-requirements.txt
export FLYVIS_ROOT_DIR="$HOME/.cache/microduck-flyvis"
export MUJOCO_GL=egl
.venv/bin/flyvis download-pretrained
```

The official pretrained archive SHA-256 is
`71c78d4070556a536b13b23ee3139cd2788aa2a9d07d430a223b4edead281db1`.
Use `results/flow/0000/000` from that archive. The adapter verifies that a checkpoint
exists and records the checkpoint/connectome hashes. Downloads stay outside Git.
The optional `flyvis[pretrained]` extra pins obsolete dependencies; use the locked
`flyvis` dependency group here instead.

The frozen evaluator caches the checkpoint's expanded parameters and computes its
ReLU/Euler dynamics using sparse target sums. Its outputs are tested against the
official flyvis integrator at `atol=rtol=1e-5`. Separable retinal box filtering is
tested against official BoxEye at `2e-6`. Effective network configuration and actual
loaded topology are hashed; other neural dynamics/activations are rejected.

For the selected calibrated walking model, the exported head camera points backward
into its own head. `prepare` corrects its optical axes **in the experiment copy**;
its attachment position and robot dynamics stay intact. The original and corrected
quaternions are recorded in `experiment-source.json`. This is a simulator optical
correction, not a physical-camera calibration. The experiment target is a dark sphere
of radius 0.08 m at height 0.30 m, with a physical collision geometry.

## Run

Set `SOURCE_BUNDLE` to a verified calibrated walking bundle, and use a new output
directory. The experiment refuses to overwrite an existing bundle/run directory.

```bash
RUN="$HOME/.local/share/unergy/microduck-flyvis/my-run"
MODEL="$FLYVIS_ROOT_DIR/results/flow/0000/000"
PY=.venv/bin/python
CLI=scripts/evaluate_flyvis_microduck.py
$PY "$CLI" prepare --source "$SOURCE_BUNDLE" --output "$RUN/bundle"

# Short end-to-end smoke before the complete experiment.
$PY "$CLI" collect --bundle "$RUN/bundle" --model "$MODEL" \
  --output "$RUN/smoke-data" --count 5 --validation-count 5 --seconds 4
$PY "$CLI" train --bundle "$RUN/bundle" --model "$MODEL" \
  --dataset "$RUN/smoke-data" --output "$RUN/smoke-trained" \
  --seeds 1 --rounds 1 --round-episodes 5 --epochs 5
$PY "$CLI" evaluate --bundle "$RUN/bundle" --model "$MODEL" \
  --checkpoints "$RUN/smoke-trained/seed-1/selected.npz" \
  --output "$RUN/smoke-evaluation" --count 5 --seconds 4

# Full protocol: 80 initial episodes, 20 validation episodes, 2 x 40 DAgger episodes,
# three independent readout seeds, and 50 held-out episodes per controller/intervention.
$PY "$CLI" collect --bundle "$RUN/bundle" --model "$MODEL" --output "$RUN/data"
for KIND in flyvis retina; do
  $PY "$CLI" train --bundle "$RUN/bundle" --model "$MODEL" \
    --dataset "$RUN/data" --kind "$KIND" --output "$RUN/$KIND-trained"
done
$PY "$CLI" evaluate --bundle "$RUN/bundle" --model "$MODEL" \
  --checkpoints "$RUN"/flyvis-trained/seed-*/selected.npz "$RUN"/retina-trained/seed-*/selected.npz \
  --output "$RUN/evaluation"
```

Each command prints progress as JSON. The smoke protocol checks plumbing and finite
values; its short episodes and training budget cannot establish tracking performance.
All physics advances in simulation time: 50 Hz walking, 25 Hz camera/readout, and
100 Hz neural integration with four held-image steps per frame. Neural state is
initialized once per episode. Sensor-selected commands start on the first frame;
training retains startup samples, and tracking scores allow two seconds for acquisition.
A standing pause was rejected in calibration because it suppressed gait startup.
Datasets, readouts and reports bind `startupProtocol=immediate_sensor_control`;
training and evaluation reject incompatible or unversioned inputs.

Robot reset noise keeps the episode seed. Target placement uses a separate stream:
`Generator(PCG64(SeedSequence([episode_seed, 0x54415247])))`. The domain constant
spells TARG. Sharing the initial random stream would let joint positions encode
target bearing and distance, invalidating sensor comparisons. Every episode,
dataset, readout and report records
`randomizationProtocol=body_seed_target_pcg64_seedsequence_54415247_v1`.
Training validates each episode as well as the dataset manifest; evaluation rejects
incompatible checkpoints. Earlier shared-stream artifacts must be regenerated.

Add `--jobs 4` to collection, training or evaluation to run independent episodes in
four local processes. The supported range is 1–8; the default is one. Each worker
owns its simulator, neural state and readout. Episode seeds, splits and budgets do
not change with worker count. Expected runtime faults are recorded, then a fresh
runtime starts the next episode. Large experiments should use parallel workers;
measure local throughput from the smoke report before scheduling the full sweep.

## Training and comparisons

Only the two-layer readout is trained (64 hidden units, ReLU, four actions). Training
uses Adam, class-balanced cross entropy, and normalization fitted on training samples.
The teacher sees simulator target geometry and rendered visibility. The learner's
call receives only sensor features, before privileged truth is fetched. DAgger labels
the learner's visited states; validation selects the readout, and held-out seeds never
enter training. The conventional baseline uses the same 721 retinal inputs and body
feedback, with the same readout architecture and episode budget.

Interventions are `frozen_camera`, `zero_vision`, and `no_body`. The last removes body
feedback only from the upper controller; the walking policy retains all its sensors.
`stale_camera` is an additional fault-injection case. Invalid/stale inputs latch zero
movement; normal target disappearance instead tests the learned STOP behavior.

Target families: stationary, lateral motion (peak 0.01 m/s), receding (0.01 m/s),
disappearance at seconds 8–10, and a 0.02 Nm yaw perturbation for 0.1 seconds at second 8.
Forward 0.2 m/s and yaw ±1 rad/s are **command values**, not measured robot velocities.
Movement changes have an 80 ms dwell; STOP bypasses it.

## Evidence and replay

`report.json` uses `MICRODUCK_FLYVIS_TRACKING_V1`; `report.md` summarizes outcomes.
Each episode has an NPZ sensor/feature trace and JSON motion trace. The first episode
of each scenario family in evaluation also stores camera frames and an MP4 showing
camera, retina, activity features, commands, and measured motion. Training records
features for every episode and camera/video for representative episodes only.

```bash
$PY "$CLI" replay --record "$RUN/evaluation/flyvis-seed-1/none/stationary-200000.npz" \
  --output "$RUN/replayed.mp4"
CUDA_VISIBLE_DEVICES='' MICRODUCK_TRACKING_BUNDLE="$SOURCE_BUNDLE" \
  FLYVIS_TRACKING_MODEL="$MODEL" \
  FLYVIS_TRACKING_REPLAY="$RUN/evaluation/flyvis-seed-1/none/stationary-200000.npz" \
  PYTHONPATH=src $PY -m pytest tests/test_flyvis_tracking*.py \
  tests/test_rom_navigation_contracts.py tests/test_rom_navigation_controller.py -q
```

Passing means at least 80% of visible samples after acquisition are within ±15°
bearing and 0.4–0.8 m, with no fall, target collision, or runtime fault. Reports also
include visibility coverage, disappearance stopping/settlement, reacquisition,
measured speed/yaw, and simulation-to-wall-time ratio. No visible scoring samples is
not a pass. Neural activity is not a spike raster, and neural replay does not establish
deterministic physics replay. Comparison results do not automatically establish a
connectome advantage; inspect failures, coverage, and interventions as well as means.

ROM launch/UI, authenticated artifact serving, obstacle navigation, physical control,
deployment and real-time scheduling are subsequent work. No ROM endpoints or defaults
are changed by this experiment.

## Historical verification before the physics correction (2026-09-15)

The targeted suite passed all 49 tests with the actual calibrated MuJoCo bundle and
official pretrained model, including 500-frame neural parity, independent target
randomization, stale/fallen/collision handling, and existing navigation contracts.
Scoped Ruff checks and formatting passed. A fresh 25-case smoke completed without
falls/collisions; all five stale-camera cases latched exact zero commands. Replayed
video frame 60 exactly matched the originally recorded video and was visually checked.

The randomization-corrected full dataset contains 80 training and 20 validation episodes, with
500 finite sensor frames each. Joint-0/bearing and joint-1/distance correlations
across training resets are 0.023 and -0.089. The completed historical run is under
`~/.local/share/unergy/microduck-flyvis/20260915-independent/`. It still used the
affected physics harness and cannot establish corrected-runtime performance.
Both its dataset and readouts require regeneration. The earlier `20260915/` data and
readouts share a random stream between target placement and joint initialization;
those results were invalidated and must not be used for sensor-comparison claims.

For current corrected-runtime evidence, see the [locomotion diagnosis](flyvis-locomotion-diagnostic.md),
[feedback reference](flyvis-feedback-reference.md) and [planned-route reference](flyvis-route-reference.md).
The latter completed 4/9 fresh-seed tours using privileged simulator pose and a known
map; it does not establish flyvis sensor-only navigation.
