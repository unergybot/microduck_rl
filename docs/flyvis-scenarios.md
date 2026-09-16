# MicroDuck outside-view and obstacle experiment

> **2026-09-15 diagnostic correction:** Earlier performance results below used a
> harness that called `mj_forward` on live runtime data after every control tick.
> A controlled comparison reproduced artificial stalls. These results are retained
> as historical observations, not corrected-runtime performance evidence. Rendering
> now refreshes a separate `MjData` copy; new artifacts require
> `physicsProtocol=runtime_owned_physics_render_copy_v1`. Existing training data and
> checkpoints must be regenerated before assessing autonomous navigation.


This extends the local flyvis tracking experiment with **visible body movement and
physical obstacles**. It uses the same calibrated 14-actuator MicroDuck, frozen
walking ONNX and frozen flyvis visual circuit. The original tracking experiment
and its running training/evaluation jobs are unchanged.

## What the video means

The left panel renders the actual MuJoCo robot and scene from an outside camera.
The right panel shows the simultaneous head-camera image, retinal input, visual
activity, selected command and measured motion. Both views are captured before the
same control tick. The outside view is recorded only for review and is never an
input to the controller. NPZ recordings contain both image streams; replay rebuilds
the synchronized video without attempting to reconstruct physics from commands.

Every video labels its controller role:

- `privileged_geometry_teacher`: a training reference that can see target and
  obstacle geometry. This demonstrates the body/scene and provides supervised
  labels; it is **not evidence of learned flyvis navigation**.
- `learned_sensor_readout`: a learned readout receiving only visual features and
  34 body channels. It has no obstacle coordinates, visibility flag, world pose,
  waypoint, or outside-camera image in its input.

## Scenarios and scoring

Five balanced scenario families cover approach, left/right turn, and an obstacle
on the left/right of the approach corridor. Targets begin 1.1–1.3 m away, outside
the success band; turning cases start at 0.40–0.55 rad bearing. A physical orange
box has half extents 0.08 × 0.10 × 0.14 m. Its collision mask includes the body and
feet. Non-obstacle cases park the box outside the course.

Robot reset retains the episode seed. Target/obstacle placement uses
`PCG64(SeedSequence([seed, 0x54415247]))`, independent of the robot reset stream.
Scenario geometry is versioned as `microduck_motion_obstacles_v2`, separate from
the easier target-tracking distribution. Legacy tracking readouts/datasets are
rejected by the scenario runner.

Passing requires all of:

- At least 0.2 m reduction in target distance and 0.2 m body displacement.
- A freshly rendered final observation after the zero-command settling attempt sees the
  target within ±15 degrees and 0.4–0.8 m.
- No robot/obstacle or robot/target collision, fall, or runtime fault throughout
  the episode and settlement.

Standing still therefore fails even when a target is visible. Contact failures are
latched at control ticks. `obstacleBaseClearanceM` is a geometric distance from the
base center to the box footprint, **not whole-body clearance or a measured sensor**.
The report includes the final measurement timestamp and definition. Performance
reports remain advisory.

The geometry teacher follows entry and exit waypoints around the box before returning to target
tracking. Detours require heading alignment within 6 degrees; the reference can
reorient toward an occluded target using its privileged geometry. Waypoints are used only by the privileged training reference and label
calculation. First verify this teacher can produce the desired motion with the
frozen walking policy; poor teacher trajectories are not useful training targets.

## Run

Use the environment and official model described in [flyvis-tracking.md](flyvis-tracking.md).
Every command requires a new output directory. All timing is offline simulation
time, with 50 Hz locomotion, 25 Hz vision/readout and 100 Hz flyvis integration.

```bash
PY=.venv/bin/python
CLI=scripts/evaluate_flyvis_microduck_scenarios.py
RUN="$HOME/.local/share/unergy/microduck-flyvis/my-scenarios"
MODEL="$HOME/.cache/microduck-flyvis/results/flow/0000/000"
$PY "$CLI" prepare --source "$SOURCE_BUNDLE" --output "$RUN/bundle"

# Explicit privileged reference: inspect actual body motion and collision geometry.
$PY "$CLI" evaluate --bundle "$RUN/bundle" --model "$MODEL" \
  --teacher --count 5 --output "$RUN/teacher-check"

# Start with short collection/training/evaluation before scaling the budget.
$PY "$CLI" collect --bundle "$RUN/bundle" --model "$MODEL" \
  --count 10 --validation-count 5 --seconds 12 --output "$RUN/data"
$PY "$CLI" train --bundle "$RUN/bundle" --model "$MODEL" \
  --dataset "$RUN/data" --epochs 20 --round-episodes 10 --output "$RUN/trained"
$PY "$CLI" evaluate --bundle "$RUN/bundle" --model "$MODEL" \
  --checkpoint "$RUN/trained/selected.npz" --count 5 --seconds 12 \
  --interventions none frozen_camera zero_vision no_body --output "$RUN/evaluation"
$PY "$CLI" replay --record "$RUN/evaluation/none/obstacle_left-300003.npz" \
  --output "$RUN/obstacle-replay.mp4"
```

`--jobs` supports 1–4 workers (default 2). Collection defaults to 30 training and
10 validation episodes; readout training defaults to one DAgger round of 20 more
episodes and 30 epochs. `--kind retina` trains a retinal baseline using the same
budget. This exploratory extension does not replace the original three-seed full
tracking benchmark. Keep held-out results separate from checkpoint selection. Training seeds start at
0, validation at 100000, development teacher probes at 200000, and learned-controller
held-out evaluation at 300000. Teacher probe outcomes can refine the curriculum;
they must not be relabeled as untouched learned-controller tests.

Datasets/checkpoints bind the scenario protocol, bundle and visual identity. Each
training episode is checked, including DAgger episodes. Retinal baseline data can
come from flyvis teacher recordings or retinal rollouts only when the retinal
preprocessing matches. Reports bind source and artifact hashes.

## Body sensors

The learned upper controller currently receives gyro (3), projected gravity (3),
14 joint positions and 14 joint velocities. Simulation gravity projection uses
model attitude; hardware transfer needs an attitude estimate with realistic error.
The model also defines `imu_accel`, which this readout does not yet consume.

Useful subsequent comparisons are accelerometer input, foot contact/load sensing
or estimation, and motor-effort feedback. MuJoCo contact forces are simulation truth
unless an explicit sensor/estimator model supplies them. Adding ToF/depth is a
separate visual-range comparison and must not silently substitute for flyvis
camera processing. Extra sensors require a versioned input contract and retraining.

## Completed exploratory result — 2026-09-15

The v2 run used 10 teacher episodes, 5 validation episodes, 20 training epochs and
one 10-episode DAgger round. The untouched learned-controller test used seeds
300000–300004, one per scenario family, with 20 seconds per episode.

| Input condition | Passed | Robot/obstacle contacts |
|---|---:|---:|
| Normal camera and body | 4/5 | 0 |
| Frozen camera | 0/5 | 0 |
| Zero visual features | 0/5 | 1 |
| No body input | 0/5 | 1 |

Normal-input approach and both turns passed; obstacle-right passed and obstacle-left
failed. No falls were recorded. Five cases are exploratory evidence, not a reliable
success-rate estimate or proof of general navigation. Artifacts and completed audits:
`~/.local/share/unergy/microduck-flyvis/20260915-scenarios-v2/`. The
`learned-obstacle-replay.mp4` there uses the learned sensor readout.

The separate full tracking benchmark completed 1,200 evaluations. Normal-input
flyvis passed 102/150 and the retinal baseline passed 103/150; this does not establish
a flyvis advantage. Its valid artifacts are under `20260915-independent/`; the
earlier shared-random-stream run under `20260915/` remains invalidated.

Changed-scene follow-up: see [flyvis-scene-transfer.md](flyvis-scene-transfer.md).
The fresh reference and editor-layout results did not reproduce reliable approach
behavior; the 4/5 result above must not be presented as general tour capability.
