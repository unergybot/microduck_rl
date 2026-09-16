# Direct locomotion diagnosis — 2026-09-15

## Finding and correction

The flyvis harness refreshed target/camera geometry with `mj_forward(model, data)`
on the live runtime state after each control tick. This does more than refresh a
picture: MuJoCo recomputes forward dynamics and derived sensor quantities. A
controlled comparison held the walking policy, robot, command and reset seed fixed.
Adding only the extra live forward call reproduced the stalled motion:

| Seed | Direct forward travel (m) | With extra live forward (m) |
|---|---:|---:|
| 400000 | 0.891 | 0.826 |
| 400001 | 0.887 | 0.003 |
| 400002 | 0.855 | 0.002 |

These are 20-second commands at 0.2 m/s, measured along the initial heading. They
are not commanded-distance guarantees. This finding corrects the earlier claim
that the stalls in those two seeds were attributable to the walking policy alone.

`TrackingWorld` now copies `MjData` before forwarding it for head/outside rendering.
Target placement only writes the intended mocap position. Runtime integration,
actuators and ONNX remain unchanged. A regression compares the full robot trajectory
and policy actions with direct runtime stepping across three seeds × 1,000 ticks;
camera observations are taken during the comparison.

The new `physicsProtocol=runtime_owned_physics_render_copy_v1` is required at dataset
and checkpoint loading boundaries. Old artifacts have `HARNESS_AFFECTED.json`
annotations; their files and source snapshots remain intact. Existing tracking,
scenario and transfer results describe the affected harness and cannot establish
corrected-runtime performance. Training data/checkpoints need regeneration.

## Remaining walking limitations

A separate direct-command battery uses ten consecutive seeds (400000–400009),
20 seconds per command followed by 2 seconds of STOP. It has no visual controller,
route planner or obstacle in the motion area. Each run checks the command actually
received by the 61D ONNX input and records the 14 outputs and actuator targets.

- Forward at 0.2 m/s: seven seeds advanced 0.854–0.891 m along their initial heading;
  seeds 400003, 400004 and 400007 advanced about 3 mm. The moving cases accumulated
  roughly 0.92–1.11 rad of heading drift despite a zero turn command.
- Left turn at +1 rad/s: all ten rotated 19.02–19.30 rad over 20 seconds.
- Right turn at −1 rad/s: all ten rotated −24.42 to −24.11 rad, showing asymmetry.
- Idle: all ten remained close to their starting position.
- All 40 runs settled under zero commands within 1.74 seconds. No fall or runtime
  fault occurred; stopped commands were confirmed in every case.

An exploratory follow-up raised the forward command to 0.3 and 0.4 m/s for the
three stalled seeds. All six runs moved, but accumulated large heading drift.
This supports a low-command startup limitation on those seeds; it does not justify
silently increasing the learned controller's speed or claim that gait reliability
is solved. Command limits and the walking policy were not changed.

## Evidence and next gate

Artifacts: `~/.local/share/unergy/microduck-flyvis/20260915-locomotion-diagnostic/`.
There are 52 diagnostic cases including the three paired reproductions, 40-case
battery and six-case speed follow-up. Two videos show a moving and a stalled direct
run, explicitly labeled without flyvis. Their move/stop measurements exactly match
the unrendered runs. `audit.json` binds the evidence checks.

The next gate is accurate closed-loop locomotion: handle startup stalls and heading
error with measured feedback, then verify a map-based reference route. Only then
regenerate the camera-training data under the corrected physics protocol and repeat
held-out sensor-driven navigation tests. No autonomous tour is qualified here.

## Sensor and scoring timing

Head and outside views are refreshed from a read-only copy of the current integrated
state. Body inputs and scoring telemetry retain the runtime's cached transforms
and sensors from its last physics evaluation; they can differ from the refreshed
image by one physics substep. Control-tick timestamps do not imply all fields have
identical substep sampling. Collision/fall latches use live runtime integration,
not contacts computed for rendering. A future exact image/pose scoring contract
must account for this timing explicitly without forwarding the live control state.

MuJoCo API semantics: [mj_forward and simulation functions](https://mujoco.readthedocs.io/en/3.10.0/APIreference/APIfunctions.html#mj-forward).

Follow-up: [feedback-assisted walking reference](flyvis-feedback-reference.md)
records the frozen controller and paired fresh-seed results.
