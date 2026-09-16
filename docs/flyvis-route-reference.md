# Planned-route reference — 2026-09-15

## Fresh-seed result

**4/9 completed tours. The route reference is not yet reliable.**

| Layout | Completed tours | Other outcomes |
|---|---:|---|
| Open gallery | 2/3 | One exhausted recovery near checkpoint 1 |
| Offset boxes | 2/3 | One timed out settling at checkpoint 1 |
| Narrow layout (wide detour) | 0/3 | Two deadlines; one exhausted recovery after checkpoint 2 |

All nine had zero recorded robot/obstacle contacts, no falls or runtime faults,
a quiet final half-second of STOP, and all recorded positions inside the execution
monitor region. Successful runs took 55.14–114.20 simulated seconds including final
stopping. Minimum sampled center-to-hazard distance was 0.167 m across the battery;
this is not a measured body-surface clearance or a continuous-time clearance bound.

The open failure's last ten seconds straddled the 10 cm arrival boundary
(9.81–10.12 cm). The offset failure's heading straddled the 0.1 rad boundary
(−0.10066 to −0.08383 rad); zero/turn commands repeatedly interrupted settling.
These traces motivate arrival hysteresis and an interior stopping target, rather
than claiming the route or flyvis capability is solved. No held-out failures were
used to change the frozen controller.

Verification: 21 existing ROM navigation controller/contract tests passed. The
arrival boundary replay distinguishes the old failure from the corrected behavior.
The independent audit verified all nine case summaries/traces, 49 frozen source
files, map/planning hashes, ONNX command slots, ordered quiet arrivals and final
stopping. Audited scoring also checks the clearance region during stopping; all
nine satisfy it. The original runner monitors that region only during route control.

## Scope

This offline experiment connects the unchanged MicroDuck walking ONNX to ROM's
Grid/A* planner and a feedback route follower. It uses the three editor-authored
MuJoCo layouts from the scene-transfer experiment. The controller receives a known
map and **simulator position, yaw and velocity**. It receives no camera or flyvis
features. It is not installed in production ROM and does not establish sensor-only
autonomy, unseen-layout generalization or hardware readiness.

The preceding [straight-line experiment](flyvis-feedback-reference.md) tested a
simpler motion task. Its 20/20 result does not predict route completion, where
turning, settling and restarting near a goal introduce additional failure modes.

## Route and scoring

Three ordered goals are (0.98, 0.08), (1.7, 0.35) and (2.1, 0.7) m; desired yaw is
zero at each. Before controller development, the first goal was relocated from
(1.2, 0), which was too close to an offset box under the planning inflation.
The old cyan floor annotation remains in the scene. Replay maps show the actual
numbered goals in yellow.

Planning uses 2 cm cells, assumed robot radius 10 cm and 10 cm clearance; the
position monitor uses the same assumed radius and 2 cm clearance. Grid inflation
also includes half a cell diagonal. These are experimental assumptions, not a
qualified whole-body envelope. Early corner transitions can use the planning
reserve. The narrow layout takes a wider route around the boxes; it does not
prove traversal of the central narrow gap.

Passing requires all three checkpoints in order, position error ≤10 cm, heading
error ≤0.1 rad, planar speed ≤0.02 m/s and yaw rate ≤0.05 rad/s for 0.5 s at each
arrival. Accepted arrivals must also remain in planning free space. After the last
checkpoint, a two-second zero command must finish with 0.5 s quiet and retain the
final goal tolerance. Contact, fall, deadline or exhausted recovery fails the run.
The audit additionally checks the execution-monitor region throughout all recorded
samples, including stopping. Robot contact with editor boxes/walls is checked at
every physics step; floor contact and noncolliding markers are excluded.

## Development findings

Development used seed 420000. The initial follower completed the open route but
failed the other two:

- Offset boxes: the −0.6 rad/s turn command barely changed heading from this reset;
  the run reached its deadline. Changing turn-in-place magnitude to 1 rad/s,
  independently of the arrival change, completed all three goals.
- Narrow layout: two checkpoints completed, then the next plan failed because
  arrival had accepted a position outside the more conservative planning grid.
  Requiring planning free space during arrival, independently of the turn change,
  completed all three goals. A replay of the observed boundary pose reproduces the
  original acceptance/replanning failure and checks the corrected behavior.

A logging bug initially hid the narrow failure when the path was `None`; a separate
rerun retained the failure and full trace. Original logs and development outcomes
remain in the artifact directory.

The frozen controller combines those changes. Forward command is 0.2 m/s with
cross-track and heading corrections; recovery uses 0.3 m/s for exactly 50 advance
ticks, at most two pulses per segment. Turn-in-place command is ±1 rad/s. The
walking ONNX, 61D input layout, 14D action layout and actuator settings are unchanged.

## Evidence

Artifacts and runnable experiment scripts:
`~/.local/share/unergy/microduck-flyvis/20260915-route-reference/`.

`heldout/manifest.json` records the controller, planning inputs and corrected-source
snapshot before evaluation. Seeds 430000–430002 are tested across the three fixed
layouts without further controller tuning. Fresh reset seeds are not fresh layouts.
`audit.json` independently checks arrivals, stopping, recorded clearance, ONNX
command slots and artifact hashes. Replays refresh a copied `MjData` and must match
the corresponding unrendered trace exactly.

## Next gate

Arrival behavior needs a separate development cycle before sensor learning. The
fresh-seed failures include motion oscillating across the position threshold and
heading corrections repeatedly interrupting the quiet interval. A useful next
experiment is to approach an interior stopping target, separate approach/settle
states with hysteresis, and verify bounded restart behavior. Freeze that design
before using another fresh seed set; retain this evaluation unchanged.

Once the map/pose reference is reliable, replace privileged pose/velocity with an
explicit estimator and compare camera/IMU/joint inputs, including noise, drift and
sensor dropout. Regenerate flyvis training data under the corrected physics
protocol. The necessary comparison is then flyvis versus a retinal baseline and
the privileged reference under the same route/scoring contract. A brain-inspired
visual representation by itself supplies neither localization nor route state.

## Replays

The artifact directory contains two verified 1024×480, 25 fps replays:

- `offset_boxes-430001.mp4`: completed obstacle tour, 2,412 frames.
- `open_gallery-430001.mp4`: retained arrival/recovery failure, 938 frames.

Both include the actual MicroDuck model and a map with goals/path/count, and match
their unrendered result, trace and numeric path exactly. The obstacle replay also
shows editor boxes/walls. Preview frames were visually inspected. An initial
path-comparison assertion treated tuples and JSON lists as unequal; canonical
comparison confirmed identical simulation data. Video audit files bind the evidence.
