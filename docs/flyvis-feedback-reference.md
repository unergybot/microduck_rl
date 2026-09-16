# Feedback-assisted walking reference — 2026-09-15

## Result

An offline feedback controller passed the straight-line criteria in **20/20 fresh
reset seeds**, compared with **0/20** for the unchanged fixed forward command.
This is a simulator-position-assisted walking reference. It does not use flyvis,
perform obstacle avoidance, or complete a tour. It is not installed in ROM.

| Paired held-out measure | Fixed command | Feedback reference |
|---|---:|---:|
| Passed all straight-line criteria | 0/20 | 20/20 |
| At least 0.5 m forward progress | 8/20 | 20/20 |
| Final forward displacement range | −0.003–0.892 m | 1.102–1.146 m |
| Worst final heading error | 60.2° | 3.4° |
| Worst lateral deviation | 70.5 cm | 7.3 cm |
| Quiet during final 0.5 s of STOP | 20/20 | 20/20 |
| Falls or runtime faults | 0 | 0 |

Ten feedback runs used one recovery pulse. Two other starts that failed under the
fixed command moved with feedback without a recovery pulse. These results are a
small held-out sample on flat ground, not a general reliability guarantee.

## Controller and information boundary

The walking ONNX, model, actuator settings, 61D observation layout, 14D action
layout and runtime command limits remain unchanged. An external offline controller
selects bounded twist commands:

- Nominal forward command: 0.2 m/s. Actual forward progress is about 1.1 m in 20 s;
  this is not accurate tracking of a 0.2 m/s physical speed.
- Heading correction: integrate the body gyro Z channel at 50 Hz; command
  `clip(-2 * estimatedHeading, -0.6, 0.6)` rad/s.
- Cross-track correction: simulator lateral position relative to the start;
  command `clip(-crossTrackM, -0.1, 0.1)` m/s sideways.
- Startup recovery: after forward simulator velocity remains below 0.015 m/s for
  at least 1 s, temporarily command 0.3 m/s forward. The tested implementation
  emits **51 control ticks = 1.02 s per pulse**, at most two pulses (2.04 s total).
  The audit found and reports the extra initiating tick; the frozen controller
  was not changed after held-out evaluation began.
- Stop is an explicit zero twist for 2 s. Independent scoring requires the final
  0.5 s to stay below 0.02 m/s planar speed and 0.05 rad/s yaw rate.

**Privileged inputs:** lateral position and forward velocity come from MuJoCo.
They are not camera-derived odometry or verified hardware sensors. Gyro integration
also has no noise/bias model here. A physical implementation requires an estimator
and failure handling; this experiment cannot establish sensor-only autonomy.

## Development and verification

Development used previous diagnostic seeds 400000–400009:

1. Heading correction alone: three moving seeds had small heading error but
   approximately 18–20 cm lateral deviation, failing the straight-line criteria.
2. Added lateral correction: the same three seeds passed, with maximum deviation
   under 7.3 cm during motion.
3. Added recovery: the three previously stalled seeds each passed with one pulse.

The controller was then frozen. Held-out seeds 410000–410019 each ran both fixed
and feedback commands, 20 s motion plus 2 s stop. Passing requires at least 0.5 m
forward progress, final heading error ≤0.15 rad, maximum lateral deviation ≤0.1 m,
no fall/fault and a quiet final stop interval. Scoring includes settlement motion.
The independently audited final-stop requirement is stricter than the initial
runner's detection of any quiet interval; all 40 runs satisfy the stricter check.

The audit verifies paired initial telemetry, all artifact/source hashes, the
commands received by the actual ONNX input, gyro integration and pulse lengths.
A single startup forward refresh after parking experiment objects is identical
in both arms. There is no per-tick refresh of the live physics state. Video refreshes
use copied `MjData`; representative replays are compared with unrendered traces.

Artifacts (including runnable spike scripts):
`~/.local/share/unergy/microduck-flyvis/20260915-feedback-control/`.
`audit.json` contains independent scores; `heldout/report.json` binds the frozen
controller and source snapshot. The prototype remains outside the repository and
is not a production navigation module.

## Next navigation gate

Integrate the feedback reference with changing route headings, corner transitions,
arrival/stop behavior and obstacle-clearance constraints, then verify a planned
route in the editor-authored scenes. The straight-line lateral-error bound must
be accounted for in obstacle clearance; passing this test does not qualify narrow
passages. Preserve ROM task ownership, cancellation and explicit simulator-truth
provenance when making that integration.

After the reference route works, regenerate flyvis data under the corrected
physics protocol and compare sensor-driven control on held-out layouts. Earlier
harness-affected datasets/checkpoints remain unsuitable for that claim.

Follow-up: [planned-route reference](flyvis-route-reference.md) tests turns,
checkpoint settling and editor-scene obstacles. Its failures show why straight-line
success alone does not qualify autonomous tours.
