# Can MicroDuck tour an unfamiliar scene using flyvis?

> **2026-09-15 diagnostic correction:** Earlier performance results below used a
> harness that called `mj_forward` on live runtime data after every control tick.
> A controlled comparison reproduced artificial stalls. These results are retained
> as historical observations, not corrected-runtime performance evidence. Rendering
> now refreshes a separate `MjData` copy; new artifacts require
> `physicsProtocol=runtime_owned_physics_render_copy_v1`. Existing training data and
> checkpoints must be regenerated before assessing autonomous navigation.


## Current mechanism

The experimental controller processes head-camera images through a frozen flyvis
visual model. A learned readout combines those visual features with gyro, projected
gravity, joint positions and joint velocities. It chooses STOP, ADVANCE, TURN_LEFT
or TURN_RIGHT. The existing walking ONNX translates that command and body feedback
into the 14 actuator targets.

The flyvis component supplies visual processing. The learned controller has no explicit itinerary,
map, route state or sensor-based localization. The external checkpoint sequencer
retains destination state, and flyvis retains visual neural activity. Body feedback helps the
controller respond to the robot's condition; adding more channels alone does not
supply destination selection or route planning.

## What the scene-editor check measures

The local `mujoco-scene-editor` State API and export backend generated three new
layouts: an open gallery, offset boxes and a narrow passage. Only exported world
geometry entered the verified MicroDuck model. Robot geometry, mass, joint damping,
actuator gains/ranges and walking policy remained identical. Walls and boxes have
physical collision masks. Goal markers are checked against obstacle volumes.

The readout and flyvis weights remain frozen. Twelve single-target trials compare
three paired fresh seeds (400000–400002) across the original obstacle setup and
three new layouts. Three additional trials allow 60 seconds to follow three
successive visible target spheres, one trial per new layout.

The marker sequencer supplies destination order externally. A checkpoint requires
at least .2 m approach, visible target at .4–.8 m and ±15°, and slow motion held
for .4 seconds. This is stand-off target following, not the robot visiting each
exact coordinate or selecting its own itinerary. Target coordinates, map and
visibility flags never enter the learned controller; they serve scoring and the
external marker sequencer only. All collisions, falls and faults prevent success.

Artifacts live outside Git:
`~/.local/share/unergy/microduck-flyvis/20260915-scene-transfer/`.
The directory contains editable scene JSON/MJCF, the verification harness, paired
records, synchronized videos, manifests and an audit script. It is a verification
spike rather than a retained production navigation API. Preflight debugging
artifacts are excluded from the final report.

## Observed result — 2026-09-15

| Layout | Single-target passes | Guided checkpoints completed | Contact cases (single / guided) |
|---|---:|---:|---:|
| Original single-obstacle reference | 0/3 | Not run | 0 / — |
| Open gallery | 0/3 | 0/3 | 0 / 1 |
| Offset boxes | 0/3 | 0/3 | 1 / 1 |
| Narrow passage | 0/3 | 0/3 | 1 / 1 |

All three guided trials failed. No falls or runtime faults occurred in the 15
trials. The original reference also failed on the three fresh seeds, so these
results do not isolate a scene-change effect. They show that reliable control was
not established even before scene changes. The earlier 4/5 scenario result was a
small sample and did not generalize to this fresh sample.

In the new layouts, seeds 400001 and 400002 received ADVANCE throughout their
single-target trials but made almost no net progress. Seed 400000 moved but lost
target alignment or made contact. The evidence exposes both walking variability
and inadequate steering; it does not identify a single causal defect in flyvis.
No controller was retrained or selected using these results.

**Conclusion:** this frozen controller cannot currently be relied on to complete
a tour in a changed scene. Even the easier externally guided checkpoint task
failed. The three seeds per layout and one guided trial per layout remain an
exploratory sample, not a population success-rate estimate.

## Relationship to ROM navigation

ROM already has a grid route planner and heading-first follower under
`src/mjlab_microduck/rom/navigation/`. Its current environment contract in
`navigation_service.py` declares `SIM_GROUND_TRUTH`: simulator position/orientation
and an installed map. Those are useful for a separate navigation reference, but
connecting that planner would not by itself demonstrate camera-based autonomy.

A sensor-driven tour would need reliable walking across resets, destination/route
state, estimates of robot location and obstacle layout from available sensors, and
recovery when the target or path is lost. These are missing connections in the
flyvis experiment. The existing planner can serve as a reference; any future
sensor adapter must preserve the distinction between estimated and privileged
simulator information.

See [direct locomotion diagnosis](flyvis-locomotion-diagnostic.md) for the
controlled reproduction, harness fix and remaining walking limitations.
