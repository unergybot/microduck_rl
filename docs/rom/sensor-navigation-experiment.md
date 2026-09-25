# MicroDuck sensor navigation experiment (2026-09-25)

This branch adds an **offline-only** controller pose source. It integrates the
installed MuJoCo model's `imu_ang_vel` and `imu_lin_vel` sensor samples, using
three-axis orientation and correcting velocity for the IMU site's offset from
the trunk origin. The qualification evaluator alone reads MuJoCo world pose.
The local controller still uses the approved static map and walking policy.

`imu_lin_vel` is a simulated velocimeter. MicroDuck hardware does not gain a
linear velocity sensor from this experiment. Leg/contact odometry, camera
landmark correction, realistic noise and dropouts, and uncertainty gating are
required before a hardware or production sensor-mode claim.

The branch also simulates raw 8×8 head ToF rays at the model's `tof` site.
It follows the [upstream MicroDuck ToF reprojector](https://github.com/pollen-robotics/microduck/blob/main/kinematics/src/tof.rs):
sensor +x forward, +y left, +z up, 45° per axis, and no-return cells unknown.
The [VL53L5CX datasheet](https://www.st.com/resource/en/datasheet/vl53l5cx.pdf)
describes the sensor class; these rays do not model cover-glass crosstalk,
multi-target returns, lighting, per-zone status, or calibrated timing. The
ToF scan does **not** currently alter navigation commands.

The corrected experimental head camera can render a known 8 cm radius marker
at roughly 0.8 m. `navigation.vision.detect_red_sphere` extracts only RGB
pixels and estimates range from marker size and camera field of view. Its
real-bundle test compares that estimate with hidden simulator truth outside
the detector. This proves the camera can provide a landmark observation; it
does not yet provide a navigation pose correction. The marker is a solid red
sphere, not an AprilTag. It cannot recover full marker pose, and the current
extractor rejects cropped or non-round blobs.

A separate, unmerged pair-marker trial at the desk exposed the remaining
geometry problem. The desk top occluded markers at 0.42–0.54 m height. Raising
them to 0.65 m made both visible from Home, but they left the 45° camera view
as the robot approached within about 1.2 m of the markers. RGB sphere-size
range estimates were 2.8–5.2% shorter than the evaluator's true camera-to-marker
distance over tested starting positions. This placement and uncalibrated
measurement cannot support an arrival gate. The trial's runtime and scene
changes were discarded; only the single-marker RGB observation probe remains.

## Reproduction

```bash
uv run python scripts/qualify_rom_navigation.py \
  --bundle /absolute/path/to/verified-bundle \
  --scenarios tests/fixtures/navigation/calibrated-scenarios.json \
  --seed-count 10 --pose-source SIM_SENSOR_ODOMETRY \
  --output /absolute/path/to/sensor-report.json
```

The first 50-case run on the calibrated bundle passed 46 cases and failed
four hidden-truth arrival checks: turn seed 7 and desk detour seeds 2, 3, and
5. All four were reported as `ARRIVED` by the sensor controller while the
true position or heading was outside the approved arrival tolerance. There
were no mapped-obstacle collisions. This **fails promotion**. The production
simulator remains on `SIM_GROUND_TRUTH`, which passed 50/50 cases at the same
scene and profile.

Terminal evaluator measurements explain the false positives: turn seed 7 had
estimated/true heading errors 0.123/0.175 rad, detour seed 2 had
estimated/true position errors to goal 0.060/0.095 m, and detour seeds 3 and
5 had true heading errors 0.152 and 0.177 rad. The approved thresholds are
0.08 m and 0.15 rad. Merely tightening integration numerics did not improve
these cases, so that experiment was discarded. The qualifier now records both
estimated and true terminal margins for future uncertainty-gate work; only
the evaluator reads truth.

Next work must add mapped visual landmarks and bounded pose correction from
their observations, use measured head joints to place ToF returns in the
body frame, classify floor and invalid ranges, and reject arrival whenever
uncertainty can exceed the remaining tolerance. Re-run the full 50-case suite
and fault injections before considering a runtime feature flag.
