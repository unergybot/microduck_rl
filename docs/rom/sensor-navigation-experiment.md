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

Next work must add real visual landmark observations or another bounded
position correction, use measured head joints to place ToF returns in the
body frame, classify floor and invalid ranges, and reject arrival whenever
uncertainty can exceed the remaining tolerance. Re-run the full 50-case suite
and fault injections before considering a runtime feature flag.
