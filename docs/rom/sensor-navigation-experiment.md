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
changes were discarded; the single-marker RGB observation probe was retained.

## AprilTag pose-correction candidate

An additional **offline-only** candidate renders five 14 cm `tag36h11`
markers with a white border in the calibrated scene: one on the desk front,
one above the door header, and three visual-only approach/goal markers. The
camera renders 640×480 RGB every 0.2 s. OpenCV detects tag corners and solves
camera pose against their declared world coordinates. A separate MuJoCo
kinematic state at the origin uses the 14 measured joint positions to recover
the camera-to-trunk transform; it never reads the live base pose. IMU-site
odometry propagates between images. The evaluator alone compares that estimate
with live MuJoCo truth.

The pose observer rejects cropped tags, projected edges under 90 px, >2 px
corner reprojection error, implausible trunk tilt/height, and a single visual
correction larger than 0.08 m or 0.1 rad. Accepted corrections are blended at
half gain to avoid repeated entry/exit at the tight arrival radius. Arrival
requires an accepted tag observation within 1.5 s, with no more than 0.03 m
and 0.08 rad estimated motion since that observation. A missing-tag fault
injection fails `LOCALIZATION_FAILED` with a confirmed stopped command. Static
rendered tests cover translated and rotated robot poses at the desk and door;
their image-based position error was under 0.02 m and yaw error under 0.03 rad
in the tested poses. One static image also tolerates small Gaussian pixel noise
and blur, while half-tag occlusion is rejected. Dynamic image noise, motion
blur, tag placement survey error, and physical marker installation remain
unqualified. The three extra approach/goal markers are visual-only fixtures,
not physical room objects or a TAIROS map update.

To run this candidate locally, install the `rom-vision` dependency group and
choose `--pose-source SIM_VISUAL_ODOMETRY` in the qualification command below.
Its report includes OpenCV version, tag world corners, and generated texture
hashes. The promotion gate remains the full deterministic scenario suite plus
noise, occlusion, and delayed-frame qualification.

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

The first AprilTag 50-case pass reached 48/50. Both failures were safe
`LOCALIZATION_FAILED` stops at the door turn after the last tag had left the
camera view for 1.22–1.30 s; hidden truth and estimated pose were already
inside the goal tolerance, and motion since the final tag fix was less than
0.002 m and 0.047 rad in the reproduced failure. The arrival confirmation
window was extended to 1.5 s while retaining the 0.03 m and 0.08 rad
post-fix motion bounds.

The revised source passed all 50 calibrated runs: ten each of open walking,
door turning, desk detour, unreachable-path rejection, and in-place settling.
The evaluator found zero mapped-obstacle contacts and zero false arrivals;
every run confirmed a stopped command. Maximum hidden-truth pose error was
0.0443 m and 0.0887 rad, below the 0.08 m and 0.15 rad evaluation bounds.
Every reachable run accepted at least 11 visual fixes. The report is
`/home/mcao/production/data/microduck/reports/experiments/2026-09-25-visual-apriltag-doorfix-50.json`;
its runtime source digest matches the revised source. The earlier 48/50 report
is retained at `.../2026-09-25-visual-apriltag-blended-50.json` for comparison.
Five held-out seeds (10–14) across all five scenarios also passed 25/25,
with zero mapped-obstacle contacts, zero false arrivals, and confirmed stopped
commands. Maximum hidden-truth pose error in that batch was 0.0311 m and
0.0620 rad. Its report is
`/home/mcao/production/data/microduck/reports/experiments/2026-09-25-visual-apriltag-heldout-25.json`;
the runtime source digest matches the 50-run report.

Next work must add dynamic camera noise, motion blur, occlusion, and delayed
frame tests, and measure an uncertainty bound for arrival. ToF still needs
head-joint reprojection and invalid/floor classification before it can alter
commands. Physical marker placement and actual hardware odometry require a
separate qualification before any runtime feature flag or robot deployment.

## Physical scene v2 qualification

`src/mjlab_microduck/rom/navigation/calibrated_v2.json` is the canonical v2
scene, profile, and five-scenario set. It keeps the calibrated v1 fixture
available. The desk footprint and the two door-post footprints in its `scene`
are the same rectangles used by the navigation grid and native MuJoCo collision
geometry. The door posts are centered at x = ±0.40 m, leaving a 0.75 m
physical opening. The overhead header is a physical collision geom above the
opening; the marker pads and 17.5 cm white AprilTag planes remain visual only.
The runtime enables the exact v2 obstacle collision set for navigation and
disables it for ordinary action tasks.

The unreachable scenario starts just outside the desk's physical contact
radius but inside the map's conservative clearance zone. This allows both
ground-truth and camera-based controllers to detect `PATH_BLOCKED` without
placing the robot inside a solid tabletop.

Use an immutable, verified locomotion bundle and keep the two reports outside
the bundle. The commands evaluate ten seeds for each of open walking, door
turning, desk detour, unreachable-path rejection, and settling. Each result
checks the expected terminal reason, confirmed stop, mapped-obstacle contacts,
and hidden-truth arrival margins. The visual-odometry report also records tag
fixes and OpenCV/texture identities.

```bash
uv run --group rom-vision python scripts/qualify_rom_navigation.py \
  --bundle /absolute/path/to/verified-bundle \
  --scenarios src/mjlab_microduck/rom/navigation/calibrated_v2.json \
  --seed-count 10 --pose-source SIM_GROUND_TRUTH \
  --output /private/path/v2-ground-truth.json
uv run --group rom-vision python scripts/qualify_rom_navigation.py \
  --bundle /absolute/path/to/verified-bundle \
  --scenarios src/mjlab_microduck/rom/navigation/calibrated_v2.json \
  --seed-count 10 --pose-source SIM_VISUAL_ODOMETRY \
  --output /private/path/v2-visual-odometry.json
```

Before release, inspect every `runs[].passed` value and the collision, stop,
and arrival fields in both reports. Record `runtimeSourceDigest`,
`scenarioDigest`, `bundleDigest`, `mapDigest`, and `profileDigest` together
with the source commit and image ID. A passing v1 report is not qualification
for v2. These deterministic simulator runs do not qualify physical MicroDuck
hardware, marker placement error, dynamic camera noise, or occlusion.

On 2026-09-26, the verified bundle
`sha256:e643915c2f90df9df7bd9b898bb544218eaadb61512ae6d6417ea03112c66f15`
passed 50/50 v2 runs in each mode. Both reports found zero mapped-obstacle
contacts, confirmed a stopped command in all 50 runs, and returned the
expected reason in each of the five scenarios. Ground truth's maximum true
arrival distance/heading were 0.07985 m/0.13653 rad. Visual odometry's were
0.07781 m/0.09972 rad; its maximum estimated-versus-true pose error was
0.05811 m/0.07465 rad, and each reachable run accepted at least five visual
fixes. Each of the ten unreachable visual runs acquired two fixes before
returning `PATH_BLOCKED`.

The reports are private at
`/home/mcao/MyCode/microduck_rl/.worktrees/microduck-scene-layout/.superpowers/sdd/2026-09-26-microduck-native-scene-layout/v2-ground-truth.json`
and
`/home/mcao/MyCode/microduck_rl/.worktrees/microduck-scene-layout/.superpowers/sdd/2026-09-26-microduck-native-scene-layout/v2-visual-odometry.json`.
They share `runtimeSourceDigest`
`sha256:f979ff7a5c0a4d9eb96f7b150da567502ceec50374f7bbf0dfc0316081c5c27d`,
`scenarioDigest`
`sha256:949f1353441d5e0218fbd8f43bc00971a5daed7fc26a566242c0d612d1d87c1a`,
`mapDigest`
`sha256:ef8343d8b5a8a00c3256860e936aebe5c973c7c8d032bc7dcd84cf50ff84fdac`,
and `profileDigest`
`sha256:0d52f5a1e24f7149ac45cd3b33cb98e4dedba542417c539d7c0c03f561f26275`.
These are local qualification results; live duale5 rollout and browser
acceptance remain separate release gates.
