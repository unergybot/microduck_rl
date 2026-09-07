# Offline SitStand comparison

This CLI evaluates experimental ONNX policies on CPU MuJoCo and ONNX Runtime.
It does not train, synchronize upstream, promote SIT, or alter ROM qualification.
Run from this fixed fork with Python 3.12, numpy, pydantic, mujoco, onnxruntime,
and system `ffmpeg` (libx264). On headless Linux set `MUJOCO_GL=egl`; rendering may
use a GPU, but inference and physics use CPU. `onnx` and `pytest` are test dependencies.

```sh
python scripts/evaluate_sitstand_comparison.py \
  --config /path/to/comparison.json \
  --output-root /data/training/policy_comparisons
```

Copy `example.json`, replace paths and every digest with independently captured
source hashes. Paths resolve against the config directory. `modelSha256` hashes
the root XML. `modelClosure` must list the root XML and every transitively included
XML, mesh, and texture; each path is relative to the root XML parent. Traversal,
symlinks, missing files and mismatched hashes cannot become evaluated evidence.
The closure aggregate hash is SHA256 of compact sorted JSON mapping path to digest.
Do not deduplicate candidates merely because their policy hashes match: the same
weights on another model or action scale are distinct experiments.

The immutable output is `<output-root>/<experimentId>`. A hidden staging directory
is atomically renamed after writing report.json, per-case CSV/MP4 and manifest.json.
Existing experiments are never replaced. The manifest hashes every artifact; the
report binds the inputs and evaluator hash. Consumers should require the manifest
and verify hashes. Rendering/encoding errors retain CSV and metrics with a null videoArtifactId and
explicit videoUnavailableReason. Invalid candidate
inputs remain explicit UNAVAILABLE cases; physical failures remain FAILED cases.

The fixed battery uses seeds 7, 11, 29, 43, 71, 101, 137, 173 and STAND_HOLD,
STAND_TO_SIT, SIT_TO_STAND, STAND_SIT_STAND. Every scenario resets once with seeded
servo perturbations uniform ±.005 rad. The three-phase sequence holds STAND, SIT,
then STAND in one continuous rollout, with no resets. Each phase must acquire 10
consecutive settled 50 Hz samples within 10 seconds, then hold continuously for
30 seconds. A broken hold fails immediately; failed acquisition cannot be retried
past the deadline. STAND uses fork settlement RMSE <=.08 rad, trunk height
.09–.14 m, tilt <=15 degrees, max joint speed <=.5 rad/s. SIT instead uses maximum
joint error <=.08 rad and target height ±.015 m. `maxPoseErrorRad` always reports
maximum individual-joint error, including transition; it is not STAND's RMSE gate.
Falls use the fork .025 m / 75-degree bounds. Actuator clamp counts are control
steps; physical limit violations count joint/substep excursions.

CSV includes post-step joint positions and velocities, inference actions, the
pre-step 61D input observation, commands, phase and aggregate diagnostics. Videos
cover seed 7 in every scenario and the first failure of each candidate. All case
CSV files are retained. ONNX metadata is recorded but missing metadata is allowed
only as experimental provenance, never represented as normalization verification.

Optional candidate settings:

- `homePose`: canonical-order 14-vector; defaults to fork shared observation HOME.
- `sitPose`: canonical-order 14-vector; defaults to fork sitting reset pose.
- `actionScale`: defaults .9; website comparison should explicitly set 1.0.
- `sitHeightM`: defaults .060; both SIT reset and diagnostic target height.
- `sitCommandDelaySeconds`: defaults 0; delays vx=1 command after phase starts.
- `resetPreviousActionOnCommandChange`: defaults false; clears action history at
  actual command edges (including a delayed SIT command) when true. Scenario
  resets always clear it.

Default observation and pose ordering comes from the fork's runtime contracts.
Explicit HOME is applied consistently to observation centering and action targets.
Head/body commands remain zero; SIT uses twist vx=1, STAND vx=0. No action EMA.
At most six candidates keep both cases and artifacts below consumer limits.
