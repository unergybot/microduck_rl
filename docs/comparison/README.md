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
source hashes. Supply top-level `codeRevisions.rl` and `codeRevisions.runtime`
as full 40-character lowercase commit hashes; branch names are rejected. They
are captured alongside the exact evaluator file SHA256 and dependency versions.
Paths resolve against the config directory. `modelSha256` hashes
the root XML. `modelClosure` must list the root XML and every transitively included
XML, mesh, and texture; each path is relative to the root XML parent. Traversal,
symlinks, missing files and mismatched hashes cannot become evaluated evidence.
The closure aggregate hash is SHA256 of compact sorted JSON mapping path to digest.
Do not deduplicate candidates merely because their policy hashes match: the same
weights on another model or action scale are distinct experiments. Verified runnable
candidates with identical policy/root/closure hashes and effective behavior are
evaluated once; later entries remain in the canonical candidate’s `aliases`.
`effectiveBehavior` records every actual pose, scale, delay and history setting;
`effectiveIdentitySha256` binds these with the artifact identities. Source labels
and local path spellings do not define behavior. Invalid source candidates remain
explicitly UNAVAILABLE and are not hidden by deduplication.

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

Per-case `phases` explicitly report STAND/SIT targets, status, reason, control and
settled sample counts, strict hold sample counts/duration, transition acquisition,
and total duration. Later phases after an early failure are NOT_RUN. Flat numeric
metrics repeat phase pass/hold counts for existing consumers, alongside totals
and `maxPoseRmseRad`, `maxStandPoseRmseRad`, `maxHeightDeviationM`. RMSE measures
against each phase's target (STAND uses HOME); STAND-only maximum excludes SIT.
Height deviation is absolute distance from SIT target or the center (.115 m) of
the STAND acceptance band, while the STAND height predicate remains .09–.14 m.
Only successfully completed physics control steps count towards reported duration;
numerical failure can interrupt the final attempted control step. Hold counts
exclude the 10 acquisition samples and exclude a sample that breaks the hold.

## Optional browser WALK / SitStand handoff

Set `browserPolicyHandoff: true` and provide `walkingPolicyPath` plus
`walkingPolicySha256` to evaluate the website's policy switches. The walking
policy must independently pass digest, float32 61D-input/14D-output and CPU-provider
validation. The existing `policyPath` continues to identify SitStand. Both use
identical physical model, HOME, action scale and observation layout. No physics
reset occurs at a policy switch; previous-action history is cleared on the actual
policy edge. The existing optional `resetPreviousActionOnCommandChange` remains a
separate ablation; leave it false for browser-style policy-only history clearing.

- STAND_HOLD runs WALK with zero commands.
- STAND_TO_SIT enters SitStand immediately, supplies flag 0 until 0.8 seconds,
  then flag 1. `sitCommandDelaySeconds` can explicitly override the 0.8-second
  opt-in default for a separately identified ablation.
- SIT_TO_STAND starts seated under SitStand flag 0, then switches to WALK at
  2.0 seconds, clearing action history.
- STAND_SIT_STAND starts on WALK, enters SitStand for the SIT phase, then uses
  the same 2.0-second return handoff without resetting body state.

The 10-second transition deadline still starts at phase entry. In handoff mode,
settlement acquisition additionally requires the final scheduled policy and
command: WALK after the 2-second return handoff, or SitStand with flag 1 after
its delay. Even an early settled STAND sample cannot start the hold before WALK
is selected. Ten qualifying samples then start a full 30-second continuous hold
under the final policy. Default single-SitStand mode retains its pose-only gate.

CSV `settled` remains the raw pose/height/tilt/speed result; `acquisitionReady`
records the scheduling gate separately. `policyMode` and `policyChanged` identify
each inference and actual policy edge. Phase evidence includes the final policy,
command, acquisition condition and policy transitions with phase-relative times.
The raw config, walking policy digest, effective behavior and deduplication identity
are bound into report provenance and therefore into the manifest report hash.

```json
{
  "browserPolicyHandoff": true,
  "walkingPolicyPath": "inputs/walking.onnx",
  "walkingPolicySha256": "sha256:REPLACE_WITH_64_LOWERCASE_HEX",
  "sitCommandDelaySeconds": 0.8,
  "resetPreviousActionOnCommandChange": false
}
```

This opt-in does not change ROM qualifications or reinterpret older experiment
results. Report `effectiveIdentitySha256` now explicitly includes the mode flag;
compare common CSV fields and numeric outcomes when checking baseline determinism
across evaluator revisions, rather than comparing identity hashes alone.
