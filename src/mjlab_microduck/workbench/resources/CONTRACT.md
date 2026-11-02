# ROM training experiment contract v1

`catalog.json` is the portable source for Java/TypeScript consumers. It contains
35 recipes in 14 action families and two named robot profiles. Recipe defaults
and command ranges were read from the registered upstream task configuration
factories at the recorded source commit. The reference studio supplied family
and variant coverage, not implementation code. The extra local SquatReference
task is outside this canonical set.

`request.schema.json` defines user intent. `experiment.schema.json` defines the
resolved immutable value. Unknown properties are rejected. `config` can be
omitted and defaults to `{}`; `schemaVersion`, `recipeId` and `robotProfileId`
are mandatory. Python integer overrides require integer values, excluding
booleans and fractional or floating point values. Real-valued fields permit
finite integer or floating point values inside their inclusive bounds.

The current bounded override surface is numEnvs, iterations, seed,
numStepsPerEnv, learningRate, episodeLengthSeconds and actionScale. The effective
configuration also freezes servo settings, 50 Hz control, observation
normalization and `actionFilter: "NONE"`. No action EMA is introduced. Advanced
PPO/network/reward/physical overrides and curriculum sample scaling belong to
the subsequent engine adaptation task; discovery does not promise those APIs.
Factory defaults are retained even where a training CLI normally overrides them
(e.g. the factory scene default is one environment). Resource admission is a
separate existing managed-training decision.

A left-kick `recipeId` maps to the registered BallKick `taskId`, together with
`factoryOverrides: {"kick_foot":"left"}`. It must be built from the left-foot
factory configuration, not a renamed right-foot policy. Slope terrain is
explicitly `slope`, despite the historical upstream task name containing Flat.
Command ranges describe initial training configuration; phase command ranges
are not velocity semantics. A joint recipe declares walk and stand teacher
requirements with matching terrain, backlash and robot profile. These describe
dependencies; they do not assert checkpoint availability or qualification.

The HD1910 profile records the published reference M6 settings and parameter
asset identity. This task does not add or load that physical asset. The engine
adaptation must supply it and verify its digest before claiming support. The
XL330 settings retain this checkout's BAM M6 settings. Profiles and catalog
exposure do not grant engine capability or training admission.

## Canonical JSON and digests

Canonical text uses UTF-8, object keys sorted lexicographically (all contract
keys are ASCII), no insignificant whitespace, JSON string escaping with
unescaped non-ASCII Unicode, and arrays in original order. Numbers are finite
plain decimal text without exponent notation or insignificant trailing decimal
zeros; integral numeric values encode as integers and negative zero as `0`.
Thus `1.0`, `1` both encode as `1`, and `1e-6` encodes as `0.000001`. Booleans
remain `true`/`false`. No custom objects, non-string keys or lone Unicode
surrogates are accepted. JSON strings and numbers must be parsed as typed data;
never construct canonical text by textual key sorting or string replacement.

`recipeDigest` is `sha256:` plus lowercase hex SHA256 of the canonical JSON of
the entire exact recipe record in `catalog.json`. `configDigest` is the same
hash format over the entire resolved specification with only `configDigest`
omitted. It includes `recipeDigest`, requested overrides, effective settings,
profile, commands and teacher requirements. Explicitly requesting a default
can therefore have a different identity from omitting it. Recipes remain
identical when a user override changes; experiment identity changes.

Verification re-resolves `requestedConfig` against the named recipe/profile
and compares the entire canonical result, including both digests. A matching
caller-supplied hash alone cannot bless changed effective settings, unknown
properties or derived task mappings. Return a detached verified value.

`example-request.json` resolves to `example-resolved.json`;
`example-resolved.canonical.json` contains its full canonical text followed by
a file newline (exclude that newline when hashing). `example-invalid.json`
changes the effective environment count without changing its identity and must
be refused by semantic/digest verification even though its types are valid.
No schema-only validator establishes semantic or digest verification.

## Engine optional typed edits

The optional groups `ppo`, `network`, `rewards`, `curriculum`, `observations`,
`events`, `commands`, `physics`, `gains`, `calibration`, `teacherCheckpoints` are
preserved verbatim in requested/effective configuration only when supplied.
All old default experiments and digests remain identical. The request schema
pins the editable data types; the engine additionally checks actual named terms,
existing numeric leaf shapes, model names and physical bounds. Functions, class
names, regex selectors, executable source and submitted filesystem paths cannot
be overridden. Group names/term names use bounded symbolic identifiers.

Reward/event/curriculum edits select existing terms, then `params` numeric data;
reward `weight` can be set separately. Observation edits select group and term,
then noise numeric bounds, scalar scale, clip pair, params or bounded delay.
Histories and term ordering remain fixed, preserving 61 inputs. Commands use
existing named numeric ranges/periods; pose command ranges are full 4/6 pairs.
Top-level commandProfile freezes actual effective ranges and semantics after
explicit range edits; effectiveConfig.commands also preserves the exact edits. Explicit
static reward/event/command changes replace their factory curricula unless the
matching curriculum term is also explicitly edited. This prevents a factory
curriculum silently overwriting the submitted setting at step zero.

Curriculum step thresholds count aggregate samples using the upstream training
reference of 4096 environments: actual stage env-step = round(base stage env-step
* 4096 / actual numEnvs). Rollout length changes batching, not the number of
samples at each stage. engine-configuration.json records effective thresholds,
reference count and rollout length. Resume requires the identical resolved
experiment and restores the upstream common_step_counter from its checkpoint.

`physics` supports gravity, solverIterations, named bodies (mass, ipos, inertia),
joints (range, stiffness), contacts (friction,
solref, solimp, margin, gap) and explicitly allowed numeric actuatorParameters.
Physical changes apply to copies of the actual model/BAM configuration.
`calibration.jointZeroOffsets` adjusts default servo reference positions by name
within 0.2 radians; passive joints cannot be calibrated as servos. These edits
are experiment data and carry no physical robot qualification.

Joint recipes require teacherCheckpoints.walk and .stand, each exactly
{jobId, checkpointSha256, configDigest}. The engine resolves private sibling
request/experiment/checkpoint manifests, checks compatible physical,
observation/action/calibration contracts, validates file hashes and uses
weights_only checkpoint loading plus strict actor/normalizer state layouts.
Teacher networks may differ in width and each applies its own frozen empirical
normalizer. A bounded supervised update (coefficient 0.1, at most 1024 rollout
samples) follows PPO using the same raw observations; gravity-z selects walking
or recovery supervision. No teacher parameters or normalizers are updated.
This independent adaptation reuses the installed rsl_rl models/PPO and adds no
reference studio implementation. Its effectiveness requires GPU qualification.

Experimental jobs publish checkpoint-index.json with ten available content-hash
generations, bounded eviction metadata, iteration/job/request/config identities
and a declared retention policy. Legacy jobs retain their two-generation path.
Experimental ONNX packages use ROM_MANAGED_POLICY_BUNDLE_V2 with exact ZIP members
manifest.json and policy.onnx; V1 stays walking-only and unchanged. V2 retains
outer image/source/request pins and distinct inner experimentConfigDigest and
experimentRecipeDigest. It embeds the immutable experiment, profile, action,
command semantics, empirical normalizer proof, executionEnvironment SIMULATOR,
provenance MANAGED and qualified false. Qualified false describes evidence only.

The optional observer taps post-step data from actual training environments,
never advances/reset/commands them, samples at most nine environments at 20 Hz,
and keeps one pending frame plus one in-flight sink write. Geometry is separate
from latest.json; both include job/request/config/model identities. GPU copies
are asynchronous pinned-buffer snapshots and their event synchronization occurs
in the sink worker. Failure is isolated from the original env.step return/error
and PPO. These files are local diagnostic artifacts; transport, permissions and
browser delivery are separate work. CPU tests cannot measure actual GPU overhead.

Named joint damping, armature and frictionloss edits are explicitly unsupported
and rejected with MANAGED_BAM_JOINT_PARAMETER_UNSUPPORTED: BAM owns those fields
and would overwrite raw spec edits. Use physics.actuatorParameters.armature for
the supported whole-actuator motor inertia, friction_base for its Coulomb budget,
and friction_viscous for its runtime viscous budget. These model parameters are
applied before actuator construction, so inertia survives in the compiled model
and friction is used by BAM at runtime. Named passive-joint edits remain refused.
Experimental resume selects available checkpoint-index generations directly; it
does not pass through the legacy latest-only checkpoint pointer. Eviction metadata
does not authorize reading an evicted file, even if a stale file still exists.

For twist, head_pose and body_pose, resampling_time_range is exactly a JSON list
of two finite numbers (booleans excluded), with 0.1 <= low <= high <= 120 seconds.
Malformed shapes, reversed intervals and out-of-bound intervals are refused during
canonical resolution, before an experiment can launch.

Experimental teacher and resume checkpoint reads require an owned regular
checkpoint-index.json, at most 2 MiB, with the complete producer identity
(schema/jobId/requestDigest/configDigest), exact retention policy 10/4096 and
strict available/evicted entry schemas. Job and request identity must equal the
immutable parent request; configuration must equal its verified experiment and
the selected reference. Available entries alone authorize a read. Latest pointers
are never unioned with the index, and missing/invalid indexes have no fallback.
Digests cannot be duplicated or conflict across either list. All checkpoint IDs
are derived from the declared SHA256, and selected bytes must match hash and size.

Metadata/checkpoints use owned regular single-link nofollow/nonblocking FD reads,
131072-byte parent metadata, 2 MiB index and 256 MiB checkpoint ceilings, with
file/path identity rechecks during verification. Group-writable files are accepted
only inside the unchanged private owned directory (matching the trainer storage
boundary); world-writable files are refused. Actual experimental consumers use
bounded verified byte snapshots: frozen teachers keep weights_only=True, while
upstream runner resume consumes BytesIO and preserves iteration/curriculum state.
The public helper's default path return and legacy latest-only behavior remain.
