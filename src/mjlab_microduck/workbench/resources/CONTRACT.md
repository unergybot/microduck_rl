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
