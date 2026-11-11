# DuckBlock / fixed-fork diagnostic evidence

Source: https://jecoprojects.com/duckblock/, captured 2026-09-07 UTC. Runtime fork
`590b986bd8c0d50ae02cb3ea2f59c463b6828168`; RL baseline
`9a03de6c63f80f762144ab00f7b614ef401ab280`. No upstream synchronization.

The website and runtime fork SitStand ONNX are byte-identical:
`c6c40e35e726eabd803d633e090d112994f469921152448367953fbaf9799bc8`.
The site's original robot XML and the fixed RL fork robot_allcollisions.xml are
also identical: `7a6fdf437f5a80c7348ad801f43f906b997a834389cb70cca5e1e8517ba38044`.
The four entries are execution conditions for this one weight identity.

| Condition | Physics step | Action scale | SIT delay | Stand /8 | Stand→sit /8 | Sit→stand /8 | Continuous /8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Website robot + flat plane | .005s | 1.0 | 0 | 8 | 0 | 7 | 0 |
| Same, scale .9 | .005s | .9 | 0 | 8 | 0 | 8 | 0 |
| Same, command delay .8s | .005s | 1.0 | .8s | 8 | 0 | 7 | 0 |
| Fixed fork scene.xml | .002s | .9 | 0 | 8 | 0 | 3 | 0 |

These are strict diagnostic results, not ROM acceptance. The continuous case
fails at its SIT phase and marks subsequent STAND NOT_RUN. In the fork scene,
sit→stand had three falls and two transition timeouts. Each scale-1 flat condition
had one strict standing-hold break. The original historical mixed-controller
experiment retains its own identity; these metrics were not reconstructed from
its video and it has not been rerun here.

For seed7, every final-second SIT sample passed height, tilt and speed but failed
maximum joint-position error. At scale1 the hip-roll errors are approximately
.175/.164rad against the required .08rad; scale.9 also has knee errors around
.148/.138rad. See `baseline-summary.json` for every joint and criterion. This
explains why a convincing seated appearance does not meet this target posture.
Do not treat it as SIT qualification. First reconcile the intended SIT target
with the learned posture and investigate the fork's step-size-sensitive standing
recovery; then choose qualification work or a training objective change.

The browser dependency capture pins31 response bodies (32.8MB) bySHA256 in
`browser-dependency-index.json`; a Cloudflare challenge redirect has no body and
is not a policy/model/runtime dependency. The exact browser versions are
MuJoCo3.11.0 and ONNX Runtime Web1.27.0, Chrome152. The cached bodies remain on
duale5 under`browser/source-cache/<sha256>`, allowing dependency reconstruction
without silently replacing them with newer site assets.

The browser uses raw61D input and14D output at50Hz. ONNX opset18 begins with
Sub(obs, embedded61D mean), Div(embedded61D scale), then the MLP. Metadata includes
`action_scale=1.0` and `run_path=None`: normalization is embedded, training lineage
is not verified. Both fork LICENSE files are Apache2.0; independent model/site
licensing and original training provenance remain explicitly unverified.

`parity-report.json` verifies100 captured observations using the browser WASM
SitStand session and Python ORT1.24.4 with atol1e-5/rtol1e-4. Maximum absolute error
is2.384185791015625e-7. `sitstand-parity.json` retains the exact float inputs and
reference actions. The browser capture visibly sits and returns to walking with
no observed fall. It starts with WALK, clears action history entering SitStand,
waits800ms before SIT=1, then holds SitStand-zero2s before returning to WALK.
A separate handoff ablation reproduces this sequence under fixed diagnostic
initialization; it does not claim identical browser entrance animation or terrain.

`capture.cjs` is the exact research capture used on duale5, with environment paths
retained for reproducibility. Its `states.json` field `second` is a sample index,
not a measured simulation timestamp: screenshot overhead adds wall time. The
browser-reference MP4 is a1fps snapshot replay. No acceptance or timing metrics
are inferred from this video. `model-comparison.json` compares captured browser
XML against the flat diagnostic adaptation: actuator gains/limits/gear and robot
body mass/inertia match. Captured XML alone is not the complete browser world;
the application changes terrain fields at runtime.

The isolated baseline used evaluator d959089, CPU MuJoCo3.10.0/ORT1.24.4,
NumPy2.4.1,4CPU,8GiB and NVIDIA EGL for video. The43 production-container identity,
image, restart and health records matched before/after; a5s guard monitored the
run.128cases produced145 verified artifacts, maximum4,248,445bytes; every requested
video exists. Nothing was placed in logs/rsl_rl and no runtime/qualification
state was changed.

On duale5 the complete pinned inputs are retained at
`/home/mcao/microduck-comparison/input`, the committed source snapshot at
`/home/mcao/microduck-comparison/source-d959089`, and original output at
`/home/mcao/microduck-comparison/output/duckblock-sitstand-20260907-v1`.
`comparison.json` and `source-inventory.json` bind the input files. To rerun, copy
the config to another filename in that input directory, change only experimentId,
then use the committed evaluator in a fresh isolated container with the original
image digest recorded in `run-evidence/.../image.json`:

```sh
docker run --rm --cpus 4 --memory 8g --gpus all --network none \
  --user 10001:10001 --read-only --tmpfs /tmp:rw,nosuid,size=512m \
  -e MUJOCO_GL=egl \
  --mount type=bind,src=/home/mcao/microduck-comparison/source-d959089,dst=/experiment,readonly \
  --mount type=bind,src=/home/mcao/microduck-comparison/input,dst=/input,readonly \
  --mount type=bind,src=/home/mcao/microduck-comparison/output,dst=/output \
  sha256:db653ec0ea349ffb8c3996d2b44368c7d24754694582253a501e8cff0b8cc77d \
  /experiment/scripts/evaluate_sitstand_comparison.py \
  --config /input/repeat.json --output-root /output
```

Use the retained guarded-run script/evidence to record and compare production
containers on every rerun. The output directory is dedicated and writable byUID10001;
no production credentials/state mounts or ports are required. Completed evidence
is separately enriched with browser/source/execution records and atomically
published under the training root's policy_comparisons directory; original
producer output remains immutable.

## Controlled policy-handoff ablation

The second immutable experiment is`duckblock-handoff-20260907-v1`, evaluator
`6fc54a0af89287e8859fb9d4b9b7da4f05d99011`, config`handoff.json`.
Both conditions retain the same flat model,scale1,HOME,seed perturbation and.8s
SIT delay. The only treatment is the website's policy-handoff protocol, including
its previous-action reset at actual policy edges and2s SitStand-zero beforeWALK.
Acquisition must wait for the final scheduled policy/command; the10s deadline
still begins atphaseentry and each successful phase holds30s under the finalpolicy.

| Condition | Stand /8 | Stand→sit /8 | Sit→stand /8 | Continuous /8 |
|---|---:|---:|---:|---:|
| Same SitStand, retained action history | 8 | 0 | 7 | 0 |
| Website WALK/SitStand handoff | 8 | 0 | 8 | 0 |

The handoff removes the observed seed11 stand-up hold break but leaves the SIT
pose failure unchanged. No falls occurred in this ablation. This is evidence for
the limited role of the switch protocol, not authorization to enable runtimeSIT.
`baseline-determinism.json` verifies all32 repeated baseline cases,43,073CSV rows
and4,996,468 shared cells match exactly, including every numeric metric.

The Walking policy is also fixed (`e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c`).
A separate100-sample browser-WASM/Python test passed with maximumabsoluteerror
1.4901161193847656e-7; inputs and outputs are in`walking-parity-samples.json`.
The report binds both policy identities and the frontend displays the additional
Walking SHA whenpresent.

Both completed enriched publications are at
`/home/mcao/MyCode/microduck_rl/policy_comparisons/<experimentId>/`.
`publish-evidence.py` validates original artifact hashes and browser policy/sample
binding, checks the production guard, preserves the original report/manifest,
then uses the evaluator's atomic publisher. It rejects an existing experimentID.
The original baseline publisher bytes are retained at commitdbb96da; the later
version adds the optionalWalking parity evidence for the secondexperiment.

ROM entry: trainingmonitor → 实验回放 → 策略对照. Choose anexperiment, candidate,
scenario andseed; comparison videos play independently and reports/CSV remain
available when no video exists. This work grants no runtimeSIT permission and
starts no training or physicalrobot execution. Deployment and authenticated
browser acceptance are tracked in platform PR https://github.com/unergybot/unergy-platform/pull/674.

## One-factor Walking-weight control

`duckblock-walk-ablation-20260907-v1` closes the acquisition-timing confound above.
Both candidates use the exact same handoff, action-history reset, final-policy
acquisition gate and deadlines. Only the policy assigned to theWALK slot differs:
SitStand itself versus the websiteWalking ONNX. `one-factor-proof.json` binds
that single effective-configuration difference.

| WALK-slot weights under identical protocol | Stand /8 | Stand→sit /8 | Sit→stand /8 | Continuous /8 |
|---|---:|---:|---:|---:|
| SitStand self-handoff control | 8 | 0 | 8 | 0 |
| WebsiteWalking | 8 | 0 | 8 | 0 |

No pass-rate benefit from Walking weights is demonstrated by this battery.
The earlier seed11 baseline acquired at.76s and rejected a hold break at.80s,
before the2s switch. Under identical handoff/acquisition rules, the self-control
acquires at2.18s and Walking at2.30s; both then hold1500steps/30s. The earlier
7/8→8/8 comparison therefore cannot be presented as proof of improved physical
stability fromWalking weights. The exact phase evidence is in`seed11-phases.json`.

The Walking treatment repeats all32 prior cases,45,248CSV rows and5,384,512shared
cells exactly. All64newcases completed, with zero falls or missing requested
videos. All73original artifacts satisfy the10MiB limit and have valid hashes;
43production containers remained unchanged across60guard checks. The enriched
third publication is under the same training-root policy_comparisons directory.

Final recommendation: preserve the current runtime qualification boundary; resolve
the SIT target-versus-learned-posture mismatch and investigate the fork scene's
2ms stand-up sensitivity before qualifying SIT. If the specified target posture
is mandatory, a training objective change may be needed. The evidence does not
justify replacing weights merely to improve the reported acceptance count.
