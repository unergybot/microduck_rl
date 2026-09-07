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
