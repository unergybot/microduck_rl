# Guarded experiment reproduction

The experiment ran on duale5 in a dedicated Docker container, with no production
mounts, credentials, published ports, or network access. The exact image was:

`sha256:db653ec0ea349ffb8c3996d2b44368c7d24754694582253a501e8cff0b8cc77d`

The first evaluator snapshot was
`d959089e59e8a2c8fa44603c32aaae2f39e9c83c`; the handoff snapshot was
`6fc54a0af89287e8859fb9d4b9b7da4f05d99011`. Both used CPU MuJoCo 3.10.0,
ONNX Runtime 1.24.4 and NumPy 2.4.1. EGL was used only for rendering.

`guarded-run-original-d959089.py` and `guarded-run-original-6fc54a0.py` preserve
exactly the launchers used on the server. Their original locations are:

- `/home/mcao/microduck-comparison/run_isolated_microduck_comparison.py`
- `/home/mcao/microduck-comparison/run_isolated_microduck_handoff.py`

`guarded-run.py` exposes only root/config/source/image identity arguments around
the same orchestration. Run experiments **serially**. It fixes user 10001:10001,
4 CPU, 8 GiB memory and total memory+swap, GPU access for EGL, a readonly root
filesystem and source/input mounts, and a writable dedicated output mount.
It records all `unergy-*` and `tairos-*` container IDs, image IDs, restart counts,
running state and health before/after and approximately every five seconds.
Any changed snapshot or unhealthy running service stops only the experiment
container. The command, image identity, EGL smoke, logs and monitoring evidence
are retained under `run-evidence/`.

Prepare a source archive from a checkout containing the exact evaluator commit:

```sh
git archive --format=tar --output=source-6fc54a0.tar \
  6fc54a0af89287e8859fb9d4b9b7da4f05d99011 \
  src scripts/evaluate_sitstand_comparison.py
scp source-6fc54a0.tar duale5:/home/mcao/microduck-comparison/
```

On the Docker host, extract into a dedicated directory (never point the source
mount at a production checkout or a symlink to one). The already captured input
assets must retain the hashes in `handoff.json`; see the research README and
source inventory. The example below uses the existing dedicated server paths:

```sh
mkdir -p /home/mcao/microduck-comparison/source-6fc54a0
mkdir -p /home/mcao/microduck-comparison/run-evidence
mkdir -p /home/mcao/microduck-comparison/output
tar -xf /home/mcao/microduck-comparison/source-6fc54a0.tar \
  -C /home/mcao/microduck-comparison/source-6fc54a0
chmod o+w /home/mcao/microduck-comparison/output
```

For another run, copy `input/handoff.json` to `input/handoff-repeat.json` and
change only `experimentId` to an unused safe ID. Existing output experiments are
never overwritten. Run the retained CLI on the Docker host:

```sh
python3 guarded-run.py \
  --root /home/mcao/microduck-comparison \
  --config input/handoff-repeat.json \
  --source-dir source-6fc54a0 \
  --source-commit 6fc54a0af89287e8859fb9d4b9b7da4f05d99011 \
  --image sha256:db653ec0ea349ffb8c3996d2b44368c7d24754694582253a501e8cff0b8cc77d
```

The original four-candidate run uses `input/comparison.json`, source directory
`source-d959089`, and commit `d959089e59e8a2c8fa44603c32aaae2f39e9c83c`.
A repeat likewise needs a copied config with a fresh experiment ID.

The container publishes only under the dedicated `output/` directory. Its final
directory initially belongs to UID 10001 with mode 0700. The completed original
runs were made readable by changing only that directory to mode 0755 using the
same UID in a separate isolated container; artifact bytes were not edited.
Copying evidence to the training replay root is a separate publication step.

The retained standard-library analyzers do not change CSVs or artifacts:

```sh
python3 analyze-csv.py /path/to/output/duckblock-handoff-20260907-v1 \
  > handoff-summary.json
python3 compare-repeat.py \
  /path/to/output/duckblock-sitstand-20260907-v1 \
  /path/to/output/duckblock-handoff-20260907-v1 \
  > baseline-determinism.json
```

`analyze-csv.py` verifies all manifest hashes/sizes and reports the 10 MiB artifact
and 2 MiB report limits, case matrix and seed-7 SIT final-second criteria/joint
errors from the final 50 CSV samples. `compare-repeat.py` compares first-run
`web-delay08` with second-run `flag-retain`: status/reason, complete numeric
metrics, and every old CSV column. The three handoff-only CSV columns are excluded
from that exact comparison. Original result: 32/32 cases and 4,996,468 shared CSV
cells matched exactly.
