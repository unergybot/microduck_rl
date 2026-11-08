"""Fixed simulator-only ROM runner. Invoked inside the pinned, network-isolated trainer image."""
import hashlib
import io
import math
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

TASK = 'Mjlab-Velocity-Flat-MicroDuck'


def require(condition, code):
    if not condition:
        raise ValueError(code)


def gpu_sharing_mode(job):
    mode = job['scope'].get('gpuSharingMode', '')
    require(mode in ('', 'EXCLUSIVE', 'SHARED_DESKTOP', 'SHARED_WORKLOADS'), 'MANAGED_GPU_SHARING_MODE_INVALID')
    return mode or 'EXCLUSIVE'


def validate_request(job):
    allowed = {'jobId', 'scope', 'requestDigest', 'kind', 'numEnvs', 'iterations', 'seed', 'timeoutSeconds',
               'parentJobId', 'checkpointSha256', 'idempotencyKey', 'limits', 'experimentJson'}
    require(set(job) <= allowed and job.get('scope', {}).get('environment') == 'SIMULATOR', 'MANAGED_SIMULATOR_REQUIRED')
    require(job.get('kind') in ('COMMISSION', 'TRAIN') and re.fullmatch(r'[-A-Za-z0-9_]{1,80}', job.get('jobId', '')), 'MANAGED_RECIPE_INVALID')
    gpu_sharing_mode(job)
    require(all(type(job.get(k)) is int for k in ('numEnvs', 'iterations', 'seed', 'timeoutSeconds')), 'MANAGED_PARAMETERS_INVALID')
    require(1 <= job['numEnvs'] <= 4096 and 1 <= job['iterations'] <= 20000 and 0 <= job['seed'] <= 2147483647 and 1 <= job['timeoutSeconds'] <= 43200, 'MANAGED_PARAMETERS_INVALID')
    if job['kind'] == 'COMMISSION':
        require((job['numEnvs'], job['iterations'], job['timeoutSeconds']) == (64, 5, 600) and not job.get('parentJobId'), 'MANAGED_SMOKE_REQUIRED')
    else:
        limits = job['limits']
        require(limits.get('normalEnabled') is True and job['numEnvs'] <= int(limits['maxEnvs']) and job['iterations'] <= int(limits['maxIterations']), 'MANAGED_ENVELOPE_EXCEEDED')
    spec = experiment(job)
    return spec['taskId'] if spec else TASK


def experiment(job):
    text = job.get('experimentJson', '')
    if not text: return None
    from mjlab_microduck.workbench import verify_experiment, canonical_json
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'MANAGED_EXPERIMENT_INVALID'); value[key] = item
        return value
    require(type(text) is str and len(text.encode('utf-8')) <= 65536, 'MANAGED_EXPERIMENT_INVALID')
    spec = verify_experiment(json.loads(text, object_pairs_hook=pairs))
    require(canonical_json(spec) == text, 'MANAGED_EXPERIMENT_INVALID')
    require(all(spec['effectiveConfig'][k] == job[k] for k in ('numEnvs', 'iterations', 'seed')), 'MANAGED_EXPERIMENT_PARAMETERS_MISMATCH')
    return spec


def verify_image_experiment(identity, spec):
    from importlib.resources import files
    from mjlab_microduck.workbench import catalog, canonical_json
    digest = lambda value: 'sha256:' + hashlib.sha256(canonical_json(value).encode()).hexdigest()
    descriptor = identity.get('trainingExperimentsV1', {})
    require(set(descriptor) == {'catalogRevision','catalogSha256','schemaSha256','recipeDigests','robotProfileIds'}, 'MANAGED_EXPERIMENT_IMAGE_UNSUPPORTED')
    schema = json.loads(files('mjlab_microduck.workbench').joinpath('resources/request.schema.json').read_text())
    require(descriptor['catalogRevision'] == spec['catalogRevision'] and descriptor['catalogSha256'] == digest(catalog()) and descriptor['schemaSha256'] == digest(schema), 'MANAGED_EXPERIMENT_IMAGE_MISMATCH')
    require(spec['recipeDigest'] in descriptor['recipeDigests'] and spec['robotProfileId'] in descriptor['robotProfileIds'], 'MANAGED_EXPERIMENT_IMAGE_UNSUPPORTED')



def observer_version(release):
    """Resolve optional observer solely from exact installed image-owned proof.

    Invalid V2 proof disables diagnostics; it never rejects normal training or
    silently falls back to V1. No job, environment variable or caller path enters
    this lookup. Resource/read failures also disable the optional observer.
    """
    from importlib.resources import files

    if not isinstance(release, dict):
        return 0
    if 'trainingWorkbenchReadV2' not in release:
        return 1
    paths = (
        'workbench/observer.py', 'workbench/geometry.py', 'workbench/checkpoints.py',
        'workbench/engine.py', 'workbench/resources/request.schema.json',
        'managed_training.py', 'rom/viewer.py',
    )
    channels = ['OBSERVER_FRAME', 'MODEL_GEOMETRY', 'CHECKPOINT_INDEX',
                'ENGINE_CONFIGURATION', 'EXPERIMENT_ARCHIVE']
    try:
        descriptor = release['trainingWorkbenchReadV2']
        if ('trainingWorkbenchReadV1' in release or type(descriptor) is not dict
                or set(descriptor) != {'schemaVersion', 'channels', 'producerSha256'}
                or descriptor['schemaVersion'] != 'ROM_TRAINING_WORKBENCH_READ_CAPABILITY_V2'
                or descriptor['channels'] != channels):
            return 0
        proof = descriptor['producerSha256']
        if type(proof) is not dict or set(proof) != set(paths):
            return 0
        package = files('mjlab_microduck')
        for path in paths:
            digest = proof[path]
            if (type(digest) is not str or not re.fullmatch(r'sha256:[a-f0-9]{64}', digest)
                    or digest != 'sha256:' + hashlib.sha256(package.joinpath(path).read_bytes()).hexdigest()):
                return 0
        return 2
    except Exception:  # noqa: BLE001 -- Invalid optional proof must not reject training.
        return 0

def atomic_json(path, value):
    path = Path(path); temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def artifact(path, kind):
    with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'artifactId': path.name, 'kind': kind, 'sha256': 'sha256:' + digest, 'sizeBytes': path.stat().st_size}


def publish_checkpoint(source, directory, *, retain_history=False):
    source = Path(source); directory = Path(directory)
    temporary = directory / 'checkpoint.tmp'
    with source.open('rb') as src, temporary.open('wb') as dst:
        shutil.copyfileobj(src, dst); dst.flush(); os.fsync(dst.fileno())
    digest = artifact(temporary, 'CHECKPOINT')['sha256'].removeprefix('sha256:')
    final = directory / ('checkpoint-' + digest + '.pt')
    previous = json.loads((directory / 'checkpoint.json').read_text()).get('artifacts', []) if (directory / 'checkpoint.json').is_file() else []
    os.replace(temporary, final)
    value = artifact(final, 'CHECKPOINT')
    atomic_json(directory / 'checkpoint.json', {'success': False, 'artifacts': [value]})
    # Only prune after the durable pointer commits; keep its predecessor for interrupted readers.
    keep = {value['artifactId']} | {a['artifactId'] for a in previous}
    for old in directory.glob('checkpoint-*.pt'):
        if not retain_history and old.name not in keep: old.unlink()
    return value


def resume_checkpoint(job, directory):
    if not job.get('parentJobId'): return None
    parent = job['parentJobId']
    require(bool(re.fullmatch(r'[-A-Za-z0-9_]{1,80}', parent)), 'MANAGED_CHECKPOINT_INVALID')
    manifest = directory.parent / parent / 'checkpoint.json'
    require(manifest.is_file() and not manifest.is_symlink(), 'MANAGED_CHECKPOINT_UNVERIFIED')
    matches = [a for a in json.loads(manifest.read_text()).get('artifacts', []) if a.get('kind') == 'CHECKPOINT' and a.get('sha256') == job['checkpointSha256']]
    require(len(matches) == 1 and bool(re.fullmatch(r'checkpoint-[a-f0-9]{64}\.pt', matches[0].get('artifactId', ''))), 'MANAGED_CHECKPOINT_UNVERIFIED')
    path = manifest.parent / matches[0]['artifactId']
    require(path.is_file() and not path.is_symlink() and manifest.is_file() and not manifest.is_symlink(), 'MANAGED_CHECKPOINT_UNVERIFIED')
    value = artifact(path, 'CHECKPOINT')
    require(value['sha256'] == job['checkpointSha256'] and value in json.loads(manifest.read_text()).get('artifacts', []), 'MANAGED_CHECKPOINT_UNVERIFIED')
    return path


def select_resume_checkpoint(job, directory, spec=None, *, snapshot=False):
    """Experiments select verified available history; legacy keeps latest-only."""
    if spec is None:
        return resume_checkpoint(job, directory)
    if not job.get('parentJobId'):
        return None
    from mjlab_microduck.workbench.checkpoints import trusted_checkpoint
    path, previous = trusted_checkpoint(directory, job, dict(
        jobId=job['parentJobId'], checkpointSha256=job['checkpointSha256'],
        configDigest=spec['configDigest']), snapshot=snapshot)
    require(previous == spec, 'MANAGED_CHECKPOINT_CONFIG_MISMATCH')
    return io.BytesIO(path) if snapshot else path


def load_resume_checkpoint(runner, checkpoint, spec=None, *, map_location):
    """The single Runtime handoff to the upstream state-restoring loader."""
    if checkpoint:
        return runner.load(checkpoint if spec is not None else str(checkpoint), map_location=map_location)


def configure_warp_runtime(wp, capability):
    """Restore mjlab's pinned Warp API and select Pascal's scalar tile backend."""
    if getattr(wp, '__version__', '') == '1.14.0':
        # Optional PCH files exceed the managed container's 256 MiB /tmp.
        wp.config.use_precompiled_headers = False
        if not hasattr(wp, 'context'):
            # mjlab 1.3 reads this module's driver version on every CUDA architecture.
            from importlib import import_module
            wp.context = import_module('warp._src.context')
    if capability != '6.1':
        return
    require(getattr(wp, '__version__', '') == '1.14.0'
            and hasattr(wp.config, 'enable_mathdx_gemm')
            and hasattr(wp.config, 'enable_mathdx_solver'),
            'MANAGED_PASCAL_RUNTIME_UNSUPPORTED')
    # MathDx's tiled solver requires sm_70+. Warp 1.14 supplies scalar GPU
    # implementations of the same operations for the pinned Pascal profile.
    wp.config.enable_mathdx_gemm = False
    wp.config.enable_mathdx_solver = False


def _diagnose(target, method, *args, **kwargs):
    try:
        if target is not None: return getattr(target, method)(*args, **kwargs)
    except Exception: pass  # optional diagnostics never acquire job authority


class MemorySampler:
    """One prompt query, then 0.25 s waits. A failed query stops this daemon."""
    def __init__(self, query, measurements=None):
        self.query = query
        self.measurements = measurements
        self.stopped = threading.Event()
        self.peak = 0
        self.count = self.failures = 0
        self.first = self.last = None
        self.status = 1
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _publish(self):
        if self.measurements is None: return
        values = {'MeasureMemory/deviceStatus': self.status, 'MeasureMemory/deviceSampleCount': self.count,
                  'MeasureMemory/deviceSampleFailures': self.failures}
        if self.count:
            values.update({'MeasureMemory/deviceUsedSampledMaxBytes': self.peak,
                           'MeasureMemory/deviceFirstSampleSeconds': self.first,
                           'MeasureMemory/deviceLastSampleSeconds': self.last})
        self.measurements.device = values

    def start(self):
        _diagnose(self, '_publish')
        self.thread.start()

    def _run(self):
        while not self.stopped.is_set():
            try:
                free, total = self.query(0)
                if (type(free) is not int or type(total) is not int or not 0 <= free <= total):
                    raise ValueError('invalid device query')
                offset = None
                if self.measurements is not None and self.measurements.started is not None:
                    offset = self.measurements.clock() - self.measurements.started
                    if not isinstance(offset, (int, float)) or not math.isfinite(offset) or offset < 0: raise ValueError('invalid sample clock')
                self.peak = max(self.peak, total-free); self.count += 1
                if self.first is None: self.first = offset
                self.last = offset
            except Exception:
                self.failures += 1
                if self.status != 5: self.status = 3
                _diagnose(self, '_publish'); return
            _diagnose(self, '_publish')
            if self.stopped.wait(.25): break
        if self.status != 5: self.status = 2 if self.count else 0
        _diagnose(self, '_publish')

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=.1)
        if self.thread.is_alive(): self.status = 5
        _diagnose(self, '_publish')


class Runtime:
    def __init__(self, *, measurements=None, collector=None):
        self.measurements = measurements
        self.collector = collector

    def preflight(self, job):
        mode = gpu_sharing_mode(job)
        shared = mode in ('SHARED_DESKTOP', 'SHARED_WORKLOADS')
        # Image content hashes are generated from the archived source, never supplied by a job.
        identity = json.loads(Path('/opt/rom-trainer/recipe.json').read_text())
        self._observer_version = observer_version(identity)
        for key in ('recipeDigest', 'modelDigest', 'dependencyDigest'):
            require(identity[key] == job['limits'][key], 'MANAGED_RECIPE_MISMATCH')
        spec = experiment(job)
        if spec: verify_image_experiment(identity, spec)
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=5)
        rows = [line.split(',') for line in raw.strip().splitlines()]
        require(len(rows) == 1 and len(rows[0]) == 3, 'MANAGED_EXACT_GPU_REQUIRED')
        gpu, driver, free = [v.strip() for v in rows[0]]
        require(gpu == job['scope']['gpuUuid'] and driver == job['scope']['driverVersion'], 'MANAGED_GPU_CHANGED')
        version = lambda value: tuple(int(p) for p in value.split('.'))
        require(version(driver) >= version(job['limits']['driverFloor']), 'MANAGED_DRIVER_UNSUPPORTED')
        memory = 'bootstrapGpuMemoryBytes' if job['kind'] == 'COMMISSION' else 'normalGpuMemoryBytes'
        free_bytes = int(free) * (1 << 20) if re.fullmatch(r'[0-9]+', free) else 0
        reserve = 2 << 30 if shared else 0
        require(free_bytes >= int(job['limits'][memory]) + reserve, 'MANAGED_GPU_MEMORY_INSUFFICIENT')
        require(shutil.disk_usage('/var/lib/rom-trainer').free >= max(10 << 30, int(job['limits']['diskBytes'])), 'MANAGED_DISK_INSUFFICIENT')
        import xml.etree.ElementTree as ET
        processes = subprocess.check_output(['nvidia-smi', '-q', '-x', '-i', gpu], text=True, timeout=5)
        inventory = ET.fromstring(processes)
        entries = inventory.findall('.//process_info')
        if shared:
            observations = inventory.findall('./gpu/processes')
            require(len(observations) == 1 and not (observations[0].text or '').strip(), 'MANAGED_GPU_IN_USE')
            # Host executable names are unavailable in the utility container's PID namespace.
            require(all(len(entry.findall('type')) == 1 and
                        (entry.findtext('type') or '').strip() in (('G', 'C+G', 'C') if mode == 'SHARED_WORKLOADS' else ('G', 'C+G'))
                        for entry in entries), 'MANAGED_GPU_IN_USE')
        else:
            require(not entries, 'MANAGED_GPU_IN_USE')
        import torch
        import warp as wp
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'MANAGED_CUDA_REQUIRED')
        capability = '.'.join(map(str, torch.cuda.get_device_capability(0)))
        require(capability in job['limits']['computeCapabilities'], 'MANAGED_GPU_UNQUALIFIED')
        configure_warp_runtime(wp, capability)
        tensor = torch.arange(8, device='cuda:0', dtype=torch.float32)
        require(torch.equal(tensor * 2, torch.arange(8, device='cuda:0', dtype=torch.float32) * 2), 'MANAGED_CUDA_FAILED')
        wp.init(); require(torch.equal(wp.to_torch(wp.from_torch(tensor)), tensor), 'MANAGED_WARP_INTEROP_FAILED')
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        return {'cudaVerified': True, 'warpInteropVerified': True, 'gpuUuid': gpu, 'driverVersion': driver, 'computeCapability': capability}

    def execute(self, job, directory):
        measurements = self.measurements
        sampler = env = observer = None
        owned_collector = self.collector is None
        failed = False
        try:
            task = validate_request(job); evidence = self.preflight(job)
            import mjlab.tasks  # Registers the installed task entry points.
            from dataclasses import asdict
            import torch
            from mjlab.envs import ManagerBasedRlEnv
            from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
            from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
            from mjlab.utils.torch import configure_torch_backends
            from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
            from mjlab_microduck.rom.export_metadata import attach_microduck_metadata
            configure_torch_backends()
            spec = experiment(job)
            _diagnose(measurements, '_observer_datum', 0 if spec else 4)
            if spec:
                from mjlab_microduck.workbench.engine import build_configs
                env_cfg, agent_cfg = build_configs(spec)
                atomic_json(directory / 'experiment.json', spec)
                atomic_json(directory / 'engine-configuration.json', dict(configDigest=spec['configDigest'], curriculumReferenceEnvs=4096, numStepsPerEnv=agent_cfg.num_steps_per_env, effectiveCurriculum={name:{key:value for key,value in term.params.items() if key.endswith('_stages')} for name,term in env_cfg.curriculum.items()}))
            else:
                env_cfg = load_env_cfg(task); agent_cfg = load_rl_cfg(task)
            env_cfg.scene.num_envs = job['numEnvs']; env_cfg.seed = job['seed']
            agent_cfg.seed = job['seed']; agent_cfg.max_iterations = job['iterations']; agent_cfg.logger = 'tensorboard'
            agent_cfg.resume = False; agent_cfg.save_interval = min(50, job['iterations'])
            work = directory / 'work'; work.mkdir(exist_ok=True)
            try:
                sampler = MemorySampler(torch.cuda.mem_get_info, measurements)
                sampler.start()
            except Exception:
                sampler = None
            _diagnose(measurements, 'read_torch', torch.cuda)
            teachers = {}
            if spec:
                from mjlab_microduck.workbench.checkpoints import load_teachers
                teachers = load_teachers(spec, job, directory, 'cuda:0')
            env = ManagerBasedRlEnv(cfg=env_cfg, device='cuda:0')
            require(abs(env.step_dt - .02) < 1e-8, 'MANAGED_CONTROL_RATE_INVALID')
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(wrapped, asdict(agent_cfg), str(work), 'cuda:0')
            try:
                from mjlab_microduck.training_console import observe_runner
                if self.collector is None:
                    self.collector = observe_runner(runner, directory, job)
                else:
                    observe_runner(runner, directory, job, collector=self.collector, measurements=measurements)
            except Exception:  # Optional observations must not change accepted jobs or PPO.
                pass
            if spec:
                if teachers:
                    from mjlab_microduck.workbench.checkpoints import attach_teachers
                    attach_teachers(runner, teachers)
                try:
                    from mjlab_microduck.workbench.observer import attach_observer
                    version = getattr(self, '_observer_version', 1)
                    if version:
                        observer = attach_observer(env, directory, job, spec, version=version, measurements=measurements)
                except Exception:
                    _diagnose(measurements, '_observer_datum', 3)
            checkpoint = select_resume_checkpoint(job, directory, spec, snapshot=spec is not None)
            load_resume_checkpoint(runner, checkpoint, spec, map_location='cuda:0')
            save = runner.save
            def managed_save(path, infos=None):
                _diagnose(measurements, 'begin', 'checkpoint')
                try:
                    require(shutil.disk_usage(directory).free >= 1 << 30, 'MANAGED_DISK_EXHAUSTED')
                    save(path, infos); published = publish_checkpoint(Path(path), directory, retain_history=bool(spec))
                    if spec:
                        index = directory / 'checkpoint-index.json'
                        previous = json.loads(index.read_text()).get('checkpoints', []) if index.is_file() else []
                        generations = [g for g in previous if g['artifact']['sha256'] != published['sha256']] + [dict(iteration=runner.current_learning_iteration, artifact=published)]
                        evicted = json.loads(index.read_text()).get('evictedCheckpoints', []) if index.is_file() else []
                        evicted = (evicted + generations[:-10])[-4096:]
                        generations = generations[-10:]
                        atomic_json(index, dict(schemaVersion='ROM_TRAINING_CHECKPOINT_INDEX_V1', jobId=job['jobId'], requestDigest=job['requestDigest'], configDigest=spec['configDigest'], retentionPolicy=dict(maxCheckpointFiles=10,metadataMaxEntries=4096), checkpoints=generations, evictedCheckpoints=evicted))
                        keep = {g['artifact']['artifactId'] for g in generations}
                        for old in directory.glob('checkpoint-*.pt'):
                            if old.name not in keep: old.unlink()
                    # Bounded immutable checkpoint generations plus the current runner file.
                    for previous in work.glob('model_*.pt'):
                        if previous != Path(path): previous.unlink()
                except BaseException:
                    _diagnose(measurements, 'end', 'checkpoint', 3)
                    raise
                else:
                    _diagnose(measurements, 'end', 'checkpoint')
            runner.save = managed_save
            _diagnose(measurements, 'end', 'initialization')
            _diagnose(measurements, 'begin', 'training')
            try:
                runner.learn(num_learning_iterations=job['iterations'], init_at_random_ep_len=True)
            except BaseException:
                _diagnose(measurements, 'end', 'training', 3)
                raise
            else:
                _diagnose(measurements, 'end', 'training')
            if not (directory / 'checkpoint.json').is_file(): managed_save(str(work / 'model_final.pt'))
            checkpoint_artifact = json.loads((directory / 'checkpoint.json').read_text())['artifacts'][0]
            atomic_json(directory / 'execution.json', {'executionState': 'SUCCEEDED'})
            _diagnose(measurements, 'begin', 'onnx')
            runner.export_policy_to_onnx(str(directory), 'policy.tmp.onnx')
            exported = directory / 'policy.tmp.onnx'
            attach_metadata_to_onnx(str(exported), get_base_metadata(env, run_path=job['jobId']))
            attach_microduck_metadata(exported, task_id=task, source_commit=Path('/opt/rom-trainer/source-revision.txt').read_text().strip(), checkpoint=checkpoint_artifact['artifactId'], run_identity=job['jobId'])
            if spec:
                import onnx
                model = onnx.load(str(exported))
                for key,value in {'microduck.experiment_config_digest':spec['configDigest'],'microduck.recipe_digest':spec['recipeDigest'],'microduck.robot_profile_id':spec['robotProfileId'],'microduck.action_id':spec['actionId'],'microduck.action_filter':'NONE','microduck.command_profile':json.dumps(spec['commandProfile'],sort_keys=True)}.items():
                    entry=model.metadata_props.add(); entry.key=key; entry.value=value
                onnx.save(model,str(exported))
            import numpy as np
            import onnxruntime as ort
            session = ort.InferenceSession(str(exported), providers=['CPUExecutionProvider'])
            require(len(session.get_inputs()) == 1 and session.get_inputs()[0].shape[-1] == 61, 'MANAGED_OBSERVATION_INVALID')
            for sample in (np.zeros((1, 61), dtype=np.float32), np.full((1, 61), .1, dtype=np.float32), np.full((1, 61), -.1, dtype=np.float32)):
                output = session.run(None, {session.get_inputs()[0].name: sample})
                require(len(output) == 1 and output[0].shape == (1, 14) and np.isfinite(output[0]).all(), 'MANAGED_ONNX_INVALID')
            with exported.open('rb') as file: os.fsync(file.fileno())
            os.replace(exported, directory / 'policy.onnx')
            policy = artifact(directory / 'policy.onnx', 'ONNX')
            _diagnose(measurements, 'end', 'onnx')
            evidence.update(onnxFinite=True, observationDimension=61, actionDimension=14, controlHz=50,
                            peakGpuMemoryBytes=max(sampler.peak if sampler is not None else 0, torch.cuda.max_memory_allocated()), policySha256=policy['sha256'])
            return {'success': True, 'executionState': 'SUCCEEDED', 'exportState': 'SUCCEEDED', 'artifacts': [checkpoint_artifact, policy], 'evidence': evidence}
        except BaseException:
            failed = True
            _diagnose(measurements, 'fail_active')
            raise
        finally:
            _diagnose(measurements, 'begin', 'cleanup')
            # Attempt every owned cleanup. Preserve an earlier authoritative failure.
            cleanup_error = None
            diagnostic_cleanup_failed = False
            if sampler is not None:
                _diagnose(measurements, 'read_torch', torch.cuda, final=True)
            for kind, close in (('observer', observer.close if observer is not None else None),
                                ('sampler', sampler.close if sampler is not None else None),
                                ('environment', env.close if env is not None else None)):
                if close is not None:
                    try: close()
                    except BaseException as error:
                        if kind == 'environment': cleanup_error = error
                        else:
                            diagnostic_cleanup_failed = True
                            if kind == 'observer': _diagnose(measurements, '_observer_datum', 3)
            _diagnose(measurements, 'end', 'cleanup', 3 if cleanup_error or diagnostic_cleanup_failed else 2)
            _diagnose(self.collector, 'flush', {})
            if owned_collector: _diagnose(self.collector, 'close')
            if cleanup_error is not None and not failed: raise cleanup_error


def publish_result_bundle(job, directory, result, publisher=None, source_commit=None):
    """Packaging failure does not rewrite an already proven training/export outcome."""
    if result.get('exportState') != 'SUCCEEDED':
        return
    try:
        spec = experiment(job)
        if publisher is None:
            if spec:
                from mjlab_microduck.workbench.bundle import publish_bundle
                publisher = lambda policy,output,pins: publish_bundle(policy,output,pins,spec)
            else:
                from mjlab_microduck.rom.managed_bundle import publish_bundle
                publisher = publish_bundle
        source_commit = source_commit or Path('/opt/rom-trainer/source-revision.txt').read_text().strip()
        checkpoints = [a for a in result['artifacts'] if a['kind'] == 'CHECKPOINT']
        pins = dict(modelDigest=job['limits']['modelDigest'], recipeDigest=job['limits']['recipeDigest'],
                    configDigest=job['requestDigest'], sourceCommit=source_commit)
        if checkpoints: pins['checkpointSha256'] = checkpoints[0]['sha256']
        bundle = publisher(directory / 'policy.onnx', directory / 'policy.zip', pins)
        result['artifacts'].append(bundle)
        result['publicationState'] = 'LOCAL_READY'
    except Exception:
        result['publicationState'] = 'FAILED'
        result['detailCode'] = 'MANAGED_BUNDLE_FAILED'


def run_managed(request_path, runtime=None):
    started = _diagnose(time, 'monotonic')
    request_path = Path(request_path); directory = request_path.parent
    measurements = collector = None
    try:
        job = json.loads(request_path.read_text()); validate_request(job)
        try:
            from mjlab_microduck.training_console import RunMeasurements, Collector
            measurements = RunMeasurements(job, started=started)
            collector = Collector(directory, job, measurements=measurements)
            _diagnose(collector, 'flush', {})
        except Exception:
            measurements = collector = None
        selected = runtime if runtime is not None else Runtime(measurements=measurements, collector=collector)
        result = selected.execute(job, directory)
        # Old injected runtimes carry no instrumentation proof.
        if runtime is not None and measurements is not None:
            _diagnose(measurements, 'end', 'initialization', 5)
            _diagnose(measurements, 'partial_training')
        if result.get('exportState') == 'SUCCEEDED':
            _diagnose(measurements, 'begin', 'package')
            try:
                publish_result_bundle(job, directory, result)
            except BaseException:
                _diagnose(measurements, 'end', 'package', 3)
                raise
            _diagnose(measurements, 'end', 'package', 2 if result.get('publicationState') == 'LOCAL_READY' else 3)
        else:
            _diagnose(measurements, 'end', 'package', 4)
    except Exception as error:
        _diagnose(measurements, 'fail_active')
        _diagnose(measurements, 'end', 'package', 4)
        checkpoint = directory / 'checkpoint.json'
        code = str(error) if isinstance(error, ValueError) and re.fullmatch(r'MANAGED_[A-Z_]{1,55}', str(error)) else 'MANAGED_TRAINING_FAILED'
        execution = directory / 'execution.json'
        trained = execution.is_file() and json.loads(execution.read_text()).get('executionState') == 'SUCCEEDED'
        if trained: code = 'MANAGED_EXPORT_FAILED'
        result = {'success': False, 'executionState': 'SUCCEEDED' if trained else 'FAILED',
                  'exportState': 'FAILED' if trained else 'PENDING', 'detailCode': code, 'artifacts': []}
        if checkpoint.is_file(): result['artifacts'] = json.loads(checkpoint.read_text()).get('artifacts', [])
    try:
        atomic_json(directory / 'result.json', result)
        _diagnose(measurements, 'terminal_written', result['success'])
        _diagnose(collector, 'flush', {})
    finally:
        _diagnose(collector, 'close')
    return 0 if result['success'] else 1


if __name__ == '__main__':
    sys.exit(run_managed(Path(sys.argv[1])))
