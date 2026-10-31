"""Fixed simulator-only ROM runner. Invoked inside the pinned, network-isolated trainer image."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
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
               'parentJobId', 'checkpointSha256', 'idempotencyKey', 'limits'}
    require(set(job) <= allowed and job.get('scope', {}).get('environment') == 'SIMULATOR', 'MANAGED_SIMULATOR_REQUIRED')
    require(job.get('kind') in ('COMMISSION', 'TRAIN') and re.fullmatch(r'[-A-Za-z0-9_]{1,80}', job.get('jobId', '')), 'MANAGED_RECIPE_INVALID')
    require(gpu_sharing_mode(job) in ('', 'EXCLUSIVE') or job['kind'] == 'COMMISSION', 'MANAGED_SHARED_COMMISSION_REQUIRED')
    require(all(type(job.get(k)) is int for k in ('numEnvs', 'iterations', 'seed', 'timeoutSeconds')), 'MANAGED_PARAMETERS_INVALID')
    require(1 <= job['numEnvs'] <= 4096 and 1 <= job['iterations'] <= 20000 and 0 <= job['seed'] <= 2147483647 and 1 <= job['timeoutSeconds'] <= 43200, 'MANAGED_PARAMETERS_INVALID')
    if job['kind'] == 'COMMISSION':
        require((job['numEnvs'], job['iterations'], job['timeoutSeconds']) == (64, 5, 600) and not job.get('parentJobId'), 'MANAGED_SMOKE_REQUIRED')
    else:
        limits = job['limits']
        require(limits.get('normalEnabled') is True and job['numEnvs'] <= int(limits['maxEnvs']) and job['iterations'] <= int(limits['maxIterations']), 'MANAGED_ENVELOPE_EXCEEDED')
    return TASK


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


def publish_checkpoint(source, directory):
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
        if old.name not in keep: old.unlink()
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


class Runtime:
    def preflight(self, job):
        mode = gpu_sharing_mode(job)
        shared = mode in ('SHARED_DESKTOP', 'SHARED_WORKLOADS')
        # Image content hashes are generated from the archived source, never supplied by a job.
        identity = json.loads(Path('/opt/rom-trainer/recipe.json').read_text())
        for key in ('recipeDigest', 'modelDigest', 'dependencyDigest'):
            require(identity[key] == job['limits'][key], 'MANAGED_RECIPE_MISMATCH')
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
        env_cfg = load_env_cfg(task); agent_cfg = load_rl_cfg(task)
        env_cfg.scene.num_envs = job['numEnvs']; env_cfg.seed = job['seed']
        agent_cfg.seed = job['seed']; agent_cfg.max_iterations = job['iterations']; agent_cfg.logger = 'tensorboard'
        agent_cfg.resume = False; agent_cfg.save_interval = min(50, job['iterations'])
        work = directory / 'work'; work.mkdir(exist_ok=True)
        import threading
        stopped = threading.Event(); peak = [0]
        def sample_memory():
            while not stopped.wait(.25):
                free, total = torch.cuda.mem_get_info(0); peak[0] = max(peak[0], total - free)
        sampler = threading.Thread(target=sample_memory, daemon=True); sampler.start()
        env = ManagerBasedRlEnv(cfg=env_cfg, device='cuda:0')
        try:
            require(abs(env.step_dt - .02) < 1e-8, 'MANAGED_CONTROL_RATE_INVALID')
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(wrapped, asdict(agent_cfg), str(work), 'cuda:0')
            checkpoint = resume_checkpoint(job, directory)
            if checkpoint: runner.load(str(checkpoint), map_location='cuda:0')
            save = runner.save
            def managed_save(path, infos=None):
                require(shutil.disk_usage(directory).free >= 1 << 30, 'MANAGED_DISK_EXHAUSTED')
                save(path, infos); publish_checkpoint(Path(path), directory)
                # Bounded immutable checkpoint generations plus the current runner file.
                for previous in work.glob('model_*.pt'):
                    if previous != Path(path): previous.unlink()
            runner.save = managed_save
            runner.learn(num_learning_iterations=job['iterations'], init_at_random_ep_len=True)
            if not (directory / 'checkpoint.json').is_file(): managed_save(str(work / 'model_final.pt'))
            checkpoint_artifact = json.loads((directory / 'checkpoint.json').read_text())['artifacts'][0]
            atomic_json(directory / 'execution.json', {'executionState': 'SUCCEEDED'})
            runner.export_policy_to_onnx(str(directory), 'policy.tmp.onnx')
            exported = directory / 'policy.tmp.onnx'
            attach_metadata_to_onnx(str(exported), get_base_metadata(env, run_path=job['jobId']))
            attach_microduck_metadata(exported, task_id=task, source_commit=Path('/opt/rom-trainer/source-revision.txt').read_text().strip(), checkpoint=checkpoint_artifact['artifactId'], run_identity=job['jobId'])
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
            evidence.update(onnxFinite=True, observationDimension=61, actionDimension=14, controlHz=50,
                            peakGpuMemoryBytes=max(peak[0], torch.cuda.max_memory_allocated()), policySha256=policy['sha256'])
            return {'success': True, 'executionState': 'SUCCEEDED', 'exportState': 'SUCCEEDED', 'artifacts': [checkpoint_artifact, policy], 'evidence': evidence}
        finally:
            stopped.set(); sampler.join(timeout=1); env.close()


def publish_result_bundle(job, directory, result, publisher=None, source_commit=None):
    """Packaging failure does not rewrite an already proven training/export outcome."""
    if result.get('exportState') != 'SUCCEEDED':
        return
    try:
        if publisher is None:
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
    request_path = Path(request_path); directory = request_path.parent
    try:
        job = json.loads(request_path.read_text()); validate_request(job)
        result = (runtime or Runtime()).execute(job, directory)
        publish_result_bundle(job, directory, result)
    except Exception as error:
        checkpoint = directory / 'checkpoint.json'
        code = str(error) if isinstance(error, ValueError) and re.fullmatch(r'MANAGED_[A-Z_]{1,55}', str(error)) else 'MANAGED_TRAINING_FAILED'
        execution = directory / 'execution.json'
        trained = execution.is_file() and json.loads(execution.read_text()).get('executionState') == 'SUCCEEDED'
        if trained: code = 'MANAGED_EXPORT_FAILED'
        result = {'success': False, 'executionState': 'SUCCEEDED' if trained else 'FAILED',
                  'exportState': 'FAILED' if trained else 'PENDING', 'detailCode': code, 'artifacts': []}
        if checkpoint.is_file(): result['artifacts'] = json.loads(checkpoint.read_text()).get('artifacts', [])
    atomic_json(directory / 'result.json', result)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    sys.exit(run_managed(Path(sys.argv[1])))
