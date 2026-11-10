"""CPU fixtures for managed simulator admission and atomic artifact publication."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from mjlab_microduck.workbench import canonical_json, resolve_experiment, catalog
WALK = 'Mjlab-Velocity-Flat-MicroDuck'

def freeze_experiment(value):
    config = {k:value[k] for k in ('numEnvs','iterations','seed')}
    value['experimentJson'] = canonical_json(resolve_experiment(dict(schemaVersion='ROM_TRAINING_EXPERIMENT_V1',recipeId=WALK,robotProfileId='rom-xl330-v1',config=config)))
    return value

spec = importlib.util.spec_from_file_location('managed', Path(__file__).parents[1] / 'src/mjlab_microduck/managed_training.py')
managed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(managed)

def job():
    return freeze_experiment({'jobId':'job-a','kind':'COMMISSION','scope':{'environment':'SIMULATOR','gpuUuid':'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee','driverVersion':'580.1'},
            'numEnvs':64,'iterations':5,'seed':7,'timeoutSeconds':600,'parentJobId':'','checkpointSha256':'',
            'limits':{'driverFloor':'580.1','computeCapabilities':['8.6'],'bootstrapGpuMemoryBytes':8<<30,'diskBytes':10<<30}})

def test_recipe_runner_rejects_physical_cpu_and_custom_commands():
    for patch in ({'scope':{'environment':'REAL'}},{'command':'bash'},{'numEnvs':4096},{'kind':'SHELL'}):
        value=job();value.update(patch)
        with pytest.raises(ValueError):managed.validate_request(value)
    assert managed.validate_request(job()) == 'Mjlab-Velocity-Flat-MicroDuck'

def test_managed_checkpoint_publication_is_atomic_and_hash_bound(tmp_path):
    source=tmp_path/'model_1.pt';source.write_bytes(b'complete checkpoint')
    result=managed.publish_checkpoint(source,tmp_path)
    assert result['sha256']=='sha256:'+hashlib.sha256(source.read_bytes()).hexdigest()
    assert json.loads((tmp_path/'checkpoint.json').read_text())['artifacts']==[result]
    assert (tmp_path/result['artifactId']).read_bytes()==source.read_bytes()
    assert not list(tmp_path.glob('*.tmp'))

def test_resume_rejects_unmanaged_or_modified_checkpoint(tmp_path):
    directory=tmp_path/'jobs/job-b';directory.mkdir(parents=True)
    value=job();value.update(jobId='job-b',kind='TRAIN',parentJobId='job-a',checkpointSha256='sha256:'+'a'*64)
    with pytest.raises(ValueError): managed.select_resume_checkpoint(value,directory,json.loads(value['experimentJson']))
    parent=directory.parent/'job-a';parent.mkdir();(parent/'checkpoint.pt').write_bytes(b'changed')
    with pytest.raises(ValueError): managed.select_resume_checkpoint(value,directory,json.loads(value['experimentJson']))

def test_failure_writes_sanitized_result_without_ready(tmp_path):
    request=tmp_path/'request.json';request.write_text(json.dumps(job()))
    class Broken:
        def execute(self, value, directory):raise RuntimeError('private host path and credentials')
    assert managed.run_managed(request, Broken())==1
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['success'] is False
    assert 'private' not in (tmp_path/'result.json').read_text()


def test_interrupted_checkpoint_publish_preserves_previous_generation(tmp_path, monkeypatch):
    source=tmp_path/'model.pt'; source.write_bytes(b'previous')
    old=managed.publish_checkpoint(source,tmp_path)
    source.write_bytes(b'next generation')
    monkeypatch.setattr(managed, 'atomic_json', lambda *args: (_ for _ in ()).throw(OSError('interrupted')))
    with pytest.raises(OSError): managed.publish_checkpoint(source,tmp_path)
    manifest=json.loads((tmp_path/'checkpoint.json').read_text())
    assert manifest['artifacts']==[old]
    assert managed.artifact(tmp_path/old['artifactId'], 'CHECKPOINT')==old


def test_completed_training_survives_export_failure(tmp_path):
    request=tmp_path/'request.json';request.write_text(json.dumps(job()))
    class ExportFailure:
        def execute(self, value, directory):
            managed.atomic_json(directory/'execution.json', {'executionState':'SUCCEEDED'})
            raise RuntimeError('export failed with private details')
    assert managed.run_managed(request, ExportFailure())==1
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['executionState']=='SUCCEEDED'
    assert result['exportState']=='FAILED'
    assert result['detailCode']=='MANAGED_EXPORT_FAILED'


def test_bundle_publication_failure_preserves_successful_export(tmp_path):
    value=job();value['requestDigest']='sha256:'+'b'*64
    result={'success':True,'executionState':'SUCCEEDED','exportState':'SUCCEEDED','artifacts':[{'artifactId':'policy.onnx','kind':'ONNX'}]}
    value['limits'].update(modelDigest='sha256:'+'c'*64,recipeDigest='sha256:'+'d'*64)
    calls=[]
    def fail(*args):
        calls.append(args)
        raise OSError('private storage error')
    managed.publish_result_bundle(value,tmp_path,result,publisher=fail,source_commit='a'*40)
    assert result['executionState']=='SUCCEEDED'
    assert result['exportState']=='SUCCEEDED'
    assert len(calls)==1
    assert result['publicationState']=='FAILED'
    assert result['detailCode']=='MANAGED_BUNDLE_FAILED'
    assert result['artifacts'][0]['kind']=='ONNX'


@pytest.mark.parametrize('mode', ['', 'EXCLUSIVE', 'SHARED_DESKTOP', 'SHARED_WORKLOADS'])
def test_commission_accepts_supported_sharing_modes_without_rewriting_scope(mode):
    value = job(); value['scope']['gpuSharingMode'] = mode
    original = json.loads(json.dumps(value))
    assert managed.validate_request(value) == WALK
    assert value == original


@pytest.mark.parametrize('mode', ['shared_desktop', 'UNKNOWN', ' ', None, 1])
def test_unknown_sharing_mode_fails_closed(mode):
    value = job(); value['scope']['gpuSharingMode'] = mode
    with pytest.raises(ValueError, match='MANAGED_GPU_SHARING_MODE_INVALID'):
        managed.validate_request(value)


@pytest.mark.parametrize('mode', ['SHARED_DESKTOP', 'SHARED_WORKLOADS'])
def test_shared_training_uses_the_published_envelope(mode):
    value = job(); value['scope']['gpuSharingMode'] = mode
    value.update(kind='TRAIN', iterations=20)
    value['limits'].update(normalEnabled=True, maxEnvs=64, maxIterations=1000)
    freeze_experiment(value)
    original = json.loads(json.dumps(value))
    assert managed.validate_request(value) == WALK
    assert value == original
    for patch in ({'numEnvs':65}, {'iterations':1001}):
        outside = json.loads(json.dumps(value)); outside.update(patch)
        with pytest.raises(ValueError, match='MANAGED_ENVELOPE_EXCEEDED'):
            managed.validate_request(outside)


@pytest.mark.parametrize('mode', [None, '', 'EXCLUSIVE'])
def test_exclusive_normal_envelope_is_preserved(mode):
    value = job(); value['kind'] = 'TRAIN'
    if mode is not None: value['scope']['gpuSharingMode'] = mode
    value['limits'].update(normalEnabled=True, maxEnvs=4096, maxIterations=20000)
    assert managed.validate_request(value) == WALK


@pytest.mark.parametrize('patch', [{'numEnvs':63}, {'iterations':4}, {'timeoutSeconds':599}, {'parentJobId':'job-parent'}])
def test_shared_commission_requires_exact_existing_smoke_envelope(patch):
    value = job(); value['scope']['gpuSharingMode'] = 'SHARED_DESKTOP'; value.update(patch)
    with pytest.raises(ValueError, match='MANAGED_SMOKE_REQUIRED'):
        managed.validate_request(value)


@pytest.fixture
def preflight_host(monkeypatch):
    """Simulate only the image, host probes and GPU libraries; run real admission."""
    value = job()
    identity = {key: 'sha256:' + digit * 64 for key, digit in
                [('recipeDigest', 'a'), ('modelDigest', 'b'), ('dependencyDigest', 'c')]}
    value['limits'].update(identity)
    from importlib.resources import files
    digest = lambda v:'sha256:'+hashlib.sha256(canonical_json(v).encode()).hexdigest()
    data=catalog()
    identity['trainingExperimentsV1']=dict(catalogRevision=data['catalogRevision'],catalogSha256=digest(data),schemaSha256=digest(json.loads(files('mjlab_microduck.workbench').joinpath('resources/request.schema.json').read_text())),recipeDigests=[digest(r) for r in data['recipes']],robotProfileIds=[p['robotProfileId'] for p in data['robotProfiles']])
    host = {'gpu': value['scope']['gpuUuid'], 'driver': '580.1', 'free': '10240',
            'processes': '<nvidia_smi_log><gpu><processes/></gpu></nvidia_smi_log>'}
    original_read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **kw:
                        json.dumps(identity) if str(path) == '/opt/rom-trainer/recipe.json' else original_read(path, *a, **kw))
    def query(command, *, text, timeout):
        assert text is True and timeout == 5
        if command == ['nvidia-smi', '--query-gpu=uuid,driver_version,memory.free', '--format=csv,noheader,nounits']:
            return f"{host['gpu']}, {host['driver']}, {host['free']}\n"
        assert command == ['nvidia-smi', '-q', '-x', '-i', host['gpu']]
        return host['processes']
    monkeypatch.setattr(managed.subprocess, 'check_output', query)
    monkeypatch.setattr(managed.shutil, 'disk_usage', lambda path: SimpleNamespace(free=20 << 30))
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                           get_device_capability=lambda index: (8, 6), synchronize=lambda: None,
                           reset_peak_memory_stats=lambda: None)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda, float32='float32',
                        arange=lambda *a, **kw: 8, equal=lambda a, b: a == b))
    monkeypatch.setitem(sys.modules, 'warp', SimpleNamespace(init=lambda: None,
                        from_torch=lambda value: value, to_torch=lambda value: value))
    return value, host


@pytest.mark.parametrize('types', [('G',), ('C+G',), ('G', 'C+G'), ()])
def test_shared_preflight_accepts_only_graphics_or_idle_with_headroom(preflight_host, types):
    value, host = preflight_host; value['scope']['gpuSharingMode'] = 'SHARED_DESKTOP'
    host['processes'] = '<nvidia_smi_log><gpu><processes>' + ''.join(
        f'<process_info><type>{kind}</type><process_name>unavailable</process_name></process_info>'
        for kind in types) + '</processes></gpu></nvidia_smi_log>'
    evidence = managed.Runtime().preflight(value)
    assert evidence['cudaVerified'] is True
    assert evidence['gpuUuid'] == value['scope']['gpuUuid']
    assert evidence['computeCapability'] == '8.6'


@pytest.mark.parametrize('process', ['<type>C</type>', '<type>UNKNOWN</type>', '<type/>', '',
                                    '<type>G</type><type>C</type>'])
def test_shared_preflight_rejects_unknown_or_compute_processes(preflight_host, process):
    value, host = preflight_host; value['scope']['gpuSharingMode'] = 'SHARED_DESKTOP'
    host['processes'] = ('<nvidia_smi_log><gpu><processes><process_info><type>G</type></process_info>'
                         '<process_info>' + process + '</process_info></processes></gpu></nvidia_smi_log>')
    with pytest.raises(ValueError, match='MANAGED_GPU_IN_USE'):
        managed.Runtime().preflight(value)


@pytest.mark.parametrize('inventory', ['<nvidia_smi_log/>', '<nvidia_smi_log><gpu/></nvidia_smi_log>',
    '<nvidia_smi_log><gpu><processes>N/A</processes></gpu></nvidia_smi_log>',
    '<nvidia_smi_log><gpu><processes/><processes/></gpu></nvidia_smi_log>'])
def test_shared_preflight_rejects_unavailable_process_inventory(preflight_host, inventory):
    value, host = preflight_host; value['scope']['gpuSharingMode'] = 'SHARED_DESKTOP'
    host['processes'] = inventory
    with pytest.raises(ValueError, match='MANAGED_GPU_IN_USE'):
        managed.Runtime().preflight(value)


@pytest.mark.parametrize('mode', [None, '', 'EXCLUSIVE'])
def test_exclusive_preflight_still_rejects_graphics_processes(preflight_host, mode):
    value, host = preflight_host
    if mode is not None: value['scope']['gpuSharingMode'] = mode
    host['processes'] = '<gpu><processes><process_info><type>G</type></process_info></processes></gpu>'
    with pytest.raises(ValueError, match='MANAGED_GPU_IN_USE'):
        managed.Runtime().preflight(value)


@pytest.mark.parametrize('free', ['10239', '8192', '0', '', 'N/A', '-1'])
def test_shared_preflight_rejects_missing_or_insufficient_headroom(preflight_host, free):
    value, host = preflight_host; value['scope']['gpuSharingMode'] = 'SHARED_DESKTOP'; host['free'] = free
    with pytest.raises(ValueError, match='MANAGED_GPU_MEMORY_INSUFFICIENT'):
        managed.Runtime().preflight(value)


def test_exclusive_preflight_does_not_require_shared_reserve(preflight_host):
    value, host = preflight_host; host['free'] = '8192'
    assert managed.Runtime().preflight(value)['cudaVerified'] is True


@pytest.mark.parametrize('mutation,code', [('gpu', 'MANAGED_GPU_CHANGED'), ('driver', 'MANAGED_GPU_CHANGED'),
    ('recipeDigest', 'MANAGED_RECIPE_MISMATCH'), ('modelDigest', 'MANAGED_RECIPE_MISMATCH'),
    ('dependencyDigest', 'MANAGED_RECIPE_MISMATCH'), ('computeCapabilities', 'MANAGED_GPU_UNQUALIFIED')])
def test_shared_preflight_preserves_gpu_and_recipe_identity(preflight_host, mutation, code):
    value, host = preflight_host; value['scope']['gpuSharingMode'] = 'SHARED_DESKTOP'
    if mutation in host: host[mutation] = 'changed'
    else: value['limits'][mutation] = ['7.5'] if mutation == 'computeCapabilities' else 'changed'
    with pytest.raises(ValueError, match=code): managed.Runtime().preflight(value)


def test_pascal_uses_pinned_scalar_tile_backend_and_legacy_context_alias(monkeypatch):
    context = SimpleNamespace(runtime=SimpleNamespace(driver_version=(13, 0)))
    monkeypatch.setitem(sys.modules, 'warp._src.context', context)
    warp = SimpleNamespace(__version__='1.14.0', config=SimpleNamespace(
        enable_mathdx_gemm=True, enable_mathdx_solver=True, use_precompiled_headers=True))
    managed.configure_warp_runtime(warp, '6.1')
    assert warp.config.enable_mathdx_gemm is False
    assert warp.config.enable_mathdx_solver is False
    assert warp.config.use_precompiled_headers is False
    assert warp.context is context


@pytest.mark.parametrize('version', ['1.12.0', '1.13.0', '1.18.0', '1.14.0+unknown'])
def test_pascal_rejects_unqualified_warp_versions(version):
    warp = SimpleNamespace(__version__=version, config=SimpleNamespace(
        enable_mathdx_gemm=True, enable_mathdx_solver=True))
    with pytest.raises(ValueError, match='MANAGED_PASCAL_RUNTIME_UNSUPPORTED'):
        managed.configure_warp_runtime(warp, '6.1')
    assert warp.config.enable_mathdx_gemm is True


def test_other_gpu_runtime_configuration_is_preserved():
    warp = SimpleNamespace(config=SimpleNamespace(enable_mathdx_gemm=True))
    managed.configure_warp_runtime(warp, '8.6')
    assert warp.config.enable_mathdx_gemm is True
    assert not hasattr(warp, 'context')


def test_ampere_pinned_warp_exposes_context_and_bounds_compilation_without_scalar_fallback(monkeypatch):
    context = SimpleNamespace(runtime=SimpleNamespace(driver_version=(13, 0)))
    monkeypatch.setitem(sys.modules, 'warp._src.context', context)
    warp = SimpleNamespace(__version__='1.14.0', config=SimpleNamespace(
        enable_mathdx_gemm=True, enable_mathdx_solver=True, use_precompiled_headers=True))
    managed.configure_warp_runtime(warp, '8.6')
    assert warp.context.runtime.driver_version == (13, 0)
    assert warp.config.enable_mathdx_gemm is True
    assert warp.config.enable_mathdx_solver is True
    assert warp.config.use_precompiled_headers is False


def test_pinned_warp_preserves_existing_public_context():
    context = SimpleNamespace(runtime=SimpleNamespace(driver_version=(13, 0)))
    warp = SimpleNamespace(__version__='1.14.0', context=context,
                           config=SimpleNamespace(use_precompiled_headers=True))
    managed.configure_warp_runtime(warp, '8.6')
    assert warp.context is context


def test_pascal_preflight_configures_scalar_backend_before_warp_initialization(preflight_host, monkeypatch):
    value, _ = preflight_host
    value['limits']['computeCapabilities'] = ['6.1']
    sys.modules['torch'].cuda.get_device_capability = lambda index: (6, 1)
    warp = sys.modules['warp']
    warp.__version__ = '1.14.0'
    warp.config = SimpleNamespace(enable_mathdx_gemm=True, enable_mathdx_solver=True,
                                  use_precompiled_headers=True)
    context = SimpleNamespace(runtime=SimpleNamespace(driver_version=(13, 0)))
    monkeypatch.setitem(sys.modules, 'warp._src.context', context)
    def initialize():
        assert warp.config.enable_mathdx_gemm is False
        assert warp.config.enable_mathdx_solver is False
        assert warp.config.use_precompiled_headers is False
        assert warp.context is context
    warp.init = initialize
    assert managed.Runtime().preflight(value)['computeCapability'] == '6.1'

@pytest.mark.parametrize('types', [('C',), ('G','C'), ('G','C+G'), ()])
def test_workload_preflight_allows_known_compute_without_changing_identity(preflight_host, types):
    value, host=preflight_host;value['scope']['gpuSharingMode']='SHARED_WORKLOADS'
    host['processes']='<nvidia_smi_log><gpu><processes>'+''.join('<process_info><type>'+kind+'</type></process_info>' for kind in types)+'</processes></gpu></nvidia_smi_log>'
    assert managed.Runtime().preflight(value)['gpuUuid']==value['scope']['gpuUuid']

@pytest.mark.parametrize('inventory', ['<nvidia_smi_log/>','<nvidia_smi_log><gpu><processes>N/A</processes></gpu></nvidia_smi_log>', '<nvidia_smi_log><gpu><processes><process_info><type>UNKNOWN</type></process_info></processes></gpu></nvidia_smi_log>'])
def test_workload_preflight_still_rejects_unknown_process_inventory(preflight_host, inventory):
    value,host=preflight_host;value['scope']['gpuSharingMode']='SHARED_WORKLOADS';host['processes']=inventory
    with pytest.raises(ValueError,match='MANAGED_GPU_IN_USE'):managed.Runtime().preflight(value)

@pytest.mark.parametrize('free', ['10239','N/A','0'])
def test_workload_preflight_retains_shared_memory_reserve(preflight_host, free):
    value,host=preflight_host;value['scope']['gpuSharingMode']='SHARED_WORKLOADS';host['free']=free
    with pytest.raises(ValueError,match='MANAGED_GPU_MEMORY_INSUFFICIENT'):managed.Runtime().preflight(value)

@pytest.mark.parametrize('mode', ['SHARED_DESKTOP', 'SHARED_WORKLOADS'])
def test_shared_training_requires_an_enabled_envelope(mode):
    value=job();value['scope']['gpuSharingMode']=mode;value['kind']='TRAIN'
    value['limits'].update(normalEnabled=False,maxEnvs=64,maxIterations=1000)
    with pytest.raises(ValueError,match='MANAGED_ENVELOPE_EXCEEDED'):managed.validate_request(value)


@pytest.mark.parametrize('diagnostics', ['missing_digest', 'import_failure', 'setup_failure'])
@pytest.mark.parametrize('learn_fails', [False, True])
def test_optional_observer_setup_preserves_learning_and_authoritative_failure(tmp_path, monkeypatch, diagnostics, learn_fails):
    import builtins
    from dataclasses import dataclass
    from types import ModuleType
    events = []
    value = job()
    value['requestDigest'] = 'sha256:' + 'a' * 64
    @dataclass
    class AgentConfig:
        clip_actions: float = 1.
        num_steps_per_env: int = 24
    class Runner:
        logger = SimpleNamespace(init_logging_writer=lambda: None, log=lambda: None)
        current_learning_iteration = 0
        def save(self, path, infos=None): Path(path).write_bytes(b'checkpoint')
        def learn(self, **kwargs):
            events.append('learn')
            if learn_fails: raise RuntimeError('authoritative learn failure')
        def export_policy_to_onnx(self, *args):
            events.append('export')
            raise RuntimeError('export reached')
    runner = Runner()
    cfg=SimpleNamespace(scene=SimpleNamespace(),curriculum={})
    env = SimpleNamespace(step_dt=.02, close=lambda: events.append('close'))
    modules = {
        'mjlab': {}, 'mjlab.tasks': {},
        'mjlab.envs': {'ManagerBasedRlEnv': lambda **kw: env},
        'mjlab.rl': {'MjlabOnPolicyRunner': lambda *a: runner, 'RslRlVecEnvWrapper': lambda *a, **kw: env},
        'mjlab.tasks.registry': {'load_env_cfg': lambda task: SimpleNamespace(scene=SimpleNamespace()),
                                'load_rl_cfg': lambda task: AgentConfig(), 'load_runner_cls': lambda task: None},
        'mjlab.utils.torch': {'configure_torch_backends': lambda: None},
        'mjlab.rl.exporter_utils': {'attach_metadata_to_onnx': lambda *a: None, 'get_base_metadata': lambda *a: {}},
        'mjlab_microduck.rom.export_metadata': {'attach_microduck_metadata': lambda *a: None},
        'torch': {'cuda': SimpleNamespace(mem_get_info=lambda _: (1, 1))},
        'mjlab_microduck.workbench.engine': {'build_configs':lambda _: (cfg,AgentConfig())},
        'mjlab_microduck.workbench.checkpoints': {'load_teachers':lambda *a:{}},
    }
    for name, attributes in modules.items():
        module = ModuleType(name); module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    observer_spec = importlib.util.spec_from_file_location('mjlab_microduck.training_console',
        Path(__file__).parents[1] / 'src/mjlab_microduck/training_console.py')
    observer = importlib.util.module_from_spec(observer_spec); observer_spec.loader.exec_module(observer)
    if diagnostics == 'missing_digest':
        original_observe=observer.observe_runner
        observer.observe_runner=lambda runner,directory,request,**kw:original_observe(runner,directory,{k:v for k,v in request.items() if k!='requestDigest'},**kw)
    if diagnostics == 'setup_failure':
        observer.observe_runner = lambda *args: (_ for _ in ()).throw(OSError('optional setup failed'))
    monkeypatch.setitem(sys.modules, 'mjlab_microduck.training_console', observer)
    original_import = builtins.__import__
    def importing(name, *args, **kwargs):
        if diagnostics == 'import_failure' and name == 'mjlab_microduck.training_console':
            raise ImportError('optional observer unavailable')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', importing)
    monkeypatch.setattr(managed.Runtime, 'preflight', lambda *a: {})
    monkeypatch.setattr(managed.shutil, 'disk_usage', lambda _: SimpleNamespace(free=20 << 30))
    request = tmp_path / 'request.json'; request.write_text(json.dumps(value))
    assert managed.run_managed(request) == 1
    assert events == (['learn', 'close'] if learn_fails else ['learn', 'export', 'close'])
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['detailCode'] == ('MANAGED_TRAINING_FAILED' if learn_fails else 'MANAGED_EXPORT_FAILED')
    assert result['executionState'] == ('FAILED' if learn_fails else 'SUCCEEDED')


def measured_host(tmp_path, monkeypatch, *, fault=None, saves=0):
    """Local measured-runtime fixture. No shared request/legacy fixture is changed."""
    import threading
    from dataclasses import dataclass
    from types import ModuleType
    import numpy as np
    console_spec = importlib.util.spec_from_file_location('mjlab_microduck.training_console',
        Path(__file__).parents[1] / 'src/mjlab_microduck/training_console.py')
    console = importlib.util.module_from_spec(console_spec); console_spec.loader.exec_module(console)
    monkeypatch.setitem(sys.modules, 'mjlab_microduck.training_console', console)
    value = job(); value['requestDigest'] = 'sha256:' + 'a' * 64
    events = []; captured = {}; queried = threading.Event()
    def fail(name):
        events.append(name)
        if fault == name: raise RuntimeError('original ' + name)
    @dataclass
    class Config:
        clip_actions: float = 1.
        num_steps_per_env: int = 7
    cfg = SimpleNamespace(scene=SimpleNamespace(), curriculum={})
    def device_query(index):
        assert index == 0
        queried.set()
        return 200, 1000
    cuda = SimpleNamespace(mem_get_info=device_query, max_memory_allocated=lambda *a: 300,
                           max_memory_reserved=lambda *a: 400)
    env = SimpleNamespace(step_dt=.02, num_envs=64, close=lambda: fail('cleanup'))
    def environment(**kwargs):
        assert queried.wait(1)
        fail('environment'); return env
    class Runner:
        current_learning_iteration = 0
        logger = SimpleNamespace(writer=None, init_logging_writer=lambda: None, log=lambda: None)
        def save(self, path, infos=None):
            fail('checkpoint'); Path(path).write_bytes(b'checkpoint')
        def learn(self, **kwargs):
            fail('training')
            for index in range(saves): self.save(str(tmp_path / 'work' / f'model_{index}.pt'))
        def export_policy_to_onnx(self, directory, name):
            fail('onnx'); (Path(directory) / name).write_bytes(b'fake ONNX test bytes')
    runner = Runner()
    def construct(*args):
        fail('runner'); return runner
    def configure(): fail('config')
    modules = {
        'mjlab': {}, 'mjlab.tasks': {},
        'mjlab.envs': {'ManagerBasedRlEnv': environment},
        'mjlab.rl': {'MjlabOnPolicyRunner': construct, 'RslRlVecEnvWrapper': lambda *a, **kw: env},
        'mjlab.tasks.registry': {'load_env_cfg': lambda _: cfg, 'load_rl_cfg': lambda _: Config(),
                                'load_runner_cls': lambda _: None},
        'mjlab.utils.torch': {'configure_torch_backends': configure},
        'mjlab.rl.exporter_utils': {'attach_metadata_to_onnx': lambda *a: fail('metadata'),
                                  'get_base_metadata': lambda *a, **kw: {}},
        'mjlab_microduck.rom.export_metadata': {'attach_microduck_metadata': lambda *a, **kw: None},
        'torch': {'cuda': cuda},
        'mjlab_microduck.workbench.engine': {'build_configs':lambda _: (cfg,Config())},
        'mjlab_microduck.workbench.checkpoints': {'load_teachers':lambda *a:{}},
        'onnx': {'load':lambda _:SimpleNamespace(metadata_props=SimpleNamespace(add=lambda:SimpleNamespace())), 'save':lambda *a:None},
        'onnxruntime': {'InferenceSession': lambda *a, **kw: SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(shape=[1, 61], name='obs')],
            run=lambda *a: [np.zeros((1, 14))])},
    }
    for name, attributes in modules.items():
        module = ModuleType(name); module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    if fault == 'teachers':
        spec = {'taskId': WALK, 'configDigest': 'sha256:' + 'b' * 64}
        monkeypatch.setattr(managed, 'experiment', lambda _: spec)
        for name, attributes in {
            'mjlab_microduck.workbench.engine': {'build_configs': lambda _: (cfg, Config())},
            'mjlab_microduck.workbench.checkpoints': {'load_teachers': lambda *a: fail('teachers')},
        }.items():
            module = ModuleType(name); module.__dict__.update(attributes)
            monkeypatch.setitem(sys.modules, name, module)
    runtime_class = managed.Runtime
    class MeasuredRuntime(runtime_class):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured.update(recorder=self.measurements, collector=self.collector)
        def preflight(self, accepted):
            fail('preflight'); return {}
    monkeypatch.setattr(managed, 'Runtime', MeasuredRuntime)
    monkeypatch.setattr(managed.shutil, 'disk_usage', lambda _: SimpleNamespace(free=20 << 30))
    read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **kw:
                        'a' * 40 if str(path) == '/opt/rom-trainer/source-revision.txt' else read(path, *a, **kw))
    def package(accepted, directory, result):
        events.append('package')
        result['publicationState'] = 'FAILED' if fault == 'package' else 'LOCAL_READY'
        if fault == 'package': result['detailCode'] = 'MANAGED_BUNDLE_FAILED'
    monkeypatch.setattr(managed, 'publish_result_bundle', package)
    request = tmp_path / 'request.json'; request.write_text(json.dumps(value))
    return request, captured, events


@pytest.mark.parametrize('fault', [None, 'preflight', 'config', 'teachers', 'environment', 'runner',
                                  'training', 'checkpoint', 'onnx', 'metadata', 'cleanup', 'package'])
def test_measurements_preserve_phase_failure_and_authoritative_results(tmp_path, monkeypatch, fault):
    request, captured, events = measured_host(tmp_path, monkeypatch, fault=fault)
    code = managed.run_managed(request)
    collector = captured['collector']; collector._thread.join(timeout=3)
    assert not collector._thread.is_alive()
    recorder = captured['recorder']; points = recorder.snapshot()[1]
    result = json.loads((tmp_path / 'result.json').read_text())
    if fault in (None, 'package'):
        assert code == 0 and result['success'] is True
        assert result['evidence']['peakGpuMemoryBytes'] == 800
        assert points['Measurement/terminalState'] == 1
        assert points['MeasurePhase/package/state'] == (3 if fault == 'package' else 2)
        assert points['MeasurePhase/onnx/state'] == 2
    else:
        assert code == 1 and result['success'] is False
        assert points['Measurement/terminalState'] == 2
        assert points['MeasurePhase/package/state'] == 4
    init_fault = fault in ('preflight', 'config', 'teachers', 'environment', 'runner')
    assert points['MeasurePhase/initialization/state'] == (3 if init_fault else 2)
    assert points['MeasurePhase/cleanup/state'] == (3 if fault == 'cleanup' else 2)
    if fault not in ('preflight', 'config'):
        assert points['MeasureMemory/deviceStatus'] == 2
        assert points['MeasureMemory/deviceSampleCount'] >= 1
        assert points['MeasureMemory/deviceUsedSampledMaxBytes'] == 800
        assert points['MeasureMemory/torchAllocatedPeakBytes'] == 300
        assert points['MeasureMemory/torchReservedPeakBytes'] == 400
    if fault in ('onnx', 'metadata'):
        assert points['MeasurePhase/onnx/state'] == 3
        assert result['detailCode'] == 'MANAGED_EXPORT_FAILED'
    if fault == 'training': assert points['MeasurePhase/training/state'] == 3
    if fault == 'checkpoint':
        assert points['MeasureCheckpoint/calls'] == 1
        assert points['MeasurePhase/checkpoint/state'] == 3
    if init_fault: assert 'training' not in events
    assert 'MeasureTraining/completedTransitions' not in points  # local fake logger has no proof


@pytest.mark.parametrize('saves,expected', [(0, 1), (2, 2)])
def test_runtime_counts_periodic_and_fallback_checkpoint_wrappers(tmp_path, monkeypatch, saves, expected):
    request, captured, _ = measured_host(tmp_path, monkeypatch, saves=saves)
    assert managed.run_managed(request) == 0
    collector = captured['collector']; collector._thread.join(timeout=3)
    assert not collector._thread.is_alive()
    points = captured['recorder'].snapshot()[1]
    assert points['MeasureCheckpoint/calls'] == expected
    assert points['MeasurePhase/checkpoint/state'] == 2
    assert points['MeasurePhase/checkpoint/seconds'] >= 0


def test_earlier_training_exception_survives_cleanup_failure(tmp_path, monkeypatch):
    request, captured, _ = measured_host(tmp_path, monkeypatch, fault='training')
    original = managed.Runtime
    class CleanupFailure(original):
        def execute(self, accepted, directory):
            import mjlab.envs
            create = sys.modules['mjlab.envs'].ManagerBasedRlEnv
            def env(**kwargs):
                instance = create(**kwargs)
                instance.close = lambda: (_ for _ in ()).throw(ValueError('MANAGED_CLEANUP_TEST'))
                return instance
            monkeypatch.setattr(sys.modules['mjlab.envs'], 'ManagerBasedRlEnv', env)
            return super().execute(accepted, directory)
    monkeypatch.setattr(managed, 'Runtime', CleanupFailure)
    assert managed.run_managed(request) == 1
    captured['collector']._thread.join(timeout=3)
    assert json.loads((tmp_path / 'result.json').read_text())['detailCode'] == 'MANAGED_TRAINING_FAILED'
    assert captured['recorder'].snapshot()[1]['MeasurePhase/cleanup/state'] == 3


@pytest.mark.parametrize('outcome', ['empty', 'one', 'invalid', 'failure'])
def test_memory_sampler_truthful_windows_and_failed_query_stops(outcome):
    import threading
    console_spec = importlib.util.spec_from_file_location('memory_console',
        Path(__file__).parents[1] / 'src/mjlab_microduck/training_console.py')
    console = importlib.util.module_from_spec(console_spec); console_spec.loader.exec_module(console)
    recorder = console.RunMeasurements({'numEnvs': 64}, clock=lambda: 10., started=2.)
    done = threading.Event(); calls = []
    def query(index):
        calls.append(index); done.set()
        if outcome == 'failure': raise OSError('query failed')
        return (-1, 1000) if outcome == 'invalid' else (200, 1000)
    sampler = managed.MemorySampler(query, recorder)
    if outcome == 'empty': sampler.stopped.set()
    sampler.start()
    if outcome != 'empty': assert done.wait(1)
    sampler.close()
    assert not sampler.thread.is_alive()
    points = recorder.snapshot()[1]
    assert points['MeasureMemory/deviceSampleCount'] == (1 if outcome == 'one' else 0)
    assert points['MeasureMemory/deviceSampleFailures'] == (1 if outcome in ('invalid', 'failure') else 0)
    if outcome == 'one':
        assert points['MeasureMemory/deviceStatus'] == 2
        assert points['MeasureMemory/deviceUsedSampledMaxBytes'] == 800
        assert points['MeasureMemory/deviceFirstSampleSeconds'] == 8
        assert points['MeasureMemory/deviceLastSampleSeconds'] == 8
    else:
        assert 'MeasureMemory/deviceUsedSampledMaxBytes' not in points
        assert 'MeasureMemory/deviceFirstSampleSeconds' not in points
        assert points['MeasureMemory/deviceStatus'] == (0 if outcome == 'empty' else 3)
    assert len(calls) <= 1


def test_memory_sampler_bounded_join_discloses_late_unfinished_query():
    import threading
    entered, release = threading.Event(), threading.Event()
    console_spec = importlib.util.spec_from_file_location('memory_console',
        Path(__file__).parents[1] / 'src/mjlab_microduck/training_console.py')
    console = importlib.util.module_from_spec(console_spec); console_spec.loader.exec_module(console)
    recorder = console.RunMeasurements({'numEnvs': 64}, clock=lambda: 10., started=2.)
    def query(index):
        entered.set(); assert release.wait(3); return 200, 1000
    sampler = managed.MemorySampler(query, recorder); sampler.start()
    try:
        assert entered.wait(1)
        sampler.close()
        points = recorder.snapshot()[1]
        assert points['MeasureMemory/deviceStatus'] == 5
        assert points['MeasureMemory/deviceSampleCount'] == 0
        assert 'MeasureMemory/deviceUsedSampledMaxBytes' not in points
    finally:
        release.set(); sampler.thread.join(timeout=2)
    assert not sampler.thread.is_alive()
    points = recorder.snapshot()[1]
    assert points['MeasureMemory/deviceStatus'] == 5
    assert points['MeasureMemory/deviceSampleCount'] == 1


def test_memory_sampler_multiple_queries_preserve_actual_window_and_cadence():
    console_spec = importlib.util.spec_from_file_location('memory_console',
        Path(__file__).parents[1] / 'src/mjlab_microduck/training_console.py')
    console = importlib.util.module_from_spec(console_spec); console_spec.loader.exec_module(console)
    clock = [3.]
    recorder = console.RunMeasurements({'numEnvs': 64}, clock=lambda: clock[0], started=1.)
    queries, waits = [], []
    def query(index):
        queries.append(index); clock[0] += 2
        return (200, 1000) if len(queries) == 1 else (100, 1000)
    class Stop:
        def is_set(self): return False
        def wait(self, interval):
            waits.append(interval); return len(waits) == 2
    sampler = managed.MemorySampler(query, recorder); sampler.stopped = Stop()
    sampler._run()
    points = recorder.snapshot()[1]
    assert waits == [.25, .25] and queries == [0, 0]
    assert points['MeasureMemory/deviceSampleCount'] == 2
    assert points['MeasureMemory/deviceUsedSampledMaxBytes'] == 900
    assert points['MeasureMemory/deviceFirstSampleSeconds'] == 4
    assert points['MeasureMemory/deviceLastSampleSeconds'] == 6
    assert points['MeasureMemory/deviceStatus'] == 2


def test_initialization_includes_request_read_and_constructor_does_not_reset_entry(tmp_path, monkeypatch):
    request, captured, _ = measured_host(tmp_path, monkeypatch, fault='preflight')
    clock = [1.]
    monkeypatch.setattr(managed.time, 'monotonic', lambda: clock[0])
    module = sys.modules['mjlab_microduck.training_console']
    original = module.RunMeasurements
    monkeypatch.setattr(module, 'RunMeasurements', lambda accepted, *, started:
                        original(accepted, started=started, clock=lambda: clock[0]))
    read = Path.read_text
    def reading(path, *args, **kwargs):
        if path == request: clock[0] = 5.
        return read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', reading)
    assert managed.run_managed(request) == 1
    captured['collector']._thread.join(timeout=3)
    points = captured['recorder'].snapshot()[1]
    assert points['MeasurePhase/initialization/seconds'] == 4
    assert points['MeasurePhase/initialization/state'] == 3


def test_measurement_setup_failure_and_old_injected_runtime_remain_optional(tmp_path, monkeypatch):
    request, captured, events = measured_host(tmp_path, monkeypatch)
    module = sys.modules['mjlab_microduck.training_console']
    monkeypatch.setattr(module, 'RunMeasurements', lambda *a, **kw: (_ for _ in ()).throw(OSError('optional recorder')))
    assert managed.run_managed(request) == 0
    assert 'training' in events and captured['recorder'] is None
    class OldRuntime:
        def execute(self, accepted, directory):
            return {'success': True, 'exportState': 'PENDING', 'artifacts': []}
    assert managed.run_managed(request, OldRuntime()) == 0


def test_optional_sampler_cleanup_error_does_not_rewrite_success(tmp_path, monkeypatch):
    request, captured, _ = measured_host(tmp_path, monkeypatch)
    close = managed.MemorySampler.close
    def failed_close(sampler):
        close(sampler)
        raise RuntimeError('optional sampler bookkeeping')
    monkeypatch.setattr(managed.MemorySampler, 'close', failed_close)
    assert managed.run_managed(request) == 0
    captured['collector']._thread.join(timeout=3)
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['success'] is True and result['exportState'] == 'SUCCEEDED'
    assert captured['recorder'].snapshot()[1]['MeasurePhase/cleanup/state'] == 3



def test_initial_console_sink_failure_preserves_authoritative_result_and_sampler_cleanup(tmp_path, monkeypatch):
    request, captured, _ = measured_host(tmp_path, monkeypatch)
    (tmp_path / 'console').write_text('unavailable optional report directory')
    assert managed.run_managed(request) == 0
    collector = captured['collector']; collector._thread.join(timeout=3)
    assert not collector._thread.is_alive()
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['success'] is True and result['exportState'] == 'SUCCEEDED'
    assert result['publicationState'] == 'LOCAL_READY'
    assert collector.state == 2
    points = captured['recorder'].snapshot(console_state=collector.state)[1]
    assert points['Measurement/terminalState'] == 1
    assert points['Measurement/consoleState'] == 2
    assert points['MeasureMemory/deviceStatus'] == 2
    assert points['MeasurePhase/cleanup/state'] == 2



def test_failed_entry_clock_keeps_accepted_execution_with_unknown_start_measurements(tmp_path, monkeypatch):
    request, captured, events = measured_host(tmp_path, monkeypatch)
    monkeypatch.setattr(managed, 'time', SimpleNamespace(
        monotonic=lambda: (_ for _ in ()).throw(OSError('entry clock unavailable'))))
    assert managed.run_managed(request) == 0
    collector = captured['collector']; collector._thread.join(timeout=3)
    assert not collector._thread.is_alive()
    assert 'training' in events and 'package' in events
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['success'] is True and result['publicationState'] == 'LOCAL_READY'
    assert result['evidence']['peakGpuMemoryBytes'] == 800
    points = captured['recorder'].snapshot()[1]
    assert points['MeasurePhase/initialization/state'] == 5
    assert 'MeasurePhase/initialization/seconds' not in points
    assert 'Measurement/snapshotElapsedSeconds' not in points
    assert points['MeasureMemory/deviceStatus'] == 2
    assert points['MeasureMemory/deviceSampleCount'] >= 1
    assert 'MeasureMemory/deviceFirstSampleSeconds' not in points
    assert points['Measurement/terminalState'] == 1


def test_disabled_observer_does_not_block_recipe_training(tmp_path, monkeypatch):
    """A missing optional read capability never blocks recipe execution."""
    request, captured, _ = measured_host(tmp_path, monkeypatch)
    monkeypatch.setattr(managed.Runtime,'_observer_version',0,raising=False)
    assert managed.run_managed(request) == 0
    values = captured['recorder'].snapshot()[1]
    assert values['MeasureObserver/state'] == 0
    assert 'MeasureObserver/hookCalls' not in values
    assert 'MeasureObserver/hookSeconds' not in values


@pytest.mark.parametrize('fault', ['attach', 'close'])
def test_runtime_observer_diagnostics_preserve_training_failure_and_known_totals(tmp_path, monkeypatch, fault):
    from dataclasses import dataclass
    from types import ModuleType

    request, captured, events = measured_host(tmp_path, monkeypatch, fault='training')
    spec = {'taskId': WALK, 'configDigest': 'sha256:' + 'b' * 64}
    monkeypatch.setattr(managed, 'experiment', lambda _: spec)
    @dataclass
    class Config:
        clip_actions: float = 1.
        num_steps_per_env: int = 7
    cfg = SimpleNamespace(scene=SimpleNamespace(), curriculum={})
    class Observer:
        def _measurement_snapshot(self):
            return {'MeasureObserver/state': 2, 'MeasureObserver/hookCalls': 2,
                    'MeasureObserver/hookSeconds': 4.}
        def close(self):
            events.append('observer-close')
            raise OSError('optional close')
    def attach(*args, version=1, measurements=None):
        assert measurements is captured['recorder']
        if fault == 'attach': raise OSError('optional setup')
        source = Observer()
        measurements._observer_datum(1, observer=source)
        return source
    for name, attributes in {
        'mjlab_microduck.workbench.engine': {'build_configs': lambda _: (cfg, Config())},
        'mjlab_microduck.workbench.checkpoints': {'load_teachers': lambda *a: {}},
        'mjlab_microduck.workbench.observer': {'attach_observer': attach},
    }.items():
        module = ModuleType(name); module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    assert managed.run_managed(request) == 1
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['detailCode'] == 'MANAGED_TRAINING_FAILED'
    assert result['executionState'] == 'FAILED'
    assert 'training' in events and 'cleanup' in events
    values = captured['recorder'].snapshot()[1]
    assert values['MeasureObserver/state'] == 3
    if fault == 'attach':
        assert 'MeasureObserver/hookCalls' not in values
        assert 'MeasureObserver/hookSeconds' not in values
    else:
        assert values['MeasureObserver/hookCalls'] == 2
        assert values['MeasureObserver/hookSeconds'] == 4


@pytest.mark.parametrize('parent', ['', 'retained-train'])
@pytest.mark.parametrize('bounds', [(1, 1, 1), (64, 5, 600), (64, 10, 600)])
def test_bounded_shared_train_and_resume_preserve_request(parent, bounds):
    value = job()
    value['scope']['gpuSharingMode'] = 'SHARED_WORKLOADS'
    value.update(kind='TRAIN', parentJobId=parent, numEnvs=bounds[0],
                 iterations=bounds[1], timeoutSeconds=bounds[2])
    value['limits'].update(normalEnabled=True, maxEnvs=64, maxIterations=10)
    freeze_experiment(value)
    original = json.loads(json.dumps(value))
    assert managed.validate_request(value) == WALK
    assert value == original


@pytest.mark.parametrize('patch', [{'numEnvs': 65}, {'iterations': 11}])
def test_bounded_shared_train_rejects_profile_overflow(patch):
    value = job()
    value['scope']['gpuSharingMode'] = 'SHARED_WORKLOADS'
    value.update(kind='TRAIN', **patch)
    value['limits'].update(normalEnabled=True, maxEnvs=64, maxIterations=10)
    with pytest.raises(ValueError, match='MANAGED_ENVELOPE_EXCEEDED'):
        managed.validate_request(value)


@pytest.mark.parametrize('enabled', [False, 1])
def test_shared_training_still_requires_explicit_normal_enablement(enabled):
    value = job()
    value['scope']['gpuSharingMode'] = 'SHARED_WORKLOADS'
    value['kind'] = 'TRAIN'
    value['limits'].update(normalEnabled=enabled, maxEnvs=64, maxIterations=10)
    with pytest.raises(ValueError, match='MANAGED_ENVELOPE_EXCEEDED'):
        managed.validate_request(value)


def test_shared_timeout_is_the_supplied_execution_budget_not_a_literal_profile():
    value = job()
    value['scope']['gpuSharingMode'] = 'SHARED_WORKLOADS'
    value.update(kind='TRAIN', timeoutSeconds=601)
    value['limits'].update(normalEnabled=True, maxEnvs=64, maxIterations=10)
    freeze_experiment(value)
    original = json.loads(json.dumps(value))
    assert managed.validate_request(value) == WALK
    assert value == original


def test_managed_training_requires_resolved_recipe():
    value=job();value.pop('experimentJson')
    with pytest.raises(ValueError, match='MANAGED_EXPERIMENT_REQUIRED'):
        managed.validate_request(value)


def test_managed_training_accepts_one_day_with_published_budget():
    value=job();value.update(kind='TRAIN',iterations=2000,timeoutSeconds=86400)
    value['limits'].update(normalEnabled=True,maxEnvs=64,maxIterations=10000)
    freeze_experiment(value)
    assert managed.validate_request(value)==WALK
    value['timeoutSeconds']=86401
    with pytest.raises(ValueError,match='MANAGED_PARAMETERS_INVALID'):managed.validate_request(value)
