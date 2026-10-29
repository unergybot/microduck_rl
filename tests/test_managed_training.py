"""CPU fixtures for managed simulator admission and atomic artifact publication."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location('managed', Path(__file__).parents[1] / 'src/mjlab_microduck/managed_training.py')
managed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(managed)

def job():
    return {'jobId':'job-a','kind':'COMMISSION','scope':{'environment':'SIMULATOR','gpuUuid':'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee','driverVersion':'580.1'},
            'numEnvs':64,'iterations':5,'seed':7,'timeoutSeconds':600,'parentJobId':'','checkpointSha256':'',
            'limits':{'driverFloor':'580.1','computeCapabilities':['8.6'],'bootstrapGpuMemoryBytes':8<<30,'diskBytes':10<<30}}

def test_fixed_recipe_rejects_physical_cpu_and_custom_commands():
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
    with pytest.raises(ValueError): managed.resume_checkpoint(value,directory)
    parent=directory.parent/'job-a';parent.mkdir();(parent/'checkpoint.pt').write_bytes(b'changed')
    with pytest.raises(ValueError): managed.resume_checkpoint(value,directory)

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


@pytest.mark.parametrize('mode', ['', 'EXCLUSIVE', 'SHARED_DESKTOP'])
def test_commission_accepts_supported_sharing_modes_without_rewriting_scope(mode):
    value = job(); value['scope']['gpuSharingMode'] = mode
    original = json.loads(json.dumps(value))
    assert managed.validate_request(value) == managed.TASK
    assert value == original


@pytest.mark.parametrize('mode', ['shared_desktop', 'UNKNOWN', ' ', None, 1])
def test_unknown_sharing_mode_fails_closed(mode):
    value = job(); value['scope']['gpuSharingMode'] = mode
    with pytest.raises(ValueError, match='MANAGED_GPU_SHARING_MODE_INVALID'):
        managed.validate_request(value)


def test_shared_desktop_rejects_train_even_with_enabled_normal_envelope():
    value = job(); value['scope']['gpuSharingMode'] = 'SHARED_DESKTOP'
    value['kind'] = 'TRAIN'
    value['limits'].update(normalEnabled=True, maxEnvs=4096, maxIterations=20000)
    with pytest.raises(ValueError, match='MANAGED_SHARED_COMMISSION_REQUIRED'):
        managed.validate_request(value)


@pytest.mark.parametrize('mode', [None, '', 'EXCLUSIVE'])
def test_legacy_exclusive_normal_envelope_is_preserved(mode):
    value = job(); value['kind'] = 'TRAIN'
    if mode is not None: value['scope']['gpuSharingMode'] = mode
    value['limits'].update(normalEnabled=True, maxEnvs=4096, maxIterations=20000)
    assert managed.validate_request(value) == managed.TASK


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
