"""CPU fixtures for managed simulator admission and atomic artifact publication."""
import hashlib
import importlib.util
import json
from pathlib import Path
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
    assert (tmp_path/'checkpoint.pt').read_bytes()==source.read_bytes()
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
