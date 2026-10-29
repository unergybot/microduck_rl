"""The managed envelope is bounded and never trusts caller compatibility claims."""
import hashlib
import json
import zipfile

import pytest
from tests.test_rom_bundle import write_normalized_onnx, write_minimal_onnx
from mjlab_microduck.rom.managed_bundle import publish_bundle, verify_bundle


def pins():
    return dict(modelDigest='sha256:'+'a'*64, recipeDigest='sha256:'+'b'*64,
                configDigest='sha256:'+'c'*64, sourceCommit='d'*40,
                checkpointSha256='sha256:'+'e'*64)


def test_publisher_proves_normalizer_and_preserves_source(tmp_path):
    policy = write_normalized_onnx(tmp_path/'policy.onnx')
    output = tmp_path/'policy.zip'
    manifest = publish_bundle(policy, output, pins())
    assert manifest['kind'] == 'BUNDLE'
    assert manifest['sha256'] == 'sha256:'+hashlib.sha256(output.read_bytes()).hexdigest()
    verified = verify_bundle(output)
    assert verified['normalizerBaked'] is True
    assert verified['sourceCommit'] == pins()['sourceCommit']
    assert verified['checkpointSha256'] == pins()['checkpointSha256']
    assert verified['observationDimension'] == 61


def test_bare_actor_is_not_publishable(tmp_path):
    with pytest.raises(ValueError):
        publish_bundle(write_minimal_onnx(tmp_path/'bare.onnx'), tmp_path/'bad.zip', pins())
    assert not (tmp_path/'bad.zip').exists()


@pytest.mark.parametrize('member', ['../policy.onnx', '/policy.onnx', 'checkpoint.pkl'])
def test_archive_members_never_execute_or_extract(tmp_path, member):
    path = tmp_path/'attack.zip'
    with zipfile.ZipFile(path, 'w') as bundle:
        bundle.writestr(member, b'not executable')
        bundle.writestr('manifest.json', json.dumps({'schema':'ROM_MANAGED_POLICY_BUNDLE_V1'}))
    with pytest.raises(ValueError):
        verify_bundle(path)
