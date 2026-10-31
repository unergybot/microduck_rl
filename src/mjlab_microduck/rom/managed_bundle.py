"""Bounded managed transport envelope; legacy release bundles keep their own schema."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import zipfile

from .contracts import CONTROLLED_SERVO_JOINTS

SCHEMA = 'ROM_MANAGED_POLICY_BUNDLE_V1'
MAX_POLICY = 256 << 20
MAX_BUNDLE = 2 << 30
DIGEST = r'sha256:[a-f0-9]{64}'


def require(condition, code):
    if not condition:
        raise ValueError(code)


def sha(data):
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'BUNDLE_DUPLICATE_KEY')
        result[key] = value
    return result


def _prove(data):
    import onnx
    import numpy as np
    import onnxruntime as ort
    from .onnx_policy import inspect_normalized_actor
    require(0 < len(data) <= MAX_POLICY, 'POLICY_SIZE_INVALID')
    model = onnx.load_model_from_string(data)
    # External initializers/custom domains are not part of the managed actor contract.
    require(not model.functions and all(op.domain in ('', 'ai.onnx') for op in model.opset_import), 'POLICY_GRAPH_INVALID')
    def local(graph):
        for tensor in graph.initializer:
            require(not tensor.external_data and tensor.data_location != onnx.TensorProto.EXTERNAL, 'POLICY_EXTERNAL_DATA_REJECTED')
        for node in graph.node:
            require(node.domain in ('', 'ai.onnx'), 'POLICY_GRAPH_INVALID')
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH: local(attribute.g)
                for nested in attribute.graphs: local(nested)
                tensors = list(attribute.tensors) + ([attribute.t] if attribute.HasField('t') else [])
                for tensor in tensors:
                    require(not tensor.external_data and tensor.data_location != onnx.TensorProto.EXTERNAL, 'POLICY_EXTERNAL_DATA_REJECTED')
    local(model.graph)
    proof = inspect_normalized_actor(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(data, sess_options=options, providers=['CPUExecutionProvider'])
    require(len(session.get_inputs()) == 1 and session.get_inputs()[0].shape == [1, 61], 'POLICY_OBSERVATION_INVALID')
    for value in (0., .1, -.1):
        result = session.run(None, {session.get_inputs()[0].name: np.full((1, 61), value, dtype=np.float32)})
        require(len(result) == 1 and result[0].shape == (1, 14) and np.isfinite(result[0]).all(), 'POLICY_ACTION_INVALID')
    return proof.fingerprint


def metadata_valid(metadata):
    required = {'policySha256', 'modelDigest', 'recipeDigest', 'configDigest', 'normalizerDigest',
                'sourceCommit', 'taskId', 'observationDimension', 'actionDimension', 'controlHz',
                'observationLayout', 'normalizerBaked', 'jointNames', 'provenance'}
    require(isinstance(metadata, dict) and required <= metadata.keys()
            and metadata.keys() <= required | {'checkpointSha256', 'subjectDigest'}, 'POLICY_METADATA_INVALID')
    for key in ('policySha256', 'modelDigest', 'recipeDigest', 'configDigest', 'normalizerDigest', 'checkpointSha256', 'subjectDigest'):
        require(key not in metadata or re.fullmatch(DIGEST, metadata[key]) is not None, 'POLICY_METADATA_INVALID')
    require(re.fullmatch(r'[a-f0-9]{40}', metadata['sourceCommit']) is not None, 'POLICY_SOURCE_INVALID')
    require(metadata['taskId'] == 'Mjlab-Velocity-Flat-MicroDuck'
            and metadata['observationDimension'] == 61 and metadata['actionDimension'] == 14
            and metadata['controlHz'] == 50 and metadata['normalizerBaked'] is True
            and metadata['observationLayout'] == 'MICRODUCK_OBS_61_V1'
            and metadata['jointNames'] == list(CONTROLLED_SERVO_JOINTS)
            and metadata['provenance'] in ('MANAGED', 'EXTERNAL_UNVERIFIED'), 'POLICY_CONTRACT_INCOMPATIBLE')


def publish_bundle(policy: Path, output: Path, pins: dict):
    require(not policy.is_symlink() and 0 < policy.stat().st_size <= MAX_POLICY, 'POLICY_SIZE_INVALID')
    data = policy.read_bytes()
    normalizer = _prove(data)
    metadata = dict(pins, policySha256=sha(data), normalizerDigest=normalizer,
                    taskId='Mjlab-Velocity-Flat-MicroDuck', observationDimension=61, actionDimension=14,
                    controlHz=50, observationLayout='MICRODUCK_OBS_61_V1', normalizerBaked=True,
                    jointNames=list(CONTROLLED_SERVO_JOINTS), provenance='MANAGED')
    metadata_valid(metadata)
    temporary = output.with_suffix('.tmp.zip')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=False) as bundle:
            bundle.writestr('manifest.json', json.dumps(dict(schema=SCHEMA, policy=metadata, policySizeBytes=len(data)), sort_keys=True, allow_nan=False))
            bundle.writestr('policy.onnx', data)
        with temporary.open('rb') as stream: os.fsync(stream.fileno())
        os.replace(temporary, output)
        with output.open('rb') as stream: digest = 'sha256:'+hashlib.file_digest(stream, 'sha256').hexdigest()
        return dict(artifactId=output.name, kind='BUNDLE', sha256=digest, sizeBytes=output.stat().st_size)
    finally:
        temporary.unlink(missing_ok=True)


def verify_bundle(path: Path):
    require(not path.is_symlink() and 0 < path.stat().st_size <= MAX_BUNDLE, 'BUNDLE_SIZE_INVALID')
    try:
        with zipfile.ZipFile(path) as bundle:
            entries = bundle.infolist()
            require(len(entries) == 2 and {e.filename for e in entries} == {'manifest.json', 'policy.onnx'}, 'BUNDLE_MEMBERS_INVALID')
            for entry in entries:
                require(not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16)
                        and not entry.flag_bits & 1, 'BUNDLE_MEMBER_INVALID')
            require(0 < bundle.getinfo('manifest.json').file_size <= 65536, 'BUNDLE_MANIFEST_INVALID')
            manifest = json.loads(bundle.read('manifest.json'), object_pairs_hook=_unique)
            require(set(manifest) == {'schema', 'policy', 'policySizeBytes'} and manifest['schema'] == SCHEMA, 'BUNDLE_SCHEMA_INVALID')
            metadata = manifest['policy']
            metadata_valid(metadata)
            length = bundle.getinfo('policy.onnx').file_size
            require(type(manifest['policySizeBytes']) is int and 0 < length <= MAX_POLICY and length == manifest['policySizeBytes'], 'POLICY_SIZE_INVALID')
            data = bundle.read('policy.onnx')
            require(sha(data) == metadata['policySha256'], 'POLICY_DIGEST_MISMATCH')
            require(_prove(data) == metadata['normalizerDigest'], 'POLICY_NORMALIZER_MISMATCH')
            return metadata
    except (zipfile.BadZipFile, KeyError, TypeError) as invalid:
        raise ValueError('BUNDLE_INVALID') from invalid
