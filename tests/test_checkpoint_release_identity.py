"""Checkpoint compatibility depends on training data/configuration, not packaging."""
import copy
import hashlib
import json

import pytest
from mjlab_microduck.workbench import canonical_json, resolve_experiment
from mjlab_microduck.workbench.checkpoints import trusted_checkpoint


def test_retained_checkpoint_survives_reader_rebuild_but_not_model_or_dependency_change(tmp_path):
    parent = tmp_path / 'parent'
    parent.mkdir(mode=0o700)
    spec = resolve_experiment(dict(schemaVersion='ROM_TRAINING_EXPERIMENT_V1',
        recipeId='Mjlab-Velocity-Flat-MicroDuck', robotProfileId='rom-xl330-v1', config={}))
    original_scope = dict(agentId='edge', agentGeneration='2', trainerId='trainer', gpuUuid='gpu',
        containerId='old', imageReference='old-image')
    limits = dict(modelDigest='model', dependencyDigest='dependencies', recipeDigest='old-source')
    original = dict(jobId='parent', requestDigest='sha256:'+'a'*64, scope=original_scope,
        limits=limits, experimentJson=canonical_json(spec))
    data = b'complete original checkpoint'
    digest = 'sha256:' + hashlib.sha256(data).hexdigest()
    artifact = dict(artifactId='checkpoint-'+digest[7:]+'.pt', kind='CHECKPOINT', sha256=digest, sizeBytes=len(data))
    index = dict(schemaVersion='ROM_TRAINING_CHECKPOINT_INDEX_V1', jobId='parent',
        requestDigest=original['requestDigest'], configDigest=spec['configDigest'],
        retentionPolicy=dict(maxCheckpointFiles=10, metadataMaxEntries=4096),
        checkpoints=[dict(iteration=10, artifact=artifact)], evictedCheckpoints=[])
    for name,value in [('request.json',original),('experiment.json',spec),('checkpoint-index.json',index)]:
        (parent/name).write_text(canonical_json(value))
    (parent/artifact['artifactId']).write_bytes(data)
    child = dict(scope={**original_scope,'containerId':'new','imageReference':'new-image'},
        limits={**limits,'recipeDigest':'new-packaging-source','maxIterations':2000})
    ref = dict(jobId='parent', checkpointSha256=digest, configDigest=spec['configDigest'])
    assert trusted_checkpoint(tmp_path/'child',child,ref,snapshot=True) == (data,spec)
    for field in ('modelDigest','dependencyDigest'):
        changed=copy.deepcopy(child)
        changed['limits'][field]='incompatible'
        with pytest.raises(ValueError):trusted_checkpoint(tmp_path/'child',changed,ref)
    changed=copy.deepcopy(child)
    changed['scope']['agentGeneration']='3'
    with pytest.raises(ValueError):trusted_checkpoint(tmp_path/'child',changed,ref)
    assert json.loads((parent/'request.json').read_text()) == original
