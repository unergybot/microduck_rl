"""Content verified, private sibling artifacts; submitted host paths never accepted."""
import json
import hashlib
import io
import os
import re
import stat
from pathlib import Path
from .configuration import verify_experiment,canonical_json
from .engine import checked


def compatible(source,target):
    for key in ('robotProfileId','backlash'):checked(source[key]==target[key])
    # Teachers have different reward/command recipes, but identical physical,
    # proprioception, action and empirical normalization contracts are required.
    a,b=source['effectiveConfig'],target['effectiveConfig']
    for key in ('servo','gains','calibration','physics','actionScale','actionFilter','controlFrequencyHz','observations'):
        checked(a.get(key)==b.get(key))
    checked(source['terrain']==target['terrain'])


INDEX_LIMIT = 2 << 20
METADATA_LIMIT = 131072
CHECKPOINT_LIMIT = 256 << 20


def _fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class _OwnedDirectory:
    """Read immutable snapshots using owned nonblocking, nofollow descriptors."""

    def __init__(self, path):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(self.fd)
            checked(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
                    and not info.st_mode & 0o022)
            self.identity = (info.st_dev, info.st_ino, info.st_mode, info.st_uid)
            self.private = not info.st_mode & 0o077
        except BaseException:
            os.close(self.fd)
            raise

    def close(self):
        os.close(self.fd)

    def unchanged(self, name, proof):
        root = os.stat(self.path, follow_symlinks=False)
        checked(stat.S_ISDIR(root.st_mode) and root.st_uid == os.geteuid()
                and not root.st_mode & 0o022 and (root.st_dev, root.st_ino, root.st_mode, root.st_uid) == self.identity)
        checked(_fingerprint(os.stat(name, dir_fd=self.fd, follow_symlinks=False)) == proof)

    def read(self, name, limit):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.fd)
        try:
            before = os.fstat(fd)
            checked(stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid()
                    and before.st_nlink == 1 and not before.st_mode & 0o002
                    and (not before.st_mode & 0o020 or self.private)
                    and 0 < before.st_size <= limit)
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                data = stream.read(limit + 1)
            proof = _fingerprint(before)
            checked(len(data) == before.st_size and _fingerprint(os.fstat(fd)) == proof)
            self.unchanged(name, proof)
            return data, proof
        finally:
            os.close(fd)

    def json(self, name, limit=METADATA_LIMIT):
        data, proof = self.read(name, limit)
        def pairs(items):
            value = {}
            for key, item in items:
                checked(key not in value)
                value[key] = item
            return value
        def invalid(_):
            raise ValueError('MANAGED_CHECKPOINT_UNVERIFIED')
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
        checked(type(value) is dict)
        return value, proof


def _available_index(index, parent, spec, ref):
    checked(set(index) == {'schemaVersion', 'jobId', 'requestDigest', 'configDigest',
                           'retentionPolicy', 'checkpoints', 'evictedCheckpoints'})
    checked(index['schemaVersion'] == 'ROM_TRAINING_CHECKPOINT_INDEX_V1'
            and index['jobId'] == ref['jobId'] == parent['jobId']
            and index['requestDigest'] == parent['requestDigest']
            and index['configDigest'] == spec['configDigest'] == ref['configDigest'])
    retention = index['retentionPolicy']
    checked(type(retention) is dict and set(retention) == {'maxCheckpointFiles', 'metadataMaxEntries'}
            and type(retention['maxCheckpointFiles']) is int
            and type(retention['metadataMaxEntries']) is int
            and retention == {'maxCheckpointFiles': 10, 'metadataMaxEntries': 4096})
    seen = set()
    for key, limit in (('checkpoints', 10), ('evictedCheckpoints', 4096)):
        entries = index[key]
        checked(type(entries) is list and len(entries) <= limit)
        for entry in entries:
            checked(type(entry) is dict and set(entry) == {'iteration', 'artifact'}
                    and type(entry['iteration']) is int and 0 <= entry['iteration'] < 1 << 31)
            artifact = entry['artifact']
            checked(type(artifact) is dict and set(artifact) == {'artifactId', 'kind', 'sha256', 'sizeBytes'})
            digest = artifact['sha256']
            checked(artifact['kind'] == 'CHECKPOINT' and type(digest) is str
                    and re.fullmatch('sha256:[a-f0-9]{64}', digest) is not None
                    and artifact['artifactId'] == 'checkpoint-' + digest[7:] + '.pt'
                    and type(artifact['sizeBytes']) is int and 0 < artifact['sizeBytes'] <= CHECKPOINT_LIMIT
                    and digest not in seen)
            seen.add(digest)
    matches = [e['artifact'] for e in index['checkpoints']
               if e['artifact']['sha256'] == ref['checkpointSha256']]
    checked(len(matches) == 1)
    return matches[0]


def trusted_checkpoint(directory, job, ref, *, snapshot=False):
    """The full-identity available index is the sole experiment read authority.

    snapshot=True returns bounded verified bytes for consumers, avoiding a reopen
    after verification. The default path return is retained for discovery callers.
    """
    checked(type(ref.get('jobId')) is str and bool(re.fullmatch('[-A-Za-z0-9_]{1,80}', ref['jobId'])))
    for key in ('checkpointSha256', 'configDigest'):
        checked(type(ref.get(key)) is str and bool(re.fullmatch('sha256:[a-f0-9]{64}', ref[key])))
    root = Path(directory).parent / ref['jobId']
    owned = _OwnedDirectory(root)
    try:
        parent, parent_proof = owned.json('request.json')
        checked(parent.get('jobId') == ref['jobId']
                and type(parent.get('requestDigest')) is str
                and bool(re.fullmatch('sha256:[a-f0-9]{64}', parent['requestDigest'])))
        checked(type(parent.get('scope')) is dict and type(parent.get('limits')) is dict)
        def scope(value):
            return {k: v for k, v in value.items() if k not in ('containerId', 'imageReference')}
        # Packaging/source revisions and release quotas are provenance, not
        # checkpoint semantics. The complete experiment below plus model and
        # dependency identities remain authoritative for resume compatibility.
        checked(scope(parent['scope']) == scope(job['scope']))
        checked(all(key in parent['limits'] and parent['limits'][key] == job['limits'].get(key)
                    for key in ('modelDigest', 'dependencyDigest')))
        source, source_proof = owned.json('experiment.json')
        spec = verify_experiment(source)
        checked(spec['configDigest'] == ref['configDigest']
                and parent.get('experimentJson') == canonical_json(spec))
        index, index_proof = owned.json('checkpoint-index.json', INDEX_LIMIT)
        selected = _available_index(index, parent, spec, ref)
        data, selected_proof = owned.read(selected['artifactId'], CHECKPOINT_LIMIT)
        checked(len(data) == selected['sizeBytes']
                and 'sha256:' + hashlib.sha256(data).hexdigest() == selected['sha256'])
        # Fail closed if ownership metadata or availability changed during the read.
        for name, proof in (('request.json', parent_proof), ('experiment.json', source_proof),
                            ('checkpoint-index.json', index_proof), (selected['artifactId'], selected_proof)):
            owned.unchanged(name, proof)
        return (data if snapshot else root / selected['artifactId']), spec
    finally:
        owned.close()


def load_teachers(spec,job,directory,device):
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from .engine import build_configs
    refs=spec['effectiveConfig'].get('teacherCheckpoints',{})
    if not spec['teacherRequirements']:
        checked(not refs);return {}
    checked(set(refs)=={'walk','stand'})
    result={}
    for requirement in spec['teacherRequirements']:
        role=requirement['actionId'];data,source=trusted_checkpoint(directory,job,refs[role],snapshot=True)
        checked(source['recipeId']==requirement['recipeId']);compatible(source,spec)
        _,agent=build_configs(source)
        actor=MLPModel(TensorDict({'actor':torch.zeros(1,61,device=device)},batch_size=[1]),{'actor':['actor']},'actor',14,hidden_dims=agent.actor.hidden_dims,activation=agent.actor.activation,obs_normalization=True,distribution_cfg=dict(agent.actor.distribution_cfg)).to(device)
        saved=torch.load(io.BytesIO(data),map_location=device,weights_only=True)
        state=saved['actor_state_dict']
        checked(any(k.startswith('obs_normalizer.') for k in state))
        actor.load_state_dict(state,strict=True);actor.eval();actor.requires_grad_(False)
        result[role]=actor
    return result


def attach_teachers(runner,teachers,coefficient=.1):
    """Independent bounded supervised step on the same PPO rollout, no extra steps.

    Teachers receive raw actor observations and apply their own frozen normalizer.
    Projected gravity z in raw proprioception chooses walk (upright) or recovery.
    """
    checked(set(teachers)=={'walk','stand'} and 0<coefficient<=1)
    import torch
    alg=runner.alg;update=alg.update
    def supervised_update():
        # Keep a bounded deterministic sample before PPO clears its rollout cursor.
        obs=alg.storage.observations.flatten(0,1)[:1024].detach().clone()
        result=update()
        with torch.no_grad():
            walk=teachers['walk'](obs);stand=teachers['stand'](obs)
            target=torch.where((obs['actor'][:,5]<-.5).unsqueeze(-1),walk,stand)
        prediction=alg.actor(obs)
        loss=coefficient*(prediction-target).square().mean()
        checked(bool(torch.isfinite(loss)))
        alg.optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(alg.actor.parameters(),alg.max_grad_norm)
        alg.optimizer.step();result['teacher_loss']=float(loss.detach())
        return result
    alg.update=supervised_update
