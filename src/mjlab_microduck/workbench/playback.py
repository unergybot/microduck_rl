"""Hash-bound normalized PT/ONNX inference; no launch or file-lookup authority."""
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
from .configuration import verify_experiment,canonical_json
from .evaluation import require

_MAX_POLICY=256*1024*1024


def _bytes(path,expected_sha256):
    require(type(expected_sha256) is str and bool(re.fullmatch('sha256:[a-f0-9]{64}',expected_sha256)),'EVALUATION_POLICY_DIGEST_INVALID')
    path=Path(path)
    require(path.is_absolute(),'EVALUATION_POLICY_FILE_INVALID')
    fd=None
    try:
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
        info=os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and info.st_uid==os.getuid() and 0<info.st_size<=_MAX_POLICY,'EVALUATION_POLICY_FILE_INVALID')
        with os.fdopen(fd,'rb') as stream:
            fd=None;data=stream.read(_MAX_POLICY+1)
        require(len(data)==info.st_size and len(data)<=_MAX_POLICY,'EVALUATION_POLICY_FILE_INVALID')
        require('sha256:'+hashlib.sha256(data).hexdigest()==expected_sha256,'EVALUATION_POLICY_DIGEST_MISMATCH')
        return data
    except OSError:raise ValueError('EVALUATION_POLICY_FILE_INVALID') from None
    finally:
        if fd is not None:os.close(fd)


class PlaybackPolicy:
    def __init__(self,kind,sha256,spec,*,module=None,session=None):
        self.kind=kind;self.sha256=sha256;self.config_digest=spec['configDigest']
        self.spec=deepcopy(spec);self.module=module;self.session=session

    @classmethod
    def from_checkpoint(cls,path,spec,expected_sha256):
        spec=verify_experiment(spec);data=_bytes(path,expected_sha256)
        import torch
        from tensordict import TensorDict
        from rsl_rl.models import MLPModel
        from .engine import build_configs
        _,agent=build_configs(spec)
        model=MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),{'actor':['actor']},'actor',14,hidden_dims=agent.actor.hidden_dims,activation=agent.actor.activation,obs_normalization=True,distribution_cfg=dict(agent.actor.distribution_cfg))
        try:
            saved=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
            require(type(saved) is dict and isinstance(saved.get('actor_state_dict'),dict),'EVALUATION_CHECKPOINT_INVALID')
            state=saved['actor_state_dict']
            require(any(k.startswith('obs_normalizer.') for k in state),'EVALUATION_NORMALIZER_MISSING')
            require(all(isinstance(v,torch.Tensor) and bool(torch.isfinite(v).all()) for v in state.values()),'EVALUATION_CHECKPOINT_INVALID')
            model.load_state_dict(state,strict=True)
        except Exception:raise ValueError('EVALUATION_CHECKPOINT_INVALID') from None
        model.eval();model.requires_grad_(False)
        return cls('PT',expected_sha256,spec,module=model)

    @classmethod
    def from_onnx(cls,path,spec,expected_sha256):
        spec=verify_experiment(spec);data=_bytes(path,expected_sha256)
        import onnx
        import onnxruntime as ort
        from mjlab_microduck.rom.managed_bundle import _prove,_unique
        try:
            normalizer=_prove(data)
            model=onnx.load_model_from_string(data)
            metadata={entry.key:entry.value for entry in model.metadata_props}
            require(len(metadata)==len(model.metadata_props),'EVALUATION_ONNX_METADATA_INVALID')
            for key,value in {'microduck.task_id':spec['taskId'],'microduck.experiment_config_digest':spec['configDigest'],'microduck.recipe_digest':spec['recipeDigest'],'microduck.robot_profile_id':spec['robotProfileId'],'microduck.action_id':spec['actionId'],'microduck.action_filter':'NONE','microduck.observation_contract':'MICRODUCK_OBS_61_V1','microduck.action_contract':'MICRODUCK_ACTION_14_V1','microduck.normalization':'EMPIRICAL_NORMALIZATION_V1','microduck.normalized_graph_fingerprint':normalizer}.items():
                require(metadata.get(key)==value,'EVALUATION_ONNX_METADATA_MISMATCH')
            command=metadata.get('microduck.command_profile','')
            require(len(command.encode())<=16384,'EVALUATION_ONNX_METADATA_INVALID')
            require(canonical_json(json.loads(command,object_pairs_hook=_unique))==canonical_json(spec['commandProfile']),'EVALUATION_ONNX_METADATA_MISMATCH')
            options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
            session=ort.InferenceSession(data,sess_options=options,providers=['CPUExecutionProvider'])
        except Exception:raise ValueError('EVALUATION_ONNX_INVALID') from None
        return cls('ONNX',expected_sha256,spec,session=session)

    def predict(self,raw):
        import torch
        require(isinstance(raw,torch.Tensor) and tuple(raw.shape)==(1,61) and raw.device.type=='cpu' and bool(torch.isfinite(raw).all()),'EVALUATION_OBSERVATION_INVALID')
        value=raw.detach().to(dtype=torch.float32)
        with torch.inference_mode():
            if self.kind=='PT':
                from tensordict import TensorDict
                result=self.module(TensorDict({'actor':value},batch_size=[1]))
            else:
                output=self.session.run(None,{self.session.get_inputs()[0].name:value.numpy()})
                require(len(output)==1,'EVALUATION_POLICY_OUTPUT_INVALID')
                result=torch.from_numpy(output[0])
        require(tuple(result.shape)==(1,14) and bool(torch.isfinite(result).all()),'EVALUATION_POLICY_OUTPUT_INVALID')
        return result


def compare_playback(pt,onnx,samples):
    import torch
    require(isinstance(pt,PlaybackPolicy) and pt.kind=='PT' and isinstance(onnx,PlaybackPolicy) and onnx.kind=='ONNX','EVALUATION_POLICY_INVALID')
    require(pt.config_digest==onnx.config_digest,'EVALUATION_POLICY_CONFIG_MISMATCH')
    require(isinstance(samples,torch.Tensor) and samples.ndim==2 and samples.shape[1]==61 and 1<=samples.shape[0]<=2048 and samples.device.type=='cpu' and bool(torch.isfinite(samples).all()),'EVALUATION_OBSERVATION_INVALID')
    errors=torch.cat([(pt.predict(row[None]).double()-onnx.predict(row[None]).double()).abs() for row in samples])
    report=dict(evaluation='CONSISTENCY_ONLY',samples=len(samples),configDigest=pt.config_digest,ptSha256=pt.sha256,onnxSha256=onnx.sha256,maxAbsoluteError=float(errors.max()),meanAbsoluteError=float(errors.mean()),rootMeanSquaredError=float(errors.square().mean().sqrt()))
    require(all(math.isfinite(report[key]) for key in ('maxAbsoluteError','meanAbsoluteError','rootMeanSquaredError')),'EVALUATION_CONSISTENCY_NONFINITE')
    return report
