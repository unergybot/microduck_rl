"""Experimental recipe-aware packages, separate from the walking-only V1 schema."""
import json
import os
from pathlib import Path
import zipfile
from .configuration import verify_experiment,canonical_json
from mjlab_microduck.rom.managed_bundle import _prove,sha,require,MAX_POLICY
from mjlab_microduck.rom.contracts import CONTROLLED_SERVO_JOINTS


def publish_bundle(policy,output,pins,spec):
    spec=verify_experiment(spec)
    require(not policy.is_symlink() and 0<policy.stat().st_size<=MAX_POLICY,'POLICY_SIZE_INVALID')
    data=policy.read_bytes();normalizer=_prove(data)
    metadata=dict(pins,policySha256=sha(data),normalizerDigest=normalizer,taskId=spec['taskId'],actionId=spec['actionId'],robotProfileId=spec['robotProfileId'],experimentConfigDigest=spec['configDigest'],experimentRecipeDigest=spec['recipeDigest'],observationDimension=61,actionDimension=14,controlHz=50,observationLayout='MICRODUCK_OBS_61_V1',normalizerBaked=True,jointNames=list(CONTROLLED_SERVO_JOINTS),commandProfile=spec['commandProfile'],actionFilter='NONE',provenance='MANAGED',executionEnvironment='SIMULATOR',qualified=False)
    manifest=dict(schema='ROM_MANAGED_POLICY_BUNDLE_V2',policy=metadata,policySizeBytes=len(data),experiment=spec)
    require(len(canonical_json(manifest).encode())<=131072,'BUNDLE_MANIFEST_INVALID')
    temporary=output.with_suffix('.tmp.zip')
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=False) as bundle:
            bundle.writestr('manifest.json',canonical_json(manifest));bundle.writestr('policy.onnx',data)
        with temporary.open('rb') as stream:os.fsync(stream.fileno())
        os.replace(temporary,output)
        return dict(artifactId=output.name,kind='BUNDLE',sha256=sha(output.read_bytes()),sizeBytes=output.stat().st_size)
    finally:temporary.unlink(missing_ok=True)
