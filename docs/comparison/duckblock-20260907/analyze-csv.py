import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
report = json.loads((root/'report.json').read_text())
manifest = json.loads((root/'manifest.json').read_text())
artifacts = {a['id']:a for a in manifest['artifacts']}
for item in manifest['artifacts']:
    path = root/item['path']
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_size == item['sizeBytes']
    assert 'sha256:'+hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
summary = dict(experimentId=report['experimentId'],candidateCount=len(report['candidates']),caseCount=len(report['cases']),artifactCount=len(artifacts),reportBytes=(root/'report.json').stat().st_size,maxArtifactBytes=max(a['sizeBytes'] for a in artifacts.values()),totalArtifactBytes=sum(a['sizeBytes'] for a in artifacts.values()),hashesVerified=True,matrix=[],seed7SitDiagnostics=[])
summary['sizeLimitsPassed'] = summary['reportBytes'] <= 2*1024**2 and summary['maxArtifactBytes'] <= 10*1024**2
names = ['left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle','neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle']
for candidate in report['candidates']:
    cases = [c for c in report['cases'] if c['candidateId']==candidate['id']]
    for scenario in ('STAND_HOLD','STAND_TO_SIT','SIT_TO_STAND','STAND_SIT_STAND'):
        subset = [c for c in cases if c['scenario']==scenario]
        summary['matrix'].append(dict(candidateId=candidate['id'],scenario=scenario,passed=sum(c['status']=='PASSED' for c in subset),total=len(subset),reasons=dict(collections.Counter(c['reason'] for c in subset))))
    case = next(c for c in cases if c['scenario']=='STAND_TO_SIT' and c['seed']==7)
    if not case['csvArtifactId']:
        continue
    with (root/artifacts[case['csvArtifactId']]['path']).open() as stream:
        rows = list(csv.DictReader(stream))[-50:]
    target = candidate['effectiveBehavior']['sitPose']
    height = candidate['effectiveBehavior']['sitHeightM']
    diag = dict(candidateId=candidate['id'],reason=case['reason'],sampleCount=len(rows),windowStartSeconds=float(rows[0]['timeSeconds']),windowEndSeconds=float(rows[-1]['timeSeconds']),criteriaPassSamples={},jointErrors=[])
    predicates = dict(pose=lambda r:max(abs(float(r[f'jointPosition{i}'])-target[i]) for i in range(14))<=.08,height=lambda r:abs(float(r['heightM'])-height)<=.015,tilt=lambda r:float(r['maxTiltRad'])<=math.radians(15),speed=lambda r:float(r['maxJointSpeedRadps'])<=.5)
    for key,predicate in predicates.items():
        diag['criteriaPassSamples'][key] = sum(predicate(r) for r in rows)
    for i,name in enumerate(names):
        errors = [abs(float(r[f'jointPosition{i}'])-target[i]) for r in rows]
        diag['jointErrors'].append(dict(joint=name,targetRad=target[i],finalPositionRad=float(rows[-1][f'jointPosition{i}']),finalErrorRad=errors[-1],minErrorRad=min(errors),maxErrorRad=max(errors),meanErrorRad=sum(errors)/len(errors),overThresholdSamples=sum(e>.08 for e in errors)))
    diag['jointErrors'].sort(key=lambda e:e['meanErrorRad'],reverse=True)
    diag['final'] = {key:float(rows[-1][key]) for key in ('maxPoseErrorRad','heightM','maxTiltRad','maxJointSpeedRadps')}
    summary['seed7SitDiagnostics'].append(diag)
summary['falls'] = sum(c['metrics'].get('falls',0) for c in report['cases'])
summary['videoUnavailableCases'] = [c['id'] for c in report['cases'] if c.get('videoUnavailableReason')]
print(json.dumps(summary,indent=2))
