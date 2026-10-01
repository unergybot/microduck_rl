"""Run only the dedicated comparison container; stop it on production changes."""
import argparse
import re
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True,help='Dedicated experiment directory on Docker host')
parser.add_argument('--config',required=True,help='Config path relative to root, e.g. input/handoff.json')
parser.add_argument('--source-dir',required=True,help='Frozen source directory relative to root')
parser.add_argument('--source-commit',required=True,help='Full evaluator source commit used by git archive')
parser.add_argument('--image',required=True,help='Exact sha256 Docker image ID, not a mutable tag')
args = parser.parse_args()
if not re.fullmatch(r'[0-9a-f]{40}',args.source_commit):
    parser.error('--source-commit requires full lowercase40hex commit')
if not re.fullmatch(r'sha256:[0-9a-f]{64}',args.image):
    parser.error('--image requires full sha256 image ID')
ROOT = args.root.resolve()
CONFIG = (ROOT/args.config).resolve()
SOURCE = (ROOT/args.source_dir).resolve()
if not CONFIG.is_relative_to(ROOT) or not SOURCE.is_relative_to(ROOT):
    parser.error('config and source must stay within dedicated root')
IMAGE = args.image
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
EVIDENCE = ROOT / 'run-evidence' / ('eval-' + args.source_commit[:7] + '-' + STAMP)
EVIDENCE.mkdir()

def command(args):
    return subprocess.check_output(args,text=True).strip()

def snapshot():
    ids = command(['docker','ps','-aq']).splitlines()
    rows = json.loads(command(['docker','inspect',*ids])) if ids else []
    return {r['Name'].lstrip('/'):dict(Id=r['Id'],Image=r['Image'],RestartCount=r['RestartCount'],Running=r['State']['Running'],Status=r['State']['Status'],Health=r['State'].get('Health',{}).get('Status','none')) for r in rows if r['Name'].startswith(('/unergy-','/tairos-'))}

def save(name,value):
    (EVIDENCE/name).write_text(json.dumps(value,indent=2)+'\n')

baseline = snapshot()
save('production-before.json',baseline)
if any(r['Running'] and r['Health'] not in ('none','healthy') for r in baseline.values()):
    raise RuntimeError('production baseline is not healthy')
image = json.loads(command(['docker','image','inspect',IMAGE]))[0]
save('image.json',{k:image[k] for k in ('Id','RepoTags','RepoDigests','Created','Architecture','Os')})
config = json.loads(CONFIG.read_text())
save('inputs.json',dict(experimentId=config['experimentId'],configSha256='sha256:'+hashlib.sha256(CONFIG.read_bytes()).hexdigest(),evaluatorSha256='sha256:'+hashlib.sha256((SOURCE/'scripts/evaluate_sitstand_comparison.py').read_bytes()).hexdigest(),codeRevisions=config['codeRevisions'],evaluatorCommit=args.source_commit))
common = ['docker','run','-d','--user','10001:10001','--cpus','4','--memory','8g','--memory-swap','8g','--gpus','all','--network','none','--read-only','--tmpfs','/tmp:rw,size=536870912','--workdir','/tmp','--env','MUJOCO_GL=egl','--env','PYTHONPATH=/experiment/src','--env','MICRODUCK_ROM_BEARER_TOKEN_FILE=','--mount',f'type=bind,src={SOURCE},dst=/experiment,readonly','--mount',f'type=bind,src={CONFIG.parent},dst=/input,readonly','--mount',f'type=bind,src={ROOT}/output,dst=/output']
monitor = (EVIDENCE/'monitor.ndjson').open('w')

def check_production():
    current = snapshot()
    monitor.write(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),unchanged=current==baseline,containers=current))+'\n')
    monitor.flush()
    return current == baseline and not any(r['Running'] and r['Health'] not in ('none','healthy') for r in current.values())

def execute(name,args):
    cmd = common + ['--name',name,IMAGE] + args
    (EVIDENCE/(name+'-command.txt')).write_text(shlex.join(cmd)+'\n')
    if not check_production():
        raise RuntimeError('production changed before start')
    cid = command(cmd)
    print('STARTED',name,cid,flush=True)
    try:
        while True:
            if not check_production():
                subprocess.run(['docker','stop','--time','2',cid],check=False,capture_output=True)
                raise RuntimeError('production identity/restart/health changed; stopped comparison only')
            state = json.loads(command(['docker','inspect','--format','{{json .State}}',cid]))
            if not state['Running']:
                logs = subprocess.run(['docker','logs',cid],check=False,capture_output=True,text=True)
                (EVIDENCE/(name+'.log')).write_text(logs.stdout+logs.stderr)
                save(name+'-state.json',state)
                print('FINISHED',name,'exit',state['ExitCode'],flush=True)
                if state['ExitCode']:
                    print((logs.stdout+logs.stderr)[-4000:],flush=True)
                    raise RuntimeError('comparison container exited nonzero')
                if 'smoke' in name:
                    print(logs.stdout,flush=True)
                return
            latest = subprocess.run(['docker','logs','--tail','1',cid],check=False,capture_output=True,text=True)
            print('MONITOR',datetime.datetime.now(datetime.timezone.utc).isoformat(),name,(latest.stdout+latest.stderr).strip()[-250:],flush=True)
            time.sleep(5)
    finally:
        # This name/ID belongs solely to this invocation; never touch production.
        state = json.loads(command(['docker','inspect','--format','{{json .State}}',cid]))
        if state['Running']:
            subprocess.run(['docker','stop','--time','2',cid],check=False,capture_output=True)
        logs = subprocess.run(['docker','logs',cid],check=False,capture_output=True,text=True)
        (EVIDENCE/(name+'.log')).write_text(logs.stdout+logs.stderr)

try:
    smoke = "import json,mujoco,numpy,onnxruntime; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type=\"sphere\" size=\".1\"/></worldbody></mujoco>'); d=mujoco.MjData(m); mujoco.mj_forward(m,d); r=mujoco.Renderer(m,height=120,width=160); r.update_scene(d); frame=r.render(); print(json.dumps(dict(mujoco=mujoco.__version__,onnxruntime=onnxruntime.__version__,numpy=numpy.__version__,eglFrameShape=list(frame.shape),eglPassed=bool(frame.size)))); r.close()"
    execute('microduck-comparison-smoke-'+STAMP.lower(),['-c',smoke])
    execute('microduck-comparison-eval-'+STAMP.lower(),['/experiment/scripts/evaluate_sitstand_comparison.py','--config','/input/'+CONFIG.name,'--output-root','/output'])
finally:
    after = snapshot()
    save('production-after.json',after)
    save('production-guard-result.json',dict(unchanged=after==baseline,baselineContainers=len(baseline)))
    monitor.close()
    print('EVIDENCE',EVIDENCE,flush=True)
