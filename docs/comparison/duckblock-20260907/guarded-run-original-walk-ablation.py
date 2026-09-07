"""Run only the dedicated comparison container; stop it on production changes."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path('/home/mcao/microduck-comparison')
IMAGE = 'sha256:db653ec0ea349ffb8c3996d2b44368c7d24754694582253a501e8cff0b8cc77d'
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
EVIDENCE = ROOT / 'run-evidence' / ('eval-6fc54a0-' + STAMP)
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
config = json.loads((ROOT/'input/walk-ablation.json').read_text())
save('inputs.json',dict(experimentId=config['experimentId'],configSha256='sha256:'+hashlib.sha256((ROOT/'input/walk-ablation.json').read_bytes()).hexdigest(),evaluatorSha256='sha256:'+hashlib.sha256((ROOT/'source-6fc54a0/scripts/evaluate_sitstand_comparison.py').read_bytes()).hexdigest(),codeRevisions=config['codeRevisions']))
common = ['docker','run','-d','--user','10001:10001','--cpus','4','--memory','8g','--memory-swap','8g','--gpus','all','--network','none','--read-only','--tmpfs','/tmp:rw,size=536870912','--workdir','/tmp','--env','MUJOCO_GL=egl','--env','PYTHONPATH=/experiment/src','--env','MICRODUCK_ROM_BEARER_TOKEN_FILE=','--mount',f'type=bind,src={ROOT}/source-6fc54a0,dst=/experiment,readonly','--mount',f'type=bind,src={ROOT}/input,dst=/input,readonly','--mount',f'type=bind,src={ROOT}/output,dst=/output']
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
    execute('microduck-comparison-eval-'+STAMP.lower(),['/experiment/scripts/evaluate_sitstand_comparison.py','--config','/input/walk-ablation.json','--output-root','/output'])
finally:
    after = snapshot()
    save('production-after.json',after)
    save('production-guard-result.json',dict(unchanged=after==baseline,baselineContainers=len(baseline)))
    monitor.close()
    print('EVIDENCE',EVIDENCE,flush=True)
