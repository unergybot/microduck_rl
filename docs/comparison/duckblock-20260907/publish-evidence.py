"""Attach captured diagnostic evidence and atomically publish; no ROM connection."""
import argparse
import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path


def sha(path):
    return 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--producer', type=Path, required=True)
    p.add_argument('--run-evidence', type=Path, required=True)
    p.add_argument('--source-inventory', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--evaluator', type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.producer / 'report.json').read_text())
    manifest = json.loads((args.producer / 'manifest.json').read_text())
    if report['experimentId'] != manifest['experimentId']:
        raise ValueError('producer identity mismatch')
    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.output_root / report['experimentId']).exists():
        raise FileExistsError(report['experimentId'])
    stage = Path(tempfile.mkdtemp(prefix='.' + report['experimentId'] + '-', dir=args.output_root))
    try:
        for a in manifest['artifacts']:
            source = args.producer / a['path']
            if source.is_symlink() or source.parent != args.producer or not source.is_file():
                raise ValueError('invalid producer path')
            if sha(source) != a['sha256'] or source.stat().st_size != a['sizeBytes']:
                raise ValueError('producer integrity failure')
            name = 'producer-report.json' if a['id'] == 'report' else source.name
            shutil.copy2(source, stage / name)
        shutil.copy2(args.producer / 'manifest.json', stage / 'producer-manifest.json')
        mappings = {
            'source-inventory.json': args.source_inventory,
            'browser-parity.json': args.workspace / 'browser/parity-report.json',
            'browser-parity-samples.json': args.workspace / 'browser/sitstand-parity.json',
            'browser-states.json': args.workspace / 'browser/states.json',
            'browser-versions.json': args.workspace / 'browser/browser-versions.json',
            'browser-dependencies.json': args.workspace / 'browser/source-cache/index.json',
            'browser-model-comparison.json': args.workspace / 'browser/model-comparison.json',
            'browser-reference.mp4': args.workspace / 'media/browser-reference.mp4',
        }
        walking_hashes = {c['walkingPolicySha256'] for c in report['candidates'] if c.get('walkingPolicySha256')}
        if walking_hashes:
            mappings['walking-parity.json'] = args.workspace / 'browser/walking-parity.json'
            mappings['walking-parity-samples.json'] = args.workspace / 'browser/walking-parity-samples.json'
        for name, source in mappings.items():
            shutil.copy2(source, stage / name)
        parity = json.loads((stage / 'browser-parity.json').read_text())
        if not parity['passed'] or any(c['policySha256'] != parity['policySha256'] for c in report['candidates']):
            raise ValueError('browser parity is not bound to all SitStand identities')
        if sha(stage / 'browser-parity-samples.json') != parity['sampleSha256']:
            raise ValueError('parity sample integrity failure')
        if walking_hashes:
            walking = json.loads((stage / 'walking-parity.json').read_text())
            if not walking['passed'] or not walking_hashes.issubset({walking['policySha256'], parity['policySha256']}):
                raise ValueError('walking parity policy identity mismatch')
            if sha(stage / 'walking-parity-samples.json') != walking['sampleSha256']:
                raise ValueError('walking parity sample integrity failure')
            report['checks'].append({'name': 'WALKING_WASM_PYTHON_PARITY', 'status': 'PASSED', 'reason': f"{walking['samples']} identical61D inputs, atol1e-5/rtol1e-4; max absolute action error {walking['maxAbsoluteError']}"})
        evidence = {f.stem: json.loads(f.read_text()) for f in args.run_evidence.glob('*.json')}
        before, after = evidence['production-before'], evidence['production-after']
        if before != after or not evidence['production-guard-result']['unchanged']:
            raise ValueError('production guard failed')
        (stage / 'execution-evidence.json').write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
        (stage / 'inputs.json').write_text(json.dumps(report['provenance']['config'], indent=2, allow_nan=False) + '\n')
        report['provenance']['attachedEvidence'] = {
            'producerReportArtifactId': 'producer-report',
            'producerManifestArtifactId': 'producer-manifest',
            'executionArtifactId': 'execution-evidence',
            'browserParityArtifactId': 'browser-parity',
            'browserReferenceVideoArtifactId': 'browser-reference',
            'sourceInventoryArtifactId': 'source-inventory',
            'publisherSha256': sha(Path(__file__)),
            'browserCaptureNote': '1fps screenshot replay; states.second is a sample index, not measured simulation time. No qualification metrics derived from video.',
            'limitations': [
                'Browser uses MuJoCo3.11.0/ORT-WASM1.27.0; CPU evaluation uses MuJoCo3.10.0/ORT1.24.4.',
                'Captured browser XML has runtime terrain changes; diagnostic cases use the explicitly hashed flat/fork scenes.',
                'Browser entrance animation and unseeded initialization are not claimed identical to diagnostic seeded HOME/SIT resets.',
                'External training lineage and independent model licensing are unverified; no ROM qualification is granted.',
            ],
        }
        report['checks'] += [
            {'name': 'BROWSER_WASM_PYTHON_PARITY', 'status': 'PASSED', 'reason': f"{parity['samples']} identical61D inputs, atol1e-5/rtol1e-4; max absolute action error {parity['maxAbsoluteError']}"},
            {'name': 'PRODUCTION_CONTAINER_GUARD', 'status': 'PASSED', 'reason': 'Production identities, images, restart counts and health unchanged; monitored during isolated execution.'},
            {'name': 'OFFLINE_DIAGNOSTIC_ONLY', 'status': 'INFO', 'reason': '离线策略评估 · 非实时训练 · 非 ROM 验收'},
        ]
        spec = importlib.util.spec_from_file_location('evaluator', args.evaluator)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if any(f.stat().st_size > 10 * 1024 * 1024 for f in stage.iterdir()):
            raise ValueError('live10MiB artifact limit exceeded')
        if len(json.dumps(report, indent=2, allow_nan=False).encode()) > 2 * 1024 * 1024:
            raise ValueError('live2MiB report limit exceeded')
        stage.chmod(0o755)
        final = module.publish(stage, args.output_root, report)
        print(final)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == '__main__':
    main()
