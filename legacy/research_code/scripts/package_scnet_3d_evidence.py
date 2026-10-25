#!/usr/bin/env python3
"""Package the accepted study's lightweight evidence and offline redraw tools."""
import argparse
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def main():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=repository/'outputs/scnet_3d_20260909')
    args = parser.parse_args()
    root = args.root.resolve()
    verification = json.loads((root/'verification.json').read_text())
    if not verification['uniform_complete'] or verification['invalid_complete_reports']:
        raise SystemExit('The complete matched study must pass verification before packaging.')

    paths = set()
    for name in ('expected_cases.json', 'extra_runs.json', 'accounting.psv',
                 'verification.json', 'results_audit_zh.md'):
        paths.add(root/name)
    for directory in ('profiles', 'manifests', 'figures'):
        paths.update(p for p in (root/directory).rglob('*') if p.is_file() and
                     p.suffix in {'.json', '.csv', '.png', '.pdf', '.svg'})
    for directory in root.glob('remote_*'):
        paths.update(p for p in directory.rglob('*') if p.is_file() and
                     p.suffix in {'.json', '.csv', '.out', '.err', '.psv', '.md'})
    for name in ('helm3d_v7_source.tar.gz', 'SHA256SUMS', 'REPRODUCE_ANALYSIS.md',
                 'analysis_environment.json', 'requirements-analysis.txt'):
        paths.add(root/'delivery'/name)
    entries = {str(Path('outputs/scnet_3d_20260909')/p.relative_to(root)): p.read_bytes()
               for p in paths}
    scripts = ('verify_scnet_3d_results.py', 'summarize_scnet_3d.py',
               'plot_scnet_3d.py', 'plot_scnet_3d_cost.py',
               'plot_scnet_3d_layer_storage.py', 'plot_scnet_3d_solver_probe.py',
               'redraw_scnet_3d_study.py', 'package_scnet_3d_evidence.py')
    for name in scripts:
        entries['scripts/'+name] = (repository/'scripts'/name).read_bytes()
    for name in ('md/scnet_3d_parallel_study_20260909.md',
                 'agent_runs/2026-09-09_scnet_3d_parallel.md'):
        entries[name] = (repository/name).read_bytes()
    entries['README.md'] = ("""# Three-dimensional Helmholtz ROM study

This archive contains the completed study's collected JSON/CSV/log evidence,
all figures, the frozen HPC source archive, and the offline analysis scripts.
Original failed configurations and adverse parameter/layer results are retained.
Large numerical MAT files remain in the remote directories recorded in the audit.
No SSH credentials, sockets, MATLAB binaries or MAT files are included.

After extraction, from this directory:

```bash
python -m pip install -r outputs/scnet_3d_20260909/delivery/requirements-analysis.txt
python scripts/redraw_scnet_3d_study.py --require-uniform-complete
```

This command checks the saved evidence and redraws the figures without SSH or
MATLAB. It verifies the true residuals exported by the numerical runs; it does
not recompute A*x locally. Regeneration also updates audit links to this location.
Numerical reproduction instructions are inside the separate frozen source archive
at outputs/scnet_3d_20260909/delivery/helm3d_v7_source.tar.gz.

The matched study uses one node/profile, separately trains a three-layer ROM at
each frequency, and tests all three held-out parameters with all three methods.
It validates Helmholtz ROM with shifted-MG completion. General complex-RHS/PML
validation of a three-dimensional WaveHoltz ROM remains outside these results.
See outputs/scnet_3d_20260909/results_audit_zh.md for results and cost definitions.
MANIFEST.json records a SHA-256 digest for every other archive entry.
""").encode()
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
                    files={name: hashlib.sha256(data).hexdigest()
                           for name, data in sorted(entries.items())})
    entries['MANIFEST.json'] = (json.dumps(manifest, indent=2)+'\n').encode()
    archive = root/'delivery/helm3d_evidence.tar.gz'
    temporary = archive.with_suffix('.tmp')
    with tarfile.open(temporary, 'w:gz') as output:
        for name, data in sorted(entries.items()):
            info = tarfile.TarInfo('helm3d_study/'+name)
            info.size = len(data)
            info.mode = 0o644
            output.addfile(info, io.BytesIO(data))
    temporary.replace(archive)
    with tarfile.open(archive, 'r:gz') as output:
        for name, expected in manifest['files'].items():
            actual = hashlib.sha256(output.extractfile('helm3d_study/'+name).read()).hexdigest()
            if actual != expected:
                raise RuntimeError('Archive digest mismatch: '+name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name('EVIDENCE_SHA256SUMS').write_text(f'{digest}  {archive.name}\n')
    print(json.dumps(dict(archive=str(archive), files=len(entries),
                          bytes=archive.stat().st_size, sha256=digest)))


if __name__ == '__main__':
    main()
