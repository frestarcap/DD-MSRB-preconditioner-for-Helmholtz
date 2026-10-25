#!/usr/bin/env python3
"""Regenerate the 3D evidence audit and plots from collected files, without SSH."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    scripts=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=scripts.parent/'outputs/scnet_3d_20260909')
    parser.add_argument('--require-uniform-complete',action='store_true')
    args=parser.parse_args(); root=args.root.resolve()
    def run(script,*options):
        subprocess.run([sys.executable,str(scripts/script),*map(str,options)],check=True)
    run('verify_scnet_3d_results.py','--root',root,
        *(['--require-uniform-complete'] if args.require_uniform_complete else []))
    selection=['--root',root,'--profile',root/'profiles/beta2_coarse12000_restart40.json','--node','g02r1n09']
    run('plot_scnet_3d.py','--root',root)
    run('plot_scnet_3d.py','--root',root,'--problem','halfspace')
    run('plot_scnet_3d.py',*selection,'--output',root/'figures/uniform_profile')
    run('plot_scnet_3d_layer_storage.py','--root',root)
    run('plot_scnet_3d_layer_storage.py',*selection,'--output',root/'figures/uniform_layers')
    run('plot_scnet_3d_cost.py',*selection,'--output',root/'figures/uniform_profile')
    for case in json.loads((root/'extra_runs.json').read_text()):
        if case['kind']!='probe': continue
        report=root/('remote_'+case['version'])/case['tag']/'probe.json'
        if report.exists():
            run('plot_scnet_3d_solver_probe.py','--report',report,'--output',root/'figures/solver_probes'/case['tag'])
    run('summarize_scnet_3d.py','--root',root)


if __name__=='__main__': main()
