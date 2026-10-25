#!/usr/bin/env python3
"""Cross-check saved reports against configuration, CSV, logs and accounting.

This checks recorded evidence; it does not recompute A*x without the remote MAT
files. True residuals were recomputed by the MATLAB experiment before export.
Failed/incomplete runs remain in the output. No successful point is selected by
its speed or storage saving.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


def close(a, b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(
        float(a), float(b), rel_tol=2e-12, abs_tol=2e-14)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(case, root, jobs, array):
    folder=root/('remote_'+case['version'])/case['tag']
    job_id=str(case['job_id']); job=jobs.get(job_id,{})
    result=dict(job_id=job_id,tag=case['tag'],state=job.get('State','UNKNOWN'),
                complete_report=False,accepted=False)
    report_path=folder/'report.json'
    if not report_path.exists():
        return result
    result['complete_report']=True
    errors=[]
    def check(condition, description):
        if not condition: errors.append(description)
    report=json.loads(report_path.read_text())
    configuration_path=folder/'configuration.json'
    configuration=json.loads(configuration_path.read_text()); cfg=configuration['cfg']
    problem=cfg[report['problem_name']+'_cfg']
    csv_path=folder/f"helmholtz_3d_candidates_{case['tag']}.csv"
    with csv_path.open() as stream: rows=list(csv.DictReader(stream))
    filename=(f"helm3d_sweep_{array}_{case['array_index']}.out" if 'array_index' in case
              else f'helm3d_{job_id}.out')
    log_path=folder.parent/'logs'/filename; log=log_path.read_text()
    check(job.get('State')=='COMPLETED' and job.get('ExitCode')=='0:0','Slurm completion / exit code')
    check('SCNET_3D_EXPERIMENT_PASS' in log,'MATLAB acceptance marker')
    check(str(report['metadata']['job_id'])==job.get('JobIDRaw'),'report / accounting job ID')
    check(report['metadata']['node']==job.get('NodeList'),'report / accounting node')
    check(job.get('AllocCPUS')=='64' and job.get('ReqMem')=='192G','64 CPU / 192 GiB allocation')
    check(cfg['parallel_workers']==8 and cfg['worker_threads']==4,'8 workers x 4 threads')
    check(report['orthogonalization']=='cgs2_mex','measured orthogonalization backend')
    if case.get('required_node'):
        check(report['metadata']['node']==case['required_node'],'requested node')
    if case.get('profile'):
        profile=json.loads((root/case['profile']).read_text())
        check(all(report['solver_config'][k]==v==cfg[k] for k,v in profile.items()),'frozen solver profile')
        check(report['metadata']['node']=='g02r1n09','matched study node')
    n=math.prod(int(a)+2*int(b) for a,b in zip(problem['Nphys'],problem['npml']))
    check(n==report['N']==configuration['N']==case['N'],'grid / unknown count')
    check(problem['omega']==report['omega']==case['omega'],'angular frequency')
    check(cfg['snapshot_tol']==1e-7 and cfg['gmres_tol']==1e-6,'unchanged tolerances')
    check(cfg['num_layers']==3 and cfg['layer_tail_tolerances']==[.003,.008,.02],'three-layer settings')
    train=problem['train_params']; query=problem['test_params']
    check(len(train)==8 and query==[.237,.317,.483],'training / held-out design')
    check(all(close(mu,.2+i*.4/7) for i,mu in enumerate(train)),'eight fixed evenly spaced training parameters')
    spacing=max(length/(count+1) for length,count in zip(problem['domain_lengths'],problem['Nphys']))
    shortest_wavelength=2*math.pi/(problem['omega']*math.sqrt(problem['background_speed']**-2+max(train)))
    check(close(report['ppw'],shortest_wavelength/spacing),'reported PPW / input geometry')
    check(all(all(abs(q-t)>1e-12 for t in train) for q in query),'queries independent of training')
    check([q['mu'] for q in report['tests']]==query,'all three queries retained')
    check(not report['training_resumed'],'fresh timing without checkpoint resume')
    snapshots=re.findall(r'^snapshot mu=([\d.]+) it=(\d+) true=([\deE.+-]+)',log,re.M)
    check(len(snapshots)==40,'40 distinct training inverse solves (shared first layer)')
    check(all(float(r)<=1e-7 for _,_,r in snapshots),'logged snapshot tolerance')
    for mu in train:
        check(sum(abs(float(m)-mu)<5.1e-7 for m,_,_ in snapshots)==5,f'five inverse solves for training mu={mu}')
    check(0<=report['snapshot_max_relres']<=1e-7,'full precision snapshot tolerance')
    check(report['all_converged'],'report numerical acceptance')
    for q in report['tests']:
        for method in ('shifted','global','dd'):
            r=q[method]; prefix=f"mu={q['mu']}, {method}"
            check(r['flag']==0 and 0<=r['true_residual']<=1e-6,prefix+' convergence')
            check(r['seconds']>0 and close(r['seconds'],r['setup_seconds']+r['solve_seconds']+r.get('rom_seconds',0)),prefix+' timing sum')
            matches=[row for row in rows if row['method']==method and close(row['mu'],q['mu'])]
            check(len(matches)==1,prefix+' unique CSV record')
            if len(matches)!=1: continue
            row=matches[0]
            for key,field in [('true_residual','final_relres'),('seconds','total_s'),('iterations','iterations'),('flag','flag')]:
                check(close(r[key],row[field]),prefix+' CSV '+key)
            if method in ('global','dd'):
                for field in ('basis_MiB','persistent_MiB'):
                    check(close(report[method+'_'+field],row[field]),prefix+' CSV '+field)
                check(close(row['speedup_vs_shifted'],q['shifted']['seconds']/r['seconds']),prefix+' CSV speedup')
    check(close(report['basis_saving_percent'],100*(1-report['dd_basis_MiB']/report['global_basis_MiB'])),'basis saving calculation')
    check(all(report[m+'_persistent_MiB']>=report[m+'_basis_MiB'] for m in ('global','dd')),'total storage includes basis')
    result.update(accepted=not errors,errors=errors,N=n,omega=report['omega'],
        snapshot_count=len(snapshots),online_count=len(report['tests'])*3,
        node=report['metadata']['node'],sources={str(p.resolve()):digest(p) for p in
            (report_path,configuration_path,csv_path,log_path)})
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/scnet_3d_20260909')
    parser.add_argument('--require-uniform-complete',action='store_true')
    args=parser.parse_args(); root=args.root
    plan=json.loads((root/'expected_cases.json').read_text())
    extras=json.loads((root/'extra_runs.json').read_text())
    with (root/'accounting.psv').open() as stream:
        jobs={row['JobID']:row for row in csv.DictReader(stream,delimiter='|')}
    cases=[dict(c,version=plan['version'],job_id=f"{plan['array_id']}_{c['array_index']}") for c in plan['cases']]
    cases.extend(c for c in extras if c['kind']=='full')
    results=[verify(c,root,jobs,plan['array_id']) for c in cases]
    matched=[c for c in extras if c['kind']=='full' and c.get('profile')]
    required={str(c['job_id']) for c in matched}
    expected_design={('waveguide',24),('waveguide',36),('waveguide',48),('halfspace',48)}
    design_complete=(len(matched)==len(expected_design) and
                     {(c['problem'],c['omega']) for c in matched}==expected_design)
    uniform_complete=design_complete and all(r['accepted'] for r in results if r['job_id'] in required)
    invalid=[r['job_id'] for r in results if r['complete_report'] and not r['accepted']]
    output=dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Saved-evidence cross-check; A*x residuals computed during MATLAB runs, not recomputed here.',
        uniform_complete=uniform_complete,uniform_design_complete=design_complete,
        expected_uniform_design=sorted(expected_design),
        invalid_complete_reports=invalid,cases=results,
        accounting_sha256=digest(root/'accounting.psv'))
    path=root/'verification.json'; path.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(accepted=sum(r['accepted'] for r in results),uniform_complete=uniform_complete,invalid=invalid)))
    for r in results:
        if r.get('errors'): print(r['job_id'],r['errors'])
    if invalid or (args.require_uniform_complete and not uniform_complete): raise SystemExit(1)


if __name__=='__main__': main()
