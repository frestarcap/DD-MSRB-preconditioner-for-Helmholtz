#!/usr/bin/env python3
"""Collect a submitted study and draw each completed case during this run.

Does not submit, cancel, or alter Slurm jobs. --watch polls until the array
leaves the queue; numerical acceptance remains a separate report field.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/scnet_3d_20260909')
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--interval',type=int,default=60)
    parser.add_argument('--control-path',help='Optional task-local SSH multiplex socket')
    args=parser.parse_args(); root=args.root.resolve()
    scripts=Path(__file__).resolve().parent
    plan=json.loads((root/'expected_cases.json').read_text())
    array=str(int(plan['array_id'])); remote=plan['remote']
    destination=root/('remote_'+plan['version'])
    connection=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15']
    environment=os.environ.copy()
    if args.control_path:
        connection+=['-o','ControlMaster=auto','-o','ControlPersist=600','-S',args.control_path]
        environment['RSYNC_RSH']=shlex.join(connection)
    previous={}; drawn={}
    while True:
        try:
            extra_file=root/'extra_runs.json'
            extras=json.loads(extra_file.read_text()) if extra_file.exists() else []
            job_ids=[array]+[str(int(e['job_id'])) for e in extras]
            def ssh(command):
                return subprocess.run(connection+['<cluster>',command],text=True,
                                      capture_output=True,timeout=45,check=True).stdout
            accounting=ssh(f"sacct -j {','.join(job_ids)} --format=JobID,JobIDRaw,JobName,State,ExitCode,Elapsed,Timelimit,AllocCPUS,ReqMem,MaxRSS,TotalCPU,NodeList -P")
            rows={r['JobID']:r for r in csv.DictReader(io.StringIO(accounting),delimiter='|')}
            temporary=root/'accounting.tmp'; temporary.write_text(accounting)
            temporary.replace(root/'accounting.psv')
            roots={remote:destination}
            for e in extras:
                if rows.get(str(e['job_id']),{}).get('State','PENDING')!='PENDING':
                    roots[e['remote']]=root/('remote_'+e['version'])
            for remote_root,local_root in roots.items():
                fetched=subprocess.run(['bash',str(scripts/'fetch_scnet_3d.sh'),remote_root,str(local_root)],
                    timeout=120,capture_output=True,text=True,env=environment)
                if fetched.returncode:
                    print('COLLECTION_INCOMPLETE '+remote_root,flush=True)
            logfiles=[(destination/'logs'/f"helm3d_sweep_{array}_{c['array_index']}.out",c['tag']) for c in plan['cases']]
            for e in extras:
                logfiles.extend((p,e['tag']) for p in (root/('remote_'+e['version'])/'logs').glob(f"*_{e['job_id']}.out"))
            for logfile,label in logfiles:
                if logfile.exists():
                    log=logfile.read_text(errors='replace')
                    updates=[line for line in log.splitlines() if line.startswith((
                        'CHECKPOINT','GLOBAL offline','DD offline','--- Online',
                        'shifted-MG:','GLOBAL TA-ROM:','DD TA-ROM:',
                        'SCNET_3D_EXPERIMENT_PASS','Error using','snapshot mu=','Snapshot mu=',
                        'SOLVER_PROBE','SCNET_3D_SOLVER_PROBE_COMPLETE'))]
                    count=previous.get(str(logfile),max(0,len(updates)-3))
                    if len(updates)>count:
                        print(f"[{label}]\n"+'\n'.join(updates[count:]),flush=True)
                    previous[str(logfile)]=len(updates)
            changed=False
            for report in root.glob('remote_*/full_*/report.json'):
                digest=hashlib.sha256(report.read_bytes()).hexdigest()
                if drawn.get(str(report))!=digest:
                    data=json.loads(report.read_text())
                    problem='halfspace' if data['problem_name']=='affine_halfspace' else 'waveguide'
                    subprocess.run([sys.executable,str(scripts/'plot_scnet_3d.py'),
                                    '--root',str(root),'--problem',problem],check=True,timeout=120)
                    drawn[str(report)]=digest; changed=True
                    compact={key:data[key] for key in ('omega','N','all_converged',
                             'basis_saving_percent','global_persistent_MiB','dd_persistent_MiB')}
                    print('REPORT '+json.dumps(compact),flush=True)
            for e in extras:
                if e['kind']!='probe': continue
                report=root/('remote_'+e['version'])/e['tag']/'probe.json'
                if not report.exists(): continue
                digest=hashlib.sha256(report.read_bytes()).hexdigest()
                if drawn.get(str(report))==digest: continue
                subprocess.run([sys.executable,str(scripts/'plot_scnet_3d_solver_probe.py'),
                    '--report',str(report),'--output',str(root/'figures/solver_probes'/e['tag'])],check=True,timeout=120)
                drawn[str(report)]=digest; changed=True
                d=json.loads(report.read_text())
                print(f"SOLVER_PROBE_REPORT job={e['job_id']} passed={d['passing_count']}",flush=True)
            subprocess.run([sys.executable,str(scripts/'summarize_scnet_3d.py'),'--root',str(root)],
                           check=True,timeout=30,stdout=subprocess.DEVNULL)
            terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY',
                      'NODE_FAIL','BOOT_FAIL','DEADLINE','PREEMPTED','REVOKED'}
            done=all(rows.get(f"{array}_{case['array_index']}",{}).get('State','').partition(' ')[0] in terminal
                     for case in plan['cases'])
            done=done and all(rows.get(str(e['job_id']),{}).get('State','').partition(' ')[0] in terminal for e in extras)
            if done:
                print('ARRAY_FINISHED: inspect accounting and numerical acceptance.',flush=True)
                break
            individual=[f"{array}_{case['array_index']}" for case in plan['cases']]+[str(e['job_id']) for e in extras]
            active=[job for job in individual if rows.get(job,{}).get('State','').partition(' ')[0] not in terminal]
            queue=ssh(f"squeue --noheader -j {','.join(active)} -o \"%.22i %.9T %.10M %.10l %.45R\"")
            signature=[tuple(line.split()[:2]+line.split()[3:]) for line in queue.splitlines()]
            if changed or previous.get('queue_signature')!=signature:
                print(queue.strip(),flush=True); previous['queue_signature']=signature
        except (subprocess.SubprocessError,ValueError,OSError) as error:
            print('COLLECTION_RETRY '+str(error),flush=True)
        if not args.watch:
            break
        time.sleep(max(15,args.interval))


if __name__=='__main__':
    main()
