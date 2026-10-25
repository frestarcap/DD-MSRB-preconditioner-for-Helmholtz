"""Run paired fresh-process RSS measurements; no training in measured processes."""
import csv
import pathlib
import re
import subprocess
root = pathlib.Path(__file__).resolve().parents[1]
out = root / 'outputs/paper_memory_benchmark'
rows = []
if (out/'process_memory.csv').exists():
    with (out/'process_memory.csv').open() as stream:
        rows=list(csv.DictReader(stream))
for problem in ('halfspace', 'ecc', 'waveguide'):
    for rep in range(1, 4):
        for method in (('global','dd') if rep % 2 else ('dd','global')):
            if any(r['problem']==problem and r['method']==method and int(r['repetition'])==rep for r in rows):
                continue
            stem = f'{problem}_{method}_run{rep}'
            command = ['/usr/bin/time','-l','/Applications/MATLAB_R2025a.app/bin/matlab',
                       '-singleCompThread','-batch',
                       f"addpath('code'); run_paper_memory_case('{problem}','{method}',{rep});"]
            with (out / (stem+'.log')).open('w') as log:
                status = subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            content = (out / (stem+'.log')).read_text()
            if status.returncode:
                raise RuntimeError(stem+' failed:\n'+content[-2500:])
            peak = int(re.search(r'(\d+)\s+maximum resident set size',content).group(1))
            model = int(re.search(r'MODEL_BYTES (\d+)',content).group(1))
            rows.append(dict(problem=problem,method=method,repetition=rep,
                             peak_RSS_MiB=peak/2**20,model_MiB=model/2**20))
            with (out/'process_memory.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
                writer.writeheader(); writer.writerows(rows)
            print(stem, 'peak RSS MiB',round(peak/2**20,3),flush=True)
