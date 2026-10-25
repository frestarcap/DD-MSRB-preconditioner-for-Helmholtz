#!/usr/bin/env python3
"""Build fixed reviewer source/evidence and separate numerical-data archives.

Raw artifacts remain ignored. The packages are local deliverables; this script
never uploads or changes repository visibility. Run after both branches commit.
"""
from pathlib import Path
import csv, hashlib, io, json, subprocess, tarfile, zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/revision_20260907'
A=OUT/'route_a'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def git(cwd,*args):return subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
versions={name:git(folder,'rev-parse','HEAD') for name,folder in [('route_b',ROOT),('route_a',A)]}
versions['checkpoint']='5668997'
source_files=[]
for name,folder in [('route_b',ROOT),('route_a',A)]:
 for rel in git(folder,'ls-files').splitlines():
  p=folder/rel
  if not p.is_file() or '/archive_legacy/' in rel:continue
  if rel.startswith(('code/','LaTeX_DL_468198_240419/','scripts/')) or rel in [
   'md/jsc_major_revision_reproduction.md','agent_runs/2026-09-07_jsc_point_by_point_response.md']:
   source_files.append((p,f'{name}/{rel}'))
# Repeated samples and complete summary/diagnostic rows are in the small package.
for p in OUT.rglob('*.csv'):
 if any(x in p.parts for x in ['route_a','python_dependencies']):continue
 source_files.append((p,str(p.relative_to(ROOT))))
for p in (OUT/'deployment').glob('*_process.log'):source_files.append((p,str(p.relative_to(ROOT))))
for folder in ['halfspace','pc']:
 for p in (OUT/folder).glob('*_timings.mat'):source_files.append((p,str(p.relative_to(ROOT))))
for p in (ROOT/'outputs/helmholtz_3d_candidates/large').glob('*waveguide*frozen_holdout_20260905.csv'):
 source_files.append((p,str(p.relative_to(ROOT))))
source_path=OUT/'JSC_reviewer_source_evidence.zip'
manifest=[]
with zipfile.ZipFile(source_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
 z.writestr('VERSIONS.json',json.dumps(versions,indent=2)+'\n')
 for p,arc in source_files:
  z.write(p,arc);manifest.append([arc,p.stat().st_size,digest(p)])
 f=io.StringIO();w=csv.writer(f);w.writerow(['path','bytes','sha256']);w.writerows(manifest)
 z.writestr('SOURCE_MANIFEST.csv',f.getvalue())
 print('Source/evidence:',source_path,source_path.stat().st_size,flush=True)
# Retain complete verified PC alternatives, the half-space models, and primary
# legacy inputs. Duplicate unverified PC copies and Python dependencies are omitted.
files=set()
for folder in ['halfspace','deployment','audit']:
 files.update((OUT/folder).glob('*.mat'))
files.update((OUT/'pc').glob('*_verified.mat'))
files.update([OUT/'pc/paper_data.mat',OUT/'pc/screen.mat'])
for rel in [
 'outputs/halfspace_dd_paper/paper_data.mat',
 'outputs/halfspace_dd_refinement/review_repaired_20260905/global.mat',
 'outputs/halfspace_dd_refinement/review_repaired_20260905/slabs1-aligned-sym8.mat',
 'outputs/helm_trajectory_evidence/review_repaired_20260905/ecc_residual_ablation_paper.mat',
 'outputs/helm_trajectory_evidence/review_repaired_20260905/pc_residual_ablation_paper.mat',
 'outputs/helmholtz_3d_candidates/large/helmholtz_3d_candidates_jsc_20260907_lens_tight.mat',
 'outputs/helmholtz_3d_candidates/large/gaussian_lens_jsc_20260907_lens_tight_training.mat',
 'outputs/helmholtz_3d_candidates/large/helmholtz_3d_candidates_waveguide_dd_memory_saving_41pct_reproduction.mat',
 'outputs/helmholtz_3d_candidates/large/gaussian_waveguide_waveguide_dd_memory_saving_41pct_reproduction_training.mat']:
 files.add(ROOT/rel)
assert all(p.is_file() for p in files)
archive=OUT/'JSC_reviewer_numerical_data.tar'
manifest=[]
with tarfile.open(archive,'w') as tar:
 for p in sorted(files):
  rel=str(p.relative_to(ROOT));tar.add(p,arcname=rel,recursive=False)
  manifest.append([rel,p.stat().st_size,digest(p)])
 f=io.StringIO();w=csv.writer(f);w.writerow(['path','bytes','sha256']);w.writerows(manifest)
 payload=f.getvalue().encode();info=tarfile.TarInfo('DATA_MANIFEST.csv');info.size=len(payload)
 tar.addfile(info,io.BytesIO(payload))
(OUT/'DATA_MANIFEST.csv').write_text(f.getvalue())
checks=[]
for p in [source_path,archive,OUT/'JSC_route_A_three_experiments.pdf',OUT/'JSC_route_B_five_experiments.pdf']:
 checks.append(f'{digest(p)}  {p.name}')
(OUT/'DELIVERY_SHA256.txt').write_text('\n'.join(checks)+'\n')
(OUT/'VERSIONS.json').write_text(json.dumps(versions,indent=2)+'\n')
print('Numerical data:',archive,archive.stat().st_size,flush=True)
