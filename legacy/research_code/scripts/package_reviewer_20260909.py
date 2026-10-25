#!/usr/bin/env python3
"""Make local reviewer PDFs and source/evidence archives; never uploads."""
from pathlib import Path
import csv, hashlib, html, io, json, re, shutil, sys, tarfile, zipfile
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reviewer_20260909'
DEL=OUT/'delivery';DEL.mkdir(exist_ok=True)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()
md=ROOT/'md/reviewer_response_20260909.md'
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='ResponseBody',fontName='Times-Roman',fontSize=10.5,leading=14,spaceAfter=7))
styles.add(ParagraphStyle(name='BodyBeforeLocation',parent=styles['ResponseBody'],keepWithNext=True))
styles.add(ParagraphStyle(name='ResponseTitle',fontName='Times-Bold',fontSize=17,leading=21,alignment=TA_CENTER,spaceAfter=16))
styles.add(ParagraphStyle(name='ResponseHeading',fontName='Times-Bold',fontSize=11.5,leading=14,spaceBefore=10,spaceAfter=6,keepWithNext=True))
def markup(t):
    t=html.escape(t).replace('  \n','<br/>').replace('\n',' ')
    t=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',t)
    return re.sub(r'\*(.*?)\*',r'<i>\1</i>',t)
flow=[]
blocks=md.read_text().strip().split('\n\n')
for i,block in enumerate(blocks):
    if block.startswith('# '):flow.append(Paragraph(markup(block[2:]),styles['ResponseTitle']))
    elif block.startswith('## '):flow.append(Paragraph(markup(block[3:]),styles['ResponseHeading']))
    else:
        before_location=i+1<len(blocks) and blocks[i+1].startswith('**Location:**')
        flow.append(Paragraph(markup(block),styles['BodyBeforeLocation' if before_location else 'ResponseBody']))
def footer(c,doc):
    c.setFont('Times-Roman',9);c.setFillColor(colors.HexColor('#444444'))
    c.drawString(48,28,'Response to the reviewer | JSC | September 2026')
    c.drawRightString(A4[0]-48,28,str(doc.page))
pdf=DEL/'Response_to_reviewer.pdf'
SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=44,title='Response to the reviewer',author='An Ping and Zhichao Peng').build(flow,onFirstPage=footer,onLaterPages=footer)
shutil.copy2(ROOT/'LaTeX_DL_468198_240419/template.pdf',DEL/'Revised_manuscript.pdf')
if '--pdf-only' in sys.argv:
    print('PDF previews:',DEL)
    raise SystemExit(0)
files=set()
for name in ['code','scripts']:
    for p in (ROOT/name).rglob('*'):
        if p.is_file() and p.suffix in {'.m','.py','.sh','.md'} and 'archive_legacy' not in p.parts:
            files.add(p)
texdir=ROOT/'LaTeX_DL_468198_240419'
for p in texdir.iterdir():
    if p.is_file() and p.suffix in {'.tex','.bib','.cls','.clo','.sty','.bst','.def','.cfg','.fd'}:files.add(p)
for p in tuple(files):
    if p.suffix!='.tex':continue
    for ref in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',p.read_text()):
        candidates=[texdir/ref,*[(texdir/ref).with_suffix(s) for s in ['.pdf','.png','.jpg']]]
        existing=[x for x in candidates if x.is_file()]
        if existing:files.add(existing[0])
for rel in ['AGENTS.md','md/reviewer_20260909_reproduction.md','md/reviewer_response_20260909.md','md/reviewer_revision_20260909_zh.md','md/jsc_major_revision_reproduction.md','md/revised_paper_roadmap.md','agent_runs/2026-09-09_reviewer_revision.md']:
    files.add(ROOT/rel)
for suffix in ['*.csv','*.json']:
    for p in OUT.rglob(suffix):
        if not set(p.relative_to(OUT).parts)&{'smoke','before','delivery','tooling_venv'} and 'smoke' not in p.name and '_scale0.25' not in p.name:files.add(p)
files.add(OUT/'reviewer_report.txt')
manifest=[]
archive=DEL/'JSC_20260909_source_evidence.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for p in sorted(files):
        rel=str(p.relative_to(ROOT));z.write(p,rel)
        manifest.append([rel,p.stat().st_size,digest(p)])
    f=io.StringIO();w=csv.writer(f);w.writerow(['path','bytes','sha256']);w.writerows(manifest)
    z.writestr('MANIFEST.csv',f.getvalue())
datafiles=[p for p in OUT.rglob('*.mat') if not set(p.relative_to(OUT).parts)&{'smoke','before','delivery','tooling_venv'} and '_scale0.25' not in p.name]
manifest=[]
archive=DEL/'JSC_20260909_supplementary_numerical_data.tar'
with tarfile.open(archive,'w') as tar:
    for p in sorted(datafiles):
        rel=str(p.relative_to(ROOT));tar.add(p,arcname=rel,recursive=False)
        manifest.append([rel,p.stat().st_size,digest(p)])
    f=io.StringIO();w=csv.writer(f);w.writerow(['path','bytes','sha256']);w.writerows(manifest)
    b=f.getvalue().encode();info=tarfile.TarInfo('DATA_MANIFEST.csv');info.size=len(b);tar.addfile(info,io.BytesIO(b))
(DEL/'DATA_MANIFEST.csv').write_text(f.getvalue())
(DEL/'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(DEL.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt'))
print('Local delivery:',DEL)
