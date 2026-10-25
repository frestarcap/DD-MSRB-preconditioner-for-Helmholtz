#!/usr/bin/env python3
"""Plot measured 3D Slurm results; no interpolated or synthetic data points."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'svg.fonttype':'none'})
COLORS={'shifted':'#667085','global':'#2875AD','dd':'#C3522A'}
LABELS={'shifted':'Shifted MG','global':'Global ROM + MG','dd':'DD ROM + MG'}


def read_report(path):
    d=json.loads(path.read_text())
    assert d['N']==int(d['N'])
    d['N']=int(d['N'])
    if 'solver_config' not in d:
        cfg=json.loads((path.parent/'configuration.json').read_text())['cfg']
        d['solver_config']={key:cfg[key] for key in (
            'shift_beta','snapshot_restart','gmres_restart',
            'snapshot_maxit','gmres_maxit','mg_coarse_max_dofs')}
    return d


def speedup(query, method):
    a,b=query['shifted'],query[method]
    if a['true_residual']>1e-6 or b['true_residual']>1e-6:
        return float('nan')
    return a['seconds']/b['seconds']


def save(fig, folder, name):
    for suffix in ('png','pdf','svg'):
        fig.savefig(folder/f'{name}.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)


def solver_caption(data):
    groups={}
    for d in data:
        s=d['solver_config']
        key=(s['shift_beta'],s['snapshot_restart'],s['gmres_restart'],s['mg_coarse_max_dofs'],d['metadata']['node'])
        groups.setdefault(key,[]).append(f"{d['omega']:g}")
    return '\n'.join(f"{node}: β={b:g}, restart={sr}/{orr}, coarse≤{c}; ω="+', '.join(frequencies)
                     for (b,sr,orr,c,node),frequencies in groups.items())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/scnet_3d_20260909')
    parser.add_argument('--problem',choices=('waveguide','halfspace'),default='waveguide')
    parser.add_argument('--profile',type=Path,help='Only reports with these exact solver settings; all reports remain in the default figures and CSV.')
    parser.add_argument('--node',help='Optional exact compute-node match for a controlled timing comparison')
    parser.add_argument('--output',type=Path,help='Optional destination for a selected comparison')
    args=parser.parse_args(); root=args.root
    out=args.output or (root/'figures' if args.problem=='waveguide' else root/'figures/halfspace')
    out.mkdir(exist_ok=True,parents=True)
    problem_title='Gaussian waveguide' if args.problem=='waveguide' else 'half-space'
    scaling=list(root.glob('**/scaling.json'))
    if scaling:
        s=json.loads(scaling[-1].read_text()); rows=s['rows']
        calibration_logs=list((scaling[-1].parent.parent/'logs').glob('*scale*.out'))
        calibration_node=calibration_logs[0].read_text().splitlines()[0] if len(calibration_logs)==1 else None
        wall=np.array([r['wall_seconds'] for r in rows])
        labels=[f"{r['workers']} × {r['threads']}\n"+('BLAS' if r.get('backend')=='cgs2_mex' else 'MATLAB') for r in rows]
        fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
        for ax,values,ylabel in zip(axes,[wall,wall[0]/wall],['Eight-task wall time (s)','Throughput relative to 1 worker']):
            bars=ax.bar(labels,values,color=['#C3522A' if i==int(wall.argmin()) else '#2875AD' for i in range(len(rows))])
            ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
            ax.set(xlabel='Workers × configured threads / orthogonalization',ylabel=ylabel)
            ax.set_ylim(0,max(values)*1.2); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
        fig.suptitle(f"3D parameter parallelism on a {s['allocation']}-CPU allocation")
        location=f' Node {calibration_node}.' if calibration_node else ''
        fig.text(.5,-.04,f"Same eight parameters; ω = {s['omega']:g}.{location} Pool startup excluded. One timing per layout.",ha='center',fontsize=9)
        save(fig,out,'parallel_scaling')
    paths=sorted(root.glob('**/full_*/report.json'))
    # One report per completed run; never silently merge replicate timings.
    data=sorted([read_report(p) for p in paths],key=lambda d:(d['omega'],d['metadata']['job_id']))
    name='gaussian_waveguide' if args.problem=='waveguide' else 'affine_halfspace'
    data=[d for d in data if d.get('problem_name','gaussian_waveguide')==name]
    if args.profile:
        profile=json.loads(args.profile.read_text())
        data=[d for d in data if all(d['solver_config'].get(k)==v for k,v in profile.items())]
    if args.node:
        data=[d for d in data if d['metadata']['node']==args.node]
    if not data:
        print('No full-frequency reports yet; no frequency figures fabricated.')
        return
    repeated={d['omega'] for d in data if sum(q['omega']==d['omega'] for q in data)>1}
    x=np.arange(len(data)); tick=[f"{d['omega']:g}{'' if d['all_converged'] else '†'}\n{d['N']/1e6:.2f}M"+
        (f"\nβ={d['solver_config']['shift_beta']:g}" if d['omega'] in repeated else '') for d in data]
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    ax=axes[0,0]
    for method,offset in [('global',-.18),('dd',.18)]:
        total=np.array([d[f'{method}_persistent_MiB'] for d in data]); basis=np.array([d[f'{method}_basis_MiB'] for d in data])
        ax.bar(x+offset,total,.34,color=COLORS[method],alpha=.3)
        ax.bar(x+offset,basis,.34,color=COLORS[method],label=method.upper()+' basis')
    ax.set(ylabel='Persistent ROM storage (MiB)',title='(a) Basis and total storage'); ax.legend()
    ax.set_ylim(0,1.22*max(d[f'{method}_persistent_MiB'] for d in data for method in ('global','dd')))
    ax.text(.02,.98,'Pale extension: operators + partition indices',transform=ax.transAxes,va='top',fontsize=8)
    for i,d in enumerate(data):
        persistent_saving=100*(1-d['dd_persistent_MiB']/d['global_persistent_MiB'])
        height=max(d['global_persistent_MiB'],d['dd_persistent_MiB'])
        ax.annotate(f"DD basis saving {d['basis_saving_percent']:.1f}%\nTotal saving {persistent_saving:.1f}%",
                    (i,height),xytext=(0,5),textcoords='offset points',ha='center',va='bottom',fontsize=8)
    ax=axes[0,1]
    for method in ('global','dd'):
        ratios=[np.array([speedup(q,method) for q in d['tests']]) for d in data]
        med=np.array([np.median(a) for a in ratios]); low=np.array([min(a) for a in ratios]); high=np.array([max(a) for a in ratios])
        ax.errorbar(x,med,yerr=[med-low,high-med],fmt='o-',capsize=4,color=COLORS[method],label=method.upper())
    ax.axhline(1,color='gray',ls='--',lw=1); ax.set(ylabel='Speedup relative to shifted MG',title='(b) Online solve: median and query range'); ax.legend()
    ax.set_yscale('log'); ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax=axes[1,0]
    for method in COLORS:
        y=[max(q[method]['true_residual'] for q in d['tests']) for d in data]
        ax.semilogy(x,y,'o-',color=COLORS[method],label=LABELS[method])
    ax.axhline(1e-6,color='black',ls='--',lw=1,label='Acceptance 10⁻⁶'); ax.legend(fontsize=8)
    ax.set(ylabel='Maximum recomputed true residual',title='(c) All held-out queries')
    residual_max=max(q[method]['true_residual'] for d in data for q in d['tests'] for method in COLORS)
    ax.set_ylim(1e-8,max(1e-5,2*residual_max))
    ax=axes[1,1]
    for method in COLORS:
        y=[np.mean([q[method]['iterations'] for q in d['tests']]) for d in data]
        ax.plot(x,y,'o-',color=COLORS[method],label=LABELS[method])
    ax.set(ylabel='Mean completion iterations',title='(d) Completion cost after three ROM steps'); ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    for ax in axes.flat:
        ax.set_xticks(x,tick); ax.set_xlabel('Angular frequency ω / unknowns'); ax.grid(alpha=.2); ax.set_axisbelow(True)
    fig.suptitle(f'3D {problem_title}: fixed minimum 9.93 points per wavelength')
    fig.text(.5,-.025,'Online: MG setup + ROM + completion; common matrix assembly and offline training excluded.\n'+solver_caption(data),
             ha='center',va='top',fontsize=8)
    if not all(d['all_converged'] for d in data):
        fig.text(.5,-.10,'† Some queries did not converge; speedup omitted when either method failed.',ha='center',fontsize=8)
    save(fig,out,'frequency_summary')

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    markers=['o','s','D','^','v']
    for i,d in enumerate(data):
        for method in ('global','dd'):
            mu=[q['mu'] for q in d['tests']]
            label=f"{method.upper()}, ω = {d['omega']:g}"
            if d['omega'] in repeated: label+=f", β={d['solver_config']['shift_beta']:g}"
            for ax,values in zip(axes,[[q[method]['rom_residual'] for q in d['tests']],
                                      [speedup(q,method) for q in d['tests']]]):
                ax.scatter(mu,values,color=COLORS[method],marker=markers[i%len(markers)],label=label,s=42)
    for ax in axes:
        ax.set(xlim=(.19,.61),xlabel='Held-out contrast parameter μ',yscale='log')
        ax.plot(np.linspace(.2,.6,8),np.full(8,.02),'|',color='#999999',
                transform=ax.get_xaxis_transform(),ms=9,clip_on=False)
        ax.grid(alpha=.2); ax.legend(fontsize=8,ncol=2)
    axes[0].set(ylabel='True residual after three ROM steps',title='(a) Reduced inverse-action quality')
    axes[0].axhline(1e-6,ls='--',color='black',lw=1)
    axes[1].set(ylabel='Online speedup relative to shifted MG',title='(b) Measured parameter dependence')
    axes[1].axhline(1,ls='--',color='gray',lw=1)
    fig.suptitle(f'3D {problem_title}: all measured queries')
    fig.text(.5,-.03,'Gray ticks mark the eight training parameters. One online timing per query and method.',ha='center',fontsize=9)
    fig.text(.5,-.075,'Online: MG setup + ROM + completion; common matrix assembly and offline training excluded.\n'+solver_caption(data),
             ha='center',va='top',fontsize=8)
    save(fig,out,'parameter_queries')

    d=data[-1]; q=min(d['tests'],key=lambda q:abs(q['mu']-.317))
    fig,ax=plt.subplots(figsize=(7.5,4.7),layout='constrained')
    for method in COLORS:
        r=q[method]; history=np.atleast_1d(r['history'])
        if method!='shifted': history=np.r_[np.atleast_1d(r['rom_history']),history[1:]]
        ax.semilogy(np.arange(len(history)),history,color=COLORS[method],label=LABELS[method],lw=1.8)
        ax.semilogy(len(history)-1,r['true_residual'],'o',color=COLORS[method],ms=5)
    ax.axhline(1e-6,color='black',ls='--',lw=1); ax.axvline(3,color='gray',ls=':',lw=1)
    ax.set(xlabel='ROM steps + completion iterations',ylabel='Relative residual',
           title=f"ω = {d['omega']:g}, N = {d['N']:,}, μ = {q['mu']:.3f}")
    ax.legend(); ax.grid(alpha=.2)
    fig.text(.5,-.02,'Krylov estimates between residual checks; final markers independently recomputed.',ha='center',fontsize=8)
    save(fig,out,'high_frequency_convergence')

    fig,axes=plt.subplots(len(data),1,figsize=(10,3.1*len(data)),squeeze=False,layout='constrained')
    for ax,d in zip(axes.flat,data):
        s=d['field_slice']; field=np.array(s['real']); peak=max(np.max(np.abs(field)),1e-30)
        image=ax.pcolormesh(s['x'],s['y'],field.T/peak,cmap='RdBu_r',vmin=-1,vmax=1,rasterized=True,shading='auto')
        for a in (0,3 if args.problem=='waveguide' else 1): ax.axvline(a,ls='--',color='black',lw=.7)
        if args.problem=='halfspace': ax.axvline(.5,ls=':',color='black',lw=.8)
        for a in (0,1): ax.axhline(a,ls='--',color='black',lw=.7)
        ax.set_aspect('equal'); ax.set(xlabel='x',ylabel='y',title=f"ω = {d['omega']:g}, N = {d['N']:,}, μ = {s['mu']:.3f}, z = {s['z']:.3f}\n"
            f"Job {d['metadata']['job_id']}; β = {d['solver_config']['shift_beta']:g}")
        fig.colorbar(image,ax=ax,label='Re(u) / slice maximum')
    fig.suptitle('3D DD-ROM solution: midplane slices (dashed lines mark physical domain)')
    save(fig,out,'wavefield_slices')

    # Embed the measured XY slice in the physical 3D domain. No unmeasured
    # out-of-plane values are inferred from this single exported slice.
    d=data[-1]; s=d['field_slice']; field=np.array(s['real'])
    peak=max(np.max(np.abs(field)),1e-30)
    xx,yy=np.meshgrid(s['x'],s['y'],indexing='ij')
    zz=np.full_like(xx,s['z']); length=3 if args.problem=='waveguide' else 1
    fig=plt.figure(figsize=(10,5.7),layout='constrained')
    ax=fig.add_subplot(111,projection='3d',computed_zorder=False)
    norm=matplotlib.colors.SymLogNorm(linthresh=.02,vmin=-1,vmax=1,base=10)
    color_label='Re(u) / slice maximum\nSymmetric log color scale'
    cmap=plt.get_cmap('RdBu_r')
    ax.plot_surface(xx,yy,zz,facecolors=cmap(norm(field/peak)),
                    rstride=1,cstride=1,shade=False,antialiased=False,rasterized=True,zorder=2)
    for z in (0,1):
        ax.plot([0,length,length,0,0],[0,0,1,1,0],[z]*5,color='#555555',lw=.8)
    for x0 in (0,length):
        for y0 in (0,1): ax.plot([x0,x0],[y0,y0],[0,1],color='#555555',lw=.8)
    if args.problem=='halfspace':
        ax.plot([.5]*5,[0,1,1,0,0],[0,0,1,1,0],color='#A47521',lw=1.2,ls='--',label='Material interface x = 0.5')
    ax.scatter([.2],[.5],[.5],s=35,c='black',marker='*',depthshade=False,label='Source center',zorder=5)
    ax.set(xlabel='x',ylabel='y',zlabel='z',zlim=(0,1),
           title=f"3D {problem_title}, ω = {d['omega']:g}, N = {d['N']:,}\nMeasured DD-ROM slice at z = {s['z']:.3f}; μ = {s['mu']:.3f}")
    ax.set_box_aspect((length+.35,1.35,1)); ax.view_init(elev=27,azim=-62)
    ax.set_xlabel('x',labelpad=10); ax.set_ylabel('y',labelpad=8)
    ax.set_zlabel('z',labelpad=4)
    ax.legend(loc='upper left',fontsize=9)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis): axis.pane.fill=False
    scalar=matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap)
    fig.colorbar(scalar,ax=ax,shrink=.65,pad=.14,label=color_label)
    fig.text(.5,-.08,'Wireframe: physical domain. Colored slice includes the PML extension.\n'
        f"Job {d['metadata']['job_id']}; node {d['metadata']['node']}; β = {d['solver_config']['shift_beta']:g}.",ha='center',fontsize=9)
    save(fig,out,'wavefield_3d_slice')
    with (out/'frequency_metrics.csv').open('w',newline='') as f:
        solver_keys=['shift_beta','snapshot_restart','gmres_restart','snapshot_maxit','gmres_maxit','mg_coarse_max_dofs']
        writer=csv.writer(f); writer.writerow(['omega','N','basis_saving_percent','persistent_saving_percent','method','mu','iterations','true_residual','online_seconds','speedup','job_id','node','all_converged']+solver_keys)
        for d in data:
            for q in d['tests']:
                for method in COLORS:
                    r=q[method]; writer.writerow([d['omega'],d['N'],d['basis_saving_percent'],100*(1-d['dd_persistent_MiB']/d['global_persistent_MiB']),method,q['mu'],r['iterations'],r['true_residual'],r['seconds'],speedup(q,method),d['metadata']['job_id'],d['metadata']['node'],d['all_converged']]+[d['solver_config'][key] for key in solver_keys])
    print(out)


if __name__=='__main__': main()
