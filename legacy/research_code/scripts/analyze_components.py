#!/usr/bin/env python3
"""Audit and summarize only completed full frozen-model diagnostic cohorts.

Plotting runtime: /opt/homebrew/Cellar/python@3.14/3.14.3_1/bin/python3.14.
The bundled Python can run --no-plots but has no matplotlib in this session.
Missing frequencies remain explicitly
missing. Smoke input is available only via --validate-smoke and is never
included in production tables/figures. Each execution writes a new timestamped
analysis directory, preserving earlier summaries and all source data.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
OMEGAS = [24, 36, 48, 60]
MUS = [.237, .317, .483]
VARIANTS = ['GG', 'GGG', 'DD', 'DDD', 'GGD', 'DDG']
COLORS = {'global': '#2364aa', 'dd': '#c95131'}
CHECKS = []


def array(value):
    return value if isinstance(value, list) else [value]


def read_records(path):
    value = json.loads(path.read_text())
    return value if isinstance(value, list) else [value]


def check(name, ok, error=0.0, tolerance=None):
    CHECKS.append(dict(check=name, passed=bool(ok), error=float(error), tolerance=tolerance))


def close(name, a, b, atol=2e-10, rtol=2e-10):
    delta = abs(a-b)
    allowed = atol + rtol*max(abs(a), abs(b))
    check(name, math.isfinite(delta) and delta <= allowed, delta, allowed)


def metric_validate(m, name):
    check(name+'.definition', m['definition_version']=='dd_component_metrics_v1')
    energy = m['energy_total']
    check(name+'.energy_finite', math.isfinite(energy) and energy>=0)
    groups = [('region', energy), ('profile_band', m['region_energy'][0]),
              ('axial_section', m['region_energy'][0]),
              ('source_region', m['region_energy'][0]),
              ('partition', energy), ('axial_fft', m['axial_fft_windowed_energy']),
              ('axial_fft_signed', m['axial_fft_windowed_energy'])]
    for prefix, total in groups:
        ek = prefix+('_power' if prefix.startswith('axial_fft') else '_energy')
        values = array(m[ek]); fractions = array(m[prefix+'_fraction'])
        check(name+'.'+prefix+'.length', len(values)==len(fractions))
        close(name+'.'+prefix+'.energy_sum', sum(values), total,
              atol=0, rtol=3e-10)
        close(name+'.'+prefix+'.fraction_sum', sum(fractions), 1 if total>0 else 0)
        for i, (value, frac) in enumerate(zip(values,fractions)):
            check(f'{name}.{prefix}[{i}].range', value>=0 and -1e-12<=frac<=1+1e-12)
            close(f'{name}.{prefix}[{i}].definition', frac, value/total if total else 0)
    close(name+'.roughness_transverse', m['roughness_transverse_over_omega_squared'],
          sum(m['roughness_over_omega_squared'][1:]))


def mg_validate(m, name):
    for field in ['relative_sum_vs_cycle_error','relative_sum_vs_Msolve_error',
                  'relative_cycle_vs_Msolve_error','relative_operator_linearity_error']:
        check(name+'.'+field, m[field]<1e-9, m[field], 1e-9)
    check(name+'.cycle', m['cycle_count']==1)
    for value in m['component_optimal_contraction']+[m['combined_optimal_contraction'],m['joint_span_contraction']]:
        check(name+'.optimal_contraction_range', -1e-10<=value<=1+1e-9)
    check(name+'.joint_not_worse_than_components',
          m['joint_span_contraction']<=min(m['component_optimal_contraction']+[m['combined_optimal_contraction']])+1e-7)


def flatten_metric(m, prefix):
    out = {prefix+'energy': m['energy_total']}
    for name,value in zip(m['region_names'],m['region_fraction']): out[prefix+name+'_fraction']=value
    out[prefix+'pml_fraction']=sum(m['region_fraction'][1:])
    out[prefix+'core_fraction_physical']=m['profile_band_fraction'][4]
    out[prefix+'core_fraction_total']=m['profile_band_fraction'][4]*m['region_fraction'][0]
    for i,value in enumerate(m['profile_band_fraction'],1): out[f'{prefix}profile_band_{i}_fraction_physical']=value
    for i,value in enumerate(m['axial_section_fraction'],1): out[f'{prefix}axial_section_{i}_fraction_physical']=value
    for name,value in zip(m['source_region_names'],m['source_region_fraction']): out[prefix+name+'_fraction_physical']=value
    for name,value in zip(['lt0p5','0p5_0p9','0p9_1p1','1p1_1p5','ge1p5'],m['axial_fft_fraction']):
        out[prefix+'fft_'+name+'_fraction_windowed']=value
    out[prefix+'roughness_axial_over_omega2']=m['roughness_axial_over_omega_squared']
    out[prefix+'roughness_transverse_over_omega2']=m['roughness_transverse_over_omega_squared']
    for name,value in zip(m['axial_fft_signed_names'],m['axial_fft_signed_fraction']):
        out[prefix+'fft_'+name+'_fraction_windowed']=value
    return out


def flatten_mg(m):
    out = {f'mg_{name}_optimal_contraction':value for name,value in
           zip(['pre','coarse','post'],m['component_optimal_contraction'])}
    for field in ['combined_optimal_contraction','combined_unit_step_contraction',
                  'joint_span_contraction','joint_span_rank']:
        out['mg_'+field]=m[field]
    return out


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def table(rows, columns, formats=None):
    formats=formats or {}
    text=['| '+' | '.join(label for key,label in columns)+' |',
          '| '+' | '.join('---' for _ in columns)+' |']
    for row in rows:
        values=[]
        for key,label in columns:
            value=row.get(key)
            if value is None: values.append('—')
            elif key in formats: values.append(formats[key](value))
            elif isinstance(value,float): values.append(f'{value:.4g}')
            else: values.append(str(value))
        text.append('| '+' | '.join(values)+' |')
    return '\n'.join(text)


def load_archive():
    path=BASE/'existing_audit/per_query_methods.csv'
    if not path.exists(): return {}
    return {(int(row['omega']),round(float(row['mu']),3),row['method']):row
            for row in csv.DictReader(path.open()) if row['method'] in COLORS}


def process_cohort(folder, production=True):
    meta=json.loads((folder/'complete.json').read_text())
    check(str(folder)+'.mode', meta['mode']==('full' if production else 'smoke'))
    check(str(folder)+'.validated', meta['validated'] is True)
    states=read_records(folder/'components.json')
    cross=read_records(folder/'cross_spaces.json')
    normalized=read_records(folder/'normalized.json')
    index={}
    flat=[]; norm_flat=[]; curves=[]; blocks=[]
    for row in states:
        key=(row['omega'],round(row['mu'],3),row['method'],row['depth'])
        name=f'w{key[0]} mu{key[1]} {key[2]} L{key[3]}'
        check(name+'.unique', key not in index); index[key]=row
        close(name+'.omega_source', row['omega'],meta['omega'],atol=0,rtol=0)
        for field in ['residual','field_error']:
            metric_validate(row[field],name+'.'+field)
            for block,e,f in zip(row[field]['partition_names'],row[field]['partition_energy'],row[field]['partition_fraction']):
                blocks.append(dict(omega=key[0],mu=key[1],method=key[2],depth=key[3],field=field,block=block,energy=e,fraction=f))
        mg_validate(row['mg'],name+'.mg')
        close(name+'.residual_norm_vs_MG_rhs',math.sqrt(row['residual']['energy_total']),row['mg']['rhs_norm'],atol=1e-14)
        for capture in ['error_capture_global','error_capture_dd']:
            check(name+'.'+capture+'.range',0<=row[capture]<=1+1e-12)
        check(name+'.replay',row['replay_relative_mismatch']<1e-8,row['replay_relative_mismatch'],1e-8)
        check(name+'.reference_residual',row['reference_residual']<2e-6,row['reference_residual'],2e-6)
        f={k:row[k] for k in ['omega','mu','method','depth','rho','relative_field_error',
                             'reference_residual','reference_disagreement_over_error',
                             'replay_relative_mismatch','error_capture_global','error_capture_dd']}
        f['reference_disagreement_exceeds_1pct']=row['reference_disagreement_over_error']>.01
        f.update(flatten_metric(row['residual'],'residual_'))
        f.update(flatten_metric(row['field_error'],'error_'))
        f.update(flatten_mg(row['mg'])); flat.append(f)
    crossindex={}
    for row in cross:
        key=(row['omega'],round(row['mu'],3),row['variant']);crossindex[key]=row
        metric_validate(row['residual'],f'{key} residual');mg_validate(row['mg'],f'{key} mg')
    for row in normalized:
        key=(row['omega'],round(row['mu'],3),row['variant'])
        cx=crossindex[key]; hist=row['history']
        metric_validate(row['after_residual'],f'{key} after')
        close(f'{key}.initial_unit',hist[0],1)
        close(f'{key}.true_final',hist[-1],row['true_relative_residual'])
        close(f'{key}.after_energy',row['after_residual']['energy_total'],row['true_relative_residual']**2)
        check(f'{key}.history_length',len(hist)==row['iteration_count']+1)
        if len(hist)>1:
            close(f'{key}.first_step_vs_independent_MG_probe',hist[1],cx['mg']['combined_optimal_contraction'],atol=1e-8)
        for j in range(1,len(hist)):
            check(f'{key}.within_cycle_monotonic_{j}',hist[j]<=hist[j-1]+1e-8)
        nf={k:row[k] for k in ['omega','mu','variant','iteration_count','true_relative_residual','seconds','fixed_step_cap']}
        nf.update(handoff_rho=cx['rho'],field_error_relative=cx['field_error_relative'])
        nf['implied_relative_residual_after_probe']=cx['rho']*row['true_relative_residual']
        nf.update(flatten_mg(cx['mg']))
        nf.update(flatten_metric(row['after_residual'],'after_residual_'))
        for target in [.5,.1,.01,.001]:
            nf[f'first_iteration_le_{target}']=next((j for j,r in enumerate(hist) if r<=target),None)
        full=row.get('full_completion') or {}
        nf['depth_full_completion_iterations']=full.get('iterations')
        nf['depth_full_completion_seconds']=full.get('seconds')
        nf['depth_full_completion_true_residual']=full.get('true_residual')
        if full:
            check(f'{key}.depth_completion_converged',full['true_residual']<=1e-6)
            close(f'{key}.depth_completion_last',full['history'][-1],full['true_residual'])
        norm_flat.append(nf)
        curves.extend(dict(omega=key[0],mu=key[1],variant=key[2],iteration=j,normalized_residual=r) for j,r in enumerate(hist))
    if production:
        expected={(meta['omega'],mu,method,depth) for mu in MUS for method in COLORS for depth in [1,2,3]}
        check(f'w{meta["omega"]}.all_18_states',set(index)==expected)
        check(f'w{meta["omega"]}.all_6_variants',set(crossindex)=={(meta['omega'],.317,v) for v in VARIANTS})
    return meta,flat,norm_flat,curves,blocks


def make_figures(out,states,normalized,curves):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'mplconfig'))
    os.environ.setdefault('XDG_CACHE_HOME',str(out/'cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.18,'savefig.dpi':180})
    final=[r for r in states if r['depth']==3]
    def save(fig,name):
        fig.savefig(out/(name+'.pdf'));fig.savefig(out/(name+'.png'));plt.close(fig)
    def frequency_panel(ax,key,label,log=False):
        for method,color in COLORS.items():
            available=[];means=[]
            for w in OMEGAS:
                vals=[r[key] for r in final if r['omega']==w and r['method']==method]
                if vals:
                    available.append(w);means.append(np.mean(vals))
                    offset=-.65 if method=='global' else .65
                    ax.scatter(np.full(len(vals),w+offset),vals,s=23,color=color,alpha=.65)
            ax.plot(available,means,'o-',color=color,lw=1.7,label='Global' if method=='global' else 'DD')
        if log:ax.set_yscale('log')
        ax.set_xticks(OMEGAS);ax.set_xlim(20,64);ax.set_xlabel(r'Frequency $\omega$');ax.set_ylabel(label)
    fig,axs=plt.subplots(2,2,figsize=(10.4,7.2),layout='constrained')
    for ax,key,label in zip(axs.flat,
        ['residual_pml_fraction','residual_core_fraction_physical','residual_fft_ge1p5_fraction_windowed','residual_roughness_transverse_over_omega2'],
        ['PML fraction of residual energy','Core fraction of physical residual energy',r'Windowed power at $|k_x|/\omega\geq1.5$',r'Transverse roughness / $\omega^2$']):
        frequency_panel(ax,key,label)
    axs.flat[0].legend(frameon=False)
    fig.suptitle('After three ROM layers: residual composition\nDots: individual queries; lines: means over three queries',fontsize=12)
    save(fig,'residual_composition')
    fig,axs=plt.subplots(2,2,figsize=(10.4,7.2),layout='constrained')
    for ax,key,label,log in zip(axs.flat,
        ['relative_field_error','error_pml_fraction','error_core_fraction_physical','error_fft_ge1p5_fraction_windowed'],
        [r'Approximate $\|e\|_2/\|u_{ref}\|_2$','PML fraction of approximate error energy','Core fraction of physical approximate error',r'Windowed error power at $|k_x|/\omega\geq1.5$'],
        [True,False,False,False]):
        frequency_panel(ax,key,label,log)
    axs.flat[0].legend(frameon=False)
    n_uncertain=sum(r['reference_disagreement_exceeds_1pct'] for r in final)
    fig.suptitle(f'Approximate field-error composition; {n_uncertain} final states exceed 1% reference disagreement\nReference disagreement is a sensitivity diagnostic, not an error bound',fontsize=11)
    save(fig,'approximate_error_composition')
    fig,axs=plt.subplots(2,2,figsize=(10.7,7.1),layout='constrained')
    keys=['mg_pre_optimal_contraction','mg_coarse_optimal_contraction','mg_post_optimal_contraction','mg_combined_optimal_contraction','mg_joint_span_contraction']
    for ax,w in zip(axs.flat,OMEGAS):
        found=False
        for method,color in COLORS.items():
            rows=[r for r in final if r['omega']==w and r['method']==method]
            if not rows:continue
            found=True;values=np.array([[r[k] for k in keys] for r in rows])
            for row in values:ax.plot(range(5),row,color=color,alpha=.22,lw=.8)
            ax.plot(range(5),values.mean(axis=0),'o-',color=color,lw=1.8,label='Global' if method=='global' else 'DD')
        if not found:ax.text(.5,.5,'Awaiting full results',ha='center',transform=ax.transAxes)
        ax.set_xticks(range(5),['Pre','Coarse','Post','V-cycle','Joint span']);ax.set_ylim(0,1.03)
        ax.set_title(rf'$\omega={w}$');ax.set_ylabel('Optimal residual fraction (lower is better)')
    axs.flat[0].legend(frameon=False)
    fig.suptitle('One-cycle component response on the true Helmholtz operator\nThin lines: queries; joint span allows independent coefficients',fontsize=12)
    save(fig,'mg_component_response')
    styles={'GG':('#2364aa','--'),'GGG':('#2364aa','-'),'DD':('#c95131','--'),
            'DDD':('#c95131','-'),'GGD':('#27945a','-.'),'DDG':('#9658a4',':')}
    fig,axs=plt.subplots(2,2,figsize=(10.5,7.1),layout='constrained')
    for ax,w in zip(axs.flat,OMEGAS):
        found=False
        for variant,(color,style) in styles.items():
            rows=[r for r in curves if r['omega']==w and r['variant']==variant]
            if not rows:continue
            found=True;ax.semilogy([r['iteration'] for r in rows],[r['normalized_residual'] for r in rows],color=color,ls=style,lw=1.8,label=variant)
        if not found:ax.text(.5,.5,'Awaiting full results',ha='center',transform=ax.transAxes)
        ax.set_title(rf'$\omega={w},\ \mu=0.317$');ax.set_xlabel('Fixed-work GMRES iteration')
        ax.set_ylabel(r'$\|r_k\|_2/\|r_0\|_2$')
    axs.flat[0].legend(frameon=False,ncol=3)
    fig.suptitle('Normalized residual directions: depth and final-layer exchange\nG / D denotes each frozen Global / DD layer; hybrids are off-trajectory interventions',fontsize=11)
    save(fig,'normalized_depth_hybrids')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-smoke',action='store_true')
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--cohort',action='append',default=[],metavar='OMEGA=PATH',
                        help='Explicit production directory, relative to this experiment folder or absolute; repeatable')
    parser.add_argument('--reference',action='append',default=[],metavar='OMEGA=PATH',
                        help='Explicit strict-reference directory; repeatable')
    args=parser.parse_args()
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out=BASE/'analysis'/stamp;out.mkdir(parents=True)
    def overrides(items):
        result={}
        for item in items:
            omega,path=item.split('=',1);omega=int(omega)
            if omega not in OMEGAS:parser.error('Omega override must be one of 24,36,48,60')
            directory=Path(path);result[omega]=directory if directory.is_absolute() else BASE/directory
        return result
    cohort_override=overrides(args.cohort);reference_override=overrides(args.reference)
    if args.validate_smoke:
        meta,states,norm,curves,blocks=process_cohort(BASE/'results/smoke_w24',False)
        report={'production':False,'smoke_schema_only':True,'states':len(states),'normalized_variants':len(norm),
                'checks':CHECKS,'passed':all(c['passed'] for c in CHECKS)}
        (out/'smoke_schema_validation.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
        print(out)
        return 0 if report['passed'] else 1
    states=[];norm=[];curves=[];blocks=[];metadata=[];status={};source_hashes={}
    for w in OMEGAS:
        folder=cohort_override.get(w,BASE/'results'/f'w{w}')
        needed=['complete.json','components.json','normalized.json','cross_spaces.json']
        missing=[name for name in needed if not (folder/name).exists()]
        if missing:
            status[str(w)]={'included':False,'status':'missing_or_incomplete','missing':missing,'directory':str(folder)};continue
        meta,st,no,cu,bl=process_cohort(folder)
        if meta['mode']!='full':
            status[str(w)]={'included':False,'status':'excluded_non_full_mode'};continue
        status[str(w)]={'included':True,'status':'completed_full','states':len(st),'normalized_variants':len(no),'job_id':meta['job_id'],'directory':str(folder)}
        states+=st;norm+=no;curves+=cu;blocks+=bl;metadata.append(meta)
        for name in needed+['input_sha256_before.txt','input_sha256_after.txt']:
            path=folder/name
            if path.exists():source_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        before=folder/'input_sha256_before.txt';after=folder/'input_sha256_after.txt'
        if before.exists() and after.exists():check(f'w{w}.source_hashes_unchanged',before.read_text()==after.read_text())
    states.sort(key=lambda r:(r['omega'],r['mu'],r['method'],r['depth']))
    norm.sort(key=lambda r:(r['omega'],r['mu'],VARIANTS.index(r['variant'])))
    refined=[];refined_history=[];refined_metadata=[]
    for w in OMEGAS:
        folder=reference_override.get(w,BASE/'results'/f'refined_w{w}')
        path=folder/'summary.json'
        if not path.exists():continue
        ref=json.loads(path.read_text());name=f'refined w{w} mu{ref["mu"]}'
        source_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        check(name+'.definition',ref['definition_version']=='dd_reference_refinement_v1')
        check(name+'.omega',ref['omega']==w)
        check(name+'.target_met',ref['target_met'] and ref['new_true_relative_residual']<=ref['requested_tolerance'])
        close(name+'.history_final',ref['residual_history'][-1],ref['new_true_relative_residual'],atol=1e-15)
        refined_metadata.append({key:value for key,value in ref.items() if key not in ['native_errors','residual_history']})
        refined_history.extend(dict(omega=w,mu=ref['mu'],iteration=j,true_relative_residual=value)
                               for j,value in enumerate(ref['residual_history']))
        for method,error in ref['native_errors'].items():
            for which in ['old','new']:metric_validate(error[which+'_components'],name+'.'+method+'.'+which)
            row=dict(omega=w,mu=ref['mu'],method=method,new_reference_true_residual=ref['new_true_relative_residual'],
                     requested_reference_tolerance=ref['requested_tolerance'],
                     old_relative_error=error['old_relative_error'],new_relative_error=error['new_relative_error'],
                     relative_error_norm_change=error['relative_error_norm_change'],
                     reference_change_over_new_error=error['reference_change_over_new_error'])
            row.update(flatten_metric(error['old_components'],'old_error_'))
            row.update(flatten_metric(error['new_components'],'new_error_'));refined.append(row)
    write_csv(out/'refined_reference_errors.csv',refined)
    write_csv(out/'refined_reference_history.csv',refined_history)
    archive=load_archive();final=[r for r in states if r['depth']==3]
    state_index={(r['omega'],round(r['mu'],3),r['method'],r['depth']):r for r in states}
    layer_changes=[]
    for row in final:
        previous=state_index[(row['omega'],round(row['mu'],3),row['method'],2)]
        layer_changes.append(dict(omega=row['omega'],mu=row['mu'],method=row['method'],
            residual_before_over_after=previous['rho']/row['rho'],
            approximate_error_before_over_after=previous['relative_field_error']/row['relative_field_error'],
            mg_one_step_L2=previous['mg_combined_optimal_contraction'],
            mg_one_step_L3=row['mg_combined_optimal_contraction'],
            mg_contraction_after_minus_before=row['mg_combined_optimal_contraction']-previous['mg_combined_optimal_contraction'],
            residual_core_fraction_change=row['residual_core_fraction_physical']-previous['residual_core_fraction_physical'],
            residual_pml_fraction_change=row['residual_pml_fraction']-previous['residual_pml_fraction'],
            residual_fft_ge1p5_fraction_change=row['residual_fft_ge1p5_fraction_windowed']-previous['residual_fft_ge1p5_fraction_windowed'],
            reference_disagreement_L3=row['reference_disagreement_over_error']))
        old=archive.get((row['omega'],round(row['mu'],3),row['method']))
        if old:
            close(f'{row["omega"]} {row["mu"]} {row["method"]}.archived_handoff',row['rho'],float(old['initial_relative_residual']),atol=1e-11)
            close(f'{row["omega"]} {row["mu"]} {row["method"]}.archived_first_MG_step',row['mg_combined_optimal_contraction'],float(old['normalized_r_at_1']),atol=2e-7)
    depth=[]
    for row in norm:
        if row['variant'] not in ['GG','DD'] or row['depth_full_completion_iterations'] is None:continue
        method='global' if row['variant']=='GG' else 'dd'
        old=archive.get((row['omega'],round(row['mu'],3),method))
        d=dict(omega=row['omega'],mu=row['mu'],method=method,
               layer2_completion_iterations=row['depth_full_completion_iterations'],
               layer2_true_residual=row['depth_full_completion_true_residual'],
               layer2_new_probe_seconds=row['depth_full_completion_seconds'],
               layer3_completion_iterations=int(old['completion_iterations']) if old else None,
               layer3_archived_completion_seconds=float(old['completion_seconds']) if old else None,
               timing_note='different runs/orthogonalization kernels; compare iterations, not seconds')
        if old:d['iterations_saved_by_layer3']=d['layer2_completion_iterations']-d['layer3_completion_iterations']
        depth.append(d)
    native_tradeoff=[]
    for w in OMEGAS:
        for method,letter in [('global','G'),('dd','D')]:
            two=next((r for r in norm if r['omega']==w and r['variant']==letter*2),None)
            three=next((r for r in norm if r['omega']==w and r['variant']==letter*3),None)
            if two is None or three is None:continue
            native_tradeoff.append(dict(omega=w,mu=two['mu'],method=method,
                L2_handoff=two['handoff_rho'],L3_handoff=three['handoff_rho'],
                handoff_L2_over_L3=two['handoff_rho']/three['handoff_rho'],
                L2_normalized_after_probe=two['true_relative_residual'],L3_normalized_after_probe=three['true_relative_residual'],
                normalized_L3_over_L2=three['true_relative_residual']/two['true_relative_residual'],
                L2_implied_residual_after_probe=two['implied_relative_residual_after_probe'],
                L3_implied_residual_after_probe=three['implied_relative_residual_after_probe'],
                implied_residual_L2_over_L3=two['implied_relative_residual_after_probe']/three['implied_relative_residual_after_probe']))
    write_csv(out/'per_state_components.csv',states);write_csv(out/'final_layer_components.csv',final)
    write_csv(out/'normalized_variants.csv',norm);write_csv(out/'normalized_histories.csv',curves)
    write_csv(out/'partition_energy.csv',blocks);write_csv(out/'depth_completion.csv',depth)
    write_csv(out/'layer3_changes.csv',layer_changes)
    write_csv(out/'native_probe_tradeoff.csv',native_tradeoff)
    verification=dict(status=status,complete=all(v['included'] for v in status.values()),
                      analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      python_executable=sys.executable,python_version=sys.version,
                      full_frequencies=[m['omega'] for m in metadata],state_count=len(states),
                      normalized_variant_count=len(norm),checks_passed=all(c['passed'] for c in CHECKS) if CHECKS else None,
                      check_count=len(CHECKS),failed_checks=[c for c in CHECKS if not c['passed']],
                      source_sha256=source_hashes,source_metadata=metadata)
    verification['strict_reference_metadata']=refined_metadata
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    (out/'checks.json').write_text(json.dumps(CHECKS,indent=2))
    (out/'data_dictionary.json').write_text(json.dumps({
        'row_identity':['omega','mu','method','depth'],
        'source_files':'results/w{omega}/components.json; normalized.json; cross_spaces.json',
        'physical_fraction':'{field}.region_fraction[0]; denominator: full-grid sum(abs(v)^2)',
        'pml_inner_fraction':'{field}.region_fraction[1]; maximum normalized outward depth <= 0.5',
        'pml_outer_fraction':'{field}.region_fraction[2]; maximum normalized outward depth > 0.5',
        'core_fraction_physical':'{field}.profile_band_fraction[4]; Gaussian profile >=0.5; denominator: physical energy',
        'core_fraction_total':'core_fraction_physical multiplied by physical_fraction',
        'fft_ge1p5_fraction_windowed':'{field}.axial_fft_fraction[4]; |kx|/omega>=1.5; denominator: physical x-Hann-windowed Fourier power',
        'roughness_transverse_over_omega2':'{field}.roughness_transverse_over_omega_squared; y,z physical forward differences divided by h^2, physical energy, and omega^2',
        'reference_disagreement_over_error':'max(||uref-uGlobal_final||,||uref-uDD_final||)/||uref-uROM||; sensitivity only',
        'error_capture_global/dd':'||U_global/dd,last^* (uref-uROM)||^2 / ||uref-uROM||^2; last-layer Euclidean span capture',
        'mg_combined_optimal_contraction':'min_alpha ||r-alpha*A*Msolve(r)||/||r||; matches one GMRES step for the fixed right preconditioner',
        'mg_joint_span_contraction':'minimum over independent coefficients on true-A images of pre/coarse/post corrections; not an actual V-cycle update',
        'first_iteration_le_target':'first recorded normalized history index satisfying target; blank means not reached within the fixed-work probe',
        'depth_completion':'new two-layer completion versus independently archived three-layer count; timings have different run/kernel provenance'
    },indent=2))
    if states and not args.no_plots:make_figures(out,states,norm,curves)
    pct=lambda x:f'{100*x:.2f}%'
    text=['# Frozen-model component analysis',f'Generated {stamp}. Only completed full runs are included; smoke runs are excluded.',
          '## Availability',table([dict(omega=w,**status[str(w)]) for w in OMEGAS],[('omega','omega'),('status','Status'),('states','States'),('normalized_variants','Variants')]),
          f'Checks: {len(CHECKS)}; failed: {sum(not c["passed"] for c in CHECKS)}. Expected complete cohort: 72 states and 24 normalized variants.',
          '## Final-layer residuals',table(final,[('omega','omega'),('mu','mu'),('method','Method'),('rho','Handoff residual'),('residual_pml_fraction','PML energy'),('residual_core_fraction_physical','Core / physical'),('residual_fft_ge1p5_fraction_windowed','FFT >=1.5'),('mg_combined_optimal_contraction','MG one-step fraction')],{k:pct for k in ['residual_pml_fraction','residual_core_fraction_physical','residual_fft_ge1p5_fraction_windowed']}),
          '## Approximate error and space capture',table(final,[('omega','omega'),('mu','mu'),('method','Method'),('relative_field_error','Relative field error'),('reference_disagreement_over_error','Reference disagreement / error'),('error_capture_global','Global last-space capture'),('error_capture_dd','DD last-space capture')],{'reference_disagreement_over_error':pct,'error_capture_global':pct,'error_capture_dd':pct}),
          '## Separate stricter-reference error study',
          'These rows use the tightened reference explicitly; original-reference tables and figures above remain unchanged. Reference change / new error measures the old-to-new vector difference, not the uncertainty of the new reference.',
          table(refined,[('omega','omega'),('mu','mu'),('method','Method'),('new_reference_true_residual','Reference true residual'),('new_relative_error','Refined relative error'),('relative_error_norm_change','Error-norm change'),('new_error_core_fraction_physical','Refined core / physical'),('new_error_fft_ge1p5_fraction_windowed','Refined FFT >=1.5'),('new_error_roughness_transverse_over_omega2','Refined transverse roughness')],{k:pct for k in ['relative_error_norm_change','new_error_core_fraction_physical','new_error_fft_ge1p5_fraction_windowed']}),
          '## Normalized depth and hybrid probes',table(norm,[('omega','omega'),('variant','Variant'),('handoff_rho','Original amplitude'),('iteration_count','Probe steps'),('true_relative_residual','Remaining normalized residual'),('first_iteration_le_0.1','Steps to 0.1'),('first_iteration_le_0.01','Steps to 0.01')]),
          '## What the third layer changes at mu=0.317',table([r for r in layer_changes if abs(r['mu']-.317)<1e-8],[('omega','omega'),('method','Method'),('residual_before_over_after','Residual reduction factor'),('approximate_error_before_over_after','Approximate error reduction factor'),('mg_one_step_L2','MG fraction after L2'),('mg_one_step_L3','MG fraction after L3')]),
          '## Depth completion',table(depth,[('omega','omega'),('method','Method'),('layer2_completion_iterations','Two-layer iterations'),('layer3_completion_iterations','Three-layer iterations'),('iterations_saved_by_layer3','Iterations saved by layer 3')]),
          '## Interpretation limits',
          '* Core means Gaussian profile >=0.5. Core/profile/axial/source fractions condition on physical-domain energy; PML fractions condition on full-grid energy. Fourier fractions condition on Hann-windowed power.',
          '* Fourier bins and finite-difference roughness describe spatial oscillation; they are not eigenmode, propagating-mode, or evanescent-mode weights.',
          '* Error fields use the saved shifted-MG solution as a reference. Reference disagreement above 1% is flagged in CSV; agreement is a sensitivity check, not a rigorous error bound.',
          '* Space capture is Euclidean energy of projection onto each last-layer space, on the same target error. It does not measure a Galerkin solve or prove a causal convergence advantage.',
          '* The one-step MG contraction is the minimum over a complex scalar times the full V-cycle action. Joint span permits three independent coefficients and is an auxiliary geometric diagnostic.',
          '* Hybrids swap a frozen final layer off its original training trajectory. Normalized probes remove residual amplitude and run a fixed iteration budget; their times are not production timings.',
          '* Depth-two completion is a new frozen-prefix ablation. Its iteration count is compared with archived depth-three completion; wall times come from different runs/kernels and are not treated as comparable.',
          '* No monotonic high-frequency improvement is assumed. Direction advantage can already exist at low frequency while a larger handoff amplitude makes DD slower overall.']
    (out/'summary.md').write_text('\n\n'.join(text)+'\n')
    (BASE/'analysis'/'latest.txt').write_text(str(out)+'\n')
    print(json.dumps({k:verification[k] for k in ['complete','full_frequencies','state_count','normalized_variant_count','check_count','checks_passed']},indent=2))
    print(out)
    return 0 if verification['checks_passed'] is not False else 1


if __name__=='__main__':
    raise SystemExit(main())
