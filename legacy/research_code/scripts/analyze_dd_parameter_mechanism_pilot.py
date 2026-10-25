"""Validate and summarize the independent 2026-09-15 small 3D pilot."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import math
import argparse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dd_parameter_mechanism_20260915'


def read_json(p):
    return json.loads(p.read_text())


def seq(x):
    return x if isinstance(x, list) else [x]


def csvwrite(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def mean(xs):
    return statistics.mean(xs)


def main(partial=False):
    hashes = read_json(OUT / 'source_hashes.json')
    assert all(hashlib.sha256((OUT/'source'/k).read_bytes()).hexdigest() == v for k,v in hashes.items())
    h2=read_json(OUT/'source_refinement_hashes.json')
    assert all(hashlib.sha256((OUT/'source_refinement'/k).read_bytes()).hexdigest()==v for k,v in h2.items())
    records, stats, directions, interventions, commutators, validation, training = [], [], [], [], [], [], []
    cases = [p for p in sorted((OUT/'results').glob('case_*')) if p.name != 'case_00']
    for p in cases:
        if not (p/'complete.json').exists():
            if partial:
                continue
            raise AssertionError(f'Incomplete case: {p}')
        info, protocol = read_json(p/'complete.json'), read_json(p/'protocol.json')
        assert info['passed']
        with (p/'online.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == info['online_solves']
        for r in rows:
            for k in ['case_id','mu','payload_MiB','pre_residual','field_error','iterations','true_residual','rom_s','setup_s','completion_s','online_s']:
                r[k] = float(r[k])
                assert math.isfinite(r[k])
            assert r['true_residual'] <= 1.001e-6
            # The driver reuses one already-built MG for paired diagnostics.
            # A deployable fixed ROM that already meets tolerance need not
            # construct MG at all. Preserve raw charged time, report both.
            r['necessary_online_s'] = r['online_s'] - (r['setup_s'] if r['pre_residual'] <= 1e-6 else 0)
        for c in seq(protocol['cohorts']):
            assert len(set(c['train'])) == len(c['train'])
            assert all(min(abs(q-t) for t in c['train']) > 1e-10 for q in c['queries'])
        refs = seq(read_json(p/'reference_training.json'))
        assert len(refs) == info['training_solves']
        assert max(x['true_residual'] for x in refs) < 2e-11
        checks = seq(read_json(p/'training_checks.json'))
        assert all(x['max_inverse_defect'] <= 1e-7 for x in checks)
        models = seq(read_json(p/'models.json'))
        for m in models:
            rr = [r for r in rows if r['cohort']==m['cohort'] and r['method']==m['method']]
            assert len(rr)==2
            assert max(abs(r['payload_MiB']-m['payload_MiB']) for r in rr)<1e-8
            st = dict(case_id=info['case_id'], cohort=m['cohort'], method=m['method'], samples=m['samples'],
                      payload_MiB=m['payload_MiB'], basis_MiB=m['basis_MiB'],operator_MiB=m['operator_MiB'],
                      ranks=','.join(str(v) for v in seq(m['layer_ranks'])),
                      mean_iterations=mean([r['iterations'] for r in rr]), max_iterations=max(r['iterations'] for r in rr),
                      mean_online_s=mean([r['online_s'] for r in rr]),
                      mean_necessary_online_s=mean([r['necessary_online_s'] for r in rr]),
                      max_pre_residual=max(r['pre_residual'] for r in rr),build_s=m['build_s'])
            stats.append(st)
            cohort=next(c for c in seq(protocol['cohorts']) if c['name']==m['cohort'])
            shared=sum(next(r['seconds'] for r in refs if abs(r['mu']-mu)<1e-12) for mu in cohort['train'])
            matched=[z for z in checks if z['cohort']==m['cohort'] and z['style']==m['method'].split('_')[0] and z['tail']==m['tail']] if '_TA3_' in m['method'] else []
            training.append(dict(case_id=info['case_id'],cohort=m['cohort'],method=m['method'],
              samples=m['samples'],common_reference_source_s=shared,model_build_source_s=m['build_s'],
              standalone_source_sum_s=shared+m['build_s'],fallback_solves=sum(z['fallback_solves'] for z in matched),
              references_reused=info['case_id'] in [101,102]))
        for d in seq(read_json(p/'mechanisms.json')) if (p/'mechanisms.json').exists() else []:
            assert d['mg']['relative_sum_vs_Msolve_error']<1e-10
            assert abs(d['history'][-1]-d['normalized20'])<1e-8
            assert abs(d['history'][1]-d['mg']['combined_optimal_contraction'])<1e-8
            directions.append(dict(case_id=info['case_id'],omega=protocol['cfg']['omega'],mu=d['mu'],method=d['method'],
              rho=d['rho'],normalized1=d['history'][1],normalized5=d['history'][5],normalized20=d['normalized20'],
              pre_smoothing=d['mg']['component_optimal_contraction'][0],
              coarse=d['mg']['component_optimal_contraction'][1],
              interface_fraction=d['residual_metrics']['interface_fraction'],interface_volume=d['residual_metrics']['interface_volume'],
              roughness=d['residual_metrics']['roughness_over_omega2'],field_error=d['field_error'],reference_over_handoff=d['reference_over_handoff']))
            if d['interventions']:
                for z in seq(d['interventions']):
                    assert abs(z['history'][-1]-z['rho20'])<1e-8
                    interventions.append(dict(case_id=info['case_id'],mu=d['mu'],method=d['method'],beta=z['beta'],sweeps=z['sweeps'],rho20=z['rho20']))
                interventions.append(dict(case_id=info['case_id'],mu=d['mu'],method=d['method'],beta=2,sweeps=2,rho20=d['normalized20']))
            if d['decomposition']:
                z=d['decomposition'];assert z['reconstruction_error']<1e-8
                commutators.append(dict(case_id=info['case_id'],mu=d['mu'],method=d['method'],
                  bulk_over_total=z['bulk_over_total'],interface_over_total=z['commutator_over_total'],cosine=z['cross_cosine'],
                  bulk20=z['bulk_rho20'],interface20=z['commutator_rho20'],support_fraction=z['commutator_metrics']['interface_fraction'],reconstruction=z['reconstruction_error']))
        validation.append(dict(case_id=info['case_id'],N=info['N'],online_solves=len(rows),training_solves=len(refs),
          new_training_reference_solves=info.get('new_training_solves',len(refs)),
          reference_source_s=sum(r['seconds'] for r in refs),fallback_solves=sum(c['fallback_solves'] for c in checks),
          max_inverse_defect=max((c['max_inverse_defect'] for c in checks),default=0),elapsed_s=info['elapsed_s']))
        records.extend(rows)
    if not partial:
        assert {s['case_id'] for s in validation} == set(range(1,7)) | {101,102}
    csvwrite(OUT/'online_all.csv',records)
    csvwrite(OUT/'model_summary.csv',stats)
    csvwrite(OUT/'directions.csv',directions)
    csvwrite(OUT/'interventions.csv',interventions)
    csvwrite(OUT/'commutators.csv',commutators)
    csvwrite(OUT/'case_validation.csv',validation)
    csvwrite(OUT/'training_summary.csv',training)
    front=[]
    for cohort in ['narrow5','medium9','wide15','wide5']:
        for cap in [0,10,20,40,80]:
            for family in ['dd_fixed','dd_TA3']:
                candidates=[s for s in stats if s['case_id'] in [1,101] and s['cohort']==cohort and s['method'].startswith(family) and s['max_iterations']<=cap]
                best=min(candidates,key=lambda s:s['payload_MiB']) if candidates else None
                front.append(dict(cohort=cohort,cap=cap,family=family,min_payload_MiB=best['payload_MiB'] if best else '',method=best['method'] if best else '',max_iterations=best['max_iterations'] if best else ''))
    csvwrite(OUT/'sampled_cost_envelope.csv',front)
    (OUT/'validation.json').write_text(json.dumps(dict(passed=not partial,cases=validation,source_hashes_unchanged=True,formal_solves=len(records)),indent=2))
    print(json.dumps(dict(cases=len(validation),formal_solves=len(records),validation=validation),indent=2))
    try:
        plot(stats,directions,interventions)
    except ImportError as exc:
        print(f'Plots skipped: {exc}')


def plot(stats,directions,interventions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cohorts=['narrow5','medium9','wide15','wide5']
    fig,axs=plt.subplots(1,4,figsize=(13,3.5),sharey=True)
    for ax,c in zip(axs,cohorts):
        for family,color,marker in [('dd_fixed','#2866a3','o'),('dd_TA3','#c85c34','s')]:
            s=sorted([s for s in stats if s['case_id'] in [1,101] and s['cohort']==c and s['method'].startswith(family)],key=lambda s:s['payload_MiB'])
            if s:ax.plot([z['payload_MiB'] for z in s],[z['max_iterations'] for z in s],marker=marker,color=color,label=family)
        ax.set_title(c);ax.set_xlabel('Deployment payload (MiB)');ax.grid(alpha=.2)
    axs[0].set_ylabel('Worst completion iterations (2 queries)');axs[0].legend(fontsize=8)
    fig.suptitle('Small 3D waveguide: sampled storage / completion tradeoffs')
    fig.tight_layout();fig.savefig(OUT/'cost_tradeoff.png',dpi=180);fig.savefig(OUT/'cost_tradeoff.pdf');plt.close(fig)
    if not directions:return
    ids=sorted({d['case_id'] for d in directions});fig,axs=plt.subplots(1,2,figsize=(10,4))
    for fam,color,mark in [('fixed_t0e+00','#2866a3','o'),('TA3_t3e-03','#c85c34','s')]:
        ratios=[];first=[]
        for cid in ids:
            dd=[d for d in directions if d['case_id']==cid and d['method']=='dd_'+fam]
            gg=[d for d in directions if d['case_id']==cid and d['method']=='global_'+fam]
            ratios.append(mean(d['normalized20'] for d in dd)/mean(d['normalized20'] for d in gg))
            first.append(mean(d['normalized1'] for d in dd)/mean(d['normalized1'] for d in gg))
        axs[0].plot(ids,first,marker=mark,color=color,label=fam.split('_')[0]);axs[1].plot(ids,ratios,marker=mark,color=color)
    for ax in axs:
        ax.axhline(1,color='gray',ls='--');ax.set_ylim(0,1.08);ax.set_yticks([0,.2,.4,.6,.8,1]);ax.set_xlabel('Case ID');ax.set_ylabel('DD / Global normalized residual');ax.grid(alpha=.2)
    axs[0].set_title('One GMRES step');axs[1].set_title('20 GMRES steps');axs[0].legend()
    fig.suptitle('Same A, MG and unit handoff residual; below 1 favors DD')
    fig.tight_layout();fig.savefig(OUT/'direction_ratios.png',dpi=180);fig.savefig(OUT/'direction_ratios.pdf');plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args();main(args.partial)
