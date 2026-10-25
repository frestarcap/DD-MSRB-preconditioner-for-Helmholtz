#!/usr/bin/env python3
"""Generate manuscript evidence from the fixed revision CSV files.

Run after serial timing, PC export, and the deployment holdouts. Outputs are
LaTeX source and intentional publication figures; raw data stay in outputs/.
"""
from pathlib import Path
import csv
import math
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/revision_20260907"
TEX = ROOT / "LaTeX_DL_468198_240419"


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def num(row, key):
    return float(row[key])


def sci(x):
    if not math.isfinite(x):
        return "--"
    mantissa, exponent = f"{x:.2e}".split("e")
    return rf"\({mantissa}\times10^{{{int(exponent)}}}\)"


def table(caption, label, columns, header, rows, size="small"):
    return (
        rf"\begin{{table}}[t]\centering\{size}" + "\n"
        + rf"\caption{{{caption}}}\label{{{label}}}" + "\n"
        + r"\setlength{\tabcolsep}{3pt}" + "\n"
        + rf"\begin{{tabular}}{{{columns}}}\toprule" + "\n"
        + " & ".join(header) + r" \\" + "\n\\midrule\n"
        + "\n".join(" & ".join(row) + r" \\" for row in rows)
        + "\n\\bottomrule\\end{tabular}\\end{table}\n"
    )


def main():
    half = read(DATA / "halfspace/summary_timing.csv")
    pc = read(DATA / "pc/paper_summary.csv")
    pc_all = read(DATA / "pc/summary_timing.csv")
    lens = []
    for name in ("all", "middle", "simple_middle"):
        lens.extend(read(DATA / f"lens/summary_{name}.csv"))
    keyed = {(r["candidate"], r["method"]): r for r in lens}
    macros = {}
    for method, title in (("global", "Global"), ("dd", "DD")):
        row = keyed[("tight", method)]
        for field, key in (("Basis", "basis_MiB"), ("Payload", "payload_MiB"), ("Offline", "offline_s")):
            macros[f"Lens{title}{field}"] = f"{num(row,key):.3f}"
        times = read(DATA / f"deployment/lens_tight_{method}_lazy_timings.csv")
        for mu, suffix in ((0.283, "A"), (0.417, "B")):
            group = [r for r in times if abs(num(r, "parameter") - mu) < 1e-9]
            assert len(group) == 5
            macros[f"Lens{title}Time{suffix}"] = f"{st.mean(num(r,'total_s') for r in group):.4f}"
            macros[f"Lens{title}SD{suffix}"] = f"{st.stdev(num(r,'total_s') for r in group):.4f}"
            macros[f"Lens{title}Residual{suffix}"] = sci(max(num(r,"true_residual") for r in group))
    g, d = keyed[("tight", "global")], keyed[("tight", "dd")]
    macros["LensBasisSaving"] = f"{100*(1-num(d,'basis_MiB')/num(g,'basis_MiB')):.1f}"
    macros["LensPayloadSaving"] = f"{100*(1-num(d,'payload_MiB')/num(g,'payload_MiB')):.1f}"
    for row, title in zip(pc, ("Global", "DD")):
        for field, key in (("Basis", "basis_MiB"), ("Payload", "payload_MiB"),
                           ("Operators", "operators_MiB"), ("Offline", "offline_s"),
                           ("Time", "total_s"), ("Completion", "completion_mean")):
            macros[f"PC{title}{field}"] = f"{num(row,key):.3f}"
        for field, key in (("Pre", "mean_precompletion_residual"), ("True", "max_true_residual"),
                           ("MeanError", "mean_error"), ("MaxError", "max_error")):
            macros[f"PC{title}{field}"] = sci(num(row,key))
    macros["PCBasisSaving"] = f"{100*(1-num(pc[1],'basis_MiB')/num(pc[0],'basis_MiB')):.1f}"
    macros["PCPayloadSaving"] = f"{100*(1-num(pc[1],'payload_MiB')/num(pc[0],'payload_MiB')):.1f}"
    (TEX / "jsc_result_macros.tex").write_text(
        "% Generated from revision CSV; do not hand edit numerical values.\n"
        + "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")

    rows = []
    for nt in (6,12,24):
        for name, label in ((f"global_n{nt}_eta-5", "Global TA-ROM"),
                            (f"dd_n{nt}_eta-5", "DD-TA-ROM")):
            r = next(r for r in half if r["candidate"] == name)
            assert int(num(r,"repetitions")) == 5
            rows.append([str(nt),label,f"{num(r,'payload_MiB'):.3f}",f"{num(r,'offline_s'):.2f}",
                         f"{num(r,'online_mean_s'):.4f}",f"{num(r,'online_worst_s'):.4f}",
                         str(int(num(r,'completion_max'))),f"{num(r,'max_query_timing_sd'):.4f}"])
    halftex = table(
        "Half-space Global/DD training-coverage comparison over nine fixed queries. "
        "Online times use five repetitions; Max. it. counts completion iterations "
        "and Max. SD is the largest per-query timing standard deviation.",
        "tab:jsc-halfspace-reuse", "rlrrrrrr", [r"$n_s$","Method","ROM (MiB)","Offline (s)","Mean (s)","Worst (s)","Max. it.","Max. SD"], rows, "scriptsize")
    (TEX / "jsc_halfspace_table.tex").write_text(halftex)

    rows=[]
    names={"edge":"$(0.012,0.016)$", "balanced":"$10^{-3}$", "middle":"$3\\times10^{-4}$", "tight":"$10^{-4}$"}
    for candidate,label in names.items():
        for method,title in (("global","Global"),("dd","DD")):
            r=keyed[(candidate,method)]
            ts=read(DATA/f"deployment/lens_{candidate}_{method}_lazy_timings.csv")
            rows.append([label,title,f"{num(r,'payload_MiB'):.3f}",
                         f"{st.mean(num(x,'total_s') for x in ts):.4f}",
                         str(int(max(num(x,'completion_iterations') for x in ts)))])
    lenstex=table("Lens storage--time tradeoff at a common $10^{-6}$ true residual target. "
        "Both models are fully retrained at each tolerance. A scalar denotes the same budget in both layers. "
        "Times use five serial repetitions at each of the two development parameters and include MG construction "
        "only when completion is needed. The primary lens model uses $10^{-4}$.",
        "tab:jsc-lens-tradeoff","llrrr",["Tail budget","Method","ROM (MiB)","Mean online (s)","Max. completion"],rows)
    (TEX/"jsc_lens_table.tex").write_text(lenstex)

    labels={"global":"Global", "current4":"Four blocks", "bands1_axial1_signed0":"Six blocks (primary)",
            "bands1_axial1_signed1":"Signed/depth", "bands1_axial3_signed0":"Three axial sections",
            "bands1_axial3_signed1":"Axial + signed", "bands3_axial1_signed0":"Four transverse bands",
            "bands3_axial3_signed1":"Bands + axial + signed"}
    rows=[]
    for r in pc_all:
        rows.append([labels[r['candidate']],f"{num(r,'basis_saving_percent'):.1f}\\%",
                     f"{num(r,'payload_MiB'):.3f}",str(int(num(r,'completion_max'))),
                     sci(num(r,'precompletion_residual_max'))])
    (TEX/"jsc_pc_partition_table.tex").write_text(table(
        "PC partition study after independent trajectory replay at unchanged snapshot budgets. "
        "Every solve passes a $10^{-6}$ true Helmholtz residual test. The final column is the worst residual "
        "before completion. All eight geometry-screened partitions and the full tested subset are archived; "
        "compression of fixed snapshots is not used as a substitute for these online measurements.",
        "tab:jsc-pc-partitions","lrrrr",["Representation","Basis saving","ROM (MiB)","Max. completion","Before completion"],rows,"scriptsize"))

    raw_layers=read(DATA/"pc/paper_layers.csv"); rows=[]
    for j in (1,2,3):
        g=next(r for r in raw_layers if r['method']=='global' and int(r['layer'])==j)
        d=next(r for r in raw_layers if r['method']!='global' and int(r['layer'])==j)
        ge,de=int(g['entries']),int(d['entries'])
        rows.append([str(j),d['ranks'],f"{ge:,}",f"{de:,}",f"{100*(1-de/ge):.1f}\\%"])
    (TEX/"jsc_pc_storage_table.tex").write_text(table("Primary PC six-block basis storage by layer.",
        "tab:pc-storage","rlrrr",["Layer","Local ranks","Global entries","DD entries","Saving"],rows,"scriptsize"))
    rows=[]
    for r,label in zip(pc,("Global TA-ROM","DD-TA-ROM")):
        rows.append([label]+[f"{num(r,k):.4f}" for k in ('reduced_s','lift_s','completion_s','total_s','max_query_sd')])
    (TEX/"jsc_pc_timing_table.tex").write_text(table(
        "Primary PC online times in seconds, averaged over the same eleven parameters and five serial "
        "repetitions after one warm-up. The last column is the largest per-query sample standard deviation "
        "of total time. Every final true Helmholtz residual is at most $10^{-6}$; phase costs include "
        "recovery and acceptance checks, including operator reconstruction inside completion. Initial full-order setup is excluded.",
        "tab:pc-speedup","lrrrrr",["Method","Reduced","Lift","Completion","Total","Max. SD"],rows,"scriptsize"))


    rows=[]
    halfdiag=read(DATA/'halfspace/online_action_diagnostics.csv')
    selected=[r for r in halfdiag if r['candidate']=='global_n6_eta-5' and abs(num(r,'parameter')-.785)<1e-10]
    for r in selected:
        rows.append(['Half-space, 6 Global',r['layer']]+[sci(num(r,k)) for k in
            ('action_defect','previous_FFOM_relative','product_bound','FGMRES_relative','condition_reduced')])
    for r in read(DATA/'deployment/waveguide_online_diagnostics.csv'):
        if abs(num(r,'parameter')-.317)<1e-10 and int(r['layer'])==2:
            rows.append(['Waveguide, '+r['method'],r['layer']]+[sci(num(r,k)) for k in
                ('action_defect','previous_FFOM_relative','product_bound','FGMRES_relative','condition_reduced')])
    intro=r"""\experimentlabel{Reduced-action defects}
Table~\ref{tab:jsc-action} evaluates \(\|AP_jq_j-q_j\|_2\), the preceding
FFOM residual, their product bound, and the FGMRES residual on the online
trajectory, without full inverse solves. Residuals are normalized by
\(\|b\|_2\). For the six-snapshot half-space
model, the second action defect exceeds one and FGMRES stagnates, exposing
insufficient training coverage. For the archived two-layer waveguide at
\(\omega=24\), DD has a smaller
second-layer defect than Global (\(0.0824\) versus \(0.381\)) but a larger
product bound (\(0.00881\) versus \(0.000536\)) because its preceding
FFOM residual is larger. Both factors matter; the action defect alone
does not rank solver performance.
"""
    (TEX/'jsc_action_diagnostics.tex').write_text(intro+table(
        'Selected online action defects and residual bound. No full inverse solve is used for this diagnostic. '
        'Queries are 0.785 for half-space and 0.317 for the waveguide.',
        'tab:jsc-action','lrrrrrr',['Model','$j$',r'$\|\varepsilon_j\|$',r'$\rho^{\rm FF}_{j-1}$','Product',
        r'$\rho^{\rm FG}_j$',r'$\kappa_2(B_j)$'],rows,'scriptsize'))
    rows=[]
    hg=read(DATA/'deployment/lens_tight_global_holdout_lazy_timings.csv')
    hd=read(DATA/'deployment/lens_tight_dd_holdout_lazy_timings.csv')
    for g,d in zip(hg,hd):
        mu=num(g,'parameter'); assert mu==num(d,'parameter')
        rows.append([f'{mu:.5f}','I' if .15<mu<.55 else 'E',
            str(int(num(g,'completion_iterations'))),str(int(num(d,'completion_iterations'))),
            sci(num(g,'true_residual')),sci(num(d,'true_residual'))])
    (TEX/'jsc_lens_holdout_table.tex').write_text(table(
        'Frozen primary lens holdouts, selected with seed 9072026 after the budget was fixed. '
        'I denotes interpolation and E extrapolation. The reduced ranks remain unchanged. '
        'Completion counts and true residuals are reported without mixing timing cohorts.',
        'tab:jsc-lens-holdout','rlrrrr',['Parameter','Set','Global it.','DD it.','Global residual','DD residual'],rows))

    plot_halfspace_tradeoff(half)


def plot_halfspace_tradeoff(half):
    # Shared figure used by both manuscript routes. Standard scientific plot.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
    fig,ax=plt.subplots(figsize=(7.3,4.5),layout='constrained')
    for prefix,label,color,marker in (("global","Global TA-ROM","#d62728","o"),
                                     ("dd","DD-TA-ROM","#207a4b","s")):
        r=[r for r in half if r['candidate'].startswith(prefix+'_n24_')]
        r.sort(key=lambda x:num(x,'payload_MiB'))
        ax.plot([num(x,'payload_MiB') for x in r],[num(x,'online_mean_s') for x in r],
                color=color,marker=marker,label=label)
        for x in r:
            exponent=x['candidate'].split('eta')[1]
            ax.annotate(rf'$10^{{{exponent}}}$',(num(x,'payload_MiB'),num(x,'online_mean_s')),
                        xytext=(3,6),textcoords='offset points',fontsize=10)
    ax.set(xlabel='Persistent ROM payload (MiB)',ylabel='Mean total online time (s)',yscale='log')
    ax.set_xlim(3,27); ax.grid(alpha=.2); ax.legend(framealpha=.95)
    path=TEX/'figures/helm_correction_rom/helm_tol_tradeoff_paper.png'
    fig.savefig(path,dpi=240); plt.close(fig)


if __name__ == '__main__':
    main()
