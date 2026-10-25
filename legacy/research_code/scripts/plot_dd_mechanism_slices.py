#!/usr/bin/env python3
"""Plot exported, actual-grid central planes; never read full 3-D vectors.

Example:
  python plot_dd_mechanism_slices.py --input-root results/interfaces_JOBID \
      --output-dir outputs/new_slice_figures --fields r0

Requires numpy/matplotlib. The output directory must not exist. Each figure
uses one shared color scale across all frequency/method panels. Also exports
cell-volume compensated plots to distinguish grid density from localization.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--fields",nargs="+",default=["r0"],choices=["r0","e0","rafter","strict_e0"])
    args=p.parse_args()
    assert not args.output_dir.exists(),"Refusing to overwrite plot output directory."
    paths=sorted(args.input_root.rglob("slices.json"))
    assert paths,"No exported slices.json found."
    data={}
    for path in paths:
        d=json.loads(path.read_text())
        assert d["validated"] and d["definition_version"]=="dd_central_slices_v1"
        w=int(d["omega"]);assert w not in data,f"Duplicate frequency{w}: select one export cohort."
        data[w]=(d,path)
    args.output_dir.mkdir(parents=True)
    os.environ.setdefault("MPLCONFIGDIR",str(args.output_dir/"mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    exports=[]
    for field in args.fields:
        selected=[]
        for omega,(d,path) in sorted(data.items()):
            records={r["variant"]:r for r in d["records"]}
            if all(v in records and field in records[v]["fields"] for v in ("GGG","DDD")):
                selected.append((omega,d,records,path))
        assert selected,f"No paired native records for{field}."
        for compensation in (False,True):
            samples=[]
            for omega,d,records,path in selected:
                dv=float(np.prod(d["geometry"]["spacing"]))
                for variant in ("GGG","DDD"):
                    a=np.asarray(records[variant]["fields"][field]["normalized_abs_slice"],float)
                    if compensation:a=a/np.sqrt(dv)
                    samples.extend(np.log10(a[a>0]).tolist())
            assert samples,"All slice values are zero."
            vmax=float(np.ceil(np.quantile(samples,.998)*2)/2);vmin=vmax-5
            fig,ax=plt.subplots(len(selected),2,figsize=(11.8,2.6*len(selected)+.6),squeeze=False)
            fig.subplots_adjust(left=.065,right=.89,bottom=.10,top=.87,wspace=.17,hspace=.42)
            for i,(omega,d,records,path) in enumerate(selected):
                g=d["geometry"];x=np.asarray(g["x"]).ravel();y=np.asarray(g["y"]).ravel()
                h=np.asarray(g["spacing"]).ravel();nphys=np.asarray(g["physical_grid_size"]).ravel().astype(int)
                pml=np.asarray(g["pml_cells"]).ravel().astype(int);L=np.asarray(g["domain_lengths"]).ravel()
                labels=np.asarray(g["block_labels"]);profile=np.asarray(g["waveguide_profile"])
                physical=(np.arange(len(x))[:,None]>=pml[0]) & (np.arange(len(x))[:,None]<pml[0]+nphys[0])
                physical=physical & (np.arange(len(y))[None,:]>=pml[1]) & (np.arange(len(y))[None,:]<pml[1]+nphys[1])
                segments=[]
                ii,jj=np.where((labels[1:,:]!=labels[:-1,:]) & physical[1:,:] & physical[:-1,:])
                for k,l in zip(ii,jj):
                    xx=(x[k]+x[k+1])/2;segments.append([(xx,y[l]-h[1]/2),(xx,y[l]+h[1]/2)])
                ii,jj=np.where((labels[:,1:]!=labels[:,:-1]) & physical[:,1:] & physical[:,:-1])
                for k,l in zip(ii,jj):
                    yy=(y[l]+y[l+1])/2;segments.append([(x[k]-h[0]/2,yy),(x[k]+h[0]/2,yy)])
                for j,variant in enumerate(("GGG","DDD")):
                    r=records[variant];f=r["fields"][field]
                    values=np.asarray(f["normalized_abs_slice"],float)
                    assert values.shape==(len(x),len(y)),"x/y matrix orientation mismatch."
                    if compensation:values=values/np.sqrt(float(np.prod(h)))
                    log=np.log10(np.maximum(values,10**(vmin-1)))
                    a=ax[i,j]
                    im=a.imshow(log.T,origin="lower",extent=[x[0]-h[0]/2,x[-1]+h[0]/2,y[0]-h[1]/2,y[-1]+h[1]/2],
                                aspect="equal",cmap="magma",vmin=vmin,vmax=vmax,interpolation="nearest",rasterized=True)
                    if segments:a.add_collection(LineCollection(segments,colors="white",linewidths=.35,alpha=.42))
                    a.add_patch(Rectangle((0,0),L[0],L[1],fill=False,edgecolor="#bdbdbd",linewidth=.8,linestyle="--"))
                    if profile.size and np.min(profile)<.5<np.max(profile):
                        a.contour(x,y,profile.T,levels=[.5],colors=["#38d9e6"],linestyles="--",linewidths=.8)
                    a.set_title(rf"$\omega={omega}$ | {'Global' if variant=='GGG' else 'DD'} | $\mu={r['mu']:.3f}$",fontsize=11)
                    a.set_ylabel("y")
                    if i==len(selected)-1:a.set_xlabel("x")
            label={"r0":"Residual at handoff","e0":"Error (original reference)",
                   "rafter":"Residual after completion","strict_e0":"Error (stricter reference)"}[field]
            normtext="volume normalization" if compensation else "nodal normalization"
            fig.suptitle(f"{label} — {normtext}",y=.98,fontsize=13)
            cax=fig.add_axes([.912,.155,.018,.64])
            cb=fig.colorbar(im,cax=cax)
            cb.set_label(r"$\log_{10}(|v|/(\|v\|_2\sqrt{\Delta V}))$" if compensation else r"$\log_{10}(|v|/\|v\|_2)$",fontsize=10)
            stem=field+("_density" if compensation else "_nodal")
            for suffix in ("pdf","png"):fig.savefig(args.output_dir/f"{stem}.{suffix}",dpi=180,bbox_inches="tight")
            plt.close(fig)
            exports.append({"field":field,"cell_volume_compensated":compensation,"color_vmin":vmin,"color_vmax":vmax,
                            "frequencies":[a[0] for a in selected],"file_stem":stem})
    manifest={"source_files":[{"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
              "plots":exports,"overlay":"Thin white: exact physical DD block boundaries (same mask on both methods); gray dashed: physical domain; cyan dashed: material profile0.5.",
              "interpretation":"Every vector is normalized by its own full Euclidean norm. Color indicates spatial concentration, not original residual magnitude. Nodal normalization depends on grid density; density plots compensate by sqrt(cell volume). Central slices alone do not determine total regional energy or causal convergence effects.",
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir/"plot_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"output_dir":str(args.output_dir.resolve()),"plots":exports},indent=2))


if __name__=="__main__":main()
