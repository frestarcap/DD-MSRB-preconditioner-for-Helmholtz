#!/usr/bin/env python3
"""Compact report comparison: two rows per frequency/stage, two columns=method.

Uses only saved actual central planes; normalizes each field by its own full
norm and cell-volume factor. This is a shape comparison, not a magnitude
reduction plot. No existing output directory is overwritten.
"""
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
    p.add_argument("--frequencies",type=int,nargs="+",default=[48])
    a=p.parse_args();assert not a.output_dir.exists(),"Refusing an existing output directory."
    sources={}
    for f in a.input_root.rglob("slices.json"):
        d=json.loads(f.read_text());w=int(d["omega"])
        if w not in a.frequencies:continue
        assert d["validated"] and w not in sources,"Invalid or duplicate frequency export."
        sources[w]=(d,f)
    assert set(sources)==set(a.frequencies),"A requested frequency is not available."
    fields=[("GGG","r0","Global: handoff"),("DDD","r0","DD: handoff"),
            ("GGG","rafter","Global: after"),("DDD","rafter","DD: after")]
    # Fixed manuscript scale, independent of selected frequencies.
    vmin,vmax=-4,1
    a.output_dir.mkdir(parents=True);os.environ.setdefault("MPLCONFIGDIR",str(a.output_dir/"mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.family":"serif","font.size":9,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(2*len(a.frequencies),2,figsize=(7.2,3.4*len(a.frequencies)),squeeze=False)
    fig.subplots_adjust(left=.10,right=.87,bottom=.075,top=.96,wspace=.12,hspace=.42)
    for row,w in enumerate(a.frequencies):
        d,_=sources[w];r={x["variant"]:x for x in d["records"]};g=d["geometry"]
        x=np.asarray(g["x"]).ravel();y=np.asarray(g["y"]).ravel();h=np.asarray(g["spacing"]).ravel()
        labels=np.asarray(g["block_labels"])
        nphys=np.asarray(g["physical_grid_size"]).ravel().astype(int);pml=np.asarray(g["pml_cells"]).ravel().astype(int)
        physical=(np.arange(len(x))[:,None]>=pml[0]) & (np.arange(len(x))[:,None]<pml[0]+nphys[0])
        physical=physical & (np.arange(len(y))[None,:]>=pml[1]) & (np.arange(len(y))[None,:]<pml[1]+nphys[1])
        segments=[]
        ii,jj=np.where((labels[1:,:]!=labels[:-1,:]) & physical[1:,:] & physical[:-1,:])
        for i,j in zip(ii,jj):
            xx=(x[i]+x[i+1])/2;segments.append([(xx,y[j]-h[1]/2),(xx,y[j]+h[1]/2)])
        ii,jj=np.where((labels[:,1:]!=labels[:,:-1]) & physical[:,1:] & physical[:,:-1])
        for i,j in zip(ii,jj):
            yy=(y[j]+y[j+1])/2;segments.append([(x[i]-h[0]/2,yy),(x[i]+h[0]/2,yy)])
        L=np.asarray(g["domain_lengths"]).ravel()
        for col,(variant,field,title) in enumerate(fields):
            values=np.asarray(r[variant]["fields"][field]["normalized_abs_slice"])/np.sqrt(np.prod(h))
            assert values.shape==(len(x),len(y))
            z=np.log10(np.maximum(values,10**(vmin-1)));axis=ax[2*row+col//2,col%2]
            im=axis.imshow(z.T,origin="lower",extent=[x[0]-h[0]/2,x[-1]+h[0]/2,y[0]-h[1]/2,y[-1]+h[1]/2],
                           aspect="equal",cmap="magma",vmin=vmin,vmax=vmax,interpolation="nearest",rasterized=True)
            if segments:axis.add_collection(LineCollection(segments,colors="white",linewidths=.25,alpha=.35))
            axis.add_patch(Rectangle((0,0),L[0],L[1],fill=False,edgecolor="#bdbdbd",linestyle="--",linewidth=.65))
            axis.set_xticks([0,1,2,3]);axis.set_yticks([0,.5,1])
            if col%2==0:axis.set_ylabel(rf"$\omega={w}$"+"\n"+("Switch" if col<2 else "40 steps")+"\n"+r"$y$")
            else:axis.tick_params(labelleft=False)
            if row==0 and col<2:axis.set_title("global" if col==0 else "DD",fontsize=11)
            if row==len(a.frequencies)-1 and col>=2:axis.set_xlabel("x")
    cax=fig.add_axes([.90,.22,.018,.56]);cb=fig.colorbar(im,cax=cax)
    cb.set_label(r"$\log_{10}(|v|/(\|v\|_2\sqrt{\Delta V}))$",fontsize=10)
    for ext in ["png","pdf"]:fig.savefig(a.output_dir/f"before_after.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)
    manifest={"frequencies":a.frequencies,"color_vmin":vmin,"color_vmax":vmax,
              "sources":[{"omega":w,"file":str(sources[w][1].resolve()),"sha256":hashlib.sha256(sources[w][1].read_bytes()).hexdigest()} for w in a.frequencies],
              "normalization":"Each vector divided by its own full Euclidean norm and sqrt(cell volume); this compares shape, not absolute magnitude reduction.",
              "overlay":"Thin white: physical DD interfaces on both methods; gray dashed: physical domain.",
              "after":"Saved rafter of fixed-work normalized completion; iteration count must be cited from accompanying solver data, not inferred from this picture.",
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.output_dir/"plot_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(str(a.output_dir.resolve()))


if __name__=="__main__":main()
