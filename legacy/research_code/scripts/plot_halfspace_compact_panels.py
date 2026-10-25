"""Render compact half-space panels from existing residual and storage evidence."""
from pathlib import Path
import csv, json, re, subprocess, tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/halfspace_dd_paper'
FIG = ROOT / 'LaTeX_DL_468198_240419/figures/helm_correction_rom'

def arrays(file, dataset):
    # h5dump resolves MATLAB cell references, without loading full solution arrays.
    text = subprocess.check_output(['h5dump', '-y', '-w', '0', '-m', '%.17g',
                                    '-d', dataset, str(file)], text=True)
    numeric = r'[-+0-9.eE,\s]+'
    return [np.fromstring(x.replace('\n', ' '), sep=',')
            for x in re.findall(r'DATA\s*\{\s*('+numeric+r')\s*\}', text)]

def save(fig, name):
    for ext in ('pdf', 'png'):
        target=FIG / f'{name}.{ext}'
        with tempfile.NamedTemporaryFile(dir=FIG, suffix='.'+ext, delete=False) as f:
            temp=Path(f.name)
        fig.savefig(temp, dpi=240)
        temp.replace(target)

def main():
    d = DATA/'true_residual_representative.mat'
    p = DATA/'fixed_pod_residual_interpolation.mat'
    dd_index=int(arrays(d, '/diagnostic/dd_index')[0][0])-1
    curves=[]
    for name, file, group, k, color, marker in [
        ('Global TA-ROM', d, '/diagnostic/global', 0, '#d90d0d', 'o'),
        ('DD-TA-ROM', d, '/diagnostic/dd', dd_index, '#0d7340', 's'),
        ('NR-ROM', d, '/diagnostic/nr', 0, '#2659bf', 'D'),
        ('Fixed solution POD', p, '/fixed/test', 0, '#d9730d', '^')]:
        if name == 'DD-TA-ROM':
            h=arrays(file,group+'/online_combined_relres_hist')[k][:4]
        else:
            h=arrays(file,group+'/online_precorr_relres_hist')[k]
        it=int(arrays(file,group+'/online_fp_iter')[0][k])
        w=arrays(file,group+'/online_true_helm_relres_hist')[k] if it else np.array([])
        curves.append((name,h,w,color,marker))
    plt.rcParams.update({'font.size':9, 'pdf.fonttype':42, 'axes.spines.top':False,
                         'axes.spines.right':False})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,3.5),width_ratios=[1.55,1],
                            layout='constrained',gridspec_kw={'wspace':.08})
    def draw(x,y,c,m,label):
        ax.semilogy(x,y,color=c,marker=m,markersize=3,markerfacecolor='white',
                    markevery=np.unique(np.linspace(0,len(y)-1,min(15,len(y))).round().astype(int)),
                    linewidth=1.25,label=label)
    for name,h,w,c,m in curves:
        n=len(h)-1
        draw(np.arange(n+1),h,c,m,name)
        if len(w):
            ax.plot([n,n],[h[-1],w[0]],':',color=c,linewidth=.7)
            draw(n+np.arange(len(w)),w,c,m,None)
            ax.plot(n,w[0],'p',color=c,markersize=5,markerfacecolor='white')
    for key,c,label in [('plain_true_helm_relres_hist','#262626','Plain WaveHoltz'),
                         ('helm_baseline_relres_hist','#660080','Plain Helmholtz GMRES')]:
        w=arrays(d,'/diagnostic/global/'+key)[0]
        draw(np.arange(len(w)),w,c,'o',label)
    ax.set(xlabel='Online step index',ylabel='Relative residual',ylim=(1e-15,2),
           yticks=10.**np.arange(-15,1,5))
    ax.set_title('(a) Convergence',loc='left',fontsize=10)
    ax.legend(fontsize=7.8,loc='upper right',framealpha=.95,borderpad=.35,handlelength=1.5)
    ax.grid(alpha=.2)
    rows=list(csv.DictReader((DATA/'layer_storage.csv').open()))
    for m,offset,c in [('global',-.18,'#d90d0d'),('dd',.18,'#0d7340')]:
        y=np.array([float(r[m+'_entries'])*16/2**20 for r in rows])
        bx.bar(np.arange(1,4)+offset,y,.34,color=c,label='Global' if m=='global' else 'DD')
    bx.set(xticks=[1,2,3],xlabel='Layer',ylabel='Basis storage (MiB)',ylim=(0,13))
    bx.set_title('(b) Layerwise storage',loc='left',fontsize=10)
    bx.legend(frameon=False,fontsize=8,loc='upper left')
    bx.grid(axis='y',alpha=.2);bx.set_axisbelow(True)
    save(fig,'helm_convergence_layer_storage');plt.close(fig)
    fig,ax=plt.subplots(figsize=(3,2.55),layout='constrained')
    physical=np.array([578125,414750])*16/2**20
    pml=np.array([743552,486384])*16/2**20
    ax.bar([0,1],physical,.6,color='#407ab8',label='Physical')
    ax.bar([0,1],pml,.6,bottom=physical,color='#bfbfbf',label='PML')
    ax.set(xticks=[0,1],xticklabels=['Global','DD'],ylabel='Basis storage (MiB)',ylim=(0,27))
    ax.legend(frameon=False,ncol=2,fontsize=8,loc='upper center')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    save(fig,'helm_regional_storage');plt.close(fig)
    print('Saved compact convergence/layer and regional storage figures.')
if __name__=='__main__': main()
