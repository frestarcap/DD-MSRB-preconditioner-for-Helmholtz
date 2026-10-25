#!/usr/bin/env python3
"""ECC region map and explanatory DD schematic on equal square axes."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Circle, Ellipse, Rectangle
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'JCP/figures/jcp_shifted'
x=np.loadtxt(ROOT/'outputs/ecc_dd_budget_20260919/paper_x.csv',delimiter=',')
y=np.loadtxt(ROOT/'outputs/ecc_dd_budget_20260919/paper_y.csv',delimiter=',')
X,Y=np.meshgrid(x,y);physical=(X>0)&(X<1)&(Y>0)&(Y<1)
region=np.full(X.shape,3);region[physical]=2;region[physical&(np.hypot(X-.5,Y-.5)<=.24)]=1
colors=['#8ebcda','#f2e3c4','#c8c8c8']
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(8,3.8),layout='constrained')
axs[0].imshow(region,origin='lower',extent=[x[0],x[-1],y[0],y[-1]],
              cmap=ListedColormap(colors),norm=BoundaryNorm([.5,1.5,2.5,3.5],3),interpolation='nearest',rasterized=True)
axs[0].add_patch(Rectangle((0,0),1,1,fill=False,ec='.3',lw=.8))
axs[0].set_title('(a) Region map')
ax=axs[1]
ax.add_patch(Rectangle((x[0],y[0]),x[-1]-x[0],y[-1]-y[0],fc=colors[2],ec='.3',lw=.8))
ax.add_patch(Rectangle((0,0),1,1,fc=colors[1],ec='.3',lw=.8))
ax.add_patch(Circle((.5,.5),.24,fc=colors[0],ec='#32678e',lw=1))
a=.12*np.sqrt(.909176375080444);b=.12/np.sqrt(.909176375080444)
ax.add_patch(Ellipse((.5,.5),2*a,2*b,fc='white',ec='black',lw=.9))
ax.annotate('',xy=(.5+.24/np.sqrt(2),.5+.24/np.sqrt(2)),xytext=(.5,.5),arrowprops={'arrowstyle':'->','lw':.9})
ax.text(.64,.74,r'$r=0.24$',ha='center',fontsize=9)
ax.text(.5,.48,'Obstacle',ha='center',va='center',fontsize=8)
ax.set_title('(b) DD blocks (schematic)')
for ax in axs:
 ax.text(.5,1.16,'3: PML',ha='center',fontsize=10)
 ax.text(.5,.89,'2: Remaining physical region',ha='center',fontsize=9)
 ax.text(.5,.32,'1: Near field',ha='center',fontsize=9)
 ax.set(xlim=(x[0],x[-1]),ylim=(y[0],y[-1]),xlabel='$x$',ylabel='$y$',xticks=[0,.5,1],yticks=[0,.5,1],aspect='equal')
 ax.set_box_aspect(1)
for ext in ['pdf','png']:fig.savefig(OUT/f'ecc_partition_explained.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)
