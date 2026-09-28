"""Fig 09_1: every matched metric vs experimental affinity, non-binders pinned at 0.

y = pK_D = -log10(K_D).  Non-binders are set to 0 ("no measurable affinity").
A raw K_D axis cannot take 0 for non-binders: 0 M would plot below the tightest
binder and read as infinite affinity. pK_D puts tighter binding higher and 0 at
the bottom, which is what "0 for non-binders" actually means.
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
plt.rcParams.update({'figure.dpi':170,'savefig.dpi':170,'font.size':9,
 'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
 'grid.alpha':.2,'grid.linewidth':.6,'axes.axisbelow':True,
 'axes.titlesize':10,'axes.titleweight':'normal','axes.labelsize':9})
OUT='proteinbase_data/figures'
HIT='#2A9D8F'; NO='#E76F51'; MINE='#264653'; ROS='#E9A13B'

D=pd.read_csv('compare_scored.csv')
HH=pd.read_csv('head_to_head.csv')
D['kd']=pd.to_numeric(D.kd,errors='coerce')
y=D.y.values

# pK_D: binders -> -log10(Kd); non-binders -> 0
pkd=np.full(len(D),np.nan)
pkd[y==0]=0.0
m=(y==1)&D.kd.notna().values&(D.kd.values>0)
pkd[m]=-np.log10(D.kd.values[m])
D['pkd']=pkd
n_drop=int(((y==1)&~m).sum())
keep=D.pkd.notna()
print(f"plotted: {int(keep.sum())}  binders with Kd: {int(m.sum())}  "
      f"non-binders at 0: {int((y==0).sum())}  binders dropped (no Kd): {n_drop}")

PRETTY={'buns_loose_density':'buried unsat. polars /kÅ²','buns_loose':'buried unsat. polars',
 'sc':'shape complementarity (proxy)','bsa':'buried surface area','bsa_apolar':'apolar BSA',
 'void_loose':'interface void','ddg_est':'estimated ΔΔG','ddg_per_bsa':'ΔΔG / BSA',
 'hb_energy':'H-bond energy','ifc_res':'interface residues',
 'ros_unsat_per_dSASA':'Δ unsat. H-bonds / dSASA','ros_unsat':'Δ unsat. H-bonds',
 'ros_sc':'shape complementarity','ros_dSASA':'dSASA','ros_dhSASA':'hydrophobic dSASA',
 'ros_packstat':'packstat','ros_dG':'ΔG separated','ros_dG_dSASA':'ΔG / dSASA',
 'ros_hb_E':'H-bond energy','ros_nres':'interface residues'}
def lab(c): return PRETTY.get(c,c.replace('ros_','').replace('_',' '))

# one panel per implementation, concepts kept adjacent
panels=[]
for _,r in HH.iterrows():
    panels.append((r.concept,r.mine,'Mine'))
    panels.append((r.concept,r.rosetta,'Rosetta'))

ncol=4; nrow=int(np.ceil(len(panels)/ncol))
fig,axes=plt.subplots(nrow,ncol,figsize=(3.35*ncol,2.95*nrow))
rng=np.random.default_rng(0)
stats=[]
for ax,(concept,col,who) in zip(np.ravel(axes),panels):
    v=pd.to_numeric(D[col],errors='coerce')
    ok=keep.values&v.notna().values
    xv=v.values[ok]; yv=D.pkd.values[ok]; yy=y[ok]
    lo,hi=np.nanpercentile(xv,[1,99])
    if not np.isfinite(lo) or hi<=lo: ax.axis('off'); continue
    # jitter the stacked non-binders so density is visible
    yj=yv.copy(); yj[yy==0]=rng.normal(0,.055,int((yy==0).sum()))
    ax.scatter(xv[yy==0],yj[yy==0],s=6,color=NO,alpha=.30,edgecolor='none',zorder=2)
    ax.scatter(xv[yy==1],yv[yy==1],s=17,color=HIT,alpha=.9,edgecolor='none',zorder=3)
    ax.axhline(0,color='#BBB',lw=.8,zorder=1)
    rho_all=spearmanr(xv,yv).correlation
    mb=yy==1
    rho_b=spearmanr(xv[mb],yv[mb]).correlation if mb.sum()>10 else np.nan
    au=roc_auc_score(yy,xv); au=max(au,1-au)
    if mb.sum()>3:
        z=np.polyfit(xv[mb],yv[mb],1); xs=np.linspace(lo,hi,20)
        ax.plot(xs,np.polyval(z,xs),color='#333',lw=1.2,ls='--',zorder=4)
    ax.set_xlim(lo,hi); ax.set_ylim(-.35,10.2)
    c=MINE if who=='Mine' else ROS
    ax.set_xlabel(f'{who}: {lab(col)}',fontsize=7.6,color=c)
    ax.set_ylabel('pK$_D$  (0 = non-binder)',fontsize=7.4)
    ax.set_title(f'ρ$_{{all}}$={rho_all:.2f}   ρ$_{{bind}}$={rho_b:.2f}   AUROC={au:.2f}',
                 fontsize=8,loc='left')
    ax.tick_params(labelsize=6.8)
    stats.append(dict(concept=concept,metric=col,source=who,rho_all=rho_all,
                      rho_binders_only=rho_b,AUROC=au))
for ax in np.ravel(axes)[len(panels):]: ax.axis('off')

h=[plt.Line2D([],[],marker='o',ls='',color=HIT,ms=6),
   plt.Line2D([],[],marker='o',ls='',color=NO,ms=6)]
fig.legend(h,[f'binder ({int(m.sum())}, measured K$_D$)',f'non-binder ({int((y==0).sum())}, pinned at 0)'],
           frameon=False,fontsize=8.5,ncol=2,loc='upper right',bbox_to_anchor=(.995,.995))
H=fig.get_figheight()
fig.suptitle('Predicted metric vs experimental affinity',x=.012,ha='left',
             y=1-0.26/H,va='top',fontsize=12,weight='bold')
fig.text(.012,1-0.52/H,
 'y = pK$_D$ = -log$_{10}$K$_D$; non-binders set to 0. '
 r'$\rho_{all}$ = Spearman over all designs, $\rho_{bind}$ = binders only. '
 'Non-binders jittered vertically for density.',
 fontsize=7.6,color='#666',va='top')
fig.tight_layout(rect=[0,0,1,1-0.80/H])
fig.savefig(f'{OUT}/09_1_metric_vs_affinity.png'); plt.close(fig)

S=pd.DataFrame(stats)
S.to_csv('metric_vs_affinity.csv',index=False)
print("\n=== correlation with pK_D ===")
print(S.reindex(S.rho_all.abs().sort_values(ascending=False).index)
       .to_string(index=False,float_format=lambda v:f"{v:.3f}"))
print("\nwrote figures/09_1_metric_vs_affinity.png")
