"""Fig 09_1: predicted metric vs experimental affinity, non-binders pinned at 0.

y = pK_D = -log10(K_D); non-binders set to 0 ("no measurable affinity").
A raw K_D axis cannot take 0 for non-binders: 0 M would plot below the tightest
binder and read as infinite affinity.

Panels are ordered by AUPRC, best first, reading left-to-right then down.
The shaded band rising from y=0 is the non-binder density along x (scaled to the
empty region between 0 and the weakest binder) so the pile-up at 0 is legible.
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score
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

pkd=np.full(len(D),np.nan)
pkd[y==0]=0.0
mb=(y==1)&D.kd.notna().values&(D.kd.values>0)
pkd[mb]=-np.log10(D.kd.values[mb])
D['pkd']=pkd
keep=D.pkd.notna()
PKD_MIN=float(np.nanmin(pkd[mb]))
print(f"plotted={int(keep.sum())}  binders with Kd={int(mb.sum())}  "
      f"non-binders at 0={int((y==0).sum())}  weakest binder pK_D={PKD_MIN:.2f}")

PRETTY={'buns_loose_density':'buried unsat. polars /kÅ²','buns_loose':'buried unsat. polars',
 'sc':'shape complementarity (proxy)','bsa':'buried surface area','bsa_apolar':'apolar BSA',
 'void_loose':'interface void','ddg_est':'estimated ΔΔG','ddg_per_bsa':'ΔΔG / BSA',
 'hb_energy':'H-bond energy','ifc_res':'interface residues',
 'ros_unsat_per_dSASA':'Δ unsat. H-bonds / dSASA','ros_unsat':'Δ unsat. H-bonds',
 'ros_sc':'shape complementarity','ros_dSASA':'dSASA','ros_dhSASA':'hydrophobic dSASA',
 'ros_packstat':'packstat','ros_dG':'ΔG separated','ros_dG_dSASA':'ΔG / dSASA',
 'ros_hb_E':'H-bond energy','ros_nres':'interface residues'}
def lab(c): return PRETTY.get(c,c.replace('ros_','').replace('_',' '))

# ---- score every panel first, then order by AUPRC -------------------------
cand=[]
for _,r in HH.iterrows():
    cand += [(r.mine,'Mine'),(r.rosetta,'Rosetta')]
rows=[]
for col,who in cand:
    v=pd.to_numeric(D[col],errors='coerce')
    ok=keep.values&v.notna().values
    if ok.sum()<100: continue
    xv=v.values[ok]; yy=y[ok]
    a=roc_auc_score(yy,xv); sg=1 if a>=.5 else -1
    rows.append(dict(metric=col,source=who,ok=ok,sign=sg,
                     AUROC=roc_auc_score(yy,sg*xv),
                     AUPRC=average_precision_score(yy,sg*xv),
                     pearson_all=pearsonr(xv,D.pkd.values[ok])[0]))
P=pd.DataFrame(rows).sort_values('AUPRC',ascending=False).reset_index(drop=True)

ncol=4; nrow=int(np.ceil(len(P)/ncol))
fig,axes=plt.subplots(nrow,ncol,figsize=(3.35*ncol,2.95*nrow))
rng=np.random.default_rng(0)
DENS_TOP=PKD_MIN*0.82          # density band fills the empty region only
for k,(ax,(_,r)) in enumerate(zip(np.ravel(axes),P.iterrows())):
    v=pd.to_numeric(D[r.metric],errors='coerce')
    ok=r.ok
    xv=v.values[ok]; yv=D.pkd.values[ok]; yy=y[ok]
    lo,hi=np.nanpercentile(xv,[1,99])
    if not np.isfinite(lo) or hi<=lo: ax.axis('off'); continue

    # non-binder density along x, scaled into the empty band
    nb=xv[yy==0]
    uniq=np.unique(np.round(xv,6))
    is_int=np.allclose(uniq,np.round(uniq))
    if is_int and len(uniq)<=30:
        # integer-valued metric: align bin edges to integer boundaries so the
        # histogram does not comb from edges straddling integers
        bins=np.arange(np.floor(lo)-.5,np.ceil(hi)+1.5,1.0)
        cnt,edges=np.histogram(nb,bins=bins)
    else:
        cnt,edges=np.histogram(nb,bins=int(min(28,max(8,len(uniq)))),range=(lo,hi))
    ctr=(edges[:-1]+edges[1:])/2
    if cnt.max()>0:
        ax.fill_between(ctr,0,cnt/cnt.max()*DENS_TOP,color=NO,alpha=.22,
                        lw=0,zorder=1)
        ax.plot(ctr,cnt/cnt.max()*DENS_TOP,color=NO,lw=1.0,alpha=.75,zorder=2)
    bx=xv[yy==1]
    ax.scatter(xv[yy==0],rng.normal(0,.045,int((yy==0).sum())),s=5,color=NO,
               alpha=.28,edgecolor='none',zorder=3)
    ax.scatter(bx,yv[yy==1],s=17,color=HIT,alpha=.9,edgecolor='none',zorder=4)
    ax.axhline(0,color='#999',lw=.9,zorder=2)

    ax.set_xlim(lo,hi); ax.set_ylim(-.5,10.2)
    c=MINE if r.source=='Mine' else ROS
    ax.set_xlabel(f'{r.source}: {lab(r.metric)}',fontsize=7.6,color=c)
    ax.set_ylabel('pK$_D$  (0 = non-binder)',fontsize=7.4)
    ax.set_title(f'{k+1}.  AUPRC={r.AUPRC:.3f}   AUROC={r.AUROC:.3f}   r={r.pearson_all:.2f}',
                 fontsize=8,loc='left')
    ax.tick_params(labelsize=6.8)
for ax in np.ravel(axes)[len(P):]: ax.axis('off')

h=[plt.Line2D([],[],marker='o',ls='',color=HIT,ms=6),
   plt.Line2D([],[],marker='o',ls='',color=NO,ms=6),
   plt.Rectangle((0,0),1,1,color=NO,alpha=.22)]
fig.legend(h,[f'binder ({int(mb.sum())}, measured K$_D$)',
              f'non-binder ({int((y==0).sum())}, pinned at 0)',
              'non-binder density along x'],
           frameon=False,fontsize=8.4,ncol=3,loc='upper right',bbox_to_anchor=(.997,.997))
H=fig.get_figheight()
fig.suptitle('Predicted metric vs experimental affinity',x=.012,ha='left',
             y=1-0.26/H,va='top',fontsize=12,weight='bold')
fig.text(.012,1-0.52/H,
 f'Ordered by AUPRC, best first (random = {y.mean():.3f}).  '
 r'y = pK$_D$ = -log$_{10}$K$_D$, non-binders at 0.  r = Pearson over all designs.  '
 'Shaded curve = non-binder density along x, scaled to fill the empty band — not pK$_D$.',
 fontsize=7.6,color='#666',va='top')
fig.tight_layout(rect=[0,0,1,1-0.82/H])
fig.savefig(f'{OUT}/09_1_metric_vs_affinity.png'); plt.close(fig)

P.drop(columns=['ok','sign']).to_csv('metric_vs_affinity.csv',index=False)
print("\n=== ordered by AUPRC ===")
print(P.drop(columns=['ok','sign']).to_string(index=False,float_format=lambda v:f"{v:.3f}"))
