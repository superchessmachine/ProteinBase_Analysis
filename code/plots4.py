"""Comparison figures: Rosetta InterfaceAnalyzer vs hand-built mechanistic descriptors."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np, os, json, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from scipy.stats import spearmanr
plt.rcParams.update({'figure.dpi':170,'savefig.dpi':170,'font.size':9,
 'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
 'grid.alpha':.2,'grid.linewidth':.6,'axes.axisbelow':True,
 'axes.titlesize':10,'axes.titleweight':'normal','axes.labelsize':9})
OUT='proteinbase_data/figures'; os.makedirs(OUT,exist_ok=True)
HIT='#2A9D8F'; NO='#E76F51'; MINE='#264653'; ROS='#E9A13B'; BOTH='#7B4B94'

D=pd.read_csv('compare_scored.csv')
RU=pd.read_csv('rosetta_univariate.csv')
HH=pd.read_csv('head_to_head.csv')
MU=pd.read_csv('univariate_benchmark.csv')
S=json.load(open('compare_summary.json'))
y=D.y.values; base=y.mean()
N=f'n = {len(D)} designs · {y.sum()} binders ({base:.1%})'

RP={'ros_sc':'shape complementarity','ros_dG':'ΔG separated','ros_dG_dSASA':'ΔG / dSASA',
 'ros_dSASA':'dSASA','ros_dhSASA':'hydrophobic dSASA','ros_packstat':'packstat',
 'ros_unsat':'Δ unsat. H-bonds','ros_unsat_per_dSASA':'Δ unsat. H-bonds / dSASA',
 'ros_hb_E':'H-bond energy','ros_hb_E_frac':'H-bond E fraction','ros_hbonds':'interface H-bonds',
 'ros_hb_per_dSASA':'H-bond E / dSASA','ros_nres':'interface residues','ros_crossterm':'crossterm energy',
 'ros_crossterm_dSASA':'crossterm / dSASA','ros_complex_E':'complex total energy',
 'ros_complexed_SASA':'complexed SASA','ros_separated_SASA':'separated SASA'}
MP={'buns_loose_density':'buried unsat. polars /kÅ²','buns_loose':'buried unsat. polars',
 'sc':'shape complementarity (proxy)','bsa':'buried surface area','bsa_apolar':'apolar BSA',
 'void_loose':'interface void','ddg_est':'estimated ΔΔG','ddg_per_bsa':'ΔΔG / BSA',
 'hb_energy':'H-bond energy','ifc_res':'interface residues'}
def rl(c): return RP.get(c,c.replace('ros_','').replace('_',' '))
def ml(c): return MP.get(c,c.replace('_',' '))
def head(fig,t,sub=None):
    H=fig.get_figheight()
    fig.suptitle(t,x=.012,ha='left',y=1-0.26/H,va='top',fontsize=12,weight='bold')
    if sub: fig.text(.012,1-0.52/H,sub,fontsize=8,color='#666',va='top')
    return 1-(0.78 if sub else 0.52)/H

# ===== FIG 8 : Rosetta metric ranking =====
T=RU.sort_values('AUROC')
fig,ax=plt.subplots(figsize=(6.6,6.2))
ax.barh([rl(c) for c in T.metric],T.AUROC-.5,left=.5,
        xerr=[T.AUROC-T.ci_lo,T.ci_hi-T.AUROC],color=ROS,height=.7,
        error_kw=dict(ecolor='#C9A97A',lw=.9,capsize=2))
ax.axvline(.5,color='#333',lw=1)
ax.set_xlim(.45,.77); ax.set_xlabel('AUROC'); ax.tick_params(labelsize=8)
top=head(fig,'Rosetta InterfaceAnalyzer metrics',N+'   ·   bars = 95% CI')
fig.tight_layout(rect=[0,0,1,top]); fig.savefig(f'{OUT}/08_rosetta_metric_auroc.png'); plt.close(fig)

# ===== FIG 9 : correlation scatters, mine vs Rosetta =====
n=len(HH); ncol=5; nrow=int(np.ceil(n/ncol))
fig,axes=plt.subplots(nrow,ncol,figsize=(3.0*ncol,2.9*nrow))
for ax,(_,r) in zip(np.ravel(axes),HH.iterrows()):
    a=pd.to_numeric(D[r.mine],errors='coerce'); b=pd.to_numeric(D[r.rosetta],errors='coerce')
    ok=a.notna()&b.notna()
    aa,bb=a[ok],b[ok]
    for lo,hi,c,lab_ in [(0,0,NO,'non-binder'),(1,1,HIT,'binder')]:
        m=(D.y[ok]==lo)
        ax.scatter(aa[m],bb[m],s=7,color=c,alpha=.45 if lo==0 else .85,edgecolor='none',
                   zorder=2 if lo==0 else 3)
    lo_a,hi_a=np.nanpercentile(aa,[1,99]); lo_b,hi_b=np.nanpercentile(bb,[1,99])
    ax.set_xlim(lo_a,hi_a); ax.set_ylim(lo_b,hi_b)
    if ok.sum()>5:
        z=np.polyfit(aa,bb,1); xs=np.linspace(lo_a,hi_a,20)
        ax.plot(xs,np.polyval(z,xs),color='#333',lw=1.2,ls='--',zorder=4)
    ax.set_xlabel(f'mine: {ml(r.mine)}',fontsize=7.5)
    ax.set_ylabel(f'Rosetta: {rl(r.rosetta)}',fontsize=7.5)
    ax.set_title(f'ρ = {r.agreement_rho:.2f}    AUROC {r.AUROC_mine:.2f} / {r.AUROC_rosetta:.2f}',
                 fontsize=8,loc='left')
    ax.tick_params(labelsize=6.5)
for ax in np.ravel(axes)[n:]: ax.axis('off')
h=[plt.Line2D([],[],marker='o',ls='',color=HIT,ms=5),plt.Line2D([],[],marker='o',ls='',color=NO,ms=5)]
fig.legend(h,['binder','non-binder'],frameon=False,fontsize=8.5,ncol=2,loc='upper right',bbox_to_anchor=(.995,.995))
top=head(fig,'Agreement: my descriptors vs Rosetta','ρ = Spearman   ·   AUROC shown as mine / Rosetta   ·   '+N)
fig.tight_layout(rect=[0,0,1,top]); fig.savefig(f'{OUT}/09_correlation_scatters.png'); plt.close(fig)

# ===== FIG 10 : method comparison, AUROC + AUPRC =====
methods=[('Rosetta only','score_ros',ROS),('Mine only','score_mine',MINE),('Both combined','score_both',BOTH)]
fig,(a1,a2)=plt.subplots(1,2,figsize=(9.4,4.2))
x=np.arange(len(methods)); w=.6
au=[roc_auc_score(y,D[c]) for _,c,_ in methods]
ap=[average_precision_score(y,D[c]) for _,c,_ in methods]
for ax,vals,ttl in [(a1,au,'AUROC'),(a2,ap,'AUPRC')]:
    ax.bar(x,vals,w,color=[c for *_,c in methods])
    for i,v in enumerate(vals): ax.text(i,v+max(vals)*.02,f'{v:.3f}',ha='center',fontsize=10,weight='bold')
    ax.set_xticks(x); ax.set_xticklabels([m for m,_,_ in methods],fontsize=8.5)
    ax.set_ylim(0,max(vals)*1.22); ax.set_title(ttl,loc='left')
    if ttl=='AUPRC':
        ax.axhline(base,ls=':',color='#999',lw=1.2)
        ax.text(len(methods)-1,base+max(vals)*.02,f'random {base:.3f}',fontsize=7,color='#888',ha='right')
    else: ax.axhline(.5,ls=':',color='#999',lw=1.2)
top=head(fig,'Method comparison','L1 logistic · nested family-grouped CV · '+N)
fig.tight_layout(rect=[0,0,1,top]); fig.savefig(f'{OUT}/10_method_comparison.png'); plt.close(fig)

# ===== FIG 11 : ROC + PR for all methods =====
fig,(a1,a2)=plt.subplots(1,2,figsize=(9.8,4.3))
allm=methods+[('Rosetta sc (best single)','ros_sc',"#B08968"),
              ('my ΔΔG/BSA (best single)','ddg_per_bsa','#5E8C7F')]
for lbl,c,col in allm:
    v=pd.to_numeric(D[c],errors='coerce').fillna(0).values
    if roc_auc_score(y,v)<.5: v=-v
    fpr,tpr,_=roc_curve(y,v); pr,rc,_=precision_recall_curve(y,v)
    ls='-' if c.startswith('score_') else '--'
    lw=2 if c.startswith('score_') else 1.4
    a1.plot(fpr,tpr,color=col,lw=lw,ls=ls,label=f'{lbl}  {roc_auc_score(y,v):.3f}')
    a2.plot(rc,pr,color=col,lw=lw,ls=ls,label=f'{lbl}  {average_precision_score(y,v):.3f}')
a1.plot([0,1],[0,1],'-',color='#E5E5E5',lw=1)
a1.set_xlabel('false positive rate'); a1.set_ylabel('true positive rate'); a1.set_title('ROC',loc='left')
a2.axhline(base,ls=':',color='#999',lw=1)
a2.set_xlabel('recall'); a2.set_ylabel('precision'); a2.set_ylim(0,1); a2.set_title('Precision–recall',loc='left')
a1.legend(frameon=False,fontsize=7.5,loc='lower right',title='AUROC',title_fontsize=7.5)
a2.legend(frameon=False,fontsize=7.5,loc='upper right',title='AUPRC',title_fontsize=7.5)
top=head(fig,'All methods','solid = fitted models, dashed = best single metric · '+N)
fig.tight_layout(rect=[0,0,1,top]); fig.savefig(f'{OUT}/11_all_methods_roc_pr.png'); plt.close(fig)

# ===== FIG 12 : head-to-head AUROC, paired =====
H=HH.sort_values('AUROC_mine')
fig,ax=plt.subplots(figsize=(7.4,4.6))
yy=np.arange(len(H)); h=.38
ax.barh(yy+h/2,H.AUROC_mine-.5,left=.5,height=h,color=MINE,label='mine (hand-built)')
ax.barh(yy-h/2,H.AUROC_rosetta-.5,left=.5,height=h,color=ROS,label='Rosetta')
ax.axvline(.5,color='#333',lw=1)
ax.set_yticks(yy); ax.set_yticklabels(H.concept,fontsize=8)
ax.set_xlim(.48,.72); ax.set_xlabel('AUROC')
for i,(m,r) in enumerate(zip(H.AUROC_mine,H.AUROC_rosetta)):
    ax.text(m+.003,i+h/2,f'{m:.2f}',va='center',fontsize=7)
    ax.text(r+.003,i-h/2,f'{r:.2f}',va='center',fontsize=7)
ax.legend(frameon=False,fontsize=8,loc='lower right')
top=head(fig,'Same concept, two implementations',N)
fig.tight_layout(rect=[0,0,1,top]); fig.savefig(f'{OUT}/12_head_to_head.png'); plt.close(fig)

print('comparison figures written')
for f in sorted(os.listdir(OUT)): print('  ',f)
