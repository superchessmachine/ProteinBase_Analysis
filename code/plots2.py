import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from scipy.stats import spearmanr
plt.rcParams.update({'figure.dpi':170,'savefig.dpi':170,'font.size':9,
 'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
 'grid.alpha':.22,'grid.linewidth':.6,'axes.axisbelow':True})
OUT='proteinbase_data/figures'; os.makedirs(OUT,exist_ok=True)
HIT='#2A9D8F'; NO='#E76F51'; PH='#264653'; ACC='#7B4B94'; REF='#B0B0B0'
R=pd.read_csv('mech2_scored.csv'); U=pd.read_csv('univariate_benchmark.csv')
CF=pd.read_csv('lasso_coefficients.csv')
y=R.y.values; base=y.mean()
SUB=(f'Nipah glycoprotein-G round · {len(R)} designs · {y.sum()} binders ({base:.1%}) · '
     f'{R.family.nunique()} sequence families · physics-only descriptors')
PRETTY={'buns':'buried unsat. polars','buns_density':'buried unsat. polars /kÅ²',
 'buns_loose_density':'buried unsat. polars /kÅ² (loose)','void':'interface void (Å³)',
 'void_loose':'interface void, loose (Å³)','void_per_bsa':'void per 100 Å² BSA','gap_index':'gap volume index',
 'sc':'shape complementarity','bsa':'buried surface area (Å²)','bsa_apolar':'apolar BSA (Å²)',
 'n_salt':'salt bridges','n_hb':'interface H-bonds','hb_energy':'H-bond energy',
 'anchor_bsa':'deepest residue burial (Å²)','elec':'Coulomb energy','elec_att':'attractive electrostatics',
 'lj_atr':'LJ attractive','lj_rep':'LJ repulsive','dG_solv':'desolvation ΔG (Eisenberg)',
 'ddg_est':'estimated ΔΔG','ddg_per_bsa':'ΔΔG per kÅ² BSA','planarity':'interface planarity',
 'eccentricity':'interface eccentricity','n_segments':'interface segments','rg_norm':'compactness Rg/Rg₀',
 'frac_helix':'binder helix fraction','frac_sheet':'binder sheet fraction','agg_max':'max hydrophobic window',
 'surf_hydrophobic_frac':'exposed hydrophobic fraction','n_cation_pi':'cation-π','n_anchor50':'anchor residues >50 Å²',
 'ct_density':'apolar contact density','rel_contact_order':'relative contact order','aliphatic_index':'aliphatic index'}
def lab(c): return PRETTY.get(c,c.replace('_',' '))

# ---- FIG 1 : univariate AUROC -------------------------------------------
T=U.sort_values('AUROC').tail(28)
fig,ax=plt.subplots(figsize=(7.6,8.6))
ax.barh([lab(c) for c in T.descriptor],T.AUROC-.5,left=.5,
        xerr=[T.AUROC-T.ci_lo,T.ci_hi-T.AUROC],color=PH,height=.74,
        error_kw=dict(ecolor='#9AB',lw=.9,capsize=2))
ax.axvline(.5,color='#444',lw=1.1)
ax.set_xlim(.44,max(.75,T.ci_hi.max()+.02)); ax.set_xlabel('AUROC  (0.5 = no discrimination; bars = 95% bootstrap CI)')
for i,(v,d) in enumerate(zip(T.AUROC,T.direction)):
    ax.text(.447,i,'↑' if d.startswith('higher') else '↓',va='center',fontsize=8,color='#666')
ax.set_title('Single mechanistic descriptors vs experimental binding',loc='left',fontsize=12,weight='bold')
ax.text(0,1.015,SUB,transform=ax.transAxes,fontsize=7.6,color='#555')
fig.tight_layout(); fig.savefig(f'{OUT}/01_single_descriptor_auroc.png'); plt.close(fig)

# ---- FIG 2 : distributions ----------------------------------------------
pick=[c for c in U.descriptor.head(8)]
fig,axes=plt.subplots(2,4,figsize=(13.2,6.2))
for ax,c in zip(axes.ravel(),pick):
    a=pd.to_numeric(R.loc[R.y==1,c],errors='coerce').dropna(); b=pd.to_numeric(R.loc[R.y==0,c],errors='coerce').dropna()
    lo,hi=np.nanpercentile(pd.concat([a,b]),[1,99])
    if not np.isfinite(lo) or hi<=lo: continue
    bins=np.linspace(lo,hi,24)
    ax.hist(b,bins=bins,color=NO,alpha=.5,density=True,label=f'non-binder ({len(b)})')
    ax.hist(a,bins=bins,color=HIT,alpha=.75,density=True,label=f'binder ({len(a)})')
    ax.axvline(b.median(),color=NO,ls='--',lw=1.3); ax.axvline(a.median(),color=HIT,ls='--',lw=1.3)
    r=U[U.descriptor==c].iloc[0]
    ax.set_title(f'{lab(c)}\nAUROC={r.AUROC:.3f}  ({r.direction})',fontsize=8.3,loc='left')
    ax.set_yticks([]); ax.tick_params(labelsize=7); ax.legend(frameon=False,fontsize=6.3)
fig.suptitle('Strongest mechanistic descriptors: binders vs non-binders   (dashed = median)',x=.006,ha='left',fontsize=12,weight='bold')
fig.text(.006,.945,SUB,fontsize=7.6,color='#555')
fig.tight_layout(rect=[0,0,1,.935]); fig.savefig(f'{OUT}/02_descriptor_distributions.png'); plt.close(fig)

# ---- FIG 3 : ROC / PR ----------------------------------------------------
curves=[('L1 logistic, all physics','score_l1',PH),('L1 logistic, curated physics','score_core',ACC)]
fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.7))
for lbl,c,col in curves:
    if c not in R: continue
    v=R[c].values
    fpr,tpr,_=roc_curve(y,v); pr,rc,_=precision_recall_curve(y,v)
    a1.plot(fpr,tpr,color=col,lw=2,label=f'{lbl} — AUROC {roc_auc_score(y,v):.3f}')
    a2.plot(rc,pr,color=col,lw=2,label=f'{lbl} — AUPRC {average_precision_score(y,v):.3f}')
bestU=U.iloc[0]
sgn=1 if bestU.direction.startswith('higher') else -1
v=sgn*pd.to_numeric(R[bestU.descriptor],errors='coerce').fillna(0).values
fpr,tpr,_=roc_curve(y,v); pr,rc,_=precision_recall_curve(y,v)
a1.plot(fpr,tpr,color=REF,lw=1.6,ls='--',label=f'best single: {lab(bestU.descriptor)} — {bestU.AUROC:.3f}')
a2.plot(rc,pr,color=REF,lw=1.6,ls='--',label=f'best single: {lab(bestU.descriptor)} — {bestU.AUPRC:.3f}')
a1.plot([0,1],[0,1],'--',color='#ccc',lw=1); a1.set_xlabel('false positive rate'); a1.set_ylabel('true positive rate')
a1.set_title('ROC',loc='left',fontsize=10); a1.legend(frameon=False,fontsize=7.2,loc='lower right')
a2.axhline(base,ls='--',color='#ccc',lw=1); a2.text(.5,base+.012,f'random = {base:.3f}',fontsize=7,color='#888')
a2.set_xlabel('recall'); a2.set_ylabel('precision'); a2.set_ylim(0,1)
a2.set_title('Precision-Recall',loc='left',fontsize=10); a2.legend(frameon=False,fontsize=7.2,loc='upper right')
fig.suptitle('Nested family-grouped cross-validation — the honest estimate',x=.006,ha='left',fontsize=12,weight='bold')
fig.text(.006,.915,'No sequence family appears in both train and test; L1 penalty chosen on an inner grouped split.',fontsize=7.6,color='#555')
fig.text(.006,.877,SUB,fontsize=7.2,color='#777')
fig.tight_layout(rect=[0,0,1,.865]); fig.savefig(f'{OUT}/03_roc_pr.png'); plt.close(fig)

# ---- FIG 4 : leakage -----------------------------------------------------
if 'score_leaky' in R:
    fig,ax=plt.subplots(1,2,figsize=(10.2,4.2))
    for a,fn,ttl in [(ax[0],average_precision_score,'AUPRC'),(ax[1],roc_auc_score,'AUROC')]:
        vals=[fn(y,R.score_leaky.values),fn(y,R.score_l1.values)]
        a.bar([0,1],vals,.5,color=['#C9C9C9',PH],edgecolor=['#888',PH])
        for i,v in enumerate(vals): a.text(i,v+.012,f'{v:.3f}',ha='center',fontsize=9,weight='bold')
        a.set_xticks([0,1]); a.set_xticklabels(['random CV\n(families leak)','family-grouped CV\n(honest)'],fontsize=8)
        a.set_ylim(0,max(vals)*1.25); a.set_title(ttl,loc='left',fontsize=10)
        if ttl=='AUPRC': a.axhline(base,ls='--',color='#bbb',lw=1); a.text(.5,base+.008,'random',fontsize=7,color='#888')
    fig.suptitle('Random cross-validation overstates every binder filter on this data',x=.006,ha='left',fontsize=12,weight='bold')
    fig.text(.006,.905,'The largest sequence family holds 82 designs and 23 of the 101 binders. Split at random, a model memorises families, not physics.',fontsize=7.4,color='#555')
    fig.tight_layout(rect=[0,0,1,.89]); fig.savefig(f'{OUT}/04_leakage.png'); plt.close(fig)

# ---- FIG 5 : lasso coefficients -----------------------------------------
C=CF.sort_values('coef')
fig,ax=plt.subplots(figsize=(7.4,max(3.2,.32*len(C)+1.6)))
ax.barh([lab(c) for c in C.descriptor],C.coef,color=[HIT if v>0 else NO for v in C.coef],height=.7)
ax.axvline(0,color='#444',lw=1.1)
ax.set_xlabel('L1 logistic coefficient  (standardised units; + = favours binding)')
ax.set_title(f'What the sparse physics model actually uses  ({len(C)} of {len(U)} descriptors survive L1)',loc='left',fontsize=11.5,weight='bold')
ax.text(0,1.012,SUB,transform=ax.transAxes,fontsize=7.4,color='#555')
fig.tight_layout(); fig.savefig(f'{OUT}/05_lasso_coefficients.png'); plt.close(fig)

# ---- FIG 6 : enrichment + packing map ------------------------------------
fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.4))
ks=np.arange(5,min(350,len(R)),5)
for lbl,c,col in curves:
    if c not in R: continue
    o=np.argsort(-R[c].values); a1.plot(ks,[y[o[:k]].mean() for k in ks],color=col,lw=2,label=lbl)
a1.axhline(base,ls='--',color='#bbb',lw=1.2); a1.text(230,base+.012,f'no filter = {base:.1%}',fontsize=7.4,color='#888')
a1.set_xlabel('designs submitted (ranked best-first)'); a1.set_ylabel('fraction that bind')
a1.set_title('Expected hit rate vs submission depth',loc='left',fontsize=10)
a1.legend(frameon=False,fontsize=7.4); a1.yaxis.set_major_formatter(lambda v,p:f'{v:.0%}')
xa,ya='buns_loose_density','void_loose'
if {xa,ya}.issubset(R.columns):
    bq=pd.qcut(pd.to_numeric(R[xa],errors='coerce'),4,duplicates='drop')
    vq=pd.qcut(pd.to_numeric(R[ya],errors='coerce'),4,duplicates='drop')
    piv=R.groupby([bq,vq]).y.mean().unstack()
    im=a2.imshow(piv.values,cmap='viridis',aspect='auto',origin='lower')
    a2.set_xticks(range(piv.shape[1])); a2.set_xticklabels([f'{i.left:.0f}–{i.right:.0f}' for i in piv.columns],fontsize=6.6,rotation=22)
    a2.set_yticks(range(piv.shape[0])); a2.set_yticklabels([f'{i.left:.1f}–{i.right:.1f}' for i in piv.index],fontsize=6.6)
    a2.set_xlabel(f'{lab(ya)}  →'); a2.set_ylabel(f'{lab(xa)}  →')
    a2.set_title('Hit rate by interface packing quality',loc='left',fontsize=10); a2.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v=piv.values[i,j]
            if v==v: a2.text(j,i,f'{v:.0%}',ha='center',va='center',color='w',fontsize=7.4,weight='bold')
    fig.colorbar(im,ax=a2,label='hit rate')
fig.suptitle('What the physics score buys you in practice',x=.006,ha='left',fontsize=12,weight='bold')
fig.text(.006,.915,SUB,fontsize=7.4,color='#555')
fig.tight_layout(rect=[0,0,1,.90]); fig.savefig(f'{OUT}/06_enrichment.png'); plt.close(fig)

# ---- FIG 7 : affinity ranking + confound ---------------------------------
try:
    B=pd.read_csv('affinity_scored.csv')
    UD=pd.read_csv('affinity_univariate.csv')
    import json as _j
    AS=_j.load(open('affinity_summary.json'))
    fig,axes=plt.subplots(1,3,figsize=(12.4,3.9))
    a=axes[0]
    a.scatter(B.kd_pred,B.logkd,s=20,color=ACC,alpha=.72,edgecolor='none')
    z=np.polyfit(B.kd_pred,B.logkd,1); xs=np.linspace(B.kd_pred.min(),B.kd_pred.max(),20)
    a.plot(xs,np.polyval(z,xs),color=NO,lw=1.7)
    a.set_xlabel('predicted log$_{10}$ K$_D$ (grouped CV)'); a.set_ylabel('measured log$_{10}$ K$_D$ (M)')
    a.set_title(f"Ridge on physics\nr = {AS['pearson']:.3f}, rho = {AS['spearman']:.3f}  (n={AS['n']})",loc='left',fontsize=9)
    a=axes[1]
    T2=UD.head(10).iloc[::-1]
    a.barh([lab(c) for c in T2.descriptor],T2.r,color=[HIT if v<0 else NO for v in T2.r],height=.7)
    a.axvline(0,color='#444',lw=1)
    a.set_xlabel('Pearson r vs log$_{10}$ K$_D$'); a.set_title('Single descriptors vs affinity',loc='left',fontsize=9)
    a.tick_params(labelsize=7)
    a=axes[2]
    strat=[('all binders',0.503),('de novo only\n(Other+Mini)',0.186),('"Other" only\n(no scFv)',-0.302)]
    a.bar(range(3),[v for _,v in strat],.55,color=[ACC,'#9A8FBF',NO])
    for i,(_,v) in enumerate(strat): a.text(i,v+(.02 if v>0 else -.05),f'{v:.3f}',ha='center',fontsize=8.5,weight='bold')
    a.axhline(0,color='#444',lw=1); a.set_xticks(range(3)); a.set_xticklabels([k for k,_ in strat],fontsize=7.4)
    a.set_ylabel('Pearson r (grouped CV)'); a.set_ylim(-.45,.62)
    a.set_title('Signal vanishes within one design format',loc='left',fontsize=9)
    fig.suptitle(f"Ranking affinity among the {AS['n']} binders with measured K$_D$ — a negative result",
                 x=.006,ha='left',fontsize=12,weight='bold')
    fig.text(.006,.905,'The model ranks affinity only because design formats differ in affinity. '
             'Stratified within a format the correlation disappears: physics does not rank K$_D$ here.',fontsize=7.4,color='#555')
    fig.tight_layout(rect=[0,0,1,.88]); fig.savefig(f'{OUT}/07_affinity_ranking.png'); plt.close(fig)
except Exception as ex:
    print('fig7 skipped:',ex)
print('figures:',sorted(os.listdir(OUT)))
