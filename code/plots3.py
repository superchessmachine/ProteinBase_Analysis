import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np, os, json, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
plt.rcParams.update({'figure.dpi':170,'savefig.dpi':170,'font.size':9,
 'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
 'grid.alpha':.2,'grid.linewidth':.6,'axes.axisbelow':True,
 'axes.titlesize':10,'axes.titleweight':'normal','axes.labelsize':9})
OUT='proteinbase_data/figures'; os.makedirs(OUT,exist_ok=True)
HIT='#2A9D8F'; NO='#E76F51'; PH='#264653'; ACC='#7B4B94'; REF='#BBBBBB'
R=pd.read_csv('mech2_scored.csv'); U=pd.read_csv('univariate_benchmark.csv')
CF=pd.read_csv('lasso_coefficients.csv')
y=R.y.values; base=y.mean()
N=f'n = {len(R)} designs · {y.sum()} binders ({base:.1%})'

P={'buns':'buried unsat. polars','buns_density':'buried unsat. polars /kÅ²',
 'buns_loose_density':'buried unsat. polars /kÅ²','buns_loose':'buried unsat. polars (loose)',
 'buns_loose_A':'buried unsat. polars (binder)','buns_A':'buried unsat. polars (binder)',
 'void':'interface void','void_loose':'interface void (loose)','void_per_bsa':'void / BSA',
 'gap_index':'gap volume index','sc':'shape complementarity','bsa':'buried surface area',
 'bsa_apolar':'apolar BSA','bsa_polar':'polar BSA','bsa_per_res':'BSA per interface residue',
 'n_salt':'salt bridges','n_hb':'interface H-bonds','hb_energy':'H-bond energy',
 'hb_density':'H-bond density','anchor_bsa':'deepest residue burial','top3_bsa':'top-3 residue burial',
 'elec':'Coulomb energy','elec_att':'attractive electrostatics','elec_rep':'repulsive electrostatics',
 'elec_per_bsa':'Coulomb / BSA','lj_atr':'LJ attractive','lj_rep':'LJ repulsive',
 'lj_atr_per_bsa':'LJ attractive / BSA','dG_solv':'desolvation ΔG','dG_solv_per_bsa':'desolvation ΔG / BSA',
 'ddg_est':'estimated ΔΔG','ddg_per_bsa':'ΔΔG / BSA','planarity':'interface planarity',
 'eccentricity':'interface eccentricity','n_segments':'interface segments','seg_per_res':'segments / residue',
 'rg':'radius of gyration','rg_norm':'compactness','frac_helix':'helix fraction','frac_sheet':'sheet fraction',
 'agg_max':'max hydrophobic window','surf_hydrophobic':'exposed hydrophobic area',
 'surf_hydrophobic_frac':'exposed hydrophobic fraction','n_cation_pi':'cation-π','n_pi_stack':'π-stacking',
 'n_anchor50':'anchor residues >50 Å²','n_anchor80':'anchor residues >80 Å²','n_clash':'steric clashes',
 'ct_density':'apolar contact density','n_apolar_ct':'apolar contacts','rel_contact_order':'relative contact order',
 'aliphatic_index':'aliphatic index','binder_len':'binder length','net_charge':'net charge',
 'charge_density':'charge density','frac_aromatic':'aromatic fraction','frac_charged':'charged fraction',
 'frac_hydrophobic':'hydrophobic fraction','n_cys':'cysteines','gravy':'GRAVY','ifc_res':'interface residues',
 'ifc_res_A':'interface residues (binder)','f_apolar':'apolar BSA fraction','salt_per_bsa':'salt bridges / BSA',
 'hb_per_bsa':'H-bond energy / BSA','anchor_frac':'anchor burial fraction'}
def L(c): return P.get(c,c.replace('_',' '))
def head(fig,t,sub=None):
    """Header with constant absolute spacing, independent of figure height."""
    H=fig.get_figheight()
    fig.suptitle(t,x=.012,ha='left',y=1-0.26/H,va='top',fontsize=12,weight='bold')
    if sub: fig.text(.012,1-0.52/H,sub,fontsize=8,color='#666',va='top')
    return 1-(0.78 if sub else 0.52)/H

# ---- FIG 1 : single-descriptor AUROC ------------------------------------
T=U.sort_values('AUROC').tail(24)
fig,ax=plt.subplots(figsize=(6.6,7.6))
ax.barh([L(c) for c in T.descriptor],T.AUROC-.5,left=.5,
        xerr=[T.AUROC-T.ci_lo,T.ci_hi-T.AUROC],color=PH,height=.7,
        error_kw=dict(ecolor='#A8BCC4',lw=.9,capsize=2))
ax.axvline(.5,color='#333',lw=1)
ax.set_xlim(.45,.77); ax.set_xlabel('AUROC')
ax.tick_params(labelsize=8)
top=head(fig,'Single mechanistic descriptors',N+'   ·   bars = 95% CI')
fig.tight_layout(rect=[0,0,1,top]); fig.savefig(f'{OUT}/01_single_descriptor_auroc.png'); plt.close(fig)

# ---- FIG 2 : distributions (6 panels) -----------------------------------
pick=list(U.descriptor.head(6))
fig,axes=plt.subplots(2,3,figsize=(10.4,5.6))
for ax,c in zip(axes.ravel(),pick):
    a=pd.to_numeric(R.loc[R.y==1,c],errors='coerce').dropna()
    b=pd.to_numeric(R.loc[R.y==0,c],errors='coerce').dropna()
    lo,hi=np.nanpercentile(pd.concat([a,b]),[1,99])
    if not np.isfinite(lo) or hi<=lo: continue
    bins=np.linspace(lo,hi,22)
    ax.hist(b,bins=bins,color=NO,alpha=.5,density=True)
    ax.hist(a,bins=bins,color=HIT,alpha=.75,density=True)
    ax.axvline(b.median(),color=NO,ls='--',lw=1.2); ax.axvline(a.median(),color=HIT,ls='--',lw=1.2)
    r=U[U.descriptor==c].iloc[0]
    ax.set_title(f'{L(c)}   AUROC {r.AUROC:.2f}',fontsize=9,loc='left')
    ax.set_yticks([]); ax.tick_params(labelsize=7.5)
h=[plt.Rectangle((0,0),1,1,color=HIT,alpha=.8),plt.Rectangle((0,0),1,1,color=NO,alpha=.6)]
fig.legend(h,['binder','non-binder'],frameon=False,fontsize=8.5,ncol=2,loc='upper right',bbox_to_anchor=(.99,.99))
top=head(fig,'Descriptor distributions',N)
fig.tight_layout(rect=[0,0,1,top]); fig.savefig(f'{OUT}/02_descriptor_distributions.png'); plt.close(fig)

# ---- FIG 3 : ROC + PR ----------------------------------------------------
curves=[('L1, all physics','score_l1',PH),('L1, curated physics','score_core',ACC)]
fig,(a1,a2)=plt.subplots(1,2,figsize=(9.6,4.3))
for lbl,c,col in curves:
    v=R[c].values
    fpr,tpr,_=roc_curve(y,v); pr,rc,_=precision_recall_curve(y,v)
    a1.plot(fpr,tpr,color=col,lw=2,label=f'{lbl}  {roc_auc_score(y,v):.3f}')
    a2.plot(rc,pr,color=col,lw=2,label=f'{lbl}  {average_precision_score(y,v):.3f}')
bu=U.iloc[0]; sg=1 if bu.direction.startswith('higher') else -1
v=sg*pd.to_numeric(R[bu.descriptor],errors='coerce').fillna(0).values
fpr,tpr,_=roc_curve(y,v); pr,rc,_=precision_recall_curve(y,v)
a1.plot(fpr,tpr,color=REF,lw=1.5,ls='--',label=f'best single  {bu.AUROC:.3f}')
a2.plot(rc,pr,color=REF,lw=1.5,ls='--',label=f'best single  {bu.AUPRC:.3f}')
a1.plot([0,1],[0,1],'-',color='#E5E5E5',lw=1)
a1.set_xlabel('false positive rate'); a1.set_ylabel('true positive rate'); a1.set_title('ROC',loc='left')
a2.axhline(base,ls=':',color='#999',lw=1)
a2.set_xlabel('recall'); a2.set_ylabel('precision'); a2.set_ylim(0,1); a2.set_title('Precision–recall',loc='left')
a1.legend(frameon=False,fontsize=8,loc='lower right',title='AUROC',title_fontsize=8)
a2.legend(frameon=False,fontsize=8,loc='upper right',title='AUPRC',title_fontsize=8)
top=head(fig,'Family-grouped cross-validation',N+f'   ·   random AUPRC = {base:.3f}')
fig.tight_layout(rect=[0,0,1,top]); fig.savefig(f'{OUT}/03_roc_pr.png'); plt.close(fig)

# ---- FIG 4 : leakage -----------------------------------------------------
fig,ax=plt.subplots(1,2,figsize=(8.2,4.0))
ap_lk=average_precision_score(y,R.score_leaky); ap_h=average_precision_score(y,R.score_l1)
au_lk=roc_auc_score(y,R.score_leaky); au_h=roc_auc_score(y,R.score_l1)
for a,(v1,v2),t in [(ax[0],(ap_lk,ap_h),'AUPRC'),(ax[1],(au_lk,au_h),'AUROC')]:
    a.bar([0,1],[v1,v2],.5,color=['#CFCFCF',PH])
    for i,v in enumerate([v1,v2]): a.text(i,v+.015,f'{v:.3f}',ha='center',fontsize=9.5,weight='bold')
    a.set_xticks([0,1]); a.set_xticklabels(['random CV','family-grouped'],fontsize=8.5)
    a.set_ylim(0,max(v1,v2)*1.28); a.set_title(t,loc='left')
top=head(fig,'Random CV overstates the filter','L1 logistic · the largest family holds 82 designs and 23 of 101 binders')
fig.tight_layout(rect=[0,0,1,top]); fig.savefig(f'{OUT}/04_leakage.png'); plt.close(fig)

# ---- FIG 5 : coefficients ------------------------------------------------
C=CF.sort_values('coef')
fig,ax=plt.subplots(figsize=(6.4,6.8))
ax.barh([L(c) for c in C.descriptor],C.coef,color=[HIT if v>0 else NO for v in C.coef],height=.68)
ax.axvline(0,color='#333',lw=1)
ax.set_xlabel('L1 coefficient (standardised)'); ax.tick_params(labelsize=8)
top=head(fig,'Sparse physics model',f'{len(C)} of {len(U)} descriptors survive L1   ·   green = favours binding')
fig.tight_layout(rect=[0,0,1,top]); fig.savefig(f'{OUT}/05_lasso_coefficients.png'); plt.close(fig)

# ---- FIG 6 : enrichment --------------------------------------------------
fig,(a1,a2)=plt.subplots(1,2,figsize=(9.8,4.2))
ks=np.arange(5,320,5)
for lbl,c,col in curves:
    o=np.argsort(-R[c].values); a1.plot(ks,[y[o[:k]].mean() for k in ks],color=col,lw=2,label=lbl)
a1.axhline(base,ls=':',color='#999',lw=1.2)
a1.set_xlabel('designs submitted (ranked)'); a1.set_ylabel('fraction that bind')
a1.set_title('Hit rate vs submission depth',loc='left')
a1.legend(frameon=False,fontsize=8); a1.yaxis.set_major_formatter(lambda v,p:f'{v:.0%}')
xa,ya='buns_loose_density','void_loose'
bq=pd.qcut(pd.to_numeric(R[xa],errors='coerce'),4,duplicates='drop')
vq=pd.qcut(pd.to_numeric(R[ya],errors='coerce'),4,duplicates='drop')
piv=R.groupby([bq,vq]).y.mean().unstack()
im=a2.imshow(piv.values,cmap='viridis',aspect='auto',origin='lower')
a2.set_xticks(range(piv.shape[1])); a2.set_xticklabels(['Q1','Q2','Q3','Q4'][:piv.shape[1]],fontsize=8)
a2.set_yticks(range(piv.shape[0])); a2.set_yticklabels(['Q1','Q2','Q3','Q4'][:piv.shape[0]],fontsize=8)
a2.set_xlabel(f'{L(ya)}  (quartile)'); a2.set_ylabel(f'{L(xa)}  (quartile)')
a2.set_title('Hit rate by packing quality',loc='left'); a2.grid(False)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v=piv.values[i,j]
        if v==v: a2.text(j,i,f'{v:.0%}',ha='center',va='center',color='w',fontsize=8.5,weight='bold')
fig.colorbar(im,ax=a2,label='hit rate')
top=head(fig,'What the score buys you',N)
fig.tight_layout(rect=[0,0,1,top]); fig.savefig(f'{OUT}/06_enrichment.png'); plt.close(fig)

# ---- FIG 7 : affinity ----------------------------------------------------
B=pd.read_csv('affinity_scored.csv'); UD=pd.read_csv('affinity_univariate.csv')
AS=json.load(open('affinity_summary.json'))
fig,axes=plt.subplots(1,3,figsize=(11.4,3.8))
a=axes[0]
a.scatter(B.kd_pred,B.logkd,s=18,color=ACC,alpha=.7,edgecolor='none')
z=np.polyfit(B.kd_pred,B.logkd,1); xs=np.linspace(B.kd_pred.min(),B.kd_pred.max(),20)
a.plot(xs,np.polyval(z,xs),color=NO,lw=1.6)
a.set_xlabel('predicted log$_{10}$K$_D$'); a.set_ylabel('measured log$_{10}$K$_D$')
a.set_title(f"all binders   r = {AS['pearson']:.2f}",loc='left')
a=axes[1]; T2=UD.head(8).iloc[::-1]
a.barh([L(c) for c in T2.descriptor],T2.r,color=[HIT if v<0 else NO for v in T2.r],height=.68)
a.axvline(0,color='#333',lw=1); a.set_xlabel('Pearson r vs log$_{10}$K$_D$')
a.set_title('single descriptors',loc='left'); a.tick_params(labelsize=8)
a=axes[2]
st=[('all\nbinders',0.503),('de novo\nonly',0.186),('one format\nonly',-0.302)]
a.bar(range(3),[v for _,v in st],.55,color=[ACC,'#9A8FBF',NO])
for i,(_,v) in enumerate(st): a.text(i,v+(.03 if v>0 else -.07),f'{v:.2f}',ha='center',fontsize=9.5,weight='bold')
a.axhline(0,color='#333',lw=1); a.set_xticks(range(3)); a.set_xticklabels([k for k,_ in st],fontsize=8)
a.set_ylabel('Pearson r'); a.set_ylim(-.45,.62); a.set_title('stratified by design format',loc='left')
top=head(fig,'Affinity ranking — a negative result',f"n = {AS['n']} binders with measured K$_D$")
fig.tight_layout(rect=[0,0,1,top]); fig.savefig(f'{OUT}/07_affinity_ranking.png'); plt.close(fig)
print('figures:',sorted(os.listdir(OUT)))
