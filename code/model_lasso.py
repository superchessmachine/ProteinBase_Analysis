import pandas as pd, numpy as np, warnings, json
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr, pearsonr

R = pd.read_csv('mech2_results.csv')
R = R[R.bsa.notna()].copy()
R['y'] = R.binding.astype(str).str.lower().eq('true').astype(int)
R['logkd'] = np.log10(pd.to_numeric(R.kd, errors='coerce'))
y = R.y.values; base = y.mean(); fam = R.family.values
DROP = {'id','name','author','designMethod','seq','url','target','binding','binding_strength',
        'expressed','kd','expression-yield','design_class','esmfold_plddt','proteinmpnn_score',
        'family','y','logkd','binder_chain','idx'}
FEAT = [c for c in R.columns if c not in DROP and pd.api.types.is_numeric_dtype(R[c])]
print(f"n={len(R)} binders={y.sum()} base={base:.4f} families={R.family.nunique()} features={len(FEAT)}")

X = R[FEAT].apply(pd.to_numeric, errors='coerce')
X = X.replace([np.inf,-np.inf], np.nan)
X = X.fillna(X.median())
keep = [c for c in FEAT if X[c].std() > 1e-9]
X = X[keep]; FEAT = keep
print("usable features:", len(FEAT))

def boot(yy, ss, fn, n=1500, seed=0):
    rng = np.random.default_rng(seed); v=[]
    for _ in range(n):
        i = rng.integers(0,len(yy),len(yy))
        if len(np.unique(yy[i]))<2: continue
        v.append(fn(yy[i], ss[i]))
    return np.percentile(v,[2.5,97.5])

# ---- 1. univariate table --------------------------------------------------
rows=[]
for c in FEAT:
    v = X[c].values
    a = roc_auc_score(y, v); sg = 1 if a>=.5 else -1
    a = roc_auc_score(y, sg*v); ap = average_precision_score(y, sg*v)
    lo,hi = boot(y, sg*v, roc_auc_score)
    m = R.logkd.notna().values & (y==1)
    rho = spearmanr(v[m], R.logkd.values[m]).correlation if m.sum()>20 else np.nan
    rows.append(dict(descriptor=c, direction='higher=better' if sg>0 else 'lower=better',
                     AUROC=a, ci_lo=lo, ci_hi=hi, AUPRC=ap, lift=ap/base, rho_logKd=rho))
U = pd.DataFrame(rows).sort_values('AUROC', ascending=False)
U.to_csv('univariate_benchmark.csv', index=False)
print("\n===== TOP 20 SINGLE MECHANISTIC DESCRIPTORS =====")
print(U.head(20).to_string(index=False, float_format=lambda v:f"{v:.3f}"))

# ---- 2. nested grouped CV, L1 logistic ------------------------------------
def nested(Xd, tag, penalty='l1', Cs=(0.003,0.01,0.03,0.1,0.3,1.0)):
    oof = np.zeros(len(y)); chosen=[]
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    for tr, te in outer.split(Xd, y, fam):
        best, bestC = -1, Cs[0]
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=1)
        for C in Cs:
            sc=[]
            for itr, ite in inner.split(Xd.iloc[tr], y[tr], fam[tr]):
                m = make_pipeline(StandardScaler(), LogisticRegression(
                    penalty=penalty, C=C, solver='liblinear', max_iter=5000, class_weight='balanced'))
                m.fit(Xd.iloc[tr].iloc[itr], y[tr][itr])
                sc.append(average_precision_score(y[tr][ite], m.predict_proba(Xd.iloc[tr].iloc[ite])[:,1]))
            if np.mean(sc) > best: best, bestC = np.mean(sc), C
        chosen.append(bestC)
        m = make_pipeline(StandardScaler(), LogisticRegression(
            penalty=penalty, C=bestC, solver='liblinear', max_iter=5000, class_weight='balanced'))
        m.fit(Xd.iloc[tr], y[tr]); oof[te] = m.predict_proba(Xd.iloc[te])[:,1]
    a=roc_auc_score(y,oof); ap=average_precision_score(y,oof)
    alo,ahi=boot(y,oof,roc_auc_score); plo,phi=boot(y,oof,average_precision_score)
    print(f"  {tag:44s} AUROC={a:.3f} [{alo:.3f},{ahi:.3f}]  AUPRC={ap:.3f} [{plo:.3f},{phi:.3f}]  lift={ap/base:.1f}x  C={chosen}")
    return oof, a, ap

print("\n===== NESTED FAMILY-GROUPED CV (honest) =====")
oof_l1,_,ap_l1 = nested(X, 'L1 logistic, all physics features')
CORE=[c for c in ['bsa','f_apolar','buns_density','buns_loose_density','void_loose','gap_index','sc',
 'n_salt','hb_energy','lj_atr','lj_rep','dG_solv','elec_att','anchor_bsa','n_anchor50','planarity',
 'eccentricity','n_segments','ddg_per_bsa','ct_density','rg_norm','frac_helix','surf_hydrophobic_frac',
 'agg_max','n_cation_pi','rel_contact_order'] if c in X.columns]
oof_core,_,ap_core = nested(X[CORE], f'L1 logistic, {len(CORE)} curated terms')

# leaky comparison
def leaky(Xd, tag):
    oof=np.zeros(len(y))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Xd,y):
        m=make_pipeline(StandardScaler(),LogisticRegression(penalty='l1',C=0.1,solver='liblinear',max_iter=5000,class_weight='balanced'))
        m.fit(Xd.iloc[tr],y[tr]); oof[te]=m.predict_proba(Xd.iloc[te])[:,1]
    print(f"  {tag:44s} AUROC={roc_auc_score(y,oof):.3f}  AUPRC={average_precision_score(y,oof):.3f}")
    return oof
print("\n===== RANDOM CV (leaky, for contrast) =====")
oof_leaky = leaky(X, 'L1 logistic, all physics features')

# ---- 3. final interpretable model on all data ------------------------------
final = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', C=0.1, solver='liblinear',
                                                          max_iter=5000, class_weight='balanced'))
final.fit(X, y)
coef = pd.DataFrame({'descriptor':FEAT,'coef':final[-1].coef_[0]})
coef['abs']=coef.coef.abs(); coef=coef[coef['abs']>1e-8].sort_values('abs',ascending=False)
mu=X.mean(); sd=X.std()
coef['mean']=coef.descriptor.map(mu); coef['sd']=coef.descriptor.map(sd)
coef.to_csv('lasso_coefficients.csv',index=False)
print(f"\n===== FINAL L1 MODEL: {len(coef)}/{len(FEAT)} descriptors retained =====")
print(coef[['descriptor','coef','mean','sd']].to_string(index=False,float_format=lambda v:f"{v:.4f}"))

# ---- 4. Kd ranking --------------------------------------------------------
m = R.logkd.notna().values
if m.sum()>20:
    s=final.predict_proba(X)[:,1][m]; lk=R.logkd.values[m]
    print(f"\n===== AFFINITY RANKING among {m.sum()} measured binders =====")
    print(f"  score vs log10(Kd): Pearson r={pearsonr(s,lk)[0]:.3f}  Spearman rho={spearmanr(s,lk).correlation:.3f}")
    print("  best single descriptors vs log10(Kd):")
    print(U.reindex(U.rho_logKd.abs().sort_values(ascending=False).index)[['descriptor','rho_logKd']].head(8).to_string(index=False,float_format=lambda v:f"{v:.3f}"))

R['score_l1']=oof_l1; R['score_core']=oof_core; R['score_leaky']=oof_leaky
R['score_final']=final.predict_proba(X)[:,1]
R.to_csv('mech2_scored.csv',index=False)
print("\n===== top-k precision (nested grouped CV) =====")
for nm,s in [('L1 all physics',oof_l1),('L1 curated',oof_core)]:
    o=np.argsort(-s); print(f"  {nm:18s} "+"  ".join(f"P@{k}={y[o[:k]].mean():.2f}" for k in (10,20,50,100,200)))
json.dump({'base':float(base),'auprc_l1':float(ap_l1),'auprc_core':float(ap_core),
           'n':int(len(R)),'binders':int(y.sum()),'families':int(R.family.nunique()),
           'n_feat':len(FEAT),'n_kept':int(len(coef))}, open('summary.json','w'), indent=1)
