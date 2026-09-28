import pandas as pd, numpy as np, warnings, json
warnings.filterwarnings('ignore')
from sklearn.linear_model import Lasso, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr, pearsonr

R=pd.read_csv('mech2_scored.csv')
B=R[(R.y==1)&R.logkd.notna()].copy()
DROP={'id','name','author','designMethod','seq','url','target','binding','binding_strength',
 'expressed','kd','expression-yield','design_class','esmfold_plddt','proteinmpnn_score',
 'family','y','logkd','binder_chain','idx','score_l1','score_core','score_leaky','score_final'}
FEAT=[c for c in B.columns if c not in DROP and pd.api.types.is_numeric_dtype(B[c])]
X=B[FEAT].replace([np.inf,-np.inf],np.nan); X=X.fillna(X.median())
X=X[[c for c in FEAT if X[c].std()>1e-9]]; FEAT=list(X.columns)
t=B.logkd.values; grp=B.family.values
print(f"binders with Kd: n={len(B)}  families={B.family.nunique()}  features={len(FEAT)}")
print(f"log10 Kd range {t.min():.2f} to {t.max():.2f} (median {np.median(t):.2f})")

def cv_pred(model_fn, ngroups=5):
    oof=np.full(len(t),np.nan)
    gkf=GroupKFold(n_splits=min(ngroups,len(np.unique(grp))))
    for tr,te in gkf.split(X,t,grp):
        m=model_fn(); m.fit(X.iloc[tr],t[tr]); oof[te]=m.predict(X.iloc[te])
    return oof

print("\n===== GROUPED CV: predicting log10 Kd from physics =====")
res={}
for name,fn in [
  ('Lasso alpha=0.30', lambda: make_pipeline(StandardScaler(),Lasso(alpha=0.30,max_iter=50000))),
  ('Lasso alpha=0.15', lambda: make_pipeline(StandardScaler(),Lasso(alpha=0.15,max_iter=50000))),
  ('Lasso alpha=0.08', lambda: make_pipeline(StandardScaler(),Lasso(alpha=0.08,max_iter=50000))),
  ('Ridge (CV alpha)', lambda: make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-1,3,30)))),
]:
    o=cv_pred(fn); m=~np.isnan(o)
    r=pearsonr(o[m],t[m])[0]; rho=spearmanr(o[m],t[m]).correlation
    rmse=np.sqrt(np.mean((o[m]-t[m])**2))
    print(f"  {name:20s} Pearson r={r:.3f}  Spearman rho={rho:.3f}  RMSE={rmse:.2f} log units")
    res[name]=(o,r,rho)

best=max(res.items(), key=lambda kv: kv[1][1])
print(f"\n  best: {best[0]}  r={best[1][1]:.3f}")
B['kd_pred']=best[1][0]
B.to_csv('affinity_scored.csv',index=False)

# final interpretable model on all binders
fm=make_pipeline(StandardScaler(),Lasso(alpha=0.15,max_iter=50000)); fm.fit(X,t)
cf=pd.DataFrame({'descriptor':FEAT,'coef':fm[-1].coef_})
cf=cf[cf.coef.abs()>1e-8].sort_values('coef',key=lambda v:v.abs(),ascending=False)
cf.to_csv('affinity_coefficients.csv',index=False)
print(f"\n===== AFFINITY LASSO: {len(cf)}/{len(FEAT)} descriptors retained =====")
print("(negative coef = lowers log10 Kd = TIGHTER binding)")
print(cf.head(18).to_string(index=False,float_format=lambda v:f"{v:.4f}"))

# per-descriptor grouped-CV-free correlation for reference
rows=[]
for c in FEAT:
    rows.append(dict(descriptor=c, r=pearsonr(X[c],t)[0], rho=spearmanr(X[c],t).correlation))
UD=pd.DataFrame(rows).sort_values('r',key=lambda v:v.abs(),ascending=False)
UD.to_csv('affinity_univariate.csv',index=False)
print("\n===== single descriptors vs log10 Kd (in-sample) =====")
print(UD.head(12).to_string(index=False,float_format=lambda v:f"{v:.3f}"))

# --- confound check: is affinity signal just design format? ---
print("\n===== CONFOUND CHECK: log10 Kd by design class =====")
print(B.groupby('design_class').logkd.agg(['count','median']).to_string(float_format=lambda v:f"{v:.2f}"))
from sklearn.linear_model import RidgeCV as _R
rr=make_pipeline(StandardScaler(),_R(alphas=np.logspace(-1,3,30))); rr.fit(X,t)
rc=pd.DataFrame({'descriptor':FEAT,'coef':rr[-1].coef_}).sort_values('coef',key=lambda v:v.abs(),ascending=False)
rc.to_csv('affinity_ridge_coefficients.csv',index=False)
print("\n===== RIDGE (deliverable affinity model) top terms =====")
print(rc.head(15).to_string(index=False,float_format=lambda v:f"{v:.4f}"))
json.dump({'n':int(len(B)),'fam':int(B.family.nunique()),'best_model':best[0],
 'pearson':float(best[1][1]),'spearman':float(best[1][2]),'n_kept':int(len(cf))},
 open('affinity_summary.json','w'),indent=1)
