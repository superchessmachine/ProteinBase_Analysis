"""Fit the selected 26-term composite on all data and expose exactly what it does."""
import pandas as pd, numpy as np, warnings, json
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score

D = pd.read_csv('mech2_scored.csv')
y = D.y.values; fam = D.family.values
CORE = ['bsa','f_apolar','buns_density','buns_loose_density','void_loose','gap_index','sc',
 'n_salt','hb_energy','lj_atr','lj_rep','dG_solv','elec_att','anchor_bsa','n_anchor50','planarity',
 'eccentricity','n_segments','ddg_per_bsa','ct_density','rg_norm','frac_helix',
 'surf_hydrophobic_frac','agg_max','n_cation_pi','rel_contact_order']
CORE = [c for c in CORE if c in D.columns]
X = D[CORE].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
X = X.fillna(X.median())

# pick C the same way the nested CV did, on grouped folds
best, bC = -1, 0.01
for C in (0.01, 0.03, 0.1, 0.3, 1.0):
    sc = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, fam):
        m = make_pipeline(StandardScaler(), LogisticRegression(
            penalty='l1', C=C, solver='liblinear', max_iter=5000, class_weight='balanced'))
        m.fit(X.iloc[tr], y[tr])
        sc.append(average_precision_score(y[te], m.predict_proba(X.iloc[te])[:,1]))
    if np.mean(sc) > best: best, bC = np.mean(sc), C
print(f"selected C = {bC}  (grouped-CV AUPRC {best:.3f})")

mdl = make_pipeline(StandardScaler(), LogisticRegression(
    penalty='l1', C=bC, solver='liblinear', max_iter=5000, class_weight='balanced'))
mdl.fit(X, y)
sc_, lr = mdl[0], mdl[-1]
coef = pd.DataFrame({'term': CORE, 'coef': lr.coef_[0],
                     'mean': sc_.mean_, 'sd': np.sqrt(sc_.var_)})
coef['kept'] = coef.coef.abs() > 1e-8
K = coef[coef.kept].sort_values('coef', key=lambda v: v.abs(), ascending=False)
print(f"\nL1 keeps {len(K)} of {len(CORE)} terms;  intercept = {lr.intercept_[0]:.4f}")
print("\n(coef is in standardised units: +ve raises P(bind), -ve lowers it)")
print(K[['term','coef','mean','sd']].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\ndropped by L1:", ', '.join(coef[~coef.kept].term) or '(none)')
coef.to_csv('composite_coefficients.csv', index=False)

# what the score does to the odds: effect of a 1-SD move in each kept term
K2 = K.copy(); K2['odds_per_SD'] = np.exp(K2.coef)
print("\nodds multiplier per +1 SD of each term:")
for _, r in K2.iterrows():
    arrow = 'up  ' if r.coef > 0 else 'down'
    print(f"  {r.term:24s} {arrow} x{r.odds_per_SD:.2f}")

# decile calibration of the out-of-fold score
B = pd.read_csv('best_composite_scores.csv')
B['y'] = B.binding.astype(str).str.lower().eq('true').astype(int)
B['dec'] = pd.qcut(B.best_score, 10, labels=False, duplicates='drop')
cal = B.groupby('dec').agg(n=('y','size'), hits=('y','sum'), rate=('y','mean'),
                           lo=('best_score','min'), hi=('best_score','max'))
print("\nout-of-fold decile calibration (base rate 0.098):")
print(cal.to_string(float_format=lambda v: f"{v:.3f}"))
json.dump({'C': bC, 'intercept': float(lr.intercept_[0]), 'n_kept': int(len(K)),
           'n_terms': len(CORE)}, open('composite_model.json','w'), indent=1)
