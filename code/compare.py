"""Head-to-head: Rosetta InterfaceAnalyzer vs the hand-built mechanistic descriptors."""
import pandas as pd, numpy as np, warnings, json
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr, pearsonr

M = pd.read_csv('mech2_scored.csv')
Rt = pd.read_csv('rosetta_results.csv')
ros_cols = [c for c in Rt.columns if c.startswith('ros_')]
Rt = Rt[['id'] + ros_cols].drop_duplicates('id')
D = M.merge(Rt, on='id', how='left')
# drop all-NaN rosetta columns
ros_cols = [c for c in ros_cols if D[c].notna().sum() > 100 and D[c].std() > 1e-9]
D['y'] = D.binding.astype(str).str.lower().eq('true').astype(int)
y = D.y.values; base = y.mean(); fam = D.family.values
print(f"n={len(D)}  binders={y.sum()}  base={base:.4f}  rosetta metrics usable={len(ros_cols)}")
print(f"rosetta rows with data: {D[ros_cols[0]].notna().sum()}")

def boot(yy, ss, fn, n=1200, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(yy), len(yy))
        if len(np.unique(yy[i])) < 2: continue
        v.append(fn(yy[i], ss[i]))
    return np.percentile(v, [2.5, 97.5])

# ---------- 1. univariate Rosetta ----------
rows = []
for c in ros_cols:
    s = pd.to_numeric(D[c], errors='coerce'); m = s.notna()
    if m.sum() < 100: continue
    yy = y[m.values]; v = s[m].values
    if len(np.unique(yy)) < 2: continue
    a = roc_auc_score(yy, v); sg = 1 if a >= .5 else -1
    a = roc_auc_score(yy, sg*v); ap = average_precision_score(yy, sg*v)
    lo, hi = boot(yy, sg*v, roc_auc_score)
    rows.append(dict(metric=c, direction='higher=better' if sg > 0 else 'lower=better',
                     AUROC=a, ci_lo=lo, ci_hi=hi, AUPRC=ap, lift=ap/base, n=int(m.sum())))
RU = pd.DataFrame(rows).sort_values('AUROC', ascending=False)
RU.to_csv('rosetta_univariate.csv', index=False)
print("\n===== ROSETTA INTERFACEANALYZER, single metrics =====")
print(RU.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ---------- 2. head-to-head on matched concepts ----------
MU = pd.read_csv('univariate_benchmark.csv').set_index('descriptor')
PAIRS = [('buried unsatisfied polars', 'buns_loose_density', 'ros_unsat_per_dSASA'),
         ('buried unsatisfied polars (raw)', 'buns_loose', 'ros_unsat'),
         ('shape complementarity', 'sc', 'ros_sc'),
         ('buried surface area', 'bsa', 'ros_dSASA'),
         ('apolar burial', 'bsa_apolar', 'ros_dhSASA'),
         ('packing / voids', 'void_loose', 'ros_packstat'),
         ('binding energy', 'ddg_est', 'ros_dG'),
         ('binding energy / area', 'ddg_per_bsa', 'ros_dG_dSASA'),
         ('H-bond energy', 'hb_energy', 'ros_hb_E'),
         ('interface residues', 'ifc_res', 'ros_nres')]
print("\n===== HEAD TO HEAD: mine vs Rosetta =====")
hh = []
for label, mine, ros in PAIRS:
    if mine not in MU.index or ros not in D.columns: continue
    a_m = MU.loc[mine, 'AUROC']
    s = pd.to_numeric(D[ros], errors='coerce'); m = s.notna()
    if m.sum() < 100: continue
    yy = y[m.values]; v = s[m].values
    a_r = max(roc_auc_score(yy, v), roc_auc_score(yy, -v))
    mv = pd.to_numeric(D[mine], errors='coerce')
    ok = mv.notna() & s.notna()
    rho = spearmanr(mv[ok], s[ok]).correlation
    hh.append(dict(concept=label, mine=mine, rosetta=ros, AUROC_mine=a_m,
                   AUROC_rosetta=a_r, agreement_rho=rho))
HH = pd.DataFrame(hh)
HH.to_csv('head_to_head.csv', index=False)
print(HH.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ---------- 3. models ----------
MECHF = [c for c in M.columns if c not in {
 'id','name','author','designMethod','seq','url','target','binding','binding_strength','expressed',
 'kd','expression-yield','design_class','esmfold_plddt','proteinmpnn_score','family','y','logkd',
 'binder_chain','idx','score_l1','score_core','score_leaky','score_final'} and pd.api.types.is_numeric_dtype(M[c])]

def nested(cols, tag, Cs=(0.01, 0.03, 0.1, 0.3, 1.0)):
    X = D[cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = X[[c for c in cols if X[c].std() > 1e-9]]
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, fam):
        best, bC = -1, Cs[0]
        for C in Cs:
            sc = []
            for it, ie in StratifiedGroupKFold(4, shuffle=True, random_state=1).split(
                    X.iloc[tr], y[tr], fam[tr]):
                mdl = make_pipeline(StandardScaler(), LogisticRegression(
                    penalty='l1', C=C, solver='liblinear', max_iter=5000, class_weight='balanced'))
                mdl.fit(X.iloc[tr].iloc[it], y[tr][it])
                sc.append(average_precision_score(y[tr][ie], mdl.predict_proba(X.iloc[tr].iloc[ie])[:, 1]))
            if np.mean(sc) > best: best, bC = np.mean(sc), C
        mdl = make_pipeline(StandardScaler(), LogisticRegression(
            penalty='l1', C=bC, solver='liblinear', max_iter=5000, class_weight='balanced'))
        mdl.fit(X.iloc[tr], y[tr]); oof[te] = mdl.predict_proba(X.iloc[te])[:, 1]
    a = roc_auc_score(y, oof); ap = average_precision_score(y, oof)
    alo, ahi = boot(y, oof, roc_auc_score); plo, phi = boot(y, oof, average_precision_score)
    o = np.argsort(-oof)
    pk = '  '.join(f'P@{k}={y[o[:k]].mean():.2f}' for k in (10, 20, 50))
    print(f"  {tag:34s} AUROC={a:.3f} [{alo:.3f},{ahi:.3f}]  AUPRC={ap:.3f} [{plo:.3f},{phi:.3f}]  {pk}")
    return oof, a, ap

print("\n===== NESTED FAMILY-GROUPED CV (L1 logistic) =====")
o_r, a_r, ap_r = nested(ros_cols, f'Rosetta only ({len(ros_cols)} metrics)')
o_m, a_m, ap_m = nested(MECHF, f'Mine only ({len(MECHF)} descriptors)')
o_b, a_b, ap_b = nested(MECHF + ros_cols, 'Both combined')
D['score_ros'] = o_r; D['score_mine'] = o_m; D['score_both'] = o_b
D.to_csv('compare_scored.csv', index=False)
json.dump({'auroc_ros': a_r, 'auprc_ros': ap_r, 'auroc_mine': a_m, 'auprc_mine': ap_m,
           'auroc_both': a_b, 'auprc_both': ap_b, 'base': float(base),
           'n_ros': len(ros_cols), 'n_mine': len(MECHF)}, open('compare_summary.json', 'w'), indent=1)
print("\nwrote compare_scored.csv, rosetta_univariate.csv, head_to_head.csv, compare_summary.json")
