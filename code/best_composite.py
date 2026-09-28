"""Re-derive the best composite score and emit its standalone panel."""
import pandas as pd, numpy as np, warnings, json
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr

M = pd.read_csv('mech2_scored.csv')
AIM = ['boltz2_min_ipsae', 'boltz2_ipsae', 'boltz2_iptm']
F = pd.read_pickle('flat.pkl')
F = F[F.target == 'nipah-glycoprotein-g'][['id'] + AIM].drop_duplicates('id')
D = M.merge(F, on='id', how='left')
R = pd.read_csv('rosetta_results.csv')
ros = [c for c in R.columns if c.startswith('ros_')]
D = D.merge(R[['id'] + ros].drop_duplicates('id'), on='id', how='left')
y = D.y.values; base = y.mean(); fam = D.family.values
print(f"n={len(D)} binders={y.sum()} base={base:.4f} families={D.family.nunique()}")

CORE = ['bsa','f_apolar','buns_density','buns_loose_density','void_loose','gap_index','sc',
 'n_salt','hb_energy','lj_atr','lj_rep','dG_solv','elec_att','anchor_bsa','n_anchor50','planarity',
 'eccentricity','n_segments','ddg_per_bsa','ct_density','rg_norm','frac_helix',
 'surf_hydrophobic_frac','agg_max','n_cation_pi','rel_contact_order']
CORE = [c for c in CORE if c in D.columns]
ALL_PHYS = [c for c in M.columns if c not in {
 'id','name','author','designMethod','seq','url','target','binding','binding_strength','expressed',
 'kd','expression-yield','design_class','esmfold_plddt','proteinmpnn_score','family','y','logkd',
 'binder_chain','idx','score_l1','score_core','score_leaky','score_final'
} and pd.api.types.is_numeric_dtype(M[c])]

def boot(yy, ss, fn, n=1500, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(yy), len(yy))
        if len(np.unique(yy[i])) < 2: continue
        v.append(fn(yy[i], ss[i]))
    return np.percentile(v, [2.5, 97.5])

def nested(cols, tag, Cs=(0.01, 0.03, 0.1, 0.3, 1.0)):
    X = D[cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()); X = X[[c for c in cols if X[c].std() > 1e-9]]
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, fam):
        best, bC = -1, Cs[0]
        for C in Cs:
            sc = []
            for it, ie in StratifiedGroupKFold(4, shuffle=True, random_state=1).split(
                    X.iloc[tr], y[tr], fam[tr]):
                m = make_pipeline(StandardScaler(), LogisticRegression(
                    penalty='l1', C=C, solver='liblinear', max_iter=5000, class_weight='balanced'))
                m.fit(X.iloc[tr].iloc[it], y[tr][it])
                sc.append(average_precision_score(y[tr][ie], m.predict_proba(X.iloc[tr].iloc[ie])[:, 1]))
            if np.mean(sc) > best: best, bC = np.mean(sc), C
        m = make_pipeline(StandardScaler(), LogisticRegression(
            penalty='l1', C=bC, solver='liblinear', max_iter=5000, class_weight='balanced'))
        m.fit(X.iloc[tr], y[tr]); oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    a = roc_auc_score(y, oof); ap = average_precision_score(y, oof)
    alo, ahi = boot(y, oof, roc_auc_score); plo, phi = boot(y, oof, average_precision_score)
    o = np.argsort(-oof)
    pk = {k: float(y[o[:k]].mean()) for k in (10, 20, 50, 100)}
    print(f"  {tag:46s} AUROC={a:.3f} [{alo:.3f},{ahi:.3f}]  AUPRC={ap:.3f} [{plo:.3f},{phi:.3f}]"
          f"  P@10={pk[10]:.2f} P@20={pk[20]:.2f} P@50={pk[50]:.2f}")
    return dict(tag=tag, oof=oof, auroc=a, auprc=ap, auroc_ci=(alo, ahi),
                auprc_ci=(plo, phi), pk=pk, n_feat=X.shape[1])

print("\n=== nested family-grouped CV, L1 logistic ===")
cands = [
    nested(CORE, 'Physics, curated (26 terms)'),
    nested(ALL_PHYS, f'Physics, all ({len(ALL_PHYS)} terms)'),
    nested(CORE + ['boltz2_min_ipsae'], 'Physics curated + min ipSAE'),
    nested(CORE + AIM, 'Physics curated + Boltz-2 confidence'),
    nested(CORE + ['ros_sc'], 'Physics curated + Rosetta sc'),
    nested(CORE + ['ros_sc', 'boltz2_min_ipsae'], 'Physics curated + ros_sc + min ipSAE'),
]
# Parsimony rule: the AUPRC spread across these variants (0.310-0.329) is far
# inside the bootstrap CIs, so they are statistically indistinguishable. Among
# models within 0.01 AUPRC of the best, take the one with fewest features.
top = max(d['auprc'] for d in cands)
tied = [d for d in cands if d['auprc'] >= top - 0.01]
best = min(tied, key=lambda d: d['n_feat'])
print(f"\n  {len(tied)} of {len(cands)} variants within 0.01 AUPRC of the best "
      f"({top:.3f}) - statistically indistinguishable.")
print(f">>> SELECTED (fewest terms among ties): {best['tag']}")
print(f"    AUPRC={best['auprc']:.3f} {list(np.round(best['auprc_ci'],3))}  "
      f"AUROC={best['auroc']:.3f} {list(np.round(best['auroc_ci'],3))}  n_feat={best['n_feat']}")
pd.DataFrame([{k: v for k, v in d.items() if k not in ('oof','pk')} | d['pk']
              for d in cands]).to_csv('composite_variants.csv', index=False)

D['best_score'] = best['oof']
D[['id', 'binding', 'kd', 'family', 'best_score']].to_csv('best_composite_scores.csv', index=False)
json.dump({k: (v if not isinstance(v, tuple) else list(v))
           for k, v in best.items() if k != 'oof'}, open('best_composite.json', 'w'), indent=1)
print("wrote best_composite_scores.csv")
