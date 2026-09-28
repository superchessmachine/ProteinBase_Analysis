"""Unified single-metric ranking across all three sources on identical rows."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, average_precision_score

D = pd.read_csv('compare_scored.csv')          # mine + rosetta + labels
F = pd.read_pickle('flat.pkl')                  # adaptyv AI metrics
F = F[F.target == 'nipah-glycoprotein-g'][[
    'id', 'boltz2_ipsae', 'boltz2_min_ipsae', 'boltz2_iptm', 'boltz2_complex_iplddt',
    'boltz2_plddt', 'boltz2_complex_plddt', 'boltz2_lis', 'boltz2_ptm',
    'shape_complimentarity_boltz2_binder_ss', 'esmfold_plddt', 'proteinmpnn_score',
    'redesigned_proteinmpnn_score', 'boltz2_complex_pde']].drop_duplicates('id')
D = D.merge(F, on='id', how='left')
y = D.y.values; base = y.mean()

SKIP = {'y', 'family', 'idx', 'logkd', 'kd', 'binding', 'binding_strength', 'expressed',
        'expression-yield', 'neutralization', 'score_l1', 'score_core', 'score_leaky',
        'score_final', 'score_ros', 'score_mine', 'score_both', 'oof_mech', 'oof_ai',
        'oof_both', 'oof_mech_leaky', 'oof_ai_leaky', 'oof_both_leaky'}
AI = {'boltz2_ipsae','boltz2_min_ipsae','boltz2_iptm','boltz2_complex_iplddt','boltz2_plddt',
      'boltz2_complex_plddt','boltz2_lis','boltz2_ptm','shape_complimentarity_boltz2_binder_ss',
      'esmfold_plddt','proteinmpnn_score','redesigned_proteinmpnn_score','boltz2_complex_pde',
      'iplddt'}

def src(c):
    if c.startswith('ros_'): return 'Rosetta'
    if c in AI: return 'Adaptyv AI'
    return 'Mine (physics)'

def boot(yy, ss, fn, n=600, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(yy), len(yy))
        if len(np.unique(yy[i])) < 2: continue
        v.append(fn(yy[i], ss[i]))
    return np.percentile(v, [2.5, 97.5])

rows = []
for c in D.columns:
    if c in SKIP or not pd.api.types.is_numeric_dtype(D[c]): continue
    s = pd.to_numeric(D[c], errors='coerce'); m = s.notna()
    if m.sum() < 300 or s[m].std() < 1e-9: continue
    yy = y[m.values]; v = s[m].values
    if len(np.unique(yy)) < 2: continue
    a = roc_auc_score(yy, v); sg = 1 if a >= .5 else -1
    a = roc_auc_score(yy, sg * v); ap = average_precision_score(yy, sg * v)
    lo, hi = boot(yy, sg * v, roc_auc_score)
    alo, ahi = boot(yy, sg * v, average_precision_score)
    rows.append(dict(metric=c, source=src(c), direction='higher' if sg > 0 else 'lower',
                     AUROC=a, auroc_lo=lo, auroc_hi=hi, AUPRC=ap, auprc_lo=alo, auprc_hi=ahi,
                     lift=ap / base, n=int(m.sum())))
U = pd.DataFrame(rows)
U.to_csv('unified_ranking.csv', index=False)
print(f"n={len(D)}  binders={y.sum()}  base={base:.4f}  metrics ranked={len(U)}")

for key in ('AUROC', 'AUPRC'):
    T = U.sort_values(key, ascending=False).head(12)
    print(f"\n{'='*82}\nTOP 12 SINGLE METRICS BY {key}   (random: AUROC 0.500, AUPRC {base:.3f})\n{'='*82}")
    for i, (_, r) in enumerate(T.iterrows(), 1):
        ci = f"[{r.auroc_lo:.3f}-{r.auroc_hi:.3f}]" if key == 'AUROC' else f"[{r.auprc_lo:.3f}-{r.auprc_hi:.3f}]"
        print(f"{i:2d}. {r.metric:42s} {r.source:15s} AUROC={r.AUROC:.3f} AUPRC={r.AUPRC:.3f} "
              f"{ci} {r.direction} is better")

print(f"\n{'='*82}\nBEST PER SOURCE\n{'='*82}")
for s in ['Mine (physics)', 'Rosetta', 'Adaptyv AI']:
    sub = U[U.source == s]
    if not len(sub): continue
    b1 = sub.loc[sub.AUROC.idxmax()]; b2 = sub.loc[sub.AUPRC.idxmax()]
    print(f"  {s:16s} best AUROC: {b1.metric:40s} {b1.AUROC:.3f}")
    print(f"  {'':16s} best AUPRC: {b2.metric:40s} {b2.AUPRC:.3f}")

# how many metrics are individually significant (CI excludes 0.5)
sig = U[U.auroc_lo > 0.5]
print(f"\n{len(sig)} of {len(U)} metrics have a 95% CI excluding AUROC 0.5")
print("source breakdown of those:", sig.source.value_counts().to_dict())
