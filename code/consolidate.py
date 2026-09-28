"""Consolidate the pipeline's intermediates into the published tables.

Produces four tables plus a data dictionary:
  designs.csv             one row per design: identity, labels, all descriptors, scores
  metrics_benchmark.csv   one row per single metric: AUROC/AUPRC with bootstrap CIs
  model_comparison.csv    one row per fitted model variant: nested grouped-CV results
  composite_coefficients.csv  the deliverable model's terms, weights and scaling
"""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')

SCR = '.'
OUT = 'proteinbase_data/data'

# ---------------------------------------------------------------- designs ---
lab = pd.read_csv('labels_fam.csv')
mech = pd.read_csv('mech2_results.csv')
ros = pd.read_csv('rosetta_results.csv')
best = pd.read_csv('best_composite_scores.csv')
flat = pd.read_pickle('flat.pkl')
flat = flat[flat.target == 'nipah-glycoprotein-g']

ID_COLS = ['id', 'name', 'author', 'designMethod', 'seq', 'url', 'target',
           'design_class', 'family']
EXP_COLS = ['binding', 'binding_strength', 'kd']   # 'expressed' comes from flat.pkl
# physics descriptors = everything mech2 added that is not identity/label
PHYS = [c for c in mech.columns
        if c not in set(ID_COLS + EXP_COLS + ['expression-yield', 'esmfold_plddt',
                                              'proteinmpnn_score', 'binder_chain'])
        and pd.api.types.is_numeric_dtype(mech[c])]
ROS = [c for c in ros.columns if c.startswith('ros_')]
BOLTZ = ['boltz2_iptm', 'boltz2_ptm', 'boltz2_ipsae', 'boltz2_min_ipsae',
         'boltz2_complex_iplddt', 'boltz2_plddt', 'boltz2_complex_plddt',
         'boltz2_lis', 'boltz2_complex_pde', 'shape_complimentarity_boltz2_binder_ss',
         'esmfold_plddt', 'proteinmpnn_score', 'redesigned_proteinmpnn_score']
BOLTZ = [c for c in BOLTZ if c in flat.columns]
EXTRA = [c for c in ['expressed', 'expression-yield'] if c in flat.columns]

D = lab[ID_COLS + EXP_COLS].drop_duplicates('id').copy()
assert D.id.is_unique, 'duplicate design ids'
D = D.merge(mech[['id'] + PHYS].drop_duplicates('id'), on='id', how='left', validate='1:1')
D = D.merge(ros[['id'] + ROS].drop_duplicates('id'), on='id', how='left', validate='1:1')
D = D.merge(flat[['id'] + BOLTZ + EXTRA].drop_duplicates('id'), on='id',
            how='left', validate='1:1')
D = D.merge(best[['id', 'best_score']].drop_duplicates('id'), on='id', how='left', validate='1:1')

# tidy identity / label columns
D = D.rename(columns={'designMethod': 'design_method', 'seq': 'sequence',
                      'url': 'structure_url', 'binding': 'exp_binding',
                      'binding_strength': 'exp_binding_strength', 'kd': 'exp_kd_M',
                      'expressed': 'exp_expressed',
                      'expression-yield': 'exp_expression_yield',
                      'best_score': 'score_composite'})
D['length'] = D.sequence.str.len()
D['exp_binding'] = D.exp_binding.astype(str).str.lower().map({'true': True, 'false': False})
D['exp_kd_M'] = pd.to_numeric(D.exp_kd_M, errors='coerce')
D['exp_pkd'] = np.where(D.exp_binding.eq(True) & D.exp_kd_M.gt(0),
                        -np.log10(D.exp_kd_M.where(D.exp_kd_M.gt(0))), np.nan)
D.loc[D.exp_binding.eq(False), 'exp_pkd'] = 0.0
# prefix physics descriptors so every column's provenance is legible
ren = {c: f'phys_{c}' for c in PHYS}
D = D.rename(columns=ren)
PHYS_P = [ren[c] for c in PHYS]

ORDER = (['id', 'name', 'author', 'design_method', 'design_class', 'family',
          'sequence', 'length', 'target', 'structure_url',
          'exp_binding', 'exp_binding_strength', 'exp_kd_M', 'exp_pkd',
          'exp_expressed', 'exp_expression_yield',
          'score_composite'] + sorted(PHYS_P) + sorted(ROS) + sorted(BOLTZ))
D = D[[c for c in ORDER if c in D.columns]]
D.to_csv(f'{OUT}/designs.csv', index=False)
print(f"designs.csv            {D.shape[0]} rows x {D.shape[1]} cols")
print(f"  physics {len(PHYS_P)}  rosetta {len(ROS)}  boltz2 {len(BOLTZ)}")
print(f"  binders {int(D.exp_binding.sum())}  families {D.family.nunique()}")

# ------------------------------------------------------- metrics benchmark ---
def src_of(c):
    if c.startswith('ros_'): return 'Rosetta'
    if c.startswith('phys_'): return 'Physics'
    return 'Boltz-2'

frames = []
for f, ren_fn in [('unified_ranking.csv', lambda m: m),
                  ('rosetta_univariate.csv', lambda m: m),
                  ('univariate_benchmark.csv', lambda m: m)]:
    try: frames.append(pd.read_csv(f))
    except FileNotFoundError: pass
U = pd.read_csv('unified_ranking.csv')
U = U.rename(columns={'auroc_lo': 'auroc_ci_lo', 'auroc_hi': 'auroc_ci_hi',
                      'auprc_lo': 'auprc_ci_lo', 'auprc_hi': 'auprc_ci_hi'})
# align metric names with designs.csv
phys_set = set(PHYS)
U['metric'] = U.metric.str.replace(r'_[xy]$', '', regex=True)
U = (U.sort_values('AUPRC', ascending=False)
       .drop_duplicates('metric', keep='first'))     # _x/_y were the same metric twice
U['metric'] = U.metric.apply(lambda m: f'phys_{m}' if m in phys_set else m)
U['source'] = U.metric.apply(src_of)
# Pearson r against exp_pkd for every metric, computed from designs.csv so the
# column is complete rather than only covering the panelled subset
from scipy.stats import pearsonr
pk = D.exp_pkd.values
def _r(metric):
    if metric not in D.columns: return np.nan
    v = pd.to_numeric(D[metric], errors='coerce').values
    m = np.isfinite(v) & np.isfinite(pk)
    if m.sum() < 50 or np.nanstd(v[m]) < 1e-12: return np.nan
    return pearsonr(v[m], pk[m])[0]
U['pearson_vs_pkd'] = U.metric.map(_r)
U = U[['metric', 'source', 'direction', 'AUROC', 'auroc_ci_lo', 'auroc_ci_hi',
       'AUPRC', 'auprc_ci_lo', 'auprc_ci_hi', 'lift',
       *( ['pearson_vs_pkd'] if 'pearson_vs_pkd' in U.columns else [] ), 'n']]
U = U.sort_values('AUPRC', ascending=False)
U.to_csv(f'{OUT}/metrics_benchmark.csv', index=False)
print(f"metrics_benchmark.csv  {U.shape[0]} metrics  "
      f"({U.source.value_counts().to_dict()})")

# -------------------------------------------------------- model comparison ---
rows = []
cv = pd.read_csv('composite_variants.csv')
for _, r in cv.iterrows():
    rows.append(dict(model=r.tag, features=int(r.n_feat), AUROC=r.auroc, AUPRC=r.auprc,
                     P_at_10=r.get('10', np.nan), P_at_20=r.get('20', np.nan),
                     P_at_50=r.get('50', np.nan), validation='nested family-grouped CV'))
cs = json.load(open('compare_summary.json'))
# 'Physics, all terms' is already covered by composite_variants; only the two
# Rosetta-containing models are unique to this run
for tag, a, p, n in [('Rosetta InterfaceAnalyzer only', cs['auroc_ros'], cs['auprc_ros'], cs['n_ros']),
                     ('Physics + Rosetta combined', cs['auroc_both'], cs['auprc_both'],
                      cs['n_ros'] + cs['n_mine'])]:
    rows.append(dict(model=tag, features=n, AUROC=a, AUPRC=p,
                     P_at_10=np.nan, P_at_20=np.nan, P_at_50=np.nan,
                     validation='nested family-grouped CV'))
base = float(D.exp_binding.mean())
rows.append(dict(model='Random baseline', features=0, AUROC=0.5, AUPRC=base,
                 P_at_10=base, P_at_20=base, P_at_50=base, validation='-'))
MC = pd.DataFrame(rows).drop_duplicates('model').sort_values('AUPRC', ascending=False)
MC.to_csv(f'{OUT}/model_comparison.csv', index=False)
print(f"model_comparison.csv   {MC.shape[0]} models")

# ------------------------------------------------------------ coefficients ---
CF = pd.read_csv('composite_coefficients.csv')
CF['term'] = CF.term.apply(lambda t: f'phys_{t}')
CF = CF.rename(columns={'mean': 'standardise_mean', 'sd': 'standardise_sd'})
CF = CF.sort_values('coef', key=lambda v: v.abs(), ascending=False)
CF.to_csv(f'{OUT}/composite_coefficients.csv', index=False)
mj = json.load(open('composite_model.json'))
print(f"composite_coefficients.csv  {CF.kept.sum()} of {len(CF)} terms kept, "
      f"intercept {mj['intercept']:.4f}, C={mj['C']}")

# --------------------------------------------------------------- integrity ---
print("\n=== integrity checks ===")
print("  duplicate ids                :", int(D.id.duplicated().sum()))
print("  rows missing a physics value :", int(D[PHYS_P].isna().all(axis=1).sum()))
print("  rows missing a rosetta value :", int(D[ROS].isna().all(axis=1).sum()))
print("  rows missing boltz2 values   :", int(D[BOLTZ].isna().all(axis=1).sum()))
print("  exp_pkd set for every design :", bool(D.exp_pkd.notna().sum() >= (D.exp_binding == False).sum()))
print("  benchmark metrics present in designs.csv:",
      int(U.metric.isin(D.columns).sum()), "/", len(U))
miss = sorted(set(U.metric) - set(D.columns))
if miss: print("  NOT in designs.csv:", miss[:8])
json.dump({'n_designs': int(len(D)), 'n_binders': int(D.exp_binding.sum()),
           'base_rate': base, 'n_families': int(D.family.nunique()),
           'n_physics': len(PHYS_P), 'n_rosetta': len(ROS), 'n_boltz2': len(BOLTZ),
           'composite_intercept': mj['intercept'], 'composite_C': mj['C'],
           'composite_terms_kept': int(CF.kept.sum())},
          open(f'{OUT}/dataset_summary.json', 'w'), indent=1)
print("\nwrote 4 tables + dataset_summary.json")
