import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, pearsonr

R=pd.read_csv('mech2_scored.csv'); U=pd.read_csv('univariate_benchmark.csv')
CF=pd.read_csv('lasso_coefficients.csv'); S=json.load(open('summary.json'))
y=R.y.values; base=y.mean()
ap_l1=average_precision_score(y,R.score_l1); au_l1=roc_auc_score(y,R.score_l1)
ap_lk=average_precision_score(y,R.score_leaky); au_lk=roc_auc_score(y,R.score_leaky)
ap_co=average_precision_score(y,R.score_core); au_co=roc_auc_score(y,R.score_core)
o=np.argsort(-R.score_l1.values)
pk={k:y[o[:k]].mean() for k in (10,20,50,100,200)}
m=R.logkd.notna()&(R.y==1)
AS=json.load(open('affinity_summary.json'))
AU=pd.read_csv('affinity_univariate.csv')
rho=AS['spearman']; pr_=AS['pearson']

def tbl(df,cols,hdr,fmt=lambda v:f"{v:.3f}"):
    out=["| "+" | ".join(hdr)+" |","|"+"|".join(["---"]*len(hdr))+"|"]
    for _,r in df.iterrows():
        out.append("| "+" | ".join(fmt(r[c]) if isinstance(r[c],(int,float,np.floating)) else str(r[c]) for c in cols)+" |")
    return "\n".join(out)

top=U.head(15).copy(); top['ci']=top.apply(lambda r:f"{r.ci_lo:.3f}–{r.ci_hi:.3f}",axis=1)
kdt=AU.head(8)
n_kd=int(m.sum())

md=f"""# Mechanistic scoring of designed binders — ProteinBase

**Data** Adaptyv Bio ProteinBase dump `proteinbase_all_data_28_01_2026.csv` (5,253 designs).
**Scope** Nipah glycoprotein-G round — the only target with enough paired *predicted complex
structures* and *wet-lab binding labels* to calibrate a scoring function.

| | |
|---|---|
| designs scored | **{S['n']}** |
| experimental binders | **{S['binders']}** ({base:.1%} base rate) |
| sequence families (30% k-mer Jaccard) | **{S['families']}** |
| mechanistic descriptors computed | **{S['n_feat']}** |
| descriptors surviving L1 | **{S['n_kept']}** |

Every descriptor is classical geometry or physics computed from coordinates.
**No neural-network confidence scores are used anywhere in the model** — no pLDDT, no ipTM,
no ipSAE, no ProteinMPNN. The only model is L1-penalised logistic regression.

---

## 1. Headline result

| model | AUROC | AUPRC | lift over random |
|---|---|---|---|
| L1 logistic, all {S['n_feat']} physics descriptors | **{au_l1:.3f}** | **{ap_l1:.3f}** | {ap_l1/base:.1f}× |
| L1 logistic, curated physics subset | {au_co:.3f} | {ap_co:.3f} | {ap_co/base:.1f}× |
| best single descriptor ({U.iloc[0].descriptor}) | {U.iloc[0].AUROC:.3f} | {U.iloc[0].AUPRC:.3f} | {U.iloc[0].lift:.1f}× |
| random | 0.500 | {base:.3f} | 1.0× |

All figures are **nested family-grouped cross-validation**: no sequence family appears in both
train and test, and the L1 penalty is chosen on an inner grouped split.

### On the AUPRC > 0.9 target

**It was not reached, and I do not believe it is reachable on this data.** The honest ceiling here
is **AUPRC ≈ {ap_l1:.2f}**. Three structural reasons:

1. **The negatives are scored on fictional complexes.** Boltz-2 returns a complex for every
   binder-target pair. When the design does not bind, that complex does not exist — the
   descriptors are measured on a pose that is an artefact. No amount of better physics fixes an
   input that is wrong.
2. **The labels are coarse and noisy.** `Weak` binders sit at µM, expression failures are recorded
   as non-binders, and Strong/Medium/Weak thresholds are assay-dependent.
3. **Nobody reports this.** Published in-silico binder filters land at AUROC 0.70–0.85. An AUPRC
   of 0.9 at a 10% base rate implies ~99% specificity at 90% recall from a predicted structure —
   that would mean binder design is solved.

I also checked how far *deliberate cheating* gets, so the ceiling is not a matter of opinion.
Using random (non-grouped) CV, a gradient-boosted model, and feeding in `author`, `designMethod`
and the family id outright:

| configuration | AUROC | AUPRC |
|---|---|---|
| honest: L1 physics, family-grouped CV | 0.691 | **0.310** |
| leaky: GBM physics, random CV | 0.846 | 0.594 |
| leaky + metadata: GBM + author/method/class, random CV | 0.860 | 0.639 |
| leaky + family id handed to the model | 0.860 | 0.633 |

**Even with every form of leakage available, AUPRC reaches 0.64 — not 0.9.** The requested target
is not attainable on this dataset by honest means or dishonest ones. What the leaky rows show is
the size of the trap: a filter validated with plain k-fold looks roughly twice as good as it is.

---

## 2. Single descriptors

![single descriptor AUROC](figures/01_single_descriptor_auroc.png)

{tbl(top,['descriptor','direction','AUROC','ci','AUPRC','lift'],['descriptor','direction','AUROC','95% CI','AUPRC','lift'],lambda v:f"{v:.3f}" if isinstance(v,(float,np.floating)) else str(v))}

![distributions](figures/02_descriptor_distributions.png)

**No single physical descriptor is strong.** The best reaches AUROC ≈ {U.iloc[0].AUROC:.2f}, and its
confidence interval nearly touches 0.5. Interface quality is genuinely multi-factorial — which is
why the sparse combination in §3 roughly doubles the AUPRC of any one term.

### On buried unsatisfied H-bonds and voids

Both of your suggested metrics are in, and they behave differently from the textbook expectation:

- **Buried unsatisfied polars** discriminate in the predicted direction — fewer unsatisfied buried
  donors/acceptors means more binding — but weakly on its own
  (`buns_loose_density` AUROC {float(U[U.descriptor=='buns_loose_density'].AUROC.iloc[0]) if 'buns_loose_density' in U.descriptor.values else float('nan'):.3f}).
  It survives L1 selection, so it carries signal the burial terms do not.
- **Interface void** came out with the *opposite* sign to the hypothesis: more void associates with
  *more* binding. That is a confound — void volume scales with interface size, and large interfaces
  bind more often. Normalising it (`void_per_bsa`, `gap_index`) weakens it rather than flipping it,
  so the cavity signal is largely subsumed by buried area. Worth knowing before you filter on it.

---

## 3. The sparse physics model

![lasso coefficients](figures/05_lasso_coefficients.png)

L1 keeps **{S['n_kept']} of {S['n_feat']}** descriptors. Coefficients are in standardised units, so
magnitude is directly comparable.

{tbl(CF.head(20),['descriptor','coef','mean','sd'],['descriptor','coefficient','mean','sd'],lambda v:f"{v:.4f}")}

![roc pr](figures/03_roc_pr.png)

---

## 4. The leakage trap

![leakage](figures/04_leakage.png)

| cross-validation | AUROC | AUPRC |
|---|---|---|
| random 5-fold, L1 (families split across folds) | {au_lk:.3f} | **{ap_lk:.3f}** |
| family-grouped 5-fold, L1 (honest) | {au_l1:.3f} | **{ap_l1:.3f}** |
| random 5-fold, GBM (families split across folds) | 0.846 | **0.594** |

The single largest sequence family holds **82 designs and 23 of the 101 binders**. Split at random,
near-duplicate designs sit in train and test simultaneously and the model memorises families rather
than learning physics. **AUPRC inflates by {ap_lk/ap_l1:.1f}× for the linear model and ~1.9× for a
gradient-boosted one** — flexible models exploit the leak harder, which is exactly why the sparse
linear model you asked for is the safer choice here.

This matters for you directly: any filter you validate on ProteinBase with plain k-fold will look
far better than it performs on genuinely novel designs.

---

## 5. What it buys you

![enrichment](figures/06_enrichment.png)

| submissions | hit rate with physics filter | no filter |
|---|---|---|
| top 10 | **{pk[10]:.0%}** | {base:.0%} |
| top 20 | **{pk[20]:.0%}** | {base:.0%} |
| top 50 | **{pk[50]:.0%}** | {base:.0%} |
| top 100 | **{pk[100]:.0%}** | {base:.0%} |
| top 200 | **{pk[200]:.0%}** | {base:.0%} |

This is the number that matters for a competition with limited slots. The score is strong at the
very top and mediocre in the middle — exactly the shape you want when you only submit your best.

---

## 6. Ranking affinity — a negative result

![affinity](figures/07_affinity_ranking.png)

Among the **{n_kd}** binders with a measured K_D, a Ridge model on the physics descriptors reaches
**Pearson r = {pr_:.3f}, Spearman rho = {rho:.3f}** under family-grouped CV.

**That number is not real physics.** Stratifying by design format destroys it:

| subset | n | Pearson r (grouped CV) |
|---|---|---|
| all binders | 100 | **0.503** |
| de novo only (Other + Miniprotein) | 87 | 0.186 |
| `Other` class only (no scFv) | 53 | **-0.302** |

The model ranks affinity only because *formats* differ in affinity — scFv designs are tighter
(median K_D 10 nM) than miniproteins (50 nM) — and the compositional descriptors that load
hardest (`frac_helix`, `n_cys`, `frac_aromatic`, `frac_sheet`) are format fingerprints, not
interface energetics. Within one format the correlation is gone or inverted.

{tbl(kdt,['descriptor','r','rho'],['descriptor','Pearson r','Spearman rho'])}

**Conclusion: these descriptors do not rank K_D.** Use the score as a binary triage filter only.
Had I reported the 0.503 without the stratification check, it would have looked like a strong
affinity predictor and been worthless in practice.

---

## 7. How to use this

1. **Filter with the sparse physics model, not with ipTM.** ipTM ranks at the base rate on this
   data. The physics composite is a genuine multiple-fold enrichment at the top of the list.
2. **Validate with grouped splits.** If you benchmark a filter on ProteinBase, cluster by sequence
   first or you will overestimate it by roughly a factor of two.
3. **Expression is a hard gate.** Across the full dataset, P(binds | did not express) = 0.000
   (n=260). Nanobody and scFv formats express worst (78–89%); miniproteins 96.7%.
4. **Don't filter on void volume alone** — on this data it tracks interface size, not packing defects.

## Reproducing

```
code/mech.py          SASA, H-bonds, BUNS, voids, shape complementarity
code/mech2.py         full descriptor set (LJ, desolvation, geometry, developability)
code/run_mech2.py     parallel driver over 1,029 complexes
code/model_lasso.py   nested family-grouped CV + L1 model
code/plots2.py        all figures
data/mech2_scored.csv per-design descriptors, labels, out-of-fold scores
data/univariate_benchmark.csv, data/lasso_coefficients.csv
```
"""
open('proteinbase_data/FINDINGS.md','w').write(md)
print("report written:", len(md), "chars")
