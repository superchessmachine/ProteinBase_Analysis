# ProteinBase_Analysis

**Mechanistic (physics-only) scoring of designed protein binders**

Physics-only scoring of Adaptyv Bio competition designs. **Start with `FINDINGS.md`.**

## What this is

Adaptyv's ProteinBase dump (`proteinbase_all_data_28_01_2026.csv`, 5,253 designs) contains
Boltz-2 *predicted complex structures* alongside *wet-lab binding results*. That makes it a
calibration set: you can measure which scores actually predicted binding instead of guessing.

The Nipah glycoprotein-G round is the only target with enough paired structures + labels
(1,029 designs, 101 binders, 9.8% hit rate) to fit anything.

## Approach

1. Downloaded all 1,029 labelled Boltz-2 complexes from ProteinBase's public storage.
2. Computed **66 mechanistic interface descriptors** from coordinates alone — no neural-network
   confidence scores anywhere:

   | block | descriptors | source |
   |---|---|---|
   | burial | `bsa`, `bsa_apolar`, `f_apolar`, `bsa_per_res` | Shrake-Rupley SASA, complex vs free chains |
   | unsatisfied polars | `buns`, `buns_loose`, `buns_density` | Rosetta-style buried unsatisfied donors/acceptors |
   | cavities | `void`, `void_loose`, `void_per_bsa`, `gap_index` | grid cavity detection; Laskowski gap index |
   | complementarity | `sc` | Lawrence & Colman-style, contact shell |
   | energetics | `lj_atr`, `lj_rep`, `dG_solv`, `elec`, `hb_energy`, `ddg_est` | softened 6-12, Eisenberg-McLachlan ASP, Coulomb (eps=4r) |
   | interactions | `n_hb`, `n_salt`, `n_cation_pi`, `n_pi_stack`, `n_clash` | geometric criteria |
   | hotspots | `anchor_bsa`, `top3_bsa`, `n_anchor50/80` | per-residue burial depth |
   | interface shape | `planarity`, `eccentricity`, `n_segments` | Jones & Thornton |
   | developability | `rg`, `rg_norm`, `rel_contact_order`, `frac_helix/sheet`, `agg_max`, `aliphatic_index`, `surf_hydrophobic_frac`, `n_cys` | backbone dihedrals, composition |

3. Benchmarked each descriptor against experimental binding, then fitted **L1-penalised logistic
   regression** under **nested family-grouped cross-validation**.

## Layout

```
FINDINGS.md    the report — read this
figures/       01 single-descriptor AUROC      05 L1 coefficients
               02 descriptor distributions     06 enrichment / packing map
               03 ROC + PR (honest CV)         07 affinity ranking (negative result)
               04 leakage
data/          mech2_scored.csv         per-design descriptors, labels, out-of-fold scores
               univariate_benchmark.csv AUROC/AUPRC per descriptor + bootstrap CIs
               lasso_coefficients.csv   the fitted sparse model
               affinity_*.csv           K_D regression outputs
               labels_fam.csv           sequence-family assignments
code/          mech.py, mech2.py        descriptor engines
               run_mech2.py             parallel driver
               model_lasso.py           nested grouped CV + L1
               affinity.py              K_D regression + confound check
               plots2.py, make_report.py
```

## Headline

| | AUROC | AUPRC (base 0.098) |
|---|---|---|
| L1 logistic, curated physics | 0.709 | 0.328 |
| best single descriptor | 0.685 | 0.188 |

Top-10 submissions: **90% hit rate** vs 9.8% unfiltered.

The requested AUPRC > 0.9 was not reached and is not reachable here — even with deliberate
leakage the ceiling is 0.64. See §1 of `FINDINGS.md`.


## Reproducing

Two inputs are not committed (see `.gitignore`):

1. **`proteinbase_all_data_28_01_2026.csv`** — the ProteinBase export (5,253 designs, 39 MB),
   dated 2026-01-28. Place it in the repo root.
2. **Complex structures** — not redistributed here. Each design's `evaluations` field carries a
   public URL for its Boltz-2 predicted complex (`boltz2_structure_prediction`); `code/labels.py`
   extracts them and they download into `cif/`.

Then:

```bash
python code/labels.py        # extract target-matched labels + structure URLs
# download the 1,029 CIFs listed by labels.py into cif/
python code/cluster.py       # sequence-family assignment (for grouped CV)
python code/run_mech2.py     # 67 mechanistic descriptors over all complexes (parallel)
python code/model_lasso.py   # nested family-grouped CV + L1 logistic model
python code/affinity.py      # K_D regression + design-format confound check
python code/plots2.py        # figures
python code/make_report.py   # regenerate FINDINGS.md
```

Requires `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`. No structural-biology
packages needed — SASA, H-bonding, cavity detection and shape complementarity are implemented
directly in `code/mech.py`.

## Caveats

- Everything here is calibrated on **one target** (Nipah glycoprotein-G) — the only round in
  ProteinBase with enough paired structures and labels. Transfer to other targets is untested.
- Descriptors are computed on **predicted** complexes. For a non-binder that complex does not
  exist, which is the ceiling on what any scoring function can do with this input.
- Benchmark with **sequence-family-grouped splits**. Plain k-fold inflates AUPRC by up to ~1.9×
  on this dataset.
