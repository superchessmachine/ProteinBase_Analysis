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
4. Ran **Rosetta `InterfaceAnalyzer`** (PyRosetta 2026.39, sidechains repacked) over the same
   1,029 complexes and compared head-to-head — see §7 of `FINDINGS.md`.

## Layout

```
FINDINGS.md          the report - read this
DATA_DICTIONARY.md   every column in every table, documented
figures/             01-13 (see below)
data/                designs.csv                 1,029 designs x 125 cols
                     metrics_benchmark.csv       97 metrics, AUROC/AUPRC + CIs
                     model_comparison.csv        9 fitted models
                     composite_coefficients.csv  the deliverable model
                     dataset_summary.json
code/                mech.py, mech2.py           descriptor engines
                     run_mech2.py                parallel driver
                     rosetta_ia.py               PyRosetta InterfaceAnalyzer
                     model_lasso.py              nested grouped CV + L1
                     best_composite.py           model selection
                     compare.py                  Rosetta vs physics
                     affinity.py                 K_D regression + confound check
                     consolidate.py              builds the published tables
                     plots3/4/5/6.py, make_report.py, make_dict.py
```

## Figures

| | |
|---|---|
| 01 | single physics descriptor AUROC |
| 02 | descriptor distributions |
| 03 | ROC + PR, honest CV |
| 04 | the leakage trap |
| 05 | L1 coefficients |
| 06 | enrichment / packing map |
| 07 | affinity ranking (negative result) |
| 08 | Rosetta metric AUROC |
| 09 | correlation scatters, physics vs Rosetta |
| 09_1 | every metric vs experimental affinity (PNAS style) |
| 10 | method comparison |
| 11 | ROC + PR, all methods |
| 12 | head-to-head, same concept |
| 13 | the best composite, standalone |


## Headline

| | AUROC | AUPRC (random 0.098) |
|---|---|---|
| **L1 logistic, 26 curated physics terms** | **0.709** | **0.328** |
| L1 logistic, Rosetta InterfaceAnalyzer | 0.651 | 0.185 |
| best single metric (Rosetta `sc_value`) | 0.678 | 0.200 |
| best single physics descriptor (Coulomb) | 0.685 | 0.188 |
| best single Boltz-2 metric (min ipSAE) | 0.638 | 0.198 |

**P@10 = 0.90** against a 9.8% base rate: submit ten, expect nine binders.

Top-10 submissions: **90% hit rate** vs 9.8% unfiltered.

The requested AUPRC > 0.9 was not reached and is not reachable here — even with deliberate
leakage the ceiling is 0.64. See §1 of `FINDINGS.md`.


## Reproducing

This repo is self-contained — the source export and all structures are included.

| input | where | size |
|---|---|---|
| `proteinbase_all_data_28_01_2026.csv` | repo root | 39 MB |
| `structures/*.cif` | 1,029 Boltz-2 predicted complexes | 446 MB |

**Provenance.** Both come from Adaptyv Bio's ProteinBase. The structures were fetched from
Adaptyv's public bucket (`proteinbase-pub.t3.storage.dev`) using the URLs carried in each
design's `evaluations` field; `designs.csv` preserves each one in `structure_url`. They are
**Boltz-2 predictions, not experimental structures** — for a non-binder the modelled complex
does not exist, which is the ceiling on everything computed from them.

The only thing not committed is `pdb/` — the two-chain PDBs Rosetta needs, derived from the
CIFs by `cif_to_pdb()` and regenerated automatically.

```bash
python code/labels.py        # target-matched labels + structure URLs
python code/cluster.py       # sequence-family assignment (for grouped CV)
python code/run_mech2.py     # 67 mechanistic descriptors over all complexes
python code/model_lasso.py   # nested family-grouped CV + L1 logistic
python code/affinity.py      # K_D regression + design-format confound check
python code/rosetta_ia.py    # PyRosetta InterfaceAnalyzer (needs the pyrosetta env)
python code/compare.py       # Rosetta vs hand-built head-to-head
python code/plots3.py        # figures 01-07
python code/plots4.py        # figures 08-12 (comparison)
python code/plots5.py        # figure 09_1
python code/plots6.py        # figure 13
python code/consolidate.py   # build the four published tables
python code/make_dict.py     # regenerate DATA_DICTIONARY.md
python code/make_report.py && python code/add_rosetta_section.py
```

Core analysis needs only `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib` — SASA,
H-bonding, cavity detection and shape complementarity are implemented directly in `code/mech.py`.
The Rosetta comparison additionally needs PyRosetta (free for academic use):

```bash
conda create -n pyrosetta python=3.11 numpy pandas scipy scikit-learn
pip install pyrosetta-installer
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta(serialization=True)"
```

## Caveats

- Calibrated on **one target** (Nipah glycoprotein-G) — the only ProteinBase round with enough
  paired structures and labels. Transfer to other targets is untested.
- Descriptors are computed on **predicted** complexes. For a non-binder that complex does not
  exist, which caps what any scoring function can achieve on this input.
- Benchmark with **sequence-family-grouped splits**. Plain k-fold inflates AUPRC by up to ~1.9×.
- Rosetta `InterfaceAnalyzer` must be run with sidechain repacking on these models; unrepacked
  `dG_separated` is clash-dominated (+319 vs +25 for the same structure).
