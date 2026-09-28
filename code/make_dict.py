"""Generate DATA_DICTIONARY.md from the published tables."""
import pandas as pd, numpy as np, json

OUT = 'proteinbase_data'
D = pd.read_csv(f'{OUT}/data/designs.csv')
U = pd.read_csv(f'{OUT}/data/metrics_benchmark.csv')
S = json.load(open(f'{OUT}/data/dataset_summary.json'))

DESC = {
 'id': 'ProteinBase design identifier',
 'name': 'Author-supplied design name',
 'author': 'Submitting team',
 'design_method': 'Method string the author declared',
 'design_class': 'Adaptyv structural class (Miniprotein, Nanobody, scFv, Peptide, Other)',
 'family': 'Sequence-family id, single-linkage clustering at 0.30 5-mer Jaccard. **Group by this for cross-validation.**',
 'sequence': 'Binder amino-acid sequence',
 'length': 'Binder length in residues',
 'target': 'Binding target (all rows: nipah-glycoprotein-g)',
 'structure_url': 'Public URL of the Boltz-2 predicted complex (mmCIF)',
 'exp_binding': 'Wet-lab binding call (True/False). **The label.**',
 'exp_binding_strength': 'Strong / Medium / Weak / None',
 'exp_kd_M': 'Measured dissociation constant, molar. Null for non-binders.',
 'exp_pkd': '-log10(exp_kd_M) for binders; 0 for non-binders. Higher = tighter.',
 'exp_expressed': 'Whether the construct expressed',
 'exp_expression_yield': 'Expression yield where reported',
 'score_composite': 'Out-of-fold composite score, nested family-grouped CV. The deliverable ranking.',
 'phys_bsa': 'Buried surface area, complex vs free chains (Shrake-Rupley SASA)',
 'phys_bsa_apolar': 'Apolar component of buried area',
 'phys_bsa_polar': 'Polar component of buried area',
 'phys_f_apolar': 'Apolar fraction of buried area',
 'phys_buns': 'Buried unsatisfied polar atoms (strict burial cutoff)',
 'phys_buns_loose': 'Buried unsatisfied polar atoms (loose burial cutoff)',
 'phys_buns_density': 'phys_buns per 1000 A^2 buried area',
 'phys_buns_loose_density': 'phys_buns_loose per 1000 A^2 buried area',
 'phys_void': 'Interface cavity volume, grid detection (A^3)',
 'phys_void_loose': 'Interface cavity volume, relaxed buriedness (A^3)',
 'phys_gap_index': 'Laskowski gap volume index',
 'phys_sc': 'Lawrence-Colman-style complementarity on the contact shell (vdW surface proxy; not equal to Rosetta sc_value)',
 'phys_lj_atr': 'Softened Lennard-Jones attractive term across the interface',
 'phys_lj_rep': 'Softened Lennard-Jones repulsive term',
 'phys_dG_solv': 'Desolvation free energy, Eisenberg-McLachlan atomic solvation parameters',
 'phys_elec': 'Coulomb energy across the interface, distance-dependent dielectric (eps = 4r)',
 'phys_elec_att': 'Attractive component of phys_elec',
 'phys_elec_rep': 'Repulsive component of phys_elec',
 'phys_hb_energy': 'Geometry-weighted interface hydrogen-bond energy',
 'phys_n_hb': 'Count of interface hydrogen bonds',
 'phys_n_salt': 'Count of interface salt bridges',
 'phys_n_cation_pi': 'Count of cation-pi interactions',
 'phys_n_pi_stack': 'Count of pi-stacking interactions',
 'phys_n_clash': 'Count of steric clashes',
 'phys_anchor_bsa': 'Buried area of the single most deeply buried binder residue',
 'phys_top3_bsa': 'Summed burial of the three deepest binder residues',
 'phys_n_anchor50': 'Binder residues burying more than 50 A^2',
 'phys_n_anchor80': 'Binder residues burying more than 80 A^2',
 'phys_planarity': 'RMS deviation of interface atoms from their least-squares plane (Jones & Thornton)',
 'phys_eccentricity': 'Principal-axis ratio of the interface patch',
 'phys_n_segments': 'Contiguous interface segments on the binder',
 'phys_rg': 'Binder radius of gyration',
 'phys_rg_norm': 'Rg normalised by 2.2*N^0.38 (compactness)',
 'phys_rel_contact_order': 'Relative contact order of the binder fold',
 'phys_frac_helix': 'Helix fraction from backbone dihedrals',
 'phys_frac_sheet': 'Sheet fraction from backbone dihedrals',
 'phys_agg_max': 'Maximum 5-residue Kyte-Doolittle hydrophobic window',
 'phys_surf_hydrophobic_frac': 'Exposed hydrophobic fraction of the free binder surface',
 'phys_ddg_est': 'Summed energy estimate: LJ + desolvation + H-bond + electrostatics',
 'phys_ddg_per_bsa': 'phys_ddg_est per 1000 A^2 buried area',
 'ros_dG': 'Rosetta dG_separated (sidechains repacked in complex and separated states)',
 'ros_dG_dSASA': 'Rosetta dG_separated / dSASA',
 'ros_dSASA': 'Rosetta interface buried area',
 'ros_dhSASA': 'Rosetta hydrophobic buried area',
 'ros_sc': 'Rosetta sc_value, true Lawrence-Colman shape complementarity',
 'ros_packstat': 'Rosetta packstat',
 'ros_unsat': 'Rosetta delta_unsat_hbonds',
 'ros_unsat_per_dSASA': 'ros_unsat per 1000 A^2 dSASA',
 'ros_hb_E': 'Rosetta total interface hydrogen-bond energy',
 'ros_hbonds': 'Rosetta interface hydrogen-bond count',
 'ros_nres': 'Rosetta interface residue count',
 'ros_crossterm': 'Rosetta crossterm_interface_energy',
 'boltz2_iptm': 'Boltz-2 interface pTM',
 'boltz2_ptm': 'Boltz-2 pTM',
 'boltz2_ipsae': 'Boltz-2 ipSAE',
 'boltz2_min_ipsae': 'Boltz-2 minimum ipSAE',
 'boltz2_complex_iplddt': 'Boltz-2 complex interface pLDDT',
 'boltz2_plddt': 'Boltz-2 pLDDT',
 'esmfold_plddt': 'ESMFold pLDDT of the binder monomer',
 'proteinmpnn_score': 'ProteinMPNN score',
}

def block(cols, title, note=''):
    out = [f'### {title}', '']
    if note: out += [note, '']
    out += ['| column | dtype | non-null | description |', '|---|---|---|---|']
    for c in cols:
        if c not in D.columns: continue
        nn = int(D[c].notna().sum())
        out.append(f'| `{c}` | {D[c].dtype} | {nn}/{len(D)} | {DESC.get(c, "-")} |')
    return '\n'.join(out) + '\n'

ident = ['id','name','author','design_method','design_class','family','sequence',
         'length','target','structure_url']
exp = [c for c in D.columns if c.startswith('exp_')]
score = ['score_composite']
phys = sorted(c for c in D.columns if c.startswith('phys_'))
ros = sorted(c for c in D.columns if c.startswith('ros_'))
bz = [c for c in D.columns if c not in ident + exp + score + phys + ros]

md = f"""# Data dictionary

Four tables, one row per entity, no redundancy between them.

| file | rows | what it is |
|---|---|---|
| `designs.csv` | {len(D)} | one row per design: identity, experimental labels, all {S['n_physics']+S['n_rosetta']+S['n_boltz2']} descriptors, composite score |
| `metrics_benchmark.csv` | {len(U)} | one row per single metric: AUROC/AUPRC with bootstrap CIs |
| `model_comparison.csv` | 10 | one row per fitted model: nested family-grouped CV results |
| `composite_coefficients.csv` | {len(pd.read_csv(f'{OUT}/data/composite_coefficients.csv'))} | the deliverable model's terms, weights and scaling constants |

**Dataset:** {S['n_designs']} designs, {S['n_binders']} binders ({S['base_rate']:.1%} base rate),
{S['n_families']} sequence families. Target: Nipah glycoprotein-G.

> **Read this before modelling.** Group by `family` when splitting. The largest family
> holds 82 designs and 23 of the {S['n_binders']} binders; a plain k-fold split inflates
> AUPRC by up to ~1.9x because near-duplicate designs land in train and test together.

---

## designs.csv

{block(ident, 'Identity')}
{block(exp, 'Experimental measurements', 'These are the labels. `exp_pkd` encodes non-binders as 0 so binders and non-binders share one axis.')}
{block(score, 'Model output')}
{block(phys, f'Physics descriptors ({len(phys)})', 'Computed from the Boltz-2 predicted complex by `code/mech.py` and `code/mech2.py`. Geometry and classical physics only; no learned parameters.')}
{block(ros, f'Rosetta InterfaceAnalyzer ({len(ros)})', 'PyRosetta 2026.39, sidechains repacked in both the complex and the separated state. Unrepacked dG_separated is clash-dominated and unusable on predicted models.')}
{block(bz, f'Boltz-2 and other predictor outputs ({len(bz)})', 'Shipped in the ProteinBase export; neural-network confidence values, not physics. Excluded from the composite model.')}

---

## metrics_benchmark.csv

| column | description |
|---|---|
| `metric` | column name in `designs.csv` |
| `source` | Physics / Rosetta / Boltz-2 |
| `direction` | whether higher or lower values indicate binding |
| `AUROC`, `auroc_ci_lo`, `auroc_ci_hi` | AUROC with 95% bootstrap CI |
| `AUPRC`, `auprc_ci_lo`, `auprc_ci_hi` | AUPRC with 95% bootstrap CI (random = {S['base_rate']:.3f}) |
| `lift` | AUPRC divided by the base rate |
| `pearson_vs_pkd` | Pearson r against `exp_pkd`, where computed |
| `n` | designs with that metric present |

Sign is applied before scoring, so AUROC is always >= 0.5 and `direction` records which way.

## model_comparison.csv

`model`, `features`, `AUROC`, `AUPRC`, `P_at_10`, `P_at_20`, `P_at_50`, `validation`.
All entries use nested family-grouped cross-validation.

## composite_coefficients.csv

`term`, `coef`, `standardise_mean`, `standardise_sd`, `kept`.

To score a new design:

```
z_i   = (x_i - standardise_mean_i) / standardise_sd_i
logit = {S['composite_intercept']:.4f} + sum(coef_i * z_i)
score = 1 / (1 + exp(-logit))
```

L1 (C={S['composite_C']}) keeps {S['composite_terms_kept']} of {len(pd.read_csv(f'{OUT}/data/composite_coefficients.csv'))} terms.

> **The coefficients are not individually interpretable.** `phys_bsa`, `phys_lj_atr`,
> `phys_dG_solv` and `phys_n_anchor50` correlate at |r| = 0.90-0.98 (VIF in the thousands);
> they all measure interface size. The fit takes a weighted contrast among them, so a
> single coefficient does not describe that term's effect in isolation. Collinearity
> degrades coefficient interpretation, not prediction.
"""
open(f'{OUT}/DATA_DICTIONARY.md', 'w').write(md)
print(f'DATA_DICTIONARY.md written ({len(md)} chars)')
