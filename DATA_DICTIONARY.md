# Data dictionary

Four tables, one row per entity, no redundancy between them.

| file | rows | what it is |
|---|---|---|
| `designs.csv` | 1029 | one row per design: identity, experimental labels, all 108 descriptors, composite score |
| `metrics_benchmark.csv` | 97 | one row per single metric: AUROC/AUPRC with bootstrap CIs |
| `model_comparison.csv` | 10 | one row per fitted model: nested family-grouped CV results |
| `composite_coefficients.csv` | 26 | the deliverable model's terms, weights and scaling constants |

**Dataset:** 1029 designs, 101 binders (9.8% base rate),
643 sequence families. Target: Nipah glycoprotein-G.

> **Read this before modelling.** Group by `family` when splitting. The largest family
> holds 82 designs and 23 of the 101 binders; a plain k-fold split inflates
> AUPRC by up to ~1.9x because near-duplicate designs land in train and test together.

---

## designs.csv

### Identity

| column | dtype | non-null | description |
|---|---|---|---|
| `id` | object | 1029/1029 | ProteinBase design identifier |
| `name` | object | 1029/1029 | Author-supplied design name |
| `author` | object | 1029/1029 | Submitting team |
| `design_method` | object | 589/1029 | Method string the author declared |
| `design_class` | object | 1021/1029 | Adaptyv structural class (Miniprotein, Nanobody, scFv, Peptide, Other) |
| `family` | int64 | 1029/1029 | Sequence-family id, single-linkage clustering at 0.30 5-mer Jaccard. **Group by this for cross-validation.** |
| `sequence` | object | 1029/1029 | Binder amino-acid sequence |
| `length` | int64 | 1029/1029 | Binder length in residues |
| `target` | object | 1029/1029 | Binding target (all rows: nipah-glycoprotein-g) |
| `structure_url` | object | 1029/1029 | Public URL of the Boltz-2 predicted complex (mmCIF) |

### Experimental measurements

These are the labels. `exp_pkd` encodes non-binders as 0 so binders and non-binders share one axis.

| column | dtype | non-null | description |
|---|---|---|---|
| `exp_binding` | bool | 1029/1029 | Wet-lab binding call (True/False). **The label.** |
| `exp_binding_strength` | object | 1029/1029 | Strong / Medium / Weak / None |
| `exp_kd_M` | float64 | 100/1029 | Measured dissociation constant, molar. Null for non-binders. |
| `exp_pkd` | float64 | 1028/1029 | -log10(exp_kd_M) for binders; 0 for non-binders. Higher = tighter. |
| `exp_expressed` | object | 1028/1029 | Whether the construct expressed |
| `exp_expression_yield` | float64 | 1025/1029 | Expression yield where reported |

### Model output

| column | dtype | non-null | description |
|---|---|---|---|
| `score_composite` | float64 | 1029/1029 | Out-of-fold composite score, nested family-grouped CV. The deliverable ranking. |

### Physics descriptors (66)

Computed from the Boltz-2 predicted complex by `code/mech.py` and `code/mech2.py`. Geometry and classical physics only; no learned parameters.

| column | dtype | non-null | description |
|---|---|---|---|
| `phys_agg_max` | float64 | 1029/1029 | Maximum 5-residue Kyte-Doolittle hydrophobic window |
| `phys_aliphatic_index` | float64 | 1029/1029 | - |
| `phys_anchor_bsa` | float64 | 1029/1029 | Buried area of the single most deeply buried binder residue |
| `phys_anchor_frac` | float64 | 1029/1029 | - |
| `phys_binder_len` | int64 | 1029/1029 | - |
| `phys_bsa` | float64 | 1029/1029 | Buried surface area, complex vs free chains (Shrake-Rupley SASA) |
| `phys_bsa_apolar` | float64 | 1029/1029 | Apolar component of buried area |
| `phys_bsa_per_res` | float64 | 1029/1029 | - |
| `phys_bsa_polar` | float64 | 1029/1029 | Polar component of buried area |
| `phys_buns` | int64 | 1029/1029 | Buried unsatisfied polar atoms (strict burial cutoff) |
| `phys_buns_A` | int64 | 1029/1029 | - |
| `phys_buns_density` | float64 | 1029/1029 | phys_buns per 1000 A^2 buried area |
| `phys_buns_loose` | int64 | 1029/1029 | Buried unsatisfied polar atoms (loose burial cutoff) |
| `phys_buns_loose_A` | int64 | 1029/1029 | - |
| `phys_buns_loose_density` | float64 | 1029/1029 | phys_buns_loose per 1000 A^2 buried area |
| `phys_charge_density` | float64 | 1029/1029 | - |
| `phys_ct_density` | float64 | 1029/1029 | - |
| `phys_dG_solv` | float64 | 1029/1029 | Desolvation free energy, Eisenberg-McLachlan atomic solvation parameters |
| `phys_dG_solv_per_bsa` | float64 | 1029/1029 | - |
| `phys_ddg_est` | float64 | 1029/1029 | Summed energy estimate: LJ + desolvation + H-bond + electrostatics |
| `phys_ddg_per_bsa` | float64 | 1029/1029 | phys_ddg_est per 1000 A^2 buried area |
| `phys_eccentricity` | float64 | 1029/1029 | Principal-axis ratio of the interface patch |
| `phys_elec` | float64 | 1029/1029 | Coulomb energy across the interface, distance-dependent dielectric (eps = 4r) |
| `phys_elec_att` | float64 | 1029/1029 | Attractive component of phys_elec |
| `phys_elec_per_bsa` | float64 | 1029/1029 | - |
| `phys_elec_rep` | float64 | 1029/1029 | Repulsive component of phys_elec |
| `phys_f_apolar` | float64 | 1029/1029 | Apolar fraction of buried area |
| `phys_frac_aromatic` | float64 | 1029/1029 | - |
| `phys_frac_charged` | float64 | 1029/1029 | - |
| `phys_frac_helix` | float64 | 1029/1029 | Helix fraction from backbone dihedrals |
| `phys_frac_hydrophobic` | float64 | 1029/1029 | - |
| `phys_frac_sheet` | float64 | 1029/1029 | Sheet fraction from backbone dihedrals |
| `phys_gap_index` | float64 | 1029/1029 | Laskowski gap volume index |
| `phys_gravy` | float64 | 1029/1029 | - |
| `phys_hb_density` | float64 | 1029/1029 | - |
| `phys_hb_energy` | float64 | 1029/1029 | Geometry-weighted interface hydrogen-bond energy |
| `phys_hb_per_bsa` | float64 | 1029/1029 | - |
| `phys_ifc_res` | int64 | 1029/1029 | - |
| `phys_ifc_res_A` | int64 | 1029/1029 | - |
| `phys_lj_atr` | float64 | 1029/1029 | Softened Lennard-Jones attractive term across the interface |
| `phys_lj_atr_per_bsa` | float64 | 1029/1029 | - |
| `phys_lj_rep` | float64 | 1029/1029 | Softened Lennard-Jones repulsive term |
| `phys_n_anchor50` | int64 | 1029/1029 | Binder residues burying more than 50 A^2 |
| `phys_n_anchor80` | int64 | 1029/1029 | Binder residues burying more than 80 A^2 |
| `phys_n_apolar_ct` | int64 | 1029/1029 | - |
| `phys_n_cation_pi` | int64 | 1029/1029 | Count of cation-pi interactions |
| `phys_n_clash` | int64 | 1029/1029 | Count of steric clashes |
| `phys_n_cys` | int64 | 1029/1029 | - |
| `phys_n_hb` | int64 | 1029/1029 | Count of interface hydrogen bonds |
| `phys_n_pi_stack` | int64 | 1029/1029 | Count of pi-stacking interactions |
| `phys_n_salt` | int64 | 1029/1029 | Count of interface salt bridges |
| `phys_n_segments` | int64 | 1029/1029 | Contiguous interface segments on the binder |
| `phys_net_charge` | int64 | 1029/1029 | - |
| `phys_planarity` | float64 | 1029/1029 | RMS deviation of interface atoms from their least-squares plane (Jones & Thornton) |
| `phys_rel_contact_order` | float64 | 1029/1029 | Relative contact order of the binder fold |
| `phys_rg` | float64 | 1029/1029 | Binder radius of gyration |
| `phys_rg_norm` | float64 | 1029/1029 | Rg normalised by 2.2*N^0.38 (compactness) |
| `phys_salt_per_bsa` | float64 | 1029/1029 | - |
| `phys_sc` | float64 | 1029/1029 | Lawrence-Colman-style complementarity on the contact shell (vdW surface proxy; not equal to Rosetta sc_value) |
| `phys_seg_per_res` | float64 | 1029/1029 | - |
| `phys_surf_hydrophobic` | float64 | 1029/1029 | - |
| `phys_surf_hydrophobic_frac` | float64 | 1029/1029 | Exposed hydrophobic fraction of the free binder surface |
| `phys_top3_bsa` | float64 | 1029/1029 | Summed burial of the three deepest binder residues |
| `phys_void` | float64 | 1029/1029 | Interface cavity volume, grid detection (A^3) |
| `phys_void_loose` | float64 | 1029/1029 | Interface cavity volume, relaxed buriedness (A^3) |
| `phys_void_per_bsa` | float64 | 1029/1029 | - |

### Rosetta InterfaceAnalyzer (29)

PyRosetta 2026.39, sidechains repacked in both the complex and the separated state. Unrepacked dG_separated is clash-dominated and unusable on predicted models.

| column | dtype | non-null | description |
|---|---|---|---|
| `ros_aromatic_dG_frac` | float64 | 0/1029 | - |
| `ros_aromatic_dSASA` | float64 | 0/1029 | - |
| `ros_centroid_dG` | float64 | 1029/1029 | - |
| `ros_complex_E` | float64 | 1029/1029 | - |
| `ros_complexed_SASA` | float64 | 1029/1029 | - |
| `ros_cplx_int_score` | float64 | 0/1029 | - |
| `ros_crossterm` | float64 | 1029/1029 | Rosetta crossterm_interface_energy |
| `ros_crossterm_dSASA` | float64 | 1029/1029 | - |
| `ros_dG` | float64 | 1029/1029 | Rosetta dG_separated (sidechains repacked in complex and separated states) |
| `ros_dG_dSASA` | float64 | 1029/1029 | Rosetta dG_separated / dSASA |
| `ros_dSASA` | float64 | 1029/1029 | Rosetta interface buried area |
| `ros_dhSASA` | float64 | 1029/1029 | Rosetta hydrophobic buried area |
| `ros_gly_dG` | float64 | 1029/1029 | - |
| `ros_hb_E` | float64 | 1029/1029 | Rosetta total interface hydrogen-bond energy |
| `ros_hb_E_frac` | float64 | 1029/1029 | - |
| `ros_hb_per_dSASA` | float64 | 1029/1029 | - |
| `ros_hbonds` | float64 | 1029/1029 | Rosetta interface hydrogen-bond count |
| `ros_int_surf_frac` | float64 | 0/1029 | - |
| `ros_nres` | float64 | 1029/1029 | Rosetta interface residue count |
| `ros_packstat` | float64 | 1029/1029 | Rosetta packstat |
| `ros_sc` | float64 | 1029/1029 | Rosetta sc_value, true Lawrence-Colman shape complementarity |
| `ros_sep_int_score` | float64 | 0/1029 | - |
| `ros_separated_E` | float64 | 0/1029 | - |
| `ros_separated_SASA` | float64 | 1029/1029 | - |
| `ros_ss_helix` | float64 | 0/1029 | - |
| `ros_ss_loop` | float64 | 0/1029 | - |
| `ros_ss_sheet` | float64 | 0/1029 | - |
| `ros_unsat` | float64 | 1029/1029 | Rosetta delta_unsat_hbonds |
| `ros_unsat_per_dSASA` | float64 | 1029/1029 | ros_unsat per 1000 A^2 dSASA |

### Boltz-2 and other predictor outputs (13)

Shipped in the ProteinBase export; neural-network confidence values, not physics. Excluded from the composite model.

| column | dtype | non-null | description |
|---|---|---|---|
| `boltz2_complex_iplddt` | float64 | 1028/1029 | Boltz-2 complex interface pLDDT |
| `boltz2_complex_pde` | float64 | 1028/1029 | - |
| `boltz2_complex_plddt` | float64 | 1028/1029 | - |
| `boltz2_ipsae` | float64 | 1028/1029 | Boltz-2 ipSAE |
| `boltz2_iptm` | float64 | 1028/1029 | Boltz-2 interface pTM |
| `boltz2_lis` | float64 | 1028/1029 | - |
| `boltz2_min_ipsae` | float64 | 1028/1029 | Boltz-2 minimum ipSAE |
| `boltz2_plddt` | float64 | 1028/1029 | Boltz-2 pLDDT |
| `boltz2_ptm` | float64 | 1028/1029 | Boltz-2 pTM |
| `esmfold_plddt` | float64 | 1020/1029 | ESMFold pLDDT of the binder monomer |
| `proteinmpnn_score` | float64 | 1020/1029 | ProteinMPNN score |
| `redesigned_proteinmpnn_score` | float64 | 1020/1029 | - |
| `shape_complimentarity_boltz2_binder_ss` | float64 | 1028/1029 | - |


---

## metrics_benchmark.csv

| column | description |
|---|---|
| `metric` | column name in `designs.csv` |
| `source` | Physics / Rosetta / Boltz-2 |
| `direction` | whether higher or lower values indicate binding |
| `AUROC`, `auroc_ci_lo`, `auroc_ci_hi` | AUROC with 95% bootstrap CI |
| `AUPRC`, `auprc_ci_lo`, `auprc_ci_hi` | AUPRC with 95% bootstrap CI (random = 0.098) |
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
logit = -0.7694 + sum(coef_i * z_i)
score = 1 / (1 + exp(-logit))
```

L1 (C=1.0) keeps 25 of 26 terms.

> **The coefficients are not individually interpretable.** `phys_bsa`, `phys_lj_atr`,
> `phys_dG_solv` and `phys_n_anchor50` correlate at |r| = 0.90-0.98 (VIF in the thousands);
> they all measure interface size. The fit takes a weighted contrast among them, so a
> single coefficient does not describe that term's effect in isolation. Collinearity
> degrades coefficient interpretation, not prediction.
