"""Insert the Rosetta comparison section into FINDINGS.md."""
import pandas as pd, numpy as np, json

RU = pd.read_csv('rosetta_univariate.csv')
HH = pd.read_csv('head_to_head.csv')
S = json.load(open('compare_summary.json'))
base = S['base']

RP = {'ros_sc':'`sc_value` (shape complementarity)','ros_dG':'`dG_separated`',
 'ros_dG_dSASA':'`dG_separated/dSASA`','ros_dSASA':'`dSASA_int`','ros_dhSASA':'`dSASA_hydrophobic`',
 'ros_packstat':'`packstat`','ros_unsat':'`delta_unsat_hbonds`',
 'ros_unsat_per_dSASA':'`delta_unsat_hbonds/dSASA`','ros_hb_E':'`total_hb_E`',
 'ros_hb_E_frac':'`hbond_E_fraction`','ros_hbonds':'`interface_hbonds`',
 'ros_hb_per_dSASA':'`total_hb_E/dSASA`','ros_nres':'`interface_nres`',
 'ros_crossterm':'`crossterm_interface_energy`','ros_crossterm_dSASA':'`crossterm/dSASA`',
 'ros_complex_E':'`complex_total_energy`','ros_complexed_SASA':'`complexed_SASA`',
 'ros_separated_SASA':'`separated_SASA`'}

def rows(df, cols, fmt):
    return '\n'.join('| ' + ' | '.join(fmt(r[c], c) for c in cols) + ' |' for _, r in df.iterrows())

f3 = lambda v, c: f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v)

ru_tbl = '| Rosetta metric | direction | AUROC | 95% CI | AUPRC | lift |\n|---|---|---|---|---|---|\n' + \
    '\n'.join(f"| {RP.get(r.metric, r.metric)} | {r.direction} | {r.AUROC:.3f} | "
              f"{r.ci_lo:.3f}–{r.ci_hi:.3f} | {r.AUPRC:.3f} | {r.lift:.2f}× |"
              for _, r in RU.iterrows())

hh_tbl = '| concept | mine | Rosetta | AUROC mine | AUROC Rosetta | agreement ρ |\n|---|---|---|---|---|---|\n' + \
    '\n'.join(f"| {r.concept} | `{r.mine}` | `{r.rosetta}` | **{r.AUROC_mine:.3f}** | "
              f"{r.AUROC_rosetta:.3f} | {r.agreement_rho:.3f} |"
              for _, r in HH.sort_values('AUROC_mine', ascending=False).iterrows())

sec = f"""## 7. Rosetta InterfaceAnalyzer comparison

PyRosetta 2026.39, `InterfaceAnalyzerMover` over the same 1,029 complexes, with sidechains
repacked in both the complex (`pack_input`) and the separated state (`pack_separated`).

> **Repacking is not optional here.** On the raw predicted models `dG_separated` came back at
> **+319** for the ephrin-B2 control — pure `fa_rep` clash, physically meaningless. Repacking
> drops it to **+25** for ~1 extra second per structure. Boltz-2 models carry enough steric
> strain that unrepacked Rosetta energies are unusable.

### Which Rosetta metrics track binding

![rosetta metrics](figures/08_rosetta_metric_auroc.png)

{ru_tbl}

**`sc_value` is Rosetta's best metric here (AUROC {RU.iloc[0].AUROC:.3f})** — the only one that
clearly beats my equivalent. `crossterm_interface_energy` has the best AUPRC ({RU.AUPRC.max():.3f},
{RU.lift.max():.2f}× lift) despite a middling AUROC, meaning it is sharp at the top of the ranking.

**The two metrics you nominated are the two that fail in Rosetta's implementation:**

- `delta_unsat_hbonds` — **AUROC 0.503**. No signal whatsoever.
- `packstat` — **AUROC 0.509**. No signal whatsoever.

My versions of both concepts do work ({HH[HH.concept.str.startswith('buried unsatisfied polars')].AUROC_mine.max():.3f}
and {float(HH[HH.concept=='packing / voids'].AUROC_mine.iloc[0]):.3f}). The difference is
normalisation and definition, not the idea — see below.

### Same concept, two implementations

![head to head](figures/12_head_to_head.png)
![correlation scatters](figures/09_correlation_scatters.png)

{hh_tbl}

Three distinct outcomes:

1. **Validation.** Buried surface area agrees at **ρ = 0.998**, apolar burial at 0.989, interface
   residue count at 0.961, with identical AUROC. My from-scratch Shrake-Rupley SASA reproduces
   Rosetta's to within rounding — the geometric core of this work is independently confirmed.

2. **Rosetta wins on shape complementarity.** Real Lawrence-Colman `sc_value` scores 0.678 against
   my proxy's 0.585, and they agree only at ρ = 0.494. My implementation measures complementarity
   on the van der Waals surface rather than the solvent-excluded surface, which is why my values
   sit near 0.26 where Rosetta reports 0.66. **Use Rosetta's `sc_value`, not mine.**

3. **Mine wins on energy and packing, and the two disagree about what they measure.**
   - `ddg_est` vs `dG_separated`: **ρ = -0.163** — anti-correlated, and mine discriminates better
     (0.666 vs 0.614). Rosetta's dG retains residual clash even after repacking; my softened LJ
     plus Eisenberg-McLachlan desolvation is more forgiving of imperfect geometry.
   - `void_loose` vs `packstat`: **ρ = -0.055** — no relationship at all. These are simply
     different quantities. Mine reaches 0.645, Rosetta's 0.509.
   - Normalising unsatisfied polars by interface area lifts AUROC from 0.531 to 0.611 in my
     implementation and from 0.503 to 0.594 in Rosetta's. **The raw count is the wrong variable
     in both packages; divide by buried area.**

### Full models

![method comparison](figures/10_method_comparison.png)
![all methods](figures/11_all_methods_roc_pr.png)

L1 logistic, nested family-grouped CV:

| feature set | AUROC | AUPRC | P@10 | P@20 | P@50 |
|---|---|---|---|---|---|
| Rosetta only ({S['n_ros']} metrics) | {S['auroc_ros']:.3f} | {S['auprc_ros']:.3f} | 0.30 | 0.30 | 0.24 |
| **Mine only ({S['n_mine']} descriptors)** | **{S['auroc_mine']:.3f}** | **{S['auprc_mine']:.3f}** | **0.80** | **0.75** | **0.46** |
| Both combined | {S['auroc_both']:.3f} | {S['auprc_both']:.3f} | 0.40 | 0.35 | 0.34 |
| random | 0.500 | {base:.3f} | — | — | — |

The hand-built set beats Rosetta's on every measure, and **combining them is worse than mine
alone**. That is not a claim that the Rosetta features are noise — it is small-sample instability:
101 positives against 84 candidate features means L1 selection varies between folds, and the inner
CV picks a penalty that does not transfer. With this many labels, a smaller feature set is the
safer choice.

### What to actually use

Take `sc_value` from Rosetta, keep everything else from the hand-built set, and normalise any
unsatisfied-polar count by interface area. Rosetta adds a real shape-complementarity term that I
cannot match; it does not add usable packing or energy terms on predicted structures.

---

"""

md = open('proteinbase_data/FINDINGS.md').read()
anchor = '## 7. How to use this'
if '## 7. Rosetta InterfaceAnalyzer comparison' in md:
    raise SystemExit('section already present')
md = md.replace(anchor, sec + '## 8. How to use this')
md = md.replace('4. **Don\'t filter on void volume alone**',
                "4. **Take `sc_value` from Rosetta** — it is the one interface term Rosetta "
                "computes better than anything here (0.678 vs 0.585).\n"
                "5. **Normalise unsatisfied polars by buried area.** The raw count carries no "
                "signal in either implementation (Rosetta's is AUROC 0.503); divided by area both work.\n"
                "6. **Don't filter on `packstat`** — AUROC 0.509 on this data.\n"
                "7. **Don't filter on void volume alone**")
open('proteinbase_data/FINDINGS.md', 'w').write(md)
print('Rosetta section inserted;', len(md), 'chars')
