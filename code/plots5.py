"""Fig 09_1: predicted metric vs experimental affinity, non-binders pinned at 0.

y = pK_D = -log10(K_D); non-binders set to 0. A raw K_D axis cannot take 0 for
non-binders: 0 M would plot below the tightest binder and read as infinite affinity.

Panels ordered by AUPRC, best first. The shaded curve rising from y=0 is the
non-binder density along x, scaled to fill the empty band between 0 and the
weakest binder.
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score

# PNAS Digital Art Guidelines: 2-column width = 42.125 picas / 7 in / 17.8 cm,
# max height 54 picas / 9 in / 22.5 cm, fonts limited to Arial/Helvetica/Times,
# all text >= 6 pt, RGB colour.  Nimbus Sans is the URW Helvetica clone.
INK = '#000000'; RULE = '#000000'
plt.rcParams.update({
    'figure.dpi': 600, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Nimbus Sans', 'Helvetica', 'Arial', 'Liberation Sans',
                        'FreeSans', 'DejaVu Sans'],
    'font.size': 6.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': RULE,
    'xtick.color': RULE, 'ytick.color': RULE,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .5, 'axes.grid': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
    'xtick.major.width': .5, 'ytick.major.width': .5,
    'xtick.labelsize': 6.0, 'ytick.labelsize': 6.0,
    'axes.labelsize': 6.5, 'legend.frameon': False,
})
OUT = 'proteinbase_data/figures'
# Okabe-Ito colourblind-safe palette (Nature Methods 2011 / Okabe & Ito 2008),
# the de-facto standard in PNAS figures.  PNAS itself mandates no palette.
OKABE_BLUE = '#0072B2'; OKABE_VERM = '#D55E00'
HIT = OKABE_BLUE; NO = OKABE_VERM

D = pd.read_csv('compare_scored.csv')
HH = pd.read_csv('head_to_head.csv')
D['kd'] = pd.to_numeric(D.kd, errors='coerce')
y = D.y.values

pkd = np.full(len(D), np.nan)
pkd[y == 0] = 0.0
mb = (y == 1) & D.kd.notna().values & (D.kd.values > 0)
pkd[mb] = -np.log10(D.kd.values[mb])
D['pkd'] = pkd
keep = D.pkd.notna()
PKD_MIN = float(np.nanmin(pkd[mb]))
print(f"plotted={int(keep.sum())}  binders={int(mb.sum())}  "
      f"non-binders={int((y==0).sum())}  weakest binder pK_D={PKD_MIN:.2f}")

PRETTY = {
 'buns_loose_density': 'Buried unsat. polars / kÅ²',
 'buns_loose': 'Buried unsat. polars',
 'sc': 'Shape complementarity',
 'bsa': 'Buried area (Å²)', 'bsa_apolar': 'Apolar buried area (Å²)',
 'void_loose': 'Interface void (Å³)', 'ddg_est': 'Estimated ΔΔG',
 'ddg_per_bsa': 'ΔΔG / buried area', 'hb_energy': 'H-bond energy',
 'ifc_res': 'Interface residues',
 'ros_unsat_per_dSASA': 'Buried unsat. polars / kÅ²',
 'ros_unsat': 'Buried unsat. polars',
 'ros_sc': 'Shape complementarity', 'ros_dSASA': 'Buried area (Å²)',
 'ros_dhSASA': 'Apolar buried area (Å²)', 'ros_packstat': 'Packstat',
 'ros_dG': 'ΔG separated', 'ros_dG_dSASA': 'ΔG / buried area',
 'ros_hb_E': 'H-bond energy', 'ros_nres': 'Interface residues'}
def lab(c, src):
    t = PRETTY.get(c, c.replace('ros_', '').replace('_', ' '))
    return t + (' (Rosetta)' if src == 'Rosetta' else '')

cand = []
for _, r in HH.iterrows():
    cand += [(r.mine, 'Mine'), (r.rosetta, 'Rosetta')]
rows = []
for col, who in cand:
    v = pd.to_numeric(D[col], errors='coerce')
    ok = keep.values & v.notna().values
    if ok.sum() < 100: continue
    xv = v.values[ok]; yy = y[ok]
    a = roc_auc_score(yy, xv); sg = 1 if a >= .5 else -1
    rows.append(dict(metric=col, source=who, ok=ok,
                     AUROC=roc_auc_score(yy, sg * xv),
                     AUPRC=average_precision_score(yy, sg * xv),
                     pearson_all=pearsonr(xv, D.pkd.values[ok])[0]))
P = pd.DataFrame(rows).sort_values('AUPRC', ascending=False).reset_index(drop=True)

ncol = 4; nrow = int(np.ceil(len(P) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 8.6))  # PNAS 2-col x < 9 in
rng = np.random.default_rng(0)
DENS_TOP = PKD_MIN * 0.80

for k, (ax, (_, r)) in enumerate(zip(np.ravel(axes), P.iterrows())):
    v = pd.to_numeric(D[r.metric], errors='coerce')
    ok = r.ok
    xv = v.values[ok]; yv = D.pkd.values[ok]; yy = y[ok]
    lo, hi = np.nanpercentile(xv, [1, 99])
    if not np.isfinite(lo) or hi <= lo: ax.axis('off'); continue

    nb = xv[yy == 0]
    uniq = np.unique(np.round(xv, 6))
    if np.allclose(uniq, np.round(uniq)) and len(uniq) <= 30:
        bins = np.arange(np.floor(lo) - .5, np.ceil(hi) + 1.5, 1.0)
        cnt, edges = np.histogram(nb, bins=bins)
    else:
        cnt, edges = np.histogram(nb, bins=int(min(28, max(8, len(uniq)))), range=(lo, hi))
    ctr = (edges[:-1] + edges[1:]) / 2
    if cnt.max() > 0:
        d = cnt / cnt.max() * DENS_TOP
        ax.fill_between(ctr, 0, d, color=NO, alpha=.18, lw=0, zorder=1)
        ax.plot(ctr, d, color=NO, lw=.6, alpha=.9, zorder=2)

    ax.scatter(xv[yy == 0], rng.normal(0, .045, int((yy == 0).sum())),
               s=1.6, color=NO, alpha=.32, edgecolor='none', zorder=3, rasterized=True)
    ax.scatter(xv[yy == 1], yv[yy == 1], s=5.0, color=HIT, alpha=.85,
               edgecolor='white', linewidth=.15, zorder=4)
    ax.axhline(0, color=RULE, lw=.4, zorder=2)

    ax.set_xlim(lo, hi); ax.set_ylim(-.6, 10.2)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel(lab(r.metric, r.source))
    if k % ncol == 0:
        ax.set_ylabel('p$K_\\mathrm{D}$  (0 = non-binder)')
    else:
        ax.set_yticklabels([])
    ax.text(-.02, 1.20, f'{k+1}', transform=ax.transAxes, fontsize=7.5,
            weight='bold', va='top', ha='left', color=INK)
    ax.text(.10, 1.20,
            f'AUPRC {r.AUPRC:.3f}   AUROC {r.AUROC:.3f}   $r$ {r.pearson_all:+.2f}',
            transform=ax.transAxes, fontsize=6.0, va='top', ha='left', color=INK)
for ax in np.ravel(axes)[len(P):]: ax.axis('off')

h = [plt.Line2D([], [], marker='o', ls='', color=HIT, ms=3.4,
                markeredgecolor='white', markeredgewidth=.2),
     plt.Line2D([], [], marker='o', ls='', color=NO, ms=3.4, alpha=.7),
     plt.Rectangle((0, 0), 1, 1, color=NO, alpha=.18)]
H = fig.get_figheight()
fig.suptitle('Evaluation Metrics', x=.5, ha='center', y=1 - .12 / H, va='top',
             fontsize=9.5, weight='bold', color=INK)
fig.legend(h, [f'Binder (n = {int(mb.sum())})',
               f'Non-binder (n = {int((y==0).sum())})',
               'Non-binder density along x'],
           fontsize=6.5, ncol=3, loc='upper center',
           bbox_to_anchor=(.5, 1 - .34 / H), handletextpad=.4, columnspacing=1.8)
fig.tight_layout(rect=[0, 0, 1, 1 - .62 / H], h_pad=1.9, w_pad=1.0)
fig.savefig(f'{OUT}/09_1_metric_vs_affinity.png')
plt.close(fig)

P.drop(columns=['ok']).to_csv('metric_vs_affinity.csv', index=False)
print("\n=== ordered by AUPRC ===")
print(P.drop(columns=['ok']).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
