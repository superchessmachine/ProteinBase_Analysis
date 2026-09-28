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

INK = '#1A1A1A'; RULE = '#4A4A4A'
plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': RULE,
    'xtick.color': RULE, 'ytick.color': RULE,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'axes.grid': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.6, 'ytick.major.size': 2.6,
    'xtick.major.width': .7, 'ytick.major.width': .7,
    'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8,
    'axes.labelsize': 7.8, 'legend.frameon': False,
})
OUT = 'proteinbase_data/figures'
HIT = '#2A7F7A'; NO = '#D96A4A'

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
 'buns_loose_density': 'Buried unsatisfied polars / kÅ²',
 'buns_loose': 'Buried unsatisfied polars',
 'sc': 'Shape complementarity',
 'bsa': 'Buried surface area (Å²)', 'bsa_apolar': 'Apolar buried area (Å²)',
 'void_loose': 'Interface void (Å³)', 'ddg_est': 'Estimated ΔΔG',
 'ddg_per_bsa': 'ΔΔG / buried area', 'hb_energy': 'Hydrogen-bond energy',
 'ifc_res': 'Interface residues',
 'ros_unsat_per_dSASA': 'Buried unsatisfied polars / kÅ²',
 'ros_unsat': 'Buried unsatisfied polars',
 'ros_sc': 'Shape complementarity', 'ros_dSASA': 'Buried surface area (Å²)',
 'ros_dhSASA': 'Apolar buried area (Å²)', 'ros_packstat': 'Packstat',
 'ros_dG': 'ΔG separated', 'ros_dG_dSASA': 'ΔG / buried area',
 'ros_hb_E': 'Hydrogen-bond energy', 'ros_nres': 'Interface residues'}
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
fig, axes = plt.subplots(nrow, ncol, figsize=(2.55 * ncol, 2.30 * nrow))
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
        ax.fill_between(ctr, 0, d, color=NO, alpha=.16, lw=0, zorder=1)
        ax.plot(ctr, d, color=NO, lw=.8, alpha=.85, zorder=2)

    ax.scatter(xv[yy == 0], rng.normal(0, .042, int((yy == 0).sum())),
               s=3.2, color=NO, alpha=.30, edgecolor='none', zorder=3, rasterized=True)
    ax.scatter(xv[yy == 1], yv[yy == 1], s=11, color=HIT, alpha=.88,
               edgecolor='white', linewidth=.25, zorder=4)
    ax.axhline(0, color=RULE, lw=.6, zorder=2)

    ax.set_xlim(lo, hi); ax.set_ylim(-.6, 10.2)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel(lab(r.metric, r.source))
    if k % ncol == 0:
        ax.set_ylabel('p$K_\\mathrm{D}$   (0 = non-binder)')
    else:
        ax.set_yticklabels([])
    ax.text(0, 1.11, f'{k+1}', transform=ax.transAxes, fontsize=8,
            weight='bold', va='top', ha='left', color=INK)
    ax.text(.10, 1.11,
            f'AUPRC {r.AUPRC:.3f}    AUROC {r.AUROC:.3f}    $r$ {r.pearson_all:+.2f}',
            transform=ax.transAxes, fontsize=6.9, va='top', ha='left', color='#3C3C3C')
for ax in np.ravel(axes)[len(P):]: ax.axis('off')

h = [plt.Line2D([], [], marker='o', ls='', color=HIT, ms=4.6,
                markeredgecolor='white', markeredgewidth=.3),
     plt.Line2D([], [], marker='o', ls='', color=NO, ms=4.6, alpha=.7),
     plt.Rectangle((0, 0), 1, 1, color=NO, alpha=.16)]
H = fig.get_figheight()
fig.suptitle('Evaluation Metrics', x=.5, ha='center', y=1 - .30 / H, va='top',
             fontsize=13.5, weight='bold', color=INK)
fig.legend(h, [f'Binder (n = {int(mb.sum())})',
               f'Non-binder (n = {int((y==0).sum())})',
               'Non-binder density along x'],
           fontsize=7.6, ncol=3, loc='upper center',
           bbox_to_anchor=(.5, 1 - .60 / H), handletextpad=.5, columnspacing=2.2)
fig.tight_layout(rect=[0, 0, 1, 1 - 1.00 / H], h_pad=2.0, w_pad=.9)
fig.savefig(f'{OUT}/09_1_metric_vs_affinity.png')
plt.close(fig)

P.drop(columns=['ok']).to_csv('metric_vs_affinity.csv', index=False)
print("\n=== ordered by AUPRC ===")
print(P.drop(columns=['ok']).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
