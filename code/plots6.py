"""Standalone panel for the best composite score, styled like the fig 09_1 array."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score

INK = '#000000'
plt.rcParams.update({
    'figure.dpi': 600, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Nimbus Sans', 'Helvetica', 'Arial', 'Liberation Sans',
                        'FreeSans', 'DejaVu Sans'],
    'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .5, 'axes.grid': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
    'xtick.major.width': .5, 'ytick.major.width': .5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.labelsize': 7, 'legend.frameon': False,
})
OUT = 'proteinbase_data/figures'
HIT = '#0072B2'; NO = '#D55E00'          # Okabe-Ito

B = pd.read_csv('best_composite_scores.csv')
S = json.load(open('best_composite.json'))
B['kd'] = pd.to_numeric(B.kd, errors='coerce')
y = B.binding.astype(str).str.lower().eq('true').astype(int).values

pkd = np.full(len(B), np.nan)
pkd[y == 0] = 0.0
mb = (y == 1) & B.kd.notna().values & (B.kd.values > 0)
pkd[mb] = -np.log10(B.kd.values[mb])
B['pkd'] = pkd
keep = B.pkd.notna().values
xv = B.best_score.values[keep]; yv = B.pkd.values[keep]; yy = y[keep]
PKD_MIN = float(np.nanmin(pkd[mb]))
r = pearsonr(xv, yv)[0]
base = y.mean()

# PNAS 1.5-column width = 27 picas / 4.5 in / 11.4 cm
fig, ax = plt.subplots(figsize=(4.5, 3.3))
lo, hi = np.nanpercentile(xv, [0.5, 99.5])
DENS_TOP = PKD_MIN * 0.80

cnt, edges = np.histogram(xv[yy == 0], bins=34, range=(lo, hi))
ctr = (edges[:-1] + edges[1:]) / 2
d = cnt / cnt.max() * DENS_TOP
ax.fill_between(ctr, 0, d, color=NO, alpha=.18, lw=0, zorder=1)
ax.plot(ctr, d, color=NO, lw=.7, alpha=.9, zorder=2)

rng = np.random.default_rng(0)
ax.scatter(xv[yy == 0], rng.normal(0, .055, int((yy == 0).sum())),
           s=2.6, color=NO, alpha=.32, edgecolor='none', zorder=3, rasterized=True)
ax.scatter(xv[yy == 1], yv[yy == 1], s=9, color=HIT, alpha=.85,
           edgecolor='white', linewidth=.2, zorder=4)
ax.axhline(0, color=INK, lw=.4, zorder=2)

ax.set_xlim(lo, hi); ax.set_ylim(-.7, 10.3)
ax.set_yticks([0, 2, 4, 6, 8, 10])
ax.set_xlabel('Composite physics score')
ax.set_ylabel('p$K_\\mathrm{D}$   (0 = non-binder)')
ax.text(0, 1.14, f"AUPRC {S['auprc']:.3f}    AUROC {S['auroc']:.3f}    $r$ {r:+.2f}",
        transform=ax.transAxes, fontsize=7.2, va='top', ha='left', color=INK)
ax.text(0, 1.055,
        f"{S['tag']}  ·  nested family-grouped CV  ·  random AUPRC {base:.3f}",
        transform=ax.transAxes, fontsize=6.2, va='top', ha='left', color='#444444')

h = [plt.Line2D([], [], marker='o', ls='', color=HIT, ms=3.4,
                markeredgecolor='white', markeredgewidth=.2),
     plt.Line2D([], [], marker='o', ls='', color=NO, ms=3.4, alpha=.7),
     plt.Rectangle((0, 0), 1, 1, color=NO, alpha=.18)]
ax.legend(h, [f'Binder (n = {int(mb.sum())})',
              f'Non-binder (n = {int((yy==0).sum())})',
              'Non-binder density'],
          fontsize=6.2, loc='center right', bbox_to_anchor=(1.0, .62),
          handletextpad=.5, labelspacing=.45)
fig.tight_layout()
fig.savefig(f'{OUT}/13_best_composite.png')
plt.close(fig)
print(f"AUPRC {S['auprc']:.3f}  AUROC {S['auroc']:.3f}  r {r:+.3f}")
print(f"P@10 {S['pk']['10']:.2f}  P@20 {S['pk']['20']:.2f}  P@50 {S['pk']['50']:.2f}"
      if isinstance(list(S['pk'])[0], str) else
      f"P@10 {S['pk'][10]:.2f}")
print('wrote figures/13_best_composite.png')
