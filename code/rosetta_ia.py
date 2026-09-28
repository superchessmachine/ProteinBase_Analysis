"""Rosetta InterfaceAnalyzer over the ProteinBase Nipah complexes.

Produces the canonical Rosetta interface metrics (dG_separated, dSASA,
delta_unsat_hbonds, packstat, sc_value) for head-to-head comparison against the
hand-built mechanistic descriptors in mech2.py.

Sidechains are repacked in the complex (pack_input) and in the separated state
(pack_separated), which is standard practice: on raw predicted models the
unrepacked dG is dominated by fa_rep clash (~+320 vs ~+25 repacked).
"""
import os, sys, traceback, warnings
warnings.filterwarnings('ignore')
SCR = '/data/yash/tmp/claude-1000/-data-yash-comp/dcc873cb-83b0-4a0a-b1ac-c9a359584756/scratchpad'
os.chdir(SCR); sys.path.insert(0, SCR)
import numpy as np, pandas as pd
from mech import parse_cif, AA3

PDBDIR = os.path.join(SCR, 'pdb'); os.makedirs(PDBDIR, exist_ok=True)


def cif_to_pdb(cif, out, binder_seq=None):
    """Write a 2-chain PDB with binder = chain A, target = chain B."""
    el, an, rn, ri, ch, xyz, bf = parse_cif(cif)
    if len(xyz) == 0: return None
    chains = list(dict.fromkeys(ch))
    if len(chains) < 2: return None

    def cseq(c):
        m = ch == c
        _, first = np.unique(ri[m], return_index=True)
        return ''.join(AA3.get(r, 'X') for r in rn[m][np.sort(first)])
    seqs = {c: cseq(c) for c in chains}
    if binder_seq:
        bs = str(binder_seq).replace(' ', '')
        def ident(a, b):
            n = min(len(a), len(b))
            return sum(1 for i in range(n) if a[i] == b[i]) / max(len(a), len(b)) if n else 0
        bc = max(chains, key=lambda c: ident(seqs[c], bs))
    else:
        bc = min(chains, key=lambda c: len(seqs[c]))
    newch = {c: ('A' if c == bc else 'B') for c in chains}
    order = sorted(range(len(xyz)), key=lambda i: (newch[ch[i]] == 'B', ri[i], i))
    lines = []
    for serial, i in enumerate(order, 1):
        nm = an[i]
        nm4 = f' {nm:<3s}' if len(nm) < 4 else nm
        lines.append(
            f"ATOM  {serial:5d} {nm4:4s} {rn[i]:>3s} {newch[ch[i]]}{ri[i]:4d}    "
            f"{xyz[i][0]:8.3f}{xyz[i][1]:8.3f}{xyz[i][2]:8.3f}{1.0:6.2f}{bf[i]:6.2f}"
            f"          {el[i]:>2s}")
    lines += ['TER', 'END']
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return out


_SF = None
def _init():
    """Initialise PyRosetta once per worker process."""
    global _SF
    if _SF is not None: return _SF
    import pyrosetta
    pyrosetta.init(
        '-mute all -ignore_unrecognized_res -ignore_zero_occupancy false '
        '-load_PDB_components false -detect_disulf true', silent=True)
    _SF = pyrosetta.get_fa_scorefxn()
    return _SF


def _g(d, name, idx=None):
    """Safely pull a field off the InterfaceData struct."""
    try:
        v = getattr(d, name)
        if idx is not None: v = v[idx]
        return float(v)
    except Exception:
        return np.nan


def analyse_one(args):
    idx, cid, seq = args
    try:
        sf = _init()
        import pyrosetta
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover as IA
        pdb = os.path.join(PDBDIR, f'{cid}.pdb')
        if not os.path.exists(pdb):
            if cif_to_pdb(f'cif/{cid}.cif', pdb, seq) is None:
                return idx, None
        pose = pyrosetta.pose_from_file(pdb)
        if pose.num_chains() < 2: return idx, None
        ia = IA(1)
        ia.set_scorefunction(sf)
        ia.set_compute_packstat(True)
        ia.set_pack_input(True)
        ia.set_pack_separated(True)
        ia.set_compute_interface_sc(True)
        ia.set_calc_dSASA(True)
        ia.apply(pose)
        d = ia.get_all_data()
        out = dict(
            ros_dG               = _g(d, 'dG', 1),
            ros_dG_dSASA         = _g(d, 'dG_dSASA_ratio'),
            ros_dSASA            = _g(d, 'dSASA', 1),
            ros_dhSASA           = _g(d, 'dhSASA', 1),
            ros_sc               = _g(d, 'sc_value'),
            ros_packstat         = _g(d, 'packstat'),
            ros_unsat            = _g(d, 'delta_unsat_hbonds'),
            ros_hb_E             = _g(d, 'total_hb_E'),
            ros_hb_E_frac        = _g(d, 'hbond_E_fraction'),
            ros_hbonds           = _g(d, 'interface_hbonds'),
            ros_nres             = _g(d, 'interface_nres', 1),
            ros_crossterm        = _g(d, 'crossterm_interface_energy'),
            ros_crossterm_dSASA  = _g(d, 'crossterm_interface_energy_dSASA_ratio'),
            ros_complex_E        = _g(d, 'complex_total_energy', 1),
            ros_separated_E      = _g(d, 'separated_total_energy'),
            ros_complexed_SASA   = _g(d, 'complexed_SASA'),
            ros_separated_SASA   = _g(d, 'separated_SASA'),
            ros_int_surf_frac    = _g(d, 'interface_to_surface_fraction'),
            ros_aromatic_dG_frac = _g(d, 'aromatic_dG_fraction'),
            ros_aromatic_dSASA   = _g(d, 'aromatic_dSASA_fraction'),
            ros_centroid_dG      = _g(d, 'centroid_dG'),
            ros_gly_dG           = _g(d, 'gly_dG'),
            ros_ss_helix         = _g(d, 'ss_helix_nres'),
            ros_ss_sheet         = _g(d, 'ss_sheet_nres'),
            ros_ss_loop          = _g(d, 'ss_loop_nres'),
            ros_sep_int_score    = _g(d, 'separated_interface_score'),
            ros_cplx_int_score   = _g(d, 'complexed_interface_score'),
        )
        # normalised forms
        if out['ros_dSASA'] and out['ros_dSASA'] == out['ros_dSASA'] and out['ros_dSASA'] > 0:
            out['ros_unsat_per_dSASA'] = out['ros_unsat'] / out['ros_dSASA'] * 1000
            out['ros_hb_per_dSASA'] = out['ros_hb_E'] / out['ros_dSASA'] * 1000
        return idx, out
    except Exception:
        return idx, {'err': traceback.format_exc()[-300:]}


if __name__ == '__main__':
    from multiprocessing import Pool
    D = pd.read_csv('labels_fam.csv')
    jobs = [(i, r.id, r.seq) for i, r in D.iterrows()]
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    out = {}
    with Pool(nproc) as p:
        for n, (i, r) in enumerate(p.imap_unordered(analyse_one, jobs, chunksize=2), 1):
            out[i] = r
            if n % 50 == 0: print(f'{n}/{len(jobs)}', flush=True)
    rec = []
    for i in range(len(D)):
        rr = {'idx': i}; v = out.get(i)
        if isinstance(v, dict) and 'err' not in v: rr.update(v)
        rec.append(rr)
    M = pd.DataFrame(rec).set_index('idx')
    R = pd.concat([D[['id', 'binding', 'binding_strength', 'kd', 'family', 'design_class']], M], axis=1)
    R.to_csv('rosetta_results.csv', index=False)
    errs = [v for v in out.values() if v is None or (isinstance(v, dict) and 'err' in v)]
    print('done. rows:', len(R), 'failed:', len(errs))
    if errs and isinstance(errs[0], dict):
        print('first error:', str(errs[0].get('err', ''))[:300])
