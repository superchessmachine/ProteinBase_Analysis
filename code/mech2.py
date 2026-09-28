"""Comprehensive mechanistic interface descriptors for designed binders.

All terms are geometry / classical-physics. No neural-network outputs, no
learned weights. Literature sources noted per block.
"""
import numpy as np
from scipy.spatial import cKDTree
from mech import (VDW, PROBE, AA3, DONORS, ACCEPTORS, POS, NEG, KD,
                  parse_cif, sasa, shape_comp, void_volume, _ident)

# Eisenberg & McLachlan (1986) atomic solvation parameters, kcal/mol/A^2
ASP = {'C': 0.0163, 'N': -0.00637, 'O': -0.00637, 'S': 0.0214}
# Rosetta-like LJ well depths (kcal/mol) and radii
EPS = {'C': 0.12, 'N': 0.16, 'O': 0.21, 'S': 0.20}
AROM = {'PHE': ['CG','CD1','CD2','CE1','CE2','CZ'],
        'TYR': ['CG','CD1','CD2','CE1','CE2','CZ'],
        'TRP': ['CD2','CE2','CE3','CZ2','CZ3','CH2']}
CATION = {('LYS','NZ'), ('ARG','CZ')}
HYDROPHOBIC = set('AVILMFWCY')


def _ss_from_backbone(xyz, an, ri, ch, cid):
    """Assign helix/sheet per residue from phi/psi computed off N-CA-C."""
    m = ch == cid
    res = {}
    for a, r, x in zip(an[m], ri[m], xyz[m]):
        if a in ('N', 'CA', 'C'):
            res.setdefault(r, {})[a] = x
    keys = sorted(k for k, v in res.items() if len(v) == 3)
    def dih(p0, p1, p2, p3):
        b0, b1, b2 = p0-p1, p2-p1, p3-p2
        b1n = b1/np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1n)*b1n
        w = b2 - np.dot(b2, b1n)*b1n
        return np.degrees(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))
    h = e = n = 0
    for i in range(1, len(keys)-1):
        a, b, c = keys[i-1], keys[i], keys[i+1]
        if c - a != 2: continue
        try:
            phi = dih(res[a]['C'], res[b]['N'], res[b]['CA'], res[b]['C'])
            psi = dih(res[b]['N'], res[b]['CA'], res[b]['C'], res[c]['N'])
        except Exception:
            continue
        n += 1
        if -100 < phi < -30 and -80 < psi < -5: h += 1
        elif -180 < phi < -80 and (90 < psi <= 180 or -180 <= psi < -150): e += 1
    if n == 0: return np.nan, np.nan, 0
    return h/n, e/n, n


def analyse2(path, binder_seq=None):
    el, an, rn, ri, ch, xyz, bf = parse_cif(path)
    if len(xyz) == 0: return None
    radii = np.array([VDW.get(e, 1.7) for e in el])
    chains = list(dict.fromkeys(ch))
    if len(chains) < 2: return None

    def cseq(c):
        m = ch == c
        _, first = np.unique(ri[m], return_index=True)
        return ''.join(AA3.get(r, 'X') for r in rn[m][np.sort(first)])
    seqs = {c: cseq(c) for c in chains}
    bc = (max(chains, key=lambda c: _ident(seqs[c], binder_seq.replace(' ', '')))
          if binder_seq else min(chains, key=lambda c: len(seqs[c])))
    mA = ch == bc; mB = ~mA
    if mA.sum() < 10 or mB.sum() < 10: return None

    s_cplx = sasa(xyz, radii)
    sA, pA, nA, _ = sasa(xyz[mA], radii[mA], ret_pts=True)
    sB, pB, nB, _ = sasa(xyz[mB], radii[mB], ret_pts=True)
    s_free = np.zeros(len(xyz)); s_free[mA] = sA; s_free[mB] = sB
    d_sasa = s_free - s_cplx
    bsa = float(d_sasa.sum())
    if bsa <= 0: return None
    D = {}

    # ---- burial / hydrophobic effect --------------------------------------
    apolar = np.isin(el, ['C', 'S'])
    D['bsa'] = bsa
    D['bsa_apolar'] = float(d_sasa[apolar].sum())
    D['bsa_polar'] = float(d_sasa[~apolar].sum())
    D['f_apolar'] = D['bsa_apolar']/bsa
    ifc = d_sasa > 0.1
    D['ifc_res'] = len(set(zip(ch[ifc], ri[ifc])))
    D['ifc_res_A'] = len(set(ri[ifc & mA]))
    D['bsa_per_res'] = bsa/max(D['ifc_res_A'], 1)

    # Eisenberg-McLachlan solvation free energy of burial
    asp = np.array([ASP.get(e, 0.0) for e in el])
    D['dG_solv'] = float(-(asp*d_sasa).sum())
    D['dG_solv_per_bsa'] = D['dG_solv']/bsa*1000

    # ---- anchor residues ---------------------------------------------------
    rk = np.array([f'{c}_{i}' for c, i in zip(ch, ri)])
    agg = {}
    for k, v in zip(rk, d_sasa): agg[k] = agg.get(k, 0.0)+v
    _bl = [v for k, v in agg.items() if k.startswith(bc+'_')]
    bv = np.array(sorted(_bl)[::-1]) if _bl else np.array([0.0])
    D['anchor_bsa'] = float(bv[0]); D['top3_bsa'] = float(bv[:3].sum())
    D['n_anchor50'] = int((bv > 50).sum()); D['n_anchor80'] = int((bv > 80).sum())
    D['anchor_frac'] = D['anchor_bsa']/bsa

    # ---- cross-interface pair terms ---------------------------------------
    tree = cKDTree(xyz)
    pr = tree.query_pairs(8.0, output_type='ndarray')
    cross = mA[pr[:, 0]] != mA[pr[:, 1]]
    pr = pr[cross]
    d = np.linalg.norm(xyz[pr[:, 0]]-xyz[pr[:, 1]], axis=1)

    # softened Lennard-Jones
    r0 = radii[pr[:, 0]]+radii[pr[:, 1]]
    ep = np.sqrt(np.array([EPS.get(e, .12) for e in el[pr[:, 0]]]) *
                 np.array([EPS.get(e, .12) for e in el[pr[:, 1]]]))
    x = np.clip(r0/np.maximum(d, 0.8*r0), 0, 2.0)
    D['lj_atr'] = float((-2*ep*x**6)[d >= r0*0.95].sum())
    D['lj_rep'] = float((ep*x**12)[d < r0*0.95].sum())
    D['n_clash'] = int((d < r0*0.75).sum())

    # electrostatics, eps = 4r
    Q = {('ASP','OD1'):-.5,('ASP','OD2'):-.5,('GLU','OE1'):-.5,('GLU','OE2'):-.5,
         ('LYS','NZ'):1.,('ARG','NH1'):.5,('ARG','NH2'):.5,('HIS','ND1'):.1,('HIS','NE2'):.1}
    qv = np.array([Q.get((r, a), 0.) for r, a in zip(rn, an)])
    qq = qv[pr[:, 0]]*qv[pr[:, 1]]
    e = 332.0*qq/(4.0*d*d)
    D['elec'] = float(e.sum()); D['elec_att'] = float(e[e < 0].sum()); D['elec_rep'] = float(e[e > 0].sum())
    D['elec_per_bsa'] = D['elec']/bsa*1000

    # H-bonds (distance + crude angular weight) and salt bridges
    polar = np.isin(el, ['N', 'O'])
    isd = np.array([(r, a) in DONORS or a == 'N' for r, a in zip(rn, an)])
    isa = np.array([(r, a) in ACCEPTORS or a == 'O' for r, a in zip(rn, an)])
    hbm = polar[pr[:, 0]] & polar[pr[:, 1]] & (d < 3.5) & \
          ((isd[pr[:, 0]] & isa[pr[:, 1]]) | (isa[pr[:, 0]] & isd[pr[:, 1]]))
    D['n_hb'] = int(hbm.sum())
    D['hb_energy'] = float(-np.exp(-((d[hbm]-2.9)**2)/0.35).sum()) if hbm.any() else 0.0
    D['hb_density'] = D['n_hb']/bsa*1000
    key = lambda i: (rn[i], an[i])
    sb = [(a, b) for a, b in pr[(d < 4.0)]
          if (key(a) in POS and key(b) in NEG) or (key(a) in NEG and key(b) in POS)]
    D['n_salt'] = len(sb)
    cc = np.isin(el, ['C'])
    D['n_apolar_ct'] = int((cc[pr[:, 0]] & cc[pr[:, 1]] & (d < 5.0)).sum())
    D['ct_density'] = D['n_apolar_ct']/bsa*1000

    # ---- buried unsatisfied polars (Rosetta buns) -------------------------
    hbcap = (isd | isa) & polar
    allp = tree.query_pairs(3.5, output_type='ndarray')
    sat = np.zeros(len(xyz), bool)
    if len(allp):
        sr = (ri[allp[:, 0]] == ri[allp[:, 1]]) & (ch[allp[:, 0]] == ch[allp[:, 1]])
        ok = polar[allp[:, 0]] & polar[allp[:, 1]] & ~sr
        for a, b in allp[ok]:
            if (isd[a] and isa[b]) or (isa[a] and isd[b]): sat[a] = sat[b] = True
    for cut, tag in ((1.0, ''), (5.0, '_loose')):
        nb = hbcap & (d_sasa > 0.1) & (s_cplx < cut)
        D['buns'+tag] = int((nb & ~sat).sum())
        D['buns'+tag+'_A'] = int((nb & ~sat & mA).sum())
        D['buns'+tag+'_density'] = D['buns'+tag]/bsa*1000

    # ---- shape complementarity + gap volume (Laskowski) -------------------
    tB, tA = cKDTree(xyz[mB]), cKDTree(xyz[mA])
    sa = tB.query(pA)[0] < 4.0; sb2 = tA.query(pB)[0] < 4.0
    D['sc'] = shape_comp(pA[sa], nA[sa], pB[sb2], nB[sb2])
    v, vl = void_volume(xyz, radii, ifc, mA, mB)
    D['void'] = v; D['void_loose'] = vl
    D['void_per_bsa'] = v/bsa*100 if v == v else np.nan
    D['gap_index'] = v/(bsa/2) if v == v else np.nan     # Laskowski gap index

    # ---- interface geometry (Jones & Thornton) ----------------------------
    ipts = xyz[ifc & mA]
    if len(ipts) >= 6:
        c0 = ipts.mean(0); U, S, Vt = np.linalg.svd(ipts-c0, full_matrices=False)
        D['planarity'] = float(np.sqrt(((ipts-c0) @ Vt[2])**2).mean())
        D['eccentricity'] = float(S[1]/S[0]) if S[0] > 0 else np.nan
    else:
        D['planarity'] = D['eccentricity'] = np.nan
    bres = sorted(set(ri[ifc & mA]))
    D['n_segments'] = 1+sum(1 for i in range(1, len(bres)) if bres[i]-bres[i-1] > 1) if bres else 0
    D['seg_per_res'] = D['n_segments']/max(len(bres), 1)

    # ---- cation-pi / pi-stacking ------------------------------------------
    cent, cres = [], []
    for c in chains:
        mm = ch == c
        for r in set(ri[mm]):
            sel = mm & (ri == r)
            nm = rn[sel][0]
            if nm in AROM:
                at = [x for a, x in zip(an[sel], xyz[sel]) if a in AROM[nm]]
                if len(at) >= 5: cent.append(np.mean(at, 0)); cres.append(c)
    npi = ncp = 0
    if cent:
        cent = np.array(cent); cres = np.array(cres)
        cat = np.array([i for i in range(len(xyz)) if (rn[i], an[i]) in CATION])
        for i in range(len(cent)):
            for j in range(i+1, len(cent)):
                if cres[i] != cres[j] and np.linalg.norm(cent[i]-cent[j]) < 6.0: npi += 1
        if len(cat):
            for i in range(len(cent)):
                same = mA if cres[i] == bc else mB
                for k in cat:
                    if same[k]: continue
                    if np.linalg.norm(cent[i]-xyz[k]) < 6.5: ncp += 1
    D['n_pi_stack'] = npi; D['n_cation_pi'] = ncp

    # ---- binder fold / developability -------------------------------------
    seqA = seqs[bc]; n = len(seqA)
    D['binder_len'] = n
    ca = xyz[mA & (an == 'CA')]
    if len(ca) > 3:
        D['rg'] = float(np.sqrt(((ca-ca.mean(0))**2).sum(1).mean()))
        D['rg_norm'] = D['rg']/(2.2*max(len(ca), 1)**0.38)
        tca = cKDTree(ca); cp = tca.query_pairs(8.0, output_type='ndarray')
        D['rel_contact_order'] = float(np.abs(cp[:, 0]-cp[:, 1]).mean()/len(ca)) if len(cp) else np.nan
    else:
        D['rg'] = D['rg_norm'] = D['rel_contact_order'] = np.nan
    h, e2, _ = _ss_from_backbone(xyz, an, ri, ch, bc)
    D['frac_helix'] = h; D['frac_sheet'] = e2
    kd = np.array([KD.get(c, 0) for c in seqA])
    D['gravy'] = float(kd.mean())
    w = 5
    D['agg_max'] = float(max(kd[i:i+w].mean() for i in range(max(len(kd)-w+1, 1)))) if len(kd) >= w else np.nan
    D['net_charge'] = sum(seqA.count(x) for x in 'KR')-sum(seqA.count(x) for x in 'DE')
    D['charge_density'] = D['net_charge']/n
    D['frac_aromatic'] = sum(seqA.count(x) for x in 'FWY')/n
    D['frac_charged'] = sum(seqA.count(x) for x in 'DEKR')/n
    D['frac_hydrophobic'] = sum(1 for c in seqA if c in HYDROPHOBIC)/n
    D['n_cys'] = seqA.count('C')
    D['aliphatic_index'] = (seqA.count('A')+2.9*seqA.count('V')+3.9*(seqA.count('I')+seqA.count('L')))*100/n
    # exposed hydrophobic surface of the free binder (aggregation risk)
    hyd = np.isin(el, ['C', 'S']) & mA
    D['surf_hydrophobic'] = float(s_free[hyd].sum())
    D['surf_hydrophobic_frac'] = D['surf_hydrophobic']/max(s_free[mA].sum(), 1)

    # normalised energetics
    D['lj_atr_per_bsa'] = D['lj_atr']/bsa*1000
    D['hb_per_bsa'] = D['hb_energy']/bsa*1000
    D['salt_per_bsa'] = D['n_salt']/bsa*1000
    D['ddg_est'] = D['lj_atr']+D['lj_rep']+D['dG_solv']+D['hb_energy']*1.5+D['elec']
    D['ddg_per_bsa'] = D['ddg_est']/bsa*1000
    D['binder_chain'] = bc
    return D
