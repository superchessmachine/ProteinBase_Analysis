"""Mechanistic interface descriptors from a predicted complex.
Pure geometry/physics: SASA, buried unsatisfied H-bond donors/acceptors,
interface void volume, Lawrence-Colman shape complementarity, salt bridges.
No learned parameters.
"""
import numpy as np
from scipy.spatial import cKDTree

VDW = {'C':1.70,'N':1.55,'O':1.52,'S':1.80,'P':1.80,'H':1.20}
PROBE = 1.4

AA3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
 'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
 'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

# H-bond donor / acceptor heavy atoms by (resname, atomname)
DONORS = {('ARG','NE'),('ARG','NH1'),('ARG','NH2'),('ASN','ND2'),('GLN','NE2'),
 ('HIS','ND1'),('HIS','NE2'),('LYS','NZ'),('SER','OG'),('THR','OG1'),('TRP','NE1'),
 ('TYR','OH'),('CYS','SG')}
ACCEPTORS = {('ASN','OD1'),('ASP','OD1'),('ASP','OD2'),('GLN','OE1'),('GLU','OE1'),
 ('GLU','OE2'),('HIS','ND1'),('HIS','NE2'),('SER','OG'),('THR','OG1'),('TYR','OH'),
 ('MET','SD'),('CYS','SG')}
POS = {('LYS','NZ'),('ARG','NH1'),('ARG','NH2'),('ARG','NE'),('HIS','ND1'),('HIS','NE2')}
NEG = {('ASP','OD1'),('ASP','OD2'),('GLU','OE1'),('GLU','OE2')}
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
 'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}


def parse_cif(path):
    """Parse model-1 ATOM records from a boltz2 mmCIF."""
    el,an,rn,ri,ch,xyz,bf = [],[],[],[],[],[],[]
    with open(path) as f:
        for line in f:
            if not line.startswith('ATOM'): continue
            p = line.split()
            if len(p) < 19 or p[18] != '1': continue
            el.append(p[2]); an.append(p[3]); rn.append(p[5])
            ri.append(int(p[7])); ch.append(p[15])
            xyz.append((float(p[10]), float(p[11]), float(p[12]))); bf.append(float(p[17]))
    return (np.array(el), np.array(an), np.array(rn), np.array(ri),
            np.array(ch), np.array(xyz, float), np.array(bf, float))


def sphere_pts(n=512):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2*i/n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.c_[np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)]

SP = sphere_pts(512)


def sasa(xyz, radii, ret_pts=False):
    """Shrake-Rupley per-atom SASA. Optionally return accessible surface points+normals."""
    ext = radii + PROBE
    tree = cKDTree(xyz)
    out = np.zeros(len(xyz))
    pts_all, nrm_all, own_all = [], [], []
    maxr = ext.max()
    for i in range(len(xyz)):
        nb = np.array(tree.query_ball_point(xyz[i], ext[i] + maxr))
        nb = nb[nb != i]
        p = xyz[i] + SP * ext[i]
        if len(nb):
            d = np.linalg.norm(p[:, None, :] - xyz[nb][None, :, :], axis=2)
            acc = ~(d < ext[nb][None, :]).any(axis=1)
        else:
            acc = np.ones(len(p), bool)
        out[i] = 4*np.pi*ext[i]**2 * acc.mean()
        if ret_pts and acc.any():
            pts_all.append(xyz[i] + SP[acc] * radii[i]); nrm_all.append(SP[acc])
            own_all.append(np.full(acc.sum(), i))
    if ret_pts:
        if pts_all:
            return out, np.vstack(pts_all), np.vstack(nrm_all), np.concatenate(own_all)
        return out, np.zeros((0,3)), np.zeros((0,3)), np.zeros(0,int)
    return out


def shape_comp(pA, nA, pB, nB, w=0.5):
    """Lawrence & Colman Sc on interface surface points."""
    if len(pA) < 10 or len(pB) < 10: return np.nan
    tA, tB = cKDTree(pA), cKDTree(pB)
    dAB, iAB = tB.query(pA); dBA, iBA = tA.query(pB)
    sAB = np.sum(nA * (-nB[iAB]), axis=1) * np.exp(-w * dAB**2)
    sBA = np.sum(nB * (-nA[iBA]), axis=1) * np.exp(-w * dBA**2)
    return float(np.median(np.concatenate([sAB, sBA])))


def void_volume(xyz, radii, ifc_mask, mA, mB, spacing=0.8):
    """Grid-based buried void volume in the interface slab."""
    if ifc_mask.sum() < 3: return np.nan, np.nan
    c = xyz[ifc_mask]
    lo, hi = c.min(0) - 3.0, c.max(0) + 3.0
    gr = [np.arange(lo[k], hi[k], spacing) for k in range(3)]
    if any(len(g) == 0 for g in gr) or np.prod([len(g) for g in gr]) > 4_000_000:
        return np.nan, np.nan
    G = np.stack(np.meshgrid(*gr, indexing='ij'), -1).reshape(-1, 3)
    tree = cKDTree(xyz)
    # empty: no atom sphere (+probe) covers the point
    d, idx = tree.query(G)
    empty = d > (radii[idx] + PROBE)
    G = G[empty]
    if not len(G): return 0.0, 0.0
    # must sit between the two chains
    tA, tB = cKDTree(xyz[mA]), cKDTree(xyz[mB])
    near = (tA.query(G)[0] < 8.0) & (tB.query(G)[0] < 8.0)
    G = G[near]
    if not len(G): return 0.0, 0.0
    # buriedness: fraction of 26 ray directions blocked within 12 A
    dirs = np.array([[i,j,k] for i in(-1,0,1) for j in(-1,0,1) for k in(-1,0,1)
                     if (i,j,k)!=(0,0,0)], float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    steps = np.arange(1.5, 12.0, 1.0)
    blocked = np.zeros(len(G))
    for dvec in dirs:
        ray = G[:, None, :] + dvec[None, None, :] * steps[None, :, None]
        dd, ii = tree.query(ray.reshape(-1, 3))
        hit = (dd < radii[ii]).reshape(len(G), -1).any(axis=1)
        blocked += hit
    buried = blocked >= 24
    vox = spacing**3
    return float(buried.sum()*vox), float((blocked >= 20).sum()*vox)


def analyse(path, binder_seq=None):
    el, an, rn, ri, ch, xyz, bf = parse_cif(path)
    if len(xyz) == 0: return None
    radii = np.array([VDW.get(e, 1.7) for e in el])
    chains = [c for c in dict.fromkeys(ch)]
    if len(chains) < 2: return None

    # chain -> sequence, to identify the binder
    def cseq(c):
        m = ch == c
        _, first = np.unique(ri[m], return_index=True)
        return ''.join(AA3.get(r, 'X') for r in rn[m][np.sort(first)])
    seqs = {c: cseq(c) for c in chains}
    if binder_seq:
        bs = binder_seq.replace(' ', '')
        bc = max(chains, key=lambda c: _ident(seqs[c], bs))
    else:
        bc = min(chains, key=lambda c: len(seqs[c]))
    tc = [c for c in chains if c != bc]
    mA = ch == bc                      # binder
    mB = np.isin(ch, tc)               # target
    if mA.sum() < 10 or mB.sum() < 10: return None

    s_cplx = sasa(xyz, radii)
    sA, pA, nA, oA = sasa(xyz[mA], radii[mA], ret_pts=True)
    sB, pB, nB, oB = sasa(xyz[mB], radii[mB], ret_pts=True)
    s_free = np.zeros(len(xyz)); s_free[mA] = sA; s_free[mB] = sB
    d_sasa = s_free - s_cplx
    bsa = float(d_sasa.sum())                     # total buried (both sides)

    apolar = np.isin(el, ['C', 'S'])
    bsa_apolar = float(d_sasa[apolar].sum())
    f_apolar = bsa_apolar / bsa if bsa > 0 else np.nan

    ifc = d_sasa > 0.1
    ifc_res = len(set(zip(ch[ifc], ri[ifc])))
    ifc_res_A = len(set(ri[ifc & mA]))

    # ---- cross-chain polar contacts -------------------------------------
    polar = np.isin(el, ['N', 'O'])
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(3.5, output_type='ndarray')
    if len(pairs):
        cross = (mA[pairs[:, 0]] != mA[pairs[:, 1]])
        pol2 = polar[pairs[:, 0]] & polar[pairs[:, 1]]
        hb = pairs[cross & pol2]
        n_hb = len(hb)
        key = lambda i: (rn[i], an[i])
        n_salt = sum(1 for a, b in hb
                     if (key(a) in POS and key(b) in NEG) or (key(a) in NEG and key(b) in POS))
        cc = np.isin(el, ['C'])
        n_apolar_ct = int((cross & cc[pairs[:, 0]] & cc[pairs[:, 1]]).sum())
    else:
        n_hb = n_salt = n_apolar_ct = 0

    # ---- buried unsatisfied polar atoms ---------------------------------
    is_don = np.array([(r, a) in DONORS or a == 'N' for r, a in zip(rn, an)])
    is_acc = np.array([(r, a) in ACCEPTORS or a == 'O' for r, a in zip(rn, an)])
    hbcap = (is_don | is_acc) & polar
    # buried on binding and essentially inaccessible in the complex
    newly_buried = hbcap & (d_sasa > 0.1) & (s_cplx < 1.0)
    newly_buried2 = hbcap & (d_sasa > 0.1) & (s_cplx < 5.0)
    part = tree.query_pairs(3.5, output_type='ndarray')
    satisfied = np.zeros(len(xyz), bool)
    if len(part):
        same_res = (ri[part[:, 0]] == ri[part[:, 1]]) & (ch[part[:, 0]] == ch[part[:, 1]])
        ok = polar[part[:, 0]] & polar[part[:, 1]] & ~same_res
        for a, b in part[ok]:
            if (is_don[a] and is_acc[b]) or (is_acc[a] and is_don[b]):
                satisfied[a] = satisfied[b] = True
    buns = int((newly_buried & ~satisfied).sum())
    buns_loose = int((newly_buried2 & ~satisfied).sum())
    buns_A = int((newly_buried & ~satisfied & mA).sum())

    # ---- shape complementarity ------------------------------------------
    tB_all = cKDTree(xyz[mB]); tA_all = cKDTree(xyz[mA])
    selA = tB_all.query(pA)[0] < 4.0
    selB = tA_all.query(pB)[0] < 4.0
    sc = shape_comp(pA[selA], nA[selA], pB[selB], nB[selB])

    # ---- interface voids -------------------------------------------------
    v_tight, v_loose = void_volume(xyz, radii, ifc, mA, mB)

    # ---- anchor residues: deep single-residue burial (hotspot proxy) -----
    res_key = np.array([f"{c}_{i}" for c, i in zip(ch, ri)])
    dfres = {}
    for k, v in zip(res_key, d_sasa):
        dfres[k] = dfres.get(k, 0.0) + v
    binder_res = {k: v for k, v in dfres.items() if k.startswith(bc + '_')}
    bvals = np.array(list(binder_res.values())) if binder_res else np.array([0.0])
    anchor_bsa = float(bvals.max())
    n_anchor50 = int((bvals > 50).sum())
    n_anchor80 = int((bvals > 80).sum())
    top3_bsa = float(np.sort(bvals)[-3:].sum())

    # ---- Coulomb electrostatics across the interface (eps = 4r) ----------
    Q = {('ASP','OD1'):-0.5,('ASP','OD2'):-0.5,('GLU','OE1'):-0.5,('GLU','OE2'):-0.5,
         ('LYS','NZ'):1.0,('ARG','NH1'):0.5,('ARG','NH2'):0.5,('ARG','NE'):0.0,
         ('HIS','ND1'):0.1,('HIS','NE2'):0.1}
    qv = np.array([Q.get((r, a), 0.0) for r, a in zip(rn, an)])
    chg_idx = np.nonzero(qv != 0.0)[0]
    elec = elec_att = elec_rep = 0.0
    if len(chg_idx) > 1:
        tq = cKDTree(xyz[chg_idx])
        pr = tq.query_pairs(12.0, output_type='ndarray')
        if len(pr):
            ga, gb = chg_idx[pr[:, 0]], chg_idx[pr[:, 1]]
            cross = mA[ga] != mA[gb]
            ga, gb = ga[cross], gb[cross]
            if len(ga):
                d = np.linalg.norm(xyz[ga] - xyz[gb], axis=1)
                e = 332.0 * qv[ga] * qv[gb] / (4.0 * d * d)
                elec = float(e.sum()); elec_att = float(e[e < 0].sum()); elec_rep = float(e[e > 0].sum())

    seqA = seqs[bc]
    n_res_A = len(seqA)
    gravy = float(np.mean([KD.get(c, 0) for c in seqA])) if n_res_A else np.nan
    chg = sum(seqA.count(x) for x in 'KR') - sum(seqA.count(x) for x in 'DE')

    return dict(
        binder_chain=bc, n_atoms=len(xyz), binder_len=n_res_A,
        bsa=bsa, bsa_apolar=bsa_apolar, f_apolar=f_apolar,
        bsa_per_res=bsa / ifc_res_A if ifc_res_A else np.nan,
        ifc_res=ifc_res, ifc_res_A=ifc_res_A,
        n_hb=n_hb, n_salt=n_salt, n_apolar_ct=n_apolar_ct,
        hb_density=n_hb / bsa * 1000 if bsa > 0 else np.nan,
        ct_density=n_apolar_ct / bsa * 1000 if bsa > 0 else np.nan,
        buns=buns, buns_A=buns_A, buns_loose=buns_loose,
        buns_density=buns / bsa * 1000 if bsa > 0 else np.nan,
        void=v_tight, void_loose=v_loose,
        void_per_bsa=v_tight / bsa * 100 if (bsa > 0 and v_tight == v_tight) else np.nan,
        sc=sc, gravy=gravy, net_charge=chg,
        anchor_bsa=anchor_bsa, n_anchor50=n_anchor50, n_anchor80=n_anchor80, top3_bsa=top3_bsa,
        elec=elec, elec_att=elec_att, elec_rep=elec_rep,
        elec_per_bsa=elec / bsa * 1000 if bsa > 0 else np.nan,
        iplddt=float(bf[ifc].mean()) if ifc.any() else np.nan,
    )


def _ident(a, b):
    n = min(len(a), len(b))
    if n == 0: return 0.0
    return sum(1 for i in range(n) if a[i] == b[i]) / max(len(a), len(b))
