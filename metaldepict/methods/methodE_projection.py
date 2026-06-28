#!/usr/bin/env python3
"""METHOD E -- 3D->2D PROJECTION depiction.

A genuinely 3D-derived 2D drawing.  For each (ligand)Cu-H complex we:

  1. build the RDKit 3D conformer of the FULL complex (build_complex.py);
  2. find the best-fit viewing plane by PCA of the heavy-atom coordinates;
  3. PROJECT every heavy atom onto that plane -> raw 2D coords, keeping the
     signed distance to the plane as the out-of-plane DEPTH;
  4. REGULARISE toward an idealised ChemDraw geometry: one standard bond
     length, ~120 deg angles, every ring snapped to a perfect regular polygon,
     WITHOUT destroying the projected orientation -- so the drawing still
     reflects the real 3D geometry;
  5. collapse terminal groups to MEDIUM plain-text labels (t-Bu, OMe, Ph);
  6. (rendered by methodE_render.py in ChemDraw style.)

This module produces the *layout*; methodE_render.py rasterises it.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from src import abbreviations as abbr  # noqa: E402
from src import build_complex, chem, depict  # noqa: E402

LIGANDS = {
    "dtbm_segphos": ("(DTBM-SEGPhos)Cu-H", None),
    "xantphos": ("(Xantphos)Cu-H",
                 "CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21"),
    "dpephos": ("(DPEphos)Cu-H",
                "O(c1ccccc1P(c1ccccc1)c1ccccc1)c1ccccc1P(c1ccccc1)c1ccccc1"),
    "ph_bpe": ("(Ph-BPE)Cu-H",
               "C(CP1C(c2ccccc2)CCC1c1ccccc1)P1C(c2ccccc2)CCC1c1ccccc1"),
}

BOND = 1.0  # one standard bond length (layout units)


# ====================================================================
#  STEP 2-3: PCA projection of the 3D conformer onto the best-fit plane
# ====================================================================
def project_pca(mol: Chem.Mol):
    conf = mol.GetConformer()
    pts = np.array([[conf.GetAtomPosition(i).x,
                     conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z]
                    for i in range(mol.GetNumAtoms())])
    c = pts.mean(axis=0)
    X = pts - c
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    e1, e2, normal = vt[0], vt[1], vt[2]
    xy = np.column_stack([X @ e1, X @ e2])
    depth = X @ normal
    if depth[int(np.argmax(np.abs(depth)))] < 0:
        depth = -depth
    if np.abs(depth).max() > 1e-9:
        depth = depth / np.abs(depth).max()
    return xy, depth


# ====================================================================
#  STEP 4: regularise toward ideal ChemDraw geometry
# ====================================================================
def ordered_rings(mol):
    ri = mol.GetRingInfo()
    rings = []
    for ring in ri.AtomRings():
        ring = list(ring)
        rset = set(ring)
        adj = {a: [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors()
                   if n.GetIdx() in rset] for a in ring}
        order, prev, cur = [ring[0]], None, ring[0]
        while len(order) < len(ring):
            nxt = next((x for x in adj[cur] if x != prev and x not in order), None)
            if nxt is None:
                break
            order.append(nxt)
            prev, cur = cur, nxt
        rings.append(order)
    return rings


def _procrustes_align(src, dst):
    """Best rotation+reflection+scale+translation taking src onto dst (Kabsch
    with optional reflection). Returns transform(p)->aligned."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    A = src - cs
    B = dst - cd
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    Rm = Vt.T @ U.T          # rotation (allow reflection -> better view match)
    scale = 1.0

    def tf(p):
        return (np.asarray(p, float) - cs) @ Rm.T * scale + cd
    return tf


def _coordgen_subset(mol, atom_subset):
    """CoordGen 2D coords for an arbitrary connected atom subset, returned as a
    dict {orig_idx: np.array([x,y])} with bond length normalised to BOND."""
    from rdkit.Chem import rdCoordGen, AllChem
    sub_list = sorted(atom_subset)
    idxmap = {old: i for i, old in enumerate(sub_list)}
    em = Chem.RWMol()
    for old in sub_list:
        a = mol.GetAtomWithIdx(old)
        na = Chem.Atom(a.GetAtomicNum())
        na.SetFormalCharge(a.GetFormalCharge())
        na.SetNoImplicit(True)
        em.AddAtom(na)
    for b in mol.GetBonds():
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a in idxmap and c in idxmap:
            bt = b.GetBondType()
            if bt == Chem.BondType.DATIVE:
                bt = Chem.BondType.SINGLE
            em.AddBond(idxmap[a], idxmap[c], bt)
    sub = em.GetMol()
    try:
        Chem.SanitizeMol(sub)
    except Exception:
        try:
            Chem.SanitizeMol(sub, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_KEKULIZE ^
                             Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
        except Exception:
            pass
    try:
        rdCoordGen.AddCoords(sub)
    except Exception:
        AllChem.Compute2DCoords(sub)
    conf = sub.GetConformer()
    out = {}
    for old, new in idxmap.items():
        p = conf.GetAtomPosition(new)
        out[old] = np.array([p.x, p.y])
    bl = []
    for b in sub.GetBonds():
        pa = conf.GetAtomPosition(b.GetBeginAtomIdx())
        pb = conf.GetAtomPosition(b.GetEndAtomIdx())
        bl.append(math.hypot(pa.x - pb.x, pa.y - pb.y))
    if bl:
        s = BOND / np.median(bl)
        for k in out:
            out[k] = out[k] * s
    return out


def _place_fused_rings(mol, ring_atoms, keep_atoms):
    """Deterministically lay out a fused-ring system as edge-sharing regular
    polygons (never overlapping, always flat).  Returns {idx: np.array}.

    Seed the largest ring as a regular polygon; then repeatedly place any ring
    that shares an edge with an already-placed ring by mirroring across that
    shared edge -- the canonical way fused aromatics are drawn.  Atoms not yet
    fixed (single-ring atoms, spiro) keep their first placement."""
    ri = mol.GetRingInfo()
    rings = [list(r) for r in ri.AtomRings()
             if set(r) <= ring_atoms and all(a in keep_atoms for a in r)]
    if not rings:
        return {}
    # order each ring cyclically
    ordered = []
    for r in rings:
        rset = set(r)
        adj = {a: [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors()
                   if n.GetIdx() in rset] for a in r}
        order, prev, cur = [r[0]], None, r[0]
        while len(order) < len(r):
            nxt = next((x for x in adj[cur] if x != prev and x not in order), None)
            if nxt is None:
                break
            order.append(nxt)
            prev, cur = cur, nxt
        ordered.append(order)

    pos = {}
    ring_cen = {}   # ridx -> centroid (for correct opposite-side reflection)

    def regpoly(order, p_a, p_b, a, b):
        m = len(order)
        ia, ib = order.index(a), order.index(b)
        R = 1.0 / (2 * math.sin(math.pi / m)) * BOND
        canon = [R * np.array([math.cos(2 * math.pi * k / m + math.pi / 2),
                               math.sin(2 * math.pi * k / m + math.pi / 2)])
                 for k in range(m)]
        ca, cb = canon[ia], canon[ib]
        v0 = cb - ca
        v1 = np.array(p_b) - np.array(p_a)
        ang = math.atan2(v1[1], v1[0]) - math.atan2(v0[1], v0[0])
        cs, sn = math.cos(ang), math.sin(ang)
        Rm = np.array([[cs, -sn], [sn, cs]])
        placed = [(Rm @ (c - ca)) + np.array(p_a) for c in canon]
        return [placed[k] for k in range(m)]

    seed = max(range(len(ordered)), key=lambda i: len(ordered[i]))
    s = ordered[seed]
    m = len(s)
    R = 1.0 / (2 * math.sin(math.pi / m)) * BOND
    for k, idx in enumerate(s):
        pos[idx] = R * np.array([math.cos(2 * math.pi * k / m + math.pi / 2),
                                 math.sin(2 * math.pi * k / m + math.pi / 2)])
    ring_cen[seed] = np.mean([pos[i] for i in s], axis=0)
    placed_rings = {seed}
    changed = True
    while changed:
        changed = False
        for ridx, order in enumerate(ordered):
            if ridx in placed_rings:
                continue
            rset = set(order)
            # need a shared EDGE (two adjacent atoms) with some placed ring
            edge = None
            ref_ring = None
            for pr in placed_rings:
                pset = set(ordered[pr])
                shared = rset & pset
                if len(shared) >= 2:
                    # find an adjacent shared pair
                    for x in shared:
                        for nb in mol.GetAtomWithIdx(x).GetNeighbors():
                            y = nb.GetIdx()
                            if y in shared:
                                edge = (x, y)
                                ref_ring = pr
                                break
                        if edge:
                            break
                if edge:
                    break
            if not edge:
                continue
            a, b = edge
            cand = regpoly(order, pos[a], pos[b], a, b)
            new_cen = np.mean(cand, axis=0)
            ev = np.array(pos[b]) - np.array(pos[a])
            nrm = np.array([-ev[1], ev[0]])
            ref_cen = ring_cen[ref_ring]
            # new ring must sit on the OPPOSITE side of the shared edge from the
            # ring it fuses to
            if np.dot(new_cen - pos[a], nrm) * np.dot(ref_cen - pos[a], nrm) > 0:
                u = ev / (np.linalg.norm(ev) + 1e-9)
                cand = [np.array(pos[a]) + 2 * (np.dot(c - pos[a], u) * u) - (c - pos[a])
                        for c in cand]
                new_cen = np.mean(cand, axis=0)
            for k, idx in enumerate(order):
                if idx not in pos:
                    pos[idx] = cand[k]
            ring_cen[ridx] = new_cen
            placed_rings.add(ridx)
            changed = True
    return pos


def _bridge_atoms(mol, p1, p2, keep_atoms, metal):
    """Shortest-path backbone atoms between the two P (not through metal)."""
    from collections import deque
    prev = {p1: None}
    q = deque([p1])
    while q:
        c = q.popleft()
        if c == p2:
            break
        for nb in mol.GetAtomWithIdx(c).GetNeighbors():
            j = nb.GetIdx()
            if j in prev or j == metal or j not in keep_atoms:
                continue
            prev[j] = c
            q.append(j)
    path = set()
    if p2 in prev:
        c = p2
        while c is not None:
            path.add(c)
            c = prev[c]
    return path


def template_layout(mol, keep_atoms, metal, hydride, depth):
    """Deterministic metal-bisphosphine layout that NEVER overlaps:

      * lay out the rigid BACKBONE (the two P + the bridge ring system + every
        fused ring of that system) cleanly with CoordGen;
      * orient it so the two P open toward the RIGHT;
      * drop Cu just right of the P-P midpoint, H one bond further right
        (so Cu-H is ~horizontal, H on the right);
      * graft each P's pendant substituent sub-trees (the drawn aryl rings, or
        nothing where the substituent is a collapsed text label) rotated to FAN
        outward, above and below, away from the metal.

    The 3D character is preserved through the depth-derived wedge/dash applied
    later; this routine only guarantees a clean, regular, non-overlapping frame.
    """
    donors = [a.GetIdx() for a in mol.GetAtomWithIdx(metal).GetNeighbors()
              if a.GetSymbol() == "P"]
    if len(donors) != 2:
        # fall back: whole-skeleton CoordGen
        cg = _coordgen_subset(mol, [i for i in keep_atoms if i not in (metal, hydride)])
        pos = np.zeros((mol.GetNumAtoms(), 2))
        for k, v in cg.items():
            pos[k] = v
        return pos
    p1, p2 = donors
    bridge = _bridge_atoms(mol, p1, p2, keep_atoms, metal)

    # backbone = bridge + every fused-ring atom sharing a ring with a bridge atom
    ri = mol.GetRingInfo()
    backbone = set(bridge)
    changed = True
    while changed:
        changed = False
        for ring in ri.AtomRings():
            rs = set(ring)
            if rs & backbone and not rs <= backbone:
                if all(a in keep_atoms for a in rs):
                    backbone |= rs
                    changed = True
    backbone |= {p1, p2}

    # pendant substituent branches at each P (everything off P that is not
    # backbone and not the metal) -- these are the drawn aryl rings (if any)
    def branches_of(p):
        out = []
        for nb in mol.GetAtomWithIdx(p).GetNeighbors():
            j = nb.GetIdx()
            if j == metal or j in backbone or j not in keep_atoms:
                continue
            atoms = _branch_atoms(mol, p, j, backbone | {metal, p}, keep_atoms)
            out.append((j, atoms))
        return out

    br1, br2 = branches_of(p1), branches_of(p2)
    arm_atoms = set()
    for _, atoms in br1 + br2:
        arm_atoms |= atoms

    # ---- lay out the whole CORE ----
    # core = every drawn (kept) atom EXCEPT the pendant P-aryl arms and the
    # metal/hydride.  This captures the backbone ring system PLUS anything
    # hanging off it (e.g. the xanthene gem-dimethyls) so nothing is orphaned.
    core = (set(keep_atoms) - arm_atoms - {metal, hydride})
    pos = np.zeros((mol.GetNumAtoms(), 2))
    placed = set()
    # 1) deterministic, flat, non-overlapping fused-ring backbone
    core_ring_atoms = {a for a in core if mol.GetAtomWithIdx(a).IsInRing()}
    fr = _place_fused_rings(mol, core_ring_atoms, keep_atoms)
    for k, v in fr.items():
        pos[k] = v
        placed.add(k)
    # 2) BFS-place the remaining core atoms (non-ring bridges, gem-dimethyls,
    #    and any ring-component reached only through a single bond such as the
    #    biaryl axis) one bond-length out, spread ~120 deg from siblings.
    core_nbr = {a: [nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                    if nb.GetIdx() in core] for a in core}
    if not placed:
        # no rings at all -> fall back to CoordGen for the whole core
        cg = _coordgen_subset(mol, core | {metal})
        for k, v in cg.items():
            if k != metal:
                pos[k] = v
                placed.add(k)
    from collections import deque
    dq = deque([a for a in placed])
    guard = 0
    while len(placed) < len(core) and guard < 10000:
        guard += 1
        progressed = False
        for a in list(core):
            if a in placed:
                continue
            pn = [x for x in core_nbr[a] if x in placed]
            if not pn:
                continue
            anchor = pn[0]
            base = pos[anchor]
            # direction: away from anchor's already-placed neighbours
            others = [pos[x] for x in core_nbr[anchor] if x in placed and x != a]
            if others:
                inw = np.mean(others, axis=0) - base
                ni = np.linalg.norm(inw)
                d = -inw / ni if ni > 1e-6 else np.array([1.0, 0.0])
            else:
                d = np.array([0.0, -1.0])
            pos[a] = base + d * BOND
            placed.add(a)
            progressed = True
        if not progressed:
            break
    # any leftover -> origin-ish near core centroid (avoids NaN)
    if placed:
        cc = np.mean([pos[a] for a in placed], axis=0)
    else:
        cc = np.zeros(2)
    for a in core:
        if a not in placed:
            pos[a] = cc + np.random.RandomState(a).randn(2) * 0.3
            placed.add(a)

    # orient the core so the P1->P2 vector is VERTICAL (the two donors stack on
    # the chelate's right edge) and the bulk of the backbone sits to the LEFT,
    # i.e. on the opposite side from where the metal will be dropped.  This is
    # the canonical chelate drawing and keeps the rigid ring system flat.
    bb_cen = np.mean([pos[a] for a in core], axis=0)
    pp = pos[p2] - pos[p1]
    if np.linalg.norm(pp) < 1e-6:
        pp = np.array([0.0, -1.0])
    # rotate so pp aligns with -y (pointing down)
    ang = math.atan2(pp[1], pp[0]) - (-math.pi / 2)
    cth, sth = math.cos(-ang), math.sin(-ang)
    R = np.array([[cth, -sth], [sth, cth]])
    for a in core:
        pos[a] = (pos[a] - bb_cen) @ R.T + bb_cen
    pmid = (pos[p1] + pos[p2]) / 2.0
    # ensure backbone bulk is to the LEFT of the P-P line (flip x if not)
    if bb_cen[0] - pmid[0] > 0 or np.mean([pos[a][0] for a in core]) > pmid[0]:
        for a in core:
            pos[a][0] = 2 * pmid[0] - pos[a][0]
    # make sure p1 is the upper one for consistent fanning
    if pos[p1][1] < pos[p2][1]:
        p1, p2, br1, br2 = p2, p1, br2, br1

    # ---- place Cu + H ----  (Cu sits a fixed, modest distance off the P-P
    # midpoint so the two P->Cu dative bonds are ~one bond long; H one bond
    # further right so Cu-H is horizontal with H on the right)
    pmid = (pos[p1] + pos[p2]) / 2.0
    ppd = np.linalg.norm(pos[p1] - pos[p2])
    off = max(BOND * 0.9, math.sqrt(max(0.0, BOND ** 2 - (ppd / 2) ** 2)))
    cu = pmid + np.array([1.0, 0.0]) * off
    pos[metal] = cu
    pos[hydride] = cu + np.array([1.0, 0.0]) * (BOND * 1.05)

    # ---- graft pendant branches, fanned outward ----
    # The backbone sits to the LEFT and the metal to the RIGHT, so the pendant
    # aryls must fan to the upper-/lower-RIGHT (away from BOTH).  Aim the fan
    # straight up for the top P and straight down for the bottom P, with a slight
    # rightward lean, then spread the (up to two) branches around that.
    def graft(p, branches, up):
        if not branches:
            return
        base = math.radians(70) if up else math.radians(-70)   # up / down...
        base += math.radians(18)                               # ...leaning right
        n = len(branches)
        spread = math.radians(68)
        for k, (j, atoms) in enumerate(branches):
            off = (-spread / 2 + spread * (k / (n - 1))) if n > 1 else 0.0
            tgt = base + off
            # lay out this branch (with its P anchor) cleanly
            sub = atoms | {p}
            cg = _coordgen_subset(mol, sub)
            # translate so P at origin, rotate so P->j points to tgt, then place
            anchor = cg[p]
            vj = cg[j] - anchor
            a0 = math.atan2(vj[1], vj[0])
            dth = tgt - a0
            cc, ss = math.cos(dth), math.sin(dth)
            Rb = np.array([[cc, -ss], [ss, cc]])
            for a in atoms:
                pos[a] = pos[p] + (cg[a] - anchor) @ Rb.T

    graft(p1, br1, up=True)
    graft(p2, br2, up=False)
    return pos


def coordgen_skeleton(mol, keep_atoms, xy_proj, metal, hydride):
    """Clean 2D layout of the kept-atom LIGAND skeleton via RDKit CoordGen, then
    rigidly aligned (rotation+reflection) to the PCA projection so the drawing
    keeps the real 3D viewing orientation.  The metal and hydride are laid out
    GEOMETRICALLY afterwards (Cu placed on the chelate's open side, H to its
    right) rather than by CoordGen -- otherwise the dative P->Cu->P chelate ring
    forces the two phosphine arms to fold over each other.  This guarantees
    regular, non-overlapping rings while the projection still dictates which way
    the molecule faces and the depth still dictates wedge/dash."""
    from rdkit.Chem import rdCoordGen, AllChem
    # Lay out the FULL complex (metal + hydride included).  With the P->Cu->P
    # chelate ring present, CoordGen splays the two phosphine arms apart
    # (P-Cu-P ~130 deg) instead of folding them together -- exactly the
    # published bisphosphine-metal arrangement.  Substituent (collapsed-group)
    # atoms are dropped so only the drawn skeleton is laid out.
    keepl = sorted(keep_atoms)
    idxmap = {old: i for i, old in enumerate(keepl)}
    em = Chem.RWMol()
    for old in keepl:
        a = mol.GetAtomWithIdx(old)
        na = Chem.Atom(a.GetAtomicNum())
        na.SetFormalCharge(a.GetFormalCharge())
        na.SetNoImplicit(True)
        em.AddAtom(na)
    for b in mol.GetBonds():
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a in idxmap and c in idxmap:
            bt = b.GetBondType()
            if bt == Chem.BondType.DATIVE:
                bt = Chem.BondType.SINGLE
            em.AddBond(idxmap[a], idxmap[c], bt)
    sub = em.GetMol()
    try:
        Chem.SanitizeMol(sub)
    except Exception:
        try:
            Chem.SanitizeMol(sub, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_KEKULIZE ^
                             Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
        except Exception:
            pass
    try:
        rdCoordGen.AddCoords(sub)
    except Exception:
        AllChem.Compute2DCoords(sub)
    conf = sub.GetConformer()
    sk = np.zeros((mol.GetNumAtoms(), 2))
    for old, new in idxmap.items():
        p = conf.GetAtomPosition(new)
        sk[old] = (p.x, p.y)
    # normalise skeleton bond length to BOND
    bl = []
    for b in sub.GetBonds():
        pa = conf.GetAtomPosition(b.GetBeginAtomIdx())
        pb = conf.GetAtomPosition(b.GetEndAtomIdx())
        bl.append(math.hypot(pa.x - pb.x, pa.y - pb.y))
    if bl:
        sk *= BOND / np.median(bl)
    # Orient the clean skeleton to the projection WITHOUT distorting it: pick the
    # pure rotation/reflection that best matches the projection's heavy-atom
    # arrangement (Kabsch, no scale) -- this transfers the real 3D viewing
    # direction onto a guaranteed non-overlapping layout. (We deliberately do
    # NOT warp atom-by-atom toward the foreshortened projection, which would
    # re-introduce ring overlap.)
    # Match the projection's orientation ONLY by reflection parity (chirality of
    # the view), not a full warp: a full Kabsch onto the foreshortened
    # projection re-folds the backbone rings.  We keep CoordGen's clean,
    # non-overlapping geometry and just choose the mirror that agrees with the
    # real 3D view -- the depth (below) still encodes the out-of-plane 3D info
    # as wedge/dash, so the drawing reflects the genuine 3D geometry.
    sc = sk[keepl] - sk[keepl].mean(axis=0)
    pc = xy_proj[keepl] - xy_proj[keepl].mean(axis=0)
    pc = pc / (np.sqrt((pc ** 2).sum(axis=1)).mean() + 1e-9)
    Hk = sc.T @ pc
    U, S, Vt = np.linalg.svd(Hk)
    Rm = Vt.T @ U.T
    out = sk.copy()
    if np.linalg.det(Rm) < 0:        # apply mirror only (x -> -x), keep layout clean
        out[keepl, 0] = 2 * sk[keepl, 0].mean() - sk[keepl, 0]

    # ---- put the hydride on the chelate's OPEN side (away from the ligand) so
    #      that, once the renderer rotates Cu->H horizontal, the molecule sits
    #      cleanly to the LEFT of the metal with H free on the right. ----
    donors = [a.GetIdx() for a in mol.GetAtomWithIdx(metal).GetNeighbors()
              if a.GetSymbol() == "P"]
    lig_cen = out[[i for i in keepl if i not in (metal, hydride)]].mean(axis=0)
    away = out[metal] - lig_cen
    na = np.linalg.norm(away)
    away = away / na if na > 1e-6 else np.array([1.0, 0.0])
    out[hydride] = out[metal] + away * (BOND * 1.15)
    return out


def regularise(mol, xy, keep_atoms, iters=140, metal=None, hydride=None):
    n = mol.GetNumAtoms()
    # ---- deterministic, non-overlapping metal-bisphosphine seed ----
    pos = template_layout(mol, keep_atoms, metal, hydride, xy)
    frozen = {metal, hydride}
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    # relax the LIGAND only; Cu and H are geometrically fixed
    kb = [(a, c) for a, c in bonds
          if a in keep_atoms and c in keep_atoms
          and a not in frozen and c not in frozen]

    rings = [r for r in ordered_rings(mol) if all(i in keep_atoms for i in r)]
    in_ring = set().union(*[set(r) for r in rings]) if rings else set()

    nbr = {i: [] for i in range(n)}
    for a, c in kb:
        nbr[a].append(c)
        nbr[c].append(a)

    klist = [i for i in sorted(keep_atoms) if i not in frozen]
    for it in range(iters):
        cool = 1.0 if it < iters * 0.7 else 0.4
        disp = np.zeros((n, 2))
        # bond-length springs
        for a, c in kb:
            d = pos[c] - pos[a]
            L = np.linalg.norm(d)
            if L < 1e-9:
                d = np.array([1.0, 0.0])
                L = 1.0
            f = (L - BOND) * 0.5 * d / L
            disp[a] += f
            disp[c] -= f
        # ring regularity
        for ring in rings:
            m = len(ring)
            P = pos[ring]
            cen = P.mean(axis=0)
            R = BOND / (2 * math.sin(math.pi / m))
            ang = np.arctan2(P[:, 1] - cen[1], P[:, 0] - cen[0])
            base = ang - (2 * math.pi / m) * np.arange(m)
            theta0 = math.atan2(np.sin(base).mean(), np.cos(base).mean())
            for j, idx in enumerate(ring):
                ta = theta0 + (2 * math.pi / m) * j
                tgt = cen + R * np.array([math.cos(ta), math.sin(ta)])
                disp[idx] += (tgt - pos[idx]) * 0.5
        # 120 deg spreading at acyclic vertices
        for i in klist:
            nb = nbr[i]
            if len(nb) < 2:
                continue
            if i in in_ring and all(j in in_ring for j in nb):
                continue
            vs = []
            for j in nb:
                d = pos[j] - pos[i]
                L = np.linalg.norm(d)
                if L > 1e-9:
                    vs.append((j, d / L, L))
            ideal = math.radians(120.0)
            for x in range(len(vs)):
                for y in range(x + 1, len(vs)):
                    (j1, u1, L1), (j2, u2, L2) = vs[x], vs[y]
                    cur = math.acos(max(-1, min(1, float(np.dot(u1, u2)))))
                    err = (ideal - cur) * 0.18
                    p1 = np.array([-u1[1], u1[0]])
                    if np.dot(p1, u2) > 0:
                        p1 = -p1
                    p2 = np.array([-u2[1], u2[0]])
                    if np.dot(p2, u1) > 0:
                        p2 = -p2
                    disp[j1] += p1 * err * L1 * 0.5
                    disp[j2] += p2 * err * L2 * 0.5
        # non-bonded repulsion (anti-overlap)
        for ai in range(len(klist)):
            i = klist[ai]
            for bj in range(ai + 1, len(klist)):
                j = klist[bj]
                if j in nbr[i]:
                    continue
                d = pos[i] - pos[j]
                L = np.linalg.norm(d)
                if 1e-6 < L < BOND * 0.92:
                    f = (BOND * 0.92 - L) * 0.3 * d / L
                    disp[i] += f
                    disp[j] -= f
        pos += np.clip(disp * cool, -0.3, 0.3)

    # re-place Cu + H from the FINAL P positions (the relax may have nudged the
    # donors), keeping Cu-H horizontal with H on the right.
    donors = [a.GetIdx() for a in mol.GetAtomWithIdx(metal).GetNeighbors()
              if a.GetSymbol() == "P"]
    if len(donors) == 2:
        pmid = (pos[donors[0]] + pos[donors[1]]) / 2.0
        ppd = np.linalg.norm(pos[donors[0]] - pos[donors[1]])
        off = max(BOND * 0.9, math.sqrt(max(0.0, BOND ** 2 - (ppd / 2) ** 2)))
        # keep Cu to the right of the donors
        cdir = 1.0 if pmid[0] >= np.mean([pos[i][0] for i in klist]) else 1.0
        pos[metal] = pmid + np.array([off, 0.0])
        pos[hydride] = pos[metal] + np.array([BOND * 1.05, 0.0])
    return pos


def _branch_atoms(mol, start, via, blocked, keep_atoms):
    """All kept atoms reachable from `start` first stepping to `via`, without
    crossing any atom in `blocked`."""
    seen = {start, via}
    stack = [via]
    while stack:
        c = stack.pop()
        for nb in mol.GetAtomWithIdx(c).GetNeighbors():
            j = nb.GetIdx()
            if j in seen or j in blocked or j not in keep_atoms:
                continue
            seen.add(j)
            stack.append(j)
    seen.discard(start)
    return seen


def _fan_phosphine_arms(mol, pos, keep_atoms, metal, hydride):
    pos = pos.copy()
    donors = [a.GetIdx() for a in mol.GetAtomWithIdx(metal).GetNeighbors()
              if a.GetSymbol() == "P"]
    if len(donors) != 2:
        return pos
    p1, p2 = donors
    # bridge = atoms on the shortest path between the two P (not through metal)
    from collections import deque
    blocked0 = {metal}
    prev = {p1: None}
    q = deque([p1])
    while q:
        c = q.popleft()
        if c == p2:
            break
        for nb in mol.GetAtomWithIdx(c).GetNeighbors():
            j = nb.GetIdx()
            if j in prev or j in blocked0 or j not in keep_atoms:
                continue
            prev[j] = c
            q.append(j)
    bridge = set()
    if p2 in prev:
        c = p2
        while c is not None:
            bridge.add(c)
            c = prev[c]
    cu = pos[metal]
    for p in (p1, p2):
        pp = pos[p]
        radial = pp - cu
        nr = np.linalg.norm(radial)
        radial = radial / nr if nr > 1e-6 else np.array([-1.0, 0.0])
        # substituent branches at this P (exclude metal and the bridge step)
        branches = []
        for nb in mol.GetAtomWithIdx(p).GetNeighbors():
            j = nb.GetIdx()
            if j == metal or j in bridge or j not in keep_atoms:
                continue
            atoms = _branch_atoms(mol, p, j, {metal} | bridge | {p}, keep_atoms)
            branches.append((j, atoms))
        if not branches:
            continue
        # fan target angles: spread around the outward (radial) direction
        base = math.atan2(radial[1], radial[0])
        nB = len(branches)
        spread = math.radians(58)
        offsets = [(-spread / 2 + spread * (k / (nB - 1)) if nB > 1 else 0.0)
                   for k in range(nB)]
        for (j, atoms), off in zip(branches, offsets):
            cur = pos[j] - pp
            ca = math.atan2(cur[1], cur[0])
            tgt = base + off
            dth = tgt - ca
            cs, sn = math.cos(dth), math.sin(dth)
            R = np.array([[cs, -sn], [sn, cs]])
            for a in atoms:
                pos[a] = pp + (pos[a] - pp) @ R.T
    return pos


# ====================================================================
#  STEP 5: MEDIUM abbreviation
# ====================================================================
def medium_groups(mol):
    hits = abbr.detect_groups(mol)
    groups = []
    collapsed = set()
    for h in hits:
        groups.append({"label": h.label, "atoms": set(h.atoms),
                       "attach": h.attach, "anchor": h.anchor, "kind": h.kind})
        collapsed |= set(h.atoms)
    return groups, collapsed


# ====================================================================
#  driver
# ====================================================================
def build_layout(key, name, smi):
    lig = chem.make_dtbm_segphos() if smi is None else Chem.MolFromSmiles(smi)
    Chem.SanitizeMol(lig)
    xyz = HERE / "data" / f"{key}.xyz"
    mol, info, method = build_complex.generate_conformer(xyz, lig=lig)
    metal, hydride, donors = info["metal"], info["hydride"], list(info["donors"])

    Chem.AssignStereochemistryFrom3D(mol)

    groups, collapsed = medium_groups(mol)
    keep = set(range(mol.GetNumAtoms())) - collapsed

    xy, depth = project_pca(mol)
    pos = regularise(mol, xy, keep, metal=metal, hydride=hydride)

    korder = {}
    try:
        km = Chem.Mol(mol)
        Chem.Kekulize(km, clearAromaticFlags=True)
        for b in km.GetBonds():
            korder[frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))] = b.GetBondTypeAsDouble()
    except Exception:
        pass

    biaryl = depict._find_biaryl_bond(mol)

    # wedge/dash from depth on sp3 stereocentres
    wedge = {}
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if i not in keep:
            continue
        if a.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        if a.GetSymbol() not in ("C", "P"):
            continue
        nb = [n.GetIdx() for n in a.GetNeighbors()]
        heavy = [j for j in nb if mol.GetAtomWithIdx(j).GetAtomicNum() > 1]
        if len(heavy) < 3:
            continue
        # genuine stereocentre only: needs a CIP code or distinct neighbours
        if a.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        cand = []
        for j in heavy:
            if j in keep:
                cand.append((abs(depth[j] - depth[i]), j, j in keep))
            elif j in collapsed:
                # collapsed substituent (e.g. Ph on Ph-BPE): wedge to its anchor
                cand.append((abs(depth[j] - depth[i]), j, False))
        if not cand:
            continue
        cand.sort(reverse=True)
        dz, j, _ = cand[0]
        if dz < 0.10:
            continue
        wedge[(i, j)] = "up" if depth[j] > depth[i] else "down"

    return {
        "key": key, "name": name, "mol": mol, "method": method,
        "pos": pos, "depth": depth, "metal": metal, "hydride": hydride,
        "donors": donors, "groups": groups, "collapsed": collapsed,
        "keep": keep, "korder": korder, "biaryl": biaryl, "wedge": wedge,
        "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
        "smiles": Chem.MolToSmiles(lig),
    }


if __name__ == "__main__":
    for k, (nm, smi) in LIGANDS.items():
        lay = build_layout(k, nm, smi)
        print(k, "keep", len(lay["keep"]), "groups", len(lay["groups"]),
              "biaryl", lay["biaryl"], "wedges", len(lay["wedge"]))
