#!/usr/bin/env python3
"""
complexes.py -- the GENERALISED organometallic gallery.

This is NOT a from-scratch renderer: it is a thin *connectivity feeder* (the
"agent") on top of the existing depiction tool.  Each species is described only
by its CONNECTIVITY -- an RDKit Mol whose donor->metal bonds are DATIVE, whose
ancillary metal-X bonds are ordinary covalent bonds, and whose eta-n
cyclopentadienyl/arene ligands have every ring carbon DATIVE-bonded to the metal
(the "cyclopentyl nu5" option).  Everything else -- 2-D seeding, the perspective
Cp disc rotated to face the metal, the dashed metal->centroid eta-n bond, the
biaryl-axis rotation into the plane for a clean metal...C_ipso contact, the
physics relax and relief -- is done by the SHARED tool:

    seed_from_smiles.scene_from_mol   (draw, with eta-n + coord perception)
    relax_harness.Harness / relief    (physics, biaryl_into_plane, declutter...)
    relaxer_energy.relax              (spring solver)

So adding a new complex == feeding new connectivity; no new drawing code.

Species: Ru-MACHO PNP pincer, Pd-SPhos biaryl (Pd...Cipso), Cu-oxalamide,
Fe(salen)Cl, Grubbs-II, Hoveyda-Grubbs-II, Cp2Rh (rhodocene), Cp2ZrCl2 (bent).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from rdkit import Chem                          # noqa: E402

import relax_harness as R                       # noqa: E402
import relaxer_energy                           # noqa: E402
import seed_from_smiles as S                    # noqa: E402
from template_draw import L                      # noqa: E402

def _safe(name):
    return name.replace("/", "-").replace(" ", "_").replace("·", ".")


CANVAS, SCALE, ANCHOR = (1240, 1040), 42.0, (620.0, 520.0)
W = {"w_bond": 6.0, "w_angle": 1.9, "w_overlap": 17.5,
     "w_rigid": 67.0, "maxiter": 180}
WHI = {"w_bond": 6.0, "w_angle": 1.6, "w_overlap": 26.0,
       "w_rigid": 67.0, "maxiter": 240}


# ======================================================================== #
#  connectivity helpers (used only to FEED the tool)
# ======================================================================== #
def _aromatic_ring(rw, n=6):
    ids = [rw.AddAtom(Chem.Atom(6)) for _ in range(n)]
    for k in range(n):
        rw.AddBond(ids[k], ids[(k + 1) % n], Chem.BondType.AROMATIC)
        rw.GetAtomWithIdx(ids[k]).SetIsAromatic(True)
    return ids


def _cyclohexyl(rw, attach):
    cs = [rw.AddAtom(Chem.Atom(6)) for _ in range(6)]
    for k in range(6):
        rw.AddBond(cs[k], cs[(k + 1) % 6], Chem.BondType.SINGLE)
    rw.AddBond(attach, cs[0], Chem.BondType.SINGLE)
    return cs


def _cp_eta5(rw, metal):
    """A cyclopentadienyl ring eta5-bound to `metal`: Kekule diene-anion, every
    ring carbon DATIVE to the metal (the tool turns this into a perspective disc)."""
    cs = [rw.AddAtom(Chem.Atom(6)) for _ in range(5)]
    orders = [Chem.BondType.DOUBLE, Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
              Chem.BondType.SINGLE, Chem.BondType.SINGLE]
    for k in range(5):
        rw.AddBond(cs[k], cs[(k + 1) % 5], orders[k])
    rw.GetAtomWithIdx(cs[4]).SetFormalCharge(-1)
    for a in cs:
        rw.AddBond(a, metal, Chem.BondType.DATIVE)
    return cs


def _simes(rw):
    """SIMes N-heterocyclic carbene fragment (carbene C = atom returned).  Both N
    carry a mesityl; the backbone is the saturated -CH2CH2-."""
    carbene = rw.AddAtom(Chem.Atom(6))
    rw.GetAtomWithIdx(carbene).SetNoImplicit(True)         # divalent carbene
    n1 = rw.AddAtom(Chem.Atom(7))
    n2 = rw.AddAtom(Chem.Atom(7))
    c4 = rw.AddAtom(Chem.Atom(6))
    c5 = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(carbene, n1, Chem.BondType.SINGLE)
    rw.AddBond(carbene, n2, Chem.BondType.SINGLE)
    rw.AddBond(n1, c4, Chem.BondType.SINGLE)
    rw.AddBond(c4, c5, Chem.BondType.SINGLE)
    rw.AddBond(c5, n2, Chem.BondType.SINGLE)
    for n in (n1, n2):
        mes = _aromatic_ring(rw, 6)
        rw.AddBond(n, mes[0], Chem.BondType.SINGLE)
        for k in (1, 3, 5):                               # 2,4,6-trimethyl
            me = rw.AddAtom(Chem.Atom(6))
            rw.AddBond(mes[k], me, Chem.BondType.SINGLE)
    return carbene


# ======================================================================== #
#  the connectivity FEED -- one builder per species, each returns a Mol
# ======================================================================== #
def build_ru_pnp():
    """RuHCl(CO)[HN(CH2CH2PiPr2)2] -- Ru-MACHO PNP pincer (DATIVE P,N,P; H, Cl,
    C#O ancillary)."""
    lig = Chem.MolFromSmiles("C(C)(C)P(C(C)C)CCNCCP(C(C)C)C(C)C")
    Chem.SanitizeMol(lig)
    rw = Chem.RWMol(lig)
    P = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "P"]
    N = next(a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "N")
    ru = rw.AddAtom(Chem.Atom(44))
    h = rw.AddAtom(Chem.Atom(1)); cl = rw.AddAtom(Chem.Atom(17))
    c = rw.AddAtom(Chem.Atom(6)); o = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(ru, h, Chem.BondType.SINGLE)
    rw.AddBond(ru, cl, Chem.BondType.SINGLE)
    rw.AddBond(ru, c, Chem.BondType.SINGLE)
    rw.AddBond(c, o, Chem.BondType.TRIPLE)
    rw.GetAtomWithIdx(c).SetFormalCharge(-1)
    rw.GetAtomWithIdx(o).SetFormalCharge(1)
    rw.AddBond(N, ru, Chem.BondType.DATIVE)
    for p in P:
        rw.AddBond(p, ru, Chem.BondType.DATIVE)
    m = rw.GetMol(); Chem.SanitizeMol(m)
    b = m.GetBondBetweenAtoms(ru, c)                       # solid Ru-C, triple C#O
    if b is not None:
        b.SetBondType(Chem.BondType.SINGLE)
    return m


def build_pd_biaryl():
    """Pd(SPhos)(Ph)(Br): P -> Pd (DATIVE) and the lower-ring ipso carbon makes the
    Pd...Cipso DATIVE contact (dashed); Ph and Br covalent."""
    lig = Chem.MolFromSmiles("COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1")
    Chem.SanitizeMol(lig)
    P = next(a.GetIdx() for a in lig.GetAtoms() if a.GetSymbol() == "P")
    ri = lig.GetRingInfo()
    arom6 = [set(r) for r in ri.AtomRings()
             if len(r) == 6 and all(lig.GetAtomWithIdx(a).GetIsAromatic() for a in r)]

    def n_ome(ring):
        return sum(1 for a in ring
                   for nb in lig.GetAtomWithIdx(a).GetNeighbors()
                   if nb.GetIdx() not in ring and nb.GetSymbol() == "O")
    dim = max(arom6, key=n_ome)
    other = next(r for r in arom6 if r is not dim)
    cipso = next(a for a in dim for nb in lig.GetAtomWithIdx(a).GetNeighbors()
                 if nb.GetIdx() in other)
    rw = Chem.RWMol(lig)
    pd = rw.AddAtom(Chem.Atom(46))
    ph = _aromatic_ring(rw, 6)
    br = rw.AddAtom(Chem.Atom(35))
    rw.AddBond(P, pd, Chem.BondType.DATIVE)
    rw.AddBond(cipso, pd, Chem.BondType.DATIVE)
    rw.AddBond(ph[0], pd, Chem.BondType.SINGLE)
    rw.AddBond(br, pd, Chem.BondType.SINGLE)
    m = rw.GetMol(); Chem.SanitizeMol(m)
    return m


def build_cu_oxalamide():
    """(N,N'-dicyclohexyloxalamide)CuBr -- a Dawei-Ma oxalamide: both amide N
    chelate Cu (DATIVE); Br ancillary.  All-N donor set."""
    lig = Chem.MolFromSmiles("O=C(NC1CCCCC1)C(=O)NC1CCCCC1")
    Chem.SanitizeMol(lig)
    rw = Chem.RWMol(lig)
    N = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "N"]
    cu = rw.AddAtom(Chem.Atom(29)); br = rw.AddAtom(Chem.Atom(35))
    for n in N:
        rw.AddBond(n, cu, Chem.BondType.DATIVE)
    rw.AddBond(cu, br, Chem.BondType.SINGLE)
    m = rw.GetMol(); Chem.SanitizeMol(m)
    return m


def build_fe_salen():
    """Fe(salen)Cl -- N2O2 Schiff base: two phenolate O + two imine N DATIVE to Fe,
    axial Cl covalent."""
    salen = Chem.MolFromSmiles(r"Oc1ccccc1/C=N/CCN=C/c1ccccc1O")
    Chem.SanitizeMol(salen)
    rw = Chem.RWMol(salen)
    O = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "O"]
    N = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "N"]
    fe = rw.AddAtom(Chem.Atom(26)); cl = rw.AddAtom(Chem.Atom(17))
    for o in O:
        rw.GetAtomWithIdx(o).SetFormalCharge(-1)
        rw.AddBond(o, fe, Chem.BondType.DATIVE)
    for n in N:
        rw.AddBond(n, fe, Chem.BondType.DATIVE)
    rw.GetAtomWithIdx(fe).SetFormalCharge(3)
    rw.AddBond(fe, cl, Chem.BondType.SINGLE)
    m = rw.GetMol(); Chem.SanitizeMol(m)
    return m


def build_grubbs2():
    """Grubbs-II: (SIMes)(PCy3)Cl2Ru=CHPh.  NHC carbene C -> Ru and PCy3 P -> Ru
    (DATIVE); Ru=CHPh alkylidene (double) and two Cl covalent."""
    rw = Chem.RWMol()
    carbene = _simes(rw)
    ru = rw.AddAtom(Chem.Atom(44))
    ch = rw.AddAtom(Chem.Atom(6))                         # =CH
    ph = _aromatic_ring(rw, 6)
    rw.AddBond(ch, ph[0], Chem.BondType.SINGLE)
    rw.AddBond(ru, ch, Chem.BondType.DOUBLE)              # Ru=C
    p = rw.AddAtom(Chem.Atom(15))
    for _ in range(3):
        _cyclohexyl(rw, p)
    rw.AddBond(p, ru, Chem.BondType.DATIVE)
    for _ in range(2):
        cl = rw.AddAtom(Chem.Atom(17)); rw.AddBond(ru, cl, Chem.BondType.SINGLE)
    rw.AddBond(carbene, ru, Chem.BondType.DATIVE)
    m = rw.GetMol(); Chem.SanitizeMol(m)
    return m


def build_hoveyda_grubbs():
    """Hoveyda-Grubbs-II: (SIMes)Cl2Ru=CH-(2-iPrO-C6H4), the ether O CHELATING Ru.
    NHC carbene C -> Ru and the isopropoxy O -> Ru (DATIVE); Ru=CH(aryl) double,
    two Cl covalent."""
    rw = Chem.RWMol()
    carbene = _simes(rw)
    ru = rw.AddAtom(Chem.Atom(44))
    ch = rw.AddAtom(Chem.Atom(6))                         # =CH
    ar = _aromatic_ring(rw, 6)                            # benzylidene aryl
    rw.AddBond(ch, ar[0], Chem.BondType.SINGLE)
    rw.AddBond(ru, ch, Chem.BondType.DOUBLE)
    o = rw.AddAtom(Chem.Atom(8))                          # ortho-O of the chelate
    rw.AddBond(ar[1], o, Chem.BondType.SINGLE)
    ipr = rw.AddAtom(Chem.Atom(6))                        # O-CH(CH3)2
    rw.AddBond(o, ipr, Chem.BondType.SINGLE)
    for _ in range(2):
        me = rw.AddAtom(Chem.Atom(6)); rw.AddBond(ipr, me, Chem.BondType.SINGLE)
    rw.AddBond(o, ru, Chem.BondType.DATIVE)               # ether O -> Ru chelate
    for _ in range(2):
        cl = rw.AddAtom(Chem.Atom(17)); rw.AddBond(ru, cl, Chem.BondType.SINGLE)
    rw.AddBond(carbene, ru, Chem.BondType.DATIVE)
    m = rw.GetMol(); Chem.SanitizeMol(m)
    return m


def build_cp2rh():
    """Rhodocene Cp2Rh -- two eta5 Cp on Rh, no sigma ligand (ferrocene-style
    parallel sandwich)."""
    rw = Chem.RWMol()
    rh = rw.AddAtom(Chem.Atom(45))
    _cp_eta5(rw, rh); _cp_eta5(rw, rh)
    m = rw.GetMol(); Chem.SanitizeMol(m)
    return m


def build_cp2zrcl2():
    """Zirconocene dichloride Cp2ZrCl2 -- two eta5 Cp + two Cl (bent metallocene)."""
    rw = Chem.RWMol()
    zr = rw.AddAtom(Chem.Atom(40))
    _cp_eta5(rw, zr); _cp_eta5(rw, zr)
    for _ in range(2):
        cl = rw.AddAtom(Chem.Atom(17)); rw.AddBond(zr, cl, Chem.BondType.SINGLE)
    m = rw.GetMol(); Chem.SanitizeMol(m)
    return m


FEED = {
    "Ru-MACHO (PNP pincer)": build_ru_pnp,
    "Pd-SPhos (biaryl, Pd···Cipso)": build_pd_biaryl,
    "Cu-oxalamide (N,N-chelate)": build_cu_oxalamide,
    "Fe(salen)Cl (N2O2)": build_fe_salen,
    "Grubbs II (SIMes/PCy3)": build_grubbs2,
    "Hoveyda–Grubbs II": build_hoveyda_grubbs,
    "Cp2Rh (rhodocene)": build_cp2rh,
    "Cp2ZrCl2 (bent)": build_cp2zrcl2,
}
ORDER = list(FEED)


# ======================================================================== #
#  the SHARED render path (same for every species)
# ======================================================================== #
def _metal_components(H):
    m = H.metal
    comps = {}
    for n in H.adj[m]:
        seen, stack = {n}, [n]
        while stack:
            x = stack.pop()
            for y in H.adj[x]:
                if y == m or y in seen:
                    continue
                seen.add(y); stack.append(y)
        comps[n] = seen
    return comps


def _rotate_component(H, comp, piv, dth):
    c, s = math.cos(dth), math.sin(dth)
    px, py = piv
    for a in comp:
        if a in H.pinned:
            continue
        x, y = H.pos[a][0] - px, H.pos[a][1] - py
        H.pos[a] = (px + c * x - s * y, py + s * x + c * y)


def metal_fan(H):
    """Spread the metal's ligands around the centre, chelate-aware: a donor whose
    cut-component still contains ANOTHER donor (a rigid chelate, e.g. SPhos
    P + Pd...Cipso, or the Hoveyda alkylidene+ether) is ANCHORED; only the free
    monodentate ancillaries are rotated, into the largest open angular gaps."""
    m = H.metal
    if m is None:
        return
    mx, my = H.pos[m]
    comps = _metal_components(H)
    nbrs = list(comps)
    if len(nbrs) < 2:
        return

    def ang(n):
        return math.atan2(H.pos[n][1] - my, H.pos[n][0] - mx)

    anchored = [n for n in nbrs if any(o in comps[n] for o in nbrs if o != n)]
    free = [n for n in nbrs if n not in anchored]
    if not free:
        order = sorted(nbrs, key=ang)
        base = ang(order[0])
        for k, nb in enumerate(order):
            _rotate_component(H, comps[nb], (mx, my),
                              (base + 2 * math.pi * k / len(order)) - ang(nb))
        return
    occupied = sorted(ang(n) for n in anchored) or [ang(free[0])]
    gaps = []
    for i in range(len(occupied)):
        a0 = occupied[i]
        a1 = occupied[(i + 1) % len(occupied)] + (2 * math.pi if i == len(occupied) - 1 else 0)
        gaps.append((a1 - a0, a0, a1))
    gaps.sort(reverse=True)
    free_sorted = sorted(free, key=ang)
    for k, nb in enumerate(free_sorted):
        span, a0, a1 = gaps[k % len(gaps)]
        share = sum(1 for j in range(len(free_sorted)) if j % len(gaps) == k % len(gaps))
        idx = k // len(gaps)
        tgt = a0 + span * (idx + 1) / (share + 1)
        _rotate_component(H, comps[nb], (mx, my),
                          (tgt - ang(nb) + math.pi) % (2 * math.pi) - math.pi)


def _declutter_metal_leaves(H, clear=0.82):
    """Swing each monodentate metal ligand (a leaf atom bonded only to the metal:
    Br, Cl, H, ...) to the least-crowded direction WITHIN its angular gap between
    the two neighbouring metal bonds -- clears e.g. a Pd-Br sitting on a biaryl
    OMe, or a Grubbs Ru-Cl grazing a mesityl, without disturbing the chelate."""
    m = H.metal
    if m is None:
        return
    mx, my = H.pos[m]
    # only swing true terminal atoms (halide / hydride); leave abbreviated groups
    # like Ph or CO where the fan placed them, so a cis OA pair (Pd-Ph next to
    # Pd-Br) is not pulled apart.
    halide = {"Cl", "Br", "I", "F", "H", ""}
    leaves = [n for n in H.adj[m] if len(H.adj[n]) == 1 and n not in H.pinned
              and H.label.get(n) in halide]
    if not leaves:
        return
    others = [a for a in H.ids if a != m]

    def ang(a):
        return math.atan2(H.pos[a][1] - my, H.pos[a][0] - mx)

    for n in leaves:
        r = math.hypot(H.pos[n][0] - mx, H.pos[n][1] - my)
        nb_angs = sorted(((ang(o) - ang(n) + math.pi) % (2 * math.pi) - math.pi, o)
                         for o in H.adj[m] if o != n)
        lo = max((d for d, _ in nb_angs if d < 0), default=-math.pi)
        hi = min((d for d, _ in nb_angs if d > 0), default=math.pi)
        base = ang(n)

        def cost(a):
            px, py = mx + r * math.cos(a), my + r * math.sin(a)
            c = 0.0
            for o in others:
                if o == n:
                    continue
                d = math.hypot(H.pos[o][0] - px, H.pos[o][1] - py)
                if d < clear * L:
                    c += (clear * L - d) ** 2
            return c
        best, bestc = base, cost(base)
        steps = 26
        for k in range(steps + 1):
            da = lo + (hi - lo) * k / steps
            if abs(da) > math.radians(85):
                continue
            c = cost(base + da)
            if c < bestc - 1e-6:
                best, bestc = base + da, c
        H.pos[n] = (mx + r * math.cos(best), my + r * math.sin(best))


def _point_in_poly(p, poly):
    x, y = p
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _bfs_path(H, src, dst, avoid):
    from collections import deque
    prev = {src: None}
    q = deque([src])
    while q:
        x = q.popleft()
        if x == dst:
            break
        for y in H.adj[x]:
            if y == avoid or y in prev:
                continue
            prev[y] = x
            q.append(y)
    if dst not in prev:
        return None
    path, x = [], dst
    while x is not None:
        path.append(x)
        x = prev[x]
    return path[::-1]


def _ring_polys(H):
    """Every ring's ordered atom-id cycle: the aromatic / aliphatic rings PLUS the
    chelate metallacycles (metal -> donor -> backbone -> donor -> metal), which the
    ring finder skips because it never traverses the metal."""
    polys = [list(H._ring_cycle(r)) for r in H.rings if len(r) >= 3]
    m = H.metal
    if m is not None:
        comps = _metal_components(H)
        donors = list(H.adj[m])
        for i in range(len(donors)):
            for jj in range(i + 1, len(donors)):
                d1, d2 = donors[i], donors[jj]
                if d2 not in comps[d1]:
                    continue                          # not a chelate pair
                path = _bfs_path(H, d1, d2, avoid=m)
                if path and 2 <= len(path) <= 9:
                    polys.append([m] + path)
    return polys


def _atoms_inside_rings(H):
    """Atom ids that sit INSIDE a ring polygon they are not part of -- a ligand
    crammed into a chelate cavity or an aryl ring (the thing to push outside)."""
    bad = set()
    for cyc in _ring_polys(H):
        ringset = set(cyc)
        poly = [H.pos[a] for a in cyc]
        if len(poly) < 3:
            continue
        for a in H.ids:
            if a in ringset or a == H.metal:
                continue
            if _point_in_poly(H.pos[a], poly):
                bad.add(a)
    return bad


def _place_ancillaries_outside(H):
    """GENERAL RULE: when the metal sits IN a chelate ring (a multidentate ligand
    bridges back to it -- Pd-SPhos, the PNP pincer, salen, ...), its monodentate
    ancillary ligands must go OUTSIDE that ring, not into the chelate cavity.  Each
    ancillary's whole sub-tree is rotated rigidly about the metal into the exterior
    arc -- the direction from the chelate centroid out through the metal -- spread
    adjacently so e.g. a Pd-Ph / Pd-Br oxidative-addition pair stays cis."""
    m = H.metal
    if m is None or R._find_biaryl(H) is not None:    # biaryl OA handled separately
        return
    mx, my = H.pos[m]
    comps = _metal_components(H)
    nbrs = list(comps)
    chel = [n for n in nbrs if any(o in comps[n] for o in nbrs if o != n)]
    anc = [n for n in nbrs if n not in chel]
    if not chel or not anc:
        return
    inside = _atoms_inside_rings(H)
    if not any(comps[n] & inside for n in anc):       # nothing crammed in a ring
        return

    def ang(a):
        return math.atan2(H.pos[a][1] - my, H.pos[a][0] - mx)
    chel_atoms = set().union(*(comps[n] for n in chel))
    cxx = sum(H.pos[a][0] for a in chel_atoms) / len(chel_atoms)
    cyy = sum(H.pos[a][1] for a in chel_atoms) / len(chel_atoms)
    if math.hypot(mx - cxx, my - cyy) < 0.25 * L:     # ligand wraps the metal (salen)
        dang = sorted(ang(n) for n in chel)           # exterior = mid of largest gap
        best = (-1.0, 0.0)
        for i in range(len(dang)):
            a0 = dang[i]
            a1 = dang[(i + 1) % len(dang)] + (2 * math.pi if i == len(dang) - 1 else 0)
            if a1 - a0 > best[0]:
                best = (a1 - a0, (a0 + a1) / 2)
        ext = best[1]
    else:
        ext = math.atan2(my - cyy, mx - cxx)
    k = len(anc)
    arc = math.radians(min(150.0, 52.0 * k))
    for idx, n in enumerate(sorted(anc, key=ang)):
        tgt = ext if k == 1 else ext - arc / 2 + arc * (idx + 0.5) / k
        _rotate_component(H, comps[n], (mx, my),
                          (tgt - ang(n) + math.pi) % (2 * math.pi) - math.pi)


# ---- self-optimisation: weight schedules + relief recipes (the "options") ---- #
WSET = {
    "base": W,
    "hi": WHI,
    "ov": {"w_bond": 5.0, "w_angle": 1.1, "w_overlap": 34.0, "w_rigid": 67.0, "maxiter": 230},
    "ang": {"w_bond": 6.0, "w_angle": 3.4, "w_overlap": 15.0, "w_rigid": 67.0, "maxiter": 240},
    "loose": {"w_bond": 4.0, "w_angle": 1.0, "w_overlap": 12.0, "w_rigid": 55.0, "maxiter": 160},
}
# each recipe is a list of (weight-key, [relief ops]) stages -- run in order, so a
# recipe = a SCHEDULE of weights and constraints introduced at different times.
RECIPES = [
    ("baseline", [("base", ["fan", "outside"]), ("hi", []),
                  ("base", ["declutter", "swing", "uncross"]),
                  ("base", ["declutter", "swing", "uncross"])]),
    ("overlap-first", [("ov", ["fan"]), ("base", ["outside"]), ("hi", []),
                       ("ov", ["declutter", "uncross"]), ("base", ["uncross"])]),
    ("angle-late", [("loose", ["fan", "outside"]), ("base", ["declutter"]),
                    ("ang", ["uncross"]), ("base", ["declutter", "uncross"])]),
    ("relief-heavy", [("base", ["fan", "outside"]), ("base", ["declutter", "swing", "uncross"]),
                      ("ov", ["uncross"]), ("base", ["declutter", "swing", "uncross"])]),
    ("overlap-heavy", [("ov", ["fan", "outside"]), ("ov", ["declutter", "uncross"]),
                       ("ov", ["declutter", "swing", "uncross"]), ("base", ["uncross"])]),
]


def _apply_op(H, op):
    if op == "fan":
        if H.metal is not None and len(H.adj[H.metal]) >= 3:
            metal_fan(H)
            H.angles = H._angle_targets()
    elif op == "outside":
        _place_ancillaries_outside(H)
        H.angles = H._angle_targets()
    elif op == "declutter":
        R.declutter(H, passes=2)
    elif op == "swing":
        R.swing_off_backbone(H)
    elif op == "uncross":
        R.uncross(H)
    elif op == "tilt":
        R.tilt_relief(H)


def _run_recipe(H, recipe):
    if recipe is None:                                # the proven default pipeline
        return _relax_proven(H)
    for wkey, ops in recipe:
        relaxer_energy.relax(H, WSET[wkey])
        for op in ops:
            _apply_op(H, op)
        relaxer_energy.relax(H, WSET[wkey])
    before = H.metrics()["overlap"]
    snap = dict(H.pos)
    _declutter_metal_leaves(H)
    if H.metrics()["overlap"] > before:
        H.pos = snap
    return H


def _relax_proven(H):
    """The proven default: relax, chelate-aware fan + push ancillaries outside any
    chelate ring (guarded), then declutter/swing/uncross relief.  Tried first; the
    weight-schedule recipes are only needed when this still leaves a problem."""
    relaxer_energy.relax(H, W)
    if H.metal is not None and len(H.adj[H.metal]) >= 3:
        before = (H.metrics()["crossings"], H.metrics()["overlap"])
        snap = dict(H.pos)
        metal_fan(H)
        _place_ancillaries_outside(H)
        H.angles = H._angle_targets()
        relaxer_energy.relax(H, WHI)
        if (H.metrics()["crossings"], H.metrics()["overlap"]) > before:
            H.pos = snap
    for _ in range(2):
        R.declutter(H, passes=2)
        R.swing_off_backbone(H)
        R.uncross(H)
        relaxer_energy.relax(H, W)
    before = H.metrics()["overlap"]
    snap = dict(H.pos)
    _declutter_metal_leaves(H)
    if H.metrics()["overlap"] > before:
        H.pos = snap
    return H


def relax_complex(H, fan=True):
    """The single, frozen-core relax used for haptic metallocenes (no fan search)."""
    relaxer_energy.relax(H, W)
    for _ in range(2):
        R.declutter(H, passes=2)
        R.uncross(H)
        relaxer_energy.relax(H, W)
    return H.metrics()


def _resettle_after_tilt(H, name, ring):
    """RE-OPTIMISE after a rotation (general rule): freeze the rotated ring -- its
    new foreshortened shape becomes the rigid target -- and re-relax so the rest of
    the structure settles AROUND it (the phosphine, OMe arms and metal ligands stop
    colliding / sitting off-centre).  Rebuilds the Harness so the tilted ring's
    distances, not the original regular hexagon's, are what the spring solver
    preserves."""
    H.commit()
    H2 = R.Harness(scene=H.scene, title=name, metal=H.metal,
                   exempt=set(ring), extra_rigid=[set(ring)])
    H2.pinned = H2.pinned | set(ring)                 # lock the straightened, tilted
    relaxer_energy.relax(H2, WHI)                     # ring; settle everything around it
    R.declutter(H2, passes=2)
    R.uncross(H2)
    relaxer_energy.relax(H2, W)
    before = (H2.metrics()["crossings"], H2.metrics()["overlap"])
    snap = dict(H2.pos)
    _declutter_metal_leaves(H2)
    if (H2.metrics()["crossings"], H2.metrics()["overlap"]) > before:
        H2.pos = snap
    if H2.metrics()["crossings"] > 0:                 # never leave a bond crossing
        R.uncross(H2)
        relaxer_energy.relax(H2, W)
    return H2


def _group_oa_leaves(H):
    """Place the covalent monodentate ligands (an oxidative-addition pair such as
    Pd-Ph and Pd-Br) ADJACENT (cis), evenly inside the largest open gap between the
    dative donors -- so the OA fragment reads as a cis pair, not split across the
    metal.  Final cosmetic transform (rotates the single leaf atoms about the
    metal)."""
    m = H.metal
    if m is None:
        return
    mx, my = H.pos[m]
    cov = [n for n in H.adj[m] if len(H.adj[n]) == 1
           and H.kind.get(frozenset((m, n))) not in ("coord", "dative")]
    others = [n for n in H.adj[m] if n not in cov]
    if len(cov) < 2 or not others:
        return

    def ang(a):
        return math.atan2(H.pos[a][1] - my, H.pos[a][0] - mx)
    oa = sorted(ang(o) for o in others)
    best = (-1.0, 0.0)
    for i in range(len(oa)):
        a0 = oa[i]
        a1 = oa[(i + 1) % len(oa)] + (2 * math.pi if i == len(oa) - 1 else 0)
        if a1 - a0 > best[0]:
            best = (a1 - a0, a0)
    span, a0 = best
    leaves = sorted(cov, key=ang)
    rr = {n: math.hypot(H.pos[n][0] - mx, H.pos[n][1] - my) for n in leaves}
    # the donor substituents (e.g. P-cyclohexyls) eat into the gap, so the even
    # split can collide; try several adjacent placements and keep the cleanest.
    snap = dict(H.pos)
    best_pos, best_score = snap, (99, 99)
    for f0, f1 in ((1 / 3, 2 / 3), (0.45, 0.72), (0.55, 0.82),
                   (0.30, 0.55), (0.62, 0.88)):
        H.pos = dict(snap)
        for fr, n in zip((f0, f1), leaves):
            tgt = a0 + span * fr
            H.pos[n] = (mx + rr[n] * math.cos(tgt), my + rr[n] * math.sin(tgt))
        m_ = H.metrics()
        score = (m_["crossings"], m_["overlap"])
        if score < best_score:
            best_score, best_pos = score, dict(H.pos)
    H.pos = best_pos


def _orient_labels(H):
    """Re-orient abbreviation labels to the FINAL geometry so each coordinating /
    attachment atom faces its parent bond (M-CO drawn OC when M is to the right,
    aryl-OMe drawn MeO when the ring is to the right, ...)."""
    for i in H.ids:
        forms = S.ABBR_FORMS.get(H.label.get(i))
        if not forms or not H.adj[i]:
            continue
        xi, yi = H.pos[i]
        parent = min(H.adj[i], key=lambda p: math.hypot(H.pos[p][0] - xi,
                                                        H.pos[p][1] - yi))
        new = S._orient_abbr(H.label[i], H.pos[parent], H.pos[i])
        H.label[i] = new
        H.scene.atoms[i].label = new


def _score(H):
    """Problem score, lower = better: bond CROSSINGS first, then OVERLAP count plus
    the number of atoms crammed INSIDE a ring, then the numerical quality_loss.
    Any non-zero count is a PROBLEM that keeps the optimiser trying more options."""
    m = H.metrics()
    inside = len(_atoms_inside_rings(H))
    return (m["crossings"], m["overlap"] + inside, round(R.quality_loss(H), 3))


def _build_one(name, mol, recipe):
    sc = S.scene_from_mol(mol, name)
    meta = getattr(sc, "meta", None) or {}
    H = R.Harness(scene=sc, title=name, metal=meta.get("metal"),
                  exempt=meta.get("exempt") or None,
                  extra_rigid=meta.get("extra_rigid") or None)
    _run_recipe(H, recipe)
    ring = R.biaryl_into_plane(H)                 # no-op unless a Pd...Cipso biaryl
    if ring:
        H = _resettle_after_tilt(H, name, ring)
        R.straighten_biaryl(H)                    # C1/C4 on the biaryl axis
        _group_oa_leaves(H)                       # cis OA pair, outside, collision-free
    return H


def render(name, mol):
    """Self-optimising render: a haptic metallocene uses the single frozen-core
    relax; everything else is run through a SEARCH over weight-schedule / relief
    recipes -- as long as a result still has a crossing or an overlap (a PROBLEM)
    the optimiser keeps trying more options, keeping the best by the judge and
    stopping the moment a clean one (0 crossings, 0 overlaps) is found."""
    sc0 = S.scene_from_mol(mol, name)
    meta0 = getattr(sc0, "meta", None) or {}
    if meta0.get("extra_rigid"):                  # haptic metallocene -> frozen core
        H = R.Harness(scene=sc0, title=name, metal=meta0.get("metal"),
                      exempt=meta0.get("exempt"), extra_rigid=meta0.get("extra_rigid"))
        relax_complex(H, fan=False)
        _orient_labels(H)
        return H
    best, bests = None, (99, 99, 1e18)
    for rname, recipe in [("proven", None)] + RECIPES:   # proven first, then search
        H = _build_one(name, mol, recipe)
        s = _score(H)
        if s < bests:
            bests, best = s, H
        if s[0] == 0 and s[1] == 0:               # clean -> no problem left to fix
            break
    _orient_labels(best)
    return best


# ======================================================================== #
def main():
    outdir = HERE / "out" / "complexes"
    outdir.mkdir(parents=True, exist_ok=True)
    Hs = {nm: render(nm, FEED[nm]()) for nm in ORDER}
    svgs, pngs, label = [], [], {}
    for nm in ORDER:
        H = Hs[nm]
        H.commit()
        m = H.metrics()
        print(f"{nm:32s} overlap={m['overlap']} crossings={m['crossings']} "
              f"angleDev={m['angleDevDeg']:.1f} crowd={m['crowd']:.2f}")
        svg = H.scene.render_svg(*CANVAS, title=nm, fixed_scale=SCALE,
                                 anchor_world=H.pos[H.metal], anchor_px=ANCHOR)
        sp = outdir / f"{_safe(nm)}.svg"
        sp.write_text(svg)
        svgs.append(str(sp)); pngs.append(str(outdir / f"{_safe(nm)}.png"))
        label[nm] = str(outdir / f"{_safe(nm)}.png")
    R._svg_to_png(svgs, pngs)
    cropdir = HERE.parent / "panels" / "complexes_cropped"
    cropdir.mkdir(parents=True, exist_ok=True)
    csvgs, cpngs = [], []
    for nm in ORDER:
        H = Hs[nm]
        H.commit()
        svg = H.scene.render_svg(*CANVAS, title=None, fixed_scale=SCALE,
                                 anchor_world=H.pos[H.metal], anchor_px=ANCHOR)
        sp = cropdir / f"{_safe(nm)}.svg"
        sp.write_text(svg)
        csvgs.append(str(sp)); cpngs.append(str(cropdir / f"{_safe(nm)}.png"))
    R._svg_to_png(csvgs, cpngs)
    from PIL import Image, ImageChops
    for p in cpngs:
        im = Image.open(p).convert("RGB")
        bg = Image.new("RGB", im.size, "white")
        bb = ImageChops.difference(im, bg).getbbox()
        if bb:
            x0, y0, x1, y1 = bb
            im = im.crop((max(0, x0 - 28), max(0, y0 - 28),
                          min(im.width, x1 + 28), min(im.height, y1 + 28)))
        im.save(p)
    _montage([label[n] for n in ORDER], 4,
             HERE.parent / "panels" / "complexes_gallery.png")
    print(f"cropped per-complex PNGs -> {cropdir}")
    print("done")


def _montage(pngs, cols, dest):
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    rows = (len(imgs) + cols - 1) // cols
    w = max(i.width for i in imgs); h = max(i.height for i in imgs)
    g = Image.new("RGB", (w * cols, h * rows), "white")
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        g.paste(im, (c * w, r * h))
    g.save(dest)
    print("montage ->", dest)


if __name__ == "__main__":
    main()
