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
    leaves = [n for n in H.adj[m] if len(H.adj[n]) == 1 and n not in H.pinned]
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


def relax_complex(H, fan=True):
    relaxer_energy.relax(H, W)
    if fan and H.metal is not None and len(H.adj[H.metal]) >= 3:
        before = (H.metrics()["crossings"], H.metrics()["overlap"])
        snap = dict(H.pos)
        metal_fan(H)
        relaxer_energy.relax(H, WHI)
        after = (H.metrics()["crossings"], H.metrics()["overlap"])
        if after > before:
            H.pos = snap
    for _ in range(2):
        R.declutter(H, passes=2)
        R.swing_off_backbone(H)
        R.uncross(H)
        relaxer_energy.relax(H, W)
    if fan:                                       # not on a frozen metallocene core
        before = H.metrics()["overlap"]
        snap = dict(H.pos)
        _declutter_metal_leaves(H)
        if H.metrics()["overlap"] > before:
            H.pos = snap
    return H.metrics()


def render(name, mol):
    """scene_from_mol (eta-n + coord perception) -> Harness -> relax/relief ->
    biaryl-into-plane tilt.  Returns the relaxed Harness."""
    sc = S.scene_from_mol(mol, name)
    meta = getattr(sc, "meta", None) or {}
    haptic = bool(meta.get("extra_rigid"))
    H = R.Harness(scene=sc, title=name, metal=meta.get("metal"),
                  exempt=meta.get("exempt") or None,
                  extra_rigid=meta.get("extra_rigid") or None)
    relax_complex(H, fan=not haptic)
    if not haptic:
        R.biaryl_into_plane(H)                    # no-op unless a Pd...Cipso biaryl
    return H


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
