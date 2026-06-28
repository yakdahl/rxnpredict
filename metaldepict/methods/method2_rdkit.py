#!/usr/bin/env python3
"""METHOD 2 -- RDKit-native depiction with condensed abbreviations.

A completely different approach from the interactive rigid-body viewer: let
RDKit / CoordGen lay out and draw the complex, with standard ABBREVIATIONS
(t-Bu, OMe, Ph ...) condensed by rdAbbreviations.  Produces a static
publication ("ChemDraw"-style) image per ligand.

MEDIUM abbreviation level == collapse terminal groups (t-Bu, OMe, and a
monosubstituted Ph) to PLAIN TEXT labels, keep the backbone rings drawn.
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdAbbreviations, rdDepictor  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402
from rdkit.Geometry import Point3D  # noqa: E402

from gen_panel import LIGANDS  # noqa: E402
from src import chem  # noqa: E402

# ------------------------------------------------------------------ #
# CUSTOM "MEDIUM" abbreviations.
#   label  SMARTS  displayLabel  displayLabelW
# first SMARTS atom = attachment dummy, remainder = the collapsed group.
# Ph is a *monosubstituted* benzene (so DTBM's substituted aryls are NOT
# collapsed -- their rings stay drawn, only their t-Bu / OMe tips collapse).
# ------------------------------------------------------------------ #
MEDIUM_ABBREVS = """tBu [*]C(C)(C)C tBu tBu
OMe [*]OC OMe MeO
Ph [*]c1ccccc1 Ph Ph
"""


def medium_abbreviations():
    return rdAbbreviations.ParseAbbreviations(MEDIUM_ABBREVS, True, False)


def _find_cu_h(mol):
    cu = h = None
    for a in mol.GetAtoms():
        if a.GetSymbol() == "Cu":
            cu = a.GetIdx()
        elif a.GetSymbol() == "H" and a.GetDegree() == 1:
            for nb in a.GetNeighbors():
                if nb.GetSymbol() == "Cu":
                    h = a.GetIdx()
    return cu, h


def _mean_bond_len(mol, conf):
    tot = n = 0.0
    for b in mol.GetBonds():
        p = conf.GetAtomPosition(b.GetBeginAtomIdx())
        q = conf.GetAtomPosition(b.GetEndAtomIdx())
        tot += math.hypot(p.x - q.x, p.y - q.y)
        n += 1
    return tot / n if n else 1.0


def _orient_cu_h_right(mol):
    """Rotate the 2-D conformer so the Cu-H vector points horizontally to the
    RIGHT, then place H at exactly one standard bond length right of Cu so the
    Cu-H bond is always clean and uncrowded."""
    cu, h = _find_cu_h(mol)
    if cu is None or h is None:
        return
    conf = mol.GetConformer()
    pcu = conf.GetAtomPosition(cu)
    ph = conf.GetAtomPosition(h)
    dx, dy = ph.x - pcu.x, ph.y - pcu.y
    ang = math.atan2(dy, dx)          # current angle of Cu->H
    theta = -ang                       # rotate to 0 rad (point +x)
    c, s = math.cos(theta), math.sin(theta)
    cx0, cy0 = pcu.x, pcu.y
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        x, y = p.x - cx0, p.y - cy0
        conf.SetAtomPosition(i, Point3D(c * x - s * y + cx0,
                                        s * x + c * y + cy0, 0.0))
    # Pin the hydride one standard bond length straight to the RIGHT of Cu.
    bl = _mean_bond_len(mol, conf)
    pcu = conf.GetAtomPosition(cu)
    conf.SetAtomPosition(h, Point3D(pcu.x + bl, pcu.y, 0.0))


def _close_pairs(mol, conf, bl, thresh=0.66):
    """Count non-bonded atom pairs closer than thresh*bondlen (proxy for visual
    crowding / overlapping labels)."""
    bonded = set()
    for b in mol.GetBonds():
        bonded.add((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        bonded.add((b.GetEndAtomIdx(), b.GetBeginAtomIdx()))
    n = mol.GetNumAtoms()
    cnt = 0
    for i in range(n):
        pi = conf.GetAtomPosition(i)
        for j in range(i + 1, n):
            if (i, j) in bonded:
                continue
            pj = conf.GetAtomPosition(j)
            if math.hypot(pi.x - pj.x, pi.y - pj.y) < thresh * bl:
                cnt += 1
    return cnt


def _mirror_x(mol):
    """Reflect the conformer about the x-axis (y -> -y).  Keeps the horizontal
    Cu-H direction; flips top/bottom, which can separate crowded substituents."""
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(p.x, -p.y, 0.0))


def _is_label_atom(mol, idx):
    """True for atoms rendered as a multi-character text label (abbreviation
    dummies, heteroatoms, the metal) which need extra clearance because their
    drawn glyph is wider than a bond vertex."""
    a = mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum() == 0:           # abbreviation surrogate (tBu/OMe/Ph)
        return True
    return a.GetSymbol() in ("O", "P", "Cu", "H")


def _relax_overlaps(mol, iters=200, step=0.12):
    """Restorative 2-D nudge that moves ONLY the abbreviation-label points
    (tBu / OMe / Ph dummy atoms).  Every backbone / ring atom is frozen, so ring
    geometry is never distorted.  Each label atom is repelled by nearby atoms,
    then snapped back to its standard bond length from its single anchor -- i.e.
    it swings about the anchor to find a clear angle and never collides."""
    conf = mol.GetConformer()
    bl = _mean_bond_len(mol, conf)
    n = mol.GetNumAtoms()
    bonded = set()
    for b in mol.GetBonds():
        bonded.add((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        bonded.add((b.GetEndAtomIdx(), b.GetBeginAtomIdx()))
    # only the abbreviation surrogate dummies are mobile
    movable = []
    anchor = {}
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 0:
            nbrs = list(a.GetNeighbors())
            if len(nbrs) == 1:
                movable.append(a.GetIdx())
                anchor[a.GetIdx()] = nbrs[0].GetIdx()
    if not movable:
        return
    for _ in range(iters):
        disp = {}
        for i in movable:
            pi = conf.GetAtomPosition(i)
            fx = fy = 0.0
            for j in range(n):
                if j == i or (i, j) in bonded:
                    continue
                pj = conf.GetAtomPosition(j)
                dx, dy = pi.x - pj.x, pi.y - pj.y
                d = math.hypot(dx, dy) or 1e-6
                # wide text labels need ~1.4 bl of clearance from each other
                other_label = mol.GetAtomWithIdx(j).GetAtomicNum() == 0
                rep = bl * (1.55 if other_label else 1.05)
                if d < rep:
                    f = (rep - d) / rep
                    fx += f * dx / d
                    fy += f * dy / d
            disp[i] = (fx, fy)
        for i, (fx, fy) in disp.items():
            p = conf.GetAtomPosition(i)
            nx, ny = p.x + step * bl * fx, p.y + step * bl * fy
            # snap back to standard bond length from the anchor (swing, don't stretch)
            pa = conf.GetAtomPosition(anchor[i])
            vx, vy = nx - pa.x, ny - pa.y
            d = math.hypot(vx, vy) or 1e-6
            conf.SetAtomPosition(i, Point3D(pa.x + bl * vx / d,
                                            pa.y + bl * vy / d, 0.0))


def _label_clashes(mol, conf, bl):
    """Count clashes using label-aware clearance (wide text glyphs need room)."""
    n = mol.GetNumAtoms()
    is_label = [_is_label_atom(mol, i) for i in range(n)]
    bonded = set()
    for b in mol.GetBonds():
        bonded.add((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        bonded.add((b.GetEndAtomIdx(), b.GetBeginAtomIdx()))
    cnt = 0
    for i in range(n):
        pi = conf.GetAtomPosition(i)
        for j in range(i + 1, n):
            if (i, j) in bonded:
                continue
            pj = conf.GetAtomPosition(j)
            d = math.hypot(pi.x - pj.x, pi.y - pj.y)
            rep = bl * (1.30 if (is_label[i] and is_label[j])
                        else 1.00 if (is_label[i] or is_label[j]) else 0.66)
            if d < rep:
                cnt += 1
    return cnt


def _p_aryl_rings(mol):
    """For each P-bound aromatic 6-ring, return (ipso, p, ring_plus_dummies)."""
    ri = mol.GetRingInfo()
    arom6 = [list(r) for r in ri.AtomRings()
             if len(r) == 6 and all(mol.GetAtomWithIdx(a).GetIsAromatic()
                                    for a in r)]
    out = []
    for ring in arom6:
        ipso = p = None
        for a in ring:
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                if nb.GetSymbol() == "P":
                    ipso, p = a, nb.GetIdx()
        if ipso is None:
            continue
        # collect ring atoms + everything hanging off the ring (the dummies)
        members = set(ring)
        for a in ring:
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                if nb.GetIdx() != p and nb.GetIdx() not in ring:
                    members.add(nb.GetIdx())
        out.append((ipso, p, members))
    return out


def _flip_group(mol, conf, ipso, p, members):
    """Reflect a substituent across the P->ipso axis (swap which meta points in)."""
    pi = conf.GetAtomPosition(ipso)
    pp = conf.GetAtomPosition(p)
    ax, ay = pi.x - pp.x, pi.y - pp.y
    L = math.hypot(ax, ay) or 1e-6
    ax, ay = ax / L, ay / L            # unit axis through ipso
    for idx in members:
        q = conf.GetAtomPosition(idx)
        vx, vy = q.x - pi.x, q.y - pi.y
        dot = vx * ax + vy * ay        # component along axis
        # reflection about the line through ipso with direction (ax,ay)
        rx = 2 * dot * ax - vx
        ry = 2 * dot * ay - vy
        conf.SetAtomPosition(idx, Point3D(pi.x + rx, pi.y + ry, 0.0))


def _rotate_group(mol, conf, pivot, members, theta):
    """Rotate `members` by theta about the in-plane `pivot` point."""
    c, s = math.cos(theta), math.sin(theta)
    px, py = pivot.x, pivot.y
    for idx in members:
        q = conf.GetAtomPosition(idx)
        x, y = q.x - px, q.y - py
        conf.SetAtomPosition(idx, Point3D(c * x - s * y + px,
                                          s * x + c * y + py, 0.0))


def _spin_aryls(mol):
    """Greedily re-orient each P-aryl ring -- try a fan of gentle rotations about
    the P->ipso pivot, keeping the one that lowers the (wide) label-clash count
    WITHOUT introducing any new tight geometric overlap (ring crossing a ring).
    This splays inward-pointing t-Bu groups out of each other's way."""
    conf = mol.GetConformer()
    bl = _mean_bond_len(mol, conf)
    angles = [math.radians(a) for a in
              (20, -20, 30, -30, 40, -40, 15, -15, 50, -50)]
    for ipso, p, members in _p_aryl_rings(mol):
        before = _label_clashes(mol, conf, bl)
        if before == 0:
            continue
        tight0 = _close_pairs(mol, conf, bl, thresh=0.80)
        pivot = conf.GetAtomPosition(ipso)
        best, best_theta = before, 0.0
        for th in angles:
            _rotate_group(mol, conf, pivot, members, th)
            lc = _label_clashes(mol, conf, bl)
            tight = _close_pairs(mol, conf, bl, thresh=0.80)
            if lc < best and tight <= tight0:
                best, best_theta = lc, th
            _rotate_group(mol, conf, pivot, members, -th)  # undo, try next
        if best_theta:
            _rotate_group(mol, conf, pivot, members, best_theta)


def _decrowd(mol):
    """De-crowd: choose the better mirror state, flip clashing P-aryl rings, then
    swing remaining abbreviation labels apart -- ring geometry stays rigid."""
    conf = mol.GetConformer()
    bl = _mean_bond_len(mol, conf)
    before = _label_clashes(mol, conf, bl)
    if before:
        _mirror_x(mol)
        if _label_clashes(mol, conf, bl) >= before:
            _mirror_x(mol)               # revert -- mirror did not help
        _spin_aryls(mol)
        if _label_clashes(mol, conf, bl):
            _relax_overlaps(mol)


def _strip_metal_charges(mol):
    for a in mol.GetAtoms():
        if a.GetSymbol() in ("Cu", "H"):
            a.SetFormalCharge(0)
            a.SetNoImplicit(True)


# Stereo-defined SMILES override (so wedges/dashes render at sp3 stereocentres).
# (R,R)-Ph-BPE: 1,2-bis[(2R,5R)-2,5-diphenylphospholano]ethane.
SMILES_OVERRIDE = {
    "ph_bpe": "C(C[P@]1[C@@H](c2ccccc2)CC[C@@H]1c1ccccc1)"
              "[P@]1[C@@H](c2ccccc2)CC[C@@H]1c1ccccc1",
}


def _find_biaryl_bond(mol):
    """Single bond joining two distinct aromatic 6-rings (the atropisomeric
    biaryl axis of DTBM-SEGPhos)."""
    ri = mol.GetRingInfo()
    arom6 = [set(r) for r in ri.AtomRings()
             if len(r) == 6 and all(mol.GetAtomWithIdx(a).GetIsAromatic()
                                    for a in r)]
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.GetIsAromatic():
            continue
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ra = next((r for r in arom6 if a1 in r), None)
        rb = next((r for r in arom6 if a2 in r), None)
        if ra is not None and rb is not None and ra is not rb:
            if not (mol.GetAtomWithIdx(a1).IsInRingSize(5)
                    or mol.GetAtomWithIdx(a2).IsInRingSize(5)):
                return b.GetIdx()
    return None


def _chemdraw_options(d):
    o = d.drawOptions()
    o.addStereoAnnotation = False
    o.bondLineWidth = 3
    o.padding = 0.10
    o.fixedBondLength = 38
    o.useBWAtomPalette()           # monochrome, ACS/ChemDraw-like
    o.annotationFontScale = 0.7
    return o


def draw(key, name, smi, out, size=(840, 760)):
    smi = SMILES_OVERRIDE.get(key, smi)
    lig = chem.make_dtbm_segphos() if smi is None else Chem.MolFromSmiles(smi)
    Chem.SanitizeMol(lig)
    cx, info = chem.build_complex(lig)

    abbr = medium_abbreviations()
    try:
        cx = rdAbbreviations.CondenseMolAbbreviations(cx, abbr, maxCoverage=1.0)
    except Exception as exc:
        print("  abbrev skip:", exc)

    _strip_metal_charges(cx)
    Chem.AssignStereochemistry(cx, cleanIt=True, force=True)
    # Classic (template-based) depictor draws PERFECT regular ring polygons --
    # CoordGen subtly distorts fused backbones (e.g. xanthene).
    rdDepictor.SetPreferCoordGen(False)
    rdDepictor.Compute2DCoords(cx)
    _orient_cu_h_right(cx)
    _decrowd(cx)
    _orient_cu_h_right(cx)             # re-pin Cu-H horizontal after relaxation

    d = rdMolDraw2D.MolDraw2DCairo(*size)
    _chemdraw_options(d)

    # Prepare (kekulize + wedge sp3 stereocentres) but keep our coordinates.
    drawn = rdMolDraw2D.PrepareMolForDrawing(cx, kekulize=True,
                                             addChiralHs=False, wedgeBonds=True)
    # Bold the atropisomeric biaryl axis (DTBM-SEGPhos) via a black highlight.
    hl_bonds = []
    hl_cols = {}
    biaryl = _find_biaryl_bond(drawn)
    if biaryl is not None:
        hl_bonds = [biaryl]
        hl_cols = {biaryl: (0, 0, 0)}
        d.drawOptions().highlightBondWidthMultiplier = 3
    d.DrawMolecule(drawn, legend=name,
                   highlightAtoms=[], highlightBonds=hl_bonds,
                   highlightAtomColors={}, highlightBondColors=hl_cols)
    d.FinishDrawing()
    Path(out).write_bytes(d.GetDrawingText())
    print("  wrote", out)


if __name__ == "__main__":
    for key, (name, smi) in LIGANDS.items():
        print("[m2]", key)
        draw(key, name, smi, f"/tmp/wf_B_{key}.png")
