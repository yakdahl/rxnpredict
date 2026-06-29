#!/usr/bin/env python3
"""
seed_from_smiles.py -- build a ChemDraw-style template SEED (a template_draw
Scene) for ANY bidentate bisphosphine, straight from a SMILES, with no
hardcoded backbone.  The Scene then feeds the same physics relaxer
(relax_harness + relaxer_energy) as the hand-built templates, so the pipeline is
fully dynamic: a new ligand needs only its SMILES.

Pipeline
--------
  SMILES -> RDKit ligand -> (chem.build_complex) Cu-H complex with dative P->Cu
         -> condense t-Bu / OMe / Ph to labelled superatoms (medium abbreviation)
         -> RDKit 2-D coordinates (classic template depictor = regular polygons)
         -> scale so the median bond = L, orient the Cu-H bond horizontal (H right)
         -> emit a template_draw.Scene (labels, colours, Kekule doubles inside
            rings, dashed P->Cu coordination bonds, bold biaryl axis, stereo
            wedges where RDKit assigns them).

Because the connectivity comes from RDKit, substitution patterns are always
correct (e.g. SEGPhos: the phosphines are ortho to the biaryl bond).

Ferrocene-based ligands (Fe with two Cp rings, e.g. DPPF / Josiphos) are drawn
with a HARDCODED ferrocene fragment (two stacked Cp pentagons + Fe); overlaps
inside that fragment are ignored, per design.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from rdkit import Chem                                  # noqa: E402
from rdkit.Chem import rdAbbreviations, rdDepictor      # noqa: E402

from template_draw import Scene, L                       # noqa: E402
from src import chem                                     # noqa: E402

P_COL, CU_COL, O_COL, FE_COL = "#c87a00", "#b05a2a", "#c0392b", "#b05a2a"
COLORS = {"P": P_COL, "Cu": CU_COL, "O": O_COL, "Fe": FE_COL,
          "N": "#2c3e9e", "S": "#b8860b", "B": "#1f8a70", "Si": "#555"}
# atoms drawn as a glyph (heteroatoms + metal + hydride); carbon stays blank
LABEL_SYMBOLS = {"P", "O", "N", "S", "B", "Si", "Cu", "Fe", "H", "Cl", "Br", "F"}

_ABBR = rdAbbreviations.ParseAbbreviations(
    "tBu [*]C(C)(C)C tBu tBu\nOMe [*]OC OMe MeO\nPh [*]c1ccccc1 Ph Ph\n",
    True, False)


# --------------------------------------------------------------------------- #
def _complex(smiles):
    lig = Chem.MolFromSmiles(smiles)
    if lig is None:
        raise ValueError(f"bad SMILES: {smiles}")
    Chem.SanitizeMol(lig)
    cx, info = chem.build_complex(lig)
    return cx, info


_LABEL_FIX = {"tBu": "t-Bu"}


def _label_of(atom):
    d = atom.GetPropsAsDict()
    if atom.GetAtomicNum() == 0:                 # abbreviation superatom
        lbl = d.get("atomLabel") or d.get("_displayLabel") or "R"
        return _LABEL_FIX.get(lbl, lbl)
    s = atom.GetSymbol()
    return s if s in LABEL_SYMBOLS else ""


def _median_bond(conf, mol):
    ds = []
    for b in mol.GetBonds():
        p = conf.GetAtomPosition(b.GetBeginAtomIdx())
        q = conf.GetAtomPosition(b.GetEndAtomIdx())
        ds.append(math.hypot(p.x - q.x, p.y - q.y))
    ds.sort()
    return ds[len(ds) // 2] if ds else 1.0


def scene_from_smiles(smiles, name=None, abbreviate=True):
    cx, _ = _complex(smiles)
    if any(a.GetSymbol() == "Fe" for a in cx.GetAtoms()):
        return _ferrocene_scene(cx, name)

    if abbreviate:
        try:
            cx = rdAbbreviations.CondenseMolAbbreviations(cx, _ABBR, maxCoverage=1.0)
        except Exception:
            pass
    Chem.SanitizeMol(cx)
    rdDepictor.SetPreferCoordGen(False)              # regular polygons
    rdDepictor.Compute2DCoords(cx)
    try:
        Chem.WedgeMolBonds(cx, cx.GetConformer())    # stereo wedges
    except Exception:
        pass
    kek = Chem.Mol(cx)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)  # explicit single/double
    except Exception:
        pass

    conf = cx.GetConformer()
    scale = L / _median_bond(conf, cx)
    pos = {a.GetIdx(): (conf.GetAtomPosition(a.GetIdx()).x * scale,
                        conf.GetAtomPosition(a.GetIdx()).y * scale)
           for a in cx.GetAtoms()}

    cu = next(a.GetIdx() for a in cx.GetAtoms() if a.GetSymbol() == "Cu")
    hyd = next((nb.GetIdx() for nb in cx.GetAtomWithIdx(cu).GetNeighbors()
                if nb.GetSymbol() == "H"), None)
    _orient_cuh(pos, cu, hyd)

    return _build_scene(cx, kek, pos), name or "molecule"


def _orient_cuh(pos, cu, hyd):
    """Rotate every atom so Cu->H points along +x (H to the right of Cu)."""
    if hyd is None:
        return
    cx_, cy_ = pos[cu]
    hx, hy = pos[hyd]
    th = -math.atan2(hy - cy_, hx - cx_)
    c, s = math.cos(th), math.sin(th)
    for i, (x, y) in list(pos.items()):
        x0, y0 = x - cx_, y - cy_
        pos[i] = (cx_ + c * x0 - s * y0, cy_ + s * x0 + c * y0)


def _ring_centroid(mol, ring, pos):
    xs = [pos[i][0] for i in ring]
    ys = [pos[i][1] for i in ring]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _build_scene(cx, kek, pos):
    sc = Scene()
    ri = cx.GetRingInfo()
    arom_rings = [set(r) for r in ri.AtomRings()
                  if all(cx.GetAtomWithIdx(a).GetIsAromatic() for a in r)]
    in_ring_bond = set()
    for r in ri.BondRings():
        in_ring_bond.update(r)

    idmap = {}
    for a in cx.GetAtoms():
        lbl = _label_of(a)
        sym = a.GetSymbol()
        col = COLORS.get(sym, "#111")
        fs = 0.92 if (a.GetAtomicNum() == 0 and len(lbl) > 2) else 1.0
        idmap[a.GetIdx()] = sc.atom(pos[a.GetIdx()], label=lbl, color=col,
                                    fontscale=fs)

    for b in cx.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        si, sj = cx.GetAtomWithIdx(i).GetSymbol(), cx.GetAtomWithIdx(j).GetSymbol()
        a, bb = idmap[i], idmap[j]
        # coordination bond P->Cu (or any X->metal) -> dashed
        if {si, sj} & {"Cu", "Fe"} and ({si, sj} - {"Cu", "Fe"}):
            if "Cu" in (si, sj):
                sc.bond(a, bb, order=1, kind="coord")
                continue
        # bold biaryl axis: single bond joining two different aromatic rings,
        # itself not in a ring
        kb = kek.GetBondBetweenAtoms(i, j)
        order = int(kb.GetBondTypeAsDouble()) if kb else 1
        if order == 1 and b.GetIdx() not in in_ring_bond:
            ra = next((r for r in arom_rings if i in r), None)
            rj = next((r for r in arom_rings if j in r), None)
            if ra is not None and rj is not None and ra is not rj:
                sc.bond(a, bb, order=1, kind="bold")
                continue
        # stereo wedge / dash from RDKit bond direction
        bd = b.GetBondDir()
        if bd == Chem.BondDir.BEGINWEDGE:
            sc.bond(a, bb, order=1, kind="wedge")
            continue
        if bd == Chem.BondDir.BEGINDASH:
            sc.bond(a, bb, order=1, kind="dash")
            continue
        # ordinary single / double (double drawn inside its ring)
        inside = None
        if order == 2:
            r = next((rr for rr in arom_rings if i in rr and j in rr), None)
            if r:
                inside = _ring_centroid(cx, r, pos)
        sc.bond(a, bb, order=order, inside=inside)
    return sc


# --------------------------------------------------------------------------- #
# Ferrocene: HARDCODED fragment (two stacked Cp pentagons + Fe). RDKit cannot
# lay out an eta5 metallocene, so a 1,1'-bis(phosphino)ferrocene is built by
# template.  Overlaps INSIDE the fragment are ignored and the whole sandwich is
# one frozen rigid body (per design).
# --------------------------------------------------------------------------- #
import method3_template as _MT                            # noqa: E402
from template_draw import polar                            # noqa: E402

_PEN_R = L / (2 * math.sin(math.pi / 5))
_APOTHEM = _PEN_R * math.cos(math.pi / 5)


def _cp_ring(sc, center, apex_deg, kek=(2, 1, 2, 1, 1)):
    """Cyclopentadienyl pentagon with its apex vertex at `apex_deg` (so the
    OPPOSITE edge -- the one facing the Fe between the rings -- is flat)."""
    ids = [sc.atom(polar(center, apex_deg + k * 72.0, _PEN_R)) for k in range(5)]
    for k in range(5):
        sc.bond(ids[k], ids[(k + 1) % 5], order=kek[k], inside=center)
    return ids


def _ferrocene_stack(sc, fe_xy=(1.6, 0.0), gap=1.2):
    """Draw a vertical SANDWICH: upper Cp (apex up) above Fe, lower Cp (apex
    down) below Fe, with Fe labelled in the centre and eta5 lines to the two
    facing carbons of each ring.  Returns (fe, upper_ids, lower_ids)."""
    fx, fy = fe_xy
    fe = sc.atom((fx, fy), label="Fe", color=FE_COL)
    up = _cp_ring(sc, (fx, fy + gap), apex_deg=90.0)     # apex up; flat edge down
    dn = _cp_ring(sc, (fx, fy - gap), apex_deg=270.0)    # apex down; flat edge up
    for ids in (up, dn):                                  # eta5: lines to facing edge
        near = sorted(ids, key=lambda i: math.hypot(
            sc.atoms[i].pos[0] - fx, sc.atoms[i].pos[1] - fy))[:2]
        for i in near:
            sc.bond(fe, i, order=1)
    return fe, up, dn


def _cp_right_vertex(sc, ids, want_up):
    """The Cp ring carbon on the metal (right) side, upper or lower half."""
    cand = [i for i in ids if (sc.atoms[i].pos[1] > 0) == want_up] or ids
    return max(cand, key=lambda i: sc.atoms[i].pos[0])


def build_ferrocene_bisphosphine(name, psub=None):
    """1,1'-bis(phosphino)ferrocene Cu-H (e.g. DPPF), hardcoded sandwich.  psub
    places the two P substituents per P (default = two Ph each)."""
    psub = psub or _MT._pphos_phenyls
    sc = Scene()
    cu, pu, pd = (5.4, 0.0), (3.5, 1.45), (3.5, -1.45)
    c = _MT.core(sc, cu, pu, pd)                          # Cu,H,P's + dashed P-Cu
    fe, up, dn = _ferrocene_stack(sc, fe_xy=(1.6, 0.0), gap=1.25)
    sc.bond(c["pu"], _cp_right_vertex(sc, up, True), order=1)   # P on upper Cp
    sc.bond(c["pd"], _cp_right_vertex(sc, dn, False), order=1)  # P on lower Cp
    psub(sc, c)
    frozen = {fe, *up, *dn}
    return sc, name, {"exempt": frozen, "extra_rigid": [frozen]}


def build_josiphos(name="josiphos", psub_p1=None, psub_p2=None):
    """Josiphos: 1,2-disubstituted ferrocene -- one Cp carbon bears P1 directly,
    the ADJACENT carbon bears a CH(CH3) tether to P2.  Both P's chelate Cu.
    Default: P1 = PPh2, P2 = PCy2 (drawn as 'Cy' labels) -- the canonical motif.
    Both substituents sit on the UPPER Cp; the lower Cp is unsubstituted."""
    psub_p1 = psub_p1 or (lambda s, pid: (_arm(s, pid, 150, "Ph"),
                                          _arm(s, pid, 210, "Ph")))
    psub_p2 = psub_p2 or (lambda s, pid: (_arm(s, pid, -35, "Cy"),
                                          _arm(s, pid, -95, "Cy")))
    sc = Scene()
    cu = sc.atom((5.4, 0.0), label="Cu", color=CU_COL)
    h = sc.atom((6.55, 0.0), label="H")
    sc.bond(cu, h, order=1)
    fe, up, dn = _ferrocene_stack(sc, fe_xy=(1.6, 0.0), gap=1.25)
    # two adjacent upper-Cp carbons on the metal side carry the substituents
    right = sorted(up, key=lambda i: sc.atoms[i].pos[0])[-2:]
    c1, c2 = sorted(right, key=lambda i: sc.atoms[i].pos[1])   # lower, upper
    # P1 directly on c1 (lower-right Cp carbon) -> chelates Cu
    p1 = sc.atom((3.5, -1.3), label="P", color=P_COL)
    sc.bond(c1, p1, order=1)
    sc.bond(p1, cu, order=1, kind="coord")
    psub_p1(sc, p1)
    # CH(CH3) tether on c2 -> P2 -> chelates Cu
    ch = sc.atom((3.0, 1.55))
    me = sc.atom((2.7, 2.5), label="CH₃", fontscale=0.8)
    p2 = sc.atom((3.7, 1.2), label="P", color=P_COL)
    sc.bond(c2, ch, order=1)
    sc.bond(ch, me, order=1, kind="wedge")
    sc.bond(ch, p2, order=1)
    sc.bond(p2, cu, order=1, kind="coord")
    psub_p2(sc, p2)
    frozen = {fe, *up, *dn}
    return sc, name, {"exempt": frozen, "extra_rigid": [frozen]}


def _arm(sc, pid, deg, label):
    p = sc.atoms[pid].pos
    tip = polar(p, deg, 1.05 * L)
    tid = sc.atom(tip, label=label)
    sc.bond(pid, tid, order=1)
    return tid


if __name__ == "__main__":
    TEST = {
        "xantphos": "CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21",
        "segphos": chem.SEGPHOS_PARENT_SMILES,
        "dpephos": "O(c1ccccc1P(c1ccccc1)c1ccccc1)c1ccccc1P(c1ccccc1)c1ccccc1",
    }
    import relax_harness as R
    for nm, smi in TEST.items():
        sc, title = scene_from_smiles(smi, nm)
        H = R.Harness(scene=sc, title=nm)
        print(f"{nm:10s} atoms={len(sc.atoms)} bonds={len(sc.bonds)} "
              f"seed_metrics={H.metrics()}")
