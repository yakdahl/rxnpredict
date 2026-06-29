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
MET_COL = "#5a6470"          # generic transition-metal glyph colour (Ru/Pd/...)
COLORS = {"P": P_COL, "Cu": CU_COL, "O": O_COL, "Fe": FE_COL,
          "N": "#2c3e9e", "S": "#b8860b", "B": "#1f8a70", "Si": "#555",
          "Ru": MET_COL, "Pd": MET_COL, "Ni": MET_COL, "Rh": MET_COL,
          "Ir": MET_COL, "Pt": MET_COL, "Co": MET_COL, "Mn": MET_COL,
          "Zr": MET_COL, "Ti": MET_COL, "Hf": MET_COL, "V": MET_COL,
          "Cr": MET_COL, "Cl": "#2e8b57", "Br": "#8b4513"}
# the catalytic metal centre (NOT ferrocene Fe); donors that coordinate it
COMPLEX_METALS = {"Cu", "Ru", "Pd", "Ni", "Rh", "Ir", "Pt", "Co", "Au", "Ag", "Mn"}
# centres that only appear via scene_from_mol on a PRE-ASSEMBLED complex (never a
# ferrocene SMILES, which is routed away earlier): Fe(salen), Cp2Zr.  Broader than
# COMPLEX_METALS so seed-building can find them without those symbols ever being
# mistaken for a coordination centre inside a ferrocene Cu-H ligand.
SEED_METALS = COMPLEX_METALS | {"Fe", "Zr", "Ti", "Hf", "V", "Cr"}
DONOR_SYMBOLS = {"P", "N", "O", "S", "C"}        # atoms that may coordinate a metal
# atoms drawn as a glyph (heteroatoms + metal + hydride); carbon stays blank
LABEL_SYMBOLS = {"P", "O", "N", "S", "B", "Si", "H", "Cl", "Br", "F",
                 "Fe", "Cu", "Ru", "Pd", "Ni", "Rh", "Ir", "Pt", "Co", "Mn",
                 "Zr", "Ti", "Hf", "V", "Cr"}

# Common organic-chemistry substituent abbreviations (from the standard reference
# list).  Split into ALWAYS-condensed (conventionally drawn as a label) and
# CROWD-GATED (drawn explicit by default, condensed only when the structure is too
# crowded -- the optimiser decides).  Order is SPECIFIC -> GENERAL so e.g. Piv/Bn
# win over the tBu/Ph fragments they contain.
_ABBR_ALWAYS_STR = (
    "Piv [*]C(=O)C(C)(C)C Piv Piv\n"             # pivaloyl  (before tBu/Ac)
    "Boc [*]OC(=O)C(C)(C)C Boc Boc\n"
    "OAc [*]OC(=O)C OAc AcO\n"
    "Ac [*]C(=O)C Ac Ac\n"
    "CO [*]C#O CO CO\n"                          # metal carbonyl M-C#O -> M-CO
    "CF3 [*]C(F)(F)F CF3 CF3\n"
    "TMS [*][Si](C)(C)C TMS TMS\n"
    "tBu [*]C(C)(C)C tBu tBu\n"
    "OMe [*]OC OMe MeO\n"
    "OEt [*]OCC OEt EtO\n"
    "NMe2 [*]N(C)C NMe2 Me2N\n"
    "nOct [*]CCCCCCCC C8H17 C8H17\n"
    "Ph [*]c1ccccc1 Ph Ph\n")
_ABBR_CROWD_STR = (                              # condensed only when crowded
    "Bn [*]Cc1ccccc1 Bn Bn\n"                    # benzyl (before Cy/Ph/Et)
    "Cy [*]C1CCCCC1 Cy Cy\n"                     # cyclohexyl
    "iPr [*]C(C)C iPr iPr\n"                     # isopropyl
    "Et [*]CC Et Et\n")
_ABBR = rdAbbreviations.ParseAbbreviations(_ABBR_ALWAYS_STR, True, False)
_ABBR_MAX = rdAbbreviations.ParseAbbreviations(
    _ABBR_ALWAYS_STR + _ABBR_CROWD_STR, True, False)


def _condense(cx, abbr):
    """Condense abbreviations CONSISTENTLY: RDKit only collapses one match per
    pass, so repeat until the atom count stops changing -- then ALL e.g. four
    P-tBu groups of a PtBu2 ligand become a t-Bu label, not a mix of label and
    explicit."""
    for _ in range(8):
        try:
            nxt = rdAbbreviations.CondenseMolAbbreviations(cx, abbr, maxCoverage=1.0)
        except Exception:
            return cx
        if nxt.GetNumAtoms() == cx.GetNumAtoms():
            return nxt
        cx = nxt
    return cx

# Abbreviations whose COORDINATING / attachment atom should face the parent bond,
# given as (forward, reversed) display forms with the attachment atom written
# FIRST in the forward form.  The renderer centres a label, so to keep the
# attachment atom next to the bond the label is flipped to the reversed form when
# the parent sits to the RIGHT of the superatom (M-CO drawn as OC, an aryl-OMe
# drawn MeO when the ring is on the right, etc.).
ABBR_FORMS = {
    "OMe": ("OMe", "MeO"), "MeO": ("OMe", "MeO"),
    "CO": ("CO", "OC"), "OC": ("CO", "OC"),
    "CF3": ("CF3", "F3C"), "F3C": ("CF3", "F3C"),
    "OPh": ("OPh", "PhO"), "PhO": ("OPh", "PhO"),
    "SMe": ("SMe", "MeS"), "MeS": ("SMe", "MeS"),
    "NMe2": ("NMe2", "Me2N"), "Me2N": ("NMe2", "Me2N"),
    "NMe3": ("NMe3", "Me3N"), "Me3N": ("NMe3", "Me3N"),
    "PPh3": ("PPh3", "Ph3P"), "Ph3P": ("PPh3", "Ph3P"),
    "NH2": ("NH2", "H2N"), "H2N": ("NH2", "H2N"),
    "OEt": ("OEt", "EtO"), "EtO": ("OEt", "EtO"),
    "OAc": ("OAc", "AcO"), "AcO": ("OAc", "AcO"),
}


def _orient_abbr(label, parent_xy, self_xy):
    """Pick the (forward|reversed) form of an abbreviation so its attachment atom
    faces the parent: reversed when the parent is to the RIGHT, forward otherwise."""
    forms = ABBR_FORMS.get(label)
    if not forms:
        return label
    return forms[1] if parent_xy[0] > self_xy[0] else forms[0]


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
    """Cu-H bisphosphine entry point: ligand SMILES -> Cu-H complex -> scene."""
    cx, _ = _complex(smiles)
    if any(a.GetSymbol() == "Fe" for a in cx.GetAtoms()):
        return _ferrocene_scene(cx, name)
    return scene_from_mol(cx, name, abbreviate=abbreviate), name or "molecule"


def scene_from_mol(cx, name=None, abbreviate=True, abbr_level="min"):
    """Build a Scene from ANY pre-assembled metal complex RDKit mol (the metal
    already present, donor->metal bonds set as DATIVE/coordinate).  Works for any
    centre in SEED_METALS and any donor atom -- Cu-H bisphosphine, Ru-PNP pincer,
    Pd biaryl-monophosphine, Cu oxalamide, Fe-salen, Grubbs, metallocenes, ...

    eta-n (HAPTIC) coordination is an OPTION fed in as connectivity: bond every
    ring carbon of an eta-n ligand to the metal with a DATIVE bond.  The tool then
    recognises the ring, rotates it into the correct perspective-disc orientation
    facing the metal (the ferrocene Cp drawing), and replaces the n spokes with a
    single dashed metal->centroid bond.  Returns a Scene whose `.meta` carries
    {metal, exempt, extra_rigid} for the Harness.

    abbr_level: "min" condenses only the conventionally-labelled groups (tBu, OMe,
    CF3, Ph, CO, ...); "max" ALSO condenses the crowd-gated alkyls (iPr, Cy, Bn,
    Et) -- the optimiser raises the level when the explicit drawing is too crowded.
    """
    if abbreviate:
        cx = _condense(cx, _ABBR_MAX if abbr_level == "max" else _ABBR)
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

    metal = next((a.GetIdx() for a in cx.GetAtoms()
                  if a.GetSymbol() in SEED_METALS), None)
    haptic = _haptic_rings(cx, metal) if metal is not None else []
    if metal is not None:
        haptic_atoms = {a for h in haptic for a in h}
        sigma = [nb.GetIdx() for nb in cx.GetAtomWithIdx(metal).GetNeighbors()
                 if nb.GetIdx() not in haptic_atoms]
        if haptic:
            _orient_for_haptic(pos, metal, sigma)
        else:
            hyd = next((nb.GetIdx() for nb in cx.GetAtomWithIdx(metal).GetNeighbors()
                        if nb.GetSymbol() == "H"), None)
            _orient_metal(pos, metal, hyd, cx)

    hinfo = _layout_haptic_discs(pos, metal, haptic, sigma, cx) if haptic else []
    return _build_scene(cx, kek, pos, metal, hinfo)


def _orient_metal(pos, metal, hyd, cx):
    """Rotate so metal->H points +x (H to the right) when there is a hydride;
    otherwise orient the FARTHEST metal->neighbour vector horizontally, giving a
    stable, readable frame for any complex (Cu-H, Ru-PNP, Pd biaryl, ...)."""
    mx, my = pos[metal]
    if hyd is not None:
        ref = hyd
    else:
        nbrs = [nb.GetIdx() for nb in cx.GetAtomWithIdx(metal).GetNeighbors()]
        if not nbrs:
            return
        ref = max(nbrs, key=lambda i: math.hypot(pos[i][0] - mx, pos[i][1] - my))
    rx, ry = pos[ref]
    th = -math.atan2(ry - my, rx - mx)
    c, s = math.cos(th), math.sin(th)
    for i, (x, y) in list(pos.items()):
        x0, y0 = x - mx, y - my
        pos[i] = (mx + c * x0 - s * y0, my + s * x0 + c * y0)


def _ring_centroid(mol, ring, pos):
    xs = [pos[i][0] for i in ring]
    ys = [pos[i][1] for i in ring]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


# --------------------------------------------------------------------------- #
#  eta-n HAPTIC coordination (cyclopentadienyl / arene), drawn as a perspective
#  disc facing the metal -- the requested general "cyclopentyl nu5" option.
# --------------------------------------------------------------------------- #
def _haptic_rings(cx, metal):
    """Rings whose carbons (mostly) DATIVE-bond the metal == an eta-n ligand.
    Returns each as a list of ring atom idxs in cyclic (adjacency) order."""
    out = []
    for ring in cx.GetRingInfo().AtomRings():
        dat = [a for a in ring
               if cx.GetBondBetweenAtoms(a, metal) is not None
               and cx.GetBondBetweenAtoms(a, metal).GetBondType() == Chem.BondType.DATIVE]
        if len(dat) >= 3 and len(dat) >= len(ring) - 1:
            out.append(list(ring))                   # AtomRings() is already cyclic
    return out


def _orient_for_haptic(pos, metal, sigma):
    """Rotate the whole complex so the sigma-ligand bundle points DOWN (-y) and
    the eta-n discs sit UP -- the canonical metallocene / piano-stool frame."""
    mx, my = pos[metal]
    if not sigma:
        return
    sx = sum(pos[s][0] - mx for s in sigma)
    sy = sum(pos[s][1] - my for s in sigma)
    if abs(sx) < 1e-9 and abs(sy) < 1e-9:
        return
    cur = math.atan2(sy, sx)
    th = (-math.pi / 2) - cur                        # send sigma bundle to 270 deg
    c, s = math.cos(th), math.sin(th)
    for i, (x, y) in list(pos.items()):
        x0, y0 = x - mx, y - my
        pos[i] = (mx + c * x0 - s * y0, my + s * x0 + c * y0)


def _disc_dirs(n, has_sigma):
    """Outward directions (deg) for n eta-n discs about the metal: a vertical
    sandwich when there is no sigma ligand (ferrocene/rhodocene), else a clamshell
    opening UP, away from the sigma ligands (bent metallocene / piano stool)."""
    if not has_sigma:
        if n == 1:
            return [90.0]
        if n == 2:
            return [90.0, 270.0]
        return [90.0 + 360.0 * k / n for k in range(n)]
    if n == 1:
        return [90.0]
    bend = 70.0
    return [90.0 + bend / 2.0 - bend * k / (n - 1) for k in range(n)]


def _subtree(cx, start, ring_set, metal):
    seen, stack = {start}, [start]
    while stack:
        x = stack.pop()
        for nb in cx.GetAtomWithIdx(x).GetNeighbors():
            y = nb.GetIdx()
            if y in seen or y in ring_set or y == metal:
                continue
            seen.add(y); stack.append(y)
    seen.discard(start)
    return seen


def _layout_haptic_discs(pos, metal, rings, sigma, cx, gap=1.62, squash=0.46):
    """Re-lay each eta-n ring as a perspective disc whose face turns TOWARD the
    metal: a regular polygon foreshortened ALONG the metal->centroid axis (so that
    axis is the ellipse's minor axis and the eta-n bond meets the ring centre
    head-on), full width across.  Centred `gap`*L from the metal; substituents
    ride rigidly with their ring atom.

    A bare metallocene (no sigma ligand: ferrocene / rhodocene) keeps both discs
    tilted the SAME way (apex up, front/lower edge bold), the classic ferrocene
    look.  With sigma ligands (bent metallocene, piano-stool half sandwich) each
    disc points its apex OUTWARD and bolds the edge nearest the metal.  Returns
    hinfo: per-ring {ring, centroid, edge styles}."""
    mx, my = pos[metal]
    dirs = _disc_dirs(len(rings), bool(sigma))
    sandwich = not sigma                              # ferrocene-style same tilt
    hinfo = []
    for ring, ddeg in zip(rings, dirs):
        n = len(ring)
        pen_r = L / (2 * math.sin(math.pi / n))
        u = (math.cos(math.radians(ddeg)), math.sin(math.radians(ddeg)))   # foreshorten axis
        up = (-u[1], u[0])                            # full (major) axis
        center = (mx + gap * L * u[0], my + gap * L * u[1])
        face = 90.0 if sandwich else ddeg            # apex orientation of the polygon
        fu = (math.cos(math.radians(face)), math.sin(math.radians(face)))
        fp = (-fu[1], fu[0])
        ring_set = set(ring)

        def vpos(kk):                                 # vertex position at local slot kk
            th = math.radians(90.0 + 360.0 * kk / n)
            lx, ly = pen_r * math.cos(th), pen_r * math.sin(th)
            wx, wy = lx * fp[0] + ly * fu[0], lx * fp[1] + ly * fu[1]
            a = (wx * u[0] + wy * u[1]) * squash
            b = wx * up[0] + wy * up[1]
            return (center[0] + a * u[0] + b * up[0], center[1] + a * u[1] + b * up[1])

        # ROTATE the Cp ring so its bulkiest substituent sits on the vertex
        # FARTHEST from the metal -- spinning the disc about its 5-fold axis (a
        # real, coordination-preserving degree of freedom) keeps a big group
        # (t-Bu, ...) pointing cleanly outward instead of folding into the ring.
        bulk = [len(_subtree(cx, a, ring_set, metal)) for a in ring]
        shift = 0
        if max(bulk) > 0:
            jmax = max(range(n), key=lambda k: bulk[k])
            far = max(range(n), key=lambda k: math.hypot(vpos(k)[0] - mx,
                                                         vpos(k)[1] - my))
            shift = (far - jmax) % n
        newpos = {atom: vpos((k + shift) % n) for k, atom in enumerate(ring)}
        for atom in ring:
            pos[atom] = newpos[atom]
        # ring substituents (Cp* methyls, ...) radiate straight OUT from the disc
        # centre through their ring carbon -- so they splay evenly around the rim
        # (foreshortened with the disc) instead of bunching at RDKit's flat-ring
        # directions.  The whole sub-tree rides on the placed first atom.
        for atom in ring:
            ox = newpos[atom][0] - center[0]
            oy = newpos[atom][1] - center[1]
            ol = math.hypot(ox, oy) or 1e-9
            ox, oy = ox / ol, oy / ol
            for nb in cx.GetAtomWithIdx(atom).GetNeighbors():
                s = nb.GetIdx()
                if s in ring_set or s == metal:
                    continue
                sub = {s} | _subtree(cx, s, ring_set | {atom}, metal)
                tgt = (newpos[atom][0] + ox * L, newpos[atom][1] + oy * L)
                dx, dy = tgt[0] - pos[s][0], tgt[1] - pos[s][1]
                for a in sub:
                    pos[a] = (pos[a][0] + dx, pos[a][1] + dy)
        # "wedges toward the forefront": the FRONT (lower, toward-viewer) edge of
        # the tilted disc is drawn BOLD, the two edges flanking it TAPER (wide end
        # at the front vertex), the receding edges plain -- the same convention as
        # the ferrocene Cp and the tilted aryls.  Front = the two lowest vertices,
        # consistently for both the sandwich and the bent metallocene.
        front = set(sorted(ring, key=lambda a: pos[a][1])[:2])
        style = {}
        for k in range(n):
            a, b = ring[k], ring[(k + 1) % n]
            fa, fb = a in front, b in front
            if fa and fb:
                style[frozenset((a, b))] = ("bold",)
            elif fa or fb:
                style[frozenset((a, b))] = ("taper", a if fa else b)  # wide at front
            else:
                style[frozenset((a, b))] = ("plain",)
        hinfo.append({"ring": ring, "center": center, "style": style})
    return hinfo


def _build_scene(cx, kek, pos, metal=None, hinfo=None):
    sc = Scene()
    hinfo = hinfo or []
    hap_atoms = {a for h in hinfo for a in h["ring"]}
    hap_edges = {}
    for h in hinfo:
        hap_edges.update(h["style"])
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
        # an abbreviation superatom with a coordinating/attachment atom (OMe, CO,
        # OPh, NMe3, PPh3, ...) is oriented so that atom faces its parent bond.
        if lbl in ABBR_FORMS:
            nbrs = [nb.GetIdx() for nb in a.GetNeighbors()]
            if nbrs:
                ax, ay = pos[a.GetIdx()]
                parent = min(nbrs, key=lambda p: math.hypot(pos[p][0] - ax,
                                                            pos[p][1] - ay))
                lbl = _orient_abbr(lbl, pos[parent], pos[a.GetIdx()])
        fs = 0.92 if (a.GetAtomicNum() == 0 and len(lbl) > 2) else 1.0
        idmap[a.GetIdx()] = sc.atom(pos[a.GetIdx()], label=lbl, color=col,
                                    fontscale=fs)

    for b in cx.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        si, sj = cx.GetAtomWithIdx(i).GetSymbol(), cx.GetAtomWithIdx(j).GetSymbol()
        a, bb = idmap[i], idmap[j]
        # eta-n spokes (metal <-> a haptic ring carbon) are not drawn individually;
        # a single dashed metal->centroid bond is added after the loop instead.
        if metal in (i, j) and ({i, j} & hap_atoms):
            continue
        # an edge of an eta-n disc -> perspective styling: bold front edge, tapers
        # flanking it (wide end toward the viewer), plain receding edges.
        es = hap_edges.get(frozenset((i, j)))
        if es is not None:
            if es[0] == "taper":
                wide = es[1]
                narrow = j if wide == i else i
                sc.bond(idmap[narrow], idmap[wide], order=1, kind="taper",
                        width=_CPW_BIG)
            elif es[0] == "bold":
                sc.bond(a, bb, order=1, kind="bold")
            else:
                sc.bond(a, bb, order=1)
            continue
        # donor->metal coordination is exactly the DATIVE bonds set when the
        # complex was assembled (P/N/O/S/C_ipso -> metal) -> dashed coord bond.
        # Covalent metal-X ancillaries (Cu-H, Ru-Cl, Pd-Br, Pd-Ph, Fe-Cl) are
        # ordinary SINGLE bonds and stay plain lines.
        if (b.GetBondType() == Chem.BondType.DATIVE
                and ({si, sj} & SEED_METALS)):
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
        # ordinary single / double (double drawn inside its ring); never re-draw a
        # double INSIDE an eta-n disc (its edges were already styled above)
        inside = None
        if order == 2 and frozenset((i, j)) not in hap_edges:
            r = next((rr for rr in arom_rings if i in rr and j in rr), None)
            if r:
                inside = _ring_centroid(cx, r, pos)
        sc.bond(a, bb, order=order, inside=inside)

    # finish each eta-n disc: aromatic circle + an invisible centroid anchor +
    # one dashed metal->centroid coord bond; freeze ring+centroid+metal rigid.
    exempt, extra_rigid = set(), []
    for h in hinfo:
        vids = [idmap[a] for a in h["ring"]]
        sc.ring_circle(vids, r_frac=0.58)
        cen = sc.atom(h["center"], label="", halo=False)
        if metal is not None:
            sc.bond(idmap[metal], cen, order=1, kind="coord")
        frozen = set(vids) | {cen}
        if metal is not None:
            frozen.add(idmap[metal])
        exempt |= frozen
        extra_rigid.append(frozen)
    sc.meta = {"metal": idmap[metal] if metal is not None else None,
               "exempt": exempt, "extra_rigid": extra_rigid}
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


_CPW_BIG = 0.17 * L      # wide end of the Cp perspective edges (== bold width)


def _cp_ring(sc, center, squash=0.6):
    """Cyclopentadienyl ring drawn in perspective (classic DPPF style): a
    vertically-squashed pentagon, apex UP, with an inscribed aromatic circle.
    The bottom FRONT edge is drawn BOLD (it is between the two perspective edges,
    so it stays big); the two edges adjacent to it TAPER from a normal bond width
    at the back up to the bold width at the front, so the disc reads as tilting
    toward the viewer; the two back edges are plain.  Both rings use the SAME
    tilt.  Vertices (CCW from apex): apex, ML, BL, BR, MR.  Returns (ids, cen)
    where `cen` is an invisible atom at the ring centre."""
    cx, cy = center
    ang = [90, 162, 234, 306, 18]                     # apex, ML, BL, BR, MR
    ids = [sc.atom((cx + _PEN_R * math.cos(math.radians(a)),
                    cy + _PEN_R * math.sin(math.radians(a)) * squash)) for a in ang]
    apex, ml, bl, br, mr = ids
    sc.bond(bl, br, order=1, kind="bold")             # FRONT edge stays BIG
    sc.bond(ml, bl, order=1, kind="taper", width=_CPW_BIG)   # normal@back -> big@front
    sc.bond(mr, br, order=1, kind="taper", width=_CPW_BIG)
    sc.bond(apex, ml, order=1)                         # back edges -> plain
    sc.bond(mr, apex, order=1)
    sc.ring_circle(ids, r_frac=0.58)
    cen = sc.atom((cx, cy), label="", halo=False)     # ring-centre anchor (eta5)
    return ids, cen


def _ferrocene_stack(sc, fe_xy=(1.6, 0.0), gap=1.25, squash=0.6,
                     metal_label="Fe", metal_col=None):
    """Reference-style SANDWICH with both Cp discs tilted the SAME way (apex up):
    upper ring above the metal, lower ring below, metal labelled in the centre,
    eta5 drawn as a DASHED line from the metal to the CENTRE of each ring.  The
    metal label/colour is parametrised so the same parallel-sandwich depiction
    serves any metallocene (ferrocene Fe, rhodocene Cp2Rh, ...).  Returns
    (metal, upper_ids, lower_ids, centre_ids)."""
    fx, fy = fe_xy
    up, cu = _cp_ring(sc, (fx, fy + gap), squash=squash)
    dn, cd = _cp_ring(sc, (fx, fy - gap), squash=squash)
    fe = sc.atom((fx, fy), label=metal_label, color=metal_col or FE_COL)
    sc.bond(fe, cu, order=1, kind="coord")            # eta5: dashed M -> ring centre
    sc.bond(fe, cd, order=1, kind="coord")
    return fe, up, dn, [cu, cd]


def _bent_cp_ring(sc, center, deg, squash=0.62, scale=1.0):
    """A Cp disc for a BENT metallocene: same perspective pentagon as the parallel
    sandwich, but ROTATED so its apex points away from the metal along `deg` (the
    metal->centroid direction), giving the open clamshell that reads as a bent
    metallocene.  Returns (vertex_ids, centre_anchor)."""
    cx, cy = center
    r = _PEN_R * scale
    base = [90, 162, 234, 306, 18]                    # apex-up reference pentagon
    rot = deg - 90.0                                  # turn apex from up to `deg`
    ids = []
    for a in base:
        th = math.radians(a + rot)
        # squash laterally (perpendicular to the metal->centroid spine) for tilt
        ux, uy = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        px, py = r * math.cos(th), r * math.sin(th)
        along = px * ux + py * uy
        perp = -px * uy + py * ux
        perp *= squash
        ids.append(sc.atom((cx + along * ux - perp * uy,
                            cy + along * uy + perp * ux)))
    n = len(ids)
    # the edge nearest the metal is drawn BOLD (front), its two neighbours taper.
    order_by_dist = sorted(range(n),
                           key=lambda k: (sc.atoms[ids[k]].pos[0] - cx) *
                           math.cos(math.radians(deg)) +
                           (sc.atoms[ids[k]].pos[1] - cy) * math.sin(math.radians(deg)))
    near = set(order_by_dist[:2])                     # two vertices closest to metal
    for k in range(n):
        a, b = ids[k], ids[(k + 1) % n]
        ka, kb = k in near, (k + 1) % n in near
        if ka and kb:
            sc.bond(a, b, order=1, kind="bold")
        elif ka or kb:
            sc.bond(a, b, order=1, kind="taper", width=_CPW_BIG)
        else:
            sc.bond(a, b, order=1)
    sc.ring_circle(ids, r_frac=0.58)
    cen = sc.atom((cx, cy), label="", halo=False)
    return ids, cen


def build_bent_metallocene(name, metal_label, ancillary, metal_col=None,
                           bend_deg=68.0, gap=1.85, squash=0.62, anc_len=1.05):
    """A BENT metallocene Cp2M(X)n -- two Cp discs opened into a clamshell with the
    metal at the vertex and `ancillary` sigma ligands (e.g. ['Cl','Cl'] for
    Cp2ZrCl2) fanning out below.  Built from the ferrocene perspective-Cp drawing
    (the requested 'starting point').  The two Cp rings + centroids + metal freeze
    into one rigid body (the metal is also the pinned centre, so the whole core is
    held in the hand-placed bent geometry); only the ancillary arms relax.
    Returns (sc, name, meta) with meta carrying the metal id for Harness(metal=)."""
    sc = Scene()
    mx, my = 2.6, 0.0
    half = bend_deg / 2.0
    left_deg, right_deg = 90.0 + half, 90.0 - half     # centroid directions (up V)
    lc = polar((mx, my), left_deg, gap * L)
    rc = polar((mx, my), right_deg, gap * L)
    left, cl_anchor = _bent_cp_ring(sc, lc, left_deg, squash=squash)
    right, cr_anchor = _bent_cp_ring(sc, rc, right_deg, squash=squash)
    metal = sc.atom((mx, my), label=metal_label, color=metal_col or MET_COL)
    sc.bond(metal, cl_anchor, order=1, kind="coord")   # eta5 dashed
    sc.bond(metal, cr_anchor, order=1, kind="coord")
    # ancillary sigma ligands fan out below the metal, centred on -y
    nx = len(ancillary)
    spread = 56.0
    for k, lbl in enumerate(ancillary):
        a = (270.0 - spread / 2.0 + spread * k / max(1, nx - 1)) if nx > 1 else 270.0
        tip = polar((mx, my), a, anc_len * L)
        col = COLORS.get(lbl, "#111")
        xid = sc.atom(tip, label=lbl, color=col)
        sc.bond(metal, xid, order=1)
    frozen = {metal, cl_anchor, cr_anchor, *left, *right}
    meta = {"metal": metal, "exempt": frozen, "extra_rigid": [frozen]}
    return sc, name, meta


def build_metallocene_sandwich(name, metal_label, metal_col=None):
    """A PARALLEL-sandwich metallocene Cp2M (rhodocene Cp2Rh, ...), the ferrocene
    depiction re-used verbatim with a different centre.  The whole sandwich is one
    frozen rigid body; the metal is the pinned centre.  Returns (sc, name, meta)."""
    sc = Scene()
    fe, up, dn, cens = _ferrocene_stack(sc, fe_xy=(2.6, 0.0), gap=1.4,
                                        metal_label=metal_label, metal_col=metal_col)
    frozen = {fe, *up, *dn, *cens}
    meta = {"metal": fe, "exempt": frozen, "extra_rigid": [frozen]}
    return sc, name, meta


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
    fe, up, dn, cens = _ferrocene_stack(sc, fe_xy=(1.6, 0.0), gap=1.25)
    sc.bond(c["pu"], _cp_right_vertex(sc, up, True), order=1)   # P on upper Cp
    sc.bond(c["pd"], _cp_right_vertex(sc, dn, False), order=1)  # P on lower Cp
    psub(sc, c)
    frozen = {fe, *up, *dn, *cens}
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
    fe, up, dn, cens = _ferrocene_stack(sc, fe_xy=(1.6, 0.0), gap=1.25)
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
    frozen = {fe, *up, *dn, *cens}
    return sc, name, {"exempt": frozen, "extra_rigid": [frozen]}


def _arm(sc, pid, deg, label, length=1.05):
    p = sc.atoms[pid].pos
    tip = polar(p, deg, length * L)
    tid = sc.atom(tip, label=label)
    sc.bond(pid, tid, order=1)
    return tid


def _aryl_para(sc, pid, deg, para_label, kek=0):
    """Place a phenyl ring on P (ipso back at P) carrying a para substituent
    label (e.g. CF3) -- for ligands whose P-aryls have a single para group."""
    p = sc.atoms[pid].pos
    ipso = polar(p, deg, L)
    center = polar(ipso, deg, _MT.HEX_R)
    ids = _MT.place_hexagon(sc, center, deg + 180.0, kek)
    sc.bond(pid, ids[0], order=1)
    _MT.text_arm(sc, ids[3], _MT.out_angle(sc, center, ids[3]), para_label,
                 length=1.0, fontscale=0.9)
    return ids


def _aryl_35(sc, pid, deg, label, kek=0, ring_scale=1.0):
    """Place a phenyl ring on P carrying TWO meta (3,5) substituent labels --
    e.g. a bis-3,5-(CF3)2-phenyl P-aryl.  `ring_scale` shrinks the hexagon (the
    P-aryl bond stays L) so the ring can be matched to a smaller reference ring."""
    p = sc.atoms[pid].pos
    ipso = polar(p, deg, L)
    center = polar(ipso, deg, _MT.HEX_R * ring_scale)
    ids = _MT.place_hexagon(sc, center, deg + 180.0, kek, scale=ring_scale)
    sc.bond(pid, ids[0], order=1)
    for m in (ids[2], ids[4]):                    # the two meta (3 and 5) carbons
        _MT.text_arm(sc, m, _MT.out_angle(sc, center, m), label,
                     length=1.0, fontscale=0.9)
    return ids


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
