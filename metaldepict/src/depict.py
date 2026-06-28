"""
depict.py
=========
Turn the 3D complex + T-REX descriptor into a *depiction graph* (JSON) that the
web viewer renders and the physics engine relaxes.

What it produces, per the design goals:
  * a clean 2D start layout   (CoordGen on the ligand + geometric metal placement)
  * abbreviation groups at two nesting levels   (t-Bu/OMe ; whole DTBM aryl)
  * symmetry: canonical-rank equivalence classes + the C2 mirror axis & pairs
  * wedge / dash perception from the 3D conformer's out-of-plane depth
  * explicit formal charges, plus the ionic (Cu(+) / H(-)) overlay
  * standardised bond lengths / metal angles for the physics rest state
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdCoordGen

try:
    from . import abbreviations as abbr
except ImportError:  # running as a script
    import abbreviations as abbr


# ---- standardised geometry (Angstrom; bonds normalised to ~1.0 in viewer) ----
IDEAL_BOND = {
    "aromatic": 1.39,
    "single": 1.50,
    "double": 1.34,
    "C-O": 1.43,
    "C-P": 1.83,
    "P-metal": 2.25,
    "H-metal": 1.50,
}


def _ligand_2d(lig: Chem.Mol) -> dict[int, tuple[float, float]]:
    """Clean 2D coordinates for the ligand via CoordGen (fallback: ETKDG-2D)."""
    m = Chem.Mol(lig)
    try:
        rdCoordGen.AddCoords(m)
    except Exception:
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(m)
    conf = m.GetConformer()
    return {i: (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y)
            for i in range(m.GetNumAtoms())}


def _pca_depth(mol3d: Chem.Mol) -> dict[int, float]:
    """Out-of-plane depth of every atom = projection on the smallest PCA axis."""
    if mol3d.GetNumConformers() == 0:
        return {i: 0.0 for i in range(mol3d.GetNumAtoms())}
    conf = mol3d.GetConformer()
    pts = np.array([[conf.GetAtomPosition(i).x,
                     conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z]
                    for i in range(mol3d.GetNumAtoms())])
    c = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - c)
    normal = vt[2]                       # smallest-variance direction
    depth = (pts - c) @ normal
    # np.linalg.svd's sign is arbitrary -> pin it so wedge parity is reproducible
    # (orient the plane normal so the most out-of-plane atom is on the + side)
    k = int(np.argmax(np.abs(depth)))
    if depth[k] < 0:
        depth = -depth
    # scale to a tidy range
    if np.abs(depth).max() > 1e-6:
        depth = depth / np.abs(depth).max()
    return {i: float(depth[i]) for i in range(len(depth))}


def _bond_type(b: Chem.Bond) -> tuple[str, float]:
    bt = b.GetBondType()
    if bt == Chem.BondType.DATIVE:
        return "dative", 1.0
    if b.GetIsAromatic() or bt == Chem.BondType.AROMATIC:
        return "aromatic", 1.5
    if bt == Chem.BondType.DOUBLE:
        return "double", 2.0
    if bt == Chem.BondType.TRIPLE:
        return "triple", 3.0
    return "single", 1.0


def _ordered_rings(mol: Chem.Mol) -> list[list[int]]:
    """SSSR rings with atoms in cyclic (bond-walk) order -- needed so the viewer
    can constrain each ring to a regular polygon."""
    ri = mol.GetRingInfo()
    out = []
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
            order.append(nxt); prev, cur = cur, nxt
        out.append(order)
    return out


def _atom_angle_deg(atom: Chem.Atom) -> float:
    """Ideal bond angle at this atom (used for 1-3 angle constraints)."""
    if atom.GetSymbol() == "Cu":
        return 120.0                       # trigonal metal
    hyb = atom.GetHybridization()
    if hyb == Chem.HybridizationType.SP3:
        return 109.5
    if hyb == Chem.HybridizationType.SP:
        return 180.0
    return 120.0                           # sp2 / aromatic


def _axial_cip(mol: Chem.Mol, biaryl) -> dict | None:
    """
    Assign the axial (atropisomeric) CIP descriptor of the biaryl from the 3D
    conformer.  RDKit does not perceive this atropisomer, so we do it directly:
    on each pivot aryl carbon the two ortho positions are the P-bearing carbon
    and the dioxole-O-bearing carbon; P (Z=15) outranks O (Z=8), so the
    P-bearing ortho is the higher-CIP substituent.  The sign of the torsion
    (highOrtho_a - axis_a - axis_b - highOrtho_b) gives aR (+) / aS (-) and the
    helicity P (+) / M (-).
    """
    if not biaryl or mol.GetNumConformers() == 0:
        return None
    from rdkit.Chem import rdMolTransforms
    a, b = biaryl
    conf = mol.GetConformer()

    def high_ortho(idx, other):
        cands = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                 if n.GetIsAromatic() and n.GetIdx() != other]
        for c in cands:                    # prefer the P-bearing ortho
            if any(nn.GetSymbol() == "P" for nn in mol.GetAtomWithIdx(c).GetNeighbors()):
                return c
        return cands[0] if cands else None

    ha, hb = high_ortho(a, b), high_ortho(b, a)
    if ha is None or hb is None:
        return None
    tor = rdMolTransforms.GetDihedralDeg(conf, ha, a, b, hb)
    descriptor = "aR" if tor > 0 else "aS"
    helicity = "P" if tor > 0 else "M"
    # wedge the two pivot->highOrtho bonds to show the twist (front/back)
    wedge = {(a, ha): ("up" if tor > 0 else "down"),
             (b, hb): ("down" if tor > 0 else "up")}
    return {"axis": [a, b], "high_ortho": [ha, hb],
            "priority_torsion_deg": round(float(tor), 1),
            "descriptor": descriptor, "helicity": helicity,
            "label": f"({descriptor})",
            "wedge_bonds": {f"{k[0]}-{k[1]}": v for k, v in wedge.items()},
            "_wedge": wedge}


def _phenyl_groups(mol: Chem.Mol, metal: int) -> list[dict]:
    """
    Level-2 abbreviation: each P-bound DTBM aryl (ring + its tBu/OMe trees)
    collapses to one 'DTBM' superatom hanging off the phosphorus.
    """
    groups = []
    gi = 0
    p_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "P"]
    ri = mol.GetRingInfo()
    for p in p_atoms:
        for nb in mol.GetAtomWithIdx(p).GetNeighbors():
            if nb.GetSymbol() != "C" or not nb.GetIsAromatic():
                continue
            ipso = nb.GetIdx()
            # ring containing ipso, all aromatic carbons
            ring = None
            for r in ri.AtomRings():
                if ipso in r and len(r) == 6 and all(
                    mol.GetAtomWithIdx(a).GetSymbol() == "C"
                    and mol.GetAtomWithIdx(a).GetIsAromatic() for a in r
                ):
                    ring = set(r)
                    break
            if ring is None:
                continue
            # a pendant phenyl is non-fused; the benzodioxole benzene shares its
            # two fusion carbons with the dioxole ring, so skip any fused ring and
            # never let the flood-fill leak across the biaryl backbone.
            if any(ri.NumAtomRings(a) > 1 for a in ring):
                continue
            # is this a *phenyl* (single external attachment = ipso->P) backbone?
            # collect the whole substituent tree: ring + everything hanging off it
            # except the bond back to P
            members = set(ring)
            stack = [a for a in ring]
            while stack:
                cur = stack.pop()
                for q in mol.GetAtomWithIdx(cur).GetNeighbors():
                    qi = q.GetIdx()
                    if qi == p or qi in members:
                        continue
                    members.add(qi)
                    stack.append(qi)
            # phenyl backbone (not the benzodioxole) -> exactly the ipso touches P
            touches_p = sum(1 for a in members
                            for q in mol.GetAtomWithIdx(a).GetNeighbors()
                            if q.GetIdx() == p)
            # benzodioxole rings also touch P at the backbone carbon; distinguish:
            # a DTBM aryl tree contains tBu (quaternary C with 3 CH3) and an OMe
            has_tbu = any(
                mol.GetAtomWithIdx(a).GetSymbol() == "C"
                and sum(1 for n in mol.GetAtomWithIdx(a).GetNeighbors()
                        if n.GetSymbol() == "C" and n.GetTotalNumHs() == 3) >= 3
                for a in members
            )
            if touches_p == 1 and has_tbu:
                gi += 1
                groups.append({
                    "id": f"l2_{gi}",
                    "label": "DTBM",
                    "atoms": sorted(members),
                    "attach": p,
                    "anchor": ipso,
                    "level": 2,
                })
    return groups


def build_depiction(complex_mol: Chem.Mol, info: dict, trex_desc: dict,
                    name: str, smiles: str, method: str) -> dict:
    mol = complex_mol
    n = mol.GetNumAtoms()
    metal = info["metal"]
    hydride = info["hydride"]
    donors = set(info["donors"])

    # ---- 2D layout: CoordGen on the FULL complex, so the P-Cu-P + biaryl
    # chelate is laid out as a real 7-membered ring.  This reproduces the
    # conventional published bisphosphine depiction (the biaryl backbone stacked
    # on one side, both phosphines pointing in to the metal, the aryls fanning
    # out) -- which laying out the ligand alone and bolting on the metal cannot. ----
    cg = Chem.Mol(mol)
    cg.RemoveAllConformers()
    try:
        rdCoordGen.AddCoords(cg)                       # clean layout (no stereo bias)
    except Exception:
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(cg)
    conf2d = cg.GetConformer()
    coords = {i: (float(conf2d.GetAtomPosition(i).x), float(conf2d.GetAtomPosition(i).y))
              for i in range(cg.GetNumAtoms())}
    P = list(donors)
    backbone_centroid = np.array([coords[i] for i in range(metal)]).mean(axis=0)
    cu_xy = np.array(coords[metal])

    depth = _pca_depth(mol)

    # ---- wedge/dash from RDKit on real (sp3) stereocentres ----
    # Assign stereo from the 3D conformer AFTER the 2D layout, then copy the
    # chiral tags onto the (already laid-out) 2D copy and wedge there -- so the
    # layout itself is never perturbed by stereo perception.
    wedge_map: dict[tuple[int, int], str] = {}
    try:
        Chem.AssignStereochemistryFrom3D(mol)
        for am, ac in zip(mol.GetAtoms(), cg.GetAtoms()):
            ac.SetChiralTag(am.GetChiralTag())
        Chem.AssignStereochemistry(cg, cleanIt=True, force=True)
        Chem.WedgeMolBonds(cg, cg.GetConformer())
        for b in cg.GetBonds():
            d = b.GetBondDir()
            if d == Chem.BondDir.BEGINWEDGE:
                wedge_map[(b.GetBeginAtomIdx(), b.GetEndAtomIdx())] = "up"
            elif d == Chem.BondDir.BEGINDASH:
                wedge_map[(b.GetBeginAtomIdx(), b.GetEndAtomIdx())] = "down"
    except Exception:
        pass

    # ---- symmetry: canonical-rank equivalence classes ----
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    classes: dict[int, list[int]] = {}
    for i, r in enumerate(ranks):
        classes.setdefault(r, []).append(i)
    sym_classes = [sorted(v) for v in classes.values() if len(v) > 1]
    # mirror axis (2D): metal -> midpoint of biaryl bond
    biaryl = _find_biaryl_bond(mol)
    # C2 mirror pairs: pair atoms ACROSS the two halves (split at the biaryl bond
    # and the metal) by matching unique canonical rank.  This avoids the bug of
    # treating any size-2 rank class as a pair -- two equivalent atoms in the
    # *same* half must not be reflected onto each other.
    # Primary: the true molecular C2 from a graph automorphism that swaps the two
    # P donors and is an involution (works for every backbone -- biaryl, ether,
    # xanthene, alkyl). Fallbacks keep the older biaryl-cut / rank heuristics.
    donor_pair = sorted(donors)
    mirror_pairs = _c2_mirror_pairs_auto(mol, metal, donor_pair)
    if not mirror_pairs:
        mirror_pairs = _c2_mirror_pairs(mol, ranks, metal, biaryl)
    if not mirror_pairs:
        mirror_pairs = [sorted(v) for v in classes.values() if len(v) == 2]
    if biaryl:
        ax_mid = np.array([coords[biaryl[0]], coords[biaryl[1]]]).mean(axis=0)
    else:
        ax_mid = backbone_centroid
    axis = {"p1": list(cu_xy), "p2": [float(ax_mid[0]), float(ax_mid[1])]}

    # ---- abbreviation groups ----
    l1_hits = abbr.detect_groups(mol)
    group_l1 = {}
    groups = []
    for h in l1_hits:
        groups.append({
            "id": h.gid, "label": h.label, "atoms": h.atoms,
            "attach": h.attach, "anchor": h.anchor, "level": 1, "kind": h.kind,
        })
        for a in h.atoms:
            group_l1[a] = h.gid
    l2_groups = _phenyl_groups(mol, metal)
    group_l2 = {}
    for g in l2_groups:
        groups.append(g)
        for a in g["atoms"]:
            group_l2[a] = g["id"]

    substituent_atoms = set(group_l2.keys())

    # ---- atoms ----
    ptable = Chem.GetPeriodicTable()
    atoms = []
    for a in mol.GetAtoms():
        i = a.GetIdx()
        sym = a.GetSymbol()
        if i == metal:
            role = "metal"
        elif i == hydride:
            role = "hydride"
        elif i in donors:
            role = "donor"
        elif i in substituent_atoms:
            role = "substituent"
        else:
            role = "backbone"
        ion = 0
        if i == metal:
            ion = +1
        elif i == hydride:
            ion = -1
        atoms.append({
            "id": i,
            "el": sym,
            "x": round(float(coords[i][0]), 4),
            "y": round(float(coords[i][1]), 4),
            "z": round(float(depth[i]), 4),
            "charge": a.GetFormalCharge(),
            "charge_ionic": ion,
            "aromatic": a.GetIsAromatic(),
            "nH": a.GetTotalNumHs(),
            "sym_class": ranks[i],
            "role": role,
            "angle_deg": _atom_angle_deg(a),
            "group_l1": group_l1.get(i),
            "group_l2": group_l2.get(i),
        })

    # ---- CIP (axial) ----
    cip = _axial_cip(mol, biaryl)
    biaryl_set = set(biaryl) if biaryl else set()

    # ---- bonds ----
    # The one real stereo element is the biaryl axis; published bisphosphine
    # drawings denote it by drawing the central biaryl bond BOLD, so we flag it
    # as `axis` rather than inventing wedges on the flanking bonds.
    # explicit Kekule double bonds (not aromatic "1.5" half-lines)
    korder: dict[tuple[int, int], float] = {}
    try:
        kmol = Chem.Mol(mol)
        Chem.Kekulize(kmol, clearAromaticFlags=True)
        for b in kmol.GetBonds():
            korder[(b.GetBeginAtomIdx(), b.GetEndAtomIdx())] = b.GetBondTypeAsDouble()
    except Exception:
        pass

    bonds = []
    for b in mol.GetBonds():
        ai, bi = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        btype, order = _bond_type(b)
        if btype == "aromatic":                 # -> explicit single/double (Kekule)
            od = korder.get((ai, bi)) or korder.get((bi, ai)) or 1.0
            order = 2.0 if od >= 2.0 else 1.0
            btype = "double" if order == 2.0 else "single"
        is_axis = bool(biaryl) and {ai, bi} == biaryl_set
        wedge, a0 = "none", ai
        if (ai, bi) in wedge_map:
            wedge, a0 = wedge_map[(ai, bi)], ai
        elif (bi, ai) in wedge_map:
            wedge, a0 = wedge_map[(bi, ai)], bi
        bonds.append({
            "a": ai, "b": bi, "a0": a0, "order": order, "type": btype,
            "wedge": wedge, "axis": is_axis,
        })

    return {
        "meta": {
            "name": name,
            "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
            "smiles": smiles,
            "conformer_method": method,
            "trex": trex_desc,
            "ideal_bond": IDEAL_BOND,
            "n_atoms": n,
            "description": "(DTBM-SEGPhos)Cu-H : bidentate phosphine chelate, "
                           "trigonal Cu(I) hydride",
        },
        "metal": _metal_block(metal, donors, hydride, trex_desc),
        "atoms": atoms,
        "bonds": bonds,
        "rings": _ordered_rings(mol),
        "groups": groups,
        "cip": {k: v for k, v in (cip or {}).items() if k != "_wedge"},
        "symmetry": {
            "classes": sym_classes,
            "mirror_pairs": mirror_pairs,
            "axis": axis,
            "point_group_hint": "C2",
        },
    }


def _c2_mirror_pairs_auto(mol, metal, donor_pair):
    """
    The molecular C2 as a graph automorphism: among all self-substructure
    matches, find an involution that fixes the metal and *swaps* the two P
    donors. The atom pairs it moves (i -> perm[i] != i) are the C2 mirror pairs.
    Bounded by maxMatches; symmetric tBu methyls only inflate the count, they do
    not break the search.
    """
    if len(donor_pair) != 2:
        return []
    d0, d1 = donor_pair
    n = mol.GetNumAtoms()
    try:
        matches = mol.GetSubstructMatches(mol, uniquify=False,
                                          maxMatches=50000, useChirality=False)
    except Exception:
        return []
    best = None
    for m in matches:
        if len(m) != n:
            continue
        if m[d0] != d1 or m[d1] != d0:
            continue
        if metal is not None and m[metal] != metal:
            continue
        if any(m[m[i]] != i for i in range(n)):     # must be an involution
            continue
        pairs = sorted({tuple(sorted((i, m[i]))) for i in range(n) if m[i] != i})
        # prefer the automorphism that moves the most atoms (the real C2, not a
        # local swap that leaves a whole arm fixed)
        if best is None or len(pairs) > len(best):
            best = pairs
    return [list(p) for p in best] if best else []


def _c2_mirror_pairs(mol, ranks, metal, biaryl):
    """
    Pair atoms related by the molecular C2 by cutting the biaryl bond and the
    metal (which together bridge the two halves) and matching, across halves,
    ranks that are unique within each half. Safe and bounded (no automorphism
    enumeration, which would explode on the symmetric tBu methyls).
    """
    if not biaryl:
        return []
    n = mol.GetNumAtoms()
    adj = {i: [] for i in range(n)}
    cut = {frozenset(biaryl)}
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if frozenset((a1, a2)) in cut or a1 == metal or a2 == metal:
            continue
        adj[a1].append(a2)
        adj[a2].append(a1)

    def comp(start):
        seen, stack = {start}, [start]
        while stack:
            c = stack.pop()
            for x in adj[c]:
                if x not in seen:
                    seen.add(x)
                    stack.append(x)
        return seen

    A, B = comp(biaryl[0]), comp(biaryl[1])
    if A & B:                       # halves not cleanly separated
        return []
    from collections import defaultdict
    ra, rb = defaultdict(list), defaultdict(list)
    for i in A:
        ra[ranks[i]].append(i)
    for i in B:
        rb[ranks[i]].append(i)
    pairs = []
    for r, ai in ra.items():
        bi = rb.get(r)
        if bi and len(ai) == 1 and len(bi) == 1:
            pairs.append(sorted((ai[0], bi[0])))
    return sorted(pairs)


def _metal_block(metal, donors, hydride, trex_desc) -> dict:
    """
    Authoritative coordination from the *known* construction (2 P donors + 1
    hydride => CN 3, trigonal).  T-REX's raw perception from an all-H xyz cannot
    pick the hydride out of ~100 hydrogens, so it is kept only as a record in
    meta.trex, while the depiction uses the correct CN/geometry here.
    """
    cn = len(donors) + (1 if hydride is not None else 0)
    geo_map = {2: ("ln", "linear"), 3: ("trpl", "trigonal planar"),
               4: ("thd", "tetrahedral"), 5: ("trbp", "trigonal bipyramidal"),
               6: ("O", "octahedral")}
    g, gname = geo_map.get(cn, ("unk", "unknown"))
    return {
        "id": metal, "el": "Cu", "cn": cn,
        "donors": sorted(donors), "hydride": hydride,
        "geometry": g, "geometry_name": gname,
        "trex_perceived_geometry": trex_desc.get("geometry_name"),
        "trex_perceived_cn": trex_desc.get("cn"),
    }


def _find_biaryl_bond(mol: Chem.Mol):
    """The single C-C bond joining the two benzodioxole aromatic systems."""
    ri = mol.GetRingInfo()
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if (a1.GetIsAromatic() and a2.GetIsAromatic()
                and not b.IsInRing()
                and a1.GetSymbol() == "C" and a2.GetSymbol() == "C"):
            # both ends must be in dioxole-bearing benzene rings (have an O 2 bonds away)
            if _near_dioxole(mol, a1.GetIdx()) and _near_dioxole(mol, a2.GetIdx()):
                return (a1.GetIdx(), a2.GetIdx())
    return None


def _near_dioxole(mol: Chem.Mol, idx: int) -> bool:
    for n in mol.GetAtomWithIdx(idx).GetNeighbors():
        for nn in n.GetNeighbors():
            if nn.GetSymbol() == "O":
                return True
    return False


def _stereo_bond_set(mol, metal, hydride, donors, biaryl):
    """Bonds that carry stereochemical meaning -> candidates for wedge/dash."""
    s = set()
    # metal coordination bonds
    for d in donors:
        s.add((metal, d))
    s.add((metal, hydride))
    # P -> ipso aryl bonds (phosphine pyramidality)
    for d in donors:
        for nb in mol.GetAtomWithIdx(d).GetNeighbors():
            if nb.GetIdx() != metal:
                s.add((d, nb.GetIdx()))
    # biaryl axis (atropisomerism)
    if biaryl:
        s.add(biaryl)
    return s


def emit(depiction: dict, data_dir: str | Path, viewer_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    viewer_dir = Path(viewer_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    viewer_dir.mkdir(parents=True, exist_ok=True)
    stem = "cuh_dtbm_segphos"
    (data_dir / f"{stem}.json").write_text(json.dumps(depiction, indent=2))
    (viewer_dir / "molecule_data.js").write_text(
        "// auto-generated by metaldepict -- window.MOLECULE\n"
        "window.MOLECULE = " + json.dumps(depiction) + ";\n"
    )
    print(f"wrote {data_dir/f'{stem}.json'} and {viewer_dir/'molecule_data.js'}")
