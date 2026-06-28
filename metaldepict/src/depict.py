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

    # ---- 2D layout: ligand via CoordGen, metal/hydride placed geometrically ----
    lig_n = metal                       # ligand atoms are 0..metal-1 (Cu appended)
    lig = Chem.RWMol(mol)
    for idx in sorted([hydride, metal], reverse=True):
        lig.RemoveAtom(idx)
    coords = _ligand_2d(lig.GetMol())   # indices 0..lig_n-1 align with mol
    P = list(donors)
    pm = np.array([coords[P[0]], coords[P[1]]]).mean(axis=0)
    backbone_centroid = np.array([coords[i] for i in range(lig_n)]).mean(axis=0)
    out_dir = pm - backbone_centroid
    if np.linalg.norm(out_dir) < 1e-6:
        out_dir = np.array([0.0, 1.0])
    out_dir = out_dir / np.linalg.norm(out_dir)
    bond_len = 1.5
    cu_xy = pm + out_dir * bond_len * 1.4
    h_xy = cu_xy + out_dir * bond_len
    coords[metal] = tuple(cu_xy)
    coords[hydride] = tuple(h_xy)

    depth = _pca_depth(mol)

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
            "group_l1": group_l1.get(i),
            "group_l2": group_l2.get(i),
        })

    # ---- bonds ----
    stereo_bonds = _stereo_bond_set(mol, metal, hydride, donors, biaryl)
    bonds = []
    for b in mol.GetBonds():
        ai, bi = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        btype, order = _bond_type(b)
        wedge = "none"
        if (ai, bi) in stereo_bonds or (bi, ai) in stereo_bonds:
            dz = depth[bi] - depth[ai]
            if abs(dz) > 0.04:
                wedge = "up" if dz > 0 else "down"
        bonds.append({
            "a": ai, "b": bi, "order": order, "type": btype,
            "wedge": wedge, "dz": round(float(depth[bi] - depth[ai]), 3),
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
        "groups": groups,
        "symmetry": {
            "classes": sym_classes,
            "mirror_pairs": mirror_pairs,
            "axis": axis,
            "point_group_hint": "C2",
        },
    }


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
