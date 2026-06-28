"""
abbreviations.py
================
Detect common organic substituent groups in an RDKit molecule and describe
how to collapse them into single labelled "superatom" nodes (e.g. t-Bu, OMe).

The depiction never *destroys* atoms -- every group records the member atom
indices, the single attachment atom, and a human label.  The web viewer can
then collapse / expand groups at render time, which is what keeps a busy
metal--organic complex readable without throwing information away.

Two nested levels are supported:

  * level 1  -- small, universally-recognised groups:  t-Bu, OMe / MeO,
                CF3, NMe2, OEt, i-Pr ...
  * level 2  -- whole-ligand-arm collapse:  an entire P(aryl)2 phosphine arm
                folds to a "PAr2" node.  Built by the depiction layer, not
                here, because it depends on the metal perception.

Only RDKit is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem


@dataclass
class GroupHit:
    """One detected abbreviation group."""

    gid: str
    label: str
    atoms: list[int]          # all member atom indices (heavy + the methyls etc.)
    attach: int               # the atom OUTSIDE the group it hangs off of
    anchor: int               # the in-group atom bonded to `attach`
    kind: str                 # 'tbu', 'ome', ...
    members: set[int] = field(default_factory=set)


# Each template: (kind, label, SMARTS, attach_query_index, anchor_pattern_index)
#   attach is matched OUTSIDE the SMARTS via the bond from `anchor`.
#   The SMARTS matches only the group itself; the first atom of the SMARTS is
#   the anchor (the atom that bonds to the rest of the molecule).
_TEMPLATES: list[tuple[str, str, str]] = [
    # tert-butyl: quaternary C with three methyls
    ("tbu", "t-Bu", "[CX4D4]([CH3])([CH3])[CH3]"),
    # trifluoromethyl
    ("cf3", "CF3", "[CX4](F)(F)F"),
    # methoxy  (O-CH3), anchor = O
    ("ome", "OMe", "[OX2D2][CH3]"),
    # ethoxy   (O-CH2-CH3)
    ("oet", "OEt", "[OX2D2][CH2][CH3]"),
    # dimethylamino
    ("nme2", "NMe2", "[NX3D3]([CH3])[CH3]"),
    # isopropyl
    ("ipr", "i-Pr", "[CX4D3H]([CH3])[CH3]"),
]


def _anchor_external_neighbor(mol: Chem.Mol, group_atoms: set[int], anchor: int) -> int | None:
    """Return the single neighbour of `anchor` that is not part of the group."""
    a = mol.GetAtomWithIdx(anchor)
    ext = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() not in group_atoms]
    if len(ext) == 1:
        return ext[0]
    return None


def detect_groups(mol: Chem.Mol) -> list[GroupHit]:
    """
    Find all non-overlapping abbreviation groups (level 1).

    Greedy by template order; an atom is claimed by at most one group.
    Returns a list of GroupHit.
    """
    claimed: set[int] = set()
    hits: list[GroupHit] = []
    counter = 0

    for kind, label, smarts in _TEMPLATES:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt, uniquify=True):
            atoms = set(match)
            if atoms & claimed:
                continue
            anchor = match[0]                       # first SMARTS atom = anchor
            attach = _anchor_external_neighbor(mol, atoms, anchor)
            if attach is None:
                continue
            # never collapse onto the metal directly
            if mol.GetAtomWithIdx(attach).GetSymbol() == "Cu":
                continue
            counter += 1
            gid = f"g{counter}_{kind}"
            hits.append(
                GroupHit(
                    gid=gid,
                    label=label,
                    atoms=sorted(atoms),
                    attach=attach,
                    anchor=anchor,
                    kind=kind,
                    members=atoms,
                )
            )
            claimed |= atoms

    return hits


def label_for_direction(label: str, dx: float) -> str:
    """
    Mirror left-pointing oxygen labels so the bond leaves the heteroatom, e.g.
    'OMe' drawn to the left of its ring reads better as 'MeO'.
    """
    flips = {"OMe": "MeO", "OEt": "EtO"}
    if dx < 0 and label in flips:
        return flips[label]
    return label
