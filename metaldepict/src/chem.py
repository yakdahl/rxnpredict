"""
chem.py
=======
Build the parent SEGPhos ligand, decorate it into DTBM-SEGPhos by *graph
substitution* (never string surgery -- that is how ring-closure digits collide
and produce spurious phosphole rings), and assemble the CuH complex.

DTBM-SEGPhos = 5,5'-bis[di(3,5-di-tert-butyl-4-methoxyphenyl)phosphino]-
               4,4'-bi-1,3-benzodioxole          (C74 H100 O8 P2, MW 1187.5)

For every P-bound phenyl ring:
  * ipso  = ring carbon bonded to phosphorus
  * meta  (the two 3,5 positions) -> tert-butyl
  * para  (the 4 position)        -> methoxy
Each phenyl attaches to exactly one phosphorus at exactly one point.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

# Verified parent (S)-SEGPhos connectivity.  Ring-closure digits are chosen so
# nothing collides: 1,2 for the two benzodioxoles' methylenedioxy/benzene,
# 3,4 for the four phenyls, 5,6 for the second benzodioxole.
SEGPHOS_PARENT_SMILES = (
    "C1Oc2ccc(P(c3ccccc3)c3ccccc3)c(-c3c(P(c4ccccc4)c4ccccc4)ccc4OCOc34)c2O1"
)


def build_segphos_parent() -> Chem.Mol:
    mol = Chem.MolFromSmiles(SEGPHOS_PARENT_SMILES)
    if mol is None:
        raise ValueError("parent SEGPhos SMILES failed to parse")
    Chem.SanitizeMol(mol)
    return mol


def _ring_atoms_containing(mol: Chem.Mol, atom_idx: int) -> list[int]:
    """The 6-membered aromatic all-carbon ring that contains `atom_idx`."""
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if atom_idx in ring and len(ring) == 6:
            if all(mol.GetAtomWithIdx(a).GetSymbol() == "C"
                   and mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
                return list(ring)
    return []


def _classify_phenyl_positions(mol: Chem.Mol, p_idx: int, ipso: int,
                               ring: list[int]) -> tuple[list[int], int]:
    """
    Within a benzene `ring`, given the ipso carbon (bonded to P), return
    (meta_indices, para_index) by ring-graph distance from ipso.
    """
    ring_set = set(ring)
    # adjacency restricted to the ring
    adj = {a: [] for a in ring}
    for a in ring:
        for n in mol.GetAtomWithIdx(a).GetNeighbors():
            if n.GetIdx() in ring_set:
                adj[a].append(n.GetIdx())
    # BFS distances from ipso
    dist = {ipso: 0}
    frontier = [ipso]
    while frontier:
        nxt = []
        for a in frontier:
            for b in adj[a]:
                if b not in dist:
                    dist[b] = dist[a] + 1
                    nxt.append(b)
        frontier = nxt
    meta = [a for a, d in dist.items() if d == 2]
    para = [a for a, d in dist.items() if d == 3]
    return meta, para[0]


def _add_tert_butyl(rw: Chem.RWMol, ring_c: int) -> None:
    cq = rw.AddAtom(Chem.Atom(6))              # quaternary carbon
    rw.AddBond(ring_c, cq, Chem.BondType.SINGLE)
    for _ in range(3):
        cm = rw.AddAtom(Chem.Atom(6))          # methyl
        rw.AddBond(cq, cm, Chem.BondType.SINGLE)


def _add_methoxy(rw: Chem.RWMol, ring_c: int) -> None:
    o = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(ring_c, o, Chem.BondType.SINGLE)
    cm = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(o, cm, Chem.BondType.SINGLE)


def make_dtbm_segphos() -> Chem.Mol:
    """Decorate parent SEGPhos -> DTBM-SEGPhos via graph substitution."""
    parent = build_segphos_parent()
    rw = Chem.RWMol(parent)

    p_atoms = [a.GetIdx() for a in parent.GetAtoms() if a.GetSymbol() == "P"]
    assert len(p_atoms) == 2, f"expected 2 P, found {len(p_atoms)}"

    # collect the four (P, ipso, ring) phenyls first (indices are stable until we edit)
    phenyls: list[tuple[int, int, list[int]]] = []
    for p in p_atoms:
        for nb in parent.GetAtomWithIdx(p).GetNeighbors():
            if nb.GetSymbol() != "C" or not nb.GetIsAromatic():
                continue
            ring = _ring_atoms_containing(parent, nb.GetIdx())
            # a P-phenyl ring: all six carbons, exactly one (ipso) bonded to P
            if not ring:
                continue
            ext = sum(
                1
                for a in ring
                for q in parent.GetAtomWithIdx(a).GetNeighbors()
                if q.GetIdx() not in ring
            )
            if ext == 1:  # only the ipso->P bond leaves the ring  => plain phenyl
                phenyls.append((p, nb.GetIdx(), ring))

    assert len(phenyls) == 4, f"expected 4 P-phenyl rings, found {len(phenyls)}"

    for p, ipso, ring in phenyls:
        meta, para = _classify_phenyl_positions(parent, p, ipso, ring)
        assert len(meta) == 2, f"expected 2 meta carbons, got {meta}"
        for m in meta:
            _add_tert_butyl(rw, m)
        _add_methoxy(rw, para)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def build_complex(lig: Chem.Mol | None = None) -> tuple[Chem.Mol, dict]:
    """
    Assemble (DTBM-SEGPhos)Cu-H:
      * dative P -> Cu bonds (the two phosphines chelate)
      * one covalent Cu-H (hydride)
    Returns (mol, info) where info records key atom indices and overall charge.
    """
    if lig is None:
        lig = make_dtbm_segphos()
    rw = Chem.RWMol(lig)
    p_idx = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "P"]

    cu = rw.AddAtom(Chem.Atom(29))
    h = rw.AddAtom(Chem.Atom(1))
    rw.AddBond(cu, h, Chem.BondType.SINGLE)
    for p in p_idx:
        rw.AddBond(p, cu, Chem.BondType.DATIVE)   # P: -> Cu

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    info = {
        "metal": cu,
        "hydride": h,
        "donors": p_idx,
        "overall_charge": Chem.GetFormalCharge(mol),
    }
    return mol, info


if __name__ == "__main__":
    from rdkit.Chem import rdMolDescriptors

    dtbm = make_dtbm_segphos()
    print("DTBM-SEGPhos formula:", rdMolDescriptors.CalcMolFormula(dtbm))
    cx, info = build_complex(dtbm)
    print("complex formula:", rdMolDescriptors.CalcMolFormula(cx), "info:", info)
