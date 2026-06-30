#!/usr/bin/env python3
"""
inchi_coord.py -- save + check the COORDINATION GEOMETRY of a complex.

Standard InChI deliberately DISCONNECTS metals (it breaks every bond to the
metal), so a bare InChI loses the coordination.  The InChI organometallics work
(v1.07 "Molecular Inorganics") keeps that information; the mechanism already in
the standard software is the RECONNECTED-metal layer, emitted with the `/RecMet`
option as an extra `/r...` layer that restores the metal-ligand bonds.  We use
that as the serialised "save" form.

On top of the string we compute a compact COORDINATION SIGNATURE -- metal, donor
set (with hapticity), coordination number and the geometry class -- which is the
thing a depiction must get right.  `signature_from_mol` derives it from the input
connectivity; `signature_from_harness` re-derives it from the FINAL 2-D drawing;
when they agree, the depiction faithfully represents the saved geometry.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from rdkit import Chem                                  # noqa: E402
from rdkit.Chem import inchi                            # noqa: E402
from rdkit import RDLogger                              # noqa: E402
RDLogger.DisableLog("rdApp.*")

import seed_from_smiles as S                            # noqa: E402

DATIVE = Chem.BondType.DATIVE


def coord_inchi(mol):
    """Reconnected-metal InChI (the `/r` layer preserves the metal-ligand bonds).
    Dative bonds are first made single (InChI cannot read RDKit's dative type) and
    H counts frozen so the high-valence metal does not trip the sanitiser."""
    rw = Chem.RWMol(mol)
    for b in rw.GetBonds():
        if b.GetBondType() == DATIVE:
            b.SetBondType(Chem.BondType.SINGLE)
    for a in rw.GetAtoms():
        a.SetNoImplicit(True)               # freeze H counts; metal stays high-valence
    m = rw.GetMol()
    Chem.SanitizeMol(m, sanitizeOps=(                 # everything EXCEPT kekulize/valence
        Chem.SanitizeFlags.SANITIZE_FINDRADICALS
        | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS), catchErrors=True)
    for opt in ("/RecMet", "/RecMet /FixedH", ""):
        try:
            s = inchi.MolToInchi(m, options=opt, treatWarningAsError=False)
            if s:
                return s
        except Exception:
            continue
    return "(InChI unavailable for this structure)"


# --------------------------------------------------------------------------- #
def _geometry_class(n_sigma, haptic_sizes):
    """Classify the coordination geometry from the donor count and hapticities."""
    nh = len(haptic_sizes)
    if nh >= 2:
        return "metallocene (sandwich)" if n_sigma == 0 else "bent metallocene"
    if nh == 1:
        return "half-sandwich (piano-stool)"
    return {0: "free", 1: "terminal", 2: "linear", 3: "trigonal planar",
            4: "square planar", 5: "square pyramidal", 6: "octahedral",
            7: "pentagonal bipyramidal"}.get(n_sigma, f"{n_sigma}-coordinate")


def signature_from_mol(mol):
    """Coordination signature from the input CONNECTIVITY: metal, donor atoms
    (DATIVE-bonded), eta-n haptic rings, coordination number, geometry class."""
    metal = next((a.GetIdx() for a in mol.GetAtoms()
                  if a.GetSymbol() in S.SEED_METALS), None)
    if metal is None:
        return None
    haptic = S._haptic_rings(mol, metal)
    hap_atoms = {a for h in haptic for a in h}
    # every ligand on the metal counts toward the geometry -- dative donors AND
    # covalent ancillaries (H, Cl, Br, CO, Ph, ...) -- except the eta-n ring atoms.
    donors = [nb.GetSymbol() for nb in mol.GetAtomWithIdx(metal).GetNeighbors()
              if nb.GetIdx() not in hap_atoms]
    hapt = [len(h) for h in haptic]
    return {
        "metal": mol.GetAtomWithIdx(metal).GetSymbol(),
        "donors": sorted(donors),
        "haptic": sorted(f"eta{n}" for n in hapt),
        "n_sigma": len(donors),
        "cn": len(donors) + len(hapt),      # one site per sigma ligand + per eta-n
        "geometry": _geometry_class(len(donors), hapt),
    }


def signature_from_harness(H):
    """Re-derive the coordination signature from the FINAL 2-D depiction: the
    drawn coord (dashed) bonds to sigma donors, the eta-n discs (metal->centroid
    dashed bonds to label-less degree-1 anchors), and the 2-D angles between the
    metal's drawn bonds (used to confirm the geometry class)."""
    m = H.metal
    if m is None:
        return None
    sigma, hapt = [], 0
    for nb in H.adj[m]:
        kind = H.kind.get(frozenset((m, nb)))
        lbl = H.label.get(nb, "")
        if lbl == "" and len(H.adj[nb]) == 1 and kind == "coord":
            hapt += 1                                   # eta-n centroid anchor
        else:
            sigma.append(lbl or "C")                    # every other drawn ligand
    # angular spread of the metal's drawn bonds, to confirm the class
    mx, my = H.pos[m]
    angs = sorted(math.degrees(math.atan2(H.pos[n][1] - my, H.pos[n][0] - mx)) % 360
                  for n in H.adj[m])
    gaps = [round((angs[(i + 1) % len(angs)] - angs[i]) % 360) for i in range(len(angs))] \
        if len(angs) > 1 else []
    return {
        "metal": H.label.get(m),
        "donors": sorted(sigma),
        "haptic": sorted(["eta5"] * hapt) if hapt else [],   # discs drawn as Cp
        "n_sigma": len(sigma),
        "cn": len(sigma) + hapt,            # one site per sigma ligand + per disc
        "geometry": _geometry_class(len(sigma), [5] * hapt),
        "drawn_bond_gaps_deg": gaps,
    }


def _match(a, b):
    """Do two signatures describe the SAME coordination geometry?"""
    if a is None or b is None:
        return False
    return (a["metal"] == b["metal"] and a["geometry"] == b["geometry"]
            and a["cn"] == b["cn"] and a["n_sigma"] == b["n_sigma"])


if __name__ == "__main__":
    import complexes as C
    print(f"{'complex':22s} {'geometry (from mol)':24s} {'(from depiction)':24s} match")
    print("-" * 86)
    for nm in C.ORDER:
        mol = C.FEED[nm]()
        sig_mol = signature_from_mol(mol)
        H = C.render(nm, mol)
        sig_dep = signature_from_harness(H)
        ok = _match(sig_mol, sig_mol and sig_dep)
        gm = sig_mol["geometry"] if sig_mol else "?"
        gd = sig_dep["geometry"] if sig_dep else "?"
        print(f"{nm:22.22s} {gm:24s} {gd:24s} {'OK' if ok else 'differ'}")
        print(f"   InChI: {coord_inchi(mol)[:96]}")
