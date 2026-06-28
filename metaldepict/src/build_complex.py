"""
build_complex.py
================
Generate a *sensible 3D conformer* of (DTBM-SEGPhos)Cu-H.

Preferred path: LANL **Architector** (purpose-built for inorganic / organometallic
conformer generation -- proper metal coordination geometry, then GFN2-xTB relax).
Architector needs xtb + openbabel, which are only available through a conda
environment, so the call is guarded: if Architector cannot be imported we fall
back to an **RDKit ETKDG + UFF** embed, which is enough to give every atom a
believable out-of-plane position for wedge/dash perception.

The 3D conformer is *not* used for the 2D layout itself (that comes from a clean
depiction layout + the physics engine).  It is used for:
  * stereochemistry  -> wedge / dash perception
  * feeding T-REX     -> metal / ligand / charge / symmetry descriptor
"""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from . import chem as _chem  # noqa  (works when imported as a package)


def _ligand_donor_indices(lig: Chem.Mol) -> list[int]:
    return [a.GetIdx() for a in lig.GetAtoms() if a.GetSymbol() == "P"]


def try_architector(lig_smiles: str, donor_idx: list[int], metal_ox: int = 1):
    """
    Build the complex with Architector if it is importable.

    Returns an ASE Atoms object on success, else None.  This is the canonical,
    chemistry-aware generator the user asked for; it runs whenever the package
    (and its xtb/openbabel backends) is present.
    """
    try:
        from architector import build_complex  # type: ignore
    except Exception:
        return None

    input_dict = {
        "core": {"metal": "Cu", "coreCN": 3},          # 3-coordinate Cu(I)
        "ligands": [
            {"smiles": lig_smiles, "coordList": donor_idx, "ligType": "bi_cis"},
            {"smiles": "[H-]", "coordList": [0], "ligType": "mono"},
        ],
        "parameters": {
            "metal_ox": metal_ox,
            "n_conformers": 1,
            "relax": True,
            "full_method": "GFN2-xTB",
            "assemble_method": "GFN2-xTB",
        },
    }
    try:
        out = build_complex(input_dict)
        key = next(iter(out))           # lowest-energy structure
        return out[key]["ase_atoms"]
    except Exception as exc:            # pragma: no cover - depends on xtb
        print(f"[architector] build failed, falling back: {exc}")
        return None


def embed_rdkit(complex_mol: Chem.Mol) -> Chem.Mol:
    """ETKDG embed + UFF relax of the full complex; robust fallbacks."""
    mol = Chem.AddHs(complex_mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    params.useRandomCoords = True
    params.maxIterations = 2000
    cid = AllChem.EmbedMolecule(mol, params)
    if cid != 0:
        # retry with plain distance geometry
        cid = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=7)
    if cid == 0:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass
    return mol


def generate_conformer(
    out_xyz: str | Path,
    lig: Chem.Mol | None = None,
) -> tuple[Chem.Mol, dict, str]:
    """
    Returns (complex_mol_3d, info, method).  Writes an .xyz file to `out_xyz`.

    `lig` is any bidentate-phosphine ligand mol (default: DTBM-SEGPhos); the two
    P atoms are chelated to Cu-H.  `complex_mol_3d` is the heavy-atom complex
    (explicit Cu and hydride H, other H implicit) carrying a 3D conformer in
    conformer id 0, so atom indices line up with the depiction graph.
    """
    if lig is None:
        lig = _chem.make_dtbm_segphos()
    lig_smiles = Chem.MolToSmiles(lig)
    complex_mol, info = _chem.build_complex(lig)

    # Always give complex_mol an index-aligned 3D conformer via RDKit, so the
    # depiction's out-of-plane depth lines up with atom ids 0..N-1.  (AddHs
    # appends H after the originals, so 0..N-1 match.)
    molH = embed_rdkit(complex_mol)
    if molH.GetNumConformers():
        src = molH.GetConformer()
        conf = Chem.Conformer(complex_mol.GetNumAtoms())
        for i in range(complex_mol.GetNumAtoms()):
            conf.SetAtomPosition(i, src.GetAtomPosition(i))
        complex_mol.RemoveAllConformers()
        complex_mol.AddConformer(conf, assignId=True)

    # Preferred conformer for the xyz fed to T-REX perception: Architector.
    # Donor indices must be in the atom space Architector actually parses
    # (the SMILES), NOT the pre-canonicalisation RWMol.
    method = "rdkit-etkdg"
    smi_mol = Chem.MolFromSmiles(lig_smiles)
    donor_idx_smiles = [a.GetIdx() for a in smi_mol.GetAtoms() if a.GetSymbol() == "P"]
    ase_atoms = try_architector(lig_smiles, donor_idx_smiles, metal_ox=1)
    if ase_atoms is not None:
        method = "architector-gfn2xtb"
        from io import StringIO

        buf = StringIO()
        ase_atoms.write(buf, format="xyz")
        Path(out_xyz).write_text(buf.getvalue())
    else:
        Chem.MolToXYZFile(molH, str(out_xyz))
    return complex_mol, info, method


if __name__ == "__main__":
    m, info, method = generate_conformer("/tmp/cuh.xyz")
    print("method:", method, "| atoms:", m.GetNumAtoms(), "| info:", info,
          "| has conf:", m.GetNumConformers())
