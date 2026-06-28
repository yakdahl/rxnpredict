#!/usr/bin/env python3
"""
generate.py -- end-to-end pipeline for the (DTBM-SEGPhos)Cu-H depiction.

    Architector/RDKit  ->  3D conformer (.xyz)
    T-REX              ->  metal/ligand/charge/symmetry/chirality descriptor
    RDKit + depict     ->  2D depiction graph (.json + viewer/molecule_data.js)

Run:
    python generate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rdkit import Chem  # noqa: E402

from src import build_complex, chem, depict, trex_bridge  # noqa: E402


def main() -> None:
    data_dir = HERE / "data"
    viewer_dir = HERE / "viewer"
    data_dir.mkdir(exist_ok=True)
    xyz_path = data_dir / "cuh_dtbm_segphos.xyz"

    print("[1/4] building 3D conformer ...")
    complex_mol, info, method = build_complex.generate_conformer(xyz_path)
    print(f"      method={method}  atoms={complex_mol.GetNumAtoms()}  "
          f"charge={info['overall_charge']}  conf={complex_mol.GetNumConformers()}")

    print("[2/4] T-REX descriptor ...")
    lig = chem.make_dtbm_segphos()
    smiles = Chem.MolToSmiles(lig)
    trex_desc = trex_bridge.describe(xyz_path, info["overall_charge"])
    trex_desc["perceived"] = {                       # raw xyz perception (kept for transparency)
        "geometry_name": trex_desc.get("geometry_name"),
        "cn": trex_desc.get("cn"), "ox": trex_desc.get("ox"),
        "trex_string": trex_desc.get("trex_string"),
    }
    trex_desc["canonical"] = trex_bridge.canonical_known_trex(smiles)
    trex_desc["ideal_geometry"] = trex_bridge.ideal_metal_geometry("Cu")
    trex_desc["is_chiral_overall"] = True            # axial (atropisomeric) backbone
    trex_desc["chirality_note"] = "axial chirality (atropisomeric biaryl backbone)"
    can = trex_desc["canonical"]
    print(f"      perceived: {trex_desc.get('geometry_name')} CN{trex_desc.get('cn')} "
          f"ox{trex_desc.get('ox')}  |  canonical: "
          f"{'CN'+str(can.get('cn'))+' '+str(can.get('geometry')) if can.get('available') else 'n/a'}")

    print("[3/4] depiction graph ...")
    depiction = depict.build_depiction(
        complex_mol, info, trex_desc,
        name="(DTBM-SEGPhos)Cu-H", smiles=smiles, method=method,
    )
    print(f"      atoms={len(depiction['atoms'])} bonds={len(depiction['bonds'])} "
          f"groups={len(depiction['groups'])} "
          f"sym_classes={len(depiction['symmetry']['classes'])} "
          f"mirror_pairs={len(depiction['symmetry']['mirror_pairs'])}")

    print("[4/4] writing outputs ...")
    depict.emit(depiction, data_dir, viewer_dir)
    print("done.")


if __name__ == "__main__":
    main()
