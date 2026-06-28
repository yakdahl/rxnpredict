#!/usr/bin/env python3
"""
gen_panel.py -- generalised generator: build the depiction for any bidentate
bisphosphine (ligated as its Cu-H complex) and write one data file per ligand
that the viewer can load with ?mol=<name>.

Demonstrates that the pipeline is NOT hard-coded to SEGPhos: DTBM-SEGPhos,
Xantphos, DPEphos and Ph-BPE all run through the same code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rdkit import Chem  # noqa: E402

from src import build_complex, chem, depict, trex_bridge  # noqa: E402

LIGANDS = {
    "dtbm_segphos": ("(DTBM-SEGPhos)Cu-H", None),
    "xantphos": ("(Xantphos)Cu-H",
                 "CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21"),
    "dpephos": ("(DPEphos)Cu-H",
                "O(c1ccccc1P(c1ccccc1)c1ccccc1)c1ccccc1P(c1ccccc1)c1ccccc1"),
    "ph_bpe": ("(Ph-BPE)Cu-H",
               "C(CP1C(c2ccccc2)CCC1c1ccccc1)P1C(c2ccccc2)CCC1c1ccccc1"),
}


def build_one(key: str, name: str, smi: str | None) -> dict:
    lig = chem.make_dtbm_segphos() if smi is None else Chem.MolFromSmiles(smi)
    Chem.SanitizeMol(lig)
    xyz = HERE / "data" / f"{key}.xyz"
    complex_mol, info, method = build_complex.generate_conformer(xyz, lig=lig)
    smiles = Chem.MolToSmiles(lig)
    trex_desc = trex_bridge.describe(xyz, info["overall_charge"])
    trex_desc["canonical"] = trex_bridge.canonical_known_trex(smiles)
    trex_desc["is_chiral_overall"] = depict._find_biaryl_bond(complex_mol) is not None
    dep = depict.build_depiction(complex_mol, info, trex_desc,
                                 name=name, smiles=smiles, method=method)
    return dep


def main() -> None:
    data_dir = HERE / "data"
    vdata = HERE / "viewer" / "data"
    data_dir.mkdir(exist_ok=True)
    vdata.mkdir(parents=True, exist_ok=True)
    index = []
    for key, (name, smi) in LIGANDS.items():
        print(f"[build] {key} ...")
        dep = build_one(key, name, smi)
        (data_dir / f"{key}.json").write_text(json.dumps(dep, indent=2))
        (vdata / f"{key}.js").write_text(
            f"// auto-generated -- window.MOLECULE for {key}\n"
            "window.MOLECULE = " + json.dumps(dep) + ";\n")
        ng = len([g for g in dep["groups"] if g["level"] == 1])
        print(f"        atoms={len(dep['atoms'])} medium-groups={ng} "
              f"formula={dep['meta']['formula']}")
        index.append({"key": key, "name": name})
    (vdata / "index.json").write_text(json.dumps(index, indent=2))
    print("done:", ", ".join(i["key"] for i in index))


if __name__ == "__main__":
    main()
