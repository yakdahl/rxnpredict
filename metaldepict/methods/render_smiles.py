#!/usr/bin/env python3
"""
render_smiles.py -- ONE entry point to depict any Cu-H bisphosphine, with NO
hardcoded backbone: give it a SMILES (or a registry name) and it builds the
template seed from the molecular graph, relaxes it with the optimised energy
physics engine, and writes a ChemDraw-style PNG.

  # a single arbitrary ligand from SMILES:
  python render_smiles.py "O(c1ccccc1P(c1ccccc1)c1ccccc1)c1ccccc1P(c1ccccc1)c1ccccc1" dpephos

  # the built-in demo set -> two montages under ../panels/:
  python render_smiles.py

Add a ligand by dropping a SMILES into LIGANDS -- nothing else to write.
Ferrocene ligands (DPPF ...) use a hardcoded sandwich fragment (RDKit can't lay
out an eta5 metallocene); pass them by registry name.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import relax_harness as R                       # noqa: E402
import relaxer_energy                           # noqa: E402
import seed_from_smiles as S                    # noqa: E402
import method3_template as MT                   # noqa: E402
from src import chem                            # noqa: E402

GLOBAL = {"w_bond": 7.96, "w_angle": 0.61, "w_overlap": 17.49,
          "w_rigid": 67.3, "maxiter": 164}

# name -> SMILES (any bidentate bisphosphine).  DTBM-SEGPhos comes from the
# graph-built mol so we don't have to type its 160-char SMILES.
LIGANDS = {
    "segphos":  chem.SEGPHOS_PARENT_SMILES,
    "xantphos": "CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21",
    "dpephos":  "O(c1ccccc1P(c1ccccc1)c1ccccc1)c1ccccc1P(c1ccccc1)c1ccccc1",
    "ph_bpe":   "C(C[P@]1[C@@H](c2ccccc2)CC[C@@H]1c1ccccc1)[P@]1[C@@H]"
                "(c2ccccc2)CC[C@@H]1c1ccccc1",
    "dppbz":    "c1ccc(P(c2ccccc2)c2ccccc2)c(P(c2ccccc2)c2ccccc2)c1",
}
# ferrocene-backbone ligands: (registry tag -> P-substituent placer)
FERROCENES = {
    "dppf":      None,                  # PPh2 on each Cp (1,1')
    "dtbm_dppf": MT._pphos_dtbm,
}
# per-ligand weight overrides for the few sterically crowded cases (optional)
OVERRIDES = {
    "dtbm_dppf": {"w_bond": 5.0, "w_angle": 1.0, "w_overlap": 26.0,
                  "w_rigid": 70.0, "maxiter": 190},
}


def relaxed(name, smiles=None, weights=None):
    """Return a relaxed Harness for `name`: from SMILES, the registry, or a
    ferrocene tag.  weights default to the global optimum (+ any OVERRIDE)."""
    weights = weights or OVERRIDES.get(name, GLOBAL)
    if name == "josiphos":
        sc, title, meta = S.build_josiphos(name)
        H = R.Harness(scene=sc, title=title,
                      exempt=meta["exempt"], extra_rigid=meta["extra_rigid"])
    elif name in FERROCENES or (smiles is None and name in FERROCENES):
        sc, title, meta = S.build_ferrocene_bisphosphine(name, FERROCENES[name])
        H = R.Harness(scene=sc, title=title,
                      exempt=meta["exempt"], extra_rigid=meta["extra_rigid"])
    else:
        smi = smiles or LIGANDS.get(name)
        if smi is None:
            # DTBM-SEGPhos convenience
            if name == "dtbm_segphos":
                from rdkit import Chem
                smi = Chem.MolToSmiles(chem.make_dtbm_segphos())
            else:
                raise KeyError(f"unknown ligand {name!r}; pass a SMILES")
        sc, title = S.scene_from_smiles(smi, name)
        H = R.Harness(scene=sc, title=title)
    relaxer_energy.relax(H, weights)
    return H


def render(name, smiles=None, out_png=None, weights=None):
    H = relaxed(name, smiles, weights)
    out_dir = HERE / "out" / "smiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    svg = out_dir / f"{name}.svg"
    H.save_svg(svg)
    png = out_png or str(out_dir / f"{name}.png")
    R._svg_to_png([str(svg)], [png])
    print(f"{name:14s} {H.metrics()}  -> {png}")
    return png


def _montage(pngs, dest, title):
    from PIL import Image, ImageDraw, ImageFont
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    cols = 2
    rows = (len(imgs) + cols - 1) // cols
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)
    pad = 6
    panel = Image.new("RGB", (w * cols + pad * (cols + 1),
                              h * rows + pad * (rows + 1) + 26), "#dddddd")
    d = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    d.text((8, 5), title, fill="black", font=font)
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, cols)
        panel.paste(im, (pad + c * (w + pad), 26 + pad + r * (h + pad)))
    panel.save(dest)
    print("montage ->", dest)


def main():
    panels = HERE.parent / "panels"
    smi_pngs = [render(n) for n in LIGANDS]
    _montage(smi_pngs, panels / "smiles_driven.png",
             "SMILES-driven (no hardcoded backbone) + energy relax")
    fc_pngs = [render(n) for n in FERROCENES]
    _montage(fc_pngs, panels / "ferrocene.png",
             "Ferrocene ligands (hardcoded sandwich) + energy relax")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] not in ("main", ""):
        smi = sys.argv[1]
        nm = sys.argv[2] if len(sys.argv) > 2 else "ligand"
        render(nm, smiles=smi, out_png=f"/tmp/{nm}.png")
    else:
        main()
