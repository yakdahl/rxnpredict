#!/usr/bin/env python3
"""
gen_named.py -- "my version" of a set of NAMED reference ligands (the ones in the
two reference figures), each rendered through the same pipeline at the locked
Cu-H scale, then montaged to match the reference layout.

Most come straight from SMILES; QuinoxP* / Me-DuPhos are P- or C-stereogenic
backbones; the Josiphos SL-J011-1 and DPPF-type ferrocenes use the hardcoded
sandwich; DM-/DTBM- variants graft the substituent onto a backbone core.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from rdkit import Chem                          # noqa: E402

import relax_harness as R                       # noqa: E402
import relaxer_energy                           # noqa: E402
import seed_from_smiles as S                    # noqa: E402
import gen_matrix as G                          # noqa: E402
from src import chem                            # noqa: E402

GLOBAL = G.GLOBAL
CANVAS, SCALE, ANCHOR = (1180, 980), 44.0, (740.0, 490.0)


def _smiles_H(smi, name):
    def fresh():
        sc, _ = S.scene_from_smiles(smi, name)
        return R.Harness(scene=sc, title=name)
    return G.best_relax(fresh)


def _frozen_H(builder):
    sc, title, meta = builder()
    H = R.Harness(scene=sc, title=title,
                  exempt=meta["exempt"], extra_rigid=meta["extra_rigid"])
    relaxer_energy.relax(H, GLOBAL)
    if H.metrics()["overlap"] > 0:
        R.declutter(H)
    return H


def _slj011():
    # Josiphos SL-J011-1: P(4-CF3-C6H4)2 directly on Cp, CH(CH3)-PtBu2 tether.
    p1 = lambda s, pid: (S._aryl_para(s, pid, 205, "CF3"),
                         S._aryl_para(s, pid, 255, "CF3"))
    p2 = lambda s, pid: (S._arm(s, pid, -30, "t-Bu"),
                         S._arm(s, pid, 30, "t-Bu"))
    return S.build_josiphos("SL-J011-1", psub_p1=p1, psub_p2=p2)


def named():
    out = {}
    # ---- figure 1 ----
    out["DTBM-SEGPHOS"] = _smiles_H(Chem.MolToSmiles(chem.make_dtbm_segphos()),
                                    "DTBM-SEGPHOS")
    out["(S,S)-Ph-BPE"] = _smiles_H(
        "C(C[P]1[C@H](c2ccccc2)CC[C@H]1c1ccccc1)[P]1[C@H]"
        "(c2ccccc2)CC[C@H]1c1ccccc1", "(S,S)-Ph-BPE")
    out["DM-SEGPHOS"] = _smiles_H(
        G.ligand_smiles(G.BACKBONES["segphos"], "c1cc(C)cc(C)c1"), "DM-SEGPHOS")
    out["(S,S)-QuinoxP*"] = _smiles_H(
        "C[P@@](C(C)(C)C)c1nc2ccccc2nc1[P@@](C)C(C)(C)C", "(S,S)-QuinoxP*")
    out["(R,R)-Me-DuPhos"] = _smiles_H(
        "C[C@@H]1CC[C@@H](C)[P]1c1ccccc1[P]1[C@H](C)CC[C@H]1C", "(R,R)-Me-DuPhos")
    out["SL-J011-1"] = _frozen_H(_slj011)
    out["DTBM-BINAP"] = _smiles_H(
        G.ligand_smiles(G.BACKBONES["binap"], G.SUBSTITUENTS["dtbm"]),
        "DTBM-BINAP")
    # ---- figure 2 ----
    out["(3,5-tBu2C6H3)-DPPBz"] = _smiles_H(
        G.ligand_smiles(G.BACKBONES["dppbz"], "c1cc(C(C)(C)C)cc(C(C)(C)C)c1"),
        "(3,5-tBu2C6H3)2P-DPPBz")
    out["(3-Oct-thienyl)-DPPBz"] = _smiles_H(
        G.ligand_smiles(G.BACKBONES["dppbz"], G.SUBSTITUENTS["octylthienyl"]),
        "(3-octyl-2-thienyl)-DPPBz")
    out["DTBM-DPPBz"] = _smiles_H(
        G.ligand_smiles(G.BACKBONES["dppbz"], G.SUBSTITUENTS["dtbm"]),
        "DTBM-DPPBz")
    out["DTBM-DPEPhos"] = _smiles_H(
        G.ligand_smiles(G.BACKBONES["dpephos"], G.SUBSTITUENTS["dtbm"]),
        "DTBM-DPEPhos")
    return out


FIG1 = ["DTBM-SEGPHOS", "(S,S)-Ph-BPE", "DM-SEGPHOS",
        "(S,S)-QuinoxP*", "(R,R)-Me-DuPhos", "SL-J011-1", "DTBM-BINAP"]
FIG2 = ["(3,5-tBu2C6H3)-DPPBz", "(3-Oct-thienyl)-DPPBz",
        "DTBM-DPPBz", "DTBM-DPEPhos"]


def main():
    outdir = HERE / "out" / "named"
    outdir.mkdir(parents=True, exist_ok=True)
    Hs = named()
    svgs, pngs, label = [], [], {}
    for nm, H in Hs.items():
        H.commit()
        svg = H.scene.render_svg(*CANVAS, title=nm, fixed_scale=SCALE,
                                 anchor_world=H.pos[H.metal], anchor_px=ANCHOR)
        sp = outdir / f"{nm}.svg"
        sp.write_text(svg)
        svgs.append(str(sp))
        pngs.append(str(outdir / f"{nm}.png"))
        label[nm] = str(outdir / f"{nm}.png")
        print(f"{nm:26s} overlap={H.metrics()['overlap']}")
    R._svg_to_png(svgs, pngs)
    _montage([label[n] for n in FIG1], 4, HERE.parent / "panels" / "named_fig1.png")
    _montage([label[n] for n in FIG2], 2, HERE.parent / "panels" / "named_fig2.png")
    print("done")


def _montage(pngs, cols, dest):
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    rows = (len(imgs) + cols - 1) // cols
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)
    g = Image.new("RGB", (w * cols, h * rows), "white")
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        g.paste(im, (c * w, r * h))
    g.save(dest)
    print("montage ->", dest)


if __name__ == "__main__":
    main()
