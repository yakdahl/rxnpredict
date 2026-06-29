#!/usr/bin/env python3
"""
gen_matrix.py -- regenerate every bisphosphine backbone with a panel of P
substituents, all drawn at ONE common scale with the Cu-H bond locked in the
same place/orientation (so the images are directly comparable / animatable).

backbone (bare-P SMILES) x substituent (SMILES fragment, attachment = atom 0)
-> RDKit grafts 2 substituents onto each P -> seed_from_smiles -> energy relax
(per-ligand black-box re-optimised only when the global weights leave overlaps)
-> fixed-scale render with Cu pinned to a constant canvas pixel.

Writes one PNG per cell to ../panels/matrix/<backbone>__<sub>.png and records the
grid in matrix_index.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from rdkit import Chem                          # noqa: E402

import relax_harness as R                       # noqa: E402
import relaxer_energy                           # noqa: E402
import seed_from_smiles as S                    # noqa: E402

GLOBAL = {"w_bond": 7.96, "w_angle": 0.61, "w_overlap": 17.49,
          "w_rigid": 67.3, "maxiter": 164}

# ---- the grid axes (order fixes the animation layout) ----------------------
BACKBONES = {                       # bare-P cores (each P gets 2 substituents)
    "segphos":  "C1Oc2ccc(P)c(-c3c(P)ccc4OCOc34)c2O1",
    "xantphos": "CC1(C)c2cccc(P)c2Oc2c(P)cccc21",
    "dpephos":  "O(c1ccccc1P)c1ccccc1P",
    "dppbz":    "c1ccc(P)c(P)c1",
    "binap":    "Pc1ccc2ccccc2c1-c1c(P)ccc2ccccc12",
    "naphthyl": "Pc1cccc2cccc(P)c12",
    "dppe":     "PCCP",
    "dppm":     "PCP",
}
SUBSTITUENTS = {                    # attachment atom = atom 0
    "dtbm":   "c1cc(C(C)(C)C)c(OC)c(C(C)(C)C)c1",
    "biscf3": "c1cc(C(F)(F)F)cc(C(F)(F)F)c1",
    "xylyl":  "c1cc(C)cc(C)c1",
    "furyl":  "c1ccco1",
    "octylthienyl": "c1sccc1CCCCCCCC",
    "cyclohexyl": "C1CCCCC1",
    "tbu":    "C(C)(C)C",
    "methyl": "C",
}
PRETTY = {"segphos": "SEGPhos", "xantphos": "Xantphos", "dpephos": "DPEphos",
          "dppbz": "DPPBz", "binap": "BINAP", "naphthyl": "1,8-naphthyl",
          "dppe": "DPPE", "dppm": "DPPM",
          "dtbm": "DTBM", "biscf3": "3,5-(CF3)2C6H3", "xylyl": "3,5-Me2C6H3",
          "furyl": "2-furyl", "octylthienyl": "3-octyl-2-thienyl",
          "cyclohexyl": "Cy", "tbu": "t-Bu", "methyl": "Me"}

# ---- fixed render frame: every image identical canvas + scale + Cu anchor ----
CANVAS = (1180, 980)
SCALE = 44.0                        # px per world bond-length L
ANCHOR_PX = (740.0, 490.0)          # where Cu sits on every canvas


def _graft(mol, p_idx, frag):
    combo = Chem.CombineMols(mol, frag)
    rw = Chem.RWMol(combo)
    rw.AddBond(p_idx, mol.GetNumAtoms(), Chem.BondType.SINGLE)  # frag atom0
    return rw.GetMol()


def ligand_smiles(core_smiles, sub_smiles):
    core = Chem.MolFromSmiles(core_smiles)
    frag = Chem.MolFromSmiles(sub_smiles)
    if core is None or frag is None:
        raise ValueError("bad core/frag SMILES")
    p_idxs = [a.GetIdx() for a in core.GetAtoms() if a.GetSymbol() == "P"]
    mol = core
    for p in p_idxs:
        for _ in range(2):
            mol = _graft(mol, p, frag)
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)


def _score(H):
    m = H.metrics()
    return (m["overlap"], round(m.get("crowd", 0.0), 2), round(m["bondCV"], 3))


def best_relax(fresh, allow_de=True):
    """Try several layout strategies on a freshly-built Harness (fresh() returns
    a new H each call) and keep the one with the fewest overlaps (then least
    crowding).  Cheap strategies first; per-scene weight re-optimisation only if
    overlaps remain."""
    cands = []

    def add(H):
        cands.append(H)
        return _score(H)

    H = fresh()
    relaxer_energy.relax(H, GLOBAL)
    if add(H)[0] == 0:
        return min(cands, key=_score)
    # strong overlap + aggressive declutter
    BIG = (12, -12, 24, -24, 36, -36, 50, -50, 68, -68, 85, -85)
    H = fresh()
    relaxer_energy.relax(H, dict(GLOBAL, w_overlap=30.0, w_rigid=72.0, maxiter=200))
    R.declutter(H, angles=BIG, passes=3)
    add(H)
    # + a rigid-body overlap push, then declutter again
    H = fresh()
    relaxer_energy.relax(H, dict(GLOBAL, w_overlap=34.0, w_rigid=72.0, maxiter=200))
    R.declutter(H, angles=BIG, passes=3)
    bo, pin, inv, tr, _ = R.body_helpers(H)
    R.overlap_relax(H, bo, pin, inv, tr, w_over=0.6, iters=140)
    R.declutter(H, angles=BIG, passes=2)
    add(H)
    # per-scene differential-evolution of the energy weights for the stubborn ones
    if allow_de and min(_score(c)[0] for c in cands) > 2:
        from scipy.optimize import differential_evolution

        def obj(x):
            h = fresh()
            relaxer_energy.relax(h, dict(zip(relaxer_energy.PARAM_NAMES, x)))
            return R.quality_loss(h)
        try:
            res = differential_evolution(obj, relaxer_energy.BOUNDS, maxiter=8,
                                         seed=0, popsize=8, polish=False, tol=1e-3)
            H = fresh()
            relaxer_energy.relax(H, dict(zip(relaxer_energy.PARAM_NAMES, res.x)))
            R.declutter(H, angles=BIG, passes=3)
            add(H)
        except Exception:
            pass
    return min(cands, key=_score)


def relaxed_cell(backbone, sub):
    smi = ligand_smiles(BACKBONES[backbone], SUBSTITUENTS[sub])

    def fresh():
        sc, _ = S.scene_from_smiles(smi, f"{backbone}_{sub}")
        return R.Harness(scene=sc, title=f"{backbone}_{sub}")
    H = best_relax(fresh)
    return H, smi, GLOBAL


def render_cell(backbone, sub, outdir):
    H, smi, weights = relaxed_cell(backbone, sub)
    H.commit()
    title = f"({PRETTY[sub]}){PRETTY[backbone]}·CuH"
    svg = H.scene.render_svg(CANVAS[0], CANVAS[1], title=title,
                             fixed_scale=SCALE,
                             anchor_world=H.pos[H.metal], anchor_px=ANCHOR_PX)
    sp = outdir / f"{backbone}__{sub}.svg"
    sp.write_text(svg)
    m = H.metrics()
    return str(sp), str(outdir / f"{backbone}__{sub}.png"), m


def main():
    outdir = HERE.parent / "panels" / "matrix"
    outdir.mkdir(parents=True, exist_ok=True)
    svgs, pngs, index = [], [], []
    for bk in BACKBONES:
        for sb in SUBSTITUENTS:
            try:
                s, p, m = render_cell(bk, sb, outdir)
                svgs.append(s)
                pngs.append(p)
                index.append({"backbone": bk, "sub": sb, "png": p,
                              "overlap": m["overlap"], "bondCV": m["bondCV"]})
                print(f"{bk:10s} {sb:13s} overlap={m['overlap']} "
                      f"bondCV={m['bondCV']:.3f}", flush=True)
            except Exception as e:
                print(f"{bk:10s} {sb:13s} ERROR {e}", flush=True)
    R._svg_to_png(svgs, pngs)
    (HERE.parent / "panels" / "matrix_index.json").write_text(
        json.dumps({"backbones": list(BACKBONES), "subs": list(SUBSTITUENTS),
                    "cells": index}, indent=2))
    print(f"done: {len(pngs)} cells")


if __name__ == "__main__":
    main()
