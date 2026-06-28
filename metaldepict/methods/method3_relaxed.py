#!/usr/bin/env python3
"""
method3_relaxed.py -- REFINED Method C.

The deterministic Method-C template (method3_template.py) gives correct
connectivity and a good fan-out but leaves a few inter-fragment bond lengths
wrong (P-Cu too long, P-aryl too short).  This driver runs the WINNING physics
relaxer from the 5-method bake-off -- global energy minimisation (relaxer_energy,
L-BFGS-B over the free atom coordinates, rings held rigid) -- with weights that
were black-box-optimised PER LIGAND against the numerical quality judge
(relax_harness.quality_loss, the crowd-aware version).

Result per ligand: uniform bond lengths, regular rings, dashed P->Cu coordination
bonds, Kekule doubles inside the rings, Cu-H horizontal (H to the right), and no
overlap/crowding -- matching the published ChemDraw references.

Run:  python method3_relaxed.py   ->  per-ligand SVG/PNG in out/relaxed/ and a
2x2 montage at ../panels/method_c_refined.png
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import relax_harness as R          # noqa: E402
import relaxer_energy              # noqa: E402

# GLOBAL energy weights (round-2 optimum): generalise to ANY ligand with a
# template builder -- no per-ligand re-optimisation needed.
GLOBAL = {"w_bond": 7.96, "w_angle": 0.61, "w_overlap": 17.49,
          "w_rigid": 67.3, "maxiter": 164}

# Per-ligand energy-term weights, black-box-optimised (scipy differential
# evolution) against the crowd-aware numerical judge, one set per molecule.
# Most ligands are happy with GLOBAL; only the bulkiest (four DTBM aryls in 2D)
# get a dedicated set so the optimiser can push the overlap weight as needed.
PER_LIGAND_WEIGHTS = {
    "dtbm_segphos": {"w_bond": 12.97, "w_angle": 3.95, "w_overlap": 18.55,
                     "w_rigid": 56.86, "maxiter": 176},
    "xantphos":     {"w_bond": 13.10, "w_angle": 0.015, "w_overlap": 17.99,
                     "w_rigid": 36.54, "maxiter": 138},
    "dpephos":      {"w_bond": 15.31, "w_angle": 0.13, "w_overlap": 6.80,
                     "w_rigid": 50.92, "maxiter": 145},
    "ph_bpe":       {"w_bond": 18.84, "w_angle": 0.002, "w_overlap": 15.31,
                     "w_rigid": 12.52, "maxiter": 196},
    # --- DTBM analogues + DPPBz ---
    "dppbz":         GLOBAL,                   # global weights suffice
    "dtbm_dppbz":    GLOBAL,                   # global weights suffice
    "dtbm_xantphos": {"w_bond": 7.16, "w_angle": 2.43, "w_overlap": 4.91,
                      "w_rigid": 51.89, "maxiter": 151},
    "dtbm_dpephos":  {"w_bond": 2.33, "w_angle": 3.35, "w_overlap": 20.66,
                      "w_rigid": 71.75, "maxiter": 185},
}

# the two montages this driver writes (classic four + the new DTBM/DPPBz variants)
GROUPS = {
    "method_c_refined": ["dtbm_segphos", "xantphos", "dpephos", "ph_bpe"],
    "method_c_dtbm_variants": ["dppbz", "dtbm_dppbz", "dtbm_dpephos",
                               "dtbm_xantphos"],
}


def build(key):
    """Return a relaxed Harness for one ligand using its own optimised weights."""
    H = R.Harness(key)
    relaxer_energy.relax(H, PER_LIGAND_WEIGHTS[key])
    return H


def render_group(name, keys):
    outdir = HERE / "out" / "relaxed"
    outdir.mkdir(parents=True, exist_ok=True)
    svgs, pngs = [], []
    for key in keys:
        H = build(key)
        svg = outdir / f"{key}.svg"
        H.save_svg(svg)
        svgs.append(str(svg))
        pngs.append(str(outdir / f"{key}.png"))
        print(f"{key:14s} {H.metrics()}")
    R._svg_to_png(svgs, pngs)

    from PIL import Image, ImageDraw, ImageFont
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)
    pad = 6
    panel = Image.new("RGB", (w * 2 + pad * 3, h * 2 + pad * 3 + 26), "#dddddd")
    d = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    d.text((8, 5), f"Method C (refined): {name}", fill="black", font=font)
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, 2)
        panel.paste(im, (pad + c * (w + pad), 26 + pad + r * (h + pad)))
    dest = HERE.parent / "panels" / f"{name}.png"
    panel.save(dest)
    print("montage ->", dest)


def main():
    for name, keys in GROUPS.items():
        render_group(name, keys)


if __name__ == "__main__":
    main()
