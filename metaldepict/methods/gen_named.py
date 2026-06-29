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


_CACHE = {}


def _smiles_H(smi, name, p_tetra=False):
    def fresh():
        sc, _ = S.scene_from_smiles(smi, name)
        return R.Harness(scene=sc, title=name, p_tetra=p_tetra)
    H, _ = G.best_relax(fresh, key=f"named::{name}", cache=_CACHE)
    return H


def _frozen_H(builder, key):
    def fresh():
        sc, title, meta = builder()
        return R.Harness(scene=sc, title=title,
                         exempt=meta["exempt"], extra_rigid=meta["extra_rigid"])
    H, _ = G.best_relax(fresh, key=f"named::{key}", cache=_CACHE)
    return H


def _extend_p_leaves(H, out_factor=1.7, in_factor=0.82):
    """Rescale leaf substituents on each P donor along their P-arm: the arm that
    carries the stereo WEDGE / DASH is pushed OUT (out_factor) so the stereo is
    clearly visible, while a plain arm (e.g. the methyl) is pulled IN (in_factor)
    so it does not stick out.  Ring substituents and the metal are left alone."""
    for p in H.donors:
        px, py = H.pos[p]
        for n in H.adj[p]:
            if H.label[n] == "Cu" or any(n in r for r in H.rings):
                continue
            kind = H.kind.get(frozenset((p, n)), "plain")
            factor = out_factor if kind in ("wedge", "dash") else in_factor
            comp, stack = {n}, [n]
            while stack:
                x = stack.pop()
                for y in H.adj[x]:
                    if y == p or y in comp:
                        continue
                    comp.add(y)
                    stack.append(y)
            dx = (H.pos[n][0] - px) * (factor - 1.0)
            dy = (H.pos[n][1] - py) * (factor - 1.0)
            for a in comp:
                H.pos[a] = (H.pos[a][0] + dx, H.pos[a][1] + dy)


def _quinoxp():
    # (S,S)-QuinoxP*: two P-stereocentres -- RDKit wedges the P-tBu bond on one
    # centre and dashes it on the other (the S,S display).  Push the t-Bu (wedge/
    # dash) arms OUT so the stereo is clear, and pull the plain methyls back IN.
    H = _smiles_H("C[P@@](C(C)(C)C)c1nc2ccccc2nc1[P@@](C)C(C)(C)C",
                  "(S,S)-QuinoxP*")
    _extend_p_leaves(H, out_factor=1.7, in_factor=0.8)
    return H


def _slj011():
    # Josiphos SL-J011-2 motif: P(3,5-(CF3)2-C6H3)2 directly on Cp, CH(CH3)-PtBu2
    # tether.  Bis-3,5-CF3 aryls shrunk to ~0.82 to match the ferrocene Cp rings
    # and fanned wide (down-left / down-right) so they do not collide; the two
    # t-Bu fan up-and-right, away from Cu.
    p1 = lambda s, pid: (S._aryl_35(s, pid, 205, "CF3", ring_scale=0.82),
                         S._aryl_35(s, pid, 320, "CF3", ring_scale=0.82))
    p2 = lambda s, pid: (S._arm(s, pid, 65, "t-Bu"),
                         S._arm(s, pid, 0, "t-Bu"))
    return S.build_josiphos("SL-J011-1", psub_p1=p1, psub_p2=p2)


def _push_labels_out(H, labels, factor):
    """Push selected leaf labels (e.g. t-Bu) on each P donor further out along
    their P-arm, AFTER relaxing (the relaxer pulls every bond back to L, so the
    extension must be applied as a post-step to persist)."""
    for p in H.donors:
        px, py = H.pos[p]
        for n in list(H.adj[p]):
            if H.label.get(n) not in labels:
                continue
            comp, stack = {n}, [n]
            while stack:
                x = stack.pop()
                for y in H.adj[x]:
                    if y == p or y in comp:
                        continue
                    comp.add(y)
                    stack.append(y)
            dx = (H.pos[n][0] - px) * (factor - 1.0)
            dy = (H.pos[n][1] - py) * (factor - 1.0)
            for a in comp:
                H.pos[a] = (H.pos[a][0] + dx, H.pos[a][1] + dy)


def _extend_methyl(H, factor=1.3):
    """Push the stereocentre CH3 (the wedged methyl) a little further out so the
    stereo wedge reads clearly."""
    for i, lbl in H.label.items():
        if lbl and lbl.startswith("CH") and lbl != "Cu":
            nb = [n for n in H.adj[i]]
            if not nb:
                continue
            px, py = H.pos[nb[0]]
            dx = (H.pos[i][0] - px) * (factor - 1.0)
            dy = (H.pos[i][1] - py) * (factor - 1.0)
            H.pos[i] = (H.pos[i][0] + dx, H.pos[i][1] + dy)


def _slj011_H():
    # build + relax + the relief CLEAN-UP steps (swing / settle / declutter /
    # uncross) but WITHOUT the tilt, so both bis-CF3 aryls stay flat and the same
    # size as each other and the Cp rings.
    sc, title, meta = _slj011()
    H = R.Harness(scene=sc, title=title,
                  exempt=meta["exempt"], extra_rigid=meta["extra_rigid"])
    G.apply_recipe(H, {"weights": dict(G.GLOBAL)})
    R.swing_off_backbone(H)
    bodies, pin, inv, translate, _ = R.body_helpers(H)
    R.overlap_relax(H, bodies, pin, inv, translate, w_over=0.5, iters=80)
    R.declutter(H, angles=R.BIG_DECL, passes=2)
    R.uncross(H)
    _push_labels_out(H, {"t-Bu"}, 1.7)           # t-Bu further out
    _extend_methyl(H, 1.3)
    return H


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
    out["(S,S)-QuinoxP*"] = _quinoxp()
    out["(R,R)-Me-DuPhos"] = _smiles_H(
        "C[C@@H]1CC[C@@H](C)[P]1c1ccccc1[P]1[C@H](C)CC[C@H]1C", "(R,R)-Me-DuPhos")
    out["SL-J011-1"] = _slj011_H()
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
        "DTBM-DPEPhos", p_tetra=True)
    out["DTBM-Xantphos"] = _smiles_H(
        G.ligand_smiles(G.BACKBONES["xantphos"], G.SUBSTITUENTS["dtbm"]),
        "DTBM-Xantphos", p_tetra=True)
    out["DM-DPEPhos"] = _smiles_H(
        G.ligand_smiles(G.BACKBONES["dpephos"], "c1cc(C)cc(C)c1"), "DM-DPEPhos")
    return out


FIG1 = ["DTBM-SEGPHOS", "(S,S)-Ph-BPE", "DM-SEGPHOS",
        "(S,S)-QuinoxP*", "(R,R)-Me-DuPhos", "SL-J011-1", "DTBM-BINAP"]
FIG2 = ["(3,5-tBu2C6H3)-DPPBz", "(3-Oct-thienyl)-DPPBz",
        "DTBM-DPPBz", "DTBM-DPEPhos", "DTBM-Xantphos", "DM-DPEPhos"]


def main():
    outdir = HERE / "out" / "named"
    outdir.mkdir(parents=True, exist_ok=True)
    global _CACHE
    _CACHE = G.load_cache()
    Hs = named()
    G.save_cache(_CACHE)
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
    # one SEPARATE, whitespace-cropped PNG per ligand (the requested deliverable):
    # render each WITHOUT the title so the crop hugs the molecule, no title gap.
    cropdir = HERE.parent / "panels" / "named_cropped"
    cropdir.mkdir(parents=True, exist_ok=True)
    csvgs, cpngs = [], []
    for nm, H in Hs.items():
        H.commit()
        svg = H.scene.render_svg(*CANVAS, title=None, fixed_scale=SCALE,
                                 anchor_world=H.pos[H.metal], anchor_px=ANCHOR)
        sp = cropdir / f"{nm}.svg"
        sp.write_text(svg)
        csvgs.append(str(sp))
        cpngs.append(str(cropdir / f"{nm}.png"))
    R._svg_to_png(csvgs, cpngs)
    for p in cpngs:
        _crop(p, p)
    print(f"cropped per-ligand PNGs -> {cropdir}")
    print("done")


def _crop(src, dest, border=28):
    """Crop the rendered PNG to its ink bounding box (+ a small uniform border),
    so each per-ligand image has minimal surrounding whitespace."""
    from PIL import Image, ImageChops
    im = Image.open(src).convert("RGB")
    bg = Image.new("RGB", im.size, "white")
    bb = ImageChops.difference(im, bg).getbbox()
    if bb:
        x0, y0, x1, y1 = bb
        x0 = max(0, x0 - border)
        y0 = max(0, y0 - border)
        x1 = min(im.width, x1 + border)
        y1 = min(im.height, y1 + border)
        im = im.crop((x0, y0, x1, y1))
    im.save(dest)


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
