#!/usr/bin/env python3
"""
gen_matrix_grid.py -- montage the backbone x substituent matrix PNGs into ONE
labelled grid (rows = backbones, columns = substituents).  Every cell is cropped
to the SAME common window (the union of all cells' ink bounding boxes), so the
locked Cu-H scale / orientation / size is preserved across the whole grid -- the
images stay directly comparable.

Reads ../panels/matrix_index.json + ../panels/matrix/*.png, writes
../panels/matrix_grid_labeled.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PANELS = HERE.parent / "panels"

# short column / row labels (match the established grid)
SUB_LABEL = {"dtbm": "DTBM", "biscf3": "3,5-(CF3)2", "xylyl": "3,5-Me2",
             "furyl": "2-furyl", "octylthienyl": "3-Oct-thienyl",
             "cyclohexyl": "Cy", "tbu": "t-Bu", "methyl": "Me"}
BK_LABEL = {"segphos": "SEGPhos", "xantphos": "Xantphos", "dpephos": "DPEphos",
            "dppbz": "DPPBz", "binap": "BINAP", "naphthyl": "1,8-napt",
            "dppe": "DPPE", "dppm": "DPPM"}


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def _ink_bbox(im):
    bg = Image.new("RGB", im.size, "white")
    return ImageChops.difference(im.convert("RGB"), bg).getbbox()


def main():
    idx = json.loads((PANELS / "matrix_index.json").read_text())
    backbones, subs = idx["backbones"], idx["subs"]
    have = {(c["backbone"], c["sub"]): c["png"] for c in idx["cells"]}
    pngs = {k: Image.open(v).convert("RGB") for k, v in have.items()}

    # common crop window = union of every cell's ink bbox (keeps a single scale)
    x0 = y0 = 10 ** 9
    x1 = y1 = 0
    for im in pngs.values():
        bb = _ink_bbox(im)
        if not bb:
            continue
        x0, y0 = min(x0, bb[0]), min(y0, bb[1])
        x1, y1 = max(x1, bb[2]), max(y1, bb[3])
    pad = 12
    crop = (max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad)
    cw0, ch0 = crop[2] - crop[0], crop[3] - crop[1]
    cell_w = 300                                   # downscale every cell to this
    sc = cell_w / cw0
    cw, ch = cell_w, int(round(ch0 * sc))

    margin_l, margin_t = 220, 90      # room for row / column headers
    gap = 10
    W = margin_l + len(subs) * (cw + gap)
    H = margin_t + len(backbones) * (ch + gap)
    grid = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(grid)
    fcol, frow = _font(34), _font(34)

    for ci, sb in enumerate(subs):
        x = margin_l + ci * (cw + gap) + cw // 2
        t = SUB_LABEL.get(sb, sb)
        w = d.textlength(t, font=fcol)
        d.text((x - w / 2, 30), t, fill="#111", font=fcol)
    for ri, bk in enumerate(backbones):
        y = margin_t + ri * (ch + gap) + ch // 2
        t = BK_LABEL.get(bk, bk)
        d.text((24, y - 17), t, fill="#111", font=frow)
        for ci, sb in enumerate(subs):
            im = pngs.get((bk, sb))
            if im is None:
                continue
            cell = im.crop(crop).resize((cw, ch), Image.LANCZOS)
            grid.paste(cell, (margin_l + ci * (cw + gap),
                              margin_t + ri * (ch + gap)))

    out = PANELS / "matrix_grid_labeled.png"
    grid.save(out)
    print(f"grid -> {out}  ({grid.size[0]}x{grid.size[1]}, "
          f"{len(backbones)}x{len(subs)} cells, common crop {cw}x{ch})")
    return out


if __name__ == "__main__":
    main()
