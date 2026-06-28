#!/usr/bin/env python3
"""Montage the four method-C PNGs into a single 2x2 panel."""
import sys
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
KEYS = ["dtbm_segphos", "xantphos", "dpephos", "ph_bpe"]
DEST = sys.argv[1] if len(sys.argv) > 1 else "/tmp/wf_C_panel.png"


def main():
    imgs = [Image.open(OUT / f"{k}.png").convert("RGB") for k in KEYS]
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)
    pad = 6
    panel = Image.new("RGB", (w * 2 + pad * 3, h * 2 + pad * 3), "#dddddd")
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, 2)
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        panel.paste(im, (x, y))
    panel.save(DEST)
    print("montage ->", DEST, panel.size)


if __name__ == "__main__":
    main()
