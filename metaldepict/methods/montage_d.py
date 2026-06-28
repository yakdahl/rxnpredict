#!/usr/bin/env python3
"""Montage the four Method-D ligand PNGs into one panel."""
import sys
from PIL import Image

prefix = sys.argv[1] if len(sys.argv) > 1 else "/tmp/wf_D_"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/wf_D_panel.png"
keys = ["dtbm_segphos", "xantphos", "dpephos", "ph_bpe"]

from PIL import ImageChops


def trim(im, border=24):
    """Crop surrounding white margin, then re-pad with a small uniform border."""
    bg = Image.new("RGB", im.size, "white")
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        im = im.crop(bbox)
    out = Image.new("RGB", (im.width + 2 * border, im.height + 2 * border), "white")
    out.paste(im, (border, border))
    return out


imgs = [trim(Image.open(f"{prefix}{k}.png").convert("RGB")) for k in keys]
# 2x2 grid, cells sized to the largest image
cw = max(im.width for im in imgs)
ch = max(im.height for im in imgs)
pad = 16
W = cw * 2 + pad * 3
H = ch * 2 + pad * 3
panel = Image.new("RGB", (W, H), "white")
for i, im in enumerate(imgs):
    r, c = divmod(i, 2)
    x = pad + c * (cw + pad) + (cw - im.width) // 2
    y = pad + r * (ch + pad) + (ch - im.height) // 2
    panel.paste(im, (x, y))
panel.save(out)
print("montage ->", out, panel.size)
