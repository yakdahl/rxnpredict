#!/usr/bin/env python3
"""
gen_gif.py -- animate the backbone x substituent matrix as a looping GIF in
which every consecutive frame changes EXACTLY ONE axis by one step (boustrophedon
/ snake traversal):

  row 0:  fix backbone, sweep all substituents      (substituent changes)
  step :  advance backbone by one (substituent held) (backbone changes)
  row 1:  sweep substituents back the other way      (substituent changes)
  step :  advance backbone by one ...                (backbone changes)

so the picture morphs one substituent-or-backbone edit at a time, 1 s per frame.
The frames are the fixed-scale renders, so Cu-H stays locked while the rest moves.
"""
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PANELS = HERE.parent / "panels"


def snake_order(backbones, subs, seed=7, turns=10):
    """Alternate full cyclic sweeps of the two axes -- one cycle, then turn 90
    deg to the other axis -- with randomised direction.  Consecutive frames
    change EXACTLY ONE of (backbone, substituent) by one step; each sweep is one
    full loop of its axis, then it turns; the fixed axis drifts by one per turn so
    the walk explores the whole grid instead of staying on a cross."""
    rnd = random.Random(seed)
    nb, ns = len(backbones), len(subs)
    i, j = rnd.randrange(nb), rnd.randrange(ns)
    idxs = [(i, j)]
    axis = rnd.randrange(2)                       # 0 = sweep substituents
    for _ in range(turns):
        d = rnd.choice((1, -1))
        if axis == 0:
            for _ in range(ns - 1):
                j = (j + d) % ns
                idxs.append((i, j))
        else:
            for _ in range(nb - 1):
                i = (i + d) % nb
                idxs.append((i, j))
        axis ^= 1
    return [(backbones[a], subs[b]) for a, b in idxs]


def main():
    idx = json.loads((PANELS / "matrix_index.json").read_text())
    backbones, subs = idx["backbones"], idx["subs"]
    have = {(c["backbone"], c["sub"]): c["png"] for c in idx["cells"]}
    order = [bs for bs in snake_order(backbones, subs) if bs in have]

    frames = []
    target_w = 940
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    for bk, sb in order:
        im = Image.open(have[(bk, sb)]).convert("RGB")
        scale = target_w / im.width
        im = im.resize((target_w, int(im.height * scale)))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, target_w, 30], fill=(20, 25, 40))
        d.text((10, 5), f"backbone={bk:10s}  substituent={sb}",
               fill="white", font=font)
        frames.append(im)

    out = PANELS / "matrix_animation.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=1000, loop=0, optimize=True)
    print(f"wrote {out}  ({len(frames)} frames, 1 s each)")
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
