#!/usr/bin/env python3
"""METHOD E renderer -- ChemDraw-style PIL rasteriser for the projected layout.

Reads the layout dict from methodE_projection.build_layout and draws:
  * THICK bonds, round caps/joins, meeting exactly at atom coords (no white gaps)
  * explicit Kekule double bonds drawn INSIDE the rings
  * MEDIUM plain-text labels (t-Bu, OMe, Ph) with a white halo behind text only
  * Cu-H bond HORIZONTAL with H to the RIGHT, no charges
  * wedge/dash on sp3 stereocentres; BOLD biaryl axis
  * P->Cu dative bonds drawn as arrows
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from methods import methodE_projection as mE  # noqa: E402

# ---- style ----
SS = 3                       # supersample factor (drawn big, downscaled = AA)
W, H = 760, 640
BONDLEN_PX = 50              # one standard bond length in px (at 1x)
BOND_W = 3.4                 # bond thickness (1x)
DOUBLE_GAP = 6.0             # inner double-bond offset (1x)
AXIS_W = 6.5                 # bold biaryl axis thickness
FONT_SZ = 20
SUB_SZ = 13

BLACK = (20, 20, 20)
WHITE = (255, 255, 255)
CU_COL = (150, 95, 40)       # subtle warm for Cu glyph
P_COL = (190, 120, 30)
O_COL = (200, 40, 40)

EL_COL = {"Cu": CU_COL, "P": P_COL, "O": O_COL, "H": BLACK, "C": BLACK}


def _font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def _font_reg(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


# ---------------------------------------------------------------- transform
def orient_and_fit(lay):
    """Rotate so Cu-H is horizontal (H right), scale to BONDLEN, centre."""
    pos = lay["pos"].copy()
    metal, hyd = lay["metal"], lay["hydride"]
    v = pos[hyd] - pos[metal]
    ang = math.atan2(v[1], v[0])
    # rotate by -ang so Cu->H points +x
    c, s = math.cos(-ang), math.sin(-ang)
    R = np.array([[c, -s], [s, c]])
    pos = pos @ R.T
    # flip y so up is up (image y grows down -> we invert at draw time anyway)
    return pos


def to_px(pos, keep_idx, label_pts):
    """Map layout coords (only kept atoms + label points) to image px."""
    allpts = [pos[i] for i in keep_idx] + label_pts
    arr = np.array(allpts)
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    span = mx - mn
    span[span < 1e-6] = 1.0
    # scale by bond length consistency: layout bond ~1.0 -> BONDLEN_PX
    scale = BONDLEN_PX
    # fit check: if too big for canvas, shrink
    pad = 70
    avail_w = (W - 2 * pad)
    avail_h = (H - 2 * pad)
    if span[0] * scale > avail_w:
        scale = avail_w / span[0]
    if span[1] * scale > avail_h:
        scale = min(scale, avail_h / span[1])
    cen = (mn + mx) / 2.0

    def f(p):
        x = (p[0] - cen[0]) * scale + W / 2
        y = -(p[1] - cen[1]) * scale + H / 2   # invert y
        return (x, y)
    return f, scale


# ---------------------------------------------------------------- drawing
def _line(dr, p, q, width, color=BLACK):
    dr.line([p[0] * SS, p[1] * SS, q[0] * SS, q[1] * SS],
            fill=color, width=int(round(width * SS)), joint="curve")
    # round caps
    r = width * SS / 2.0
    for (x, y) in (p, q):
        dr.ellipse([x * SS - r, y * SS - r, x * SS + r, y * SS + r], fill=color)


def _draw_double(dr, p, q, ring_centers, width):
    """Single line + an inner parallel line offset toward the nearest ring centre."""
    p = np.array(p, float)
    q = np.array(q, float)
    d = q - p
    L = np.linalg.norm(d)
    if L < 1e-6:
        return
    u = d / L
    nrm = np.array([-u[1], u[0]])
    mid = (p + q) / 2
    # choose offset side: toward nearest ring centre if available, else default
    side = nrm
    if ring_centers:
        nearest = min(ring_centers, key=lambda c: np.hypot(*(np.array(c) - mid)))
        if np.dot(np.array(nearest) - mid, nrm) < 0:
            side = -nrm
    # main line full length
    _line(dr, p, q, width)
    # inner line shortened
    off = side * DOUBLE_GAP
    sh = 0.16 * L
    ip = p + u * sh + off
    iq = q - u * sh + off
    _line(dr, ip, iq, width)


def _wedge(dr, p, q, width, kind):
    """Solid wedge ('up') or hashed ('down') from p (narrow) to q (wide)."""
    p = np.array(p, float)
    q = np.array(q, float)
    d = q - p
    L = np.linalg.norm(d)
    if L < 1e-6:
        return
    u = d / L
    nrm = np.array([-u[1], u[0]])
    wide = width * 2.4
    if kind == "up":
        a = q + nrm * wide
        b = q - nrm * wide
        dr.polygon([(p[0] * SS, p[1] * SS), (a[0] * SS, a[1] * SS),
                    (b[0] * SS, b[1] * SS)], fill=BLACK)
    else:
        nh = 6
        for k in range(1, nh + 1):
            t = k / (nh + 0.0)
            w = wide * t
            c = p + d * t
            e1 = c + nrm * w
            e2 = c - nrm * w
            _line(dr, e1, e2, width * 0.7)


def _arrow(dr, p, q, width, color=BLACK):
    """Dative arrow from p (donor) to q (metal)."""
    p = np.array(p, float)
    q = np.array(q, float)
    d = q - p
    L = np.linalg.norm(d)
    if L < 1e-6:
        return
    u = d / L
    nrm = np.array([-u[1], u[0]])
    # shorten so the head sits just before the metal glyph
    tip = q - u * (width * 1.0)
    _line(dr, p, tip, width, color)
    hl = width * 2.6
    base = tip - u * hl
    a = base + nrm * hl * 0.55
    b = base - nrm * hl * 0.55
    dr.polygon([(tip[0] * SS, tip[1] * SS), (a[0] * SS, a[1] * SS),
                (b[0] * SS, b[1] * SS)], fill=color)


def _text_halo(dr, xy, text, font, color, anchor="mm"):
    """Draw text with a white halo behind the text ONLY (no box)."""
    x, y = xy[0] * SS, xy[1] * SS
    halo = max(3, int(BOND_W * SS * 0.55))
    for dx in range(-halo, halo + 1, 2):
        for dy in range(-halo, halo + 1, 2):
            if dx * dx + dy * dy <= halo * halo:
                dr.text((x + dx, y + dy), text, font=font, fill=WHITE, anchor=anchor)
    dr.text((x, y), text, font=font, fill=color, anchor=anchor)


# ---------------------------------------------------------------- main render
def render(lay, out_png):
    mol = lay["mol"]
    pos = orient_and_fit(lay)
    keep = lay["keep"]
    metal, hyd = lay["metal"], lay["hydride"]
    depth = lay["depth"]
    donors = set(lay["donors"])
    collapsed = lay["collapsed"]
    groups = lay["groups"]
    korder = lay["korder"]
    biaryl = lay["biaryl"]
    wedge = lay["wedge"]

    # ---- label placement for each collapsed group ----
    # label sits beyond the attach atom along (anchor->attach extended), at
    # ~1 bond length from attach.
    keep_idx = sorted(keep)
    label_pts = []
    label_info = []  # (point, text, attach)
    for g in groups:
        attach = g["attach"]
        anchor = g["anchor"]
        # direction from attach outward = away from attach's kept neighbours mean
        a = pos[attach]
        kn = [pos[j] for j in [nb.GetIdx() for nb in mol.GetAtomWithIdx(attach).GetNeighbors()]
              if j in keep and j != anchor]
        if kn:
            inward = np.mean(kn, axis=0) - a
            ninw = np.linalg.norm(inward)
            direction = -inward / ninw if ninw > 1e-6 else np.array([1.0, 0.0])
        else:
            # use projected anchor position relative to attach
            direction = (pos[anchor] - a)
            nd = np.linalg.norm(direction)
            direction = direction / nd if nd > 1e-6 else np.array([1.0, 0.0])
        lp = a + direction * (mE.BOND * 1.0)
        label_pts.append(lp)
        label_info.append((lp, g["label"], attach, direction))

    f, scale = to_px(pos, keep_idx, label_pts)

    # ring centres (px) for double-bond inner side
    from methods.methodE_projection import ordered_rings
    rings = [r for r in ordered_rings(mol) if all(i in keep for i in r)]
    ring_centers = []
    for r in rings:
        c = np.mean([f(pos[i]) for i in r], axis=0)
        ring_centers.append(tuple(c))

    img = Image.new("RGB", (W * SS, H * SS), WHITE)
    dr = ImageDraw.Draw(img)

    # ---- bonds ----
    bw = BOND_W
    for b in mol.GetBonds():
        ai, bi = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        # skip bonds wholly inside collapsed groups; skip the anchor->inside-group
        if ai in collapsed and bi in collapsed:
            continue
        # bond from a kept atom to a collapsed group -> drawn as stub to label
        if (ai in collapsed) ^ (bi in collapsed):
            continue  # handled by label stub below
        if ai not in keep or bi not in keep:
            continue
        pa, pb = f(pos[ai]), f(pos[bi])
        bt = b.GetBondType()
        # dative P->Cu
        if bt == __import__("rdkit").Chem.BondType.DATIVE:
            donor = ai if ai in donors else bi
            _arrow(dr, f(pos[donor]), f(pos[metal]), bw)
            continue
        # bold biaryl axis
        if biaryl and {ai, bi} == set(biaryl):
            _line(dr, pa, pb, AXIS_W)
            continue
        # wedge?
        wk = wedge.get((ai, bi)) or wedge.get((bi, ai))
        if wk:
            # narrow end at the stereocentre
            if (ai, bi) in wedge:
                _wedge(dr, pa, pb, bw, wk)
            else:
                _wedge(dr, pb, pa, bw, wk)
            continue
        order = korder.get(frozenset((ai, bi)), b.GetBondTypeAsDouble())
        if order >= 2.0:
            _draw_double(dr, pa, pb, ring_centers, bw)
        else:
            _line(dr, pa, pb, bw)

    # ---- Cu-H bond (horizontal, H to the right) ----
    _line(dr, f(pos[metal]), f(pos[hyd]), bw)

    # ---- label stubs: bond from attach atom to label point ----
    fnt = _font(FONT_SZ)
    fnt_sub = _font(SUB_SZ)
    for lp, text, attach, direction in label_info:
        pa = f(pos[attach])
        plab = f(lp)
        # draw stub, then label over it (halo hides the overlap = no gap)
        # shorten stub so it ends at label edge
        _line(dr, pa, plab, bw)

    # ---- atom glyphs (heteroatoms + metal + hydride) ----
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if i not in keep:
            continue
        sym = a.GetSymbol()
        if sym == "C":
            continue  # carbons implicit
        p = f(pos[i])
        col = EL_COL.get(sym, BLACK)
        _text_halo(dr, p, sym, fnt, col)

    # Cu and H glyphs (no charges)
    _text_halo(dr, f(pos[metal]), "Cu", fnt, CU_COL)
    _text_halo(dr, f(pos[hyd]), "H", fnt, BLACK)

    # ---- group labels (plain text, halo) ----
    for lp, text, attach, direction in label_info:
        plab = f(lp)
        _text_halo(dr, plab, text, fnt, BLACK)

    # ---- molecule name ----
    nm_fnt = _font(22)
    dr.text((14 * SS, 12 * SS), lay["name"], font=nm_fnt, fill=BLACK)

    # downscale (anti-alias)
    img = img.resize((W, H), Image.LANCZOS)
    img.save(out_png)
    return out_png


def emit_json(lay, out_json):
    """Viewer-style depiction JSON for Method E: the projected+regularised 2D
    coords, the out-of-plane depth, Kekule bond orders, wedge/dash, MEDIUM
    plain-text groups, and the chelate metadata."""
    import json
    mol = lay["mol"]
    pos = mE.orient_for_export(lay) if hasattr(mE, "orient_for_export") else lay["pos"]
    pos = orient_and_fit(lay)
    keep = lay["keep"]
    depth = lay["depth"]
    metal, hyd = lay["metal"], lay["hydride"]
    donors = set(lay["donors"])
    biaryl = lay["biaryl"]
    wedge = lay["wedge"]
    korder = lay["korder"]
    atoms = []
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if i not in keep:
            continue
        atoms.append({
            "id": i, "el": a.GetSymbol(),
            "x": round(float(pos[i][0]), 4), "y": round(float(pos[i][1]), 4),
            "depth": round(float(depth[i]), 4),
            "role": ("metal" if i == metal else "hydride" if i == hyd
                     else "donor" if i in donors else "backbone"),
        })
    bonds = []
    for b in mol.GetBonds():
        ai, bi = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if ai not in keep or bi not in keep:
            continue
        bt = b.GetBondType()
        if bt == __import__("rdkit").Chem.BondType.DATIVE:
            typ, order = "dative", 1.0
        else:
            order = korder.get(frozenset((ai, bi)), b.GetBondTypeAsDouble())
            typ = "double" if order >= 2.0 else "single"
        wk = wedge.get((ai, bi)) or wedge.get((bi, ai))
        bonds.append({"a": ai, "b": bi, "type": typ, "order": order,
                      "wedge": wk or "none",
                      "axis": bool(biaryl) and {ai, bi} == set(biaryl)})
    groups = [{"label": g["label"], "atoms": sorted(g["atoms"]),
               "attach": g["attach"], "anchor": g["anchor"]}
              for g in lay["groups"]]
    data = {
        "meta": {"name": lay["name"], "formula": lay["formula"],
                 "smiles": lay["smiles"], "method": "E:3D->2D PCA projection",
                 "conformer_method": lay["method"], "abbreviation": "medium"},
        "metal": {"id": metal, "el": "Cu", "hydride": hyd,
                  "donors": sorted(donors), "geometry": "trigonal planar"},
        "atoms": atoms, "bonds": bonds, "groups": groups,
        "cu_h_horizontal": True, "charges": "none",
    }
    Path(out_json).write_text(json.dumps(data, indent=2))
    return out_json


def main():
    prefix = os.environ.get("PANEL_PREFIX", "/tmp/wf_E_")
    outs = []
    for k, (nm, smi) in mE.LIGANDS.items():
        lay = mE.build_layout(k, nm, smi)
        out = f"{prefix}{k}.png"
        render(lay, out)
        emit_json(lay, f"{prefix}{k}.json")
        outs.append(out)
        print("rendered", k, "->", out, "(+ json)")
    # montage
    montage(outs, f"{prefix}panel.png")
    print("montage ->", f"{prefix}panel.png")


def montage(paths, out):
    imgs = [Image.open(p) for p in paths]
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)
    M = Image.new("RGB", (w * 2, h * 2), WHITE)
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, 2)
        M.paste(im, (c * w, r * h))
    # grid separators
    d = ImageDraw.Draw(M)
    d.line([w, 0, w, h * 2], fill=(210, 210, 210), width=2)
    d.line([0, h, w * 2, h], fill=(210, 210, 210), width=2)
    M.save(out)


if __name__ == "__main__":
    main()
