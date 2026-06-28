#!/usr/bin/env python3
"""
template_draw.py -- a tiny deterministic 2D molecule-drawing toolkit that emits
ChemDraw-style SVG.  No force-directed physics: every atom is placed by explicit
template rules (regular polygons, fixed bond length, ideal ~120 deg angles).

Primitives
----------
  Scene        : collects atoms/bonds, then renders to SVG.
  L            : the single standard bond length (world units).
  ring(...)    : place a regular polygon and return its vertex ids.
  bond(...)    : single / double (Kekule, inside-offset) / wedge / dash / dative.
  label(...)   : plain-text atom/abbreviation label with a white halo (no box).

Coordinates are in an abstract world frame with +x right and +y *up* (chemistry
convention); the SVG writer flips y.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---- global drawing constants -------------------------------------------------
L = 1.0                      # standard bond length (world units)
BOND_W = 0.090 * L           # thick bond stroke width
DOUBLE_GAP = 0.165 * L       # offset of the inner line of a double bond
DOUBLE_SHRINK = 0.175 * L    # how much the inner double-bond line is shortened
WEDGE_WIDE = 0.27 * L        # wide end of a stereo wedge
FONT = 0.50 * L              # text height for atom labels


def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def vscale(a, s):
    return (a[0] * s, a[1] * s)


def vlen(a):
    return math.hypot(a[0], a[1])


def vnorm(a):
    n = vlen(a)
    return (a[0] / n, a[1] / n) if n else (0.0, 0.0)


def perp(a):
    """left-hand perpendicular"""
    return (-a[1], a[0])


def polar(origin, ang_deg, r):
    a = math.radians(ang_deg)
    return (origin[0] + r * math.cos(a), origin[1] + r * math.sin(a))


# ------------------------------------------------------------------------------
@dataclass
class Atom:
    id: int
    pos: tuple
    label: str = ""
    color: str = "#111"
    halo: bool = True
    fontscale: float = 1.0


@dataclass
class Bond:
    a: int
    b: int
    order: int = 1
    kind: str = "plain"      # plain | wedge | dash | dative | bold
    inside: tuple = None


@dataclass
class Scene:
    atoms: dict = field(default_factory=dict)
    bonds: list = field(default_factory=list)
    _next: int = 0

    def atom(self, pos, label="", color="#111", halo=True, fontscale=1.0):
        i = self._next
        self._next += 1
        self.atoms[i] = Atom(i, pos, label, color, halo=halo, fontscale=fontscale)
        return i

    def move(self, i, pos):
        self.atoms[i].pos = pos

    def bond(self, a, b, order=1, kind="plain", inside=None):
        self.bonds.append(Bond(a, b, order, kind, inside))

    def ring(self, center, n, start_deg, r=None, kekule=None):
        """Place an n-gon with edge length == L. start_deg = angle of vertex 0.
        kekule: list[int] of length n giving ring-edge orders, drawn inside.
        Returns (vertex_ids, circumradius)."""
        if r is None:
            r = L / (2 * math.sin(math.pi / n))
        ids = []
        for k in range(n):
            ang = start_deg + k * 360.0 / n
            ids.append(self.atom(polar(center, ang, r)))
        if kekule is not None:
            for k in range(n):
                a, b = ids[k], ids[(k + 1) % n]
                self.bond(a, b, order=kekule[k], inside=center)
        return ids, r

    # -- SVG output ------------------------------------------------------------
    def render_svg(self, width=760, height=640, pad=0.7, bg="#ffffff",
                   title=None, title_color="#1a1a1a"):
        xs = [a.pos[0] for a in self.atoms.values()]
        ys = [a.pos[1] for a in self.atoms.values()]
        minx, maxx = min(xs) - pad, max(xs) + pad
        miny, maxy = min(ys) - pad, max(ys) + pad
        wspan = maxx - minx
        hspan = maxy - miny
        scale = min(width / wspan, height / hspan)
        offx = (width - wspan * scale) / 2
        offy = (height - hspan * scale) / 2

        def tx(p):
            X = offx + (p[0] - minx) * scale
            Y = height - (offy + (p[1] - miny) * scale)
            return X, Y

        bw = BOND_W * scale
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
               f'height="{height}" viewBox="0 0 {width} {height}">']
        out.append(f'<rect width="{width}" height="{height}" fill="{bg}"/>')
        out.append('<g stroke-linecap="round" stroke-linejoin="round">')

        def trim_for_label(pa, pb, ia, ib):
            ax, ay = pa
            bx, by = pb
            d = vnorm((bx - ax, by - ay))
            la = self.atoms[ia].label
            lb = self.atoms[ib].label
            ta = (0.34 + 0.16 * max(0, len(la) - 1)) if la else 0.0
            tb = (0.34 + 0.16 * max(0, len(lb) - 1)) if lb else 0.0
            return (ax + d[0] * ta, ay + d[1] * ta), (bx - d[0] * tb, by - d[1] * tb)

        for bd in self.bonds:
            A = self.atoms[bd.a].pos
            B = self.atoms[bd.b].pos
            A2, B2 = trim_for_label(A, B, bd.a, bd.b)
            pa = tx(A2)
            pb = tx(B2)
            col = "#111"
            if bd.kind == "wedge":
                d = vnorm(vsub(pb, pa))
                pr = perp(d)
                w = WEDGE_WIDE * scale
                p1 = (pa[0], pa[1])
                p2 = (pb[0] + pr[0] * w / 2, pb[1] + pr[1] * w / 2)
                p3 = (pb[0] - pr[0] * w / 2, pb[1] - pr[1] * w / 2)
                out.append(f'<polygon points="{p1[0]:.2f},{p1[1]:.2f} '
                           f'{p2[0]:.2f},{p2[1]:.2f} {p3[0]:.2f},{p3[1]:.2f}" '
                           f'fill="{col}"/>')
            elif bd.kind == "dash":
                d = vnorm(vsub(pb, pa))
                pr = perp(d)
                n = 6
                for k in range(1, n + 1):
                    t = k / (n + 0.5)
                    cx = pa[0] + (pb[0] - pa[0]) * t
                    cy = pa[1] + (pb[1] - pa[1]) * t
                    w = (WEDGE_WIDE * scale) * t / 2
                    q1 = (cx + pr[0] * w, cy + pr[1] * w)
                    q2 = (cx - pr[0] * w, cy - pr[1] * w)
                    out.append(f'<line x1="{q1[0]:.2f}" y1="{q1[1]:.2f}" '
                               f'x2="{q2[0]:.2f}" y2="{q2[1]:.2f}" '
                               f'stroke="{col}" stroke-width="{bw*0.75:.2f}"/>')
            elif bd.kind == "dative":
                out.append(f'<line x1="{pa[0]:.2f}" y1="{pa[1]:.2f}" '
                           f'x2="{pb[0]:.2f}" y2="{pb[1]:.2f}" '
                           f'stroke="{col}" stroke-width="{bw:.2f}"/>')
                d = vnorm(vsub(pb, pa))
                pr = perp(d)
                ah = 0.32 * L * scale
                aw = 0.17 * L * scale
                base = (pb[0] - d[0] * ah, pb[1] - d[1] * ah)
                t1 = (base[0] + pr[0] * aw, base[1] + pr[1] * aw)
                t2 = (base[0] - pr[0] * aw, base[1] - pr[1] * aw)
                out.append(f'<polygon points="{pb[0]:.2f},{pb[1]:.2f} '
                           f'{t1[0]:.2f},{t1[1]:.2f} {t2[0]:.2f},{t2[1]:.2f}" '
                           f'fill="{col}"/>')
            elif bd.kind == "bold":
                out.append(f'<line x1="{pa[0]:.2f}" y1="{pa[1]:.2f}" '
                           f'x2="{pb[0]:.2f}" y2="{pb[1]:.2f}" '
                           f'stroke="{col}" stroke-width="{bw*1.9:.2f}"/>')
            else:
                out.append(f'<line x1="{pa[0]:.2f}" y1="{pa[1]:.2f}" '
                           f'x2="{pb[0]:.2f}" y2="{pb[1]:.2f}" '
                           f'stroke="{col}" stroke-width="{bw:.2f}"/>')
                if bd.order == 2:
                    d = vnorm(vsub(B2, A2))
                    pr = perp(d)
                    mid = vscale(vadd(A2, B2), 0.5)
                    if bd.inside is not None:
                        toward = vnorm(vsub(bd.inside, mid))
                        if (toward[0] * pr[0] + toward[1] * pr[1]) < 0:
                            pr = (-pr[0], -pr[1])
                    gap = DOUBLE_GAP
                    sh = DOUBLE_SHRINK
                    iA = (A2[0] + pr[0] * gap + d[0] * sh,
                          A2[1] + pr[1] * gap + d[1] * sh)
                    iB = (B2[0] + pr[0] * gap - d[0] * sh,
                          B2[1] + pr[1] * gap - d[1] * sh)
                    qa = tx(iA)
                    qb = tx(iB)
                    out.append(f'<line x1="{qa[0]:.2f}" y1="{qa[1]:.2f}" '
                               f'x2="{qb[0]:.2f}" y2="{qb[1]:.2f}" '
                               f'stroke="{col}" stroke-width="{bw:.2f}"/>')

        out.append('</g>')

        fs = FONT * scale
        for a in self.atoms.values():
            if not a.label:
                continue
            X, Y = tx(a.pos)
            f = fs * a.fontscale
            common = (f'text-anchor="middle" dominant-baseline="central" '
                      f'font-family="Helvetica, Arial, sans-serif" '
                      f'font-size="{f:.1f}" font-weight="600"')
            if a.halo:
                out.append(f'<text x="{X:.2f}" y="{Y:.2f}" {common} '
                           f'stroke="#ffffff" stroke-width="{f*0.34:.2f}" '
                           f'fill="#ffffff">{_esc(a.label)}</text>')
            out.append(f'<text x="{X:.2f}" y="{Y:.2f}" {common} '
                       f'fill="{a.color}">{_esc(a.label)}</text>')

        if title:
            out.append(f'<text x="16" y="30" font-family="Helvetica, Arial, '
                       f'sans-serif" font-size="20" font-weight="700" '
                       f'fill="{title_color}">{_esc(title)}</text>')
        out.append('</svg>')
        return "\n".join(out)


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
