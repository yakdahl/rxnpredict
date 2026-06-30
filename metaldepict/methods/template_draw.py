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
    width: float = None      # override wedge wide-end half-width (world units)


@dataclass
class Scene:
    atoms: dict = field(default_factory=dict)
    bonds: list = field(default_factory=list)
    ring_circles: list = field(default_factory=list)  # (vertex_ids, r_frac) aromatic circle
    wedges: list = field(default_factory=list)         # (tip_id, base_id, half_w) eta5 wedge
    _next: int = 0

    def atom(self, pos, label="", color="#111", halo=True, fontscale=1.0):
        i = self._next
        self._next += 1
        self.atoms[i] = Atom(i, pos, label, color, halo=halo, fontscale=fontscale)
        return i

    def move(self, i, pos):
        self.atoms[i].pos = pos

    def bond(self, a, b, order=1, kind="plain", inside=None, width=None):
        self.bonds.append(Bond(a, b, order, kind, inside, width))

    def ring_circle(self, vertex_ids, r_frac=0.62):
        """An aromatic-ring circle inscribed in the ring of `vertex_ids`; centre
        and radii are recomputed from CURRENT vertex positions at render time, so
        it follows the ring (e.g. a ferrocene Cp as it relaxes)."""
        self.ring_circles.append((list(vertex_ids), r_frac))

    def wedge(self, tip_id, base_id, half_w=0.32):
        """A solid metallocene-style eta5 wedge: a filled triangle, narrow at
        `tip_id` (Fe) and `half_w`*L wide at `base_id` (a Cp carbon)."""
        self.wedges.append((tip_id, base_id, half_w))

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
                   title=None, title_color="#1a1a1a",
                   fixed_scale=None, anchor_world=None, anchor_px=None):
        # fixed_scale (px per world-unit) + anchor pins a world point to a pixel
        # point with NO auto-fit, so every image shares one scale and a locked
        # Cu-H bond -- atoms (incl. Cu/H glyphs) render at identical size.
        if fixed_scale is not None:
            scale = fixed_scale
            aw = anchor_world or (0.0, 0.0)
            ap = anchor_px or (width / 2.0, height / 2.0)

            def tx(p):
                return ap[0] + (p[0] - aw[0]) * scale, ap[1] - (p[1] - aw[1]) * scale
        else:
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
                w = (bd.width if bd.width is not None else WEDGE_WIDE) * scale
                p1 = (pa[0], pa[1])
                p2 = (pb[0] + pr[0] * w / 2, pb[1] + pr[1] * w / 2)
                p3 = (pb[0] - pr[0] * w / 2, pb[1] - pr[1] * w / 2)
                out.append(f'<polygon points="{p1[0]:.2f},{p1[1]:.2f} '
                           f'{p2[0]:.2f},{p2[1]:.2f} {p3[0]:.2f},{p3[1]:.2f}" '
                           f'fill="{col}"/>')
            elif bd.kind == "taper":
                # trapezoid: NORMAL bond width at `a`, growing to `width` at `b`
                # (a perspective edge coming toward the viewer)
                d = vnorm(vsub(pb, pa))
                pr = perp(d)
                w = (bd.width if bd.width is not None else WEDGE_WIDE) * scale
                n = bw
                a1 = (pa[0] + pr[0] * n / 2, pa[1] + pr[1] * n / 2)
                a2 = (pa[0] - pr[0] * n / 2, pa[1] - pr[1] * n / 2)
                b1 = (pb[0] + pr[0] * w / 2, pb[1] + pr[1] * w / 2)
                b2 = (pb[0] - pr[0] * w / 2, pb[1] - pr[1] * w / 2)
                out.append(f'<polygon points="{a1[0]:.2f},{a1[1]:.2f} '
                           f'{b1[0]:.2f},{b1[1]:.2f} {b2[0]:.2f},{b2[1]:.2f} '
                           f'{a2[0]:.2f},{a2[1]:.2f}" fill="{col}"/>')
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
            elif bd.kind in ("dative", "coord"):
                # coordinate / dative bond -> a simple DASHED line (no arrowhead),
                # the conventional way to draw a metal<-donor coordination bond.
                dash = f'{0.16 * L * scale:.2f},{0.13 * L * scale:.2f}'
                out.append(f'<line x1="{pa[0]:.2f}" y1="{pa[1]:.2f}" '
                           f'x2="{pb[0]:.2f}" y2="{pb[1]:.2f}" '
                           f'stroke="{col}" stroke-width="{bw:.2f}" '
                           f'stroke-dasharray="{dash}"/>')
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
                        gap, sh = DOUBLE_GAP, DOUBLE_SHRINK   # ring: offset + inset
                    else:
                        gap, sh = DOUBLE_GAP, 0.0             # terminal C=O / C=N:
                    iA = (A2[0] + pr[0] * gap + d[0] * sh,    # full-length second line
                          A2[1] + pr[1] * gap + d[1] * sh)
                    iB = (B2[0] + pr[0] * gap - d[0] * sh,
                          B2[1] + pr[1] * gap - d[1] * sh)
                    qa = tx(iA)
                    qb = tx(iB)
                    out.append(f'<line x1="{qa[0]:.2f}" y1="{qa[1]:.2f}" '
                               f'x2="{qb[0]:.2f}" y2="{qb[1]:.2f}" '
                               f'stroke="{col}" stroke-width="{bw:.2f}"/>')
                elif bd.order == 3:
                    # triple bond (e.g. a metal carbonyl C#O): the centre line
                    # above plus a symmetric flanking line on EACH side.
                    d = vnorm(vsub(B2, A2))
                    pr = perp(d)
                    sh = DOUBLE_SHRINK
                    gap = DOUBLE_GAP * 1.15
                    for sgn in (1, -1):
                        iA = (A2[0] + sgn * pr[0] * gap + d[0] * sh,
                              A2[1] + sgn * pr[1] * gap + d[1] * sh)
                        iB = (B2[0] + sgn * pr[0] * gap - d[0] * sh,
                              B2[1] + sgn * pr[1] * gap - d[1] * sh)
                        qa, qb = tx(iA), tx(iB)
                        out.append(f'<line x1="{qa[0]:.2f}" y1="{qa[1]:.2f}" '
                                   f'x2="{qb[0]:.2f}" y2="{qb[1]:.2f}" '
                                   f'stroke="{col}" stroke-width="{bw:.2f}"/>')

        out.append('</g>')

        # ferrocene eta5 wedges (filled triangles) -- resolved from CURRENT atoms
        for tip, base, hw in self.wedges:
            pt = self.atoms[tip].pos
            pb = self.atoms[base].pos
            d = vnorm(vsub(pb, pt))
            pr = perp(d)
            w = hw * L
            t = tx(pt)
            b1 = tx((pb[0] + pr[0] * w, pb[1] + pr[1] * w))
            b2 = tx((pb[0] - pr[0] * w, pb[1] - pr[1] * w))
            out.append(f'<polygon points="{t[0]:.2f},{t[1]:.2f} '
                       f'{b1[0]:.2f},{b1[1]:.2f} {b2[0]:.2f},{b2[1]:.2f}" '
                       f'fill="#111"/>')
        # aromatic-ring circles (ellipses matching the ring's drawn shape, via
        # the vertex cloud's PRINCIPAL axes -- so a foreshortened / tilted ring
        # gets a correctly-oriented ellipse, not an axis-aligned bbox blob)
        for vids, rf in self.ring_circles:
            pts = [self.atoms[i].pos for i in vids]
            n = len(pts)
            cx = sum(p[0] for p in pts) / n
            cy = sum(p[1] for p in pts) / n
            sxx = syy = sxy = 0.0
            for (x, y) in pts:
                dx, dy = x - cx, y - cy
                sxx += dx * dx
                syy += dy * dy
                sxy += dx * dy
            sxx /= n
            syy /= n
            sxy /= n
            tr = sxx + syy
            disc = math.sqrt(max(0.0, (tr * 0.5) ** 2 - (sxx * syy - sxy * sxy)))
            l1, l2 = tr * 0.5 + disc, tr * 0.5 - disc
            if abs(sxy) > 1e-9:
                ang = math.atan2(l1 - sxx, sxy)          # major-axis angle
            else:
                ang = 0.0 if sxx >= syy else math.pi / 2
            rx = rf * math.sqrt(2.0 * max(l1, 1e-12))    # =0.58*Rc for a regular ring
            ry = rf * math.sqrt(2.0 * max(l2, 1e-12))
            X, Y = tx((cx, cy))
            deg = -math.degrees(ang)                     # SVG y points down
            out.append(f'<ellipse cx="{X:.2f}" cy="{Y:.2f}" rx="{rx * scale:.2f}" '
                       f'ry="{ry * scale:.2f}" fill="none" stroke="#111" '
                       f'stroke-width="{BOND_W * scale:.2f}" '
                       f'transform="rotate({deg:.1f} {X:.2f} {Y:.2f})"/>')

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
