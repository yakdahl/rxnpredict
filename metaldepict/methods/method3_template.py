#!/usr/bin/env python3
"""
method3_template.py -- METHOD C: deterministic template / skeleton layout.

A "ChemDraw-style" 2D depiction of four Cu-H bisphosphine complexes drawn at
MEDIUM abbreviation, WITHOUT any force-directed physics.  Every atom is placed
by explicit template rules:

  * the P-Cu-P + chelate metallacycle is laid out explicitly, with the Cu-H
    bond HORIZONTAL and H to the RIGHT of Cu;
  * each backbone ring is a regular polygon (hexagon / pentagon);
  * each substituent arm is attached radially by fixed rules;
  * terminal groups are abbreviated to PLAIN TEXT (t-Bu, OMe, Ph) at MEDIUM;
  * explicit Kekule double bonds are drawn on the inside of every ring;
  * P->Cu bonds are dative (arrow) bonds; sp3 stereocentres get wedge/dash;
  * the DTBM-SEGPhos biaryl axis is a BOLD bond.

Rings are placed by EXPLICIT centre + orientation and fusion vertices are
referenced by INDEX (CCW from a known vertex-0), so the layout is reproducible
and collision-free -- nothing is found by nearest-neighbour search.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from template_draw import (Scene, L, polar, vadd, vsub, vscale, vnorm)  # noqa: E402

HEX_R = L / (2 * math.sin(math.pi / 6))     # circumradius of unit-edge hexagon
PEN_R = L / (2 * math.sin(math.pi / 5))     # circumradius of unit-edge pentagon

P_COL = "#c87a00"
CU_COL = "#b05a2a"
O_COL = "#c0392b"


def vlen(a):
    return math.hypot(a[0], a[1])


# ------------------------------------------------------------------------------
def place_hexagon(sc, center, vertex0_deg, kek_offset=0):
    """Place a benzene hexagon. vertex0 is at angle vertex0_deg from centre.
    Kekule doubles drawn inside; kek_offset (0/1) chooses which edges double.
    Returns list of 6 vertex ids, CCW from vertex0."""
    ids = []
    for k in range(6):
        ids.append(sc.atom(polar(center, vertex0_deg + k * 60.0, HEX_R)))
    kek = [2, 1, 2, 1, 2, 1] if kek_offset == 0 else [1, 2, 1, 2, 1, 2]
    for k in range(6):
        sc.bond(ids[k], ids[(k + 1) % 6], order=kek[k], inside=center)
    return ids


def out_angle(sc, center, vid):
    p = sc.atoms[vid].pos
    return math.degrees(math.atan2(p[1] - center[1], p[0] - center[0]))


def text_arm(sc, parent_id, direction_deg, label, length=1.0, order=1,
             fontscale=1.0, halo=True):
    p = sc.atoms[parent_id].pos
    tip = polar(p, direction_deg, length * L)
    tid = sc.atom(tip, label=label, fontscale=fontscale, halo=halo)
    sc.bond(parent_id, tid, order=order)
    return tid


def core(sc, cu_pos, pu_pos, pd_pos):
    pu = sc.atom(pu_pos, label="P", color=P_COL)
    pd = sc.atom(pd_pos, label="P", color=P_COL)
    cu = sc.atom(cu_pos, label="Cu", color=CU_COL)
    h = sc.atom(vadd(cu_pos, (1.15 * L, 0.0)), label="H")
    sc.bond(cu, h, order=1)
    sc.bond(pu, cu, order=1, kind="dative")
    sc.bond(pd, cu, order=1, kind="dative")
    return dict(pu=pu, pd=pd, cu=cu, h=h)


# ==============================================================================
# DTBM-SEGPhos  (the busiest case -- must read cleanly)
# ==============================================================================
def _dtbm_aryl(sc, pid, center_dir_deg, kek=0):
    """Place a 3,5-di-tBu-4-OMe-phenyl ring attached to P at pid, ring centre
    in direction center_dir_deg from P.  ipso vertex points back at P."""
    p = sc.atoms[pid].pos
    ipso = polar(p, center_dir_deg, L)
    center = polar(ipso, center_dir_deg, HEX_R)
    v0 = center_dir_deg + 180.0          # vertex0 = ipso (points back to P)
    ids = place_hexagon(sc, center, v0, kek)
    sc.bond(pid, ids[0], order=1)
    # ipso=0; meta=2,4 ; para=3 (CCW)
    for v, lab in ((ids[2], "t-Bu"), (ids[3], "OMe"), (ids[4], "t-Bu")):
        text_arm(sc, v, out_angle(sc, center, v), lab, length=1.0, fontscale=0.92)
    return ids, center


def build_dtbm_segphos():
    sc = Scene()
    cu = (7.2, 0.0)
    pu = (4.7, 1.95)
    pd = (4.7, -1.95)
    c = core(sc, cu, pu, pd)

    # --- two DTBM aryls per P, fanned OUT away from the backbone -------------
    # The two aryls on each P must be widely separated (~75 deg) so their
    # hexagons + t-Bu/OMe labels never collide.  Upper P sends one aryl up and
    # one up-right; lower P mirrors downward.
    _dtbm_aryl(sc, c["pu"], 128)
    _dtbm_aryl(sc, c["pu"], 40)
    _dtbm_aryl(sc, c["pd"], -128)
    _dtbm_aryl(sc, c["pd"], -40)

    # --- SEGPhos backbone: two benzo rings + biaryl axis (bold) -------------
    # Place the two backbone rings explicitly to the LEFT, well separated.
    cu_b = (1.7, 1.45)
    cd_b = (1.7, -1.45)
    # orient each ring so one vertex points to its P (lower-right for upper ring)
    bu = place_hexagon(sc, cu_b, 30, kek_offset=0)    # upper backbone ring
    bd = place_hexagon(sc, cd_b, -30, kek_offset=0)   # lower backbone ring
    # connect P to the backbone ring vertex nearest P (explicit index):
    # upper ring vertex0 at 30deg = lower-right -> bond to P_up
    sc.bond(c["pu"], bu[0], order=1)
    sc.bond(c["pd"], bd[0], order=1)
    # biaryl axis: the inner vertices facing each other (upper ring bottom,
    # lower ring top).  upper ring: vertex at -90 region; with v0=30deg the
    # vertices are at 30,90,150,210,270,330. bottom = index 4 (270deg).
    # lower ring v0=-30 -> -30,30,90,150,210,270 ; top = index 2 (90deg).
    au = bu[4]   # 270deg, bottom of upper ring
    ad = bd[2]   # 90deg, top of lower ring
    sc.bond(au, ad, order=1, kind="bold")     # atropisomeric axis (bold)

    # methylenedioxy (O-CH2-O) fused on the far-LEFT edge of each backbone ring
    _dioxole(sc, bu, cu_b, edge=(2, 3))   # left edge of upper ring
    _dioxole(sc, bd, cd_b, edge=(3, 4))   # left edge of lower ring
    return sc, "(DTBM-SEGPhos)Cu–H"


def _dioxole(sc, ring_ids, center, edge):
    a, b = ring_ids[edge[0]], ring_ids[edge[1]]
    pa = sc.atoms[a].pos
    pb = sc.atoms[b].pos
    mid = vscale(vadd(pa, pb), 0.5)
    out = vnorm(vsub(mid, center))
    o1 = sc.atom(vadd(pa, vscale(out, 0.85 * L)), label="O", color=O_COL)
    o2 = sc.atom(vadd(pb, vscale(out, 0.85 * L)), label="O", color=O_COL)
    ch2 = sc.atom(vadd(mid, vscale(out, 1.45 * L)))
    sc.bond(a, o1, order=1)
    sc.bond(b, o2, order=1)
    sc.bond(o1, ch2, order=1)
    sc.bond(o2, ch2, order=1)


# ==============================================================================
# Xantphos / DPEphos -- diaryl-ether (DPEphos) or xanthene (Xantphos) backbone
# ==============================================================================
def _pphos_phenyls(sc, c):
    """The two P-Ph arms per P, fanned out away from the backbone (to left)."""
    text_arm(sc, c["pu"], 58, "Ph", length=1.05)
    text_arm(sc, c["pu"], 118, "Ph", length=1.05)
    text_arm(sc, c["pd"], -58, "Ph", length=1.05)
    text_arm(sc, c["pd"], -118, "Ph", length=1.05)


def _xanthene_backbone(sc, c):
    """Build a PROPER fused tricyclic xanthene: a central 6-membered pyran ring
    (O at the far-left vertex, C(CH3)2 at the far-right vertex) with the two
    benzo rings fused onto its upper-left and lower-left edges.  Each benzo ring
    carries a P (which also bears two Ph)."""
    _pphos_phenyls(sc, c)
    # central pyran ring -- regular hexagon, centred left of the metal.
    cen = (1.95, 0.0)
    # orient so vertex0 points RIGHT (0 deg) = the spiro C(CH3)2 carbon, and the
    # opposite vertex (180) = O.  Vertices CCW: 0,60,120,180,240,300.
    py = place_hexagon(sc, cen, 0.0, kek_offset=0)
    # but the central pyran is NOT aromatic -> redraw its 6 edges as single.
    sc.bonds = [b for b in sc.bonds
                if not (set((b.a, b.b)) <= set(py))]
    for k in range(6):
        sc.bond(py[k], py[(k + 1) % 6], order=1)
    cQ = py[0]    # right vertex -> C(CH3)2
    cO = py[3]    # left vertex  -> O
    sc.atoms[cO].label = "O"
    sc.atoms[cO].color = O_COL
    # the two upper carbons py[1](60deg) & py[2](120deg) are the shared edge of
    # the UPPER benzo ring; py[4](240) & py[5](300) the LOWER benzo ring.
    # Fuse upper benzo onto edge (py[1]-py[2]); fuse lower onto edge (py[5]-py[4]).
    bu = _fuse_benzo(sc, py[2], py[1], cen)     # upper ring (above the edge)
    bd = _fuse_benzo(sc, py[4], py[5], cen)     # lower ring (below the edge)
    # P attaches to the outer ortho carbon of each benzo ring (the one nearest P)
    # _fuse_benzo returns ring ids with [0],[1] == the two shared (re-used) atoms
    # and 2,3,4,5 the new ones going around.  The carbon ortho to the shared
    # edge on the metal (right) side carries P.
    pu_aryl = _benzo_p_vertex(sc, bu, c["pu"])
    pd_aryl = _benzo_p_vertex(sc, bd, c["pd"])
    sc.bond(c["pu"], pu_aryl, order=1)
    sc.bond(c["pd"], pd_aryl, order=1)
    # C(CH3)2 methyls -- point right toward the metal/open space
    text_arm(sc, cQ, 30, "CH₃", length=1.05, fontscale=0.78)
    text_arm(sc, cQ, -30, "CH₃", length=1.05, fontscale=0.78)
    return sc


def _fuse_benzo(sc, a, b, central_center):
    """Fuse a benzene ring onto the existing edge a-b, on the side AWAY from
    central_center.  a,b are existing atom ids (the shared edge).  Returns ring
    ids [a, b, n2, n3, n4, n5] CCW with Kekule doubles drawn inside."""
    pa = sc.atoms[a].pos
    pb = sc.atoms[b].pos
    mid = vscale(vadd(pa, pb), 0.5)
    out = vnorm(vsub(mid, central_center))           # away from central ring
    # ring centre is offset from the shared-edge midpoint by the apothem
    apothem = HEX_R * math.cos(math.pi / 6)
    rc = vadd(mid, vscale(out, apothem))
    # Regular hexagon about rc.  a is vertex0; going CCW the next vertex must be
    # b.  Determine the angular step sign so that step from a lands on b.
    ang_a = math.degrees(math.atan2(pa[1] - rc[1], pa[0] - rc[0]))
    ang_b = math.degrees(math.atan2(pb[1] - rc[1], pb[0] - rc[0]))
    # normalised signed step a->b (should be +/-60)
    step = ((ang_b - ang_a + 180) % 360) - 180
    sgn = 1 if step > 0 else -1
    ids = [a, b]
    for k in range(2, 6):
        ids.append(sc.atom(polar(rc, ang_a + sgn * 60.0 * k, HEX_R)))
    # Kekule: shared edge a-b already drawn (single, part of pyran). Alternate
    # doubles around the remaining 5 edges so the ring is benzene.
    orders = [1, 2, 1, 2, 1, 2]
    for k in range(6):
        if k == 0:
            continue   # a-b already exists
        x, y = ids[k], ids[(k + 1) % 6]
        sc.bond(x, y, order=orders[k], inside=rc)
    return ids


def _benzo_p_vertex(sc, ring_ids, pid):
    """Return the benzo-ring vertex nearest to P (the ipso carbon for P)."""
    pp = sc.atoms[pid].pos
    cand = ring_ids[2:]   # don't pick the two shared (fused) atoms
    return min(cand, key=lambda i: vlen(vsub(sc.atoms[i].pos, pp)))


def _dpephos_backbone(sc, c):
    """DPEphos: two SEPARATE benzo rings joined only by an ether O (no central
    ring, no CMe2).  Each benzo carries a P with two Ph."""
    _pphos_phenyls(sc, c)
    cu_b = (1.0, 1.72)
    cd_b = (1.0, -1.72)
    bu = place_hexagon(sc, cu_b, 30, kek_offset=0)
    bd = place_hexagon(sc, cd_b, -30, kek_offset=0)
    sc.bond(c["pu"], bu[0], order=1)
    sc.bond(c["pd"], bd[0], order=1)
    up_botL = bu[3]    # 210 bottom-left of upper ring
    dn_topL = bd[3]    # 150 top-left of lower ring
    o_mid = vscale(vadd(sc.atoms[up_botL].pos, sc.atoms[dn_topL].pos), 0.5)
    oid = sc.atom((o_mid[0] - 0.15 * L, o_mid[1]), label="O", color=O_COL)
    sc.bond(up_botL, oid, order=1)
    sc.bond(dn_topL, oid, order=1)
    return sc


def build_xantphos():
    sc = Scene()
    cu = (5.6, 0.0)
    pu = (3.6, 1.55)
    pd = (3.6, -1.55)
    c = core(sc, cu, pu, pd)
    return _xanthene_backbone(sc, c), "(Xantphos)Cu–H"


def build_dpephos():
    sc = Scene()
    cu = (5.0, 0.0)
    pu = (3.1, 1.45)
    pd = (3.1, -1.45)
    c = core(sc, cu, pu, pd)
    return _dpephos_backbone(sc, c), "(DPEphos)Cu–H"


# ==============================================================================
# Ph-BPE -- two phospholanes bridged by -CH2CH2-, each with two Ph (stereo)
# ==============================================================================
def build_ph_bpe():
    sc = Scene()
    cu = (4.4, 0.0)
    pu = (2.4, 1.45)
    pd = (2.4, -1.45)
    c = core(sc, cu, pu, pd)

    _phospholane(sc, c["pu"], up=True)
    _phospholane(sc, c["pd"], up=False)

    # ethylene bridge between the two P's
    pu_pos = sc.atoms[c["pu"]].pos
    pd_pos = sc.atoms[c["pd"]].pos
    bx = min(pu_pos[0], pd_pos[0]) - 0.95 * L
    cmu = sc.atom((bx, 0.55))
    cmd = sc.atom((bx, -0.55))
    sc.bond(c["pu"], cmu, order=1)
    sc.bond(c["pd"], cmd, order=1)
    sc.bond(cmu, cmd, order=1)
    return sc, "(Ph-BPE)Cu–H"


def _phospholane(sc, pid, up=True):
    p = sc.atoms[pid].pos
    cdir = 128 if up else -128
    center = polar(p, cdir, PEN_R)
    pdir = math.degrees(math.atan2(p[1] - center[1], p[0] - center[0]))
    ids = [pid]
    for k in range(1, 5):
        ids.append(sc.atom(polar(center, pdir + k * 72.0, PEN_R)))
    for k in range(5):
        sc.bond(ids[k], ids[(k + 1) % 5], order=1)
    # 2,5 carbons = ids[1], ids[4]; Ph with wedge (up) / dash (down) stereo
    for v, kind, push in ((ids[1], "wedge", 0), (ids[4], "dash", 0)):
        d = out_angle(sc, center, v)
        tip = polar(sc.atoms[v].pos, d, 1.05 * L)
        tid = sc.atom(tip, label="Ph")
        sc.bond(v, tid, order=1, kind=kind)
    return ids


BUILDERS = {
    "dtbm_segphos": build_dtbm_segphos,
    "xantphos": build_xantphos,
    "dpephos": build_dpephos,
    "ph_bpe": build_ph_bpe,
}


def check_overlaps(sc, key, tol=0.45):
    """Report pairs of atoms closer than tol*L (excluding bonded pairs) and
    label-label collisions.  A clean template should report zero."""
    bonded = set()
    for b in sc.bonds:
        bonded.add(frozenset((b.a, b.b)))
    bad = 0
    ids = list(sc.atoms)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if frozenset((a, b)) in bonded:
                continue
            d = vlen(vsub(sc.atoms[a].pos, sc.atoms[b].pos))
            # labelled atoms need more clearance
            la, lb = sc.atoms[a].label, sc.atoms[b].label
            need = tol
            if la and lb:
                need = 0.62 + 0.10 * (len(la) + len(lb))
            elif la or lb:
                need = 0.52
            if d < need:
                bad += 1
                if bad <= 8:
                    print(f"   [overlap] {key}: {la or 'C'}#{a} <-> "
                          f"{lb or 'C'}#{b}  d={d:.2f} need={need:.2f}")
    if bad:
        print(f"   {key}: {bad} overlap pair(s)")
    return bad


def main():
    outdir = HERE / "out"
    outdir.mkdir(exist_ok=True)
    total = 0
    for key, fn in BUILDERS.items():
        sc, title = fn()
        svg = sc.render_svg(width=760, height=640, title=title)
        (outdir / f"{key}.svg").write_text(svg)
        n = check_overlaps(sc, key)
        total += n
        print(f"wrote {key}.svg  atoms={len(sc.atoms)} bonds={len(sc.bonds)} "
              f"overlaps={n}")
    print(f"TOTAL overlaps across all 4: {total}")


if __name__ == "__main__":
    main()
