#!/usr/bin/env python3
"""
method_d_force.py -- METHOD D: force-directed graph-drawing layout.

Distinct from the rigid-body physics engine (Method 1). Here the *global*
placement comes from a force-directed graph layout (networkx spring /
Kamada-Kawai) run on a REDUCED graph in which every fused-ring system and every
MEDIUM abbreviation group (t-Bu / OMe / Ph) is collapsed to a single super-node.
After the spring layout settles the super-nodes, each ring system is EXPANDED
back into a rigid regular polygon oriented along its attachment bond(s), and the
ChemDraw-style depiction is drawn directly with matplotlib.

Pipeline:
  load depiction JSON (atoms/bonds/rings/groups/metal, with explicit Kekule
  bond orders already assigned)
    -> collapse ring-systems + abbreviation groups + leave P/Cu/H/linker atoms
       as point super-nodes
    -> networkx force-directed layout on the reduced graph  (GLOBAL placement)
    -> expand rings as regular polygons (rigid, oriented by attachment)
    -> place P, Cu, H (Cu-H horizontal, H to the right), linkers, group labels
    -> draw: THICK bonds, inside Kekule double bonds, plain-text labels with a
       text-only white halo, wedge/dash, dative P->Cu arrows, bold biaryl axis.

Usage:
  python3 method_d_force.py <key> <out.png>
  python3 method_d_force.py all <prefix>          # renders all four ligands
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon, FancyArrowPatch
import matplotlib.patheffects as pe

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

LIGANDS = {
    "dtbm_segphos": "data/dtbm_segphos.json",
    "xantphos": "data/xantphos.json",
    "dpephos": "data/dpephos.json",
    "ph_bpe": "data/ph_bpe.json",
}

# ChemDraw-ish palette
COLORS = {"C": "#1a1d22", "P": "#e8821e", "O": "#cc2b2b", "N": "#2b5fcc",
          "F": "#2aa84a", "Cl": "#2aa84a", "Cu": "#b5651d", "H": "#1a1d22",
          "S": "#c9a21a", "B": "#cc7a55"}
LABEL_ELEMENTS = {"P", "O", "N", "F", "Cl", "Cu", "S", "B", "Si"}

BL = 1.0            # one standard bond length (world units)
HALO = "#ffffff"


# --------------------------------------------------------------------------- IO
def load(key: str) -> dict:
    return json.loads((HERE.parent / LIGANDS[key]).read_text())


# ---------------------------------------------------- reduced-graph construction
def ring_systems(rings: list[list[int]]) -> list[set[int]]:
    """Merge rings that share atoms into fused ring systems."""
    systems: list[set[int]] = []
    for ring in rings:
        rs = set(ring)
        hit = [s for s in systems if s & rs]
        if not hit:
            systems.append(rs)
        else:
            base = hit[0]
            for other in hit[1:]:
                base |= other
                systems.remove(other)
            base |= rs
    return systems


def build_reduced(D: dict):
    """Return (G, node_of_atom, info) for the collapsed super-node graph."""
    atoms = {a["id"]: a for a in D["atoms"]}
    groups = [g for g in D["groups"] if g["level"] == 1]

    # atoms that belong to a MEDIUM abbreviation group collapse to a plain-text
    # label and must NOT be drawn as a ring -- so exclude any ring fully inside a
    # group (the four "Ph" phenyls etc.) from the ring-system perception.
    group_atoms: set[int] = set()
    for g in groups:
        group_atoms |= set(g["atoms"])
    backbone_rings = [r for r in D["rings"] if not (set(r) <= group_atoms)]
    rsystems = ring_systems(backbone_rings)

    node_of_atom: dict[int, str] = {}
    info: dict[str, dict] = {}

    # abbreviation-group super-nodes FIRST (highest priority -> plain-text label)
    for g in groups:
        nid = f"G:{g['id']}"
        info[nid] = {"kind": "group", "label": g["label"], "attach": g["attach"],
                     "anchor": g.get("anchor"), "atoms": list(g["atoms"]), "group": g}
        for a in g["atoms"]:
            node_of_atom[a] = nid

    # ring-system super-nodes (backbone rings only)
    for i, sys_atoms in enumerate(rsystems):
        nid = f"R{i}"
        info[nid] = {"kind": "ring", "atoms": sorted(sys_atoms),
                     "rings": [r for r in backbone_rings if set(r) & sys_atoms]}
        for a in sys_atoms:
            node_of_atom.setdefault(a, nid)

    # remaining free atoms (P, Cu, H, linker O / CH2 ...) are point super-nodes
    for a in D["atoms"]:
        if a["id"] not in node_of_atom:
            nid = f"A{a['id']}"
            info[nid] = {"kind": "atom", "atom": a["id"], "el": a["el"]}
            node_of_atom[a["id"]] = nid

    # edges of the reduced graph (collapse bonds crossing super-nodes)
    G = nx.Graph()
    G.add_nodes_from(info.keys())
    cross = {}                       # (na, nb) -> list of (atomA, atomB, bond)
    for b in D["bonds"]:
        na, nb = node_of_atom[b["a"]], node_of_atom[b["b"]]
        if na == nb:
            continue
        key = (na, nb) if na < nb else (nb, na)
        cross.setdefault(key, []).append((b["a"], b["b"], b))
        G.add_edge(na, nb)
    return G, node_of_atom, info, cross, atoms


# --------------------------------------------------------- force-directed layout
def force_layout(G: nx.Graph, info: dict) -> dict[str, np.ndarray]:
    """GLOBAL placement via a force-directed graph-drawing algorithm.

    Kamada-Kawai (energy-minimising spring layout based on graph-theoretic
    distances) for the global skeleton, refined by a Fruchterman-Reingold
    spring pass. This is the heart of Method D -- positions of every ring/group
    super-node come purely from the graph layout, not from any 3D conformer.
    """
    # desired edge length ~ sum of the two node radii so big ring systems get
    # pushed apart proportionally to their drawn size.
    n = G.number_of_nodes()
    try:
        pos = nx.kamada_kawai_layout(G, scale=1.0)
    except Exception:
        pos = nx.spring_layout(G, seed=7, k=1.5 / math.sqrt(max(n, 2)),
                               iterations=400)
    # Fruchterman-Reingold refinement seeded by KK -> cleaner, less cramped
    pos = nx.spring_layout(G, pos=pos, seed=7,
                           k=1.7 / math.sqrt(max(n, 2)), iterations=350)

    P = {k: np.array(v, float) for k, v in pos.items()}
    # scale so the median super-node spacing equals a comfortable multiple of BL
    cen = np.mean(list(P.values()), axis=0)
    for k in P:
        P[k] = P[k] - cen
    dists = [np.linalg.norm(P[a] - P[b]) for a, b in G.edges()]
    med = np.median(dists) if dists else 1.0
    target = 3.9 * BL
    s = target / med if med > 1e-6 else 1.0
    for k in P:
        P[k] = P[k] * s

    # ---- de-overlap pass: push ring/group super-nodes apart by their drawn
    # radius so big fused systems (DTBM aryls) stop sitting on top of each other.
    # This stays a force-directed step (repulsion along graph-free pairs).
    def node_radius(nid):
        ninfo = info[nid]
        if ninfo["kind"] == "ring":
            n = len(ninfo["atoms"])
            # fused-system footprint + room for radial substituent labels so the
            # de-overlap pass also keeps t-Bu/OMe labels of neighbouring rings apart
            return 1.0 + 0.34 * n
        if ninfo["kind"] == "group":
            return 1.3
        return 0.65
    keys = list(P.keys())
    for _ in range(120):
        moved = 0.0
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                d = P[b] - P[a]
                dist = np.linalg.norm(d) or 1e-6
                mind = node_radius(a) + node_radius(b)
                if dist < mind:
                    push = (mind - dist) * 0.5
                    dirv = d / dist
                    P[a] = P[a] - dirv * push
                    P[b] = P[b] + dirv * push
                    moved += push
        if moved < 0.01:
            break
    return P


# ---------------------------------------------- ring expansion (regular polygons)
def order_ring(ring_atoms: list[int], bondset: dict) -> list[int]:
    """Return ring atom ids in cyclic connectivity order."""
    adj = {a: [] for a in ring_atoms}
    rs = set(ring_atoms)
    for (a, b) in bondset:
        if a in rs and b in rs:
            adj[a].append(b)
            adj[b].append(a)
    start = ring_atoms[0]
    order = [start]
    prev = None
    cur = start
    while len(order) < len(ring_atoms):
        nxts = [x for x in adj[cur] if x != prev and (x not in order or x == start)]
        nxts = [x for x in adj[cur] if x != prev and x not in order]
        if not nxts:
            break
        prev, cur = cur, nxts[0]
        order.append(cur)
    return order


def regular_polygon(center: np.ndarray, n: int, start_angle: float,
                     radius: float) -> list[np.ndarray]:
    return [center + radius * np.array([math.cos(start_angle + 2 * math.pi * k / n),
                                        math.sin(start_angle + 2 * math.pi * k / n)])
            for k in range(n)]


def circumradius(n: int) -> float:
    return BL / (2 * math.sin(math.pi / n))


def expand_ring_system(nid: str, info_node: dict, center: np.ndarray,
                       neighbor_dirs: list[np.ndarray], bondset: dict) -> dict[int, np.ndarray]:
    """Expand a fused ring system into rigid regular polygons.

    Single ring: one regular n-gon centred at `center`, rotated so the average
    attachment direction points to a vertex (puts substituents radially out).
    Fused systems: place the first ring, then attach each subsequent ring across
    its shared edge as a regular polygon on the far side.
    """
    rings = [order_ring(r, bondset) for r in info_node["rings"]]
    coords: dict[int, np.ndarray] = {}

    # average attachment direction (toward the neighbour super-nodes)
    if neighbor_dirs:
        avg = np.mean(neighbor_dirs, axis=0)
        if np.linalg.norm(avg) < 1e-6:
            avg = neighbor_dirs[0]
        attach_ang = math.atan2(avg[1], avg[0])
    else:
        attach_ang = 0.0

    # ---- place first ring as a regular polygon ----
    r0 = rings[0]
    n0 = len(r0)
    R0 = circumradius(n0)
    # rotate so a vertex sits along the attachment direction (substituents out)
    pts = regular_polygon(center, n0, attach_ang, R0)
    for a, p in zip(r0, pts):
        coords[a] = p

    placed = set(r0)
    # ---- fuse remaining rings across shared edges ----
    remaining = rings[1:]
    guard = 0
    while remaining and guard < 50:
        guard += 1
        progress = False
        for ring in list(remaining):
            shared = [a for a in ring if a in placed]
            if len(shared) < 2:
                continue
            # find a shared edge (two adjacent placed atoms also adjacent in ring)
            ringset = ring
            edge = None
            for i in range(len(ring)):
                a, b = ring[i], ring[(i + 1) % len(ring)]
                if a in placed and b in placed:
                    edge = (a, b)
                    break
            if edge is None:
                continue
            n = len(ring)
            Rn = circumradius(n)
            pa, pb = coords[edge[0]], coords[edge[1]]
            mid = (pa + pb) / 2
            # apothem direction = away from already-placed centroid
            cen_placed = np.mean([coords[a] for a in placed], axis=0)
            edir = pb - pa
            edge_len = np.linalg.norm(edir) or 1.0
            normal = np.array([-edir[1], edir[0]]) / edge_len
            if np.dot(mid - cen_placed, normal) < 0:
                normal = -normal
            apothem = Rn * math.cos(math.pi / n)
            new_center = mid + normal * apothem
            # angle of edge start vertex on the new polygon
            ang_a = math.atan2(pa[1] - new_center[1], pa[0] - new_center[0])
            # order the ring so it starts at edge[0] and goes toward edge[1]
            idx = ring.index(edge[0])
            if ring[(idx + 1) % n] != edge[1]:
                ring = ring[::-1]
                idx = ring.index(edge[0])
            step = 2 * math.pi / n
            # confirm direction sign by matching vertex 1 to pb
            cand1 = new_center + Rn * np.array([math.cos(ang_a + step), math.sin(ang_a + step)])
            sign = 1 if np.linalg.norm(cand1 - pb) < np.linalg.norm(
                (new_center + Rn * np.array([math.cos(ang_a - step), math.sin(ang_a - step)])) - pb) else -1
            for k in range(n):
                a = ring[(idx + k) % n]
                if a in coords:
                    continue
                ang = ang_a + sign * step * k
                coords[a] = new_center + Rn * np.array([math.cos(ang), math.sin(ang)])
            placed |= set(ring)
            remaining.remove(ring if ring in remaining else ring[::-1]) if False else None
            try:
                remaining.remove(ring)
            except ValueError:
                # ring was reversed; remove the matching original
                for rr in list(remaining):
                    if set(rr) == set(ring):
                        remaining.remove(rr)
                        break
            progress = True
        if not progress:
            break
    # any unplaced (degenerate) -> drop at center
    for ring in remaining:
        for a in ring:
            coords.setdefault(a, center.copy())
    return coords


# ----------------------------------------------------------------- full assembly
def assemble(D: dict):
    G, node_of_atom, info, cross, atoms = build_reduced(D)
    P = force_layout(G, info)

    bondset = {(min(b["a"], b["b"]), max(b["a"], b["b"])): b for b in D["bonds"]}
    coords: dict[int, np.ndarray] = {}

    metal = D["metal"]
    metal_id, hyd_id = metal["id"], metal["hydride"]

    # ---- expand ring systems ----
    for nid, ninfo in info.items():
        if ninfo["kind"] != "ring":
            continue
        center = P[nid]
        # neighbour directions in the reduced layout
        ndirs = []
        for nb in G.neighbors(nid):
            d = P[nb] - center
            if np.linalg.norm(d) > 1e-6:
                ndirs.append(d / np.linalg.norm(d))
        rc = expand_ring_system(nid, ninfo, center, ndirs, bondset)
        coords.update(rc)

    # ---- place free atoms (P, Cu, H, linkers) from the GLOBAL layout ----
    for nid, ninfo in info.items():
        if ninfo["kind"] != "atom":
            continue
        coords[ninfo["atom"]] = P[nid].copy()

    # adjacency over real atoms (for local geometric cleanup)
    adj = {a["id"]: [] for a in D["atoms"]}
    for b in D["bonds"]:
        adj[b["a"]].append(b["b"])
        adj[b["b"]].append(b["a"])

    donors = metal["donors"]
    free_ids = [ninfo["atom"] for nid, ninfo in info.items() if ninfo["kind"] == "atom"]

    # ---- local geometric cleanup: snap every FREE atom to ideal bond geometry
    # relative to its already-placed (ring / other free) neighbours, keeping the
    # force-directed global arrangement but giving clean ~120 deg bond angles and
    # one standard bond length. Cu and H are templated separately below. ----
    fixed = set()        # atoms whose position is authoritative (ring atoms)
    for nid, ninfo in info.items():
        if ninfo["kind"] == "ring":
            fixed |= set(ninfo["atoms"])
    # linkers / P first toward their ring neighbour(s)
    for _ in range(40):
        for aid in free_ids:
            if aid in (metal_id, hyd_id):
                continue
            nbrs = [n for n in adj[aid] if n in coords and n != hyd_id and n != metal_id]
            ringn = [n for n in nbrs if n in fixed]
            placed_dir = P[info_atom_node(info, aid)] - np.mean(
                [coords[n] for n in nbrs], axis=0) if nbrs else np.array([1.0, 0.0])
            if not nbrs:
                continue
            if len(ringn) >= 2:
                # bridge atom (e.g. ether O / CH2 between two rings): bisector,
                # pushed away from the mean of those rings so it bows outward
                a, b = ringn[0], ringn[1]
                mid = (coords[a] + coords[b]) / 2
                base = coords[a] - coords[b]
                bl = np.linalg.norm(base) or 1.0
                normal = np.array([-base[1], base[0]]) / bl
                # choose the side matching the global force-layout position
                if np.dot(coords[aid] - mid, normal) < 0:
                    normal = -normal
                half = bl / 2
                h = math.sqrt(max(BL * BL - half * half, 0.04 * BL * BL))
                coords[aid] = mid + normal * h
            else:
                # terminal-ish free atom: keep direction from neighbour centroid
                cen = np.mean([coords[n] for n in nbrs], axis=0)
                d = coords[aid] - cen
                if np.linalg.norm(d) < 1e-6:
                    d = placed_dir
                d = d / (np.linalg.norm(d) or 1.0)
                coords[aid] = cen + d * BL

    # ---- TEMPLATE the Cu coordination sphere symmetrically (clean ChemDraw) ----
    # Place Cu so the two P donors sit symmetrically; Cu-H horizontal, H right.
    # The donors keep their force-directed positions (so the backbone hangs off
    # them naturally); Cu goes on the bisector of the P...P opening, outboard.
    p0, p1 = coords[donors[0]], coords[donors[1]]
    pmid = (p0 + p1) / 2
    lig_pts = [coords[a["id"]] for a in D["atoms"]
               if a["id"] in coords and a["id"] not in (metal_id, hyd_id)]
    backbone_cen = np.mean(lig_pts, axis=0)
    away = pmid - backbone_cen
    if np.linalg.norm(away) < 1e-6:
        away = np.array([0.0, -1.0])
    away = away / np.linalg.norm(away)
    ppdist = np.linalg.norm(p0 - p1)
    # Cu far enough out that both P->Cu bonds are a sensible length
    cu = pmid + away * max(1.6 * BL, 1.0 * ppdist)
    coords[metal_id] = cu
    coords[hyd_id] = cu + away * BL * 1.6     # provisional; made horizontal later

    # group-label positions computed after final coords (in draw), so just stub
    group_label_pos: dict[str, np.ndarray] = {}

    return D, info, node_of_atom, cross, coords, group_label_pos, bondset


def info_atom_node(info, aid):
    for nid, ninfo in info.items():
        if ninfo.get("kind") == "atom" and ninfo.get("atom") == aid:
            return nid
    return None


def align_cu_h(coords: dict[int, np.ndarray], metal_id: int, hyd_id: int):
    """Rotate whole assembly so Cu-H is horizontal with H to the RIGHT."""
    cu, h = coords[metal_id], coords[hyd_id]
    d = h - cu
    if np.linalg.norm(d) < 1e-6:
        return
    ang = -math.atan2(d[1], d[0])
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s], [s, c]])
    for k in coords:
        coords[k] = cu + R @ (coords[k] - cu)
    # ensure H strictly to the right
    coords[hyd_id] = cu + np.array([np.linalg.norm(d), 0.0])


# --------------------------------------------------------------------- rendering
def draw(D, info, node_of_atom, cross, coords, group_label_pos, bondset, out_png):
    metal = D["metal"]
    metal_id, hyd_id = metal["id"], metal["hydride"]
    atoms = {a["id"]: a for a in D["atoms"]}

    # ---- group-label positions: fan each label into an OPEN direction off its
    # attach atom (away from every other neighbour of that atom: the metal, the
    # backbone, and sibling group labels). This keeps Ph/t-Bu/OMe labels from
    # stacking and from crossing the coordination bonds. ----
    adj = {a["id"]: [] for a in D["atoms"]}
    for b in D["bonds"]:
        adj[b["a"]].append(b["b"])
        adj[b["b"]].append(b["a"])
    # group attach atom -> list of group nids hanging off it
    groups_on = {}
    for nid, ninfo in info.items():
        if ninfo["kind"] == "group":
            groups_on.setdefault(ninfo["attach"], []).append(nid)

    glabel = {}
    metal_id = D["metal"]["id"]
    # ring centroid lookup for radial placement of ring substituents
    ring_centroid = {}
    for nid, ninfo in info.items():
        if ninfo["kind"] == "ring":
            c = np.mean([coords[a] for a in ninfo["atoms"] if a in coords], axis=0)
            for a in ninfo["atoms"]:
                ring_centroid[a] = c

    for attach, gnids in groups_on.items():
        ap = coords[attach]
        attach_on_ring = attach in ring_centroid
        if attach_on_ring:
            # substituent on a ring atom (t-Bu / OMe on DTBM aryls): point
            # RADIALLY outward from the ring centroid -> never overlaps the ring
            base = ap - ring_centroid[attach]
            if np.linalg.norm(base) < 1e-6:
                base = np.array([1.0, 0.0])
            base = base / np.linalg.norm(base)
            base_ang = math.atan2(base[1], base[0])
            ng = len(gnids)
            spread = math.radians(70) if ng > 1 else 0.0
        else:
            # substituent on a free atom (P): point away from every non-group
            # neighbour (Cu, backbone) so labels fan into open space
            occupied = []
            for n in adj[attach]:
                if node_of_atom.get(n) and info[node_of_atom[n]]["kind"] == "group":
                    continue
                if n in coords:
                    d = coords[n] - ap
                    if np.linalg.norm(d) > 1e-6:
                        occupied.append(d / np.linalg.norm(d))
            if occupied:
                base = -np.mean(occupied, axis=0)
                if np.linalg.norm(base) < 1e-6:
                    base = np.array([0.0, 1.0])
            else:
                base = np.array([1.0, 0.0])
            base = base / (np.linalg.norm(base) or 1.0)
            base_ang = math.atan2(base[1], base[0])
            ng = len(gnids)
            spread = math.radians(94) if ng > 1 else 0.0
        for i, nid in enumerate(gnids):
            off = (i - (ng - 1) / 2) * (spread / max(ng - 1, 1)) if ng > 1 else 0.0
            ang = base_ang + off
            out = np.array([math.cos(ang), math.sin(ang)])
            dist = BL * 1.32 if not attach_on_ring else BL * 1.2
            glabel[nid] = ap + out * dist

    # bounds
    allpts = list(coords.values()) + list(glabel.values())
    xs = [p[0] for p in allpts]
    ys = [p[1] for p in allpts]
    pad = 1.1
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad + 1.0   # headroom for the title

    W = xmax - xmin
    Hh = ymax - ymin
    DPI = 150
    # constant pixels-per-bond-length so every ligand draws at the SAME bond
    # length and line/label weight (DTBM no longer shrinks to illegibility).
    PX_PER_BL = 66.0
    fig_w = max(5.6, min(12.0, W * PX_PER_BL / DPI))
    fig_h = fig_w * Hh / W
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # line-width proportions tied to bond length in *data* units -> consistent
    LW = 5.2          # thick bond
    BOLDW = 9.2       # bold biaryl axis
    DGAP = 0.21 * BL  # inside double-bond offset (visible inside thick bonds)
    DSHRINK = 0.15    # double-bond inner line shortening at each end
    FONT = 19
    SFONT = 17

    label_atoms = set()   # atoms drawn as text -> shrink bonds back from them
    for aid, a in atoms.items():
        if aid not in coords:
            continue
        if a["el"] in LABEL_ELEMENTS or aid == hyd_id or aid == metal_id:
            label_atoms.add(aid)

    # group attach atoms are drawn as the ring vertex; the label is a separate
    # node one bond away.

    def is_hidden(aid):
        nid = node_of_atom.get(aid)
        return nid is not None and info[nid]["kind"] == "group"

    # ---- ring interior directions for inside double bonds ----
    ring_inside = {}
    for nid, ninfo in info.items():
        if ninfo["kind"] != "ring":
            continue
        for ring in ninfo["rings"]:
            pts = [coords[a] for a in ring if a in coords]
            if len(pts) < 3:
                continue
            c = np.mean(pts, axis=0)
            ro = order_ring(ring, bondset)
            for i in range(len(ro)):
                a, b = ro[i], ro[(i + 1) % len(ro)]
                if a not in coords or b not in coords:
                    continue
                mid = (coords[a] + coords[b]) / 2
                d = c - mid
                nl = np.linalg.norm(d) or 1.0
                ring_inside[(min(a, b), max(a, b))] = d / nl

    def inset_for(aid):
        if aid == metal_id:
            return 0.30 * BL
        if aid in label_atoms:
            return 0.24 * BL
        return 0.0

    # ---- draw bonds (skip bonds internal to a collapsed group) ----
    for b in D["bonds"]:
        a1, a2 = b["a"], b["b"]
        if a1 not in coords or a2 not in coords:
            continue
        if is_hidden(a1) or is_hidden(a2):
            continue   # internal to / dangling into a collapsed group; drawn as label bond below
        p1 = coords[a1].astype(float).copy()
        p2 = coords[a2].astype(float).copy()
        u = p2 - p1
        L = np.linalg.norm(u) or 1.0
        u = u / L
        # inset behind labels
        p1 = p1 + u * inset_for(a1)
        p2 = p2 - u * inset_for(a2)
        perp = np.array([-u[1], u[0]])

        if b["type"] == "dative":
            # arrow from donor (P) to Cu
            if a1 == metal_id:
                frm, to = p2, p1
                frm_id, to_id = a2, a1
            else:
                frm, to = p1, p2
                frm_id, to_id = a1, a2
            v = to - frm
            vl = np.linalg.norm(v) or 1.0
            v = v / vl
            frm2 = frm + v * inset_for(frm_id)
            to2 = to - v * (inset_for(to_id) + 0.10 * BL)
            arr = FancyArrowPatch(frm2, to2, arrowstyle="-|>", mutation_scale=18,
                                  lw=3.4, color="#3a3a3a", shrinkA=0, shrinkB=0,
                                  joinstyle="round", capstyle="round", zorder=2)
            ax.add_patch(arr)
            continue

        # wedge / dash on sp3 stereocentres
        if b.get("wedge") in ("up", "down"):
            a0 = b.get("a0", a1)
            narrow = p1 if a0 == a1 else p2
            wide = p2 if a0 == a1 else p1
            w = 0.13 * BL
            if b["wedge"] == "up":
                tri = MplPolygon([narrow, wide + perp * w, wide - perp * w],
                                 closed=True, facecolor="#1a1d22",
                                 edgecolor="#1a1d22", lw=0.5, zorder=3,
                                 joinstyle="round")
                ax.add_patch(tri)
            else:
                n = 6
                for k in range(1, n + 1):
                    t = k / (n + 1)
                    w2 = w * t
                    c = narrow + (wide - narrow) * t
                    ax.add_line(Line2D([c[0] + perp[0] * w2, c[0] - perp[0] * w2],
                                       [c[1] + perp[1] * w2, c[1] - perp[1] * w2],
                                       color="#1a1d22", lw=LW * 0.8,
                                       solid_capstyle="round", zorder=3))
            continue

        bold = b.get("axis", False)
        lw = BOLDW if bold else LW
        ax.add_line(Line2D([p1[0], p2[0]], [p1[1], p2[1]], color="#1a1d22",
                           lw=lw, solid_capstyle="round", solid_joinstyle="round",
                           zorder=2))

        # explicit Kekule double bond -> inside line
        if b.get("order") == 2 or b.get("order") == 2.0:
            key = (min(a1, a2), max(a1, a2))
            ins = ring_inside.get(key)
            if ins is not None:
                off = ins * DGAP
            else:
                off = perp * DGAP
            q1 = p1 + off + u * (L * DSHRINK)
            q2 = p2 + off - u * (L * DSHRINK)
            ax.add_line(Line2D([q1[0], q2[0]], [q1[1], q2[1]], color="#1a1d22",
                               lw=LW * 0.92, solid_capstyle="round", zorder=2))

    # ---- group label bonds: from attach atom out to the label. If the bond
    # from the attach (stereocentre) to the group anchor carries a wedge/dash
    # (Ph-BPE), draw the label bond AS that wedge -> stereochemistry survives the
    # MEDIUM abbreviation. ----
    for nid, ninfo in info.items():
        if ninfo["kind"] != "group":
            continue
        attach = ninfo["attach"]
        if attach not in coords:
            continue
        p1 = coords[attach].astype(float).copy()
        lp = glabel[nid]
        u = lp - p1
        L = np.linalg.norm(u) or 1.0
        u = u / L
        p1 = p1 + u * inset_for(attach)
        lab = ninfo["label"]
        gap = 0.16 * BL + 0.052 * BL * len(lab)
        p2 = lp - u * gap
        perp = np.array([-u[1], u[0]])
        # find a wedge on attach<->anchor (or any attach<->group-atom) bond
        wedge = None
        anchor = ninfo.get("anchor")
        gatomset = set(ninfo["atoms"])
        for bb in D["bonds"]:
            o = bb["b"] if bb["a"] == attach else (bb["a"] if bb["b"] == attach else None)
            if o is not None and o in gatomset and bb.get("wedge") in ("up", "down"):
                wedge = bb["wedge"]
                # narrow end at the stereocentre (attach) if a0==attach
                break
        if wedge == "up":
            w = 0.13 * BL
            tri = MplPolygon([p1, p2 + perp * w, p2 - perp * w], closed=True,
                             facecolor="#1a1d22", edgecolor="#1a1d22", lw=0.5,
                             zorder=3, joinstyle="round")
            ax.add_patch(tri)
        elif wedge == "down":
            w = 0.13 * BL
            n = 6
            for k in range(1, n + 1):
                t = k / (n + 1)
                w2 = w * t
                c = p1 + (p2 - p1) * t
                ax.add_line(Line2D([c[0] + perp[0] * w2, c[0] - perp[0] * w2],
                                   [c[1] + perp[1] * w2, c[1] - perp[1] * w2],
                                   color="#1a1d22", lw=LW * 0.8,
                                   solid_capstyle="round", zorder=3))
        else:
            ax.add_line(Line2D([p1[0], p2[0]], [p1[1], p2[1]], color="#1a1d22",
                               lw=LW, solid_capstyle="round", zorder=2))

    halo_pe = [pe.withStroke(linewidth=5.0, foreground=HALO)]

    # ---- atom labels ----
    for aid, a in atoms.items():
        if aid not in coords or is_hidden(aid):
            continue
        p = coords[aid]
        el = a["el"]
        txt = None
        if aid == metal_id:
            txt, col, fs = "Cu", COLORS["Cu"], FONT
        elif aid == hyd_id:
            txt, col, fs = "H", COLORS["H"], FONT
        elif el in LABEL_ELEMENTS:
            txt, col, fs = el, COLORS.get(el, "#1a1d22"), FONT
        if txt is not None:
            ax.text(p[0], p[1], txt, ha="center", va="center", fontsize=fs,
                    fontweight="bold", color=col, zorder=5,
                    path_effects=halo_pe, fontfamily="sans-serif")

    # ---- group labels (plain text, halo only behind text) ----
    for nid, ninfo in info.items():
        if ninfo["kind"] != "group":
            continue
        lp = glabel[nid]
        ax.text(lp[0], lp[1], ninfo["label"], ha="center", va="center",
                fontsize=SFONT, fontweight="bold", color="#1a1d22", zorder=5,
                path_effects=halo_pe, fontfamily="sans-serif")

    # ---- title ----
    name = D["meta"]["name"]
    ax.text(0.02, 0.97, name, transform=ax.transAxes, ha="left", va="top",
            fontsize=15, fontweight="bold", color="#222")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_png, dpi=DPI, facecolor="white")
    plt.close(fig)


def render(key: str, out_png: str):
    D = load(key)
    (D, info, node_of_atom, cross, coords,
     glp, bondset) = assemble(D)
    align_cu_h(coords, D["metal"]["id"], D["metal"]["hydride"])
    draw(D, info, node_of_atom, cross, coords, glp, bondset, out_png)
    print(f"[D] {key} -> {out_png}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    key, out = sys.argv[1], sys.argv[2]
    if key == "all":
        for k in LIGANDS:
            render(k, f"{out}{k}.png")
    else:
        render(key, out)


if __name__ == "__main__":
    main()
