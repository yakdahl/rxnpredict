"""
method3_template.py
===================
METHOD C (REVISED) -- a DETERMINISTIC TEMPLATE / skeleton layout for the four
Cu-H bisphosphine complexes, then cleaned by the existing homebrew viewer's
physics.

Design
------
Instead of asking RDKit's CoordGen for 2D coordinates (Method A) we PLACE the
depiction by explicit chemical-drawing rules and emit exactly the same viewer
JSON schema (`data/<lig>.json`) so the existing homebrew viewer renders it and
its little physics engine (bond/angle springs + repulsion) cleans up the few
collisions a purely deterministic template leaves behind.

The rules, in drawing order:

  1.  Put the metallacycle on the canvas FIRST:
        * Cu at the origin.
        * Cu-H bond HORIZONTAL with H to the RIGHT.
        * the two P donors symmetric about the horizontal axis, up-left and
          down-left of Cu (P-Cu-P opening to the left, ~110 deg bite).
        * each P then carries the chelate backbone toward the left, and its two
          substituent aryls/alkyls fan out to the left.

  2.  GROW THE GRAPH by breadth-first traversal from Cu.  Every atom is given a
        position the first time it is reached, using a parent position + an
        outgoing direction:
          - ring atoms: when we first enter a ring, lay the WHOLE ring down as a
            perfect regular n-gon, oriented so the entry bond points "outward"
            (away from the metal / parent), giving perfect polygons for free.
          - chain atoms: a 120-deg zig-zag, alternating the turn so chains do
            not curl back on themselves.
          - substituents off a ring/chain atom: radial, bisecting the exterior
            angle so they point away from the ring interior.

  3.  Collapsed abbreviation groups (t-Bu / OMe / Ph) are positioned as super
        nodes by the viewer at their member centroid; because we lay the member
        atoms out radially, the centroid sits where a chemist would draw the
        label.

The result is a clean, deterministic skeleton; the viewer's physics only has to
nudge the handful of arms that land near a neighbour, while every ring stays a
frozen regular polygon (the viewer treats ring systems as rigid bodies).
"""

from __future__ import annotations

import math
from collections import deque

# one standard bond length for the whole drawing (viewer normalises again)
BL = 1.5


# --------------------------------------------------------------------------- #
#  small geometry helpers
# --------------------------------------------------------------------------- #
def _unit(dx: float, dy: float) -> tuple[float, float]:
    d = math.hypot(dx, dy) or 1.0
    return dx / d, dy / d


def _rot(vx: float, vy: float, ang: float) -> tuple[float, float]:
    c, s = math.cos(ang), math.sin(ang)
    return vx * c - vy * s, vx * s + vy * c


def _regular_polygon(center, start_vertex, n, ccw=True):
    """
    Vertices of a regular n-gon with circumradius R chosen so the edge length is
    BL, the FIRST vertex placed at `start_vertex` and the rest walking around
    `center` (CCW or CW).  Returns a list of (x, y) of length n with index 0 ==
    start_vertex.
    """
    cx, cy = center
    sx, sy = start_vertex
    a0 = math.atan2(sy - cy, sx - cx)
    R = BL / (2.0 * math.sin(math.pi / n))
    step = (2 * math.pi / n) * (1 if ccw else -1)
    return [(cx + R * math.cos(a0 + step * k),
             cy + R * math.sin(a0 + step * k)) for k in range(n)]


# --------------------------------------------------------------------------- #
#  the template layout
# --------------------------------------------------------------------------- #
class TemplateLayout:
    """Compute deterministic 2D coordinates for one depiction graph (dict)."""

    def __init__(self, dep: dict):
        self.dep = dep
        self.metal = dep["metal"]["id"]
        self.hydride = dep["metal"]["hydride"]
        self.donors = list(dep["metal"]["donors"])
        self.atoms = {a["id"]: a for a in dep["atoms"]}
        self.pos: dict[int, tuple[float, float]] = {}
        self.placed: set[int] = set()

        # adjacency (heavy-atom graph, undirected) from the bond list
        self.adj: dict[int, list[int]] = {a["id"]: [] for a in dep["atoms"]}
        for b in dep["bonds"]:
            self.adj[b["a"]].append(b["b"])
            self.adj[b["b"]].append(b["a"])

        # rings, in walk order, indexed by member atom
        self.rings = [list(r) for r in dep["rings"]]
        self.ring_of: dict[int, list[list[int]]] = {}
        for r in self.rings:
            for a in r:
                self.ring_of.setdefault(a, []).append(r)

        # fused-ring SYSTEMS (share an atom) -> placed together
        self.ring_systems = self._fuse_ring_systems()
        self.system_of: dict[int, frozenset] = {}
        for sysset in self.ring_systems:
            for a in sysset:
                self.system_of[a] = frozenset(sysset)

    # -- ring fusion ------------------------------------------------------- #
    def _fuse_ring_systems(self) -> list[set]:
        systems: list[set] = []
        for r in self.rings:
            hit = [s for s in systems if any(a in s for a in r)]
            if not hit:
                systems.append(set(r))
            else:
                base = hit[0]
                for other in hit[1:]:
                    base |= other
                    systems.remove(other)
                base.update(r)
        return systems

    # -- chelate backbone path -------------------------------------------- #
    def _chelate_path(self) -> list[int]:
        """Shortest donor->donor path NOT through the metal (P..backbone..P)."""
        if len(self.donors) < 2:
            return list(self.donors)
        s, t = sorted(self.donors)[:2]
        prev = {s: None}
        q = deque([s])
        while q:
            c = q.popleft()
            if c == t:
                break
            for nb in self.adj[c]:
                if nb == self.metal or nb in prev:
                    continue
                prev[nb] = c
                q.append(nb)
        if t not in prev:
            return [s, t]
        path = []
        c = t
        while c is not None:
            path.append(c)
            c = prev.get(c)
        return path[::-1]

    # -- seed the metallacycle -------------------------------------------- #
    def _seed(self):
        """
        Place the WHOLE chelate metallacycle explicitly: Cu at the origin, the
        Cu-H bond horizontal with H to the right, the two P donors symmetric
        up-left / down-left, and the backbone bridge atoms strung along a
        circular arc that bulges to the LEFT (away from H), so the metallacycle
        reads as one tidy ring with both phosphines pointing in to Cu.
        """
        Cu = self.metal
        self.pos[Cu] = (0.0, 0.0)
        self.placed.add(Cu)

        if self.hydride is not None:
            self.pos[self.hydride] = (BL, 0.0)
            self.placed.add(self.hydride)

        # the two phosphines open to the LEFT, symmetric about the x-axis, with a
        # WIDE P-Cu-P bite (~130 deg) so the two ligand halves start well apart.
        ds = sorted(self.donors)
        half = math.radians(65.0)
        pdir = {ds[0]: (-math.cos(half), math.sin(half)),
                ds[1]: (-math.cos(half), -math.sin(half))}
        # the dative P-Cu bond is legitimately a bit longer than a C-C bond
        PCU = 1.35 * BL
        for d in ds:
            ux, uy = pdir[d]
            self.pos[d] = (ux * PCU, uy * PCU)
            self.placed.add(d)

        # Pre-place the chelate BACKBONE deterministically along the chelate path
        # so the two ligand halves do not collide.  ACYCLIC bridge atoms (O, CH2,
        # C(CH3)2 ...) are strung on a shallow arc bulging LEFT; the FIRST
        # ring-system atom met on each side from a P is laid down as a clean
        # regular-polygon block opening LEFT, and the rest of that fused system
        # tiles off it.  Ring atoms are NEVER linearly interpolated, so backbone
        # aryls stay perfect polygons.
        path = self._chelate_path()
        if len(path) >= 3 and path[0] in self.donors and path[-1] in self.donors:
            P0, P1 = self.pos[path[0]], self.pos[path[-1]]
            interior = path[1:-1]
            m = len(interior)
            mx, my = (P0[0] + P1[0]) / 2, (P0[1] + P1[1]) / 2
            bulge = (1.2 + 0.45 * m) * BL
            apex = (mx - bulge, my)
            for k, aid in enumerate(interior, start=1):
                if aid in self.system_of:
                    continue                      # ring atoms placed by tiling
                t = k / (m + 1)
                if t <= 0.5:
                    s = t / 0.5
                    x = P0[0] + (apex[0] - P0[0]) * s
                    y = P0[1] + (apex[1] - P0[1]) * s
                else:
                    s = (t - 0.5) / 0.5
                    x = apex[0] + (P1[0] - apex[0]) * s
                    y = apex[1] + (P1[1] - apex[1]) * s
                self.pos[aid] = (x, y)
                self.placed.add(aid)
            # now seat each backbone ring system met along the path as a clean
            # polygon block to the LEFT, then tile its fused partners.
            for idx, aid in enumerate(interior):
                if aid not in self.system_of or aid in self.placed:
                    continue
                # parent = previous placed path atom
                parent = None
                for j in range(idx, -1, -1):
                    cand = path[1:-1][j] if 0 <= j < m else None
                    if cand is None:
                        cand = path[0]
                    if cand in self.placed:
                        parent = cand
                        break
                if parent is None:
                    parent = path[0]
                px, py = self.pos[parent]
                # entry direction: LEFT and biased to the parent's own side
                # (upper P -> up-left, lower P -> down-left) so the two backbone
                # ring systems fan apart instead of stacking on the apex.
                side = 1.0 if py >= 0 else -1.0
                edx, edy = _unit(-1.0, 0.55 * side)   # mostly LEFT, slight fan
                self.pos[aid] = (px + edx * BL, py + edy * BL)
                self.placed.add(aid)
                self._place_ring_system(self.system_of[aid], aid, parent)

    # -- ring placement ---------------------------------------------------- #
    def _place_ring_system(self, sysset: frozenset, entry_atom: int,
                           parent: int):
        """
        Lay a fused ring system down as regular polygons.  `entry_atom` is the
        ring atom we arrived at from `parent` (already positioned); orient the
        ring so its centre is on the far side of entry_atom from the parent.
        """
        # find one ring of the system that contains entry_atom
        host = next((r for r in self.rings
                     if entry_atom in r and set(r) <= set(sysset)
                     or (entry_atom in r and r and set(r) & sysset)), None)
        if host is None:
            host = next(r for r in self.rings if entry_atom in r)

        ex, ey = self.pos[entry_atom]
        if parent in self.pos:
            px, py = self.pos[parent]
            inx, iny = _unit(ex - px, ey - py)   # bond direction = outward
        else:
            inx, iny = (-1.0, 0.0)

        n = len(host)
        # interior angle / 2: centre lies at distance = apothem along the
        # OUTWARD bisector from the entry vertex... simpler: place centre so the
        # entry vertex sits on the circle and the ring opens away from parent.
        R = BL / (2.0 * math.sin(math.pi / n))
        # direction from entry vertex to centre = outward direction (away from
        # parent) -- this makes the ring grow into open space.
        cx, cy = ex + inx * R, ey + iny * R

        # which winding keeps the ring inside the canvas / consistent? choose the
        # winding that puts the next walk atom on the upper side for upper P arms.
        verts = _regular_polygon((cx, cy), (ex, ey), n, ccw=True)
        # rotate ring index so vertex 0 == entry_atom
        k0 = host.index(entry_atom)
        ordered = host[k0:] + host[:k0]
        newly = []
        for idx, a in enumerate(ordered):
            if a not in self.placed:
                self.pos[a] = verts[idx]
                self.placed.add(a)
                newly.append(a)

        # place any OTHER rings of the system that share an edge with placed
        # atoms, repeatedly, so fused systems (benzodioxole, xanthene) tile.
        changed = True
        while changed:
            changed = False
            for r in self.rings:
                if not (set(r) & sysset):
                    continue
                if all(a in self.placed for a in r):
                    continue
                placed_in = [a for a in r if a in self.placed]
                if len(placed_in) < 2:
                    continue
                # find two adjacent placed atoms in this ring -> a shared edge
                seed_edge = None
                m = len(r)
                for i in range(m):
                    a, b = r[i], r[(i + 1) % m]
                    if a in self.placed and b in self.placed:
                        seed_edge = (i, a, b)
                        break
                if seed_edge is None:
                    continue
                i, a, b = seed_edge
                pa, pb = self.pos[a], self.pos[b]
                # ring centre is on the side AWAY from the rest of the molecule;
                # approximate "away" as away from the system centroid of placed
                sysc = self._placed_centroid(sysset)
                mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
                ox, oy = _unit(mx - sysc[0], my - sysc[1])
                apo = R * math.cos(math.pi / m) if m == len(r) else \
                    (BL / (2 * math.sin(math.pi / m))) * math.cos(math.pi / m)
                Rr = BL / (2.0 * math.sin(math.pi / m))
                ccx, ccy = mx + ox * apo, my + oy * apo
                verts2 = _regular_polygon((ccx, ccy), pa, m, ccw=True)
                # make sure verts2[1] lands on b; if not, flip winding
                if math.hypot(verts2[1][0] - pb[0], verts2[1][1] - pb[1]) > 0.4 * BL:
                    verts2 = _regular_polygon((ccx, ccy), pa, m, ccw=False)
                # re-seat centre so apothem is on the correct outward side again
                for off, atom in enumerate(r[i:] + r[:i]):
                    if atom not in self.placed:
                        self.pos[atom] = verts2[off]
                        self.placed.add(atom)
                        newly.append(atom)
                        changed = True
        return newly

    def _placed_centroid(self, ids) -> tuple[float, float]:
        xs = [self.pos[a][0] for a in ids if a in self.placed]
        ys = [self.pos[a][1] for a in ids if a in self.placed]
        if not xs:
            return (0.0, 0.0)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    # -- fan-out directions at a branch ------------------------------------ #
    @staticmethod
    def _fan(in_ang: float, k: int) -> list[float]:
        """
        `k` outgoing bond angles around an atom whose incoming bond points along
        `in_ang`.  The outgoing bonds continue roughly OPPOSITE the incoming
        direction (so the chain flows outward) and fan symmetrically with ~120
        deg between the incoming bond and the outer arms:
          k=1 -> straight on, single ~120-deg zig (here: 180 deg away = anti)
          k=2 -> +/-120 deg off the incoming (classic sp2/sp3 trigonal)
          k=3 -> the third arm fills the remaining gap
        """
        # outgoing bonds CONTINUE outward (centred on in_ang), fanned to the
        # sides -- never back toward where we came from (that would aim the
        # substituents at the metal and pile them up in the centre).
        if k == 1:
            return [in_ang + math.radians(60.0)]   # a single 120-deg zig
        if k == 2:
            return [in_ang - math.radians(60.0), in_ang + math.radians(60.0)]
        if k == 3:
            return [in_ang, in_ang - math.radians(100.0),
                    in_ang + math.radians(100.0)]
        span = math.radians(200.0)
        return [in_ang - span / 2 + span * i / (k - 1) for i in range(k)]

    # -- grow ring systems off the pre-placed metallacycle ----------------- #
    def _place_backbone_rings(self):
        """
        Any backbone (chelate-path) atom that belongs to a ring system: tile that
        whole ring system from the >=2 already-placed backbone atoms it shares,
        so the backbone aryls (xanthene, the two SEGPhos benzodioxoles, the
        DPEphos benzenes) are perfect polygons fused onto the metallacycle.
        """
        seeded = set()
        for aid in list(self.placed):
            if aid in self.system_of and self.system_of[aid] not in seeded:
                sysset = self.system_of[aid]
                # need >=2 placed atoms of a ring of this system to orient it
                if sum(1 for x in sysset if x in self.placed) >= 2:
                    self._tile_system_from_placed(sysset)
                    seeded.add(sysset)

    def _tile_system_from_placed(self, sysset):
        """Place every ring of `sysset` from already-placed shared edges,
        opening AWAY from the metal."""
        Cu = self.pos[self.metal]
        changed = True
        guard = 0
        while changed and guard < 50:
            guard += 1
            changed = False
            for r in self.rings:
                if not (set(r) & sysset) or all(a in self.placed for a in r):
                    continue
                m = len(r)
                edge = None
                for i in range(m):
                    a, b = r[i], r[(i + 1) % m]
                    if a in self.placed and b in self.placed:
                        edge = (i, a, b)
                        break
                if edge is None:
                    continue
                i, a, b = edge
                pa, pb = self.pos[a], self.pos[b]
                mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
                # open away from the metal
                ox, oy = _unit(mx - Cu[0], my - Cu[1])
                Rr = BL / (2.0 * math.sin(math.pi / m))
                apo = Rr * math.cos(math.pi / m)
                ccx, ccy = mx + ox * apo, my + oy * apo
                verts = _regular_polygon((ccx, ccy), pa, m, ccw=True)
                if math.hypot(verts[1][0] - pb[0], verts[1][1] - pb[1]) > 0.4 * BL:
                    verts = _regular_polygon((ccx, ccy), pa, m, ccw=False)
                for off, atom in enumerate(r[i:] + r[:i]):
                    if atom not in self.placed:
                        self.pos[atom] = verts[off]
                        self.placed.add(atom)
                        changed = True

    # -- main BFS ---------------------------------------------------------- #
    def layout(self) -> None:
        self._seed()
        # frontier item: (atom, in_dir_angle) -- in_dir is the angle of the bond
        # that arrived at `atom`.  Grow outward from the two phosphines; each P's
        # three substituents (backbone path + 2 aryls/alkyls) fan into distinct
        # sectors so the two ligand halves never pile up on each other.
        q: deque = deque()
        Cu = self.pos[self.metal]
        bbc = self._placed_centroid([a for a in self.placed
                                     if a not in (self.metal, self.hydride)])
        for aid in list(self.placed):
            if aid in (self.metal, self.hydride):
                continue
            if aid in self.donors:
                dx, dy = self.pos[aid][0] - Cu[0], self.pos[aid][1] - Cu[1]
                q.append((aid, math.atan2(dy, dx)))
            else:
                ax, ay = self.pos[aid]
                q.append((aid, math.atan2(ay - bbc[1], ax - bbc[0])))

        guard = 0
        while q and guard < 100000:
            guard += 1
            atom, in_ang = q.popleft()
            kids = [nb for nb in self.adj[atom]
                    if nb != self.metal and nb not in self.placed
                    and not (nb in self.system_of and atom in self.system_of
                             and self.system_of[nb] == self.system_of[atom])]
            if not kids:
                # still walk into a ring system this atom is part of, if unplaced
                continue
            # deterministic order: ring-bearing kids first, then by id, so the
            # bulky rings claim the outer sectors.
            kids.sort(key=lambda nb: (nb not in self.system_of, nb))
            if atom in self.system_of:
                # `atom` is a ring vertex -> its exocyclic substituents point
                # RADIALLY OUTWARD from the ring centre (exterior bisector). Fan
                # multiple substituents tightly about that outward radial.
                sysc = self._placed_centroid(self.system_of[atom])
                ax, ay = self.pos[atom]
                rad = math.atan2(ay - sysc[1], ax - sysc[0])
                k = len(kids)
                spread = math.radians(50.0)
                angs = [rad] if k == 1 else \
                    [rad - spread / 2 + spread * i / (k - 1) for i in range(k)]
            else:
                angs = self._fan(in_ang, len(kids))
            for nb, a in zip(kids, angs):
                if nb in self.placed:
                    continue
                if nb in self.system_of:
                    newly = self._enter_ring(nb, atom, a)
                    # enqueue EVERY freshly placed ring atom with a radial-outward
                    # in_dir so its exocyclic substituents (t-Bu / OMe / aryl /
                    # the ipso->P bond) fan outward off the ring.
                    sysc = self._placed_centroid(self.system_of[nb])
                    for rid in newly:
                        rx, ry = self.pos[rid]
                        rad = math.atan2(ry - sysc[1], rx - sysc[0])
                        q.append((rid, rad))
                else:
                    px, py = self.pos[atom]
                    self.pos[nb] = (px + math.cos(a) * BL, py + math.sin(a) * BL)
                    self.placed.add(nb)
                    q.append((nb, a))

        # any stragglers -> drop near a placed neighbour
        for a in self.atoms:
            if a not in self.placed:
                nbs = [nb for nb in self.adj[a] if nb in self.placed]
                if nbs:
                    bx, by = self.pos[nbs[0]]
                    dx, dy = self._radial_dir_from(a, nbs[0])
                    self.pos[a] = (bx + dx * BL, by + dy * BL)
                else:
                    self.pos[a] = (0.0, 0.0)
                self.placed.add(a)

    def _radial_dir_from(self, atom, base):
        bx, by = self.pos[base]
        sx = sy = 0.0
        for nb in self.adj[base]:
            if nb in self.placed and nb != atom:
                ux, uy = _unit(self.pos[nb][0] - bx, self.pos[nb][1] - by)
                sx += ux
                sy += uy
        if sx == 0 and sy == 0:
            return (-1.0, 0.0)
        return _unit(-sx, -sy)

    def _enter_ring(self, entry_atom: int, parent: int, in_ang: float):
        """Place the entry vertex along `in_ang` from parent, then tile its ring
        system as regular polygons opening away from the parent.  Returns the
        list of ring-system atoms newly placed by this call."""
        sysset = self.system_of[entry_atom]
        newly = []
        if entry_atom not in self.placed:
            px, py = self.pos[parent]
            self.pos[entry_atom] = (px + math.cos(in_ang) * BL,
                                    py + math.sin(in_ang) * BL)
            self.placed.add(entry_atom)
            newly.append(entry_atom)
        if all(a in self.placed for a in sysset):
            return newly
        newly += self._place_ring_system(sysset, entry_atom, parent)
        return newly

    # -- write back -------------------------------------------------------- #
    def apply(self) -> dict:
        self.layout()
        # normalise so the metal is at the origin and rotate Cu-H to +x exactly
        Cu = self.pos[self.metal]
        out = {a: (x - Cu[0], y - Cu[1]) for a, (x, y) in self.pos.items()}
        if self.hydride is not None and self.hydride in out:
            hx, hy = out[self.hydride]
            ang = -math.atan2(hy, hx)
            out = {a: _rot(x, y, ang) for a, (x, y) in out.items()}
        for a in self.dep["atoms"]:
            x, y = out[a["id"]]
            a["x"] = round(float(x), 4)
            a["y"] = round(float(y), 4)
        self.dep["meta"]["layout_method"] = "method3-template"
        return self.dep


def relayout(dep: dict) -> dict:
    """Replace a depiction's coordinates with the deterministic template."""
    return TemplateLayout(dep).apply()
