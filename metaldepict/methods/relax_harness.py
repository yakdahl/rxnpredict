#!/usr/bin/env python3
"""
relax_harness.py -- shared scaffold for experimenting with 2-D physics relaxers
that clean up a Method-C TEMPLATE SEED of the Cu-H bisphosphine depictions.

WHY: the deterministic template (method3_template.py) places Cu / P / rings at
fixed absolute coordinates, so the few INTER-FRAGMENT bonds come out the wrong
length (e.g. Xantphos P->aryl = 0.68 L, P->Cu = 2.53 L).  A 2-D physics pass can
pull every bond back to a standard length and fan overlapping groups apart while
keeping the (already perfect) rings rigid.

This module exposes the molecule to a relaxer as BOTH:
  * a per-atom graph   -- H.pos, H.bonds (with target lengths), H.angles
                          (junction triples with ideal angles), H.rigid_pairs
                          (intra-ring distances to keep rings rigid for per-atom
                          solvers), H.overlaps (non-bonded pairs + min sep).
  * a rigid-body view  -- H.bodies (ring SYSTEMS frozen as rigid bodies; every
                          other atom is its own 1-atom body), H.joints
                          (inter-body bonds), H.body_of.

A RELAXER is any callable                 relax(H, w) -> None
that mutates H.pos in place.  `w` is a dict of weights; targets are expressed as
multiples of the standard bond length L so weights are scale-free.  H.pinned
atoms (Cu and its hydride) must never move -- that keeps the Cu-H bond exactly
horizontal with H to the right.

Helpers: H.metrics() (bondCV / angleDevDeg / overlap / ringEdgeCV), H.save_svg(),
and make_panel() which renders all four ligands through a relaxer to a montage.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import method3_template as M           # noqa: E402
from template_draw import L            # noqa: E402

LIGANDS = ["dtbm_segphos", "xantphos", "dpephos", "ph_bpe"]

# ---- default target lengths (multiples of L) --------------------------------
TGT_PCU = 1.32      # P -> Cu coordination (drawn slightly longer, conventional)
TGT_CUH = 1.10      # Cu -> H
TGT_BOND = 1.00     # every ordinary skeletal / substituent bond
IDEAL_RING = {3: 60.0, 4: 90.0, 5: 108.0, 6: 120.0, 7: 128.57}


def _vlen(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def _seg_cross(p1, p2, p3, p4):
    """True iff segment p1-p2 properly crosses segment p3-p4."""
    return (_ccw(p1, p3, p4) != _ccw(p2, p3, p4)
            and _ccw(p1, p2, p3) != _ccw(p1, p2, p4))


def _ang(p, q, r):
    """angle (deg) at q for the triple p-q-r."""
    a = (p[0] - q[0], p[1] - q[1])
    b = (r[0] - q[0], r[1] - q[1])
    na = math.hypot(*a) or 1e-9
    nb = math.hypot(*b) or 1e-9
    c = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))
    return math.degrees(math.acos(c))


class Harness:
    """Loads one ligand's Method-C seed and derives everything a relaxer needs."""

    def __init__(self, key=None, scene=None, title=None,
                 exempt=None, extra_rigid=None):
        # Two seed sources: a named hardcoded template (key -> M.BUILDERS[key]())
        # or any pre-built Scene (e.g. generated from a SMILES). The rest of the
        # harness -- ring/body detection, joints, angles, overlaps, metrics --
        # is derived purely from the Scene, so the physics is fully dynamic.
        # exempt:      atom-id set whose MUTUAL overlaps are ignored (e.g. the
        #              ferrocene sandwich, which is intentionally close-packed).
        # extra_rigid: list of atom-id sets to FREEZE into one rigid body each
        #              (e.g. a metallocene, whose Cp rings + Fe must move as one).
        if scene is not None:
            sc = scene
            title = title or "molecule"
        else:
            sc, title = M.BUILDERS[key]()
        self.key = key or "custom"
        self.scene = sc
        self.title = title
        self._exempt = set(exempt) if exempt else set()
        self._extra_rigid = [set(g) for g in (extra_rigid or [])]
        self.pos = {i: (float(a.pos[0]), float(a.pos[1]))
                    for i, a in sc.atoms.items()}
        self.label = {i: a.label for i, a in sc.atoms.items()}
        self.ids = list(sc.atoms)

        # ---- adjacency + bond kinds ----
        self.adj = {i: [] for i in self.ids}
        self.kind = {}
        self.order = {}
        self._raw_bonds = []
        for b in sc.bonds:
            self.adj[b.a].append(b.b)
            self.adj[b.b].append(b.a)
            self.kind[frozenset((b.a, b.b))] = b.kind
            self.order[frozenset((b.a, b.b))] = b.order
            self._raw_bonds.append((b.a, b.b))

        # ---- rings + fused ring systems -> rigid bodies ----
        self.rings = self._find_rings()
        # bulky saturated carbocycles (e.g. cyclohexyl): all-single-bond, all
        # unlabelled-carbon rings -> they get extra clearance so the layout
        # pushes them further from everything else.
        self.bulky = set()
        for r in self.rings:
            edges = [frozenset((a, b)) for a in r for b in self.adj[a]
                     if b in r and b > a]
            if (edges and all(self.order.get(e, 1) == 1 for e in edges)
                    and all(not self.label[a] for a in r) and len(r) >= 5):
                self.bulky |= set(r)
        self.bodies = self._build_bodies()
        self.body_of = {}
        for bi, bd in enumerate(self.bodies):
            for a in bd:
                self.body_of[a] = bi

        # ---- metal + hydride (pins) ----
        self.metal = next((i for i, l in self.label.items() if l == "Cu"), None)
        self.hyd = None
        if self.metal is not None:
            for nb in self.adj[self.metal]:
                if self.label[nb] == "H":
                    self.hyd = nb
        self.donors = [i for i, l in self.label.items() if l == "P"]
        self.pinned = set(x for x in (self.metal, self.hyd) if x is not None)

        # ---- bonds with target lengths ----
        self.bonds = [(a, b, self._target_len(a, b)) for (a, b) in self._raw_bonds]
        self.joints = [(a, b, t) for (a, b, t) in self.bonds
                       if self.body_of[a] != self.body_of[b]]

        # ---- intra-body rigid distances (edges + diagonals) ----
        self.rigid_pairs = self._rigid_pairs()

        # ---- junction angle targets ----
        self.angles = self._angle_targets()

        # ---- non-bonded overlap pairs + min separation ----
        self.overlaps = self._overlap_pairs()

    # -------------------------------------------------------------- geometry --
    def _find_rings(self):
        """All simple carbo/heterocyclic rings up to size 7 via shortest-path-
        after-edge-removal.  The metal is NEVER traversed, so the P-Cu-P chelate
        metallacycle is excluded -- its coordination bonds must stay flexible and
        must not fuse Cu+P into a rigid body."""
        rings = set()
        adj = self.adj
        metal_ids = {i for i, l in self.label.items() if l == "Cu"}
        for (u, v) in self._raw_bonds:
            if u in metal_ids or v in metal_ids:
                continue
            # shortest path u->v without the direct edge, never through metal
            from collections import deque
            prev = {u: None}
            q = deque([u])
            while q:
                x = q.popleft()
                if x == v:
                    break
                for y in adj[x]:
                    if y in metal_ids:
                        continue
                    if y == v and x == u:
                        continue            # skip the direct edge
                    if y not in prev:
                        prev[y] = x
                        q.append(y)
            if v not in prev:
                continue
            path = []
            x = v
            ok = True
            while x is not None:
                path.append(x)
                x = prev[x]
                if len(path) > 8:
                    ok = False
                    break
            if ok and 3 <= len(path) <= 7:
                rings.add(frozenset(path))
        return [set(r) for r in rings]

    def _build_bodies(self):
        """Fuse rings sharing an atom into rigid bodies; lone atoms are 1-bodies.
        Forced `extra_rigid` groups (e.g. a ferrocene sandwich) join the fusion so
        their rings + bridging metal move as one rigid unit."""
        groups = [set(r) for r in self.rings] + [set(g) for g in self._extra_rigid]
        merged = True
        while merged:
            merged = False
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if groups[i] & groups[j]:
                        groups[i] |= groups[j]
                        groups.pop(j)
                        merged = True
                        break
                if merged:
                    break
        in_ring = set().union(*groups) if groups else set()
        bodies = [frozenset(g) for g in groups]
        for i in self.ids:
            if i not in in_ring:
                bodies.append(frozenset((i,)))
        return bodies

    def _target_len(self, a, b):
        la, lb = self.label[a], self.label[b]
        k = self.kind[frozenset((a, b))]
        if {la, lb} == {"P", "Cu"} or k in ("dative", "coord"):
            return TGT_PCU * L
        if {la, lb} == {"Cu", "H"}:
            return TGT_CUH * L
        return TGT_BOND * L

    def _rigid_pairs(self):
        """For per-atom solvers: every intra-body atom pair with its seed
        distance, so rings stay rigid and regular under spring relaxation."""
        out = []
        for bd in self.bodies:
            if len(bd) < 2:
                continue
            members = list(bd)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, c = members[i], members[j]
                    out.append((a, c, _vlen(self.pos[a], self.pos[c])))
        return out

    def _angle_targets(self):
        """Ideal angles at JUNCTION atoms (atoms carrying >=1 inter-body joint).
        Ring-internal angles are handled by rigid bodies / rigid_pairs, so we
        only constrain the flexible junctions (P, Cu, ipso carbons, bridges)."""
        joint_atoms = set()
        for (a, b, _) in self.joints:
            joint_atoms.add(a)
            joint_atoms.add(b)
        out = []
        for j in joint_atoms:
            nbrs = sorted(self.adj[j],
                          key=lambda n: math.atan2(self.pos[n][1] - self.pos[j][1],
                                                   self.pos[n][0] - self.pos[j][0]))
            n = len(nbrs)
            if n < 2:
                continue
            if self.label[j] == "Cu":
                ideal = 120.0                      # trigonal metal
            elif self.label[j] == "P":
                ideal = 112.0                      # ~tetrahedral-ish P
            else:
                ideal = 360.0 / n if n >= 3 else 120.0
            # consecutive neighbour pairs only (the drawn wedge between them)
            for t in range(n):
                i_, k_ = nbrs[t], nbrs[(t + 1) % n]
                if n == 2 and t == 1:
                    break
                out.append((i_, j, k_, ideal))
        return out

    def _overlap_pairs(self):
        """Non-bonded atom pairs in DIFFERENT bodies, with a label-aware minimum
        separation (wide text glyphs need more clearance)."""
        bonded = set(frozenset((a, b)) for (a, b) in self._raw_bonds)
        out = []
        for ii in range(len(self.ids)):
            for jj in range(ii + 1, len(self.ids)):
                a, b = self.ids[ii], self.ids[jj]
                if frozenset((a, b)) in bonded:
                    continue
                if self.body_of[a] == self.body_of[b]:
                    continue
                if a in self._exempt and b in self._exempt:
                    continue                    # intra-fragment (e.g. ferrocene)
                la, lb = self.label[a], self.label[b]
                if la and lb:
                    mn = 0.62 * L + 0.10 * L * (len(la) + len(lb))
                elif la or lb:
                    mn = 0.78 * L
                else:
                    mn = 0.55 * L
                if a in self.bulky or b in self.bulky:    # cyclohexyl etc. -- more room
                    mn *= 1.45
                out.append((a, b, mn))
        return out

    # --------------------------------------------------------------- metrics --
    def metrics(self):
        # skeletal single-ish bond lengths (exclude P-Cu coordination + Cu-H)
        sk = []
        for (a, b, t) in self.bonds:
            if {self.label[a], self.label[b]} & {"Cu"}:
                continue
            sk.append(_vlen(self.pos[a], self.pos[b]))
        bondCV = _cv(sk)
        # ring edge CV
        redge = []
        for r in self.rings:
            rl = list(r)
            for x in rl:
                for y in self.adj[x]:
                    if y in r and y > x:
                        redge.append(_vlen(self.pos[x], self.pos[y]))
        ringCV = _cv(redge)
        # junction angle deviation from ideal
        devs = [abs(_ang(self.pos[i], self.pos[j], self.pos[k]) - tgt)
                for (i, j, k, tgt) in self.angles]
        angleDev = sum(devs) / len(devs) if devs else 0.0
        # overlaps
        ov = sum(1 for (a, b, mn) in self.overlaps
                 if _vlen(self.pos[a], self.pos[b]) < mn)
        # coordination-bond length error (how far P-Cu / Cu-H are from target)
        coorderr = []
        for (a, b, t) in self.bonds:
            if {self.label[a], self.label[b]} & {"Cu"}:
                coorderr.append(abs(_vlen(self.pos[a], self.pos[b]) - t) / L)
        return {
            "bondCV": round(bondCV, 4),
            "ringEdgeCV": round(ringCV, 4),
            "ringAngleDev": round(self._ring_angle_dev(), 2),
            "angleDevDeg": round(angleDev, 2),
            "overlap": ov,
            "crossings": self.count_crossings(),
            "crowd": round(self._crowd(), 3),
            "coordLenErr": round(sum(coorderr) / len(coorderr), 3) if coorderr else 0,
            "symDev": round(self._sym_dev(), 3),
        }

    def count_crossings(self):
        """Number of pairs of drawn bonds that visually CROSS (share no atom and
        properly intersect).  Two bonds crossing is a worse depiction artefact
        than a near-contact, and the atom-overlap count does not detect it (the
        endpoints can be far apart).  Bonds inside one exempt fragment (the
        ferrocene sandwich) are ignored against each other."""
        bonds = self._raw_bonds
        n = 0
        for i in range(len(bonds)):
            a, b = bonds[i]
            pa, pb = self.pos[a], self.pos[b]
            for j in range(i + 1, len(bonds)):
                c, d = bonds[j]
                if a == c or a == d or b == c or b == d:
                    continue
                if {a, b, c, d} <= self._exempt:
                    continue
                if _seg_cross(pa, pb, self.pos[c], self.pos[d]):
                    n += 1
        return n

    def _crowd(self, factor=1.5):
        """Soft near-contact / anti-collapse penalty: sum over cross-body
        non-bonded pairs of the squared penetration into a clearance band of
        factor*min_sep.  Unlike the hard `overlap` count this rewards genuine
        WHITESPACE, so a method cannot game the judge by compressing groups
        toward the metal until they merely-touch without strictly overlapping."""
        tot = 0.0
        for (a, b, mn) in self.overlaps:
            d = _vlen(self.pos[a], self.pos[b])
            clear = mn * factor
            if d < clear:
                tot += ((clear - d) / clear) ** 2
        return tot

    def _ring_cycle(self, r):
        """Order a ring's atom set into a cyclic walk via ring-internal adjacency."""
        rs = set(r)
        start = next(iter(rs))
        cyc = [start]
        prev = None
        cur = start
        while True:
            nxt = None
            for nb in self.adj[cur]:
                if nb in rs and nb != prev:
                    if nb == start and len(cyc) > 2:
                        return cyc
                    if nb not in cyc:
                        nxt = nb
                        break
            if nxt is None:
                return cyc
            cyc.append(nxt)
            prev, cur = cur, nxt
            if len(cyc) > len(rs):
                return cyc

    def _ring_angle_dev(self):
        """Mean |interior-angle - regular-polygon-ideal| over all ring vertices.
        Penalises squished / irregular rings (the references are regular n-gons)."""
        devs = []
        for r in self.rings:
            cyc = self._ring_cycle(r)
            n = len(cyc)
            if n < 3:
                continue
            ideal = 180.0 * (n - 2) / n
            for t in range(n):
                p = self.pos[cyc[t - 1]]
                q = self.pos[cyc[t]]
                s = self.pos[cyc[(t + 1) % n]]
                devs.append(abs(_ang(p, q, s) - ideal))
        return sum(devs) / len(devs) if devs else 0.0

    def _sym_dev(self):
        """C2 deviation: reflect every atom across the horizontal Cu-H axis and
        measure the mean residual to the nearest same-label atom (the references
        are C2-symmetric about that axis).  Normalised by L."""
        if self.metal is None:
            return 0.0
        cy = self.pos[self.metal][1]
        tot = 0.0
        n = 0
        for i in self.ids:
            x, y = self.pos[i]
            rx, ry = x, 2 * cy - y
            best = 1e9
            for j in self.ids:
                if self.label[j] != self.label[i]:
                    continue
                d = math.hypot(self.pos[j][0] - rx, self.pos[j][1] - ry)
                if d < best:
                    best = d
            tot += best
            n += 1
        return (tot / n) / L if n else 0.0

    # ----------------------------------------------------------------- output --
    def commit(self):
        # recompute every ring's CURRENT centroid so Kekule double bonds are
        # re-pointed to the inside of their ring -- the 'inside' coordinate baked
        # in at template time goes stale once rigid bodies translate/rotate, which
        # would flip doubles to the outside.
        ring_cent = []
        for r in self.rings:
            xs = [self.pos[i][0] for i in r]
            ys = [self.pos[i][1] for i in r]
            ring_cent.append((r, (sum(xs) / len(xs), sum(ys) / len(ys))))
        for i, p in self.pos.items():
            self.scene.atoms[i].pos = p
        for bd in self.scene.bonds:
            if bd.order == 2 and bd.inside is not None:
                for (r, c) in ring_cent:
                    if bd.a in r and bd.b in r:
                        bd.inside = c
                        break

    def to_svg(self, width=760, height=640):
        self.commit()
        return self.scene.render_svg(width=width, height=height, title=self.title)

    def save_svg(self, path, width=760, height=640):
        Path(path).write_text(self.to_svg(width, height))


def _cv(xs):
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    if m == 0:
        return 0.0
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var) / m


# ============================================================================ #
#  NUMERICAL JUDGE  +  BLACK-BOX OPTIMISER
#  quality_loss() turns a relaxed depiction into ONE scalar (lower = better,
#  reference-like). optimize_method() runs scipy differential-evolution over a
#  method's OWN parameters to minimise the mean loss across all four ligands.
# ============================================================================ #
# Coefficients that DEFINE "good" (reference-like) -- uniform bonds, regular
# rings, correct coordination lengths, no overlap, C2-symmetric. Junction angles
# are intentionally NOT scored (the bite angle etc. are set by the layout, not a
# single ideal). These are the judge's weights, fixed; the METHODS tune their own.
DEFAULT_JUDGE = {
    "bond": 6.5,        # bondCV  -- uniform skeletal bond lengths (down a bit)
    "ring": 7.0,        # ringEdgeCV
    "ringang": 0.05,    # ringAngleDev (deg) -- regular polygons
    "jang": 0.08,       # angleDevDeg -- junction angles off is bothersome (up)
    "coord": 1.6,       # coordLenErr -- P-Cu / Cu-H at target length
    "overlap": 1.0,     # per strictly-overlapping non-bonded pair (count)
    "crowd": 1.0,       # soft near-contact / anti-collapse (rewards whitespace)
    "sym": 1.2,         # symDev -- C2 symmetry about the Cu-H axis
}


def fan_substituents(H, margin_deg=22.0):
    """MERGE step: RDKit gives the connectivity + ring shapes, but clusters the
    two P-substituents; this re-fans them.  For each P donor, the substituent
    groups (the small components hanging off P, i.e. not the backbone, not Cu)
    are rotated about P to spread evenly through the largest free angular gap
    between the backbone bond and the P->Cu bond.  Whole rigid bodies rotate, so
    ring shapes are preserved; the energy relax then only has to fix lengths."""
    bodies, pin, inv, translate, rotate = body_helpers(H)
    for p in H.donors:
        nbrs = [n for n in H.adj[p] if H.label[n] != "Cu"]
        if len(nbrs) < 2:
            continue
        # connected component reached from each neighbour with the P-n bond cut
        comp = {}
        for n in nbrs:
            seen = {p, n}
            stack = [n]
            while stack:
                x = stack.pop()
                for y in H.adj[x]:
                    if y == p or y in seen or H.label[y] == "Cu":
                        continue
                    seen.add(y)
                    stack.append(y)
            comp[n] = seen - {p}
        backbone_n = max(nbrs, key=lambda n: len(comp[n]))
        subs = [n for n in nbrs if n != backbone_n]
        if not subs:
            continue

        def ang(a, b):
            return math.atan2(H.pos[b][1] - H.pos[a][1],
                              H.pos[b][0] - H.pos[a][0])
        a_back = ang(p, backbone_n)
        a_cu = ang(p, H.metal) if H.metal is not None else a_back + math.pi
        # the two arcs between a_back and a_cu; fan into the larger one
        d = (a_cu - a_back) % (2 * math.pi)
        if d >= math.pi:
            lo, span = a_back, d
        else:
            lo, span = a_cu, 2 * math.pi - d
        m = math.radians(margin_deg)
        span = max(0.0, span - 2 * m)
        k = len(subs)
        targets = [lo + m + span * (j + 1) / (k + 1) for j in range(k)]
        # assign each sub to the nearest target (stable), then rotate its body
        subs_sorted = sorted(subs, key=lambda n: (ang(p, n) - lo) % (2 * math.pi))
        for n, tgt in zip(subs_sorted, targets):
            cur = ang(p, n)
            dth = (tgt - cur + math.pi) % (2 * math.pi) - math.pi
            c_, s_ = math.cos(dth), math.sin(dth)
            px, py = H.pos[p]
            for a in comp[n]:                    # rotate the WHOLE substituent
                if a in H.pinned:
                    continue
                x, y = H.pos[a][0] - px, H.pos[a][1] - py
                H.pos[a] = (px + c_ * x - s_ * y, py + s_ * x + c_ * y)


def declutter(H, angles=(10, -10, 20, -20, 32, -32, 45, -45), passes=2):
    """SLIGHT in-place rotation to relieve crowding: for each P substituent group
    that participates in an overlap, try small rotations of the WHOLE group about
    its P and keep the one that removes the most local overlaps (never adds one).
    Conservative -- only crowded groups move, ring shapes are preserved."""
    def local_cost(atoms):
        s = set(atoms)
        return sum(1 for (a, b, mn) in H.overlaps
                   if (a in s or b in s) and _vlen(H.pos[a], H.pos[b]) < mn)

    for _ in range(passes):
        for p in H.donors:
            nbrs = [n for n in H.adj[p] if H.label[n] != "Cu"]
            if len(nbrs) < 2:
                continue
            comp = {}
            for n in nbrs:
                seen = {p, n}
                stack = [n]
                while stack:
                    x = stack.pop()
                    for y in H.adj[x]:
                        if y == p or y in seen or H.label[y] == "Cu":
                            continue
                        seen.add(y)
                        stack.append(y)
                comp[n] = seen - {p}
            backbone_n = max(nbrs, key=lambda n: len(comp[n]))
            for n in nbrs:
                if n == backbone_n:
                    continue
                atoms = [a for a in comp[n] if a not in H.pinned]
                if not atoms or local_cost(atoms) == 0:
                    continue
                px, py = H.pos[p]
                orig = {a: H.pos[a] for a in atoms}
                best_cost = local_cost(atoms)
                best_deg = 0.0
                for deg in angles:
                    th = math.radians(deg)
                    cz, sz = math.cos(th), math.sin(th)
                    for a in atoms:
                        x, y = orig[a][0] - px, orig[a][1] - py
                        H.pos[a] = (px + cz * x - sz * y, py + sz * x + cz * y)
                    cst = local_cost(atoms)
                    for a in atoms:
                        H.pos[a] = orig[a]
                    if cst < best_cost:
                        best_cost, best_deg = cst, deg
                if best_deg:
                    th = math.radians(best_deg)
                    cz, sz = math.cos(th), math.sin(th)
                    for a in atoms:
                        x, y = orig[a][0] - px, orig[a][1] - py
                        H.pos[a] = (px + cz * x - sz * y, py + sz * x + cz * y)


def _p_components(H):
    """For every P donor: {neighbour -> set of atoms reachable from it with the
    P bond cut (Cu excluded)} plus which neighbour is the backbone (biggest)."""
    res = {}
    for p in H.donors:
        nbrs = [n for n in H.adj[p] if H.label[n] != "Cu"]
        if len(nbrs) < 2:
            continue
        comp = {}
        for n in nbrs:
            seen = {p, n}
            stack = [n]
            while stack:
                x = stack.pop()
                for y in H.adj[x]:
                    if y == p or y in seen or H.label[y] == "Cu":
                        continue
                    seen.add(y)
                    stack.append(y)
            comp[n] = seen - {p}
        backbone_n = max(nbrs, key=lambda n: len(comp[n]))
        res[p] = (comp, backbone_n)
    return res


def _aryl_substituent_rings(H):
    """Candidate aromatic substituent rings for tilting: for each P donor, the
    5/6-ring carrying the ipso atom of a NON-backbone arm.  Returns
    (p, ipso, ring_set, component_set) tuples."""
    out = []
    rings = [set(r) for r in H.rings]
    for p, (comp, backbone_n) in _p_components(H).items():
        for n, atoms in comp.items():
            if n == backbone_n:
                continue
            for r in rings:
                if n in r and 5 <= len(r) <= 6 and r <= atoms:
                    out.append((p, n, set(r), set(atoms)))
                    break
    return out


def _restyle_tilted_ring(H, ring, perp):
    """Redraw a foreshortened ring with thickening WEDGES on the edges that come
    TOWARD the viewer and NOTHING (plain bonds) on the edges going away, plus an
    aromatic circle.

    Conventions (fixed): the wedge (toward-viewer) side is always the LEFT side
    of the ring, so every tilted ring -- top or bottom -- reads the same way.
    Never two taper wedges in a row: the first edge of a run of near-side edges
    is a thickening taper, any further connected near edge is drawn BOLD (uniform
    full width) instead, so a wedge never feeds straight into another wedge."""
    prx, pry = perp
    cx = sum(H.pos[a][0] for a in ring) / len(ring)
    cy = sum(H.pos[a][1] for a in ring) / len(ring)
    depth = {a: (H.pos[a][0] - cx) * prx + (H.pos[a][1] - cy) * pry for a in ring}
    pos = [a for a in ring if depth[a] > 0]
    neg = [a for a in ring if depth[a] < 0]

    def meanx(s):
        return sum(H.pos[a][0] for a in s) / len(s) if s else 0.0
    if pos and neg and meanx(pos) > meanx(neg):  # near (wedge) side := LEFT (small x)
        depth = {a: -v for a, v in depth.items()}
    cyc = H._ring_cycle(ring)
    edges = [(cyc[i], cyc[(i + 1) % len(cyc)]) for i in range(len(cyc))]
    near = [(depth[e[0]] + depth[e[1]]) / 2 > 0 for e in edges]
    bonds = {frozenset((b.a, b.b)): b for b in H.scene.bonds}
    # rotate the edge list so a run of near-side edges does not wrap the seam
    start = next((i for i in range(len(edges)) if not near[i]), 0)
    order = [(start + k) % len(edges) for k in range(len(edges))]
    # within a run of near edges ALTERNATE taper, bold, taper, ... -- so a run
    # starts and ends with a taper and a bold is always flanked by tapers (never
    # a dangling bold whose far end touches no wedge).
    run_idx = 0
    for i in order:
        e = edges[i]
        bd = bonds.get(frozenset(e))
        if bd is None:
            continue
        bd.order = 1
        bd.inside = None
        if near[i]:
            if run_idx % 2 == 1:                 # interior of the run -> bold
                bd.kind = "bold"
                bd.width = None
            else:                                # run ends (and start) -> taper
                lo, hi = (e[0], e[1]) if depth[e[0]] <= depth[e[1]] else (e[1], e[0])
                bd.a, bd.b = lo, hi              # narrow at far vertex, wide toward viewer
                bd.kind = "taper"
                bd.width = 0.17 * L
            run_idx += 1
        else:
            bd.kind = "plain"
            bd.width = None
            run_idx = 0
    H.scene.ring_circle(list(ring), r_frac=0.58)


def _exo_subtrees(H, ring):
    """For each ring atom, the subtrees hanging off it (substituents/labels):
    {anchor_ring_atom -> [(neighbour, {subtree atom ids})...]}, never crossing
    back into the ring or into the P donor."""
    rs = set(ring)
    res = {}
    for c in ring:
        for d in H.adj[c]:
            if d in rs or H.label[d] == "P" or H.label[d] == "Cu":
                continue
            seen = {d}
            stack = [d]
            while stack:
                x = stack.pop()
                for y in H.adj[x]:
                    if y in rs or y in seen or H.label[y] in ("P", "Cu"):
                        continue
                    seen.add(y)
                    stack.append(y)
            res.setdefault(c, []).append((d, seen))
    return res


def _tilt_geometry(H, ring, comp, squash):
    """Foreshorten `ring`'s atoms laterally about its radial spine (P->ring) and
    re-hang every exocyclic subtree rigidly off its moved ring carbon (so the
    decorations keep their fan, not pile on the spine).  Returns (orig, perp):
    `orig` = pre-move positions for the whole component (to revert), `perp` = the
    lateral unit used (the depth axis)."""
    pid = next((a for a in ring if any(H.label[n] == "P" for n in H.adj[a])), None)
    p = next(n for n in H.adj[pid] if H.label[n] == "P")
    px, py = H.pos[p]
    cx = sum(H.pos[a][0] for a in ring) / len(ring)
    cy = sum(H.pos[a][1] for a in ring) / len(ring)
    ax, ay = cx - px, cy - py
    an = math.hypot(ax, ay) or 1e-9
    ax, ay = ax / an, ay / an                    # spine unit (radial, kept)
    prx, pry = -ay, ax                           # lateral unit (squashed = depth)
    exo = _exo_subtrees(H, ring)
    orig = {a: H.pos[a] for a in comp}
    rs = set(ring)
    for a in ring:                               # foreshorten ring atoms only
        if a in H.pinned:
            continue
        rx, ry = H.pos[a][0] - px, H.pos[a][1] - py
        u = rx * ax + ry * ay
        v = rx * prx + ry * pry
        H.pos[a] = (px + u * ax + squash * v * prx, py + u * ay + squash * v * pry)
    # re-hang each exocyclic subtree along the EXTERIOR-ANGLE BISECTOR at its ring
    # carbon (straight out from the ring), so decorations splay at clean, even
    # angles instead of inheriting their stale pre-tilt directions.
    ncx = sum(H.pos[a][0] for a in ring) / len(ring)
    ncy = sum(H.pos[a][1] for a in ring) / len(ring)
    for c, subs in exo.items():
        sx = sy = 0.0
        for n in (n for n in H.adj[c] if n in rs):
            vx, vy = H.pos[n][0] - H.pos[c][0], H.pos[n][1] - H.pos[c][1]
            ln = math.hypot(vx, vy) or 1e-9
            sx += vx / ln
            sy += vy / ln
        ox, oy = -sx, -sy                        # away from the ring neighbours
        ol = math.hypot(ox, oy)
        if ol < 1e-6:                            # straight carbon -> radial outward
            ox, oy = H.pos[c][0] - ncx, H.pos[c][1] - ncy
            ol = math.hypot(ox, oy) or 1e-9
        ox, oy = ox / ol, oy / ol
        for (d, atoms) in subs:
            blen = _vlen(orig[c], orig[d])
            tgt = (H.pos[c][0] + ox * blen, H.pos[c][1] + oy * blen)
            dx, dy = tgt[0] - orig[d][0], tgt[1] - orig[d][1]
            for a in atoms:
                if a in H.pinned:
                    continue
                H.pos[a] = (orig[a][0] + dx, orig[a][1] + dy)
    return orig, (prx, pry)


def _mirror_partner(H, ring, cands, done):
    """The substituent ring on the OTHER side of the Cu-H axis whose centroid is
    the mirror image (across y = Cu_y) of `ring`'s -- so a tilt can be matched
    symmetrically.  None if there is no metal or no close mirror."""
    if H.metal is None:
        return None
    cuy = H.pos[H.metal][1]
    cx = sum(H.pos[a][0] for a in ring) / len(ring)
    cy = sum(H.pos[a][1] for a in ring) / len(ring)
    mx, my = cx, 2 * cuy - cy
    best = None
    bestd = 0.9 * L
    for (p2, ipso2, ring2, comp2) in cands:
        if frozenset(ring2) == frozenset(ring) or frozenset(ring2) in done:
            continue
        c2x = sum(H.pos[a][0] for a in ring2) / len(ring2)
        c2y = sum(H.pos[a][1] for a in ring2) / len(ring2)
        d = math.hypot(c2x - mx, c2y - my)
        if d < bestd:
            bestd, best = d, (p2, ipso2, ring2, comp2)
    return best


def tilt_relief(H, squash=0.5):
    """LAST-RESORT relief for crowded aromatic substituents: tilt a ring 'into
    the plane of the canvas' -- foreshorten its ring atoms laterally about the
    radial P->ring spine (decorations re-hung so they keep their fan) -- so it
    takes up less sideways room and any bond CROSSING through it is relieved.
    The near (outer) side is redrawn with thickening WEDGES, the receding side
    plain (see _restyle_tilted_ring for the 3-D depth logic).  Fires on rings
    that overlap, cross, OR are merely crowded; when a ring is tilted its MIRROR
    partner across the Cu-H axis is tilted to match, so the depiction stays
    symmetric.  Kept only when it adds no crossing/overlap and removes a crossing,
    an overlap, or a meaningful amount of crowding; else reverted."""
    def n_over():
        return sum(1 for (a, b, mn) in H.overlaps
                   if _vlen(H.pos[a], H.pos[b]) < mn)

    def ring_crowd(ring):
        """Crowd-band penetration (squared) that this ring is involved in -- the
        same soft measure as H._crowd but restricted to pairs touching the ring,
        so a ring that is merely tight (no strict overlap) still scores > 0."""
        rs = set(ring)
        tot = 0.0
        for (a, b, mn) in H.overlaps:
            if a not in rs and b not in rs:
                continue
            d = _vlen(H.pos[a], H.pos[b])
            clear = mn * 1.5
            if d < clear:
                tot += ((clear - d) / clear) ** 2
        return tot

    def ring_bad(ring):
        rs = set(ring)
        ov = sum(1 for (a, b, mn) in H.overlaps
                 if (a in rs or b in rs) and _vlen(H.pos[a], H.pos[b]) < mn)
        cr = sum(1 for (a, b) in H._raw_bonds if (a in rs) != (b in rs)
                 for (c, d) in H._raw_bonds
                 if c not in rs and d not in rs and len({a, b, c, d}) == 4
                 and _seg_cross(H.pos[a], H.pos[b], H.pos[c], H.pos[d]))
        return (ov + cr, ring_crowd(ring))

    cands = [c for c in _aryl_substituent_rings(H)]
    # a ring is worth tilting if it overlaps / crosses, OR is genuinely crowded
    bad = [c for c in cands if ring_bad(c[2])[0] > 0 or ring_bad(c[2])[1] > 0.25]
    bad.sort(key=lambda c: (-ring_bad(c[2])[0], -ring_bad(c[2])[1]))

    tilted = 0
    done = set()
    for (p, ipso, ring, comp) in bad:
        bb = ring_bad(ring)
        if frozenset(ring) in done or (bb[0] == 0 and bb[1] <= 0.25):
            continue
        before = (H.count_crossings(), n_over(), H._crowd())
        orig, perp = _tilt_geometry(H, ring, comp, squash)
        after = (H.count_crossings(), n_over(), H._crowd())
        # keep a tilt that does not add crossings/overlaps and either removes a
        # crossing / overlap, gains whitespace, OR depicts a clearly crowded ring
        # in perspective -- the no-worse guard stops any label pile-up.
        improved = (after[0] <= before[0] and after[1] <= before[1]
                    and (after[0] < before[0] or after[1] < before[1]
                         or after[2] < before[2] - 0.02 or bb[1] > 0.40))
        if not improved:
            for a, q in orig.items():
                H.pos[a] = q
            continue
        _restyle_tilted_ring(H, ring, perp)
        done.add(frozenset(ring))
        tilted += 1
        # match the mirror partner so the picture stays symmetric
        part = _mirror_partner(H, ring, cands, done)
        if part:
            p2, ipso2, ring2, comp2 = part
            c0 = H.count_crossings()
            orig2, perp2 = _tilt_geometry(H, ring2, comp2, squash)
            if H.count_crossings() <= c0:
                _restyle_tilted_ring(H, ring2, perp2)
                done.add(frozenset(ring2))
                tilted += 1
            else:
                for a, q in orig2.items():
                    H.pos[a] = q
    if tilted:
        frozen = set().union(*done) if done else set()
        _settle_around_tilts(H, frozen)
    return tilted


def _settle_around_tilts(H, frozen):
    """After tilting, a SLIGHT relax of the REST of the molecule with the tilted
    rings (and Cu/H) held fixed, so the non-tilted bonds tidy up -- decorations
    settle to even lengths and the other substituents take cleaner junction
    angles (which also nudges them a little further out).  Angle-favouring
    weights; reverted if it introduces a crossing or extra overlap."""
    import relaxer_energy
    def n_over():
        return sum(1 for (a, b, mn) in H.overlaps
                   if _vlen(H.pos[a], H.pos[b]) < mn)
    before = (H.count_crossings(), n_over(), H.metrics()["angleDevDeg"])
    saved_pos = {i: H.pos[i] for i in H.ids}
    saved_pin = set(H.pinned)
    H.pinned = saved_pin | set(frozen)
    try:
        relaxer_energy.relax(H, {"w_bond": 4.0, "w_angle": 3.0, "w_overlap": 18.0,
                                 "w_rigid": 67.0, "maxiter": 60})
    finally:
        H.pinned = saved_pin
    after = (H.count_crossings(), n_over(), H.metrics()["angleDevDeg"])
    # only keep the settle if it makes NOTHING worse (no new crossing / overlap,
    # and the junction angles do not deteriorate)
    if after[0] > before[0] or after[1] > before[1] or after[2] > before[2] + 0.5:
        for i, q in saved_pos.items():
            H.pos[i] = q


def _backbone_atoms(H):
    """The set of backbone atoms (the largest non-Cu component hanging off each
    P), i.e. everything that is NOT a P substituent."""
    bb = set()
    for p, (comp, backbone_n) in _p_components(H).items():
        bb |= comp[backbone_n]
    return bb


def swing_off_backbone(H, max_deg=95, step=5):
    """Rotate any P substituent that CROSSES THE BACKBONE bodily about its P,
    scanning +/- up to max_deg, to the angle that minimises (#overlaps it makes
    with backbone atoms, then its total #overlaps).  Unlike declutter's small
    steps this clears head-on substituent-on-backbone ring crossings (e.g. a
    DPEphos P-aryl folded over the diaryl-ether phenyl).  A pure rigid rotation,
    so ring shapes are preserved and decorations keep their spread."""
    bb = _backbone_atoms(H)

    def over_with(atoms, against):
        s = set(atoms)
        return sum(1 for (a, b, mn) in H.overlaps
                   if ((a in s and b in against) or (b in s and a in against))
                   and _vlen(H.pos[a], H.pos[b]) < mn)

    def tot_over(atoms):
        s = set(atoms)
        return sum(1 for (a, b, mn) in H.overlaps
                   if (a in s or b in s) and _vlen(H.pos[a], H.pos[b]) < mn)

    degs = [d for k in range(1, max_deg // step + 1) for d in (k * step, -k * step)]
    moved = 0
    for p, (comp, backbone_n) in _p_components(H).items():
        for n, atoms in comp.items():
            if n == backbone_n:
                continue
            mv = [a for a in atoms if a not in H.pinned]
            if not mv or over_with(mv, bb) == 0:
                continue
            px, py = H.pos[p]
            orig = {a: H.pos[a] for a in mv}
            best, bestd = (over_with(mv, bb), tot_over(mv)), 0.0
            for deg in degs:
                th = math.radians(deg)
                c, s = math.cos(th), math.sin(th)
                for a in mv:
                    x, y = orig[a][0] - px, orig[a][1] - py
                    H.pos[a] = (px + c * x - s * y, py + s * x + c * y)
                cost = (over_with(mv, bb), tot_over(mv))
                for a in mv:
                    H.pos[a] = orig[a]
                if cost < best:
                    best, bestd = cost, deg
            if bestd:
                th = math.radians(bestd)
                c, s = math.cos(th), math.sin(th)
                for a in mv:
                    x, y = orig[a][0] - px, orig[a][1] - py
                    H.pos[a] = (px + c * x - s * y, py + s * x + c * y)
                moved += 1
    return moved


def uncross(H, pushes=(0.0, 0.14, 0.28, 0.42), max_deg=95, step=5):
    """Resolve substituent CROSSINGS (bonds that visually intersect) and the
    overlaps that go with them.  For each crossing substituent: rotate it about
    its P and, if rotation alone will not separate it, push it radially OUTWARD
    along its spine -- ELONGATING the P-substituent bond.  A few slightly-long
    bonds are an acceptable price for removing a crossing, per the brief.  Greedy
    per group; keeps the move minimising (crossings, overlaps, elongation)."""
    degs = [0.0] + [d for k in range(1, max_deg // step + 1)
                    for d in (k * step, -k * step)]

    def group_cost(mv, p):
        s2 = set(mv) | {p}
        gb = [(a, b) for (a, b) in H._raw_bonds if a in s2 and b in s2]
        ob = [(a, b) for (a, b) in H._raw_bonds if not (a in s2 and b in s2)]
        cr = 0
        for (a, b) in gb:
            pa, pb = H.pos[a], H.pos[b]
            for (c, d) in ob:
                if len({a, b, c, d}) == 4 and _seg_cross(pa, pb, H.pos[c], H.pos[d]):
                    cr += 1
        s = set(mv)
        ov = sum(1 for (a, b, mn) in H.overlaps
                 if (a in s or b in s) and _vlen(H.pos[a], H.pos[b]) < mn)
        return cr, ov

    moved = 0
    for p, (comp, backbone_n) in _p_components(H).items():
        for n, atoms in comp.items():
            if n == backbone_n:
                continue
            mv = [a for a in atoms if a not in H.pinned]
            if not mv or group_cost(mv, p)[0] == 0:      # only crossing groups
                continue
            px, py = H.pos[p]
            orig = {a: H.pos[a] for a in mv}
            base = group_cost(mv, p)
            best = (base[0], base[1], 0.0)        # (crossings, overlaps, push)
            bestmove = None
            for deg in degs:
                th = math.radians(deg)
                cz, sz = math.cos(th), math.sin(th)
                rot = {}
                for a in mv:
                    x, y = orig[a][0] - px, orig[a][1] - py
                    rot[a] = (px + cz * x - sz * y, py + sz * x + cz * y)
                gx = sum(rot[a][0] for a in mv) / len(mv) - px
                gy = sum(rot[a][1] for a in mv) / len(mv) - py
                gn = math.hypot(gx, gy) or 1e-9
                ux, uy = gx / gn, gy / gn         # outward spine after rotation
                for push in pushes:
                    for a in mv:
                        H.pos[a] = (rot[a][0] + ux * push * L,
                                    rot[a][1] + uy * push * L)
                    cr, ov = group_cost(mv, p)
                    for a in mv:
                        H.pos[a] = orig[a]
                    cost = (cr, ov, push)
                    if cost < best:
                        best, bestmove = cost, (deg, push)
            if bestmove and (best[0] < base[0] or best[1] < base[1]):
                deg, push = bestmove
                th = math.radians(deg)
                cz, sz = math.cos(th), math.sin(th)
                rot = {}
                for a in mv:
                    x, y = orig[a][0] - px, orig[a][1] - py
                    rot[a] = (px + cz * x - sz * y, py + sz * x + cz * y)
                gx = sum(rot[a][0] for a in mv) / len(mv) - px
                gy = sum(rot[a][1] for a in mv) / len(mv) - py
                gn = math.hypot(gx, gy) or 1e-9
                ux, uy = gx / gn, gy / gn
                for a in mv:
                    H.pos[a] = (rot[a][0] + ux * push * L, rot[a][1] + uy * push * L)
                moved += 1
    return moved


def relief_pass(H, squash=0.5):
    """LAST-RESORT relief for genuinely stubborn cells, applied in order:
      1. swing any substituent folded over the backbone out into open space,
      2. a gentle overlap-separation settle,
      3. declutter (small in-place rotations of crowded groups),
      4. uncross: rotate / radially elongate substituents to remove CROSSINGS,
      5. tilt a still-crowded aromatic substituent 'into the plane' -- self-gated.
    Deterministic, so it reproduces exactly on a cached replay."""
    swing_off_backbone(H)
    bodies, pin, inv, translate, _ = body_helpers(H)
    overlap_relax(H, bodies, pin, inv, translate, w_over=0.5, iters=80)
    declutter(H, angles=BIG_DECL, passes=2)
    uncross(H)
    tilt_relief(H, squash=squash)


BIG_DECL = (12, -12, 24, -24, 36, -36, 50, -50, 68, -68, 85, -85)


def quality_loss(H, c=DEFAULT_JUDGE):
    """Scalar numerical quality of a (relaxed) Harness -- lower is better."""
    m = H.metrics()
    return (c["bond"] * m["bondCV"]
            + c["ring"] * m["ringEdgeCV"]
            + c["ringang"] * m["ringAngleDev"]
            + c.get("jang", 0.0) * m.get("angleDevDeg", 0.0)
            + c["coord"] * m["coordLenErr"]
            + c["overlap"] * m["overlap"]
            + c.get("crowd", 0.0) * m.get("crowd", 0.0)
            + c["sym"] * m["symDev"])


def mean_loss(relax_fn, params, ligands=LIGANDS, judge=DEFAULT_JUDGE):
    """Mean quality_loss of relax_fn(params) across the ligand set (the optimiser
    target). Returns a large penalty if the relaxer raises."""
    tot = 0.0
    for key in ligands:
        H = Harness(key)
        try:
            relax_fn(H, params)
        except Exception:
            return 1e6
        tot += quality_loss(H, judge)
    return tot / len(ligands)


def optimize_method(relax_fn, bounds, param_names, ligands=LIGANDS,
                    judge=DEFAULT_JUDGE, optimizer="de", maxiter=25, seed=0):
    """Black-box optimise a method's parameters against the numerical judge.

    `relax_fn(H, params)` takes a dict {name: value}; `bounds` is a list of
    (lo, hi) aligned with `param_names`. Returns (best_params_dict, best_loss,
    per_ligand_metrics). optimizer: 'de' (differential evolution, global),
    'anneal' (dual annealing), or 'nm' (Nelder-Mead, local)."""
    from scipy.optimize import differential_evolution, dual_annealing, minimize

    def obj(x):
        return mean_loss(relax_fn, dict(zip(param_names, x)), ligands, judge)

    if optimizer == "de":
        res = differential_evolution(obj, bounds, maxiter=maxiter, seed=seed,
                                     tol=1e-4, popsize=10, polish=True,
                                     mutation=(0.4, 1.0), recombination=0.8)
        best, loss = res.x, res.fun
    elif optimizer == "anneal":
        res = dual_annealing(obj, bounds, maxiter=maxiter, seed=seed)
        best, loss = res.x, res.fun
    else:
        x0 = [(lo + hi) / 2 for lo, hi in bounds]
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": maxiter * 10 * len(bounds),
                                "xatol": 1e-3, "fatol": 1e-4})
        best, loss = res.x, res.fun

    params = dict(zip(param_names, [float(v) for v in best]))
    per = {}
    for key in ligands:
        H = Harness(key)
        relax_fn(H, params)
        per[key] = H.metrics()
    return params, float(loss), per


# ============================================================================ #
#  panel renderer: run a relaxer over all 4 ligands -> SVG -> PNG -> montage
# ============================================================================ #
PW_CORE = "/tmp/node_modules/playwright-core"


def make_panel(relax_fn, weights, out_png, tag="variant", steps_note="",
               outdir=None):
    """Relax all four ligands with `relax_fn(H, weights)`, render a 2x2 montage
    to out_png, and return {ligand: metrics}.  Requires node + playwright-core."""
    from PIL import Image, ImageDraw, ImageFont
    outdir = Path(outdir or (HERE / "out" / tag))
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    keys = []
    for key in LIGANDS:
        H = Harness(key)
        relax_fn(H, weights)
        H.save_svg(outdir / f"{key}.svg")
        results[key] = H.metrics()
        keys.append(key)
    # SVG -> PNG via playwright (svg2png.mjs reads out/<tag>/<key>.svg? no -> reads
    # methods/out by default; we point it at our dir through an env override)
    svgs = [str(outdir / f"{k}.svg") for k in keys]
    pngs = [str(outdir / f"{k}.png") for k in keys]
    _svg_to_png(svgs, pngs)
    # montage 2x2
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)
    pad = 6
    panel = Image.new("RGB", (w * 2 + pad * 3, h * 2 + pad * 3 + 26), "#dddddd")
    d = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    d.text((8, 5), f"{tag}  {steps_note}", fill="black", font=font)
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, 2)
        panel.paste(im, (pad + c * (w + pad), 26 + pad + r * (h + pad)))
    panel.save(out_png)
    return results


def _svg_to_png(svg_paths, png_paths):
    """Rasterise SVGs with headless Chromium (inline node script)."""
    pairs = list(zip(svg_paths, png_paths))
    script = r"""
const { createRequire } = require('module');
const path = require('path'); const fs = require('fs');
const { chromium } = require(process.env.PW_CORE);
const chrome = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
const pairs = JSON.parse(process.env.PAIRS);
(async () => {
  const b = await chromium.launch({ executablePath: chrome,
      args: ['--no-sandbox','--disable-gpu'] });
  for (const [svg, png] of pairs) {
    const s = fs.readFileSync(svg, 'utf8');
    const mw = s.match(/width="(\d+)"/), mh = s.match(/height="(\d+)"/);
    const vw = mw ? +mw[1] : 760, vh = mh ? +mh[1] : 640;
    const p = await b.newPage({ viewport: { width: vw, height: vh },
        deviceScaleFactor: 2 });
    await p.setContent('<!doctype html><body style="margin:0">'+s+'</body>',
        { waitUntil: 'load' });
    await p.waitForTimeout(40);
    await p.screenshot({ path: png });
    await p.close();
  }
  await b.close();
})();
"""
    import json
    env = {"PW_CORE": PW_CORE, "PAIRS": json.dumps(pairs)}
    import os
    e = dict(os.environ)
    e.update(env)
    subprocess.run(["node", "-e", script], check=True, env=e,
                   capture_output=True)


# ============================================================================ #
#  REFERENCE relaxer -- a plain rigid-body spring solver, the baseline that the
#  workflow variants are measured against (and a worked example of the API).
# ============================================================================ #
def body_helpers(H):
    """Return (bodies, pin, inv, translate, rotate) -- the standard rigid-body
    movement primitives every relaxer needs.  bodies[i] is a list of atom ids;
    pin[i] True if the body contains a pinned atom (Cu/H); inv[i] = 1/size (0 if
    pinned).  translate(i,dx,dy) and rotate(i,(px,py),dth) move a whole body."""
    bodies = [list(b) for b in H.bodies]
    pin = [any(a in H.pinned for a in bd) for bd in bodies]
    inv = [0.0 if pin[i] else 1.0 / len(bodies[i]) for i in range(len(bodies))]

    def translate(bi, dx, dy):
        if pin[bi]:
            return
        for a in bodies[bi]:
            H.pos[a] = (H.pos[a][0] + dx, H.pos[a][1] + dy)

    def rotate(bi, piv, dth):
        if pin[bi] or abs(dth) < 1e-12:
            return
        c, s = math.cos(dth), math.sin(dth)
        for a in bodies[bi]:
            x, y = H.pos[a][0] - piv[0], H.pos[a][1] - piv[1]
            H.pos[a] = (piv[0] + c * x - s * y, piv[1] + s * x + c * y)

    return bodies, pin, inv, translate, rotate


def overlap_relax(H, bodies, pin, inv, translate, w_over=0.5, iters=120):
    """Shared overlap-separation pass: translate non-bonded overlapping bodies
    apart (label-aware minimum separation). Pinned bodies are immovable."""
    for _ in range(iters):
        for (a, b, mn) in H.overlaps:
            ba, bb = H.body_of[a], H.body_of[b]
            pa, pb = H.pos[a], H.pos[b]
            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
            d = math.hypot(dx, dy) or 1e-9
            if d < mn:
                corr = w_over * (mn - d)
                s = inv[ba] + inv[bb]
                if s == 0:
                    continue
                ux, uy = dx / d, dy / d
                translate(ba, -ux * corr * inv[ba] / s, -uy * corr * inv[ba] / s)
                translate(bb, ux * corr * inv[bb] / s, uy * corr * inv[bb] / s)


def reference_relax(H, w):
    """Baseline = MINIMAL, placement-preserving rigid-body fix.

    The Method-C template already has good ANGLES and near-zero overlap; only a
    few inter-fragment bond LENGTHS are wrong (P-Cu too long, P-aryl too short).
    So:
      1. translate the WHOLE ligand toward the pinned Cu until the mean P-Cu
         coordination length hits target -- a rigid shift that preserves every
         internal angle and bond;
      2. snap each leaf body (aryl rings, label arms) radially along its joint to
         the target length, leaves first, so directions (hence angles) are kept;
      3. a gentle overlap-separation pass.
    Weights: w['overlap'] (separation strength), w['ov_iters'].
    """
    bodies, pin, inv, translate, rotate = body_helpers(H)
    nonpin = [bi for bi in range(len(bodies)) if not pin[bi]]

    # 1) rigid translate to set mean P-Cu length
    cu = H.pos[H.metal]
    tgtP = TGT_PCU * L
    for _ in range(40):
        fx = fy = 0.0
        for p in H.donors:
            dx, dy = H.pos[p][0] - cu[0], H.pos[p][1] - cu[1]
            d = math.hypot(dx, dy) or 1e-9
            f = (d - tgtP) / d
            fx += f * dx
            fy += f * dy
        n = max(1, len(H.donors))
        for bi in nonpin:
            translate(bi, -0.5 * fx / n, -0.5 * fy / n)

    # 2) leaf-inward radial joint-length snap
    deg = {bi: 0 for bi in range(len(bodies))}
    for (a, b, t) in H.joints:
        deg[H.body_of[a]] += 1
        deg[H.body_of[b]] += 1
    for _ in range(6):
        for (a, b, t) in sorted(
                H.joints,
                key=lambda j: -max(deg[H.body_of[j[0]]], deg[H.body_of[j[1]]])):
            ba, bb = H.body_of[a], H.body_of[b]
            if deg[ba] <= deg[bb] and not pin[ba]:
                mv, pm, pa_ = ba, a, b
            elif not pin[bb]:
                mv, pm, pa_ = bb, b, a
            else:
                continue
            panch, pmov = H.pos[pa_], H.pos[pm]
            dx, dy = pmov[0] - panch[0], pmov[1] - panch[1]
            d = math.hypot(dx, dy) or 1e-9
            ux, uy = dx / d, dy / d
            translate(mv, panch[0] + ux * t - pmov[0], panch[1] + uy * t - pmov[1])

    # 3) overlap cleanup
    overlap_relax(H, bodies, pin, inv, translate,
                  w_over=w.get("overlap", 0.5), iters=w.get("ov_iters", 150))


if __name__ == "__main__":
    # smoke test: seed metrics vs reference-relax metrics for every ligand
    for key in LIGANDS:
        H = Harness(key)
        seed = H.metrics()
        reference_relax(H, {"bond": 1.0, "angle": 0.5, "overlap": 0.6})
        print(f"{key:14s} bodies={len(H.bodies):2d} rings={len(H.rings):2d} "
              f"joints={len(H.joints):2d} angles={len(H.angles):2d}")
        print(f"   seed   {seed}")
        print(f"   relax  {H.metrics()}")
