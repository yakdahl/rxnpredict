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

    def __init__(self, key):
        self.key = key
        sc, title = M.BUILDERS[key]()
        self.scene = sc
        self.title = title
        self.pos = {i: (float(a.pos[0]), float(a.pos[1]))
                    for i, a in sc.atoms.items()}
        self.label = {i: a.label for i, a in sc.atoms.items()}
        self.ids = list(sc.atoms)

        # ---- adjacency + bond kinds ----
        self.adj = {i: [] for i in self.ids}
        self.kind = {}
        self._raw_bonds = []
        for b in sc.bonds:
            self.adj[b.a].append(b.b)
            self.adj[b.b].append(b.a)
            self.kind[frozenset((b.a, b.b))] = b.kind
            self._raw_bonds.append((b.a, b.b))

        # ---- rings + fused ring systems -> rigid bodies ----
        self.rings = self._find_rings()
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
        """Fuse rings sharing an atom into rigid bodies; lone atoms are 1-bodies."""
        groups = [set(r) for r in self.rings]
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
                la, lb = self.label[a], self.label[b]
                if la and lb:
                    mn = 0.62 * L + 0.10 * L * (len(la) + len(lb))
                elif la or lb:
                    mn = 0.78 * L
                else:
                    mn = 0.55 * L
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
            "coordLenErr": round(sum(coorderr) / len(coorderr), 3) if coorderr else 0,
            "symDev": round(self._sym_dev(), 3),
        }

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
    "bond": 9.0,        # bondCV  -- uniform skeletal bond lengths (top priority)
    "ring": 7.0,        # ringEdgeCV
    "ringang": 0.05,    # ringAngleDev (deg) -- regular polygons
    "coord": 1.6,       # coordLenErr -- P-Cu / Cu-H at target length
    "overlap": 1.0,     # per overlapping non-bonded pair (count) -- hard penalty
    "sym": 1.2,         # symDev -- C2 symmetry about the Cu-H axis
}


def quality_loss(H, c=DEFAULT_JUDGE):
    """Scalar numerical quality of a (relaxed) Harness -- lower is better."""
    m = H.metrics()
    return (c["bond"] * m["bondCV"]
            + c["ring"] * m["ringEdgeCV"]
            + c["ringang"] * m["ringAngleDev"]
            + c["coord"] * m["coordLenErr"]
            + c["overlap"] * m["overlap"]
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
    const p = await b.newPage({ viewport: { width: 760, height: 640 },
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
