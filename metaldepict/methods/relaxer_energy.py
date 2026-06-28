"""energy.py -- GLOBAL ENERGY MINIMISATION relaxer for the Cu-H bisphosphine
Method-C template seeds (relax_harness.Harness).

METHOD
------
Define a scalar potential energy over the FREE (non-pinned) atom coordinates:

    E(x) =  w_bond    * sum_(a,b,t)      (|p_a - p_b| - t)^2          # bonds -> target len
          + w_angle   * sum_(i,j,k,a0)   (theta_ijk - a0)^2           # junction angles
          + w_overlap * sum_(a,b,mn)     max(0, mn - |p_a-p_b|)^2     # non-bonded clearance
          + w_rigid   * sum_(a,b,s)      (|p_a - p_b| - s)^2          # intra-ring rigidity

The intra-body rigid_pairs term (every atom-pair distance inside a ring system,
seeded from the already-regular template) is what keeps the rings RIGID and
REGULAR -- with a large w_rigid the polygons cannot distort, so ringEdgeCV and
ringAngleDev stay ~0 even though every atom moves independently.

We minimise E with scipy.optimize.minimize(method='L-BFGS-B') over the free atom
coordinates only (pinned Cu + hydride held fixed, so the Cu-H bond stays exactly
horizontal with H to the right).  An ANALYTIC, fully-vectorised (numpy) gradient
is supplied for every term, so each energy+gradient evaluation is a handful of
array ops and L-BFGS converges within the ~150 ms-per-relax budget.

PARAMETERIZATION (this method's own parameters = the ENERGY-TERM WEIGHTS):
    w_bond, w_angle, w_overlap, w_rigid   -- relative strengths of the four terms
    maxiter                               -- L-BFGS iteration cap (cost knob)
The black-box optimiser (optimize_method) tunes these weights against the judge.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import minimize

PARAM_NAMES = ["w_bond", "w_angle", "w_overlap", "w_rigid", "maxiter"]
# sensible search ranges.  w_rigid is allowed to be large (rings must win), the
# overlap penalty is one-sided so it can also be strong, angle is the soft term.
BOUNDS = [
    (1.0, 20.0),     # w_bond     -- pull inter-fragment bonds to target length
    (0.0, 4.0),      # w_angle    -- soft junction-angle preference (judge ignores it)
    (1.0, 30.0),     # w_overlap  -- non-bonded clearance (one-sided hinge)
    (5.0, 80.0),     # w_rigid    -- keep rings rigid + regular (must dominate)
    (60.0, 200.0),   # maxiter    -- L-BFGS iteration cap
]


class _Problem:
    """Pre-compiled, vectorised energy/gradient for one Harness.

    Layout: a single position array `P` (n_atoms x 2) holds EVERY atom.  Free
    atoms map to slots in the optimisation vector x; pinned atoms keep their
    fixed seed coordinates.  `free_slot[i]` is the x-slot index of atom i, or -1
    if pinned.  Gradients are accumulated on the full array then scattered to the
    free rows via np.add.at, so pinned atoms never receive a displacement.
    """

    def __init__(self, H):
        ids = list(H.ids)
        self.ids = ids
        self.row = {i: r for r, i in enumerate(ids)}          # atom id -> row
        self.n = len(ids)
        self.P = np.array([H.pos[i] for i in ids], dtype=float)

        free_ids = [i for i in ids if i not in H.pinned]
        self.free_ids = free_ids
        self.free_rows = np.array([self.row[i] for i in free_ids], dtype=int)
        self.nfree = len(free_ids)
        # row -> slot in x (-1 if pinned)
        self.slot = np.full(self.n, -1, dtype=int)
        for s, i in enumerate(free_ids):
            self.slot[self.row[i]] = s

        # ---- distance terms: bonds, rigid pairs, overlaps -------------------
        self.bA, self.bB, self.bT = self._pairs(H.bonds)
        self.rA, self.rB, self.rT = self._pairs(H.rigid_pairs)
        self.oA, self.oB, self.oT = self._pairs(H.overlaps)

        # ---- angle terms: (i,j,k, ideal_rad) --------------------------------
        if H.angles:
            self.aI = np.array([self.row[t[0]] for t in H.angles], dtype=int)
            self.aJ = np.array([self.row[t[1]] for t in H.angles], dtype=int)
            self.aK = np.array([self.row[t[2]] for t in H.angles], dtype=int)
            self.a0 = np.array([math.radians(t[3]) for t in H.angles], dtype=float)
        else:
            self.aI = self.aJ = self.aK = np.zeros(0, dtype=int)
            self.a0 = np.zeros(0, dtype=float)

    def _pairs(self, table):
        if not table:
            z = np.zeros(0, dtype=int)
            return z, z, np.zeros(0, dtype=float)
        A = np.array([self.row[t[0]] for t in table], dtype=int)
        B = np.array([self.row[t[1]] for t in table], dtype=int)
        T = np.array([t[2] for t in table], dtype=float)
        return A, B, T

    def pack(self):
        return self.P[self.free_rows].reshape(-1).copy()

    def _scatter(self, P):
        """Write the current x into the full position array's free rows."""
        self.P[self.free_rows] = P.reshape(-1, 2)

    def _dist_grad(self, A, B, T, weight, one_sided, gradP):
        """Accumulate sum_w*(|d|-T)^2 and its gradient into gradP. Returns E."""
        if weight == 0.0 or A.size == 0:
            return 0.0
        d = self.P[B] - self.P[A]                       # (m,2)
        r = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        r_safe = np.where(r < 1e-12, 1e-12, r)
        diff = r - T
        if one_sided:
            mask = diff < 0.0                           # only penalise overlap
            if not mask.any():
                return 0.0
            diff = np.where(mask, diff, 0.0)
        E = weight * float(np.sum(diff * diff))
        coef = (2.0 * weight * diff / r_safe)[:, None]  # (m,1)
        f = coef * d                                    # force-ish on B
        np.add.at(gradP, B, f)
        np.add.at(gradP, A, -f)
        return E

    def _angle_grad(self, weight, gradP):
        if weight == 0.0 or self.aI.size == 0:
            return 0.0
        pj = self.P[self.aJ]
        u = self.P[self.aI] - pj
        v = self.P[self.aK] - pj
        ru = np.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2)
        rv = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
        ru_s = np.where(ru < 1e-9, 1e-9, ru)
        rv_s = np.where(rv < 1e-9, 1e-9, rv)
        dot = u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1]
        cosang = np.clip(dot / (ru_s * rv_s), -1.0, 1.0)
        theta = np.arccos(cosang)
        diff = theta - self.a0
        E = weight * float(np.sum(diff * diff))
        sin_t = np.sqrt(np.maximum(1e-12, 1.0 - cosang * cosang))
        c = cosang[:, None]
        ru2 = (ru_s * ru_s)[:, None]
        rv2 = (rv_s * rv_s)[:, None]
        ruv = (ru_s * rv_s)[:, None]
        st = sin_t[:, None]
        gi = (c * u / ru2 - v / ruv) / st
        gk = (c * v / rv2 - u / ruv) / st
        gj = -(gi + gk)
        coef = (2.0 * weight * diff)[:, None]
        np.add.at(gradP, self.aI, coef * gi)
        np.add.at(gradP, self.aK, coef * gk)
        np.add.at(gradP, self.aJ, coef * gj)
        return E

    def energy_grad(self, x, weights):
        wb, wa, wo, wr = weights
        self._scatter(x)
        gradP = np.zeros((self.n, 2), dtype=float)
        E = 0.0
        E += self._dist_grad(self.bA, self.bB, self.bT, wb, False, gradP)
        E += self._dist_grad(self.rA, self.rB, self.rT, wr, False, gradP)
        E += self._dist_grad(self.oA, self.oB, self.oT, wo, True, gradP)
        E += self._angle_grad(wa, gradP)
        g = gradP[self.free_rows].reshape(-1)           # pinned rows dropped
        return E, g


def relax(H, p):
    """Global energy minimisation: minimise E over free atom coords with L-BFGS-B.

    p: dict with keys PARAM_NAMES (energy-term weights + iteration cap).
    Mutates H.pos in place.  Pinned atoms (Cu, hydride) are never moved.
    Rings stay rigid via the strong rigid_pairs term.
    """
    wb = float(p.get("w_bond", 8.0))
    wa = float(p.get("w_angle", 1.0))
    wo = float(p.get("w_overlap", 10.0))
    wr = float(p.get("w_rigid", 40.0))
    maxiter = int(round(float(p.get("maxiter", 120))))
    maxiter = max(20, min(400, maxiter))

    prob = _Problem(H)
    if prob.nfree == 0:
        return
    x0 = prob.pack()
    weights = (wb, wa, wo, wr)

    res = minimize(
        prob.energy_grad, x0, args=(weights,),
        method="L-BFGS-B", jac=True,
        options={"maxiter": maxiter, "maxfun": maxiter * 4,
                 "ftol": 1e-9, "gtol": 1e-7},
    )
    xf = res.x
    if not np.all(np.isfinite(xf)):
        return                       # safety: never write NaN/inf back
    prob._scatter(xf)
    for r, i in enumerate(prob.free_ids):
        row = prob.free_rows[r]
        H.pos[i] = (float(prob.P[row, 0]), float(prob.P[row, 1]))
