/* metaldepict viewer — adjustable 2D depiction + 2D physics engine.
 * No dependencies. Reads window.MOLECULE (see molecule_data.js).
 *
 * Design:
 *  - persistent per-atom world coordinates (y-up)
 *  - abbreviation collapse levels rebuild a "view graph" of visible nodes
 *  - physics: 1-2 bond springs + 1-3 angle springs (standardised geometry)
 *             + fragment repulsion / collision (de-overlap)
 *             + optional C2 symmetrisation
 *  - drag to adjust, double-click to pin, scroll zoom, background pan
 */
(function () {
  "use strict";
  const M = window.MOLECULE;
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.getElementById("svg");

  // ---------------------------------------------------------------- styling
  const COLORS = { C: "#23262b", P: "#e8821e", O: "#cc2b2b", N: "#2b5fcc",
    F: "#2aa84a", Cl: "#2aa84a", Cu: "#c06a2b", H: "#5566aa", S: "#c9a21a", B: "#c97" };
  const LABEL_ELEMENTS = new Set(["P", "O", "N", "F", "Cl", "Cu", "S", "B", "Si"]);

  // -------------------------------------------------------- persistent state
  const atom = new Map();        // id -> {x,y,vx,vy,pinned}
  const atomData = new Map();    // id -> source record
  M.atoms.forEach(a => { atom.set(a.id, { x: a.x, y: a.y, vx: 0, vy: 0, pinned: false }); atomData.set(a.id, a); });
  const sup = new Map();         // group id -> {x,y,vx,vy,pinned}
  const groupById = new Map(M.groups.map(g => [g.id, g]));
  const metalId = M.metal.id, hydrideId = M.metal.hydride;
  const donorSet = new Set(M.metal.donors);

  let LEVEL = "l1", running = true, showWedge = true, showCharge = true,
      showAllH = false, enforceSym = true, repScale = 0.55;

  // characteristic bond length (world units) from the initial layout
  let L0 = (function () {
    const ds = [];
    M.bonds.forEach(b => {
      const A = atom.get(b.a), B = atom.get(b.b);
      ds.push(Math.hypot(A.x - B.x, A.y - B.y));
    });
    ds.sort((p, q) => p - q);
    return ds.length ? ds[Math.floor(ds.length / 2)] || 1.5 : 1.5;
  })();

  // ----------------------------------------------------------- view graph
  let view = null;
  function nodeRadius(kind, el, label) {
    if (kind === "super") return (0.95 + 0.22 * (label ? label.length : 3)) * L0;
    if (el === "Cu") return 0.62 * L0;
    if (LABEL_ELEMENTS.has(el) || el === "H") return 0.46 * L0;
    return 0.40 * L0;
  }
  let relaxBoost = 1.0;   // annealing multiplier on repulsion during relax()
  // every drawn bond is one standard length (superatom bonds a touch longer so
  // the label has room); metal/dative/double are NOT shortened -> uniform bonds
  function lenFactor(type) {
    if (type === "super") return 1.4;      // room for the abbreviation label
    if (type === "dative") return 1.25;    // M-P coordination bonds are legitimately longer
    return 1.0;                            // everything else: one standard length
  }
  // chord of a regular n-gon between vertices k edges apart (edge = L0)
  function chord(k, n) { return L0 * Math.sin(k * Math.PI / n) / Math.sin(Math.PI / n); }

  function buildView() {
    const collapsed = LEVEL === "l1" ? M.groups.filter(g => g.level === 1)
                    : LEVEL === "l2" ? M.groups.filter(g => g.level === 2)
                    : [];
    const hidden = new Map();          // atomId -> groupId
    collapsed.forEach(g => g.atoms.forEach(a => hidden.set(a, g.id)));

    // (re)position super nodes at the centroid of their member atoms
    collapsed.forEach(g => {
      let sx = 0, sy = 0; g.atoms.forEach(a => { const p = atom.get(a); sx += p.x; sy += p.y; });
      const cx = sx / g.atoms.length, cy = sy / g.atoms.length;
      if (!sup.has(g.id)) sup.set(g.id, { x: cx, y: cy, vx: 0, vy: 0, pinned: false });
      else { const s = sup.get(g.id); if (!s.pinned) { s.x = cx; s.y = cy; s.vx = s.vy = 0; } }
    });

    const keyForAtom = id => hidden.has(id) ? "g:" + hidden.get(id) : "a:" + id;
    const nodeByKey = new Map();
    const nodes = [];
    function ensureNode(key) {
      if (nodeByKey.has(key)) return nodeByKey.get(key);
      let n;
      if (key[0] === "a") {
        const id = +key.slice(2), a = atomData.get(id), ref = atom.get(id);
        n = { key, kind: "atom", id, el: a.el, data: a, ref };
      } else {
        const gid = key.slice(2), g = groupById.get(gid), ref = sup.get(gid);
        n = { key, kind: "super", gid, label: g.label, attach: g.attach, data: g, ref };
      }
      n.r = nodeRadius(n.kind, n.el, n.label);
      nodeByKey.set(key, n); nodes.push(n); return n;
    }
    M.atoms.forEach(a => { if (!hidden.has(a.id)) ensureNode("a:" + a.id); });
    collapsed.forEach(g => ensureNode("g:" + g.id));

    // edges
    const edgeMap = new Map();
    M.bonds.forEach(b => {
      const ka = keyForAtom(b.a), kb = keyForAtom(b.b);
      if (ka === kb) return;                       // internal to a collapsed group
      const na = ensureNode(ka), nb = ensureNode(kb);
      const k = ka < kb ? ka + "|" + kb : kb + "|" + ka;
      if (edgeMap.has(k)) return;
      let type = b.type, order = b.order, wedge = b.wedge;
      const a0 = (b.a0 !== undefined ? b.a0 : b.a), b0 = (a0 === b.a ? b.b : b.a);
      if (na.kind === "super" || nb.kind === "super") { type = "super"; order = 1; wedge = "none"; }
      else if (b.type === "dative") { type = "dative"; }
      else if ((b.a === metalId && b.b === hydrideId) || (b.b === metalId && b.a === hydrideId)) type = "metalH";
      edgeMap.set(k, { na, nb, type, order, wedge, a0, b0 });
    });
    const edges = [...edgeMap.values()];

    const pk = (a, b) => a.key < b.key ? a.key + "|" + b.key : b.key + "|" + a.key;

    // ---- HARD constraints: uniform bonds + regular-polygon rings ----
    const hard = [];
    edges.forEach(e => hard.push({ a: e.na, b: e.nb, d: L0 * lenFactor(e.type) }));
    const ringPair = new Set();
    (M.rings || []).forEach(ring => {
      if (ring.some(id => hidden.has(id))) return;          // ring not fully visible
      const ns = ring.map(id => nodeByKey.get("a:" + id));
      if (ns.some(x => !x)) return;
      const n = ns.length;
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++) {
          let k = j - i; k = Math.min(k, n - k);
          hard.push({ a: ns[i], b: ns[j], d: chord(k, n) });
          ringPair.add(pk(ns[i], ns[j]));
        }
    });

    // ---- SOFT constraints: 1-3 bond angles (flex a little to de-overlap) ----
    const nbr = new Map(nodes.map(n => [n.key, []]));
    edges.forEach(e => { nbr.get(e.na.key).push(e); nbr.get(e.nb.key).push(e); });
    const angles = [];
    nodes.forEach(c => {
      if (c.kind !== "atom") return;                        // angles about real atoms
      const es = nbr.get(c.key);
      const ang = (c.data.angle_deg || 120) * Math.PI / 180;
      for (let i = 0; i < es.length; i++)
        for (let j = i + 1; j < es.length; j++) {
          const oi = es[i].na === c ? es[i].nb : es[i].na;
          const oj = es[j].na === c ? es[j].nb : es[j].na;
          if (ringPair.has(pk(oi, oj))) continue;            // ring already fixes it (hard)
          const li = L0 * lenFactor(es[i].type), lj = L0 * lenFactor(es[j].type);
          const target = Math.sqrt(li * li + lj * lj - 2 * li * lj * Math.cos(ang));
          // the metal's trigonal angles are HARD -> a clean 120 deg coordination
          if (c.id === metalId) hard.push({ a: oi, b: oj, d: target });
          else angles.push({ i: oi, j: oj, target });
        }
    });

    // excluded pairs for repulsion (anything already geometrically constrained)
    // -- but bulky abbreviation superatoms must still repel their 1-3 siblings,
    // otherwise two DTBM groups on the same P collapse onto each other.
    const excl = new Set();
    hard.forEach(c => excl.add(pk(c.a, c.b)));   // bonded/ring pairs never repel
    // 1-3 angle pairs normally don't repel, but a bulky superatom must repel its
    // siblings so two DTBM groups on one P don't collapse together.
    angles.forEach(p => { if (p.i.kind !== "super" && p.j.kind !== "super") excl.add(pk(p.i, p.j)); });

    let c0x = 0, c0y = 0; nodes.forEach(n => { c0x += n.ref.x; c0y += n.ref.y; });
    const centroid0 = { x: c0x / nodes.length, y: c0y / nodes.length };
    view = { nodes, nodeByKey, edges, hard, angles, excl, keyForAtom, pk, centroid0 };
  }

  function nodeForAtom(id) { return view.nodeByKey.get(view.keyForAtom(id)); }

  // ---------------------------------------------------------------- physics
  // Hybrid Position-Based Dynamics, strictly 2D:
  //   soft pass  -> angle springs + gentle repulsion (the "little relaxation"
  //                 that keeps things from being locked into an overlap)
  //   hard pass  -> project bond + ring constraints so bond lengths stay uniform
  //                 and every ring stays a regular polygon.
  function step(dt) {
    const nodes = view.nodes;
    nodes.forEach(n => { n.fx = 0; n.fy = 0; });

    // soft: 1-3 angle springs (flexible)
    const K13 = 0.18;
    view.angles.forEach(p => {
      const A = p.i.ref, B = p.j.ref;
      let dx = B.x - A.x, dy = B.y - A.y, d = Math.hypot(dx, dy) || 1e-6;
      const f = K13 * (d - p.target) / d;
      p.i.fx += f * dx; p.i.fy += f * dy; p.j.fx -= f * dx; p.j.fy -= f * dy;
    });

    // soft: repulsion to spread overlapping, non-bonded fragments
    // (relaxBoost anneals this high early in relax() to untangle, then eases)
    const kRep = repScale * relaxBoost * 2.6 * L0 * L0;
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i], A = a.ref;
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j], B = b.ref;
        if (view.excl.has(a.key < b.key ? a.key + "|" + b.key : b.key + "|" + a.key)) continue;
        let dx = B.x - A.x, dy = B.y - A.y, d2 = dx * dx + dy * dy;
        if (d2 < 1e-6) { dx = (i - j) * 1e-3 || 1e-3; dy = 1e-3; d2 = dx * dx + dy * dy; }
        const d = Math.sqrt(d2), minD = a.r + b.r;
        let f = kRep / d2;
        if (d < minD) f += 0.6 * (minD - d) / d;       // soft collision
        const fx = f * dx / d, fy = f * dy / d;
        a.fx -= fx; a.fy -= fy; b.fx += fx; b.fy += fy;
      }
    }

    if (enforceSym) symmetrise(0.15);

    // integrate the soft forces (damped, clamped, 2D)
    const damp = 0.82, maxV = 0.4 * L0;
    nodes.forEach(n => {
      const p = n.ref;
      if (p.pinned) { p.vx = p.vy = 0; return; }
      p.vx = (p.vx + n.fx * dt) * damp; p.vy = (p.vy + n.fy * dt) * damp;
      const v = Math.hypot(p.vx, p.vy);
      if (v > maxV) { p.vx *= maxV / v; p.vy *= maxV / v; }
      p.x += p.vx * dt; p.y += p.vy * dt;
    });

    // hard pass: project bond + ring distance constraints (Gauss-Seidel).
    // Bonds last each pass so uniform bond length wins any residual conflict.
    for (let pass = 0; pass < 18; pass++) {
      for (const c of view.hard) {
        const A = c.a.ref, B = c.b.ref;
        let dx = B.x - A.x, dy = B.y - A.y, d = Math.hypot(dx, dy) || 1e-6;
        const diff = (d - c.d) / d;
        const pa = A.pinned, pb = B.pinned;
        if (pa && pb) continue;
        const sa = pa ? 0 : (pb ? 1 : 0.5), sb = pb ? 0 : (pa ? 1 : 0.5);
        A.x += sa * diff * dx; A.y += sa * diff * dy;
        B.x -= sb * diff * dx; B.y -= sb * diff * dy;
      }
    }

    // drift-free recentring (translation only; skipped while the user drags)
    if (!nodes.some(n => n.ref.pinned)) {
      let cx = 0, cy = 0; nodes.forEach(n => { cx += n.ref.x; cy += n.ref.y; });
      cx = cx / nodes.length - view.centroid0.x; cy = cy / nodes.length - view.centroid0.y;
      nodes.forEach(n => { n.ref.x -= cx; n.ref.y -= cy; });
    }
  }

  function backboneCentroid() {
    let sx = 0, sy = 0, n = 0;
    M.atoms.forEach(a => { if (a.role === "backbone") { const p = atom.get(a.id); sx += p.x; sy += p.y; n++; } });
    return n ? { x: sx / n, y: sy / n } : { x: 0, y: 0 };
  }
  function symmetrise(k) {
    const Mt = atom.get(metalId), bc = backboneCentroid();
    let ax = bc.x - Mt.x, ay = bc.y - Mt.y; const al = Math.hypot(ax, ay) || 1; ax /= al; ay /= al;
    const reflect = (P) => {
      const vx = P.x - Mt.x, vy = P.y - Mt.y;
      const dot = vx * ax + vy * ay;
      const px = dot * ax, py = dot * ay;            // parallel comp
      const ex = vx - px, ey = vy - py;              // perpendicular
      return { x: Mt.x + px - ex, y: Mt.y + py - ey };
    };
    M.symmetry.mirror_pairs.forEach(([i, j]) => {
      const ni = nodeForAtom(i), nj = nodeForAtom(j);
      if (!ni || !nj || ni === nj) return;
      const ti = reflect(nj.ref), tj = reflect(ni.ref);
      ni.fx += k * (ti.x - ni.ref.x); ni.fy += k * (ti.y - ni.ref.y);
      nj.fx += k * (tj.x - nj.ref.x); nj.fy += k * (tj.y - nj.ref.y);
    });
  }

  function relax(n, anneal = false) {
    for (let i = 0; i < n; i++) {
      relaxBoost = anneal ? (i < n * 0.4 ? 3.0 : 1.0) : 1.0;
      step(1.0);
    }
    relaxBoost = 1.0;
  }

  // ---------------------------------------------------------------- view xform
  let scale = 40, tx = 0, ty = 0;
  function W() { return svg.clientWidth || 900; }
  function H() { return svg.clientHeight || 600; }
  function sx(x) { return tx + x * scale; }
  function sy(y) { return ty - y * scale; }
  function inv(px, py) { return { x: (px - tx) / scale, y: (ty - py) / scale }; }

  function fit() {
    const ns = view.nodes; if (!ns.length) return;
    let xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9;
    ns.forEach(n => { const p = n.ref;
      xmin = Math.min(xmin, p.x - n.r); xmax = Math.max(xmax, p.x + n.r);
      ymin = Math.min(ymin, p.y - n.r); ymax = Math.max(ymax, p.y + n.r); });
    const bw = xmax - xmin || 1, bh = ymax - ymin || 1, m = 40;
    scale = Math.min((W() - 2 * m) / bw, (H() - 2 * m) / bh);
    scale = Math.max(6, Math.min(120, scale));
    const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
    tx = W() / 2 - cx * scale; ty = H() / 2 + cy * scale;
  }

  // ---------------------------------------------------------------- rendering
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function clear() { while (svg.firstChild) svg.removeChild(svg.firstChild); }

  function render() {
    clear();
    const defs = el("defs", {}, svg);
    // arrowhead for dative bonds
    const mk = el("marker", { id: "arrow", viewBox: "0 0 10 10", refX: "9", refY: "5",
      markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" }, defs);
    el("path", { d: "M0,0 L10,5 L0,10 z", fill: "#444" }, mk);

    const gBonds = el("g", {}, svg);
    const gAtoms = el("g", {}, svg);

    // ---- bonds ----
    view.edges.forEach(e => {
      const A = e.na, B = e.nb;
      let x1 = sx(A.ref.x), y1 = sy(A.ref.y), x2 = sx(B.ref.x), y2 = sy(B.ref.y);
      let ux = x2 - x1, uy = y2 - y1; const L = Math.hypot(ux, uy) || 1; ux /= L; uy /= L;
      // inset bond ends behind labels / super boxes
      const insetA = labelInset(A), insetB = labelInset(B);
      x1 += ux * insetA; y1 += uy * insetA; x2 -= ux * insetB; y2 -= uy * insetB;
      const px = -uy, py = ux;       // perpendicular

      if (e.type === "dative") {
        // arrow from donor (P) to metal (Cu)
        let from = A, to = B;
        if (A.id === metalId) { from = B; to = A; }
        let fx = sx(from.ref.x), fy = sy(from.ref.y), txx = sx(to.ref.x), tyy = sy(to.ref.y);
        let vx = txx - fx, vy = tyy - fy; const vl = Math.hypot(vx, vy) || 1; vx /= vl; vy /= vl;
        fx += vx * labelInset(from); fy += vy * labelInset(from);
        txx -= vx * (labelInset(to) + 8); tyy -= vy * (labelInset(to) + 8);
        el("line", { x1: fx, y1: fy, x2: txx, y2: tyy, stroke: "#555",
          "stroke-width": 1.7, "stroke-dasharray": "1 0", "marker-end": "url(#arrow)" }, gBonds);
        return;
      }
      if (showWedge && (e.type !== "super") && (e.wedge === "up" || e.wedge === "down")) {
        // narrow end at a0, wide end at b0
        const narrow = (e.a0 === A.id) ? { x: x1, y: y1 } : { x: x2, y: y2 };
        const wide = (e.a0 === A.id) ? { x: x2, y: y2 } : { x: x1, y: y1 };
        const w = 5;
        if (e.wedge === "up") {
          el("polygon", { points: `${narrow.x},${narrow.y} ${wide.x + px * w},${wide.y + py * w} ${wide.x - px * w},${wide.y - py * w}`,
            fill: "#23262b", stroke: "#23262b", "stroke-width": 0.5 }, gBonds);
        } else {
          // hashed wedge
          const n = 6;
          for (let k = 1; k <= n; k++) {
            const t = k / (n + 1), w2 = w * t;
            const cx = narrow.x + (wide.x - narrow.x) * t, cy = narrow.y + (wide.y - narrow.y) * t;
            el("line", { x1: cx + px * w2, y1: cy + py * w2, x2: cx - px * w2, y2: cy - py * w2,
              stroke: "#23262b", "stroke-width": 1.2 }, gBonds);
          }
        }
        return;
      }

      const stroke = "#2b2f36", sw = 1.6;
      const drawLine = (ox, oy) => el("line", { x1: x1 + ox, y1: y1 + oy, x2: x2 + ox, y2: y2 + oy,
        stroke, "stroke-width": sw, "stroke-linecap": "round" }, gBonds);
      if (e.order === 2) {
        const o = 2.4; drawLine(px * o, py * o); drawLine(-px * o, -py * o);
      } else if (e.order === 1.5 || e.type === "aromatic") {
        drawLine(0, 0);
        // inner shorter line to read as aromatic
        const o = 3.0, s = 0.14;
        el("line", { x1: x1 + px * o + ux * L * s, y1: y1 + py * o + uy * L * s,
          x2: x2 + px * o - ux * L * s, y2: y2 + py * o - uy * L * s,
          stroke, "stroke-width": 1.1, "stroke-linecap": "round", opacity: .85 }, gBonds);
      } else {
        drawLine(0, 0);
      }
    });

    // ---- nodes ----
    view.nodes.forEach(n => {
      const X = sx(n.ref.x), Y = sy(n.ref.y);
      if (n.kind === "super") {
        const wpx = (n.label.length * 7.4 + 14), hpx = 20;
        el("rect", { x: X - wpx / 2, y: Y - hpx / 2, width: wpx, height: hpx, rx: 6,
          fill: "#fff", stroke: "#888", "stroke-width": 1.3, "data-key": n.key, class: "node" }, gAtoms);
        const t = el("text", { x: X, y: Y, "text-anchor": "middle", "dominant-baseline": "central",
          "font-size": 13, "font-weight": 600, fill: "#333", "data-key": n.key, class: "node" }, gAtoms);
        t.textContent = n.label;
        return;
      }
      const el_ = n.el;
      const labelled = LABEL_ELEMENTS.has(el_) || el_ === "Cu" || (n.id === hydrideId) || (showAllH && el_ === "H");
      // invisible hit target always present
      el("circle", { cx: X, cy: Y, r: Math.max(n.r * scale, 9), fill: "transparent",
        "data-key": n.key, class: "node hit" }, gAtoms);
      let glyphR = 6;
      if (n.id === metalId) {
        el("circle", { cx: X, cy: Y, r: 11, fill: COLORS.Cu, stroke: "#7a4317", "stroke-width": 1.5, class: "glyph" }, gAtoms);
        const t = el("text", { x: X, y: Y, "text-anchor": "middle", "dominant-baseline": "central",
          "font-size": 12, "font-weight": 700, fill: "#fff" }, gAtoms);
        t.textContent = "Cu"; glyphR = 11;
      } else if (labelled) {
        const col = COLORS[el_] || "#23262b";
        el("circle", { cx: X, cy: Y, r: 9.5, fill: "#fff", class: "glyph" }, gAtoms);   // halo
        const t = el("text", { x: X, y: Y, "text-anchor": "middle", "dominant-baseline": "central",
          "font-size": 13.5, "font-weight": 700, fill: col }, gAtoms);
        t.textContent = (n.id === hydrideId) ? "H" : el_; glyphR = 9.5;
      }
      // charges as full +/- symbols (offset by the fixed visible glyph radius)
      if (showCharge) {
        const q = (n.data.charge !== 0) ? n.data.charge : (n.data.charge_ionic || 0);
        if (q) drawCharge(gAtoms, X, Y, glyphR, q);
      }
    });
  }

  function labelInset(n) {
    if (n.kind === "super") return n.label.length * 3.7 + 8;
    if (n.id === metalId) return 12;
    if (LABEL_ELEMENTS.has(n.el) || n.id === hydrideId) return 10;
    return 1.5;
  }
  // charge as drawn line shapes (never a font glyph), geometrically centred in
  // the badge circle: a minus is one centred segment, a plus adds a centred
  // vertical segment -- both symmetric about (cx, cy).
  function drawCharge(g, X, Y, r, q) {
    const cx = X + r * 0.9 + 7, cy = Y - r * 0.9 - 5;
    const col = q > 0 ? "#c0392b" : "#2b5fcc";
    el("circle", { cx, cy, r: 7.6, fill: "#fff", stroke: col, "stroke-width": 1.4, class: "glyph" }, g);
    const a = 4.0;
    el("line", { x1: cx - a, y1: cy, x2: cx + a, y2: cy, stroke: col,
      "stroke-width": 1.9, "stroke-linecap": "round", class: "glyph" }, g);
    if (q > 0)
      el("line", { x1: cx, y1: cy - a, x2: cx, y2: cy + a, stroke: col,
        "stroke-width": 1.9, "stroke-linecap": "round", class: "glyph" }, g);
  }

  // ---------------------------------------------------------------- legend
  function legend() {
    const t = M.meta.trex || {};
    const can = t.canonical || {};
    const trexStr = (can.string || t.trex_string || "").toString();
    const lg = document.getElementById("legend");
    lg.innerHTML =
      `<b>${M.meta.name}</b><br>` +
      `metal: <b>Cu</b> · CN ${M.metal.cn} · ${M.metal.geometry_name}<br>` +
      `T-REX: <span style="font-family:monospace">${trexStr.slice(0, 40)}${trexStr.length > 40 ? "…" : ""}</span><br>` +
      `config: ${M.cip && M.cip.descriptor ? `<b>${M.cip.label}</b> axial · ${M.cip.helicity}-helical` : "—"}` +
      `<br><span class="swatch" style="background:${COLORS.P}"></span>P` +
      `<span class="swatch" style="background:${COLORS.O};margin-left:8px"></span>O` +
      `<span class="swatch" style="background:${COLORS.Cu};margin-left:8px"></span>Cu`;
  }

  // ---------------------------------------------------------------- header
  document.getElementById("title").textContent = "metaldepict — " + M.meta.name;
  document.getElementById("formula").textContent = M.meta.formula;
  document.getElementById("geom").textContent = "Cu · " + M.metal.geometry_name + " (CN " + M.metal.cn + ")";
  document.getElementById("method").textContent = "conformer: " + (M.meta.conformer_method || "");

  // ---------------------------------------------------------------- loop
  function frame() {
    if (running) { for (let s = 0; s < 3; s++) step(1.0); render(); }
    requestAnimationFrame(frame);
  }

  // ---------------------------------------------------------------- interaction
  let drag = null, pan = null;
  svg.addEventListener("pointerdown", ev => {
    const key = ev.target.getAttribute && ev.target.getAttribute("data-key");
    if (key) {
      const n = view.nodeByKey.get(key);
      // hold the node during the gesture; a plain click (no move) is reverted
      // on pointerup so it does not count as a pin -- that keeps double-click
      // free to toggle the pin cleanly.
      drag = { n, wasPinned: n.ref.pinned, moved: false };
      n.ref.pinned = true; n.ref.vx = n.ref.vy = 0;
      svg.setPointerCapture(ev.pointerId); svg.classList.add("dragging");
    } else {
      pan = { x: ev.clientX, y: ev.clientY, tx, ty };
      svg.classList.add("dragging");
    }
  });
  svg.addEventListener("pointermove", ev => {
    if (drag) {
      const rect = svg.getBoundingClientRect();
      const w = inv(ev.clientX - rect.left, ev.clientY - rect.top);
      drag.n.ref.x = w.x; drag.n.ref.y = w.y; drag.n.ref.vx = drag.n.ref.vy = 0;
      drag.moved = true;
      if (!running) render();
    } else if (pan) {
      tx = pan.tx + (ev.clientX - pan.x); ty = pan.ty + (ev.clientY - pan.y);
      if (!running) render();
    }
  });
  function endPointer() {
    if (drag && !drag.moved) drag.n.ref.pinned = drag.wasPinned;  // a click is not a pin
    drag = null; pan = null; svg.classList.remove("dragging");
  }
  svg.addEventListener("pointerup", endPointer);
  svg.addEventListener("pointercancel", endPointer);
  svg.addEventListener("dblclick", ev => {
    const key = ev.target.getAttribute && ev.target.getAttribute("data-key");
    if (key) { const n = view.nodeByKey.get(key); n.ref.pinned = !n.ref.pinned; if (!running) render(); }
  });
  svg.addEventListener("wheel", ev => {
    ev.preventDefault();
    const rect = svg.getBoundingClientRect();
    const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    const w = inv(mx, my);
    const factor = Math.exp(-ev.deltaY * 0.0012);
    scale = Math.max(5, Math.min(200, scale * factor));
    tx = mx - w.x * scale; ty = my + w.y * scale;
    if (!running) render();
  }, { passive: false });

  // ---------------------------------------------------------------- controls
  document.querySelectorAll("[data-level]").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-level]").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      LEVEL = btn.getAttribute("data-level");
      buildView(); relax(240, true); fit(); render();
    });
  });
  const playBtn = document.getElementById("btn-play");
  playBtn.addEventListener("click", () => {
    running = !running; playBtn.classList.toggle("active", running);
    playBtn.textContent = running ? "▶ run" : "❚❚ paused";
    if (!running) render();
  });
  document.getElementById("btn-relax").addEventListener("click", () => { relax(300, true); render(); });
  document.getElementById("chk-sym").addEventListener("change", e => { enforceSym = e.target.checked; });
  document.getElementById("chk-wedge").addEventListener("change", e => { showWedge = e.target.checked; render(); });
  document.getElementById("chk-charge").addEventListener("change", e => { showCharge = e.target.checked; render(); });
  document.getElementById("chk-carbonH").addEventListener("change", e => { showAllH = e.target.checked; render(); });
  document.getElementById("rng-rep").addEventListener("input", e => { repScale = e.target.value / 100; });
  document.getElementById("btn-fit").addEventListener("click", () => { fit(); render(); });
  document.getElementById("btn-reset").addEventListener("click", () => {
    M.atoms.forEach(a => { const p = atom.get(a.id); p.x = a.x; p.y = a.y; p.vx = p.vy = 0; p.pinned = false; });
    sup.forEach(s => s.pinned = false);
    buildView(); relax(240, true); fit(); render();   // relax like the initial load
  });
  document.getElementById("btn-svg").addEventListener("click", () => {
    const w = W(), h = H();
    const clone = svg.cloneNode(true);          // give the standalone file real dimensions
    clone.setAttribute("width", w); clone.setAttribute("height", h);
    clone.setAttribute("viewBox", `0 0 ${w} ${h}`);
    const s = new XMLSerializer().serializeToString(clone);
    downloadBlob(new Blob(['<?xml version="1.0" encoding="UTF-8"?>\n' + s], { type: "image/svg+xml" }), "metaldepict.svg");
  });
  document.getElementById("btn-json").addEventListener("click", () => {
    const out = { name: M.meta.name, level: LEVEL,
      atoms: M.atoms.map(a => { const p = atom.get(a.id); return { id: a.id, el: a.el, x: p.x, y: p.y, pinned: p.pinned }; }) };
    downloadBlob(new Blob([JSON.stringify(out, null, 2)], { type: "application/json" }), "metaldepict_layout.json");
  });
  function downloadBlob(blob, name) {
    const u = URL.createObjectURL(blob); const a = document.createElement("a");
    a.href = u; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(u), 1000);
  }
  window.addEventListener("resize", () => { fit(); render(); });

  // ---------------------------------------------------------------- go
  buildView(); relax(280, true); fit(); legend(); render();
  requestAnimationFrame(frame);

  // expose for headless testing / screenshots
  window.__metaldepict = {
    relax: (n) => { relax(n || 200, true); render(); },
    setLevel: (lv) => { LEVEL = lv; buildView(); relax(240, true); fit(); render(); },
    fit: () => { fit(); render(); },
    overlapScore, metrics, view: () => view, stop: () => { running = false; }
  };

  // quantitative depiction-quality metrics (consistency of bonds/angles/rings)
  function metrics() {
    const v = view, avg = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
    const cvOf = a => { const m = avg(a); return m ? Math.sqrt(avg(a.map(x => (x - m) ** 2))) / m : 0; };
    // uniformity is measured over the standard organic bonds; metal/dative
    // bonds are a separate (legitimately longer) class and abbreviation bonds
    // are not real bonds.
    const bl = v.edges.filter(e => !["super", "dative", "metalH"].includes(e.type))
      .map(e => Math.hypot(e.na.ref.x - e.nb.ref.x, e.na.ref.y - e.nb.ref.y));
    const ringEdgeCV = [], ringAngleDev = [];
    (M.rings || []).forEach(ring => {
      const ns = ring.map(id => v.nodeByKey.get("a:" + id));
      if (ns.some(x => !x)) return;
      const n = ns.length, e = [];
      for (let i = 0; i < n; i++) {
        const A = ns[i].ref, B = ns[(i + 1) % n].ref; e.push(Math.hypot(A.x - B.x, A.y - B.y));
      }
      ringEdgeCV.push(cvOf(e));
      const ideal = (n - 2) * 180 / n;
      for (let i = 0; i < n; i++) {
        const P = ns[(i - 1 + n) % n].ref, Q = ns[i].ref, R = ns[(i + 1) % n].ref;
        let ang = Math.abs((Math.atan2(P.y - Q.y, P.x - Q.x) - Math.atan2(R.y - Q.y, R.x - Q.x)) * 180 / Math.PI);
        if (ang > 180) ang = 360 - ang;
        ringAngleDev.push(Math.abs(ang - ideal));
      }
    });
    // C2 symmetry deviation: reflect each mirror atom across the
    // metal->backbone-centroid axis and measure residual to its partner (/ L0)
    let symDev = 0, sn = 0;
    const Mt = atom.get(metalId), bc = backboneCentroid();
    let axx = bc.x - Mt.x, axy = bc.y - Mt.y; const al = Math.hypot(axx, axy) || 1; axx /= al; axy /= al;
    const reflect = P => { const vx = P.x - Mt.x, vy = P.y - Mt.y, dot = vx * axx + vy * axy;
      const px = dot * axx, py = dot * axy; return { x: Mt.x + px - (vx - px), y: Mt.y + py - (vy - py) }; };
    (M.symmetry.mirror_pairs || []).forEach(([i, j]) => {
      const ni = v.nodeByKey.get("a:" + i), nj = v.nodeByKey.get("a:" + j);
      if (!ni || !nj) return;
      const r = reflect(nj.ref);
      symDev += Math.hypot(ni.ref.x - r.x, ni.ref.y - r.y); sn++;
    });
    return {
      bondLenCV: +cvOf(bl).toFixed(3),
      ringEdgeCV: +avg(ringEdgeCV).toFixed(3),
      ringAngleDevDeg: +avg(ringAngleDev).toFixed(2),
      symDev: +(sn ? symDev / sn / L0 : 0).toFixed(3),
      overlap: overlapScore(),
    };
  }

  // overlap metric: count non-bonded node pairs whose *visual* glyphs intersect
  // (use the drawn label size, not the larger repulsion radius)
  function visualR(n) {
    if (n.kind === "super") return 0.45 * L0;        // abbreviation box ~ label
    if (n.el === "Cu") return 0.34 * L0;
    if (LABEL_ELEMENTS.has(n.el) || n.id === hydrideId) return 0.30 * L0;
    return 0.16 * L0;                                 // bare vertex
  }
  function overlapScore() {
    let c = 0; const ns = view.nodes;
    for (let i = 0; i < ns.length; i++) for (let j = i + 1; j < ns.length; j++) {
      const a = ns[i], b = ns[j];
      const key = a.key < b.key ? a.key + "|" + b.key : b.key + "|" + a.key;
      if (view.excl.has(key)) continue;
      const d = Math.hypot(a.ref.x - b.ref.x, a.ref.y - b.ref.y);
      if (d < (visualR(a) + visualR(b))) c++;
    }
    return c;
  }
})();
