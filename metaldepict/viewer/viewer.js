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
      showAllH = false, enforceSym = false, repScale = 0.45;

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
    if (kind === "super") return (0.55 + 0.16 * (label ? label.length : 3)) * L0;
    if (el === "Cu") return 0.62 * L0;
    if (LABEL_ELEMENTS.has(el) || el === "H") return 0.46 * L0;
    return 0.34 * L0;
  }
  function restLength(type) {
    switch (type) {
      case "dative": return 1.55 * L0;     // P -> Cu
      case "metalH": return 1.15 * L0;     // Cu - H
      case "super":  return 1.55 * L0;
      case "double": return 0.90 * L0;
      default:       return 1.00 * L0;     // single / aromatic
    }
  }

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
      let type = b.type, order = b.order, wedge = b.wedge, a0 = b.a, b0 = b.b;
      if (na.kind === "super" || nb.kind === "super") { type = "super"; order = 1; wedge = "none"; }
      else if (b.type === "dative") { type = "dative"; }
      else if ((b.a === metalId && b.b === hydrideId) || (b.b === metalId && b.a === hydrideId)) type = "metalH";
      edgeMap.set(k, { na, nb, type, order, wedge, a0, b0 });
    });
    const edges = [...edgeMap.values()];

    // neighbour map + 1-3 angle pairs
    const nbr = new Map(nodes.map(n => [n.key, []]));
    edges.forEach(e => { nbr.get(e.na.key).push(e); nbr.get(e.nb.key).push(e); });
    const pairs13 = [];
    nodes.forEach(c => {
      const es = nbr.get(c.key);
      for (let i = 0; i < es.length; i++)
        for (let j = i + 1; j < es.length; j++) {
          const oi = es[i].na === c ? es[i].nb : es[i].na;
          const oj = es[j].na === c ? es[j].nb : es[j].na;
          // target distance: metal centre -> ideal angle; else preserve current
          let target;
          const li = restLength(es[i].type), lj = restLength(es[j].type);
          if (c.kind === "atom" && c.id === metalId) {
            const ang = 120 * Math.PI / 180;
            target = Math.sqrt(li * li + lj * lj - 2 * li * lj * Math.cos(ang));
          } else {
            target = Math.hypot(oi.ref.x - oj.ref.x, oi.ref.y - oj.ref.y);
            target = Math.max(0.7 * L0, target);
          }
          pairs13.push({ i: oi, j: oj, target });
        }
    });

    // excluded pairs for repulsion (bonded + 1-3)
    const excl = new Set();
    const pk = (a, b) => a.key < b.key ? a.key + "|" + b.key : b.key + "|" + a.key;
    edges.forEach(e => excl.add(pk(e.na, e.nb)));
    pairs13.forEach(p => excl.add(pk(p.i, p.j)));

    view = { nodes, nodeByKey, edges, pairs13, excl, keyForAtom };
  }

  function nodeForAtom(id) { return view.nodeByKey.get(view.keyForAtom(id)); }

  // ---------------------------------------------------------------- physics
  function step(dt) {
    const nodes = view.nodes;
    nodes.forEach(n => { n.fx = 0; n.fy = 0; });

    // 1-2 bond springs
    const K12 = 0.55;
    view.edges.forEach(e => {
      const A = e.na.ref, B = e.nb.ref;
      let dx = B.x - A.x, dy = B.y - A.y, d = Math.hypot(dx, dy) || 1e-6;
      const rest = restLength(e.type);
      const f = K12 * (d - rest) / d;
      const fx = f * dx, fy = f * dy;
      e.na.fx += fx; e.na.fy += fy; e.nb.fx -= fx; e.nb.fy -= fy;
    });

    // 1-3 angle springs
    const K13 = 0.22;
    view.pairs13.forEach(p => {
      const A = p.i.ref, B = p.j.ref;
      let dx = B.x - A.x, dy = B.y - A.y, d = Math.hypot(dx, dy) || 1e-6;
      const f = K13 * (d - p.target) / d;
      const fx = f * dx, fy = f * dy;
      p.i.fx += fx; p.i.fy += fy; p.j.fx -= fx; p.j.fy -= fy;
    });

    // repulsion + collision (de-overlap)
    const kRep = repScale * 2.4 * L0 * L0;
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i], A = a.ref;
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j], B = b.ref;
        const key = a.key < b.key ? a.key + "|" + b.key : b.key + "|" + a.key;
        let dx = B.x - A.x, dy = B.y - A.y, d2 = dx * dx + dy * dy;
        if (d2 < 1e-6) { dx = (Math.random() - 0.5) * 0.01; dy = (Math.random() - 0.5) * 0.01; d2 = dx * dx + dy * dy; }
        const d = Math.sqrt(d2);
        const minD = a.r + b.r;
        if (!view.excl.has(key)) {
          // soft long-range repulsion
          let f = kRep / d2;
          // hard collision when bounding circles overlap (this is what spreads fragments)
          if (d < minD) f += 0.9 * (minD - d) / d;
          const fx = f * dx / d, fy = f * dy / d;
          a.fx -= fx; a.fy -= fy; b.fx += fx; b.fy += fy;
        }
      }
    }

    // optional C2 symmetrisation across metal -> backbone-centroid axis
    if (enforceSym) symmetrise(0.12);

    // gentle centring
    let cx = 0, cy = 0; nodes.forEach(n => { cx += n.ref.x; cy += n.ref.y; });
    cx /= nodes.length; cy /= nodes.length;
    nodes.forEach(n => { n.fx -= 0.002 * (n.ref.x - cx); n.fy -= 0.002 * (n.ref.y - cy); });

    // integrate (semi-implicit, damped)
    const damp = 0.86, maxV = 0.6 * L0;
    nodes.forEach(n => {
      const p = n.ref;
      if (p.pinned) { p.vx = p.vy = 0; return; }
      p.vx = (p.vx + n.fx * dt) * damp;
      p.vy = (p.vy + n.fy * dt) * damp;
      const v = Math.hypot(p.vx, p.vy);
      if (v > maxV) { p.vx *= maxV / v; p.vy *= maxV / v; }
      p.x += p.vx * dt; p.y += p.vy * dt;
    });
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

  function relax(n) { for (let i = 0; i < n; i++) step(1.0); }

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
  function drawCharge(g, X, Y, r, q) {
    const cx = X + r * 0.9 + 6, cy = Y - r * 0.9 - 4;
    el("circle", { cx, cy, r: 7.5, fill: "#fff", stroke: q > 0 ? "#c0392b" : "#2b5fcc", "stroke-width": 1.4 }, g);
    const sign = q > 0 ? "+" : "−";
    const txt = Math.abs(q) > 1 ? Math.abs(q) + sign : sign;
    const t = el("text", { x: cx, y: cy + 0.5, "text-anchor": "middle", "dominant-baseline": "central",
      "font-size": 11, "font-weight": 800, fill: q > 0 ? "#c0392b" : "#2b5fcc" }, g);
    t.textContent = txt;
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
      `chiral: ${t.is_chiral_overall ? "yes — axial backbone" : "—"}` +
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
      buildView(); relax(60); fit(); render();
    });
  });
  const playBtn = document.getElementById("btn-play");
  playBtn.addEventListener("click", () => {
    running = !running; playBtn.classList.toggle("active", running);
    playBtn.textContent = running ? "▶ run" : "❚❚ paused";
    if (!running) render();
  });
  document.getElementById("btn-relax").addEventListener("click", () => { relax(300); render(); });
  document.getElementById("chk-sym").addEventListener("change", e => { enforceSym = e.target.checked; });
  document.getElementById("chk-wedge").addEventListener("change", e => { showWedge = e.target.checked; render(); });
  document.getElementById("chk-charge").addEventListener("change", e => { showCharge = e.target.checked; render(); });
  document.getElementById("chk-carbonH").addEventListener("change", e => { showAllH = e.target.checked; render(); });
  document.getElementById("rng-rep").addEventListener("input", e => { repScale = e.target.value / 100; });
  document.getElementById("btn-fit").addEventListener("click", () => { fit(); render(); });
  document.getElementById("btn-reset").addEventListener("click", () => {
    M.atoms.forEach(a => { const p = atom.get(a.id); p.x = a.x; p.y = a.y; p.vx = p.vy = 0; p.pinned = false; });
    sup.forEach(s => s.pinned = false);
    buildView(); relax(120); fit(); render();   // relax like the initial load
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
  buildView(); relax(120); fit(); legend(); render();
  requestAnimationFrame(frame);

  // expose for headless testing / screenshots
  window.__metaldepict = {
    relax: (n) => { relax(n || 200); render(); },
    setLevel: (lv) => { LEVEL = lv; buildView(); relax(120); fit(); render(); },
    overlapScore, view: () => view, stop: () => { running = false; }
  };

  // crude overlap metric: count node-pairs whose circles intersect (non-bonded)
  function overlapScore() {
    let c = 0; const ns = view.nodes;
    for (let i = 0; i < ns.length; i++) for (let j = i + 1; j < ns.length; j++) {
      const a = ns[i], b = ns[j];
      const key = a.key < b.key ? a.key + "|" + b.key : b.key + "|" + a.key;
      if (view.excl.has(key)) continue;
      const d = Math.hypot(a.ref.x - b.ref.x, a.ref.y - b.ref.y);
      if (d < (a.r + b.r) * 0.85) c++;
    }
    return c;
  }
})();
