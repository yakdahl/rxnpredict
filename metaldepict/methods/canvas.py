#!/usr/bin/env python3
"""
canvas.py -- export the relaxed complexes to JSON and emit a single self-contained
HTML page where each depiction can be DRAGGED and ROTATED by hand.

`scene_to_dict(H, name)` serialises a relaxed Harness (atoms, bonds, ring circles,
adjacency, the metal id) to a plain dict.  `write_canvas(list_of_dicts, path)`
writes a standalone editor (no build step, no server, no dependencies): open the
.html, pick a complex, drag an atom to move it (bonds follow), or Shift-drag to
rotate that atom's whole sub-tree about the metal, then download the adjusted SVG
or the JSON.  This is the "ship an online canvas to adjust the complexes" piece.
"""
from __future__ import annotations

import json


def scene_to_dict(H, name):
    sc = H.scene
    atoms = [{"id": i, "x": round(H.pos[i][0], 4), "y": round(H.pos[i][1], 4),
              "label": H.label.get(i, a.label), "color": a.color,
              "fs": a.fontscale, "halo": bool(a.halo)}
             for i, a in sc.atoms.items()]
    bonds = [{"a": b.a, "b": b.b, "order": b.order, "kind": b.kind,
              "width": b.width,
              "inside": [round(b.inside[0], 4), round(b.inside[1], 4)] if b.inside else None}
             for b in sc.bonds]
    circles = [{"ids": list(v), "rfrac": r} for (v, r) in sc.ring_circles]
    adj = {}
    for b in sc.bonds:
        adj.setdefault(b.a, []).append(b.b)
        adj.setdefault(b.b, []).append(b.a)
    return {"name": name, "metal": H.metal, "atoms": atoms, "bonds": bonds,
            "circles": circles, "adj": {str(k): v for k, v in adj.items()}}


def write_canvas(complexes, path):
    data = json.dumps(complexes)
    html = _HTML.replace("/*__DATA__*/", data)
    with open(path, "w") as f:
        f.write(html)


_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>metaldepict canvas</title>
<style>
  body{font:14px system-ui,sans-serif;margin:0;background:#fafafa;color:#222}
  header{padding:10px 16px;background:#fff;border-bottom:1px solid #ddd;display:flex;
         gap:14px;align-items:center;flex-wrap:wrap}
  select,button{font:13px system-ui;padding:5px 9px;border:1px solid #bbb;border-radius:6px;
         background:#fff;cursor:pointer}
  button:hover{background:#f0f0f0}
  #wrap{display:flex;justify-content:center;padding:18px}
  svg{background:#fff;border:1px solid #e2e2e2;border-radius:8px;touch-action:none}
  .hint{color:#888;font-size:12px}
  .at{cursor:grab}
</style></head><body>
<header>
  <b>metaldepict</b>
  <select id="pick"></select>
  <button id="reset">Reset</button>
  <button id="svg">Download SVG</button>
  <button id="json">Download JSON</button>
  <span class="hint">drag an atom to move &middot; Shift-drag to rotate its group about the metal &middot; Alt-drag rotates the whole molecule</span>
</header>
<div id="wrap"><svg id="cv" width="760" height="640"></svg></div>
<script>
const DATA = /*__DATA__*/;
const SVGNS="http://www.w3.org/2000/svg";
const L=44, BW=2.4*1.0;            // px per world unit is set by fit()
let cur=null, P={}, fit_={s:1,ox:0,oy:0};
const cv=document.getElementById("cv");
const pick=document.getElementById("pick");
DATA.forEach((c,i)=>{const o=document.createElement("option");o.value=i;o.textContent=c.name;pick.appendChild(o);});

function load(i){
  cur=JSON.parse(JSON.stringify(DATA[i]));   // deep copy so Reset works
  P={}; cur.atoms.forEach(a=>P[a.id]=[a.x,a.y]);
  cur.amap={}; cur.atoms.forEach(a=>cur.amap[a.id]=a);
  render();
}
function fit(){
  const xs=cur.atoms.map(a=>P[a.id][0]), ys=cur.atoms.map(a=>P[a.id][1]);
  const pad=1.0, minx=Math.min(...xs)-pad,maxx=Math.max(...xs)+pad,
        miny=Math.min(...ys)-pad,maxy=Math.max(...ys)+pad;
  const W=cv.width.baseVal.value, H=cv.height.baseVal.value;
  const s=Math.min(W/(maxx-minx),H/(maxy-miny));
  fit_={s, ox:(W-(maxx-minx)*s)/2 - minx*s, oy:miny, maxy};
  fit_.tx=p=>[fit_.ox+p[0]*s, H-((p[1]-miny)*s+(H-(maxy-miny)*s)/2)];
}
function el(n,at){const e=document.createElementNS(SVGNS,n);for(const k in at)e.setAttribute(k,at[k]);return e;}
function unit(a,b){const d=[b[0]-a[0],b[1]-a[1]];const n=Math.hypot(d[0],d[1])||1;return [d[0]/n,d[1]/n];}
function trim(pa,pb,la,lb){const d=unit(pa,pb);
  const ta=la?0.34+0.16*Math.max(0,la.length-1):0, tb=lb?0.34+0.16*Math.max(0,lb.length-1):0;
  return [[pa[0]+d[0]*ta,pa[1]+d[1]*ta],[pb[0]-d[0]*tb,pb[1]-d[1]*tb]];}
function render(){
  fit(); while(cv.firstChild)cv.removeChild(cv.firstChild);
  const tx=fit_.tx, s=fit_.s, bw=0.055*L*s/ (44/ s) ; const lw=Math.max(1.6,0.05*s);
  // bonds
  cur.bonds.forEach(b=>{
    const A=P[b.a],B=P[b.b],la=cur.amap[b.a].label,lb=cur.amap[b.b].label;
    const [A2,B2]=trim(A,B,la,lb); let pa=tx(A2),pb=tx(B2);
    const d=unit(pa,pb), pr=[-d[1],d[0]];
    if(b.kind==="wedge"){const w=(b.width||0.34)*s;
      cv.appendChild(el("polygon",{points:`${pa[0]},${pa[1]} ${pb[0]+pr[0]*w/2},${pb[1]+pr[1]*w/2} ${pb[0]-pr[0]*w/2},${pb[1]-pr[1]*w/2}`,fill:"#111"}));}
    else if(b.kind==="taper"){const w=(b.width||0.17*L)*s, n=lw;
      cv.appendChild(el("polygon",{points:`${pa[0]+pr[0]*n/2},${pa[1]+pr[1]*n/2} ${pb[0]+pr[0]*w/2},${pb[1]+pr[1]*w/2} ${pb[0]-pr[0]*w/2},${pb[1]-pr[1]*w/2} ${pa[0]-pr[0]*n/2},${pa[1]-pr[1]*n/2}`,fill:"#111"}));}
    else if(b.kind==="dash"){const n=6;for(let k=1;k<=n;k++){const t=k/(n+0.5);
      const cx=pa[0]+(pb[0]-pa[0])*t, cy=pa[1]+(pb[1]-pa[1])*t, w=0.34*s*t/2;
      cv.appendChild(el("line",{x1:cx+pr[0]*w,y1:cy+pr[1]*w,x2:cx-pr[0]*w,y2:cy-pr[1]*w,stroke:"#111","stroke-width":lw*0.75}));}}
    else if(b.kind==="coord"){cv.appendChild(el("line",{x1:pa[0],y1:pa[1],x2:pb[0],y2:pb[1],stroke:"#111","stroke-width":lw,"stroke-dasharray":`${0.16*L*s/(44/s)*0+5},4`}));}
    else if(b.kind==="bold"){cv.appendChild(el("line",{x1:pa[0],y1:pa[1],x2:pb[0],y2:pb[1],stroke:"#111","stroke-width":lw*1.9}));}
    else{cv.appendChild(el("line",{x1:pa[0],y1:pa[1],x2:pb[0],y2:pb[1],stroke:"#111","stroke-width":lw}));
      if(b.order===2){const ins=b.inside?tx(b.inside):null; let p=pr.slice();
        if(ins){const mid=[(pa[0]+pb[0])/2,(pa[1]+pb[1])/2],to=unit(mid,ins);if(to[0]*p[0]+to[1]*p[1]<0)p=[-p[0],-p[1]];}
        const g=0.13*s, sh=0.16*s;
        cv.appendChild(el("line",{x1:pa[0]+p[0]*g+d[0]*sh,y1:pa[1]+p[1]*g+d[1]*sh,x2:pb[0]+p[0]*g-d[0]*sh,y2:pb[1]+p[1]*g-d[1]*sh,stroke:"#111","stroke-width":lw}));}}
  });
  // ring circles (ellipse via principal axes)
  cur.circles.forEach(c=>{const pts=c.ids.map(i=>P[i]);const n=pts.length;
    let cx=0,cy=0;pts.forEach(p=>{cx+=p[0];cy+=p[1];});cx/=n;cy/=n;
    let sxx=0,syy=0,sxy=0;pts.forEach(p=>{const dx=p[0]-cx,dy=p[1]-cy;sxx+=dx*dx;syy+=dy*dy;sxy+=dx*dy;});
    sxx/=n;syy/=n;sxy/=n;const tr=sxx+syy,det=sxx*syy-sxy*sxy;
    const l1=tr/2+Math.sqrt(Math.max(0,tr*tr/4-det)),l2=tr/2-Math.sqrt(Math.max(0,tr*tr/4-det));
    const ang=Math.atan2(l1-sxx,sxy)*180/Math.PI;
    const C=fit_.tx([cx,cy]);
    cv.appendChild(el("ellipse",{cx:C[0],cy:C[1],rx:Math.sqrt(l1)*1.16*fit_.s*c.rfrac,ry:Math.sqrt(Math.max(l2,0.01))*1.16*fit_.s*c.rfrac,
      transform:`rotate(${-ang} ${C[0]} ${C[1]})`,fill:"none",stroke:"#111","stroke-width":lw}));});
  // atoms
  cur.atoms.forEach(a=>{const p=fit_.tx(P[a.id]);
    if(a.label){const fs=15*(a.fs||1);
      const t=el("text",{x:p[0],y:p[1],"text-anchor":"middle","dominant-baseline":"central",
        "font-family":"system-ui,Arial","font-weight":"700","font-size":fs,fill:a.color||"#111"});
      t.textContent=a.label;
      const halo=el("text",{x:p[0],y:p[1],"text-anchor":"middle","dominant-baseline":"central",
        "font-family":"system-ui,Arial","font-weight":"700","font-size":fs,fill:"none",stroke:"#fff","stroke-width":4.5});
      halo.textContent=a.label; cv.appendChild(halo); cv.appendChild(t);}
    const hit=el("circle",{cx:p[0],cy:p[1],r:11,fill:"transparent",class:"at"});
    hit.dataset.id=a.id; cv.appendChild(hit);});
}
// ---- interaction ----
function comp(id){const seen=new Set([id]),st=[id],m=cur.metal;
  while(st.length){const x=st.pop();(cur.adj[String(x)]||[]).forEach(y=>{if(y!==m&&!seen.has(y)){seen.add(y);st.push(y);}});}
  return seen;}
let drag=null;
function world(ev){const r=cv.getBoundingClientRect();const px=ev.clientX-r.left,py=ev.clientY-r.top;
  // invert fit_.tx
  const H=cv.height.baseVal.value, s=fit_.s, miny=fit_.oy;
  const wx=(px-fit_.ox)/s; const wy=miny+((H-py)-(H-(fit_.maxy-miny)*s)/2)/s; return [wx,wy];}
cv.addEventListener("pointerdown",ev=>{const t=ev.target;if(!t.dataset||t.dataset.id===undefined)return;
  const id=+t.dataset.id; const w=world(ev);
  drag={id,start:w,mode:ev.shiftKey?"rot":ev.altKey?"rotall":"move",
        comp:ev.shiftKey?comp(id):(ev.altKey?new Set(cur.atoms.map(a=>a.id)):new Set([id])),
        orig:{}};
  drag.comp.forEach(i=>drag.orig[i]=P[i].slice());
  cv.setPointerCapture(ev.pointerId);});
cv.addEventListener("pointermove",ev=>{if(!drag)return;const w=world(ev);
  if(drag.mode==="move"){P[drag.id]=[drag.orig[drag.id][0]+(w[0]-drag.start[0]),drag.orig[drag.id][1]+(w[1]-drag.start[1])];}
  else{const piv=drag.mode==="rot"?P[cur.metal]:centroid(drag.comp,drag.orig);
    const a0=Math.atan2(drag.start[1]-piv[1],drag.start[0]-piv[0]),a1=Math.atan2(w[1]-piv[1],w[0]-piv[0]);
    const dth=a1-a0,c=Math.cos(dth),s=Math.sin(dth);
    drag.comp.forEach(i=>{const x=drag.orig[i][0]-piv[0],y=drag.orig[i][1]-piv[1];P[i]=[piv[0]+c*x-s*y,piv[1]+s*x+c*y];});}
  render();});
function centroid(set,src){let x=0,y=0,n=0;set.forEach(i=>{x+=src[i][0];y+=src[i][1];n++;});return[x/n,y/n];}
cv.addEventListener("pointerup",()=>{drag=null;});
document.getElementById("reset").onclick=()=>load(+pick.value);
pick.onchange=()=>load(+pick.value);
document.getElementById("svg").onclick=()=>{const blob=new Blob([new XMLSerializer().serializeToString(cv)],{type:"image/svg+xml"});dl(blob,cur.name+".svg");};
document.getElementById("json").onclick=()=>{const out={...cur,atoms:cur.atoms.map(a=>({...a,x:P[a.id][0],y:P[a.id][1]}))};
  dl(new Blob([JSON.stringify(out,null,1)],{type:"application/json"}),cur.name+".json");};
function dl(blob,name){const u=URL.createObjectURL(blob);const a=document.createElement("a");a.href=u;a.download=name;a.click();URL.revokeObjectURL(u);}
load(0);
</script></body></html>
"""
