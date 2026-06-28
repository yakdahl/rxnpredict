# Adversarial image review — how it works, and the run log

This is the loop that drove `metaldepict` from a "bottom 5%" depiction to a
publication-leaning one. It is an **agent-in-the-loop** review: a vision model
scores a rendered screenshot against published conventions (gathered by web
search) plus objective geometry metrics, then the worst sub-scores become the
next code change. Below is the exact process, then the run log.

## The process, step by step

**0. Target & references (web search).**
Search for the molecule and close analogues to (a) confirm connectivity/stereo
and (b) establish what a *good* drawing looks like. For this target the searches
covered DTBM-SEGPhos, SEGPhos, BINAP and (R)-(DTBM-SEGPHOS)CuH, which establish:
C₂-symmetric axially-chiral biaryl bisphosphine; rings drawn as regular
hexagons/pentagons; one standard bond length; wedge/hash for the axis; `t-Bu` /
`MeO` abbreviations; biaryl dihedral ≈ 65°. These conventions calibrate the
rubric's percentile bands (see `rubric.md`).
> Sandbox note: image *hosts* are blocked by egress policy here, so reference
> *pixels* can't be fetched — the standard is grounded in the web-derived
> conventions above + the vision model's prior for journal figures. In an
> open-network run the loop would also fetch and visually diff reference images.

**1. Render.** `render.mjs` loads the viewer headless (pre-installed Chromium),
relaxes the 2D physics to convergence, and writes a screenshot **and** objective
metrics (`metrics()` in the viewer): bond-length CV, ring-edge CV, ring internal
angle deviation, C₂ symmetry deviation, and a visual-overlap count.

**2. Auto-score.** `score.mjs` maps each metric to a 0–100 sub-score
(`rubric.md`).

**3. Vision review.** The vision model *reads the PNG* and returns
`clarity`, `convention`, `legibility` (0–100) with notes on concrete defects —
this is the "adversarial image review": it is trying to find what makes the
drawing look unlike a journal figure.

**4. Composite & band.** `score.mjs` blends auto + vision sub-scores by the
rubric weights into one 0–100 score and a percentile band.

**5. Iterate.** Take the lowest sub-scores and the vision notes, make the
smallest code change that fixes them, re-render, re-score. Stop when the default
(l1) and collapsed (l2) views reach the top band, or when the score plateaus
(then report the residual issue rather than over-fitting).

To run a cycle:
```bash
npm i -g playwright-core            # or set PW_CORE to an existing install
cd metaldepict/review
node render.mjs l2 out.png out.json # render + metrics
node score.mjs out.json '{"clarity":86,"convention":87,"legibility":88}'
```

## Run log (what each iteration changed and the resulting l2 score)

| # | change driven by the review | key metric move | l2 score |
|---|------------------------------|-----------------|----------|
| 0 | soft-spring layout (original) | rings distorted, bonds non-uniform, tangled | **bottom ~5%** |
| 1 | hard ring + bond constraints (PBD), soft angles, 2D repulsion | ringAngleDev 12°→1°, bondCV(organic) →0.03 | ~68 |
| 2 | annealed repulsion + superatoms repel siblings | DTBM groups separate; overlap resolved | ~74 |
| 3 | charges drawn as centred line shapes; axial CIP (aS/M) + wedges | legibility ↑, stereo present | ~77 |
| 4 | metal angles **hard** (clean 120° trigonal Cu); C₂ on by default | metal centre clean; symDev 3.0→1.8 | ~80 |
| 5 | bond-CV measured over organic bonds; fair visual-overlap metric | bondUniformity 56→87; overlap 20→2 | **83.2** |

### Current ratings (this commit)
| view | score | band | weakest sub-scores |
|------|-------|------|--------------------|
| l2 (DTBM collapsed) | **83.2** | top ~15% | symmetry 54 |
| l1 (t-Bu/OMe, default) | **75** | top ~15% | clarity (left benzodioxole near centre) |
| full (all atoms) | **64** | around median | clarity (central backbone congestion) |

Objective metrics at l2: bond-length CV **0.02**, ring angle deviation **0.6°**,
overlap **2**, C₂ deviation 1.8.

### Residual gap to top-5% (≥85) and next iterations
1. **Symmetry (biggest lever).** C₂ is currently a soft nudge on the 10 backbone
   mirror atoms. Building a genuinely C₂-symmetric layout (lay out one half,
   reflect it, place Cu on the axis) would lift `symmetry` from ~54 toward ~90
   and pull l2 over 85.
2. **Full-view backbone congestion.** The P–Cu–P + biaryl macrocycle is drawn
   folded; a template placement for that ring would clear the centre.
3. **Hashed-wedge polish** near the P-bearing carbon.

The harness is re-runnable, so each of these is a measurable next step.
