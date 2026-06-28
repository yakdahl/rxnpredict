# Depiction-quality rubric (adversarial image review)

The reviewer rates a rendered depiction 0–100 by blending **objective metrics**
(computed from the laid-out geometry) with **vision sub-scores** (a vision model
reads the screenshot and judges it against published depictions of the same /
similar molecules, gathered by web search — BINAP, SEGPhos, metal–bisphosphine
chelates, (R)-(DTBM-SEGPHOS)CuH).

## Objective sub-scores (auto, from `metrics()`)
| sub-score | metric | mapping | weight |
|-----------|--------|---------|--------|
| bond uniformity | `bondLenCV` (coeff. of variation of bond lengths) | `100 − 650·CV` | 0.14 |
| ring regularity | `ringAngleDevDeg` (mean | internal angle − ideal |) | `100 − 22·dev` | 0.16 |
| symmetry | `symDev` (C₂ reflection residual / L0) | `100 − 90·symDev` | 0.10 |
| overlap (coarse) | `overlap` (intersecting non-bonded node pairs) | `100 − 7·n` | 0.05 |

## Vision sub-scores (image reviewer, 0–100)
| sub-score | what it judges | weight |
|-----------|----------------|--------|
| clarity | reads cleanly; no tangled/overlapping fragments or crossing bonds | 0.25 |
| convention | regular rings, ~120° angles, standard bond length, wedge stereo, abbreviations — i.e. looks like a journal figure | 0.18 |
| legibility | element labels, abbreviations and charge symbols readable and unambiguous | 0.12 |

## Percentile bands (calibrated to published figures)
- **≥ 85** — top 5%, publication-grade
- 75–84 — top ~15%
- 65–74 — top ~30%
- 50–64 — around median
- 30–49 — below median
- **≤ 29** — bottom ~5%

A clean ChemDraw-style figure of a metal–bisphosphine (regular rings, uniform
bonds, clear `t-Bu`/`MeO` abbreviations, axial wedge, C₂-symmetric layout) sets
the ~85–95 bar. The reviewer iterates the depiction workflow until the **default
(l1) and collapsed (l2) views reach ≥ 85** and reports residual issues for the
busiest (full) view.

## What "uses web search" means here
The reviewer runs web searches to (1) confirm the correct connectivity/【stereo
of the target and close analogues and (2) establish the drawing conventions the
band thresholds encode. In a fully-networked run it would additionally fetch
reference images and visually diff them; in this sandbox image hosts are blocked
by egress policy, so reference *pixels* are unavailable and the standard is
grounded in the web-derived conventions + the vision model's trained prior for
publication-quality structures.
