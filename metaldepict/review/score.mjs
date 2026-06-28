/* score.mjs -- turn metrics + vision sub-scores into one 0-100 rating.
 *
 *   node score.mjs <metrics.json> '<visionJSON>'
 *
 * visionJSON (from the image reviewer, each 0-100):
 *   { "clarity": .., "convention": .., "legibility": .. , "notes": ".." }
 *
 * Auto sub-scores are derived from the objective metrics; the two are blended
 * by the weights below. The percentile bar is calibrated from published
 * metal-bisphosphine depictions (see rubric.md).
 */
import fs from "fs";

const clamp = (x, lo = 0, hi = 100) => Math.max(lo, Math.min(hi, x));
const m = JSON.parse(fs.readFileSync(process.argv[2], "utf8")).metrics;
const vision = JSON.parse(process.argv[3] || "{}");

// ---- auto sub-scores from objective metrics ----
const auto = {
  bondUniformity: clamp(100 - 650 * m.bondLenCV),       // CV 0.05->68, 0.10->35
  ringRegularity: clamp(100 - 22 * m.ringAngleDevDeg),  // 0.5deg->89, 2deg->56
  symmetry:       clamp(100 - 25 * m.symDev),           // symDev 1.0->75, 2.0->50
  overlapAuto:    clamp(100 - 12 * m.overlap),          // coarse; vision dominates clarity
};

// ---- weights (sum = 1.0) ----
const W = {
  bondUniformity: 0.14,
  ringRegularity: 0.16,
  symmetry:       0.10,
  overlapAuto:    0.05,
  clarity:        0.25,   // vision: reads cleanly, no tangles
  convention:     0.18,   // vision: looks like a publication figure
  legibility:     0.12,   // vision: labels/abbreviations/charges legible
};

const parts = {
  bondUniformity: auto.bondUniformity,
  ringRegularity: auto.ringRegularity,
  symmetry: auto.symmetry,
  overlapAuto: auto.overlapAuto,
  clarity: vision.clarity ?? null,
  convention: vision.convention ?? null,
  legibility: vision.legibility ?? null,
};

let total = 0, wsum = 0;
for (const k in W) if (parts[k] != null) { total += W[k] * parts[k]; wsum += W[k]; }
const score = wsum ? +(total / wsum).toFixed(1) : null;

const band = s => s >= 85 ? "top 5% (publication-grade)"
  : s >= 75 ? "top ~15%"
  : s >= 65 ? "top ~30%"
  : s >= 50 ? "around median"
  : s >= 30 ? "below median"
  : "bottom ~5%";

console.log(JSON.stringify({ parts, weights: W, score, band: score == null ? null : band(score),
  metrics: m, visionNotes: vision.notes || "" }, null, 2));
