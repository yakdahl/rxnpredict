/* render.mjs -- adversarial-review renderer.
 *
 *   node render.mjs <level> <out.png> <out.metrics.json>
 *
 * Loads the viewer headless, relaxes the physics to convergence, writes a
 * screenshot AND the objective depiction-quality metrics (bond-length CV, ring
 * regularity, C2 symmetry deviation, overlap). The screenshot is what the
 * vision reviewer scores; the metrics are the auto part of the rubric.
 *
 * Chromium path / playwright-core location are overridable by env:
 *   PW_CHROMIUM   (default: pre-installed at /opt/pw-browsers/chromium-1194/...)
 *   PW_CORE       (default: playwright-core resolved from node_modules)
 */
import { createRequire } from "module";
import path from "path";
import url from "url";

const require = createRequire(import.meta.url);
const here = path.dirname(url.fileURLToPath(import.meta.url));

const level = process.argv[2] || "l1";
const outPng = process.argv[3] || `/tmp/metaldepict_${level}.png`;
const outJson = process.argv[4] || `/tmp/metaldepict_${level}.metrics.json`;
const STEPS = +(process.env.PW_STEPS || 80);

const corePath = process.env.PW_CORE || "playwright-core";
const { chromium } = require(corePath);
const chromePath = process.env.PW_CHROMIUM ||
  "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const viewer = "file://" + path.resolve(here, "..", "viewer", "index.html");

const browser = await chromium.launch({ executablePath: chromePath, args: ["--no-sandbox", "--disable-gpu"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const errors = [];
page.on("pageerror", e => errors.push(e.message));
await page.goto(viewer, { waitUntil: "load" });
await page.waitForTimeout(400);
await page.evaluate(l => window.__metaldepict.setLevel(l), level);
await page.evaluate(s => { window.__metaldepict.relax(s); window.__metaldepict.fit(); }, STEPS);
await page.waitForTimeout(150);
const metrics = await page.evaluate(() => window.__metaldepict.metrics());
await page.screenshot({ path: outPng });
const fs = require("fs");
fs.writeFileSync(outJson, JSON.stringify({ level, metrics, errors }, null, 2));
console.log(`level=${level} -> ${outPng}`);
console.log(JSON.stringify(metrics));
if (errors.length) console.log("JS errors:", errors.slice(0, 5));
await browser.close();
