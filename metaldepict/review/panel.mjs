import { createRequire } from "module"; import path from "path"; import url from "url";
const require = createRequire(import.meta.url);
const here = path.dirname(url.fileURLToPath(import.meta.url));
const { chromium } = require(process.env.PW_CORE || "playwright-core");
const chrome = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const viewer = "file://" + path.resolve(here, "..", "viewer", "index.html");
const ligs = process.argv.slice(2);
const browser = await chromium.launch({ executablePath: chrome, args: ["--no-sandbox","--disable-gpu"] });
for (const key of ligs) {
  const page = await browser.newPage({ viewport: { width: 760, height: 640 } });
  await page.goto(viewer + "?mol=" + key, { waitUntil: "load" });
  await page.waitForTimeout(400);
  await page.evaluate(() => { window.__metaldepict.setLevel("l1"); window.__metaldepict.relax(280); window.__metaldepict.fit(); });
  // hide UI chrome for a clean panel; draw the molecule name
  await page.evaluate(() => {
    document.querySelector("header").style.display = "none";
    document.querySelector(".controls").style.display = "none";
    document.querySelector(".legend").style.display = "none";
    document.querySelector(".hint").style.display = "none";
    const nm = (window.MOLECULE && window.MOLECULE.meta && window.MOLECULE.meta.name) || "";
    const t = document.createElement("div");
    t.textContent = nm; t.style.cssText = "position:absolute;left:10px;top:8px;font:700 16px sans-serif;color:#222;z-index:9";
    document.getElementById("stage").appendChild(t);
    window.__metaldepict.fit();
  });
  await page.evaluate(() => window.__metaldepict.relax(2));
  await page.waitForTimeout(120);
  await page.screenshot({ path: "/tmp/panel_" + key + ".png" });
  console.log("rendered", key);
  await page.close();
}
await browser.close();
