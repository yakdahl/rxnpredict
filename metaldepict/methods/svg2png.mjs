import { createRequire } from "module";
import path from "path";
import url from "url";
import fs from "fs";
const require = createRequire(import.meta.url);
const here = path.dirname(url.fileURLToPath(import.meta.url));
const { chromium } = require(process.env.PW_CORE || "/tmp/node_modules/playwright-core");
const chrome = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const outdir = path.resolve(here, "out");
const keys = process.argv.slice(2);
const browser = await chromium.launch({ executablePath: chrome, args: ["--no-sandbox","--disable-gpu"] });
for (const key of keys) {
  const svg = fs.readFileSync(path.join(outdir, key + ".svg"), "utf8");
  const page = await browser.newPage({ viewport: { width: 760, height: 640 }, deviceScaleFactor: 2 });
  await page.setContent(`<!doctype html><body style="margin:0;padding:0">${svg}</body>`, { waitUntil: "load" });
  await page.waitForTimeout(60);
  await page.screenshot({ path: path.join(outdir, key + ".png") });
  console.log("png", key);
  await page.close();
}
await browser.close();
