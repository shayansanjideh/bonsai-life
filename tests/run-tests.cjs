// Runs the GPU surgeon test and the stub UI smoke test in headless Chromium (SwiftShader WebGPU).
// Usage: node tests/run-tests.cjs   (from the repo root; needs playwright + python3)
const { chromium } = require("playwright");
const { spawn } = require("child_process");
const path = require("path");
(async () => {
  const root = path.join(__dirname, "..");
  const srv = spawn("python3", ["-m", "http.server", "8777", "--bind", "127.0.0.1"], { cwd: root, stdio: "ignore" });
  await new Promise(r => setTimeout(r, 800));
  const browser = await chromium.launch({ args: ["--enable-unsafe-swiftshader", "--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--no-sandbox"] });
  let failed = false;
  try {
    const page = await browser.newPage(); const errs = []; page.on("pageerror", e => errs.push(String(e)));
    await page.goto("http://127.0.0.1:8777/tests/surgeon-test.html");
    await page.waitForFunction(() => window.done || window.failed, null, { timeout: 120000 }).catch(() => {});
    const out = await page.evaluate(() => document.getElementById("o").textContent);
    console.log(out); if (errs.length) console.log("page errors:", errs);
    if (!/PASS/.test(out) || /FAIL/.test(out) || errs.length) failed = true;
    // Stub-model UI smoke test: every control path without a GPU model.
    const ui = await browser.newPage({ viewport: { width: 1360, height: 1000 } }); const uerr = [];
    ui.on("pageerror", e => uerr.push(String(e))); ui.on("console", m => { if (m.type() === "error") uerr.push(m.text()); });
    await ui.goto("http://127.0.0.1:8777/web/brain.html?stub=1&autoload=1");
    await ui.waitForFunction(() => document.getElementById("mLoss0").textContent.includes("nats"), null, { timeout: 20000 });
    await ui.click("#btnGen"); await ui.waitForFunction(() => document.getElementById("btnGen").disabled === false && document.querySelectorAll(".tok").length > 3, null, { timeout: 20000 });
    const n1 = await ui.locator(".tok").count();
    await ui.locator(".tok").nth(2).click(); await ui.locator("#insp .cand:not(.chosen)").first().click();
    await ui.waitForFunction(() => document.querySelector(".tok.forked") && !document.getElementById("btnGen").disabled, null, { timeout: 20000 });
    await ui.click('.cell[data-k="L0.mlp"]'); await ui.waitForTimeout(400);
    await ui.waitForFunction(() => !document.getElementById("btnGen").disabled && !document.getElementById("compareWrap").hidden, null, { timeout: 20000 });
    const off = await ui.textContent("#kOff"), loss = await ui.textContent("#kLoss");
    if (process.env.SHOT_DIR) await ui.screenshot({ path: process.env.SHOT_DIR + "/write.png" });
    await ui.click('[data-tab="scan"]').catch(() => {});
    await ui.evaluate(() => { for (const b of document.querySelectorAll(".tabs button")) b.classList.toggle("on", b.dataset.tab === "scan"); for (const p of document.querySelectorAll(".tabpane")) p.classList.toggle("on", p.id === "tab-scan"); });
    await ui.click("#btnScan"); await ui.waitForFunction(() => /done/.test(document.getElementById("scanStat").textContent), null, { timeout: 60000 });
    const bars = await ui.locator("#scanChart rect").count();
    await ui.click("#btnSweep"); await ui.waitForFunction(() => /sweep done/.test(document.getElementById("scanStat").textContent), null, { timeout: 60000 });
    const sweepRows = await ui.locator("#sweepOut tr").count();
    if (process.env.SHOT_DIR) await ui.screenshot({ path: process.env.SHOT_DIR + "/scan.png" });
    await ui.evaluate(() => { for (const b of document.querySelectorAll(".tabs button")) b.classList.toggle("on", b.dataset.tab === "game"); for (const p of document.querySelectorAll(".tabpane")) p.classList.toggle("on", p.id === "tab-game"); });
    await ui.fill("#gCustom", "cat"); await ui.dispatchEvent("#gCustom", "change");
    await ui.click("#btnGame");
    for (let i = 0; i < 40; i++) { await ui.waitForSelector("#live .cand", { timeout: 10000 }).catch(() => {}); if (await ui.locator(".win").count()) break; const c = ui.locator("#live .cand"); if (!(await c.count())) break; await c.first().click(); await ui.waitForTimeout(60); }
    const gameMsg = await ui.textContent("#gMsg"), picks = await ui.textContent("#gPicks");
    if (process.env.SHOT_DIR) await ui.screenshot({ path: process.env.SHOT_DIR + "/game.png" });
    console.log(`UI: ${n1} tokens, fork ok, ${off} off, loss ${loss}, ${bars} scan bars, ${sweepRows - 1} sweep rows, game: "${gameMsg}" picks ${picks}`);
    if (uerr.length) { console.log("UI errors:", uerr); failed = true; }
    if (bars !== 48 || sweepRows !== 9) failed = true;
  } finally { await browser.close(); srv.kill(); }
  process.exit(failed ? 1 : 0);
})();
