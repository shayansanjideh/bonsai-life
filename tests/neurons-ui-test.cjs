// Stub-model UI test for web/neurons.html: load, map neurons, generate, click a word and a neuron. No GPU needed.
// THREE_DIR=<unpacked three@0.160.0> serves three.js locally when unpkg is unreachable. SHOT_DIR=<dir> saves screenshots.
const { chromium } = require("playwright"); const { spawn } = require("child_process"); const path = require("path");
(async () => {
  const srv = spawn("python3", ["-m", "http.server", "8783", "--bind", "127.0.0.1"], { cwd: path.join(__dirname, ".."), stdio: "ignore" });
  await new Promise(r => setTimeout(r, 800));
  const b = await chromium.launch({ args: ["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist", "--no-sandbox"] });
  let fail = false;
  try {
    const p = await b.newPage({ viewport: { width: 1440, height: 1000 } }); const errs = [];
    p.on("pageerror", e => errs.push(String(e))); p.on("console", m => { if (m.type() === "error" && (process.env.THREE_DIR || !/three|unpkg/.test(m.text()))) errs.push(m.text()); });
    if (process.env.THREE_DIR) await p.route("https://unpkg.com/three@0.160.0/**", r => r.fulfill({ path: path.join(process.env.THREE_DIR, r.request().url().split("three@0.160.0/")[1]), contentType: "application/javascript" }));
    await p.goto("http://127.0.0.1:8783/web/neurons.html?stub=1");
    await p.click("#btnLoad"); await p.waitForFunction(() => !document.getElementById("btnGen").disabled, null, { timeout: 90000 }).catch(e => { throw new Error(e.message + " errors: " + errs.join(" / ") + " status: "); });
    await p.click("#btnGen"); await p.waitForFunction(() => !document.getElementById("btnGen").disabled, null, { timeout: 180000 });
    const st = await p.evaluate(() => ({ toks: cloudState.toks.length, S: cloudState.map.S, edges: cloudState.map.edges.length, live: cloudState.map.live, wm: document.getElementById("wm").width }));
    await p.click('.tok[data-i="20"]'); await p.waitForTimeout(800);
    const cur = await p.evaluate(() => document.querySelector(".tok.cur")?.dataset.i);
    if (process.env.SHOT_DIR) await p.screenshot({ path: path.join(process.env.SHOT_DIR, "neurons.png") });
    console.log("neurons UI:", JSON.stringify({ ...st, cur }), "errors:", errs.slice(0, 3));
    fail = st.toks < 20 || st.S < 100 || st.edges < 10 || cur !== "20" || errs.length > 0;
  } catch (e) { console.log("FAIL", e.message); fail = true; }
  await b.close(); srv.kill(); console.log(fail ? "FAIL neurons UI" : "PASS neurons UI"); process.exit(fail ? 1 : 0);
})();
