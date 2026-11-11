# Kaggle kernel: test the secret-message page on the real Bonsai model in headless Chromium.
# 1. Probe which WebGPU adapters Chrome gets (T4 via Vulkan, default, SwiftShader CPU) without loading anything.
# 2. Load the model on the best one and run short, time-capped tasks, writing summary.json after each,
#    so a killed or capped run still leaves results. Whole run is capped at ~2.5 h.
import json, os, subprocess, sys, time, traceback

URL = os.environ.get("SECRET_URL", "https://shayansanjideh.github.io/bonsai-life/secret.html")
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
DEADLINE = time.time() + float(os.environ.get("BUDGET_S", 9000))
LOG = open(os.path.join(OUT, "log.txt"), "a")
summary = {"url": URL}
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(map(str, a)); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def save(): json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
def sh(cmd):
    log("$", cmd); r = subprocess.run(cmd, shell=True, capture_output=True, text=True); log((r.stdout + r.stderr)[-1500:]); return r.returncode

if not os.environ.get("SKIP_INSTALL"):
    sh("nvidia-smi --query-gpu=name,driver_version --format=csv || true")
    sh(f"{sys.executable} -m pip install -q playwright==1.48.0")
    sh(f"{sys.executable} -m playwright install --with-deps chromium 2>&1 | tail -3")
CHROME = os.environ.get("CHROME_PATH") or None
from playwright.sync_api import sync_playwright

COMMON = ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--no-sandbox", "--enable-dawn-features=allow_unsafe_apis"]
BACKENDS = {
    "t4_vulkan": COMMON + ["--enable-features=Vulkan,WebGPU", "--use-angle=vulkan", "--use-vulkan=native", "--disable-vulkan-surface"],
    "gpu_default": COMMON + ["--enable-features=Vulkan"],
    "swiftshader": COMMON + ["--enable-features=Vulkan", "--use-webgpu-adapter=swiftshader", "--use-angle=swiftshader", "--use-vulkan=swiftshader"],
}
ADAPTER_JS = """async () => { const a = await navigator.gpu?.requestAdapter(); if (!a) return null;
  const i = a.info || (a.requestAdapterInfo ? await a.requestAdapterInfo() : {}); return {vendor: i.vendor, architecture: i.architecture, description: i.description}; }"""

def run_task(p, name, js, cap_s):
    """Start async JS in the page, wait up to cap_s, record result or TIMEOUT."""
    cap_s = min(cap_s, DEADLINE - time.time())
    if cap_s < 30: summary[name] = "SKIPPED (out of time)"; save(); return None
    t0 = time.time()
    p.evaluate(f"() => {{ window.__r = undefined; (async () => {{ {js} }})().then(r => window.__r = {{ok: r}}, e => window.__r = {{err: String(e && e.stack || e)}}); }}")
    try:
        p.wait_for_function("() => window.__r !== undefined", timeout=cap_s * 1000, polling=3000)
        r = p.evaluate("() => window.__r")
    except Exception:
        r = {"err": f"TIMEOUT after {cap_s:.0f}s", "progress": p.evaluate("() => window.__progress || null")}
    r["seconds"] = round(time.time() - t0)
    summary[name] = r; save(); log(name, json.dumps(r)[:3000]); return r

with sync_playwright() as pw:
    adapters = {}
    for name, args in BACKENDS.items():
        try:
            b = pw.chromium.launch(headless=True, executable_path=CHROME, args=args + ["--headless=new"])
            p = b.new_page(); p.goto(URL, timeout=120000)
            adapters[name] = p.evaluate(ADAPTER_JS); b.close()
        except Exception as e:
            adapters[name] = {"error": str(e)[:300]}
        log("adapter", name, adapters[name])
    summary["adapters"] = adapters; save()
    real = [n for n in ("t4_vulkan", "gpu_default") if adapters.get(n) and "error" not in adapters[n] and adapters[n].get("architecture") != "swiftshader"]
    pick = real[0] if real else ("swiftshader" if adapters.get("swiftshader") and "error" not in adapters["swiftshader"] else None)
    summary["backend"] = pick; save()
    if not pick: log("no usable WebGPU adapter"); sys.exit(0)

    b = pw.chromium.launch(headless=True, executable_path=CHROME, args=BACKENDS[pick] + ["--headless=new"])
    p = b.new_page(); p.on("pageerror", lambda e: log("pageerror:", str(e)[:300]))
    p.goto(URL, timeout=120000)
    t0 = time.time(); p.click("#wake")
    try:
        p.wait_for_function("() => /ready|Couldn't/.test(document.querySelector('#loadtxt').textContent)", timeout=1800_000, polling=3000)
    except Exception: pass
    summary["load"] = {"text": p.text_content("#loadtxt"), "seconds": round(time.time() - t0)}; save(); log("load", summary["load"])
    if "ready" not in summary["load"]["text"]: sys.exit(0)

    # Speed: 16 decode steps.
    run_task(p, "speed", """const {lm, S} = window.stegoCtx(); let lg = await lm.begin(lm.promptIds(S.TOPICS[0])); const t = performance.now();
      for (let i = 0; i < 16; i++) { window.__progress = i; lg = await lm.step(lm.encode(' the')[0]); } return {tokPerSec: 16000 / (performance.now() - t)};""", 900)
    # Fingerprint + same-device repeat (the cross-device baseline).
    run_task(p, "fingerprint", """const {lm, mask, P, L} = window.stegoCtx(); const a = await L.fingerprint(lm, mask); const b = await L.fingerprint(lm, mask);
      window.stegoReport.fingerprint = a; return {repeat: L.compareFingerprints(a, b, P), fp: a};""", 1800)
    # One real encode -> decode, generous length so we learn how many characters a secret needs.
    rt_js = """const {lm, mask, P, S} = window.stegoCtx(); const p = {...P, maxChars: 3000, maxTokens: 260}; const out = [];
      for (const [secret, topic] of SECRETS) { for (let seed = 1000; seed < 1000 + TRIES; seed++) {
        const t0 = performance.now(); const e = await S.encodeOnce(lm, mask, secret, topic, p, seed, (ids) => { window.__progress = {secret, seed, tokens: ids.length}; });
        const r = {secret, topic, seed, ok: e.ok, reason: e.reason || null, text: e.text, tokens: e.ids && e.ids.length, chars: e.text && e.text.length, st: e.st, encodeS: (performance.now() - t0) / 1000};
        if (e.ok) { const t1 = performance.now(); const N = await S.readDigits(lm, mask, lm.encode(e.text), topic, p); r.decoded = S.unpackSecret(N); r.correct = r.decoded === S.normalizeSecret(secret); r.decodeS = (performance.now() - t1) / 1000; }
        out.push(r); window.__partial = out; if (e.ok) break; } } return out;"""
    gpu = pick != "swiftshader"
    secrets = [["meet at 7", 0], ["bring snacks", 1], ["the code is 4417", 2], ["yes", 3]] if gpu else [["meet at 7", 0]]
    r = run_task(p, "roundtrip", rt_js.replace("SECRETS", json.dumps(secrets)).replace("TRIES", "3" if gpu else "2"), 3600 if gpu else 5400)
    if r and "err" in r: summary["roundtrip_partial"] = p.evaluate("() => window.__partial || null"); save()
    if gpu:
        run_task(p, "capacity", """const {lm, mask, L} = window.stegoCtx(); const nat = await L.sampleNatural(lm, mask, 6, 90, (s, i) => window.__progress = [s, i]);
          return {top: L.capacity(nat).slice(0, 8), texts: nat.map(s => s.text)};""", 2400)
    b.close()
log("done", json.dumps(summary)[:4000])
