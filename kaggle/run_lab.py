# Kaggle kernel: run the secret-message Lab headlessly on WebGPU backends and save the reports.
#   backends: NVIDIA T4 through Vulkan (real GPU) and SwiftShader (CPU software GPU).
# Two different backends on one box = a real cross-device determinism test.
# Output: /kaggle/working/report_<backend>.json and summary.json
import json, os, subprocess, sys, time, traceback

URL = os.environ.get("SECRET_URL", "https://shayansanjideh.github.io/bonsai-life/secret.html")
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
LOG = open(os.path.join(OUT, "log.txt"), "a")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(map(str, a)); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

def sh(cmd):
    log("$", cmd); r = subprocess.run(cmd, shell=True, capture_output=True, text=True); log((r.stdout + r.stderr)[-1500:]); return r.returncode

sh("nvidia-smi --query-gpu=name,driver_version --format=csv || true")
if not os.environ.get("SKIP_INSTALL"):
    sh(f"{sys.executable} -m pip install -q playwright==1.48.0")
    sh(f"{sys.executable} -m playwright install --with-deps chromium 2>&1 | tail -3")
CHROME = os.environ.get("CHROME_PATH") or None
sh("ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null; vulkaninfo --summary 2>/dev/null | head -20 || true")

from playwright.sync_api import sync_playwright

COMMON = ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan,WebGPU", "--no-sandbox",
          "--disable-vulkan-surface", "--enable-dawn-features=allow_unsafe_apis"]
BACKENDS = {
    "t4_vulkan": COMMON + ["--use-angle=vulkan", "--use-vulkan=native"],
    "gpu_default": ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--no-sandbox", "--enable-features=Vulkan"],
    "swiftshader": COMMON + ["--use-webgpu-adapter=swiftshader", "--use-angle=swiftshader", "--use-vulkan=swiftshader"],
}
# Which lab steps per backend: SwiftShader is slow, so only the fingerprint there.
STEPS = {"t4_vulkan": ["lab-fp", "lab-cap", "lab-rt"], "gpu_default": ["lab-fp", "lab-cap", "lab-rt"], "swiftshader": ["lab-fp"]}
DONE = {"lab-fp": "Fingerprint recorded", "lab-cap": "Capacity from", "lab-rt": "of 4 decoded"}

summary = {}
with sync_playwright() as pw:
    for name, args in BACKENDS.items():
        if name == "gpu_default" and "seconds" in summary.get("t4_vulkan", {}): continue
        log("=== backend", name)
        try:
            b = pw.chromium.launch(headless=True, executable_path=CHROME, args=args + ["--headless=new"])
            p = b.new_page()
            p.on("console", lambda m: log("console:", m.text[:300]))
            p.goto(URL, timeout=120000)
            info = p.evaluate("""async () => { const a = await navigator.gpu?.requestAdapter(); if (!a) return null;
              const i = a.info || (a.requestAdapterInfo ? await a.requestAdapterInfo() : {}); return {vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description, fallback: a.isFallbackAdapter}; }""")
            log("adapter:", info)
            if not info: summary[name] = {"error": "no WebGPU adapter"}; b.close(); continue
            t0 = time.time()
            p.click("#wake")
            p.wait_for_function("() => /ready|Couldn't/.test(document.querySelector('#loadtxt').textContent)", timeout=3600_000, polling=2000)
            log("load:", p.text_content("#loadtxt"), f"{time.time()-t0:.0f}s")
            if "Couldn't" in p.text_content("#loadtxt"): summary[name] = {"error": p.text_content("#loadtxt"), "adapter": info}; b.close(); continue
            p.click("[data-pane=lab]")
            for step in STEPS[name]:
                t1 = time.time(); p.click("#" + step)
                p.wait_for_function(f"() => /{DONE[step]}|Error/.test(document.querySelector('#labmeta').textContent)", timeout=4 * 3600_000, polling=5000)
                log(step, p.text_content("#labmeta"), f"{time.time()-t1:.0f}s")
                log(p.text_content("#labout")[:1500])
                rep = p.evaluate("() => JSON.parse(JSON.stringify(window.stegoReport))")
                json.dump(rep, open(os.path.join(OUT, f"report_{name}.json"), "w"))
            summary[name] = {"adapter": info, "seconds": round(time.time() - t0)}
            b.close()
        except Exception as e:
            log("FAILED", name, traceback.format_exc()[-2000:]); summary[name] = {"error": str(e)[:500]}

# Cross-backend comparison, using the page's own compare function in a plain (no-GPU) page.
reps = {n: json.load(open(os.path.join(OUT, f"report_{n}.json"))) for n in BACKENDS if os.path.exists(os.path.join(OUT, f"report_{n}.json"))}
if len(reps) >= 2 and all("fingerprint" in r for r in reps.values()):
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True, executable_path=CHROME, args=["--no-sandbox"]); p = b.new_page(); p.goto(URL)
        a, c = list(reps.values())
        summary["cross_backend"] = p.evaluate("""async ([a, c]) => { const L = await import('./stegolab.mjs'); const S = await import('./stego.mjs');
            return L.compareFingerprints(a.fingerprint, c.fingerprint, S.DEFAULTS); }""", [a, c])
        b.close()
for n, r in reps.items():
    summary.setdefault(n, {}).update({k: r.get(k) for k in ("sameDeviceRepeat", "prefillVsDecode")})
    if r.get("capacity"): summary[n]["capacityTop5"] = r["capacity"][:5]
    if r.get("roundTrips"): summary[n]["roundTrips"] = [{k: x.get(k) for k in ("secret", "correct", "reason", "tries", "tokens", "chars", "text")} for x in r["roundTrips"]]
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
log(json.dumps(summary, indent=1)[:6000])
