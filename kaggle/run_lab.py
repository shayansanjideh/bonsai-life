# Kaggle kernel: run the secret-message Lab headlessly on WebGPU backends and save the reports.
#   backends: NVIDIA T4 through Vulkan (real GPU) and SwiftShader (CPU software GPU).
# Two different backends on one box = a real cross-device determinism test.
# Output: /kaggle/working/report_<backend>.json, summary.json, log.txt
# Design rules (runs used to hang for hours with no output):
#   - probe every backend's WebGPU adapter first (cheap), record it, then decide what to run
#   - hard overall budget (BUDGET_MIN, default 45) and per-step caps; a TIMEOUT is logged, never hangs
#   - summary.json + reports are rewritten after every step, so a killed run still leaves results
import json, os, subprocess, sys, time, traceback

T_START = time.time()
BUDGET = float(os.environ.get("BUDGET_MIN", "45")) * 60
URL = os.environ.get("SECRET_URL", "https://shayansanjideh.github.io/bonsai-life/secret.html")
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
LOG = open(os.path.join(OUT, "log.txt"), "a")
def log(*a):
    s = time.strftime("%H:%M:%S ") + f"[{(time.time()-T_START)/60:5.1f}m] " + " ".join(map(str, a)); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def left(): return BUDGET - (time.time() - T_START)
def cap(sec): return max(5_000, int(min(sec, left() - 60) * 1000))  # ms, always leave a minute to write results

def sh(cmd, t=600):
    log("$", cmd)
    try: r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t); log((r.stdout + r.stderr)[-1500:]); return r.returncode
    except subprocess.TimeoutExpired: log("TIMEOUT", cmd); return -1

summary = {"budget_min": BUDGET / 60, "probe": {}}
def save():
    summary["elapsed_min"] = round((time.time() - T_START) / 60, 1)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

sh("nvidia-smi --query-gpu=name,driver_version --format=csv || true")
sh("echo NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES; ls /usr/lib/x86_64-linux-gnu | grep -i -E 'nvidia|GLX' | head -20")
if not os.environ.get("SKIP_INSTALL"):
    sh("apt-get -qq update >/dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get -qq install -y libvulkan1 vulkan-tools >/dev/null 2>&1; echo apt done")
    # NVIDIA containers often ship the driver libs but no Vulkan ICD manifest, so Vulkan sees no GPU.
    if not any(os.path.exists(p) for p in ("/usr/share/vulkan/icd.d/nvidia_icd.json", "/etc/vulkan/icd.d/nvidia_icd.json")):
        os.makedirs("/etc/vulkan/icd.d", exist_ok=True)
        json.dump({"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0"}},
                  open("/etc/vulkan/icd.d/nvidia_icd.json", "w"))
        log("wrote /etc/vulkan/icd.d/nvidia_icd.json")
    sh(f"{sys.executable} -m pip install -q playwright==1.48.0")
    sh(f"{sys.executable} -m playwright install --with-deps chromium 2>&1 | tail -3")
CHROME = os.environ.get("CHROME_PATH") or None
sh("ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null; vulkaninfo --summary 2>&1 | head -30 || true", t=60)

from playwright.sync_api import sync_playwright

COMMON = ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan,WebGPU", "--no-sandbox",
          "--disable-vulkan-surface", "--enable-dawn-features=allow_unsafe_apis"]
BACKENDS = {
    "t4_vulkan": COMMON + ["--use-angle=vulkan", "--use-vulkan=native"],
    "gpu_default": ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--no-sandbox", "--enable-features=Vulkan"],
    "swiftshader": COMMON + ["--use-webgpu-adapter=swiftshader", "--use-angle=swiftshader", "--use-vulkan=swiftshader"],
}
DONE = {"lab-fp": "Fingerprint recorded", "lab-cap": "Capacity from", "lab-rt": "of 4 decoded"}
LOAD_CAP = {"gpu": 15 * 60, "cpu": 20 * 60}
STEP_CAP = {"gpu": 12 * 60, "cpu": 15 * 60}
ADAPTER_JS = """async () => { if (!navigator.gpu) return {error: 'navigator.gpu missing'}; const a = await navigator.gpu.requestAdapter(); if (!a) return null;
  const i = a.info || (a.requestAdapterInfo ? await a.requestAdapterInfo() : {}); return {vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description, fallback: a.isFallbackAdapter}; }"""

def is_soft(info):
    s = json.dumps(info or {}).lower()
    return (not info) or "error" in info or info.get("fallback") or "swiftshader" in s or "llvmpipe" in s

def open_page(pw, name):
    b = pw.chromium.launch(headless=True, executable_path=CHROME, args=BACKENDS[name] + ["--headless=new"])
    p = b.new_page(); p.on("console", lambda m: log("console:", m.text[:300]))
    p.goto(URL, timeout=120_000)
    return b, p

with sync_playwright() as pw:
    # 1) Probe: which adapter does each backend really get? ~1 min total.
    for name in BACKENDS:
        try:
            b, p = open_page(pw, name); info = p.evaluate(ADAPTER_JS); b.close()
        except Exception as e:
            info = {"error": str(e)[:300]}
        summary["probe"][name] = info; log("probe", name, info)
    save()
    real = [n for n in ("t4_vulkan", "gpu_default") if not is_soft(summary["probe"][n])]
    summary["real_gpu_backend"] = real[0] if real else None
    if not real: log("NO REAL GPU ADAPTER: Chrome on this box only gets a software/none adapter. Running fingerprint on SwiftShader only.")
    plan = ([(real[0], "gpu", ["lab-fp", "lab-cap", "lab-rt"])] if real else []) + [("swiftshader", "cpu", ["lab-fp"])]
    save()

    # 2) Run the lab, bounded.
    for name, kind, steps in plan:
        if left() < 5 * 60: log("skip", name, "- out of budget"); summary[name] = {"skipped": "budget"}; save(); continue
        if not summary["probe"].get(name) or "error" in summary["probe"][name]: summary[name] = {"skipped": "no adapter"}; save(); continue
        log("=== backend", name, steps); rec = summary[name] = {"adapter": summary["probe"][name], "steps": {}}
        b = None
        try:
            b, p = open_page(pw, name)
            t0 = time.time(); p.click("#wake")
            try:
                p.wait_for_function("() => /ready|Couldn't/.test(document.querySelector('#loadtxt').textContent)", timeout=cap(LOAD_CAP[kind]), polling=2000)
            except Exception:
                rec["error"] = f"TIMEOUT loading model after {time.time()-t0:.0f}s: " + p.text_content("#loadtxt")[:200]; log(rec["error"]); save(); b.close(); continue
            rec["load_s"] = round(time.time() - t0); log("load:", p.text_content("#loadtxt"), f"{rec['load_s']}s")
            if "Couldn't" in p.text_content("#loadtxt"): rec["error"] = p.text_content("#loadtxt"); save(); b.close(); continue
            p.click("[data-pane=lab]")
            for step in steps:
                if left() < 3 * 60: rec["steps"][step] = "skipped: budget"; break
                t1 = time.time(); p.click("#" + step)
                try:
                    p.wait_for_function(f"() => /{DONE[step]}|Error/.test(document.querySelector('#labmeta').textContent)", timeout=cap(STEP_CAP[kind]), polling=5000)
                    rec["steps"][step] = f"{p.text_content('#labmeta')[:200]} ({time.time()-t1:.0f}s)"
                except Exception:
                    rec["steps"][step] = f"TIMEOUT after {time.time()-t1:.0f}s at: {p.text_content('#labmeta')[:200]}"
                log(step, rec["steps"][step]); log(p.text_content("#labout")[:1500])
                try: json.dump(p.evaluate("() => JSON.parse(JSON.stringify(window.stegoReport))"), open(os.path.join(OUT, f"report_{name}.json"), "w"))
                except Exception as e: log("report save failed", e)
                save()
                if rec["steps"][step].startswith("TIMEOUT"): break
            rec["seconds"] = round(time.time() - t0)
        except Exception as e:
            log("FAILED", name, traceback.format_exc()[-2000:]); rec["error"] = str(e)[:500]
        finally:
            try: b and b.close()
            except Exception: pass
            save()

reps = {n: json.load(open(os.path.join(OUT, f"report_{n}.json"))) for n in BACKENDS if os.path.exists(os.path.join(OUT, f"report_{n}.json"))}
if len(reps) >= 2 and all(r and "fingerprint" in r for r in reps.values()) and left() > 60:
    try:
        with sync_playwright() as pw:
            b = pw.chromium.launch(headless=True, executable_path=CHROME, args=["--no-sandbox"]); p = b.new_page(); p.goto(URL)
            a, c = list(reps.values())[:2]
            summary["cross_backend"] = p.evaluate("""async ([a, c]) => { const L = await import('./stegolab.mjs'); const S = await import('./stego.mjs');
                return L.compareFingerprints(a.fingerprint, c.fingerprint, S.DEFAULTS); }""", [a, c])
            b.close()
    except Exception as e: summary["cross_backend"] = {"error": str(e)[:300]}
for n, r in reps.items():
    if not r: continue
    summary.setdefault(n, {}).update({k: r.get(k) for k in ("sameDeviceRepeat", "prefillVsDecode")})
    if r.get("capacity"): summary[n]["capacityTop5"] = r["capacity"][:5]
    if r.get("roundTrips"): summary[n]["roundTrips"] = [{k: x.get(k) for k in ("secret", "correct", "reason", "tries", "tokens", "chars", "text")} for x in r["roundTrips"]]
save()
log(json.dumps(summary, indent=1)[:6000])
