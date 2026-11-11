# Kaggle script kernel: run tests/real-model-test.html against the real Ternary-Bonsai-2-2B in headless Chromium.
# Tries the T4 through Vulkan first, then falls back to SwiftShader (CPU WebGPU, slow, quick mode).
# Outputs land in /kaggle/working: result_<config>.json, shot_<config>.png, log.txt.
import json, os, shutil, subprocess, sys, time, urllib.request, glob

W = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
ROOT = "/tmp/bonsai-life"
LOG = open(os.path.join(W, "log.txt"), "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def sh(cmd, timeout=900):
    log("$", cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = (r.stdout + r.stderr)[-3000:]
    if out.strip(): log(out)
    return r.returncode

FILES = ["web/brain-engine.mjs", "web/naturalist.mjs", "web/vendor/bonsai.mjs", "tests/real-model-test.html"]
def fetch_repo():
    if sh(f"timeout 90 git clone --depth 1 https://github.com/shayansanjideh/bonsai-life {ROOT}", 120) == 0:
        return
    shutil.rmtree(ROOT, ignore_errors=True)
    for f in FILES:
        for base in ["https://cdn.jsdelivr.net/gh/shayansanjideh/bonsai-life@main/", "https://raw.githubusercontent.com/shayansanjideh/bonsai-life/main/"]:
            try:
                os.makedirs(os.path.dirname(os.path.join(ROOT, f)), exist_ok=True)
                data = urllib.request.urlopen(base + f, timeout=60).read()
                open(os.path.join(ROOT, f), "wb").write(data); log("got", f, len(data), "from", base); break
            except Exception as e:
                log("fetch failed", base + f, e)
        else:
            raise SystemExit("could not fetch " + f)

def setup_vulkan():
    sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")
    sh("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libvulkan1 vulkan-tools >/dev/null", 600)
    icds = glob.glob("/usr/share/vulkan/icd.d/nvidia*") + glob.glob("/etc/vulkan/icd.d/nvidia*")
    if not icds:
        libs = [p for d in ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/nvidia/lib64", "/usr/local/nvidia/lib", "/usr/lib"] for p in glob.glob(d + "/libGLX_nvidia.so*")]
        log("libGLX_nvidia candidates:", libs)
        if libs:
            os.makedirs("/etc/vulkan/icd.d", exist_ok=True)
            json.dump({"file_format_version": "1.0.0", "ICD": {"library_path": libs[0], "api_version": "1.3.0"}}, open("/etc/vulkan/icd.d/nvidia_icd.json", "w"))
            log("wrote nvidia ICD ->", libs[0])
    sh("vulkaninfo --summary 2>&1 | head -40")

def main():
    log("=== start", time.ctime())
    fetch_repo()
    sh(f"{sys.executable} -m pip install -q playwright", 600)
    sh(f"{sys.executable} -m playwright install --with-deps chromium", 900)
    setup_vulkan()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", "8777", "--bind", "127.0.0.1"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    from playwright.sync_api import sync_playwright
    configs = [
        ("t4_vulkan", ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan", "--use-vulkan=native", "--ignore-gpu-blocklist", "--disable-vulkan-surface", "--enable-gpu", "--no-sandbox"], False, 40 * 60),
        ("swiftshader", ["--enable-unsafe-swiftshader", "--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--no-sandbox"], True, 150 * 60),
    ]
    summary = {}
    with sync_playwright() as p:
        for name, args, quick, budget in configs:
            log(f"=== config {name}")
            try:
                b = p.chromium.launch(channel="chromium", args=args, headless=True)
            except Exception as e:
                log("launch failed", e); continue
            pg = b.new_page(viewport={"width": 1000, "height": 1400})
            pg.on("console", lambda m: LOG.write(f"[console {m.type}] {m.text[:400]}\n"))
            pg.on("pageerror", lambda e: log("[pageerror]", e))
            pg.goto("http://127.0.0.1:8777/")
            info = pg.evaluate("""async () => { const a = await navigator.gpu?.requestAdapter(); if (!a) return null;
                const i = a.info || {}; return { vendor: i.vendor, arch: i.architecture, desc: i.description, f16: a.features.has('shader-f16'), maxBuf: a.limits.maxBufferSize, maxSB: a.limits.maxStorageBufferBindingSize }; }""")
            log("adapter:", info)
            if not info or (name == "t4_vulkan" and "swiftshader" in json.dumps(info).lower()):
                log("no usable adapter for", name); b.close(); continue
            t0 = time.time()
            pg.goto("http://127.0.0.1:8777/tests/real-model-test.html" + ("?quick=1" if quick else ""))
            res = None
            while time.time() - t0 < budget:
                time.sleep(10)
                try:
                    res = pg.evaluate("window.result || null")
                except Exception as e:
                    log("evaluate failed", e); break
                if res and res.get("done"): break
                if int(time.time() - t0) % 120 < 10: log(f"[{int(time.time() - t0)} s]", (pg.inner_text("#o") or "")[-300:].replace("\n", " | "))
            text = pg.inner_text("#o")
            log(f"--- {name} output ---\n{text}")
            pg.screenshot(path=os.path.join(W, f"shot_{name}.png"), full_page=True)
            json.dump({"adapter": info, "result": res, "text": text, "seconds": time.time() - t0}, open(os.path.join(W, f"result_{name}.json"), "w"), indent=1)
            summary[name] = {"adapter": info, "done": bool(res and res.get("done")), "failed": res.get("failed") if res else None, "seconds": round(time.time() - t0)}
            b.close()
            if res and res.get("done") and not res.get("failed") and res.get("scan"): break  # full run on the GPU, no need for the fallback
    srv.kill()
    log("=== summary", json.dumps(summary, indent=1))
    json.dump(summary, open(os.path.join(W, "summary.json"), "w"), indent=1)

main()
