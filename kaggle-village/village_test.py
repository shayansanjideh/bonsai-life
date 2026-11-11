# Real-model test of Bonsai Village on a Kaggle T4.
# 1) Probe whether headless Chromium exposes a real WebGPU adapter on the T4; if yes, run the page in true in-browser mode.
# 2) Otherwise (and in addition) run the page with ?llama= pointing at PrismML's llama.cpp serving the same Bonsai 2B weights.
import os, json, time, glob, subprocess, tarfile, shutil, urllib.request, threading, http.server, socketserver, functools
W = "/kaggle/working"; os.makedirs(W, exist_ok=True); os.chdir(W)
LOG = open(f"{W}/log.txt", "a")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def retry(f, *a, **k):
    for i in range(6):
        try: return f(*a, **k)
        except Exception as e: log("retry", i, repr(e)[:150]); time.sleep(10 * (i + 1))
    return f(*a, **k)
SITE = "https://shayansanjideh.github.io/bonsai-life/"
R = {}
def save(): json.dump(R, open(f"{W}/results.json", "w"), indent=1)

# ---------- site files (served locally so the page and llama-server are both on localhost) ----------
os.makedirs("site/vendor", exist_ok=True)
for f in ["index.html", "naturalist.mjs", "vendor/bonsai.mjs"]:
    retry(urllib.request.urlretrieve, SITE + f, "site/" + f)
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
srv = socketserver.ThreadingTCPServer(("127.0.0.1", 8000), functools.partial(Q, directory=f"{W}/site"))
threading.Thread(target=srv.serve_forever, daemon=True).start()
log("site served")

# ---------- browser ----------
subprocess.run("pip install -q playwright==1.48.0 && python -m playwright install --with-deps chromium > pw.log 2>&1", shell=True)
from playwright.sync_api import sync_playwright
log("playwright ready")

# ---------- llama.cpp server (same weights) ----------
def setup_llama():
    os.makedirs("bin", exist_ok=True); os.makedirs("models", exist_ok=True)
    found = glob.glob("/kaggle/input/**/llama-server", recursive=True)
    if found: shutil.copytree(os.path.dirname(found[0]), "bin/x", dirs_exist_ok=True)
    else: tarfile.open(glob.glob("/kaggle/input/**/*.tar.gz", recursive=True)[0]).extractall("bin")
    for f in glob.glob("bin/**/*", recursive=True):
        if os.path.isfile(f): os.chmod(f, 0o755)
    for f in glob.glob("bin/**/*.so*", recursive=True):
        if os.path.getsize(f) == 0:
            base = f.split(".so")[0] + ".so"; real = [g for g in glob.glob(base + "*") if os.path.getsize(g) > 0]
            if real: os.remove(f); os.symlink(os.path.basename(max(real, key=len)), f)
    subprocess.run("pip install -q nvidia-cuda-runtime-cu12 nvidia-cublas-cu12", shell=True)
    import site
    libs = glob.glob(f"{site.getsitepackages()[0]}/nvidia/*/lib")
    srvbin = glob.glob("bin/**/llama-server", recursive=True)[0]
    from huggingface_hub import hf_hub_download
    m = retry(hf_hub_download, "prism-ml/Ternary-Bonsai-2-2B-gguf", "Ternary-Bonsai-2-2B-PQ2_0.gguf", local_dir="models")
    env = dict(os.environ, LD_LIBRARY_PATH=":".join([os.path.dirname(os.path.abspath(srvbin))] + libs + [os.environ.get("LD_LIBRARY_PATH", "")]))
    p = subprocess.Popen([srvbin, "-m", m, "-ngl", "99", "-c", "8192", "-np", "1", "--port", "8080", "--host", "127.0.0.1"], stdout=open(f"{W}/llama.log", "w"), stderr=subprocess.STDOUT, env=env)
    for _ in range(300):
        try:
            if json.load(urllib.request.urlopen("http://127.0.0.1:8080/health")).get("status") == "ok": break
        except Exception: pass
        if p.poll() is not None: raise RuntimeError(open(f"{W}/llama.log").read()[-2000:])
        time.sleep(2)
    log("llama up"); return p

SNAP = """() => ({ stats: BV.stats, picks: BV.picks, status: document.getElementById('status').textContent,
  creatures: BV.C.map(c => ({ name: c.name, brain: c.brain, alive: c.alive, energy: +c.energy.toFixed(2), decisions: c.decisions, rumor: c.rumor, action: c.action, last: c.last && { action: c.last.action, target: c.last.target, say: c.last.say, thought: c.last.thought, reason: c.last.reason, valid: c.last.valid } })),
  feed: [...document.querySelectorAll('#feed .ev')].map(e => e.innerText.replace(/\\n/g, ' ')).reverse(),
  board: document.getElementById('board') ? document.getElementById('board').innerText : '' })"""

def run_village(page, tag, minutes, video_ok=True):
    out = {"snaps": []}
    t0 = time.time(); k = 0; whispered = rethought = False
    while time.time() - t0 < minutes * 60:
        page.wait_for_timeout(30000); k += 1
        el = (time.time() - t0) / 60
        if not whispered and el > 2:
            page.click("#tab-mind"); page.evaluate("BV.select('Pip')")
            page.click("#whisperBtn"); whispered = True; log(tag, "whispered to Pip")
        if not rethought and el > 5:
            page.click("#tab-mind"); page.evaluate("BV.select(BV.C.find(c => c.brain === 'bonsai' && c.alive)?.name || 'Mo')")
            page.click("#rethink"); page.wait_for_timeout(20000)
            out["rethink"] = {"out": page.inner_text("#rethinkOut"), "A": page.inner_text("#abA"), "B": page.inner_text("#abB")}
            try: page.click("#undo")
            except Exception: pass
            rethought = True; log(tag, "rethink", out["rethink"])
        s = page.evaluate(SNAP); out["snaps"].append({"min": round(el, 1), "stats": s["stats"], "picks": s["picks"]})
        if k % 2 == 0: page.screenshot(path=f"{W}/{tag}_{k:02d}.png")
        log(tag, f"{el:.1f} min", json.dumps(s["stats"]), json.dumps(s["picks"]))
    out["final"] = page.evaluate(SNAP)
    return out

def browser_run(pw, url, tag, minutes, arena=False, video=False):
    b = pw.chromium.launch(args=["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan", "--ignore-gpu-blocklist", "--use-vulkan"])
    ctx = b.new_context(viewport={"width": 1440, "height": 900}, record_video_dir=f"{W}/video_{tag}" if video else None, record_video_size={"width": 1440, "height": 900} if video else None)
    page = ctx.new_page(); errs = []
    page.on("pageerror", lambda e: errs.append(str(e))); page.on("console", lambda m: errs.append(m.text) if m.type in ("error", "warning") else None)
    page.goto(url); page.wait_for_timeout(3000)
    if arena: page.click("#mode")
    try: out = run_village(page, tag, minutes)
    finally:
        out = locals().get("out", {}); out["errors"] = errs[:30]; ctx.close(); b.close()
    return out

with sync_playwright() as pw:
    # 1. WebGPU probe
    b = pw.chromium.launch(args=["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan", "--ignore-gpu-blocklist", "--use-vulkan"])
    p = b.new_page(); p.goto("http://127.0.0.1:8000/index.html?stub=1")
    R["webgpu_probe"] = p.evaluate("""async () => { if (!navigator.gpu) return { gpu: false }; const a = await navigator.gpu.requestAdapter(); if (!a) return { gpu: true, adapter: null };
        const i = a.info || (a.requestAdapterInfo ? await a.requestAdapterInfo() : {}); return { gpu: true, vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description, isFallback: a.isFallbackAdapter }; }""")
    b.close(); log("webgpu probe", R["webgpu_probe"]); save()
    probe = R["webgpu_probe"]; real_gpu = probe.get("gpu") and probe.get("vendor") and "google" not in str(probe.get("vendor")).lower() and not probe.get("isFallback")
    # 2. true in-browser run if a real adapter exists
    if real_gpu:
        try:
            R["webgpu_village"] = browser_run(pw, "http://127.0.0.1:8000/index.html", "webgpu", 8, video=True); save()
        except Exception as e: R["webgpu_error"] = repr(e)[:500]; log("webgpu run failed", e); save()
    # 3. llama.cpp-backed runs (same weights) for decision quality + bias + arena
    lp = setup_llama()
    R["llama_village"] = browser_run(pw, "http://127.0.0.1:8000/index.html?llama=http://127.0.0.1:8080", "village", 10, video=not real_gpu); save()
    R["llama_arena"] = browser_run(pw, "http://127.0.0.1:8000/index.html?llama=http://127.0.0.1:8080", "arena", 26, arena=True); save()
    lp.terminate()
log("ALL DONE")
