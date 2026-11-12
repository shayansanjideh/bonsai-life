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

FILES = ["web/brain-engine.mjs", "web/naturalist.mjs", "web/neurons.mjs", "web/vendor/bonsai.mjs", "tests/real-model-test.html", "tests/neuron-trace-test.html", "tests/trace-dump.html"]
# Which test page to run. TEST=neuron runs the activation-tap self-test (minutes); TEST=brain runs the surgery test and,
# on a real GPU, the 6-passage lesion scan (far too slow for the CPU fallback, so it runs in quick mode there).
TEST = os.environ.get("BONSAI_TEST", "neuron")
PAGE = {"neuron": "tests/neuron-trace-test.html", "brain": "tests/real-model-test.html", "truth": "tests/trace-dump.html"}[TEST]
TRUTH_PROMPT = "The capital of France is Paris, and the capital of Japan is"

# ---- TEST=truth: ground truth from PrismML's llama.cpp (CPU), via llama-eval-callback's tensor dumps ----
# The WebGPU tap claims: resid = residual stream after layer L (llama.cpp "l_out-L"), neurons = silu(gate)*up of layer
# L's MLP (llama.cpp "ffn_swiglu-L" or "ffn_gate_par-L"). eval-callback prints, per tensor, the first and last 3 tokens x
# first and last 3 values, and the total sum. We compare the values of the last 3 tokens.
PRISM_TAG = "prism-b12128-d7b2896"
def llama_reference():
    import re, tarfile
    d = "/tmp/llama"; os.makedirs(d, exist_ok=True)
    tgz = f"{d}/llama.tgz"
    if not os.path.exists(tgz):
        urllib.request.urlretrieve(f"https://github.com/PrismML-Eng/llama.cpp/releases/download/{PRISM_TAG}/llama-{PRISM_TAG}-bin-ubuntu-x64.tar.gz", tgz)
        tarfile.open(tgz).extractall(d)
    bins = {os.path.basename(p): p for p in glob.glob(d + "/**/llama-*", recursive=True) if os.path.isfile(p)}
    log("llama.cpp binaries:", sorted(bins))
    ev = bins.get("llama-eval-callback")
    if not ev: return {"error": "no llama-eval-callback in the release", "bins": sorted(bins)}
    for name in ["Ternary-Bonsai-2-2B-PTQ1_0.gguf", "Ternary-Bonsai-2-2B-PQ2_0.gguf"]:
        mp = f"{d}/{name}"
        if not os.path.exists(mp):
            try: urllib.request.urlretrieve(f"https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf/resolve/main/{name}", mp)
            except Exception as e: log("download failed", name, e); continue
        env = dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(ev) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
        r = subprocess.run([ev, "-m", mp, "-p", TRUTH_PROMPT, "-ngl", "0", "-c", "256", "-t", str(os.cpu_count() or 4)], capture_output=True, text=True, timeout=1800, env=env)
        out = r.stdout + r.stderr
        open(os.path.join(W, f"evalcb_{name}.txt"), "w").write(out)
        log(name, "eval-callback exit", r.returncode, "bytes", len(out))
        if r.returncode == 0 and "l_out-" in out:
            tok = bins.get("llama-tokenize"); ids = None
            if tok:
                t = subprocess.run([tok, "-m", mp, "-p", TRUTH_PROMPT, "--ids", "--log-disable"], capture_output=True, text=True, env=env, timeout=600).stdout
                m = re.search(r"\[[\d,\s]+\]", t); ids = json.loads(m.group(0)) if m else None
            return {"model": name, "text": out, "ids": ids}
        log(out[-2000:])
    return {"error": "eval-callback produced no l_out tensors"}

def parse_evalcb(text):
    """{tensor name: {"rows": [[6 numbers] per printed token row], "sum": float, "shape": [...]}}"""
    import re
    res, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"\s*(?:ggml_debug|common_debug_cb_eval):\s+(\S+)\s*=.*=\s*\{([\d,\s]+)\}", line)
        if m: cur = m.group(1); res[cur] = {"rows": [], "shape": [int(x) for x in m.group(2).split(",")]}; continue
        if cur is None: continue
        m = re.match(r"\s*sum\s*=\s*([-\d.eE+naif]+)", line)
        if m: res[cur]["sum"] = float(m.group(1)); cur = None; continue
        nums = re.findall(r"-?\d+\.\d+(?:e[-+]?\d+)?|-?nan|-?inf", line)
        if nums and "[" in line: res[cur]["rows"].append([float(x) for x in nums])
    return res

def compare(ref, web):
    """Correlate the WebGPU values with llama.cpp's for the last 3 tokens of every layer."""
    import math
    t = parse_evalcb(ref["text"]); out = {"model": ref["model"], "llama_ids": ref.get("ids"), "web_ids": web.get("ids"), "layers": []}
    n = len(web["ids"]); webtok = web["tokens"]
    def stats(a, b):
        if not a: return None
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        va = sum((x - ma) ** 2 for x in a); vb = sum((y - mb) ** 2 for y in b)
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        return {"corr": cov / math.sqrt(va * vb) if va > 0 and vb > 0 else None, "max_abs_diff": max(abs(x - y) for x, y in zip(a, b)), "max_abs_ref": max(abs(x) for x in a), "n": len(a)}
    names_mlp = ["ffn_swiglu", "ffn_gate_par", "ffn_par"]
    out["mlp_name"] = next((nm for nm in names_mlp if f"{nm}-0" in t), None)
    allr, allw = {"resid": [], "neurons": []}, {"resid": [], "neurons": []}
    for l in range(web["dims"]["L"]):
        row = {"layer": l}
        for kind, name in [("resid", f"l_out-{l}"), ("neurons", f"{out['mlp_name']}-{l}" if out["mlp_name"] else None)]:
            e = t.get(name) if name else None
            if not e or len(e["rows"]) < 6: row[kind] = None; continue
            ref_vals, web_vals = [], []
            for k, tokidx in enumerate([n - 3, n - 2, n - 1]):
                rr = e["rows"][3 + k]; w = webtok[str(tokidx)][l][kind]
                ref_vals += rr[:3] + rr[-3:]; web_vals += w["head"] + w["tail"]
            row[kind] = stats(ref_vals, web_vals); allr[kind] += ref_vals; allw[kind] += web_vals
        out["layers"].append(row)
    out["overall"] = {k: stats(allr[k], allw[k]) for k in allr}
    return out
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
            pg.goto(f"http://127.0.0.1:8777/{PAGE}" + ("?quick=1" if quick and TEST == "brain" else ""))
            if TEST == "truth": log("(tracing the truth prompt)")
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
            if res and res.get("done") and not res.get("failed") and (res.get("scan") or TEST in ("neuron", "truth")): break  # done on the GPU, no need for the fallback
    srv.kill()
    if TEST == "truth":
        web = next((json.load(open(f))["result"] for f in sorted(glob.glob(os.path.join(W, "result_*.json"))) if (json.load(open(f)).get("result") or {}).get("ids")), None)
        try:
            ref = llama_reference()
            if "error" in ref or not web: summary["truth"] = {"error": ref.get("error") or "no WebGPU dump", "ref": {k: v for k, v in ref.items() if k != "text"}}
            else:
                c = compare(ref, web); summary["truth"] = c
                log("=== ground truth (llama.cpp vs WebGPU tap)", json.dumps({"model": c["model"], "ids_match": c["llama_ids"] == c["web_ids"], "mlp_name": c["mlp_name"], "overall": c["overall"]}, indent=1))
                for r in c["layers"]: log(f"L{r['layer']:2d} resid {r['resid']}  neurons {r['neurons']}")
        except Exception as e:
            import traceback; summary["truth"] = {"error": traceback.format_exc()}; log(traceback.format_exc())
    log("=== summary", json.dumps(summary, indent=1))
    json.dump(summary, open(os.path.join(W, "summary.json"), "w"), indent=1)

if len(sys.argv) == 4 and sys.argv[1] == "--compare":   # offline: python kaggle_brain_test.py --compare evalcb.txt result_x.json
    c = compare({"text": open(sys.argv[2]).read(), "model": os.path.basename(sys.argv[2])}, json.load(open(sys.argv[3]))["result"])
    print(json.dumps({"mlp_name": c["mlp_name"], "web_ids": c["web_ids"], "overall": c["overall"]}, indent=1))
    for r in c["layers"]: print(f"L{r['layer']:2d} resid {r['resid']}  neurons {r['neurons']}")
else:
    main()
