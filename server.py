#!/usr/bin/env python3
"""Bonsai-Life local server: serves the web app and proxies the naturalist call to llama-server.

The browser never talks to the model directly; this keeps one origin and lets us validate the JSON.
Stdlib only. Usage: python3 server.py [--port 8000] [--llama http://127.0.0.1:8080]
"""
import argparse, json, os, random, time, urllib.request
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
ap.add_argument("--llama", default=os.environ.get("LLAMA_URL", "http://127.0.0.1:8080"))
ap.add_argument("--timeout", type=float, default=float(os.environ.get("NATURALIST_TIMEOUT", 240)))
args = ap.parse_args()
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
COLORS = ["red", "orange", "yellow", "green", "blue", "purple"]

SCHEMA = {
    "type": "object",
    "properties": {
        "species_name": {"type": "string"},
        "field_notes": {"type": "string"},
        "proposed_edits": {
            "type": "array", "minItems": 2, "maxItems": 4,
            "items": {"type": "object",
                      "properties": {"source_color": {"type": "string", "enum": COLORS},
                                     "target_color": {"type": "string", "enum": COLORS},
                                     "attraction": {"type": "number"}},
                      "required": ["source_color", "target_color", "attraction"]}},
    },
    "required": ["species_name", "field_notes", "proposed_edits"],
}

def prompt(matrix, goal):
    rows = "\n".join(f"{COLORS[i]:>7}: " + " ".join(f"{COLORS[j]}={matrix[i*6+j]:+.2f}" for j in range(6)) for i in range(6))
    return (
        "You are a field biologist studying a particle-life petri dish. The image shows the current champion organism: "
        "colored particles, with faint streaks as motion trails.\n"
        f"The user's goal: \"{goal}\".\n"
        "Attraction matrix (row color is pulled toward column color when positive, pushed away when negative; "
        "asymmetric pairs like A->B positive with B->A negative make chasers and swimmers):\n"
        f"{rows}\n\n"
        "Reply in JSON with: species_name (a made-up two-word Latin binomial like 'Vermis fluorescens'), field_notes (1-2 vivid sentences about what you "
        "see in the image: clumps, worms, rings, swimmers, chasing), and proposed_edits (2-4 edits that push the dish toward "
        "the goal; attraction between -1 and 1)."
    )

def llama(path, body=None, timeout=5):
    req = urllib.request.Request(args.llama + path, json.dumps(body).encode() if body else None,
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def model_name():
    try:
        h = llama("/health", timeout=2)
        if h.get("status") != "ok": return None
        try:
            m = llama("/v1/models", timeout=2)["data"][0]["id"]
            return os.path.basename(m)
        except Exception:
            return "llama-server"
    except Exception:
        return None

def naturalist(req):
    matrix = [float(x) for x in req["matrix"]][:36]
    goal = str(req.get("goal", ""))[:200]
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": req["image"]}},
            {"type": "text", "text": prompt(matrix, goal)}]}],
        "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
        "response_format": {"type": "json_object", "schema": SCHEMA},
    }
    t = time.time()
    r = llama("/v1/chat/completions", body, timeout=args.timeout)
    txt = r["choices"][0]["message"].get("content") or ""
    out = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
    edits = []
    for e in out.get("proposed_edits", [])[:4]:
        try:
            s, d = COLORS.index(str(e["source_color"]).lower()), COLORS.index(str(e["target_color"]).lower())
            edits.append({"source": s, "target": d, "attraction": max(-1.0, min(1.0, float(e["attraction"])))})
        except Exception:
            continue
    return {"species_name": str(out.get("species_name", ""))[:60], "field_notes": str(out.get("field_notes", ""))[:400],
            "edits": edits, "ms": int((time.time() - t) * 1000), "raw": txt[:600],
            "tokens": r.get("usage", {})}

class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=WEB, **k)
    def log_message(self, fmt, *a):
        if "/api/" in (self.path or ""): super().log_message(fmt, *a)
    def send_json(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/api/health"): return self.send_json(200, {"model": model_name()})
        return super().do_GET()
    def do_POST(self):
        if not self.path.startswith("/api/naturalist"): return self.send_json(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            res = naturalist(req)
            print(f"[naturalist] {res['ms']} ms  {res['species_name']!r}  edits={len(res['edits'])}", flush=True)
            return self.send_json(200, res)
        except Exception as e:
            print("[naturalist] error:", repr(e)[:300], flush=True)
            return self.send_json(200, {"error": repr(e)[:300]})

if __name__ == "__main__":
    print(f"Bonsai-Life on http://{args.host}:{args.port}  (model server: {args.llama}, online: {model_name()})", flush=True)
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()
