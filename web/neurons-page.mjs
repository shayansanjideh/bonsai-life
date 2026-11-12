// neurons.html: load Bonsai, map a sample of its neurons on a warm-up text, then trace every word it reads or writes.
import { loadRealEngine, StubEngine, sample } from "./brain-engine.mjs";
import { NeuronMap, WordMap } from "./neurons.mjs";

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const show = p => p.replace(/\n/g, "↵").replace(/^ /, "·") || "∅";
const tip = $("tip");
function setStatus(t, cls = "") { const s = $("status"); s.textContent = t; s.className = "status " + cls; }

// Warm-up text for choosing and placing neurons: varied on purpose (facts, story, code, instructions, a letter).
const WARMUP = [
  "The capital of France is Paris, and the capital of Japan is Tokyo. Water boils at 100 degrees Celsius.",
  "Once upon a time, a small fox lived at the edge of a forest. Every evening she climbed the hill to watch the sun go down.",
  "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
  "To make tea, boil water and pour it over the leaves. Wait three minutes, then add milk if you like.",
  "Dear Sam, thank you for your letter. I am sorry it took me so long to reply. One, two, three, four, five.",
];

let eng = null, map = null, cloud = null, wm = null, dims = null;
// one entry per traced token: { id, piece, prompt, z (Float32Array S), path (Float32Array L*3) }
let toks = [], cur = -1, picked = -1, busy = false, stopFlag = false;

$("btnLoad").onclick = async () => {
  $("btnLoad").disabled = true;
  try {
    if (params.has("stub")) { eng = new StubEngine({ delayMs: 4 }); $("stubNote").hidden = false; }
    else {
      $("prog").hidden = false;
      eng = await loadRealEngine({
        onProgress: f => { $("prog").firstElementChild.style.width = (f * 100).toFixed(1) + "%"; setStatus(`downloading ${Math.round(f * 429)} / 429 MB`); },
        onStatus: m => setStatus(m.length > 46 ? m.slice(0, 46) + "…" : m),
      });
    }
    dims = eng.traceDims;
    $("wmLayer").max = dims.L - 1; if (+$("wmLayer").value > dims.L - 1) $("wmLayer").value = Math.round(dims.L * 0.66);
    $("wmLayerV").textContent = $("wmLayer").value;
    await calibrate();
    setStatus("Ready", "on");
    $("btnLoad").textContent = "Loaded"; $("btnLoad").classList.remove("primary");
    $("btnGen").disabled = false;
    window.cloudState = { eng, map, toks };
    if (params.has("autogen")) $("btnGen").click();
  } catch (e) {
    console.error(e); setStatus("load failed", "err"); $("btnLoad").disabled = false;
    $("cloudMsg").textContent = "Couldn't start: " + (e.message || e);
  }
  $("prog").hidden = true;
};

async function calibrate() {
  const perLayer = Math.min(192, dims.F);
  map = new NeuronMap(dims, { perLayer });
  const lists = WARMUP.map(t => eng.encodePrompt(t, false));
  const total = lists.reduce((a, l) => a + l.length - 1, 0); let done = 0;
  $("prog").hidden = false; const t0 = performance.now();
  for (const ids of lists) {
    await eng.startTraced(ids, (i, tr) => {
      map.addCalibration(tr.neurons); done++;
      $("prog").firstElementChild.style.width = (done / total * 100).toFixed(1) + "%";
      setStatus(`mapping neurons · ${done} / ${total} words`, "busy");
    });
  }
  $("kSpeed").textContent = (done / ((performance.now() - t0) / 1000)).toFixed(1) + " tok/s";
  setStatus("placing neurons…", "busy"); await new Promise(r => setTimeout(r, 30));
  map.build();
  const S = map.S, F = dims.F;
  $("kN").textContent = `${S.toLocaleString()} of ${(dims.L * F).toLocaleString()}`;
  $("howDots").textContent = `Each layer of Bonsai has an MLP with ${F.toLocaleString()} neurons. Showing all ${(dims.L * F).toLocaleString()} would be a fog, so each layer shows the ${map.K} that vary most on a ${map.calibTokens ?? total}-word warm-up text. Their activations are copied off the GPU after every layer, for every word.`;
  if (eng.kind === "stub") $("cloudFoot").textContent = "Test model: these activations are generated, not Bonsai's.";
  try {
    const { NeuronCloud } = await import("./neurons3d.mjs");
    $("cloud").innerHTML = "";
    cloud = new NeuronCloud($("cloud"), map, { onHover, onPick });
    window.cloud = cloud;
  } catch (e) { console.error(e); $("cloud").innerHTML = `<div class="fallback">3D view unavailable here (${esc(e.message || e)}).</div>`; }
  wm = new WordMap(dims);
}

function onHover(h) {
  if (!h) { tip.style.display = "none"; return; }
  const t = toks[cur], z = t ? t.z[h.s] : null;
  tip.innerHTML = `<b>Layer ${h.layer} · neuron ${h.neuron}</b>${z !== null ? `<br>${z >= 0 ? "+" : ""}${z.toFixed(1)}σ vs usual on “${esc(show(t.piece))}”` : ""}<br><span style="color:var(--muted)">click to follow it</span>`;
  tip.style.display = "block"; tip.style.left = Math.min(innerWidth - 280, h.x + 14) + "px"; tip.style.top = h.y + 16 + "px";
}
function onPick(p) { picked = p ? p.s : -1; renderNeuron(); }

function renderNeuron() {
  const c = $("spark"), g = c.getContext("2d"), w = c.width = c.clientWidth * devicePixelRatio, h = c.height = c.clientHeight * devicePixelRatio;
  g.clearRect(0, 0, w, h);
  if (picked < 0) { $("nTitle").textContent = "Click a dot in the cloud"; $("nToks").innerHTML = ""; $("nHint").textContent = "You'll see which words in the text make it fire."; return; }
  const d = map.describe(picked); $("nTitle").textContent = `Layer ${d.layer} · neuron ${d.neuron}`;
  if (!toks.length) { $("nHint").textContent = "Generate some text to see what makes it fire."; $("nToks").innerHTML = ""; return; }
  const zs = toks.map(t => t.z[picked]), m = Math.max(3, ...zs.map(Math.abs));
  g.strokeStyle = "rgba(16,40,80,.15)"; g.beginPath(); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke();
  const bw = w / zs.length;
  zs.forEach((z, i) => { const bh = (z / m) * (h / 2 - 2); g.fillStyle = i === cur ? "#0d1526" : z > 0 ? "#ff2a2a" : "#2a7bff"; g.fillRect(i * bw + 0.5, h / 2 - Math.max(0, bh), Math.max(1, bw - 1), Math.abs(bh)); });
  const top = toks.map((t, i) => ({ t, z: zs[i] })).sort((a, b) => b.z - a.z).slice(0, 8).filter(x => x.z > 0.5);
  $("nHint").textContent = top.length ? "Strongest on these words in the text (σ above its usual level):" : "It hasn't fired well above its usual level on this text yet.";
  $("nToks").innerHTML = top.map(x => `<span>${esc(show(x.t.piece))}<b>${x.z.toFixed(1)}σ</b></span>`).join("");
}

// ---- word map (2D canvas) ----
let wmPts = [], wmDirty = true;
function drawWordMap() {
  const c = $("wm"), g = c.getContext("2d"), dpr = devicePixelRatio, w = c.width = c.clientWidth * dpr, h = c.height = c.clientHeight * dpr;
  g.clearRect(0, 0, w, h);
  if (!wm || wm.items.length < 4) { g.fillStyle = "#8a95a8"; g.font = `${13 * dpr}px -apple-system,system-ui,sans-serif`; g.textAlign = "center"; g.fillText(wm ? "Generate a few words to fill the map." : "Wake Bonsai first.", w / 2, h / 2); return; }
  if (wmDirty) { wmPts = wm.project(+$("wmLayer").value); wmDirty = false; }
  const pad = 26 * dpr, X = x => pad + (x + 1) / 2 * (w - 2 * pad), Y = y => pad + (1 - (y + 1) / 2) * (h - 2 * pad);
  const n = wmPts.length;
  g.lineWidth = 1 * dpr; g.strokeStyle = "rgba(155,123,255,.25)"; g.beginPath();
  wmPts.forEach((p, i) => i ? g.lineTo(X(p.x), Y(p.y)) : g.moveTo(X(p.x), Y(p.y))); g.stroke();
  const order = wmPts.map((p, i) => i).sort((a, b) => wmPts[a].z - wmPts[b].z);
  g.font = `${12 * dpr}px ui-monospace,SFMono-Regular,Menlo,monospace`; g.textAlign = "left";
  for (const i of order) {
    const p = wmPts[i], isCur = p.meta === cur, t = toks[p.meta], r = (2.4 + 1.4 * (p.z + 1) / 2) * dpr;
    g.fillStyle = isCur ? "#ff2a2a" : t?.prompt ? "rgba(86,98,117,.5)" : `hsl(${255 - 40 * i / n},70%,58%)`;
    g.beginPath(); g.arc(X(p.x), Y(p.y), isCur ? r * 1.7 : r, 0, 7); g.fill();
  }
  // labels: current word first, then the model's own words, then the prompt; skip any that would overlap
  const boxes = [], fs = 12 * dpr, lab = wmPts.map((p, i) => i).sort((a, b) => (wmPts[b].meta === cur) - (wmPts[a].meta === cur) || (toks[wmPts[a].meta]?.prompt ? 1 : 0) - (toks[wmPts[b].meta]?.prompt ? 1 : 0));
  for (const i of lab) {
    const p = wmPts[i], isCur = p.meta === cur, txt = p.text.trim().slice(0, 14); if (!txt) continue;
    const x = X(p.x) + 6 * dpr, y = Y(p.y) + 4 * dpr, bw = g.measureText(txt).width, bx = [x - 1, y - fs, x + bw + 1, y + 3];
    if (!isCur && boxes.some(o => bx[0] < o[2] && bx[2] > o[0] && bx[1] < o[3] && bx[3] > o[1])) continue;
    boxes.push(bx); g.fillStyle = isCur ? "#0d1526" : toks[p.meta]?.prompt ? "rgba(86,98,117,.75)" : "rgba(13,21,38,.82)";
    if (isCur) { g.font = `600 ${fs}px ui-monospace,SFMono-Regular,Menlo,monospace`; }
    g.fillText(txt, x, y); g.font = `${fs}px ui-monospace,SFMono-Regular,Menlo,monospace`;
  }
}
$("wmLayer").oninput = () => { $("wmLayerV").textContent = $("wmLayer").value; wmDirty = true; drawWordMap(); };
new ResizeObserver(() => drawWordMap()).observe($("wm"));
let wmT = 0; const wmSoon = () => { wmDirty = true; clearTimeout(wmT); wmT = setTimeout(drawWordMap, 120); };

// ---- text ----
function renderText() {
  $("out").innerHTML = toks.map((t, i) => `<span class="tok${t.prompt ? " pr" : ""}${i === cur ? " cur" : ""}" data-i="${i}">${esc(t.piece)}</span>`).join("") + (busy ? '<span class="caret"></span>' : "");
}
$("out").onclick = e => { const el = e.target.closest(".tok"); if (!el || busy) return; focusTok(+el.dataset.i); };
function focusTok(i, ms) {
  cur = i; const t = toks[i]; if (!t) return;
  cloud?.show(t.z, t.path, { ms: ms ?? ($("slow").checked ? 2400 : 650) });
  $("kTok").textContent = show(t.piece).trim().slice(0, 12) || "·";
  for (const el of $("out").querySelectorAll(".tok")) el.classList.toggle("cur", +el.dataset.i === i);
  drawWordMap(); renderNeuron();
}
const special = p => /^<\|.*\|>$/.test(p.trim()) || !p.trim();
function addTok(id, trace, prompt) {
  const piece = eng.piece(id), z = map.activity(trace.neurons), path = map.path(z);
  const i = toks.push({ id, piece, prompt, z, path }) - 1;
  if (!special(piece)) wm.add(piece, trace.resid, i);
  return i;
}
const wait = ms => new Promise(r => setTimeout(r, ms));

$("btnGen").onclick = async () => {
  if (busy) return; busy = true; stopFlag = false;
  for (const b of ["btnGen", "btnReplay", "btnLoad"]) $(b).disabled = true; $("btnStop").disabled = false;
  toks = []; cur = -1; wm.clear(); wmDirty = true; if (cloud) cloud.busy = true;
  const slow = () => $("slow").checked;
  try {
    const chat = $("mode").value === "chat", ids = eng.encodePrompt($("prompt").value, chat);
    setStatus("reading the prompt", "busy");
    toks.push({ id: ids[0], piece: eng.piece(ids[0]), prompt: true, z: new Float32Array(map.S), path: new Float32Array(dims.L * 3) });
    let logits = await eng.startTraced(ids, async (i, tr) => { const k = addTok(ids[i], tr, true); renderText(); focusTok(k, 300); if (slow()) await wait(400); });
    setStatus("writing", "busy");
    const temp = +$("temp").value, max = +$("maxTok").value, t0 = performance.now(); let n = 0;
    for (; n < max && !stopFlag; n++) {
      const id = sample(logits, temp);
      if (eng.stopIds.has(id)) break;
      const r = await eng.stepTraced(id); logits = r.logits;
      const k = addTok(id, r.trace, false); renderText(); focusTok(k); if (n % 4 === 0) wmSoon();
      $("kSpeed").textContent = ((n + 1) / ((performance.now() - t0) / 1000)).toFixed(1) + " tok/s";
      await wait(slow() ? 2600 : 120);
    }
    setStatus("Ready", "on");
  } catch (e) { console.error(e); setStatus("error: " + (e.message || e).slice(0, 60), "err"); }
  busy = false; if (cloud) cloud.busy = false; renderText(); wmSoon(); renderNeuron();
  for (const b of ["btnGen", "btnReplay"]) $(b).disabled = false; $("btnStop").disabled = true;
  window.cloudState = { eng, map, toks };
};
$("btnStop").onclick = () => { stopFlag = true; };
$("btnReplay").onclick = async () => {
  if (busy || !toks.length) return; busy = true; stopFlag = false; $("btnStop").disabled = false; $("btnReplay").disabled = $("btnGen").disabled = true;
  for (let i = 1; i < toks.length && !stopFlag; i++) { focusTok(i); await wait($("slow").checked ? 2600 : toks[i].prompt ? 260 : 520); }
  busy = false; $("btnStop").disabled = true; $("btnReplay").disabled = $("btnGen").disabled = false; renderText();
};
$("temp").oninput = () => $("tempV").textContent = $("temp").value;
if (params.has("stub")) $("btnLoad").textContent = "Load test model";
drawWordMap();
if (params.has("autoload")) $("btnLoad").click();
