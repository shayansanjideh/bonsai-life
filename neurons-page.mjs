// neurons.html: load Bonsai, map a sample of its neurons on a warm-up text, then trace every word it reads or writes.
import { loadRealEngine, StubEngine, sample, softmaxTop, logProb } from "./brain-engine.mjs";
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
    $("wmLayer").max = dims.L - 1; $("wmLayerMax").textContent = dims.L - 1; if (+$("wmLayer").value > dims.L - 1) $("wmLayer").value = Math.round(dims.L * 0.66);
    $("wmLayerV").textContent = $("wmLayer").value;
    await calibrate();
    setStatus("Ready", "on");
    $("btnLoad").textContent = "Loaded"; $("btnLoad").closest(".ld")?.setAttribute("hidden", ""); $("btnLoad").classList.remove("primary");
    $("btnGen").disabled = false; buildBrain();
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
  $("kN").textContent = S.toLocaleString(); $("rShown").textContent = `${S.toLocaleString()} / ${(dims.L * F).toLocaleString()}`;
  $("howDots").textContent = `Each layer of Bonsai has an MLP with ${F.toLocaleString()} neurons. Showing all ${(dims.L * F).toLocaleString()} would be a fog, so each layer shows the ${map.K} that vary most on a ${map.calibTokens ?? total}-word warm-up text. Their activations are copied off the GPU after every layer, for every word, and match PrismML's own llama.cpp to a correlation of 0.9999.`;
  try {
    const { NeuronCloud } = await import("./neurons3d.mjs");
    $("cloud").innerHTML = "";
    cloud = new NeuronCloud($("cloud"), map, { onHover, onPick, onView: v => { $("rAz").textContent = v.azimuth.toFixed(1).padStart(5, "0") + "°"; $("rEl").textContent = v.elevation.toFixed(1) + "°"; drawRuler(v.ruler); } });
    window.cloud = cloud;
  } catch (e) { console.error(e); $("cloud").innerHTML = `<div class="fallback">3D view unavailable here (${esc(e.message || e)}).</div>`; }
  wm = new WordMap(dims);
}

function drawRuler(r) {
  const el = $("ruler"); if (!r || r.x < 40) { el.hidden = true; return; }
  el.hidden = false; const top = 28 + Math.min(...r.ys), bot = 28 + Math.max(...r.ys);
  el.style.cssText = `left:${r.x - 40}px;top:${top}px;height:${bot - top}px;bottom:auto`;
  const h1 = document.querySelector(".stage .ov h1, .stage .h1"), cb = $("cloud").getBoundingClientRect(), hb = h1?.getBoundingClientRect();
  const clear = y => !hb || cb.top + y > hb.bottom + 6 || cb.left + r.x - 40 > hb.right + 6;   // keep labels off the headline
  el.innerHTML = r.ys.map((y, l) => `<span class="${(l % 4 === 0 || l === r.ys.length - 1) && clear(y) ? "l" : ""}" data-l="L${String(l).padStart(2, "0")}" style="top:${y + 28 - top}px"></span>`).join("");
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
  if (picked < 0) { $("nTitle").textContent = "–"; $("nToks").innerHTML = ""; $("nHint").textContent = "Click a dot in the cloud to follow one neuron."; return; }
  const d = map.describe(picked); $("nTitle").innerHTML = `L${String(d.layer).padStart(2, "0")} <span>/</span> N${d.neuron}`;
  if (!toks.length) { $("nHint").textContent = "Generate some text to see what makes it fire."; $("nToks").innerHTML = ""; return; }
  const zs = toks.map(t => t.z[picked]), m = Math.max(3, ...zs.map(Math.abs));
  g.fillStyle = "#c9d0d4"; g.fillRect(0, h * 0.6, w, 1);
  const bw = w / zs.length, base = h * 0.6;
  zs.forEach((z, i) => { const bh = z > 0 ? (z / m) * (base - 2) : (z / m) * (h - base - 2); g.fillStyle = z >= 1.9 ? "#ff5a1f" : z > 0 ? "#12161a" : "#9aa4aa"; g.fillRect(i * bw + 0.5, base - Math.max(0, bh), Math.max(1, bw - 1.5), Math.max(1, Math.abs(bh))); });
  if (cur >= 0) { g.fillStyle = "#ff5a1f"; g.fillRect(cur * bw, h - 3 * devicePixelRatio, Math.max(2, bw), 3 * devicePixelRatio); }   // marks the selected word
  const top = toks.map((t, i) => ({ t, z: zs[i] })).sort((a, b) => b.z - a.z).slice(0, 6).filter(x => x.z > 0.5);
  $("nHint").textContent = top.length ? "Words in the text that make it fire most:" : "It hasn't fired well above its usual level on this text yet.";
  $("nToks").innerHTML = top.map(x => `<div><span>${esc(x.t.piece.trim() || show(x.t.piece))}</span><em>+${x.z.toFixed(1)}σ</em></div>`).join("");
}

// ---- word map (2D canvas) ----
let wmPts = [], wmDirty = true;
function drawWordMap() {
  const c = $("wm"), g = c.getContext("2d"), dpr = devicePixelRatio, w = c.width = c.clientWidth * dpr, h = c.height = c.clientHeight * dpr;
  g.clearRect(0, 0, w, h);
  g.strokeStyle = "rgba(18,22,26,.08)"; g.lineWidth = 1;
  for (let k = 1; k < 6; k++) { const x = Math.round(w * k / 6) + .5, y = Math.round(h * k / 6) + .5; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); }
  const font = (wt, px) => `${wt} ${px * dpr}px "Instrument Sans",-apple-system,Inter,system-ui,sans-serif`;
  if (!wm || wm.items.length < 4) { g.fillStyle = "#6c757d"; g.font = font(400, 13); g.textAlign = "center"; g.fillText(wm ? "Generate a few words to fill the map." : "Wake Bonsai first.", w / 2, h / 2); return; }
  if (wmDirty) { wmPts = wm.project(+$("wmLayer").value); wmDirty = false; }
  const pad = 26 * dpr, X = x => pad + (x + 1) / 2 * (w - 2 * pad), Y = y => pad + (1 - (y + 1) / 2) * (h - 2 * pad);
  g.lineWidth = 0.6 * dpr; g.strokeStyle = "rgba(18,22,26,.2)"; g.beginPath();
  wmPts.forEach((p, i) => i ? g.lineTo(X(p.x), Y(p.y)) : g.moveTo(X(p.x), Y(p.y))); g.stroke();
  for (const p of wmPts) {
    const isCur = p.meta === cur, t = toks[p.meta];
    g.fillStyle = isCur ? "#ff5a1f" : t?.prompt ? "#9aa4aa" : "#12161a";
    g.beginPath(); g.arc(X(p.x), Y(p.y), (isCur ? 4.5 : 2.4) * dpr, 0, 7); g.fill();
  }
  // labels: current word first, then the model's own words, then the prompt; skip any that would overlap
  const boxes = [], fs = 12.5 * dpr, lab = wmPts.map((p, i) => i).sort((a, b) => (wmPts[b].meta === cur) - (wmPts[a].meta === cur) || (toks[wmPts[a].meta]?.prompt ? 1 : 0) - (toks[wmPts[b].meta]?.prompt ? 1 : 0));
  for (const i of lab) {
    const p = wmPts[i], isCur = p.meta === cur, txt = p.text.trim().slice(0, 14); if (!txt) continue;
    g.font = font(isCur ? 600 : 400, 12.5);
    const x = X(p.x) + 7 * dpr, y = Y(p.y) + 4 * dpr, bw = g.measureText(txt).width, bx = [x - 1, y - fs, x + bw + 1, y + 3];
    if (!isCur && boxes.some(o => bx[0] < o[2] && bx[2] > o[0] && bx[1] < o[3] && bx[3] > o[1])) continue;
    boxes.push(bx); g.fillStyle = isCur ? "#ff5a1f" : toks[p.meta]?.prompt ? "#6c757d" : "#12161a";
    g.fillText(txt, x, y);
  }
}
$("wmLayer").oninput = () => { $("wmLayerV").textContent = $("wmLayer").value; wmDirty = true; drawWordMap(); };
renderNext();
new ResizeObserver(() => drawWordMap()).observe($("wm"));
let wmT = 0; const wmSoon = () => { wmDirty = true; clearTimeout(wmT); wmT = setTimeout(drawWordMap, 120); };

// ---- text ----
// chat-template plumbing (<|im_start|>user, <think></think> and the newlines around them) is hidden from the text view
const isTag = p => /^<\|.*\|>$/.test(p.trim()) || /^<\/?think>$/.test(p.trim());
function hiddenToks() {
  const h = new Set();
  toks.forEach((t, i) => {
    if (isTag(t.piece)) h.add(i);
    else if (i > 0 && /<\|im_start\|>/.test(toks[i - 1].piece)) h.add(i);             // role name after <|im_start|>
    else if (!t.piece.trim() && (h.has(i - 1) || (toks[i + 1] && isTag(toks[i + 1].piece)))) h.add(i);
  });
  return h;
}
function renderText() {
  const h = hiddenToks();
  let lastPrompt = false;
  $("out").innerHTML = toks.map((t, i) => { if (h.has(i)) return ""; const br = lastPrompt && !t.prompt ? "<br>" : ""; lastPrompt = t.prompt; return br + `<span class="tok${t.prompt ? " pr" : ""}${i === cur ? " cur" : ""}" data-i="${i}">${esc(t.piece)}</span>`; }).join("") + (busy ? '<span class="caret"></span>' : stoppedAtOnce ? '<br><span class="hint">It ended its answer straight away without writing anything: its top choice was the end-of-answer token.</span>' : "");
}
$("out").onclick = e => { const el = e.target.closest(".tok"); if (!el || busy) return; focusTok(+el.dataset.i); };
function renderNext() {
  const t = toks[cur], el = $("live");
  if (!t || !t.cands) { el.innerHTML = `<p class="hint">${t ? "This is a prompt word, so the model didn't choose it." : "Generate to see the top 5 choices at each word."}</p>`; return; }
  el.innerHTML = t.cands.map((c, k) => `<div class="cand static${c.id === t.id ? " chosen" : ""}"><span class="n">${String(k + 1).padStart(2, "0")}</span><span class="t">${esc(show(eng.piece(c.id)))}</span><span class="track"><span class="bar" style="transform:scaleX(${c.p.toFixed(4)})"></span></span><span class="p">${(c.p * 100).toFixed(1)}%</span></div>`).join("");
}
function focusTok(i, ms) {
  cur = i; const t = toks[i]; if (!t) return; renderNext();
  cloud?.show(t.z, t.path, { ms: ms ?? ($("slow").checked ? 2400 : 650) });
  $("kTok").textContent = show(t.piece).trim().slice(0, 12) || "·";
  for (const el of $("out").querySelectorAll(".tok")) el.classList.toggle("cur", +el.dataset.i === i);
  drawWordMap(); renderNeuron();
}
const special = p => /^<\|.*\|>$/.test(p.trim()) || !p.trim();
function addTok(id, trace, prompt, cands) {
  const piece = eng.piece(id), z = map.activity(trace.neurons), path = map.path(z);
  const i = toks.push({ id, piece, prompt, z, path, cands }) - 1;
  const role = i > 0 && /<\|im_start\|>/.test(toks[i - 1].piece);   // "user" / "assistant" after <|im_start|>
  if (!special(piece) && !isTag(piece) && !role) wm.add(piece, trace.resid, i);
  return i;
}
const wait = ms => new Promise(r => setTimeout(r, ms));

let stoppedAtOnce = false;
$("btnGen").onclick = async () => {
  if (busy) return; busy = true; stopFlag = false; stoppedAtOnce = false;
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
      const id = sample(logits, temp), cands = softmaxTop(logits, 5, 1).top;
      if (eng.stopIds.has(id)) { if (n === 0) stoppedAtOnce = true; break; }
      const r = await eng.stepTraced(id); logits = r.logits;
      const k = addTok(id, r.trace, false, cands); renderText(); focusTok(k); if (n % 4 === 0) wmSoon();
      $("kSpeed").textContent = ((n + 1) / ((performance.now() - t0) / 1000)).toFixed(1) + " tok/s";
      await wait(slow() ? 2600 : 120);
    }
    setStatus(knock.size ? `${pct(eng.surgeon.damagedFraction({ knock }))} off` : "Ready", knock.size ? "busy" : "on");
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

// ---------- surgery: switch blocks off in GPU memory, then rewrite so the cloud shows the damage ----------
const REF_TEXT = "The cat sat on the mat. Water boils at 100 degrees Celsius. The capital of France is Paris, and the capital of Japan is Tokyo. One, two, three, four, five, six, seven, eight.";
const knock = new Set(); let intactLoss = null, surgQ = Promise.resolve();
const pct = f => (f * 100 < 10 ? (f * 100).toFixed(1) : Math.round(f * 100)) + "%";
async function refLoss() {
  const ids = eng.encodePrompt(REF_TEXT, false);
  let logits = await eng.start(ids.slice(0, 1)), nll = 0;
  for (let i = 1; i < ids.length; i++) { nll -= logProb(logits, ids[i]); if (i < ids.length - 1) logits = await eng.step(ids[i]); }
  return nll / (ids.length - 1);
}
function buildBrain() {
  const g = $("brain"), ls = eng.layers;
  g.style.gridTemplateColumns = `auto repeat(${ls.length}, minmax(14px, 1fr))`;
  g.innerHTML = ['<span></span>', ...ls.map(l => `<span class="lh">${l.i}</span>`),
    '<span class="rl">Mixer</span>', ...ls.map(l => `<button class="cell${l.type === "full" ? " full" : ""}" data-k="L${l.i}.mixer" title="layer ${l.i} ${l.type === "full" ? "full attention" : "linear attention"}">${l.type === "full" ? "A" : ""}</button>`),
    '<span class="rl">MLP</span>', ...ls.map(l => `<button class="cell mlp" data-k="L${l.i}.mlp" title="layer ${l.i} MLP"></button>`)].join("");
}
function paintBrain() {
  for (const c of $("brain").querySelectorAll(".cell")) c.classList.toggle("off", knock.has(c.dataset.k));
  $("mOff").textContent = pct(eng.surgeon.damagedFraction({ knock })); $("btnRestore").disabled = !knock.size;
  cloud?.setKnocked(new Set([...knock].map(k => +k.slice(1, k.indexOf(".")))));
}
function surgery() {                              // queue edits so rapid clicks apply in order
  paintBrain(); stopFlag = true;
  surgQ = surgQ.then(async () => {
    while (busy) await wait(50);
    busy = true; for (const b of ["btnGen", "btnReplay"]) $(b).disabled = true;
    try {
      if (intactLoss === null) { setStatus("measuring the intact model", "busy"); intactLoss = await refLoss(); $("mLoss0").textContent = intactLoss.toFixed(3); }
      setStatus("editing weights", "busy"); await eng.surgeon.apply({ knock });
      setStatus("measuring loss", "busy"); const L = await refLoss();
      $("mLoss").textContent = L.toFixed(3) + (knock.size ? ` (${L >= intactLoss ? "+" : ""}${(L - intactLoss).toFixed(2)})` : "");
    } catch (e) { console.error(e); setStatus("edit failed: " + (e.message || e).slice(0, 50), "err"); busy = false; return; }
    busy = false; $("btnGen").disabled = false; $("btnGen").click();
  });
}
$("brain").onclick = e => { const c = e.target.closest(".cell"); if (!c || !eng) return; knock.has(c.dataset.k) ? knock.delete(c.dataset.k) : knock.add(c.dataset.k); surgery(); };
$("btnRestore").onclick = () => { knock.clear(); surgery(); };
if (params.has("stub")) $("btnLoad").textContent = "Load test model";
drawWordMap();
if (params.has("autoload")) $("btnLoad").click();
