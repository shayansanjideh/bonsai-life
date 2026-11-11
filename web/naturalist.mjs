// In-browser naturalist: Ternary-Bonsai-2-2B on WebGPU via PrismML's Talk engine (vendor/bonsai.mjs, Apache-2.0).
// Weights stream from Hugging Face once and are cached by the browser for later visits.
const REV = "a6032a5089ac27e96be3a80c7b42d1ad0bd89887";
const BASE = `https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf/resolve/${REV}/`;
const WEIGHTS = [
  { name: "Ternary-Bonsai-2-2B-PTQ1_0.gguf", size: 429183456 },
  { name: "Ternary-Bonsai-2-2B-mmproj-Q8_0.gguf", size: 364664352 },
];
export const TOTAL_MB = Math.round(WEIGHTS.reduce((a, w) => a + w.size, 0) / 1e6);
const COLORS = ["red", "orange", "yellow", "green", "blue", "purple"];

export async function webgpuAvailable() {
  try { return !!(navigator.gpu && (await navigator.gpu.requestAdapter())); } catch { return false; }
}

export async function loadModel({ onProgress, onStatus, signal, vision = true } = {}) {
  if (!(await webgpuAvailable())) throw new Error("This browser has no WebGPU. Try Chrome or Edge on a laptop, or Safari 26+.");
  const { Bonsai, CachedByteSource, UrlByteSource } = await import("./vendor/bonsai.mjs");
  const files = vision ? WEIGHTS : WEIGHTS.slice(0, 1);
  const total = files.reduce((a, w) => a + w.size, 0), stats = new Map(), sources = [];
  const t0 = performance.now();
  try {
    for (const w of files) {
      const url = BASE + w.name;
      let src = await UrlByteSource.open(url, { knownSize: w.size, signal });
      try {
        src = await CachedByteSource.wrap(src, url, {
          signal,
          onCache: s => { stats.set(url, s); onProgress?.(Math.min(1, [...stats.values()].reduce((a, x) => a + Math.max(x.fromNetwork, x.stored), 0) / total)); },
        });
      } catch { /* storage disabled: load uncached */ }
      sources.push(src);
    }
    const model = await Bonsai.load({
      model: sources[0], ...(vision ? { mmproj: sources[1], maxImageTokens: 1024 } : {}), contextLength: 4096, signal,
      onProgress: e => onStatus?.(e.message || "Preparing WebGPU…"),
    });
    return { model, loadMs: Math.round(performance.now() - t0) };
  } finally {
    for (const s of sources) s.close?.();
  }
}

export function promptFor(A, goal) {
  const rows = COLORS.map((c, i) => `${c}: ` + COLORS.map((d, j) => `${d}=${A[i * 6 + j].toFixed(2)}`).join(" ")).join("\n");
  return `You are a field biologist studying a particle-life petri dish. The image shows the current champion organism: colored particles, with faint streaks as motion trails.
The user's goal: "${goal}".
Attraction matrix (row color is pulled toward column color when positive, pushed away when negative; asymmetric pairs like A->B positive with B->A negative make chasers and swimmers):
${rows}

Reply with only a JSON object, no other text:
{"species_name": "<made-up two-word Latin binomial like Vermis fluorescens>", "field_notes": "<1-2 vivid sentences about what you see: clumps, worms, rings, swimmers, chasing>", "proposed_edits": [{"source_color": "<color>", "target_color": "<color>", "attraction": <number between -1 and 1>}, ...2 to 4 edits that push the dish toward the goal]}
Colors: red, orange, yellow, green, blue, purple.`;
}

// Bonsai has no JSON-schema mode in the browser engine, so parse leniently.
export function parseReply(text) {
  let obj = null;
  const s = text.indexOf("{"), e = text.lastIndexOf("}");
  if (s >= 0 && e > s) { try { obj = JSON.parse(text.slice(s, e + 1)); } catch { /* fall through */ } }
  const edits = [];
  const list = obj?.proposed_edits ?? [];
  for (const x of Array.isArray(list) ? list : []) {
    const a = COLORS.indexOf(String(x?.source_color).toLowerCase()), b = COLORS.indexOf(String(x?.target_color).toLowerCase());
    const v = Number(x?.attraction);
    if (a >= 0 && b >= 0 && Number.isFinite(v)) edits.push({ source: a, target: b, attraction: Math.max(-1, Math.min(1, v)) });
  }
  if (!edits.length) { // salvage "red -> blue 0.8" style edits from free text
    const re = new RegExp(`(${COLORS.join("|")})\\W{1,8}(${COLORS.join("|")})\\D{0,20}?(-?\\d*\\.\\d+|-?1|0)`, "gi");
    for (const m of text.matchAll(re)) { if (edits.length >= 4) break; edits.push({ source: COLORS.indexOf(m[1].toLowerCase()), target: COLORS.indexOf(m[2].toLowerCase()), attraction: Math.max(-1, Math.min(1, +m[3])) }); }
  }
  const name = String(obj?.species_name ?? (text.match(/species_name"?\s*:\s*"([^"]+)/)?.[1] ?? "")).slice(0, 60);
  const notes = String(obj?.field_notes ?? (text.match(/field_notes"?\s*:\s*"([^"]+)/)?.[1] ?? "")).slice(0, 400);
  return { species_name: name, field_notes: notes, edits: edits.slice(0, 4) };
}

export async function observe(model, blob, A, goal, signal) {
  const t0 = performance.now(); let raw = "";
  for await (const ev of model.chat([{ role: "user", content: promptFor(A, goal), images: [blob] }],
                                    { signal, thinking: false, maxNewTokens: 260, sampling: { temperature: 0.7 } })) {
    if (ev.type === "token" && ev.phase === "answer") raw += ev.text;
  }
  return { ...parseReply(raw), raw: raw.slice(0, 600), ms: Math.round(performance.now() - t0) };
}

// Text-only call (used by the living-cells narrator).
export async function ask(model, prompt, { maxNewTokens = 300, signal, onToken } = {}) {
  const t0 = performance.now(); let raw = "", tokens = 0;
  for await (const ev of model.chat([{ role: "user", content: prompt }], { signal, thinking: false, maxNewTokens, sampling: { temperature: 0.7 } })) {
    if (ev.type === "token" && ev.phase === "answer") { raw += ev.text; tokens++; onToken?.(raw); }
  }
  const ms = Math.round(performance.now() - t0);
  return { raw, ms, tokens, tps: tokens / Math.max(0.001, ms / 1000) };
}
export const TEXT_MB = Math.round(WEIGHTS[0].size / 1e6);

// Multi-turn chat (system + alternating user/assistant messages): small models follow a real conversation far better than a pasted transcript.
export async function chat(model, messages, { maxNewTokens = 120, signal, temperature = 0.8 } = {}) {
  const t0 = performance.now(); let raw = "", tokens = 0;
  for await (const ev of model.chat(messages, { signal, thinking: false, maxNewTokens, sampling: { temperature } })) {
    if (ev.type === "token" && ev.phase === "answer") { raw += ev.text; tokens++; }
  }
  const ms = Math.round(performance.now() - t0);
  return { raw, ms, tokens, tps: tokens / Math.max(0.001, ms / 1000) };
}
