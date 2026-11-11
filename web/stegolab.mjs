// De-risking experiments for the secret-message page. Everything here works with any `lm` that
// matches the interface in stego.mjs, so it runs on the stub model in node and on Bonsai in a browser.
import * as S from "./stego.mjs";

// Fixed cover text used for fingerprints: every device replays exactly these tokens.
export const FP_TEXT = "I finally tried the little coffee place by the station this morning. The cup was strong, the room was quiet, and I stayed way longer than I planned.";
const FP_TOP = 8;

// Replay FP_TEXT under topic 0 and record the top logits at each step.
export async function fingerprint(lm, mask) {
  const text = lm.fpText ?? FP_TEXT, ids = lm.encode(text), steps = [];
  let logits = await lm.begin(lm.promptIds(S.TOPICS[0]));
  for (let i = 0; i < ids.length; i++) {
    const t = S.topN(logits, mask, FP_TOP);
    steps.push({ ids: t.ids, vals: t.vals.map(v => Math.round(v * 1e4) / 1e4) });
    if (i + 1 < ids.length) logits = await lm.step(ids[i]);
  }
  return { text, ids, steps };
}

// Compare two fingerprints: how far apart are the logits, and would any carrying decision flip?
export function compareFingerprints(a, b, p = S.DEFAULTS) {
  if (a.ids.join() !== b.ids.join()) return { error: "different tokenization of the fixed text" };
  let maxDiff = 0, sumDiff = 0, n = 0, setFlips = 0, classFlips = 0, safeFlips = 0;
  for (let i = 0; i < a.steps.length; i++) {
    const A = a.steps[i], B = b.steps[i];
    for (let j = 0; j < A.ids.length; j++) {
      const k = B.ids.indexOf(A.ids[j]);
      if (k >= 0) { const d = Math.abs(A.vals[j] - B.vals[k]); maxDiff = Math.max(maxDiff, d); sumDiff += d; n++; }
    }
    if (A.ids.slice(0, 4).sort().join() !== B.ids.slice(0, 4).sort().join()) setFlips++;
    const ca = S.classify(A.vals, { ...p, K: Math.min(p.K, FP_TOP - 1) }, 0), cb = S.classify(B.vals, { ...p, K: Math.min(p.K, FP_TOP - 1) }, 0);
    if (ca.k !== cb.k) {
      classFlips++;
      // A flip only matters if the encoder would have used that step: was A firmly decided?
      const firm = S.classify(A.vals, { ...p, K: Math.min(p.K, FP_TOP - 1) }, p.delta);
      if (!firm.ambiguous) safeFlips++;
    }
  }
  return { steps: a.steps.length, maxDiff: +maxDiff.toFixed(4), meanDiff: +(sumDiff / Math.max(1, n)).toFixed(5), top4SetChanges: setFlips, classFlips, flipsTheMarginMissed: safeFlips };
}

// Same device, different kernels: logits after a one-shot prefill of (prompt + first i tokens) vs the
// step-by-step decode path. Proxy for cross-device noise when only one device is at hand.
export async function prefillVsDecode(lm, mask, positions = [4, 10, 16, 22, 28]) {
  const fp = await fingerprint(lm, mask), prompt = lm.promptIds(S.TOPICS[0]);
  let maxDiff = 0;
  for (const i of positions) {
    if (i >= fp.ids.length) continue;
    const logits = await lm.begin([...prompt, ...fp.ids.slice(0, i)]);
    const t = S.topN(logits, mask, FP_TOP), ref = fp.steps[i];
    for (let j = 0; j < t.ids.length; j++) { const k = ref.ids.indexOf(t.ids[j]); if (k >= 0) maxDiff = Math.max(maxDiff, Math.abs(t.vals[j] - ref.vals[k])); }
  }
  return { positions: positions.length, maxDiff: +maxDiff.toFixed(4) };
}

// Sample natural tweets and record the top logits at every step (for offline capacity estimates).
export async function sampleNatural(lm, mask, n = 6, maxTokens = 90, onProgress) {
  const out = [];
  for (let s = 0; s < n; s++) {
    const topic = s % S.TOPICS.length, rng = S.mulberry32(500 + s), steps = [], ids = [];
    let logits = await lm.begin(lm.promptIds(S.TOPICS[topic]));
    for (let i = 0; i < maxTokens; i++) {
      const t = S.topN(logits, mask, 12);
      steps.push(t.vals.map(v => Math.round(v * 1e3) / 1e3));
      // temperature 0.8 / top-p 0.9 like the encoder's free steps
      const m = t.vals[0]; const w = t.vals.map(v => Math.exp((v - m) / 0.8)); const z = w.reduce((a, b) => a + b, 0);
      let r = rng() * z, tok = t.ids[0];
      for (let j = 0; j < w.length; j++) { r -= w[j]; if (r <= 0) { tok = t.ids[j]; break; } }
      ids.push(tok);
      onProgress?.(s, i);
      if (/[.!?]$/.test(lm.decode([tok])) && i > 30) break;
      logits = await lm.step(tok);
    }
    out.push({ topic, text: lm.decode(ids), steps });
  }
  return out;
}

export const GRID = [];
for (const G of [0.5, 0.8, 1.2]) for (const W of [2, 3, 4]) for (const delta of [0.1, 0.25]) for (const K of [2, 4, 8]) GRID.push({ G, W, delta, K });

// For each parameter set: bits per token, and tokens needed for a 55-bit payload ("meet at 7").
export function capacity(samples, payloadBits = 55) {
  return GRID.map(g => {
    let bits = 0, toks = 0, amb = 0;
    for (const s of samples) for (const vals of s.steps) {
      const K = Math.min(g.K, vals.length - 1), c = S.classify(vals, { ...g, K }, g.delta);
      toks++;
      if (c.ambiguous) amb++; else if (c.k > 0) bits += Math.log2(c.k) * 0.98; // digit-0 padding wastes a little at the end
    }
    const bpt = bits / Math.max(1, toks);
    return { ...g, bitsPerToken: +bpt.toFixed(3), ambiguousPct: +(100 * amb / Math.max(1, toks)).toFixed(1), tokensFor55: bpt ? Math.ceil(payloadBits / bpt) : Infinity };
  }).sort((a, b) => b.bitsPerToken - a.bitsPerToken);
}

// Encode then decode on this device.
export async function roundTrips(lm, mask, p, n = 4, onResult) {
  const secrets = ["meet at 7", "bring snacks", "the code is 4417", "yes"], res = [];
  for (let i = 0; i < n; i++) {
    const t0 = performance.now(), secret = secrets[i % secrets.length], topic = i % S.TOPICS.length;
    const e = await S.encodeSecret(lm, mask, secret, topic, p);
    const t1 = performance.now();
    const d = e.ok ? await S.decodeText(lm, mask, e.text, p) : null;
    const r = { secret, topic, encodeOk: e.ok, reason: e.reason, tries: e.try, text: e.text, chars: e.text?.length, tokens: e.ids?.length,
      bits: e.payloadBits, stats: e.st, decoded: d?.secret ?? null, correct: d?.secret === secret, encodeMs: Math.round(t1 - t0), decodeMs: Math.round(performance.now() - t1) };
    res.push(r); onResult?.(r);
  }
  return res;
}
