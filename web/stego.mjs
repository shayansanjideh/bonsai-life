// Hide a short secret in a model-written tweet, and read it back.
//
// Both sides run the same model on the same prompt. At each token the model's top logits decide
// whether this step "carries" a digit: it does when the top k tokens are all plausible (within W of
// the best) and clearly separated from token k+1 (gap of at least G). The encoder then picks one of
// those k tokens, ordered by token id, so the digit never depends on the order of near-tied logits.
// The encoder only uses a step when every threshold clears by `delta`, so small logit differences
// between GPUs, browsers or kernels can't change how the reader classifies it. When a step is too close
// to call, the encoder writes a token outside every set the reader could compute, and the reader skips it.
//
// The payload is a big integer read in mixed radix (one digit of base k per carrying step):
//   N = binary("1" + huffman(secret) + crc16(secret)).
// Steps after N runs out carry digit 0, so the reader can keep reading to the end of the text.

export const DEFAULTS = Object.freeze({
  G: 0.8,        // min logit gap between the k-th and (k+1)-th candidate
  W: 3.0,        // max logit spread inside the carrying set (e^-3 ~ 5% of the top token)
  delta: 0.25,   // safety margin the encoder keeps from every threshold
  K: 4,          // largest carrying set (base 4 = 2 bits)
  temp: 0.8,     // sampling for free (non-carrying) steps
  topP: 0.9,
  maxTokens: 160,
  maxChars: 280,
  tries: 6,
});

export const TOPICS = [
  "coffee", "a rainy weekend", "learning to cook", "a morning walk", "a book you just finished", "the train ride home",
];
export const promptFor = topic =>
  `Write one casual tweet (two or three short sentences) about ${topic}. Plain text only: no hashtags, no emojis, no quotes.`;

// ---------- payload: fixed Huffman code + CRC16 ----------
const ALPHABET = " etaoinshrdlcumwfgypbvkjxqz0123456789.,?!'-:";
const FREQ = [180, 120, 90, 80, 75, 70, 67, 63, 61, 60, 43, 40, 28, 28, 27, 24, 24, 22, 20, 20, 19, 15, 10, 8, 2, 2, 1,
  12, 12, 10, 8, 8, 8, 8, 8, 8, 8, 12, 9, 4, 4, 4, 3, 3];
const CODE = (() => {
  let nodes = [...ALPHABET].map((c, i) => ({ w: FREQ[i], o: i, sym: c }));
  let order = ALPHABET.length;
  while (nodes.length > 1) {
    nodes.sort((a, b) => a.w - b.w || a.o - b.o);
    const [a, b] = nodes.splice(0, 2);
    nodes.push({ w: a.w + b.w, o: order++, kids: [a, b] });
  }
  const map = new Map();
  (function walk(n, p) { if (n.sym !== undefined) map.set(n.sym, p || "0"); else { walk(n.kids[0], p + "0"); walk(n.kids[1], p + "1"); } })(nodes[0], "");
  return map;
})();
const DECODE = new Map([...CODE].map(([c, b]) => [b, c]));

export function normalizeSecret(s) {
  return s.toLowerCase().replace(/[‘’]/g, "'").replace(/\s+/g, " ").trim();
}
export function badChars(s) { return [...new Set([...normalizeSecret(s)].filter(c => !CODE.has(c)))]; }

function crc16(str) { // CRC-16/CCITT-FALSE
  let c = 0xffff;
  for (const ch of str) {
    c ^= ch.charCodeAt(0) << 8;
    for (let i = 0; i < 8; i++) c = c & 0x8000 ? ((c << 1) ^ 0x1021) & 0xffff : (c << 1) & 0xffff;
  }
  return c;
}

export function packSecret(secret) {
  const s = normalizeSecret(secret);
  const bad = badChars(s);
  if (bad.length) throw new Error(`can't encode: ${bad.map(c => JSON.stringify(c)).join(" ")}`);
  if (!s) throw new Error("empty secret");
  const bits = [...s].map(c => CODE.get(c)).join("") + crc16(s).toString(2).padStart(16, "0");
  return { N: BigInt("0b1" + bits), bits: bits.length };
}

export function unpackSecret(N) {
  if (N <= 1n) return null;
  const bits = N.toString(2).slice(1);
  if (bits.length <= 16) return null;
  const body = bits.slice(0, -16), crc = parseInt(bits.slice(-16), 2);
  let out = "", cur = "";
  for (const b of body) { cur += b; const c = DECODE.get(cur); if (c !== undefined) { out += c; cur = ""; } if (cur.length > 16) return null; }
  if (cur || !out || crc16(out) !== crc) return null;
  return out;
}

// ---------- token filter (depends only on the tokenizer, so both sides agree) ----------
export function buildMask(lm) {
  const mask = new Uint8Array(lm.vocabSize);
  for (let id = 0; id < lm.vocabSize; id++) {
    let s;
    try { s = lm.decode([id]); } catch { continue; }
    if (!s || !/^[\x20-\x7e]+$/.test(s)) continue;     // printable ASCII only (drops partial UTF-8, newlines, emoji)
    if (/ {2}|<\||\|>|[@#*`_\\{}<>\[\]|~^]|http|www/.test(s)) continue; // no mentions, hashtags, markup, links, specials
    mask[id] = 1;
  }
  return mask;
}

// Top-n allowed tokens, sorted by logit (desc), ties broken by id so every device sorts the same way.
export function topN(logits, mask, n) {
  const ids = new Array(n).fill(-1), vals = new Array(n).fill(-Infinity);
  for (let i = 0; i < logits.length; i++) {           // ascending ids, strict ">": ties keep the lower id first
    const v = logits[i];
    if (!mask[i] || !(v > vals[n - 1])) continue;
    let j = n - 1;
    while (j > 0 && v > vals[j - 1]) { vals[j] = vals[j - 1]; ids[j] = ids[j - 1]; j--; }
    vals[j] = v; ids[j] = i;
  }
  return { ids, vals };
}

// Classify one step. With delta = 0 this is the reader's rule. With delta > 0 it also reports whether the
// answer could flip under logit noise smaller than delta/2 per logit.
export function classify(vals, p, delta = 0) {
  const possible = new Set();
  let ambiguous = false;
  for (let k = p.K; k >= 2; k--) {
    const gap = vals[k - 1] - vals[k], spread = vals[0] - vals[k - 1];
    if (!Number.isFinite(vals[k - 1])) continue;
    const yes = gap >= p.G + delta && spread <= p.W - delta;
    const no = gap < p.G - delta || spread > p.W + delta;
    if (yes) { possible.add(k); return { k, ambiguous, possible }; }
    if (!no) { ambiguous = true; possible.add(k); }
  }
  possible.add(0);
  return { k: 0, ambiguous, possible };
}

export function carrySet(ids, k) { return ids.slice(0, k).sort((a, b) => a - b); }

// Pick the highest-ranked token that is outside the top-m set on every device (gap >= delta below rank m-1).
function outside(top, m, delta) {
  for (let r = m; r < top.ids.length; r++) if (top.ids[r] >= 0 && Number.isFinite(top.vals[r]) && top.vals[m - 1] - top.vals[r] >= delta) return top.ids[r];
  return -1;
}

export function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

function sample(top, temp, topP, rng) {
  const m = top.vals[0], w = [];
  let z = 0;
  for (let i = 0; i < top.ids.length && top.ids[i] >= 0; i++) { const x = Math.exp((top.vals[i] - m) / temp); w.push(x); z += x; }
  let acc = 0, keep = 0;
  for (; keep < w.length; keep++) { acc += w[keep] / z; if (acc >= topP) { keep++; break; } }
  keep = Math.max(1, Math.min(keep, w.length));
  let s = 0; for (let i = 0; i < keep; i++) s += w[i];
  let r = rng() * s;
  for (let i = 0; i < keep; i++) { r -= w[i]; if (r <= 0) return top.ids[i]; }
  return top.ids[keep - 1];
}

const TOPW = 24;

// lm: { vocabSize, encode(text) -> ids, decode(ids) -> text, async begin(promptIds) -> logits, async step(id) -> logits, promptIds(topic) -> ids }
export async function encodeOnce(lm, mask, secret, topicIdx, p, seed, onStep) {
  let { N } = packSecret(secret);
  const payloadBits = N.toString(2).length - 1;
  const rng = mulberry32(seed);
  const ids = [];
  const st = { carry: 0, bits: 0, ambiguous: 0, free: 0, steps: 0 };
  let logits = await lm.begin(lm.promptIds(TOPICS[topicIdx]));
  for (let i = 0; i < p.maxTokens; i++) {
    const top = topN(logits, mask, TOPW);
    const c = classify(top.vals, p, p.delta);
    let tok;
    if (c.ambiguous) {
      tok = outside(top, Math.max(...c.possible), p.delta);
      if (tok < 0) return { ok: false, reason: "no safe token at a borderline step", ids, st };
      st.ambiguous++;
    } else if (c.k > 0) {
      const K = BigInt(c.k);
      tok = carrySet(top.ids, c.k)[Number(N % K)];
      N /= K; st.carry++; st.bits += Math.log2(c.k);
    } else { tok = sample(top, p.temp, p.topP, rng); st.free++; }
    ids.push(tok); st.steps++;
    onStep?.(ids, st, N);
    if (N === 0n && /[.!?]$/.test(lm.decode([tok])) && i >= 8) break;
    logits = await lm.step(tok);
  }
  const text = lm.decode(ids);
  if (N !== 0n) return { ok: false, reason: `ran out of tokens (${payloadBits} bits needed)`, text, ids, st };
  let re = [];
  try { re = lm.encode(text); } catch { /* treated as a mismatch */ }
  if (re.length !== ids.length || re.some((x, j) => x !== ids[j])) return { ok: false, reason: "text re-tokenizes differently", text, ids, st };
  if (text.length > p.maxChars) return { ok: false, reason: `too long (${text.length} chars)`, text, ids, st };
  return { ok: true, text, ids, st, payloadBits };
}

export async function encodeSecret(lm, mask, secret, topicIdx, p = DEFAULTS, onTry) {
  let best = null;
  for (let t = 0; t < p.tries; t++) {
    const r = await encodeOnce(lm, mask, secret, topicIdx, p, 1000 + t);
    r.try = t + 1;
    onTry?.(r);
    if (r.ok) return r;
    best = r;
  }
  return best;
}

// Read the digits of `text` under one topic's prompt.
export async function readDigits(lm, mask, ids, topicIdx, p = DEFAULTS) {
  let N = 0n, M = 1n, logits = await lm.begin(lm.promptIds(TOPICS[topicIdx]));
  for (let i = 0; i < ids.length; i++) {
    const top = topN(logits, mask, TOPW);
    const c = classify(top.vals, p, 0);
    if (c.k > 0) {
      const set = carrySet(top.ids, c.k), d = set.indexOf(ids[i]);
      if (d >= 0) { N += BigInt(d) * M; M *= BigInt(c.k); }
    }
    if (i + 1 < ids.length) logits = await lm.step(ids[i]);
  }
  return N;
}

export async function decodeText(lm, mask, text, p = DEFAULTS, onTopic) {
  const ids = lm.encode(text);
  for (let t = 0; t < TOPICS.length; t++) {
    onTopic?.(t);
    const secret = unpackSecret(await readDigits(lm, mask, ids, t, p));
    if (secret !== null) return { secret, topic: t };
  }
  return null;
}
