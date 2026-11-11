// Stub-model tests for web/stego.mjs: no GPU or weights needed.
// The stub's "reader" copy adds logit noise, standing in for a different GPU/browser.
import * as S from "../web/stego.mjs";
import assert from "node:assert/strict";

import { makeLM } from "../web/stublm.mjs";
const V = makeLM().vocabSize;

// payload codec
for (const s of ["meet at 7", "the eagle lands at noon!", "x", "pin 0427"]) {
  const { N } = S.packSecret(s);
  assert.equal(S.unpackSecret(N), S.normalizeSecret(s));
}
assert.equal(S.unpackSecret(S.packSecret("meet at 7").N ^ 4n), null, "crc catches a flipped bit");
assert.throws(() => S.packSecret("café"));
console.log("payload ok; 'meet at 7' =", S.packSecret("meet at 7").bits, "bits");

const P = { ...S.DEFAULTS, maxChars: 4000, maxTokens: 600 };
async function trial(noise, delta, n = 40) {
  const enc = makeLM(0), mask = S.buildMask(enc);
  assert.equal(mask[V - 1], 0); assert.equal(mask[V - 2], 0); assert.equal(mask[V - 3], 0);
  let ok = 0, wrong = 0, encFail = 0, toks = 0;
  for (let t = 0; t < n; t++) {
    const secret = ["meet at 7", "bring the blue folder", "no", "abort, abort"][t % 4];
    const r = await S.encodeSecret(enc, mask, secret, t % S.TOPICS.length, { ...P, delta });
    if (!r.ok) { encFail++; continue; }
    toks += r.ids.length;
    const dec = await S.decodeText(makeLM(noise, 100 + t), mask, r.text, P);
    if (dec?.secret === S.normalizeSecret(secret) && dec.topic === t % S.TOPICS.length) ok++; else wrong++;
  }
  return { noise, delta, ok, wrong, encFail, avgTokens: +(toks / Math.max(1, ok + wrong)).toFixed(1) };
}

const rows = [];
for (const [noise, delta] of [[0, 0], [0, 0.25], [0.05, 0], [0.05, 0.25], [0.1, 0.25], [0.2, 0.25]]) rows.push(await trial(noise, delta));
console.table(rows);
const get = (n, d) => rows.find(r => r.noise === n && r.delta === d);
assert.equal(get(0, 0.25).wrong, 0, "no noise: always decodes");
assert.equal(get(0.05, 0.25).wrong, 0, "noise below delta/2: always decodes");
assert.equal(get(0.1, 0.25).wrong, 0, "noise at delta/2 (worst-case bound): always decodes");
assert.ok(get(0.05, 0).wrong > 0, "control: without the margin, noise breaks decoding");
console.log("all stego tests passed");

// lab helpers on the stub (FP_TEXT isn't in the stub vocab, so swap in a stub-friendly text)
const L = await import("../web/stegolab.mjs");
{
  const lm = makeLM(0), mask = S.buildMask(lm);
  const nat = await L.sampleNatural(lm, mask, 4, 60);
  const cap = L.capacity(nat);
  assert.ok(cap.length === L.GRID.length && cap[0].bitsPerToken > 0);
  console.log("stub capacity, best 3:", cap.slice(0, 3));
  const rt = await L.roundTrips(lm, mask, P, 4);
  assert.ok(rt.every(r => r.correct), "lab round trips decode");
}
console.log("lab tests passed");
