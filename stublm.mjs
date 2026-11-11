// Tiny fake language model for tests and the ?stub page mode (no GPU, no weights).
import { mulberry32 } from "./stego.mjs";
const WORDS = ("the a of to and in is it that was for on with as at by this from my i you we they he she but not or have had " +
  "coffee rain walk book train home morning cook today really just so very good new long quiet cold warm cup slow day night " +
  "week time again still back out up over into after before like feel felt love made read finished took went came about little").split(" ");
const PIECES = [".", ",", "!", "?", ...WORDS, ...WORDS.map(w => " " + w), "é", "#tag", " @you"];
const V = PIECES.length;

function hash(...xs) { let h = 2166136261; for (const x of xs) { h ^= x; h = Math.imul(h, 16777619); } return h >>> 0; }

export function makeLM(noise = 0, noiseSeed = 7) {
  let ctx = [], rngN = mulberry32(noiseSeed);
  const logitsFor = () => {
    const r = mulberry32(hash(...ctx));
    const out = new Float32Array(V).fill(-12);
    for (let j = 0; j < 14; j++) out[Math.floor(r() * V)] = -Math.log(-Math.log(r() + 1e-9)) * 1.6; // Gumbel-ish heads
    if (noise) for (let i = 0; i < V; i++) out[i] += (rngN() * 2 - 1) * noise;
    return out;
  };
  return {
    vocabSize: V,
    fpText: "the coffee was good today. i took a long walk in the rain and read my new book on the train home.",
    decode: ids => ids.map(i => PIECES[i]).join(""),
    encode(text) { // greedy longest match
      const out = []; let i = 0;
      while (i < text.length) {
        let best = -1, len = 0;
        PIECES.forEach((p, id) => { if (p.length > len && text.startsWith(p, i)) { best = id; len = p.length; } });
        if (best < 0) throw new Error("untokenizable at " + i);
        out.push(best); i += len;
      }
      return out;
    },
    promptIds: topic => [...topic].map(c => c.charCodeAt(0) % V),
    async begin(ids) { ctx = [...ids]; return logitsFor(); },
    async step(id) { ctx.push(id); return logitsFor(); },
  };
}

