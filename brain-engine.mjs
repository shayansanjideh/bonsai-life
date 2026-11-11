// Engine layer for the brain explorer (brain.html).
// RealEngine drives Ternary-Bonsai-2-2B through PrismML's WebGPU engine (vendor/bonsai.mjs) one token at a time,
// reading the full next-token distribution after every step. Surgeon edits the model's ternary weights in GPU
// memory. StubEngine is a tiny fake model with the same interface, for testing the page without a GPU.

// ---------------------------------------------------------------------------------------------------------------
// Surgeon: zero (or sign-flip) groups of 128 ternary weights by rewriting their f16 group scale on the GPU.
//
// How Bonsai stores a ternary matrix on the GPU (read from vendor/bonsai.mjs):
//   * "lut2" format: 2-bit codes in pack.bits, and one f16 scale per 128 weights in pack.scales, two scales per u32
//     (scale s lives in u32 s>>1, low half when s is even). A tensor at blockOffset B (in 32-weight blocks) owns
//     scale words [B/8, (B+count)/8).
//   * "ptq1" format: 28-byte superblocks per 128 weights (26 bytes of base-3 packed trits, then the f16 scale in
//     bytes 26-27). A tensor at sbOffset S owns bytes [28S, 28(S+count)); the scale is the high half of word 7k+6.
// Setting a group's scale to 0 sets all 128 of its weights to 0. Flipping the sign bit negates them.
// Before the first edit we copy every scale word of every ternary tensor into a backup buffer, so "restore" is exact.

const SURGERY_WGSL = /* wgsl */ `
struct P { wordBase: u32, stride: u32, nWords: u32, mode: u32, thresh: u32, seed: u32, flags: u32, pad: u32 }
@group(0) @binding(0) var<storage, read_write> w: array<u32>;
@group(0) @binding(1) var<storage, read_write> bk: array<u32>;
@group(0) @binding(2) var<uniform> p: P;
fn hash(x: u32) -> u32 {
  var h = x * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return (h >> 22u) ^ h;
}
fn hit(g: u32) -> bool {
  if ((p.flags & 2u) != 0u) { return true; }
  return hash(g ^ p.seed) < p.thresh;
}
fn edit(h: u32) -> u32 { if (p.mode == 1u) { return 0u; } return h ^ 0x8000u; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 64u;
  if (i >= p.nWords) { return; }
  let a = p.wordBase + i * p.stride;
  if (p.mode == 9u) { bk[i] = w[a]; return; }          // backup
  let o = bk[i];
  var lo = o & 0xFFFFu;
  var hi = o >> 16u;
  if (p.mode != 0u) {                                 // 0 = restore, 1 = zero, 2 = flip
    if ((p.flags & 1u) != 0u) {                       // lut2: two scales per word
      if (hit(2u * i)) { lo = edit(lo); }
      if (hit(2u * i + 1u)) { hi = edit(hi); }
    } else {                                          // ptq1: scale is the high half only
      if (hit(i)) { hi = edit(hi); }
    }
  }
  w[a] = lo | (hi << 16u);
}`;

const MIXER_FULL = ["attn_q", "attn_k", "attn_v", "attn_output"];
const MIXER_LINEAR = ["attn_qkv", "attn_gate", "ssm_out"];
const MLP = ["ffn_gate", "ffn_up", "ffn_down"];

function hash32(x) { // same as WGSL hash, for seeds
  let h = (Math.imul(x >>> 0, 747796405) + 2891336453) >>> 0;
  h = Math.imul(((h >>> ((h >>> 28) + 4)) ^ h) >>> 0, 277803737) >>> 0;
  return ((h >>> 22) ^ h) >>> 0;
}

export class Surgeon {
  // weights: bonsai.weights; config: bonsai.config (blockCount, layers[]).
  constructor(device, weights, config) {
    this.device = device;
    this.format = weights.format ?? (weights.ptq1 ? "ptq1" : "lut2");
    this.tensors = [];
    this.layers = Array.from({ length: config.blockCount }, (_, i) => ({ i, type: config.layers?.[i] ?? "?", mixer: [], mlp: [] }));
    const align = device.limits.minStorageBufferOffsetAlignment || 256;
    const add = (name, layer, part, buffer, byteOff, nWords, stride, lut2, weightsCount) => {
      // Bind a 256-aligned window of the weight buffer that covers this tensor's words.
      const first = byteOff, last = byteOff + ((nWords - 1) * stride + 1) * 4;
      const bindOff = Math.floor(first / align) * align, bindSize = Math.ceil((last - bindOff) / 4) * 4;
      const t = { id: this.tensors.length, name, layer, part, buffer, bindOff, bindSize, wordBase: (first - bindOff) / 4, nWords, stride, lut2, weights: weightsCount, key: "intact" };
      this.tensors.push(t); this.layers[layer]?.[part].push(t);
    };
    for (let L = 0; L < config.blockCount; L++) {
      const mixer = config.layers?.[L] === "full" ? MIXER_FULL : MIXER_LINEAR;
      for (const [part, list] of [["mixer", mixer], ["mlp", MLP]]) for (const base of list) {
        const name = `blk.${L}.${base}.weight`;
        if (this.format === "ptq1") {
          const pl = weights.ptq1?.placement?.get(name); if (!pl) continue;
          const pack = weights.ptq1.packs[pl.packIndex];
          add(name, L, part, pack.buffer, pl.sbOffset * 28 + 24, pl.sbCount, 7, false, pl.sbCount * 128);
        } else {
          const pl = weights.lut2Placement?.get(name); if (!pl) continue;
          const pack = weights.lut2Packs[pl.packIndex];
          if (pl.blockOffset % 8 || pl.blockCount % 8) throw new Error(`${name}: unexpected block alignment`);
          add(name, L, part, pack.scales, pl.blockOffset / 2, pl.blockCount / 8, 1, true, pl.blockCount * 32);
        }
      }
    }
    if (!this.tensors.length) throw new Error(`no ternary tensors found (weight format "${this.format}")`);
    this.totalWeights = this.tensors.reduce((a, t) => a + t.weights, 0);
    this.ready = null;
  }

  async init() {
    if (this.ready) return this.ready;
    this.ready = (async () => {
      const d = this.device;
      this.pipeline = await d.createComputePipelineAsync({ layout: "auto", compute: { module: d.createShaderModule({ code: SURGERY_WGSL }), entryPoint: "main" } });
      for (const t of this.tensors) t.backup = d.createBuffer({ size: Math.max(16, t.nWords * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, label: `surgeon.backup.${t.name}` });
      const enc = d.createCommandEncoder({ label: "surgeon.backup" });
      for (const t of this.tensors) this.#pass(enc, t, { mode: 9 });
      d.queue.submit([enc.finish()]);
      await d.queue.onSubmittedWorkDone();
    })();
    return this.ready;
  }

  #pass(enc, t, { mode, thresh = 0, seed = 0, knock = false }) {
    const d = this.device;
    const u = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
    new Uint32Array(u.getMappedRange()).set([t.wordBase, t.stride, t.nWords, mode, thresh >>> 0, seed >>> 0, (t.lut2 ? 1 : 0) | (knock ? 2 : 0), 0]);
    u.unmap();
    const bg = d.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: t.buffer, offset: t.bindOff, size: t.bindSize } },
      { binding: 1, resource: { buffer: t.backup } },
      { binding: 2, resource: { buffer: u } },
    ] });
    const groups = Math.ceil(t.nWords / 64), maxX = d.limits.maxComputeWorkgroupsPerDimension || 65535;
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.min(groups, maxX), Math.ceil(groups / maxX), 1);
    pass.end();
    (this._uniforms ??= []).push(u);
  }

  // spec: { knock: Set of "L<i>.mixer" / "L<i>.mlp", random: 0..1 fraction of groups, mode: "zero" | "flip", seed }
  async apply(spec = {}) {
    await this.init();
    const knock = spec.knock ?? new Set(), p = Math.max(0, Math.min(1, spec.random ?? 0));
    const mode = spec.mode === "flip" ? 2 : 1, seed = spec.seed ?? 1;
    const enc = this.device.createCommandEncoder({ label: "surgeon.apply" });
    let n = 0;
    for (const t of this.tensors) {
      const knocked = knock.has(`L${t.layer}.${t.part}`);
      const key = knocked ? `knock:${mode}` : p > 0 ? `rand:${p}:${mode}:${seed}` : "intact";
      if (key === t.key) continue;
      if (knocked) this.#pass(enc, t, { mode, knock: true });
      else if (p > 0) this.#pass(enc, t, { mode, thresh: p >= 1 ? 0xFFFFFFFF : Math.floor(p * 4294967296), seed: hash32(seed * 7919 + t.id + 1) });
      else this.#pass(enc, t, { mode: 0 });
      t.key = key; n++;
    }
    if (n) { this.device.queue.submit([enc.finish()]); await this.device.queue.onSubmittedWorkDone(); }
    for (const u of this._uniforms ?? []) u.destroy(); this._uniforms = [];
    return n;
  }

  restore() { return this.apply({}); }

  // Fraction of all ternary weights currently switched off (exact for knockouts, expected value for random damage).
  damagedFraction(spec = {}) {
    const knock = spec.knock ?? new Set(), p = spec.random ?? 0;
    let w = 0;
    for (const t of this.tensors) w += knock.has(`L${t.layer}.${t.part}`) ? t.weights : p * t.weights;
    return w / this.totalWeights;
  }

  // Test helper: read back the scale words of a tensor.
  async readWords(t) {
    const d = this.device, size = t.bindSize;
    const st = d.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = d.createCommandEncoder(); enc.copyBufferToBuffer(t.buffer, t.bindOff, st, 0, size); d.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const all = new Uint32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy();
    return Uint32Array.from({ length: t.nWords }, (_, i) => all[t.wordBase + i * t.stride]);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Shared helpers

export function softmaxTop(logits, k = 5, temperature = 1) {
  const T = temperature > 0 ? temperature : 1;
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp((logits[i] - max) / T);
  const top = [];
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i];
    if (top.length < k) { top.push(i); top.sort((a, b) => logits[b] - logits[a]); }
    else if (v > logits[top[k - 1]]) { top[k - 1] = i; top.sort((a, b) => logits[b] - logits[a]); }
  }
  const logZ = Math.log(sum) + max / T;
  return { top: top.map(id => ({ id, p: Math.exp(logits[id] / T - logZ) })), logZ, T };
}

export function logProb(logits, id) { // natural-log probability at temperature 1
  let max = -Infinity; for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0; for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - max);
  return logits[id] - max - Math.log(sum);
}

export function sample(logits, temperature, rng = Math.random) {
  if (!(temperature > 0)) { let b = 0; for (let i = 1; i < logits.length; i++) if (logits[i] > logits[b]) b = i; return b; }
  let max = -Infinity; for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const w = new Float64Array(logits.length); let sum = 0;
  for (let i = 0; i < logits.length; i++) { w[i] = Math.exp((logits[i] - max) / temperature); sum += w[i]; }
  let r = rng() * sum;
  for (let i = 0; i < w.length; i++) { r -= w[i]; if (r <= 0) return i; }
  return w.length - 1;
}

// ---------------------------------------------------------------------------------------------------------------
// RealEngine

export class RealEngine {
  constructor(bonsai) {
    this.b = bonsai; this.m = bonsai.model; this.cfg = bonsai.config;
    this.kind = "real";
    const stops = new Set(this.cfg.eosTokenIds ?? []);
    for (const s of ["<|im_end|>", "<|endoftext|>"]) { const id = bonsai.tokenizer.tokenToId?.(s); if (id !== undefined) stops.add(id); }
    this.stopIds = stops;
    const device = this.m.device ?? bonsai.gpu?.device;
    this.surgeon = new Surgeon(device, bonsai.weights, this.cfg);
    this.layers = this.surgeon.layers.map(l => ({ i: l.i, type: l.type }));
    this.pieceCache = new Map();
  }
  get name() { return "Ternary-Bonsai-2-2B (WebGPU)"; }
  encodePrompt(text, chat) {
    return chat ? this.b.encode(this.b.renderPrompt([{ role: "user", content: text }], { thinking: false })) : this.b.encode(text);
  }
  decode(ids) { return this.b.decode(ids); }
  piece(id) { let s = this.pieceCache.get(id); if (s === undefined) { s = this.b.decode([id]); this.pieceCache.set(id, s); } return s; }
  async start(ids) { this.m.reset(); await this.m.prefill(ids); return this.m.readLogits(); }
  async step(id) { await this.m.decodeStep(id); return this.m.readLogits(); }
  get position() { return this.m.position ?? this.m.cache?.seqLength; }
}

export async function loadRealEngine(opts) {
  const NAT = await import("./naturalist.mjs");
  const { model, loadMs } = await NAT.loadModel({ ...opts, vision: false });
  const eng = new RealEngine(model);
  eng.loadMs = loadMs;
  return eng;
}

// ---------------------------------------------------------------------------------------------------------------
// StubEngine: a fake 24-layer "model" over a small word vocabulary. Bigram logits from a fixed seed; surgery adds
// noise proportional to the damaged fraction, so every UI path (forking, surgery, scans, the game) can be exercised
// without WebGPU. Open brain.html?stub=1.

const STUB_WORDS = ("<eos> . , the a cat dog sat on mat and then ran to park big small red blue happy sad tree sun moon " +
  "water boils at 100 degrees Celsius capital of France is Paris Japan Tokyo one two three four five six seven eight " +
  "I am robot banana purple elephant tell me about your day today was good walked saw friend ate lunch it very " +
  "nice sky rain cloud bird sang song we you they said hello world once upon time there lived king queen castle").split(" ");

function mulberry(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

export class StubEngine {
  constructor({ layers = 24, delayMs = 15 } = {}) {
    this.kind = "stub"; this.delayMs = delayMs;
    this.vocab = STUB_WORDS; this.V = STUB_WORDS.length;
    this.stopIds = new Set([0]);
    this.layers = Array.from({ length: layers }, (_, i) => ({ i, type: (i + 1) % 4 === 0 ? "full" : "linear" }));
    const r = mulberry(7);
    this.table = Array.from({ length: this.V }, () => Float32Array.from({ length: this.V }, () => (r() - 0.5) * 6));
    // make the reference sentence and a few phrases likely
    const seq = "the cat sat on the mat . water boils at 100 degrees Celsius . the capital of France is Paris . one two three four five six seven eight".split(" ");
    for (let i = 0; i + 1 < seq.length; i++) this.table[this.id(seq[i])][this.id(seq[i + 1])] += 8;
    this.spec = {}; this.ctx = [];
    const self = this;
    this.surgeon = {
      layers: this.layers.map(l => ({ ...l, mixer: [{}], mlp: [{}] })),
      totalWeights: 1.7e9,
      async init() {},
      async apply(spec) { self.spec = spec ?? {}; return 1; },
      restore() { self.spec = {}; return 1; },
      damagedFraction(spec = {}) { const k = spec.knock?.size ?? 0; return Math.min(1, k / (2 * layers) + (spec.random ?? 0) * (1 - k / (2 * layers))); },
    };
  }
  get name() { return "Stub model (no GPU)"; }
  id(w) { const i = this.vocab.indexOf(w); return i < 0 ? 1 : i; }
  encodePrompt(text) { return text.trim().toLowerCase().split(/\s+/).filter(Boolean).map(w => this.id(w.replace(/[^a-z0-9]/g, "") || w)); }
  decode(ids) { return ids.map(i => this.piece(i)).join(""); }
  piece(id) { return id === 0 ? "" : " " + this.vocab[id]; }
  #logits() {
    const last = this.ctx[this.ctx.length - 1] ?? 1, row = this.table[last];
    const dmg = this.surgeon.damagedFraction(this.spec), critical = this.spec.knock?.has("L0.mlp") ? 1 : 0;
    const r = mulberry(last * 131 + this.ctx.length * 7 + Math.round(dmg * 1000) + (this.spec.seed ?? 0));
    const out = new Float32Array(this.V);
    for (let i = 0; i < this.V; i++) out[i] = row[i] * (1 - Math.min(0.95, dmg * 1.6 + critical * 0.8)) + (r() - 0.5) * 10 * (dmg + critical);
    if (this.ctx.length > 30) out[0] += 6;
    return out;
  }
  async #wait() { if (this.delayMs) await new Promise(r => setTimeout(r, this.delayMs)); }
  async start(ids) { this.ctx = [...ids]; await this.#wait(); return this.#logits(); }
  async step(id) { this.ctx.push(id); await this.#wait(); return this.#logits(); }
  get position() { return this.ctx.length; }
}
