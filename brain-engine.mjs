// Engine layer for the brain explorer (brain.html, and the older layers.html).
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
  // Traced generation: every token (after the first) goes through decodeStep with the tracer on.
  get traceDims() { return { L: this.cfg.blockCount, H: this.cfg.hiddenSize, F: this.cfg.ffnSize }; }
  async startTraced(ids, onTrace) {
    this.tracer ??= new Tracer(this.m, this.cfg);
    this.m.reset(); await this.m.prefill(ids.slice(0, 1));
    for (let i = 1; i < ids.length; i++) await onTrace?.(i, await this.tracer.step(ids[i]));
    return this.m.readLogits();
  }
  async stepTraced(id) { const trace = await this.tracer.step(id); return { trace, logits: await this.m.readLogits() }; }
}

// ---------------------------------------------------------------------------------------------------------------
// Tracer: reads what happens inside the model, one token at a time. After each layer's MLP (the `down` projection,
// which also adds into the residual stream) it copies two GPU buffers into a capture buffer:
//   neurons: the MLP's hidden activations silu(gate)·up, ffnSize floats (6,144 on Bonsai 2B), before rotation
//   resid:   the residual stream after the layer, hiddenSize floats (2,048)
// It works by wrapping the decode graph's record() so every program still runs in the same order; the copies are
// extra commands in the same command encoder. Prefill uses a batched graph, so a traced prompt is fed one token at a
// time through decodeStep (slower, but every token gets a trace).

export class Tracer {
  constructor(model, cfg) {
    this.m = model; this.cfg = model.config ?? cfg; this.device = model.device;
    this.L = this.cfg.blockCount; this.H = this.cfg.hiddenSize; this.F = this.cfg.ffnSize;
    this.on = false; this.g = null;
  }
  async attach() {
    const g = await this.m.decode();
    if (g === this.g) return;
    const inter = g.arena?.buffers?.find(b => b.label === "decode.intermediate");
    if (!inter || !g.hidden?.buffer) throw new Error("tracing: this engine build has no decode.intermediate / hidden buffers");
    const after = new Map();
    for (const p of g.layerPrograms) { const m = /^decode\.L(\d+)\.down$/.exec(p.label ?? ""); if (m) after.set(p, +m[1]); }
    if (after.size !== this.L) throw new Error(`tracing: found ${after.size} MLP outputs for ${this.L} layers`);
    const row = (this.H + this.F) * 4, bytes = row * this.L;
    this.cap?.destroy(); this.staging?.destroy();
    this.cap = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label: "trace.capture" });
    this.staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: "trace.staging" });
    const self = this, hid = g.hidden, H4 = this.H * 4, F4 = this.F * 4, orig = g.record;
    g.record = function (r, n, a, s) {
      if (!self.on) return orig.call(this, r, n, a, s);
      const run = r.run;
      r.run = function (prog) {
        run.call(this, prog);
        const L = after.get(prog);
        if (L !== undefined) { this.copyBuffer(hid.buffer, hid.byteOffset, self.cap, L * row, H4); this.copyBuffer(inter, 0, self.cap, L * row + H4, F4); }
      };
      try { return orig.call(this, r, n, a, s); } finally { r.run = run; }
    };
    this.g = g; this.row = row; this.bytes = bytes;
  }
  async read() {
    const enc = this.device.createCommandEncoder({ label: "trace.read" });
    enc.copyBufferToBuffer(this.cap, 0, this.staging, 0, this.bytes);
    this.device.queue.submit([enc.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(this.staging.getMappedRange().slice(0)); this.staging.unmap();
    const { L, H, F } = this, neurons = new Float32Array(L * F), resid = new Float32Array(L * H), r = H + F;
    for (let l = 0; l < L; l++) { resid.set(all.subarray(l * r, l * r + H), l * H); neurons.set(all.subarray(l * r + H, (l + 1) * r), l * F); }
    return { neurons, resid };
  }
  async step(id) { await this.attach(); this.on = true; try { await this.m.decodeStep(id); } finally { this.on = false; } return this.read(); }
}

export async function loadRealEngine(opts) {
  const NAT = await import("./naturalist.mjs");
  const { model, loadMs } = await NAT.loadModel({ ...opts, vision: false });
  const eng = new RealEngine(model);
  eng.loadMs = loadMs;
  return eng;
}

// ---------------------------------------------------------------------------------------------------------------
// StubEngine: a scripted fake model for testing the page without a GPU (open brain.html?stub=1). It follows a few
// canned answers word by word with plausible alternatives, so screenshots look like real use. Damage adds noise and
// repetition in proportion to the fraction of weights switched off, and layer 0's MLP is made critical.

const SCRIPTS = [
  { match: /sky/i, text: "The sky looks blue because sunlight scatters off the tiny molecules in the air. Blue light has a shorter wavelength, so it scatters much more than red light, and that scattered blue reaches our eyes from every direction." },
  { match: /day/i, text: "My day was pretty good! I took a long walk in the park, had a banana for lunch, and spent the evening reading a book about the moon." },
  { match: /once upon/i, text: "Once upon a time there was a little robot who lived in a quiet town. Every morning it watered the garden and said hello to the birds." },
  { match: /banana/i, text: "Paris is the elegant capital of France, famous for the Eiffel Tower and its quiet cafes. Tokyo is the busy capital of Japan, full of neon lights and fast trains. A banana is a soft yellow fruit that is sweet and easy to peel." },
  { match: /capital/i, text: "The capital of France is Paris." },
  { match: /cat sat/i, text: "The cat sat on the mat. Water boils at 100 degrees Celsius. The capital of France is Paris, and the capital of Japan is Tokyo. One, two, three, four, five, six, seven, eight." },
];
const POOL = ("the a it and of to is was in that this because light air blue red sun sky so very some more most its from our every with on at they we I you one time little big then there day good about".split(" "));

function mulberry(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const words = t => t.trim().split(/\s+/).filter(Boolean);

export class StubEngine {
  constructor({ layers = 24, delayMs = 15 } = {}) {
    this.kind = "stub"; this.delayMs = delayMs;
    this.vocab = ["<eos>"]; this.index = new Map([["<eos>", 0]]);
    this.stopIds = new Set([0]);
    for (const sc of SCRIPTS) for (const w of words(sc.text)) this.id(w);
    for (const w of POOL) this.id(w);
    this.layers = Array.from({ length: layers }, (_, i) => ({ i, type: (i + 1) % 4 === 0 ? "full" : "linear" }));
    this.spec = {}; this.ctx = []; this.script = null; this.promptLen = 0;
    const self = this;
    this.surgeon = {
      layers: this.layers.map(l => ({ ...l, mixer: [{}], mlp: [{}] })),
      totalWeights: 1.372e9,
      async init() {},
      async apply(spec) { self.spec = spec ?? {}; return 1; },
      restore() { self.spec = {}; return 1; },
      damagedFraction(spec = {}) { const k = spec.knock?.size ?? 0; return Math.min(1, k / (2 * layers) + (spec.random ?? 0) * (1 - k / (2 * layers))); },
    };
  }
  get name() { return "Stub model (no GPU)"; }
  id(w) { let i = this.index.get(w); if (i === undefined) { i = this.vocab.length; this.vocab.push(w); this.index.set(w, i); } return i; }
  encodePrompt(text) { return words(text).map(w => this.id(w)); }
  decode(ids) { return ids.map(i => this.piece(i)).join("").replace(/^ /, ""); }
  piece(id) { return id === 0 ? "" : " " + this.vocab[id]; }
  #logits() {
    const V = this.vocab.length, n = this.ctx.length, last = this.ctx[n - 1] ?? 0;
    const sw = this.script ? words(this.script.text) : null;
    const pos = this.script ? n - this.promptLen + this.offset : -1;
    const expected = sw ? (pos < sw.length ? this.id(sw[pos]) : 0) : 0;
    const knock = this.spec.knock ?? new Set(), p = this.spec.random ?? 0;
    let dmg = p * (this.spec.mode === "flip" ? 2.2 : 1) + knock.size * 0.004;
    if (knock.has("L0.mlp")) dmg += 0.45;
    const r = mulberry(n * 977 + last * 131 + (this.spec.seed ?? 1) * 7);
    const out = new Float32Array(V).fill(-8);
    for (let k = 0; k < 6; k++) out[1 + Math.floor(r() * (V - 1))] = 1.5 + r() * 2.2;  // plausible alternatives
    out[expected] = Math.max(out[expected], 7.5 - 2.5 * r());
    if (dmg > 0) {
      const rn = mulberry(n * 31 + last + Math.round(dmg * 1e4));
      for (let i = 0; i < V; i++) out[i] += (rn() - 0.5) * 18 * dmg;
      out[expected] -= 9 * dmg;
      if (n > this.promptLen + 2) out[this.ctx[n - 2]] += 14 * dmg;                        // loops: "a time, a time"
    }
    return out;
  }
  async #wait() { if (this.delayMs) await new Promise(r => setTimeout(r, this.delayMs)); }
  async start(ids) {
    this.ctx = [...ids];
    const text = this.decode(ids);
    this.script = SCRIPTS.find(s => s.match.test(text)) ?? (ids.length === 1 ? SCRIPTS.at(-1) : null); // one token = the reference-loss paragraph
    // a script that starts with the prompt's own words (raw continuation) is followed from inside; otherwise from its start
    this.offset = 0; this.promptLen = ids.length;
    if (this.script) { const sw = words(this.script.text), pw = words(text); if (sw.slice(0, pw.length).join(" ") === pw.join(" ")) this.offset = pw.length; }
    await this.#wait(); return this.#logits();
  }
  async step(id) { this.ctx.push(id); await this.#wait(); return this.#logits(); }
  get position() { return this.ctx.length; }
  // Fake traces for layout tests: each word has a random 16-d meaning, each neuron a random preference; higher layers
  // blend in more of the recent context. Shapes are small (512 neurons, 128-d residual). Never shown as real data.
  get traceDims() { return { L: this.layers.length, H: 128, F: 512 }; }
  #vec(seed, n, scale = 1) { const r = mulberry(seed), v = new Float32Array(n); for (let i = 0; i < n; i++) v[i] = (r() + r() + r() - 1.5) * scale; return v; }
  #trace() {
    const { L, H, F } = this.traceDims, D = 16;
    this.tw ??= { n: this.#vec(11, L * F * D), h: this.#vec(12, H * D) };
    const n = this.ctx.length, cur = this.#vec(1000 + this.ctx[n - 1], D), ctx = new Float32Array(D);
    for (let k = 2; k <= Math.min(6, n); k++) { const e = this.#vec(1000 + this.ctx[n - k], D); for (let d = 0; d < D; d++) ctx[d] += e[d] / k; }
    const neurons = new Float32Array(L * F), resid = new Float32Array(L * H), x = new Float32Array(D);
    // fake damage: a switched-off MLP has silent neurons, and anything switched off scrambles the layers above it
    const knock = this.spec.knock ?? new Set(), noise = this.#vec(7 + n, D, 1.4); let hurt = (this.spec.random ?? 0) * 2;
    for (let l = 0; l < L; l++) {
      const a = l / (L - 1);
      if (knock.has(`L${l}.mixer`) || knock.has(`L${l}.mlp`)) hurt += 0.5;
      for (let d = 0; d < D; d++) x[d] = (1 - a) * cur[d] + a * ctx[d] * 1.6 + Math.min(1.5, hurt) * noise[d];
      for (let j = 0; j < F; j++) { let z = 0; const o = (l * F + j) * D; for (let d = 0; d < D; d++) z += this.tw.n[o + d] * x[d]; z = z * 0.9 - 1.2; neurons[l * F + j] = knock.has(`L${l}.mlp`) ? 0 : z / (1 + Math.exp(-z)); }
      for (let i = 0; i < H; i++) { let z = 0; for (let d = 0; d < D; d++) z += this.tw.h[i * D + d] * x[d]; resid[l * H + i] = z * (1 + l * 0.1); }
    }
    return { neurons, resid };
  }
  async startTraced(ids, onTrace) {
    this.ctx = ids.slice(0, 1);
    for (let i = 1; i < ids.length; i++) { this.ctx.push(ids[i]); await this.#wait(); await onTrace?.(i, this.#trace()); }
    return this.start(ids);
  }
  async stepTraced(id) { const logits = await this.step(id); return { trace: this.#trace(), logits }; }
}
