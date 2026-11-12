// Neuron cloud analysis: pure functions over traces from brain-engine.mjs (no GPU, no DOM), so they run in tests too.
//
// NeuronMap picks a sample of MLP neurons in every layer and lays them out so neurons that fire together sit together:
//   1. calibrate on a short text: keep every neuron's activation for every token (int8, per token-layer scale)
//   2. per layer, keep the `perLayer` neurons whose activation varies most across tokens (the ones doing something)
//   3. describe each kept neuron by its z-scored activations over the calibration tokens; PCA over all of them gives
//      two directions; a neuron's angle on its layer's disc comes from those, its radius from its rank (fills the disc)
//   4. edges: for each neuron, its most correlated neuron in the next layer, when the correlation is strong
// WordMap keeps the residual stream of every token in the current text and projects one layer's states to 3D (PCA),
// so tokens the model represents similarly land close together.

function power(G, n, k, iters = 80) {           // top-k eigenpairs of a symmetric n×n matrix (Float64Array, row-major)
  const A = Float64Array.from(G), out = [];
  for (let c = 0; c < k; c++) {
    let v = new Float64Array(n); for (let i = 0; i < n; i++) v[i] = Math.sin(i * 12.9898 + c * 78.233) + 0.01;
    let lam = 0;
    for (let it = 0; it < iters; it++) {
      const w = new Float64Array(n);
      for (let i = 0; i < n; i++) { let s = 0; const o = i * n; for (let j = 0; j < n; j++) s += A[o + j] * v[j]; w[i] = s; }
      let nn = 0; for (let i = 0; i < n; i++) nn += w[i] * w[i]; nn = Math.sqrt(nn) || 1;
      lam = nn; for (let i = 0; i < n; i++) v[i] = w[i] / nn;
    }
    out.push({ v, lam });
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i * n + j] -= lam * v[i] * v[j];
  }
  return out;
}

export class NeuronMap {
  constructor({ L, F }, { perLayer = 192 } = {}) {
    this.L = L; this.F = F; this.K = Math.min(perLayer, F);
    this.q = []; this.scales = []; this.built = false;
  }
  get tokens() { return this.q.length; }
  addCalibration(neurons) {                       // neurons: Float32Array(L*F) for one token
    const { L, F } = this, q = new Int8Array(L * F), sc = new Float32Array(L);
    for (let l = 0; l < L; l++) {
      let m = 1e-12; for (let j = 0; j < F; j++) m = Math.max(m, Math.abs(neurons[l * F + j]));
      sc[l] = m / 127; for (let j = 0; j < F; j++) q[l * F + j] = Math.round(neurons[l * F + j] / sc[l]);
    }
    this.q.push(q); this.scales.push(sc);
  }
  build({ radius = 3.2, spacing = 0.42, edgeCorr = 0.5, maxEdges = 1400 } = {}) {
    const { L, F, K } = this, T = this.q.length;
    if (T < 4) throw new Error("NeuronMap: need at least 4 calibration tokens");
    const val = (t, l, j) => this.q[t][l * F + j] * this.scales[t][l];
    // 1. pick the K most variable neurons per layer
    let live = 0;
    const ids = new Int32Array(L * K), mean = new Float32Array(L * K), std = new Float32Array(L * K);
    for (let l = 0; l < L; l++) {
      const mu = new Float64Array(F), sq = new Float64Array(F);
      for (let t = 0; t < T; t++) for (let j = 0; j < F; j++) { const x = val(t, l, j); mu[j] += x; sq[j] += x * x; }
      const sd = new Float64Array(F); for (let j = 0; j < F; j++) { mu[j] /= T; sd[j] = Math.sqrt(Math.max(0, sq[j] / T - mu[j] * mu[j])); }
      const order = Array.from({ length: F }, (_, j) => j).sort((a, b) => sd[b] - sd[a]).slice(0, K).sort((a, b) => a - b);
      order.forEach((j, k) => { ids[l * K + k] = j; mean[l * K + k] = mu[j]; std[l * K + k] = sd[j] || 1; });
      for (let j = 0; j < F; j++) if (sd[j] > 1e-6) live++;
    }
    this.live = live / (L * F);
    if (this.live < 0.05) throw new Error(`NeuronMap: only ${(this.live * 100).toFixed(1)}% of neurons ever changed; the trace looks empty`);
    // 2. z-scored token vectors, S × T
    const S = L * K, X = new Float32Array(S * T);
    for (let s = 0; s < S; s++) { const l = Math.floor(s / K), j = ids[s]; for (let t = 0; t < T; t++) X[s * T + t] = (val(t, l, j) - mean[s]) / std[s]; }
    // 3. PCA via the T×T Gram matrix
    const G = new Float64Array(T * T);
    for (let s = 0; s < S; s++) { const o = s * T; for (let a = 0; a < T; a++) { const xa = X[o + a]; if (!xa) continue; for (let b = a; b < T; b++) G[a * T + b] += xa * X[o + b]; } }
    for (let a = 0; a < T; a++) for (let b = 0; b < a; b++) G[a * T + b] = G[b * T + a];
    const [p1, p2] = power(G, T, 2);
    const c1 = new Float32Array(S), c2 = new Float32Array(S);
    for (let s = 0; s < S; s++) { let u = 0, w = 0; for (let t = 0; t < T; t++) { u += X[s * T + t] * p1.v[t]; w += X[s * T + t] * p2.v[t]; } c1[s] = u; c2[s] = w; }
    const pos = new Float32Array(S * 3);
    for (let l = 0; l < L; l++) {
      const rows = Array.from({ length: K }, (_, k) => l * K + k), mag = rows.map(s => Math.hypot(c1[s], c2[s]));
      const rank = rows.map((_, i) => i).sort((a, b) => mag[a] - mag[b]);
      rank.forEach((i, r) => {
        const s = rows[i], ang = Math.atan2(c2[s], c1[s]), rad = radius * Math.sqrt((r + 0.5) / K);
        pos[s * 3] = Math.cos(ang) * rad; pos[s * 3 + 1] = l * spacing; pos[s * 3 + 2] = Math.sin(ang) * rad;
      });
    }
    // 4. edges to the most correlated neuron in the next layer
    const edges = [];
    for (let l = 0; l < L - 1; l++) for (let a = 0; a < K; a++) {
      const sa = l * K + a; let best = -1, bc = edgeCorr;
      for (let b = 0; b < K; b++) { const sb = (l + 1) * K + b; let c = 0; for (let t = 0; t < T; t++) c += X[sa * T + t] * X[sb * T + t]; c /= T; if (c > bc) { bc = c; best = sb; } }
      if (best >= 0) edges.push([sa, best, bc]);
    }
    edges.sort((a, b) => b[2] - a[2]); edges.length = Math.min(edges.length, maxEdges);
    Object.assign(this, { S, ids, mean, std, pos, edges, spacing, radius, built: true, explained: [p1.lam, p2.lam] });
    this.q = []; this.scales = [];                // calibration data no longer needed
    return this;
  }
  activity(neurons) {                             // z-scores of the sampled neurons for one token
    const { S, K, F } = this, z = new Float32Array(S);
    for (let s = 0; s < S; s++) { const l = Math.floor(s / K); z[s] = (neurons[l * F + this.ids[s]] - this.mean[s]) / this.std[s]; }
    return z;
  }
  path(z, thresh = 1) {                           // per layer: centre of the neurons firing well above their usual level
    const { L, K, pos } = this, pts = new Float32Array(L * 3);
    for (let l = 0; l < L; l++) {
      let w = 0, x = 0, y = 0;
      for (let k = 0; k < K; k++) { const s = l * K + k, e = Math.max(0, z[s] - thresh); if (!e) continue; const q = e * e; w += q; x += q * pos[s * 3]; y += q * pos[s * 3 + 2]; }
      pts[l * 3] = w ? x / w : 0; pts[l * 3 + 1] = l * this.spacing; pts[l * 3 + 2] = w ? y / w : 0;
    }
    return pts;
  }
  describe(s) { return { layer: Math.floor(s / this.K), neuron: this.ids[s] }; }
}

export class WordMap {
  constructor({ L, H }) { this.L = L; this.H = H; this.items = []; }
  add(text, resid, meta) {                        // resid: Float32Array(L*H); each layer's state is RMS-normalised
    const { L, H } = this, q = new Int8Array(L * H), sc = new Float32Array(L);
    for (let l = 0; l < L; l++) {
      let ss = 0, m = 1e-12; for (let i = 0; i < H; i++) { const x = resid[l * H + i]; ss += x * x; }
      const rms = Math.sqrt(ss / H) || 1;
      for (let i = 0; i < H; i++) m = Math.max(m, Math.abs(resid[l * H + i] / rms));
      sc[l] = m / 127;                            // stored values are resid/rms in units of sc
      for (let i = 0; i < H; i++) q[l * H + i] = Math.round(resid[l * H + i] / rms / sc[l]);
    }
    this.items.push({ text, q, sc, meta });
  }
  clear() { this.items = []; }
  project(layer, skip = 0) {                      // → [{text, i, x, y, z}] scaled to about ±1
    const { H } = this, its = this.items.slice(skip), T = its.length;
    if (T < 4) return its.map((it, i) => ({ text: it.text, meta: it.meta, i: i + skip, x: 0, y: 0, z: 0 }));
    const X = new Float32Array(T * H), mu = new Float64Array(H);
    its.forEach((it, t) => { for (let i = 0; i < H; i++) { const x = it.q[layer * H + i] * it.sc[layer]; X[t * H + i] = x; mu[i] += x / T; } });
    for (let t = 0; t < T; t++) for (let i = 0; i < H; i++) X[t * H + i] -= mu[i];
    const G = new Float64Array(T * T);
    for (let a = 0; a < T; a++) for (let b = a; b < T; b++) { let s = 0; for (let i = 0; i < H; i++) s += X[a * H + i] * X[b * H + i]; G[a * T + b] = G[b * T + a] = s; }
    const pcs = power(G, T, 3, 60);
    const out = its.map((it, t) => ({ text: it.text, meta: it.meta, i: t + skip, x: pcs[0].v[t] * Math.sqrt(pcs[0].lam), y: pcs[1].v[t] * Math.sqrt(pcs[1].lam), z: pcs[2].v[t] * Math.sqrt(pcs[2].lam) }));
    let m = 1e-9; for (const p of out) m = Math.max(m, Math.abs(p.x), Math.abs(p.y), Math.abs(p.z));
    for (const p of out) { p.x /= m; p.y /= m; p.z /= m; }
    return out;
  }
}
