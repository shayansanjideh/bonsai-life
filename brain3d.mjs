// 3D view of Bonsai's layer stack for layers.html: one flat disc per block, the token mixer on the left and the MLP on
// the right, threaded on an orange residual stream. Click a block to knock it out. Pulses run up the stack for every
// generated token, damage dims and reddens blocks, and lesion-scan results light blocks by how much they matter.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const COL = { full: new THREE.Color("#12161a"), linear: new THREE.Color("#9aa4aa"), mlp: new THREE.Color("#c9d0d4"), off: new THREE.Color("#ff5a1f"), hot: new THREE.Color("#ff5a1f"), pearl: new THREE.Color("#f8faf9"), ink: new THREE.Color("#12161a") };

export class Brain3D {
  constructor(host, layers, { onToggle, onHover } = {}) {
    this.host = host; this.layers = layers; this.onToggle = onToggle; this.onHover = onHover;
    this.knock = new Set(); this.random = 0; this.scan = new Map(); this.pulses = []; this.tokens = [];
    const r = this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    r.setPixelRatio(Math.min(2, devicePixelRatio)); r.toneMapping = THREE.NoToneMapping;
    r.setClearColor(0x000000, 0);
    host.appendChild(r.domElement);
    const scene = this.scene = new THREE.Scene();
    
    const cam = this.camera = new THREE.PerspectiveCamera(32, 1, 0.1, 200);
    const H = this.H = layers.length * 0.36;
    cam.position.set(14, H * 0.55, 18);
    const ctl = this.controls = new OrbitControls(cam, r.domElement);
    ctl.target.set(0, H / 2 - 0.9, 0); ctl.enableDamping = true; ctl.dampingFactor = 0.08; ctl.autoRotate = true; ctl.autoRotateSpeed = 0.55;
    ctl.minDistance = 6; ctl.maxDistance = 30; ctl.enablePan = false;
    ctl.addEventListener("start", () => { ctl.autoRotate = false; clearTimeout(this.idleT); });
    ctl.addEventListener("end", () => { this.idleT = setTimeout(() => ctl.autoRotate = true, 6000); });

    // flat colours only (MeshBasicMaterial), so no lights

    // floor: a polar grid and one thin ink ring, nothing glossy
    const grid = new THREE.PolarGridHelper(8.6, 12, 6, 96, 0xc9d0d4, 0xdde3e6); grid.position.y = -0.44; scene.add(grid);
    const ring = new THREE.Mesh(new THREE.RingGeometry(3.08, 3.12, 128), new THREE.MeshBasicMaterial({ color: 0x12161a }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = -0.435; scene.add(ring);

    // residual stream
    this.stream = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, H + 0.6, 16), new THREE.MeshBasicMaterial({ color: 0xff5a1f, transparent: true, opacity: 0.9 }));
    this.stream.position.y = H / 2 - 0.2; scene.add(this.stream);

    // blocks
    // each block is a flat disc with a crisp ring edge (rings and dots, not slabs)
    const geo = new THREE.CylinderGeometry(0.74, 0.74, 0.05, 64);
    const edgeGeo = new THREE.EdgesGeometry(new THREE.CylinderGeometry(0.745, 0.745, 0.055, 64), 30);
    this.blocks = [];
    for (const l of layers) for (const part of ["mixer", "mlp"]) {
      const base = part === "mlp" ? COL.mlp : l.type === "full" ? COL.full : COL.linear;
      const mat = new THREE.MeshBasicMaterial({ color: base.clone(), transparent: true, opacity: 0.9 });
      const m = new THREE.Mesh(geo, mat);
      m.position.set(part === "mixer" ? -0.95 : 0.95, l.i * 0.36, 0);
      const edges = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: COL.ink, transparent: true, opacity: 0.6 }));
      m.add(edges);
      m.userData = { key: `L${l.i}.${part}`, layer: l.i, part, type: l.type, base, edges, glow: 0 };
      scene.add(m); this.blocks.push(m);
    }
    // layer index labels on the left
    for (const l of layers) if (l.i % 4 === 0 || l.i === layers.length - 1) { const s = this.#label("L" + String(l.i).padStart(2, "0"), 0x12161a, 30); s.position.set(-2.15, l.i * 0.36, 0); s.scale.multiplyScalar(0.5); scene.add(s); }
    const lm = this.#label("Mixer", 0x6c757d, 30); lm.position.set(-0.95, -0.36, 1.0); lm.scale.multiplyScalar(0.6); scene.add(lm);
    const lp = this.#label("MLP", 0x6c757d, 30); lp.position.set(0.95, -0.36, 1.0); lp.scale.multiplyScalar(0.6); scene.add(lp);

    // picking
    this.ray = new THREE.Raycaster(); this.mouse = new THREE.Vector2(); this.hover = null;
    const dom = r.domElement; let down = null;
    dom.addEventListener("pointermove", e => { this.#setMouse(e); this.#pick(e); });
    dom.addEventListener("pointerleave", () => { this.hover = null; this.onHover?.(null); dom.style.cursor = ""; });
    dom.addEventListener("pointerdown", e => down = [e.clientX, e.clientY]);
    dom.addEventListener("pointerup", e => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
      this.#setMouse(e); const hit = this.#hit(); if (hit) this.onToggle?.(hit.userData.key);
    });
    new ResizeObserver(() => this.resize()).observe(host); this.resize();
    this.clock = new THREE.Clock(); this.running = true;
    // Share the GPU politely with the model: render at most ~20 fps while the model is busy, and not at all when off screen.
    this.busy = false; this.visible = true; this.n = 0;
    new IntersectionObserver(es => this.visible = es[0].isIntersecting).observe(host);
    const loop = () => { if (!this.running) return; requestAnimationFrame(loop); this.n++;
      if (!this.visible || document.hidden) { this.clock.getDelta(); return; }
      if (this.busy && this.n % 3) return;
      this.#frame(); }; loop();
  }

  #label(text, color, px) {
    const c = document.createElement("canvas"), g = c.getContext("2d"); g.font = `500 ${px}px "Instrument Sans", -apple-system, Inter, system-ui, sans-serif`;
    const w = Math.ceil(g.measureText(text).width) + 16; c.width = w; c.height = px + 16;
    g.font = `500 ${px}px "Instrument Sans", -apple-system, Inter, system-ui, sans-serif`; g.fillStyle = "#" + new THREE.Color(color).getHexString(); g.textBaseline = "middle"; g.fillText(text, 8, c.height / 2);
    const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true, depthWrite: false }));
    s.scale.set(w / 60, c.height / 60, 1); return s;
  }
  #setMouse(e) { const b = this.renderer.domElement.getBoundingClientRect(); this.mouse.set(((e.clientX - b.left) / b.width) * 2 - 1, -((e.clientY - b.top) / b.height) * 2 + 1); }
  #hit() { this.ray.setFromCamera(this.mouse, this.camera); return this.ray.intersectObjects(this.blocks, false)[0]?.object ?? null; }
  #pick(e) {
    const h = this.#hit(); this.hover = h; this.renderer.domElement.style.cursor = h ? "pointer" : "";
    this.onHover?.(h ? { ...h.userData, x: e.clientX, y: e.clientY, delta: this.scan.get(h.userData.key) } : null);
  }
  resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight; if (!w || !h) return;
    this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  setDamage(knock, random) { this.knock = new Set(knock); this.random = random; }
  setScan(results) { this.scan = new Map(results.map(r => [`L${r.layer}.${r.part}`, r.delta])); this.scanMax = Math.max(0.05, ...results.map(r => r.delta)); }
  clearScan() { this.scan = new Map(); }
  focus(key) { const b = this.blocks.find(b => b.userData.key === key); if (b) b.userData.glow = 1.6; }
  pulse(p = 1) { this.pulses.push(0); this.pulseCol = COL.off.clone().lerp(COL.ink, Math.max(0, Math.min(1, 1 - -Math.log10(Math.max(p, 1e-4)) / 2)) * 0.5); if (this.pulses.length > 6) this.pulses.shift(); }
  emitToken(text) {
    if (!text || !text.trim()) return;
    const s = this.#label(text.length > 18 ? text.slice(0, 17) + "…" : text, 0x0d1526, 44);
    this.tokN = (this.tokN ?? 0) + 1; const lane = [0, 1.3, -1.3, 2.4, -2.4][this.tokN % 5]; s.position.set(lane, this.H + 0.2, (Math.random() - 0.5) * 0.6); s.scale.multiplyScalar(0.55);
    s.userData.age = 0; this.scene.add(s); this.tokens.push(s);
    if (this.tokens.length > 10) { const o = this.tokens.shift(); this.scene.remove(o); o.material.map.dispose(); o.material.dispose(); }
  }

  #frame() {
    const dt = Math.min(0.05, this.clock.getDelta()), t = this.clock.elapsedTime;
    this.controls.update();
    // pulses travel up the residual stream at ~2 stack-heights per second
    this.pulses = this.pulses.map(p => p + dt * 2.2).filter(p => p < 1.15);
    for (const b of this.blocks) {
      const u = b.userData, m = b.material, off = this.knock.has(u.key);
      const y = u.layer / Math.max(1, this.layers.length - 1);
      let pulse = 0; for (const p of this.pulses) pulse = Math.max(pulse, Math.exp(-((p - y) ** 2) / 0.0018));
      u.glow = Math.max(0, u.glow - dt * 1.5);
      const scanV = this.scan.has(u.key) ? Math.max(0, this.scan.get(u.key)) / this.scanMax : null;
      const flick = this.random > 0 ? 1 - this.random * (0.75 + 0.25 * Math.sin(t * 9 + u.layer * 2.3 + (u.part === "mlp" ? 1.7 : 0))) : 1;
      let lift = 0;
      let col = u.base;
      if (scanV !== null) { lift = 1.6 * Math.sqrt(Math.min(1, scanV)); col = u.base.clone().lerp(COL.hot, Math.min(1, scanV)); }
      if (off) col = COL.off;
      m.color.copy(col).lerp(COL.pearl, off ? 0 : (1 - flick) * 0.8);
      if (!off && (pulse > 0.02 || u.glow > 0)) m.color.lerp(this.pulseCol ?? COL.off, Math.min(1, pulse * 0.7 + u.glow * 0.5));
      m.opacity = off ? 1 : 0.9;
      u.edges.material.color.copy(off ? COL.off : COL.ink); u.edges.material.opacity = off ? 1 : 0.55 + pulse * 0.4;
      u.lift = (u.lift ?? 0) + (lift - (u.lift ?? 0)) * Math.min(1, dt * 6); b.position.x = (u.part === "mixer" ? -1 : 1) * (0.95 + u.lift);
      const s = 1 + (b === this.hover ? 0.05 : 0) + pulse * 0.04; b.scale.set(s, 1, s);
    }
    this.stream.material.opacity = 0.55 + 0.4 * (this.pulses.length ? 1 : 0) * (1 - this.random);
    this.stream.material.color.copy(COL.off);
    for (const s of this.tokens) { s.userData.age += dt; s.position.y += dt * 0.9; s.material.opacity = Math.max(0, 1 - s.userData.age / 3); }
    this.renderer.render(this.scene, this.camera);
  }
  dispose() { this.running = false; this.renderer.dispose(); }
}
