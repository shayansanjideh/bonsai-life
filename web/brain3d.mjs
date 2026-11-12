// 3D view of Bonsai's layer stack for brain.html: one glass slab per layer, the token mixer on the left and the MLP on
// the right, threaded on a glowing residual stream. Click a block to knock it out. Pulses run up the stack for every
// generated token, damage dims and reddens blocks, and lesion-scan results light blocks by how much they matter.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const COL = { full: new THREE.Color("#3aa0ff"), linear: new THREE.Color("#22c9b8"), mlp: new THREE.Color("#9b7bff"), off: new THREE.Color("#ff2a2a"), hot: new THREE.Color("#ff2a2a"), pearl: new THREE.Color("#f4f8ff") };

export class Brain3D {
  constructor(host, layers, { onToggle, onHover } = {}) {
    this.host = host; this.layers = layers; this.onToggle = onToggle; this.onHover = onHover;
    this.knock = new Set(); this.random = 0; this.scan = new Map(); this.pulses = []; this.tokens = [];
    const r = this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    r.setPixelRatio(Math.min(2, devicePixelRatio)); r.toneMapping = THREE.ACESFilmicToneMapping; r.toneMappingExposure = 1.15;
    r.shadowMap.enabled = true; r.shadowMap.type = THREE.PCFSoftShadowMap; r.setClearColor(0x000000, 0);
    host.appendChild(r.domElement);
    const scene = this.scene = new THREE.Scene();
    
    const pm = new THREE.PMREMGenerator(r); scene.environment = pm.fromScene(new RoomEnvironment(), 0.02).texture;
    const cam = this.camera = new THREE.PerspectiveCamera(32, 1, 0.1, 200);
    const H = this.H = layers.length * 0.36;
    cam.position.set(14, H * 0.55, 18);
    const ctl = this.controls = new OrbitControls(cam, r.domElement);
    ctl.target.set(0, H / 2 - 0.9, 0); ctl.enableDamping = true; ctl.dampingFactor = 0.08; ctl.autoRotate = true; ctl.autoRotateSpeed = 0.55;
    ctl.minDistance = 6; ctl.maxDistance = 30; ctl.enablePan = false;
    ctl.addEventListener("start", () => { ctl.autoRotate = false; clearTimeout(this.idleT); });
    ctl.addEventListener("end", () => { this.idleT = setTimeout(() => ctl.autoRotate = true, 6000); });

    scene.add(new THREE.HemisphereLight(0xdcefff, 0xf2eefc, 0.9));
    const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(7, H + 9, 6); key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048); Object.assign(key.shadow.camera, { left: -8, right: 8, top: 8, bottom: -8, near: 1, far: H + 30 }); key.shadow.bias = -0.0004; key.shadow.radius = 6;
    key.target.position.set(0, H / 2, 0); scene.add(key, key.target);

    // floor: dark mirror-ish disc with a soft ring
    const floor = new THREE.Mesh(new THREE.CircleGeometry(9, 128), new THREE.MeshPhysicalMaterial({ color: 0xffffff, metalness: 0.05, roughness: 0.22, clearcoat: 1, clearcoatRoughness: 0.08 }));
    floor.rotation.x = -Math.PI / 2; floor.position.y = -0.45; floor.receiveShadow = true; scene.add(floor);
    const grid = new THREE.PolarGridHelper(8.6, 12, 6, 96, 0xb9c6d8, 0xd6dee9); grid.position.y = -0.44; grid.material.transparent = true; grid.material.opacity = 0.7; scene.add(grid);
    const ring = new THREE.Mesh(new THREE.RingGeometry(3.05, 3.16, 128), new THREE.MeshBasicMaterial({ color: 0xff2a2a }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = -0.435; scene.add(ring);

    // residual stream
    this.stream = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, H + 0.6, 24), new THREE.MeshBasicMaterial({ color: 0xff2a2a, transparent: true, opacity: 0.9 }));
    this.stream.position.y = H / 2 - 0.2; scene.add(this.stream);

    // blocks
    const geo = new RoundedBoxGeometry(1.55, 0.24, 1.55, 4, 0.08);
    const edgeGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(1.56, 0.25, 1.56));
    this.blocks = [];
    for (const l of layers) for (const part of ["mixer", "mlp"]) {
      const base = part === "mlp" ? COL.mlp : l.type === "full" ? COL.full : COL.linear;
      const mat = new THREE.MeshPhysicalMaterial({ color: COL.pearl.clone().lerp(base, 0.35), metalness: 0.55, roughness: 0.12, clearcoat: 1, clearcoatRoughness: 0.05,
        iridescence: 1, iridescenceIOR: 1.35, iridescenceThicknessRange: [180, 520], emissive: base.clone(), emissiveIntensity: 0, transparent: true, opacity: 1 });
      const m = new THREE.Mesh(geo, mat); m.castShadow = true; m.receiveShadow = true;
      m.position.set(part === "mixer" ? -0.95 : 0.95, l.i * 0.36, 0);
      const edges = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: base, transparent: true, opacity: 0.55 }));
      m.add(edges);
      m.userData = { key: `L${l.i}.${part}`, layer: l.i, part, type: l.type, base, edges, glow: 0 };
      scene.add(m); this.blocks.push(m);
    }
    // layer index labels on the left
    for (const l of layers) if (l.i % 4 === 0 || l.i === layers.length - 1) { const s = this.#label(String(l.i), 0x566275, 34); s.position.set(-2.15, l.i * 0.36, 0); s.scale.multiplyScalar(0.5); scene.add(s); }
    const lm = this.#label("mixer", 0x566275, 30); lm.position.set(-0.95, -0.36, 1.0); lm.scale.multiplyScalar(0.6); scene.add(lm);
    const lp = this.#label("MLP", 0x566275, 30); lp.position.set(0.95, -0.36, 1.0); lp.scale.multiplyScalar(0.6); scene.add(lp);

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
    const c = document.createElement("canvas"), g = c.getContext("2d"); g.font = `600 ${px}px -apple-system, "SF Pro Text", Inter, system-ui, sans-serif`;
    const w = Math.ceil(g.measureText(text).width) + 16; c.width = w; c.height = px + 16;
    g.font = `600 ${px}px -apple-system, "SF Pro Text", Inter, system-ui, sans-serif`; g.fillStyle = "#" + new THREE.Color(color).getHexString(); g.textBaseline = "middle"; g.fillText(text, 8, c.height / 2);
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
  pulse(p = 1) { this.pulses.push(0); this.pulseCol = new THREE.Color("#3aa0ff").lerp(COL.off, Math.max(0, Math.min(1, -Math.log10(Math.max(p, 1e-4)) / 2))); if (this.pulses.length > 6) this.pulses.shift(); }
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
      let tint = 0.35, ei = 0, lift = 0;
      let col = u.base;
      if (scanV !== null) { tint = 0.2 + 0.8 * Math.min(1, scanV); lift = 1.6 * Math.sqrt(Math.min(1, scanV)); if (scanV > 0.5) col = u.base.clone().lerp(COL.hot, (scanV - 0.5) * 2); }
      if (off) { col = COL.off; tint = 1; ei = 0.25 + 0.15 * Math.sin(t * 4); }
      m.color.copy(COL.pearl).lerp(col, tint).multiplyScalar(off ? 1 : flick * 0.6 + 0.4);
      m.emissive.copy(off ? COL.off : pulse > 0.05 && this.pulseCol ? this.pulseCol : u.base); m.emissiveIntensity = ei + pulse * (off ? 0 : 0.9) + u.glow * 0.6;
      m.opacity = off ? 0.7 : 1;
      u.edges.material.color.copy(off ? COL.off : col); u.edges.material.opacity = off ? 1 : 0.45 + pulse * 0.5;
      u.lift = (u.lift ?? 0) + (lift - (u.lift ?? 0)) * Math.min(1, dt * 6); b.position.x = (u.part === "mixer" ? -1 : 1) * (0.95 + u.lift);
      const s = 1 + (b === this.hover ? 0.05 : 0) + pulse * 0.04; b.scale.set(s, 1, s);
    }
    this.stream.material.opacity = 0.55 + 0.4 * (this.pulses.length ? 1 : 0) * (1 - this.random);
    this.stream.material.color.copy(this.pulses.length && this.pulseCol ? this.pulseCol : COL.off);
    for (const s of this.tokens) { s.userData.age += dt; s.position.y += dt * 0.9; s.material.opacity = Math.max(0, 1 - s.userData.age / 3); }
    this.renderer.render(this.scene, this.camera);
  }
  dispose() { this.running = false; this.renderer.dispose(); }
}
