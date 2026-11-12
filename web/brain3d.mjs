// 3D view of Bonsai's layers: a rack of chrome blocks, one column per layer (token mixer on top, MLP below).
// Talks to brain.html only through window.brainAPI and "brain:*" DOM events, so if three.js can't load,
// the rest of the page keeps working.
const $ = id => document.getElementById(id);
const wrap = $("s3Wrap"), canvas = $("s3"), msg = $("s3Msg"), tipEl = $("s3Tip"), badge = $("s3Badge");
const fail = t => { msg.textContent = t; msg.hidden = false; };

let THREE, OrbitControls, RoomEnvironment, RoundedBoxGeometry;
try {
  THREE = await import("three");
  ({ OrbitControls } = await import("three/addons/controls/OrbitControls.js"));
  ({ RoomEnvironment } = await import("three/addons/environments/RoomEnvironment.js"));
  ({ RoundedBoxGeometry } = await import("three/addons/geometries/RoundedBoxGeometry.js"));
} catch (e) {
  fail("The 3D view needs three.js from a CDN and couldn't load it. Everything else on the page still works.");
  throw e;
}

let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" });
} catch (e) { fail("This browser has WebGL turned off, so the 3D view is unavailable."); throw e; }
// Software GL (no GPU, e.g. headless CI or GPU blocklisted) draws a frame in seconds; there the view only
// redraws when something changes, with no sway, pulses or dust, so it never starves the rest of the page.
const glInfo = (() => { const gl = renderer.getContext(), ext = gl.getExtension("WEBGL_debug_renderer_info"); return ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : ""; })();
const SOFT = /swiftshader|llvmpipe|softpipe|software/i.test(glInfo) || new URLSearchParams(location.search).has("still3d");
renderer.setPixelRatio(SOFT ? 1 : Math.min(devicePixelRatio || 1, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.3;
renderer.outputColorSpace = THREE.SRGBColorSpace;
msg.hidden = true;

const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.035).texture;
const key = new THREE.DirectionalLight(0xffffff, 1.1); key.position.set(4, 8, 6); scene.add(key);
const rim = new THREE.DirectionalLight(0x8ab4ff, 0.6); rim.position.set(-6, 2, -6); scene.add(rim);

const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 200);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = !SOFT; controls.dampingFactor = 0.07;
let spinning = !SOFT, swayT = 0; // gentle side-to-side sway instead of a full spin
controls.enablePan = false; controls.minDistance = 6; controls.maxDistance = 45;
controls.minPolarAngle = 0.25; controls.maxPolarAngle = Math.PI - 0.35;

// ---------- palette ----------
const C = {
  linear: new THREE.Color("#e4e8ee"), full: new THREE.Color("#3d86ff"), mlp: new THREE.Color("#cbbb9f"),
  off: new THREE.Color("#ff453a"), worn: new THREE.Color("#2a2b2f"), graphite: new THREE.Color("#7d8189"),
  hot: new THREE.Color("#ff8a00"), black: new THREE.Color(0), white: new THREE.Color(0xffffff),
  blue: new THREE.Color("#64b5ff"), red: new THREE.Color("#ff453a"),
};

// ---------- geometry ----------
const SPACING = 0.52, THICK = 0.3, H = 1.25, DEPTH = 1.8, GAP = 0.2;
const geo = new RoundedBoxGeometry(THICK, H, DEPTH, 3, 0.07);
const rack = new THREE.Group(); scene.add(rack);
let blocks = [], layerCount = 0, rackLen = 0, live = false;

// floor glow: soft radial gradient under the rack
function radialTexture(inner, outer) {
  const c = document.createElement("canvas"); c.width = c.height = 256;
  const g = c.getContext("2d"), gr = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  gr.addColorStop(0, inner); gr.addColorStop(1, outer); g.fillStyle = gr; g.fillRect(0, 0, 256, 256);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
const floor = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: radialTexture("rgba(120,150,210,0.22)", "rgba(0,0,0,0)"), transparent: true, depthWrite: false }));
floor.rotation.x = -Math.PI / 2; scene.add(floor);

// the residual stream: a thin glowing rail through the middle of the rack
const spineMat = new THREE.MeshBasicMaterial({ color: 0x5aa9ff, transparent: true, opacity: 0.35 });
const spine = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 1, 16), spineMat);
spine.rotation.z = Math.PI / 2; rack.add(spine);

function label(text, size = 0.32, color = "#98989f", weight = 600) {
  const c = document.createElement("canvas"), g = c.getContext("2d"), px = 64;
  g.font = `${weight} ${px}px -apple-system, "SF Pro Text", Inter, system-ui, sans-serif`;
  const w = Math.ceil(g.measureText(text).width) + 24; c.width = w; c.height = px + 24;
  g.font = `${weight} ${px}px -apple-system, "SF Pro Text", Inter, system-ui, sans-serif`;
  g.fillStyle = color; g.textBaseline = "middle"; g.fillText(text, 12, c.height / 2);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true, depthWrite: false }));
  s.scale.set(size * c.width / c.height, size, 1); return s;
}
const labels = new THREE.Group(); rack.add(labels);

// dust: one particle per ~0.1% of weights knocked out by random damage
const DUST = 1400;
const dustGeo = new THREE.BufferGeometry(), dustPos = new Float32Array(DUST * 3), dustVel = new Float32Array(DUST * 3);
dustGeo.setAttribute("position", new THREE.BufferAttribute(dustPos, 3)); dustGeo.setDrawRange(0, 0);
const dust = new THREE.Points(dustGeo, new THREE.PointsMaterial({ size: 0.045, color: 0xff6a61, transparent: true, opacity: 0.8, depthWrite: false, blending: THREE.AdditiveBlending }));
rack.add(dust);

function build(layers) {
  for (const b of blocks) { rack.remove(b.mesh); b.mesh.material.dispose(); }
  labels.clear(); blocks = [];
  layerCount = layers.length; rackLen = (layerCount - 1) * SPACING;
  for (const l of layers) {
    const x = l.i * SPACING - rackLen / 2;
    for (const part of ["mixer", "mlp"]) {
      const base = part === "mlp" ? C.mlp : l.type === "full" ? C.full : C.linear;
      const mat = new THREE.MeshPhysicalMaterial({ color: base.clone(), metalness: 1, roughness: part === "mlp" ? 0.24 : 0.16, clearcoat: 0.7, clearcoatRoughness: 0.12, envMapIntensity: 1.6, emissive: new THREE.Color(0) });
      const mesh = new THREE.Mesh(geo, mat);
      const sign = part === "mixer" ? 1 : -1;
      mesh.position.set(x, sign * (GAP + H / 2), 0);
      const b = { mesh, k: `L${l.i}.${part}`, layer: l.i, part, type: l.type, sign, x, base,
        baseRough: mat.roughness, impact: 0, delta: null, flash: 0, hover: 0, z: 0, sy: 1 };
      mesh.userData.block = b; rack.add(mesh); blocks.push(b);
    }
    if (l.i % 4 === 0 || l.i === layerCount - 1) { const s = label(String(l.i), 0.28); s.position.set(x, -(GAP + H + 0.42), DEPTH / 2); labels.add(s); }
  }
  const head = -rackLen / 2 - 1.5, tail = rackLen / 2 + 0.75;
  for (const [t, y] of [["mixer", GAP + H / 2], ["MLP", -(GAP + H / 2)]]) { const s = label(t, 0.3, "#c7c7cc"); s.position.set(head - 0.35, y, 0); labels.add(s); }
  const a = label("tokens in", 0.26, "#6e6e75"); a.position.set(head - 0.1, 0, DEPTH / 2); labels.add(a);
  const z = label("next token", 0.26, "#6e6e75"); z.position.set(tail + 0.5, 0, 0); labels.add(z);
  const lay = label("layer", 0.24, "#6e6e75", 500); lay.position.set(head - 0.35, -(GAP + H + 0.42), DEPTH / 2); labels.add(lay);
  spine.scale.y = rackLen + 1.6;
  floor.scale.set(rackLen + 9, 7, 1); floor.position.y = -(GAP + H + 0.75);
  for (let i = 0; i < DUST; i++) resetDust(i, true);
  frame();
}
function resetDust(i, anywhere) {
  dustPos[i * 3] = (Math.random() - 0.5) * (rackLen + 1);
  dustPos[i * 3 + 1] = (Math.random() < 0.5 ? 1 : -1) * (GAP + Math.random() * H);
  dustPos[i * 3 + 2] = (Math.random() - 0.5) * DEPTH * (anywhere ? 1.6 : 1);
  dustVel[i * 3] = (Math.random() - 0.5) * 0.004; dustVel[i * 3 + 1] = -0.002 - Math.random() * 0.006; dustVel[i * 3 + 2] = (Math.random() - 0.5) * 0.01;
}

function frame() {
  const w = wrap.clientWidth, h = wrap.clientHeight; if (!w || !h) return;
  renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
}
function fitDistance() {
  const halfW = (rackLen + 4.5) / 2, halfH = GAP + H + 1.2, t = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  return Math.max(halfH / t, halfW / (t * camera.aspect)) * 1.12 + DEPTH;
}
function resetView() {
  frame();
  // portrait screens: look down the rack diagonally so it fills the frame instead of shrinking to a sliver
  const tall = camera.aspect < 1;
  const d = tall ? fitDistance() * 0.5 + 4.5 : fitDistance(), dir = tall ? new THREE.Vector3(1, 0.5, 0.8).normalize() : new THREE.Vector3(0.22, 0.3, 1).normalize();
  camera.position.copy(dir.multiplyScalar(d)); controls.target.set(0, -0.1, 0); controls.update();
}
new ResizeObserver(frame).observe(wrap);

// ---------- state from the page ----------
const st = { knock: new Set(), random: 0, mode: "zero", colorBy: "type", busy: 0, pulses: [], maxDelta: 0 };
document.addEventListener("brain:ready", e => { live = true; build(e.detail.layers); resetView(); setBadge(); });
document.addEventListener("brain:damage", e => { const d = e.detail; st.knock = new Set(d.knock); st.random = d.random; st.mode = d.mode; setBadge(d.fraction); });
document.addEventListener("brain:busy", e => { st.busy = e.detail.busy ? performance.now() : 0; });
document.addEventListener("brain:token", e => {
  if (SOFT) return;
  if (st.pulses.length > 8) st.pulses.shift();
  st.pulses.push({ x: -rackLen / 2 - 0.9, p: e.detail.p ?? 1, mesh: pulseMesh(e.detail.p ?? 1) });
});
document.addEventListener("brain:scan", e => {
  const res = e.detail.res; st.maxDelta = Math.max(1e-6, ...res.map(r => r.delta));
  for (const b of blocks) { const r = res.find(r => r.layer === b.layer && r.part === b.part); b.delta = r ? r.delta : null; }
  const btn = $("s3Color").querySelector('[data-c="impact"]'); btn.disabled = false; btn.title = "";
  if (st.colorBy !== "impact") setColorBy("impact");
});
function setBadge(f) {
  const n = st.knock.size;
  badge.textContent = !live ? "preview · wake Bonsai to operate" : n || st.random ? `${n} block${n === 1 ? "" : "s"} off${st.random ? ` · ${Math.round(st.random * 100)}% random ${st.mode === "flip" ? "flips" : "zeros"}` : ""}${f != null ? ` · ${(f * 100).toFixed(1)}% of weights` : ""}` : "intact";
}

// token pulses travel through the rack along the residual stream; color = how sure the model was
const pulseGeo = new THREE.SphereGeometry(0.11, 20, 20);
const glowTex = radialTexture("rgba(255,255,255,1)", "rgba(255,255,255,0)");
function pulseMesh(p) {
  const col = C.blue.clone().lerp(C.red, Math.min(1, Math.max(0, -Math.log10(Math.max(p, 1e-6)) / 2)));
  const m = new THREE.Mesh(pulseGeo, new THREE.MeshBasicMaterial({ color: col }));
  const g = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: col, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  g.scale.set(0.55, 0.55, 1); g.material.opacity = 0.7; m.add(g); rack.add(m); m.userData.col = col; return m;
}

// ---------- interaction ----------
const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
let hovered = null, down = null, userSpin = true;
function pick(e) {
  const r = canvas.getBoundingClientRect();
  ptr.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ptr, camera);
  const hit = ray.intersectObjects(blocks.map(b => b.mesh), false)[0];
  return hit ? hit.object.userData.block : null;
}
function showTip(b, e) {
  if (!b) { tipEl.style.display = "none"; return; }
  const r = wrap.getBoundingClientRect();
  const name = b.part === "mlp" ? "MLP" : b.type === "full" ? "full attention" : "linear attention";
  const off = st.knock.has(b.k);
  const d = b.delta != null ? `<br>Lesion scan: loss ${b.delta >= 0 ? "+" : ""}${b.delta.toFixed(3)} when off` : "";
  const act = !live ? "Wake Bonsai to operate." : off ? "Click to switch it back on." : "Click to switch it off.";
  tipEl.innerHTML = `<b>Layer ${b.layer} · ${name}</b>${off ? ` <span style="color:#ff8a82">off</span>` : ""}${d}<br><span style="color:#98989f">${act}</span>`;
  tipEl.style.display = "block";
  const x = e.clientX - r.left + 14, y = e.clientY - r.top + 14;
  tipEl.style.left = Math.min(x, r.width - tipEl.offsetWidth - 8) + "px";
  tipEl.style.top = Math.min(y, r.height - tipEl.offsetHeight - 8) + "px";
}
canvas.addEventListener("pointermove", e => {
  if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) down.drag = true;
  hovered = down?.drag ? null : pick(e);
  wrap.classList.toggle("hot", !!hovered); showTip(hovered, e);
});
canvas.addEventListener("pointerleave", () => { hovered = null; showTip(null); wrap.classList.remove("hot"); });
canvas.addEventListener("pointerdown", e => { down = { x: e.clientX, y: e.clientY, drag: false }; });
canvas.addEventListener("pointerup", e => {
  const wasDrag = down?.drag; down = null; if (wasDrag) return;
  const b = pick(e); if (!b) return;
  if (window.brainAPI) { window.brainAPI.toggle(b.k); b.flash = 1; }
  else { badge.textContent = "Wake Bonsai first, then click blocks to switch them off"; }
  setTimeout(() => showTip(pick(e), e), 30);
});
controls.addEventListener("start", () => { if (spinning) setSpin(false); });
function setSpin(on) { spinning = userSpin = on; wake(); const b = $("s3Spin"); b.textContent = on ? "Pause rotation" : "Rotate"; b.setAttribute("aria-pressed", on); }
$("s3Spin").onclick = () => setSpin(!spinning);
$("s3Reset").onclick = () => { swayT = 0; resetView(); setSpin(true); };
function setColorBy(c) {
  st.colorBy = c; wake();
  for (const b of $("s3Color").children) b.classList.toggle("on", b.dataset.c === c);
  $("s3LegImpact").hidden = c !== "impact";
}
$("s3Color").onclick = e => { const b = e.target.closest("button[data-c]"); if (b && !b.disabled) setColorBy(b.dataset.c); };

// ---------- animation ----------
const tmpC = new THREE.Color(), tmpE = new THREE.Color();
let visible = true, last = performance.now(), wakeUntil = 0, lastDraw = 0, pending = true;
// software GL: one redraw per change (k = 1 below, so a single frame lands on the final state)
const wake = (ms = 1600) => { pending = true; wakeUntil = Math.max(wakeUntil, performance.now() + (SOFT ? 0 : ms)); };
for (const ev of ["brain:ready", "brain:damage", "brain:scan", ...(SOFT ? [] : ["brain:busy", "brain:token"])]) document.addEventListener(ev, () => wake());
for (const ev of ["pointermove", "pointerdown", "wheel"]) canvas.addEventListener(ev, () => wake(), { passive: true });
controls.addEventListener("change", () => wake(400));
new ResizeObserver(() => wake(300)).observe(wrap);
new IntersectionObserver(es => { visible = es[0].isIntersecting; }, { threshold: 0.01 }).observe(wrap);
function tick(now) {
  requestAnimationFrame(tick);
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  if (!visible || document.hidden) return;
  const animating = !SOFT && (spinning || st.pulses.length || st.random > 0 || (st.busy && now - st.busy < 8000));
  if (!animating && now > wakeUntil && !pending) return;  // idle: leave the GPU to the model
  if (SOFT && now - lastDraw < 250) return;                // software GL: a few frames a second at most
  lastDraw = now; pending = false;
  const k = SOFT ? 1 : 1 - Math.exp(-dt * 9);
  const busy = st.busy && now - st.busy < 8000;
  const wear = Math.min(1, st.random / 0.6);

  // pulses
  for (let i = st.pulses.length - 1; i >= 0; i--) {
    const p = st.pulses[i]; p.x += dt * (rackLen + 2) / 0.9;
    p.mesh.position.set(p.x, 0, 0);
    if (p.x > rackLen / 2 + 1) { rack.remove(p.mesh); p.mesh.material.dispose(); st.pulses.splice(i, 1); continue; }
    for (const b of blocks) if (Math.abs(b.x - p.x) < SPACING * 0.6) { b.flash = Math.max(b.flash, 0.55); b.flashCol = p.mesh.userData.col; }
  }
  spineMat.opacity = 0.25 + 0.25 * Math.min(1, st.pulses.length / 2) + (busy ? 0.15 * Math.sin(now / 120) : 0);

  for (const b of blocks) {
    const m = b.mesh.material, off = st.knock.has(b.k);
    const t = st.colorBy === "impact" && b.delta != null ? Math.sqrt(Math.max(0, b.delta) / st.maxDelta) : 0;
    // target appearance
    if (off) { tmpC.copy(C.off); tmpE.copy(C.off).multiplyScalar(0.28); }
    else if (st.colorBy === "impact") { tmpC.copy(C.graphite).lerp(C.hot, t * 0.85); tmpE.copy(C.hot).multiplyScalar(0.55 * t * t + 0.04 * t); }
    else { tmpC.copy(b.base).lerp(C.worn, wear * 0.75); tmpE.setRGB(0, 0, 0); }
    if (b.flash > 0.01) tmpE.lerp(b.flashCol || C.white, Math.min(1, b.flash) * 0.6);
    if (hovered === b) tmpE.add(C.white.clone().multiplyScalar(0.08));
    if (busy && !off) tmpE.add(C.blue.clone().multiplyScalar(0.05 + 0.05 * Math.sin(now / 140 + b.layer * 0.5)));
    m.color.lerp(tmpC, k); m.emissive.lerp(tmpE, k);
    const rough = off ? 0.5 : Math.min(0.85, b.baseRough + wear * 0.55);
    m.roughness += (rough - m.roughness) * k;
    m.metalness += ((off ? 0.35 : st.colorBy === "impact" ? 0.45 : 1) - m.metalness) * k;
    // pulled out of the rack when off; taller when it matters more (impact view)
    b.z += ((off ? 0.65 : 0) - b.z) * k;
    b.sy += ((st.colorBy === "impact" ? 0.55 + 0.9 * t : 1) - b.sy) * k;
    b.mesh.scale.y = b.sy;
    b.mesh.position.set(b.x, b.sign * (GAP + H * b.sy / 2), b.z);
    b.mesh.rotation.x = b.sign * -0.08 * (b.z / 0.65);
    b.flash *= Math.exp(-dt * 3.2);
  }

  // dust from random damage
  const nDust = SOFT ? 0 : Math.round(DUST * Math.min(1, st.random / 0.6));
  dustGeo.setDrawRange(0, nDust); dust.material.color.set(st.mode === "flip" ? 0xbf5af2 : 0xff6a61);
  for (let i = 0; i < nDust; i++) {
    dustPos[i * 3] += dustVel[i * 3]; dustPos[i * 3 + 1] += dustVel[i * 3 + 1]; dustPos[i * 3 + 2] += dustVel[i * 3 + 2];
    if (dustPos[i * 3 + 1] < -(GAP + H + 0.7)) resetDust(i, false);
  }
  if (nDust) dustGeo.attributes.position.needsUpdate = true;

  if (spinning) {
    const a0 = 0.42 * Math.sin(swayT * 0.22); swayT += dt; const a1 = 0.42 * Math.sin(swayT * 0.22);
    camera.position.sub(controls.target).applyAxisAngle(THREE.Object3D.DEFAULT_UP, a1 - a0).add(controls.target);
  }
  controls.update();
  renderer.render(scene, camera);
}

// preview until the model loads: same layout as Bonsai 2B (24 layers, full attention every 4th)
build(Array.from({ length: 24 }, (_, i) => ({ i, type: (i + 1) % 4 === 0 ? "full" : "linear" })));
if (window.brainAPI) { live = true; build(window.brainAPI.layers); const d = window.brainAPI.damaged(); st.knock = new Set(d.knock); st.random = d.random; }
resetView(); setBadge(); setSpin(spinning); wake(2500);
requestAnimationFrame(tick);
// for tests and console poking: screen position of a block's centre
window.brain3d = { project(k) { const b = blocks.find(b => b.k === k); if (!b) return null; const v = b.mesh.getWorldPosition(new THREE.Vector3()).project(camera); const r = canvas.getBoundingClientRect(); return { x: r.left + (v.x + 1) / 2 * r.width, y: r.top + (1 - v.y) / 2 * r.height }; }, setSpin };
