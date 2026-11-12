// 3D neuron cloud for neurons.html. Each dot is one sampled MLP neuron (layout from neurons.mjs NeuronMap): height is
// the layer, neurons that fire together sit together. For every token the dots light up layer by layer, bottom to top
// (orange = far above its usual level, ink = above, teal = below), co-firing links between neighbouring layers glow when both ends
// are active, and a red line traces where the activity is centred in each layer.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const IDLE = new THREE.Color("#9aa4aa"), HOT = new THREE.Color("#ff5a1f"), WARM = new THREE.Color("#12161a"), COOL = new THREE.Color("#0c8f8a");

const LINK_IDLE = new THREE.Color("#e3e8e7").toArray(), LINK_HOT = new THREE.Color("#ff5a1f").toArray();
const VERT = `
attribute float aAct; attribute float aSel; uniform float uScale; varying float vAct; varying float vSel;
void main(){ vAct=aAct; vSel=aSel; vec4 mv=modelViewMatrix*vec4(position,1.0); gl_Position=projectionMatrix*mv;
  float a=abs(aAct); gl_PointSize=uScale*(0.5+(aAct>0.62?1.5:0.8*a*a)+1.4*aSel)/-mv.z; }`;
const FRAG = `
uniform vec3 cIdle; uniform vec3 cHot; uniform vec3 cWarm; uniform vec3 cCool; varying float vAct; varying float vSel;
void main(){ vec2 p=gl_PointCoord*2.0-1.0; float r=dot(p,p); if(r>1.0) discard;
  float a=clamp(vAct,-1.0,1.0); vec3 c=a>0.0?(a>0.62?cHot:mix(cIdle,cWarm,smoothstep(0.2,0.45,a))):mix(cIdle,cCool,smoothstep(0.35,0.7,-a));
  c=mix(c,vec3(0.07,0.086,0.1),vSel); float edge=smoothstep(1.0,0.8,r); float core=0.0;
  float alpha=edge*(0.55+0.45*smoothstep(0.15,0.5,abs(a)))+vSel; gl_FragColor=vec4(c+core,min(alpha,1.0));
  #include <colorspace_fragment>
}`;

export class NeuronCloud {
  constructor(host, map, { onHover, onPick, onView } = {}) {
    this.host = host; this.map = map; this.onHover = onHover; this.onPick = onPick; this.onView = onView;
    const r = this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    r.setPixelRatio(Math.min(2, devicePixelRatio)); r.setClearColor(0x000000, 0); host.appendChild(r.domElement);
    const scene = this.scene = new THREE.Scene();
    const Hgt = this.Hgt = (map.L - 1) * map.spacing;
    const cam = this.camera = new THREE.PerspectiveCamera(34, 1, 0.1, 200);
    cam.position.set(13.5, Hgt / 2 + 5.2, 15.5);
    const ctl = this.controls = new OrbitControls(cam, r.domElement);
    ctl.target.set(0, Hgt / 2, 0); ctl.enableDamping = true; ctl.dampingFactor = 0.08; ctl.autoRotate = true; ctl.autoRotateSpeed = 0.45;
    ctl.minDistance = 4; ctl.maxDistance = 40;
    // the mouse wheel scrolls the page; zoom with a trackpad pinch or ctrl+wheel (both arrive as wheel events with ctrlKey)
    r.domElement.addEventListener("wheel", e => { if (!e.ctrlKey) e.stopImmediatePropagation(); }, { capture: true });
    ctl.addEventListener("start", () => { ctl.autoRotate = false; clearTimeout(this.idleT); });
    ctl.addEventListener("end", () => { this.idleT = setTimeout(() => ctl.autoRotate = true, 8000); });

    // faint layer rings and a floor ring, so the eye can read height as depth through the network
    const ringMat = new THREE.LineBasicMaterial({ color: 0x12161a, transparent: true, opacity: 0.13 });
    const circle = n => { const g = new THREE.BufferGeometry(), p = []; for (let i = 0; i <= n; i++) { const a = i / n * Math.PI * 2; p.push(Math.cos(a), 0, Math.sin(a)); } g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3)); return g; };
    const cg = circle(96);
    this.rings = [];
    for (let l = 0; l < map.L; l += 1) { const m = new THREE.Line(cg, ringMat.clone()); m.scale.setScalar(map.radius * 1.04); m.position.y = l * map.spacing; scene.add(m); this.rings.push(m); }

    // neurons
    const S = map.S, g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(map.pos, 3));
    this.act = new Float32Array(S); this.target = new Float32Array(S); this.sel = new Float32Array(S);
    g.setAttribute("aAct", new THREE.BufferAttribute(this.act, 1).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute("aSel", new THREE.BufferAttribute(this.sel, 1).setUsage(THREE.DynamicDrawUsage));
    this.mat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false,
      uniforms: { uScale: { value: 300 }, cIdle: { value: IDLE }, cHot: { value: HOT }, cWarm: { value: WARM }, cCool: { value: COOL } } });
    this.points = new THREE.Points(g, this.mat); scene.add(this.points);

    // co-firing links between neighbouring layers
    const E = map.edges.length, ep = new Float32Array(E * 6); this.ecol = new Float32Array(E * 6);
    map.edges.forEach(([a, b], i) => { ep.set(map.pos.subarray(a * 3, a * 3 + 3), i * 6); ep.set(map.pos.subarray(b * 3, b * 3 + 3), i * 6 + 3); });
    const eg = new THREE.BufferGeometry(); eg.setAttribute("position", new THREE.BufferAttribute(ep, 3));
    eg.setAttribute("color", new THREE.BufferAttribute(this.ecol, 3).setUsage(THREE.DynamicDrawUsage));
    this.links = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55, depthWrite: false }));
    scene.add(this.links); this.paintLinks();

    // the token's path: where activity is centred in each layer
    this.pathMat = new THREE.MeshBasicMaterial({ color: 0xff5a1f, transparent: true, opacity: 0.95, depthTest: false });
    this.path = new THREE.Mesh(new THREE.BufferGeometry(), this.pathMat); this.path.renderOrder = 10; scene.add(this.path); this.pathN = -1;
    this.head = new THREE.Mesh(new THREE.SphereGeometry(0.08, 20, 14), new THREE.MeshBasicMaterial({ color: 0xff5a1f, depthTest: false })); this.head.renderOrder = 11; this.head.visible = false; scene.add(this.head);

    // hover and click on neurons
    this.ray = new THREE.Raycaster(); this.ray.params.Points.threshold = 0.07; this.mouse = new THREE.Vector2(); this.hover = -1; this.picked = -1;
    const pick = e => { const b = r.domElement.getBoundingClientRect(); this.mouse.set((e.clientX - b.left) / b.width * 2 - 1, -(e.clientY - b.top) / b.height * 2 + 1); this.ray.setFromCamera(this.mouse, cam); const h = this.ray.intersectObject(this.points)[0]; return h ? h.index : -1; };
    r.domElement.addEventListener("pointermove", e => { const i = pick(e); if (i !== this.hover) { this.hover = i; r.domElement.style.cursor = i >= 0 ? "pointer" : ""; this.onHover?.(i >= 0 ? { s: i, z: this.target[i] * 3, ...map.describe(i), x: e.clientX, y: e.clientY } : null); } else if (i >= 0) this.onHover?.({ s: i, z: this.target[i] * 3, ...map.describe(i), x: e.clientX, y: e.clientY }); });
    r.domElement.addEventListener("pointerleave", () => { this.hover = -1; this.onHover?.(null); });
    let down = null; r.domElement.addEventListener("pointerdown", e => down = [e.clientX, e.clientY]);
    r.domElement.addEventListener("pointerup", e => { if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return; this.pickNeuron(pick(e)); });

    new ResizeObserver(() => this.resize()).observe(host); this.resize();
    this.busy = false; this.visible = true; this.n = 0; this.t0 = 0; this.pathTarget = null;
    new IntersectionObserver(es => this.visible = es[0].isIntersecting).observe(host);
    const loop = () => { if (!this.running) return; requestAnimationFrame(loop); if (!this.visible || document.hidden) return; this.n++; if (this.busy && this.n % 2) return; this.frame(); };
    this.running = true; loop();
  }
  pickNeuron(i) { this.sel.fill(0); this.picked = i; if (i >= 0) this.sel[i] = 1; this.points.geometry.attributes.aSel.needsUpdate = true; this.onPick?.(i >= 0 ? { s: i, ...this.map.describe(i) } : null); }
  // show one token: z-scores from NeuronMap.activity; the wave climbs the layers over `ms`
  setKnocked(layers) {                           // layers with a switched-off block get an orange ring
    this.rings.forEach((m, l) => { const on = layers.has(l); m.material.color.set(on ? 0xff5a1f : 0x12161a); m.material.opacity = on ? 0.95 : 0.13; });
  }
  show(z, path, { ms = 650 } = {}) {
    for (let s = 0; s < this.map.S; s++) this.target[s] = Math.max(-1, Math.min(1, z[s] / 3));
    this.pathTarget = path; this.t0 = performance.now(); this.waveMs = ms;
  }
  paintLinks() {
    const { edges } = this.map, c = this.ecol, idle = LINK_IDLE, hot = LINK_HOT;
    edges.forEach(([a, b], i) => {
      const k = Math.min(1, Math.max(0, Math.min(this.act[a], this.act[b]) - 0.3) / 0.35);
      for (const o of [0, 3]) for (let ch = 0; ch < 3; ch++) c[i * 6 + o + ch] = idle[ch] + (hot[ch] - idle[ch]) * k;
    });
    this.links.geometry.attributes.color.needsUpdate = true;
  }
  frame() {
    const { map } = this, now = performance.now(), K = map.K, sp = this.waveMs / map.L;
    const reached = this.t0 ? Math.min(map.L, Math.floor((now - this.t0) / sp) + 1) : 0;
    let changed = false;
    for (let l = 0; l < map.L; l++) {
      const on = l < reached;
      for (let k = 0; k < K; k++) { const s = l * K + k, t = on ? this.target[s] : this.act[s] * 0.9, d = t - this.act[s]; if (Math.abs(d) > 0.002) { this.act[s] += d * (on ? 0.35 : 0.2); changed = true; } }
    }
    if (changed) { this.points.geometry.attributes.aAct.needsUpdate = true; if (this.n % 2 === 0) this.paintLinks(); }
    if (this.pathTarget) {
      const n = Math.max(1, reached);
      if (n !== this.pathN || this.pathSrc !== this.pathTarget) {
        this.pathN = n; this.pathSrc = this.pathTarget; this.path.geometry.dispose();
        const pts = []; for (let l = 0; l < n; l++) pts.push(new THREE.Vector3().fromArray(this.pathTarget, l * 3));
        this.path.geometry = n > 1 ? new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), n * 6, 0.022, 8, false) : new THREE.BufferGeometry();
        this.head.visible = true; this.head.position.copy(pts[n - 1]);
      }
    }
    this.controls.update(); this.renderer.render(this.scene, this.camera);
    if (this.onView && this.n % 10 === 0) { const sp = new THREE.Spherical().setFromVector3(this.camera.position.clone().sub(this.controls.target)); this.onView({ azimuth: (THREE.MathUtils.radToDeg(sp.theta) + 360) % 360, elevation: 90 - THREE.MathUtils.radToDeg(sp.phi), ruler: this.ruler() }); }
  }
  // screen positions for a layer ruler drawn beside the cloud: y of each layer's centre, x just left of the rings
  ruler() {
    const w = this.host.clientWidth, h = this.host.clientHeight, m = this.map, right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
    const px = v => { const p = v.clone().project(this.camera); return [(p.x + 1) / 2 * w, (1 - p.y) / 2 * h]; };
    const ys = Array.from({ length: m.L }, (_, l) => px(new THREE.Vector3(0, l * m.spacing, 0))[1]);
    const mid = new THREE.Vector3(0, (m.L - 1) * m.spacing / 2, 0), [cx] = px(mid), [ex] = px(mid.clone().addScaledVector(right, m.radius * 1.1));
    return { ys, x: cx - Math.abs(ex - cx) - 24 };
  }
  resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight; if (!w || !h) return;
    this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    this.mat.uniforms.uScale.value = h * 0.17 * Math.min(2, devicePixelRatio);
  }
  dispose() { this.running = false; this.renderer.dispose(); this.renderer.domElement.remove(); }
}
