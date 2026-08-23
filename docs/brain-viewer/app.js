/**
 * app.js — 3D Brain Viewer
 *
 * ── Coordinate system ───────────────────────────────────────────────────────
 * Marching cubes runs on a (Z, H, W) volume with spacing [dz, dy, dx].
 * trimesh exports vertex coordinates as-is into GLB.
 * Three.js GLTFLoader imports them unchanged, so in world space:
 *
 *   Three.js X  ←  verts[:,0] = Z_voxel × dz   (axial / slice-depth)
 *   Three.js Y  ←  verts[:,1] = H_voxel × dy   (height  / coronal)
 *   Three.js Z  ←  verts[:,2] = W_voxel × dx   (width   / sagittal)
 *
 * ── Clipping convention ──────────────────────────────────────────────────────
 * THREE.Plane keeps geometry where:  dot(normal, point) + constant ≥ 0
 *
 * To keep geometry where  point[dim] ≤ worldPos  (show the "lower" half):
 *   normal   = unit vector in the -dim direction
 *   constant = worldPos
 *   → dot(-ê_dim, point) + worldPos ≥ 0  →  -point[dim] + worldPos ≥ 0  →  point[dim] ≤ worldPos ✓
 *
 * Slider at 100 % → worldPos = box.max[dim] → entire brain is kept.
 * Slider at   0 % → worldPos = box.min[dim] → nothing is kept.
 *
 * ── Slice plane orientation ──────────────────────────────────────────────────
 * PlaneGeometry default: lies in XY, normal along +Z.
 *   axial    (cut ⊥ X):  rotation.y = +π/2    → normal along +X
 *   coronal  (cut ⊥ Y):  rotation.x = -π/2    → normal along +Y
 *   sagittal (cut ⊥ Z):  no rotation           → normal along +Z
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader }    from 'three/addons/loaders/GLTFLoader.js';

// ── Base URL (works under any sub-path, including GitHub Pages) ──────────────
const BASE = new URL('.', import.meta.url).href;
function asset(rel) { return BASE + rel; }

// ── DOM refs ─────────────────────────────────────────────────────────────────
const container       = document.getElementById('canvas-container');
const loadingOverlay  = document.getElementById('loading-overlay');
const loadingSub      = document.getElementById('loading-sub');
const errorOverlay    = document.getElementById('error-overlay');
const errorBody       = document.getElementById('error-body');
const clipSlider      = document.getElementById('clip-slider');
const clipVal         = document.getElementById('clip-val');
const opacitySlider   = document.getElementById('opacity-slider');
const opacityVal      = document.getElementById('opacity-val');
const contrastSlider  = document.getElementById('contrast-slider');
const contrastVal     = document.getElementById('contrast-val');
const brightnessSlider= document.getElementById('brightness-slider');
const brightnessVal   = document.getElementById('brightness-val');
const autoRotateToggle= document.getElementById('autorotate-toggle');
const resetBtn        = document.getElementById('reset-btn');
const screenshotBtn   = document.getElementById('screenshot-btn');
const axisBtns        = document.querySelectorAll('.axis-btn');
const variantGroup    = document.getElementById('variant-group');
const variantToggle   = document.getElementById('variant-toggle');
const panelEl         = document.getElementById('panel');
const panelToggle     = document.getElementById('panel-toggle');
const metricsToggle   = document.getElementById('metrics-toggle');
const tumorGroupEl    = document.getElementById('tumor-group');
const tumorDivider    = document.getElementById('tumor-divider');
const tumorToggle     = document.getElementById('tumor-toggle');
const tumorOpSlider   = document.getElementById('tumor-opacity-slider');
const tumorOpVal      = document.getElementById('tumor-opacity-val');
const tumorLegend     = document.getElementById('tumor-legend');

// ── Renderer ──────────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.localClippingEnabled = true;       // required for per-material clipping planes
container.appendChild(renderer.domElement);

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x080810);

const camera = new THREE.PerspectiveCamera(
  45, container.clientWidth / container.clientHeight, 0.1, 2000
);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance   = 10;
controls.maxDistance   = 1000;

// ── Lighting ──────────────────────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0xfff0e0, 0.5));

const keyLight = new THREE.DirectionalLight(0xffeedd, 1.1);
keyLight.position.set(200, 300, 200);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xd0e8ff, 0.35);
fillLight.position.set(-200, -100, 100);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0xffe0c0, 0.2);
rimLight.position.set(0, -300, -200);
scene.add(rimLight);

// ── Clipping plane (shared by all brain shell meshes) ─────────────────────────
// Initialised to a safe default (keeps everything).
const clipPlane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 1e9);

// The tumour is cut fractionally deeper than the brain. Its cross-section would
// otherwise be exactly coplanar with the MRI slice quad and z-fight with it.
const TUMOR_CLIP_EPS = 0.6;   // mm
const tumorClipPlane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 1e9);

// ── Per-axis configuration ────────────────────────────────────────────────────
// normal points in the -dim direction so the plane keeps dim ≤ worldPos.
const AXIS_CFG = {
  axial:    { dim: 'x', normal: new THREE.Vector3(-1,  0,  0) },
  coronal:  { dim: 'y', normal: new THREE.Vector3( 0, -1,  0) },
  sagittal: { dim: 'z', normal: new THREE.Vector3( 0,  0, -1) },
};

// Shell opacity used whenever the tumour layer is on: opaque enough to read as
// a brain, sheer enough to see the tumour suspended inside it.
const SHELL_OPACITY_WITH_TUMOR = 0.35;

// ── App state ─────────────────────────────────────────────────────────────────
const state = {
  axis:       'axial',
  clipT:      1.0,          // 1 = no cut, 0 = fully cut
  opacity:    SHELL_OPACITY_WITH_TUMOR,
  contrast:   1.0,
  brightness: 1.0,
  autoRotate: false,
  showTumor:  true,         // tumour visible on load; the shell dims to suit
  tumorOpacity: 1.0,
  hasTumor:   false,        // whether this variant shipped a tumour mesh at all
  meta:       null,
  metrics:    null,         // assets/<variant>/metrics.json (optional, may be null)
  brainBox:   null,
  variants:   [{ id: 'default', label: 'Brain', base: 'assets/' }],
  variant:    0,            // index into state.variants (the active volume)
  firstLoad:  true,
};

// Asset base for the active variant (e.g. 'assets/initial/' or legacy 'assets/').
function vbase() { return state.variants[state.variant].base; }
function vid()   { return state.variants[state.variant].id; }

let brainGroup = null;    // THREE.Group from GLTFLoader
let tumorGroup  = null;   // tumour as it is now        (cyan)
let growthGroup = null;   // tissue predicted to be new (red), evolved variant only
let sliceQuad  = null;    // PlaneGeometry quad showing MRI at cut position
const texCache = {};      // "axis_idx" → THREE.Texture
const texLoader = new THREE.TextureLoader();

// ── UI helpers ────────────────────────────────────────────────────────────────
function setLoadingMsg(msg) { if (loadingSub) loadingSub.textContent = msg; }

function showError(msg) {
  loadingOverlay.classList.add('hidden');
  errorOverlay.classList.add('visible');
  if (msg) errorBody.textContent = msg;
}

function hideLoading() {
  loadingOverlay.style.transition = 'opacity 0.6s ease';
  loadingOverlay.style.opacity    = '0';
  setTimeout(() => { loadingOverlay.style.display = 'none'; }, 650);
}

// ── Slice count from metadata ─────────────────────────────────────────────────
function getSliceCount(axis) {
  if (!state.meta || !state.meta.slices) return 1;
  const s = state.meta.slices;
  // New format: slices.axial.count
  if (s[axis] && s[axis].count !== undefined) return s[axis].count;
  // Legacy flat format: slices.count
  if (s.count !== undefined) return s.count;
  return 1;
}

// ── Texture loader with per-axis path ─────────────────────────────────────────
function loadSliceTex(axis, idx) {
  const base = vbase();
  const key = `${vid()}_${axis}_${idx}`;   // variant-scoped so volumes don't collide
  if (texCache[key]) return Promise.resolve(texCache[key]);

  return new Promise((resolve) => {
    const padded = String(idx).padStart(3, '0');
    // Primary: RGBA PNG with transparent background (new format)
    texLoader.load(
      asset(`${base}slices/${axis}/slice_${padded}.png`),
      (tex) => { tex.colorSpace = THREE.SRGBColorSpace; texCache[key] = tex; resolve(tex); },
      undefined,
      () => {
        // Fallback A: legacy flat-directory PNG
        texLoader.load(
          asset(`${base}slices/slice_${padded}.png`),
          (tex) => { tex.colorSpace = THREE.SRGBColorSpace; texCache[key] = tex; resolve(tex); },
          undefined,
          () => {
            // Fallback B: legacy JPG (opaque)
            texLoader.load(
              asset(`${base}slices/${axis}/slice_${padded}.jpg`),
              (tex) => { tex.colorSpace = THREE.SRGBColorSpace; texCache[key] = tex; resolve(tex); },
              undefined,
              () => resolve(null)
            );
          }
        );
      }
    );
  });
}

// ── Build / rebuild the slice quad for a given axis ───────────────────────────
function rebuildSliceQuad(box, axis) {
  if (sliceQuad) {
    scene.remove(sliceQuad);
    sliceQuad.geometry.dispose();
    sliceQuad.material.dispose();
    sliceQuad = null;
  }

  // Prefer the cropped world dimensions stored in metadata — these match the
  // exported slice images exactly (no black border).  Fall back to bbox size.
  const sm = state.meta && state.meta.slices && state.meta.slices[axis];
  let qw, qh;
  if (sm && sm.world_w && sm.world_h) {
    qw = sm.world_w;
    qh = sm.world_h;
  } else {
    const s = new THREE.Vector3();
    box.getSize(s);
    if      (axis === 'axial')   { qw = s.z; qh = s.y; }
    else if (axis === 'coronal') { qw = s.x; qh = s.z; }
    else                         { qw = s.x; qh = s.y; }
  }

  const geo = new THREE.PlaneGeometry(qw, qh);
  const mat = new THREE.MeshBasicMaterial({
    color:       new THREE.Color(1, 1, 1),
    side:        THREE.DoubleSide,
    transparent: true,        // needed for RGBA PNG alpha to work
    alphaTest:   0.02,        // discard pixels where alpha < 2% (background)
    depthWrite:  true,
    depthTest:   true,
  });
  sliceQuad = new THREE.Mesh(geo, mat);
  sliceQuad.visible = false;
  scene.add(sliceQuad);
}

// ── Core update: sync clip plane + slice quad with current state ──────────────
async function updateScene() {
  if (!brainGroup || !state.brainBox || !state.meta) return;

  const box          = state.brainBox;
  const { dim, normal } = AXIS_CFG[state.axis];

  const minVal   = box.min[dim];
  const maxVal   = box.max[dim];
  const worldPos = minVal + state.clipT * (maxVal - minVal);

  // ── 1. Update clipping plane
  // Keeps geometry where: dot(-ê_dim, point) + worldPos ≥ 0  →  point[dim] ≤ worldPos
  clipPlane.normal.copy(normal);    // already points in -dim direction
  clipPlane.constant = worldPos;

  // Cut the tumour marginally deeper so its cross-section sits just behind the
  // slice quad rather than coplanar with it.
  tumorClipPlane.normal.copy(normal);
  tumorClipPlane.constant = worldPos - TUMOR_CLIP_EPS;

  // ── 2. Shell opacity
  brainGroup.traverse((child) => {
    if (!child.isMesh) return;
    child.material.opacity     = state.opacity;
    child.material.transparent = state.opacity < 1.0;
    // A transparent DoubleSide shell that still writes depth self-occludes and
    // culls whatever is inside it. Drop depth writes while it is see-through,
    // and draw it after the tumour so the blend lands on top.
    child.material.depthWrite  = state.opacity >= 1.0;
    child.renderOrder          = 1;
  });

  // ── 2b. Tumour layer
  for (const g of [tumorGroup, growthGroup]) {
    if (!g) continue;
    g.visible = state.showTumor;
    g.traverse((child) => {
      if (!child.isMesh) return;
      child.material.opacity     = state.tumorOpacity;
      child.material.transparent = state.tumorOpacity < 1.0;
      child.material.depthWrite  = state.tumorOpacity >= 1.0;
    });
  }

  // ── 3. Slice quad
  if (!sliceQuad) return;

  const showSlice = state.clipT < 0.995;
  sliceQuad.visible = showSlice;

  if (!showSlice) return;

  // Position quad at the cut plane.
  // Use crop center from metadata for the two in-plane axes so the MRI image
  // lands exactly where the mesh cross-section is (no offset from black border).
  const meshCenter = new THREE.Vector3();
  box.getCenter(meshCenter);
  const pos = meshCenter.clone();
  pos[dim] = worldPos;  // set the clip-axis position

  const sm = state.meta && state.meta.slices && state.meta.slices[state.axis];
  if (sm && sm.center_world) {
    const cw = sm.center_world;  // [Three.js-X center, Three.js-Y center, Three.js-Z center]
    if (dim === 'x') { pos.y = cw[1]; pos.z = cw[2]; }
    else if (dim === 'y') { pos.x = cw[0]; pos.z = cw[2]; }
    else                  { pos.x = cw[0]; pos.y = cw[1]; }
  }
  sliceQuad.position.copy(pos);

  // Orient quad perpendicular to cut axis
  sliceQuad.rotation.set(0, 0, 0);
  if      (dim === 'x') sliceQuad.rotation.y = Math.PI / 2;   // face toward +X
  else if (dim === 'y') sliceQuad.rotation.x = -Math.PI / 2;  // face toward +Y
  // dim === 'z': PlaneGeometry default already faces +Z

  // Map worldPos → slice index (0 = box.min side, nSlices-1 = box.max side)
  const nSlices  = getSliceCount(state.axis);
  const sliceIdx = Math.round(
    Math.max(0, Math.min(1, (worldPos - minVal) / Math.max(maxVal - minVal, 1e-6)))
    * (nSlices - 1)
  );

  const tex = await loadSliceTex(state.axis, sliceIdx);
  if (tex && sliceQuad.material.map !== tex) {
    sliceQuad.material.map = tex;
    sliceQuad.material.needsUpdate = true;
  }

  // Brightness + contrast multiply the texture colour
  const v = state.brightness * state.contrast;
  sliceQuad.material.color.setScalar(Math.min(v, 4.0));
}

// ── Load volume metadata ──────────────────────────────────────────────────────
async function loadMeta() {
  setLoadingMsg('Loading metadata…');
  const resp = await fetch(asset(vbase() + 'volume_meta.json'));
  if (!resp.ok) throw new Error(`volume_meta.json returned HTTP ${resp.status}`);
  return resp.json();
}

// ── Load brain GLB ────────────────────────────────────────────────────────────
function loadBrainGLB() {
  return new Promise((resolve, reject) => {
    setLoadingMsg('Loading brain mesh…');
    new GLTFLoader().load(
      asset(vbase() + 'brain_surface.glb'),
      (gltf) => {
        gltf.scene.traverse((child) => {
          if (!child.isMesh) return;
          // Use vertex colours baked by generate_brain.py (MRI intensity mapped
          // to a warm anatomical ramp).  vertexColors:true means the material
          // color property acts as a multiplier — keep it white so colours are
          // unmodified by default.
          const hasVC = child.geometry.attributes.color !== undefined;
          child.material = new THREE.MeshStandardMaterial({
            vertexColors:   hasVC,
            color:          new THREE.Color(hasVC ? 0xffffff : 0xc8907a),
            roughness:      0.75,
            metalness:      0.04,
            side:           THREE.DoubleSide,
            clippingPlanes: [clipPlane],
            clipShadows:    true,
          });
        });
        resolve(gltf.scene);
      },
      (xhr) => {
        if (xhr.total) setLoadingMsg(`Mesh: ${Math.round(xhr.loaded / xhr.total * 100)} %`);
      },
      reject
    );
  });
}

/* ── Tumour meshes (optional) ──────────────────────────────────────────────
   assets/<variant>/tumor_surface.glb  — the tumour as it is now, cyan
   assets/<variant>/tumor_growth.glb   — tissue the PINN predicts, red
   Both optional, exactly like metrics.json: a 404 means this variant has no
   tumour layer, never an error and never a blocked brain mesh. The GLBs carry
   flat vertex colours baked by generate_brain.py, so the material only has to
   respect them.                                                            */
function loadTumorGLB(file) {
  return new Promise((resolve) => {
    new GLTFLoader().load(
      asset(vbase() + file),
      (gltf) => {
        gltf.scene.traverse((child) => {
          if (!child.isMesh) return;
          child.material = new THREE.MeshStandardMaterial({
            vertexColors:   child.geometry.attributes.color !== undefined,
            color:          new THREE.Color(0xffffff),
            roughness:      0.45,
            metalness:      0.0,
            emissive:       new THREE.Color(0x101820),
            side:           THREE.DoubleSide,
            clippingPlanes: [tumorClipPlane],
            clipShadows:    true,
          });
          // Drawn before the shell and writing depth, so the translucent brain
          // blends over it instead of culling it.
          child.renderOrder = 0;
          child.material.depthWrite = true;
        });
        resolve(gltf.scene);
      },
      undefined,
      () => resolve(null),        // 404 or parse failure → no tumour layer
    );
  });
}

// Show/hide the tumour control group and build its legend from metrics.json,
// whose tumor.volume_ml / growth.delta_volume_ml are the same numbers the
// quantitative panel is already displaying.
function renderTumorControls() {
  const on = state.hasTumor;
  if (tumorGroupEl) tumorGroupEl.style.display = on ? '' : 'none';
  if (tumorDivider) tumorDivider.style.display = on ? '' : 'none';
  if (!on || !tumorLegend) return;

  const m = state.metrics || {};
  const rows = [];
  const nowMl = m.tumor && typeof m.tumor.volume_ml === 'number'
    ? (growthGroup ? m.tumor.volume_ml - (m.growth ? m.growth.delta_volume_ml : 0)
                   : m.tumor.volume_ml)
    : null;
  rows.push(['#7ec8c8', 'Current',
             Number.isFinite(nowMl) ? nowMl.toFixed(1) + ' mL' : '']);
  if (growthGroup && m.growth && typeof m.growth.delta_volume_ml === 'number') {
    rows.push(['#ff6a6a', 'New growth', '+' + m.growth.delta_volume_ml.toFixed(1) + ' mL']);
  }
  tumorLegend.innerHTML = rows.map(([c, name, val]) =>
    `<div class="legend-row"><span class="legend-dot" style="background:${c}"></span>` +
    `<span class="legend-name">${name}</span><span class="legend-val">${val}</span></div>`
  ).join('');
}

// ── Fit camera to bounding box ────────────────────────────────────────────────
function fitCamera(box) {
  const center = new THREE.Vector3();
  const size   = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);
  const dist = Math.max(size.x, size.y, size.z) * 1.9;
  camera.position.copy(center).add(new THREE.Vector3(0, 0, dist));
  controls.target.copy(center);
  controls.update();
}

// ── Variant manifest (initial / evolved / …) ──────────────────────────────────
async function loadManifest() {
  // Optional: assets/manifest.json lists the available volumes. If absent, the
  // viewer falls back to the legacy single flat assets/ layout.
  try {
    const resp = await fetch(asset('assets/manifest.json'));
    if (!resp.ok) return;
    const data = await resp.json();
    if (Array.isArray(data.variants) && data.variants.length) {
      state.variants = data.variants.map(v => ({
        id: v.id, label: v.label || v.id, base: `assets/${v.dir || v.id}/`,
      }));
      const di = state.variants.findIndex(v => v.id === data.default);
      state.variant = di >= 0 ? di : 0;
    }
  } catch (_) { /* keep flat-layout default */ }
}

function buildVariantUI() {
  if (!variantToggle || !variantGroup) return;
  if (state.variants.length < 2) return;       // nothing to toggle
  variantGroup.style.display = '';
  variantToggle.innerHTML = '';
  state.variants.forEach((v, i) => {
    const btn = document.createElement('button');
    btn.className = 'variant-btn' + (i === state.variant ? ' active' : '');
    btn.textContent = v.label;
    btn.dataset.idx = String(i);
    btn.addEventListener('click', () => switchVariant(i));
    variantToggle.appendChild(btn);
  });
}

function disposeGroup(g) {
  if (!g) return;
  scene.remove(g);
  g.traverse((c) => {
    if (!c.isMesh) return;
    c.geometry.dispose();
    if (Array.isArray(c.material)) c.material.forEach(m => m.dispose());
    else c.material.dispose();
  });
}

function disposeBrain() {
  disposeGroup(brainGroup);
  disposeGroup(tumorGroup);
  disposeGroup(growthGroup);
  brainGroup = tumorGroup = growthGroup = null;
}

// Load (or hot-swap to) a variant. keepView=true preserves camera + clip so the
// initial and evolved volumes can be compared from the exact same viewpoint.
// ── Mobile drawers (≤900px) ──────────────────────────────────────────────────
// Above 900px the two panels are fixed rails and these toggles are display:none,
// so every function here is a no-op on desktop. Below it, only one panel may be
// open at a time — two open drawers would leave no canvas visible on a phone.
function setDrawer(el, btn, open) {
  if (!el || !btn) return;
  el.classList.toggle('open', open);
  btn.classList.toggle('active', open);
  btn.setAttribute('aria-expanded', String(open));
}

function toggleDrawer(which) {
  const mEl = document.getElementById('metrics-panel');
  const wantPanel   = which === 'panel'   && !panelEl.classList.contains('open');
  const wantMetrics = which === 'metrics' && !!mEl && !mEl.classList.contains('open');
  setDrawer(panelEl, panelToggle, wantPanel);
  setDrawer(mEl, metricsToggle, wantMetrics);
}

// Used by the guided tour, which must reveal the metrics panel before pointing at it.
function openMetricsDrawer() {
  const mEl = document.getElementById('metrics-panel');
  if (!mEl || mEl.style.display === 'none') return;
  setDrawer(panelEl, panelToggle, false);
  setDrawer(mEl, metricsToggle, true);
}

function closeDrawers() {
  setDrawer(panelEl, panelToggle, false);
  setDrawer(document.getElementById('metrics-panel'), metricsToggle, false);
}

if (panelToggle)   panelToggle.addEventListener('click',   () => toggleDrawer('panel'));
if (metricsToggle) metricsToggle.addEventListener('click', () => toggleDrawer('metrics'));

/* ── Quantitative metrics (optional) ───────────────────────────────────────
   assets/<variant>/metrics.json, schema 'oracle.metrics/1'. The file is
   optional by contract: a 404, unparseable JSON, or an unsupported major
   version all mean "hide the panel" — never an error, never a blocked mesh.
   Rendering wrong numbers on a medical page is worse than rendering none,
   so an unrecognised major version refuses rather than guesses.          */
async function loadMetrics() {
  try {
    const resp = await fetch(asset(vbase() + 'metrics.json'));
    if (!resp.ok) return null;
    const d = await resp.json();
    if (!d || typeof d.schema !== 'string' || !d.schema.startsWith('oracle.metrics/1')) {
      console.warn('[brain-viewer] unsupported metrics schema:', d && d.schema);
      return null;
    }
    return d;
  } catch (_) {
    return null;
  }
}

function hideMetrics() {
  const p = document.getElementById('metrics-panel');
  if (p) p.style.display = 'none';
  // No metrics → no drawer toggle. '' hands display back to the media query.
  if (metricsToggle) metricsToggle.style.display = 'none';
  setDrawer(p, metricsToggle, false);
}

// Set a value, or hide the whole row when the datum is absent.
function setMetric(id, value, fmt) {
  const el = document.getElementById(id);
  if (!el) return;
  const row = el.closest('.control-group') || el.parentElement;
  const missing = value === null || value === undefined ||
                  (typeof value === 'number' && !Number.isFinite(value));
  if (missing) {
    if (row) row.style.display = 'none';
    return;
  }
  if (row) row.style.display = '';
  el.textContent = fmt ? fmt(value) : String(value);
}

function renderMetrics() {
  const panel = document.getElementById('metrics-panel');
  if (!panel) return;
  const m = state.metrics;
  if (!m) { hideMetrics(); return; }
  panel.style.display = '';
  if (metricsToggle) metricsToggle.style.display = '';

  const t = m.tumor || {};
  setMetric('m-variant',   m.label || m.variant || null);
  setMetric('m-volume-ml', t.volume_ml,              v => v.toFixed(2) + ' mL');
  setMetric('m-voxels',    t.voxels,                 v => v.toLocaleString());
  setMetric('m-diameter',  t.equivalent_diameter_mm, v => v.toFixed(1) + ' mm');
  setMetric('m-spacing',   Array.isArray(m.voxel && m.voxel.spacing_mm) ? m.voxel.spacing_mm : null,
                                                     a => a.map(x => x.toFixed(2)).join(' × ') + ' mm');

  // Growth: null on the initial variant by contract.
  const g = m.growth;
  const gGroup = document.getElementById('m-growth-group');
  if (gGroup) gGroup.style.display = g ? '' : 'none';
  if (g) {
    setMetric('m-growth-ratio', g.ratio,            v => v.toFixed(2) + '×');
    setMetric('m-growth-delta', g.delta_volume_ml,  v => (v >= 0 ? '+' : '') + v.toFixed(2) + ' mL');
    setMetric('m-horizon',      m.horizon_days,     v => '+' + v.toFixed(0) + ' d');
  }

  // Classification: null when no classifier ran.
  const c = m.classification;
  const cGroup = document.getElementById('m-cls-group');
  if (cGroup) cGroup.style.display = c ? '' : 'none';
  if (c) {
    setMetric('m-cls-label', c.label || null, v => String(v).toUpperCase());
    setMetric('m-cls-conf',  c.confidence,    v => (v * 100).toFixed(1) + '%');

    const bars = document.getElementById('m-cls-bars');
    if (bars) {
      bars.innerHTML = '';
      const probs = c.probabilities || {};
      Object.keys(probs).forEach(k => {
        const p = probs[k];
        if (typeof p !== 'number' || !Number.isFinite(p)) return;
        const row = document.createElement('div');
        row.className = 'prob-row' + (k === c.label ? ' is-top' : '');
        const pct = (p * 100).toFixed(1);
        row.innerHTML =
          `<span class="prob-name"></span>` +
          `<span class="prob-bar"><span class="prob-fill" style="width:${pct}%"></span></span>` +
          `<span class="prob-pct">${pct}%</span>`;
        row.querySelector('.prob-name').textContent = k;   // textContent: never inject data as HTML
        bars.appendChild(row);
      });
    }

    const ood = document.getElementById('m-ood');
    if (ood) {
      const show = c.out_of_distribution && c.ood_note;
      ood.style.display = show ? '' : 'none';
      ood.textContent = show ? c.ood_note : '';
    }
  }

  // C1 rule 8: initial and evolved are thresholded on *different* quantities,
  // so the source and threshold must always be visible, never implied.
  const src = document.getElementById('m-source');
  if (src) {
    const parts = [];
    if (t.source) parts.push(String(t.source).replace(/_/g, ' '));
    if (typeof t.threshold === 'number' && Number.isFinite(t.threshold)) {
      parts.push('threshold ' + t.threshold);
    }
    src.style.display = parts.length ? '' : 'none';
    src.textContent = parts.length ? 'Source: ' + parts.join(' · ') : '';
  }
}

async function loadVariant(index, keepView) {
  state.variant = index;
  state.meta = await loadMeta();

  // Optional metrics — isolated so a render bug can never reject the promise
  // init() awaits (which would raise the full-screen "assets not found" overlay).
  state.metrics = await loadMetrics();
  try {
    renderMetrics();
  } catch (e) {
    console.warn('[brain-viewer] metrics render failed', e);
    hideMetrics();
  }

  // On the very first load honour the axis used during preprocessing
  if (!keepView) {
    const primaryAxis = state.meta?.volume?.primary_axis || state.meta?.volume?.axis;
    if (primaryAxis && AXIS_CFG[primaryAxis]) {
      state.axis = primaryAxis;
      axisBtns.forEach(b => b.classList.toggle('active', b.dataset.axis === state.axis));
    }
  }

  // ~54 MB per variant (4.6 MB GLB + 3 x 128 slice PNGs). Say so on touch
  // devices, where this is far more likely to be a metered connection.
  if (state.firstLoad && window.matchMedia('(pointer: coarse)').matches) {
    setLoadingMsg('Loading ~54 MB — best on Wi-Fi');
  }

  disposeBrain();
  brainGroup = await loadBrainGLB();
  scene.add(brainGroup);

  // Tumour layer — resolves to null when this variant has no mesh, in which
  // case the controls stay hidden and everything else behaves as before.
  setLoadingMsg('Loading tumour mesh…');
  [tumorGroup, growthGroup] = await Promise.all([
    loadTumorGLB('tumor_surface.glb'),
    loadTumorGLB('tumor_growth.glb'),
  ]);
  if (tumorGroup)  scene.add(tumorGroup);
  if (growthGroup) scene.add(growthGroup);
  state.hasTumor = !!(tumorGroup || growthGroup);
  renderTumorControls();

  const box = new THREE.Box3().setFromObject(brainGroup);
  state.brainBox = box;
  if (!keepView) fitCamera(box);
  rebuildSliceQuad(box, state.axis);

  // Without a tumour there is nothing to see through the shell, so open opaque.
  setShellOpacity(state.hasTumor ? SHELL_OPACITY_WITH_TUMOR : 1.0);

  setLoadingMsg('Preloading textures…');
  const nPre = Math.min(6, getSliceCount(state.axis));
  await Promise.all(Array.from({ length: nPre }, (_, i) => loadSliceTex(state.axis, i)));

  await updateScene();
}

async function switchVariant(index) {
  if (index === state.variant || !state.variants[index]) return;
  variantToggle.querySelectorAll('.variant-btn').forEach((b, i) =>
    b.classList.toggle('active', i === index));
  try {
    await loadVariant(index, /* keepView */ true);   // compare from the same viewpoint
  } catch (err) {
    console.error('[brain-viewer] variant switch failed', err);
    showError(`Could not load "${state.variants[index].label}": ${err.message}`);
  }
}

/* ── Tour surface ──────────────────────────────────────────────────────────
   The one export in this file. tour.js drives the viewer through these and
   through the real UI controls below — it never reaches into module state
   directly, so the tour cannot desynchronise the panel from the scene.    */
export const tourApi = {
  state, camera, controls,
  fitCamera, switchVariant, openMetricsDrawer, closeDrawers, setShellOpacity,
  els: { clipSlider, axisBtns, autoRotateToggle, opacitySlider, tumorToggle },
};

// ── Initialise ────────────────────────────────────────────────────────────────
async function init() {
  try {
    await loadManifest();
    buildVariantUI();
    await loadVariant(state.variant, /* keepView */ false);
    state.firstLoad = false;
    hideLoading();
    animate();

    // Optional, like metrics.json: always loaded (it installs its own start
    // button), but only autostarted by ?tour=1 — which is what makes the
    // screencast reproducible. A 404 or throw here leaves the viewer as-is.
    import('./tour.js')
      .then(m => m.installTour(tourApi, {
        autostart: new URLSearchParams(location.search).has('tour'),
      }))
      .catch(e => console.warn('[brain-viewer] tour unavailable', e));
  } catch (err) {
    console.error('[brain-viewer]', err);
    showError(
      `Load failed: ${err.message}.\n\n` +
      'Run  python run_pipeline.py  first, then serve this directory ' +
      'over HTTP (not file://).'
    );
  }
}

// ── Animation loop ────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.autoRotate = state.autoRotate;
  controls.update();
  renderer.render(scene, camera);
}

// ── Window resize ─────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {
  const w = container.clientWidth;
  const h = container.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
});

// ── UI event handlers ─────────────────────────────────────────────────────────
clipSlider.addEventListener('input', () => {
  state.clipT = parseInt(clipSlider.value, 10) / 100;
  clipVal.textContent = clipSlider.value + '%';
  updateScene();
});

opacitySlider.addEventListener('input', () => {
  state.opacity = parseInt(opacitySlider.value, 10) / 100;
  opacityVal.textContent = opacitySlider.value + '%';
  updateScene();
});

contrastSlider.addEventListener('input', () => {
  state.contrast = parseFloat(contrastSlider.value);
  contrastVal.textContent = state.contrast.toFixed(2);
  updateScene();
});

brightnessSlider.addEventListener('input', () => {
  state.brightness = parseFloat(brightnessSlider.value);
  brightnessVal.textContent = state.brightness.toFixed(2);
  updateScene();
});

autoRotateToggle.addEventListener('change', () => {
  state.autoRotate = autoRotateToggle.checked;
});

if (tumorToggle) tumorToggle.addEventListener('change', () => {
  state.showTumor = tumorToggle.checked;
  // Hiding the tumour has no reason to keep the brain see-through; showing it
  // again would be pointless behind an opaque shell. Move the shell with it,
  // unless the user has already dialled in something of their own.
  const target = state.showTumor ? SHELL_OPACITY_WITH_TUMOR : 1.0;
  const wasDefault = Math.abs(state.opacity -
    (state.showTumor ? 1.0 : SHELL_OPACITY_WITH_TUMOR)) < 0.01;
  if (wasDefault) setShellOpacity(target);
  updateScene();
});

if (tumorOpSlider) tumorOpSlider.addEventListener('input', () => {
  state.tumorOpacity = parseInt(tumorOpSlider.value, 10) / 100;
  tumorOpVal.textContent = tumorOpSlider.value + '%';
  updateScene();
});

axisBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    if (!AXIS_CFG[btn.dataset.axis]) return;
    state.axis = btn.dataset.axis;
    axisBtns.forEach(b => b.classList.toggle('active', b === btn));
    // Reset clip and rebuild quad geometry for the new axis
    clipSlider.value = '100';
    state.clipT = 1.0;
    clipVal.textContent = '100%';
    if (state.brainBox) rebuildSliceQuad(state.brainBox, state.axis);
    updateScene();
  });
});

// Keep the shell slider, its label and state in one place — three call sites
// set it now (reset, the tumour toggle, and first load).
function setShellOpacity(v) {
  state.opacity = v;
  const pct = Math.round(v * 100);
  opacitySlider.value = String(pct);
  opacityVal.textContent = pct + '%';
}

resetBtn.addEventListener('click', () => {
  if (!state.brainBox) return;
  fitCamera(state.brainBox);
  clipSlider.value = '100';    state.clipT = 1.0;  clipVal.textContent = '100%';
  setShellOpacity(state.hasTumor ? SHELL_OPACITY_WITH_TUMOR : 1.0);
  if (tumorToggle) { tumorToggle.checked = true; state.showTumor = true; }
  if (tumorOpSlider) {
    tumorOpSlider.value = '100'; state.tumorOpacity = 1.0;
    tumorOpVal.textContent = '100%';
  }
  updateScene();
});

screenshotBtn.addEventListener('click', () => {
  renderer.render(scene, camera);
  const link = document.createElement('a');
  link.href     = renderer.domElement.toDataURL('image/png');
  link.download = 'brain_viewer.png';
  link.click();
});

// ── Boot ──────────────────────────────────────────────────────────────────────
init();
