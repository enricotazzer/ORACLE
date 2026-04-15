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

// ── Per-axis configuration ────────────────────────────────────────────────────
// normal points in the -dim direction so the plane keeps dim ≤ worldPos.
const AXIS_CFG = {
  axial:    { dim: 'x', normal: new THREE.Vector3(-1,  0,  0) },
  coronal:  { dim: 'y', normal: new THREE.Vector3( 0, -1,  0) },
  sagittal: { dim: 'z', normal: new THREE.Vector3( 0,  0, -1) },
};

// ── App state ─────────────────────────────────────────────────────────────────
const state = {
  axis:       'axial',
  clipT:      1.0,          // 1 = no cut, 0 = fully cut
  opacity:    1.0,
  contrast:   1.0,
  brightness: 1.0,
  autoRotate: false,
  meta:       null,
  brainBox:   null,
};

let brainGroup = null;    // THREE.Group from GLTFLoader
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
  const key = `${axis}_${idx}`;
  if (texCache[key]) return Promise.resolve(texCache[key]);

  return new Promise((resolve) => {
    const padded = String(idx).padStart(3, '0');
    // Primary: RGBA PNG with transparent background (new format)
    texLoader.load(
      asset(`assets/slices/${axis}/slice_${padded}.png`),
      (tex) => { tex.colorSpace = THREE.SRGBColorSpace; texCache[key] = tex; resolve(tex); },
      undefined,
      () => {
        // Fallback A: legacy flat-directory PNG
        texLoader.load(
          asset(`assets/slices/slice_${padded}.png`),
          (tex) => { tex.colorSpace = THREE.SRGBColorSpace; texCache[key] = tex; resolve(tex); },
          undefined,
          () => {
            // Fallback B: legacy JPG (opaque)
            texLoader.load(
              asset(`assets/slices/${axis}/slice_${padded}.jpg`),
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

  // ── 2. Shell opacity
  brainGroup.traverse((child) => {
    if (!child.isMesh) return;
    child.material.opacity     = state.opacity;
    child.material.transparent = state.opacity < 1.0;
  });

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
  const resp = await fetch(asset('assets/volume_meta.json'));
  if (!resp.ok) throw new Error(`volume_meta.json returned HTTP ${resp.status}`);
  return resp.json();
}

// ── Load brain GLB ────────────────────────────────────────────────────────────
function loadBrainGLB() {
  return new Promise((resolve, reject) => {
    setLoadingMsg('Loading brain mesh…');
    new GLTFLoader().load(
      asset('assets/brain_surface.glb'),
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

          // Mirror the mesh along Three.js Z (= W_voxel axis) to match the
          // fliplr applied to axial slice exports.
          // Flip about the geometry's own Z centre so the position is unchanged.
          child.geometry.computeBoundingBox();
          const bz = child.geometry.boundingBox;
          const cz = (bz.min.z + bz.max.z) / 2;
          child.geometry.applyMatrix4(new THREE.Matrix4().set(
            1, 0,  0,      0,
            0, 1,  0,      0,
            0, 0, -1, 2 * cz,
            0, 0,  0,      1
          ));
          child.geometry.computeVertexNormals();
          child.geometry.computeBoundingBox();
          child.geometry.computeBoundingSphere();
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

// ── Initialise ────────────────────────────────────────────────────────────────
async function init() {
  try {
    state.meta = await loadMeta();

    // Honour the axis that was used during preprocessing
    const primaryAxis = state.meta?.volume?.primary_axis
                     || state.meta?.volume?.axis;       // legacy key
    if (primaryAxis && AXIS_CFG[primaryAxis]) {
      state.axis = primaryAxis;
      axisBtns.forEach(b => b.classList.toggle('active', b.dataset.axis === state.axis));
    }

    brainGroup = await loadBrainGLB();
    scene.add(brainGroup);

    const box = new THREE.Box3().setFromObject(brainGroup);
    state.brainBox = box;
    fitCamera(box);
    rebuildSliceQuad(box, state.axis);

    // Warm the texture cache for the first few slices
    setLoadingMsg('Preloading textures…');
    const nPre = Math.min(6, getSliceCount(state.axis));
    await Promise.all(
      Array.from({ length: nPre }, (_, i) => loadSliceTex(state.axis, i))
    );

    hideLoading();
    await updateScene();
    animate();

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

resetBtn.addEventListener('click', () => {
  if (!state.brainBox) return;
  fitCamera(state.brainBox);
  clipSlider.value = '100';    state.clipT    = 1.0;  clipVal.textContent    = '100%';
  opacitySlider.value = '100'; state.opacity  = 1.0;  opacityVal.textContent = '100%';
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
