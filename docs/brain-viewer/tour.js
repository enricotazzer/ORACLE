/* ─────────────────────────────────────────────────────────────────────────────
   Guided tour — the viewer plays itself.

   Loaded by a dynamic import from app.js and entirely optional: if this file
   404s or throws, app.js catches it and the viewer behaves exactly as it did
   before the tour existed. Nothing here is on the critical path.

   Two rules keep it honest:

   1. Drive the real UI, never a private copy of it. Beats set a slider's value
      and dispatch 'input', or call .click() on an axis button, so app.js's own
      handlers run — state, label text and updateScene() stay in sync with zero
      duplicated logic.
   2. Any *trusted* user gesture ends the tour immediately. The synthetic events
      this file dispatches carry isTrusted === false, so the tour never cancels
      itself.

   Captions read live values out of state.metrics rather than hard-coding them,
   so the numbers can never drift from what the panel is showing.
   ──────────────────────────────────────────────────────────────────────────── */

const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

let api      = null;
let card     = null;
let beats    = [];
let idx      = 0;
let gen      = 0;      // bumped on every start/stop; stale async frames bail out
let running  = false;

/* ── Timing primitives (all cancellable via `gen`) ─────────────────────────── */

const lerp = (a, b, t) => a + (b - a) * t;
const easeInOutCubic = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

function tween(ms, onFrame) {
  if (REDUCED) { onFrame(1); return Promise.resolve(); }
  const myGen = gen;
  return new Promise(resolve => {
    let done = false;
    const finish = (snap) => {
      if (done) return;
      done = true;
      clearTimeout(watchdog);
      if (snap && myGen === gen) onFrame(1);   // land on the target, not mid-tween
      resolve();
    };

    // requestAnimationFrame does not fire in a backgrounded tab. Without this
    // the promise never settles and the tour freezes on the current beat with
    // only Skip to escape it, so fall back to jumping to the end state.
    const watchdog = setTimeout(() => finish(true), ms * 3 + 500);

    const t0 = performance.now();
    (function step(now) {
      if (done) return;
      if (myGen !== gen) return finish(false);
      const t = Math.min(1, (now - t0) / ms);
      onFrame(easeInOutCubic(t));
      if (t < 1) requestAnimationFrame(step); else finish(false);
    })(performance.now());
  });
}

function wait(ms) {
  const myGen = gen;
  return new Promise(resolve => setTimeout(resolve, REDUCED ? Math.min(ms, 300) : ms))
    .then(() => myGen === gen);
}

/* ── Viewer actuators — all go through app.js's own event handlers ─────────── */

function setSlider(el, v) {
  if (!el || parseInt(el.value, 10) === v) return;
  el.value = String(v);
  el.dispatchEvent(new Event('input'));
}

async function clipTo(pct, ms = 2000) {
  const el = api.els.clipSlider;
  if (!el) return;
  const from = parseInt(el.value, 10);
  if (from === pct) return;
  await tween(ms, e => setSlider(el, Math.round(lerp(from, pct, e))));
}

function setAxis(axis) {
  const btn = Array.from(api.els.axisBtns).find(b => b.dataset.axis === axis);
  if (btn && !btn.classList.contains('active')) btn.click();   // also resets clip to 100%
}

function setTumor(on) {
  const el = api.els.tumorToggle;
  if (!el || el.checked === on) return;
  el.checked = on;
  el.dispatchEvent(new Event('change'));
}

// Straight through app.js's setter so slider, label and state stay in step.
function setShell(v) {
  if (!api.setShellOpacity) return;
  api.setShellOpacity(v);
}

async function shellTo(v, ms = 1200) {
  const from = api.state.opacity;
  if (Math.abs(from - v) < 0.01) return;
  await tween(ms, e => setShell(lerp(from, v, e)));
}

function setAutoRotate(on) {
  const el = api.els.autoRotateToggle;
  if (!el || el.checked === on) return;
  el.checked = on;
  el.dispatchEvent(new Event('change'));
}

// Centre and extent without importing THREE — Box3 exposes plain min/max vectors.
function boxInfo() {
  const b = api.state.brainBox;
  if (!b) return null;
  return {
    cx: (b.min.x + b.max.x) / 2,
    cy: (b.min.y + b.max.y) / 2,
    cz: (b.min.z + b.max.z) / 2,
    maxSize: Math.max(b.max.x - b.min.x, b.max.y - b.min.y, b.max.z - b.min.z),
  };
}

// Move to a direction *relative to the bounding box*, so the framing is
// identical whatever the volume's world scale turns out to be.
async function camTo(dir, factor = 1.9, ms = 1000) {
  const bi = boxInfo();
  if (!bi) return;
  const n = Math.hypot(dir[0], dir[1], dir[2]) || 1;
  const d = bi.maxSize * factor;
  const to = [bi.cx + (dir[0] / n) * d, bi.cy + (dir[1] / n) * d, bi.cz + (dir[2] / n) * d];

  const p = api.camera.position;
  const t = api.controls.target;
  const p0 = [p.x, p.y, p.z];
  const t0 = [t.x, t.y, t.z];

  await tween(ms, e => {
    api.camera.position.set(lerp(p0[0], to[0], e), lerp(p0[1], to[1], e), lerp(p0[2], to[2], e));
    api.controls.target.set(lerp(t0[0], bi.cx, e), lerp(t0[1], bi.cy, e), lerp(t0[2], bi.cz, e));
    api.controls.update();
  });
}

/* ── Metrics-panel highlighting ────────────────────────────────────────────── */

function rowsFor(ids) {
  return ids
    .map(id => document.getElementById(id))
    .filter(Boolean)
    .map(el => el.closest('.control-group') || el.parentElement)
    .filter(Boolean);
}

async function pulse(els, ms = 2000) {
  if (!els.length) return;
  api.openMetricsDrawer();                       // no-op above 900px
  els[0].scrollIntoView({ block: 'nearest', behavior: REDUCED ? 'auto' : 'smooth' });
  els.forEach(el => el.classList.add('tour-pulse'));
  await wait(ms);
  els.forEach(el => el.classList.remove('tour-pulse'));
}

/* ── Live caption values ───────────────────────────────────────────────────── */

const M = () => api.state.metrics || {};
const num = (v, fmt, fallback) =>
  (typeof v === 'number' && Number.isFinite(v)) ? fmt(v) : fallback;

/* ── The beats ─────────────────────────────────────────────────────────────── */

const ALL_BEATS = [
  {
    title: 'A brain, reconstructed',
    body: () => 'Sparse MRI slices in, dense volume out. What you are orbiting is a ' +
                'marching-cubes shell over a GAN-reconstructed volume — not a scan.',
    dwell: 2400,
    async run() {
      await api.switchVariant(0);                // idempotent when already active
      setAutoRotate(false);
      setAxis('axial');
      setSlider(api.els.clipSlider, 100);
      setTumor(true);                            // pin the defaults: a second run
      setShell(0.35);                            // must not inherit the first's state
      api.fitCamera(api.state.brainBox);
      setAutoRotate(true);                       // spin only once the camera has settled
    },
  },
  {
    title: 'Cut into it',
    body: () => 'The plane at the cut is the reconstructed MRI at that depth, ' +
                'resampled live from the volume — not a stored screenshot.',
    dwell: 1600,
    async run() {
      setAutoRotate(false);                      // autoRotate would fight the tween
      await camTo([0.85, 0.28, 0.95], 1.75, 900);
      await clipTo(42, 2200);
    },
  },
  {
    title: 'Dense in all three planes',
    body: () => 'Only one axis was ever acquired. Axial, coronal and sagittal all ' +
                'resolve because the GAN filled in what was never scanned.',
    dwell: 600,
    async run() {
      setAutoRotate(false);
      for (const [axis, dir] of [
        ['axial',    [1.0, 0.30, 0.60]],         // clip normal -X → cut face looks +X
        ['coronal',  [0.30, 1.0, 0.55]],         // clip normal -Y
        ['sagittal', [0.55, 0.30, 1.0]],         // clip normal -Z
      ]) {
        setAxis(axis);
        // The axis click resets the clip, but it only fires when the axis
        // actually changes — 'axial' is already active on the first pass. Reset
        // explicitly so each sub-step starts from a full volume either way.
        setSlider(api.els.clipSlider, 100);
        await camTo(dir, 1.8, 900);
        if (gen !== this._gen) return;
        await clipTo(50, 1200);
        if (!(await wait(500))) return;
      }
    },
  },
  {
    title: 'The tumour, measured',
    need: () => !!api.state.metrics,
    body: () => {
      const t = M().tumor || {};
      return `${num(t.volume_ml, v => v.toFixed(1) + ' mL', '—')} · ` +
             `${num(t.voxels, v => v.toLocaleString(), '—')} voxels · ` +
             `${num(t.equivalent_diameter_mm, v => v.toFixed(1) + ' mm', '—')} equivalent ` +
             'diameter. Segmented by nnU-Net 2D with EMA weights, 16-augmentation TTA ' +
             'and morphological post-processing.';
    },
    dwell: 700,
    async run() {
      setAutoRotate(false);
      await camTo([0.9, 0.3, 0.95], 1.8, 900);
      await pulse(rowsFor(['m-volume-ml', 'm-voxels', 'm-diameter']), 2200);
    },
  },
  {
    title: 'The shape itself',
    need: () => !!api.state.hasTumor,
    body: () => 'Not a bright patch on a slice — the segmentation mask meshed and ' +
                'placed back in the volume. The brain fades so you can see it whole.',
    dwell: 900,
    async run() {
      setAutoRotate(false);
      setTumor(true);
      await Promise.all([shellTo(0.10, 1400), camTo([0.8, 0.35, 1.0], 1.25, 1400)]);
      setAutoRotate(true);
    },
  },
  {
    title: 'What the classifier saw',
    need: () => !!(api.state.metrics && api.state.metrics.classification),
    body: () => {
      const c = M().classification || {};
      const pct = num(c.confidence, v => (v * 100).toFixed(1) + '%', '—');
      return `EfficientNet-B3 over every slice, aggregated on tumour evidence: ` +
             `<b>${c.label || '—'}</b> at ${pct}. Grad-CAM in the notebook shows which ` +
             'pixels drove it. Out-of-distribution here — a demonstration, not a diagnosis.';
    },
    dwell: 700,
    async run() {
      await pulse([document.getElementById('m-cls-group')].filter(Boolean), 2600);
    },
  },
  {
    title: 'Six months later',
    need: () => api.state.variants.length > 1,
    body: () => 'A Fisher–KPP PINN evolves the tumour density forward, nnU-Net ' +
                're-segments the result, and the GAN rebuilds the whole volume from it. ' +
                'Same camera — watch the shell grow.',
    dwell: 2000,
    async run() {
      setAutoRotate(false);
      await shellTo(0.30, 700);
      await api.switchVariant(1);                // keepView: true — grows in place
    },
  },
  {
    title: 'Growth, quantified',
    // Needs metrics as well as a second volume — growth is null on `initial` by
    // contract and only populates after beat 6 switches, so the test is "are
    // there metrics at all", not "is growth set right now".
    need: () => api.state.variants.length > 1 && !!api.state.metrics,
    body: () => {
      const m = M(), g = m.growth || {}, r = m.tumor_resegmented || {};
      const ratio = num(g.ratio, v => '×' + v.toFixed(2), '—');
      const delta = num(g.delta_volume_ml, v => (v >= 0 ? '+' : '') + v.toFixed(1) + ' mL', '—');
      const days  = num(m.horizon_days, v => v.toFixed(0) + ' days', 'the horizon');
      const check = num(r.volume_ml, v => ` Re-segmenting independently gives ${v.toFixed(1)} mL.`, '');
      const key = api.state.hasTumor
        ? ' <b>Cyan</b> is the tumour today; <b>red</b> is what the PINN predicts will be new.'
        : '';
      return `${ratio}, ${delta} over ${days}.${check}${key}`;
    },
    dwell: 700,
    async run() {
      await pulse(rowsFor(['m-growth-ratio', 'm-growth-delta', 'm-horizon']), 2600);
    },
  },
  {
    title: 'Your turn',
    body: () => 'Drag to rotate, use the clip slider to cut, and switch <b>Volume</b> ' +
                'to compare the initial scan against the predicted one.',
    dwell: 2600,
    async run() {
      setAutoRotate(false);
      api.closeDrawers();
    },
  },
];

/* ── Caption card ──────────────────────────────────────────────────────────── */

function buildCard() {
  const el = document.createElement('div');
  el.id = 'tour-card';
  el.innerHTML =
    '<div class="tour-head">' +
      '<span class="tour-step"></span>' +
      '<button class="tour-btn tour-skip" type="button">Skip</button>' +
    '</div>' +
    '<div class="tour-title"></div>' +
    '<div class="tour-body"></div>' +
    '<div class="tour-nav">' +
      '<button class="tour-btn" type="button" data-step="-1" aria-label="Previous">‹</button>' +
      '<button class="tour-btn" type="button" data-step="1" aria-label="Next">›</button>' +
    '</div>';

  el.querySelector('.tour-skip').addEventListener('click', () => stop());
  el.querySelectorAll('[data-step]').forEach(b =>
    b.addEventListener('click', () => jump(idx + Number(b.dataset.step))));
  document.body.appendChild(el);
  return el;
}

function showCaption(beat) {
  card.querySelector('.tour-step').textContent  = `${idx + 1} / ${beats.length}`;
  card.querySelector('.tour-title').textContent = beat.title;
  card.querySelector('.tour-body').innerHTML    = beat.body();   // authored here, not user input
  card.querySelector('[data-step="-1"]').disabled = idx === 0;
}

/* ── Run loop ──────────────────────────────────────────────────────────────── */

function jump(to) {
  if (to < 0 || to >= beats.length) return;
  start(to);
}

async function start(from = 0) {
  const myGen = ++gen;
  beats = ALL_BEATS.filter(b => !b.need || b.need());
  if (!beats.length) return;

  running = true;
  document.body.classList.add('tour-active');
  card.classList.add('visible');

  for (idx = Math.min(from, beats.length - 1); idx < beats.length; idx++) {
    if (myGen !== gen) return;
    const beat = beats[idx];
    beat._gen = myGen;
    showCaption(beat);
    try {
      await beat.run();
    } catch (e) {
      console.warn('[brain-viewer] tour beat failed', beat.title, e);
    }
    if (myGen !== gen) return;
    if (!(await wait(beat.dwell ?? 1400))) return;
  }
  stop();
}

function stop() {
  gen++;                                    // orphans every in-flight tween/wait
  running = false;
  setAutoRotate(false);                     // never hand back a spinning camera
  document.body.classList.remove('tour-active');
  if (card) card.classList.remove('visible');
  document.querySelectorAll('.tour-pulse').forEach(el => el.classList.remove('tour-pulse'));
}

/* ── Install ───────────────────────────────────────────────────────────────── */

export function installTour(externalApi, opts = {}) {
  api  = externalApi;
  card = buildCard();

  // A real gesture always wins. Synthetic events dispatched by this file carry
  // isTrusted === false, so the tour cannot cancel itself.
  const bail = e => { if (e.isTrusted && running) stop(); };
  const canvas = document.getElementById('canvas-container');
  if (canvas) {
    canvas.addEventListener('pointerdown', bail);
    canvas.addEventListener('wheel', bail, { passive: true });
  }
  const panel = document.getElementById('panel');
  if (panel) panel.addEventListener('pointerdown', bail);
  window.addEventListener('keydown', e => { if (e.key === 'Escape') bail(e); });

  // The start button is created here, not in index.html, so a missing tour.js
  // never leaves a dead control behind.
  const btnRow = document.querySelector('#panel .btn-row');
  if (btnRow) {
    const b = document.createElement('button');
    b.className = 'btn btn-tour';
    b.id = 'tour-btn';
    b.type = 'button';
    b.textContent = '▶ Guided tour';
    b.addEventListener('click', () => (running ? stop() : start(0)));
    btnRow.parentNode.insertBefore(b, btnRow);
  }

  if (opts.autostart) setTimeout(() => start(0), 600);
}
