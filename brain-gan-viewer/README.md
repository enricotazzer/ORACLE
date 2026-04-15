# Brain GAN Viewer

A complete pipeline that takes GAN-generated MRI slices and produces a production-ready,
interactive 3D brain visualization deployable on GitHub Pages.

---

## Overview

```
GAN MRI slices (PNG/JPG)
        │
        ▼
  generate_brain.py
  ├── normalise + Gaussian denoise
  ├── Otsu threshold + morphological cleaning
  ├── largest-component brain mask
  ├── Taubin-smoothed marching cubes → brain_surface.glb
  └── export MRI slice JPGs + volume_meta.json
        │
        ▼
  viewer/  (Three.js static site)
  ├── realistic cortex shell
  ├── clipping plane reveals actual MRI content
  └── orbit / zoom / pan / screenshot
        │
        ▼
  docs/brain-viewer/  (GitHub Pages build)
```

---

## Input format

Place your GAN-generated brain MRI slices here:

```
data/gan_slices/
  slice_000.png
  slice_001.png
  ...
  slice_127.png
```

Requirements:
- PNG or JPG (or TIFF)
- Grayscale or RGB (auto-converted to grayscale)
- All same width × height
- Natural-sorted filenames
- Black/near-black background, brain in foreground
- No skull expected — pipeline isolates brain tissue only

---

## Installation

```bash
cd brain-gan-viewer
pip install -r requirements.txt
```

Python ≥ 3.9 recommended.

---

## Run the full pipeline

```bash
python run_pipeline.py \
  --input_dir  ./data/gan_slices \
  --axis       axial \
  --pixel_spacing    1.0 \
  --slice_thickness  1.0 \
  --project_repo_name      my-main-project \
  --github_pages_subpath   /my-main-project/brain-viewer/
```

This:
1. Generates the brain mesh + slice exports → `viewer/assets/`
2. Validates all output assets
3. Copies the final static site into `docs/brain-viewer/`

---

## Run locally

```bash
cd docs/brain-viewer
python -m http.server 8080
# open http://localhost:8080
```

> **Important:** The viewer must be served over HTTP (not `file://`) because
> the browser blocks cross-origin fetch of local files.

---

## How realism is achieved

### Mesh smoothing — Taubin method

Naive Laplacian smoothing shrinks the mesh volume over iterations.
Taubin (1995) alternates a positive step (λ) and a negative step (μ = −0.53)
so high-frequency noise is removed while volume is approximately preserved.
This produces a smooth cortex-like surface without artefact shrinkage.

### Pre-smoothing the scalar field

Before marching cubes, the binary brain mask is Gaussian-blurred.
This ensures the iso-surface is smooth even at step_size=1 and avoids
the voxel-staircase artefact typical of raw boolean masks.

### No skull artefacts

The pipeline uses Otsu thresholding on the GAN volume rather than
atlas-based skull-stripping, because GAN-generated slices already represent
brain tissue without a bony calvarium. The threshold naturally separates
brain from dark background.

---

## How the clip plane works

1. The `clip-slider` drives a single parameter `t ∈ [0, 1]`.
2. `t` maps linearly into the bounding box of the brain along the chosen axis.
3. A `THREE.Plane` clips the shell at that world position — the renderer
   discards everything on the "cut" side.
4. The same `t` maps to a slice index in the exported `assets/slices/` folder.
5. A quad (`PlaneGeometry`) is placed exactly at the clip plane and textured
   with the corresponding MRI slice image.

This means the cut surface and the displayed slice are always synchronised
from the same parameter — no shader tricks, no fragile UV projection.

---

## Preprocessing parameters

| Flag | Default | Purpose |
|------|---------|---------|
| `--smooth_sigma` | 0.8 | Gaussian σ applied to volume after normalisation (GAN artefact reduction) |
| `--field_sigma` | 1.5 | Gaussian σ applied to mask field before marching cubes |
| `--closing_radius` | 4 | Ball radius for binary closing (fills gaps in mask) |
| `--dilation_radius` | 1 | Final mask dilation (ensures continuous cortex) |
| `--taubin_iter` | 25 | Number of Taubin smoothing iterations |
| `--decimate_fraction` | 0.85 | Target face fraction after decimation (1.0 = no decimation) |
| `--max_slices` | 128 | Number of MRI slice JPGs exported for the viewer |

---

## Deploy to GitHub Pages

### Option A — `docs/` folder on `main` branch (recommended)

1. Copy `docs/brain-viewer/` into your main repo:
   ```bash
   cp -r docs/brain-viewer/ /path/to/my-main-repo/docs/brain-viewer/
   ```
2. Commit and push.
3. In your repo → **Settings → Pages → Source**: `main` branch, `/docs` folder.
4. Viewer URL: `https://<USERNAME>.github.io/<REPO_NAME>/brain-viewer/`

### Option B — `gh-pages` branch

1. Push only the contents of `docs/brain-viewer/` to a `gh-pages` branch.
2. In **Settings → Pages → Source**: `gh-pages` branch, `/` root.
3. Viewer URL: `https://<USERNAME>.github.io/<REPO_NAME>/`

### Base-path rewriting

If served under a sub-path, pass `--github_pages_subpath` to `run_pipeline.py`.
The script injects a `<base href="…">` tag into `index.html` so all relative
asset URLs resolve correctly.

---

## Integrate into your main repo homepage

In your main `README.md` or `index.html`:

```markdown
[View 3D Brain Model](https://<USERNAME>.github.io/<REPO_NAME>/brain-viewer/)
```

Or embed as an iframe:
```html
<iframe
  src="https://<USERNAME>.github.io/<REPO_NAME>/brain-viewer/"
  width="100%" height="600"
  frameborder="0">
</iframe>
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Blank canvas, no mesh | Run `generate_brain.py` first; check that `viewer/assets/brain_surface.glb` exists |
| "Cannot load volume_meta.json" | Must be served via HTTP, not `file://` |
| Slice plane shows blank / black | `assets/slices/slice_000.jpg` missing — re-run generate_brain.py |
| Mesh looks hollow / empty | Increase `--closing_radius` or `--dilation_radius` |
| Mesh too jagged | Increase `--field_sigma` and/or `--taubin_iter` |
| Too few slices visible | Increase `--max_slices` (up to the number of input slices) |
| GAN checkerboard artefacts in slices | Increase `--smooth_sigma` (try 1.2–2.0) |
| GitHub Pages 404 | Confirm the `docs/` folder is committed; check the Pages source in repo Settings |

---

## Notes on GAN-generated MRI

- GAN outputs may contain checkerboard or high-frequency noise. The `--smooth_sigma`
  parameter applies a mild Gaussian blur before thresholding to reduce these artefacts.
- GAN slices typically represent a single anatomy (brain without skull), so classical
  skull-stripping (FreeSurfer, FSL BET) is unnecessary and not used here.
- If your GAN was trained on axial slices, use `--axis axial`. If coronal, use
  `--axis coronal`, etc. The pipeline stacks slices depth-first along the chosen axis.
- Slice ordering is determined by natural sort on filenames — ensure filenames encode
  anatomical order (slice_000 → slice_127 = inferior → superior for axial).
