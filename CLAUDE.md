# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ORACLE is a **research prototype, marked complete**, with a classification + viewer-metrics extension layered on top (see the Project Status section of [README.md](README.md)). It is a notebook-driven brain-tumor pipeline: classify → segment → reconstruct a 3D volume → predict tumor growth → re-segment/re-reconstruct the predicted future → render both in a browser.

The three original modules are untouched by the extension. Classification is a **parallel diagnostic branch**: it feeds `metrics.json` and does **not** gate Stage 1 (the segmenter is class-agnostic), and Stages 1–4 run unchanged when its checkpoint is missing.

There is **no Python package, no test suite, no linter, and no build step** for the ML side. All model code lives inside Jupyter notebooks at the repo root. The only conventional Python is the standalone visualization pipeline in [brain-gan-viewer/](brain-gan-viewer/).

## Repository layout

Root notebooks, in pipeline order:

| Notebook | Role |
| -------- | ---- |
| [dataset_analysis.ipynb](dataset_analysis.ipynb) | Exploratory stats on the MU-Glioma-Post cohort (class imbalance, tumor size distribution) |
| [effnet_tumor_classification.ipynb](effnet_tumor_classification.ipynb) | Stage 0b — EfficientNet-B3 4-class classifier + `ClsGradCAM`. Produces `best_effnetb3_cls.pth` (EMA) and `effnetb3_cls_final.pt` |
| [nnUnet.ipynb](nnUnet.ipynb) | Stage 1 — trains the custom nnU-Net 2D segmenter. Produces `best_nnunet2d.pth` (EMA weights) and `nnunet2d_final.pth` |
| [3d-recon-gen.ipynb](3d-recon-gen.ipynb) | Stage 2a — generator-only pretraining of the Fast2p5D sparse→dense reconstructor |
| [3d-recon-disc.ipynb](3d-recon-disc.ipynb) | Stage 2b — GAN adversarial fine-tuning; also drives the brain-gan-viewer pipeline and serves the viewer locally |
| [pinn_tumor_growth.ipynb](pinn_tumor_growth.ipynb) | Stage 3 — inverse Fisher–KPP PINN on BraTS 2024; produces `pinn_tumor_growth.pt` |
| [full_pipeline_testing.ipynb](full_pipeline_testing.ipynb) | Chains all four checkpoints end-to-end on one patient timepoint (Stage 0b classification + Stages 1-4, including the closed evolution loop) |

Untracked-but-present directories (all in [.gitignore](.gitignore)): `models/` (six checkpoints once the classifier is published), `assets/`, `old_approaches/`, `viewer_slices/` (`initial/` + `evolved/` PNG stacks exported by the full pipeline), `PKG-MU-Glioma-partial/` (local dataset copy), `roadmap.md`.

`roadmap.md` documents the *superseded* UNet++/EfficientNet-B4 approach; do not treat it as current.

## Running things

### Notebooks

Open in Jupyter and run top-to-bottom. There is no `requirements.txt` for the notebooks; dependencies are installed by in-notebook `pip install` cells or assumed present in the Kaggle image (torch, monai, albumentations, ttach, nibabel, scikit-image, trimesh, plotly).

**Device handling is not uniform** — check before editing:

- `nnUnet.ipynb`, `3d-recon-gen.ipynb`, `3d-recon-disc.ipynb`, `full_pipeline_testing.ipynb` fall back `cuda → mps → cpu`, so they run on this Mac.
- `pinn_tumor_growth.ipynb` is `cuda → cpu` only (double-backward autograd through the PDE residual is unreliable on MPS). Do not "fix" this by adding MPS.
- AMP `GradScaler` is explicitly gated on `torch.cuda.is_available()`; batch sizes shrink on non-CUDA (e.g. `BATCH_SIZE = 12 if cuda else 4`).

**Dataset paths are hardcoded per notebook and differ by origin:**

- Kaggle-authored: `nnUnet.ipynb` and `full_pipeline_testing.ipynb` use `/kaggle/input/datasets/prishapgpg/pkg-mu/PKG - MU-Glioma-Post/MU-Glioma-Post`; `full_pipeline_testing.ipynb` also reads checkpoints from `/kaggle/input/models/aneurisma/oracle-pipeline/...` and has a `_DATA_CANDIDATES` list for local fallback.
- Locally authored: the two `3d-recon-*` notebooks use `/Volumes/T7/ORACLE project/PKG-MU-Glioma-partial` via `CFG['data_root']`.
- `effnet_tumor_classification.ipynb` reads `masoudnickparvar/brain-tumor-mri-dataset` at `/kaggle/input/brain-tumor-mri-dataset` (`Training/` + `Testing/`, four class folders each), with an `ORACLE_CLS_DATA_ROOT` override — same `_first_existing_dir` idiom as `full_pipeline_testing.ipynb`.
- `pinn_tumor_growth.ipynb` reads BraTS 2024 from `/kaggle/input/brats2024-small-dataset` and **silently falls back to a synthetic finite-difference Fisher–KPP problem** if no cases are found (`USE_REAL_DATA`). If results look suspiciously clean, check that flag first.

When adapting a notebook to run locally, change the path constant — don't restructure the loaders.

### 3D viewer pipeline

The only part with real dependencies and a CLI:

```bash
cd brain-gan-viewer
pip install -r requirements.txt

# full run: generate_brain.py + prepare_github_pages.py
python run_pipeline.py \
  --input_dir ./data/gan_slices \
  --axis axial \
  --github_pages_subpath /ORACLE/brain-viewer/

# preview
cd ../docs/brain-viewer && python -m http.server 8080
```

`--variant initial|evolved` builds the two volumes the viewer's `Volume` toggle switches between; both must be generated for the toggle to work. See [brain-gan-viewer/README.md](brain-gan-viewer/README.md) for the full flag reference.

The viewer itself is plain Three.js loaded from a CDN importmap — **no npm, no bundler**. `brain-gan-viewer/viewer/` is gitignored and `docs/brain-viewer/` is the tracked deploy artifact; the two are kept byte-identical. Edit the tracked copy and mirror it by hand — see the warning under [Deployment](#deployment) before reaching for `prepare_github_pages.py`.

## Deployment

> **⚠️ Do not run `prepare_github_pages.py` for code-only viewer edits.** Its `copy_viewer()` does `shutil.rmtree(docs/brain-viewer)` and then `copytree` from the **gitignored** `brain-gan-viewer/viewer/` — it deletes ~1261 tracked files (173 MB) and repopulates from an untracked source. It is safe only while the two trees are identical. For editing `index.html`/`app.js`/`style.css`, change the tracked copy under `docs/brain-viewer/`, then `cp` the three files into `brain-gan-viewer/viewer/`, and confirm `git status --short` shows exactly three paths.

[.github/workflows/static.yml](.github/workflows/static.yml) publishes `./docs` to GitHub Pages on every push to `main`. Anything committed under `docs/brain-viewer/` goes live at `https://enricotazzer.github.io/ORACLE/brain-viewer/`. Note `brain-gan-viewer/viewer/` and `brain-gan-viewer/docs/` are gitignored while the top-level `docs/` is not — `docs/brain-viewer/` is the deploy artifact and must be committed.

## Architecture notes worth knowing before editing

**The three stages are coupled by tensor conventions, not by an API.** Each notebook redefines the model classes it needs, so changing an architecture means changing it in *both* its training notebook and in `full_pipeline_testing.ipynb`, which re-declares `nnUNet2D`, `Fast2p5D`, and the PINN classes in order to load the checkpoints. Checkpoints are plain `state_dict`s — a silently renamed layer breaks loading at Stage N with a `strict=True` error.

- **Classification**: `BrainTumorClassifier` is a **fourth** class re-declared in `full_pipeline_testing.ipynb`. Its `state_dict` keys are namespaced under `backbone.` and mirror torchvision's `efficientnet_b3` naming — switching to `timm`, or doing 1-channel stem surgery instead of replicating grayscale to 3 channels, breaks `strict=True` loading at Stage 0b. Input is 300×300 (B3's native resolution), deliberately *not* the segmenter's 256. Grad-CAM hooks `backbone.features.8` (last 1536-ch `Conv2dNormActivation`, 10×10 at 300²). Reload via `load_classifier()`, which returns the same `(net, cfg, ckpt)` 3-tuple shape as `load_pinn()`.
- **Segmentation input**: 4-channel `(t1n, t1c, t2w, t2f)`, binary mask out. Deep supervision means the model returns a *list* of logits (full-res head first) rather than a single tensor — `full_pipeline_testing.ipynb` wraps it in `_DSStripWrapper` to take head 0 at inference. Trained at `IMG_SIZE = 192` but run at `CFG['img_size'] = 256` in the full pipeline, which shares one resolution across all stages; the network is fully convolutional so this works, but keep any new layer resolution-agnostic.
- **Reconstruction input**: 5-channel `(t1n, t1c, t2w, t2f, mask_density)` over a 5-slice window `[z..z+4]` predicting `[z+5]`. The 5th channel is the segmentation mask — Stage 2 depends on Stage 1's output, which is why the pipeline order is fixed.
- **PINN**: outputs `u ∈ [0,1]` by construction (sigmoid head) and estimates `D_wm`, `D_gm`, `ρ` **in log space**. Reload via the `load_pinn()` helper stored alongside the checkpoint, not by hand-constructing the module.
- **Stage 4** re-uses Stage 1 and Stage 2 on synthetically evolved slices. All growth knobs are in a single `GROWTH_CFG` dict (`rho_scale`, `D_scale`, `horizon_days`); appearance compositing (necrotic core / enhancing rim / edema) is in `APPEAR`. Defaults are tuned for *visible* change, not physical fidelity — set the scales to `1.0` for the faithful regime.

**Deployment-matched evaluation is deliberate.** Segmentation model selection uses EMA weights + TTA + morphological post-processing, not raw single-pass Dice; validation therefore runs only every `VALIDATE_EVERY` epochs because it's ~8× slower. Don't "optimize" this by validating on raw forward passes — the reported ≈0.84 Dice is the post-processed number and the two are not comparable.

## Local dry-running

There is **no MRI data on this machine** (`PKG-MU-Glioma-partial/` is gone), so nothing trains locally. But the conda env **`oracle`** (`~/miniconda3/envs/oracle/bin/python`) has torch 2.10, torchvision 0.25, cv2 4.13, sklearn 1.8, **albumentations 2.0.8**, skimage, trimesh, nibabel, timm, ttach — enough to construct models, fire Grad-CAM hooks, build transforms, and `import generate_brain`. Use it for every local check; do not `pip install` into `base`.

**albumentations 2.x trap:** `nnUnet.ipynb`'s augmentation idioms must not be copied blindly. In 2.0.8 the names `ShiftScaleRotate`, `CoarseDropout` and `GaussNoise` all still exist, but `A.GaussNoise(var_limit=…)` does **not** raise — it emits `UserWarning: Argument(s) 'var_limit' are not valid` and **silently drops the kwarg**, turning the augmentation into a no-op. try/except cannot catch that. Build transforms under `warnings.simplefilter("error", UserWarning)` to make it fail loudly.

## EfficientNet-B3 weights

Kaggle notebooks default to **Internet OFF**, and without it torchvision cannot fetch ImageNet weights — the model then trains from random init and still reports plausible-looking accuracy. `load_imagenet_b3_weights()` therefore falls back torchvision → `/kaggle/input/**/efficientnet_b3*.pth` glob → random init with a `!!!` banner, and records the result as `CLS_CFG['pretrained']` **inside both checkpoints**. Check that flag before trusting any reported metric; §13 refuses to declare success when it is `False`. `torch.hub.set_dir('/kaggle/working/torch_hub')` puts the download in the notebook *output* so it can be republished as a dataset for offline reruns.

## Viewer metrics contract

`assets/<variant>/metrics.json`, schema **`oracle.metrics/1`** — written by `full_pipeline_testing.ipynb`, copied into place by `generate_brain.py --metrics_json`, read by `app.js::loadMetrics()`. [brain-gan-viewer/metrics.example.json](brain-gan-viewer/metrics.example.json) is the reference document and the local test fixture.

- The file is **optional**: 404 or invalid JSON means "hide the panel", never an error, never a blocked mesh. `verify_assets()` reports it but must never fold it into `ok`, or every pre-existing deployment starts exiting 1.
- The reader **refuses** a major version it does not know rather than guessing — wrong numbers on a medical page are worse than no numbers. Additive fields keep major `1`; any rename/unit/nullability change bumps to `oracle.metrics/2`.
- `growth` is null on `initial`; `classification` is null when no classifier ran; `tumor_resegmented` is null unless a re-segmentation exists. Missing values are `null`, never `"N/A"`.
- `voxel.spacing_mm` is the **native NIfTI grid** (`header.get_zooms()[:3]`). `volume_meta.json`'s `pixel_spacing_mm` describes the viewer's 256² resampled *display* grid. They are different quantities — **never reconcile them.**
- Writers must wrap values in `float()`/`int()` (numpy scalars aren't JSON-serializable) and guarantee finiteness — `json.dumps` emits bare `NaN`/`Infinity`, which is invalid JSON and makes `JSON.parse` throw in the browser.

## Conventions

- `SEED = 42` everywhere; keep it when adding cells.
- Config lives in ALL-CAPS module-level constants or a single `CFG`/`INFER_CFG`/`GROWTH_CFG` dict near the top of the relevant section — add new knobs there rather than inline.
- **Edit notebooks programmatically**, never by hand: load with `json.load`, splice cell dicts, and write back as `json.dumps(nb, indent=1, ensure_ascii=False) + "\n"` — that round-trips byte-exactly, so diffs contain only the cells you changed. New cell `id` = `uuid4().hex[:8]`; code cells carry `execution_count: null` and `outputs: []`. Gate with an AST check (blank `!`/`%` lines, then `ast.parse` every code cell) — all notebooks pass it today.
- Notebooks are committed **with outputs stripped** (except `dataset_analysis.ipynb` and `3d-recon-disc.ipynb`, which retain some). Figures shown in the README live as static files in [readme_assets/](readme_assets/) — regenerate and re-commit those rather than relying on embedded cell output.
- Section headings are numbered markdown cells (`## 7. Model Initialization`); the numbering has gaps from removed sections — leave them alone rather than renumbering, since prose references section numbers.
