# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ORACLE is a **research prototype, marked complete** (see the Project Status section of [README.md](README.md)). It is a notebook-driven brain-tumor pipeline: segment → reconstruct a 3D volume → predict tumor growth → re-segment/re-reconstruct the predicted future → render both in a browser.

There is **no Python package, no test suite, no linter, and no build step** for the ML side. All model code lives inside Jupyter notebooks at the repo root. The only conventional Python is the standalone visualization pipeline in [brain-gan-viewer/](brain-gan-viewer/).

## Repository layout

Root notebooks, in pipeline order:

| Notebook | Role |
| -------- | ---- |
| [dataset_analysis.ipynb](dataset_analysis.ipynb) | Exploratory stats on the MU-Glioma-Post cohort (class imbalance, tumor size distribution) |
| [nnUnet.ipynb](nnUnet.ipynb) | Stage 1 — trains the custom nnU-Net 2D segmenter. Produces `best_nnunet2d.pth` (EMA weights) and `nnunet2d_final.pth` |
| [3d-recon-gen.ipynb](3d-recon-gen.ipynb) | Stage 2a — generator-only pretraining of the Fast2p5D sparse→dense reconstructor |
| [3d-recon-disc.ipynb](3d-recon-disc.ipynb) | Stage 2b — GAN adversarial fine-tuning; also drives the brain-gan-viewer pipeline and serves the viewer locally |
| [pinn_tumor_growth.ipynb](pinn_tumor_growth.ipynb) | Stage 3 — inverse Fisher–KPP PINN on BraTS 2024; produces `pinn_tumor_growth.pt` |
| [full_pipeline_testing.ipynb](full_pipeline_testing.ipynb) | Chains all three checkpoints end-to-end on one patient timepoint, including the Stage-4 closed evolution loop |

Untracked-but-present directories (all in [.gitignore](.gitignore)): `models/` (the five checkpoints), `assets/`, `old_approaches/`, `viewer_slices/` (`initial/` + `evolved/` PNG stacks exported by the full pipeline), `PKG-MU-Glioma-partial/` (local dataset copy), `roadmap.md`.

Note the README references a notebook name (`unet_plusplus_brain_tumor_segmentation.ipynb`) that no longer exists — the segmenter was rewritten and the file is now `nnUnet.ipynb`. `roadmap.md` documents the *superseded* UNet++/EfficientNet-B4 approach; do not treat it as current.

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

The viewer itself ([brain-gan-viewer/viewer/](brain-gan-viewer/viewer/)) is plain Three.js loaded from a CDN importmap — **no npm, no bundler**. Edit `app.js`/`index.html` directly, then re-run `prepare_github_pages.py` to sync into `docs/brain-viewer/`.

## Deployment

[.github/workflows/static.yml](.github/workflows/static.yml) publishes `./docs` to GitHub Pages on every push to `main`. Anything committed under `docs/brain-viewer/` goes live at `https://enricotazzer.github.io/ORACLE/brain-viewer/`. Note `brain-gan-viewer/viewer/` and `brain-gan-viewer/docs/` are gitignored while the top-level `docs/` is not — `docs/brain-viewer/` is the deploy artifact and must be committed.

## Architecture notes worth knowing before editing

**The three stages are coupled by tensor conventions, not by an API.** Each notebook redefines the model classes it needs, so changing an architecture means changing it in *both* its training notebook and in `full_pipeline_testing.ipynb`, which re-declares `nnUNet2D`, `Fast2p5D`, and the PINN classes in order to load the checkpoints. Checkpoints are plain `state_dict`s — a silently renamed layer breaks loading at Stage N with a `strict=True` error.

- **Segmentation input**: 4-channel `(t1n, t1c, t2w, t2f)`, binary mask out. Deep supervision means the model returns a *list* of logits (full-res head first) rather than a single tensor — `full_pipeline_testing.ipynb` wraps it in `_DSStripWrapper` to take head 0 at inference. Trained at `IMG_SIZE = 192` but run at `CFG['img_size'] = 256` in the full pipeline, which shares one resolution across all stages; the network is fully convolutional so this works, but keep any new layer resolution-agnostic.
- **Reconstruction input**: 5-channel `(t1n, t1c, t2w, t2f, mask_density)` over a 5-slice window `[z..z+4]` predicting `[z+5]`. The 5th channel is the segmentation mask — Stage 2 depends on Stage 1's output, which is why the pipeline order is fixed.
- **PINN**: outputs `u ∈ [0,1]` by construction (sigmoid head) and estimates `D_wm`, `D_gm`, `ρ` **in log space**. Reload via the `load_pinn()` helper stored alongside the checkpoint, not by hand-constructing the module.
- **Stage 4** re-uses Stage 1 and Stage 2 on synthetically evolved slices. All growth knobs are in a single `GROWTH_CFG` dict (`rho_scale`, `D_scale`, `horizon_days`); appearance compositing (necrotic core / enhancing rim / edema) is in `APPEAR`. Defaults are tuned for *visible* change, not physical fidelity — set the scales to `1.0` for the faithful regime.

**Deployment-matched evaluation is deliberate.** Segmentation model selection uses EMA weights + TTA + morphological post-processing, not raw single-pass Dice; validation therefore runs only every `VALIDATE_EVERY` epochs because it's ~8× slower. Don't "optimize" this by validating on raw forward passes — the reported ≈0.84 Dice is the post-processed number and the two are not comparable.

## Conventions

- `SEED = 42` everywhere; keep it when adding cells.
- Config lives in ALL-CAPS module-level constants or a single `CFG`/`INFER_CFG`/`GROWTH_CFG` dict near the top of the relevant section — add new knobs there rather than inline.
- Notebooks are committed **with outputs stripped** (except `dataset_analysis.ipynb` and `3d-recon-disc.ipynb`, which retain some). Figures shown in the README live as static files in [readme_assets/](readme_assets/) — regenerate and re-commit those rather than relying on embedded cell output.
- Section headings are numbered markdown cells (`## 7. Model Initialization`); the numbering has gaps from removed sections — leave them alone rather than renumbering, since prose references section numbers.
