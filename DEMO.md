# Demoing ORACLE

A script for presenting this live — interview, defence, or a five-minute stand-up.
The guided tour does the driving; this tells you what to say over it and what to
answer when someone pushes back.

**Live site:** <https://enricotazzer.github.io/ORACLE/>
**Straight to the tour:** <https://enricotazzer.github.io/ORACLE/brain-viewer/?tour=1>

---

## Before you start

- [ ] Open the viewer once beforehand. It streams ~54 MB per volume; the second load is cached, the first is not.
- [ ] If the connection is unknown, serve locally instead: `cd docs/brain-viewer && python3 -m http.server 8080`, then `localhost:8080/?tour=1`.
- [ ] Full screen, bookmarks bar hidden. The viewer is a fixed-layout WebGL page — it looks best with nothing else on screen.
- [ ] Know that **any drag, scroll, or `Esc` ends the tour instantly** and hands you the controls. That is a feature: use it to take over mid-beat.

---

## The five-minute version

The tour has nine beats and runs about 40 seconds on its own. You are going to
interrupt it constantly. That's the point — it guarantees you never lose your place.

### Opening line, before you click anything

> "This takes a sparse MRI acquisition — a handful of slices — and returns a dense
> 3D volume, a segmented tumour, and a prediction of what that tumour looks like six
> months from now. Then it runs the whole thing again on the prediction. Everything
> you'll see is model output, not a rendering of stored scans."

### Beat 1 — "A brain, reconstructed"

Let it spin. Say:

> "This shell is marching cubes over a reconstructed volume. The network only ever
> saw a fraction of these slices."

### Beat 2 — "Cut into it"

The clip plane opens to 42%. **Pause here** — drag the model yourself.

> "The image on the cut face is the reconstruction at that depth, resampled as I move.
> It isn't a stored screenshot."

### Beat 3 — "Dense in all three planes"

Axial, coronal, sagittal in turn. This is the strongest evidence in the demo — spend
time on it.

> "Only one axis was ever acquired. The other two resolve because the GAN filled in
> what was never scanned. If the reconstruction were bad, the off-axis planes would
> be the first thing to fall apart."

### Beat 4 — "The tumour, measured"

The volumetry rows pulse.

> "96.9 millilitres, 57 millimetre equivalent diameter. Segmented by a custom 2D
> nnU-Net — EMA weights, sixteen-augmentation test-time augmentation, and morphological
> post-processing. Dice is about 0.84, and that's the post-processed number, measured
> the same way it runs here."

### Beat 5 — "The shape itself"

The brain fades to 10% and the tumour is left hanging in space. Let it rotate.

> "That's not a bright patch on a slice — that's the segmentation mask meshed with
> marching cubes and put back in the volume at the same coordinates. The 96.9 millilitres
> in the panel is the voxel count behind this exact surface."

Good moment to drag the clip slider: the tumour cuts with the brain, and its cross-section
lines up with the bright region in the MRI on the cut face. That alignment is the check —
it means the mask really is registered to the volume, not just plausible-looking.

### Beat 6 — "What the classifier saw"

**Get ahead of the caveat. Say it before you're asked:**

> "There's a classifier upstream — EfficientNet-B3, four classes, with Grad-CAM. It
> scores 0.98 macro-F1 on its own held-out data. But it's out of distribution here:
> trained on pre-operative JPEGs, applied to post-operative volumes with no tumour-type
> ground truth. It's wired in to show the plumbing, and it's flagged as
> out-of-distribution in the notebook, in the metrics file, and on that amber strip.
> I'd rather show you that than hide it."

### Beat 7 — "Six months later" ← **this is the payoff**

The volume switches, the camera does not move, the mesh grows in place.

> "Same camera. A Fisher–KPP physics-informed network recovered this patient's
> diffusion and proliferation constants, integrated 180 days forward, and then the
> segmenter and the GAN ran *again* on the predicted future. That's the closed loop."

### Beat 8 — "Growth, quantified"

> "The cyan is the tumour today. The red shell around it is what the PINN predicts will
> be new tissue in six months — ×1.76, plus 73.6 millilitres. And here's the check I care
> about: the PINN's density threshold says 170.6 mL, while re-segmenting the predicted
> volume from scratch says 169.7. Those are thresholded on completely different quantities
> and they agree to half a percent."

### Beat 9 — "Your turn"

Toggle **Volume** back and forth manually a few times. The A/B is more convincing under
your hand than on rails.

---

## The ninety-second version

When you get cut short: beat 1 → skip to beat 5 → beat 7 → beat 8. Use `›` on the caption card.

> "Sparse slices in, dense 3D volume out. → That's the segmented tumour, in 3D, at the
> volume the panel reports. → Six months forward, predicted by a physics-informed network
> and re-segmented. → Cyan is now, red is new: ×1.76 growth, with two independent
> estimates agreeing to half a percent."

Then stop talking.

---

## Questions you will get, and honest answers

**"Is this clinically usable?"**
No. It's a research prototype and it says so on the landing page. Single-fold,
single-institution evaluation, no external test set, no clinical validation against
follow-up imaging.

**"Your classifier scores 0.98 — isn't that dataset leaky?"**
The shipped train/test split of `brain-tumor-mri-dataset` is widely reported as leaky,
so ORACLE doesn't use it. It pools both folders, fingerprints every image with a
256-bit perceptual hash, groups near-duplicates with union-find, and re-splits so no
group spans train, validation and test. 0.9802 accuracy / 0.9800 macro-F1 / 0.9990 AUC
is on *that* split. Worth adding, unprompted: the shipped boundary contained zero
byte-identical images when measured — weaker evidence of leakage than its reputation
suggests. What is *not* solved is patient identity; the source datasets carry no patient
IDs, so two slices from one patient can still land on opposite sides if they aren't
near-duplicates.

**"Is the predicted growth physically calibrated?"**
No, and this matters. The Stage-4 growth knobs are tuned for *visible* change so the
demo reads on screen. Set `rho_scale` and `D_scale` to 1.0 in `GROWTH_CFG` for the
physically faithful regime — the evolution is much subtler. The PDE and the recovered
parameters are real; the display scaling is not.

**"What's the actual segmentation Dice?"**
≈0.84 on the held-out test split, with post-processing. On this particular demo patient,
mean per-slice Dice is 0.802. Model selection used the deployed path — EMA + TTA +
post-processing — deliberately, so the reported number is the number you get at
inference, not a flattering raw forward pass.

**"Why is the confidence only 74.7% if the classifier is that accurate?"**
Two reasons, both honest. It's out of distribution here. And softmax confidence is
uncalibrated and currently *under*-confident — mean max-softmax 0.9525 against 0.9802
accuracy, a side effect of 0.05 label smoothing. Read it as a ranking score.

**"Is that red region actually predicted, or just the tumour drawn bigger?"**
Predicted, and the loop is closed. The Fisher–KPP PINN recovers this patient's diffusion and
proliferation constants, integrates the density forward 180 days, and the red mesh is that
density thresholded at 0.25. Then — independently — nnU-Net re-segments the synthesised
future volume and gets 169.7 mL against the PDE's 170.6. The red is a prediction that two
different models agree on, not a dilation.

**"How do you aggregate per-slice predictions into one answer?"**
Not by averaging — that's the interesting part. Most slices contain no tumour, so a
naive mean calls this glioma patient `notumor` at 64.3%. Ranking slices by tumour
evidence and averaging the top ten gives `glioma` at 74.7%. Both are printed at runtime
so the bias stays visible.

**"Can I run it?"**
Yes. `full_pipeline_testing.ipynb` is inference-only; attach the dataset and the model
on Kaggle and Run All — a preflight cell stops immediately with instructions if either
mount is missing. If you'd rather not attach anything,
[`demo_full_run_executed.ipynb`](demo_full_run_executed.ipynb) is the same notebook
already run, with every figure retained.

**"What would you do next?"**
True 3D volumetric PDE instead of slice-wise 2D; longitudinal ground truth to validate
the growth horizon; calibrating the UQ intervals; and multi-centre external validation.
These are in the README's Future Work section as open directions, not pending tasks.

---

## If the network dies

In order of preference:

1. **Local server** — `cd docs/brain-viewer && python3 -m http.server 8080` → `localhost:8080/?tour=1`. Fully offline, identical to the live site.
2. **The recorded screencast** — `demo.mp4`, if you've captured it.
3. **[`demo_full_run_executed.ipynb`](demo_full_run_executed.ipynb)** in Jupyter — every stage with its figures, no data or GPU required.
4. **Static figures** in [`readme_assets/`](readme_assets/).

**Do not promise a live pipeline run.** There is no MRI data on the presenting machine
and the pipeline needs a GPU session. The viewer, the notebook outputs, and the figures
are the demo; the training and inference happened elsewhere.
