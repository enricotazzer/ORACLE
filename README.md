# 🔮 ORACLE Oncology Reconstruction And Clinical Learning Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)

**Predicting Tomorrow's Tumors from Today's MRI**

</div>

---

## 📋 Overview

ORACLE is an end-to-end deep learning framework for brain tumor analysis that combines **detection**, **3D reconstruction**, and **physics-informed growth prediction**. Using multimodal MRI inputs, ORACLE can:

- ✅ Segment and localize tumors with pixel-level precision using 4 MRI modalities (`t1n`, `t1c`, `t2w`, `t2f`)
- 🔄 Reconstruct full 3D brain volumes
- 📈 Predict tumor evolution 3-6 months into the future using Physics-Informed Neural Networks (PINNs)
- 🔍 Provide explainable predictions with Grad-CAM visualizations

This project addresses the critical clinical need for **early intervention planning** by simulating glioma growth patterns based on reaction-diffusion biomechanical models.

---

## ✨ Features

### 🎯 1. Tumor Segmentation & Localization
- UNet++ architecture for pixel-level tumor segmentation (`segmentation_models_pytorch`)
- 4-channel multimodal MRI input (`t1n`, `t1c`, `t2w`, `t2f`) with binary tumor mask targets
- Outputs spatial tumor density maps ready for PINN input
- Visual segmentation masks for clinical interpretability
- Patient-level train/validation/test split to prevent leakage

### 🧊 2. Single-Slice to 3D Volume Reconstruction
- 5-channel / 5-slice context reconstruction model
- Inputs: `t1n`, `t1c`, `t2w`, `t2f`, `mask_density` over depth window `[z..z+4]`
- Target: next non-overlapping slice `t1n[z+5]`
- Bidirectional autoregressive full-volume inference with forward/backward fusion
- Preserves anatomical features and tumor morphology
- Quantitative quality reported with PSNR/SSIM on held-out evaluation

### ⏱️ 3. Physics-Informed Tumor Growth Prediction
- PINN implementation of Fisher-Kolmogorov reaction-diffusion equation
- Patient-specific parameter estimation (diffusion D, proliferation ρ)
- Forward prediction with uncertainty quantification
- Ensemble-based confidence intervals

---

## 🏗️ Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ORACLE Pipeline                          │
└─────────────────────────────────────────────────────────────┘
                            │
                     Multimodal MRI Slice/Context Input
                            │
                            ▼
                  ┌─────────────────────┐
                  │ Segmentation Module │ → Tumor Detection + Tumor Mask
                  │   (Unet++)          │
                  │   Encoder-Decoder   |
                  │   Skip Connections  │  
                  └─────────────────────┘
                            │
                            ▼ (Yes)
                  ┌────────────────────────┐
                  │ Reconstruction Module  │ → Full 3D Volume
                  │ (5-ch/5-slice context  │
                  │ + autoregressive fuse) │
                  └────────────────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │   PINN Module     │ → Growth Prediction
                  │ ∂u/∂t = D∇²u +    │    (t + 3-6 months)
                  │      ρu(1-u)      │    + Uncertainty
                  └───────────────────┘
                            │
                            ▼
                 Visualization Dashboard
                 (3D rendering + heatmaps)
```

### Module Details

| Module | Input | Output | Technology |
|--------|-------|--------|------------|
| **Detection** | 4-channel MRI slice (`t1n`,`t1c`,`t2w`,`t2f`) | Detection + Mask | Unet++ (EfficientNet encoder) |
| **Reconstruction** | 5-channel, 5-slice context window | 3D volume (autoregressive) | Context model + bidirectional fusion |
| **PINN** | u₀(x), time t | Predicted u(t,x) | Physics-informed neural network |

---
