# 🔮 ORACLE Oncology Reconstruction And Clinical Learning Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)

**Predicting Tomorrow's Tumors from Today's MRI**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Datasets](#-datasets) • [Results](#-results) • [Roadmap](#-roadmap)

</div>

---

## 📋 Overview

ORACLE is an end-to-end deep learning framework for brain tumor analysis that combines **detection**, **3D reconstruction**, and **physics-informed growth prediction**. Starting from a single 2D MRI slice, CHRONOS can:

- ✅ Segment and localize tumors with pixel-level precision (DICE >0.90)
- 🔄 Reconstruct full 3D brain volumes
- 📈 Predict tumor evolution 3-6 months into the future using Physics-Informed Neural Networks (PINNs)
- 🔍 Provide explainable predictions with Grad-CAM visualizations

This project addresses the critical clinical need for **early intervention planning** by simulating glioma growth patterns based on reaction-diffusion biomechanical models.

---

## ✨ Features

### 🎯 1. Tumor Segmentation & Localization
- U-Net architecture for pixel-level tumor segmentation
- Precise tumor boundary delineation with DICE score >0.90
- Outputs spatial tumor density maps ready for PINN input
- Visual segmentation masks for clinical interpretability
- Handles both tumor detection and localization in one step

### 🧊 2. Single-Slice to 3D Volume Reconstruction
- Slice-based latent diffusion model with positional encoding
- Generates complete 3D MRI volumes from a single 2D slice
- Preserves anatomical features and tumor morphology
- PSNR >26 dB reconstruction quality

### ⏱️ 3. Physics-Informed Tumor Growth Prediction
- PINN implementation of Fisher-Kolmogorov reaction-diffusion equation
- Patient-specific parameter estimation (diffusion D, proliferation ρ)
- Forward prediction with uncertainty quantification
- Ensemble-based confidence intervals

---

## 🏗️ Architecture

Segmentation Module     â”‚ â†’ Tumor Mask uâ‚€(x)
                â”‚      (U-Net)             â”‚   + Tumor Detection
                â”‚  â€¢ Encoder-Decoder       â”‚   (Binary: Present/Absent)
                â”‚  â€¢ Skip Connections  
### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ORACLE Pipeline                          │
└─────────────────────────────────────────────────────────────┘
                            │
                   Single MRI Slice Input
                            │
                            ▼
                  ┌─────────────────────┐
                  │ Segmentation Module │ → Tumor Detection + Tumor Mask
                  │   (Unet++)          │
                  │   Encoder-Decoder   │
                  │   Skip Connections  │  
                  └─────────────────────┘
                            │
                            ▼ (Yes)
                  ┌────────────────────────┐
                  │ Reconstruction Module  │ → Full 3D Volume
                  │  (Diffusion Model +    │
                  │   Positional Encoding) │
                  └────────────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │  Segmentation    │ → Tumor Mask u₀(x)
                  └──────────────────┘
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
| **Detection** | 2D MRI slice (224×224) | Binary classification + heatmap | Transfer learning (ResNet50/EfficientNet) |
| **Reconstruction** | Single slice + position | 3D volume (256×256×N) | Latent diffusion with positional embeddings |
| **Segmentation** | 3D volume | Tumor density map u(x) | U-Net or from existing masks |
| **PINN** | u₀(x), time t | Predicted u(t,x) | Physics-informed neural network |

---
