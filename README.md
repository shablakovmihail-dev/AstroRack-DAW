# AstroRack DAW v8.3.6 🪐🎛️

> **Desktop Digital Audio Workstation-style Lunar & Planetary Astrophotography Post-Processor.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-41CD52?logo=qt&logoColor=white)](https://riverbankcomputing.com/software/pyqt/)
[![NumPy](https://img.shields.io/badge/Math-NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![SciPy](https://img.shields.io/badge/DSP-SciPy-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![License: CC0](https://img.shields.io/badge/License-Public%20Domain%20%2F%20CC0-00E5FF.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

---

## 🌟 Overview

**AstroRack DAW** reimagines planetary, lunar, and solar astrophotography post-processing through the familiar, tactile paradigm of a **Digital Audio Workstation (DAW) 19-inch mastering rack**. 

Instead of opaque, destructive one-click filters, AstroRack DAW gives astrophotographers parametric control over non-destructive modular inserts, real-time Dry/Wet mixing, sub-pixel atmospheric dispersion correction, À trous wavelet decomposition, and edge-protected Richardson-Lucy deconvolution — all driven by a responsive, multithreaded DSP pipeline.

---

## ✨ Key Features & Highlights

- 🌊 **Multi-layer À trous Wavelet Sharpening**:
  - High-precision dyadic B3-spline wavelet decomposition.
  - Independent per-layer gain and threshold denoising to reveal delicate planetary cloud belts, lunar rilles, and solar granules without noise amplification.
  - Seamless parallel Dry/Wet blend controls for each rack module.

- 🔬 **Richardson-Lucy Deconvolution with Ringing Protection**:
  - Accelerated iterative deconvolution modeling telescope point-spread functions (PSF).
  - Built-in dynamic planetary limb detection and edge-ringing dampening mask to prevent Gibbs artifacts on bright planet edges (Jupiter, Saturn, Venus).

- 🌈 **Atmospheric Dispersion Corrector (Auto RGB Phase Align)**:
  - Sub-pixel Fourier phase-correlation and normalized cross-correlation (NCC) to correct atmospheric prismatic chromatic aberration across Red, Green, and Blue channels.

- 🌓 **Interactive Split View with Dynamic Lunar Phase Handle**:
  - Side-by-side or sliding split-screen view comparing raw and mastered frames.
  - Ergonomic handle with dynamic lunar phase rendering (from New Moon crescent to Full Moon disk) tracking split position.

- 📊 **Logarithmic Luminance / RGB Spectrum Histogram**:
  - Real-time 32-bit floating-point histogram with switchable linear/logarithmic scales.
  - Auto-Planet Region-of-Interest (ROI) background-sky suppression mask to measure celestial body dynamics accurately.
  - Hardware-style clipping LED warning indicators for black point clipping and highlight saturation.

- 🎚️ **Dual-State A/B Snapshot Comparison & Master Bypass**:
  - Instant toggle between two independent rack parameter states (State A vs. State B).
  - Master Bypass with analog-style tactile relay sound feedback for critical mastering decisions.

- 🔭 **Live Pixel Telemetry & Probing**:
  - High-precision 32-bit float coordinate inspector tracking RGB values, luminance, signal levels, and zoom ratios.

---

## 🛠️ Technology Stack

- **Core Framework**: [Python 3.10+](https://www.python.org/)
- **UI Architecture**: [PyQt6](https://riverbankcomputing.com/software/pyqt/) (Hardware-accelerated custom widgets, modern dark DAW theme)
- **DSP & Mathematics**: [NumPy](https://numpy.org/) & [SciPy](https://scipy.org/) (Convolution kernels, À trous wavelets, matrix algebra)
- **Computer Vision & Image Processing**: [OpenCV](https://opencv.org/) (Phase correlation, morphological filters, fast multi-channel blending)

---

## 🚀 Quick Start Guide

### Prerequisites

Ensure you have Python 3.10 or higher installed:

```bash
python --version
```

### 1. Clone the Repository

```bash
git clone https://github.com/shablakovmihail-dev/AstroRack-DAW.git
cd AstroRack-DAW
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run AstroRack DAW

```bash
python main.py
```

---

## 📁 Repository Structure

```
AstroRack-DAW/
├── assets/                 # Icons, splash imagery, fonts, and tactile sound effects
├── core/                   # DSP engines, À trous wavelets, Richardson-Lucy, pipeline thread
│   ├── analysis.py         # Dynamic histogram, ROI segmentation, and clipping telemetry
│   ├── enhance.py          # Color grading, contrast, and local micro-structure algorithms
│   ├── pipeline.py         # Asynchronous worker QThread with stage-level caching
│   ├── processing.py       # Core DSP routines (wavelets, deconvolution, alignment)
│   └── sound.py            # Tactile audio feedback (analog relay clicks, switches)
├── ui/                     # Dark DAW GUI components and custom canvas
│   ├── canvas.py           # Pan/zoom viewport with lunar split-view slider
│   ├── dialogs.py          # About dialog, presets, and dialog helpers
│   ├── main_window.py      # Main rack frame, toolbars, and signal routing
│   ├── splash.py           # Startup calibration splash screen
│   └── widgets.py          # DAW rack modules, knobs, LED meters, and sliders
├── samples/                # Sample reference planetary and lunar targets
├── tests/                  # Unit and integration test suite
├── .gitignore              # Production Git exclusion rules
├── requirements.txt        # Minimal production dependencies
├── main.py                 # Application entry point
└── README.md               # Project documentation
```

---

## 🛰️ Developer Observatory

```
📡 Ground Station: Astana, KZ • Epoch J2026.8
```

---

## 📄 License

This project is released under the **Public Domain / Zero Rights Reserved (CC0 1.0 Universal)** license.  
Dedicated to amateur astronomers and astrophotographers worldwide — free for personal and commercial use without restrictions.
