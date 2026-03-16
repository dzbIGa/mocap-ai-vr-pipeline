# Motion Capture and AI Integration for VR-Based 3D Animation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Unreal Engine](https://img.shields.io/badge/Unreal%20Engine-5.x-black)](https://www.unrealengine.com/)

> **This repository is directly related to the manuscript:**
>
> He, F., Liu, X., Zhu, Z., Zhai, L., & Liu, J. (2025).  
> *"The Role of Motion Capture and AI in VR-Based 3D Animation Design and Production."*  
> Submitted to **The Visual Computer** (Springer).
>
> **If you use this code or data in your research, please cite this manuscript.**  
> See the [Citation](#citation) section below.

---

## Overview

This repository implements the **Motion Capture–AI Animation Pipeline** described in the manuscript — an integrated, latency-aware framework for VR-based 3D animation production. The pipeline combines:

- **Motion capture acquisition** — loading and normalizing optical/inertial/markerless data
- **Kalman filter smoothing** — noise reduction under real-time VR constraints (Eq. 2)
- **IK retargeting** — inverse kinematics solver for cross-character motion adaptation (Eq. 3)
- **CNN-LSTM facial animation** — blendshape weight prediction from facial markers (Eq. 4)
- **Latency monitoring** — frame-level pipeline delay measurement (Eq. 5)
- **Evaluation module** — accuracy, efficiency, and latency metrics from the paper

All code is implemented in pure Python (NumPy/SciPy), with no deep learning framework required for inference.

---

## Repository Structure

```
mocap-ai-vr-pipeline/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── pipeline/
│   ├── acquisition/
│   │   └── mocap_reader.py          # Load & normalize .json / .c3d / .bvh data
│   ├── preprocessing/
│   │   ├── kalman_smoother.py       # Kalman filter noise reduction (Eq. 2)
│   │   └── normalize.py             # Frame cleaning, resampling, amplitude scaling
│   ├── facial_animation/
│   │   ├── cnn_lstm_model.py        # CNN-LSTM blendshape predictor (Eq. 4)
│   │   └── train.py                 # Training script (MSE, Adam, early stopping)
│   ├── retargeting/
│   │   └── ik_solver.py             # IK retargeting solver (Eq. 3)
│   └── rendering/
│       └── latency_monitor.py       # Latency measurement (Eq. 5)
│
├── evaluation/
│   ├── metrics.py                   # Accuracy, efficiency, latency metrics
│   └── compare_systems.py           # Optical vs inertial vs markerless comparison
│
├── data/
│   ├── README.md                    # Dataset description
│   ├── sample/                      # Ready-to-use sample data (8 subjects)
│   │   ├── raw_mocap.json           # Raw 3D marker trajectories
│   │   ├── smoothed_mocap.json      # Post-Kalman smoothed trajectories
│   │   ├── facial_animation.json    # Facial markers + blendshape weights
│   │   ├── ik_retargeting_results.json  # IK convergence data
│   │   ├── evaluation_metrics.json  # All quantitative results from paper
│   │   ├── fig_system_comparison.png
│   │   ├── fig_efficiency.png
│   │   ├── fig_latency.png
│   │   └── fig_kalman.png
│   └── model_weights/
│       ├── cnn_lstm_facial.npz      # Pre-trained CNN-LSTM weights
│       └── training_info.json       # Training hyperparameters
│
├── configs/
│   └── pipeline_config.yaml         # All pipeline parameters
│
├── notebooks/
│   ├── 01_pipeline_demo.ipynb       # End-to-end pipeline walkthrough
│   └── 02_results_reproduction.ipynb # Reproduce all paper figures & tables
│
└── scripts/
    └── generate_all.py              # Regenerate sample data from scratch
```

---

## Dependencies and Requirements

### System Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.9 or higher |
| OS | Windows 10/11, Ubuntu 20.04+, macOS 12+ |
| RAM | 4 GB minimum (8 GB recommended) |
| GPU | Not required for inference (optional for retraining) |
| Unreal Engine | 5.x (optional, for real-time rendering integration) |

### Python Installation

```bash
pip install -r requirements.txt
```

**Contents of `requirements.txt`:**

```
numpy>=1.23.0
scipy>=1.9.0
matplotlib>=3.6.0
jupyter>=1.0.0
pyyaml>=6.0
tqdm>=4.65.0
```

> PyTorch is **not required** for inference. The CNN-LSTM model ships with pre-trained NumPy weights. PyTorch is only needed to retrain from scratch.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/mocap-ai-vr-pipeline.git
cd mocap-ai-vr-pipeline
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the end-to-end demo

```bash
jupyter notebook notebooks/01_pipeline_demo.ipynb
```

### 4. Reproduce paper results (all figures and tables)

```bash
jupyter notebook notebooks/02_results_reproduction.ipynb
```

### 5. Run as Python script (no Jupyter needed)

```python
import sys
sys.path.insert(0, '.')

import numpy as np
from pipeline.acquisition.mocap_reader      import load_mocap_json
from pipeline.preprocessing.kalman_smoother import kalman_smooth
from pipeline.retargeting.ik_solver          import retarget_sequence
from pipeline.facial_animation.cnn_lstm_model import CNNLSTMFacialAnimator
from pipeline.rendering.latency_monitor      import simulate_latency_report
from evaluation.metrics                      import load_and_report

# Load data
raw = load_mocap_json('data/sample/raw_mocap.json')

# Kalman smoothing (Eq. 2, K=0.3)
smoothed = kalman_smooth(raw['subject_01'], gain=0.3)

# IK retargeting (Eq. 3)
ik_result = retarget_sequence(smoothed[:50])
print(f"IK convergence: {ik_result['convergence_rate_pct']:.1f}%")

# CNN-LSTM facial animation (Eq. 4)
model = CNNLSTMFacialAnimator('data/model_weights/cnn_lstm_facial.npz')
import json
with open('data/sample/facial_animation.json') as f:
    fac = json.load(f)
import numpy as np
markers = np.array(fac['subject_01']['markers'], dtype=np.float32)
weights = model.predict(markers[:100])
print(f"Blendshape weights shape: {weights.shape}")

# Latency (Eq. 5)
report = simulate_latency_report()
print(f"Mean latency: {report['mean_latency_ms']} ms  |  Compliance: {report['budget_compliance_pct']}%")

# Full metrics report
load_and_report('data/sample/evaluation_metrics.json')
```

---

## Key Algorithms

### Eq. 2 — Kalman Filter Noise Reduction

$$\hat{P}_i(t) = K \cdot P_i(t) + (1 - K) \cdot \hat{P}_i(t-1)$$

| Parameter | Value |
|-----------|-------|
| Kalman gain K | 0.3 (range 0.2–0.4) |
| Measurement noise | 0.01–0.05 |
| Process noise | low |
| Noise reduction achieved | ~77% |

---

### Eq. 3 — IK Retargeting

$$\theta_{ik} = \arg\min_\theta \sum_j \| M_j(\theta) - P_i \|^2$$

| Parameter | Value |
|-----------|-------|
| Convergence tolerance ε | 10⁻³ m |
| Max iterations | 100 |
| Convergence rate | 100% on sample data |

---

### Eq. 4 — CNN-LSTM Facial Animation

$$\mathbf{w}(t) = f_{\text{CNN-LSTM}}(\mathbf{F}(t))$$

| Component | Details |
|-----------|---------|
| Input | 68 facial markers × 3 coords per frame |
| CNN | 2 × Conv1D, 64 filters, kernel=3, ReLU |
| LSTM | 128 hidden units, 2 layers |
| Output | 52 blendshape weights ∈ [0, 1] |
| Loss | MSE |
| Optimiser | Adam (lr=1e-3) |
| Stopping | Early stopping, patience=10 |
| Val MSE | 0.0023 |

---

### Eq. 5 — VR Rendering Latency

$$\Delta t = t_{\text{render}} - t_{\text{mocap}}$$

| Metric | Value |
|--------|-------|
| Target FPS | 90 |
| Mean latency | 17.4 ms |
| Max latency | 19.8 ms |
| Budget | 20 ms |
| Compliance | 98.2% |

---

## Dataset

The motion capture dataset includes **8 participants** performing locomotion, upper-body gestures, interaction tasks, and full-body movements in a controlled indoor environment.

| File | Description | Shape |
|------|-------------|-------|
| `raw_mocap.json` | Raw 3D marker trajectories | 8 subjects × 300 frames × 31 markers × 3 |
| `smoothed_mocap.json` | Post-Kalman trajectories | same |
| `facial_animation.json` | Facial markers + blendshape labels | 8 subjects × 300 frames |
| `ik_retargeting_results.json` | IK convergence per frame | 8 subjects × 300 frames |
| `evaluation_metrics.json` | All quantitative results | — |

> **Full dataset DOI:** *to be assigned on Zenodo upon manuscript acceptance.*  
> The `data/sample/` directory contains complete synthetic representative data matching all statistics reported in the paper.

---

## Results

| System | Accuracy | Cost |
|--------|----------|------|
| Optical (Vicon) | **95%** | $200K |
| Inertial (Xsens) | 85% | $15K |
| Markerless (DeepMotion) | 70% | $10K |

**AI Integration Impact:**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Production time | 14 days | 7 days | −50% |
| Motion cleanup | 8.0 h | 1.6 h | −80% |
| Facial animation | 6.0 h | 1.2 h | −80% |

---

## Citation

If you use this code, pipeline, or data in your research, please cite:

```bibtex
@article{he2025mocap_vr,
  author  = {He, Fang and Liu, Xun and Zhu, Zhaoye and Zhai, Lanru and Liu, Jiaxin},
  title   = {The Role of Motion Capture and AI in VR-Based 3D Animation Design and Production},
  journal = {The Visual Computer},
  year    = {2025},
  note    = {Submitted},
  url     = {https://github.com/<your-username>/mocap-ai-vr-pipeline}
}
```

> **Note:** This repository is directly associated with the manuscript submitted to *The Visual Computer*.  
> Readers are encouraged to cite the manuscript when referencing or building upon this code.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

**Corresponding author:** Xun Liu — xunliu2@cafaedu.com  
Academy of Fine Arts, YunNan Arts University, Kunming, China
