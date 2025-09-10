# Unified FSDD Trainer — GRU vs LSTM (GPU-optimized)

This repository provides a **unified training pipeline** for the [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset), allowing you to **compare GRU vs LSTM** models under different constraints:

- **Deterministic training** (seed control, reproducibility)
- **Light augmentation** (SpecAugment-like masking)
- **CUDA optimizations** (AMP, TF32, cuDNN tweaks)
- **Multiple experimental tasks** (A, B1, B2, C)

Outputs include:
- **Training curves**
- **Confusion matrices**
- **Metrics tables**
- **Flowchart of the pipeline**

---

## 📂 Tasks

| Task | Description |
|------|-------------|
| **A** | Baseline — no constraints |
| **B1** | Each recurrent layer limited to **36 kB memory** (hidden size auto-calculated) |
| **B2** | B1 + **integer-only inference** (dynamic INT8 quantization of RNN/Linear; evaluated on CPU) |
| **C**  | **Power-of-two weights (projection)** starting from B1 topology.<br>Supports multiple approaches: `ema`, `snap_epoch`, `snap_step`, `stochastic`, `inq`, `row_shared`, `apot2`, `mixed_pot`. Knowledge distillation (KD) optional. |

---

## 🧩 Models & Features

- **Models:** `gru`, `lstm`, or `both` (compare side-by-side)
- **Features:**
  - `mfcc20` (20-dim MFCC via torchaudio)
  - `mfcc39` (39-dim MFCC + Δ/ΔΔ via librosa)

---

## ⚙️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
