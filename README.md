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

## Approaches for Task C (PoT)

`C_APPROACH` options (used after `C_PROJ_START_EPOCH`):

- **`ema`**: Exponential-moving blend toward nearest ±2^k (gentle, default).
- **`snap_epoch`**: Hard snap at end of each epoch during PoT phase.
- **`snap_step`**: Hard snap every N steps (`C_PROJ_EVERY_N_STEPS`).
- **`stochastic`**: Stochastic rounding between neighbor exponents.
- **`row_shared`**: Per-row shared exponent for 2D weights.
- **`apot2`**: Approximate as sum of two powers of two (±2^k1 ± 2^k2).
- **`mixed_pot`**: Apply PoT to RNN weights only; keep fc/bias float (`C_TARGET_SCOPE="rnn_only"`).
- **`inq`**: Incremental Network Quantization-style staging (progressively freeze top-|w|%).

Optional **KD** (knowledge distillation) for any C_* approach:
- `KD_ENABLE=True`, with `KD_TAU`, `KD_ALPHA` controlling soft loss mix.

---

> **GPU note:** If you use CUDA, prefer installing the correct PyTorch build from https://pytorch.org/get-started/locally/ and then keep the rest of the requirements as-is.

---

## Quickstart

```bash
# 0) (Optional) Create a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 1) Install dependencies
pip install -r requirements.txt

# 2) Get the dataset (FSDD) next to this repo
git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git

# 3) Run (defaults are set at top of script)
python fsdd_trainer.py
