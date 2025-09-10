# Unified FSDD Trainer — GRU vs LSTM (GPU-optimized)

Compare GRU and LSTM on the Free Spoken Digit Dataset (FSDD) with deterministic runs, light augmentation, and optional constraints:
- **Tasks**: A (baseline), B1 (36 kB/layer), B2 (INT8 dynamic quant), C (Power-of-two weights w/ multiple approaches)
- **Preproc**: MFCC-20 (torchaudio) or MFCC-39 (librosa Δ/ΔΔ)
- **Outputs**: training curves, confusion matrix, metrics table, flowchart

## Quickstart

```bash
# 1) Create and activate a virtual env (optional but recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Get the dataset (FSDD) and set the path
# Clone the dataset into ./free-spoken-digit-dataset
git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git

# 4) Run
python fsdd_trainer.py
