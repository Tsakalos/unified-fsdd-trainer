"""
Unified FSDD Trainer — Compare GRU vs LSTM (GPU-optimized, AMP fixed)

- TASK:
    "A"  -> Baseline (no constraints)
    "B1" -> Layer memory limit: 36 kB per layer (auto-solve hidden size)
    "B2" -> B1 + integer-only inference (dynamic INT8 quantization of RNN/Linear; eval on CPU)
    "C"  : Power-of-two weights (projection), starting from B1 topology
           C_APPROACH: "ema" | "snap_epoch" | "snap_step" | "stochastic" | "inq" | "row_shared" | "apot2" | "mixed_pot"
           KD optional for any C_* approach

- MODEL:  "gru" | "lstm" | "both"
- PREPROC: "mfcc20" (torchaudio, 20-D) | "mfcc39" (librosa, 39-D)
- Same split/loaders/training for fair comparisons
- CUDA optimizations: cuDNN benchmark, TF32, pin_memory, non_blocking, AMP
""
"""

# =========================
# CONFIG
# =========================
TASK    = "C"          # "A" | "B1" | "B2" | "C"
MODEL   = "gru"        # "gru" | "lstm" | "both"
PREPROC = "mfcc20"     # "mfcc20" | "mfcc39"

DATASET_PATH = "free-spoken-digit-dataset/recordings"
SAMPLE_RATE  = 8000
BATCH_SIZE   = 64        # 64/128 on GPU -- 32 on CPU
EPOCHS       = 32
LR           = 1e-3
WEIGHT_DECAY = 0         # 1e-4
CLIP_NORM    = 1.0
PATIENCE     = 6
NUM_CLASSES  = 10
SEED         = 42

# Determinism / Data loading
DETERMINISTIC         = True
NUM_WORKERS           = 2       # 0 for hard determinism across OS
PIN_MEMORY_ON_CUDA    = True
PERSISTENT_WORKERS    = True
SPEAKER_AWARE_SPLIT   = True    # avoid speaker leakage

# Augmentation (light SpecAugment-ish on MFCC)
AUG_ENABLE            = True
TIME_MASK_PARAM       = 10      # frames (set 0 to disable)
TIME_MASK_N           = 1       # how many time masks per sample

# Baseline sizes (Task A)
H_GRU  = 64
H_GRU_BASE   = 64
H_LSTM = 128
H_LSTM_BASE  = 128
BIDIR_GRU_A  = True
BIDIR_LSTM_A = False

# Task B/C constraints
LAYER_BUDGET_BYTES = 36 * 1024  # 36 kB
# If you want to re-maximize hidden size in B2 using int8 bytes, set True.
B2_EXPAND_TO_INT8_BUDGET = False

# ===== Task C knobs (new) =====
# Approach selector:
C_APPROACH = "ema"          # "ema" | "snap_epoch" | "snap_step" | "stochastic" | "inq" | "row_shared" | "apot2" | "mixed_pot" | "hard"
C_TARGET_SCOPE = "rnn_only"      # "all" (default) or "rnn_only"
C_PROJ_START_EPOCH   = 10    # when to start enforcing PoT
C_LR_FINE_TUNE       = 5e-4 # LR after C_PROJ_START_EPOCH
C_PROJ_EVERY_N_STEPS = None # or int (e.g., 100) for step-level projection
C_EMA_ALPHA          = 0.85 # EMA blend for "ema" -- higher = gentler
C_CLAMP_EXP_MIN      = -8
C_CLAMP_EXP_MAX      = +8

# INQ staged quantization (for C_APPROACH="inq"): list of (fraction_quantize, epochs_for_stage)
C_INQ_STAGES = [(0.3, 4), (0.6, 4), (1.0, 6)]

# Knowledge Distillation (teacher is the same topology trained in float)
KD_ENABLE             = False    # set True to distill to Task C student
KD_TAU                = 3.0      # temperature
KD_ALPHA              = 0.6      # loss = alpha*KD + (1-alpha)*CE

# =========================
import os, random, time
os.environ["MPLBACKEND"] = "Agg"
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.ao.quantization import get_default_qat_qconfig

# PyTorch light on CPU threads
try:
    torch.set_num_threads(2)
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

# Force safer dataloader defaults on Windows (avoid multiprocessing)
if os.name == "nt":
    NUM_WORKERS = 0            # NO worker processes (prevents re-import in children)
    PIN_MEMORY_ON_CUDA = False # pin_memory doesn’t help on CPU and costs RAM
    PERSISTENT_WORKERS = False

try:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False

# Audio libs
import soundfile as sf
import torchaudio
import torchaudio.transforms as T


# Librosa only if needed >> PREPROC = "mfcc39"
try:
    import librosa
except Exception as e:
    raise RuntimeError("PREPROC='mfcc39' requires librosa.") from e

# === Repro & device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Speed knobs (safe defaults)
if device.type == "cuda" and not DETERMINISTIC:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

PIN_MEMORY   = (device.type=="cuda") and PIN_MEMORY_ON_CUDA
PERSISTENT   = PERSISTENT_WORKERS and (NUM_WORKERS > 0)

# =========================
# Plot utilities
# =========================
class TrainRecorder:
    def __init__(self):
        self.epochs, self.train_loss, self.train_acc, self.val_loss, self.val_acc = [], [], [], [], []
    def add(self, ep, tr_loss, tr_acc, va_loss, va_acc):
        self.epochs.append(ep); self.train_loss.append(tr_loss); self.train_acc.append(tr_acc)
        self.val_loss.append(va_loss); self.val_acc.append(va_acc)

def plot_training_curves(rec, tag, show=False):
    if not rec.epochs: print("[plot] No epochs; skipping curves."); return
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.set_title(f"Training & Validation Curves ({tag})")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.plot(rec.epochs, rec.train_loss, label="train loss")
    ax1.plot(rec.epochs, rec.val_loss, label="val loss")
    ax1.grid(alpha=0.3); ax1.legend(loc="upper right")
    ax2 = ax1.twinx(); ax2.set_ylabel("Accuracy")
    ax2.plot(rec.epochs, rec.train_acc, linestyle="--", label="train acc")
    ax2.plot(rec.epochs, rec.val_acc, linestyle="--", label="val acc")
    ax2.legend(loc="lower right")
    plt.tight_layout(); fname = f"training_curves_{tag}.png"
    plt.savefig(fname, dpi=160)
    if show: plt.show()
    plt.close(); print(f"[plot] Saved training curves → {fname}")

@torch.no_grad()
def collect_predictions(model, loader, device_override=None):
    dev = device_override if device_override is not None else device
    model.eval(); ys_true, ys_pred = [], []
    for xb, lengths, yb in loader:
        xb = xb.to(dev, non_blocking=(dev.type=="cuda"))
        lengths = lengths.to(dev, non_blocking=(dev.type=="cuda"))
        yb = yb.to(dev, non_blocking=(dev.type=="cuda"))
        with torch.amp.autocast("cuda", enabled=(dev.type=="cuda")):
            logits = model(xb, lengths)
        ys_true.extend(yb.tolist()); ys_pred.extend(logits.argmax(dim=1).tolist())
    return ys_true, ys_pred

@torch.no_grad()
def measure_latency(model, loader, device, warmup_batches=3, measure_batches=10):
    """
    Measures per-sample latency (ms/sample): mean and p95.
    - Uses a few batches from `loader` (test loader is fine).
    - For CUDA, synchronizes before/after timing.
    """
    model.eval()
    times_per_sample = []
    it = iter(loader)

    # Warmup
    for _ in range(warmup_batches):
        try:
            xb, lengths, _ = next(it)
        except StopIteration:
            it = iter(loader)
            xb, lengths, _ = next(it)
        xb = xb.to(device, non_blocking=(device.type=="cuda"))
        lengths = lengths.to(device, non_blocking=(device.type=="cuda"))
        if device.type == "cuda": torch.cuda.synchronize()
        _ = model(xb, lengths)
        if device.type == "cuda": torch.cuda.synchronize()

    # Measure
    for _ in range(measure_batches):
        try:
            xb, lengths, _ = next(it)
        except StopIteration:
            it = iter(loader)
            xb, lengths, _ = next(it)
        xb = xb.to(device, non_blocking=(device.type=="cuda"))
        lengths = lengths.to(device, non_blocking=(device.type=="cuda"))

        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(xb, lengths)
        if device.type == "cuda": torch.cuda.synchronize()
        dt = time.perf_counter() - t0  # seconds for the batch

        times_per_sample.append(1000.0 * dt / xb.size(0))  # ms/sample

    arr = np.array(times_per_sample, dtype=np.float64)
    return float(arr.mean()), float(np.percentile(arr, 95.0))

def plot_confusion_counts(y_true, y_pred, tag, class_names=None, show=False):
    if not _HAS_SK:
        print("[plot] sklearn not available; skipping confusion matrix.")
        return
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names) if class_names else 10))
    if class_names is None: class_names = [str(i) for i in range(cm.shape[0])]
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.set_title(f"Confusion Matrix — {tag}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:d}", ha="center", va="center",
                    color="black" if cm[i,j] > cm.max()/2 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{tag}.png", dpi=160)
    if show: plt.show()
    plt.close(); print(f"[plot] Saved confusion matrix → confusion_matrix_{tag}.png")

def compute_overall_metrics(y_true, y_pred):
    """
    Returns dict with only what we need for a balanced dataset:
    - accuracy
    - macro F1
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

def compute_per_class_metrics(y_true, y_pred):
    """Returns dict with precision, recall, f1, support (per class)."""
    if not _HAS_SK: return None
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    return {"precision": prec, "recall": rec, "f1": f1, "support": supp}

def print_overall_result(tag, metrics, mean_ms, p95_ms):
    """
    Expects metrics dict from overall_metrics(...) and latency numbers (can be NaN placeholders).
    """
    print(f"\n[{tag}]")
    print(f"  Accuracy: {metrics['acc']:.4f}")
    print(f"  Precision/Recall/F1 (micro): {metrics['precision_micro']:.4f} / "
          f"{metrics['recall_micro']:.4f} / {metrics['f1_micro']:.4f}")
    print(f"  Precision/Recall/F1 (macro): {metrics['precision_macro']:.4f} / "
          f"{metrics['recall_macro']:.4f} / {metrics['f1_macro']:.4f}")
    print(f"  Latency ms/sample (mean / p95): {mean_ms} / {p95_ms}")

def print_metrics_table_compare(metrics_gru, metrics_lstm):
    """Terminal table comparing GRU vs LSTM per class."""
    if metrics_gru is None or metrics_lstm is None:
        print("[metrics] scikit-learn not found; skipping per-class table."); return
    header = (
        "Class |  P_gru  R_gru  F1_gru |  P_lstm R_lstm F1_lstm | Support\n" + "-" * 64
    )
    print(header)
    for c in range(NUM_CLASSES):
        print(
            f"{c:^5} | "
            f"{metrics_gru['precision'][c]:6.3f} {metrics_gru['recall'][c]:6.3f} {metrics_gru['f1'][c]:6.3f} | "
            f"{metrics_lstm['precision'][c]:6.3f} {metrics_lstm['recall'][c]:6.3f} {metrics_lstm['f1'][c]:6.3f} | "
            f"{int(metrics_gru['support'][c]):7d}"
        )
    print("-" * 64)

def plot_metrics_table_compare(metrics_gru, metrics_lstm, tag, show=False):
    """
    Heatmap-like table of per-class metrics comparing GRU vs LSTM.
    Rows: classes 0..9
    Cols: [P_gru, R_gru, F1_gru, P_lstm, R_lstm, F1_lstm]
    """
    if metrics_gru is None or metrics_lstm is None:
        print("[plot] scikit-learn not found; skipping metrics table figure.");
        return
    import numpy as np
    data = np.stack(
        [
            metrics_gru["precision"], metrics_gru["recall"], metrics_gru["f1"],
            metrics_lstm["precision"], metrics_lstm["recall"], metrics_lstm["f1"],
        ],
        axis=1,  # (C, 6)
    )
    colnames = ["P_gru", "R_gru", "F1_gru", "P_lstm", "R_lstm", "F1_lstm"]
    rownames = [str(i) for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    im = ax.imshow(data, aspect="auto", vmin=0.9, vmax=1.0)
    ax.set_title(f"Per-class metrics (GRU vs LSTM) — {tag}")
    ax.set_xlabel("Metric"); ax.set_ylabel("Class")
    ax.set_xticks(range(len(colnames))); ax.set_xticklabels(colnames)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(rownames)

    # Annotate with values
    for i in range(NUM_CLASSES):
        for j in range(len(colnames)):
            ax.text(j, i, f"{data[i, j]:.2f}",
                    ha="center", va="center",
                    color="black" if data[i, j] > 0.5 else "white")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fname = f"metrics_table_{tag}.png"
    plt.savefig(fname, dpi=160)
    if show: plt.show()
    plt.close(); print(f"[plot] Saved metrics table → {fname}")

def plot_mfcc_image(mfcc_TF, tag, show=False):
    if hasattr(mfcc_TF, "detach"): mfcc_TF = mfcc_TF.detach().cpu().numpy()
    img = mfcc_TF.T
    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.imshow(img, aspect="auto", origin="lower")
    ax.set_title(f"MFCC (T×F) — {tag}"); ax.set_xlabel("Time frames"); ax.set_ylabel("MFCC coeff")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fname = f"mfcc_{tag}.png"
    plt.savefig(fname, dpi=160)
    if show: plt.show()
    plt.close(); print(f"[plot] Saved MFCC image → {fname}")

def plot_pipeline_flowchart(tag, preproc_label, model_names, extra_note=""):
    fig, ax = plt.subplots(figsize=(6, 12)); ax.axis("off")
    def box(x,y,w,h,text):
        rect = patches.FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.02,rounding_size=8",
                                      edgecolor="black", facecolor="#f2eaf7")
        ax.add_patch(rect); ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=11)
    def arrow(x1,y1,x2,y2):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle="->", lw=1.8))
    W,H=0.72,0.10; x=0.14; y0=0.90; dy=0.12
    box(x,y0,W,H,"Audio (.wav)")
    box(x,y0-dy,W,H,f"Preprocess (mono, {SAMPLE_RATE/1000:.0f} kHz)")
    box(x,y0-2*dy,W,H,f"Features: {preproc_label}")
    box(x,y0-3*dy,W,H,"Batch Pad (T → T_max)")
    box(x,y0-4*dy,W,H,"Model: " + " & ".join(model_names))
    box(x,y0-5*dy,W,H,"Output logits → Softmax")
    box(x,y0-6*dy,W,H,"Metrics: Acc / Confusion / Per-class table")
    if extra_note:
        box(x,y0-7*dy,W,H,extra_note)
        arrow(x+W/2, y0-6*dy, x+W/2, y0-7*dy)
    for k in range(6):
        arrow(x+W/2, y0-k*dy, x+W/2, y0-(k+1)*dy)
    plt.tight_layout(); fname=f"flowchart_{tag}.png"
    plt.savefig(fname, dpi=160); plt.close(); print(f"[plot] Saved flowchart → {fname}")

# =========================
# Data, features, split
# =========================

def collate_pad(batch):
    """batch of (T, F), label -> (B, T_max, F), lengths, labels"""
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.int64)
    F = seqs[0].shape[1]; T_max = int(lengths.max()); B = len(seqs)
    padded = torch.zeros(B, T_max, F, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        t = s.shape[0]; padded[i, :t, :] = s
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels

def safe_load_wav(path):
    """Try torchaudio; if it fails, fall back to soundfile. Returns (C, T), sr."""
    try:
        wf, sr = torchaudio.load(path)  # (C, T)
        return wf, sr
    except Exception:
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
        wf = torch.from_numpy(data).transpose(0, 1).contiguous()
        return wf, sr

# =========================
# Feature extractors (plug-in style)
# =========================
class ExtractorMFCC20:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sr = sample_rate
        self.mfcc = T.MFCC(
            sample_rate=sample_rate, n_mfcc=20,
            melkwargs={"n_fft": 256, "hop_length": 128, "n_mels": 40, "center": True, "power": 2.0},
        )
    def __call__(self, wf_tensor_1xT, sr):
        if sr != self.sr:
            wf_tensor_1xT = T.Resample(orig_freq=sr, new_freq=self.sr)(wf_tensor_1xT) # (1, F, Tm) -> (Tm, F)
        mfcc = self.mfcc(wf_tensor_1xT).squeeze(0).transpose(0, 1).contiguous()  # torch.FloatTensor (T, 20)
        return mfcc

class ExtractorMFCC39:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sr = sample_rate
    def __call__(self, wf_tensor_1xT, sr):
        if sr != self.sr:
            wf_tensor_1xT = T.Resample(orig_freq=sr, new_freq=self.sr)(wf_tensor_1xT)
        y = wf_tensor_1xT.squeeze(0).cpu().numpy()
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13, n_fft=256, hop_length=128, center=True)
        d1 = librosa.feature.delta(mfcc, order=1); d2 = librosa.feature.delta(mfcc, order=2)
        feat = np.concatenate([mfcc, d1, d2], axis=0).T.astype(np.float32)  # (T, 39)
        return torch.from_numpy(feat)

def make_extractor(preproc):
    if preproc == "mfcc20": return ExtractorMFCC20(SAMPLE_RATE)
    if preproc == "mfcc39": return ExtractorMFCC39(SAMPLE_RATE)
    raise ValueError("PREPROC must be 'mfcc20' or 'mfcc39'")

# =========================
# Dataset
# =========================
class FSDDDataset(Dataset):
    def __init__(self, dataset_path, extractor):
        self.dataset_path = dataset_path
        self.files = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith(".wav")])
        if not self.files: raise RuntimeError(f"No .wav files under {dataset_path}")
        self.extractor = extractor
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        fname = self.files[idx]; path = os.path.join(self.dataset_path, fname)
        wf, sr = safe_load_wav(path)  # (C, T)
        if wf.shape[0] > 1: wf = wf.mean(dim=0, keepdim=True)
        x = self.extractor(wf, sr)  # (T, F)
        y = int(fname.split("_")[0])
        return x, y

def split_balanced_indices(files, val_frac=0.1, test_frac=0.1):
    by_label = {d: [] for d in range(NUM_CLASSES)}
    for i, fname in enumerate(files):
        lab = int(fname.split("_")[0]); by_label[lab].append(i)
    tr, va, te = [], [], []
    for d in range(NUM_CLASSES):
        idxs = by_label[d]; random.shuffle(idxs); n = len(idxs)
        n_train = int((1 - val_frac - test_frac) * n)
        n_val   = int(val_frac * n)
        tr += idxs[:n_train]; va += idxs[n_train:n_train+n_val]; te += idxs[n_train+n_val:]
    return tr, va, te

# =========================
# Parameter-budget helpers
# =========================
def max_hidden_under_budget_gru(D, bidirectional, bytes_per_param=4):
    """Return max H s.t. GRU layer params ≤ 36kB; params per direction = 3H(D+H+2)."""
    budget_params = LAYER_BUDGET_BYTES // int(bytes_per_param)
    best = 1
    for H in range(1, 1024):
        per_dir = 3*H*(D + H + 2)
        total = per_dir * (2 if bidirectional else 1)
        if total <= budget_params: best = H
        else: break
    return best

def max_hidden_under_budget_lstm(D, bidirectional, bytes_per_param=LAYER_BUDGET_BYTES/9216*4):
    """Return max H s.t. LSTM layer params ≤ 36kB; params per direction = 4H(D+H+2)."""
    budget_params = LAYER_BUDGET_BYTES // int(bytes_per_param)
    best = 1
    for H in range(1, 1024):
        per_dir = 4*H*(D + H + 2)
        total = per_dir * (2 if bidirectional else 1)
        if total <= budget_params: best = H
        else: break
    return best

# =========================
# Models
# =========================
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, num_classes=10, bidirectional=True, dropout=0.1):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)  # (layers*dirs, B, H)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        return self.fc(self.dropout(h))

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, num_classes=10, bidirectional=False, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, c_n) = self.lstm(packed)  # h_n: (layers*dirs, B, H)
        h = h_n[-1]
        return self.fc(self.dropout(h))

# =========================
# Task C: PoT utilities
# =========================
def _iter_params_by_scope(model, scope="all"):
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if scope == "rnn_only":
            if not (("gru" in name.lower()) or ("lstm" in name.lower()) or ("weight_ih" in name) or ("weight_hh" in name)):
                # skip non-recurrent params (e.g., fc.weight, biases)
                continue
        yield name, p

@torch.no_grad()
def snap_to_pot_(p, kmin, kmax, eps=1e-8):
    w = p.data; mask = w.ne(0)
    if not mask.any(): return
    mag = w.abs().clamp_min(eps)
    exp = mag.log2().round().clamp_(kmin, kmax)
    w.copy_(torch.where(mask, w.sign() * (2.0 ** exp), w))

@torch.no_grad()
def snap_to_pot_stochastic_(p, kmin, kmax, eps=1e-8):
    w = p.data; a = w.abs().clamp_min(eps)
    kf = a.log2(); kl = kf.floor(); kh = kl + 1
    ph = (kf - kl).clamp(0,1)
    pick_high = (torch.rand_like(ph) < ph).float()
    k = (kl*(1-pick_high) + kh*pick_high).clamp_(kmin, kmax)
    q = w.sign() * (2.0 ** k)
    w.copy_(q)

@torch.no_grad()
def snap_row_shared_exp_(W, kmin, kmax, eps=1e-8):
    # W: (out, in)
    sign = W.sign()
    A = W.abs().clamp_min(eps)
    k = A.log2().median(dim=1).values.round().clamp_(kmin, kmax)  # per-row exponent
    Q = sign * (2.0 ** k.view(-1,1))
    W.copy_(Q)

@torch.no_grad()
def snap_to_apot2_(p, kmin, kmax, eps=1e-8):
    w = p.data; s = w.sign(); a = w.abs().clamp_min(eps)
    k1 = a.log2().round().clamp_(kmin, kmax)
    r  = (a - (2.0 ** k1)).clamp_min(0)
    k2 = r.clamp_min(eps).log2().round().clamp_(kmin, kmax)
    q  = s * ((2.0 ** k1) + (2.0 ** k2))
    w.copy_(q)

@torch.no_grad()
def project_model(model, approach, scope, alpha, kmin, kmax):
    for name, p in _iter_params_by_scope(model, scope):
        if approach == "ema":
            # EMA blend toward nearest PoT
            w = p.data; mask = w.ne(0)
            if not mask.any(): continue
            mag = w.abs().clamp_min(1e-8)
            exp = mag.log2().round().clamp_(kmin, kmax)
            w_pot = w.sign() * (2.0 ** exp)
            p.data = torch.where(mask, alpha * w + (1.0 - alpha) * w_pot, w)
        elif approach == "snap_epoch" or approach == "snap_step":
            snap_to_pot_(p, kmin, kmax)
        elif approach == "stochastic":
            snap_to_pot_stochastic_(p, kmin, kmax)
        elif approach == "row_shared":
            if p.ndim == 2: snap_row_shared_exp_(p, kmin, kmax)
        elif approach == "apot2":
            snap_to_apot2_(p, kmin, kmax)
        elif approach == "mixed_pot":
            # apply only to recurrent weights; scope already restricts; keep fc/bias float
            snap_to_pot_(p, kmin, kmax)
        elif approach == "inq":
            # handled by INQ controller; skip here
            pass

# INQ controller
class INQController:
    def __init__(self, model, scope, stages, kmin, kmax):
        self.model = model; self.scope = scope
        self.stages = stages  # list of (frac, epochs)
        self.stage_idx = 0
        self.masks = None  # tensor masks for frozen/snap areas
        self.kmin = kmin; self.kmax = kmax
    def _params(self):
        return [(n,p) for n,p in _iter_params_by_scope(self.model, self.scope) if p.ndim >= 1]
    def _build_mask(self, frac):
        masks = {}
        for n,p in self._params():
            w = p.data.view(-1)
            k = int(len(w) * frac)
            if k <= 0:
                mask = torch.zeros_like(w, dtype=torch.bool)
            else:
                idx = w.abs().topk(k).indices
                mask = torch.zeros_like(w, dtype=torch.bool); mask[idx] = True
            masks[n] = mask.view_as(p.data)
        self.masks = masks
    @torch.no_grad()
    def snap_and_freeze(self):
        for n,p in self._params():
            mask = self.masks[n]
            # snap masked positions to PoT and freeze by zeroing future grads there
            tmp = p.data.clone()
            snap_to_pot_(tmp, self.kmin, self.kmax)
            p.data = torch.where(mask, tmp, p.data)
            def _hook(grad, mask=mask):
                return grad.masked_fill(mask, 0)
            if hasattr(p, "_inq_hook"): p._inq_hook.remove()
            p._inq_hook = p.register_hook(_hook)
    def maybe_step_stage(self, epoch):
        # sum of previous epochs to decide stage transition
        pass
    def run_stages(self, current_epoch, local_epoch_in_stage):
        # Called each epoch: if entering a new stage, build mask & snap
        # We track stage transitions externally in train() for simplicity.
        pass

# =========================
# Train / Eval (with AMP & non_blocking)
# =========================
def evaluate(model, loader, criterion, device_override=None):
    dev = device_override if device_override is not None else device
    model.eval(); total_loss=0.0; total_acc=0.0; n=0
    with torch.no_grad():
        for xb, lengths, yb in loader:
            xb = xb.to(dev, non_blocking=(dev.type=="cuda"))
            lengths = lengths.to(dev, non_blocking=(dev.type=="cuda"))
            yb = yb.to(dev, non_blocking=(dev.type=="cuda"))
            with torch.amp.autocast("cuda", enabled=(dev.type=="cuda")):
                logits = model(xb, lengths)
                loss = criterion(logits, yb)
            acc = (logits.argmax(dim=1) == yb).float().mean().item()
            b = xb.size(0)
            total_loss += loss.item()*b; total_acc += acc*b; n += b
    return total_loss/max(n,1), total_acc/max(n,1)


def train(model, train_loader, val_loader, epochs, lr, weight_decay, clip_norm, patience, recorder=None, task="A"):
    # For Task C we use two-phase schedule + different LR (fine-tune)
    if task == "C":
        weight_decay = 0.0  # avoid pushing w->0 in PoT regime

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))

    # INQ bookkeeping
    inq = None
    if task == "C" and C_APPROACH == "inq":
        inq = INQController(model, scope=C_TARGET_SCOPE, stages=C_INQ_STAGES,
                            kmin=C_CLAMP_EXP_MIN, kmax=C_CLAMP_EXP_MAX)
        # Prepare stage counters
        stage_ptr = 0
        stage_epoch_left = C_INQ_STAGES[0][1] if C_INQ_STAGES else 0
        # initialize first stage mask/snap
        if C_INQ_STAGES:
            frac, _ = C_INQ_STAGES[0]
            inq._build_mask(frac)
            inq.snap_and_freeze()

    best_val = float("inf"); best_state=None; patience_left=patience
    global_step = 0

    for ep in range(1, epochs+1):
        model.train(); run_loss=0.0; run_acc=0.0; seen=0

        # enter PoT fine-tune phase (LR drop)
        if task == "C" and ep == C_PROJ_START_EPOCH:
            for pg in optimizer.param_groups:
                pg["lr"] = C_LR_FINE_TUNE

        for xb, lengths, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
                logits = model(xb, lengths)
                if task == "C" and KD_ENABLE and (kd_teacher is not None):
                    kd_teacher.eval()
                    with torch.no_grad():
                        t_logits = kd_teacher(xb, lengths)
                    ce = criterion(logits, yb)
                    kd = F.kl_div(
                        F.log_softmax(logits / KD_TAU, dim=1),
                        F.softmax(t_logits / KD_TAU, dim=1),
                        reduction="batchmean"
                    ) * (KD_TAU * KD_TAU)
                    loss = ce + KD_ALPHA * kd
                else:
                    loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            if clip_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer); scaler.update()

            # C: per-step projection
            global_step += 1
            if task == "C" and ep >= C_PROJ_START_EPOCH and C_PROJ_EVERY_N_STEPS and C_APPROACH in {"snap_step", "ema",
                                                                                                    "stochastic",
                                                                                                    "row_shared",
                                                                                                    "apot2",
                                                                                                    "mixed_pot"}:
                if global_step % C_PROJ_EVERY_N_STEPS == 0:
                    project_model(model, approach=C_APPROACH, scope=C_TARGET_SCOPE,
                                  alpha=C_EMA_ALPHA, kmin=C_CLAMP_EXP_MIN, kmax=C_CLAMP_EXP_MAX)

            b = xb.size(0)
            run_loss += loss.item()*b
            run_acc  += (logits.argmax(dim=1) == yb).float().sum().item()
            seen     += b

        # End-of-epoch projection for Task C (recommended default)
        if task == "C" and ep >= C_PROJ_START_EPOCH and C_PROJ_EVERY_N_STEPS is None and C_APPROACH in {"ema",
                                                                                                        "snap_epoch",
                                                                                                        "stochastic",
                                                                                                        "row_shared",
                                                                                                        "apot2",
                                                                                                        "mixed_pot"}:
            project_model(model, approach=C_APPROACH, scope=C_TARGET_SCOPE,
                          alpha=C_EMA_ALPHA, kmin=C_CLAMP_EXP_MIN, kmax=C_CLAMP_EXP_MAX)

        # INQ stage management (after each epoch)
        if inq is not None and C_INQ_STAGES:
            stage_epoch_left -= 1
            if stage_epoch_left <= 0 and (stage_ptr + 1) < len(C_INQ_STAGES):
                stage_ptr += 1
                frac, n_ep = C_INQ_STAGES[stage_ptr]
                inq._build_mask(frac)
                inq.snap_and_freeze()
                stage_epoch_left = n_ep

        tr_loss = run_loss/seen; tr_acc = run_acc/seen
        va_loss, va_acc = evaluate(model, val_loader, nn.CrossEntropyLoss())
        print(f"Epoch {ep:02d}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
        if recorder is not None: recorder.add(ep, tr_loss, tr_acc, va_loss, va_acc)

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping."); break

    if best_state is not None:
        model.load_state_dict(best_state)

# =========================
# Quantization helper for B2
# =========================
def quantize_dynamic_int8(model):
    """Dynamic quantization to int8 for GRU/LSTM/Linear (inference, CPU-only)."""
    from torch.ao.quantization import quantize_dynamic
    model = model.to("cpu").eval()  # <-- move to CPU before quantizing
    qmodel = quantize_dynamic(
        model, {nn.GRU, nn.LSTM, nn.Linear}, dtype=torch.qint8
    )
    return qmodel

def add_qat_to_linear_head(model):
    """
    Attaches a QAT qconfig to the Linear head only.
    Returns a model ready for QAT prepare/convert.
    """
    qconfig = get_default_qat_qconfig("fbgemm")
    # clone module refs we want to QAT
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc.qconfig = qconfig
    else:
        print("[qat] No Linear head found; skipping QAT attachment.")
    return model

def train_qat_linear_head(
    model, train_loader, val_loader, epochs_qat=6, base_lr=5e-4, clip_norm=1.0
):
    """
    Short fine-tune while fake-quant is active on the Linear head.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))
    best_val = float("inf"); best_state = None

    for ep in range(1, epochs_qat+1):
        model.train(); run_loss = 0.0; seen = 0
        for xb, lengths, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
                logits = model(xb, lengths)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            if clip_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer); scaler.update()

            run_loss += loss.item() * xb.size(0); seen += xb.size(0)

        va_loss, _ = evaluate(model, val_loader, criterion)
        print(f"[QAT] Epoch {ep:02d}/{epochs_qat} | train loss {run_loss/seen:.4f} | val loss {va_loss:.4f}")
        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

def run_experiment(
    dataset_path: str,
    task: str,
    model: str,            # "gru" | "lstm" | "both"
    preproc: str,          # "mfcc20" | "mfcc39"
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    clip_norm: float = CLIP_NORM,
    patience: int = PATIENCE,
    num_workers: int = NUM_WORKERS,
    c_params: dict | None = None,
    teacher_model: nn.Module | None = None,  # for KD in Task C
    B2_QAT_ENABLE: bool = False,
    B2_QAT_EPOCHS: int = 6,
):
    """
    Runs one experiment with the given configuration:
      - builds dataset & loaders for `preproc`
      - trains model(s) respecting `task` constraints (incl. B2 quantized eval on CPU)
      - creates the same plots as main (training curves, confusion matrix, metrics table, flowchart)
      - returns a dict of metrics and artifact paths
    """

    # Task-C override support
    if c_params is None: c_params = {}
    saved_c = {}
    if task == "C":
        for k in ["C_APPROACH", "C_TARGET_SCOPE", "C_PROJ_START_EPOCH", "C_PROJ_EVERY_N_STEPS",
                  "C_EMA_ALPHA", "C_LR_FINE_TUNE", "C_CLAMP_EXP_MIN", "C_CLAMP_EXP_MAX",
                  "C_INQ_STAGES", "KD_ENABLE", "KD_TAU", "KD_ALPHA"]:
            if k in c_params:
                try:
                    saved_c[k] = globals()[k]
                except KeyError:
                    continue
                globals()[k] = c_params[k]

    # --- Build dataset for the requested preproc
    extractor = make_extractor(preproc)
    full_ds = FSDDDataset(dataset_path, extractor=extractor)
    tr_idx, va_idx, te_idx = split_balanced_indices(full_ds.files, val_frac=0.1, test_frac=0.1)
    train_ds, val_ds, test_ds = Subset(full_ds, tr_idx), Subset(full_ds, va_idx), Subset(full_ds, te_idx)

    # --- DataLoaders (respect GPU-friendly args)
    loader_args = dict(
        batch_size=batch_size, collate_fn=collate_pad,
        num_workers=num_workers, pin_memory=(device.type=="cuda"),
        persistent_workers=(device.type=="cuda" and num_workers>0)
    )

    # g = torch.Generator(); g.manual_seed(SEED)    # stabilize + possibly squeeze more accuracy
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_args)#, generator=g, worker_init_fn=lambda wid: np.random.seed(SEED + wid))
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_args)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_args)

    # --- For report: preview MFCC image
    if len(train_ds) > 0:
        x0, _ = train_ds[0]
        plot_mfcc_image(x0, tag=f"{task}_{preproc}_preview", show=False)
        input_dim = x0.shape[1]
    else:
        input_dim = (20 if preproc=="mfcc20" else 39)

    # --- Choose topology under constraints
    def choose_topology(model_name, D):
        if task == "A":
            if model_name == "gru":
                return H_GRU_BASE, BIDIR_GRU_A
            else:
                return H_LSTM_BASE, BIDIR_LSTM_A
        else:
            bidir = False
            if task == "B2" and B2_EXPAND_TO_INT8_BUDGET:
                bytes_per_param = 1
            else:
                bytes_per_param = 4
            if model_name == "gru":
                H = max_hidden_under_budget_gru(D, bidirectional=bidir, bytes_per_param=bytes_per_param)
                return max(8, H), bidir
            else:
                H = max_hidden_under_budget_lstm(D, bidirectional=bidir, bytes_per_param=bytes_per_param)
                return max(8, H), bidir

    def plot_hidden_contribution_time_occlusion(
            model, sample_mfcc, sample_label, tag, win=8, stride=4, device_override=None
    ):
        """
        Occlude small time windows in MFCC and measure drop in the correct-class logit.
        Saves: hidden_contrib_<tag>.png
        """
        dev = device_override if device_override is not None else device
        model.eval()

        # Prepare baseline logit on the full sequence
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
            x = sample_mfcc.unsqueeze(0).to(dev)  # (1, T, F)
            lengths = torch.tensor([sample_mfcc.shape[0]], device=dev)
            logits_full = model(x, lengths)  # (1, C)
            c = int(sample_label)
            base = logits_full[0, c].item()

        T = sample_mfcc.shape[0]
        scores, centers = [], []
        for t0 in range(0, T, stride):
            t1 = min(T, t0 + win)
            x_occ = sample_mfcc.clone()
            x_occ[t0:t1, :] = 0.0  # zero a window
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                x_occ = x_occ.unsqueeze(0).to(dev)
                lengths = torch.tensor([T], device=dev)
                logits = model(x_occ, lengths)
                drop = base - logits[0, c].item()  # positive ⇒ hurts correct class
            scores.append(max(0.0, drop))
            centers.append((t0 + t1) / 2.0)

        # Plot
        fig, ax = plt.subplots(figsize=(7.0, 2.6))
        ax.plot(centers, scores)
        ax.set_title(f"Hidden-state contribution via time occlusion — {tag}")
        ax.set_xlabel("Time (frames)")
        ax.set_ylabel("Δ logit (correct class)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fname = f"hidden_contrib_{tag}.png"
        plt.savefig(fname, dpi=160)
        plt.close()
        print(f"[plot] Saved hidden-state contribution → {fname}")

    # --- Single-model runner (returns metrics + paths)
    def run_one(model_name: str):
        H, bidir = choose_topology(model_name, input_dim)
        tag = f"{task}_{model_name}_{preproc}_H{H}{'_bi' if bidir else ''}"

        # Build model
        if model_name == "gru":
            net = GRUClassifier(input_dim=input_dim, hidden_dim=H,
                                num_classes=NUM_CLASSES, bidirectional=bidir, dropout=0.1).to(device)
        else:
            net = LSTMClassifier(input_dim=input_dim, hidden_dim=H,
                                 num_classes=NUM_CLASSES, bidirectional=bidir, dropout=0.1).to(device)
        print(net)

        # silence cudnn GRU warning / trim jitter on CUDA
        if hasattr(net, "gru"):
            try:
                net.gru.flatten_parameters()
            except Exception:
                pass

        # Hidden-state contribution artifact (puts the model in eval mode internally)
        try:
            sx, sy = train_ds[0]
            plot_hidden_contribution_time_occlusion(
                net, sx, sy, tag=f"{tag}_occl", win=8, stride=4, device_override=device
            )
        except Exception as e:
            print(f"[warn] hidden contribution plot skipped: {e}")

        # Train
        rec = TrainRecorder()
        kd_teacher = None
        if task == "C" and KD_ENABLE and (teacher_model is not None):
            kd_teacher = teacher_model.to(device).eval()
        train(net, train_loader, val_loader, epochs, lr, weight_decay, clip_norm, patience,
              recorder=rec, task=task)
        plot_training_curves(rec, tag=tag, show=False); training_curves_path = f"training_curves_{tag}.png"

        # B2 quantized eval on CPU
        eval_model, eval_device, extra_note = net, device, ""
        if task == "B2":
            eval_model = quantize_dynamic_int8(net)      # CPU int8 model
            eval_device = torch.device("cpu")
            extra_note = "Dynamic int8 quantized (CPU inference)"

        # Hidden-state contribution artifact on one train sample (story-telling figure)
        try:
            sx, sy = train_ds[0]  # any representative sample works
            plot_hidden_contribution_time_occlusion(
                net, sx, sy, tag=f"{tag}_occl", win=8, stride=4, device_override=device
            )
        except Exception as e:
            print(f"[warn] hidden contribution plot skipped: {e}")

        # Evaluate + predictions
        crit = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate(eval_model, test_loader, crit, device_override=eval_device)
        y_true, y_pred = collect_predictions(eval_model, test_loader, device_override=eval_device)

        #  overall metrics (lean)
        overall = overall_metrics(y_true, y_pred)  # {"accuracy": ..., "f1_macro": ...}

        #  latency (GPU for A/B1/C, CPU for B2 int8)
        lat_mean_ms, lat_p95_ms = measure_latency(eval_model, test_loader, eval_device, warmup_batches=3,
                                                  measure_batches=10)

        # --- Print metrics-summary for this model/task
        print_overall_result(tag, overall, lat_mean_ms, lat_p95_ms)

        plot_confusion_counts(y_true, y_pred, tag=tag, class_names=[str(i) for i in range(NUM_CLASSES)], show=False)
        cm_path = f"confusion_matrix_{tag}.png"

        # # Per-class metrics
        # per_class = compute_per_class_metrics(y_true, y_pred)

        return {
            "name": model_name,
            "tag": tag,
            "hidden": H,
            "bidirectional": bidir,
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "metrics": {
                "accuracy": float(overall["acc"]),
                "f1_macro": float(overall["f1_macro"]),
            },
            "latency_ms_per_sample": {
                "mean": lat_mean_ms,
                "p95": lat_p95_ms,
            },
            "paths": {
                "training_curves": training_curves_path,
                "confusion_matrix": cm_path,
            },
            "note": extra_note,
        }


    # --- Run requested model(s)
    results = []
    if model == "both":
        results.append(run_one("gru"))
        results.append(run_one("lstm"))
    elif model in ("gru","lstm"):
        results.append(run_one(model))
    else:
        raise ValueError("model must be 'gru', 'lstm', or 'both'")


    # --- Return a compact summary
    summary = {
        "task": task,
        "preproc": preproc,
        "dataset_sizes": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "results": results,

    }
    return summary

# === helpers (no external deps beyond sklearn if available) ===
def overall_metrics(y_true, y_pred):
    """
    Returns dict with acc, precision/recall/f1 for micro and macro averaging.
    If scikit-learn is unavailable, falls back to micro ~ accuracy and zeros for macro.
    """
    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        acc = accuracy_score(y_true, y_pred)
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
    except Exception:
        # Fallback without sklearn (single-label multiclass: micro ≈ accuracy)
        import numpy as _np
        y_true = _np.asarray(y_true); y_pred = _np.asarray(y_pred)
        acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
        p_micro = r_micro = f1_micro = acc
        p_macro = r_macro = f1_macro = 0.0

    return dict(
        acc=acc,
        precision_micro=p_micro, recall_micro=r_micro, f1_micro=f1_micro,
        precision_macro=p_macro, recall_macro=r_macro, f1_macro=f1_macro,
    )

def _latency_placeholders():
    # Until run_experiment returns eval handles for timing
    return {"mean": float("nan"), "p95": float("nan")}

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Task A
    expA_mfcc20 = run_experiment(
        dataset_path=DATASET_PATH,
        task="A",
        model="both",
        preproc="mfcc20",
        epochs=20,
        batch_size=64
    )

    print("\n>>> Summary:", {
        "task": expA_mfcc20["task"],
        "preproc": expA_mfcc20["preproc"],
        "dataset_sizes": expA_mfcc20["dataset_sizes"],
    })
    for r in expA_mfcc20["results"]:
        print(f"- {r['tag']}: acc={r['metrics']['accuracy']:.4f}, "
              f"f1_macro={r['metrics']['f1_macro']:.4f}, "
              f"lat_mean={r['latency_ms_per_sample']['mean']:.2f} ms, "
              f"p95={r['latency_ms_per_sample']['p95']:.2f} ms")

    expA_mfcc39 = run_experiment(
        dataset_path=DATASET_PATH,
        task="A",
        model="both",
        preproc="mfcc39",
        epochs=20,
        batch_size=64
    )

    print("\n>>> Summary:", {
        "task": expA_mfcc39["task"],
        "preproc": expA_mfcc39["preproc"],
        "dataset_sizes": expA_mfcc39["dataset_sizes"],
    })
    for r in expA_mfcc39["results"]:
        print(f"- {r['tag']}: acc={r['metrics']['accuracy']:.4f}, "
              f"f1_macro={r['metrics']['f1_macro']:.4f}, "
              f"lat_mean={r['latency_ms_per_sample']['mean']:.2f} ms, "
              f"p95={r['latency_ms_per_sample']['p95']:.2f} ms")

    # Task B1 — 36 kB per-layer budget, GRU only
    expB1 = run_experiment(
        dataset_path=DATASET_PATH,
        task="B1",
        model="gru",  # set "both" if you want GRU+LSTM under B1
        preproc="mfcc20",
        epochs=20,
        batch_size=64
    )

    print("\n>>> B1 summary:", {
        "task": expB1["task"],
        "preproc": expB1["preproc"],
        "dataset_sizes": expB1["dataset_sizes"],
    })
    for r in expB1["results"]:
        print(f"- {r['tag']}: acc={r['metrics']['accuracy']:.4f}, "
              f"f1_macro={r['metrics']['f1_macro']:.4f}, "
              f"lat_mean={r['latency_ms_per_sample']['mean']:.2f} ms, "
              f"p95={r['latency_ms_per_sample']['p95']:.2f} ms")

    # Task B2 — dynamic INT8 (CPU eval), GRU only
    expB2 = run_experiment(
        dataset_path=DATASET_PATH,
        task="B2",
        model="gru",
        preproc="mfcc20",
        epochs=20,
        batch_size=64
    )

    print("\n>>> B2 summary:", {
        "task": expB2["task"],
        "preproc": expB2["preproc"],
        "dataset_sizes": expB2["dataset_sizes"],
    })
    for r in expB2["results"]:
        print(f"- {r['tag']}: acc={r['metrics']['accuracy']:.4f}, "
              f"f1_macro={r['metrics']['f1_macro']:.4f}, "
              f"lat_mean={r['latency_ms_per_sample']['mean']:.2f} ms, "
              f"p95={r['latency_ms_per_sample']['p95']:.2f} ms")
    print("\n>>> B2 GRU int8 acc:", expB2["results"][0]["test_acc"])

    # Task C
    # 3a) EMA projection 
    exp3a = run_experiment(
        dataset_path=DATASET_PATH, task="C", model="gru", preproc="mfcc20",
        epochs=35, batch_size=64, lr=1e-3, weight_decay=0.0,
        c_params=dict(
            C_APPROACH="ema", C_TARGET_SCOPE="rnn_only",
            C_PROJ_START_EPOCH=9, C_PROJ_EVERY_N_STEPS=None, C_EMA_ALPHA=0.85,
            C_LR_FINE_TUNE=2e-4, C_CLAMP_EXP_MIN=-8, C_CLAMP_EXP_MAX=8,
        ),
        teacher_model=None,
    )
    print("\n>>> C-ema summary:", {
        "task": exp3a["task"], "preproc": exp3a["preproc"], "dataset_sizes": exp3a["dataset_sizes"],
    })
    for r in exp3a["results"]:
        print(f"- {r['tag']}: acc={r['metrics']['accuracy']:.4f}, "
              f"f1_macro={r['metrics']['f1_macro']:.4f}, "
              f"lat_mean={r['latency_ms_per_sample']['mean']:.2f} ms, "
              f"p95={r['latency_ms_per_sample']['p95']:.2f} ms")
        print(f">>> C-ema {r['name']} acc:", r["test_acc"])

    # 3b) One-shot snap each epoch
    exp3b = run_experiment(
        dataset_path=DATASET_PATH, task="C", model="gru", preproc="mfcc20",
        epochs=30, lr=1e-3, weight_decay=0.0,
        c_params=dict(C_APPROACH="snap_epoch", C_TARGET_SCOPE="all", C_PROJ_START_EPOCH=6),
    )
    print("\n>>> C-snap_epoch summary:", {
        "task": exp3b["task"], "preproc": exp3b["preproc"], "dataset_sizes": exp3b["dataset_sizes"],
    })
    for r in exp3b["results"]:
        print(f"- {r['tag']}: acc={r['metrics']['accuracy']:.4f}, "
              f"f1_macro={r['metrics']['f1_macro']:.4f}, "
              f"lat_mean={r['latency_ms_per_sample']['mean']:.2f} ms, "
              f"p95={r['latency_ms_per_sample']['p95']:.2f} ms")
        print(f">>> C-snap_epoch {r['name']} acc:", r["test_acc"])

    # 3c) Stochastic rounding
    exp3c = run_experiment(
        dataset_path=DATASET_PATH, task="C", model="gru", preproc="mfcc20",
        epochs=30, lr=1e-3, weight_decay=0.0,
        c_params=dict(C_APPROACH="stochastic", C_TARGET_SCOPE="all", C_PROJ_START_EPOCH=6),
    )
    print("\n>>> C-stochastic summary:", {
        "task": exp3c["task"], "preproc": exp3c["preproc"], "dataset_sizes": exp3c["dataset_sizes"],
    })
    for r in exp3c["results"]:
        print(f"- {r['tag']}: acc={r['metrics']['accuracy']:.4f}, "
              f"f1_macro={r['metrics']['f1_macro']:.4f}, "
              f"lat_mean={r['latency_ms_per_sample']['mean']:.2f} ms, "
              f"p95={r['latency_ms_per_sample']['p95']:.2f} ms")
        print(f">>> C-stochastic {r['name']} acc:", r["test_acc"])

    # 3d) Mixed-PoT: only RNN weights PoT (fc/bias float)
    exp3d = run_experiment(
        dataset_path=DATASET_PATH, task="C", model="gru", preproc="mfcc20",
        epochs=30, lr=1e-3, weight_decay=0.0,
        c_params=dict(C_APPROACH="mixed_pot", C_TARGET_SCOPE="rnn_only", C_PROJ_START_EPOCH=6),
    )
    print("\n>>> C-mixed_pot summary:", {
        "task": exp3d["task"], "preproc": exp3d["preproc"], "dataset_sizes": exp3d["dataset_sizes"],
    })
    for r in exp3d["results"]:
        print(f"- {r['tag']}: acc={r['metrics']['accuracy']:.4f}, "
              f"f1_macro={r['metrics']['f1_macro']:.4f}, "
              f"lat_mean={r['latency_ms_per_sample']['mean']:.2f} ms, "
              f"p95={r['latency_ms_per_sample']['p95']:.2f} ms")
        print(f">>> C-mixed_pot {r['name']} acc:", r["test_acc"])

    # 3e) Row-shared exponent
    exp3e = run_experiment(
        dataset_path=DATASET_PATH, task="C", model="gru", preproc="mfcc20",
        epochs=30, lr=1e-3, weight_decay=0.0,
        c_params=dict(C_APPROACH="row_shared", C_TARGET_SCOPE="rnn_only", C_PROJ_START_EPOCH=6),
    )
    print("\n>>> C-row_shared summary:", {
        "task": exp3e["task"], "preproc": exp3e["preproc"], "dataset_sizes": exp3e["dataset_sizes"],
    })
    for r in exp3e["results"]:
        print(f"- {r['tag']}: acc={r['metrics']['accuracy']:.4f}, "
              f"f1_macro={r['metrics']['f1_macro']:.4f}, "
              f"lat_mean={r['latency_ms_per_sample']['mean']:.2f} ms, "
              f"p95={r['latency_ms_per_sample']['p95']:.2f} ms")
        print(f">>> C-row_shared {r['name']} acc:", r["test_acc"])

    # 3f) APoT-2
    exp3f = run_experiment(
        dataset_path=DATASET_PATH, task="C", model="gru", preproc="mfcc20",
        epochs=30, lr=1e-3, weight_decay=0.0,
        c_params=dict(C_APPROACH="apot2", C_TARGET_SCOPE="rnn_only", C_PROJ_START_EPOCH=6),
    )
    print("\n>>> C-apot2 summary:", {
        "task": exp3f["task"], "preproc": exp3f["preproc"], "dataset_sizes": exp3f["dataset_sizes"],
    })
    for r in exp3f["results"]:
        print(f"- {r['tag']}: acc={r['metrics']['accuracy']:.4f}, "
              f"f1_macro={r['metrics']['f1_macro']:.4f}, "
              f"lat_mean={r['latency_ms_per_sample']['mean']:.2f} ms, "
              f"p95={r['latency_ms_per_sample']['p95']:.2f} ms")
        print(f">>> C-apot2 {r['name']} acc:", r["test_acc"])

    # 3g) INQ staged snapping
    exp3g = run_experiment(
        dataset_path=DATASET_PATH, task="C", model="gru", preproc="mfcc20",
        epochs=sum(ep for _, ep in C_INQ_STAGES) + 5, lr=1e-3, weight_decay=0.0,
        c_params=dict(
            C_APPROACH="inq", C_TARGET_SCOPE="all", C_PROJ_START_EPOCH=1,
            C_INQ_STAGES=[(0.3, 4), (0.6, 4), (1.0, 6)]
        ),
    )
    print("\n>>> C-inq summary:", {
        "task": exp3g["task"], "preproc": exp3g["preproc"], "dataset_sizes": exp3g["dataset_sizes"],
    })
    for r in exp3g["results"]:
        print(f"- {r['tag']}: acc={r['metrics']['accuracy']:.4f}, "
              f"f1_macro={r['metrics']['f1_macro']:.4f}, "
              f"lat_mean={r['latency_ms_per_sample']['mean']:.2f} ms, "
              f"p95={r['latency_ms_per_sample']['p95']:.2f} ms")
        print(f">>> C-inq {r['name']} acc:", r["test_acc"])
