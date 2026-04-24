# Self-Pruning Neural Network

**Tredence Analytics — AI Engineering Internship Case Study**  
**Submitted by :** Navitha E (22MIA1051) | VIT Chennai

A PyTorch implementation of a feed-forward neural network that **learns to prune its own weights during training** via learnable sigmoid gates and a combined L1 + entropy sparsity loss — no separate post-training pruning step required.

---

## Results

| Lambda (λ) | Soft Accuracy | Hard-Pruned Accuracy | Sparsity |
|:----------:|:-------------:|:--------------------:|:--------:|
| `1e-04` | 60.02% | **48.91%** | 75.7% |
| `1e-03` | 60.62% | 10.00% | 97.1% |
| `5e-03` | 60.08% | 10.00% | 99.3% |

Dataset: CIFAR-10 (10-class image classification) | Epochs: 30 | Device: CPU

---

## How It Works

Each weight in the network has a corresponding learnable **gate score** `s`. During the forward pass:

```
gate   = sigmoid(s)              ∈ (0, 1)
output = (weight ⊙ gate) @ x + bias
```

The total loss combines classification and sparsity:

```
L_total = CrossEntropy + λ_L1 × Σ sigmoid(s) + λ_ent × Σ H(sigmoid(s))
```

- **L1 term**: constant gradient pressure drives gates to exactly 0
- **Entropy term** (novel): pushes gates away from 0.5 toward binary values, accelerating polarisation

---

## Novel Contributions

Beyond the base assignment specification:

| Feature | Why |
|---------|-----|
| `gate_scores` init = **+3.0** (not 0.0) | Fixes 0% sparsity bug — see below |
| **Dual-rate optimiser** — gates use 10× LR | Gates must travel ~7.6 units; weights need small adjustments |
| **Entropy regulariser** on gates | Synergises with L1 for faster binary polarisation |
| **Hard-prune inference** mode | Honest evaluation: gates < 0.01 exactly zeroed at test time |
| **Layer-wise sparsity** report | Shows which layers compress most aggressively |
| **BatchNorm** after each hidden layer | Stabilises training as weights get masked |
| **Data augmentation** (flip + crop) | Improves accuracy baseline |
| **Accuracy vs Sparsity dual-axis plot** | Visual summary of the λ trade-off |

---

## The Bug I Fixed

The original implementation initialised `gate_scores = 0` → `sigmoid(0) = 0.5`. At this point Adam's adaptive moments neutralise the near-constant L1 gradient, so gates never move. Result: **0% sparsity** regardless of λ.

**Fix:** Initialise `gate_scores = +3.0` → `sigmoid(+3) ≈ 0.95`. Gates start fully open; L1 now has a clear direction to drive them negative. Problem solved in one line.

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/<your-username>/tredence-self-pruning-network
cd tredence-self-pruning-network

# 2. Install dependencies
pip install torch torchvision matplotlib numpy

# 3. Run
python3 self_pruning_network.py
```

CIFAR-10 (~170 MB) downloads automatically to `./data/`.  
Outputs saved to `./outputs/`:

```
outputs/
├── gate_distribution.png       # Gate value histogram (bimodal = success)
├── accuracy_vs_sparsity.png    # Dual-axis trade-off chart
└── results_summary.csv         # Lambda, SoftAcc, HardPrunedAcc, Sparsity
```

**Runtime:** ~25–35 min on CPU (MacBook M-series). Faster with CUDA — detected automatically.

---

## Mac SSL Fix

If you get `SSL: CERTIFICATE_VERIFY_FAILED` when downloading CIFAR-10:

```bash
/Applications/Python\ 3.13/Install\ Certificates.command
```

The script also includes `ssl._create_default_https_context = ssl._create_unverified_context` as a fallback.

---

## Project Structure

```
tredence-self-pruning-network/
├── self_pruning_network.py   # All code: model, training, evaluation, plots
├── REPORT.md                 # Technical report with analysis
├── README.md                 # This file
└── outputs/                  # Generated after running
    ├── gate_distribution.png
    ├── accuracy_vs_sparsity.png
    └── results_summary.csv
```

---

## Key Design Decisions

**Why sigmoid gates (not ReLU or hard threshold)?**  
Sigmoid is differentiable everywhere and bounded in (0, 1) — essential for gradient-based learning. Hard thresholds have zero gradient almost everywhere; ReLU is unbounded.

**Why L1 on gate values (not gate_scores)?**  
Penalising `sigmoid(s)` directly penalises the actual multiplicative effect on weights. Penalising raw scores would be less interpretable and would interact poorly with the sigmoid saturation regions.

**Why 10× LR for gate_scores?**  
A gate_score must travel from +3 to −4.6 (a range of 7.6) to go from fully open to pruned. Weights need only small adjustments around their initialised values. Using a shared LR either makes weights update too aggressively or keeps gates too slow.

**Why entropy + L1 (not L1 alone)?**  
L1 is weakest near `g = 0.5` (sigmoid gradient ≈ 0.25 there). Entropy gradient is largest at `g = 0.5`. They fill each other's blind spots, producing faster and cleaner binary polarisation.

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
numpy>=1.24.0
```

---

## License

MIT
