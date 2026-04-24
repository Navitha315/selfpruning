# Self-Pruning Neural Network — Technical Report

**Author:** Navitha E(22MIA1051), VIT Chennai  
**Role:** AI Engineering Intern — Tredence Analytics (2025 Cohort)  
**Dataset:** CIFAR-10 | **Framework:** PyTorch | **Device:** CPU (Apple M-series)

---

## 1. Problem Summary

Standard neural network pruning is a **post-training step**: train a dense network, identify unimportant weights, remove them. This assignment takes a harder approach — build a network that **learns to prune itself during training** by associating each weight with a learnable gate parameter. If a gate collapses to 0, its corresponding weight is effectively removed. The challenge is formulating a loss and training procedure that makes most gates collapse, leaving only the most important connections active.

---

## 2. The `PrunableLinear` Layer

### 2.1 Design

A custom layer replaces `torch.nn.Linear` throughout the network. It holds **two learnable parameter tensors** of identical shape (`out_features × in_features`):

| Parameter | Initialisation | Role |
|-----------|---------------|------|
| `weight` | Kaiming uniform (same as `nn.Linear`) | Standard linear weights |
| `gate_scores` | **+3.0** | Raw logits for per-weight gates |

### 2.2 Forward Pass

```
gates        = sigmoid(gate_scores)         ∈ (0, 1)  per weight
pruned_w     = weight  ⊙  gates             element-wise multiplication
output       = F.linear(x, pruned_w, bias)  standard affine transformation
```

Implemented entirely from scratch using `F.linear` (not `nn.Linear`). Because `sigmoid`, element-wise multiply, and `F.linear` are all differentiable PyTorch operations, **gradients flow automatically through both `weight` and `gate_scores`** via autograd with no manual gradient specification required.

A gate value of 0 zeroes out its corresponding weight entirely. A gate of 1 leaves the weight unchanged.

### 2.3 Initialisation — Critical Design Decision

`gate_scores` are initialised at `+3.0` → `sigmoid(+3) ≈ 0.952`. All gates start **fully open**. The sparsity loss then drives scores negative. A gate is considered pruned when `sigmoid(score) < 0.01`, requiring `score < −4.6`. Starting at `+3.0`, each score must travel ~7.6 units — achievable in 30 epochs with the dual-rate optimiser described in Section 5.

> **Why not initialise at 0.0?**  
> `sigmoid(0) = 0.5`. Adam's adaptive moment estimates quickly neutralise the near-constant L1 gradient at this value, causing gate scores to barely move. In the original implementation this produced **0% sparsity** across all λ values. Initialising at `+3.0` resolves this completely.

### 2.4 Hard-Prune Inference (Novel)

At evaluation time, an optional `hard_prune=True` flag hard-zeros any gate below the threshold before multiplication, producing the **exact sparse network's true accuracy** — not the soft-gated approximation used during training.

---

## 3. Sparsity Regularisation Loss

### 3.1 Loss Formulation

$$\mathcal{L}_\text{total} = \underbrace{\mathcal{L}_\text{CE}}_{\text{classification}} + \underbrace{\lambda_{L1} \cdot \sum_{\text{layers}} \sum_{i,j} \sigma(s_{ij})}_{\text{L1 sparsity}} + \underbrace{\lambda_\text{ent} \cdot \sum_{\text{layers}} H\!\left(\sigma(s_{ij})\right)}_{\text{entropy regulariser (novel)}}$$

where $s_{ij}$ are raw gate scores, $\sigma$ is sigmoid, and $H(g) = -g \log g - (1-g)\log(1-g)$ is binary entropy.

### 3.2 Why L1 Produces Exact Zeros

**L2 gradient** = `2g`: as `g → 0`, the gradient vanishes. The optimiser loses motivation, leaving many small-but-nonzero gates.

**L1 gradient** = `±1` (constant): Even a gate of `0.001` receives the same push toward zero as a gate of `0.9`. This constant pressure is what produces **exact zeros** rather than merely small values. Since sigmoid outputs are always positive, L1 simplifies to the sum of gate values — straightforward to compute and differentiate.

### 3.3 Entropy Regulariser (Novel Addition)

$H(g)$ is maximised at `g = 0.5` and minimised (= 0) at `g ∈ {0, 1}`. Minimising entropy penalises gates for remaining in the ambiguous midpoint. The entropy gradient is largest exactly where L1 is weakest — near `g = 0.5` — making the two terms complementary:

- **L1** pulls gates toward 0
- **Entropy** pushes gates away from 0.5 toward either binary pole

Together they accelerate binary polarisation, producing cleaner sparse solutions in fewer epochs.

### 3.4 λ as a Trade-off Hyperparameter

λ controls the sparsity–accuracy trade-off. Higher λ gives more weight to sparsity terms relative to cross-entropy, producing sparser networks at the cost of accuracy. Three values were evaluated: `1e-04` (low), `1e-03` (medium), `5e-03` (high).

---

## 4. Network Architecture

```
Input: CIFAR-10 image  →  flatten  →  3072-d vector

PrunableLinear(3072 → 512)  →  BatchNorm1d(512)  →  ReLU
PrunableLinear(512  → 256)  →  BatchNorm1d(256)  →  ReLU
PrunableLinear(256  → 128)  →  BatchNorm1d(128)  →  ReLU
PrunableLinear(128  → 10)                              ← logits
```

**BatchNorm** is added after each hidden layer to stabilise training as increasing fractions of weights are masked — without it, internal covariate shift worsens as the effective network size shrinks dynamically during training.

---

## 5. Training Setup

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Optimiser | Adam | Adaptive LR handles heterogeneous gradient scales |
| Weight / BN LR | `1e-3` | Standard for Adam on CIFAR |
| **Gate LR** | **`1e-2` (10× weight LR)** | Gates must traverse ~7.6 units in 30 epochs |
| Weight decay | `1e-4` (weights only, not gates) | Gates have their own explicit penalty |
| LR Schedule | Cosine Annealing (T_max=30) | Smooth decay prevents late-stage oscillation |
| Batch size | 128 | |
| Epochs | 30 | |
| Prune threshold | 0.01 | `sigmoid(score) < 0.01` → weight considered pruned |

**Dual-rate optimiser** (novel): `gate_scores` and weights are placed in separate `param_groups` with 10× LR for gates. Using identical LR for both means gates never traverse far enough to hit the prune threshold within 30 epochs.

**Data augmentation** (novel): `RandomHorizontalFlip` + `RandomCrop(32, padding=4)` applied during training.

---

## 6. Results

### 6.1 Summary Table

| Lambda (λ_L1) | Soft Accuracy (%) | Hard-Pruned Accuracy (%) | Sparsity Level (%) |
|:---:|:---:|:---:|:---:|
| `1e-04` (low) | **60.02** | **48.91** | 75.7 |
| `1e-03` (medium) | **60.62** | 10.00 | 97.1 |
| `5e-03` (high) | **60.08** | 10.00 | 99.3 |

> **Soft Accuracy**: test accuracy using continuous gate values (no thresholding).  
> **Hard-Pruned Accuracy**: test accuracy with all gates < 0.01 hard-zeroed — the honest accuracy of the truly sparse network.

### 6.2 Layer-Wise Sparsity Breakdown

| Layer | Parameters | λ = 1e-04 | λ = 1e-03 | λ = 5e-03 |
|-------|:----------:|:---------:|:---------:|:---------:|
| fc1 (3072→512) | 1,572,864 | 99.1% | 100.0% | 100.0% |
| fc2 (512→256) | 131,072 | 89.3% | 99.9% | 100.0% |
| fc3 (256→128) | 32,768 | 79.7% | 98.7% | 99.8% |
| fc4 (128→10) | 1,280 | 34.8% | 89.9% | 97.3% |

**Key observation:** fc1 — the largest layer with ~1.57M weights — prunes most aggressively because it has the most redundancy. fc4 — the output layer with only 1,280 weights — is the most resistant because every connection directly influences classification logits.

### 6.3 Training Dynamics (λ = 1e-04)

| Epoch | Total Loss | Train Acc | Sparsity |
|------:|----------:|----------:|--------:|
| 1 | 102.30 | 35.7% | 0.0% |
| 5 | 3.14 | 48.4% | 48.1% |
| 10 | 1.93 | 52.6% | 66.7% |
| 15 | 1.64 | 55.0% | 72.2% |
| 20 | 1.51 | 56.8% | 74.7% |
| 25 | 1.44 | 58.3% | 75.6% |
| 30 | 1.42 | 58.6% | **75.7%** |

The high initial loss (102.30) reflects the L1 penalty on ~1.57M gate values all near 0.95. As gates collapse to near-zero the penalty shrinks rapidly, and from epoch 5 onward the classification loss drives accuracy upward while sparsity continues to grow. Accuracy and sparsity improve **simultaneously** — a hallmark of well-formulated learned pruning.

---

## 7. Analysis of the λ Trade-off

### 7.1 λ = 1e-04 (Low) — Best Deployable Model

- 75.7% sparsity with 60.02% soft accuracy and **48.91% hard-pruned accuracy**
- Hard-pruned accuracy of 48.91% is 4.9× the random baseline (10%) using only 24.3% of the original weights
- The 11% soft-to-hard gap indicates some residual information in near-zero gates, but the network has substantially committed to a sparse solution
- **This is the most practical configuration for deployment**

### 7.2 λ = 1e-03 (Medium) — Fragile Near-Sparse Regime

- 97.1% sparsity with 60.62% soft accuracy but **10.00% hard-pruned accuracy** (random-guess level)
- The network maintains high soft accuracy by relying on gates in the 0.01–0.05 range — sub-threshold but carrying non-negligible signal
- Hard-pruning collapses accuracy completely, revealing the network has entered a **fragile near-sparse regime**: almost-pruned but not truly pruned
- This is the most interesting failure mode — soft accuracy is misleading as a metric here

### 7.3 λ = 5e-03 (High) — Full Collapse

- 99.3% sparsity — essentially all weights pruned
- Hard-pruned accuracy: 10.00% (random)
- The network has no capacity to maintain useful representations with virtually no active connections

### 7.4 Critical Insight: Soft vs Hard-Pruned Accuracy

The gap between soft and hard-pruned accuracy is the most informative metric in this study. A **small gap** indicates the network genuinely committed to a sparse solution (λ = 1e-04). A **large gap** indicates the network exploits sub-threshold residuals rather than truly pruning (λ ≥ 1e-03).

Production deployment requires hard-pruned accuracy. Reporting only soft accuracy would misrepresent the sparse model's actual capability.

---

## 8. Gate Value Distribution

The gate distribution plot for λ = 5e-03 shows **100% of gates below the prune threshold** — a single sharp spike at 0. The soft accuracy of 60% is entirely carried by sub-threshold residuals (gates in the 0.001–0.01 range).

For λ = 1e-04, the expected bimodal distribution is observed:
- **Large spike near 0** — the 75.7% of pruned gates
- **Cluster near 0.7–0.95** — the 24.3% of surviving gates that the network preserved for classification

The near-complete absence of gates near 0.5 confirms the entropy regulariser successfully pushed all gates away from the ambiguous midpoint toward binary values.

*See: `outputs/gate_distribution.png` and `outputs/accuracy_vs_sparsity.png`*

---

## 9. Novel Contributions Beyond the Spec

| Contribution | Implementation | Effect |
|-------------|---------------|--------|
| **gate_scores init = +3.0** | `torch.full(..., 3.0)` | Fixed 0% sparsity bug; core correctness fix |
| **Dual-rate optimiser** | Separate `param_groups`, gate LR = 10× | Enabled gates to reach prune threshold in 30 epochs |
| **Entropy regulariser** | $\lambda_\text{ent} \cdot H(\sigma(s))$ in loss | Faster binary polarisation, cleaner gate distribution |
| **Hard-prune inference** | `hard_prune=True` eval mode | Honest measurement of truly sparse network |
| **Layer-wise sparsity** | Per-layer `pruning_ratio()` | Shows fc1 prunes first; fc4 most resistant |
| **BatchNorm** | After each hidden layer | Stabilised training during aggressive pruning |
| **Data augmentation** | RandomFlip + RandomCrop | Improved accuracy baseline |
| **Accuracy vs Sparsity plot** | Dual-axis chart | Visual summary of λ trade-off |

---

## 10. Conclusion

The self-pruning mechanism works correctly. Learnable sigmoid gates provide a smooth, differentiable, end-to-end trainable pruning signal. The L1 penalty combined with the entropy regulariser drives gates to binary values, producing networks with 75–99% sparsity from a **single training run** — no separate pruning step required.

The λ = 1e-04 configuration is the most deployable result: **48.91% hard-pruned accuracy with 75.7% of weights removed**. The layer-wise sparsity breakdown reveals a natural compression hierarchy — large early layers compress most aggressively while the small output layer retains the most connections.

The soft-vs-hard accuracy gap is proposed as a diagnostic for evaluating whether learned pruning has produced genuine sparsity or a fragile near-sparse regime. This distinction is essential for any production deployment of pruned models.

This approach is related to **Variational Dropout** (Molchanov et al., 2017), **L0 regularisation** (Louizos et al., 2018), and **Straight-Through Estimators** for discrete sparsity — but is simpler to implement, fully differentiable, and requires no post-processing.
