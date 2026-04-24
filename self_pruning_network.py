"""
Self-Pruning Neural Network — Tredence AI Engineering Internship Case Study

Submitted by  : Navitha E (22MIA1051), VIT Chennai

ADDITIONS (beyond the specifications mentioned in assignment)
-----------------------------------
1. Dual-rate Optimiser      — gate_scores use 10× LR vs weights; they need
                              to traverse a much larger range in the same epochs.
2. Gate Entropy Regulariser — penalises gates near 0.5, accelerating binary
                              polarisation toward 0 or 1.  Works synergistically
                              with L1.
3. Hard-prune Inference     — at eval time, gates < threshold are hard-zeroed,
                              showing the exact sparse network's accuracy.
4. LayerWise Sparsity       — per-layer sparsity printed after each run.
5. Accuracy vs Sparsity plot — dual-axis tradeoff chart across all λ values.
6. Data Augmentation        — random flip + crop; improves test accuracy baseline.
7. BatchNorm between layers — stabilises training when many weights are masked.
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context   # Mac SSL fix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os


# 1.  PrunableLinear  (fixed + novel)

class PrunableLinear(nn.Module):
    """
    Linear layer with per-weight learnable sigmoid gates.

    Training forward:
        gates        = sigmoid(gate_scores)      ∈ (0, 1)
        pruned_w     = weight ⊙ gates            element-wise mask
        output       = F.linear(x, pruned_w, bias)

    Inference forward (hard_prune=True):
        gates below PRUNE_THR are hard-zeroed before multiplication,
        producing an exactly sparse computation graph.

    Gradient flow:
        Both weight and gate_scores receive gradients automatically
        through sigmoid → element-wise mul → F.linear (all differentiable).
    """

    PRUNE_THR  = 1e-2   
    GATE_INIT  = 3.0     

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), self.GATE_INIT)
        )
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

    def forward(self, x: torch.Tensor, hard_prune: bool = False) -> torch.Tensor:
        g = self.gates()
        if hard_prune:

            with torch.no_grad():
                mask = (g >= self.PRUNE_THR).float()
            g = g * mask
        pruned_w = self.weight * g
        return F.linear(x, pruned_w, self.bias)

    # Loss components 

    def l1_gate_loss(self) -> torch.Tensor:
        """L1 norm of gate values — constant gradient drives gates to 0."""
        return self.gates().sum()

    def entropy_gate_loss(self) -> torch.Tensor:
        """
        Novel: Entropy penalty on gate distribution.
        H(g) = -g·log(g) - (1-g)·log(1-g) is maximised at g=0.5.
        Minimising entropy pushes gates toward binary {0,1}, working
        synergistically with L1 to accelerate polarisation.
        """
        g   = self.gates().clamp(1e-6, 1 - 1e-6)
        ent = -g * g.log() - (1 - g) * (1 - g).log()
        return ent.mean()

    # Metrics

    def pruning_ratio(self) -> float:
        with torch.no_grad():
            return (self.gates() < self.PRUNE_THR).float().mean().item()

# 2.  Self-Pruning Network
# ═════════════════════════════════════════════════════════════

class SelfPruningNet(nn.Module):
    """
    4-layer feed-forward net for CIFAR-10 (input: 3072-d flattened).
    Novel: BatchNorm after each hidden layer stabilises training
    when large fractions of weights are being masked.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = PrunableLinear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = PrunableLinear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = PrunableLinear(128, 10)

    def forward(self, x: torch.Tensor, hard_prune: bool = False) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x, hard_prune)))
        x = F.relu(self.bn2(self.fc2(x, hard_prune)))
        x = F.relu(self.bn3(self.fc3(x, hard_prune)))
        return self.fc4(x, hard_prune)

    def prunable_layers(self):
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def total_sparsity_loss(self, lam_l1: float, lam_ent: float) -> torch.Tensor:
        """
        Combined:
          lam_l1 × Σ L1(gates)     — primary sparsity driver
          lam_ent × Σ H(gates)     — novel binary polarisation term
        """
        l1_loss  = sum(l.l1_gate_loss()   for l in self.prunable_layers())
        ent_loss = sum(l.entropy_gate_loss() for l in self.prunable_layers())
        return lam_l1 * l1_loss + lam_ent * ent_loss

    def overall_pruning_ratio(self) -> float:
        return float(np.mean([l.pruning_ratio() for l in self.prunable_layers()]))

    def layerwise_sparsity(self) -> dict:
        return {
            f"fc{i+1}": f"{l.pruning_ratio()*100:.1f}%"
            for i, l in enumerate(self.prunable_layers())
        }

    def all_gate_values(self) -> np.ndarray:
        parts = []
        for l in self.prunable_layers():
            with torch.no_grad():
                parts.append(l.gates().cpu().numpy().flatten())
        return np.concatenate(parts)

    def param_groups(self, base_lr: float):
        """
        Novel: Dual-rate optimiser groups.
        gate_scores need 10× LR — they must travel from +3 to <−4.6
        (a range of ~8) while weights only need small adjustments.
        Using the same LR for both means gates move far too slowly.
        """
        gates, others = [], []
        for name, p in self.named_parameters():
            (gates if "gate_scores" in name else others).append(p)
        return [
            {"params": others, "lr": base_lr,        "weight_decay": 1e-4},
            {"params": gates,  "lr": base_lr * 10.0, "weight_decay": 0.0},
        ]


# ═══════════════════════════════════════════════════════════════
# 3.  Data Loaders
# ═══════════════════════════════════════════════════════════════

def get_cifar10_loaders(batch_size: int = 128, data_dir: str = "./data"):
    # Novel: augmentation improves accuracy on CPU runs
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    pin = torch.cuda.is_available()   # disable pin_memory on CPU/MPS
    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=pin)
    return train_loader, test_loader


# ═══════════════════════════════════════════════════════════════
# 4.  Training & Evaluation
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device, lam_l1, lam_ent):
    model.train()
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits      = model(images)
        cls_loss    = F.cross_entropy(logits, labels)
        sparse_loss = model.total_sparsity_loss(lam_l1, lam_ent)
        loss        = cls_loss + sparse_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device, hard_prune: bool = False):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds   = model(images, hard_prune=hard_prune).argmax(1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)
    return correct / total


def train_and_evaluate(lam_l1, lam_ent, epochs, device, train_loader, test_loader):
    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.param_groups(base_lr=1e-3))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n── λ_L1={lam_l1:.0e}  λ_ent={lam_ent:.0e} {'─'*38}")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, lam_l1, lam_ent
        )
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            sp = model.overall_pruning_ratio() * 100
            print(f"  Epoch {epoch:>2}/{epochs} | Loss: {train_loss:.3f} | "
                  f"Train Acc: {train_acc*100:.1f}% | Sparsity: {sp:.1f}%")

    soft_acc = evaluate(model, test_loader, device, hard_prune=False)
    hard_acc = evaluate(model, test_loader, device, hard_prune=True)
    sparsity = model.overall_pruning_ratio() * 100
    print(f"  ✓ Soft Acc: {soft_acc*100:.2f}%  |  Hard-Pruned Acc: {hard_acc*100:.2f}%  |  Sparsity: {sparsity:.1f}%")
    print(f"  Layer-wise: {model.layerwise_sparsity()}")
    return soft_acc, hard_acc, sparsity, model


# ═══════════════════════════════════════════════════════════════
# 5.  Plots
# ═══════════════════════════════════════════════════════════════

def plot_gate_distribution(model, lam_l1, save_path):
    vals = model.all_gate_values()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(vals, bins=120, color="#4C72B0", edgecolor="white", linewidth=0.2)
    ax.axvline(x=PrunableLinear.PRUNE_THR, color="crimson", linestyle="--",
               linewidth=1.5, label=f"Prune threshold ({PrunableLinear.PRUNE_THR})")
    pct = (vals < PrunableLinear.PRUNE_THR).sum() / len(vals) * 100
    ax.text(0.55, 0.82, f"Pruned gates: {pct:.1f}%", transform=ax.transAxes,
            fontsize=11, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
    ax.set_xlabel("Gate Value  σ(gate_score)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Gate Value Distribution  (λ_L1 = {lam_l1:.0e})", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → {save_path}")


def plot_tradeoff(results, save_path):
    """Novel: dual-axis accuracy vs sparsity chart."""
    lambdas   = [f"λ={r[0]:.0e}" for r in results]
    soft_accs = [r[1] * 100 for r in results]
    hard_accs = [r[2] * 100 for r in results]
    sparses   = [r[3]       for r in results]
    x = list(range(len(lambdas)))

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(x, soft_accs, "o-",  color="#1565C0", lw=2, label="Soft Accuracy (%)")
    ax1.plot(x, hard_accs, "s--", color="#0D47A1", lw=2, label="Hard-Pruned Acc (%)")
    ax2.bar(x, sparses, alpha=0.22, color="#E64A19", label="Sparsity (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(lambdas, fontsize=11)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=11, color="#1565C0")
    ax2.set_ylabel("Sparsity Level (%)", fontsize=11, color="#E64A19")
    ax1.set_title("Accuracy vs Sparsity Trade-off", fontsize=13)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → {save_path}")


# ═══════════════════════════════════════════════════════════════
# 6.  Main
# ═══════════════════════════════════════════════════════════════

def main():
    EPOCHS     = 30
    BATCH_SIZE = 128
    OUTPUT_DIR = "./outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Three λ_L1 settings; entropy penalty fixed at 1e-4 across all
    CONFIGS = [
        (1e-4, 1e-4),   # low   — minimal pruning
        (1e-3, 1e-4),   # med   — balanced
        (5e-3, 1e-4),   # high  — aggressive
    ]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*64}")
    print(f"  Self-Pruning Neural Network — Tredence Case Study (v2)")
    print(f"  Device: {DEVICE}  |  Epochs: {EPOCHS}")
    print(f"  Fixes : gate_scores init=+3, dual-rate LR for gates")
    print(f"  Novel : Entropy reg, Hard-prune eval, BN, Augmentation")
    print(f"{'='*64}")

    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)

    results    = []
    best_model = None
    best_lam   = None
    best_sp    = 0.0

    for lam_l1, lam_ent in CONFIGS:
        sa, ha, sp, model = train_and_evaluate(
            lam_l1, lam_ent, EPOCHS, DEVICE, train_loader, test_loader
        )
        results.append((lam_l1, sa, ha, sp, model))
        if sp > best_sp:
            best_sp, best_model, best_lam = sp, model, lam_l1

    # ── Summary table ────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"{'Lambda':<10} {'Soft Acc (%)':>14} {'Hard-Pruned (%)':>17} {'Sparsity (%)':>14}")
    print(f"{'-'*58}")
    for lam, sa, ha, sp, _ in results:
        print(f"{lam:<10.0e} {sa*100:>14.2f} {ha*100:>17.2f} {sp:>14.1f}")
    print(f"{'='*72}\n")

    # ── Plots ────────────────────────────────────────────────
    plot_gate_distribution(best_model, best_lam,
                           os.path.join(OUTPUT_DIR, "gate_distribution.png"))
    plot_tradeoff([(r[0], r[1], r[2], r[3]) for r in results],
                  os.path.join(OUTPUT_DIR, "accuracy_vs_sparsity.png"))

    # ── CSV ──────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
    with open(csv_path, "w") as f:
        f.write("Lambda,SoftAccuracy,HardPrunedAccuracy,SparsityLevel\n")
        for lam, sa, ha, sp, _ in results:
            f.write(f"{lam:.0e},{sa*100:.2f},{ha*100:.2f},{sp:.1f}\n")
    print(f"  → {csv_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
