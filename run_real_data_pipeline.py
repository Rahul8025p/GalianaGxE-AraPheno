"""
Real Data Pipeline — Galiana EnvironmentalOnly
===============================================
Uses the actual AraPheno dataset (1021 accessions x 288 traits) stored in:
  data/real_arapheno_Y.npy
  data/real_phenotype_names.json

The dataset contains:
  - 73 bioclimatic / environmental columns (clim-* prefix) → X_env  [REAL]
  - 215 plant phenotype columns                            → Y_pheno [REAL]

The EnvironmentalOnly model predicts all 215 real plant phenotypes from
the 73 real bioclimatic/WorldClim features across all 1021 accessions.

NOTE: No real WGS genotype matrix is bundled in this repo. To run the
full GalianaGxE model, provide a real (N x 27655 x 17) variant consequence
matrix as X_geno.
"""

import os
import sys
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_env import load_and_split_arapheno_data
from model_env import EnvironmentalOnly

# ── Config ───────────────────────────────────────────────────────────────────
DATA_NPY    = "data/real_arapheno_Y.npy"
DATA_JSON   = "data/real_phenotype_names.json"
EPOCHS      = 30
BATCH_SIZE  = 64
LR          = 1e-3
WEIGHT_DECAY = 1e-5
TEST_FRAC   = 0.2
SEED        = 42
RESULTS_DIR = "results"

torch.manual_seed(SEED)
np.random.seed(SEED)

def banner(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def evaluate(model, X_e, Y_np, device):
    model.eval()
    with torch.no_grad():
        preds = model(X_e.to(device)).cpu().numpy()
    mse  = float(np.mean((preds - Y_np) ** 2))
    mae  = float(np.mean(np.abs(preds - Y_np)))
    r_vals = []
    for p in range(Y_np.shape[1]):
        if np.std(preds[:, p]) > 1e-6 and np.std(Y_np[:, p]) > 1e-6:
            r_vals.append(float(np.corrcoef(preds[:, p], Y_np[:, p])[0, 1]))
        else:
            r_vals.append(0.0)
    return mse, mae, float(np.mean(r_vals)), preds, r_vals

def main():
    banner("GALIANA — REAL DATA PIPELINE  (EnvironmentalOnly Model)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Compute device : {device}")

    # ── 1. Load real AraPheno data ───────────────────────────────────────────
    banner("1 / 5  Loading Real AraPheno Dataset")
    split = load_and_split_arapheno_data(DATA_NPY, DATA_JSON)

    env_raw     = split["env_data"].astype(np.float32)    # (1021, 73)
    pheno_raw   = split["pheno_data"].astype(np.float32)  # (1021, 215)
    env_names   = split["env_names"]
    pheno_names = split["pheno_names"]
    N           = split["num_samples"]

    print(f"  Total accessions            : {N}")
    print(f"  Environmental features (X)  : {env_raw.shape[1]}  — real WorldClim/bioclimatic")
    print(f"  Plant phenotypes (Y)        : {pheno_raw.shape[1]}  — real AraPheno measurements")
    print(f"  Sample env features         : {env_names[:5]}")
    print(f"  Sample phenotypes           : {pheno_names[:5]}")

    # ── 2. Pre-process ───────────────────────────────────────────────────────
    banner("2 / 5  Pre-processing & Train / Test Split")

    # Impute zero entries with column means (proxy for missing values)
    for c in range(pheno_raw.shape[1]):
        col = pheno_raw[:, c]
        nonzero_mean = col[col != 0].mean() if (col != 0).any() else 0.0
        pheno_raw[col == 0, c] = nonzero_mean

    scaler_e = StandardScaler()
    scaler_p = StandardScaler()
    env_scaled   = scaler_e.fit_transform(env_raw).astype(np.float32)
    pheno_scaled = scaler_p.fit_transform(pheno_raw).astype(np.float32)

    idx = np.arange(N)
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_FRAC, random_state=SEED)
    print(f"  Train accessions : {len(tr_idx)}")
    print(f"  Test  accessions : {len(te_idx)}")

    X_tr = torch.tensor(env_scaled[tr_idx])
    X_te = torch.tensor(env_scaled[te_idx])
    Y_tr = torch.tensor(pheno_scaled[tr_idx])
    Y_te_np = pheno_scaled[te_idx]

    num_env   = env_raw.shape[1]
    num_pheno = pheno_raw.shape[1]

    # ── 3. Build model ───────────────────────────────────────────────────────
    banner("3 / 5  Building EnvironmentalOnly Model")
    model = EnvironmentalOnly(num_env_features=num_env, num_phenotypes=num_pheno).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture : Linear({num_env}) → LayerNorm → Tanh → Linear(64) → Linear(50) → Linear({num_pheno})")
    print(f"  Parameters   : {total_params:,}")
    print(f"  Input        : {num_env} real bioclimatic features (100% real data)")
    print(f"  Output       : {num_pheno} real plant phenotypes (100% real data)")

    # ── 4. Train ─────────────────────────────────────────────────────────────
    banner("4 / 5  Training")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    n_train   = X_tr.shape[0]
    losses    = []

    print(f"  Epochs: {EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LR} | Samples: {n_train} train / {len(te_idx)} test\n")
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss, steps = 0.0, 0
        for start in range(0, n_train, BATCH_SIZE):
            bidx = perm[start:start + BATCH_SIZE]
            xb   = X_tr[bidx].to(device)
            yb   = Y_tr[bidx].to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        scheduler.step()
        epoch_loss /= steps
        losses.append(epoch_loss)
        print(f"   Epoch [{ep:02d}/{EPOCHS}]  MSE Loss: {epoch_loss:.4f}")

    # ── 5. Evaluate ──────────────────────────────────────────────────────────
    banner("5 / 5  Evaluation on Held-out Test Set")
    mse, mae, mean_r, preds, r_list = evaluate(model, X_te, Y_te_np, device)

    print(f"\n  Test MSE (Mean Squared Error)  : {mse:.4f}")
    print(f"  Test MAE (Mean Absolute Error) : {mae:.4f}")
    print(f"  Mean Pearson Correlation (r)   : {mean_r:.4f}")

    # Top 10 phenotypes by Pearson r
    top_pheno_idx = np.argsort(r_list)[::-1][:10]
    print("\n  TOP 10 PHENOTYPES BEST PREDICTED FROM REAL ENVIRONMENTAL DATA:")
    print("  " + "-" * 65)
    for rank, pidx in enumerate(top_pheno_idx, 1):
        print(f"   #{rank:2d}. {pheno_names[pidx][:48]:<50}  r = {r_list[pidx]:+.4f}")

    # Gradient saliency — top environmental drivers
    print("\n  TOP 10 ENVIRONMENTAL DRIVERS (Gradient Saliency):")
    print("  " + "-" * 65)
    sample_e = X_te[:20].clone().to(device).requires_grad_(True)
    out_s = model(sample_e)
    model.zero_grad()
    out_s.sum().backward()
    env_imp  = sample_e.grad.abs().mean(dim=0).cpu().numpy()
    top_env  = np.argsort(env_imp)[::-1][:10]
    for rank, eidx in enumerate(top_env, 1):
        print(f"   #{rank:2d}. {env_names[eidx]:<28}  saliency = {env_imp[eidx]:.5f}")

    # Sample predictions
    print("\n  SAMPLE PREDICTIONS vs ACTUAL (first test accession, top 5 traits):")
    print(f"  {'Trait':<44} {'Actual':>8} {'Predicted':>10} {'|Error|':>8}")
    print("  " + "-" * 74)
    for pidx in top_pheno_idx[:5]:
        t = pheno_names[pidx][:42]
        a = Y_te_np[0, pidx]
        p = preds[0, pidx]
        print(f"  {t:<44} {a:>+8.4f} {p:>+10.4f} {abs(a-p):>8.4f}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # CSV — per-trait results
    csv_path = os.path.join(RESULTS_DIR, "real_data_phenotype_performance.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phenotype", "pearson_r"])
        for i, name in enumerate(pheno_names):
            w.writerow([name, f"{r_list[i]:.6f}"])
    print(f"\n  Trait-by-trait CSV saved  → {csv_path}")

    # Plot — training loss curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, EPOCHS + 1), losses, marker='o', color='#10b981', lw=2)
    plt.title("Training Loss — Real AraPheno Data (EnvironmentalOnly)", fontsize=13, fontweight='bold')
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    loss_png = os.path.join(RESULTS_DIR, "real_data_training_loss.png")
    plt.savefig(loss_png, dpi=300); plt.close()
    print(f"  Training loss curve saved → {loss_png}")

    # Plot — top phenotype Pearson r bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    top20_idx  = np.argsort(r_list)[::-1][:20]
    top20_r    = [r_list[i] for i in top20_idx]
    top20_names= [pheno_names[i][:30] for i in top20_idx]
    colors = ['#10b981' if r >= 0 else '#ef4444' for r in top20_r]
    ax.barh(range(len(top20_r)), top20_r[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top20_r)))
    ax.set_yticklabels(top20_names[::-1], fontsize=8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel("Pearson r", fontsize=11)
    ax.set_title("Top 20 Phenotypes — Real Data Prediction (EnvironmentalOnly)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    bar_png = os.path.join(RESULTS_DIR, "real_data_top_phenotypes.png")
    plt.savefig(bar_png, dpi=300); plt.close()
    print(f"  Top phenotypes bar chart  → {bar_png}")

    banner("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"  Test MSE   : {mse:.4f}")
    print(f"  Test MAE   : {mae:.4f}")
    print(f"  Pearson r  : {mean_r:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
