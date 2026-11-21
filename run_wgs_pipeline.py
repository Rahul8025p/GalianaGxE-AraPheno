"""
run_wgs_pipeline.py
====================
Full GalianaGxE pipeline on REAL data:
  - X_geno : real 1001 Genomes imputed SNP matrix   (N x 27655 x 17)  [REAL]
  - X_env  : real AraPheno bioclimatic features     (N x 73)           [REAL]
  - Y_pheno: real AraPheno plant phenotypes          (N x 215)          [REAL]

Run build_genotype_matrix.py first to generate data/real_genotype_X.npy
"""

import os, sys, json, csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_env import load_and_split_arapheno_data
from model_env import GalianaGxE, EnvironmentalOnly

# ── Config ───────────────────────────────────────────────────────────────────
DATA_NPY      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "real_arapheno_Y.npy")
DATA_JSON     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "real_phenotype_names.json")
GENO_NPY      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "real_wgs_genotype_100.npy")
if not os.path.exists(GENO_NPY):
    GENO_NPY  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "real_genotype_X.npy")
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EPOCHS        = 20
BATCH_SIZE    = 16
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
TEST_FRAC     = 0.2
SEED          = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

def banner(t):
    print("\n" + "="*80); print(f"  {t}"); print("="*80)

def pearson_r_per_trait(preds, targets):
    r_list = []
    for p in range(targets.shape[1]):
        if np.std(preds[:,p]) > 1e-6 and np.std(targets[:,p]) > 1e-6:
            r_list.append(float(np.corrcoef(preds[:,p], targets[:,p])[0,1]))
        else:
            r_list.append(0.0)
    return r_list

def evaluate_model(model, X_g_te, X_e_te, Y_te_np, device, use_geno=True):
    model.eval()
    with torch.no_grad():
        if use_geno:
            preds = model(X_g_te.to(device), X_e_te.to(device)).cpu().numpy()
        else:
            preds = model(X_e_te.to(device)).cpu().numpy()
    mse   = float(np.mean((preds - Y_te_np)**2))
    mae   = float(np.mean(np.abs(preds - Y_te_np)))
    r_list = pearson_r_per_trait(preds, Y_te_np)
    return mse, mae, float(np.mean(r_list)), preds, r_list

def train_loop(model, data_tuple, label, epochs, batch_size, lr, wd, device, use_geno=True):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    losses = []
    if use_geno:
        X_g_tr, X_e_tr, Y_tr = data_tuple
        N = X_g_tr.shape[0]
    else:
        X_e_tr, Y_tr = data_tuple
        N = X_e_tr.shape[0]

    print(f"\n  ── {label} ──")
    for ep in range(1, epochs+1):
        model.train()
        perm = torch.randperm(N)
        ep_loss, steps = 0.0, 0
        for start in range(0, N, batch_size):
            bidx = perm[start:start+batch_size]
            if use_geno:
                xg = X_g_tr[bidx].to(device)
                xe = X_e_tr[bidx].to(device)
                yb = Y_tr[bidx].to(device)
                pred = model(xg, xe)
            else:
                xe = X_e_tr[bidx].to(device)
                yb = Y_tr[bidx].to(device)
                pred = model(xe)
            loss = criterion(pred, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep_loss += loss.item(); steps += 1
        scheduler.step()
        ep_loss /= steps
        losses.append(ep_loss)
        print(f"   Epoch [{ep:02d}/{epochs}]  MSE Loss: {ep_loss:.4f}")
    return losses

def main():
    banner("GALIANA GxE — FULL REAL DATA PIPELINE (WGS + ENV + PHENOTYPE)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Compute device : {device}")

    # ── 1. Check genotype matrix ─────────────────────────────────────────────
    banner("1 / 6  Loading Genotype Data")
    if not os.path.exists(GENO_NPY):
        print(f"  ERROR: {GENO_NPY} not found!")
        print("  Please run:  python build_genotype_matrix.py  first.")
        sys.exit(1)

    X_geno_all = np.load(GENO_NPY)   # (N, 27655, 17) float32
    N_geno, num_genes, num_var = X_geno_all.shape
    print(f"  Genotype matrix loaded : {X_geno_all.shape}  ({X_geno_all.nbytes/1e6:.0f} MB)")
    print(f"  Accessions             : {N_geno}")
    print(f"  Genes                  : {num_genes:,}")
    print(f"  Variant bins           : {num_var}")

    # ── 2. Load real env + phenotype data ────────────────────────────────────
    banner("2 / 6  Loading AraPheno Environmental & Phenotype Data")
    split = load_and_split_arapheno_data(DATA_NPY, DATA_JSON)
    env_raw   = split["env_data"].astype(np.float32)[:N_geno]
    pheno_raw = split["pheno_data"].astype(np.float32)[:N_geno]
    env_names   = split["env_names"]
    pheno_names = split["pheno_names"]
    N = env_raw.shape[0]

    print(f"  Accessions (aligned)   : {N}")
    print(f"  Environmental features : {env_raw.shape[1]}  (real WorldClim)")
    print(f"  Plant phenotypes       : {pheno_raw.shape[1]}  (real AraPheno)")

    # Impute zero entries
    for c in range(pheno_raw.shape[1]):
        col = pheno_raw[:, c]
        nz_mean = col[col != 0].mean() if (col != 0).any() else 0.0
        pheno_raw[col == 0, c] = nz_mean

    scaler_e = StandardScaler()
    scaler_p = StandardScaler()
    env_scaled   = scaler_e.fit_transform(env_raw).astype(np.float32)
    pheno_scaled = scaler_p.fit_transform(pheno_raw).astype(np.float32)
    num_env   = env_raw.shape[1]
    num_pheno = pheno_raw.shape[1]

    # ── 3. Train / Test split ────────────────────────────────────────────────
    banner("3 / 6  Train / Test Split")
    idx = np.arange(N)
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_FRAC, random_state=SEED)
    print(f"  Train : {len(tr_idx)} accessions")
    print(f"  Test  : {len(te_idx)} accessions")

    X_g_tr = torch.tensor(X_geno_all[tr_idx])
    X_g_te = torch.tensor(X_geno_all[te_idx])
    X_e_tr = torch.tensor(env_scaled[tr_idx])
    X_e_te = torch.tensor(env_scaled[te_idx])
    Y_tr   = torch.tensor(pheno_scaled[tr_idx])
    Y_te_np = pheno_scaled[te_idx]

    # ── 4. Build models ──────────────────────────────────────────────────────
    banner("4 / 6  Building Models")
    model_gxe = GalianaGxE(
        num_genes=num_genes,
        num_variant_types=num_var,
        num_env_features=num_env,
        num_phenotypes=num_pheno
    ).to(device)
    model_env = EnvironmentalOnly(
        num_env_features=num_env,
        num_phenotypes=num_pheno
    ).to(device)

    p_gxe = sum(p.numel() for p in model_gxe.parameters() if p.requires_grad)
    p_env = sum(p.numel() for p in model_env.parameters() if p.requires_grad)
    print(f"  GalianaGxE        : {p_gxe:,} parameters  ← uses REAL genotype + env")
    print(f"  EnvironmentalOnly : {p_env:,} parameters  ← env-only baseline")

    # ── 5. Train ─────────────────────────────────────────────────────────────
    banner("5 / 6  Training")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    losses_env = train_loop(
        model_env, (X_e_tr, Y_tr),
        "EnvironmentalOnly (real env → real phenotypes)",
        EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, device, use_geno=False
    )
    losses_gxe = train_loop(
        model_gxe, (X_g_tr, X_e_tr, Y_tr),
        "GalianaGxE (REAL genotype + real env → real phenotypes)",
        EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, device, use_geno=True
    )

    # ── 6. Evaluate ──────────────────────────────────────────────────────────
    banner("6 / 6  Evaluation on Held-out Test Set")
    mse_e, mae_e, r_e, preds_e, r_list_e = evaluate_model(model_env, None, X_e_te, Y_te_np, device, use_geno=False)
    mse_g, mae_g, r_g, preds_g, r_list_g = evaluate_model(model_gxe, X_g_te, X_e_te, Y_te_np, device, use_geno=True)

    print(f"\n  {'Model':<40} {'MSE':>8} {'MAE':>8} {'Pearson r':>10}")
    print("  " + "-"*68)
    print(f"  {'EnvironmentalOnly (real env)':<40} {mse_e:>8.4f} {mae_e:>8.4f} {r_e:>10.4f}")
    print(f"  {'GalianaGxE (REAL geno + real env)':<40} {mse_g:>8.4f} {mae_g:>8.4f} {r_g:>10.4f}")

    # Top 10 phenotypes
    top_idx = np.argsort(r_list_g)[::-1][:10]
    print("\n  TOP 10 PHENOTYPES PREDICTED BY GalianaGxE (Real Genotype):")
    print("  " + "-"*65)
    for rank, pidx in enumerate(top_idx, 1):
        delta = r_list_g[pidx] - r_list_e[pidx]
        print(f"   #{rank:2d}. {pheno_names[pidx][:44]:<46}  r={r_list_g[pidx]:+.4f}  Δ={delta:+.4f}")

    # Env saliency
    print("\n  TOP 10 ENVIRONMENTAL DRIVERS (GxE model saliency):")
    sample_e = X_e_te[:20].clone().to(device).requires_grad_(True)
    sample_g = X_g_te[:20].clone().to(device)
    out_s = model_gxe(sample_g, sample_e)
    model_gxe.zero_grad(); out_s.sum().backward()
    env_imp = sample_e.grad.abs().mean(dim=0).cpu().numpy()
    for rank, eidx in enumerate(np.argsort(env_imp)[::-1][:10], 1):
        print(f"   #{rank:2d}. {env_names[eidx]:<28}  saliency={env_imp[eidx]:.5f}")

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "wgs_phenotype_performance.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phenotype", "pearson_r_envonly", "pearson_r_gxe", "delta_r"])
        for i, name in enumerate(pheno_names):
            w.writerow([name, f"{r_list_e[i]:.6f}", f"{r_list_g[i]:.6f}",
                        f"{r_list_g[i]-r_list_e[i]:.6f}"])
    print(f"\n  Trait results saved → {csv_path}")

    # Save plots
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(range(1, EPOCHS+1), losses_env, 'o-', color='#10b981', lw=2, label='EnvOnly')
    axes[0].plot(range(1, EPOCHS+1), losses_gxe, 's--', color='#1d4ed8', lw=2, label='GalianaGxE')
    axes[0].set_title("Training Loss — Real WGS + AraPheno Data", fontweight='bold')
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE"); axes[0].legend(); axes[0].grid(alpha=0.4)

    top20 = np.argsort(r_list_g)[::-1][:20]
    bars_g = [r_list_g[i] for i in top20]
    bars_e = [r_list_e[i] for i in top20]
    names20 = [pheno_names[i][:25] for i in top20]
    x = np.arange(len(top20))
    axes[1].barh(x+0.2, bars_g[::-1], 0.4, color='#1d4ed8', label='GalianaGxE (real geno)')
    axes[1].barh(x-0.2, bars_e[::-1], 0.4, color='#10b981', label='EnvOnly baseline')
    axes[1].set_yticks(x); axes[1].set_yticklabels(names20[::-1], fontsize=7)
    axes[1].set_xlabel("Pearson r"); axes[1].legend(fontsize=8)
    axes[1].set_title("Top 20 Phenotypes (Real WGS Model)", fontweight='bold')
    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "wgs_model_results.png")
    plt.savefig(png_path, dpi=300); plt.close()
    print(f"  Results plot saved  → {png_path}")

    banner("PIPELINE COMPLETE")
    print(f"  GalianaGxE (Real WGS) — MSE: {mse_g:.4f} | MAE: {mae_g:.4f} | Pearson r: {r_g:.4f}")
    print(f"  EnvironmentalOnly     — MSE: {mse_e:.4f} | MAE: {mae_e:.4f} | Pearson r: {r_e:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
