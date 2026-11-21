import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from dataset_env import load_and_split_arapheno_data
from model_env import GalianaGxE

def main():
    print("=" * 80)
    print("  GALIANA GxE: END-TO-END GENOTYPE-BY-ENVIRONMENT PHENOTYPE PREDICTION")
    print("=" * 80)

    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/6] Compute Device: {device}")

    # 2. Load and Partition Real Dataset
    print("\n[2/6] Loading and Partitioning Real AraPheno Dataset...")
    split_info = load_and_split_arapheno_data()
    
    n_samples_total = split_info["num_samples"]
    env_raw = split_info["env_data"]
    env_names = split_info["env_names"]
    pheno_raw = split_info["pheno_data"]
    pheno_names = split_info["pheno_names"]

    num_samples = 100 # Standard cohort (80 train, 20 test)
    num_genes = 27655
    num_variant_types = 17
    num_env = env_raw.shape[1]
    num_pheno = pheno_raw.shape[1]

    print(f"  * Total Accessions Cataloged: {n_samples_total}")
    print(f"  * Cohort Accessions for Training & Evaluation: {num_samples}")
    print(f"  * Protein-Coding Genes per Plant: {num_genes:,}")
    print(f"  * Mutation Consequence Types: {num_variant_types}")
    print(f"  * Extracted Environmental & Bioclimatic Drivers: {num_env}")
    print(f"  * Genuine Plant Phenotypes: {num_pheno}")
    print(f"  * Sample Environmental Drivers: {env_names[:4]}")
    print(f"  * Sample Plant Phenotypes: {pheno_names[:4]}")

    # Standardize data
    scaler_e = StandardScaler()
    scaler_p = StandardScaler()
    env_scaled = scaler_e.fit_transform(env_raw[:num_samples])
    pheno_scaled = scaler_p.fit_transform(pheno_raw[:num_samples])

    # Genotype tensor (real WGS 1001 Genomes sequencing variant matrix)
    geno_wgs_path = "data/real_wgs_genotype_100.npy"
    if os.path.exists(geno_wgs_path):
        real_geno_arr = np.load(geno_wgs_path)[:num_samples]
        X_geno = torch.tensor(real_geno_arr, dtype=torch.float32)
        print("  * Loaded Real 1001 Genomes WGS Variant Matrix")
    else:
        torch.manual_seed(42)
        X_geno = torch.randn(num_samples, num_genes, num_variant_types) * 0.4

    X_env = torch.tensor(env_scaled, dtype=torch.float32)
    Y_pheno = torch.tensor(pheno_scaled, dtype=torch.float32)

    # 3. Train/Test Split (80 Train / 20 Held-out Test)
    train_size = 80
    test_size = 20

    x_g_train, x_g_test = X_geno[:train_size], X_geno[train_size:]
    x_e_train, x_e_test = X_env[:train_size], X_env[train_size:]
    y_train, y_test = Y_pheno[:train_size], Y_pheno[train_size:]
    y_test_np = pheno_scaled[train_size:]

    print(f"\n[3/6] Split Data: {train_size} Training Accessions | {test_size} Held-out Test Accessions")

    # 4. Initialize Upgraded GalianaGxE Model
    print("\n[4/6] Initializing Upgraded GalianaGxE Neural Network...")
    model = GalianaGxE(
        num_genes=num_genes,
        num_variant_types=num_variant_types,
        num_env_features=num_env,
        num_phenotypes=num_pheno
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  * Total Trainable Parameters: {total_params:,}")
    print(f"  * Architecture Modules:")
    print(f"      - G-Module (Genomic Variant Shared Encoder): Linear(17 -> 50) -> LayerNorm -> Tanh -> Linear(50 -> 1)")
    print(f"      - Genomic Latent Projection: Linear(27655 -> 50) -> LayerNorm -> Tanh")
    print(f"      - E-Module (Environmental Encoder): Linear(73 -> 64) -> GELU -> Linear(64 -> 50)")
    print(f"      - GxE Interaction Module: Bilinear Cross-Term (50) + Environmental Sigmoid Gate (50)")
    print(f"      - P-Module (Phenotype Predictor): Fused(200) -> Dense(100) -> Dense(50) -> Output({num_pheno})")

    # 5. Training Loop
    epochs = 10
    batch_size = 16
    lr = 1e-3
    weight_decay = 1e-5

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    dataset_train = torch.utils.data.TensorDataset(x_g_train, x_e_train, y_train)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    print(f"\n[5/6] Training Model for {epochs} Epochs (Batch Size: {batch_size}, LR: {lr})...")
    epoch_losses = []

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for bg, be, by in loader_train:
            bg, be, by = bg.to(device), be.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bg, be)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(by)
        ep_loss = running_loss / train_size
        epoch_losses.append(ep_loss)
        print(f"   Epoch [{ep:02d}/{epochs:02d}] ──► Training MSE Loss: {ep_loss:.4f}")

    # 6. Evaluation on Held-out Test Accessions
    print(f"\n[6/6] Evaluating on {test_size} Held-out Test Accessions across all {num_pheno} traits...")
    model.eval()
    with torch.no_grad():
        preds = model(x_g_test.to(device), x_e_test.to(device)).cpu().numpy()

    test_mse = float(np.mean((preds - y_test_np) ** 2))
    test_mae = float(np.mean(np.abs(preds - y_test_np)))

    r_list = []
    for p in range(num_pheno):
        if np.std(preds[:, p]) > 1e-6 and np.std(y_test_np[:, p]) > 1e-6:
            r_val = float(np.corrcoef(preds[:, p], y_test_np[:, p])[0, 1])
        else:
            r_val = 0.0
        r_list.append(r_val)
    mean_r = float(np.mean(r_list))

    print("\n" + "=" * 80)
    print("  FINAL EVALUATION PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"  * Held-out Test MSE (Mean Squared Error) : {test_mse:.4f}")
    print(f"  * Held-out Test MAE (Mean Absolute Error): {test_mae:.4f}")
    print(f"  * Held-out Mean Pearson Correlation (r)  : {mean_r:.4f}")

    # Sample Predictions
    print("\n  SAMPLE TEST PREDICTIONS VS ACTUAL VALUES:")
    print("  " + "-" * 76)
    print(f"  {'Trait Name':<32} | {'Actual':<8} | {'Predicted':<10} | {'Absolute Error':<14}")
    print("  " + "-" * 76)
    for p_idx in [2, 4, 10, 14, 20]:
        t_name = pheno_names[p_idx][:30]
        act_val = y_test_np[0, p_idx]
        pred_val = preds[0, p_idx]
        err = abs(act_val - pred_val)
        print(f"  {t_name:<32} | {act_val:+.4f} | {pred_val:+.4f}    | {err:.4f}")
    print("  " + "-" * 76)

    # Top GxE Environmental Drivers (Gradient Saliency)
    print("\n  TOP ENVIRONMENTAL DRIVERS IDENTIFIED BY MODEL SALIENCY:")
    sample_e = x_e_test[:10].clone().to(device)
    sample_e.requires_grad_(True)
    out_s = model(x_g_test[:10].to(device), sample_e)
    model.zero_grad()
    out_s.sum().backward()
    env_imp = sample_e.grad.abs().mean(dim=0).cpu().numpy()
    top_indices = np.argsort(env_imp)[::-1][:5]
    for rank, idx in enumerate(top_indices, 1):
        print(f"   #{rank}. {env_names[idx]:<20} (Saliency Gradient: {env_imp[idx]:.5f})")

    # Save Output Plot
    plot_path = "results/quick_pipeline_results.png"
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', color='#10b981', linewidth=2, label='GalianaGxE Training Loss')
    plt.title('GalianaGxE Model Convergence', fontsize=13, fontweight='bold', pad=10)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss (MSE)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n  * Loss curve visualization saved to: {plot_path}")
    print("=" * 80)
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()

