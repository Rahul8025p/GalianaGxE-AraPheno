import os
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from dataset_env import load_and_split_arapheno_data
from model_env import GalianaGxE, GalianaEnvAdditive, GalianaGenomicOnly, EnvironmentalOnly

def run_environmental_upgrade_experiments(num_samples=100, epochs=20, batch_size=16):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    print("=" * 75)
    print("STARTING GENOTYPE-BY-ENVIRONMENT (GxE) UPGRADE EXPERIMENTS")
    print("=" * 75)

    # 1. Load Data
    split_info = load_and_split_arapheno_data()
    env_raw_all = split_info["env_data"]
    env_names = split_info["env_names"]
    pheno_raw_all = split_info["pheno_data"]
    pheno_names = split_info["pheno_names"]

    # Select top accessions (matching 100 accessions in original study)
    env_raw = env_raw_all[:num_samples]
    pheno_raw = pheno_raw_all[:num_samples]

    n_samples = len(env_raw)
    num_env = env_raw.shape[1]
    num_phenotypes = pheno_raw.shape[1]
    num_genes = 27655
    num_variant_types = 17

    print(f"Number of Arabidopsis accessions: {n_samples}")
    print(f"Bioclimatic / Environmental Features (WorldClim & Climate): {num_env}")
    print(f"Target Plant Phenotypic Traits: {num_phenotypes}")
    print(f"Number of Genes per Accession: {num_genes:,}")
    print(f"Variant Consequence Dimensions per Gene: {num_variant_types}")

    # Standardize Environmental Factors and Phenotypes
    scaler_env = StandardScaler()
    env_scaled = scaler_env.fit_transform(env_raw)
    env_tensor = torch.tensor(env_scaled, dtype=torch.float32)

    scaler_pheno = StandardScaler()
    pheno_scaled = scaler_pheno.fit_transform(pheno_raw)
    pheno_tensor = torch.tensor(pheno_scaled, dtype=torch.float32)

    # 2. Simulate biological genomic variants with G, E, and GxE effect components
    print("\nSynthesizing biological genomic variation with QTLs and GxE interaction...")
    torch.manual_seed(42)
    np.random.seed(42)

    X_raw = torch.randn(n_samples, num_genes, num_variant_types) * 0.4
    
    # Genetic QTL signals: 40 causal gene loci per phenotype
    for p_idx in range(num_phenotypes):
        causal_genes = np.random.choice(num_genes, 40, replace=False)
        for c_g in causal_genes:
            sig = pheno_tensor[:, p_idx].unsqueeze(1) * 0.35
            X_raw[:, c_g, :4] += sig.repeat(1, 4)

    # Inject GxE interaction: environmental factors modulate allele effects on climate-responsive traits
    for p_idx in range(num_phenotypes):
        trait_vals = pheno_scaled[:, p_idx]
        corrs = [np.corrcoef(trait_vals, env_scaled[:, e_i])[0, 1] for e_i in range(num_env)]
        top_env_idx = int(np.nanargmax(np.abs(corrs))) if len(corrs) > 0 else 0

        # GxE responsive loci
        gxe_genes = np.random.choice(num_genes, 20, replace=False)
        gxe_signal = (pheno_tensor[:, p_idx] * env_tensor[:, top_env_idx]).unsqueeze(1) * 0.28
        for g_locus in gxe_genes:
            X_raw[:, g_locus, 4:8] += gxe_signal.repeat(1, 4)

    # Scale genotype features
    X_flat = X_raw.view(-1, num_variant_types).numpy()
    scaler_x = StandardScaler()
    X_scaled_flat = scaler_x.fit_transform(X_flat)
    X_geno_tensor = torch.tensor(X_scaled_flat, dtype=torch.float32).view(n_samples, num_genes, num_variant_types)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device}")

    # 3. Model Architecture Comparison
    models_to_test = ["geno_only", "env_only", "additive", "gxe"]
    model_names_display = {
        "geno_only": "Genotype-Only (Galiana Baseline)",
        "env_only": "Environment-Only (Bioclim Baseline)",
        "additive": "G + E Additive Fusion",
        "gxe": "Full GxE Interaction Model"
    }

    # 80/20 Train / Held-out Test Split
    train_size = int(0.8 * n_samples)
    test_size = n_samples - train_size
    train_idx = np.arange(train_size)
    test_idx = np.arange(train_size, n_samples)

    x_g_train, x_g_test = X_geno_tensor[train_idx], X_geno_tensor[test_idx]
    x_e_train, x_e_test = env_tensor[train_idx], env_tensor[test_idx]
    y_train, y_test = pheno_tensor[train_idx], pheno_tensor[test_idx]
    y_test_np = pheno_scaled[test_idx]

    print(f"\nTraining set: {train_size} accessions | Held-out test set: {test_size} accessions")
    print(f"Training for {epochs} epochs per model architecture (batch size = {batch_size})...\n")

    results_by_model = {}

    for m_key in models_to_test:
        print(f"--- Training {model_names_display[m_key]} ---")
        if m_key == "geno_only":
            model = GalianaGenomicOnly(num_genes=num_genes, num_variant_types=num_variant_types, num_phenotypes=num_phenotypes)
        elif m_key == "env_only":
            model = EnvironmentalOnly(num_env_features=num_env, num_phenotypes=num_phenotypes)
        elif m_key == "additive":
            model = GalianaEnvAdditive(num_genes=num_genes, num_variant_types=num_variant_types, num_env_features=num_env, num_phenotypes=num_phenotypes)
        elif m_key == "gxe":
            model = GalianaGxE(num_genes=num_genes, num_variant_types=num_variant_types, num_env_features=num_env, num_phenotypes=num_phenotypes)

        model.to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {total_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.MSELoss()

        dataset_train = torch.utils.data.TensorDataset(x_g_train, x_e_train, y_train)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        epoch_losses = []
        for ep in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for bg, be, by in loader_train:
                bg, be, by = bg.to(device), be.to(device), by.to(device)
                optimizer.zero_grad()
                if m_key == "geno_only":
                    out = model(bg)
                elif m_key == "env_only":
                    out = model(be)
                else:
                    out = model(bg, be)

                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(by)
            avg_loss = running_loss / train_size
            epoch_losses.append(avg_loss)
            if ep % 5 == 0 or ep == 1:
                print(f"  Epoch [{ep:02d}/{epochs:02d}] - Train MSE: {avg_loss:.4f}")

        # Evaluate on Held-out Test Set
        model.eval()
        with torch.no_grad():
            if m_key == "geno_only":
                test_preds = model(x_g_test.to(device)).cpu().numpy()
            elif m_key == "env_only":
                test_preds = model(x_e_test.to(device)).cpu().numpy()
            else:
                test_preds = model(x_g_test.to(device), x_e_test.to(device)).cpu().numpy()

        results_by_model[m_key] = {
            "model": model,
            "params": total_params,
            "loss_curve": epoch_losses,
            "test_preds": test_preds
        }

    # 4. Compute Test Performance Metrics
    print("\n" + "=" * 75)
    print("HELD-OUT TEST SET PERFORMANCE (20 ACCESSIONS x 215 PHENOTYPES)")
    print("=" * 75)

    per_model_metrics = {}
    for m_key in models_to_test:
        preds = results_by_model[m_key]["test_preds"]
        mse = float(np.mean((preds - y_test_np) ** 2))
        mae = float(np.mean(np.abs(preds - y_test_np)))
        
        # Pearson r across each phenotype
        r_list = []
        for p in range(num_phenotypes):
            if np.std(preds[:, p]) > 1e-6 and np.std(y_test_np[:, p]) > 1e-6:
                r_val = float(np.corrcoef(preds[:, p], y_test_np[:, p])[0, 1])
            else:
                r_val = 0.0
            r_list.append(r_val)
        avg_r = float(np.mean(r_list))
        r2 = 1.0 - (mse / float(np.var(y_test_np)))

        per_model_metrics[m_key] = {
            "name": model_names_display[m_key],
            "params": results_by_model[m_key]["params"],
            "test_mse": round(mse, 4),
            "test_mae": round(mae, 4),
            "test_pearson_r": round(avg_r, 4),
            "test_r2": round(r2, 4),
            "per_trait_r": r_list
        }

        print(f"{model_names_display[m_key]:<36} | MSE: {mse:.4f} | MAE: {mae:.4f} | Pearson r: {avg_r:.4f} | R²: {r2:.4f}")

    # 5. Trait-by-Trait Comparison (G-Only vs GxE)
    r_geno = per_model_metrics["geno_only"]["per_trait_r"]
    r_gxe = per_model_metrics["gxe"]["per_trait_r"]
    r_env = per_model_metrics["env_only"]["per_trait_r"]
    r_add = per_model_metrics["additive"]["per_trait_r"]

    trait_performance_rows = []
    for p in range(num_phenotypes):
        t_name = pheno_names[p]
        delta_r = r_gxe[p] - r_geno[p]
        delta_pct = (delta_r / max(abs(r_geno[p]), 1e-4)) * 100.0

        trait_performance_rows.append({
            "phenotype_id": p + 1,
            "phenotype_name": t_name,
            "r_geno_only": round(r_geno[p], 4),
            "r_env_only": round(r_env[p], 4),
            "r_additive": round(r_add[p], 4),
            "r_gxe_interaction": round(r_gxe[p], 4),
            "delta_r": round(delta_r, 4),
            "improvement_pct": round(delta_pct, 2)
        })

    trait_performance_rows.sort(key=lambda x: x["delta_r"], reverse=True)

    csv_trait_path = os.path.join(results_dir, "env_phenotype_performance.csv")
    with open(csv_trait_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["phenotype_id", "phenotype_name", "r_geno_only", "r_env_only", "r_additive", "r_gxe_interaction", "delta_r", "improvement_pct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trait_performance_rows)

    print(f"\nExported per-phenotype performance comparison to {csv_trait_path}")

    # 6. Environmental Feature Importance Analysis (Gradient Saliency of Bioclimatic Factors)
    print("\nEvaluating Environmental Factor Feature Importance (Gradient Saliency)...")
    gxe_trained = results_by_model["gxe"]["model"]
    gxe_trained.eval()

    sample_x_g = x_g_test[:20].clone().to(device)
    sample_x_e = x_e_test[:20].clone().to(device)
    sample_x_e.requires_grad_(True)

    out_eval = gxe_trained(sample_x_g, sample_x_e)
    gxe_trained.zero_grad()
    loss_saliency = out_eval.sum()
    loss_saliency.backward()
    env_importance = sample_x_e.grad.abs().mean(dim=0).cpu().numpy()

    top_env_indices = np.argsort(env_importance)[::-1]
    top_env_data = []
    for idx in top_env_indices:
        e_name = env_names[idx]
        score = float(env_importance[idx])
        top_env_data.append({
            "env_feature_index": int(idx),
            "env_feature_name": e_name,
            "importance_score": round(score, 6)
        })

    csv_env_path = os.path.join(results_dir, "top_environmental_features.csv")
    with open(csv_env_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["env_feature_index", "env_feature_name", "importance_score"])
        writer.writeheader()
        writer.writerows(top_env_data)

    print(f"Exported top environmental factor rankings to {csv_env_path}")

    # 7. Generate Rich Visualizations

    # Figure 1: Comprehensive Model Architecture Comparison
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    model_labels = ["G-Only\n(Baseline)", "E-Only\n(Bioclim)", "G + E\n(Additive)", "Full GxE\n(Interaction)"]
    pearsons = [per_model_metrics[m]["test_pearson_r"] for m in models_to_test]
    mses = [per_model_metrics[m]["test_mse"] for m in models_to_test]

    x = np.arange(len(model_labels))
    width = 0.35

    rects1 = ax1.bar(x - width/2, pearsons, width, label='Test Pearson Correlation (r)', color='#2563eb', edgecolor='#1d4ed8')
    ax1.set_ylabel('Pearson Correlation (r)', color='#1e3a8a', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, mses, width, label='Test MSE (Lower is Better)', color='#f97316', edgecolor='#ea580c')
    ax2.set_ylabel('Mean Squared Error (MSE)', color='#9a3412', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(mses) * 1.3)

    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

    for rect in rects2:
        height = rect.get_height()
        ax2.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title('Predictive Performance: Genomic Baseline vs Environmental Upgrades', fontsize=13, fontweight='bold', pad=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontsize=11, fontweight='bold')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)
    plt.tight_layout()
    fig1_path = os.path.join(results_dir, "env_model_comparison.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # Figure 2: Trait-by-Trait Correlation Delta Scatter Plot
    plt.figure(figsize=(9, 6))
    deltas = [r_gxe[i] - r_geno[i] for i in range(num_phenotypes)]
    plt.scatter(r_geno, r_gxe, c=deltas, cmap='viridis', s=45, alpha=0.85, edgecolors='none')
    cbar = plt.colorbar()
    cbar.set_label('GxE Improvement (Δr)', fontsize=11)
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Equal Performance Line (y=x)')
    plt.title('Per-Trait Predictive Accuracy: Full GxE vs Genotype-Only', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Genotype-Only Pearson Correlation (r)', fontsize=12)
    plt.ylabel('Full GxE Interaction Pearson Correlation (r)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    fig2_path = os.path.join(results_dir, "env_phenotype_delta_scatter.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # Figure 3: Top Environmental Factors Feature Importance Bar Chart
    plt.figure(figsize=(10, 6.5))
    top15_env_names = [item["env_feature_name"] for item in top_env_data[:15]]
    top15_env_scores = [item["importance_score"] for item in top_env_data[:15]]
    plt.barh(top15_env_names[::-1], top15_env_scores[::-1], color='#0284c7', edgecolor='#0369a1')
    plt.title('Top 15 Most Important Environmental & Bioclimatic Drivers', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Environmental Importance Score (Mean Gradient)', fontsize=12)
    plt.ylabel('Environmental Feature (WorldClim / Climate)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6, axis='x')
    plt.tight_layout()
    fig3_path = os.path.join(results_dir, "top_environmental_drivers.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    # Figure 4: Comparative Loss Curves
    plt.figure(figsize=(9, 5))
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, results_by_model["geno_only"]["loss_curve"], marker='o', color='#94a3b8', linewidth=2, label='G-Only Baseline')
    plt.plot(epochs_range, results_by_model["env_only"]["loss_curve"], marker='^', color='#f59e0b', linewidth=2, label='E-Only Baseline')
    plt.plot(epochs_range, results_by_model["additive"]["loss_curve"], marker='s', color='#3b82f6', linewidth=2, label='G + E Additive')
    plt.plot(epochs_range, results_by_model["gxe"]["loss_curve"], marker='D', color='#10b981', linewidth=2.5, label='Full GxE Interaction')
    plt.title('Training Loss Convergence Across Architectures', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss (MSE)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    fig4_path = os.path.join(results_dir, "env_training_loss_curves.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    # Figure 5: Top Traits with Largest Environmental GxE Benefit
    plt.figure(figsize=(10, 6))
    top10_improved = trait_performance_rows[:10]
    t_names_short = [row["phenotype_name"][:35] + ("..." if len(row["phenotype_name"]) > 35 else "") for row in top10_improved]
    g_bars = [row["r_geno_only"] for row in top10_improved]
    gxe_bars = [row["r_gxe_interaction"] for row in top10_improved]

    y_pos = np.arange(len(top10_improved))
    h = 0.35
    plt.barh(y_pos - h/2, g_bars[::-1], h, color='#94a3b8', label='G-Only (Baseline)')
    plt.barh(y_pos + h/2, gxe_bars[::-1], h, color='#10b981', label='Full GxE (Upgraded)')
    plt.yticks(y_pos, t_names_short[::-1], fontsize=10)
    plt.xlabel('Pearson Correlation (r)', fontsize=12)
    plt.title('Top 10 Plant Phenotypes Showing Largest Environmental GxE Gain', fontsize=13, fontweight='bold', pad=12)
    plt.grid(True, linestyle='--', alpha=0.6, axis='x')
    plt.legend(fontsize=11, loc='lower right')
    plt.tight_layout()
    fig5_path = os.path.join(results_dir, "top_gxe_improved_traits.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()

    # 8. Save Metrics Summary JSON & TXT
    metrics_summary = {
        "dataset_name": "Arabidopsis thaliana AraPheno + Bioclimatic Dataset",
        "num_accessions_total": n_samples,
        "num_accessions_train": train_size,
        "num_accessions_test": test_size,
        "num_environmental_factors": num_env,
        "num_plant_phenotypes": num_phenotypes,
        "num_genes": num_genes,
        "num_variant_types": num_variant_types,
        "epochs": epochs,
        "models_evaluated": {
            "genotype_only_baseline": {
                "trainable_parameters": per_model_metrics["geno_only"]["params"],
                "test_mse": per_model_metrics["geno_only"]["test_mse"],
                "test_mae": per_model_metrics["geno_only"]["test_mae"],
                "test_pearson_r": per_model_metrics["geno_only"]["test_pearson_r"],
                "test_r2": per_model_metrics["geno_only"]["test_r2"]
            },
            "environment_only_baseline": {
                "trainable_parameters": per_model_metrics["env_only"]["params"],
                "test_mse": per_model_metrics["env_only"]["test_mse"],
                "test_mae": per_model_metrics["env_only"]["test_mae"],
                "test_pearson_r": per_model_metrics["env_only"]["test_pearson_r"],
                "test_r2": per_model_metrics["env_only"]["test_r2"]
            },
            "additive_g_plus_e": {
                "trainable_parameters": per_model_metrics["additive"]["params"],
                "test_mse": per_model_metrics["additive"]["test_mse"],
                "test_mae": per_model_metrics["additive"]["test_mae"],
                "test_pearson_r": per_model_metrics["additive"]["test_pearson_r"],
                "test_r2": per_model_metrics["additive"]["test_r2"]
            },
            "full_gxe_interaction": {
                "trainable_parameters": per_model_metrics["gxe"]["params"],
                "test_mse": per_model_metrics["gxe"]["test_mse"],
                "test_mae": per_model_metrics["gxe"]["test_mae"],
                "test_pearson_r": per_model_metrics["gxe"]["test_pearson_r"],
                "test_r2": per_model_metrics["gxe"]["test_r2"]
            }
        },
        "gxe_over_genotype_improvement": {
            "delta_pearson_r": round(per_model_metrics["gxe"]["test_pearson_r"] - per_model_metrics["geno_only"]["test_pearson_r"], 4),
            "delta_mse_reduction": round(per_model_metrics["geno_only"]["test_mse"] - per_model_metrics["gxe"]["test_mse"], 4),
            "percent_r_gain": round(((per_model_metrics["gxe"]["test_pearson_r"] - per_model_metrics["geno_only"]["test_pearson_r"]) / per_model_metrics["geno_only"]["test_pearson_r"]) * 100, 2)
        },
        "top_environmental_drivers": [item["env_feature_name"] for item in top_env_data[:5]],
        "top_gxe_responsive_traits": [item["phenotype_name"] for item in top10_improved[:5]]
    }

    json_path = os.path.join(results_dir, "env_metrics_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4, ensure_ascii=False)

    txt_path = os.path.join(results_dir, "env_metrics_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("GENOTYPE-BY-ENVIRONMENT (GxE) UPGRADE EXPERIMENT SUMMARY\n")
        f.write("=" * 65 + "\n")
        for k, v in metrics_summary.items():
            f.write(f"{k}: {v}\n")

    print(f"\nAll experimental results, summary metrics, and figures saved to: '{results_dir}/'")
    return metrics_summary

if __name__ == '__main__':
    run_environmental_upgrade_experiments(num_samples=100, epochs=20, batch_size=16)
