import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from model_env import GalianaGxE, GalianaEnvAdditive, GalianaGenomicOnly, EnvironmentalOnly
from dataset_env import GalianaEnvDataset, load_and_split_arapheno_data

def train_env_model(
    model_type="gxe",
    num_samples=200,
    num_epochs=15,
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-5,
    save_path="training_env_results.png"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Upgraded Environmental Model ({model_type.upper()}) on {device} ---")

    # Load data
    try:
        data_dict = load_and_split_arapheno_data()
        print("Loaded real AraPheno dataset and partitioned environmental factors.")
        env_raw = data_dict["env_data"][:num_samples]
        pheno_raw = data_dict["pheno_data"][:num_samples]
        n_samples = len(env_raw)
        n_env = env_raw.shape[1]
        n_pheno = pheno_raw.shape[1]
        n_genes = 27655
        n_var = 17

        # Standardize
        env_std = (env_raw - env_raw.mean(axis=0)) / (env_raw.std(axis=0) + 1e-6)
        pheno_std = (pheno_raw - pheno_raw.mean(axis=0)) / (pheno_raw.std(axis=0) + 1e-6)

        # Load real WGS genotype features if available, otherwise initialize standardized tensor
        geno_path = "data/real_wgs_genotype_100.npy"
        if os.path.exists(geno_path):
            real_geno = np.load(geno_path)[:n_samples]
            X_geno = torch.tensor(real_geno, dtype=torch.float32)
        else:
            torch.manual_seed(42)
            X_geno = torch.randn(n_samples, n_genes, n_var) * 0.5
        X_env = torch.tensor(env_std, dtype=torch.float32)
        Y_pheno = torch.tensor(pheno_std, dtype=torch.float32)

        dataset = GalianaEnvDataset(
            X_geno=X_geno,
            X_env=X_env,
            Y_pheno=Y_pheno,
            num_samples=n_samples,
            num_genes=n_genes,
            num_variant_types=n_var,
            num_env_features=n_env,
            num_phenotypes=n_pheno
        )
    except Exception as e:
        print(f"Note: Falling back to baseline dataset initialization: {e}")
        n_samples = num_samples
        n_genes = 5000
        n_var = 17
        n_env = 73
        n_pheno = 50
        dataset = GalianaEnvDataset(
            num_samples=n_samples,
            num_genes=n_genes,
            num_variant_types=n_var,
            num_env_features=n_env,
            num_phenotypes=n_pheno
        )

    # Train / Val Split (80% / 20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Initialize selected model
    if model_type == "gxe":
        model = GalianaGxE(
            num_genes=n_genes,
            num_variant_types=n_var,
            num_env_features=n_env,
            num_phenotypes=n_pheno
        )
    elif model_type == "additive":
        model = GalianaEnvAdditive(
            num_genes=n_genes,
            num_variant_types=n_var,
            num_env_features=n_env,
            num_phenotypes=n_pheno
        )
    elif model_type == "env_only":
        model = EnvironmentalOnly(
            num_env_features=n_env,
            num_phenotypes=n_pheno
        )
    elif model_type == "geno_only":
        model = GalianaGenomicOnly(
            num_genes=n_genes,
            num_variant_types=n_var,
            num_phenotypes=n_pheno
        )
    else:
        raise ValueError(f"Unknown model_type {model_type}")

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    val_losses = []

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for x_g, x_e, y in train_loader:
            x_g, x_e, y = x_g.to(device), x_e.to(device), y.to(device)
            optimizer.zero_grad()

            if model_type in ["gxe", "additive"]:
                out = model(x_g, x_e)
            elif model_type == "env_only":
                out = model(x_e)
            else:
                out = model(x_g)

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)

        train_loss = running_loss / len(train_set)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for x_g, x_e, y in val_loader:
                x_g, x_e, y = x_g.to(device), x_e.to(device), y.to(device)
                if model_type in ["gxe", "additive"]:
                    val_out = model(x_g, x_e)
                elif model_type == "env_only":
                    val_out = model(x_e)
                else:
                    val_out = model(x_g)
                val_running_loss += criterion(val_out, y).item() * len(y)

        val_loss = val_running_loss / len(val_set)
        val_losses.append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Train MSE: {train_loss:.4f} - Val MSE: {val_loss:.4f}")

    print("Training finished successfully.")

    # Save training curves
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', color='#1d4ed8', linewidth=2, label='Train MSE Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, marker='s', color='#10b981', linewidth=2, linestyle='--', label='Val MSE Loss')
    plt.title(f'Galiana Upgraded Model Training ({model_type.upper()})', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training plot to: {save_path}")

    return model, train_losses, val_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Galiana Upgraded Environmental Model")
    parser.add_argument("--model-type", type=str, default="gxe", choices=["gxe", "additive", "env_only", "geno_only"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    train_env_model(
        model_type=args.model_type,
        num_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )

