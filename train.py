import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import Galiana
from model_env import GalianaGxE
from dataset import GalianaDataset

def train(num_epochs=10, batch_size=10, lr=1e-3, weight_decay=1e-5, use_env=True, save_path="results/training_results.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on {device} (use_env={use_env})...")

    # 1. Dataset Initialization
    dataset = GalianaDataset(num_samples=100, include_env=use_env)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Model Initialization
    if use_env:
        model = GalianaGxE(
            num_genes=27655,
            num_variant_types=17,
            num_env_features=73,
            num_phenotypes=215
        )
    else:
        model = Galiana(
            num_genes=27655,
            num_variant_types=17,
            num_phenotypes=288
        )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # 3. Optimizer & Loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            if use_env:
                data_g, data_e, targets = batch
                data_g, data_e, targets = data_g.to(device), data_e.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(data_g, data_e)
            else:
                data_g, targets = batch
                data_g, targets = data_g.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(data_g)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(targets)

        avg_loss = epoch_loss / len(dataset)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}], Loss: {avg_loss:.4f}")

    print("Training finished.")

    # Save plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', color='#2563eb', linewidth=2, label='Training Loss (MSE)')
    plt.title(f'{"Galiana Upgraded GxE" if use_env else "Galiana Baseline"} Training Loss', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training loss curve saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Galiana (Genotype or GxE)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--use-env", action="store_true", default=True, help="Include environmental factors")
    parser.add_argument("--no-env", dest="use_env", action="store_false", help="Genotype only")
    args = parser.parse_args()

    train(num_epochs=args.epochs, batch_size=args.batch_size, use_env=args.use_env)
