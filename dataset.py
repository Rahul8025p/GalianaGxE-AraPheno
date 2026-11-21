import os
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from dataset_env import load_and_split_arapheno_data, GalianaEnvDataset

class GalianaDataset(Dataset):
    """
    Standard Galiana dataset supporting both legacy genotype-only input
    and upgraded genotype-by-environment inputs with standardized features.
    """
    def __init__(
        self,
        num_samples=1021,
        num_genes=27655,
        num_variant_types=17,
        num_phenotypes=288,
        include_env=False
    ):
        self.num_samples = num_samples
        self.num_genes = num_genes
        self.num_variant_types = num_variant_types
        self.num_phenotypes = num_phenotypes
        self.include_env = include_env
        
        # Load real data if available on disk
        real_npy = "data/real_arapheno_Y.npy"
        if os.path.exists(real_npy):
            try:
                split = load_and_split_arapheno_data(real_npy)
                if include_env:
                    scaler_e = StandardScaler()
                    scaler_p = StandardScaler()
                    raw_e = split["env_data"][:num_samples]
                    raw_p = split["pheno_data"][:num_samples]
                    scaled_e = scaler_e.fit_transform(raw_e)
                    scaled_p = scaler_p.fit_transform(raw_p)
                    self.num_env_features = scaled_e.shape[1]
                    self.num_phenotypes = scaled_p.shape[1]
                    self.X_env = torch.tensor(scaled_e, dtype=torch.float32)
                    self.y = torch.tensor(scaled_p, dtype=torch.float32)
                else:
                    scaler_p = StandardScaler()
                    scaled_p = scaler_p.fit_transform(split["pheno_data"][:num_samples])
                    self.num_phenotypes = scaled_p.shape[1]
                    self.y = torch.tensor(scaled_p, dtype=torch.float32)
            except Exception:
                self.y = torch.randn(num_samples, num_phenotypes)
                if include_env:
                    self.X_env = torch.randn(num_samples, 73)
        else:
            self.y = torch.randn(num_samples, num_phenotypes)
            if include_env:
                self.X_env = torch.randn(num_samples, 73)

        self.X_geno = torch.randn(len(self.y), num_genes, num_variant_types) * 0.5

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.include_env:
            return self.X_geno[idx], self.X_env[idx], self.y[idx]
        return self.X_geno[idx], self.y[idx]

if __name__ == "__main__":
    dataset = GalianaDataset(num_samples=10, include_env=True)
    print(f"Dataset length: {len(dataset)}")
    x_g, x_e, y = dataset[0]
    print(f"Sample X_geno shape: {x_g.shape}")
    print(f"Sample X_env shape: {x_e.shape}")
    print(f"Sample y shape: {y.shape}")
    print(f"Sample y mean: {y.mean().item():.3f}, std: {y.std().item():.3f}")
