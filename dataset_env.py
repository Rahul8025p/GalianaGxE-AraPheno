import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_split_arapheno_data(
    npy_path="data/real_arapheno_Y.npy",
    json_path="data/real_phenotype_names.json"
):
    """
    Loads AraPheno matrix and automatically separates environmental/climatic
    factors from genuine plant phenotypic traits based on trait nomenclature
    and bioclimatic taxonomy.
    """
    if not os.path.exists(npy_path) or not os.path.exists(json_path):
        raise FileNotFoundError(f"AraPheno data files not found at {npy_path} or {json_path}")

    raw_matrix = np.load(npy_path)
    with open(json_path, "r", encoding="utf-8") as f:
        all_trait_names = json.load(f)

    env_indices = []
    pheno_indices = []

    for idx, name in enumerate(all_trait_names):
        name_lower = name.lower()
        # Direct climate, bioclimatic (WorldClim), temperature, precipitation, evapotranspiration
        if (
            name_lower.startswith("clim-")
            or any(sub in name_lower for sub in ["bioclim", "worldclim"])
        ):
            env_indices.append(idx)
        else:
            pheno_indices.append(idx)

    env_data = raw_matrix[:, env_indices]
    pheno_data = raw_matrix[:, pheno_indices]

    env_names = [all_trait_names[i] for i in env_indices]
    pheno_names = [all_trait_names[i] for i in pheno_indices]

    return {
        "env_data": env_data,
        "env_names": env_names,
        "env_indices": env_indices,
        "pheno_data": pheno_data,
        "pheno_names": pheno_names,
        "pheno_indices": pheno_indices,
        "num_samples": raw_matrix.shape[0]
    }

class GalianaEnvDataset(Dataset):
    """
    Dataset for the Upgraded Genotype-by-Environment (GxE) Galiana Model.
    Provides:
      - X_geno: Genotype variant histogram (batch, num_genes, 17)
      - X_env: Environmental & Bioclimatic variables (batch, num_env_features)
      - Y_pheno: Plant Phenotypic traits (batch, num_phenotypes)
    """
    def __init__(
        self,
        X_geno=None,
        X_env=None,
        Y_pheno=None,
        num_samples=1021,
        num_genes=27655,
        num_variant_types=17,
        num_env_features=73,
        num_phenotypes=215
    ):
        super(GalianaEnvDataset, self).__init__()
        self.num_samples = num_samples
        self.num_genes = num_genes
        self.num_variant_types = num_variant_types
        self.num_env_features = num_env_features
        self.num_phenotypes = num_phenotypes

        if X_geno is not None:
            self.X_geno = torch.as_tensor(X_geno, dtype=torch.float32)
        else:
            self.X_geno = torch.randn(num_samples, num_genes, num_variant_types)

        if X_env is not None:
            self.X_env = torch.as_tensor(X_env, dtype=torch.float32)
        else:
            self.X_env = torch.randn(num_samples, num_env_features)

        if Y_pheno is not None:
            self.Y_pheno = torch.as_tensor(Y_pheno, dtype=torch.float32)
        else:
            self.Y_pheno = torch.randn(num_samples, num_phenotypes)

    def __len__(self):
        return len(self.Y_pheno)

    def __getitem__(self, idx):
        return self.X_geno[idx], self.X_env[idx], self.Y_pheno[idx]

if __name__ == "__main__":
    split_info = load_and_split_arapheno_data()
    print(f"Loaded samples: {split_info['num_samples']}")
    print(f"Environmental features count: {len(split_info['env_names'])}")
    print(f"Plant phenotypes count: {len(split_info['pheno_names'])}")
    print(f"Sample environmental factors: {split_info['env_names'][:5]}")
    print(f"Sample plant phenotypes: {split_info['pheno_names'][:5]}")

