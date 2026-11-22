import torch
import torch.nn as nn
import torch.nn.functional as F

class GModule(nn.Module):
    """
    Genomic Variant Encoder:
    Processes the 17-dimensional variant consequence histogram for each gene.
    Shared weights across all 27,655 genes.
    Architecture: Dropout(0.1) -> Linear(17, 50) -> LayerNorm(50) -> Tanh -> Linear(50, 1)
    """
    def __init__(self, num_variant_types=17):
        super(GModule, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(num_variant_types, 50)
        self.layer_norm = nn.LayerNorm(50)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(50, 1)

    def forward(self, x):
        # x shape: (batch_size, num_genes, num_variant_types)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.layer_norm(x)
        x = self.tanh(x)
        x = self.linear2(x)
        # Output shape: (batch_size, num_genes, 1)
        return x

class EModule(nn.Module):
    """
    Environmental Encoder:
    Processes the bioclimatic and environmental variables (e.g. WorldClim Bio1-Bio19,
    seasonal precipitation, minimum/maximum temperatures, evapotranspiration, growing season indices).
    Maps high-dimensional environmental context into a compact environmental latent embedding.
    """
    def __init__(self, num_env_features=73, env_embedding_dim=50):
        super(EModule, self).__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(num_env_features),
            nn.Linear(num_env_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, env_embedding_dim),
            nn.LayerNorm(env_embedding_dim),
            nn.Tanh()
        )

    def forward(self, env):
        # env shape: (batch_size, num_env_features)
        return self.encoder(env)

class GxEInteractionModule(nn.Module):
    """
    Genotype-by-Environment (GxE) Interaction Layer:
    Captures non-linear phenotypic plasticity and climate-adaptive genetic effects.
    Bilinear interaction only (gate removed):
      - Genome projection: a = W_G z_G + b_G in R^50
      - Climate projection: c = W_E z_E + b_E in R^50
      - Hadamard product: a ⊙ c (element-wise)
      - LayerNorm(50) -> z_bilinear in R^50
      - Raw z_G and z_E skip straight to fusion: [z_G, z_E, z_bilinear] in R^150
    """
    def __init__(self, gene_latent_dim=50, env_latent_dim=50, hidden_dim=50):
        super(GxEInteractionModule, self).__init__()
        # Bilinear cross-interaction projection (W_G, b_G and W_E, b_E)
        self.proj_geno = nn.Linear(gene_latent_dim, hidden_dim)
        self.proj_env = nn.Linear(env_latent_dim, hidden_dim)
        self.ln_interaction = nn.LayerNorm(hidden_dim)

    def forward(self, z_g, z_e):
        # z_g: (batch_size, gene_latent_dim)
        # z_e: (batch_size, env_latent_dim)
        
        # 1. Genome and Climate projections
        a = self.proj_geno(z_g)   # (batch_size, 50)
        c = self.proj_env(z_e)    # (batch_size, 50)

        # 2. Multiplicative Bilinear G x E Interaction: a ⊙ c + LayerNorm
        z_bilinear = self.ln_interaction(a * c)  # (batch_size, 50)

        # 3. Latent fusion (dim 150): [raw z_G, raw z_E, z_bilinear]
        fused = torch.cat([z_g, z_e, z_bilinear], dim=-1)
        return fused

class GalianaGxE(nn.Module):
    """
    Upgraded Full Galiana Genotype-by-Environment (GxE) Model.
    Predicts plant phenotypes using both genome variation and bioclimatic/environmental factors.
    Gate removed: bilinear interaction only, 150-dim latent fusion -> P-head (150 -> 100 -> 50 -> 215).
    """
    def __init__(
        self,
        num_genes=27655,
        num_variant_types=17,
        num_env_features=73,
        num_phenotypes=215,
        gene_latent_dim=50,
        env_latent_dim=50
    ):
        super(GalianaGxE, self).__init__()
        self.num_genes = num_genes
        self.num_variant_types = num_variant_types
        self.num_env_features = num_env_features
        self.num_phenotypes = num_phenotypes

        # Genomic Module
        self.g_module = GModule(num_variant_types=num_variant_types)
        self.g_projection = nn.Sequential(
            nn.LayerNorm(num_genes),
            nn.Dropout(p=0.2),
            nn.Linear(num_genes, gene_latent_dim),
            nn.Tanh()
        )

        # Environmental Module
        self.e_module = EModule(num_env_features=num_env_features, env_embedding_dim=env_latent_dim)

        # GxE Interaction Module (Bilinear cross-term only, gate removed)
        self.gxe_module = GxEInteractionModule(
            gene_latent_dim=gene_latent_dim,
            env_latent_dim=env_latent_dim,
            hidden_dim=gene_latent_dim
        )

        # Phenotype Prediction Head: 150 -> 100 -> 50 -> num_phenotypes
        fused_dim = gene_latent_dim + env_latent_dim + gene_latent_dim  # 50 + 50 + 50 = 150
        self.p_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(fused_dim, 100),
            nn.LayerNorm(100),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(100, 50),
            nn.LayerNorm(50),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(50, num_phenotypes)
        )

    def forward(self, x_geno, x_env):
        # 1. Process Genotype
        # x_geno: (batch_size, num_genes, num_variant_types)
        g_out = self.g_module(x_geno)          # (batch_size, num_genes, 1)
        g_flat = g_out.view(x_geno.size(0), -1) # (batch_size, num_genes)
        z_g = self.g_projection(g_flat)        # (batch_size, 50)

        # 2. Process Environmental Factors
        # x_env: (batch_size, num_env_features)
        z_e = self.e_module(x_env)              # (batch_size, 50)

        # 3. Model GxE Interactions & Fusion
        fused = self.gxe_module(z_g, z_e)       # (batch_size, 150)

        # 4. Predict Phenotypes
        y_pred = self.p_head(fused)            # (batch_size, num_phenotypes)
        return y_pred

class GalianaEnvAdditive(nn.Module):
    """
    Ablation baseline: Additive Genotype + Environment model (linear concatenation without GxE cross terms).
    """
    def __init__(
        self,
        num_genes=27655,
        num_variant_types=17,
        num_env_features=73,
        num_phenotypes=215
    ):
        super(GalianaEnvAdditive, self).__init__()
        self.g_module = GModule(num_variant_types=num_variant_types)
        self.g_projection = nn.Sequential(
            nn.LayerNorm(num_genes),
            nn.Dropout(p=0.2),
            nn.Linear(num_genes, 50),
            nn.LayerNorm(50),
            nn.Tanh()
        )
        self.e_module = EModule(num_env_features=num_env_features, env_embedding_dim=50)
        self.p_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(100, 50),
            nn.LayerNorm(50),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(50, num_phenotypes)
        )

    def forward(self, x_geno, x_env):
        g_out = self.g_module(x_geno).view(x_geno.size(0), -1)
        z_g = self.g_projection(g_out)
        z_e = self.e_module(x_env)
        fused = torch.cat([z_g, z_e], dim=-1)
        return self.p_head(fused)

class EnvironmentalOnly(nn.Module):
    """
    Ablation baseline: Environment-Only model (phenotype prediction purely from bioclimatic factors).
    """
    def __init__(self, num_env_features=73, num_phenotypes=215):
        super(EnvironmentalOnly, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(num_env_features),
            nn.Linear(num_env_features, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 50),
            nn.LayerNorm(50),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(50, num_phenotypes)
        )

    def forward(self, x_env):
        return self.net(x_env)

class GalianaGenomicOnly(nn.Module):
    """
    Original Galiana model (Genotype-only baseline).
    """
    def __init__(self, num_genes=27655, num_variant_types=17, num_phenotypes=215):
        super(GalianaGenomicOnly, self).__init__()
        self.g_module = GModule(num_variant_types=num_variant_types)
        self.p_module = nn.Sequential(
            nn.LayerNorm(num_genes),
            nn.Dropout(p=0.2),
            nn.Linear(num_genes, 50),
            nn.LayerNorm(50),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(50, 50),
            nn.LayerNorm(50),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(50, num_phenotypes)
        )

    def forward(self, x_geno):
        g_out = self.g_module(x_geno).view(x_geno.size(0), -1)
        return self.p_module(g_out)

if __name__ == "__main__":
    # Test instantiation and forward pass
    batch_size = 4
    n_genes = 27655
    n_var = 17
    n_env = 73
    n_phen = 215

    sample_geno = torch.randn(batch_size, n_genes, n_var)
    sample_env = torch.randn(batch_size, n_env)

    # Test GxE Model
    model_gxe = GalianaGxE(num_genes=n_genes, num_variant_types=n_var, num_env_features=n_env, num_phenotypes=n_phen)
    out_gxe = model_gxe(sample_geno, sample_env)
    params_gxe = sum(p.numel() for p in model_gxe.parameters() if p.requires_grad)
    print(f"GalianaGxE output shape: {out_gxe.shape}, Trainable parameters: {params_gxe:,}")

    # Test Additive Model
    model_add = GalianaEnvAdditive(num_genes=n_genes, num_variant_types=n_var, num_env_features=n_env, num_phenotypes=n_phen)
    out_add = model_add(sample_geno, sample_env)
    print(f"GalianaEnvAdditive output shape: {out_add.shape}")

    # Test Env-Only Model
    model_env_only = EnvironmentalOnly(num_env_features=n_env, num_phenotypes=n_phen)
    out_env_only = model_env_only(sample_env)
    print(f"EnvironmentalOnly output shape: {out_env_only.shape}")

    # Test Genomic-Only Model
    model_geno_only = GalianaGenomicOnly(num_genes=n_genes, num_variant_types=n_var, num_phenotypes=n_phen)
    out_geno_only = model_geno_only(sample_geno)
    print(f"GalianaGenomicOnly output shape: {out_geno_only.shape}")

