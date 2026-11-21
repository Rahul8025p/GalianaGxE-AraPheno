# Galiana GxE: Upgraded Deep Genotype-by-Environment Phenotype Prediction

This repository is an upgraded implementation of the **Galiana** neural network architecture for *Arabidopsis thaliana*, extended with a **Genotype-by-Environment (G×E) Interaction Module** to predict complex quantitative plant traits from both whole-genome sequencing variants and real bioclimatic/environmental factors.

Based on the foundational paper:
> *From genotype to phenotype in Arabidopsis thaliana: in-silico genome interpretation predicts 288 phenotypes from sequencing data*

---

## 🌿 Motivation: Why Environmental Factors Matter

In quantitative genetics, evolutionary ecology, and plant biology, phenotype determination is governed by the fundamental equation:

$$\text{Phenotype } (P) = \text{Genotype } (G) + \text{Environment } (E) + G \times E \text{ Interaction} + \epsilon$$

The baseline Galiana model evaluated phenotypes exclusively from genomic variant counts ($P = f(G)$). However:
1. Plants are sessile organisms whose phenotypes (flowering time, root architecture, germination, biomass, seed longevity, and ion uptake) exhibit extensive **phenotypic plasticity** governed by environmental gradients.
2. In the AraPheno database (1001 Genomes Project), **73 variables are bioclimatic and meteorological factors** (WorldClim bioclimatic indices Bio1–Bio19, seasonal temperatures, rainfall, potential/actual evapotranspiration, growing season indices) that act as **environmental drivers**, not phenotypic traits.
3. Incorporating environmental factors allows the model to capture **Gene-by-Environment ($G \times E$) cross-regulations** (e.g. drought and cold response genes whose effect sizes change across climatic regimes).

---

## 🧬 Architecture Overview

The upgraded model (`GalianaGxE`) features four coordinated neural modules:

```
[ Genomic Variants ] (27,655 genes x 17 types)
         │
    ┌────▼────┐
    │ G-Module│ (Weight-shared linear layers)
    └────┬────┘
         │  Scalar gene representations (27,655)
    ┌────▼────────┐
    │ Projection  │ ──► z_G (Genomic Latent Vector, dim 50)
    └─────────────┘          │
                             ├──────────────┬─────────────┐
                             │              │             │
[ Bioclimatic Factors ]      │              │             │
(73 Climate / WorldClim)     │              │             │
         │                   │              │             │
    ┌────▼────┐              │              ▼             ▼
    │ E-Module│ ──► z_E ─────┴──────► [ Bilinear ]   [ Sigmoid Gate ]
    └─────────┘ (dim 50)               (G x E)          (z_G * gate)
                             │              │             │
                             └──────────────┼─────────────┘
                                            ▼
                               [ Fused Latent Vector ] (dim 200)
                                            │
                                       ┌────▼────┐
                                       │ P-Module│
                                       └────┬────┘
                                            ▼
                             [ 215 Quantitative Traits ]
```

1. **G-Module (Genomic Encoder):**
   - Applies weight sharing across all 27,655 genes to project 17-dimensional variant consequence histograms into scalar gene representations.
2. **E-Module (Environmental Encoder):**
   - Encodes the 73 bioclimatic variables (WorldClim Bio1–Bio19, evapotranspiration, seasonal temperatures, growing season indicators) through a deep MLP with LayerNorm, GELU, and Dropout into an environmental latent embedding $z_E \in \mathbb{R}^{50}$.
3. **G×E Interaction Module:**
   - **Bilinear Multiplicative Interaction:** Models allele-specific climate responses via low-rank Hadamard product $z_{G \times E} = (W_G z_G) \odot (W_E z_E)$.
   - **Environmental Gating:** Dynamically modulates genomic pathway fluxes via an environmental sigmoid gate $g_E = \sigma(W_{gate} z_E)$.
4. **P-Module (Multi-task Phenotype Predictor):**
   - Maps the joint genomic-environmental representation ($[z_G, z_E, z_{G \times E}, z_G \odot g_E]$) to 215 real *A. thaliana* phenotypes simultaneously.

---

## 📊 Comparative Performance Results

Evaluated on 100 *Arabidopsis thaliana* ecotypes with held-out test evaluation:

| Architecture | Input Features | Parameters | Test MSE (↓) | Test MAE (↓) | **Test Pearson $r$ (↑)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Genotype-Only (Galiana Baseline)** | 27,655 Genes $\times$ 17 Variants | 1,452,876 | 0.7759 | 0.6284 | `0.4122` |
| **Environment-Only (Bioclim Baseline)** | 73 Bioclimatic Factors | 19,325 | 1.0854 | 0.7330 | `0.1081` |
| **G + E Additive Model** | Genotype + Environment (Concat) | 1,463,736 | 0.7917 | 0.6319 | `0.4135` |
| **Full G×E Interaction Model** | Genotype + Environment + Interaction | **1,491,786** | 0.7992 | 0.6352 | **`0.4178`** |

### Top Traits Benefiting from Environmental G×E Interactions
- **Lateral / Main Root Length ratio (`LRLpMRL0`):** Pearson $r$ improved from $-0.072$ to **$+0.5537$** ($\Delta r = +0.6258$).
- **Gravitropic Setpoint Angle (`GSA`):** Pearson $r$ improved from $-0.4409$ to **$+0.1468$** ($\Delta r = +0.5877$).
- **Natural Seed Aging Treatment (`NAT`):** Pearson $r$ improved from $-0.3107$ to **$+0.2168$** ($\Delta r = +0.5275$).
- **Shoot Primordia Development:** Pearson $r$ improved from $+0.2069$ to **$+0.6132$** ($\Delta r = +0.4062$).
- **Days to Flowering in Sweden (`DTFplantingSummerLocSweden2009`):** Pearson $r$ improved by **$\Delta r = +0.3092$**.

### Top Environmental & Bioclimatic Drivers
1. `clim-gs11`: Growing season month index
2. `clim-aet10`: Actual Evapotranspiration (Autumn)
3. `clim-cyclelength`: Life cycle duration
4. `clim-bio18`: Precipitation of Warmest Quarter
5. `clim-gs5`: Growing season thermal index
6. `clim-tmin7`: July Minimum Temperature

---

## 📁 Repository Structure

- [`model_env.py`](file:///e:/RM/model_env.py): Upgraded PyTorch module definitions (`GModule`, `EModule`, `GxEInteractionModule`, `GalianaGxE`, `GalianaEnvAdditive`, `EnvironmentalOnly`, `GalianaGenomicOnly`).
- [`dataset_env.py`](file:///e:/RM/dataset_env.py): Automatic dataset splitter that parses real AraPheno data into environmental factors and plant phenotypic traits.
- [`train.py`](file:///e:/RM/train.py): Training script supporting both Genotype-only and G×E environmental training.
- [`generate_env_results.py`](file:///e:/RM/generate_env_results.py): Comprehensive benchmarking pipeline across all 4 architectures with automatic visual generation.
- [`model.py`](file:///e:/RM/model.py): Original legacy Galiana architecture.
- [`results/`](file:///e:/RM/results):
  - [`env_model_comparison.png`](file:///e:/RM/results/env_model_comparison.png): Comparative bar chart across all 4 architectures.
  - [`env_phenotype_delta_scatter.png`](file:///e:/RM/results/env_phenotype_delta_scatter.png): Trait-by-trait scatter plot of G-only vs G×E performance.
  - [`top_environmental_drivers.png`](file:///e:/RM/results/top_environmental_drivers.png): Feature importance ranking of bioclimatic factors.
  - [`top_gxe_improved_traits.png`](file:///e:/RM/results/top_gxe_improved_traits.png): Bar chart of the top phenotypes showing the greatest G×E gain.
  - [`env_training_loss_curves.png`](file:///e:/RM/results/env_training_loss_curves.png): Convergence curves across architectures.
  - [`env_metrics_summary.json`](file:///e:/RM/results/env_metrics_summary.json): Complete numeric metrics summary.
  - [`env_phenotype_performance.csv`](file:///e:/RM/results/env_phenotype_performance.csv): Trait-by-trait performance comparison table.
  - [`top_environmental_features.csv`](file:///e:/RM/results/top_environmental_features.csv): Ranking of all 73 bioclimatic drivers.
  - [`ENVIRONMENTAL_UPGRADE_REPORT.md`](file:///e:/RM/results/ENVIRONMENTAL_UPGRADE_REPORT.md): Comprehensive scientific report.

---

## 🚀 Quickstart

### 1. Run the Upgraded Environmental Training Loop
```bash
# Train full GxE model with environmental factors
python train.py --use-env --epochs 10

# Train legacy genotype-only model
python train.py --no-env --epochs 10
```

### 2. Run Comprehensive Comparative Benchmark & Regenerate Plots
```bash
python generate_env_results.py
```
