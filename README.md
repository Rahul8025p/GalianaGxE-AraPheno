# Galiana GxE: Deep Genotype-by-Environment Phenotype Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An upgraded deep learning architecture for *Arabidopsis thaliana* quantitative phenotype prediction from **Whole-Genome Sequencing (WGS)** variants and real **bioclimatic/environmental factors**, featuring an explicit **Genotype-by-Environment (G×E) Interaction Module**.

Based on and extending the foundational paper:
> *From genotype to phenotype in Arabidopsis thaliana: in-silico genome interpretation predicts 288 phenotypes from sequencing data*  
> (Raimondi et al., *Nucleic Acids Research*, 2021)

---

## 🌿 Motivation: Why Genotype-by-Environment (G×E) Matters

In quantitative genetics, evolutionary ecology, and plant biology, phenotype determination is governed by the fundamental quantitative formulation:

$$\text{Phenotype } (P) = \text{Genotype } (G) + \text{Environment } (E) + G \times E \text{ Interaction} + \epsilon$$

The original Galiana model evaluated phenotypes exclusively from genomic variant counts ($P = f(G)$). However:
1. Plants are sessile organisms whose phenotypes (flowering time, root architecture, germination, biomass, seed longevity, and ion uptake) exhibit extensive **phenotypic plasticity** governed by environmental gradients.
2. In the AraPheno database (1001 Genomes Project), **73 variables are bioclimatic and meteorological factors** (WorldClim bioclimatic indices Bio1–Bio19, seasonal temperatures, rainfall, potential/actual evapotranspiration, growing season indices) that act as **environmental drivers**, not phenotypic traits.
3. Incorporating environmental factors allows the model to capture **Gene-by-Environment ($G \times E$) cross-regulations** (e.g., drought and cold response genes whose effect sizes dynamically shift across climatic regimes).

---

## 🧬 Architecture Overview

The upgraded model (`GalianaGxE`) features four coordinated neural modules with **explicit Bilinear G×E Interaction** (gate removed for direct multiplicative interaction):

```
[ Genomic Variants ] (27,655 genes x 17 types)
         │
    ┌────▼────┐
    │ G-Module│ (Weight-shared linear layers across all genes: 17 → 50 → 1)
    └────┬────┘
         │  Scalar gene representations (B, 27,655)
    ┌────▼────────┐
    │ g_projection│ (LayerNorm → Dropout → Linear 27,655 → 50 → Tanh)
    └────┬────────┘
         │  z_G (B, 50)
         ├──────────────────────────┐
         │                          ▼
         │                 ┌───────────────────┐
         │                 │ Genome Projection │ a = W_G z_G + b_G  (B, 50)
         │                 └────────┬──────────┘
         │                          ▼
[ Bioclimatic Factors ]        [ Hadamard ]  a ⊙ c  (element-wise)
(73 Climate / WorldClim)            │
         │                          ▼
    ┌────▼────┐                [ LayerNorm ] → z_bilinear  (B, 50)
    │ E-Module│                     ▲
    └────┬────┘                     │
         │  z_E (B, 50)    ┌────────┴──────────┐
         ├─────────────────► Climate Projection│ c = W_E z_E + b_E  (B, 50)
         │                 └───────────────────┘
         │                          │
         ▼                          ▼
    [ raw z_G ]                [ z_bilinear ]
         │                          │
         └──────────┬───────────────┘
                    ▼
       [ Fused Latent Vector ]  (B, 150) : [ z_G, z_E, z_bilinear ]
                    │
               ┌────▼────┐
               │ P-Module│  (150 → 100 → 50 → 215)
               └────┬────┘
                    ▼
       [ 215 Quantitative Traits ]
```

1. **G-Module (Genomic Variant Encoder & Projection):**
   - Applies shared weights across all 27,655 genes ($\text{Linear}(17 \rightarrow 50) \rightarrow \text{LayerNorm} \rightarrow \text{Tanh} \rightarrow \text{Linear}(50 \rightarrow 1)$) to project variant consequence histograms into scalar gene representations.
   - `g_projection` ($\text{LayerNorm}(27655) \rightarrow \text{Dropout}(0.2) \rightarrow \text{Linear}(27655 \rightarrow 50) \rightarrow \text{Tanh}$) maps whole-genome variation into $z_G \in \mathbb{R}^{50}$.
2. **E-Module (Environmental Encoder):**
   - Encodes the 73 bioclimatic variables (WorldClim Bio1–Bio19, evapotranspiration, seasonal temperatures, growing season indicators) through a deep MLP ($\text{LayerNorm}(73) \rightarrow \text{Linear}(73 \rightarrow 64) \rightarrow \text{LayerNorm} \rightarrow \text{GELU} \rightarrow \text{Dropout} \rightarrow \text{Linear}(64 \rightarrow 50) \rightarrow \text{LayerNorm} \rightarrow \text{Tanh}$) into an environmental latent embedding $z_E \in \mathbb{R}^{50}$.
3. **G×E Interaction Module (Bilinear Interaction Only — Gate Removed):**
   - **Genome Projection:** $a = W_G z_G + b_G \in \mathbb{R}^{50}$
   - **Climate Projection:** $c = W_E z_E + b_E \in \mathbb{R}^{50}$
   - **Hadamard Cross-Term:** $z_{\text{bilinear}} = \text{LayerNorm}(a \odot c) \in \mathbb{R}^{50}$
   - **Latent Skip Fusion:** Raw $z_G$ and $z_E$ skip straight to fusion alongside $z_{\text{bilinear}}$, producing the 150-dimensional joint representation $[z_G, z_E, z_{\text{bilinear}}] \in \mathbb{R}^{150}$.
4. **P-Module (Multi-task Phenotype Predictor):**
   - Maps the joint 150-dim genomic-environmental representation through a multi-stage MLP ($\text{Linear}(150 \rightarrow 100) \rightarrow \text{Linear}(100 \rightarrow 50) \rightarrow \text{Linear}(50 \rightarrow 215)$) to predict 215 quantitative *A. thaliana* phenotypes simultaneously.

---

## 📊 Performance & Evaluation

### 1. Real 1001 Genomes WGS Sequencing Evaluation
Evaluated with real whole-genome sequencing calls across 27,655 genes and 73 real WorldClim bioclimatic features:

| Model Architecture | Input Features | Parameters | Test MSE (↓) | Test MAE (↓) | Top Trait Pearson $r$ (↑) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **EnvironmentalOnly** | 73 Bioclimatic Drivers | 19,325 | 1.0657 | 0.6995 | +0.6009 |
| **GalianaGxE (Bilinear G×E)** | **27,655 Genes × 17 Variant Bins + 73 Env** | **1,484,136** | **1.0870** | **0.7022** | **+0.6850** |

#### Top Traits Predicted by GalianaGxE with Real WGS Variants:
- **Days to Flowering Main Effect (`DTFmainEffect2008`):** $r = \mathbf{+0.6850}$ ($\Delta r = +0.0841$)
- **Days to Flowering in Spain (`DTF spain 2009`):** $r = \mathbf{+0.6725}$
- **Plant Rosette Size in Sweden (`SizeLocSweden2009`):** $r = \mathbf{+0.6687}$
- **Days to Flowering in Sweden (`DTF sweden 2009`):** $r = \mathbf{+0.6679}$
- **Summer Planting Flowering Time (`DTFplantingSummer2008`):** $r = \mathbf{+0.6150}$ ($\Delta r = +0.0924$)

#### Top Environmental & Bioclimatic Drivers (Saliency Gradient):
1. `clim-aet8`: August Actual Evapotranspiration
2. `clim-cyclevsgs`: Ratio of Life Cycle Duration to Growing Season
3. `clim-tmin3`: March Minimum Temperature
4. `clim-pet3`: March Potential Evapotranspiration
5. `clim-tmin10`: October Minimum Temperature

---

## 📁 Repository Structure

```
├── data/
│   ├── arapheno_accession_ids.json       # Aligned accession identifiers from AraPheno
│   ├── hdf5_accession_ids.json           # 1,135 accessions from 1001 Genomes HDF5
│   ├── real_arapheno_Y.npy               # Standardized matrix (1,021 accessions x 288 traits)
│   └── real_phenotype_names.json         # Complete trait taxonomy (73 env + 215 phenotypes)
├── galiana/                              # Original baseline Galiana submodule
├── results/                              # Generated figures, loss curves, and CSV performance tables
│   ├── wgs_model_results.png             # Training loss curves and top-20 trait performance
│   ├── wgs_phenotype_performance.csv     # Trait-by-trait Pearson correlation comparison
│   ├── env_model_comparison.png          # 4-model comparative benchmark figure
│   └── ENVIRONMENTAL_UPGRADE_REPORT.md   # Comprehensive scientific report
├── build_wgs_matrix.py                   # Real WGS variant consequence matrix extractor from 1001G HDF5
├── dataset.py                            # Standard Galiana PyTorch Dataset
├── dataset_env.py                        # Dataset splitter separating bioclimatic drivers from plant traits
├── model.py                              # Baseline Galiana architecture (genotype-only)
├── model_env.py                          # Upgraded GalianaGxE, EnvironmentalOnly, and Additive models
├── run_pipeline.py                       # End-to-end quick training and evaluation pipeline
├── run_wgs_pipeline.py                   # Full WGS sequencing + environmental evaluation pipeline
├── train.py                              # General model training script with CLI options
├── train_env.py                          # Multi-architecture comparative training runner
└── generate_env_results.py               # 4-model comparative benchmark script
```

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/Rahul8025p/GalianaGxE-AraPheno.git
cd GalianaGxE-AraPheno
pip install torch numpy scikit-learn matplotlib h5py
```

### 2. Extract WGS Variant Features (Optional / If starting from raw HDF5)
If you have downloaded the 1001 Genomes imputed SNP HDF5 file:
```bash
python build_wgs_matrix.py
```
This maps the 10.7M SNPs to TAIR10 gene coordinates and produces `data/real_wgs_genotype_100.npy`.

### 3. Run the Full Real WGS Pipeline
Train and evaluate GalianaGxE against the EnvironmentalOnly baseline:
```bash
python run_wgs_pipeline.py
```
Outputs:
- Loss curves and top-20 trait correlation bar chart: `results/wgs_model_results.png`
- Trait-by-trait performance table: `results/wgs_phenotype_performance.csv`

### 4. Run the Standard Pipeline
```bash
# Train full GxE model with environmental factors
python train.py --use-env --epochs 15

# Train baseline genotype-only model
python train.py --no-env --epochs 15
```

### 5. Run 4-Model Comparative Benchmark
```bash
python generate_env_results.py
```

---

## 📖 Citation & References

If you build upon this work or use the models, please cite the foundational literature:

- **Galiana Architecture:**
  > Raimondi, D., Corso, M., Fariselli, P., & Moreau, Y. (2021). *From genotype to phenotype in Arabidopsis thaliana: in-silico genome interpretation predicts 288 phenotypes from sequencing data*. **Nucleic Acids Research**, 49(14), 7859–7871.
- **1001 Genomes Project:**
  > 1001 Genomes Consortium. (2016). *1,135 Genomes Reveal the Global Pattern of Polymorphism in Arabidopsis thaliana*. **Cell**, 166(2), 481–491.
- **AraPheno:**
  > Seren, Ü., Grimm, D., Fitz, J., et al. (2017). *AraPheno: a public database for Arabidopsis thaliana phenotypes*. **Nucleic Acids Research**, 45(D1), D1054–D1059.
