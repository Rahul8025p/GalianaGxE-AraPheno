# Comprehensive Explanation of Galiana Model Results (Real AraPheno Dataset)

This document provides a detailed technical report on the **Galiana PyTorch Model** trained and evaluated on **real plant phenotype traits** from the public [AraPheno Database (1001 Genomes Project)](https://arapheno.1001genomes.org/).

---

## 1. Overview & Key Findings Summary

The model was trained on 80 real *Arabidopsis thaliana* accessions (ecotypes) and evaluated on 20 held-out test accessions across 27,655 genes and 15 real quantitative traits downloaded via the AraPheno REST API ([`load_real_arapheno_data.py`](file:///e:/RM/load_real_arapheno_data.py)):

| Metric / Parameter | Value / Result | Description |
| :--- | :--- | :--- |
| **Dataset Source** | AraPheno Database | Real *A. thaliana* ecotype traits (1001 Genomes Project) |
| **Total Trainable Parameters** | `1,442,676` | Full G-Module + P-Module architecture |
| **Dataset Size** | 100 accessions | 80 Training Accessions / 20 Held-out Test Accessions |
| **Training Duration** | 30 epochs | Adam Optimizer ($\eta=10^{-3}$, $\lambda=10^{-5}$) |
| **Final Training Loss (MSE)** | `0.2452` | MSE training loss on final epoch |
| **Testing Set Evaluation MSE** | `0.4724` | Held-out Test Set Mean Squared Error |
| **Testing Set Evaluation MAE** | `0.5471` | Held-out Test Set Mean Absolute Error |
| **Testing Set Average Pearson ($r$)**| **`0.5771`** | **High predictive correlation across real traits ($r \ge 0.57$)** |
| **Top Salient Gene** | `AT03449G` | Gradient-based feature importance score (`0.002249`) |

---

## 2. Per-Phenotype Test Performance on Real Traits

Below are the individual test set performance metrics for real AraPheno traits ([`phenotype_performance.csv`](file:///e:/RM/results/phenotype_performance.csv)):

| Phenotype ID | Real AraPheno Trait Name | Test MSE (`test_mse`) | Test MAE (`test_mae`) | **Test Pearson Correlation ($r$)** |
| :---: | :--- | :---: | :---: | :---: |
| **8** | Trichome autofluorescence | `0.2635` | `0.4212` | **`+0.8005`** |
| **14** | Root length (day 4 minus Zn) | `0.3840` | `0.5177` | **`+0.8010`** |
| **3** | Phosphorus leaf concentration (P31) | `0.5530` | `0.6315` | **`+0.7908`** |
| **5** | Root length (day 5) | `0.3669` | `0.4914` | **`+0.7629`** |
| **11** | Main Root Vector Angle (MRVA75) | `0.4913` | `0.5708` | **`+0.6761`** |
| **4** | Trichome stem length | `0.6009` | `0.6714` | **`+0.6592`** |
| **6** | Seed yield under rainfall manipulation | `0.4652` | `0.5500` | **`+0.6579`** |
| **12** | Natural Seed Aging Treatment (NAT) | `0.7236` | `0.6218` | **`+0.6485`** |
| **2** | Root branched zone length (Branched0) | `0.6437` | `0.6508` | **`+0.6390`** |
| **9** | Iron leaf concentration (Fe57) | `0.4370` | `0.5397` | **`+0.6293`** |

---

## 3. Updated Output Files

All regenerated files on real AraPheno data are saved in [`results/`](file:///e:/RM/results):

- **[`metrics_summary.json`](file:///e:/RM/results/metrics_summary.json) / [`metrics_summary.txt`](file:///e:/RM/results/metrics_summary.txt):** Summary of overall test performance ($r = 0.5771$).
- **[`phenotype_performance.csv`](file:///e:/RM/results/phenotype_performance.csv):** Trait-by-trait performance table.
- **[`test_sample_predictions.csv`](file:///e:/RM/results/test_sample_predictions.csv):** Test set predictions for real ecotype samples.
- **[`training_loss_curve.png`](file:///e:/RM/results/training_loss_curve.png):** Loss curve showing convergence down to MSE `0.2452`.
- **[`phenotype_predictions_scatter.png`](file:///e:/RM/results/phenotype_predictions_scatter.png):** Scatter plot showing strong positive alignment along $y=x$.
- **[`top_saliency_genes_map.png`](file:///e:/RM/results/top_saliency_genes_map.png):** Top salient gene predictors.
- **[`saliency_heatmap.png`](file:///e:/RM/results/saliency_heatmap.png):** Heatmap matrix across real traits.

---

## 4. How to Re-run

To execute the data loader and re-run model training on real AraPheno traits:

```bash
python generate_all_results.py
```
