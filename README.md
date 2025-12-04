# Predicting Transcription Factor Perturbations with Genomic Embeddings

**10-701: Introduction to Machine Learning - Final Project**
**Carnegie Mellon University, Computational Biology Department**
**December 2025**

## Team Members

- **Palak Chaudhry** - pchaudhr@andrew.cmu.edu
- **Julie Bocetti** - jbocetti@andrew.cmu.edu
- **Yu-Hsiang Chen** - yuhsianc@andrew.cmu.edu
- **Marios Gavrielatos** - mgavriel@andrew.cmu.edu

## Project Overview

This project predicts genome-wide gene expression responses to transcription factor (TF) perturbations using genomic sequence embeddings from the Enformer model. We trained multiple regression models (ElasticNet, XGBoost, MLP, kNN) on CRISPRa Perturb-seq data from human fibroblasts to predict mean log-fold expression changes across 231 TF perturbations.

**Key Findings:**
- ElasticNet with 1,024-dimensional embeddings achieved the best performance
- TSS-centered sequences outperformed full gene sequences
- 38.5% of perturbations showed significant predictive performance (Pearson r > 0.1)
- Clustering analysis revealed biologically meaningful TF groupings

---

# Project Organization

This directory contains all analysis scripts and results for the Gene Perturbation Prediction project.

## Directory Structure

```
project/
├── analysis/                           # All analysis scripts and results
│   ├── 0_data/                        # Data preprocessing scripts
│   ├── 1_embeddings/                  # Enformer embedding generation
│   ├── 2_regression/                  # Regression model training
│   ├── 3_weight_analysis/             # Model weight analysis and clustering
│   └── 4_compare_regression/          # Model performance comparison
├── embeddings/                         # Generated Enformer embeddings (.npy files)
├── perturb_seq_data/                   # CRISPRa Perturb-seq data (download from [Zenodo](https://zenodo.org/records/15200179#:~:text=from%20overloaded%20droplets%3A-,fibroblast_CRISPRa_mean_pop.h5ad,-Files))
├── genomic_sequences.tar.gz            # Compressed genomic sequence files
└── genomic_sequences/                  # Extracted genomic sequences (FASTA files)
```

---

## Data Directories

### embeddings/
Contains pre-computed Enformer embeddings for all genes:
- `embeddings_enformer_tss_full.npy` - Full 3,072-dim TSS embeddings from Enformer trunk
- `embeddings_enformer_tss_1024.npy` - PCA-reduced 1,024-dim TSS embeddings
- `embeddings_enformer_gencode.v49.pc_transcripts_full.npy` - Full 3,072-dim GENCODE embeddings
- `embeddings_enformer_gencode.v49.pc_transcripts_1024.npy` - PCA-reduced 1,024-dim GENCODE embeddings

### perturb_seq_data/
Contains processed CRISPRa Perturb-seq data from human fibroblasts:
- `fibroblast_CRISPRa_mean_pop.h5ad` - Original single-cell data
- `fibroblast_CRISPRa_mean_pop_mean.csv` - Mean expression changes per TF (231 perturbations)
- `fibroblast_CRISPRa_mean_pop_median.csv` - Median expression changes per TF

### genomic_sequences.tar.gz
Compressed archive containing genomic sequence FASTA files (stored with Git LFS):
- `gencode.v49.pc_transcripts.gene_names.fa` - Protein-coding gene sequences
- `TSS.fa` - Transcription start site-centered sequences

**To extract**: `tar -xzf genomic_sequences.tar.gz`

---

## 1_embeddings/

Generate genomic sequence embeddings using pre-trained Enformer model.

```
1_embeddings/
├── notebooks/
   ├── enformer_tss.ipynb                              # 32-dim TSS embeddings
   ├── enformer_tss_1024.ipynb                         # 1024-dim TSS embeddings
   ├── enformer_gencode.v49.pc_transcripts.ipynb       # 32-dim GENCODE
   └── enformer_gencode.v49.pc_transcripts_1024.ipynb  # 1024-dim GENCODE

```

**Outputs**: Embeddings saved to `/embeddings/` directory at project root

---

## 2_regression/

Train regression models on genomic embeddings to predict perturbation responses.

### ElasticNet (32-dim embeddings)
```
2_regression/elasticnet/
├── scripts/
│   ├── glmnet_script_param.R                    # Hyperparameter tuning
│   ├── glmnet_script_eval_with_best_params.R    # Final evaluation
│   ├── run_glmnet_gencode.sbatch                # SLURM job for GENCODE
│   ├── run_glmnet_tss.sbatch                    # SLURM job for TSS
│   ├── run_elasticnet_eval_gencode.sh           # Evaluation script
│   └── run_elasticnet_eval_tss.sh
└── results/
    ├── gencode_coefficients.csv                 # Model weights
    ├── gencode_eval_summary.csv
    ├── gencode_eval_with_best_params.csv
    ├── gencode_results_cv.csv
    └── tss_* (same structure)
```

### ElasticNet_1024 (1024-dim embeddings)
```
2_regression/elasticnet_1024/
├── scripts/
│   ├── glmnet_script_param_1024.R
│   ├── glmnet_script_eval_with_best_params_1024.R
│   ├── run_glmnet_gencode_1024.sbatch
│   ├── run_glmnet_tss_1024.sbatch
│   ├── run_elasticnet_eval_gencode_1024.sh
│   ├── run_elasticnet_eval_tss_1024.sh
└── results/
    └── (same structure as elasticnet/)
```

### XGBoost
```
2_regression/xgboost/
├── scripts/
│   ├── XGBRegressor_Enformer_param.py
│   ├── XGBRegressor_Enformer_eval_with_best_params.py
│   ├── run_xgboost_gencode.sbatch
│   ├── run_xgboost_tss.sbatch
│   ├── run_xgboost_eval_gencode.sh
│   └── run_xgboost_eval_tss.sh
└── results/
    ├── gencode_eval_summary.csv
    ├── gencode_eval_with_best_params.csv
    ├── gencode_feature_importance.csv
    ├── gencode_results.csv
    ├── gencode_results.json
    └── tss_* (same structure)
```

### MLP
```
2_regression/mlp/
├── scripts/
│   ├── mlp.py
│   └── mlp_slurm.sbatch
├── notebooks/
│   └── test.ipynb
└── results/
    ├── gencode_heldout_embeddings.csv
    ├── gencode_heldout_embeddings.json
    ├── gencode_heldout_results.csv
    ├── gencode_heldout_results.json
    ├── gencode_results_summary.csv
    └── tss_* (same structure)
```

---

## 3_weight_analysis/

Analyze and cluster model weights to identify biological patterns.

```
3_weight_analysis/
├── notebooks/
│   ├── elasticnet_DimReduction_Clustering.ipynb        # 32-dim analysis
│   ├── elasticnet_DimReduction_Clustering_1024.ipynb   # 1024-dim analysis
│   └── xgboost_DimReduction_Clustering.ipynb
├── figures/
│   ├── 32dim/
│   │   ├── gencode_clustering_2d.png
│   │   ├── gencode_clustering_3d.png
│   │   ├── tss_clustering_2d.png
│   │   └── tss_clustering_3d.png
│   ├── 1024dim/
│   │   └── (same structure)
│   ├── interactive/
│   │   ├── elasticnet_gencode_gmm_3d_umap.html
│   │   ├── elasticnet_gencode_kmeans_3d_umap.html
│   │   ├── elasticnet_tss_gmm_3d_umap.html
│   │   └── elasticnet_tss_kmeans_3d_umap.html
│   └── xgboost_* (XGBoost figures)
└── cluster_assignments/
    ├── elasticnet_gencode_2d_k4.json
    ├── elasticnet_gencode_3d_k4.json
    ├── elasticnet_tss_2d_k4.json
    ├── elasticnet_tss_3d_k4.json
    └── elasticnet_1024_* (1024-dim assignments)
```

---

## 4_compare_regression/

Compare performance across all models and embedding types.

```
4_compare_regression/
├── notebooks/
│   └── comparisons.ipynb                        # Main comparison analysis
├── figures/
│   ├── model_comparison.png                     # All models compared
│   ├── elasticnet_comparison_detailed.png       # ElasticNet vs ElasticNet_1024
│   ├── elasticnet_vs_1024_performance_changes.png
│   └── high_performers_analysis.png
├── interactive/
│   └── elasticnet_comparison_interactive.html   # Interactive plotly visualization
└── results/
    ├── unified_results_gencode.csv              # All model results (GENCODE)
    ├── unified_results_tss.csv                  # All model results (TSS)
    ├── gencode_elasticnet_improvements.csv
    ├── tss_elasticnet_improvements.csv
    ├── high_performers_gencode.csv
    └── high_performers_tss.csv
```

---

## File Naming Conventions

- **Embedding type**: `gencode` or `tss`
- **Model**: `elasticnet`, `elasticnet_1024`, `xgboost`, `mlp`, `knn`
- **Output type**: `coefficients`, `results`, `eval_summary`, `feature_importance`, etc.

---

