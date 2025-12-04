#!/usr/bin/env python

import pandas as pd
import numpy as np
import scanpy as sc
from scipy import io
import anndata as ad
import matplotlib.pyplot as plt

adata_mean_pop_path = "../../perturb_seq_data/fibroblast_CRISPRa_mean_pop.h5ad"
fibroblast_CRISPRa_mean_pop = sc.read_h5ad(adata_mean_pop_path)

df_X = pd.DataFrame(fibroblast_CRISPRa_mean_pop.X,
                    index=fibroblast_CRISPRa_mean_pop.obs['target_gene'],
                    columns=fibroblast_CRISPRa_mean_pop.var_names)

index_counts = df_X.index.value_counts()
all_six = (index_counts == 6).all()
print("All index values are repeated 6 times:", all_six)

not_six = index_counts[index_counts != 6]
print("Indexes not repeated 6 times:\n", not_six)

df_X = pd.DataFrame(fibroblast_CRISPRa_mean_pop.X,
                    index=fibroblast_CRISPRa_mean_pop.obs_names,
                    columns=fibroblast_CRISPRa_mean_pop.var_names)
meta = fibroblast_CRISPRa_mean_pop.obs.copy()

grouped_all = df_X.groupby(meta['target_gene'], sort=True)
X_target_median = grouped_all.median()
X_target_mean = grouped_all.mean()
X_target_std = grouped_all.std(ddof=0)
X_target_zscore = X_target_mean / (X_target_std + 1e-8)

X_target_mean.to_csv("../../perturb_seq_data/fibroblast_CRISPRa_mean_pop_mean.csv")
X_target_median.to_csv("../../perturb_seq_data/fibroblast_CRISPRa_mean_pop_median.csv")
