#!/usr/bin/env python
# Converted from comparisons.ipynb

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# File paths
gencode_input = '../../2_regression/mlp/results/gencode_heldout_results.csv'
tss_input = '../../2_regression/mlp/results/tss_heldout_results.csv'

gencode_output = '../../2_regression/mlp/results/gencode_results_summary.csv'
tss_output = '../../2_regression/mlp/results/tss_results_summary.csv'

# Process gencode file - try reading as CSV
df_gencode = pd.read_csv(gencode_input)
print("Columns available:", df_gencode.columns.tolist())
df_gencode_summary = df_gencode[['perturbation', 'rmse_test', 'pearson_corr_test', 'cv_rmse_mean']]
df_gencode_summary.to_csv(gencode_output, index=False)
print(f"Gencode summary saved to: {gencode_output}")
print(f"Shape: {df_gencode_summary.shape}")
print(df_gencode_summary.head())

# Process TSS file
df_tss = pd.read_csv(tss_input)
df_tss_summary = df_tss[['perturbation', 'rmse_test', 'pearson_corr_test', 'cv_rmse_mean']]
df_tss_summary.to_csv(tss_output, index=False)
print(f"\nTSS summary saved to: {tss_output}")
print(f"Shape: {df_tss_summary.shape}")
print(df_tss_summary.head())


# File paths
gencode_json = '../../2_regression/KNN/results_gencode.json'
tss_json = '../../2_regression/KNN/results_TSS.json'

gencode_output = '../../2_regression/KNN/results_gencode.csv'
tss_output = '../../2_regression/KNN/results_TSS.csv'

# Load and process gencode
with open(gencode_json, 'r') as f:
    data_gencode = json.load(f)

df_gencode = pd.DataFrame(data_gencode)
# Rename columns to match other models - convert MSE to RMSE
df_gencode['RMSE'] = np.sqrt(df_gencode['MSE'])
df_gencode_output = df_gencode[['TF', 'k', 'MSE', 'RMSE', 'Pearson', 'Spearman']]
df_gencode_output.to_csv(gencode_output, index=False)
print(f"Gencode CSV saved to: {gencode_output}")
print(f"Shape: {df_gencode_output.shape}")
print(df_gencode_output.head())

# Load and process TSS
with open(tss_json, 'r') as f:
    data_tss = json.load(f)

df_tss = pd.DataFrame(data_tss)
# Convert MSE to RMSE
df_tss['RMSE'] = np.sqrt(df_tss['MSE'])
df_tss_output = df_tss[['TF', 'k', 'MSE', 'RMSE', 'Pearson', 'Spearman']]
df_tss_output.to_csv(tss_output, index=False)
print(f"\nTSS CSV saved to: {tss_output}")
print(f"Shape: {df_tss_output.shape}")
print(df_tss_output.head())

# File paths

# KNN
knn_gencode = '../../2_regression/KNN/results_gencode.json'
knn_tss = '../../2_regression/KNN/results_TSS.json'

# XGBoost
xgb_gencode = '../../2_regression/xgboost/results/gencode_eval_summary.csv'
xgb_tss = '../../2_regression/xgboost/results/tss_eval_summary.csv'

# MLP
mlp_gencode = '../../2_regression/mlp/results/gencode_results_summary.csv'
mlp_tss = '../../2_regression/mlp/results/tss_results_summary.csv'

# ElasticNet - use the eval_with_best_params files instead
enet_gencode = '../../2_regression/elasticnet/results/gencode_eval_with_best_params.csv'
enet_tss = '../../2_regression/elasticnet/results/tss_eval_with_best_params.csv'

# ElasticNet - use the eval_with_best_params files instead
enet_1024_gencode = '../../2_regression/elasticnet_1024/results/gencode_eval_with_best_params.csv'
enet_1024_tss = '../../2_regression/elasticnet_1024/results/tss_eval_with_best_params.csv'

# Output
output_gencode = '../results/unified_results_gencode.csv'
output_tss = '../results/unified_results_tss.csv'


# Load KNN data
with open(knn_gencode, 'r') as f:
    knn_g = pd.DataFrame(json.load(f))
    knn_g['RMSE'] = np.sqrt(knn_g['MSE'])
    knn_g = knn_g.rename(columns={'TF': 'perturbation', 'Pearson': 'pearson'})
    knn_g = knn_g[['perturbation', 'RMSE', 'pearson']]
    knn_g = knn_g.rename(columns={'RMSE': 'KNN_RMSE', 'pearson': 'KNN_Pearson'})

with open(knn_tss, 'r') as f:
    knn_t = pd.DataFrame(json.load(f))
    knn_t['RMSE'] = np.sqrt(knn_t['MSE'])
    knn_t = knn_t.rename(columns={'TF': 'perturbation', 'Pearson': 'pearson'})
    knn_t = knn_t[['perturbation', 'RMSE', 'pearson']]
    knn_t = knn_t.rename(columns={'RMSE': 'KNN_RMSE', 'pearson': 'KNN_Pearson'})

# Load XGBoost data
xgb_g = pd.read_csv(xgb_gencode)
xgb_g = xgb_g[['perturbation', 'mean_rmse', 'mean_pearson']]
xgb_g = xgb_g.rename(columns={'mean_rmse': 'XGBoost_RMSE', 'mean_pearson': 'XGBoost_Pearson'})

xgb_t = pd.read_csv(xgb_tss)
xgb_t = xgb_t[['perturbation', 'mean_rmse', 'mean_pearson']]
xgb_t = xgb_t.rename(columns={'mean_rmse': 'XGBoost_RMSE', 'mean_pearson': 'XGBoost_Pearson'})

# Load MLP data
mlp_g = pd.read_csv(mlp_gencode)
mlp_g = mlp_g[['perturbation', 'rmse_test', 'pearson_corr_test']]
mlp_g = mlp_g.rename(columns={'rmse_test': 'MLP_RMSE', 'pearson_corr_test': 'MLP_Pearson'})

mlp_t = pd.read_csv(mlp_tss)
mlp_t = mlp_t[['perturbation', 'rmse_test', 'pearson_corr_test']]
mlp_t = mlp_t.rename(columns={'rmse_test': 'MLP_RMSE', 'pearson_corr_test': 'MLP_Pearson'})

# Load ElasticNet data - aggregate per-fold results
enet_g_raw = pd.read_csv(enet_gencode)
# Group by perturbation and calculate mean RMSE and Pearson
enet_g = enet_g_raw.groupby('perturbation').agg({
    'rmse': 'mean',
    'pearson_corr': 'mean'
}).reset_index()
enet_g = enet_g.rename(columns={'rmse': 'ElasticNet_RMSE', 'pearson_corr': 'ElasticNet_Pearson'})

enet_t_raw = pd.read_csv(enet_tss)
enet_t = enet_t_raw.groupby('perturbation').agg({
    'rmse': 'mean',
    'pearson_corr': 'mean'
}).reset_index()
enet_t = enet_t.rename(columns={'rmse': 'ElasticNet_RMSE', 'pearson_corr': 'ElasticNet_Pearson'})

# Load ElasticNet data - aggregate per-fold results
enet_1024_g_raw = pd.read_csv(enet_1024_gencode)
# Group by perturbation and calculate mean RMSE and Pearson
enet_1024_g = enet_1024_g_raw.groupby('perturbation').agg({
    'rmse': 'mean',
    'pearson_corr': 'mean'
}).reset_index()
enet_1024_g = enet_1024_g.rename(columns={'rmse': 'ElasticNet_1024_RMSE', 'pearson_corr': 'ElasticNet_1024_Pearson'})

enet_1024_t_raw = pd.read_csv(enet_1024_tss)
enet_1024_t = enet_1024_t_raw.groupby('perturbation').agg({
    'rmse': 'mean',
    'pearson_corr': 'mean'
}).reset_index()
enet_1024_t = enet_1024_t.rename(columns={'rmse': 'ElasticNet_1024_RMSE', 'pearson_corr': 'ElasticNet_1024_Pearson'})


# Merge all datasets - GENCODE
unified_gencode = knn_g.merge(xgb_g, on='perturbation', how='outer')
unified_gencode = unified_gencode.merge(mlp_g, on='perturbation', how='outer')
unified_gencode = unified_gencode.merge(enet_g, on='perturbation', how='outer')
unified_gencode = unified_gencode.merge(enet_1024_g, on='perturbation', how='outer')
# Reorder columns
cols_order = ['perturbation', 
              'KNN_RMSE', 'KNN_Pearson',
              'XGBoost_RMSE', 'XGBoost_Pearson',
              'MLP_RMSE', 'MLP_Pearson',
              'ElasticNet_RMSE', 'ElasticNet_Pearson',
              'ElasticNet_1024_RMSE','ElasticNet_1024_Pearson']

unified_gencode = unified_gencode[cols_order]

# Save
unified_gencode.to_csv(output_gencode, index=False)
print(f"Gencode unified results saved to: {output_gencode}")
print(f"Shape: {unified_gencode.shape}")
print("\nFirst few rows:")
print(unified_gencode.head(10))
print("\nSummary statistics:")
print(unified_gencode.describe())


# Merge all datasets - TSS
unified_tss = knn_t.merge(xgb_t, on='perturbation', how='outer')
unified_tss = unified_tss.merge(mlp_t, on='perturbation', how='outer')
unified_tss = unified_tss.merge(enet_t, on='perturbation', how='outer')
unified_tss = unified_tss.merge(enet_1024_t, on='perturbation', how='outer')

# Reorder columns
unified_tss = unified_tss[cols_order]

# Save
unified_tss.to_csv(output_tss, index=False)
print(f"\n\nTSS unified results saved to: {output_tss}")
print(f"Shape: {unified_tss.shape}")
print("\nFirst few rows:")
print(unified_tss.head(10))
print("\nSummary statistics:")
print(unified_tss.describe())

# STEP 1: Regenerate unified CSV with corrected ElasticNet data

print("STEP 1: REGENERATING UNIFIED CSV FILES WITH CORRECTED ELASTICNET")

# File paths

# KNN
knn_gencode = '../../2_regression/KNN/results_gencode.json'
knn_tss = '../../2_regression/KNN/results_TSS.json'

# XGBoost
xgb_gencode = '../../2_regression/xgboost/results/gencode_eval_summary.csv'
xgb_tss = '../../2_regression/xgboost/results/tss_eval_summary.csv'

# MLP
mlp_gencode = '../../2_regression/mlp/results/gencode_results_summary.csv'
mlp_tss = '../../2_regression/mlp/results/tss_results_summary.csv'

# ElasticNet - use the eval_with_best_params files instead
enet_gencode = '../../2_regression/elasticnet/results/gencode_eval_with_best_params.csv'
enet_tss = '../../2_regression/elasticnet/results/tss_eval_with_best_params.csv'

# ElasticNet - use the eval_with_best_params files instead
enet_1024_gencode = '../../2_regression/elasticnet_1024/results/gencode_eval_with_best_params.csv'
enet_1024_tss = '../../2_regression/elasticnet_1024/results/tss_eval_with_best_params.csv'

# Output
output_gencode = '../results/unified_results_gencode.csv'
output_tss = '../results/unified_results_tss.csv'


# Load KNN
with open(knn_gencode, 'r') as f:
    knn_g = pd.DataFrame(json.load(f))
    knn_g['RMSE'] = np.sqrt(knn_g['MSE'])
    knn_g = knn_g.rename(columns={'TF': 'perturbation', 'Pearson': 'pearson'})
    knn_g = knn_g[['perturbation', 'RMSE', 'pearson']]
    knn_g = knn_g.rename(columns={'RMSE': 'KNN_RMSE', 'pearson': 'KNN_Pearson'})

with open(knn_tss, 'r') as f:
    knn_t = pd.DataFrame(json.load(f))
    knn_t['RMSE'] = np.sqrt(knn_t['MSE'])
    knn_t = knn_t.rename(columns={'TF': 'perturbation', 'Pearson': 'pearson'})
    knn_t = knn_t[['perturbation', 'RMSE', 'pearson']]
    knn_t = knn_t.rename(columns={'RMSE': 'KNN_RMSE', 'pearson': 'KNN_Pearson'})

# Load XGBoost
xgb_g = pd.read_csv(xgb_gencode)
xgb_g = xgb_g[xgb_g['perturbation'] != 'Overall']  # Remove overall row
xgb_g = xgb_g[['perturbation', 'mean_rmse', 'mean_pearson']]
xgb_g = xgb_g.rename(columns={'mean_rmse': 'XGBoost_RMSE', 'mean_pearson': 'XGBoost_Pearson'})

xgb_t = pd.read_csv(xgb_tss)
xgb_t = xgb_t[xgb_t['perturbation'] != 'Overall']
xgb_t = xgb_t[['perturbation', 'mean_rmse', 'mean_pearson']]
xgb_t = xgb_t.rename(columns={'mean_rmse': 'XGBoost_RMSE', 'mean_pearson': 'XGBoost_Pearson'})

# Load MLP
mlp_g = pd.read_csv(mlp_gencode)
mlp_g = mlp_g[['perturbation', 'rmse_test', 'pearson_corr_test']]
mlp_g = mlp_g.rename(columns={'rmse_test': 'MLP_RMSE', 'pearson_corr_test': 'MLP_Pearson'})

mlp_t = pd.read_csv(mlp_tss)
mlp_t = mlp_t[['perturbation', 'rmse_test', 'pearson_corr_test']]
mlp_t = mlp_t.rename(columns={'rmse_test': 'MLP_RMSE', 'pearson_corr_test': 'MLP_Pearson'})


# Load ElasticNet data - aggregate per-fold results
enet_g_raw = pd.read_csv(enet_gencode)
# Group by perturbation and calculate mean RMSE and Pearson
enet_g = enet_g_raw.groupby('perturbation').agg({
    'rmse': 'mean',
    'pearson_corr': 'mean'
}).reset_index()
enet_g = enet_g.rename(columns={'rmse': 'ElasticNet_RMSE', 'pearson_corr': 'ElasticNet_Pearson'})

enet_t_raw = pd.read_csv(enet_tss)
enet_t = enet_t_raw.groupby('perturbation').agg({
    'rmse': 'mean',
    'pearson_corr': 'mean'
}).reset_index()
enet_t = enet_t.rename(columns={'rmse': 'ElasticNet_RMSE', 'pearson_corr': 'ElasticNet_Pearson'})

# Load ElasticNet data - aggregate per-fold results
enet_1024_g_raw = pd.read_csv(enet_1024_gencode)
# Group by perturbation and calculate mean RMSE and Pearson
enet_1024_g = enet_1024_g_raw.groupby('perturbation').agg({
    'rmse': 'mean',
    'pearson_corr': 'mean'
}).reset_index()
enet_1024_g = enet_1024_g.rename(columns={'rmse': 'ElasticNet_1024_RMSE', 'pearson_corr': 'ElasticNet_1024_Pearson'})

enet_1024_t_raw = pd.read_csv(enet_1024_tss)
enet_1024_t = enet_1024_t_raw.groupby('perturbation').agg({
    'rmse': 'mean',
    'pearson_corr': 'mean'
}).reset_index()
enet_1024_t = enet_1024_t.rename(columns={'rmse': 'ElasticNet_1024_RMSE', 'pearson_corr': 'ElasticNet_1024_Pearson'})

# Merge all - GENCODE
unified_gencode = knn_g.merge(xgb_g, on='perturbation', how='outer')
unified_gencode = unified_gencode.merge(mlp_g, on='perturbation', how='outer')
unified_gencode = unified_gencode.merge(enet_g, on='perturbation', how='outer')
unified_gencode = unified_gencode.merge(enet_1024_g, on='perturbation', how='outer')


cols_order = ['perturbation', 
              'KNN_RMSE', 'KNN_Pearson',
              'XGBoost_RMSE', 'XGBoost_Pearson',
              'MLP_RMSE', 'MLP_Pearson',
              'ElasticNet_RMSE', 'ElasticNet_Pearson', 
              'ElasticNet_1024_RMSE','ElasticNet_1024_Pearson']

unified_gencode = unified_gencode[cols_order]
unified_gencode.to_csv(output_gencode, index=False)
print(f"Gencode unified saved: {output_gencode}")

# Merge all - TSS
unified_tss = knn_t.merge(xgb_t, on='perturbation', how='outer')
unified_tss = unified_tss.merge(mlp_t, on='perturbation', how='outer')
unified_tss = unified_tss.merge(enet_t, on='perturbation', how='outer')
unified_tss = unified_tss.merge(enet_1024_t, on='perturbation', how='outer')

unified_tss = unified_tss[cols_order]
unified_tss.to_csv(output_tss, index=False)
print(f"TSS unified saved: {output_tss}")



# STEP 2: Load and analyze
print("STEP 2: MODEL COMPARISON ANALYSIS")

df_gencode = pd.read_csv(output_gencode)
df_tss = pd.read_csv(output_tss)

print("\n1. CALCULATING STATISTICS...")

models = ['KNN', 'XGBoost', 'MLP', 'ElasticNet', 'ElasticNet_1024']

summary_stats = []
for model in models:
    rmse_col = f'{model}_RMSE'
    pearson_col = f'{model}_Pearson'
    
    stats = {
        'Model': model,
        'Mean_RMSE': df_gencode[rmse_col].mean(),
        'Std_RMSE': df_gencode[rmse_col].std(),
        'Median_RMSE': df_gencode[rmse_col].median(),
        'Mean_Pearson': df_gencode[pearson_col].mean(),
        'Std_Pearson': df_gencode[pearson_col].std(),
        'Median_Pearson': df_gencode[pearson_col].median(),
    }
    summary_stats.append(stats)

summary_df_gencode = pd.DataFrame(summary_stats)

summary_stats_tss = []
for model in models:
    rmse_col = f'{model}_RMSE'
    pearson_col = f'{model}_Pearson'
    
    stats = {
        'Model': model,
        'Mean_RMSE': df_tss[rmse_col].mean(),
        'Std_RMSE': df_tss[rmse_col].std(),
        'Median_RMSE': df_tss[rmse_col].median(),
        'Mean_Pearson': df_tss[pearson_col].mean(),
        'Std_Pearson': df_tss[pearson_col].std(),
        'Median_Pearson': df_tss[pearson_col].median(),
    }
    summary_stats_tss.append(stats)

summary_df_tss = pd.DataFrame(summary_stats_tss)

# Print summary statistics
print("\n" + "="*80)
print("GENCODE EMBEDDINGS - SUMMARY STATISTICS")
print("="*80)
print(summary_df_gencode.to_string(index=False))

print("\n" + "="*80)
print("TSS EMBEDDINGS - SUMMARY STATISTICS")
print("="*80)
print(summary_df_tss.to_string(index=False))

# Calculate best models per perturbation
rmse_cols = [f'{model}_RMSE' for model in models]
df_gencode['Best_RMSE_Model'] = df_gencode[rmse_cols].idxmin(axis=1).str.replace('_RMSE', '')
df_tss['Best_RMSE_Model'] = df_tss[rmse_cols].idxmin(axis=1).str.replace('_RMSE', '')

pearson_cols = [f'{model}_Pearson' for model in models]
df_gencode['Best_Pearson_Model'] = df_gencode[pearson_cols].idxmax(axis=1).str.replace('_Pearson', '')
df_tss['Best_Pearson_Model'] = df_tss[pearson_cols].idxmax(axis=1).str.replace('_Pearson', '')

# Print win counts
print("\n" + "="*80)
print("GENCODE EMBEDDINGS - WIN COUNTS")
print("="*80)
print("\nBest RMSE Model Counts:")
rmse_counts_gencode = df_gencode['Best_RMSE_Model'].value_counts()
rmse_counts_gencode = rmse_counts_gencode.reindex(models, fill_value=0)
print(rmse_counts_gencode.to_string())

print("\nBest Pearson Model Counts:")
pearson_counts_gencode = df_gencode['Best_Pearson_Model'].value_counts()
pearson_counts_gencode = pearson_counts_gencode.reindex(models, fill_value=0)
print(pearson_counts_gencode.to_string())

print("\n" + "="*80)
print("TSS EMBEDDINGS - WIN COUNTS")
print("="*80)
print("\nBest RMSE Model Counts:")
rmse_counts_tss = df_tss['Best_RMSE_Model'].value_counts()
rmse_counts_tss = rmse_counts_tss.reindex(models, fill_value=0)
print(rmse_counts_tss.to_string())

print("\nBest Pearson Model Counts:")
pearson_counts_tss = df_tss['Best_Pearson_Model'].value_counts()
pearson_counts_tss = pearson_counts_tss.reindex(models, fill_value=0)
print(pearson_counts_tss.to_string())


# 3. Visualizations - Updated with TSS distributions
print("2. CREATING VISUALIZATIONS...")

# Create a 3x4 grid to show all comparisons
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle('Model Comparison: Gencode vs TSS', fontsize=18, fontweight='bold')

# Row 1: Bar plots with error bars
# Plot 1: RMSE - Gencode
ax = axes[0, 0]
rmse_means_gencode = [df_gencode[f'{model}_RMSE'].mean() for model in models]
rmse_stds_gencode = [df_gencode[f'{model}_RMSE'].std() for model in models]
ax.bar(models, rmse_means_gencode, yerr=rmse_stds_gencode, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_ylabel('Mean RMSE', fontweight='bold')
ax.set_title('RMSE by Model (Gencode)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: RMSE - TSS
ax = axes[0, 1]
rmse_means_tss = [df_tss[f'{model}_RMSE'].mean() for model in models]
rmse_stds_tss = [df_tss[f'{model}_RMSE'].std() for model in models]
ax.bar(models, rmse_means_tss, yerr=rmse_stds_tss, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_ylabel('Mean RMSE', fontweight='bold')
ax.set_title('RMSE by Model (TSS)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 3: Pearson - Gencode
ax = axes[0, 2]
pearson_means_gencode = [df_gencode[f'{model}_Pearson'].mean() for model in models]
pearson_stds_gencode = [df_gencode[f'{model}_Pearson'].std() for model in models]
ax.bar(models, pearson_means_gencode, yerr=pearson_stds_gencode, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_ylabel('Mean Pearson', fontweight='bold')
ax.set_title('Pearson by Model (Gencode)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 4: Pearson - TSS
ax = axes[0, 3]
pearson_means_tss = [df_tss[f'{model}_Pearson'].mean() for model in models]
pearson_stds_tss = [df_tss[f'{model}_Pearson'].std() for model in models]
ax.bar(models, pearson_means_tss, yerr=pearson_stds_tss, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_ylabel('Mean Pearson', fontweight='bold')
ax.set_title('Pearson by Model (TSS)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Row 2: RMSE Distributions
# Plot 5: RMSE Distribution - Gencode
ax = axes[1, 0]
data_to_plot = [df_gencode[f'{model}_RMSE'].dropna() for model in models]
bp = ax.boxplot(data_to_plot, tick_labels=models, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('RMSE', fontweight='bold')
ax.set_title('RMSE Distribution (Gencode)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 6: RMSE Distribution - TSS
ax = axes[1, 1]
data_to_plot = [df_tss[f'{model}_RMSE'].dropna() for model in models]
bp = ax.boxplot(data_to_plot, tick_labels=models, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('RMSE', fontweight='bold')
ax.set_title('RMSE Distribution (TSS)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 7: Pearson Distribution - Gencode
ax = axes[1, 2]
data_to_plot = [df_gencode[f'{model}_Pearson'].dropna() for model in models]
bp = ax.boxplot(data_to_plot, tick_labels=models, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Pearson', fontweight='bold')
ax.set_title('Pearson Distribution (Gencode)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 8: Pearson Distribution - TSS
ax = axes[1, 3]
data_to_plot = [df_tss[f'{model}_Pearson'].dropna() for model in models]
bp = ax.boxplot(data_to_plot, tick_labels=models, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Pearson', fontweight='bold')
ax.set_title('Pearson Distribution (TSS)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')



# Row 3: Win counts
# Plot 9: Best RMSE counts - Gencode
ax = axes[2, 0]
rmse_counts_gencode = df_gencode['Best_RMSE_Model'].value_counts()
rmse_counts_gencode = rmse_counts_gencode.reindex(models, fill_value=0)
ax.bar(rmse_counts_gencode.index, rmse_counts_gencode.values, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Best RMSE Wins (Gencode)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 10: Best RMSE counts - TSS
ax = axes[2, 1]
rmse_counts_tss = df_tss['Best_RMSE_Model'].value_counts()
rmse_counts_tss = rmse_counts_tss.reindex(models, fill_value=0)
ax.bar(rmse_counts_tss.index, rmse_counts_tss.values, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Best RMSE Wins (TSS)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 11: Best Pearson counts - Gencode
ax = axes[2, 2]
pearson_counts_gencode = df_gencode['Best_Pearson_Model'].value_counts()
pearson_counts_gencode = pearson_counts_gencode.reindex(models, fill_value=0)
ax.bar(pearson_counts_gencode.index, pearson_counts_gencode.values, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Best Pearson Wins (Gencode)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 12: Best Pearson counts - TSS
ax = axes[2, 3]
pearson_counts_tss = df_tss['Best_Pearson_Model'].value_counts()
pearson_counts_tss = pearson_counts_tss.reindex(models, fill_value=0)
ax.bar(pearson_counts_tss.index, pearson_counts_tss.values, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Best Pearson Wins (TSS)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('../figures/model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparisos.png")
plt.show()


# ============================================================================
# ANALYSIS OF HIGH-PERFORMING PERTURBATIONS (Pearson > 0.1)
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS OF HIGH-PERFORMING PERTURBATIONS (Pearson > 0.1)")
print("="*80)

# Define threshold
pearson_threshold = 0.1

# Function to identify high performers for each model
def get_high_performers(df, model, threshold=0.1):
    """Get perturbations with Pearson > threshold for a given model"""
    col = f'{model}_Pearson'
    high_performers = df[df[col] > threshold][['perturbation', col]]
    return high_performers.sort_values(col, ascending=False)

# Analyze for both Gencode and TSS
print("\n" + "-"*80)
print("GENCODE EMBEDDINGS - High Performers by Model")
print("-"*80)

gencode_high_performers = {}
for model in models:
    high_perf = get_high_performers(df_gencode, model, pearson_threshold)
    gencode_high_performers[model] = high_perf
    print(f"\n{model}: {len(high_perf)} perturbations with Pearson > {pearson_threshold}")
    if len(high_perf) > 0:
        print(high_perf.to_string(index=False))
        print(f"  Mean Pearson: {high_perf[f'{model}_Pearson'].mean():.4f}")
        print(f"  Max Pearson: {high_perf[f'{model}_Pearson'].max():.4f}")

print("\n" + "-"*80)
print("TSS EMBEDDINGS - High Performers by Model")
print("-"*80)

tss_high_performers = {}
for model in models:
    high_perf = get_high_performers(df_tss, model, pearson_threshold)
    tss_high_performers[model] = high_perf
    print(f"\n{model}: {len(high_perf)} perturbations with Pearson > {pearson_threshold}")
    if len(high_perf) > 0:
        print(high_perf.to_string(index=False))
        print(f"  Mean Pearson: {high_perf[f'{model}_Pearson'].mean():.4f}")
        print(f"  Max Pearson: {high_perf[f'{model}_Pearson'].max():.4f}")

# Find perturbations that are high performers across ALL models
print("\n" + "="*80)
print("PERTURBATIONS THAT ARE HIGH PERFORMERS ACROSS MULTIPLE MODELS")
print("="*80)

# Gencode: Count how many models each perturbation is high-performing in
gencode_high_count = {}
for model in models:
    high_perf_list = gencode_high_performers[model]['perturbation'].tolist()
    for pert in high_perf_list:
        gencode_high_count[pert] = gencode_high_count.get(pert, 0) + 1

# Sort by count
gencode_high_count_sorted = sorted(gencode_high_count.items(), key=lambda x: x[1], reverse=True)

print("\nGENCODE - Perturbations appearing as high performers in multiple models:")
print(f"{'Perturbation':<15} {'# Models':<10} {'Model Performance'}")
print("-"*80)
for pert, count in gencode_high_count_sorted:
    # Get Pearson values for this perturbation across all models
    row = df_gencode[df_gencode['perturbation'] == pert].iloc[0]
    perf_str = " | ".join([f"{model}: {row[f'{model}_Pearson']:.4f}" for model in models])
    print(f"{pert:<15} {count:<10} {perf_str}")

# TSS: Count how many models each perturbation is high-performing in
tss_high_count = {}
for model in models:
    high_perf_list = tss_high_performers[model]['perturbation'].tolist()
    for pert in high_perf_list:
        tss_high_count[pert] = tss_high_count.get(pert, 0) + 1

# Sort by count
tss_high_count_sorted = sorted(tss_high_count.items(), key=lambda x: x[1], reverse=True)

print("\nTSS - Perturbations appearing as high performers in multiple models:")
print(f"{'Perturbation':<15} {'# Models':<10} {'Model Performance'}")
print("-"*80)
for pert, count in tss_high_count_sorted:
    # Get Pearson values for this perturbation across all models
    row = df_tss[df_tss['perturbation'] == pert].iloc[0]
    perf_str = " | ".join([f"{model}: {row[f'{model}_Pearson']:.4f}" for model in models])
    print(f"{pert:<15} {count:<10} {perf_str}")

# Create detailed comparison table for high performers
print("\n" + "="*80)
print("DETAILED COMPARISON TABLE - HIGH PERFORMERS (Pearson > 0.1 in ANY model)")
print("="*80)

# Get union of all high performers from all models (both Gencode and TSS)
all_high_gencode = set()
for model in models:
    all_high_gencode.update(gencode_high_performers[model]['perturbation'].tolist())

all_high_tss = set()
for model in models:
    all_high_tss.update(tss_high_performers[model]['perturbation'].tolist())

print(f"\nGENCODE: {len(all_high_gencode)} unique high-performing perturbations")
high_perf_gencode_df = df_gencode[df_gencode['perturbation'].isin(all_high_gencode)].copy()
high_perf_gencode_df = high_perf_gencode_df.sort_values('ElasticNet_Pearson', ascending=False)
print(high_perf_gencode_df.to_string(index=False))

print(f"\nTSS: {len(all_high_tss)} unique high-performing perturbations")
high_perf_tss_df = df_tss[df_tss['perturbation'].isin(all_high_tss)].copy()
high_perf_tss_df = high_perf_tss_df.sort_values('ElasticNet_Pearson', ascending=False)
print(high_perf_tss_df.to_string(index=False))

# Save high performers to CSV
high_perf_gencode_df.to_csv('../results/high_performers_gencode.csv', index=False)
high_perf_tss_df.to_csv('../results/high_performers_tss.csv', index=False)
print("\nSaved high performer tables to CSV files")

# Visualize high performers
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('High-Performing Perturbations (Pearson > 0.1)', fontsize=16, fontweight='bold')

# Plot 1: Distribution of Pearson correlations for high performers - Gencode
ax = axes[0, 0]
for i, model in enumerate(models):
    high_perf = gencode_high_performers[model]
    if len(high_perf) > 0:
        ax.hist(high_perf[f'{model}_Pearson'], alpha=0.6, label=model, bins=15)
ax.set_xlabel('Pearson Correlation', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Distribution of High Pearson Values (Gencode)', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: Distribution of Pearson correlations for high performers - TSS
ax = axes[0, 1]
for i, model in enumerate(models):
    high_perf = tss_high_performers[model]
    if len(high_perf) > 0:
        ax.hist(high_perf[f'{model}_Pearson'], alpha=0.6, label=model, bins=15)
ax.set_xlabel('Pearson Correlation', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Distribution of High Pearson Values (TSS)', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 3: Count of high performers per model - Gencode
ax = axes[1, 0]
counts = [len(gencode_high_performers[model]) for model in models]
ax.bar(models, counts, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_ylabel('Number of High Performers', fontweight='bold')
ax.set_title('High Performer Count by Model (Gencode)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
# Add count labels on bars
for i, count in enumerate(counts):
    ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold')

# Plot 4: Count of high performers per model - TSS
ax = axes[1, 1]
counts = [len(tss_high_performers[model]) for model in models]
ax.bar(models, counts, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_ylabel('Number of High Performers', fontweight='bold')
ax.set_title('High Performer Count by Model (TSS)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
# Add count labels on bars
for i, count in enumerate(counts):
    ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/high_performers_analysis.png', dpi=300, bbox_inches='tight')
print("\nSaved: high_performers_analysis.png")
plt.show()

# ============================================================================
# GENERAL STATISTICS - EASY TO READ SUMMARY
# ============================================================================

print("\n" + "="*80)
print("GENERAL STATISTICS SUMMARY")
print("="*80)

# 1. Overall Performance Summary
print("\n" + "─"*80)
print("1. OVERALL PERFORMANCE METRICS")
print("─"*80)

summary_table = []
for model in models:
    gencode_rmse = df_gencode[f'{model}_RMSE'].mean()
    gencode_pearson = df_gencode[f'{model}_Pearson'].mean()
    tss_rmse = df_tss[f'{model}_RMSE'].mean()
    tss_pearson = df_tss[f'{model}_Pearson'].mean()
    
    summary_table.append({
        'Model': model,
        'Gencode RMSE': f'{gencode_rmse:.4f}',
        'Gencode Pearson': f'{gencode_pearson:.4f}',
        'TSS RMSE': f'{tss_rmse:.4f}',
        'TSS Pearson': f'{tss_pearson:.4f}'
    })

summary_df = pd.DataFrame(summary_table)
print(summary_df.to_string(index=False))

# 2. Best Model Summary
print("\n" + "─"*80)
print("2. BEST MODEL BY METRIC")
print("─"*80)

best_models = []

# Gencode - Best RMSE
best_rmse_gencode = summary_df.loc[summary_df['Gencode RMSE'].astype(float).idxmin(), 'Model']
best_rmse_gencode_val = df_gencode[[f'{model}_RMSE' for model in models]].mean().min()

# Gencode - Best Pearson
best_pearson_gencode = summary_df.loc[summary_df['Gencode Pearson'].astype(float).idxmax(), 'Model']
best_pearson_gencode_val = df_gencode[[f'{model}_Pearson' for model in models]].mean().max()

# TSS - Best RMSE
best_rmse_tss = summary_df.loc[summary_df['TSS RMSE'].astype(float).idxmin(), 'Model']
best_rmse_tss_val = df_tss[[f'{model}_RMSE' for model in models]].mean().min()

# TSS - Best Pearson
best_pearson_tss = summary_df.loc[summary_df['TSS Pearson'].astype(float).idxmax(), 'Model']
best_pearson_tss_val = df_tss[[f'{model}_Pearson' for model in models]].mean().max()

print(f"Gencode - Best RMSE:    {best_rmse_gencode:<12} ({best_rmse_gencode_val:.4f})")
print(f"Gencode - Best Pearson: {best_pearson_gencode:<12} ({best_pearson_gencode_val:.4f})")
print(f"TSS     - Best RMSE:    {best_rmse_tss:<12} ({best_rmse_tss_val:.4f})")
print(f"TSS     - Best Pearson: {best_pearson_tss:<12} ({best_pearson_tss_val:.4f})")

# 3. Win Rate Summary
print("\n" + "─"*80)
print("3. WIN RATES (% of perturbations where model is best)")
print("─"*80)

total_perturbations = len(df_gencode)

win_rate_table = []
for model in models:
    # Gencode wins
    gencode_rmse_wins = (df_gencode['Best_RMSE_Model'] == model).sum()
    gencode_pearson_wins = (df_gencode['Best_Pearson_Model'] == model).sum()
    
    # TSS wins
    tss_rmse_wins = (df_tss['Best_RMSE_Model'] == model).sum()
    tss_pearson_wins = (df_tss['Best_Pearson_Model'] == model).sum()
    
    win_rate_table.append({
        'Model': model,
        'Gencode RMSE Wins': f'{gencode_rmse_wins} ({gencode_rmse_wins/total_perturbations*100:.1f}%)',
        'Gencode Pearson Wins': f'{gencode_pearson_wins} ({gencode_pearson_wins/total_perturbations*100:.1f}%)',
        'TSS RMSE Wins': f'{tss_rmse_wins} ({tss_rmse_wins/total_perturbations*100:.1f}%)',
        'TSS Pearson Wins': f'{tss_pearson_wins} ({tss_pearson_wins/total_perturbations*100:.1f}%)'
    })

win_rate_df = pd.DataFrame(win_rate_table)
print(win_rate_df.to_string(index=False))

# 4. High Performer Statistics (Pearson > 0.1)
print("\n" + "─"*80)
print("4. HIGH PERFORMER STATISTICS (Pearson > 0.1)")
print("─"*80)

high_perf_stats = []
for model in models:
    gencode_high = (df_gencode[f'{model}_Pearson'] > 0.1).sum()
    tss_high = (df_tss[f'{model}_Pearson'] > 0.1).sum()
    
    high_perf_stats.append({
        'Model': model,
        'Gencode High Performers': f'{gencode_high} ({gencode_high/total_perturbations*100:.1f}%)',
        'TSS High Performers': f'{tss_high} ({tss_high/total_perturbations*100:.1f}%)'
    })

high_perf_df = pd.DataFrame(high_perf_stats)
print(high_perf_df.to_string(index=False))

# 5. Performance Variability (Standard Deviation)
print("\n" + "─"*80)
print("5. PERFORMANCE VARIABILITY (Lower std = More consistent)")
print("─"*80)

variability_table = []
for model in models:
    gencode_rmse_std = df_gencode[f'{model}_RMSE'].std()
    gencode_pearson_std = df_gencode[f'{model}_Pearson'].std()
    tss_rmse_std = df_tss[f'{model}_RMSE'].std()
    tss_pearson_std = df_tss[f'{model}_Pearson'].std()
    
    variability_table.append({
        'Model': model,
        'Gencode RMSE Std': f'{gencode_rmse_std:.4f}',
        'Gencode Pearson Std': f'{gencode_pearson_std:.4f}',
        'TSS RMSE Std': f'{tss_rmse_std:.4f}',
        'TSS Pearson Std': f'{tss_pearson_std:.4f}'
    })

variability_df = pd.DataFrame(variability_table)
print(variability_df.to_string(index=False))

# 6. Median Performance (Robust to outliers)
print("\n" + "─"*80)
print("6. MEDIAN PERFORMANCE (Robust to outliers)")
print("─"*80)

median_table = []
for model in models:
    gencode_rmse_med = df_gencode[f'{model}_RMSE'].median()
    gencode_pearson_med = df_gencode[f'{model}_Pearson'].median()
    tss_rmse_med = df_tss[f'{model}_RMSE'].median()
    tss_pearson_med = df_tss[f'{model}_Pearson'].median()
    
    median_table.append({
        'Model': model,
        'Gencode RMSE': f'{gencode_rmse_med:.4f}',
        'Gencode Pearson': f'{gencode_pearson_med:.4f}',
        'TSS RMSE': f'{tss_rmse_med:.4f}',
        'TSS Pearson': f'{tss_pearson_med:.4f}'
    })

median_df = pd.DataFrame(median_table)
print(median_df.to_string(index=False))

# 7. Performance Range (Min/Max)
print("\n" + "─"*80)
print("7. PERFORMANCE RANGE")
print("─"*80)

for model in models:
    print(f"\n{model}:")
    print(f"  Gencode RMSE:    [{df_gencode[f'{model}_RMSE'].min():.4f}, {df_gencode[f'{model}_RMSE'].max():.4f}]")
    print(f"  Gencode Pearson: [{df_gencode[f'{model}_Pearson'].min():.4f}, {df_gencode[f'{model}_Pearson'].max():.4f}]")
    print(f"  TSS RMSE:        [{df_tss[f'{model}_RMSE'].min():.4f}, {df_tss[f'{model}_RMSE'].max():.4f}]")
    print(f"  TSS Pearson:     [{df_tss[f'{model}_Pearson'].min():.4f}, {df_tss[f'{model}_Pearson'].max():.4f}]")

# 8. Gencode vs TSS Comparison
print("\n" + "─"*80)
print("8. GENCODE vs TSS COMPARISON (Which embedding is better?)")
print("─"*80)

embedding_comparison = []
for model in models:
    gencode_rmse = df_gencode[f'{model}_RMSE'].mean()
    tss_rmse = df_tss[f'{model}_RMSE'].mean()
    gencode_pearson = df_gencode[f'{model}_Pearson'].mean()
    tss_pearson = df_tss[f'{model}_Pearson'].mean()
    
    better_rmse = 'Gencode' if gencode_rmse < tss_rmse else 'TSS'
    better_pearson = 'Gencode' if gencode_pearson > tss_pearson else 'TSS'
    rmse_diff = abs(gencode_rmse - tss_rmse)
    pearson_diff = abs(gencode_pearson - tss_pearson)
    
    embedding_comparison.append({
        'Model': model,
        'Better RMSE': f'{better_rmse} (Δ={rmse_diff:.4f})',
        'Better Pearson': f'{better_pearson} (Δ={pearson_diff:.4f})'
    })

embedding_comp_df = pd.DataFrame(embedding_comparison)
print(embedding_comp_df.to_string(index=False))

# 9. Key Insights Summary
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Determine overall best model
rmse_scores = {model: (df_gencode[f'{model}_RMSE'].mean() + df_tss[f'{model}_RMSE'].mean()) / 2 for model in models}
pearson_scores = {model: (df_gencode[f'{model}_Pearson'].mean() + df_tss[f'{model}_Pearson'].mean()) / 2 for model in models}

best_overall_rmse = min(rmse_scores, key=rmse_scores.get)
best_overall_pearson = max(pearson_scores, key=pearson_scores.get)

print(f"\n✓ Best Model (RMSE):    {best_overall_rmse} with average RMSE of {rmse_scores[best_overall_rmse]:.4f}")
print(f"✓ Best Model (Pearson): {best_overall_pearson} with average Pearson of {pearson_scores[best_overall_pearson]:.4f}")

# Count total high performers
total_high_gencode = len(all_high_gencode)
total_high_tss = len(all_high_tss)
print(f"\n✓ Total perturbations with Pearson > 0.1:")
print(f"  - Gencode: {total_high_gencode}/{total_perturbations} ({total_high_gencode/total_perturbations*100:.1f}%)")
print(f"  - TSS:     {total_high_tss}/{total_perturbations} ({total_high_tss/total_perturbations*100:.1f}%)")

# Determine which embedding is generally better
gencode_better_count = sum(1 for model in models if df_gencode[f'{model}_RMSE'].mean() < df_tss[f'{model}_RMSE'].mean())
tss_better_count = len(models) - gencode_better_count

print(f"\n✓ Embedding Preference (by RMSE):")
if gencode_better_count > tss_better_count:
    print(f"  - Gencode is better for {gencode_better_count}/{len(models)} models")
else:
    print(f"  - TSS is better for {tss_better_count}/{len(models)} models")

print("\n" + "="*80)

# ============================================================================
# HIGH PERFORMER STATISTICS - SIDE BY SIDE COMPARISON
# ============================================================================

print("\n" + "="*80)
print("HIGH PERFORMER STATISTICS AT DIFFERENT PEARSON THRESHOLDS")
print("="*80)

# Define thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

for threshold in thresholds:
    print("\n" + "─"*80)
    print(f"HIGH PERFORMERS (Pearson > {threshold})")
    print("─"*80)
    
    combined_stats = []
    for model in models:
        gencode_count = (df_gencode[f'{model}_Pearson'] > threshold).sum()
        tss_count = (df_tss[f'{model}_Pearson'] > threshold).sum()
        
        combined_stats.append({
            'Model': model,
            'Gencode High Performers': f'{gencode_count} ({gencode_count/total_perturbations*100:.1f}%)',
            'TSS High Performers': f'{tss_count} ({tss_count/total_perturbations*100:.1f}%)'
        })
    
    combined_df = pd.DataFrame(combined_stats)
    print(combined_df.to_string(index=False))

from sklearn.decomposition import PCA 
import numpy as np
# After loading embeddings 
embeddings = np.load('../../../embeddings/embeddings_enformer_gencode.v49.pc_transcripts_full.npy', allow_pickle=True) # Shape: (N, 3072) 
# Reduce to 1024 dimensions with PCA 
pca = PCA(n_components=1024, random_state=42) 
embeddings_1024 = pca.fit_transform(embeddings) # Shape: (N, 512) 
print(f"Variance explained by 1024 components: {pca.explained_variance_ratio_.sum():.4f}")


np.save(
    '../../../embeddings/embeddings_enformer_gencode.v49.pc_transcripts_1024.npy',
    embeddings_1024
)

from sklearn.decomposition import PCA 
import numpy as np
# After loading embeddings 
embeddings = np.load('../../../embeddings/embeddings_enformer_tss_full.npy', allow_pickle=True) # Shape: (N, 3072) 
# Reduce to 1024 dimensions with PCA 
pca = PCA(n_components=1024, random_state=42) 
embeddings_1024 = pca.fit_transform(embeddings) # Shape: (N, 512) 
print(f"Variance explained by 1024 components: {pca.explained_variance_ratio_.sum():.4f}")


np.save(
    '../../../embeddings/embeddings_enformer_tss_1024.npy',
    embeddings_1024
)

from sklearn.decomposition import PCA 
import numpy as np
# After loading embeddings 
embeddings = np.load('../../../embeddings/embeddings_enformer_gencode.v49.pc_transcripts_full.npy', allow_pickle=True) # Shape: (N, 3072) 
# Reduce to 1024 dimensions with PCA 

np.shape(embeddings)

pca = PCA(n_components=512, random_state=42) 

embeddings_1024 = pca.fit_transform(embeddings)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df_gencode = pd.read_csv('../results/unified_results_gencode.csv')
df_tss = pd.read_csv('../results/unified_results_tss.csv')

print("="*80)
print("PERFORMANCE CHANGE ANALYSIS: ElasticNet vs ElasticNet_1024")
print("="*80)

def analyze_performance_changes(df, embedding_type):
    """Analyze which perturbations improved/worsened with 1024-dim embeddings"""
    
    print(f"\n{'='*80}")
    print(f"{embedding_type.upper()} EMBEDDINGS - PERFORMANCE CHANGES")
    print(f"{'='*80}")
    
    # Calculate performance differences
    df['Pearson_Improvement'] = df['ElasticNet_1024_Pearson'] - df['ElasticNet_Pearson']
    df['RMSE_Improvement'] = df['ElasticNet_RMSE'] - df['ElasticNet_1024_RMSE']  # Positive = better (lower RMSE)
    
    # Overall statistics
    print(f"\nOVERALL STATISTICS:")
    print(f"  Mean Pearson improvement: {df['Pearson_Improvement'].mean():.4f}")
    print(f"  Median Pearson improvement: {df['Pearson_Improvement'].median():.4f}")
    print(f"  Mean RMSE improvement: {df['RMSE_Improvement'].mean():.4f}")
    print(f"  Median RMSE improvement: {df['RMSE_Improvement'].median():.4f}")
    
    # Count improvements/declines
    pearson_improved = (df['Pearson_Improvement'] > 0).sum()
    pearson_declined = (df['Pearson_Improvement'] < 0).sum()
    rmse_improved = (df['RMSE_Improvement'] > 0).sum()
    rmse_declined = (df['RMSE_Improvement'] < 0).sum()
    
    print(f"\n  Perturbations with improved Pearson: {pearson_improved}/{len(df)} ({pearson_improved/len(df)*100:.1f}%)")
    print(f"  Perturbations with declined Pearson: {pearson_declined}/{len(df)} ({pearson_declined/len(df)*100:.1f}%)")
    print(f"  Perturbations with improved RMSE: {rmse_improved}/{len(df)} ({rmse_improved/len(df)*100:.1f}%)")
    print(f"  Perturbations with declined RMSE: {rmse_declined}/{len(df)} ({rmse_declined/len(df)*100:.1f}%)")
    
    # Top 20 biggest improvements (by Pearson)
    print(f"\n{'─'*80}")
    print(f"TOP 20 BIGGEST IMPROVEMENTS (by Pearson correlation)")
    print(f"{'─'*80}")
    top_improved = df.nlargest(20, 'Pearson_Improvement')[
        ['perturbation', 'ElasticNet_Pearson', 'ElasticNet_1024_Pearson', 'Pearson_Improvement']
    ]
    print(top_improved.to_string(index=False))
    
    # Top 20 biggest declines (by Pearson)
    print(f"\n{'─'*80}")
    print(f"TOP 20 BIGGEST DECLINES (by Pearson correlation)")
    print(f"{'─'*80}")
    top_declined = df.nsmallest(20, 'Pearson_Improvement')[
        ['perturbation', 'ElasticNet_Pearson', 'ElasticNet_1024_Pearson', 'Pearson_Improvement']
    ]
    print(top_declined.to_string(index=False))
    
    # Perturbations that crossed performance thresholds
    print(f"\n{'─'*80}")
    print(f"PERTURBATIONS CROSSING PERFORMANCE THRESHOLDS")
    print(f"{'─'*80}")
    
    # Became high performers (crossed 0.1 threshold)
    became_high_perf = df[
        (df['ElasticNet_Pearson'] <= 0.1) & (df['ElasticNet_1024_Pearson'] > 0.1)
    ][['perturbation', 'ElasticNet_Pearson', 'ElasticNet_1024_Pearson', 'Pearson_Improvement']]
    
    print(f"\nBecame high performers (Pearson crossed 0.1): {len(became_high_perf)}")
    if len(became_high_perf) > 0:
        print(became_high_perf.to_string(index=False))
    
    # Lost high performer status
    lost_high_perf = df[
        (df['ElasticNet_Pearson'] > 0.1) & (df['ElasticNet_1024_Pearson'] <= 0.1)
    ][['perturbation', 'ElasticNet_Pearson', 'ElasticNet_1024_Pearson', 'Pearson_Improvement']]
    
    print(f"\nLost high performer status (Pearson dropped below 0.1): {len(lost_high_perf)}")
    if len(lost_high_perf) > 0:
        print(lost_high_perf.to_string(index=False))
    
    # Crossed 0.2 threshold
    crossed_02 = df[
        (df['ElasticNet_Pearson'] <= 0.2) & (df['ElasticNet_1024_Pearson'] > 0.2)
    ][['perturbation', 'ElasticNet_Pearson', 'ElasticNet_1024_Pearson', 'Pearson_Improvement']]
    
    print(f"\nCrossed Pearson 0.2 threshold: {len(crossed_02)}")
    if len(crossed_02) > 0:
        print(crossed_02.to_string(index=False))
    
    # Crossed 0.3 threshold
    crossed_03 = df[
        (df['ElasticNet_Pearson'] <= 0.3) & (df['ElasticNet_1024_Pearson'] > 0.3)
    ][['perturbation', 'ElasticNet_Pearson', 'ElasticNet_1024_Pearson', 'Pearson_Improvement']]
    
    print(f"\nCrossed Pearson 0.3 threshold: {len(crossed_03)}")
    if len(crossed_03) > 0:
        print(crossed_03.to_string(index=False))
    
    return df

# Analyze both embedding types
df_gencode_analyzed = analyze_performance_changes(df_gencode, 'Gencode')
df_tss_analyzed = analyze_performance_changes(df_tss, 'TSS')

# Create visualizations
print(f"\n{'='*80}")
print("CREATING VISUALIZATIONS...")
print(f"{'='*80}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Gencode
# Plot 1: Scatter plot of Pearson improvements
ax = axes[0, 0]
ax.scatter(df_gencode_analyzed['ElasticNet_Pearson'], 
           df_gencode_analyzed['ElasticNet_1024_Pearson'], 
           alpha=0.6, s=50)
ax.plot([0, 0.5], [0, 0.5], 'r--', label='No change', linewidth=2)
ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Pearson = 0.1')
ax.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5)
ax.set_xlabel('ElasticNet (32-dim) Pearson', fontweight='bold')
ax.set_ylabel('ElasticNet_1024 Pearson', fontweight='bold')
ax.set_title('Gencode: Pearson Comparison', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Distribution of improvements
ax = axes[0, 1]
ax.hist(df_gencode_analyzed['Pearson_Improvement'], bins=30, alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
ax.set_xlabel('Pearson Improvement', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Gencode: Distribution of Improvements', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Top/Bottom performers
ax = axes[0, 2]
top_10_improved = df_gencode_analyzed.nlargest(10, 'Pearson_Improvement')
top_10_declined = df_gencode_analyzed.nsmallest(10, 'Pearson_Improvement')
combined = pd.concat([top_10_improved, top_10_declined])
colors = ['green' if x > 0 else 'red' for x in combined['Pearson_Improvement']]
y_pos = range(len(combined))
ax.barh(y_pos, combined['Pearson_Improvement'], color=colors, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(combined['perturbation'], fontsize=8)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Pearson Improvement', fontweight='bold')
ax.set_title('Gencode: Top Gainers/Losers', fontweight='bold')
ax.grid(alpha=0.3, axis='x')

# Row 2: TSS (same plots)
# Plot 4: Scatter plot
ax = axes[1, 0]
ax.scatter(df_tss_analyzed['ElasticNet_Pearson'], 
           df_tss_analyzed['ElasticNet_1024_Pearson'], 
           alpha=0.6, s=50, color='orange')
ax.plot([0, 0.5], [0, 0.5], 'r--', label='No change', linewidth=2)
ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Pearson = 0.1')
ax.axvline(x=0.1, color='green', linestyle='--', alpha=0.5)
ax.set_xlabel('ElasticNet (32-dim) Pearson', fontweight='bold')
ax.set_ylabel('ElasticNet_1024 Pearson', fontweight='bold')
ax.set_title('TSS: Pearson Comparison', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Distribution
ax = axes[1, 1]
ax.hist(df_tss_analyzed['Pearson_Improvement'], bins=30, alpha=0.7, 
        edgecolor='black', color='orange')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
ax.set_xlabel('Pearson Improvement', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('TSS: Distribution of Improvements', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Top/Bottom performers
ax = axes[1, 2]
top_10_improved = df_tss_analyzed.nlargest(10, 'Pearson_Improvement')
top_10_declined = df_tss_analyzed.nsmallest(10, 'Pearson_Improvement')
combined = pd.concat([top_10_improved, top_10_declined])
colors = ['green' if x > 0 else 'red' for x in combined['Pearson_Improvement']]
y_pos = range(len(combined))
ax.barh(y_pos, combined['Pearson_Improvement'], color=colors, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(combined['perturbation'], fontsize=8)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Pearson Improvement', fontweight='bold')
ax.set_title('TSS: Top Gainers/Losers', fontweight='bold')
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('../figures/elasticnet_vs_1024_performance_changes.png', 
            dpi=300, bbox_inches='tight')
print("Saved: elasticnet_vs_1024_performance_changes.png")
plt.show()

# Save detailed results to CSV
print(f"\nSaving detailed results to CSV...")
df_gencode_analyzed.to_csv(
    '../results/gencode_elasticnet_improvements.csv',
    index=False
)
print("Saved: gencode_elasticnet_improvements.csv")

df_tss_analyzed.to_csv(
    '../results/tss_elasticnet_improvements.csv',
    index=False
)
print("Saved: tss_elasticnet_improvements.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Load data
df_gencode = pd.read_csv('../results/unified_results_gencode.csv')
df_tss = pd.read_csv('../results/unified_results_tss.csv')

# Calculate improvements
df_gencode['Pearson_Improvement'] = df_gencode['ElasticNet_1024_Pearson'] - df_gencode['ElasticNet_Pearson']
df_tss['Pearson_Improvement'] = df_tss['ElasticNet_1024_Pearson'] - df_tss['ElasticNet_Pearson']

# Create figure with both embeddings side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

def create_detailed_scatter(ax, df, embedding_type, color='steelblue'):
    """Create detailed scatter plot with annotations and statistics"""
    
    # Color points by improvement magnitude
    improvements = df['Pearson_Improvement'].values
    colors = plt.cm.RdYlGn(plt.Normalize(vmin=-0.2, vmax=0.2)(improvements))
    
    # Main scatter plot
    scatter = ax.scatter(df['ElasticNet_Pearson'], 
                        df['ElasticNet_1024_Pearson'],
                        c=improvements,
                        cmap='RdYlGn',
                        vmin=-0.2, vmax=0.2,
                        alpha=0.7, 
                        s=60,
                        edgecolors='black',
                        linewidth=0.5)
    
    # Diagonal line (no change)
    max_val = max(df['ElasticNet_Pearson'].max(), df['ElasticNet_1024_Pearson'].max())
    min_val = min(df['ElasticNet_Pearson'].min(), df['ElasticNet_1024_Pearson'].min())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', 
            label='No change', linewidth=2, alpha=0.7)
    
    # Threshold lines
    ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    
    # Shade quadrants
    # Upper left: Improved to high performer
    ax.add_patch(Rectangle((min_val, 0.1), 0.1 - min_val, max_val - 0.1, 
                           facecolor='green', alpha=0.05))
    # Lower right: Declined from high performer
    ax.add_patch(Rectangle((0.1, min_val), max_val - 0.1, 0.1 - min_val, 
                           facecolor='red', alpha=0.05))
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Pearson Improvement', fontweight='bold', fontsize=11)
    
    # Calculate statistics
    above_diagonal = (df['ElasticNet_1024_Pearson'] > df['ElasticNet_Pearson']).sum()
    below_diagonal = (df['ElasticNet_1024_Pearson'] < df['ElasticNet_Pearson']).sum()
    
    # Threshold crossing stats
    became_high_perf = ((df['ElasticNet_Pearson'] <= 0.1) & 
                        (df['ElasticNet_1024_Pearson'] > 0.1)).sum()
    lost_high_perf = ((df['ElasticNet_Pearson'] > 0.1) & 
                      (df['ElasticNet_1024_Pearson'] <= 0.1)).sum()
    crossed_02 = ((df['ElasticNet_Pearson'] <= 0.2) & 
                  (df['ElasticNet_1024_Pearson'] > 0.2)).sum()
    
    mean_improvement = df['Pearson_Improvement'].mean()
    median_improvement = df['Pearson_Improvement'].median()
    
    # Top 3 improved
    top_3 = df.nlargest(3, 'Pearson_Improvement')
    
    # Annotate top 3 improvements
    for idx, row in top_3.iterrows():
        ax.annotate(row['perturbation'], 
                   xy=(row['ElasticNet_Pearson'], row['ElasticNet_1024_Pearson']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                                 color='black', lw=1.5))
    
    # Add statistics text box
    stats_text = (
        f"Total: {len(df)} perturbations\n"
        f"Improved: {above_diagonal} ({above_diagonal/len(df)*100:.1f}%)\n"
        f"Declined: {below_diagonal} ({below_diagonal/len(df)*100:.1f}%)\n"
        f"\n"
        f"Mean Δ: {mean_improvement:+.4f}\n"
        f"Median Δ: {median_improvement:+.4f}\n"
        # f"\n"
        # f"Crossed 0.1→: {became_high_perf}\n"
        # f"Crossed 0.2→: {crossed_02}\n"
        # f"Lost 0.1←: {lost_high_perf}"
    )
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                     edgecolor='black', linewidth=1.5))
    
    # Labels and title
    ax.set_xlabel('ElasticNet (32-dim) Pearson', fontweight='bold', fontsize=12)
    ax.set_ylabel('ElasticNet (1024-dim) Pearson', fontweight='bold', fontsize=12)
    ax.set_title(f'{embedding_type} Embeddings: Model Performance Comparison', 
                fontweight='bold', fontsize=13)
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    return ax

# Create plots for both embedding types
create_detailed_scatter(axes[0], df_gencode, 'GENCODE', color='steelblue')
create_detailed_scatter(axes[1], df_tss, 'TSS', color='darkorange')

plt.suptitle('ElasticNet (32-dim) vs ElasticNet (1024-dim): Individual Perturbation Performance', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('../figures/elasticnet_comparison_detailed.png', 
            dpi=300, bbox_inches='tight')
print("Saved: elasticnet_comparison_detailed.png")
plt.show()

# Print detailed statistics
print("="*80)
print("DETAILED COMPARISON STATISTICS")
print("="*80)

for df, name in [(df_gencode, 'GENCODE'), (df_tss, 'TSS')]:
    print(f"\n{name} Embeddings:")
    print(f"  Mean improvement: {df['Pearson_Improvement'].mean():+.4f}")
    print(f"  Median improvement: {df['Pearson_Improvement'].median():+.4f}")
    print(f"  Std improvement: {df['Pearson_Improvement'].std():.4f}")
    print(f"  Max improvement: {df['Pearson_Improvement'].max():+.4f} ({df.loc[df['Pearson_Improvement'].idxmax(), 'perturbation']})")
    print(f"  Max decline: {df['Pearson_Improvement'].min():+.4f} ({df.loc[df['Pearson_Improvement'].idxmin(), 'perturbation']})")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data
df_gencode = pd.read_csv('../results/unified_results_gencode.csv')
df_tss = pd.read_csv('../results/unified_results_tss.csv')

# Calculate improvements
df_gencode['Pearson_Improvement'] = df_gencode['ElasticNet_1024_Pearson'] - df_gencode['ElasticNet_Pearson']
df_tss['Pearson_Improvement'] = df_tss['ElasticNet_1024_Pearson'] - df_tss['ElasticNet_Pearson']

# Create static matplotlib figure with both embeddings side by side
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

def create_detailed_scatter(ax, df, embedding_type, color='steelblue'):
    """Create detailed scatter plot with annotations and statistics"""
    
    # Color points by improvement magnitude
    improvements = df['Pearson_Improvement'].values
    colors = plt.cm.RdYlGn(plt.Normalize(vmin=-0.2, vmax=0.2)(improvements))
    
    # Main scatter plot
    scatter = ax.scatter(df['ElasticNet_Pearson'], 
                        df['ElasticNet_1024_Pearson'],
                        c=improvements,
                        cmap='RdYlGn',
                        vmin=-0.2, vmax=0.2,
                        alpha=0.7, 
                        s=60,
                        edgecolors='black',
                        linewidth=0.5)
    
    # Diagonal line (no change)
    max_val = max(df['ElasticNet_Pearson'].max(), df['ElasticNet_1024_Pearson'].max())
    min_val = min(df['ElasticNet_Pearson'].min(), df['ElasticNet_1024_Pearson'].min())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', 
            label='No change', linewidth=2, alpha=0.7)
    
    # Threshold lines
    ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    
    # Shade quadrants
    # Upper left: Improved to high performer
    ax.add_patch(Rectangle((min_val, 0.1), 0.1 - min_val, max_val - 0.1, 
                           facecolor='green', alpha=0.05))
    # Lower right: Declined from high performer
    ax.add_patch(Rectangle((0.1, min_val), max_val - 0.1, 0.1 - min_val, 
                           facecolor='red', alpha=0.05))
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Pearson Improvement', fontweight='bold', fontsize=11)
    
    # Calculate statistics
    above_diagonal = (df['ElasticNet_1024_Pearson'] > df['ElasticNet_Pearson']).sum()
    below_diagonal = (df['ElasticNet_1024_Pearson'] < df['ElasticNet_Pearson']).sum()
    
    # Threshold crossing stats
    became_high_perf = ((df['ElasticNet_Pearson'] <= 0.1) & 
                        (df['ElasticNet_1024_Pearson'] > 0.1)).sum()
    lost_high_perf = ((df['ElasticNet_Pearson'] > 0.1) & 
                      (df['ElasticNet_1024_Pearson'] <= 0.1)).sum()
    crossed_02 = ((df['ElasticNet_Pearson'] <= 0.2) & 
                  (df['ElasticNet_1024_Pearson'] > 0.2)).sum()
    
    mean_improvement = df['Pearson_Improvement'].mean()
    median_improvement = df['Pearson_Improvement'].median()
    
    # Top 3 improved
    top_3 = df.nlargest(3, 'Pearson_Improvement')
    
    # Annotate top 3 improvements
    positions = [(10, 10), (10, -10), (-10, 10)]
    
    for idx, (row_idx, row) in enumerate(top_3.iterrows()):
        offset = positions[idx % len(positions)]
        ax.annotate(f"{row['perturbation']}\nΔ={row['Pearson_Improvement']:.3f}", 
                   xy=(row['ElasticNet_Pearson'], row['ElasticNet_1024_Pearson']),
                   xytext=offset, textcoords='offset points',
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', 
                                 color='black', lw=1.2))
    
    # Add statistics text box
    stats_text = (
        f"Total: {len(df)} perturbations\n"
        f"Improved: {above_diagonal} ({above_diagonal/len(df)*100:.1f}%)\n"
        f"Declined: {below_diagonal} ({below_diagonal/len(df)*100:.1f}%)\n"
        f"\n"
        f"Mean Δ: {mean_improvement:+.4f}\n"
        f"Median Δ: {median_improvement:+.4f}\n"
        f"\n"
        f"Crossed 0.1→: {became_high_perf}\n"
        f"Crossed 0.2→: {crossed_02}\n"
        f"Lost 0.1←: {lost_high_perf}"
    )
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                     edgecolor='black', linewidth=1.5))
    
    # Labels and title
    ax.set_xlabel('ElasticNet (32-dim) Pearson', fontweight='bold', fontsize=12)
    ax.set_ylabel('ElasticNet(1024-dim)Pearson', fontweight='bold', fontsize=12)
    ax.set_title(f'{embedding_type} Embeddings: Model Performance Comparison', 
                fontweight='bold', fontsize=13)
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    return ax

# Create static plots
create_detailed_scatter(axes[0], df_gencode, 'GENCODE', color='steelblue')
create_detailed_scatter(axes[1], df_tss, 'TSS', color='darkorange')

plt.suptitle('ElasticNet (32-dim) vs ElasticNet (1024-dim): Individual Perturbation Performance', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('../figures/elasticnet_comparison_detailed.png', 
            dpi=300, bbox_inches='tight')
print("Saved: elasticnet_comparison_detailed.png")
plt.show()

# ============================================================================
# CREATE INTERACTIVE HTML PLOT
# ============================================================================

print("\nCreating interactive HTML plot...")

# Create subplots for interactive version
fig_interactive = make_subplots(
    rows=1, cols=2,
    subplot_titles=('GENCODE Embeddings', 'TSS Embeddings'),
    horizontal_spacing=0.12
)

def add_interactive_scatter(fig, df, embedding_type, row, col):
    """Add interactive scatter plot to plotly figure"""
    
    # Calculate max/min for diagonal line
    max_val = max(df['ElasticNet_Pearson'].max(), df['ElasticNet_1024_Pearson'].max())
    min_val = min(df['ElasticNet_Pearson'].min(), df['ElasticNet_1024_Pearson'].min())
    
    # Create hover text with detailed information
    hover_text = []
    for idx, row_data in df.iterrows():
        text = (
            f"<b>{row_data['perturbation']}</b><br>"
            f"ElasticNet (32-dim): {row_data['ElasticNet_Pearson']:.4f}<br>"
            f"ElasticNet (1024-dim): {row_data['ElasticNet_1024_Pearson']:.4f}<br>"
            f"<b>Improvement: {row_data['Pearson_Improvement']:+.4f}</b><br>"
            f"<br>"
            f"RMSE (32-dim): {row_data['ElasticNet_RMSE']:.4f}<br>"
            f"RMSE (1024-dim): {row_data['ElasticNet_1024_RMSE']:.4f}<br>"
        )
        hover_text.append(text)
    
    # Main scatter plot
    fig.add_trace(
        go.Scatter(
            x=df['ElasticNet_Pearson'],
            y=df['ElasticNet_1024_Pearson'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['Pearson_Improvement'],
                colorscale='RdYlGn',
                cmin=-0.2,
                cmax=0.2,
                showscale=(col == 2),  # Show colorbar only on right plot
                colorbar=dict(
                    title=dict(text="Pearson<br>Improvement"),
                    x=1.15
                ),
                line=dict(width=0.5, color='black')
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name=embedding_type
        ),
        row=row, col=col
    )
    
    # Diagonal line (no change)
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='No change',
            showlegend=(col == 1),
            hoverinfo='skip'
        ),
        row=row, col=col
    )
    
    # Threshold lines at 0.1
    fig.add_hline(y=0.1, line=dict(color='orange', width=1.5, dash='dash'), 
                  opacity=0.5, row=row, col=col)
    fig.add_vline(x=0.1, line=dict(color='orange', width=1.5, dash='dash'), 
                  opacity=0.5, row=row, col=col)
    
    # Threshold lines at 0.2
    fig.add_hline(y=0.2, line=dict(color='red', width=1.5, dash='dash'), 
                  opacity=0.4, row=row, col=col)
    fig.add_vline(x=0.2, line=dict(color='red', width=1.5, dash='dash'), 
                  opacity=0.4, row=row, col=col)
    
    # Update axes
    fig.update_xaxes(title_text="ElasticNet (32-dim) Pearson", row=row, col=col)
    fig.update_yaxes(title_text="ElasticNet (1024-dim) Pearson", row=row, col=col)

# Add both embeddings to interactive plot
add_interactive_scatter(fig_interactive, df_gencode, 'GENCODE', 1, 1)
add_interactive_scatter(fig_interactive, df_tss, 'TSS', 1, 2)

# Update layout
fig_interactive.update_layout(
    title_text="ElasticNet (32-dim) vs ElasticNet (1024-dim): Interactive Performance Comparison",
    title_font_size=16,
    height=600,
    width=1400,
    showlegend=True,
    hovermode='closest',
    template='plotly_white'
)

# Save interactive HTML
fig_interactive.write_html(
    '../interactive/elasticnet_comparison_interactive.html'
)
print("Saved: elasticnet_comparison_interactive.html")

# Print detailed statistics
print("\n" + "="*80)
print("DETAILED COMPARISON STATISTICS")
print("="*80)

for df, name in [(df_gencode, 'GENCODE'), (df_tss, 'TSS')]:
    print(f"\n{name} Embeddings:")
    print(f"  Mean improvement: {df['Pearson_Improvement'].mean():+.4f}")
    print(f"  Median improvement: {df['Pearson_Improvement'].median():+.4f}")
    print(f"  Std improvement: {df['Pearson_Improvement'].std():.4f}")
    print(f"  Max improvement: {df['Pearson_Improvement'].max():+.4f} ({df.loc[df['Pearson_Improvement'].idxmax(), 'perturbation']})")
    print(f"  Max decline: {df['Pearson_Improvement'].min():+.4f} ({df.loc[df['Pearson_Improvement'].idxmin(), 'perturbation']})")
    
    print(f"\n  Top 3 Most Improved Perturbations:")
    top_3 = df.nlargest(3, 'Pearson_Improvement')[['perturbation', 'ElasticNet_Pearson', 'ElasticNet_1024_Pearson', 'Pearson_Improvement']]
    for idx, row in top_3.iterrows():
        print(f"    {row['perturbation']:<15} : {row['ElasticNet_Pearson']:.4f} → {row['ElasticNet_1024_Pearson']:.4f} (Δ={row['Pearson_Improvement']:+.4f})")