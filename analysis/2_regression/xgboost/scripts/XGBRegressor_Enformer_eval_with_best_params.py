import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from xgboost import XGBRegressor
import sys


def cv_single_target_with_params(X, y, pert_name, params, n_splits=5):
    """
    Run KFold CV using provided best parameters for one perturbation vector.
    Reports RMSE and Pearson correlation for each fold.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    fold_id = 0
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Use the best parameters for this gene
        model = XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        corr, _ = pearsonr(y_test, y_pred)

        results.append({
            "perturbation": pert_name,
            "fold": fold_id,
            "rmse": rmse,
            "pearson_corr": corr
        })

        fold_id += 1

    return results


# Get command line arguments
if len(sys.argv) < 2:
    raise ValueError("Please provide embedding type as argument: 'tss' or 'gencode'")

EMBEDDING_TYPE = sys.argv[1]

print("=" * 50)
print(f"Embedding type: {EMBEDDING_TYPE}")
print("=" * 50)

# best parameters from previous run
input_file = "../../../data/active_guides_CRISPRa_mean_pop_mean.csv"
param_file = f"../results/{EMBEDDING_TYPE}_results.csv"

print(f"Loading best parameters from: {param_file}")
param_results = pd.read_csv(param_file)

# Get the best parameters for each gene (average across folds)
best_params = param_results.groupby('perturbation').agg({
    'best_n_estimators': lambda x: int(x.mode()[0]),  # most common value
    'best_max_depth': lambda x: int(x.mode()[0]),
    'best_learning_rate': lambda x: float(x.mode()[0]),
    'best_subsample': lambda x: float(x.mode()[0])
}).reset_index()

print(f"Loaded best parameters for {len(best_params)} genes")

# input data
mean_pop = pd.read_csv(input_file, index_col=0)

# embeddings and metadata based on selected type
if EMBEDDING_TYPE == "tss":
    embeddings_path = "../../../embeddings/embeddings_enformer_tss.npy"
    metadata_path = "../../../embeddings/enformer_gene_names.txt"
    print(f"Loading TSS embeddings from: {embeddings_path}")
elif EMBEDDING_TYPE == "gencode":
    embeddings_path = "../../../embeddings/embeddings_enformer_gencode.v49.pc_transcripts.npy"
    metadata_path = "../../../embeddings/gencode.v49.pc_transcripts_gene_names.txt"
    print(f"Loading Gencode v49 PC transcripts embeddings from: {embeddings_path}")
else:
    raise ValueError(f"Unknown embedding type: {EMBEDDING_TYPE}. Use 'tss' or 'gencode'.")

# embeddings (allow_pickle for compatibility)
embeddings = np.load(embeddings_path, allow_pickle=True)
print(f"Embeddings shape: {embeddings.shape}")

# gene names from metadata
meta_table = pd.read_csv(metadata_path, sep="\t", header=None, names=["index", "gene"])
print(f"Loaded {len(meta_table)} gene names from metadata")

embeddings = pd.DataFrame(embeddings, index=meta_table['gene'])

# Align datasets
embeddings = embeddings.groupby(embeddings.index).mean()
common_genes = mean_pop.columns.intersection(embeddings.index)
X = embeddings.loc[common_genes]
Y = mean_pop[common_genes].T

perturb_names = Y.columns

print("X shape:", X.shape, "| Y shape:", Y.shape)

# Run CV for each perturbation with best params AND train final model
all_results = []
all_weights = []

for i in range(Y.shape[1]):
    pert_name = perturb_names[i]

    # Get best parameters for this gene
    params_row = best_params[best_params['perturbation'] == pert_name]

    if len(params_row) == 0:
        print(f"\n>>> WARNING: No best params found for {pert_name}, skipping...")
        continue

    params = {
        'n_estimators': params_row['best_n_estimators'].values[0],
        'max_depth': params_row['best_max_depth'].values[0],
        'learning_rate': params_row['best_learning_rate'].values[0],
        'subsample': params_row['best_subsample'].values[0]
    }

    print(f"\n>>> Running 5-fold CV for {pert_name} ({i+1}/{Y.shape[1]}) with params: {params}")

    results = cv_single_target_with_params(X.values, Y.iloc[:, i].values, pert_name, params)
    all_results.extend(results)

    # Train final model on ALL data
    print(f">>> Training final model on all data for {pert_name}")
    final_model = XGBRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(X.values, Y.iloc[:, i].values)

    # Extract feature importance (weights)
    feature_importance = final_model.feature_importances_

    # Store weights for this gene
    weight_row = {'perturbation': pert_name}
    for feat_idx, importance in enumerate(feature_importance):
        weight_row[f'feature_{feat_idx}'] = importance
    all_weights.append(weight_row)

# Save per-fold results
results_df = pd.DataFrame(all_results)

csv_file = f"../results/{EMBEDDING_TYPE}_eval_with_best_params.csv"
results_df.to_csv(csv_file, index=False)

print(f"\nPer-fold results saved to {csv_file}")

# Calculate and save summary statistics
# Calculate mean metrics per gene
summary_per_gene = results_df.groupby('perturbation').agg({
    'rmse': ['mean', 'std'],
    'pearson_corr': ['mean', 'std']
}).reset_index()

summary_per_gene.columns = ['perturbation', 'mean_rmse', 'std_rmse', 'mean_pearson', 'std_pearson']

# Calculate overall average
overall_mean_rmse = summary_per_gene['mean_rmse'].mean()
overall_mean_pearson = summary_per_gene['mean_pearson'].mean()

print("\n" + "=" * 50)
print("SUMMARY STATISTICS")
print("=" * 50)
print(f"Overall average RMSE: {overall_mean_rmse:.6f}")
print(f"Overall average Pearson correlation: {overall_mean_pearson:.6f}")
print("=" * 50)

# Save summary
summary_file = f"../results/{EMBEDDING_TYPE}_eval_summary.csv"
summary_per_gene.to_csv(summary_file, index=False)

# Add overall statistics at the end
with open(summary_file, 'a') as f:
    f.write(f"\nOverall,{overall_mean_rmse},{np.nan},{overall_mean_pearson},{np.nan}\n")

print(f"\nSummary statistics saved to {summary_file}")

# Save feature importance weights
weights_df = pd.DataFrame(all_weights)
weights_file = f"../results/{EMBEDDING_TYPE}_feature_importance.csv"
weights_df.to_csv(weights_file, index=False)

print(f"\nFeature importance weights saved to {weights_file}")
print(f"Embedding type used: {EMBEDDING_TYPE}")
print("Done!")
