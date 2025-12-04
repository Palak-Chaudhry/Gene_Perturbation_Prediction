import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import pearsonr
from xgboost import XGBRegressor
import sys


def cv_single_target(X, y, pert_name, n_splits=5):
    """
    Run KFold CV with hyperparameter tuning for one perturbation vector.
    Reports RMSE and Pearson correlation for each fold.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    # Define hyperparameter grid for tuning
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0]
    }

    fold_id = 0
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Base model with fixed parameters
        base_model = XGBRegressor(
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )

        # GridSearchCV with 3-fold inner CV for hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train, y_train)

        # Use best model for prediction
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        corr, _ = pearsonr(y_test, y_pred)

        results.append({
            "perturbation": pert_name,
            "fold": fold_id,
            "rmse": rmse,
            "pearson_corr": corr,
            "best_n_estimators": grid_search.best_params_['n_estimators'],
            "best_max_depth": grid_search.best_params_['max_depth'],
            "best_learning_rate": grid_search.best_params_['learning_rate'],
            "best_subsample": grid_search.best_params_['subsample']
        })

        fold_id += 1

    return results

# ======================
# Get command line arguments
# ======================
if len(sys.argv) < 2:
    raise ValueError("Please provide embedding type as argument: 'tss' or 'gencode'")

EMBEDDING_TYPE = sys.argv[1]

print("=" * 50)
print(f"Embedding type: {EMBEDDING_TYPE}")
print("=" * 50)

# ======================
# Load input data
# ======================
# Configuration: Choose which embeddings to use
# Option 1: TSS-based embeddings (19685 genes, 32 features)
# Option 2: Gencode v49 PC transcripts embeddings (19058 genes, 32 features)

input_file = "../../../data/active_guides_CRISPRa_mean_pop_mean.csv"
mean_pop = pd.read_csv(input_file, index_col=0)

# Load embeddings and metadata based on selected type
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

# Load embeddings (allow_pickle for compatibility)
embeddings = np.load(embeddings_path, allow_pickle=True)
print(f"Embeddings shape: {embeddings.shape}")

# Load gene names from metadata
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

# ======================
# Run CV for each perturbation
# ======================
all_results = []

for i in range(Y.shape[1]):
    pert_name = perturb_names[i]
    print(f"\n>>> Running 5-fold CV for {pert_name} ({i+1}/{Y.shape[1]})")

    results = cv_single_target(X.values, Y.iloc[:, i].values, pert_name)
    all_results.extend(results)

# ======================
# Save results
# ======================
results_df = pd.DataFrame(all_results)

# Include embedding type in output filename
csv_file = f"../results/{EMBEDDING_TYPE}_results.csv"
json_file = f"../results/{EMBEDDING_TYPE}_results.json"

results_df.to_csv(csv_file, index=False)
results_df.to_json(json_file, orient="records", lines=True)

print(f"Done! Results saved to {csv_file} and {json_file}")
print(f"Embedding type used: {EMBEDDING_TYPE}")
