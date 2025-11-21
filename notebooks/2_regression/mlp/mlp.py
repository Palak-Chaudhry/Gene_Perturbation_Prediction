import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from joblib import Parallel, delayed

def build_mlp_pipeline(hidden_layer_sizes=(64, 32),
                       alpha=1e-3,
                       learning_rate_init=1e-3,
                       random_state=42):
    """
    Build a Pipeline: StandardScaler -> MLPRegressor
    with the given hyperparameters.
    """
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,                  # L2 regularization
        learning_rate="adaptive",
        learning_rate_init=learning_rate_init,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=random_state
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", mlp)
    ])


def cv_tune_mlp_on_train(X_train, y_train,
                         param_grid,
                         n_splits=5,
                         random_state=42):
    """
    Hyperparameter tuning using K-fold CV on the 80% train set.

    For each hyperparameter configuration:
      - Run K-fold CV on (X_train, y_train)
      - Compute mean RMSE across folds
    Choose the configuration with the lowest mean RMSE.

    Returns:
      best_params (dict), best_mean_rmse (float)
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    best_rmse = np.inf
    best_params = None

    for params in param_grid:
        fold_rmses = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = build_mlp_pipeline(
                hidden_layer_sizes=params["hidden_layer_sizes"],
                alpha=params["alpha"],
                learning_rate_init=params["learning_rate_init"],
                random_state=random_state
            )

            model.fit(X_tr, y_tr)
            y_val_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            fold_rmses.append(rmse)

        mean_rmse = np.mean(fold_rmses)

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params

    return best_params, best_rmse


def run_tf_training_with_heldout_test(X, y, pert_name,
                                      train_idx,
                                      test_idx,
                                      param_grid,
                                      cv_splits=5,
                                      random_state=42):
    """
    For one TF (one target y):
      - Use global train_idx/test_idx to define 80%/20% split.
      - On the 80% (train) perform CV-based hyperparameter tuning.
      - Retrain best model on full 80%.
      - Evaluate once on the 20% held-out test set.
    """

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Hyperparameter tuning on train set only
    best_params, best_cv_rmse = cv_tune_mlp_on_train(
        X_train, y_train,
        param_grid=param_grid,
        n_splits=cv_splits,
        random_state=random_state
    )

    # Train final model on full 80% using best hyperparams
    final_model = build_mlp_pipeline(
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        alpha=best_params["alpha"],
        learning_rate_init=best_params["learning_rate_init"],
        random_state=random_state
    )
    final_model.fit(X_train, y_train)

    # Evaluate on the untouched 20%
    y_pred = final_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_corr, _ = pearsonr(y_test, y_pred)

    return {
        "perturbation": pert_name,
        "rmse_test": test_rmse,
        "pearson_corr_test": test_corr,
        "cv_rmse_mean": best_cv_rmse,
        "hidden_layer_sizes": best_params["hidden_layer_sizes"],
        "alpha": best_params["alpha"],
        "learning_rate_init": best_params["learning_rate_init"]
    }


# ======================
# Load input data
# ======================
EMBEDDING_TYPE = "gencode" # "tss"  # or 

input_file = "/ocean/projects/cis240075p/pchaudhr/Gene_Perturbation_Prediction/data/active_guides_CRISPRa_mean_pop_mean.csv"
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

embeddings = np.load(embeddings_path, allow_pickle=True)
print(f"Embeddings shape: {embeddings.shape}")

meta_table = pd.read_csv(metadata_path, sep="\t", header=None, names=["index", "gene"])
print(f"Loaded {len(meta_table)} gene names from metadata")

embeddings = pd.DataFrame(embeddings, index=meta_table['gene'])

# Align datasets: same genes in embeddings and mean_pop
embeddings = embeddings.groupby(embeddings.index).mean()
common_genes = mean_pop.columns.intersection(embeddings.index)
X = embeddings.loc[common_genes]            # genes x features
Y = mean_pop[common_genes].T               # genes x perturbations

perturb_names = Y.columns

print("X shape:", X.shape, "| Y shape:", Y.shape)

# ======================
# Create a single 80/20 train/test split for genes
# ======================
n_genes = X.shape[0]
all_indices = np.arange(n_genes)

train_idx, test_idx = train_test_split(
    all_indices,
    test_size=0.2,
    shuffle=True,
    random_state=42
)

print(f"Train genes: {len(train_idx)}, Test genes: {len(test_idx)}")

# Convert X to numpy once
X_np = X.values

# ======================
# Define hyperparameter grid for tuning
# ======================
param_grid = [
    {"hidden_layer_sizes": (32,),
     "alpha": 1e-3,
     "learning_rate_init": 1e-3},

    {"hidden_layer_sizes": (64,),
     "alpha": 1e-3,
     "learning_rate_init": 1e-3},

    {"hidden_layer_sizes": (64, 32),
     "alpha": 1e-3,
     "learning_rate_init": 1e-3},

    {"hidden_layer_sizes": (64, 32),
     "alpha": 1e-2,
     "learning_rate_init": 1e-3},

    {"hidden_layer_sizes": (128, 64, 32),
     "alpha": 1e-3,
     "learning_rate_init": 1e-4},

    {"hidden_layer_sizes": (128, 64, 32),
     "alpha": 1e-2,
     "learning_rate_init": 1e-4},
]
# ======================
# Run training/tuning/testing for each perturbation (TF) IN PARALLEL
# ======================

def process_one_tf(j):
    pert_name = perturb_names[j]
    print(f"\n>>> TF {pert_name} ({j+1}/{Y.shape[1]})")

    y_full = Y.iloc[:, j].values

    res = run_tf_training_with_heldout_test(
        X=X_np,
        y=y_full,
        pert_name=pert_name,
        train_idx=train_idx,
        test_idx=test_idx,
        param_grid=param_grid,
        cv_splits=5,
        random_state=42
    )
    return res


# Number of parallel workers:
# -1 = use all available cores
# or set explicitly, e.g., n_jobs=8
N_JOBS = -1

# On Linux/HPC this is fine directly.
# On Windows you'd want this inside: if __name__ == "__main__": ...
all_results = Parallel(n_jobs=N_JOBS)(
    delayed(process_one_tf)(j) for j in range(Y.shape[1])
)

# # ======================
# # Run training/tuning/testing for each perturbation (TF)
# # ======================
# all_results = []

# for j in range(Y.shape[1]):
#     pert_name = perturb_names[j]
#     print(f"\n>>> TF {pert_name} ({j+1}/{Y.shape[1]})")

#     y_full = Y.iloc[:, j].values

#     res = run_tf_training_with_heldout_test(
#         X=X_np,
#         y=y_full,
#         pert_name=pert_name,
#         train_idx=train_idx,
#         test_idx=test_idx,
#         param_grid=param_grid,
#         cv_splits=5,
#         random_state=42
#     )

#     all_results.append(res)

# ======================
# Save results
# ======================
results_df = pd.DataFrame(all_results)

csv_file = f"enformer_{EMBEDDING_TYPE}_mlp_80_20_heldout_results_" + input_file.split("/")[-1] + ".csv"
json_file = f"enformer_{EMBEDDING_TYPE}_mlp_80_20_heldout_results_" + input_file.split("/")[-1] + ".json"

results_df.to_csv(csv_file, index=False)
results_df.to_json(json_file, orient="records", lines=True)

print(f"Done! Results saved to {csv_file} and {json_file}")
print(f"Embedding type used: {EMBEDDING_TYPE}")
