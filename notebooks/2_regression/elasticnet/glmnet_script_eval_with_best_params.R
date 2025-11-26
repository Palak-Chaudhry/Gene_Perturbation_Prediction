# ===============================
# Libraries
# ===============================
library(data.table)
library(glmnet)
library(reticulate)
library(doSNOW)
library(foreach)

# Optional: point to local Python if needed
# use_python("/usr/bin/python3", required = TRUE)

# ===============================
# Get command line arguments
# ===============================
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Please provide embedding type as argument: 'tss' or 'gencode'")
}
EMBEDDING_TYPE <- args[1]

cat("==============================\n")
cat("Embedding type:", EMBEDDING_TYPE, "\n")
cat("==============================\n")

# ===============================
# Load input data
# ===============================
input_file <- "../../../data/active_guides_CRISPRa_mean_pop_mean.csv"

# Load best parameters from previous run
param_file <- paste0("glmnet_", EMBEDDING_TYPE, "_results_cv_", basename(input_file))
cat("Loading best parameters from:", param_file, "\n")
best_params <- fread(param_file)

cat("Loaded best parameters for", nrow(best_params), "genes\n")

# ===============================
# Load Enformer embeddings
# ===============================
np <- import("numpy")

if (EMBEDDING_TYPE == "tss") {
  embeddings_path <- "../../../embeddings/embeddings_enformer_tss.npy"
  metadata_path <- "../../../embeddings/enformer_gene_names.txt"
  cat("Loading TSS embeddings from:", embeddings_path, "\n")
} else if (EMBEDDING_TYPE == "gencode") {
  embeddings_path <- "../../../embeddings/embeddings_enformer_gencode.v49.pc_transcripts.npy"
  metadata_path <- "../../../embeddings/gencode.v49.pc_transcripts_gene_names.txt"
  cat("Loading Gencode v49 PC transcripts embeddings from:", embeddings_path, "\n")
} else {
  stop("Unknown embedding type: ", EMBEDDING_TYPE, ". Use 'tss' or 'gencode'.")
}

# Load embeddings
embeddings <- np$load(embeddings_path, allow_pickle = TRUE)
cat("Embeddings shape:", dim(embeddings), "\n")

# Load gene names from metadata
gene_table <- fread(metadata_path, sep = "\t", header = FALSE, col.names = c("index", "gene"))
cat("Loaded", nrow(gene_table), "gene names from metadata\n")

rownames(embeddings) <- gene_table$gene

# Load mean-pop matrix
mean_pop <- fread(input_file)
gene_ids <- mean_pop[[1]]
mean_pop <- as.matrix(mean_pop[, -1])
rownames(mean_pop) <- gene_ids

# Align genes
common_genes <- intersect(colnames(mean_pop), rownames(embeddings))
X <- embeddings[common_genes, ]
Y <- t(mean_pop[, common_genes, drop=FALSE])

n_samples <- nrow(X)
cat("X shape:", dim(X), "| Y shape:", dim(Y), "\n")

# ===============================
# Set up k-fold CV
# ===============================
k <- 5  # number of folds
set.seed(42)  # Match Python's random_state
folds <- sample(rep(1:k, length.out = n_samples))

# ===============================
# Parallel setup
# ===============================
ncores <- parallel::detectCores() - 1
cl <- makeCluster(ncores, type = "SOCK")
registerDoSNOW(cl)
cat("Running on", ncores, "cores...\n")

# Progress bar
total <- ncol(Y)
pb <- txtProgressBar(min = 0, max = total, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

# ===============================
# Run CV for each perturbation with best params
# ===============================
# Convert best_params to a named vector for easy access in parallel
lambda_vec <- setNames(best_params$lambda_min, best_params$perturbation)

all_results <- foreach(i = seq_len(total), .combine = rbind,
                       .packages = c("glmnet"),
                       .options.snow = opts) %dopar% {

  pert_name <- colnames(Y)[i]

  # Get best lambda for this gene
  lambda_best <- lambda_vec[pert_name]

  if (is.na(lambda_best) || length(lambda_best) == 0) {
    return(NULL)
  }

  y <- scale(Y[, i])
  fold_results <- data.frame()

  for(f in 1:k){
    train_idx <- which(folds != f)
    test_idx  <- which(folds == f)

    X_train <- X[train_idx, ]
    X_test  <- X[test_idx, ]
    y_train <- y[train_idx]
    y_test  <- y[test_idx]

    # Train with alpha=0.5 (elastic net) and best lambda
    model <- glmnet(x = X_train, y = y_train, alpha = 0.5, lambda = lambda_best)

    # Predict
    preds <- predict(model, newx = X_test, s = lambda_best)

    # Calculate metrics
    rmse <- sqrt(mean((y_test - preds)^2))
    pearson <- cor(preds, y_test)

    fold_results <- rbind(fold_results, data.frame(
      perturbation = pert_name,
      fold = f - 1,  # 0-indexed to match Python
      rmse = rmse,
      pearson_corr = pearson
    ))
  }

  fold_results
}

close(pb)
stopCluster(cl)

# ===============================
# Train final models on ALL data and extract coefficients
# ===============================
cat("\n==============================\n")
cat("Training final models on all data...\n")
cat("==============================\n")

all_coefficients <- data.frame()

for(i in seq_len(ncol(Y))){
  pert_name <- colnames(Y)[i]

  # Get best lambda for this gene
  lambda_best <- lambda_vec[pert_name]

  if (is.na(lambda_best) || length(lambda_best) == 0) {
    next
  }

  cat("Training final model for", pert_name, "...\n")

  y <- scale(Y[, i])

  # Train with alpha=0.5 (elastic net) and best lambda on ALL data
  final_model <- glmnet(x = X, y = y, alpha = 0.5, lambda = lambda_best)

  # Extract coefficients
  coefs <- as.matrix(coef(final_model, s = lambda_best))
  coef_names <- rownames(coefs)
  coef_values <- as.vector(coefs)

  # Create a data frame with one row per coefficient
  coef_df <- data.frame(
    perturbation = pert_name,
    feature = coef_names,
    coefficient = coef_values
  )

  all_coefficients <- rbind(all_coefficients, coef_df)
}

# Save coefficients
coef_file <- paste0("glmnet_", EMBEDDING_TYPE, "_coefficients_", basename(input_file))
fwrite(all_coefficients, coef_file)
cat("\nModel coefficients saved to", coef_file, "\n")

# ===============================
# Save per-fold results
# ===============================
output_file <- paste0("glmnet_", EMBEDDING_TYPE, "_eval_with_best_params_", basename(input_file))
fwrite(all_results, output_file)
cat("\nPer-fold results saved to", output_file, "\n")

# ===============================
# Calculate and save summary statistics
# ===============================
summary_per_gene <- all_results[, .(
  mean_rmse = mean(rmse),
  std_rmse = sd(rmse),
  mean_pearson = mean(pearson_corr),
  std_pearson = sd(pearson_corr)
), by = perturbation]

# Calculate overall averages
overall_mean_rmse <- mean(summary_per_gene$mean_rmse)
overall_mean_pearson <- mean(summary_per_gene$mean_pearson)

cat("\n==============================\n")
cat("SUMMARY STATISTICS\n")
cat("==============================\n")
cat("Overall average RMSE:", overall_mean_rmse, "\n")
cat("Overall average Pearson correlation:", overall_mean_pearson, "\n")
cat("==============================\n")

# Save summary
summary_file <- paste0("glmnet_", EMBEDDING_TYPE, "_eval_summary_", basename(input_file))
fwrite(summary_per_gene, summary_file)

# Add overall statistics at the end
overall_row <- data.frame(
  perturbation = "Overall",
  mean_rmse = overall_mean_rmse,
  std_rmse = NA,
  mean_pearson = overall_mean_pearson,
  std_pearson = NA
)
fwrite(overall_row, summary_file, append = TRUE)

cat("\nSummary statistics saved to", summary_file, "\n")
cat("Embedding type used:", EMBEDDING_TYPE, "\n")
cat("Done!\n")
