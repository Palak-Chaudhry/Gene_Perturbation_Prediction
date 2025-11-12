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
# List of mean-pop datasets
# ===============================
input_files <- c(
  "../data/active_strong_guides_CRISPRa_mean_pop_zscore.csv"
)

# ===============================
# Load Enformer embeddings
# ===============================
# Configuration: Choose which embeddings to use
# Option 1: TSS-based embeddings (19685 genes, 32 features)
# Option 2: Gencode v49 PC transcripts embeddings (19058 genes, 32 features)
EMBEDDING_TYPE <- "tss"  # Change to "gencode" to use the other embeddings

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

# Load embeddings (allow_pickle for compatibility)
embeddings <- np$load(embeddings_path, allow_pickle = TRUE)
cat("Embeddings shape:", dim(embeddings), "\n")

# Load gene names from metadata
gene_table <- fread(metadata_path, sep = "\t", header = FALSE, col.names = c("index", "gene"))
cat("Loaded", nrow(gene_table), "gene names from metadata\n")

rownames(embeddings) <- gene_table$gene

# ===============================
# Loop over datasets
# ===============================
for(input_file in input_files){
  
  cat("\n==============================\n")
  cat("Processing dataset:", input_file, "\n")
  
  # Step 1. Load mean-pop matrix
  mean_pop <- fread(input_file)
  gene_ids <- mean_pop[[1]]
  mean_pop <- as.matrix(mean_pop[, -1])
  rownames(mean_pop) <- gene_ids
  
  # Step 2. Align genes
  common_genes <- intersect(colnames(mean_pop), rownames(embeddings))
  X <- embeddings[common_genes, ]
  Y <- t(mean_pop[, common_genes, drop=FALSE])
  
  n_samples <- nrow(X)
  
  # Step 3. Parallel setup
  ncores <- parallel::detectCores() - 1
  cl <- makeCluster(ncores, type = "SOCK")
  registerDoSNOW(cl)
  cat("Running on", ncores, "cores...\n")
  
  # progress bar
  total <- ncol(Y)
  pb <- txtProgressBar(min = 0, max = total, style = 3)
  progress <- function(n) setTxtProgressBar(pb, n)
  opts <- list(progress = progress)
  
  # Step 4. k-fold CV for each perturbation
  k <- 5  # number of folds
  set.seed(1)
  folds <- sample(1:k, size = n_samples, replace = TRUE)
  
  results <- foreach(i = seq_len(total), .combine = rbind,
                     .packages = c("glmnet"),
                     .options.snow = opts) %dopar% {
                       
                       y <- scale(Y[, i])
                       fold_cor <- numeric(k)
                       fold_sqrt1mse <- numeric(k)
                       
                       for(f in 1:k){
                         train_idx <- which(folds != f)
                         test_idx  <- which(folds == f)
                         
                         X_train <- X[train_idx, ]
                         X_test  <- X[test_idx, ]
                         y_train <- y[train_idx]
                         y_test  <- y[test_idx]
                         
                         gres <- cv.glmnet(y = y_train, x = X_train, alpha = 0.5,
                                           nlambda = 25, nfolds = 5)
                         
                         preds <- predict(gres, newx = X_test, s = "lambda.min")
                         
                         fold_cor[f] <- cor(preds, y_test)
                         fold_sqrt1mse[f] <- sqrt(1 - min(gres$cvm))
                       }
                       
                       data.frame(
                         perturbation = colnames(Y)[i],
                         mean_cor = mean(fold_cor),
                         mean_sqrt1mse = mean(fold_sqrt1mse),
                         lambda_min = gres$lambda.min,
                         lambda_1se = gres$lambda.1se
                       )
                     }
  
  close(pb)
  stopCluster(cl)
  
  # Step 5. Save results
  output_file <- paste0("glmnet_", EMBEDDING_TYPE, "_results_cv_", basename(input_file))
  fwrite(results, output_file)
  cat("Done! Results saved to", output_file, "\n")
  cat("Embedding type used:", EMBEDDING_TYPE, "\n")
}
