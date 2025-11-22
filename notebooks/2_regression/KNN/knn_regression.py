# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 20:34:39 2025

@author: sammy
"""

import numpy as np
import pandas as pd
import json

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler


def load_embedding(embedding_path, gene_name_path):
    # Load embedding matrix
    embedding = np.load(embedding_path)      # (N, D)

    # Load gene names
    gene_names = []
    with open(gene_name_path) as f:
        for line in f:
            parts = line.strip().split()
            # parts[0] = index, parts[1] = gene name
            gene_names.append(parts[1])

    # Map gene name → index in embedding
    name_to_idx = {g: i for i, g in enumerate(gene_names)}

    return embedding, name_to_idx


def extract_embedding_for_csv_cols(csv_path, embedding, name_to_idx):
    # Load CSV
    df = pd.read_csv(csv_path, index_col=0)

    col_names = df.columns.tolist()

    # For each gene in CSV columns, find its embedding row
    selected_embeds = []
    missing = []

    for g in col_names:
        if g in name_to_idx:
            idx = name_to_idx[g]
            selected_embeds.append(embedding[idx])
        else:
            missing.append(g)
            selected_embeds.append(np.full(embedding.shape[1], np.nan))

    selected_embeds = np.vstack(selected_embeds)   # (G, D)
    
    valid_mask = ~np.isnan(selected_embeds).any(axis=1)
    selected_embeds = selected_embeds[valid_mask]
    df = df.loc[:, valid_mask]

    return selected_embeds, missing, df


def split_data(X, Y):
    """Split into 70% train, 10% val, 20% test."""
    # After first split: train 70%, temp 30%
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )

    # From temp: val = 1/3 of temp (0.1), test = 2/3 of temp (0.2)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=(2/3), random_state=42
    )

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def nested_cv_split(X, Y, n_outer=5, random_state=42):
    """
    Yields 5 folds:
      - Outer: 80% trainval, 20% test
      - Inner: Split trainval into 70% train, 10% val   (7:1 ratio)
    """
    kf = KFold(n_splits=n_outer, shuffle=True, random_state=random_state)

    for trainval_idx, test_idx in kf.split(X):
        X_trainval = X[trainval_idx]
        Y_trainval = Y[trainval_idx]

        X_test = X[test_idx]
        Y_test = Y[test_idx]

        # Inner split: 7:1 ratio inside trainval
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_trainval, Y_trainval,
            test_size=1/8,      # 1 part validation out of 8 total
            random_state=random_state
        )

        yield X_train, Y_train, X_val, Y_val, X_test, Y_test


class KNNRegressorModel:
    """Simple wrapper for KNN regression."""

    def __init__(self, n_neighbors=5):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)

    def train(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, Y):
        pred = self.predict(X)
        mse = mean_squared_error(Y, pred)
        corr, _ = pearsonr(Y, pred)
        spe_corr, _ = spearmanr(Y, pred)
        return mse, corr, spe_corr


def main():
    
    # ----------------------------------------------------------------------
    # Define 
    # ----------------------------------------------------------------------
    
    embedding_types = ['gene', 'TSS']
    embedding_files = ['../../../embeddings/embeddings_enformer_gencode.v49.pc_transcripts.npy',
                       '../../../embeddings/embeddings_enformer_tss.npy']
    gene_names = ['../../../embeddings/gencode.v49.pc_transcripts_gene_names.txt',
                  '../../../embeddings/enformer_gene_names.txt']
    data_file = '../../../data/active_guides_CRISPRa_mean_pop_mean.csv'
    
    # ----------------------------------------------------------------------
    # Performance matrices
    # ----------------------------------------------------------------------
    
    performances = []
    
    for i in range(len(embedding_types)):
        
        # ----------------------------------------------------------------------
        # Load data
        # ----------------------------------------------------------------------
        embedding, name_to_idx = load_embedding(embedding_files[i], gene_names[i])
        
        selected_embedding, missing_genes, data = extract_embedding_for_csv_cols(
                            data_file, embedding, name_to_idx)
        
        embedding_type_performance = []
        
        for j in range(len(data)):
            
            X = selected_embedding
            Y = data.iloc[j,:].values
            
            output_dict = {'TF': data.index[j],
                           'k': None,
                           'MSE': None,
                           'Pearson': None,
                           'Spearman': None}
            
            mse = np.zeros(5)
            pearson = np.zeros(5)
            spearman = np.zeros(5)
            
            global_k = None
            
            for fold_id, (X_train, Y_train, X_val, Y_val, X_test, Y_test) in enumerate(
                nested_cv_split(X, Y, n_outer=5)):
                
                print("===== Fold", fold_id, "=====")

                # Standardization
                scaler = StandardScaler()
                scaler.fit(X_train)
            
                X_train_s = scaler.transform(X_train)
                X_val_s   = scaler.transform(X_val)
            
                # -------------------------------------------------------------
                # Hyperparameter search ONLY on the first fold
                # -------------------------------------------------------------
                if fold_id == 0:
                    candidates = [5, 10, 25, 50, 75, 100]
                    best_k = None
                    best_val_score = np.inf
                    
                    for k in candidates:
                        model = KNNRegressorModel(n_neighbors=k)
                        model.train(X_train_s, Y_train)
                        val_mse, _, _ = model.evaluate(X_val_s, Y_val)
            
                        if val_mse < best_val_score:
                            best_val_score = val_mse
                            best_k = k
            
                    global_k = best_k
                    print(f"[Fold 0] Selected global k = {global_k}")
            
                else:
                    # later folds use the k chosen from fold 0
                    best_k = global_k
                    print(f"[Fold {fold_id}] Using fixed k = {best_k}")
            
                # -------------------------------------------------------------
                # Retrain using both train + val (80% block)
                # -------------------------------------------------------------
                X_comb = np.vstack([X_train, X_val])
                Y_comb = np.hstack([Y_train, Y_val])
                
                # Standardization
                scaler_comb = StandardScaler()
                scaler_comb.fit(X_comb)
            
                X_comb_s = scaler_comb.transform(X_comb)
                X_test_s  = scaler_comb.transform(X_test)
            
                final = KNNRegressorModel(n_neighbors=best_k)
                final.train(X_comb_s, Y_comb)
            
                # -------------------------------------------------------------
                # Evaluate on outer test block (20%)
                # -------------------------------------------------------------
                test_mse, test_corr, test_spear = final.evaluate(X_test_s, Y_test)
                print(f"Test MSE = {test_mse}")
                print(f"Test Pearson = {test_corr}")
                
                mse[fold_id] = test_mse
                pearson[fold_id] = test_corr
                spearman[fold_id] = test_spear
                
            
            output_dict['k'] = global_k
            output_dict['MSE'] = float(np.mean(mse))
            output_dict['Pearson'] = float(np.mean(pearson))
            output_dict['Spearman'] = float(np.mean(spearman))
            
            embedding_type_performance.append(output_dict)
            
        performances.append(embedding_type_performance)
        
        with open("results.json", "w") as f:
            json.dump(performances, f, indent=2)
        
        
if __name__ == "__main__":
    main()
