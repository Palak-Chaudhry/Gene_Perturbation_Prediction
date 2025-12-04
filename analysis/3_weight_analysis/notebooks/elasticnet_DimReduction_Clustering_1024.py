#!/usr/bin/env python
# Converted from elasticnet_DimReduction_Clustering_1024.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import umap
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Paths to feature importance files
gencode_path = '../../2_regression/elasticnet_1024/results/gencode_coefficients.csv'
tss_path = '../../2_regression/elasticnet_1024/results/tss_coefficients.csv'

# Load data
df_gencode = pd.read_csv(gencode_path)
wide = df_gencode.pivot(index="perturbation", columns="feature", values="coefficient")
df_gencode = wide.reindex(
    ['(Intercept)'] + [f"V{i}" for i in range(1, 33)],
    axis=1
).reset_index()

df_tss = pd.read_csv(tss_path)
wide = df_tss.pivot(index="perturbation", columns="feature", values="coefficient")
df_tss = wide.reindex(
    ['(Intercept)'] + [f"V{i}" for i in range(1, 33)],
    axis=1
).reset_index()


print(f"Gencode feature importance shape: {df_gencode.shape}")
print(f"TSS feature importance shape: {df_tss.shape}")
print(f"\nNumber of perturbations: {len(df_gencode)}")
print(f"Number of features: {len([col for col in df_gencode.columns if col.startswith('feature_')])}")

# Display first few rows
df_gencode.head()

# Extract feature columns (32 features)
feature_cols = [col for col in df_gencode.columns if col.startswith('(') or col.startswith('V')]
print(f"Feature columns: {len(feature_cols)}")

# Extract feature matrices
X_gencode = df_gencode[feature_cols].values
X_tss = df_tss[feature_cols].values
perturbations = df_gencode['perturbation'].values

print(f"\nGencode feature matrix shape: {X_gencode.shape}")
print(f"TSS feature matrix shape: {X_tss.shape}")

# Standardize the features (important for dimensionality reduction)
scaler_gencode = StandardScaler()
scaler_tss = StandardScaler()

X_gencode_scaled = scaler_gencode.fit_transform(X_gencode)
X_tss_scaled = scaler_tss.fit_transform(X_tss)

print("\nFeatures standardized successfully")

# PCA
pca_gencode = PCA(n_components=2, random_state=42)
X_gencode_pca = pca_gencode.fit_transform(X_gencode_scaled)

pca_tss = PCA(n_components=2, random_state=42)
X_tss_pca = pca_tss.fit_transform(X_tss_scaled)

print("PCA completed")
print(f"Gencode explained variance: {pca_gencode.explained_variance_ratio_.sum():.4f}")
print(f"TSS explained variance: {pca_tss.explained_variance_ratio_.sum():.4f}")

# Perform PCA with ALL components to see full variance explained
pca_full_gencode = PCA(random_state=42)
pca_full_gencode.fit(X_gencode_scaled)

pca_full_tss = PCA(random_state=42)
pca_full_tss.fit(X_tss_scaled)

print(f"Total number of components: {len(pca_full_gencode.components_)}")
print(f"Shape of feature matrix: {X_gencode_scaled.shape}")

# Create scree plots showing variance explained by each PC
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Gencode - Variance explained per component
axes[0, 0].bar(range(1, len(pca_full_gencode.explained_variance_ratio_) + 1), 
               pca_full_gencode.explained_variance_ratio_,
               alpha=0.7, color='steelblue')
axes[0, 0].set_xlabel('Principal Component')
axes[0, 0].set_ylabel('Variance Explained Ratio')
axes[0, 0].set_title('Gencode - Variance Explained per PC', fontweight='bold', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)

# Gencode - Cumulative variance explained
cumsum_gencode = np.cumsum(pca_full_gencode.explained_variance_ratio_)
axes[0, 1].plot(range(1, len(cumsum_gencode) + 1), cumsum_gencode, 
                'o-', linewidth=2, markersize=6, color='steelblue')
axes[0, 1].axhline(y=0.90, color='r', linestyle='--', label='90% variance')
axes[0, 1].axhline(y=0.95, color='orange', linestyle='--', label='95% variance')
axes[0, 1].set_xlabel('Number of Components')
axes[0, 1].set_ylabel('Cumulative Variance Explained')
axes[0, 1].set_title('Gencode - Cumulative Variance Explained', fontweight='bold', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# TSS - Variance explained per component
axes[1, 0].bar(range(1, len(pca_full_tss.explained_variance_ratio_) + 1), 
               pca_full_tss.explained_variance_ratio_,
               alpha=0.7, color='darkorange')
axes[1, 0].set_xlabel('Principal Component')
axes[1, 0].set_ylabel('Variance Explained Ratio')
axes[1, 0].set_title('TSS - Variance Explained per PC', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# TSS - Cumulative variance explained
cumsum_tss = np.cumsum(pca_full_tss.explained_variance_ratio_)
axes[1, 1].plot(range(1, len(cumsum_tss) + 1), cumsum_tss, 
                'o-', linewidth=2, markersize=6, color='darkorange')
axes[1, 1].axhline(y=0.90, color='r', linestyle='--', label='90% variance')
axes[1, 1].axhline(y=0.95, color='orange', linestyle='--', label='95% variance')
axes[1, 1].set_xlabel('Number of Components')
axes[1, 1].set_ylabel('Cumulative Variance Explained')
axes[1, 1].set_title('TSS - Cumulative Variance Explained', fontweight='bold', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('pca_variance_explained.png', dpi=300, bbox_inches='tight')
plt.show()

# Find number of components for 90% and 95% variance
n_components_90_gencode = np.argmax(cumsum_gencode >= 0.90) + 1
n_components_95_gencode = np.argmax(cumsum_gencode >= 0.95) + 1
n_components_90_tss = np.argmax(cumsum_tss >= 0.90) + 1
n_components_95_tss = np.argmax(cumsum_tss >= 0.95) + 1

print(f"\nGencode:")
print(f"  Components for 90% variance: {n_components_90_gencode}")
print(f"  Components for 95% variance: {n_components_95_gencode}")
print(f"\nTSS:")
print(f"  Components for 90% variance: {n_components_90_tss}")
print(f"  Components for 95% variance: {n_components_95_tss}")

# t-SNE
tsne_gencode = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_gencode_tsne = tsne_gencode.fit_transform(X_gencode_scaled)

tsne_tss = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tss_tsne = tsne_tss.fit_transform(X_tss_scaled)

print("t-SNE completed")

# UMAP
umap_gencode = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
X_gencode_umap = umap_gencode.fit_transform(X_gencode_scaled)

umap_tss = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
X_tss_umap = umap_tss.fit_transform(X_tss_scaled)

print("UMAP completed")

# Visualize all dimensionality reductions for TSS
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X_tss_pca[:, 0], X_tss_pca[:, 1], alpha=0.6, s=50, color='orange')
axes[0].set_title('PCA - TSS', fontsize=14, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca_tss.explained_variance_ratio_[0]:.2%})')
axes[0].set_ylabel(f'PC2 ({pca_tss.explained_variance_ratio_[1]:.2%})')

axes[1].scatter(X_tss_tsne[:, 0], X_tss_tsne[:, 1], alpha=0.6, s=50, color='orange')
axes[1].set_title('t-SNE - TSS', fontsize=14, fontweight='bold')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

axes[2].scatter(X_tss_umap[:, 0], X_tss_umap[:, 1], alpha=0.6, s=50, color='orange')
axes[2].set_title('UMAP - TSS', fontsize=14, fontweight='bold')
axes[2].set_xlabel('UMAP 1')
axes[2].set_ylabel('UMAP 2')

plt.tight_layout()
# plt.savefig('tss_dimensionality_reduction.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize all dimensionality reductions for Gencode
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X_gencode_pca[:, 0], X_gencode_pca[:, 1], alpha=0.6, s=50)
axes[0].set_title('PCA - Gencode', fontsize=14, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca_gencode.explained_variance_ratio_[0]:.2%})')
axes[0].set_ylabel(f'PC2 ({pca_gencode.explained_variance_ratio_[1]:.2%})')

axes[1].scatter(X_gencode_tsne[:, 0], X_gencode_tsne[:, 1], alpha=0.6, s=50)
axes[1].set_title('t-SNE - Gencode', fontsize=14, fontweight='bold')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

axes[2].scatter(X_gencode_umap[:, 0], X_gencode_umap[:, 1], alpha=0.6, s=50)
axes[2].set_title('UMAP - Gencode', fontsize=14, fontweight='bold')
axes[2].set_xlabel('UMAP 1')
axes[2].set_ylabel('UMAP 2')

plt.tight_layout()
# plt.savefig('gencode_dimensionality_reduction.png', dpi=300, bbox_inches='tight')
plt.show()

# Set optimal number of clusters (adjust this based on the plots above)
optimal_k = 4

def apply_clustering(X, n_clusters=5):
    """Apply both KMeans and GMM clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
    gmm_labels = gmm.fit_predict(X)
    
    return kmeans_labels, gmm_labels

# Apply clustering to ALL THREE dimensionality reductions for BOTH embeddings

# Gencode
kmeans_labels_gencode_pca, gmm_labels_gencode_pca = apply_clustering(X_gencode_pca, optimal_k)
kmeans_labels_gencode_tsne, gmm_labels_gencode_tsne = apply_clustering(X_gencode_tsne, optimal_k)
kmeans_labels_gencode_umap, gmm_labels_gencode_umap = apply_clustering(X_gencode_umap, optimal_k)

# TSS
kmeans_labels_tss_pca, gmm_labels_tss_pca = apply_clustering(X_tss_pca, optimal_k)
kmeans_labels_tss_tsne, gmm_labels_tss_tsne = apply_clustering(X_tss_tsne, optimal_k)
kmeans_labels_tss_umap, gmm_labels_tss_umap = apply_clustering(X_tss_umap, optimal_k)

print(f"Clustering completed with k={optimal_k} clusters!")

import json

# Create dictionaries mapping perturbation names to cluster assignments
print("Saving cluster assignments to JSON files...")

# Gencode cluster assignments
gencode_clusters = {
    'PCA': {
        'KMeans': {pert: int(label) for pert, label in zip(perturbations, kmeans_labels_gencode_pca)},
        'GMM': {pert: int(label) for pert, label in zip(perturbations, gmm_labels_gencode_pca)}
    },
    'tSNE': {
        'KMeans': {pert: int(label) for pert, label in zip(perturbations, kmeans_labels_gencode_tsne)},
        'GMM': {pert: int(label) for pert, label in zip(perturbations, gmm_labels_gencode_tsne)}
    },
    'UMAP': {
        'KMeans': {pert: int(label) for pert, label in zip(perturbations, kmeans_labels_gencode_umap)},
        'GMM': {pert: int(label) for pert, label in zip(perturbations, gmm_labels_gencode_umap)}
    }
}

# TSS cluster assignments
tss_clusters = {
    'PCA': {
        'KMeans': {pert: int(label) for pert, label in zip(perturbations, kmeans_labels_tss_pca)},
        'GMM': {pert: int(label) for pert, label in zip(perturbations, gmm_labels_tss_pca)}
    },
    'tSNE': {
        'KMeans': {pert: int(label) for pert, label in zip(perturbations, kmeans_labels_tss_tsne)},
        'GMM': {pert: int(label) for pert, label in zip(perturbations, gmm_labels_tss_tsne)}
    },
    'UMAP': {
        'KMeans': {pert: int(label) for pert, label in zip(perturbations, kmeans_labels_tss_umap)},
        'GMM': {pert: int(label) for pert, label in zip(perturbations, gmm_labels_tss_umap)}
    }
}

# Save to JSON files
with open('../cluster_assignments/elasticnet_1024_gencode_2d_k4.json', 'w') as f:
    json.dump(gencode_clusters, f, indent=2)
print("Saved: elasticnet_1024_gencode_cluster_assignments_2d_4k.json")

with open('../cluster_assignments/elasticnet_1024_tss_2d_k4.json', 'w') as f:
    json.dump(tss_clusters, f, indent=2)
print("Saved: elasticnet_1024_tss_cluster_assignments_2d_4k.json")

print("\nCluster assignments saved successfully!")
print(f"Total TFs/perturbations: {len(perturbations)}")
print(f"Number of clusters: {optimal_k}")

# Visualize TSS clustering results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

data_list = [X_tss_pca, X_tss_tsne, X_tss_umap]
kmeans_labels_list = [kmeans_labels_tss_pca, kmeans_labels_tss_tsne, kmeans_labels_tss_umap]
gmm_labels_list = [gmm_labels_tss_pca, gmm_labels_tss_tsne, gmm_labels_tss_umap]
methods = ['PCA', 't-SNE', 'UMAP']

for col, (X_reduced, kmeans_labels, gmm_labels, method) in enumerate(zip(data_list, kmeans_labels_list, gmm_labels_list, methods)):
    # KMeans clustering
    scatter = axes[0, col].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                    c=kmeans_labels, cmap='tab10', alpha=0.7, s=50)
    axes[0, col].set_title(f'KMeans - {method}', fontsize=14, fontweight='bold')
    axes[0, col].set_xlabel(f'{method} 1')
    axes[0, col].set_ylabel(f'{method} 2')
    plt.colorbar(scatter, ax=axes[0, col], label='Cluster')
    
    # GMM clustering
    scatter = axes[1, col].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                    c=gmm_labels, cmap='tab10', alpha=0.7, s=50)
    axes[1, col].set_title(f'GMM - {method}', fontsize=14, fontweight='bold')
    axes[1, col].set_xlabel(f'{method} 1')
    axes[1, col].set_ylabel(f'{method} 2')
    plt.colorbar(scatter, ax=axes[1, col], label='Cluster')

plt.suptitle(f'ElasticNet - TSS - Clustering Results (k={optimal_k})', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/1024dim/tss_clustering_2d.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize Gencode clustering results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

data_list = [X_gencode_pca, X_gencode_tsne, X_gencode_umap]
kmeans_labels_list = [kmeans_labels_gencode_pca, kmeans_labels_gencode_tsne, kmeans_labels_gencode_umap]
gmm_labels_list = [gmm_labels_gencode_pca, gmm_labels_gencode_tsne, gmm_labels_gencode_umap]
methods = ['PCA', 't-SNE', 'UMAP']

for col, (X_reduced, kmeans_labels, gmm_labels, method) in enumerate(zip(data_list, kmeans_labels_list, gmm_labels_list, methods)):
    # KMeans clustering
    scatter = axes[0, col].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                    c=kmeans_labels, cmap='tab10', alpha=0.7, s=50)
    axes[0, col].set_title(f'KMeans - {method}', fontsize=14, fontweight='bold')
    axes[0, col].set_xlabel(f'{method} 1')
    axes[0, col].set_ylabel(f'{method} 2')
    plt.colorbar(scatter, ax=axes[0, col], label='Cluster')
    
    # GMM clustering
    scatter = axes[1, col].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                    c=gmm_labels, cmap='tab10', alpha=0.7, s=50)
    axes[1, col].set_title(f'GMM - {method}', fontsize=14, fontweight='bold')
    axes[1, col].set_xlabel(f'{method} 1')
    axes[1, col].set_ylabel(f'{method} 2')
    plt.colorbar(scatter, ax=axes[1, col], label='Cluster')

plt.suptitle(f'ElasticNet - Gencode - Clustering Results (k={optimal_k})', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/1024dim/gencode_clustering_2d.png', dpi=300, bbox_inches='tight')
plt.show()

# PCA
pca_gencode = PCA(n_components=3, random_state=42)
X_gencode_pca = pca_gencode.fit_transform(X_gencode_scaled)

pca_tss = PCA(n_components=3, random_state=42)
X_tss_pca = pca_tss.fit_transform(X_tss_scaled)

print("PCA completed")
print(f"Gencode explained variance: {pca_gencode.explained_variance_ratio_.sum():.4f}")
print(f"TSS explained variance: {pca_tss.explained_variance_ratio_.sum():.4f}")
# t-SNE
tsne_gencode = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1000)
X_gencode_tsne = tsne_gencode.fit_transform(X_gencode_scaled)

tsne_tss = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1000)
X_tss_tsne = tsne_tss.fit_transform(X_tss_scaled)

print("t-SNE completed")
# UMAP
umap_gencode = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
X_gencode_umap = umap_gencode.fit_transform(X_gencode_scaled)

umap_tss = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
X_tss_umap = umap_tss.fit_transform(X_tss_scaled)

print("UMAP completed")
# Visualize all dimensionality reductions for TSS
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X_tss_pca[:, 0], X_tss_pca[:, 1], alpha=0.6, s=50, color='orange')
axes[0].set_title('PCA - TSS', fontsize=14, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca_tss.explained_variance_ratio_[0]:.2%})')
axes[0].set_ylabel(f'PC2 ({pca_tss.explained_variance_ratio_[1]:.2%})')

axes[1].scatter(X_tss_tsne[:, 0], X_tss_tsne[:, 1], alpha=0.6, s=50, color='orange')
axes[1].set_title('t-SNE - TSS', fontsize=14, fontweight='bold')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

axes[2].scatter(X_tss_umap[:, 0], X_tss_umap[:, 1], alpha=0.6, s=50, color='orange')
axes[2].set_title('UMAP - TSS', fontsize=14, fontweight='bold')
axes[2].set_xlabel('UMAP 1')
axes[2].set_ylabel('UMAP 2')

plt.tight_layout()
# plt.savefig('tss_dimensionality_reduction.png', dpi=300, bbox_inches='tight')
plt.show()
# Visualize all dimensionality reductions for Gencode
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X_gencode_pca[:, 0], X_gencode_pca[:, 1], alpha=0.6, s=50)
axes[0].set_title('PCA - Gencode', fontsize=14, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca_gencode.explained_variance_ratio_[0]:.2%})')
axes[0].set_ylabel(f'PC2 ({pca_gencode.explained_variance_ratio_[1]:.2%})')

axes[1].scatter(X_gencode_tsne[:, 0], X_gencode_tsne[:, 1], alpha=0.6, s=50)
axes[1].set_title('t-SNE - Gencode', fontsize=14, fontweight='bold')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

axes[2].scatter(X_gencode_umap[:, 0], X_gencode_umap[:, 1], alpha=0.6, s=50)
axes[2].set_title('UMAP - Gencode', fontsize=14, fontweight='bold')
axes[2].set_xlabel('UMAP 1')
axes[2].set_ylabel('UMAP 2')

plt.tight_layout()
# plt.savefig('gencode_dimensionality_reduction.png', dpi=300, bbox_inches='tight')
plt.show()

# Set optimal number of clusters (adjust this based on the plots above)
optimal_k = 4

def apply_clustering(X, n_clusters=5):
    """Apply both KMeans and GMM clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
    gmm_labels = gmm.fit_predict(X)
    
    return kmeans_labels, gmm_labels


# Gencode
kmeans_labels_gencode_pca, gmm_labels_gencode_pca = apply_clustering(X_gencode_pca, optimal_k)
kmeans_labels_gencode_tsne, gmm_labels_gencode_tsne = apply_clustering(X_gencode_tsne, optimal_k)
kmeans_labels_gencode_umap, gmm_labels_gencode_umap = apply_clustering(X_gencode_umap, optimal_k)

# TSS
kmeans_labels_tss_pca, gmm_labels_tss_pca = apply_clustering(X_tss_pca, optimal_k)
kmeans_labels_tss_tsne, gmm_labels_tss_tsne = apply_clustering(X_tss_tsne, optimal_k)
kmeans_labels_tss_umap, gmm_labels_tss_umap = apply_clustering(X_tss_umap, optimal_k)

print(f"Clustering completed with k={optimal_k} clusters!")

import json

# Create dictionaries mapping perturbation names to cluster assignments
print("Saving cluster assignments to JSON files...")

# Gencode cluster assignments
gencode_clusters = {
    'PCA': {
        'KMeans': {pert: int(label) for pert, label in zip(perturbations, kmeans_labels_gencode_pca)},
        'GMM': {pert: int(label) for pert, label in zip(perturbations, gmm_labels_gencode_pca)}
    },
    'tSNE': {
        'KMeans': {pert: int(label) for pert, label in zip(perturbations, kmeans_labels_gencode_tsne)},
        'GMM': {pert: int(label) for pert, label in zip(perturbations, gmm_labels_gencode_tsne)}
    },
    'UMAP': {
        'KMeans': {pert: int(label) for pert, label in zip(perturbations, kmeans_labels_gencode_umap)},
        'GMM': {pert: int(label) for pert, label in zip(perturbations, gmm_labels_gencode_umap)}
    }
}

# TSS cluster assignments
tss_clusters = {
    'PCA': {
        'KMeans': {pert: int(label) for pert, label in zip(perturbations, kmeans_labels_tss_pca)},
        'GMM': {pert: int(label) for pert, label in zip(perturbations, gmm_labels_tss_pca)}
    },
    'tSNE': {
        'KMeans': {pert: int(label) for pert, label in zip(perturbations, kmeans_labels_tss_tsne)},
        'GMM': {pert: int(label) for pert, label in zip(perturbations, gmm_labels_tss_tsne)}
    },
    'UMAP': {
        'KMeans': {pert: int(label) for pert, label in zip(perturbations, kmeans_labels_tss_umap)},
        'GMM': {pert: int(label) for pert, label in zip(perturbations, gmm_labels_tss_umap)}
    }
}

# Save to JSON files
with open('../cluster_assignments/elasticnet_1024_gencode_3d_k4.json', 'w') as f:
    json.dump(gencode_clusters, f, indent=2)
print("Saved: elasticnet_1024_gencode_cluster_assignments_3d_4k.json")

with open('../cluster_assignments/elasticnet_1024_tss_3d_k4.json', 'w') as f:
    json.dump(tss_clusters, f, indent=2)
print("Saved: elasticnet_1024_tss_cluster_assignments_3d_4k.json")

print("\nCluster assignments saved successfully!")
print(f"Total TFs/perturbations: {len(perturbations)}")
print(f"Number of clusters: {optimal_k}")

# Visualize TSS clustering results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

data_list = [X_tss_pca, X_tss_tsne, X_tss_umap]
kmeans_labels_list = [kmeans_labels_tss_pca, kmeans_labels_tss_tsne, kmeans_labels_tss_umap]
gmm_labels_list = [gmm_labels_tss_pca, gmm_labels_tss_tsne, gmm_labels_tss_umap]
methods = ['PCA', 't-SNE', 'UMAP']

for col, (X_reduced, kmeans_labels, gmm_labels, method) in enumerate(zip(data_list, kmeans_labels_list, gmm_labels_list, methods)):
    # KMeans clustering
    scatter = axes[0, col].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                    c=kmeans_labels, cmap='tab10', alpha=0.7, s=50)
    axes[0, col].set_title(f'KMeans - {method}', fontsize=14, fontweight='bold')
    axes[0, col].set_xlabel(f'{method} 1')
    axes[0, col].set_ylabel(f'{method} 2')
    plt.colorbar(scatter, ax=axes[0, col], label='Cluster')
    
    # GMM clustering
    scatter = axes[1, col].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                    c=gmm_labels, cmap='tab10', alpha=0.7, s=50)
    axes[1, col].set_title(f'GMM - {method}', fontsize=14, fontweight='bold')
    axes[1, col].set_xlabel(f'{method} 1')
    axes[1, col].set_ylabel(f'{method} 2')
    plt.colorbar(scatter, ax=axes[1, col], label='Cluster')

plt.suptitle(f'ElasticNet - TSS - Clustering Results (k={optimal_k})', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/1024dim/tss_clustering_3d.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize Gencode clustering results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

data_list = [X_gencode_pca, X_gencode_tsne, X_gencode_umap]
kmeans_labels_list = [kmeans_labels_gencode_pca, kmeans_labels_gencode_tsne, kmeans_labels_gencode_umap]
gmm_labels_list = [gmm_labels_gencode_pca, gmm_labels_gencode_tsne, gmm_labels_gencode_umap]
methods = ['PCA', 't-SNE', 'UMAP']

for col, (X_reduced, kmeans_labels, gmm_labels, method) in enumerate(zip(data_list, kmeans_labels_list, gmm_labels_list, methods)):
    # KMeans clustering
    scatter = axes[0, col].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                    c=kmeans_labels, cmap='tab10', alpha=0.7, s=50)
    axes[0, col].set_title(f'KMeans - {method}', fontsize=14, fontweight='bold')
    axes[0, col].set_xlabel(f'{method} 1')
    axes[0, col].set_ylabel(f'{method} 2')
    plt.colorbar(scatter, ax=axes[0, col], label='Cluster')
    
    # GMM clustering
    scatter = axes[1, col].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                    c=gmm_labels, cmap='tab10', alpha=0.7, s=50)
    axes[1, col].set_title(f'GMM - {method}', fontsize=14, fontweight='bold')
    axes[1, col].set_xlabel(f'{method} 1')
    axes[1, col].set_ylabel(f'{method} 2')
    plt.colorbar(scatter, ax=axes[1, col], label='Cluster')

plt.suptitle(f'ElasticNet - Gencode - Clustering Results (k={optimal_k})', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/1024dim/gencode_clustering_3d.png', dpi=300, bbox_inches='tight')
plt.show()

import plotly.express as px
import plotly.graph_objects as go
import json

print("Creating 3D UMAP visualizations...")

# For Gencode - KMeans
df_plot_gencode_kmeans = pd.DataFrame({
    'x': X_gencode_umap[:, 0],
    'y': X_gencode_umap[:, 1],
    'z': X_gencode_umap[:, 2],
    'perturbation': perturbations,
    'cluster': kmeans_labels_gencode_umap
})

fig = px.scatter_3d(df_plot_gencode_kmeans, 
                    x='x', y='y', z='z',
                    color='cluster', 
                    hover_data=['perturbation'],
                    title='ElasticNet Gencode - KMeans Clustering (3D UMAP)',
                    labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
                    color_continuous_scale='Viridis')
fig.update_traces(marker=dict(size=5))
fig.write_html('elasticnet_gencode_kmeans_3d_umap.html')
print("Saved: elasticnet_gencode_kmeans_3d_umap.html")

# For Gencode - GMM
df_plot_gencode_gmm = pd.DataFrame({
    'x': X_gencode_umap[:, 0],
    'y': X_gencode_umap[:, 1],
    'z': X_gencode_umap[:, 2],
    'perturbation': perturbations,
    'cluster': gmm_labels_gencode_umap
})

fig = px.scatter_3d(df_plot_gencode_gmm, 
                    x='x', y='y', z='z',
                    color='cluster', 
                    hover_data=['perturbation'],
                    title='ElasticNet Gencode - GMM Clustering (3D UMAP)',
                    labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
                    color_continuous_scale='Viridis')
fig.update_traces(marker=dict(size=5))
fig.write_html('elasticnet_gencode_gmm_3d_umap.html')
print("Saved: elasticnet_gencode_gmm_3d_umap.html")

# For TSS - KMeans
df_plot_tss_kmeans = pd.DataFrame({
    'x': X_tss_umap[:, 0],
    'y': X_tss_umap[:, 1],
    'z': X_tss_umap[:, 2],
    'perturbation': perturbations,
    'cluster': kmeans_labels_tss_umap
})

fig = px.scatter_3d(df_plot_tss_kmeans, 
                    x='x', y='y', z='z',
                    color='cluster', 
                    hover_data=['perturbation'],
                    title='ElasticNet TSS - KMeans Clustering (3D UMAP)',
                    labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
                    color_continuous_scale='Viridis')
fig.update_traces(marker=dict(size=5))
fig.write_html('elasticnet_tss_kmeans_3d_umap.html')
print("Saved: elasticnet_tss_kmeans_3d_umap.html")

# For TSS - GMM
df_plot_tss_gmm = pd.DataFrame({
    'x': X_tss_umap[:, 0],
    'y': X_tss_umap[:, 1],
    'z': X_tss_umap[:, 2],
    'perturbation': perturbations,
    'cluster': gmm_labels_tss_umap
})

fig = px.scatter_3d(df_plot_tss_gmm, 
                    x='x', y='y', z='z',
                    color='cluster', 
                    hover_data=['perturbation'],
                    title='ElasticNet TSS - GMM Clustering (3D UMAP)',
                    labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
                    color_continuous_scale='Viridis')
fig.update_traces(marker=dict(size=5))
fig.write_html('elasticnet_tss_gmm_3d_umap.html')
print("Saved: elasticnet_tss_gmm_3d_umap.html")

