#!/usr/bin/env python
# Converted from enformer_gencode.v49.pc_trascripts.ipynb

!nvidia-smi

import os

# Configure environment
os.environ['WANDB_API_KEY'] = '' # replace with your key
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # Set available GPU device

# Login to Weights & Biases for experiment tracking
!wandb login

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Grelu library for genomic deep learning
from grelu.model.models import EnformerPretrainedModel
from grelu.io.fasta import read_fasta
from grelu.sequence.format import convert_input_type

import warnings
warnings.filterwarnings('ignore')

# Configuration
SAVE = True
SEED = 1182024
TARGET_LENGTH = 8192  # Fixed sequence length for padding
BATCH_SIZE = 8

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def pad_sequence(seq, target_len):
    """
    Pad or truncate sequence to target length.
    
    Args:
        seq: DNA sequence string
        target_len: Desired length
    
    Returns:
        Padded/truncated sequence
    """
    if len(seq) > target_len:
        return seq[:target_len]
    else:
        return seq + "N" * (target_len - len(seq))

# Load sequences from FASTA file
input_fasta = "../../genomic_sequences/gencode.v49.pc_transcripts.gene_names.fa"

sequences = read_fasta(input_fasta)
padded_sequences = [pad_sequence(seq, TARGET_LENGTH) for seq in sequences]

print(f"Processed {len(padded_sequences)} sequences padded to length {TARGET_LENGTH} bp.")

# Convert sequences to one-hot encoding format
ohes = convert_input_type(
    inputs=padded_sequences,
    output_type="one_hot",
    genome="hg38",
    add_batch_axis=True,
)

print(f"One-hot encoded shape: {ohes[0].shape}")

# Initialize Enformer pre-trained model
feature_extractor = EnformerPretrainedModel(
    n_tasks=32,
    device=device
)
feature_extractor = feature_extractor.to(device)

# Calculate total parameters
total_params = sum(p.numel() for p in feature_extractor.parameters())
print(f"Model loaded successfully")
print(f"Total parameters: {total_params:,}")

# Create DataLoader for batch processing
test_loader = DataLoader(
    dataset=ohes,
    batch_size=BATCH_SIZE,
    pin_memory=True
)

# Extract features from all sequences
embeddings = []
feature_extractor.eval()
with torch.no_grad():
    for batch_idx, data in tqdm(enumerate(test_loader), total=len(test_loader), desc="Extracting features"):
        # Move data to device
        curr_data = torch.tensor(data, dtype=torch.float32).to(device)
        
        # Get embeddings from model
        batch_embeddings = feature_extractor(curr_data).squeeze().detach().cpu().numpy()
        embeddings.extend(batch_embeddings)

# Convert to numpy array
embedding = np.array(embeddings, dtype=np.float32)
embedding = np.stack(embedding, axis=0)

print(f"\nEmbedding shape: {embedding.shape}")
print(f"Total sequences processed: {len(embedding)}")

if SAVE:
    output_file = "../../embeddings/embeddings_enformer_gencode.v49.pc_transcripts.npy"
    np.save(output_file, embedding)
    print(f"Embeddings saved to: {output_file}")
else:
    print("SAVE is set to False - embeddings not saved")

import os
import numpy as np

def extract_gene_names(fasta_file):
    """
    Extract gene names from FASTA file headers.

    Args:
        fasta_file: Path to FASTA file

    Returns:
        List of gene names in the order they appear in the file
    """
    gene_names = []

    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Extract gene name (remove '>' and strip whitespace)
                gene_name = line[1:].strip()
                gene_names.append(gene_name)

    return gene_names

# Define paths
fasta_file = "../../../genomic_sequences/gencode.v49.pc_transcripts.gene_names.fa"
embeddings_file = "../../../embeddings/embeddings_enformer_gencode.v49.pc_transcripts.npy"
metadata_dir = "../../1_embeddings"
output_file = os.path.join(metadata_dir, "gencode.v49.pc_transcripts_gene_names.txt")


# Create metadata directory if it doesn't exist
os.makedirs(metadata_dir, exist_ok=True)
print(f"Created metadata directory: {metadata_dir}")
# Extract gene names from FASTA file
print(f"Extracting gene names from {fasta_file}...")
gene_names = extract_gene_names(fasta_file)
print(f"Extracted {len(gene_names)} gene names")
# Load embeddings to verify alignment (allow pickle for this file)
print(f"\nLoading embeddings from {embeddings_file}...")
embeddings = np.load(embeddings_file, allow_pickle=True)
print(f"Embeddings shape: {embeddings.shape}")
# Verify that the number of genes matches the number of embeddings
if len(gene_names) != embeddings.shape[0]:
    print(f"\nWARNING: Mismatch detected!")
    print(f"  Gene names: {len(gene_names)}")
    print(f"  Embeddings: {embeddings.shape[0]}")
else:
    print(f"\n✓ Verified: {len(gene_names)} genes match {embeddings.shape[0]} embeddings")
# Save gene names to metadata file
print(f"\nSaving gene names to {output_file}...")
with open(output_file, 'w') as f:
    for idx, gene_name in enumerate(gene_names):
        f.write(f"{idx}\t{gene_name}\n")
print(f"✓ Metadata saved successfully!")
print(f"\nMetadata file format:")
print(f"  Column 1: Index (0-based, corresponds to embedding row)")
print(f"  Column 2: Gene name")
# Show first few lines as preview
print(f"\nFirst 5 entries:")
for i in range(min(5, len(gene_names))):
    print(f"  {i}\t{gene_names[i]}")
# Save as numpy array as well for easier loading
npy_output = os.path.join(metadata_dir, "gencode.v49.pc_transcripts_gene_names.npy")
np.save(npy_output, np.array(gene_names))
print(f"\n✓ Also saved as numpy array: {npy_output}")
