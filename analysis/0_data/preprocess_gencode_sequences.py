#!/usr/bin/env python

import re
from statistics import mean, median

tss_fa = "../../genomic_sequences/TSS.fa"
gencode_fa = "../../genomic_sequences/gencode.v49.pc_transcripts.fa"
output_fa = "../../genomic_sequences/gencode.v49.pc_transcripts.gene_names.fa"

genes_of_interest = set()
with open(tss_fa) as f:
    for line in f:
        if line.startswith(">"):
            genes_of_interest.add(line.strip().lstrip(">"))

gene_pattern = re.compile(r"\|([A-Za-z0-9\-]+)\|([A-Za-z0-9]+)\|")

written_genes = set()
with open(gencode_fa) as fin, open(output_fa, "w") as fout:
    keep = False
    for line in fin:
        if line.startswith(">"):
            m = gene_pattern.search(line)
            if m:
                gene_name = m.group(2)
                if gene_name in genes_of_interest and gene_name not in written_genes:
                    fout.write(f">{gene_name}\n")
                    keep = True
                    written_genes.add(gene_name)
                else:
                    keep = False
            else:
                keep = False
        elif keep:
            fout.write(line)

print(f"Loaded {len(genes_of_interest)} genes of interest.")
print(f"Wrote {len(written_genes)} unique genes to {output_fa}.")

gene_lengths = {}
gene_name = None
seq_chunks = []

with open(output_fa) as f:
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if gene_name and seq_chunks:
                seq = "".join(seq_chunks)
                gene_lengths[gene_name] = len(seq)
            gene_name = line.lstrip(">")
            seq_chunks = []
        else:
            seq_chunks.append(line)

if gene_name and seq_chunks:
    seq = "".join(seq_chunks)
    gene_lengths[gene_name] = len(seq)

lengths = list(gene_lengths.values())
shortest_gene = [k for k, v in gene_lengths.items() if v == min(lengths)][0]
longest_gene = [k for k, v in gene_lengths.items() if v == max(lengths)][0]

print(f"\nTotal genes: {len(gene_lengths)}")
print(f"Shortest: {min(lengths)} bp ({shortest_gene})")
print(f"Longest: {max(lengths)} bp ({longest_gene})")
print(f"Mean: {mean(lengths):.1f} bp")
print(f"Median: {median(lengths):.1f} bp")
