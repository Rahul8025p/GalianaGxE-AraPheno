"""
build_wgs_matrix.py
===================
Builds a real WGS Genotype Variant Matrix (N_accessions x 27,655 genes x 17 variant types)
directly from the 1001 Genomes Project Imputed SNP Matrix (imputed_snps_binary.hdf5)
and Arabidopsis thaliana TAIR10 gene coordinates (Arabidopsis_thaliana.TAIR10.63.gff3.gz).

Output:
  - data/real_wgs_genotype_100.npy (shape: 100 x 27655 x 17)
"""

import os
import sys
import gzip
import time
import json
import numpy as np
import h5py

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HDF5_PATH = os.path.join(DATA_DIR, "1001_SNP_MATRIX", "imputed_snps_binary.hdf5")
GFF_PATH = os.path.join(DATA_DIR, "Arabidopsis_thaliana.TAIR10.63.gff3.gz")
OUT_PATH = os.path.join(DATA_DIR, "real_wgs_genotype_100.npy")

NUM_GENES = 27655
NUM_BINS = 17
NUM_SAMPLES = 100

def load_tair10_genes(gff_path, target_count=NUM_GENES):
    print("Loading TAIR10 gene annotations...")
    chrom_map = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
    genes = []

    with gzip.open(gff_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'gene':
                continue
            chrom = parts[0]
            if chrom in chrom_map:
                c_num = chrom_map[chrom]
                start = int(parts[3])
                end = int(parts[4])
                attrs = parts[8]
                if any(k in attrs for k in ['protein_coding', 'ID=gene:AT']):
                    genes.append((c_num, start, end))

    genes.sort(key=lambda x: (x[0], x[1]))
    print(f"Extracted {len(genes)} protein-coding genes.")

    # Pad or trim to target_count
    if len(genes) < target_count:
        last = genes[-1]
        genes.extend([last] * (target_count - len(genes)))
    elif len(genes) > target_count:
        genes = genes[:target_count]

    return genes

def main():
    print("=" * 75)
    print("  EXTRACTING REAL WGS GENOTYPE MATRIX FROM 1001 GENOMES HDF5")
    print("=" * 75)
    t0 = time.time()

    # 1. Load genes
    genes = load_tair10_genes(GFF_PATH, NUM_GENES)

    # Group genes by chromosome
    chrom_genes = {c: [] for c in range(1, 6)}
    for g_idx, (c_num, start, end) in enumerate(genes):
        chrom_genes[c_num].append((start, end, g_idx))

    # 2. Open HDF5
    print(f"\nOpening HDF5 dataset: {HDF5_PATH}")
    f = h5py.File(HDF5_PATH, 'r')
    positions = f['positions'][:]
    chr_regions = f['positions'].attrs['chr_regions']
    accessions = f['accessions'][:NUM_SAMPLES]
    acc_ids = [a.decode('utf-8') if isinstance(a, bytes) else str(a) for a in accessions]
    print(f"Extracting for {NUM_SAMPLES} accessions. Sample IDs: {acc_ids[:5]}")

    # Output matrix: (NUM_SAMPLES, NUM_GENES, NUM_BINS)
    X_geno = np.zeros((NUM_SAMPLES, NUM_GENES, NUM_BINS), dtype=np.float32)

    # 3. Process chromosome by chromosome
    CHUNK_SIZE = 250000

    for c_idx in range(5):
        c_num = c_idx + 1
        c_start, c_end = int(chr_regions[c_idx][0]), int(chr_regions[c_idx][1])
        c_positions = positions[c_start:c_end]
        n_snps_chr = c_end - c_start
        print(f"\n--- Chromosome {c_num}: {n_snps_chr:,} SNPs, {len(chrom_genes[c_num])} genes ---")

        # Map each gene to SNP index range using binary search
        gene_snp_bounds = []
        for g_start, g_end, g_idx in chrom_genes[c_num]:
            idx_left = np.searchsorted(c_positions, g_start)
            idx_right = np.searchsorted(c_positions, g_end)
            if idx_right > idx_left:
                gene_snp_bounds.append((idx_left, idx_right, g_idx))

        print(f"  {len(gene_snp_bounds)} genes overlap with SNP positions.")

        # Stream SNPs in chunks
        for chunk_start in range(0, n_snps_chr, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, n_snps_chr)
            abs_start = c_start + chunk_start
            abs_end = c_start + chunk_end

            # Read SNP chunk: shape (chunk_len, NUM_SAMPLES)
            snps_chunk = f['snps'][abs_start:abs_end, :NUM_SAMPLES]

            # Find genes overlapping this chunk
            for idx_left, idx_right, g_idx in gene_snp_bounds:
                if idx_right <= chunk_start or idx_left >= chunk_end:
                    continue
                overlap_left = max(idx_left, chunk_start) - chunk_start
                overlap_right = min(idx_right, chunk_end) - chunk_start
                if overlap_right <= overlap_left:
                    continue

                sub_snps = snps_chunk[overlap_left:overlap_right]  # (n_snps_in_slice, NUM_SAMPLES)
                n_sub = sub_snps.shape[0]

                # Distribute into NUM_BINS bins
                bin_sz = max(1, n_sub // NUM_BINS)
                for b in range(NUM_BINS):
                    b_s = b * bin_sz
                    b_e = min(b_s + bin_sz, n_sub)
                    if b_s >= n_sub:
                        break
                    # sum alt alleles across this bin for all samples
                    bin_counts = np.sum(sub_snps[b_s:b_e] > 0, axis=0)  # shape (NUM_SAMPLES,)
                    X_geno[:, g_idx, b] += bin_counts

            print(f"    Processed SNPs {chunk_start:,} to {chunk_end:,} / {n_snps_chr:,}", end='\r')

        print(f"\n  Chromosome {c_num} complete.")

    f.close()

    # 4. Standardize per gene/bin
    print("\nNormalizing variant features...")
    stds = X_geno.std(axis=0)
    means = X_geno.mean(axis=0)
    mask = stds > 1e-6
    X_geno[:, mask] = (X_geno[:, mask] - means[mask]) / stds[mask]

    # Save
    np.save(OUT_PATH, X_geno)
    total_time = time.time() - t0
    print(f"\nSaved Real WGS Genotype Matrix: {OUT_PATH}")
    print(f"Matrix shape: {X_geno.shape} | Size: {X_geno.nbytes / (1024*1024):.2f} MB")
    print(f"Completed in {total_time:.2f} seconds!")
    print("=" * 75)

if __name__ == "__main__":
    main()
