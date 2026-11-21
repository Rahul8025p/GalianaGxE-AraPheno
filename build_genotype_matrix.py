"""
build_genotype_matrix.py
=========================
Converts the 1001 Genomes imputed SNP HDF5 matrix into a gene-level
variant count matrix suitable for GalianaGxE:
  Output shape: (N_accessions x 27655_genes x 17_variant_bins)

Since the HDF5 provides raw binary SNPs (0/1) per position, we:
1. Load the Arabidopsis TAIR10 gene coordinates (chromosome + start/end)
2. For each gene, count SNPs per accession in 17 position-based bins
   (proxy for the 17 variant consequence types Galiana originally used)
3. Align accessions with those in AraPheno (real_arapheno_Y.npy)
4. Save as data/real_genotype_X.npy  (float32, N x 27655 x 17)
         and data/genotype_accession_ids.json
"""

import os, sys, json, gzip, io
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler

HDF5_PATH   = "data/1001_SNP_MATRIX/imputed_snps_binary.hdf5"
OUT_GENO    = "data/real_genotype_X.npy"
OUT_ACC_IDS = "data/genotype_accession_ids.json"
PHENO_NPY   = "data/real_arapheno_Y.npy"
PHENO_JSON  = "data/real_phenotype_names.json"

# TAIR10 gene coordinates — 27,655 protein-coding genes
# We'll download a minimal gene BED from Ensembl Plants
GENE_BED_URL = "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/bed/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.57.bed.gz"
GENE_BED_PATH = "data/TAIR10_genes.bed.gz"

NUM_GENES = 27655
NUM_BINS  = 17   # positional bins within each gene (proxy for consequence types)

def download_gene_bed():
    if os.path.exists(GENE_BED_PATH):
        print(f"  Gene BED already exists: {GENE_BED_PATH}")
        return
    import urllib.request
    print(f"  Downloading TAIR10 gene coordinates...")
    urllib.request.urlretrieve(GENE_BED_URL, GENE_BED_PATH)
    print(f"  Saved to {GENE_BED_PATH}")

def parse_gene_bed(bed_path, num_genes=NUM_GENES):
    """Parse BED file → list of (chrom_int, start, end) for protein-coding genes."""
    genes = []
    chrom_map = {'1':1,'2':2,'3':3,'4':4,'5':5,
                 'Chr1':1,'Chr2':2,'Chr3':3,'Chr4':4,'Chr5':5}
    opener = gzip.open if bed_path.endswith('.gz') else open
    with opener(bed_path, 'rt') as f:
        for line in f:
            if line.startswith('#') or line.startswith('track'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            chrom_raw = parts[0].replace('Chr','').replace('chr','')
            if chrom_raw not in chrom_map and ('Chr'+chrom_raw) not in chrom_map:
                continue
            # Map to int chromosome
            chrom_key = chrom_raw if chrom_raw in chrom_map else 'Chr'+chrom_raw
            chrom_int = chrom_map.get(chrom_key, 0)
            if chrom_int == 0:
                continue
            try:
                start = int(parts[1])
                end   = int(parts[2])
            except ValueError:
                continue
            # Only keep protein_coding if column 7 available
            if len(parts) > 7 and 'protein_coding' not in parts[7]:
                continue
            genes.append((chrom_int, start, end))
    # Deduplicate and sort
    genes = sorted(set(genes))
    print(f"  Parsed {len(genes)} protein-coding gene loci from BED")
    # Pad or trim to NUM_GENES
    while len(genes) < num_genes:
        genes.append(genes[-1])
    return genes[:num_genes]

def build_matrix():
    print("=" * 70)
    print("  BUILDING REAL GENOTYPE MATRIX FROM 1001 GENOMES HDF5")
    print("=" * 70)

    # 1. Load HDF5
    print(f"\n[1/5] Opening HDF5: {HDF5_PATH}")
    f = h5py.File(HDF5_PATH, 'r')
    all_accs  = f['accessions'][:]           # shape: (N_accs,) — accession IDs as bytes/str
    positions = f['positions'][:]            # shape: (total_snps,) — genome-wide positions
    chr_regions = f['positions'].attrs['chr_regions']  # array of (start_idx, end_idx) per chrom

    all_accs_str = [a.decode('utf-8') if isinstance(a, bytes) else str(a) for a in all_accs]
    print(f"  HDF5 accessions  : {len(all_accs_str)}")
    print(f"  Total SNP sites  : {len(positions):,}")
    print(f"  Chromosomes      : {len(chr_regions)}")
    print(f"  Sample acc IDs   : {all_accs_str[:6]}")

    # 2. Load AraPheno accession list (from phenotype data)
    print(f"\n[2/5] Loading AraPheno accession IDs from phenotype file...")
    # The AraPheno npy rows correspond to accession order from the API
    # We load the saved IDs if available, else use all HDF5 accs
    arapheno_acc_path = "data/arapheno_accession_ids.json"
    if os.path.exists(arapheno_acc_path):
        with open(arapheno_acc_path) as fj:
            arapheno_acc_ids = [str(a) for a in json.load(fj)]
        print(f"  Loaded {len(arapheno_acc_ids)} AraPheno accession IDs")
    else:
        # Use first N HDF5 accessions aligned with phenotype rows
        pheno_mat = np.load(PHENO_NPY)
        N = pheno_mat.shape[0]
        arapheno_acc_ids = all_accs_str[:N]
        print(f"  No ID file found — using first {N} HDF5 accessions as proxy")

    # Find overlap between HDF5 and AraPheno accessions
    hdf5_acc_set = {a: i for i, a in enumerate(all_accs_str)}
    matched = [(i_pheno, hdf5_acc_set[acc]) for i_pheno, acc in enumerate(arapheno_acc_ids)
               if acc in hdf5_acc_set]
    print(f"  Matched accessions (HDF5 ∩ AraPheno): {len(matched)}")

    if len(matched) < 10:
        print("  WARNING: Very few matches! Using first N HDF5 accessions as proxy.")
        N = min(len(arapheno_acc_ids), len(all_accs_str))
        matched = [(i, i) for i in range(N)]

    pheno_indices = [m[0] for m in matched]
    hdf5_indices  = [m[1] for m in matched]
    N_matched = len(matched)
    print(f"  Building genotype matrix for {N_matched} accessions")

    # 3. Load gene coordinates
    print(f"\n[3/5] Loading TAIR10 gene coordinates...")
    download_gene_bed()
    genes = parse_gene_bed(GENE_BED_PATH, NUM_GENES)

    # Build per-chromosome gene lookup
    chrom_genes = {1:[], 2:[], 3:[], 4:[], 5:[]}
    for g_idx, (chrom, start, end) in enumerate(genes):
        if chrom in chrom_genes:
            chrom_genes[chrom].append((start, end, g_idx))

    # 4. Build genotype matrix chromosome by chromosome
    print(f"\n[4/5] Building (N={N_matched} x {NUM_GENES} genes x {NUM_BINS} bins) matrix...")
    print(f"  This may take a few minutes (reading SNP data per chromosome)...")

    X_geno = np.zeros((N_matched, NUM_GENES, NUM_BINS), dtype=np.float32)
    hdf5_indices_arr = np.array(hdf5_indices)

    for chrom_idx in range(5):   # chromosomes 1-5
        chrom_num = chrom_idx + 1
        cr = chr_regions[chrom_idx]
        snp_start, snp_end = int(cr[0]), int(cr[1])
        chrom_positions = positions[snp_start:snp_end]
        n_snps_chrom = snp_end - snp_start
        print(f"  Chromosome {chrom_num}: {n_snps_chrom:,} SNPs, {len(chrom_genes[chrom_num])} genes")

        # Read SNPs for matched accessions on this chromosome
        # snps shape: (n_snps_chrom, total_accessions) — read in chunks
        CHUNK = 50000
        for gene_start_bp, gene_end_bp, g_idx in chrom_genes[chrom_num]:
            # Find SNP indices within this gene's coordinates
            in_gene = np.where(
                (chrom_positions >= gene_start_bp) & (chrom_positions <= gene_end_bp)
            )[0]
            if len(in_gene) == 0:
                continue
            # Map to global SNP indices
            global_snp_idx = snp_start + in_gene
            # Read SNP values for matched accessions
            # h5py fancy indexing: snps[snp_idx_list, acc_idx_list]
            snp_vals = f['snps'][global_snp_idx[0]:global_snp_idx[-1]+1, :][:, hdf5_indices_arr]
            # snp_vals shape: (n_snps_in_gene, N_matched)
            # Distribute into NUM_BINS positional bins
            n_snps = snp_vals.shape[0]
            bin_size = max(1, n_snps // NUM_BINS)
            for b in range(NUM_BINS):
                b_start = b * bin_size
                b_end   = min(b_start + bin_size, n_snps)
                if b_start >= n_snps:
                    break
                # Count alt alleles per accession in this bin
                X_geno[:, g_idx, b] = snp_vals[b_start:b_end, :].sum(axis=0).T

        print(f"  Chr {chrom_num} done ✓")

    f.close()

    # Normalize per gene
    print("\n[5/5] Normalizing and saving...")
    # Scale each gene's variant counts to zero-mean unit-var across accessions
    for g in range(NUM_GENES):
        for b in range(NUM_BINS):
            col = X_geno[:, g, b]
            std = col.std()
            if std > 1e-8:
                X_geno[:, g, b] = (col - col.mean()) / std

    np.save(OUT_GENO, X_geno)
    with open(OUT_ACC_IDS, 'w') as fj:
        json.dump([arapheno_acc_ids[i] for i in pheno_indices], fj)

    print(f"  Genotype matrix saved → {OUT_GENO}  shape: {X_geno.shape}")
    print(f"  Accession IDs saved   → {OUT_ACC_IDS}")
    print(f"  Memory: {X_geno.nbytes / 1e6:.1f} MB")
    print("=" * 70)
    print("  GENOTYPE MATRIX BUILD COMPLETE")
    print("=" * 70)
    return X_geno, pheno_indices

if __name__ == "__main__":
    build_matrix()
