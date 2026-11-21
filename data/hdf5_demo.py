#!/usr/bin/env python
#
# (c) 2017 by Joffrey Fitz (joffrey.fitz@tuebingen.mpg.de),
# Max Planck Institute for Developmental Biology,
# Tuebingen, Germany
#

import h5py, numpy


f = h5py.File('1001_SNP_MATRIX/imputed_snps_binary.hdf5','r')

# Coordinates for At1g01070.1
start_pos = 38752
end_pos = 40944

# Get all SNP positions for all chromosomes (len=10709949)
positions = f['positions'][:]

# Array of tupels with start/stop indices for each chromosome
chr_regions = f['positions'].attrs['chr_regions']
indices_for_chr1 = chr_regions[0]

# Subset positions for SNPs on Chr1
positions_on_chr1 = positions[indices_for_chr1[0]:indices_for_chr1[1]]

# Index for start_pos/end_pos
idx_start_pos = numpy.where( numpy.logical_and(positions_on_chr1>=start_pos, positions_on_chr1<=end_pos) )[0][0]
idx_end_pos = numpy.where( numpy.logical_and(positions_on_chr1>=start_pos, positions_on_chr1<=end_pos) )[0][-1]

# Get SNPs
snps_in_region = f['snps'][idx_start_pos:idx_end_pos]

# Find index of a specific position 
ix = numpy.where(positions_on_chr1 == 39751)[0][0]

# Get the corresponding SNP for that position
snp = f['snps'][ix]

# Get index of specific accessions
accs = f['accessions'][:]
ix_of_acc = numpy.where(accs == '139')[0][0]

# Get SNP for pos and acc
snp_for_specific_pos_and_acc = snp[ix_of_acc]

print "SNP %s" % snp_for_specific_pos_and_acc
