#!/usr/bin/env python3
"""
Extract features from MSA files for model training.

This script extracts three types of features from MSA files:
1. QFinder features: For QFinder model (7-class substitution model classification)
2. RASFinder features: For RASFinder model (4-class RHAS model classification)
3. FFinder features: For FFinder model (2-class +F classification)

All extraction methods are optimized for extremely high efficiency.

Usage:
    # Extract features from training/validation set
    python data_preparation/feature_extraction.py \
        --alignments_dir ./simulated_alignments \
        --output_dir ./extracted_features \
        --data_type train_val \
        --num_workers 7
    
    # Extract features from test set
    python data_preparation/feature_extraction.py \
        --alignments_dir ./simulated_alignments \
        --output_dir ./extracted_features \
        --data_type test \
        --num_workers 7
"""

import argparse
import os
import sys
import re
import time
import pickle
import traceback
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import Counter
from Bio import AlignIO
from io import StringIO
import numpy as np
from scipy.stats import entropy, skew, kurtosis
from scipy.spatial.distance import jensenshannon

# Global random seed
RANDOM_SEED = 723

RASFinderFeatures = namedtuple('RASFinderFeatures', ['sitewise_feature', 'summary_feature'])

# Amino acid mapping to integer codes
AMINO_ACID_MAP = {
    'A': 1, 'R': 3, 'N': 5, 'D': 7, 'C': 9, 'Q': 11, 'E': 13, 'G': 15, 'H': 17, 'I': 19,
    'L': 21, 'K': 23, 'M': 25, 'F': 27, 'P': 29, 'S': 31, 'T': 33, 'W': 35, 'Y': 37, 'V': 39
}

AA_CODES = np.array(list(AMINO_ACID_MAP.values()), dtype=np.int16)
AA_INDICES = np.arange(20, dtype=np.int32)

# Fast look-up table for character to integer code conversion
# Any character not in AMINO_ACID_MAP is treated as a gap (0)
CHAR_CODE_LUT = np.zeros(256, dtype=np.int16)
for aa, code in AMINO_ACID_MAP.items():
    CHAR_CODE_LUT[ord(aa)] = code
    CHAR_CODE_LUT[ord(aa.lower())] = code

MAX_CODE = AA_CODES.max() + 1

CODE_TO_INDEX = np.full(MAX_CODE, -1, dtype=np.int16)
CODE_TO_INDEX[AA_CODES] = AA_INDICES

def generate_replacements():
    """
    Generate all possible amino acid replacement pair codes.
    
    Uses bitwise shift to create unique integer codes for each pair.

    This is inspired by the code in the ModelRevelator repository.
    
    Returns:
        Array of replacement codes
    """
    aa_codes = AA_CODES.astype(np.uint64)
    single_replacements = aa_codes << aa_codes
    aa1, aa2 = np.meshgrid(aa_codes, aa_codes, indexing='ij')
    pair_replacements = aa1.ravel() << aa2.ravel()
    pair_replacements = pair_replacements[aa1.ravel() != aa2.ravel()]
    
    return np.concatenate([single_replacements, pair_replacements]).astype(np.uint64)


REPLACEMENTS = generate_replacements()
N_REPLACEMENTS = len(REPLACEMENTS)

# Pre-compute sorted indices and sorted array for efficient lookup in extract_qfinder_features
REPLACEMENTS_SORTER = np.argsort(REPLACEMENTS)
REPLACEMENTS_SORTED = REPLACEMENTS[REPLACEMENTS_SORTER]

from Bio import AlignIO
import numpy as np

def msa_code_to_index(msa_matrix):
    """
    Convert code matrix (0,1,3,5,...39) → index matrix (0..19)
    gap → -1
    """
    out = np.full(msa_matrix.shape, -1, dtype=np.int16)

    mask = msa_matrix < CODE_TO_INDEX.shape[0]
    out[mask] = CODE_TO_INDEX[msa_matrix[mask]]

    return out

def aa_frequency_from_msa(msa_matrix):

    msa_idx = msa_code_to_index(msa_matrix)

    valid = msa_idx >= 0
    flat = msa_idx[valid]

    counts = np.bincount(flat, minlength=20)
    freq = counts / counts.sum()

    return freq


def convert_msa(file_path):
    """
    Convert an MSA in sequential PHYLIP format to a NumPy array.
    
    Args:
        file_path: Path to the alignment file
    
    Returns:
        2D array representing the MSA
    """
    """
    aln = AlignIO.read(file_path, "phylip-relaxed")
    arr = np.array([list(rec.seq) for rec in aln])
    msa = CHAR_CODE_LUT[arr.view(np.uint8)]
    return msa
    """
    with open(file_path, 'rt', encoding='utf8') as f:
        lines = f.readlines()
    
    seqs = [re.split(r'\s+', ln.strip())[-1] for ln in lines[1:] if ln.strip()]
    if not seqs:
        raise ValueError(f"No sequences found in file: {file_path}")
    
    n_taxa, n_sites = map(int, lines[0].strip().split())
    
    # Vectorized conversion using pre-computed look-up table
    flattened_bytes = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8)
    return CHAR_CODE_LUT[flattened_bytes].reshape(n_taxa, n_sites)


def replace_gaps(msa, rng):
    """
    Replace gaps (0) in the MSA with the most common amino acid in that site.
    
    Ties are broken randomly. Sites containing only gaps are filled randomly.
    
    Args:
        msa: The MSA array
        rng: Random number generator instance
    
    Returns:
        MSA with gaps filled
    """
    msa_filled = msa.copy()
    gap_mask = (msa_filled == 0)
    
    if not gap_mask.any():
        return msa_filled
    
    # Count amino acids per site
    counts = (msa_filled[:, :, None] == AA_CODES).sum(axis=0, dtype=np.int32)  # This is a 2D array of counts for each amino acid at each site
    
    # Identify maximums and randomly choose one for ties
    max_counts = counts.max(axis=1, keepdims=True)
    is_max = (counts == max_counts)
    masked_randoms = np.where(is_max, rng.random(counts.shape), -1.0)
    modal_idx = masked_randoms.argmax(axis=1)  # Index of the most common amino acid for each site
    
    # Handle sites containing only gaps
    all_gap_cols = (counts.sum(axis=1) == 0)  # This is a 1D array of booleans, True if the site contains only gaps
    if all_gap_cols.any():
        modal_idx[all_gap_cols] = rng.integers(0, 20, size=all_gap_cols.sum())  # Randomly choose an amino acid for each site containing only gaps
    
    # Convert modal indices back to amino acid codes
    modal_code = (modal_idx * 2 + 1).astype(np.int16)
    
    # Replace gaps with the most common amino acid for each site
    rows, cols = np.where(gap_mask)
    msa_filled[rows, cols] = modal_code[cols]
    
    return msa_filled


def calculate_seq_frequencies(msa, n_sites):
    """
    Calculate 20 amino acid frequencies for every sequence.
    
    Args:
        msa: The MSA array
        n_sites: Number of sites (columns) in the alignment
    
    Returns:
        (n_taxa, 20) array of amino acid frequencies
    """
    counts = (msa[:, :, None] == AA_CODES).sum(axis=1, dtype=np.int32)
    return (counts / n_sites).astype(np.float32)


def calculate_overall_frequencies(msa):
    """
    Calculate overall amino acid frequencies.
    
    Args:
        msa: The MSA array
    
    Returns:
        20-element vector of amino acid frequencies
    """
    non_gap_aas = msa[msa != 0]
    aa_indices = (non_gap_aas - 1) // 2
    counts = np.bincount(aa_indices, minlength=20)
    return (counts / counts.sum()).astype(np.float32)


# -----------------------------------------------------------------------------


def bimodality_coefficient(data):
    """
    Calculate the bimodality coefficient for a 1D dataset.
    
    Args:
        data: 1D array of values
    
    Returns:
        Bimodality coefficient
    """
    n = len(data)
    if n <= 3:
        return np.nan
    
    g = skew(data)
    k = kurtosis(data, fisher=False)
    correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    
    return (g ** 2 + 1) / (k + correction)


def extract_entropy_features(entropy_values, gap_ratio_site):
    """
    Calculate summary statistics from a distribution of entropy values.
    Each site has one entropy value.
    
    Args:
        entropy_values: Array of entropy values
    
    Returns:
        10-element array of 10 summary statistics for the distribution of entropy values
    """
    if entropy_values.size == 0:
        return np.zeros(10, dtype=np.float32)
    
    n_sites = len(entropy_values)
    """
    features = np.array([
        np.sum(entropy_values == 0) / n_sites,  # Proportion of invariant sites
        np.sum(entropy_values < 0.01) / n_sites,
        np.sum(entropy_values < 0.1) / n_sites,
        np.sum(entropy_values > 1.5) / n_sites,
        entropy_values.mean(),
        entropy_values.var(),
        entropy_values.std(),
        np.corrcoef(entropy_values[:-1], entropy_values[1:])[0,1]
            if len(entropy_values) > 2 and entropy_values.std()>1e-6 else 0,
        np.min(entropy_values),
        gap_ratio_site.mean(),
        np.percentile(entropy_values, 10),
        np.percentile(entropy_values, 25),
        np.percentile(entropy_values,90) - np.percentile(entropy_values,10),
        np.median(entropy_values),
        skew(entropy_values),
        kurtosis(entropy_values),
        bimodality_coefficient(entropy_values)
    ], dtype=np.float32)
    """
    features = np.array([
		np.sum(entropy_values == 0) / n_sites,  # Proportion of invariant sites
        np.sum(entropy_values < 0.01) / n_sites,
        np.sum(entropy_values < 0.1) / n_sites,
        np.percentile(entropy_values, 10),
        np.percentile(entropy_values, 25),
        gap_ratio_site.mean(),
        np.var(entropy_values),
        skew(entropy_values),
        kurtosis(entropy_values),
        bimodality_coefficient(entropy_values)
    ], dtype=np.float32)
    return np.nan_to_num(features, nan=0.0) # Replace NaN with 0.0


# -----------------------------------------------------------------------------


def extract_qfinder_features(msa_filled, n_taxa, n_sites, rng):
    """
    Extract QFinder features based on pairwise sequence comparisons.
    625 sequence pairs are randomly sampled from the MSA.
    Each pair of sequences has 440 features: 
    - 400 replacement pattern frequencies
    - 20 sequence frequencies for each sequence in the pair
    
    Args:
        msa_filled: Gap-filled MSA
        n_taxa: Number of taxa
        n_sites: Number of sites
        rng: Random number generator instance
    
    Returns:
        (625, 440) array of QFinder features
    """
    num_pairs = 625

    # Calculate sequence frequencies
    seq_freqs = calculate_seq_frequencies(msa_filled, n_sites)
    
    # Sample pairs of sequences, ensuring no self-pairs
    pairs = rng.choice(n_taxa, (num_pairs, 2), replace=True).astype(np.uint16)
    self_pair_mask = (pairs[:, 0] == pairs[:, 1])
    n_self_pairs = np.sum(self_pair_mask)
    if n_self_pairs > 0:
        new_pairs = np.empty((n_self_pairs, 2), dtype=np.uint16)
        for i in range(n_self_pairs):
            new_pairs[i] = rng.choice(n_taxa, 2, replace=False)
        pairs[self_pair_mask] = new_pairs
    
    # Find unique pairs to avoid redundant calculations
    # This is a critical step to improve efficiency
    unique_pairs, inverse_indices = np.unique(pairs, axis=0, return_inverse=True)
    n_unique = len(unique_pairs)
    
    # Calculate replacement codes for all unique pairs
    seq1s = msa_filled[unique_pairs[:, 0]].astype(np.uint64)
    seq2s = msa_filled[unique_pairs[:, 1]].astype(np.uint64)
    all_replacements = seq1s << seq2s  # (n_unique, n_sites)
    
    # Map replacement codes to indices in REPLACEMENTS array
    positions_in_sorted = np.searchsorted(REPLACEMENTS_SORTED, all_replacements)  # Indices of all_replacements in sorted REPLACEMENTS array, (n_unique, n_sites)
    mapped_indices = REPLACEMENTS_SORTER[positions_in_sorted]  # Indices of all_replacements in original REPLACEMENTS array, (n_unique, n_sites)
    
    # Calculate frequencies using vectorized grouped histogram approach
    offset = np.arange(n_unique)[:, None] * N_REPLACEMENTS  # (n_unique, 1): [[0],[400], [800], ...]
    linearized_indices = mapped_indices + offset  # Assign a unique offset to each unique sequence pair to avoid loops of np.bincount because each sequence pair now is in its own bin
    all_counts = np.bincount(linearized_indices.ravel(), minlength=n_unique * N_REPLACEMENTS)
    unique_freqs = all_counts.reshape(n_unique, N_REPLACEMENTS).astype(np.float32) / n_sites
    
    # Assemble final feature matrix
    final_replacement_freqs = unique_freqs[inverse_indices]  # (num_pairs, 400)
    features = np.hstack([final_replacement_freqs, seq_freqs[pairs[:, 0]], seq_freqs[pairs[:, 1]]])  # (num_pairs, 440)
    return features.astype(np.float32)


def extract_rasfinder_features(msa):
    """
    Extract RASFinder features based on sitewise and summary statistics.
    Each site has 23 features:
    - 20 amino acid frequencies
    - Entropy value of amino acid frequencies
    - Proportion of unique amino acids, e.g. if the site has 2 kinds of amino acids, this feature will be 2/20 = 0.1
    - Invariant site indicator, True if the site is invariant
    The summary features are 10 statistics for the distribution of entropy values.

    Args:
        msa: MSA (with gaps)
    
    Returns:
        RASFinderFeatures: A tuple containing sitewise and summary features.
            - sitewise_features: Sitewise features of shape (n_sites, 23)
            - summary_features: Summary features of shape (10,)
    """
    n_taxa, _ = msa.shape
    gap_mask = (msa == 0)
    
    # One-hot encode the MSA
    aa_indices = (msa - 1) // 2
    one_hot = (aa_indices[..., None] == AA_INDICES).astype(np.int32)  # (n_taxa, n_sites, 20)
    one_hot[gap_mask] = 0
    
    # Calculate sitewise frequencies, ignoring all-gap sites
    site_counts = one_hot.sum(axis=0)
    non_gap_counts = n_taxa - gap_mask.sum(axis=0)
    valid_sites_mask = (non_gap_counts > 0)
    gap_ratio_site = gap_mask.sum(axis=0)[valid_sites_mask] / n_taxa 
    site_frequencies = site_counts[valid_sites_mask] / non_gap_counts[valid_sites_mask, None]
    
    # Calculate statistics from valid site frequencies
    entropies = entropy(site_frequencies.T)
    unique_aa_props = (site_frequencies > 0).sum(axis=1) / 20.0
    invariant_indicator = (unique_aa_props == 0.05)
    
    sitewise_features = np.column_stack([site_frequencies, entropies, unique_aa_props, invariant_indicator]).astype(np.float32)
    summary_features = extract_entropy_features(entropies, gap_ratio_site)
    
    return RASFinderFeatures(sitewise_features, summary_features)


BACKGROUND_FREQS = {
    "LG": [
        0.079066, 0.055941, 0.041977, 0.053052, 0.012937,
        0.040767, 0.071586, 0.057337, 0.022355, 0.062157,
        0.099081, 0.064600, 0.022951, 0.042302, 0.044040,
        0.061197, 0.053287, 0.012066, 0.034155, 0.070772
    ],

    "WAG": [
        0.0866279, 0.043972, 0.0390894, 0.0570451, 0.0193078,
        0.0367281, 0.0580589, 0.0832518, 0.0244313, 0.048466,
        0.086209, 0.0620286, 0.0195027, 0.0384319, 0.0457631,
        0.0695179, 0.0610127, 0.0143859, 0.0352742, 0.0708956
    ],

    "JTT": [
        0.076747, 0.051691, 0.042645, 0.051544, 0.019803,
        0.040752, 0.061830, 0.073152, 0.022944, 0.053761,
        0.091904, 0.058676, 0.023826, 0.040126, 0.050901,
        0.068765, 0.058565, 0.014261, 0.032102, 0.066005
    ],

    "Q.plant": [
        0.074923000, 0.050500000, 0.038734000, 0.053195000, 0.011300000,
        0.037499000, 0.068513000, 0.059627000, 0.021204000, 0.058991000,
        0.102504000, 0.067306000, 0.022371000, 0.043798000, 0.037039000,
        0.084451000, 0.047850000, 0.012322000, 0.030777000, 0.077097000
    ],

    "Q.bird": [
        0.066363000, 0.054021000, 0.037784000, 0.047511000, 0.022651000,
        0.048841000, 0.071571000, 0.058368000, 0.025403000, 0.045108000,
        0.100181000, 0.061361000, 0.021069000, 0.038230000, 0.053861000,
        0.089298000, 0.053536000, 0.012313000, 0.027173000, 0.065359000
    ],

    "Q.mammal": [
        0.067997000, 0.055503000, 0.036288000, 0.046867000, 0.021435000,
        0.050281000, 0.068935000, 0.055323000, 0.026410000, 0.041953000,
        0.101191000, 0.060037000, 0.019662000, 0.036237000, 0.055146000,
        0.096864000, 0.057136000, 0.011785000, 0.024730000, 0.066223000
    ],

    "Q.pfam": [
        0.085788000, 0.057731000, 0.042028000, 0.056462000, 0.010447000,
        0.039548000, 0.067799000, 0.064861000, 0.021040000, 0.055398000,
        0.100413000, 0.059401000, 0.019898000, 0.042789000, 0.039579000,
        0.069262000, 0.055498000, 0.014430000, 0.033233000, 0.064396000
    ]
}

AA_ORDER = [
    "A","R","N","D","C","Q","E","G","H","I",
    "L","K","M","F","P","S","T","W","Y","V"
]
MODEL_NAMES = list(BACKGROUND_FREQS.keys())


def aa_freq(aln):
    c = Counter(aln.flatten())
    total = sum(c.get(a, 0) for a in AA_ORDER)
    return np.array([c.get(a, 0)/total for a in AA_ORDER]).astype(np.float32)

def read_phylip(path):
    with open(path) as f:
        content = f.read()
    content = content.replace(".", "-")
    aln = AlignIO.read(StringIO(content), "phylip-relaxed")
    return np.array([list(str(r.seq)) for r in aln])

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
NUM_AA = len(AA_LIST)

def kl_divergence(emp_freq, model_freq, eps=1e-10):
    """
    KL(empirical || model)
    emp_freq, model_freq: list hoặc np.array, length = 20
    return: float
    """
    emp = np.asarray(emp_freq, dtype=np.float64)
    mod = np.asarray(model_freq, dtype=np.float64)

    emp = np.clip(emp, eps, 1.0)
    mod = np.clip(mod, eps, 1.0)

    emp /= emp.sum()
    mod /= mod.sum()

    return float(np.sum(emp * np.log(emp / mod)))

def compute_js_divergence(emp_freq, model_freq):
    emp = np.asarray(emp_freq, dtype=np.float64)
    mod = np.asarray(model_freq, dtype=np.float64)
    eps = 1e-10
    emp = np.clip(emp, eps, 1.0)
    mod = np.clip(mod, eps, 1.0)

    emp /= emp.sum()
    mod /= mod.sum()
    return jensenshannon(emp, mod, base=2.0)

def sitewise_js_stats(msa_array):
    js_values = [[] for _ in range(7)]
    js_final_values = []
    for col in range(msa_array.shape[1]):
        col_aa = msa_array[:, col]
        aa, counts = np.unique(col_aa, return_counts=True)
        freq = np.zeros(20)

        for a, c in zip(aa, counts):
            if a in AA_TO_IDX:
                freq[AA_TO_IDX[a]] = c

        if freq.sum() == 0:
            continue

        freq /= freq.sum()
        id = 0
        for m in MODEL_NAMES:
            eq_freq = BACKGROUND_FREQS[m]
            js = compute_js_divergence(freq, eq_freq)
            js_values[id].append(js)
            id += 1
    for i in range(6):
        js_final_values.append(np.mean(js_values[i]))
        js_final_values.append(np.std(js_values[i]))
        js_final_values.append(np.max(js_values[i]))


    return js_final_values

def compute_sitewise_aa_variance(msa):
    """
    Returns:
        mean_variance, max_variance
    """
    msa_array = np.array([list(seq) for seq in msa])
    n_sites = msa_array.shape[1]

    site_variances = []

    for i in range(n_sites):
        column = msa_array[:, i]
        counts = Counter(aa for aa in column if aa in AA_TO_IDX)

        total = sum(counts.values())
        if total == 0:
            continue

        freqs = np.array([counts.get(aa, 0) / total for aa in AA_LIST])
        site_variances.append(np.var(freqs))

    if not site_variances:
        return 0.0, 0.0

    return float(np.mean(site_variances)), float(np.max(site_variances))

def count_dominant_sites(msa, threshold=0.6):
    """
    threshold: dominance cutoff (0.5–0.7 thường tốt)
    """
    msa_array = np.array([list(seq) for seq in msa])
    n_sites = msa_array.shape[1]

    dominant_sites = 0

    for i in range(n_sites):
        column = msa_array[:, i]
        counts = Counter(aa for aa in column if aa in AA_TO_IDX)

        total = sum(counts.values())
        if total == 0:
            continue

        max_freq = max(counts.values()) / total
        if max_freq >= threshold:
            dominant_sites += 1

    return dominant_sites / n_sites

def extract_ffinder_features(msa, file_path):
    """
    Extract FFinder features based on alignment statistics.
    Each alignment has 22 features:
    - 20 amino acid frequencies
    - Number of taxa
    - Number of sites
    
    Args:
        msa: MSA (with gaps)
    
    Returns:
        22-element array of FFinder features
    """
    n_taxa, n_sites = msa.shape
    overall_freqs = calculate_overall_frequencies(msa)
    f = aa_frequency_from_msa(msa)
    kl_vals = [kl_divergence(f, BACKGROUND_FREQS[m]) for m in MODEL_NAMES]
    kl_min = min(kl_vals)
    js_vals = [compute_js_divergence(f, BACKGROUND_FREQS[m]) for m in MODEL_NAMES]
    js_min = min(js_vals)
    return np.concatenate([overall_freqs, [n_taxa, n_sites], kl_vals, [k - kl_min for k in kl_vals], [j - js_min for j in js_vals], js_vals]).astype(np.float32)


# -----------------------------------------------------------------------------


def extract_features_from_file(filename, alignments_dir, gap_rng, pair_rng):
    """
    Extract all features from a single MSA file.
    
    Args:
        filename: Name of the .phy file to process
        alignments_dir: Directory containing alignment files
        gap_rng: Random number generator for gap filling
        pair_rng: Random number generator for pair sampling
    
    Returns:
        Tuple containing all extracted features and timing information
    """
    start_time = time.perf_counter()
    
    file_path = alignments_dir / filename
    msa = convert_msa(str(file_path))
    n_taxa, n_sites = msa.shape
    conversion_time = time.perf_counter()
    
    msa_filled = replace_gaps(msa, gap_rng)
    qfinder_features = extract_qfinder_features(msa_filled, n_taxa, n_sites, pair_rng)
    qfinder_time = time.perf_counter()
    
    ras_features = extract_rasfinder_features(msa)
    rasfinder_time = time.perf_counter()
    
    ffinder_features = extract_ffinder_features(msa, file_path)
    ffinder_time = time.perf_counter()
    
    timing_info = {
        "conversion": conversion_time - start_time,
        "qfinder": qfinder_time - conversion_time,
        "rasfinder": rasfinder_time - qfinder_time,
        "ffinder": ffinder_time - rasfinder_time,
        "total": ffinder_time - start_time
    }
    
    return qfinder_features, ras_features, ffinder_features, timing_info


def extract_labels(filename):
    """
    Extract labels from the filename.
    Note: The label for QFinder is the first character of the filename [0-6], which is already determined in simulation.py!
    
    Args:
        filename: Filename to extract labels from
    
    Returns:
        A tuple (label_ras, label_f) where:
            - label_ras: RHAS model label (0-3)
            - label_f: +F label (0-1)
    """
    label_ras = int("+I" in filename) + 2 * int("+G4" in filename)
    label_f = int("+F" in filename)
    return label_ras, label_f


def process_single_file(filename, alignments_dir, output_dict):
    """
    Process a single MSA file: extract features and save to disk.
    
    Args:
        filename: Name of the .phy file
        alignments_dir: Directory containing alignment files
        output_dict: Dictionary with keys 'feature_QFinder', 'feature_RASFinder', 'feature_FFinder' and values as Path objects
    
    Returns:
        tuple: (file_basename, timing) or None if error
    """
    try:
        # Create RNGs for each process
        gap_rng = np.random.default_rng(RANDOM_SEED)
        pair_rng = np.random.default_rng(RANDOM_SEED)
        
        # Extract all features
        qfinder_features, ras_features, ffinder_features, timing_info = extract_features_from_file(
            filename, alignments_dir, gap_rng, pair_rng
        )
        
        # Extract base name and labels
        file_basename = os.path.splitext(filename)[0]
        rhas_label, f_label = extract_labels(file_basename)
        
        # Save QFinder features
        qfinder_path = output_dict['feature_QFinder'] / f"{file_basename}.npy"
        np.save(qfinder_path, qfinder_features)
        
        # Save RASFinder features
        # Sitewise and summary features are saved together in a single .npz file
        ras_path = output_dict['feature_RASFinder'] / f"{rhas_label}{file_basename[1:]}.npz"
        np.savez(ras_path, sitewise=ras_features.sitewise_feature, summary=ras_features.summary_feature)
        
        # Save FFinder features
        ffinder_path = output_dict['feature_FFinder'] / f"{f_label}{file_basename[1:]}.npy"
        np.save(ffinder_path, ffinder_features)
        
        return file_basename, timing_info
    
    except Exception as e:
        print(f"Error processing {filename}: {e}", file=sys.stderr)
        traceback.print_exc()
        return None


def main():
    """
    Processes all .phy alignment files in the specified directory, extracts
    QFinder, RASFinder, and FFinder features, and saves them to output directories.
    """

    parser = argparse.ArgumentParser(
        description="Extract features from MSA files for model training"
    )
    parser.add_argument(
        "--alignments_dir",
        type=str,
        default="./simulated_alignments",
        help="Directory containing .phy alignment files (default: ./simulated_alignments)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./extracted_features",
        help="Base directory for output features (default: ./extracted_features)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 7)"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["train_val", "test"],
        default="test",
        help="Data type: 'train_val' for training/validation set, 'test' for test set (default: train_val). Automatically appends suffix to alignments_dir and output_dir."
    )
    
    args = parser.parse_args()
    
    # Automatically add suffix to paths based on data_type
    alignments_dir_base = Path(args.alignments_dir)
    output_dir_base = Path(args.output_dir)
    
    suffix = f"_{args.data_type}"
    if not alignments_dir_base.name.endswith(suffix):
        alignments_dir = alignments_dir_base.parent / f"{alignments_dir_base.name}{suffix}"
    else:
        alignments_dir = alignments_dir_base
    
    if not output_dir_base.name.endswith(suffix):
        output_dir = output_dir_base.parent / f"{output_dir_base.name}{suffix}"
    else:
        output_dir = output_dir_base
    
    if not alignments_dir.exists():
        raise FileNotFoundError(f"Alignments directory not found: {alignments_dir}")
    
    # Create output directory structure
    output_dict = {
        'feature_QFinder': output_dir / "QFinder",
        'feature_RASFinder': output_dir / "RASFinder",
        'feature_FFinder': output_dir / "FFinder"
    }
    
    for dir_path in output_dict.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .phy files in the alignments directory
    files_to_process = [f for f in os.listdir(alignments_dir) if f.endswith('.phy') or f.endswith('.phyml')]
    
    # Process files in parallel
    start_total = time.perf_counter()
    timing_results = {}  # Dictionary to store timing information for each file
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        tasks = [
            executor.submit(
                process_single_file,
                filename,
                alignments_dir,
                output_dict
            )
            for filename in files_to_process
        ]
        
        for i, task in enumerate(tasks):
            result = task.result()
            if result:
                file_basename, timing_info = result
                timing_results[file_basename] = timing_info
    
    end_total = time.perf_counter()
    total_time = end_total - start_total
    
    # Save timing results to a pickle file
    timing_path = output_dir / "feature_extraction_time.pkl"
    with open(timing_path, 'wb') as f:
        pickle.dump(timing_results, f)
    
    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == '__main__':
    main()
