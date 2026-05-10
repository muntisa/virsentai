#!/usr/bin/env python3
"""
Dataset Undersampling

Balances a viral genome dataset using MinHash + clustering undersampling.
Creates a diverse balanced dataset from the majority class.

Usage:
    python DS-Undersampling.py

Requirements:
    - pandas, numpy, scikit-learn, datasketch, tqdm
    - Input: ds/ds_160k.tsv (from DS-Filter.py)

Process:
    1. Load ds/ds_160k.tsv and filter labels 0 and 1 (remove -1)
    2. Generate MinHash signatures for label 0 sequences (parallel CPU)
    3. Cluster using MiniBatchKMeans into target groups
    4. Select one representative per cluster (diverse subset)
    5. Merge with label 1 and save balanced dataset

Output:
    - ds/balanced_diverse_genomes.tsv: Balanced dataset with maximum diversity

Notes:
    - Adaptive stride: every k-mer for short sequences (<5kb)
    - Stride of 15 for long genomes (up to 160kb) for speed
    - k=21, num_perm=128 for MinHash signatures
"""

import pandas as pd
import numpy as np
import time
import multiprocessing as mp
from datasketch import MinHash
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from config import *

def compute_sketch(sequence, ksize=21, num_perm=128):
    """
    Worker function: Hashes viral genomes.
    Adaptive stride: Use every k-mer for short sequences (<5kb), 
    stride of 15 for long genomes (up to 160kb) for maximum speed.
    """
    m = MinHash(num_perm=num_perm)
    seq_len = len(sequence)
    # Sampling every 15th k-mer for long sequences keeps it fast 
    # while maintaining a strong diversity signature.
    stride = 1 if seq_len < 5000 else 15
    
    for i in range(0, seq_len - ksize + 1, stride):
        kmer = sequence[i:i+ksize].encode('utf8')
        m.update(kmer)
    return m.hashvalues

def main(input_path, output_path):
    total_start = time.perf_counter()
    
    # --- [1/5] Data Loading ---
    print("\n[1/5] Loading and filtering dataset...")
    s1_start = time.perf_counter()
    df = pd.read_csv(input_path, sep='\t')
    
    # Filter: Keep only labels 0 and 1 (ignore -1)
    df = df[df['label'].isin([0, 1])].copy()
    
    df_1 = df[df['label'] == 1].reset_index(drop=True)
    df_0 = df[df['label'] == 0].reset_index(drop=True)
    target_n = len(df_1)
    s1_end = time.perf_counter()
    print(f"Done. Target size: {target_n} | Pool (Label 0): {len(df_0)}")

    # --- [2/5] Feature Extraction (CPU Parallel) ---
    print(f"\n[2/5] Generating MinHash signatures (Parallel CPU)...")
    s2_start = time.perf_counter()
    sequences = df_0['sequence'].tolist()
    
    # On Windows, ProcessPoolExecutor is the safest way to use all cores
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        features = list(tqdm(
            executor.map(compute_sketch, sequences, chunksize=100), 
            total=len(sequences), 
            unit="seq",
            desc="Hashing Progress"
        ))
    
    X = np.array(features, dtype=np.float32)
    s2_end = time.perf_counter()

    # --- [3/5] Clustering (Optimized MiniBatchKMeans) ---
    print(f"\n[3/5] Clustering into {target_n} diverse groups...")
    s3_start = time.perf_counter()
    
    # MiniBatchKMeans is much faster than standard KMeans for 50k rows
    kmeans = MiniBatchKMeans(
        n_clusters=target_n,
        batch_size=2048,
        random_state=42,
        n_init='auto'
    )
    kmeans.fit(X)
    s3_end = time.perf_counter()

    # --- [4/5] Selection ---
    print("\n[4/5] Selecting diverse representatives...")
    s4_start = time.perf_counter()
    
    # Find the index of the sequence closest to each cluster centroid
    closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    unique_indices = np.unique(closest_indices)
    
    # If duplicates occurred, fill with remaining rows
    if len(unique_indices) < target_n:
        needed = target_n - len(unique_indices)
        remaining = np.setdiff1d(np.arange(len(df_0)), unique_indices)
        extra = np.random.choice(remaining, size=needed, replace=False)
        final_indices = np.concatenate([unique_indices, extra])
    else:
        final_indices = unique_indices

    df_0_sampled = df_0.iloc[final_indices].copy()
    s4_end = time.perf_counter()

    # --- [5/5] Merging and Saving ---
    print("\n[5/5] Saving final balanced dataset...")
    s5_start = time.perf_counter()
    final_df = pd.concat([df_1, df_0_sampled]).sample(frac=1).reset_index(drop=True)
    final_df.to_csv(output_path, sep='\t', index=False)
    s5_end = time.perf_counter()

    total_end = time.perf_counter()
    
    # --- Performance Report ---
    print("\n" + "="*40)
    print("PERFORMANCE REPORT")
    print("="*40)
    print(f"1. Loading/Filtering: {(s1_end - s1_start)/60:8.2f} min")
    print(f"2. Parallel Hashing:  {(s2_end - s2_start)/60:8.2f} min")
    print(f"3. Fast Clustering:   {(s3_end - s3_start)/60:8.2f} min")
    print(f"4. Selection:         {(s4_end - s4_start)/60:8.2f} min")
    print(f"5. Saving File:       {(s5_end - s5_start)/60:8.2f} min")
    print("-" * 40)
    print(f"TOTAL RUNTIME:       {(total_end - total_start)/60:8.2f} min")
    print("="*40)

# IMPORTANT: On Windows, the main code must be inside this block
if __name__ == "__main__":
    main(DS_160K_FILE, DS_BALANCED_FILE)