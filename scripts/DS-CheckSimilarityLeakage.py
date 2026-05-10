#!/usr/bin/env python3
"""
Similarity Leakage Check

Verifies no similarity leakage exists between train and validation sets.
Computes pairwise Jaccard similarity between random samples.

Usage:
    python DS-CheckSimilarityLeakage.py

Input:
    - ds/train_split.tsv (from DS-Split.py)
    - ds/val_split.tsv (from DS-Split.py)

Output:
    - ds/similarity_leakage_report.csv

Methodology:
    1. Sample SAMPLE_SIZE sequences from train and val
    2. Encode k-mers as integers (2 bits/base)
    3. Compute Jaccard similarity for all pairs
    4. Analyze distribution and flag high-similarity pairs

Parameters (must match DS-Split.py):
    - K = 9 (k-mer size)
    - THRESHOLD = 0.60
    - SAMPLE_SIZE = 1000 (sequences per set)
    - MAX_SEQ_LENGTH = 80000 (skip longer for speed)
"""

import pandas as pd
import numpy as np
import random
import csv
import time

csv.field_size_limit(2_000_000)

from config import *

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
K         = 9     # k-mer size           — MUST match DS-Split.py
THRESHOLD = 0.60  # leakage threshold    — MUST match DS-Split.py

SAMPLE_SIZE    = 1000     # sequences sampled from each set (500×500 = 250k comparisons)
MAX_SEQ_LENGTH = 80_000  # skip longer sequences for speed

RANDOM_SEED = 2025

# ── BASE ENCODING LOOKUP ──────────────────────────────────────────────────────
# Identical to DS-Split.py: A=0, C=1, G=2, T=3, N/other=0
_BASE_MAP = np.zeros(256, dtype=np.uint32)
for _char, _val in zip("AaCcGgTtNn", [0, 0, 1, 1, 2, 2, 3, 3, 0, 0]):
    _BASE_MAP[ord(_char)] = _val


# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def sequence_to_kmer_ints(seq: str, k: int = K) -> np.ndarray:
    """Convert a DNA sequence to a sorted array of unique integer k-mers.

    Identical implementation to DS-Split.py — ensures both scripts use
    exactly the same k-mer representation.

    Args:
        seq: DNA string.
        k:   k-mer length.

    Returns:
        uint32 numpy array of unique integer k-mers.
    """
    arr = _BASE_MAP[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    n_kmers = len(arr) - k + 1
    if n_kmers <= 0:
        return np.array([], dtype=np.uint32)

    powers  = (4 ** np.arange(k - 1, -1, -1)).astype(np.uint32)
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=(n_kmers, k), strides=strides)
    return np.unique((windows @ powers).astype(np.uint32))


def jaccard_from_int_sets(a: np.ndarray, b: np.ndarray) -> float:
    """Exact Jaccard similarity between two sorted integer k-mer arrays.

    Uses np.intersect1d / np.union1d — faster than converting to Python sets.

    Args:
        a, b: sorted uint32 arrays of unique k-mer integers.

    Returns:
        Jaccard similarity in [0, 1].
    """
    inter = len(np.intersect1d(a, b, assume_unique=True))
    union = len(a) + len(b) - inter
    return inter / union if union > 0 else 0.0


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    random.seed(RANDOM_SEED)
    start_time = time.time()

    print("=" * 60)
    print("DS-CheckSimilarityLeakage: Similarity Leakage Check")
    print("=" * 60)

    # ── Step 1: Load splits ───────────────────────────────────────────────
    df_train = pd.read_csv(TRAIN_SPLIT_FILE, sep="\t")
    df_val   = pd.read_csv(VAL_SPLIT_FILE,   sep="\t")

    print(f"\nLoaded splits:")
    print(f"  Train: {len(df_train):,}")
    print(f"  Val:   {len(df_val):,}")

    # Filter by length for speed (long sequences slow down exact Jaccard)
    df_train = df_train[df_train["sequence"].str.len() < MAX_SEQ_LENGTH].reset_index(drop=True)
    df_val   = df_val  [df_val  ["sequence"].str.len() < MAX_SEQ_LENGTH].reset_index(drop=True)

    print(f"\nAfter filtering (length < {MAX_SEQ_LENGTH:,}):")
    print(f"  Train: {len(df_train):,}")
    print(f"  Val:   {len(df_val):,}")

    if len(df_train) < SAMPLE_SIZE or len(df_val) < SAMPLE_SIZE:
        raise ValueError(
            f"Not enough sequences after filtering. "
            f"Reduce MAX_SEQ_LENGTH or SAMPLE_SIZE."
        )

    # ── Step 2: Sample ────────────────────────────────────────────────────
    print(f"\nParameters:")
    print(f"  k-mer size:        {K}")
    print(f"  Threshold:         {THRESHOLD}  ← must match DS-Split.py")
    print(f"  Sample size:       {SAMPLE_SIZE} × {SAMPLE_SIZE} = {SAMPLE_SIZE**2:,} comparisons")
    print(f"  Max seq length:    {MAX_SEQ_LENGTH:,}")

    train_sample = df_train["sequence"].sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).tolist()
    val_sample   = df_val  ["sequence"].sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).tolist()

    # ── Step 3: Pre-compute k-mer integer arrays ──────────────────────────
    print(f"\nPre-computing k-mer sets...")
    t0 = time.time()
    train_kmers = [sequence_to_kmer_ints(s) for s in train_sample]
    val_kmers   = [sequence_to_kmer_ints(s) for s in val_sample]
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Step 4: Compute exact pairwise Jaccard ────────────────────────────
    print(f"\nComputing {SAMPLE_SIZE**2:,} pairwise Jaccard similarities...")
    t0 = time.time()

    all_sims = np.zeros((SAMPLE_SIZE, SAMPLE_SIZE), dtype=np.float32)
    total    = SAMPLE_SIZE * SAMPLE_SIZE
    done     = 0

    for i, t_kmers in enumerate(train_kmers):
        for j, v_kmers in enumerate(val_kmers):
            all_sims[i, j] = jaccard_from_int_sets(t_kmers, v_kmers)
            done += 1

        if (i + 1) % 50 == 0 or i + 1 == SAMPLE_SIZE:
            print(f"  {done:>8,}/{total:,}  ({100*done/total:.0f}%)  "
                  f"{time.time()-t0:.0f}s")

    flat = all_sims.ravel()

    # ── Step 5: Report ────────────────────────────────────────────────────
    max_j        = float(flat.max())
    mean_j       = float(flat.mean())
    high_pairs   = flat[flat >= THRESHOLD]
    n_high       = len(high_pairs)
    n_total      = len(flat)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Comparisons:          {n_total:,}")
    print(f"  Max Jaccard:          {max_j:.4f}")
    print(f"  Mean Jaccard:         {mean_j:.4f}")
    print(f"  Pairs >= {THRESHOLD}: {n_high}  ({100*n_high/n_total:.4f}%)")

    if max_j < THRESHOLD:
        print(f"\n  ✓ OK — max similarity {max_j:.4f} is below threshold {THRESHOLD}")
        print(f"    No significant leakage detected.")
    else:
        print(f"\n  ✗ WARNING — {n_high} pair(s) at or above threshold {THRESHOLD}!")
        print(f"    Potential data leakage detected. Re-run DS-Split.py.")

    print(f"\nPercentile distribution:")
    for p in [50, 75, 90, 95, 99, 99.9]:
        print(f"  {p:5.1f}th: {np.percentile(flat, p):.4f}")

    # ── Step 6: Save report ───────────────────────────────────────────────
    report = {
        "n_train":            len(df_train),
        "n_val":              len(df_val),
        "sample_size":        SAMPLE_SIZE,
        "k":                  K,
        "threshold":          THRESHOLD,
        "max_seq_length":     MAX_SEQ_LENGTH,
        "max_jaccard":        round(max_j, 4),
        "mean_jaccard":       round(mean_j, 4),
        "n_above_threshold":  n_high,
        "pct_above_threshold": round(100 * n_high / n_total, 4),
        "status":             "OK" if max_j < THRESHOLD else "LEAKAGE_DETECTED",
    }

    out_path = SIMILARITY_REPORT_FILE
    pd.DataFrame([report]).to_csv(out_path, index=False)
    print(f"\nReport saved to {out_path}")

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed/60:.2f} minutes")
