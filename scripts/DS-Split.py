#!/usr/bin/env python3
"""
Data Splitting Script

Splits balanced dataset into train/validation subsets using MinHash + LSH
to avoid similarity leakage between sets.

Usage:
    python DS-Split.py

Input:
    - ds/ds_160k_balanced.tsv (from DS-Undersampling.py)

Output:
    - ds/train_split.tsv: Training set (80%)
    - ds/val_split.tsv: Validation set (20%)

Algorithm:
    1. Encode k-mers as integers (2 bits/base)
    2. Generate MinHash signatures (128 hash functions)
    3. LSH banding to find candidate pairs (32 bands)
    4. Verify candidates with full MinHash Jaccard
    5. Union-Find for transitive clustering
    6. StratifiedGroupKFold for balanced split

Parameters (must match DS-CheckSimilarityLeakage.py):
    - K = 9 (k-mer size)
    - SIMILARITY_THRESHOLD = 0.60

Notes:
    - ~30 MB for 30k sequences vs ~42 GB for raw k-mer sets
    - P(detect pair | Jaccard=0.60) ≈ 98.8%
    - No cluster overlap between train/val
"""

import os
import csv
csv.field_size_limit(2_000_000)

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedGroupKFold
from config import *

print("=" * 60)
print("DS-Split: Data Preparation Script (MinHash + LSH)")
print("=" * 60)

INPUT_FILE   = DS_BALANCED_INPUT
TRAIN_OUTPUT = TRAIN_SPLIT_FILE
VAL_OUTPUT   = VAL_SPLIT_FILE

# ── SHARED PARAMETERS (must match DS-CheckSimilarityLeakage.py) ──────────────
K                    = 9     # k-mer size
SIMILARITY_THRESHOLD = 0.60  # Jaccard threshold for same-cluster assignment

# ── MINHASH + LSH PARAMETERS ─────────────────────────────────────────────────
N_HASH  = 128   # MinHash functions. More → better Jaccard estimate.
                # StdErr(estimate) ≈ 1/√N_HASH ≈ 0.089 with 128.
N_BANDS = 32    # LSH bands. N_ROWS = N_HASH // N_BANDS = 4.
                # P(detect | J=0.60) = 1-(1-0.60^4)^32 ≈ 98.8%
                # P(false candidate | J=0.30) ≈ 23% → filtered by verify step.

RANDOM_SEED = 2025

# ── BASE ENCODING LOOKUP ──────────────────────────────────────────────────────
# Maps ASCII byte → 2-bit value: A=0, C=1, G=2, T=3, N/other=0
_BASE_MAP = np.zeros(256, dtype=np.uint32)
for _char, _val in zip("AaCcGgTtNn", [0, 0, 1, 1, 2, 2, 3, 3, 0, 0]):
    _BASE_MAP[ord(_char)] = _val


# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def sequence_to_kmer_ints(seq: str, k: int = K) -> np.ndarray:
    """Convert a DNA sequence to a sorted array of unique integer k-mers.

    Encoding: each base → 2 bits (A=00, C=01, G=10, T=11).
    A 9-mer fits in 18 bits → uint32. Zero string allocation.

    Uses stride tricks to build sliding windows without copying the array,
    then a dot product with powers-of-4 to encode each window as an integer.

    Args:
        seq: DNA string (uppercase or lowercase; N treated as A).
        k:   k-mer length.

    Returns:
        uint32 numpy array of unique integer k-mers (sorted).
    """
    arr = _BASE_MAP[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    n_kmers = len(arr) - k + 1
    if n_kmers <= 0:
        return np.array([], dtype=np.uint32)

    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.uint32)

    # as_strided creates an (n_kmers × k) view — no copy, no extra allocation.
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=(n_kmers, k), strides=strides)

    return np.unique((windows @ powers).astype(np.uint32))


def compute_minhash_signature(
    kmer_ints: np.ndarray,
    hash_a: np.ndarray,
    hash_b: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Compute a MinHash signature via universal hashing.

    h_i(x) = a_i * x + b_i  (uint64 arithmetic, wraps at 2^64)
    MinHash_i = min over all k-mers of h_i(k-mer)

    Processes k-mers in chunks to cap peak memory at
    n_hash × chunk_size × 8 bytes = 128 × 4096 × 8 = 4 MB.

    Args:
        kmer_ints:  uint32 array of unique k-mer integers.
        hash_a:     uint64 array of shape (n_hash,) — multipliers.
        hash_b:     uint64 array of shape (n_hash,) — offsets.
        chunk_size: k-mers processed per iteration.

    Returns:
        uint64 array of shape (n_hash,) — MinHash signature.
    """
    n_hash = len(hash_a)
    sig = np.full(n_hash, np.iinfo(np.uint64).max, dtype=np.uint64)

    for start in range(0, len(kmer_ints), chunk_size):
        chunk = kmer_ints[start : start + chunk_size].astype(np.uint64)
        # shape: (n_hash, chunk_size) — uint64 overflow is intentional (mod 2^64)
        hashed = hash_a[:, None] * chunk[None, :] + hash_b[:, None]
        sig = np.minimum(sig, hashed.min(axis=1))

    return sig


def cluster_sequences(
    sequences: list,
    k: int = K,
    threshold: float = SIMILARITY_THRESHOLD,
    n_hash: int = N_HASH,
    n_bands: int = N_BANDS,
    seed: int = RANDOM_SEED,
) -> list:
    """Cluster DNA sequences by k-mer Jaccard similarity.

    Three-phase pipeline:

      Phase 1 — MinHash signatures
        Compress each sequence's k-mer set into a compact n_hash-integer
        signature. Memory: O(n × n_hash) instead of O(n × unique_kmers).

      Phase 2 — LSH banding
        Split each signature into n_bands bands of n_rows integers.
        Sequences sharing at least one band hash to the same bucket and
        become candidate pairs — without scanning all O(n²) pairs.

      Phase 3 — Verify + Union-Find
        For each candidate pair, estimate Jaccard = fraction of n_hash
        positions with equal MinHash value. Merge pairs above threshold
        into connected components via Union-Find (path-compressed).
        Handles transitive similarity correctly: if A~B and B~C, all
        three end up in the same cluster regardless of A~C similarity.

    Args:
        sequences: list of DNA strings.
        k:         k-mer size.
        threshold: Jaccard similarity threshold.
        n_hash:    number of MinHash functions (accuracy).
        n_bands:   number of LSH bands (recall/precision tradeoff).
        seed:      random seed for reproducibility.

    Returns:
        List of integer cluster IDs, one per sequence.
    """
    n = len(sequences)
    n_rows = n_hash // n_bands

    rng = np.random.default_rng(seed)
    hash_a = rng.integers(1, 2**64, size=n_hash, dtype=np.uint64)   # non-zero multipliers
    hash_b = rng.integers(0, 2**64, size=n_hash, dtype=np.uint64)

    # ── Phase 1: MinHash signatures ───────────────────────────────────────
    print(f"   Phase 1/3 — MinHash signatures  "
          f"(n_hash={n_hash}, k={k}, n={n:,})")
    t0 = time.time()
    signatures = np.zeros((n, n_hash), dtype=np.uint64)

    for i, seq in enumerate(sequences):
        kmer_ints = sequence_to_kmer_ints(seq, k)
        if len(kmer_ints) > 0:
            signatures[i] = compute_minhash_signature(kmer_ints, hash_a, hash_b)
        if (i + 1) % 5_000 == 0 or i + 1 == n:
            print(f"   {i+1:>6,}/{n:,}  ({100*(i+1)/n:.0f}%)  "
                  f"{time.time()-t0:.0f}s elapsed")

    sig_mb  = signatures.nbytes / 1e6
    raw_est = n * 24_000 * 40 / 1e9   # rough estimate for raw Python sets
    print(f"   Signatures: {sig_mb:.0f} MB  "
          f"(vs ~{raw_est:.0f} GB for raw k-mer sets)\n")

    # ── Phase 2: LSH banding → candidate pairs ────────────────────────────
    print(f"   Phase 2/3 — LSH banding  "
          f"({n_bands} bands × {n_rows} rows)")
    t0 = time.time()
    candidate_pairs: set = set()

    for band in range(n_bands):
        cs = band * n_rows
        ce = cs + n_rows
        band_sigs = signatures[:, cs:ce]   # (n, n_rows)

        buckets: dict = {}
        for i in range(n):
            key = band_sigs[i].tobytes()
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(i)

        for bucket in buckets.values():
            if len(bucket) < 2:
                continue
            for a in range(len(bucket)):
                for b in range(a + 1, len(bucket)):
                    u, v = bucket[a], bucket[b]
                    if u > v:
                        u, v = v, u
                    candidate_pairs.add((u, v))

    all_pairs = n * (n - 1) // 2
    print(f"   Candidate pairs: {len(candidate_pairs):,}  "
          f"(vs {all_pairs:,} all-pairs, "
          f"{100*len(candidate_pairs)/all_pairs:.3f}%)  "
          f"{time.time()-t0:.1f}s\n")

    # ── Phase 3: Verify + Union-Find ──────────────────────────────────────
    print(f"   Phase 3/3 — Verifying {len(candidate_pairs):,} candidates "
          f"+ Union-Find clustering...")
    t0 = time.time()

    parent = list(range(n))

    def find(x: int) -> int:
        """Path-compressed find."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path halving
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        """Merge clusters containing x and y."""
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    merged = 0
    for i, j in candidate_pairs:
        # Unbiased Jaccard estimate: fraction of positions where MinHash agrees
        est_jaccard = float(np.mean(signatures[i] == signatures[j]))
        if est_jaccard >= threshold:
            union(i, j)
            merged += 1

    print(f"   Merged: {merged:,} pairs above threshold  "
          f"{time.time()-t0:.1f}s")

    root_to_id: dict = {}
    assignments = []
    for i in range(n):
        root = find(i)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        assignments.append(root_to_id[root])

    print(f"   Clusters: {len(root_to_id):,}")
    return assignments


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Step 1: Load data
    print(f"\n1. Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, sep="\t")
    print(f"   Loaded {len(df):,} rows")

    seq_len = df["sequence"].str.len()
    print(f"   Sequence lengths — "
          f"min: {seq_len.min():,}  "
          f"max: {seq_len.max():,}  "
          f"mean: {seq_len.mean():,.0f}")

    # Step 2: Build or load splits
    if os.path.exists(TRAIN_OUTPUT) and os.path.exists(VAL_OUTPUT):
        print(f"\nExisting splits found — loading to verify...")
        train_df = pd.read_csv(TRAIN_OUTPUT, sep="\t")
        val_df   = pd.read_csv(VAL_OUTPUT,   sep="\t")
        print(f"   Train: {len(train_df):,} rows")
        print(f"   Val:   {len(val_df):,} rows")

    else:
        print(f"\n2. Clustering sequences "
              f"(k={K}, threshold={SIMILARITY_THRESHOLD})...")
        df["cluster_id"] = cluster_sequences(df["sequence"].tolist())

        # StratifiedGroupKFold guarantees:
        #   • No cluster_id appears in both train AND val
        #   • Label distribution (0/1) is balanced in both sets
        print(f"\n3. Splitting with StratifiedGroupKFold (80% train / 20% val)...")
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        train_idx, val_idx = next(
            sgkf.split(df, y=df["label"], groups=df["cluster_id"])
        )

        train_df = df.iloc[train_idx].copy()
        val_df   = df.iloc[val_idx].copy()
        print(f"   Train: {len(train_df):,} rows")
        print(f"   Val:   {len(val_df):,} rows")

        print(f"\n4. Saving...")
        train_df.to_csv(TRAIN_OUTPUT, sep="\t", index=False)
        val_df.to_csv(VAL_OUTPUT,     sep="\t", index=False)
        print(f"   {TRAIN_OUTPUT}")
        print(f"   {VAL_OUTPUT}")

    # Step 3: Verify — no cluster overlap, balanced labels
    print(f"\n5. Verification...")
    overlap = set(train_df["cluster_id"]).intersection(set(val_df["cluster_id"]))
    if overlap:
        print(f"   ✗ Leakage check: FAILED — {len(overlap)} overlapping clusters!")
    else:
        print(f"   ✓ Leakage check: OK — no cluster overlap")

    t_dist = dict(train_df["label"].value_counts(normalize=True).round(3) * 100)
    v_dist = dict(val_df["label"].value_counts(normalize=True).round(3) * 100)
    print(f"   Label distribution (%):")
    print(f"   Train: {t_dist}")
    print(f"   Val:   {v_dist}")

    print("\n" + "=" * 60)
    print("DS-Split complete!  Run DS-CheckSimilarityLeakage.py to verify.")
    print("=" * 60)
