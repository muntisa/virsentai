#!/usr/bin/env python3
# =============================================================================
# CheckSplitClasses.py - Verify class distribution in train/val splits
# =============================================================================
# Purpose:
#   Check that train and val sets have balanced class distribution
# =============================================================================

import pandas as pd
import csv
import sys

# Increase field size limit for long sequences
csv.field_size_limit(2000000)


def main():
    # Load data
    train_df = pd.read_csv('fine-tuning/train_split_160k.tsv', sep='\t')
    val_df = pd.read_csv('fine-tuning/val_split_160k.tsv', sep='\t')
    
    print("=" * 50)
    print("Class Distribution Check")
    print("=" * 50)
    
    # Train set
    print("\n=== TRAIN SET ===")
    print(f"Total: {len(train_df)}")
    train_counts = train_df['label'].value_counts().sort_index()
    for label, count in train_counts.items():
        ratio = count / len(train_df) * 100
        print(f"  Label {label}: {count} ({ratio:.1f}%)")
    
    # Validation set
    print("\n=== VAL SET ===")
    print(f"Total: {len(val_df)}")
    val_counts = val_df['label'].value_counts().sort_index()
    for label, count in val_counts.items():
        ratio = count / len(val_df) * 100
        print(f"  Label {label}: {count} ({ratio:.1f}%)")
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Train: {len(train_df)} samples (80%)")
    print(f"Val:   {len(val_df)} samples (20%)")
    print(f"Total: {len(train_df) + len(val_df)}")


if __name__ == "__main__":
    main()