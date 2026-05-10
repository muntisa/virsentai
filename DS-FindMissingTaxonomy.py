#!/usr/bin/env python3
"""
Find Missing Taxonomy

Identifies organisms in the balanced dataset that don't have taxonomy data yet.

Usage:
    python DS-FindMissingTaxonomy.py

Input:
    - fine-tuning/ds_160k_balanced.tsv (original balanced dataset)
    - ds/ds_160k_balanced_taxonomy_incremental.tsv (taxonomy data fetched so far)

Output:
    - ds/ds_160k_balanced_missing_taxonomy.tsv (records with missing taxonomy)

Process:
    1. Read original balanced dataset
    2. Read incremental taxonomy file
    3. Find organisms NOT in taxonomy file
    4. Export missing records to TSV

Notes:
    - Shows list of missing organisms
    - Useful for resuming interrupted taxonomy fetching
"""