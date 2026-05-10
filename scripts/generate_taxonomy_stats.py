#!/usr/bin/env python3
"""
Generate Taxonomy Statistics

Analyzes the taxonomy-enriched balanced dataset and generates comprehensive
statistics for the taxonomy dashboard.

Usage:
    python generate_taxonomy_stats.py

Input:
    - ds/ds_160k_balanced_taxonomy.tsv (with taxonomy columns from NCBI)

Output:
    - ds/taxonomy_stats.json (pre-calculated statistics for dashboard)
    - webapp/taxonomy_dashboard.html (updated dashboard)

Statistics Generated:
    - Overview: total records, zoonotic/non-zoonotic counts, unique counts
    - Sequence length: min, max, mean, median, std, percentiles
    - Taxonomy counts: top 15 for each rank (realm, kingdom, phylum, class, order, family, genus, species)
    - Host distribution: top 25 hosts with label breakdown
    - Data sources: BVBRC, VirusHostDB, RefSeq counts
    - Geographic: top 25 countries
    - Completeness: sequence completeness status

Requirements:
    - pandas
    - taxonomy_html.py for dashboard generation

Notes:
    - Runs after taxonomy data is fully fetched
    - Generates JSON for fast dashboard loading
    - Creates interactive HTML dashboard
"""

import pandas as pd
import json
import re

def analyze_taxonomy_data(input_file='ds/ds_160k_balanced_taxonomy.tsv', output_file='webapp/taxonomy_stats.json'):
    """
    Analyzes taxonomy dataset and generates statistics JSON.
    Also updates the taxonomy dashboard HTML.
    """
    df = pd.read_csv(input_file, sep='\t')
    
    total = len(df)
    zoonotic = df[df['label'] == 1].shape[0]
    non_zoonotic = df[df['label'] == 0].shape[0]
    
    length_stats = {
        'min': int(df['length'].min()),
        'max': int(df['length'].max()),
        'mean': float(df['length'].mean()),
        'median': float(df['length'].median()),
        'std': float(df['length'].std()),
        'p25': float(df['length'].quantile(0.25)),
        'p75': float(df['length'].quantile(0.75))
    }
    
    source_counts = df['source'].value_counts().to_dict()
    host_counts = df['host'].value_counts().head(25).to_dict()
    country_counts = df['country'].value_counts().head(25).to_dict()
    
    completeness = df['completeness'].str.lower().value_counts().to_dict()
    
    ranks = ['realm', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    taxonomy_counts = {}
    
    for rank in ranks:
        taxonomy_counts[rank] = df[rank].value_counts().head(15).to_dict()
        taxonomy_counts[f'{rank}_zoonotic'] = df[df['label']==1][rank].value_counts().head(15).to_dict()
        taxonomy_counts[f'{rank}_non_zoonotic'] = df[df['label']==0][rank].value_counts().head(15).to_dict()
    
    host_zoonotic = df[df['label']==1]['host'].value_counts().head(20).to_dict()
    host_non_zoonotic = df[df['label']==0]['host'].value_counts().head(20).to_dict()
    
    stats = {
        'total': total,
        'zoonotic': zoonotic,
        'non_zoonotic': non_zoonotic,
        'unique_organisms': int(df['organism'].nunique()),
        'unique_hosts': int(df['host'].nunique()),
        'unique_species': int(df['species'].nunique()),
        'length_stats': length_stats,
        'source_counts': source_counts,
        'host_counts': host_counts,
        'host_zoonotic': host_zoonotic,
        'host_non_zoonotic': host_non_zoonotic,
        'taxonomy_counts': taxonomy_counts,
        'completeness': completeness,
        'country_counts': country_counts
    }
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to {output_file}")
    
    return stats

if __name__ == "__main__":
    analyze_taxonomy_data()