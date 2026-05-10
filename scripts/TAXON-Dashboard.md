# Taxonomy Dashboard Pipeline

This document describes the taxonomy data collection and visualization pipeline for the VirSeNtAI project.

## Overview

The pipeline consists of 5 scripts that:
1. Fetch taxonomy data for all organisms in the dataset
2. Find organisms with missing taxonomy
3. Fetch taxonomy one by one from NCBI
4. Generate statistics JSON
5. Create interactive HTML dashboard

| Step | Script | Description |
|------|--------|-------------|
| 1 | `DS-GetOrganismTaxonomy.py` | Fetch taxonomy for all organisms in dataset |
| 2 | `DS-FindMissingTaxonomy.py` | Find organisms with missing taxonomy |
| 3 | `DS-taxonomy_fetcher.py` | Fetch taxonomy one by one from NCBI |
| 4 | `generate_taxonomy_stats.py` | Generate statistics JSON |
| 5 | `taxonomy_html.py` | Generate taxonomy dashboard HTML |

## Prerequisites

- Python 3.x with .venv virtual environment
- Required packages: biopython, pandas
- NCBI API key (recommended for faster rate limits)
- config.py with NCBI_ENTREZ_EMAIL and NCBI_API_KEY

## Pipeline Steps

### Step 1: Get Organism Taxonomy

```powershell
python DS-GetOrganismTaxonomy.py
```

**What it does:**
- Reads organisms from the balanced dataset
- Fetches taxonomy data from NCBI for each organism
- Saves full taxonomy information

**Input:**
- `fine-tuning/ds_160k_balanced.tsv`

**Output:**
- `ds/ds_160k_balanced_taxonomy.tsv` (all columns + taxonomy)
- `ds/ds_160k_balanced_taxonomy_incremental.tsv` (backup after each organism)

**Taxonomy Columns Added:**
- TaxID, Name, Rank, realm, kingdom, phylum, class, order, family, genus, species

**Performance:**
- With API key: ~50 minutes for 15,000 organisms (10 req/sec)
- Without API key: ~3 hours (3 req/sec)
- Delay between calls: 0.1s

---

### Step 2: Find Missing Taxonomy

```powershell
python DS-FindMissingTaxonomy.py
```

**What it does:**
- Compares original dataset with fetched taxonomy
- Identifies organisms still missing taxonomy data

**Input:**
- `fine-tuning/ds_160k_balanced.tsv`
- `ds/ds_160k_balanced_taxonomy_incremental.tsv`

**Output:**
- `ds/ds_160k_balanced_missing_taxonomy.tsv`

**Notes:**
- Shows list of missing organisms
- Useful for resuming interrupted fetching

---

### Step 3: Fetch Taxonomy (One by One)

```powershell
python DS-taxonomy_fetcher.py
```

**What it does:**
- Fetches taxonomy data for individual organisms
- Processes hosts from non-human dataset

**Input:**
- `ds/ds_160k_nonHuman.tsv` OR
- `ds/organisms_without_taxonomy.txt`

**Output:**
- `ds/hosts_taxonomy.tsv`

**Output Columns:**
- name, taxid, rank, superkingdom, kingdom, phylum, class, order, family, subfamily, genus, species

**Notes:**
- Processes organisms one by one
- Useful for incremental processing

---

### Step 4: Generate Statistics JSON

```powershell
python generate_taxonomy_stats.py
```

**What it does:**
- Analyzes taxonomy-enriched dataset
- Generates comprehensive statistics
- Creates taxonomy dashboard HTML

**Input:**
- `ds/ds_160k_balanced_taxonomy.tsv`

**Output:**
- `ds/taxonomy_stats.json`
- `webapp/taxonomy_dashboard.html`

**Statistics Generated:**
- Overview: total records, zoonotic/non-zoonotic counts, unique counts
- Sequence length: min, max, mean, median, std, percentiles
- Taxonomy counts: top 15 for each rank (realm, kingdom, phylum, class, order, family, genus, species)
- Host distribution: top 25 hosts with label breakdown
- Data sources: BVBRC, VirusHostDB, RefSeq counts
- Geographic: top 25 countries
- Completeness: sequence completeness status

---

### Step 5: Generate Dashboard HTML

```powershell
python taxonomy_html.py
```

**What it does:**
- Generates interactive HTML dashboard from JSON statistics

**Input:**
- `webapp/taxonomy_stats.json` (from Step 4)

**Output:**
- `webapp/taxonomy_dashboard.html`

**Dashboard Features:**
- Statistics overview cards
- Sequence length statistics
- Taxonomy distribution (pie charts for each rank)
- Host distribution (top 15 non-human hosts)
- Data source distribution
- Geographic distribution (top 15 countries)

**Requirements:**
- echarts (for pie charts)
- tailwindcss (for styling)
- Inter font

---

## Configuration (config.py)

Key configuration variables:

```python
NCBI_ENTREZ_EMAIL = "your_email@domain.com"  # Required for NCBI API
NCBI_API_KEY = "your_api_key"                  # Recommended for faster rate limits
LOG_PATH = "logs"                              # Log directory
DS_BALANCED_FILE = "ds/ds_160k_balanced.tsv"   # Input dataset
```

## Output Files

| File | Description |
|------|-------------|
| ds/ds_160k_balanced_taxonomy.tsv | Full dataset with taxonomy |
| ds/ds_160k_balanced_taxonomy_incremental.tsv | Backup during fetching |
| ds/ds_160k_balanced_missing_taxonomy.tsv | Missing taxonomy records |
| ds/hosts_taxonomy.tsv | Host taxonomy data |
| ds/taxonomy_stats.json | Statistics JSON |
| webapp/taxonomy_dashboard.html | Interactive dashboard |

## Example Usage

### Complete Pipeline

```powershell
# Step 1: Fetch taxonomy for all organisms
python DS-GetOrganismTaxonomy.py

# Step 2: Find missing taxonomy
python DS-FindMissingTaxonomy.py

# Step 3: Fetch remaining taxonomy (if needed)
python DS-taxonomy_fetcher.py

# Step 4: Generate statistics and dashboard
python generate_taxonomy_stats.py
```

### Resume Interrupted Process

```powershell
# After interruption, find what's missing
python DS-FindMissingTaxonomy.py

# Fetch missing organisms
python DS-taxonomy_fetcher.py

# Regenerate statistics
python generate_taxonomy_stats.py
```

## Database Tables

| Field | Description |
|-------|-------------|
| realm | Highest taxonomic rank (e.g., Riboviria) |
| kingdom | Second rank (e.g., Orthornavirae) |
| phylum | Third rank |
| class | Fourth rank |
| order | Fifth rank |
| family | Sixth rank |
| genus | Seventh rank |
| species | Eighth rank |

## Summary

This pipeline enables comprehensive taxonomy analysis by:
1. Fetching NCBI taxonomy data for all organisms
2. Handling missing data with incremental processing
3. Generating statistics for visualization
4. Creating interactive HTML dashboard