# Raw Database Cleaning Pipeline

This guide documents the pipeline for cleaning and deduplicating the raw virus database to produce a refined dataset.

## Overview

Process `raw-viruses.sqlite3` to create a filtered, deduplicated database `db-viruses.sqlite3`.

## Source Data

- **File**: `db/raw-viruses.sqlite3`
- **Initial records**: 99,816 (BVBRC: 47,043 | RefSeq: 18,016 | VirusHostDB: 34,757)

## Target Database

- **File**: `db/db-viruses.sqlite3`
- **Schema**: Same as `raw-viruses.sqlite3`

## Source Priority (highest to lowest)

| Priority | Source | Rationale |
|----------|--------|-----------|
| 1 | VirusHostDB | Most curated data with rich metadata (host, country, collection_date) |
| 2 | BVBRC | Includes host info from VIPR database |
| 3 | RefSeq | Generic reference sequences, often minimal host info |

## Duplicate Resolution Criteria

When multiple records share the same accession or sequence, the script keeps the one from the highest priority source. If equal priority:
1. Prefer records with `collection_date`
2. Prefer records with complete host values (not "unknown"/"NA")

## Processing Steps

### Step 0: Load raw data

- Connect to `db/raw-viruses.sqlite3`
- Read all records into pandas DataFrame
- Print counts by source (BVBRC, RefSeq, VirusHostDB)

### Step 1: Remove invalid sequences (not A,T,G,C only)

Remove sequences containing characters other than A, T, G, C.

```python
def is_valid_sequence(seq):
    if pd.isna(seq) or seq == '':
        return False
    seq_upper = str(seq).upper()
    valid_chars = set('ATGC')
    return all(c in valid_chars for c in seq_upper if c not in ' \n\t\r')
```

### Step 2: Collapse RefSeq accessions to highest version

For RefSeq records:
- Parse accession base (e.g., `NC_001234` from `NC_001234.1`)
- Sort by version number descending
- Keep highest version number per organism
- Remove version suffix

```python
def get_accession_base(accession):
    if pd.isna(accession):
        return None
    return str(accession).split('.')[0]

def get_version(accession):
    if pd.isna(accession):
        return 0
    parts = str(accession).split('.')
    try:
        return int(parts[1]) if len(parts) > 1 else 0
    except:
        return 0
```

### Step 3: Deduplicate by accession (different sources)

- Parse accession base
- Score by: source_order, date_score, host_complete
- Sort by scores (ascending for priority)
- Keep first record per accession

### Step 4: Deduplicate by identical sequences

- Convert sequences to uppercase
- Normalize: remove whitespace
- Score by: source_order, date_score, host_complete
- Sort by scores
- Keep first record per normalized sequence

### Step 5: Normalize of host values and correction of human host

SPLIT FIRST (delimiters between hosts):
- Split on: `,`, `;`, `|`, `/`
- Protect strain patterns (e.g., H3N2/2009) from being split

REMOVE per token (noise within a single host):
- Remove: `"`, `'`, `` ` ``, `\`, `{`, `}`, `*`, `#`, `?`, `=`
- Replace with space: `_`, `-`
- Replace with space: `(`, `)`, `[`, `]` (preserving content inside)
- Remove: age text, passage numbers, accession suffixes

CORRECT:
- If any normalized value is human -> return "Homo sapiens"
- Otherwise -> return first host value (lowercased)

AGE-ONLY:
- If value matches age-only pattern (e.g., "68 years old", "78 yo", "1.5 months old") -> "Homo sapiens"

### Step 6: Replace NULL/empty/invalid host values with "Unknown"

Records with NULL or empty host values are replaced with "Unknown" to ensure all records have a valid host for taxonomy mapping.

Additionally, invalid host values that do not represent real organisms are replaced with "Unknown":
- env, environment, environmental, layer, root, marc 145, not available, rd

```python
df['host'] = df['host'].fillna('Unknown')
df.loc[df['host'].astype(str).str.strip() == '', 'host'] = 'Unknown'

INVALID_HOSTS = ['env', 'environment', 'environmental', 'layer', 'root', 'marc 145', 'not available', 'rd']
df.loc[df['host'].isin(INVALID_HOSTS), 'host'] = 'Unknown'
```

### Step 7: Insert to new database

- Create `db/db-viruses.sqlite3`
- Insert cleaned records

### Step 8: Export to TSV

- Export to `db/db-viruses.tsv`
- Print final counts by source
- Print host summary (human vs non-human)

### Step 9: Remove fake/invalid virus records

Remove synthetic, recombinant, and other non-natural virus records from `db/db-viruses.sqlite3`.

**Fake virus indicators** (searched in organism field, case-insensitive):
- `recombinant`: Engineered recombinants
- `vector`: Expression vectors
- `construct`: Laboratory constructs
- `synthetic`: Artificial synthetic viruses
- `chimera`: Chimeric constructs
- `pseudotype`: Pseudotyped viruses
- `clone`: Cloned sequences
- `cDNA`: Complementary DNA constructs
- `GFP`: Green fluorescent protein fusions
- `luciferase`: Reporter gene fusions
- `wastewater`: Environmental waste samples
- `metagenome`: Uncultured metagenomic sequences
- `uncultured`: Uncultured virus sequences
- `satRNA`: Satellite RNA (not independent viruses)

**Results:**
- Before: 68,741 rows
- After: 68,650 rows
- Deleted: 91 rows containing fake indicators (recombinant, uncultured, metagenome, etc.)

```powershell
& .venv\Scripts\activate.ps1; python DB-RemoveFakeViruses.py
```

## Execution Commands

### Step 1: Clean and deduplicate raw data

```powershell
python DB-clean_raw_sqlite.py
```

*Output: `db/db-viruses.sqlite3`, `db/db-viruses.tsv`*

### Step 2: Remove fake/invalid virus records

```powershell
python DB-RemoveFakeViruses.py
```

*Removes synthetic, recombinant, and non-natural virus records.*

### Step 3: Correct unknown hosts using LLM-found data

- Purpose: Retrieve missing hosts for Unknown records using the LM Studio model and produce a TSV of Organism-Host mappings.
- Input: db/UNK_initial_host.txt extract from database `db/db-viruses.sqlite3`
- Output: db/UNK_LLM_hosts.tsv
- Notes: Requires LM Studio server on port 1234 and the DuckDuckGo plugin; logs to logs/.
- Prints each correction made

```powershell
& .venv\Scripts\activate.ps1; python DB-LLMFetchHostUNK.py
```

*Requirements: LM Studio server on port 1234 with DuckDuckGo plugin*
*Output: `db/UNK_LLM_hosts.tsv`*

### Step 4: Update database with the corrected unknown hosts

- Purpose: Apply corrections from UNK_LLM_hosts.tsv to db/db-viruses.sqlite3, replacing Unknown hosts with corrected values.
- Input: db/UNK_LLM_hosts.tsv
- Output: Updated db/db-viruses.sqlite3
- Notes: Logs to logs/
- prints number of corrected rows.

```powershell
& .venv\Scripts\activate.ps1; python DB-AddCorrectedUNKHosts.py
```

*Output: `db/db-viruses.sqlite3`*

### Step 5: Generate analytics dashboard

Generate an interactive HTML dashboard showing host distribution and sequence length statistics.

```powershell
& .venv\Scripts\activate.ps1; python DB-Dashboard.py
```

### Dashboard Features
- Summary statistics for each host class
- Sequence length distribution histograms by host type
- Mean vs Median comparison charts

*Output: `DB-Dashboard.html`*

## Scripts

| Script | Description |
|--------|-------------|
| `DB-clean_raw_sqlite.py` | Main cleaning and deduplication script |
| `DB-RemoveFakeViruses.py` | Remove fake/invalid virus records |
| `DB-LLMFetchHostUNK.py` | Fetch unknown hosts via LLM (gemma-4-e4b) |
| `DB-AddCorrectedUNKHosts.py` | Update database with corrected hosts |
| `DB-Dashboard.py` | Generate HTML analytics dashboard |

## Files

| File | Description |
|------|-------------|
| `db/raw-viruses.sqlite3` | Source database |
| `db/db-viruses.sqlite3` | Target database (cleaned) |
| `db/db-viruses.tsv` | TSV export of cleaned data |
| `db/UNK_LLM_hosts.tsv` | LLM-corrected host mappings |
| `DB-Dashboard.html` | Generated HTML dashboard |
| `logs/*.log` | Log files |