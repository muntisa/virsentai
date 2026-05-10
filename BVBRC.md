# Comprehensive Download of Virus Genomes from BV-BRC

This guide specifies how to download complete virus genomes from the Bacterial Viral Bioinformatics Resource Center (BV-BRC), including sequences and host annotations.

## 1. Strategy: BV-BRC API Retrieval
BV-BRC provides a comprehensive API for downloading virus genome data including metadata, sequences, and host information.

### Quality Filters Used:
- **Complete Genomes**: Only monolithic, non-segmented genomes.
- **Host Annotation**: Provides curated host information from BV-BRC database.
- **Sequence Data**: Full genomic sequences in FASTA format.

## 2. Prerequisites
- **Python 3.x** and **.venv** virtual environment.
- Sufficient disk space for genomic FASTA data.

## 3. Step-by-Step Instructions

### Step 1: Activate Environment
```powershell
.venv\Scripts\activate
```

### Step 2: Download Metadata
Download virus metadata from BV-BRC API.
```powershell
python raw-data\BVBRC\ds-3_1-Viprbrc_get_viruses_metadata.py
```
*Output: `raw-data/BVBRC/all_hosts_viruses_metadata_noSeq.csv`*

### Step 3: Filter Metadata
Filter and clean the metadata to keep only the latest version of each virus.
```powershell
python raw-data\BVBRC\ds-3_2-Viprbrc_filter_viruses_metadata.py
```
*Output: `raw-data/BVBRC/all_hosts_viruses_metadata_noSeq_filtered.csv`*

### Step 4: Download Sequence Data
Download genomic sequences for the filtered viruses.
```powershell
python raw-data\BVBRC\ds-3_3-Viprbrc_get_viruses_data.py
```
*Output: `raw-data/BVBRC/Viprbrc_all_hosts_viruses_with_seqs.tsv`*

## 4. Automation Scripts
The following scripts manage the process:
- `raw-data\BVBRC\ds-3_1-Viprbrc_get_viruses_metadata.py`: API metadata retrieval
- `raw-data\BVBRC\ds-3_2-Viprbrc_filter_viruses_metadata.py`: Metadata filtering and cleaning
- `raw-data\BVBRC\ds-3_3-Viprbrc_get_viruses_data.py`: Genomic sequence download

## 5. Verification
- Metadata: `raw-data/BVBRC/all_hosts_viruses_metadata_noSeq.csv`
- Filtered: `raw-data/BVBRC/all_hosts_viruses_metadata_noSeq_filtered.csv`
- Sequences: `raw-data/BVBRC/Viprbrc_all_hosts_viruses_with_seqs.tsv`