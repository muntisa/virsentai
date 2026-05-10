# Comprehensive Download of All Complete Virus Genomes from VirusHostDB

This guide specifies how to download the entire set of complete virus genomes from VirusHostDB (Genome.jp), including sequences and host annotations.

## 1. Strategy: Direct FTP Retrieval
VirusHostDB provides curated, pre-packaged files for all complete virus genomes in their database. This is more efficient than individual API queries for this specific source.

### Quality Filters Used:
- **Complete Genomes**: Strictly filters for monolithic, **non-segmented** genomes by cross-referencing with `non-segmented_virus_list.tsv`.
- **Host Annotation**: Provides manually curated host information from VirusHostDB.
- **Large Records**: Optimized to handle large genomic sequences (up to 10MB per record) by adjusting the CSV field limit.

## 2. Prerequisites
- **Python 3.x** and **.venv** virtual environment.
- Sufficient disk space for genomic FASTA data (~100-200MB compressed, 1GB+ uncompressed).

## 3. Step-by-Step Instructions

### Step 1: Activate Environment
```powershell
.\.venv\Scripts\activate
```

### Step 2: Download Data
Download metadata and genomic sequences directly from the Genome.jp FTP.
```powershell
python .\raw-data\VirusHostDB\download_data.py
```
*Outputs:*
- `raw-data/VirusHostDB/virushostdb.tsv`
- `raw-data/VirusHostDB/virushostdb.formatted.genomic.fna.gz`

### Step 4: Process and Unify
Parse the headers and metadata to create a unified TSV file.
```powershell
python .\raw-data\VirusHostDB\process_data.py
```
*Output: `raw-data/VirusHostDB/VirHostDB_raw.tsv`*

## 4. Automation Scripts
The following scripts manage the process:
- `raw-data\VirusHostDB\download_data.py`: FTP acquisition.
- `raw-data\VirusHostDB\process_data.py`: Parsing and schema mapping.

## 5. Verification
- Metadata: `raw-data/VirusHostDB/virushostdb.tsv`
- Sequences: `raw-data/VirusHostDB/virushostdb.formatted.genomic.fna.gz`
- Dataset: `raw-data/VirusHostDB/VirHostDB_raw.tsv`
