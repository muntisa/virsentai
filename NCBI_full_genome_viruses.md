# Comprehensive Download of All Complete Virus Genomes from NCBI

This guide specifies how to download the entire set of complete virus genomes from NCBI, including sequences and metadata (ID, Name, Host).

## 1. Strategy: Hybrid Discovery & Enrichment
Due to the large number of sequences (~15,000+ RefSeq records), we use a high-reliability **Hybrid Strategy** to avoid server timeouts and ensure metadata completeness (especially **Host Information**).

### Quality Filters Used:
- **Taxon: Viruses**: Uses the formal scientific name for the kingdom-level group.
- **Complete RefSeq**: Restricts the dataset to curated, full-length sequences validated by NCBI staff.

### How it works:
1.  **Phase 1: Discovery (TaxID-based)**: We use the robust `genome` subcommand to discover the unique **Taxonomy IDs** for all complete RefSeq viruses. This avoids timeouts associated with broad taxonomic searches on the `virus` endpoint.
2.  **Phase 2: Enrichment (Batched)**: We take the discovered TaxIDs and fetch their full metadata in batches of 100 via the `virus` subcommand. We use `--inputfile` for batching to avoid "Header Too Large" (HTTP 431) errors.
3.  **Phase 3: Sequence Download**: Sequences are batched (5,000 per ZIP) for maximum network stability.

## 2. Prerequisites
- **Python 3.x** and **.venv** virtual environment.
- **NCBI Datasets CLI** tools in `raw-data/NCBI/`.
- **NCBI API Key** (Highly Recommended): Add your key to `config.py` to increase your rate limit from 3 to 10 requests per second.

## 3. Step-by-Step Instructions

### Step 1: Activate Environment
```powershell
.\.venv\Scripts\activate
```

### Step 2: Download Full Metadata
Fetch the high-fidelity metadata for all complete RefSeq viruses. 
```powershell
python .\raw-data\NCBI\fetch_virus_metadata.py
```
*Output: `raw-data/NCBI/all_complete_metadata.jsonl`*

### Step 3: Run Batch Download
Download the genomic sequences in batches of 5,000.
```powershell
python .\raw-data\NCBI\batch_download_viruses.py
```
*Output: `raw-data/NCBI/batches/virus_batch_*.zip`*

### Step 4: Convert to Unified TSV
Merge metadata and genomic sequences into a single TSV file.
```powershell
python .\raw-data\NCBI\convert_to_tsv.py
```
*Output: `raw-data/NCBI/NCBI_raw.tsv`*

### Step 5: (Optional) Download Sample
For quick testing, download a small sample of 5 complete RefSeq viruses.
```powershell
python .\raw-data\NCBI\download_sample.py
```
*Output: `accessions_sample.txt`, `sample_viruses.zip`*

## 4. Automation Scripts
The following scripts manage the process:
- `raw-data\NCBI\fetch_virus_metadata.py`: Phase 1 & 2 logic.
- `raw-data\NCBI\batch_download_viruses.py`: Phase 3 batching.
- `raw-data\NCBI\convert_to_tsv.py`: Merging into TSV.
- `raw-data\NCBI\download_sample.py`: Download sample for testing.

## 5. Verification
- Metadata: `raw-data/NCBI/all_complete_metadata.jsonl`
- Sequences: `raw-data/NCBI/batches/*.zip`
- Dataset: `raw-data/NCBI/NCBI_raw.tsv`
