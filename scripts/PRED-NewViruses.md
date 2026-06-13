# VIRSentAI RefSeq Pipeline Documentation

## Overview

The VIRSentAI pipeline scans NCBI or RefSeq for new virus sequences, predicts their zoonotic potential using a fine-tuned HyenaDNA model, stores results in a SQLite database, and updates the webapp.

You can choose between two data sources:
- **NCBI GenBank**: General NCBI database (default)
- **RefSeq**: Curated RefSeq database (higher quality, manually reviewed)

## Pipeline Steps

### RefSeq Pipeline
| Step | Script | Description |
|------|--------|-------------|
| 1 | 01a_scan_viruses_raw_RefSeq.py | Scan RefSeq for viruses, save raw data with "Unknown" for missing hosts |
| 2 | 01b_fetch_unknown_hosts_RefSeq.py | Find hosts for "Unknown" entries using gemma4 |
| 3 | 02_predict_NEW_SCAN.py | Predict zoonotic potential using HyenaDNA model |
| 4 | 03_add_to_db_NEW_SCAN.py | Add predictions to SQLite database |
| 5 | 04_update_webapp.py | Update webapp with JSON/TSV files |

---

### Step 1a: 01a_scan_viruses_raw_RefSeq.py - Scanning raw viruses from RefSeq

#### Purpose
Scans the RefSeq nucleotide database for new virus sequences within a specified date range: only complete viral genomes from RefSeq curated records. If host cannot be found from RefSeq, use "Unknown".

#### Input
- **Arguments**: `--start-date` and `--end-date` (format: YYYY/MM/DD)
- **Example**: `python 01a_scan_viruses_raw_RefSeq.py --start-date 2026/04/16 --end-date 2026/05/04`

#### Output
- `virus-scan/virus_scan_2026-04-16_to_2026-05-04_RefSeq_raw.tsv`
- Columns: `Virus_ID`, `Virus_Name`, `Virus_Seq`, `Host`, `Country`, `Collection_Date`, `Registration_Date`, `Database`

#### Usage
```powershell
python 01a_scan_viruses_raw_RefSeq.py --start-date 2026/04/16 --end-date 2026/05/04
```

---

### Step 1b: 01b_fetch_unknown_hosts_RefSeq.py - LLM Host Correction

#### Purpose
Reads the raw RefSeq TSV file from Step 1a, finds hosts for viruses where host is "Unknown" using gemma4 LLM with DuckDuckGo search, and outputs a corrected TSV file.

#### Input
- **Arguments**: `--start-date` and `--end-date` (format: YYYY/MM/DD)
- **Input file**: `virus-scan/virus_scan_YYYY-MM-DD_to_YYYY-MM-DD_RefSeq_raw.tsv`

#### Output
- `virus-scan/virus_scan_2026-04-16_to_2026-05-04_RefSeq.tsv`
- Columns: `Virus_ID`, `Virus_Name`, `Virus_Seq`, `Host`, `Country`, `Collection_Date`, `Registration_Date`, `Database`, `LLM_host`

#### Usage
```powershell
python 01b_fetch_unknown_hosts_RefSeq.py --start-date 2026/04/16 --end-date 2026/05/04
```

#### Requirements
- LM Studio server running on port 1234
- DuckDuckGo search plugin enabled in LM Studio
- Model: google/gemma-4-e4b (gemma-4-E4B-it-Q4_K_M.gguf)

---

## Step 2: 02_predict_NEW_SCAN.py

### Purpose
Uses a fine-tuned HyenaDNA model to make predictions on new virus sequences from a TSV file.

### Input
- **Arguments**: `--input` (path to TSV file with 'sequence' column)
- **Input file**: `virus-scan/virus_scan_2026-04-16_to_2026-05-04_RefSeq.tsv`

### Output
- `virus-scan/virus_scan_2026-04-16_to_2026-05-04_RefSeq_predictions.tsv`
- Columns: All original + `PClass_1` (zoonotic probability)

### Usage
```powershell
python 02_predict_NEW_SCAN.py --input virus-scan/virus_scan_2026-04-16_to_2026-05-04_RefSeq.tsv
```

### Requirements
- Fine-tuned model from FT-virsentai_v3 scripts

---

## Step 3: 03_add_to_db_NEW_SCAN.py

### Purpose
Reads virus scan data with predictions from TSV and inserts into the SQLite database.

### Input
- **Arguments**: `--input` (path to prediction TSV file)
- **Input file**: `virus-scan/virus_scan_2026-04-16_to_2026-05-04_RefSeq_predictions.tsv`

### Output
- Records added to `db/virsentai.sqlite3` predictions table

### Usage
```powershell
python 03_add_to_db_NEW_SCAN.py --input virus-scan/virus_scan_2026-04-16_to_2026-05-04_RefSeq_predictions.tsv
```

---

## Step 4: 04_update_webapp.py

### Purpose
Creates TSV files with all predictions and models from the database, generates JSON summary data for the webapp, and updates the index HTML page.

### Input
- SQLite database: `db/virsentai.sqlite3`
- No command line arguments required

### Output Files
- TSV: `db/all_predictions.tsv`
- TSV: `db/all_models.tsv`
- JSON: `webapp/summary_stats.json`
- Updated: `webapp/index.html`
- Backup: `webapp/index_old.html`

### Usage
```powershell
python 04_update_webapp.py
```

---

## Running the Pipeline Manually

### Step 1a: Scan RefSeq for viruses (raw)
```powershell
python 01a_scan_viruses_raw_RefSeq.py --start-date 2026/04/16 --end-date 2026/05/04
```

### Step 1b: Fetch unknown hosts using LLM
```powershell
python 01b_fetch_unknown_hosts_RefSeq.py --start-date 2026/04/16 --end-date 2026/05/04
```

### Step 2: Predict zoonotic potential
```powershell
python 02_predict_NEW_SCAN.py --input virus-scan/virus_scan_2026-04-16_to_2026-05-04_RefSeq.tsv
```

### Step 3: Add predictions to database
```powershell
python 03_add_to_db_NEW_SCAN.py --input virus-scan/virus_scan_2026-04-16_to_2026-05-04_RefSeq_predictions.tsv
```

### Step 4: Update webapp
```powershell
python 04_update_webapp.py
```

---

## Configuration (config.py)

Key configuration variables used by the pipeline:

```python
VIRUS_SCAN_DIR = "virus-scan"      # Output folder for scan files
SQLite_PRED_FILE = "db/virsentai.sqlite3"  # Prediction database
SQLITE_VIRUSES_FILE = "db/db-viruses.sqlite3"  # Virus database
SEQ_MAX_LENGTH = 160_000           # Max sequence length for tokenization
VIRSENTAI_PROB_CUTOFF = 0.8        # Zoonotic threshold
WEBAPP_DIR = "webapp"              # Webapp directory
LOG_PATH = "logs"                  # Log directory
```

---

## Running with Virtual Environment

To use the project's virtual environment:

```powershell
.venv\Scripts\python.exe 01a_scan_viruses_raw_RefSeq.py --start-date 2026/04/16 --end-date 2026/05/04
```