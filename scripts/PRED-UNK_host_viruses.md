# Prediction Pipeline for Unknown Host Viruses

This document describes the pipeline for predicting zoonotic potential of viruses with unknown hosts.

## Overview

The pipeline consists of 4 sequential scripts that:
1. Create a prediction database
2. Run predictions on unknown host viruses
3. Add predictions to the database
4. Update the webapp with results

| Step | Script | Description |
|------|--------|-------------|
| 1 | `CreateEmplySQLitePred.py` | Create empty prediction database |
| 2 | `02_predict_NEW.py` | Run predictions on unknown host viruses |
| 3 | `03_add_to_db_NEW.py` | Add predictions to database |
| 4 | `04_update_webapp.py` | Update webapp with results |

## Prerequisites

- Python 3.x with .venv virtual environment
- Fine-tuned HyenaDNA model (from FT-virsentai_v3 scripts)
- Input data: ds/ds_160k_UNK.tsv (viruses with unknown hosts)
- Required packages: torch, transformers, pandas, scipy

## Pipeline Steps

### Step 1: Create Empty Prediction Database

```powershell
python CreateEmplySQLitePred.py
```

**What it does:**
- Creates an empty SQLite database
- Sets up the following tables:
  - predictions: Stores virus prediction results
  - models: Stores model metadata
  - chembl_approved_drug: Stores drug information
  - PLAPT_AE: Stores PLAPT adverse events data

**Output:**
- `db/virsentai.sqlite3`

**Notes:**
- Schema definitions imported from config.py
- Creates database directory if it doesn't exist

---

### Step 2: Run Predictions on Unknown Host Viruses

```powershell
python 02_predict_NEW.py --input ds/ds_160k_UNK.tsv
```

**What it does:**
- Loads the fine-tuned HyenaDNA model
- Processes sequences from input TSV
- Runs inference to predict zoonotic potential
- Adds PClass_1 column with probability scores

**Input:**
- ds/ds_160k_UNK.tsv (viruses with unknown hosts)

**Output:**
- ds/ds_160k_UNK_predictions.tsv
  - All original columns preserved
  - New column: PClass_1 (probability of zoonotic potential)

**Process:**
1. Load original HyenaDNA architecture from HuggingFace
2. Copy best model weights from fine-tuned model
3. Load sequences from input TSV file
4. Tokenize sequences (max length: SEQ_MAX_LENGTH)
5. Run inference
6. Apply softmax to get probabilities
7. Extract probability of class 1 (zoonotic potential)

---

### Step 3: Add Predictions to Database

```powershell
python 03_add_to_db_NEW.py --input ds/ds_160k_UNK_predictions.tsv
```

**What it does:**
- Reads prediction TSV file
- Maps columns to database schema
- Inserts records into predictions table

**Input:**
- ds/ds_160k_UNK_predictions.tsv (from Step 2)

**Output:**
- Records added to db/virsentai.sqlite3 predictions table

**Column Mapping:**
| TSV Column | Database Column |
|------------|-----------------|
| created_at | Registration_Date |
| accession | Virus_ID |
| organism | Virus_Name |
| host | Host |
| source | Database |
| PClass_1 | prediction_score |

**Database Schema (predictions table):**
- prediction_id INTEGER PRIMARY KEY
- prediction_date TEXT
- prediction_score REAL
- model_id INTEGER
- virus_id TEXT
- virus_name TEXT
- virus_host TEXT
- virus_db TEXT
- created_at TEXT

---

### Step 4: Update Webapp

```powershell
python 04_update_webapp.py
```

**What it does:**
- Loads all predictions from database
- Creates TSV exports
- Generates JSON summary for webapp
- Updates HTML page with new statistics

**Input:**
- db/virsentai.sqlite3 (predictions table)
- webapp/index.html template

**Output Files:**
- db/all_predictions.tsv: All predictions sorted by score
- db/all_models.tsv: All models from database
- webapp/summary_stats.json: JSON summary for webapp
- webapp/index.html: Updated HTML page
- webapp/index_old.html: Backup of previous version

**JSON Summary Includes:**
- Total Scanned Viruses: count of all predictions
- Zoonotic Viruses: count where prediction_score >= threshold
- Monthly New Cases: dict of month -> count
- Top Predictions: list of top 10 zoonotic viruses

**Blacklist Filter:**
Excludes synthetic/construct/vector-like entries from zoonotic count:
- synthetic, construct, vector, plasmid, clone
- pseudovirus, pseudovirion, VLP, viriform
- metagenome, satellite, etc.

## Example Usage

```powershell
# Complete pipeline
python CreateEmplySQLitePred.py
python 02_predict_NEW.py --input ds/ds_160k_UNK.tsv
python 03_add_to_db_NEW.py --input ds/ds_160k_UNK_predictions.tsv
python 04_update_webapp.py
```

## Configuration

Key configuration from config.py:
- SQLite_PRED_FILE: Path to prediction database
- VIRSENTAI_PROB_CUTOFF: Threshold for zoonotic classification
- SEQ_MAX_LENGTH: Maximum sequence length for tokenization

## Logs

All scripts generate timestamped logs in the `logs/` directory:
- CreateEmplySQLitePred: logs/CreateEmplySQLitePred_<timestamp>.log
- 02_predict_NEW: logs/02_predict_NEW_<timestamp>.log
- 03_add_to_db_NEW: logs/03_add_to_db_NEW_<timestamp>.log
- 04_update_webapp: logs/04_update_webapp_<timestamp>.log

## Summary

This pipeline enables prediction of zoonotic potential for viruses with unknown hosts:
1. Set up the prediction database
2. Run the model on unknown host sequences
3. Store results for querying
4. Display results on the webapp dashboard