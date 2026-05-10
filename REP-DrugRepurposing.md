# Drug Repurposing Pipeline

This document describes the drug repurposing pipeline for identifying potential antiviral drugs against zoonotic viruses using the PLAPT (Protein-Ligand Affinity Prediction) model.

## Overview

The pipeline consists of 3 sequential scripts that:
1. Retrieve approved drugs from ChEMBL database
2. Run PLAPT predictions for zoonotic viruses
3. Export filtered results and update webapp

| Step | Script | Description |
|------|--------|-------------|
| 1 | `REP-GetApprovedDrugs.py` | Retrieve approved drugs from ChEMBL, filter by MW |
| 2 | `REP-drug_repurposing.py` | Run PLAPT predictions for zoonotic viruses |
| 3 | `REP-exportFilteredPLAPT.py` | Export results and update webapp |

## Prerequisites

- Python 3.x with .venv virtual environment
- Required packages: chembl_webresource_client, rdkit, biopython, torch, onnxruntime, xplapt
- SQLite database with predictions table populated
- PLAPT model files (affinity_predictor.onnx)

## Pipeline Steps

### Step 1: Get Approved Drugs from ChEMBL

```powershell
python REP-GetApprovedDrugs.py --min-mw 200 --max-mw 500
```

**What it does:**
- Queries ChEMBL database for approved drugs (max_phase=4)
- Calculates molecular weight using RDKit
- Filters by specified MW range (default: 200-500 Da)

**Arguments:**
- `--min-mw`: Minimum molecular weight (Da) [default: 200]
- `--max-mw`: Maximum molecular weight (Da) [default: 500]

**Output:**
- `approved_drugs_200_500_MW.tsv`
- Columns: molecule_chembl_id, pref_name, canonical_smiles, MW

**Notes:**
- Filters out drugs without SMILES structure
- Uses RDKit for MW calculation

---

### Step 2: Drug Repurposing Predictions

```powershell
python REP-drug_repurposing.py
```

**What it does:**
- Syncs approved drugs from TSV to database
- Loads PLAPT model (protein & ligand encoders + ONNX predictor)
- Fetches zoonotic viruses (prediction_score >= 0.8)
- Extracts viral proteins from NCBI using Entrez API
- Calculates affinity against all approved drugs
- Saves high-affinity hits (>= 0.9) to database

**Input:**
- `approved_drugs_200_500_MW.tsv` (from Step 1)
- `db/virsentai.sqlite3` (predictions table)
- `PLAPT/models/affinity_predictor.onnx`

**Output:**
- `virsentai_PLAPT_AE_*.csv` (all PLAPT_AE records)

**Configuration:**
- VIRSENTAI_PROB_CUTOFF: 0.8 (minimum prediction score)
- PLAPT_AFFINITY_CUTOFF: 0.9 (minimum affinity)

**Process:**
1. Connect to SQLite database
2. Sync approved drugs to 'chembl_approved_drug' table
3. Load PLAPT ONNX model
4. Fetch viruses with score >= 0.8
5. For each virus:
   - Skip if already processed
   - Extract proteins from NCBI
   - Calculate drug-protein affinity
   - Save hits >= 0.9 to 'PLAPT_AE' table

**Notes:**
- Skips already processed viruses to avoid duplication
- Uses Entrez API for protein extraction
- Logs to logs/AE_<timestamp>.log

---

### Step 3: Export and Update Webapp

```powershell
python REP-exportFilteredPLAPT.py
```

**What it does:**
- Queries PLAPT_AE joined with predictions and drug data
- Applies filters (prediction_score >= 0.8, affinity >= 0.9)
- Applies blacklist filter (removes synthetic/construct/vector entries)
- Fetches protein names from NCBI
- Exports filtered data to CSV
- Updates webapp HTML file

**Input:**
- `db/virsentai.sqlite3` (with PLAPT_AE, predictions, chembl_approved_drug tables)
- `webapp/repurposing.html` template

**Output:**
- `virsentai_PLAPT_AE_prob-0.8_AE-0.9.csv`: Filtered results
- `webapp/repurposing.html`: Updated with new data
- `webapp/repurposing_old.html`: Backup

**Blacklist Filter:**
Removes entries where virus_name contains:
- synthetic, construct, vector, plasmid, clone
- pseudovirus, VLP, metagenome, satellite, etc.

---

## Configuration (config.py)

Key configuration variables:

```python
SQLite_PRED_FILE = "db/virsentai.sqlite3"  # Prediction database
VIRSENTAI_PROB_CUTOFF = 0.8                # Zoonotic threshold
PLAPT_AFFINITY_CUTOFF = 0.9                # Minimum affinity
PLAPT_MODEL_PATH = "PLAPT/models/affinity_predictor.onnx"
PLAPT_UPDATE_HTML = True                  # Enable HTML update
WEBAPP_DIR = "webapp"                      # Webapp directory
LOG_PATH = "logs"                          # Log directory
NCBI_ENTREZ_EMAIL = "your_email@domain.com"  # NCBI email
```

## Example Usage

### Complete Pipeline

```powershell
# Step 1: Get approved drugs
python REP-GetApprovedDrugs.py

# Step 2: Run repurposing predictions
python REP-drug_repurposing.py

# Step 3: Export and update webapp
python REP-exportFilteredPLAPT.py
```

### Custom MW Range

```powershell
# Get drugs with MW 300-600 Da
python REP-GetApprovedDrugs.py --min-mw 300 --max-mw 600
```

## Database Tables

| Table | Description |
|-------|-------------|
| predictions | Virus predictions from VirSeNtAI |
| chembl_approved_drug | ChEMBL-approved drugs (MW 200-500) |
| PLAPT_AE | Drug-virus-protein affinity results |

## Output Files

| File | Description |
|------|-------------|
| approved_drugs_200_500_MW.tsv | ChEMBL approved drugs |
| virsentai_PLAPT_AE_*.csv | All PLAPT results |
| webapp/repurposing.html | Updated webapp page |
| webapp/repurposing_old.html | Backup of previous version |

## Summary

This pipeline enables drug repurposing by:
1. Filtering suitable drug candidates by molecular weight
2. Predicting protein-drug affinity for zoonotic viruses
3. Exporting results and displaying on webapp dashboard