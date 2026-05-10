# SQLite Database Documentation

This document explains the schema and field mapping for the unified virus genome database used in this project.

## Database Location
- **Path**: `db/raw-viruses.sqlite3`
- **Table**: `viruses`

## Field Definitions

### 1. `db_id` (Internal Identifier)
The primary internal identifier used to track the record's origin and biological entity.
- **Format**: `[Source]:[InternalID]`
- **NCBI/RefSeq**: `NCBI` (source identifier for all RefSeq records)
- **VirusHostDB**: `VirusHostDB:[TaxID]` (e.g., `VirusHostDB:11287`)
- **BVBRC**: `BVBRC:[Virus_ID]` (e.g., `BVBRC:1000288.9`)
- **Purpose**: Groups data by source and prevents ID collisions across different databases.

#### Why the differences?
- **NCBI/RefSeq Strategy**: Uses `NCBI` as the source identifier. The unique record lookup is done via the `accession` field instead.
- **VirusHostDB Strategy**: Uses the **TaxID** because a single VirusHostDB entry represents a curated virus entity. In cases of segmented viruses (though filtered out in the final dataset), the TaxID serves as the anchor for the entire genomic set, whereas individual accessions only represent fragments.
- **BVBRC Strategy**: Uses the **Virus_ID** from VIPR, which is the stable internal identifier for each virus in the BVBRC database.

### 2. `accession` (Sequence Identifier)
The universal accession number for the genomic sequence as found in GenBank/ENA/DDBJ.
- **Purpose**: Direct lookup in international sequence databases.
*Note: In VirusHostDB, this is the raw sequence accession, which is distinct from the TaxID-based `db_id`.*

### 3. `source`
The origin of the data (e.g., `RefSeq`, `VirusHostDB`, `BVBRC`).

### 4. `organism`
The name of the virus as provided by the source.

### 5. `host`
The manual or curated host organism associated with the virus.

### 6. `completeness`
Indicates if the genome is **complete** or **partial**.
- For VirusHostDB, this is strictly filtered to only include **non-segmented** genomes.

---

## Technical Maintenance
- **Field Limit**: Genomic sequences can be large. Python's CSV reader is configured to handle fields up to 10MB during ingestion.
- **Schema Reference**: Always check `config.py` for the definitive `SQLITE_VIRUSES_SCHEMA`.

---

## Data Import Steps

### Step 1: Load NCBI/RefSeq Data

```powershell
$ .venv\Scripts\python.exe SQL_load_raw_ncbi_from_TSV.py
```

---

### Step 2: Load VirusHostDB Data

```powershell
$ .venv\Scripts\python.exe SQL-load_virushostdb_from_TSV.py
```

---

### Step 3: Load BVBRC Data

```powershell
$ .venv\Scripts\python.exe SQL-load_raw_bvbrc_from_TSV.py
```

---

## Inspecting the Database

Use the `read_sqlite.py` utility to inspect the database contents:

```powershell
python read_sqlite.py db/raw-viruses.sqlite3
```

This will display the first 5 records of each table with truncated sequences for readability.
