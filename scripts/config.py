import os

# VirSentAI: Viral Sentry AI – Intelligent Zoonotic Surveillance Platform
# Configuration for all the scripts

# Viral Full Genome DNA Datasets
RAW_DATA_NCBI_PATH        = "raw-data/NCBI"
RAW_DATA_VirusHostDB_PATH = "raw-data/VirusHostDB"
RAW_DATA_BVBRC_PATH       = "raw-data/BVBRC"

# BVBRC_CLI_BIN_PATH        = os.path.join(RAW_DATA_Viprbrc_PATH, "cli", "bin")

# db
DB_PATH = "db"
SQLITE_VIRUSES_FILE      = "db/raw-viruses.sqlite3" # mixed raw data
SQLITE_CORR_VIRUSES_FILE = "db/db-viruses.sqlite3"  # mixed corrected data
TSV_CORR_VIRUSES_FILE    = "db/db-viruses.tsv"      # the same as TSV

# LOGS
LOG_PATH = "logs"
LOG_DIR  = "logs"

# UNK host viruses
UNK_DATA_FILE     = "ds/ds_160k_UNK.tsv"
UNK_PREDICT_FILE  = "predictions/ds_160k_UNK_preds.tsv"
UNK_TOKENIZED_DIR = "fine-tuning/tokenized_UNK_dataset"

# Dataset files
DS_160K_FILE =      "ds/ds_160k.tsv"
DS_HUMAN_FILE =     "ds/ds_160k_Human.tsv"
DS_NONHUMAN_FILE =      "ds/ds_160k_nonHuman.tsv"
DS_TAXONOMY_FILE =       "ds/ds_160k_balanced_taxonomy.tsv"
DS_TAXONOMY_INCREMENTAL_FILE = "ds/ds_160k_balanced_taxonomy_incremental.tsv"
DS_UNK_FILE =       "ds/ds_160k_UNK.tsv"
DS_BALANCED_FILE =  "ds/balanced_diverse_genomes.tsv"
DS_BALANCED_INPUT = "fine-tuning/ds_160k_balanced.tsv"
TRAIN_SPLIT_FILE =  "fine-tuning/train_split_160k.tsv"
VAL_SPLIT_FILE =    "fine-tuning/val_split_160k.tsv"
SIMILARITY_REPORT_FILE = "fine-tuning/similarity_leakage_report.csv"

# ChEMBL Drug Filter
DEFAULT_MIN_MW = 200
DEFAULT_MAX_MW = 500
DRUG_OUTPUT_FILE = "approved_drugs_{min_mw}_{max_mw}_MW.tsv"

# Virus scan output directory
VIRUS_SCAN_DIR = "virus-scan"

# Model configuration
NEW_MODEL_DIR = "fine-tuning/models/hyena_fine_tunning"
SEQ_MAX_LENGTH = 160_000
EVAL_BATCH_SIZE = 4
PREDICT_TEMP_SUBFOLDER = "predict_temp"

# Zoonotic threshold (virsentai probability)
VIRSENTAI_PROB_CUTOFF = 0.8

# Webapp configuration
WEBAPP_DIR = "webapp"
JSON_FILE = "webapp/summary_stats.json"
ALL_PREDS_FILE = "db/all_predictions.tsv"
ALL_MODELS_FILE = "db/all_models.tsv"

# NCBI Entrez
NCBI_ENTREZ_EMAIL = "your_email@gmail.com" # change with your email
NCBI_API_KEY      = "YOUR_NCBI_API" # Add your NCBI API Key here for better performance
NCBI_ENTREZ_MAX_RESULTS = 10000
NCBI_BATCH_SIZE = 200
NCBI_ENTREZ_DELAY = 0.34
NCBI_MAX_RETRIES = 3
NCBI_RETRY_DELAY = 5

# Virsentai model version
DEFAULT_MODEL_ID = 3

# SQLite schema for raw-viruses.sqlite3 and db-viruses.sqlite3
SQLITE_VIRUSES_SCHEMA = """
CREATE TABLE IF NOT EXISTS viruses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    db_id TEXT,
    accession TEXT,
    sequence TEXT,
    length INTEGER,
    source TEXT,
    collection_date TEXT,
    country TEXT,
    host TEXT,
    organism TEXT,
    segment TEXT,
    completeness TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Blacklist terms for filtering
ZOONOTIC_VIRUS_KEYWORDS = [
    'synthetic construct', 'synthetic', 'construct', 'vector', 'plasmid', 'clone',
    'pseudovirus', 'pseudovirion', 'virus-like particle', 'virus like particle', 'VLP',
    'viriform', 'gene transfer agent', 'gene-transfer-agent', 'GTA', 'artificial',
    'alphasatellite', 'betasatellite', 'satellite', 'MAG', 'metagenome', 'Lake Sarah', 'TYR', 'BGG_'
]

# Drug Repurposing (PLAPT)
PLAPT_AFFINITY_CUTOFF = 9
PLAPT_UPDATE_HTML = True
PLAPT_EXPORT_FILE = f"virsentai_PLAPT_AE_prob-{VIRSENTAI_PROB_CUTOFF}_AEcuroff-{PLAPT_AFFINITY_CUTOFF}.csv"
PLAPT_MODEL_PATH = "PLAPT/models/affinity_predictor.onnx"

# SQLite predictions database
SQLite_PRED_FILE = "db/virsentai.sqlite3"

PREDICTION_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY,
    prediction_date DATE,
    prediction_score REAL,
    model_id SMALLINT,
    virus_id TEXT,
    virus_name TEXT,
    virus_host TEXT,
    virus_db TEXT,
    created_at TIMESTAMP
);
"""
MODELS_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS models (
    model_id SMALLINT PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_description TEXT,
    model_train_date DATE,
    model_accuracy REAL,
    model_auroc REAL, 
    created_at TIMESTAMP 
);
"""

DRUGS_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS chembl_approved_drug (
    drug_chembl_id TEXT PRIMARY KEY, 
    pref_name TEXT, 
    canonical_smiles TEXT, 
    MW REAL,
    created_at DATE
);
"""

PLAPT_TABLE_QUERY ="""
CREATE TABLE IF NOT EXISTS PLAPT_AE (
    AE_id INTEGER PRIMARY KEY, 
    created_at TIMESTAMP, 
    virus_id TEXT, 
    protein_id TEXT, 
    drug_ID TEXT, 
    neg_log10_affinity_M REAL,
    FOREIGN KEY (drug_ID) REFERENCES chembl_approved_drug(drug_chembl_id)
);
"""