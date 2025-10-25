# VirSentAI: Viral Sentry AI – Intelligent Zoonotic Surveillance Platform
# Configuration for all the scripts

DATASET_FILE = "dataset/df_max_160k.csv"
SEQ_MAX_LENGTH = 160_000
EVAL_BATCH_SIZE = 4 # decrease the value for less GPU

# Training
TRAINING_DIR  = "hyena_dna_training" # Where you are training the model
NEW_MODEL_DIR = "models/ver2"        # Where you save the final trained model
PREDICT_TEMP_SUBFOLDER  = "prediction_temp"

TOKEN_TRAIN_PATH = "tokenized_train_dataset" # tokenize dataset
TOKEN_VAL_PATH   = "tokenized_val_dataset"

# Virus scans
VIRUS_SCAN_DIR = "virus_scan"
ALL_PREDS_FILE  = "virsentai_all_predictions.tsv"
ALL_MODELS_FILE = "virsentai_all_models.tsv"

# SQLite db
SQLite_FILE = "db/virsentai.sqlite3"
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

# LOGS
LOG_DIR = "logs"

# Webapp
WEBAPP_DIR = "webapp"
JSON_FILE  = "webapp/summary_stats_from_db.json"
AUROC_CUTOFF = 0.80 # used to decide possible zoonotic viruses

# NCBI Entrez
NCBI_ENTREZ_EMAIL = "your_email@your_center.com"
NCBI_ENTREZ_MAX_RESULTS = 10000 # maximum API records from NCBI query
