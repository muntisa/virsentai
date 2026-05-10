#!/usr/bin/env python3
"""
Drug Repurposing Pipeline

Uses the PLAPT (Protein-Ligand Affinity Prediction) model to identify
potential drug repurposing candidates for viruses with high zoonotic
probability predicted by VirSeNtAI.

Usage:
    python REP-drug_repurposing.py

Workflow:
    1. Connect to SQLite database and ensure tables exist
    2. Sync approved drugs from TSV file to 'chembl_approved_drug' table
    3. Load the PLAPT model (protein & ligand encoders + ONNX predictor)
    4. Fetch viruses from 'predictions' table with score >= 0.8
    5. For each virus:
       - Check if already processed (skip if exists in PLAPT_AE)
       - Extract viral proteins from NCBI using Entrez API
       - Calculate affinity against all approved drugs
       - Save high-affinity hits (>= 0.9) to 'PLAPT_AE' table
    6. Export all results to CSV file

Input:
    - approved_drugs_200_500_MW.tsv (from REP-GetApprovedDrugs.py)
    - db/virsentai.sqlite3 (with predictions table)
    - PLAPT/models/affinity_predictor.onnx

Output:
    - virsentai_PLAPT_AE_*.csv: All PLAPT_AE records exported

Configuration:
    - VIRSENTAI_PROB_CUTOFF: 0.8 (minimum prediction score)
    - PLAPT_AFFINITY_CUTOFF: 0.9 (minimum affinity)

Requirements:
    - biopython, torch, onnxruntime, xplapt
    - ChEMBL drugs TSV file
    - PLAPT ONNX model

Notes:
    - Skips already processed viruses
    - Uses Entrez API for protein extraction
    - Logs to logs/AE_<timestamp>.log
"""

import os
import sqlite3
import pandas as pd
import logging
import sys
import time
from datetime import datetime

# --- Biopython for NCBI ---
from Bio import Entrez
from Bio import SeqIO

# --- PLAPT for Affinity Calculation ---
import torch
import onnxruntime
from PLAPT.plapt import Plapt

from config import *

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================

# --- Database & File Paths ---
DB_FILE = SQLite_PRED_FILE
DRUGS_CSV = DRUG_OUTPUT_FILE.format(min_mw=DEFAULT_MIN_MW, max_mw=DEFAULT_MAX_MW)
FINAL_CSV_EXPORT = PLAPT_EXPORT_FILE
PLAPT_MODEL_PATH = PLAPT_MODEL_PATH

# --- NCBI Configuration ---
ENTREZ_EMAIL = NCBI_ENTREZ_EMAIL
Entrez.email = ENTREZ_EMAIL

# --- Pipeline Cutoffs ---

# --- Timestamp for Log File ---
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"AE-{TIMESTAMP}.log")

# ==============================================================================
# --- 2. LOGGING SETUP (To Console & File) ---
# ==============================================================================

# Set default logging level to ERROR for libraries (e.g., ONNX)
# We will set our own logger to INFO
logging.basicConfig(level=logging.ERROR)
onnxruntime.set_default_logger_severity(3) # Suppress ONNX warnings

# Create our specific logger
logger = logging.getLogger("DrugRepurposing")
logger.setLevel(logging.INFO)

# 2a. File Handler (saves to LOG_FILE)
file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 2b. Stream Handler (prints to console)
stream_handler = logging.StreamHandler(sys.stdout)
stream_formatter = logging.Formatter('%(message)s') # Simpler format for console
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)

logger.info(f"Logging configured. Saving logs to {LOG_FILE}")

# ==============================================================================
# --- 3. NCBI PROTEIN FETCHING ---
# ==============================================================================

NCBI_DELAY = 0.5  # Delay between NCBI API requests (seconds)


def extract_virus_proteins(nucleotide_id, max_retries=5, base_delay=5):
    """
    Extracts all protein sequences linked to an NCBI Nucleotide ID.
    
    Uses the Entrez API to:
        1. Find linked protein IDs using elink
        2. Fetch protein sequences using efetch
    
    Features:
        - Retry logic with exponential backoff
        - Uses NCBI API key for higher rate limits
        - Graceful error handling
    
    Args:
        nucleotide_id (str): NCBI Nucleotide accession (e.g., 'NC_022919')
        max_retries (int): Maximum retry attempts on failure (default: 5)
        base_delay (int): Base delay between retries in seconds (default: 5)
    
    Returns:
        list: List of dicts with keys: Protein_ID, Name, Sequence
              Returns empty list if no proteins found or on error.
    
    NCBI Rate Limits:
        - Without API key: 3 requests/second
        - With API key: 10 requests/second
    """
    """
    Extracts all protein sequences linked to an NCBI Nucleotide ID.
    Uses retry logic with exponential backoff and API key for higher rate limits.
    """
    linked_protein_ids = []
    protein_data = []

    def ncbi_request(request_func, *args, **kwargs):
        """Retry wrapper for NCBI requests with exponential backoff."""
        for attempt in range(max_retries):
            try:
                time.sleep(NCBI_DELAY)
                return request_func(*args, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"   [NCBI] Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
                    time.sleep(delay)
                else:
                    raise

    logger.info(f"   [NCBI] Step 1: Searching for linked protein IDs for {nucleotide_id}...")
    try:
        handle = ncbi_request(
            Entrez.elink,
            dbfrom="nuccore", db="protein", id=nucleotide_id, 
            rettype="fasta", api_key=NCBI_API_KEY if NCBI_API_KEY else None
        )
        record = Entrez.read(handle)
        handle.close()

        if record and record[0].get("LinkSetDb"):
            linked_proteins = record[0]["LinkSetDb"][0]["Link"]
            linked_protein_ids = [link["Id"] for link in linked_proteins]

        logger.info(f"   [NCBI]  -> Found {len(linked_protein_ids)} linked protein IDs.")

    except Exception as e:
        logger.error(f"   [NCBI] Error during Entrez.elink for ID {nucleotide_id}: {e}")
        return []

    if not linked_protein_ids:
        logger.warning(f"   [NCBI] No linked protein IDs found for {nucleotide_id}.")
        return []

    logger.info("   [NCBI] Step 2: Fetching protein sequences in FASTA format...")
    try:
        ids_string = ",".join(linked_protein_ids)
        handle = ncbi_request(
            Entrez.efetch,
            db="protein", id=ids_string, rettype="fasta", retmode="text",
            api_key=NCBI_API_KEY if NCBI_API_KEY else None
        )

        for seq_record in SeqIO.parse(handle, "fasta"):
            protein_id = seq_record.id.split("|")[1] if "|" in seq_record.id else seq_record.id
            protein_name = seq_record.description.split(seq_record.id)[-1].strip()

            protein_data.append({
                "Protein_ID": protein_id,
                "Name": protein_name,
                "Sequence": str(seq_record.seq)
            })

        handle.close()
        logger.info(f"   [NCBI]  -> Successfully extracted {len(protein_data)} protein sequences.")

    except Exception as e:
        logger.error(f"   [NCBI] Error during Entrez.efetch for proteins: {e}")
        return []

    return protein_data

# ==============================================================================
# --- 4. DATABASE SETUP FUNCTIONS ---
# ==============================================================================

def create_db_connection(db_file):
    """
    Create a database connection to the SQLite database.
    
    Args:
        db_file (str): Path to SQLite database file
    
    Returns:
        sqlite3.Connection: Database connection object, or None on error
    """
    """ Create a database connection to the SQLite database """
    logger.info(f"Connecting to database: {db_file}")
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        logger.info(f"SQLite connection successful (Version: {sqlite3.sqlite_version})")
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
    return conn

def create_tables(conn):
    """
    Create required tables if they don't exist.
    
    Tables created:
        - chembl_approved_drug: Stores ChEMBL-approved drug information
        - PLAPT_AE: Stores affinity calculation results
    
    Uses schema definitions from config.py:
        - DRUGS_TABLE_QUERY
        - PLAPT_TABLE_QUERY
    """
    logger.info("Checking/Creating database tables...")
    try:
        cursor = conn.cursor()
        
        # --- chembl_approved_drug Table ---
        # Note: We use TEXT PRIMARY KEY for drug_chembl_id as it's like 'CHEMBL123'
        cursor.execute(DRUGS_TABLE_QUERY)
        
        # --- PLAPT_AE Table ---
        # Note: AE_id is INTEGER PRIMARY KEY, which is an alias for LONG INTEGER in SQLite
        cursor.execute(PLAPT_TABLE_QUERY)
        
        conn.commit()
        logger.info("Tables 'chembl_approved_drug' and 'PLAPT_AE' are ready.")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")


def sync_drugs_to_db(conn, drug_csv_path):
    """
    Reads approved drugs from TSV and inserts new drugs into database.
    
    This function is idempotent - it skips drugs that already exist.
    
    Args:
        conn (sqlite3.Connection): Database connection
        drug_csv_path (str): Path to TSV file with drug data
    
    Returns:
        DataFrame: DataFrame of loaded drugs, or None on error
    
    TSV Expected Columns:
        - molecule_chembl_id: ChEMBL ID (e.g., 'CHEMBL123')
        - pref_name: Drug name
        - canonical_smiles: SMILES string
        - MW: Molecular weight
    """
    logger.info(f"Loading approved drugs from {drug_csv_path}...")
    try:
        df_drugs = pd.read_csv(drug_csv_path, sep="\t")
        # Ensure correct datatypes, especially for MW
        df_drugs['MW'] = pd.to_numeric(df_drugs['MW'], errors='coerce')
        # Drop any drugs that failed to load properly
        df_drugs = df_drugs.dropna(subset=['molecule_chembl_id', 'canonical_smiles', 'MW'])
        logger.info(f"Loaded {len(df_drugs)} drugs from CSV.")

        logger.info("Syncing drugs to 'chembl_approved_drug' table (skipping existing)...")
        cursor = conn.cursor()
        
        insert_count = 0
        # Get the current date *once* for this sync operation
        current_date = datetime.now().date()
        
        for row in df_drugs.itertuples():
            # Check if drug already exists
            cursor.execute("SELECT 1 FROM chembl_approved_drug WHERE drug_chembl_id = ?", (row.molecule_chembl_id,))
            if cursor.fetchone() is None:
                # Does not exist, insert it
                cursor.execute("""
                INSERT INTO chembl_approved_drug (drug_chembl_id, pref_name, canonical_smiles, MW, created_at)
                VALUES (?, ?, ?, ?, ?)
                """, (row.molecule_chembl_id, row.pref_name, row.canonical_smiles, row.MW, current_date))
                insert_count += 1
        
        conn.commit()
        logger.info(f"Drug sync complete. Added {insert_count} new drugs to the database.")
        return df_drugs
        
    except Exception as e:
        logger.error(f"Error loading or syncing drugs: {e}")
        return None

# ==============================================================================
# --- 5. PLAPT MODEL LOADING ---
# ==============================================================================

def load_plapt_model():
    """
    Initializes and returns the PLAPT model for drug-protein affinity prediction.
    
    PLAPT consists of:
        1. Protein Encoder: Rostlab/prot_bert (BERT for protein sequences)
        2. Ligand Encoder: seyonec/ChemBERTa-zinc-base-v1 (BERT for molecules)
        3. Prediction Module: ONNX model for affinity score regression
    
    The model is loaded once and reused for all predictions.
    
    Returns:
        Plapt: Initialized PLAPT model object, or None on error
    
    Model Configuration:
        - Device: CUDA if available, else CPU
        - Cache: Local disk cache for embeddings
        - Model path: Configured via PLAPT_MODEL_PATH in config.py
    """
    logger.info(f"Loading PLAPT model from {PLAPT_MODEL_PATH}...")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        logger.info("GPU is available. Loading model to CUDA.")
        logger.info(f"   CUDNN Version: {torch.backends.cudnn.version()}")
        logger.info(f"   Device Name: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        logger.info("GPU not available. Loading model to CPU.")
        device = "cpu"
        
    try:
        plapt_model = Plapt(prediction_module_path=PLAPT_MODEL_PATH, device=device)
        logger.info("PLAPT model loaded successfully.")
        return plapt_model
    except Exception as e:
        logger.error(f"Fatal error: Could not load PLAPT model: {e}")
        return None

# ==============================================================================
# --- 6. MAIN EXECUTION ---
# ==============================================================================

def main():
    """
    Main execution pipeline for drug repurposing analysis.
    
    Pipeline Steps:
        1. Connect to SQLite database and create tables if needed
        2. Load and sync approved drugs from TSV file
        3. Load PLAPT model (protein-ligand affinity predictor)
        4. Fetch viruses with prediction_score >= VIRSENTAI_PROB_CUTOFF
        5. Process each virus:
           - Skip if already processed (exists in PLAPT_AE)
           - Extract proteins from NCBI
           - Calculate drug affinities for each protein
           - Save high-affinity hits to PLAPT_AE table
        6. Export all results to CSV file
        7. Cleanup and close database connection
    
    Global Variables Used:
        - DB_FILE: SQLite database path
        - DRUGS_CSV: Approved drugs TSV path
        - FINAL_CSV_EXPORT: Output CSV path
        - PLAPT_MODEL_PATH: PLAPT model path
        - VIRSENTAI_PROB_CUTOFF: Minimum prediction score threshold
        - PLAPT_AFFINITY_CUTOFF: Minimum affinity for saving hits
    
    Output Files:
        - logs/AE-YYYYMMDD_HHMMSS.log: Execution log
        - virsentai_PLAPT_AE_*.csv: Exported PLAPT_AE records
    """
    logger.info("====== STARTING DRUG REPURPOSING PIPELINE ======")
    
    # --- Step 1: Connect to DB and Create Tables ---
    conn = create_db_connection(DB_FILE)
    if conn is None:
        return # Error already logged
    create_tables(conn)
    cursor = conn.cursor()

    # --- Step 2: Load and Sync Drugs ---
    df_drugs = sync_drugs_to_db(conn, DRUGS_CSV)
    if df_drugs is None or df_drugs.empty:
        logger.error("No drugs loaded. Exiting.")
        return

    # --- Step 3: Load PLAPT Model ---
    plapt_model = load_plapt_model()
    if plapt_model is None:
        return # Error already logged

    # --- Step 4: Get Target Viruses ---
    logger.info(f"Fetching target virus IDs (Score >= {VIRSENTAI_PROB_CUTOFF})...")
    try:
        cursor.execute("""
        SELECT DISTINCT virus_id 
        FROM predictions 
        WHERE prediction_score >= ?
        """, (VIRSENTAI_PROB_CUTOFF,))
        
        target_viruses = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found {len(target_viruses)} target viruses to process.")
        if not target_viruses:
            logger.info("No viruses met the cutoff. Exiting pipeline.")
            return

    except Exception as e:
        logger.error(f"Error fetching target viruses: {e}")
        return

    # --- Step 5: Main Processing Loop (Virus -> Protein -> Drugs) ---
    logger.info(f"* VIRUSES > 0.80\n{target_viruses}\nTOTAL={len(target_viruses)}")

    total_hits_saved = 0
    for i, virus_id in enumerate(target_viruses):
        cursor.execute("SELECT 1 FROM PLAPT_AE WHERE virus_id = ? LIMIT 1", (virus_id,))
        if cursor.fetchone():
            logger.info(f"[SKIP] {virus_id} already processed. Continuing to next virus.")
            continue
        
        logger.info(f"\n--- Processing Virus {i+1}/{len(target_viruses)}: {virus_id} ---")
        
        # 5a. Get Proteins for this Virus
        proteins = extract_virus_proteins(virus_id)
        logger.info(proteins)
        if not proteins:
            logger.warning(f"No proteins found for {virus_id}. Skipping.")
            continue
            
        logger.info(f"Found {len(proteins)} proteins for {virus_id}.")

        # 5b. Loop through each protein
        for protein in proteins:
            protein_id = protein['Protein_ID']
            protein_seq = protein['Sequence']
            logger.info(f"\n   Processing Protein: {protein_id} ({protein['Name'][:50]}...)")
            
            # 5c. Check for existing calculations to avoid re-work
            cursor.execute("""
            SELECT drug_ID FROM PLAPT_AE WHERE virus_id = ? AND protein_id = ?
            """, (virus_id, protein_id))
            existing_drug_ids = {row[0] for row in cursor.fetchall()}
            logger.info(f"   -> Found {len(existing_drug_ids)} existing calculations for this protein.")
            
            # 5d. Prepare drug batch (only drugs not yet calculated)
            drugs_to_process = []
            drug_id_map = {} # Map SMILES back to ChEMBL ID
            
            for row in df_drugs.itertuples():
                if row.molecule_chembl_id not in existing_drug_ids:
                    drugs_to_process.append(row.canonical_smiles)
                    drug_id_map[row.canonical_smiles] = row.molecule_chembl_id
            
            if not drugs_to_process:
                logger.info("   -> No new drugs to process for this protein. Skipping.")
                continue
                
            logger.info(f"   -> Calculating affinity for {len(drugs_to_process)} new drugs...")

            # 5e. Run PLAPT calculation (BATCH MODE)
            try:
                # score_candidates takes one protein and a list of molecules
                
                # NOTE: Removing the verbose log/print lines from the original file
                # to keep the log clean.
                # logger.info("** PLAPT - NO PROTEINS\n", len(protein_seq), "\n** NO DRUGS\n", len(drugs_to_process))
                # print(protein_seq)

                results = plapt_model.score_candidates(protein_seq, drugs_to_process)
                # 'results' is a list of dicts: [{'neg_log10_affinity_M': ...}, ...]

            except Exception as e:
                logger.error(f"   -> ERROR: PLAPT calculation failed for {protein_id}: {e}")
                continue # Skip to the next protein
                
            # --- START OF NEW FIX ---

            # 5f. Process PLAPT results and save high-affinity hits to DB
            logger.info(f"   -> Calculation complete. Processing {len(results)} scores...")

            # Safeguard: Check for a length mismatch, which would be a critical error
            if len(results) != len(drugs_to_process):
                logger.error(f"   -> CRITICAL: Mismatch in results length. Expected {len(drugs_to_process)}, got {len(results)}. Skipping this protein.")
                continue # Go to the next protein
            
            hits_found_this_batch = 0
            records_to_insert = []
            current_time = datetime.now()
            
            # We iterate in parallel using zip().
            # res_dict is a dictionary like {'neg_log10_affinity_M': 5.49}
            # smiles is a string from our 'drugs_to_process' list
            
            for res_dict, smiles in zip(results, drugs_to_process):
                try:
                    # 1. Get the affinity score from the dictionary
                    # Convert to float just in case it's a string or other numeric type
                    affinity_float = float(res_dict['neg_log10_affinity_M'])
                    
                    # 2. Now, compare the float value
                    if affinity_float >= PLAPT_AFFINITY_CUTOFF:
                        
                        # 3. Get the corresponding drug_ID from our map
                        drug_id = drug_id_map[smiles]
                        
                        # 4. Append the full record for database insertion
                        records_to_insert.append((
                            current_time, 
                            virus_id, 
                            protein_id, 
                            drug_id, 
                            affinity_float
                        ))
                        hits_found_this_batch += 1
                        
                except (ValueError, TypeError, KeyError) as e:
                    # This catches multiple potential errors:
                    # - KeyError: 'neg_log10_affinity_M' key doesn't exist in res_dict
                    # - ValueError: The value isn't a valid number
                    # - TypeError: The value is None or another invalid type
                    logger.warning(f"   -> Skipping invalid/missing affinity score for SMILES {smiles}. Error: {e}")
                    continue # Skip to the next drug
            
            # --- END OF NEW FIX ---
            
            # This part of the code remains the same and will now work correctly
            if records_to_insert:
                cursor.executemany("""
                INSERT INTO PLAPT_AE (created_at, virus_id, protein_id, drug_ID, neg_log10_affinity_M)
                VALUES (?, ?, ?, ?, ?)
                """, records_to_insert)
                
                conn.commit()
                logger.info(f"   -> SUCCESS: Saved {hits_found_this_batch} high-affinity hits to 'PLAPT_AE' table.")
                total_hits_saved += hits_found_this_batch
            else:
                logger.info("   -> No new hits found above the cutoff.")

    logger.info(f"\n--- Pipeline processing complete. Total new hits saved: {total_hits_saved} ---")

    # --- Step 6: Export all results to CSV ---
    logger.info(f"Exporting all records from 'PLAPT_AE' to {FINAL_CSV_EXPORT}...")
    try:
        df_export = pd.read_sql_query("SELECT * FROM PLAPT_AE", conn)
        df_export.to_csv(FINAL_CSV_EXPORT, index=False)
        logger.info(f"Successfully exported {len(df_export)} records.")
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")

    # --- Step 7: Cleanup ---
    conn.close()
    logger.info("Database connection closed.")
    logger.info("====== SCRIPT FINISHED ======")


if __name__ == "__main__":
    if ENTREZ_EMAIL == "your_email@example.com":
        logger.error("="*60)
        logger.error("ERROR: Please edit the script and set the 'ENTREZ_EMAIL' variable.")
        logger.error("NCBI requires a valid email to use their services.")
        logger.error("="*60)
    else:
        main()