#!/usr/bin/env python3
"""
NCBI Data to SQLite Loader

Loads NCBI/RefSeq virus data from TSV into the SQLite database.
Appends records to the 'viruses' table in the database.

Usage:
    python SQL_load_raw_ncbi_from_TSV.py

Requirements:
    - pandas library
    - raw-data/NCBI/NCBI_raw.tsv (from convert_to_tsv.py)
    - config.py with SQLITE_VIRUSES_FILE and SQLITE_VIRUSES_SCHEMA

Input:
    - raw-data/NCBI/NCBI_raw.tsv

Output:
    - db/raw-viruses.sqlite3 (viruses table)

Notes:
    - Appends to existing data (if_exists='append')
    - Drops 'id' column from TSV to let SQLite auto-increment
    - Requires database directory to exist
"""

import os
import sys
import sqlite3
import pandas as pd
import time
import io
from datetime import datetime

# Add root to path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from config import SQLITE_VIRUSES_FILE, SQLITE_VIRUSES_SCHEMA, RAW_DATA_NCBI_PATH, LOG_PATH
except ImportError:
    SQLITE_VIRUSES_FILE = "db/raw-viruses.sqlite3"
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
    RAW_DATA_NCBI_PATH = "raw-data/NCBI"
    LOG_PATH = "logs"

def setup_logging(script_name):
    """
    Sets up logging to both terminal and a file in the LOG_PATH.
    Matches the style used in other scripts.
    """
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_PATH, f"DB_{script_name}_{timestamp}.log")
    
    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(log_file, "w", encoding="utf-8")
            self.at_start_of_line = True

        def write(self, message):
            if not message:
                return
            
            processed_message = ""
            lines = message.splitlines(keepends=True)
            
            for line in lines:
                if self.at_start_of_line and line.strip():
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    processed_message += f"{timestamp} "
                
                processed_message += line
                self.at_start_of_line = line.endswith('\n')

            self.terminal.write(processed_message)
            self.log.write(processed_message)
            self.log.flush()

        def flush(self):
            pass

    sys.stdout = Logger()
    sys.stderr = sys.stdout
    print(f"Logging started: {log_file}")
    return log_file

def load_tsv_to_sqlite():
    """
    Reads the NCBI_raw.tsv and loads its content into the SQLite database.
    """
    script_start_time = time.time()
    setup_logging("load_ncbi")
    
    tsv_file = os.path.join(RAW_DATA_NCBI_PATH, "NCBI_raw.tsv")
    
    if not os.path.exists(tsv_file):
        print(f"Error: TSV file {tsv_file} not found.")
        return

    # Ensure database directory exists
    db_dir = os.path.dirname(SQLITE_VIRUSES_FILE)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)

    print(f"Connecting to database: {SQLITE_VIRUSES_FILE}")
    try:
        conn = sqlite3.connect(SQLITE_VIRUSES_FILE)
        cursor = conn.cursor()

        # Phase 1: Ensure Schema
        print("Ensuring table 'viruses' exists with proper schema...")
        cursor.executescript(SQLITE_VIRUSES_SCHEMA)
        conn.commit()

        # Phase 2: Read and Load Data
        print(f"Reading data from {tsv_file}...")
        
        # Load TSV using pandas
        # We specify low_memory=False for safety with large files
        df = pd.read_csv(tsv_file, sep='\t', low_memory=False)
        
        # The 'id' column in the TSV is a sequential counter from the script.
        # However, the database 'id' is PRIMARY KEY AUTOINCREMENT.
        # To avoid conflicts and let SQLite manage IDs (especially for subsequent appends),
        # we drop the 'id' column from the dataframe if it exists.
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
            
        initial_count = len(df)
        print(f"Read {initial_count} records from TSV.")

        # Phase 3: Insert into SQLite
        # Using to_sql with if_exists='append' to add to existing data
        print(f"Appending records to 'viruses' table...")
        df.to_sql('viruses', conn, if_exists='append', index=False)
        conn.commit()
        
        # Verification
        cursor.execute("SELECT COUNT(*) FROM viruses")
        total_rows = cursor.fetchone()[0]
        
        print(f"Successfully loaded data.")
        print(f"Records added in this run: {initial_count}")
        print(f"Total records in 'viruses' table: {total_rows}")

    except Exception as e:
        print(f"Error during database operation: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

    script_end_time = time.time()
    elapsed = script_end_time - script_start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

if __name__ == "__main__":
    load_tsv_to_sqlite()
