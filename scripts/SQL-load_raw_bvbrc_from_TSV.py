#!/usr/bin/env python3
"""
BV-BRC Data to SQLite Loader

Loads BV-BRC (VIPR) virus data from TSV into the SQLite database.
Filters out rows with missing accessions or sequences before import.

Usage:
    python SQL-load_raw_bvbrc_from_TSV.py

Requirements:
    - pandas library
    - raw-data/BVBRC/Viprbrc_all_hosts_viruses_with_seqs.tsv (from ds-3_3)
    - config.py with SQLITE_VIRUSES_FILE and SQLITE_VIRUSES_SCHEMA

Input:
    - raw-data/BVBRC/Viprbrc_all_hosts_viruses_with_seqs.tsv

Output:
    - db/raw-viruses.sqlite3 (viruses table)

Column Mapping:
    - Virus_ID -> db_id (format: BVBRC:{Virus_ID})
    - Genbank_Accessions -> accession
    - Sequence -> sequence
    - Genome_Length -> length
    - Host_Name -> host
    - Genome_Name -> organism

Import Rules:
    - Skip rows where Genbank_Accessions is empty/missing
    - Skip rows where Sequence is empty/missing
    - Set completeness = "complete"
    - Set source = "BVBRC"
"""

import os
import sys
import sqlite3
import pandas as pd
import time
from datetime import datetime

# Add root to path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from config import SQLITE_VIRUSES_FILE, SQLITE_VIRUSES_SCHEMA, RAW_DATA_BVBRC_PATH, LOG_PATH
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
    RAW_DATA_BVBRC_PATH = "raw-data/BVBRC"
    LOG_PATH = "logs"

def setup_logging(script_name):
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
            if not message: return
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
        def flush(self): pass

    sys.stdout = Logger()
    sys.stderr = sys.stdout
    print(f"Logging started: {log_file}")
    return log_file

def load_tsv_to_sqlite():
    script_start_time = time.time()
    setup_logging("load_bvbrc")
    
    tsv_file = os.path.join(RAW_DATA_BVBRC_PATH, "Viprbrc_all_hosts_viruses_with_seqs.tsv")
    
    if not os.path.exists(tsv_file):
        print(f"Error: TSV file {tsv_file} not found.")
        return

    db_dir = os.path.dirname(SQLITE_VIRUSES_FILE)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)

    print(f"Connecting to database: {SQLITE_VIRUSES_FILE}")
    try:
        conn = sqlite3.connect(SQLITE_VIRUSES_FILE)
        cursor = conn.cursor()

        print("Ensuring table 'viruses' exists...")
        cursor.executescript(SQLITE_VIRUSES_SCHEMA)
        conn.commit()

        print(f"Reading data from {tsv_file}...")
        df = pd.read_csv(tsv_file, sep='\t', low_memory=False)
        
        initial_count = len(df)
        print(f"Read {initial_count} records from TSV.")

        print("Filtering: removing rows with missing Genbank_Accessions or Sequence...")
        df = df.dropna(subset=['Genbank_Accessions', 'Sequence'])
        df = df[df['Genbank_Accessions'].astype(str).str.strip() != '']
        df = df[df['Sequence'].astype(str).str.strip() != '']
        
        filtered_count = len(df)
        skipped = initial_count - filtered_count
        print(f"Filtered out {skipped} rows. Records to import: {filtered_count}")

        print("Mapping columns to database schema...")
        df['db_id'] = 'BVBRC:' + df['Virus_ID'].astype(str)
        df['accession'] = df['Genbank_Accessions']
        df['sequence'] = df['Sequence']
        df['length'] = df['Genome_Length']
        df['source'] = 'BVBRC'
        df['collection_date'] = None
        df['country'] = None
        df['host'] = df['Host_Name']
        df['organism'] = df['Genome_Name']
        df['segment'] = None
        df['completeness'] = 'complete'

        df = df[["db_id", "accession", "sequence", "length", "source", 
                "collection_date", "country", "host", "organism", "segment", "completeness"]]

        print(f"Appending records to 'viruses' table...")
        df.to_sql('viruses', conn, if_exists='append', index=False)
        conn.commit()
        
        cursor.execute("SELECT COUNT(*) FROM viruses")
        total_rows = cursor.fetchone()[0]
        
        print(f"Successfully loaded data.")
        print(f"Records added in this run: {filtered_count}")
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
