#!/usr/bin/env python3
"""
VirusHostDB Data to SQLite Loader

Loads VirusHostDB virus data from TSV into the SQLite database.
Imports records in batches of 1000 for performance.

Usage:
    python SQL-load_virushostdb_from_TSV.py

Requirements:
    - raw-data/VirusHostDB/VirHostDB_raw.tsv (from process_data.py)
    - config.py with SQLITE_VIRUSES_FILE and SQLITE_VIRUSES_SCHEMA

Input:
    - raw-data/VirusHostDB/VirHostDB_raw.tsv

Output:
    - db/raw-viruses.sqlite3 (viruses table)

Notes:
    - Uses batch inserts (1000 records per batch)
    - Increases CSV field size limit to 10MB for large sequences
    - Appends to existing data
"""

import sqlite3
import csv
import os
import sys
import time
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import RAW_DATA_VirusHostDB_PATH, SQLITE_VIRUSES_FILE, SQLITE_VIRUSES_SCHEMA, LOG_PATH

def setup_logging(script_name):
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_PATH, f"VHDB_{script_name}_{timestamp}.log")
    
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

def main():
    setup_logging("load_to_sqlite")
    
    # Increase CSV field size limit for large genomic sequences
    csv.field_size_limit(10 * 1024 * 1024)
    
    tsv_path = os.path.join(RAW_DATA_VirusHostDB_PATH, "VirHostDB_raw.tsv")
    
    if not os.path.exists(tsv_path):
        print(f"Error: Unified TSV not found at {tsv_path}")
        return

    print(f"Connecting to database: {SQLITE_VIRUSES_FILE}")
    conn = sqlite3.connect(SQLITE_VIRUSES_FILE)
    cursor = conn.cursor()

    # Ensure table exists
    cursor.executescript(SQLITE_VIRUSES_SCHEMA)
    
    print(f"Loading data from {tsv_path}...")
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        # Prepare insert statement
        columns = ["db_id", "accession", "sequence", "length", "source", "collection_date", "country", "host", "organism", "segment", "completeness"]
        placeholders = ", ".join(["?" for _ in columns])
        sql = f"INSERT INTO viruses ({', '.join(columns)}) VALUES ({placeholders})"
        
        count = 0
        batch = []
        for row in reader:
            data = tuple(row[col] for col in columns)
            batch.append(data)
            count += 1
            
            if len(batch) >= 1000:
                cursor.executemany(sql, batch)
                conn.commit()
                batch = []
                print(f"Inserted {count} records...")
        
        if batch:
            cursor.executemany(sql, batch)
            conn.commit()
            
    print(f"Finished loading VirusHostDB data. Total records: {count}")
    conn.close()

if __name__ == "__main__":
    main()
