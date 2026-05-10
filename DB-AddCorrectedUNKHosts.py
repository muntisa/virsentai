#!/usr/bin/env python3
"""
Add Corrected Unknown Hosts

Updates unknown host values in db/db-viruses.sqlite3 using LLM-corrected TSV file.

Usage:
    python DB-AddCorrectedUNKHosts.py

Input:
    - db/db-viruses.sqlite3 (cleaned viruses table)
    - db/UNK_LLM_hosts.tsv (with 'Organism' and 'Host' columns)

Output:
    - Modified db/db-viruses.sqlite3 (updated host values)

Process:
    1. Read organism-host pairs from UNK_LLM_hosts.tsv
    2. Connect to db-viruses.sqlite3
    3. For records where host is "Unknown", look up organism in TSV
    4. Replace "Unknown" with corrected host value from TSV
    5. Print number of corrected rows

Notes:
    - Only updates records where host = "Unknown"
    - Matches on organism field
    - Logs to logs/Process_partialHosts_<timestamp>.log
"""

import os
import sys
import csv
import sqlite3
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from config import *

def setup_logging(script_name):
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_PATH, f"Process_{script_name}_{timestamp}.log")

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
    setup_logging("DB-AddCorrectedUNKHosts")

    # Step 1: Read TSV file with organism-host pairs
    tsv_path = UNK_LLM_HOSTS_FILE
    if not os.path.exists(tsv_path):
        print(f"TSV file not found: {tsv_path}")
        return

    organism_host_map = {}
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            organism = row.get('Organism', '').strip()
            host = row.get('Host', '').strip()
            if organism and host and host != 'Unknown':
                organism_host_map[organism] = host

    print(f"Loaded {len(organism_host_map)} organism-host pairs from {tsv_path}")

    # Step 2: Connect to database
    db_path = SQLITE_CORR_VIRUSES_FILE
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Step 3: Find records with Unknown host
    cursor.execute("SELECT COUNT(*) FROM viruses WHERE host = 'Unknown'")
    unknown_count = cursor.fetchone()[0]
    print(f"Total records with Unknown host: {unknown_count}")

    # Step 4: Update records
    corrected = 0
    for organism, new_host in organism_host_map.items():
        cursor.execute(
            "UPDATE viruses SET host = ? WHERE organism = ? AND host = 'Unknown'",
            (new_host, organism)
        )
        if cursor.rowcount > 0:
            print(f"Corrected: {organism} --> {new_host}")
        corrected += cursor.rowcount

    conn.commit()

    print(f"Number of corrected rows: {corrected}")

    # Verify remaining Unknown records
    cursor.execute("SELECT COUNT(*) FROM viruses WHERE host = 'Unknown'")
    remaining_unknown = cursor.fetchone()[0]
    print(f"Remaining records with Unknown host: {remaining_unknown}")

    conn.close()
    print("\nDone!")

if __name__ == '__main__':
    main()
