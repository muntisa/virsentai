#!/usr/bin/env python3
"""
Remove Fake Virus Records

Removes fake/invalid virus records from db/db-viruses.sqlite3.
Identifies synthetic, engineered, or non-biological virus constructs.

Usage:
    python DB-RemoveFakeViruses.py

Input:
    - db/db-viruses.sqlite3 (cleaned viruses table)

Output:
    - Modified db/db-viruses.sqlite3 (fake records deleted)

Fake Virus Indicators (case-insensitive, searched in organism field):
    - recombinant: Engineered recombinants
    - vector: Expression vectors
    - construct: Laboratory constructs
    - synthetic: Artificial synthetic viruses
    - chimera: Chimeric constructs
    - pseudotype: Pseudotyped viruses
    - clone: Cloned sequences
    - cDNA: Complementary DNA constructs
    - GFP: Green fluorescent protein fusions
    - luciferase: Reporter gene fusions
    - wastewater: Environmental waste samples
    - metagenome: Uncultured metagenomic sequences
    - uncultured: Uncultured virus sequences
    - satRNA: Satellite RNA (not independent viruses)
"""

import os
import sys
import sqlite3
from datetime import datetime
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

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

def main():
    setup_logging("DB-RemoveFakeViruses")

    db_path = SQLITE_CORR_VIRUSES_FILE
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # List of substrings to search for (case-insensitive)
    substrings = [
        'recombinant', 'vector', 'construct', 'synthetic', 'chimera',
        'pseudotype', 'clone', 'cDNA', 'GFP', 'luciferase',
        'wastewater', 'metagenome', 'uncultured',
        'satRNA'
    ]

    # Build WHERE clause
    conditions = []
    for sub in substrings:
        conditions.append(f"UPPER(organism) LIKE UPPER('%{sub}%')")
    where_clause = " OR ".join(conditions)

    # Count total rows before deletion
    cursor.execute("SELECT COUNT(*) FROM viruses")
    total_before = cursor.fetchone()[0]
    print(f"Total rows before cleanup: {total_before}")

    # Count rows to be deleted
    query_count = f"SELECT COUNT(*) FROM viruses WHERE {where_clause}"
    cursor.execute(query_count)
    count_to_delete = cursor.fetchone()[0]
    print(f"Rows to delete (fake viruses): {count_to_delete}")

    # List sample organisms to be deleted
    cursor.execute(f"SELECT DISTINCT organism FROM viruses WHERE {where_clause} ORDER BY organism LIMIT 20")
    rows = cursor.fetchall()
    print("\nSample organisms to be deleted (up to 20):")
    for row in rows:
        print(f"  {row[0]}")

    # Delete the fake virus records
    if count_to_delete > 0:
        print(f"\nDeleting {count_to_delete} rows...")
        delete_query = f"DELETE FROM viruses WHERE {where_clause}"
        cursor.execute(delete_query)
        conn.commit()
        print("Deletion complete.")
    else:
        print("\nNo rows to delete.")

    # Count total rows after deletion
    cursor.execute("SELECT COUNT(*) FROM viruses")
    total_after = cursor.fetchone()[0]
    print(f"\nTotal rows after cleanup: {total_after}")
    print(f"Total rows deleted: {total_before - total_after}")

    conn.close()
    print("\nDone!")

if __name__ == '__main__':
    main()
