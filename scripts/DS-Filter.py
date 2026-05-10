#!/usr/bin/env python3
"""
Virus Data Filter

Extracts and filters virus data from SQLite database to create classified TSV datasets.

Usage:
    python DS-Filter.py

Input:
    - db/db-viruses.sqlite3 (cleaned viruses table)

Output:
    - ds/ds_160k.tsv: All filtered rows (length <= 160,000)
    - ds/ds_160k_Human.tsv: Only label 1 (Homo sapiens)
    - ds/ds_160k_nonHuman.tsv: Only label 0 (other hosts)
    - ds/ds_160k_UNK.tsv: Only label -1 (Unknown)

Label Classification:
    - label 1: Host is "Homo sapiens" (human)
    - label -1: Host is "Unknown"
    - label 0: All other hosts (non-human, known)

Output Columns:
    id, db_id, accession, sequence, length, source, collection_date,
    country, host, organism, segment, completeness, created_at, label
"""

import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime

from config import *

def setup_logging(script_name):
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_PATH, f"{script_name}_{timestamp}.log")

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

def classify_host(host):
    if pd.isna(host) or host == '':
        return 0
    if host == 'Homo sapiens':
        return 1
    if host == 'Unknown':
        return -1
    return 0

def main():
    setup_logging("DS-Filter")

    conn = sqlite3.connect(SQLITE_CORR_VIRUSES_FILE)
    df = pd.read_sql_query('SELECT * FROM viruses', conn)
    conn.close()

    print('Step 1 - Original DataFrame:')
    print(f'Number of rows: {len(df)}')
    print(f'Number of columns: {len(df.columns)}')
    print(f'Header columns: {list(df.columns)}')
    print()

    filtered_df = df[df['length'] <= 160000].copy()

    print('Step 2 - Filtered DataFrame (length <= 160,000 bases):')
    print(f'Number of rows: {len(filtered_df)}')
    print(f'Number of columns: {len(filtered_df.columns)}')
    print(f'Header columns: {list(filtered_df.columns)}')
    print()

    filtered_df['label'] = filtered_df['host'].apply(classify_host)

    class_stats = filtered_df['label'].value_counts().sort_index()
    print('Step 2b - label Statistics:')
    print(f'label 1 (Homo sapiens): {class_stats.get(1, 0)}')
    print(f'label -1 (Unknown): {class_stats.get(-1, 0)}')
    print(f'label 0 (Other): {class_stats.get(0, 0)}')
    print(f'Total: {len(filtered_df)}')
    print()

    filtered_df.to_csv(DS_160K_FILE, sep='\t', index=False)
    print('Step 3 - Saved to DS_160K_FILE')

    human_df = filtered_df[filtered_df['label'] == 1]
    human_df.to_csv(DS_HUMAN_FILE, sep='\t', index=False)
    print(f'Step 4 - Saved {len(human_df)} rows to DS_HUMAN_FILE')

    nonhuman_df = filtered_df[filtered_df['label'] == 0]
    nonhuman_df.to_csv(DS_NONHUMAN_FILE, sep='\t', index=False)
    print(f'Step 5 - Saved {len(nonhuman_df)} rows to DS_NONHUMAN_FILE')

    unk_df = filtered_df[filtered_df['label'] == -1]
    unk_df.to_csv(DS_UNK_FILE, sep='\t', index=False)
    print(f'Step 6 - Saved {len(unk_df)} rows to DS_UNK_FILE')

    print('Done!')

if __name__ == "__main__":
    main()