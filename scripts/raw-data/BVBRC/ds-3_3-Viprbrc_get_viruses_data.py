#!/usr/bin/env python3
"""
BV-BRC Virus Data Downloader

Fetches metadata and nucleotide sequences for all viral genomes from BV-BRC API.
Combines genome and sequence endpoints to get complete data including genomic sequences.

Usage:
    python raw-data/BVBRC/ds-3_3-Viprbrc_get_viruses_data.py

Requirements:
    - requests library
    - tqdm library for progress bar
    - all_hosts_viruses_metadata_noSeq_filtered.csv (from ds-3_2)
    - config.py with RAW_DATA_BVBRC_PATH and LOG_PATH

Input:
    - raw-data/BVBRC/all_hosts_viruses_metadata_noSeq_filtered.csv

Output:
    - raw-data/BVBRC/Viprbrc_all_hosts_viruses_with_seqs.tsv

Notes:
    - Processes in batches of 20 virus IDs (API limit is 25)
    - Includes 0.2s delay between API calls
    - Logs to logs/ds-3_Viprbrc_get_viruses_data.log
"""

import requests
import csv
import os
import time
import json
from tqdm import tqdm
from datetime import datetime
from config import RAW_DATA_BVBRC_PATH, LOG_PATH

# --- Configuration ---
GENOME_API_URL = "https://www.bv-brc.org/api/genome"
SEQUENCE_API_URL = "https://www.bv-brc.org/api/genome_sequence"
OUTPUT_DIR = RAW_DATA_BVBRC_PATH
INPUT_CSV  = os.path.join(OUTPUT_DIR, "all_hosts_viruses_metadata_noSeq_filtered.csv")
OUTPUT_TSV = os.path.join(OUTPUT_DIR, "Viprbrc_all_hosts_viruses_with_seqs.tsv")
LOG_FILE   = os.path.join(LOG_PATH, "ds-3_Viprbrc_get_viruses_data.log")
BATCH_SIZE = 20 # there is a limit of 25 from the server, but we use 20 to be safe

# --- Logging setup ---
def log_message(msg, print_to_screen=False):
    if print_to_screen:
        print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as logf:
        logf.write(f"{msg}\n")

def fetch_sequences_batch(virus_ids):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/rqlquery+x-www-form-urlencoded"
    }
    query_ids = ",".join(virus_ids)
    query = f"in(genome_id,({query_ids}))&select(genome_id,sequence)"
    try:
        response = requests.post(SEQUENCE_API_URL, data=query, headers=headers, timeout=(30, 180))
        response.raise_for_status()
        time.sleep(0.2)  # Wait for 0.5 seconds
        return response.json()
    except Exception as e:
        log_message(f"Error fetching sequences batch: {e}", print_to_screen=True)
        return []
    
# --- Helper functions ---
def fetch_batch(virus_ids):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/rqlquery+x-www-form-urlencoded"
    }
    query_ids = ",".join(virus_ids)
    query = f"in(genome_id,({query_ids}))&select(genome_id,genome_name,host_name,genome_length,genbank_accessions,sequence)"
    try:
        response = requests.post(GENOME_API_URL, data=query, headers=headers, timeout=(30, 120))
        response.raise_for_status()

        time.sleep(0.2)  # Wait for 0.5 seconds
        return response.json()
    except Exception as e:
        log_message(f"Error fetching metadata batch: {e}", print_to_screen=True)
        return []

# --- Main script ---
def main():
    start_time = time.time()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Clear and initialize log file
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Log started at {datetime.now()}\n")

    # Step 1: Read virus IDs from CSV
    virus_ids = []
    try:
        with open(INPUT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                virus_ids.append(row["Virus_ID"].strip())
    except Exception as e:
        log_message(f"Error reading input CSV: {e}", print_to_screen=True)
        return

    if not virus_ids:
        log_message("No Virus_IDs found. Exiting.", print_to_screen=True)
        return

    log_message(f"Found {len(virus_ids)} Virus_IDs to process.")

    # limit the number of virus_ids to process
    # virus_ids = virus_ids[:40]  # For testing, limit to first 100 IDs

    results = []
    print(f"Processing {len(virus_ids)} virus IDs in batches of {BATCH_SIZE}...")
    for i in tqdm(range(0, len(virus_ids), BATCH_SIZE), desc="Processing virus batches"):
        batch_ids = virus_ids[i:i + BATCH_SIZE]
        metadata_batch = fetch_batch(batch_ids)
        metadata_dict = {item['genome_id']: item for item in metadata_batch}
        sequence_batch = fetch_sequences_batch(batch_ids)
        sequence_dict = {item['genome_id']: item.get("sequence", "N/A") for item in sequence_batch}
        for vid in batch_ids:
            item = metadata_dict.get(vid, {})
            seq = sequence_dict.get(vid, {})
            # Extract and clean fields
            host = item.get("host_name", "N/A")
            if isinstance(host, list):
                host = host[0] if host else "N/A"

            acc = item.get("genbank_accessions", "N/A")
            if isinstance(acc, list):
                acc = ";".join(acc)

            processed = {
                "Virus_ID": vid,
                "Genome_Name": item.get("genome_name", "N/A"),
                "Host_Name": host,
                "Genome_Length": item.get("genome_length", "N/A"),
                "Genbank_Accessions": acc,
                "Sequence": seq.upper() if seq else "N/A"
            }
            
            # log_message(json.dumps(processed, indent=2))  # Log to file only
            results.append(processed)

    # Step 3: Write to TSV
    try:
        with open(OUTPUT_TSV, 'w', newline='', encoding='utf-8') as tsvfile:
            fieldnames = ["Virus_ID", "Genome_Name", "Host_Name", "Genome_Length", "Genbank_Accessions", "Sequence"]
            writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        log_message(f"\nSuccessfully wrote {len(results)} records to {OUTPUT_TSV}", print_to_screen=True)
    except Exception as e:
        log_message(f"Error writing TSV file: {e}", print_to_screen=True)

    total_time = time.time() - start_time
    log_message(f"\nTotal execution time: {total_time:.2f} seconds", print_to_screen=True)

if __name__ == "__main__":
    main()

