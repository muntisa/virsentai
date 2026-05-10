#!/usr/bin/env python3
"""
Step 1a: Scan RefSeq for New Viruses (Raw Data)

Scans the RefSeq nucleotide database for new virus sequences within a
specified date range and saves raw data without host correction.

Usage:
    python 01a_scan_viruses_raw_RefSeq.py --start-date 2026/04/16 --end-date 2026/05/04

Input:
    - --start-date: Start date in YYYY/MM/DD format (required)
    - --end-date: End date in YYYY/MM/DD format (required)

Output:
    - virus-scan/virus_scan_2026-04-16_to_2026-05-04_RefSeq_raw.tsv

Process:
    1. Search RefSeq Nucleotide database using Entrez
       - Query: complete viral genomes, RefSeq curated, date range
    2. Batch fetch records (200 per batch)
    3. Extract fields: Virus ID, Name, Sequence, Host, Country, Collection Date, Registration Date
    4. Filter out human hosts
    5. Remove already-scanned viruses from database

Notes:
    - Host "Unknown" means not available in RefSeq
    - Uses Bio.SeqIO for GenBank parsing
    - Logs to logs/01a_scan_viruses_raw_RefSeq_<timestamp>.log
"""
import os
import time
import csv
import logging
import sys
from datetime import datetime
from Bio import Entrez, SeqIO
import sqlite3
import argparse

from config import *

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------
parser = argparse.ArgumentParser(description="Scan RefSeq for new viruses - Raw data (no LLM)")
parser.add_argument("--start-date", required=True, help="Start date in YYYY/MM/DD format")
parser.add_argument("--end-date", required=True, help="End date in YYYY/MM/DD format")
args = parser.parse_args()

# Get dates from command line arguments
start_date = args.start_date
end_date   = args.end_date

# Reformat dates from YYYY/MM/DD to YYYY-MM-DD
start_date_formatted = datetime.strptime(start_date, "%Y/%m/%d").strftime("%Y-%m-%d")
end_date_formatted   = datetime.strptime(end_date,   "%Y/%m/%d").strftime("%Y-%m-%d")

# Output file and folder
output_folder = VIRUS_SCAN_DIR
output_file = os.path.join(output_folder, f"virus_scan_{start_date_formatted}_to_{end_date_formatted}_RefSeq_raw.tsv")

# Batch settings
BATCH_SIZE = NCBI_BATCH_SIZE
DELAY = NCBI_ENTREZ_DELAY

print(f"Scanning RefSeq viruses from {start_date} to {end_date}")

# --- Global Variables ---
Entrez.email = NCBI_ENTREZ_EMAIL

# Create Virus scan folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Logging setup ---
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(LOG_DIR, exist_ok=True)
log_file_name = os.path.join(LOG_DIR, f"{script_name}_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_name),
        logging.StreamHandler()
    ]
)

def safe_get(qualifiers, key):
    """Safely get a value from qualifiers dictionary."""
    return qualifiers.get(key, [""])[0] if qualifiers.get(key) else ""

def build_refseq_query(start_date, end_date):
    """Build the RefSeq query string."""
    return (
        'txid10239[Organism:exp]'
        ' AND NOT human[Host] '
        ' AND srcdb_refseq[PROP] '
        ' AND refseq[filter] '
        ' AND "complete genome"[Title] '
        ' NOT "partial"[Title]'
        ' NOT "Human"[Title]'
        ' NOT "human"[Title]'
        f' AND ("{start_date}"[PDAT] : "{end_date}"[PDAT])'
    )

def search_ids():
    """Search for RefSeq virus IDs."""
    query = build_refseq_query(start_date, end_date)
    logging.info(f"RefSeq Search term: {query}")
    
    handle = Entrez.esearch(
        db="nuccore",
        term=query,
        retmax=NCBI_ENTREZ_MAX_RESULTS
    )
    record = Entrez.read(handle)
    handle.close()
    
    id_list = record.get("IdList", [])
    logging.info(f"Number of RefSeq hits: {len(id_list)}")
    return id_list

def fetch_batches(ids):
    """Fetch records in batches for efficiency."""
    for i in range(0, len(ids), BATCH_SIZE):
        batch = ids[i:i + BATCH_SIZE]
        
        try:
            handle = Entrez.efetch(
                db="nuccore",
                id=",".join(batch),
                rettype="gb",
                retmode="text"
            )
            
            for record in SeqIO.parse(handle, "genbank"):
                yield record
            
            handle.close()
            time.sleep(DELAY)
            
        except Exception as e:
            logging.error(f"Batch error {i}: {e}")
            print(f"Batch error {i}: {e}", file=sys.stderr)

def extract_metadata(record):
    """Extract metadata from a GenBank record."""
    virus_id = record.id  # RefSeq accession (e.g., NC_XXXXXX)
    virus_name = record.annotations.get("organism", "")
    sequence = str(record.seq).upper()
    
    host = ""
    country = ""
    collection_date = ""
    
    # Extract from source feature qualifiers
    for feature in record.features:
        if feature.type == "source":
            qualifiers = feature.qualifiers
            host = safe_get(qualifiers, "host")
            country = safe_get(qualifiers, "country")
            collection_date = safe_get(qualifiers, "collection_date")
            break
    
    # Get registration date from LOCUS
    registration_date = ""
    if record.annotations.get("date"):
        try:
            registration_date = datetime.strptime(record.annotations.get("date"), "%d-%b-%Y").strftime("%Y-%m-%d")
        except:
            registration_date = record.annotations.get("date", "")
    
    return {
        "Virus_ID": virus_id,
        "Virus_Name": virus_name,
        "Virus_Seq": sequence,
        "Host": host if host else "Unknown",
        "Country": country,
        "Collection_Date": collection_date,
        "Registration_Date": registration_date,
        "Database": "RefSeq"
    }

def filter_viruses_by_db_id(viruses, sqlite_file_path):
    """
    Filters a list of virus dictionaries, removing those whose 'Virus_ID' 
    already exists in the 'predictions' table of the SQLite database.
    """
    existing_ids = set()
    filtered_viruses = []
    
    try:
        with sqlite3.connect(sqlite_file_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT virus_id FROM predictions")
            existing_ids = {row[0] for row in cursor.fetchall()}
        
        logging.info(f"Found {len(existing_ids)} existing Virus IDs in the database.")
        
    except sqlite3.OperationalError as e:
        logging.error(f"Database error: {e}")
        return viruses
    
    for virus in viruses:
        virus_id = virus.get("Virus_ID")
        if virus_id is not None and virus_id not in existing_ids:
            filtered_viruses.append(virus)
    
    logging.info(f"Filtering complete. {len(filtered_viruses)} unique viruses remain.")
    return filtered_viruses

def fetch_virus_data():
    """Main function to fetch virus data from RefSeq."""
    viruses = []
    
    # Search for IDs
    id_list = search_ids()
    
    if not id_list:
        logging.info("No new RefSeq virus sequences found matching the criteria. Exiting.")
        return viruses
    
    if len(id_list) >= NCBI_ENTREZ_MAX_RESULTS:
        logging.warning(f"Warning: Number of hits reached the maximum limit of {NCBI_ENTREZ_MAX_RESULTS}. Exiting.")
        return viruses
    
    # Fetch and process records
    no = 1
    for record in fetch_batches(id_list):
        logging.info(f"Processing {no}/{len(id_list)}: RefSeq ID {record.id}")
        no += 1
        
        # Extract metadata
        virus_data = extract_metadata(record)
        
        # Skip human hosts
        if "homo sapiens" in virus_data["Host"].lower() or "covid-19" in virus_data["Host"].lower():
            logging.info(f"Skipping record {virus_data['Virus_ID']} because host is filtered: '{virus_data['Host']}'.")
            continue
        
        # Skip if missing essential data
        if not virus_data["Virus_Name"] or not virus_data["Virus_Seq"] or not virus_data["Virus_ID"]:
            logging.info(f"Skipping record {virus_data.get('Virus_ID', 'unknown')}: missing essential data.")
            continue
        
        if not virus_data["Registration_Date"]:
            logging.info(f"Skipping record {virus_data['Virus_ID']}: missing registration date.")
            continue
        
        viruses.append(virus_data)
        logging.info(f"Processed: {virus_data['Virus_ID']}, Host: {virus_data['Host']}, Registration: {virus_data['Registration_Date']}")
    
    # Filter by database
    viruses = filter_viruses_by_db_id(viruses, SQLite_FILE)
    
    logging.info("\n--- Final Filtered List (Viruses to Insert) ---")
    if viruses:
        for virus in viruses:
            logging.info(f"ID: {virus['Virus_ID']}, Name: {virus['Virus_Name']}, Host: {virus['Host']}")
    else:
        logging.info("The filtered list is empty or an error occurred.")
    
    return viruses

if __name__ == "__main__":
    print(f"Scanning for RefSeq virus sequences from {start_date} to {end_date}...")
    
    if os.path.exists(output_file):
        print(f"Output file already exists: {output_file}. Scan will be skipped.")
    else:
        virus_data = fetch_virus_data()
        
        if virus_data:
            with open(output_file, "w", newline="", encoding="utf-8") as tsvfile:
                fieldnames = ["Virus_ID", "Virus_Name", "Virus_Seq", "Host", "Country", "Collection_Date", "Registration_Date", "Database"]
                writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerows(virus_data)
            print(f"Data saved to {output_file} with {len(virus_data)} records.")
        else:
            print("No data to save.")
    
    # Verification step
    if os.path.exists(output_file):
        print("\n--- Verification ---")
        print(f"Reading first 2 rows from {output_file}...")
        try:
            with open(output_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                header = next(reader)
                print(" | ".join(header))
                print("-" * 80)
                for i in range(2):
                    try:
                        row = next(reader)
                        row_to_log = row[:]
                        if len(row_to_log) > 2 and len(row_to_log[2]) > 50:
                            row_to_log[2] = row_to_log[2][:50] + "..."
                        print(" | ".join(row_to_log))
                    except StopIteration:
                        break
        except Exception as e:
            print(f"Error during verification: {e}")