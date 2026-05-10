#!/usr/bin/env python3
"""
BV-BRC Virus Metadata Fetcher

Fetches metadata for all complete viral genomes from BV-BRC API.
Retrieves genome_id, genome_name, host_name, genbank_accessions, and genome_length.
Does NOT fetch sequences - only metadata.

Usage:
    python raw-data/BVBRC/ds-3_1-Viprbrc_get_viruses_metadata.py

Requirements:
    - requests library
    - Internet connection to BV-BRC API
    - config.py with RAW_DATA_BVBRC_PATH

Output:
    - raw-data/BVBRC/all_hosts_viruses_metadata_noSeq.csv

Notes:
    - Filters for Viruses (superkingdom) and Complete genome status
    - Uses pagination with 10000 records per batch
    - Includes 0.5s delay between API calls to be polite
"""

import requests
import json
import os
import csv # For writing CSV files
import time # For delays between paginated requests
from config import RAW_DATA_BVBRC_PATH

# --- Configuration ---
# API endpoint for genome data
GENOME_API_URL = "https://www.bv-brc.org/api/genome"

# Directory to save the output CSV
OUTPUT_DIR = RAW_DATA_BVBRC_PATH
CSV_FILENAME = "all_hosts_viruses_metadata_noSeq.csv"

# --- Helper Functions ---

def fetch_all_virus_data_for_csv():
    """
    Fetches specified metadata (genome_id, host_name, genbank_accessions)
    for ALL viral genomes from BV-BRC using pagination.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              processed data for a virus. Returns an empty list on failure.
    """
    all_processed_results = []
    batch_size = 10000  # Number of records to fetch per API call (max is often 25000, but smaller is safer for web APIs)
    offset = 0
    total_fetched = 0

    # Fields to select. 'genbank_accessions' is the field name for Genbank Accession(s).
    select_fields = "genome_id,genome_name,host_name,genbank_accessions,genome_name,genome_length" # Added genome_name for context
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/rqlquery+x-www-form-urlencoded"
    }

    print("Starting to fetch all virus metadata. This may take a long time...")

    while True:
        # Construct the RQL query with pagination
        # We sort by genome_id to ensure consistent order if the process is interrupted and restarted,
        # though this script fetches all in one go.
        query = (
            f"limit({batch_size},{offset})&"
            f"select({select_fields})&"
            "eq(superkingdom,Viruses)&"
            "eq(genome_status,Complete)&"
            "sort(+genome_id)" # Sorting is good practice for large queries
        )
        
        print(f"Fetching records from offset {offset} with batch size {batch_size}...")
        # print(f"Current query: {query}") # Uncomment for debugging query string

        try:
            response = requests.post(GENOME_API_URL, data=query, headers=headers, timeout=(30, 120)) # (connect_timeout, read_timeout)
            response.raise_for_status()
            
            current_batch = response.json()

            if not current_batch: # No more data
                print("No more results from API. All data fetched.")
                break
            
            num_in_batch = len(current_batch)
            total_fetched += num_in_batch
            print(f"  Fetched {num_in_batch} records in this batch. Total fetched so far: {total_fetched}")

            for item in current_batch:
                # Process host_name: take the first if it's a list, otherwise use as is.
                host_name_raw = item.get("host_name")
                if isinstance(host_name_raw, list):
                    print(host_name_raw)
                    processed_host_name = host_name_raw[0] if host_name_raw else "N/A"
                elif host_name_raw: # If it's a non-empty string
                    processed_host_name = host_name_raw
                else:
                    processed_host_name = "N/A"
                    # Skip human viruses
                    # print(f"Skipping human virus: {item.get('genome_id')}")
                    continue

                # Process genbank_accessions: join list with semicolon if it's a list.
                accessions_raw = item.get("genbank_accessions")
                if isinstance(accessions_raw, list):
                    processed_accessions = ";".join(accessions_raw) if accessions_raw else "N/A"
                elif accessions_raw: # If it's a non-empty string (though less likely for this field)
                    processed_accessions = accessions_raw
                else:
                    processed_accessions = "N/A"
                
                processed_item = {
                    "Virus_ID": item.get("genome_id", "N/A"),
                    "Genome_Name": item.get("genome_name", "N/A"),
                    "Genbank_Accessions": processed_accessions,
                    "Genome_Length": item.get("genome_length", "N/A"),
                    "Host_Name": processed_host_name
                }
                all_processed_results.append(processed_item)
            
            if num_in_batch < batch_size: # This was the last page
                print(f"Fetched {num_in_batch} records, which is less than batch size {batch_size}. Assuming end of data.")
                break
            
            offset += batch_size
            time.sleep(0.5) # Be polite to the API: 0.5-second delay between requests

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data at offset {offset}: {e}")
            print("Stopping data fetching due to error.")
            # Optionally, you could return partial results or implement more robust retries here
            return all_processed_results # Return what has been fetched so far
        except json.JSONDecodeError:
            print(f"Error decoding JSON response at offset {offset}.")
            if 'response' in locals() and response:
                print(f"Response text (first 500 chars): {response.text[:500]}...")
            print("Stopping data fetching due to JSON error.")
            return all_processed_results

    print(f"Finished fetching all data. Total records processed: {len(all_processed_results)}")
    return all_processed_results

# --- Main Script Logic ---
if __name__ == "__main__":
    start_time = time.time()

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # 1. Fetch all virus metadata
    virus_metadata_list = fetch_all_virus_data_for_csv()

    if not virus_metadata_list:
        print("No virus metadata fetched. Exiting.")
    else:
        csv_file_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
        print(f"\nWriting {len(virus_metadata_list)} records to CSV file: {csv_file_path}")
        
        # Define the fieldnames for the CSV header
        # Ensure these keys match what's in the dictionaries in virus_metadata_list
        
        fieldnames = ["Virus_ID", "Genome_Name", "Genbank_Accessions", "Genome_Length", "Host_Name"]

        try:
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader() # Write the header row
                for virus_data in virus_metadata_list:
                    # Ensure only the selected fields are written, in case processed_item had extra keys
                    row_to_write = {field: virus_data.get(field, "N/A") for field in fieldnames}
                    writer.writerow(row_to_write)
            
            print(f"Successfully wrote data to {csv_file_path}")
            print(f"Total number of viruses written to {csv_file_path}: {len(virus_metadata_list)}")
        except IOError:
            print(f"Error writing to CSV file at {csv_file_path}. Check permissions or disk space.")
        except Exception as e:
            print(f"An unexpected error occurred while writing the CSV: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total running time: {elapsed_time:.2f} seconds")