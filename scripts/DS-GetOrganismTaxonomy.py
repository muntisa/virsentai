#!/usr/bin/env python3
"""
Organism Taxonomy Fetcher

Fetches taxonomy data from NCBI for all organisms in the balanced dataset.

Usage:
    python DS-GetOrganismTaxonomy.py

Input:
    - fine-tuning/ds_160k_balanced.tsv

Output:
    - ds/ds_160k_balanced_taxonomy.tsv (all columns except sequence + taxonomy)
    - ds/ds_160k_balanced_taxonomy_incremental.tsv (backup - saved after each organism)

Taxonomy Columns Added:
    TaxID, Name, Rank, realm, kingdom, phylum, class, order, family, genus, species

Performance:
    - Uses NCBI API key for faster rate limits (10 req/sec vs 3 req/sec)
    - Delay between calls: 0.1s (with API key)
    - For 15,000 organisms: ~50 minutes (vs ~3 hours without key)

Requirements:
    - biopython
    - NCBI API key (recommended for speed)
    - config.py with NCBI_ENTREZ_EMAIL and NCBI_API_KEY

Notes:
    - Saves incremental backups after each organism
    - Includes retry logic for NCBI service
    - Logs to logs/ directory
"""

import os
import sys
import time
import logging
from datetime import datetime
import pandas as pd
import re
from Bio import Entrez
from config import *

Entrez.email = NCBI_ENTREZ_EMAIL
Entrez.api_key = NCBI_API_KEY
Entrez.tool = "DS-GetOrganismTaxonomy"

TAXONOMY_RANKS = ['realm', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

NCBI_MAX_RETRIES = 3
NCBI_RETRY_DELAY = 5

NCBI_DELAY_MIN = NCBI_ENTREZ_DELAY # 0.34
NCBI_DELAY_MAX = 1.0
NCBI_DELAY_INCREMENT = 0.05

consecutive_errors = 0
current_delay = NCBI_DELAY_MIN

def setup_logging():
    """Configure logging to both console and file."""
    os.makedirs(LOG_DIR, exist_ok=True)

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"{script_name}_{timestamp}.log")

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file

def get_ncbi_tax_with_retry(taxon, max_retries=NCBI_MAX_RETRIES):
    """
    Fetch taxonomy data for a single organism from NCBI Taxonomy database.
    Includes retry logic for handling NCBI service errors.
    Uses dynamic delay that increases on consecutive errors.

    Args:
        taxon (str): Organism name or TaxID
        max_retries (int): Maximum number of retry attempts

    Returns:
        dict: Taxonomy data with keys: taxid, scientific_name, rank, and taxonomy ranks
    """
    global consecutive_errors, current_delay

    for attempt in range(max_retries):
        try:
            if not re.match(r'^\d+$', str(taxon)):
                handle = Entrez.esearch(db="taxonomy", term=f'"{taxon}"', retmode="xml")
                record = Entrez.read(handle, validate=False)
                handle.close()

                if not record["IdList"]:
                    consecutive_errors = 0
                    return {
                        "taxid": "",
                        "scientific_name": "",
                        "rank": "",
                        "error": f"Taxon '{taxon}' not found in NCBI Taxonomy"
                    }

                tax_id = record["IdList"][0]
            else:
                tax_id = str(taxon)

            time.sleep(current_delay)

            handle = Entrez.efetch(db="taxonomy", id=tax_id, retmode="xml")
            record = Entrez.read(handle, validate=False)
            handle.close()

            tax = record[0]
            lineage = {(x["Rank"] if x["Rank"] else "unknown"): x["ScientificName"] for x in tax["LineageEx"]}

            consecutive_errors = 0
            result = {
                "taxid": tax_id,
                "scientific_name": tax["ScientificName"],
                "rank": tax["Rank"],
            }

            for rank in TAXONOMY_RANKS:
                result[rank] = lineage.get(rank, "")

            return result

        except Exception as e:
            error_msg = str(e)
            consecutive_errors += 1

            if consecutive_errors > 5:
                current_delay = min(current_delay + NCBI_DELAY_INCREMENT, NCBI_DELAY_MAX)
                logging.warning(f"Increasing delay to {current_delay:.2f}s after {consecutive_errors} consecutive errors")

            if attempt < max_retries - 1:
                logging.warning(f"Retry {attempt + 1}/{max_retries} for '{taxon}': {error_msg}")
                time.sleep(NCBI_RETRY_DELAY)
            else:
                logging.error(f"Failed after {max_retries} attempts for '{taxon}': {error_msg}")
                return {
                    "taxid": "",
                    "scientific_name": "",
                    "rank": "",
                    "error": error_msg
                }

def append_taxonomy_result(taxonomy_file, organism, tax_data, first_row):
    """Append a single taxonomy result to the TSV file."""
    row_data = {
        "organism": organism,
        "taxid": tax_data.get('taxid', ''),
        "scientific_name": tax_data.get('scientific_name', ''),
        "rank": tax_data.get('rank', ''),
    }
    for rank in TAXONOMY_RANKS:
        row_data[rank] = tax_data.get(rank, '')

    df_row = pd.DataFrame([row_data])

    if first_row:
        df_row.to_csv(taxonomy_file, sep='\t', index=False, mode='w', header=True)
    else:
        df_row.to_csv(taxonomy_file, sep='\t', index=False, mode='a', header=False)

    return False

def get_taxonomy_batch(organisms_df, progress_interval=100):
    """
    Fetch taxonomy data for all organisms in a dataframe.
    Saves results incrementally to backup TSV file.

    Args:
        organisms_df (pd.DataFrame): DataFrame with 'organism' column
        progress_interval (int): Print progress every N organisms

    Returns:
        dict: Mapping of organism name -> taxonomy data
    """
    unique_organisms = organisms_df['organism'].dropna().unique()
    total = len(unique_organisms)

    taxonomy_map = {}
    failed_count = 0
    start_time = time.time()

    taxonomy_file = DS_TAXONOMY_INCREMENTAL_FILE
    first_row = not os.path.exists(taxonomy_file)

    logging.info(f"Fetching taxonomy for {total} unique organisms...")
    logging.info(f"Estimated time: {total * 2 * current_delay / 3600:.1f} hours (at ~{current_delay:.2f}s per call)")
    logging.info(f"Incremental backup: {taxonomy_file}")

    for i, organism in enumerate(unique_organisms, 1):
        try:
            tax_data = get_ncbi_tax_with_retry(organism)
            taxonomy_map[organism] = tax_data

            first_row = append_taxonomy_result(taxonomy_file, organism, tax_data, first_row)

            if 'error' in tax_data:
                failed_count += 1

            if i % progress_interval == 0 or i == total:
                elapsed = time.time() - start_time
                percent = (i / total) * 100
                eta = (elapsed / i) * (total - i) if i > 0 else 0

                logging.info(
                    f"  Progress: {i}/{total} ({percent:.1f}%) | "
                    f"Failed: {failed_count} | "
                    f"Elapsed: {elapsed/60:.1f}min | "
                    f"ETA: {eta/60:.1f}min"
                )

        except Exception as e:
            logging.error(f"Unexpected error for '{organism}': {e}")
            taxonomy_map[organism] = {
                "taxid": "",
                "scientific_name": "",
                "rank": "",
                "error": str(e)
            }
            failed_count += 1

    logging.info(f"Completed. Fetched {len(taxonomy_map)} organisms, {failed_count} failed.")
    return taxonomy_map

def format_duration(seconds):
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h {(seconds%3600)/60:.0f}min"

def main():
    logger, log_file = setup_logging()

    print(f"\n{'='*60}")
    print("DS-GetOrganismTaxonomy.py - Fetch Taxonomy Data")
    print(f"{'='*60}\n")

    start_time = time.time()
    logging.info("=" * 50)
    logging.info("DS-GetOrganismTaxonomy.py started")
    logging.info(f"NCBI Email: {NCBI_ENTREZ_EMAIL}")
    logging.info(f"NCBI API Key: {'*' * 20}{NCBI_API_KEY[-10:] if len(NCBI_API_KEY) > 10 else NCBI_API_KEY}")
    logging.info(f"Rate limit: {NCBI_DELAY_MIN}s per query (with API key, auto-adjusts up to {NCBI_DELAY_MAX}s)")
    logging.info(f"Incremental backup: {DS_TAXONOMY_INCREMENTAL_FILE}")
    logging.info(f"Log file: {log_file}")

    input_file  = "ds/organisms_without_taxonomy.tsv" # DS_BALANCED_INPUT # "ds/ds_160k_balanced_missing_taxonomy1.tsv" 
    output_file = "ds/ds/ds_160k_balanced_taxonomy3.tsv" # DS_TAXONOMY_FILE # "ds/ds_160k_balanced_taxonomy2.tsv"

    logging.info(f"Reading input file: {input_file}")
    print(f"Reading input file: {input_file}")

    df = pd.read_csv(input_file, sep='\t')
    record_count = len(df)
    unique_organisms = df['organism'].nunique()

    logging.info(f"Loaded {record_count} records with {unique_organisms} unique organisms")
    print(f"  Loaded {record_count} records with {unique_organisms} unique organisms")

    logging.info("Fetching taxonomy data from NCBI...")
    print("\nFetching taxonomy data from NCBI...")

    taxonomy_map = get_taxonomy_batch(df, progress_interval=100)

    logging.info("Adding taxonomy columns to dataframe...")
    print("\nAdding taxonomy columns to dataframe...")

    for rank in TAXONOMY_RANKS:
        df[rank] = df['organism'].map(lambda x: taxonomy_map.get(x, {}).get(rank, ""))

    df['TaxID'] = df['organism'].map(lambda x: taxonomy_map.get(x, {}).get('taxid', ""))
    df['Name'] = df['organism'].map(lambda x: taxonomy_map.get(x, {}).get('scientific_name', ""))
    df['Rank'] = df['organism'].map(lambda x: taxonomy_map.get(x, {}).get('rank', ""))

    failed_organisms = sum(1 for v in taxonomy_map.values() if 'error' in v)
    logging.info(f"Taxonomy fetch complete. Failed organisms: {failed_organisms}")

    columns_to_drop = ['sequence']
    existing_cols_to_drop = [c for c in columns_to_drop if c in df.columns]
    df = df.drop(columns=existing_cols_to_drop)

    taxonomy_cols_order = ['TaxID', 'Name', 'Rank'] + TAXONOMY_RANKS
    other_cols = [c for c in df.columns if c not in taxonomy_cols_order]
    new_col_order = taxonomy_cols_order + other_cols
    df = df[new_col_order]

    os.makedirs(os.path.dirname(output_file), exist_ok=True) if os.path.dirname(output_file) else None
    logging.info(f"Saving output to: {output_file}")
    print(f"\nSaving output to: {output_file}")

    df.to_csv(output_file, sep='\t', index=False)
    logging.info(f"Saved {len(df)} records with {len(df.columns)} columns")

    print(f"  Saved {len(df)} records with {len(df.columns)} columns")

    print("\nColumn summary:")
    logging.info("Columns: " + ", ".join(df.columns))
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    end_time = time.time()
    total_duration = end_time - start_time

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Total records: {record_count}")
    print(f"  Unique organisms: {unique_organisms}")
    print(f"  Failed lookups: {failed_organisms}")
    print(f"  Output file: {output_file}")
    print(f"  Log file: {log_file}")
    print(f"  Total running time: {format_duration(total_duration)}")
    print(f"{'='*60}")

    logging.info("=" * 50)
    logging.info("DS-GetOrganismTaxonomy.py completed successfully")
    logging.info(f"Total running time: {format_duration(total_duration)}")

if __name__ == "__main__":
    main()