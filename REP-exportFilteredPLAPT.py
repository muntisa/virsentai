#!/usr/bin/env python3
"""
Export Filtered PLAPT Results

Exports filtered PLAPT_AE data from SQLite database to CSV
and updates the webapp HTML file for visualization.

Usage:
    python REP-exportFilteredPLAPT.py

Workflow:
    1. Connect to SQLite database
    2. Query PLAPT_AE joined with predictions and drug data
       - Filter: prediction_score >= 0.8
       - Filter: neg_log10_affinity_M >= 0.9
    3. Apply blacklist filter (remove synthetic/construct/vector entries)
    4. Fetch protein names from NCBI for entries marked with '*'
    5. Clean protein names (remove bracketed virus names)
    6. Export filtered data to CSV
    7. Update webapp HTML file

Input:
    - db/virsentai.sqlite3 (with PLAPT_AE, predictions, chembl_approved_drug tables)
    - webapp/repurposing.html template

Output:
    - virsentai_PLAPT_AE_prob-0.8_AE-0.9.csv: Filtered results
    - webapp/repurposing.html: Updated with new data
    - webapp/repurposing_old.html: Backup

Blacklist Filter:
    Removes entries with virus_name containing:
    synthetic, construct, vector, plasmid, clone, pseudovirus, etc.

Configuration:
    - VIRSENTAI_PROB_CUTOFF: 0.8
    - PLAPT_AFFINITY_CUTOFF: 0.9

Notes:
    - Requires database with populated PLAPT_AE table
    - Logs to logs/REP-exportFilteredPLAPT_<timestamp>.log
    - webapp/repurposing.html file exists
    - Python packages: pandas, biopython (optional, for protein name enrichment)
"""

# export filtered PLAPT_AE data from SQLite DB to CSV and update HTML file
# This script connects to the SQLite database, retrieves PLAPT_AE data
# based on specified criteria, saves the filtered data to a CSV file,  
# and updates an HTML file with the new CSV data (if enabled).

import sqlite3
import pandas as pd
import os
import shutil # Import the shutil module for file operations
import time
import re

# Try to import Biopython Entrez/SeqIO for fetching protein names from NCBI
try:
    from Bio import Entrez, SeqIO
except Exception:
    Entrez = None
    SeqIO = None

from config import *

# --- Configuration ---
HTML_FILE = os.path.join(WEBAPP_DIR, "repurposing.html")
BACKUP_FILE = os.path.join(WEBAPP_DIR, "repurposing_old.html")
OUTPUT_CSV_FILE = PLAPT_EXPORT_FILE
flag_AddPLAPT2HTML = PLAPT_UPDATE_HTML

# --- SQL Query (Remains the same as it fetches all data before filtering) ---
SQL_QUERY = f"""
SELECT
    A.virus_id,
    P.virus_name,
    A.protein_id,
    -- Hardcode the protein_name as requested
    '*' AS protein_name, 
    A.drug_ID,
    -- Get the drug name from the 'pref_name' column
    D.pref_name AS drug_name,
    A.neg_log10_affinity_M
FROM
    PLAPT_AE A
INNER JOIN
    predictions P ON A.virus_id = P.virus_id
INNER JOIN
    chembl_approved_drug D ON A.drug_ID = D.drug_chembl_id
WHERE
    P.prediction_score > {VIRSENTAI_PROB_CUTOFF}
AND
    A.neg_log10_affinity_M > {PLAPT_AFFINITY_CUTOFF}
ORDER BY
    A.neg_log10_affinity_M DESC;
"""

def process_database():
    """
    Main function to process and export filtered PLAPT data.
    
    Connects to the SQLite database, executes the query, applies filters,
    saves the CSV, and updates the HTML file.
    
    Steps:
        1. Connect to SQLite database
        2. Execute SQL query (PLAPT_AE + predictions + drugs join)
        3. Apply blacklist filter (remove synthetic/construct entries)
        4. Fetch protein names from NCBI for unknown entries
        5. Clean protein names (remove virus brackets)
        6. Export to CSV file
        7. Update webapp HTML file (if enabled)
    
    Output Files:
        - CSV: Configured via PLAPT_EXPORT_FILE
        - HTML: webapp/repurposing.html (backup: webapp/repurposing_old.html)
    
    Error Handling:
        - Database not found
        - SQLite operational errors
        - NCBI fetch failures (non-fatal)
        - HTML file not found
    """
    print(f"Connecting to database: {SQLite_PRED_FILE}")
    
    if not os.path.exists(SQLite_PRED_FILE):
        print(f"Error: Database file not found at {SQLite_PRED_FILE}")
        return

    try:
        # 1. Fetch and Filter Data
        conn = sqlite3.connect(SQLite_PRED_FILE)
        print("Executing SQL query and applying initial filters...")
        df = pd.read_sql_query(SQL_QUERY, conn)
        conn.close()

        initial_count = len(df)

        # Filter out records whose `virus_name` indicates synthetic/construct/vector-like entries.
        # Exclude if virus_name matches (case-insensitive) any of the blacklist terms below.
        blacklist_terms = ZOONOTIC_VIRUS_KEYWORDS
        # Build a regex pattern that matches any of the terms (escaped) — spaces and hyphens are preserved.
        # Use non-capturing group (?:...) to avoid UserWarning about match groups
        pattern = r"(?:" + r"|".join([re.escape(t) for t in blacklist_terms]) + r")"

        # Create a boolean mask for rows to exclude (case-insensitive)
        mask_exclude = df['virus_name'].astype(str).str.contains(pattern, case=False, regex=True, na=False)

        df_filtered = df[~mask_exclude].copy()

        final_count = len(df_filtered)
        removed_count = initial_count - final_count

        print(f"Initial records: {initial_count}. After blacklist filter, final records: {final_count} (Removed: {removed_count})")

        # NEW: complete the protein names! that are *
        # modify df_filtered
        # 1) Normalize `virus_name`: if there's a comma, keep only text before it
        df_filtered.loc[:, 'virus_name'] = df_filtered['virus_name'].astype(str).apply(lambda v: v.split(',', 1)[0].strip())
        # Fill the 'protein_name' column by querying NCBI Protein DB using the protein_id
        
        if Entrez is None or SeqIO is None:
            print("Biopython not available -- skipping protein name enrichment. Install biopython to enable this feature.")
        else:
            # Configure Entrez email from config
            Entrez.email = NCBI_ENTREZ_EMAIL

            def fetch_protein_names(protein_ids, batch_size=200, sleep_interval=0.34):
                """
                Fetch protein descriptions for a list of protein IDs from NCBI.
                
                Uses NCBI Entrez API to fetch GenBank records for protein accessions.
                
                Args:
                    protein_ids (list): List of NCBI protein accession IDs
                    batch_size (int): Number of IDs per batch request (default: 200)
                    sleep_interval (float): Delay between requests in seconds (default: 0.34)
                
                Returns:
                    dict: Mapping of protein_id -> protein description/title
                
                Features:
                    - Batch processing to respect NCBI rate limits
                    - Deduplication while preserving order
                    - Handles version suffixes (e.g., NP_001234.1)
                    - Non-fatal error handling (prints warning on failure)
                """
                mapping = {}
                # Clean and unique
                ids = [str(x).strip() for x in protein_ids if pd.notna(x) and str(x).strip()]
                # Remove entries that are just '*' or empty
                ids = [i for i in ids if i != '*' and i != '']
                # Unique while preserving order
                seen = set()
                unique_ids = []
                for i in ids:
                    if i not in seen:
                        seen.add(i)
                        unique_ids.append(i)

                for start in range(0, len(unique_ids), batch_size):
                    batch = unique_ids[start:start+batch_size]
                    ids_str = ",".join(batch)
                    try:
                        time.sleep(sleep_interval)
                        handle = Entrez.efetch(db="protein", id=ids_str, rettype="gb", retmode="text")
                        for rec in SeqIO.parse(handle, "gb"):
                            # rec.id is normally the accession (may include version)
                            pid = rec.id
                            desc = rec.description
                            mapping[pid] = desc
                        handle.close()
                    except Exception as e:
                        print(f"Warning: error fetching protein batch starting at {start}: {e}")
                return mapping

            # Determine rows that need updating (where protein_name is '*' or missing)
            try:
                need_mask = df_filtered['protein_name'].isin(['*']) | df_filtered['protein_name'].isna()
            except Exception:
                need_mask = df_filtered['protein_name'].isna()

            ids_to_lookup = df_filtered.loc[need_mask, 'protein_id'].astype(str).tolist()

            if ids_to_lookup:
                print(f"Fetching protein names for {len(set(ids_to_lookup))} unique protein IDs from NCBI...")
                name_map = fetch_protein_names(ids_to_lookup)

                if name_map:
                    # Apply mapping to dataframe. Try matching exact accession first; if not found, try without version suffix
                    def resolve_name(pid, current_name):
                        pid_str = str(pid)
                        if pid_str in name_map:
                            return name_map[pid_str]
                        # try removing version (.1, .2, etc.)
                        if '.' in pid_str:
                            base = pid_str.split('.')[0]
                            for k in name_map:
                                if k.startswith(base + '.') or k == base:
                                    return name_map[k]
                        return current_name

                    df_filtered.loc[:, 'protein_name'] = df_filtered.apply(lambda r: resolve_name(r['protein_id'], r['protein_name']), axis=1)

                    # 2) Clean protein_name: remove bracketed virus name e.g. '... [Virus name]'
                    df_filtered.loc[:, 'protein_name'] = (
                        df_filtered['protein_name']
                        .astype(str)
                        .str.replace(r"\s*\[.*?\]\s*", "", regex=True)
                        .str.strip()
                    )
                else:
                    print("No protein names retrieved from NCBI. Leaving '*' values as-is.")
        
        # 2. Generate CSV String and Save Output CSV
        
        # Generate the CSV content as a single string
        csv_data_string = df_filtered.to_csv(index=False)
        
        # Save the filtered DataFrame to the specified CSV file
        df_filtered.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"Data saved successfully to {OUTPUT_CSV_FILE}")

        # 3. Update HTML File

        if not flag_AddPLAPT2HTML:
            print("HTML update flag is disabled. Skipping HTML update.")
            return
        
        if not os.path.exists(HTML_FILE):
            print(f"Error: HTML file not found at {HTML_FILE}. Skipping HTML update.")
            return

        print(f"\nUpdating HTML file: {HTML_FILE}")
        
        # Copy the original file to a backup file
        shutil.copy(HTML_FILE, BACKUP_FILE)
        print(f"Original HTML backed up to {BACKUP_FILE}")

        # Read the content of the HTML file
        with open(HTML_FILE, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Define the markers for the data block
        START_MARKER = "window.RAW_CSV_DATA = `"
        END_MARKER = "`;\n    </script>"
        
        # Find the start and end indices for replacement
        start_index = html_content.find(START_MARKER)
        
        if start_index != -1:
            # Adjust start_index to point after the marker
            start_index += len(START_MARKER)
            
            # Find the end marker starting from the new start_index
            end_index = html_content.find(END_MARKER, start_index)

            if end_index != -1:
                # Construct the new HTML content
                new_html_content = (
                    html_content[:html_content.find(START_MARKER) + len(START_MARKER)] + 
                    csv_data_string.strip() + # Insert the CSV string, stripping leading/trailing whitespace
                    html_content[end_index:]
                )
                
                # Write the new content back to the HTML file
                with open(HTML_FILE, 'w', encoding='utf-8') as f:
                    f.write(new_html_content)
                
                print("HTML file updated successfully.")
            else:
                print("Error: Could not find the closing data marker in HTML. Skipping update.")
        else:
            print("Error: Could not find the starting data marker in HTML. Skipping update.")

    except sqlite3.OperationalError as e:
        print(f"An error occurred during database operation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Create the webapp directory if it doesn't exist for testing purposes
    os.makedirs(WEBAPP_DIR, exist_ok=True)
    
    # NOTE: The actual repurposing.html file must be in WEBAPP_DIR before running this script. 

    process_database()