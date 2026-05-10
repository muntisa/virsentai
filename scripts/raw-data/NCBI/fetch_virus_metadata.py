#!/usr/bin/env python3
"""
NCBI Virus Metadata Fetcher

Fetches metadata for all complete RefSeq virus genomes from NCBI Datasets.
Uses a two-phase approach:
1. Discovery: Find unique Tax IDs for complete RefSeq viruses
2. Enrichment: Get detailed metadata (accession, host, location, etc.)

Usage:
    python raw-data/NCBI/fetch_virus_metadata.py

Requirements:
    - datasets CLI tool (NCBI Datasets)
    - Optional: NCBI_API_KEY in config.py for higher rate limits

Output:
    - raw-data/NCBI/all_complete_metadata.jsonl (JSON Lines format)
"""

import subprocess
import os
import time
import sys
import json
from datetime import datetime

# Add root to path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
try:
    from config import LOG_PATH
except ImportError:
    LOG_PATH = "logs"

def setup_logging(script_name):
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_PATH, f"NCBI_{script_name}_{timestamp}.log")
    
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

def fetch_metadata():
    script_start_time = time.time()
    setup_logging("fetch_metadata")

    tool_path = os.path.join(os.path.dirname(__file__), "datasets.exe")
    output_file = os.path.join("raw-data", "NCBI", "all_complete_metadata.jsonl")
    
    # Read API Key if available
    try:
        from config import NCBI_API_KEY
    except ImportError:
        NCBI_API_KEY = ""
        
    base_cmd = [tool_path]
    if NCBI_API_KEY:
        base_cmd += ["--api-key", NCBI_API_KEY]

    # PHASE 1: Discovery (Use genome subcommand to get unique Tax IDs)
    print("PHASE 1: Discovering unique Tax IDs for complete RefSeq viruses...")
    # Matches: Complete assembly, RefSeq source, Viruses taxon (10239)
    discovery_cmd = base_cmd + ["summary", "genome", "taxon", "10239", "--assembly-level", "complete", "--assembly-source", "RefSeq", "--as-json-lines"]
    
    unique_tax_ids = set()
    try:
        process = subprocess.Popen(discovery_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in process.stdout:
            if line.strip():
                try:
                    data = json.loads(line)
                    tax_id = data.get('organism', {}).get('tax_id')
                    if tax_id:
                        unique_tax_ids.add(str(tax_id))
                except:
                    continue
        
        process.wait()
        if process.returncode != 0:
            print(f"Error during discovery: {process.stderr.read()}")
            return
            
        print(f"Discovery complete. Total unique Tax IDs found: {len(unique_tax_ids)}")
        
    except Exception as e:
        print(f"An error occurred during discovery: {e}")
        return

    if not unique_tax_ids:
        print("No Tax IDs found. Exiting.")
        return

    # PHASE 2: Enrichment (Use virus subcommand in batches of Tax IDs to get full metadata)
    print("\nPHASE 2: Enriching metadata (ID, Name, Host) via batched Tax ID queries...")
    tax_ids_list = list(unique_tax_ids)
    batch_size = 100 # summary virus genome taxon --inputfile supports up to 100
    total_ids = len(tax_ids_list)
    
    seen_accessions = set()
    
    try:
        with open(output_file, "w", encoding="utf-8") as out_f:
            for i in range(0, total_ids, batch_size):
                batch = tax_ids_list[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (total_ids + batch_size - 1) // batch_size
                
                print(f"Enriching batch {batch_num}/{total_batches} ({len(batch)} Tax IDs)...")
                
                # Save batch Tax IDs to a temp file
                batch_tax_file = os.path.join("raw-data", "NCBI", "batch_tax_temp.txt")
                with open(batch_tax_file, "w") as f:
                    for tid in batch:
                        f.write(tid + "\n")

                # Fetch virus metadata using --inputfile for the tax ids
                # We also include --refseq and --complete-only again for safety
                enrich_cmd = base_cmd + ["summary", "virus", "genome", "taxon", "--inputfile", batch_tax_file, "--complete-only", "--refseq", "--as-json-lines"]
                
                enrich_process = subprocess.run(enrich_cmd, capture_output=True, text=True, encoding="utf-8")
                
                # Clean up temp file
                if os.path.exists(batch_tax_file):
                    os.remove(batch_tax_file)
                
                if enrich_process.returncode == 0:
                    lines_added = 0
                    for line in enrich_process.stdout.splitlines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                acc = data.get('accession')
                                if acc and acc not in seen_accessions:
                                    out_f.write(line + "\n")
                                    seen_accessions.add(acc)
                                    lines_added += 1
                            except:
                                continue
                    
                    # Dynamic sleep to optimize for rate limits
                    sleep_time = 0.15 if NCBI_API_KEY else 0.4
                    time.sleep(sleep_time)
                else:
                    print(f"Warning: Batch {batch_num} failed: {enrich_process.stderr}")
                    
        print(f"\nMetadata enrichment complete. Total unique viruses saved: {len(seen_accessions)}")
        print(f"Final metadata file: {output_file}")
                
    except Exception as e:
        print(f"An error occurred during enrichment: {e}")
    
    script_end_time = time.time()
    elapsed = script_end_time - script_start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")


if __name__ == "__main__":
    fetch_metadata()
