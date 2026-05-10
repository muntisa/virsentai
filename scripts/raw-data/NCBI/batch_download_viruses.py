#!/usr/bin/env python3
"""
NCBI Virus Batch Downloader

Downloads virus genome packages from NCBI in batches.
Reads accessions from all_complete_metadata.jsonl and downloads
each batch as a ZIP file containing genomic sequences.

Usage:
    python raw-data/NCBI/batch_download_viruses.py

Requirements:
    - datasets CLI tool (NCBI Datasets)
    - all_complete_metadata.jsonl (from fetch_virus_metadata.py)
    - Optional: NCBI_API_KEY in config.py for higher rate limits

Output:
    - raw-data/NCBI/batches/virus_batch_N.zip (one per batch)
    - Accession list files (temp, cleaned up after download)

Configuration:
    - batch_size: Number of accessions per batch (default: 5000)
"""

import json
import subprocess
import os
import time
import sys
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

def batch_download():
    script_start_time = time.time()
    setup_logging("batch_download")

    tool_path = os.path.join(os.path.dirname(__file__), "datasets.exe")
    metadata_file = os.path.join("raw-data", "NCBI", "all_complete_metadata.jsonl")
    batch_dir = os.path.join("raw-data", "NCBI", "batches")
    batch_size = 5000 # Number of genomes per batch
    
    # Read API Key if available
    try:
        from config import NCBI_API_KEY
    except ImportError:
        NCBI_API_KEY = ""
        
    base_cmd = [tool_path]
    if NCBI_API_KEY:
        base_cmd += ["--api-key", NCBI_API_KEY]
    
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file {metadata_file} not found. Please run fetch_virus_metadata.py first.")
        return
    
    if not os.path.exists(batch_dir):
        os.makedirs(batch_dir)

    print("Reading accessions from metadata...")
    accessions = []
    with open(metadata_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    accessions.append(data['accession'])
                except:
                    continue
    
    total = len(accessions)
    print(f"Total accessions found: {total}")
    
    for i in range(0, total, batch_size):
        batch = accessions[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        batch_filename = os.path.join(batch_dir, f"virus_batch_{batch_num}.zip")
        acc_list_file = os.path.join(batch_dir, f"acc_batch_{batch_num}.txt")
        
        # Save batch accessions to a temp file
        with open(acc_list_file, "w") as f:
            for acc in batch:
                f.write(acc + "\n")
        
        print(f"Downloading batch {batch_num}/{total_batches} ({len(batch)} accessions)...")
        
        cmd = base_cmd + [
            "download", "virus", "genome", "accession",
            "--inputfile", acc_list_file,
            "--filename", batch_filename,
            "--no-progressbar"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"Batch {batch_num} completed in {end_time - start_time:.1f} seconds.")
            # Remove the temp accession list
            os.remove(acc_list_file)
        else:
            print(f"Error in batch {batch_num}: {result.stderr}")
            # Keep the accession list for retrying
            
    print("All batches processed.")
    
    script_end_time = time.time()
    elapsed = script_end_time - script_start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")


if __name__ == "__main__":
    batch_download()
