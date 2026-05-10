#!/usr/bin/env python3
"""
NCBI Virus Sample Downloader

Downloads a sample of complete RefSeq virus genomes from NCBI Datasets CLI.
This script retrieves 5 complete RefSeq viral genomes for testing purposes.

Usage:
    python raw-data/NCBI/download_sample.py

Requirements:
    - datasets CLI tool (NCBI Datasets)
    - Place datasets.exe in the same directory as this script

Output:
    - accessions_sample.txt: List of downloaded accession IDs
    - sample_viruses.zip: Downloaded genome package
"""

import subprocess
import json
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

def run():
    script_start_time = time.time()
    setup_logging("download_sample")

    print("Getting summary for 5 complete RefSeq viruses...")
    datasets_tool = os.path.join(os.path.dirname(__file__), "datasets.exe")
    cmd_summary = [datasets_tool, "summary", "virus", "genome", "taxon", "Viruses", "--complete-only", "--refseq", "--limit", "5", "--as-json-lines"]
    result = subprocess.run(cmd_summary, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error getting summary: {result.stderr}")
        return

    lines = result.stdout.strip().split('\n')
    accessions = []
    for line in lines:
        if line:
            try:
                data = json.loads(line)
                accessions.append(data['accession'])
            except Exception as e:
                print(f"Error parsing line: {e}")

    print(f"Found accessions: {accessions}")
    
    if not accessions:
        print("No accessions found.")
        return

    with open("accessions_sample.txt", "w") as f:
        for acc in accessions:
            f.write(acc + "\n")

    print("Downloading sample package for these accessions...")
    # Using datasets download genome accession --inputfile to download specific accessions
    cmd_download = [datasets_tool, "download", "virus", "genome", "accession", "--inputfile", "accessions_sample.txt", "--filename", "sample_viruses.zip"]
    result_dl = subprocess.run(cmd_download, capture_output=True, text=True)
    
    if result_dl.returncode != 0:
        print(f"Error downloading package: {result_dl.stderr}")
    else:
        print("Download successful: sample_viruses.zip")

    script_end_time = time.time()
    elapsed = script_end_time - script_start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

if __name__ == "__main__":

    run()
