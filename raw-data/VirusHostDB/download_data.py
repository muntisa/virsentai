#!/usr/bin/env python3
"""
VirusHostDB Data Downloader

Downloads data files from the VirusHostDB FTP server.
Retrieves metadata, genomic sequences, and virus lists.

Usage:
    python raw-data/VirusHostDB/download_data.py

Requirements:
    - requests library
    - Internet connection to access genome.jp FTP

Output:
    - raw-data/VirusHostDB/virushostdb.tsv
    - raw-data/VirusHostDB/virushostdb.formatted.genomic.fna.gz
    - raw-data/VirusHostDB/non-segmented_virus_list.tsv
    - raw-data/VirusHostDB/segmented_virus_list.tsv

Configuration:
    - RAW_DATA_VirusHostDB_PATH: Target directory for downloads
"""

import os
import requests
import sys
import time
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import RAW_DATA_VirusHostDB_PATH, LOG_PATH

def setup_logging(script_name):
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_PATH, f"VHDB_{script_name}_{timestamp}.log")
    
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

FTP_BASE_URL = "https://www.genome.jp/ftp/db/virushostdb/"
FILES_TO_DOWNLOAD = [
    "virushostdb.tsv",
    "virushostdb.formatted.genomic.fna.gz",
    "non-segmented_virus_list.tsv",
    "segmented_virus_list.tsv"
]

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {target_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        sys.exit(1)

def main():
    setup_logging("download_data")
    # Ensure directory exists
    os.makedirs(RAW_DATA_VirusHostDB_PATH, exist_ok=True)
    
    for filename in FILES_TO_DOWNLOAD:
        url = FTP_BASE_URL + filename
        target_path = os.path.join(RAW_DATA_VirusHostDB_PATH, filename)
        download_file(url, target_path)

if __name__ == "__main__":
    main()
