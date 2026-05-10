#!/usr/bin/env python3
"""
NCBI Data TSV Converter

Converts downloaded NCBI virus genome packages into a unified TSV format.
Reads metadata from all_complete_metadata.jsonl and genomic sequences from
downloaded ZIP packages, then merges them into a single TSV file.

Usage:
    python raw-data/NCBI/convert_to_tsv.py

Requirements:
    - all_complete_metadata.jsonl in raw-data/NCBI/
    - Batch ZIP files in raw-data/NCBI/batches/

Output:
    - raw-data/NCBI/NCBI_raw.tsv with columns:
      id, db_id, accession, sequence, length, source, collection_date,
      country, host, organism, segment, completeness, created_at

Configuration:
    - RAW_DATA_NCBI_PATH: Directory containing NCBI data
    - LOG_PATH: Directory for log files
"""

import os
import sys
import json
import zipfile
import time
import io
from datetime import datetime

# Add root to path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
try:
    from config import RAW_DATA_NCBI_PATH, LOG_PATH
except ImportError:
    RAW_DATA_NCBI_PATH = "raw-data/NCBI"
    LOG_PATH = "logs"

def setup_logging(script_name):
    """
    Sets up logging to both terminal and a file in the LOG_PATH.
    Matches the style used in other NCBI scripts.
    """
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

def parse_fasta_stream(fasta_stream):
    """
    Manually parses FASTA format from an IO stream to avoid external dependencies.
    Yields (accession, sequence) tuples.
    """
    acc = None
    seq = []
    
    # Use io.TextIOWrapper to handle the binary stream from zip
    wrapper = io.TextIOWrapper(fasta_stream, encoding='utf-8')
    
    for line in wrapper:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if acc:
                yield acc, "".join(seq)
            # Extract accession (first word after >)
            acc = line[1:].split()[0]
            seq = []
        else:
            seq.append(line)
    
    if acc:
        yield acc, "".join(seq)

def convert_to_tsv():
    """
    Reads metadata and downloaded genomic sequences, then saves them as a TSV file.
    Includes all fields defined in the database schema.
    """
    script_start_time = time.time()
    setup_logging("convert_to_tsv")
    
    metadata_file = os.path.join(RAW_DATA_NCBI_PATH, "all_complete_metadata.jsonl")
    batches_dir = os.path.join(RAW_DATA_NCBI_PATH, "batches")
    output_file = os.path.join(RAW_DATA_NCBI_PATH, "NCBI_raw.tsv")
    
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file {metadata_file} not found.")
        return

    print("Loading metadata...")
    metadata_map = {}
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        acc = data.get('accession')
                        if acc:
                            # Extract schema fields
                            virus_name = data.get('virus', {}).get('organism_name', 'Unknown')
                            host_name = data.get('host', {}).get('organism_name', 'Unknown')
                            length = data.get('length', 0)
                            source = data.get('source_database', 'Unknown')
                            collection_date = data.get('isolate', {}).get('collection_date', 'Unknown')
                            country = data.get('location', {}).get('geographic_location', 'Unknown')
                            segment = data.get('segment', 'ANONYMOUS')
                            completeness = data.get('completeness', 'Unknown')
                            
                            metadata_map[acc] = {
                                'organism': virus_name,
                                'host': host_name,
                                'length': length,
                                'source': source,
                                'collection_date': collection_date,
                                'country': country,
                                'segment': segment,
                                'completeness': completeness
                            }
                    except Exception as e:
                        continue
    except Exception as e:
        print(f"Error reading metadata file: {e}")
        return
        
    print(f"Loaded metadata for {len(metadata_map)} accessions.")
    
    if not os.path.exists(batches_dir):
        print(f"Error: Batches directory {batches_dir} not found.")
        return

    zip_files = [f for f in os.listdir(batches_dir) if f.endswith(".zip")]
    if not zip_files:
        print(f"No zip files found in {batches_dir}")
        return
        
    print(f"Found {len(zip_files)} zip files. Starting conversion...")
    
    processed_count = 0
    missing_meta_count = 0
    created_at = datetime.now().isoformat()
    
    try:
        with open(output_file, "w", encoding="utf-8") as out_f:
            # Write header matching SQLITE_VIRUSES_SCHEMA (excluding id autoincrement maybe, or include as row number)
            # Columns: id, db_id, accession, sequence, length, source, collection_date, country, host, organism, segment, completeness, created_at
            header = ["id", "db_id", "accession", "sequence", "length", "source", "collection_date", "country", "host", "organism", "segment", "completeness", "created_at"]
            out_f.write("\t".join(header) + "\n")
            
            for zip_name in zip_files:
                zip_path = os.path.join(batches_dir, zip_name)
                print(f"Processing {zip_name}...")
                
                try:
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        fasta_path = "ncbi_dataset/data/genomic.fna"
                        
                        if fasta_path in z.namelist():
                            with z.open(fasta_path) as fasta_file:
                                for acc, seq in parse_fasta_stream(fasta_file):
                                    processed_count += 1
                                    
                                    meta = metadata_map.get(acc)
                                    if not meta:
                                        base_acc = acc.split('.')[0]
                                        meta = metadata_map.get(base_acc)
                                    
                                    if meta:
                                        m = meta
                                    else:
                                        m = {k: 'Unknown' for k in ['organism', 'host', 'source', 'collection_date', 'country', 'segment', 'completeness']}
                                        m['length'] = len(seq)
                                        missing_meta_count += 1
                                    
                                    # Prepare row
                                    row = [
                                        str(processed_count),           # id
                                        "NCBI",                         # db_id
                                        acc,                            # accession
                                        seq,                            # sequence
                                        str(m['length']),               # length
                                        m['source'],                    # source
                                        m['collection_date'],           # collection_date
                                        m['country'],                   # country
                                        m['host'],                      # host
                                        m['organism'],                  # organism
                                        m['segment'],                   # segment
                                        m['completeness'],              # completeness
                                        created_at                      # created_at
                                    ]
                                    
                                    # Clean tabs from all fields
                                    clean_row = [str(val).replace("\t", " ") for val in row]
                                    out_f.write("\t".join(clean_row) + "\n")
                                    
                        else:
                            print(f"Warning: {fasta_path} not found in {zip_name}")
                except Exception as e:
                    print(f"Error processing {zip_name}: {e}")

        print(f"\nConversion complete.")
        print(f"Total sequences processed: {processed_count}")
        if missing_meta_count > 0:
            print(f"Warning: {missing_meta_count} sequences had no matching metadata.")
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"Failed to write output TSV: {e}")

    script_end_time = time.time()
    elapsed = script_end_time - script_start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

if __name__ == "__main__":
    convert_to_tsv()
