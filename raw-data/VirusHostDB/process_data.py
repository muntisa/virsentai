#!/usr/bin/env python3
"""
VirusHostDB Data Processor

Processes downloaded VirusHostDB files and converts them to unified TSV format.
Merges metadata with genomic sequences, filtering for complete genomes only.

Usage:
    python raw-data/VirusHostDB/process_data.py

Requirements:
    - virushostdb.tsv (from download_data.py)
    - virushostdb.formatted.genomic.fna.gz (from download_data.py)
    - non-segmented_virus_list.tsv (from download_data.py)

Output:
    - raw-data/VirusHostDB/VirHostDB_raw.tsv with columns:
      db_id, accession, sequence, length, source, collection_date,
      country, host, organism, segment, completeness

Notes:
    - Only keeps non-segmented (complete) genomes
    - Filters based on complete_accessions list
"""

import os
import csv
import gzip
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

def parse_fasta(fasta_path):
    """Parses the formatted FASTA file and yields records."""
    print(f"Parsing FASTA: {fasta_path}")
    current_header = None
    current_seq = []
    
    # Use gzip if the file ends with .gz
    open_func = gzip.open if fasta_path.endswith('.gz') else open
    mode = 'rt' if fasta_path.endswith('.gz') else 'r'
    
    with open_func(fasta_path, mode, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header:
                    yield current_header, "".join(current_seq)
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_header:
            yield current_header, "".join(current_seq)

def main():
    setup_logging("process_data")
    metadata_path = os.path.join(RAW_DATA_VirusHostDB_PATH, "virushostdb.tsv")
    fasta_path = os.path.join(RAW_DATA_VirusHostDB_PATH, "virushostdb.formatted.genomic.fna.gz")
    output_path = os.path.join(RAW_DATA_VirusHostDB_PATH, "VirHostDB_raw.tsv")
    
    # Check if files exist
    if not os.path.exists(metadata_path) or not os.path.exists(fasta_path):
        print(f"Error: Missing input files in {RAW_DATA_VirusHostDB_PATH}")
        return

    # Load metadata into a lookup dictionary (by refseq id / accession)
    # The README says refseq id column contains RefSeq IDs. 
    # But FASTA header has Sequence_accession.
    # Note: one virus tax id can have multiple refseq ids.
    metadata_lookup = {}
    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            refseq_ids = row.get('refseq id', '').split(',')
            for rid in refseq_ids:
                rid = rid.strip()
                if rid:
                    metadata_lookup[rid] = row

    # Load non-segmented list (Complete genomes only)
    non_segmented_path = os.path.join(RAW_DATA_VirusHostDB_PATH, "non-segmented_virus_list.tsv")
    complete_accessions = set()
    print(f"Loading complete/non-segmented accession list from {non_segmented_path}...")
    with open(non_segmented_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                # The 3rd column is a comma-separated list of accessions
                accs = [a.strip() for a in parts[2].split(',')]
                for a in accs:
                    if a:
                        complete_accessions.add(a)
    
    # Process FASTA and merge
    unified_columns = ["db_id", "accession", "sequence", "length", "source", "collection_date", "country", "host", "organism", "segment", "completeness"]
    
    print(f"Processing FASTA and writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=unified_columns, delimiter='\t')
        writer.writeheader()
        
        count = 0
        skipped_segmented = 0
        for header, sequence in parse_fasta(fasta_path):
            # Header format: Sequence_accession virus name | Hostname | ...
            parts = header.split('|')
            first_part = parts[0].strip()
            accession = first_part.split(' ')[0]
            
            # Filter: ONLY keep non-segmented (complete) genomes
            if accession not in complete_accessions:
                skipped_segmented += 1
                continue
                
            virus_name = " ".join(first_part.split(' ')[1:])
            
            host = ""
            if len(parts) > 1:
                host = parts[1].strip()
            
            meta = metadata_lookup.get(accession, {})
            db_id = f"VirusHostDB:{meta.get('virus tax id', 'unknown')}"
            
            writer.writerow({
                "db_id": db_id,
                "accession": accession,
                "sequence": sequence,
                "length": len(sequence),
                "source": "VirusHostDB",
                "collection_date": "",
                "country": "",
                "host": host or meta.get('host name', ''),
                "organism": virus_name or meta.get('virus name', ''),
                "segment": "", 
                "completeness": "complete"
            })
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} records (Skipped {skipped_segmented} segmented/other)...")
                
    print(f"Finished. Processed {count} records. Skipped {skipped_segmented} records.")
                
    print(f"Finished. Processed {count} records.")

if __name__ == "__main__":
    main()
