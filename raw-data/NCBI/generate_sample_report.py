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

def generate_report():
    script_start_time = time.time()
    setup_logging("generate_sample_report")

    # Assuming these are in the project root relative to the raw-data/NCBI folder
    # if running from root, these are correct.
    # But better to be absolute if we know where they are.
    data_path = os.path.join("raw-data", "NCBI", "sample_data", "ncbi_dataset", "data", "data_report.jsonl")
    fasta_path = os.path.join("raw-data", "NCBI", "sample_data", "ncbi_dataset", "data", "genomic.fna")
    
    if not os.path.exists(data_path) or not os.path.exists(fasta_path):
        print("Required files not found.")
        return

    metadata = {}
    with open(data_path) as f:
        for line in f:
            d = json.loads(line)
            acc = d.get('accession')
            if acc:
                metadata[acc] = {
                    'name': d.get('virus', {}).get('organismName', 'N/A'),
                    'host': d.get('host', {}).get('organismName', 'N/A')
                }

    sequences = {}
    current_id = None
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                current_id = line.split()[0][1:]
                sequences[current_id] = ''
            elif current_id:
                sequences[current_id] += line.strip()

    print(f"{'Accession':<15} | {'Virus Name':<40} | {'Host Name':<30} | {'Seq Preview'}")
    print("-" * 110)
    for acc in sorted(metadata.keys()):
        name = metadata[acc]['name']
        host = metadata[acc]['host']
        seq = sequences.get(acc, 'N/A')
        preview = seq[:40] + "..." if len(seq) > 40 else seq
        print(f"{acc:<15} | {name:<40} | {host:<30} | {preview}")

    script_end_time = time.time()
    elapsed = script_end_time - script_start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

if __name__ == "__main__":

    generate_report()
