import subprocess
import json
import os

def fetch_accessions():
    # Paths are relative to the script location or project root
    # We assume we are running from project root
    tool_path = os.path.join("raw-data", "NCBI", "datasets.exe")
    output_file = os.path.join("raw-data", "NCBI", "all_complete_accessions.txt")
    
    print("Fetching all complete virus accessions from NCBI...")
    # Note: --complete-only ensures we only get complete genomes
    cmd = [tool_path, "summary", "virus", "genome", "taxon", "viruses", "--complete-only", "--limit", "all", "--as-json-lines"]
    
    try:
        # We use a pipe to stream the output since it can be very large
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        count = 0
        with open(output_file, "w") as f:
            for line in process.stdout:
                if line.strip():
                    try:
                        data = json.loads(line)
                        f.write(data['accession'] + "\n")
                        count += 1
                        if count % 1000 == 0:
                            print(f"Found {count} accessions...")
                    except Exception as e:
                        continue
        
        process.wait()
        if process.returncode != 0:
            print(f"Error: {process.stderr.read()}")
        else:
            print(f"Done! Saved {count} accessions to {output_file}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    fetch_accessions()
