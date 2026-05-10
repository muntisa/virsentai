#!/usr/bin/env python3
"""
NCBI Taxonomy Fetcher

Fetches NCBI taxonomy data for organisms one at a time. Extracts unique hosts
from the non-human dataset and fetches their taxonomy information.

Usage:
    python DS-taxonomy_fetcher.py

Input:
    - ds/ds_160k_nonHuman.tsv OR
    - ds/organisms_without_taxonomy.txt (list of organisms)

Output:
    - ds/hosts_taxonomy.tsv

Output Columns:
    name, taxid, rank, superkingdom, kingdom, phylum, class, order, family, subfamily, genus, species

Process:
    1. Read unique host values from input
    2. Save host list to ds/hosts.txt
    3. Fetch taxonomy data from NCBI for each host
    4. Output to ds/hosts_taxonomy.tsv

Requirements:
    - biopython
    - config.py with NCBI_ENTREZ_EMAIL and NCBI_API_KEY
    - Rate limit: 0.1s between requests

Notes:
    - Processes organisms one by one
    - Supports incremental processing
    - Logs to logs/ directory
"""

import argparse
import csv
import sys
import time
import logging
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

from Bio import Entrez

# Load config for defaults
sys.path.append(str(Path(__file__).parent))
try:
    from config import NCBI_ENTREZ_EMAIL, NCBI_API_KEY, LOG_PATH
except ImportError:
    NCBI_ENTREZ_EMAIL = "c.munteanu@udc.es"
    NCBI_API_KEY = None
    LOG_PATH = "logs"

DEFAULT_INPUT = "ds/organisms_without_taxonomy.txt"
DEFAULT_OUTPUT = "ds/missing_organism_taxonomy.tsv"

def setup_logging(script_name):
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_PATH, f"Taxonomy_{script_name}_{timestamp}.log")

    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(log_file, "w", encoding="utf-8")
            self.at_start_of_line = True

        def write(self, message):
            if not message: return
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

        def flush(self): pass

    sys.stdout = Logger()
    sys.stderr = sys.stdout
    print(f"Logging started: {log_file}")
    return log_file

class LogWrapper:
    def info(self, msg, *args):
        if args:
            print("  " + (msg % args))
        else:
            print("  " + msg)
    def warning(self, msg, *args):
        if args:
            print("  " + (msg % args), file=sys.stderr)
        else:
            print("  " + msg, file=sys.stderr)
    def error(self, msg, *args):
        if args:
            print("  " + (msg % args), file=sys.stderr)
        else:
            print("  " + msg, file=sys.stderr)

log = LogWrapper()

def get_taxonomy_fallback(name: str) -> dict:
    """Get taxonomy using pattern matching when NCBI lookup fails."""
    import re
    row = {col: "" for col in OUTPUT_COLUMNS}
    row["name"] = name
    name_lower = name.lower()

    TAXONOMY_PATTERNS = [
        (["escherichia", "e. coli", "coli o", "coli"], ("Bacteria", "Bacteria", "Proteobacteria", "Gammaproteobacteria", "Enterobacterales", "Enterobacteriaceae", "Escherichia", "Escherichia coli")),
        (["salmonella", "enterica"], ("Bacteria", "Bacteria", "Proteobacteria", "Gammaproteobacteria", "Enterobacterales", "Enterobacteriaceae", "Salmonella", "Salmonella enterica")),
        (["yersinia", "enterocolitica"], ("Bacteria", "Bacteria", "Proteobacteria", "Gammaproteobacteria", "Enterobacterales", "Enterobacteriaceae", "Yersinia", "Yersinia enterocolitica")),
        (["bacillus"], ("Bacteria", "Bacteria", "Firmicutes", "Bacilli", "Bacillales", "Bacillaceae", "Bacillus", "Bacillus species")),
        (["aedes", "culex", "anopheles", "mosquito"], ("Eukaryota", "Animals", "Arthropoda", "Insecta", "Diptera", "Culicidae", "Aedes", "Aedes species")),
        (["ixodes", "tick"], ("Eukaryota", "Animals", "Arthropoda", "Insecta", "Ixodida", "Ixodidae", "Ixodes", "Ixodes species")),
        (["homo sapiens", "human"], ("Eukaryota", "Animals", "Chordata", "Mammalia", "Primates", "Hominidae", "Homo", "Homo sapiens")),
        (["mus", "mouse", "murine"], ("Eukaryota", "Animals", "Chordata", "Mammalia", "Rodentia", "Muridae", "Mus", "Mus species")),
        (["rat", "rattus"], ("Eukaryota", "Animals", "Chordata", "Mammalia", "Rodentia", "Muridae", "Rattus", "Rattus species")),
        (["bat", "chiroptera"], ("Eukaryota", "Animals", "Chordata", "Mammalia", "Chiroptera", "Vespertilionidae", "Myotis", "Myotis species")),
        (["bird", "avian", "passer", "sparrow"], ("Eukaryota", "Animals", "Chordata", "Aves", "Passeriformes", "Passeridae", "Passer", "Passer domesticus")),
        (["guinea pig", "cavia"], ("Eukaryota", "Animals", "Chordata", "Mammalia", "Rodentia", "Caviidae", "Cavia", "Cavia porcellus")),
        (["rabbit", "oryctolagus"], ("Eukaryota", "Animals", "Chordata", "Mammalia", "Lagomorpha", "Leporidae", "Oryctolagus", "Oryctolagus cuniculus")),
        (["pig", "sus scrofa", "swine"], ("Eukaryota", "Animals", "Chordata", "Mammalia", "Artiodactyla", "Suidae", "Sus", "Sus scrofa")),
        (["cattle", "cow", "bos"], ("Eukaryota", "Animals", "Chordata", "Mammalia", "Artiodactyla", "Bovidae", "Bos", "Bos taurus")),
        (["sheep", "ovis"], ("Eukaryota", "Animals", "Chordata", "Mammalia", "Artiodactyla", "Bovidae", "Ovis", "Ovis aries")),
        (["goat", "capra"], ("Eukaryota", "Animals", "Chordata", "Mammalia", "Artiodactyla", "Bovidae", "Capra", "Capra hircus")),
        (["chicken", "gallus"], ("Eukaryota", "Animals", "Chordata", "Aves", "Galliformes", "Phasianidae", "Gallus", "Gallus gallus")),
        (["duck", "anas"], ("Eukaryota", "Animals", "Chordata", "Aves", "Anseriformes", "Anatidae", "Anas", "Anas platyrhynchos")),
        (["frog", "xenopus", "rana"], ("Eukaryota", "Animals", "Chordata", "Amphibia", "Anura", "Pipidae", "Xenopus", "Xenopus laevis")),
        (["fish", "zebrafish", "danio"], ("Eukaryota", "Animals", "Chordata", "Actinopterygii", "Cypriniformes", "Cyprinidae", "Danio", "Danio rerio")),
        (["cat", "felis"], ("Eukaryota", "Animals", "Chordata", "Mammalia", "Carnivora", "Felidae", "Felis", "Felis catus")),
        (["dog", "canis"], ("Eukaryota", "Animals", "Chordata", "Mammalia", "Carnivora", "Canidae", "Canis", "Canis lupus")),
        (["horse", "equus"], ("Eukaryota", "Animals", "Chordata", "Mammalia", "Perissodactyla", "Equidae", "Equus", "Equus caballus")),
        (["plant", "arabidopsis", "rice", "wheat", "corn", "maize", "tomato"], ("Eukaryota", "Plants", "Streptophyta", "Magnoliopsida", "Rosales", "Malvaceae", "Unknown", "Unknown species")),
        (["fungi", "yeast", "saccharomyces"], ("Eukaryota", "Fungi", "Ascomycota", "Saccharomycetes", "Saccharomycetales", "Saccharomycetaceae", "Saccharomyces", "Saccharomyces cerevisiae")),
        (["alectorobius"], ("Eukaryota", "Animals", "Arthropoda", "Insecta", "Ixodida", "Argasidae", "Alectorobius", "Alectorobius species")),
        (["austrosimulium"], ("Eukaryota", "Animals", "Arthropoda", "Insecta", "Diptera", "Simuliidae", "Austrosimulium", "Austrosimulium species")),
        (["mops"], ("Eukaryota", "Animals", "Arthropoda", "Insecta", "Diptera", "Culicidae", "Mops", "Mops species")),
        (["morganella"], ("Bacteria", "Bacteria", "Proteobacteria", "Gammaproteobacteria", "Enterobacterales", "Enterobacteriaceae", "Morganella", "Morganella species")),
        (["rhizophagus"], ("Eukaryota", "Fungi", "Glomeromycota", "Glomeromycetes", "Glomerales", "Glomeraceae", "Rhizophagus", "Rhizophagus species")),
        (["sida"], ("Eukaryota", "Plants", "Streptophyta", "Magnoliopsida", "Malvales", "Malvaceae", "Sida", "Sida species")),
        (["trichoprosopon"], ("Eukaryota", "Animals", "Arthropoda", "Insecta", "Diptera", "Culicidae", "Trichoprosopon", "Trichoprosopon species")),
        (["aethomys"], ("Eukaryota", "Animals", "Chordata", "Mammalia", "Rodentia", "Muridae", "Aethomys", "Aethomys species")),
    ]

    matched = False
    for pattern_list, taxonomy in TAXONOMY_PATTERNS:
        if any(p in name_lower for p in pattern_list):
            row["superkingdom"] = taxonomy[0]
            row["kingdom"] = taxonomy[1]
            row["phylum"] = taxonomy[2]
            row["class"] = taxonomy[3]
            row["order"] = taxonomy[4]
            row["family"] = taxonomy[5]
            row["genus"] = taxonomy[6]
            row["species"] = taxonomy[7]
            matched = True
            break

    if not matched:
        parts = name.split()
        if parts:
            row["genus"] = parts[0]
        row["notes"] = "Not found in NCBI Taxonomy"
    else:
        row["rank"] = "species" if row.get("species") else "genus"
        lineage = []
        for col in ["kingdom", "phylum", "class", "order", "family", "genus"]:
            if row.get(col):
                lineage.append(row[col])
        if lineage:
            row["lineage_summary"] = " > ".join(lineage)
        row["notes"] = "Found via pattern matching"

    return row

# Map NCBI rank variations to our standard ranks
RANK_MAP = {
    "domain": "superkingdom",
    "superkingdom": "superkingdom",
    "kingdom": "kingdom",
    "subkingdom": "kingdom",
    "phylum": "phylum",
    "subphylum": "subphylum",
    "class": "class",
    "subclass": "subclass",
    "order": "order",
    "suborder": "suborder",
    "family": "family",
    "subfamily": "subfamily",
    "genus": "genus",
    "subgenus": "subgenus",
    "species": "species",
    "subspecies": "species",
    "strain": "species",
    "variety": "species",
    "subvariety": "species",
    "clade": "kingdom",   # NCBI uses "clade" for many taxonomic groups
    "no rank": "",
}

# Taxonomic ranks we want to capture (mapped from NCBI rank names)
TARGET_RANKS = [
    "superkingdom",   
    "kingdom",        
    "phylum",        
    "subphylum",
    "class",         
    "subclass",
    "order",
    "suborder",
    "family",
    "subfamily",
    "genus",
    "subgenus",
    "species",
]

OUTPUT_COLUMNS = (
    ["name", "taxid", "rank"]
    + TARGET_RANKS
)

# NCBI helpers
def search_taxid(name: str, retries: int = 3, delay: float = 1.0) -> Optional[str]:
    for attempt in range(retries):
        try:
            handle = Entrez.esearch(db="taxonomy", term=name, retmax=1)
            record = Entrez.read(handle)
            handle.close()
            ids = record.get("IdList", [])
            return ids[0] if ids else None
        except Exception as exc:
            log.warning("esearch attempt %d/%d for '%s': %s", attempt + 1, retries, name, exc)
            time.sleep(delay * (attempt + 1))
    return None

def fetch_tax_record(taxid: str, retries: int = 3, delay: float = 1.0) -> Optional[dict]:
    for attempt in range(retries):
        try:
            handle = Entrez.efetch(db="taxonomy", id=taxid, retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            return records[0] if records else None
        except Exception as exc:
            log.warning("efetch attempt %d/%d for taxid %s: %s", attempt + 1, retries, taxid, exc)
            time.sleep(delay * (attempt + 1))
    return None

def parse_lineage(record: dict) -> dict:
    result = {rank: "" for rank in TARGET_RANKS}
    for node in record.get("LineageEx", []):
        rank = node.get("Rank", "").lower().strip()
        sci = node.get("ScientificName", "").strip()
        # Map NCBI rank to our standard rank
        mapped_rank = RANK_MAP.get(rank, "")
        if mapped_rank and mapped_rank in result:
            # Only fill if empty (take first occurrence)
            if not result[mapped_rank]:
                result[mapped_rank] = sci
    
    # Also check the record's own rank
    own_rank = record.get("Rank", "").lower().strip()
    own_name = record.get("ScientificName", "").strip()
    mapped_own_rank = RANK_MAP.get(own_rank, "")
    if mapped_own_rank and mapped_own_rank in result and not result[mapped_own_rank]:
        result[mapped_own_rank] = own_name
    
    return result

def lineage_summary(record: dict) -> str:
    raw = record.get("Lineage", "")
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    if len(parts) > 6:
        parts = [parts[0], ".."] + parts[-5:]
    return " > ".join(parts)

def process(delay: float) -> None:
    input_path = 'ds/organisms_without_taxonomy.txt'
    print(f"Reading organisms from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        unique_hosts = [line.strip() for line in f if line.strip()]
    
    print(f"Number of organisms to process: {len(unique_hosts)}")

    output_path = Path('ds/missing_organism_taxonomy.tsv')
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()

        start_time = time.time()
        for idx, name in enumerate(unique_hosts, start=1):
            percent = int((idx / len(unique_hosts)) * 100)
            log.info("[%d/%d %3d%%] %s", idx, len(unique_hosts), percent, name)
            row = {col: "" for col in OUTPUT_COLUMNS}
            row["name"] = name

            taxid = search_taxid(name, delay=delay)
            time.sleep(delay)

            if taxid is None:
                log.warning("  -> not found in NCBI, skipped")
                continue

            record = fetch_tax_record(taxid, delay=delay)
            time.sleep(delay)

            if record is None:
                log.warning("  -> fetch failed for taxid %s, skipped", taxid)
                continue

            tax = parse_lineage(record)
            row["taxid"] = record.get("TaxId", taxid)
            row["rank"] = record.get("Rank", "")

            for rank in TARGET_RANKS:
                row[rank] = tax.get(rank, "")

            writer.writerow(row)
            log.info("  -> taxid=%s  rank=%s  family=%s", row["taxid"], row["rank"], row["family"])

    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds}s"
    else:
        time_str = f"{seconds}s"
    log.info("Done - results written to '%s'", output_path)
    log.info("Total running time: %s", time_str)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map taxonomy for virus hosts from NCBI."
    )
    parser.add_argument("--email", default=NCBI_ENTREZ_EMAIL)
    parser.add_argument("--api-key", default=NCBI_API_KEY)
    parser.add_argument("--delay", type=float, default=0.1)
    args = parser.parse_args()

    Entrez.email = args.email
    if args.api_key:
        Entrez.api_key = args.api_key
        log.info("API key set - using up to 10 req/s")
    else:
        log.info("No API key - limited to ~3 req/s")

    process(delay=args.delay)

if __name__ == "__main__":
    main()