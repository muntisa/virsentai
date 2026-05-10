#!/usr/bin/env python3
"""
Raw Database Cleaner

Cleans and deduplicates the raw virus database (db/raw-viruses.sqlite3)
to produce a refined database (db/db-viruses.sqlite3) with higher quality data.

Usage:
    python DB-clean_raw_sqlite.py

Input:
    - db/raw-viruses.sqlite3 (raw viruses table)

Output:
    - db/db-viruses.sqlite3 (cleaned viruses table)
    - db/db-viruses.tsv (exported TSV)

Processing Steps:
1. Remove invalid sequences (not A,T,G,C only)
2. Collapse RefSeq accessions to highest version
3. Deduplicate by accession (different sources)
4. Deduplicate by identical sequences
5. Normalize host values and correct human host:
   - SPLIT: delimiters between hosts (, ; | /) - protect strain patterns (H3N2/2009)
   - REMOVE: strange characters, age text, passage, accession suffixes
   - CORRECT: if human in any value -> "Homo sapiens"
   - AGE-ONLY: if value is age only (e.g., "68 years old") -> "Homo sapiens"
6. Replace NULL/empty/invalid host values with "Unknown"
   - Invalid: env, environment, environmental, layer, root, marc 145, not available, rd
7. Insert to new database
8. Export to TSV

Source Priority (highest to lowest):
1. VirusHostDB (most curated, rich metadata)
2. BVBRC (includes host info from VIPR)
3. RefSeq (generic reference, minimal host info)

Duplicate Resolution:
- Prefer higher priority source
- Prefer records with collection_date
- Prefer records with complete host values

Configuration:
    - SQLITE_CORR_VIRUSES_FILE: Output database path
    - TSV_CORR_VIRUSES_FILE: Output TSV path
"""

import os
import sys
import sqlite3
import pandas as pd
import re
import unicodedata
from datetime import datetime
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

OUTPUT_DB = SQLITE_CORR_VIRUSES_FILE
OUTPUT_TSV = TSV_CORR_VIRUSES_FILE

def setup_logging(script_name):
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_PATH, f"Process_{script_name}_{timestamp}.log")

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

def get_accession_base(accession):
    if pd.isna(accession):
        return None
    return str(accession).split('.')[0]

def get_version(accession):
    if pd.isna(accession):
        return 0
    parts = str(accession).split('.')
    try:
        return int(parts[1]) if len(parts) > 1 else 0
    except:
        return 0

def is_host_complete(host):
    if pd.isna(host):
        return False
    host_str = str(host).strip().lower()
    return host_str != '' and host_str != 'unknown' and host_str != 'na'

def get_collection_date_score(date):
    if pd.isna(date) or str(date).strip() == '':
        return 0
    return 1

def is_valid_sequence(seq):
    if pd.isna(seq) or seq == '':
        return False
    seq_upper = str(seq).upper()
    valid_chars = set('ATGC')
    return all(c in valid_chars for c in seq_upper if c not in ' \n\t\r')

def print_host_stats(df, label=""):
    if label:
        print(f"\n{label}")
    print("Host\tCount")
    human = len(df[df['host'] == 'Homo sapiens'])
    unknown = len(df[df['host'] == 'Unknown'])
    rest = len(df) - human - unknown
    print(f"Homo sapiens\t{human}")
    print(f"Unknown\t{unknown}")
    print(f"Rest\t{rest}")
    print(f"Total\t{len(df)}")

def process_db():
    setup_logging("db_process")

    print(f"Loading data from {SQLITE_VIRUSES_FILE}...")
    conn = sqlite3.connect(SQLITE_VIRUSES_FILE)
    df = pd.read_sql("SELECT * FROM viruses", conn)
    conn.close()

    print(f"Loaded {len(df)} records")
    print(f"  BVBRC: {len(df[df['source'] == 'BVBRC'])}")
    print(f"  RefSeq: {len(df[df['source'] == 'RefSeq'])}")
    print(f"  VirusHostDB: {len(df[df['source'] == 'VirusHostDB'])}")

    print_host_stats(df, "\nHost stats before cleaning:")

    print("\n=== Null values (missing values) per field ===")
    for col in df.columns:
        null_count = df[col].isna().sum()
        print(f"{col}: {null_count}")

    print("\n=== Step 1: Remove invalid sequences (not A,T,G,C only) ===")
    df['valid_seq'] = df['sequence'].apply(is_valid_sequence)
    invalid_count = len(df[~df['valid_seq']])
    print(f"Records with invalid sequences: {invalid_count}")
    df = df[df['valid_seq']].drop(columns=['valid_seq'])
    print(f"After Step 1: {len(df)} records")
    print_host_stats(df, "Host stats after Step 1:")

    print("\n=== Step 2: Collapse RefSeq accessions to highest version ===")
    refseq_df = df[df['source'] == 'RefSeq'].copy()
    refseq_df['accession_base'] = refseq_df['accession'].apply(get_accession_base)
    refseq_df['version'] = refseq_df['accession'].apply(get_version)
    refseq_df = refseq_df.sort_values('version', ascending=False)
    refseq_df = refseq_df.drop_duplicates(subset=['organism', 'accession_base'], keep='first')
    refseq_df['accession'] = refseq_df['accession_base']
    refseq_df = refseq_df.drop(columns=['accession_base', 'version'])
    df = pd.concat([df[df['source'] != 'RefSeq'], refseq_df])
    print(f"After Step 2: {len(df)} records")
    print_host_stats(df, "Host stats after Step 2:")

    print("\n=== Step 3: Deduplicate by accession (different sources) ===")
    df['accession_base'] = df['accession'].apply(get_accession_base)
    df['date_score'] = df['collection_date'].apply(get_collection_date_score)
    df['host_complete'] = df['host'].apply(is_host_complete).astype(int)
    df['source_order'] = df['source'].map({'VirusHostDB': 0, 'BVBRC': 1, 'RefSeq': 2})
    df = df.sort_values(['source_order', 'date_score', 'host_complete'], ascending=[True, False, False])
    df = df.drop_duplicates(subset=['accession_base'], keep='first')
    df = df.drop(columns=['accession_base', 'date_score', 'host_complete', 'source_order'])
    print(f"After Step 3: {len(df)} records")
    print_host_stats(df, "Host stats after Step 3:")

    print("\n=== Step 4: Deduplicate by identical sequences ===")
    print("Converting sequences to uppercase...")
    df['sequence'] = df['sequence'].str.upper()
    df['seq_normalized'] = df['sequence'].fillna('').str.replace(r'\s+', '', regex=True)
    df['date_score'] = df['collection_date'].apply(get_collection_date_score)
    df['host_complete'] = df['host'].apply(is_host_complete).astype(int)
    df['source_order'] = df['source'].map({'VirusHostDB': 0, 'BVBRC': 1, 'RefSeq': 2})
    df = df.sort_values(['source_order', 'date_score', 'host_complete'], ascending=[True, False, False])
    df = df.drop_duplicates(subset=['seq_normalized'], keep='first')
    df = df.drop(columns=['seq_normalized', 'date_score', 'host_complete', 'source_order'])
    print(f"After Step 4: {len(df)} records")
    print_host_stats(df, "Host stats after Step 4:")

    print("\n=== Step 5: Normalize of host values and correction of human host ===")

    PLACEHOLDER = "__PROTECTED__"

    HUMAN_HOST_VALUES = {

        # -------------------------------------------------------------------------
        # CANONICAL / TAXONOMY (NCBI, VirHostDB, BV-BRC)
        # -------------------------------------------------------------------------
        "homo sapiens",              # NCBI standard; BV-BRC host_name; VirHostDB display
        "homo sapiens sapiens",      # Subspecies designation
        "9606",                      # NCBI Taxonomy ID; VirHostDB host_tax_id
        "taxid:9606",                # NCBI prefixed tax ID
        "taxon:9606",                # VirHostDB / GenBank alternate prefix
        "tax_id:9606",               # Alternative separator format
        "ncbi:9606",                 # Source-prefixed variant
        "homo sapiens (human)",      # UniProt display style

        # -------------------------------------------------------------------------
        # COMMON NAME VARIANTS (NCBI GenBank free-text; BV-BRC host_common_name)
        # -------------------------------------------------------------------------
        "human",                     # BV-BRC host_common_name canonical
        "humans",
        "human being",
        "human beings",
        "human subject",             # Clinical submissions
        "human subjects",            # Plural clinical
        "human host",                # Compound form found in GenBank
        "human sample",              # BioSample informal
        "human specimen",            # Clinical specimen notation

        # -------------------------------------------------------------------------
        # CLINICAL / EPIDEMIOLOGICAL (NCBI BioSample; BV-BRC host_health context)
        # -------------------------------------------------------------------------
        "patient",
        "patients",                  # Plural
        "person",
        "persons",                   # Plural
        "individual",                # Epidemiological term
        "individuals",               # Plural
        "clinical",                  # Shorthand in some BioSample records
        "clinical isolate",          # BioSample / BV-BRC raw value
        "volunteer",                 # Trial/study submissions

        # -------------------------------------------------------------------------
        # ABBREVIATED FORMS (GenBank submitter shortcuts)
        # -------------------------------------------------------------------------
        "h. sapiens",
        "h.sapiens",
        "h sapiens",
        "h.s.",                      # Ultra-abbreviated

        # -------------------------------------------------------------------------
        # SEX-SPECIFIC VARIANTS (NCBI GenBank; BV-BRC host_sex combinations)
        # -------------------------------------------------------------------------
        "human male",
        "human female",
        "male human",                # Inverted form
        "female human",              # Inverted form
        "human adult",
        "human infant",
        "human child",
        "homo sapiens male",         # Full binomial + sex
        "homo sapiens female",       # Full binomial + sex

        # -------------------------------------------------------------------------
        # AGE / DEMOGRAPHIC VARIANTS (NCBI BioSample free-text)
        # -------------------------------------------------------------------------
        "adult",                     # Context-dependent; common in clinical
        "adult human",               # Explicit adult qualifier
        "pediatric",                 # Age group
        "pediatric patient",         # Compound clinical term
        "neonate",                   # Age group
        "neonatal",                  # Adjectival form
        "elderly",                   # Age qualifier
        "elderly human",             # Compound

        # -------------------------------------------------------------------------
        # CASE / FORMAT ERRORS (normalization handled by .lower(), included for doc)
        # -------------------------------------------------------------------------
        "homo sapien",               # Missing trailing 's' — very common typo
        "homo_sapiens",              # Underscore separator
        "homosapiens",               # Missing space
        "homo-sapiens",              # Hyphen separator
        "humen",                     # Misspelling
        "humon",                     # Misspelling
        "homo sapeins",              # Transposition typo
        "homo spaiens",              # Transposition typo
        "homo sapein",               # Combined misspelling
        "h.sapien",                  # Abbreviated + missing 's'

        # -------------------------------------------------------------------------
        # NON-ENGLISH VARIANTS (found in GenBank international submissions)
        # -------------------------------------------------------------------------

        # Spanish
        "humano",
        "humana",
        "ser humano",
        "ser humana",                # Feminine form
        "homo sapiens (humano)",     # Spanish parenthetical

        # French
        "humain",
        "humaine",                   # Feminine form
        "être humain",               #
        "etre humain",               # Without accent

        # German
        "mensch",
        "menschen",
        "homo sapiens (mensch)",     # German parenthetical

        # Italian
        "uomo",
        "essere umano",              #
        "umano",                     #

        # Portuguese
        "humanos",                   # Plural

        # Japanese
        "ヒト",                       # katakana
        "人間",                        # kanji, 'ningen'
        "人類",                        # kanji, 'jinrui'

        # Chinese (Simplified)
        "人",                         # single character
        "人类",                        # 'renlei'
        "人體",                        # body/organism, traditional

        # Russian
        "человек",                   #

        # -------------------------------------------------------------------------
        # BV-BRC SPECIFIC (after HostAnnotation pipeline; controlled vocabulary)
        # -------------------------------------------------------------------------
        "human host",                # BV-BRC surveillance field
    }

    def protect_strain_patterns(text):
        if pd.isna(text):
            return "", []
        text = str(text)
        strain_pattern = re.compile(r"\b[Hh]\d[Nn]\d(?:/\d+)?\b")
        strains = strain_pattern.findall(text)
        protected = strain_pattern.sub(PLACEHOLDER, text)
        return protected, strains

    def restore_strain_patterns(text, strains):
        result = text
        for strain in strains:
            result = result.replace(PLACEHOLDER, strain, 1)
        return result

    def split_host_field(raw):
        if pd.isna(raw):
            return []
        protected, strains = protect_strain_patterns(raw)
        parts = re.split(r"[,;/|]", protected)
        result = []
        strain_idx = 0
        for part in parts:
            sp = part.strip()
            if not sp:
                continue
            if PLACEHOLDER in sp and strain_idx < len(strains):
                sp = sp.replace(PLACEHOLDER, strains[strain_idx], 1)
                strain_idx += 1
            result.append(sp)
        return result

    def clean_host_token(token):
        protected, strains = protect_strain_patterns(token)
        s = protected
        s = unicodedata.normalize("NFC", s)
        s = s.replace("\ufeff", "")
        s = s.replace("\u200b", "")
        s = s.replace("\u00a0", " ")
        s = s.replace("\t", " ")
        s = s.replace("\n", " ")
        s = s.replace("\r", " ")
        s = re.sub(r"^(taxid|taxon|tax_id|ncbi|host)\s*=\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(
            r"\b\d+\s*[-]?\s*\d*\s*(year[s]?\s*old|months?\s*old|weeks?\s*old|days?\s*old)\b",
            "", s, flags=re.IGNORECASE
        )
        s = re.sub(r"\bP\d+\b", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\b[NAX][CP]_\d+\b", "", s, flags=re.IGNORECASE)
        s = s.replace("(", " ").replace(")", " ")
        s = s.replace("[", " ").replace("]", " ")
        s = s.replace("_", " ")
        s = s.replace("-", " ")
        s = re.sub(r"[\"'`\\|{}*#?=\t\n\r]+", "", s)
        s = re.sub(r"\s+", " ", s)
        s = restore_strain_patterns(s, strains)
        s = s.strip()
        return s

    def normalize_host(host_value, original=None):
        if pd.isna(host_value):
            return None
        host_str = str(host_value).strip().lower()
        if host_str in HUMAN_HOST_VALUES:
            return "Homo sapiens"

        check_str = str(original).lower() if original else host_str
        AGE_ONLY_PATTERN = re.compile(
            r"^\s*\d[\d.,]*\s*(year|yr|y\.?o\.?|y\.?o|yrs?|month|mo|mos?|wk|day)s?\s*(old)?\s*$",
            re.IGNORECASE
        )
        if AGE_ONLY_PATTERN.match(check_str):
            return "Homo sapiens"

        words = host_str.split()
        for word in words:
            if word in HUMAN_HOST_VALUES:
                return "Homo sapiens"
        return host_value

    def process_host_field(raw):
        parts = split_host_field(raw)
        results = []
        for part in parts:
            cleaned = clean_host_token(part)
            normalized = normalize_host(cleaned, original=part)
            if normalized:
                results.append(normalized)
        seen = set()
        deduped = []
        for r in results:
            r_lower = r.lower()
            if r_lower not in seen:
                seen.add(r_lower)
                deduped.append(r)
        has_human = any(r == "Homo sapiens" for r in deduped)
        if has_human:
            return "Homo sapiens"
        if deduped:
            return deduped[0].lower()
        return None

    host_changes = {}
    original_hosts = df['host'].copy()
    df['host'] = df['host'].apply(process_host_field)

    for orig, corr in zip(original_hosts, df['host']):
        if pd.notna(orig) and str(orig).strip():
            key = (str(orig).strip(), corr if corr else "None")
            host_changes[key] = host_changes.get(key, 0) + 1

    for (orig, corr), count in sorted(host_changes.items(), key=lambda x: -x[1]):
        print(f"{orig} - {corr}")

    print_host_stats(df, "\nHost stats after Step 5:")

    print("\n=== Step 6: Replace NULL/empty host values with 'Unknown' ===")
    null_empty_before = len(df[df['host'].isna() | (df['host'].astype(str).str.strip() == '')])
    df['host'] = df['host'].fillna('Unknown')
    df.loc[df['host'].astype(str).str.strip() == '', 'host'] = 'Unknown'
    print(f"Replaced {null_empty_before} NULL/empty host values with 'Unknown'")

    INVALID_HOSTS = ['env', 'environment', 'environmental', 'layer', 'root', 'marc 145', 'not available', 'rd']
    invalid_count = len(df[df['host'].isin(INVALID_HOSTS)])
    df.loc[df['host'].isin(INVALID_HOSTS), 'host'] = 'Unknown'
    print(f"Replaced {invalid_count} invalid host values with 'Unknown': {INVALID_HOSTS}")

    print_host_stats(df, "Host stats after Step 6:")
    print(f"Replaced {null_empty_before + invalid_count} host values total")

    print("\n=== Step 7: Insert to new database ===")
    if os.path.exists(OUTPUT_DB):
        os.remove(OUTPUT_DB)

    db_dir = os.path.dirname(OUTPUT_DB)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)

    conn = sqlite3.connect(OUTPUT_DB)
    cursor = conn.cursor()
    cursor.executescript(SQLITE_VIRUSES_SCHEMA)
    conn.commit()

    columns = ['db_id', 'accession', 'sequence', 'length', 'source', 'collection_date',
             'country', 'host', 'organism', 'segment', 'completeness']
    df = df[columns]

    df.to_sql('viruses', conn, if_exists='append', index=False)
    conn.commit()

    cursor.execute("SELECT source, COUNT(*) FROM viruses GROUP BY source")
    counts = dict(cursor.fetchall())

    print(f"\n=== Final records ===")
    print(f"  BVBRC: {counts.get('BVBRC', 0)}")
    print(f"  RefSeq: {counts.get('RefSeq', 0)}")
    print(f"  VirusHostDB: {counts.get('VirusHostDB', 0)}")
    print(f"  Total: {len(df)}")

    print_host_stats(df, "\n=== Host summary ===")

    print("\n=== Step 8: Exporting to TSV ===")
    df.to_csv(OUTPUT_TSV, sep="\t", index=False)
    print(f"Exported to {OUTPUT_TSV}")

    conn.close()
    print("\nDone!")

if __name__ == "__main__":
    process_db()