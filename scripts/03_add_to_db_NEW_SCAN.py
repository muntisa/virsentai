#!/usr/bin/env python3
"""
Step 3: Add Scan Predictions to Database

Reads virus prediction data from scan results and inserts it
into the 'predictions' table in the SQLite database.

Usage:
    python 03_add_to_db_NEW_SCAN.py --input virus-scan/virus_scan_2026-04-16_to_2026-05-04_RefSeq_predictions.tsv

Input:
    - TSV file with prediction data (must contain PClass_1 column)
    - SQLite database at db/virsentai.sqlite3

Output:
    - Records added to db/virsentai.sqlite3 predictions table

Column Mapping:
    - created_at → Registration_Date
    - accession → Virus_ID
    - organism → Virus_Name
    - host → Host
    - source → Database
    - PClass_1 → prediction_score

Notes:
    - Same as 03_add_to_db_NEW.py but for scan results
    - Logs to logs/03_add_to_db_NEW_SCAN_<timestamp>.log
"""
import os
import csv
import logging
import argparse
from datetime import datetime

import pandas as pd
csv.field_size_limit(2000000)

import sqlite3

from config import *

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

parser = argparse.ArgumentParser(description="Add predictions to SQLite database")
parser.add_argument(
    "--input",
    required=True,
    help="Path to the input TSV file with predictions.",
)
args = parser.parse_args()

input_tsv_path = os.path.abspath(args.input)
modelid = DEFAULT_MODEL_ID

script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%MM%SS")
os.makedirs(LOG_DIR, exist_ok=True)
log_file_name = os.path.join(LOG_DIR, f"{script_name}_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_name),
        logging.StreamHandler(),
    ],
)


def add_predictions_to_db(tsv_file_path, sqlite_db_path, modelid):
    logging.info(f"Reading predictions from: {tsv_file_path}")

    try:
        df = pd.read_csv(tsv_file_path, sep="\t")
        logging.info(f"Found {len(df):,} predictions to process.")

        logging.info(f"Connecting to database: {sqlite_db_path}")
        conn = sqlite3.connect(sqlite_db_path)
        cursor = conn.cursor()

        insert_query = """
        INSERT INTO predictions (
            prediction_date,
            prediction_score,
            model_id,
            virus_id,
            virus_name,
            virus_host,
            virus_db,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """

        column_mapping = {
            "Registration_Date": "created_at",
            "Virus_ID": "accession",
            "Virus_Name": "organism",
            "Host": "host",
            "Database": "source",
        }

        for index, row in df.iterrows():
            prediction_data = (
                row["Registration_Date"],
                row["PClass_1"],
                modelid,
                row["Virus_ID"],
                row["Virus_Name"],
                row["Host"],
                row["Database"],
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            cursor.execute(insert_query, prediction_data)

        conn.commit()
        conn.close()

        logging.info(f"Successfully inserted {len(df):,} records into the database.")
        print(f"Successfully inserted {len(df):,} records into the database.")

    except FileNotFoundError:
        logging.error(f"Error: The file was not found at {tsv_file_path}")
    except KeyError as e:
        logging.error(f"Error: Missing column in TSV file: {e}")
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    if not os.path.exists(input_tsv_path):
        print(f"Error: Input file not found: {input_tsv_path}")
        exit(1)

    print(f"Adding predictions from: {input_tsv_path}")
    print(f"Using model ID: {modelid}")
    add_predictions_to_db(input_tsv_path, SQLite_PRED_FILE, modelid=modelid)

    print("\n--- Verification ---")
    try:
        conn = sqlite3.connect(SQLite_PRED_FILE)
        verify_query = "SELECT * FROM predictions ORDER BY prediction_id DESC LIMIT 5;"
        df_verify = pd.read_sql_query(verify_query, conn)
        conn.close()

        if not df_verify.empty:
            print("Last 5 predictions:")
            print(df_verify.to_string())
        else:
            print("No predictions found in the table.")
    except sqlite3.Error as e:
        print(f"Database error: {e}")