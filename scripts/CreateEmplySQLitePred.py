#!/usr/bin/env python3
"""
Create Empty Prediction Database

Creates an empty SQLite database for storing predictions with the following tables:
    - predictions: Stores virus prediction results
    - models: Stores model metadata
    - chembl_approved_drug: Stores drug information
    - PLAPT_AE: Stores PLAPT adverse events data

Usage:
    python CreateEmplySQLitePred.py

Output:
    - db/virsentai.sqlite3 (created if not exists)

Notes:
    - All schema definitions imported from config.py
    - Creates database directory if it doesn't exist
    - Verifies table creation after setup
"""

import os
import sqlite3

from config import SQLite_PRED_FILE, PREDICTION_TABLE_QUERY, MODELS_TABLE_QUERY, DRUGS_TABLE_QUERY, PLAPT_TABLE_QUERY


def create_database(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    print(f"Creating database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("Creating 'predictions' table...")
    cursor.execute(PREDICTION_TABLE_QUERY)

    print("Creating 'models' table...")
    cursor.execute(MODELS_TABLE_QUERY)

    print("Creating 'chembl_approved_drug' table...")
    cursor.execute(DRUGS_TABLE_QUERY)

    print("Creating 'PLAPT_AE' table...")
    cursor.execute(PLAPT_TABLE_QUERY)

    conn.commit()

    print("\nVerifying tables...")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables created: {[t[0] for t in tables]}")

    conn.close()
    print(f"\nDatabase created successfully: {db_path}")


if __name__ == "__main__":
    if os.path.exists(SQLite_PRED_FILE):
        print(f"Database already exists: {SQLite_PRED_FILE}")
        response = input("Overwrite? (y/N): ")
        if response.lower() == "y":
            os.remove(SQLite_PRED_FILE)
            create_database(SQLite_PRED_FILE)
        else:
            print("Aborted.")
    else:
        create_database(SQLite_PRED_FILE)