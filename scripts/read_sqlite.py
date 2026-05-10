#!/usr/bin/env python3
"""
SQLite Database Reader

Reads and displays the contents of a SQLite database in a readable format.
Shows the first 5 rows of each table with truncated sequence columns for
better terminal visibility.

Usage:
    python read_sqlite.py <database_file>

Example:
    python read_sqlite.py ds/raw-viruses.sqlite3

Output:
    - Displays all tables with first 5 rows each
    - Shows total row count per table
"""

import sqlite3
import pandas as pd
import argparse
import os
import sys

def read_sqlite(db_file):
    """
    Connects to a SQLite database and prints the head of all tables.
    Truncates long genomic sequences for better visibility in the terminal.
    """
    if not os.path.exists(db_file):
        print(f"Error: File '{db_file}' not found.")
        return

    try:
        # Connect to the database
        conn = sqlite3.connect(db_file)
        
        # Fetch the list of all tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        
        if not tables:
            print(f"No user tables found in '{db_file}'.")
            return

        print("=" * 60)
        print(f"DATABASE: {os.path.abspath(db_file)}")
        print(f"TABLES FOUND: {', '.join(tables)}")
        print("=" * 60 + "\n")

        # Set pandas options for better terminal display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 50)

        for table in tables:
            print(f"--- TABLE: {table} ---")
            
            try:
                # Load the first 5 rows
                df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", conn)
                
                if df.empty:
                    print("[Table is empty]\n")
                    continue
                
                # Special handling for genomic sequences to prevent terminal clutter
                if 'sequence' in df.columns:
                    df['sequence'] = df['sequence'].apply(lambda x: (str(x)[:40] + "...") if len(str(x)) > 40 else x)
                
                # Print the head
                print(df.to_string(index=False))
                
                # Show record count for the table
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"\n[Total rows in table: {count}]\n")
                
            except Exception as table_error:
                print(f"Error reading table '{table}': {table_error}\n")

        print("=" * 60)
        conn.close()
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and print heads of all tables in a SQLite database.")
    parser.add_argument("db_file", help="Path to the SQLite database file.")
    
    # If no arguments are provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    read_sqlite(args.db_file)
