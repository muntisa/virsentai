#!/usr/bin/env python3
"""
Update Webapp with Predictions

Creates TSV files with all predictions and models from the database,
generates JSON summary data for the webapp, and updates the HTML page.

Usage:
    python 04_update_webapp.py

Input:
    - SQLite database: db/virsentai.sqlite3
    - webapp/index.html template

Output:
    - db/all_predictions.tsv: All predictions sorted by score (descending)
    - db/all_models.tsv: All models from database
    - webapp/summary_stats.json: JSON summary for webapp
    - webapp/index.html: Updated HTML page
    - webapp/index_old.html: Backup of previous version

JSON Summary Includes:
    - Total Scanned Viruses: count of all predictions
    - Zoonotic Viruses: count where prediction_score >= threshold
    - Monthly New Cases: dict of month -> count
    - Top Predictions: list of top 10 zoonotic viruses

Blacklist Filter:
    Excludes synthetic/construct/vector-like entries from zoonotic count:
    synthetic, construct, vector, plasmid, clone, pseudovirus, VLP,
    metagenome, satellite, etc.

Notes:
    - No command line arguments required
    - Reads from db/virsentai.sqlite3
    - Logs to logs/04_update_webapp_<timestamp>.log
"""

How to Run:
    python 04_update_webapp.py
    
    Note: No arguments required. Reads from SQLite database and updates webapp.

Requirements:
    - SQLite database at db/virsentai.sqlite3 with predictions
    - webapp/index.html template file
"""
import os
import logging
from datetime import datetime
import pandas as pd
import sqlite3
import json
import re
import shutil

from config import *

# --- Logging setup ---
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(LOG_DIR, exist_ok=True)
log_file_name = os.path.join(LOG_DIR, f"{script_name}_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_name),
        logging.StreamHandler()
    ]
)

def embed_json_into_html(html_template_path, json_data_path, output_html_path):
    """
    Reads a JSON file, embeds its content into a specified HTML file,
    and saves the result to a new HTML file.
    """
    try:
        # Read the JSON data
        with open(json_data_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # Convert to JSON string
        json_string_for_js = json.dumps(json_data, indent=10)

        # Construct the new JavaScript block
        new_summary_stats_js_block = f"""      // Data from summary_stats.json embedded directly
      const summaryStats = {json_string_for_js};"""

        # Read the HTML template
        with open(html_template_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Find and replace the old 'const summaryStats = {...};' block
        pattern = re.compile(
            r'\s*// Data from summary_stats\.json embedded directly.*?const summaryStats\s*=\s*\{.*?};\s*',
            re.DOTALL
        )
        
        match = pattern.search(html_content)
        if match:
            updated_html_content = pattern.sub(new_summary_stats_js_block, html_content, 1)
            print(f"Successfully replaced 'const summaryStats' block in '{html_template_path}'.")
        else:
            print(f"Warning: The 'const summaryStats = {{...}};' block was not found in '{html_template_path}'.")
            updated_html_content = html_content

        # Write the updated HTML content
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(updated_html_content)

        print(f"Successfully created '{output_html_path}' with embedded JSON data.")

    except Exception as e:
        logging.error(f"Error during embedding: {e}")
        raise

def update_webapp():
    """
    Loads data from SQLite database, creates TSV and JSON files, and updates HTML.
    """
    # Load data from SQLite database
    logging.info("Loading data from SQLite database for webapp update...")
    
    try:
        conn = sqlite3.connect(SQLite_PRED_FILE)
        df_db  = pd.read_sql_query("SELECT * FROM predictions", conn)
        df_db2 = pd.read_sql_query("SELECT * FROM models", conn)
        conn.close()
        logging.info(f"Successfully loaded {len(df_db)} records from {SQLite_PRED_FILE}")

        # Create TSV file with all predictions
        logging.info(f"Saving all predictions to TSV file: {ALL_PREDS_FILE}")
        df_db = df_db.drop('created_at', axis=1)
        df_db.sort_values(by='prediction_score', ascending=False, inplace=True)
        df_db.to_csv(ALL_PREDS_FILE, sep='\t', index=False)

        # Create TSV file with all models
        logging.info(f"Saving all models to TSV file: {ALL_MODELS_FILE}")
        df_db2 = df_db2.drop('created_at', axis=1)
        df_db2.to_csv(ALL_MODELS_FILE, sep='\t', index=False)

        # Create JSON Summary from Database Data
        df_db['prediction_date'] = pd.to_datetime(df_db['prediction_date'], format='mixed')

        # Calculate total scanned viruses
        total_scanned_db = len(df_db)

        # Calculate total zoonotic viruses (prediction_score >= 0.8)
        zoonotic_df_db = df_db[df_db['prediction_score'] >= VIRSENTAI_PROB_CUTOFF].copy()

        # Filter out synthetic/construct/vector-like entries
        pattern = r"(?:" + r"|".join([re.escape(t) for t in ZOONOTIC_VIRUS_KEYWORDS]) + r")"
        mask_exclude = zoonotic_df_db['virus_name'].astype(str).str.contains(pattern, case=False, regex=True, na=False)
        zoonotic_df_db = zoonotic_df_db[~mask_exclude].copy()
        
        total_zoonotic_db = len(zoonotic_df_db)

        # Calculate monthly new cases
        zoonotic_df_db['Month'] = zoonotic_df_db['prediction_date'].dt.to_period('M').astype(str)
        monthly_new_cases_db = zoonotic_df_db.groupby('Month').size().to_dict()

        # Top 10 predictions
        top_preds_db = (
            zoonotic_df_db
            .sort_values(by='prediction_score', ascending=False)
            .head(10)
            .loc[:, ['virus_id', 'virus_name', 'virus_host', 'virus_db', 'prediction_score', 'prediction_date']]
            .rename(columns={
                'virus_id': 'Virus ID',
                'virus_name': 'Virus Name',
                'virus_host': 'Virus Host',
                'virus_db': 'Database',
                'prediction_score': 'Probability',
                'prediction_date': 'Scan Date'
            })
        )
        
        top_preds_db['Scan Date'] = top_preds_db['Scan Date'].dt.strftime('%Y-%m-%d')
        top_preds_db = top_preds_db.to_dict(orient='records')
        
        # Create summary dictionary
        summary_db = {
            "Total Scanned Viruses": total_scanned_db,
            "Zoonotic Viruses": total_zoonotic_db,
            "Monthly New Cases": monthly_new_cases_db,
            "Top Predictions": top_preds_db
        }

        # Save to JSON file
        with open(JSON_FILE, 'w') as f:
            json.dump(summary_db, f, indent=4)

        logging.info(f"Summary JSON from database saved to {JSON_FILE}")

        # Embed JSON data into HTML page
        html_template_file = os.path.join(WEBAPP_DIR, "index.html")
        json_input_file = JSON_FILE
        temp_updated_html_file = os.path.join(WEBAPP_DIR, "index_updated.html")
        old_html_backup_file   = os.path.join(WEBAPP_DIR, "index_old.html")

        # Ensure webapp directory exists
        os.makedirs(WEBAPP_DIR, exist_ok=True)

        # Check if JSON file exists
        if not os.path.exists(json_input_file):
            logging.error(f"Error: JSON data file '{json_input_file}' not found.")
            exit(1)

        # Create backup
        if os.path.exists(html_template_file):
            shutil.copyfile(html_template_file, old_html_backup_file)
            logging.info(f"Backed up current '{html_template_file}' to '{old_html_backup_file}'.")

        # Create updated HTML
        embed_json_into_html(html_template_file, json_input_file, temp_updated_html_file)

        # Replace index.html
        shutil.copyfile(temp_updated_html_file, html_template_file)
        logging.info(f"Copied '{temp_updated_html_file}' to '{html_template_file}'.")

        # Remove temporary file
        os.remove(temp_updated_html_file)
        logging.info(f"Removed temporary file '{temp_updated_html_file}'.")

        print("\nScript completed successfully. Your 'index.html' has been updated.")

    except FileNotFoundError:
        logging.error(f"Error: The database file was not found at {SQLite_FILE}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    print("Updating webapp with latest data...")
    update_webapp()
    print("\nWebapp update complete!")